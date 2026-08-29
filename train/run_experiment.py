"""
run_experiment.py - EXPERIMENTS.md の命名規則・保存規約に沿った実験ランナー。

train_rocm.py (ROCm + TunableOp常時有効) をベースに、実データで壁時計時間
ベースの学習を行い、SAVE_INTERVAL_SEC(デフォルト30分)ごとにevaluate()を
実行してJSON Linesでログを残す。NaN/Infを検知した場合は完走を待たず
即座に停止し、その時点のチェックポイントを保存して終了する。

🚨 P1-2以降: Binned Value (HL-Gauss) + Result Head 構成が標準。
label_smoothing=0.03・policyロジットclamp撤廃はP1-1_bugfixで確定済みの
ため固定値とし、CLIオプションからは削除した。

保存規約 (EXPERIMENTS.md準拠):
  model/checkpoints/{test-id}.pt
  model/logs/{test-id}.jsonl
"""
import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import train_rocm as train
from train_rocm import (
    DEVICE,
    DATA_DIR,
    TEST_DATA_DIR,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    ETA_MIN_RATIO,
    TOTAL_OPT_STEPS,
    PHYSICAL_BATCH_SIZE,
    ACCUM_STEPS,
    VALUE_LOSS_SCALE,
    VALUE_MIX_LAMBDA,
    SAVE_INTERVAL_SEC,
    POSITIONS_PER_EPOCH,
    MultiFileHCPEDataset,
    ValidationDataset,
    get_network,
    get_warmup_cosine_scheduler,
    build_param_groups,
    evaluate,
    estimate_dataset_size,
)

NUM_WORKERS = 4  # train.py の train() 関数内のローカル定数と同じ値
LABEL_SMOOTHING = 0.03  # P1-1_bugfixで確定済み


def collect_relpos_bias_stats(model):
    """
    🚨 P1-3以降: モデルに relpos_bias パラメータ(2D相対位置バイアス)が
    存在する場合、ブロックごとのstd/平均絶対偏差をJSONLに記録するための
    統計を集める。存在しないネットワーク(P1-2以前)では空dictを返す
    (キー自体がJSONLに出ないので後方互換)。
    再学習時に「序盤時点でどれだけ学習が進んでいるか」を追跡できるようにする。
    """
    stats = {}
    for name, p in model.named_parameters():
        if name.endswith("relpos_bias"):
            block_name = name.rsplit(".relpos_bias", 1)[0]
            stats[f"relpos_bias_std__{block_name}"] = p.detach().std().item()
            stats[f"relpos_bias_meanabs__{block_name}"] = p.detach().abs().mean().item()
    return stats


def main():
    parser = argparse.ArgumentParser(description="EXPERIMENTS.md準拠の壁時計時間ベース実験ランナー")
    parser.add_argument("--test-id", type=str, required=True, help="例: P1-2_binnedvalue")
    parser.add_argument("--max-hours", type=float, default=None, help="壁時計時間での学習上限(時間)")
    parser.add_argument("--max-steps", type=int, default=None,
                         help="optimizer.step数での学習上限。EXPERIMENTS.mdの例外規定"
                              "(速度コスト増加5%未満の場合)でstep数を揃えて比較する際に使用。"
                              "--max-hoursと併用可、どちらか早く到達した方で停止する")
    parser.add_argument("--network", type=str, default="network.network_chunagon_v28.ChunagonNetV3")
    parser.add_argument("--eval-interval-sec", type=float, default=SAVE_INTERVAL_SEC)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="訓練データのディレクトリパス (デフォルト: ./data/traindata)")
    parser.add_argument("--val-dir", type=str, default=TEST_DATA_DIR, help="検証データのディレクトリパス (デフォルト: ./data/train_test)")
    args = parser.parse_args()

    if args.max_hours is None and args.max_steps is None:
        parser.error("--max-hours か --max-steps のどちらかを指定してください")

    os.makedirs("model/checkpoints", exist_ok=True)
    os.makedirs("model/logs", exist_ok=True)

    data_dir = args.data_dir
    val_dir = args.val_dir

    ckpt_path = f"model/checkpoints/{args.test_id}.pt"
    jsonl_path = f"model/logs/{args.test_id}.jsonl"

    print(f"[INFO] test_id = {args.test_id}")
    print(f"[INFO] DEVICE = {DEVICE}  TunableOp enabled = {torch.cuda.tunable.is_enabled()}")
    print(f"[INFO] max_hours = {args.max_hours}  max_steps = {args.max_steps}  eval_interval_sec = {args.eval_interval_sec}")
    print(f"[INFO] data_dir = {data_dir}  val_dir = {val_dir}")
    print(f"[INFO] network = {args.network}")
    print(f"[INFO] label_smoothing = {LABEL_SMOOTHING} (固定, P1-1由来)  VALUE_MIX_LAMBDA = {VALUE_MIX_LAMBDA}")
    print(f"[INFO] checkpoint -> {ckpt_path}")
    print(f"[INFO] jsonl log -> {jsonl_path}")

    total_records, num_files = estimate_dataset_size(data_dir)
    actual_positions_per_epoch = min(total_records, POSITIONS_PER_EPOCH)
    total_batches_per_epoch = actual_positions_per_epoch // PHYSICAL_BATCH_SIZE
    # 🚨 P1-5: total_opt_stepsはEPOCHSベースの計算式ではなく、実測(6時間で
    # 約20,583 optimizer.stepに収束)に基づく固定値TOTAL_OPT_STEPSを
    # スケジューラの前提として用いる
    total_opt_steps = TOTAL_OPT_STEPS
    print(f"[INFO] 全データ: {num_files}ファイル / 推定{total_records:,}局面")
    print(f"[INFO] 総最適化ステップ数(スケジューラ用固定値): {total_opt_steps:,}")

    dataset = MultiFileHCPEDataset(data_dir)
    loader = DataLoader(dataset, batch_size=PHYSICAL_BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True)
    val_dataset = ValidationDataset(val_dir)
    val_loader = DataLoader(val_dataset, batch_size=PHYSICAL_BATCH_SIZE, num_workers=0)

    model = get_network(args.network).to(DEVICE)

    param_groups = build_param_groups(model, WEIGHT_DECAY)
    optimizer = optim.AdamW(param_groups, lr=LEARNING_RATE)
    scheduler = get_warmup_cosine_scheduler(optimizer, WARMUP_STEPS, total_opt_steps, ETA_MIN_RATIO)

    model.train()

    criterion_p_fn = lambda p, t: nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)(p, t)
    criterion_result_fn = nn.BCEWithLogitsLoss()

    t_start = time.perf_counter()
    max_seconds = args.max_hours * 3600.0 if args.max_hours is not None else None
    last_eval_wall = t_start

    batch_count = 0
    step_count = 0
    gc_counter = 0
    stop_reason = None
    last_loss_p = last_loss_v = last_loss_v_binned = last_loss_v_result = float('nan')

    jsonl_f = open(jsonl_path, "a", encoding="utf-8")

    try:
        for x, t_m, t_v, t_score, soft_value_target, result_target in loader:
            batch_count += 1
            x = x.to(DEVICE)
            t_m = t_m.to(DEVICE)
            soft_value_target = soft_value_target.to(DEVICE)
            result_target = result_target.to(DEVICE)

            # 🚨 P1-2: Binned Value + Result Head。訓練時は
            # (p, value_logits[B,NUM_VALUE_BINS], result_logit[B,1]) を返す
            p, value_logits, result_logit = model(x)

            loss_p = criterion_p_fn(p, t_m)

            loss_v_binned = -(soft_value_target * torch.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()
            loss_v_result = criterion_result_fn(result_logit.squeeze(), result_target)
            loss_v = (VALUE_MIX_LAMBDA * loss_v_binned + (1 - VALUE_MIX_LAMBDA) * loss_v_result) * VALUE_LOSS_SCALE

            loss = (loss_p + loss_v) / ACCUM_STEPS
            loss.backward()

            if batch_count % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                current_loss = loss.item() * ACCUM_STEPS
                last_loss_p = loss_p.item()
                last_loss_v = loss_v.item()
                last_loss_v_binned = loss_v_binned.item()
                last_loss_v_result = loss_v_result.item()

                if not math.isfinite(current_loss):
                    stop_reason = f"non_finite_loss (loss={current_loss}, step={step_count})"
                    print(f"[ERROR] {stop_reason}")
                    break

                if step_count % 50 == 0:
                    elapsed = time.perf_counter() - t_start
                    print(f"[TRAIN] step={step_count}  elapsed={elapsed / 60:.1f}min  "
                          f"loss={current_loss:.4f}  loss_p={last_loss_p:.4f}  loss_v={last_loss_v:.4f}  "
                          f"loss_v_binned={last_loss_v_binned:.4f}  loss_v_result={last_loss_v_result:.4f}")

                now_wall = time.perf_counter()
                if now_wall - last_eval_wall >= args.eval_interval_sec:
                    print(f"\n[EVAL] step={step_count} 時点でevaluate()を実行...")
                    acc, val_loss_p, _val_loss_v_logit_deprecated, val_loss_v_prob, result_acc = evaluate(model, val_loader)
                    model.train()

                    elapsed = time.perf_counter() - t_start
                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "elapsed_sec": round(elapsed, 1),
                        "optimizer_step": step_count,
                        "policy_accuracy": acc,
                        "value_loss_prob": val_loss_v_prob,
                        "result_head_accuracy": result_acc,
                        "loss_p": last_loss_p,
                        "loss_v": last_loss_v,
                        "loss_v_binned": last_loss_v_binned,
                        "loss_v_result": last_loss_v_result,
                    }
                    record.update(collect_relpos_bias_stats(model))
                    jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_f.flush()
                    print(f"[EVAL] acc={acc:.4%}  value_loss_prob={val_loss_v_prob:.5f}  "
                          f"result_head_accuracy={result_acc:.4%}  elapsed={elapsed / 3600:.2f}h")

                    if not all(math.isfinite(v) for v in (acc, val_loss_p, val_loss_v_prob, result_acc)):
                        stop_reason = f"non_finite_eval_metric (step={step_count})"
                        print(f"[ERROR] {stop_reason}")
                        break

                    last_eval_wall = now_wall

                if max_seconds is not None and time.perf_counter() - t_start >= max_seconds:
                    stop_reason = "max_hours_reached"
                    break

                if args.max_steps is not None and step_count >= args.max_steps:
                    stop_reason = "max_steps_reached"
                    break

            del p, value_logits, result_logit, loss_p, loss_v, loss_v_binned, loss_v_result, loss
            gc_counter += 1
            if gc_counter % 100 == 0:
                import gc
                gc.collect()

    except Exception as e:
        stop_reason = f"exception: {type(e).__name__}: {e}"
        print(f"[ERROR] {stop_reason}")

    finally:
        jsonl_f.close()

    total_elapsed = time.perf_counter() - t_start
    print(f"\n[INFO] 学習ループ終了。理由: {stop_reason}")
    print(f"[INFO] 総経過時間: {total_elapsed / 3600:.2f}h  optimizer.step: {step_count}")

    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': 0,
        'global_step': step_count,
        'stop_reason': stop_reason,
        'elapsed_sec': total_elapsed,
        'test_id': args.test_id,
        'network': args.network,
        'label_smoothing': LABEL_SMOOTHING,
        'value_mix_lambda': VALUE_MIX_LAMBDA,
        'score_cutline': train.SCORE_CUTLINE,
    }
    torch.save(ckpt, ckpt_path)
    print(f"[OK] チェックポイント保存: {ckpt_path} ({os.path.getsize(ckpt_path) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
