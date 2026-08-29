import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import os
import math
import copy
import features
import cshogi
import struct
import glob
import random
import time
import importlib
import argparse
import sys
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

# ==========================================
# ⚙️ グローバル設定
# ==========================================
DEVICE = torch.device('cuda')

# 🚨 ROCm(gfx120X系)はデフォルトのhipBLAS/rocBLASカーネル選択が最適でない
# ことが確認されている(AMD ROCm issue #5674相当)。TunableOpのオンライン
# チューニングを有効化することで、DirectML比 約1.08倍まで高速化する
# (ベンチマーク済み: 1.7085 sec/step -> 1.1666 sec/step)。
# 初回のみ未知の演算形状に対してチューニングコストが発生するが、結果は
# tunableop_results0.csv に永続化され、次回実行時は即座に再利用される。
torch.cuda.tunable.enable(True)
torch.cuda.tunable.tuning_enable(True)
DATA_DIR = os.environ.get("DAINAGON_DATA_DIR", os.path.join("data", "traindata"))
TEST_DATA_DIR = os.environ.get("DAINAGON_VAL_DIR", os.path.join("data", "train_test"))
MODEL_SAVE_PATH = "model/chunagon_v30.pt" 
GRAPH_SAVE_PATH = "model/learning_curve_v30.png" 

# ハイパーパラメータ
EPOCHS              = 3
INPUT_CHANNELS      = 55
LEARNING_RATE       = 3e-4
WEIGHT_DECAY        = 1e-3
WARMUP_STEPS        = 800  # 🚨 P1-5: 2000 -> 800 (TOTAL_OPT_STEPSの約3.9%相当)
ETA_MIN_RATIO       = 1e-3
# 🚨 P1-5: これまでの全実験が6時間の壁時計学習で約20,583 optimizer.stepに
# 収束している実測値に基づき、スケジューラの総ステップ数を固定値で与える
# (EPOCHS×1エポック分のstep数という従来の計算式には依存させない)
TOTAL_OPT_STEPS     = 20583
TARGET_BATCH_SIZE   = 2048
PHYSICAL_BATCH_SIZE = 256
ACCUM_STEPS         = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE
VALUE_LOSS_SCALE    = 1.5
VALUE_MIX_LAMBDA    = 0.5  # 🚨 P1-2設計書§6: Binned Value / Result Head の混合比。固定値、自己判断で変更しない
SAVE_INTERVAL_SEC   = 1800
BACKUP_SAVE_INTERVAL_SEC = 36000
POSITIONS_PER_EPOCH = 100_000_000  # 1エポックあたりの局面数（1億局面）
SHUFFLE_BUFFER_SIZE = 500_000      # シャッフルバッファ（大きくするとより混ざる。50万件で約19MB）
CHUNK_SIZE          = 4096         # 1回のランダムアクセスで読み込む連続件数（IO速度担保）
MAE_RATE            = 0.20
SCORE_CUTLINE       = 3000  # 🚨 P1-1_bugfix: score_cpはint16(struct '<h')で±32767が上限のため、
                            # 旧値100000では abs(score_cp) > SCORE_CUTLINE が常にFalseとなり
                            # 実質不発だった(実測: 0%フィルタ)。詰み付近の極端スコアを除外する
                            # 意図に沿って有効な値に修正(実測: 17.36%フィルタ)。
                            # 🚨 P1-2: SCORE_CUTLINEのロジックには一切触れない(設計書§7の留保事項)。

# ==========================================
# 🎲 P1-2: Binned Value (HL-Gauss) 設定
# ==========================================
NUM_VALUE_BINS = 128
_VALUE_BIN_EDGES = [i / NUM_VALUE_BINS for i in range(NUM_VALUE_BINS + 1)]
_VALUE_BIN_SIGMA = 0.75 * (1.0 / NUM_VALUE_BINS)  # 設計書§4: ビン幅の0.75倍を初期値とする


def _normal_cdf(x, loc, scale):
    return 0.5 * (1.0 + math.erf((x - loc) / (scale * math.sqrt(2.0))))


def make_hl_gauss_target(target_prob):
    """
    target_prob([0,1]のスカラー勝率ターゲット)を中心としたガウス分布を
    NUM_VALUE_BINS個のビンに割り当て、正規化した確率分布(合計1)を返す。
    (設計書§4 HL-Gauss。データローダー内でCPU側/pure pythonで計算する)
    """
    cdf = [_normal_cdf(edge, target_prob, _VALUE_BIN_SIGMA) for edge in _VALUE_BIN_EDGES]
    probs = [cdf[i + 1] - cdf[i] for i in range(NUM_VALUE_BINS)]
    total = sum(probs)
    probs = [p / total for p in probs]  # 端の裾切れを正規化で補正
    return torch.tensor(probs, dtype=torch.float32)

# ==========================================
# 🔌 ネットワーク動的ローダー
# ==========================================
def get_network(network_path):
    """
    例: "network.network_chunagon_v25.ChunagonNetV3" 
    のように指定された文字列からクラスを動的にインポートしてインスタンス化する。
    """
    try:
        module_name, class_name = network_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        NetworkClass = getattr(module, class_name)
        print(f"[OK] ネットワーク '{class_name}' を '{module_name}' からロードしました。")
        return NetworkClass()
    except Exception as e:
        print(f"[ERROR] ネットワークの読み込みに失敗しました ({network_path}): {e}")
        sys.exit(1)

# ==========================================
# 🎛️ Optimizer param_groups (weight decay 分類)
# ==========================================
def build_param_groups(model, weight_decay):
    """
    🚨 P1-1_bugfix: 旧実装は名前の部分文字列 ('bias','norm','bn') で
    no-decay判定していたが、nn.Sequential内でインデックス番号アクセスに
    なるBatchNorm2d(例: stem.1, value_conv.1, mae_conv.1)は名前に
    'bn'/'norm'が含まれず、weightが誤ってweight decay対象になっていた。
    モジュール型(BatchNorm/LayerNorm/GroupNorm)で判定することで、
    名前の付け方に依存せず正しくno-decay側に分類する。
    """
    decay_params = []
    no_decay_params = []
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)

    for module in model.modules():
        for pname, p in module.named_parameters(recurse=False):
            if isinstance(module, norm_types) or pname == 'bias':
                no_decay_params.append(p)
            else:
                decay_params.append(p)

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

# ==========================================
# 📉 ユーティリティ & スケジューラ
# ==========================================
def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, eta_min_ratio=1e-3):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_decay * (1.0 - eta_min_ratio) + eta_min_ratio
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def estimate_dataset_size(data_dir, record_size=38):
    files = glob.glob(os.path.join(data_dir, "*.hcpe"))
    total_bytes = sum(os.path.getsize(f) for f in files)
    return total_bytes // record_size, len(files)

# ==========================================
# 🎯 指し手変換 (dlshogi 2187次元方式)
# ==========================================
def usi_to_action(usi):
    """USI文字列(例: 7g7f)を、[27方向 × 81マス = 2187次元]のインデックスに変換する"""
    if usi[1] == '*': # 持ち駒の打ち込み
        piece_type = usi[0]
        drop_dict = {'P': 0, 'L': 1, 'N': 2, 'S': 3, 'G': 4, 'B': 5, 'R': 6}
        dir_idx = 20 + drop_dict[piece_type]
        
        to_file = int(usi[2])
        to_rank = ord(usi[3]) - ord('a')
        to_x = 9 - to_file
        to_y = to_rank
    else: # 盤上の移動
        from_file = int(usi[0])
        from_rank = ord(usi[1]) - ord('a')
        to_file = int(usi[2])
        to_rank = ord(usi[3]) - ord('a')
        
        from_x = 9 - from_file
        from_y = from_rank
        to_x = 9 - to_file
        to_y = to_rank
        
        dx = to_x - from_x
        dy = to_y - from_y
        
        # 移動方向の特定 (0~9)
        if dy < 0 and dx == 0: dir_idx = 0
        elif dy < 0 and dx < 0 and dy == dx: dir_idx = 1
        elif dy < 0 and dx > 0 and -dy == dx: dir_idx = 2
        elif dy == 0 and dx < 0: dir_idx = 3
        elif dy == 0 and dx > 0: dir_idx = 4
        elif dy > 0 and dx == 0: dir_idx = 5
        elif dy > 0 and dx < 0 and dy == -dx: dir_idx = 6
        elif dy > 0 and dx > 0 and dy == dx: dir_idx = 7
        elif dy == -2 and dx == -1: dir_idx = 8
        elif dy == -2 and dx == 1: dir_idx = 9
        else: raise ValueError(f"Invalid move direction: {usi}")
            
        # 成る場合は +10
        if len(usi) == 5 and usi[4] == '+':
            dir_idx += 10
            
    to_sq = to_y * 9 + to_x
    return dir_idx * 81 + to_sq

# ==========================================
# 🗂️ データセット (score_cpを出力に追加)
# ==========================================
class MultiFileHCPEDataset(IterableDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.hcpe")))
        if not self.file_list:
            raise ValueError("データが見つかりません。")
        self.RECORD_SIZE = 38
        self.TBL_NUM = str.maketrans("123456789", "987654321")
        self.TBL_ALP = str.maketrans("abcdefghi", "ihgfedcba")
    
    def _process_record(self, record, cb):
        hcp = np.frombuffer(record[:32], dtype=np.uint8).copy()
        score_cp = struct.unpack('<h', record[32:34])[0]

        if abs(score_cp) > SCORE_CUTLINE:
            return None

        move16 = struct.unpack('<H', record[34:36])[0]

        # 🚨 P1-2: record[36] (gameResult) 実機確認結果 (2026-08-17):
        #   gameResult は符号付き1byte (struct '<b') だが、実際に取り得る値は
        #   cshogi公式定数と完全一致する 0/1/2 のみ (負値は観測されない):
        #     gameResult == cshogi.DRAW (0)       : 引き分け
        #     gameResult == cshogi.BLACK_WIN (1)  : 先手勝ち
        #     gameResult == cshogi.WHITE_WIN (2)  : 後手勝ち
        #   検証: 実データ474,430,891局面中の1ファイル(約940万局面)で
        #   値は{0,1,2}の3種のみ。さらに |score_cp|>=25000 の75,804局面で
        #   score_cp>0(手番有利)と「手番側が最終的に勝利」の一致率100%
        #   (不一致0件)を確認済み。score_cpは手番側視点で確定。
        game_result = struct.unpack('<b', record[36:37])[0]

        cb.set_hcp(hcp)
        move_usi = cshogi.move_to_usi(move16)
        is_white = (cb.turn == cshogi.WHITE)

        x = features.make_features_v16_fast(cb, is_white)
        label = self._rotate_move(move_usi) if is_white else move_usi

        try:
            move_id = usi_to_action(label)
        except ValueError:
            return None

        # 勝率 (BCEターゲット用)
        #win_rate = 1.0 / (1.0 + math.exp(-score_cp / 600.0))
        target_logit = score_cp / 600.0
        # Sigmoidを通して 0.0〜1.0 の「確率(勝率)」に変換する
        target_prob = 1.0 / (1.0 + math.exp(max(min(-target_logit, 50.0), -50.0)))

        # 🚨 P1-2: Result Head用ターゲット。手番視点で「この局面の手番側が
        # 最終的に勝ったか」に統一する (設計書§3)
        if game_result == cshogi.DRAW:
            result_target = 0.5
        elif game_result == cshogi.BLACK_WIN:
            result_target = 1.0 if (cb.turn == cshogi.BLACK) else 0.0
        else:  # cshogi.WHITE_WIN
            result_target = 1.0 if (cb.turn == cshogi.WHITE) else 0.0

        soft_value_target = make_hl_gauss_target(target_prob)

        return (torch.from_numpy(x),
                torch.tensor(move_id, dtype=torch.long),
                torch.tensor(target_prob, dtype=torch.float32),
                torch.tensor(score_cp, dtype=torch.float32),
                soft_value_target,
                torch.tensor(result_target, dtype=torch.float32))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # ワーカー1つあたりのノルマを計算（例：1億 ÷ 4ワーカー = 2500万件）
        yield_limit = POSITIONS_PER_EPOCH // num_workers
        yielded = 0

        cb = cshogi.Board()
        buffer = []
        
        # ワーカーごとに異なる乱数シードを持たせて、全員でバラバラの場所を漁る
        seed = (torch.initial_seed() + worker_id) % (2**32)
        rng = random.Random(seed)

        while yielded < yield_limit:
            # 1. 完全ランダムにファイルを選ぶ
            fp = rng.choice(self.file_list)
            fsize = os.path.getsize(fp)
            num_rec = fsize // self.RECORD_SIZE
            if num_rec == 0: continue

            # 2. ファイルのランダムな位置（オフセット）から読み始める
            start_rec = rng.randrange(num_rec)
            
            with open(fp, "rb") as fh:
                fh.seek(start_rec * self.RECORD_SIZE)
                
                # 3. 効率化のため、チャンクサイズ分だけ連続でガサッと読む
                for _ in range(CHUNK_SIZE):
                    raw = fh.read(self.RECORD_SIZE)
                    if not raw or len(raw) < self.RECORD_SIZE:
                        break # EOFに達したら次のファイルへ
                    buffer.append(raw)

                    # 4. シャッフルバッファが満タンになったらランダムに排出
                    if len(buffer) >= SHUFFLE_BUFFER_SIZE:
                        # バッファの中からテキトーに1個選んで末尾とスワップして取り出す（O(1)で高速）
                        idx = rng.randrange(len(buffer))
                        buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                        raw_to_process = buffer.pop()

                        result = self._process_record(raw_to_process, cb)
                        if result is not None:
                            yield result
                            yielded += 1
                            # ノルマ（1エポック分）に達したら即終了
                            if yielded >= yield_limit:
                                return
        
        # もしバッファに端数が残っていたら排出して終了
        rng.shuffle(buffer)
        for raw in buffer:
            if yielded >= yield_limit: return
            result = self._process_record(raw, cb)
            if result is not None:
                yield result
                yielded += 1
    def _rotate_move(self, move_usi):
        if "*" in move_usi:
            return move_usi[0] + "*" + move_usi[2:].translate(self.TBL_NUM).translate(self.TBL_ALP)
        return move_usi[:2].translate(self.TBL_NUM).translate(self.TBL_ALP) + \
               move_usi[2:4].translate(self.TBL_NUM).translate(self.TBL_ALP) + move_usi[4:]

# ==========================================
# 📊 評価用 Dataset & 関数
# ==========================================
class ValidationDataset(IterableDataset):
    def __init__(self, data_dir, num_samples=5000):
        self.data_dir = data_dir
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.hcpe")))
        if not self.file_list:
            raise ValueError(f"❌ テストデータが見つかりません: {data_dir}")
        self.num_samples = num_samples
        self.RECORD_SIZE = 38
        self.TBL_NUM = str.maketrans("123456789", "987654321")
        self.TBL_ALP = str.maketrans("abcdefghi", "ihgfedcba")

    def _rotate_move(self, move_usi):
        if "*" in move_usi:
            return move_usi[0] + "*" + move_usi[2:].translate(self.TBL_NUM).translate(self.TBL_ALP)
        return move_usi[:2].translate(self.TBL_NUM).translate(self.TBL_ALP) + \
               move_usi[2:4].translate(self.TBL_NUM).translate(self.TBL_ALP) + move_usi[4:]

    def __iter__(self):
        target_file = self.file_list[-1]
        file_size = os.path.getsize(target_file)
        num_records = file_size // self.RECORD_SIZE
        start_rec = max(0, num_records - self.num_samples)

        cb = cshogi.Board()
        count = 0

        with open(target_file, "rb") as f:
            f.seek(start_rec * self.RECORD_SIZE)
            while count < self.num_samples:
                record = f.read(self.RECORD_SIZE)
                if not record or len(record) < self.RECORD_SIZE: break

                hcp = np.frombuffer(record[:32], dtype=np.uint8).copy()
                score_cp = struct.unpack('<h', record[32:34])[0]
                move16 = struct.unpack('<H', record[34:36])[0]
                game_result = struct.unpack('<b', record[36:37])[0]  # 🚨 P1-2: §3参照

                cb.set_hcp(hcp)
                move_usi = cshogi.move_to_usi(move16)
                is_white = (cb.turn == cshogi.WHITE)

                x = features.make_features_v16_fast(cb, is_white)

                if is_white: move_usi = self._rotate_move(move_usi)

                try: move_id = usi_to_action(move_usi)
                except ValueError: continue

                #win_rate = 1.0 / (1.0 + math.exp(-score_cp / 600.0))
                target_logit = score_cp / 600.0

                # 🚨 P1-2: Result Head用ターゲット(手番視点、_process_recordと同一ロジック)
                if game_result == cshogi.DRAW:
                    result_target = 0.5
                elif game_result == cshogi.BLACK_WIN:
                    result_target = 1.0 if (cb.turn == cshogi.BLACK) else 0.0
                else:  # cshogi.WHITE_WIN
                    result_target = 1.0 if (cb.turn == cshogi.WHITE) else 0.0

                yield (torch.from_numpy(x),
                       torch.tensor(move_id, dtype=torch.long),
                       torch.tensor(score_cp, dtype=torch.float32),
                       torch.tensor(result_target, dtype=torch.float32))
                count += 1

def evaluate(model, val_loader):
    """
    🚨 P1-2: Binned Value + Result Head を持つネットワーク専用。
    eval modeのforward()は (p, v_scalar, result_prob) の3値を返す前提。
    value_loss_prob(勝率空間のMSE)は P1-0/P1-1と完全に同一定義のまま
    維持している(設計書§6、比較可能性の生命線のため変更禁止)。
    ロジット空間MSE(value_loss_v_logit)はBinned Valueでは定義できないため
    廃止し、代わりにResult Headの正解率(引き分け局面は正誤判定から除外)を返す。
    """
    model.eval() # 🚨 推論モード（Valueには自動的にSigmoidがかかる）
    correct = 0
    total = 0
    val_loss_p = 0.0
    val_loss_v_prob = 0.0
    result_correct = 0
    result_total = 0
    batch_count = 0

    criterion_p = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()

    with torch.no_grad():
        for x, t_m, t_score, result_target in val_loader:
            x = x.to(DEVICE)
            t_m = t_m.to(DEVICE)
            t_score = t_score.to(DEVICE)
            result_target = result_target.to(DEVICE)

            # p: ロジット(2187), v_scalar: 勝率(0.0~1.0), result_prob: 勝率(Result Head)
            p, v_scalar, result_prob = model(x)

            # 1. Policy 精度と Loss (CrossEntropy)
            pred_move = p.argmax(1)
            correct += (pred_move == t_m).sum().item()
            total += t_m.size(0)
            val_loss_p += criterion_p(p, t_m).item()

            # 2. 正解データの作成 (水匠定数600)
            t_v_logit = t_score / 600.0
            t_v_prob = 1.0 / (1.0 + torch.exp(-t_v_logit))

            # 3. Value Loss計算: 勝率空間のMSE (🚨 P1-0/P1-1と同一定義、変更禁止)
            val_loss_v_prob += criterion_v(v_scalar.squeeze(), t_v_prob).item()

            # 4. Result Headの正解率 (引き分け(0.5)は正誤判定の対象外)
            decisive_mask = (result_target != 0.5)
            if decisive_mask.any():
                pred_result = (result_prob.squeeze() > 0.5).float()
                result_correct += (pred_result[decisive_mask] == result_target[decisive_mask]).sum().item()
                result_total += decisive_mask.sum().item()

            batch_count += 1

    acc = correct / total if total > 0 else 0
    avg_loss_p = val_loss_p / batch_count if batch_count > 0 else 0
    avg_loss_v_prob = val_loss_v_prob / batch_count if batch_count > 0 else 0
    # 🚨 P1-2: ロジット空間MSEはBinned Valueでは定義不可のため廃止(常に0.0のプレースホルダ)
    avg_loss_v_logit_deprecated = 0.0
    result_acc = result_correct / result_total if result_total > 0 else 0

    return acc, avg_loss_p, avg_loss_v_logit_deprecated, avg_loss_v_prob, result_acc

def plot_learning_curve(loss_history, save_path):
    if not loss_history: return
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Chunagon v3 (Y-shape) Learning Curve')
    plt.xlabel('Optimizer Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# ==========================================
# 🔄 メイン訓練ループ
# ==========================================
def train(args):
    os.makedirs("model", exist_ok=True)
    
    data_dir = getattr(args, "data_dir", None) or DATA_DIR
    test_data_dir = getattr(args, "val_dir", None) or TEST_DATA_DIR

    # 全体容量の推定は情報として残す
    total_records, num_files = estimate_dataset_size(data_dir)
    
    # 🚨 学習の基準を POSITIONS_PER_EPOCH (1億) に固定する
    actual_positions_per_epoch = min(total_records, POSITIONS_PER_EPOCH)
    total_batches_per_epoch = actual_positions_per_epoch // PHYSICAL_BATCH_SIZE
    # 🚨 P1-5: total_opt_stepsはEPOCHSベースの計算式ではなく、実測に基づく
    # 固定値TOTAL_OPT_STEPSをスケジューラの前提として用いる
    total_opt_steps = TOTAL_OPT_STEPS

    print(f"[INFO] 全データ: {num_files} ファイル / 推定 {total_records:,} 局面")
    print(f"[INFO] 1エポックの学習サイズ: {actual_positions_per_epoch:,} 局面")
    print(f"[INFO] 総最適化ステップ数(スケジューラ用固定値): {total_opt_steps:,} ステップ")

    # データローダー
    NUM_WORKERS = 4
    dataset = MultiFileHCPEDataset(data_dir)
    loader = DataLoader(dataset, batch_size=PHYSICAL_BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS>0))
    val_dataset = ValidationDataset(test_data_dir)
    val_loader = DataLoader(val_dataset, batch_size=PHYSICAL_BATCH_SIZE, num_workers=0)

    # 🚀 ネットワークの動的ロード
    model = get_network(args.network).to(DEVICE)

    # Optimizer & Scheduler
    param_groups = build_param_groups(model, WEIGHT_DECAY)
    optimizer = optim.AdamW(param_groups, lr=LEARNING_RATE)
    scheduler = get_warmup_cosine_scheduler(optimizer, WARMUP_STEPS, total_opt_steps, ETA_MIN_RATIO)

    # 💾 レジューム用変数
    start_epoch = 0
    start_global_step = 0
    loss_history = []

    if args.resume and os.path.exists(MODEL_SAVE_PATH):
        print(f"[INFO] チェックポイントを復元します: {MODEL_SAVE_PATH}")
        ckpt = torch.load(MODEL_SAVE_PATH, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        model.to(DEVICE)
        
        if 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
        if 'epoch' in ckpt: start_epoch = ckpt['epoch']
        if 'global_step' in ckpt: start_global_step = ckpt['global_step']
        print(f"[OK] Epoch {start_epoch+1}, 累積 {start_global_step} ステップから再開します。")

    last_save_time = time.time()
    last_backup_time = time.time()
    gc_counter = 0

    print(f"=== Dainagon Training Started ===")

    for epoch in range(start_epoch, EPOCHS):
        model.train() # 🚨 訓練モード (Valueは生ロジットを出力する)
        pbar = tqdm(loader, total=total_batches_per_epoch, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)
        
        batch_count = 0
        step_count = 0
        running_loss = 0.0

        for i, (x, t_m, t_v, t_score, soft_value_target, result_target) in enumerate(pbar):
            batch_count += 1
            """
            # 🚨 完全レジューム処理 (過去のステップまで空読みスキップ)
            current_total_steps = (batch_count // ACCUM_STEPS)
            if current_total_steps <= start_global_step and epoch == start_epoch:
                if batch_count % ACCUM_STEPS == 0:
                    pbar.set_postfix_str(f"Skipping to resume point... ({current_total_steps}/{start_global_step})")
                continue
            """
            x = x.to(DEVICE)
            t_m = t_m.to(DEVICE)
            t_score = t_score.to(DEVICE)
            soft_value_target = soft_value_target.to(DEVICE)
            result_target = result_target.to(DEVICE)

            # Forward (🚨 P1-2: Binned Value + Result Head。訓練時は
            # (p, value_logits[B,NUM_VALUE_BINS], result_logit[B,1]) を返す)
            p, value_logits, result_logit = model(x)
            # 🚨 P1-1_bugfix: policyロジットのclamp(-10,10)を撤廃
            # (勾配を不必要に歪める可能性があるため)

            # --- ⚔️ Loss 計算 ---
            # 1. Policy Loss (🚨 P1-1_bugfix: label_smoothing 0.1 -> 0.03)
            loss_p = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.03)(p, t_m)

            # 2. Value Loss (🚨 P1-2: Binned Value(HL-Gauss, CE) + Result Head(BCE) を混合)
            loss_v_binned = -(soft_value_target * torch.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()
            loss_v_result = nn.BCEWithLogitsLoss()(result_logit.squeeze(), result_target)
            loss_v = (VALUE_MIX_LAMBDA * loss_v_binned + (1 - VALUE_MIX_LAMBDA) * loss_v_result) * VALUE_LOSS_SCALE

            loss = (loss_p + loss_v) / ACCUM_STEPS
            loss.backward()

            if batch_count % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                actual_global_step = (start_global_step if epoch == start_epoch else 0) + step_count
                step_count += 1
                
                # 🚨 追加: loss_historyに記録する
                current_loss = loss.item() * ACCUM_STEPS
                running_loss += current_loss
                loss_history.append(current_loss)
                
                pbar.set_postfix({
                    "Loss": f"{current_loss:.4f}",
                    "L_p": f"{loss_p.item():.4f}",
                    "L_v": f"{loss_v.item():.4f}",
                    "L_vbin": f"{loss_v_binned.item():.4f}",
                    "L_vres": f"{loss_v_result.item():.4f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
                })

                # ⏱️ 30分ごとの保存と評価
                current_time = time.time()
                if current_time - last_save_time > SAVE_INTERVAL_SEC:
                    
                    # 🚨 追加: テスト評価の実行
                    acc, val_loss_p, _val_loss_v_logit_deprecated, val_loss_v_prob, result_acc = evaluate(model, val_loader)
                    print(f"\n[EVAL] Policy精度: {acc:.2%}, Policy Loss: {val_loss_p:.4f}, Value MSE(勝率): {val_loss_v_prob:.5f}, Result Head精度: {result_acc:.2%}")
                    model.train() # 評価が終わったら必ず訓練モード(生ロジット出力)に戻す
                    
                    # 🚨 追加: グラフの保存
                    plot_learning_curve(loss_history, GRAPH_SAVE_PATH)

                    ckpt = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_step': actual_global_step,
                    }
                    torch.save(ckpt, MODEL_SAVE_PATH)
                    print(f"[SAVE] Checkpoint updated. (Step: {actual_global_step})")
                    last_save_time = current_time

                    if current_time - last_backup_time > BACKUP_SAVE_INTERVAL_SEC:
                        time_str = time.strftime('%Y%m%d_%H%M%S')
                        base_name, ext = os.path.splitext(MODEL_SAVE_PATH)
                        backup_path = f"{base_name}_backup_{time_str}{ext}"
                        
                        torch.save(ckpt, backup_path) # 直前で作った ckpt をそのまま別名で保存
                        print(f"🛡️ [BACKUP] 10時間経過: バックアップを保存しました -> {backup_path}")
                        last_backup_time = current_time

            # 🛠️ 対Windows メモリリーク防御
            del p, value_logits, result_logit, loss_p, loss_v, loss_v_binned, loss_v_result, loss
            gc_counter += 1
            if gc_counter % 100 == 0:
                import gc
                gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dainagon Training Script (ROCm)")
    parser.add_argument(
        "--network", 
        type=str, 
        default="network.network_chunagon_v28.ChunagonNetV3",
        help="ロードするネットワークのパス (例: network.network_chunagon_v28.ChunagonNetV3)"
    )
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="訓練データのディレクトリパス (デフォルト: ./data/traindata)")
    parser.add_argument("--val-dir", type=str, default=TEST_DATA_DIR, help="検証データのディレクトリパス (デフォルト: ./data/train_test)")
    parser.add_argument("--resume", action="store_true", help="チェックポイントから再開する")
    args = parser.parse_args()

    train(args)