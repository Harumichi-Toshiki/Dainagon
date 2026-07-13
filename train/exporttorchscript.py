import torch
import numpy as np
import os
import sys
import importlib
import argparse

# ==========================================
# 🔌 ネットワーク動的ローダー
# ==========================================
def get_network(network_path):
    try:
        module_name, class_name = network_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        NetworkClass = getattr(module, class_name)
        return NetworkClass()
    except Exception as e:
        print(f"[ERROR] ネットワークの読み込みに失敗しました: {e}")
        sys.exit(1)

# ==========================================
# 📦 エクスポートメイン処理
# ==========================================
def export(args):
    # 引数からチェックポイントパスを設定
    checkpoint_path = os.path.join("model", f"{args.name}.pt")
    
    if args.output:
        save_path = args.output
    else:
        suffix = "_fp16" if args.fp16 else "_fp32"
        save_path = os.path.join("model", f"{args.name}{suffix}_traced.pt")

    print(f"[INFO] ネットワーク定義をロード中: {args.network}")
    model = get_network(args.network)

    # 重みの読み込み
    if os.path.exists(checkpoint_path):
        print(f"[INFO] チェックポイントをロード中: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        print("[OK] 重みのインポートに成功しました。")
    else:
        print(f"[WARN] チェックポイントが見つかりません: {checkpoint_path}")
        print("[WARN] 初期化状態の重みのままエクスポートを続行します。")

    # 評価モード固定
    model.eval()

    # 将棋盤入力 [B, 55, 9, 9]
    dummy_input = torch.randn(1, 55, 9, 9, dtype=torch.float32)

    # FP16 変換
    if args.fp16:
        print("[INFO] モデルおよびダミー入力をFP16(半精度)に変換中...")
        model = model.half()
        dummy_input = dummy_input.half()

    print(f"[INFO] TorchScriptエクスポートを開始します... 保存先: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        # torch.jit.traceを使用して静的グラフモデルを作成
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(save_path)
        print(f"[SUCCESS] TorchScriptモデルのエクスポートが完了しました！ -> {save_path}")
    except Exception as e:
        print(f"[ERROR] TorchScriptエクスポート中にエラーが発生しました: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dainagon TorchScript Export Script")
    parser.add_argument(
        "--network", 
        type=str, 
        default="network.network_chunagon_v25_2.ChunagonNetV3",
        help="対象ネットワーククラスパス"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="chunagon_v25_2",
        help="モデルファイル名（拡張子 .pt なし、model/ フォルダ内から自動検索）"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="個別にフルパス指定したい場合のみ使用"
    )
    parser.add_argument(
        "--fp16", 
        action="store_true", 
        default=False,
        help="FP16（半精度）形式で出力する"
    )
    args = parser.parse_args()

    export(args)
