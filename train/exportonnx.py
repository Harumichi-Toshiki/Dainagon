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
    # 🚨 引数からインプットとアウトプットのパスを自動生成
    checkpoint_path = os.path.join("model", f"{args.name}.pt")
    
    if args.output:
        save_path = args.output
    else:
        suffix = "_fp16" if args.fp16 else "_fp32"
        save_path = os.path.join("model", f"{args.name}{suffix}.onnx")

    print(f"[INFO] ネットワーク定義をロード中: {args.network}")
    model = get_network(args.network)

    # 💾 重みの読み込み
    if os.path.exists(checkpoint_path):
        print(f"[INFO] チェックポイントをロード中: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        print("[OK] 重みのインポートに成功しました。")
    else:
        print(f"[WARN] チェックポイントが見つかりません: {checkpoint_path}")
        print("[WARN] 初期化状態の重みのままエクスポートを続行します。")

    # 🚨 評価モード固定 (ValueのSigmoidを強制ON)
    model.eval()

    # 将棋盤入力 [B, 55, 9, 9]
    dummy_input = torch.randn(1, 55, 9, 9, dtype=torch.float32)

    # FP16 変換
    if args.fp16:
        print("[INFO] モデルおよびダミー入力をFP16(半精度)に変換中...")
        model = model.half()
        dummy_input = dummy_input.half()

    # 可変バッチサイズ設定
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'policy': {0: 'batch_size'},
        'value': {0: 'batch_size'}
    }

    print(f"[INFO] ONNXエクスポートを開始します... 保存先: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['policy', 'value'],
            dynamic_axes=dynamic_axes
        )
        print(f"[SUCCESS] ONNXモデルのエクスポートが完了しました！ -> {save_path}")
    except Exception as e:
        print(f"[ERROR] ONNXエクスポート中にエラーが発生しました: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dainagon ONNX Export Script (Smart Path Mode)")
    parser.add_argument(
        "--network", 
        type=str, 
        default="network.network_chunagon_v28.ChunagonNetV3",
        help="対象ネットワーククラスパス"
    )
    # 🚨 パス指定から「名前指定」に変更
    parser.add_argument(
        "--name", 
        type=str, 
        default="chunagon_v28",
        help="モデルファイル名（拡張子 .pt なし、model/ フォルダ内から自動検索）"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="個別にフルパス指定したい場合のみ使用（通常は省略してください）"
    )
    parser.add_argument(
        "--fp16", 
        action="store_true", 
        default=True,
        help="FP16（半精度）形式で出力する"
    )
    args = parser.parse_args()

    export(args)