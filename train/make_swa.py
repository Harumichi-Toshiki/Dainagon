import torch

# ==========================================
# ⚙️ SWA錬成設定
# ==========================================
# 平均化したい優秀なチェックポイント（学習終盤のもの）をリストアップ
# ※必ず同じ構造（v21.1）のモデルを指定してください
MODELS_TO_BLEND = [
    "model/chunagon_v27.pt",
    "model/chunagon_v27_2.pt",
    "model/chunagon_v27_3.pt",
    "model/chunagon_v27_b8.pt",
    "model/chunagon_v27_b9.pt",
]

# 保存する新しい最強モデルの名前
OUTPUT_SWA_MODEL = "model/chunagon_v27_swa.pt"

def create_swa_model():
    print(f"🔄 {len(MODELS_TO_BLEND)}つのモデルをブレンド（SWA）します...")
    
    # 1. ベースとなる1つ目のモデルを読み込む
    base_checkpoint = torch.load(MODELS_TO_BLEND[0], map_location='cpu')
    base_state_dict = base_checkpoint['model'] if 'model' in base_checkpoint else base_checkpoint
    
    # 2. 2つ目以降のモデルの重みをひたすら足し算していく
    for path in MODELS_TO_BLEND[1:]:
        print(f"   ➕ 足し合わせ中: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        for key in base_state_dict.keys():
            # テンソルの型が整数型（バッチノルムのstep等）の場合は平均化しない
            if base_state_dict[key].is_floating_point():
                base_state_dict[key] += state_dict[key]
                
    # 3. 足した数（モデルの数）で割って平均を出す
    num_models = len(MODELS_TO_BLEND)
    for key in base_state_dict.keys():
        if base_state_dict[key].is_floating_point():
            base_state_dict[key] = base_state_dict[key] / num_models
            
    # 4. SWAモデルとして保存
    if 'model' in base_checkpoint:
        base_checkpoint['model'] = base_state_dict
        torch.save(base_checkpoint, OUTPUT_SWA_MODEL)
    else:
        torch.save(base_state_dict, OUTPUT_SWA_MODEL)
        
    print(f"\n✨ SWA錬成完了！最強モデルを保存しました: {OUTPUT_SWA_MODEL}")

if __name__ == "__main__":
    create_swa_model()