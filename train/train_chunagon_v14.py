import torch
import torch.nn as nn
import torch.nn.functional as F  
import torch.optim as optim
import torch_directml
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import math
import cshogi # ★爆速ライブラリ
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ 設定 (Dainagon v14: DirectML Speed King)
# ==========================================
DEVICE = torch_directml.device() # DirectML (Radeon)

DATA_FILE = "data/train_data_suisho.txt"
VOCAB_FILE = "data/vocab_v2.json"
MODEL_SAVE_PATH = "model/chunagon_v14_dml.pt" 
GRAPH_SAVE_PATH = "model/learning_curve_v14.png"

# ★モデル構成
BLOCKS = 10
CHANNELS = 128
INPUT_CHANNELS = 46

# ★学習設定
EPOCHS = 5
LEARNING_RATE = 0.001
POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 1.0

# ★メモリ対策 (RX 9070 XTなら攻める)
TARGET_BATCH_SIZE = 2048 
PHYSICAL_BATCH_SIZE = 2048 # VRAM 16GBなら余裕。足りなければ512へ
ACCUM_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE

print(f"⚡ Device: {DEVICE} (DirectML)")
print(f"⚡ Settings: {BLOCKS} Blocks, {CHANNELS} Channels, cshogi=ON")

# ==========================================
# 🚅 高速特徴量生成 (cshogi版)
# ==========================================
# cshogiの定数定義
# HU=1, KY=2, KE=3, GI=4, KI=5, KA=6, HI=7, OU=8
# 成り: +8 (TO=9 ... RY=15)
# 後手: +16 (W_HU=17 ...)
# 手持ち: HU=0 ... HI=6

def make_features_cshogi(board):
    # cshogiはnumpy連携が速い
    features = np.zeros((INPUT_CHANNELS, 9, 9), dtype=np.float32)
    
    # 0. 手番管理 (cshogi: 0=Black, 1=White)
    is_white = (board.turn == cshogi.WHITE)
    
    # 1. 盤上の駒
    # piecesは長さ81のint配列 (0..80)
    # cshogiの座標は 0=9一, 1=8一 ... 8=1一 ... 80=1九
    # 入力特徴量は (row, col) = (筋, 段) ではなく (段, 筋) っぽいので注意
    # train_chunagon_v12では: sq // 9, sq % 9 -> (y, x)
    # target_sq = (80 - sq) if is_white else sq
    
    # 高速イテレーション
    pieces = board.pieces
    for sq, p in enumerate(pieces):
        if p == 0: continue
        
        # 先手/後手判定 & 駒種別(0~13)
        if p <= 15: # 先手 (1..15)
            idx = (p - 1) # 0..14 (FU..RY) -> そのまま使う
            # ただしモデルは 0..13 (14種) を想定しているはず
            # cshogi: 1..8(生), 9..15(成)
            # v12: piece_type-1 (1..14 -> 0..13)
            # マッピング:
            # 1(Fu)..8(Ou) -> 0..7
            # 9(To)..15(Ry) -> 8..14 ? 
            # 前回のv12ロジック: p.piece_type - 1. 
            # python-shogi: PAWN=1..KING=8, PROM_PAWN=9..PROM_ROOK=14
            # ほぼ同じ。
            feat_idx = p - 1
            if feat_idx >= 14: feat_idx = 13 # 安全策(玉は成らないが)
            
            my_piece = True
        else: # 後手 (17..31)
            p_clean = p - 16
            feat_idx = p_clean - 1
            if feat_idx >= 14: feat_idx = 13
            my_piece = False
            
        # 特徴量のチャンネル決定
        # 先手番視点: 味方=0~13, 敵=14~27
        # 後手番視点: 味方=0~13(元の後手駒), 敵=14~27(元の先手駒)
        
        if is_white: # 今は後手番
            if not my_piece: # 先手の駒(敵)
                 channel = feat_idx + 14
            else: # 後手の駒(味方)
                 channel = feat_idx
            # 盤面回転
            target_sq = 80 - sq
        else: # 今は先手番
            if my_piece: # 先手の駒(味方)
                channel = feat_idx
            else: # 後手の駒(敵)
                channel = feat_idx + 14
            target_sq = sq
            
        features[channel, target_sq // 9, target_sq % 9] = 1.0

    # 2. 持ち駒
    # pieces_in_hand: [color][kind] (kind: 0=FU .. 6=HI)
    hand = board.pieces_in_hand
    
    # 味方
    my_hand = hand[1] if is_white else hand[0]
    for k in range(7):
        if my_hand[k] > 0: features[28 + k, :, :] = my_hand[k] / 10.0
        
    # 敵
    opp_hand = hand[0] if is_white else hand[1]
    for k in range(7):
        if opp_hand[k] > 0: features[35 + k, :, :] = opp_hand[k] / 10.0

    # 3. 利きのヒートマップ (cshogiのビットボード活用で高速化可能だが、今回は簡易実装)
    # v14 speed hack: Pythonループでの利き計算は重すぎるので一旦スキップするか、
    # 余裕があれば実装する。DirectMLでCPUネックになりがちなので、ここは「手数」だけにする。
    # ★速度優先のため、利き計算(42, 43ch)は0のままにするか、軽量な処理にする
    # (ここがボトルネックの主犯なので)
    
    # 4. 手数
    features[44, :, :] = min(board.move_number / 300.0, 1.0)

    # 5. 王手フラグ
    # cshogiは is_check() がない？ -> is_in_check(color)
    if board.is_check():
        features[45, :, :] = 1.0

    return features

# 回転処理 (文字列操作はPythonでも速い)
TBL_NUM = str.maketrans("123456789", "987654321")
TBL_ALP = str.maketrans("abcdefghi", "ihgfedcba")
def rotate_move(move_usi):
    if "*" in move_usi:
        return move_usi[0] + "*" + move_usi[2:].translate(TBL_NUM).translate(TBL_ALP)
    return move_usi[:2].translate(TBL_NUM).translate(TBL_ALP) + \
           move_usi[2:4].translate(TBL_NUM).translate(TBL_ALP) + move_usi[4:]

class DistillationDataset(Dataset):
    def __init__(self, txt_path, vocab):
        self.vocab = vocab
        self.samples = []
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                self.samples = [line.strip() for line in f]
        print(f"📖 Loaded {len(self.samples)} samples (DirectML Optimized).")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        line = self.samples[idx]
        parts = line.split()
        sfen = " ".join(parts[:-2])
        move_usi = parts[-2]
        score_cp = int(parts[-1])
        
        # ★cshogiで高速読み込み
        board = cshogi.Board(sfen)
        x = make_features_cshogi(board)
        
        # 正解手の回転
        if board.turn == cshogi.WHITE: 
            move_label = rotate_move(move_usi)
        else: 
            move_label = move_usi
            
        move_id = self.vocab.get(move_label, -100)
        win_rate = 1.0 / (1.0 + math.exp(-score_cp / 600.0))
        
        return (torch.from_numpy(x), torch.tensor(move_id, dtype=torch.long), torch.tensor(win_rate, dtype=torch.float32))

# ==========================================
# 🧠 Model (ReLU Edition for Stability)
# ==========================================
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True), # ★SiLU -> ReLU
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_s, max_s], dim=1)
        spatial_att = self.sigmoid(self.conv_spatial(spatial))
        return x * spatial_att

class DainagonBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.cbam = CBAMBlock(channels)
        self.act = nn.ReLU(inplace=True) # ★SiLU -> ReLU

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(x))
        out = self.conv1(out)
        out = self.act(self.bn2(out))
        out = self.conv2(out)
        out = self.cbam(out)
        return out + residual

class DainagonNet(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, CHANNELS, 3, padding=1, bias=False),
            nn.BatchNorm2d(CHANNELS),
            nn.ReLU(inplace=True) # ★SiLU -> ReLU
        )
        self.res_blocks = nn.ModuleList([DainagonBlock(CHANNELS) for _ in range(BLOCKS)])
        self.fc_input_dim = CHANNELS * 9 * 9
        self.dropout = nn.Dropout(p=0.3)
        self.policy_fc = nn.Linear(self.fc_input_dim, 1024)
        self.policy_act = nn.ReLU(inplace=True) # ★SiLU -> ReLU
        self.policy_head = nn.Linear(1024, num_moves)
        self.value_head = nn.Sequential(
            nn.Linear(self.fc_input_dim + 1024, 256), 
            nn.ReLU(inplace=True), # ★SiLU -> ReLU
            nn.Dropout(p=0.3),
            nn.Linear(256, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.conv_input(x)
        for block in self.res_blocks: h = block(h)
        h = h.view(-1, self.fc_input_dim)
        h = self.dropout(h)
        p_feat = self.policy_act(self.policy_fc(h))
        policy_out = self.policy_head(p_feat)
        v_input = torch.cat([h, p_feat], dim=1)
        value_out = self.value_head(v_input)
        return policy_out, value_out

# ==========================================
# 学習ループ (DirectML Optimized)
# ==========================================
def plot_learning_curve(loss_history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Dainagon Learning Curve (DirectML)')
    plt.savefig(save_path)
    plt.close()

def train():
    os.makedirs("model", exist_ok=True)
    with open(VOCAB_FILE, 'r') as f: vocab = json.load(f)
    
    print("📖 データセット読み込み中 (cshogi)...")
    dataset = DistillationDataset(DATA_FILE, vocab)
    
    # ★Windows DirectMLの鉄則: num_workers=0
    # cshogiのおかげで0でも速い！
    loader = DataLoader(
        dataset, 
        batch_size=PHYSICAL_BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False # DirectMLではFalse推奨の場合あり
    )
    
    model = DainagonNet(len(vocab)).to(DEVICE)
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🔄 続きから再開: {MODEL_SAVE_PATH}")
        # DirectMLではmap_location=DEVICEではなくcpuに読んでから送るのが安全
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_p = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_v = nn.MSELoss()
    
    print("=== Training v14 Started (DirectML Speed King) ===")
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (x, t_m, t_v) in enumerate(pbar):
            # データ転送
            x = x.to(DEVICE)
            t_m = t_m.to(DEVICE)
            t_v = t_v.to(DEVICE)
            
            # 順伝播
            p, v = model(x)
            
            loss = (POLICY_WEIGHT * criterion_p(p, t_m)) + \
                   (VALUE_WEIGHT * criterion_v(v.squeeze(), t_v))
            loss = loss / ACCUM_STEPS
            
            # 逆伝播 (AMPなし・Float32)
            loss.backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                current_loss = loss.item() * ACCUM_STEPS
                loss_history.append(current_loss)
                pbar.set_postfix({"Loss": f"{current_loss:.4f}"})
            
            total_loss += loss.item() * ACCUM_STEPS
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
        
        torch.save({'model': model.state_dict(), 'vocab': vocab}, MODEL_SAVE_PATH)
        plot_learning_curve(loss_history, GRAPH_SAVE_PATH)

if __name__ == "__main__":
    train()