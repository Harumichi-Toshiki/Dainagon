import torch
import torch.nn as nn

# ==========================================
# ⚙️ 基本設定 (15b フルサイズ純粋ResNet仕様)
# ==========================================
INPUT_CHANNELS = 55
CHANNELS = 256 # 🚨 1000万パラメータに抑えるなら 192 に変更推奨

# ==========================================
# 🟦 CBAM (Channel & Spatial Attention)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        hidden = max(8, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_pool = x.mean(dim=(2, 3), keepdim=True)
        max_pool = x.amax(dim=(2, 3), keepdim=True)
        return self.sigmoid(self.fc2(self.relu(self.fc1(avg_pool))) + self.fc2(self.relu(self.fc1(max_pool))))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

# ==========================================
# 🧩 1. 純粋フルサイズ ResNet ブロック (Nestedなし)
# ==========================================
class FullResNetBlock(nn.Module):
    """
    圧縮(1x1)を行わず、CHANNELSのまま3x3を2回かける最もハードウェアに優しい構造。
    """
    def __init__(self, channels=CHANNELS):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

# ==========================================
# 🧩 2. CBAM付き フルサイズ ResNet ブロック
# ==========================================
class CBAMFullResNetBlock(nn.Module):
    """
    大局観を補うため、フルサイズResNetの出力にCBAM(空間＆チャンネル注意)を掛け合わせる。
    """
    def __init__(self, channels=CHANNELS):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out) * out
        out = self.sa(out) * out
        return self.relu(out + res)

# ==========================================
# 🧠 Dainagon 15b フルサイズ 本体
# ==========================================
class ChunagonNetV3(nn.Module): # 💡 train.pyの呼び出しに合わせるため名前は維持
    def __init__(self):
        super().__init__()
        
        # --- (1) Stem ---
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(CHANNELS),
            nn.ReLU(inplace=True)
        )

        # --- (2) 15ブロックの太い幹 (純粋ResNet + たまにCBAM) ---
        blocks = []
        for i in range(1, 16):
            # 💡 5層、10層、15層目（要所）にのみCBAMを配置し、大局観を補正
            if i in {5, 10, 15}: 
                blocks.append(CBAMFullResNetBlock(CHANNELS))
            else:
                blocks.append(FullResNetBlock(CHANNELS))
        self.trunk = nn.Sequential(*blocks)

        # 🚨 Y-shapeの枝は廃止し、最速の直接出力へ

        # --- (3) Heads ---
        self.policy_head = nn.Conv2d(CHANNELS, 27, 1)

        self.value_conv = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 1, bias=False), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True)
        )
        self.value_head = nn.Sequential(
            nn.Linear(32 * 81, 256), 
            nn.ReLU(inplace=True), 
            nn.Linear(256, 1)
        )

        # 初期局面バグ対策のゼロ初期化
        nn.init.constant_(self.value_head[-1].bias, 0.0)
        nn.init.normal_(self.value_head[-1].weight, mean=0.0, std=0.01)

        # --- (4) MAE (訓練時用) ---
        self.mae_dropout = nn.Dropout2d(p=0.15)
        self.mae_conv = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mae_head = nn.Conv2d(64, INPUT_CHANNELS, kernel_size=1)

    def forward(self, x):
        # 1. 幹を15ブロック貫通
        h = self.stem(x)
        h = self.trunk(h)
        
        # 2. Heads (直接出力)
        p = self.policy_head(h).view(-1, 2187)
        v = self.value_head(self.value_conv(h).view(-1, 32 * 81))
        
        # 3. MAE & 出力切り替え
        if self.training:
            h_mae = self.mae_dropout(h)
            x_hat = self.mae_head(self.mae_conv(h_mae))
            return p, v, x_hat
        else:
            # 💡 以前実証された通り、C++エンジンは確率(0.0~1.0)を待っているためSigmoidは必須
            v = torch.sigmoid(v) 
            return p, v