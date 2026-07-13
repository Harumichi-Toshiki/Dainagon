import torch
import torch.nn as nn
import math

# ==========================================
# ⚙️ 基本設定 (1004万パラメータ仕様)
# ==========================================
INPUT_CHANNELS = 55
CHANNELS = 256
MID_CHANNELS = 64  # 1/4 Nested ボトルネック用

# ==========================================
# 🧩 1. Ryfcブロック (飛角十字インセプション)
# ==========================================
class RyfcNestedBlock(nn.Module):
    def __init__(self, channels=CHANNELS, mid_channels=MID_CHANNELS):
        super().__init__()
        self.reduce = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(mid_channels)
        
        self.conv_v = nn.Conv2d(mid_channels, mid_channels, kernel_size=(9, 1), padding=(4, 0), bias=False)
        self.conv_h = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 9), padding=(0, 4), bias=False)
        self.conv_1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        
        self.bn_v = nn.BatchNorm2d(mid_channels)
        self.bn_h = nn.BatchNorm2d(mid_channels)
        self.bn_1 = nn.BatchNorm2d(mid_channels)
        
        self.expand = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        out = self.relu(self.bn_reduce(self.reduce(x)))
        cross_out = self.relu(self.bn_v(self.conv_v(out)) + self.bn_h(self.conv_h(out)) + self.bn_1(self.conv_1(out)))
        out = self.bn_expand(self.expand(cross_out))
        return self.relu(out + res)

# ==========================================
# 🧩 2. スタンダード Nested ブロック (3x3)
# ==========================================
class StandardNestedBlock(nn.Module):
    def __init__(self, channels=CHANNELS, mid_channels=MID_CHANNELS):
        super().__init__()
        self.reduce = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(mid_channels)
        self.conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn_conv = nn.BatchNorm2d(mid_channels)
        self.expand = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        out = self.relu(self.bn_reduce(self.reduce(x)))
        out = self.relu(self.bn_conv(self.conv(out)))
        out = self.bn_expand(self.expand(out))
        return self.relu(out + res)

# ==========================================
# 🧩 3. ラージカーネル Nested ブロック (7x7)
# ==========================================
class LargeKernelNestedBlock(nn.Module):
    def __init__(self, channels=CHANNELS, mid_channels=MID_CHANNELS):
        super().__init__()
        self.reduce = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(mid_channels)
        self.dwconv = nn.Conv2d(mid_channels, mid_channels, kernel_size=7, padding=3, groups=mid_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid_channels)
        self.expand = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        out = self.relu(self.bn_reduce(self.reduce(x)))
        out = self.relu(self.bn_dw(self.dwconv(out)))
        out = self.bn_expand(self.expand(out))
        return self.relu(out + res)

# ==========================================
# 🧩 4. Transformer ブロック (ONNX互換)
# ==========================================
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels=CHANNELS, height=9, width=9):
        super().__init__()
        half_ch = channels // 2
        self.file_embed = nn.Parameter(torch.randn(1, width, half_ch) * 0.02)
        self.rank_embed = nn.Parameter(torch.randn(height, 1, half_ch) * 0.02)

    def forward(self, x):
        f_emb = self.file_embed.expand(9, -1, -1)
        r_emb = self.rank_embed.expand(-1, 9, -1)
        pos_2d = torch.cat([f_emb, r_emb], dim=-1).view(81, -1).unsqueeze(0)
        return x + pos_2d.to(x.device)

class ONNXCompatibleAttention(nn.Module):
    def __init__(self, channels=CHANNELS, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.q_linear = nn.Linear(channels, channels, bias=False)
        self.k_linear = nn.Linear(channels, channels, bias=False)
        self.v_linear = nn.Linear(channels, channels, bias=False)
        self.out_linear = nn.Linear(channels, channels, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_linear(x).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(x).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(x).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.out_linear(out)

class HybridTransformerBlock(nn.Module):
    def __init__(self, channels=CHANNELS):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.pos_encoder = PositionalEncoding2D(channels)
        self.attn = ONNXCompatibleAttention(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(inplace=True), 
            nn.Linear(channels * 4, channels)
        )
        self.gate_attn = nn.Parameter(torch.tensor([0.01]))
        self.gate_mlp = nn.Parameter(torch.tensor([0.01]))

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        out = self.bn1(x) 
        out = out.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        out = self.pos_encoder(out)
        out = self.attn(out)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = residual + self.gate_attn * out
        
        residual = x
        out = self.bn2(x)
        out = out.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        out = self.mlp(out)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = residual + self.gate_mlp * out
        return x

# ==========================================
# 🧠 中納言 v27 本体 (Y-shape バンパー付き最強版)
# ==========================================
class ChunagonNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- (1) 拡張レシーバー (Stem) ---
        self.stem_conv = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(CHANNELS),
            nn.ReLU(inplace=True)
        )
        self.stem_ryfc1 = RyfcNestedBlock(CHANNELS)
        self.stem_ryfc2 = RyfcNestedBlock(CHANNELS)

        # --- (2) 深層・共通幹 (Common Trunk) : 24ブロック ---
        common_blocks = []
        for i in range(1, 25):
            if i in {4, 9, 14}: 
                common_blocks.append(LargeKernelNestedBlock(CHANNELS))
            elif i in {7, 18, 22}: 
                common_blocks.append(RyfcNestedBlock(CHANNELS))
            elif i in {6, 12, 19, 24}: 
                common_blocks.append(HybridTransformerBlock(CHANNELS))
            else:
                common_blocks.append(StandardNestedBlock(CHANNELS))
        self.common_trunk = nn.Sequential(*common_blocks)

        # --- (3) Policy 専用枝 (5ブロック) ---
        policy_blocks = []
        for i in range(1, 6):
            policy_blocks.append(StandardNestedBlock(CHANNELS))
        self.policy_branch = nn.Sequential(*policy_blocks)
        self.policy_head = nn.Conv2d(CHANNELS, 27, 1)

        # --- (4) Value 専用枝 (5ブロック) ---
        value_blocks = []
        for i in range(1, 6):
            value_blocks.append(StandardNestedBlock(CHANNELS))
        self.value_branch = nn.Sequential(*value_blocks)
        
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

        nn.init.constant_(self.value_head[-1].bias, 0.0)
        nn.init.normal_(self.value_head[-1].weight, mean=0.0, std=0.01)

        # --- (5) MAE (訓練時用) ---
        self.mae_dropout = nn.Dropout2d(p=0.15)
        self.mae_conv = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mae_head = nn.Conv2d(64, INPUT_CHANNELS, kernel_size=1)

    def forward(self, x):
        # 1. Stem & 共通幹
        h = self.stem_conv(x)
        h = self.stem_ryfc1(h)
        h = self.stem_ryfc2(h)
        h = self.common_trunk(h)
        
        # 2. Y-shape 分岐 (バンパー)
        h_policy = self.policy_branch(h)
        h_value = self.value_branch(h)
        
        # 3. Heads
        p = self.policy_head(h_policy).view(-1, 2187)
        v = self.value_head(self.value_conv(h_value).view(-1, 32 * 81))
        
        # 4. MAE
        if self.training:
            h_mae = self.mae_dropout(h)
            x_hat = self.mae_head(self.mae_conv(h_mae))
            return p, v, x_hat
        else:
            v = torch.sigmoid(v)
            return p, v