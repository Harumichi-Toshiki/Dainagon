import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import os
import math
import copy
import shogi
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
DEVICE = torch_directml.device()
DATA_DIR = r"E:\DainagonProject（E）\traindata"
TEST_DATA_DIR = r"C:\Users\harum\Desktop\MyProject\DainagonProject\data\train_test"
MODEL_SAVE_PATH = "model/chunagon_v28.pt" 
GRAPH_SAVE_PATH = "model/learning_curve_v28.png" 

# ハイパーパラメータ
EPOCHS              = 3
INPUT_CHANNELS      = 55
LEARNING_RATE       = 3e-4
WEIGHT_DECAY        = 1e-3
WARMUP_STEPS        = 2000
ETA_MIN_RATIO       = 1e-3
TARGET_BATCH_SIZE   = 2048
PHYSICAL_BATCH_SIZE = 256
ACCUM_STEPS         = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE
VALUE_LOSS_SCALE    = 1.5
SAVE_INTERVAL_SEC   = 1800
BACKUP_SAVE_INTERVAL_SEC = 36000
POSITIONS_PER_EPOCH = 100_000_000  # 1エポックあたりの局面数（1億局面）
SHUFFLE_BUFFER_SIZE = 500_000      # シャッフルバッファ（大きくするとより混ざる。50万件で約19MB）
CHUNK_SIZE          = 4096         # 1回のランダムアクセスで読み込む連続件数（IO速度担保）
MAE_RATE            = 0.20
SCORE_CUTLINE       = 100000

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
# 🛠️ ビットボード展開関数
# ==========================================
def bb_to_squares(bb):
    """整数のビットボードから、立っているビットの座標を返すジェネレータ"""
    while bb > 0:
        r = bb & -bb
        yield r.bit_length() - 1
        bb ^= r

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
# ⚡ 特徴量生成 (python-shogi 55ch版フル実装)
# ==========================================
def make_features_v16(board, is_white):
    features = np.zeros((INPUT_CHANNELS, 9, 9), dtype=np.float32)
    
    # 1. 盤上の駒 (0-27)
    for sq in range(81):
        p = board.piece_at(sq)
        if p:
            if p.color == board.turn: # 手番側
                idx = p.piece_type - 1
            else: # 相手側
                idx = p.piece_type - 1 + 14
                
            target_sq = (80 - sq) if is_white else sq
            y, x = divmod(target_sq, 9)
            features[idx, y, x] = 1.0

    # 2. 持ち駒 (28-41)
    piece_order = [shogi.PAWN, shogi.LANCE, shogi.KNIGHT, shogi.SILVER, shogi.GOLD, shogi.BISHOP, shogi.ROOK]
    colors = [(board.turn, 28), (not board.turn, 35)]
    for color, offset in colors:
        hand = board.pieces_in_hand[color]
        for i, p_type in enumerate(piece_order):
            count = hand[p_type]
            if count > 0:
                features[offset + i, :, :] = count / 10.0

    # 3. 大駒の利き (42-45)
    for sq in range(81):
        p = board.piece_at(sq)
        if not p: continue
        pt = p.piece_type
        if pt not in [shogi.ROOK, shogi.PROM_ROOK, shogi.BISHOP, shogi.PROM_BISHOP]:
            continue
            
        is_self = (p.color == board.turn)
        is_rook = (pt == shogi.ROOK or pt == shogi.PROM_ROOK)
        
        if is_self: ch = 42 if is_rook else 43
        else:       ch = 44 if is_rook else 45
        
        attacks_bb = board.attacks_from(pt, sq, board.occupied, p.color)
        
        for t_sq in bb_to_squares(attacks_bb):
            target_sq = (80 - t_sq) if is_white else t_sq
            y, x = divmod(target_sq, 9)
            features[ch, y, x] = 1.0

    # 4. 玉の周辺 (46-49)
    k_sq_self = board.king_squares[board.turn]
    k_sq_opp = board.king_squares[not board.turn]
    kings = [(k_sq_self, 46), (k_sq_opp, 48)]
    
    for k_sq, base_ch in kings:
        if k_sq is None: continue
        ky, kx = divmod(k_sq, 9)
        
        for r, r_ch in [(3, 1), (2, 0)]:
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    ny, nx = ky + dy, kx + dx
                    if 0 <= ny < 9 and 0 <= nx < 9:
                        orig_sq = ny * 9 + nx
                        target_sq = (80 - orig_sq) if is_white else orig_sq
                        ty, tx = divmod(target_sq, 9)
                        features[base_ch + r_ch, ty, tx] = 1.0

    # 5. その他 (50-54)
    if board.is_check(): features[50, :, :] = 1.0
    # ch51: 予約（白紙）
    # features[51] は常に 0.0 のまま (エンジン側と整合済み)
    
    for i in range(9):
        features[52, :, i] = i / 8.0  # File (筋: 0〜8)
        features[53, i, :] = i / 8.0  # Rank (段: 0〜8)
    # ch54: 予約（白紙）
    # features[54] は常に 0.0 のまま (エンジン側と整合済み)
        
    return features

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

        cb.set_hcp(hcp)
        move_usi = cshogi.move_to_usi(move16)
        is_white = (cb.turn == cshogi.WHITE)

        sfen = cb.sfen()
        board = shogi.Board(sfen)
        x = make_features_v16(board, is_white) # 既存の特徴量関数を呼び出す
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
        
        return (torch.from_numpy(x),
                torch.tensor(move_id, dtype=torch.long),
                torch.tensor(target_prob, dtype=torch.float32), 
                torch.tensor(score_cp, dtype=torch.float32))

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

                cb.set_hcp(hcp)
                move_usi = cshogi.move_to_usi(move16)
                is_white = (cb.turn == cshogi.WHITE)

                sfen = cb.sfen()
                board = shogi.Board(sfen)
                x = make_features_v16(board, is_white)

                if is_white: move_usi = self._rotate_move(move_usi)

                try: move_id = usi_to_action(move_usi)
                except ValueError: continue

                #win_rate = 1.0 / (1.0 + math.exp(-score_cp / 600.0))
                target_logit = score_cp / 600.0

                yield (torch.from_numpy(x),
                       torch.tensor(move_id, dtype=torch.long),
                       torch.tensor(score_cp, dtype=torch.float32))
                count += 1

def evaluate(model, val_loader):
    model.eval() # 🚨 推論モード（Valueには自動的にSigmoidがかかる）
    correct = 0
    total = 0
    val_loss_p = 0.0
    val_loss_v_logit = 0.0
    val_loss_v_prob = 0.0
    batch_count = 0

    criterion_p = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()

    with torch.no_grad():
        for x, t_m, t_score in val_loader:
            x, t_m, t_score = x.to(DEVICE), t_m.to(DEVICE), t_score.to(DEVICE)
            
            # p: ロジット(2187), v: 勝率(0.0~1.0)
            p, v = model(x)
            
            # 1. Policy 精度と Loss (CrossEntropy)
            pred_move = p.argmax(1)
            correct += (pred_move == t_m).sum().item()
            total += t_m.size(0)
            val_loss_p += criterion_p(p, t_m).item()
            
            # 2. 正解データの作成 (水匠定数600)
            t_v_logit = t_score / 600.0
            t_v_prob = 1.0 / (1.0 + torch.exp(-t_v_logit))
            
            # 3. Value Loss計算
            # ① 勝率空間のMSE (Vision版と同じ、直感的な勝率の誤差)
            val_loss_v_prob += criterion_v(v.squeeze(), t_v_prob).item()
            
            # ② ロジット空間のMSE (訓練時の純粋な誤差と比較するため)
            # vを逆Sigmoid関数(logit)で元のロジットに戻してから比較する
            v_logit = torch.logit(v.squeeze(), eps=1e-7)
            val_loss_v_logit += criterion_v(v_logit, t_v_logit).item()
            
            batch_count += 1
            
    acc = correct / total if total > 0 else 0
    avg_loss_p = val_loss_p / batch_count if batch_count > 0 else 0
    avg_loss_v_prob = val_loss_v_prob / batch_count if batch_count > 0 else 0
    avg_loss_v_logit = val_loss_v_logit / batch_count if batch_count > 0 else 0
    
    return acc, avg_loss_p, avg_loss_v_logit, avg_loss_v_prob

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
    
    # 全体容量の推定は情報として残す
    total_records, num_files = estimate_dataset_size(DATA_DIR)
    
    # 🚨 学習の基準を POSITIONS_PER_EPOCH (1億) に固定する
    actual_positions_per_epoch = min(total_records, POSITIONS_PER_EPOCH)
    total_batches_per_epoch = actual_positions_per_epoch // PHYSICAL_BATCH_SIZE
    total_opt_steps_per_epoch = total_batches_per_epoch // ACCUM_STEPS
    total_opt_steps = total_opt_steps_per_epoch * EPOCHS
    
    print(f"[INFO] 全データ: {num_files} ファイル / 推定 {total_records:,} 局面")
    print(f"[INFO] 1エポックの学習サイズ: {actual_positions_per_epoch:,} 局面")
    print(f"[INFO] 総最適化ステップ数: {total_opt_steps:,} ステップ")

    # データローダー
    NUM_WORKERS = 4
    dataset = MultiFileHCPEDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=PHYSICAL_BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS>0))
    val_dataset = ValidationDataset(TEST_DATA_DIR)
    val_loader = DataLoader(val_dataset, batch_size=PHYSICAL_BATCH_SIZE, num_workers=0)

    # 🚀 ネットワークの動的ロード
    model = get_network(args.network).to(DEVICE)

    # Optimizer & Scheduler
    _no_decay_kw = ('bias', 'norm', 'bn')
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if not any(kw in n for kw in _no_decay_kw)], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(kw in n for kw in _no_decay_kw)], 'weight_decay': 0.0},
    ]
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

        for i, (x, t_m, t_v, t_score) in enumerate(pbar):
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
            t_v = t_v.to(DEVICE)
            t_score = t_score.to(DEVICE)

            # 🎯 ターゲットスムージング (MSEでは不要ですが、あっても無害なので残してもOK)
            #t_v = torch.clamp(t_v, min=0.005, max=0.995)

            # Forward (🚨 訓練時は p, v に加えて x_hat が返ってくる)
            # 隠された masked_x を入力し、モデルに元の x を想像させる
            p, v, x_hat = model(x)
            p = torch.clamp(p, min=-10.0, max=10.0)

            # --- ⚔️ Loss 計算 ---
            # 1. Policy Loss
            loss_p = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)(p, t_m)
            
            # 2. Value Loss (🚨 BCEから、生ロジット同士のMSELossに変更！)
            loss_fn_v = nn.BCEWithLogitsLoss(reduction='none')
            bce_loss = loss_fn_v(v.squeeze(), t_v)
            
            weight_v = torch.where(torch.abs(t_score) <= 2000, 2.0, 0.2)
            loss_v = (bce_loss * weight_v).mean() * VALUE_LOSS_SCALE
            
            # 3. MAE Loss (🚨 隠された入力から、元の完璧な盤面 x を復元させるMSE)
            #loss_mae = nn.MSELoss()(x_hat, x) * 2.0 # スケール調整
            loss_mae = nn.BCEWithLogitsLoss()(x_hat, x) * 2.0

            # 🚨 3つのLossを合算
            loss = (loss_p + loss_v + loss_mae) / ACCUM_STEPS
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
                    "L_m": f"{loss_mae.item():.4f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
                })

                # ⏱️ 30分ごとの保存と評価
                current_time = time.time()
                if current_time - last_save_time > SAVE_INTERVAL_SEC:
                    
                    # 🚨 追加: テスト評価の実行
                    acc, val_loss_p, val_loss_v_logit, val_loss_v_prob = evaluate(model, val_loader)
                    print(f"\n[EVAL] Policy精度: {acc:.2%}, Policy Loss: {val_loss_p:.4f}, Value Loss(Logit): {val_loss_v_logit:.4f}, Value MSE(勝率): {val_loss_v_prob:.5f}")
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
            del p, v, x_hat, loss_p, bce_loss, loss_v, loss_mae, loss
            gc_counter += 1
            if gc_counter % 100 == 0:
                import gc
                gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dainagon SOTA Training Script")
    parser.add_argument(
        "--network", 
        type=str, 
        default="network.network_chunagon_v28.ChunagonNetV3",
        help="ロードするネットワークのパス (例: network.network_chunagon_v25.ChunagonNetV3)"
    )
    parser.add_argument("--resume", action="store_true", help="チェックポイントから再開する")
    args = parser.parse_args()

    train(args)