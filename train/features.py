"""
features.py - 高速版 55ch 特徴量生成エンジン (cshogi 直接呼び出し)

【実機確認結果 (Section 10)】
- cshogi.NONE: 0
- 駒の先後判定: 先手駒コード 1..14 (p < 17), 後手駒コード 17..30 (p >= 17)
- 先手(cshogi.BLACK)=0, 後手(cshogi.WHITE)=1
- cshogi.piece_to_piece_type(p) の返り値:
    PAWN=1, LANCE=2, KNIGHT=3, SILVER=4, BISHOP=5, ROOK=6, GOLD=7, KING=8,
    PROM_PAWN=9, PROM_LANCE=10, PROM_KNIGHT=11, PROM_SILVER=12, PROM_BISHOP=13, PROM_ROOK=14
- cshogi.Board.pieces_in_hand[color] のインデックス順序:
    [0:歩, 1:香, 2:桂, 3:銀, 4:金, 5:角, 6:飛]
- 座標系:
    cshogi_file, rank_idx = divmod(sq, 9)
    y = rank_idx (0: a段, 8: i段)
    x = 8 - cshogi_file (0: 9筋, 8: 1筋)
"""

import numpy as np
import cshogi

INPUT_CHANNELS = 55

CSHOGI_PT_TO_SHOGI_IDX = np.array([0, 0, 1, 2, 3, 5, 6, 4, 7, 8, 9, 10, 11, 12, 13], dtype=np.int32)

ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# Pre-computed rank (y) and file (x) for each cshogi square (0..80)
CSHOGI_SQ_TO_Y = np.zeros(81, dtype=np.int32)
CSHOGI_SQ_TO_X = np.zeros(81, dtype=np.int32)

for sq in range(81):
    cshogi_file, rank_idx = divmod(sq, 9)
    CSHOGI_SQ_TO_Y[sq] = rank_idx
    CSHOGI_SQ_TO_X[sq] = 8 - cshogi_file

# Ramp values for ch52 (File) and ch53 (Rank)
RAMP = np.arange(9, dtype=np.float32) / 8.0

def make_features_v16_fast(cb: cshogi.Board, is_white: bool) -> np.ndarray:
    """
    cshogi.Board から55chの特徴量テンソルを生成する。
    features_legacy.make_features_v16(shogi.Board, is_white) と
    ビット単位で完全に同一の出力を返す。
    """
    features = np.zeros((INPUT_CHANNELS, 9, 9), dtype=np.float32)

    turn = cb.turn
    opp_turn = 1 - turn

    pieces = cb.pieces
    occupied = np.zeros((9, 9), dtype=bool)

    # 1. 盤上の駒 (0-27) & occupied grid (9x9)
    for cshogi_sq in range(81):
        p = pieces[cshogi_sq]
        if p == 0:
            continue

        y = CSHOGI_SQ_TO_Y[cshogi_sq]
        x = CSHOGI_SQ_TO_X[cshogi_sq]
        occupied[y, x] = True

        p_color = 1 if p >= 17 else 0
        is_self = (p_color == turn)

        cshogi_pt = cshogi.piece_to_piece_type(p)
        pt_idx = CSHOGI_PT_TO_SHOGI_IDX[cshogi_pt]

        idx = pt_idx if is_self else (pt_idx + 14)

        ty = (8 - y) if is_white else y
        tx = (8 - x) if is_white else x
        features[idx, ty, tx] = 1.0

    # 2. 持ち駒 (28-41)
    hand_self = cb.pieces_in_hand[turn]
    for i in range(7):
        count = hand_self[i]
        if count > 0:
            features[28 + i, :, :] = count / 10.0

    hand_opp = cb.pieces_in_hand[opp_turn]
    for i in range(7):
        count = hand_opp[i]
        if count > 0:
            features[35 + i, :, :] = count / 10.0

    # 3. 大駒の利き (42-45)
    for cshogi_sq in range(81):
        p = pieces[cshogi_sq]
        if p == 0:
            continue

        cshogi_pt = cshogi.piece_to_piece_type(p)
        if cshogi_pt not in (5, 6, 13, 14): # BISHOP, ROOK, PROM_BISHOP, PROM_ROOK
            continue

        p_color = 1 if p >= 17 else 0
        is_self = (p_color == turn)
        is_rook = (cshogi_pt == 6 or cshogi_pt == 14)

        ch = (42 if is_rook else 43) if is_self else (44 if is_rook else 45)

        y0 = CSHOGI_SQ_TO_Y[cshogi_sq]
        x0 = CSHOGI_SQ_TO_X[cshogi_sq]

        if cshogi_pt == 6: # ROOK
            dirs, single_dirs = ROOK_DIRS, ()
        elif cshogi_pt == 5: # BISHOP
            dirs, single_dirs = BISHOP_DIRS, ()
        elif cshogi_pt == 14: # PROM_ROOK
            dirs, single_dirs = ROOK_DIRS, BISHOP_DIRS
        else: # PROM_BISHOP
            dirs, single_dirs = BISHOP_DIRS, ROOK_DIRS

        # 光線射出
        for dy, dx in dirs:
            y, x = y0 + dy, x0 + dx
            while 0 <= y < 9 and 0 <= x < 9:
                ty = (8 - y) if is_white else y
                tx = (8 - x) if is_white else x
                features[ch, ty, tx] = 1.0
                if occupied[y, x]:
                    break
                y += dy
                x += dx

        # 隣接1マス
        for dy, dx in single_dirs:
            y, x = y0 + dy, x0 + dx
            if 0 <= y < 9 and 0 <= x < 9:
                ty = (8 - y) if is_white else y
                tx = (8 - x) if is_white else x
                features[ch, ty, tx] = 1.0

    # 4. 玉の周辺 (46-49)
    k_sq_self = cb.king_square(turn)
    k_sq_opp = cb.king_square(opp_turn)

    for k_sq, base_ch in ((k_sq_self, 46), (k_sq_opp, 48)):
        if 0 <= k_sq < 81:
            ky = CSHOGI_SQ_TO_Y[k_sq]
            kx = CSHOGI_SQ_TO_X[k_sq]

            for r, r_ch in ((3, 1), (2, 0)):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        ny, nx = ky + dy, kx + dx
                        if 0 <= ny < 9 and 0 <= nx < 9:
                            ty = (8 - ny) if is_white else ny
                            tx = (8 - nx) if is_white else nx
                            features[base_ch + r_ch, ty, tx] = 1.0

    # 5. その他 (50-54)
    if cb.is_check():
        features[50, :, :] = 1.0

    features[52, :, :] = RAMP
    features[53, :, :] = RAMP[:, None]

    return features
