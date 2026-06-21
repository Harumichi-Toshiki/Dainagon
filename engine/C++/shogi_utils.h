#pragma once
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <map>
#include <onnxruntime_cxx_api.h>
#include "bitboard.h"

// 定数定義
//enum Color { BLACK = 0, WHITE = 1 };
//enum Piece {
//    NO_PIECE = 0, PAWN = 1, LANCE = 2, KNIGHT = 3, SILVER = 4, GOLD = 5, BISHOP = 6, ROOK = 7, KING = 8,
//    PROM_PAWN = 9, PROM_LANCE = 10, PROM_KNIGHT = 11, PROM_SILVER = 12,
//    PROM_BISHOP = 14, PROM_ROOK = 15
//}; // 13欠番

// 方向定数
const int DIR_UP = -9; const int DIR_UP_LEFT = -10; const int DIR_UP_RIGHT = -8;
const int DIR_LEFT = -1; const int DIR_RIGHT = 1;
const int DIR_DOWN = 9; const int DIR_DOWN_LEFT = 8; const int DIR_DOWN_RIGHT = 10;

inline std::string sq_to_usi(int sq) {
    int x = 9 - (sq % 9);
    int y = (sq / 9) + 1;
    return std::string{ (char)('0' + x), (char)('a' + y - 1) };
}

inline std::string move_to_usi(const Move& m) {
    if (m.from == -1 && m.to == -1) return "resign";
    if (m.is_drop) {
        char drop_chars[] = { '?', 'P', 'L', 'N', 'S', 'G', 'B', 'R' };
        return std::string{ drop_chars[m.type], '*' } + sq_to_usi(m.to);
    }
    else {
        std::string s = sq_to_usi(m.from) + sq_to_usi(m.to);
        if (m.is_promote) s += "+";
        return s;
    }
}

struct StateInfo {
    int captured_piece;           // 取られた駒
    int last_to_sq;               // 直前の着手マス
    uint64_t previous_zobrist_key;// 1手前のハッシュキー
    int previous_king_sq;         // 動かす前の玉の位置
    Position previous_bit_pos;    // ビットボードのバックアップ（安全に復元するため）
};

struct Board {
    int pieces[81];
    int hand[2][8];
    int turn;
    int move_count;
    int king_sq[2];
    int last_to_sq;

    Position bit_pos;

    uint64_t zobrist_key;

    Board() { reset(); }

    uint64_t get_zobrist_key() const { return zobrist_key; }

    // ★追加：最初だけ全計算するための関数宣言
    uint64_t calc_zobrist_key_full() const;

    void update_bitboard() {
        bit_pos.set_from_array(pieces, turn, hand);
    }

    void reset() {
        std::fill(pieces, pieces + 81, 0);
        std::memset(hand, 0, sizeof(hand));
        turn = BLACK;
        move_count = 1;
        last_to_sq = -1;

        int setup[] = { 2,3,4,5,8,5,4,3,2 };
        for (int i = 0; i < 9; i++) {
            put_piece(i, 0, setup[i] + 16); // 後手(0段目)
            put_piece(i, 8, setup[i]);      // 先手(8段目)
            put_piece(i, 2, 1 + 16);
            put_piece(i, 6, 1);
        }
        put_piece(1, 7, 6); put_piece(7, 7, 7);
        put_piece(7, 1, 6 + 16); put_piece(1, 1, 7 + 16);
        update_bitboard();
        zobrist_key = calc_zobrist_key_full();
    }

    void put_piece(int x, int y, int val) {
        int sq = y * 9 + x;
        pieces[sq] = val;
        if ((val & 15) == KING) king_sq[(val > 16) ? WHITE : BLACK] = sq;
    }

    //std::string sq_to_usi(int sq) const {
    //    int x = 9 - (sq % 9);
    //    int y = (sq / 9) + 1;
    //    char c1 = '0' + x;
    //    char c2 = 'a' + (y - 1);
    //    return std::string{ c1, c2 };
    //}

    void set_sfen(const std::string& sfen) {
        if (sfen == "startpos") { reset(); return; }
        reset();
    }

    void push_usi(std::string usi) {
        move_count++;
        int from_sq, to_sq;
        bool promote = (usi.back() == '+');
        if (promote) usi.pop_back();

        auto parse_sq = [](char c1, char c2) { return (c2 - 'a') * 9 + ('9' - c1); };

        if (usi[1] == '*') {
            int type = 0;
            switch (usi[0]) {
            case 'P': type = PAWN; break; case 'L': type = LANCE; break;
            case 'N': type = KNIGHT; break; case 'S': type = SILVER; break;
            case 'G': type = GOLD; break; case 'B': type = BISHOP; break; case 'R': type = ROOK; break;
            }
            to_sq = parse_sq(usi[2], usi[3]);
            hand[turn][type]--;
            pieces[to_sq] = type + (turn == WHITE ? 16 : 0);
            last_to_sq = to_sq;
        }
        else {
            from_sq = parse_sq(usi[0], usi[1]);
            to_sq = parse_sq(usi[2], usi[3]);
            int val = pieces[from_sq];
            pieces[from_sq] = 0;
            int captured = pieces[to_sq];
            if (captured > 0) {
                int type = captured & 15;
                if (type > 8) type -= 8;
                hand[turn][type]++;
            }
            if (promote) val += 8;
            pieces[to_sq] = val;
            if ((val & 15) == KING) king_sq[turn] = to_sq;
            last_to_sq = to_sq;
        }
        turn = 1 - turn;
        update_bitboard();
        zobrist_key = calc_zobrist_key_full();
    }

#define IS_VALID_XY(x, y) ((x) >= 0 && (x) < 9 && (y) >= 0 && (y) < 9)

    // ★ ステップ3：超高速・逆レイキャスト版 is_attacked
    bool is_attacked(int sq, int attacker_color) const {
        int kx = sq % 9;
        int ky = sq / 9;
        // attacker_colorから見た「前」の方向 (BLACK=0なら上(-1)、WHITE=1なら下(+1))
        int fw = (attacker_color == BLACK) ? -1 : 1;

        // --- 1. 飛び駒（香・飛・角・龍・馬）の光線チェック ---
        // 直線(上下左右)
        int dx4[4] = { 0, 0, -1, 1 };
        int dy4[4] = { -1, 1, 0, 0 };
        for (int d = 0; d < 4; ++d) {
            for (int step = 1; step < 9; ++step) {
                int nx = kx + dx4[d] * step;
                int ny = ky + dy4[d] * step;
                if (!IS_VALID_XY(nx, ny)) break; // 盤外に出たら光線ストップ

                int p = pieces[ny * 9 + nx];
                if (p != 0) { // 何かの駒にぶつかった
                    if ((p > 16 ? WHITE : BLACK) == attacker_color) {
                        int type = p & 15;
                        if (type == ROOK || type == PROM_ROOK) return true;
                        // 敵の香車（敵から見て「前」に進む光線のみ）
                        if (type == LANCE && dx4[d] == 0 && dy4[d] == -fw) return true;
                    }
                    break; // 自分の駒、または攻撃権のない敵駒が「壁」になったのでこの方向は安全
                }
            }
        }

        // 斜め(4方向)
        int dxd[4] = { -1, 1, -1, 1 };
        int dyd[4] = { -1, -1, 1, 1 };
        for (int d = 0; d < 4; ++d) {
            for (int step = 1; step < 9; ++step) {
                int nx = kx + dxd[d] * step;
                int ny = ky + dyd[d] * step;
                if (!IS_VALID_XY(nx, ny)) break;

                int p = pieces[ny * 9 + nx];
                if (p != 0) {
                    if ((p > 16 ? WHITE : BLACK) == attacker_color) {
                        int type = p & 15;
                        if (type == BISHOP || type == PROM_BISHOP || type == PROM_ROOK) return true;
                    }
                    break;
                }
            }
        }

        // --- 2. 近接駒（歩・桂）のピンポイントチェック ---
        // 歩 (敵の歩が一つ手前にいるか)
        if (IS_VALID_XY(kx, ky - fw)) {
            int p = pieces[(ky - fw) * 9 + kx];
            if (p != 0 && (p > 16 ? WHITE : BLACK) == attacker_color && (p & 15) == PAWN) return true;
        }
        // 桂馬 (敵の桂馬がピョンと飛んでくる位置にいるか)
        if (IS_VALID_XY(kx - 1, ky - fw * 2)) {
            int p = pieces[(ky - fw * 2) * 9 + (kx - 1)];
            if (p != 0 && (p > 16 ? WHITE : BLACK) == attacker_color && (p & 15) == KNIGHT) return true;
        }
        if (IS_VALID_XY(kx + 1, ky - fw * 2)) {
            int p = pieces[(ky - fw * 2) * 9 + (kx + 1)];
            if (p != 0 && (p > 16 ? WHITE : BLACK) == attacker_color && (p & 15) == KNIGHT) return true;
        }

        // --- 3. その他の近接駒（銀・金・玉・成駒）の周囲8マスチェック ---
        int dx8[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };
        int dy8[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };
        for (int d = 0; d < 8; ++d) {
            int nx = kx + dx8[d];
            int ny = ky + dy8[d];
            if (IS_VALID_XY(nx, ny)) {
                int p = pieces[ny * 9 + nx];
                if (p != 0 && (p > 16 ? WHITE : BLACK) == attacker_color) {
                    int type = p & 15;
                    if (type == KING) return true; // 敵玉が隣接（あり得ないが念のため）

                    // 位置関係のフラグ
                    bool is_forward = (dy8[d] == -fw);
                    bool is_side = (dy8[d] == 0);
                    bool is_backward = (dy8[d] == fw);
                    bool is_straight = (dx8[d] == 0);
                    bool is_diag = (dx8[d] != 0 && dy8[d] != 0);

                    // 銀の攻撃範囲（前3方向 ＋ 後ろ斜め2方向）
                    if (type == SILVER && (is_forward || is_diag)) return true;
                    // 金・と金・成香・成桂・成銀の攻撃範囲（前3 ＋ 横2 ＋ 後ろ真っ直ぐ1）
                    if ((type == GOLD || type == PROM_PAWN || type == PROM_LANCE || type == PROM_KNIGHT || type == PROM_SILVER)
                        && (is_forward || is_side || (is_backward && is_straight))) return true;

                    // 龍の斜め1歩
                    if (type == PROM_ROOK && is_diag) return true;
                    // 馬の縦横1歩
                    if (type == PROM_BISHOP && (is_straight || is_side)) return true;
                }
            }
        }

        return false;
    }

    Bitboard get_attackers_to(int sq, int attacker_color) const {
        Bitboard attackers(0, 0); // 初期状態はゼロ（誰も攻撃していない）
        int kx = sq % 9;
        int ky = sq / 9;
        int fw = (attacker_color == BLACK) ? -1 : 1;

        // --- 1. 飛び駒（香・飛・角・龍・馬）の光線チェック ---
        int dx4[4] = { 0, 0, -1, 1 };
        int dy4[4] = { -1, 1, 0, 0 };
        for (int d = 0; d < 4; ++d) {
            for (int step = 1; step < 9; ++step) {
                int nx = kx + dx4[d] * step;
                int ny = ky + dy4[d] * step;
                if (!IS_VALID_XY(nx, ny)) break;

                int p = pieces[ny * 9 + nx];
                if (p != 0) {
                    if ((p > 16 ? WHITE : BLACK) == attacker_color) {
                        int type = p & 15;
                        // return true の代わりに、攻撃駒のマスを attackers に記録する
                        if (type == ROOK || type == PROM_ROOK) attackers.set(ny * 9 + nx);
                        if (type == LANCE && dx4[d] == 0 && dy4[d] == -fw) attackers.set(ny * 9 + nx);
                    }
                    break;
                }
            }
        }

        int dxd[4] = { -1, 1, -1, 1 };
        int dyd[4] = { -1, -1, 1, 1 };
        for (int d = 0; d < 4; ++d) {
            for (int step = 1; step < 9; ++step) {
                int nx = kx + dxd[d] * step;
                int ny = ky + dyd[d] * step;
                if (!IS_VALID_XY(nx, ny)) break;

                int p = pieces[ny * 9 + nx];
                if (p != 0) {
                    if ((p > 16 ? WHITE : BLACK) == attacker_color) {
                        int type = p & 15;
                        if (type == BISHOP || type == PROM_BISHOP || type == PROM_ROOK) attackers.set(ny * 9 + nx);
                    }
                    break;
                }
            }
        }

        // --- 2. 近接駒（歩・桂）のピンポイントチェック ---
        if (IS_VALID_XY(kx, ky - fw)) {
            int p = pieces[(ky - fw) * 9 + kx];
            if (p != 0 && (p > 16 ? WHITE : BLACK) == attacker_color && (p & 15) == PAWN)
                attackers.set((ky - fw) * 9 + kx);
        }
        if (IS_VALID_XY(kx - 1, ky - fw * 2)) {
            int p = pieces[(ky - fw * 2) * 9 + (kx - 1)];
            if (p != 0 && (p > 16 ? WHITE : BLACK) == attacker_color && (p & 15) == KNIGHT)
                attackers.set((ky - fw * 2) * 9 + (kx - 1));
        }
        if (IS_VALID_XY(kx + 1, ky - fw * 2)) {
            int p = pieces[(ky - fw * 2) * 9 + (kx + 1)];
            if (p != 0 && (p > 16 ? WHITE : BLACK) == attacker_color && (p & 15) == KNIGHT)
                attackers.set((ky - fw * 2) * 9 + (kx + 1));
        }

        // --- 3. その他の近接駒（銀・金・玉・成駒）の周辺8マスチェック ---
        int dx8[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };
        int dy8[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };
        for (int d = 0; d < 8; ++d) {
            int nx = kx + dx8[d];
            int ny = ky + dy8[d];
            if (IS_VALID_XY(nx, ny)) {
                int p = pieces[ny * 9 + nx];
                if (p != 0 && (p > 16 ? WHITE : BLACK) == attacker_color) {
                    int type = p & 15;
                    if (type == KING) { attackers.set(ny * 9 + nx); continue; }

                    bool is_forward = (dy8[d] == -fw);
                    bool is_side = (dy8[d] == 0);
                    bool is_backward = (dy8[d] == fw);
                    bool is_straight = (dx8[d] == 0);
                    bool is_diag = (dx8[d] != 0 && dy8[d] != 0);

                    if (type == SILVER && (is_forward || is_diag)) attackers.set(ny * 9 + nx);
                    else if ((type == GOLD || type == PROM_PAWN || type == PROM_LANCE || type == PROM_KNIGHT || type == PROM_SILVER)
                        && (is_forward || is_side || (is_backward && is_straight))) attackers.set(ny * 9 + nx);
                    else if (type == PROM_ROOK && is_diag) attackers.set(ny * 9 + nx);
                    else if (type == PROM_BISHOP && (is_straight || is_side)) attackers.set(ny * 9 + nx);
                }
            }
        }

        return attackers; // 最後に、攻撃しているすべての敵駒の位置を返す
    }

    // ★追加：指定した手番の玉が王手されているか判定する
    bool in_check(int side) const {
        // 相手(1 - side)の駒によって、自分の玉が攻撃されているか
        return is_attacked(king_sq[side], 1 - side);
    }

    // ★追加：王手になる手（攻め方用）と、王手を回避する手（受け方用）だけを抽出する
    std::vector<std::string> get_tsume_moves(bool is_attacker) {
        std::vector<std::string> all_moves = get_legal_moves();
        std::vector<std::string> filtered_moves;

        for (const auto& m : all_moves) {
            Board temp = *this;
            temp.push_usi(m);

            // 自分が指した後に自分の玉が取られる手（非合法手・自殺手）は除外
            if (temp.in_check(turn)) continue;

            if (is_attacker) {
                // 攻め方：相手玉に王手をかけている手だけを残す
                if (temp.in_check(1 - turn)) {
                    filtered_moves.push_back(m);
                }
            }
            else {
                // 受け方：すべての合法手（自殺手以外）が逃げの手となる
                filtered_moves.push_back(m);
            }
        }
        return filtered_moves;
    }

    //std::string move_to_usi(const Move& m) const {
    //    if (m.is_drop) {
    //        char drop_chars[] = { '?', 'P', 'L', 'N', 'S', 'G', 'B', 'R' };
    //        std::string s = "";
    //        s += drop_chars[m.type];
    //        s += "*";
    //        s += sq_to_usi(m.to);
    //        return s;
    //    }
    //    else {
    //        std::string s = sq_to_usi(m.from) + sq_to_usi(m.to);
    //        if (m.is_promote) s += "+";
    //        return s;
    //    }
    //}

    // ★追加2：文字列を一切使わずに盤面を更新する（超高速！）
    void push_move(const Move& m, StateInfo& st);

    void undo_move(const Move& m, const StateInfo& st) {
        turn = 1 - turn; // 手番を戻す
        move_count--;

        // 1. 状態の直接復元
        last_to_sq = st.last_to_sq;
        zobrist_key = st.previous_zobrist_key;
        king_sq[turn] = st.previous_king_sq;
        bit_pos = st.previous_bit_pos; // ビットボードの逆演算をコピーで一撃解決

        // 2. 盤面と持ち駒の復元
        if (m.is_drop) {
            pieces[m.to] = 0; // 置いた駒を消す
            hand[turn][m.type]++; // 持ち駒を増やす
        }
        else {
            int val = pieces[m.to];
            if (m.is_promote) val -= 8; // 成りを戻す

            pieces[m.from] = val; // 駒を元の場所に戻す
            pieces[m.to] = st.captured_piece; // 取られていた相手の駒を復活させる

            if (st.captured_piece > 0) {
                int cap_type = st.captured_piece & 15;
                if (cap_type > 8) cap_type -= 8; // 成り駒は生駒に戻して計算
                hand[turn][cap_type]--; // 増えていた自分の持ち駒を減らす
            }
        }
    }
    // ★追加3：df-pn専用の高速王手生成
    // ★追加3：df-pn専用の高速王手生成（オーダリング完全版）
    std::vector<Move> get_tsume_moves_fast(bool is_attacker) {
        Move all_moves[MAX_LEGAL_MOVES];
        Move* end_moves = generate_moves(bit_pos, all_moves);
        std::vector<Move> filtered_moves;

        for (Move* m = all_moves; m != end_moves; ++m) {
            if (is_attacker) {
                StateInfo st;
                push_move(*m, st); // ★実体(*m)を渡す

                bool is_check = is_attacked(king_sq[turn], 1 - turn);
                undo_move(*m, st);

                if (is_check) {
                    filtered_moves.push_back(*m);
                }
            }
            else {
                filtered_moves.push_back(*m);
            }
        }

        // ==========================================
        // ★ オーダリング（Move Ordering）処理
        // ==========================================
        // 駒の価値（大駒を取る手ほどスコアが高くなる）
        auto get_piece_value = [](int type) {
            switch (type & 15) {
            case PAWN: return 10; case LANCE: return 20; case KNIGHT: return 30;
            case SILVER: return 40; case GOLD: case PROM_PAWN: case PROM_LANCE: case PROM_KNIGHT: case PROM_SILVER: return 50;
            case BISHOP: return 70; case PROM_BISHOP: return 80; case ROOK: return 80; case PROM_ROOK: return 90;
            case KING: return 1000; default: return 0;
            }
            };

        int opp_king_sq = king_sq[1 - turn];
        int opp_king_x = opp_king_sq % 9;
        int opp_king_y = opp_king_sq / 9;

        auto score_move = [&](const Move& m) {
            int score = 0;

            int to_x = m.to % 9;
            int to_y = m.to / 9;
            int dist_x = std::abs(to_x - opp_king_x);
            int dist_y = std::abs(to_y - opp_king_y);
            int max_dist = (std::max)(dist_x, dist_y); // チェビシェフ距離（マス目の遠さ）

            if (is_attacker) {
                // 攻め方：とにかく「玉の近く（急所）」を最優先する！
                if (max_dist == 1) score += 5000; // 王の隣(距離1)への着手は超絶ボーナス
                else if (max_dist == 2) score += 2000; // 距離2も強力

                score -= max_dist * 100; // 遠い手には厳しいペナルティ

                if (!m.is_drop) {
                    int captured = pieces[m.to];
                    if (captured != 0) {
                        score += get_piece_value(captured) * 10; // 駒取りのボーナスを劇的に下げる
                    }
                }
                if (m.is_promote) score += 500;
                if (m.is_drop) score += get_piece_value(m.type) * 10;
            }
            else {
                // 受け方：玉が逃げる手は後回しにし、合駒や駒取りを先に読む
                if (!m.is_drop) {
                    int captured = pieces[m.to];
                    if (captured != 0) {
                        score += get_piece_value(captured) * 10;
                    }
                    // ★重要：玉自身の移動（逃げ）は証明が長引くので後回し
                    if (pieces[m.from] != 0 && (pieces[m.from] & 15) == KING) {
                        score -= 2000;
                    }
                }
            }

            return score;
            };

        std::sort(filtered_moves.begin(), filtered_moves.end(), [&](const Move& a, const Move& b) {
            return score_move(a) > score_move(b);
            });
        // ==========================================

        return filtered_moves;
    }

    // 詳しい利き生成（特徴量用）
    void get_attacks(int sq, int type, int color, std::vector<int>& out_squares) const {
        // 1. スライドする利き（香・飛・角・龍・馬）
        int sliding_dirs[8] = { -9, -1, 1, 9, -10, -8, 8, 10 };
        bool is_rook_type = (type == ROOK || type == PROM_ROOK);
        bool is_bishop_type = (type == BISHOP || type == PROM_BISHOP);
        bool is_lance = (type == LANCE);

        for (int d = 0; d < 8; ++d) {
            if (d < 4 && !is_rook_type && !is_lance) continue;
            if (d >= 4 && !is_bishop_type) continue;
            if (is_lance && ((color == BLACK && sliding_dirs[d] != -9) || (color == WHITE && sliding_dirs[d] != 9))) continue;

            int curr = sq + sliding_dirs[d];
            while (curr >= 0 && curr < 81) {
                int prev_x = (curr - sliding_dirs[d]) % 9;
                int curr_x = curr % 9;
                if (std::abs(prev_x - curr_x) > 1) break;

                out_squares.push_back(curr);
                if (pieces[curr] != 0) break;
                if (is_lance) break; // 香車はスライド処理内で1回で止める（ループさせない）
                curr += sliding_dirs[d];
            }
        }
        if (type == PROM_ROOK || type == PROM_BISHOP) {
            int king_dirs[8] = { -9, -1, 1, 9, -10, -8, 8, 10 };
            for (int d = 0; d < 8; ++d) {
                // 龍なら斜め、馬なら上下左右の1マスをチェック
                bool is_extra_dir = (type == PROM_ROOK) ? (d >= 4) : (d < 4);
                if (is_extra_dir) {
                    int target = sq + king_dirs[d];
                    if (target >= 0 && target < 81) {
                        int sx = sq % 9; int tx = target % 9;
                        if (std::abs(sx - tx) <= 1) out_squares.push_back(target);
                    }
                }
            }
        }
    }

    bool can_move_to(int from, int to, int type, int color) const {
        int diff = to - from;
        int ax = std::abs(from % 9 - to % 9);
        int ay = std::abs(from / 9 - to / 9);

        if (type != KNIGHT && type != LANCE && type != ROOK && type != BISHOP && type != PROM_ROOK && type != PROM_BISHOP) {
            if (ax > 1 || ay > 1) return false;
            if (from == to) return false;
        }

        if (type == PAWN) {
            if (color == BLACK) { if (diff != DIR_UP) return false; }
            else { if (diff != DIR_DOWN) return false; }
        }
        else if (type == KNIGHT) {
            if (color == BLACK) { if (diff != DIR_UP_LEFT - 9 && diff != DIR_UP_RIGHT - 9) return false; }
            else { if (diff != DIR_DOWN_LEFT + 9 && diff != DIR_DOWN_RIGHT + 9) return false; }
        }
        else if (type == SILVER) {
            if (color == BLACK) { if (diff == DIR_LEFT || diff == DIR_RIGHT || diff == DIR_DOWN) return false; }
            else { if (diff == DIR_LEFT || diff == DIR_RIGHT || diff == DIR_UP) return false; }
        }
        else if (type == GOLD || type == PROM_PAWN || type == PROM_LANCE || type == PROM_KNIGHT || type == PROM_SILVER) {
            if (color == BLACK) { if (diff == DIR_DOWN_LEFT || diff == DIR_DOWN_RIGHT) return false; }
            else { if (diff == DIR_UP_LEFT || diff == DIR_UP_RIGHT) return false; }
        }
        else if (type == KING) {}
        else if (type == BISHOP || type == PROM_BISHOP) {
            if (ax != ay) { if (!(type == PROM_BISHOP && ax <= 1 && ay <= 1)) return false; }
        }
        else if (type == ROOK || type == PROM_ROOK) {
            if (ax != 0 && ay != 0) { if (!(type == PROM_ROOK && ax <= 1 && ay <= 1)) return false; }
        }
        else if (type == LANCE) {
            if (ax != 0) return false;
            if (color == BLACK) { if (diff >= 0) return false; }
            else { if (diff <= 0) return false; }
        }

        if (type == LANCE || type == BISHOP || type == ROOK || type == PROM_BISHOP || type == PROM_ROOK) {
            if ((type == PROM_BISHOP || type == PROM_ROOK) && ax <= 1 && ay <= 1) return true;
            int dir = 0;
            if (ax == 0) dir = (diff > 0) ? 9 : -9;
            else if (ay == 0) dir = (diff > 0) ? 1 : -1;
            else if (ax == ay) {
                int from_x = from % 9; int to_x = to % 9;
                if (to > from) dir = (to_x > from_x) ? 10 : 8;
                else dir = (to_x > from_x) ? -8 : -10;
            }
            if (dir == 0) return false;
            int curr = from + dir;
            while (curr != to) {
                if (pieces[curr] != 0) return false;
                curr += dir;
                if (curr < 0 || curr > 80) return false;
            }
        }
        return true;
    }

    std::vector<std::string> get_legal_moves() {
        // 1. ビットボード版の合法手生成を呼び出す（爆速！）
        Move bb_moves[MAX_LEGAL_MOVES];
        Move* end_moves = generate_moves(bit_pos, bb_moves);

        // 2. 出力された Move 構造体を、今まで通りの USI文字列 に変換する
        std::vector<std::string> usi_moves;
        char drop_chars[] = { '?', 'P', 'L', 'N', 'S', 'G', 'B', 'R' };

        // ★ポインタを使ったループに変更
        for (Move* m = bb_moves; m != end_moves; ++m) {
            if (m->is_drop) {
                std::string s = "";
                s += drop_chars[m->type];
                s += "*";
                s += sq_to_usi(m->to);
                usi_moves.push_back(s);
            }
            else {
                std::string s = sq_to_usi(m->from) + sq_to_usi(m->to);
                if (m->is_promote) s += "+";
                usi_moves.push_back(s);
            }
        }
        return usi_moves;
    }
    // ★追加: 現在の盤面からSFEN文字列（検索キー）を生成する
    std::string to_sfen_key() const {
        std::string sfen = "";

        // 1. 盤上の駒
        for (int y = 0; y < 9; ++y) {
            int empty_count = 0;
            for (int x = 0; x < 9; ++x) { // 9筋(x=0) -> 1筋(x=8)
                int sq = y * 9 + x;
                int val = pieces[sq];

                if (val == 0) {
                    empty_count++;
                }
                else {
                    if (empty_count > 0) {
                        sfen += std::to_string(empty_count);
                        empty_count = 0;
                    }
                    int type = val & 15;
                    bool is_promoted = (type > 8);
                    if (is_promoted) { sfen += "+"; type -= 8; }

                    char c = '?';
                    switch (type) {
                    case PAWN: c = 'P'; break; case LANCE: c = 'L'; break;
                    case KNIGHT: c = 'N'; break; case SILVER: c = 'S'; break;
                    case GOLD: c = 'G'; break; case BISHOP: c = 'B'; break;
                    case ROOK: c = 'R'; break; case KING: c = 'K'; break;
                    }

                    bool is_white = (val > 16);
                    if (is_white) c = std::tolower(c);
                    sfen += c;
                }
            }
            if (empty_count > 0) sfen += std::to_string(empty_count);
            if (y < 8) sfen += "/";
        }

        // 2. 手番
        sfen += (turn == BLACK) ? " b " : " w ";

        // 3. 持ち駒
        std::string hand_str = "";
        // 順番: 飛, 角, 金, 銀, 桂, 香, 歩
        int order[] = { ROOK, BISHOP, GOLD, SILVER, KNIGHT, LANCE, PAWN };
        char chars[] = { 'R', 'B', 'G', 'S', 'N', 'L', 'P' };

        bool has_hand = false;
        // 先手 (大文字)
        for (int i = 0; i < 7; ++i) {
            int cnt = hand[BLACK][order[i]];
            if (cnt > 0) {
                has_hand = true;
                if (cnt > 1) hand_str += std::to_string(cnt);
                hand_str += chars[i];
            }
        }
        // 後手 (小文字)
        for (int i = 0; i < 7; ++i) {
            int cnt = hand[WHITE][order[i]];
            if (cnt > 0) {
                has_hand = true;
                if (cnt > 1) hand_str += std::to_string(cnt);
                hand_str += (char)std::tolower(chars[i]);
            }
        }

        if (!has_hand) hand_str = "-";
        sfen += hand_str;

        // ※手数は含めない（定跡キー用）
        return sfen;
    }
    
};



// ★完全版 特徴量生成 (55チャンネルすべて埋める)
void make_features(const Board& b, std::vector<Ort::Float16_t>& features) {
    // コンストラクタを使って 0.0f で初期化
    std::fill(features.begin(), features.end(), Ort::Float16_t(0.0f));
    bool is_white = (b.turn == WHITE);

    // 1.0f を表す FP16 定数
    const Ort::Float16_t f16_one(1.0f);

    // 180度回転 (80 - sq)
    auto get_py_target = [&](int sq) {
        return is_white ? (80 - sq) : sq;
        };

    auto set_feat = [&](int ch, int sq) {
        features[ch * 81 + get_py_target(sq)] = f16_one;
        };

    // 駒番号の翻訳 (金=6, 角=4, 飛=5)
    auto get_py_offset = [](int type) {
        if (type == PROM_BISHOP) return 12; // 馬 (26ch)
        if (type == PROM_ROOK) return 13;   // 龍 (27ch)
        return type - 1;
        };

    // 1. 盤上の駒 (0-27)
    for (int sq = 0; sq < 81; ++sq) {
        int val = b.pieces[sq];
        if (val == 0) continue;
        int type = val & 15;
        int color = (val > 16) ? WHITE : BLACK;
        int ch_base = (color == b.turn) ? 0 : 14;
        set_feat(ch_base + get_py_offset(type), sq);
    }

    // 2. 持ち駒 (28-41)
    int piece_order[] = { PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK };
    for (int i = 0; i < 7; ++i) {
        int pt = piece_order[i];
        // floatで計算してからOrt::Float16_tへ変換
        Ort::Float16_t self_v16(b.hand[b.turn][pt] / 10.0f);
        Ort::Float16_t opp_v16(b.hand[1 - b.turn][pt] / 10.0f);

        for (int s = 0; s < 81; s++) {
            features[(28 + i) * 81 + s] = self_v16;
            features[(35 + i) * 81 + s] = opp_v16;
        }
    }

    // 3. 大駒の利き (42-45)
    for (int sq = 0; sq < 81; ++sq) {
        int val = b.pieces[sq];
        if (val == 0) continue;
        int type = val & 15;
        if (type != ROOK && type != BISHOP && type != PROM_ROOK && type != PROM_BISHOP) continue;

        int color = (val > 16) ? WHITE : BLACK;
        bool is_self = (color == b.turn);
        bool is_rook_type = (type == ROOK || type == PROM_ROOK);
        int ch = is_self ? (is_rook_type ? 42 : 43) : (is_rook_type ? 44 : 45);

        std::vector<int> attacks;
        b.get_attacks(sq, type, color, attacks);
        for (int t_sq : attacks) set_feat(ch, t_sq);
    }

    // 4. 玉の周辺 (46-49)
    int kings[2] = { b.king_sq[b.turn], b.king_sq[1 - b.turn] };
    int base_chs[2] = { 46, 48 };
    for (int k = 0; k < 2; ++k) {
        int k_sq = kings[k];
        if (k_sq < 0) continue;
        int ky = k_sq / 9; int kx = k_sq % 9;
        for (int r = 2; r <= 3; ++r) {
            int ch = base_chs[k] + (r - 2);
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    int ny = ky + dy; int nx = kx + dx;
                    if (ny >= 0 && ny < 9 && nx >= 0 && nx < 9) set_feat(ch, ny * 9 + nx);
                }
            }
        }
    }

    // 5. その他 (50-54)
    if (b.is_attacked(b.king_sq[b.turn], 1 - b.turn)) {
        for (int s = 0; s < 81; s++) features[50 * 81 + s] = f16_one;
    }

    Ort::Float16_t prog_v16((std::min)(b.move_count / 200.0f, 1.0f));
    //for (int s = 0; s < 81; s++) features[51 * 81 + s] = prog_v16;

    for (int i = 0; i < 9; ++i) {
        Ort::Float16_t v16(i / 8.0f);
        for (int j = 0; j < 9; ++j) {
            features[52 * 81 + get_py_target(j * 9 + i)] = v16;
            features[53 * 81 + get_py_target(i * 9 + j)] = v16;
        }
    }
    //if (b.last_to_sq != -1) set_feat(54, b.last_to_sq);
}

#include "zobrist.h"

inline void Board::push_move(const Move& m, StateInfo& st) {
    // ★追加：手を進める前に、失われる情報をすべて StateInfo に退避する
    st.captured_piece = m.is_drop ? 0 : pieces[m.to];
    st.last_to_sq = last_to_sq;
    st.previous_zobrist_key = zobrist_key; // （※ステップ1を実装済みの場合）
    st.previous_king_sq = king_sq[turn];
    st.previous_bit_pos = bit_pos;
    move_count++;

    // 1. 手番が入れ替わるので必ず XOR
    zobrist_key ^= ZOBRIST_TURN;

    if (m.is_drop) {
        // 2. 持ち駒が1つ減るハッシュ更新
        zobrist_key ^= ZOBRIST_HAND[turn][m.type][hand[turn][m.type]];
        hand[turn][m.type]--;
        zobrist_key ^= ZOBRIST_HAND[turn][m.type][hand[turn][m.type]];

        pieces[m.to] = m.type + (turn == WHITE ? 16 : 0);

        // 3. 盤面に駒が増えるハッシュ更新
        zobrist_key ^= ZOBRIST_BOARD[turn][m.type][m.to];

        last_to_sq = m.to;
    }
    else {
        int val = pieces[m.from];
        int type = val & 15;
        pieces[m.from] = 0;

        // 4. 移動元の駒が盤上から消えるハッシュ更新
        zobrist_key ^= ZOBRIST_BOARD[turn][type][m.from];

        int captured = pieces[m.to];
        if (captured > 0) {
            int cap_type = captured & 15;
            if (cap_type > 8) cap_type -= 8; // 成り駒は生の駒に戻す

            // 5. 取られた相手の駒が盤上から消えるハッシュ更新
            zobrist_key ^= ZOBRIST_BOARD[1 - turn][captured & 15][m.to];

            // 6. 自分の持ち駒が増えるハッシュ更新
            zobrist_key ^= ZOBRIST_HAND[turn][cap_type][hand[turn][cap_type]];
            hand[turn][cap_type]++;
            zobrist_key ^= ZOBRIST_HAND[turn][cap_type][hand[turn][cap_type]];
        }

        if (m.is_promote) { val += 8; type += 8; }
        pieces[m.to] = val;

        // 7. 移動先に駒が出現するハッシュ更新
        zobrist_key ^= ZOBRIST_BOARD[turn][type][m.to];

        if ((val & 15) == KING) king_sq[turn] = m.to;
        last_to_sq = m.to;
    }
    turn = 1 - turn;
    bit_pos.do_move(m); // ビットボードの更新も専用関数で爆速化
}

// ★ get_zobrist_key から calc_zobrist_key_full に名前を変更
inline uint64_t Board::calc_zobrist_key_full() const {
    uint64_t key = 0;

    // 1. 盤上の駒
    for (int sq = 0; sq < 81; ++sq) {
        int val = pieces[sq];
        if (val != 0) {
            int color = (val > 16) ? WHITE : BLACK;
            int type = val & 15;
            key ^= ZOBRIST_BOARD[color][type][sq];
        }
    }

    // 2. 持ち駒 (0枚でもXORするように変更し、差分更新と完全同期)
    int hand_types[] = { PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK };
    for (int c = 0; c < 2; ++c) {
        for (int type : hand_types) {
            int count = hand[c][type];
            key ^= ZOBRIST_HAND[c][type][count];
        }
    }

    // 3. 手番 (後手番のときだけXORする)
    if (turn == WHITE) {
        key ^= ZOBRIST_TURN;
    }

    return key;
}