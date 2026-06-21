#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>

#ifdef _MSC_VER
#include <intrin.h>
#endif




// --- 定数定義 ---
constexpr int BLACK = 0;
constexpr int WHITE = 1;

constexpr int PAWN = 1, LANCE = 2, KNIGHT = 3, SILVER = 4,
GOLD = 5, BISHOP = 6, ROOK = 7, KING = 8;
constexpr int PROM_PAWN = 9, PROM_LANCE = 10, PROM_KNIGHT = 11,
PROM_SILVER = 12, PROM_BISHOP = 14, PROM_ROOK = 15;

constexpr int MAX_LEGAL_MOVES = 2048;

// --- 1. ビットボード構造体 ---
struct Bitboard {
    uint64_t p0 = 0; // 0〜63マス
    uint32_t p1 = 0; // 64〜80マス

    Bitboard() = default;
    Bitboard(uint64_t _p0, uint32_t _p1) : p0(_p0), p1(_p1) {}

    inline void set(int sq) {
        if (sq < 64) p0 |= (1ULL << sq);
        else p1 |= (1U << (sq - 64));
    }
    inline bool is_not_empty() const {
        return p0 != 0 || p1 != 0;
    }
    inline void clear(int sq) {
        if (sq < 64) p0 &= ~(1ULL << sq);
        else p1 &= ~(1U << (sq - 64));
    }
    inline bool test(int sq) const {
        if (sq < 64) return (p0 >> sq) & 1ULL;
        return (p1 >> (sq - 64)) & 1U;
    }

    inline Bitboard operator&(const Bitboard& b) const { return { p0 & b.p0, p1 & b.p1 }; }
    inline Bitboard operator|(const Bitboard& b) const { return { p0 | b.p0, p1 | b.p1 }; }
    inline Bitboard operator^(const Bitboard& b) const { return { p0 ^ b.p0, p1 ^ b.p1 }; }
    inline Bitboard operator~() const { return { ~p0, ~p1 & 0x1FFFFU }; }

    inline int pop_lsb() {
        if (p0) {
#ifdef _MSC_VER
            unsigned long sq;
            _BitScanForward64(&sq, p0); // Windows用の爆速命令
            p0 &= p0 - 1;
            return (int)sq;
#else
            int sq = __builtin_ctzll(p0); // Mac/Linux用の爆速命令
            p0 &= p0 - 1;
            return sq;
#endif
        }
        if (p1) {
#ifdef _MSC_VER
            unsigned long sq;
            _BitScanForward(&sq, p1);
            p1 &= p1 - 1;
            return (int)sq + 64;
#else
            int sq = __builtin_ctz(p1) + 64;
            p1 &= p1 - 1;
            return sq;
#endif
        }
        return -1;
    }
    explicit operator bool() const { return p0 || p1; }
};

// --- 2. 事前計算テーブル (C++17 inline変数) ---
inline Bitboard PAWN_ATTACKS[2][81];
inline Bitboard KNIGHT_ATTACKS[2][81];
inline Bitboard SILVER_ATTACKS[2][81];
inline Bitboard GOLD_ATTACKS[2][81];
inline Bitboard KING_ATTACKS[81];
inline Bitboard FILE_MASKS[9];
inline Bitboard RANK_MASKS[2][9]; // [color][rank] (行き所のない駒判定用)
inline Bitboard LINE_BB[81][81];    // ★追加：2つのマスを通る無限の直線
inline Bitboard BETWEEN_BB[81][81];

// 起動時に1回だけ呼ぶ初期化関数
inline void init_bitboards() {
    for (int sq1 = 0; sq1 < 81; ++sq1) {
        for (int sq2 = 0; sq2 < 81; ++sq2) {
            LINE_BB[sq1][sq2] = Bitboard();
            BETWEEN_BB[sq1][sq2] = Bitboard();
            if (sq1 == sq2) continue;

            int x1 = sq1 % 9, y1 = sq1 / 9;
            int x2 = sq2 % 9, y2 = sq2 / 9;
            int dx = x2 - x1, dy = y2 - y1;

            if (dx == 0 || dy == 0 || std::abs(dx) == std::abs(dy)) {
                int step_x = (dx == 0) ? 0 : (dx > 0 ? 1 : -1);
                int step_y = (dy == 0) ? 0 : (dy > 0 ? 1 : -1);

                int cx = x1, cy = y1;
                while (cx >= 0 && cx < 9 && cy >= 0 && cy < 9) {
                    LINE_BB[sq1][sq2].set(cy * 9 + cx);
                    cx += step_x; cy += step_y;
                }
                cx = x1; cy = y1;
                while (cx >= 0 && cx < 9 && cy >= 0 && cy < 9) {
                    LINE_BB[sq1][sq2].set(cy * 9 + cx);
                    cx -= step_x; cy -= step_y;
                }

                cx = x1 + step_x; cy = y1 + step_y;
                while (cx != x2 || cy != y2) {
                    BETWEEN_BB[sq1][sq2].set(cy * 9 + cx);
                    cx += step_x; cy += step_y;
                }
            }
        }
    }
    for (int sq = 0; sq < 81; ++sq) {
        int x = sq % 9;
        int y = sq / 9;

        // 筋マスクの設定
        FILE_MASKS[x].set(sq);

        // 歩の利き
        if (y > 0) PAWN_ATTACKS[BLACK][sq].set(sq - 9);
        if (y < 8) PAWN_ATTACKS[WHITE][sq].set(sq + 9);

        // 桂馬の利き (ワープバグもここで完全に防げる)
        if (y > 1) {
            if (x > 0) KNIGHT_ATTACKS[BLACK][sq].set(sq - 18 - 1); // 左上
            if (x < 8) KNIGHT_ATTACKS[BLACK][sq].set(sq - 18 + 1); // 右上
        }
        if (y < 7) {
            if (x > 0) KNIGHT_ATTACKS[WHITE][sq].set(sq + 18 - 1); // 左下
            if (x < 8) KNIGHT_ATTACKS[WHITE][sq].set(sq + 18 + 1); // 右下
        }

        auto add_attack = [&](Bitboard& bb, int dx, int dy) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < 9 && ny >= 0 && ny < 9) {
                bb.set(ny * 9 + nx);
            }
            };

        // 【銀の利き】(前3方向、斜め後ろ2方向)
        add_attack(SILVER_ATTACKS[BLACK][sq], -1, -1); add_attack(SILVER_ATTACKS[BLACK][sq], 0, -1); add_attack(SILVER_ATTACKS[BLACK][sq], 1, -1);
        add_attack(SILVER_ATTACKS[BLACK][sq], -1, 1); add_attack(SILVER_ATTACKS[BLACK][sq], 1, 1);

        add_attack(SILVER_ATTACKS[WHITE][sq], -1, 1); add_attack(SILVER_ATTACKS[WHITE][sq], 0, 1); add_attack(SILVER_ATTACKS[WHITE][sq], 1, 1);
        add_attack(SILVER_ATTACKS[WHITE][sq], -1, -1); add_attack(SILVER_ATTACKS[WHITE][sq], 1, -1);

        // 【金の利き】(前3方向、横2方向、真後ろ1方向)
        add_attack(GOLD_ATTACKS[BLACK][sq], -1, -1); add_attack(GOLD_ATTACKS[BLACK][sq], 0, -1); add_attack(GOLD_ATTACKS[BLACK][sq], 1, -1);
        add_attack(GOLD_ATTACKS[BLACK][sq], -1, 0); add_attack(GOLD_ATTACKS[BLACK][sq], 1, 0);
        add_attack(GOLD_ATTACKS[BLACK][sq], 0, 1);

        add_attack(GOLD_ATTACKS[WHITE][sq], -1, 1); add_attack(GOLD_ATTACKS[WHITE][sq], 0, 1); add_attack(GOLD_ATTACKS[WHITE][sq], 1, 1);
        add_attack(GOLD_ATTACKS[WHITE][sq], -1, 0); add_attack(GOLD_ATTACKS[WHITE][sq], 1, 0);
        add_attack(GOLD_ATTACKS[WHITE][sq], 0, -1);

        // 【玉の利き】(全8方向)
        int king_dx[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
        int king_dy[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
        for (int i = 0; i < 8; ++i) {
            add_attack(KING_ATTACKS[sq], king_dx[i], king_dy[i]);
        }
    }

    // 行き所のないマス（1段目や2段目）のマスク作成
    for (int x = 0; x < 9; ++x) {
        RANK_MASKS[BLACK][0].set(x);         // 先手の1段目
        RANK_MASKS[BLACK][1].set(x + 9);     // 先手の2段目
        RANK_MASKS[WHITE][8].set(x + 72);    // 後手の9段目
        RANK_MASKS[WHITE][7].set(x + 63);    // 後手の8段目
    }
}

// --- 3. 手の構造体 ---
struct Move {
    int from, to, type;
    bool is_promote, is_drop;
};

// --- 4. 盤面クラス (変換処理つき) ---
class Position {
public:
    Bitboard color_bb[2];
    Bitboard type_bb[16];
    int hand[2][8] = { 0 };
    int turn = BLACK;
    int king_sq[2] = { -1, -1 };

    Position() { turn = BLACK; }

    // 【重要】既存の配列からビットボードへ変換する関数！
    void set_from_array(const int pieces[81], const int current_turn, const int hand_pieces[2][8]) {
        for (int i = 0; i < 2; ++i) color_bb[i] = Bitboard();
        for (int i = 0; i < 16; ++i) type_bb[i] = Bitboard();

        turn = current_turn;
        for (int c = 0; c < 2; ++c) {
            for (int p = 1; p <= 7; ++p) hand[c][p] = hand_pieces[c][p];
        }

        for (int sq = 0; sq < 81; ++sq) {
            int val = pieces[sq];
            if (val == 0) continue;

            int color = (val > 16) ? WHITE : BLACK;
            int type = val & 15;

            color_bb[color].set(sq);
            type_bb[type].set(sq);
            if (type == KING) king_sq[color] = sq;
        }
    }

    Bitboard occupied() const { return color_bb[BLACK] | color_bb[WHITE]; }
    Bitboard targets(int color) const { return ~color_bb[color]; }

    struct Dir { int dx, dy; };
    Bitboard get_slider_attacks(int sq, const std::vector<Dir>& dirs) const {
        Bitboard attacks;
        Bitboard occ = occupied();
        int kx = sq % 9;
        int ky = sq / 9;

        for (const auto& d : dirs) {
            int x = kx;
            int y = ky;
            while (true) {
                x += d.dx;
                y += d.dy;
                if (x < 0 || x > 8 || y < 0 || y > 8) break; // 盤外に出たらストップ

                int target = y * 9 + x;
                attacks.set(target); // 利きをセット
                if (occ.test(target)) break; // 何かの駒にぶつかったらストップ（その駒を取る手は合法なので含める）
            }
        }
        return attacks;
    }

    void do_move(const Move& m) {
        int us = turn;
        int them = 1 - us;

        if (m.is_drop) {
            // 持ち駒を打つ
            color_bb[us].set(m.to);
            type_bb[m.type].set(m.to);
            hand[us][m.type]--;
        }
        else {
            // 盤上の駒を動かす
            color_bb[us].clear(m.from);
            type_bb[m.type].clear(m.from);

            // 相手の駒を取る処理
            if (color_bb[them].test(m.to)) {
                int cap_type = 0;
                for (int t = 1; t <= 15; ++t) {
                    if (type_bb[t].test(m.to)) { cap_type = t; break; }
                }
                color_bb[them].clear(m.to);
                type_bb[cap_type].clear(m.to);
                int base_type = (cap_type > 8) ? cap_type - 8 : cap_type; // 成駒は元に戻す
                hand[us][base_type]++;
            }

            // 移動先に駒を置く
            int new_type = m.is_promote ? m.type + 8 : m.type;
            color_bb[us].set(m.to);
            type_bb[new_type].set(m.to);

            // 玉が動いた場合は王様の位置を更新
            if (m.type == KING) king_sq[us] = m.to;
        }
        turn = them; // 手番を交代
    }

    bool is_attacked(int sq, int by_color) const {
        Bitboard attackers = color_bb[by_color];

        // 1. 歩・桂・銀・金の利きチェック（相手の色の利きテーブルを逆向きに使う）
        if (PAWN_ATTACKS[1 - by_color][sq] & type_bb[PAWN] & attackers) return true;
        if (KNIGHT_ATTACKS[1 - by_color][sq] & type_bb[KNIGHT] & attackers) return true;
        if (SILVER_ATTACKS[1 - by_color][sq] & type_bb[SILVER] & attackers) return true;

        Bitboard golds = type_bb[GOLD] | type_bb[PROM_PAWN] | type_bb[PROM_LANCE] | type_bb[PROM_KNIGHT] | type_bb[PROM_SILVER];
        if (GOLD_ATTACKS[1 - by_color][sq] & golds & attackers) return true;

        // 2. 玉の利きチェック
        if (KING_ATTACKS[sq] & type_bb[KING] & attackers) return true;

        // 3. 馬と龍の1マス斜め・上下左右の利きチェック
        Bitboard prom_sliders = type_bb[PROM_BISHOP] | type_bb[PROM_ROOK];
        if (KING_ATTACKS[sq] & prom_sliders & attackers) return true;

        // 4. 香車の利き（相手が先手なら上から降ってくるので、下(0, 1)に向かって光線を飛ばす）
        std::vector<Dir> lance_dirs = (by_color == BLACK) ? std::vector<Dir>{{0, 1}} : std::vector<Dir>{ {0, -1} };
        if (get_slider_attacks(sq, lance_dirs) & type_bb[LANCE] & attackers) return true;

        // 5. 角・馬の斜めスライド利き
        std::vector<Dir> bishop_dirs = { {-1, -1}, {1, -1}, {-1, 1}, {1, 1} };
        if (get_slider_attacks(sq, bishop_dirs) & (type_bb[BISHOP] | type_bb[PROM_BISHOP]) & attackers) return true;

        // 6. 飛・龍の上下左右スライド利き
        std::vector<Dir> rook_dirs = { {0, -1}, {0, 1}, {-1, 0}, {1, 0} };
        if (get_slider_attacks(sq, rook_dirs) & (type_bb[ROOK] | type_bb[PROM_ROOK]) & attackers) return true;

        return false;
    }
    inline Bitboard get_attackers_to(int sq, int attacker_color) const {
        Bitboard attackers(0, 0);
        int us = 1 - attacker_color;
        int kx = sq % 9; int ky = sq / 9;

        // 1. 歩
        int p_from = sq + (attacker_color == BLACK ? 9 : -9);
        if (p_from >= 0 && p_from < 81 && type_bb[PAWN].test(p_from) && color_bb[attacker_color].test(p_from)) attackers.set(p_from);

        // 2. 桂馬
        int fw = (attacker_color == BLACK) ? 1 : -1;
        if (kx - 1 >= 0 && ky + fw * 2 >= 0 && ky + fw * 2 < 9) {
            int n_from = (ky + fw * 2) * 9 + (kx - 1);
            if (type_bb[KNIGHT].test(n_from) && color_bb[attacker_color].test(n_from)) attackers.set(n_from);
        }
        if (kx + 1 < 9 && ky + fw * 2 >= 0 && ky + fw * 2 < 9) {
            int n_from = (ky + fw * 2) * 9 + (kx + 1);
            if (type_bb[KNIGHT].test(n_from) && color_bb[attacker_color].test(n_from)) attackers.set(n_from);
        }

        // 3. 銀・金・玉・成駒 (近接)
        int dx8[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };
        int dy8[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };
        for (int d = 0; d < 8; ++d) {
            int nx = kx + dx8[d]; int ny = ky + dy8[d];
            if (nx >= 0 && nx < 9 && ny >= 0 && ny < 9) {
                int from = ny * 9 + nx;
                if (color_bb[attacker_color].test(from)) {
                    bool is_forward = (dy8[d] == fw);
                    bool is_side = (dy8[d] == 0);
                    bool is_backward = (dy8[d] == -fw);
                    bool is_straight = (dx8[d] == 0);
                    bool is_diag = (dx8[d] != 0 && dy8[d] != 0);

                    if (type_bb[KING].test(from)) attackers.set(from);
                    else if (type_bb[SILVER].test(from) && (is_forward || is_diag)) attackers.set(from);
                    else if ((type_bb[GOLD].test(from) || type_bb[PROM_PAWN].test(from) || type_bb[PROM_LANCE].test(from) || type_bb[PROM_KNIGHT].test(from) || type_bb[PROM_SILVER].test(from))
                        && (is_forward || is_side || (is_backward && is_straight))) attackers.set(from);
                    else if (type_bb[PROM_ROOK].test(from) && is_diag) attackers.set(from);
                    else if (type_bb[PROM_BISHOP].test(from) && (is_straight || is_side)) attackers.set(from);
                }
            }
        }

        // 4. 香・飛・角 (スライダー)
        int dx4[4] = { 0, 0, -1, 1 };
        int dy4[4] = { -1, 1, 0, 0 };
        for (int d = 0; d < 4; ++d) {
            for (int step = 1; step < 9; ++step) {
                int nx = kx + dx4[d] * step; int ny = ky + dy4[d] * step;
                if (nx < 0 || nx >= 9 || ny < 0 || ny >= 9) break;
                int from = ny * 9 + nx;
                if (color_bb[us].test(from)) break;
                if (color_bb[attacker_color].test(from)) {
                    if (type_bb[ROOK].test(from) || type_bb[PROM_ROOK].test(from)) attackers.set(from);
                    if (type_bb[LANCE].test(from) && dx4[d] == 0 && dy4[d] == fw) attackers.set(from);
                    break;
                }
            }
        }
        int dxd[4] = { -1, 1, -1, 1 };
        int dyd[4] = { -1, -1, 1, 1 };
        for (int d = 0; d < 4; ++d) {
            for (int step = 1; step < 9; ++step) {
                int nx = kx + dxd[d] * step; int ny = ky + dyd[d] * step;
                if (nx < 0 || nx >= 9 || ny < 0 || ny >= 9) break;
                int from = ny * 9 + nx;
                if (color_bb[us].test(from)) break;
                if (color_bb[attacker_color].test(from)) {
                    if (type_bb[BISHOP].test(from) || type_bb[PROM_BISHOP].test(from)) attackers.set(from);
                    break;
                }
            }
        }
        return attackers;
    }
    Bitboard get_pinned(int us) const {
        Bitboard pinned;
        int ksq = king_sq[us];
        if (ksq == -1) return pinned;

        int them = 1 - us;
        Bitboard occ = occupied();

        // 相手の飛び駒（飛・角・香・龍・馬）
        Bitboard sliders = (type_bb[ROOK] | type_bb[PROM_ROOK] | type_bb[BISHOP] | type_bb[PROM_BISHOP] | type_bb[LANCE]) & color_bb[them];

        Bitboard temp_sliders = sliders;
        while (temp_sliders) {
            int s = temp_sliders.pop_lsb();

            // 玉との間に直線関係がなければ無視
            if (!LINE_BB[ksq][s]) continue;

            // ==========================================
            // ★究極のバグ修正：飛車が斜めにピンしたり、角が縦横にピンする「幻のピン」を防ぐ
            // ==========================================
            int kx = ksq % 9, ky = ksq / 9;
            int sx = s % 9, sy = s / 9;
            int dx = std::abs(kx - sx);
            int dy = std::abs(ky - sy);

            if (dx == dy) {
                // 斜めの線なら、角か馬でなければピンできない
                if (!type_bb[BISHOP].test(s) && !type_bb[PROM_BISHOP].test(s)) continue;
            }
            else {
                // 縦横の線なら、飛・龍・香でなければピンできない
                if (!type_bb[ROOK].test(s) && !type_bb[PROM_ROOK].test(s) && !type_bb[LANCE].test(s)) continue;

                // 香車の場合、前方にしか飛ばないので向きをチェック
                if (type_bb[LANCE].test(s)) {
                    if (them == BLACK && ky >= sy) continue; // 先手(0)の香は上(yが減る方向)にしか利かない
                    if (them == WHITE && ky <= sy) continue; // 後手(1)の香は下(yが増える方向)にしか利かない
                }
            }

            // 玉と飛び駒の「間」にある駒を取得
            Bitboard bet = BETWEEN_BB[ksq][s] & occ;

            // 間に駒が「ちょうど1つ」ある場合のみピンが発生
            if (bet) {
                int blocker = bet.pop_lsb();
                if (!bet) { // pop_lsb後にもう駒が残っていなければ（＝1つだけなら）
                    if (color_bb[us].test(blocker)) {
                        pinned.set(blocker); // それが自分の駒ならピン認定！
                    }
                }
            }
        }
        return pinned;
    }
};




inline Move* generate_moves_simple(const Position& pos, Move* move_list) {
    int us = pos.turn;
    int opp = 1 - us;
    Bitboard target_sq = pos.targets(us);
    Bitboard empty_sq = ~pos.occupied();

    // 【歩の移動】
    Bitboard pawns = pos.type_bb[PAWN] & pos.color_bb[us];
    while (pawns) {
        int from = pawns.pop_lsb();
        Bitboard attacks = PAWN_ATTACKS[us][from] & target_sq;
        while (attacks) {
            int to = attacks.pop_lsb();
            int to_y = to / 9; int from_y = from / 9;
            bool promote_zone = (us == BLACK) ? (to_y <= 2 || from_y <= 2) : (to_y >= 6 || from_y >= 6);
            bool must_promote = (us == BLACK) ? (to_y == 0) : (to_y == 8);

            if (promote_zone) *move_list++ = { from, to, PAWN, true, false };
            if (!must_promote) *move_list++ = { from, to, PAWN, false, false };
        }
    }

    // 【桂馬の移動】
    Bitboard knights = pos.type_bb[KNIGHT] & pos.color_bb[us];
    while (knights) {
        int from = knights.pop_lsb();
        Bitboard attacks = KNIGHT_ATTACKS[us][from] & target_sq;
        while (attacks) {
            int to = attacks.pop_lsb();

            int to_y = to / 9; int from_y = from / 9;
            bool promote_zone = (us == BLACK) ? (to_y <= 2 || from_y <= 2) : (to_y >= 6 || from_y >= 6);
            // 桂馬は1段目・2段目(後手なら8段目・9段目)に行った時は強制成り！
            bool must_promote = (us == BLACK) ? (to_y <= 1) : (to_y >= 7);

            if (promote_zone) *move_list++ = { from, to, KNIGHT, true, false };
            if (!must_promote) *move_list++ = { from, to, KNIGHT, false, false };
        }
    }

    // 【歩の打牌 (Drop)】
    if (pos.hand[us][PAWN] > 0) {
        Bitboard valid_drops = empty_sq;
        // 行き所のない1段目を消す
        valid_drops = valid_drops & ~RANK_MASKS[us][(us == BLACK) ? 0 : 8];

        // 二歩の排除
        Bitboard my_pawns = pos.type_bb[PAWN] & pos.color_bb[us];
        for (int x = 0; x < 9; ++x) {
            if (my_pawns & FILE_MASKS[x]) {
                valid_drops = valid_drops & ~FILE_MASKS[x];
            }
        }

        while (valid_drops) {
            int to = valid_drops.pop_lsb();
            *move_list++ = { -1, to, PAWN, false, true };
        }
    }

    // 【香・桂・銀・金・角・飛の打牌 (Drop)】
    for (int type = LANCE; type <= ROOK; ++type) {
        if (pos.hand[us][type] > 0) {
            Bitboard valid_drops = empty_sq;

            // 香車は行き所のない1段目(後手は9段目)には打てない
            if (type == LANCE) {
                valid_drops = valid_drops & ~RANK_MASKS[us][(us == BLACK) ? 0 : 8];
            }
            // 桂馬は行き所のない1・2段目(後手は8・9段目)には打てない
            else if (type == KNIGHT) {
                valid_drops = valid_drops & ~RANK_MASKS[us][(us == BLACK) ? 0 : 8];
                valid_drops = valid_drops & ~RANK_MASKS[us][(us == BLACK) ? 1 : 7];
            }
            // 銀・金・角・飛はどこに打ってもOK（そのまま）

            while (valid_drops) {
                int to = valid_drops.pop_lsb();
                *move_list++ = { -1, to, type, false, true };
            }
        }
    }

    // 【銀の移動】
    Bitboard silvers = pos.type_bb[SILVER] & pos.color_bb[us];
    while (silvers) {
        int from = silvers.pop_lsb();
        Bitboard attacks = SILVER_ATTACKS[us][from] & target_sq;
        while (attacks) {
            int to = attacks.pop_lsb();
            int to_y = to / 9; int from_y = from / 9;
            bool promote_zone = (us == BLACK) ? (to_y <= 2 || from_y <= 2) : (to_y >= 6 || from_y >= 6);

            *move_list++ = { from, to, SILVER, false, false };
            if (promote_zone) *move_list++ = { from, to, SILVER, true, false };
        }
    }

    // 【金、および金と同じ動きをする成駒（と金・成香・成桂・成銀）の移動】
    // ビット演算 OR (|) で該当する駒を全てかき集める！
    Bitboard golds = (pos.type_bb[GOLD] | pos.type_bb[PROM_PAWN] |
        pos.type_bb[PROM_LANCE] | pos.type_bb[PROM_KNIGHT] |
        pos.type_bb[PROM_SILVER]) & pos.color_bb[us];
    while (golds) {
        int from = golds.pop_lsb();
        Bitboard attacks = GOLD_ATTACKS[us][from] & target_sq;
        while (attacks) {
            int to = attacks.pop_lsb();

            // from のマスにある駒の「本当の種類」を調べて type にセット
            int real_type = GOLD;
            for (int t = 1; t <= 15; ++t) {
                if (pos.type_bb[t].test(from)) { real_type = t; break; }
            }
            *move_list++ = { from, to, real_type, false, false }; // 金や成駒はそれ以上成れない
        }
    }

    // 【玉の移動】
    Bitboard kings = pos.type_bb[KING] & pos.color_bb[us];
    while (kings) {
        int from = kings.pop_lsb();
        Bitboard attacks = KING_ATTACKS[from] & target_sq;
        while (attacks) {
            int to = attacks.pop_lsb();
            *move_list++ = { from, to, KING, false, false };
        }
    }
    Bitboard lances = pos.type_bb[LANCE] & pos.color_bb[us];
    std::vector<Position::Dir> lance_dirs = (us == BLACK) ? std::vector<Position::Dir>{{0, -1}} : std::vector<Position::Dir>{ {0, 1} };
    while (lances) {
        int from = lances.pop_lsb();
        Bitboard attacks = pos.get_slider_attacks(from, lance_dirs) & target_sq;
        while (attacks) {
            int to = attacks.pop_lsb();
            int to_y = to / 9; int from_y = from / 9;
            bool promote_zone = (us == BLACK) ? (to_y <= 2 || from_y <= 2) : (to_y >= 6 || from_y >= 6);
            bool must_promote = (us == BLACK) ? (to_y == 0) : (to_y == 8); // 1段目(9段目)は強制成り

            if (promote_zone) *move_list++ = { from, to, LANCE, true, false };
            if (!must_promote) *move_list++ = { from, to, LANCE, false, false };
        }
    }

    // 【角と馬の移動】
    Bitboard bishops = (pos.type_bb[BISHOP] | pos.type_bb[PROM_BISHOP]) & pos.color_bb[us];
    std::vector<Position::Dir> bishop_dirs = { {-1, -1}, {1, -1}, {-1, 1}, {1, 1} }; // 斜め4方向
    while (bishops) {
        int from = bishops.pop_lsb();
        Bitboard attacks = pos.get_slider_attacks(from, bishop_dirs);

        bool is_promoted = pos.type_bb[PROM_BISHOP].test(from);
        if (is_promoted) attacks = attacks | KING_ATTACKS[from]; // 馬なら玉の動きを追加！

        attacks = attacks & target_sq;

        while (attacks) {
            int to = attacks.pop_lsb();
            int to_y = to / 9; int from_y = from / 9;
            bool promote_zone = (us == BLACK) ? (to_y <= 2 || from_y <= 2) : (to_y >= 6 || from_y >= 6);

            *move_list++ = { from, to, is_promoted ? PROM_BISHOP : BISHOP, false, false };
            if (!is_promoted && promote_zone) {
                *move_list++ = { from, to, BISHOP, true, false };
            }
        }
    }

    // 【飛車と龍の移動】
    Bitboard rooks = (pos.type_bb[ROOK] | pos.type_bb[PROM_ROOK]) & pos.color_bb[us];
    std::vector<Position::Dir> rook_dirs = { {0, -1}, {0, 1}, {-1, 0}, {1, 0} }; // 上下左右
    while (rooks) {
        int from = rooks.pop_lsb();
        Bitboard attacks = pos.get_slider_attacks(from, rook_dirs);

        bool is_promoted = pos.type_bb[PROM_ROOK].test(from);
        if (is_promoted) attacks = attacks | KING_ATTACKS[from]; // 龍なら玉の動きを追加！

        attacks = attacks & target_sq;

        while (attacks) {
            int to = attacks.pop_lsb();
            int to_y = to / 9; int from_y = from / 9;
            bool promote_zone = (us == BLACK) ? (to_y <= 2 || from_y <= 2) : (to_y >= 6 || from_y >= 6);

            *move_list++ = { from, to, is_promoted ? PROM_ROOK : ROOK, false, false };
            if (!is_promoted && promote_zone) {
                *move_list++ = { from, to, ROOK, true, false };
            }
        }
    }

    
    return move_list;
}


inline bool has_legal_moves(const Position& pos) {
    Move temp_moves[MAX_LEGAL_MOVES];
    Move* temp_end = generate_moves_simple(pos, temp_moves);
    int us = pos.turn; int opp = 1 - us;

    Bitboard checkers = pos.get_attackers_to(pos.king_sq[us], opp);
    Bitboard pinned = pos.get_pinned(us);

    // ★ if (checkers) ではなく is_not_empty() を使う
    if (checkers.is_not_empty()) {
        Bitboard target_sq;
        int checker_sq = checkers.pop_lsb();
        if (checkers.is_not_empty()) target_sq = Bitboard(0, 0); // 両王手
        else { target_sq = BETWEEN_BB[pos.king_sq[us]][checker_sq]; target_sq.set(checker_sq); }

        for (Move* m = temp_moves; m != temp_end; ++m) {
            if (m->type == KING) {
                Position next_pos = pos; next_pos.do_move(*m);
                if (!next_pos.is_attacked(next_pos.king_sq[us], opp)) return true;
            }
            else if (target_sq.test(m->to)) {
                if (!pinned.test(m->from) || LINE_BB[pos.king_sq[us]][m->from].test(m->to)) return true;
            }
        }
        return false;
    }

    // 王手されていない場合
    for (Move* m = temp_moves; m != temp_end; ++m) {
        if (m->type == KING) {
            Position next_pos = pos; next_pos.do_move(*m);
            if (!next_pos.is_attacked(next_pos.king_sq[us], opp)) return true;
        }
        else {
            if (m->is_drop) return true;
            if (!pinned.test(m->from) || LINE_BB[pos.king_sq[us]][m->from].test(m->to)) return true;
        }
    }
    return false;
}

// ★ 修正版：本番の手生成（王手回避フィルター付き）
inline Move* generate_moves(const Position& pos, Move* move_list) {
    int us = pos.turn; int opp = 1 - us;
    Bitboard checkers = pos.get_attackers_to(pos.king_sq[us], opp);

    Move temp_moves[MAX_LEGAL_MOVES];
    Move* temp_end = generate_moves_simple(pos, temp_moves);
    Bitboard pinned = pos.get_pinned(us);

    // 1. 王手されている場合（爆速フィルター）
    if (checkers.is_not_empty()) {
        Bitboard target_sq;
        int checker_sq = checkers.pop_lsb();
        if (checkers.is_not_empty()) target_sq = Bitboard(0, 0);
        else { target_sq = BETWEEN_BB[pos.king_sq[us]][checker_sq]; target_sq.set(checker_sq); }

        for (Move* m = temp_moves; m != temp_end; ++m) {
            if (m->type == KING) {
                Position next_pos = pos; next_pos.do_move(*m);
                if (!next_pos.is_attacked(next_pos.king_sq[us], opp)) *move_list++ = *m;
            }
            else if (target_sq.test(m->to)) { // ターゲットマスのみを通す！
                if (!pinned.test(m->from) || LINE_BB[pos.king_sq[us]][m->from].test(m->to)) {
                    if (m->is_drop && m->type == PAWN) { // 打ち歩詰めチェック
                        Position next_pos = pos; next_pos.do_move(*m);
                        if (next_pos.is_attacked(next_pos.king_sq[opp], us)) {
                            if (!has_legal_moves(next_pos)) continue;
                        }
                    }
                    *move_list++ = *m;
                }
            }
        }
        return move_list;
    }

    // 2. 王手されていない場合（通常のフィルター）
    for (Move* m = temp_moves; m != temp_end; ++m) {
        if (m->type == KING) {
            Position next_pos = pos; next_pos.do_move(*m);
            if (next_pos.is_attacked(next_pos.king_sq[us], opp)) continue;
        }
        else {
            if (!m->is_drop && pinned.test(m->from)) {
                if (!LINE_BB[pos.king_sq[us]][m->from].test(m->to)) continue;
            }
        }
        if (m->is_drop && m->type == PAWN) {
            Position next_pos = pos; next_pos.do_move(*m);
            if (next_pos.is_attacked(next_pos.king_sq[opp], us)) {
                if (!has_legal_moves(next_pos)) continue;
            }
        }
        *move_list++ = *m;
    }
    return move_list;
}