#pragma once
#include <cstdint>
#include <random>
#include "shogi_utils.h"

inline uint64_t ZOBRIST_BOARD[2][16][81];
inline uint64_t ZOBRIST_HAND[2][8][19];
inline uint64_t ZOBRIST_TURN;
inline uint64_t ZOBRIST_LAST_TO[81];      // ★追加：相手の最終手用

inline void init_zobrist() {
    std::mt19937_64 rng(123456789ULL);

    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < 16; ++p) {
            for (int sq = 0; sq < 81; ++sq) {
                ZOBRIST_BOARD[c][p][sq] = rng();
            }
        }
    }
    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < 8; ++p) {
            for (int count = 0; count < 19; ++count) {
                ZOBRIST_HAND[c][p][count] = rng();
            }
        }
    }
    ZOBRIST_TURN = rng();

    for (int sq = 0; sq < 81; ++sq) {
        ZOBRIST_LAST_TO[sq] = rng(); // ★追加
    }
}
