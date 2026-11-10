// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The x all-gather kernels' compile-time args.
// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once
#include <type_traits>
constexpr uint32_t k_step_tiles = get_compile_time_arg_val(0), row_tiles_per_core = get_compile_time_arg_val(1),
                   k_steps_per_shard = get_compile_time_arg_val(2);
constexpr uint32_t RP = get_compile_time_arg_val(3), CR = get_compile_time_arg_val(4),
                   row_chunks = get_compile_time_arg_val(5);
constexpr uint32_t NV = get_compile_time_arg_val(6), GROUP = get_compile_time_arg_val(7);
constexpr uint32_t INC = get_compile_time_arg_val(8), engine_columns = get_compile_time_arg_val(9),
                   engine_rows = get_compile_time_arg_val(10);
constexpr uint32_t S0 = get_compile_time_arg_val(11), S1 = get_compile_time_arg_val(12),
                   G = get_compile_time_arg_val(13);
// (a core's tiles of block b from its counts word: a byte a block, or past TP 4 two (block, count) pairs -
// ReduceScatterPlan::counts_word)
FORCE_INLINE uint32_t cnt_of(uint32_t cnts, uint32_t b) {
    if constexpr (G > 4) {
        return ((cnts & 0xFF) == b ? (cnts >> 8) & 0xFF : 0) + (((cnts >> 16) & 0xFF) == b ? cnts >> 24 : 0);
    } else {
        return (cnts >> (8 * b)) & 0xFF;
    }
}
// the call's live row chunks (SKIP_PADDING_CHUNKS, a padded forward: up to the last with a real row - the kernel sets
// it at its start; else every chunk)
inline uint32_t x_gather_num_live_chunks = row_chunks;
#ifdef SKIP_PADDING_CHUNKS
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/padded_rows.hpp"
static_assert(CR <= 64 && row_chunks <= 16, "a chunk's real row tiles: one 64-bit mask a chunk");
using x_gather_mask_t =
    std::conditional_t<(CR <= 32), uint32_t, uint64_t>;  // (a register on the 32-bit RISCs when it fits)
inline x_gather_mask_t x_gather_real_tiles[row_chunks];  // (chunk m's real chunk-local row tiles, bit lr)
FORCE_INLINE uint32_t x_gather_arg(uint32_t i) { return moe_block_args::common(i); }
FORCE_INLINE void x_gather_find_live_chunks() {
    const PaddedRows rr = padded_read_rows(x_gather_arg, 0, (x_gather_arg(10) + 63) & ~63u);
    const uint32_t row0 = x_gather_arg(5) + token_loop::index() * TOKEN_LOOP_PADDED_ROW_STRIDE, ocr = x_gather_arg(8),
                   ort = x_gather_arg(9);  // (loop l's rows)
    x_gather_num_live_chunks = 1;
    for (uint32_t m = 0; m < row_chunks; ++m) {
        x_gather_mask_t b = 0;
        for (uint32_t lr = 0; lr < CR; ++lr) {
            if (padded_tile_real(rr, row0, padded_global_row_tile(m, lr, CR, ocr, ort))) {
                b |= x_gather_mask_t(1) << lr;
            }
        }
        x_gather_real_tiles[m] = b;
        if (b) {
            x_gather_num_live_chunks = m + 1;
        }
    }
}
// is chunk m's chunk-local row tile lr real (a padding tile's x / reduce-scatter payload is not sent); x_gather_mask(m)
// once a chunk, the bit tested a packet
FORCE_INLINE bool x_gather_tile_real(uint32_t m, uint32_t lr) { return (x_gather_real_tiles[m] >> lr) & 1; }
FORCE_INLINE x_gather_mask_t x_gather_mask(uint32_t m) { return x_gather_real_tiles[m]; }
#else
constexpr bool x_gather_tile_real(uint32_t, uint32_t) { return true; }
using x_gather_mask_t = uint32_t;
constexpr x_gather_mask_t x_gather_mask(uint32_t) { return ~x_gather_mask_t(0); }
#endif
// (down + reduce-scatter) does core k's sub-block of block b in chunk m hold a real row (its rows: its chunk-local row
// tiles k / engine_columns row_tiles_per_core + r in block b's RB = CR / G)
FORCE_INLINE bool x_gather_block_real(uint32_t m, uint32_t k, uint32_t b) {
#ifndef SKIP_PADDING_CHUNKS
    return true;  // (every row real - and a latent's blocks are columns)
#endif
    const uint32_t cy = k / engine_columns;
    for (uint32_t r = 0; r < row_tiles_per_core; ++r) {
        const uint32_t lr = cy * row_tiles_per_core + r;
        if (lr / (CR / G) == b && x_gather_tile_real(m, lr)) {
            return true;
        }
    }
    return false;  // (computed once a sub-block, not a packet)
}
constexpr uint32_t RSX = 12;  // (the RS block's words past the entry order: FINB, mailbox, zone, counter, P, FINB-free,
                              // zone credit, zone free, the zone-free signals a chunk, the credits a call expected,
                              // the "router done" word, the matmul cores it counts (0: no router-done credit))
