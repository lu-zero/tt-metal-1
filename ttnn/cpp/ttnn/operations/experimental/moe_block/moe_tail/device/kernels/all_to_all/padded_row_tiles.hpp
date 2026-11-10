// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The all-to-all tail in a padded forward (the named compile-time arg skip_padding): which of my owned row tiles hold
// a real row (moe_head's padded_rows.hpp). The padding row tiles are not read, tilized or sent: each chip writes zeros
// to its own output's padding row tiles (the workers' writers, a share each), so the padding rows of the output are
// zero - finite, as the model's next ops need.
//
// The padding args at `a` (the kernel's args, moe_block_args.hpp): [0] actual_start's / the metadata's address, [1]
// actual_end's (0 with PADDED_FORWARD_METADATA), [2] my SP row, [3] SP rows, [4] tokens an SP row, [5] my owned rows'
// first row in my SP row (token loop 0), [6] the rows a token loop moves it. PADDED_FORWARD_CHUNKS /
// PADDED_FORWARD_METADATA stay defines: moe_head's padded_rows.hpp, shared by both ops, reads them.
#pragma once

#include <cstdint>

// (moe_tail_program_factory's all_to_all_cb::CB_PADDING_SCRATCH, 128 B pages: the reader's / port reader's metadata
// words at its base (on the workers the page it hands the compute), the writer's / port sender's 256 B on, the workers'
// zero tile 512 B on)
constexpr uint32_t CB_PADDING_SCRATCH = 6;
constexpr uint32_t PADDING_SCRATCH_WRITER_OFFSET = 256, PADDING_SCRATCH_ZERO_TILE_OFFSET = 512;
// (all_to_all_cb::CB_LIVE_ROW_TILES: the workers' reader -> compute hand-off of a token loop's live row tiles, its own
// one-page CB, so the page it advances to each loop never lands on the scratch above)
constexpr uint32_t CB_LIVE_ROW_TILES = 7;

// is row tile r live: with skip_padding bit r of the mask (r < 64, the host checks), else every row tile
template <bool skip_padding>
FORCE_INLINE bool row_tile_is_live(uint64_t live_mask, uint32_t r) {
    if constexpr (skip_padding) {
        return (live_mask >> r) & 1;
    } else {
        return true;
    }
}

#ifndef COMPILE_FOR_TRISC
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/padded_rows.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
FORCE_INLINE uint32_t padding_arg(uint32_t i) { return moe_block_args::arg(i); }
// my SP row's real rows: the metadata words read into l1 (128 B, 64 B aligned)
FORCE_INLINE PaddedRows read_padded_rows(uint32_t a, uint32_t l1) { return padded_read_rows(padding_arg, a, l1); }
// my owned rows' first row in my SP row, token loop `loop`'s
FORCE_INLINE uint32_t first_owned_row(uint32_t a, uint32_t loop) {
    return padding_arg(a + 5) + loop * padding_arg(a + 6);
}
// bit r: row tile r (of row_tiles <= 64) from first_row holds a real row
FORCE_INLINE uint64_t live_row_tile_mask(const PaddedRows& rows, uint32_t first_row, uint32_t row_tiles) {
    uint64_t mask = 0;
    for (uint32_t r = 0; r < row_tiles; ++r) {
        if (padded_tile_real(rows, first_row, r)) {
            mask |= uint64_t(1) << r;
        }
    }
    return mask;
}
FORCE_INLINE uint64_t read_live_row_tile_mask(uint32_t a, uint32_t l1, uint32_t row_tiles, uint32_t loop) {
    return live_row_tile_mask(read_padded_rows(a, l1), first_owned_row(a, loop), row_tiles);
}
#endif
