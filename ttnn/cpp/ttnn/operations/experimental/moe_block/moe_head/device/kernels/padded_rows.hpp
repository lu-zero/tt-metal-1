// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A padded forward's real rows (router_common_args.hpp PADDED_CHUNK): PADDED_FORWARD_CHUNKS chunks packed in the
// forward, each SP row holding every chunk's B rows back to back, chunk k's real rows a prefix of its block (main's
// rotated layout's closed form). The metadata at l1: the forward's metadata words (PADDED_FORWARD_METADATA: at
// PADDED_FORWARD_CHUNKS 1 (slot, start, end), above {num_chunks, 0} + PADDED_FORWARD_CHUNKS (slot, start, end)
// records), else one chunk's actual_start at l1 and actual_end at l1 + 64. SKIP_PADDING_CHUNKS's effective row chunks:
// a head's row chunks past the last with a real row are skipped (padded_live_chunks).
#pragma once

#include <cstdint>
#include "api/tile_constants.h"

#ifndef PADDED_FORWARD_CHUNKS
#define PADDED_FORWARD_CHUNKS 1
#endif
static_assert(
    PADDED_FORWARD_CHUNKS >= 1 && PADDED_FORWARD_CHUNKS <= 8,
    "a forward of 1 .. 8 chunks (the metadata's 104 B in a 128 B slot)");

// positions in [base, base + len) below end
FORCE_INLINE uint32_t padded_prefix_count(uint32_t end, uint32_t base, uint32_t len) {
    return end <= base ? 0 : (end - base < len ? end - base : len);
}
// SP row my's real rows of a chunk [start, end) of tpc rows an SP row (moe_padding_config's closed form)
FORCE_INLINE uint32_t padded_chunk_rows(uint32_t start, uint32_t end, uint32_t my, uint32_t sp, uint32_t tpc) {
    const uint32_t chunk = sp * tpc, slab = start / chunk, bchip = (start / tpc) % sp, boff = start % tpc;
    if (my != bchip) {  // (one run: rows before the boundary row a slab on)
        return padded_prefix_count(end, ((my < bchip ? slab + 1 : slab) * chunk) + my * tpc, tpc);
    }
    // (the boundary row starts mid-slab and wraps into the next: two runs)
    return padded_prefix_count(end, slab * chunk + my * tpc + boff, tpc - boff) +
           padded_prefix_count(end, (slab + 1) * chunk + my * tpc, boff);
}
// my SP row's real rows of each chunk of the forward (its block's real prefix), from the words at l1
struct PaddedRows {
    uint32_t B;                         // an SP row's rows a chunk
    uint32_t v[PADDED_FORWARD_CHUNKS];  // chunk k's real rows: rows k B .. k B + v[k] of the SP row
    // is row `row` of my SP row real (rows past every chunk's - the block's padding rows - are not)
    FORCE_INLINE bool real(uint32_t row) const {
        const uint32_t k = row / B;
        return k < PADDED_FORWARD_CHUNKS && row - k * B < v[k];
    }
    // the real rows in [f, f + n)
    FORCE_INLINE uint32_t count(uint32_t f, uint32_t n) const {
        uint32_t c = 0;
        for (uint32_t k = 0; k < PADDED_FORWARD_CHUNKS; ++k) {
            const uint32_t a = k * B, b = a + v[k];
            const uint32_t lo = a > f ? a : f, hi = b < f + n ? b : f + n;
            c += hi > lo ? hi - lo : 0;
        }
        return c;
    }
};
FORCE_INLINE PaddedRows padded_rows_of(uint32_t l1, uint32_t my, uint32_t sp, uint32_t row_tokens) {
    PaddedRows r;
    r.B = row_tokens / PADDED_FORWARD_CHUNKS;
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
#ifdef PADDED_FORWARD_METADATA
    const uint32_t n = PADDED_FORWARD_CHUNKS == 1 ? 1u : w[0];  // (the chunks in use: the rest are empty)
    for (uint32_t k = 0; k < PADDED_FORWARD_CHUNKS; ++k) {
        const uint32_t o = PADDED_FORWARD_CHUNKS == 1 ? 1u : 3u + 3u * k;
        r.v[k] = k < n ? padded_chunk_rows(w[o], w[o + 1], my, sp, r.B) : 0;
    }
#else
    r.v[0] = padded_chunk_rows(w[0], w[16], my, sp, r.B);
#endif
    return r;
}

#ifndef COMPILE_FOR_TRISC
// The live row chunks of a head call (SKIP_PADDING_CHUNKS): args at a - [0] actual_start's / the metadata's address,
// [1] actual_end's (0 with PADDED_FORWARD_METADATA), [2] my SP row, [3] SP rows, [4] tokens an SP row, [5] the call's
// first row in my SP row, [6] chunks row_chunks, [7] row tiles a chunk CR, [8] GLOBAL_ROW_TILE's own CR (0:
// chunk-major), [9] its own row tiles. 1 + the last chunk with a real row tile (at least 1). The scalars / words:
// one-page DRAM tensors (page 0 in bank 0), read into l1 (128 B, 64 B aligned).
template <typename Arg>
FORCE_INLINE PaddedRows padded_read_rows(Arg arg, uint32_t a, uint32_t l1) {
#ifdef PADDED_FORWARD_METADATA
    constexpr uint32_t bytes = PADDED_FORWARD_CHUNKS == 1 ? 16 : (4 * (2 + 3 * PADDED_FORWARD_CHUNKS) + 15) / 16 * 16;
    noc_async_read(get_noc_addr_from_bank_id<true>(0, arg(a)), l1, bytes);
#else
    noc_async_read(get_noc_addr_from_bank_id<true>(0, arg(a)), l1, 16);
    noc_async_read(get_noc_addr_from_bank_id<true>(0, arg(a + 1)), l1 + 64, 16);
#endif
    noc_async_read_barrier();
    invalidate_l1_cache();
    return padded_rows_of(l1, arg(a + 2), arg(a + 3), arg(a + 4));
}
// GLOBAL_ROW_TILE (shared_expert_macros.hpp) from the args: chunk m's chunk-local row tile lr's row tile of the call
FORCE_INLINE uint32_t padded_global_row_tile(uint32_t m, uint32_t lr, uint32_t cr, uint32_t ocr, uint32_t ort) {
    return ocr ? (lr / ocr) * ort + m * ocr + lr % ocr : m * cr + lr;
}
// is row tile t of the call (rows row0 + TILE_HEIGHT t ..) real anywhere
FORCE_INLINE bool padded_tile_real(const PaddedRows& rr, uint32_t row0, uint32_t t) {
    return rr.count(row0 + tt::constants::TILE_HEIGHT * t, tt::constants::TILE_HEIGHT) != 0;
}
template <typename Arg>
FORCE_INLINE uint32_t padded_live_of(const PaddedRows& rr, Arg arg, uint32_t a) {
    const uint32_t row0 = arg(a + 5), nch = arg(a + 6), cr = arg(a + 7), ocr = arg(a + 8), ort = arg(a + 9);
    for (uint32_t m = nch; m > 1; --m) {
        for (uint32_t lr = 0; lr < cr; ++lr) {
            if (padded_tile_real(rr, row0, padded_global_row_tile(m - 1, lr, cr, ocr, ort))) {
                return m;
            }
        }
    }
    return 1;
}
template <typename Arg>
FORCE_INLINE uint32_t padded_live_chunks(Arg arg, uint32_t a, uint32_t l1) {
    return padded_live_of(padded_read_rows(arg, a, l1), arg, a);
}
#endif
