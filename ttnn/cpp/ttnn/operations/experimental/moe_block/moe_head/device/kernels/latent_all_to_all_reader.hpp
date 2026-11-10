// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
// (K3's latent all-to-all) per destination block b of my direction, per core of my link holding rows of b: wait its
// hand-off on my latent word of it, NoC-read its rows [r0, r0 + nr) of its latent block [row_tiles_per_core][LWMAX] in
// RP-tile chunks (the batch rules of reduce_scatter_gather). args at a0: nu, (b | slot << 8 | chip << 16, mesh) x nu,
// nc, (x | y << 8 | k << 16, rows: 4 bits a block - r0 | nr << 2) x nc, LAT address, word address, LATZ address,
// counter address, LWMAX. Chunk m: its LAT at + m LATENT_CHUNK_BYTES bytes, the words' targets count up (seen, the
// caller's), re-armed after the last
FORCE_INLINE void latent_all_to_all_gather(uint32_t a0, uint32_t& pos, uint32_t m, uint32_t* seen) {
    const uint32_t nu = moe_block_args::arg(a0), u0 = a0 + 1, nc = moe_block_args::arg(u0 + 2 * nu);
    if (nu == 0) {
        return;
    }
    const uint32_t c0 = u0 + 2 * nu + 1, t0a = c0 + 2 * nc;
    const uint32_t lat = moe_block_args::arg(t0a) + m * LATENT_CHUNK_BYTES, word = moe_block_args::arg(t0a + 1),
                   LW = moe_block_args::arg(t0a + 4);
    if (m == 0) {
        for (uint32_t i = 0; i < nc; ++i) {
            seen[i] = 0;
        }
    }
    uint32_t c = 0, need = 0, wptr = 0;  // (pos: chunks pushed since the phases began - the CB's batch alignment)
    auto flush = [&]() {
        noc_async_read_barrier();
        cb_push_back(0, c * RP);
        pos += c;
        c = 0;
    };
    for (uint32_t un = 0; un < nu; ++un) {
        const uint32_t b = moe_block_args::arg(u0 + 2 * un) & 0xFF;
        for (uint32_t i = 0; i < nc; ++i) {
            const uint32_t cw = moe_block_args::arg(c0 + 2 * i),
                           rw = (moe_block_args::arg(c0 + 2 * i + 1) >> (4 * b)) & 0xF;
            const uint32_t r0 = rw & 3, nr = rw >> 2, cnt = nr * LW;
            if (cnt == 0) {
                continue;
            }
            volatile tt_l1_ptr uint32_t* mb =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word + X_GATHER_HANDOFF_BYTES * (cw >> 16));
            const uint32_t target = ++seen[i];
            invalidate_l1_cache();
            if (*mb < target && c > 0) {
                flush();
            }
            noc_semaphore_wait_min(mb, target);
            const uint64_t src =
                get_noc_addr(cw & 0xFF, (cw >> 8) & 0xFF, lat + r0 * LW * moe_block_tiles::kBf16TileBytes);
            for (uint32_t t0 = 0; t0 < cnt; t0 += RP) {
                if (c == 0) {
                    need = GROUP - pos % GROUP;
                    cb_reserve_back(0, need * RP);
                    wptr = get_write_ptr(0);
                }
                const uint32_t n = cnt - t0 < RP ? cnt - t0 : RP;
                noc_async_read(
                    src + t0 * moe_block_tiles::kBf16TileBytes,
                    wptr + c * RP * moe_block_tiles::kBf16TileBytes,
                    n * moe_block_tiles::kBf16TileBytes);
                if (++c == need) {
                    flush();
                }
            }
        }
        if (c > 0) {
            flush();
        }
    }
    if (m + 1 < x_gather_num_live_chunks) {
        return;
    }
    for (uint32_t i = 0; i < nc; ++i) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            word + X_GATHER_HANDOFF_BYTES * (moe_block_args::arg(c0 + 2 * i) >> 16)) = 0;
    }
}
