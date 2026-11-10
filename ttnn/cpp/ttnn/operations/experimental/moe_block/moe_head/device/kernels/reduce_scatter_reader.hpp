// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
// (down + reduce-scatter) gather my units: per unit (entry e, block b, halves), per core of my link in the unit's
// halves holding tiles of b: wait its hand-off on my mailbox word of it, NoC-read its bf16 sub-block from its FINB
// (compact in the chip's entry order) in chunks of RP tiles; a batch ends at a multiple of GROUP chunks, at the end
// of a unit and before a blocking wait. At the end my mailbox words re-armed.
// args at r0: nu, (e | b << 8 | mask << 16, slot0 | slot1 << 16) x nu, nc, (x | y << 8 | half << 16 | k << 20,
// counts (8 bits a block)) x nc, the chip's entry order [G], FINB address, mailbox address, .., FINB-free address.
// Chunk m of row_chunks: the mailbox targets count up (seen, the caller's, over the chunks); after chunk m (not the
// last) +1 on the FINB-free word of every core read - its FINB is free for chunk m + 1; after the last the words
// re-armed
FORCE_INLINE void reduce_scatter_gather(uint32_t r0, uint32_t& pos, uint32_t m, uint32_t* seen) {
    const uint32_t nu = moe_block_args::arg(r0), u0 = r0 + 1, nc = moe_block_args::arg(u0 + 2 * nu);
    if (nu == 0) {
        return;
    }
    const uint32_t c0 = u0 + 2 * nu + 1, en0 = c0 + 2 * nc;
    const uint32_t finb = moe_block_args::arg(en0 + G), mbox = moe_block_args::arg(en0 + G + 1);
    if (m == 0) {
        for (uint32_t i = 0; i < nc; ++i) {
            seen[i] = 0;
        }
    }
    uint8_t rdc[128];
    for (uint32_t i = 0; i < nc; ++i) {
        rdc[i] = 0;
    }
    uint32_t c = 0, need = 0, wptr = 0;  // (pos: chunks pushed since the phases began - the CB's batch alignment)
    auto flush = [&]() {
        noc_async_read_barrier();
        cb_push_back(0, c * RP);
        pos += c;
        c = 0;
    };
    for (uint32_t un = 0; un < nu; ++un) {
        const uint32_t w0 = moe_block_args::arg(u0 + 2 * un), e = w0 & 0xFF, b = (w0 >> 8) & 0xFF,
                       mask = (w0 >> 16) & 3;
        // (pieces (w0 bit 28): the cores hand it pass by pass - walk pass-major over my cores, each core's pass piece
        // its nr x pw tiles at nr x (the earlier passes' widths) in its pass-major sub-block; else core by core whole)
        const uint32_t np_ = (w0 >> 28) & 1 ? REDUCE_SCATTER_PASSES : 1;
        for (uint32_t ip = 0; ip < np_; ++ip) {
            for (uint32_t i = 0; i < nc; ++i) {
                const uint32_t cw = moe_block_args::arg(c0 + 2 * i), cnts = moe_block_args::arg(c0 + 2 * i + 1);
                uint32_t cnt = cnt_of(cnts, b);
                if (!((mask >> ((cw >> 16) & 1)) & 1) || cnt == 0) {
                    continue;
                }
                uint32_t fo = 0;  // (the sub-block's FINB offset: the core's tiles of the entries before e)
                for (uint32_t e2 = 0; e2 < e; ++e2) {
                    fo += cnt_of(cnts, moe_block_args::arg(en0 + e2));
                }
                if (np_ > 1) {  // (pass ip's piece: the column's widths are split_even(pn, REDUCE_SCATTER_PASSES))
                    const uint32_t pn = cw >> 27, nr = cnt / pn, bw = pn / REDUCE_SCATTER_PASSES,
                                   rm = pn % REDUCE_SCATTER_PASSES;
                    fo += nr * (ip * bw + (ip < rm ? ip : rm));
                    cnt = nr * (bw + (ip < rm ? 1 : 0));
                }
                volatile tt_l1_ptr uint32_t* mb =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mbox + X_GATHER_HANDOFF_BYTES * ((cw >> 20) & 0x7F));
                const uint32_t target = ++seen[i];
                invalidate_l1_cache();
                if (*mb < target && c > 0) {
                    flush();  // (never hold reads while blocked on a producer)
                }
                noc_semaphore_wait_min(mb, target);
                rdc[i] = 1;
                const uint64_t src =
                    get_noc_addr(cw & 0xFF, (cw >> 8) & 0xFF, finb + fo * moe_block_tiles::kBf16TileBytes);
                const bool live = x_gather_block_real(m, (cw >> 20) & 0x7F, b);  // (padding rows only: not read)
                for (uint32_t t0 = 0; t0 < cnt; t0 += RP) {
                    if (c == 0) {
                        need = GROUP - pos % GROUP;
                        cb_reserve_back(0, need * RP);
                        wptr = get_write_ptr(0);
                    }
                    const uint32_t n = cnt - t0 < RP ? cnt - t0 : RP;
                    if (live) {
                        noc_async_read(
                            src + t0 * moe_block_tiles::kBf16TileBytes,
                            wptr + c * RP * moe_block_tiles::kBf16TileBytes,
                            n * moe_block_tiles::kBf16TileBytes);
                    }
                    if (++c == need) {
                        flush();
                    }
                }
            }
        }
        if (c > 0) {
            flush();
        }
    }
    if (m + 1 < x_gather_num_live_chunks) {  // (every read of the chunk landed - each unit ends in a flush: FINB free)
        const uint32_t ffa = moe_block_args::arg(en0 + G + 5);
        for (uint32_t i = 0; i < nc; ++i) {
            if (rdc[i]) {
                const uint32_t cw = moe_block_args::arg(c0 + 2 * i);
                noc_semaphore_inc(get_noc_addr(cw & 0xFF, (cw >> 8) & 0xFF, ffa), 1);
            }
        }
        noc_async_atomic_barrier();
        return;
    }
    for (uint32_t i = 0; i < nc; ++i) {  // (re-arm: every hand-off of the call was waited for)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            mbox + X_GATHER_HANDOFF_BYTES * ((moe_block_args::arg(c0 + 2 * i) >> 20) & 0x7F)) = 0;
    }
}
