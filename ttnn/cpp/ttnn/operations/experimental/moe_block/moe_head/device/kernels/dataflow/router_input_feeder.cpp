// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// router_input_feeder: feeds the router compute its inputs and builds a share of the joint routing rows
// Runs on: router cores, RISCV_1 (NCRISC), NoC 0
// Responsibilities:
//  - reads the router bias and expert index tiles from DRAM into CBs 5 / 11 (once a call)
//  - waits for the matmul cores' partial logits in HOLD (engine-done semaphore), then the RDY page (live row tiles,
//    chip index, TP position) to the compute
//  - hands the other chips' logits partials (landed over the fabric) to the compute (ARRB) in a fixed order
//  - builds tokens 8 .. 15 of the joint routing rows in L1 once the top-K is done
//  - under ROUTING_PLANNER: waits for its rows word, then routing_planner::plan if the host gave it an instance
//    (planner_slot currently places every instance on a BRISC)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/token_loop.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/router_common_args.hpp"
#ifdef ROUTING_PLANNER
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_planner_select.hpp"
#endif

#include <cstdint>
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/router_routing_rows.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
constexpr uint32_t expert_tiles = get_compile_time_arg_val(1), G = get_compile_time_arg_val(2);
constexpr auto b_args = TensorAccessorArgs<4>();
constexpr auto i_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
constexpr uint32_t ARRB = 4, BIAS = 5, IDX = 11, K = get_compile_time_arg_val(3), RDY = 12;
// args: bias^T, expert index tiles; my rows [G]; the arrivals (na: every other chip's partial of my owner
// rows, slot = its source index) and their words (a word a slot, 16 B apart); the slots in their fixed sum order [G -
// 1]
// One token loop of this kernel's work: kernel_main below runs it once per loop (token_loop.hpp).
void process_token_loop(uint32_t loop) {
    const uint32_t b_addr = moe_block_args::arg(0), i_addr = moe_block_args::arg(1);
    const uint32_t r0 = 2, na = moe_block_args::arg(r0 + G);
    volatile tt_l1_ptr uint32_t* ctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(moe_block_args::arg(r0 + G + 1));
    const uint32_t ab0 = r0 + G, nab = G - 1;  // (the order: ab0 + 2 ..; the tail: ab0 + nab + 2 ..)
    {  // (the top-K's done words: zero before anyone can write them - they are written after the scores; with the
       // split word 128 too, the partner's best K landed - the arena is call scratch: stale, it let the merge take a
       // stale page, the picks a call behind)
        const uint32_t w0 = moe_block_args::arg(ab0 + nab + 2);
        // (and 144: the BRISC's live row tiles, written after my BIAS push - in loop 0 only: in a later token loop the
        // BIAS is already there, so the BRISC may write 144 before I get here; I clear it after its last read instead)
        const uint32_t nz = loop == 0 ? 10 : 9;
        for (uint32_t i = 0; i < nz; ++i) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w0 + 16 * i) = 0;
        }
    }
    if (loop == 0) {  // (bias and expert indices: the call's, once - the compute reads them every loop, never pops)
        const auto bg = TensorAccessor(b_args, b_addr, moe_block_tiles::kFp32TileBytes);
        const auto ig = TensorAccessor(i_args, i_addr, moe_block_tiles::kBf16TileBytes);
        cb_reserve_back(BIAS, expert_tiles);
        cb_reserve_back(IDX, expert_tiles);
        for (uint32_t t = 0; t < expert_tiles; ++t) {
            noc_async_read(
                bg.get_noc_addr(t),
                get_write_ptr(BIAS) + t * moe_block_tiles::kFp32TileBytes,
                moe_block_tiles::kFp32TileBytes);
        }
        for (uint32_t t = 0; t < expert_tiles; ++t) {
            noc_async_read(
                ig.get_noc_addr(t),
                get_write_ptr(IDX) + t * moe_block_tiles::kBf16TileBytes,
                moe_block_tiles::kBf16TileBytes);
        }
        noc_async_read_barrier();
        cb_push_back(BIAS, expert_tiles);
        cb_push_back(IDX, expert_tiles);
    }
#ifdef ROUTER_ITERATIONS_ON_ROUTER
    constexpr uint32_t router_iterations = ROUTER_ITERATIONS_ON_ROUTER;  // (router iterations: my row tile of each)
#else
    constexpr uint32_t router_iterations = 1;
#endif
    for (uint32_t it = 0; it < router_iterations; ++it) {
        {  // the engine computed my partial logits (every engine core its expert tiles x its rows, written into my HOLD
           // with an increment of this counter a tile): all G x expert_tiles in -> the compute (RDY)
            DeviceZoneScopedN("SE-RENGWAIT");
            volatile tt_l1_ptr uint32_t* ec =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ROUTER_ENGINE_DONE_SEM_ADDR);
            noc_semaphore_wait_min(ec, expert_tiles);  // (my own block's tiles: the others' go straight to the ports)
            noc_semaphore_inc(get_noc_addr(ROUTER_ENGINE_DONE_SEM_ADDR), 0u - expert_tiles);  // (re-arm)
            noc_async_atomic_barrier();
            cb_reserve_back(RDY, 1);
#ifdef PADDED_CHUNK
            {  // (the BRISC's live row tiles into the RDY page: the compute's three threads read the same word)
                volatile tt_l1_ptr uint32_t* lw =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(moe_block_args::arg(ab0 + nab + 2) + 144);
                uint32_t lv;
                while (!((lv = *lw) >> 31)) {
                    invalidate_l1_cache();
                }
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(RDY)) = lv;
            }
#endif
#ifdef ROUTER_MY_CHIP
            {  // (this chip's index and TP position for the compute: one program for the mesh)
                volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(RDY));
                w[1] = ROUTER_MY_CHIP;
                w[2] = ROUTER_TP_POSITION;
            }
#endif
            cb_push_back(RDY, 1);
            // (and word 112 for the compute's PACK thread: the RDY wait holds only UNPACK, and PACK's hand-off pushes
            // (FINC) would hand HOLD to the ports before the engine's partials landed - stale HOLD: the last call's
            // planner overlay in it, K3's picks 99.10%)
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(moe_block_args::arg(ab0 + nab + 2) + 112) = 1;
        }
        if (na) {  // the sources' partials as they land (a fused increment of the source slot's own word, 16 B apart):
                   // each slot to the compute (ARRB) in a fixed order - the order args, the predicted landing order -
                   // so the sum is the same every run at ~no wait (the head is always deterministic)
            DeviceZoneScopedN("SE-RLWAIT");
            for (uint32_t a = 0, seen = 0; a < na;) {
                invalidate_l1_cache();
                for (uint32_t s = 0; s < na; ++s) {
                    if (s != moe_block_args::arg(ab0 + 2 + a)) {  // (the fixed order)
                        continue;
                    }
                    if ((seen >> s & 1) || ctr[4 * s] == 0) {
                        continue;
                    }
                    ctr[4 * s] =
                        0;  // (re-arm: one increment a call - the source's next one comes after this chip's exchange)
                    seen |= 1u << s;
                    ++a;
                    cb_reserve_back(ARRB, 1);
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(ARRB)) = s;
                    cb_push_back(ARRB, 1);
                }
            }
        }
        {  // (my share of the joint rows - tokens 8 .. 15 - once the top-K is done)
            const uint32_t t2 = ab0 + nab + 2, words = moe_block_args::arg(t2);
            volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 64);
            while (*done == 0) {
                invalidate_l1_cache();
            }
            *done = 0;
            DeviceZoneScopedN("SE-RROWS");
#ifdef PADDED_CHUNK
            // (my iterations' live row tiles: the BRISC's, words + 144 - an all-padding tile has no rows)
            if ((*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 144) >> it) & 1)
#endif
            {
                set_l1_data_cache<true>();
                invalidate_l1_cache();
                rows_::make<K, expert_tiles>(
                    moe_block_args::arg(t2 + 1),
                    moe_block_args::arg(t2 + 4),
                    moe_block_args::arg(t2 + 2),
                    moe_block_args::arg(t2 + 3),
                    8,
                    16);
                set_l1_data_cache<false>();
            }
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 16) = 1;
        }
    }  // (router iterations)
#if defined(PADDED_CHUNK) && TOKEN_LOOPS > 1
    // (the live row tiles read: zero for the next token loop's BRISC write - see the top)
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(moe_block_args::arg(ab0 + nab + 2) + 144) = 0;
#endif
#ifdef ROUTING_PLANNER
    {  // the second planner instance on this core (the routing exchange + planner)
        const uint32_t rr = ab0 + nab + 2 + 5, rows2 = moe_block_args::arg(rr), B = moe_block_args::arg(rr + 1);
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows2), ROUTING_EXCHANGE_ROUTER_ROW_TILES);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows2) = 0;
        if (B) {
            ROUTING_PLANNER_RUN(rr + 2, B);  // (no instance here when the chip has fewer experts than planner RISCs)
        }
    }
#endif
}

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    for (uint32_t loop = 0; loop < token_loop::LOOPS; ++loop) {
        token_loop::begin(loop);
        process_token_loop(loop);
        token_loop::end(loop);
    }
}
