// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// router_output_writer: writes the router's outputs to DRAM and runs the core's routing planner instance
// Runs on: router cores, RISCV_0 (BRISC), NoC 1
// Responsibilities:
//  - builds tokens 0 .. 7 of the joint routing rows, waits for the other shares, writes the owner page to the routing
//    rows in DRAM, then +1 on every planner core's rows words; writes the compute's route words to DRAM
//  - padded chunk: reads actual_start / actual_end from DRAM, publishes the live row tiles, zero routes for padding
//  - router iterations: +1 on the ports' and router core 0's iteration words; router core 0 tells the matmul cores go
//  - under ROUTING_PLANNER: once every router core's rows are in, routing_planner::plan (reads the exchange's DRAM
//    slots; writes counts / regions / token index / table / plan to DRAM)
// Note: fabric-written DRAM is read on NoC 1, the fabric routers' NoC: the other chips' routing packets land in the
//  DRAM slots through its endpoint (a read through the other NoC's DRAM endpoint can return stale data)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/token_loop.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/router_common_args.hpp"
#ifdef ROUTING_PLANNER
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_planner_select.hpp"
#endif

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
constexpr uint32_t expert_tiles = get_compile_time_arg_val(1), K = get_compile_time_arg_val(3);
constexpr auto o_args = TensorAccessorArgs<4>();
constexpr uint32_t BIAS = 5;
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/router_routing_rows.hpp"

// args: the out address, my owner page, route scale bits, the joint rows' L1 scratch, the done word, the BEST page
// One token loop of this kernel's work: kernel_main below runs it once per loop (token_loop.hpp).
void process_token_loop(uint32_t loop) {
#if defined(ROUTER_ROUTE_WORDS) && defined(PADDED_CHUNK)
    // (a padded chunk's actual_start / actual_end, for the route words: into the route words' L1 (256 B) - the compute
    // writes them only after word 80, by when they are read - early, read once the rows are written, long after)
    const uint32_t vl1 = padded_metadata_slot(moe_block_args::arg(6));
    padded_metadata_read(vl1);
    PaddedRows rr{};  // (my SP row's real rows, read below)
#endif
    constexpr uint32_t t0 = 0;
    // (token loops: loop l's owner rows TOKEN_LOOP_ROUTER_PAGE_STRIDE pages on, its route words
    // TOKEN_LOOP_ROUTE_WORDS_STRIDE bytes on - token_loop.hpp)
    const uint32_t o_addr = moe_block_args::arg(t0),
                   page = moe_block_args::arg(t0 + 1) + loop * TOKEN_LOOP_ROUTER_PAGE_STRIDE;
    const uint32_t o_l1 = moe_block_args::arg(t0 + 3);
    const uint32_t words = moe_block_args::arg(t0 + 4);
    // the top K of every token column, sorted descending in rows 0 .. K - 1 of the best page;
    // tokens 0 .. 7 here, 8 .. 15 the NCRISC, 16 .. 23 the compute's MATH thread, 24 .. 31 its PACK (done words 16 / 32
    // / 48)
    const uint32_t bv = moe_block_args::arg(t0 + 5);
    const auto og = TensorAccessor(o_args, o_addr, tt::constants::TILE_HEIGHT * sizeof(uint32_t) * K);
#ifdef ROUTER_ITERATIONS_ON_ROUTER
    // router iterations (ROUTER_ITERATION_ARGS args at t0 + 10): am I router core 0, my chip's "rit" word, router core
    // 0's NoC, the logits ports' count, the ports, the ewords, the matmul cores' NoC x's and y's
    constexpr uint32_t router_iterations = ROUTER_ITERATIONS_ON_ROUTER, ti = t0 + 10;
    const bool c0me = moe_block_args::arg(ti) != 0;
    const uint32_t rit = moe_block_args::arg(ti + 1), c0 = moe_block_args::arg(ti + 2);
    const uint32_t nact = moe_block_args::arg(ti + 3), npts = moe_block_args::arg(ti + 4), pt0 = ti + 5;
    const uint32_t ew = moe_block_args::arg(pt0 + npts), gx = moe_block_args::arg(pt0 + npts + 1);
    const uint32_t xs0 = pt0 + npts + 2, gy = moe_block_args::arg(xs0 + gx), ys0 = xs0 + gx + 1;
#else
    constexpr uint32_t router_iterations = 1;
#endif
#ifdef PADDED_CHUNK
    // a padded forward: my SP row's real rows now, and which of my iterations' row tiles hold a real token (bit it, bit
    // 31 valid) - an all-padding tile's top-K and joint rows are not computed (its route words are zero). Published at
    // words + 144 once the NCRISC zeroed the words (the arena is call scratch: its BIAS push comes after), for it and,
    // through each iteration's RDY page, the compute's three threads alike
    noc_async_read_barrier();
    invalidate_l1_cache();
    rr = padded_rows(vl1);
    uint32_t live = 0x80000000u;
    for (uint32_t it = 0; it < router_iterations; ++it) {
#ifdef ROUTER_ITERATIONS_ON_ROUTER
        const uint32_t tk =
            moe_block_args::arg(t0 + 8) + it * tt::constants::TILE_HEIGHT * ROUTER_ROW_TILES_PER_ITERATION;
#else
        const uint32_t tk = moe_block_args::arg(t0 + 8);
#endif
        if (rr.count(
                moe_block_args::common(7) + loop * TOKEN_LOOP_PADDED_ROW_STRIDE + tk, tt::constants::TILE_HEIGHT)) {
            live |= 1u << it;
        }
    }
    auto live_it = [&](uint32_t it) -> bool { return (live >> it) & 1; };
#else
    auto live_it = [](uint32_t) -> bool { return true; };
#endif
    auto route_out = [&](uint32_t it) {  // dispatch prep: my owned tokens' route words (the compute's, word 96)
#ifdef ROUTER_ROUTE_WORDS
        volatile tt_l1_ptr uint32_t* rd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 96);
        while (*rd == 0) {
            invalidate_l1_cache();
        }
        *rd = 0;  // (re-arm)
        const uint32_t rl1 = moe_block_args::arg(t0 + 6);
#ifdef ROUTER_ITERATIONS_ON_ROUTER
        const uint32_t tok0 =
            moe_block_args::arg(t0 + 8) +
            it * tt::constants::TILE_HEIGHT * ROUTER_ROW_TILES_PER_ITERATION;  // (my row tile of iteration it)
#else
        const uint32_t tok0 = moe_block_args::arg(t0 + 8);
#endif
#ifdef PADDED_CHUNK
        {  // (padding tokens - their rows not real - get no route: not dispatched)
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rl1);
            const uint32_t row0 = moe_block_args::common(7) + loop * TOKEN_LOOP_PADDED_ROW_STRIDE +
                                  tok0;  // (token c's row in my SP row: row0 + c)
            for (uint32_t c = 0; c < 32; ++c) {
                if (!rr.real(row0 + c)) {
                    for (uint32_t i = 0; i < ROUTER_ROUTE_WORDS_PER_TOKEN; ++i) {
                        o[ROUTER_ROUTE_WORDS_PER_TOKEN * c + i] = 0;
                    }
                }
            }
        }
#endif
        const auto rg = TensorAccessor(
            TensorAccessorArgs<ROUTER_ROUTE_ACCESSOR_CT_OFFSET>(),
            moe_block_args::arg(t0 + 7) + loop * TOKEN_LOOP_ROUTE_WORDS_STRIDE,
            ROUTER_ROUTE_PAGE_BYTES);
        noc_async_write(
            rl1,
            rg.get_noc_addr(0) + tok0 * sizeof(uint32_t) * ROUTER_ROUTE_WORDS_PER_TOKEN,
            tt::constants::TILE_HEIGHT * sizeof(uint32_t) * ROUTER_ROUTE_WORDS_PER_TOKEN);
        noc_async_write_barrier();
#endif
    };
    cb_wait_front(BIAS, expert_tiles);
#ifdef PADDED_CHUNK
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 144) = live;
#endif
    for (uint32_t it = 0; it < router_iterations; ++it) {
        volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words);
        while (*done == 0) {
            invalidate_l1_cache();
        }
        *done = 0;  // (re-arm)
        {
            {
                DeviceZoneScopedN("SE-RROWS");
                if (live_it(it)) {              // (an all-padding tile: no rows)
                    set_l1_data_cache<true>();  // (write-through; the packed tiles read after an invalidate)
                    invalidate_l1_cache();
                    rows_::make<K, expert_tiles>(
                        bv, moe_block_args::arg(t0 + 9), o_l1, moe_block_args::arg(t0 + 2), 0, 8);
                    set_l1_data_cache<false>();
                }
            }
            for (uint32_t i = 1; i < 4; ++i) {
                volatile tt_l1_ptr uint32_t* wd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 16 * i);
                while (*wd == 0) {
                    invalidate_l1_cache();
                }
                *wd = 0;  // (re-arm)
            }
#ifdef ROUTER_ROUTE_WORDS
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + 80) = 1;  // (every share of the rows written)
#endif
        }
#ifdef ROUTER_ITERATIONS_ON_ROUTER
        noc_async_write(
            o_l1,
            og.get_noc_addr(page + it * ROUTER_ROW_TILES_PER_ITERATION),
            tt::constants::TILE_HEIGHT * sizeof(uint32_t) * K);  // (my row tile of iteration it)
#else
        noc_async_write(o_l1, og.get_noc_addr(page), tt::constants::TILE_HEIGHT * sizeof(uint32_t) * K);
#endif
        noc_async_write_barrier();
#ifdef ROUTING_PLANNER
        {  // (my rows of this iteration are written: every router core's two planner RISCs told)
            const uint32_t r0 = t0 + 10 + ROUTER_ITERATION_ARGS, rows = moe_block_args::arg(r0),
                           rows2 = moe_block_args::arg(r0 + 1);
            for (uint32_t i = 0; i < ROUTING_EXCHANGE_ROUTER_CORES; ++i) {
                const uint32_t x = moe_block_args::arg(r0 + 3 + 2 * i), y = moe_block_args::arg(r0 + 4 + 2 * i);
                noc_semaphore_inc(get_noc_addr(x, y, rows), 1);
                noc_semaphore_inc(get_noc_addr(x, y, rows2), 1);
            }
        }
#endif
#ifdef ROUTER_ITERATIONS_ON_ROUTER
        if (it + 1 < router_iterations) {  // (not the last iteration: its route words now - the L1 is reused - then
                                           // this core is free)
            route_out(it);
            noc_async_atomic_barrier();
            for (uint32_t i = 0; i < npts;
                 ++i) {  // (every port: the chips sending me logits may send iteration it + 1)
                const uint32_t p = moe_block_args::arg(pt0 + i);
                noc_semaphore_inc(get_noc_addr(p & 0xFFFF, p >> 16, rit + 16), 1);
            }
            noc_semaphore_inc(get_noc_addr(c0 & 0xFFFF, c0 >> 16, rit), 1);
            if (c0me) {  // (router core 0: every router core and logits port done with it -> every matmul core's go)
                volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rit);
                while (*cnt < (ROUTER_ROW_TILES_PER_ITERATION + nact) * (it + 1)) {
                    invalidate_l1_cache();
                }
                for (uint32_t x = 0; x < gx; ++x) {
                    for (uint32_t y = 0; y < gy; ++y) {
                        noc_semaphore_inc(
                            get_noc_addr(
                                moe_block_args::arg(xs0 + x), moe_block_args::arg(ys0 + y), ew + ROUTER_GO_WORD),
                            1);
                    }
                }
                if (it + 2 == router_iterations) {  // (re-arm: the last count came)
                    noc_semaphore_inc(
                        get_noc_addr(rit), 0u - (ROUTER_ROW_TILES_PER_ITERATION + nact) * (router_iterations - 1));
                }
            }
            noc_async_atomic_barrier();
        }
#endif
    }
#ifdef ROUTING_PLANNER
    {  // the routing exchange + planner: every router core's rows written, then the routing planner
        DeviceZoneScopedN("SE-RPLAN");
        const uint32_t r0 = t0 + 10 + ROUTER_ITERATION_ARGS, rows = moe_block_args::arg(r0);
        const uint32_t B = moe_block_args::arg(r0 + 2);
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows), ROUTING_EXCHANGE_ROUTER_ROW_TILES);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows) = 0;  // (re-arm: all increments landed)
        if (B) {
            ROUTING_PLANNER_RUN(
                r0 + 3 + 2 * ROUTING_EXCHANGE_ROUTER_CORES,
                B);  // (no instance here: fewer planner instances than router cores)
        }
    }
#endif
    route_out(router_iterations - 1);  // (the last iteration's after the planner: off the routing chain)
}

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    for (uint32_t loop = 0; loop < token_loop::LOOPS; ++loop) {
        token_loop::begin(loop);
        process_token_loop(loop);
        token_loop::end(loop);
    }
}
