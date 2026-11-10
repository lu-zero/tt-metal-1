// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Token loops of the MoE head, run by the kernels themselves (MoEBlockConfig.token_loops k).
//
// Why: a chunk run as k token loops processes k slices of the rows one after another. The head runs them all in one
// launch: every kernel loops k times over its work instead of the host enqueuing the head k times.
//
// Every MoE head kernel's kernel_main has this shape (the helpers below):
//
//     moe_block_args::load();                    // this core's args from the DRAM table
//     for (uint32_t loop = 0; loop < token_loop::LOOPS; ++loop) {
//         token_loop::begin(loop);              // the loop index the shared headers read (token_loop::index())
//         ...this kernel's work for one token loop...
//         token_loop::end(loop);                // the chip barrier before the next loop; after the last, the reset
//     }
//
// What changes from one loop to the next - each kernel computes it from the loop index:
//   * x and the shared expert's output: loop l's rows are TOKEN_LOOP_X_STRIDE_BYTES / TOKEN_LOOP_Y_STRIDE_BYTES bytes
//   further in every bank;
//   * the dispatch buffer's rows of this chip's tokens: TOKEN_LOOP_DISPATCH_ROW_STRIDE rows further (the prep units);
//   * the call's first row in the SP row (a padded forward's span): TOKEN_LOOP_PADDED_ROW_STRIDE rows further;
//   * the router's owner page and route words: TOKEN_LOOP_ROUTER_PAGE_STRIDE pages / TOKEN_LOOP_ROUTE_WORDS_STRIDE
//   bytes further;
//   * the routing planner plans in the last loop only (every loop's tokens); earlier loops run it empty.
// Everything else (CB layouts, slot rings, counters) carries on from loop to loop as from one row chunk to the next:
// the kernels apply their chunk-boundary rules at a loop boundary too, and count ring slots across the loops.
//
// The barrier between two loops (TOKEN_LOOPS > 1, token_loop::end). Every RISC of every core finishes loop l: its NoC
// traffic flushed (data movement) or its Tensix pipeline idle (compute), then it marks its done word. Each core has one
// lead (its NCRISC, else its BRISC): the lead waits for its core's RISCs, counts in on the coordinator core and waits
// for the go (phase A: no core still runs loop l, nothing of loop l is in flight). The lead then resets its core's
// program semaphores to 0 and counts in again (phase B: every core reset); after the second go every RISC runs loop
// l + 1. That is plain synchronization with public APIs: semaphores and NoC atomics, nothing of the launch state is
// touched.
//
// The words: persistent, zero, the same L1 address on every core (the host's "loops" words, 16 B apart): RISC r's
// done word at 16 r (BRISC 0, NCRISC 1, TRISC 2 .. 4), the go word at 80, the coordinator's count at 96. They are back
// at 0 when the kernels end, for the next launch.
//
// The barrier's table: common runtime args from TOKEN_LOOP_BARRIER_ARG (real common args, after the DRAM table's - the
// kernel's own are in its block), from the host (head_program.cpp add_head_loops):
//   [0] the words' address
//   [1] the coordinator: logical x | y << 8 | NoC x << 16 | NoC y << 24
//   [2] the program's cores (the barrier's count)
//   then (a lead kernel, TOKEN_LOOP_LEAD) the cores it leads: either 1 << 31 and one role for every core it runs on, or
//   n and n x (logical x | y << 8, role); a role = (the core's RISCs: bit r for RISC r, the core's program semaphores:
//   bit s) then (the coordinator kernel, TOKEN_LOOP_COORDINATOR) the cores it releases: n, n x (NoC x | y << 16)
#pragma once

#include <cstdint>
// (every kernel's args: moe_block_args::arg / moe_block_args::common, read from this core's block of the DRAM tables -
// head_program.cpp head_args_to_dram; kernel_main loads them first; it also brings the compute / dataflow APIs the
// barrier below uses)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"

// (the host's token loop defines - head_program.cpp; one loop when absent)
#ifndef TOKEN_LOOPS
#define TOKEN_LOOPS 1
#endif
#ifndef TOKEN_LOOP_X_STRIDE_BYTES
#define TOKEN_LOOP_X_STRIDE_BYTES 0
#endif
#ifndef TOKEN_LOOP_Y_STRIDE_BYTES
#define TOKEN_LOOP_Y_STRIDE_BYTES 0
#endif
#ifndef TOKEN_LOOP_DISPATCH_ROW_STRIDE
#define TOKEN_LOOP_DISPATCH_ROW_STRIDE 0
#endif
#ifndef TOKEN_LOOP_PADDED_ROW_STRIDE
#define TOKEN_LOOP_PADDED_ROW_STRIDE 0
#endif
#ifndef TOKEN_LOOP_ROUTER_PAGE_STRIDE
#define TOKEN_LOOP_ROUTER_PAGE_STRIDE 0
#endif
#ifndef TOKEN_LOOP_ROUTE_WORDS_STRIDE
#define TOKEN_LOOP_ROUTE_WORDS_STRIDE 0
#endif

namespace token_loop {
constexpr uint32_t LOOPS = TOKEN_LOOPS;
inline uint32_t g_current_loop = 0;  // (set by begin(): the shared headers read it through index() / is_last())
inline __attribute__((always_inline)) uint32_t index() { return g_current_loop; }
inline __attribute__((always_inline)) bool is_last() { return g_current_loop + 1 == LOOPS; }

#if TOKEN_LOOPS > 1
// (this RISC's done word: 16 x its index)
#if defined(COMPILE_FOR_BRISC)
constexpr uint32_t MY_RISC = 0;
#elif defined(COMPILE_FOR_NCRISC)
constexpr uint32_t MY_RISC = 1;
#elif defined(UCK_CHLKC_UNPACK)
constexpr uint32_t MY_RISC = 2;
#elif defined(UCK_CHLKC_MATH)
constexpr uint32_t MY_RISC = 3;
#else
constexpr uint32_t MY_RISC = 4;
#endif
constexpr uint32_t RISCS = 5, WORDS = 6;  // (the done words of the 5 RISCs, then the go word)
constexpr uint32_t GO_WORD = 80, COUNT_WORD = 96;

inline __attribute__((always_inline)) uint32_t barrier_arg(uint32_t i) {
    return get_common_arg_val<uint32_t>(TOKEN_LOOP_BARRIER_ARG + i);
}
inline __attribute__((always_inline)) volatile tt_l1_ptr uint32_t* word(uint32_t offset) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_arg(0) + offset);
}
inline __attribute__((always_inline)) void wait_at_least(uint32_t offset, uint32_t value) {
    while (*word(offset) < value) {
#ifndef COMPILE_FOR_TRISC
        invalidate_l1_cache();  // (the data movement RISCs' L1 data cache; the compute RISCs have none)
#endif
    }
}

// (every RISC) my loop is over here: my NoC traffic out of L1 and acked, or my Tensix pipeline idle
__attribute__((noinline)) inline void finish() {
#ifndef COMPILE_FOR_TRISC
    noc_async_full_barrier();
#else
    ckernel::tensix_sync();
#endif
}

#if !defined(COMPILE_FOR_TRISC) && defined(TOKEN_LOOP_LEAD)
struct Role {
    uint32_t riscs = 0, semaphores = 0;
};
// my core's role if this kernel leads it (riscs 0: not the lead)
__attribute__((noinline)) inline Role my_role() {
    uint32_t i = 3;
    if (barrier_arg(i) >> 31) {  // (every core of the kernel alike)
        return Role{barrier_arg(i + 1), barrier_arg(i + 2)};
    }
    const uint32_t me = get_absolute_logical_x() | (uint32_t(get_absolute_logical_y()) << 8);
    for (uint32_t core = 0, cores = barrier_arg(i++); core < cores; ++core, i += 3) {
        if (barrier_arg(i) == me) {
            return Role{barrier_arg(i + 1), barrier_arg(i + 2)};
        }
    }
    return Role{};
}
// (the coordinator's release list: past the roles)
inline __attribute__((always_inline)) uint32_t releases_at() {
    const uint32_t i = 3;
    return (barrier_arg(i) >> 31) ? i + 3 : i + 1 + 3 * barrier_arg(i);
}
// (lead) one phase of the barrier: count in on the coordinator, which releases every core with go = value
__attribute__((noinline)) inline void chip_phase(uint32_t value) {
    const uint32_t coordinator = barrier_arg(1);
    noc_semaphore_inc(get_noc_addr((coordinator >> 16) & 0xFF, coordinator >> 24, barrier_arg(0) + COUNT_WORD), 1);
    noc_async_atomic_barrier();
#ifdef TOKEN_LOOP_COORDINATOR
    if ((coordinator & 0xFF) == get_absolute_logical_x() && ((coordinator >> 8) & 0xFF) == get_absolute_logical_y()) {
        const uint32_t cores = barrier_arg(2);
        wait_at_least(COUNT_WORD, cores);
        noc_semaphore_inc(get_noc_addr(barrier_arg(0) + COUNT_WORD), 0u - cores);  // (re-armed before any release)
        noc_async_atomic_barrier();
        uint32_t i = releases_at();
        for (uint32_t core = 0, released = barrier_arg(i++); core < released; ++core) {
            const uint32_t noc_xy = barrier_arg(i++);
            noc_inline_dw_write(get_noc_addr(noc_xy & 0xFFFF, noc_xy >> 16, barrier_arg(0) + GO_WORD), value);
        }
        noc_async_write_barrier();
    }
#endif
    wait_at_least(GO_WORD, value);
}
#endif

// between loop l and l + 1
__attribute__((noinline)) inline void between(uint32_t loop) {
    finish();
    const uint32_t phase_a = 2 * loop + 1, phase_b = 2 * loop + 2;
    *word(16 * MY_RISC) = phase_a;
#if !defined(COMPILE_FOR_TRISC) && defined(TOKEN_LOOP_LEAD)
    if (const Role role = my_role(); role.riscs) {  // the lead: my core's RISCs, the chip; my semaphores; the chip
        for (uint32_t risc = 0; risc < RISCS; ++risc) {
            if (role.riscs >> risc & 1) {
                wait_at_least(16 * risc, phase_a);
            }
        }
        chip_phase(phase_a);
        for (uint32_t semaphore = 0; semaphore < 32; ++semaphore) {
            if (role.semaphores >> semaphore & 1) {
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(semaphore)), 0);
            }
        }
        chip_phase(phase_b);
    } else {
        wait_at_least(GO_WORD, phase_b);
    }
#else
    wait_at_least(GO_WORD, phase_b);
#endif
}

// after the last loop: the words back to 0 for the next launch (the lead, once its core's RISCs are done)
__attribute__((noinline)) inline void after_last() {
    finish();
    const uint32_t last_phase = 2 * TOKEN_LOOPS - 1;
    *word(16 * MY_RISC) = last_phase;
#if !defined(COMPILE_FOR_TRISC) && defined(TOKEN_LOOP_LEAD)
    if (const Role role = my_role(); role.riscs) {
        for (uint32_t risc = 0; risc < RISCS; ++risc) {
            if (role.riscs >> risc & 1) {
                wait_at_least(16 * risc, last_phase);
            }
        }
        for (uint32_t w = 0; w < WORDS; ++w) {
            *word(16 * w) = 0;
        }
    }
#endif
}
#endif  // TOKEN_LOOPS > 1

// At the top of each loop: the loop the shared headers see.
inline __attribute__((always_inline)) void begin(uint32_t loop) { g_current_loop = loop; }

// At the end of each loop: the chip barrier before the next loop, or after the last loop the barrier words' reset
// (nothing with one loop).
inline __attribute__((always_inline)) void end([[maybe_unused]] uint32_t loop) {
#if TOKEN_LOOPS > 1
    if (loop + 1 < LOOPS) {
        between(loop);
    } else {
        after_last();
    }
#endif
}
}  // namespace token_loop
