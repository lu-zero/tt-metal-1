// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// shared_expert_row_dataflow: the shared expert's row-direction dataflow on a matmul core (x in, x / h along the row)
// Runs on: matmul cores (the shared expert's engine_columns x engine_rows grid), RISCV_0 (BRISC), NoC 0
// Responsibilities:
//  - reads this core's steps of the chip's own x shard from DRAM into x-own L1; then +1 on every port core's
//    "x loaded" word and the row lead's word, and lets its NCRISC's W load start
//  - row lead: reads the row's router x from the row cores' x-own L1 and multicasts it along the row (RX, x-ready word)
//  - x steps along the row into CB 0 (kernel_lib multicasts x0 / x1), a gathered shard once XREADY (x_gather_watcher)
//    says it landed; h along the row into every row core's h-all slot (multicast h, CB 3)
//  - hand-offs to the port cores (+1 on their mailbox words): the latent block packed (LATENT_CB), each reduce-scatter
//    sub-block done (REDUCE_SCATTER_FINAL_COUNTER_CB); publishes the live row chunks to its NCRISC and compute
// Note: row multicasts on NoC 0, column multicasts (shared_expert_column_dataflow) on NoC 1: same-NoC row and column
//  multicasts stall each other
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/token_loop.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/shared_expert_macros.hpp"

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "api/debug/kernel_profiler.hpp"
#ifdef SKIP_PADDING_CHUNKS
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/padded_rows.hpp"
#endif
using namespace dataflow_kernel_lib;
constexpr uint32_t engine_columns = get_compile_time_arg_val(0), row_tiles_per_core = get_compile_time_arg_val(1),
                   k_step_tiles = get_compile_time_arg_val(2);
constexpr uint32_t gate_up_k_steps = get_compile_time_arg_val(3), gate_up_passes = get_compile_time_arg_val(4),
                   row_chunks = get_compile_time_arg_val(5);
constexpr uint32_t max_column_pairs = get_compile_time_arg_val(6), S0 = get_compile_time_arg_val(7),
                   S1 = get_compile_time_arg_val(8);
constexpr uint32_t XREADY = get_compile_time_arg_val(9);
constexpr auto x_args = TensorAccessorArgs<10>();
constexpr uint32_t CB_XIN = 0, CB_HOWN = 2, CB_HALL = 3;
constexpr uint32_t NF = gate_up_k_steps / 2, NOWN0 = NF / S0, NOWN1 = NF / S1;  // (steps a family a pass; per owner)
// x position j (the j-th K step this chip processes: its own shard's first, then each shard as it lands): family j % 2,
// owner = the family's subset[(j / 2) % S] (S divides a shard's steps / 2, so the rotation over (shard, pass, step)
// rounds keeps every position on one owner), its local index (j / 2) / S
FORCE_INLINE uint32_t own_slot(uint32_t j) { return (j & 1) ? (j / 2) / S1 : (j / 2) / S0; }
// (arg i of the core's full list: its block of the DRAM table - head_program.cpp head_args_to_dram; common arg 0 is
// ewords, ROUTER_PHASE_WORDS_ADDR)
FORCE_INLINE uint32_t shared_expert_arg(uint32_t i) { return moe_block_args::arg(i); }
// token loops (token_loop.hpp): the row chunks of the earlier loops - the h-all slots alternate over every chunk of the
// call, the loops' one after another (the compute pops HALL a chunk at a time)
inline uint32_t token_loop_m0 = 0;
// One token loop of this kernel's work: kernel_main below runs it once per loop (token_loop.hpp).
void process_token_loop(uint32_t loop) {
    if (loop == 0) {
        token_loop_m0 = 0;  // (a launch starts the count)
    }
    size_t a = 0;
    // (mine first - then from x_addr to y's address the chip's: shared_expert_arg's common span)
    const uint32_t cx = shared_expert_arg(a++), cy = shared_expert_arg(a++);
    const uint32_t i0 = shared_expert_arg(a++),
                   i1 = shared_expert_arg(a++);  // my index in family 0 / 1 (or engine_columns)
    const uint32_t ry = shared_expert_arg(a++);
    const uint32_t x_addr = shared_expert_arg(a++) + loop * TOKEN_LOOP_X_STRIDE_BYTES, x_ct = shared_expert_arg(a++),
                   x_k0 = shared_expert_arg(a++);  // (loop l's x)
    const uint32_t xown = shared_expert_arg(a++), xin = shared_expert_arg(a++), hown = shared_expert_arg(a++);
    const uint32_t hall = shared_expert_arg(a++), lx = shared_expert_arg(a++), rx = shared_expert_arg(a++);
    const uint32_t rtr_xa = shared_expert_arg(a++);  // (RX)
    const uint32_t ft0 = a;  // (the x families: family 0 / 1 columns [S0] [S1], per column (nown, family-1 base slot))
    a += S0 + S1 + 2 * engine_columns;
    const uint32_t vx0 = a, sps = shared_expert_arg(vx0 + engine_columns),
                   g = shared_expert_arg(vx0 + engine_columns + 1);
    const uint32_t ord0 =
        vx0 + engine_columns + 2;  // (the row's NoC x; steps a shard, shards; the shard order: own, then arrival)
    auto step_of = [&](uint32_t j) { return shared_expert_arg(ord0 + j / sps) * sps + j % sps; };  // (position j)
    const uint32_t nown = (i0 < engine_columns ? NOWN0 : 0) + (i1 < engine_columns ? NOWN1 : 0);   // (my steps a chunk)
    auto mine = [&](uint32_t j) { return (j & 1) ? ((j / 2) % S1 == i1) : ((j / 2) % S0 == i0); };
    // my steps' x-own slot: [chunk][family 0 slots, then family 1][row_tiles_per_core][k_step_tiles]
    auto own_addr = [&](uint32_t m, uint32_t j) {  // (every row chunk's x own resident)
        const uint32_t slot = (j & 1) ? (i0 < engine_columns ? NOWN0 : 0) + own_slot(j) : own_slot(j);
        return xown + ((m * nown + slot) * row_tiles_per_core) * k_step_tiles * moe_block_tiles::kBf16TileBytes;
    };
    auto rtrx = [&]() {  // (called after my own-shard x load: the gather's ports wait on every matmul core's - a row
        // lead's x load behind its ~20 us of router x held the gather back that long: GLM-5.3 ring 640 235.8 -> 219.2
        // us)
        if (cx == 0) {  // the router's x: the row's first core reads the row's x (row_tiles_per_core row tiles, the
                        // chip's K slice) a K chunk
            // a round into RX and multicasts it along the row on NOC0 (the NCRISC's W_r^T goes down the columns on
            // NOC1), then sets the round on the row's RX-ready word (from my own source word RXS: the NCRISC re-arms
            // the ready word)
            DeviceZoneScopedN("SE-RTRX");
            constexpr uint32_t XS =
                ROUTER_K_CHUNK_TILES * ROUTER_CORE_ROW_TILES *
                moe_block_tiles::kBf16TileBytes;  // (ROUTER_CORE_ROW_TILES: my row_tiles_per_core rows of every chunk -
                                                  // row rr of chunk rr / row_tiles_per_core)
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ROUTER_PHASE_WORDS_ADDR + ROUTER_X_SENT_FLAG_ID);
            constexpr uint32_t NRD = ROUTER_K_TILES / ROUTER_K_CHUNK_TILES;
#ifdef ROUTER_ITERATIONS
            constexpr uint32_t router_iterations =
                ROUTER_ITERATIONS;  // (router iterations: ROUTER_CORE_ROW_TILES of the row's rows each)
#else
            constexpr uint32_t router_iterations = 1;
#endif
            constexpr uint32_t router_rounds =
                NRD * router_iterations;  // (the rounds of every iteration: their counts run on)
            volatile tt_l1_ptr uint32_t* xack =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ROUTER_PHASE_WORDS_ADDR + X_ACK_FLAG_ID);
            {  // (the row's x is the own shard the row's cores just loaded into x own: read from their L1, not the 2 MB
               // again from DRAM beside the own-x loads and W_r^T - once every row core counted its load on my XRL
               // word)
                volatile tt_l1_ptr uint32_t* xrl =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ROUTER_PHASE_WORDS_ADDR + X_READ_LOCAL_FLAG_ID);
                while (*xrl < engine_columns) invalidate_l1_cache();
                *xrl = 0;  // (re-armed: one count a row core a call, all in)
            }
            for (uint32_t g = 0; g < router_rounds; ++g) {
                const uint32_t rd = g % NRD, it = g / NRD;  // (round rd of iteration it)
                // (two buffers, alternating over every round of the call - the loops' one after another: the receivers
                // push a round each, the compute pops; past two rounds of a loop the row's acks first)
                const uint32_t xa = rtr_xa + ((loop * router_rounds + g) & 1) * XS;
                if (router_rounds > 2 && g >= 2)
                    while (*xack < (g - 1) * engine_columns) invalidate_l1_cache();
                for (uint32_t r = 0; r < ROUTER_CORE_ROW_TILES; ++r)
                    for (uint32_t k = 0; k < ROUTER_K_CHUNK_TILES; ++k) {
                        const uint32_t rg = it * ROUTER_CORE_ROW_TILES + r;  // (the row's row rg of the call)
                        // (K tile kt of the chip's slice: own-shard position kt / k_step_tiles - its owner column and
                        // x-own slot as x_gather_schedule.hpp's dst_of; row r: chunk r / row_tiles_per_core, local row
                        // r % row_tiles_per_core)
                        const uint32_t kt = rd * ROUTER_K_CHUNK_TILES + k, j = kt / k_step_tiles, hh = j / 2;
                        const uint32_t col =
                            (j & 1) ? shared_expert_arg(ft0 + S0 + hh % S1) : shared_expert_arg(ft0 + hh % S0);
                        const uint32_t nw = shared_expert_arg(ft0 + S0 + S1 + 2 * col),
                                       b1 = shared_expert_arg(ft0 + S0 + S1 + 2 * col + 1);
                        const uint32_t slot = (j & 1) ? b1 + hh / S1 : hh / S0;
                        const uint32_t xa_ =
                            xown +
                            ((((rg / row_tiles_per_core) * nw + slot) * row_tiles_per_core + rg % row_tiles_per_core) *
                                 k_step_tiles +
                             kt % k_step_tiles) *
                                moe_block_tiles::kBf16TileBytes;
                        noc_async_read(
                            get_noc_addr(shared_expert_arg(vx0 + col), ry, xa_),
                            xa + (k * ROUTER_CORE_ROW_TILES + r) * moe_block_tiles::kBf16TileBytes,
                            moe_block_tiles::kBf16TileBytes);
                    }
                noc_async_read_barrier();
                noc_async_write_multicast_loopback_src(
                    xa, get_noc_multicast_addr(lx, ry, rx, ry, xa), XS, engine_columns);
                noc_async_writes_flushed();
                *src = g + 1;
                noc_semaphore_set_multicast_loopback_src(
                    ROUTER_PHASE_WORDS_ADDR + ROUTER_X_SENT_FLAG_ID,
                    get_noc_multicast_addr(lx, ry, rx, ry, ROUTER_PHASE_WORDS_ADDR + ROUTER_X_READY_FLAG_ID),
                    engine_columns);
                noc_async_writes_flushed();
            }
            if (router_rounds > 2) {  // (re-arm: every ack landed - the last were waited for)
                noc_semaphore_inc(
                    get_noc_addr(ROUTER_PHASE_WORDS_ADDR + X_ACK_FLAG_ID), 0u - (router_rounds - 2) * engine_columns);
                noc_async_atomic_barrier();
            }
        }
    };
    const auto xg = TensorAccessor(x_args, x_addr, moe_block_tiles::kBf16TileBytes);
    auto xload_step = [&](uint32_t m, uint32_t j) {  // (one owned step of chunk m from DRAM x [rows, x_ct])
        const uint32_t s = step_of(j);
        if (!(mine(j) && s * k_step_tiles >= x_k0 && s * k_step_tiles < x_k0 + x_ct)) {
            return;
        }
        for (uint32_t r = 0; r < row_tiles_per_core; ++r) {
            for (uint32_t kk = 0; kk < k_step_tiles; ++kk) {
                noc_async_read(
                    xg.get_noc_addr(
                        GLOBAL_ROW_TILE(m, cy * row_tiles_per_core + r) * x_ct + s * k_step_tiles + kk - x_k0),
                    own_addr(m, j) + (r * k_step_tiles + kk) * moe_block_tiles::kBf16TileBytes,
                    moe_block_tiles::kBf16TileBytes);
            }
        }
    };
    if (x_ct) {  // (alone) my steps of every chunk from DRAM x [rows, x_ct]: columns x_k0 ..
        DeviceZoneScopedN("SE-XLOAD");
        for (uint32_t m = 0; m < row_chunks; ++m) {
            for (uint32_t j = 0; j < gate_up_k_steps; ++j) {
                xload_step(m, j);
            }
        }
        noc_async_read_barrier();
    }
    {  // (my own-shard x is loaded: count it on the gather's ports - they send it from here - and let my W load start)
        const uint32_t np0 = ord0 + g, nports = shared_expert_arg(np0), xl_addr = shared_expert_arg(np0 + 1);
        for (uint32_t i = 0; i < nports; ++i) {
            noc_semaphore_inc(
                get_noc_addr(shared_expert_arg(np0 + 2 + 2 * i), shared_expert_arg(np0 + 3 + 2 * i), xl_addr), 1);
        }
        noc_semaphore_inc(
            get_noc_addr(shared_expert_arg(vx0), ry, ROUTER_PHASE_WORDS_ADDR + X_READ_LOCAL_FLAG_ID),
            1);  // (and on my row's lead: the router's x)
        noc_async_atomic_barrier();
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(shared_expert_arg(np0 + 2 + 2 * nports))) = 1;
    }
#ifdef SKIP_PADDING_CHUNKS
    // a padded forward: the row chunks up to the last with a real row only (the rest are padding: no x, no shared
    // expert, no reduce-scatter) - for my NCRISC and compute too (ewords + LIVE_CHUNKS_FLAG_ID, zeroed at my end).
    // After my x load and its count on the gather's ports (they go first), before the router's x (my NCRISC's weights
    // wait on it)
    const uint32_t live_chunks = padded_live_chunks(
        [](uint32_t i) {
            return moe_block_args::common(SKIP_PADDING_ARGS_OFFSET + i) +
                   (i == 5 ? loop * TOKEN_LOOP_PADDED_ROW_STRIDE : 0u);
        },
        0,
        (moe_block_args::common(SKIP_PADDING_ARGS_OFFSET + 10) + 63) & ~63u);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ROUTER_PHASE_WORDS_ADDR + LIVE_CHUNKS_FLAG_ID) = live_chunks;
#else
    constexpr uint32_t live_chunks = row_chunks;
#endif
    rtrx();  // (the router's x after my own x)
    constexpr auto mx0 =
        McastArgs<get_named_compile_time_arg_val("x0_ct_offset"), get_named_compile_time_arg_val("x0_rt_offset")>();
    constexpr auto mx1 =
        McastArgs<get_named_compile_time_arg_val("x1_ct_offset"), get_named_compile_time_arg_val("x1_rt_offset")>();
    constexpr auto mh =
        McastArgs<get_named_compile_time_arg_val("h_ct_offset"), get_named_compile_time_arg_val("h_rt_offset")>();
    Noc noc;
    auto sh = mh.optional_sender(noc);
    auto rh = mh.optional_receiver(noc);
    auto s0 = mx0.optional_sender(noc);
    auto r0 = mx0.optional_receiver(noc);
    auto s1 = mx1.optional_sender(noc);
    auto r1 = mx1.optional_receiver(noc);
    volatile tt_l1_ptr uint32_t* xready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(XREADY));
    for (uint32_t m = 0; m < live_chunks; ++m) {
        {
            DeviceZoneScopedN("SE-XROUNDS");
            // shard-major: each shard (own first, then as it lands) through every pass before the next shard
            for (uint32_t k = 0; k < g; ++k) {
                for (uint32_t pg = 0; pg < gate_up_passes; ++pg) {
                    for (uint32_t st = 0; st < sps; ++st) {
                        const uint32_t j =
                            k * sps + st;  // (position; its family j % 2 = st % 2: a block's rounds are even)
                        const uint32_t round =
                            ((m * g + k) * gate_up_passes + pg) * (sps / 2) + st / 2;  // (the family's)
                        cb_reserve_back(CB_XIN, row_tiles_per_core * k_step_tiles);
                        const uint32_t dst =
                            xin + (j & 1) * row_tiles_per_core * k_step_tiles * moe_block_tiles::kBf16TileBytes;
                        if (mine(j)) {  // (position j: the shard of rank j / sps must be in - the own shard always is)
                            if (j >= sps) {
                                noc_semaphore_wait_min(xready, m * (g - 1) + j / sps);
                            }
                            constexpr uint32_t XBYTES =
                                row_tiles_per_core * k_step_tiles * moe_block_tiles::kBf16TileBytes;
                            if (j & 1) {
                                s1->send(own_addr(m, j), dst, XBYTES);
                            } else {
                                s0->send(own_addr(m, j), dst, XBYTES);
                            }
                        } else {
                            if (j & 1) {
                                r1->receive(round);
                            } else {
                                r0->receive(round);
                            }
                        }
                        cb_push_back(CB_XIN, row_tiles_per_core * k_step_tiles);
                    }
                }
            }
        }
#ifdef LATENT_CB
        {  // (the latent's all-to-all: once it is packed, +1 on each destination's port word of mine)
            DeviceZoneScopedN("SE-LHAND");
            const uint32_t np0 = ord0 + g, nports = shared_expert_arg(np0), r0 = np0 + 5 + 2 * nports;
            const uint32_t l0 = r0 + 2 + 2 * shared_expert_arg(r0), nl = shared_expert_arg(l0);
            const uint32_t lw = shared_expert_arg(l0 + 1);
            cb_wait_front(LATENT_CB, (m + 1) * LATENT_PAGES);  // (chunk m's block; the words count up over the chunks)
            for (uint32_t j = 0; j < nl; ++j) {
                noc_semaphore_inc(
                    get_noc_addr(shared_expert_arg(l0 + 2 + 2 * j), shared_expert_arg(l0 + 3 + 2 * j), lw), 1);
            }
        }
#endif
        {  // (its own zone: the hand-offs below are not h exchange)
            DeviceZoneScopedN("SE-HX");
            // h: every core's h block [row_tiles_per_core][max_column_pairs] into every row core's h-all slot (m % 2)
            // as block cx
            // ([engine_columns][row_tiles_per_core][max_column_pairs]): one kernel_lib rotating round a column; the
            // receivers' readiness ack is the slot's credit
            constexpr uint32_t h_gathered_tiles = engine_columns * row_tiles_per_core * max_column_pairs;
            cb_wait_front(CB_HOWN, row_tiles_per_core * max_column_pairs);
            cb_reserve_back(CB_HALL, h_gathered_tiles);
            const uint32_t hs = hall + (SHARED_EXPERT_H_SLOTS > 1 ? ((token_loop_m0 + m) & 1) : 0) * h_gathered_tiles *
                                           moe_block_tiles::kBf16TileBytes;  // (h-all slots: SHARED_EXPERT_H_SLOTS)
            for (uint32_t c = 0; c < engine_columns; ++c) {
                if (c == cx) {
                    sh->send(
                        hown,
                        hs + c * row_tiles_per_core * max_column_pairs * moe_block_tiles::kBf16TileBytes,
                        row_tiles_per_core * max_column_pairs * moe_block_tiles::kBf16TileBytes);
                } else {
                    rh->receive(m * engine_columns + c);
                }
                cb_push_back(
                    CB_HALL,
                    row_tiles_per_core *
                        max_column_pairs);  // (column c's block: the down's first pass starts on its K tiles)
            }
            cb_pop_front(CB_HOWN, row_tiles_per_core * max_column_pairs);
        }
        {  // (down + reduce-scatter) each finished sub-block's hand-off: +1 on its port's mailbox word of mine (the
           // words count up over the chunks: the port's targets do)
            DeviceZoneScopedN("SE-HAND");
            const uint32_t np0 = ord0 + g, nports = shared_expert_arg(np0), r0 = np0 + 5 + 2 * nports;
            const uint32_t nh = shared_expert_arg(r0), mbox = shared_expert_arg(r0 + 1);
            for (uint32_t j = 0; j < nh; ++j) {
                cb_wait_front(REDUCE_SCATTER_FINAL_COUNTER_CB, j + 1);
                noc_semaphore_inc(
                    get_noc_addr(shared_expert_arg(r0 + 2 + 2 * j), shared_expert_arg(r0 + 3 + 2 * j), mbox), 1);
            }
            if constexpr (row_chunks > 1 || TOKEN_LOOPS > 1) {
                cb_pop_front(
                    REDUCE_SCATTER_FINAL_COUNTER_CB,
                    nh);  // (the next chunk's - or loop's - hand-offs from the front again)
            }
        }
    }
#if defined(LATENT_CB) && TOKEN_LOOPS > 1
    // (token loops: the latent's whole region popped - the compute pads the skipped chunks - so the next loop's packs
    // start at its start, where the ports read them)
    cb_wait_front(LATENT_CB, row_chunks * LATENT_PAGES);
    cb_pop_front(LATENT_CB, row_chunks * LATENT_PAGES);
#endif
    token_loop_m0 += live_chunks;
    noc_async_atomic_barrier();
#ifdef SKIP_PADDING_CHUNKS
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ROUTER_PHASE_WORDS_ADDR + LIVE_CHUNKS_FLAG_ID) =
        0;  // (re-arm: my NCRISC and compute read it)
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
