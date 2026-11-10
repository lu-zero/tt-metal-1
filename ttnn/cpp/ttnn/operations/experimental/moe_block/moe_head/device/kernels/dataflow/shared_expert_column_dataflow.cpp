// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// shared_expert_column_dataflow: the shared expert's column-direction dataflow on a matmul core (W, outputs)
// Runs on: matmul cores (the shared expert's engine_columns x engine_rows grid), RISCV_1 (NCRISC), NoC 1
// Responsibilities:
//  - router: the column lead reads W_r^T expert tiles from DRAM and multicasts them down the column; pushes the x / W
//    rounds to the compute; writes its partial logits to the router core's HOLD or the port's sub-block (+ counter);
//    then +1 on every port's "router done" word
//  - after its BRISC's x load: reads its W blocks from DRAM (windowed prefetch) and multicasts each down the column
//    into the W slots (CB 1, WV / WFREE semaphores)
//  - reduce-scatter: pushes the fabric-landed arrival zones to the compute, credits the ports' zone-free words, writes
//    the final tiles to y in DRAM, frees FINB once the ports read it; latent arrivals counted -> LATENT_READY_CB
//  - dispatch prep: the compute's rows into the dispatch buffer in DRAM (posted writes, a fence at the end)
// Note: column multicasts on NoC 1, row multicasts (shared_expert_row_dataflow) on NoC 0: same-NoC row and column
//  multicasts stall each other
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/token_loop.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/shared_expert_macros.hpp"

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
constexpr uint32_t engine_rows = get_compile_time_arg_val(0), row_tiles_per_core = get_compile_time_arg_val(1),
                   k_step_tiles = get_compile_time_arg_val(2);
constexpr uint32_t gate_up_k_steps = get_compile_time_arg_val(3), gate_up_passes = get_compile_time_arg_val(4),
                   row_chunks = get_compile_time_arg_val(5);
constexpr uint32_t down_k_step_tiles = get_compile_time_arg_val(6), down_k_steps = get_compile_time_arg_val(7),
                   down_passes = get_compile_time_arg_val(8);
constexpr uint32_t w_slots = get_compile_time_arg_val(9), w_slot_tiles = get_compile_time_arg_val(10),
                   WTB = get_compile_time_arg_val(11);
constexpr uint32_t WV = get_compile_time_arg_val(12), WFREE = get_compile_time_arg_val(13);
constexpr auto w_args = TensorAccessorArgs<14>();
constexpr auto y_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto p_args =
    TensorAccessorArgs<DISPATCH_PREP_ACCESSOR_CT_OFFSET>();  // (the dispatch buffer [N S, W] row-major)
FORCE_INLINE volatile tt_l1_ptr uint32_t* ew(uint32_t o) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ROUTER_PHASE_WORDS_ADDR + o);
}
constexpr auto rx_args = TensorAccessorArgs<ROUTER_ACCESSOR_CT_OFFSET>();  // (the chip's x shard [rows, H / TP], DRAM)
constexpr auto router_w_args = TensorAccessorArgs<rx_args.next_compile_time_args_offset()>();  // (W_r^T [E, H / TP])
constexpr uint32_t CB_WIN = 1, w_blocks_per_chunk =
                                   gate_up_passes * gate_up_k_steps + down_passes * down_k_steps;  // (W blocks a chunk)
#ifdef ROUTER_ITERATIONS
constexpr uint32_t router_iterations =
    ROUTER_ITERATIONS;  // (router iterations: the router phase over ROUTER_CORE_ROW_TILES of my rows at a time)
#else
constexpr uint32_t router_iterations = 1;
#endif
constexpr uint32_t NSD1 = down_k_steps ? down_k_steps : 1;  // (never divided by 0)
FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i));
}
// the column's W stream: block b of a chunk (gate / up pass b / gate_up_k_steps, step b % gate_up_k_steps; then down)
// held by row b % engine_rows, multicast down the column into W slot (round % w_slots); every core pushes what lands
// and credits the slot's next holder (arg i of the core's full list: its block of the DRAM table - head_program.cpp
// head_args_to_dram; common arg 0 is ewords, ROUTER_PHASE_WORDS_ADDR)
FORCE_INLINE uint32_t shared_expert_arg(uint32_t i) { return moe_block_args::arg(i); }
// (a row of an untilized dispatch-prep block: DISPATCH_PREP_FULL_COLUMN_TILES tiles' bf16 columns)
constexpr uint32_t prep_row_bytes = DISPATCH_PREP_FULL_COLUMN_TILES * tt::constants::TILE_WIDTH * sizeof(uint16_t);
// token loops (token_loop.hpp): the W rounds of the earlier loops - a round's slot (and its WV / WFREE semaphore) is
// its index over every round of the call, the loops' one after another (the compute pops WIN a round at a time; the
// semaphores start at 0 every loop)
inline uint32_t token_loop_rb0 = 0;
FORCE_INLINE uint32_t wslot(uint32_t w) { return (token_loop_rb0 + w) % w_slots; }
// One token loop of this kernel's work: kernel_main below runs it once per loop (token_loop.hpp).
void process_token_loop(uint32_t loop) {
    if (loop == 0) {
        token_loop_rb0 = 0;  // (a launch starts the count)
    }
    size_t a = 0;
    // (mine first; from ty to the tail the chip's: shared_expert_arg's common span)
    const uint32_t cy = shared_expert_arg(a++), cxv = shared_expert_arg(a++);
    const uint32_t wbase = shared_expert_arg(a++);
    const uint32_t wbank = shared_expert_arg(a++);  // (the DRAM bank holding my W-own region)
    const uint32_t R0 = a;                          // (the router logits' args)
    a += 10 + ROUTER_DEPTH * ROUTER_CORE_ROW_TILES * router_iterations;
    const uint32_t wg0 = a, wd0 = wg0 + gate_up_passes;  // pass widths (gate / up, down)
    a = wd0 + down_passes;
    const uint32_t ty = shared_expert_arg(a++), by = shared_expert_arg(a++);
    const uint32_t w_addr = shared_expert_arg(a++), wown = shared_expert_arg(a++), win = shared_expert_arg(a++);
    const uint32_t words = shared_expert_arg(a++),
                   y_addr = shared_expert_arg(a++) + loop * TOKEN_LOOP_Y_STRIDE_BYTES;  // (loop l's y)
    const uint32_t ys0 = a;                                                             // (the column's NoC y)
    const uint32_t sps = shared_expert_arg(ys0 + engine_rows),
                   ord0 = ys0 + engine_rows + 2;  // (steps a shard; the shard order)
    // round w's block (natural index): a gate / up pass's steps in this chip's processing order, then down's
    auto bsize = [&](uint32_t b) {  // (tiles)
        return b < gate_up_passes * gate_up_k_steps
                   ? k_step_tiles * shared_expert_arg(wg0 + b / gate_up_k_steps)
                   : down_k_step_tiles * shared_expert_arg(wd0 + (b - gate_up_passes * gate_up_k_steps) / NSD1);
    };
    // (a chunk's round -> block, and my blocks' offsets in my W-own, tabled once: computed in place, the round's block
    // (4 runtime divisions - sps is a runtime arg) and the offset's walk over my blocks ran on every pass of the
    // stream's loop below (the prefetch scan, the send's skip, the credit) - ~340 cycles a pass, a W round ~2.7
    // passes; tabled ~285 (K3 head handshake-only gate / up 48.8 -> 41.1 us, traced head ~-4 us; GLM-5.3 -5.6 us))
    static_assert(w_blocks_per_chunk <= 1024, "W rounds a chunk");
    uint16_t rblk[w_blocks_per_chunk], woffs[w_blocks_per_chunk / engine_rows + 1];
    for (uint32_t c = 0; c < w_blocks_per_chunk; ++c) {
        uint32_t b = c;
        if (c < gate_up_passes * gate_up_k_steps) {
            const uint32_t k = c / (gate_up_passes * sps), rem = c % (gate_up_passes * sps), pg = rem / sps,
                           st = rem % sps;
            b = pg * gate_up_k_steps + shared_expert_arg(ord0 + k) * sps + st;  // (shard-major: shard, pass, step)
        }
        rblk[c] = b;
    }
    for (uint32_t kb = 0, o = 0; cy + kb * engine_rows < w_blocks_per_chunk;
         ++kb) {  // (my blocks: cy, cy + engine_rows, ..)
        woffs[kb] = o;
        o += bsize(cy + kb * engine_rows);
    }
    auto blk = [&](uint32_t w) -> uint32_t { return rblk[w % w_blocks_per_chunk]; };
    const auto yg = TensorAccessor(y_args, y_addr, moe_block_tiles::kBf16TileBytes);
    // my blocks from DRAM, once: my W-own is one contiguous region in one bank (per-bank layout, wbase its offset),
    // read block by block in large bursts, every read issued now - the k-th block's under transaction id
    // min(k, 14) + 1; a block is sent once its reads landed (the column streams while the weights are still arriving)
    // the router's logits on the engine: the row's first core reads the row's x (row_tiles_per_core row tiles, the
    // chip's K slice) and multicasts it along the row; the column's first core reads the column's W_r^T expert tiles
    // and multicasts them down the column; then the compute's partial logits go to the router cores' HOLD (the main
    // loop below) (args at R0: x, W, my column's expert tiles (valid count, first), RX / RW addresses, the row's NoC x
    // (far end, first core), my row's NoC y, the router cores' NoC [RB], their HOLD, their counter, my column has
    // experts)
    const uint32_t etc = shared_expert_arg(R0 + 2), re0 = shared_expert_arg(R0 + 3);
    const uint32_t rwa = shared_expert_arg(R0 + 5);        // (R0 + 4: the row's RX, the row lead's)
    for (uint32_t it = 0; it < router_iterations; ++it) {  // (router iterations: my rows it ROUTER_CORE_ROW_TILES ..)
        {
            // K chunks of ROUTER_K_CHUNK_TILES tiles, at most two rounds, a buffer each (no slot reused: no acks): the
            // row's first core's BRISC multicasts the row's x along the row (NOC0), the column's first core multicasts
            // its W_r^T expert tiles down the column (here, NOC1 - same-NoC row and column multicasts stall each
            // other); a lead reads round rd + 1 while round rd's multicast goes; every core pushes a round once its
            // words say it landed
            DeviceZoneScopedN("SE-RTRLD");
            // (the column's W_r^T lead on row NoC x % engine_rows, not row 0: the NOC1 read responses come back along
            // the lead's row (-y, then -x), and 11 leads on one row shared its links - ~3 us a round's reads vs ~0.6
            // for the x leads, who sit on different rows)
            const uint32_t wly = shared_expert_arg(ys0 + cxv % engine_rows);
            const bool lw = etc && cy == cxv % engine_rows;
            const bool hw = shared_expert_arg(R0 + 9);
            const auto wg2 = TensorAccessor(router_w_args, shared_expert_arg(R0 + 1), moe_block_tiles::kBf16TileBytes);
            constexpr uint32_t NRD = ROUTER_K_TILES / ROUTER_K_CHUNK_TILES, WS = ROUTER_EXPERT_TILES_PER_COLUMN *
                                                                                 ROUTER_K_CHUNK_TILES *
                                                                                 moe_block_tiles::kBf16TileBytes;
            constexpr uint32_t router_rounds =
                NRD * router_iterations;  // (the rounds of every iteration: their counts run on across them)
            // (two buffers: past two rounds a buffer is reused - every receiver acks a round once its compute popped it
            // (its next reserve went through): +1 on the row's x lead and the column's W lead, who wait for all of
            // them)
            const uint32_t xlx = shared_expert_arg(R0 + 7), xly = shared_expert_arg(R0 + 8);
            for (uint32_t rd = 0; rd < NRD; ++rd) {
                const uint32_t g = it * NRD + rd;  // (the round over every iteration)
                cb_reserve_back(ROUTER_X_CB, ROUTER_K_CHUNK_TILES * ROUTER_CORE_ROW_TILES);
                cb_reserve_back(ROUTER_W_CB, ROUTER_EXPERT_TILES_PER_COLUMN * ROUTER_K_CHUNK_TILES);
                if (router_rounds > 2 && g >= 2) {
                    noc_semaphore_inc(get_noc_addr(xlx, xly, ROUTER_PHASE_WORDS_ADDR + X_ACK_FLAG_ID), 1);
                    if (hw) {
                        noc_semaphore_inc(get_noc_addr(cxv, wly, ROUTER_PHASE_WORDS_ADDR + W_ACK_FLAG_ID), 1);
                    }
                }
                if (lw) {  // (the column's first core: the chunk of its expert tiles of W_r^T down the column)
                    const uint32_t wa = rwa + ((loop * router_rounds + g) & 1) * WS;  // (over the call's rounds)
                    if (router_rounds > 2 && g >= 2) {
                        while (*ew(W_ACK_FLAG_ID) < (g - 1) * engine_rows) {
                            invalidate_l1_cache();
                        }
                    }
                    for (uint32_t e = 0; e < etc; ++e) {
                        for (uint32_t k = 0; k < ROUTER_K_CHUNK_TILES; ++k) {
                            noc_async_read(
                                wg2.get_noc_addr((re0 + e) * ROUTER_K_TILES + rd * ROUTER_K_CHUNK_TILES + k),
                                wa + (e * ROUTER_K_CHUNK_TILES + k) * moe_block_tiles::kBf16TileBytes,
                                moe_block_tiles::kBf16TileBytes);
                        }
                    }
                    noc_async_read_barrier();
                    noc_async_write_multicast_loopback_src(
                        wa,
                        get_noc_multicast_addr(cxv, by, cxv, ty, wa),
                        etc * ROUTER_K_CHUNK_TILES * moe_block_tiles::kBf16TileBytes,
                        engine_rows);
                    noc_async_writes_flushed();
                    *ew(ROUTER_W_READY_FLAG_ID) = g + 1;
                    noc_semaphore_set_multicast_loopback_src(
                        (ROUTER_PHASE_WORDS_ADDR + ROUTER_W_READY_FLAG_ID),
                        get_noc_multicast_addr(cxv, by, cxv, ty, (ROUTER_PHASE_WORDS_ADDR + ROUTER_W_READY_FLAG_ID)),
                        engine_rows);
                }
                while (*ew(ROUTER_X_READY_FLAG_ID) < g + 1) {
                    invalidate_l1_cache();
                }
                cb_push_back(ROUTER_X_CB, ROUTER_K_CHUNK_TILES * ROUTER_CORE_ROW_TILES);
                while (hw && *ew(ROUTER_W_READY_FLAG_ID) < g + 1) {
                    invalidate_l1_cache();
                }
                cb_push_back(ROUTER_W_CB, ROUTER_EXPERT_TILES_PER_COLUMN * ROUTER_K_CHUNK_TILES);
            }
            if (it + 1 == router_iterations) {
                noc_async_writes_flushed();  // (my sets' source words read: re-arm them)
                *ew(ROUTER_X_READY_FLAG_ID) =
                    0;  // (re-armed: no more sets come this call - the x lead sets from its own word, RXS)
                *ew(ROUTER_W_READY_FLAG_ID) = 0;
                if (router_rounds > 2 && lw) {  // (every ack landed: the last were waited for)
                    noc_semaphore_inc(
                        get_noc_addr(ROUTER_PHASE_WORDS_ADDR + W_ACK_FLAG_ID), 0u - (router_rounds - 2) * engine_rows);
                    noc_async_atomic_barrier();
                }
            }
        }
        {  // my partial logits (the compute's, a few us later) to their router cores' HOLD - before my W stream starts
            DeviceZoneScopedN("SE-RTROUT");
#ifdef ROUTER_ITERATIONS
            if (it > 0) {  // (router core 0's go: every router core and logits port of the chip is done with it - 1)
                while (*ew(ROUTER_GO_FLAG_ID) < it) {
                    invalidate_l1_cache();
                }
            }
#endif
            cb_wait_front(ROUTER_OUT_CB, ROUTER_EXPERT_TILES_PER_COLUMN * ROUTER_CORE_ROW_TILES);
            // (row tile r of mine: my own owner's block into its router core's HOLD, another owner's straight into the
            // port sub-block it leaves from - (NoC, base, count) a row tile; the count ETC tiles, after they landed)
            const uint32_t ro = get_read_ptr(ROUTER_OUT_CB);
            for (uint32_t r = 0; r < ROUTER_CORE_ROW_TILES; ++r) {
                const uint32_t rg = it * ROUTER_CORE_ROW_TILES + r;  // (my row rg of the call)
                const uint32_t dw = shared_expert_arg(R0 + 10 + 3 * rg), dx = dw & 0xFFFF, dy = dw >> 16;
                const uint32_t base = shared_expert_arg(R0 + 11 + 3 * rg), ctr = shared_expert_arg(R0 + 12 + 3 * rg);
                for (uint32_t e = 0; e < etc; ++e) {
                    noc_async_write(
                        ro + (e * ROUTER_CORE_ROW_TILES + r) * moe_block_tiles::kFp32TileBytes,
                        get_noc_addr(dx, dy, base + (re0 + e) * moe_block_tiles::kFp32TileBytes),
                        moe_block_tiles::kFp32TileBytes);
                }
                noc_async_write_barrier();  // (landed before the count)
                if (etc) {
                    noc_semaphore_inc(get_noc_addr(dx, dy, ctr), etc);
                }
            }
            cb_pop_front(ROUTER_OUT_CB, ROUTER_EXPERT_TILES_PER_COLUMN * ROUTER_CORE_ROW_TILES);
        }
    }  // (router iterations)
#ifdef ROUTER_ITERATIONS
    *ew(ROUTER_GO_FLAG_ID) = 0;  // (re-armed: router core 0's last go came before my last iteration's sends)
#endif
#ifdef ROUTER_DONE_CREDIT
    {  // my router phase is over - x rows / W_r^T read, the partials out: +1 on every port's "router done" word (the
       // ports credit this chip's arrival zones / tell the latent's senders "started" once every core's is: the
       // router's buffers overlay them - rtr_overlay)
        const uint32_t t0 = ord0 + shared_expert_arg(ys0 + engine_rows + 1), np_ = shared_expert_arg(t0),
                       rdw = shared_expert_arg(t0 + 3 + 2 * np_);
        for (uint32_t i = 0; i < np_; ++i) {
            noc_semaphore_inc(
                get_noc_addr(shared_expert_arg(t0 + 2 + 2 * i), shared_expert_arg(t0 + 3 + 2 * i), rdw), 1);
        }
    }
#endif
    {
        {  // (after my core's x load: the DRAM serves the x first - the gather is waiting on it)
            const uint32_t t0 = ord0 + shared_expert_arg(ys0 + engine_rows + 1);  // (past the shard order: the tail)
            const uint32_t xld = shared_expert_arg(t0 + 2 + 2 * shared_expert_arg(t0));
            while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(xld)) == 0) {
                invalidate_l1_cache();
            }
        }
    }
#ifdef SKIP_PADDING_CHUNKS
    uint32_t live_chunks;  // (a padded forward: the row chunks up to the last with a real row - my BRISC's, in ewords)
    {
        volatile tt_l1_ptr uint32_t* ne =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ROUTER_PHASE_WORDS_ADDR + LIVE_CHUNKS_FLAG_ID);
        while ((live_chunks = *ne) == 0) {
            invalidate_l1_cache();
        }
    }
#else
    constexpr uint32_t live_chunks = row_chunks;
#endif
    // the weights' DRAM reads go in stream order a window ahead of the sends (WPF of my blocks in flight): spread
    // over gate / up at the rate the stream uses them - a burst of all 35 MB at the start starved the gather
    constexpr uint8_t WNOC = noc_index;  // (the W DRAM reads: the NCRISC's own NoC)
    constexpr uint32_t WPF = 3;
    constexpr uint32_t WLA = 6;  // (the window in rounds: K3 head traced 440 -> 397 us at 6, flat over 5-8)
    uint8_t trid_of[w_blocks_per_chunk / engine_rows + 1];  // (a core's blocks: every engine_rows-th of the chunk's
                                                            // w_blocks_per_chunk)
    uint32_t wi = 0, ni = 0, pending = 0;  // (prefetch cursor over the first chunk's rounds; issued; issued unsent)
#ifdef SHARED_EXPERT_W_RING_SLOTS
    // (one chunk: my W-own a ring of SHARED_EXPERT_W_RING_SLOTS block slots - a block read into slot (its issue index %
    // SHARED_EXPERT_W_RING_SLOTS) once the block SHARED_EXPERT_W_RING_SLOTS before it was sent (multicast flushed); the
    // DRAM region keeps every block)
    uint8_t slot_of[w_blocks_per_chunk / engine_rows + 1];
    uint32_t nsent = 0;
#endif
#ifdef SHARED_EXPERT_W_RING_SLOTS
    const uint32_t WPN =
        live_chunks * w_blocks_per_chunk;  // (rounds the prefetch walks: every chunk reads its blocks again)
#else
    constexpr uint32_t WPN = w_blocks_per_chunk;  // (the first chunk's: resident after it)
#endif
    auto offset_of = [&](uint32_t b) { return static_cast<uint32_t>(woffs[b / engine_rows]); };  // (b: one of mine)
    uint32_t wla = 0;                     // (rounds landed here - the window's base)
    auto prefetch = [&](uint32_t upto) {  // (issue my blocks of rounds < upto, and the window)
        // (the window in the column's rounds, not my blocks - a block only once its round is within WLA of
        // the stream; 3 of mine in flight on every core put ~30 rounds of the whole grid in the DRAM queues at once,
        // and the first-needed block landed ~50 us after the burst was issued)
        while (wi < WPN && (wi < upto || (pending < WPF && wi < wla + WLA))) {
            const uint32_t b = blk(wi);
            if (b % engine_rows != cy) {
                ++wi;
                continue;
            }
#ifdef SHARED_EXPERT_W_RING_SLOTS
            if (ni >= nsent + SHARED_EXPERT_W_RING_SLOTS)
                break;  // (its slot still sending)
            const uint32_t lo = (ni % SHARED_EXPERT_W_RING_SLOTS) * w_slot_tiles;
            slot_of[b / engine_rows] = ni % SHARED_EXPERT_W_RING_SLOTS;
#endif
            ++wi;
            const uint32_t t = (ni++ % 14) + 1, o = offset_of(b);
#ifndef SHARED_EXPERT_W_RING_SLOTS
            const uint32_t lo = o;
#endif
            trid_of[b / engine_rows] = t;
            noc_async_read_set_trid(t, WNOC);
            noc_async_read(
                get_noc_addr_from_bank_id<true>(wbank, w_addr + wbase + o * WTB, WNOC),
                wown + lo * WTB,
                bsize(b) * WTB,
                WNOC);
            ++pending;
        }
    };
    const uint32_t w_rounds = live_chunks * w_blocks_per_chunk;  // (W rounds a call)
    uint32_t ws = 0, woff = 0, wr = 0, wc = w_slots;             // (send, my offset, pushed, credited)
    uint32_t nf[w_slots] = {};
    while (ws < w_rounds && blk(ws) % engine_rows != cy) {
        ++ws;  // (skip rounds I do not hold)
    }
    // (down + reduce-scatter) arrivals per direction in zone-slot order: slot i credited (its pages pushed - the arg's
    // high half: P, or with compact zones my tiles of its block) once my arrival counter reached its threshold (the
    // low half; 0: nothing comes - a block I hold no tile of); the fin sub-block to y
    const uint32_t t0r = ord0 + shared_expert_arg(ys0 + engine_rows + 1);
    const uint32_t q0 = t0r + 4 + 2 * shared_expert_arg(t0r);  // (past the tail)
    const uint32_t naf = shared_expert_arg(q0), af0 = q0 + 1, nab = shared_expert_arg(af0 + naf), ab0 = af0 + naf + 1;
    const uint32_t f0 = ab0 + nab;
    const uint32_t fcount = shared_expert_arg(f0), fy0 = shared_expert_arg(f0 + 1);
    const uint32_t fstride = shared_expert_arg(f0 + 2), fw = shared_expert_arg(f0 + 3);
    volatile tt_l1_ptr uint32_t* arrf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(shared_expert_arg(f0 + 4));
    volatile tt_l1_ptr uint32_t* arrb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(shared_expert_arg(f0 + 5));
    uint32_t af = 0, ab = 0, fo = 0;
// (chunks: the counters count up over them - a chunk's thresholds + its index x the arrivals a chunk; args at
// f0 + 6: the fin's y0 of chunks 1 .., the ports facing my fwd / bwd upstream (NoC x | y << 16), their zone-free
// word, my FINB-free word, the ports reading my FINB)
#ifdef REDUCE_SCATTER_FINAL_ROWS  // (a core's row_tiles_per_core rows straddle owner blocks: each (chunk, row)'s own y
                                  // base, at f0 + 6
                                  // + row_chunks + 4)
    constexpr uint32_t CHX = row_chunks + 4 + row_chunks * row_tiles_per_core;
    const uint32_t NFF = live_chunks - 1;
#else
    constexpr uint32_t CHX = row_chunks > 1 ? row_chunks + 4 : 0;
    const uint32_t NFF = row_chunks > 1 ? live_chunks - 1 : 0;
#endif
    uint32_t nfmax = 0, nbmax = 0, zuf = 0, zub = 0;  // (the thresholds' max; the zone pages my slots take)
    for (uint32_t i = 0; i < naf; ++i) {
        const uint32_t a = shared_expert_arg(af0 + i);
        nfmax = (a & 0xFFFF) > nfmax ? (a & 0xFFFF) : nfmax;
        zuf += a >> 16;
    }
    for (uint32_t i = 0; i < nab; ++i) {
        const uint32_t a = shared_expert_arg(ab0 + i);
        nbmax = (a & 0xFFFF) > nbmax ? (a & 0xFFFF) : nbmax;
        zub += a >> 16;
    }
    uint32_t zsf = 0, zsb = 0, ff = 0;  // (zone-free signals sent a direction; FINB-free credits pushed)
    const uint32_t zpf = row_chunks > 1 ? shared_expert_arg(f0 + 6 + row_chunks - 1) : 0,
                   zpb = row_chunks > 1 ? shared_expert_arg(f0 + 6 + row_chunks) : 0;
    const uint32_t zfa = row_chunks > 1 ? shared_expert_arg(f0 + 6 + row_chunks + 1) : 0;
    volatile tt_l1_ptr uint32_t* ffw =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_chunks > 1 ? shared_expert_arg(f0 + 6 + row_chunks + 2) : 0);
    const uint32_t fnr = row_chunks > 1 ? shared_expert_arg(f0 + 6 + row_chunks + 3) : 0;
#ifdef LATENT_CB
    // (the latent: once every other chip's slice of my rows has landed (lctr: lnsrc arrivals a chunk), the compute's
    // prep may untilize them)
    const uint32_t L0 = f0 + 6 + CHX, lnr = shared_expert_arg(L0), lnsrc = shared_expert_arg(L0 + 1);
    volatile tt_l1_ptr uint32_t* lctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(shared_expert_arg(L0 + 2));
    const uint32_t NLI = lnr == 0 || lnsrc == 0 ? 0 : live_chunks;
    uint32_t li = 0;
#else
    constexpr uint32_t NLI = 0, li = 0;
#endif
#ifdef LATENT_CB
    const uint32_t P0 = f0 + 6 + 3 + CHX;
#else
    const uint32_t P0 = f0 + 6 + CHX;
#endif
    const uint32_t pnu =
        shared_expert_arg(P0) / row_chunks * live_chunks;  // (prep units: chunk-major, as many a chunk)
    const auto pg = TensorAccessor(p_args, shared_expert_arg(P0 + 1), DISPATCH_PREP_ROW_BYTES);
    uint32_t fsrc = 0, frow = 0,
             fco = 0;  // (the last prep unit: the fence's source / rows - mine: nothing else writes them)
    uint32_t pu = 0;
    while (ws < w_rounds || wr < w_rounds || wc < w_rounds || af < live_chunks * naf || ab < live_chunks * nab ||
           fo < live_chunks * fcount || ff < NFF || li < NLI || pu < pnu) {
        invalidate_l1_cache();
#ifdef LATENT_CB
        if (li < NLI && *lctr >= (li + 1) * lnsrc) {
            cb_reserve_back(LATENT_READY_CB, 1);  // (the compute's prep may untilize the arrivals)
            cb_push_back(LATENT_READY_CB, 1);
            ++li;
        }
#endif
        if (pu < pnu &&
            cb_pages_available_at_front(
                DISPATCH_PREP_OUT_CB, DISPATCH_PREP_FULL_COLUMN_TILES)) {  // a prep unit's 32 rows to the buffer
            const uint32_t src = get_read_ptr(DISPATCH_PREP_OUT_CB), q = P0 + 2 + 3 * pu;
            const uint32_t row = shared_expert_arg(q) + loop * TOKEN_LOOP_DISPATCH_ROW_STRIDE,
                           co = shared_expert_arg(q + 1), nb = shared_expert_arg(q + 2);
            // (posted: non-posted DRAM writes on NOC1 - their acks - under the W stream's NOC1 DRAM reads wedged a
            // chip's NOC1, ~1 in 200 calls with the prep in every chunk; landed by the fence at the end:
            // test_posted_fence)
            for (uint32_t i = 0; i < 32; ++i) {
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                    src + i * prep_row_bytes, pg.get_noc_addr(row + i) + co, nb);
            }
            noc_async_posted_writes_flushed();  // (PO may be refilled once they left L1)
            fsrc = src, frow = row, fco = co;
            cb_pop_front(DISPATCH_PREP_OUT_CB, DISPATCH_PREP_FULL_COLUMN_TILES);
            ++pu;
        }
        if (af < live_chunks * naf) {
            const uint32_t sl = af % naf, mc = af / naf;
            if (row_chunks > 1 && sl == 0 &&
                mc > zsf) {  // (chunk mc's first slot: chunk mc - 1's zone all read (popped) -
                // +1 on the zone-free word of the port facing my fwd upstream: it credits the upstream's sends)
                if (cb_pages_reservable_at_back(REDUCE_SCATTER_ARRIVAL_FWD_CB, REDUCE_SCATTER_ZONE_TILES_FWD)) {
                    // (only a core something lands on signals: one with only "nothing comes" slots runs ahead of
                    // the credit - its signals would count for a slower one's)
                    if (nfmax) {
                        noc_semaphore_inc(get_noc_addr(zpf & 0xFFFF, zpf >> 16, zfa), 1);
                    }
                    ++zsf;
                }
            } else {
                const uint32_t a = shared_expert_arg(af0 + sl), thr = a & 0xFFFF, pz = a >> 16;
                if (thr == 0 || *arrf >= mc * nfmax + thr) {
                    if (pz) {
                        cb_reserve_back(REDUCE_SCATTER_ARRIVAL_FWD_CB, pz);
                        cb_push_back(REDUCE_SCATTER_ARRIVAL_FWD_CB, pz);
                    }
                    ++af;
                    if ((row_chunks > 1 || TOKEN_LOOPS > 1) && sl + 1 == naf &&
                        REDUCE_SCATTER_ZONE_TILES_FWD > zuf) {  // (the chunk padded to the whole zone: the
                        // compute pops it all, the next chunk's slots from the zone's start again)
                        cb_reserve_back(REDUCE_SCATTER_ARRIVAL_FWD_CB, REDUCE_SCATTER_ZONE_TILES_FWD - zuf);
                        cb_push_back(REDUCE_SCATTER_ARRIVAL_FWD_CB, REDUCE_SCATTER_ZONE_TILES_FWD - zuf);
                    }
                }
            }
        }
        if (ab < live_chunks * nab) {
            const uint32_t sl = ab % nab, mc = ab / nab;
            if (row_chunks > 1 && sl == 0 && mc > zsb) {
                if (cb_pages_reservable_at_back(REDUCE_SCATTER_ARRIVAL_BWD_CB, REDUCE_SCATTER_ZONE_TILES_BWD)) {
                    if (nbmax) {
                        noc_semaphore_inc(get_noc_addr(zpb & 0xFFFF, zpb >> 16, zfa), 1);
                    }
                    ++zsb;
                }
            } else {
                const uint32_t a = shared_expert_arg(ab0 + sl), thr = a & 0xFFFF, pz = a >> 16;
                if (thr == 0 || *arrb >= mc * nbmax + thr) {
                    if (pz) {
                        cb_reserve_back(REDUCE_SCATTER_ARRIVAL_BWD_CB, pz);
                        cb_push_back(REDUCE_SCATTER_ARRIVAL_BWD_CB, pz);
                    }
                    ++ab;
                    if ((row_chunks > 1 || TOKEN_LOOPS > 1) && sl + 1 == nab && REDUCE_SCATTER_ZONE_TILES_BWD > zub) {
                        cb_reserve_back(REDUCE_SCATTER_ARRIVAL_BWD_CB, REDUCE_SCATTER_ZONE_TILES_BWD - zub);
                        cb_push_back(REDUCE_SCATTER_ARRIVAL_BWD_CB, REDUCE_SCATTER_ZONE_TILES_BWD - zub);
                    }
                }
            }
        }
        if (fo < live_chunks * fcount && cb_pages_available_at_front(REDUCE_SCATTER_OUTPUT_FWD_CB, 1)) {
            const uint32_t mc = fo / fcount, i = fo % fcount;
#ifdef REDUCE_SCATTER_FINAL_ROWS
            const uint32_t ya = shared_expert_arg(f0 + 6 + row_chunks + 4 + mc * row_tiles_per_core + i / fw) + i % fw;
#elif defined(SHARED_EXPERT_REDUCE_SCATTER_PASSES)
            // (the rows scatter's pass-major sub-block: pass q's nr x pw tiles after the earlier passes' - widths
            // split_even(fw, SHARED_EXPERT_REDUCE_SCATTER_PASSES))
            const uint32_t y0 = mc == 0 ? fy0 : shared_expert_arg(f0 + 6 + mc - 1);  // (chunk mc's rows)
            const uint32_t nr = fcount / fw, bw = fw / SHARED_EXPERT_REDUCE_SCATTER_PASSES,
                           rm = fw % SHARED_EXPERT_REDUCE_SCATTER_PASSES;
            uint32_t q = 0, j = i, c0 = 0;
            while (j >= nr * (bw + (q < rm ? 1 : 0))) {
                j -= nr * (bw + (q < rm ? 1 : 0));
                c0 += bw + (q < rm ? 1 : 0);
                ++q;
            }
            const uint32_t pw = bw + (q < rm ? 1 : 0);
            const uint32_t ya = y0 + (j / pw) * fstride + c0 + j % pw;
#else
            const uint32_t y0 = mc == 0 ? fy0 : shared_expert_arg(f0 + 6 + mc - 1);  // (chunk mc's rows)
            const uint32_t ya = y0 + (i / fw) * fstride + i % fw;
#endif
            cb_wait_front(REDUCE_SCATTER_OUTPUT_FWD_CB, 1);
            noc_async_write(
                get_read_ptr(REDUCE_SCATTER_OUTPUT_FWD_CB), yg.get_noc_addr(ya), moe_block_tiles::kBf16TileBytes);
            noc_async_writes_flushed();  // (the page read out: free for the next fin tiles - its ack is the kernel
                                         // end's barrier; a barrier a tile serialized ~35 DRAM round trips behind the
                                         // last fin)
            cb_pop_front(REDUCE_SCATTER_OUTPUT_FWD_CB, 1);
            ++fo;
        }
        if (ff < NFF && *ffw >= (ff + 1) * fnr) {  // (the ports read chunk ff's FINB: the compute may write it again)
            cb_reserve_back(REDUCE_SCATTER_FINAL_FREE_CB, 1);
            cb_push_back(REDUCE_SCATTER_FINAL_FREE_CB, 1);
            ++ff;
        }
        if (wi < WPN) {
            prefetch(0);  // (keep the window full)
        }
        if (ws < w_rounds && (ws < w_slots || *sem(WFREE + wslot(ws)) >=
                                                  engine_rows * (nf[wslot(ws)] + 1))) {  // my block: down the column
            const uint32_t b = blk(ws), n = bsize(b), kb = b / engine_rows;  // (kb: its index among my blocks)
            woff = 0;  // (its offset in my W-own: my blocks before it)
#ifdef SHARED_EXPERT_W_RING_SLOTS
            woff = slot_of[kb] * w_slot_tiles;
#else
            woff = offset_of(b);
#endif
            if (ws < WPN) {  // (first chunk, or every chunk with the ring: its reads issued and landed)
                prefetch(ws + 1);
                noc_async_read_barrier_with_trid(trid_of[kb], WNOC);
                --pending;
            }
            const uint32_t dst = win + wslot(ws) * w_slot_tiles * WTB, vw = words + (ws & 3) * 16;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vw) = ws + 1;
            noc_async_write_multicast_loopback_src(
                wown + woff * WTB, get_noc_multicast_addr(cxv, by, cxv, ty, dst), n * WTB, engine_rows);
            noc_async_writes_flushed();
            noc_semaphore_set_multicast_loopback_src(
                vw, get_noc_multicast_addr(cxv, by, cxv, ty, get_semaphore(WV + wslot(ws))), engine_rows);
            noc_async_writes_flushed();
            if (ws >= w_slots) {
                ++nf[wslot(ws)];
            }
            ++ws;
#ifdef SHARED_EXPERT_W_RING_SLOTS
            ++nsent;  // (flushed: its slot may take a later block)
#endif
            while (ws < w_rounds && blk(ws) % engine_rows != cy) {
                ++ws;
            }
        }
        if (wr < w_rounds && (*sem(WV + wslot(wr)) >= wr + 1)) {  // W round wr landed here
            cb_reserve_back(CB_WIN, w_slot_tiles);
            cb_push_back(CB_WIN, w_slot_tiles);
            wla = ++wr;
        }
        if (wc < w_rounds && wc < wr + w_slots &&
            cb_pages_reservable_at_back(CB_WIN, w_slot_tiles * (wc - wr + 1))) {  // slot of wc free
            noc_semaphore_inc(
                get_noc_addr(cxv, shared_expert_arg(ys0 + blk(wc) % engine_rows), get_semaphore(WFREE + wslot(wc))), 1);
            ++wc;
        }
    }
#if defined(LATENT_CB) && TOKEN_LOOPS > 1
    for (; li < live_chunks;
         ++li) {  // (token loops: a page a chunk in every loop - the compute pops live_chunks - fillers past mine)
        cb_reserve_back(LATENT_READY_CB, 1);
        cb_push_back(LATENT_READY_CB, 1);
    }
#endif
    token_loop_rb0 += w_rounds;
    if (pu) {  // (the prep's fence: a non-posted rewrite of 4 B in every DRAM bank - the last unit's rows, one a bank -
        // after the posted writes on the same VC: its ack means they all landed)
        for (uint32_t i = 0; i < NUM_DRAM_BANKS; ++i) {
            noc_async_write(fsrc + i * prep_row_bytes, pg.get_noc_addr(frow + i) + fco, 4);
        }
    }
    noc_async_write_barrier();
    {  // re-arm my arrival counters (every increment of the call landed: each was waited to its final count)
        if (nfmax) {
            noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(arrf)), 0u - live_chunks * nfmax);
        }
        if (nbmax) {
            noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(arrb)), 0u - live_chunks * nbmax);
        }
        if (NFF && fnr) {
            noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(ffw)), 0u - NFF * fnr);
        }
    }
#ifdef LATENT_CB
    if (lnr && lnsrc) {
        noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(lctr)), 0u - live_chunks * lnsrc);  // (re-arm)
    }
#endif
    noc_async_atomic_barrier();
}

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    for (uint32_t loop = 0; loop < token_loop::LOOPS; ++loop) {
        token_loop::begin(loop);
        process_token_loop(loop);
        token_loop::end(loop);
    }
}
