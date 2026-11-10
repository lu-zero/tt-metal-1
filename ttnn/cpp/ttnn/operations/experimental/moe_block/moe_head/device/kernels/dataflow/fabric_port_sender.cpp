// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fabric_port_sender: sends everything this chip sends through one fabric port core (a direction and link)
// Runs on: port cores (one per X direction and fabric link), RISCV_1 (NCRISC), NoC 0 (RISCV_0_default)
// Responsibilities:
//  - x all-gather: pops CB 0 (fabric_port_stager) and fabric-writes each packet into the neighbour's matmul cores'
//    x-own L1, incrementing the downstream port's arrival counter; +1 on the neighbour's x_gather_watcher a shard
//  - from CB 0: reduce-scatter sub-blocks (the downstream chip's zones), latent all-to-all rows (the destination's
//    LATZ); from L1: router logits partials (the owner chips' router arrival zones), the routing exchange's X packets
//    (the destination chips' DRAM slots)
//  - fabric fences and credits: "ready" / "started" words, zone credits after "router done"; re-arms its words
// Note: in this op the reading side runs on BRISC (RISCV_0) and the sending side on NCRISC (RISCV_1), opposite to
//  tt-metal's Reader/Writer config convention, on purpose
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/token_loop.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "api/debug/kernel_profiler.hpp"
using namespace tt::tt_fabric;
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/x_gather_compile_args.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/x_gather_schedule.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/latent_all_to_all_sender.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/reduce_scatter_sender.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/router_logits_stream.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_exchange_send.hpp"

// One token loop of this kernel's work: kernel_main below runs it once per loop (token_loop.hpp).
void process_token_loop(uint32_t loop) {
    // (token loops: every loop allocates its packet headers anew - the last loop's packets were flushed by its
    // connection's close)
    PacketHeaderPool::reset();
#ifdef SKIP_PADDING_CHUNKS
    x_gather_find_live_chunks();  // (a padded forward: the row chunks up to the last with a real row)
#endif
    size_t a = 0;
    const uint32_t xown = moe_block_args::arg(a++), v = moe_block_args::arg(a++);
    const uint32_t arrival = moe_block_args::arg(a++), expect_in = moe_block_args::arg(a++);  // (my counter)
    const uint32_t ready = moe_block_args::arg(a++), wait_ready = moe_block_args::arg(a++);
    const uint32_t send_ready = moe_block_args::arg(a++), rx = moe_block_args::arg(a++), ry = moe_block_args::arg(a++);
    const uint32_t px = moe_block_args::arg(a++), py = moe_block_args::arg(a++);      // downstream port (NoC)
    const uint16_t mesh = moe_block_args::arg(a++), chip = moe_block_args::arg(a++);  // the neighbour
    const uint32_t wx = moe_block_args::arg(a++), wy = moe_block_args::arg(a++), watch = moe_block_args::arg(a++);
    const uint32_t n = moe_block_args::arg(a++);
    const uint32_t ent = a, T = ent + 3 * n;  // entries (q, part, packets a chunk); tables
    a = T + G + S0 + S1 + 3 * engine_columns +
        engine_rows;  // (past the tables: the RS units, then the fabric connection)
    const uint32_t rs0 = a, rnu = moe_block_args::arg(rs0), rnc = moe_block_args::arg(rs0 + 1 + 2 * rnu);
    const uint32_t ls0 = rs0 + 2 + 2 * rnu + 2 * rnc + G + RSX;
    // (chunks: the zone credits - my zone-free word (my chip's cores, a chunk's arrivals from my peer read), the
    // signals a chunk; forwarded to the peer's port facing me (its zone-credit word): row_chunks - 1 a call)
    const uint32_t rse = rs0 + 2 + 2 * rnu + 2 * rnc + G;
    volatile tt_l1_ptr uint32_t* zfree = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(moe_block_args::arg(rse + 7));
    const uint32_t zneed = moe_block_args::arg(rse + 8), zexp0 = moe_block_args::arg(rse + 9);
    const uint32_t zexp = zexp0 ? zexp0 - (row_chunks - x_gather_num_live_chunks)
                                : 0;  // (the credits a call expects: one a live chunk on)
    const uint32_t zcred = moe_block_args::arg(rse + 6);
    const uint32_t NZF = zneed && row_chunks > 1 ? x_gather_num_live_chunks - 1 : 0;
    // (the router's buffers over this chip's arrival zones / LATZ: every matmul core counts its router phase over on
    // my "router done" word (rneed of them) - only then the peer's chunk 0 credit and the latent's "started" go)
    volatile tt_l1_ptr uint32_t* rdone = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(moe_block_args::arg(rse + 10));
    const uint32_t rw_ = moe_block_args::arg(rse + 11), rneed = rw_ & 0xFFFF;
    const bool rzone = (rw_ >> 16) & 1, rlat = (rw_ >> 17) & 1;  // (what waits for it: chunk 0's credit, "started")
    bool rtold = rneed == 0;
    uint32_t zfw = 0;
    LogitsStream ls;
    ls.init(ls0);
    const uint32_t la0 = ls0 + LogitsStream::len(ls0), lnu = moe_block_args::arg(la0);
    // (the latent's end-to-end fence: my fence word, the "started" count my sends need (one a destination chip), then
    // the chips I tell "started" - their latent ports toward me: (chip | mesh << 16, x | y << 8))
    const uint32_t lf0 = la0 + 2 + 2 * lnu + 2 * moe_block_args::arg(la0 + 1 + 2 * lnu) + 5;
    const uint32_t lfence = moe_block_args::arg(lf0), lneed = moe_block_args::arg(lf0 + 1),
                   lns = moe_block_args::arg(lf0 + 2);
    // (the logits' end-to-end fence: my fence word, the "started" count my logits need (one an owner chip I send to),
    // then the chips whose logits come to me through this port's direction and link: route, their port)
    const uint32_t lg0 = lf0 + 3 + 2 * lns;
    const uint32_t lgfence = moe_block_args::arg(lg0), lgneed = moe_block_args::arg(lg0 + 1),
                   lgns = moe_block_args::arg(lg0 + 2);
    volatile tt_l1_ptr uint32_t* lgf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lgfence);
    const uint32_t ra0 = lg0 + 3 + 2 * lgns;
    RoutingExchangeSend ra;
    ra.init(ra0);
    a = ra0 + RoutingExchangeSend::len(ra0);
    if (n > 0 || send_ready || rnu > 0 || ls.nu > 0 || lnu > 0 || lns > 0 || lgns > 0 || !ra.done || NZF ||
        (rzone && zneed)) {
        // (the fabric connection's three values - eth channel, teardown and buffer-index semaphores - from the block)
        auto conn = WorkerToFabricEdmSender::build_from_eth_channel<ProgrammableCoreType::TENSIX>(
            moe_block_args::arg(a), moe_block_args::arg(a + 1), moe_block_args::arg(a + 2));
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[GROUP + 1 + 3];
        for (uint32_t h = 0; h <= GROUP; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            (void)fabric_set_unicast_route(hdrs[h], chip, mesh);
        }
        volatile tt_l1_ptr PACKET_HEADER_TYPE* rahdr =
            PacketHeaderPool::allocate_header();  // (routing: routed a packet)
        volatile tt_l1_ptr PACKET_HEADER_TYPE* lshdr = PacketHeaderPool::allocate_header();  // (logits: routed a unit)
        for (uint32_t un = 0; un < lnu && un < 3; ++un) {  // (the latent's destinations: routed a unit)
            const uint32_t w0 = moe_block_args::arg(la0 + 1 + 2 * un);
            hdrs[GROUP + 1 + un] = PacketHeaderPool::allocate_header();
            (void)fabric_set_unicast_route(
                hdrs[GROUP + 1 + un],
                static_cast<uint16_t>(w0 >> 16),
                static_cast<uint16_t>(moe_block_args::arg(la0 + 2 + 2 * un)));
        }
        conn.open();
        if (send_ready) {  // fence: I started this call - the neighbour's port facing me may write into my L1
            hdrs[GROUP]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, ready), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
        }
        volatile tt_l1_ptr PACKET_HEADER_TYPE* fh = lns > 0 ? PacketHeaderPool::allocate_header() : nullptr;
        auto lstarted = [&]() {  // (every chip sending me latent may write my LATZ - its latent port's fence word)
            for (uint32_t i = 0; i < lns; ++i) {
                const uint32_t w = moe_block_args::arg(lf0 + 3 + 2 * i), xy = moe_block_args::arg(lf0 + 4 + 2 * i);
                (void)fabric_set_unicast_route(fh, static_cast<uint16_t>(w & 0xFFFF), static_cast<uint16_t>(w >> 16));
                fh->to_noc_unicast_atomic_inc(
                    NocUnicastAtomicIncCommandHeader{get_noc_addr(xy & 0xFF, (xy >> 8) & 0xFF, lfence), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(fh), sizeof(PACKET_HEADER_TYPE));
            }
        };
        if (!rlat) {
            lstarted();  // (I started this call: LATZ is free - no router buffers over it)
        }
        if (lgns > 0) {  // (I started this call: every chip whose logits come to me through this port may write my
            // router cores' arrival zone - its logits port's fence word; my neighbour's ready only fences the first
            // hop)
            volatile tt_l1_ptr PACKET_HEADER_TYPE* gh = lshdr;  // (free until the first logits packet: re-routed each)
            for (uint32_t i = 0; i < lgns; ++i) {
                const uint32_t w = moe_block_args::arg(lg0 + 3 + 2 * i), xy = moe_block_args::arg(lg0 + 4 + 2 * i);
                (void)fabric_set_unicast_route(gh, static_cast<uint16_t>(w & 0xFFFF), static_cast<uint16_t>(w >> 16));
                gh->to_noc_unicast_atomic_inc(
                    NocUnicastAtomicIncCommandHeader{get_noc_addr(xy & 0xFF, (xy >> 8) & 0xFF, lgfence), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(gh), sizeof(PACKET_HEADER_TYPE));
            }
        }
        bool lgok = lgneed == 0;  // (every owner I send logits to started this call)
#ifdef ROUTER_LOGITS_ITERATIONS
        // router iterations: the iteration whose logits I send, and the "started" rounds told past the first - round k
        // once my chip's router cores are all done with iteration k - 1 (their count on my "rit" word, +16)
        uint32_t lit = 0, ltok = 0;
        volatile tt_l1_ptr uint32_t* rfree = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ls.rit + 16);
        auto lgtok = [&]() {
            if (ltok + 1 < ROUTER_LOGITS_ITERATIONS && lgns > 0) {
                invalidate_l1_cache();
                if (*rfree >= ls.rbi * (ltok + 1)) {
                    noc_async_writes_flushed();  // (the logits header's last packet left: re-routed here)
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* gh = lshdr;
                    for (uint32_t i = 0; i < lgns; ++i) {
                        const uint32_t w = moe_block_args::arg(lg0 + 3 + 2 * i),
                                       xy = moe_block_args::arg(lg0 + 4 + 2 * i);
                        (void)fabric_set_unicast_route(
                            gh, static_cast<uint16_t>(w & 0xFFFF), static_cast<uint16_t>(w >> 16));
                        gh->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                            get_noc_addr(xy & 0xFF, (xy >> 8) & 0xFF, lgfence), 1, true});
                        conn.wait_for_empty_write_slot();
                        conn.send_payload_flush_blocking_from_address(
                            reinterpret_cast<uint32_t>(gh), sizeof(PACKET_HEADER_TYPE));
                    }
                    ++ltok;
                }
            } else if (ltok + 1 < ROUTER_LOGITS_ITERATIONS) {
                ++ltok;  // (no chip sends me logits through this port)
            }
        };
#endif
        if (wait_ready) {
            DeviceZoneScopedN("SE-RDYWAIT");  // (its end: the neighbour's port started - launch skew + a hop)
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), 1);
            noc_semaphore_inc(get_noc_addr(ready), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t arr_noc = get_noc_addr(px, py, arrival);
        uint32_t h = 0, unflushed = 0;
        auto rpoll = [&]() {  // (every matmul core's router phase is over: the zones, LATZ free - also between the x
            // gather's packets: the upstream's chunk 0 reduce-scatter waits for it)
            if (!rtold && *rdone >= rneed) {
                if (rzone && zneed) {  // (chunk 0's credit)
                    hdrs[GROUP]->to_noc_unicast_atomic_inc(
                        NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, zcred), 1, true});
                    conn.wait_for_empty_write_slot();
                    conn.send_payload_flush_blocking_from_address(
                        reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
                }
                if (rlat) {
                    lstarted();
                }
                rtold = true;
            }
        };
        auto zpoll = [&]() {  // (my chip's cores read a chunk's zones: credit the peer's sends, a blocking header -
            // the gather's fence / watcher header, never in flight across this)
            rpoll();
            if (rtold && zfw < NZF && *zfree >= (zfw + 1) * zneed) {
                hdrs[GROUP]->to_noc_unicast_atomic_inc(
                    NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, zcred), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
                ++zfw;
            }
        };
        auto lpoll = [&]() {
            zpoll();
#ifdef ROUTER_LOGITS_ITERATIONS
            lgtok();
            if (!lgok) {
                invalidate_l1_cache();
                lgok = *lgf >= lgneed * (lit + 1);  // (every owner I send to: its router cores free for iteration lit)
            }
            if (lgok) {
                ls.poll(conn, lshdr, false);
                if (ls.done() &&
                    lit + 1 < ROUTER_LOGITS_ITERATIONS) {  // (iteration lit's sub-blocks all left: its slots are free)
                    noc_async_writes_flushed();
                    if (ls.active()) {
                        noc_semaphore_inc(get_noc_addr(ls.c0 & 0xFFFF, ls.c0 >> 16, ls.rit), 1);
                    }
                    ls.next_iteration();
                    ++lit;
                    lgok = lgneed == 0;
                }
            }
#else
            if (!lgok) {
                invalidate_l1_cache();
                lgok = *lgf >= lgneed;
            }
            if (lgok) {
                ls.poll(conn, lshdr, false);
            }
#endif
            ra.poll(
                conn,
                rahdr,
                false,
                ROUTING_EXCHANGE_NUM_CHIPS,
                ROUTING_EXCHANGE_PLANNER_INSTANCES,
                ROUTING_EXCHANGE_EXPERTS_PER_CHIP,
                ROUTING_EXCHANGE_MY_CHIP,
                ROUTING_EXCHANGE_HEADER_BYTES,
                ROUTING_EXCHANGE_EXPERT_COUNT_BLOCK_BYTES,
                ROUTING_EXCHANGE_INSTANCE_BLOCK_BYTES,
                ROUTING_EXCHANGE_MAX_PACKET_PAYLOAD_BYTES);
        };
        for (uint32_t m = 0; m < x_gather_num_live_chunks && n > 0; ++m) {
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t q = moe_block_args::arg(ent + 3 * k), part = moe_block_args::arg(ent + 3 * k + 1);
                const uint32_t npk = moe_block_args::arg(ent + 3 * k + 2);
                uint32_t sent = 0;
                const auto mk = x_gather_mask(m);  // (chunk m's real row tiles)
                walk(m, q, part, v, [&](uint32_t R, uint32_t j, uint32_t, uint32_t kk0, uint32_t np, uint32_t) {
                    if (row_chunks > 1 && !rtold) {  // (chunks: the upstream's chunk 0 reduce-scatter comes before
                        // my gather of the last chunk's x ends; one chunk: after it - no blocking header between
                        // the x packets, it cost K3 640 ~3 us of gather)
                        invalidate_l1_cache();
                        rpoll();
                    }
#if ROUTER_LOGITS_FIRST
                    lpoll();  // (GLM-type: the logits ahead of the x gather)
                    while (!cb_pages_available_at_front(0, RP * (unflushed + 1))) lpoll();
#endif
                    cb_wait_front(0, RP * (unflushed + 1));
                    const uint32_t src = get_read_ptr(0) + unflushed * RP * moe_block_tiles::kBf16TileBytes;
                    const bool inc = (++sent % INC) == 0 || sent == npk;
                    if ((mk >> (R - m * CR)) & 1) {
                        const Dst d = dst_of(T, xown, m, R, j, kk0);
                        const uint64_t dst = get_noc_addr(d.x, d.y, d.addr);
                        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                        h = h + 1 == GROUP ? 0 : h + 1;
                        if (inc) {
                            hdr->to_noc_fused_unicast_write_atomic_inc(
                                NocUnicastAtomicIncFusedCommandHeader{dst, arr_noc, 1, true},
                                np * moe_block_tiles::kBf16TileBytes);
                        } else {
                            hdr->to_noc_unicast_write(
                                NocUnicastCommandHeader{dst}, np * moe_block_tiles::kBf16TileBytes);
                        }
                        conn.wait_for_empty_write_slot();
                        conn.send_current_slot_non_blocking(
                            src, np * moe_block_tiles::kBf16TileBytes, reinterpret_cast<uint32_t>(hdr));
                    } else if (inc) {  // (a padding row tile: no x - only the count its packets carry)
                        hdrs[GROUP]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{arr_noc, 1, true});
                        conn.wait_for_empty_write_slot();
                        conn.send_payload_flush_blocking_from_address(
                            reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
                    }
                    if (++unflushed == GROUP || sent == npk) {
                        noc_async_writes_flushed();
                        cb_pop_front(0, RP * GROUP);
                        unflushed = 0;
                    }
                });
                // this chunk's shard k all sent: the neighbour's watcher (behind its packets on this link)
                hdrs[GROUP]->to_noc_unicast_atomic_inc(
                    NocUnicastAtomicIncCommandHeader{get_noc_addr(wx, wy, watch), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
            }
        }
        lpoll();
        uint32_t rpos = 0;  // (as the reader's pos)
        for (uint32_t m = 0; m < x_gather_num_live_chunks; ++m) {
            latent_all_to_all_send(la0, conn, hdrs, lpoll, rpos, m);
            reduce_scatter_send(rs0, conn, hdrs, lpoll, rpos, m);
        }
#if TOKEN_LOOPS > 1
        if (rpos % GROUP) {  // (token loops: the reader padded its last group - the next loop's gather starts aligned)
            cb_wait_front(0, (GROUP - rpos % GROUP) * RP);
            cb_pop_front(0, (GROUP - rpos % GROUP) * RP);
        }
#endif
        while (zfw < NZF || !rtold) {  // (the chunks' credits still owed; the router-done credit / "started")
            invalidate_l1_cache();
            lpoll();
        }
#ifdef ROUTER_LOGITS_ITERATIONS
        while (lit + 1 < ROUTER_LOGITS_ITERATIONS || !ls.done() ||
               ltok + 1 < ROUTER_LOGITS_ITERATIONS) {  // (every iteration's logits; every "started")
            lpoll();
        }
#else
        if (lgneed) {
            noc_semaphore_wait_min(lgf, lgneed);
        }
        ls.poll(conn, lshdr, true);  // (whatever is left of the logits)
#endif
        ra.poll(
            conn,
            rahdr,
            true,
            ROUTING_EXCHANGE_NUM_CHIPS,
            ROUTING_EXCHANGE_PLANNER_INSTANCES,
            ROUTING_EXCHANGE_EXPERTS_PER_CHIP,
            ROUTING_EXCHANGE_MY_CHIP,
            ROUTING_EXCHANGE_HEADER_BYTES,
            ROUTING_EXCHANGE_EXPERT_COUNT_BLOCK_BYTES,
            ROUTING_EXCHANGE_INSTANCE_BLOCK_BYTES,
            ROUTING_EXCHANGE_MAX_PACKET_PAYLOAD_BYTES);  // (the routing)
        noc_async_writes_flushed();
        if (NZF) {
            noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(zfree)), 0u - NZF * zneed);  // (re-arm)
        }
        if (zexp) {  // (every credit of the call came: they were waited for, the last before the last chunk)
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zcred), zexp);
            noc_semaphore_inc(get_noc_addr(zcred), 0u - zexp);
        }
        ls.rearm();
        if (lneed) {
            noc_semaphore_inc(get_noc_addr(lfence), 0u - lneed);  // (re-arm: a next call's early "started" stays)
        }
#ifdef ROUTER_LOGITS_ITERATIONS
        constexpr uint32_t LG_ROUNDS =
            ROUTER_LOGITS_ITERATIONS;  // (a "started" a router iteration from each owner I send to)
#else
        constexpr uint32_t LG_ROUNDS = 1;
#endif
        if (lgneed) {
            noc_semaphore_inc(get_noc_addr(lgfence), 0u - lgneed * LG_ROUNDS);  // (re-arm, as the latent's)
        }
        noc_async_atomic_barrier();
        conn.close();
    }
#ifdef ROUTER_LOGITS_ITERATIONS
    {  // (re-arm: my chip's router cores counted every iteration but the last here - also on a port with nothing to
       // send)
        volatile tt_l1_ptr uint32_t* rf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ls.rit + 16);
        noc_semaphore_wait_min(rf, ls.rbi * (ROUTER_LOGITS_ITERATIONS - 1));
        noc_semaphore_inc(get_noc_addr(ls.rit + 16), 0u - ls.rbi * (ROUTER_LOGITS_ITERATIONS - 1));
        noc_async_atomic_barrier();
    }
#endif
    if (rneed) {  // (re-arm: every matmul core counted its router phase here - also on a port with nothing to send)
        noc_semaphore_wait_min(rdone, rneed);
        noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(rdone)), 0u - rneed);
        noc_async_atomic_barrier();
    }
    if (expect_in > 0) {  // everything from upstream landed; re-arm for the next call
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival);
        const uint32_t ein = expect_in / row_chunks * x_gather_num_live_chunks;  // (as many a chunk: the live ones)
        noc_semaphore_wait_min(arrived, ein);
        noc_semaphore_inc(get_noc_addr(arrival), 0u - ein);
        noc_async_atomic_barrier();
    }
}

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    for (uint32_t loop = 0; loop < token_loop::LOOPS; ++loop) {
        token_loop::begin(loop);
        process_token_loop(loop);
        token_loop::end(loop);
    }
}
