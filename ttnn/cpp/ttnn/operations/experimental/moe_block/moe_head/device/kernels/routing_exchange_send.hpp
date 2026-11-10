// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The routing exchange's send (ROUTING_EXCHANGE_SEND).
// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_accessors.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"

// one packet straight to its destination, the fabric routing every hop (Y then X, its turn included): into the
// destination's DRAM slot page par N + src, chunks of MAXP, the last a fused increment of its arrival counter (par: c1,
// else c0) on planner core 0. No relay on the way: no chip re-reads and re-sends another chip's packet, so a packet's
// parity is always its sender's own call count.
template <typename C, typename H>
FORCE_INLINE void routing_exchange_hop(
    C& conn,
    H hdr,
    uint32_t src_l1,
    uint32_t bytes,
    uint32_t chip,
    uint32_t par,
    uint32_t srcid,
    uint32_t PCAP,
    uint32_t sl,
    uint32_t c0,
    uint32_t c1,
    uint32_t px,
    uint32_t py,
    uint32_t mesh,
    uint32_t num_chips,
    uint32_t MAXP) {
    // fabric router writes on NOC1: resolve the DRAM endpoint for its NoC (SYS-1419)
    const uint64_t dst = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
        TensorAccessor(routing_accessors::slots, sl, PCAP), par * num_chips + srcid);
    const uint64_t inc = get_noc_addr(px, py, par ? c1 : c0);
    for (uint32_t off = 0; off < bytes; off += MAXP) {
        const uint32_t sz = bytes - off < MAXP ? bytes - off : MAXP;
        noc_async_writes_flushed();  // (the header's previous packet has left)
        (void)fabric_set_unicast_route(hdr, static_cast<uint16_t>(chip), static_cast<uint16_t>(mesh));
        if (off + sz >= bytes) {
            hdr->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{dst + off, inc, 1, true}, sz);
        } else {
            hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst + off}, sz);
        }
        conn.wait_for_empty_write_slot();
        conn.send_current_slot_non_blocking(src_l1 + off, sz, reinterpret_cast<uint32_t>(hdr));
    }
}
FORCE_INLINE uint32_t routing_exchange_bytes(uint32_t pb, uint32_t planner_instances, uint32_t header_bytes) {
    const uint32_t* nq = reinterpret_cast<const uint32_t*>(pb);
    uint32_t n = 0;
    for (uint32_t q = 0; q < planner_instances; ++q) {
        n += nq[q];
    }
    return (header_bytes + 4 * n + 63) / 64 * 64;
}
struct RoutingExchangeSend {
    uint32_t r0, nd, nr, self_, B, PCAP, sl, c0, c1, px, py, calls, ready, mesh, done;
    void init(uint32_t a) {
        r0 = a;
        nd = moe_block_args::arg(a);
        if (!nd) {
            done = 1;
            return;
        }
        nr = moe_block_args::arg(a + 1);
        self_ = moe_block_args::arg(a + 2);
        B = moe_block_args::arg(a + 3);
        PCAP = moe_block_args::arg(a + 4);
        sl = moe_block_args::arg(a + 5);
        c0 = moe_block_args::arg(a + 6);
        c1 = moe_block_args::arg(a + 7);
        px = moe_block_args::arg(a + 8);
        py = moe_block_args::arg(a + 9);
        calls = moe_block_args::arg(a + 10);
        ready = moe_block_args::arg(a + 11);
        mesh = moe_block_args::arg(a + 12);
        done = 0;
    }
    static uint32_t len(uint32_t a) {
        const uint32_t n = moe_block_args::arg(a);
        return n ? 13 + n : 1;  // (then a word a destination: its chip id)
    }
    template <typename C, typename H>
    void poll(
        C& conn,
        H hdr,
        bool blocking,
        uint32_t num_chips,
        uint32_t planner_instances,
        uint32_t experts_per_chip,
        uint32_t my_chip,
        uint32_t header_bytes,
        uint32_t expert_count_block_bytes,
        uint32_t instance_block_bytes,
        uint32_t MAXP) {
        if (done) {
            return;
        }
        volatile tt_l1_ptr uint32_t* rd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
        invalidate_l1_cache();
        if (*rd < planner_instances) {
            if (!blocking) {
                return;
            }
            noc_semaphore_wait_min(rd, planner_instances);
        }
        DeviceZoneScopedN("SE-RASEND");
        noc_semaphore_inc(get_noc_addr(ready), 0u - planner_instances);
        noc_async_atomic_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* cl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(calls);
        const uint32_t par = cl[0] & 1;  // (every port core and planner core counts its calls: they stay in step)
        cl[0] = cl[0] + 1;
        for (uint32_t j = 0; j < nd; ++j) {
            const uint32_t pb = B + j * PCAP;
            const uint32_t* nq = reinterpret_cast<const uint32_t*>(pb);
            // (the per-expert totals: the planners' - the destination's owner instance wrote them into the header)
            reinterpret_cast<uint32_t*>(pb)[14] = par;  // (the call's parity and the source: a hang dump reads them)
            reinterpret_cast<uint32_t*>(pb)[15] = my_chip;
            (void)nq;
            const uint32_t bytes = routing_exchange_bytes(pb, planner_instances, header_bytes);
            if (j < nr) {
                routing_exchange_hop(
                    conn,
                    hdr,
                    pb,
                    bytes,
                    moe_block_args::arg(r0 + 13 + j),
                    par,
                    my_chip,
                    PCAP,
                    sl,
                    c0,
                    c1,
                    px,
                    py,
                    mesh,
                    num_chips,
                    MAXP);
            } else {  // (SELF: my own chip's packet)
                noc_async_write(
                    pb,
                    TensorAccessor(routing_accessors::slots, sl, PCAP).get_noc_addr(par * num_chips + my_chip),
                    bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(px, py, par ? c1 : c0), 1);
            }
        }
        noc_async_writes_flushed();
        done = 1;
    }
};
