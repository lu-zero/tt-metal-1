// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// routing_exchange_y_sender: the routing exchange's send in the first Y direction
// Runs on: the watcher core (one a chip), RISCV_0 (BRISC), NoC 0 (RISCV_0_default)
// Responsibilities:
//  - waits for this chip's planner instances (ready word), then sends their routing packets from L1 over the fabric
//    straight into each destination chip's DRAM slot, with a fused increment of its arrival counter on planner core 0;
//    this chip's own packet: a NoC write into its own slot
//  - Y hello (launch-skew measurement only): one fabric increment to the Y neighbour's watcher core; waits for the
//    hellos that land here (a runtime arg: 0, 1 or 2) and re-arms the word
// Note: the routing planner reads this chip's own packet from the sender's L1, not from the DRAM slot written on NoC 0
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/token_loop.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_exchange_send.hpp"

// One token loop of this kernel's work: kernel_main below runs it once per loop (token_loop.hpp).
void process_token_loop(uint32_t loop) {
    // (token loops: every loop allocates its packet headers anew - the last loop's packets were flushed by its
    // connection's close)
    PacketHeaderPool::reset();
    RoutingExchangeSend rs;
    rs.init(0);
    size_t a = RoutingExchangeSend::len(0);
    const uint32_t remote = moe_block_args::arg(a++);
    if (remote) {
        const uint32_t hello = moe_block_args::arg(a++), pc = moe_block_args::arg(a++), pm = moe_block_args::arg(a++);
        // (the fabric connection's three values - eth channel, teardown and buffer-index semaphores - from the block)
        auto conn = WorkerToFabricEdmSender::build_from_eth_channel<ProgrammableCoreType::TENSIX>(
            moe_block_args::arg(a), moe_block_args::arg(a + 1), moe_block_args::arg(a + 2));
        a += 3;
        // the hellos that land here a call (the rows whose first Y direction points at this row): one on a ring; on a
        // line none at the near end, two at the row before the far end
        const uint32_t hellos = moe_block_args::arg(a++);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
        conn.open();
        // measurement only: "started" to the Y neighbour's watcher; the neighbour's landing here is stamped when seen
        // (between the sends' polls, never waited on before they are done) - the rows' launch skew
        (void)fabric_set_unicast_route(hdr, static_cast<uint16_t>(pc), static_cast<uint16_t>(pm));
        hdr->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{get_noc_addr(my_x[0], my_y[0], hello), 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
        volatile tt_l1_ptr uint32_t* hw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hello);
        bool seen = hellos == 0;
        while (!rs.done) {
            invalidate_l1_cache();
            if (!seen && *hw) {
                DeviceTimestampedData("SE-YHELLO", 1);
                seen = true;
            }
            rs.poll(
                conn,
                hdr,
                false,
                ROUTING_EXCHANGE_NUM_CHIPS,
                ROUTING_EXCHANGE_PLANNER_INSTANCES,
                ROUTING_EXCHANGE_EXPERTS_PER_CHIP,
                ROUTING_EXCHANGE_MY_CHIP,
                ROUTING_EXCHANGE_HEADER_BYTES,
                ROUTING_EXCHANGE_EXPERT_COUNT_BLOCK_BYTES,
                ROUTING_EXCHANGE_INSTANCE_BLOCK_BYTES,
                ROUTING_EXCHANGE_MAX_PACKET_PAYLOAD_BYTES);
        }
        if (hellos) {
            noc_semaphore_wait_min(hw, hellos);
            if (!seen) {
                DeviceTimestampedData("SE-YHELLO", 0);  // (late: after the sends)
            }
            noc_semaphore_inc(get_noc_addr(hello), 0u - hellos);  // (re-arm)
            noc_async_atomic_barrier();
        }
        rs.poll(
            conn,
            hdr,
            true,
            ROUTING_EXCHANGE_NUM_CHIPS,
            ROUTING_EXCHANGE_PLANNER_INSTANCES,
            ROUTING_EXCHANGE_EXPERTS_PER_CHIP,
            ROUTING_EXCHANGE_MY_CHIP,
            ROUTING_EXCHANGE_HEADER_BYTES,
            ROUTING_EXCHANGE_EXPERT_COUNT_BLOCK_BYTES,
            ROUTING_EXCHANGE_INSTANCE_BLOCK_BYTES,
            ROUTING_EXCHANGE_MAX_PACKET_PAYLOAD_BYTES);
        conn.close();
    } else {
        struct NoConn {
            void wait_for_empty_write_slot() {}
            void send_current_slot_non_blocking(uint32_t, uint32_t, uint32_t) {}
            void send_payload_flush_blocking_from_address(uint32_t, uint32_t) {}
        } nc;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = nullptr;
        rs.poll(
            nc,
            hdr,
            true,
            ROUTING_EXCHANGE_NUM_CHIPS,
            ROUTING_EXCHANGE_PLANNER_INSTANCES,
            ROUTING_EXCHANGE_EXPERTS_PER_CHIP,
            ROUTING_EXCHANGE_MY_CHIP,
            ROUTING_EXCHANGE_HEADER_BYTES,
            ROUTING_EXCHANGE_EXPERT_COUNT_BLOCK_BYTES,
            ROUTING_EXCHANGE_INSTANCE_BLOCK_BYTES,
            ROUTING_EXCHANGE_MAX_PACKET_PAYLOAD_BYTES);
    }
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
