// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// x_gather_watcher: tells the matmul cores when each gathered x shard has fully landed on this chip
// Runs on: the watcher core (one a chip), RISCV_1 (NCRISC), NoC 1 (RISCV_1_default)
// Responsibilities:
//  - polls its L1 counters, which the upstream fabric_port_sender cores increment over the fabric (a shard a stream)
//  - once every stream carrying shard k of chunk m counted it: +1 on XREADY on every matmul core
//    (shared_expert_row_dataflow waits on it); re-arms the counters
//  - with a second Y direction (WATCHER_ROUTING_ARGS_OFFSET): that direction's routing exchange send, once the planner
//    instances are ready, over the fabric into the destination chips' DRAM slots (routing_exchange_send.hpp)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/token_loop.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#ifdef SKIP_PADDING_CHUNKS
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/padded_rows.hpp"
#endif
#ifdef WATCHER_ROUTING_ARGS_OFFSET
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_exchange_send.hpp"

#endif
// chunk m's shard of rank k (k = 1 .. G - 1 in the engine's order) is in once each stream carrying it counted it
// (a stream counts every shard it sends: m entries + its index + 1): then XREADY + 1 on every engine core
// (XREADY = m (G - 1) + k). args: chunks, the XREADY id, ranks G - 1, engine cores nc, counters ncnt; (counter address,
// final) x ncnt; cores (x, y) x nc; per rank: nreq, (counter address, the entry's index, the stream's entries a chunk)
// One token loop of this kernel's work: kernel_main below runs it once per loop (token_loop.hpp).
void process_token_loop(uint32_t loop) {
    // (token loops: every loop allocates its packet headers anew - the last loop's packets were flushed by its
    // connection's close)
#ifdef WATCHER_ROUTING_ARGS_OFFSET
    PacketHeaderPool::reset();
#endif
    size_t a = 0;
    const uint32_t nch0 = moe_block_args::arg(a++), sem = get_semaphore(moe_block_args::arg(a++));
#ifdef SKIP_PADDING_CHUNKS
    // (a padded forward: the row chunks up to the last with a real row - the counters count those)
    const uint32_t nch = padded_live_chunks(
        [](uint32_t i) { return moe_block_args::common(i) + (i == 5 ? loop * TOKEN_LOOP_PADDED_ROW_STRIDE : 0u); },
        0,
        (moe_block_args::common(10) + 63) & ~63u);  // (loop l's first row: TOKEN_LOOP_PADDED_ROW_STRIDE rows on)
#else
    const uint32_t nch = nch0;
#endif
    const uint32_t nk = moe_block_args::arg(a++), nc = moe_block_args::arg(a++), ncnt = moe_block_args::arg(a++);
    const uint32_t cnt0 = a, c0 = cnt0 + 2 * ncnt, r0 = c0 + 2 * nc;
    for (uint32_t m = 0; m < nch; ++m) {
        uint32_t b = r0;
        for (uint32_t k = 0; k < nk; ++k) {
            const uint32_t nreq = moe_block_args::arg(b++);
            for (uint32_t i = 0; i < nreq; ++i, b += 3) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(moe_block_args::arg(b)),
                    m * moe_block_args::arg(b + 2) + moe_block_args::arg(b + 1) + 1);
            }
            for (uint32_t c = 0; c < nc; ++c) {
                noc_semaphore_inc(
                    get_noc_addr(moe_block_args::arg(c0 + 2 * c), moe_block_args::arg(c0 + 2 * c + 1), sem), 1);
            }
        }
    }
    for (uint32_t i = 0; i < ncnt; ++i) {  // (re-arm)
        noc_semaphore_inc(
            get_noc_addr(moe_block_args::arg(cnt0 + 2 * i)), 0u - moe_block_args::arg(cnt0 + 2 * i + 1) / nch0 * nch);
    }
    noc_async_atomic_barrier();
#ifdef WATCHER_ROUTING_ARGS_OFFSET
    {  // (the routing exchange's second Y direction: after the watch)
        RoutingExchangeSend rs;
        rs.init(WATCHER_ROUTING_ARGS_OFFSET);
        size_t a2 = WATCHER_ROUTING_ARGS_OFFSET + RoutingExchangeSend::len(WATCHER_ROUTING_ARGS_OFFSET);
        if (moe_block_args::arg(a2++)) {
            a2 += 3;  // (the Y hello's args: the BRISC's direction only)
            // (the fabric connection's three values from the block)
            auto conn = WorkerToFabricEdmSender::build_from_eth_channel<ProgrammableCoreType::TENSIX>(
                moe_block_args::arg(a2), moe_block_args::arg(a2 + 1), moe_block_args::arg(a2 + 2));
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
            conn.open();
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
        }
        noc_async_atomic_barrier();
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
