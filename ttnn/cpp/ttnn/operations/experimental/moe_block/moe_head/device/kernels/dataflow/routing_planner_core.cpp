// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// routing_planner_core: the routing planner on a planner-only core (token loops)
// Runs on: planner-only cores, RISCV_0 (BRISC), NoC 1; or RISCV_1 (NCRISC), NoC 0 (the core's second planner instance)
// Responsibilities:
//  - waits for every router core's rows (ROUTING_EXCHANGE_ROUTER_ROW_TILES increments of its rows word by
//    router_output_writer), then re-arms the word
//  - with an instance here (its region arg non-zero): routing_planner::plan - reads the routing rows and the
//    exchange's DRAM slots, writes counts / regions / token index / table / plan to DRAM
// Note: fabric-written DRAM is read on NoC 1, the fabric routers' NoC (a read through the other NoC's DRAM endpoint
//  can return stale data); planner_slot currently gives the NCRISC (NoC 0) instance no region: it only waits

// A planner-only core's kernel (token loops: the routing planner's instances past the router cores, BRISC and NCRISC):
// once every router core's rows are written (ROUTING_EXCHANGE_ROUTER_ROW_TILES increments of my rows word),
// routing_planner::plan. Args: my rows word, my instance's region (0: none here), then routing_planner::plan's.
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/token_loop.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/router_common_args.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_planner_select.hpp"

// One token loop of this kernel's work: kernel_main below runs it once per loop (token_loop.hpp).
void process_token_loop(uint32_t loop) {
    const uint32_t rows = moe_block_args::arg(0), B = moe_block_args::arg(1);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows), ROUTING_EXCHANGE_ROUTER_ROW_TILES);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows) = 0;  // (re-arm: all increments landed)
    if (B) {
        DeviceZoneScopedN("SE-RPLAN");
        ROUTING_PLANNER_RUN(2, B);
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
