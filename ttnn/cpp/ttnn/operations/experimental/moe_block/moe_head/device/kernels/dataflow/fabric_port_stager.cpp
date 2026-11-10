// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fabric_port_stager: stages what a fabric port core sends into CB 0, for fabric_port_sender on the same core
// Runs on: port cores (one per X direction and fabric link), RISCV_0 (BRISC), NoC 1 (RISCV_1_default)
// Responsibilities:
//  - x all-gather: NoC-reads this chip's own x shard from the matmul cores' x-own L1 (once every matmul core counted
//    its load) and, relaying, the upstream shards the fabric landed there (arrival counter) into CB 0
//  - reduce-scatter: per matmul core's hand-off (mailbox word) reads its sub-block from its FINB into CB 0, then +1 on
//    its FINB-free word; latent all-to-all: the cores' latent rows from LAT the same way
//  - skips padding row tiles; pushes whole groups (GROUP x RP tiles) so CB 0 stays in step with the sender
// Note: in this op the reading side runs on BRISC (RISCV_0) and the sending side on NCRISC (RISCV_1), opposite to
//  tt-metal's Reader/Writer config convention, on purpose; the fabric-landed shards are read on NoC 1, the fabric
//  routers' NoC
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/token_loop.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/x_gather_compile_args.hpp"
constexpr auto x_args = TensorAccessorArgs<14>();
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/x_gather_schedule.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/latent_all_to_all_reader.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/reduce_scatter_reader.hpp"

// One token loop of this kernel's work: kernel_main below runs it once per loop (token_loop.hpp).
void process_token_loop(uint32_t loop) {
    size_t a = 0;
#ifdef SKIP_PADDING_CHUNKS
    x_gather_find_live_chunks();  // (a padded forward: the row chunks up to the last with a real row)
#endif
    const uint32_t x_addr = moe_block_args::arg(a++), xown = moe_block_args::arg(a++), v = moe_block_args::arg(a++);
    const uint32_t arrival = moe_block_args::arg(a++), n = moe_block_args::arg(a++), up_ipc = moe_block_args::arg(a++);
    const uint32_t ent = a, T = ent + 4 * n;  // entries (q, part, up_base within a chunk, packets a chunk); tables
    // (after the tables: the engine's "own shard loaded" counter here and its full count)
    const uint32_t xl0 = T + G + S0 + S1 + 3 * engine_columns + engine_rows;
    volatile tt_l1_ptr uint32_t* xloaded = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(moe_block_args::arg(xl0));
    const uint32_t xl_need = moe_block_args::arg(xl0 + 1);
    bool xl = false;
    const auto x = TensorAccessor(x_args, x_addr, moe_block_tiles::kBf16TileBytes);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival);
    uint32_t batch = 0, wptr = 0;
    for (uint32_t m = 0; m < x_gather_num_live_chunks; ++m) {
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t q = moe_block_args::arg(ent + 4 * k), part = moe_block_args::arg(ent + 4 * k + 1);
            const uint32_t up_base = moe_block_args::arg(ent + 4 * k + 2), npk = moe_block_args::arg(ent + 4 * k + 3);
            uint32_t done = 0;
            const auto mk = x_gather_mask(m);  // (chunk m's real row tiles)
            walk(m, q, part, v, [&](uint32_t R, uint32_t j, uint32_t kj, uint32_t kk0, uint32_t np, uint32_t idx) {
                if (batch == 0) {
                    cb_reserve_back(0, RP * GROUP);
                    wptr = get_write_ptr(0);
                }
                const uint32_t dst = wptr + batch * RP * moe_block_tiles::kBf16TileBytes;
                if (k == 0) {  // my own shard: from the engine's x-own L1 once every engine core loaded it (no DRAM
                               // here: the DRAM is the engine's weight load's)
                    if (!xl) {
                        noc_semaphore_wait_min(xloaded, xl_need);
                        xl = true;
                    }
                    if ((mk >> (R - m * CR)) & 1) {  // (a padding row tile: nothing to read)
                        const Dst d = dst_of(T, xown, m, R, j, kk0);
                        noc_async_read(get_noc_addr(d.x, d.y, d.addr), dst, np * moe_block_tiles::kBf16TileBytes);
                    }
                } else {  // relay: upstream's packet idx of (chunk m, its entry k - 1) landed here
                    noc_semaphore_wait_min(arrived, m * up_ipc + up_base + idx / INC + 1);
                    if ((mk >> (R - m * CR)) & 1) {
                        const Dst d = dst_of(T, xown, m, R, j, kk0);
                        noc_async_read(get_noc_addr(d.x, d.y, d.addr), dst, np * moe_block_tiles::kBf16TileBytes);
                    }
                }
                ++batch;
                ++done;
                if (batch == GROUP || done == npk) {
                    noc_async_read_barrier();
                    cb_push_back(0, RP * GROUP);
                    batch = 0;
                }
            });
        }
    }
    noc_semaphore_wait_min(xloaded, xl_need);  // (re-arm: every engine core counted itself on every port)
    noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(xloaded)), 0u - xl_need);
    noc_async_atomic_barrier();
    const uint32_t rsa = xl0 + 2, rnu = moe_block_args::arg(rsa), rnc = moe_block_args::arg(rsa + 1 + 2 * rnu);
    uint32_t pos = 0;  // (the gather left the CB batch-aligned; the phases after it share one chunk count)
    uint32_t seen[128], lseen[128];
    for (uint32_t m = 0; m < x_gather_num_live_chunks; ++m) {
        latent_all_to_all_gather(rsa + 2 + 2 * rnu + 2 * rnc + G + RSX, pos, m, lseen);
        reduce_scatter_gather(rsa, pos, m, seen);
    }
#if TOKEN_LOOPS > 1
    if (pos % GROUP) {  // (token loops: the last group padded - the next loop's gather writes whole groups from here)
        cb_reserve_back(0, (GROUP - pos % GROUP) * RP);
        cb_push_back(0, (GROUP - pos % GROUP) * RP);
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
