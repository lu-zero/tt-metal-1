// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router-core kernels' addresses, as common runtime args (the program hash then holds none). Every router kernel
// (NCRISC, compute, BRISC) gets the same two common args:
//   [0] the router cores' arena base (the CB range's address on the router cores)
//   [1] the engine-done semaphore address (the matmul cores count their partial logits tiles into it):
//   ROUTER_ENGINE_DONE_SEM_ADDR
// and the host defines the offset the macro below builds an address from:
//   ROUTER_MSYNC_OFFSET    ROUTER_MSYNC_ADDR = arena base + ROUTER_MSYNC_OFFSET
#pragma once

// (moe_block_args::common - token_loop.hpp - reads them: from this core's block of the DRAM table, head_args_to_dram)
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_accessors.hpp"
#endif

#define ROUTER_ENGINE_DONE_SEM_ADDR (moe_block_args::common(1))
#ifdef ROUTER_MSYNC_OFFSET
#define ROUTER_MSYNC_ADDR (moe_block_args::common(0) + ROUTER_MSYNC_OFFSET)
#endif

// PADDED_CHUNK (a padded forward), read on device every call (a trace replays with the values of the moment):
//  * moe_head(..., actual_start=, actual_end=): one chunk, main's chunked-prefill metadata scalars - two one-element
//    uint32 DRAM tensors, the chunk's first real token's absolute position and one past its last (common [2], [3]);
//  * PADDED_FORWARD_METADATA, moe_head(..., metadata=): the forward's metadata words (common/prefill/prefill_chunk.py,
//  one uint32
//    DRAM tensor, common [2]) - PADDED_FORWARD_CHUNKS (forward_chunks) chunks packed in the forward: at 1 (slot, start,
//    end), above {num_chunks, 0} then PADDED_FORWARD_CHUNKS (slot, start, end) records (past num_chunks unused, empty).
// Each SP row holds every chunk's B = [6] / PADDED_FORWARD_CHUNKS rows back to back (chunk k: rows k B ..), each chunk
// laid out as main's KV-pad-aware rotated block-cyclic layout over the SP rows (moe_padding_config's writer,
// update_padded_kv_cache): an SP row's real rows of chunk k are a prefix of its block, their count main's closed form.
// Within the SP row, TP chip p owns rows p S .. (p + 1) S. Common args [4] my SP row r, [5] SP rows, [6] tokens an SP
// row, [7] my first row in it (p S; with token loops the loop's first row plus p S), [8] S, [9] a token loop's rows,
// [10] the loops, [11] the first loop's first row of mine. A token of this chip is real when its row is
// (padded_row_real): the planner counts, stages and plans only those, and the route words of the rest are zero - a
// padding token is not routed, dispatched, computed or combined.
#if defined(PADDED_CHUNK) && !defined(COMPILE_FOR_TRISC)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/padded_rows.hpp"
// the metadata's slot in a scratch from a: its first 64 B aligned address and 64 B on (the DRAM source is 64 B
// aligned; into a 16 B aligned, not 64 B aligned slot the read came back zero) - give it 192 B
FORCE_INLINE uint32_t padded_metadata_slot(uint32_t a) { return (a + 63) & ~63u; }
// issue only - noc_async_read_barrier + invalidate_l1_cache before reading them: the metadata words into l1, or
// actual_start's 16 B into l1 and actual_end's into l1 + 64
FORCE_INLINE void padded_metadata_read(uint32_t l1) {
#ifdef PADDED_FORWARD_METADATA
    constexpr uint32_t bytes = PADDED_FORWARD_CHUNKS == 1 ? 16 : (4 * (2 + 3 * PADDED_FORWARD_CHUNKS) + 15) / 16 * 16;
    noc_async_read(TensorAccessor(routing_accessors::start, moe_block_args::common(2), 64).get_noc_addr(0), l1, bytes);
#else
    noc_async_read(TensorAccessor(routing_accessors::start, moe_block_args::common(2), 64).get_noc_addr(0), l1, 16);
    noc_async_read(TensorAccessor(routing_accessors::end, moe_block_args::common(3), 64).get_noc_addr(0), l1 + 64, 16);
#endif
}
FORCE_INLINE PaddedRows padded_rows(uint32_t l1) {
    return padded_rows_of(l1, moe_block_args::common(4), moe_block_args::common(5), moe_block_args::common(6));
}
#endif
