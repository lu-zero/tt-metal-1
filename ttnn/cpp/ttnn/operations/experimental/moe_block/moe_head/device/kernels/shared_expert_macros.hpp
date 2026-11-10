// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The matmul-core kernels' macros, built from host defines that carry no address.
#pragma once

// GLOBAL_ROW_TILE(m, lr): the global row tile of chunk m's chunk-local row tile lr. With row chunks over TP shards
// (row_chunks > 1 and shards > 1) chunk m is block m of every owner, so the owners' rows stay contiguous: the host
// defines SHARED_EXPERT_OWNER_CHUNK_ROW_TILES = CR / shards and SHARED_EXPERT_OWNER_ROW_TILES = rows / 32 / shards;
// else chunk-major (SHARED_EXPERT_CHUNK_ROW_TILES = CR, row tiles a chunk).
#ifdef SHARED_EXPERT_OWNER_CHUNK_ROW_TILES
#define GLOBAL_ROW_TILE(m, lr)                                                      \
    (((lr) / SHARED_EXPERT_OWNER_CHUNK_ROW_TILES) * SHARED_EXPERT_OWNER_ROW_TILES + \
     (m) * SHARED_EXPERT_OWNER_CHUNK_ROW_TILES + (lr) % SHARED_EXPERT_OWNER_CHUNK_ROW_TILES)
#else
#define GLOBAL_ROW_TILE(m, lr) ((m) * SHARED_EXPERT_CHUNK_ROW_TILES + (lr))
#endif

// ROUTER_PHASE_WORDS_ADDR: the router-phase words (a persistent L1 tensor a core). The BRISC / NCRISC read its address
// from common arg 0 (their common args are [ewords, N, the slot map, the shared slots]: head_program.cpp slot_map).
// Only expanded by the dataflow kernels (the compute never names it).
#define ROUTER_PHASE_WORDS_ADDR (moe_block_args::common(0))
