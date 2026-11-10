// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE head's layout planner: every candidate layout of the shape (matmul cores engine_columns x engine_rows,
// row_tiles_per_core row tiles a core row, row_chunks row chunks, K step k_step_tiles, the router's logits on the
// matmul cores or on router cores, P-tile or compact RS zones), the ones whose L1 fits and whose per-core requirements
// hold (head_sizing.hpp), ranked by a cost model calibrated on measured heads. A chunk no layout fits runs as token
// loops (MoEBlock's loop planner). fit() is the head's whole host plan: a shape it accepts builds; a shape it refuses
// says which requirement failed, with the numbers and what would fit. Host only.

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "head_sizing.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

// Host-only defaults (a shape planned without a device): a Galaxy chip's compute grid and free L1. A block plans on
// its device's grid and measured free L1 (moe_head_l1_free).
inline constexpr std::array<uint32_t, 2> kHeadGrid = {12, 10};
inline constexpr int64_t kL1Free = 1383 * 1024;
// L1 the block allocates after the head is planned (its own words: 128 B a bank in L1 with the persistent words in
// L1_SMALL, ~4.3 KB without) and the allocator's alignment
inline constexpr int64_t kL1Margin = 8 * 1024;
inline constexpr int64_t kUnlimited = int64_t(1) << 40;

// What the head computes, a chip.
struct HeadShape {
    uint32_t rows = 0;  // a mesh row's tokens
    uint32_t hidden_dim = 0;
    uint32_t shared_expert_dim = 0;  // the shared expert's intermediate a TP chip
    uint32_t tp_size = 0;
    uint32_t router_candidates = 0;  // router candidates (num_experts x fold)
    uint32_t top_k = 0;
    std::optional<uint32_t> latent;
    uint32_t fold = 1;
    uint32_t links = 2;
    bool ring = false;  // the TP axis wraps
    std::array<uint32_t, 2> grid = kHeadGrid;
    uint32_t sp_rows = 1;     // mesh rows
    bool sp_ring = false;     // the SP axis wraps
    bool transposed = false;  // SP on mesh axis 1 (mesh_routing.hpp MeshTopology.transposed)
    // the fabric's first hop between every pair of chips (MoEBlockResources::first_hops: FirstHops.dirs, [src N + dst]
    // in the block's chip order): the routing exchange's ports follow it
    std::vector<uint8_t> first_hops;
    uint32_t payload = 8256;  // the fabric's packet payload bytes
    uint32_t accumulators = 16;  // the combine's accumulators
    WeightDtype w_dtype = WeightDtype::Bf8;
    // token loops: a chunk's mesh row runs as `loops` head calls of `rows` rows each (MoEBlockConfig.token_loops);
    // the routing planner and exchange are sized for all of them (SA tokens a chip) and plan them in the last call
    uint32_t loops = 1;
    uint32_t splits = 1;  // the block's splits (chunk_splits): a padded chunk's rows of the whole chunk
    // the caller's tokens an SP row (chunk_tokens / SP: without the padding rows, MoEBlockConfig.pad_tokens) - main's
    // closed form of an SP row's real rows; 0: splits x loops x rows
    uint32_t row_tokens = 0;
    // the chunks packed in a forward (MoEBlockConfig.forward_chunks): an SP row holds each chunk's row_tokens /
    // fwd_chunks rows back to back
    uint32_t fwd_chunks = 1;

    int reduce_scatter_dim() const { return latent ? -1 : -2; }
    uint32_t num_chips() const { return sp_rows * tp_size; }
    uint32_t tokens_per_chip() const { return rows / tp_size; }
    uint32_t tokens_per_chip_all_loops() const {
        return tokens_per_chip() * loops;
    }  // a chip's owned tokens of the whole chunk (every loop's)
    uint32_t experts_per_chip() const { return router_candidates / fold / num_chips(); }
    bool operator==(const HeadShape&) const = default;
};

// A layout of the head; ordered field by field (the planner's tie order).
struct HeadLayout {
    uint32_t engine_rows = 0;
    uint32_t row_tiles_per_core = 0;
    uint32_t row_chunks = 0;
    uint32_t k_step_tiles = 0;
    uint32_t engine_columns = 11;
    bool compact_reduce_scatter_zones = false;  // compact RS arrival zones
    // router iterations: the router stage (logits, top-K, routing rows) runs router_iterations times, each over
    // row_chunks / router_iterations row chunks on the router cores of one iteration (a barrier across the TP ring
    // between iterations); 1 - all rows at once
    uint32_t router_iterations = 1;
    auto operator<=>(const HeadLayout&) const = default;
};

// The cost model's rates, us.
struct HeadCost {
    double clk = 1.35e3;
    double tile = 20;
    double round_us = 0.9774;
    double link_gbs = 20.89;
    double dram_gbs = 244.5;
    double rtr_tile = 20;
    double chain_us = 83.11;
    double chunk_us = 48.04;
    double base_us = 26.21;
    double noc_gbs = 11.69;
    double chain_ek = 0.1852;
    double router_barrier_us = 5.0;  // (a router iteration's barrier across the TP ring: estimated, not calibrated)
    double xmc_gbs = 43.98;
};

using Parts = std::vector<std::pair<std::string, int64_t>>;

// A sized layout and its predicted time; predicted_us is +inf for a refused layout (note: the reason).
struct HeadPlanned {
    HeadLayout layout;
    int64_t arena = 0;  // bytes a core
    Regions sizes;
    double predicted_us = 0;
    Parts parts;
    std::string note;
};

// head_plan_fit's answer.
struct HeadFit {
    bool ok = false;
    std::vector<HeadPlanned> planned;  // best first
    std::string reason;
    std::optional<HeadPlanned> unconstrained;  // the cost model's pick with unlimited L1
    std::string note;                 // what L1 forced
};

// The arena cap a K step allows.
using ArenaMax = std::function<int64_t(uint32_t kb)>;
ArenaMax arena_max_of(int64_t l1_free, uint32_t payload, int64_t cb_floor = 0);
ArenaMax arena_max_fixed(int64_t bytes);

struct HeadFitOptions {
    std::optional<int64_t> l1_free;  // free L1 a core at the head's build (default kL1Free - kL1Margin)
    int64_t cb_floor = 0;  // other programs' static CBs under the arena (arena_cap_of: the MoE block's flat + dispatch)
    bool largest = true;   // on an L1 refusal, also the largest rows a mesh row that fit
    bool explain = true;   // the unconstrained pick and the L1 note
    uint32_t min_router_iters = 1;  // MoEBlockConfig.head_router_iterations (at least; 1: the planner's)
};

// head_plan_fit: the head's whole host plan.
HeadFit head_plan_fit(const HeadShape& shape, const HeadFitOptions& options = {});
HeadFit head_plan_fit(const HeadShape& shape, const ArenaMax& arena_max, const HeadFitOptions& options);

// The pieces fit() is built from (exposed for tests and the device op's build).
uint32_t engine_width(const HeadShape& shape);
std::vector<std::array<uint32_t, 3>> head_layouts(
    const HeadShape& shape, uint32_t mt_max = 2);  // (engine_rows, row_tiles_per_core, row_chunks)
std::vector<uint32_t> k_steps(const HeadShape& shape);
std::string shape_checks(const HeadShape& shape);
std::string describe_layout(const HeadLayout& lay);
// (sized engine or nullopt, arena bytes, region sizes, why not)
struct HeadSized {
    std::optional<SharedExpertSizing> eng;
    int64_t arena = 0;
    Regions sizes;
    std::string reason;
};
HeadSized head_size(const HeadShape& shape, const HeadLayout& lay, int64_t cap);
std::pair<double, Parts> head_cost(
    const HeadShape& shape, const HeadLayout& lay, const SharedExpertSizing& eng, const HeadCost& c = {});
std::vector<HeadPlanned> head_plan_layouts(
    const HeadShape& shape,
    const ArenaMax& arena_max,
    bool keep_unfit = false,
    uint32_t min_router_iters = 1);

}  // namespace ttnn::operations::experimental::moe_block::moe_head
