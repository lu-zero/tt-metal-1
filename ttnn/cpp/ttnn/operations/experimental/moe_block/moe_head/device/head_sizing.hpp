// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE head's host sizing, no device: the pieces head_plan_fit needs to decide whether a layout builds.
//  * ReduceScatterPlan: the SE down + reduce-scatter's tiling and transport schedule (over plan_transport).
//  * SharedExpertSizing: the matmul cores' arena plan.
//  * router_layout / router_min_bytes: the router cores' arena regions.
//  * routing_exchange_sizes / worst_rows / slot_bytes: the routing exchange's buffers.
//  * routing_paths: the routing exchange's senders.
// A requirement a layout does not meet is a SizingError saying which one, with its numbers.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ttnn::operations::experimental::moe_block::moe_head {

namespace head_math {
inline int64_t al64(int64_t v) { return (v + 63) / 64 * 64; }
inline int64_t al16(int64_t v) { return (v + 15) / 16 * 16; }
inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
}  // namespace head_math

// A requirement of the build failed (what(): which one, with its numbers).
struct SizingError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The matmul cores' arena does not fit its cap at any W ring / h slot count.
struct ArenaTooLarge : SizingError {
    ArenaTooLarge(const std::string& what, int64_t need, int64_t cap) : SizingError(what), need(need), cap(cap) {}
    int64_t need;
    int64_t cap;
};

// Ordered (name, bytes) regions, in arena order.
using Regions = std::vector<std::pair<std::string, int64_t>>;

inline constexpr int64_t kArenaMax = 1344 * 1024;  // the arena's largest size a core
// The kernels' args blocks (head_program.cpp head_args_to_dram): a region past the arena's regions on every arena core
// - the RISCV_0 kernel's block, the RISCV_1 kernel's, the compute kernel's page. Taken off the arena's cap.
inline constexpr int64_t kHeadArgsBytes = 12 * 1024;
inline constexpr uint32_t kDestTiles = 8;  // the compute kernel's DEST tiles

// port_cb_bytes: a port core's packet CB at K step kb (an arena region on the port cores).
int64_t port_cb_bytes(uint32_t kb, uint32_t payload, uint32_t group = 4);
// arena_cap_of: L1 the arena may take from a free-L1 figure, above cb_floor bytes of other static CBs (every
// head CB is in the arena; the MoE block's programs that run in it after the head have none either: 0 there) and below
// the kernels' args region (kHeadArgsBytes).
int64_t arena_cap_of(int64_t l1_free, int64_t cb_floor = 0);

// n into `parts` widths, multiples of unit, as even as possible (the first n / unit % parts one unit wider).
std::vector<uint32_t> column_split(uint32_t n, uint32_t parts, uint32_t unit);

// ---- the reduce-scatter's transport schedule (mirror off, one wave) ----
enum class ReduceScatterKind : uint8_t { Line, RingBalanced, RingUnbalanced };
enum class ReduceScatterDir : uint8_t { Fwd = 0, Bwd = 1, Fin = 2 };
ReduceScatterKind reduce_scatter_kind(uint32_t G, bool ring);

struct ReduceScatterRole {
    ReduceScatterDir dir;
    bool start;
    uint32_t dist;
};

struct ReduceScatterChip {
    std::map<std::pair<uint32_t, uint32_t>, ReduceScatterRole> roles;                 // (block, half)
    std::vector<uint32_t> entries[2];                                                 // per half
    std::map<std::pair<uint32_t, ReduceScatterDir>, std::vector<uint32_t>> arrivals;  // (half, fwd | bwd)
};

struct ReduceScatterTransport {
    std::vector<ReduceScatterChip> chips;
    uint32_t slots[2] = {1, 1};  // fwd, bwd
};
ReduceScatterTransport plan_transport(uint32_t G, ReduceScatterKind kind);

// ---- ReduceScatterPlan (the parts the sizing reads) ----
class ReduceScatterPlan {
public:
    struct Pass {
        uint32_t c0;
        uint32_t pw;
        std::optional<uint32_t> entry;
    };
    enum class Op : uint8_t { Pass, Hand, Fin };

    ReduceScatterPlan(
        uint32_t G,
        bool ring,
        uint32_t engine_columns,
        uint32_t engine_rows,
        uint32_t row_tiles_per_core,
        uint32_t KT,
        int scatter,
        uint32_t nv,
        uint32_t pwmax,
        bool compact_reduce_scatter_zones);

    uint32_t G, engine_columns, engine_rows, row_tiles_per_core, KT, nv, pwmax, chunk_row_tiles, ncores;
    int scatter;
    bool compact_reduce_scatter_zones;
    ReduceScatterKind kind;
    ReduceScatterTransport transport;
    uint32_t P = 0;
    // scatter -1
    uint32_t NB = 0, npb = 0;
    std::vector<uint32_t> w, s;
    // scatter -2
    uint32_t RB = 0, npass = 0;
    std::vector<uint32_t> pn, n0;
    std::optional<uint32_t> hcol;
    bool pm = false;

    uint32_t k(uint32_t cx, uint32_t cy) const { return cy * engine_columns + cx; }
    uint32_t half(uint32_t k) const;
    std::pair<uint32_t, uint32_t> rows_in(uint32_t cy, uint32_t b) const;
    uint32_t count(uint32_t cx, uint32_t cy, uint32_t b) const;
    const std::vector<uint32_t>& entries(uint32_t p, uint32_t h = 0) const { return transport.chips[p].entries[h]; }
    const ReduceScatterRole& role(uint32_t p, uint32_t b, uint32_t h) const;
    std::vector<Pass> column_passes(uint32_t p, uint32_t cx) const;
    uint32_t finb_tiles(uint32_t cx, uint32_t cy) const;
    // (offsets by entry, HOLD tiles used)
    std::pair<std::map<uint32_t, uint32_t>, uint32_t> hold_offs(uint32_t p, uint32_t cx, uint32_t cy) const;
    std::vector<std::pair<Op, uint32_t>> ops(uint32_t p, uint32_t cx, uint32_t cy) const;
    std::vector<std::pair<uint32_t, ReduceScatterDir>> handoffs(uint32_t p, uint32_t cx, uint32_t cy) const;
    std::optional<uint32_t> down(uint32_t p, ReduceScatterDir d) const;
    std::optional<uint32_t> piece_entry(uint32_t p, uint32_t h, ReduceScatterDir d) const;
    std::vector<std::pair<uint32_t, uint32_t>> zone_slots(
        uint32_t p, uint32_t cx, uint32_t cy, ReduceScatterDir d) const;
    uint32_t zone_tiles(ReduceScatterDir d) const;

    // ---- the device build's tables ----
    struct Arrival {
        uint32_t entry, slot, threshold;
    };
    struct PortUnit {
        uint32_t entry, block, mask, slot0, slot1;
        bool pieces;
    };
    uint32_t link(uint32_t k) const { return k % nv; }
    std::optional<uint32_t> up(uint32_t p, ReduceScatterDir d) const;
    // per direction (fwd, bwd): [(entry, zone slot, threshold)]
    std::array<std::vector<Arrival>, 2> arrivals(uint32_t p, uint32_t cx, uint32_t cy) const;
    std::vector<PortUnit> port_units(uint32_t p, ReduceScatterDir d) const;
    std::vector<ReduceScatterDir> finb_readers(uint32_t p, uint32_t cx, uint32_t cy) const;
    std::vector<std::pair<uint32_t, uint32_t>> port_cores(uint32_t l) const;  // (cx, cy) of link l, k order
    std::tuple<uint32_t, uint32_t, uint32_t> fin_map(uint32_t p, uint32_t cx, uint32_t cy) const;
    uint32_t prefix_mask(uint32_t q, uint32_t h, ReduceScatterDir d, uint32_t slot) const;
    uint32_t counts_word(uint32_t cx, uint32_t cy) const;
    uint32_t finb_off(uint32_t p, uint32_t cx, uint32_t cy, uint32_t e) const;
    std::vector<uint32_t> pass_widths(uint32_t cx) const;
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> pass_bases(uint32_t p, uint32_t cx, uint32_t cy) const;
    std::vector<uint32_t> compute_args(uint32_t p, uint32_t cx, uint32_t cy) const;

private:
    bool needs_hold(uint32_t p, uint32_t cx, uint32_t cy, uint32_t e) const;
};

// ---- the shared expert's (matmul cores') host sizing ----
enum class WeightDtype : uint8_t { Bf4, Bf8, Bf16 };

struct SharedExpertArgs {
    uint32_t rows = 0, hidden_dim = 0, shared_expert_dim = 0;
    uint32_t engine_columns = 11, engine_rows = 10, row_tiles_per_core = 1, k_step_tiles = 16;
    uint32_t shards = 1;
    const ReduceScatterPlan* rs = nullptr;   // the down + reduce-scatter's plan (required)
    std::optional<uint32_t> latent;          // tiles a chip
    std::pair<uint32_t, uint32_t> router{};  // (ET, KTO): the router logits, computed on the engine
    uint32_t router_iters = 1;               // the router stage over row_chunks / router_iters row chunks at a time
    int64_t arena_max = kArenaMax;
    WeightDtype w_dtype = WeightDtype::Bf8;
};

struct SharedExpertSizing {
    explicit SharedExpertSizing(const SharedExpertArgs& a);  // throws SizingError / ArenaTooLarge

    SharedExpertArgs args;  // (args.rs: not owned - valid while the caller's ReduceScatterPlan is)
    uint32_t engine_columns, engine_rows, row_tiles_per_core, chunk_row_tiles, rows, hidden_dim, shared_expert_dim,
        shards;
    uint32_t row_chunks, KT, shared_expert_tiles, k_step_tiles, num_k_steps, k_steps_per_shard, S0, S1, gate_up_passes,
        gate_up_passes_without_latent, down_passes, down_k_step_tiles, down_k_steps, NB, w_slot_tiles, w_tile_bytes;
    uint32_t max_column_pairs, max_down_pass_width, gate_up_pass_width,
        max_latent_column_tiles = 0, latent_tiles = 0, w_ring_slots = 0, h_slots = 1, prep_full_column_tiles = 0,
        prep_block_column_tiles = 0;
    int64_t latent_pages = 0, xown_t = 0, wown_t = 0, acc_pages = 0;
    std::vector<uint32_t> fam0, fam1;  // the x families' owner columns
    std::vector<uint32_t> po, h0, lw, l0;
    std::vector<std::vector<uint32_t>> wg, wd;
    // the router logits on the engine (args.router)
    uint32_t router_expert_tiles_per_column = 0, RG = 0, router_k_chunk_tiles = 0;
    bool rtr_zones = false, rtr_latz = false, rtr_overlay = false;
    Regions reduce_scatter_sizes;  // the reduce-scatter's regions
    int64_t arena_bytes = 0;
    Regions off;                    // region offsets, arena order
    Regions sizes;                  // each region's bytes (overlays share their host's)
    std::vector<int64_t> seq_ends;  // the sequential regions' bounds (the CB ring check's)
    // compute kernel groups: columns with the same (gate / up widths, down widths), in first-column order
    std::vector<std::pair<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>, std::vector<uint32_t>>> groups;

    int64_t dram_w_bytes() const {
        return int64_t(KT * 2 * shared_expert_tiles + shared_expert_tiles * KT) * w_tile_bytes;
    }
    int64_t at(const std::string& region) const;  // off[region]
    bool in_fam0(uint32_t c) const;
    bool in_fam1(uint32_t c) const;
    // (owner column, family, index in the family) of x step j
    std::tuple<uint32_t, uint32_t, uint32_t> owner(uint32_t j) const;
    // chunk m's local row tile lr -> the global row tile
    uint32_t grow(uint32_t m, uint32_t lr) const;
    // (first local row, rows) of core row cy's row_tiles_per_core row tiles in owner block b (latent / prep)
    std::pair<uint32_t, uint32_t> latent_rows(uint32_t cy, uint32_t b) const;
    uint32_t bsize(uint32_t c, uint32_t b) const;
    int64_t lrdy_bytes() const { return 16 * int64_t(std::max<uint32_t>(4, row_chunks)); }
    bool fin_rows() const;
};

// ---- the router cores ----
// (offsets, bytes) of the router cores' arena regions.
// arr_first (token loops): the peers' logits arrivals before the bias, so the last loop's routing planner overlays them
// too (they are dead once every router core's rows are written; an earlier loop's planner never overlays them - a peer
// one loop ahead writes its next logits there)
std::pair<Regions, int64_t> router_layout(
    uint32_t G, uint32_t ET, uint32_t K, uint32_t ETF = 0, bool arr_first = false);
int64_t router_min_bytes(uint32_t G, uint32_t E, uint32_t K);

// ---- the routing exchange ----
int64_t routing_header_bytes(uint32_t EPC, uint32_t M);
int64_t routing_slot_bytes(uint32_t S, uint32_t K, uint32_t EPC, uint32_t M);
int64_t routing_worst_rows(uint32_t N, uint32_t S, uint32_t K, uint32_t EPC);
struct RoutingSizes {
    uint32_t M;
    int64_t inst_cap, plan_l1, port_l1;
    bool fits;
    std::string reason;
};
RoutingSizes routing_exchange_sizes(
    uint32_t N, uint32_t S, uint32_t K, uint32_t EPC, uint32_t A, int64_t rows_cap, uint32_t M = 0, uint32_t TPP = 32);

// The routing planner's instances on the router cores (nc of them, one each - its BRISC, NOC1): the most instances M
// (<= nc + nx, EPC, A, 14) whose regions (one a core, step bytes) fit - over the router's dead buffers [0, overlay)
// when they fit there, else past the router's bytes - in the arena. fixed_M: that many (an earlier token loop's
// planner: the last's instances over the router's local buffers [0, arr) - never the peers' logits arrivals).
// nx (token loops): planner-only cores - their whole arena the planner's. The instances fill the planner-only cores
// first (q < nx: core q), then the router cores (q - nx); without planner-only cores instance q is router core q's.
// Every instance runs on its core's BRISC, on NOC1 - the fabric routers' NoC: the routing packets land in DRAM through
// the NOC1 endpoint and the router's rows are written on NOC1, so every planner reads them through the port they were
// written through (a read through the other NoC's endpoint can return the page's previous contents after the write's
// ack: the BH DRAM cross-port stale read). One NOC1 RISC a core (two RISCs on one NoC share its counters).
// pbase(q): its region's offset.
struct PlannerSlot {
    bool router;    // a router core's (else a planner-only core's)
    uint32_t core;  // its index among the router / the planner-only cores
    uint32_t risc;  // 0 BRISC, 1 NCRISC (its region: the core's risc-th)
};
inline PlannerSlot planner_slot(uint32_t q, uint32_t nc, uint32_t nx) {
    if (q < nx) {
        return {false, q, 0};
    }
    q -= nx;
    return {true, q % nc, 0};  // (place_planner: M <= nc + nx - one instance a core)
}
struct PlannerPlace {
    bool ok = false;
    uint32_t M = 0;
    RoutingSizes sz{};
    int64_t step = 0, pl0 = 0;
    uint32_t nc = 0, nx = 0;  // router cores, planner-only cores
    int64_t pbase(uint32_t q) const {
        const auto s = planner_slot(q, nc, nx);
        return (s.router ? pl0 : 0) + int64_t(s.risc) * step;
    }
};
PlannerPlace place_planner(
    uint32_t N,
    uint32_t S,
    uint32_t K,
    uint32_t EPC,
    uint32_t A,
    uint32_t nc,
    int64_t overlay,
    int64_t rbytes,
    int64_t arena,
    uint32_t fixed_M = 0,
    uint32_t nx = 0);
// token loops: the planner cores a head wants (router cores and planner-only ones, two instances each)
inline constexpr uint32_t kPlannerCores = 14;
// The cells no head piece takes, (x, y): left of the mover column (the grid's last), past the matmul cores
// (engine_columns x engine_rows), less the router cores placed under the matmul cores (the first `under` of rows
// engine_rows .. by row, columns 0 .. engine_columns); the planner-only cores (token loops) are the first of them.
inline std::vector<std::pair<uint32_t, uint32_t>> head_free_cells(
    uint32_t grid_x, uint32_t grid_y, uint32_t engine_columns, uint32_t engine_rows, uint32_t under) {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    uint32_t skipped = 0;
    for (uint32_t y = engine_rows; y < grid_y; ++y) {
        for (uint32_t x = 0; x < engine_columns; ++x) {
            if (skipped < under) {
                ++skipped;
                continue;
            }
            out.emplace_back(x, y);
        }
    }
    for (uint32_t y = 0; y < grid_y; ++y) {
        for (uint32_t x = engine_columns; x + 1 < grid_x; ++x) {
            out.emplace_back(x, y);
        }
    }
    return out;
}
// token loops: how many planner-only cores a head with nc router cores takes, of `free` free cells - up to every
// instance's (14 at most, one a core): with them all past the router cores, the router kernels carry no planner (one
// program running every loop holds both planners - an earlier loop's and the last's - and the router cores' would not
// fit the kernel config buffer)
inline uint32_t planner_only_cores(uint32_t loops, size_t free) {
    if (loops <= 1) {
        return 0;
    }
    return static_cast<uint32_t>(std::min<size_t>(free, kPlannerCores));
}
// an earlier token loop's planner plans no token: sized for one row tile
inline constexpr uint32_t kPlannerIdleS = 32;
// ---- routing_paths ----
struct RoutingPaths {
    uint32_t nx;     // the most packets an X sender sends
    uint32_t ny;     // ... a Y sender
    uint32_t nsend;  // the most senders a chip
};
// The routing exchange's paths on an R x C mesh, or why they do not build.
// first_hops: the fabric's first hop between every pair of chips (HeadShape.first_hops).
std::variant<RoutingPaths, std::string> routing_paths(
    uint32_t R, uint32_t C, bool ring, uint32_t links, const std::vector<uint8_t>& first_hops);

// "%.<p>f" of v, and "[a, b, ...]" of a list (the planner's messages).
std::string fmt_fixed(double v, int precision);
std::string list_str(const std::vector<uint32_t>& v);

}  // namespace ttnn::operations::experimental::moe_block::moe_head
