// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "head_plan.hpp"
#include "ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/operations/experimental/moe_block/common/kernels/combine_plan_words.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>

using moe_block_tiles::kBf16TileBytes;
using moe_block_tiles::kFp32TileBytes;

namespace ttnn::operations::experimental::moe_block::moe_head {

// The cost model's arithmetic with every operation rounded on its own, no fused multiply-add: the ranking of close
// layouts does not depend on what the compiler contracts.
#if defined(__clang__)
#define HEAD_PLAN_NO_FP_CONTRACT _Pragma("clang fp contract(off)")
#else
#define HEAD_PLAN_NO_FP_CONTRACT
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

namespace head_plan_detail {

constexpr uint32_t kPortGroup = 4;     // the x gather: a port sender's packet group
constexpr uint32_t kHeaders = 12;      // fabric packet headers a RISC
constexpr uint32_t kRouterArgs = 341;  // a router core's runtime-arg bound
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kEngOverhead = 3.0;  // the logits on the matmul cores: their matmul's cost over the router cores'

using head_math::al64;
using head_math::ceil_div;
std::string kb0(double bytes) { return fmt_fixed(bytes / 1024, 0); }
std::string s(int64_t v) { return std::to_string(v); }

int64_t round_us(double v) { return static_cast<int64_t>(std::nearbyint(v)); }

enum class Cells : uint8_t { None, Column, Row, Rows };

// router_cells: where the router cores go.
// nrt: the router cores (a row tile each of the chip's owned rows of one router iteration).
Cells router_cells(const HeadShape& shape, uint32_t engine_rows, uint32_t engine_columns, uint32_t nrt) {
    const uint32_t gy = shape.grid[1];
    if (nrt + 2 * shape.links + 1 <= gy) {
        return Cells::Column;
    }
    if (engine_rows < gy && nrt <= engine_columns) {
        return Cells::Row;
    }
    if (engine_rows < gy && nrt <= engine_columns * (gy - engine_rows)) {
        return Cells::Rows;
    }
    return Cells::None;
}

uint32_t latent_dests(uint32_t G, bool ring) { return ring && G > 2 ? G / 2 : G - 1; }

int64_t region_of(const Regions& r, const std::string& k) {
    for (const auto& [n, v] : r) {
        if (n == k) {
            return v;
        }
    }
    return 0;
}

// cores_check: every per-core requirement past the matmul cores' arena; "" or why not. The ports' buffers
// (logits sub-blocks, routing exchange packets, x-gather staging) and the watcher's grow the arena up to cap, as the
// program grows it (head_program: the arena is one size on every core).
std::string cores_check(
    const HeadShape& shape, const HeadLayout& lay, const ReduceScatterPlan& rs, int64_t& arena, int64_t cap) {
    const uint32_t G = shape.tp_size, L = shape.links, gy = shape.grid[1];
    const bool ring = shape.ring && G > 2;
    if (rs.P >= 256 || (G > 4 && rs.scatter != -2)) {
        return "reduce-scatter counts word: blocks of < 256 tiles (P " + s(rs.P) + "), past TP 4 the rows scatter only";
    }
    if (kPortGroup + 1 > kHeaders) {
        return "packet headers";
    }
    if (shape.latent) {
        const uint32_t nd = latent_dests(G, ring);
        if (nd > 3 || kPortGroup + 1 + nd > kHeaders) {
            return "latent all-to-all: " + s(nd) + " destination blocks a port > 3 (packet headers: " + s(kHeaders) +
                   " a RISC)";
        }
    }
    const uint32_t nrt =
        shape.rows / 32 / G / lay.router_iterations;  // (the router cores: one router iteration's row tiles)
    if (lay.engine_columns * lay.engine_rows / L > 128) {
        return "a port's reduce-scatter cores: " + s(lay.engine_columns * lay.engine_rows / L) +
               " > 128 (the port reader's rdc / seen tables)";
    }
    if (3 * L > gy) {
        return "mover column: " + s(2 * L) + " ports + " + s(L) + " copy cores > " + s(gy) + " cells";
    }
    if (nrt + 2 * L + 1 > gy) {
        const int64_t free = (int64_t(gy) - lay.engine_rows) * lay.engine_columns;
        if (lay.engine_rows >= gy || !(nrt <= lay.engine_columns || nrt <= free)) {
            return "router cores: " + s(nrt) +
                   " (a row tile each of a chip's owned rows of a router iteration) fit neither the mover column (" +
                   s(gy) + " cells less " + s(2 * L) + " ports and the watcher) nor the " + s(free) +
                   " free cells under the matmul cores (" + s(lay.engine_columns) + "x" + s(lay.engine_rows) + ")";
        }
    }
    const uint32_t nc = nrt;
    const uint32_t num_planner_only_cores = planner_only_cores(
        shape.loops,
        head_free_cells(shape.grid[0], gy, lay.engine_columns, lay.engine_rows, nrt + 2 * L + 1 > gy ? nrt : 0).size());
    const uint32_t ET = shape.router_candidates / 32;
    const auto [roff, rbytes] = router_layout(G, ET, shape.top_k, ET, shape.loops > 1);
    if (rbytes > arena) {
        return "router cores: " + kb0(rbytes) + " KB > the arena's " + kb0(arena) + " KB";
    }
    const int64_t ns = ceil_div(nc, L);
    const int64_t pbuf = int64_t(G - 1) * ns * ET * kFp32TileBytes;
    if (pbuf > cap) {
        return "the ports' logits sub-blocks: " + kb0(pbuf) + " KB > the arena's " + kb0(cap) + " KB";
    }
    // (the planner and the exchange: the whole chunk's tokens - every token loop's)
    const uint32_t N = shape.num_chips(), S = shape.tokens_per_chip_all_loops(), K = shape.top_k,
                   EPC = shape.experts_per_chip(), A = shape.accumulators;
    if (N * EPC > 1024 || S > 2048 || EPC > 64 || shape.tokens_per_chip() % 32) {
        return "routing planner: N EPC " + s(N * EPC) + " <= 1024, S " + s(S) + " <= 2048 tokens a chip, EPC " +
               s(EPC) + " <= 64";
    }
    // (the last token loop's planner - or the only one's - over the router's dead buffers [0, bias) where it fits)
    const auto pp =
        place_planner(N, S, K, EPC, A, nc, region_of(roff, "bias"), rbytes, arena, 0, num_planner_only_cores);
    const uint32_t M = pp.M;
    if (!pp.ok) {
        const int64_t per = ceil_div(std::max<uint32_t>(M, 1), nc);
        const std::string what = !pp.sz.reason.empty()
                                     ? pp.sz.reason
                                     : s(per) + " region(s) of " + kb0(pp.step) + " KB past " + kb0(pp.pl0) + " KB";
        return "routing planner: " + what + " > the arena's " + kb0(arena) + " KB";
    }
    if (shape.loops > 1) {  // (an earlier loop's: M instances planning no token, past the router's bytes)
        const auto idle = place_planner(
            N, kPlannerIdleS, K, EPC, A, nc, region_of(roff, "arr"), rbytes, arena, M, num_planner_only_cores);
        if (!idle.ok) {
            return "routing planner: an earlier token loop's " + s(M) + " instances past the router's " + kb0(rbytes) +
                   " KB > the arena's " + kb0(arena) + " KB";
        }
    }
    const auto rt = routing_paths(shape.sp_rows, G, ring, L, shape.first_hops);
    if (std::holds_alternative<std::string>(rt)) {
        return "routing exchange: " + std::get<std::string>(rt);
    }
    const auto [nx, ny, nsend] = std::get<RoutingPaths>(rt);
    const int64_t pcap = routing_slot_bytes(S, K, EPC, M);
    const int64_t xg =
        port_cb_bytes(lay.k_step_tiles, shape.payload, kPortGroup);  // (the x-gather's staging, after them)
    const int64_t port = al64(al64(pbuf) + int64_t(nx) * pcap) + xg;
    if (port > cap) {
        return "routing exchange: a port's buffers " + kb0(port) + " KB (logits " + kb0(pbuf) + " + " + s(nx) +
               " packets of " + fmt_fixed(pcap / 1024.0, 1) + " KB + x-gather staging " + kb0(xg) +
               " KB) > the arena's " + kb0(cap) + " KB";
    }
    if (2 * int64_t(ny) * pcap > cap) {
        return "routing exchange: the watcher's Y buffers " + kb0(2 * int64_t(ny) * pcap) + " KB > the arena's " +
               kb0(cap) + " KB";
    }
    arena = std::max({arena, al64(port), al64(2 * int64_t(ny) * pcap)});
    const int64_t plen = 26 + 5 * int64_t(M) + 3 * int64_t(N) + 3 * int64_t(nsend);
    const std::vector<std::pair<std::string, int64_t>> nargs = {
        {"BRISC", 1 + 10 + 3 + 2 * int64_t(nc) + plen},
        {"NCRISC", 2 * int64_t(G) + 10 + (nc < M ? 2 + plen : 2)},
        {"compute", G + 9},
    };
    std::string big;
    for (const auto& [k, v] : nargs) {
        if (v > kRouterArgs) {
            big += (big.empty() ? "" : ", ") + ("'" + k + "': ") + s(v);
        }
    }
    if (!big.empty()) {
        return "router cores' runtime args: {" + big + "} > " + s(kRouterArgs) + " (" + s(nc) + " router cores, " +
               s(M) + " planner instances, N " + s(N) + ")";
    }
    return "";
}

bool is_l1(const HeadPlanned& p) {
    return p.note.rfind("arena", 0) == 0 || p.note.rfind("router cores' arena", 0) == 0 ||
           p.note.find("of L1") != std::string::npos;
}

// summary: why no candidate fits.
std::string summary(const std::vector<HeadPlanned>& unfit, const ArenaMax& arena_max) {
    std::vector<const HeadPlanned*> l1, other;
    for (const auto& p : unfit) {
        (is_l1(p) ? l1 : other).push_back(&p);
    }
    std::vector<std::string> out;
    if (!l1.empty()) {
        const HeadPlanned* least = nullptr;
        int64_t bestk = 0;
        for (const auto* p : l1) {
            int64_t k = (p->arena ? p->arena : kUnlimited) - arena_max(p->layout.k_step_tiles);
            if (!least || k < bestk) {
                least = p;
                bestk = k;
            }
        }
        std::set<uint32_t> kbs, nchs;
        for (const auto* p : l1) {
            kbs.insert(p->layout.k_step_tiles);
            nchs.insert(p->layout.row_chunks);
        }
        out.push_back(
            "L1: the least arena any layout needs is " + kb0(least->arena) + " KB (" + describe_layout(least->layout) +
            "), a core has " + kb0(arena_max(least->layout.k_step_tiles)) + " KB; tried K steps " +
            list_str({kbs.begin(), kbs.end()}) + ", row chunks " + list_str({nchs.begin(), nchs.end()}) +
            ", W rings off / 5 / 4 / 3");
    }
    if (!other.empty()) {
        auto sorted = other;
        std::stable_sort(sorted.begin(), sorted.end(), [](const HeadPlanned* a, const HeadPlanned* b) {
            return std::make_pair(a->layout.row_chunks, -int64_t(a->layout.k_step_tiles)) <
                   std::make_pair(b->layout.row_chunks, -int64_t(b->layout.k_step_tiles));
        });
        std::vector<std::string> reasons;
        const HeadPlanned* first = nullptr;
        for (const auto* p : sorted) {
            std::string r = p->note.substr(0, p->note.find(':'));
            if (std::find(reasons.begin(), reasons.end(), r) == reasons.end()) {
                reasons.push_back(r);
                if (!first) {
                    first = p;
                }
            }
        }
        std::string more = reasons.size() > 1 ? " (and " + s(reasons.size() - 1) + " other reason(s))" : std::string();
        out.push_back(first->note + " at " + describe_layout(first->layout) + more);
    }
    std::string res;
    for (size_t i = 0; i < out.size(); ++i) {
        res += (i ? "; " : "") + out[i];
    }
    return res;
}

// degradation: what L1 forced.
std::string degradation(const HeadPlanned& chosen, const std::optional<HeadPlanned>& free) {
    HEAD_PLAN_NO_FP_CONTRACT
    if (!free || chosen.predicted_us <= free->predicted_us * 1.005) {
        return "";
    }
    const auto &a = chosen.layout, &b = free->layout;
    std::vector<std::string> d;
    if (a.k_step_tiles != b.k_step_tiles) {
        d.push_back("K step " + s(b.k_step_tiles) + " -> " + s(a.k_step_tiles));
    }
    if (a.row_chunks != b.row_chunks) {
        d.push_back("row chunks " + s(b.row_chunks) + " -> " + s(a.row_chunks));
    }
    if (a.engine_rows != b.engine_rows || a.row_tiles_per_core != b.row_tiles_per_core) {
        d.push_back(
            "matmul core rows x row_tiles_per_core " + s(b.engine_rows) + "x" + s(b.row_tiles_per_core) + " -> " +
            s(a.engine_rows) + "x" + s(a.row_tiles_per_core));
    }
    if (a.compact_reduce_scatter_zones != b.compact_reduce_scatter_zones) {
        d.push_back("compact RS zones");
    }
    for (const auto& [k, v] : chosen.parts) {
        if (k == "wring" && v) {
            d.push_back("W streamed from DRAM every chunk (a ring of " + s(v) + " slots)");
        }
    }
    std::string ds;
    for (size_t i = 0; i < d.size(); ++i) {
        ds += (i ? ", " : "") + d[i];
    }
    return "L1-limited: " + (ds.empty() ? std::string("another layout") : ds) + " - predicted " +
           fmt_fixed(chosen.predicted_us, 0) + " us vs " + fmt_fixed(free->predicted_us, 0) +
           " us with unlimited L1 (+" + fmt_fixed(chosen.predicted_us - free->predicted_us, 0) + " us, +" +
           fmt_fixed(100 * (chosen.predicted_us / free->predicted_us - 1), 0) + "%)";
}

}  // namespace head_plan_detail

using namespace head_plan_detail;

ArenaMax arena_max_of(int64_t l1_free, uint32_t payload, int64_t cb_floor) {
    (void)payload;
    return [l1_free, cb_floor](uint32_t) { return arena_cap_of(l1_free, cb_floor); };
}

ArenaMax arena_max_fixed(int64_t bytes) {
    return [bytes](uint32_t) { return bytes; };
}

uint32_t engine_width(const HeadShape& shape) {
    int64_t engine_columns = std::min<int64_t>(int64_t(shape.grid[0]) - 1, shape.shared_expert_dim / 32);
    if (shape.latent) {
        engine_columns = std::min<int64_t>(engine_columns, *shape.latent / 32 / shape.tp_size);
    }
    return engine_columns < 0 ? 0 : uint32_t(engine_columns);
}

std::vector<std::array<uint32_t, 3>> head_layouts(const HeadShape& shape, uint32_t mt_max) {
    const uint32_t G = shape.tp_size, gy = shape.grid[1];
    const uint32_t engine_columns = engine_width(shape);
    const uint32_t RT = shape.rows / 32;
    std::vector<std::array<uint32_t, 3>> out;
    if (shape.rows % 32 || RT % G) {
        return out;
    }
    const uint32_t nrt = RT / G;
    for (uint32_t nch = 1; nch <= nrt; ++nch) {
        if (nrt % nch) {
            continue;
        }
        const uint32_t chunk_row_tiles = RT / nch;
        for (uint32_t engine_rows = gy; engine_rows > 0; --engine_rows) {
            if (chunk_row_tiles % engine_rows || chunk_row_tiles / engine_rows > mt_max) {
                continue;
            }
            bool cells = false;  // (the router cores fit for some router iteration count)
            for (uint32_t nit = 1; nit <= nch && !cells; ++nit) {
                cells = nch % nit == 0 && router_cells(shape, engine_rows, engine_columns, nrt / nit) != Cells::None;
            }
            if (!cells) {
                continue;
            }
            out.push_back({engine_rows, chunk_row_tiles / engine_rows, nch});
        }
    }
    return out;
}

std::vector<uint32_t> k_steps(const HeadShape& shape) {
    const uint32_t klt = shape.hidden_dim / 32 / shape.tp_size;
    std::vector<uint32_t> out;
    for (uint32_t d = 16; d >= 1; --d) {
        if (klt % d == 0 && (klt / d) % 2 == 0) {
            out.push_back(d);
        }
    }
    return out;
}

std::string describe_layout(const HeadLayout& lay) {
    return "matmul cores " + s(lay.engine_columns) + "x" + s(lay.engine_rows) + " row_tiles_per_core " +
           s(lay.row_tiles_per_core) + ", " + s(lay.row_chunks) + " row chunk" + (lay.row_chunks > 1 ? "s" : "") +
           ", K step " + s(lay.k_step_tiles) +
           (lay.router_iterations > 1 ? ", " + s(lay.router_iterations) + " router iterations" : "");
}

HeadSized head_size(const HeadShape& shape, const HeadLayout& lay, int64_t cap) {
    std::optional<ReduceScatterPlan> rs;
    std::optional<SharedExpertSizing> eng;
    try {
        rs.emplace(
            shape.tp_size,
            shape.ring && shape.tp_size > 2,
            lay.engine_columns,
            lay.engine_rows,
            lay.row_tiles_per_core,
            shape.hidden_dim / 32,
            shape.reduce_scatter_dim(),
            shape.links,
            kDestTiles / lay.row_tiles_per_core,
            lay.compact_reduce_scatter_zones);
        SharedExpertArgs a;
        a.rows = shape.rows;
        a.hidden_dim = shape.hidden_dim;
        a.shared_expert_dim = shape.shared_expert_dim;
        a.engine_columns = lay.engine_columns;
        a.engine_rows = lay.engine_rows;
        a.row_tiles_per_core = lay.row_tiles_per_core;
        a.k_step_tiles = lay.k_step_tiles;
        a.shards = shape.tp_size;
        a.rs = &*rs;
        if (shape.latent) {
            a.latent = *shape.latent / 32 / shape.tp_size;
        }
        a.router = std::make_pair(
            shape.router_candidates / 32, shape.hidden_dim / shape.tp_size / 32);  // (the logits on the matmul cores)
        a.router_iters = lay.router_iterations;
        a.arena_max = cap;
        a.w_dtype = shape.w_dtype;
        eng.emplace(a);
    } catch (const ArenaTooLarge& err) {
        return {
            std::nullopt, err.need, {}, "arena " + kb0(err.need) + " KB > " + kb0(cap) + " KB (W ring at the least)"};
    } catch (const SizingError& err) {
        const std::string why = err.what();
        return {std::nullopt, 0, {}, why.empty() ? "sizing failed" : why};
    }
    const int64_t need = al64(router_min_bytes(shape.tp_size, shape.router_candidates, shape.top_k));
    if (need > cap) {
        return {std::nullopt, need, eng->sizes, "router cores' arena " + kb0(need) + " KB > " + kb0(cap) + " KB"};
    }
    int64_t arena = std::max(eng->arena_bytes, need);
    std::string why = cores_check(shape, lay, *rs, arena, cap);
    if (!why.empty()) {
        return {std::nullopt, arena, eng->sizes, why};
    }
    Regions sizes = eng->sizes;
    return {std::move(eng), arena, std::move(sizes), ""};
}

std::pair<double, Parts> head_cost(
    const HeadShape& shape, const HeadLayout& lay, const SharedExpertSizing& eng, const HeadCost& c) {
    HEAD_PLAN_NO_FP_CONTRACT
    const int64_t G = shape.tp_size, row_tiles_per_core = lay.row_tiles_per_core, n = lay.row_chunks;
    const int64_t KT = shape.hidden_dim / 32, shared_expert_tiles = shape.shared_expert_dim / 32;
    int64_t gu = 0;
    for (uint32_t p : eng.po) {
        gu = std::max<int64_t>(gu, 2 * p);
    }
    gu += shape.latent ? eng.max_latent_column_tiles : 0;
    int64_t dn = 0;
    for (const auto& w : eng.wd) {
        int64_t sum = 0;
        for (uint32_t x : w) {
            sum += x;
        }
        dn = std::max(dn, sum);
    }
    const double t_gu = double(n * row_tiles_per_core * KT * gu) * c.tile / c.clk +
                        double(n * eng.gate_up_passes * eng.num_k_steps) * c.round_us;
    const double t_dn = double(n * row_tiles_per_core * shared_expert_tiles * dn) * c.tile / c.clk +
                        double(n * eng.down_passes * eng.down_k_steps) * c.round_us;
    const bool fhops = shape.ring && G > 2;
    const double rate = double(shape.links) * c.link_gbs * 1e3;
    double t_x, t_lat = 0.0;
    if (fhops) {
        const double hops = double(G - 1) / 2;
        t_x = std::floor(hops * shape.rows * shape.hidden_dim / double(G)) * 2 / rate;
        if (shape.latent) {
            t_lat = std::floor(std::floor(hops * shape.rows * *shape.latent / double(G)) / double(G)) * 2 / rate;
        }
    } else {
        const int64_t hops = G - 1;
        t_x = double(hops * shape.rows * shape.hidden_dim / G * 2) / rate;
        if (shape.latent) {
            t_lat = double(hops * shape.rows * int64_t(*shape.latent) / G / G * 2) / rate;
        }
    }
    const double t_rs = t_x;
    const int64_t w = eng.dram_w_bytes();
    const double t_w = double(w * (eng.w_ring_slots ? n : 1)) / (c.dram_gbs * 1e3);
    const double t_wn = double(n * w) / double(lay.engine_columns) / (c.noc_gbs * 1e3);
    const double t_math = t_gu + t_dn;
    const double t_xm = double(n * eng.gate_up_passes) * (double(KT) / 2) * double(row_tiles_per_core) *
                        kBf16TileBytes / double(std::min(eng.S0, eng.S1)) / (c.xmc_gbs * 1e3);
    const double comp = std::max({t_math, t_wn, t_xm}) / double(n) + c.chunk_us * (n > 1 ? 1 : 0);
    const double steady = std::max({comp, (t_x + t_lat + t_rs) / double(n), t_w / double(n)});
    double t_shared_expert = (t_x + t_lat) / double(n) + double(n - 1) * steady + comp + t_rs / double(n);
    if (n == 1) {
        t_shared_expert = std::max(t_shared_expert, t_w + t_dn + t_rs);
    }
    const int64_t ET = shape.router_candidates / 32;
    const double chain = c.chain_us + c.chain_ek * double(ET) * double(shape.top_k);
    const int64_t etc = ceil_div(ET, lay.engine_columns);
    const double t_eng =
        kEngOverhead * double(n) * double(row_tiles_per_core) * double(etc) * double(KT / G) * c.rtr_tile / c.clk;
    t_shared_expert += t_eng;  // (the logits on the matmul cores)
    // (router iterations: the router cores' chain once an iteration, a barrier across the TP ring between them)
    const double t_chain =
        t_eng + double(lay.router_iterations) * chain + double(lay.router_iterations - 1) * c.router_barrier_us;
    const double t = c.base_us + std::max(t_shared_expert, t_chain);
    Parts parts = {
        {"gu", round_us(t_gu)},
        {"down", round_us(t_dn)},
        {"x", round_us(t_x)},
        {"rs", round_us(t_rs)},
        {"w", round_us(t_w)},
        {"wnoc", round_us(t_wn)},
        {"xmc", round_us(t_xm)},
        {"owners", int64_t(std::min(eng.S0, eng.S1))},
        {"se", round_us(t_shared_expert)},
        {"chain", round_us(t_chain)},
        {"wring", int64_t(eng.w_ring_slots)},
    };
    return {t, parts};
}

std::vector<HeadPlanned> head_plan_layouts(
    const HeadShape& shape, const ArenaMax& arena_max, bool keep_unfit, uint32_t min_router_iters) {
    const uint32_t GXe = engine_width(shape);
    // each layout with the fewest router iterations (from min_router_iters) that fit - for one layout more iterations
    // only add router chains and barriers - and the layouts ranked by cost: a layout that needs router iterations wins
    // where its matmul cores outweigh them (e.g. 10x10 row_tiles_per_core 2 at 1280 rows on a 10-wide grid over 10x8
    // row_tiles_per_core 1 in 5 chunks)
    std::vector<HeadPlanned> out;
    for (const auto& [engine_rows, row_tiles_per_core, row_chunks] : head_layouts(shape)) {
        for (uint32_t kb : k_steps(shape)) {
            // (the fewest router iterations that fit: router_iterations divides the row chunks)
            for (uint32_t nit = min_router_iters; nit <= row_chunks; ++nit) {
                if (row_chunks % nit) {
                    continue;
                }
                const std::vector<bool> compacts = shape.reduce_scatter_dim() == -2 && shape.tp_size <= 4
                                                       ? std::vector<bool>{false, true}
                                                       : std::vector<bool>{false};
                HeadLayout lay;
                HeadSized sized;
                for (bool compact : compacts) {
                    lay = HeadLayout{
                        .engine_rows = engine_rows,
                        .row_tiles_per_core = row_tiles_per_core,
                        .row_chunks = row_chunks,
                        .k_step_tiles = kb,
                        .engine_columns = GXe,
                        .compact_reduce_scatter_zones = compact,
                        .router_iterations = nit};
                    sized = head_size(shape, lay, arena_max(kb));
                    if (sized.eng) {
                        break;
                    }
                }
                if (!sized.eng) {
                    if (keep_unfit) {
                        out.push_back(HeadPlanned{lay, sized.arena, sized.sizes, kInf, {}, sized.reason});
                    }
                    continue;
                }
                auto [t, parts] = head_cost(shape, lay, *sized.eng);
                out.push_back(HeadPlanned{lay, sized.arena, sized.sizes, t, parts, ""});
                break;
            }
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const HeadPlanned& a, const HeadPlanned& b) {
        return std::make_pair(a.predicted_us, -int64_t(a.layout.k_step_tiles)) <
               std::make_pair(b.predicted_us, -int64_t(b.layout.k_step_tiles));
    });
    return out;
}

std::string shape_checks(const HeadShape& shape) {
    const uint32_t G = shape.tp_size, H = shape.hidden_dim;
    if (G < 2) {
        return "TP " + s(G) + ": the MoE head gathers x over the TP axis (TP >= 2)";
    }
    if (G > 4 && shape.latent) {
        return "TP " + s(G) +
               " > 4 with a latent: the latent all-to-all sends a port's destination blocks under one packet header "
               "each - at most 3 (a port's latent block) - a mesh with TP 2 or 4";
    }
    if (G > 8) {
        return "TP " + s(G) + " > 8: the routing / combine port tables (a mesh with TP <= 8)";
    }
    if (shape.grid[0] < 2 || shape.grid[0] > 12) {
        return "a " + s(shape.grid[0]) + "-column compute grid: the head is at most 11 matmul-core columns and the " +
               "movers' column (2 to 12 columns)";
    }
    if (shape.links != 1 && shape.links != 2) {
        return s(shape.links) + " links: 1 or 2 (a port per direction and link)";
    }
    if (H % (32 * G)) {
        return "H " + s(H) + " is not whole tiles over TP " + s(G) + " (a multiple of " + s(32 * G) + ")";
    }
    const uint32_t klt = H / 32 / G;
    if (k_steps(shape).empty()) {
        return "H / 32 / TP = " + s(klt) +
               " tiles a shard: no K step <= 16 gives an even count a shard (an even shard)";
    }
    if (shape.shared_expert_dim < 32 || shape.shared_expert_dim % 32) {
        return "the shared expert's I / TP = " + s(shape.shared_expert_dim) +
               ": whole tiles, at least one (I_s a multiple of " + s(32 * G) + ")";
    }
    if (shape.latent && (*shape.latent % (32 * G))) {
        return "latent " + s(*shape.latent) + " is not whole tiles over TP " + s(G) + " (a multiple of " + s(32 * G) +
               ")";
    }
    if (shape.top_k % 2 || shape.top_k > 16) {
        return "top-K " + s(shape.top_k) +
               ": even and <= 16 (joint routing rows - K ids + K weights in 64 B - dispatch prep, routing exchange)";
    }
    if (shape.router_candidates % 32) {
        return s(shape.router_candidates) + " router candidates: whole tiles (a multiple of 32)";
    }
    if (shape.rows % (32 * G)) {
        const int64_t nxt = ceil_div(shape.rows, 32 * G) * 32 * G;
        return "rows a mesh row " + s(shape.rows) + " must be a multiple of 32 x TP = " + s(32 * G) +
               "; nearest valid " + s(nxt);
    }
    const uint32_t N = shape.num_chips(), S = shape.tokens_per_chip_all_loops(), EPC = shape.experts_per_chip();
    if (N > moe_block_plan_words::kMaxChips || shape.accumulators > moe_block_plan_words::kAccMax) {
        return "routing planner: N " + s(N) + " <= 32 chips, accumulators " + s(shape.accumulators) +
               " <= " + s(moe_block_plan_words::kAccMax);
    }
    if (S > 2048) {
        return "routing planner: " + s(S) + " tokens a chip > 2048 (an entry's token field); chunk <= " + s(2048 * N) +
               " tokens";
    }
    if (EPC > 64 || N * EPC > 1024) {
        return "routing planner: " + s(EPC) + " experts a chip <= 64 and N x EPC " + s(N * EPC) +
               " <= 1024 (fold the router)";
    }
    if (engine_width(shape) < 1) {
        return "no matmul core column: I / TP " + s(shape.shared_expert_dim) + " or latent tiles a chip";
    }
    return "";
}

HeadFit head_plan_fit(const HeadShape& shape, const HeadFitOptions& options) {
    return head_plan_fit(
        shape, arena_max_of(options.l1_free.value_or(kL1Free - kL1Margin), shape.payload, options.cb_floor), options);
}

HeadFit head_plan_fit(const HeadShape& shape, const ArenaMax& arena_max, const HeadFitOptions& options) {
    HeadFit res;
    res.reason = shape_checks(shape);
    if (!res.reason.empty()) {
        return res;
    }
    if (head_layouts(shape).empty()) {
        const uint32_t nrt = shape.rows / 32 / shape.tp_size;
        res.reason =
            "no matmul-core layout for " + s(shape.rows) + " rows a mesh row (" + s(nrt) +
            " row tiles an owner: row chunks of whole owner blocks, row_tiles_per_core <= 2, the router cores' cells)";
        return res;
    }
    const auto ps = head_plan_layouts(shape, arena_max, true, options.min_router_iters);
    std::vector<HeadPlanned> ok;
    for (const auto& p : ps) {
        if (p.predicted_us != kInf) {
            ok.push_back(p);
        }
    }
    if (ok.empty()) {
        std::vector<HeadPlanned> unfit;
        for (const auto& p : ps) {
            if (p.predicted_us == kInf) {
                unfit.push_back(p);
            }
        }
        if (unfit.empty()) {
            res.reason = "no layout";
            return res;
        }
        std::string why = summary(unfit, arena_max);
        if (options.largest && why.find("L1:") != std::string::npos) {
            const uint32_t step = 32 * shape.tp_size;
            int64_t lo = 0, hi = int64_t(shape.rows / step) - 1;
            HeadFitOptions sub = options;
            sub.largest = false;
            sub.explain = false;
            while (lo < hi) {
                const int64_t mid = (lo + hi + 1) / 2;
                HeadShape smaller = shape;
                smaller.rows = uint32_t(mid * step);
                if (head_plan_fit(smaller, arena_max, sub).ok) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            if (lo) {
                why += "; the largest rows a mesh row that fit: " + s(lo * step) + " (chunk " +
                       s(lo * step * shape.sp_rows) + ")";
            }
        }
        res.reason = why;
        return res;
    }
    res.ok = true;
    res.planned = ok;
    if (!options.explain) {
        return res;
    }
    try {
        const auto free = head_plan_layouts(shape, arena_max_fixed(kUnlimited), false);
        if (!free.empty()) {
            res.unconstrained = free.front();
        }
    } catch (const std::exception&) {  // (the unconstrained pick only explains the note)
        res.unconstrained.reset();
    }
    res.note = degradation(res.planned[0], res.unconstrained);
    return res;
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head

#if !defined(__clang__)
#pragma GCC pop_options
#endif
#undef HEAD_PLAN_NO_FP_CONTRACT
