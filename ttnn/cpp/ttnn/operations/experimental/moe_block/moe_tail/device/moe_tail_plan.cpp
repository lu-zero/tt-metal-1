// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// moe_tail's host plans (moe_tail_plan.hpp has the overview): the pure sizing and scheduling functions first, then
// TailSchedule (the all-to-all's routing and fence counts), the all-to-all plan, and the latent engine's geometry,
// watchers and plan.

#include "moe_tail_plan.hpp"

#include <algorithm>
#include <numeric>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"
#include "ttnn/operations/experimental/moe_block/common/port_placement.hpp"

namespace ttnn::operations::experimental::moe_block::moe_tail {

namespace moe_tail_plan_detail {

using moe_block_tiles::kBf16TileBytes;
using moe_block_tiles::kFp32TileBytes;
using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// len(range(start, stop, step))
uint32_t range_len(uint32_t start, uint32_t stop, uint32_t step) {
    return start < stop ? ceil_div(stop - start, step) : 0;
}

}  // namespace moe_tail_plan_detail
using namespace moe_tail_plan_detail;

std::optional<uint32_t> row_neighbor(uint32_t p, uint32_t tp_size, bool ring, TailDirection d) {
    if (d == TailDirection::Forward) {
        return (ring || p + 1 < tp_size) ? std::optional<uint32_t>((p + 1) % tp_size) : std::nullopt;
    }
    return (ring || p > 0) ? std::optional<uint32_t>((p + tp_size - 1) % tp_size) : std::nullopt;
}

std::pair<std::vector<Unit>, std::vector<Unit>> gather_schedule(uint32_t p, uint32_t tp_size, bool ring) {
    std::vector<Unit> forward, backward;
    if (ring && tp_size % 2 == 0 && tp_size >= 4) {
        const uint32_t half = tp_size / 2;
        for (uint32_t i = 0; i < half; ++i) {
            forward.push_back({(p + tp_size - i) % tp_size, i == half - 1 ? 1u : 0u});
            backward.push_back({(p + i) % tp_size, i == half - 1 ? 2u : 0u});
        }
        return {forward, backward};
    }
    if (ring) {
        const uint32_t num_forward = tp_size / 2, num_backward = tp_size - 1 - num_forward;
        for (uint32_t i = 0; i < num_forward; ++i) {
            forward.push_back({(p + tp_size - i) % tp_size, 0});
        }
        for (uint32_t i = 0; i < num_backward; ++i) {
            backward.push_back({(p + i) % tp_size, 0});
        }
        return {forward, backward};
    }
    if (p + 1 < tp_size) {
        for (uint32_t i = 0; i <= p; ++i) {
            forward.push_back({p - i, 0});
        }
    }
    if (p > 0) {
        for (uint32_t i = 0; i < tp_size - p; ++i) {
            backward.push_back({p + i, 0});
        }
    }
    return {forward, backward};
}

std::vector<uint32_t> chunk_order(uint32_t p, uint32_t tp_size, bool ring) {
    auto distance = [&](uint32_t q) {
        return ring ? std::min((q + tp_size - p) % tp_size, (p + tp_size - q) % tp_size) : (q > p ? q - p : p - q);
    };
    std::vector<uint32_t> out(tp_size);
    std::iota(out.begin(), out.end(), 0u);
    std::stable_sort(out.begin(), out.end(), [&](uint32_t a, uint32_t b) {
        const auto key = [&](uint32_t q) {
            return std::make_tuple(distance(q), (q + tp_size - p) % tp_size != tp_size - 1, q);
        };
        return key(a) < key(b);
    });
    return out;
}

std::vector<std::pair<uint32_t, uint32_t>> even_split(uint32_t n, uint32_t parts) {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    const uint32_t base = n / parts, remainder = n % parts;
    uint32_t k = 0;
    for (uint32_t i = 0; i < parts; ++i) {
        const uint32_t width = base + (i < remainder ? 1 : 0);
        out.emplace_back(k, k + width);
        k += width;
    }
    return out;
}

std::vector<uint32_t> column_split(uint32_t n_tiles, uint32_t columns, uint32_t unit) {
    TT_FATAL(n_tiles % unit == 0, "moe_tail: {} tiles in units of {}", n_tiles, unit);
    const uint32_t base = n_tiles / unit / columns, remainder = n_tiles / unit % columns;
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < columns; ++i) {
        out.push_back((base + (i < remainder ? 1 : 0)) * unit);
    }
    return out;
}

uint32_t ring_pages(uint32_t pages, const std::vector<uint32_t>& pushes) {
    const std::set<uint32_t> sizes(pushes.begin(), pushes.end());
    if (sizes.size() == 1 && pages % *sizes.begin() == 0) {
        return pages;
    }
    return pages + *sizes.rbegin() - 1;
}

// ---- TailSchedule ----

TailSchedule::TailSchedule(
    uint32_t tp_size,
    bool ring,
    uint32_t num_links,
    uint32_t block_tiles,
    uint32_t owned_row_tiles,
    uint32_t num_dram_banks,
    uint32_t pages_per_run,
    std::map<std::pair<uint32_t, uint32_t>, TailDirection> ties) :
    tp_size(tp_size),
    num_links(num_links),
    block_tiles(block_tiles),
    owned_row_tiles(owned_row_tiles),
    num_dram_banks(num_dram_banks),
    pages_per_run(pages_per_run),
    ring(ring),
    ties(std::move(ties)) {
    for (uint32_t p = 0; p < tp_size; ++p) {
        for (TailDirection d : {TailDirection::Forward, TailDirection::Backward}) {
            for (uint32_t link = 0; link < num_links; ++link) {
                for (const auto& [q, part] : destinations(p, d)) {
                    // (my arrivals: the destination's port facing the side my packets come in by; my ready: its port
                    // facing the fabric's way back to me)
                    expected_arrivals_[{q, opposite(d), link}] +=
                        ceil_div(owned_row_tiles * runs(part, link), kArrivalIncrementEvery);
                    readies_[{q, direction(q, p), link}].insert({p, d, link});
                }
            }
        }
    }
}

TailDirection TailSchedule::direction(uint32_t p, uint32_t q) const {
    if (!ring) {
        return q > p ? TailDirection::Forward : TailDirection::Backward;
    }
    const uint32_t forward_distance = (q + tp_size - p) % tp_size;
    if (2 * forward_distance == tp_size) {
        auto it = ties.find({p, q});
        return it == ties.end() ? TailDirection::Forward : it->second;
    }
    return 2 * forward_distance < tp_size ? TailDirection::Forward : TailDirection::Backward;
}

std::vector<Unit> TailSchedule::destinations(uint32_t p, TailDirection d) const {
    std::vector<Unit> out;
    if (!ring) {
        if (d == TailDirection::Forward) {
            for (uint32_t q = p + 1; q < tp_size; ++q) {
                out.push_back({q, 0});
            }
        } else {
            for (uint32_t q = p; q-- > 0;) {
                out.push_back({q, 0});
            }
        }
        return out;
    }
    for (uint32_t k = 1; k < tp_size; ++k) {
        const uint32_t q = d == TailDirection::Forward ? (p + k) % tp_size : (p + tp_size - k) % tp_size;
        if (direction(p, q) == d) {
            out.push_back({q, 0});
        }
    }
    return out;
}

std::vector<std::pair<uint32_t, uint32_t>> TailSchedule::run_list(uint32_t part, uint32_t link) const {
    const uint32_t banks_used = std::min(num_dram_banks, block_tiles);
    const uint32_t my_offsets = range_len(link, banks_used, num_links);
    const uint32_t lo = part == 2 ? my_offsets / 2 : 0, hi = part == 1 ? my_offsets / 2 : my_offsets;
    std::vector<std::pair<uint32_t, uint32_t>> out;
    for (uint32_t i = lo; i < hi; ++i) {
        const uint32_t t0 = link + i * num_links;
        for (uint32_t m = t0; m < block_tiles; m += pages_per_run * num_dram_banks) {
            out.emplace_back(m, std::min(pages_per_run, ceil_div(block_tiles - m, num_dram_banks)));
        }
    }
    return out;
}

uint32_t TailSchedule::hops(uint32_t p, uint32_t q, TailDirection d) const {
    return d == TailDirection::Forward ? (q + tp_size - p) % tp_size : (p + tp_size - q) % tp_size;
}

uint32_t TailSchedule::expected_arrivals(const Port& port) const {
    auto it = expected_arrivals_.find(port);
    return it == expected_arrivals_.end() ? 0 : it->second;
}

std::vector<TailSchedule::Port> TailSchedule::readies(const Port& port) const {
    auto it = readies_.find(port);
    if (it == readies_.end()) {
        return {};
    }
    std::vector<Port> out(it->second.begin(), it->second.end());
    // (sorted by (p, direction, link), Backward first)
    std::sort(out.begin(), out.end(), [](const Port& a, const Port& b) {
        return std::make_tuple(std::get<0>(a), std::get<1>(a) == TailDirection::Forward, std::get<2>(a)) <
               std::make_tuple(std::get<0>(b), std::get<1>(b) == TailDirection::Forward, std::get<2>(b));
    });
    return out;
}

TailAllToAllGeometry TailAllToAllGeometry::make(
    uint32_t tp_size, uint32_t rows, uint32_t hidden_dim, uint32_t num_dram_banks, size_t max_payload_bytes) {
    TailAllToAllGeometry g;
    g.tp_size = tp_size;
    g.rows = rows;
    g.hidden_dim = hidden_dim;
    g.owned_rows = rows / tp_size;
    TT_FATAL(
        rows % tp_size == 0 && g.owned_rows % TILE_HEIGHT == 0 && hidden_dim % (TILE_WIDTH * tp_size) == 0,
        "moe_tail (all-to-all): {} rows, H {} over TP {}: whole tiles",
        rows,
        hidden_dim,
        tp_size);
    g.owned_row_tiles = g.owned_rows / TILE_HEIGHT;
    g.hidden_tiles = hidden_dim / TILE_WIDTH;
    g.block_tiles = g.hidden_tiles / tp_size;
    for (uint32_t d = 1; d <= 8; ++d) {  // (a worker's tile columns: up to 8, the add's DEST tiles at bfloat16)
        if (g.block_tiles % d == 0) {
            g.worker_columns = d;
        }
    }
    g.num_workers = g.hidden_tiles / g.worker_columns;
    g.num_dram_banks = num_dram_banks;
    g.pages_per_run = std::max<uint32_t>(1, static_cast<uint32_t>(max_payload_bytes / kBf16TileBytes));
    // (a port's staging CB: two groups in flight within 96 KB, at most 8 runs a group)
    g.runs_per_group =
        std::max<uint32_t>(1, std::min<uint32_t>(8, (96 * 1024) / (2 * g.pages_per_run * kBf16TileBytes)));
    return g;
}

std::shared_ptr<const TailAllToAllPlan> make_tail_all_to_all_plan(
    MoEBlockResources& resources, uint32_t rows, uint32_t hidden_dim) {
    auto& mesh = resources.mesh_device();
    const auto& topology = resources.topology();
    const uint32_t tp_size = topology.C;
    TT_FATAL(tp_size >= 2, "moe_tail (all-to-all): a TP group (mesh row) of {} chips", tp_size);
    TT_FATAL(
        tp_size - 1 <= kMaxPortDestinations,
        "moe_tail (all-to-all): TP {}: a port sends to at most {} chips",
        tp_size,
        kMaxPortDestinations);
    const auto dram = mesh.dram_grid_size();
    auto plan = std::make_shared<TailAllToAllPlan>();
    plan->geometry =
        TailAllToAllGeometry::make(tp_size, rows, hidden_dim, dram.x * dram.y, resources.fabric_max_payload_bytes());
    plan->topology = topology;
    plan->ring = topology.wrap_x;
    plan->num_links = resources.config().num_links;
    const auto& g = plan->geometry;

    auto node = [&](ChipCoord ch) { return mesh.get_fabric_node_id(mesh_coord(ch, topology)); };
    // The ring's far side (distance tp_size / 2): the way the fabric routes it from p, the same on every mesh row.
    std::map<std::pair<uint32_t, uint32_t>, TailDirection> ties;
    if (plan->ring && tp_size % 2 == 0) {
        for (uint32_t r = 0; r < topology.R; ++r) {
            for (uint32_t p = 0; p < tp_size; ++p) {
                const uint32_t q = (p + tp_size / 2) % tp_size;
                const auto forward =
                    tt::tt_fabric::get_eth_forwarding_direction(node({r, p}), node({r, (p + 1) % tp_size}));
                const auto toward_q = tt::tt_fabric::get_eth_forwarding_direction(node({r, p}), node({r, q}));
                const TailDirection d = toward_q == forward ? TailDirection::Forward : TailDirection::Backward;
                const auto [it, fresh] = ties.emplace(std::make_pair(p, q), d);
                TT_FATAL(fresh || it->second == d, "moe_tail (all-to-all): ring ties differ between mesh rows");
            }
        }
    }
    plan->links = row_links(resources.mesh_plan());
    plan->schedule = TailSchedule(
        tp_size,
        plan->ring,
        plan->num_links,
        g.block_tiles,
        g.owned_row_tiles,
        g.num_dram_banks,
        g.pages_per_run,
        std::move(ties));

    // Ports: one per (direction, link) with a neighbour on the row, next to its link's Ethernet core (NoC1).
    MeshPlan row_plan = resources.mesh_plan();
    for (auto& [chip, dirs] : row_plan.chips) {
        dirs[static_cast<size_t>(Dir::YPlus)].reset();
        dirs[static_cast<size_t>(Dir::YMinus)].reset();
    }
    const auto cores = place_ports(mesh, row_plan);
    std::set<std::pair<size_t, size_t>> taken;
    std::vector<CoreRange> ranges;
    for (const auto& [key, core] : cores) {
        const TailDirection d = key.dir == Dir::XPlus ? TailDirection::Forward : TailDirection::Backward;
        plan->ports[{key.chip, d, key.link_index}] = core;
        if (taken.insert({core.y, core.x}).second) {
            ranges.emplace_back(core, core);
        }
    }
    plan->port_union = CoreRangeSet(ranges);
    const auto grid = mesh.compute_with_storage_grid_size();
    for (uint32_t y = 0; y < grid.y && plan->workers.size() < g.num_workers; ++y) {
        for (uint32_t x = 0; x < grid.x && plan->workers.size() < g.num_workers; ++x) {
            if (!taken.contains({y, x})) {
                plan->workers.emplace_back(x, y);
            }
        }
    }
    TT_FATAL(
        plan->workers.size() == g.num_workers,
        "moe_tail (all-to-all): {} workers, {} free cores",
        g.num_workers,
        plan->workers.size());
    return plan;
}

// ---- latent engine ----

TailEngineGeometry TailEngineGeometry::make(
    uint32_t rows,
    uint32_t latent_dim,
    uint32_t output_dim,
    uint32_t num_chunks,
    uint32_t engine_columns,
    uint32_t engine_rows,
    uint32_t w_tile_bytes,
    uint32_t cap_bytes) {
    TailEngineGeometry g;
    TT_FATAL(engine_rows % 2 == 0, "moe_tail (engine): {} engine rows is odd", engine_rows);
    TT_FATAL(
        latent_dim % (2 * TILE_WIDTH) == 0 && output_dim % TILE_WIDTH == 0,
        "moe_tail (engine): latent_dim {} / output_dim {}: whole tiles of each K half",
        latent_dim,
        output_dim);
    TT_FATAL(
        rows % (TILE_HEIGHT * num_chunks) == 0,
        "moe_tail (engine): {} rows: whole row tiles a chunk ({} chunks)",
        rows,
        num_chunks);
    g.rows = rows;
    g.latent_dim = latent_dim;
    g.output_dim = output_dim;
    g.num_chunks = num_chunks;
    g.engine_columns = engine_columns;
    g.engine_rows = engine_rows;
    g.w_tile_bytes = w_tile_bytes;
    g.latent_tiles = latent_dim / TILE_WIDTH;
    g.output_tiles = output_dim / TILE_WIDTH;
    g.row_tiles_per_chunk = rows / TILE_HEIGHT / num_chunks;
    g.num_pairs = engine_rows / 2;
    const uint32_t row_tiles = num_chunks * g.row_tiles_per_chunk;
    for (uint32_t p = 0; p < g.num_pairs; ++p) {
        g.steps.push_back(range_len(p, row_tiles, g.num_pairs));
        g.own_steps.push_back(range_len(p, g.row_tiles_per_chunk, g.num_pairs));
    }
    g.max_steps = *std::max_element(g.steps.begin(), g.steps.end());
    g.max_own_steps = *std::max_element(g.own_steps.begin(), g.own_steps.end());
    g.k_halves = even_split(g.latent_tiles, 2);
    g.k_half_tiles = g.k_halves[0].second;
    TT_FATAL(g.k_halves[1].second - g.k_halves[1].first == g.k_half_tiles, "moe_tail (engine): equal K halves");
    g.column_tiles = column_split(g.output_tiles, engine_columns, 1);
    uint32_t first_tile = 0;
    for (uint32_t c = 0; c < engine_columns; ++c) {
        g.column_first_tile.push_back(first_tile);
        first_tile += g.column_tiles[c];
    }
    g.max_column_tiles = *std::max_element(g.column_tiles.begin(), g.column_tiles.end());
    // (a step's column tiles in DEST at once: 8 fp32 tiles at full sync, one kept for the scale)
    TT_FATAL(g.max_column_tiles <= 7, "moe_tail (engine): a step's {} fp32 tiles in DEST", g.max_column_tiles);
    // (the columns a core finishes: half 0 the first ceil(tiles / 2), half 1 the rest - at most kExchangeTiles - 1
    // either way)
    std::set<uint32_t> counts;
    for (uint32_t tiles : g.column_tiles) {
        const uint32_t first_half = ceil_div(tiles, 2);
        g.finished_columns.push_back({std::make_pair(0u, first_half), std::make_pair(first_half, tiles - first_half)});
        for (const auto& [offset, count] : g.finished_columns.back()) {
            TT_FATAL(
                count > 0 && count <= kExchangeTiles - 1 && tiles - count <= kExchangeTiles - 1,
                "moe_tail (engine): {} columns a core",
                tiles);
            counts.insert(count);
        }
    }
    g.finished_counts.assign(counts.begin(), counts.end());
    uint32_t lcm = 1;
    for (uint32_t count : g.finished_counts) {
        lcm = std::lcm(lcm, count);
    }
    g.finished_ring_pages = lcm * ceil_div(2 * g.finished_counts.back(), lcm);
    g.w_k_ranges = even_split(g.k_half_tiles, g.num_pairs);
    g.share_k_ranges = even_split(g.k_half_tiles, engine_columns);
    for (const auto& [k_first, k_end] : g.share_k_ranges) {
        g.max_share_tiles = std::max(g.max_share_tiles, k_end - k_first);
    }
    // the z ring and the squares slots: a slot a step where L1 has room (no slot reused: no free waits), else fewer
    const uint32_t lowest = std::min<uint32_t>(2, g.max_steps);
    uint32_t slots = g.max_steps, total = 0;
    for (;; --slots) {
        total = g.layout(slots).second;
        if (total <= cap_bytes || slots == lowest) {
            break;
        }
    }
    TT_FATAL(
        total <= cap_bytes,
        "moe_tail (engine): its L1 {} KB > {} KB at {} ring slots",
        total >> 10,
        cap_bytes >> 10,
        lowest);
    g.ring_slots = slots;
    g.arena_bytes = total;
    return g;
}

std::pair<std::vector<uint32_t>, uint32_t> TailEngineGeometry::layout(uint32_t slots) const {
    auto align = [](uint32_t v) { return (v + 63) / 64 * 64; };
    const std::vector<uint32_t> sizes = {
        k_half_tiles * max_column_tiles * w_tile_bytes,    // up weights
        slots * k_half_tiles * kBf16TileBytes,             // ring
        max_steps * max_share_tiles * kBf16TileBytes,      // landing
        slots * engine_columns * kFp32TileBytes,           // row squares
        kFp32TileBytes,                                    // my squares
        kBf16TileBytes,                                    // ones
        finished_ring_pages * kFp32TileBytes,              // finished
        2 * kExchangeTiles * kFp32TileBytes,               // exchange send
        2 * kFp32TileBytes,                                // sum of squares
        max_steps * kExchangeTiles * kFp32TileBytes,       // exchange receive
        finished_ring_pages * kBf16TileBytes,              // shared expert
        finished_ring_pages * kBf16TileBytes,              // output
        max_own_steps * max_share_tiles * kBf16TileBytes,  // own rows
    };
    std::vector<uint32_t> offsets;
    uint32_t o = 0;
    for (uint32_t v : sizes) {
        offsets.push_back(o);
        o = align(o + v);
    }
    return {offsets, o};
}

uint32_t TailEngineGeometry::region_tiles() const {
    uint32_t m = 0;
    for (const auto& [k_first, k_end] : w_k_ranges) {
        m = std::max(m, k_end - k_first);
    }
    return m * max_column_tiles;
}

std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> engine_watchers(
    uint32_t p, uint32_t tp_size, bool ring, const TailEngineGeometry& g, uint32_t group_columns) {
    std::set<Unit> units;  // (every block landed here by a port: from either neighbour)
    for (uint32_t c = 0; c < tp_size; ++c) {
        const auto [forward, backward] = gather_schedule(c, tp_size, ring);
        if (row_neighbor(c, tp_size, ring, TailDirection::Forward) == p) {
            units.insert(forward.begin(), forward.end());
        }
        if (row_neighbor(c, tp_size, ring, TailDirection::Backward) == p) {
            units.insert(backward.begin(), backward.end());
        }
    }
    const auto order = chunk_order(p, g.num_chunks, ring);
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> out;
    for (const auto& [q, part] : units) {
        const uint32_t first_column = part == 2 ? g.engine_columns / 2 : 0;
        const uint32_t end_column = part == 1 ? g.engine_columns / 2 : g.engine_columns;
        const uint32_t position = static_cast<uint32_t>(std::find(order.begin(), order.end(), q) - order.begin());
        for (uint32_t r = 0; r < g.row_tiles_per_chunk; ++r) {
            const uint32_t row_tile = position * g.row_tiles_per_chunk + r, pair = row_tile % g.num_pairs,
                           step = row_tile / g.num_pairs;
            for (uint32_t c0 = first_column; c0 < end_column; c0 += group_columns) {
                const uint32_t c1 = std::min(c0 + group_columns, end_column);
                for (uint32_t half = 0; half < 2; ++half) {
                    if (c1 - c0 > 1) {
                        out[{c1 - 1, pair + g.num_pairs * half}].push_back(step | c0 << 8 | c1 << 16);
                    }
                }
            }
        }
    }
    for (auto& [k, v] : out) {
        TT_FATAL(v.size() <= std::min<uint32_t>(g.max_steps, 32), "moe_tail (engine): {} watched steps", v.size());
        std::sort(v.begin(), v.end());
    }
    return out;
}

std::shared_ptr<const TailEnginePlan> make_tail_engine_plan(
    MoEBlockResources& resources,
    uint32_t rows,
    uint32_t latent_dim,
    uint32_t output_dim,
    uint32_t w_tile_bytes,
    uint32_t cap_bytes) {
    auto& mesh = resources.mesh_device();
    const auto& topology = resources.topology();
    const auto grid = mesh.compute_with_storage_grid_size();
    // (the engine: every row of the grid, an even count of them, and every column but the last)
    const uint32_t engine_rows = grid.y / 2 * 2;
    TT_FATAL(
        grid.x >= 2 && engine_rows >= 4,
        "moe_tail (engine): a {}x{} grid (at least 2 columns, 4 rows)",
        grid.x,
        grid.y);
    TT_FATAL(
        resources.config().num_links == 2,
        "moe_tail (engine): {} links (a link a K half)",
        resources.config().num_links);
    const uint32_t engine_columns = grid.x - 1;  // (the last column: the gather's ports)
    auto plan = std::make_shared<TailEnginePlan>();
    plan->geometry = TailEngineGeometry::make(
        rows, latent_dim, output_dim, topology.C, engine_columns, engine_rows, w_tile_bytes, cap_bytes);
    plan->topology = topology;
    plan->ring = topology.wrap_x;
    plan->num_links = 2;
    plan->num_dram_banks = mesh.dram_grid_size().x;
    plan->tiles_per_packet =
        std::max<uint32_t>(1, static_cast<uint32_t>(resources.fabric_max_payload_bytes() / kBf16TileBytes));
    plan->norm_eps = resources.dims().norm_eps;
    plan->links = row_links(resources.mesh_plan());
    // (the ports' column has no row with room for a placement search: placed down it - forward above, backward below,
    // a link apart)
    const std::array<uint32_t, 4> port_rows = {
        engine_rows / 10, 3 * engine_rows / 10, 6 * engine_rows / 10, 8 * engine_rows / 10};
    std::vector<CoreRange> ranges;
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t link = 0; link < 2; ++link) {
            const CoreCoord core(engine_columns, port_rows[2 * i + link]);
            plan->ports[{i == 0 ? TailDirection::Forward : TailDirection::Backward, link}] = core;
            ranges.emplace_back(core, core);
        }
    }
    plan->port_cores = CoreRangeSet(ranges);
    plan->engine_cores = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(engine_columns - 1, engine_rows - 1)));
    const auto& g = plan->geometry;
    // (a port core holds the landing CB (its base names the engine's landing slots), its staging (three groups: one
    // sending, one landing, one being issued), the group counts and the poll words)
    const uint32_t port_bytes = g.max_steps * g.max_share_tiles * kBf16TileBytes +
                                3 * kPacketsPerGroup * plan->tiles_per_packet * kBf16TileBytes + 64 + 16 +
                                g.row_tiles_per_chunk * g.engine_columns * 16;
    TT_FATAL(
        port_bytes <= cap_bytes,
        "moe_tail (engine): a port core's L1 {} KB > {} KB",
        port_bytes >> 10,
        cap_bytes >> 10);
    return plan;
}

std::map<std::pair<ChipCoord, TailDirection>, std::vector<uint32_t>> row_links(const MeshPlan& plan) {
    std::map<std::pair<ChipCoord, TailDirection>, std::vector<uint32_t>> out;
    for (const auto& [chip, ports] : plan.chips) {
        for (TailDirection d : {TailDirection::Forward, TailDirection::Backward}) {
            if (const auto& port = ports[static_cast<size_t>(mesh_dir(d))]) {
                out[{chip, d}] = port->links;
            }
        }
    }
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block::moe_tail
