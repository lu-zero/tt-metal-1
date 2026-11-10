// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE tail's program, one for the mesh (TailEngineFactory: the latent engine, kernels/latent_engine/;
// TailAllToAllFactory: the all-to-all, kernels/all_to_all/):
//  * built a chip at a time, then every data-movement kernel's per-core args go to a per-chip DRAM table, read into
//    an args CB at the kernel's start (common/args_table.hpp): the chips' programs are then identical - one program;
//  * no per-call arena tensor: the engine's regions are program CBs; the landing CB comes first, on the engine and
//    the port cores, so it sits at the same address there and on every chip, and the kernels read the slots' bases
//    with get_write_ptr instead of taking them as runtime args;
//  * the fence's ready / arrival words are persistent word tensors of the resources (not global semaphores);
//  * the all-to-all creates no port on a direction without a neighbour.

#include <map>
#include <tracy/Tracy.hpp>
#include "moe_tail_program_factory.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <set>
#include <unordered_map>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::moe_block::moe_tail {

namespace moe_tail_factory_detail {

using tt::tt_metal::Buffer;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::DataMovementProcessor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::NOC;
using tt::tt_metal::ProgramDescriptor;

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_tail/device/kernels/";
using moe_block_tiles::kBf16TileBytes;
using moe_block_tiles::kFp32TileBytes;
using tt::constants::TILE_HEIGHT;

// The latent engine's CB ids (engine_common.hpp, engine_norm_projection_compute.cpp and gather_port_common.hpp use the
// same names).
namespace engine_cb {
enum : uint8_t {
    CB_UP_WEIGHTS = 0,        // W' (resident: its K ranges pushed in order, never popped)
    CB_RING = 1,              // the row's shares of a step: all of my K half (ring_slots slots)
    CB_ROW_SQUARES = 2,       // the row's squares tiles of a step (fp32, ring_slots slots)
    CB_MY_SQUARES = 3,        // my share's squares (fp32)
    CB_ONES = 4,              // the all-ones tile of the squares' row sums
    CB_FINISHED = 5,          // my finished columns' partials (fp32)
    CB_EXCHANGE_SEND = 6,     // the partner's columns' partials + my half's sum of squares (fp32)
    CB_SUM_OF_SQUARES = 7,    // my half's sum of squares (fp32)
    CB_EXCHANGE_RECEIVE = 8,  // the partner's exchange (fp32, a slot a step of a loop)
    CB_SHARED_EXPERT = 9,
    CB_OUTPUT = 10,
    CB_LANDING = 11,   // the landing slots of every step: own steps tilized here, the others landed by the gather
    CB_OWN_ROWS = 12,  // my own shares, row-major
    CB_PORT_STAGING = 13,
    CB_PORT_PACKET_COUNTS = 14,
    CB_PORT_POLL = 15,
    CB_STEP_WORDS = 16,  // the engine's step words, this call's (zeroed each call): after the landing CB, one address
    CB_ARGS = 17,        // the data-movement kernels' args blocks (one_program): the last CB, sized to them
};
}  // namespace engine_cb
// The all-to-all's (the all_to_all kernels use the same names).
namespace all_to_all_cb {
enum : uint8_t {
    CB_ROWS = 0,  // a worker's owned rows, row-major
    CB_TILIZED = 1,
    CB_SHARED_EXPERT = 2,
    CB_OUTPUT = 3,
    CB_PORT_STAGING = 4,     // port_staging_reader -> port_fabric_sender
    CB_Z_READY = 5,          // the z-ready words, a row tile's 16 B apart
    CB_PADDING_SCRATCH = 6,  // a padded forward: the span's words, the workers' zero tile (padded_row_tiles.hpp)
    CB_LIVE_ROW_TILES = 7,   // a padded forward: the workers' reader -> compute hand-off of a loop's live row tiles
    CB_ARGS = 8,             // the data-movement kernels' args blocks (one_program): the last CB, sized to them
};
}  // namespace all_to_all_cb

KernelDescriptor kernel(const std::string& file, const CoreRangeSet& cores, KernelDescriptor::ConfigDescriptor config) {
    KernelDescriptor k;
    k.kernel_source = std::string(kKernelDir) + file;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.config = std::move(config);
    return k;
}

DataMovementConfigDescriptor dm(DataMovementProcessor risc, NOC noc) {
    return DataMovementConfigDescriptor{.processor = risc, .noc = noc};
}

CBDescriptor cb(uint8_t index, const CoreRangeSet& cores, uint32_t total, tt::DataFormat fmt, uint32_t page) {
    return CBDescriptor{
        .total_size = total,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{.buffer_index = index, .data_format = fmt, .page_size = page}},
    };
}

// (TailEngine's CB check: a block addressed from the CB's pointer must not straddle its end - every core pushes one
// size there and the dataflow ends read page by page, so each size must divide the CB)
void check_blocks(const char* name, uint32_t size, uint32_t page, const std::vector<uint32_t>& blocks) {
    for (uint32_t blk : std::set<uint32_t>(blocks.begin(), blocks.end())) {
        TT_FATAL(
            ring_pages(size / page, {blk}) == size / page,
            "moe_tail (engine): CB {} ({} pages) takes blocks of {}: one would straddle its end",
            name,
            size / page,
            blk);
    }
}

void append_accessor(KernelDescriptor& k, const Buffer* b) {
    tt::tt_metal::TensorAccessorArgs(*b).append_to(k.compile_time_args);
}

uint32_t u32(size_t v) { return static_cast<uint32_t>(v); }
uint32_t xy(const CoreCoord& v) { return u32(v.x | v.y << 16); }

std::vector<CoreCoord> cores_of_set(const CoreRangeSet& s) {
    std::vector<CoreCoord> v;
    for (const auto& r : s.ranges()) {
        for (auto y = r.start_coord.y; y <= r.end_coord.y; ++y) {
            for (auto x = r.start_coord.x; x <= r.end_coord.x; ++x) {
                v.emplace_back(x, y);
            }
        }
    }
    return v;
}

CoreRangeSet cells(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    for (const auto& c : cores) {
        ranges.emplace_back(c, c);
    }
    return CoreRangeSet(ranges);
}

std::vector<ttnn::MeshCoordinate> coords_of(const ttnn::MeshCoordinateRangeSet& set) {
    std::vector<ttnn::MeshCoordinate> out;
    for (const auto& range : set.ranges()) {
        for (const auto& c : range) {
            out.push_back(c);
        }
    }
    return out;
}

uint32_t index_of(const std::vector<uint32_t>& v, uint32_t x) {
    return static_cast<uint32_t>(std::find(v.begin(), v.end(), x) - v.begin());
}

// The buffers the runtime args address, by source (address_patches.hpp; nullptr: absent).
std::array<Buffer*, 9> sources_of(const MoETailInputs& a, const Tensor& output) {
    const auto opt = [](const std::optional<Tensor>& t) { return t.has_value() ? t->buffer() : nullptr; };
    return {
        a.y_own.buffer(),
        opt(a.shared_expert),
        opt(a.w_up),
        output.buffer(),
        a.ready.buffer(),
        opt(a.arrival),
        opt(a.staging),
        a.metadata.has_value() ? a.metadata->buffer() : opt(a.actual_start),
        opt(a.actual_end)};
}

// The cached workload of the program descriptors, with each program's address patches.
TailEngineFactory::cached_mesh_workload_t to_cached(
    tt::tt_metal::WorkloadDescriptor descriptors,
    const MoETailInputs& args,
    const Tensor& output,
    std::shared_ptr<const void> args_tables) {
    const auto sources = sources_of(args, output);
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, TailEngineFactory::shared_variables_t> shared;
    for (auto& [range, desc] : descriptors.programs) {
        TailEngineFactory::shared_variables_t sv{
            .patches = take_address_patches(desc, sources), .args_tables = args_tables};
        workload.add_program(range, tt::tt_metal::Program(desc));
        shared.emplace(range, std::move(sv));
    }
    return {std::move(workload), std::move(shared)};
}

void patch_cached(TailEngineFactory::cached_mesh_workload_t& cached, const MoETailInputs& args, const Tensor& output) {
    const auto sources = sources_of(args, output);
    for (auto& [range, sv] : cached.shared_variables) {
        apply_address_patches(cached.workload.get_programs().at(range), sv.patches, sources);
    }
}

// Token loops in one call (attrs.loops k > 1): every kernel of both tails loops over them itself, loop l's rows of
// y_own / shared_expert / the output l x loop_stride further (compile-time args).
// A token loop's rows of an interleaved DRAM tensor: whole pages of every bank, so loop l's rows start l x the returned
// bytes further in every bank (0 with one loop).
uint32_t loop_stride(const Tensor& t, uint32_t k, const char* what) {
    if (k <= 1) {
        return 0;
    }
    const auto* b = t.buffer();
    const uint32_t pages = b->num_pages(), nb = b->allocator()->get_num_banks(b->buffer_type());
    TT_FATAL(
        !t.is_sharded() && pages % (k * nb) == 0,
        "moe_tail: token loops in one call need {}'s pages ({}) interleaved, a whole number of every bank's ({} "
        "banks) a loop ({} loops)",
        what,
        pages,
        nb,
        k);
    return static_cast<uint32_t>(pages / k / nb * b->aligned_page_size());
}

// Two data-movement kernels on the same cores (their indices in the program).
struct DmPair {
    size_t reader, writer;
    CoreRangeSet cores;
};

// The chips' programs as one program for the mesh (common/args_table.hpp): each data-movement kernel's per-core args
// go to a per-chip DRAM table; at start the kernel loads its core's block into CB `cb_id`, added here on each pair's
// cores and sized to that pair's blocks (the reader's at 0, the writer's right after). The compute kernels keep their
// runtime args (the same on every chip).
std::pair<tt::tt_metal::WorkloadDescriptor, std::shared_ptr<const void>> one_program(
    tt::tt_metal::WorkloadDescriptor w,
    tt::tt_metal::distributed::MeshDevice& mesh,
    uint8_t cb_id,
    const std::vector<DmPair>& pairs,
    const char* mode) {
    std::vector<std::pair<ttnn::MeshCoordinate, ProgramDescriptor>> chips;
    for (auto& prog : w.programs) {
        TT_FATAL(prog.range.shape().mesh_size() == 1, "moe_tail ({}): one program a chip before the args move", mode);
        chips.emplace_back(prog.range.start_coord(), std::move(prog.descriptor));
    }
    // a kernel's block: its longest core's args on any chip, whole 64 B (as args_table.cpp pages it)
    auto block = [&](size_t k) {
        size_t words = 0;
        for (const auto& [coord, p] : chips) {
            for (const auto& [core, a] : p.kernels[k].runtime_args) {
                words = std::max(words, a.size());
            }
        }
        return std::max<uint32_t>(64, (4 * u32(words) + 63) / 64 * 64);
    };
    std::vector<std::optional<ArgsScratch>> scratch(chips[0].second.kernels.size());
    for (const auto& [rd, wr, cores] : pairs) {
        const uint32_t br = block(rd), bw = block(wr);
        scratch[rd] = ArgsScratch{.cb = cb_id, .offset = 0, .bytes = br};
        scratch[wr] = ArgsScratch{.cb = cb_id, .offset = br, .bytes = bw};
        log_debug(tt::LogOp, "moe_tail ({}): kernels {} / {}: args blocks {} + {} B", mode, rd, wr, br, bw);
        for (auto& [coord, p] : chips) {
            p.cbs.push_back(cb(cb_id, cores, br + bw, tt::DataFormat::UInt32, br + bw));
        }
    }
    std::string why;
    auto tables = runtime_args_to_dram(mesh, chips, scratch, why);
    TT_FATAL(tables != nullptr, "moe_tail ({}): the kernels' args to DRAM: {}", mode, why);
    for (size_t i = 0; i < chips.size(); ++i) {
        w.programs[i].descriptor = std::move(chips[i].second);
    }
    std::string merge_why;
    auto one = merge_chip_workload(std::move(w), "moe_tail", &merge_why);
    TT_FATAL(
        one.programs.size() == 1,
        "moe_tail ({}): the chips' programs still differ with their args in DRAM: {}",
        mode,
        merge_why);
    return {std::move(one), std::move(tables)};
}

tt::tt_metal::WorkloadDescriptor engine_workload(
    const MoETailParams& attrs,
    const MoETailInputs& args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords);
tt::tt_metal::WorkloadDescriptor all_to_all_workload(
    const MoETailParams& attrs,
    const MoETailInputs& args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords);

}  // namespace moe_tail_factory_detail
using namespace moe_tail_factory_detail;

TailEngineFactory::cached_mesh_workload_t TailEngineFactory::create_mesh_workload(
    const MoETailParams& attrs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const MoETailInputs& tensor_args,
    Tensor& output) {
    // (token loops: the engine's kernels loop themselves - no merged per-loop programs)
    const auto& plan = *attrs.engine_plan;
    // (kernels: 0 reader, 1 writer, 2 compute on the engine; 3 port reader, 4 port writer on the ports)
    auto [w, tables] = one_program(
        engine_workload(attrs, tensor_args, output, tensor_coords),
        *tensor_args.y_own.device(),
        engine_cb::CB_ARGS,
        {{0, 1, plan.engine_cores}, {3, 4, plan.port_cores}},
        "engine");
    return to_cached(std::move(w), tensor_args, output, std::move(tables));
}

void TailEngineFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached, const MoETailParams& /*attrs*/, const MoETailInputs& tensor_args, Tensor& output) {
    ZoneScopedN("moe_tail: override (hit, engine)");
    patch_cached(cached, tensor_args, output);
}

TailAllToAllFactory::cached_mesh_workload_t TailAllToAllFactory::create_mesh_workload(
    const MoETailParams& attrs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const MoETailInputs& tensor_args,
    Tensor& output) {
    // (token loops: the all-to-all kernels loop themselves - no merged per-loop programs)
    const auto& plan = *attrs.all_to_all_plan;
    // (kernels: 0 reader, 1 compute, 2 writer on the workers; 3 port reader, 4 port writer on the ports' union)
    auto [w, tables] = one_program(
        all_to_all_workload(attrs, tensor_args, output, tensor_coords),
        *tensor_args.y_own.device(),
        all_to_all_cb::CB_ARGS,
        {{0, 2, cells(plan.workers)}, {3, 4, plan.port_union}},
        "all-to-all");
    return to_cached(std::move(w), tensor_args, output, std::move(tables));
}

void TailAllToAllFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached, const MoETailParams& /*attrs*/, const MoETailInputs& tensor_args, Tensor& output) {
    ZoneScopedN("moe_tail: override (hit)");
    patch_cached(cached, tensor_args, output);
}

// ---------------------------------------------------------------------------------------------------------------------
// TailEngine
// ---------------------------------------------------------------------------------------------------------------------

tt::tt_metal::WorkloadDescriptor moe_tail_factory_detail::engine_workload(
    const MoETailParams& attrs,
    const MoETailInputs& args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    using namespace engine_cb;
    const auto& plan = *attrs.engine_plan;
    const auto& g = plan.geometry;
    auto* mesh = args.y_own.device();
    const uint32_t engine_columns = g.engine_columns, engine_rows = g.engine_rows, num_pairs = g.num_pairs;
    const uint32_t tp_size = g.num_chunks;
    const auto w_format = tt::tt_metal::datatype_to_dataformat_converter(args.w_up->dtype());
    const auto bf16 = tt::DataFormat::Float16_b, fp32 = tt::DataFormat::Float32;

    Buffer* y = args.y_own.buffer();
    Buffer* w = args.w_up->buffer();
    Buffer* se = args.shared_expert->buffer();
    Buffer* out = output.buffer();
    Buffer* ready = args.ready.buffer();
    // token loops: every kernel loops over them itself (loop l's rows of y_own / se / the output l x stride further)
    const uint32_t loops = std::max<uint32_t>(1, attrs.loops);
    const uint32_t sy = loop_stride(args.y_own, loops, "y_own");
    const uint32_t ss = loop_stride(*args.shared_expert, loops, "shared_expert");
    const uint32_t so = loop_stride(output, loops, "the output");

    auto virt = [&](uint32_t column, uint32_t row) {
        return mesh->worker_core_from_logical_core(CoreCoord(column, row));
    };
    auto node = [&](uint32_t r, uint32_t c) {
        return mesh->get_fabric_node_id(mesh_coord(ChipCoord{r, c}, plan.topology));
    };
    auto schedule = [&](uint32_t p, TailDirection d) {
        if (!row_neighbor(p, tp_size, plan.ring, d)) {
            return std::vector<Unit>{};
        }
        const auto s = gather_schedule(p, tp_size, plan.ring);
        return d == TailDirection::Forward ? s.first : s.second;
    };

    const CoreRangeSet& engine = plan.engine_cores;
    const CoreRangeSet& ports = plan.port_cores;
    const CoreRangeSet both = engine.merge(ports);
    // (a port's staging: three groups - one sending, one landing, one being issued)
    const uint32_t port_staging_pages = 3 * kPacketsPerGroup * plan.tiles_per_packet;

    // CBs: the landing slots first (engine and ports: one address everywhere), then the engine's, then the ports'.
    std::vector<CBDescriptor> cbs;
    auto engine_buffer = [&](uint8_t i,
                             const char* name,
                             uint32_t size,
                             tt::DataFormat format,
                             uint32_t page,
                             std::vector<uint32_t> blocks) {
        check_blocks(name, size, page, blocks);
        cbs.push_back(cb(i, i == CB_LANDING ? both : engine, size, format, page));
    };
    engine_buffer(
        CB_LANDING,
        "landing",
        g.max_steps * g.max_share_tiles * kBf16TileBytes,
        bf16,
        kBf16TileBytes,
        {g.max_share_tiles});
    // (the step words right after it, also on engine and ports: 16 B each - engine_common.hpp has the layout)
    const uint32_t step_words_bytes = 16 * (3 * g.max_steps + 1 + g.max_steps * engine_columns);
    cbs.push_back(cb(CB_STEP_WORDS, both, step_words_bytes, tt::DataFormat::UInt32, step_words_bytes));
    engine_buffer(
        CB_UP_WEIGHTS,
        "up_weights",
        g.k_half_tiles * g.max_column_tiles * g.w_tile_bytes,
        w_format,
        g.w_tile_bytes,
        {1});
    engine_buffer(
        CB_RING, "ring", g.ring_slots * g.k_half_tiles * kBf16TileBytes, bf16, kBf16TileBytes, {g.k_half_tiles});
    engine_buffer(
        CB_ROW_SQUARES,
        "row_squares",
        g.ring_slots * engine_columns * kFp32TileBytes,
        fp32,
        kFp32TileBytes,
        {engine_columns});
    engine_buffer(CB_MY_SQUARES, "my_squares", kFp32TileBytes, fp32, kFp32TileBytes, {1});
    engine_buffer(CB_ONES, "ones", kBf16TileBytes, bf16, kBf16TileBytes, {1});
    engine_buffer(
        CB_FINISHED, "finished", g.finished_ring_pages * kFp32TileBytes, fp32, kFp32TileBytes, g.finished_counts);
    engine_buffer(
        CB_EXCHANGE_SEND, "exchange_send", 2 * kExchangeTiles * kFp32TileBytes, fp32, kFp32TileBytes, {kExchangeTiles});
    engine_buffer(CB_SUM_OF_SQUARES, "sum_of_squares", 2 * kFp32TileBytes, fp32, kFp32TileBytes, {1});
    engine_buffer(
        CB_EXCHANGE_RECEIVE,
        "exchange_receive",
        g.max_steps * kExchangeTiles * kFp32TileBytes,
        fp32,
        kFp32TileBytes,
        {kExchangeTiles});
    engine_buffer(
        CB_SHARED_EXPERT,
        "shared_expert",
        g.finished_ring_pages * kBf16TileBytes,
        bf16,
        kBf16TileBytes,
        g.finished_counts);
    engine_buffer(CB_OUTPUT, "output", g.finished_ring_pages * kBf16TileBytes, bf16, kBf16TileBytes, g.finished_counts);
    // (my own shares' row-major source: a whole max_share_tiles block a push (my share's tiles in it), so a block never
    // straddles the CB's end, loop after loop)
    engine_buffer(
        CB_OWN_ROWS,
        "own_rows",
        g.max_own_steps * g.max_share_tiles * kBf16TileBytes,
        bf16,
        kBf16TileBytes,
        {g.max_share_tiles});
    // the ports' staging, their group packet counts (a 16 B page each) and their poll words (16 B a share)
    cbs.push_back(cb(CB_PORT_STAGING, ports, port_staging_pages * kBf16TileBytes, bf16, kBf16TileBytes));
    cbs.push_back(cb(CB_PORT_PACKET_COUNTS, ports, 4 * 16, tt::DataFormat::UInt32, 16));
    cbs.push_back(
        cb(CB_PORT_POLL, ports, 16 + g.row_tiles_per_chunk * engine_columns * 16, tt::DataFormat::UInt32, 16));

    // the engine's dataflow kernels' compile-time args 0 - 10 (engine_common.hpp)
    const std::vector<uint32_t> engine_compile_args = {
        g.k_half_tiles,
        g.max_column_tiles,
        g.row_tiles_per_chunk,
        g.num_chunks,
        engine_columns,
        g.w_tile_bytes,
        g.ring_slots,
        g.latent_tiles,
        kExchangeTiles,
        num_pairs,
        g.max_steps};
    std::vector<uint32_t> w_k_bounds;
    for (const auto& [k_first, k_end] : g.w_k_ranges) {
        w_k_bounds.push_back(k_first);
    }
    w_k_bounds.push_back(g.k_half_tiles);
    std::vector<uint32_t> share_k_ranges;
    for (const auto& [k_first, k_end] : g.share_k_ranges) {
        share_k_ranges.push_back(k_first);
        share_k_ranges.push_back(k_end);
    }
    std::vector<uint32_t> column_xs;  // (the engine's NoC x by column)
    for (uint32_t c = 0; c < engine_columns; ++c) {
        column_xs.push_back(u32(virt(c, 0).x));
    }
    // (the gates: lead core (0, 0), the engine's NoC rectangle, its cores)
    const CoreCoord lead = virt(0, 0), far_corner = virt(engine_columns - 1, engine_rows - 1);
    const std::vector<uint32_t> gate = {
        xy(lead),
        u32(std::min(lead.x, far_corner.x)),
        u32(std::min(lead.y, far_corner.y)),
        u32(std::max(lead.x, far_corner.x)),
        u32(std::max(lead.y, far_corner.y)),
        engine_columns * engine_rows};
    const uint32_t w_region_tiles = g.region_tiles();
    std::vector<uint32_t> port_xys;  // (the port cores the zero gate releases once each: x | y << 16)
    for (const auto& range : ports.ranges()) {
        for (const auto& core : range) {
            port_xys.push_back(xy(virt(core.x, core.y)));
        }
    }

    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& coord : coords_of(tensor_coords)) {
        const ChipCoord chip = chip_of(coord, plan.topology);
        const uint32_t r = chip.r, p = chip.c;
        const auto order = chunk_order(p, g.num_chunks, plan.ring);
        const auto watchers = engine_watchers(p, tp_size, plan.ring, g);

        ProgramDescriptor desc;
        desc.cbs = ProgramDescriptor::CBDescriptors(cbs.begin(), cbs.end());
        // (the ports' zero-gate go, PORT_READER_GO_SEM / PORT_WRITER_GO_SEM: ids 0 and 1, before the fabric
        // connections take the next free ones)
        for (uint32_t id = 0; id < 2; ++id) {
            desc.semaphores.push_back(
                tt::tt_metal::SemaphoreDescriptor{.id = id, .core_ranges = ports, .initial_value = 0});
        }

        auto reader =
            kernel("latent_engine/engine_dram_reader.cpp", engine, dm(DataMovementProcessor::RISCV_0, NOC::NOC_0));
        reader.compile_time_args = engine_compile_args;
        append_accessor(reader, y);
        append_accessor(reader, se);
        reader.compile_time_args.insert(reader.compile_time_args.end(), {loops, sy, ss, g.max_share_tiles});
        auto writer = kernel(
            "latent_engine/engine_row_exchange_writer.cpp", engine, dm(DataMovementProcessor::RISCV_1, NOC::NOC_1));
        writer.compile_time_args = engine_compile_args;
        append_accessor(writer, out);
        writer.compile_time_args.insert(writer.compile_time_args.end(), {loops, so});
        ComputeConfigDescriptor compute_config{
            .math_fidelity = attrs.math_fidelity,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .math_approx_mode = false,
        };
        compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
        for (uint8_t i : {CB_FINISHED, CB_SUM_OF_SQUARES, CB_EXCHANGE_RECEIVE}) {
            compute_config.unpack_to_dest_mode[i] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        }
        auto compute = kernel("latent_engine/engine_norm_projection_compute.cpp", engine, compute_config);
        compute.compile_time_args = {
            g.k_half_tiles,
            engine_columns,
            g.max_share_tiles,
            std::bit_cast<uint32_t>(static_cast<float>(1.0 / g.latent_dim)),
            std::bit_cast<uint32_t>(plan.norm_eps),
            kExchangeTiles,
            loops,
            g.max_steps};

        for (uint32_t column = 0; column < engine_columns; ++column) {
            const uint32_t column_tiles = g.column_tiles[column];
            const auto [share_k_first, share_k_end] = g.share_k_ranges[column];
            for (uint32_t row = 0; row < engine_rows; ++row) {
                const uint32_t pair = row % num_pairs, half = row / num_pairs;
                const auto [finished_offset, finished_count] = g.finished_columns[column][half];
                const CoreCoord core(column, row);
                const uint32_t noc_x = u32(virt(column, row).x), noc_y = u32(virt(column, row).y);
                const uint32_t half_column_top = u32(virt(column, half * num_pairs).y);
                const uint32_t half_column_bottom = u32(virt(column, half * num_pairs + num_pairs - 1).y);
                const auto [w_k_first, w_k_end] = g.w_k_ranges[pair];
                const uint32_t region = (half * engine_columns + column) * num_pairs + pair;

                KernelDescriptor::RTArgList reader_args;
                for (uint32_t x : {pair, column_tiles, g.k_halves[half].first}) {
                    reader_args.push_back(x);
                }
                reader_args.push_back(w);
                for (uint32_t x :
                     {region % plan.num_dram_banks,
                      (region / plan.num_dram_banks) * w_region_tiles * g.w_tile_bytes,
                      w_k_first,
                      w_k_end,
                      noc_x,
                      half_column_top,
                      half_column_bottom}) {
                    reader_args.push_back(x);
                }
                reader_args.push_back(y);
                reader_args.push_back(share_k_first);
                reader_args.push_back(share_k_end);
                reader_args.push_back(se);
                for (uint32_t x :
                     {g.output_tiles,
                      g.column_first_tile[column] + finished_offset,
                      finished_count,
                      g.steps[pair],
                      g.own_steps[pair]}) {
                    reader_args.push_back(x);
                }
                reader_args.append(gate);
                reader_args.push_back(static_cast<uint32_t>(column == 0 && row == 0));
                if (column == 0 && row == 0) {  // (the zero gate's lead releases the port cores)
                    reader_args.push_back(static_cast<uint32_t>(port_xys.size()));
                    reader_args.append(port_xys);
                } else {
                    reader_args.push_back(0u);
                }
                reader_args.append(order);
                reader.emplace_runtime_args(core, reader_args);

                KernelDescriptor::RTArgList writer_args;
                for (uint32_t x :
                     {pair,
                      column_tiles,
                      g.column_first_tile[column] + finished_offset,
                      finished_count,
                      share_k_first,
                      share_k_end,
                      u32(virt(0, row).x),
                      u32(virt(engine_columns - 1, row).x),
                      column}) {
                    writer_args.push_back(x);
                }
                writer_args.push_back(out);
                for (uint32_t x :
                     {g.output_tiles,
                      g.max_share_tiles,
                      noc_x,
                      noc_y,
                      u32(virt(column, (row + num_pairs) % engine_rows).y)}) {
                    writer_args.push_back(x);
                }
                writer_args.push_back(g.steps[pair]);
                writer_args.push_back(g.own_steps[pair]);
                writer_args.append(order);
                for (uint32_t c = 0; c < engine_columns; ++c) {
                    writer_args.push_back(u32(virt(c, row).x));
                }
                writer_args.append(w_k_bounds);
                const auto it = watchers.find({column, row});
                std::vector<uint32_t> watched = it == watchers.end() ? std::vector<uint32_t>{} : it->second;
                writer_args.push_back(static_cast<uint32_t>(watched.size()));
                watched.resize(g.max_steps, 0);  // (a fixed length)
                writer_args.append(watched);
                // (the gate's lead: every writer counts in there after each loop but the last)
                writer_args.push_back(xy(lead));
                writer.emplace_runtime_args(core, writer_args);

                std::vector<uint32_t> compute_args = {
                    column_tiles,
                    finished_offset,
                    finished_count,
                    share_k_first,
                    share_k_end,
                    g.steps[pair],
                    g.own_steps[pair],
                    0};
                compute_args.insert(compute_args.end(), w_k_bounds.begin() + 1, w_k_bounds.end());
                compute.runtime_args.emplace_back(core, std::move(compute_args));
            }
        }

        // ---- the gather's ports ---- (compile-time args: gather_port_common.hpp)
        const std::vector<uint32_t> port_compile_args = {
            g.row_tiles_per_chunk,
            engine_columns,
            plan.tiles_per_packet,
            g.max_share_tiles,
            g.latent_tiles,
            kPacketsPerGroup,
            num_pairs,
            kLandedGroupColumns,
            CB_PORT_STAGING,
            CB_PORT_PACKET_COUNTS,
            CB_PORT_POLL,
            CB_LANDING,
            loops};
        auto port_reader = kernel(
            "latent_engine/gather_port_staging_reader.cpp", ports, dm(DataMovementProcessor::RISCV_1, NOC::NOC_0));
        port_reader.compile_time_args = port_compile_args;
        auto port_writer = kernel(
            "latent_engine/gather_port_fabric_sender.cpp", ports, dm(DataMovementProcessor::RISCV_0, NOC::NOC_1));
        port_writer.compile_time_args = port_compile_args;
        for (const auto& [key, core] : plan.ports) {
            const auto [direction, link] = key;
            const auto peer = row_neighbor(p, tp_size, plan.ring, direction);
            const auto sends = schedule(p, direction);
            std::vector<uint32_t> pair_ys;
            for (uint32_t q = 0; q < num_pairs; ++q) {
                pair_ys.push_back(u32(virt(0, q + num_pairs * link).y));
            }
            KernelDescriptor::RTArgList reader_args;
            reader_args.push_back(static_cast<uint32_t>(sends.size()));
            for (const auto& u : sends) {
                reader_args.push_back(u.part | index_of(order, u.pos) << 8);
            }
            reader_args.append(share_k_ranges);
            reader_args.append(column_xs);
            reader_args.append(pair_ys);
            // (the gate's lead: every port reader counts in there after each loop but the last)
            reader_args.push_back(xy(lead));
            port_reader.emplace_runtime_args(core, reader_args);

            // (my ready to the peer's port that writes into this chip; it waits for it before writing my landing slots)
            const bool send_ready = peer.has_value() && !schedule(*peer, opposite(direction)).empty();
            const CoreCoord peer_port = plan.ports.at({opposite(direction), link});
            const CoreCoord ready_core = send_ready ? virt(peer_port.x, peer_port.y) : virt(0, 0);
            KernelDescriptor::RTArgList sender_args;
            sender_args.push_back(ready);
            for (uint32_t x : {static_cast<uint32_t>(send_ready), u32(ready_core.x), u32(ready_core.y)}) {
                sender_args.push_back(x);
            }
            const bool linked = peer.has_value() && (!sends.empty() || send_ready);
            std::vector<uint32_t> fabric_args;
            if (linked) {
                const auto peer_node = node(r, *peer);
                const auto peer_order = chunk_order(*peer, g.num_chunks, plan.ring);
                sender_args.push_back(*peer_node.mesh_id);
                sender_args.push_back(static_cast<uint32_t>(peer_node.chip_id));
                sender_args.push_back(static_cast<uint32_t>(sends.size()));
                for (const auto& u : sends) {
                    sender_args.push_back(u.part | index_of(peer_order, u.pos) << 8);
                }
                const auto& links = plan.links.at({ChipCoord{r, p}, direction});
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    *mesh, node(r, p), peer_node, links.at(link), desc, core, fabric_args);
            } else {
                for (uint32_t x : {0u, 0u, 0u}) {
                    sender_args.push_back(x);
                }
            }
            sender_args.append(share_k_ranges);
            sender_args.append(column_xs);
            sender_args.append(pair_ys);
            sender_args.append(fabric_args);
            port_writer.emplace_runtime_args(core, sender_args);
        }

        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(writer));
        desc.kernels.push_back(std::move(compute));
        desc.kernels.push_back(std::move(port_reader));
        desc.kernels.push_back(std::move(port_writer));
        // (engine_common.hpp: PARTNER_EXCHANGE, SLOTS_FREE, W_LOADED, W_GATE_COUNT, W_GATE_GO, LOOP_GATE_COUNT,
        // LOOP_GATE_GO)
        for (uint32_t id = 0; id < 7; ++id) {
            desc.semaphores.push_back(
                tt::tt_metal::SemaphoreDescriptor{.id = id, .core_ranges = engine, .initial_value = 0});
        }

        workload.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(desc)});
    }
    return workload;
}

// ---------------------------------------------------------------------------------------------------------------------
// all-to-all
// ---------------------------------------------------------------------------------------------------------------------
tt::tt_metal::WorkloadDescriptor moe_tail_factory_detail::all_to_all_workload(
    const MoETailParams& attrs,
    const MoETailInputs& args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    using namespace all_to_all_cb;
    const auto& plan = *attrs.all_to_all_plan;
    const auto& g = plan.geometry;
    const auto& schedule = plan.schedule;
    auto* mesh = args.y_own.device();
    const auto bf16 = tt::DataFormat::Float16_b;

    Buffer* y = args.y_own.buffer();
    Buffer* se = args.shared_expert.has_value() ? args.shared_expert->buffer() : nullptr;
    Buffer* out = output.buffer();
    Buffer* staging = args.staging->buffer();
    Buffer* arrival = args.arrival->buffer();
    Buffer* ready = args.ready.buffer();

    auto virt = [&](const CoreCoord& c) { return mesh->worker_core_from_logical_core(c); };
    auto node = [&](uint32_t r, uint32_t c) {
        return mesh->get_fabric_node_id(mesh_coord(ChipCoord{r, c}, plan.topology));
    };
    const CoreRangeSet workers = cells(plan.workers);
    std::vector<uint32_t> worker_xys;  // (x | y << 16 each: the carrying ports release them once a call)
    for (const auto& c : plan.workers) {
        worker_xys.push_back(xy(virt(c)));
    }
    TT_FATAL(
        worker_xys.size() == g.num_workers,
        "moe_tail (all-to-all): {} workers, {} planned",
        worker_xys.size(),
        g.num_workers);
    const uint32_t worker_columns = g.worker_columns, has_shared_expert = se != nullptr ? 1u : 0u;
    // token loops: every kernel loops over them itself (loop l's rows of y / se / the output: l x stride further)
    const uint32_t loops = std::max<uint32_t>(1, attrs.loops);
    const uint32_t sy = loop_stride(args.y_own, loops, "y_own");
    const uint32_t ss = se != nullptr ? loop_stride(*args.shared_expert, loops, "shared_expert") : 0u;
    const uint32_t so = loop_stride(output, loops, "the output");
    // a padded forward: the padding row tiles skipped (padded_row_tiles.hpp) - the span's buffers and each chip's
    // padding args
    Buffer* span_first = args.metadata.has_value()       ? args.metadata->buffer()
                         : args.actual_start.has_value() ? args.actual_start->buffer()
                                                         : nullptr;
    Buffer* span_end = args.metadata.has_value() || !args.actual_end.has_value() ? nullptr : args.actual_end->buffer();
    const bool skip_padding = span_first != nullptr;
    TT_FATAL(
        !skip_padding || g.owned_row_tiles <= 64,
        "moe_tail (all-to-all): a padded forward's {} row tiles a chip (at most 64)",
        g.owned_row_tiles);
    TT_FATAL(
        g.block_tiles % worker_columns == 0,
        "moe_tail (all-to-all): a worker's {} columns straddle a chip's block of {}",
        worker_columns,
        g.block_tiles);
    auto padding_args = [&](KernelDescriptor::RTArgList& list, uint32_t sp_row, uint32_t p) {
        list.push_back(span_first);
        if (span_end != nullptr) {
            list.push_back(span_end);
        } else {
            list.push_back(0u);
        }
        const uint32_t call_rows = TILE_HEIGHT * g.owned_row_tiles * g.tp_size;  // (a loop's rows of my SP row)
        for (uint32_t x :
             {sp_row,
              plan.topology.R,
              attrs.row_tokens,
              attrs.loop * call_rows + p * TILE_HEIGHT * g.owned_row_tiles,
              call_rows}) {
            list.push_back(x);
        }
    };
    // skip_padding: a named compile-time arg of every all-to-all kernel; the padded forward's chunks / metadata stay
    // defines (moe_head's padded_rows.hpp, shared with the head, reads them)
    auto padding_config = [&](KernelDescriptor& k) {
        k.named_compile_time_args.emplace_back("skip_padding", skip_padding ? 1u : 0u);
        if (skip_padding) {
            k.defines.emplace_back("PADDED_FORWARD_CHUNKS", std::to_string(attrs.forward_chunks));
            if (args.metadata.has_value()) {
                k.defines.emplace_back("PADDED_FORWARD_METADATA", "1");
            }
        }
    };

    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& coord : coords_of(tensor_coords)) {
        const ChipCoord chip = chip_of(coord, plan.topology);
        const uint32_t r = chip.r, p = chip.c;
        // (this chip's port cores; the port kernels and their CB go on every chip's - the union, the same program on
        // every chip (one program for the mesh): a core that is no port of this chip runs them idle)
        std::vector<std::pair<TailSchedule::Port, CoreCoord>> my_ports;
        std::set<std::pair<size_t, size_t>> mine;
        for (const auto& [key, core] : plan.ports) {
            if (std::get<0>(key) == chip) {
                my_ports.push_back({{p, std::get<1>(key), std::get<2>(key)}, core});
                mine.insert({core.x, core.y});
            }
        }
        const CoreRangeSet& port_cores = plan.port_union;

        ProgramDescriptor desc;
        // (the z-ready words, a row tile's 16 B apart: this call's, zeroed by each carrying port - the first CB on the
        // workers and the ports, so one address on all of them: a worker's get_write_ptr is the ports')
        desc.cbs.push_back(
            cb(CB_Z_READY,
               workers.merge(port_cores),
               16 * g.owned_row_tiles,
               tt::DataFormat::UInt32,
               16 * g.owned_row_tiles));
        // (every block worker_columns pages - the port's pages_per_run x runs_per_group: no block straddles a ring's
        // end)
        desc.cbs.push_back(cb(
            CB_ROWS, workers, ring_pages(2 * worker_columns, {worker_columns}) * kBf16TileBytes, bf16, kBf16TileBytes));
        desc.cbs.push_back(cb(
            CB_TILIZED, workers, ring_pages(worker_columns, {worker_columns}) * kBf16TileBytes, bf16, kBf16TileBytes));
        desc.cbs.push_back(
            cb(CB_SHARED_EXPERT,
               workers,
               ring_pages(2 * worker_columns, {worker_columns}) * kBf16TileBytes,
               bf16,
               kBf16TileBytes));
        desc.cbs.push_back(
            cb(CB_OUTPUT,
               workers,
               ring_pages(2 * worker_columns, {worker_columns}) * kBf16TileBytes,
               bf16,
               kBf16TileBytes));
        desc.cbs.push_back(cb(
            CB_PORT_STAGING,
            port_cores,
            ring_pages(2 * g.runs_per_group * g.pages_per_run, {g.runs_per_group * g.pages_per_run}) * kBf16TileBytes,
            bf16,
            kBf16TileBytes));
        if (skip_padding) {
            // (128 B pages: the readers' metadata words, the writers' 256 B on, the workers' zero tile 512 B on -
            // padded_row_tiles.hpp)
            desc.cbs.push_back(
                cb(CB_PADDING_SCRATCH, workers.merge(port_cores), 512 + kBf16TileBytes, tt::DataFormat::UInt32, 128));
            desc.cbs.push_back(cb(CB_LIVE_ROW_TILES, workers, 128, tt::DataFormat::UInt32, 128));
        }

        auto worker_reader =
            kernel("all_to_all/worker_input_reader.cpp", workers, dm(DataMovementProcessor::RISCV_1, NOC::NOC_1));
        worker_reader.compile_time_args = {worker_columns, has_shared_expert};
        append_accessor(worker_reader, y);
        append_accessor(worker_reader, se != nullptr ? se : y);
        worker_reader.compile_time_args.insert(worker_reader.compile_time_args.end(), {loops, sy, ss});
        auto worker_compute = kernel(
            "all_to_all/worker_tilize_add_compute.cpp",
            workers,
            ComputeConfigDescriptor{
                .math_fidelity = tt::tt_metal::MathFidelity::HiFi2,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false});
        worker_compute.compile_time_args = {worker_columns, has_shared_expert, loops};
        // (NOC0: the staging buffer lands in DRAM through the NOC0 endpoint the port reader reads it through - a read
        // through the other NoC endpoint can return a page's previous contents after the write's ack, the BH DRAM
        // cross-port stale read; the worker reader, NOC1, reads only an earlier program's y and shared expert)
        auto worker_writer =
            kernel("all_to_all/worker_output_writer.cpp", workers, dm(DataMovementProcessor::RISCV_0, NOC::NOC_0));
        worker_writer.compile_time_args = {worker_columns};
        append_accessor(worker_writer, out);
        append_accessor(worker_writer, staging);
        worker_writer.compile_time_args.insert(worker_writer.compile_time_args.end(), {loops, so});
        // (the port kernels' compile-time args 0 - 5: port_common.hpp)
        const std::vector<uint32_t> port_compile_args = {
            CB_PORT_STAGING,
            kBf16TileBytes,
            g.pages_per_run,
            g.num_dram_banks,
            g.runs_per_group,
            kArrivalIncrementEvery};
        auto port_reader =
            kernel("all_to_all/port_staging_reader.cpp", port_cores, dm(DataMovementProcessor::RISCV_1, NOC::NOC_0));
        port_reader.compile_time_args = port_compile_args;
        append_accessor(port_reader, staging);
        port_reader.compile_time_args.push_back(loops);
        auto port_sender =
            kernel("all_to_all/port_fabric_sender.cpp", port_cores, dm(DataMovementProcessor::RISCV_0, NOC::NOC_1));
        port_sender.compile_time_args = port_compile_args;
        append_accessor(port_sender, out);
        port_sender.compile_time_args.insert(port_sender.compile_time_args.end(), {loops, so});
        for (auto* k : {&worker_reader, &worker_compute, &worker_writer, &port_reader, &port_sender}) {
            padding_config(*k);
        }

        for (const auto& [port, core] : my_ports) {
            const auto [position, direction, link] = port;
            const auto destinations = schedule.destinations(p, direction);
            TT_FATAL(
                destinations.size() <= kMaxPortDestinations,
                "moe_tail (all-to-all): a port's {} destinations",
                destinations.size());
            KernelDescriptor::RTArgList reader_args;
            reader_args.push_back(staging);
            reader_args.push_back(g.num_workers);
            for (uint32_t x :
                 {g.owned_row_tiles,
                  g.hidden_tiles,
                  g.block_tiles,
                  link,
                  plan.num_links,
                  static_cast<uint32_t>(destinations.size())}) {
                reader_args.push_back(x);
            }
            if (skip_padding) {  // (at a fixed place: the same args bind the same buffers on every chip)
                padding_args(reader_args, r, p);
            }
            for (const auto& u : destinations) {
                reader_args.push_back(u.pos | u.part << 8);
            }
            reader_args.append(worker_xys);  // (the workers it releases once its z-ready words are zeroed)
            port_reader.emplace_runtime_args(core, reader_args);

            const auto readies = schedule.readies(port);
            std::set<uint32_t> destination_chips;
            for (const auto& u : destinations) {
                destination_chips.insert(u.pos);
            }
            KernelDescriptor::RTArgList sender_args;
            sender_args.push_back(out);
            for (uint32_t x : {p, g.owned_row_tiles, g.block_tiles, link, plan.num_links}) {
                sender_args.push_back(x);
            }
            sender_args.push_back(arrival);
            sender_args.push_back(schedule.expected_arrivals(port));
            sender_args.push_back(ready);
            sender_args.push_back(static_cast<uint32_t>(destination_chips.size()));
            if (skip_padding) {
                padding_args(sender_args, r, p);
            }
            sender_args.push_back(static_cast<uint32_t>(readies.size()));
            for (const auto& [ready_position, ready_direction, ready_link] : readies) {
                const auto n = node(r, ready_position);
                sender_args.push_back(
                    xy(virt(plan.ports.at({ChipCoord{r, ready_position}, ready_direction, ready_link}))));
                sender_args.push_back(static_cast<uint32_t>(n.chip_id));
                sender_args.push_back(*n.mesh_id);
                sender_args.push_back(schedule.hops(p, ready_position, direction));
            }
            sender_args.push_back(static_cast<uint32_t>(destinations.size()));
            for (const auto& u : destinations) {
                const auto n = node(r, u.pos);
                sender_args.push_back(u.pos | u.part << 8);
                sender_args.push_back(static_cast<uint32_t>(n.chip_id));
                sender_args.push_back(*n.mesh_id);
                sender_args.push_back(schedule.hops(p, u.pos, direction));
                sender_args.push_back(g.owned_row_tiles * schedule.runs(u.part, link));
                sender_args.push_back(xy(virt(plan.ports.at({ChipCoord{r, u.pos}, opposite(direction), link}))));
            }
            if (!destinations.empty() || !readies.empty()) {
                const auto peer = row_neighbor(p, g.tp_size, plan.ring, direction);
                TT_FATAL(
                    peer.has_value(),
                    "moe_tail (all-to-all): ({}, {}) has no {} link",
                    r,
                    p,
                    direction == TailDirection::Forward ? "forward" : "backward");
                const auto& links = plan.links.at({chip, direction});
                std::vector<uint32_t> fabric_args;
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    *mesh, node(r, p), node(r, *peer), links.at(link), desc, core, fabric_args);
                sender_args.append(fabric_args);
            }
            port_sender.emplace_runtime_args(core, sender_args);
        }
        // (the union's cores that are no port of this chip: idle - no destination, no ready, nothing expected)
        for (const auto& core : cores_of_set(port_cores)) {
            if (mine.contains({core.x, core.y})) {
                continue;
            }
            KernelDescriptor::RTArgList reader_args;
            reader_args.push_back(staging);
            reader_args.push_back(g.num_workers);
            for (uint32_t x : {g.owned_row_tiles, g.hidden_tiles, g.block_tiles, 0u, plan.num_links, 0u}) {
                reader_args.push_back(x);
            }
            if (skip_padding) {
                padding_args(reader_args, r, p);
            }
            port_reader.emplace_runtime_args(core, reader_args);
            KernelDescriptor::RTArgList sender_args;
            sender_args.push_back(out);
            for (uint32_t x : {p, g.owned_row_tiles, g.block_tiles, 0u, plan.num_links}) {
                sender_args.push_back(x);
            }
            sender_args.push_back(arrival);
            sender_args.push_back(0u);
            sender_args.push_back(ready);
            sender_args.push_back(0u);
            if (skip_padding) {
                padding_args(sender_args, r, p);
            }
            sender_args.push_back(0u);  // (no ready)
            sender_args.push_back(0u);  // (no destination)
            port_sender.emplace_runtime_args(core, sender_args);
        }

        // (the ports that carry something: the workers' z-ready words; a port with no destination neither waits nor
        // re-arms)
        std::vector<uint32_t> carrying_port_xys;
        for (const auto& [port, core] : my_ports) {
            if (!schedule.destinations(p, std::get<1>(port)).empty()) {
                carrying_port_xys.push_back(xy(virt(core)));
            }
        }
        for (uint32_t i = 0; i < plan.workers.size(); ++i) {
            const CoreCoord& core = plan.workers[i];
            const uint32_t first_column = i * worker_columns;
            KernelDescriptor::RTArgList reader_args;
            reader_args.push_back(y);
            if (se != nullptr) {
                reader_args.push_back(se);
            } else {
                reader_args.push_back(0u);
            }
            for (uint32_t x : {first_column, g.owned_row_tiles, g.hidden_tiles}) {
                reader_args.push_back(x);
            }
            if (skip_padding) {
                padding_args(reader_args, r, p);
            }
            worker_reader.emplace_runtime_args(core, reader_args);
            worker_compute.runtime_args.emplace_back(core, std::vector<uint32_t>{g.owned_row_tiles});
            KernelDescriptor::RTArgList writer_args;
            writer_args.push_back(out);
            writer_args.push_back(staging);
            for (uint32_t x : {first_column, g.owned_row_tiles, g.hidden_tiles, g.block_tiles, p}) {
                writer_args.push_back(x);
            }
            if (skip_padding) {
                padding_args(writer_args, r, p);
            }
            writer_args.push_back(static_cast<uint32_t>(carrying_port_xys.size()));
            writer_args.append(carrying_port_xys);
            worker_writer.emplace_runtime_args(core, writer_args);
        }

        desc.kernels.push_back(std::move(worker_reader));
        desc.kernels.push_back(std::move(worker_compute));
        desc.kernels.push_back(std::move(worker_writer));
        desc.kernels.push_back(std::move(port_reader));
        desc.kernels.push_back(std::move(port_sender));
        // (the workers' zero gate, ZERO_GATE_SEM)
        desc.semaphores.push_back(
            tt::tt_metal::SemaphoreDescriptor{.id = 0, .core_ranges = workers, .initial_value = 0});
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(desc)});
    }
    return workload;
}
}  // namespace ttnn::operations::experimental::moe_block::moe_tail
