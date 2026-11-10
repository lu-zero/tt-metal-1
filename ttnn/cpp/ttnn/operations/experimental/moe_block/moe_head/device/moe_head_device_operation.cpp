// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_head_device_operation.hpp"
#include <tracy/Tracy.hpp>

#include <tt-metalium/program.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

namespace moe_head_op_detail {

void check_dram(const Tensor& t, const char* what) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr, "moe_head: {} must be on device", what);
    TT_FATAL(t.memory_config().buffer_type() == BufferType::DRAM, "moe_head: {} must be in DRAM", what);
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

}  // namespace moe_head_op_detail
using namespace moe_head_op_detail;

ttsl::hash::hash_t MoEHeadDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& c = attrs.config;
    return tt::tt_metal::operation::hash_operation<MoEHeadDeviceOperation>(
        attrs.plan_key,
        c.layout.engine_rows,
        c.layout.row_tiles_per_core,
        c.layout.row_chunks,
        c.layout.k_step_tiles,
        c.layout.engine_columns,
        c.layout.compact_reduce_scatter_zones,
        c.layout.router_iterations,
        c.shape.rows,
        c.shape.accumulators,
        c.fidelity,
        c.w_dtype,
        c.l1_free,
        c.cb_floor,
        args.x.tensor_spec(),
        args.w.tensor_spec(),
        args.w_router.tensor_spec(),
        args.actual_start.has_value(),
        args.metadata.has_value(),
        attrs.loop,
        attrs.all_loops);
}

MoEHeadDeviceOperation::program_factory_t MoEHeadDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoEHeadFactory{};
}

void MoEHeadDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    TT_FATAL(attrs.model != nullptr, "moe_head: the host model is missing");
    const auto s = head_tensor_shapes(*attrs.model);
    check_dram(args.x, "x");
    const auto& xs = args.x.logical_shape();
    const uint32_t nl = attrs.all_loops ? attrs.loops : 1;  // (the loops the call runs)
    TT_FATAL(
        xs[-2] == attrs.config.shape.rows * nl && xs[-1] == attrs.config.shape.hidden_dim / attrs.config.shape.tp_size,
        "moe_head: x must be [.., {}, {}] a chip, got {}",
        attrs.config.shape.rows * nl,
        attrs.config.shape.hidden_dim / attrs.config.shape.tp_size,
        xs);
    TT_FATAL(args.x.layout() == Layout::TILE && args.x.dtype() == DataType::BFLOAT16, "moe_head: x is bf16 TILE");
    for (const auto* t :
         {&args.w,
          &args.w_router,
          &args.router_bias,
          &args.router_index,
          &args.y,
          &args.rows,
          &args.xbuf,
          &args.route,
          &args.slots,
          &args.counts,
          &args.regions,
          &args.token_index,
          &args.table,
          &args.plan}) {
        check_dram(*t, "an argument");
    }
    TT_FATAL(
        args.actual_start.has_value() == args.actual_end.has_value(),
        "moe_head: a padded chunk takes both actual_start and actual_end");
    TT_FATAL(
        !(args.metadata.has_value() && args.actual_start.has_value()),
        "moe_head: a padded forward as actual_start / actual_end or as metadata, not both");
    TT_FATAL(
        attrs.config.shape.fwd_chunks == 1 || !args.actual_start.has_value(),
        "moe_head: a forward of {} chunks takes the forward's metadata words (one chunk's actual_start / actual_end "
        "cannot describe it)",
        attrs.config.shape.fwd_chunks);
    if (args.metadata) {
        check_dram(*args.metadata, "metadata");
        const uint32_t fw = attrs.config.shape.fwd_chunks, words = fw == 1 ? 3 : 2 + 3 * fw;
        TT_FATAL(
            args.metadata->dtype() == DataType::UINT32 && args.metadata->logical_volume() >= words &&
                !args.metadata->is_sharded(),
            "moe_head: the forward's metadata is {} uint32 words (DRAM interleaved), got {} of {} elements",
            words,
            args.metadata->dtype(),
            args.metadata->logical_volume());
        TT_FATAL(
            args.metadata->buffer()->aligned_page_size() * args.metadata->buffer()->num_pages() >= 4 * words &&
                args.metadata->buffer()->num_pages() == 1,
            "moe_head: the forward's metadata words must be one page");
    }
    for (const auto* v : {&args.actual_start, &args.actual_end}) {
        if (*v) {
            check_dram(**v, "actual_start / actual_end");
            TT_FATAL(
                (*v)->dtype() == DataType::UINT32 && (*v)->logical_volume() == 1 && !(*v)->is_sharded(),
                "moe_head: actual_start / actual_end are one uint32 each (DRAM interleaved), got {} of {} elements",
                (*v)->dtype(),
                (*v)->logical_volume());
        }
    }
    TT_FATAL(
        args.y.logical_shape()[-2] == s.y_rows * nl && args.y.logical_shape()[-1] == s.y_cols,
        "moe_head: y must be [{}, {}]",
        s.y_rows * nl,
        s.y_cols);
    TT_FATAL(
        !attrs.all_loops || attrs.loop % attrs.loops == 0,
        "moe_head: every token loop in one call starts at a split's first loop, got loop {} of {}",
        attrs.loop,
        attrs.loops);
}

void MoEHeadDeviceOperation::validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}

MoEHeadDeviceOperation::spec_return_value_t MoEHeadDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args) {
    return args.y.tensor_spec();
}

MoEHeadDeviceOperation::tensor_return_value_t MoEHeadDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& args) {
    return args.y;
}

HeadBuffers head_buffers(const MoEHeadParams& attrs, const MoEHeadInputs& a) {
    HeadBuffers b;
    TT_FATAL(SRC_WORD0 + attrs.words.size() <= kHeadMaxSrc, "moe_head: {} persistent words", attrs.words.size());
    auto set = [&](HeadSrc s, const Tensor* t) {
        if (t != nullptr) {
            b.buffers[s] = t->buffer();
            b.addresses[s] = static_cast<uint32_t>(t->buffer()->address());
        }
    };
    set(SRC_X, &a.x);
    set(SRC_W, &a.w);
    set(SRC_Y, &a.y);
    set(SRC_ROWS, &a.rows);
    set(SRC_XBUF, &a.xbuf);
    set(SRC_ROUTE, &a.route);
    set(SRC_RW, &a.w_router);
    set(SRC_RB, &a.router_bias);
    set(SRC_RIX, &a.router_index);
    set(SRC_SLOTS, &a.slots);
    set(SRC_COUNTS, &a.counts);
    set(SRC_REGIONS, &a.regions);
    set(SRC_TOKEN_INDEX, &a.token_index);
    set(SRC_TABLE, &a.table);
    set(SRC_PLAN, &a.plan);
    set(SRC_ARENA, &attrs.arena);
    set(SRC_START, a.actual_start ? &*a.actual_start : (a.metadata ? &*a.metadata : nullptr));
    b.meta = a.metadata.has_value();
    set(SRC_END, a.actual_end ? &*a.actual_end : nullptr);
    for (size_t i = 0; i < attrs.words.size(); ++i) {
        b.addresses[SRC_WORD0 + i] = attrs.words[i];
    }
    b.loops = attrs.loops;
    b.loop = attrs.loop % attrs.loops;
    b.split = attrs.loop / attrs.loops;
    if (attrs.all_loops && attrs.loops > 1) {
        // (a loop's rows of an interleaved DRAM tensor l x stride a bank further: whole pages of every bank a loop)
        auto stride = [&](const Tensor& t, const char* what) -> uint32_t {
            TT_FATAL(!t.is_sharded(), "moe_head: token loops in one call take {} interleaved", what);
            const auto* buf = t.buffer();
            const uint32_t pages = buf->num_pages(), nb = buf->allocator()->get_num_banks(buf->buffer_type());
            TT_FATAL(
                pages % (attrs.loops * nb) == 0,
                "moe_head: token loops in one call need {}'s pages ({}) a whole number of every bank's ({} banks) a "
                "loop ({} loops)",
                what,
                pages,
                nb,
                attrs.loops);
            return static_cast<uint32_t>(pages / attrs.loops / nb * buf->aligned_page_size());
        };
        b.merged = true;
        b.x_stride = stride(a.x, "x");
        b.y_stride = stride(a.y, "y");
        const auto specs = head_words(*attrs.model);
        for (size_t i = 0; i < specs.size(); ++i) {
            if (specs[i].name == "loops") {
                b.loop_words = attrs.words.at(i);
            }
        }
        TT_FATAL(b.loop_words != 0, "moe_head: no loop words for token loops in one call");
    }
    return b;
}

tt::tt_metal::WorkloadDescriptor MoEHeadFactory::create_workload_descriptor(
    const MoEHeadParams& attrs,
    const MoEHeadInputs& args,
    Tensor& /*output*/,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    ZoneScopedN("moe_head: create (miss)");
    HeadPatches layout;
    auto w = build_head_workload(*attrs.model, head_buffers(attrs, args), coords_of(tensor_coords), &layout);
    std::lock_guard lock(attrs.patches->mutex);
    if (!attrs.patches->layout) {
        attrs.patches->layout = std::move(layout);
    }
    return w;
}

void MoEHeadFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MoEHeadParams& attrs,
    const MoEHeadInputs& args,
    Tensor& /*output*/,
    const std::optional<ttnn::MeshCoordinate>& /*coord*/) {
    ZoneScopedN("moe_head: override (hit)");
    const HeadBuffers b = head_buffers(attrs, args);
    auto& store = *attrs.patches;
    HeadLastAddresses* last = nullptr;
    const HeadPatches* layout = nullptr;
    {
        // (a program another resources object built: what to rewrite, once - its layout, no tables)
        std::lock_guard lock(store.mutex);
        last = &store.last[&program];  // (a map node: stable while we patch)
        if (!store.layout) {
            const auto& sh = attrs.config.shape;
            const MeshTopology topo{.R = sh.sp_rows, .C = sh.tp_size, .transposed = sh.transposed};
            std::vector<ttnn::MeshCoordinate> all;
            for (uint32_t i = 0; i < sh.num_chips(); ++i) {
                all.push_back(mesh_coord(ChipCoord{i / sh.tp_size, i % sh.tp_size}, topo));
            }
            HeadPatches l;
            build_head_workload(*attrs.model, b, all, &l, /*layout_only=*/true);
            store.layout = std::move(l);
        }
        layout = &*store.layout;
    }
    patch_head_program(program, *layout, b, *last);
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head

namespace ttnn::prim {

Tensor moe_block_head(
    const ttnn::operations::experimental::moe_block::moe_head::MoEHeadParams& params,
    ttnn::operations::experimental::moe_block::moe_head::MoEHeadInputs inputs) {
    using OperationType = ttnn::operations::experimental::moe_block::moe_head::MoEHeadDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(params, std::move(inputs));
}

}  // namespace ttnn::prim
