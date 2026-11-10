// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// worker_output_writer: an all-to-all worker's writer - own block to the output, the rest to staging, then z-ready
// Runs on: all-to-all worker cores (plan.workers), RISCV_0 (BRISC), NoC 0
// Responsibilities:
//  - reads the compute's tiles from CB_OUTPUT (L1)
//  - writes my own block's tiles to the output and the other chips' tiles to the staging buffer (DRAM) over NoC 0;
//    skip_padding: zeros into its share of the output's padding row tiles
//  - per token loop waits on its ZERO_GATE_SEM semaphore for every carrying port's release, then +1 on each carrying
//    port's z-ready word of the row tile (port_staging_reader); staging reuse across loops is ordered by that gate
//    (the ports zero their words and release only once done reading the previous loop's staging)
// Note: NoC 0 so the staging buffer lands through the DRAM endpoint port_staging_reader reads it through (BH cross-port
//       stale read)

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_tail/device/kernels/all_to_all/padded_row_tiles.hpp"

// (moe_tail_program_factory's all_to_all_cb: the compute's output, the z-ready words' CB - the same address on the
// ports; the zero gate's semaphore id)
constexpr uint32_t CB_OUTPUT = 3, CB_Z_READY = 5, ZERO_GATE_SEM = 0;

void kernel_main() {
    moe_block_args::load();  // (my args: this chip's block of the DRAM table, common/args_table.hpp)
    constexpr uint32_t worker_columns = get_compile_time_arg_val(0);
    constexpr auto output_args = TensorAccessorArgs<1>();
    constexpr auto staging_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    // token loops: their count and, a loop, the output's rows that much further in every bank (bytes)
    constexpr uint32_t token_loops = get_compile_time_arg_val(staging_args.next_compile_time_args_offset());
    constexpr uint32_t output_loop_stride_bytes =
        get_compile_time_arg_val(staging_args.next_compile_time_args_offset() + 1);
    constexpr bool skip_padding = get_named_compile_time_arg_val("skip_padding") != 0;
    constexpr uint32_t tile_bytes = moe_block_tiles::kBf16TileBytes;
    size_t a = 0;
    const uint32_t output_address = moe_block_args::arg(a++), staging_address = moe_block_args::arg(a++);
    const uint32_t first_column = moe_block_args::arg(a++), owned_row_tiles = moe_block_args::arg(a++);
    const uint32_t hidden_tiles = moe_block_args::arg(a++), block_tiles = moe_block_args::arg(a++);
    const uint32_t my_position = moe_block_args::arg(a++), z_ready_address = get_write_ptr(CB_Z_READY);
    const uint32_t padding_args = a;  // (the padding args, at a fixed place)
    if constexpr (skip_padding) {
        a += 7;
    }
    const uint32_t num_carrying_ports = moe_block_args::arg(a++);
    const uint32_t carrying_ports = a;  // (x | y << 16 each)
    const auto staging = TensorAccessor(staging_args, staging_address, tile_bytes);
    const uint32_t my_block_owner = first_column / block_tiles;  // (the TP position my columns' block goes to)
    for (uint32_t loop = 0; loop < token_loops; ++loop) {
        const auto output = TensorAccessor(output_args, output_address + loop * output_loop_stride_bytes, tile_bytes);
        if (num_carrying_ports > 0) {  // (every carrying port's z-ready words zeroed for this loop; re-armed here)
            volatile tt_l1_ptr uint32_t* gate =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ZERO_GATE_SEM));
            noc_semaphore_wait_min(gate, num_carrying_ports);
            noc_semaphore_inc(get_noc_addr(get_semaphore(ZERO_GATE_SEM)), 0u - num_carrying_ports);
            noc_async_atomic_barrier();
        }
        PaddedRows padded_rows{};
        uint32_t first_row = 0;
        uint64_t live_mask = 0;
        if constexpr (skip_padding) {
            padded_rows =
                read_padded_rows(padding_args, get_write_ptr(CB_PADDING_SCRATCH) + PADDING_SCRATCH_WRITER_OFFSET);
            first_row = first_owned_row(padding_args, loop);
            live_mask = live_row_tile_mask(padded_rows, first_row, owned_row_tiles);
        }
        for (uint32_t r = 0; r < owned_row_tiles; ++r) {
            if (!row_tile_is_live<skip_padding>(live_mask, r)) {
                continue;  // (nor counted in: the ports skip it too)
            }
            cb_wait_front(CB_OUTPUT, worker_columns);
            const uint32_t tiles_l1 = get_read_ptr(CB_OUTPUT);
            for (uint32_t t = 0; t < worker_columns; ++t) {
                const uint64_t destination = my_block_owner == my_position
                                                 ? output.get_noc_addr(
                                                       (my_position * owned_row_tiles + r) * block_tiles +
                                                       first_column - my_block_owner * block_tiles + t)
                                                 : staging.get_noc_addr(r * hidden_tiles + first_column + t);
                noc_async_write(tiles_l1 + t * tile_bytes, destination, tile_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_OUTPUT, worker_columns);
            for (uint32_t i = 0; i < num_carrying_ports; ++i) {
                const uint32_t xy = moe_block_args::arg(carrying_ports + i);
                noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, z_ready_address + 16 * r), 1);
            }
        }
        if constexpr (skip_padding) {
            // my output's padding row tiles - every chip's owned rows of my SP row, my block's columns: zeros, the
            // units (row tile, worker_columns columns) round-robin over the workers; nothing else writes them
            const uint32_t column_groups = block_tiles / worker_columns, num_workers = hidden_tiles / worker_columns;
            const uint32_t worker_index = first_column / worker_columns;
            const uint32_t units = (hidden_tiles / block_tiles) * owned_row_tiles * column_groups;
            const uint32_t sp_row_first_row = first_row - my_position * tt::constants::TILE_HEIGHT * owned_row_tiles;
            const uint32_t zero_tile = get_write_ptr(CB_PADDING_SCRATCH) + PADDING_SCRATCH_ZERO_TILE_OFFSET;
            bool filled = false;
            for (uint32_t u = worker_index; u < units; u += num_workers) {
                const uint32_t row_tile = u / column_groups, column_group = u % column_groups;
                if (padded_tile_real(padded_rows, sp_row_first_row, row_tile)) {
                    continue;
                }
                if (!filled) {
                    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_tile);
                    for (uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
                        words[i] = 0;
                    }
                    filled = true;
                }
                for (uint32_t t = 0; t < worker_columns; ++t) {
                    noc_async_write(
                        zero_tile,
                        output.get_noc_addr(row_tile * block_tiles + column_group * worker_columns + t),
                        tile_bytes);
                }
            }
            noc_async_write_barrier();
        }
        noc_async_atomic_barrier();
    }
}
