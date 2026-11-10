// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The shared expert's matmul cores' compute kernel (one kernel a column group): the router's logits matmul, then per
// row chunk gate / up with the activation (h), down + the reduce-scatter's adds, and the dispatch prep's untilize.
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/token_loop.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/shared_expert_macros.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack_untilize.h"
#include "api/debug/kernel_profiler.hpp"
#ifdef SKIP_PADDING_CHUNKS
#include "internal/tt-1xx/cache.h"
#endif
#ifdef TRISC_PACK
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "ckernel_sfpu_situ_glu.h"
#include "ckernel_sfpu_clamped_silu_glu.h"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/swiglu_sfpu.h"
namespace ckernel::sfpu {
// (the model's activation constants, compile-time: SiTU's betas, the clamp limit, SwiGLU-OAI's alpha / limit; their
// defines come with the kernel source - head_program.cpp glu_defines)
struct SharedExpertActivationConfig {
    static constexpr float beta_gate = SHARED_EXPERT_GLU_BETA, beta_up = SHARED_EXPERT_GLU_LINEAR_BETA;
    static constexpr float limit = SHARED_EXPERT_GLU_LIMIT, clamp_limit = SHARED_EXPERT_GLU_LIMIT,
                           alpha = SHARED_EXPERT_GLU_ALPHA;
};
}  // namespace ckernel::sfpu
#endif

// Compile-time args (head_program.cpp, the shared expert's compute kernel)
constexpr uint32_t row_tiles_per_core = get_compile_time_arg_val(0);  // (row tiles of a chunk this core computes)
constexpr uint32_t k_step_tiles = get_compile_time_arg_val(1);        // (K tiles a gate / up step)
constexpr uint32_t gate_up_passes = get_compile_time_arg_val(2);      // (gate / up passes; latent-down passes after)
constexpr uint32_t row_chunks = get_compile_time_arg_val(3);
constexpr uint32_t max_column_pairs = get_compile_time_arg_val(4);   // (h tiles a row of a core, at most)
constexpr uint32_t down_k_step_tiles = get_compile_time_arg_val(5);  // (K tiles a down step)
constexpr uint32_t down_k_steps = get_compile_time_arg_val(6);
constexpr uint32_t down_passes = get_compile_time_arg_val(7);
constexpr uint32_t w_slot_tiles = get_compile_time_arg_val(8);  // (a W slot: the tiles a step pops)
constexpr uint32_t k_shards = get_compile_time_arg_val(9);      // (K shards: one a TP chip)
constexpr uint32_t k_steps_per_shard = get_compile_time_arg_val(10);
constexpr uint32_t engine_columns = get_compile_time_arg_val(11);
// (h of the whole row, gathered: [column][row tile][max_column_pairs])
constexpr uint32_t h_gathered_tiles = engine_columns * row_tiles_per_core * max_column_pairs;

// The GLU activation of gate / up (the named compile-time arg: head_program.cpp glu_defines' code)
enum class Activation : uint32_t {
    Silu = 1,         // silu(gate) x up
    Situ = 2,         // SiTU (Kimi)
    ClampedSilu = 3,  // silu(min(gate, limit)) x clamp(up, +-limit): GLM-5.3's swiglu_limit, DeepSeek V4's
    SwigluOai = 4,    // (clamp(up, +-limit) + 1) x g sigmoid(alpha g), g = min(gate, limit): MiniMax M3
};
constexpr Activation activation = static_cast<Activation>(get_named_compile_time_arg_val("shared_expert_activation"));

// Circular buffers
constexpr uint32_t cb_x = 0;             // x steps (the row's x, gathered along the row)
constexpr uint32_t cb_w = 1;             // W steps (gate / up, then down; the column's W stream)
constexpr uint32_t cb_h_local = 2;       // this core's h columns of the chunk
constexpr uint32_t cb_h_gathered = 3;    // h of every column of the row (the down's input)
constexpr uint32_t cb_partial_sums = 5;  // fp32 gate / up partials of the K shards (packer L1 accumulation)

// The full arg list: the per-group block (gate / down tables, h column offsets, the shared expert's tiles), then the
// core's reduce-scatter / prep tables. Indices stay absolute (the host's tables point at them): arg i of the full list,
// the page the core's BRISC hands over from its block of the DRAM table - head_program.cpp head_args_to_dram.
// Runtime args: the column's gate / up pass widths [gate_up_passes], then the down pass widths [down_passes].
FORCE_INLINE uint32_t shared_expert_arg(uint32_t i) { return moe_block_args::arg(i); }

// The router's matmul at HiFi3 (the engine runs HiFi2: the routing picks need HiFi3): matmul_block's LLK calls with the
// fidelity fixed.
ALWI void router_matmul_hifi3_init(uint32_t in0_cb, uint32_t in1_cb, uint32_t column_tiles, uint32_t k_tiles) {
    state_configure(in1_cb, in0_cb, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul_init(in0_cb, in1_cb, 1, column_tiles, 1, k_tiles)));
    MATH((llk_math_matmul_init<MathFidelity::HiFi3, MM_THROTTLE>(in0_cb, in1_cb, 1, column_tiles, 1)));
}
ALWI void router_matmul_hifi3(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t in0_tile,
    uint32_t in1_tile,
    uint32_t dest_tile,
    uint32_t column_tiles,
    uint32_t k_tiles) {
    state_configure(in1_cb, in0_cb, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul(in0_cb, in1_cb, in0_tile, in1_tile, column_tiles, 1, k_tiles)));
    MATH((llk_math_matmul<MathFidelity::HiFi3, MM_THROTTLE>(dest_tile, column_tiles, 1)));
}

// The activation's SFPU init (on the PACK thread, which runs the GLU between the matmul and the pack)
template <Activation kind>
ALWI void activation_init() {
    if constexpr (kind == Activation::Situ) {
        PACK((SFPU_BINARY_INIT_FN_NO_ARGS(situ_glu, sfpu::situ_glu_init)));
    } else if constexpr (kind == Activation::ClampedSilu) {
        PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::clamped_silu_glu_init)));
    } else if constexpr (kind == Activation::SwigluOai) {
        PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::swiglu_init)));
    } else {
        silu_tile_init_pack();
        PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
    }
}
// The activation of one (gate, up) DEST tile pair, in place: DEST gate_tile <- act(gate, up), up at gate_tile + 1
template <Activation kind>
ALWI void activation_tile_pair(uint32_t gate_tile) {
    if constexpr (kind == Activation::Situ) {
        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_situ_glu,
            (DST_ACCUM_MODE, 8, sfpu::SharedExpertActivationConfig),
            gate_tile,
            gate_tile + 1,
            gate_tile,
            VectorMode::RC)));
    } else if constexpr (kind == Activation::ClampedSilu) {
        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_clamped_silu_glu,
            (DST_ACCUM_MODE, 8, sfpu::SharedExpertActivationConfig),
            gate_tile,
            gate_tile + 1,
            gate_tile,
            VectorMode::RC)));
    } else if constexpr (kind == Activation::SwigluOai) {
        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_swiglu,
            (DST_ACCUM_MODE, 8, sfpu::SharedExpertActivationConfig, true),
            gate_tile,
            gate_tile + 1,
            gate_tile,
            VectorMode::RC)));
    } else {
        silu_tile_pack(gate_tile);
        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary_mul,
            (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE),
            gate_tile,
            gate_tile + 1,
            gate_tile,
            VectorMode::RC)));
    }
}

// The router's logits: my expert tiles x my row tiles over the chip's K slice (every chain starts here)
void router_logits() {
    DeviceZoneScopedN("SE-RTRMM");
    // (every core the same count; past the experts: unused tiles)
    constexpr uint32_t expert_tiles = ROUTER_EXPERT_TILES_PER_COLUMN;
#ifdef ROUTER_ITERATIONS
    constexpr uint32_t router_iterations = ROUTER_ITERATIONS;  // (ROUTER_CORE_ROW_TILES of my rows each)
#else
    constexpr uint32_t router_iterations = 1;
#endif
    for (uint32_t iteration = 0; iteration < router_iterations; ++iteration) {
        // (a K chunk's expert_tiles x row tiles partials in DEST, then fp32 into ROUTER_OUT_CB with the packer's L1
        // accumulation)
        reconfig_data_format(ROUTER_X_CB, ROUTER_W_CB);
        pack_reconfig_data_format(ROUTER_OUT_CB);
        // (row groups of ROUTER_ROW_TILES_PER_GROUP rows: expert_tiles x ROUTER_ROW_TILES_PER_GROUP tiles in DEST - all
        // rows at once where they fit)
        router_matmul_hifi3_init(ROUTER_W_CB, ROUTER_X_CB, ROUTER_ROW_TILES_PER_GROUP, ROUTER_K_CHUNK_TILES);
        cb_reserve_back(ROUTER_OUT_CB, expert_tiles * ROUTER_CORE_ROW_TILES);
        for (uint32_t k_chunk = 0; k_chunk < ROUTER_K_TILES / ROUTER_K_CHUNK_TILES; ++k_chunk) {
            cb_wait_front(ROUTER_X_CB, ROUTER_K_CHUNK_TILES * ROUTER_CORE_ROW_TILES);
            cb_wait_front(ROUTER_W_CB, expert_tiles * ROUTER_K_CHUNK_TILES);
            PACK((llk_pack_reconfig_l1_acc(k_chunk > 0 ? 1 : 0)));
            for (uint32_t row_group = 0; row_group < ROUTER_CORE_ROW_TILES; row_group += ROUTER_ROW_TILES_PER_GROUP) {
                tile_regs_acquire();
                for (uint32_t expert = 0; expert < expert_tiles; ++expert) {
                    for (uint32_t k = 0; k < ROUTER_K_CHUNK_TILES; ++k) {
                        router_matmul_hifi3(
                            ROUTER_W_CB,
                            ROUTER_X_CB,
                            expert * ROUTER_K_CHUNK_TILES + k,
                            k * ROUTER_CORE_ROW_TILES + row_group,
                            expert * ROUTER_ROW_TILES_PER_GROUP,
                            ROUTER_ROW_TILES_PER_GROUP,
                            ROUTER_K_CHUNK_TILES);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t expert = 0; expert < expert_tiles; ++expert) {
                    for (uint32_t row = 0; row < ROUTER_ROW_TILES_PER_GROUP; ++row) {
                        pack_tile<true>(
                            expert * ROUTER_ROW_TILES_PER_GROUP + row,
                            ROUTER_OUT_CB,
                            expert * ROUTER_CORE_ROW_TILES + row_group + row);
                    }
                }
                tile_regs_release();
            }
            cb_pop_front(ROUTER_X_CB, ROUTER_K_CHUNK_TILES * ROUTER_CORE_ROW_TILES);
            cb_pop_front(ROUTER_W_CB, expert_tiles * ROUTER_K_CHUNK_TILES);
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
        cb_push_back(ROUTER_OUT_CB, expert_tiles * ROUTER_CORE_ROW_TILES);
        reconfig_data_format(cb_w, cb_x);  // (back to gate / up's formats)
    }
}

// One token loop of this kernel's work: kernel_main below runs it once per loop (token_loop.hpp).
// output_tiles_pushed: REDUCE_SCATTER_OUTPUT_FWD_CB tiles pushed over the whole call - its blocks end at multiples of
// SHARED_EXPERT_DEST_TILES (the NCRISC reads it page by page, pack_tile does not wrap): a loop's adds continue where
// the last loop's left it, as a chunk's do after the last chunk's.
void process_token_loop(uint32_t& output_tiles_pushed) {
    router_logits();
    activation_init<activation>();
#ifdef SKIP_PADDING_CHUNKS
    uint32_t live_chunks;  // (a padded forward: the row chunks up to the last with a real row - the BRISC's word)
    {
        volatile tt_l1_ptr uint32_t* live_chunks_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            moe_block_args::common(SKIP_PADDING_ARGS_OFFSET) + LIVE_CHUNKS_FLAG_ID);
        while ((live_chunks = *live_chunks_word) == 0) {
            invalidate_l1_cache();
        }
    }
#else
    constexpr uint32_t live_chunks = row_chunks;
#endif
    for (uint32_t chunk = 0; chunk < live_chunks; ++chunk) {
        // gate / up: per pass all of K in DEST, the activation, h packed once
        {
            DeviceZoneScopedN("SE-GU");
            if (row_chunks > 1 && chunk > 0) {
                reconfig_data_format(cb_w, cb_x);  // (the last chunk's down / reduce-scatter left other formats)
            }
            cb_reserve_back(cb_h_local, row_tiles_per_core * max_column_pairs);
            // shard-major: shard k (own first, then as each lands) through every pass; a pass's fp32 partial is packed
            // onto its fixed cb_partial_sums slot (row tiles x pass width tiles at the pass's offset) with the packer's
            // L1 accumulation - shard 0 writes it, shards 1 .. k_shards - 2 add onto it in L1 - and only the last shard
            // brings the sum back into DEST (unpacked straight to DEST, fp32) under its own matmuls; then the
            // activation and h. Shard k_shards - 2 pushes each pass's slot as it is done (the write pointer then sits
            // at the next pass's slot); the chunk's tail is pushed and popped as padding, so every chunk starts at the
            // region's start
            if (k_shards > 1) {
                cb_reserve_back(cb_partial_sums, SHARED_EXPERT_ACC_PAGES);
            }
            // (cb_partial_sums pages pushed this chunk - the pack offsets are relative to the write pointer)
            uint32_t partial_pages_pushed = 0;
            for (uint32_t shard = 0; shard < k_shards; ++shard) {
                uint32_t h_column_offset = 0, latent_column_offset = 0, partial_offset = 0;
                for (uint32_t pass = 0; pass < gate_up_passes; ++pass) {
                    const uint32_t pass_width = shared_expert_arg(pass), pass_tiles = row_tiles_per_core * pass_width;
                    tile_regs_acquire();
                    if (shard + 1 == k_shards && shard > 0) {
                        cb_wait_front(cb_partial_sums, pass_tiles);
                        reconfig_data_format_srca(cb_w, cb_partial_sums);
                        copy_init(cb_partial_sums);
                        for (uint32_t tile = 0; tile < pass_tiles; ++tile) {
                            copy_tile(cb_partial_sums, tile, tile);
                        }
                        cb_pop_front(cb_partial_sums, pass_tiles);
                        reconfig_data_format_srca(cb_partial_sums, cb_w);
                    }
                    matmul_block_init(cb_x, cb_w, 0, pass_width, row_tiles_per_core, k_step_tiles);
                    for (uint32_t step = 0; step < k_steps_per_shard; ++step) {
                        cb_wait_front(cb_x, row_tiles_per_core * k_step_tiles);
                        cb_wait_front(cb_w, w_slot_tiles);
                        for (uint32_t k = 0; k < k_step_tiles; ++k) {
                            matmul_block(
                                cb_x, cb_w, k, k * pass_width, 0, false, pass_width, row_tiles_per_core, k_step_tiles);
                        }
                        cb_pop_front(cb_x, row_tiles_per_core * k_step_tiles);
                        cb_pop_front(cb_w, w_slot_tiles);
                    }
                    tile_regs_commit();
                    if (shard + 1 < k_shards) {
                        tile_regs_wait();
                        pack_reconfig_data_format(cb_partial_sums);
                        PACK((llk_pack_reconfig_l1_acc(shard > 0 ? 1 : 0)));
                        for (uint32_t tile = 0; tile < pass_tiles; ++tile) {
                            pack_tile<true>(tile, cb_partial_sums, partial_offset + tile - partial_pages_pushed);
                        }
                        PACK((llk_pack_reconfig_l1_acc(0)));
                        tile_regs_release();
                        partial_offset += pass_tiles;
                        if (shard + 2 == k_shards) {  // (complete but for the last shard's: the last shard may read it)
                            cb_push_back(cb_partial_sums, pass_tiles);
                            partial_pages_pushed += pass_tiles;
                        }
                        continue;
                    }
                    tile_regs_wait();
#ifdef LATENT_CB
                    // (a latent-down pass: no activation, the chip's latent columns packed bf16 to LATENT_CB)
                    if (pass >= SHARED_EXPERT_GATE_UP_PASSES) {
                        pack_reconfig_data_format(LATENT_CB);
                        for (uint32_t row = 0; row < row_tiles_per_core; ++row) {
                            for (uint32_t tile = 0; tile < pass_width; ++tile) {
                                pack_tile<true>(
                                    row * pass_width + tile,
                                    LATENT_CB,
                                    row * LATENT_MAX_COLUMN_TILES + latent_column_offset + tile);
                            }
                        }
                        tile_regs_release();
                        latent_column_offset += pass_width;
                        continue;
                    }
#endif
                    pack_reconfig_data_format(cb_h_local);
                    // the activation on the PACK thread's SFPU, (gate, up) tile pairs in place
                    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                    for (uint32_t tile = 0; tile < pass_tiles; tile += 2) {
                        activation_tile_pair<activation>(tile);
                    }
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    for (uint32_t row = 0; row < row_tiles_per_core; ++row) {
                        for (uint32_t tile = 0; tile < pass_width; tile += 2) {
                            pack_tile<true>(
                                row * pass_width + tile,
                                cb_h_local,
                                row * max_column_pairs + h_column_offset + tile / 2);
                        }
                    }
                    tile_regs_release();
                    h_column_offset += pass_width / 2;
                }
            }
            cb_push_back(cb_h_local, row_tiles_per_core * max_column_pairs);
            // (the region's tail: padding, so the next chunk starts at its start)
            if (k_shards > 1 && partial_pages_pushed < SHARED_EXPERT_ACC_PAGES) {
                cb_push_back(cb_partial_sums, SHARED_EXPERT_ACC_PAGES - partial_pages_pushed);
                cb_wait_front(cb_partial_sums, SHARED_EXPERT_ACC_PAGES - partial_pages_pushed);
                cb_pop_front(cb_partial_sums, SHARED_EXPERT_ACC_PAGES - partial_pages_pushed);
            }
#ifdef LATENT_CB
            // (the latent: the BRISC hands it to the ports, the NCRISC writes mine; with chunks a chunk's latent pages,
            // the next chunk's packed past them)
            cb_reserve_back(LATENT_CB, LATENT_PAGES);
            cb_push_back(LATENT_CB, LATENT_PAGES);
#endif
        }
        {
            DeviceZoneScopedN("SE-DOWNRS");
            // (down + reduce-scatter) the host's program of ops: a down pass - all of K in DEST over h-gathered and the
            // streamed W_down; row r's tiles packed at its base: into REDUCE_SCATTER_HOLD_CB (fp32: an entry that adds
            // an arrival here) or REDUCE_SCATTER_FINAL_BLOCK_CB (bf16: a start entry, exchanged as is); a hand-off -
            // entry e's start sub-block is complete (a REDUCE_SCATTER_FINAL_COUNTER_CB page: the BRISC's hand-off); a
            // finish - the held partial + the arrival(s) in fp32 DEST, packed bf16 into the final block (relay: handed
            // off) or REDUCE_SCATTER_OUTPUT_FWD_CB (fin: y). Partials accumulate fp32 in L1; only the exchanged
            // sub-blocks are bf16.
            const uint32_t ops_at = gate_up_passes + down_passes + engine_columns + 1;
            const uint32_t num_ops = shared_expert_arg(ops_at), down_pass_table = ops_at + 1 + num_ops;
            const uint32_t entry_table = down_pass_table + down_passes * (1 + row_tiles_per_core);
            // (h-gathered columns waited for: the first pass follows the exchange's rounds)
            uint32_t down_passes_done = 0, h_columns_waited = 0;
            if (row_chunks > 1 && chunk > 0) {  // (the ports read the last chunk's final block: mine to write again -
                                                // the NCRISC's credit)
                cb_wait_front(REDUCE_SCATTER_FINAL_FREE_CB, 1);
                cb_pop_front(REDUCE_SCATTER_FINAL_FREE_CB, 1);
            }
            for (uint32_t op_index = 0; op_index < num_ops; ++op_index) {
                const uint32_t op = shared_expert_arg(ops_at + 1 + op_index), kind = op & 0xFF, index = op >> 8;
                if (kind == 0) {  // a down pass
                    const uint32_t down_pass = index;
                    const uint32_t pass_width =
                        shared_expert_arg(down_pass_table + down_pass * (1 + row_tiles_per_core));
                    reconfig_data_format(cb_w, cb_h_gathered);
                    matmul_block_init(cb_h_gathered, cb_w, 0, pass_width, row_tiles_per_core, max_column_pairs);
                    uint32_t h_column = 0;
                    tile_regs_acquire();
                    for (uint32_t step = 0; step < down_k_steps; ++step) {
                        cb_wait_front(cb_w, w_slot_tiles);
                        for (uint32_t k = 0; k < down_k_step_tiles; ++k) {
                            const uint32_t k_tile = step * down_k_step_tiles + k;
                            while (k_tile >= shared_expert_arg(gate_up_passes + down_passes + h_column + 1)) {
                                ++h_column;
                            }
                            if (h_column >= h_columns_waited) {
                                h_columns_waited = h_column + 1;
                                cb_wait_front(cb_h_gathered, h_columns_waited * row_tiles_per_core * max_column_pairs);
                            }
                            const uint32_t h_tile = h_column * row_tiles_per_core * max_column_pairs + k_tile -
                                                    shared_expert_arg(gate_up_passes + down_passes + h_column);
                            matmul_block(
                                cb_h_gathered,
                                cb_w,
                                h_tile,
                                k * pass_width,
                                0,
                                false,
                                pass_width,
                                row_tiles_per_core,
                                max_column_pairs);
                        }
                        cb_pop_front(cb_w, w_slot_tiles);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t row = 0; row < row_tiles_per_core; ++row) {
                        const uint32_t base_word =
                            shared_expert_arg(down_pass_table + down_pass * (1 + row_tiles_per_core) + 1 + row);
                        const uint32_t out_cb =
                            (base_word >> 31) ? REDUCE_SCATTER_HOLD_CB : REDUCE_SCATTER_FINAL_BLOCK_CB;
                        const uint32_t base = base_word & 0x7FFFFFFF;
                        pack_reconfig_data_format(out_cb);
                        for (uint32_t tile = 0; tile < pass_width; ++tile) {
                            pack_tile<true>(row * pass_width + tile, out_cb, base + tile);
                        }
                    }
                    tile_regs_release();
                    // (pass done: the UNPACK of a later finish may read its held tiles)
                    cb_reserve_back(REDUCE_SCATTER_PASS_SYNC_CB, 1);
                    cb_push_back(REDUCE_SCATTER_PASS_SYNC_CB, 1);
                    ++down_passes_done;
                } else if (kind == 1) {  // a hand-off
                    cb_reserve_back(REDUCE_SCATTER_FINAL_COUNTER_CB, 1);
                    cb_push_back(REDUCE_SCATTER_FINAL_COUNTER_CB, 1);
                } else if (kind == 3) {
                    // dispatch prep: my rows of the chip's owned tokens (x own / the latent) untilized into
                    // DISPATCH_PREP_OUT_CB, a unit (n tiles of one row tile, contiguous in L1 at DISPATCH_PREP_UNIT_CB
                    // + offset) at a time; the NCRISC writes its rows into the dispatch buffer. After the last pass:
                    // under the reduce-scatter's waits for arrivals.
                    DeviceZoneScopedN("SE-PREP");
#ifdef DISPATCH_PREP_LAST
                    // (with chunks: every chunk's units in the last chunk's op - no prep writes under the W re-reads)
                    const uint32_t units_per_chunk = chunk + 1 == live_chunks ? shared_expert_arg(index) : 0;
                    const uint32_t first_unit = 0, end_unit = live_chunks * units_per_chunk;
#else
                    const uint32_t units_per_chunk = shared_expert_arg(index);
                    const uint32_t first_unit = chunk * units_per_chunk, end_unit = (chunk + 1) * units_per_chunk;
#endif
                    uint32_t unit_cb_read_pointer = 0;
                    UNPACK((unit_cb_read_pointer = get_local_cb_interface(DISPATCH_PREP_UNIT_CB).fifo_rd_ptr));
                    [[maybe_unused]] uint32_t latent_chunks_ready = 0;  // (chunks' latent arrivals waited for)
                    if (units_per_chunk) {
                        // (the last pass left srcA at the weights' format)
                        reconfig_data_format_srca(DISPATCH_PREP_UNIT_CB);
                        pack_reconfig_data_format(DISPATCH_PREP_OUT_CB);
                        pack_untilize_init<DISPATCH_PREP_BLOCK_COLUMN_TILES, DISPATCH_PREP_FULL_COLUMN_TILES>(
                            DISPATCH_PREP_UNIT_CB, DISPATCH_PREP_OUT_CB);
                    }
                    for (uint32_t unit = first_unit; unit < end_unit; ++unit) {
                        const uint32_t unit_offset = shared_expert_arg(index + 1 + 2 * unit),
                                       unit_flags = shared_expert_arg(index + 2 + 2 * unit);
                        const uint32_t unit_tiles = unit_flags & 0xFFFF;
#ifdef LATENT_READY_CB
                        // (a unit of the latent's arrivals: once they all landed)
                        if ((unit_flags >> 16) && latent_chunks_ready < unit / units_per_chunk + 1) {
                            latent_chunks_ready = unit / units_per_chunk + 1;
                            cb_wait_front(LATENT_READY_CB, latent_chunks_ready);
                        }
#endif
                        cb_reserve_back(DISPATCH_PREP_OUT_CB, DISPATCH_PREP_FULL_COLUMN_TILES);
                        for (uint32_t block = 0; block * DISPATCH_PREP_BLOCK_COLUMN_TILES < unit_tiles; ++block) {
                            // (the unit's block in L1: the CB's read pointer is in 16 B words)
                            UNPACK(
                                (get_local_cb_interface(DISPATCH_PREP_UNIT_CB).fifo_rd_ptr =
                                     unit_cb_read_pointer + (unit_offset + block * DISPATCH_PREP_BLOCK_COLUMN_TILES *
                                                                               moe_block_tiles::kBf16TileBytes) /
                                                                16));
                            pack_untilize_block<DISPATCH_PREP_BLOCK_COLUMN_TILES, DISPATCH_PREP_FULL_COLUMN_TILES>(
                                DISPATCH_PREP_UNIT_CB, 1, DISPATCH_PREP_OUT_CB, block);
                        }
                        cb_push_back(DISPATCH_PREP_OUT_CB, DISPATCH_PREP_FULL_COLUMN_TILES);
                    }
                    if (units_per_chunk) {
                        pack_untilize_uninit(DISPATCH_PREP_OUT_CB);
                    }
                    UNPACK((get_local_cb_interface(DISPATCH_PREP_UNIT_CB).fifo_rd_ptr = unit_cb_read_pointer));
                } else {  // a finish: entry `index` - its held partial + the arrival(s)
                    const uint32_t entry_args = entry_table + 6 * index;
                    const uint32_t entry_tiles = shared_expert_arg(entry_args),
                                   mode = shared_expert_arg(entry_args + 1);
                    // (the arrivals' zone offsets, tiles - every slot before them pushed with them)
                    const uint32_t forward_slot = shared_expert_arg(entry_args + 2),
                                   backward_slot = shared_expert_arg(entry_args + 3);
                    const uint32_t hold_offset = shared_expert_arg(entry_args + 4),
                                   final_block_offset = shared_expert_arg(entry_args + 5);
                    // (mode: bit 0 a forward arrival, bit 1 a backward arrival, bit 2 the output - else relayed)
                    const bool has_forward = mode & 1, has_backward = mode & 2, is_output = mode & 4;
                    cb_wait_front(REDUCE_SCATTER_PASS_SYNC_CB, down_passes_done);
                    if (has_forward) {
                        cb_wait_front(REDUCE_SCATTER_ARRIVAL_FWD_CB, forward_slot + entry_tiles);
                    }
                    if (has_backward) {
                        cb_wait_front(REDUCE_SCATTER_ARRIVAL_BWD_CB, backward_slot + entry_tiles);
                    }
                    const uint32_t first_arrival_cb =
                        has_forward ? REDUCE_SCATTER_ARRIVAL_FWD_CB : REDUCE_SCATTER_ARRIVAL_BWD_CB;
                    const uint32_t first_arrival_slot = has_forward ? forward_slot : backward_slot;
                    const uint32_t out_cb = is_output ? REDUCE_SCATTER_OUTPUT_FWD_CB : REDUCE_SCATTER_FINAL_BLOCK_CB;
                    pack_reconfig_data_format(out_cb);
                    for (uint32_t first = 0, tiles = 0; first < entry_tiles; first += tiles) {
                        const uint32_t limit =
                            is_output ? SHARED_EXPERT_DEST_TILES - output_tiles_pushed % SHARED_EXPERT_DEST_TILES
                                      : SHARED_EXPERT_DEST_TILES;
                        tiles = entry_tiles - first < limit ? entry_tiles - first : limit;
                        if (is_output) {
                            output_tiles_pushed += tiles;
                        }
                        tile_regs_acquire();
                        reconfig_data_format(REDUCE_SCATTER_HOLD_CB, first_arrival_cb);
                        add_init(REDUCE_SCATTER_HOLD_CB, first_arrival_cb);
                        for (uint32_t j = 0; j < tiles; ++j) {
                            add_tiles(
                                REDUCE_SCATTER_HOLD_CB,
                                first_arrival_cb,
                                hold_offset + first + j,
                                first_arrival_slot + first + j,
                                j);
                        }
                        if (has_forward && has_backward) {  // (the second arrival: DEST -> srcA at fp32 width, + bwd)
                            reconfig_data_format(REDUCE_SCATTER_HOLD_CB, REDUCE_SCATTER_ARRIVAL_BWD_CB);
                            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                                REDUCE_SCATTER_ARRIVAL_BWD_CB);
                            for (uint32_t j = 0; j < tiles; ++j) {
                                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                                    REDUCE_SCATTER_ARRIVAL_BWD_CB, backward_slot + first + j, j);
                            }
                        }
                        tile_regs_commit();
                        if (is_output) {
                            cb_reserve_back(REDUCE_SCATTER_OUTPUT_FWD_CB, tiles);
                        }
                        tile_regs_wait();
                        for (uint32_t j = 0; j < tiles; ++j) {
                            if (is_output) {
                                pack_tile(j, REDUCE_SCATTER_OUTPUT_FWD_CB);
                            } else {
                                pack_tile<true>(j, REDUCE_SCATTER_FINAL_BLOCK_CB, final_block_offset + first + j);
                            }
                        }
                        tile_regs_release();
                        if (is_output) {
                            cb_push_back(REDUCE_SCATTER_OUTPUT_FWD_CB, tiles);
                        }
                    }
                    if (!is_output) {
                        cb_reserve_back(REDUCE_SCATTER_FINAL_COUNTER_CB, 1);
                        cb_push_back(REDUCE_SCATTER_FINAL_COUNTER_CB, 1);
                    }
                }
            }
            cb_pop_front(cb_h_gathered, h_gathered_tiles);
            // (the chunk's syncs and arrival zones popped: the next chunk's - or the next token loop's - from their
            // start)
            if constexpr (row_chunks > 1 || TOKEN_LOOPS > 1) {
                cb_wait_front(REDUCE_SCATTER_PASS_SYNC_CB, down_passes_done);
                cb_pop_front(REDUCE_SCATTER_PASS_SYNC_CB, down_passes_done);
                uint32_t arrival_directions = 0;
                for (uint32_t entry = 0; entry < REDUCE_SCATTER_ENTRIES; ++entry) {
                    arrival_directions |= shared_expert_arg(entry_table + 6 * entry + 1);  // (the entries: TP of them)
                }
                if (arrival_directions & 1) {
                    cb_wait_front(REDUCE_SCATTER_ARRIVAL_FWD_CB, REDUCE_SCATTER_ZONE_TILES_FWD);
                    cb_pop_front(REDUCE_SCATTER_ARRIVAL_FWD_CB, REDUCE_SCATTER_ZONE_TILES_FWD);
                }
                if (arrival_directions & 2) {
                    cb_wait_front(REDUCE_SCATTER_ARRIVAL_BWD_CB, REDUCE_SCATTER_ZONE_TILES_BWD);
                    cb_pop_front(REDUCE_SCATTER_ARRIVAL_BWD_CB, REDUCE_SCATTER_ZONE_TILES_BWD);
                }
            }
        }
    }
#if TOKEN_LOOPS > 1
#ifdef LATENT_CB
    // (token loops: the skipped chunks' latent pages as padding - the BRISC pops the whole region)
    if (live_chunks < row_chunks) {
        cb_reserve_back(LATENT_CB, (row_chunks - live_chunks) * LATENT_PAGES);
        cb_push_back(LATENT_CB, (row_chunks - live_chunks) * LATENT_PAGES);
    }
#endif
#ifdef LATENT_READY_CB
    // (token loops: the NCRISC's page a chunk, filled up to live_chunks - the next loop's from 0)
    cb_wait_front(LATENT_READY_CB, live_chunks);
    cb_pop_front(LATENT_READY_CB, live_chunks);
#endif
#endif
}

void kernel_main() {
    moe_block_args::load();  // (this core's args: the page the BRISC hands over)
    // (once: every reconfiguration after this is an explicit *_init / reconfig_data_format call)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_h_local);
    uint32_t output_tiles_pushed = 0;
    for (uint32_t loop = 0; loop < token_loop::LOOPS; ++loop) {
        token_loop::begin(loop);
        process_token_loop(output_tiles_pushed);
        token_loop::end(loop);
    }
}
