// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The routing planner (routing_planner.hpp) for a kernel: ROUTING_PLANNER_RUN(a, B). Token loops in one program
// (ROUTING_PLANNER_PER_LOOP, token_loop.hpp): one planner, its tokens' layout and instance bases picked by the loop (an
// earlier loop's plans no token: ROUTING_PLANNER_IDLE_*). ROUTING_PLANNER_NONE: no instance on this kernel's cores - no
// planner code.
#pragma once

#ifdef ROUTING_PLANNER_NONE
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_accessors.hpp"
#define ROUTING_PLANNER_RUN(a, B) ((void)(a), (void)(B))
#else
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_planner.hpp"
#define ROUTING_PLANNER_RUN(a, B) routing_planner::plan((a), (B))
#endif
