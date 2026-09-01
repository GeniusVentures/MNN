//
//  sgfp4_encode.hpp
//  MNN
//
//  Created by MNN on 2026/08/29.
//  Copyright © 2018, Alibaba Group Holding Limited.
//
// SGFP4 v2 adaptive quadtree weight encoder (Phase 9, Plan 09-01).
//
// One-shot public API. Mirrors gnus-poc fp4_exporter.py --adaptive:
// DEFAULT_V2_THRESHOLDS split policy, Laplacian-pyramid weighted error,
// FP4_AFFINE / T158_AFFINE dual code modes, LAYOUT_MIXED + uniform 0-3 +
// LAYOUT_FULL_4X4 superblock layouts, internal zero-padding of non-64-
// aligned {dimO, dimI} planes, and the SGF4|0x02 container framing that
// the MNN CPU/Vulkan decoders consume unchanged.
//
// Returns an empty vector on invalid input: non-finite (NaN/Inf) weights,
// non-positive dims, or dims exceeding 65536.
//

#ifndef TOOLS_FP4_SGFP4_ENCODE_HPP
#define TOOLS_FP4_SGFP4_ENCODE_HPP

#include <cstdint>
#include <vector>

#include "MNN/SGFP4DequantUtils.hpp"

namespace sgfp4_encode {

// Phase 10, Plan 10-03 (D-08): explicit threshold tunable (Phase 9's D-10
// deferral resolved). Defaults are the Python-identical
// DEFAULT_V2_THRESHOLDS values (fp4_exporter.py); revisions land only as
// deliberate, data-justified deltas from those defaults. Future knobs are
// deliberately absent (D-10: thresholds only).
struct EncodeConfig {
    struct Gate {
        int leafSize;       // 64 / 32 / 16 / 8 / 4 (kDefaultV2Thresholds order)
        double maxMse;
        double maxRelative;
    };
    Gate leafGates[5];
};

// The shipped default config: values identical to gnus-poc
// DEFAULT_V2_THRESHOLDS. Defined in sgfp4_encode.cpp (extern linkage;
// do not define another copy at namespace scope in this header).
extern const EncodeConfig kDefaultEncodeConfig;

// Encode the dimO x dimI row-major FP32 weight plane into an SGFP4 v2
// adaptive container (byte vector). See file header for the invalid-input
// contract (empty vector return).
std::vector<uint8_t> encode(const float* weights, int dimO, int dimI);

// Config-carrying overload: identical to the one-shot encode above when
// handed kDefaultEncodeConfig; threads caller gate values through the
// quadtree split policy otherwise. An overload (NOT a default argument) so
// "which call site uses tuned values" stays greppable. Same invalid-input
// contract (empty vector return).
std::vector<uint8_t> encode(const float* weights, int dimO, int dimI, const EncodeConfig& config);

} // namespace sgfp4_encode

#endif /* TOOLS_FP4_SGFP4_ENCODE_HPP */
