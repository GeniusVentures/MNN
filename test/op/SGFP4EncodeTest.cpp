//
//  SGFP4EncodeTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/08/29.
//  Copyright © 2018, Alibaba Group Holding Limited.
//
// Plan 09-04: CPU decode-parity suite for the C++ encoder (D-04/D-08 CPU
// leg). Encodes the kRealShapeFixtures input weights via
// sgfp4_encode::encode(), decodes via the CPU oracle (direct for aligned
// shapes, _crop overload for padded shapes) and asserts parity with the
// Python-encoded decoded reference at rtol 1e-4. Also covers the encode()
// security gates (NaN/Inf, invalid dims, all-zero input) and a hard
// LAYOUT_FULL_4X4 assertion on a constructed all-split plane (D-11b).
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "MNN/SGFP4DequantUtils.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "SGFP4RealShapeFixtures.h"
#include "sgfp4_encode.hpp"

namespace {

// D-04 decode-vs-decode bar (decode-vs-decode, not encode-vs-source).
constexpr float kEncodeRelTol = 1e-4f;

inline int paddedDim(int d) {
    return ((d + 63) / 64) * 64;
}

inline bool isAligned(int dimO, int dimI) {
    return paddedDim(dimO) == dimO && paddedDim(dimI) == dimI;
}

} // namespace

class SGFP4EncodeTest : public MNNTestCase {
public:
    SGFP4EncodeTest()  = default;
    virtual ~SGFP4EncodeTest()  = default;

    virtual bool run(int precision) {
        (void)precision;
        if (!testContainerFraming()) {
            return false;
        }
        if (!testFixtureParity()) {
            return false;
        }
        if (!testSecurityGates()) {
            return false;
        }
        if (!testAllZeroInput()) {
            return false;
        }
        if (!testFull4x4Layout()) {
            return false;
        }
        MNN_PRINT("SGFP4EncodeTest: all layers PASSED\n");
        return true;
    }

private:
    // Container framing: magic + version for every encoded fixture.
    bool testContainerFraming() {
        for (size_t i = 0; i < sgfp4_real_shape_fixtures::kRealShapeFixtureCount; ++i) {
            const auto& f = sgfp4_real_shape_fixtures::kRealShapeFixtures[i];
            auto cppContainer = sgfp4_encode::encode(f.inputWeights, f.dimO, f.dimI);
            if (cppContainer.size() < 16) {
                MNN_ERROR("SGFP4EncodeTest: container too small for '%s'\n", f.name);
                return false;
            }
            uint32_t magic = 0;
            std::memcpy(&magic, cppContainer.data(), 4);
            if (magic != MNN::kSGFP4Magic) {
                MNN_ERROR("SGFP4EncodeTest: bad magic for '%s'\n", f.name);
                return false;
            }
            if (cppContainer[4] != 0x02u) {
                MNN_ERROR("SGFP4EncodeTest: bad version for '%s'\n", f.name);
                return false;
            }
        }
        MNN_PRINT("SGFP4EncodeTest: framing PASSED\n");
        return true;
    }

    // Parity: C++ encode → CPU decode == Python encode → Python decode
    // (via the committed expected array) within rtol 1e-4. Aligned shapes
    // use the direct oracle; padded shapes use the _crop overload. This
    // inherently also verifies that std::rint rounding ties do not cause
    // out-of-tolerance divergence vs the numpy half-to-even oracle.
    bool testFixtureParity() {
        for (size_t i = 0; i < sgfp4_real_shape_fixtures::kRealShapeFixtureCount; ++i) {
            const auto& f = sgfp4_real_shape_fixtures::kRealShapeFixtures[i];
            auto cppContainer = sgfp4_encode::encode(f.inputWeights, f.dimO, f.dimI);
            if (cppContainer.empty()) {
                MNN_ERROR("SGFP4EncodeTest: encode returned empty for '%s'\n", f.name);
                return false;
            }
            std::vector<float> cppOut(f.expectedCount, 0.0f);
            bool ok;
            if (isAligned(f.dimO, f.dimI)) {
                ok = MNN::dequant_sgfp4_container_cpu(cppContainer.data(), cppContainer.size(), cppOut.data(),
                                                       f.expectedCount);
            } else {
                int pO = paddedDim(f.dimO);
                int pI = paddedDim(f.dimI);
                ok = MNN::dequant_sgfp4_container_cpu_crop(cppContainer.data(), cppContainer.size(), cppOut.data(),
                                                            f.dimO, f.dimI, pO, pI);
            }
            if (!ok) {
                MNN_ERROR("SGFP4EncodeTest: decode returned false for '%s'\n", f.name);
                return false;
            }
            if (!checkVectorByRelativeError<float>(cppOut.data(), f.expected, static_cast<int>(f.expectedCount),
                                                   kEncodeRelTol)) {
                MNN_ERROR("SGFP4EncodeTest: parity failed for '%s'\n", f.name);
                return false;
            }
        }
        MNN_PRINT("SGFP4EncodeTest: fixture parity (%zu shapes, rtol 1e-4) PASSED\n",
                  sgfp4_real_shape_fixtures::kRealShapeFixtureCount);
        return true;
    }

    // Security gates (T-09-01/T-09-02): NaN/Inf planes and invalid dims
    // must produce the empty-vector contract.
    bool testSecurityGates() {
        std::vector<float> v(64 * 64, 1.0f);

        {
            std::vector<float> nanW = v;
            nanW[100] = std::numeric_limits<float>::quiet_NaN();
            if (!sgfp4_encode::encode(nanW.data(), 64, 64).empty()) {
                MNN_ERROR("SGFP4EncodeTest: NaN input not rejected\n");
                return false;
            }
        }
        {
            std::vector<float> infW = v;
            infW[100] = std::numeric_limits<float>::infinity();
            if (!sgfp4_encode::encode(infW.data(), 64, 64).empty()) {
                MNN_ERROR("SGFP4EncodeTest: +Inf input not rejected\n");
                return false;
            }
        }
        {
            std::vector<float> negInfW = v;
            negInfW[100] = -std::numeric_limits<float>::infinity();
            if (!sgfp4_encode::encode(negInfW.data(), 64, 64).empty()) {
                MNN_ERROR("SGFP4EncodeTest: -Inf input not rejected\n");
                return false;
            }
        }
        if (!sgfp4_encode::encode(v.data(), 0, 64).empty()) {
            MNN_ERROR("SGFP4EncodeTest: zero dimO not rejected\n");
            return false;
        }
        if (!sgfp4_encode::encode(v.data(), 64, 0).empty()) {
            MNN_ERROR("SGFP4EncodeTest: zero dimI not rejected\n");
            return false;
        }
        if (!sgfp4_encode::encode(v.data(), -1, 64).empty()) {
            MNN_ERROR("SGFP4EncodeTest: negative dim not rejected\n");
            return false;
        }
        if (!sgfp4_encode::encode(nullptr, 64, 64).empty()) {
            MNN_ERROR("SGFP4EncodeTest: null weights not rejected\n");
            return false;
        }
        MNN_PRINT("SGFP4EncodeTest: security gates PASSED\n");
        return true;
    }

    // All-zero plane (maxabs == 0): must produce a valid container that
    // decodes to all zeros (T-09-03 guard round-trip).
    bool testAllZeroInput() {
        std::vector<float> zeros(64 * 64, 0.0f);
        auto container = sgfp4_encode::encode(zeros.data(), 64, 64);
        if (container.empty()) {
            MNN_ERROR("SGFP4EncodeTest: all-zero input produced empty container\n");
            return false;
        }
        std::vector<float> out(64 * 64, 0.0f);
        if (!MNN::dequant_sgfp4_container_cpu(container.data(), container.size(), out.data(), 64 * 64)) {
            MNN_ERROR("SGFP4EncodeTest: all-zero container failed decode\n");
            return false;
        }
        for (float value : out) {
            if (std::fabs(value) > 1e-4f) {
                MNN_ERROR("SGFP4EncodeTest: all-zero decode produced %f\n", value);
                return false;
            }
        }
        MNN_PRINT("SGFP4EncodeTest: all-zero input PASSED\n");
        return true;
    }

    // D-11b: high-variance uniform noise forces every quadtree level past
    // its gate, splitting all the way to the 4x4 floor → LAYOUT_FULL_4X4
    // (verified against the Python encoder: default_rng(42) uniform noise
    // in [-8,8] yields layout distribution {5:1}). Hard assertion on the
    // record's sb_header low 3 bits (not just informational). Deterministic
    // LCG so the test never depends on <random> implementation details.
    bool testFull4x4Layout() {
        std::vector<float> weights(64 * 64, 0.0f);
        uint32_t lcgState = 42u; // numerically-verified recipe (see comment)
        auto nextUniform = [&lcgState]() {
            lcgState = lcgState * 1664525u + 1013904223u;
            // Map to [-8, 8) with ample mantissa entropy per sample.
            return (static_cast<float>(lcgState >> 8) / 16777216.0f) * 16.0f - 8.0f;
        };
        for (int r = 0; r < 64; ++r) {
            for (int c = 0; c < 64; ++c) {
                weights[static_cast<size_t>(r) * 64 + c] = nextUniform();
            }
        }
        auto container = sgfp4_encode::encode(weights.data(), 64, 64);
        if (container.empty()) {
            MNN_ERROR("SGFP4EncodeTest: checkerboard encode returned empty\n");
            return false;
        }

        // Record 0 starts at the record-region base: fixed header (16) +
        // offset table (B=1 × 4 bytes) padded to 16 = byte 32.
        const size_t kExpectedSbHeaderOffset = 32;
        if (container.size() < kExpectedSbHeaderOffset + 4) {
            MNN_ERROR("SGFP4EncodeTest: container too small for sb_header probe\n");
            return false;
        }
        uint32_t sbHeader = MNN::sgfp4_read_u32_le(container.data() + kExpectedSbHeaderOffset);
        uint32_t layout   = sbHeader & MNN::kSGFP4LayoutEnumMask;
        if (layout != MNN::kSGFP4LayoutFull4x4) {
            MNN_ERROR("SGFP4EncodeTest: expected LAYOUT_FULL_4X4 (%u), got %u\n",
                      static_cast<unsigned>(MNN::kSGFP4LayoutFull4x4), static_cast<unsigned>(layout));
            return false;
        }

        // Layout distribution check only: for FULL_4X4 every 4x4 leaf is
        // quantized independently, so pixel-level round-trip parity is not
        // meaningful here (the fixture-parity layer owns numerical parity).
        MNN_PRINT("SGFP4EncodeTest: LAYOUT_FULL_4X4 coverage PASSED\n");
        return true;
    }
};

MNNTestSuiteRegister(SGFP4EncodeTest, "op/sgfp4/encode");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
