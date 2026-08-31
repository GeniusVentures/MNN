//
//  SGFP4VulkanEncodeParityTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/08/29.
//  Copyright © 2018, Alibaba Group Holding Limited.
//
// Plan 09-05: Vulkan encode-parity suite (D-08 verification target).
// C++-encodes each fixture's input weights via sgfp4_encode::encode(),
// writes the container to a temp sidecar, runs a Vulkan SGFP4Dequant
// Session, and compares the decoded output against the Python-encoded
// decoded reference at rtol 1e-4. Covers both the aligned leg (128x64,
// both dims 64-multiples) and the padded-crop leg (D-11a: 100x36, 37x91
// and the other non-aligned shapes decode to TRUE dims -- no pad-column
// contamination). Gracefully skips on machines without a Vulkan device.
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "MNN_generated.h"
#include "MNN/SGFP4DequantUtils.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/expr/Executor.hpp"
#include "MNN/expr/Module.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "SGFP4RealShapeFixtures.h"
#include "sgfp4_encode.hpp"

using namespace MNN::Express;

namespace {

// D-04 decode-vs-decode bar for the Vulkan leg.
constexpr float kVkEncodeRelTol = 1e-4f;

inline int paddedDimOf(int d) {
    return ((d + 63) / 64) * 64;
}

// Run one fixture's C++-encoded container through a Vulkan Session with
// the production external-sidecar plumbing ({0, size} descriptor +
// op->externalPath set directly on the OpT). Returns nullptr on failure
// (MNN_ERROR already printed); the output count is returned via outCount.
const float* runVulkanSession(const std::string& sidecarPath, const std::vector<uint8_t>& container, int dimO,
                              int dimI, int& outCount, std::vector<float>& outStorage) {
    std::shared_ptr<MNN::OpT> op(new MNN::OpT);
    op->type      = MNN::OpType_SGFP4Dequant;
    op->main.type = MNN::OpParameter_SGFP4DequantParam;
    auto* param   = new MNN::SGFP4DequantParamT;
    param->magic    = MNN::kSGFP4Magic;
    param->external = {0, static_cast<int64_t>(container.size())};
    param->dims     = {dimO, dimI};
    op->main.value  = param;
    // Op.externalPath must be set directly here -- the creator reads
    // op->externalPath() (not covered by createExecutionWithExternal).
    op->externalPath = sidecarPath;

    auto output = Variable::create(Expr::create(op.get(), {}));
    auto buffer = Variable::save({output});

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_VULKAN;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High; // FP32 shader variant
    backendConfig.memory    = MNN::BackendConfig::Memory_High;
    config.backendConfig    = &backendConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setExternalFile(sidecarPath);

    std::shared_ptr<Module> m(Module::load({}, {}, reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size(),
                                            rtmgr));
    if (nullptr == m) {
        MNN_ERROR("SGFP4VulkanEncodeParityTest: Module::load returned null\n");
        return nullptr;
    }

    auto outputs = m->onForward({});
    if (outputs.empty()) {
        MNN_ERROR("SGFP4VulkanEncodeParityTest: module produced no outputs\n");
        return nullptr;
    }
    auto outVar  = outputs[0];
    auto* outPtr = outVar->readMap<float>();
    auto outInfo = outVar->getInfo();
    if (nullptr == outPtr || nullptr == outInfo) {
        MNN_ERROR("SGFP4VulkanEncodeParityTest: output has no data/info\n");
        return nullptr;
    }
    outCount = static_cast<int>(outInfo->size);
    // CRITICAL: copy BEFORE the VARP dies. The outputs vector is local to
    // this function; returning outPtr directly would hand the caller a
    // dangling pointer into a freed Express variable (whose tensor memory
    // returns to the allocator pool -- surfacing as 0xdddddddd canary
    // reads). Copy into caller-owned storage so the data survives.
    outStorage.assign(outPtr, outPtr + outCount);
    return outStorage.data();
}

// Encode fixture → temp sidecar → Vulkan Session → parity check. The pad
// filter selects aligned-only (false) or padded-only (true) fixtures.
bool runVulkanFixtureParity(bool paddedOnly, int& checked) {
    for (size_t i = 0; i < sgfp4_real_shape_fixtures::kRealShapeFixtureCount; ++i) {
        const auto& f = sgfp4_real_shape_fixtures::kRealShapeFixtures[i];
        bool isPadded = (paddedDimOf(f.dimO) != f.dimO) || (paddedDimOf(f.dimI) != f.dimI);
        if (paddedOnly != isPadded) {
            continue;
        }

        auto container = sgfp4_encode::encode(f.inputWeights, f.dimO, f.dimI);
        if (container.empty()) {
            MNN_ERROR("SGFP4VulkanEncodeParityTest: encode returned empty for '%s'\n", f.name);
            return false;
        }

        std::ostringstream oss;
        oss << "sgfp4_vk_encode_parity_" << i << "_" << static_cast<unsigned long>(std::time(nullptr)) << "_"
            << static_cast<unsigned long>(rand()) << ".mnn.weight";
        std::string sidecarPath = oss.str();
        {
            std::ofstream ofs(sidecarPath, std::ios::binary | std::ios::trunc);
            if (!ofs) {
                MNN_ERROR("SGFP4VulkanEncodeParityTest: failed to open sidecar '%s'\n", sidecarPath.c_str());
                return false;
            }
            ofs.write(reinterpret_cast<const char*>(container.data()),
                      static_cast<std::streamsize>(container.size()));
        }

        int outCount = 0;
        std::vector<float> outStorage;
        const float* outPtr = runVulkanSession(sidecarPath, container, f.dimO, f.dimI, outCount, outStorage);
        std::remove(sidecarPath.c_str());
        if (nullptr == outPtr) {
            return false;
        }

        // True dims only -- never the padded count (T-09-11).
        if (outCount != f.dimO * f.dimI) {
            MNN_ERROR("SGFP4VulkanEncodeParityTest: '%s' output count %d != true %d\n", f.name, outCount,
                      f.dimO * f.dimI);
            return false;
        }
        if (!checkVectorByRelativeError<float>(outPtr, f.expected, outCount, kVkEncodeRelTol)) {
            MNN_ERROR("SGFP4VulkanEncodeParityTest: parity mismatch for '%s' (rtol 1e-4)\n", f.name);
            return false;
        }
        ++checked;
    }
    return true;
}

// Explicit crop-correctness probe (Pitfall 5): on a padded fixture, the
// value at the START of output row 1 must equal the reference's row-1
// start, NOT the flat-prefix value that pad columns would have shifted in.
bool testVulkanCropCorrectness() {
    for (size_t i = 0; i < sgfp4_real_shape_fixtures::kRealShapeFixtureCount; ++i) {
        const auto& f = sgfp4_real_shape_fixtures::kRealShapeFixtures[i];
        // Use shape_100x36 or shape_37x91 (any clearly non-aligned shape
        // with dimI far from its padded stride).
        if (f.dimI % 64 == 0 || f.dimO % 64 == 0) {
            continue;
        }
        int paddedI = paddedDimOf(f.dimI);
        if (paddedI - f.dimI < 2) {
            continue; // need at least one pad column for a decisive probe
        }

        auto container = sgfp4_encode::encode(f.inputWeights, f.dimO, f.dimI);
        if (container.empty()) {
            MNN_ERROR("SGFP4VulkanCropCorrectness: encode returned empty for '%s'\n", f.name);
            return false;
        }

        std::ostringstream oss;
        oss << "sgfp4_vk_crop_" << i << "_" << static_cast<unsigned long>(std::time(nullptr)) << ".bin";
        std::string sidecarPath = oss.str();
        {
            std::ofstream ofs(sidecarPath, std::ios::binary | std::ios::trunc);
            ofs.write(reinterpret_cast<const char*>(container.data()),
                      static_cast<std::streamsize>(container.size()));
        }

        int outCount = 0;
        std::vector<float> outStorage;
        const float* outPtr = runVulkanSession(sidecarPath, container, f.dimO, f.dimI, outCount, outStorage);
        std::remove(sidecarPath.c_str());
        if (nullptr == outPtr) {
            return false;
        }
        if (outCount != f.dimO * f.dimI) {
            MNN_ERROR("SGFP4VulkanCropCorrectness: '%s' outCount %d != %d\n", f.name, outCount, f.dimO * f.dimI);
            return false;
        }

        // Row-0 parity + row-boundary probe: outPtr[dimI] (row 1, col 0)
        // must match expected[dimI], NOT expected[paddedI] (the value a
        // flat-prefix copy would have placed there).
        if (!checkVectorByRelativeError<float>(outPtr, f.expected, f.dimI, kVkEncodeRelTol)) {
            MNN_ERROR("SGFP4VulkanCropCorrectness: row 0 mismatch for '%s'\n", f.name);
            return false;
        }
        float rowBoundary    = outPtr[f.dimI];
        float want           = f.expected[f.dimI];
        float flatPrefixWrong = f.expected[paddedI];
        if (std::fabs(rowBoundary - flatPrefixWrong) < 1e-9f &&
            std::fabs(want - flatPrefixWrong) > 1e-9f) {
            MNN_ERROR("SGFP4VulkanCropCorrectness: flat-prefix contamination detected for '%s'\n", f.name);
            return false;
        }
        if (std::fabs(rowBoundary - want) > 1e-2f * (std::fabs(want) + 1.0f)) {
            MNN_ERROR("SGFP4VulkanCropCorrectness: row boundary mismatch for '%s'\n", f.name);
            return false;
        }
        MNN_PRINT("SGFP4VulkanCropCorrectness: '%s' crop probe PASSED\n", f.name);
        return true;
    }
    MNN_ERROR("SGFP4VulkanCropCorrectness: no qualifying padded fixture found\n");
    return false;
}

} // namespace

class SGFP4VulkanEncodeParityTest : public MNNTestCase {
public:
    SGFP4VulkanEncodeParityTest()  = default;
    virtual ~SGFP4VulkanEncodeParityTest()  = default;

    virtual bool run(int precision) {
        (void)precision;

        // Graceful skip: no Vulkan device → suite still passes (D-07).
        auto vulkanCreator = MNN::MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN);
        if (nullptr == vulkanCreator) {
            MNN_PRINT("Vulkan backend not available — skipping SGFP4 Vulkan encode parity test\n");
            return true;
        }

        int checked = 0;
        if (!runVulkanFixtureParity(false, checked)) { // aligned shapes (e.g. 128x64)
            return false;
        }
        if (!runVulkanFixtureParity(true, checked)) { // padded shapes (100x36, 37x91, ...)
            return false;
        }
        if (!testVulkanCropCorrectness()) {
            return false;
        }
        MNN_PRINT("SGFP4VulkanEncodeParityTest: %d fixtures (aligned + padded-crop) matched Python reference on "
                  "Vulkan (rtol 1e-4)\n",
                  checked);
        return true;
    }
};

MNNTestSuiteRegister(SGFP4VulkanEncodeParityTest, "op/sgfp4/vulkan_encode_parity");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
