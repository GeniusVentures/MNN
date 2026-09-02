//
//  SGFP4VulkanDequantTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/08/24.
//  Copyright © 2018, Alibaba Group Holding Limited.
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <vector>

#include "MNN_generated.h"
#include "MNN/SGFP4DequantUtils.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/expr/Executor.hpp"
#include "MNN/expr/Module.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "SGFP4DequantFixtures.h"

using namespace MNN::Express;

namespace {

// Tight pass: Precision_High forces FP32 tensors + the FP32 shader variant,
// so GPU-vs-CPU agreement is limited only by ordinary float32 rounding.
constexpr float kFixtureRelativeTolerance = 1e-4f;
// Relaxed pass: default precision may select the FP16 shader variant; FP16
// output storage warrants a looser tolerance (Pitfall 2).
constexpr float kFp16RelativeTolerance = 2e-3f;

// Run one fixture through a Vulkan module session using the SAME production
// external-sidecar plumbing as the CPU path: {0, size} descriptor and
// op->externalPath set directly on the OpT (this op type is not one of the
// types OpCommonUtils rewrites with a session-derived externalPath).
// Returns false (with an MNN_ERROR naming the fixture) on any failure.
bool runSgfp4VulkanModule(const std::string& sidecarPath, const sgfp4_fixtures::Fixture& fixture,
                          const float* cpuRef, float rtol, bool highPrecision) {
    std::shared_ptr<MNN::OpT> op(new MNN::OpT);
    op->type      = MNN::OpType_SGFP4Dequant;
    op->main.type = MNN::OpParameter_SGFP4DequantParam;
    auto* param   = new MNN::SGFP4DequantParamT;
    param->magic   = MNN::kSGFP4Magic;
    param->external = {0, static_cast<int64_t>(fixture.containerSize)};
    param->dims     = {fixture.dimO, fixture.dimI};
    op->main.value  = param;
    // Op.externalPath must be set directly here — VulkanSGFP4Dequant's
    // creator reads op->externalPath() (same as CPUSGFP4Dequant).
    op->externalPath = sidecarPath;

    // SGFP4Dequant is a 0-input, Const-like source op.
    auto output = Variable::create(Expr::create(op.get(), {}));
    auto buffer = Variable::save({output});

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_VULKAN;
    MNN::BackendConfig backendConfig;
    if (highPrecision) {
        backendConfig.precision = MNN::BackendConfig::Precision_High;
    }
    backendConfig.memory = MNN::BackendConfig::Memory_High;
    config.backendConfig = &backendConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setExternalFile(sidecarPath);

    std::shared_ptr<Module> m(Module::load({}, {}, reinterpret_cast<const uint8_t*>(buffer.data()),
                                            buffer.size(), rtmgr));
    if (nullptr == m) {
        MNN_ERROR("SGFP4VulkanDequantTest: Module::load returned null for '%s'\n", fixture.name);
        return false;
    }

    auto outputs = m->onForward({});
    if (outputs.empty()) {
        MNN_ERROR("SGFP4VulkanDequantTest: module produced no outputs for '%s'\n", fixture.name);
        return false;
    }
    auto outVar  = outputs[0];
    auto* outPtr = outVar->readMap<float>();
    auto outInfo = outVar->getInfo();
    if (nullptr == outPtr || nullptr == outInfo) {
        MNN_ERROR("SGFP4VulkanDequantTest: output has no data/info for '%s'\n", fixture.name);
        return false;
    }
    size_t outCount = static_cast<size_t>(outInfo->size);
    if (outCount != fixture.expectedCount) {
        MNN_ERROR("SGFP4VulkanDequantTest: '%s' output count %zu != expected %zu\n", fixture.name, outCount,
                  fixture.expectedCount);
        return false;
    }
    if (!checkVectorByRelativeError<float>(outPtr, cpuRef, static_cast<int>(outCount), rtol)) {
        MNN_ERROR("SGFP4VulkanDequantTest: GPU/CPU parity mismatch for '%s' (rtol=%e)\n", fixture.name, rtol);
        return false;
    }
    return true;
}

} // namespace

class SGFP4VulkanDequantTest : public MNNTestCase {
public:
    SGFP4VulkanDequantTest()  = default;
    virtual ~SGFP4VulkanDequantTest() = default;

    virtual bool run(int precision) {
        (void)precision;

        // D-07 graceful skip: no Vulkan device → suite still passes.
        auto vulkanCreator = MNN::MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN);
        if (nullptr == vulkanCreator) {
            MNN_PRINT("Vulkan backend not available — skipping SGFP4 Vulkan parity test\n");
            return true;
        }

        int checked = 0;
        for (size_t i = 0; i < sgfp4_fixtures::kFixtureCount; ++i) {
            const sgfp4_fixtures::Fixture& fixture = sgfp4_fixtures::kFixtures[i];

            ++checked;

            // Per-fixture temp sidecar (unique name; removed after both passes).
            std::ostringstream oss;
            oss << "sgfp4_vk_sidecar_" << i << "_" << static_cast<unsigned long>(std::time(nullptr)) << "_"
                << static_cast<unsigned long>(rand()) << ".mnn.weight";
            std::string sidecarPath = oss.str();
            {
                std::ofstream ofs(sidecarPath, std::ios::binary | std::ios::trunc);
                if (!ofs) {
                    MNN_ERROR("SGFP4VulkanDequantTest: failed to open sidecar '%s' for write\n",
                              sidecarPath.c_str());
                    return false;
                }
                ofs.write(reinterpret_cast<const char*>(fixture.container),
                          static_cast<std::streamsize>(fixture.containerSize));
            }

            bool pass = true;
            // (b+c) CPU reference decode + fixture-drift guard.
            // Phase 12 codec fix: fixtures are in the normative SPATIAL
            // plane order now; the CPU reference decode must be spatial too.
            std::vector<float> cpuOut(fixture.expectedCount);
            {
                const int pdO = ((fixture.dimO + 63) / 64) * 64;
                const int pdI = ((fixture.dimI + 63) / 64) * 64;
                if (!MNN::dequant_sgfp4_container_cpu_crop(fixture.container, fixture.containerSize, cpuOut.data(),
                                                           fixture.dimO, fixture.dimI, pdO, pdI)) {
                    MNN_ERROR("SGFP4VulkanDequantTest: CPU reference decode failed for '%s'\n", fixture.name);
                    std::remove(sidecarPath.c_str());
                    return false;
                }
            }
            if (!checkVectorByRelativeError<float>(cpuOut.data(), fixture.expected,
                                                   static_cast<int>(fixture.expectedCount),
                                                   kFixtureRelativeTolerance)) {
                MNN_ERROR("SGFP4VulkanDequantTest: fixture '%s' drifted from CPU reference decode\n",
                          fixture.name);
                std::remove(sidecarPath.c_str());
                return false;
            }

            // (d+e) Tight GPU pass: Precision_High → FP32 shader variant.
            if (!runSgfp4VulkanModule(sidecarPath, fixture, cpuOut.data(), kFixtureRelativeTolerance, true)) {
                pass = false;
            }

            // Optional relaxed pass (D-06): default precision may select the
            // FP16 shader variant; harmless re-run at FP32 if the device
            // lacks FP16. Only if the tight pass already succeeded.
            if (pass &&
                !runSgfp4VulkanModule(sidecarPath, fixture, cpuOut.data(), kFp16RelativeTolerance, false)) {
                pass = false;
            }

            std::remove(sidecarPath.c_str());
            if (!pass) {
                return false;
            }
        }

        MNN_PRINT("SGFP4VulkanDequantTest: %d fixtures (including LAYOUT_MIXED) matched CPU reference on Vulkan "
                  "(FP32 tight + default-precision passes)\n",
                  checked);
        return true;
    }
};

MNNTestSuiteRegister(SGFP4VulkanDequantTest, "op/sgfp4/vulkan_uniform_parity");

// ===========================================================================
// op/sgfp4/vulkan_buffer_parity — Vulkan buffer-mode (inline param->buffer)
// decode parity (Plan 08-05, D-08): GPU buffer-mode decode == CPU oracle,
// using the SAME fixtures as the sidecar suite. No sidecar file is written
// and no setExternalFile is called — the container is inline. Pass-skips
// with no Vulkan device (D-07).
// ===========================================================================
class SGFP4VulkanBufferParityTest : public MNNTestCase {
public:
    SGFP4VulkanBufferParityTest()  = default;
    virtual ~SGFP4VulkanBufferParityTest() = default;

    virtual bool run(int precision) {
        (void)precision;

        // D-07 graceful skip: no Vulkan device → suite still passes.
        auto vulkanCreator = MNN::MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN);
        if (nullptr == vulkanCreator) {
            MNN_PRINT("Vulkan backend not available — skipping SGFP4 Vulkan buffer parity test\n");
            return true;
        }

        int checked = 0;
        for (size_t i = 0; i < sgfp4_fixtures::kFixtureCount; ++i) {
            const sgfp4_fixtures::Fixture& fixture = sgfp4_fixtures::kFixtures[i];
            ++checked;

            // CPU oracle reference (Phase 12 codec fix: spatial / normative).
            std::vector<float> cpuOut(fixture.expectedCount);
            {
                const int pdO = ((fixture.dimO + 63) / 64) * 64;
                const int pdI = ((fixture.dimI + 63) / 64) * 64;
                if (!MNN::dequant_sgfp4_container_cpu_crop(fixture.container, fixture.containerSize, cpuOut.data(),
                                                           fixture.dimO, fixture.dimI, pdO, pdI)) {
                    MNN_ERROR("SGFP4VulkanBufferParityTest: CPU reference decode failed for '%s'\n", fixture.name);
                    return false;
                }
            }

            // Tight pass (Precision_High → FP32 shader variant); relaxed
            // default-precision pass only if tight succeeded (D-06 mirror).
            bool pass = runBufferVulkanModule(fixture, cpuOut.data(), kFixtureRelativeTolerance, true);
            if (pass) {
                pass = runBufferVulkanModule(fixture, cpuOut.data(), kFp16RelativeTolerance, false);
            }
            if (!pass) {
                return false;
            }
        }

        MNN_PRINT("SGFP4VulkanBufferParityTest: %d fixtures matched CPU reference on Vulkan via inline buffer "
                  "(FP32 tight + default-precision passes)\n",
                  checked);
        return true;
    }

private:
    bool runBufferVulkanModule(const sgfp4_fixtures::Fixture& fixture, const float* cpuRef, float rtol,
                               bool highPrecision) {
        std::shared_ptr<MNN::OpT> op(new MNN::OpT);
        op->type      = MNN::OpType_SGFP4Dequant;
        op->main.type = MNN::OpParameter_SGFP4DequantParam;
        auto* param   = new MNN::SGFP4DequantParamT;
        param->magic  = MNN::kSGFP4Magic;
        // Buffer mode (D-01): inline bytes; external stays EMPTY and
        // externalPath stays UNSET (buffer-first branch, 08-03). No sidecar
        // file, no setExternalFile.
        param->buffer.assign(fixture.container, fixture.container + fixture.containerSize);
        param->dims    = {fixture.dimO, fixture.dimI};
        op->main.value = param;

        // 0-input Const-like source op.
        auto output = Variable::create(Expr::create(op.get(), {}));
        auto buffer = Variable::save({output});

        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_VULKAN;
        MNN::BackendConfig backendConfig;
        if (highPrecision) {
            backendConfig.precision = MNN::BackendConfig::Precision_High;
        }
        backendConfig.memory = MNN::BackendConfig::Memory_High;
        config.backendConfig = &backendConfig;
        std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));

        std::shared_ptr<Module> m(Module::load({}, {}, reinterpret_cast<const uint8_t*>(buffer.data()),
                                                buffer.size(), rtmgr));
        if (nullptr == m) {
            MNN_ERROR("SGFP4VulkanBufferParityTest: Module::load returned null for '%s'\n", fixture.name);
            return false;
        }

        auto outputs = m->onForward({});
        if (outputs.empty()) {
            MNN_ERROR("SGFP4VulkanBufferParityTest: module produced no outputs for '%s'\n", fixture.name);
            return false;
        }
        auto outVar  = outputs[0];
        auto* outPtr = outVar->readMap<float>();
        auto outInfo = outVar->getInfo();
        if (nullptr == outPtr || nullptr == outInfo) {
            MNN_ERROR("SGFP4VulkanBufferParityTest: output has no data/info for '%s'\n", fixture.name);
            return false;
        }
        size_t outCount = static_cast<size_t>(outInfo->size);
        if (outCount != fixture.expectedCount) {
            MNN_ERROR("SGFP4VulkanBufferParityTest: '%s' output count %zu != expected %zu\n", fixture.name,
                      outCount, fixture.expectedCount);
            return false;
        }
        if (!checkVectorByRelativeError<float>(outPtr, cpuRef, static_cast<int>(outCount), rtol)) {
            MNN_ERROR("SGFP4VulkanBufferParityTest: GPU/CPU buffer-mode parity mismatch for '%s' (rtol=%e)\n",
                      fixture.name, rtol);
            return false;
        }
        return true;
    }
};

MNNTestSuiteRegister(SGFP4VulkanBufferParityTest, "op/sgfp4/vulkan_buffer_parity");

#endif
