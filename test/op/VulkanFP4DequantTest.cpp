//
//  VulkanFP4DequantTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/05/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <cmath>
#include <cstring>
#include "MNN_generated.h"
#include "MNN/FP4DequantUtils.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/expr/Module.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

#define TEST_RANDOM_SEED 2024

/**
 * @brief Build a minimal Dequantize module using the OpT / Express API pattern.
 *
 * Constructs a single-op model with OpType_Dequantize and wraps it in a Module
 * that dispatches through the Vulkan backend (via the supplied RuntimeManager).
 *
 * The Dequantize op parameters are set for identity-scale dequantization
 * (scale=1.0, zeroPoint=0), which matches the FP4 E2M1 decode behavior where
 * the nibble encoding maps directly to float values without quantization math.
 */
static std::shared_ptr<Module> makeDequantModule(
        const std::shared_ptr<Executor::RuntimeManager>& rtmgr) {
    std::shared_ptr<MNN::OpT> dequantOp(new MNN::OpT);
    dequantOp->type       = MNN::OpType_Dequantize;
    dequantOp->main.type  = MNN::OpParameter_Dequantize;
    dequantOp->main.value = new MNN::DequantizeT;

    auto* dq = dequantOp->main.AsDequantize();
    dq->inputQuantizedParam.reset(new MNN::QuantizedParamT);
    dq->inputQuantizedParam->zeroPoint = 0;
    dq->inputQuantizedParam->scale     = 1.0f;
    dq->mode       = MNN::QuantizeMode_MIN_COMBINED;
    dq->modelFormat = MNN::ModeFormat_TENSORFLOW;
    dq->type       = MNN::DataType_DT_FLOAT;

    // Placeholder input (shape determined at runtime by the caller).
    auto placeholder = _Input();
    auto output      = Variable::create(Expr::create(dequantOp.get(), {placeholder}));
    auto buffer      = Variable::save({output});

    std::shared_ptr<Module> m(Module::load({}, {}, (uint8_t*)buffer.data(),
                                           buffer.size(), rtmgr));
    return m;
}

class VulkanFP4DequantTest : public MNNTestCase {
public:
    VulkanFP4DequantTest()  = default;
    virtual ~VulkanFP4DequantTest() = default;

    virtual bool run(int precision) {
        srand(TEST_RANDOM_SEED);

        // Vulkan availability guard (Phase 1 pattern).
        auto vulkanCreator = MNN::MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN);
        if (nullptr == vulkanCreator) {
            MNN_PRINT("Vulkan backend not available — skipping FP4 dequant test\n");
            return true;
        }

        // Build Vulkan schedule config (Phase 1 pattern).
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_VULKAN;
        MNN::BackendConfig backendConfig;
        backendConfig.precision = MNN::BackendConfig::Precision_High;
        backendConfig.memory    = MNN::BackendConfig::Memory_High;
        config.backendConfig    = &backendConfig;

        std::shared_ptr<Executor::RuntimeManager> rtmgr(
            Executor::RuntimeManager::createRuntimeManager(config));
        auto m = makeDequantModule(rtmgr);

        // Tolerance: wider for FP16 (precision == 3 maps to FP32Converter[3]).
        float rtol = (precision == 3) ? 0.02f : 0.01f;

        // ================================================================
        // Test case 1: E2M1 exact values (all 16 nibble encodings)
        // ================================================================
        {
            const int elementCount = 16;
            uint8_t packed[8];  // 16 nibbles → 8 bytes

            // Pack all 16 E2M1 values in order (low nibble first).
            for (int i = 0; i < 16; i += 2) {
                packed[i >> 1] = MNN::pack_fp4_byte(
                    static_cast<uint8_t>(i),       // low nibble  (even index)
                    static_cast<uint8_t>(i + 1));  // high nibble (odd index)
            }

            // CPU reference.
            float cpuRef[16];
            for (int i = 0; i < 16; ++i) {
                cpuRef[i] = MNN::dequant_e2m1_cpu(static_cast<uint8_t>(i));
            }

            // Build input VARP with packed data.
            std::vector<int> shape = {1, 1, 1, elementCount};
            auto inputVar = _Input(shape, NCHW, halide_type_of<int8_t>());
            {
                auto* ptr = inputVar->writeMap<int8_t>();
                ::memcpy(ptr, packed, sizeof(packed));
                inputVar->unMap();
            }

            auto outputVar = m->onForward({inputVar})[0];
            auto* outPtr   = outputVar->readMap<float>();
            auto outInfo   = outputVar->getInfo();

            if (outInfo->dim[3] != elementCount) {
                MNN_ERROR("VulkanFP4DequantTest: E2M1 exact values — unexpected output size %d (expected %d)\n",
                          outInfo->dim[3], elementCount);
                return false;
            }

            // Special handling for Inf/NaN: the shader may produce slightly different NaN representations.
            // Validate all finite values with relative error; skip Inf/NaN element-wise comparison.
            for (int i = 0; i < elementCount; ++i) {
                float expected = cpuRef[i];
                float actual   = outPtr[i];

                if (std::isnan(expected)) {
                    if (!std::isnan(actual)) {
                        MNN_ERROR("VulkanFP4DequantTest: E2M1 nibble 0x%X — expected NaN, got %f\n", i, actual);
                        return false;
                    }
                    continue;
                }
                if (std::isinf(expected)) {
                    if (!std::isinf(actual) || std::signbit(expected) != std::signbit(actual)) {
                        MNN_ERROR("VulkanFP4DequantTest: E2M1 nibble 0x%X — expected %f, got %f\n",
                                  i, expected, actual);
                        return false;
                    }
                    continue;
                }

                float absErr = std::fabs(actual - expected);
                float maxVal = std::max(std::fabs(expected), 1.0f);
                if (absErr > maxVal * rtol) {
                    MNN_ERROR("VulkanFP4DequantTest: E2M1 nibble 0x%X — expected %f, got %f (absErr=%e, rtol=%e)\n",
                              i, expected, actual, absErr, rtol);
                    return false;
                }
            }
            MNN_PRINT("VulkanFP4DequantTest: E2M1 exact values PASSED\n");
        }

        // ================================================================
        // Test case 2: Random packed FP4 at multiple sizes
        // ================================================================
        {
            std::vector<int> testSizes = {64, 256, 1024, 4096};

            for (auto elementCount : testSizes) {
                int packedBytes = (elementCount + 1) / 2;

                // Generate random nibbles and pack them.
                std::vector<uint8_t> packed(packedBytes, 0);
                for (int i = 0; i < elementCount; ++i) {
                    uint8_t nibble = static_cast<uint8_t>(rand() % 16);
                    int byteIdx    = i >> 1;
                    if (i & 1) {
                        packed[byteIdx] |= (nibble << 4);   // high nibble
                    } else {
                        packed[byteIdx] |= nibble;           // low nibble
                    }
                }

                // CPU reference.
                std::vector<float> cpuRef(elementCount);
                MNN::dequant_fp4_packed_cpu(packed.data(), cpuRef.data(), elementCount);

                // Build input VARP.
                std::vector<int> shape = {1, 1, 1, elementCount};
                auto inputVar = _Input(shape, NCHW, halide_type_of<int8_t>());
                {
                    auto* ptr = inputVar->writeMap<int8_t>();
                    ::memcpy(ptr, packed.data(), packedBytes);
                    inputVar->unMap();
                }

                auto outputVar = m->onForward({inputVar})[0];
                auto* outPtr   = outputVar->readMap<float>();
                auto outInfo   = outputVar->getInfo();

                int outSize = outInfo->dim[3];
                if (outSize != elementCount) {
                    MNN_ERROR("VulkanFP4DequantTest: random FP4 size %d — unexpected output size %d\n",
                              elementCount, outSize);
                    return false;
                }

                // FP32 comparison with rtol.
                if (!checkVectorByRelativeError<float>(outPtr, cpuRef.data(), elementCount, rtol)) {
                    MNN_ERROR("VulkanFP4DequantTest: random FP4 size %d FAILED\n", elementCount);
                    return false;
                }

                // For FP16 precision, additionally validate against FP16-rounded reference.
                if (precision == 3) {
                    std::vector<float> fp16Ref(elementCount);
                    for (int i = 0; i < elementCount; ++i) {
                        fp16Ref[i] = FP32Converter[3](cpuRef[i]);
                    }
                    if (!checkVectorByRelativeError<float>(outPtr, cpuRef.data(), fp16Ref.data(),
                                                            elementCount, rtol)) {
                        MNN_ERROR("VulkanFP4DequantTest: random FP4 size %d FP16 comparison FAILED\n",
                                  elementCount);
                        return false;
                    }
                }
            }
            MNN_PRINT("VulkanFP4DequantTest: random packed FP4 PASSED\n");
        }

        // ================================================================
        // Test case 3: Boundary conditions
        // ================================================================
        {
            // 3a. Zero elements — model should handle gracefully.
            {
                std::vector<int> shape = {1, 1, 1, 0};
                auto inputVar = _Input(shape, NCHW, halide_type_of<int8_t>());
                auto outputVar = m->onForward({inputVar})[0];
                auto outInfo = outputVar->getInfo();
                if (outInfo->dim[3] != 0) {
                    MNN_ERROR("VulkanFP4DequantTest: zero-element output size %d (expected 0)\n",
                              outInfo->dim[3]);
                    return false;
                }
                MNN_PRINT("VulkanFP4DequantTest: zero-element case PASSED\n");
            }

            // 3b. Odd element counts.
            {
                std::vector<int> oddSizes = {7, 15, 127};
                for (auto elementCount : oddSizes) {
                    int packedBytes = (elementCount + 1) / 2;
                    std::vector<uint8_t> packed(packedBytes, 0);
                    for (int i = 0; i < elementCount; ++i) {
                        uint8_t nibble = static_cast<uint8_t>(rand() % 16);
                        int byteIdx    = i >> 1;
                        if (i & 1) {
                            packed[byteIdx] |= (nibble << 4);
                        } else {
                            packed[byteIdx] |= nibble;
                        }
                    }

                    std::vector<float> cpuRef(elementCount);
                    MNN::dequant_fp4_packed_cpu(packed.data(), cpuRef.data(), elementCount);

                    std::vector<int> shape = {1, 1, 1, elementCount};
                    auto inputVar = _Input(shape, NCHW, halide_type_of<int8_t>());
                    {
                        auto* ptr = inputVar->writeMap<int8_t>();
                        ::memcpy(ptr, packed.data(), packedBytes);
                        inputVar->unMap();
                    }

                    auto outputVar = m->onForward({inputVar})[0];
                    auto* outPtr   = outputVar->readMap<float>();
                    auto outInfo   = outputVar->getInfo();

                    if (outInfo->dim[3] != elementCount) {
                        MNN_ERROR("VulkanFP4DequantTest: odd element count %d — unexpected output size %d\n",
                                  elementCount, outInfo->dim[3]);
                        return false;
                    }
                    if (!checkVectorByRelativeError<float>(outPtr, cpuRef.data(), elementCount, rtol)) {
                        MNN_ERROR("VulkanFP4DequantTest: odd element count %d FAILED\n", elementCount);
                        return false;
                    }
                }
                MNN_PRINT("VulkanFP4DequantTest: odd element counts PASSED\n");
            }

            // 3c. Large element count (multi-workgroup dispatch).
            {
                const int elementCount = 65536;
                int packedBytes = (elementCount + 1) / 2;
                std::vector<uint8_t> packed(packedBytes, 0);
                for (int i = 0; i < elementCount; ++i) {
                    uint8_t nibble = static_cast<uint8_t>(rand() % 16);
                    int byteIdx    = i >> 1;
                    if (i & 1) {
                        packed[byteIdx] |= (nibble << 4);
                    } else {
                        packed[byteIdx] |= nibble;
                    }
                }

                // CPU reference — verify first and last 128 elements for large test.
                std::vector<float> cpuRef(elementCount);
                MNN::dequant_fp4_packed_cpu(packed.data(), cpuRef.data(), elementCount);

                std::vector<int> shape = {1, 1, 1, elementCount};
                auto inputVar = _Input(shape, NCHW, halide_type_of<int8_t>());
                {
                    auto* ptr = inputVar->writeMap<int8_t>();
                    ::memcpy(ptr, packed.data(), packedBytes);
                    inputVar->unMap();
                }

                auto outputVar = m->onForward({inputVar})[0];
                auto* outPtr   = outputVar->readMap<float>();
                auto outInfo   = outputVar->getInfo();

                if (outInfo->dim[3] != elementCount) {
                    MNN_ERROR("VulkanFP4DequantTest: large element count — unexpected output size %d\n",
                              outInfo->dim[3]);
                    return false;
                }
                if (!checkVectorByRelativeError<float>(outPtr, cpuRef.data(), elementCount, rtol)) {
                    MNN_ERROR("VulkanFP4DequantTest: large element count %d FAILED\n", elementCount);
                    return false;
                }
                MNN_PRINT("VulkanFP4DequantTest: large element count PASSED\n");
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(VulkanFP4DequantTest, "op/vulkan/fp4_dequant_correctness");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
