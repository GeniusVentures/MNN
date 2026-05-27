//
//  VulkanFP4Dequant.cpp
//  MNN
//
//  Created by MNN on 2026/05/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanFP4Dequant.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

struct FP4DequantConst {
    uint32_t elementCount;
    uint32_t srcBytes;
};

VulkanFP4Dequant::VulkanFP4Dequant(Backend* bn, bool useFP32Output) : VulkanBasicExecution(bn) {
    mUseFP32Output = useFP32Output;
    auto vkBn = static_cast<VulkanBackend*>(backend());
    mConstBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, sizeof(FP4DequantConst), nullptr,
                                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,   // binding 0: SrcRaw[]
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,   // binding 1: Dst[]
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER    // binding 2: ConstBuffer
    };

    // Select shader pipeline per D-04: FP16 default, FP32 via flag
    std::string shaderName;
    if (mUseFP32Output) {
        // Force FP32 output variant
        shaderName = "glsl_fp4_dequant_comp";
    } else if (vkBn->useFP16()) {
        // Default FP16 for GPU bandwidth efficiency
        shaderName = "glsl_fp4_dequant_FP16_comp";
    } else {
        // Backend doesn't support FP16, fall back to FP32
        shaderName = "glsl_fp4_dequant_comp";
    }

    mDequantPipeline = vkBn->getPipeline(shaderName, types);
    mDescriptorSet.reset(mDequantPipeline->createSet());
}

VulkanFP4Dequant::~VulkanFP4Dequant() {
}

ErrorCode VulkanFP4Dequant::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                      const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto extra  = static_cast<VulkanBackend*>(backend());

    // Compute element count and packed source byte count (2 FP4 values per byte, D-03)
    uint32_t elementCount = static_cast<uint32_t>(output->elementSize());
    uint32_t srcBytes     = (elementCount + 1u) / 2u;

    // Write push constants via uniform buffer
    {
        auto dequantConst = reinterpret_cast<FP4DequantConst*>(mConstBuffer->map());
        dequantConst->elementCount = elementCount;
        dequantConst->srcBytes     = srcBytes;
        mConstBuffer->unmap();
    }

    auto inputBuffer  = extra->getTensorBuffer(input);
    auto outputBuffer = extra->getTensorBuffer(output);
    auto inputSize    = extra->getTensorSize(input);
    auto outputSize   = extra->getTensorSize(output);

    // Write descriptor set (binding indices must match fp4_dequant.comp layout exactly)
    mDescriptorSet->writeBuffer(inputBuffer.first->buffer(), 0, inputSize, inputBuffer.second);
    mDescriptorSet->writeBuffer(outputBuffer.first->buffer(), 1, outputSize, outputBuffer.second);
    mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());

    // Bind pipeline and dispatch (256 threads per workgroup matches local_size_x)
    mDequantPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(elementCount, 256), 1, 1);

    // Output barrier: make output visible to downstream ops
    cmdBuffer->barrierSource(outputBuffer.first->buffer(), outputBuffer.second, outputSize);

    return NO_ERROR;
}

class VulkanFP4DequantCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs,
                                            const std::vector<Tensor*>& outputs,
                                            const MNN::Op* op,
                                            Backend* backend) const override {
        // Default to FP16 output per D-04.
        // FP32 can be forced by checking op parameters or quantization attributes
        // (future: read FP32 flag from op->main_as_QuantizedFloatParam()).
        bool useFP32Output = false;
        return new VulkanFP4Dequant(backend, useFP32Output);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Dequantize, new VulkanFP4DequantCreator);
    return true;
}();

} // namespace MNN
