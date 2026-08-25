//
//  VulkanSGFP4Dequant.cpp
//  MNN
//
//  Created by MNN on 2026/08/24.
//  Copyright © 2018, Alibaba Group Holding Limited.
//

#include <fstream>

#include "VulkanSGFP4Dequant.hpp"
#include "MNN/SGFP4DequantUtils.hpp"
#include "VulkanBackend.hpp"
#include "core/FileLoader.hpp"
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

namespace {

// Must equal local_size_x in sgfp4_dequant.comp (named on both sides so the
// link is greppable, no bare 256 at the dispatch site).
constexpr uint32_t kSgfp4WorkgroupSize = 256;

struct SGFP4DequantConst {
    uint32_t outElementCount;
    uint32_t containerBytes;
};

// FileLoader::size() only reflects bytes already pulled into its internal
// cache blocks; it is NOT a filesystem stat. A real on-disk size probe is
// required to bound the DoS check below (T-03-02) against the sidecar's
// actual size BEFORE any container or VulkanBuffer allocation. Exact clone
// of CPUSGFP4Dequant.cpp's helper (Phase-1-tested).
bool queryFileSize(const std::string& path, size_t& outSize) {
    std::ifstream probe(path, std::ios::binary | std::ios::ate);
    if (!probe.is_open()) {
        return false;
    }
    auto pos = probe.tellg();
    if (pos < 0) {
        return false;
    }
    outSize = static_cast<size_t>(pos);
    return true;
}

} // namespace

VulkanSGFP4Dequant::VulkanSGFP4Dequant(Backend* bn, std::vector<uint8_t> container, uint32_t outElementCount,
                                       bool useFP32Output)
    : VulkanBasicExecution(bn) {
    mUseFP32Output  = useFP32Output;
    mOutElementCount = outElementCount;
    mContainer       = std::move(container);
    mContainerBytes  = static_cast<uint32_t>(mContainer.size());

    auto vkBn = static_cast<VulkanBackend*>(backend());
    mConstBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, sizeof(SGFP4DequantConst), nullptr,
                                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // binding 0: Container[] (validated bytes)
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // binding 1: Dst[]
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // binding 2: ConstBuffer
    };

    // Shader-variant selection per D-04/D-06, cloning VulkanFP4Dequant:
    // FP16 default, FP32 via flag or backend capability.
    std::string shaderName;
    if (mUseFP32Output) {
        shaderName = "glsl_sgfp4_dequant_comp";
    } else if (vkBn->useFP16()) {
        shaderName = "glsl_sgfp4_dequant_FP16_comp";
    } else {
        shaderName = "glsl_sgfp4_dequant_comp";
    }

    mDequantPipeline = vkBn->getPipeline(shaderName, types);
    mDescriptorSet.reset(mDequantPipeline->createSet());

    // D-01: upload AFTER validation (the creator only reaches this ctor with
    // validated bytes). Host-data ctor keeps onEncode copy-free.
    mContainerBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, mContainerBytes,
                                                      mContainer.data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

VulkanSGFP4Dequant::~VulkanSGFP4Dequant() {
}

ErrorCode VulkanSGFP4Dequant::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const VulkanCommandPool::Buffer* cmdBuffer) {
    (void)inputs; // 0-input Const-like op: source is the validated container SSBO
    auto output = outputs[0];
    auto extra  = static_cast<VulkanBackend*>(backend());

    uint32_t elementCount = static_cast<uint32_t>(output->elementSize());

    {
        auto c = reinterpret_cast<SGFP4DequantConst*>(mConstBuffer->map());
        c->outElementCount = elementCount;
        c->containerBytes  = mContainerBytes;
        mConstBuffer->unmap();
    }

    auto outputBuffer = extra->getTensorBuffer(output);
    auto outputSize   = extra->getTensorSize(output);

    // Binding indices match sgfp4_dequant.comp exactly.
    mDescriptorSet->writeBuffer(mContainerBuffer->buffer(), 0, mContainerBuffer->size(), 0);
    mDescriptorSet->writeBuffer(outputBuffer.first->buffer(), 1, outputSize, outputBuffer.second);
    mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());

    mDequantPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(elementCount, kSgfp4WorkgroupSize), 1, 1);

    // Output barrier: make output visible to downstream ops.
    cmdBuffer->barrierSource(outputBuffer.first->buffer(), outputBuffer.second, outputSize);

    return NO_ERROR;
}

class VulkanSGFP4DequantCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs,
                                            const std::vector<Tensor*>& outputs,
                                            const MNN::Op* op,
                                            Backend* backend) const override {
        (void)inputs; // 0-input op
        auto param = op->main_as_SGFP4DequantParam();
        if (nullptr == param) {
            MNN_ERROR("VulkanSGFP4Dequant: missing SGFP4DequantParam\n");
            return nullptr;
        }
        // Same external-sidecar gate as CPUSGFP4Dequant (ConvolutionCommon
        // USE_EXTERNAL_DATA + externalPath pattern).
        if (!USE_EXTERNAL_DATA(param) || nullptr == op->externalPath()) {
            MNN_ERROR("VulkanSGFP4Dequant: op requires external sidecar data\n");
            return nullptr;
        }
        auto external = param->external()->data();
        int64_t offset = external[0];
        int64_t size   = external[1];
        if (offset < 0 || size <= 0) {
            MNN_ERROR("VulkanSGFP4Dequant: invalid external offset/size\n");
            return nullptr;
        }

        // T-03-02: probe the real on-disk size BEFORE any allocation.
        size_t fileSize = 0;
        if (!queryFileSize(op->externalPath()->str(), fileSize)) {
            MNN_ERROR("VulkanSGFP4Dequant: cannot open sidecar %s\n", op->externalPath()->c_str());
            return nullptr;
        }
        size_t offsetSize = static_cast<size_t>(offset);
        size_t readSize   = static_cast<size_t>(size);
        if (offsetSize > fileSize || readSize > fileSize - offsetSize) {
            MNN_ERROR("VulkanSGFP4Dequant: external {offset,size} exceeds sidecar size\n");
            return nullptr;
        }

        FileLoader loader(op->externalPath()->c_str(), true);
        if (!loader.valid()) {
            MNN_ERROR("VulkanSGFP4Dequant: FileLoader invalid for sidecar\n");
            return nullptr;
        }

        std::vector<uint8_t> container(readSize);
        loader.offset(offset);
        if (!loader.read(reinterpret_cast<char*>(container.data()), size)) {
            MNN_ERROR("VulkanSGFP4Dequant: bounded sidecar read failed\n");
            return nullptr;
        }

        // Output element count: from the output tensor (shape-inference
        // product), consistent with what onEncode will dispatch on.
        if (outputs.empty() || nullptr == outputs[0]) {
            MNN_ERROR("VulkanSGFP4Dequant: no output tensor\n");
            return nullptr;
        }
        auto elementCount = outputs[0]->elementSize();
        if (elementCount <= 0) {
            MNN_ERROR("VulkanSGFP4Dequant: empty output\n");
            return nullptr;
        }

        // D-05: one-time host pre-validation with the Phase-1-tested
        // fully-bounds-checked CPU walk (T-03-03). A false return prevents
        // Execution construction entirely: no upload, no dispatch, no
        // partial output writes.
        std::vector<float> scratch(static_cast<size_t>(elementCount));
        if (!dequant_sgfp4_container_cpu(container.data(), container.size(), scratch.data(),
                                         static_cast<size_t>(elementCount))) {
            MNN_ERROR("VulkanSGFP4Dequant: container failed host pre-validation\n");
            return nullptr;
        }
        scratch.clear();
        scratch.shrink_to_fit();

        // FP32 can later be forced from op parameters; FP16 default per D-04.
        bool useFP32Output = false;
        return new VulkanSGFP4Dequant(backend, std::move(container), static_cast<uint32_t>(elementCount),
                                      useFP32Output);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_SGFP4Dequant, new VulkanSGFP4DequantCreator);
    return true;
}();

} // namespace MNN
