//
//  VulkanSGFP4Dequant.hpp
//  MNN
//
//  Created by MNN on 2026/08/24.
//  Copyright © 2018, Alibaba Group Holding Limited.
//

#ifndef VulkanSGFP4Dequant_hpp
#define VulkanSGFP4Dequant_hpp

#include <memory>
#include <vector>
#include <cstdint>
#include "VulkanBasicExecution.hpp"

namespace MNN {

// Decodes uniform-layout SGFP4 v2 external-sidecar containers on the Vulkan
// buffer backend. The creator loads + host-pre-validates the container
// bytes (dequant_sgfp4_container_cpu as the validator, D-05) and hands over
// ONLY validated bytes; the constructor then uploads them once to a storage
// buffer and onEncode is a pure bind + dispatch (0-input Const-like op).
class VulkanSGFP4Dequant : public VulkanBasicExecution {
public:
    // Plan 09-02: true {dimO, dimI} replace the bare outElementCount so
    // onEncode can derive the padded dispatch geometry (D-11a crop write).
    VulkanSGFP4Dequant(Backend* bn, std::vector<uint8_t> container, int dimO, int dimI, bool useFP32Output);
    virtual ~VulkanSGFP4Dequant();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mContainerBuffer; // binding 0: validated container bytes (SSBO)
    std::shared_ptr<VulkanBuffer> mConstBuffer;     // binding 2: {outElementCount, containerBytes}
    const VulkanPipeline* mDequantPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    std::vector<uint8_t> mContainer; // kept alive alongside the upload for host reference
    bool mUseFP32Output;
    uint32_t mContainerBytes;
    int mDimO;
    int mDimI;
};

} // namespace MNN

#endif /* VulkanSGFP4Dequant_hpp */
