//
//  VulkanFP4Dequant.hpp
//  MNN
//
//  Created by MNN on 2026/05/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanFP4Dequant_hpp
#define VulkanFP4Dequant_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanFP4Dequant : public VulkanBasicExecution {
public:
    VulkanFP4Dequant(Backend* bn, bool useFP32Output);
    virtual ~VulkanFP4Dequant();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mDequantPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    bool mUseFP32Output;
};

} // namespace MNN

#endif /* VulkanFP4Dequant_hpp */
