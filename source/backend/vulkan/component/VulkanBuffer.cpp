//
//  VulkanBuffer.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/component/VulkanBuffer.hpp"
#include <string.h>
namespace MNN {

VulkanBuffer::VulkanBuffer(const VulkanMemoryPool& pool, bool separate, size_t size, const void* hostData,
                           VkBufferUsageFlags usage, VkSharingMode shared, VkFlags requirements_mask)
    : mPool(pool) {
    MNN_ASSERT(size > 0);
    mSize = size;
    mShared = shared;
    mBuffer = const_cast<VulkanMemoryPool&>(mPool).allocBuffer(size, usage, shared);
    mUsage = usage;

    VkMemoryRequirements memReq;
    mPool.device().getBufferMemoryRequirements(mBuffer, memReq);
    mMemory = const_cast<VulkanMemoryPool&>(mPool).allocMemory(memReq, requirements_mask, separate);
    if (nullptr == mMemory.first) {
        MNN_ERROR("VulkanBuffer allocMemory failed: request=%zu, memReq.size=%zu, memReq.align=%zu, typeBits=0x%x, reqMask=0x%x\n",
                  size, (size_t)memReq.size, (size_t)memReq.alignment, memReq.memoryTypeBits, (uint32_t)requirements_mask);
        MNN_ASSERT(false);
        return;
    }
    //        FUNC_PRINT(mMemory->type());
    auto realMem = (VulkanMemory*)mMemory.first;

    if (nullptr != hostData) {
        void* data = nullptr;
        auto mapRes = mPool.device().mapMemory(realMem->get(), mMemory.second, size, 0 /*flag, not used*/, &data);
        if (mapRes != VK_SUCCESS) {
            MNN_ERROR("VulkanBuffer mapMemory failed: vkResult=%d, request=%zu, offset=%zu, memReq.size=%zu\n", mapRes,
                      size, (size_t)mMemory.second, (size_t)memReq.size);
        }
        CALL_VK(mapRes);
        ::memcpy(data, hostData, size);
        mPool.device().unmapMemory(realMem->get());
    }
    auto bindRes = mPool.device().bindBufferMemory(mBuffer, realMem->get(), mMemory.second);
    if (bindRes != VK_SUCCESS) {
        MNN_ERROR("VulkanBuffer bindBufferMemory failed: vkResult=%d, request=%zu, memReq.size=%zu, memReq.align=%zu, offset=%zu\n",
                  bindRes, size, (size_t)memReq.size, (size_t)memReq.alignment, (size_t)mMemory.second);
    }
    CALL_VK(bindRes);
}

VulkanBuffer::~VulkanBuffer() {
    const_cast<VulkanMemoryPool&>(mPool).returnBuffer(mBuffer, mSize, mUsage, mShared);
    if (!mReleased) {
        const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory);
    }
}
void* VulkanBuffer::map(int start, int size) const {
    const auto& limits = mPool.device().proty().limits;
    if (size < 0) {
        size = mSize;
    }
    auto realMem = (VulkanMemory*)mMemory.first;
    void* data = nullptr;
    CALL_VK(mPool.device().mapMemory(realMem->get(), start + mMemory.second, size, 0, &data));
    return data;
}
void VulkanBuffer::unmap() const {
    auto realMem = (VulkanMemory*)mMemory.first;
    mPool.device().unmapMemory(realMem->get());
}
void VulkanBuffer::release() {
    if (mReleased) {
        return;
    }
    mReleased = true;
    const_cast<VulkanMemoryPool&>(mPool).returnMemory(mMemory);
}

void VulkanBuffer::flush(bool write, int start, int size) const {
    // Do nothing
}

} // namespace MNN
