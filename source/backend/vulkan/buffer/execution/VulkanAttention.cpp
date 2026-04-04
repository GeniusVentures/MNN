#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "VulkanAttention.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
#include <climits>
#include <limits>

namespace MNN {

static inline float _invSqrt(float x) {
    return 1.0f / ::sqrtf(x);
}

static constexpr int kAttentionVecSize = 4;
static constexpr int kAttentionDispatchTile = 8;
static constexpr int kAttentionMaxHeadDim = 256;
static constexpr uint32_t kAttentionSoftmaxLocalSizeCap = 128;
static constexpr uint32_t kAttentionInitStateElementsPerDispatch = 256;
static constexpr size_t kAttentionQueryInputIndex = 0;
static constexpr size_t kAttentionKeyInputIndex = 1;
static constexpr size_t kAttentionValueInputIndex = 2;
static constexpr size_t kAttentionMaskInputIndex = 3;
static constexpr size_t kAttentionRequiredInputCount = 3;
static constexpr int kAttentionBatchSize = 1;
static constexpr int kScalarMaskElementCount = 1;
static constexpr int kMaskMinDimensions = 2;
static constexpr int kMaskQueryAxisOffset = 2;
static constexpr int kMaskKeyAxisOffset = 1;
static constexpr int kTurboQuantKBlockD4 = 8;
static constexpr int kTurboQuantKBlockSize = kTurboQuantKBlockD4 * kAttentionVecSize;
static constexpr int kTurboQuantKPackedWordCount = 4;

static inline int _getAttentionVecCount(int size) {
    return size / kAttentionVecSize;
}

static inline int _padToAttentionVec(int size) {
    return UP_DIV(size, kAttentionVecSize) * kAttentionVecSize;
}

static inline const Tensor* _getOptionalAttentionMask(const std::vector<Tensor*>& inputs) {
    if (inputs.size() <= kAttentionMaskInputIndex) {
        return nullptr;
    }
    return inputs[kAttentionMaskInputIndex];
}

static inline bool _useTurboQuantK(const KVMeta* meta, int headDim) {
    return nullptr != meta && meta->turboquant_k_enable && meta->turboquant_format == 0 &&
           meta->turboquant_block_size == kTurboQuantKBlockSize && headDim > 0 && (headDim % kTurboQuantKBlockSize) == 0;
}

static inline bool _useTurboQuantV(const KVMeta* meta, int headDim) {
    return nullptr != meta && meta->turboquant_v_enable && meta->turboquant_format == 0 &&
           meta->turboquant_block_size == kTurboQuantKBlockSize && headDim > 0 && (headDim % kTurboQuantKBlockSize) == 0;
}

static inline bool _supportAttentionPrefill(const std::vector<Tensor*>& inputs, bool needKvCache, int queryLen) {
    if (!needKvCache || queryLen <= 1) {
        return false;
    }
    auto mask = _getOptionalAttentionMask(inputs);
    if (nullptr == mask) {
        return true;
    }
    if (mask->elementSize() == kScalarMaskElementCount) {
        return mask->getType() == halide_type_of<float>();
    }
    MNN_ASSERT(mask->getType() == halide_type_of<float>());
    const int md = mask->dimensions();
    MNN_ASSERT(md >= kMaskMinDimensions);
    MNN_ASSERT(mask->length(md - kMaskQueryAxisOffset) == queryLen);
    MNN_ASSERT(mask->length(md - kMaskKeyAxisOffset) > 0);
    return false;
}

static inline bool _supportTurboQuantKPrefill(const std::vector<Tensor*>& inputs, bool needKvCache, int queryLen) {
    if (!needKvCache || queryLen <= 1) {
        return false;
    }
    const Tensor* mask = _getOptionalAttentionMask(inputs);
    if (nullptr == mask) {
        return true;
    }
    // Scalar causal-mask placeholders keep the prefill route, but remain on the dense-K path until the compressed
    // prefill implementation is proven numerically safe for that case.
    return false;
}

static inline int _getTurboQuantKBlockCount(int headDim) {
    return headDim / kTurboQuantKBlockSize;
}

static inline size_t _getTurboQuantKBufferSize(int maxLen, int kvHeadNum, int headDim) {
    return (size_t)maxLen * (size_t)kvHeadNum * (size_t)_getTurboQuantKBlockCount(headDim) *
           (size_t)kTurboQuantKPackedWordCount * sizeof(uint32_t);
}

static uint32_t _selectSoftmaxLocalSize(int totalLen, uint32_t maxSizeX, uint32_t maxInvocations) {
    if (totalLen <= 1) {
        return 1;
    }
    uint32_t cap = kAttentionSoftmaxLocalSizeCap;
    cap = ALIMIN(cap, maxSizeX);
    cap = ALIMIN(cap, maxInvocations);
    cap = ALIMIN(cap, (uint32_t)totalLen);
    uint32_t localSize = 1;
    while ((localSize << 1) <= cap) {
        localSize <<= 1;
    }
    return localSize;
}

static constexpr int kAttentionPrefillKBlock = 512;

static bool _supportDecodeQ1Subgroup(const VulkanDevice& device) {
    const auto& subgroup = device.getSubgroupInfo();
    if (0 == subgroup.size) {
        return false;
    }
    if (0 == (subgroup.stages & VK_SHADER_STAGE_COMPUTE_BIT)) {
        return false;
    }
    const VkSubgroupFeatureFlags required = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    if ((subgroup.ops & required) != required) {
        return false;
    }
    return true;
}

void VulkanAttention::KVCache::reset() {
    maxLen = 0;
    kvHeadNum = 0;
    headDim = 0;
    fp16 = false;
    key = nullptr;
    packedKey = nullptr;
    value = nullptr;
    packedValue = nullptr;
    turboQuantKBlockSize = 0;
    turboQuantVBlockSize = 0;
}

void VulkanAttention::KVCache::ensureCapacity(VulkanBackend* vkBn, int requiredLen, int kvH, int d, bool useFP16, bool useTurboQuantK,
                                              bool useTurboQuantV, int useTurboQuantBlockSize) {
    MNN_ASSERT(requiredLen >= 0);
    MNN_ASSERT(kvH > 0);
    MNN_ASSERT(d > 0);
    if (useTurboQuantK) {
        MNN_ASSERT(useTurboQuantBlockSize == kTurboQuantKBlockSize);
        MNN_ASSERT((d % kTurboQuantKBlockSize) == 0);
    }
    if (useTurboQuantV) {
        MNN_ASSERT(useTurboQuantBlockSize == kTurboQuantKBlockSize);
        MNN_ASSERT((d % kTurboQuantKBlockSize) == 0);
    }
    if (kvHeadNum != kvH || headDim != d || fp16 != useFP16 || nullptr == key || nullptr == value) {
        reset();
        kvHeadNum = kvH;
        headDim = d;
        fp16 = useFP16;
        maxLen = requiredLen + expandChunk;
        maxLen = ALIMAX(maxLen, expandChunk);
        const size_t bytes = fp16 ? sizeof(uint16_t) : sizeof(float);
        const size_t bufSize = (size_t)maxLen * (size_t)kvHeadNum * (size_t)headDim * bytes;
        key.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, bufSize, nullptr,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT));
        value.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, bufSize, nullptr,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    } else if (requiredLen > maxLen) {
        const int oldMaxLen = maxLen;
        maxLen = requiredLen + expandChunk;
        const size_t bytes = fp16 ? sizeof(uint16_t) : sizeof(float);
        const size_t newSize = (size_t)maxLen * (size_t)kvHeadNum * (size_t)headDim * bytes;
        std::shared_ptr<VulkanBuffer> newKey(new VulkanBuffer(vkBn->getMemoryPool(), false, newSize, nullptr,
                                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT));
        std::shared_ptr<VulkanBuffer> newValue(new VulkanBuffer(vkBn->getMemoryPool(), false, newSize, nullptr,
                                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT));
        // Preserve old content.
        //
        // cacheKey is packed as [kvHeadNum, headDim/4, maxLen, 4], so changing maxLen changes the row stride and we must repack.
        // cacheValue is kvh-major as [kvHeadNum, maxLen, headDim], so changing maxLen changes the kvh stride and we must repack too.
        const size_t oldSize = key->size();
        if (oldSize > 0) {
            // Value: repack kvh blocks with new stride.
            {
                const VkDeviceSize rowBytes = (VkDeviceSize)oldMaxLen * (VkDeviceSize)headDim * (VkDeviceSize)bytes;
                const VkDeviceSize srcStride = rowBytes;
                const VkDeviceSize dstStride = (VkDeviceSize)maxLen * (VkDeviceSize)headDim * (VkDeviceSize)bytes;
                std::vector<VkBufferCopy> regions;
                regions.reserve((size_t)kvHeadNum);
                for (int kvh = 0; kvh < kvHeadNum; ++kvh) {
                    VkBufferCopy c;
                    c.srcOffset = (VkDeviceSize)kvh * srcStride;
                    c.dstOffset = (VkDeviceSize)kvh * dstStride;
                    c.size = rowBytes;
                    regions.emplace_back(c);
                }
                vkBn->copyGPUToGPUBufferRegions(value->buffer(), newValue->buffer(), regions.data(), (uint32_t)regions.size());
            }

            // Key: repack rows with new stride.
            const int d4Size = _getAttentionVecCount(headDim);
            MNN_ASSERT(d4Size > 0);
            const uint32_t rowCount = (uint32_t)kvHeadNum * (uint32_t)d4Size;
            const VkDeviceSize vec4Bytes = (VkDeviceSize)(kAttentionVecSize * bytes);
            const VkDeviceSize srcRowStride = (VkDeviceSize)oldMaxLen * vec4Bytes;
            const VkDeviceSize dstRowStride = (VkDeviceSize)maxLen * vec4Bytes;
            std::vector<VkBufferCopy> regions;
            regions.reserve(rowCount);
            for (uint32_t r = 0; r < rowCount; ++r) {
                VkBufferCopy c;
                c.srcOffset = (VkDeviceSize)r * srcRowStride;
                c.dstOffset = (VkDeviceSize)r * dstRowStride;
                c.size = srcRowStride;
                regions.emplace_back(c);
            }
            vkBn->copyGPUToGPUBufferRegions(key->buffer(), newKey->buffer(), regions.data(), (uint32_t)regions.size());
        }
        key = newKey;
        value = newValue;
    }

    if (!useTurboQuantK) {
        packedKey = nullptr;
        turboQuantKBlockSize = 0;
    } else {
        const int turboQuantKBlockCount = _getTurboQuantKBlockCount(headDim);
        MNN_ASSERT(turboQuantKBlockCount > 0);
        const size_t packedSize = _getTurboQuantKBufferSize(maxLen, kvHeadNum, headDim);
        if (nullptr == packedKey || turboQuantKBlockSize != useTurboQuantBlockSize || packedKey->size() != packedSize) {
            std::shared_ptr<VulkanBuffer> newPackedKey(new VulkanBuffer(vkBn->getMemoryPool(), false, packedSize, nullptr,
                                                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT));
            if (nullptr != packedKey) {
                const VkDeviceSize srcStride = (VkDeviceSize)turboQuantKBlockCount * (VkDeviceSize)maxLen *
                                               (VkDeviceSize)kTurboQuantKPackedWordCount * (VkDeviceSize)sizeof(uint32_t);
                const VkDeviceSize dstStride = srcStride;
                std::vector<VkBufferCopy> regions;
                regions.reserve((size_t)kvHeadNum);
                for (int kvh = 0; kvh < kvHeadNum; ++kvh) {
                    VkBufferCopy c;
                    c.srcOffset = (VkDeviceSize)kvh * srcStride;
                    c.dstOffset = (VkDeviceSize)kvh * dstStride;
                    c.size = srcStride;
                    regions.emplace_back(c);
                }
                vkBn->copyGPUToGPUBufferRegions(packedKey->buffer(), newPackedKey->buffer(), regions.data(),
                                                (uint32_t)regions.size());
            }
            packedKey = newPackedKey;
        }
        turboQuantKBlockSize = useTurboQuantBlockSize;
    }

    if (!useTurboQuantV) {
        packedValue = nullptr;
        turboQuantVBlockSize = 0;
    } else {
        const int turboQuantVBlockCount = _getTurboQuantKBlockCount(headDim);
        MNN_ASSERT(turboQuantVBlockCount > 0);
        const size_t packedValueSize = _getTurboQuantKBufferSize(maxLen, kvHeadNum, headDim);
        if (nullptr == packedValue || turboQuantVBlockSize != useTurboQuantBlockSize || packedValue->size() != packedValueSize) {
            std::shared_ptr<VulkanBuffer> newPackedValue(new VulkanBuffer(vkBn->getMemoryPool(), false, packedValueSize, nullptr,
                                                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT));
            if (nullptr != packedValue) {
                const VkDeviceSize srcStride = (VkDeviceSize)turboQuantVBlockCount * (VkDeviceSize)maxLen *
                                               (VkDeviceSize)kTurboQuantKPackedWordCount * (VkDeviceSize)sizeof(uint32_t);
                const VkDeviceSize dstStride = srcStride;
                std::vector<VkBufferCopy> regions;
                regions.reserve((size_t)kvHeadNum);
                for (int kvh = 0; kvh < kvHeadNum; ++kvh) {
                    VkBufferCopy c;
                    c.srcOffset = (VkDeviceSize)kvh * srcStride;
                    c.dstOffset = (VkDeviceSize)kvh * dstStride;
                    c.size = srcStride;
                    regions.emplace_back(c);
                }
                vkBn->copyGPUToGPUBufferRegions(packedValue->buffer(), newPackedValue->buffer(), regions.data(),
                                                (uint32_t)regions.size());
            }
            packedValue = newPackedValue;
        }
        turboQuantVBlockSize = useTurboQuantBlockSize;
    }
}

VulkanAttention::VulkanAttention(const Op* op, Backend* bn) : VulkanBasicExecution(bn), mOp(op) {
    auto vkBn = static_cast<VulkanBackend*>(bn);
    mUseFP16 = vkBn->useFP16();
    mMeta = reinterpret_cast<KVMeta*>(vkBn->getMetaPtr());
    if (nullptr != op && nullptr != op->main_as_AttentionParam()) {
        mNeedKvCache = op->main_as_AttentionParam()->kv_cache();
    }
    mKVCache.reset(new KVCache);
    mParam = vkBn->allocUniform(nullptr, sizeof(GpuParam));
    mTurboQuantVParam = vkBn->allocUniform(nullptr, sizeof(TurboQuantVParam));
    if (!mNeedKvCache) {
        std::vector<VkDescriptorType> typesAttn{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // output
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // query
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // keyIn
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // valueIn
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheKey
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheValue
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // mask
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string attnName = "glsl_attention_fused_";
        if (mUseFP16) {
            attnName += "FP16_";
        }
        attnName += "comp";
        mAttentionLegacyPipeline = vkBn->getPipeline(attnName, typesAttn);
        MNN_ASSERT(nullptr != mAttentionLegacyPipeline);
        mAttentionLegacySet.reset(mAttentionLegacyPipeline->createSet());
        return;
    }

    // kv_cache=true path: pre-create update/prefill/decode pipelines to avoid resize cold-start.
    {
        std::vector<VkDescriptorType> typesUpdate{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // keyIn
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // valueIn
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheKey
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheValue
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // packedCacheKey
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // packedCacheValue
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // turboQuantVParam
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string updateName = "glsl_attention_kvcache_update_";
        if (mUseFP16) {
            updateName += "FP16_";
        }
        updateName += "comp";
        mUpdatePipeline = vkBn->getPipeline(updateName, typesUpdate);
        MNN_ASSERT(nullptr != mUpdatePipeline);
        mUpdateSet.reset(mUpdatePipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> typesRearrange{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // queryOut
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // queryIn
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string rqName = "glsl_attention_prefill_rearrange_q_";
        if (mUseFP16) {
            rqName += "FP16_";
        }
        rqName += "comp";
        mRearrangeQPipeline = vkBn->getPipeline(rqName, typesRearrange);
        MNN_ASSERT(nullptr != mRearrangeQPipeline);
        mRearrangeQSet.reset(mRearrangeQPipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> typesInit{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // m
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // l
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // alpha
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // oAcc
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string initName = "glsl_attention_prefill_kblock_init_state_";
        if (mUseFP16) {
            initName += "FP16_";
        }
        initName += "comp";
        mInitStatePipeline = vkBn->getPipeline(initName, typesInit);
        MNN_ASSERT(nullptr != mInitStatePipeline);
        mInitStateSet.reset(mInitStatePipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> typesQK{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // qk
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // query
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheKey
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // packedCacheKey
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // mask
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };

        std::string qkName = "glsl_attention_prefill_kblock_qk_";
        if (mUseFP16) {
            qkName += "FP16_";
        }
        qkName += "comp";
        mQKBlockPipeline = vkBn->getPipeline(qkName, typesQK);
        MNN_ASSERT(nullptr != mQKBlockPipeline);
        mQKBlockSet.reset(mQKBlockPipeline->createSet());

        std::string qkFullName = "glsl_attention_prefill_kblock_qk_full_";
        if (mUseFP16) {
            qkFullName += "FP16_";
        }
        qkFullName += "comp";
        mQKBlockFullPipeline = vkBn->getPipeline(qkFullName, typesQK);
        MNN_ASSERT(nullptr != mQKBlockFullPipeline);
        mQKBlockFullSet.reset(mQKBlockFullPipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> typesSoftmax{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // w
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // qk
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // m
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // l
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // alpha
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string softmaxName = "glsl_attention_prefill_kblock_softmax_online_";
        if (mUseFP16) {
            softmaxName += "FP16_";
        }
        softmaxName += "comp";
        const auto& limits = vkBn->getDevice().proty().limits;
        const int kBlock4 = UP_DIV(kAttentionPrefillKBlock, 4) * 4;
        const int maxK4 = UP_DIV(kBlock4, 4);
        uint32_t localSize = _selectSoftmaxLocalSize(maxK4, (uint32_t)limits.maxComputeWorkGroupSize[0],
                                                      (uint32_t)limits.maxComputeWorkGroupInvocations);
        mSoftmaxOnlinePipeline = vkBn->getPipeline(softmaxName, typesSoftmax, {localSize});
        MNN_ASSERT(nullptr != mSoftmaxOnlinePipeline);
        mSoftmaxOnlineSet.reset(mSoftmaxOnlinePipeline->createSet());
        mSoftmaxOnlineLocalSize = localSize;
    }

    {
        std::vector<VkDescriptorType> typesQKV{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // oAcc
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // w
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheValue
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // alpha
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // param
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // turboQuantVParam
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER  // packedCacheValue
        };
        std::string qkvName = "glsl_attention_prefill_kblock_qkv_acc_";
        if (mUseFP16) {
            qkvName += "FP16_";
        }
        qkvName += "comp";
        mQKVAccPipeline = vkBn->getPipeline(qkvName, typesQKV);
        MNN_ASSERT(nullptr != mQKVAccPipeline);
        mQKVAccSet.reset(mQKVAccPipeline->createSet());

        std::string qkvFullName = "glsl_attention_prefill_kblock_qkv_acc_full_";
        if (mUseFP16) {
            qkvFullName += "FP16_";
        }
        qkvFullName += "comp";
        mQKVAccFullPipeline = vkBn->getPipeline(qkvFullName, typesQKV);
        MNN_ASSERT(nullptr != mQKVAccFullPipeline);
        mQKVAccFullSet.reset(mQKVAccFullPipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> typesFinal{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // output
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // oAcc
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // l
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
        };
        std::string finalName = "glsl_attention_prefill_kblock_finalize_";
        if (mUseFP16) {
            finalName += "FP16_";
        }
        finalName += "comp";
        mFinalizePipeline = vkBn->getPipeline(finalName, typesFinal);
        MNN_ASSERT(nullptr != mFinalizePipeline);
        mFinalizeSet.reset(mFinalizePipeline->createSet());
    }

    {
        std::vector<VkDescriptorType> typesAttnFused{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // output
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // query
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // keyIn
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // valueIn
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheKey
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // packedCacheKey
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheValue
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // mask
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // param
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // turboQuantVParam
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER  // packedCacheValue
        };
        std::string attnName = "glsl_attention_fused_packed_";
        if (mUseFP16) {
            attnName += "FP16_";
        }
        attnName += "comp";
        mAttentionPipeline = vkBn->getPipeline(attnName, typesAttnFused);
        MNN_ASSERT(nullptr != mAttentionPipeline);
        mAttentionSet.reset(mAttentionPipeline->createSet());

        if (_supportDecodeQ1Subgroup(vkBn->getDevice())) {
            std::vector<VkDescriptorType> typesAttnDense{
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // output
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // query
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // keyIn
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // valueIn
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheKey
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // cacheValue
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // mask
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // param
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // turboQuantVParam
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER  // packedCacheValue
            };
            mDecodeQ1SubgroupLocalSize = vkBn->getDevice().getSubgroupSize();
            if (mDecodeQ1SubgroupLocalSize > 0) {
                std::string decodeQ1Name = "glsl_attention_decode_q1_subgroup_";
                if (mUseFP16) {
                    decodeQ1Name += "FP16_";
                }
                decodeQ1Name += "comp";
                mDecodeQ1SubgroupPipeline = vkBn->getPipeline(decodeQ1Name, typesAttnDense, {mDecodeQ1SubgroupLocalSize});
                if (nullptr != mDecodeQ1SubgroupPipeline) {
                    mDecodeQ1SubgroupSet.reset(mDecodeQ1SubgroupPipeline->createSet());
                }

                std::string decodeQ1HD128Name = "glsl_attention_decode_q1_subgroup_hd128_";
                if (mUseFP16) {
                    decodeQ1HD128Name += "FP16_";
                }
                decodeQ1HD128Name += "comp";
                mDecodeQ1SubgroupHD128Pipeline = vkBn->getPipeline(decodeQ1HD128Name, typesAttnDense, {mDecodeQ1SubgroupLocalSize});
                if (nullptr != mDecodeQ1SubgroupHD128Pipeline) {
                    mDecodeQ1SubgroupHD128Set.reset(mDecodeQ1SubgroupHD128Pipeline->createSet());
                }
            }
        }
    }
}

VulkanAttention::~VulkanAttention() {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    if (mTempQuery) {
        vkBn->onReleaseBuffer(mTempQuery.get(), Backend::DYNAMIC);
        mTempQuery.reset();
    }
    if (mTempQKBlock) {
        vkBn->onReleaseBuffer(mTempQKBlock.get(), Backend::DYNAMIC);
        mTempQKBlock.reset();
    }
    if (mTempWBlock) {
        vkBn->onReleaseBuffer(mTempWBlock.get(), Backend::DYNAMIC);
        mTempWBlock.reset();
    }
    if (mTempM) {
        vkBn->onReleaseBuffer(mTempM.get(), Backend::DYNAMIC);
        mTempM.reset();
    }
    if (mTempL) {
        vkBn->onReleaseBuffer(mTempL.get(), Backend::DYNAMIC);
        mTempL.reset();
    }
    if (mTempAlpha) {
        vkBn->onReleaseBuffer(mTempAlpha.get(), Backend::DYNAMIC);
        mTempAlpha.reset();
    }
    if (mTempOAcc) {
        vkBn->onReleaseBuffer(mTempOAcc.get(), Backend::DYNAMIC);
        mTempOAcc.reset();
    }
    if (mSyntheticMask) {
        vkBn->onReleaseBuffer(mSyntheticMask.get(), Backend::DYNAMIC);
        mSyntheticMask.reset();
    }
    vkBn->recycleUniform(mTurboQuantVParam);
    vkBn->recycleUniform(mParam);
}

bool VulkanAttention::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto res = new VulkanAttention(op, bn);
    res->mKVCache = mKVCache;
    res->mMeta = mMeta;
    *dst = res;
    return true;
}

ErrorCode VulkanAttention::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const VulkanCommandPool::Buffer* cmdBuffer) {
    MNN_ASSERT(inputs.size() >= kAttentionRequiredInputCount);
    MNN_ASSERT(!outputs.empty());
    auto query = inputs[kAttentionQueryInputIndex];
    auto key = inputs[kAttentionKeyInputIndex];
    auto value = inputs[kAttentionValueInputIndex];
    MNN_ASSERT(nullptr != query && nullptr != key && nullptr != value);
    MNN_ASSERT(query->dimensions() == 4);
    MNN_ASSERT(key->dimensions() == 4);
    MNN_ASSERT(value->dimensions() == 4);
    MNN_ASSERT(query->length(0) == kAttentionBatchSize);
    MNN_ASSERT(key->length(0) == kAttentionBatchSize);
    MNN_ASSERT(value->length(0) == kAttentionBatchSize);
    mQueryLen = query->length(1);
    mKeyLen = key->length(1);
    mHeadNum = query->length(2);
    mHeadDim = query->length(3);
    mKvHeadNum = key->length(2);
    MNN_ASSERT(mHeadNum > 0 && mKvHeadNum > 0);
    MNN_ASSERT(mHeadNum % mKvHeadNum == 0);
    MNN_ASSERT(mHeadDim > 0);
    MNN_ASSERT((mHeadDim & (kAttentionVecSize - 1)) == 0);
    MNN_ASSERT(mHeadDim <= kAttentionMaxHeadDim);
    MNN_ASSERT(value->length(1) == mKeyLen);
    MNN_ASSERT(value->length(2) == mKvHeadNum);
    MNN_ASSERT(value->length(3) == mHeadDim);

    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto cmd = cmdBuffer->get();

#ifdef ENABLE_VULKAN_TIME_PROFILE
    auto dispatchWithProfile = [&](const char* name, const VulkanPipeline* pipeline,
                                   const std::shared_ptr<VulkanLayout::DescriptorSet>& set, uint32_t x, uint32_t y,
                                   uint32_t z) {
        auto* profiler = vkBn->timeProfiler();
        if (nullptr != profiler) {
            VulkanTimeProfileScope scope(profiler, cmd, name, VulkanTimeProfiler::Kind::Shader);
            pipeline->bind(cmd, set->get());
            vkCmdDispatch(cmd, x, y, z);
            return;
        }
        pipeline->bind(cmd, set->get());
        vkCmdDispatch(cmd, x, y, z);
    };
#else
    auto dispatchWithProfile = [&](const char*, const VulkanPipeline* pipeline,
                                   const std::shared_ptr<VulkanLayout::DescriptorSet>& set, uint32_t x, uint32_t y,
                                   uint32_t z) {
        pipeline->bind(cmd, set->get());
        vkCmdDispatch(cmd, x, y, z);
    };
#endif

    const bool usePrefill = _supportAttentionPrefill(inputs, mNeedKvCache, mQueryLen);
    const bool useTurboQuantK = mNeedKvCache && _useTurboQuantK(mMeta, mHeadDim) &&
                                (mQueryLen == 1 || _supportTurboQuantKPrefill(inputs, mNeedKvCache, mQueryLen));
    mUsePrefill = usePrefill;

    if (mNeedKvCache) {
        MNN_ASSERT(nullptr != mUpdatePipeline);
        MNN_ASSERT(nullptr != mUpdateSet);

        // Dispatch: KV update (x=dim/4, y=keyLen, z=kvHeadNum).
        dispatchWithProfile(mUseFP16 ? "glsl_attention_kvcache_update_FP16_comp" : "glsl_attention_kvcache_update_comp",
                            mUpdatePipeline, mUpdateSet, UP_DIV(_getAttentionVecCount(mHeadDim), kAttentionDispatchTile), mKeyLen,
                            mKvHeadNum);
        // NOTE: KV cache buffers may be reallocated in onBeforeExecute (descriptor set updated there), so we must not
        // record a VkBufferMemoryBarrier with a stale VkBuffer handle here. Use a global memory barrier instead.
        {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier,
                                 0, nullptr, 0, nullptr);
        }
    }

    if (usePrefill) {
        constexpr int K_BLOCK = kAttentionPrefillKBlock;
        int pastLenForPrefill = 0;
        if (mNeedKvCache) {
            MNN_ASSERT(nullptr != mMeta);
            const int reverseSize = mMeta->computeReverseSize();
            const int reverse = reverseSize > 0 ? reverseSize : 0;
            const int previous = (int)mMeta->previous;
            const int remove = (int)mMeta->remove;
            MNN_ASSERT(previous >= 0);
            MNN_ASSERT(remove >= 0);
            MNN_ASSERT(remove <= previous);
            pastLenForPrefill = previous - remove + reverse;
        }
        mPrefillTotalLen = pastLenForPrefill + mKeyLen;
        mQueryLen4 = _padToAttentionVec(mQueryLen);
        MNN_ASSERT(mPrefillTotalLen > 0);

        const int64_t queryElementsI64 = (int64_t)mHeadNum * (int64_t)mHeadDim * (int64_t)mQueryLen4;
        MNN_ASSERT(queryElementsI64 > 0 && queryElementsI64 <= (int64_t)INT_MAX);
        const int queryElements = (int)queryElementsI64;

        if (!mTempQuery || (size_t)mTempQuery->elementSize() != (size_t)queryElements) {
            if (mTempQuery) {
                vkBn->onReleaseBuffer(mTempQuery.get(), Backend::DYNAMIC);
                mTempQuery.reset();
            }
            mTempQuery.reset(Tensor::createDevice<float>({queryElements}));
            bool res = vkBn->onAcquireBuffer(mTempQuery.get(), Backend::DYNAMIC);
            if (!res) {
                return OUT_OF_MEMORY;
            }
        }

        MNN_ASSERT(nullptr != mRearrangeQPipeline);
        MNN_ASSERT(nullptr != mRearrangeQSet);

        const int kBlock4 = _padToAttentionVec(K_BLOCK);
        const int64_t rowCountI64 = (int64_t)mQueryLen * (int64_t)mHeadNum;
        MNN_ASSERT(rowCountI64 > 0 && rowCountI64 <= (int64_t)INT_MAX);
        const int rowCount = (int)rowCountI64;

        const int64_t qkElementsI64 = (int64_t)rowCount * (int64_t)kBlock4;
        MNN_ASSERT(qkElementsI64 > 0 && qkElementsI64 <= (int64_t)INT_MAX);
        const int qkElements = (int)qkElementsI64;

        const int64_t oaccElementsI64 = (int64_t)rowCount * (int64_t)mHeadDim;
        MNN_ASSERT(oaccElementsI64 > 0 && oaccElementsI64 <= (int64_t)INT_MAX);
        const int oaccElements = (int)oaccElementsI64;

        if (!mTempQKBlock || (size_t)mTempQKBlock->elementSize() != (size_t)qkElements) {
            if (mTempQKBlock) {
                vkBn->onReleaseBuffer(mTempQKBlock.get(), Backend::DYNAMIC);
                mTempQKBlock.reset();
            }
            mTempQKBlock.reset(Tensor::createDevice<float>({qkElements}));
            if (!vkBn->onAcquireBuffer(mTempQKBlock.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }
        if (!mTempWBlock || (size_t)mTempWBlock->elementSize() != (size_t)qkElements) {
            if (mTempWBlock) {
                vkBn->onReleaseBuffer(mTempWBlock.get(), Backend::DYNAMIC);
                mTempWBlock.reset();
            }
            mTempWBlock.reset(Tensor::createDevice<float>({qkElements}));
            if (!vkBn->onAcquireBuffer(mTempWBlock.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }

        // State buffers must be FP32 even when VulkanBackend runs in FP16 mode. Use int tensors to force 4-byte storage.
        if (!mTempM || (size_t)mTempM->elementSize() != (size_t)rowCount) {
            if (mTempM) {
                vkBn->onReleaseBuffer(mTempM.get(), Backend::DYNAMIC);
                mTempM.reset();
            }
            mTempM.reset(Tensor::createDevice<int>({rowCount}));
            if (!vkBn->onAcquireBuffer(mTempM.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }
        if (!mTempL || (size_t)mTempL->elementSize() != (size_t)rowCount) {
            if (mTempL) {
                vkBn->onReleaseBuffer(mTempL.get(), Backend::DYNAMIC);
                mTempL.reset();
            }
            mTempL.reset(Tensor::createDevice<int>({rowCount}));
            if (!vkBn->onAcquireBuffer(mTempL.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }
        if (!mTempAlpha || (size_t)mTempAlpha->elementSize() != (size_t)rowCount) {
            if (mTempAlpha) {
                vkBn->onReleaseBuffer(mTempAlpha.get(), Backend::DYNAMIC);
                mTempAlpha.reset();
            }
            mTempAlpha.reset(Tensor::createDevice<int>({rowCount}));
            if (!vkBn->onAcquireBuffer(mTempAlpha.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }
        if (!mTempOAcc || (size_t)mTempOAcc->elementSize() != (size_t)oaccElements) {
            if (mTempOAcc) {
                vkBn->onReleaseBuffer(mTempOAcc.get(), Backend::DYNAMIC);
                mTempOAcc.reset();
            }
            mTempOAcc.reset(Tensor::createDevice<int>({oaccElements}));
            if (!vkBn->onAcquireBuffer(mTempOAcc.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }

        MNN_ASSERT(nullptr != mInitStatePipeline);
        MNN_ASSERT(nullptr != mInitStateSet);

        MNN_ASSERT(nullptr != mQKBlockPipeline);
        MNN_ASSERT(nullptr != mQKBlockSet);
        MNN_ASSERT(nullptr != mQKBlockFullPipeline);
        MNN_ASSERT(nullptr != mQKBlockFullSet);

        MNN_ASSERT(nullptr != mSoftmaxOnlinePipeline);
        MNN_ASSERT(nullptr != mSoftmaxOnlineSet);

        MNN_ASSERT(nullptr != mQKVAccPipeline);
        MNN_ASSERT(nullptr != mQKVAccSet);
        MNN_ASSERT(nullptr != mQKVAccFullPipeline);
        MNN_ASSERT(nullptr != mQKVAccFullSet);

        MNN_ASSERT(nullptr != mFinalizePipeline);
        MNN_ASSERT(nullptr != mFinalizeSet);

        // 1) Rearrange Q to packed-D Qtmp: (x=qLen4, y=headDim/4, z=headNum)
        dispatchWithProfile(mUseFP16 ? "glsl_attention_prefill_rearrange_q_FP16_comp" : "glsl_attention_prefill_rearrange_q_comp",
                            mRearrangeQPipeline, mRearrangeQSet, UP_DIV(mQueryLen4, kAttentionDispatchTile),
                            UP_DIV(_getAttentionVecCount(mHeadDim), kAttentionDispatchTile), mHeadNum);
        {
            auto qBuf = vkBn->getTensorBuffer(mTempQuery.get());
            cmdBuffer->barrierSource(qBuf.first->buffer(), qBuf.second, vkBn->getTensorSize(mTempQuery.get()));
        }

        // K-block prefill: online softmax in K dimension to avoid O(qLen*totalLen) intermediates.
        auto stateMBuf = vkBn->getTensorBuffer(mTempM.get());
        auto stateLBuf = vkBn->getTensorBuffer(mTempL.get());
        auto stateABuf = vkBn->getTensorBuffer(mTempAlpha.get());
        auto oaccBuf = vkBn->getTensorBuffer(mTempOAcc.get());

        dispatchWithProfile(mUseFP16 ? "glsl_attention_prefill_kblock_init_state_FP16_comp" : "glsl_attention_prefill_kblock_init_state_comp",
                            mInitStatePipeline, mInitStateSet,
                            UP_DIV((uint32_t)mQueryLen * (uint32_t)mHeadNum * (uint32_t)mHeadDim,
                                   kAttentionInitStateElementsPerDispatch),
                            1, 1);
        cmdBuffer->barrierSource(stateMBuf.first->buffer(), stateMBuf.second, vkBn->getTensorSize(mTempM.get()));
        cmdBuffer->barrierSource(stateLBuf.first->buffer(), stateLBuf.second, vkBn->getTensorSize(mTempL.get()));
        cmdBuffer->barrierSource(stateABuf.first->buffer(), stateABuf.second, vkBn->getTensorSize(mTempAlpha.get()));
        cmdBuffer->barrierSource(oaccBuf.first->buffer(), oaccBuf.second, vkBn->getTensorSize(mTempOAcc.get()));

        struct QKPushConst {
            uint32_t kStart;
            uint32_t blockLen;
        };
        struct SoftmaxPushConst {
            uint32_t blockLen;
        };

        auto dispatchWithPushConst = [&](const char* name, const VulkanPipeline* pipeline,
                                         const std::shared_ptr<VulkanLayout::DescriptorSet>& set, uint32_t x, uint32_t y,
                                         uint32_t z, const void* pcData, uint32_t pcSize) {
#ifdef ENABLE_VULKAN_TIME_PROFILE
            auto* profiler = vkBn->timeProfiler();
            if (nullptr != profiler) {
                VulkanTimeProfileScope scope(profiler, cmd, name, VulkanTimeProfiler::Kind::Shader);
                pipeline->bind(cmd, set->get());
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pcData);
                vkCmdDispatch(cmd, x, y, z);
                return;
            }
#endif
            pipeline->bind(cmd, set->get());
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pcData);
            vkCmdDispatch(cmd, x, y, z);
        };

        const int totalLen = mPrefillTotalLen;
        const int kBlock = K_BLOCK;
        for (int kStart = 0; kStart < totalLen; kStart += kBlock) {
            const int blockLen = ALIMIN(kBlock, totalLen - kStart);
            const int blockLen4 = _padToAttentionVec(blockLen);
            const int blockLen4_4 = _getAttentionVecCount(blockLen4);

            // 2) QK block: (x=blockLen4/4, y=qLen4/4, z=headNum)
            QKPushConst pcQK{(uint32_t)kStart, (uint32_t)blockLen};
            const bool fullBlock = (blockLen == kBlock) && (kStart + kBlock <= totalLen);
            const VulkanPipeline* qkPipe = fullBlock ? mQKBlockFullPipeline : mQKBlockPipeline;
            const std::shared_ptr<VulkanLayout::DescriptorSet>& qkSet = fullBlock ? mQKBlockFullSet : mQKBlockSet;
            const char* qkName = nullptr;
            if (fullBlock) {
                qkName = mUseFP16 ? "glsl_attention_prefill_kblock_qk_full_FP16_comp" : "glsl_attention_prefill_kblock_qk_full_comp";
            } else {
                qkName = mUseFP16 ? "glsl_attention_prefill_kblock_qk_FP16_comp" : "glsl_attention_prefill_kblock_qk_comp";
            }
            dispatchWithPushConst(qkName, qkPipe, qkSet, UP_DIV((uint32_t)blockLen4_4, kAttentionDispatchTile),
                                  UP_DIV((uint32_t)_getAttentionVecCount(mQueryLen4), kAttentionDispatchTile), (uint32_t)mHeadNum,
                                  &pcQK, sizeof(pcQK));
            {
                auto qkBuf = vkBn->getTensorBuffer(mTempQKBlock.get());
                cmdBuffer->barrierSource(qkBuf.first->buffer(), qkBuf.second, vkBn->getTensorSize(mTempQKBlock.get()));
            }

            // 3) Softmax online: updates m/l and writes unnormalized w (x=headNum, y=qLen)
            SoftmaxPushConst pcSM{(uint32_t)blockLen};
            dispatchWithPushConst(mUseFP16 ? "glsl_attention_prefill_kblock_softmax_online_FP16_comp"
                                           : "glsl_attention_prefill_kblock_softmax_online_comp",
                                  mSoftmaxOnlinePipeline, mSoftmaxOnlineSet, (uint32_t)mHeadNum, (uint32_t)mQueryLen, 1, &pcSM,
                                  sizeof(pcSM));
            {
                auto wBuf = vkBn->getTensorBuffer(mTempWBlock.get());
                cmdBuffer->barrierSource(wBuf.first->buffer(), wBuf.second, vkBn->getTensorSize(mTempWBlock.get()));
                cmdBuffer->barrierSource(stateMBuf.first->buffer(), stateMBuf.second, vkBn->getTensorSize(mTempM.get()));
                cmdBuffer->barrierSource(stateLBuf.first->buffer(), stateLBuf.second, vkBn->getTensorSize(mTempL.get()));
                cmdBuffer->barrierSource(stateABuf.first->buffer(), stateABuf.second, vkBn->getTensorSize(mTempAlpha.get()));
            }

            // 4) QKV accumulate: (x=headDim/4, y=qLen/2, z=headNum)
            const VulkanPipeline* qkvPipe = fullBlock ? mQKVAccFullPipeline : mQKVAccPipeline;
            const std::shared_ptr<VulkanLayout::DescriptorSet>& qkvSet = fullBlock ? mQKVAccFullSet : mQKVAccSet;
            const char* qkvName = nullptr;
            if (fullBlock) {
                qkvName = mUseFP16 ? "glsl_attention_prefill_kblock_qkv_acc_full_FP16_comp"
                                   : "glsl_attention_prefill_kblock_qkv_acc_full_comp";
            } else {
                qkvName =
                    mUseFP16 ? "glsl_attention_prefill_kblock_qkv_acc_FP16_comp" : "glsl_attention_prefill_kblock_qkv_acc_comp";
            }
            dispatchWithPushConst(qkvName, qkvPipe, qkvSet, UP_DIV((uint32_t)_getAttentionVecCount(mHeadDim), kAttentionDispatchTile),
                                  UP_DIV((uint32_t)UP_DIV(mQueryLen, 2), kAttentionDispatchTile), (uint32_t)mHeadNum, &pcQK,
                                  sizeof(pcQK));
            cmdBuffer->barrierSource(oaccBuf.first->buffer(), oaccBuf.second, vkBn->getTensorSize(mTempOAcc.get()));
        }

        // 5) Finalize: output = oAcc / l
        dispatchWithProfile(mUseFP16 ? "glsl_attention_prefill_kblock_finalize_FP16_comp" : "glsl_attention_prefill_kblock_finalize_comp",
                            mFinalizePipeline, mFinalizeSet, UP_DIV((uint32_t)_getAttentionVecCount(mHeadDim), kAttentionDispatchTile),
                            UP_DIV((uint32_t)UP_DIV(mQueryLen, 2), kAttentionDispatchTile), (uint32_t)mHeadNum);
        return NO_ERROR;
    }

    // Decode (or kv_cache disabled): keep fused shader.
    mQueryLen4 = 0;
    if (mTempQuery) {
        vkBn->onReleaseBuffer(mTempQuery.get(), Backend::DYNAMIC);
        mTempQuery.reset();
    }
    if (mTempQKBlock) {
        vkBn->onReleaseBuffer(mTempQKBlock.get(), Backend::DYNAMIC);
        mTempQKBlock.reset();
    }
    if (mTempWBlock) {
        vkBn->onReleaseBuffer(mTempWBlock.get(), Backend::DYNAMIC);
        mTempWBlock.reset();
    }
    if (mTempM) {
        vkBn->onReleaseBuffer(mTempM.get(), Backend::DYNAMIC);
        mTempM.reset();
    }
    if (mTempL) {
        vkBn->onReleaseBuffer(mTempL.get(), Backend::DYNAMIC);
        mTempL.reset();
    }
    if (mTempAlpha) {
        vkBn->onReleaseBuffer(mTempAlpha.get(), Backend::DYNAMIC);
        mTempAlpha.reset();
    }
    if (mTempOAcc) {
        vkBn->onReleaseBuffer(mTempOAcc.get(), Backend::DYNAMIC);
        mTempOAcc.reset();
    }
    mPrefillTotalLen = 0;

    if (mNeedKvCache) {
        const bool useDecodeQ1Subgroup =
            !useTurboQuantK && (mQueryLen == 1) && (nullptr != mDecodeQ1SubgroupPipeline) && (nullptr != mDecodeQ1SubgroupSet);
        if (useDecodeQ1Subgroup) {
            const bool useHD128 = (mHeadDim == 128) && (nullptr != mDecodeQ1SubgroupHD128Pipeline) &&
                                  (nullptr != mDecodeQ1SubgroupHD128Set);
            if (useHD128) {
                dispatchWithProfile(mUseFP16 ? "glsl_attention_decode_q1_subgroup_hd128_FP16_comp"
                                             : "glsl_attention_decode_q1_subgroup_hd128_comp",
                                    mDecodeQ1SubgroupHD128Pipeline, mDecodeQ1SubgroupHD128Set, (uint32_t)mHeadNum, 1, 1);
            } else {
                dispatchWithProfile(mUseFP16 ? "glsl_attention_decode_q1_subgroup_FP16_comp"
                                             : "glsl_attention_decode_q1_subgroup_comp",
                                    mDecodeQ1SubgroupPipeline, mDecodeQ1SubgroupSet, (uint32_t)mHeadNum, 1, 1);
            }
        } else {
            MNN_ASSERT(nullptr != mAttentionPipeline);
            MNN_ASSERT(nullptr != mAttentionSet);
            dispatchWithProfile(mUseFP16 ? "glsl_attention_fused_packed_FP16_comp" : "glsl_attention_fused_packed_comp",
                                mAttentionPipeline, mAttentionSet, UP_DIV(mHeadNum, 8), UP_DIV(mQueryLen, 8), 1);
        }
    } else {
        MNN_ASSERT(nullptr != mAttentionLegacyPipeline);
        MNN_ASSERT(nullptr != mAttentionLegacySet);
        dispatchWithProfile(mUseFP16 ? "glsl_attention_fused_FP16_comp" : "glsl_attention_fused_comp",
                            mAttentionLegacyPipeline, mAttentionLegacySet, UP_DIV(mHeadNum, 8), UP_DIV(mQueryLen, 8), 1);
    }

    return NO_ERROR;
}

ErrorCode VulkanAttention::onBeforeExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() >= kAttentionRequiredInputCount);
    MNN_ASSERT(!outputs.empty());
    auto query = inputs[kAttentionQueryInputIndex];
    auto key = inputs[kAttentionKeyInputIndex];
    auto value = inputs[kAttentionValueInputIndex];
    auto output = outputs[0];
    MNN_ASSERT(nullptr != query && nullptr != key && nullptr != value && nullptr != output);
    MNN_ASSERT(query->length(1) == mQueryLen);
    MNN_ASSERT(key->length(1) == mKeyLen);
    MNN_ASSERT(query->length(2) == mHeadNum);
    MNN_ASSERT(key->length(2) == mKvHeadNum);
    MNN_ASSERT(query->length(3) == mHeadDim);
    MNN_ASSERT(key->length(3) == mHeadDim);
    MNN_ASSERT(value->length(1) == mKeyLen);
    MNN_ASSERT(value->length(2) == mKvHeadNum);
    MNN_ASSERT(value->length(3) == mHeadDim);
    MNN_ASSERT(query->length(0) == kAttentionBatchSize);

    auto vkBn = static_cast<VulkanBackend*>(backend());

    int hasMask = 0;
    int lowerTriangularMask = 0;
    int maskQlen = 0;
    int maskKvlen = 0;
    const Tensor* mask = _getOptionalAttentionMask(inputs);
    if (nullptr != mask) {
        // Keep CUDA/OpenCL compatibility: scalar mask is a placeholder in kv-cache mode.
        if (mNeedKvCache && mask->elementSize() == kScalarMaskElementCount) {
            if (mask->getType() == halide_type_of<float>()) {
                lowerTriangularMask = 1;
            }
        } else {
            hasMask = 1;
            MNN_ASSERT(mask->getType() == halide_type_of<float>());
            const int md = mask->dimensions();
            MNN_ASSERT(md >= kMaskMinDimensions);
            maskQlen = mask->length(md - kMaskQueryAxisOffset);
            maskKvlen = mask->length(md - kMaskKeyAxisOffset);
            MNN_ASSERT(maskQlen == mQueryLen);
            MNN_ASSERT(maskKvlen > 0);
        }
    }

    const bool turboQuantKRequested = mNeedKvCache && _useTurboQuantK(mMeta, mHeadDim) &&
                                      (mQueryLen == 1 || _supportTurboQuantKPrefill(inputs, mNeedKvCache, mQueryLen));
    const bool turboQuantVRequested = mNeedKvCache && _useTurboQuantV(mMeta, mHeadDim);
    const int turboQuantKBlockSize = turboQuantKRequested ? mMeta->turboquant_block_size : 0;
    const int turboQuantVBlockSize = turboQuantVRequested ? mMeta->turboquant_block_size : 0;

    int pastLenForCompute = 0;
    if (mNeedKvCache) {
        MNN_ASSERT(nullptr != mMeta);
        const int reverseSize = mMeta->computeReverseSize();
        const int reverse = reverseSize > 0 ? reverseSize : 0;
        const int previous = (int)mMeta->previous;
        const int remove = (int)mMeta->remove;
        const int add = (int)mMeta->add;
        MNN_ASSERT(previous >= 0);
        MNN_ASSERT(remove >= 0);
        MNN_ASSERT(add >= 0);
        MNN_ASSERT(add <= mKeyLen);
        MNN_ASSERT(remove <= previous);

        const int start = previous - remove;
        pastLenForCompute = start + reverse;

        // Ensure capacity for compute window (pastLen + keyLen), because shaders read only from KV cache.
        mKVCache->ensureCapacity(vkBn, pastLenForCompute + mKeyLen, mKvHeadNum, mHeadDim, mUseFP16, turboQuantKRequested,
                                 turboQuantVRequested, ALIMAX(turboQuantKBlockSize, turboQuantVBlockSize));

        // Compact reserved spans into a contiguous kept region: dst starts at (previous - remove).
        if (mMeta->n_reserve > 0 && reverse > 0) {
            MNN_ASSERT(nullptr != mMeta->reserve);
            MNN_ASSERT(start >= 0);
            MNN_ASSERT(nullptr != mKVCache->key && nullptr != mKVCache->value);

            const size_t bytes = mUseFP16 ? sizeof(uint16_t) : sizeof(float);
            const int d4Size = mHeadDim / 4;
            MNN_ASSERT(d4Size > 0);

            std::shared_ptr<VulkanBuffer> compactKey(new VulkanBuffer(vkBn->getMemoryPool(), false, mKVCache->key->size(), nullptr,
                                                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT));
            std::shared_ptr<VulkanBuffer> compactValue(new VulkanBuffer(vkBn->getMemoryPool(), false, mKVCache->value->size(), nullptr,
                                                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT));

            const VkDeviceSize vec4Bytes = (VkDeviceSize)(4 * bytes);
            const VkDeviceSize keyRowStride = (VkDeviceSize)mKVCache->maxLen * vec4Bytes;
            const VkDeviceSize valueTokenBytes = (VkDeviceSize)mHeadDim * (VkDeviceSize)bytes;
            const VkDeviceSize valueHeadStride = (VkDeviceSize)mKVCache->maxLen * valueTokenBytes;

            std::vector<VkBufferCopy> keyRegions;
            std::vector<VkBufferCopy> valueRegions;
            keyRegions.reserve((size_t)mKvHeadNum * (size_t)d4Size * (size_t)mMeta->n_reserve);
            valueRegions.reserve((size_t)mKvHeadNum * (size_t)mMeta->n_reserve);

            int dstPos = 0;
            for (int n = 0; n < mMeta->n_reserve; ++n) {
                const int begin = mMeta->reserve[2 * n + 0];
                const int length = mMeta->reserve[2 * n + 1];
                MNN_ASSERT(begin >= 0);
                MNN_ASSERT(length > 0);

                const int srcPos = start + begin;
                const int dstBase = start + dstPos;
                MNN_ASSERT(srcPos >= 0);
                MNN_ASSERT(srcPos + length <= previous);
                MNN_ASSERT(srcPos + length <= mKVCache->maxLen);
                MNN_ASSERT(dstBase >= 0);
                MNN_ASSERT(dstBase + length <= mKVCache->maxLen);

                for (int kvh = 0; kvh < mKvHeadNum; ++kvh) {
                    VkBufferCopy valueCopy;
                    valueCopy.srcOffset = (VkDeviceSize)kvh * valueHeadStride + (VkDeviceSize)srcPos * valueTokenBytes;
                    valueCopy.dstOffset = (VkDeviceSize)kvh * valueHeadStride + (VkDeviceSize)dstBase * valueTokenBytes;
                    valueCopy.size = (VkDeviceSize)length * valueTokenBytes;
                    valueRegions.emplace_back(valueCopy);
                }

                const int rowBase = mKvHeadNum * d4Size;
                for (int row = 0; row < rowBase; ++row) {
                    VkBufferCopy keyCopy;
                    keyCopy.srcOffset = (VkDeviceSize)row * keyRowStride + (VkDeviceSize)srcPos * vec4Bytes;
                    keyCopy.dstOffset = (VkDeviceSize)row * keyRowStride + (VkDeviceSize)dstBase * vec4Bytes;
                    keyCopy.size = (VkDeviceSize)length * vec4Bytes;
                    keyRegions.emplace_back(keyCopy);
                }

                dstPos += length;
            }
            MNN_ASSERT(dstPos == reverse);

            if (!keyRegions.empty()) {
                vkBn->copyGPUToGPUBufferRegions(mKVCache->key->buffer(), compactKey->buffer(), keyRegions.data(),
                                                (uint32_t)keyRegions.size());
            }
            if (!valueRegions.empty()) {
                vkBn->copyGPUToGPUBufferRegions(mKVCache->value->buffer(), compactValue->buffer(), valueRegions.data(),
                                                (uint32_t)valueRegions.size());
            }

            mKVCache->key = compactKey;
            mKVCache->value = compactValue;
            if (turboQuantKRequested && nullptr != mKVCache->packedKey) {
                const int turboQuantKBlockCount = _getTurboQuantKBlockCount(mHeadDim);
                const VkDeviceSize packedTokenBytes = (VkDeviceSize)turboQuantKBlockCount *
                                                      (VkDeviceSize)kTurboQuantKPackedWordCount *
                                                      (VkDeviceSize)sizeof(uint32_t);
                const VkDeviceSize packedHeadStride = (VkDeviceSize)mKVCache->maxLen * packedTokenBytes;
                const size_t packedSize = _getTurboQuantKBufferSize(mKVCache->maxLen, mKvHeadNum, mHeadDim);
                std::shared_ptr<VulkanBuffer> compactPackedKey(new VulkanBuffer(vkBn->getMemoryPool(), false, packedSize, nullptr,
                                                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT));
                std::vector<VkBufferCopy> packedRegions;
                packedRegions.reserve((size_t)mKvHeadNum * (size_t)mMeta->n_reserve);
                int dstPosPacked = 0;
                for (int n = 0; n < mMeta->n_reserve; ++n) {
                    const int begin = mMeta->reserve[2 * n + 0];
                    const int length = mMeta->reserve[2 * n + 1];
                    const int srcPos = start + begin;
                    const int dstBase = start + dstPosPacked;
                    for (int kvh = 0; kvh < mKvHeadNum; ++kvh) {
                        VkBufferCopy c;
                        c.srcOffset = (VkDeviceSize)kvh * packedHeadStride + (VkDeviceSize)srcPos * packedTokenBytes;
                        c.dstOffset = (VkDeviceSize)kvh * packedHeadStride + (VkDeviceSize)dstBase * packedTokenBytes;
                        c.size = (VkDeviceSize)length * packedTokenBytes;
                        packedRegions.emplace_back(c);
                    }
                    dstPosPacked += length;
                }
                if (!packedRegions.empty()) {
                    vkBn->copyGPUToGPUBufferRegions(mKVCache->packedKey->buffer(), compactPackedKey->buffer(), packedRegions.data(),
                                                    (uint32_t)packedRegions.size());
                }
                mKVCache->packedKey = compactPackedKey;
            }
            if (turboQuantVRequested && nullptr != mKVCache->packedValue) {
                const int turboQuantVBlockCount = _getTurboQuantKBlockCount(mHeadDim);
                const VkDeviceSize packedTokenBytes = (VkDeviceSize)turboQuantVBlockCount *
                                                      (VkDeviceSize)kTurboQuantKPackedWordCount *
                                                      (VkDeviceSize)sizeof(uint32_t);
                const VkDeviceSize packedHeadStride = (VkDeviceSize)mKVCache->maxLen * packedTokenBytes;
                const size_t packedSize = _getTurboQuantKBufferSize(mKVCache->maxLen, mKvHeadNum, mHeadDim);
                std::shared_ptr<VulkanBuffer> compactPackedValue(new VulkanBuffer(vkBn->getMemoryPool(), false, packedSize, nullptr,
                                                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT));
                std::vector<VkBufferCopy> packedRegions;
                packedRegions.reserve((size_t)mKvHeadNum * (size_t)mMeta->n_reserve);
                int dstPosPacked = 0;
                for (int n = 0; n < mMeta->n_reserve; ++n) {
                    const int begin = mMeta->reserve[2 * n + 0];
                    const int length = mMeta->reserve[2 * n + 1];
                    const int srcPos = start + begin;
                    const int dstBase = start + dstPosPacked;
                    for (int kvh = 0; kvh < mKvHeadNum; ++kvh) {
                        VkBufferCopy c;
                        c.srcOffset = (VkDeviceSize)kvh * packedHeadStride + (VkDeviceSize)srcPos * packedTokenBytes;
                        c.dstOffset = (VkDeviceSize)kvh * packedHeadStride + (VkDeviceSize)dstBase * packedTokenBytes;
                        c.size = (VkDeviceSize)length * packedTokenBytes;
                        packedRegions.emplace_back(c);
                    }
                    dstPosPacked += length;
                }
                if (!packedRegions.empty()) {
                    vkBn->copyGPUToGPUBufferRegions(mKVCache->packedValue->buffer(), compactPackedValue->buffer(),
                                                    packedRegions.data(), (uint32_t)packedRegions.size());
                }
                mKVCache->packedValue = compactPackedValue;
            }
        }
    }

    const int group = mHeadNum / mKvHeadNum;
    const int totalLenForCompute = pastLenForCompute + mKeyLen;

    if (lowerTriangularMask != 0) {
        const int maskElements = mQueryLen * totalLenForCompute;
        MNN_ASSERT(maskElements > 0);
        if (!mSyntheticMask || mSyntheticMask->elementSize() != maskElements) {
            if (mSyntheticMask) {
                vkBn->onReleaseBuffer(mSyntheticMask.get(), Backend::DYNAMIC);
                mSyntheticMask.reset();
            }
            mSyntheticMask.reset(Tensor::createDevice<float>({mQueryLen, totalLenForCompute}));
            if (!vkBn->onAcquireBuffer(mSyntheticMask.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }

        std::shared_ptr<Tensor> hostMask(Tensor::create<float>({mQueryLen, totalLenForCompute}));
        auto hostMaskPtr = hostMask->host<float>();
        const float negativeInfinity = -std::numeric_limits<float>::max();
        for (int q = 0; q < mQueryLen; ++q) {
            const int causalLimit = pastLenForCompute + q;
            for (int k = 0; k < totalLenForCompute; ++k) {
                hostMaskPtr[q * totalLenForCompute + k] = (k <= causalLimit) ? 0.0f : negativeInfinity;
            }
        }
        vkBn->onCopyBuffer(hostMask.get(), mSyntheticMask.get());
        mask = mSyntheticMask.get();
        hasMask = 1;
        maskQlen = mQueryLen;
        maskKvlen = totalLenForCompute;
        lowerTriangularMask = 0;
    }
    mHasAttentionMask = (hasMask != 0);
    const bool useTurboQuantK = turboQuantKRequested;

    auto gpuParam = reinterpret_cast<GpuParam*>(mParam->map());
    gpuParam->s0[0] = mQueryLen;
    gpuParam->s0[1] = mKeyLen;
    gpuParam->s0[2] = mHeadNum;
    gpuParam->s0[3] = mKvHeadNum;
    gpuParam->s1[0] = mHeadDim;
    gpuParam->s1[1] = group;
    gpuParam->s1[2] = pastLenForCompute;
    gpuParam->s1[3] = totalLenForCompute;
    gpuParam->s2[0] = maskQlen;
    gpuParam->s2[1] = maskKvlen;
    gpuParam->s2[2] = hasMask;
    gpuParam->s2[3] = mNeedKvCache ? mKVCache->maxLen : 0;
    gpuParam->f0[0] = _invSqrt((float)mHeadDim);
    const float sparseTau = (mNeedKvCache && nullptr != mMeta && mMeta->sparse_v_enable) ? mMeta->sparse_v_tau : -1.0f;
    gpuParam->f0[1] = sparseTau;
    gpuParam->f0[2] = (float)lowerTriangularMask;
    gpuParam->f0[3] = useTurboQuantK ? (float)turboQuantKBlockSize : 0.0f;
    mParam->unmap();

    auto turboQuantVParam = reinterpret_cast<TurboQuantVParam*>(mTurboQuantVParam->map());
    turboQuantVParam->f0[0] = turboQuantVRequested ? 1.0f : 0.0f;
    turboQuantVParam->f0[1] = turboQuantVRequested ? (float)turboQuantVBlockSize : 0.0f;
    turboQuantVParam->f0[2] = 0.0f;
    turboQuantVParam->f0[3] = 0.0f;
    mTurboQuantVParam->unmap();

    // Bind buffers (update + attention). Note: when hasMask == 0, bind query buffer as placeholder.
    auto queryBuf = vkBn->getTensorBuffer(query);
    auto keyBuf = vkBn->getTensorBuffer(key);
    auto valueBuf = vkBn->getTensorBuffer(value);
    auto outBuf = vkBn->getTensorBuffer(output);
    const VkDeviceSize queryOffset = queryBuf.second;

    const VulkanBuffer* denseCacheKeyBuf = nullptr;
    const VulkanBuffer* packedCacheKeyBuf = nullptr;
    const VulkanBuffer* cacheValueBuf = nullptr;
    const VulkanBuffer* packedCacheValueBuf = nullptr;
    VkDeviceSize denseCacheKeyOffset = 0;
    VkDeviceSize packedCacheKeyOffset = 0;
    VkDeviceSize cacheValueOffset = 0;
    VkDeviceSize packedCacheValueOffset = 0;
    size_t denseCacheKeySize = 0;
    size_t packedCacheKeySize = 0;
    size_t cacheValueSize = 0;
    size_t packedCacheValueSize = 0;

    if (mNeedKvCache) {
        denseCacheKeyBuf = mKVCache->key.get();
        packedCacheKeyBuf = useTurboQuantK ? mKVCache->packedKey.get() : mKVCache->key.get();
        cacheValueBuf = mKVCache->value.get();
        packedCacheValueBuf = turboQuantVRequested ? mKVCache->packedValue.get() : mKVCache->value.get();
        MNN_ASSERT(nullptr != denseCacheKeyBuf && nullptr != packedCacheKeyBuf && nullptr != cacheValueBuf &&
                   nullptr != packedCacheValueBuf);
        denseCacheKeySize = denseCacheKeyBuf->size();
        packedCacheKeySize = packedCacheKeyBuf->size();
        cacheValueSize = cacheValueBuf->size();
        packedCacheValueSize = packedCacheValueBuf->size();
    } else {
        // KV cache disabled: alias cache buffers to current K/V (shaders read only from cache bindings).
        denseCacheKeyBuf = keyBuf.first;
        packedCacheKeyBuf = keyBuf.first;
        cacheValueBuf = valueBuf.first;
        packedCacheValueBuf = valueBuf.first;
        denseCacheKeyOffset = keyBuf.second;
        packedCacheKeyOffset = keyBuf.second;
        cacheValueOffset = valueBuf.second;
        packedCacheValueOffset = valueBuf.second;
        denseCacheKeySize = vkBn->getTensorSize(key);
        packedCacheKeySize = vkBn->getTensorSize(key);
        cacheValueSize = vkBn->getTensorSize(value);
        packedCacheValueSize = vkBn->getTensorSize(value);
    }

    // Update set (only when KV cache is enabled; kv_cache=false uses legacy fused shader directly on input K/V).
    if (mNeedKvCache) {
        mUpdateSet->writeBuffer(keyBuf.first->buffer(), 0, vkBn->getTensorSize(key), keyBuf.second);
        mUpdateSet->writeBuffer(valueBuf.first->buffer(), 1, vkBn->getTensorSize(value), valueBuf.second);
        mUpdateSet->writeBuffer(mKVCache->key->buffer(), 2, mKVCache->key->size(), 0);
        mUpdateSet->writeBuffer(cacheValueBuf->buffer(), 3, cacheValueSize, cacheValueOffset);
        mUpdateSet->writeBuffer((useTurboQuantK ? mKVCache->packedKey.get() : mKVCache->key.get())->buffer(), 4,
                                useTurboQuantK ? mKVCache->packedKey->size() : mKVCache->key->size(), 0);
        mUpdateSet->writeBuffer((turboQuantVRequested ? mKVCache->packedValue.get() : mKVCache->value.get())->buffer(), 5,
                                turboQuantVRequested ? mKVCache->packedValue->size() : mKVCache->value->size(), 0);
        mUpdateSet->writeBuffer(mTurboQuantVParam->buffer(), 6, mTurboQuantVParam->size());
        mUpdateSet->writeBuffer(mParam->buffer(), 7, mParam->size());
    }

    if (mUsePrefill) {
        MNN_ASSERT(totalLenForCompute == mPrefillTotalLen);
        MNN_ASSERT(mQueryLen4 == UP_DIV(mQueryLen, 4) * 4);

        MNN_ASSERT(nullptr != mTempQuery);
        auto tqBuf = vkBn->getTensorBuffer(mTempQuery.get());

        // Rearrange Q set: queryTmp <- query
        MNN_ASSERT(nullptr != mRearrangeQSet);
        mRearrangeQSet->writeBuffer(tqBuf.first->buffer(), 0, vkBn->getTensorSize(mTempQuery.get()), tqBuf.second);
        mRearrangeQSet->writeBuffer(queryBuf.first->buffer(), 1, vkBn->getTensorSize(query), queryBuf.second);
        mRearrangeQSet->writeBuffer(mParam->buffer(), 2, mParam->size());

        MNN_ASSERT(nullptr != mTempQKBlock && nullptr != mTempWBlock);
        MNN_ASSERT(nullptr != mTempM && nullptr != mTempL && nullptr != mTempAlpha && nullptr != mTempOAcc);
        auto qkBuf = vkBn->getTensorBuffer(mTempQKBlock.get());
        auto wBuf = vkBn->getTensorBuffer(mTempWBlock.get());
        auto mBuf = vkBn->getTensorBuffer(mTempM.get());
        auto lBuf = vkBn->getTensorBuffer(mTempL.get());
        auto aBuf = vkBn->getTensorBuffer(mTempAlpha.get());
        auto oBuf = vkBn->getTensorBuffer(mTempOAcc.get());

        // Init state set
        mInitStateSet->writeBuffer(mBuf.first->buffer(), 0, vkBn->getTensorSize(mTempM.get()), mBuf.second);
        mInitStateSet->writeBuffer(lBuf.first->buffer(), 1, vkBn->getTensorSize(mTempL.get()), lBuf.second);
        mInitStateSet->writeBuffer(aBuf.first->buffer(), 2, vkBn->getTensorSize(mTempAlpha.get()), aBuf.second);
        mInitStateSet->writeBuffer(oBuf.first->buffer(), 3, vkBn->getTensorSize(mTempOAcc.get()), oBuf.second);
        mInitStateSet->writeBuffer(mParam->buffer(), 4, mParam->size());

        // QK block set
        mQKBlockSet->writeBuffer(qkBuf.first->buffer(), 0, vkBn->getTensorSize(mTempQKBlock.get()), qkBuf.second);
        mQKBlockSet->writeBuffer(tqBuf.first->buffer(), 1, vkBn->getTensorSize(mTempQuery.get()), tqBuf.second);
        mQKBlockSet->writeBuffer(denseCacheKeyBuf->buffer(), 2, denseCacheKeySize, denseCacheKeyOffset);
        mQKBlockSet->writeBuffer(packedCacheKeyBuf->buffer(), 3, packedCacheKeySize, packedCacheKeyOffset);
        if (hasMask) {
            auto maskBuf = vkBn->getTensorBuffer(mask);
            mQKBlockSet->writeBuffer(maskBuf.first->buffer(), 4, vkBn->getTensorSize(mask), maskBuf.second);
        } else {
            mQKBlockSet->writeBuffer(queryBuf.first->buffer(), 4, vkBn->getTensorSize(query), queryBuf.second);
        }
        mQKBlockSet->writeBuffer(mParam->buffer(), 5, mParam->size());

        // QK full-block set (same bindings as tail-safe set)
        mQKBlockFullSet->writeBuffer(qkBuf.first->buffer(), 0, vkBn->getTensorSize(mTempQKBlock.get()), qkBuf.second);
        mQKBlockFullSet->writeBuffer(tqBuf.first->buffer(), 1, vkBn->getTensorSize(mTempQuery.get()), tqBuf.second);
        mQKBlockFullSet->writeBuffer(denseCacheKeyBuf->buffer(), 2, denseCacheKeySize, denseCacheKeyOffset);
        mQKBlockFullSet->writeBuffer(packedCacheKeyBuf->buffer(), 3, packedCacheKeySize, packedCacheKeyOffset);
        if (hasMask) {
            auto maskBuf = vkBn->getTensorBuffer(mask);
            mQKBlockFullSet->writeBuffer(maskBuf.first->buffer(), 4, vkBn->getTensorSize(mask), maskBuf.second);
        } else {
            mQKBlockFullSet->writeBuffer(queryBuf.first->buffer(), 4, vkBn->getTensorSize(query), queryBuf.second);
        }
        mQKBlockFullSet->writeBuffer(mParam->buffer(), 5, mParam->size());

        // Softmax online set (writes w, updates m/l/alpha)
        mSoftmaxOnlineSet->writeBuffer(wBuf.first->buffer(), 0, vkBn->getTensorSize(mTempWBlock.get()), wBuf.second);
        mSoftmaxOnlineSet->writeBuffer(qkBuf.first->buffer(), 1, vkBn->getTensorSize(mTempQKBlock.get()), qkBuf.second);
        mSoftmaxOnlineSet->writeBuffer(mBuf.first->buffer(), 2, vkBn->getTensorSize(mTempM.get()), mBuf.second);
        mSoftmaxOnlineSet->writeBuffer(lBuf.first->buffer(), 3, vkBn->getTensorSize(mTempL.get()), lBuf.second);
        mSoftmaxOnlineSet->writeBuffer(aBuf.first->buffer(), 4, vkBn->getTensorSize(mTempAlpha.get()), aBuf.second);
        mSoftmaxOnlineSet->writeBuffer(mParam->buffer(), 5, mParam->size());

        // QKV accumulate set
        mQKVAccSet->writeBuffer(oBuf.first->buffer(), 0, vkBn->getTensorSize(mTempOAcc.get()), oBuf.second);
        mQKVAccSet->writeBuffer(wBuf.first->buffer(), 1, vkBn->getTensorSize(mTempWBlock.get()), wBuf.second);
        mQKVAccSet->writeBuffer(cacheValueBuf->buffer(), 2, cacheValueSize, cacheValueOffset);
        mQKVAccSet->writeBuffer(aBuf.first->buffer(), 3, vkBn->getTensorSize(mTempAlpha.get()), aBuf.second);
        mQKVAccSet->writeBuffer(mParam->buffer(), 4, mParam->size());
        mQKVAccSet->writeBuffer(mTurboQuantVParam->buffer(), 5, mTurboQuantVParam->size());
        mQKVAccSet->writeBuffer(packedCacheValueBuf->buffer(), 6, packedCacheValueSize, packedCacheValueOffset);

        // QKV accumulate full-block set (same bindings as tail-safe set)
        mQKVAccFullSet->writeBuffer(oBuf.first->buffer(), 0, vkBn->getTensorSize(mTempOAcc.get()), oBuf.second);
        mQKVAccFullSet->writeBuffer(wBuf.first->buffer(), 1, vkBn->getTensorSize(mTempWBlock.get()), wBuf.second);
        mQKVAccFullSet->writeBuffer(cacheValueBuf->buffer(), 2, cacheValueSize, cacheValueOffset);
        mQKVAccFullSet->writeBuffer(aBuf.first->buffer(), 3, vkBn->getTensorSize(mTempAlpha.get()), aBuf.second);
        mQKVAccFullSet->writeBuffer(mParam->buffer(), 4, mParam->size());
        mQKVAccFullSet->writeBuffer(mTurboQuantVParam->buffer(), 5, mTurboQuantVParam->size());
        mQKVAccFullSet->writeBuffer(packedCacheValueBuf->buffer(), 6, packedCacheValueSize, packedCacheValueOffset);

        // Finalize set
        mFinalizeSet->writeBuffer(outBuf.first->buffer(), 0, vkBn->getTensorSize(output), outBuf.second);
        mFinalizeSet->writeBuffer(oBuf.first->buffer(), 1, vkBn->getTensorSize(mTempOAcc.get()), oBuf.second);
        mFinalizeSet->writeBuffer(lBuf.first->buffer(), 2, vkBn->getTensorSize(mTempL.get()), lBuf.second);
        mFinalizeSet->writeBuffer(mParam->buffer(), 3, mParam->size());
        return NO_ERROR;
    }

    // Attention set (fused). Keep packed fused set for fallback even when decode-q1 subgroup is available.
    auto writeAttentionSet = [&](const std::shared_ptr<VulkanLayout::DescriptorSet>& set) {
        MNN_ASSERT(nullptr != set);
        set->writeBuffer(outBuf.first->buffer(), 0, vkBn->getTensorSize(output), outBuf.second);
        set->writeBuffer(queryBuf.first->buffer(), 1, vkBn->getTensorSize(query), queryBuf.second);
        set->writeBuffer(keyBuf.first->buffer(), 2, vkBn->getTensorSize(key), keyBuf.second);
        set->writeBuffer(valueBuf.first->buffer(), 3, vkBn->getTensorSize(value), valueBuf.second);
        set->writeBuffer(denseCacheKeyBuf->buffer(), 4, denseCacheKeySize, denseCacheKeyOffset);
        set->writeBuffer(packedCacheKeyBuf->buffer(), 5, packedCacheKeySize, packedCacheKeyOffset);
        set->writeBuffer(cacheValueBuf->buffer(), 6, cacheValueSize, cacheValueOffset);
        if (hasMask) {
            auto maskBuf = vkBn->getTensorBuffer(mask);
            set->writeBuffer(maskBuf.first->buffer(), 7, vkBn->getTensorSize(mask), maskBuf.second);
        } else {
            set->writeBuffer(queryBuf.first->buffer(), 7, vkBn->getTensorSize(query), queryBuf.second);
        }
        set->writeBuffer(mParam->buffer(), 8, mParam->size());
        set->writeBuffer(mTurboQuantVParam->buffer(), 9, mTurboQuantVParam->size());
        set->writeBuffer(packedCacheValueBuf->buffer(), 10, packedCacheValueSize, packedCacheValueOffset);
    };
    auto writeDenseAttentionSet = [&](const std::shared_ptr<VulkanLayout::DescriptorSet>& set) {
        MNN_ASSERT(nullptr != set);
        set->writeBuffer(outBuf.first->buffer(), 0, vkBn->getTensorSize(output), outBuf.second);
        set->writeBuffer(queryBuf.first->buffer(), 1, vkBn->getTensorSize(query), queryBuf.second);
        set->writeBuffer(keyBuf.first->buffer(), 2, vkBn->getTensorSize(key), keyBuf.second);
        set->writeBuffer(valueBuf.first->buffer(), 3, vkBn->getTensorSize(value), valueBuf.second);
        set->writeBuffer(denseCacheKeyBuf->buffer(), 4, denseCacheKeySize, denseCacheKeyOffset);
        set->writeBuffer(cacheValueBuf->buffer(), 5, cacheValueSize, cacheValueOffset);
        if (hasMask) {
            auto maskBuf = vkBn->getTensorBuffer(mask);
            set->writeBuffer(maskBuf.first->buffer(), 6, vkBn->getTensorSize(mask), maskBuf.second);
        } else {
            set->writeBuffer(queryBuf.first->buffer(), 6, vkBn->getTensorSize(query), queryBuf.second);
        }
        set->writeBuffer(mParam->buffer(), 7, mParam->size());
    };
    auto writeDecodeAttentionSet = [&](const std::shared_ptr<VulkanLayout::DescriptorSet>& set) {
        MNN_ASSERT(nullptr != set);
        set->writeBuffer(outBuf.first->buffer(), 0, vkBn->getTensorSize(output), outBuf.second);
        set->writeBuffer(queryBuf.first->buffer(), 1, vkBn->getTensorSize(query), queryBuf.second);
        set->writeBuffer(keyBuf.first->buffer(), 2, vkBn->getTensorSize(key), keyBuf.second);
        set->writeBuffer(valueBuf.first->buffer(), 3, vkBn->getTensorSize(value), valueBuf.second);
        set->writeBuffer(denseCacheKeyBuf->buffer(), 4, denseCacheKeySize, denseCacheKeyOffset);
        set->writeBuffer(cacheValueBuf->buffer(), 5, cacheValueSize, cacheValueOffset);
        if (hasMask) {
            auto maskBuf = vkBn->getTensorBuffer(mask);
            set->writeBuffer(maskBuf.first->buffer(), 6, vkBn->getTensorSize(mask), maskBuf.second);
        } else {
            set->writeBuffer(queryBuf.first->buffer(), 6, vkBn->getTensorSize(query), queryBuf.second);
        }
        set->writeBuffer(mParam->buffer(), 7, mParam->size());
        set->writeBuffer(mTurboQuantVParam->buffer(), 8, mTurboQuantVParam->size());
        set->writeBuffer(packedCacheValueBuf->buffer(), 9, packedCacheValueSize, packedCacheValueOffset);
    };
    if (mNeedKvCache) {
        MNN_ASSERT(nullptr != mAttentionSet);
        writeAttentionSet(mAttentionSet);
        if (mQueryLen == 1 && nullptr != mDecodeQ1SubgroupSet) {
            writeDecodeAttentionSet(mDecodeQ1SubgroupSet);
        }
        if (mQueryLen == 1 && nullptr != mDecodeQ1SubgroupHD128Set) {
            writeDecodeAttentionSet(mDecodeQ1SubgroupHD128Set);
        }
    } else {
        MNN_ASSERT(nullptr != mAttentionLegacySet);
        writeDenseAttentionSet(mAttentionLegacySet);
    }

    return NO_ERROR;
}

class VulkanAttentionCreator : public VulkanBackend::Creator {
public:
    VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                   Backend* backend) const override {
        return new VulkanAttention(op, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Attention, new VulkanAttentionCreator);
    return true;
}();

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
