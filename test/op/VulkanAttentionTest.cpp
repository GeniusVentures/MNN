//
//  VulkanAttentionTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/05/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "core/OpCommonUtils.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <stdlib.h>
#include <vector>
#include <stdio.h>

using namespace MNN::Express;

#define TEST_RANDOM_SEED 2024

int NumHead   = 16;
int KvNumHead = 2;
int HeadDim   = 128;
const int pastLength = 101;

struct KVMeta {
    enum {
        NoChange,
        PendingWrite,
        PendingRead
    } file_operation;
    size_t block = 4096;
    size_t previous = 0;
    size_t remove = 0;
    int* reserve = nullptr;
    int n_reserve = 0;
    size_t add = 0;
    std::string file_name = "";
    int file_flag = NoChange;
    int seqlen_in_disk = 0;
    int layer_index = 0;
    int layer_nums = 0;
    bool sparse_v_enable = false;
    float sparse_v_tau = 1.0e-6f;
    bool turboquant_k_enable = false;
    bool turboquant_v_enable = false;
    int turboquant_block_size = 32;
    int turboquant_format = 0;
    std::vector<int> reserveHost;
    void sync() {
        int revertNumber = 0;
        for (int i=0; i<n_reserve; ++i) {
            revertNumber += reserve[2*i+1];
        }
        previous = previous - remove + add + revertNumber;
        n_reserve = 0;
        reserve = nullptr;
        remove = 0;
        add = 0;
    }
};

static KVMeta gMeta;

static std::shared_ptr<Module> _makeVulkanAttentionModule(int attentionMode = 8) {
    auto Q = _Input();
    auto K = _Input();
    auto V = _Input();
    auto mask = _Input();
    std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
    attention->type = MNN::OpType_Attention;
    attention->main.type = MNN::OpParameter_AttentionParam;
    attention->main.value = new MNN::AttentionParamT;
    attention->main.AsAttentionParam()->kv_cache = true;
    auto o = Variable::create(Expr::create(attention.get(), {Q, K, V, mask}));
    auto buffer = Variable::save({o});
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_VULKAN;
    MNN::BackendConfig bnConfig;
    bnConfig.precision = MNN::BackendConfig::Precision_High;
    bnConfig.memory = MNN::BackendConfig::Memory_High;
    config.backendConfig = &bnConfig;
    config.numThread = 1;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, &gMeta);
    rtmgr->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
    std::shared_ptr<Module> m(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
    return m;
}

static VARP _computeAttentionExpr(VARP Q, VARP K, VARP V, VARP mask, int kvNumHead,
                                   int kvPastLength, int headDim) {
    auto qinfo = Q->getInfo();
    auto kinfo = K->getInfo();
    auto seqLength = qinfo->dim[1];
    auto numHead = qinfo->dim[2];
    auto batch = qinfo->dim[0];
    auto group = numHead / kvNumHead;
    if (mask->getInfo()->type.code == halide_type_int) {
        mask = (_Scalar<float>(1.0) - _Cast<float>(mask)) * _Scalar<float>(std::numeric_limits<float>::lowest());
    }
    Q = _Reshape(Q, {batch, seqLength, kvNumHead, group, headDim});
    Q = _Transpose(Q, {0, 2, 3, 1, 4});
    K = _Reshape(K, {batch, seqLength, kvNumHead, 1, headDim});
    K = _Transpose(K, {0, 2, 3, 1, 4});
    auto scale = 1.0f / sqrtf(headDim);
    K = K * _Scalar<float>(scale);
    K.fix(VARP::CONSTANT);
    auto QK = _MatMul(Q, K, false, true);
    QK = QK + mask;
    QK = _Softmax(QK, -1);
    V = _Reshape(V, {batch, seqLength, kvNumHead, 1, headDim});
    V = _Transpose(V, {0, 2, 3, 1, 4});
    V.fix(VARP::CONSTANT);
    auto QKV = _MatMul(QK, V, false, false);
    auto O = _Transpose(QKV, {0, 3, 1, 2, 4});
    O = _Reshape(O, {batch, seqLength, -1});
    O.fix(VARP::CONSTANT);
    return O;
}

static std::vector<std::vector<std::vector<float>>> generateRandTensor(int C, int H, int W, int precision) {
    std::vector<std::vector<std::vector<float>>> a;
    a.resize(C);
    for (int i = 0; i < C; i++) {
        a[i].resize(H);
        for (int j = 0; j < H; j++) {
            a[i][j].resize(W);
            for (int k = 0; k < W; k++) {
                if (precision == 2) {
                    a[i][j][k] = ((i + j + k) % 10) * 0.002;
                } else {
                    a[i][j][k] = ((i + j + k) % 10) * 0.16 - 5.6;
                }
            }
        }
    }
    return a;
}

static VARP vector_to_var(std::vector<std::vector<std::vector<float>>>& a) {
    int C = a.size();
    int H = a[0].size();
    int W = a[0][0].size();
    VARP var = _Input({1, C, H, W}, NCHW, halide_type_of<float>());
    float* ptr = var->writeMap<float>();
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < H; j++) {
            for (int k = 0; k < W; k++) {
                ptr[i * H * W + j * W + k] = a[i][j][k];
            }
        }
    }
    var->unMap();
    return var;
}

static VARP vector_to_var(std::vector<std::vector<int>>& a) {
    int H = a.size();
    int W = a[0].size();
    VARP var = _Input({1, 1, H, W}, NCHW, halide_type_of<int>());
    int* ptr = var->writeMap<int>();
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            ptr[i * W + j] = a[i][j];
        }
    }
    var->unMap();
    return var;
}

static std::vector<std::vector<int>> generateCausalMask(int seqLen, int kvSeqLen) {
    std::vector<std::vector<int>> mask(seqLen);
    for (int i = 0; i < seqLen; i++) {
        mask[i].resize(kvSeqLen);
        for (int j = 0; j < kvSeqLen; j++) {
            if (j <= i + (kvSeqLen - seqLen)) {
                mask[i][j] = 1;
            } else {
                mask[i][j] = 0;
            }
        }
    }
    return mask;
}

class VulkanAttentionCorrectnessTest : public MNNTestCase {
public:
    VulkanAttentionCorrectnessTest() = default;
    virtual ~VulkanAttentionCorrectnessTest() = default;

    virtual bool run(int precision) {
        srand(TEST_RANDOM_SEED);

        // Check Vulkan backend availability
        auto vulkanCreator = MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN);
        if (nullptr == vulkanCreator) {
            MNN_PRINT("Vulkan backend not available — skipping VulkanAttentionCorrectnessTest\n");
            return true;
        }

        // Test case 1: GQA (Grouped Query Attention) — group=8
        {
            int numHead = 16;
            int kvNumHead = 2; // group = 8
            int headDim = 128;
            int queryLen = 16;
            int pastLen = 101;

            auto query  = generateRandTensor(queryLen, numHead, headDim, precision);
            auto key    = generateRandTensor(pastLen + queryLen, kvNumHead, headDim, precision);
            auto value  = generateRandTensor(pastLen + queryLen, kvNumHead, headDim, precision);
            auto maskInt = generateCausalMask(queryLen, pastLen + queryLen);

            auto Q = vector_to_var(query);
            auto K = vector_to_var(key);
            auto V = vector_to_var(value);
            auto M = vector_to_var(maskInt);

            // CPU reference (self-attention only, no past KVCache)
            auto refOutput = _computeAttentionExpr(Q, K, V, M, kvNumHead, pastLen, headDim);
            auto refPtr = refOutput->readMap<float>();
            auto refInfo = refOutput->getInfo();
            int refSize = refInfo->size;

            // Vulkan output (KVCache-based)
            gMeta = KVMeta();
            gMeta.sparse_v_enable = false;
            gMeta.turboquant_k_enable = false;
            gMeta.turboquant_v_enable = false;
            auto attn = _makeVulkanAttentionModule();
            // Prefill: first queryLen tokens go into KVCache
            auto qPrefill = _Input({1, queryLen, numHead, headDim}, NCHW, halide_type_of<float>());
            auto kPrefill = _Input({1, queryLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());
            auto vPrefill = _Input({1, queryLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());
            ::memcpy(qPrefill->writeMap<float>(), Q->readMap<float>(), queryLen * numHead * headDim * sizeof(float));
            ::memcpy(kPrefill->writeMap<float>(), K->readMap<float>(), queryLen * kvNumHead * headDim * sizeof(float));
            ::memcpy(vPrefill->writeMap<float>(), V->readMap<float>(), queryLen * kvNumHead * headDim * sizeof(float));

            // Mask: causal mask for prefill
            auto maskPrefillInt = generateCausalMask(queryLen, queryLen);
            auto maskPrefill = vector_to_var(maskPrefillInt);

            gMeta.previous = 0;
            gMeta.add = queryLen;
            auto output = attn->onForward({qPrefill, kPrefill, vPrefill, maskPrefill})[0];
            gMeta.sync();

            auto outPtr = output->readMap<float>();
            if (!checkVectorByRelativeError<float>(outPtr, refPtr, refSize, 0.01f)) {
                MNN_ERROR("VulkanAttentionCorrectnessTest: GQA case failed!\n");
                return false;
            }
            MNN_PRINT("VulkanAttentionCorrectnessTest: GQA (group=8) case PASSED\n");
        }

        // Test case 2: MHA (Multi-Head Attention) — group=1
        {
            int numHead = 16;
            int kvNumHead = 16; // group = 1
            int headDim = 128;
            int queryLen = 16;
            int pastLen = 101;

            auto query  = generateRandTensor(queryLen, numHead, headDim, precision);
            auto key    = generateRandTensor(pastLen + queryLen, kvNumHead, headDim, precision);
            auto value  = generateRandTensor(pastLen + queryLen, kvNumHead, headDim, precision);
            auto maskInt = generateCausalMask(queryLen, pastLen + queryLen);

            auto Q = vector_to_var(query);
            auto K = vector_to_var(key);
            auto V = vector_to_var(value);
            auto M = vector_to_var(maskInt);

            auto refOutput = _computeAttentionExpr(Q, K, V, M, kvNumHead, pastLen, headDim);
            auto refPtr = refOutput->readMap<float>();
            auto refInfo = refOutput->getInfo();
            int refSize = refInfo->size;

            gMeta = KVMeta();
            gMeta.sparse_v_enable = false;
            auto attn = _makeVulkanAttentionModule();
            auto qPrefill = _Input({1, queryLen, numHead, headDim}, NCHW, halide_type_of<float>());
            auto kPrefill = _Input({1, queryLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());
            auto vPrefill = _Input({1, queryLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());
            ::memcpy(qPrefill->writeMap<float>(), Q->readMap<float>(), queryLen * numHead * headDim * sizeof(float));
            ::memcpy(kPrefill->writeMap<float>(), K->readMap<float>(), queryLen * kvNumHead * headDim * sizeof(float));
            ::memcpy(vPrefill->writeMap<float>(), V->readMap<float>(), queryLen * kvNumHead * headDim * sizeof(float));

            auto maskPrefillInt = generateCausalMask(queryLen, queryLen);
            auto maskPrefill = vector_to_var(maskPrefillInt);

            gMeta.previous = 0;
            gMeta.add = queryLen;
            auto output = attn->onForward({qPrefill, kPrefill, vPrefill, maskPrefill})[0];
            gMeta.sync();

            auto outPtr = output->readMap<float>();
            if (!checkVectorByRelativeError<float>(outPtr, refPtr, refSize, 0.01f)) {
                MNN_ERROR("VulkanAttentionCorrectnessTest: MHA case failed!\n");
                return false;
            }
            MNN_PRINT("VulkanAttentionCorrectnessTest: MHA (group=1) case PASSED\n");
        }

        // Test case 3: MQA (Multi-Query Attention) — kvHeadNum=1
        {
            int numHead = 16;
            int kvNumHead = 1;
            int headDim = 128;
            int queryLen = 16;
            int pastLen = 101;

            auto query  = generateRandTensor(queryLen, numHead, headDim, precision);
            auto key    = generateRandTensor(pastLen + queryLen, kvNumHead, headDim, precision);
            auto value  = generateRandTensor(pastLen + queryLen, kvNumHead, headDim, precision);
            auto maskInt = generateCausalMask(queryLen, pastLen + queryLen);

            auto Q = vector_to_var(query);
            auto K = vector_to_var(key);
            auto V = vector_to_var(value);
            auto M = vector_to_var(maskInt);

            auto refOutput = _computeAttentionExpr(Q, K, V, M, kvNumHead, pastLen, headDim);
            auto refPtr = refOutput->readMap<float>();
            auto refInfo = refOutput->getInfo();
            int refSize = refInfo->size;

            gMeta = KVMeta();
            gMeta.sparse_v_enable = false;
            auto attn = _makeVulkanAttentionModule();
            auto qPrefill = _Input({1, queryLen, numHead, headDim}, NCHW, halide_type_of<float>());
            auto kPrefill = _Input({1, queryLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());
            auto vPrefill = _Input({1, queryLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());
            ::memcpy(qPrefill->writeMap<float>(), Q->readMap<float>(), queryLen * numHead * headDim * sizeof(float));
            ::memcpy(kPrefill->writeMap<float>(), K->readMap<float>(), queryLen * kvNumHead * headDim * sizeof(float));
            ::memcpy(vPrefill->writeMap<float>(), V->readMap<float>(), queryLen * kvNumHead * headDim * sizeof(float));

            auto maskPrefillInt = generateCausalMask(queryLen, queryLen);
            auto maskPrefill = vector_to_var(maskPrefillInt);

            gMeta.previous = 0;
            gMeta.add = queryLen;
            auto output = attn->onForward({qPrefill, kPrefill, vPrefill, maskPrefill})[0];
            gMeta.sync();

            auto outPtr = output->readMap<float>();
            if (!checkVectorByRelativeError<float>(outPtr, refPtr, refSize, 0.01f)) {
                MNN_ERROR("VulkanAttentionCorrectnessTest: MQA case failed!\n");
                return false;
            }
            MNN_PRINT("VulkanAttentionCorrectnessTest: MQA (kvHeadNum=1) case PASSED\n");
        }

        // Test case 4: KVCache Multi-Turn
        {
            int numHead = 16;
            int kvNumHead = 2;
            int headDim = 128;
            int turnLen = 8;
            int numTurns = 3;

            gMeta = KVMeta();
            gMeta.sparse_v_enable = false;
            auto attn = _makeVulkanAttentionModule();

            // Generate input data for 3 turns totalling 24 tokens
            auto allQuery  = generateRandTensor(numTurns * turnLen, numHead, headDim, precision);
            auto allKey    = generateRandTensor(numTurns * turnLen, kvNumHead, headDim, precision);
            auto allValue  = generateRandTensor(numTurns * turnLen, kvNumHead, headDim, precision);

            int cumulativeLen = 0;
            for (int turn = 0; turn < numTurns; turn++) {
                cumulativeLen += turnLen;
                int totalLen = pastLength + cumulativeLen;

                // CPU reference for this turn (self-attention over all tokens so far)
                auto refMaskInt = generateCausalMask(turnLen, totalLen);
                auto refQ = _Input({1, turnLen, numHead, headDim}, NCHW, halide_type_of<float>());
                auto refK = _Input({1, totalLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());
                auto refV = _Input({1, totalLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());
                auto refM = _Input({1, 1, turnLen, totalLen}, NCHW, halide_type_of<int>());

                // Fill Q with this turn's slice
                for (int i = 0; i < turnLen; i++) {
                    for (int j = 0; j < numHead; j++) {
                        ::memcpy(refQ->writeMap<float>() + i * numHead * headDim + j * headDim,
                                allQuery[turn * turnLen + i][j].data(), headDim * sizeof(float));
                    }
                }
                // Fill K/V with all accumulated tokens
                for (int i = 0; i < cumulativeLen; i++) {
                    int srcIdx = i;
                    for (int j = 0; j < kvNumHead; j++) {
                        ::memcpy(refK->writeMap<float>() + i * kvNumHead * headDim + j * headDim,
                                allKey[srcIdx][j].data(), headDim * sizeof(float));
                        ::memcpy(refV->writeMap<float>() + i * kvNumHead * headDim + j * headDim,
                                allValue[srcIdx][j].data(), headDim * sizeof(float));
                    }
                }
                // Cast int mask to float mask for Express ref
                auto refMFloat = vector_to_var(refMaskInt);

                auto refOutput = _computeAttentionExpr(refQ, refK, refV, refMFloat, kvNumHead, pastLength, headDim);
                auto refPtr = refOutput->readMap<float>();
                auto refInfo = refOutput->getInfo();
                int refSize = refInfo->size;

                // Vulkan: KVCache-based (feed one turn at a time)
                auto qTurn = _Input({1, turnLen, numHead, headDim}, NCHW, halide_type_of<float>());
                auto kTurn = _Input({1, turnLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());
                auto vTurn = _Input({1, turnLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());
                for (int i = 0; i < turnLen; i++) {
                    for (int j = 0; j < numHead; j++) {
                        ::memcpy(qTurn->writeMap<float>() + i * numHead * headDim + j * headDim,
                                allQuery[turn * turnLen + i][j].data(), headDim * sizeof(float));
                    }
                }
                for (int i = 0; i < turnLen; i++) {
                    for (int j = 0; j < kvNumHead; j++) {
                        ::memcpy(kTurn->writeMap<float>() + i * kvNumHead * headDim + j * headDim,
                                allKey[turn * turnLen + i][j].data(), headDim * sizeof(float));
                        ::memcpy(vTurn->writeMap<float>() + i * kvNumHead * headDim + j * headDim,
                                allValue[turn * turnLen + i][j].data(), headDim * sizeof(float));
                    }
                }
                auto maskTurnInt = generateCausalMask(turnLen, turnLen);
                auto maskTurn = vector_to_var(maskTurnInt);

                gMeta.add = turnLen;
                auto output = attn->onForward({qTurn, kTurn, vTurn, maskTurn})[0];
                gMeta.sync();

                auto outPtr = output->readMap<float>();
                if (!checkVectorByRelativeError<float>(outPtr, refPtr, refSize, 0.01f)) {
                    MNN_ERROR("VulkanAttentionCorrectnessTest: KVCache multi-turn case failed at turn %d!\n", turn + 1);
                    return false;
                }
            }
            MNN_PRINT("VulkanAttentionCorrectnessTest: KVCache multi-turn case PASSED\n");
        }

        // Test case 5: Variable Sequence Lengths
        {
            int numHead = 16;
            int kvNumHead = 2;
            int headDim = 128;
            std::vector<int> queryLengths = {1, 8, 32, 128};

            for (auto queryLen : queryLengths) {
                int pastLen = 101;
                auto query  = generateRandTensor(queryLen, numHead, headDim, precision);
                auto key    = generateRandTensor(pastLen + queryLen, kvNumHead, headDim, precision);
                auto value  = generateRandTensor(pastLen + queryLen, kvNumHead, headDim, precision);
                auto maskInt = generateCausalMask(queryLen, pastLen + queryLen);

                auto Q = vector_to_var(query);
                auto K = vector_to_var(key);
                auto V = vector_to_var(value);
                auto M = vector_to_var(maskInt);

                auto refOutput = _computeAttentionExpr(Q, K, V, M, kvNumHead, pastLen, headDim);
                auto refPtr = refOutput->readMap<float>();
                auto refInfo = refOutput->getInfo();
                int refSize = refInfo->size;

                gMeta = KVMeta();
                gMeta.sparse_v_enable = false;
                auto attn = _makeVulkanAttentionModule();
                auto qPrefill = _Input({1, queryLen, numHead, headDim}, NCHW, halide_type_of<float>());
                auto kPrefill = _Input({1, queryLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());
                auto vPrefill = _Input({1, queryLen, kvNumHead, headDim}, NCHW, halide_type_of<float>());
                ::memcpy(qPrefill->writeMap<float>(), Q->readMap<float>(), queryLen * numHead * headDim * sizeof(float));
                ::memcpy(kPrefill->writeMap<float>(), K->readMap<float>(), queryLen * kvNumHead * headDim * sizeof(float));
                ::memcpy(vPrefill->writeMap<float>(), V->readMap<float>(), queryLen * kvNumHead * headDim * sizeof(float));

                auto maskPrefillInt = generateCausalMask(queryLen, queryLen);
                auto maskPrefill = vector_to_var(maskPrefillInt);

                gMeta.previous = 0;
                gMeta.add = queryLen;
                auto output = attn->onForward({qPrefill, kPrefill, vPrefill, maskPrefill})[0];
                gMeta.sync();

                auto outPtr = output->readMap<float>();
                if (!checkVectorByRelativeError<float>(outPtr, refPtr, refSize, 0.01f)) {
                    MNN_ERROR("VulkanAttentionCorrectnessTest: variable seqLen=%d case failed!\n", queryLen);
                    return false;
                }
            }
            MNN_PRINT("VulkanAttentionCorrectnessTest: Variable sequence length case PASSED\n");
        }

        return true;
    }
};

MNNTestSuiteRegister(VulkanAttentionCorrectnessTest, "op/vulkan/attention_correctness");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
