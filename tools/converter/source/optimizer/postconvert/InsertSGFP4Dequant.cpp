//
//  InsertSGFP4Dequant.cpp
//  MNNConverter
//
//  Created by MNN on 2026/09/01.
//  Copyright © 2018, Alibaba Group Holding Limited.
//
// Phase 11, Plan 11-01 (SGV2-28): the SGFP4 v2 graph-rewrite PostConverter
// pass. Rewrites target conv-family FP32 weights into OpType_SGFP4Dequant
// producer nodes (graph surgery -- a new node feeding the original,
// type-unchanged conv's inputs[1]; never in-place op mutation).
//
// Decisions implemented (see 11-CONTEXT.md):
//   D-01  registered pass, invoked from PostConverter.cpp's final batch
//         BEFORE ReIndexTensor (research KEY Q2 order lock: ReIndexTensor
//         then compacts/dedups this pass's tensorName additions for free)
//   D-03  full net coverage: net->oplists AND every subgraph->nodes
//   D-06  exactly Convolution / ConvolutionDepthwise / Deconvolution /
//         DeconvolutionDepthwise; dims arithmetic mirrors
//         WeightQuantAndCoding.cpp (oc = outputCount, kernelSize =
//         weightSize / oc, dimO = oc, dimI = kernelSize)
//   D-07  light-tier floor: elements < 4096 OR dimI == 1 -> FP32 untouched
//   D-08  encode config via the greppable file-scope alias below
//   D-14  dead code when modelConfig::useSGFP4 is false (the default)
//   Phase 8 D-11 buffer contract: inserted ops carry param->buffer
//         populated, external == {} and empty externalPath; the pass never
//         writes external/externalPath itself (externalization rides
//         RemoveAndStoreParam/storeSGFP4Container untouched)
//   Idempotency (research Pitfall 3): rewrite condition is strictly
//         conv-family AND inputIndexes.size() == 1 AND quanParameter ==
//         nullptr, so the double-RunOptimize final-batch execution and
//         per-subgraph round-trips never double-rewrite
//   Weight-spill path (research KEY Q3): convs arriving with
//         param->external.size() == 3 are reloaded from
//         .__convert_external_data.bin via FileLoader after flushing
//         config->externalFile (null-checked); bias restored into
//         param->bias; param->external cleared
//   Transactional failure safety (T-11-01/T-11-03): a conv is only mutated
//         (weight cleared, input index pushed) AFTER its container encoded
//         successfully; on encode/reload failure the conv is skipped
//         untouched, MNN_ERROR names the op, and the pass returns false
//

#include <cstring>
#include <fstream>
#include <vector>

#include <MNN/MNNDefine.h>
#include "../PostTreatUtils.hpp"
#include "config.hpp"
#include "../Global.hpp"
#include "MNN/SGFP4DequantUtils.hpp"
#include "sgfp4_encode.hpp"

using namespace MNN;

namespace {

// D-08: the single edit point when the Phase 10 validated delta is adopted
// upstream. Python-identical defaults by design -- cross-repo parity
// outranks one-sided promotion (Phase 10 D-09); the validated (unused by
// default) table lives in tools/fp4/real_weight_validation_report.json.
// File-scope in this .cpp ONLY -- kDefaultEncodeConfig is an extern with
// linkage in sgfp4_encode.hpp; redefining it at namespace scope in another
// TU is an MSVC C2086 trap. No CLI threshold override (D-08).
static const sgfp4_encode::EncodeConfig& kSGFP4ConverterEncodeConfig = sgfp4_encode::kDefaultEncodeConfig;

// The temp spill file the converter itself wrote spilled conv weights into
// (optimizeNet opens it as .__convert_external_data.bin; writeFb.cpp
// removes it only after postTreat, i.e. after this pass has run).
const char* kConvertExternalDataFile = ".__convert_external_data.bin";

bool isSgfp4TargetOpType(MNN::OpType t) {
    return t == OpType_Convolution || t == OpType_ConvolutionDepthwise || t == OpType_Deconvolution ||
           t == OpType_DeconvolutionDepthwise;
}

// Reload spilled FP32 weights (and bias) for one conv from the converter's
// temp external-data file. external layout for a non-quan conv is exactly
// {weightOffset, weightBytes, biasBytes} (storeWeight pairs in
// RemoveParams.cpp). Returns false on any open/seek/read failure or short
// read -- the caller skips the conv untouched (T-11-01).
//
// NOTE (deviation from research KEY Q3): reads via std::ifstream, NOT
// FileLoader. This pass runs INSIDE optimizeNet, while the converter's
// own externalFile ofstream still holds the temp bin open. MSVC's
// fopen_s/_wfopen_s (FileLoader's open path) requests EXCLUSIVE sharing,
// so opening the bin there is a guaranteed sharing violation at this
// point in the pipeline (writeFb.cpp's reload works only because it runs
// after the stream closed). std::ifstream opens deny-none and reads the
// flushed bytes back correctly (probe: TestSGFP4Converter PHASE C T7).
bool reloadSpilledConvWeights(Convolution2DT* param, const std::string& opName) {
    if (param->external.size() != 3) {
        return false;
    }
    std::ifstream bin(kConvertExternalDataFile, std::ios::binary);
    if (!bin.is_open()) {
        MNN_ERROR("InsertSGFP4Dequant: op '%s': cannot open %s for spilled-weight reload\n", opName.c_str(),
                  kConvertExternalDataFile);
        return false;
    }
    bin.seekg((std::streamoff)param->external[0], std::ios::beg);
    if (!bin.good()) {
        MNN_ERROR("InsertSGFP4Dequant: op '%s': seek to offset %lld in %s failed\n", opName.c_str(),
                  (long long)param->external[0], kConvertExternalDataFile);
        return false;
    }
    std::vector<float> weight;
    weight.resize((size_t)(param->external[1] / (int64_t)sizeof(float)));
    if (weight.empty()) {
        MNN_ERROR("InsertSGFP4Dequant: op '%s': spilled weight size %lld is not a positive float multiple\n",
                  opName.c_str(), (long long)param->external[1]);
        return false;
    }
    bin.read(reinterpret_cast<char*>(weight.data()), (std::streamsize)param->external[1]);
    if (bin.gcount() != (std::streamsize)param->external[1]) {
        MNN_ERROR("InsertSGFP4Dequant: op '%s': short read of %lld weight bytes from %s\n", opName.c_str(),
                  (long long)param->external[1], kConvertExternalDataFile);
        return false;
    }
    if (param->external[2] > 0) {
        param->bias.resize((size_t)(param->external[2] / (int64_t)sizeof(float)));
        bin.read(reinterpret_cast<char*>(param->bias.data()), (std::streamsize)param->external[2]);
        if (bin.gcount() != (std::streamsize)param->external[2]) {
            MNN_ERROR("InsertSGFP4Dequant: op '%s': short read of %lld bias bytes from %s\n", opName.c_str(),
                      (long long)param->external[2], kConvertExternalDataFile);
            return false;
        }
    }
    param->weight     = std::move(weight);
    param->external.clear();
    return true;
}

// Build the buffer-staged SGFP4 producer OpT (Phase 8 D-11 contract; shape
// from TestSGFP4Converter.cpp's makeSgfp4Op reference).
// D-13 deviation (Phase 11): dims carries the CONV-WEIGHT geometry
// {O, I, kH, kW} (MatMul-derived convs included -- see caller); the flat
// decode plane stays dimO x dimI by construction, and the decoder derives
// it as dims[0] x product(dims[1..]).
std::unique_ptr<OpT> makeSgfp4DequantOp(const std::vector<uint8_t>& container,
                                        const std::vector<int>& decodeDims,
                                        const std::string& name, int outputIndex) {
    std::unique_ptr<OpT> op(new OpT);
    op->type            = OpType_SGFP4Dequant;
    op->name            = name;
    op->main.type       = OpParameter_SGFP4DequantParam;
    auto* param         = new SGFP4DequantParamT;
    param->magic        = kSGFP4Magic;
    param->dims         = decodeDims;
    // SGFP4DequantParamT::buffer is std::vector<int8_t> (flatc [byte]);
    // copy the raw container bytes across.
    param->buffer.resize(container.size());
    if (!container.empty()) {
        ::memcpy(param->buffer.data(), container.data(), container.size());
    }
    // external stays {} and externalPath stays empty -- externalization
    // rides RemoveAndStoreParam/storeSGFP4Container untouched.
    op->main.value      = param;
    op->outputIndexes   = {outputIndex};
    return op;
}

// Process one node list (either net->oplists or a subgraph->nodes). The
// name-appender callback abstracts the per-scope tensor namespace: the root
// net grows net->tensorName, a subgraph grows subgraph->tensors; the
// returned index is the pre-push size (NEVER renumber existing indices --
// research KEY Q8). Returns false if any conv failed transactionally.
template <typename NameAppender>
bool processOplist(std::vector<std::unique_ptr<OpT>>& ops, NameAppender appendTensorName, bool* failed) {
    for (auto iter = ops.begin(); iter != ops.end(); ++iter) {
        OpT* op = iter->get();
        if (nullptr == op || !isSgfp4TargetOpType(op->type)) {
            continue;
        }
        // Idempotency + D-02 fingerprint: an original converter conv has
        // only its input-activation index; a second input is the visible
        // fingerprint of an SGFP4-rewritten conv. int8 weights
        // (quanParameter) are not FP32 encode targets either.
        if (op->inputIndexes.size() != 1) {
            continue;
        }
        auto param = op->main.AsConvolution2D();
        if (nullptr == param || nullptr == param->quanParameter.get()) {
            // quanParameter == nullptr is required; a nullptr param for a
            // conv-family op is malformed -- skip defensively.
            if (nullptr == param) {
                continue;
            }
        } else {
            continue;
        }

        // Weight acquisition (research KEY Q3).
        if (param->weight.empty()) {
            if (param->external.size() == 3) {
                // The temp-bin ofstream may still be open and buffered --
                // flush before reading the file back (Pitfall 5). The
                // pointer is null when the open failed
                // (PostConverter.cpp:645-647).
                auto config = Global<modelConfig>::Get();
                if (nullptr != config && nullptr != config->externalFile) {
                    config->externalFile->flush();
                }
                const std::string opName = op->name.empty() ? "<unnamed>" : op->name;
                if (!reloadSpilledConvWeights(param, opName)) {
                    *failed = true;
                    continue;
                }
            } else {
                // No weights at all -- nothing to encode.
                continue;
            }
        }
        const std::string opName = op->name.empty() ? "<unnamed>" : op->name;

        // Dims (Pitfall 8 -- mirror WeightQuantAndCoding exactly).
        auto* common = param->common.get();
        if (nullptr == common) {
            continue;
        }
        const int oc          = common->outputCount;
        const int kernelSize  = (int)(param->weight.size() / oc);
        const int dimO        = oc;
        const int dimI        = kernelSize;
        const size_t elements = (size_t)dimO * (size_t)dimI;
        if (param->weight.size() != elements) {
            // T-11-02: pre-encode assertion in size_t.
            MNN_ERROR("InsertSGFP4Dequant: op '%s': weight size %zu != dimO*dimI %zu -- skipping\n", opName.c_str(),
                      param->weight.size(), elements);
            *failed = true;
            continue;
        }
        // D-07 light-tier floor: tiny tensors are pad-overhead-dominated
        // (Phase 10 validated corpus tiering rule).
        if (elements < 4096 || dimI == 1) {
            continue;
        }

        // Encode. Empty container is the encoder's invalid-input contract
        // (NaN/Inf, bad dims) -- never encode garbage (V5 / T-11-03).
        auto container = sgfp4_encode::encode(param->weight.data(), dimO, dimI, kSGFP4ConverterEncodeConfig);
        if (container.empty()) {
            MNN_ERROR("InsertSGFP4Dequant: op '%s': sgfp4_encode returned an empty container -- skipping\n",
                      opName.c_str());
            *failed = true;
            continue;
        }

        // Conv-weight geometry for the emitted tensor (D-13 deviation):
        // recover kH/kW from the common param; the input-channel count is
        // kernelSize / (kH*kW). MatMul-derived convs carry 1x1 kernels, so
        // they generalize cleanly. If common disagrees with the weight
        // layout (kernelSize not divisible), fall back to flat {dimO, dimI}
        // -- the 2-D legacy form decodes identically.
        std::vector<int> tensorDims = {dimO, dimI};
        const int kx = common->kernelX;
        const int ky = common->kernelY;
        if (kx > 0 && ky > 0 && (kernelSize % (kx * ky)) == 0) {
            tensorDims = {dimO, kernelSize / (kx * ky), ky, kx};
        }

        // Node construction + splice. Producer precedes consumer.
        const int newIndex = (int)appendTensorName(std::string());
        std::string nodeName = op->name.empty() ? ("sgfp4_weight_" + std::to_string(newIndex))
                                                : (op->name + "_sgfp4");
        appendTensorName(nodeName);
        auto dequantOp = makeSgfp4DequantOp(container, tensorDims, nodeName, newIndex);
        // (sub)graph tensor namespace grew before mutation; failure paths
        // above have already exited. Mutate only now (transactional rule).
        iter = ops.insert(iter, std::move(dequantOp));
        // iter now points at the inserted producer; advance back onto the
        // conv, push its new weight input, clear FP32 weight + spill
        // descriptor (swap-empty idiom, RemoveParams.cpp precedent).
        ++iter;
        OpT* conv = iter->get();
        conv->inputIndexes.push_back(newIndex);
        {
            std::vector<float> emptyWeight;
            param->weight.swap(emptyWeight);
        }
        param->external.clear();
    }
    return true;
}

} // namespace

class InsertSGFP4Dequant : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        // D-14: dead code when the flag is absent.
        auto config = Global<modelConfig>::Get();
        if (nullptr == config || !config->useSGFP4) {
            return true;
        }
        bool failed = false;

        // D-03: root oplist.
        {
            auto& tensorName = net->tensorName;
            auto appender    = [&tensorName](const std::string& name) -> size_t {
                if (!name.empty()) {
                    tensorName.emplace_back(name);
                }
                return tensorName.size();
            };
            processOplist(net->oplists, appender, &failed);
        }
        // D-03: every subgraph (exact saveExternalData walk shape).
        for (auto& subgraph : net->subgraphs) {
            auto& tensors = subgraph->tensors;
            auto appender = [&tensors](const std::string& name) -> size_t {
                if (!name.empty()) {
                    tensors.emplace_back(name);
                }
                return tensors.size();
            };
            processOplist(subgraph->nodes, appender, &failed);
        }
        return failed ? false : true;
    }
};
static PostConverterRegister<InsertSGFP4Dequant> __l("InsertSGFP4Dequant");
