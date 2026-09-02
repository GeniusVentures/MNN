//
//  CPUSGFP4Dequant.cpp
//  MNN
//
//  Created by MNN on 2026/08/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>

#include "CPUSGFP4Dequant.hpp"
#include "MNN/SGFP4DequantUtils.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/FileLoader.hpp"
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"

namespace MNN {

namespace {

// FileLoader::size() only reflects bytes already pulled into its internal
// cache blocks by the parameterless, whole-file FileLoader::read(); it is
// NOT a filesystem stat and stays 0 for the offset+size-bounded read used
// here (FileLoader::offset()/read(buffer,size)). A real on-disk size probe
// is required to bound the DoS check below (T-01-04) against the sidecar's
// actual size before attempting an allocation of the declared `size`.
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

// Dims interpretation (Phase 11, D-13 deviation): dims[0] is always the
// encode plane's dimO; the remaining dims multiply to dimI. The 2-D
// {dimO, dimI} form (injection-tool artifacts, MatMul consumers) reads
// identically to before; the 4-D conv-weight form {O, I, kH, kW} written
// by the converter's InsertSGFP4Dequant pass yields I*kH*kW -- the same
// flat decode plane, now exposed with conv-weight geometry so
// ShapeConvolution / ConvolutionTiledExecutorMultiInput see a valid
// weight tensor.
bool readDecodeDims(const SGFP4DequantParam* param, int& dimO, int& dimI) {
    if (nullptr == param || nullptr == param->dims() || param->dims()->size() < 2) {
        return false;
    }
    dimO = param->dims()->Get(0);
    dimI = 1;
    for (flatbuffers::uoffset_t i = 1; i < param->dims()->size(); ++i) {
        dimI *= param->dims()->Get(i);
    }
    return dimO > 0 && dimI > 0;
}

} // namespace

ErrorCode CPUSGFP4Dequant::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto param = mOp->main_as_SGFP4DequantParam();
    if (nullptr == param) {
        return INVALID_VALUE;
    }

    // Padded-geometry derivation (Plan 09-02): the output tensor keeps the
    // TRUE {dimO, dimI} shape; a container encoded from a zero-padded plane
    // (D-06) decodes through the crop overload. When both dims are
    // 64-multiples this is a no-op and the original path runs unchanged.
    if (outputs.empty() || nullptr == outputs[0]) {
        return INVALID_VALUE;
    }
    int dimO = 0, dimI = 0;
    if (!readDecodeDims(param, dimO, dimI)) {
        return INVALID_VALUE;
    }
    mPaddedDimO = ((dimO + 63) / 64) * 64;
    mPaddedDimI = ((dimI + 63) / 64) * 64;
    // Buffer-first dispatch (D-01/D-02, Plan 08-03): a non-empty inline
    // `buffer` is the live decode source -- no FileLoader, no externalPath.
    // Copy into mContainer for safety (the FlatBuffers buffer may point
    // into the model buffer, which must not be treated as owned storage).
    const auto* buf = param->buffer();
    if (nullptr != buf && buf->size() > 0) {
        mContainer.assign(buf->data(), buf->data() + buf->size());
        // Entry gate: magic/version framing check (D-02).
        if (!sgfp4_is_v2_container(mContainer.data(), mContainer.size())) {
            mContainer.clear();
            return INVALID_VALUE;
        }
        // Dims-consistency: eager oracle decode into scratch (Q2 decision --
        // the eager oracle doubles as the buffer-mode replacement for the
        // sidecar path's T-01-04 file-size DoS bound; the buffer is already
        // fully materialized in memory).
        //
        // Phase 12 (Plan 12-02) codec fix: ALWAYS decode through the spatial
        // padded-plane overload. The flat dequant_sgfp4_container_cpu is a
        // leaf-major LINEAR STREAM that equals the row-major plane only for
        // one-superblock-wide grids (tiles_x == 1); the normative convention
        // (gnus-poc decode_v2, matched by the encoder's superblock ordering)
        // places record (by, bx) at rows by*64..+64 / cols bx*64..+64. Real
        // conv weights with dimI > 64 decoded to garbage through the flat
        // path (reproducer: encode->flat-decode on 128+-col planes diverges
        // ~2x; spatial round-trips at FP4 noise level). The crop overload's
        // aligned case is the spatial decode with a no-op stride crop.
        std::vector<float> scratch(outputs[0]->elementSize(), 0.0f);
        if (!dequant_sgfp4_container_cpu_crop(mContainer.data(), mContainer.size(), scratch.data(), dimO, dimI,
                                              mPaddedDimO, mPaddedDimI)) {
            mContainer.clear();
            return INVALID_VALUE;
        }
        return NO_ERROR;
    }
    // Mirrors ConvolutionCommon.cpp's USE_EXTERNAL_DATA(param) + externalPath
    // gate: the empty-buffer fallback is the original external-sidecar
    // container form (D-04 -- unchanged path).
    if (!USE_EXTERNAL_DATA(param) || nullptr == mOp->externalPath()) {
        return NOT_SUPPORT;
    }

    auto external = param->external()->data();
    int64_t offset = external[0];
    int64_t size   = external[1];
    if (offset < 0 || size <= 0) {
        return INVALID_VALUE;
    }

    // T-01-04: bound the declared size against the sidecar's real on-disk
    // size BEFORE allocating mContainer, so an attacker-controlled
    // external()[1] can't force an oversized allocation against a small
    // (or missing) file.
    size_t fileSize = 0;
    if (!queryFileSize(mOp->externalPath()->str(), fileSize)) {
        return NOT_SUPPORT;
    }
    size_t offsetSize = static_cast<size_t>(offset);
    size_t readSize   = static_cast<size_t>(size);
    if (offsetSize > fileSize || readSize > fileSize - offsetSize) {
        return INVALID_VALUE;
    }

    // Construct with init=true so valid() is usable immediately, mirroring
    // Interpreter.cpp's FileLoader(file, true) + valid() pattern.
    FileLoader loader(mOp->externalPath()->c_str(), true);
    if (!loader.valid()) {
        return NOT_SUPPORT;
    }

    mContainer.resize(readSize);
    loader.offset(offset);
    if (!loader.read(reinterpret_cast<char*>(mContainer.data()), size)) {
        mContainer.clear();
        return INVALID_VALUE;
    }
    return NO_ERROR;
}

ErrorCode CPUSGFP4Dequant::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto output = outputs[0];
    size_t elementCount = output->elementSize();
    if (elementCount == 0) {
        return NO_ERROR;
    }
    if (mContainer.empty()) {
        return INVALID_VALUE;
    }

    float* dest = output->host<float>();
    if (nullptr == dest) {
        return INVALID_VALUE;
    }
    // Malformed container: return an error rather than writing partial
    // garbage silently. All planes decode through the SPATIAL crop overload
    // (Phase 12 codec fix -- see onResize); the aligned case is the same
    // spatial decode with a no-op stride crop.
    auto param = mOp->main_as_SGFP4DequantParam();
    int dimO = 0, dimI = 0;
    if (!readDecodeDims(param, dimO, dimI)) {
        return INVALID_VALUE;
    }
    bool ok = dequant_sgfp4_container_cpu_crop(mContainer.data(), mContainer.size(), dest, dimO, dimI, mPaddedDimO,
                                               mPaddedDimI);
    if (!ok) {
        return INVALID_VALUE;
    }
    return NO_ERROR;
}

class CPUSGFP4DequantCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUSGFP4Dequant(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSGFP4DequantCreator, OpType_SGFP4Dequant);

} // namespace MNN
