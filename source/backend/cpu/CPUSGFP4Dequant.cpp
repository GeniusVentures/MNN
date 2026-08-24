//
//  CPUSGFP4Dequant.cpp
//  MNN
//
//  Created by MNN on 2026/08/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUSGFP4Dequant.hpp"
#include "MNN/SGFP4DequantUtils.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/FileLoader.hpp"
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"

namespace MNN {

ErrorCode CPUSGFP4Dequant::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto param = mOp->main_as_SGFP4DequantParam();
    if (nullptr == param) {
        return INVALID_VALUE;
    }
    // Mirrors ConvolutionCommon.cpp's USE_EXTERNAL_DATA(param) + externalPath
    // gate: this op only supports the external-sidecar container form.
    if (!USE_EXTERNAL_DATA(param) || nullptr == mOp->externalPath()) {
        return NOT_SUPPORT;
    }

    auto external = param->external()->data();
    int64_t offset = external[0];
    int64_t size   = external[1];
    if (offset < 0 || size <= 0) {
        return INVALID_VALUE;
    }

    // Construct with init=true so valid()/size() are usable immediately,
    // mirroring Interpreter.cpp's FileLoader(file, true) + valid() pattern.
    FileLoader loader(mOp->externalPath()->c_str(), true);
    if (!loader.valid()) {
        return NOT_SUPPORT;
    }

    // T-01-04: never trust a declared size that exceeds the sidecar's own
    // size (DoS guard against an oversized external()[1]).
    size_t fileSize   = loader.size();
    size_t offsetSize = static_cast<size_t>(offset);
    size_t readSize   = static_cast<size_t>(size);
    if (offsetSize > fileSize || readSize > fileSize - offsetSize) {
        return INVALID_VALUE;
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
    // garbage silently.
    bool ok = dequant_sgfp4_container_cpu(mContainer.data(), mContainer.size(), dest, elementCount);
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
