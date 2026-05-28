//
//  CPUFP4Dequant.cpp
//  MNN
//
//  Created by MNN on 2026/05/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUFP4Dequant.hpp"
#include "MNN/FP4DequantUtils.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"

namespace MNN {

ErrorCode CPUFP4Dequant::onExecute(const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    size_t elementCount = output->elementSize();
    if (elementCount == 0) {
        return NO_ERROR;
    }

    const uint8_t* packed = input->host<uint8_t>();
    if (packed == nullptr) {
        return NO_ERROR;
    }

    float* dest = output->host<float>();
    dequant_fp4_packed_cpu(packed, dest, elementCount);

    return NO_ERROR;
}

class CPUFP4DequantCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (inputs.empty() || outputs.empty()) {
            return nullptr;
        }
        auto input  = inputs[0];
        auto output = outputs[0];
        if (input->getType().code != halide_type_uint || input->getType().bits != 8) {
            return nullptr;
        }
        size_t elementCount = output->elementSize();
        size_t expectedPackedBytes = (elementCount + 1) / 2;
        if (input->elementSize() == expectedPackedBytes) {
            return new CPUFP4Dequant(backend);
        }
        return nullptr;
    }
};

// CPUFP4DequantCreator registration is handled by CPUDequantizeCreator::onCreate
// which detects FP4-packed data and routes to CPUFP4Dequant internally.

} // namespace MNN
