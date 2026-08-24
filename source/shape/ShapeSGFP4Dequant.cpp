//
//  ShapeSGFP4Dequant.cpp
//  MNN
//
//  Created by MNN on 2026/08/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

// SGFP4Dequant is Const-like: it produces a float weight tensor decoded from
// an external sidecar container, so it takes no data input. Output geometry
// comes from SGFP4DequantParam::dims (manifest-resident per spec section
// 6.1), not from copying an input tensor's shape.
class ShapeSGFP4Dequant : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == outputs.size());
        auto param = op->main_as_SGFP4DequantParam();
        if (nullptr == param || nullptr == param->dims()) {
            return false;
        }
        auto dims = param->dims();
        auto output = outputs[0];
        output->buffer().dimensions = dims->size();
        for (int i = 0; i < dims->size(); ++i) {
            output->setLength(i, dims->Get(i));
        }
        output->buffer().type = halide_type_of<float>();
        TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        return true;
    }
};
REGISTER_SHAPE(ShapeSGFP4Dequant, OpType_SGFP4Dequant);
} // namespace MNN
