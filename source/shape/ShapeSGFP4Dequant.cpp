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
        // Phase 11 (D-13 deviation): a rank>=3 output is conv-weight
        // geometry {O, I, kH, kW} written by the converter's
        // InsertSGFP4Dequant pass -- NCHW (CAFFE-equivalent) format so
        // Tensor::channel() resolves dim[1] (the input-channel) the way
        // conv shape inference and ConvolutionTiledExecutorMultiInput
        // expect. Rank-2 outputs (injection-tool artifacts consumed by
        // MatMul) keep the original NHWC tag -- unchanged behavior.
        TensorUtils::getDescribe(output)->dimensionFormat =
            (dims->size() >= 3) ? MNN_DATA_FORMAT_NCHW : MNN_DATA_FORMAT_NHWC;
        return true;
    }
};
REGISTER_SHAPE(ShapeSGFP4Dequant, OpType_SGFP4Dequant);
} // namespace MNN
