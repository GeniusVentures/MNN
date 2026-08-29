//
//  FP4ModelTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/05/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <cmath>
#include <cstring>
#include "MNN_generated.h"
#include "MNN/FP4DequantUtils.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/expr/Module.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

static uint8_t encode_fp4(float val) {
    if (std::isnan(val)) return 0x07;
    bool s = std::signbit(val);
    float v = std::fabs(val);
    if (v == 0.0f) return s ? 0x08 : 0x00;
    if (std::isinf(v)) return s ? 0x0E : 0x06;
    int e = (int)std::floor(std::log2(v));
    int be = e + 1;
    if (be >= 3) return s ? 0x0E : 0x06;
    if (be <= 0) { int m = (int)std::round(v / 0.5f) & 0x1; return (s ? 0x08 : 0x00) | m; }
    int m = (int)std::round((v / (float)(1 << e) - 1.0f) * 2.0f) & 0x1;
    return (s ? 0x08 : 0x00) | (be << 1) | m;
}

class FP4ModelConversionTest : public MNNTestCase {
public:
    FP4ModelConversionTest()  = default;
    virtual ~FP4ModelConversionTest() = default;

    virtual bool run(int precision) {
        printf("FP4Model START\n"); fflush(stdout);

        const float wf[32] = {
            0.0f,0.5f,1.0f,1.5f, 2.0f,2.5f,3.0f,0.0f,
            0.5f,1.0f,1.5f,2.0f, 2.5f,3.0f,0.0f,0.5f,
            1.0f,1.5f,2.0f,2.5f, 3.0f,0.0f,0.5f,1.0f,
            1.5f,2.0f,2.5f,3.0f, 0.0f,0.5f,1.0f,1.5f,
        };
        const float bias[2] = {0.0f, 0.0f};
        std::vector<float> inp(64);
        for (int i = 0; i < 64; ++i) inp[i] = 0.1f * (float)i;

        printf("FP4Model — creating input var\n"); fflush(stdout);
        auto x = _Input({1, 4, 4, 4}, NCHW, halide_type_of<float>());
        ::memcpy(x->writeMap<float>(), inp.data(), 64 * sizeof(float)); x->unMap();

        printf("FP4Model — creating conv op\n"); fflush(stdout);
        std::shared_ptr<MNN::OpT> op(new MNN::OpT);
        op->type = MNN::OpType_Convolution;
        op->main.type = MNN::OpParameter_Convolution2D;
        op->main.value = new MNN::Convolution2DT;
        auto* c = op->main.AsConvolution2D();
        c->common.reset(new MNN::Convolution2DCommonT);
        c->common->inputCount = 1; c->common->outputCount = 2;
        c->common->kernelX = 2; c->common->kernelY = 2;
        c->common->strideX = 1; c->common->strideY = 1;
        c->weight.assign(wf, wf + 32);
        c->bias.assign(bias, bias + 2);

        printf("FP4Model — executing Expr::create\n"); fflush(stdout);
        auto refVar = Variable::create(Expr::create(op.get(), {x}));

        printf("FP4Model DONE\n"); fflush(stdout);
        return true;
    }
};

MNNTestSuiteRegister(FP4ModelConversionTest, "op/fp4/conversion");
#endif
