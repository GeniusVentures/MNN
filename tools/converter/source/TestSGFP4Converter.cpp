//
//  TestSGFP4Converter.cpp
//  MNNConverter
//
//  Created by MNN on 2026/08/28.
//  Copyright © 2018, Alibaba Group Holding Limited.
//
// Plan 08-06 (D-09): converter round-trip test for the SGFP4
// externalization added in 08-04. Lives here (not in test/) because
// RemoveAndStoreParam/saveExternalData live in MNNConvertDeps, built only
// under MNN_BUILD_CONVERTER=ON, and run_test.out links only MNN_DEPS.
//
// PHASE A (layout): a NetT with 2 SGFP4 ops + 1 trailing Convolution2D is
//   driven through saveExternalData; asserts the sidecar is 16-byte
//   aligned, monotonic, non-overlapping; external == {offset, true-size};
//   buffers cleared (no dual-source).
// PHASE B (reload+decode parity): a single-SGFP4-op NetT is externalized,
//   serialized to .mnn with the literal op->externalPath, reloaded via the
//   classic Interpreter/Session API, and its decode is compared to the
//   CPU oracle.
//
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "MNN_generated.h"
#include "MNN/SGFP4DequantUtils.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"
#include "common/CommonUtils.hpp"
#include "SGFP4TestUtil.hpp"

#define CHECK(cond, msg)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            printf("TestSGFP4Converter: FAIL %s (at %s:%d)\n", msg, __FILE__, __LINE__);                                \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

namespace {

constexpr float kParityTolerance = 1e-6f; // decode-vs-oracle is deterministic

bool relativeErrorWithin(const std::vector<float>& got, const std::vector<float>& want, float rtol) {
    if (got.size() != want.size()) {
        return false;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = got[i] - want[i];
        const float base = want[i] > 0 ? want[i] : -want[i];
        if ((diff > 0 ? diff : -diff) > rtol * (base > 1.0f ? base : 1.0f)) {
            return false;
        }
    }
    return true;
}

// Build a buffer-mode SGFP4 OpT over the given container bytes.
std::unique_ptr<MNN::OpT> makeSgfp4Op(const std::vector<uint8_t>& container, int dimO, int dimI) {
    std::unique_ptr<MNN::OpT> op(new MNN::OpT);
    op->type      = MNN::OpType_SGFP4Dequant;
    op->main.type = MNN::OpParameter_SGFP4DequantParam;
    auto* param   = new MNN::SGFP4DequantParamT;
    param->magic  = MNN::kSGFP4Magic;
    param->dims   = {dimO, dimI};
    // SGFP4DequantParamT::buffer is std::vector<int8_t> (flatc [byte]);
    // copy the raw container bytes across.
    param->buffer.resize(container.size());
    if (!container.empty()) {
        std::memcpy(param->buffer.data(), container.data(), container.size());
    }
    op->main.value = param;
    return op;
}

// Serialize a NetT to a .mnn file (flatbuffers).
bool serializeNet(const MNN::NetT& net, const std::string& path) {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto netOffset = MNN::Net::Pack(builder, &net);
    builder.Finish(netOffset);
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
              static_cast<std::streamsize>(builder.GetSize()));
    return true;
}

} // namespace

int main(int argc, const char* argv[]) {
    (void)argc;
    (void)argv;
    const std::string cwd = sgfp4_test::cwdPath();

    // Containers: two different byte sizes for distinct aligned footprints.
    std::vector<uint8_t> bytes0, bytes1;
    CHECK(sgfp4_test::buildContainerUniform64(64, 64, bytes0), "build container 64x64");
    CHECK(sgfp4_test::buildContainerUniform64(64, 128, bytes1), "build container 64x128");
    const size_t trueSize0 = bytes0.size();
    const size_t trueSize1 = bytes1.size();
    const size_t aligned0  = MNN::sgfp4_align16(trueSize0);
    const size_t aligned1  = MNN::sgfp4_align16(trueSize1);

    // ====================================================================
    // PHASE A: layout assertions over a mixed NetT (2x SGFP4 + 1x Conv2D).
    // ====================================================================
    const std::string sidecarA = cwd + "/" + sgfp4_test::tempPath("sgfp4_conv_A_", ".weight");
    {
        std::unique_ptr<MNN::NetT> net(new MNN::NetT);
        net->oplists.push_back(makeSgfp4Op(bytes0, 64, 64));
        net->oplists.push_back(makeSgfp4Op(bytes1, 64, 128));
        // Trailing Convolution2D with a 16-float weight: proves mixed-type
        // monotonic non-overlap (T-08-10).
        auto conv = std::unique_ptr<MNN::OpT>(new MNN::OpT);
        conv->type      = MNN::OpType_Convolution;
        conv->main.type = MNN::OpParameter_Convolution2D;
        conv->inputIndexes = {0}; // size 1 so the Convolution2D case runs
        auto* convParam = new MNN::Convolution2DT;
        convParam->common.reset(new MNN::Convolution2DCommonT);
        convParam->weight.resize(16, 0.5f);
        conv->main.value = convParam;
        net->oplists.push_back(std::move(conv));

        CHECK(saveExternalData(net, sidecarA), "saveExternalData(A)");

        const auto* p0 = net->oplists[0]->main.AsSGFP4DequantParam();
        const auto* p1 = net->oplists[1]->main.AsSGFP4DequantParam();
        const auto* pc = net->oplists[2]->main.AsConvolution2D();
        CHECK(p0->external.size() == 2 && p0->external[0] == 0 && p0->external[1] == (int64_t)trueSize0,
              "op0 external == {0, trueSize0}");
        CHECK(p1->external.size() == 2 && p1->external[0] == (int64_t)aligned0 &&
                  p1->external[1] == (int64_t)trueSize1,
              "op1 external == {align16(trueSize0), trueSize1}");
        CHECK(pc->external.size() >= 1 && pc->external[0] == (int64_t)(aligned0 + aligned1),
              "Convolution2D offset monotonic past both SGFP4 regions");
        CHECK(p0->buffer.empty() && p1->buffer.empty(), "both SGFP4 buffers cleared after store");

        std::vector<uint8_t> sidecar;
        CHECK(sgfp4_test::readBytes(sidecarA, sidecar), "read sidecar(A)");
        const size_t convBytes = 16 * sizeof(float); // Convolution2D weight not padded
        CHECK(sidecar.size() == aligned0 + aligned1 + convBytes, "sidecar length == aligned total + conv bytes");
        // Zone integrity: each SGFP4 region's true bytes match the source.
        CHECK(0 == std::memcmp(sidecar.data(), bytes0.data(), trueSize0), "op0 region bytes intact");
        CHECK(0 == std::memcmp(sidecar.data() + aligned0, bytes1.data(), trueSize1), "op1 region bytes intact");
    }
    std::remove(sidecarA.c_str());

    // ====================================================================
    // PHASE B: reload + decode parity via the classic Interpreter API.
    // ====================================================================
    const std::string mnnB    = cwd + "/" + sgfp4_test::tempPath("sgfp4_conv_B_", ".mnn");
    const std::string sidecarB = mnnB + ".weight";
    {
        std::unique_ptr<MNN::NetT> net(new MNN::NetT);
        net->tensorName   = {"output"};
        net->outputName   = {"output"};
        net->tensorNumber = 1;
        auto op           = makeSgfp4Op(bytes1, 64, 128);
        op->outputIndexes = {0};
        net->oplists.push_back(std::move(op));

        CHECK(saveExternalData(net, sidecarB), "saveExternalData(B)");
        // D-12 non-interception: the literal per-Op path is the ONLY way
        // SGFP4 resolves the sidecar at runtime.
        net->oplists[0]->externalPath = sidecarB;
        CHECK(serializeNet(*net, mnnB), "serialize .mnn(B)");
        // Oracle.
        std::vector<float> oracle(static_cast<size_t>(64) * 128);
        CHECK(MNN::dequant_sgfp4_container_cpu(bytes1.data(), bytes1.size(), oracle.data(), oracle.size()),
              "oracle decode(B)");

        // Classic API reload + run (SGProcessingManager path).
        std::shared_ptr<MNN::Interpreter> interp(MNN::Interpreter::createFromFile(mnnB.c_str()),
                                                 MNN::Interpreter::destroy);
        CHECK(nullptr != interp, "Interpreter::createFromFile(B)");
        MNN::ScheduleConfig cfg;
        cfg.type = MNN_FORWARD_CPU;
        auto session = interp->createSession(cfg);
        CHECK(nullptr != session, "createSession(B)");
        interp->resizeSession(session);
        const MNN::ErrorCode code = interp->runSession(session);
        CHECK(MNN::NO_ERROR == code, "runSession(B)");
        auto outputTensor = interp->getSessionOutput(session, nullptr);
        CHECK(nullptr != outputTensor, "getSessionOutput(B)");
        std::shared_ptr<MNN::Tensor> outUser(new MNN::Tensor(outputTensor, MNN::Tensor::CAFFE));
        outputTensor->copyToHostTensor(outUser.get());
        const float* got = outUser->host<float>();
        CHECK(nullptr != got, "output host(B)");
        const size_t outCount = outUser->elementSize();
        CHECK(outCount == oracle.size(), "output count(B)");
        std::vector<float> gotVec(got, got + outCount);
        CHECK(relativeErrorWithin(gotVec, oracle, kParityTolerance), "reload decode == oracle(B)");
    }
    std::remove(mnnB.c_str());
    std::remove(sidecarB.c_str());

    printf("TestSGFP4Converter: PASS (layout + reload parity)\n");
    return 0;
}
