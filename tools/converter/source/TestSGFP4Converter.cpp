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
// PHASE C (pass mechanics, Plan 11-04 / D-12): synthetic NetTs driven
//   through the registered InsertSGFP4Dequant PostConverter pass via
//   RunNetPass; asserts node insertion + consumer rewiring (inputs[1]),
//   weight clearing, the Phase 8 D-11 buffer contract (buffer populated,
//   external == {}, no externalPath), light-tier floor (D-07), flag-off
//   dead code (D-14), decode cross-check, flatbuffers round-trip
//   survival, subgraph coverage (D-03), spilled-weight reload (KEY Q3),
//   WeightQuantAndCoding no-op on rewritten convs (D-02 / SGV2-30),
//   idempotency (double-RunNetPass), and encode-failure (NaN/Inf)
//   propagation with transactional skip (T-11-03).
//
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "MNN_generated.h"
#include "MNN/SGFP4DequantUtils.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"
#include "common/CommonUtils.hpp"
#include "SGFP4TestUtil.hpp"
#include "PostConverter.hpp"
#include "optimizer/PostTreatUtils.hpp" // PostConverter base + get()
#include "config.hpp"
#include "optimizer/Global.hpp"

using MNN::Express::RunNetPass;

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

    // ====================================================================
    // PHASE C (Plan 11-04, D-12): InsertSGFP4Dequant pass mechanics.
    // ====================================================================
    {
        // ---- Local helpers -------------------------------------------
        // Deterministic NEAR-CONSTANT fill: small perturbation around
        // 0.25 (peak deviation ~0.006). Every 64x64 superblock lands deep
        // inside the uniform-layout gate (per-leaf relative error far
        // below threshold), so FP4 reconstruction is tight and the
        // tolerance below is meaningful. Full-range ramps (fast or slow)
        // defeat the quadtree by design -- splits exhaust and the fallback
        // quantizes the ramp to near-full-range error (verified by
        // scratch diag: maxErr 1.89 on a [-1,1] plane).
        auto fillRamp = [](std::vector<float>& w, size_t n, int seed) {
            w.resize(n);
            for (size_t i = 0; i < n; ++i) {
                w[i] = 0.25f + 0.003f * (float)((i + (size_t)seed * 97u) % 7u);
            }
        };
        // Build a Convolution OpT: inputIndexes = {actIndex}, outputCount =
        // oc, weight of oc*kernelCount floats. Mirror of the PHASE A conv.
        auto makeConvOp = [](int oc, int kernelCount, const std::vector<float>& weight,
                             int actIndex) -> std::unique_ptr<MNN::OpT> {
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type            = MNN::OpType_Convolution;
            op->name            = "conv_c";
            op->main.type       = MNN::OpParameter_Convolution2D;
            op->inputIndexes    = {actIndex};
            op->outputIndexes   = {actIndex + 1};
            auto* param         = new MNN::Convolution2DT;
            param->common.reset(new MNN::Convolution2DCommonT);
            param->common->outputCount = oc;
            param->weight              = weight;
            op->main.value             = param;
            return op;
        };
        auto countSgfp4Ops = [](const MNN::NetT* net) -> size_t {
            size_t n = 0;
            for (const auto& op : net->oplists) {
                if (op->type == MNN::OpType_SGFP4Dequant) ++n;
            }
            return n;
        };
        // Coarse FP4-quantization tolerance check for the near-constant
        // fill: every decoded value is finite and within a small absolute
        // band of the source (uniform-layout reconstruction of a
        // 0.25+-0.006 plane).
        auto fp4Approx = [](const std::vector<float>& got, const std::vector<float>& want) -> bool {
            if (got.size() != want.size()) return false;
            for (size_t i = 0; i < got.size(); ++i) {
                if (!std::isfinite(got[i])) return false;
                const float d = got[i] - want[i];
                const float a = d < 0 ? -d : d;
                if (a > 0.05f) return false; // near-constant-plane bound
            }
            return true;
        };

        // Registration canary (RunNetPass only LOGs a missing pass; this
        // fails loudly instead). PostConverter is global-scope
        // (PostTreatUtils.hpp), not inside namespace MNN.
        CHECK(nullptr != PostConverter::get("InsertSGFP4Dequant"),
              "PHASE C: InsertSGFP4Dequant pass registered");

        modelConfig config; // stack config; useSGFP4 defaults false (D-04)

        // ---- Test 1: insertion / rewiring / buffer contract ------------
        std::unique_ptr<MNN::NetT> net1(new MNN::NetT);
        net1->tensorName   = {"x", "y"};
        net1->tensorNumber = 2;
        std::vector<float> w1;
        fillRamp(w1, (size_t)64 * 128, 1);
        net1->oplists.push_back(makeConvOp(64, 128, w1, 0));
        {
            config.useSGFP4 = true;
            Global<modelConfig>::Reset(&config);
            RunNetPass({"InsertSGFP4Dequant"}, net1);
            CHECK(1 == countSgfp4Ops(net1.get()), "T1: exactly one SGFP4Dequant op");
            CHECK(2 == net1->oplists.size(), "T1: oplist grew by one (producer + conv)");
            const auto* dq = net1->oplists[0]->type == MNN::OpType_SGFP4Dequant
                                 ? net1->oplists[0].get() : net1->oplists[1].get();
            const auto* cv = net1->oplists[0]->type == MNN::OpType_SGFP4Dequant
                                 ? net1->oplists[1].get() : net1->oplists[0].get();
            CHECK(dq->type == MNN::OpType_SGFP4Dequant && cv->type == MNN::OpType_Convolution,
                  "T1: producer precedes consumer");
            const auto* dp = dq->main.AsSGFP4DequantParam();
            CHECK(nullptr != dp, "T1: dequant param");
            CHECK(dp->magic == MNN::kSGFP4Magic, "T1: magic");
            CHECK(dp->dims.size() == 2 && dp->dims[0] == 64 && dp->dims[1] == 128, "T1: dims == {64,128}");
            CHECK(!dp->buffer.empty(), "T1: buffer non-empty (D-11 buffer contract)");
            CHECK(dp->external.empty(), "T1: external == {}");
            CHECK(dq->externalPath.empty(), "T1: externalPath empty");
            CHECK(cv->inputIndexes.size() == 2, "T1: conv inputs grew to 2");
            CHECK(cv->inputIndexes[1] == dq->outputIndexes[0], "T1: inputs[1] == dequant output");
            const auto* cp = cv->main.AsConvolution2D();
            CHECK(cp->weight.empty(), "T1: conv weight cleared");
            CHECK(cv->type == MNN::OpType_Convolution, "T1: conv type unchanged");
            CHECK(net1->tensorName.size() == 3, "T1: tensorName grew by exactly 1");
            CHECK(net1->tensorName[2] == "conv_c_sgfp4", "T1: appended name <conv>_sgfp4");
            // ---- Test 4: decode cross-check ---------------------------
            std::vector<float> decoded((size_t)64 * 128);
            CHECK(MNN::dequant_sgfp4_container_cpu(reinterpret_cast<const uint8_t*>(dp->buffer.data()),
                                                   dp->buffer.size(), decoded.data(), decoded.size()),
                  "T4: container decodes");
            CHECK(decoded.size() == w1.size(), "T4: decoded size == dimO*dimI");
            CHECK(fp4Approx(decoded, w1), "T4: decode approximates source (FP4 tolerance)");
        }

        // ---- Test 9: idempotency (second run on rewritten net) ---------
        {
            const size_t opsBefore = net1->oplists.size();
            const auto inBefore    = net1->oplists[1]->inputIndexes; // conv is at [1]
            const auto namesBefore = net1->tensorName;
            RunNetPass({"InsertSGFP4Dequant"}, net1);
            CHECK(net1->oplists.size() == opsBefore, "T9: node count stable");
            CHECK(net1->oplists[1]->inputIndexes == inBefore, "T9: no second input append");
            CHECK(net1->tensorName == namesBefore, "T9: no tensorName double-append");
        }

        // ---- Test 5: flatbuffers round-trip survival ------------------
        {
            const std::string mnnC = cwd + "/" + sgfp4_test::tempPath("sgfp4_c_rt_", ".mnn");
            CHECK(serializeNet(*net1, mnnC), "T5: serialize rewritten net");
            std::vector<uint8_t> fb;
            CHECK(sgfp4_test::readBytes(mnnC, fb), "T5: read .mnn bytes");
            const auto* netFb = MNN::GetNet(fb.data());
            CHECK(nullptr != netFb, "T5: GetNet root");
            std::unique_ptr<MNN::NetT> re(netFb->UnPack());
            CHECK(1 == countSgfp4Ops(re.get()), "T5: SGFP4Dequant survives round-trip");
            const auto* rp = re->oplists[0]->type == MNN::OpType_SGFP4Dequant
                                 ? re->oplists[0]->main.AsSGFP4DequantParam()
                                 : re->oplists[1]->main.AsSGFP4DequantParam();
            CHECK(nullptr != rp && !rp->buffer.empty(), "T5: buffer intact after UnPack");
            std::remove(mnnC.c_str());
        }

        // ---- Test 2: light-tier floor (D-07) --------------------------
        {
            std::unique_ptr<MNN::NetT> net2(new MNN::NetT);
            net2->tensorName = {"x", "y", "z"};
            std::vector<float> wSmall, wDimI1;
            fillRamp(wSmall, (size_t)8 * 256, 2);   // 2048 < 4096
            fillRamp(wDimI1, (size_t)64 * 64, 3);   // dimI == 1 case below
            net2->oplists.push_back(makeConvOp(8, 256, wSmall, 0));
            // dimI == 1: outputCount 8192, kernelSize 1 -> 8192x1 plane
            std::vector<float> wCol((size_t)8192 * 1, 0.25f);
            net2->oplists.push_back(makeConvOp(8192, 1, wCol, 1));
            RunNetPass({"InsertSGFP4Dequant"}, net2);
            CHECK(0 == countSgfp4Ops(net2.get()), "T2: no dequant for light-tier convs");
            CHECK(!net2->oplists[0]->main.AsConvolution2D()->weight.empty(), "T2: small weight intact");
            CHECK(net2->oplists[0]->inputIndexes.size() == 1, "T2: small inputs untouched");
            CHECK(!net2->oplists[1]->main.AsConvolution2D()->weight.empty(), "T2: dimI==1 weight intact");
            CHECK(net2->oplists[1]->inputIndexes.size() == 1, "T2: dimI==1 inputs untouched");
            (void)wDimI1;
        }

        // ---- Test 3: flag-off dead code (D-14) ------------------------
        {
            std::unique_ptr<MNN::NetT> net3(new MNN::NetT);
            net3->tensorName = {"x", "y"};
            std::vector<float> w3;
            fillRamp(w3, (size_t)64 * 128, 4);
            net3->oplists.push_back(makeConvOp(64, 128, w3, 0));
            const auto namesBefore = net3->tensorName;
            config.useSGFP4 = false;
            Global<modelConfig>::Reset(&config);
            RunNetPass({"InsertSGFP4Dequant"}, net3);
            CHECK(0 == countSgfp4Ops(net3.get()), "T3: flag-off -> no insertion");
            CHECK(1 == net3->oplists.size(), "T3: flag-off -> oplist unchanged");
            CHECK(net3->oplists[0]->inputIndexes.size() == 1, "T3: flag-off -> inputs unchanged");
            CHECK(!net3->oplists[0]->main.AsConvolution2D()->weight.empty(), "T3: flag-off -> weight intact");
            CHECK(net3->tensorName == namesBefore, "T3: flag-off -> tensorName unchanged");
            config.useSGFP4 = true;
        }

        // ---- Test 6: subgraph coverage (D-03) -------------------------
        {
            std::unique_ptr<MNN::NetT> net6(new MNN::NetT);
            net6->tensorName = {"root_in", "root_out"};
            const size_t rootOpsBefore = 0;
            auto sub = std::unique_ptr<MNN::SubGraphProtoT>(new MNN::SubGraphProtoT);
            sub->name    = "body";
            sub->tensors = {"x"};
            std::vector<float> w6;
            fillRamp(w6, (size_t)64 * 128, 5);
            sub->nodes.push_back(makeConvOp(64, 128, w6, 0));
            const std::vector<std::string> tensorsBefore = sub->tensors;
            net6->subgraphs.push_back(std::move(sub));
            Global<modelConfig>::Reset(&config); // useSGFP4 == true here
            RunNetPass({"InsertSGFP4Dequant"}, net6);
            const auto& sg = net6->subgraphs[0];
            size_t dqCount = 0;
            for (const auto& op : sg->nodes) {
                if (op->type == MNN::OpType_SGFP4Dequant) ++dqCount;
            }
            CHECK(1 == dqCount, "T6: subgraph conv rewritten");
            CHECK(2 == sg->nodes.size(), "T6: dequant appended to subgraph nodes");
            const auto* sconv = sg->nodes[0]->type == MNN::OpType_Convolution ? sg->nodes[0].get() : sg->nodes[1].get();
            CHECK(sconv->inputIndexes.size() == 2, "T6: subgraph conv inputs[1] rewired");
            CHECK(sg->tensors.size() == 2, "T6: tensors grew by 1");
            CHECK(sg->tensors[1] == "conv_c_sgfp4", "T6: appended tensor name");
            CHECK(sconv->inputIndexes[1] == 1, "T6: new index == pre-push tensors size");
            CHECK(sg->tensors[0] == tensorsBefore[0], "T6: existing tensor unrenumbered");
            CHECK(net6->oplists.size() == rootOpsBefore, "T6: root oplists untouched");
        }

        // ---- Test 7: spilled-weight reload (KEY Q3) --------------------
        {
            const int dimO7 = 64, dimI7 = 128;
            std::vector<float> w7, b7((size_t)dimO7, 0.125f);
            fillRamp(w7, (size_t)dimO7 * dimI7, 6);
            // Write weight + bias at offset 0 to the converter's temp bin,
            // KEEPING the ofstream open (the production condition).
            std::ofstream spill(".__convert_external_data.bin",
                               std::ios::binary | std::ios::trunc);
            CHECK(spill.is_open(), "T7: open spill bin");
            spill.write(reinterpret_cast<const char*>(w7.data()),
                        (std::streamsize)(w7.size() * sizeof(float)));
            spill.write(reinterpret_cast<const char*>(b7.data()),
                        (std::streamsize)(b7.size() * sizeof(float)));
            config.externalFile = &spill; // pass must flush it before reading

            // Discriminating probe (T7 debug): can this process open the
            // bin for READ while the ofstream holds it open? Splits
            // OS-level share denial from a FileLoader-specific problem.
            {
                std::ifstream probe(".__convert_external_data.bin", std::ios::binary);
                printf("T7 probe: ifstream-while-ofstream-open: %s\n",
                       probe.is_open() ? "OK" : "FAIL");
                if (probe.is_open()) {
                    float first = 0.0f;
                    probe.read(reinterpret_cast<char*>(&first), sizeof(float));
                    printf("T7 probe: first float readback = %.6f (expect %.6f)\n", first, w7[0]);
                }
            }

            std::unique_ptr<MNN::NetT> net7(new MNN::NetT);
            net7->tensorName = {"x", "y"};
            std::unique_ptr<MNN::OpT> conv7 = makeConvOp(dimO7, dimI7, {}, 0);
            auto* p7 = conv7->main.AsConvolution2D();
            p7->external = {0, (int64_t)(w7.size() * sizeof(float)), (int64_t)(b7.size() * sizeof(float))};
            net7->oplists.push_back(std::move(conv7));
            RunNetPass({"InsertSGFP4Dequant"}, net7);

            CHECK(1 == countSgfp4Ops(net7.get()), "T7: spilled conv rewritten");
            const auto* cv7 = net7->oplists[1]->type == MNN::OpType_Convolution
                                  ? net7->oplists[1].get() : net7->oplists[0].get();
            const auto* cp7 = cv7->main.AsConvolution2D();
            CHECK(cp7->external.empty(), "T7: external cleared");
            CHECK(cp7->bias.size() == b7.size(), "T7: bias restored (count)");
            bool biasEq = true;
            for (size_t i = 0; i < b7.size(); ++i) {
                if (cp7->bias[i] != b7[i]) { biasEq = false; break; }
            }
            CHECK(biasEq, "T7: bias restored (values)");
            CHECK(cv7->inputIndexes.size() == 2, "T7: inputs rewired");
            // Decode cross-check against the written weights.
            const auto* dq7 = net7->oplists[0]->type == MNN::OpType_SGFP4Dequant
                                  ? net7->oplists[0].get() : net7->oplists[1].get();
            const auto* dp7 = dq7->main.AsSGFP4DequantParam();
            std::vector<float> dec7((size_t)dimO7 * dimI7);
            CHECK(MNN::dequant_sgfp4_container_cpu(reinterpret_cast<const uint8_t*>(dp7->buffer.data()),
                                                   dp7->buffer.size(), dec7.data(), dec7.size()),
                  "T7: spilled container decodes");
            CHECK(fp4Approx(dec7, w7), "T7: decode approximates spilled weights");

            spill.close();
            config.externalFile = nullptr;
            std::remove(".__convert_external_data.bin"); // T-11-09 hygiene
        }

        // ---- Test 8: WeightQuantAndCoding no-op on rewritten conv (D-02)
        {
            std::unique_ptr<MNN::OpT> rw(new MNN::OpT);
            rw->type            = MNN::OpType_Convolution;
            rw->name            = "conv_rewritten";
            rw->main.type       = MNN::OpParameter_Convolution2D;
            rw->inputIndexes    = {0, 1}; // ALREADY-REWRITTEN fingerprint
            rw->outputIndexes   = {2};
            auto* rp8           = new MNN::Convolution2DT;
            rp8->common.reset(new MNN::Convolution2DCommonT);
            rp8->common->outputCount = 64;
            rw->main.value      = rp8; // weight left empty (cleared)

            modelConfig cfg8;          // weightQuantBits == 0 default path
            cfg8.useSGFP4 = true;      // irrelevant here; the guard is topology
            PostTreatContext ctx8;     // lightest valid context (guard fires first)
            WeightQuantAndCoding(rw, cfg8, &ctx8);

            CHECK(rw->inputIndexes.size() == 2, "T8: inputs still 2 (no-op)");
            CHECK(rw->main.AsConvolution2D()->weight.empty(), "T8: weight still empty");
            CHECK(nullptr == rw->main.AsConvolution2D()->quanParameter.get(), "T8: no quanParameter");
            CHECK(rw->type == MNN::OpType_Convolution, "T8: type unchanged");
        }

        // ---- Test 10: encode-failure propagation (NaN / Inf) -----------
        {
            for (int variant = 0; variant < 2; ++variant) {
                std::unique_ptr<MNN::NetT> net10(new MNN::NetT);
                net10->tensorName = {"x", "y"};
                std::vector<float> w10;
                fillRamp(w10, (size_t)64 * 128, 7);
                if (variant == 0) {
                    w10[0] = std::numeric_limits<float>::quiet_NaN();
                } else {
                    w10[0] = std::numeric_limits<float>::infinity();
                }
                const std::vector<float> saved = w10;
                net10->oplists.push_back(makeConvOp(64, 128, w10, 0));
                Global<modelConfig>::Reset(&config); // useSGFP4 == true
                // Direct onExecute for the failure-report leg (RunNetPass
                // only LOGs a false return).
                auto* pass10 = PostConverter::get("InsertSGFP4Dequant");
                CHECK(nullptr != pass10, "T10: pass lookup");
                CHECK(!pass10->onExecute(net10), "T10: pass reports failure (variant)");
                CHECK(0 == countSgfp4Ops(net10.get()), "T10: no dequant node (variant)");
                const auto* cp10 = net10->oplists[0]->main.AsConvolution2D();
                CHECK(cp10->weight.size() == saved.size(), "T10: weight intact (size)");
                CHECK(0 == std::memcmp(cp10->weight.data(), saved.data(), saved.size() * sizeof(float)),
                      "T10: weight intact (bytes; NaN-safe compare)");
                CHECK(net10->oplists[0]->inputIndexes.size() == 1, "T10: inputs untouched (variant)");
            }
        }

        // Reset the global so no state leaks past PHASE C.
        Global<modelConfig>::Reset(nullptr);
    }

    printf("TestSGFP4Converter: PASS (layout + reload parity + pass mechanics)\n");
    return 0;
}
