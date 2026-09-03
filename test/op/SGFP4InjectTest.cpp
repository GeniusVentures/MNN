//
//  SGFP4InjectTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/08/26.
//  Copyright © 2018, Alibaba Group Holding Limited.
//
// Plan 05-01: prove the Express VARP-level graph-surgery recipe (research
// assumption A1) at the runtime level BEFORE the standalone injection tool
// (Plan 05-02) exists: build a minimal 2-op MatMul model, splice an
// OpType_SGFP4Dequant node in place of the weight Const via
// Variable::replace, save with the direct-to-file overload, reload through
// Module::load + setExternalFile, and compare the decode against the
// dequant_sgfp4_container_cpu oracle. Plus the byte-level v2 version-gate
// unit test (sgfp4_is_v2_container, SGINJ-01).
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "MNN_generated.h"
#include "MNN/SGFP4DequantUtils.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/expr/Executor.hpp"
#include "MNN/expr/Module.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "SGFP4TestUtil.hpp"
#include "SGFP4DequantFixtures.h"

using namespace MNN::Express;

namespace {

// Same cross-language tolerance as SGFP4DequantTest: decode-vs-decode is
// deterministic; this tolerates only ordinary float32 arithmetic noise.
constexpr float kFixtureRelativeTolerance = 1e-4f;

// The spliced test graph is a single-row MatMul: input {1, dimI}.
constexpr int kMatMulInputRows = 1;

// Deterministic input/weight generators (no rand() -- reproducible).
constexpr int kInputValueModulus = 977;
constexpr float kInputValueScale = 0.001f;
constexpr float kInputValueOffset = -0.4f;
constexpr int kWeightValueModulus = 251;
constexpr int kWeightValueCenter = 125;
constexpr float kWeightValueScale = 0.01f;

const sgfp4_fixtures::Fixture* findFixture(const char* name) {
    for (size_t i = 0; i < sgfp4_fixtures::kFixtureCount; ++i) {
        if (std::strcmp(sgfp4_fixtures::kFixtures[i].name, name) == 0) {
            return &sgfp4_fixtures::kFixtures[i];
        }
    }
    return nullptr;
}

// Deterministic input vector: kMatMulInputRows x dimI floats.
std::vector<float> makeInputValues(int dimI) {
    std::vector<float> values(static_cast<size_t>(kMatMulInputRows) * dimI, 0.0f);
    for (size_t k = 0; k < values.size(); ++k) {
        values[k] = kInputValueScale * static_cast<float>(k % kInputValueModulus) + kInputValueOffset;
    }
    return values;
}

// Deterministic original weight: dimO x dimI floats (replaced by the splice,
// but must be present so the pre-surgery model is a real 2-op graph).
std::vector<float> makeWeightValues(int dimO, int dimI) {
    std::vector<float> values(static_cast<size_t>(dimO) * dimI, 0.0f);
    for (size_t k = 0; k < values.size(); ++k) {
        values[k] = kWeightValueScale * static_cast<float>((static_cast<int>(k) * 37) % kWeightValueModulus - kWeightValueCenter);
    }
    return values;
}

void fillVarpWithValues(VARP var, const std::vector<float>& values) {
    auto* ptr = var->writeMap<float>();
    if (nullptr != ptr) {
        std::memcpy(ptr, values.data(), values.size() * sizeof(float));
    }
    var->unMap();
}

// Collect every expr reachable (through inputs) from the given roots.
void collectExprs(const EXPRP& expr, std::set<EXPRP>& visited) {
    if (nullptr == expr || 0 != visited.count(expr)) {
        return;
    }
    visited.insert(expr);
    for (const auto& input : expr->inputs()) {
        collectExprs(input->expr().first, visited);
    }
}

} // namespace

// ====================================================================
// op/sgfp4/inject -- the full graph-surgery recipe the injection tool
// (Plan 05-02) near-copies (SGINJ-02/03/04, A1/D-06/D-07/D-08).
// ====================================================================
class SGFP4InjectTest : public MNNTestCase {
public:
    SGFP4InjectTest()  = default;
    virtual ~SGFP4InjectTest() = default;

    virtual bool run(int precision) {
        const sgfp4_fixtures::Fixture* fixture = findFixture("mode0_uniform64");
        if (nullptr == fixture) {
            MNN_ERROR("SGFP4InjectTest: could not locate 'mode0_uniform64' fixture\n");
            return false;
        }
        const int dimO = fixture->dimO;
        const int dimI = fixture->dimI;

        const std::string modelPath   = sgfp4_test::tempPath("sgfp4_inject_model_", ".mnn");
        const std::string outPath     = sgfp4_test::tempPath("sgfp4_inject_out_", ".mnn");
        const std::string sidecarPath = sgfp4_test::tempPath("sgfp4_inject_sidecar_", ".mnn.weight");

        // Sidecar first: exactly fixture.containerSize bytes at offset 0.
        if (!sgfp4_test::writeBytes(sidecarPath, fixture->container, fixture->containerSize)) {
            MNN_ERROR("SGFP4InjectTest: failed to write sidecar '%s'\n", sidecarPath.c_str());
            return false;
        }

        bool pass        = false;
        auto cleanupGuard = [&]() {
            std::remove(modelPath.c_str());
            std::remove(outPath.c_str());
            std::remove(sidecarPath.c_str());
        };

        do {
            // -- 1. Build a minimal 2-op Express graph and save it (the
            // "normally-converted .mnn" stand-in).
            {
                const std::vector<float> weightVals = makeWeightValues(dimO, dimI);
                auto input  = _Input({kMatMulInputRows, dimI}, NHWC, halide_type_of<float>());
                auto weight = _Const(weightVals.data(), {dimO, dimI}, NHWC, halide_type_of<float>());
                weight->setName("weight");
                auto out = _MatMul(input, weight);
                Variable::save({out}, modelPath.c_str());
            }

            // -- 2. Reload as a VARP map.
            auto varMap = Variable::loadMap(modelPath.c_str());
            if (varMap.empty()) {
                MNN_ERROR("SGFP4InjectTest: loadMap of '%s' returned an empty map\n", modelPath.c_str());
                break;
            }
            auto weightIter = varMap.find("weight");
            if (varMap.end() == weightIter) {
                MNN_ERROR("SGFP4InjectTest: reloaded model has no variable named 'weight'\n");
                break;
            }
            VARP weightVar = weightIter->second;

            // -- 3. Build the SGFP4Dequant node from a hand-built OpT
            // (near-copy of SGFP4DequantTest::runSgfp4Module).
            std::shared_ptr<MNN::OpT> op(new MNN::OpT);
            op->type     = MNN::OpType_SGFP4Dequant;
            op->main.type = MNN::OpParameter_SGFP4DequantParam;
            auto* param   = new MNN::SGFP4DequantParamT;
            param->magic   = MNN::kSGFP4Magic;
            param->external = {0, static_cast<int64_t>(fixture->containerSize)};
            param->dims     = {dimO, dimI};
            op->main.value  = param;
            // Op.externalPath must be set literally on the op: this op type
            // is not one of the types createExecutionWithExternal rewrites
            // with a session-derived externalPath (Pitfall 2).
            op->externalPath = sidecarPath;
            auto dequantVar = Variable::create(Expr::create(op.get(), {}));
            dequantVar->setName(std::string("weight") + "_sgfp4");

            // -- 4. Rewire consumers: after replace, weightVar is the live
            // node; dequantVar must NOT be kept in any save set (Pitfall 4).
            Variable::replace(weightVar, dequantVar);

            // -- 5. Recompute outputs AFTER rewiring and save direct-to-file.
            auto inputOutputs = Variable::getInputAndOutput(varMap);
            auto varOutputs   = Variable::mapToSequence(inputOutputs.second);
            Variable::save(varOutputs, outPath.c_str());

            // -- 6. Reload the spliced artifact via Module::load. The module
            // is 1-input, so feed a fresh input VARP (NOT the 0-input form).
            MNN::ScheduleConfig config;
            config.type = MNN_FORWARD_CPU;
            std::shared_ptr<Executor::RuntimeManager> rtmgr(
                Executor::RuntimeManager::createRuntimeManager(config));
            // Before load -- Pitfall 5.
            rtmgr->setExternalFile(sidecarPath);
            std::shared_ptr<Module> m(Module::load({}, {}, outPath.c_str(), rtmgr));
            if (nullptr == m) {
                MNN_ERROR("SGFP4InjectTest: Module::load of spliced artifact returned null\n");
                break;
            }

            const std::vector<float> inputVals = makeInputValues(dimI);
            auto inputVarp = _Input({kMatMulInputRows, dimI}, NHWC, halide_type_of<float>());
            fillVarpWithValues(inputVarp, inputVals);
            auto outputs = m->onForward({inputVarp});
            if (outputs.empty()) {
                MNN_ERROR("SGFP4InjectTest: spliced module produced no outputs\n");
                break;
            }
            auto outVar  = outputs[0];
            auto* outPtr = outVar->readMap<float>();
            auto outInfo = outVar->getInfo();
            if (nullptr == outPtr || nullptr == outInfo) {
                MNN_ERROR("SGFP4InjectTest: spliced module output has no data/info\n");
                break;
            }
            const int outCount = kMatMulInputRows * dimO;
            if (static_cast<int>(outInfo->size) != outCount) {
                MNN_ERROR("SGFP4InjectTest: output element count %d != expected %d\n",
                          static_cast<int>(outInfo->size), outCount);
                break;
            }

            // -- 7. Oracle comparison (SGINJ-04): reference =
            // _MatMul(_Const(input), _Const(oracleWeight)).
            std::vector<float> oracle(static_cast<size_t>(dimO) * dimI, 0.0f);
            if (!MNN::dequant_sgfp4_container_cpu(fixture->container, fixture->containerSize, oracle.data(),
                                                  oracle.size())) {
                MNN_ERROR("SGFP4InjectTest: oracle decode of fixture container failed\n");
                break;
            }
            auto refIn   = _Const(inputVals.data(), {kMatMulInputRows, dimI}, NHWC, halide_type_of<float>());
            auto refW    = _Const(oracle.data(), {dimO, dimI}, NHWC, halide_type_of<float>());
            auto refOut  = _MatMul(refIn, refW);
            const float* refPtr = refOut->readMap<float>();
            if (nullptr == refPtr) {
                MNN_ERROR("SGFP4InjectTest: oracle reference MatMul produced no data\n");
                break;
            }
            if (!checkVectorByRelativeError<float>(outPtr, refPtr, outCount, kFixtureRelativeTolerance)) {
                MNN_ERROR("SGFP4InjectTest: spliced-artifact decode mismatch vs dequant_sgfp4_container_cpu oracle\n");
                break;
            }

            // -- 8. Graph-structure assertion (A1/D-07): the reloaded
            // artifact must contain exactly one SGFP4Dequant op and zero
            // weight-Const exprs (the original weight is dead-dropped).
            {
                auto reloaded = Variable::loadMap(outPath.c_str());
                if (reloaded.empty()) {
                    MNN_ERROR("SGFP4InjectTest: loadMap of spliced artifact returned an empty map\n");
                    break;
                }
                std::set<EXPRP> exprs;
                for (const auto& nameVar : reloaded) {
                    collectExprs(nameVar.second->expr().first, exprs);
                }
                int dequantCount = 0;
                int constCount   = 0;
                for (const auto& expr : exprs) {
                    if (nullptr != expr->get() && MNN::OpType_SGFP4Dequant == expr->get()->type()) {
                        ++dequantCount;
                    }
                    if (VARP::CONSTANT == expr->inputType()) {
                        ++constCount;
                    }
                }
                if (1 != dequantCount) {
                    MNN_ERROR("SGFP4InjectTest: expected exactly 1 SGFP4Dequant expr, found %d\n", dequantCount);
                    break;
                }
                if (0 != constCount) {
                    MNN_ERROR("SGFP4InjectTest: expected 0 weight-Const exprs after splice, found %d\n", constCount);
                    break;
                }
            }

            pass = true;
        } while (false);

        cleanupGuard();
        if (pass) {
            MNN_PRINT("SGFP4InjectTest: graph surgery + save/reload + oracle decode PASSED\n");
        }
        return pass;
    }
};
MNNTestSuiteRegister(SGFP4InjectTest, "op/sgfp4/inject");

// ====================================================================
// op/sgfp4/inject_v1_reject -- byte-level version gate (SGINJ-01).
// ====================================================================
class SGFP4InjectV1RejectTest : public MNNTestCase {
public:
    SGFP4InjectV1RejectTest()  = default;
    virtual ~SGFP4InjectV1RejectTest() = default;

    virtual bool run(int precision) {
        const sgfp4_fixtures::Fixture* fixture = findFixture("mode0_uniform64");
        if (nullptr == fixture) {
            MNN_ERROR("SGFP4InjectV1RejectTest: could not locate 'mode0_uniform64' fixture\n");
            return false;
        }

        // Known-good v2 bytes must pass the gate.
        if (!MNN::sgfp4_is_v2_container(fixture->container, fixture->containerSize)) {
            MNN_ERROR("SGFP4InjectV1RejectTest: known-good v2 fixture was rejected by the version gate\n");
            return false;
        }

        // Bad magic: flip the first byte (SGFP4DequantTest bad-magic pattern).
        {
            std::vector<uint8_t> bad(fixture->container, fixture->container + fixture->containerSize);
            bad[0] ^= 0xFF;
            if (MNN::sgfp4_is_v2_container(bad.data(), bad.size())) {
                MNN_ERROR("SGFP4InjectV1RejectTest: bad-magic container was accepted\n");
                return false;
            }
        }

        // Bad version.
        {
            std::vector<uint8_t> bad(fixture->container, fixture->container + fixture->containerSize);
            bad[MNN::kSGFP4VersionByteOffset] = kBadVersionByte;
            if (MNN::sgfp4_is_v2_container(bad.data(), bad.size())) {
                MNN_ERROR("SGFP4InjectV1RejectTest: bad-version container was accepted\n");
                return false;
            }
        }

        // v1-layout buffer: the legacy headers[B]|offsets[B]|codes_blob form
        // has no SGF4 magic at all -- any 32 bytes without the magic stand in.
        {
            const std::vector<uint8_t> v1Layout(kV1LayoutProbeBytes, 0x00);
            if (MNN::sgfp4_is_v2_container(v1Layout.data(), v1Layout.size())) {
                MNN_ERROR("SGFP4InjectV1RejectTest: v1-layout (magic-less) buffer was accepted\n");
                return false;
            }
        }

        // Null pointer.
        if (MNN::sgfp4_is_v2_container(nullptr, fixture->containerSize)) {
            MNN_ERROR("SGFP4InjectV1RejectTest: null buffer was accepted\n");
            return false;
        }

        // Too short: one byte below the fixed-header size.
        {
            const std::vector<uint8_t> shortBuf(MNN::kSGFP4FixedHeaderSize - 1, 0x00);
            if (MNN::sgfp4_is_v2_container(shortBuf.data(), shortBuf.size())) {
                MNN_ERROR("SGFP4InjectV1RejectTest: too-short buffer was accepted\n");
                return false;
            }
        }

        MNN_PRINT("SGFP4InjectV1RejectTest: version-gate accept/reject cases PASSED\n");
        return true;
    }

private:
    static const uint8_t kBadVersionByte;
    static const size_t kV1LayoutProbeBytes;
};
const uint8_t SGFP4InjectV1RejectTest::kBadVersionByte   = 0xFF;
const size_t SGFP4InjectV1RejectTest::kV1LayoutProbeBytes = 32;
MNNTestSuiteRegister(SGFP4InjectV1RejectTest, "op/sgfp4/inject_v1_reject");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
