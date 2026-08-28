//
//  SGFP4ClassicAPITest.cpp
//  MNNTests
//
//  Created by MNN on 2026/08/27.
//  Copyright © 2018, Alibaba Group Holding Limited.
//
// Plan 06-02: prove the Phase 5 injected artifact loads and runs through
// the CLASSIC Interpreter/Session API -- createFromFile -> createSession ->
// resizeSession -> runSession -- the exact path downstream
// SGProcessingManager::MNN_Tensor::Process() uses (SGINJ-05), with named
// input/output identification (D-16), FP32-baseline parity (D-05..D-08),
// and sidecar resolution via the op's literal externalPath with NO
// session-level setExternalFile (SGINJ-06/SC3, D-13).
//
// Two suites in one self-contained file (no committed fixtures, D-01/D-10):
//   op/sgfp4/classic_api                 -- full happy path + parity
//   op/sgfp4/classic_api_missing_sidecar -- graceful-failure probe (D-13)
//
// Corrupted-payload probing (D-14) and hand-tampered out-of-bounds offsets
// (D-15) are explicitly Phase 7 territory and absent here.
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "MNN_generated.h"
#include "MNN/SGFP4DequantUtils.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "fp4/sgfp4_inject_core.hpp"
#include "SGFP4TestUtil.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace {

// ====================================================================
// Named fixtures / framing constants (D-04/D-08: every non-obvious literal
// is a named constant; format constants come from SGFP4DequantUtils.hpp).
// ====================================================================

// Base-model / container geometry: 512x512 weight, all-UNIFORM_64 v2
// container (Phase 5 demo-container lineage, D-04).
constexpr int kMatrixDim           = 512;
constexpr int kMacroblockEdge      = 64;
constexpr int kRecordCount         = (kMatrixDim * kMatrixDim) / (kMacroblockEdge * kMacroblockEdge); // 64

// LCG input generator (D-08): identical values feed injected + baseline.
constexpr uint32_t kLcgSeed     = 0x9E3779B9u;
constexpr uint32_t kLcgMul      = 1664525u;
constexpr uint32_t kLcgAdd      = 1013904223u;
constexpr float kLcgNormalize   = 1.0f / 16777216.0f; // (state >> 8) is 24-bit

// FP32-baseline parity tolerance (D-07).
constexpr float kParityRelativeTolerance = 1e-4f;

// ====================================================================
// Plan 08-02 (D-10 pull-forward of the W-1 fix): the shared helpers in
// SGFP4TestUtil.hpp now provide the container builder and niche-dir
// writer. The former LOCAL buildContainerUniform64 here wrote ABSOLUTE
// offset-table entries (kRecordRegionStart + b*kRecordSize) -- the W-1
// offset-convention divergence -- and is replaced by the shared
// REGION-RELATIVE builder sgfp4_test::buildContainerUniform64, whose
// entries are relative to the record-region start (encoder convention).
// ====================================================================

// Named-I/O base model (D-16/D-02): Input[1,512] 'input' -> MatMul with
// Const weight[512,512] 'weight' -> 'output'; also serves as the FP32
// baseline when the weight already equals the oracle decode (D-05/D-06).
bool buildNamedBaseModel(const std::vector<float>& weight, const std::string& path) {
    auto input  = _Input({1, kMatrixDim}, NHWC, halide_type_of<float>());
    auto weightConst = _Const(weight.data(), {kMatrixDim, kMatrixDim}, NHWC, halide_type_of<float>());
    weightConst->setName("weight");
    auto out = _MatMul(input, weightConst);
    input->setName("input");
    out->setName("output");
    Variable::save({out}, path.c_str());
    return true;
}

void fillLcgInput(std::vector<float>& input) {
    input.resize(kMatrixDim, 0.0f);
    uint32_t state = kLcgSeed;
    for (int i = 0; i < kMatrixDim; ++i) {
        state     = state * kLcgMul + kLcgAdd;
        input[i]  = static_cast<float>(state >> 8) * kLcgNormalize;
    }
}

// Run one classic-API session over the given model with the given input and
// capture its 512-float output (pictureRecognition.cpp flow).
bool runClassicSession(const std::string& modelPath, const std::vector<float>& inputVals, std::vector<float>& output) {
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(modelPath.c_str()), Interpreter::destroy);
    if (nullptr == net) {
        MNN_ERROR("SGFP4ClassicAPITest: createFromFile('%s') returned null\n", modelPath.c_str());
        return false;
    }
    ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_CPU;
    auto session = net->createSession(cfg);
    if (nullptr == session) {
        MNN_ERROR("SGFP4ClassicAPITest: createSession('%s') returned null\n", modelPath.c_str());
        return false;
    }
    auto inputTensor = net->getSessionInput(session, nullptr);
    if (nullptr == inputTensor) {
        MNN_ERROR("SGFP4ClassicAPITest: no session input for '%s'\n", modelPath.c_str());
        return false;
    }
    // resizeSession returns VOID -- a resize error is swallowed here and
    // must be observed at runSession (Pitfall 1).
    net->resizeSession(session);
    ::memcpy(inputTensor->host<float>(), inputVals.data(), kMatrixDim * sizeof(float));
    const ErrorCode code = net->runSession(session);
    if (NO_ERROR != code) {
        MNN_ERROR("SGFP4ClassicAPITest: runSession('%s') returned %d\n", modelPath.c_str(), static_cast<int>(code));
        return false;
    }
    auto outputTensor = net->getSessionOutput(session, nullptr);
    if (nullptr == outputTensor) {
        MNN_ERROR("SGFP4ClassicAPITest: no session output for '%s'\n", modelPath.c_str());
        return false;
    }
    std::shared_ptr<Tensor> outUser(new Tensor(outputTensor, Tensor::CAFFE));
    outputTensor->copyToHostTensor(outUser.get());
    const float* got = outUser->host<float>();
    if (nullptr == got) {
        MNN_ERROR("SGFP4ClassicAPITest: output unreadable for '%s'\n", modelPath.c_str());
        return false;
    }
    output.assign(got, got + kMatrixDim);
    return true;
}

// Shared fixture pipeline: container -> oracle -> base model -> niche dir
// -> in-process injection (D-12) -> outPath + outPath.weight.
struct ClassicFixture {
    std::string basePath;
    std::string outPath;
    std::string sidecarPath;
    std::string nicheDir;
};

bool buildInjectedArtifact(ClassicFixture& fx) {
    const std::string cwd = sgfp4_test::cwdPath();

    // 1. In-test 512x512 all-UNIFORM_64 container; oracle-valid by
    //    construction (research A1 closed by the assertions below).
    std::vector<uint8_t> containerBytes;
    sgfp4_test::buildContainerUniform64(kMatrixDim, kMatrixDim, containerBytes);
    if (!MNN::sgfp4_is_v2_container(containerBytes.data(), containerBytes.size())) {
        MNN_ERROR("SGFP4ClassicAPITest: generated container failed sgfp4_is_v2_container\n");
        return false;
    }

    // 2. Oracle decode -- also the FP32 baseline weight source (D-06).
    std::vector<float> oracleBuf(static_cast<size_t>(kMatrixDim) * kMatrixDim, 0.0f);
    if (!MNN::dequant_sgfp4_container_cpu(containerBytes.data(), containerBytes.size(), oracleBuf.data(),
                                          oracleBuf.size())) {
        MNN_ERROR("SGFP4ClassicAPITest: oracle decode of generated container failed\n");
        return false;
    }

    // 3. Base model with weight == oracle (also the FP32 baseline, D-05).
    fx.basePath = cwd + "/" + sgfp4_test::tempPath("sgfp4_classic_base_", ".mnn");
    if (!buildNamedBaseModel(oracleBuf, fx.basePath)) {
        MNN_ERROR("SGFP4ClassicAPITest: failed to build base model '%s'\n", fx.basePath.c_str());
        return false;
    }

    // 4. Synthetic niche dir (absolute paths, Pitfall 3).
    fx.nicheDir = cwd + "/" + sgfp4_test::tempPath("sgfp4_classic_niche_", ".d");
    if (!sgfp4_test::writeNicheDir(containerBytes, fx.nicheDir, "phase6_fixture.sgfp4", kMatrixDim, kMatrixDim)) {
        MNN_ERROR("SGFP4ClassicAPITest: failed to write niche dir '%s'\n", fx.nicheDir.c_str());
        return false;
    }

    // 5. In-process injection via the shared core (D-12).
    fx.outPath     = cwd + "/" + sgfp4_test::tempPath("sgfp4_classic_out_", ".mnn");
    fx.sidecarPath = fx.outPath + ".weight";
    const char* argv[] = {"sgfp4_inject",                          // 0
                          "--model",     fx.basePath.c_str(),       // 1..2
                          "--niche-dir", fx.nicheDir.c_str(),       // 3..4
                          "--output",    fx.outPath.c_str()};       // 5..6
    if (0 != sgfp4_inject::run(7, argv)) {
        MNN_ERROR("SGFP4ClassicAPITest: sgfp4_inject::run failed\n");
        return false;
    }
    return true;
}

void cleanupFixture(const ClassicFixture& fx) {
    std::remove(fx.basePath.c_str());
    std::remove(fx.outPath.c_str());
    std::remove(fx.sidecarPath.c_str());
    std::remove((fx.nicheDir + "/phase6_fixture.sgfp4").c_str());
    std::remove((fx.nicheDir + "/manifest.json").c_str());
    sgfp4_test::removeDir(fx.nicheDir);
}

} // namespace

// ====================================================================
// op/sgfp4/classic_api -- happy path: classic load/run + named I/O +
// FP32-baseline parity (SGINJ-05, SGINJ-06; D-01..D-09, D-16).
// ====================================================================
class SGFP4ClassicAPITest : public MNNTestCase {
public:
    SGFP4ClassicAPITest()  = default;
    virtual ~SGFP4ClassicAPITest() = default;

    virtual bool run(int precision) {
        ClassicFixture fx;
        if (!buildInjectedArtifact(fx)) {
            cleanupFixture(fx);
            return false;
        }
        bool pass = false;
        do {
            // -- Classic load of the INJECTED artifact (SGINJ-05, D-03).
            std::shared_ptr<Interpreter> net(Interpreter::createFromFile(fx.outPath.c_str()), Interpreter::destroy);
            if (nullptr == net) {
                MNN_ERROR("SGFP4ClassicAPITest: createFromFile of injected '%s' returned null\n", fx.outPath.c_str());
                break;
            }
            ScheduleConfig cfg;
            cfg.type = MNN_FORWARD_CPU;
            auto session = net->createSession(cfg);
            if (nullptr == session) {
                MNN_ERROR("SGFP4ClassicAPITest: createSession of injected artifact returned null\n");
                break;
            }

            // -- Named-I/O identification (D-16, ROADMAP criterion 1): the
            // base-model names must survive injection.
            const auto& inAll  = net->getSessionInputAll(session);
            const auto& outAll = net->getSessionOutputAll(session);
            if (1 != inAll.count("input")) {
                MNN_ERROR("SGFP4ClassicAPITest: getSessionInputAll has no 'input'");
                for (const auto& kv : inAll) {
                    MNN_PRINT(" [input key '%s']", kv.first.c_str());
                }
                MNN_PRINT("\n");
                break;
            }
            if (1 != outAll.count("output")) {
                MNN_ERROR("SGFP4ClassicAPITest: getSessionOutputAll has no 'output'");
                for (const auto& kv : outAll) {
                    MNN_PRINT(" [output key '%s']", kv.first.c_str());
                }
                MNN_PRINT("\n");
                break;
            }

            // -- Feed + run (D-08). NO setExternalFile anywhere: the sidecar
            // resolves via the op's literal externalPath (SGINJ-06/SC3).
            std::vector<float> inputVals;
            fillLcgInput(inputVals);
            auto inputTensor = net->getSessionInput(session, nullptr);
            // resizeSession returns VOID (Pitfall 1): resize errors surface
            // at runSession below.
            net->resizeSession(session);
            ::memcpy(inputTensor->host<float>(), inputVals.data(), kMatrixDim * sizeof(float));
            const ErrorCode code = net->runSession(session);
            if (NO_ERROR != code) {
                MNN_ERROR("SGFP4ClassicAPITest: runSession of injected artifact returned %d\n",
                          static_cast<int>(code));
                break;
            }
            auto outputTensor = net->getSessionOutput(session, nullptr);
            std::shared_ptr<Tensor> outUser(new Tensor(outputTensor, Tensor::CAFFE));
            outputTensor->copyToHostTensor(outUser.get());
            const float* got = outUser->host<float>();
            if (nullptr == got) {
                MNN_ERROR("SGFP4ClassicAPITest: injected output unreadable\n");
                break;
            }

            // -- FP32 baseline (D-05..D-07): classic session over the base
            // model whose weight already IS the oracle decode.
            std::vector<float> baseline;
            if (!runClassicSession(fx.basePath, inputVals, baseline)) {
                break;
            }
            if (!checkVectorByRelativeError<float>(got, baseline.data(), kMatrixDim, kParityRelativeTolerance)) {
                MNN_ERROR("SGFP4ClassicAPITest: injected output != FP32 baseline within rtol %g\n",
                          kParityRelativeTolerance);
                break;
            }
            pass = true;
        } while (false);

        cleanupFixture(fx);
        if (pass) {
            MNN_PRINT("SGFP4ClassicAPITest: classic load/run + named I/O + FP32 parity PASSED\n");
        }
        return pass;
    }
};
MNNTestSuiteRegister(SGFP4ClassicAPITest, "op/sgfp4/classic_api");

// ====================================================================
// op/sgfp4/classic_api_missing_sidecar -- D-13 graceful-failure probe:
// with the sidecar deleted, load/create still succeed but runSession
// returns a non-zero ErrorCode and nothing crashes (the failure mode the
// downstream SGProcessingManager team needs documented).
// ====================================================================
class SGFP4ClassicAPIMissingSidecarTest : public MNNTestCase {
public:
    SGFP4ClassicAPIMissingSidecarTest()  = default;
    virtual ~SGFP4ClassicAPIMissingSidecarTest() = default;

    virtual bool run(int precision) {
        ClassicFixture fx;
        if (!buildInjectedArtifact(fx)) {
            cleanupFixture(fx);
            return false;
        }
        bool pass = false;
        do {
            // D-13: delete the sidecar the injected op points at.
            std::remove(fx.sidecarPath.c_str());

            // The model itself is still a valid .mnn: load + session create
            // must succeed (the failure is deferred to resize/run).
            std::shared_ptr<Interpreter> net(Interpreter::createFromFile(fx.outPath.c_str()), Interpreter::destroy);
            if (nullptr == net) {
                MNN_ERROR("SGFP4ClassicAPIMissingSidecarTest: createFromFile returned null\n");
                break;
            }
            ScheduleConfig cfg;
            cfg.type = MNN_FORWARD_CPU;
            auto session = net->createSession(cfg);
            if (nullptr == session) {
                MNN_ERROR("SGFP4ClassicAPIMissingSidecarTest: createSession returned null\n");
                break;
            }

            // resizeSession returns VOID and discards the Session::resize
            // ErrorCode (Interpreter.cpp) -- CPUSGFP4Dequant::onResize
            // returns NOT_SUPPORT on the missing sidecar, leaving
            // mNeedResize true. The observable failure is therefore at
            // runSession (returns COMPUTE_SIZE_ERROR), not here (Pitfall 1).
            net->resizeSession(session);

            const ErrorCode code = net->runSession(session);
            if (NO_ERROR == code) {
                // Research A4 fallback: probe the secondary observable.
                int status = 0;
                net->getSessionInfo(session, Interpreter::RESIZE_STATUS, &status);
                MNN_ERROR("SGFP4ClassicAPIMissingSidecarTest: runSession unexpectedly NO_ERROR "
                          "(RESIZE_STATUS %d) -- expected non-zero\n",
                          status);
                break;
            }
            pass = true;
        } while (false);

        cleanupFixture(fx);
        if (pass) {
            MNN_PRINT("SGFP4ClassicAPIMissingSidecarTest: missing sidecar fails gracefully PASSED\n");
        }
        return pass;
    }
};
MNNTestSuiteRegister(SGFP4ClassicAPIMissingSidecarTest, "op/sgfp4/classic_api_missing_sidecar");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
