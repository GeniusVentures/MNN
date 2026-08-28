//
//  SGFP4MultiTensorTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/08/28.
//  Copyright © 2018, Alibaba Group Holding Limited.
//
// Plan 07-03: prove multi-tensor injection + structured (LAYOUT_MIXED)
// end-to-end coverage and enforce the full malformed-input failure matrix
// with D-11 atomicity assertions (SGINJ-07/SGINJ-08; D-04/D-12: both
// halves live in ONE file, two suites).
//
//   op/sgfp4/multi_tensor      -- 2 containers (structured MIXED fixture +
//                                  in-test uniform) -> single artifact with
//                                  disjoint 16-byte-aligned sidecar ranges,
//                                  byte-identical to sources; classic
//                                  Interpreter/Session run; FP32 parity.
//   op/sgfp4/malformed_inputs  -- probe matrix: every malformed input exits
//                                  non-zero with a diagnostic AND leaves NO
//                                  output files behind (D-11 regression
//                                  against Plan 07-01's failCleanup).
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
#include "SGFP4StructuredFixtures.h"
#include "SGFP4TestUtil.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace {

// ====================================================================
// Geometry (D-05/D-06: chained two-MatMul graph, distinct weight shapes).
// ====================================================================

constexpr int kInputDim     = 512; // 'input' width; w1 is [512,512]
constexpr int kStructDimO   = 512; // structured fixture (w1) dims
constexpr int kStructDimI   = 512;
constexpr int kUniformDimO  = 512; // in-test uniform container (w2) dims
constexpr int kUniformDimI  = 64;

// LCG input generator: identical values feed injected + baseline runs.
constexpr uint32_t kLcgSeed   = 0x9E3779B9u;
constexpr uint32_t kLcgMul    = 1664525u;
constexpr uint32_t kLcgAdd    = 1013904223u;
constexpr float kLcgNormalize = 1.0f / 16777216.0f; // (state >> 8) is 24-bit

constexpr float kParityRelativeTolerance = 1e-4f;

// ====================================================================
// Plan 08-02: the shared filesystem/serialization helpers and the
// generalized REGION-RELATIVE container builder now live in
// SGFP4TestUtil.hpp (namespace sgfp4_test).
// ====================================================================

// Overload with a raw manifest string (malformed probes rewrite the
// manifest wholesale; the caller computes the JSON).
bool writeNicheDirRawManifest(const std::vector<uint8_t>& containerBytes, const std::string& dir,
                              const std::string& containerName, const std::string& manifest) {
    if (!sgfp4_test::makeDir(dir)) {
        return false;
    }
    if (!sgfp4_test::writeBytes(dir + "/" + containerName, containerBytes.data(), containerBytes.size())) {
        return false;
    }
    return sgfp4_test::writeBytes(dir + "/manifest.json", reinterpret_cast<const uint8_t*>(manifest.data()),
                                  manifest.size());
}

std::string manifestJsonFor(const std::vector<uint8_t>& bytes, const std::string& containerName, int dimO, int dimI) {
    const std::string digest = sgfp4::sha256_hex(bytes.data(), bytes.size());
    std::ostringstream oss;
    oss << "{\"fp4_binary\":{\"path\":\"" << containerName << "\",\"sha256\":\"" << digest
        << "\",\"stats\":{\"shape\":[" << dimO << "," << dimI << "]}}}";
    return oss.str();
}

void fillLcgInput(std::vector<float>& input, int count) {
    input.resize(count, 0.0f);
    uint32_t state = kLcgSeed;
    for (int i = 0; i < count; ++i) {
        state    = state * kLcgMul + kLcgAdd;
        input[i] = static_cast<float>(state >> 8) * kLcgNormalize;
    }
}

// Classic-API session runner (Phase 6 runClassicSession, output width
// generalized): named I/O + runSession + host copy-out.
bool runClassicSession(const std::string& modelPath, const std::vector<float>& inputVals, int outputCount,
                       std::vector<float>& output) {
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(modelPath.c_str()), Interpreter::destroy);
    if (nullptr == net) {
        MNN_ERROR("SGFP4MultiTensorTest: createFromFile('%s') returned null\n", modelPath.c_str());
        return false;
    }
    ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_CPU;
    auto session = net->createSession(cfg);
    if (nullptr == session) {
        MNN_ERROR("SGFP4MultiTensorTest: createSession('%s') returned null\n", modelPath.c_str());
        return false;
    }
    auto inputTensor = net->getSessionInput(session, nullptr);
    if (nullptr == inputTensor) {
        MNN_ERROR("SGFP4MultiTensorTest: no session input for '%s'\n", modelPath.c_str());
        return false;
    }
    net->resizeSession(session); // void return; errors surface at runSession
    ::memcpy(inputTensor->host<float>(), inputVals.data(), inputVals.size() * sizeof(float));
    const ErrorCode code = net->runSession(session);
    if (NO_ERROR != code) {
        MNN_ERROR("SGFP4MultiTensorTest: runSession('%s') returned %d\n", modelPath.c_str(), static_cast<int>(code));
        return false;
    }
    auto outputTensor = net->getSessionOutput(session, nullptr);
    if (nullptr == outputTensor) {
        MNN_ERROR("SGFP4MultiTensorTest: no session output for '%s'\n", modelPath.c_str());
        return false;
    }
    std::shared_ptr<Tensor> outUser(new Tensor(outputTensor, Tensor::CAFFE));
    outputTensor->copyToHostTensor(outUser.get());
    const float* got = outUser->host<float>();
    if (nullptr == got) {
        MNN_ERROR("SGFP4MultiTensorTest: output unreadable for '%s'\n", modelPath.c_str());
        return false;
    }
    output.assign(got, got + outputCount);
    return true;
}

// Collect every expr reachable (through inputs) from the given roots
// (Phase 5 walker precedent).
void collectExprs(const EXPRP& expr, std::set<EXPRP>& visited) {
    if (nullptr == expr || 0 != visited.count(expr)) {
        return;
    }
    visited.insert(expr);
    for (const auto& input : expr->inputs()) {
        collectExprs(input->expr().first, visited);
    }
}

struct DequantNodeInfo {
    int64_t offset;
    int64_t size;
    int dimO;
    int dimI;
};

} // namespace

// ====================================================================
// op/sgfp4/multi_tensor -- SGINJ-07 core + structured half of SGINJ-08.
// ====================================================================
class SGFP4MultiTensorTest : public MNNTestCase {
public:
    SGFP4MultiTensorTest()  = default;
    virtual ~SGFP4MultiTensorTest() = default;

    virtual bool run(int precision) {
        // -- Test 3 first (cheapest): MIXED provenance guard. The C++ side
        // cannot re-derive MIXED-ness; kStructuredMixedCount is the frozen
        // authoring-time count (Plan 07-02). Guards fixture-regeneration
        // drift (RESEARCH Pitfall 2).
        if (kStructuredMixedCount <= 0) {
            MNN_ERROR("SGFP4MultiTensorTest: kStructuredMixedCount = %d -- structured fixture has no MIXED "
                      "superblocks; regenerate via tools/fp4/author_structured_fixture.py\n",
                      kStructuredMixedCount);
            return false;
        }
        if (!MNN::sgfp4_is_v2_container(kStructuredMixedData, kStructuredSize)) {
            MNN_ERROR("SGFP4MultiTensorTest: structured fixture failed the v2 version gate\n");
            return false;
        }
        MNN_PRINT("SGFP4MultiTensorTest: structured fixture MIXED superblock count = %d (authoring-time "
                  "provenance)\n",
                  kStructuredMixedCount);

        const std::string cwd = sgfp4_test::cwdPath();
        const std::string basePath  = cwd + "/" + sgfp4_test::tempPath("sgfp4_mt_base_", ".mnn");
        const std::string outPath   = cwd + "/" + sgfp4_test::tempPath("sgfp4_mt_out_", ".mnn");
        const std::string sidecarPath = outPath + ".weight";
        const std::string structDir  = cwd + "/" + sgfp4_test::tempPath("sgfp4_mt_nicheA_", ".d");
        const std::string uniformDir = cwd + "/" + sgfp4_test::tempPath("sgfp4_mt_nicheB_", ".d");

        auto cleanupGuard = [&]() {
            std::remove(basePath.c_str());
            std::remove(outPath.c_str());
            std::remove(sidecarPath.c_str());
            std::remove((structDir + "/struct_fixture.sgfp4").c_str());
            std::remove((structDir + "/manifest.json").c_str());
            sgfp4_test::removeDir(structDir);
            std::remove((uniformDir + "/uniform_fixture.sgfp4").c_str());
            std::remove((uniformDir + "/manifest.json").c_str());
            sgfp4_test::removeDir(uniformDir);
        };

        bool pass = false;
        do {
            // -- 1. Oracle decodes: w1 = structured fixture, w2 = uniform.
            std::vector<float> w1(static_cast<size_t>(kStructDimO) * kStructDimI, 0.0f);
            if (!MNN::dequant_sgfp4_container_cpu(kStructuredMixedData, kStructuredSize, w1.data(), w1.size())) {
                MNN_ERROR("SGFP4MultiTensorTest: oracle decode of structured fixture failed\n");
                break;
            }
            std::vector<uint8_t> uniformBytes;
            if (!sgfp4_test::buildContainerUniform64(kUniformDimO, kUniformDimI, uniformBytes)) {
                MNN_ERROR("SGFP4MultiTensorTest: uniform container builder failed\n");
                break;
            }
            if (!MNN::sgfp4_is_v2_container(uniformBytes.data(), uniformBytes.size())) {
                MNN_ERROR("SGFP4MultiTensorTest: generated uniform container failed the v2 gate\n");
                break;
            }
            std::vector<float> w2(static_cast<size_t>(kUniformDimO) * kUniformDimI, 0.0f);
            if (!MNN::dequant_sgfp4_container_cpu(uniformBytes.data(), uniformBytes.size(), w2.data(), w2.size())) {
                MNN_ERROR("SGFP4MultiTensorTest: oracle decode of uniform container failed\n");
                break;
            }

            // -- 2. Base model (D-05, chained per RESEARCH Q5):
            //    input[1,512] -MatMul(w1[512,512])-> h -MatMul(w2[512,64])-> output
            {
                auto input = _Input({1, kInputDim}, NHWC, halide_type_of<float>());
                auto c1    = _Const(w1.data(), {kStructDimO, kStructDimI}, NHWC, halide_type_of<float>());
                c1->setName("weight1");
                auto h = _MatMul(input, c1);
                auto c2 = _Const(w2.data(), {kUniformDimO, kUniformDimI}, NHWC, halide_type_of<float>());
                c2->setName("weight2");
                auto out = _MatMul(h, c2);
                input->setName("input");
                out->setName("output");
                Variable::save({out}, basePath.c_str());
            }

            // -- 3. Two niche dirs (D-06): A = structured fixture bytes with
            //    real-encoder provenance; B = in-test uniform container.
            const std::vector<uint8_t> structBytes(kStructuredMixedData,
                                                   kStructuredMixedData + kStructuredSize);
            if (!sgfp4_test::writeNicheDir(structBytes, structDir, "struct_fixture.sgfp4", kStructDimO, kStructDimI)) {
                MNN_ERROR("SGFP4MultiTensorTest: failed to write structured niche dir '%s'\n", structDir.c_str());
                break;
            }
            if (!sgfp4_test::writeNicheDir(uniformBytes, uniformDir, "uniform_fixture.sgfp4", kUniformDimO,
                                            kUniformDimI)) {
                MNN_ERROR("SGFP4MultiTensorTest: failed to write uniform niche dir '%s'\n", uniformDir.c_str());
                break;
            }

            // -- 4. In-process injection, two --niche-dir pairs, argc = 9.
            const char* argv[] = {"sgfp4_inject",                 // 0
                                  "--model",     basePath.c_str(), // 1..2
                                  "--niche-dir", structDir.c_str(),// 3..4
                                  "--niche-dir", uniformDir.c_str(),// 5..6
                                  "--output",    outPath.c_str()};// 7..8
            if (0 != sgfp4_inject::run(9, argv)) {
                MNN_ERROR("SGFP4MultiTensorTest: sgfp4_inject::run(9, ...) failed\n");
                break;
            }

            // -- 5. Structure + collision assertions (SGINJ-07 core):
            //    2 dequant ops, disjoint 16-aligned ranges, dims as
            //    injected, sidecar bytes equal to the source containers.
            std::vector<DequantNodeInfo> nodes;
            {
                auto reloaded = Variable::loadMap(outPath.c_str());
                if (reloaded.empty()) {
                    MNN_ERROR("SGFP4MultiTensorTest: loadMap of injected artifact returned an empty map\n");
                    break;
                }
                std::set<EXPRP> exprs;
                for (const auto& nameVar : reloaded) {
                    collectExprs(nameVar.second->expr().first, exprs);
                }
                for (const auto& expr : exprs) {
                    if (nullptr == expr->get() || MNN::OpType_SGFP4Dequant != expr->get()->type()) {
                        continue;
                    }
                    const auto* param = expr->get()->main_as_SGFP4DequantParam();
                    if (nullptr == param || nullptr == param->external() || nullptr == param->dims() ||
                        param->external()->size() != 2 || param->dims()->size() != 2) {
                        MNN_ERROR("SGFP4MultiTensorTest: dequant op carries malformed SGFP4DequantParam\n");
                        break;
                    }
                    DequantNodeInfo info;
                    info.offset = param->external()->Get(0);
                    info.size   = param->external()->Get(1);
                    info.dimO   = param->dims()->Get(0);
                    info.dimI   = param->dims()->Get(1);
                    nodes.push_back(info);
                }
                if (2 != nodes.size()) {
                    MNN_ERROR("SGFP4MultiTensorTest: expected exactly 2 SGFP4Dequant ops, found %zu\n", nodes.size());
                    break;
                }
                bool sawStruct = false;
                bool sawUniform = false;
                bool structFail = false;
                for (const auto& n : nodes) {
                    if (0 != n.offset % 16) {
                        MNN_ERROR("SGFP4MultiTensorTest: offset %lld not 16-byte aligned\n",
                                  static_cast<long long>(n.offset));
                        structFail = true;
                        break;
                    }
                    if (n.dimO == kStructDimO && n.dimI == kStructDimI) {
                        sawStruct = true;
                        if (static_cast<size_t>(n.size) != kStructuredSize) {
                            MNN_ERROR("SGFP4MultiTensorTest: structured node size %lld != fixture %zu\n",
                                      static_cast<long long>(n.size), kStructuredSize);
                            structFail = true;
                            break;
                        }
                    } else if (n.dimO == kUniformDimO && n.dimI == kUniformDimI) {
                        sawUniform = true;
                        if (static_cast<size_t>(n.size) != uniformBytes.size()) {
                            MNN_ERROR("SGFP4MultiTensorTest: uniform node size %lld != container %zu\n",
                                      static_cast<long long>(n.size), uniformBytes.size());
                            structFail = true;
                            break;
                        }
                    } else {
                        MNN_ERROR("SGFP4MultiTensorTest: unexpected dims {%d,%d} on dequant op\n", n.dimO, n.dimI);
                        structFail = true;
                        break;
                    }
                }
                if (structFail) {
                    break;
                }
                // disjoint-range check (pairwise)
                const bool overlap = (nodes[0].offset < nodes[1].offset + nodes[1].size) &&
                                     (nodes[1].offset < nodes[0].offset + nodes[0].size);
                if (overlap) {
                    MNN_ERROR("SGFP4MultiTensorTest: sidecar ranges overlap: [%lld,%lld) vs [%lld,%lld)\n",
                              static_cast<long long>(nodes[0].offset),
                              static_cast<long long>(nodes[0].offset + nodes[0].size),
                              static_cast<long long>(nodes[1].offset),
                              static_cast<long long>(nodes[1].offset + nodes[1].size));
                    break;
                }
                if (!sawStruct || !sawUniform) {
                    MNN_ERROR("SGFP4MultiTensorTest: expected one {%d,%d} and one {%d,%d} node (saw struct=%d "
                              "uniform=%d)\n",
                              kStructDimO, kStructDimI, kUniformDimO, kUniformDimI, sawStruct, sawUniform);
                    break;
                }

                // byte-identity: sidecar range == source container bytes
                // (dims matched as PAIRS -- both nodes have dimO == 512, so
                // dimI is the discriminator)
                std::vector<uint8_t> sidecar;
                if (!sgfp4_test::readBytes(sidecarPath, sidecar)) {
                    MNN_ERROR("SGFP4MultiTensorTest: cannot read sidecar '%s'\n", sidecarPath.c_str());
                    break;
                }
                bool identityFail = false;
                for (const auto& n : nodes) {
                    const uint8_t* src = nullptr;
                    size_t srcSize    = 0;
                    if (n.dimO == kStructDimO && n.dimI == kStructDimI) {
                        src     = kStructuredMixedData;
                        srcSize = kStructuredSize;
                    } else {
                        src     = uniformBytes.data();
                        srcSize = uniformBytes.size();
                    }
                    if (n.offset < 0 || static_cast<size_t>(n.offset) + n.size > sidecar.size()) {
                        MNN_ERROR("SGFP4MultiTensorTest: sidecar range [%lld,%lld) out of bounds (sidecar %zu)\n",
                                  static_cast<long long>(n.offset), static_cast<long long>(n.offset + n.size),
                                  sidecar.size());
                        identityFail = true;
                        break;
                    }
                    if (0 != std::memcmp(sidecar.data() + n.offset, src, srcSize)) {
                        MNN_ERROR("SGFP4MultiTensorTest: sidecar bytes at offset %lld differ from source "
                                  "container\n",
                                  static_cast<long long>(n.offset));
                        identityFail = true;
                        break;
                    }
                }
                if (identityFail) {
                    break;
                }
            }
            if (2 != nodes.size()) {
                break; // inner break paths above already logged
            }

            // -- 6. Classic-API run of the injected artifact (D-07):
            //    named I/O 'input'/'output', [1,64] output, FP32 parity vs
            //    the pre-injection base model (weights already == oracle
            //    decodes; zero-by-construction parity, Phase 6 D-06
            //    extended to two weights).
            std::vector<float> inputVals;
            fillLcgInput(inputVals, kInputDim);
            std::vector<float> injected;
            {
                std::shared_ptr<Interpreter> net(Interpreter::createFromFile(outPath.c_str()), Interpreter::destroy);
                if (nullptr == net) {
                    MNN_ERROR("SGFP4MultiTensorTest: createFromFile of injected artifact returned null\n");
                    break;
                }
                ScheduleConfig cfg;
                cfg.type = MNN_FORWARD_CPU;
                auto session = net->createSession(cfg);
                if (nullptr == session) {
                    MNN_ERROR("SGFP4MultiTensorTest: createSession of injected artifact returned null\n");
                    break;
                }
                const auto& inAll  = net->getSessionInputAll(session);
                const auto& outAll = net->getSessionOutputAll(session);
                if (1 != inAll.count("input") || 1 != outAll.count("output")) {
                    MNN_ERROR("SGFP4MultiTensorTest: named I/O mismatch (inputs %zu with 'input'=%d, outputs %zu "
                              "with 'output'=%d)\n",
                              inAll.size(), static_cast<int>(inAll.count("input")), outAll.size(),
                              static_cast<int>(outAll.count("output")));
                    break;
                }
                auto inputTensor = net->getSessionInput(session, nullptr);
                net->resizeSession(session);
                ::memcpy(inputTensor->host<float>(), inputVals.data(), kInputDim * sizeof(float));
                const ErrorCode code = net->runSession(session);
                if (NO_ERROR != code) {
                    MNN_ERROR("SGFP4MultiTensorTest: runSession of injected artifact returned %d\n",
                              static_cast<int>(code));
                    break;
                }
                auto outputTensor = net->getSessionOutput(session, nullptr);
                std::shared_ptr<Tensor> outUser(new Tensor(outputTensor, Tensor::CAFFE));
                outputTensor->copyToHostTensor(outUser.get());
                const float* got = outUser->host<float>();
                if (nullptr == got) {
                    MNN_ERROR("SGFP4MultiTensorTest: injected output unreadable\n");
                    break;
                }
                injected.assign(got, got + kUniformDimI);
            }
            std::vector<float> baseline;
            if (!runClassicSession(basePath, inputVals, kUniformDimI, baseline)) {
                break;
            }
            if (!checkVectorByRelativeError<float>(injected.data(), baseline.data(), kUniformDimI,
                                                   kParityRelativeTolerance)) {
                MNN_ERROR("SGFP4MultiTensorTest: injected output != FP32 baseline within rtol %g\n",
                          kParityRelativeTolerance);
                break;
            }
            pass = true;
        } while (false);

        cleanupGuard();
        if (pass) {
            MNN_PRINT("SGFP4MultiTensorTest: 2-container disjoint ranges + byte identity + classic run parity "
                      "PASSED\n");
        }
        return pass;
    }
};
MNNTestSuiteRegister(SGFP4MultiTensorTest, "op/sgfp4/multi_tensor");

// ====================================================================
// op/sgfp4/malformed_inputs -- D-09/D-10/D-11 failure matrix. Each probe:
// fresh temp paths (Pitfall 6), expect run() != 0, then assert BOTH output
// files are absent (the D-11 atomicity regression against Plan 07-01's
// failCleanup). Everything runs in-process; no crash is tolerated.
// ====================================================================
class SGFP4MalformedInputsTest : public MNNTestCase {
public:
    SGFP4MalformedInputsTest()  = default;
    virtual ~SGFP4MalformedInputsTest() = default;

    virtual bool run(int precision) {
        // Probe table (RESEARCH Q4). Each entry rebuilds its niche dir from
        // the shared pristine uniform container + manifest, applies a
        // mutation, runs a SINGLE-niche injection against the 2-weight base
        // model (probe 7 builds its own two-same-shape model), and asserts
        // fail + no-partial-output.
        enum ProbeKind {
            kEmptyContainer,        // 1: 0-byte file
            kTruncated,             // 2: first 15 bytes
            kBadSha,                // 3: one hex digit flipped
            kBadMagic,              // 4a: bytes[0] ^= 0xFF (sha recomputed)
            kVersionOne,            // 4b: version byte 0x01 (sha recomputed)
            kMissingSha,            // 5: omit sha256
            kMissingPath,           // 5: omit path
            kMissingShape,          // 5: omit stats.shape
            kRankThreeShape,        // 5: shape rank 3
            kNonPositiveDim,        // 5: shape dim 0
            kZeroMatch,             // 6: shape [256,256] matches nothing
            kMultiMatch,            // 7: two [512,512] weights, one niche
            kGarbageBody,           // 8: payload byte flipped, sha recomputed
            kProbeCount
        };
        static const char* kProbeNames[kProbeCount] = {
            "empty_container",  "truncated",     "bad_sha",        "bad_magic",       "version_one",
            "missing_sha",      "missing_path",  "missing_shape",  "rank3_shape",     "nonpositive_dim",
            "zero_match",       "multi_match",   "garbage_body",
        };

        const std::string cwd = sgfp4_test::cwdPath();

        // Shared pristine uniform container ([512,64]) + oracle weights for
        // the base model.
        std::vector<uint8_t> pristine;
        if (!sgfp4_test::buildContainerUniform64(kUniformDimO, kUniformDimI, pristine)) {
            MNN_ERROR("SGFP4MalformedInputsTest: uniform container builder failed\n");
            return false;
        }
        std::vector<float> w2(static_cast<size_t>(kUniformDimO) * kUniformDimI, 0.0f);
        if (!MNN::dequant_sgfp4_container_cpu(pristine.data(), pristine.size(), w2.data(), w2.size())) {
            MNN_ERROR("SGFP4MalformedInputsTest: oracle decode of pristine container failed\n");
            return false;
        }
        // Structured fixture decode for the base model's w1 (and the probe-7
        // second weight source is a second structured-like decode reuse).
        std::vector<float> w1(static_cast<size_t>(kStructDimO) * kStructDimI, 0.0f);
        if (!MNN::dequant_sgfp4_container_cpu(kStructuredMixedData, kStructuredSize, w1.data(), w1.size())) {
            MNN_ERROR("SGFP4MalformedInputsTest: oracle decode of structured fixture failed\n");
            return false;
        }

        // 2-weight base model (same topology as the multi_tensor suite; the
        // uniform-shaped weight is the pairing target except probe 7).
        const std::string basePath = cwd + "/" + sgfp4_test::tempPath("sgfp4_mi_base_", ".mnn");
        {
            auto input = _Input({1, kInputDim}, NHWC, halide_type_of<float>());
            auto c1    = _Const(w1.data(), {kStructDimO, kStructDimI}, NHWC, halide_type_of<float>());
            c1->setName("weight1");
            auto h = _MatMul(input, c1);
            auto c2 = _Const(w2.data(), {kUniformDimO, kUniformDimI}, NHWC, halide_type_of<float>());
            c2->setName("weight2");
            auto out = _MatMul(h, c2);
            input->setName("input");
            out->setName("output");
            Variable::save({out}, basePath.c_str());
        }

        // Probe-7 model: TWO [512,512] weights -> one [512,512] niche must
        // hard-fail with "found 2" (D-08 behavior lock).
        const std::string basePath2 = cwd + "/" + sgfp4_test::tempPath("sgfp4_mi_base2_", ".mnn");
        {
            auto input = _Input({1, kInputDim}, NHWC, halide_type_of<float>());
            auto c1    = _Const(w1.data(), {kStructDimO, kStructDimI}, NHWC, halide_type_of<float>());
            c1->setName("weight1");
            auto h1 = _MatMul(input, c1);
            auto c2 = _Const(w1.data(), {kStructDimO, kStructDimI}, NHWC, halide_type_of<float>());
            c2->setName("weight2");
            auto h2 = _MatMul(input, c2);
            auto out = _Add(h1, h2);
            input->setName("input");
            out->setName("output");
            Variable::save({out}, basePath2.c_str());
        }

        auto cleanupShared = [&]() {
            std::remove(basePath.c_str());
            std::remove(basePath2.c_str());
        };

        bool allPass   = true;
        bool anyRan    = false;

        for (int probe = 0; probe < kProbeCount; ++probe) {
            const std::string outPath   = cwd + "/" + sgfp4_test::tempPath("sgfp4_mi_out_", ".mnn");
            const std::string sidecarPath = outPath + ".weight";
            const std::string nicheDir  = cwd + "/" + sgfp4_test::tempPath("sgfp4_mi_niche_", ".d");
            const std::string containerName = "probe_fixture.sgfp4";
            auto cleanupProbe = [&](void) {
                std::remove(outPath.c_str());
                std::remove(sidecarPath.c_str());
                std::remove((nicheDir + "/" + containerName).c_str());
                std::remove((nicheDir + "/manifest.json").c_str());
                sgfp4_test::removeDir(nicheDir);
            };

            bool setupOk = true;
            do {
                // -- Build the probe's container + manifest ----------------
                std::vector<uint8_t> bytes;
                std::string manifest;

                switch (probe) {
                    case kEmptyContainer: {
                        bytes.clear(); // 0-byte container; sha over empty
                        const std::string digest = sgfp4::sha256_hex(bytes.data(), bytes.size());
                        std::ostringstream oss;
                        oss << "{\"fp4_binary\":{\"path\":\"" << containerName << "\",\"sha256\":\"" << digest
                            << "\",\"stats\":{\"shape\":[" << kUniformDimO << "," << kUniformDimI << "]}}}";
                        manifest = oss.str();
                        break;
                    }
                    case kTruncated: {
                        bytes.assign(pristine.begin(), pristine.begin() + 15); // header cut short
                        const std::string digest = sgfp4::sha256_hex(bytes.data(), bytes.size());
                        std::ostringstream oss;
                        oss << "{\"fp4_binary\":{\"path\":\"" << containerName << "\",\"sha256\":\"" << digest
                            << "\",\"stats\":{\"shape\":[" << kUniformDimO << "," << kUniformDimI << "]}}}";
                        manifest = oss.str();
                        break;
                    }
                    case kBadSha: {
                        bytes = pristine;
                        manifest = manifestJsonFor(bytes, containerName, kUniformDimO, kUniformDimI);
                        // Flip one hex digit of the (valid) sha.
                        const size_t firstQuote = manifest.find("\"sha256\":\"") + 10;
                        manifest[firstQuote] = (manifest[firstQuote] == '0') ? '1' : '0';
                        break;
                    }
                    case kBadMagic: {
                        bytes = pristine;
                        bytes[0] ^= 0xFF; // destroy the magic, keep framing otherwise
                        const std::string digest = sgfp4::sha256_hex(bytes.data(), bytes.size());
                        std::ostringstream oss;
                        oss << "{\"fp4_binary\":{\"path\":\"" << containerName << "\",\"sha256\":\"" << digest
                            << "\",\"stats\":{\"shape\":[" << kUniformDimO << "," << kUniformDimI << "]}}}";
                        manifest = oss.str();
                        break;
                    }
                    case kVersionOne: {
                        bytes = pristine;
                        bytes[MNN::kSGFP4VersionByteOffset] = 0x01; // valid magic, v1 version byte
                        const std::string digest = sgfp4::sha256_hex(bytes.data(), bytes.size());
                        std::ostringstream oss;
                        oss << "{\"fp4_binary\":{\"path\":\"" << containerName << "\",\"sha256\":\"" << digest
                            << "\",\"stats\":{\"shape\":[" << kUniformDimO << "," << kUniformDimI << "]}}}";
                        manifest = oss.str();
                        break;
                    }
                    case kMissingSha: {
                        bytes = pristine;
                        const std::string digest = sgfp4::sha256_hex(bytes.data(), bytes.size());
                        std::ostringstream oss;
                        oss << "{\"fp4_binary\":{\"path\":\"" << containerName
                            << "\",\"stats\":{\"shape\":[" << kUniformDimO << "," << kUniformDimI << "]}}}";
                        manifest = oss.str();
                        break;
                    }
                    case kMissingPath: {
                        bytes = pristine;
                        const std::string digest = sgfp4::sha256_hex(bytes.data(), bytes.size());
                        std::ostringstream oss;
                        oss << "{\"fp4_binary\":{\"sha256\":\"" << digest << "\",\"stats\":{\"shape\":["
                            << kUniformDimO << "," << kUniformDimI << "]}}}";
                        manifest = oss.str();
                        break;
                    }
                    case kMissingShape: {
                        bytes = pristine;
                        const std::string digest = sgfp4::sha256_hex(bytes.data(), bytes.size());
                        std::ostringstream oss;
                        oss << "{\"fp4_binary\":{\"path\":\"" << containerName << "\",\"sha256\":\"" << digest
                            << "\"}}";
                        manifest = oss.str();
                        break;
                    }
                    case kRankThreeShape: {
                        bytes = pristine;
                        const std::string digest = sgfp4::sha256_hex(bytes.data(), bytes.size());
                        std::ostringstream oss;
                        oss << "{\"fp4_binary\":{\"path\":\"" << containerName << "\",\"sha256\":\"" << digest
                            << "\",\"stats\":{\"shape\":[" << kUniformDimO << "," << kUniformDimI << ",3]}}}";
                        manifest = oss.str();
                        break;
                    }
                    case kNonPositiveDim: {
                        bytes = pristine;
                        const std::string digest = sgfp4::sha256_hex(bytes.data(), bytes.size());
                        std::ostringstream oss;
                        oss << "{\"fp4_binary\":{\"path\":\"" << containerName << "\",\"sha256\":\"" << digest
                            << "\",\"stats\":{\"shape\":[" << kUniformDimO << ",0]}}}";
                        manifest = oss.str();
                        break;
                    }
                    case kZeroMatch: {
                        bytes = pristine;
                        // [256,256] matches no weight in the 2-weight model
                        manifest = manifestJsonFor(bytes, containerName, 256, 256);
                        break;
                    }
                    case kMultiMatch: {
                        bytes = pristine;
                        // Rebuild bytes at [512,512]? No: use the STRUCTURED
                        // fixture (already 512x512, valid v2). One [512,512]
                        // niche vs the two [512,512] weights in basePath2.
                        const std::vector<uint8_t> sb(kStructuredMixedData,
                                                       kStructuredMixedData + kStructuredSize);
                        bytes = sb;
                        manifest = manifestJsonFor(bytes, containerName, kStructDimO, kStructDimI);
                        break;
                    }
                    case kGarbageBody: {
                        bytes = pristine;
                        // Garbage INSIDE the framing contract (D-10): keep the
                        // fixed header (magic/version) and offset table intact
                        // so the byte-level v2 gate and sha256 pass, but write
                        // an INVALID layout enum (7 > kSGFP4LayoutEnumCount-1)
                        // into record 0's sb_header. The injected op's decode
                        // (and the tool's in-tool verify decode) then fails
                        // structurally -> run exits non-zero -> failCleanup
                        // must remove the freshly written outputs.
                        // (A garbage nibble-PAYLOAD byte is also valid D-10
                        // input, but the affine decode is total over payload
                        // nibbles -- garbage payloads decode to different
                        // values rather than failing, so a payload-byte-only
                        // flip is a structural SUCCESS by design. The
                        // framing-corruption variant below is the probe that
                        // reaches the D-11 cleanup path.)
                        const size_t recordRegionStart = MNN::sgfp4_align16(
                            MNN::kSGFP4RecordOffsetTableStart +
                            8 * MNN::kSGFP4RecordOffsetEntrySize); // 8 records (512x64)
                        const size_t sbHeader0 = recordRegionStart;
                        const uint32_t garbage = (sgfp4_read_u32_le(bytes.data() + sbHeader0) &
                                                  ~MNN::kSGFP4LayoutEnumMask) |
                                                 0x7u;
                        sgfp4_test::writeU32Le(bytes, sbHeader0, garbage);
                        const std::string digest = sgfp4::sha256_hex(bytes.data(), bytes.size());
                        std::ostringstream oss;
                        oss << "{\"fp4_binary\":{\"path\":\"" << containerName << "\",\"sha256\":\"" << digest
                            << "\",\"stats\":{\"shape\":[" << kUniformDimO << "," << kUniformDimI << "]}}}";
                        manifest = oss.str();
                        break;
                    }
                    default:
                        setupOk = false;
                        break;
                }
                if (!setupOk) {
                    break;
                }

                // Seed a stale artifact so the no-partial-output assertion
                // also proves stale removal (D-11 semantics: a failed run
                // removes ANY files at the output paths).
                if (!sgfp4_test::writeBytes(outPath, reinterpret_cast<const uint8_t*>("stale"), 5) ||
                    !sgfp4_test::writeBytes(sidecarPath, reinterpret_cast<const uint8_t*>("stale"), 5)) {
                    MNN_ERROR("SGFP4MalformedInputsTest[%d/%s]: failed to seed stale artifacts\n", probe,
                              kProbeNames[probe]);
                    setupOk = false;
                    break;
                }

                if (!writeNicheDirRawManifest(bytes, nicheDir, containerName, manifest)) {
                    MNN_ERROR("SGFP4MalformedInputsTest[%d/%s]: failed to write niche dir\n", probe,
                              kProbeNames[probe]);
                    setupOk = false;
                    break;
                }
            } while (false);

            if (!setupOk) {
                cleanupProbe();
                allPass = false;
                continue;
            }

            // -- Run: single-niche injection against the appropriate model.
            const std::string& model = (probe == kMultiMatch) ? basePath2 : basePath;
            const char* argv[] = {"sgfp4_inject",                  // 0
                                  "--model",     model.c_str(),    // 1..2
                                  "--niche-dir", nicheDir.c_str(), // 3..4
                                  "--output",    outPath.c_str()};// 5..6
            const int rc = sgfp4_inject::run(7, argv);
            anyRan = true;

            // -- Assertions: fail + NO output files (D-11).
            bool probePass = (0 != rc);
            if (probePass) {
                if (sgfp4_test::fileExists(outPath) || sgfp4_test::fileExists(sidecarPath)) {
                    MNN_ERROR("SGFP4MalformedInputsTest[%d/%s]: run failed cleanly (rc=%d) BUT output files "
                              "remain (out=%d sidecar=%d) -- D-11 atomicity violated\n",
                              probe, kProbeNames[probe], rc, static_cast<int>(sgfp4_test::fileExists(outPath)),
                              static_cast<int>(sgfp4_test::fileExists(sidecarPath)));
                    probePass = false;
                }
            } else {
                MNN_ERROR("SGFP4MalformedInputsTest[%d/%s]: run unexpectedly returned 0\n", probe,
                          kProbeNames[probe]);
            }
            if (probePass) {
                MNN_PRINT("SGFP4MalformedInputsTest[%d/%s]: failed cleanly, no partial output PASSED\n", probe,
                          kProbeNames[probe]);
            }
            if (!probePass) {
                allPass = false;
            }

            cleanupProbe();
        }

        cleanupShared();
        if (!anyRan) {
            MNN_ERROR("SGFP4MalformedInputsTest: no probes executed\n");
            return false;
        }
        // Note (anti-case, RESEARCH Q4 note 6): a "dims disagreement" probe
        // is UNREACHABLE by construction -- the pairing key and the op's
        // dims both come from the same manifest stats.shape field (D-14), so
        // any disagreement collapses into the zero-match probe above.
        if (allPass) {
            MNN_PRINT("SGFP4MalformedInputsTest: all %d probes failed cleanly with no partial artifacts PASSED\n",
                      static_cast<int>(kProbeCount));
        }
        return allPass;
    }
};
MNNTestSuiteRegister(SGFP4MalformedInputsTest, "op/sgfp4/malformed_inputs");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
