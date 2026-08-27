//
//  sgfp4_inject_core.hpp
//  MNN SGFP4 v2 injection tool -- shared core (header-only)
//
//  Created by MNN on 2026/08/27.
//  Copyright © 2018, Alibaba Group Holding Limited.
//
// Plan 06-01: the entire Phase 5 injection tool core (everything except
// main()) moved verbatim from tools/fp4/sgfp4_inject.cpp into this
// header-only core under namespace sgfp4_inject, so the Phase 6 test can
// drive the injection in-process via sgfp4_inject::run(argc, argv) with no
// subprocess and no re-implementation (decision D-12). Zero behavior
// change: the Phase 5 CLI worker was renamed to run and every free
// function marked inline; tools/fp4/sgfp4_inject.cpp is now a thin shim.
//
// Given a normally-converted .mnn plus one or more gnus-poc
// fp4_exporter.py --adaptive output directories, produce a new .mnn +
// merged external sidecar where each target weight tensor is produced by
// an OpType_SGFP4Dequant node. The graph-surgery recipe is the one proven
// by Plan 05-01 (test/op/SGFP4InjectTest.cpp).
//
// CLI: sgfp4_inject --model <path> --niche-dir <dir> [--niche-dir <dir>...]
//                   --output <path>
// Sidecar: <output>.weight
//
#ifndef TOOLS_FP4_SGFP4_INJECT_CORE_HPP
#define TOOLS_FP4_SGFP4_INJECT_CORE_HPP

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "MNN/MNNDefine.h"
#include "MNN_generated.h"
#include "MNN/SGFP4DequantUtils.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/expr/Executor.hpp"
#include "MNN/expr/Module.hpp"
#include "rapidjson/document.h"
#include "sha256.hpp"

namespace sgfp4_inject {

using namespace MNN::Express;

// Decode-vs-decode tolerance (D-13): deterministically identical decodes,
// this only absorbs float32 arithmetic noise.
constexpr float kOracleRelativeTolerance = 1e-4f;

// manifest fp4_binary.stats.shape must be exactly this long (D-05).
constexpr rapidjson::SizeType kManifestShapeRank = 2;

struct InjectedNode {
    std::string weightName;
    VARP weightVar;             // original weight VARP (live node after replace)
    std::shared_ptr<MNN::OpT> op;
    std::vector<uint8_t> containerBytes;
    int dimO = 0;
    int dimI = 0;
    size_t sidecarOffset = 0;
    size_t sidecarSize    = 0;
};

struct NicheDir {
    std::string dir;
    std::string containerPath;
    std::string containerBase;
    std::string manifestPathSha; // manifest fp4_binary.sha256
    std::string manifestPathBase; // basename of manifest fp4_binary.path (cross-check only)
    std::vector<uint8_t> bytes;
    int dimO = 0;
    int dimI = 0;
};

inline std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Basename splitting on both separators (no <filesystem>; C++11 default).
inline std::string basenameOf(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return (slash == std::string::npos) ? path : path.substr(slash + 1);
}

inline bool readFileBytes(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        return false;
    }
    const std::streamsize size = ifs.tellg();
    if (size < 0) {
        return false;
    }
    ifs.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    if (size > 0) {
        ifs.read(reinterpret_cast<char*>(out.data()), size);
    }
    return static_cast<std::streamsize>(ifs.gcount()) == size;
}

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX // windows.h min/max macros clash with std::max/std::min
#endif
#include <windows.h>
// Directory listing via FindFirstFile (portable-enough for this tool; no
// <filesystem> under the C++11 default).
inline std::vector<std::string> listDirEntries(const std::string& dir) {
    std::vector<std::string> names;
    const std::string pattern = dir + "\\*";
    WIN32_FIND_DATAA data;
    HANDLE handle = FindFirstFileA(pattern.c_str(), &data);
    if (INVALID_HANDLE_VALUE == handle) {
        return names;
    }
    do {
        const std::string name = data.cFileName;
        if (name != "." && name != ".." && !(data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            names.push_back(name);
        }
    } while (FindNextFileA(handle, &data));
    FindClose(handle);
    return names;
}
#else
#include <dirent.h>
inline std::vector<std::string> listDirEntries(const std::string& dir) {
    std::vector<std::string> names;
    DIR* d = opendir(dir.c_str());
    if (nullptr == d) {
        return names;
    }
    while (auto* entry = readdir(d)) {
        const std::string name = entry->d_name;
        if (name != "." && name != ".." && entry->d_type != DT_DIR) {
            names.push_back(name);
        }
    }
    closedir(d);
    return names;
}
#endif

inline void usage() {
    MNN_PRINT("Usage: sgfp4_inject --model <path> --niche-dir <dir> [--niche-dir <dir>...] --output <path>\n");
    MNN_PRINT("  Each --niche-dir is an unmodified fp4_exporter.py --adaptive output dir\n");
    MNN_PRINT("  (manifest.json + <niche>.sgfp4). Emits <output> plus <output>.weight.\n");
}

// Per --niche-dir validation (D-01, D-03, D-05, Pitfall 3, T-05-01..03).
inline bool loadNicheDir(const std::string& dir, NicheDir& niche) {
    niche.dir = dir;

    // 1. Discover the unique *.sgfp4 file in the dir (case-insensitive).
    std::vector<std::string> matches;
    for (const auto& name : listDirEntries(dir)) {
        const std::string lower = toLower(name);
        if (lower.size() > 6 && lower.substr(lower.size() - 6) == ".sgfp4") {
            matches.push_back(name);
        }
    }
    if (matches.size() != 1) {
        MNN_ERROR("sgfp4_inject: niche dir '%s' must contain exactly one *.sgfp4 file, found %zu\n",
                  dir.c_str(), matches.size());
        return false;
    }
    niche.containerBase = matches[0];
    niche.containerPath = dir + "/" + matches[0];

    // 2. Parse manifest.json (rapidjson DOM).
    std::vector<uint8_t> manifestBytes;
    const std::string manifestPath = dir + "/manifest.json";
    if (!readFileBytes(manifestPath, manifestBytes)) {
        MNN_ERROR("sgfp4_inject: cannot read '%s'\n", manifestPath.c_str());
        return false;
    }
    rapidjson::Document doc;
    doc.Parse(reinterpret_cast<const char*>(manifestBytes.data()), manifestBytes.size());
    if (doc.HasParseError() || !doc.IsObject()) {
        MNN_ERROR("sgfp4_inject: '%s' is not valid JSON\n", manifestPath.c_str());
        return false;
    }
    const char* kMissing = "sgfp4_inject: manifest '%s' missing field fp4_binary.%s\n";
    if (!doc.HasMember("fp4_binary") || !doc["fp4_binary"].IsObject()) {
        MNN_ERROR(kMissing, manifestPath.c_str(), "(object)");
        return false;
    }
    const auto& fp4 = doc["fp4_binary"];

    // fp4_binary.sha256 (D-03).
    if (!fp4.HasMember("sha256") || !fp4["sha256"].IsString()) {
        MNN_ERROR(kMissing, manifestPath.c_str(), "sha256");
        return false;
    }
    niche.manifestPathSha = fp4["sha256"].GetString();

    // fp4_binary.path -- basename only, cross-validated against the
    // discovered container (never resolved literally; root-relative with
    // backslashes, T-05-03).
    if (!fp4.HasMember("path") || !fp4["path"].IsString()) {
        MNN_ERROR(kMissing, manifestPath.c_str(), "path");
        return false;
    }
    niche.manifestPathBase = basenameOf(fp4["path"].GetString());

    // fp4_binary.stats.shape (D-05): exactly 2 positive ints.
    if (!fp4.HasMember("stats") || !fp4["stats"].IsObject() || !fp4["stats"].HasMember("shape") ||
        !fp4["stats"]["shape"].IsArray()) {
        MNN_ERROR(kMissing, manifestPath.c_str(), "stats.shape");
        return false;
    }
    const auto& shape = fp4["stats"]["shape"];
    if (shape.Size() != kManifestShapeRank || !shape[0].IsInt() || !shape[1].IsInt() || shape[0].GetInt() <= 0 ||
        shape[1].GetInt() <= 0) {
        MNN_ERROR("sgfp4_inject: manifest '%s' fp4_binary.stats.shape must be 2 positive ints\n", manifestPath.c_str());
        return false;
    }
    niche.dimO = shape[0].GetInt();
    niche.dimI = shape[1].GetInt();

    // 3. Cross-check container basename vs manifest path basename (Pitfall 3).
    if (toLower(niche.containerBase) != toLower(niche.manifestPathBase)) {
        MNN_ERROR("sgfp4_inject: niche dir '%s': discovered container '%s' != manifest fp4_binary.path basename '%s'\n",
                  dir.c_str(), niche.containerBase.c_str(), niche.manifestPathBase.c_str());
        return false;
    }

    // 4. Read container bytes; sha256 integrity vs manifest (D-03).
    if (!readFileBytes(niche.containerPath, niche.bytes)) {
        MNN_ERROR("sgfp4_inject: cannot read container '%s'\n", niche.containerPath.c_str());
        return false;
    }
    const std::string digest = sgfp4::sha256_hex(niche.bytes.data(), niche.bytes.size());
    if (toLower(digest) != toLower(niche.manifestPathSha)) {
        MNN_ERROR("sgfp4_inject: container '%s' sha256 mismatch: computed %s != manifest %s\n",
                  niche.containerPath.c_str(), digest.c_str(), niche.manifestPathSha.c_str());
        return false;
    }

    // 5. Byte-level version gate (SGINJ-01) -- never consult fp4_binary.format.
    if (!MNN::sgfp4_is_v2_container(niche.bytes.data(), niche.bytes.size())) {
        MNN_ERROR("sgfp4_inject: container '%s' failed the SGFP4 v2 version gate "
                  "(magic/version bytes: v1-layout or malformed container rejected)\n",
                  niche.containerPath.c_str());
        return false;
    }
    return true;
}

// Build the SGFP4Dequant OpT (Plan 05-01 recipe; externalPath literal, Pitfall 2).
inline std::shared_ptr<MNN::OpT> makeDequantOp(const std::string& sidecarPath, size_t offset, size_t size, int dimO,
                                               int dimI) {
    std::shared_ptr<MNN::OpT> op(new MNN::OpT);
    op->type      = MNN::OpType_SGFP4Dequant;
    op->main.type = MNN::OpParameter_SGFP4DequantParam;
    auto* param   = new MNN::SGFP4DequantParamT;
    param->magic   = MNN::kSGFP4Magic;
    param->external = {static_cast<int64_t>(offset), static_cast<int64_t>(size)};
    param->dims     = {dimO, dimI};
    op->main.value  = param;
    op->externalPath = sidecarPath;
    return op;
}

inline int run(int argc, const char* argv[]) {
    std::string modelPath;
    std::string outputPath;
    std::vector<std::string> nicheDirs;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--niche-dir" && i + 1 < argc) {
            nicheDirs.push_back(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        } else {
            usage();
            return 1;
        }
    }
    if (modelPath.empty() || outputPath.empty() || nicheDirs.empty()) {
        usage();
        return 1;
    }
    const std::string sidecarPath = outputPath + ".weight";

    // ---- Per-niche-dir validation --------------------------------------
    std::vector<NicheDir> niches;
    niches.reserve(nicheDirs.size());
    for (const auto& dir : nicheDirs) {
        NicheDir niche;
        if (!loadNicheDir(dir, niche)) {
            return 1;
        }
        niches.push_back(niche);
    }

    // ---- Model load + exact-shape pairing (D-02, D-04, T-05-05) --------
    auto varMap = Variable::loadMap(modelPath.c_str());
    if (varMap.empty()) {
        MNN_ERROR("sgfp4_inject: '%s' loaded as an empty variable map\n", modelPath.c_str());
        return 1;
    }
    auto inputOutputs = Variable::getInputAndOutput(varMap);

    std::vector<InjectedNode> injected;
    injected.reserve(niches.size());
    for (auto& niche : niches) {
        // Enumerate candidate weights: non-input 2-D vars with exact dims.
        std::vector<std::pair<std::string, VARP>> candidates;
        for (const auto& nameVar : varMap) {
            if (0 != inputOutputs.first.count(nameVar.first)) {
                continue; // graph inputs are not weights
            }
            auto info = nameVar.second->getInfo();
            if (nullptr == info || info->dim.size() != 2 || info->dim[0] != niche.dimO || info->dim[1] != niche.dimI) {
                continue;
            }
            candidates.emplace_back(nameVar.first, nameVar.second);
        }
        if (candidates.size() != 1) {
            MNN_ERROR("sgfp4_inject: niche dir '%s' (shape {%d,%d}): expected exactly 1 weight match, found %zu:",
                      niche.dir.c_str(), niche.dimO, niche.dimI, candidates.size());
            for (const auto& cand : candidates) {
                auto info = cand.second->getInfo();
                MNN_PRINT(" candidate '%s' {idim %d}", cand.first.c_str(),
                          info ? info->dim[0] : -1);
            }
            MNN_PRINT("\n");
            return 1;
        }
        InjectedNode node;
        node.weightName = candidates[0].first;
        node.weightVar  = candidates[0].second;
        node.dimO       = niche.dimO;
        node.dimI       = niche.dimI;
        node.containerBytes = niche.bytes;
        node.sidecarSize    = niche.bytes.size();
        injected.push_back(node);
    }

    // ---- Sidecar merge (D-11, SGINJ-03): write all containers into one
    // stream, non-overlapping, 16-byte-aligned offsets. Offsets are known
    // before Op construction so the spliced ops carry final {offset, size}.
    {
        std::ofstream ofs(sidecarPath, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            MNN_ERROR("sgfp4_inject: cannot open sidecar '%s' for write\n", sidecarPath.c_str());
            return 1;
        }
        size_t offsetCursor = 0;
        for (auto& node : injected) {
            node.sidecarOffset = offsetCursor;
            ofs.write(reinterpret_cast<const char*>(node.containerBytes.data()),
                      static_cast<std::streamsize>(node.containerBytes.size()));
            // Pad to the 16-byte alignment the format mandates.
            const size_t aligned = MNN::sgfp4_align16(node.containerBytes.size());
            const size_t pad     = aligned - node.containerBytes.size();
            static const char kZero = '\0';
            for (size_t p = 0; p < pad; ++p) {
                ofs.put(kZero);
            }
            offsetCursor += aligned;
            if (ofs.fail()) {
                MNN_ERROR("sgfp4_inject: sidecar write failed at offset %zu\n", node.sidecarOffset);
                return 1;
            }
        }
    }

    // ---- Op construction + surgery (D-06, D-07, D-08, SGINJ-02) --------
    for (auto& node : injected) {
        node.op = makeDequantOp(sidecarPath, node.sidecarOffset, node.sidecarSize, node.dimO, node.dimI);
        auto dequantVar = Variable::create(Expr::create(node.op.get(), {}));
        dequantVar->setName(node.weightName + "_sgfp4");
        // After replace, weightVar is the live node; dequantVar is NOT kept
        // in any save set (Plan 05-01 Pitfall 4).
        Variable::replace(node.weightVar, dequantVar);
    }

    // ---- Serialize (SGINJ-04): recompute outputs AFTER all rewiring ----
    auto outputs = Variable::mapToSequence(Variable::getInputAndOutput(varMap).second);
    Variable::save(outputs, outputPath.c_str());
    MNN_PRINT("sgfp4_inject: wrote '%s' + '%s' (%zu node(s) injected)\n", outputPath.c_str(), sidecarPath.c_str(),
              injected.size());

    // ---- In-tool verify (D-12, D-13): unconditional reload + per-node
    // decode-vs-oracle comparison, isolated 0-input sub-modules.
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setExternalFile(sidecarPath); // BEFORE load (Pitfall 5)
    {
        std::shared_ptr<Module> full(Module::load({}, {}, outputPath.c_str(), rtmgr));
        if (nullptr == full) {
            MNN_ERROR("sgfp4_inject: verification reload of '%s' returned null module\n", outputPath.c_str());
            return 1;
        }
    }
    for (const auto& node : injected) {
        // Isolated 0-input module over just the dequant op (same offset/size/
        // externalPath as the spliced node) -- decode-vs-decode, deterministic.
        std::ostringstream oss;
        oss << outputPath << ".verify_" << (&node - injected.data()) << ".mnn";
        const std::string tempPath = oss.str();
        auto op       = makeDequantOp(sidecarPath, node.sidecarOffset, node.sidecarSize, node.dimO, node.dimI);
        auto nodeVar  = Variable::create(Expr::create(op.get(), {}));
        Variable::save({nodeVar}, tempPath.c_str());
        std::shared_ptr<Module> m(Module::load({}, {}, tempPath.c_str(), rtmgr));
        std::remove(tempPath.c_str());
        if (nullptr == m) {
            MNN_ERROR("sgfp4_inject: verification sub-module for '%s' failed to load\n", node.weightName.c_str());
            return 1;
        }
        auto outs = m->onForward({});
        if (outs.empty()) {
            MNN_ERROR("sgfp4_inject: verification sub-module for '%s' produced no output\n", node.weightName.c_str());
            return 1;
        }
        const float* got = outs[0]->readMap<float>();
        auto info        = outs[0]->getInfo();
        if (nullptr == got || nullptr == info) {
            MNN_ERROR("sgfp4_inject: verification sub-module for '%s' output unreadable\n", node.weightName.c_str());
            return 1;
        }
        const size_t elementCount = static_cast<size_t>(node.dimO) * node.dimI;
        if (static_cast<size_t>(info->size) != elementCount) {
            MNN_ERROR("sgfp4_inject: verification sub-module for '%s': %d elements != expected %zu\n",
                      node.weightName.c_str(), static_cast<int>(info->size), elementCount);
            return 1;
        }
        std::vector<float> oracle(elementCount, 0.0f);
        if (!MNN::dequant_sgfp4_container_cpu(node.containerBytes.data(), node.containerBytes.size(), oracle.data(),
                                              elementCount)) {
            MNN_ERROR("sgfp4_inject: oracle decode of container for '%s' failed\n", node.weightName.c_str());
            return 1;
        }
        bool ok = true;
        for (size_t k = 0; k < elementCount; ++k) {
            const float diff = got[k] - oracle[k];
            const float denom = std::max(std::fabs(oracle[k]), 1e-6f);
            if (std::fabs(diff) > kOracleRelativeTolerance * denom) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            MNN_ERROR("sgfp4_inject: verification mismatch for '%s' (decode vs oracle, rtol %g)\n",
                      node.weightName.c_str(), kOracleRelativeTolerance);
            return 1;
        }
        MNN_PRINT("sgfp4_inject: node '%s' {%d,%d} offset=%zu size=%zu verified (decode==oracle)\n",
                  node.weightName.c_str(), node.dimO, node.dimI, node.sidecarOffset, node.sidecarSize);
    }
    MNN_PRINT("sgfp4_inject: done\n");
    return 0;
}

} // namespace sgfp4_inject

#endif // TOOLS_FP4_SGFP4_INJECT_CORE_HPP
