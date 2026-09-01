//
//  sgfp4_encode_dump.cpp
//  MNN SGFP4 v2 dump-driven encode harness (Phase 10 parity tool)
//
//  Created by MNN on 2026/08/31.
//  Copyright © 2018, Alibaba Group Holding Limited.
//
// Plan 10-02 (D-11, user-approved): a dump->container transducer only. It
// reads a raw little-endian FP32 row-major dump (dimO*dimI values), runs
// the shipped Phase 9 C++ encoder (sgfp4_encode::encode), and writes the
// SGFP4 v2 container -- so the Python validation driver (Plan 10-03) can
// parity-sample the *shipped C++ encoder itself* against gnus-poc
// fp4_exporter.py --adaptive on real weights. No model parsing, no decode,
// no other responsibilities. Unlike sgfp4_inject.out (which deliberately
// does NOT link the encoder), this target links sgfp4_encode.
//
// CLI: sgfp4_encode_dump.out --weights <f32-dump> --dimO <N> --dimI <M>
//                           --out <container-path>
//
// Exit codes: 0 = container written on success
//             1 = usage / IO / size-mismatch error (no output file)
//             2 = encoder rejected the input (NaN/Inf/bad dims contract:
//                 empty-vector return) -- no output file
//

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "MNN/MNNDefine.h"
#include "sgfp4_encode.hpp"

namespace {

// Dims beyond this are rejected by the encoder contract anyway; check here
// so usage errors exit 1 before any allocation (mirrors sgfp4_encode.hpp).
constexpr int kMaxDim = 65536;

void printUsage(const char* argv0) {
    MNN_PRINT("usage: %s --weights <f32-dump-path> --dimO <N> --dimI <M> --out <container-path>\n", argv0);
}

bool parseIntArg(const char* text, int& out) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    if (value < 1 || value > kMaxDim) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool readFileBytes(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        return false;
    }
    const std::streamsize size = ifs.tellg();
    if (size < 0) {
        return false;
    }
    out.resize(static_cast<size_t>(size));
    ifs.seekg(0, std::ios::beg);
    if (!ifs.read(reinterpret_cast<char*>(out.data()), size)) {
        return false;
    }
    return true;
}

} // namespace

int main(int argc, const char* argv[]) {
    std::string weightsPath;
    std::string outPath;
    int dimO = 0;
    int dimI = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) {
            weightsPath = argv[++i];
        } else if (arg == "--dimO" && i + 1 < argc) {
            if (!parseIntArg(argv[++i], dimO)) {
                MNN_ERROR("sgfp4_encode_dump: --dimO must be a positive integer <= %d\n", kMaxDim);
                return 1;
            }
        } else if (arg == "--dimI" && i + 1 < argc) {
            if (!parseIntArg(argv[++i], dimI)) {
                MNN_ERROR("sgfp4_encode_dump: --dimI must be a positive integer <= %d\n", kMaxDim);
                return 1;
            }
        } else if (arg == "--out" && i + 1 < argc) {
            outPath = argv[++i];
        } else {
            MNN_ERROR("sgfp4_encode_dump: unknown or incomplete argument '%s'\n", arg.c_str());
            printUsage(argv[0]);
            return 1;
        }
    }

    if (weightsPath.empty() || outPath.empty() || dimO == 0 || dimI == 0) {
        printUsage(argv[0]);
        return 1;
    }

    std::vector<uint8_t> bytes;
    if (!readFileBytes(weightsPath, bytes)) {
        MNN_ERROR("sgfp4_encode_dump: cannot read weights dump '%s'\n", weightsPath.c_str());
        return 1;
    }

    // Exact size contract: raw little-endian FP32, row-major, exactly
    // dimO*dimI values.
    const size_t expected = static_cast<size_t>(dimO) * static_cast<size_t>(dimI) * sizeof(float);
    if (bytes.size() != expected) {
        MNN_ERROR("sgfp4_encode_dump: size mismatch for '%s': got %zu bytes, expected %zu "
                  "(%d x %d floats)\n",
                  weightsPath.c_str(), bytes.size(), expected, dimO, dimI);
        return 1;
    }

    const float* weights = reinterpret_cast<const float*>(bytes.data());
    std::vector<uint8_t> container = sgfp4_encode::encode(weights, dimO, dimI);
    if (container.empty()) {
        // Encoder invalid-input contract: non-finite (NaN/Inf) weights or
        // bad dims. No output file is written.
        MNN_ERROR("sgfp4_encode_dump: encoder rejected input '%s' (NaN/Inf or bad dims)\n",
                  weightsPath.c_str());
        return 2;
    }

    // Write only after the full encode succeeded (atomic-enough for the
    // tooling loop: a failed run never leaves a stale/partial container).
    std::ofstream ofs(outPath, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        MNN_ERROR("sgfp4_encode_dump: cannot open output '%s' for writing\n", outPath.c_str());
        return 1;
    }
    if (!ofs.write(reinterpret_cast<const char*>(container.data()),
                   static_cast<std::streamsize>(container.size()))) {
        MNN_ERROR("sgfp4_encode_dump: short write to '%s'\n", outPath.c_str());
        return 1;
    }
    if (!ofs.good()) {
        MNN_ERROR("sgfp4_encode_dump: stream error writing '%s'\n", outPath.c_str());
        return 1;
    }

    // One machine-parseable summary line (dims + container size).
    MNN_PRINT("sgfp4_encode_dump: dimO=%d dimI=%d container_bytes=%zu\n", dimO, dimI, container.size());
    return 0;
}
