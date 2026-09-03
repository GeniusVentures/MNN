//
//  SGFP4TestUtil.hpp
//  MNNTests
//
//  Created by MNN on 2026/08/28.
//  Copyright © 2018, Alibaba Group Holding Limited.
//
// Plan 08-02: single header-only home for the SGFP4 test helpers that were
// previously duplicated across SGFP4ClassicAPITest.cpp /
// SGFP4MultiTensorTest.cpp / SGFP4InjectTest.cpp (v2.0 audit D-10 debt).
// The canonical container builder is the GENERALIZED REGION-RELATIVE
// variant from SGFP4MultiTensorTest.cpp (offset-table entries are relative
// to the record-region start, matching the gnus-poc encoder convention).
// The absolute-offset variant that lived in SGFP4ClassicAPITest.cpp is
// deliberately NOT carried forward (W-1 offset-convention divergence).
//
// Pure helpers only -- no MNNTestSuite.h dependency, so non-test tools may
// include it. Every function is `inline` (C++11, multiple-TU safety).
//
#ifndef SGFP4TestUtil_hpp
#define SGFP4TestUtil_hpp

#include <cstdint>
#include <cstdlib>
#include <ctime>
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

#include "MNN/SGFP4DequantUtils.hpp"
#include "fp4/sha256.hpp"

namespace sgfp4_test {

// ====================================================================
// Uniform-container framing constants (all-UNIFORM_64 degenerate case,
// mirrors encode_sgfp4.py encode_macroblock).
// ====================================================================

constexpr int kMacroblockEdge = 64;

constexpr size_t kElementsPerLeaf   = static_cast<size_t>(kMacroblockEdge) * kMacroblockEdge;              // 4096
constexpr size_t kNibblePayloadBytes = (kElementsPerLeaf / MNN::kSGFP4NibblesPerWord) * sizeof(uint32_t);  // 2048
constexpr size_t kRecordPrePayload  = 2 * sizeof(uint32_t);                                                // 8
constexpr size_t kRecordPadBytes    = MNN::kSGFP4Alignment - kRecordPrePayload;                            // 8
constexpr size_t kRecordSize        = kRecordPrePayload + kRecordPadBytes + kNibblePayloadBytes;           // 2064

constexpr uint32_t kLeafScaleOneBits = 0x3C00; // IEEE754 half(1.0)
constexpr uint32_t kNibbleCode       = 0x1;    // decodes to S*(+1)+bias = 1.0f

// ====================================================================
// Filesystem / serialization helpers (Phase 6 precedents).
// ====================================================================

inline std::string tempPath(const char* prefix, const char* suffix) {
    std::ostringstream oss;
    oss << prefix << static_cast<unsigned long>(std::time(nullptr)) << "_"
        << static_cast<unsigned long>(std::rand()) << suffix;
    return oss.str();
}

inline std::string cwdPath() {
    std::vector<char> buf(1024, '\0');
#if defined(_WIN32)
    return std::string(_getcwd(buf.data(), static_cast<int>(buf.size() - 1)));
#else
    return std::string(getcwd(buf.data(), buf.size() - 1));
#endif
}

inline bool makeDir(const std::string& path) {
#if defined(_WIN32)
    return 0 == _mkdir(path.c_str());
#else
    return 0 == mkdir(path.c_str(), 0755);
#endif
}

inline void removeDir(const std::string& path) {
#if defined(_WIN32)
    _rmdir(path.c_str());
#else
    rmdir(path.c_str());
#endif
}

inline bool fileExists(const std::string& path) {
    std::ifstream ifs(path);
    return ifs.good();
}

inline void writeU32Le(std::vector<uint8_t>& out, size_t offset, uint32_t value) {
    out[offset]     = static_cast<uint8_t>(value & 0xFFu);
    out[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    out[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xFFu);
    out[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xFFu);
}

inline bool writeBytes(const std::string& path, const uint8_t* data, size_t size) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    return static_cast<size_t>(ofs.tellp()) == size;
}

inline bool readBytes(const std::string& path, std::vector<uint8_t>& out) {
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

// Generalized all-UNIFORM_64 container builder (canonical region-relative
// convention): any [dimO, dimI] with 64-multiple dims; recordCount =
// (dimO/64)*(dimI/64) sequential records of the degenerate leaf (S=1.0,
// all nibble codes 1 -> decode 1.0).
//
// Offset-table entries are RELATIVE to the record-region start (the
// encoder's convention: offsets.append(cursor) with cursor starting at 0
// -- encode_sgfp4.py; the decoder recomputes recStart = regionStart +
// recOffRel). Writing absolute offsets leaves the decode reading
// payload-pattern bytes as headers -- deterministic but semantically
// wrong framing.
inline bool buildContainerUniform64(int dimO, int dimI, std::vector<uint8_t>& out) {
    const int tilesY      = dimO / kMacroblockEdge;
    const int tilesX      = dimI / kMacroblockEdge;
    const int recordCount = tilesY * tilesX;

    // Offset table lives at byte 16, record region at align16(16+B*4).
    // Computed arithmetically; agreement with the format's own inline
    // helper asserted at runtime below (constexpr calling the inline
    // sgfp4_align16 trips MSVC C2131 -- Pitfall 4, Phase 6 auto-fix).
    const size_t offsetTableBytes =
        MNN::kSGFP4RecordOffsetTableStart + static_cast<size_t>(recordCount) * MNN::kSGFP4RecordOffsetEntrySize;
    const size_t recordRegionStart = (offsetTableBytes + MNN::kSGFP4Alignment - 1) & ~(MNN::kSGFP4Alignment - 1);
    if (recordRegionStart != MNN::sgfp4_align16(offsetTableBytes)) {
        return false;
    }

    out.assign(recordRegionStart + static_cast<size_t>(recordCount) * kRecordSize, 0);
    writeU32Le(out, 0, MNN::kSGFP4Magic);
    out[MNN::kSGFP4VersionByteOffset] = MNN::kSGFP4Version;
    writeU32Le(out, MNN::kSGFP4RecordCountOffset, static_cast<uint32_t>(recordCount));
    for (int b = 0; b < recordCount; ++b) {
        writeU32Le(out, MNN::kSGFP4RecordOffsetTableStart + b * MNN::kSGFP4RecordOffsetEntrySize,
                   static_cast<uint32_t>(b * kRecordSize));
    }
    const uint32_t sbHeader    = static_cast<uint32_t>(MNN::kSGFP4LayoutUniform64) & MNN::kSGFP4LayoutEnumMask;
    const uint32_t leafHeader  = (kLeafScaleOneBits << MNN::kSGFP4LeafHeaderScaleShift);
    const uint32_t payloadWord = kNibbleCode * 0x11111111u; // code in all 8 nibbles
    for (int b = 0; b < recordCount; ++b) {
        const size_t rec = recordRegionStart + b * kRecordSize;
        writeU32Le(out, rec, sbHeader);
        writeU32Le(out, rec + sizeof(uint32_t), leafHeader);
        const size_t payloadStart = rec + kRecordPrePayload + kRecordPadBytes;
        for (size_t w = 0; w < kNibblePayloadBytes / sizeof(uint32_t); ++w) {
            writeU32Le(out, payloadStart + w * sizeof(uint32_t), payloadWord);
        }
        // pad bytes stay zero from assign()
    }
    return true;
}

// Parameterized synthetic niche dir writer (the manifest `path` is the
// container BASENAME, cross-checked case-insensitively by the tool;
// sha256 always over the exact container bytes via sgfp4::sha256_hex).
inline bool writeNicheDir(const std::vector<uint8_t>& containerBytes, const std::string& dir,
                          const std::string& containerName, int dimO, int dimI) {
    if (!makeDir(dir)) {
        return false;
    }
    const std::string containerPath = dir + "/" + containerName;
    if (!writeBytes(containerPath, containerBytes.data(), containerBytes.size())) {
        return false;
    }
    const std::string digest = sgfp4::sha256_hex(containerBytes.data(), containerBytes.size());
    std::ostringstream oss;
    oss << "{\"fp4_binary\":{\"path\":\"" << containerName << "\",\"sha256\":\"" << digest
        << "\",\"stats\":{\"shape\":[" << dimO << "," << dimI << "]}}}";
    const std::string manifest = oss.str();
    return writeBytes(dir + "/manifest.json", reinterpret_cast<const uint8_t*>(manifest.data()), manifest.size());
}

} // namespace sgfp4_test

#endif /* SGFP4TestUtil_hpp */
