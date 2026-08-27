//
//  SGFP4DequantUtils.hpp
//  MNN
//
//  Created by MNN on 2026/08/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SGFP4DequantUtils_hpp
#define SGFP4DequantUtils_hpp

#include <cstdint>
#include <cstddef>
#include <cstring>
#include "half.hpp"

namespace MNN {

// ---------------------------------------------------------------------------
// SGFP4 v2 container format constants (sgfp4-arxiv-v2.txt sections 3.2, 4.3,
// 6.1, 6.2). All multi-byte values in the container are little-endian.
// ---------------------------------------------------------------------------

// v2 magic 'SGF4', assembled little-endian from its ASCII bytes so the
// literal documents itself (matches how read_u32_le() would interpret the
// four on-disk bytes 'S','G','F','4').
constexpr uint32_t kSGFP4Magic =
    (static_cast<uint32_t>('S')) |
    (static_cast<uint32_t>('G') << 8) |
    (static_cast<uint32_t>('F') << 16) |
    (static_cast<uint32_t>('4') << 24);
constexpr uint8_t kSGFP4Version = 0x02;

// Fixed header layout: magic(4) + version(1) + B(4) + pad0(7) = 16 bytes.
constexpr size_t kSGFP4FixedHeaderSize   = 16;
constexpr size_t kSGFP4VersionByteOffset = 4;  // right after magic
constexpr size_t kSGFP4RecordCountOffset = 5;  // right after magic+version
constexpr size_t kSGFP4RecordOffsetTableStart = kSGFP4FixedHeaderSize; // byte 16
constexpr size_t kSGFP4RecordOffsetEntrySize  = 4; // each record_offsets[b] is a u32
constexpr size_t kSGFP4Alignment              = 16; // 16-byte alignment throughout

// Per-leaf FP16 header unpack (spec section 6.2, Eq. 6):
// S = half(h >> 16); bias = half(h & 0xFFF0); flags = h & 0xF; mode = flags & 0x1.
constexpr uint32_t kSGFP4LeafHeaderScaleShift = 16;
constexpr uint32_t kSGFP4LeafHeaderBiasMask   = 0xFFF0u;
constexpr uint32_t kSGFP4LeafHeaderFlagsMask  = 0xFu;
constexpr uint32_t kSGFP4LeafHeaderModeBit    = 0x1u;

// Dual-mode payload packing (spec section 4.3, Eq. 3/4).
constexpr int kSGFP4NibblesPerWord         = 8;  // mode 0: 8 x 4-bit codes / u32 word
constexpr int kSGFP4SymbolsPerWord         = 16; // mode 1: 16 x 2-bit codes / u32 word
constexpr int kSGFP4NibbleBitWidth         = 4;
constexpr int kSGFP4SymbolBitWidth         = 2;
constexpr int kSGFP4NibbleMask             = 0xF;
constexpr int kSGFP4SymbolMask             = 0x3;
constexpr int kSGFP4TwosComplementSignBias = 0x8; // sign-extend a 4-bit two's complement value

// sb_header (spec section 6.2): layout enum lives in bits 0-2.
constexpr uint32_t kSGFP4LayoutEnumMask = 0x7u;

// split-map (spec section 6.2): LAYOUT_MIXED serializes the quadtree as a
// pre-order-DFS bitmap -- one bit per node of size >= 8 (1 = split, 0 = leaf),
// quadrant order TL/TR/BL/BR; nodes of size 4 never split and emit no bit.
// At most 1 + 4 + 16 + 64 = 85 bits are required, stored in three
// little-endian uint32 words (bit k = bit k%32 of word k/32), upper bits zero.
constexpr size_t kSGFP4SplitMapWords     = 3;
constexpr size_t kSGFP4SplitMapBytes     = 12;
constexpr int    kSGFP4MaxQuadTreeBits   = 85; // 1 + 4 + 16 + 64
constexpr int    kSGFP4QuadTreeMinSplitSize = 8; // nodes of size 4 are always leaves

// Table 3 uniform-layout map (Phase 1 subset -- LAYOUT_MIXED is Phase 2).
enum SGFP4UniformLayout : uint32_t {
    kSGFP4LayoutUniform64 = 0, // N=1,   leaf edge 64
    kSGFP4LayoutUniform32 = 1, // N=4,   leaf edge 32
    kSGFP4LayoutUniform16 = 2, // N=16,  leaf edge 16
    kSGFP4LayoutUniform8  = 3, // N=64,  leaf edge 8
    kSGFP4LayoutMixed     = 4, // Phase 2 -- quadtree leaves via split map
    kSGFP4LayoutFull4x4   = 5, // N=256, leaf edge 4
    kSGFP4LayoutEnumCount = 6, // anything >= this is invalid
};

/**
 * @brief Round `x` up to the next multiple of the container's 16-byte
 * alignment (spec section 6.1: record region start, per-payload padding).
 */
inline size_t sgfp4_align16(size_t x) {
    return (x + (kSGFP4Alignment - 1)) & ~(kSGFP4Alignment - 1);
}

/**
 * @brief Read a little-endian uint32 from an arbitrary byte pointer.
 * Caller must have already bounds-checked `p + 4 <= end`.
 */
inline uint32_t sgfp4_read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

/**
 * @brief Byte-level SGFP4 v2 version gate (SGINJ-01).
 *
 * Probes the container's own bytes -- magic == 'SGF4' AND version == 0x02 --
 * without decoding anything. Legacy v1 fixed-payload containers
 * (headers[B]|offsets[B]|codes_blob) carry no SGF4 magic, so this probe
 * rejects them without ever trusting a manifest's format label (the
 * gnus-poc manifest mislabels v2 as "fp4_ultra_v0.2"; never consult it).
 */
inline bool sgfp4_is_v2_container(const uint8_t* data, size_t size) {
    return data != nullptr && size >= kSGFP4FixedHeaderSize &&
           sgfp4_read_u32_le(data) == kSGFP4Magic &&
           data[kSGFP4VersionByteOffset] == kSGFP4Version;
}

/**
 * @brief Resolve a Table 3 uniform layout enum to its leaf count N and leaf
 * edge size n. Returns false for LAYOUT_MIXED (Phase 2) or any enum >= 6.
 */
inline bool sgfp4_resolve_uniform_layout(uint32_t layoutEnum, int& leafCount, int& leafEdge) {
    switch (layoutEnum) {
        case kSGFP4LayoutUniform64:
            leafCount = 1;
            leafEdge  = 64;
            return true;
        case kSGFP4LayoutUniform32:
            leafCount = 4;
            leafEdge  = 32;
            return true;
        case kSGFP4LayoutUniform16:
            leafCount = 16;
            leafEdge  = 16;
            return true;
        case kSGFP4LayoutUniform8:
            leafCount = 64;
            leafEdge  = 8;
            return true;
        case kSGFP4LayoutFull4x4:
            leafCount = 256;
            leafEdge  = 4;
            return true;
        case kSGFP4LayoutMixed:
        default:
            // LAYOUT_MIXED is decoded by the quadtree walk in
            // sgfp4_walk_quadtree (never through this Table 3 resolver);
            // any enum >= kSGFP4LayoutEnumCount is malformed. Both are
            // rejected here.
            return false;
    }
}

// ---------------------------------------------------------------------------
// LAYOUT_MIXED quadtree split-map walk (spec section 6.2, Phase 2).
// ---------------------------------------------------------------------------

/**
 * @brief Bounds-checked reader over the 3-word / <=85-bit split map.
 *
 * Bit k of the map is bit (k % 32) of little-endian word (k / 32). Reading
 * past kSGFP4MaxQuadTreeBits (a malformed/oversized tree) fails closed via
 * a false return, which the caller turns into a decode failure (T-02-01).
 */
struct SGFP4SplitMapReader {
    const uint32_t* words;
    int bit;

    explicit SGFP4SplitMapReader(const uint32_t* w) : words(w), bit(0) {}

    bool next(bool& out) {
        if (bit >= kSGFP4MaxQuadTreeBits) {
            return false;
        }
        out = ((words[bit >> 5] >> (bit & 31)) & 1u) != 0;
        ++bit;
        return true;
    }
};

/** @brief One quadtree node: macroblock-relative origin (x, y) and edge n. */
struct QuadNode {
    int x;
    int y;
    int n;
};

/**
 * @brief Iterative pre-order DFS walk of the split map (D-01).
 *
 * Quadrant pop order is TL, TR, BL, BR. Nodes of size >= 8 read one split
 * bit; nodes of size 4 are always leaves and read nothing. Leaves are
 * collected in traversal order -- the order the block headers and payloads
 * are stored in (spec section 6.2). Recursion-free with an explicit
 * fixed-size stack so the identical algorithm ports to the Phase 4 GLSL
 * shader, where recursion is impossible.
 *
 * @param map       pointer to the 3-word split map (bounds-checked by caller)
 * @param leafCount output: number of leaves collected
 * @param leaves    output leaf array (decoder-owned)
 * @param maxLeaves capacity of `leaves`
 * @return true on a well-formed walk, false on a past-end bit read (T-02-01)
 *         or stack/leaf overflow (T-02-04)
 */
inline bool sgfp4_walk_quadtree(const uint32_t* map, int& leafCount, QuadNode* leaves, int maxLeaves) {
    QuadNode stack[kSGFP4MaxQuadTreeBits];
    int top      = 0;
    stack[top++] = QuadNode{0, 0, 64};

    SGFP4SplitMapReader r(map);
    leafCount = 0;

    while (top > 0) {
        QuadNode node = stack[--top];
        if (node.n >= kSGFP4QuadTreeMinSplitSize) {
            bool split = false;
            if (!r.next(split)) {
                return false; // bit read past 85 -- malformed map
            }
            if (split) {
                int h = node.n / 2;
                if (top + 4 > kSGFP4MaxQuadTreeBits) {
                    return false;
                }
                // Push in reverse so TL pops first (pre-order DFS).
                stack[top++] = QuadNode{node.x + h, node.y + h, h}; // BR
                stack[top++] = QuadNode{node.x, node.y + h, h};     // BL
                stack[top++] = QuadNode{node.x + h, node.y, h};     // TR
                stack[top++] = QuadNode{node.x, node.y, h};         // TL
                continue;
            }
        }
        if (leafCount >= maxLeaves) {
            return false;
        }
        leaves[leafCount++] = node;
    }
    return true;
}

/**
 * @brief Unpack a leaf's FP16 header word into (S, bias, mode).
 *
 * S = half(h >> 16), bias = half(h & 0xFFF0) (top 12 mantissa/exponent bits;
 * the low 4 bits are repurposed as flags), mode = flags & 0x1.
 * Uses the vendored half_float::half (3rd_party/half/half.hpp) -- no
 * hand-rolled FP16 decode.
 */
inline void unpack_leaf_header(uint32_t h, float& S, float& bias, int& mode) {
    uint16_t sBits    = static_cast<uint16_t>(h >> kSGFP4LeafHeaderScaleShift);
    uint16_t biasBits = static_cast<uint16_t>(h & kSGFP4LeafHeaderBiasMask);

    half_float::half hs;
    half_float::half hb;
    std::memcpy(&hs, &sBits, sizeof(hs));
    std::memcpy(&hb, &biasBits, sizeof(hb));

    S    = static_cast<float>(hs);
    bias = static_cast<float>(hb);
    mode = static_cast<int>((h & kSGFP4LeafHeaderFlagsMask) & kSGFP4LeafHeaderModeBit);
}

/**
 * @brief Decode one leaf's dual-mode payload into `leafEdge * leafEdge`
 * reconstructed floats (row-major within the leaf), via w = S*c + bias.
 *
 * Mode 0 (FP4_AFFINE): plain 4-bit two's-complement codes in [-8,7], 8 codes
 * per little-endian u32 word (n*n/8 words total). NOT E2M1.
 * Mode 1 (T158_AFFINE): 2-bit ternary codes, 16 per word (n*n/16 words
 * total); 00->0, 01->+1, 10->-1, 11->0 (reserved).
 *
 * @param words      pointer to the leaf's payload words (already bounds-checked by the caller)
 * @param leafEdge   leaf edge size n
 * @param S          decoded scale
 * @param bias       decoded bias
 * @param mode       0 = FP4_AFFINE, 1 = T158_AFFINE
 * @param out        destination for leafEdge*leafEdge floats
 */
inline void sgfp4_decode_leaf_payload(const uint32_t* words, int leafEdge, float S, float bias, int mode,
                                       float* out) {
    const int elementCount = leafEdge * leafEdge;
    if (mode == 0) {
        for (int i = 0; i < elementCount; ++i) {
            uint32_t w   = words[i / kSGFP4NibblesPerWord];
            int shift    = kSGFP4NibbleBitWidth * (i % kSGFP4NibblesPerWord);
            int nib      = static_cast<int>((w >> shift) & kSGFP4NibbleMask);
            int c        = (nib ^ kSGFP4TwosComplementSignBias) - kSGFP4TwosComplementSignBias;
            out[i]       = S * static_cast<float>(c) + bias;
        }
    } else {
        for (int i = 0; i < elementCount; ++i) {
            uint32_t w   = words[i / kSGFP4SymbolsPerWord];
            int shift    = kSGFP4SymbolBitWidth * (i % kSGFP4SymbolsPerWord);
            int sym      = static_cast<int>((w >> shift) & kSGFP4SymbolMask);
            int c        = (sym == 1) ? 1 : (sym == 2) ? -1 : 0; // 00->0,01->+1,10->-1,11->0(reserved)
            out[i]       = S * static_cast<float>(c) + bias;
        }
    }
}

/**
 * @brief Decode a full SGFP4 v2 uniform-layout container into a flat float
 * output buffer.
 *
 * Container layout (spec sections 6.1/6.2, little-endian throughout):
 *   fixed header (16B): magic[4]='SGF4', version(u8)=0x02, B(u32), pad0[7]
 *   record_offsets[B] (u32 each), starting at byte 16
 *   record region, starting at align16(16 + 4*B); record_offsets[b] is
 *     relative to this region start.
 *   per record: sb_header(u32) -> layout enum in bits 0-2 (Table 3);
 *     block_headers[N] (u32 each, N from Table 3); pad to a 16-byte
 *     boundary; payloads[N] (each padded to a 16-byte multiple).
 *
 * Decode order is fully sequential: record 0's leaves fill the first
 * N0*n0*n0 output elements, record 1's leaves fill the next N1*n1*n1, and
 * so on. Uniform-layout leaves are appended in row-major raster order of
 * the tile grid (row-major within each leaf); LAYOUT_MIXED leaves are
 * appended in pre-order DFS traversal order of the split map (TL/TR/BL/BR),
 * each leaf still row-major internally (spec section 6.2) -- a leaf-major
 * linear stream, not a spatial scatter.
 *
 * Every read is bounds-checked against `containerSize` before it happens
 * (ASVS V5); malformed/out-of-bounds containers return false without ever
 * dereferencing past the buffer. LAYOUT_MIXED is decoded via the
 * sgfp4_walk_quadtree split-map walk (with strict tiling validation); any
 * layout enum >= 6 is still rejected by sgfp4_resolve_uniform_layout. The
 * total decoded element count is bounded
 * against and must exactly equal `outElementCount` -- this also bounds
 * per-record work against the declared output size (no unbounded
 * allocation/looping from an attacker-controlled B).
 *
 * @param container       pointer to the raw container bytes (untrusted)
 * @param containerSize   size of `container` in bytes
 * @param out             destination float buffer (must hold outElementCount floats)
 * @param outElementCount expected total decoded element count (product of manifest dims)
 * @return true on success, false on any malformed/out-of-bounds input
 */
inline bool dequant_sgfp4_container_cpu(const uint8_t* container, size_t containerSize, float* out,
                                         size_t outElementCount) {
    if (container == nullptr || out == nullptr) {
        return false;
    }
    if (containerSize < kSGFP4FixedHeaderSize) {
        return false;
    }

    uint32_t magic = sgfp4_read_u32_le(container);
    if (magic != kSGFP4Magic) {
        return false;
    }
    uint8_t version = container[kSGFP4VersionByteOffset];
    if (version != kSGFP4Version) {
        return false;
    }
    // B is read as a u32 at byte offset 5; this straddles the version byte,
    // matching the spec's packed fixed header (magic4+version1+B4+pad7=16).
    if (kSGFP4RecordCountOffset + 4 > containerSize) {
        return false;
    }
    uint32_t B = sgfp4_read_u32_le(container + kSGFP4RecordCountOffset);

    // Bound the record-offset table itself against containerSize before
    // trusting B for any further arithmetic (DoS guard, T-01-02).
    size_t offsetTableBytes = static_cast<size_t>(B) * kSGFP4RecordOffsetEntrySize;
    if (offsetTableBytes / kSGFP4RecordOffsetEntrySize != static_cast<size_t>(B)) {
        return false; // overflow
    }
    if (kSGFP4RecordOffsetTableStart > containerSize ||
        offsetTableBytes > containerSize - kSGFP4RecordOffsetTableStart) {
        return false;
    }

    size_t regionStart = sgfp4_align16(kSGFP4RecordOffsetTableStart + offsetTableBytes);
    if (regionStart > containerSize) {
        return false;
    }

    size_t outCursor = 0;
    for (uint32_t b = 0; b < B; ++b) {
        // Extra records beyond the declared output geometry: reject rather
        // than silently ignore (geometry/container mismatch).
        if (outCursor >= outElementCount) {
            return false;
        }

        size_t offEntry = kSGFP4RecordOffsetTableStart + static_cast<size_t>(b) * kSGFP4RecordOffsetEntrySize;
        if (offEntry + 4 > containerSize) {
            return false;
        }
        uint32_t recOffRel = sgfp4_read_u32_le(container + offEntry);
        if (recOffRel > containerSize - regionStart) {
            return false;
        }
        size_t recStart = regionStart + recOffRel;
        if (recStart + 4 > containerSize) {
            return false;
        }

        uint32_t sbHeader   = sgfp4_read_u32_le(container + recStart);
        uint32_t layoutEnum = sbHeader & kSGFP4LayoutEnumMask;
        int leafCount = 0;
        int leafEdge  = 0;
        QuadNode leaves[256];
        bool isMixed = (layoutEnum == kSGFP4LayoutMixed);
        size_t blockHeadersStart = 0;
        if (isMixed) {
            // A 12-byte split map sits between sb_header and block headers
            // (spec section 6.2). Bounds-check the map span, walk the
            // quadtree, then validate strict tiling: leaf areas must sum to
            // exactly 4096 (64*64) (T-02-02). Any malformed map fails decode.
            if (kSGFP4SplitMapBytes > containerSize - (recStart + 4)) {
                return false;
            }
            const uint32_t* map = reinterpret_cast<const uint32_t*>(container + recStart + 4);
            if (!sgfp4_walk_quadtree(map, leafCount, leaves, 256)) {
                return false;
            }
            int area = 0;
            for (int i = 0; i < leafCount; ++i) {
                area += leaves[i].n * leaves[i].n;
            }
            if (area != 64 * 64) {
                return false;
            }
            blockHeadersStart = recStart + 4 + kSGFP4SplitMapBytes;
        } else {
            if (!sgfp4_resolve_uniform_layout(layoutEnum, leafCount, leafEdge)) {
                return false;
            }
            blockHeadersStart = recStart + 4;
        }
        size_t blockHeadersBytes = static_cast<size_t>(leafCount) * 4;
        if (blockHeadersBytes > containerSize - blockHeadersStart) {
            return false;
        }
        const uint8_t* blockHeaders = container + blockHeadersStart;

        size_t payloadsStart = sgfp4_align16(blockHeadersStart + blockHeadersBytes);
        if (payloadsStart > containerSize) {
            return false;
        }

        size_t payloadCursor = payloadsStart;
        for (int leaf = 0; leaf < leafCount; ++leaf) {
            uint32_t header = sgfp4_read_u32_le(blockHeaders + leaf * 4);
            float S = 0.0f, bias = 0.0f;
            int mode = 0;
            unpack_leaf_header(header, S, bias, mode);

            // MIXED leaves carry a per-leaf edge size from the split-map
            // walk; uniform leaves all share the Table 3 edge size.
            int n = isMixed ? leaves[leaf].n : leafEdge;
            int elementCount = n * n;
            int wordsPerLeaf = (mode == 0) ? (elementCount / kSGFP4NibblesPerWord)
                                            : (elementCount / kSGFP4SymbolsPerWord);
            size_t payloadBytes = static_cast<size_t>(wordsPerLeaf) * 4;
            if (payloadBytes > containerSize - payloadCursor) {
                return false;
            }
            if (static_cast<size_t>(elementCount) > outElementCount - outCursor) {
                return false;
            }

            sgfp4_decode_leaf_payload(reinterpret_cast<const uint32_t*>(container + payloadCursor), n, S,
                                      bias, mode, out + outCursor);

            outCursor += static_cast<size_t>(elementCount);
            payloadCursor += sgfp4_align16(payloadBytes);
        }
    }

    return outCursor == outElementCount;
}

} // namespace MNN

#endif /* SGFP4DequantUtils_hpp */
