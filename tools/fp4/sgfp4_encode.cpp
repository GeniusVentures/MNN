//
//  sgfp4_encode.cpp
//  MNN
//
//  Created by MNN on 2026/08/29.
//  Copyright © 2018, Alibaba Group Holding Limited.
//
// SGFP4 v2 adaptive quadtree weight encoder (Phase 9, Plan 09-01).
//
// Functional port of the gnus-poc adaptive exporter stack:
//   - fp4_exporter.py: _export_v2_adaptive (container framing, zero-pad,
//     _fit_affine 16-candidate scale search, _fit_ternary, _classify_layout,
//     _build_split_map, record assembly, 16-byte alignment rules)
//   - quadtree.py: QuadtreeEncoder._try_block (recursive accept/split with
//     hysteresis slack, outlier veto), _combined_gate_error
//   - laplacian.py: LaplacianWeightedError.compute (separable Gaussian
//     Laplacian pyramid, sigma 2*2^level, reflect boundary)
//
// Framing constants are REUSED from MNN::SGFP4DequantUtils.hpp (single
// source of truth shared with the decoders) -- never redefined here.
//
// All fit/error accumulation runs in double (Pitfall 3); code rounding uses
// std::rint with FE_TONEAREST (numpy np.round half-to-even parity, Pitfall 2
// -- never std::round). The quadtree accepts with _kHysteresisSlack = 1.1;
// the 0.8 improvement factor is dead code upstream and intentionally absent.
//

#include "sgfp4_encode.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "half.hpp"

namespace sgfp4_encode {

namespace {

using MNN::kSGFP4Alignment;
using MNN::kSGFP4LeafHeaderBiasMask;
using MNN::kSGFP4LeafHeaderModeBit;
using MNN::kSGFP4LeafHeaderScaleShift;
using MNN::kSGFP4LayoutFull4x4;
using MNN::kSGFP4LayoutMixed;
using MNN::kSGFP4LayoutUniform16;
using MNN::kSGFP4LayoutUniform32;
using MNN::kSGFP4LayoutUniform64;
using MNN::kSGFP4LayoutUniform8;
using MNN::kSGFP4Magic;
using MNN::kSGFP4NibblesPerWord;
using MNN::kSGFP4SplitMapWords;
using MNN::kSGFP4SymbolsPerWord;
using MNN::kSGFP4Version;

constexpr int    kMacroblockSize    = 64;
constexpr int    kMinLeafSize       = 4;
constexpr int    kMaxDim            = 65536;
constexpr double kTernaryDelta      = 0.10;
constexpr double kHysteresisSlack   = 1.1;
constexpr double kRelativeEpsilon   = 1e-12;
constexpr double kT158OutlierScale  = 5.0;
constexpr double kFP16Max           = 65504.0;
constexpr int    kSplitMapMaxBits   = MNN::kSGFP4MaxQuadTreeBits;

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

double rintHalfToEven(double v) {
    return std::rint(v);
}

uint16_t floatToHalfBits(float v) {
    half_float::half h(v);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

void appendU32Le(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

size_t align16Size(size_t n) {
    return MNN::sgfp4_align16(n);
}

// ---------------------------------------------------------------------------
// Gaussian filter (separable, reflect boundary) -- mirrors scipy
// gaussian_filter(mode='reflect', truncate=4.0) on a row-major rows x cols
// plane. Hand-rolled per 09-RESEARCH Pitfall 1 exception: no scipy in the
// C++ toolchain.
// ---------------------------------------------------------------------------

int reflectIndex(int i, int n) {
    // scipy 'reflect' = (d c b a | a b c d | d c b a): no edge repeat.
    while (i < 0 || i >= n) {
        if (i < 0) {
            i = -i;
        }
        if (i >= n) {
            i = 2 * n - 2 - i;
        }
    }
    return i;
}

// Symmetric 1D convolution along columns (vertical) with an odd-length kernel.
void convolveVertical(const double* src, double* dst, int rows, int cols, const std::vector<double>& kernel) {
    const int radius = static_cast<int>(kernel.size() / 2);
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                acc += kernel[k + radius] * src[reflectIndex(r + k, rows) * cols + c];
            }
            dst[r * cols + c] = acc;
        }
    }
}

// Symmetric 1D convolution along rows (horizontal).
void convolveHorizontal(const double* src, double* dst, int rows, int cols, const std::vector<double>& kernel) {
    const int radius = static_cast<int>(kernel.size() / 2);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                acc += kernel[k + radius] * src[r * cols + reflectIndex(c + k, cols)];
            }
            dst[r * cols + c] = acc;
        }
    }
}

std::vector<double> gaussianKernel1D(double sigma) {
    int radius = static_cast<int>(4.0 * sigma + 0.5);
    std::vector<double> kernel(2 * radius + 1, 0.0);
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        double w = std::exp(-(static_cast<double>(i) * static_cast<double>(i)) / (2.0 * sigma * sigma));
        kernel[i + radius] = w;
        sum += w;
    }
    for (auto& w : kernel) {
        w /= sum;
    }
    return kernel;
}

// Separable 2D Gaussian on a rows x cols double plane (vertical then
// horizontal, like scipy's default order (0,1) == (rows, cols) axes).
void gaussianFilter2D(const std::vector<double>& src, std::vector<double>& dst, int rows, int cols, double sigma) {
    std::vector<double> kernel = gaussianKernel1D(sigma);
    std::vector<double> tmp(static_cast<size_t>(rows) * cols);
    convolveVertical(src.data(), tmp.data(), rows, cols, kernel);
    convolveHorizontal(tmp.data(), dst.data(), rows, cols, kernel);
}

// ---------------------------------------------------------------------------
// Laplacian-weighted error -- mirrors laplacian.py compute(). `smooth`
// starts as the residual plane and is progressively replaced by
// smooth_base[::2,::2] between levels.
// ---------------------------------------------------------------------------

double laplacianWeightedError(std::vector<double> smooth, int rows, int cols, int blockSize) {
    int levels = 0;
    switch (blockSize) {
        case 32: levels = 2; break;
        case 64: levels = 3; break;
        case 16: levels = 1; break;
        default: levels = 0; break;
    }
    if (levels == 0) {
        double acc = 0.0;
        const size_t n = static_cast<size_t>(rows) * cols;
        for (size_t i = 0; i < n; ++i) {
            acc += smooth[i] * smooth[i];
        }
        return acc / static_cast<double>(n);
    }

    double totalError = 0.0;
    double weightSum  = 0.0;
    double sigmaBase  = 2.0;
    for (int level = 0; level < levels; ++level) {
        double sigma = sigmaBase * std::pow(2.0, static_cast<double>(level));
        std::vector<double> base(static_cast<size_t>(rows) * cols);
        gaussianFilter2D(smooth, base, rows, cols, sigma);
        double bandAcc = 0.0;
        const size_t n = static_cast<size_t>(rows) * cols;
        for (size_t i = 0; i < n; ++i) {
            double band = smooth[i] - base[i];
            bandAcc += band * band;
        }
        double levelWeight = 1.0 / std::pow(2.0, static_cast<double>(level));
        totalError += levelWeight * (bandAcc / static_cast<double>(n));
        weightSum += levelWeight;

        if (level < levels - 1) {
            // Subsample smooth = smooth_base[::2, ::2].
            int halfRows = rows / 2;
            int halfCols = cols / 2;
            std::vector<double> down(static_cast<size_t>(halfRows) * halfCols);
            for (int r = 0; r < halfRows; ++r) {
                for (int c = 0; c < halfCols; ++c) {
                    down[static_cast<size_t>(r) * halfCols + c] = base[static_cast<size_t>(2 * r) * cols + 2 * c];
                }
            }
            smooth = std::move(down);
            rows   = halfRows;
            cols   = halfCols;
        }
    }
    return totalError / weightSum;
}

// ---------------------------------------------------------------------------
// Code fitting -- mirrors fp4_exporter.py _fit_affine / _fit_ternary /
// _encode_fp4_affine_variable / _encode_t158_affine_variable, and
// quadtree.py _reconstruct / _t158_has_outlier. All math in double.
// ---------------------------------------------------------------------------

struct LeafResult {
    int    mode = 0;              // 0 = FP4_AFFINE, 1 = T158_AFFINE
    double l2Error = 0.0;         // plain leaf MSE (diagnostic)
    double scale = 1.0;
    double bias  = 0.0;
    std::vector<int8_t> codes;    // FP4: [-8,7]; T158: {-1,0,+1}
};

// bias = mean; 16 scale candidates over initial_scale * logspace(0.5, 1.5).
LeafResult fitAffine(const double* vals, int n) {
    LeafResult out;
    out.mode = 0;
    double sum = 0.0;
    double maxabs = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += vals[i];
        double a = std::fabs(vals[i]);
        if (a > maxabs) {
            maxabs = a;
        }
    }
    double bias = sum / static_cast<double>(n);
    double initialScale = maxabs > 0.0 ? maxabs / 7.0 : 1.0;

    double bestError = std::numeric_limits<double>::infinity();
    double bestScale = initialScale;
    // numpy.logspace(log10(0.5), log10(1.5), 16): t = 0.5 * (1.5/0.5)^(k/15).
    const double logLo       = std::log10(0.5);
    const double logHi       = std::log10(1.5);
    const double denominator = static_cast<double>(16 - 1);
    for (int k = 0; k < 16; ++k) {
        double multiplier = std::pow(10.0, logLo + (logHi - logLo) * (static_cast<double>(k) / denominator));
        double scale = initialScale * multiplier;
        double acc = 0.0;
        for (int i = 0; i < n; ++i) {
            double code = rintHalfToEven((vals[i] - bias) / scale);
            if (code > 7.0) code = 7.0;
            if (code < -8.0) code = -8.0;
            double recon = scale * code + bias;
            double d = vals[i] - recon;
            acc += d * d;
        }
        double error = acc / static_cast<double>(n);
        if (error < bestError) {
            bestError = error;
            bestScale = scale;
        }
    }

    out.scale = bestScale;
    out.bias  = bias;
    out.l2Error = bestError;
    out.codes.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        double code = rintHalfToEven((vals[i] - bias) / bestScale);
        if (code > 7.0) code = 7.0;
        if (code < -8.0) code = -8.0;
        out.codes[static_cast<size_t>(i)] = static_cast<int8_t>(code);
    }
    return out;
}

// bias = mean; scale = mean(|centered|) floored at 1e-8.
LeafResult fitTernary(const double* vals, int n) {
    LeafResult out;
    out.mode = 1;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += vals[i];
    }
    double bias = sum / static_cast<double>(n);
    double absSum = 0.0;
    for (int i = 0; i < n; ++i) {
        absSum += std::fabs(vals[i] - bias);
    }
    double scale = absSum / static_cast<double>(n);
    if (scale < 1e-8) {
        scale = 1e-8;
    }
    out.scale = scale;
    out.bias  = bias;

    double threshold = 0.5 * scale;
    double acc = 0.0;
    out.codes.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        double centered = vals[i] - bias;
        int code = 0;
        if (centered > threshold) {
            code = 1;
        } else if (centered < -threshold) {
            code = -1;
        }
        out.codes[static_cast<size_t>(i)] = static_cast<int8_t>(code);
        double recon = scale * static_cast<double>(code) + bias;
        double d = vals[i] - recon;
        acc += d * d;
    }
    out.l2Error = acc / static_cast<double>(n);
    return out;
}

// quadtree.py _t158_has_outlier: max per-weight error vs 5 * scale.
bool t158HasOutlier(const double* vals, int n, const LeafResult& t158) {
    double threshold = 0.5 * t158.scale;
    double maxError  = 0.0;
    for (int i = 0; i < n; ++i) {
        double centered = vals[i] - t158.bias;
        int code = 0;
        if (centered > threshold) {
            code = 1;
        } else if (centered < -threshold) {
            code = -1;
        }
        double recon = t158.scale * static_cast<double>(code) + t158.bias;
        double e = std::fabs(vals[i] - recon);
        if (e > maxError) {
            maxError = e;
        }
    }
    return maxError > kT158OutlierScale * t158.scale;
}

// ---------------------------------------------------------------------------
// Payload packing (row-major within the leaf, little-endian u32 words).
// ---------------------------------------------------------------------------

std::vector<uint8_t> packNibbles(const std::vector<int8_t>& codes, int mode) {
    const int perWord = (mode == 0) ? kSGFP4NibblesPerWord : kSGFP4SymbolsPerWord;
    const int bitWidth = (mode == 0) ? 4 : 2;
    const int n = static_cast<int>(codes.size());
    const int wordCount = n / perWord;
    std::vector<uint32_t> words(static_cast<size_t>(wordCount), 0u);
    for (int i = 0; i < n; ++i) {
        uint32_t sym;
        if (mode == 0) {
            sym = static_cast<uint32_t>(static_cast<uint8_t>(codes[static_cast<size_t>(i)])) & 0xFu;
        } else {
            int t = codes[static_cast<size_t>(i)];
            sym = (t == 1) ? 1u : (t == -1) ? 2u : 0u;
        }
        int word  = i / perWord;
        int shift = bitWidth * (i % perWord);
        words[static_cast<size_t>(word)] |= (sym << static_cast<uint32_t>(shift));
    }
    std::vector<uint8_t> out(static_cast<size_t>(wordCount) * 4, 0);
    for (int w = 0; w < wordCount; ++w) {
        appendU32Le(out, words[static_cast<size_t>(w)]);
    }
    if (out.size() % kSGFP4Alignment != 0) {
        out.resize(align16Size(out.size()), 0);
    }
    return out;
}

uint32_t packLeafHeader(double scale, double bias, int mode) {
    float sClip = static_cast<float>(scale);
    float bClip = static_cast<float>(bias);
    if (sClip > static_cast<float>(kFP16Max)) sClip = static_cast<float>(kFP16Max);
    if (sClip < -static_cast<float>(kFP16Max)) sClip = -static_cast<float>(kFP16Max);
    if (bClip > static_cast<float>(kFP16Max)) bClip = static_cast<float>(kFP16Max);
    if (bClip < -static_cast<float>(kFP16Max)) bClip = -static_cast<float>(kFP16Max);
    uint16_t sBits = floatToHalfBits(sClip);
    uint16_t bBits = floatToHalfBits(bClip);
    // S in the upper 16 bits; bias occupies the top 12 bits of the low half;
    // bit 0 carries MODE, bits 1-3 reserved and written 0 (Pitfall 4).
    return (static_cast<uint32_t>(sBits) << kSGFP4LeafHeaderScaleShift) |
           (static_cast<uint32_t>(bBits) & kSGFP4LeafHeaderBiasMask) |
           (static_cast<uint32_t>(mode) & kSGFP4LeafHeaderModeBit);
}

// ---------------------------------------------------------------------------
// Quadtree -- mirrors quadtree.py QuadtreeEncoder.encode/_try_block with
// DEFAULT_V2_THRESHOLDS and hysteresis slack 1.1 (dead 0.8 improvement
// deliberately absent).
// ---------------------------------------------------------------------------

struct Leaf {
    int y = 0;
    int x = 0;
    int size = 0;
    LeafResult result;
};

struct Threshold {
    int    leafSize;
    double maxMse;
    double maxRelative;
};

// DEFAULT_V2_THRESHOLDS (fp4_exporter.py): keyed by leaf size.
constexpr Threshold kDefaultV2Thresholds[5] = {
    {64, 0.01, 0.05},
    {32, 0.005, 0.03},
    {16, 0.002, 0.02},
    {8, 0.001, 0.01},
    {4, 0.0005, 0.005},
};

// double accumulation gate (quadtree.py _combined_gate_error).
double combinedGateError(const double* region, int n, double selectedError, double maxMse, double maxRelative) {
    if (maxRelative <= 0.0) {
        return selectedError;
    }
    double power = 0.0;
    for (int i = 0; i < n; ++i) {
        power += region[i] * region[i];
    }
    double signalPower = power / static_cast<double>(n);
    if (signalPower <= kRelativeEpsilon) {
        return selectedError;
    }
    double relativeEquivalent = maxMse * ((selectedError / signalPower) / maxRelative);
    return std::max(selectedError, relativeEquivalent);
}

struct QuadtreeContext {
    const double* plane; // padded plane, row-major paddedDimI-strided
    int paddedDimI;
    Threshold thresholds[5];
};

void extractRegion(const QuadtreeContext& ctx, int y, int x, int size, std::vector<double>& region) {
    region.resize(static_cast<size_t>(size) * size);
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            region[static_cast<size_t>(r) * size + c] =
                ctx.plane[static_cast<size_t>(y + r) * ctx.paddedDimI + (x + c)];
        }
    }
}

void tryBlock(const QuadtreeContext& ctx, int y, int x, int size, std::vector<Leaf>& leaves) {
    std::vector<double> region;
    extractRegion(ctx, y, x, size, region);
    const int n = size * size;

    // Threshold for this leaf size (DEFAULT_V2_THRESHOLDS covers all sizes).
    const Threshold* threshold = &kDefaultV2Thresholds[4];
    for (int i = 0; i < 5; ++i) {
        if (kDefaultV2Thresholds[i].leafSize == size) {
            threshold = &kDefaultV2Thresholds[i];
            break;
        }
    }

    LeafResult fp4  = fitAffine(region.data(), n);
    LeafResult t158 = fitTernary(region.data(), n);

    std::vector<double> residual(static_cast<size_t>(n));

    auto fillResidual = [&](const LeafResult& res) {
        if (res.mode == 1) {
            double thr = 0.5 * res.scale;
            for (int i = 0; i < n; ++i) {
                double centered = region[i] - res.bias;
                int code = 0;
                if (centered > thr) code = 1;
                else if (centered < -thr) code = -1;
                residual[static_cast<size_t>(i)] = region[i] - (res.scale * code + res.bias);
            }
        } else {
            for (int i = 0; i < n; ++i) {
                double recon = res.scale * static_cast<double>(res.codes[static_cast<size_t>(i)]) + res.bias;
                residual[static_cast<size_t>(i)] = region[i] - recon;
            }
        }
    };

    fillResidual(fp4);
    double fp4Error = laplacianWeightedError(residual, size, size, size);

    fillResidual(t158);
    double t158Error = laplacianWeightedError(residual, size, size, size);

    bool t158Preferred = t158Error <= (1.0 + kTernaryDelta) * fp4Error;
    if (t158Preferred && t158HasOutlier(region.data(), n, t158)) {
        t158Preferred = false;
    }

    LeafResult selected = t158Preferred ? t158 : fp4;
    double selectedError = t158Preferred ? t158Error : fp4Error;

    double gateError = combinedGateError(region.data(), n, selectedError, threshold->maxMse, threshold->maxRelative);

    bool accept = gateError <= threshold->maxMse;
    if (!accept && size > kMinLeafSize) {
        accept = gateError <= threshold->maxMse * kHysteresisSlack;
    }
    if (size <= kMinLeafSize) {
        accept = true;
    }

    if (accept) {
        Leaf leaf;
        leaf.y = y;
        leaf.x = x;
        leaf.size = size;
        leaf.result = std::move(selected);
        leaves.push_back(std::move(leaf));
        return;
    }

    int half = size / 2;
    tryBlock(ctx, y, x, half, leaves);
    tryBlock(ctx, y, x + half, half, leaves);
    tryBlock(ctx, y + half, x, half, leaves);
    tryBlock(ctx, y + half, x + half, half, leaves);
}

// ---------------------------------------------------------------------------
// Layout classification + split map (fp4_exporter.py _classify_layout /
// _build_split_map). Leaves arrive in pre-order DFS TL/TR/BL/BR order.
// ---------------------------------------------------------------------------

uint32_t classifyLayout(const std::vector<Leaf>& leaves) {
    if (leaves.empty()) {
        return kSGFP4LayoutMixed;
    }
    int size = leaves[0].size;
    for (const auto& leaf : leaves) {
        if (leaf.size != size) {
            return kSGFP4LayoutMixed;
        }
    }
    int expectedCount = (kMacroblockSize / size) * (kMacroblockSize / size);
    if (static_cast<int>(leaves.size()) != expectedCount) {
        return kSGFP4LayoutMixed;
    }
    switch (size) {
        case 64: return kSGFP4LayoutUniform64;
        case 32: return kSGFP4LayoutUniform32;
        case 16: return kSGFP4LayoutUniform16;
        case 8:  return kSGFP4LayoutUniform8;
        case kMinLeafSize: return kSGFP4LayoutFull4x4;
        default: return kSGFP4LayoutMixed;
    }
}

// Pre-order DFS bitmap consumed by the decoder walk (TL/TR/BL/BR).
void buildSplitMapBits(const std::vector<Leaf>& leaves, uint32_t (&words)[kSGFP4SplitMapWords]) {
    const int total = static_cast<int>(leaves.size());
    int leafIndex = 0;
    int bitIndex = 0;
    uint32_t bits[kSplitMapMaxBits];
    bool overflow = false;

    struct WalkFrame {
        int y, x, size;
    };

    // Iterative walk mirroring the Python recursive walk.
    WalkFrame stack[16];
    int top = 0;
    stack[top++] = WalkFrame{0, 0, kMacroblockSize};

    while (top > 0) {
        WalkFrame f = stack[--top];
        if (leafIndex >= total) {
            overflow = true;
            break;
        }
        const Leaf& leaf = leaves[static_cast<size_t>(leafIndex)];
        if (f.size == kMinLeafSize) {
            // Forced leaf; emits no bit. The leaf must match.
            leafIndex++;
            continue;
        }
        if (leaf.y == f.y && leaf.x == f.x && leaf.size == f.size) {
            if (bitIndex < kSplitMapMaxBits) {
                bits[bitIndex++] = 0; // leaf
            } else {
                overflow = true;
                break;
            }
            leafIndex++;
            continue;
        }
        // Split node.
        if (bitIndex < kSplitMapMaxBits) {
            bits[bitIndex++] = 1;
        } else {
            overflow = true;
            break;
        }
        int half = f.size / 2;
        // Push reverse so TL pops first.
        stack[top++] = WalkFrame{f.y + half, f.x + half, half};
        stack[top++] = WalkFrame{f.y, f.x + half, half};
        stack[top++] = WalkFrame{f.y + half, f.x, half};
        stack[top++] = WalkFrame{f.y, f.x, half};
    }
    (void)overflow;

    for (int i = 0; i < kSGFP4SplitMapWords; ++i) {
        words[i] = 0;
    }
    for (int i = 0; i < bitIndex; ++i) {
        if (bits[i] != 0) {
            words[i / 32] |= (1u << static_cast<uint32_t>(i % 32));
        }
    }
}

// ---------------------------------------------------------------------------
// Record + container assembly (fp4_exporter.py _export_v2_adaptive).
// ---------------------------------------------------------------------------

std::vector<uint8_t> assembleSuperblockRecord(const std::vector<Leaf>& leaves) {
    uint32_t layout = classifyLayout(leaves);

    // sb_header: layout enum in bits 0-2, reserved bits 3-31 written 0.
    std::vector<uint8_t> record;
    appendU32Le(record, layout & 0x7u);

    std::vector<Leaf> ordered;
    if (layout == kSGFP4LayoutMixed) {
        uint32_t words[kSGFP4SplitMapWords];
        buildSplitMapBits(leaves, words);
        for (int i = 0; i < kSGFP4SplitMapWords; ++i) {
            appendU32Le(record, words[i]);
        }
        ordered = leaves; // pre-order DFS traversal order
    } else {
        // Uniform layouts: leaves sorted row-major (y then x).
        // tryBlock emits TL/TR/BL/BR which for equal sizes is already
        // row-major, but sort defensively.
        ordered = leaves;
        std::sort(ordered.begin(), ordered.end(), [](const Leaf& a, const Leaf& b) {
            if (a.y != b.y) return a.y < b.y;
            return a.x < b.x;
        });
    }

    // Leaf headers.
    for (const auto& leaf : ordered) {
        appendU32Le(record, packLeafHeader(leaf.result.scale, leaf.result.bias, leaf.result.mode));
    }
    // Header section pads to a 16-byte boundary before payloads.
    if (record.size() % kSGFP4Alignment != 0) {
        record.resize(align16Size(record.size()), 0);
    }

    // Payloads (each already padded to a 16-byte multiple by packNibbles).
    for (const auto& leaf : ordered) {
        std::vector<uint8_t> payload = packNibbles(leaf.result.codes, leaf.result.mode);
        record.insert(record.end(), payload.begin(), payload.end());
    }

    // Whole record padded to a 16-byte multiple.
    if (record.size() % kSGFP4Alignment != 0) {
        record.resize(align16Size(record.size()), 0);
    }
    return record;
}

std::vector<uint8_t> assembleContainer(const std::vector<std::vector<uint8_t>>& records) {
    // Fixed header: magic(4) + version(1) + B(4) + pad0(7).
    std::vector<uint8_t> out;
    out.reserve(64);
    appendU32Le(out, kSGFP4Magic);
    out.push_back(static_cast<uint8_t>(kSGFP4Version));
    appendU32Le(out, static_cast<uint32_t>(records.size()));
    for (int i = 0; i < 7; ++i) {
        out.push_back(0);
    }
    // record_offsets table (relative to the aligned record-region base).
    const size_t offsetTableEnd = 16 + 4 * records.size();
    const size_t regionStart = align16Size(offsetTableEnd);
    std::vector<uint32_t> offsets(records.size(), 0);
    size_t running = 0;
    for (size_t i = 0; i < records.size(); ++i) {
        offsets[i] = static_cast<uint32_t>(running);
        running += records[i].size();
    }
    for (uint32_t off : offsets) {
        appendU32Le(out, off);
    }
    // pad1: zeros up to the aligned record-region base.
    while (out.size() < regionStart) {
        out.push_back(0);
    }
    for (const auto& rec : records) {
        out.insert(out.end(), rec.begin(), rec.end());
    }
    return out;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

std::vector<uint8_t> encode(const float* weights, int dimO, int dimI) {
    if (weights == nullptr || dimO <= 0 || dimI <= 0 || dimO > kMaxDim || dimI > kMaxDim) {
        return {};
    }

    // Security gate: reject NaN/Inf planes (ASVS V5; empty-vector contract).
    const size_t inputCount = static_cast<size_t>(dimO) * static_cast<size_t>(dimI);
    for (size_t i = 0; i < inputCount; ++i) {
        if (!std::isfinite(weights[i])) {
            return {};
        }
    }

    // Zero-pad to 64-multiples (fp4_exporter._export_v2_adaptive).
    const int paddedDimO = ((dimO + kMacroblockSize - 1) / kMacroblockSize) * kMacroblockSize;
    const int paddedDimI = ((dimI + kMacroblockSize - 1) / kMacroblockSize) * kMacroblockSize;
    const size_t paddedCount = static_cast<size_t>(paddedDimO) * static_cast<size_t>(paddedDimI);
    std::vector<double> plane(paddedCount, 0.0);
    for (int r = 0; r < dimO; ++r) {
        for (int c = 0; c < dimI; ++c) {
            plane[static_cast<size_t>(r) * paddedDimI + c] = static_cast<double>(weights[static_cast<size_t>(r) * dimI + c]);
        }
    }

    const int tilesY = (paddedDimO + kMacroblockSize - 1) / kMacroblockSize;
    const int tilesX = (paddedDimI + kMacroblockSize - 1) / kMacroblockSize;

    QuadtreeContext ctx;
    ctx.plane = plane.data();
    ctx.paddedDimI = paddedDimI;

    std::vector<std::vector<uint8_t>> records;
    records.reserve(static_cast<size_t>(tilesY) * tilesX);
    for (int sbR = 0; sbR < tilesY; ++sbR) {
        for (int sbC = 0; sbC < tilesX; ++sbC) {
            std::vector<Leaf> leaves;
            tryBlock(ctx, sbR * kMacroblockSize, sbC * kMacroblockSize, kMacroblockSize, leaves);
            records.push_back(assembleSuperblockRecord(leaves));
        }
    }

    return assembleContainer(records);
}

} // namespace sgfp4_encode
