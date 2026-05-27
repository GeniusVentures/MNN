//
//  FP4DequantUtils.hpp
//  MNN
//
//  Created by MNN on 2026/05/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef FP4DequantUtils_hpp
#define FP4DequantUtils_hpp

#include <cmath>
#include <cstdint>

namespace MNN {

/**
 * @brief Decode a single E2M1-encoded FP4 nibble to IEEE 754 float.
 *
 * E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit (bias = 1).
 * Subnormal (e == 0): (-1)^s × m × 0.5
 * Normal   (e == 1,2): (-1)^s × 2^(e-1) × (1 + m/2)
 * Special  (e == 3): m == 0 → Inf, m == 1 → NaN
 *
 * Known E2M1 test vectors:
 * ┌──────────────┬────────────────┬────────┐
 * │ Nibble (hex) │ Binary (s|ee|m)│ Value  │
 * ├──────────────┼────────────────┼────────┤
 * │ 0x0          │ 0|00|0         │ +0.0   │
 * │ 0x1          │ 0|00|1         │ +0.5   │
 * │ 0x2          │ 0|01|0         │ +1.0   │
 * │ 0x3          │ 0|01|1         │ +1.5   │
 * │ 0x4          │ 0|10|0         │ +2.0   │
 * │ 0x5          │ 0|10|1         │ +3.0   │
 * │ 0x6          │ 0|11|0         │ +Inf   │
 * │ 0x7          │ 0|11|1         │ NaN    │
 * │ 0x8          │ 1|00|0         │ -0.0   │
 * │ 0x9          │ 1|00|1         │ -0.5   │
 * │ 0xA          │ 1|01|0         │ -1.0   │
 * │ 0xB          │ 1|01|1         │ -1.5   │
 * │ 0xC          │ 1|10|0         │ -2.0   │
 * │ 0xD          │ 1|10|1         │ -3.0   │
 * │ 0xE          │ 1|11|0         │ -Inf   │
 * │ 0xF          │ 1|11|1         │ NaN    │
 * └──────────────┴────────────────┴────────┘
 *
 * @param nibble  The 4-bit E2M1 value (lower 4 bits used, bits 4-7 ignored).
 * @return        Decoded IEEE 754 float value.
 */
inline float dequant_e2m1_cpu(uint8_t nibble) {
    uint8_t s = (nibble >> 3) & 0x1;
    uint8_t e = (nibble >> 1) & 0x3;
    uint8_t m = nibble & 0x1;

    float sign = (s == 1) ? -1.0f : 1.0f;

    if (e == 0) {
        // Subnormal: (-1)^s × m × 0.5 (bias = 1)
        return sign * static_cast<float>(m) * 0.5f;
    }

    if (e == 3) {
        // Special: Inf (m = 0) or NaN (m = 1)
        return (m == 0) ? sign * INFINITY : NAN;
    }

    // Normal: (-1)^s × 2^(e-1) × (1 + m/2)
    float expVal = static_cast<float>(1 << (e - 1));
    float mantissa = 1.0f + static_cast<float>(m) * 0.5f;
    return sign * expVal * mantissa;
}

/**
 * @brief Pack two FP4 nibbles into a single byte (little-endian nibble order).
 *
 * Low nibble  (bits 0-3): first FP4 value  (even element indices).
 * High nibble (bits 4-7): second FP4 value (odd element indices).
 *
 * This matches the swizzling used by fp4_dequant.comp (D-03 packing layout).
 *
 * @param low_nibble   The first FP4 value (stored in bits 0-3).
 * @param high_nibble  The second FP4 value (stored in bits 4-7).
 * @return             Packed byte with two FP4 values.
 */
inline uint8_t pack_fp4_byte(uint8_t low_nibble, uint8_t high_nibble) {
    return (low_nibble & 0x0F) | ((high_nibble & 0x0F) << 4);
}

/**
 * @brief Dequantize an entire packed FP4 buffer to float.
 *
 * Unpacks 2 FP4 values per byte: even indices from low nibble,
 * odd indices from high nibble. Each nibble is decoded via
 * dequant_e2m1_cpu().
 *
 * @param packed        Pointer to packed byte buffer (2 FP4 values per byte).
 * @param output        Pointer to output float array (must have elementCount elements).
 * @param elementCount  Number of float values to produce.
 */
inline void dequant_fp4_packed_cpu(const uint8_t* packed, float* output, size_t elementCount) {
    for (size_t i = 0; i < elementCount; ++i) {
        size_t byteIdx = i >> 1;
        uint8_t byteVal = packed[byteIdx];
        uint8_t nibble = (i & 1) ? (byteVal >> 4) : (byteVal & 0x0F);
        output[i] = dequant_e2m1_cpu(nibble);
    }
}

} // namespace MNN

#endif /* FP4DequantUtils_hpp */
