#!/usr/bin/env python3
"""
Regression test for the MAX_E2M1_VALUE scale-calibration bug in quantize_fp4.py.

Per .planning/quick/260821-p1q-evaluate-current-fp4-ultra-fp4-implement/
SGFP4-PIVOT-ANALYSIS.md Section 1: MAX_E2M1_VALUE was previously 6.0, which
guaranteed that every channel's max-magnitude weight normalized to exactly
6.0 and saturated to +/-Inf in encode_fp4_e2m1 (since E2M1's largest finite
magnitude is 3.0, and any normalized magnitude >= 4.0 saturates to +/-Inf).
This suite proves the fix (MAX_E2M1_VALUE == 3.0) and guards against the
constant ever silently regressing back to 6.0 (or any other wrong value).

Run directly with:
    python3 tools/fp4/test_quantize_fp4.py -v

No MNNConvert, no built MNN binaries, and no model files are required.
"""

import math
import os
import sys
import unittest

# Ensure quantize_fp4 is importable regardless of the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import quantize_fp4 as q

# Ground-truth E2M1 test-vector table, transcribed exactly from the
# doc-comment table in include/MNN/FP4DequantUtils.hpp's dequant_e2m1_cpu.
E2M1_TABLE = {
    0x0: 0.0,
    0x1: 0.5,
    0x2: 1.0,
    0x3: 1.5,
    0x4: 2.0,
    0x5: 3.0,
    0x6: math.inf,
    0x7: math.nan,
    0x8: -0.0,
    0x9: -0.5,
    0xA: -1.0,
    0xB: -1.5,
    0xC: -2.0,
    0xD: -3.0,
    0xE: -math.inf,
    0xF: math.nan,
}


def dequant_e2m1(nibble):
    """Pure-Python mirror of dequant_e2m1_cpu from FP4DequantUtils.hpp.

    Maintained for test purposes only -- keep in sync with the C++
    implementation in include/MNN/FP4DequantUtils.hpp.
    """
    s = (nibble >> 3) & 0x1
    e = (nibble >> 1) & 0x3
    m = nibble & 0x1

    sign = -1.0 if s == 1 else 1.0

    if e == 0:
        # Subnormal: (-1)^s * m * 0.5 (bias = 1)
        return sign * float(m) * 0.5

    if e == 3:
        # Special: Inf (m == 0) or NaN (m == 1)
        return sign * math.inf if m == 0 else math.nan

    # Normal: (-1)^s * 2^(e-1) * (1 + m/2)
    exp_val = float(1 << (e - 1))
    mantissa = 1.0 + float(m) * 0.5
    return sign * exp_val * mantissa


def _unpack_nibble(packed, index):
    """Unpack the nibble for element `index`, matching pack_fp4_byte's
    low-nibble-even/high-nibble-odd convention."""
    byte_val = packed[index >> 1]
    return (byte_val >> 4) if (index & 1) else (byte_val & 0x0F)


class MaxE2M1ValueCalibrationTest(unittest.TestCase):
    def test_python_decode_mirror_matches_ground_truth_table(self):
        """Validate the Python mirror's fidelity before trusting it to
        validate quantize_fp4.py's encoder."""
        for nibble, expected in E2M1_TABLE.items():
            actual = dequant_e2m1(nibble)
            if math.isnan(expected):
                self.assertTrue(
                    math.isnan(actual),
                    "nibble 0x%x: expected NaN, got %r" % (nibble, actual),
                )
            else:
                self.assertEqual(
                    actual, expected,
                    "nibble 0x%x: expected %r, got %r" % (nibble, expected, actual),
                )

    def test_max_e2m1_value_equals_true_max_finite_magnitude(self):
        """Direct regression guard: fails immediately if MAX_E2M1_VALUE is
        ever changed back to 6.0 or any other wrong value."""
        true_max_finite = max(v for v in E2M1_TABLE.values() if math.isfinite(v))
        self.assertEqual(true_max_finite, 3.0)
        self.assertEqual(q.MAX_E2M1_VALUE, true_max_finite)

    def test_channel_max_magnitude_weight_round_trips_finite(self):
        """Core defect guard: previously the max-magnitude element always
        decoded to +/-Inf. Two elements (3.7 and -3.7) tie for max
        magnitude here -- check both."""
        channel = [0.5, -1.2, 3.7, -3.7, 2.0, 0.0]
        packed, scale = q.quantize_channel_weights(channel)

        round_tripped = []
        for i in range(len(channel)):
            nibble = _unpack_nibble(packed, i)
            value = dequant_e2m1(nibble) * scale
            round_tripped.append(value)
            self.assertTrue(
                math.isfinite(value),
                "element %d (%.4f) round-tripped to non-finite value %r"
                % (i, channel[i], value),
            )

        max_abs = max(abs(w) for w in channel)
        for i, orig in enumerate(channel):
            if abs(orig) == max_abs:
                self.assertAlmostEqual(
                    abs(round_tripped[i]), max_abs, delta=1e-6,
                    msg="max-magnitude element %d did not round-trip precisely" % i,
                )

    def test_multiple_channels_never_saturate_to_inf(self):
        """Representative channels covering different shapes and
        magnitudes must never produce +/-Inf or NaN after round-trip."""
        channels = [
            [4.2],  # single-element channel
            [0.1, 0.2, 0.3, 0.05],  # several small positive floats
            [-1.0, 2.5, -0.75, 0.0, 5.0],  # mixed-sign channel
            [1000.0, -2500.0, 750.0, -10.0],  # large-magnitude values
        ]
        for channel in channels:
            packed, scale = q.quantize_channel_weights(channel)
            for i in range(len(channel)):
                nibble = _unpack_nibble(packed, i)
                value = dequant_e2m1(nibble) * scale
                self.assertFalse(
                    math.isinf(value),
                    "channel %r element %d produced +/-Inf" % (channel, i),
                )
                self.assertFalse(
                    math.isnan(value),
                    "channel %r element %d produced NaN" % (channel, i),
                )


if __name__ == "__main__":
    unittest.main()
