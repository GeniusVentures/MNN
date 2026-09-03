#!/usr/bin/env python3
"""
FP4 E2M1 Model Weight Quantization Tool

Converts float .mnn model Conv2D/InnerProduct weights to FP4 E2M1 format.
Produces .mnn models with symmetricQuan (nbits=4) weight encoding that the
Vulkan and CPU FP4 dequant runtimes can execute via OpType_Dequantize.

FP4 E2M1 format: 1 sign | 2 exponent (bias=1) | 1 mantissa
  Subnormal (e=0): (-1)^s * m * 0.5
  Normal   (e=1,2): (-1)^s * 2^(e-1) * (1 + m/2)
  Special  (e=3): m=0 Inf, m=1 NaN
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

import math

# Normalizing a channel's max-magnitude weight by dividing by MAX_E2M1_VALUE
# makes that weight's normalized value equal exactly MAX_E2M1_VALUE. E2M1 can
# only represent magnitudes < 4.0 (biased_e >= 3, i.e. magnitude >= 4.0,
# saturates to +/-Inf in encode_fp4_e2m1 below) as finite values, and the
# largest finite magnitude it can encode is 3.0 (nibble 0x5/0xD: biased_e=2,
# m=1 -> 2^(2-1)*(1+0.5) = 3.0; see FP4DequantUtils.hpp's test-vector table).
# A divisor of 6.0 made every channel's largest-magnitude weight normalize to
# exactly 6.0, which always fell into the >= 4.0 saturation branch, so every
# channel's max-magnitude weight was quantized to +/-Inf on every run.
MAX_E2M1_VALUE = 3.0


def encode_fp4_e2m1(val):
    """Encode IEEE 754 float to E2M1 4-bit nibble."""
    if np.isnan(val) or val is None:
        return 0x07  # positive NaN
    s = 1 if math.copysign(1.0, val) < 0 else 0
    val = abs(val)
    if val == 0.0:
        return s << 3  # positive or negative zero
    if np.isinf(val):
        return 0x06 | (s << 3)  # Inf
    e = int(np.floor(np.log2(val)))
    biased_e = e + 1  # bias = 1
    if biased_e >= 3:
        # Magnitudes with biased_e >= 3 (i.e. >= 4.0) have no finite E2M1
        # encoding and saturate to +/-Inf. This is unchanged by the
        # MAX_E2M1_VALUE fix above -- the largest finite-magnitude code
        # E2M1 can represent remains 3.0 (nibble 0x5/0xD).
        return 0x06 | (s << 3)  # saturate to +/-Inf
    if biased_e <= 0:
        m_bit = int(round(val / 0.5)) & 0x1
        return (s << 3) | m_bit
    m_bit = int(round((val / float(1 << e) - 1.0) * 2.0)) & 0x1
    return (s << 3) | (biased_e << 1) | m_bit


def pack_fp4_byte(low_nibble, high_nibble):
    """Pack two FP4 nibbles into one byte, little-endian (low nibble first)."""
    return (low_nibble & 0x0F) | ((high_nibble & 0x0F) << 4)


def quantize_channel_weights(channel_weights):
    """Quantize a list of floats for one output channel to packed FP4 bytes."""
    n = len(channel_weights)
    packed = bytearray((n + 1) // 2)
    max_abs = max(abs(w) for w in channel_weights)
    scale = max_abs / MAX_E2M1_VALUE if max_abs > 0.0 else 1.0
    if scale == 0.0:
        scale = 1.0
    for i in range(n):
        quantized_val = channel_weights[i] / scale
        nibble = encode_fp4_e2m1(quantized_val)
        byte_idx = i >> 1
        if (i & 1) == 0:
            packed[byte_idx] = (packed[byte_idx] & 0xF0) | (nibble & 0x0F)
        else:
            packed[byte_idx] = (packed[byte_idx] & 0x0F) | ((nibble & 0x0F) << 4)
    return bytes(packed), scale


def find_mnnconvert():
    """Locate the MNNConvert binary."""
    candidates = [
        "./MNNConvert",
        "./build/MNNConvert",
        "MNNConvert",
    ]
    for c in candidates:
        if shutil_which(c):
            return c
    raise RuntimeError(
        "MNNConvert not found. Build MNN with -DMNN_BUILD_CONVERTER=ON or set PATH."
    )


def shutil_which(name):
    """Cross-platform which() equivalent."""
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def quantize_model(input_path, output_path, bits=4):
    """Main quantization pipeline: MNN -> JSON, quantize weights, JSON -> MNN."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input model not found: {input_path}")
    if bits != 4:
        raise ValueError(f"Only 4-bit (FP4 E2M1) quantization is supported, got {bits}")

    try:
        mnnconvert = find_mnnconvert()
    except RuntimeError:
        sys.stderr.write("WARNING: MNNConvert not found. Running JSON-only mode.\n")
        sys.stderr.write("Install MNNConvert or add it to PATH for full .mnn output.\n")
        mnnconvert = None

    tmp_json_input = tempfile.mktemp(suffix=".json")
    tmp_json_output = tempfile.mktemp(suffix=".json")

    try:
        if mnnconvert:
            cmd = [mnnconvert, "-f", "MNN", "--modelFile", input_path, "--jsonFile", tmp_json_input]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            sys.stderr.write("Skipping MNN -> JSON step (no MNNConvert).\n")
            sys.stderr.write("Provide a .json model file directly with --input.\n")
            if input_path.endswith(".json"):
                with open(input_path, "r") as f:
                    data = json.load(f)
                packed_bytes, scales = quantize_model_json(data)
                _write_output_json(data, tmp_json_output)
                _write_quant_stats(tmp_json_output, packed_bytes, scales)
                return
            sys.exit(1)

        with open(tmp_json_input, "r") as f:
            data = json.load(f)

        total_ops = 0
        quantized_ops = 0
        total_orig_bytes = 0
        total_packed_bytes = 0

        for op in data.get("oplists", []):
            op_type = op.get("type", "")
            if op_type not in ("Convolution", "InnerProduct"):
                continue
            total_ops += 1
            main = op.get("main", {})
            weights = main.get("weight")
            if not weights or not isinstance(weights, list) or len(weights) == 0:
                continue

            if op_type == "Convolution":
                common = main.get("common", {})
                oc = common.get("outputCount", 0)
            else:
                oc = main.get("outputCount", 0)

            if oc <= 0:
                continue

            total_floats = len(weights)
            channel_size = total_floats // oc if oc > 0 else total_floats

            all_packed = bytearray()
            all_scales = []

            for c in range(oc):
                start = c * channel_size
                end = start + channel_size
                channel_w = weights[start:end]
                packed_channel, scale = quantize_channel_weights(channel_w)
                all_packed.extend(packed_channel)
                all_scales.append(float(scale))

            packed_int_list = [int(b) for b in all_packed]

            main["symmetricQuan"] = {
                "nbits": 4,
                "weight": packed_int_list,
                "scale": all_scales,
                "bias": [0.0] * oc,
                "outputDataType": 0,
                "clampMin": 0,
                "clampMax": 0,
            }
            del main["weight"]
            quantized_ops += 1
            total_orig_bytes += total_floats * 4
            total_packed_bytes += len(all_packed)

        _write_output_json(data, tmp_json_output)

        if mnnconvert:
            final_cmd = [
                mnnconvert, "-f", "JSON", "--modelFile", tmp_json_output,
                "--MNNModel", output_path,
            ]
            subprocess.run(final_cmd, check=True, capture_output=True, text=True)
        else:
            with open(tmp_json_output, "r") as src:
                with open(output_path, "w") as dst:
                    dst.write(src.read())
            sys.stderr.write(f"Wrote JSON output to {output_path}\n")

        _write_quant_stats(output_path, total_packed_bytes, all_scales,
                          total_ops=total_ops, quantized_ops=quantized_ops,
                          orig_bytes=total_orig_bytes)

    finally:
        for f in [tmp_json_input, tmp_json_output]:
            if os.path.isfile(f):
                os.remove(f)


def _write_output_json(data, path):
    """Write modified model JSON to file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_quant_stats(output_path, packed_bytes, scales,
                       total_ops=0, quantized_ops=0, orig_bytes=0):
    """Print quantization statistics."""
    if total_ops > 0:
        ratio = (1.0 - float(packed_bytes) / float(orig_bytes)) * 100.0 if orig_bytes > 0 else 0.0
        sys.stderr.write(f"Quantization complete: {quantized_ops}/{total_ops} ops quantized\n")
        sys.stderr.write(f"  Original weight bytes: {orig_bytes}\n")
        sys.stderr.write(f"  Packed FP4 bytes: {packed_bytes}\n")
        sys.stderr.write(f"  Size reduction: {ratio:.1f}%\n")
    sys.stderr.write(f"Output: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert float .mnn model weights to FP4 E2M1 format"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input float .mnn model file"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output FP4-quantized .mnn model file"
    )
    parser.add_argument(
        "--bits", type=int, default=4,
        help="Quantization bits (default: 4, only 4-bit FP4 supported)"
    )
    parser.add_argument(
        "--sym", type=bool, default=True,
        help="Use symmetric quantization (default: True)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.stderr.write(f"ERROR: Input file not found: {args.input}\n")
        sys.exit(1)

    quantize_model(args.input, args.output, bits=args.bits)


if __name__ == "__main__":
    main()
