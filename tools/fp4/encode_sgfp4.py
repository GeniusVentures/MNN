#!/usr/bin/env python3
"""
SGFP4 v2 Reference Encoder (uniform layouts only)

Standalone reference/test-oracle encoder for the SGFP4 v2 self-framed weight
container (sgfp4-arxiv-v2.txt, Sections 3.2, 4.3, 4.4, 6.1, 6.2). Implements
the spec's *exemplary* affine encode for both code modes plus per-block
(per-leaf) mode selection, and writes v2 containers restricted to the five
Table 3 "uniform" layouts (LAYOUT_MIXED / quadtree is out of scope -- Phase 2).

This file is intentionally independent of tools/fp4/quantize_fp4.py (the
existing E2M1 encoder): it does not import or modify it. SGFP4 v2 is an
additive format with completely different decode math (affine integer
codes, not E2M1 float microcodes).

The encoder doubles as the Python-side oracle for the C++ round-trip tests:
`--selftest` proves encode->decode is invertible (independent Python
reference decoder mirrors MNN::dequant_sgfp4_container_cpu), and
`--emit-cpp-fixture` writes a committed C++ header of encoder-produced
containers + expected weights for test/op/SGFP4DequantTest.cpp to consume.

Usage:
    python tools/fp4/encode_sgfp4.py --selftest
    python tools/fp4/encode_sgfp4.py --emit-cpp-fixture test/op/SGFP4DequantFixtures.h
"""

import argparse
import struct
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Container format constants (mirrors include/MNN/SGFP4DequantUtils.hpp).
# All multi-byte container fields are little-endian.
# ---------------------------------------------------------------------------

SGFP4_MAGIC = ord('S') | (ord('G') << 8) | (ord('F') << 16) | (ord('4') << 24)
SGFP4_VERSION = 0x02

FIXED_HEADER_SIZE = 16          # magic(4) + version(1) + B(4) + pad0(7)
VERSION_BYTE_OFFSET = 4         # right after magic
RECORD_COUNT_OFFSET = 5         # right after magic+version
RECORD_OFFSET_TABLE_START = FIXED_HEADER_SIZE
RECORD_OFFSET_ENTRY_SIZE = 4    # each record_offsets[b] is a u32
ALIGNMENT = 16                  # 16-byte alignment throughout

# Per-leaf FP16 header packing (spec section 6.2, Eq. 6):
# S = half(h >> 16); bias = half(h & 0xFFF0); mode = (h & 0xF) & 0x1.
LEAF_HEADER_SCALE_SHIFT = 16
LEAF_HEADER_BIAS_MASK = 0xFFF0
LEAF_HEADER_MODE_BIT = 0x1

# Dual-mode payload packing (spec section 4.3, Eq. 3/4).
NIBBLES_PER_WORD = 8   # mode 0: 8 x 4-bit codes / u32 word
SYMBOLS_PER_WORD = 16  # mode 1: 16 x 2-bit codes / u32 word
NIBBLE_BIT_WIDTH = 4
SYMBOL_BIT_WIDTH = 2
NIBBLE_MASK = 0xF
SYMBOL_MASK = 0x3
TWOS_COMPLEMENT_SIGN_BIAS = 0x8

LAYOUT_ENUM_MASK = 0x7  # sb_header bits 0-2 (spec section 6.2)

# Table 3 uniform-layout map: layout_enum -> (leafCount N, leaf edge n).
LAYOUT_UNIFORM_64 = 0
LAYOUT_UNIFORM_32 = 1
LAYOUT_UNIFORM_16 = 2
LAYOUT_UNIFORM_8 = 3
LAYOUT_MIXED = 4       # quadtree -- Phase 2, not emitted by this encoder
LAYOUT_FULL_4X4 = 5

LAYOUT_TABLE = {
    LAYOUT_UNIFORM_64: (1, 64),
    LAYOUT_UNIFORM_32: (4, 32),
    LAYOUT_UNIFORM_16: (16, 16),
    LAYOUT_UNIFORM_8: (64, 8),
    LAYOUT_FULL_4X4: (256, 4),
}

MACROBLOCK_EDGE = 64  # every uniform layout tiles one 64x64 macroblock exactly

MODE_FP4_AFFINE = 0
MODE_T158_AFFINE = 1

FP16_MAX = 65504.0  # spec section 3.2: S and bias clipped to +/-65504 at encode time

# Exemplary v1/v2 encoder policy (spec section 4.4).
S_SEARCH_CANDIDATES = 16
S_SEARCH_LOW_FACTOR = 0.5
S_SEARCH_HIGH_FACTOR = 1.5
S_SEARCH_DIVISOR = 7.0        # candidates span [0.5, 1.5] * maxi|wi| / 7
MODE_SELECT_EPS = 0.10        # default eps in [0.05, 0.20] (Eq. 5)
FP4_CODE_MIN = -8
FP4_CODE_MAX = 7


def align16(x):
    """Round `x` up to the next multiple of the container's 16-byte alignment."""
    return (x + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1)


def clip_fp16_range(x):
    """Clip a float to the FP16-representable range (spec section 3.2)."""
    return max(-FP16_MAX, min(FP16_MAX, x))


def float_to_half_bits(x):
    """Convert a Python float to its IEEE-754 binary16 bit pattern (as an int)."""
    half = np.array([x], dtype=np.float32).astype(np.float16)
    return int(half.view(np.uint16)[0])


def half_bits_to_float(bits):
    """Convert an IEEE-754 binary16 bit pattern back to a Python float."""
    arr = np.array([bits & 0xFFFF], dtype=np.uint16)
    return float(arr.view(np.float16)[0])


# ---------------------------------------------------------------------------
# Per-leaf encode (spec section 4.4)
# ---------------------------------------------------------------------------

def encode_leaf_fp4(w):
    """FP4_AFFINE (mode 0): bias = mean(w); S from a 16-candidate log search
    over [0.5, 1.5] * maxi|wi|/7 minimizing round-trip MSE; codes clipped to
    [-8, 7]. Returns (S, bias, codes:int32[n*n], l2_err)."""
    bias = float(np.mean(w))
    maxabs = float(np.max(np.abs(w)))
    if maxabs == 0.0:
        # Degenerate constant-zero leaf: any nonzero S works since every code is 0.
        return 1.0, bias, np.zeros(w.shape, dtype=np.int32), 0.0

    base = maxabs / S_SEARCH_DIVISOR
    candidates = np.geomspace(S_SEARCH_LOW_FACTOR * base, S_SEARCH_HIGH_FACTOR * base,
                               num=S_SEARCH_CANDIDATES)
    best_err = None
    best_S = float(candidates[0])
    best_codes = None
    for S in candidates:
        codes = np.clip(np.round((w - bias) / S), FP4_CODE_MIN, FP4_CODE_MAX)
        recon = codes * S + bias
        err = float(np.sum((recon - w) ** 2))
        if best_err is None or err < best_err:
            best_err = err
            best_S = float(S)
            best_codes = codes
    return best_S, bias, best_codes.astype(np.int32), best_err


def encode_leaf_t158(w):
    """T158_AFFINE (mode 1): bias = mean(w); S = mean|w - bias|; codes assigned
    by thresholding at tau = S/2 -> {-1, 0, +1}. Returns (S, bias, codes, l2_err)."""
    bias = float(np.mean(w))
    d = w - bias
    S = float(np.mean(np.abs(d)))
    if S == 0.0:
        S = 1.0
    tau = S / 2.0
    codes = np.where(np.abs(d) < tau, 0, np.sign(d)).astype(np.int32)
    recon = codes.astype(np.float64) * S + bias
    err = float(np.sum((recon - w) ** 2))
    return S, bias, codes, err


def select_mode(err_fp4, err_t158, eps=MODE_SELECT_EPS):
    """Eq. 5: choose T158 iff e_T158 <= (1 + eps) * e_FP4."""
    return MODE_T158_AFFINE if err_t158 <= (1.0 + eps) * err_fp4 else MODE_FP4_AFFINE


def encode_leaf(w, force_mode=None):
    """Encode one leaf with both modes and select per Eq. 5 (or use
    `force_mode` to pin a mode, e.g. for generating single-mode test
    fixtures -- the underlying per-mode math is identical either way).
    Returns (mode, S, bias, codes)."""
    s_fp4, b_fp4, c_fp4, e_fp4 = encode_leaf_fp4(w)
    s_t158, b_t158, c_t158, e_t158 = encode_leaf_t158(w)
    mode = force_mode if force_mode is not None else select_mode(e_fp4, e_t158)
    if mode == MODE_FP4_AFFINE:
        return mode, s_fp4, b_fp4, c_fp4
    return mode, s_t158, b_t158, c_t158


# ---------------------------------------------------------------------------
# Packing (spec sections 4.3, 6.1, 6.2 -- inverse of SGFP4DequantUtils.hpp)
# ---------------------------------------------------------------------------

def pack_leaf_header(mode, S, bias):
    """Pack (mode, S, bias) into the leaf header word (Eq. 6, inverse).
    Returns (header_word, S_bits, bias_bits_masked) -- the bits are returned
    too so callers can compute the exact FP16-truncated (S, bias) the decoder
    will recover, for building bit-exact 'expected' fixture values."""
    S_bits = float_to_half_bits(clip_fp16_range(S))
    bias_bits_masked = float_to_half_bits(clip_fp16_range(bias)) & LEAF_HEADER_BIAS_MASK
    header = ((S_bits << LEAF_HEADER_SCALE_SHIFT) | bias_bits_masked | (mode & LEAF_HEADER_MODE_BIT))
    return header & 0xFFFFFFFF, S_bits, bias_bits_masked


def pack_payload(mode, codes):
    """Pack a leaf's codes into little-endian uint32 words (spec section 4.3),
    padded to a 16-byte multiple."""
    n2 = codes.shape[0]
    if mode == MODE_FP4_AFFINE:
        per_word = NIBBLES_PER_WORD
        bitwidth = NIBBLE_BIT_WIDTH
        values = [int(c) & NIBBLE_MASK for c in codes]  # 4-bit two's complement
    else:
        per_word = SYMBOLS_PER_WORD
        bitwidth = SYMBOL_BIT_WIDTH

        def sym_of(c):
            if c == 1:
                return 0b01
            if c == -1:
                return 0b10
            return 0b00  # 0 -> 00

        values = [sym_of(int(c)) for c in codes]

    num_words = n2 // per_word
    words = [0] * num_words
    for i, v in enumerate(values):
        words[i // per_word] |= (v << (bitwidth * (i % per_word)))

    payload = b''.join(struct.pack('<I', w) for w in words)
    pad_len = align16(len(payload)) - len(payload)
    return payload + b'\x00' * pad_len


def encode_macroblock(leaves_weights, layout_enum, force_mode=None):
    """Encode one 64x64 macroblock's leaves into a v2 record (spec section
    6.2). `leaves_weights` is a list of N flat float32 arrays, each n*n long,
    in row-major raster order of the tile grid. Returns (record_bytes,
    expected:float32[N*n*n]) where `expected` is the exact reconstruction the
    decoder will produce (S/bias already FP16-truncated as the container
    stores them)."""
    N, n = LAYOUT_TABLE[layout_enum]
    if len(leaves_weights) != N:
        raise ValueError(f"layout {layout_enum} expects {N} leaves, got {len(leaves_weights)}")

    header_words = []
    payload_chunks = []
    expected_chunks = []
    for leaf_w in leaves_weights:
        mode, S, bias, codes = encode_leaf(leaf_w, force_mode=force_mode)
        header, S_bits, bias_bits_masked = pack_leaf_header(mode, S, bias)
        header_words.append(header)
        payload_chunks.append(pack_payload(mode, codes))

        S_used = half_bits_to_float(S_bits)
        bias_used = half_bits_to_float(bias_bits_masked)
        recon = (np.float32(S_used) * codes.astype(np.float32) + np.float32(bias_used))
        expected_chunks.append(recon.astype(np.float32))

    sb_header = layout_enum & LAYOUT_ENUM_MASK
    block_headers_bytes = b''.join(struct.pack('<I', h) for h in header_words)
    pre_payload_len = 4 + len(block_headers_bytes)
    pad_len = align16(pre_payload_len) - pre_payload_len
    record = struct.pack('<I', sb_header) + block_headers_bytes + b'\x00' * pad_len
    for chunk in payload_chunks:
        record += chunk

    expected = np.concatenate(expected_chunks) if expected_chunks else np.zeros((0,), dtype=np.float32)
    return record, expected


def encode_container(macroblocks):
    """Encode a full v2 container (spec section 6.1) from a list of
    (leaves_weights, layout_enum, force_mode) macroblock tuples, in row-major
    macroblock order. Returns (container_bytes, expected:float32[...]) where
    `expected` is the linear-order concatenation of every macroblock's
    reconstruction (matches dequant_sgfp4_container_cpu's fully sequential
    decode order: record 0's leaves fill the first span, record 1's the
    next, etc.)."""
    B = len(macroblocks)
    records = []
    expected_parts = []
    for leaves_weights, layout_enum, force_mode in macroblocks:
        record, expected = encode_macroblock(leaves_weights, layout_enum, force_mode)
        records.append(record)
        expected_parts.append(expected)

    offsets = []
    cursor = 0
    for r in records:
        offsets.append(cursor)
        cursor += len(r)
        assert cursor % ALIGNMENT == 0, "record length must stay a 16-byte multiple"

    header = struct.pack('<4sB', b'SGF4', SGFP4_VERSION) + struct.pack('<I', B)
    header += b'\x00' * (FIXED_HEADER_SIZE - len(header))  # pad0

    offset_table = b''.join(struct.pack('<I', o) for o in offsets)
    pre_region_len = FIXED_HEADER_SIZE + len(offset_table)
    pad1_len = align16(pre_region_len) - pre_region_len

    container = bytearray(header + offset_table + b'\x00' * pad1_len)
    for r in records:
        container += r

    expected = (np.concatenate(expected_parts) if expected_parts
                else np.zeros((0,), dtype=np.float32))
    return bytes(container), expected


# ---------------------------------------------------------------------------
# Independent Python reference decoder (mirrors
# MNN::dequant_sgfp4_container_cpu in include/MNN/SGFP4DequantUtils.hpp) --
# used by --selftest to prove the *encoded bytes* round-trip, not just the
# in-process arrays produced while encoding.
# ---------------------------------------------------------------------------

def _read_u32_le(buf, offset):
    return struct.unpack_from('<I', buf, offset)[0]


def unpack_leaf_header_ref(h):
    s_bits = (h >> LEAF_HEADER_SCALE_SHIFT) & 0xFFFF
    bias_bits = h & LEAF_HEADER_BIAS_MASK
    S = half_bits_to_float(s_bits)
    bias = half_bits_to_float(bias_bits)
    mode = h & LEAF_HEADER_MODE_BIT
    return S, bias, mode


def _decode_leaf_payload_ref(payload_bytes, leaf_edge, S, bias, mode):
    n2 = leaf_edge * leaf_edge
    out = np.zeros(n2, dtype=np.float32)
    if mode == MODE_FP4_AFFINE:
        num_words = n2 // NIBBLES_PER_WORD
        words = struct.unpack_from('<%dI' % num_words, payload_bytes, 0)
        for i in range(n2):
            w = words[i // NIBBLES_PER_WORD]
            shift = NIBBLE_BIT_WIDTH * (i % NIBBLES_PER_WORD)
            nib = (w >> shift) & NIBBLE_MASK
            c = (nib ^ TWOS_COMPLEMENT_SIGN_BIAS) - TWOS_COMPLEMENT_SIGN_BIAS
            out[i] = np.float32(S) * np.float32(c) + np.float32(bias)
    else:
        num_words = n2 // SYMBOLS_PER_WORD
        words = struct.unpack_from('<%dI' % num_words, payload_bytes, 0)
        for i in range(n2):
            w = words[i // SYMBOLS_PER_WORD]
            shift = SYMBOL_BIT_WIDTH * (i % SYMBOLS_PER_WORD)
            sym = (w >> shift) & SYMBOL_MASK
            c = 1 if sym == 1 else (-1 if sym == 2 else 0)
            out[i] = np.float32(S) * np.float32(c) + np.float32(bias)
    return out


def decode_container_ref(container):
    """Independent Python reference decode of a v2 uniform-layout container.
    Mirrors dequant_sgfp4_container_cpu byte-for-byte; raises AssertionError
    on malformed framing (selftest only feeds well-formed containers)."""
    size = len(container)
    assert size >= FIXED_HEADER_SIZE
    magic = _read_u32_le(container, 0)
    assert magic == SGFP4_MAGIC, f"bad magic 0x{magic:08x}"
    version = container[VERSION_BYTE_OFFSET]
    assert version == SGFP4_VERSION, f"bad version 0x{version:02x}"
    B = _read_u32_le(container, RECORD_COUNT_OFFSET)
    region_start = align16(RECORD_OFFSET_TABLE_START + RECORD_OFFSET_ENTRY_SIZE * B)

    out_parts = []
    for b in range(B):
        off_entry = RECORD_OFFSET_TABLE_START + b * RECORD_OFFSET_ENTRY_SIZE
        rec_off_rel = _read_u32_le(container, off_entry)
        rec_start = region_start + rec_off_rel

        sb_header = _read_u32_le(container, rec_start)
        layout_enum = sb_header & LAYOUT_ENUM_MASK
        assert layout_enum in LAYOUT_TABLE, f"unsupported layout enum {layout_enum}"
        N, n = LAYOUT_TABLE[layout_enum]

        block_headers_start = rec_start + 4
        payloads_start = align16(block_headers_start + N * 4)
        cursor = payloads_start
        for leaf in range(N):
            h = _read_u32_le(container, block_headers_start + leaf * 4)
            S, bias, mode = unpack_leaf_header_ref(h)
            n2 = n * n
            words_per_leaf = (n2 // NIBBLES_PER_WORD) if mode == MODE_FP4_AFFINE else (n2 // SYMBOLS_PER_WORD)
            payload_bytes = words_per_leaf * 4
            leaf_bytes = container[cursor:cursor + payload_bytes]
            out_parts.append(_decode_leaf_payload_ref(leaf_bytes, n, S, bias, mode))
            cursor += align16(payload_bytes)

    return np.concatenate(out_parts) if out_parts else np.zeros((0,), dtype=np.float32)


# ---------------------------------------------------------------------------
# --selftest
# ---------------------------------------------------------------------------

def selftest():
    rng = np.random.default_rng(20260824)
    ok = True

    def run_case(name, macroblocks, leaves_by_record=None):
        nonlocal ok
        container, expected = encode_container(macroblocks)
        decoded = decode_container_ref(container)

        if decoded.shape != expected.shape:
            print(f"[FAIL] {name}: shape mismatch decoded={decoded.shape} expected={expected.shape}")
            ok = False
            return
        # (1) Wire-format fidelity: decoding the actual bytes must reproduce
        # exactly what the encoder computed it would store (S/bias already
        # FP16-truncated), independent of the in-process arrays.
        diff = float(np.max(np.abs(decoded - expected))) if decoded.size else 0.0
        if diff > 1e-3:
            print(f"[FAIL] {name}: decode-vs-encoded-expected mismatch, max diff={diff}")
            ok = False
            return

        # (2) Encoding-quality sanity: decoded values stay within a generous
        # multiple of each leaf's own (S, bias) affine grid relative to the
        # original random weights (i.e. genuinely round-trips the underlying
        # data, not just self-consistent bit-shuffling).
        if leaves_by_record is not None:
            offset = 0
            for record_idx, leaves in enumerate(leaves_by_record):
                for leaf_w in leaves:
                    n2 = leaf_w.shape[0]
                    leaf_decoded = decoded[offset:offset + n2]
                    bound = 12.0 * (float(np.std(leaf_w)) + 1.0)
                    leaf_err = float(np.max(np.abs(leaf_decoded - leaf_w)))
                    if leaf_err > bound:
                        print(f"[FAIL] {name}: record {record_idx} leaf reconstruction error "
                              f"{leaf_err} exceeds bound {bound}")
                        ok = False
                        return
                    offset += n2

        print(f"[PASS] {name}: max wire-format diff={diff:.6g}")

    # Both modes x all five uniform layouts, single macroblock (B=1).
    for layout_enum, (N, n) in LAYOUT_TABLE.items():
        for mode in (MODE_FP4_AFFINE, MODE_T158_AFFINE):
            leaves = [rng.standard_normal(n * n).astype(np.float32) for _ in range(N)]
            run_case(f"layout={layout_enum} mode={mode}", [(leaves, layout_enum, mode)], [leaves])

    # B != 0 (mod 4) alignment case: 3 macroblocks (pad1 exercise, spec Pitfall 3).
    layout_enum = LAYOUT_UNIFORM_16
    mode = MODE_FP4_AFFINE
    N, n = LAYOUT_TABLE[layout_enum]
    macroblocks = []
    leaves_by_record = []
    for _ in range(3):
        leaves = [rng.standard_normal(n * n).astype(np.float32) for _ in range(N)]
        macroblocks.append((leaves, layout_enum, mode))
        leaves_by_record.append(leaves)
    run_case("B=3 alignment (64x192)", macroblocks, leaves_by_record)

    # Automatic (non-forced) per-block mode selection, exercised end-to-end:
    # a near-constant leaf (ternary-friendly) and a smoothly-varying leaf
    # (FP4-friendly) in the same macroblock, letting Eq. 5 pick per leaf.
    layout_enum = LAYOUT_UNIFORM_32
    N, n = LAYOUT_TABLE[layout_enum]
    mixed_leaves = []
    for i in range(N):
        if i % 2 == 0:
            leaf = rng.choice([-1.0, 0.0, 1.0], size=n * n).astype(np.float32) * 0.01
        else:
            leaf = rng.standard_normal(n * n).astype(np.float32)
        mixed_leaves.append(leaf)
    run_case("automatic mode selection (Eq. 5)", [(mixed_leaves, layout_enum, None)], [mixed_leaves])

    return ok


# ---------------------------------------------------------------------------
# --emit-cpp-fixture
# ---------------------------------------------------------------------------

def build_fixture_cases():
    rng = np.random.default_rng(20260824)
    cases = []
    layout_names = {
        LAYOUT_UNIFORM_64: "uniform64",
        LAYOUT_UNIFORM_32: "uniform32",
        LAYOUT_UNIFORM_16: "uniform16",
        LAYOUT_UNIFORM_8: "uniform8",
        LAYOUT_FULL_4X4: "full4x4",
    }
    for layout_enum, lname in layout_names.items():
        N, n = LAYOUT_TABLE[layout_enum]
        for mode, mname in ((MODE_FP4_AFFINE, "mode0"), (MODE_T158_AFFINE, "mode1")):
            leaves = [rng.standard_normal(n * n).astype(np.float32) for _ in range(N)]
            container, expected = encode_container([(leaves, layout_enum, mode)])
            cases.append({
                "name": f"{mname}_{lname}",
                "container": container,
                "dims": (MACROBLOCK_EDGE, MACROBLOCK_EDGE),
                "mode": mode,
                "layout": layout_enum,
                "expected": expected,
            })

    # B != 0 (mod 4) alignment case.
    layout_enum = LAYOUT_UNIFORM_16
    mode = MODE_FP4_AFFINE
    N, n = LAYOUT_TABLE[layout_enum]
    macroblocks = []
    for _ in range(3):
        leaves = [rng.standard_normal(n * n).astype(np.float32) for _ in range(N)]
        macroblocks.append((leaves, layout_enum, mode))
    container, expected = encode_container(macroblocks)
    cases.append({
        "name": "mode0_uniform16_b3",
        "container": container,
        "dims": (MACROBLOCK_EDGE, MACROBLOCK_EDGE * 3),
        "mode": mode,
        "layout": layout_enum,
        "expected": expected,
    })
    return cases


def emit_cpp_fixture(path):
    cases = build_fixture_cases()
    lines = []
    lines.append("// Auto-generated by tools/fp4/encode_sgfp4.py --emit-cpp-fixture.")
    lines.append("// DO NOT EDIT BY HAND -- regenerate via:")
    lines.append("//   python tools/fp4/encode_sgfp4.py --emit-cpp-fixture test/op/SGFP4DequantFixtures.h")
    lines.append("#ifndef SGFP4DequantFixtures_h")
    lines.append("#define SGFP4DequantFixtures_h")
    lines.append("")
    lines.append("#include <cstddef>")
    lines.append("")
    lines.append("// Cross-language round-trip fixtures for SGFP4 v2 uniform-layout containers.")
    lines.append("// The Python-side oracle is tools/fp4/encode_sgfp4.py --selftest.")
    lines.append("namespace sgfp4_fixtures {")
    lines.append("")
    lines.append("struct Fixture {")
    lines.append("    const char* name;")
    lines.append("    const unsigned char* container;")
    lines.append("    size_t containerSize;")
    lines.append("    int dimO;")
    lines.append("    int dimI;")
    lines.append("    int mode;")
    lines.append("    int layout;")
    lines.append("    const float* expected;")
    lines.append("    size_t expectedCount;")
    lines.append("};")
    lines.append("")

    for case in cases:
        cbytes = case["container"]
        arr_name = f"kFixture_{case['name']}_data"
        hex_bytes = ", ".join(f"0x{b:02x}" for b in cbytes)
        lines.append(f"static const unsigned char {arr_name}[] = {{")
        lines.append(hex_bytes + ",")
        lines.append("};")

        exp_name = f"kFixture_{case['name']}_expected"
        exp_vals = ", ".join(f"{float(v):.9e}f" for v in case["expected"])
        lines.append(f"static const float {exp_name}[] = {{")
        lines.append(exp_vals + ",")
        lines.append("};")
        lines.append("")

    lines.append("static const Fixture kFixtures[] = {")
    for case in cases:
        arr_name = f"kFixture_{case['name']}_data"
        exp_name = f"kFixture_{case['name']}_expected"
        lines.append(
            f'    {{"{case["name"]}", {arr_name}, sizeof({arr_name}), '
            f'{case["dims"][0]}, {case["dims"][1]}, {case["mode"]}, {case["layout"]}, '
            f'{exp_name}, sizeof({exp_name}) / sizeof(float)}},'
        )
    lines.append("};")
    lines.append("static const size_t kFixtureCount = sizeof(kFixtures) / sizeof(kFixtures[0]);")
    lines.append("")
    lines.append("} // namespace sgfp4_fixtures")
    lines.append("")
    lines.append("#endif // SGFP4DequantFixtures_h")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="SGFP4 v2 uniform-layout reference encoder (affine dual-mode, per-block mode selection)"
    )
    parser.add_argument("--selftest", action="store_true",
                         help="Encode->decode round-trip self-test (both modes x all uniform layouts + B!=0(mod4))")
    parser.add_argument("--emit-cpp-fixture", metavar="PATH",
                         help="Write the committed C++ cross-language fixture header to PATH")
    args = parser.parse_args()

    if not args.selftest and not args.emit_cpp_fixture:
        parser.print_help()
        sys.exit(1)

    ok = True
    if args.selftest:
        ok = selftest()
        if not ok:
            print("SELFTEST FAILED")
        else:
            print("SELFTEST PASSED")

    if args.emit_cpp_fixture:
        emit_cpp_fixture(args.emit_cpp_fixture)
        print(f"Wrote fixture header to {args.emit_cpp_fixture}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
