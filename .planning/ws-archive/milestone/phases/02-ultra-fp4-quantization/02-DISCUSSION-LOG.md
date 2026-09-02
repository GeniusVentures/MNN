# Phase 2: Ultra FP4 Quantization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 02-ultra-fp4-quantization
**Areas discussed:** FP4 format encoding, Op integration pattern, Weight packing, Dequantization target precision

---

## FP4 Format Encoding

| Option | Description | Selected |
|--------|-------------|----------|
| E2M1 | 1 sign, 2 exponent, 1 mantissa — standard 4-bit float | ✓ |
| E3M0 | 3 exponent, 0 mantissa — power-of-two only | |
| Custom | Application-specific encoding | |

**User's choice:** E2M1 (standard). Dequantization formula: `(-1)^s × 2^(e-1) × (1 + m/2)`.

---

## Op Integration Pattern

| Option | Description | Selected |
|--------|-------------|----------|
| New OpType | Register a new `OpType_FP4Dequant` through schema/shape/geometry/Vulkan | |
| Existing dequant variant | Integrate into existing dequantization flow, reuse loaded shader infrastructure | ✓ |

**User's choice:** Implemented as an existing dequant variant — not a standalone new op type. Leverages the loaded shader pattern rather than creating a new registration chain.

---

## Weight Packing

| Option | Description | Selected |
|--------|-------------|----------|
| 1 value per byte (with padding) | Simpler but wastes 4 bits per value | |
| 2 values per byte | Packed sequentially, GLSL-optimized | ✓ |
| 2 values per byte + swizzling | Packed with swizzling for GPU memory coalescing | ✓ |

**User's choice:** 2 values per byte, packed optimally for GLSL storage buffer reads, with swizzling as needed for GPU memory access patterns.

---

## Dequantization Target Precision

| Option | Description | Selected |
|--------|-------------|----------|
| FP16 only | Half precision, best GPU memory bandwidth | |
| FP32 only | Full precision, best accuracy | |
| FP16 default, FP32 flag | FP16 by default, FP32 via runtime flag | ✓ |

**User's choice:** Default FP16 for GPU efficiency. FP32 output supported via a flag (GLSL preprocessor define or runtime uniform).

---

## the agent's Discretion

None — all gray areas were decided by the user.

## Deferred Ideas

None — discussion stayed within phase scope.
