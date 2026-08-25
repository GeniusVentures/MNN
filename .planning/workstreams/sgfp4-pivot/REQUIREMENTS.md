# Requirements: MNN SGFP4 v2 Quadtree-Adaptive Quantization

**Defined:** 2026-08-22
**Core Value:** A working SGFP4 v2 (quadtree-adaptive, affine dual-mode) weight-decode path in MNN — CPU and Vulkan — additive to the existing E2M1 "Ultra FP4" implementation.

> **Additive, not a replacement.** These requirements add a new container format and a new dedicated
> Execution class. The existing E2M1 path (`FP4DequantUtils.hpp`, `CPUFP4Dequant`, `VulkanFP4Dequant`,
> `tools/fp4/quantize_fp4.py`) is untouched; `MNN::dequant_fp4_packed_cpu()` keeps its current
> signature and E2M1 semantics (live cross-repo contract with `SGProcessingManager`).

## v1 Requirements

### Affine Dual-Mode Decode Core — CPU, Uniform Layouts

- [x] **SGV2-01**: Affine reconstruction `w = S·c + bias` is implemented for both code modes — FP4_AFFINE (mode 0, 4-bit two's-complement, codes [-8,7]) and T158_AFFINE (mode 1, ternary, codes {-1,0,+1}, with reserved symbol `11` decoded as 0)
- [x] **SGV2-02**: FP16 (IEEE 754 binary16) scale+bias unpack from a packed uint32 in packHalf2x16 order (S upper 16 bits, bias lower 16 bits), including the v2 leaf-header 12-bit truncated-bias recovery `S=half(h>>16)`, `bias=half(h & 0xFFF0)`, `flags=h & 0xF`
- [x] **SGV2-03**: v2 self-framed stream parsing — magic `'SGF4'`, version `0x02`, `B` record count, 16-byte-aligned little-endian record-offset table, and per-record `sb_header` layout enum (bits 0–2)
- [x] **SGV2-04**: Uniform-layout record walk (LAYOUT_UNIFORM_64/32/16/8, LAYOUT_FULL_4x4) with deterministic leaf count/geometry, row-major raster leaf order, and normative per-leaf payload packing (n²/8 words mode 0, n²/16 words mode 1)
- [x] **SGV2-05**: New FlatBuffers op descriptor carrying only `{magic, offset, size}`, with the SGFP4 v2 container stored in a `.mnn.weight`-style external sidecar file (mirroring `Convolution2D.external`) and loaded via `FileLoader` — no macroblock/quadtree typed fields in the schema
- [x] **SGV2-06**: New dedicated CPU Execution class parses the container internally and produces a decoded float tensor, additive to (not replacing) the existing E2M1 `CPUFP4Dequant`
- [x] **SGV2-07**: Minimal Python encoder produces uniform-layout v2 containers (reference round-to-nearest affine encode + per-block mode selection), and CPU unit tests validate round-trip decode for both modes across all uniform layouts via `./run_test.out`

### Adaptive Quadtree Layout — CPU, LAYOUT_MIXED

- [ ] **SGV2-08**: LAYOUT_MIXED split-map parsing — 12-byte / 3-word little-endian bitmap, pre-order DFS traversal, quadrant order TL/TR/BL/BR, nodes of size ≥8 contribute a split bit, 4×4 nodes are always leaves
- [ ] **SGV2-09**: Variable per-leaf decode for mixed records — leaf headers and payloads consumed in traversal order, each leaf's edge size n driving its payload word count, honoring 16-byte record and payload alignment
- [ ] **SGV2-10**: Error-driven quadtree encoder — recursive subdivision with per-level MSE/relative-error thresholds, per-region mode selection (`choose T158 iff e_T158 ≤ (1+ε)·e_FP4`, default ε=0.10), ternary outlier veto, and uniform-layout collapse when all leaves share one size
- [ ] **SGV2-11**: CPU round-trip tests for mixed/adaptive containers, including a golden split-map traversal-order check, passing via `./run_test.out`

### Vulkan Decode — Uniform Layouts

- [x] **SGV2-12**: GLSL compute shader decodes uniform-layout SGFP4 v2 containers on the Vulkan buffer backend (FP4_AFFINE + T158_AFFINE affine reconstruction via shift-mask-FMA), embedded via `makeshader.py` with regenerated `AllShader.cpp` / `AllShader.h` / `VulkanShaderMap.cpp`
- [x] **SGV2-13**: New Vulkan Execution class registered in the buffer-backend execution table, reading the same `{magic, offset, size}` external-sidecar descriptor as the CPU path
- [x] **SGV2-14**: CPU/Vulkan decode-parity test for uniform-layout containers within float tolerance, passing via `./run_test.out`

### Vulkan Decode — Adaptive Quadtree (LAYOUT_MIXED)

- [x] **SGV2-15**: Vulkan shader extended to walk the LAYOUT_MIXED split-map and decode variable per-leaf-size records on GPU (one workgroup per macroblock)
- [x] **SGV2-16**: CPU/Vulkan decode-parity test for mixed/adaptive containers within float tolerance, passing via `./run_test.out`

## v2 Requirements

- **SGV2-17**: End-to-end model integration — convert a real model's weights to SGFP4 v2 and run inference through the new op in a full graph (beyond isolated decode tests)
- **SGV2-18**: GPU decode performance tuning — workgroup sizing, coalesced loads, shared-memory fused dequantize→matmul
- **SGV2-19**: Additional macroblock geometries (e.g. 128×32, 32×64, 128×64) and alternate payload sizes/alignments (Section 10 design variants)
- **SGV2-20**: Extend SGFP4 v2 decode to other backends (Metal / CUDA / OpenCL)
- **SGV2-21**: Optional Laplacian-pyramid error weighting in the encoder's quadtree cost function

## Out of Scope

| Feature | Reason |
|---------|--------|
| SGFP4 v1 fixed-payload profile | Locked to v2-only; v1 dropped from scope entirely |
| Attestation / verifiable-execution support | MNN runs AI processing and returns a result; SuperGenius verifies separately |
| Conformance-vector ("golden container") / byte-exactness test infrastructure | Cross-device bit-exact attestation is SuperGenius's concern, not MNN's |
| Integer-exact-only ternary kernel variants | Only needed for the attestation determinism classes — out of scope |
| SuperGenius / `SGProcessingManager`-side integration | Deferred to a separate GSD plan in that repo (consume new entry point, fix stale test, backfill D-04/D-09/D-13 docs) |
| Changes to the existing E2M1 `dequant_fp4_packed_cpu()` contract | Live cross-repo API contract; SGFP4 v2 is additive, not a modification |
| Modeling macroblocks/quadtrees as typed FlatBuffers fields | Locked to external-file + minimal `{magic, offset, size}` descriptor |
| Encoder accuracy / perplexity benchmarking | Reference encoder is exemplary; empirical evaluation is out of scope here |

## Traceability

| Requirement | Category | Phase | Status |
|-------------|----------|-------|--------|
| SGV2-01 | Affine dual-mode math | Phase 1 | Complete |
| SGV2-02 | FP16 param packing | Phase 1 | Complete |
| SGV2-03 | v2 stream framing | Phase 1 | Complete |
| SGV2-04 | Uniform record walk | Phase 1 | Complete |
| SGV2-05 | Schema descriptor + external sidecar | Phase 1 | Complete |
| SGV2-06 | CPU Execution class | Phase 1 | Complete |
| SGV2-07 | Minimal encoder + round-trip tests | Phase 1 | Pending |
| SGV2-08 | Quadtree split-map parse | Phase 2 | Pending |
| SGV2-09 | Variable per-leaf decode | Phase 2 | Pending |
| SGV2-10 | Error-driven encoder | Phase 2 | Pending |
| SGV2-11 | Mixed round-trip tests | Phase 2 | Pending |
| SGV2-12 | GLSL uniform decode shader | Phase 3 | Pending |
| SGV2-13 | Vulkan Execution + registration | Phase 3 | Pending |
| SGV2-14 | CPU/Vulkan uniform parity | Phase 3 | Pending |
| SGV2-15 | Vulkan quadtree decode | Phase 4 | Pending |
| SGV2-16 | CPU/Vulkan mixed parity | Phase 4 | Pending |

**Coverage:**

- v1 requirements: 16 total
- Mapped to phases: 16 (100%)
- Phase 1: 7 requirements (SGV2-01 through SGV2-07)
- Phase 2: 4 requirements (SGV2-08 through SGV2-11)
- Phase 3: 3 requirements (SGV2-12 through SGV2-14)
- Phase 4: 2 requirements (SGV2-15 through SGV2-16)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-08-22*
*Last updated: 2026-08-22 after roadmap creation (traceability populated)*
