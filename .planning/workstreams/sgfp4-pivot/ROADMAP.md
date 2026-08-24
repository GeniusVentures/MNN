# Roadmap: MNN — SGFP4 v2 Quadtree-Adaptive Quantization

## Overview

This workstream adds **SGFP4 v2** (Super Genius FP4, quadtree-adaptive profile) as a new
weight-compression decode path in MNN. SGFP4 v2 defines an affine-integer, dual-mode container
(`w = S·c + bias`) with a self-framed stream (magic `'SGF4'`, version `0x02`), a record-offset
table, variable-size per-macroblock records, and an error-driven quadtree layout.

**This work is additive, not a replacement.** The existing E2M1 "Ultra FP4" microformat
(`include/MNN/FP4DequantUtils.hpp`, `source/backend/cpu/CPUFP4Dequant.cpp`,
`source/backend/vulkan/buffer/execution/VulkanFP4Dequant.cpp`, `tools/fp4/quantize_fp4.py` — built
in the `milestone` workstream's Phases 2 and 4) stays in place and its behavior is unchanged. In
particular `MNN::dequant_fp4_packed_cpu()` is a live cross-repo API contract consumed by
SuperGenius's `SGProcessingManager` and must keep its current signature and E2M1 semantics. SGFP4
v2 introduces a *new* dedicated Execution class and a *new* container format alongside that path;
the two do not share code beyond incidental helpers.

The journey: **Phase 1** proves the affine dual-mode reconstruction math, FP16 parameter packing,
v2 stream framing, and the container plumbing on CPU for uniform layouts only — establishing decode
correctness for both `FP4_AFFINE` and `T158_AFFINE` without quadtree complexity. **Phase 2** extends
CPU decode to the adaptive `LAYOUT_MIXED` quadtree and adds the error-driven encoder. **Phase 3**
ports the (now CPU-validated) uniform decode to a Vulkan GLSL Execution. **Phase 4** extends the
Vulkan shader to the quadtree layout, completing GPU parity with the CPU reference.

This ordering follows MNN's own `skills/add-new-op/SKILL.md` process (schema → shape → CPU backend →
unit tests → extend to other backends): the full decode is made correct and tested on CPU before
being ported to Vulkan, and the hardest GPU piece (quadtree-in-a-shader) is isolated last.

### Roadmap Notes (locked scope, not phase blockers)

1. **v2 only.** No SGFP4 v1 fixed-payload work, staged or otherwise — v1 is dropped from scope
   entirely.

2. **Container adoption is locked to external-file + minimal descriptor.** The SGFP4 v2 container
   (magic, record-offset table, macroblock records) is stored in a `.mnn.weight`-style external
   sidecar file; the op carries only a small `{magic, offset, size}` descriptor, mirroring
   `Convolution2D.external` (`schema/default/CaffeOp.fbs:109`, `source/core/ConvolutionCommon.cpp`,
   `source/core/Interpreter.cpp:96`). The FlatBuffers schema never models macroblocks or quadtrees
   as typed fields — a dedicated Execution class parses SGFP4's own byte-level offset-table/quadtree
   structure internally.

3. **Attestation / verifiable-execution is OUT of scope.** MNN's job is to run AI processing and
   return a result; SuperGenius verifies that result separately. No conformance-vector ("golden
   container") tests, no byte-exactness test infrastructure, and no integer-exact-only kernel
   variants belong in this workstream. (Parity tests here compare CPU vs Vulkan *within float
   tolerance* for functional correctness, not for cross-device bit-exactness attestation.)

4. **MNN-only scope.** Any SuperGenius/`SGProcessingManager`-side integration — consuming the new
   decode entry point this produces, fixing the stale `Fp4UltraRecognizedButDecodeUnavailable`
   test, backfilling the D-04/D-09/D-13 design docs — is deferred to a separate GSD plan in that
   repo. No cross-repo integration phases are scoped here.

5. **Vulkan target is the buffer backend.** SGFP4 v2 Vulkan work lives under
   `source/backend/vulkan/buffer/execution/` alongside the existing `VulkanFP4Dequant`; editing
   buffer-backend GLSL requires regenerating embedded shaders with
   `source/backend/vulkan/buffer/compiler/makeshader.py` and committing the regenerated
   `AllShader.cpp` / `AllShader.h` / `VulkanShaderMap.cpp` (per CLAUDE.md).

6. **Open (independent, not a blocker):** whether to first execute the `milestone` workstream's
   Phase 4 plan 04-02 (E2E test of the existing E2M1 pipeline). Assumed to proceed in parallel /
   independently, since SGFP4 v2 is additive and touches different files.

Reference source: `.planning/quick/260821-p1q-evaluate-current-fp4-ultra-fp4-implement/SGFP4-PIVOT-ANALYSIS.md`
and the spec `.planning/sgfp4-arxiv-v2.txt` (Sections 3, 4, 6).

## Phases

- [x] **Phase 1: Affine Dual-Mode Decode Core (CPU, Uniform Layouts)** — Prove `w = S·c + bias` for FP4_AFFINE + T158_AFFINE, v2 stream framing, uniform-layout record walk, and external-sidecar container plumbing on CPU (completed 2026-08-24)
- [ ] **Phase 2: Adaptive Quadtree Layout (CPU, LAYOUT_MIXED)** — Extend CPU decode to the pre-order-DFS quadtree split-map and add the error-driven encoder (mode selection ε=0.10, outlier veto)
- [ ] **Phase 3: Vulkan Decode — Uniform Layouts** — Port uniform-layout SGFP4 v2 decode to a Vulkan GLSL Execution with CPU/Vulkan parity
- [ ] **Phase 4: Vulkan Decode — Adaptive Quadtree (LAYOUT_MIXED)** — Extend the Vulkan shader to walk the quadtree split-map, completing GPU parity with the CPU reference

## Phase Details

### Phase 1: Affine Dual-Mode Decode Core (CPU, Uniform Layouts)

**Goal**: A new dedicated CPU Execution class decodes SGFP4 v2 uniform-layout containers to float weights via the affine dual-mode rule `w = S·c + bias`, loading the container from an external `.mnn.weight`-style sidecar through a minimal `{magic, offset, size}` op descriptor. Establishes decode correctness for both code modes and the container plumbing, with no quadtree complexity.
**Depends on**: Nothing (first phase)
**Requirements**: SGV2-01, SGV2-02, SGV2-03, SGV2-04, SGV2-05, SGV2-06, SGV2-07
**Success Criteria** (what must be TRUE):

  1. An encoder-produced SGFP4 v2 uniform-layout container (magic `'SGF4'`, version `0x02`) round-trips through the new CPU Execution and reconstructs weights via `w = S·c + bias` for both FP4_AFFINE (codes [-8,7]) and T158_AFFINE (codes {-1,0,+1}) within the encoder's round-trip error bound
  2. The op loads its container from an external `.mnn.weight`-style sidecar using only a `{magic, offset, size}` descriptor — no macroblock/quadtree fields appear anywhere in the FlatBuffers schema
  3. All five uniform layouts (LAYOUT_UNIFORM_64/32/16/8, LAYOUT_FULL_4x4) decode with correct leaf count, row-major raster leaf order, and normative payload word counts (n²/8 words mode 0, n²/16 words mode 1), verified via `./run_test.out`
  4. FP16 scale+bias unpack (packHalf2x16 order; v2 leaf's 12-bit truncated-bias recovery `S=half(h>>16)`, `bias=half(h & 0xFFF0)`, `flags=h & 0xF`) matches a reference half→float within FP16 precision, and the ternary reserved symbol `11` decodes to 0
  5. The existing E2M1 `CPUFP4Dequant` / `dequant_fp4_packed_cpu` path and its tests are unchanged (additive, not a replacement)

**Plans**: 2/2 plans complete
Plans:

- [x] 01-01-PLAN.md — Schema (OpType_SGFP4Dequant + SGFP4DequantParam) + shape computer + SGFP4 v2 decode core (framing, uniform record walk, FP16 leaf-header unpack, dual-mode payload decode, affine reconstruct) + CPU Execution with external-sidecar loading (SGV2-01..06)
- [x] 01-02-PLAN.md — Reference uniform-layout v2 Python encoder + CPU round-trip / edge-case / op-level tests (SGV2-07)

### Phase 2: Adaptive Quadtree Layout (CPU, LAYOUT_MIXED)

**Goal**: CPU decode handles the variable-size LAYOUT_MIXED record via the pre-order-DFS quadtree split-map, and a new error-driven encoder produces adaptive quadtree layouts using per-level thresholds, per-region mode selection, and the ternary outlier veto. Completes the full SGFP4 v2 feature set on CPU.
**Depends on**: Phase 1
**Requirements**: SGV2-08, SGV2-09, SGV2-10, SGV2-11
**Success Criteria** (what must be TRUE):

  1. A LAYOUT_MIXED container with a known quadtree split-map decodes on CPU with leaves visited in pre-order DFS order (quadrants TL/TR/BL/BR) matching a golden traversal; nodes of size 4 contribute no split bit
  2. The error-driven encoder collapses to a uniform layout when all leaves share one size and otherwise emits LAYOUT_MIXED, selecting T158 iff `e_T158 ≤ (1+ε)·e_FP4` (default ε=0.10) with the ternary outlier veto respected
  3. Mixed/adaptive containers round-trip encode→decode within the encoder's per-level error thresholds, verified via `./run_test.out`
  4. Variable per-leaf payload sizes and 16-byte record/payload alignment are honored — a leaf of edge size n consumes exactly n²/8 (mode 0) or n²/16 (mode 1) little-endian uint32 words

**Plans**: 2 plans
Plans:
**Wave 1**

- [ ] 02-01-PLAN.md — LAYOUT_MIXED decode core: split-map constants + iterative fixed-size-stack walker + MIXED branch in `dequant_sgfp4_container_cpu()` (SGV2-08, SGV2-09)

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 02-02-PLAN.md — Error-driven quadtree encoder (`subdivide_macroblock`/`build_split_map`/`classify_layout`) + committed mixed fixtures + golden-traversal/mixed round-trip/negative split-map tests (SGV2-10, SGV2-11)

### Phase 3: Vulkan Decode — Uniform Layouts

**Goal**: A Vulkan buffer-backend GLSL Execution decodes uniform-layout SGFP4 v2 containers on GPU (FP4_AFFINE + T158_AFFINE, shift-mask-FMA), reading the same external-sidecar descriptor as the CPU path, with output matching the CPU reference decode within float tolerance.
**Depends on**: Phase 1
**Requirements**: SGV2-12, SGV2-13, SGV2-14
**Success Criteria** (what must be TRUE):

  1. A GLSL compute shader decodes uniform-layout SGFP4 v2 containers on the Vulkan buffer backend and is embedded via `makeshader.py` with regenerated `AllShader.cpp` / `AllShader.h` / `VulkanShaderMap.cpp` committed
  2. The new Vulkan Execution class is registered in the buffer-backend execution table and loads the same `{magic, offset, size}` external-sidecar descriptor as the CPU path, producing decoded weights for both code modes
  3. Vulkan decode output matches the CPU reference decode for uniform-layout containers within float tolerance, verified via `./run_test.out`

**Plans**: ~2 (not yet broken down)
Plans:

- [ ] TBD (run /gsd-plan-phase 3 to break down)

### Phase 4: Vulkan Decode — Adaptive Quadtree (LAYOUT_MIXED)

**Goal**: The Vulkan shader walks the LAYOUT_MIXED split-map and decodes variable per-leaf-size records on GPU (one workgroup per macroblock), achieving CPU/Vulkan parity across the complete SGFP4 v2 feature set.
**Depends on**: Phase 2, Phase 3
**Requirements**: SGV2-15, SGV2-16
**Success Criteria** (what must be TRUE):

  1. The Vulkan shader walks the LAYOUT_MIXED split-map (pre-order DFS, TL/TR/BL/BR) and decodes variable per-leaf-size records on GPU, with regenerated embedded shaders committed
  2. Vulkan decode output matches the CPU reference decode for mixed/adaptive containers within float tolerance, verified via `./run_test.out`
  3. The complete SGFP4 v2 feature set (both code modes, all uniform layouts, and LAYOUT_MIXED) decodes consistently on CPU and Vulkan within float tolerance

**Plans**: ~1-2 (not yet broken down)
Plans:

- [ ] TBD (run /gsd-plan-phase 4 to break down)

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Affine Dual-Mode Decode Core (CPU, Uniform) | 2/2 | Complete    | 2026-08-24 |
| 2. Adaptive Quadtree Layout (CPU, LAYOUT_MIXED) | 0/~2 | Not started | — |
| 3. Vulkan Decode — Uniform Layouts | 0/~2 | Not started | — |
| 4. Vulkan Decode — Adaptive Quadtree | 0/~1-2 | Not started | — |
