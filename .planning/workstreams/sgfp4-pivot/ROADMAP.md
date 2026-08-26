# Roadmap: MNN — SGFP4 v2 Quadtree-Adaptive Quantization

## Milestones

- ✅ **v1.0 SGFP4 v2 Decode (Vulkan-parity)** — Phases 1-4 (shipped 2026-08-26)
- 🚧 **v2.0 SGFP4 v2 Converter Integration** — Phases 5-9 (planning)

## Phases

<details>
<summary>✅ v1.0 SGFP4 v2 Decode (Vulkan-parity) (Phases 1-4) — SHIPPED 2026-08-26</summary>

Full detail archived to `.planning/milestones/v1.0-ROADMAP.md`; phase docs (PLAN/SUMMARY/VERIFICATION/etc.) archived to `.planning/milestones/v1.0-phases/`.

- [x] Phase 1: Affine Dual-Mode Decode Core (CPU, Uniform Layouts) (2/2 plans) — completed 2026-08-24
- [x] Phase 2: Adaptive Quadtree Layout (CPU, LAYOUT_MIXED) (2/2 plans) — completed 2026-08-24
- [x] Phase 3: Vulkan Decode — Uniform Layouts (4/4 plans) — completed 2026-08-25
- [x] Phase 4: Vulkan Decode — Adaptive Quadtree (LAYOUT_MIXED) (2/2 plans) — completed 2026-08-25

</details>

### 🚧 v2.0 SGFP4 v2 Converter Integration (Planning)

**Milestone Goal:** A pytorch → onnx → mnnconvert pipeline can produce a `.mnn` file that runs SGFP4 v2 (quadtree-adaptive) FP4 inference, via a native mnnconvert CLI flag.

Builds on v1.0's decode-only path (CPU + Vulkan Executions, synthetic-fixture-validated). This milestone closes the gap to a usable end-user pipeline: quantizing real model weights and wiring the result natively into `mnnconvert`. Ordering follows the architectural correction from research — "op-type rewrite" is graph surgery (a new `SGFP4Dequant` node feeding the original conv op's `inputs[1]`), not in-place mutation, and real-weight validation is scheduled *before* full graph-rewrite integration because unvalidated synthetic-fixture assumptions (macroblock tiling, split thresholds) are the top-flagged risk — cheaper to catch against an isolated encoder (Phase 7) than after it's wired into the converter graph (Phase 8).

- [ ] **Phase 5: Schema + Sidecar Wiring** - Add `buffer:[byte]` to `SGFP4DequantParam` and wire `RemoveParams.cpp` to externalize it through the existing shared sidecar mechanism
- [ ] **Phase 6: Real-Weight C++ Encoder Port** - Port the Python quadtree/dual-mode encoder to C++, operating on real (non-64-aligned) weight tensor shapes
- [ ] **Phase 7: Real-Weight Validation Against Actual Model Statistics** - Validate/revise encoder parameters against a real model's weight distributions before graph-rewrite integration
- [ ] **Phase 8: Graph-Rewrite PostConverter Pass + CLI Flag** - New `PostConverter` pass inserts `SGFP4Dequant` nodes; new CLI flag triggers it; `WeightQuantAndCoding.cpp` skip-guard prevents double-processing
- [ ] **Phase 9: End-to-End Validation** - A real model converts and runs correct inference on CPU and Vulkan via the new flag

## Phase Details

### Phase 5: Schema + Sidecar Wiring
**Goal**: The converter's SGFP4Dequant op descriptor gains a raw-bytes staging field, and `RemoveParams.cpp` externalizes it through the same shared sidecar mechanism already used by `Convolution2D`, unblocking every later encoder and graph-rewrite phase.
**Depends on**: Phase 4
**Requirements**: SGV2-22, SGV2-23
**Success Criteria** (what must be TRUE):
  1. `SGFP4DequantParam` in `schema/default/CaffeOp.fbs` has a `buffer:[byte]` field, and regenerated FlatBuffers headers compile cleanly across both the converter and the runtime.
  2. `RemoveParams.cpp` recognizes `OpParameter_SGFP4DequantParam` and externalizes its `buffer` bytes through the existing shared `ofstream`/threaded-offset sidecar mechanism (same code path as `Convolution2D.external`), producing a correct `{magic, offset, size}` descriptor.
  3. A test op carrying `SGFP4DequantParam.buffer` converts through `RemoveParams` alongside pre-existing `Convolution2D` external sidecar entries in the same model without offset collisions or corruption of either op's data.
**Plans**: TBD

### Phase 6: Real-Weight C++ Encoder Port
**Goal**: Real `Convolution2D`/`ConvolutionDepthwise`/`Deconvolution` weight tensors — of arbitrary shape, not just synthetic 64x64-aligned fixtures — can be SGFP4 v2 encoded in-process by a C++ port of the Python encoder.
**Depends on**: Phase 5
**Requirements**: SGV2-24, SGV2-25
**Success Criteria** (what must be TRUE):
  1. Given the same weight tensor fed to `encode_sgfp4.py --selftest`, the C++ encoder produces output that decodes (via the existing CPU decode Execution) to the same reconstructed weights within the oracle's established tolerance, across the full existing selftest fixture set.
  2. The C++ encoder accepts real conv/deconv weight tensor shapes that are not multiples of the 64x64 macroblock edge, applying a defined and documented tiling/padding policy rather than failing or silently truncating.
  3. A unit test exercising at least one non-64-multiple weight shape passes: encode via the C++ encoder, decode via the existing CPU Execution, and verify round-trip correctness.
**Plans**: TBD

### Phase 7: Real-Weight Validation Against Actual Model Statistics
**Goal**: SGFP4 v2 encoder tiling/threshold parameters are validated — and revised if necessary — against real trained-model weight distributions before being wired into the automatic conversion pipeline.
**Depends on**: Phase 6
**Requirements**: SGV2-26, SGV2-27
**Success Criteria** (what must be TRUE):
  1. A roadmap-selected real test model's `Convolution2D`/`ConvolutionDepthwise`/`Deconvolution` weights are run through the Phase 6 C++ encoder, producing a quantization-error characterization (vs. FP32 reference) distinct from and in addition to the synthetic-fixture selftest.
  2. Where the real-weight error report shows synthetic-fixture-tuned defaults (macroblock size, subdivision thresholds, mode-selection epsilon) don't hold, those parameters are revised in the encoder and re-validated against the same real weights, with the error characterization improving or the deviation explicitly justified.
  3. The resulting encoder parameter set (revised or confirmed) is documented as the default the Phase 8 graph-rewrite pass will invoke.
**Plans**: TBD

### Phase 8: Graph-Rewrite PostConverter Pass + CLI Flag
**Goal**: mnnconvert exposes a native SGFP4 v2 CLI flag that, during conversion, inserts a decode-ready `SGFP4Dequant` weight producer ahead of eligible conv/deconv ops without corrupting or double-processing those ops' weight data.
**Depends on**: Phase 5, Phase 7
**Requirements**: SGV2-28, SGV2-29, SGV2-30
**Success Criteria** (what must be TRUE):
  1. A new `PostConverter` pass (modeled on `SplitBlockQuantConvolution.cpp`) inserts an `SGFP4Dequant` node — with correct buffer bytes and `externalPath` set — that feeds the original, type-unchanged `Convolution2D`/`ConvolutionDepthwise`/`Deconvolution` op as `inputs[1]`, and the existing CPU/Vulkan SGFP4 decode Executions load and execute it without returning `NOT_SUPPORT`.
  2. A new mnnconvert CLI flag — peer to `--weightQuantBits`, distinctly named from the sibling `milestone` workstream's Ultra FP4 flag — triggers SGFP4 v2 quantization end-to-end in a single `mnnconvert` invocation, self-gated via `Global<modelConfig>::Get()` (the rest of the default optimizer pipeline still runs, unlike the `--splitQuantBlock`/`expectedPass` short-circuit).
  3. `WeightQuantAndCoding.cpp` skips ops already rewritten to consume an `SGFP4Dequant` input — a model converted with the new flag shows no double-quantization artifacts on those weights (verified by inspecting the output graph/weight buffers).
**Plans**: TBD

### Phase 9: End-to-End Validation
**Goal**: A real trained model, converted through the new SGFP4 v2 flag, produces correct inference results on both CPU and Vulkan.
**Depends on**: Phase 8
**Requirements**: SGV2-31, SGV2-32
**Success Criteria** (what must be TRUE):
  1. A real test model converts via pytorch → onnx → mnnconvert(SGFP4 flag) → `.mnn` with no conversion errors.
  2. The converted `.mnn` model runs correct inference on CPU, matching an FP32/reference baseline within acceptable tolerance.
  3. The same converted `.mnn` model runs correct inference on Vulkan, matching the CPU/reference baseline within acceptable tolerance.
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 1. Affine Dual-Mode Decode Core (CPU, Uniform) | v1.0 | 2/2 | Complete | 2026-08-24 |
| 2. Adaptive Quadtree Layout (CPU, LAYOUT_MIXED) | v1.0 | 2/2 | Complete | 2026-08-24 |
| 3. Vulkan Decode — Uniform Layouts | v1.0 | 4/4 | Complete | 2026-08-25 |
| 4. Vulkan Decode — Adaptive Quadtree | v1.0 | 2/2 | Complete | 2026-08-25 |
| 5. Schema + Sidecar Wiring | v2.0 | 0/TBD | Not started | - |
| 6. Real-Weight C++ Encoder Port | v2.0 | 0/TBD | Not started | - |
| 7. Real-Weight Validation Against Actual Model Statistics | v2.0 | 0/TBD | Not started | - |
| 8. Graph-Rewrite PostConverter Pass + CLI Flag | v2.0 | 0/TBD | Not started | - |
| 9. End-to-End Validation | v2.0 | 0/TBD | Not started | - |
