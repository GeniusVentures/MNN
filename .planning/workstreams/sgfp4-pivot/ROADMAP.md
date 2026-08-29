# Roadmap: MNN — SGFP4 v2 Quadtree-Adaptive Quantization

## Milestones

- ✅ **v1.0 SGFP4 v2 Decode (Vulkan-parity)** — Phases 1-4 (shipped 2026-08-26)
- ✅ **v2.0 SGFP4 v2 Model-Artifact Injection Tool** — Phases 5-7 (shipped 2026-08-28)
- 📋 **v3.0 SGFP4 v2 Converter Integration** — Phases 8-12 (planned; re-scoped at plan time)

## Phases

<details>
<summary>✅ v1.0 SGFP4 v2 Decode (Vulkan-parity) (Phases 1-4) — SHIPPED 2026-08-26</summary>

Full detail archived to `.planning/milestones/v1.0-ROADMAP.md`; phase docs (PLAN/SUMMARY/VERIFICATION/etc.) archived to `.planning/milestones/v1.0-phases/`.

- [x] Phase 1: Affine Dual-Mode Decode Core (CPU, Uniform Layouts) (2/2 plans) — completed 2026-08-24
- [x] Phase 2: Adaptive Quadtree Layout (CPU, LAYOUT_MIXED) (2/2 plans) — completed 2026-08-24
- [x] Phase 3: Vulkan Decode — Uniform Layouts (4/4 plans) — completed 2026-08-25
- [x] Phase 4: Vulkan Decode — Adaptive Quadtree (LAYOUT_MIXED) (2/2 plans) — completed 2026-08-25

</details>

<details>
<summary>✅ v2.0 SGFP4 v2 Model-Artifact Injection Tool (Phases 5-7) — SHIPPED 2026-08-28</summary>

Full detail archived to `.planning/milestones/v2.0-ROADMAP.md`; phase docs archived to `.planning/milestones/v2.0-phases/`; milestone audit at `.planning/milestones/v2.0-MILESTONE-AUDIT.md` (passed 8/8).

- [x] Phase 5: Injection Core — Artifact Construction & Graph Splicing (2/2 plans) — completed 2026-08-26
- [x] Phase 6: Classic-API Load & Run Validation (2/2 plans) — completed 2026-08-27
- [x] Phase 7: Multi-Tensor Hardening & Structured-Data Coverage (3/3 plans) — completed 2026-08-28

</details>

### 📋 v2.0 Milestone Goal (shipped)

**A standalone tool takes a normally-converted `.mnn` plus one or more real SGFP4 v2 container files (gnus-poc `fp4_exporter.py --adaptive` output) and produces a final `.mnn` + external sidecar where target weight tensors are produced by `OpType_SGFP4Dequant` nodes — verified loadable/runnable via the classic Interpreter/Session API (the downstream `SGProcessingManager` path).** Shipped as `tools/fp4/sgfp4_inject` (`MNN_BUILD_SGFP4_TOOLS=ON`), with 9/9 `op/sgfp4/` test suites green and a 13-probe malformed-input clean-failure matrix. **v3.0 planning must absorb its sidecar/rewiring learnings** (see v2.0-MILESTONE-AUDIT.md tech debt: W-1 classic_api offset-convention retrofit, W-2 arg-stage failCleanup, W-3 portable gnus-poc root, test-helper dedup).

**Terminology lock:** the format is called **"SGFP4 v2"** — never "Ultra FP4" (that is the sibling `milestone` workstream's unrelated E2M1/`FP4_ULTRA` format, despite gnus-poc manifest labels suggesting otherwise).

### 📋 v3.0 SGFP4 v2 Converter Integration (Planned)

**Milestone Goal:** A pytorch → onnx → mnnconvert pipeline can produce a `.mnn` file that runs SGFP4 v2 (quadtree-adaptive) FP4 inference, via a native mnnconvert CLI flag.

Formerly the v2.0 milestone (roadmapped 2026-08-25 at 0% execution; moved 2026-08-26 when the Injection Tool milestone was inserted ahead of it). **v3.0 planning notes:** the SGFP4-pivot handoff (2026-08-26) designates gnus-poc's `fp4_exporter.py` as the canonical real-weight encoder — Phase 9 (C++ encoder port) must be re-evaluated at plan time against direct consumption of exporter output (as the injection tool does), and Phase 11 should absorb the injection tool's sidecar/rewiring learnings and retire the v2.0 tech debt (W-1 classic_api offset-convention retrofit, W-2 arg-stage failCleanup, W-3 portable gnus-poc root env-var override, test-helper dedup into `SGFP4TestUtil.hpp`).

## Phase Details

*(v1.0 phase details: see `.planning/milestones/v1.0-ROADMAP.md`. v2.0 phase details: see `.planning/milestones/v2.0-ROADMAP.md`.)*

- [x] **Phase 8: Schema + Sidecar Wiring** - Add `buffer:[byte]` to `SGFP4DequantParam` and wire `RemoveParams.cpp` to externalize it through the existing shared sidecar mechanism (completed 2026-08-28)
- [ ] **Phase 9: Real-Weight C++ Encoder Port** - Port the Python quadtree/dual-mode encoder to C++, operating on real (non-64-aligned) weight tensor shapes
- [ ] **Phase 10: Real-Weight Validation Against Actual Model Statistics** - Validate/revise encoder parameters against a real model's weight distributions before graph-rewrite integration
- [ ] **Phase 11: Graph-Rewrite PostConverter Pass + CLI Flag** - New `PostConverter` pass inserts `SGFP4Dequant` nodes; new CLI flag triggers it; `WeightQuantAndCoding.cpp` skip-guard prevents double-processing
- [ ] **Phase 12: End-to-End Validation** - A real model converts and runs correct inference on CPU and Vulkan via the new flag

### Phase 8: Schema + Sidecar Wiring

**Goal**: Add `buffer:[byte]` to `SGFP4DequantParam` (schema evolution) and wire `RemoveParams.cpp` so SGFP4 container bytes carried in the op param can be externalized through the converter's existing shared `.mnn.weight` sidecar mechanism (`saveExternalData` / `_largeModel` auto-trigger). The CPU and Vulkan runtime decoders accept **both** data placements — inline buffer and external sidecar — with one unified convention. This is the serialization foundation Phases 9–11 build on.
**Depends on**: Phase 4 (v1.0 — existing CPU/Vulkan decode Executions are the consumers) + v2.0 injection-tool learnings (16-byte sidecar alignment, offset conventions, classic-API load behavior)
**Requirements**: SGV2-22, SGV2-23 (`.planning/milestones/v2.0-REQUIREMENTS.md` §"v3.0 Converter Integration")
**Success Criteria** (what must be TRUE):

1. `schema/default/CaffeOp.fbs` `SGFP4DequantParam` contains `buffer:[byte]`; generated headers regenerated and committed; existing artifacts (injection-tool output) and all existing `op/sgfp4/` suites remain green unchanged (sidecar mode stays the original supported path — buffer mode is additive).
2. CPU (`CPUSGFP4Dequant::onResize`) and Vulkan Executions dispatch **buffer-first**: non-empty `param->buffer()` decodes directly from the inline bytes (no FileLoader, magic + dims-consistency entry checks retained); empty buffer falls back to the existing `external = {offset, size}` + `op->externalPath` path with all current validation (incl. T-01-04 file-size bounds) preserved.
3. `RemoveAndStoreParam` handles `OpParameter_SGFP4DequantParam` via an aligned `storeWeight` variant: sidecar region padded to a 16-byte multiple (zero-filled) before advancing the shared offset, `external = {offset, true-size}` (pad inert), buffer cleared after store (no dual-source duplication); externalization remains gated by the existing `config.saveExternalData` / `_largeModel` flags — no new converter flag.
4. Decode-parity tests prove buffer-mode decode == sidecar-mode decode == existing oracle (`SGFP4DequantFixtures` / `dequant_sgfp4_container_cpu`) on both CPU and Vulkan using identical container bytes across the two placements; a converter-path round-trip test drives `RemoveAndStoreParam`/`saveExternalData` on a synthetic `NetT` and asserts 16-byte-aligned, monotonic, non-overlapping sidecar layout, `external == {offset, true-size}`, buffer cleared in the serialized op, and reload+decode parity.
5. `SGFP4TestUtil.hpp` extracted from the duplicated helpers in `SGFP4ClassicAPITest.cpp` / `SGFP4MultiTensorTest.cpp` / `SGFP4InjectTest.cpp` (retrofitted onto it; correct region-relative offset convention from `SGFP4MultiTensorTest.cpp:190-199`), and the Phase 11 hand-off contract is documented: buffer-staging convention (pass writes `buffer`, `external = {}`, no `externalPath` — zero byte I/O) plus the non-interception note that `SGFP4Dequant` intentionally stays out of `createExecutionWithExternal`.

**Plans**: 6/6 plans complete

Plans:
**Wave 1**

- [x] 08-01-PLAN.md — Append `buffer:[byte]` to `SGFP4DequantParam` + regenerate/commit schema headers (SGV2-22, D-04 regression)
- [x] 08-02-PLAN.md — Extract `SGFP4TestUtil.hpp` and retrofit the three test files (D-10 region-relative builder)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 08-03-PLAN.md — Buffer-first dispatch in CPU + Vulkan decoders + D-12 non-interception comment
- [x] 08-04-PLAN.md — Aligned `storeWeight` + `loadExternalParam` SGFP4 cases in `RemoveParams.cpp` (SGV2-23)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 08-05-PLAN.md — Buffer-mode decode-parity suites (CPU + Vulkan) vs oracle (D-08)
- [x] 08-06-PLAN.md — Converter round-trip test target `TestSGFP4Converter` (D-09)

### Phase 9: Real-Weight C++ Encoder Port

**Goal**: Port the Python quadtree/dual-mode SGFP4 v2 encoder (`fp4_exporter.py --adaptive`) to C++ as `tools/fp4/sgfp4_encode.hpp/.cpp`, operating on real (non-64-aligned) weight shapes; extend CPU and Vulkan decoders with a minimal padded-crop path (D-11a); verify decode-parity (rtol 1e-4) against Python goldens on both CPU oracle and Vulkan Execution; ship deterministic real-shape golden fixtures (D-05).
**Depends on**: Phase 8
**Requirements**: SGV2-24, SGV2-25

**Plans**: 5 plans

Plans:
**Wave 1** *(independent — run in parallel)*

- [ ] 09-01-PLAN.md — Encoder core `sgfp4_encode.hpp/.cpp` + CMake `sgfp4_encode` STATIC lib (SGV2-24, SGV2-25)
- [ ] 09-02-PLAN.md — Padded-crop decode path: CPU oracle new overload, `CPUSGFP4Dequant` dispatch, Vulkan Execution + GLSL shader + mandatory regen (SGV2-25)

**Wave 2** *(blocked on 09-01)*

- [ ] 09-03-PLAN.md — Golden fixture generator `author_real_shape_fixture.py` + committed `SGFP4RealShapeFixtures.h` (SGV2-24, SGV2-25)

**Wave 3** *(blocked on 09-01, 09-02, 09-03 — run in parallel)*

- [ ] 09-04-PLAN.md — CPU encode-parity + edge-case tests `SGFP4EncodeTest.cpp` "op/sgfp4/encode" (SGV2-24, SGV2-25)
- [ ] 09-05-PLAN.md — Vulkan encode-parity tests `SGFP4VulkanEncodeParityTest.cpp` "op/sgfp4/vulkan_encode_parity" (SGV2-24, SGV2-25)

### Phase 10: Real-Weight Validation Against Actual Model Statistics

**Goal**: Validate/revise encoder parameters against a real model's weight distributions before graph-rewrite integration (synthetic-fixture-tuned assumptions are the top-flagged risk).
**Depends on**: Phase 9
**Requirements**: SGV2-26, SGV2-27
**Success Criteria**: TBD at plan time (phase detail to be finalized when planned).

**Plans**: 0/TBD

### Phase 11: Graph-Rewrite PostConverter Pass + CLI Flag

**Goal**: New `PostConverter` pass inserts `SGFP4Dequant` nodes (buffer-staged per Phase 8's D-11 contract); new CLI flag triggers it; `WeightQuantAndCoding.cpp` skip-guard prevents double-processing. Absorbs the v2.0 injection tool's sidecar/rewiring learnings and retires tech debt W-1 (classic_api offset-convention retrofit) and W-2 (arg-stage failCleanup) per the v2.0 milestone audit placement.
**Depends on**: Phase 8, Phase 10
**Requirements**: SGV2-28, SGV2-29, SGV2-30
**Success Criteria**: TBD at plan time (phase detail to be finalized when planned).

**Plans**: 0/TBD

### Phase 12: End-to-End Validation

**Goal**: A real model converts and runs correct inference on CPU and Vulkan via the new flag.
**Depends on**: Phase 11
**Requirements**: SGV2-31, SGV2-32
**Success Criteria**: TBD at plan time (phase detail to be finalized when planned).

**Plans**: 0/TBD

**(Renumbering provenance: Phase 8 was Phase 5, 9←6, 10←7, 11←8, 12←9. Dependencies: Phase 8 ← Phase 4 (v1.0) + v2.0 learnings; Phase 9 ← 8; Phase 10 ← 9; Phase 11 ← 8, 10; Phase 12 ← 11.)**

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 1. Affine Dual-Mode Decode Core (CPU, Uniform) | v1.0 | 2/2 | Complete | 2026-08-24 |
| 2. Adaptive Quadtree Layout (CPU, LAYOUT_MIXED) | v1.0 | 2/2 | Complete | 2026-08-24 |
| 3. Vulkan Decode — Uniform Layouts | v1.0 | 4/4 | Complete | 2026-08-25 |
| 4. Vulkan Decode — Adaptive Quadtree | v1.0 | 2/2 | Complete | 2026-08-25 |
| 5. Injection Core — Artifact Construction & Graph Splicing | v2.0 | 2/2 | Complete | 2026-08-26 |
| 6. Classic-API Load & Run Validation | v2.0 | 2/2 | Complete    | 2026-08-27 |
| 7. Multi-Tensor Hardening & Structured-Data Coverage | v2.0 | 3/3 | Complete    | 2026-08-28 |
| 8. Schema + Sidecar Wiring | v3.0 | 6/6 | Complete    | 2026-08-28 |
| 9. Real-Weight C++ Encoder Port | v3.0 | 0/TBD | Not started | - |
| 10. Real-Weight Validation Against Actual Model Statistics | v3.0 | 0/TBD | Not started | - |
| 11. Graph-Rewrite PostConverter Pass + CLI Flag | v3.0 | 0/TBD | Not started | - |
| 12. End-to-End Validation | v3.0 | 0/TBD | Not started | - |
