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

- [ ] **Phase 8: Schema + Sidecar Wiring** - Add `buffer:[byte]` to `SGFP4DequantParam` and wire `RemoveParams.cpp` to externalize it through the existing shared sidecar mechanism
- [ ] **Phase 9: Real-Weight C++ Encoder Port** - Port the Python quadtree/dual-mode encoder to C++, operating on real (non-64-aligned) weight tensor shapes
- [ ] **Phase 10: Real-Weight Validation Against Actual Model Statistics** - Validate/revise encoder parameters against a real model's weight distributions before graph-rewrite integration
- [ ] **Phase 11: Graph-Rewrite PostConverter Pass + CLI Flag** - New `PostConverter` pass inserts `SGFP4Dequant` nodes; new CLI flag triggers it; `WeightQuantAndCoding.cpp` skip-guard prevents double-processing
- [ ] **Phase 12: End-to-End Validation** - A real model converts and runs correct inference on CPU and Vulkan via the new flag

**(Phase details for 8-12 preserved as previously roadmapped — goals, dependencies, and success criteria transfer with renumbering: Phase 8 was Phase 5, 9←6, 10←7, 11←8, 12←9. Dependencies: Phase 8 ← Phase 4 (v1.0) + v2.0 learnings; Phase 9 ← 8; Phase 10 ← 9; Phase 11 ← 8, 10; Phase 12 ← 11.)**

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
| 8. Schema + Sidecar Wiring | v3.0 | 0/TBD | Not started | - |
| 9. Real-Weight C++ Encoder Port | v3.0 | 0/TBD | Not started | - |
| 10. Real-Weight Validation Against Actual Model Statistics | v3.0 | 0/TBD | Not started | - |
| 11. Graph-Rewrite PostConverter Pass + CLI Flag | v3.0 | 0/TBD | Not started | - |
| 12. End-to-End Validation | v3.0 | 0/TBD | Not started | - |
