# Roadmap: MNN — SGFP4 v2 Quadtree-Adaptive Quantization

## Milestones

- ✅ **v1.0 SGFP4 v2 Decode (Vulkan-parity)** — Phases 1-4 (shipped 2026-08-26)
- ✅ **v2.0 SGFP4 v2 Model-Artifact Injection Tool** — Phases 5-7 (shipped 2026-08-28)
- ✅ **v3.0 SGFP4 v2 Converter Integration** — Phases 8-12 (shipped 2026-09-02)

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

<details>
<summary>✅ v3.0 SGFP4 v2 Converter Integration (Phases 8-12) — SHIPPED 2026-09-02</summary>

Full detail archived to `.planning/milestones/v3.0-ROADMAP.md`; requirements archived to `.planning/milestones/v3.0-REQUIREMENTS.md` (11/11 complete).

**Milestone Goal (shipped):** A pytorch → onnx → mnnconvert pipeline can produce a `.mnn` file that runs SGFP4 v2 (quadtree-adaptive) FP4 inference, via a native `mnnconvert --sgfp4` CLI flag. Shipped as the `InsertSGFP4Dequant` PostConverter pass + byte-exact C++ encoder (`tools/fp4/sgfp4_encode`) + `tools/fp4/e2e_validation.ps1` CPU+Vulkan accuracy gate (PASS vs FP32 baseline, locked tolerances). Two correctness-critical codec fixes landed under its E2E gate (spatial decode convention; encoder split-map coordinates/order). v2.0 tech debt W-1/W-2/W-3 retired in Phase 11.

- [x] Phase 8: Schema + Sidecar Wiring (6/6 plans) — completed 2026-08-28
- [x] Phase 9: Real-Weight C++ Encoder Port (5/5 plans) — completed 2026-08-31
- [x] Phase 10: Real-Weight Validation Against Actual Model Statistics (3/3 plans) — completed 2026-09-01
- [x] Phase 11: Graph-Rewrite PostConverter Pass + CLI Flag (5/5 plans) — completed 2026-09-01
- [x] Phase 12: End-to-End Validation (2/2 plans) — completed 2026-09-02

</details>

**Terminology lock:** the format is called **"SGFP4 v2"** — never "Ultra FP4" (that is the sibling `milestone` workstream's unrelated E2M1/`FP4_ULTRA` format, despite gnus-poc manifest labels suggesting otherwise).

## Phase Details

*(v1.0 phase details: see `.planning/milestones/v1.0-ROADMAP.md`. v2.0: `.planning/milestones/v2.0-ROADMAP.md`. v3.0: `.planning/milestones/v3.0-ROADMAP.md`.)*

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
| 9. Real-Weight C++ Encoder Port | v3.0 | 5/5 | Complete    | 2026-08-31 |
| 10. Real-Weight Validation Against Actual Model Statistics | v3.0 | 3/3 | Complete    | 2026-09-01 |
| 11. Graph-Rewrite PostConverter Pass + CLI Flag | v3.0 | 5/5 | Complete    | 2026-09-01 |
| 12. End-to-End Validation | v3.0 | 2/2 | Complete | 2026-09-02 |

---
*Next milestone: not yet defined — run `/gsd-new-milestone --ws sgfp4-pivot` to define v4.0 requirements and roadmap. Deferred candidates: SGV2-33..37 (Performance & Coverage, see `.planning/milestones/v3.0-REQUIREMENTS.md`).*

