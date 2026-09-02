---
status: passed
phase: 09-real-weight-c-encoder-port
verified: 2026-09-02
consolidated: true  # generated at v3.0 milestone close from 09-UAT.md (5/5 pass, 2026-08-31) + plan summaries — v2.0 Phase-5 consolidation precedent
requirements: [SGV2-24, SGV2-25]
score: 5/5
---

# Phase 9 Verification Report (Consolidated)

**Method**: goal-backward check of the ROADMAP goal (Python quadtree/dual-mode encoder ported to C++ for real non-64-aligned shapes; padded-crop decode path; decode-parity rtol 1e-4 vs Python goldens on CPU + Vulkan; deterministic committed fixtures) against executed evidence. Evidence sources: `09-UAT.md` (5/5 automated tests passed 2026-08-31), the five plan SUMMARYs, and the Phase 12 independent re-verification.

## Goal Truths vs. Evidence

### 1. Encoder core shipped (SGV2-24) — VERIFIED
- `tools/fp4/sgfp4_encode.hpp/.cpp`: faithful functional port of gnus-poc `fp4_exporter.py --adaptive` + `quadtree.py` + `laplacian.py` — separable Gaussian Laplacian pyramid (sigma=2·2^level, reflect boundary, truncate 4.0), `_fit_affine`/`_fit_ternary` in double accumulation, `std::rint` half-to-even rounding, `_try_block` with DEFAULT_V2_THRESHOLDS, `_classify_layout` incl. LAYOUT_FULL_4X4 (D-11b), 16-byte-aligned container assembly, framing constants reused from `MNN::kSGFP4*` (09-01-SUMMARY; commits 510fad98, ff822e7c).
- Input-validation gates: non-finite weights / non-positive dims / dims > 65536 / null → empty-vector contract; all-zero input → valid zero container (UAT test 5).
- **Byte-exactness**: post ff822e7c fixes (packNibbles double-alloc; RNE `half_cast` explicit-rounding template defeating MSVC COMDAT folding), C++ containers are byte-identical to the gnus-poc exporter (0 diff bytes on 100×36 and 250×128) — independently re-proven in Phase 12 UAT test 10 on a fresh 250×128 seed-42 MIXED weight (python `decode_v2` maxAbs 0.087659 = FP4-noise level).

### 2. Padded-crop decode path, both backends (SGV2-25) — VERIFIED
- CPU oracle new overload + `CPUSGFP4Dequant` dispatch + Vulkan Execution/GLSL (shader regen per CLAUDE.md rule) — UAT test 3: 7/7 fixtures incl. padded shapes (100×36, 37×91, 64×36, 5×5, 1×1) decode to TRUE dims on a real Vulkan device (Precision_High), 100×36 crop probe confirms row-boundary correctness.
- Internal zero-padding of non-64-aligned planes in the encoder (D-06) — decoder consumes unchanged containers.

### 3. Decode-parity vs Python goldens (SGV2-24) — VERIFIED
- `op/sgfp4/encode` suite: 7/7 real-shape fixtures decode back to the Python `decode_v2` reference at rtol 1e-4, `SGF4` magic + version 0x02 on every container (UAT test 1).

### 4. Deterministic committed fixtures (D-05) — VERIFIED
- `author_real_shape_fixture.py` regenerates `SGFP4RealShapeFixtures.h` byte-identical (seeded RNG, sha256 provenance, no timestamps) — UAT test 4 (Compare-Object clean, git status clean).

### 5. No regression — VERIFIED
- Full `op/sgfp4` family 13/13 green after the encoder/decoder changes (UAT test 2).

## Requirement Traceability

| Req | Status | Evidence |
|---|---|---|
| SGV2-24 | Met | Truths 1, 3; commits 510fad98, ff822e7c |
| SGV2-25 | Met | Truths 1 (padding), 2 |

## Notes & Carried Items

- Deviation (documented in 09-01-SUMMARY): pre-existing broken dead code `test/op/FP4ModelTest.cpp` (milestone workstream, STATE.md blocker) was minimally repaired to link `run_test.out` (orphaned second `run()`-body fragment deleted; intact `FP4ModelConversionTest` preserved). Recommended the `milestone` workstream ratify.
- Spatial decode convention defect found later in Phase 12 (multi-column grids, tiles_x ≥ 2) was fixed at ff822e7c-lineage commit 54bbeaf8 in Phase 12 — the Phase 9 parity suites predated the exposing shape class; final cross-repo convention parity is proven in `12-UAT.md` test 10.
