---
status: passed
phase: 12-end-to-end-validation
verified: 2026-09-02
consolidated: true  # generated at v3.0 milestone close from 12-UAT.md (10/10 pass, 2026-09-02) + plan summaries — v2.0 Phase-5 consolidation precedent
requirements: [SGV2-31, SGV2-32]
score: 4/4
---

# Phase 12 Verification Report (Consolidated)

**Method**: goal-backward check of every ROADMAP success criterion (D-11/D-12 exit-code honesty; SGV2-31 CPU gate; SGV2-32 Vulkan gate; committed documented artifact) against executed evidence. Evidence sources: `12-UAT.md` (10/10 automated tests passed 2026-09-02, several independently rerun by the verifier), `12-01-SUMMARY.md`, `12-02-SUMMARY.md`, and `11-VERIFICATION.md` (Phase 11 handoff).

## Success Criteria vs. Evidence

### 1. D-11 exit-code honesty (converter prerequisite) — VERIFIED
- `RunNetPass` bool contract + SGFP4-gated nullptr failure chain: RunNetPass(false) → optimizeNetImpl(nullptr + MNN_ERROR) → convertModel(false, null-guard in cli.cpp) → main(exit 1) — `TestSGFP4Converter.exe` exit 0 with PASS incl. T10 sibling assertion (UAT test 5).
- Negative leg: corrupt.onnx + `--sgfp4` → exit 1, no "Converted Success!" (UAT tests 1, 4).
- D-12 flag-off byte-identical behavior: corrupt flag-off exit 0 (same messages), corpus flag-off exit 0 with artifact (UAT test 6, rerun by verifier).
- All 13 `op/sgfp4` suites (13/13) + `TestSGFP4Converter` green (UAT test 7, rerun by verifier: passed:13 failed:0 skipped:0).

### 2. SGV2-31 CPU gate — VERIFIED
- `tools/fp4/e2e_validation.ps1 -Corpus <alexnet.onnx>`: converts FP32 + `--sgfp4`, runs both artifacts via the classic API on CPU, SGFP4 output matches FP32 baseline within locked tolerances — `PASS: cpu max-abs=5.07216500E+000 (idx 533), max-rel=2.37302592E+002 (idx 573)`; `E2E VALIDATION: PASS`, exit 0 (UAT tests 2, 9, verifier-rerun).

### 3. SGV2-32 Vulkan gate — VERIFIED
- Same artifact via classic API + `MNN_FORWARD_VULKAN` (Precision_High): `PASS: vulkan` with argmax diagnostics (idx 533/638); genuine-device assertion `vulkan backend confirmed: backendType is 7` from stdout + vulkaninfo pre-check; no-Vulkan = exit 2 hard FAIL, never SKIP (D-07) (UAT tests 2, 3).

### 4. One committed, documented artifact — VERIFIED
- Tolerances locked with recorded derivation: TolAbs=10.14433 / TolRel=948.601606 (measured-worst × 2.0: cpu max-abs 5.072165 / vulkan max-abs 3.926990; dated 2026-09-01, commit 54bbeaf8, Phase 10 anchor, 1e-5 text-dump floor caveat) (UAT test 8).
- D-10 diagnostics on every verdict line (per-backend max-abs/max-rel + argmax indices) (UAT test 9).
- `tools/fp4/README.md` documents usage, tolerance derivation, hard Vulkan requirement; zero "Ultra FP4" occurrences (terminology lock honored).

## Correctness Fixes Surfaced by the Gate (Rule 3, deviation-documented)

The accuracy gate surfaced two codec blockers, fixed within 12-02 (commits c6d6906e..54bbeaf8 lineage):
1. **Spatial decode convention**: runtime decoders moved to the normative SPATIAL padded-plane convention (gnus-poc `decode_v2`); the legacy flat-stream append is only plane-correct for one-superblock-wide grids (multi-column, tiles_x ≥ 2 exposed it).
2. **Encoder split-map bugs**: global-vs-local coordinates + TL/TR/BL/BR child order in `buildSplitMapBits`; C++ containers now pass strict python `decode_v2` with errors identical to the python exporter's own roundtrip — independently reproduced by the verifier on a fresh 250×128 seed-42 MIXED weight (UAT test 10: maxAbs 0.087659 = FP4-noise level).

## Requirement Traceability

| Req | Status | Evidence |
|---|---|---|
| SGV2-31 | Met | Criteria 1, 2, 4; commits c6d6906e, a0728c4c, 54bbeaf8 |
| SGV2-32 | Met | Criterion 3 |

## Notes & Carried Items

- None open. The 4-D dims convention deviation (Phase 11) was exercised end-to-end here on both backends and survived the accuracy gate.
