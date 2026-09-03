---
phase: 07-multi-tensor-hardening-structured-data-coverage
plan: 02
subsystem: testing
tags: [sgfp4, fp4, fixture, quadtree, layout-mixed, gnus-poc]

requires:
  - phase: 05-injection-core-artifact-construction-graph-splicing
    provides: niche-dir/manifest input contract the fixture must satisfy
provides:
  - tools/fp4/author_structured_fixture.py — re-runnable authoring script calling the REAL gnus-poc FP4Exporter.export_weights(..., adaptive=True)
  - test/op/SGFP4StructuredFixtures.h — self-contained structured LAYOUT_MIXED container C-array fixture (140240 bytes, 12 MIXED superblocks)
affects: [07-03 multi-tensor suite (consumes kStructuredMixedData)]

tech-stack:
  added: [numpy (authoring-time only, gnus-poc repo env)]
  patterns: [deterministic fixture authoring — no timestamps, seeded RNG, byte-identical regeneration asserted via fc.exe]

key-files:
  created: [tools/fp4/author_structured_fixture.py, test/op/SGFP4StructuredFixtures.h]
  modified: []

key-decisions:
  - "Weight recipe amplitude iterated to 12.0 (from 1.0): size-64 gate is max_mse 0.01 — amp-1.0 ramp quantizes at MSE ~1e-4 (stays uniform, first run produced 64/64 UNIFORM_64); amp-12 TL-quadrant ramp splits while flat quadrants accept → MIXED (matches STATE.md Phase-2 encoder finding)"
  - "Assert uses stats[\"layout_distribution\"][4] > 0 literally; loud non-zero exit before any header emission on failure"
  - "Distribution 52× UNIFORM_64 + 12× MIXED — both kinds present, preferred per plan"

patterns-established:
  - "Real-encoder fixture authoring: cross-repo dependency (gnus-poc FP4Exporter) exists only at authoring time; tests consume the frozen C-array only (D-01..D-03)"

requirements-completed: [SGINJ-08]

coverage:
  - id: D1
    description: "Structured LAYOUT_MIXED container from the real gnus-poc encoder frozen as committed C-array fixture with provenance"
    requirement: "SGINJ-08"
    verification:
      - kind: unit
        ref: "python tools/fp4/author_structured_fixture.py → exit 0, layout_distribution {0:52, 4:12}, MIXED assert passed"
        status: pass
      - kind: unit
        ref: "fc.exe /b committed header vs TEMP regeneration → no differences (deterministic, git diff clean after verify run)"
        status: pass
      - kind: unit
        ref: "grep header: kStructuredMixedData / kStructuredDimO = 512 / kStructuredMixedCount = 12 / sha256 9ebb8c1f... / layout_distribution provenance present"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-08-28
status: complete
---

# Plan 07-02: Structured fixture from real gnus-poc encoder Summary

**First real-encoder LAYOUT_MIXED container: 512×512 structured weights → 140,240-byte SGFP4 v2 container with 12 MIXED + 52 uniform superblocks, frozen as a self-contained C-array fixture with byte-identical regeneration.**

## Performance

- **Duration:** ~15 min
- **Completed:** 2026-08-28
- **Tasks:** 1
- **Files modified:** 2 (both created)

## Accomplishments
- `tools/fp4/author_structured_fixture.py`: calls `FP4Exporter.export_weights(weights, "phase7_structured", adaptive=True)` programmatically (never the dummy-noise `__main__`); mandatory `layout_distribution[4] > 0` self-assert; v2 framing validated before emitting; deterministic (no timestamps, seed-20260828 RNG only).
- `test/op/SGFP4StructuredFixtures.h`: 140,240-byte container as `kStructuredMixedData[]` + `kStructuredDimO/DimI = 512` + `kStructuredSize` + `kStructuredMixedCount = 12`; provenance block records recipe, full layout_distribution, sha256 `9ebb8c1f8530ab3000c680007759122b5ffee612e3e0d35bfdc9cd23a9ed4257`, byte length.
- Retires the STATE.md pending todo "structured second artifact required before Phase 7".

## Task Commits

1. **Task 1: Authoring script + structured container export with MIXED self-assert** - `c72f1d1b` (script + committed header; one atomic deliverable per plan)

## Files Created/Modified
- `tools/fp4/author_structured_fixture.py` — new (153 lines)
- `test/op/SGFP4StructuredFixtures.h` — new (8,789 lines generated)

## Decisions Made
- Ramp amplitude fixed at 12.0 based on the gate arithmetic: DEFAULT_V2_THRESHOLDS[64] = {max_mse: 0.01, max_relative: 0.05}; an amp-1.0 ramp leaves whole-block quantization MSE ~1e-4 → uniform collapse (observed: first run {0:64}); amp-12 forces split, then asymmetry (only TL quadrant ramps) forces different leaf depths → MIXED.
- Recipe: background 0.002 ± 0.00025 seeded texture; every 2nd-row/3rd-column superblock gets diagonal ramp 0.05..12.0 (outer-product/12) confined to its TL 32×32 quadrant.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Recipe amplitude produced all-uniform container on first run**
- **Found during:** Task 1 (first authoring run)
- **Issue:** Plan's suggested "amplitude ~1.0" ramp quantizes with whole-block MSE ~1e-4, far under the size-64 gate max_mse=0.01 → all 64 superblocks stayed UNIFORM_64; MIXED assert failed (loud, exit 1) as designed.
- **Fix:** Read quadtree.py/laplacian.py threshold arithmetic; raised ramp amplitude to 12.0 (matching the STATE.md-recorded Phase-2 finding "TL-quadrant ramp amp 12 = asymmetric MIXED"). Plan explicitly allowed iterating amplitudes until the assert passes.
- **Files modified:** tools/fp4/author_structured_fixture.py
- **Verification:** Re-run → {0:52, 4:12} distribution, exit 0, deterministic round-trip.
- **Committed in:** c72f1d1b

**2. [Rule 2 - Style] Assert literal form**
- **Found during:** Task 1 (acceptance-criteria check)
- **Issue:** Initial implementation used a manual get()+if instead of the plan's literal `assert stats["layout_distribution"][4] > 0`.
- **Fix:** Switched to the literal assert (acceptance criterion greps for it); failure remains loud via traceback with the distribution in the printed summary.
- **Files modified:** tools/fp4/author_structured_fixture.py
- **Verification:** Re-run exit 0; byte-identical output unchanged.
- **Committed in:** c72f1d1b

---

**Total deviations:** 2 auto-fixed (1 bug-class recipe tuning, 1 style conformance)
**Impact on plan:** None — both behaviors the plan actually required (MIXED present, loud failure) now hold.

## Issues Encountered
None beyond the recipe tuning above.

## User Setup Required
None - no external service configuration required. (Authoring-time numpy/gnus-poc env already present on this machine.)

## Next Phase Readiness
- `kStructuredMixedData` / `kStructuredDimO=512` / `kStructuredDimI=512` / `kStructuredSize` / `kStructuredMixedCount=12` ready for Plan 07-03's multi-tensor suite (structured niche dir A).

*Plan: 07-02*
*Completed: 2026-08-28*
