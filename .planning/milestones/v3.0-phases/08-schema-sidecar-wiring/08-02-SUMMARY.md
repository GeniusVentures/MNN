---
phase: 08-schema-sidecar-wiring
plan: 02
subsystem: testing
tags: [test-infrastructure, dedup, sgfp4]

requires:
  - phase: 08-01 buffer field
    provides: schema surface (not directly consumed; independent wave-1 plan)
provides:
  - test/op/SGFP4TestUtil.hpp — shared header-only sgfp4_test helpers (region-relative builder canonical)
  - W-1 offset-convention fix pulled forward into classic_api (absolute-offset builder deleted)
affects: [08-05 buffer tests, 08-06 converter test (consumes SGFP4TestUtil.hpp)]

tech-stack:
  added: []
  patterns: [header-only inline helper namespace shared across test TUs]

key-files:
  created: [test/op/SGFP4TestUtil.hpp]
  modified: [test/op/SGFP4ClassicAPITest.cpp, test/op/SGFP4MultiTensorTest.cpp, test/op/SGFP4InjectTest.cpp]

key-decisions:
  - "Canonical builder = SGFP4MultiTensorTest's GENERALIZED REGION-RELATIVE variant (entries b*kRecordSize relative to record-region start); classic_api's absolute-offset variant deleted (W-1 fix pulled forward per ROADMAP success criterion 5)"
  - "All helpers inline in namespace sgfp4_test — C++11 ODR-safe for multi-TU inclusion; no MNNTestSuite.h dependency so tools (TestSGFP4Converter) can include it"
  - "Framing constants (kRecordSize family) moved into the header; classic_api keeps only non-helper constants it still references (kMatrixDim, kRecordCount, LCG, tolerance)"
  - "D-13 reconciliation: only the builder migration moved into Phase 8; the remaining W-1 classic_api behavioral retrofit stays Phase 11"

patterns-established:
  - "sgfp4_test:: shared helper namespace for all SGFP4 test/container construction"

requirements-completed: [SGV2-22, SGV2-23]

coverage:
  - id: D1
    description: "Single shared SGFP4TestUtil.hpp with the region-relative container builder; three test files retrofitted; no local helper definitions remain"
    requirement: "SGV2-22"
    verification:
      - kind: unit
        ref: "run_test.out op/sgfp4/ (classic_api, multi_tensor, inject, malformed_inputs all green post-retrofit) — exit 0"
        status: pass
      - kind: unit
        ref: "grep gate: no 'bool buildContainerUniform64|bool tempPath|void writeU32Le' definitions in the three files; kRecordRegionStart + b absent from header"
        status: pass
    human_judgment: false

duration: 55min
completed: 2026-08-28
status: complete
---

# Phase 08 Plan 02: SGFP4 test helper dedup

**Duplicated SGFP4 test helpers unified into SGFP4TestUtil.hpp with the region-relative builder canonical; classic_api's absolute-offset (W-1) copy deleted.**

## Performance

- **Duration:** ~55 min
- **Tasks:** 2/2
- **Files modified:** 4 (1 created, 3 retrofitted); net −75 lines

## Accomplishments
- Created `test/op/SGFP4TestUtil.hpp`: `sgfp4_test::` namespace, 10 inline helpers (tempPath, cwdPath, makeDir, removeDir, fileExists, writeU32Le, writeBytes, readBytes, buildContainerUniform64, writeNicheDir) + framing constants
- `SGFP4MultiTensorTest.cpp`: local helper block + framing constants deleted; keepers (writeNicheDirRawManifest, manifestJsonFor) rebuilt on sgfp4_test:: primitives; all call sites qualified
- `SGFP4ClassicAPITest.cpp`: ABSOLUTE-offset local builder REPLACED by `sgfp4_test::buildContainerUniform64(kMatrixDim, kMatrixDim, ...)` — the W-1 offset-convention divergence is gone; local writeNicheDir replaced by the shared parameterized form preserving the exact `phase6_fixture.sgfp4` layout
- `SGFP4InjectTest.cpp`: tempPath/writeBytes duplicates removed, call sites qualified
- Grep gates: zero local definitions of the moved helpers across the three files; `kRecordRegionStart + b` (absolute form) absent from the header; ≥10 `inline` functions present
- All affected suites green post-retrofit (classic_api, classic_api_missing_sidecar, multi_tensor, inject, inject_v1_reject, malformed_inputs)

## Deviations
- None.

## Issues
- None.
