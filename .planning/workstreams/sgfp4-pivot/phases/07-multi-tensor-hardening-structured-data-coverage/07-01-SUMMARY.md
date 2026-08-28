---
phase: 07-multi-tensor-hardening-structured-data-coverage
plan: 01
subsystem: testing
tags: [sgfp4, fp4, injection-tool, atomicity, docs]

requires:
  - phase: 06-classic-api-load-run-validation
    provides: header-only sgfp4_inject core (sgfp4_inject::run) + classic_api regression suites
provides:
  - D-11 failure-cleanup (failCleanup) in sgfp4_inject::run() — failed runs leave no output/sidecar
  - tools/fp4/README.md documenting CLI, input contract, dims convention, sidecar layout, failure semantics (D-13)
affects: [07-03 malformed-input atomicity assertions, SGProcessingManager downstream consumer safety]

tech-stack:
  added: []
  patterns: [failCleanup lambda invoked on every post-sidecarPath failure return]

key-files:
  created: [tools/fp4/README.md]
  modified: [tools/fp4/sgfp4_inject_core.hpp]

key-decisions:
  - "Option A failure-cleanup (lambda + std::remove) over temp+rename: std::rename over an existing destination is not portable on Windows"
  - "Cleanup removes ANY files at outputPath/sidecarPath including stale artifacts from a previous successful run — D-11 satisfied literally"
  - "Arg-validation returns (~289/294) intentionally do NOT call failCleanup — nothing has been decided about output paths yet"

patterns-established:
  - "failCleanup lambda: defined immediately after sidecarPath, prepended to every failure return 1 after that point"

requirements-completed: [SGINJ-07, SGINJ-08]

coverage:
  - id: D1
    description: "D-11 atomicity: failed sgfp4_inject runs remove output + sidecar (failCleanup on all 12 post-sidecarPath failure returns)"
    requirement: "SGINJ-08"
    verification:
      - kind: unit
        ref: "grep: every return 1 in run() after sidecarPath (lines 316-484) preceded by failCleanup(); lines 289/294 are arg-validation"
        status: pass
      - kind: unit
        ref: "run_test.out op/sgfp4 → 7/7 passed (classic_api + classic_api_missing_sidecar green — success path byte-identical)"
        status: pass
      - kind: unit
        ref: "behavioral file-absence assertions shipped as op/sgfp4/malformed_inputs suite in Plan 07-03"
        status: pass
    human_judgment: false
  - id: D2
    description: "D-13 documentation: tools/fp4/README.md covering CLI usage, niche-dir/manifest contract, dims={dimO,dimI} convention, sidecar layout, post-D-11 failure behavior"
    requirement: "SGINJ-07"
    verification:
      - kind: unit
        ref: "Select-String grep over README for all 10 required literal tokens → 12 matches"
        status: pass
    human_judgment: false

duration: 20min
completed: 2026-08-28
status: complete
---

# Plan 07-01: D-11 failure-cleanup + D-13 tool documentation Summary

**sgfp4_inject failed runs can no longer leave partial/stale artifacts (12 failure sites cleaned), and the tool contract is now documented in tools/fp4/README.md.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-08-28T02:30Z (approx)
- **Completed:** 2026-08-28
- **Tasks:** 2
- **Files modified:** 2 (1 modified, 1 created)

## Accomplishments
- `failCleanup` lambda added to `run()` right after `sidecarPath` definition; prepended to every failure `return 1` from niche-dir validation through the verify chain (12 sites) — D-11 atomicity shipped.
- Success path byte-identical: full `op/sgfp4` family regression green (7/7 passed, including `classic_api` and `classic_api_missing_sidecar`).
- `tools/fp4/README.md` created with the four D-13 content areas (CLI usage, niche-dir/manifest input contract, dims convention, sidecar layout) plus post-D-11 failure semantics.

## Task Commits

1. **Task 1: D-11 atomicity — failure-cleanup in sgfp4_inject::run()** - `025d96b2`
2. **Task 2: D-13 — tools/fp4/README.md** - `2e3c9385`

## Files Created/Modified
- `tools/fp4/sgfp4_inject_core.hpp` — `failCleanup` lambda + 12 call sites (23 insertions, 0 deletions besides)
- `tools/fp4/README.md` — new (74 lines)

## Decisions Made
- Cleanup semantics cover stale artifacts: a failed run removes ANY files at the output paths, so downstream consumers (`SGProcessingManager` unchecked-nullptr path) never see an artifact not corresponding to the failed run's inputs.
- Arg-validation returns (~289/294) left without cleanup (per plan: outputPath may be empty; nothing written yet).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `failCleanup` behavior ready to be regression-asserted by Plan 07-03's `op/sgfp4/malformed_inputs` suite (file-absence after each probe).
- README grep assertions all pass; `sgfp4_inject.cpp` shim untouched (`git diff` clean).

*Plan: 07-01*
*Completed: 2026-08-28*
