---
phase: quick-260821-rql
plan: 1
subsystem: quantization
tags: [fp4, e2m1, quantize_fp4.py, unittest, regression-test]

# Dependency graph
requires:
  - phase: quick-260821-p1q
    provides: SGFP4-PIVOT-ANALYSIS.md Section 1 root-cause analysis of the MAX_E2M1_VALUE calibration bug
provides:
  - Corrected MAX_E2M1_VALUE constant (3.0) in tools/fp4/quantize_fp4.py so channel max-magnitude weights no longer saturate to +/-Inf
  - Standalone stdlib-only regression test tools/fp4/test_quantize_fp4.py guarding against the constant regressing
affects: [phase-4-plan-04-02, sgprocessingmanager-fp4-consumer]

# Tech tracking
tech-stack:
  added: []
  patterns: [pure-Python mirror of a C++ decode function kept in sync via ground-truth test-vector table, used to validate an encoder without requiring a build]

key-files:
  created: [tools/fp4/test_quantize_fp4.py]
  modified: [tools/fp4/quantize_fp4.py]

key-decisions:
  - "Fixed MAX_E2M1_VALUE from 6.0 to 3.0 (true max finite E2M1 magnitude, nibble 0x5/0xD) rather than changing encode_fp4_e2m1's saturation branch, since the defect is entirely a calibration-constant bug"
  - "Regression test is self-contained (stdlib unittest/math/os/sys only) and mirrors dequant_e2m1_cpu in pure Python, validated first against FP4DequantUtils.hpp's ground-truth test-vector table before being trusted to validate the encoder"

patterns-established:
  - "Python-side test mirrors of C++ decode/encode logic must include a dedicated test proving the mirror matches the ground-truth table before using it to validate other code"

requirements-completed: []

coverage:
  - id: D1
    description: "MAX_E2M1_VALUE corrected to 3.0 in tools/fp4/quantize_fp4.py; channel max-magnitude weights no longer saturate to +/-Inf"
    verification:
      - kind: unit
        ref: "tools/fp4/test_quantize_fp4.py#MaxE2M1ValueCalibrationTest.test_max_e2m1_value_equals_true_max_finite_magnitude"
        status: pass
      - kind: unit
        ref: "tools/fp4/test_quantize_fp4.py#MaxE2M1ValueCalibrationTest.test_channel_max_magnitude_weight_round_trips_finite"
        status: pass
    human_judgment: false
  - id: D2
    description: "Self-contained stdlib-only regression suite (4 tests) proving the fix and guarding against recurrence, confirmed to fail against the pre-fix constant"
    verification:
      - kind: unit
        ref: "python tools/fp4/test_quantize_fp4.py -v (4/4 pass against fixed code; manually confirmed 3/4 fail when MAX_E2M1_VALUE is reverted to 6.0)"
        status: pass
    human_judgment: false

# Metrics
duration: 15min
completed: 2026-08-21
status: complete
---

# Quick Task 260821-rql: Fix MAX_E2M1_VALUE Scale-Calibration Bug Summary

**Corrected `MAX_E2M1_VALUE` from 6.0 to 3.0 in `tools/fp4/quantize_fp4.py` (the true max finite E2M1 magnitude) and added a 4-test stdlib-only regression suite in `tools/fp4/test_quantize_fp4.py` that fails if the constant ever regresses.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-08-21
- **Completed:** 2026-08-21
- **Tasks:** 2
- **Files modified:** 2 (1 modified, 1 created)

## Accomplishments
- Fixed the root-caused MAX_E2M1_VALUE scale-calibration defect from SGFP4-PIVOT-ANALYSIS.md Section 1: the divisor used to compute per-channel scale was 6.0, which guaranteed every channel's max-magnitude weight normalized to exactly 6.0 and hit `encode_fp4_e2m1`'s `biased_e >= 3` saturation branch, corrupting every channel's largest weight to +/-Inf on every quantization run.
- Added accurate rationale comments above the constant and on the saturation-return line explaining the correct semantics (largest finite E2M1 magnitude is 3.0, saturation applies to magnitudes >= 4.0).
- Created a standalone, stdlib-only regression suite (`unittest`, `math`, `os`, `sys` — no new dependencies) with 4 test methods: a fidelity check of a pure-Python mirror of `dequant_e2m1_cpu` against the ground-truth 16-entry E2M1 table in `FP4DequantUtils.hpp`, a direct guard on `MAX_E2M1_VALUE == 3.0`, a round-trip finiteness/precision check on a representative channel with tied max-magnitude elements, and a multi-channel finiteness sweep.
- Manually verified the regression suite actually catches the original bug: reverting `MAX_E2M1_VALUE` to 6.0 causes 3 of 4 tests to fail (`test_max_e2m1_value_equals_true_max_finite_magnitude`, `test_channel_max_magnitude_weight_round_trips_finite`, `test_multiple_channels_never_saturate_to_inf`); restored the fix and confirmed all 4 pass again after clearing a stale `__pycache__` artifact from the temporary revert.

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix MAX_E2M1_VALUE scale-calibration constant** - `13164404` (fix)
2. **Task 2: Add self-contained regression test for the round-trip finiteness fix** - `ee8d54b7` (test)

_No separate plan-metadata commit was created per this quick task's constraints (ROADMAP.md not updated); STATE.md and this SUMMARY.md are committed together as the final docs commit._

## Files Created/Modified
- `tools/fp4/quantize_fp4.py` - `MAX_E2M1_VALUE` changed from 6.0 to 3.0; added rationale comments above the constant and on the saturation-branch return in `encode_fp4_e2m1`
- `tools/fp4/test_quantize_fp4.py` - New standalone regression suite (`MaxE2M1ValueCalibrationTest`, 4 test methods) proving the fix and guarding against recurrence

## Decisions Made
- Fixed the calibration constant itself (6.0 -> 3.0) rather than touching `encode_fp4_e2m1`'s control flow, `pack_fp4_byte`, or `quantize_model`, since the defect is isolated entirely to the scale-calibration constant per the plan's scope and the SGFP4-PIVOT-ANALYSIS.md root cause.
- Kept the Python-side E2M1 decode mirror (`dequant_e2m1`) as a dedicated test-only function validated against the ground-truth table first, rather than importing/binding to the C++ implementation, to keep the test suite dependency-free and buildless.

## Deviations from Plan

None - plan executed exactly as written. Both tasks' `<verify>` blocks were run and passed as specified; the plan's overall `<verification>` step 3 (manually reverting to 6.0 and confirming the test suite fails) was also executed and confirmed as an extra correctness check beyond the per-task verify blocks.

## Issues Encountered
- During the manual revert-and-confirm step (plan `<verification>` item 3), a stale `tools/fp4/__pycache__/quantize_fp4.cpython-313.pyc` bytecode cache from the temporary revert initially caused the restored-fix test run to show the reverted (failing) behavior even though the source file was correctly restored to `MAX_E2M1_VALUE = 3.0`. Removed `tools/fp4/__pycache__` and re-ran; all 4 tests passed as expected. No cache directory was left behind or committed (git status confirms `tools/fp4/__pycache__` is absent).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 4 plan 04-02 (E2E FP4 model test) is no longer blocked by this defect: `quantize_channel_weights` now produces finite E2M1-encoded max-magnitude weights instead of unconditionally saturating them to +/-Inf.
- No files outside `tools/fp4/` were touched; `include/MNN/FP4DequantUtils.hpp`, `CPUFP4Dequant.cpp`, and `VulkanFP4Dequant.cpp` remain untouched as required, since the bug was isolated to the Python encoder's calibration constant.
- SGFP4-PIVOT-ANALYSIS.md's open questions (v1-vs-v2 target, container adoption depth, verifiability scope) remain unresolved and are unaffected by this fix.

---
*Phase: quick-260821-rql*
*Completed: 2026-08-21*

## Self-Check: PASSED

- FOUND: tools/fp4/quantize_fp4.py
- FOUND: tools/fp4/test_quantize_fp4.py
- FOUND: .planning/quick/260821-rql-fix-max-e2m1-value-scale-calibration-bug/260821-rql-SUMMARY.md
- FOUND: commit 13164404 (fix)
- FOUND: commit ee8d54b7 (test)
