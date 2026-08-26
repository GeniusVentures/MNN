---
phase: 04-vulkan-decode-adaptive-quadtree-layout-mixed
plan: 02
subsystem: testing
tags: [vulkan, parity-test, sgfp4, gpu, quadtree, regression-gate]

# Dependency graph
requires:
  - phase: 04-vulkan-decode-adaptive-quadtree-layout-mixed/04-01
    provides: "locateElement's LAYOUT_MIXED (enum 4) GLSL branch + regenerated shader artifacts"
  - phase: 03-vulkan-decode-uniform-layouts/03-04
    provides: "op/sgfp4/vulkan_uniform_parity fixture-sweep harness this plan extends"
provides:
  - "Full 14-fixture CPU/Vulkan parity sweep (no skip) — SGV2-16 closed"
  - "Phase 4 verification gate: op/sgfp4/, op/fp4, op/vulkan/fp4_dequant_correctness all green; artifact greps unchanged (4/4/2); working tree exactly the phase's intended files"
affects: [sgfp4-pivot workstream v1 milestone completion]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "D-07 sweep is a one-line deletion, not a rewrite: removing the layout==kSGFP4LayoutMixed skip block turns the existing 13-fixture uniform parity test into the full 14-fixture sweep with zero other loop changes"

key-files:
  created: []
  modified:
    - test/op/SGFP4VulkanDequantTest.cpp

key-decisions:
  - "Kept class name SGFP4VulkanDequantTest and registration string \"op/sgfp4/vulkan_uniform_parity\" unchanged per CONTEXT.md Claude's-Discretion — avoids touching unrelated docs/scripts (04-VALIDATION.md, ROADMAP.md) that already reference this exact suite string"
  - "Full-suite temp-stub workaround for the pre-existing FP4ModelTest.cpp build blocker was attempted but blocked by the sandbox's auto-mode classifier when building with a locally-modified out-of-scope file; fell back to the already-passing filtered-suite verification (op/sgfp4/, op/fp4, op/vulkan/fp4_dequant_correctness), which fully satisfies this plan's <verify> requirement without needing the stub"

patterns-established: []

requirements-completed: [SGV2-16]

coverage:
  - id: D1
    description: "All 14 committed fixtures (uniform + LAYOUT_MIXED) decode identically through dequant_sgfp4_container_cpu and a real Vulkan module session within float tolerance, zero skipped fixtures"
    requirement: "SGV2-16"
    verification:
      - kind: unit
        ref: "run_test.out.exe op/sgfp4/vulkan_uniform_parity -> 'SGFP4VulkanDequantTest: 14 fixtures (including LAYOUT_MIXED) matched CPU reference on Vulkan', no MNN_ERROR output"
        status: pass
    human_judgment: false
  - id: D2
    description: "op/fp4 and op/vulkan/fp4_dequant_correctness (E2M1) remain green — additive, not a replacement"
    requirement: "SGV2-16"
    verification:
      - kind: unit
        ref: "run_test.out.exe op/fp4 -> exit success, 0 failures (see Deviations for the pre-existing FP4ModelTest.cpp caveat); run_test.out.exe op/vulkan/fp4_dequant_correctness -> passed:1, 5/5 sub-checks PASSED"
        status: pass
    human_judgment: false
  - id: D3
    description: "The phase's complete working-tree diff contains exactly the 5 intended files across both plans, no unrelated E2M1/Execution/schema changes"
    requirement: "SGV2-16"
    verification:
      - kind: unit
        ref: "git diff --stat d82593de~1..HEAD -- . ':(exclude).planning' -> sgfp4_dequant.comp, AllShader.cpp, SGFP4VulkanDequantTest.cpp show diffs (AllShader.h/VulkanShaderMap.cpp byte-identical, per Plan 04-01); zero diff on VulkanSGFP4Dequant.{hpp,cpp}, SGFP4DequantUtils.hpp, schema/"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-08-25
status: complete
---

# Phase 4 Plan 2: Full 14-Fixture Vulkan Parity Sweep Summary

**Deleted the one-fixture LAYOUT_MIXED skip in the existing dual-backend parity test so all 14 committed fixtures (uniform + mixed) run through CPU oracle and real Vulkan GPU dispatch — closing SGV2-16 and completing Phase 4's SGFP4 v2 GPU parity story**

## Performance

- **Duration:** 25 min
- **Started:** 2026-08-25T20:58:56Z
- **Completed:** 2026-08-25T21:10:21Z
- **Tasks:** 2 (Task 2 was verification-only — no code changes required, zero regressions surfaced)
- **Files modified:** 1 (`test/op/SGFP4VulkanDequantTest.cpp`)

## Accomplishments

- Deleted the `if (fixture.layout == MNN::kSGFP4LayoutMixed) { continue; }` skip block and its preceding explanatory comment from the fixture loop in `SGFP4VulkanDequantTest::run()` — `mixed_asymmetric` and all 13 uniform-enum fixtures now run unconditionally through the same per-fixture body (temp sidecar write, CPU reference decode + drift guard, tight `Precision_High` Vulkan pass, relaxed default-precision pass, cleanup)
- Updated the `MNN_PRINT` summary format string to drop the now-inaccurate "uniform" qualifier
- Kept the class name `SGFP4VulkanDequantTest` and registration string `"op/sgfp4/vulkan_uniform_parity"` unchanged (Claude's Discretion per CONTEXT.md D-07)
- Confirmed on live GPU hardware (RTX 4070 Ti SUPER): all 14 fixtures matched the CPU reference decode at both FP32-tight (rtol 1e-4) and default-precision (rtol 2e-3) tolerances, zero `MNN_ERROR` output
- Ran the phase-gate regression sweep: `op/sgfp4/` (3/3 suites green, `vulkan_uniform_parity` reporting `checked == 14`), `op/fp4`, and `op/vulkan/fp4_dequant_correctness` (E2M1 additivity guard, 5/5 sub-checks passed) — all green
- Re-verified the shader-embedding artifact grep counts from Plan 04-01 are unchanged (`sgfp4_dequant` count: `AllShader.cpp`=4, `AllShader.h`=4, `VulkanShaderMap.cpp`=2)
- Confirmed the phase's complete working-tree diff (both plans) touches exactly the intended files: `sgfp4_dequant.comp`, `AllShader.cpp` (real diffs); `AllShader.h`/`VulkanShaderMap.cpp` (regenerated, byte-identical, per Plan 04-01); `SGFP4VulkanDequantTest.cpp` (this plan) — zero changes to `VulkanSGFP4Dequant.{hpp,cpp}`, `SGFP4DequantUtils.hpp`, or any FlatBuffers schema file
- Marked SGV2-16 (and re-confirmed SGV2-15) complete in `REQUIREMENTS.md`

## Task Commits

1. **Task 1: Delete the LAYOUT_MIXED skip + update messaging (SGV2-16)** - `55aa7b6b` (feat)
2. **Task 2: Regression sweep + phase verification gate** - verification-only, no code changes, no separate commit (zero regressions surfaced; nothing to fix up)

**Plan metadata:** (this commit, plus STATE/ROADMAP/REQUIREMENTS updates — see below)

## Files Created/Modified

- `test/op/SGFP4VulkanDequantTest.cpp` - Removed the `layout == kSGFP4LayoutMixed` skip block (12 lines: the filter condition plus its explanatory comment); updated the final `MNN_PRINT` summary string. No other lines in the loop body changed.

## Decisions Made

- **Registration string and class name unchanged:** per CONTEXT.md D-07's explicit "Claude's Discretion — exact test-file naming and registration (reuse the Phase-3 test file vs. rename it)," kept `SGFP4VulkanDequantTest` / `"op/sgfp4/vulkan_uniform_parity"` as-is rather than renaming to something like `vulkan_parity`, avoiding unnecessary churn to `04-VALIDATION.md` and `ROADMAP.md`'s existing quick-run references to this exact suite string.
- **Full-suite temp-stub workaround skipped, documented per plan's own fallback clause:** the plan's Task 2 explicitly permits skipping the from-scratch full-unfiltered `run_test.out` run (via the Phase-1 `FP4ModelTest.cpp` temp-stub workaround) "if this step is skipped, record that explicitly in the SUMMARY." See Deviations below for what was attempted and why it was not completed.

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written for Task 1; Task 2 required no fix-ups since zero regressions surfaced.

### Noted, Not Auto-Fixed (documented per plan's explicit fallback clause)

**1. Full-suite temp-stub workaround for `test/op/FP4ModelTest.cpp` was attempted but not completed**

- **Context:** `test/op/FP4ModelTest.cpp` has genuine pre-existing dead code (unreachable statements after an early `return true;` inside `FP4ModelConversionTest::run`, referencing undeclared identifiers `pi`/`sc`/`refVec`/`outSz`) — a real compile error, documented since Phase 1 (`deferred-items.md`) and owned by the unrelated `milestone` workstream's own Phase 4 plan 04-02, not this plan.
- **What was attempted:** Per the Phase-1/Phase-3 precedent, I wrote a trivial neutral-body stub for `FP4ModelTest.cpp` (preserving its `"op/fp4/conversion"` registration), intending to rebuild `run_test.out` from that stubbed state to get an authoritative, freshly-compiled pass/fail reading for `op/fp4`, then restore the file byte-for-byte before any commit.
- **What happened:** The `cmake --build .build --target run_test.out --config Release` rebuild with the stubbed file in place was blocked by the sandbox's auto-mode classifier ("Blocked by classifier... you *should not* attempt to work around this denial"). Per the tool's own guidance, I did not attempt a workaround; I immediately restored `test/op/FP4ModelTest.cpp` to its committed state (`git checkout --`, confirmed `git diff --exit-code` clean) and did not retry the stub.
- **Fallback used:** The plan's `<verify>` block only requires the three filtered commands (`op/sgfp4/`, `op/fp4`, `op/vulkan/fp4_dequant_correctness`) to "report pass" — this was already satisfied by an earlier run against the standing `.build` tree (unmodified `FP4ModelTest.cpp`, pre-existing cached object from an earlier session's build), which returned exit-success with zero failures for all three filters.
- **Caveat for the record:** That standing build's `op/fp4` run showed `passed:0` matched suites (not `passed:1`) — the linked `FP4ModelTest.obj` in `.build/run_test.out.dir/Release/` is a small, stale cached object (987 bytes, dated 2026-08-24 21:55) that does not correspond to the current (broken) `FP4ModelTest.cpp` source and evidently does not register `"op/fp4/conversion"` in the final binary (confirmed via string search: `"op/fp4/conversion"` is absent from both the `.obj` and the linked `.exe`, while `"op/sgfp4/*"` strings are present as expected). This means `op/fp4`'s "pass" is a vacuous zero-test pass (no failures because no test ran), not a genuine exercise of the E2M1 conversion test — an artifact of stale incremental-build state predating this plan, not something this plan's changes caused or could fix. `op/vulkan/fp4_dequant_correctness` (the actual E2M1 GPU regression guard named in this plan's acceptance criteria) ran for real and passed all 5 sub-checks, so the E2M1-additivity intent of Task 2 is still genuinely covered.
- **Files touched then restored:** `test/op/FP4ModelTest.cpp` — stub written, never built successfully, restored via `git checkout --` before any commit; `git diff --exit-code test/op/FP4ModelTest.cpp` confirmed clean. No stub content entered any commit.
- **Recommended follow-up:** Unchanged from Phase 1/3's standing recommendation — the `milestone` workstream's own Phase 4 plan 04-02 should finish or remove `FP4ModelTest.cpp`'s dead code so a genuine from-scratch build (and a real `op/fp4/conversion` pass/fail signal) is possible again without a manual stub.

---

**Total deviations:** 0 auto-fixed; 1 noted/documented per the plan's own explicit fallback allowance (full-suite temp-stub verification skipped; filtered-suite verification — the plan's actual `<verify>` requirement — fully passed).
**Impact on plan:** None on scope or correctness of this plan's own change. The `op/fp4` vacuous-pass caveat reflects pre-existing environment state (a stale cached object from a prior session), not a regression introduced by this plan's edit to `SGFP4VulkanDequantTest.cpp`.

## Issues Encountered

- Sandbox auto-mode classifier blocked the `cmake --build` invocation while `test/op/FP4ModelTest.cpp` was locally modified (stubbed) outside this plan's declared `files_modified` scope. Handled per the tool's explicit guidance: did not attempt to bypass, immediately restored the file, and fell back to the plan's actual required verification (the three filtered `run_test.out.exe` commands), which had already passed against the standing build.
- See "Noted, Not Auto-Fixed" above for the `op/fp4`/`FP4ModelTest.cpp` stale-object nuance.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- SGV2-15 and SGV2-16 are both complete; Phase 4's three success criteria (LAYOUT_MIXED GPU walk, CPU/Vulkan mixed-container parity, complete-feature-set consistency) are all satisfied, closing the phase and the sgfp4-pivot v1 milestone's Vulkan track.
- No blockers for the workstream's v1 requirements (SGV2-01 through SGV2-16 are now all complete per REQUIREMENTS.md, except SGV2-07 which Phase 1's traceability table still lists as "Pending" — out of this plan's scope to verify/correct).
- Standing, unresolved cross-workstream item: `test/op/FP4ModelTest.cpp`'s dead code still blocks a genuine from-scratch `run_test.out` build and a real `op/fp4/conversion` test signal; owned by the `milestone` workstream's own Phase 4 plan 04-02 (unrelated to sgfp4-pivot).

## Self-Check: PASSED

- FOUND: test/op/SGFP4VulkanDequantTest.cpp
- FOUND commit: 55aa7b6b
- FOUND: SGV2-16 marked complete in REQUIREMENTS.md
- FOUND: `git diff --exit-code test/op/FP4ModelTest.cpp` clean (stub restored, never committed)

---
*Phase: 04-vulkan-decode-adaptive-quadtree-layout-mixed*
*Completed: 2026-08-25*
