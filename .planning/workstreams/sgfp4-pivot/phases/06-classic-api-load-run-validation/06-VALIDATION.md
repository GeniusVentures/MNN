---
phase: 6
slug: classic-api-load-run-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-27
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | MNN native test framework (`MNNTestSuite`, `run_test.out`); classic-API suites consumed via in-test injection (shared core header) |
| **Config file** | `test/CMakeLists.txt` (existing glob) + `tools/fp4/CMakeLists.txt` (existing `sgfp4_inject.out` target) |
| **Quick run command** | `./run_test.out op/sgfp4/classic_api` (filtered suite) |
| **Full suite command** | `./run_test.out op/sgfp4` (all SGFP4 suites; full `run_test.out` still blocked by pre-existing unrelated `FP4ModelTest.cpp` breakage — STATE.md) |
| **Estimated runtime** | ~60–120 seconds (filtered) |

---

## Sampling Rate

- **After every task commit:** Run `./run_test.out op/sgfp4/classic_api` (or `op/sgfp4` once later suites exist)
- **After every plan wave:** Run `./run_test.out op/sgfp4` + rebuild `sgfp4_inject.out` smoke (`--help` / demo-container E2E from Phase 5)
- **Before `/gsd-verify-work`:** Filtered-suite set green; named I/O assertions + FP32-baseline parity + missing-sidecar probe all green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 06-01-T1 (core refactor) | 06-01 | 1 | SGINJ-05 (build path) | T-06-01 | `sgfp4_inject_core.hpp` exists; `sgfp4_inject.cpp` main() thin; `sgfp4_inject.out` still builds and Phase 5 E2E unchanged | source assertion + build | `cd build && cmake .. -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_SGFP4_TOOLS=ON && make sgfp4_inject.out && ./tools/fp4/sgfp4_inject.out --model minimal_512.mnn --niche-dir "<demo fp4 dir>" --output out.mnn && test -f out.mnn && test -f out.mnn.weight` | no | pending |
| 06-02-T1 (classic load/run + named I/O) | 06-02 | 2 | SGINJ-05, SGINJ-06 | T-06-02 | `createFromFile→createSession→runSession` returns NO_ERROR; `getSessionInputAll`/`getSessionOutputAll` return base-model names (D-16); output matches FP32 baseline (decoded-container weight) within rtol 1e-4 (D-05..D-08); external path resolved via op itself, no `setExternalFile` | integration | `cd build && ./run_test.out op/sgfp4/classic_api` | no | pending |
| 06-02-T2 (missing-sidecar probe) | 06-02 | 2 | SGINJ-06 (failure mode) | T-06-03 | With `.weight` sidecar absent: no crash; failure observable via `runSession` != NO_ERROR or `getSessionInfo(RESIZE_STATUS)` (resizeSession swallows the ErrorCode — probe via runSession) | integration | `cd build && ./run_test.out op/sgfp4/classic_api_missing_sidecar` | no | pending |

> Incremental feedback (<30s) after first configure: rebuild `run_test.out` (filtered build) + run `./run_test.out op/sgfp4/classic_api`.
> Task IDs map to plan tasks; final numbering fixed by PLAN.md.

---

## Wave 0 (Validation Fallback)

- **Fallback trigger:** `run_test.out` full build blocked by the pre-existing unrelated
  `test/op/FP4ModelTest.cpp` breakage (documented in STATE.md).

- **Fallback procedure:** Build and run filtered suites only (`op/sgfp4/` and friends) —
  the same workaround validated in Phase 04 P02 and reused in Phase 5. The classic-API
  suites are self-contained (in-test injection + temp-dir niche artifacts), so they need
  no external fixtures.

---

## Coverage Signals

- **Green:** `op/sgfp4/classic_api` and `op/sgfp4/classic_api_missing_sidecar` pass; `sgfp4_inject.out` Phase 5 E2E still passes post-refactor.
- **Yellow:** Classic run passes but names mismatch after injection (D-16) or parity near tolerance — investigate tensor naming serialization first, then sidecar offsets.
- **Red:** Classic load/run crashes on missing sidecar, `runSession` error not propagated, or injected output diverges from FP32 baseline beyond rtol.
