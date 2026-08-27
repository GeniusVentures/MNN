---
phase: 5
slug: injection-core-artifact-construction-graph-splicing
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-26
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | MNN native test framework (`MNNTestSuite`, `run_test.out`) + plan-level tool E2E checks |
| **Config file** | `test/CMakeLists.txt` (existing); tool build gated by new CMake option per RESEARCH.md |
| **Quick run command** | `./run_test.out op/sgfp4` (filtered SGFP4 suites) |
| **Full suite command** | `./run_test.out` (note: pre-existing unrelated `FP4ModelTest.cpp` breakage — see STATE.md; use filtered suites when it blocks) |
| **Estimated runtime** | ~60–120 seconds (filtered) |

---

## Sampling Rate

- **After every task commit:** Run `./run_test.out op/sgfp4`
- **After every plan wave:** Run filtered suites covering changed area (tool E2E commands from PLAN.md `<verify>` blocks)
- **Before `/gsd-verify-work`:** Full filtered-suite set green; injected artifact reload + oracle-parity E2E green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-T1 (A1 spike) | 05-01 | 1 | SGINJ-02, SGINJ-03, SGINJ-04 | T-05-03, T-05-02 | Saved artifact has exactly 1 SGFP4Dequant op and 0 dead weight Consts; decode within 1e-4f | integration | `cd build && ./run_test.out op/sgfp4/inject` | no | pending |
| 05-01-T2 (version gate) | 05-01 | 1 | SGINJ-01 | T-05-01 | `sgfp4_is_v2_container` rejects bad-magic/bad-version/v1-layout/null/short; accepts known-good v2 | unit | `cd build && ./run_test.out op/sgfp4/inject_v1_reject && ./run_test.out op/sgfp4/inject` | no | pending |
| 05-02-T1 (SHA-256 + CMake) | 05-02 | 2 | SGINJ-02 (build path) | T-05-SC | Vendored single-header sha256 (no OpenSSL/registry installs); wiring grep-clean | source assertion | `grep -n "MNN_BUILD_SGFP4_TOOLS\|tools/fp4/CMakeLists.txt\|sgfp4_inject.out\|sha256_hex" CMakeLists.txt tools/fp4/CMakeLists.txt tools/fp4/sha256.hpp` | no | pending |
| 05-02-T2 (sgfp4_inject tool) | 05-02 | 2 | SGINJ-01, SGINJ-02, SGINJ-03, SGINJ-04 | T-05-01..T-05-06 | v1 rejected at byte level before decode; sha256 mismatch hard-errors; sidecar ranges monotonic/16-aligned; reload+oracle compare within 1e-4f | E2E | `cd build && cmake .. -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_SGFP4_TOOLS=ON && make sgfp4_inject.out && ./tools/fp4/sgfp4_inject.out --model minimal_512.mnn --niche-dir "<demo fp4 dir>" --output out.mnn && test -f out.mnn && test -f out.mnn.weight` (shell: MSYS2/MinGW bash) | no | pending |

> Incremental feedback (<30s) after first configure: `cd build && make sgfp4_inject.out` + the E2E line above.
> Task IDs map to plan tasks: 05-01 Task 1/Task 2, 05-02 Task 1/Task 2.

---

## Wave 0 (Validation Fallback)

The following fallback validates execution when the primary verification is unavailable:

- **Fallback trigger:** `run_test.out` full build blocked by the pre-existing unrelated
  `test/op/FP4ModelTest.cpp` breakage (documented in STATE.md).

- **Fallback procedure:** Build and run with filtered suites only
  (`op/sgfp4/`, `op/fp4`, `op/vulkan/fp4_dequant_correctness`) — the same workaround
  validated in Phase 04 P02 — plus the injection tool's own E2E exit-code check on the
  demo container `gnus-poc/models/specialists_mlx/demo/fp4/demo.sgfp4`.

---

## Coverage Signals

- **Green:** All filtered SGFP4 suites pass; injected artifact E2E passes on demo container.
- **Yellow:** Suites pass but oracle parity exceeds tolerance on ≥1 tensor (indicates
  decode mismatch — investigate offsets/sizes first).
- **Red:** Tool crashes, v1 container accepted silently, or artifact fails `Module::load`.

---

## Read-Only Enforcement

`wave_0_complete: false` until the first plan's Wave-0 spike (A1: `Variable::replace`
in-place semantics on a minimal input→MatMul→output graph) lands its minimal graph test.
