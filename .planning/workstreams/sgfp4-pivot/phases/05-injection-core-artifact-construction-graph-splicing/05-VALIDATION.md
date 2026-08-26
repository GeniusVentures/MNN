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
| TBD-by-planner | TBD | TBD | SGINJ-01 | TBD | v1 container rejected via magic/version byte check | unit | TBD | no | pending |
| TBD-by-planner | TBD | TBD | SGINJ-02 | TBD | Constructed op matches pinned OpT recipe (type/param/externalPath) | unit+E2E | TBD | no | pending |
| TBD-by-planner | TBD | TBD | SGINJ-03 | TBD | Merged sidecar byte ranges non-overlapping & match {offset,size} | unit | TBD | no | pending |
| TBD-by-planner | TBD | TBD | SGINJ-04 | TBD | Artifact reloads via Module::load + setExternalFile, decodes within oracle tolerance | E2E | TBD | no | pending |

> The planner fills Task ID / Plan / Wave / Automated Command / File Exists columns when
> concrete task breakdowns exist. Threat Ref filled from PLAN.md threat models (security
> capability inactive this run — leave TBD or n/a).

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
