---
phase: 7
slug: multi-tensor-hardening-structured-data-coverage
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-27
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | MNN native test framework (`MNNTestSuite`, `run_test.out`) — in-process tool invocation via shared core header |
| **Config file** | `test/CMakeLists.txt` (existing glob — new files picked up on reconfigure) + `tools/fp4/CMakeLists.txt` (existing target) |
| **Quick run command** | `cd .build; .\run_test.out op/sgfp4/multi_tensor` (PowerShell; same for `op/sgfp4/malformed_inputs`) |
| **Full suite command** | `cd .build; .\run_test.out op/sgfp4` (family; full binary still blocked by unrelated `FP4ModelTest.cpp` — filtered-workaround per STATE.md) + `cmake --build . --target sgfp4_inject.out` |
| **Estimated runtime** | ~2-3 minutes (rebuild + family run) |

---

## Sampling Rate

- **After every task commit:** Run the specific new suite (`op/sgfp4/multi_tensor` or `op/sgfp4/malformed_inputs`) — target < 60s
- **After every plan wave:** Run `.\run_test.out op/sgfp4` family + `cmake --build . --target sgfp4_inject.out` + one full CLI E2E asserting artifact files appear on success
- **Before `/gsd-verify-work`:** All `op/sgfp4/*` green, standalone build OK, no regressions in `classic_api*` after core-header edit, README reviewed
- **Max feedback latency:** ~2-3 minutes

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | SGINJ-08 | Tampering | malformed container → exit ≠ 0 + diagnostic, no crash | unit | `.\run_test.out op/sgfp4/malformed_inputs` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | D-11 (SGINJ-07/08) | DoS | failed run leaves NO partial out.mnn/out.mnn.weight | unit | (same suite, probe 8 + absence assert) | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 1 | SGINJ-07 | — | 2 containers → 1 artifact, disjoint aligned ranges, byte-identical ranges, classic run parity | integration | `.\run_test.out op/sgfp4/multi_tensor` | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 1 | SGINJ-08 | — | structured container decoded through classic path, parity vs oracle baseline | integration | (same suite) | ❌ W0 (fixture authoring) | ⬜ pending |
| 07-02-03 | 02 | 1 | SGINJ-08 (D-13) | — | README documents dims convention, niche-dir/manifest contract, CLI usage, sidecar layout | source assertion | `git show` review + verify-work | ❌ (new) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Critical Failure Modes

1. **Atomicity regression/bug** — failed run leaves partial artifact (downstream = SGProcessingManager nullptr crash). Highest severity.
2. **Sidecar offset collision** between the two containers (silent weight corruption if unasserted).
3. **Structured fixture not actually MIXED** — quadtree coverage claimed but all-uniform bytes shipped.
4. **Tool crash on garbage payload** (D-10) — must be clean failure/structural success.
5. **Pairing ambiguity mishandled** — 2+ matches must stay a hard fail (D-08).
6. **Standalone tool build break** from core-header edits (must keep `sgfp4_inject.out` + Phase 5/6 suites green).

---

## Wave 0 Requirements

- [ ] `test/op/SGFP4MultiTensorTest.cpp` — both suites (`op/sgfp4/multi_tensor`, `op/sgfp4/malformed_inputs`) for SGINJ-07/SGINJ-08
- [ ] `test/op/SGFP4StructuredFixtures.h` — generated structured-mixed C-array fixture (blocks structured suite; authoring quick-task using gnus-poc `FP4Exporter` with `adaptive=True`, MIXED self-asserted)
- [ ] `tools/fp4/README.md` — D-13 four content areas
- [ ] No framework install/config needed — existing infrastructure covers all phase requirements.

*Existing `SGFP4ClassicAPITest.cpp` template + `.build` MSVC tree carry the rest.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Structured fixture authoring (gnus-poc export, MIXED assert) | SGINJ-08 | Cross-repo Python env; one-time authoring-time step (D-03) | Run `FP4Exporter.export_weights(..., adaptive=True)` on asymmetric-quadrant weights; assert `stats["layout_distribution"][4] > 0`; convert to C-array header |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
