---
phase: 8
slug: schema-sidecar-wiring
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-28
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `08-RESEARCH.md` §"Validation Architecture" (HIGH confidence).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | MNN's custom runner (`run_test.out`, `MNNTestSuite`/`MNNTestCase` registration via `MNNTestSuite::add`) |
| **Config file** | none — `test/CMakeLists.txt` auto-globs `test/**/*.cpp` |
| **Quick run command** | `run_test.out op/sgfp4/<suite>` (filter = `test->name.find(prefix) == 0`, `test/main.cpp` + `MNNTestSuite.cpp:43`) |
| **Full suite command** | `run_test.out op/sgfp4/` (all SGFP4 suites; full unfiltered `run_test.out` is pre-existing-blocked by dead `test/op/FP4ModelTest.cpp` — out of scope, tracked in STATE.md) |
| **Estimated runtime** | ~seconds per filtered suite; `op/sgfp4/` family well under a minute on CPU (Vulkan suites pass-skip with no device) |

**Converter-side tests:** `RemoveAndStoreParam`/`saveExternalData` live in `MNNConvertDeps` (built only under `MNN_BUILD_CONVERTER=ON`, already ON in the workspace build); converter round-trip tests link a converter-side test target, not `run_test.out` (RESEARCH Open Question Q1 → planner decides exact target wiring).

---

## Sampling Rate

- **After every task commit:** Run `run_test.out op/sgfp4/<touched-suite>` (fast filtered)
- **After every plan wave:** Run `run_test.out op/sgfp4/` (all SGFP4 CPU suites) + converter round-trip target (D-09) + Vulkan parity suite (skips gracefully if no device)
- **Before `/gsd-verify-work`:** Full `op/sgfp4/` family + converter round-trip + an existing v2.0 injected artifact still loads via classic API (D-04 regression guard)
- **Max feedback latency:** < 60s (filtered suite rebuild + run)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD (schema) | 01 | 1 | SGV2-22 | — | N/A (schema edit; backward compat by FlatBuffers append semantics) | compile + regression | `run_test.out op/sgfp4/dequant` | ✅ existing suites | ⬜ pending |
| TBD (CPU dispatch) | 01 | 1 | SGV2-22 | T-08 buffer-tamper | `sgfp4_is_v2_container` entry gate + bounds-checked decode; no partial output on malformed buffer | unit | `run_test.out op/sgfp4/dequant` | ❌ W0 (buffer-mode variants) | ⬜ pending |
| TBD (Vulkan dispatch) | 02 | 1 | SGV2-22 | T-08 buffer-tamper | mirror of CPU host pre-validation | unit | `run_test.out op/sgfp4/vulkan_<parity>` | ❌ W0 | ⬜ pending |
| TBD (RemoveParams) | 03 | 2 | SGV2-23 | T-08 overlap | aligned monotonic non-overlapping sidecar regions; `external == {offset, true-size}` | converter round-trip | converter-side test target | ❌ W0 | ⬜ pending |
| TBD (parity D-08) | 04 | 2 | SGV2-22 | — | buffer-mode == sidecar-mode == oracle | unit/parity | `run_test.out op/sgfp4/<new parity suite>` | ❌ W0 | ⬜ pending |
| TBD (test util D-10) | 05 | 1 | — | — | N/A (test-infra dedup; keeps region-relative offset convention correct) | unit (compile + existing suites) | `run_test.out op/sgfp4/classic_api` / `multi_tensor` / `inject_*` | ✅ retrofit existing | ⬜ pending |
| TBD (docs D-11/D-12) | 06 | 3 | — | — | N/A (comments/docs) | n/a | — | ❌ comment add | ⬜ pending |

*Task IDs finalized when PLAN.md files exist — update this map at execution time.*

---

## Wave 0 Requirements

- [ ] `test/op/SGFP4TestUtil.hpp` — shared helpers (tempPath, cwdPath, makeDir/removeDir, fileExists, writeU32Le, writeBytes/readBytes, generalized region-relative container builders, niche-dir writer) — extracted from the three duplicated test files and retrofitted (D-10)
- [ ] Converter round-trip test target (D-09) — placement + CMake wiring linking `MNNConvertDeps` (Q1 decision lands in PLAN.md)
- [ ] Buffer-mode parity test files/suites (D-08) — CPU + Vulkan (`op/sgfp4/` registration)
- [ ] Optional: `loadExternalParam` SGFP4 read-back case if planner decides converter-reload symmetry is in-scope (Q3)

*Wave 0 is produced as part of Phase 8 execution itself (test-first per plan waves) — these are the artifacts the plans must schedule before the verifications that depend on them.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Existing v2.0 injected artifact still loads via classic API | D-04 regression | Requires a real previously-injected artifact + classic API run outside the suite filters | Load a shipped v2.0 artifact (sidecar mode, no `buffer` field present) via `Interpreter::createFromFile` → `createSession` → `runSession`; confirm unchanged behavior |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
