---
phase: 06-classic-api-load-run-validation
verified: 2026-08-27T00:00:00Z
status: passed
score: 3/3 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: none
  note: "Initial verification — no previous VERIFICATION.md existed"
human_verification: []
---

# Phase 06: Classic-API Load & Run Validation — Verification Report

**Phase Goal:** The injected artifact loads and runs through the classic Interpreter/Session API — `Interpreter::createFromFile/createSession → runSession` — the exact path `SGProcessingManager::MNN_Tensor::Process()` uses downstream; never previously verified end-to-end.
**Verified:** 2026-08-27
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `Interpreter::createFromFile` → `createSession` → `runSession` succeeds on an injected artifact, with correct session input/output tensor identification | ✓ VERIFIED | `op/sgfp4/classic_api` PASSED (see Spot-Check 1). Source: `test/op/SGFP4ClassicAPITest.cpp:355-375` (`createFromFile` non-null, `createSession` non-null, `getSessionInputAll`/`getSessionOutputAll` assert keys `"input"`/`"output"`), `:404` (`runSession` asserted `NO_ERROR`). Runtime output: `SGFP4ClassicAPITest: classic load/run + named I/O + FP32 parity PASSED`, `runSession` cost 27.3 ms |
| 2 | End-to-end inference with at least one injected weight tensor matches an FP32/reference baseline within defined tolerance on CPU, via the existing decode Execution | ✓ VERIFIED | `test/op/SGFP4ClassicAPITest.cpp:415-432`: baseline session over `fx.basePath` (weight = `dequant_sgfp4_container_cpu` oracle), identical LCG input feeds both sessions (D-08), `checkVectorByRelativeError<float>(got, baseline, 512, 1e-4f)` at `:427`. Suite PASSED at runtime — parity holds within rtol 1e-4 on `MNN_FORWARD_CPU` (`cfg.type = MNN_FORWARD_CPU`, `:364`) |
| 3 | External-sidecar resolution works under the classic API path (external path via the op itself, not session-level `setExternalFile`) | ✓ VERIFIED | Grep of `SGFP4ClassicAPITest.cpp`: only 2 `setExternalFile` mentions, both in comments (`:14`, `:397` — zero API invocations). Sidecar resolves via the op's literal `externalPath`: happy-path suite passes with no external-file API call. Negative probe `op/sgfp4/classic_api_missing_sidecar` PASSED individually (exit 0): after `std::remove(fx.sidecarPath)` (`:467`), load/create succeed but `runSession` returns non-zero (`Can't run session because not resized`) with no crash — graceful failure documented |

**Score:** 3/3 truths verified

### Plan 06-01 Must-Have Truths (refactor enabler)

| Truth | Status | Evidence |
|-------|--------|----------|
| `sgfp4_inject.out` still builds, CLI contract unchanged | ✓ VERIFIED | `W:\gnus\GeniusNetwork\thirdparty\MNN\.build\Release\sgfp4_inject.out.exe` exists (Test-Path: True); `tools/fp4/sgfp4_inject.cpp` is a 27-line shim (`#include "sgfp4_inject_core.hpp"` + `main` returning `sgfp4_inject::run(argc, argv)`) preserving the `--model/--niche-dir/--output → out.mnn + out.mnn.weight` contract (header comment `:19-21`) |
| Injection core callable in-process as `sgfp4_inject::run(argc, argv)` (no subprocess, D-12) | ✓ VERIFIED | `tools/fp4/sgfp4_inject_core.hpp:50` `namespace sgfp4_inject {`, `:275` `inline int run(int argc, const char* argv[])`; test calls it in-process at `SGFP4ClassicAPITest.cpp:327` (`sgfp4_inject::run(7, argv)`) — proven at runtime by `sgfp4_inject: node 'weight' {512,512} offset=0 size=132368 verified (decode==oracle)` inside the test binary's own process |
| Phase 5 E2E decode==oracle still verifies after refactor | ✓ VERIFIED | The in-tool verification printed during both suite runs (line above); full `op/sgfp4/` family green post-refactor (7/7) |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `test/op/SGFP4ClassicAPITest.cpp` | Two suites registered `op/sgfp4/classic_api` + `op/sgfp4/classic_api_missing_sidecar`; classic-API flow; in-process injection; oracle + parity check; no functional `setExternalFile` | ✓ VERIFIED | Exists (508+ lines, substantive). `MNNTestSuiteRegister` at `:442` and `:508`. Uses `Interpreter::createFromFile` (`:357`), `createSession` (`:363`), `resizeSession` (`:401`), `runSession` (`:404`), `getSessionInputAll`/`getSessionOutputAll` (`:373-374`), `sgfp4_inject::run` (`:327`), `dequant_sgfp4_container_cpu` (`:300`), `sgfp4_is_v2_container` (`:293`), `checkVectorByRelativeError` (`:427`). Gated `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE` (`:23`). Zero functional `setExternalFile` (comments only) |
| `tools/fp4/sgfp4_inject_core.hpp` | `namespace sgfp4_inject` + `inline int run` | ✓ VERIFIED | `:50` and `:275`; namespace closes `:470`; helpers all present |
| `tools/fp4/sgfp4_inject.cpp` | Thin main() shim | ✓ VERIFIED | 27 lines: include + `int main` → `sgfp4_inject::run(argc, argv)` |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| `SGFP4ClassicAPITest.cpp` | `include/MNN/Interpreter.hpp` | `createFromFile/createSession/runSession/getSessionInputAll/getSessionOutputAll` | ✓ WIRED — all five tokens present and used in the happy path |
| `SGFP4ClassicAPITest.cpp` | `tools/fp4/sgfp4_inject_core.hpp` | `#include "fp4/sgfp4_inject_core.hpp"` (`:44`) + `sgfp4_inject::run` (`:327`) | ✓ WIRED — in-process call verified at runtime |
| `SGFP4ClassicAPITest.cpp` | `include/MNN/SGFP4DequantUtils.hpp` | `dequant_sgfp4_container_cpu` + `sgfp4_is_v2_container` | ✓ WIRED — `:293`, `:300`; oracle round-trip ran in-suite |
| `sgfp4_inject.cpp` | `sgfp4_inject_core.hpp` | `#include` + `run` delegation | ✓ WIRED — binary builds and runs |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full SGFP4 family regression | `.\Release\run_test.out.exe op/sgfp4/` (from `.build`) | `TEST_CASE_AMOUNT_UNIT: {"blocked":0,"failed":0,"passed":7,"skipped":0}`, `EXIT=0`; includes `SGFP4ClassicAPITest: classic load/run + named I/O + FP32 parity PASSED` and `SGFP4ClassicAPIMissingSidecarTest: missing sidecar fails gracefully PASSED` | ✓ PASS |
| Missing-sidecar negative probe (individual) | `.\Release\run_test.out.exe op/sgfp4/classic_api_missing_sidecar` | `passed:1, failed:0`, `EXIT=0`; observable failure: `Can't run session because not resized` (runSession non-zero, no crash) after sidecar deletion | ✓ PASS |
| Injector binary exists | `Test-Path .\Release\sgfp4_inject.out.exe` | `True` | ✓ PASS |
| Phase commits present | `git log --oneline -8` | `4cc1851b` / `69f22e8d` (06-01), `6483a954` (06-02), summaries `f863a83f` / `df70ea65`, ROADMAP marks `1fcec178` / `b824f4a0` | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared for this phase; Step 7c satisfied by in-suite runtime verification above (the run_test.out suites ARE the behavioral probes, executed directly by the verifier).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SGINJ-05 | 06-01, 06-02 (both declare `[SGINJ-05, SGINJ-06]`; both summaries' `requirements-completed: [SGINJ-05, SGINJ-06]`) | Injected artifact loads/runs via classic API (createFromFile/createSession/runSession, named-I/O friction expected) | ✓ SATISFIED | `op/sgfp4/classic_api` PASSED; named I/O survived injection verbatim (`input`/`output` asserts held unmodified — the anticipated friction did not materialize) |
| SGINJ-06 | 06-01, 06-02 | E2E inference matches FP32 baseline within tolerance on CPU; sidecar resolution via op's externalPath, not session-level setExternalFile | ✓ SATISFIED | Parity rtol 1e-4 PASSED on CPU; no `setExternalFile` call; missing-sidecar graceful-failure probe PASSED |

**Orphaned requirements:** None — REQUIREMENTS.md maps exactly SGINJ-05/SGINJ-06 to Phase 6, both claimed by both plans.

ℹ️ **Note (non-blocking):** the REQUIREMENTS.md traceability table rows for SGINJ-05/SGINJ-06 still read `Pending` while ROADMAP.md marks 06 complete — a planning-doc sync lag, not a code gap. Recommend updating to `Complete` at workstream close.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | Zero hits for TBD/FIXME/XXX/HACK/placeholder across the three phase files | — | — |

Deviations found in 06-02 SUMMARY were compile-fix class only (`using namespace MNN;` addition; constexpr→runtime-guard constant) — no scope or behavior impact; both validated by the runtime pass.

### Human Verification Required

None. All success criteria are machine-verified end-to-end (real Interpreter/Session execution, real injected artifact, real parity comparison, real negative probe) — no visual/UX/external-service surface exists for this phase.

### Gaps Summary

No gaps. All three roadmap success criteria verified against source code and confirmed by direct execution of the test binaries by the verifier (not SUMMARY claims): classic-API load/run succeeds with named I/O, injected inference matches the FP32 baseline within rtol 1e-4 on CPU, and sidecar resolution flows through the op's own externalPath with a documented graceful missing-sidecar failure. Both phase requirements (SGINJ-05, SGINJ-06) are satisfied and traceable through plan frontmatter and summary completion fields.

**Final verdict: PASSED** — phase goal achieved; workstream core-value claim (classic path never previously verified) is now closed with executable evidence.

---

_Verified: 2026-08-27_
_Verifier: the agent (gsd-verifier)_
