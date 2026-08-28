---
phase: 06-classic-api-load-run-validation
plan: 06-02
subsystem: testing
tags: [sgfp4, classic-api, interpreter, session, mnn-test-suite, d-12]

requires:
  - phase: 06-classic-api-load-run-validation
    provides: sgfp4_inject_core.hpp with in-process sgfp4_inject::run (06-01)
  - phase: 05-injection-core-artifact-construction-graph-splicing
    provides: injected-artifact format (OpType_SGFP4Dequant + externalPath sidecar)
provides:
  - op/sgfp4/classic_api — proves injected artifacts load/run via classic Interpreter/Session API with named I/O and FP32-baseline parity (closes the workstream core-value claim)
  - op/sgfp4/classic_api_missing_sidecar — documents graceful missing-sidecar failure (non-zero runSession, no crash) for the downstream SGProcessingManager team
affects: [07-*, SGProcessingManager integration (separate workstream)]

tech-stack:
  added: []
  patterns:
    - "In-test container generation: degenerate all-UNIFORM_64 v2 framing written from kSGFP4* header constants only (no re-derived magic numbers), validated by sgfp4_is_v2_container + dequant_sgfp4_container_cpu round-trip"
    - "In-process tool invocation in tests: argv-array call into sgfp4_inject::run — the Phase 5 CLI exercised without a subprocess"

key-files:
  created:
    - test/op/SGFP4ClassicAPITest.cpp
  modified: []

key-decisions:
  - "Named I/O survives injection verbatim: getSessionInputAll/getSessionOutputAll return exactly 'input'/'output' — research A2/Open Q2 resolved, no suffix appended (asserts held unmodified)"
  - "Missing-sidecar primary observable confirmed: load/create succeed, runSession returns non-zero (Session::run refuses while mNeedResize after CPUSGFP4Dequant::onResize NOT_SUPPORT) — the RESIZE_STATUS fallback branch was not needed"
  - "Absolute temp paths everywhere (cwd-anchored tempPath) so the op's literal externalPath and the tool's niche-dir reads never depend on the runner's cwd (Pitfall 3)"
  - "No test/CMakeLists.txt change needed: 'tools' is already on run_test.out's include path, so #include \"fp4/sgfp4_inject_core.hpp\" resolves; Open Q4 closed with no MSVC /W3 warning fix required"

patterns-established:
  - "Classic-API parity harness: build baseline model with weight = oracle decode, then one runClassicSession helper drives both baseline and injected artifacts through identical Interpreter flows"

requirements-completed: [SGINJ-05, SGINJ-06]

coverage:
  - id: D1
    description: "op/sgfp4/classic_api — self-contained happy path: in-test container/base model/niche dir, in-process injection, classic load/run NO_ERROR, named 'input'/'output' I/O, FP32 parity rtol 1e-4, no setExternalFile"
    requirement: SGINJ-05
    verification:
      - kind: unit
        ref: "run_test.out op/sgfp4/classic_api → PASS (exit 0)"
        status: pass
  - id: D2
    description: "op/sgfp4/classic_api_missing_sidecar — sidecar deleted: createFromFile/createSession still succeed, runSession returns non-zero, no crash"
    requirement: SGINJ-06
    verification:
      - kind: unit
        ref: "run_test.out op/sgfp4/classic_api_missing_sidecar → PASS (exit 0)"
        status: pass
  - id: D3
    description: "No regression to pre-existing SGFP4 suites and sgfp4_inject.out target still builds (D-12 refactor intact)"
    requirement: SGINJ-05
    verification:
      - kind: unit
        ref: "run_test.out op/sgfp4/ → 7/7 passed (exit 0); cmake --build --target sgfp4_inject.out → OK"
        status: pass

human_judgment: false

duration: 40min
completed: 2026-08-27
status: complete
---

# Plan 06-02: Classic-API Load & Run Test Summary

**The injected SGFP4 artifact now provably loads and runs through the classic Interpreter/Session API — the exact downstream SGProcessingManager path — with named I/O intact, FP32 parity within 1e-4, and a documented graceful missing-sidecar failure.**

## Performance

- **Duration:** ~40 min
- **Tasks:** 2
- **Files created:** 1

## Accomplishments
- `test/op/SGFP4ClassicAPITest.cpp` (behind `MNN_SUPPORT_TRANSFORMER_FUSE`): two suites in one self-contained file — zero committed fixtures, zero env-var skips (D-01/D-10).
- **op/sgfp4/classic_api**: in-test 512×512 all-UNIFORM_64 container (framing from `kSGFP4*` constants, framing sizes 2064 B/record and 132,368 B total — matches the real demo container), oracle decode as weight, in-process injection via `sgfp4_inject::run(7, argv)` (D-12), then `createFromFile → createSession → resizeSession → runSession = NO_ERROR`; `getSessionInputAll`/`getSessionOutputAll` return exactly `input`/`output` (D-16); injected output matches the FP32 baseline within rtol 1e-4 (D-05..D-08); no `setExternalFile` anywhere (sidecar resolves via the op's literal `externalPath`, SGINJ-06/SC3).
- **op/sgfp4/classic_api_missing_sidecar** (D-13): after deleting the sidecar, load/session-create still succeed and `runSession` returns a non-zero ErrorCode (`Can't run session because not resized` path) with no crash.
- Full `op/sgfp4/` regression green (7/7) and `sgfp4_inject.out` still builds.

## Task Commits

1. **Task 1: SGFP4ClassicAPITest happy-path suite** — `6483a954` (test)
2. **Task 2: SGFP4ClassicAPIMissingSidecarTest probe** — `6483a954` (same file, single commit)

**Plan metadata:** this SUMMARY commit (docs)

## Files Created/Modified
- `test/op/SGFP4ClassicAPITest.cpp` — both suites + file-static fixture helpers (created)

## Decisions Made
- Reused the 05-01 `tempPath` pattern but anchored every temp path to the current working directory (cwd-anchored) so the absolute-path contract of `op->externalPath` holds regardless of runner cwd.
- Shared `buildInjectedArtifact`/`runClassicSession` helpers between the two suites — the missing-sidecar probe re-runs the full happy-path fixture then deletes the sidecar, exercising the exact artifact a downstream consumer would hold.
- Named-I/O asserts held unmodified (`input`/`output` survive injection verbatim) — flagged friction from ROADMAP success criterion 1 did not materialize; recorded via this summary.

## Deviations from Plan

### Auto-fixed Issues

**1. Missing namespace qualifications (MSVC compile errors)**
- **Found during:** Task 1 (first build)
- **Issue:** `Tensor`, `Interpreter`, `ScheduleConfig`, `ErrorCode`, `NO_ERROR` failed to resolve — `using namespace MNN::Express` alone does not pull in the parent `MNN` namespace on MSVC.
- **Fix:** Added `using namespace MNN;` alongside `using namespace MNN::Express;`.
- **Files modified:** `test/op/SGFP4ClassicAPITest.cpp`
- **Verification:** clean compile
- **Committed in:** `6483a954`

**2. constexpr call to inline helper rejected by MSVC (C2131)**
- **Found during:** Task 1 (second build)
- **Issue:** `MNN::sgfp4_align16` is `inline`, not `constexpr`; MSVC refused it as a constant-expression initializer for `kRecordRegionStart` (both direct use and inside `static_assert`).
- **Fix:** Computed the constant arithmetically and added a runtime guard assertion in `buildContainerUniform64` that `kRecordRegionStart == sgfp4_align16(offsetTableEnd)` — same guarantee, compiler-compatible.
- **Files modified:** `test/op/SGFP4ClassicAPITest.cpp`
- **Verification:** clean compile; test PASS validates the framing end-to-end (oracle round-trip + in-tool `decode==oracle`)
- **Committed in:** `6483a954`

**3. vcxproj glob regeneration cycle (build scaffolding, not a code fix)**
- **Found during:** Task 1 (build)
- **Issue:** the new test file was not in the generated `run_test.out.vcxproj` (CMake glob ran at configure time), and re-running `cmake .` re-added the known-broken `FP4ModelTest.cpp`.
- **Fix:** re-ran `cmake .` to pick up the new file, then re-applied the documented STATE.md workaround (filter `FP4ModelTest.cpp` out of the untracked generated `.vcxproj`). No tracked file touched; permanent fix remains owned by the `milestone` workstream (04-02).
- **Verification:** full build + all suites green

## Self-Check

PASSED — every acceptance token present (gated `#ifdef`, both `MNNTestSuiteRegister` strings exactly once, all six classic-API tokens, `setName("input")`/`setName("output")`, `sgfp4_inject::run(`, `dequant_sgfp4_container_cpu`, `checkVectorByRelativeError`, `sgfp4::sha256_hex`, `std::remove(` on the sidecar); zero functional `setExternalFile` (only two comment mentions — the API call does not appear); `op/sgfp4/classic_api` exit 0 PASS, `op/sgfp4/classic_api_missing_sidecar` exit 0 PASS, `op/sgfp4/` 7/7 exit 0, `sgfp4_inject.out` builds.
