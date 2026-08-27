---
phase: 05-injection-core
plan: 01
subsystem: sgfp4-injection
tags: [sgfp4, express, graph-surgery, version-gate, test]
key-files:
  - test/op/SGFP4InjectTest.cpp
  - include/MNN/SGFP4DequantUtils.hpp
metrics:
  tests_passed: 5
  tests_failed: 0
  suites_new: ["op/sgfp4/inject", "op/sgfp4/inject_v1_reject"]
---

# Plan 05-01 Summary: SGFP4 Inject Graph-Surgery Spike + Version Gate

## Objective

De-risk the Express VARP-level graph-surgery recipe (research assumption A1) and
prove it at the runtime level before the standalone tool exists; add the
byte-level v2 version-gate primitive `sgfp4_is_v2_container` (SGINJ-01).

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 + 2 | 67fc8947 | `test/op/SGFP4InjectTest.cpp` (SGFP4InjectTest + SGFP4InjectV1RejectTest) and `sgfp4_is_v2_container` in `include/MNN/SGFP4DequantUtils.hpp` |

(Tasks 1 and 2 share the same test file and header; single atomic commit.)

## What Was Done

- **A1 spike (`op/sgfp4/inject`)**: builds a minimal 2-op MatMul Express graph
  (`input {1,dimI} × weight Const {dimO,dimI}`, `mode0_uniform64` fixture),
  saves direct-to-file, reloads via `Variable::loadMap`, splices a 0-input
  `OpType_SGFP4Dequant` node in place of `"weight"` via `Variable::replace`
  (externalPath set literally on the OpT; name `weight_sgfp4` per D-08),
  recomputes outputs post-rewiring, saves via the direct-to-file
  `Variable::save(vars, fileName)` overload, reloads through
  `Module::load` + `rtmgr->setExternalFile(sidecar)` called BEFORE load, and
  runs the 1-input module with a fresh input VARP.
- **Oracle comparison (SGINJ-04)**: spliced-artifact MatMul output vs
  `_MatMul(_Const(input), _Const(oracle))` where oracle =
  `dequant_sgfp4_container_cpu` decode of the same container bytes —
  `checkVectorByRelativeError<float>` at 1e-4f. PASSED.
- **Graph-structure assertion (A1/D-07)**: reloads the saved artifact via
  `Variable::loadMap`, walks every reachable expr — asserts exactly 1
  `OpType_SGFP4Dequant` expr and 0 `VARP::CONSTANT` exprs (original weight
  Const dead-dropped). PASSED.
- **Version gate (SGINJ-01)**: header-only inline
  `MNN::sgfp4_is_v2_container(data, size)` probing the container's own bytes
  (`sgfp4_read_u32_le(data) == kSGFP4Magic && data[kSGFP4VersionByteOffset] ==
  kSGFP4Version`, plus null/min-size guards) — no manifest
  `fp4_binary.format` consultation. `SGFP4InjectV1RejectTest`
  (`op/sgfp4/inject_v1_reject`) asserts: known-good v2 accepted; bad-magic
  (`[0] ^= 0xFF`), bad-version (0xFF at version byte), v1-layout magic-less
  32-byte buffer, nullptr, and 15-byte truncated buffer all rejected.

## Verification Results

- `run_test.out op/sgfp4/inject` — PASS (18.1 ms)
- `run_test.out op/sgfp4/inject_v1_reject` — PASS (0.05 ms)
- `run_test.out op/sgfp4/` — **all 5 suites passed, 0 failed**:
  `uniform_decode`, `mixed_decode`, `inject` (new), `inject_v1_reject` (new),
  `vulkan_uniform_parity` — no regression to pre-existing suites.

## Assumption A1 — Confirmed

`Variable::replace(dst, src)` performs in-place consumer rewiring, and
`Variable::save(outputs)` naturally drops the now-unreachable original weight
Const (verified structurally, not just numerically). The exact recipe the tool
(Plan 05-02) needs is proven and committed.

## Deviations

- **Build unblock (environment, not code)**: the pre-existing, out-of-scope
  `test/op/FP4ModelTest.cpp` dead-code blocker (STATE.md known issue, owned by
  the `milestone` workstream) still breaks a full `run_test.out` build under
  MSVC. Instead of the Phase 04 P02 temp-stub workaround (which was previously
  sandbox-blocked), the file was excluded from the **untracked, generated**
  `.build/run_test.out.vcxproj` only (no tracked file touched;
  `git status` clean apart from expected paths). This is a strictly local,
  repeatable unblock: re-run the one-line filter after every `cmake` configure.
- Tests run under Windows PowerShell / MSVC `.build\Release\run_test.out.exe`
  (the plan's `<automated>` shell notes assumed MSYS2 bash; exit codes and
  PASS lines are equivalent).

## Self-Check

PASSED — all acceptance criteria verified (registration strings, replace/save
call forms, literal externalPath + kSGFP4Magic, graph-structure assertion
present, both suites exit 0 with PASS lines, full `op/sgfp4/` family green).
