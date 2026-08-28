---
phase: 06-classic-api-load-run-validation
plan: 06-01
subsystem: testing
tags: [sgfp4, injection-tool, refactor, header-only, d-12]

requires:
  - phase: 05-injection-core-artifact-construction-graph-splicing
    provides: sgfp4_inject.cpp standalone tool (05-02), SGFP4InjectTest graph-surgery recipe (05-01)
provides:
  - Header-only injection core tools/fp4/sgfp4_inject_core.hpp with sgfp4_inject::run(argc, argv) callable in-process (D-12 enabler for 06-02)
  - Thin sgfp4_inject.cpp main() shim with unchanged CLI contract and behavior
affects: [06-classic-api-load-run-validation, 07-* coverage/tooling phases]

tech-stack:
  added: []
  patterns:
    - "Shared core header pattern: tool logic in namespace sgfp4_inject with all free functions inline, so a tool main() and a run_test.out suite share one implementation with no subprocess and no re-implementation (ODR-safe across TUs, T-06-02)"

key-files:
  created:
    - tools/fp4/sgfp4_inject_core.hpp
  modified:
    - tools/fp4/sgfp4_inject.cpp

key-decisions:
  - "Mechanical zero-behavior-change move: every function body moved verbatim from the .cpp anonymous namespace into namespace sgfp4_inject, marked inline; the CLI worker renamed to run with byte-identical CLI-parse and exit semantics"
  - "NOMINMAX guard stays self-contained inside the header before <windows.h> (Pitfall 6), so any includer is safe without pre-defining it"
  - "No CMake change: tools/fp4/CMakeLists.txt already globs *.hpp at configure time and the header is inert in the executable source list"

patterns-established:
  - "Tool-core-as-header: future in-process consumers (06-02 classic-API test) include the header as `\"fp4/sgfp4_inject_core.hpp\"` (resolved via the project's global `tools/` include dir, the same mechanism the mnncli tests use) and call sgfp4_inject::run with an argv array instead of spawning the .out"

requirements-completed: [SGINJ-05, SGINJ-06]

coverage:
  - id: D1
    description: "Header-only injection core sgfp4_inject_core.hpp (namespace sgfp4_inject, inline helpers, run(argc,argv)); sgfp4_inject.cpp reduced to a main() shim; CLI contract unchanged"
    requirement: SGINJ-05
    verification:
      - kind: unit
        ref: "Select-String token scan: namespace sgfp4_inject / inline int run / all helper names / NOMINMAX / sgfp4_is_v2_container / dequant_sgfp4_container_cpu / sgfp4::sha256_hex / Variable::replace / Variable::save / op->externalPath present; int main( and injectMain absent (0 hits)"
        status: pass
      - kind: integration
        ref: "cmake --build . --target sgfp4_inject.out --config Release (MSVC, .build) — builds clean"
        status: pass
  - id: D2
    description: "Phase 5 end-to-end regression: demo-container injection still produces out.mnn + out.mnn.weight with in-tool decode==oracle verification after the refactor"
    requirement: SGINJ-06
    verification:
      - kind: e2e
        ref: "sgfp4_inject.out --model minimal_512.mnn --niche-dir <gnus-poc demo fp4 dir> --output out.mnn → exit 0, both artifacts exist, 'node ... verified (decode==oracle)' printed"
        status: pass

human_judgment: false

duration: 30min
completed: 2026-08-27
status: complete
---

# Plan 06-01: Shared Injection-Core Refactor Summary

**The Phase 5 injection tool's entire core now lives in a header-only `namespace sgfp4_inject` with an in-process `run(argc, argv)` entry point — a zero-behavior-change move that lets the Phase 6 classic-API test drive real injection without a subprocess.**

## Performance

- **Duration:** ~30 min
- **Tasks:** 2
- **Files modified:** 2 (1 created, 1 thinned)

## Accomplishments
- Created `tools/fp4/sgfp4_inject_core.hpp`: all helpers (`toLower`, `basenameOf`, `readFileBytes`, `listDirEntries` ×2, `usage`, `loadNicheDir`, `makeDequantOp`) plus `NicheDir`/`InjectedNode` structs moved verbatim into `namespace sgfp4_inject`; every free function `inline`; CLI worker renamed to `run` with identical parse/exit semantics.
- Thinned `tools/fp4/sgfp4_inject.cpp` to a 27-line shim: include + `int main` returning `sgfp4_inject::run(argc, argv)`.
- Re-verified the full Phase 5 end-to-end path post-refactor: real demo niche dir (`demo.sgfp4`, 132,368 B) → injected `out.mnn` + `out.mnn.weight`, exit 0, in-tool per-node `verified (decode==oracle)`.

## Task Commits

1. **Task 1: Create sgfp4_inject_core.hpp (move all non-main code)** — `4cc1851b` (refactor)
2. **Task 2: Thin sgfp4_inject.cpp to main-only + verify no regression** — `69f22e8d` (refactor + smoke)

## Files Created/Modified
- `tools/fp4/sgfp4_inject_core.hpp` — header-only injection core (created)
- `tools/fp4/sgfp4_inject.cpp` — thin main() shim (modified, −443/+8)

## Decisions Made
- Kept `using namespace MNN::Express;` inside `namespace sgfp4_inject` (as the plan specified, mirroring sha256.hpp's inline pattern) — it is scoped to the header's namespace and does not leak into includers.
- Removed the literal token `injectMain` even from the header's explanatory comment so the plan's grep-based acceptance criterion (`injectMain` count == 0) holds strictly.
- Restored the CLI `--help`-style usage semantics untouched; `_CRT_SECURE_NO_WARNINGS` stays tool-target-scoped in `tools/fp4/CMakeLists.txt` (no CMake edit required).

## Deviations from Plan

### Auto-fixed Issues

**1. Missing E2E base model — fallback path exercised**
- **Found during:** Task 2 (E2E smoke)
- **Issue:** Phase 5's `minimal_512.mnn` was throwaway scaffolding (never committed); neither it nor a rebuilt copy existed on disk, and the plan's primary CLI path needs it.
- **Fix:** Applied the plan's explicit fallback: regenerated an equivalent in-run base model `minimal_512.mnn` (Input[1,512] → MatMul(weight[512,512]), LCG-filled weight) via a throwaway generator (`tmp/gen_base_model.cpp`, compiled ad-hoc against the built `MNN.lib` with `/MT` to match the static-CRT build), then ran the real demo-dir E2E against it. The throwaway generator and its artifact live under untracked `tmp/` and are NOT committed.
- **Files modified:** none in-repo (tmp/ only)
- **Verification:** exit 0, `out.mnn` (real demo container injected) + `out.mnn.weight` present, `node 'Const2' {512,512} offset=0 size=132368 verified (decode==oracle)`
- **Committed in:** n/a (untracked scaffolding)

## Self-Check

PASSED — all acceptance criteria hold: header contains every required token (namespace, `inline int run`, helpers, `NOMINMAX`, oracle/version-gate/sha256/`Variable::replace`/`Variable::save`/`op->externalPath`); header has 0 hits for `int main(`/`injectMain`; shim contains the include + `return sgfp4_inject::run(argc, argv);` and 0 hits for any moved symbol; `sgfp4_inject.out` builds clean (MSVC Release); Phase 5 demo-dir E2E green with `decode==oracle`.
