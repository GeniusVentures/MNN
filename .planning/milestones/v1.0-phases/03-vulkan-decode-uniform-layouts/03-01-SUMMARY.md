---
phase: 03-vulkan-decode-uniform-layouts
plan: 01
subsystem: infra
tags: [vulkan, glslang, shaderc, cmake, toolchain]

requires:
  - phase: 02-adaptive-quadtree-layout-cpu-layout-mixed
    provides: "CPU decode core + committed fixtures used for the regression baseline"
provides:
  - "glslangValidator + spirv-opt reachable from WSL (~/.local/bin symlinks to shaderc build)"
  - "Vulkan buffer backend build configuration (.build: MNN_VULKAN=ON, MNN_VULKAN_IMAGE=OFF)"
  - "run_test.out with Vulkan backend linked; GPU device path proven by op/vulkan/fp4_dequant_correctness"
affects: [03-02, 03-03, 03-04]

tech-stack:
  added: ["shaderc glslangValidator 11:14.3.0 via WSL interop (no new packages)"]
  patterns: ["WSL symlink to Windows shaderc exes; relative-path invocation (Win32 exes cannot resolve Linux absolute paths)"]

key-files:
  created: []
  modified: ["source/backend/vulkan/buffer/execution/glsl/fp4_dequant.comp (deviation fix)", ".build/CMakeCache.txt (untracked build dir)"]

key-decisions:
  - "User-directed toolchain: reuse the existing shaderc build at thirdparty/build/Windows/Release/shaderc/bin instead of installing glslang-tools in WSL"
  - "WSL executes the Windows glslangValidator.exe via interop; symlinks are extensionless so subprocess.run(['glslangValidator']) resolves"

patterns-established:
  - "makeshader.py regeneration host: WSL bash with PATH=$HOME/.local/bin prepended; run from source/backend/vulkan/buffer/compiler"
  - "WSL find ordering differs from the environment that produced the committed artifacts — regeneration reorders entries (content-equivalent churn)"

requirements-completed: [SGV2-12]

coverage:
  - id: D1
    description: "glslangValidator reachable from the POSIX environment that runs makeshader.py"
    requirement: SGV2-12
    verification:
      - kind: other
        ref: "wsl: glslangValidator --version -> Glslang Version: 11:14.3.0"
        status: pass
    human_judgment: false
  - id: D2
    description: ".build configured with MNN_VULKAN=ON / MNN_VULKAN_IMAGE=OFF producing run_test.out with the Vulkan buffer backend linked"
    requirement: SGV2-12
    verification:
      - kind: other
        ref: "Select-String .build/CMakeCache.txt 'MNN_VULKAN:BOOL=ON' + run_test.out.exe link success"
        status: pass
    human_judgment: false
  - id: D3
    description: "op/vulkan/fp4_dequant_correctness passes on the physical RTX GPU (device path end-to-end)"
    requirement: SGV2-12
    verification:
      - kind: unit
        ref: "run_test.out.exe op/vulkan/fp4_dequant_correctness -> passed:1 failed:0"
        status: pass
    human_judgment: false
  - id: D4
    description: "CPU regression baseline op/sgfp4/ stays green (Phase 1/2 suites)"
    requirement: SGV2-12
    verification:
      - kind: unit
        ref: "run_test.out.exe op/sgfp4/ -> passed:2 failed:0 (uniform_decode + mixed_decode)"
        status: pass
    human_judgment: false

duration: 95min
completed: 2026-08-24
status: complete
---

# Phase 03 Plan 01: Toolchain + Vulkan Build Gate Summary

Offline-shader toolchain provisioned via existing shaderc build (WSL interop), build reconfigured to the Vulkan buffer backend, and the GPU device path proven with a green Vulkan FP4 test after fixing two latent shader bugs it exposed.

## Performance

- **Duration:** ~95 min
- **Tasks:** 2/2 (1 checkpoint decision + 1 auto)
- **Files modified:** 1 source file (deviation) + build cache

## Accomplishments

- Checkpoint resolved by user direction: reuse `thirdparty/build/Windows/Release/shaderc/bin` (glslangValidator 11:14.3.0 + spirv-opt + full SPIRV-Tools). Extensionless symlinks created in WSL `~/.local/bin`; `makeshader.py` runs green end-to-end from WSL.
- `.build` reconfigured: `MNN_VULKAN:BOOL=ON`, `MNN_VULKAN_IMAGE:BOOL=OFF` (MSVC 17 2022 generator, cached). `run_test.out` links with the Vulkan buffer backend.
- `op/vulkan/fp4_dequant_correctness` passes on the RTX 4070 Ti SUPER — first verified GPU dispatch round-trip on this machine.
- `op/sgfp4/` CPU suites pass (2/2) — no Phase 1/2 regressions.
- FP4ModelTest.cpp pre-existing build blocker handled exactly per the Phase-1 documented temp-stub workaround (stub → build → restore → `git diff --exit-code` clean; nothing stubbed committed).

## Deviations from Plan

1. **[Rule 1 - bug] `fp4_dequant.comp` byte-vs-word indexing** — Found during: Task 2 smoke test. The pre-existing shader indexed `SrcRaw[]` (a u32 word array) with a byte offset, so element 2 read word 1 instead of byte 0 (expected +1.0, got −0.0). Fix: select word via `byteIndex>>2`, byte via `(byteIndex&3)*8`. Commit `d6ad5242`.
2. **[Rule 1 - bug] `fp4_dequant.comp` E2M1 special e=3/m=1 decoded to ±Inf** — Found during: same test run after fix 1. Spec/test contract expects NaN. Fix: `0.0/0.0` for m=1. Commit `d6ad5242`.
3. **[Rule 3 - blocker] makeshader.py regeneration churn** — WSL `find` ordering differs from the original generator environment, so all three regenerated artifacts reorder existing entries (~139k lines of content-equivalent churn). User approved accepting the churn (checkpoint). Churn landed in the plan 03-02 regeneration commit; future regenerations are stable.
4. **[Rule 2 - missing tool] `makeshader.py` exit code is not trustworthy** — On glslang failure it catches, prints, and continues (exception traceback, exit 0). Always grep its log for `error` instead of trusting `$?`.

## Authentication Gates

None.

## Issues Encountered

None open. The pre-existing `FP4ModelTest.cpp` build blocker remains owned by the milestone workstream's Phase 4 plan 04-02 (unchanged; workaround documented in Phase 1 deferred-items.md and reused here exactly).

## Environment Record (for plan 03-02's regeneration)

- Host: **WSL bash** with `export PATH="$HOME/.local/bin:$PATH"`; run `python3 makeshader.py` from `source/backend/vulkan/buffer/compiler`.
- Windows glslang exes cannot resolve Linux absolute paths — invoke with relative paths from a drvfs cwd (WSL translates the cwd).

## Self-Check: PASSED

- `.build/CMakeCache.txt` contains `MNN_VULKAN:BOOL=ON` + `MNN_VULKAN_IMAGE:BOOL=OFF` ✓
- `run_test.out` builds with Vulkan buffer backend linked ✓
- `op/vulkan/fp4_dequant_correctness` passes (passed:1, failed:0) ✓
- `op/sgfp4/` passes (passed:2, failed:0) ✓
- `git diff --exit-code test/op/FP4ModelTest.cpp` clean after stub cycles ✓
