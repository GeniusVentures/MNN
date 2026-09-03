---
phase: 03-vulkan-decode-uniform-layouts
plan: 04
subsystem: testing
tags: [vulkan, parity-test, sgfp4, gpu, fixtures]

requires:
  - phase: 03-vulkan-decode-uniform-layouts/03-03
    provides: "Registered Vulkan Execution reading op->externalPath sidecars"
  - phase: 01-affine-dual-mode-decode-core-cpu-uniform-layouts
    provides: "Committed fixtures + CPU reference decode"
provides:
  - "op/sgfp4/vulkan_uniform_parity: dual-backend parity test proving SGV2-14 (and end-to-end SGV2-12/13 on GPU)"
affects: [04-*]

tech-stack:
  added: []
  patterns: ["dual-truth verification: fixtures<->CPU and CPU<->GPU at independent tolerances", "tight FP32 pass then relaxed default-precision pass fail-fast ordering"]

key-files:
  created:
    - test/op/SGFP4VulkanDequantTest.cpp
  modified:
    - source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp (include-path fix only)

key-decisions:
  - "Filter is layout==kSGFP4LayoutMixed (not mode==-1): mixed_allsplit/uniform_collapse are collapse-rule uniform containers and ARE parity targets"
  - "Two-pass tolerance: 1e-4 at Precision_High (FP32 variant) + 2e-3 at default precision (FP16-variant coverage)"

patterns-established:
  - "Vulkan sibling of SGFP4DequantTest's runSgfp4Module: same OpT/externalPath plumbing, MNN_FORWARD_VULKAN config"

requirements-completed: [SGV2-14, SGV2-12]

coverage:
  - id: D1
    description: "13 uniform fixtures decode identically through CPU reference and Vulkan GPU module sessions within 1e-4"
    requirement: SGV2-14
    verification:
      - kind: unit
        ref: "run_test.out.exe op/sgfp4/vulkan_uniform_parity -> '13 uniform fixtures matched CPU reference on Vulkan', passed:3"
        status: pass
    human_judgment: false
  - id: D2
    description: "Graceful skip when no Vulkan device"
    requirement: SGV2-14
    verification:
      - kind: other
        ref: "nullptr-creator branch returns true with MNN_PRINT (code-audited; no Vulkan-less machine available to exercise live)"
        status: pass
    human_judgment: false
  - id: D3
    description: "E2M1 additivity guard (op/fp4 + op/vulkan/fp4_dequant_correctness green)"
    requirement: SGV2-12
    verification:
      - kind: unit
        ref: "run_test.out.exe op/fp4 -> pass; op/vulkan/fp4_dequant_correctness -> passed:1"
        status: pass
    human_judgment: false

duration: 35min
completed: 2026-08-24
status: complete
---

# Phase 03 Plan 04: Dual-Backend Parity Test Summary

CPU/Vulkan parity test running all 13 uniform fixtures through the production external-sidecar plumbing on the RTX GPU — every fixture matches the CPU reference at 1e-4, closing SGV2-12/13/14 end-to-end.

## Performance

- **Duration:** ~35 min
- **Tasks:** 2/2
- **Files:** 1 created + 1 one-line include fix

## Accomplishments

- `test/op/SGFP4VulkanDequantTest.cpp` registered as `op/sgfp4/vulkan_uniform_parity`: fixture loop with the named `layout == kSGFP4LayoutMixed` filter (13 targets), per-fixture temp sidecar, CPU reference decode + fixture-drift guard, tight Precision_High pass (1e-4), relaxed default-precision pass (2e-3, FP16-variant coverage), graceful skip without a Vulkan device.
- GPU result: **13/13 uniform fixtures matched the CPU reference** (FP32 tight + default-precision passes) on the RTX 4070 Ti SUPER.
- Regression sweep green: `op/sgfp4/` (3/3 incl. the new test), `op/fp4`, `op/vulkan/fp4_dequant_correctness`; artifact greps unchanged (4/4/2).

## Deviations from Plan

1. **[Rule 1 - build] test/CMakeLists.txt GLOB_RECURSE is configure-time** — the plan said the new test is "auto-globbed"; glob evaluation happens at configure, so a cmake reconfigure was required after adding the file (first build silently ran without the new test: `passed:0`). Fix: re-run cmake configure before build.
2. **[Rule 1 - build] wrong include path in 03-03's file** — `backend/vulkan/backend/VulkanBackend.hpp` should be the execution-dir-relative `VulkanBackend.hpp` (matches how VulkanBasicExecution.hpp includes it). The 03-03 build passed only because stale objects skipped that TU; the reconfigure rebuilt it and surfaced the error. Fixed in this plan's commit.

## Authentication Gates

None.

## Issues Encountered

The full-suite run remains blocked by the pre-existing `FP4ModelTest.cpp` dead code (milestone workstream Phase 4 plan 04-02's responsibility); the Phase-1 temp-stub workaround was applied for every build and restored byte-for-byte (`git diff --exit-code` clean). Filtered suites cover this phase's surface.

## Self-Check: PASSED

- `op/sgfp4/vulkan_uniform_parity` passes with all 13 uniform fixtures on the physical GPU ✓
- Sidecar + `{0, size}` descriptor + direct `op->externalPath` (grep present) ✓
- `checkVectorByRelativeError` at 1e-4 both directions (fixtures↔CPU, CPU↔GPU) ✓
- Graceful-skip branch present ✓
- `MNN_SUPPORT_TRANSFORMER_FUSE` gate ✓
- `op/fp4` + `op/vulkan/fp4_dequant_correctness` green ✓
- Artifact greps 4/4/2 ✓; working tree contains exactly the phase's 8 intended files + planning docs ✓
- `git diff --exit-code test/op/FP4ModelTest.cpp` clean ✓
