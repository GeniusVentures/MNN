---
phase: 03-vulkan-decode-uniform-layouts
plan: 03
subsystem: vulkan
tags: [vulkan, execution, sgfp4, sidecar, fileloader]

requires:
  - phase: 03-vulkan-decode-uniform-layouts/03-02
    provides: "Embedded shader keys glsl_sgfp4_dequant[_FP16]_comp + descriptor contract"
  - phase: 01-affine-dual-mode-decode-core-cpu-uniform-layouts
    provides: "dequant_sgfp4_container_cpu as the host pre-validator + CPUSGFP4Dequant loading pattern"
provides:
  - "Registered Vulkan buffer-backend Execution for OpType_SGFP4Dequant with host-pre-validated container upload"
affects: [03-04]

tech-stack:
  added: []
  patterns: ["creator-side load+validate+upload (compensates VulkanBasicExecution's missing onResize hook)", "named kSgfp4WorkgroupSize=256 linked to shader local_size_x"]

key-files:
  created:
    - source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.hpp
    - source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp
  modified: []

key-decisions:
  - "Constructor takes already-validated container bytes (std::vector<uint8_t>) — the creator is the single validation point"
  - "outElementCount derived from outputs[0]->elementSize() in the creator (consistent with onEncode dispatch); scratch vector validates then frees before construction"
  - "FP32 flag reserved (useFP32Output=false default), FP16 selected when vkBn->useFP16()"

patterns-established:
  - "0-input Const-like Vulkan op: onEncode ignores inputs entirely; source is the container SSBO"

requirements-completed: [SGV2-13]

coverage:
  - id: D1
    description: "Vulkan Execution class registered on OpType_SGFP4Dequant with host sidecar load + one-time pre-validation"
    requirement: SGV2-13
    verification:
      - kind: other
        ref: "build green with auto-globbed files; grep: OpType_SGFP4Dequant, dequant_sgfp4_container_cpu, queryFileSize in VulkanSGFP4Dequant.cpp"
        status: pass
    human_judgment: false
  - id: D2
    description: "GPU decode dispatch executes end-to-end (validation ensures decodeability; correctness proven in plan 03-04)"
    requirement: SGV2-13
    verification: []
    human_judgment: true
    rationale: "First live GPU execution of this Execution is plan 03-04's parity test; not observable in this plan"

duration: 30min
completed: 2026-08-24
status: complete
---

# Phase 03 Plan 03: VulkanSGFP4Dequant Execution Summary

Registered Vulkan buffer-backend Execution that loads the same external-sidecar descriptor as the CPU path, bounds-checks the sidecar size before any allocation, pre-validates the container host-side once, and dispatches both shader variants — E2M1 path untouched.

## Performance

- **Duration:** ~30 min
- **Tasks:** 2/2
- **Files:** 2 created (258 lines)

## Accomplishments

- `VulkanSGFP4Dequant.hpp/.cpp`: `VulkanBasicExecution` subclass; creator clones `CPUSGFP4Dequant.cpp`'s `queryFileSize` + USE_EXTERNAL_DATA/externalPath gate + bounded `FileLoader` read (T-03-02 probe-before-alloc, source-order preserved).
- D-05 pre-validation: scratch-buffer `dequant_sgfp4_container_cpu` pass; false → `nullptr` from `onCreate` + `MNN_ERROR` (no upload, no dispatch, no partial writes).
- D-01 upload only after validation: host-data `VulkanBuffer` ctor for the SSBO; `onEncode` is copy-free bind/write/dispatch with `UP_DIV(elementCount, kSgfp4WorkgroupSize)` and `barrierSource`.
- Registered via `static bool gResistor` lambda → `VulkanBackend::addCreator(OpType_SGFP4Dequant, ...)`.
- Build green (MSVC, no RTTI/exceptions — error paths are returns); `op/sgfp4/` CPU suites pass (2/2).

## Deviations from Plan

None - plan executed exactly as written.

## Authentication Gates

None.

## Issues Encountered

None.

## Next Plan Readiness

Ready for 03-04: parity test should build a Dequantize-free model with `OpType_SGFP4Dequant` + `SGFP4DequantParam` (external `{offset,size}`, `op->externalPath` set), run it on a `MNN_FORWARD_VULKAN` schedule config, and compare against the CPU reference with `checkVectorByRelativeError`.

## Self-Check: PASSED

- Header guard `VulkanSGFP4Dequant_hpp`; no onResize override ✓
- `queryFileSize` invoked before `container` allocation and any VulkanBuffer construction ✓
- Scratch validation false → no Execution construction ✓
- Descriptor path identical to CPUSGFP4Dequant (`main_as_SGFP4DequantParam` + `externalPath`) ✓
- Both shader keys + `useFP16()` selection chain present ✓
- onEncode reads no input tensor; `kSgfp4WorkgroupSize == 256` ✓
- `git diff --exit-code` clean on `VulkanFP4Dequant.*` ✓
- Build green; `op/sgfp4/` passed:2 ✓
