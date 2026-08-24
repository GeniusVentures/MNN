---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 3 — Vulkan LAYOUT_MIXED Port (GLSL walker)
current_plan: Not started
status: executing
stopped_at: Phase 3 context gathered
last_updated: "2026-08-24T23:41:31.858Z"
last_activity: 2026-08-24
last_activity_desc: Phase 2 execution complete — 2/2 plans (SGV2-08..11)
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 4
  completed_plans: 4
  percent: 50
---

# Project State

## Project Reference

See: ROADMAP.md, REQUIREMENTS.md (both created 2026-08-22)
See also: `.planning/quick/260821-p1q-evaluate-current-fp4-ultra-fp4-implement/SGFP4-PIVOT-ANALYSIS.md` for full gap analysis and decision history.

**Core value:** A working SGFP4 v2 (quadtree-adaptive, affine dual-mode) weight-decode path in MNN — CPU and Vulkan — additive to the existing E2M1 "Ultra FP4" implementation.

## Current Position

**Status:** Phase 2 executed — awaiting verify
**Current Phase:** 3 — Vulkan LAYOUT_MIXED Port (GLSL walker)
**Last Activity:** 2026-08-24 — Phase 2 execution complete (2/2 plans)
**Last Activity Description:** Phase 2 execution complete — 2/2 plans (SGV2-08..11)

## Progress

**Phases Complete:** 2/4
**Current Plan:** Not started

## Accumulated Context

### Locked Decisions (see SGFP4-PIVOT-ANALYSIS.md for full rationale)

- Target SGFP4 v2 only — no v1 work (2026-08-22)
- GNUS Execution Integrity / attestation out of scope for MNN — MNN runs AI processing and returns a result, SuperGenius verifies (2026-08-22)
- MNN-only scope — SuperGenius/SGProcessingManager integration is a separate GSD plan in that repo (2026-08-22)
- Container: external `.mnn.weight`-style sidecar file + minimal `{magic, offset, size}` op descriptor, mirroring `Convolution2D.external`; no macroblock/quadtree typed FlatBuffers fields (2026-08-22)

### Pending Todos

- Phase 02 executed (both plans); next: `/gsd-verify-work` for Phase 02, then plan Phase 3 (Vulkan LAYOUT_MIXED port per ROADMAP.md)
- `test/op/FP4ModelTest.cpp` (pre-existing, unrelated dead code from `milestone` workstream commit `cffaf4bd`) blocks a from-scratch `run_test.out` build; see `01-affine-dual-mode-decode-core-cpu-uniform-layouts/deferred-items.md`. Recommend the `milestone` workstream's Phase 4 plan 04-02 fix or remove it.
- Open, non-blocking: whether to execute the `milestone` workstream's Phase 4 plan 04-02 before/alongside this work

## Session Continuity

**Last session:** 2026-08-24T23:41:31.854Z

**Stopped At:** Phase 3 context gathered
**Resume File:** .planning/workstreams/sgfp4-pivot/phases/03-vulkan-decode-uniform-layouts/03-CONTEXT.md

## Performance Metrics

| Phase | Plan | Duration | Notes |
|-------|------|----------|-------|
| Phase 01 P01 | 20min | 3 tasks | 11 files |
| Phase 01 P02 | 40min | 2 tasks | 4 files |
| Phase 02 P01 | 25min | 2 tasks | 2 files |
| Phase 02 P02 | 35min | 3 tasks | 3 files |

## Decisions

- [Phase 02]: Encoder subdivision hysteresis blocks noise-scaling splits — selftest/fixtures use constructive ramp tiles (full ramp amp 60 = all-split → FULL_4X4 collapse; TL-quadrant ramp amp 12 = asymmetric MIXED)
- [Phase 02]: Spec §6.3 uniform collapse is normative — all-split and constant tiles MUST emit uniform enums (5 / 0), not MIXED; asserted in selftest
- [Phase 02]: Split-map negatives: a complete bitmap always tiles exactly, so area!=4096 is defense-in-depth (unreachable from pure bitmaps); observable negatives are bit-exhaustion (85-bit cap) and truncation cases

- [Phase 01]: SGFP4 decode order is fully sequential/linear (records then leaves); Plan 01-02 encoder must match this byte order
- [Phase 01]: Manual minimal append to ShapeRegister.cpp/CPUOPRegister.cpp instead of full register.py regen, since Windows directory ordering reorders the whole file
- [Phase 01]: Pitfall 2 resolved: buffer-based Module::load does not auto-set externalPath; Plan 01-02 tests must call rtmgr->setExternalFile() before Module::load(buffer,...)
- [Phase 01]: Op.externalPath must be set directly on the OpT for OpType_SGFP4Dequant (rtmgr->setExternalFile alone does not populate it, since createExecutionWithExternal only rewrites externalPath for Convolution2D/Scale/LayerNorm)
- [Phase 01]: Fixed CPUSGFP4Dequant's broken T-01-04 DoS bound: FileLoader::size() is only populated by the whole-file read(), not the offset+size read this op uses; replaced with a direct std::ifstream file-size probe
