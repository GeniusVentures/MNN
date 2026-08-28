---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: SGFP4 v2 Model-Artifact Injection Tool
current_phase: 07
status: executing
stopped_at: Phase 7 context gathered
last_updated: "2026-08-28T03:11:30.590Z"
last_activity: 2026-08-28
last_activity_desc: Phase 07 complete
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 7
  completed_plans: 7
  percent: 100
current_phase_name: Multi-Tensor Hardening & Structured-Data Coverage
current_plan: 05-02 complete
---

# Project State

## Project Reference

See: ROADMAP.md, REQUIREMENTS.md (both created 2026-08-22; ROADMAP.md v2.0 section added 2026-08-25; v2.0 restructured to Injection Tool 2026-08-26)
See also: `.planning/quick/260821-p1q-evaluate-current-fp4-ultra-fp4-implement/SGFP4-PIVOT-ANALYSIS.md` for full gap analysis and decision history.

**Core value:** A standalone tool takes a normally-converted `.mnn` plus real SGFP4 v2 container files (gnus-poc `fp4_exporter.py --adaptive` output) and produces a final `.mnn` + external sidecar where target weight tensors are produced by `OpType_SGFP4Dequant` nodes — verified loadable/runnable via the classic Interpreter/Session API (the downstream `SGProcessingManager` path).

## Current Position

Phase: 07
Plan: Not started
Status: Executing Phase 7
Last activity: 2026-08-28 — Phase 07 complete

## Progress

**Phases Complete:** 1/3 (v2.0 milestone)

Progress: [███░░░░░░░] 33%

## Accumulated Context

### Locked Decisions (see SGFP4-PIVOT-ANALYSIS.md for full rationale)

- Target SGFP4 v2 only — no v1 work (2026-08-22)
- GNUS Execution Integrity / attestation out of scope for MNN — MNN runs AI processing and returns a result, SuperGenius verifies (2026-08-22)
- MNN-only scope — SuperGenius/SGProcessingManager integration is a separate GSD plan in that repo (2026-08-22)
- Container: external `.mnn.weight`-style sidecar file + minimal `{magic, offset, size}` op descriptor, mirroring `Convolution2D.external`; no macroblock/quadtree typed FlatBuffers fields (2026-08-22)
- v2.0: "op-type rewrite" is graph surgery — a new `SGFP4Dequant` node feeding the original, type-unchanged conv/deconv op's `inputs[1]`, not in-place op mutation (research, 2026-08-25; carried into v2.0 injection design as consumer-rewiring)
- v2.0: real-weight validation scheduled before graph-rewrite integration — synthetic-fixture-tuned assumptions are the top-flagged risk (research, 2026-08-25; now applies to v3.0 Phases 10→11 ordering)
- v2.0 restructure (2026-08-26): injection tool (post-hoc graph surgery on converted `.mnn` + gnus-poc containers) inserted as v2.0 ahead of converter integration (now v3.0) — chosen at 0% execution to front-load a real loadable artifact and de-risk the converter milestone
- Injection-tool contract: `op->externalPath` is set literally on the `SGFP4Dequant` op (not session-derived — `createExecutionWithExternal` doesn't cover this op type); serialization uses `Variable::save` direct-to-file overload; final artifact must verify via classic Interpreter/Session API, not just Express `Module::load`
- Canonical real-weight encoder is gnus-poc `fp4_exporter.py --adaptive` (v2, byte-verified vs `SGFP4DequantUtils.hpp`); MNN's `tools/fp4/encode_sgfp4.py` is test-oracle-only (2026-08-26)

### Pending Todos

- v1.0 archived 2026-08-26 (`.planning/milestones/v1.0-ROADMAP.md`, `v1.0-REQUIREMENTS.md`). SGV2-07..11 checkboxes and traceability rows corrected during archival (were stale — work was done, verify step just never flipped them).
- **2026-08-26 restructure:** v2.0 is now the Model-Artifact Injection Tool (Phases 5-7, SGINJ-01..08), drafted from the SGFP4 handoff. The former v2.0 Converter Integration moved to v3.0 (Phases 8-12, SGV2-22..32) at 0% execution / zero plans — no renumbering cost. Next: `/gsd-plan-phase 5`.
- v3.0 planning-time re-evaluations (noted in ROADMAP.md/REQUIREMENTS.md): whether the Phase 9 C++ encoder port is still justified vs. direct consumption of gnus-poc exporter output (injection tool consumes Python-produced containers directly); the real-validation model/corpus selection and non-64-multiple tiling/padding convention gaps move with Phase 10; the CLI flag naming question moves with Phase 11.
- Starter artifact for Phase 5: `gnus-poc/models/specialists_mlx/demo/fp4/demo.sgfp4` (132,368 bytes, 512×512, byte-verified) — uniform random noise, all `UNIFORM_64`; a structured (non-uniform) second artifact is REQUIRED before Phase 7's quadtree coverage criterion can pass.
- Do NOT consume gnus-poc `pipeline/runner.py` default quantize output (invokes exporter without `--adaptive` → legacy v1, unsupported by the decoder). gnus-poc-side fix needed; not this workstream's job.
- Terminology: the format is "SGFP4 v2" — never "Ultra FP4" (that name collision with gnus-poc manifests refers to a different, unrelated E2M1 format from the sibling `milestone` workstream).
- `test/op/FP4ModelTest.cpp` (pre-existing, unrelated dead code from `milestone` workstream commit `cffaf4bd`) still blocks a from-scratch `run_test.out` build; see `01-affine-dual-mode-decode-core-cpu-uniform-layouts/deferred-items.md`. Recommend the `milestone` workstream's own Phase 4 plan 04-02 fix or remove it — confirmed still broken/unfixed as of this session (04-02-SUMMARY.md's Deviations section).
- Doc debt: no `02-VERIFICATION.md` exists (Phases 1 and 3 have one) — deferred, acknowledged in MILESTONES.md v1.0 entry; run `/gsd-verify-work 2` retroactively if a formal verification artifact is ever wanted

## Session Continuity

**Last session:** 2026-08-28T01:19:46.808Z

**Stopped At:** Phase 7 context gathered
**Resume File:** .planning/workstreams/sgfp4-pivot/phases/07-multi-tensor-hardening-structured-data-coverage/07-CONTEXT.md

## Performance Metrics

| Phase | Plan | Duration | Notes |
|-------|------|----------|-------|
| Phase 01 P01 | 20min | 3 tasks | 11 files |
| Phase 01 P02 | 40min | 2 tasks | 4 files |
| Phase 02 P01 | 25min | 2 tasks | 2 files |
| Phase 02 P02 | 35min | 3 tasks | 3 files |
| Phase 03 P01 | 95min | toolchain + build gate | 0 code files |
| Phase 03 P02 | 45min | GLSL uniform decode shader | makeshader regen |
| Phase 03 P03 | 30min | Vulkan Execution + registration | 2 files |
| Phase 03 P04 | 35min | dual-backend parity test | 2 files |
| Phase 04 P01 | 10min | LAYOUT_MIXED GLSL decode + shader regen | 2 files (2 more regenerated byte-identical) |
| Phase 04 P02 | 25min | 2 tasks | 1 files |
| Phase 05 P01 | 30min | graph-surgery spike + version gate | 2 files |
| Phase 05 P02 | 40min | sgfp4_inject tool + end-to-end | 4 files |

## Quick Tasks Completed

| Date | Slug | Result | Commit |
|------|------|--------|--------|
| 2026-08-25 | backfill-sgfp4-pivot-phase2-completion | ROADMAP/STATE Phase 2 completion backfill (docs only) | 2333a38b |
| 2026-08-26 | v2.0-restructure-injection-tool | v2.0 → Injection Tool restructure; Converter Integration → v3.0 | 8cc5e2f9 |

## Decisions

- [Phase 04 P01]: GLSL quadtree walk stack holds only edge-size `n` (never x/y) — dequant_sgfp4_container_cpu never reads QuadNode.x/.y and all 4 split children share edge n/2u, so push/pop order among identical values is irrelevant
- [Phase 04 P01]: MIXED branch in locateElement is self-contained with its own `continue` rather than falling through the shared uniform N*n*n tail, since MIXED has no fixed per-leaf n
- [Phase 04 P01]: WSL-hosted glslang toolchain (symlinked Windows .exe) requires relative paths from a drvfs (/mnt/...) cwd — /tmp-based Linux paths are unresolvable even after wslpath -w conversion
- [Phase 02]: Encoder subdivision hysteresis blocks noise-scaling splits — selftest/fixtures use constructive ramp tiles (full ramp amp 60 = all-split → FULL_4X4 collapse; TL-quadrant ramp amp 12 = asymmetric MIXED)
- [Phase 02]: Spec §6.3 uniform collapse is normative — all-split and constant tiles MUST emit uniform enums (5 / 0), not MIXED; asserted in selftest
- [Phase 02]: Split-map negatives: a complete bitmap always tiles exactly, so area!=4096 is defense-in-depth (unreachable from pure bitmaps); observable negatives are bit-exhaustion (85-bit cap) and truncation cases

- [Phase 01]: SGFP4 decode order is fully sequential/linear (records then leaves); Plan 01-02 encoder must match this byte order
- [Phase 01]: Manual minimal append to ShapeRegister.cpp/CPUOPRegister.cpp instead of full register.py regen, since Windows directory ordering reorders the whole file
- [Phase 01]: Pitfall 2 resolved: buffer-based Module::load does not auto-set externalPath; Plan 01-02 tests must call rtmgr->setExternalFile() before Module::load(buffer,...)
- [Phase 01]: Op.externalPath must be set directly on the OpT for OpType_SGFP4Dequant (rtmgr->setExternalFile alone does not populate it, since createExecutionWithExternal only rewrites externalPath for Convolution2D/Scale/LayerNorm)
- [Phase 01]: Fixed CPUSGFP4Dequant's broken T-01-04 DoS bound: FileLoader::size() is only populated by the whole-file read(), not the offset+size read this op uses; replaced with a direct std::ifstream file-size probe
- [Phase 04 P02]: Kept SGFP4VulkanDequantTest class name and op/sgfp4/vulkan_uniform_parity registration string unchanged after removing the LAYOUT_MIXED skip, per CONTEXT.md Claude's-Discretion, avoiding churn to docs/scripts referencing that exact suite string
- [Phase 04 P02]: Full-suite FP4ModelTest.cpp temp-stub workaround was attempted but blocked by the sandbox classifier while building with a locally-modified out-of-scope file; fell back to the plan's actual required filtered-suite verification (op/sgfp4/, op/fp4, op/vulkan/fp4_dequant_correctness), which had already passed

## Operator Next Steps

- Run `/gsd-verify-work 5` to formally verify Phase 5 (optional — Phase 6's classic-API suites re-prove artifact validity load-level; or verify both phases together after Phase 6)
- Then `/gsd-execute-phase 6` (06-01 shared-core refactor → 06-02 classic-API test)
- Before/during Phase 6: request or generate a structured (non-uniform) SGFP4 v2 container from gnus-poc — needed for Phase 7's quadtree coverage criterion; the existing demo artifact is all `UNIFORM_64`
- FP4ModelTest.cpp full-suite build blocker: local unblock is now cleaner than the temp-stub cycle — after every `cmake` configure, filter the file out of the untracked generated `.build/run_test.out.vcxproj` (see 05-01-SUMMARY Deviations); permanent fix still owned by the `milestone` workstream (04-02)
