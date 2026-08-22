---
workstream: sgfp4-pivot
created: 2026-08-22
---

# Project State

## Project Reference

See: ROADMAP.md, REQUIREMENTS.md (both created 2026-08-22)
See also: `.planning/quick/260821-p1q-evaluate-current-fp4-ultra-fp4-implement/SGFP4-PIVOT-ANALYSIS.md` for full gap analysis and decision history.

**Core value:** A working SGFP4 v2 (quadtree-adaptive, affine dual-mode) weight-decode path in MNN — CPU and Vulkan — additive to the existing E2M1 "Ultra FP4" implementation.

## Current Position

**Status:** Roadmap defined, Phase 1 not yet planned
**Current Phase:** 1 (Affine Dual-Mode Decode Core — CPU, Uniform Layouts)
**Last Activity:** 2026-08-22
**Last Activity Description:** ROADMAP.md + REQUIREMENTS.md created (4 phases, 16 v1 requirements, SGV2-01..16)

## Progress

**Phases Complete:** 0/4
**Current Plan:** N/A — Phase 1 plan(s) not yet broken down

## Accumulated Context

### Locked Decisions (see SGFP4-PIVOT-ANALYSIS.md for full rationale)

- Target SGFP4 v2 only — no v1 work (2026-08-22)
- GNUS Execution Integrity / attestation out of scope for MNN — MNN runs AI processing and returns a result, SuperGenius verifies (2026-08-22)
- MNN-only scope — SuperGenius/SGProcessingManager integration is a separate GSD plan in that repo (2026-08-22)
- Container: external `.mnn.weight`-style sidecar file + minimal `{magic, offset, size}` op descriptor, mirroring `Convolution2D.external`; no macroblock/quadtree typed FlatBuffers fields (2026-08-22)

### Pending Todos

- Plan Phase 1 (run /gsd-plan-phase 1 or equivalent)
- Open, non-blocking: whether to execute the `milestone` workstream's Phase 4 plan 04-02 before/alongside this work

## Session Continuity

**Stopped At:** Roadmap + requirements created, Phase 1 planning next
**Resume File:** None
