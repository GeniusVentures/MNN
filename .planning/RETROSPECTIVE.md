# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — SGFP4 v2 Decode (Vulkan-parity)

**Shipped:** 2026-08-26
**Phases:** 4 | **Plans:** 10 | **Sessions:** ~4 (2026-08-22 → 2026-08-25)

### What Was Built
- CPU affine dual-mode decode core (`w = S·c + bias`, FP4_AFFINE + T158_AFFINE), v2 stream framing, external-sidecar container loading, and a reference Python encoder with round-trip tests across all five uniform layouts
- CPU LAYOUT_MIXED adaptive quadtree decode (pre-order DFS split-map walk) plus an error-driven encoder with per-level MSE thresholds and ternary outlier veto
- Vulkan GLSL uniform-layout decode shader with CPU/Vulkan parity
- Vulkan LAYOUT_MIXED quadtree walk on GPU, closing out full CPU/Vulkan parity across 14 committed fixtures

### What Worked
- Following MNN's own `skills/add-new-op/SKILL.md` order (schema → shape → CPU backend → tests → other backends) kept each phase's blast radius small — CPU correctness was locked down before the harder GPU quadtree-in-a-shader work was attempted
- Locking scope early (v2-only, external-sidecar container, attestation explicitly out of scope) in the pre-milestone gap analysis prevented scope creep across all 4 phases
- Building the SGFP4 v2 work as strictly additive to the existing E2M1 "Ultra FP4" path (separate op type, separate Execution classes) meant zero regression risk to the live cross-repo `dequant_fp4_packed_cpu()` contract

### What Was Inefficient
- Phase 2's formal verification step never ran, leaving `02-VERIFICATION.md` missing and REQUIREMENTS.md checkboxes/traceability rows stale (SGV2-08..11 unchecked, SGV2-07 traceability row not updated) until caught during milestone close — cost a full review pass to reconcile at the end instead of at Phase 2's own close
- WSL-hosted glslang toolchain path resolution (Phase 3) cost significant time (95min on Wave 0 alone) working out that relative paths from a drvfs cwd are required

### Patterns Established
- New backend Executions for additive formats should register under their own `OpType` (not overload an existing dequant op) and reuse the `Convolution2D.external`-style `{magic, offset, size}` sidecar descriptor pattern rather than modeling format internals as typed FlatBuffers fields
- GLSL quadtree/stack-walking kernels should carry only the minimum state needed per node (e.g. edge size `n`, not `x`/`y`) when the CPU reference doesn't need positional state either — keeps shader register pressure down

### Key Lessons
1. Flip requirement checkboxes and write the phase's VERIFICATION.md as part of that phase's own close-out, not deferred — catching it 3 phases later at milestone-close time means re-deriving context that was fresh right after Phase 2 shipped
2. When a `/gsd-new-milestone` run discovers the previous milestone was never formally closed (no archive), stop and run `/gsd-complete-milestone` first — running `phases.clear` against unarchived phase directories would have destroyed the only easily-navigable copy of 4 phases' PLAN/SUMMARY/VERIFICATION docs

### Cost Observations
- Model mix: researcher/synthesizer/roadmapper agents ran on `deepseek-direct/deepseek-v4-pro` (configured model_profile_overrides for the `opencode` runtime, not Claude-native)
- Sessions: ~4 across 2026-08-22 → 2026-08-25
- Notable: Phase 3's Wave 0 (toolchain provisioning) was the single most expensive plan (95min) despite touching zero code files — environment setup cost exceeded implementation cost for that wave

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | ~4 | 4 | First milestone for the sgfp4-pivot workstream; established additive-format pattern (separate OpType, external sidecar) |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|--------------------|
| v1.0 | 14 CPU/Vulkan parity fixtures + uniform/mixed round-trip suites | Both code modes × all 5 uniform layouts + LAYOUT_MIXED | Reference Python encoder (`encode_sgfp4.py`), no new runtime deps |

### Top Lessons (Verified Across Milestones)

1. Close out each phase's verification before moving to the next — deferred verification compounds into milestone-close overhead
