# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v2.0 — SGFP4 v2 Model-Artifact Injection Tool

**Shipped:** 2026-08-28
**Phases:** 3 | **Plans:** 7 | **Sessions:** ~3 (2026-08-26 → 2026-08-28) | **Commits:** 38 | **Code:** 12 files, +11,627 lines

### What Was Built
- Standalone `sgfp4_inject` tool: normally-converted `.mnn` + gnus-poc SGFP4 v2 containers → new `.mnn` + merged external sidecar with `OpType_SGFP4Dequant` weight-producing nodes
- Classic Interpreter/Session API proof: injected artifacts run the exact downstream `SGProcessingManager` path with FP32 parity (rtol 1e-4) and named I/O preserved
- Structured LAYOUT_MIXED fixture from the real gnus-poc encoder (byte-deterministic, 12 MIXED superblocks) + multi-tensor/malformed-input suites (9/9 family green, 13/13 clean-fail probes)

### What Worked
- Front-loading a real loadable artifact via post-hoc graph surgery (the 2026-08-26 restructure at 0% execution) de-risked the converter milestone before any converter code was written — the tool now serves as the reference implementation v3.0 Phase 11 can absorb
- Proving the graph-surgery recipe as a runtime-level spike (05-01) *before* building the tool (05-02) meant the tool was a transcription of a proven recipe, not an experiment
- Tool-core-as-header pattern (`sgfp4_inject_core.hpp` + `sgfp4_inject::run`): one implementation shared by the CLI binary and three test files in-process, no subprocess, no re-implementation
- Stale-artifact seeding in every malformed probe — made the atomicity (D-11) guarantee regression-tested rather than asserted

### What Was Inefficient
- Phase 5's formal verification artifact was never generated at phase close (UAT existed, VERIFICATION.md + summary frontmatter did not) — recurred from v1.0 Phase 2 despite that lesson; cost an audit-gate failure and a consolidation pass at milestone close
- README over-promised failure semantics before arg-stage cleanup was covered (W-2) — docs written one plan ahead of the code's actual guarantees
- Test helper duplication across three suite files invited a real convention drift bug (W-1: absolute vs. region-relative offsets) that was only caught because Phase 7 happened to re-derive the encoder's convention

### Patterns Established
- Byte-level container gating (version/magic in the container's own bytes) over manifest-field gating — manifests can lie about format lineage
- Fixture authoring from the REAL upstream encoder with frozen C-array output + provenance header + deterministic regeneration assert — cross-repo dependency exists only at authoring time
- Negative-probe tables as suites: mutation-kind loop + stale-artifact seeding + exit-code AND file-absence assertions per probe

### Key Lessons
1. Same as v1.0 Lesson 1, now verified twice: write the phase's VERIFICATION.md and flip requirement checkboxes **in the phase's own close-out** — the "UAT exists so it's fine" half-measure still fails the formal audit gate
2. One container/fixture builder per codebase: when two tests hand-roll the same binary format, one will drift (W-1) — extract the shared builder the moment the second consumer appears
3. A verification gap that *has* runtime evidence closes in minutes (consolidation from existing UAT); one that doesn't closes in hours — always run verify-work in-phase even if the audit feels distant

### Cost Observations
- Sessions: ~3 across 2026-08-26 → 2026-08-28 (restructure → 3 phases → audit/close)
- Notable: all 7 plans ran on Windows/MSVC `.build` despite plan shells assuming MSYS2 — exit-code/PASS semantics proved equivalent, no re-planning needed

---

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
| v2.0 | ~3 | 3 | Injection-tool restructure at 0% execution front-loaded a real artifact; tool-core-as-header; real-encoder fixture authoring |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|--------------------|
| v1.0 | 14 CPU/Vulkan parity fixtures + uniform/mixed round-trip suites | Both code modes × all 5 uniform layouts + LAYOUT_MIXED | Reference Python encoder (`encode_sgfp4.py`), no new runtime deps |
| v2.0 | 9 `op/sgfp4/` suites incl. classic-API parity + 13-probe malformed matrix | Express + classic API; single- & multi-tensor; uniform + structured MIXED | Vendored SHA-256 header, no new runtime deps |

### Top Lessons (Verified Across Milestones)

1. Close out each phase's verification before moving to the next — deferred verification compounds into milestone-close overhead *(verified in both v1.0 and v2.0 — v2.0's UAT-exists-but-no-VERIFICATION.md half-measure still failed the audit gate)*
2. Share binary-format builders across tests — duplicated hand-rolled builders drift (v2.0 W-1)
