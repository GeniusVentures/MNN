# Phase 3: Vulkan Decode — Uniform Layouts - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-24
**Phase:** 3-vulkan-decode-uniform-layouts
**Areas discussed:** Sidecar IO, Shader structure, Parity test, Output precision, GPU validation, Thread-locating strategy

---

## Sidecar IO

| Option | Description | Selected |
|--------|-------------|----------|
| Upload whole container | onResize reads sidecar via FileLoader + size probe (same as CPU), copies into a VulkanBuffer SSBO; shader reads raw container bytes; parity test identical to CPU test | ✓ |
| Device-local via sparse | Vulkan sparse-memory/file-backed extensions to map the sidecar; avoids a copy but adds extension deps and device variance — overkill for Phase 3 | |
| Stage parsed payloads | CPU parses framing, uploads only leaf payloads; less GPU bounds-checking but duplicates record-walk on CPU and diverges from "one traversal algorithm" goal | |

**User's choice:** Upload whole container
**Notes:** Parity test drives identical fixtures through both backends — same sidecar file, same descriptor.

---

## Shader structure

| Option | Description | Selected |
|--------|-------------|----------|
| Thread-per-weight | One linear thread per output weight; each thread re-walks framing to find its leaf; embarrassingly parallel, trivially matches CPU math | ✓ |
| Workgroup-per-record | Each workgroup owns a macroblock, walks records once in shared memory; more efficient but more complex — the structure Phase 4 heads toward anyway | |
| Mirror fp4_dequant | Flat precomputed-offset element mapping, but SGFP4 v2 needs the framing walk, so reduces to option 1 with extra indirection | |

**User's choice:** Thread-per-weight
**Notes:** Workgroup sizing is planner's discretion, following existing VulkanFP4Dequant conventions.

---

## Parity test

| Option | Description | Selected |
|--------|-------------|----------|
| Single dual-backend test | One C++ test decoding each fixture via dequant_sgfp4_container_cpu() AND via a Vulkan session, compared within float tolerance (checkVectorByRelativeError); graceful skip without a Vulkan device | ✓ |
| Fixtures vs both refs | CPU and Vulkan tests each compare against committed expected weights — indirect parity through golden data | |

**User's choice:** Single dual-backend test
**Notes:** Windows build/test machine has a Vulkan device; test still degrades gracefully elsewhere.

---

## Output precision

| Option | Description | Selected |
|--------|-------------|----------|
| FP16 default + FP32 | Mirror VulkanFP4Dequant D-04: useFP16() selects FP16-output variant, else FP32; two shader variants via makeshader.py | ✓ |
| FP32 only this phase | Single FP32 shader, safest parity target; defer FP16 to perf pass (SGV2-18) | |

**User's choice:** FP16 default + FP32
**Notes:** Shader naming follows backend convention (glsl_sgfp4_dequant_FP16_comp / glsl_sgfp4_dequant_comp).

---

## GPU validation

| Option | Description | Selected |
|--------|-------------|----------|
| Host pre-validates | Execution onResize validates full container once via existing header-only checks; shader indexes validated values; malformed container rejected pre-dispatch with CPU-matching error semantics | ✓ |
| Shader checks all | Shader bounds-checks every read and writes zero/skips on OOB; defense-in-depth but duplicates validation in GLSL and forces an OOB output convention | |
| Host + shader clamps | Host validates structure AND shader keeps cheap clamps on indexed loads; strongest posture, slight complexity increase | |

**User's choice:** Host pre-validates
**Notes:** No defined-OOB-output convention needed; only validated containers are uploaded and dispatched.

---

## Thread-locating strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Full re-walk/thread | Each thread redundantly re-walks framing (magic/version → offset table → record → leaf header → payload word); stateless, no barriers; redundant reads cached in L2 | ✓ |
| Prefix/index kernel | Two-kernel or single-kernel prefix pass computing record/leaf output offsets, then binary search; more code, marginal gain, Phase 4 rebuilds it for quadtree anyway | |
| Staged pre-walk | Workgroup 0 pre-walks into staging buffer with barrier; inter-workgroup sync complexity without clear need | |

**User's choice:** Full re-walk/thread
**Notes:** Deliberate trade: simplicity now over indexing machinery Phase 4 would have to rebuild for the quadtree walk.

---

## Claude's Discretion

- GLSL helper decomposition inside the .comp file(s), subject to makeshader.py pipeline constraints
- Workgroup size and dispatch arithmetic (named constants, no magic numbers)
- Host-side validation reuse: direct SGFP4DequantUtils.hpp calls vs. small Execution wrappers
- Test registration naming within the existing op/sgfp4/... namespace

## Deferred Ideas

None — discussion stayed within phase scope. (GPU perf tuning is SGV2-18 backlog; quadtree GPU walk is Phase 4 by design.)
