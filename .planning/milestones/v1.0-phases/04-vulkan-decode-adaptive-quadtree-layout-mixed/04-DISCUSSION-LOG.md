# Phase 4: Vulkan Decode — Adaptive Quadtree (LAYOUT_MIXED) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-25
**Phase:** 4-vulkan-decode-adaptive-quadtree-layout-mixed
**Areas discussed:** GPU parallel strategy, Shader organization, Parity test + final sweep, Host-side aux data

---

## GPU parallel strategy

### Parallelization model

| Option | Description | Selected |
|--------|-------------|----------|
| Thread-per-weight walk | Each thread re-walks split-map (≤85 nodes, ≤12 dependent u32 reads) to locate its leaf then decodes one weight; stateless, no shared memory, no sync | ✓ |
| Workgroup-per-macroblock | One workgroup per macroblock; first phase walks tree into shared memory, then threads decode cooperatively; complex (barriers, shared-memory layout) | |
| Thread-per-leaf | One thread decodes an entire leaf serially; simplest indexing but terrible load balance | |

**User's choice:** Thread-per-weight walk
**Notes:** Extends Phase-3 D-03/D-04 directly; minimal shader diff.

### SGV2-15 "one workgroup per macroblock" wording

| Option | Description | Selected |
|--------|-------------|----------|
| Waive phrasing | "One workgroup per macroblock" is an implementation hint, not a success criterion; binding criteria are split-map walk correctness + CPU/Vulkan parity | ✓ |
| Honor literally | Implementation must demonstrably dispatch one workgroup per macroblock | |
| Amend requirement | Edit REQUIREMENTS.md SGV2-15 text to remove the parenthetical before planning | |

**User's choice:** Waive phrasing
**Notes:** Recorded explicitly so planner/verifier don't fight over requirement wording.

### Walk structure in GLSL

| Option | Description | Selected |
|--------|-------------|----------|
| Bounded walk, no stack | Fixed loop depth 4 (64→4); matches CPU fixed-size-stack discipline; no unbounded loops | ✓ |
| Explicit fixed stack | Port CPU walker 1:1 with fixed 85-entry stack array; easiest to eyeball-verify | |
| Recursive function | GLSL recursion — discouraged/undefined in compute shaders; Phase-2 D-01 rejected this | |

**User's choice:** Bounded walk, no stack

### Multi-record containers (B>1)

| Option | Description | Selected |
|--------|-------------|----------|
| Linear record scan | Keep existing per-thread offset-table scan, accumulate consumed outputs; walk quadtree only for enum-4 records; mirrors CPU | ✓ |
| Record base table | Host-precomputed aux SSBO with binary search; plumbing for few records | |

**User's choice:** Linear record scan

## Shader organization

### Shader file

| Option | Description | Selected |
|--------|-------------|----------|
| Extend single .comp | MIXED branch inside existing locateElement; one code path for all layouts; FP16/FP32 variants unchanged from FLOAT macro | ✓ |
| Separate mixed .comp | New shader + pipeline keys; Execution picks shader per record layout; doubles embedded variants (4 total) | |

**User's choice:** Extend single .comp
**Notes:** Shader header comment already anticipated ("a later phase replaces locateElement").

### Const buffer

| Option | Description | Selected |
|--------|-------------|----------|
| No const changes | Keep outElementCount + containerBytes; shader derives the rest from container bytes | ✓ |
| Extend const buffer | Add macroblockSize/flags word — container is already self-describing | |

**User's choice:** No const changes

## Parity test + final sweep

### Test shape

| Option | Description | Selected |
|--------|-------------|----------|
| One full-sweep test | Extend parity test to all 14 fixtures (uniform + mixed + b3) through both backends; covers SC 1–3 at once | ✓ |
| Separate mixed test | Keep uniform parity untouched as regression anchor; add op/sgfp4/vulkan_mixed_parity for the 3 mixed fixtures | |

**User's choice:** One full-sweep test

### Tolerance / no-device handling

| Option | Description | Selected |
|--------|-------------|----------|
| rtol 1e-4 + skip | checkVectorByRelativeError rtol 1e-4; graceful skip when no Vulkan device; Phase-3 convention | ✓ |
| Tighter tolerance | Smaller rtol or absolute floor for near-zero ternary weights | |

**User's choice:** rtol 1e-4 + skip

### SC-1 traversal proof

| Option | Description | Selected |
|--------|-------------|----------|
| Parity implies walk | Weight parity transitively proves walk order/geometry; golden traversal already proven on CPU (Phase-2 enumerator) | ✓ |
| Explicit quadrant assert | Additional per-leaf (x,y) coordinate checks via asymmetric fixture | |

**User's choice:** Parity implies walk

## Host-side aux data

### Aux SSBO decision

| Option | Description | Selected |
|--------|-------------|----------|
| No aux — defer to SGV2-18 | Keep shader stateless (D-A1 re-walk); all indexing machinery deferred to GPU-perf backlog | ✓ |
| Emit leaf-table SSBO | Host pre-validation already decodes; derive leaf table (origin + payload offset) so threads skip walks | |

**User's choice:** No aux — defer to SGV2-18
**Notes:** Phase 4 is deliberately the simplest correct GPU quadtree; nothing in this phase's criteria justifies the plumbing.

---

## Claude's Discretion

- GLSL helper decomposition for the MIXED branch (helper function vs. inline)
- Loop/branch arrangement within the bounded walk (no unbounded loops, no stacks)
- Test file naming/registration within `op/sgfp4/...` namespace
- Verification handling of the known `test/op/FP4ModelTest.cpp` build blocker (temporary-local-stub workaround)

## Deferred Ideas

None new. Standing: GPU perf/indexing machinery (SGV2-18), E2E integration (SGV2-17), FP4ModelTest.cpp fix (milestone workstream plan 04-02).
