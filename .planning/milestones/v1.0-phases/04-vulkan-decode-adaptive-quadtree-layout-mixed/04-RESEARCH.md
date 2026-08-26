# Phase 4: Vulkan Decode — Adaptive Quadtree (LAYOUT_MIXED) - Research

**Researched:** 2026-08-25
**Domain:** GPU compute shader (GLSL/Vulkan) port of a CPU quadtree/bitmap walk; MNN buffer-backend Execution + shader-embedding pipeline; dual-backend numeric parity testing
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**GPU parallel strategy (SGV2-15)**
- D-01: Thread-per-weight walk — extends Phase-3 D-03/D-04 directly. Each thread re-walks the quadtree split-map (≤85 nodes, ≤12-byte/3-word bitmap) to locate its leaf, then decodes one weight via the existing shift-mask-FMA path. Stateless: no shared memory, no inter-thread sync, no separate kernel phases.
- D-02: SGV2-15's parenthetical "one workgroup per macroblock" is descriptive, not normative. The binding success criteria are (a) correct pre-order-DFS split-map walk with variable per-leaf-size decode on GPU, and (b) CPU/Vulkan parity within float tolerance. The verifier checks function, not dispatch shape.
- D-03: GLSL walk structure: bounded loop, no stack — fixed loop bound of 4 tree levels (64→32→16→8→4; nodes ≥8 carry split bits, 4×4 always leaf per spec §6.2). No dynamic stack arrays, no unbounded loops.
- D-04: Multi-record containers (B>1, e.g. the committed b3 fixture): keep the existing per-thread linear record scan of the offset table, accumulating consumed output elements. Walk the quadtree only when the owning record's layout enum is 4.

**Shader organization**
- D-05: Extend the single existing `sgfp4_dequant.comp` — `locateElement` gains the MIXED branch (enum 4) instead of returning false. FP16/FP32 variants remain generated from the single `.comp` via the `FLOAT` macro / `macro.json` (no new variant count). Regenerate and commit `AllShader.cpp` / `AllShader.h` / `VulkanShaderMap.cpp` via `makeshader.py`.
- D-06: `SGFP4DequantConst` stays unchanged (`outElementCount`, `containerBytes`). The shader derives everything else from container bytes via `read_u32_le`. No `VulkanSGFP4Dequant` C++ changes are expected beyond what the shader extension requires.

**Parity test + final sweep (SGV2-16, Success Criteria 1–3)**
- D-07: One full-sweep parity test — extend/rename the Phase-3 `op/sgfp4/vulkan_uniform_parity` test to iterate over ALL 14 committed fixtures in `test/op/SGFP4DequantFixtures.h`, each decoded via CPU reference AND Vulkan, compared with `checkVectorByRelativeError`. This single test satisfies Success Criteria 1–3 at once.
- D-08: Comparison uses `checkVectorByRelativeError` rtol 1e-4 with the Phase-3 graceful-skip convention when no Vulkan device is available.
- D-09: SC-1 walk correctness is proven transitively by weight parity — the golden pre-order-DFS traversal order is already independently verified on CPU (Phase-2 D-05 enumerator + golden-traversal test). No separate GPU structural traversal test.

**Host-side aux data**
- D-10: No aux data — stay stateless. Threads re-walk per D-01; no leaf-index SSBO, no host-emitted per-record leaf table, no ConstBuffer additions. All indexing machinery is deferred to SGV2-18 GPU-perf backlog.

### Claude's Discretion
- GLSL helper decomposition inside `sgfp4_dequant.comp` for the MIXED branch (helper function vs. inline in `locateElement`).
- Loop/branch arrangement within the bounded (≤4-level) walk, provided no unbounded loops and no stacks.
- Exact test-file naming and registration (reuse the Phase-3 test file vs. rename it), within the existing `op/sgfp4/...` namespace.
- Handling of the known `test/op/FP4ModelTest.cpp` build blocker during verification (Phase-1 `deferred-items.md` temporary-local-stub workaround, never committed).

### Deferred Ideas (OUT OF SCOPE)
None new this phase. Standing deferrals carried for visibility:
- GPU perf / indexing machinery (leaf-table SSBO, record base tables, workgroup-per-macroblock shape, coalescing) — SGV2-18 v2 backlog (D-10).
- E2E model integration (SGV2-17), encoder-side/benchmark work, and the `test/op/FP4ModelTest.cpp` fix — owned elsewhere (`milestone` workstream Phase 4 plan 04-02).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-------------------|
| SGV2-15 | Vulkan shader extended to walk the LAYOUT_MIXED split-map and decode variable per-leaf-size records on GPU (one workgroup per macroblock — descriptive per D-02) | §"Code Examples" gives the exact current `locateElement` state and the CPU walker it must mirror; §"Architecture Patterns" gives the bounded-4-level GLSL loop translation; §"Common Pitfalls" covers GLSL shift/keyword/type traps specific to this port |
| SGV2-16 | CPU/Vulkan decode-parity test for mixed/adaptive containers within float tolerance, passing via `./run_test.out` | §"Validation Architecture" and §"Architecture Patterns → Pattern 4" give the exact one-line D-07 change to `SGFP4VulkanDequantTest.cpp` (delete the `layout == kSGFP4LayoutMixed` skip) plus rename/message updates |
</phase_requirements>

## Summary

This phase is a narrow, well-scoped GLSL port: the CPU-side quadtree walk in `include/MNN/SGFP4DequantUtils.hpp::sgfp4_walk_quadtree` (an iterative, fixed-size-stack pre-order DFS over a ≤85-bit / 3-word split-map) needs a GLSL-legal, stateless, bounded-loop equivalent inside `locateElement()` in `source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp`. The uniform-layout branch already present in `locateElement` establishes every reusable piece: `read_u32_le()` for framing reads, `unpackLeafHeader()` via `unpackHalf2x16` for FP16 params, `codeMode0`/`codeMode1` for dual-mode symbol decode, and the sequential per-thread leaf-payload-cursor walk for multi-leaf/multi-record containers (D-04). The MIXED branch is additive to this same function — it only needs to compute, per output index, which flat leaf (in traversal order) owns it and that leaf's edge size `n`, then fall into the *same* payload-cursor/decode tail the uniform branch already uses.

The critical translation risk is recursion: `sgfp4_walk_quadtree` uses an explicit `QuadNode stack[85]` and unbounded-looking `while (top > 0)` loop, which is legal C++ but is exactly the pattern D-03 forbids in GLSL (no dynamic stack, no unbounded loop). Because the CPU walker was *designed* for this port (Phase-2 D-01 says so explicicitly — "the identical algorithm ports to the Phase 4 GLSL shader"), the fix is not a new algorithm but a re-expression: since the tree has a hard-known max depth of 4 (edge 64→32→16→8→4) and pre-order-DFS with fixed quadrant order (TL/TR/BL/BR) means a leaf's ancestry is uniquely determined by a 4-symbol path (one 2-bit quadrant choice per level) once you know which levels split, the GPU walk can determine "does this output element's leaf split at level L" via *linear leaf-count occupancy*, not via a stack — walk down from the root exactly 4 bounded iterations, at each iteration reading one split bit (word `node>>5`, bit `node&31`, mirroring the CPU reader's own masking, which is also GLSL-shift-UB-safe per the GLSL ES spec) and deciding whether the current node contains the target flat-leaf-index or has already been fully consumed, without ever storing more than the current node's (x, y, n) and a leaf-counter accumulator. This is a strict `for (level = 0; level < 4; ++level)` bound — exactly D-03's requirement — and requires no stack because "descend into the child that contains leaf index K" is a single decision per level, not a branch-and-backtrack search.

The shader-embedding pipeline (`makeshader.py` → `AllShader.cpp`/`AllShader.h`/`VulkanShaderMap.cpp`) and the parity-test harness (`SGFP4VulkanDequantTest.cpp`, `op/sgfp4/vulkan_uniform_parity`) are both already fully built and exercised by Phase 3 — this phase reuses both mechanically rather than designing them. The single fixture that exercises true LAYOUT_MIXED (`mixed_asymmetric`, layout enum 4) is already committed in `test/op/SGFP4DequantFixtures.h` and is the one fixture Phase 3's test explicitly skips (`if (fixture.layout == MNN::kSGFP4LayoutMixed) continue;`); D-07's "full sweep" is executed by deleting that skip.

**Primary recommendation:** Add a GLSL helper (e.g. `bool locateMixedLeaf(uint recStart, uint leafFlatIndex, out uint leafByteOffset, out uint leafEdge)` or equivalent) that performs the bounded 4-level descent over the record's 3-word split map to resolve a flat traversal-order leaf index to its `(x, y, n)` — call it from a new `else if (layoutEnum == 4u)` branch inside `locateElement`'s per-record loop, reusing the exact same block-header/payload-cursor tail the uniform branch already has (generalized to accept a per-leaf `n` instead of the record-wide `n`). Then delete the one-line MIXED skip in `SGFP4VulkanDequantTest.cpp` and rename the registered test name to reflect the full sweep.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Split-map bit parsing (which nodes split) | GPU Compute (Vulkan GLSL shader) | CPU decode core (`SGFP4DequantUtils.hpp`) | The shader must independently re-derive tree structure per-thread (D-01 stateless re-walk); the CPU core is the reference algorithm the GLSL must match bit-for-bit, not a runtime dependency |
| Leaf → flat-output-index mapping (pre-order DFS traversal order) | GPU Compute (Vulkan GLSL shader) | — | Owned entirely by `locateElement`'s per-thread walk; no host-computed index table exists (D-10 forbids one) |
| Per-leaf FP16 header unpack (S, bias, mode) | GPU Compute (Vulkan GLSL shader) | — | Already implemented via `unpackHalf2x16` in the uniform branch; MIXED branch reuses it unchanged once it has a leaf byte offset |
| Dual-mode symbol decode + affine reconstruct (`w = S*c + bias`) | GPU Compute (Vulkan GLSL shader) | — | Unchanged — `codeMode0`/`codeMode1` and the final FMA are layout-agnostic and already shared by all layouts |
| Container well-formedness validation (bounds, magic, tiling-sums-to-4096) | Host/CPU (`VulkanSGFP4Dequant` Execution creator, via `dequant_sgfp4_container_cpu`) | — | D-06: host pre-validation already fully decodes the container (including the MIXED branch, added in Phase 2) before any GPU dispatch; the shader trusts this and stays defensive-only (`idx >= outElementCount` guard) |
| Shader embedding / SPIR-V generation | Build tooling (`makeshader.py` + WSL glslang) | — | Mechanical regeneration step, not new design; Phase 3 already proved the pipeline against this exact `.comp` file |
| CPU/GPU parity verification | Test tier (`test/op/SGFP4VulkanDequantTest.cpp`, `./run_test.out`) | CPU decode core (oracle) | The CPU decoder is the ground truth the GPU output is checked against; no new oracle needed |

## Standard Stack

This phase adds no new external dependencies (no npm/pip/cargo packages). The "stack" is entirely first-party MNN infrastructure already in place from Phases 1–3:

### Core (already in the codebase — no installation needed)
| Component | Location | Purpose | Why Standard (for this repo) |
|-----------|----------|---------|-------------------------------|
| `sgfp4_dequant.comp` | `source/backend/vulkan/buffer/execution/glsl/` | The single shader this phase extends | D-05 locks extension into this file; GLSL 4.5-class compute shader, `#version`/`FLOAT` prepended by `makeshader.py` [VERIFIED: source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp] |
| `SGFP4DequantUtils.hpp` | `include/MNN/` | Normative CPU decode + quadtree walker (porting reference) | Header-only, already ported once (uniform layouts) [VERIFIED: include/MNN/SGFP4DequantUtils.hpp:141-282] |
| `makeshader.py` + `VulkanCodeGen.py` | `source/backend/vulkan/buffer/compiler/` | Embeds `.comp` → C++ byte arrays across FP32/FP16 variants | Locked pipeline per repo CLAUDE.md; exercised twice already in Phase 3 [VERIFIED: 03-01-SUMMARY.md, 03-02-SUMMARY.md] |
| glslangValidator 11:14.3.0 | WSL interop, symlinked from `thirdparty/build/Windows/Release/shaderc/bin` | GLSL→SPIR-V compilation used by `makeshader.py` | Already provisioned and confirmed reachable this session (`glslangValidator --version` → `11:14.3.0`) [VERIFIED: WSL shell probe this session] |
| `checkVectorByRelativeError<float>` | `test/TestUtils.h` (MNN test harness) | Float-tolerance vector comparison | Already the comparison primitive used by both Phase 3 parity passes [VERIFIED: test/op/SGFP4VulkanDequantTest.cpp:98] |

### Supporting
| Component | Purpose | When Used |
|-----------|---------|-----------|
| `MNN::Express::Module::load` + `Executor::RuntimeManager` | Runs the Vulkan op end-to-end through the production graph (not a raw pipeline invocation) | Parity test only — unchanged pattern from Phase 3 |
| `unpackHalf2x16` (GLSL built-in) | Exact FP16→FP32 leaf-header unpack | Already used by the uniform branch; MIXED leaves use the identical per-leaf header format, so this is reused verbatim, not reimplemented |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Extending the single `.comp` (D-05, locked) | A separate `sgfp4_dequant_mixed.comp` | Rejected by CONTEXT.md D-05 — would double the FP16/FP32 variant count and duplicate the framing-constant block; the shader's own header comment already anticipates the drop-in extension |
| Bounded 4-level loop (D-03, locked) | GLSL-side fixed-capacity stack array (e.g. `uint stack[8]`) mimicking the CPU structure literally | A stack-array port is *possible* in GLSL (arrays of known size are legal) but D-03 explicitly forbids "dynamic stack arrays" for this phase to hard-cap per-thread divergence cost; the level-bounded descent (see Architecture Patterns) achieves the same result without any array beyond a few scalars |

**Installation:** None — no new packages. If glslang/WSL interop is ever unavailable on a fresh machine, see Phase 3's `03-01-SUMMARY.md` for the exact provisioning recipe (reuse `thirdparty/build/Windows/Release/shaderc/bin` via WSL symlinks); this phase does not need to redo that provisioning, it was verified live this session.

**Version verification:** N/A — no package-manager-tracked dependency is introduced by this phase. The only "version" surface is glslang, confirmed live: `Glslang Version: 11:14.3.0` (WSL, this session), matching the version Phase 3 recorded.

## Package Legitimacy Audit

**Not applicable.** This phase installs no external packages (no `npm install`, `pip install`, or `cargo add`). All components used are either first-party MNN source files already committed by Phases 1–3, or the glslang toolchain already provisioned in Phase 3 (an existing, non-npm/pip binary reused via WSL interop, not a new registry dependency). No package-legitimacy check is required.

## Architecture Patterns

### System Architecture Diagram

```
                  ┌───────────────────────────────────────────┐
                  │  Host (C++) — VulkanSGFP4Dequant creator   │
                  │  (UNCHANGED this phase, D-06)               │
                  │                                             │
  sidecar file →  │  queryFileSize → bounded read → container  │
  {offset,size}   │  bytes[] --> dequant_sgfp4_container_cpu    │
                  │  (full CPU decode incl. MIXED walk, Phase2) │
                  │  ── pass ──► upload container SSBO,         │
                  │              build ConstBuffer{outElem,     │
                  │              containerBytes}, select        │
                  │              FP16/FP32 pipeline              │
                  └───────────────────┬─────────────────────────┘
                                      │ vkCmdDispatch(UP_DIV(elem,256),1,1)
                                      ▼
     ┌────────────────────────────────────────────────────────────────┐
     │  GPU — sgfp4_dequant.comp, one thread per output element idx   │
     │                                                                  │
     │  main(): idx >= outElementCount? → return (sole guard, D-06)   │
     │       │                                                          │
     │       ▼                                                          │
     │  locateElement(idx, ...)                                        │
     │       │  for each record b in [0,B):                            │
     │       │    layoutEnum = read_u32_le(recStart) & 0x7              │
     │       │    ┌─────────────────────┬───────────────────────────┐ │
     │       │    │ enum 0/1/2/3/5      │ enum 4 (THIS PHASE, NEW)  │ │
     │       │    │ (existing, Phase 3) │                            │ │
     │       │    │ Table-3 lookup →    │ read 3-word split map →   │ │
     │       │    │ fixed N,n           │ bounded 4-level descent   │ │
     │       │    │                     │ → resolves leaf n, flat   │ │
     │       │    │                     │   leaf index (D-03: no    │ │
     │       │    │                     │   stack, no unbounded     │ │
     │       │    │                     │   loop)                   │ │
     │       │    └─────────┬───────────┴─────────────┬─────────────┘ │
     │       │              └───────────┬──────────────┘               │
     │       │                          ▼                               │
     │       │        shared tail: block-header read → unpackLeafHeader │
     │       │        (unpackHalf2x16) → sequential payload-cursor walk │
     │       │        to this leaf's words (D-04, unchanged pattern)    │
     │       ▼                                                          │
     │  code = nibble/symbol extracted from read_u32_le(payloadWordByte)│
     │  c = codeMode0(code) or codeMode1(code)                          │
     │  Dst[idx] = FLOAT(S)*FLOAT(c) + FLOAT(bias)                      │
     └────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
              ┌───────────────────────────────────────────┐
              │ Test tier — SGFP4VulkanDequantTest.cpp     │
              │ (op/sgfp4/vulkan_uniform_parity → renamed) │
              │                                             │
              │ for ALL 14 fixtures (D-07, skip DELETED):  │
              │   cpuOut = dequant_sgfp4_container_cpu(...) │
              │   gpuOut = runSgfp4VulkanModule(...)        │
              │   checkVectorByRelativeError(gpuOut, cpuOut,│
              │                               rtol=1e-4)    │
              └───────────────────────────────────────────┘
```

### Recommended Project Structure

No new files/directories. Every touched path already exists:
```
include/MNN/SGFP4DequantUtils.hpp                                  # unchanged — porting reference only
source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp     # EDITED — MIXED branch in locateElement
source/backend/vulkan/buffer/compiler/AllShader.cpp                # REGENERATED (makeshader.py)
source/backend/vulkan/buffer/shaders/AllShader.h                   # REGENERATED (makeshader.py)
source/backend/vulkan/buffer/compiler/VulkanShaderMap.cpp          # REGENERATED (makeshader.py)
source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.{hpp,cpp}# UNCHANGED (D-06) — verify, don't edit
test/op/SGFP4VulkanDequantTest.cpp                                  # EDITED — delete MIXED skip, D-07 sweep
test/op/SGFP4DequantFixtures.h                                      # UNCHANGED — mixed_asymmetric already committed
```

### Pattern 1: Bounded 4-level descent replaces the CPU's stack-based DFS

**What:** The CPU walker (`sgfp4_walk_quadtree`) pushes/pops an explicit `QuadNode stack[85]` to do pre-order DFS, because it needs to *enumerate* all leaves for validation (tiling-sum check, leaf array fill). The GPU shader does not need enumeration — it needs to answer one question per thread: "for flat traversal-index K (this thread's leaf), what is this leaf's `(x, y, n)` and payload word offset?" That is a *root-to-leaf descent*, not a full-tree traversal, and a descent has a hard-known depth bound (4, since edge only ever halves 64→32→16→8→4 and node≥8 always tests a bit while node==4 never does — matching `kSGFP4QuadTreeMinSplitSize = 8` in the header [VERIFIED: include/MNN/SGFP4DequantUtils.hpp:69]).

**When to use:** Any time a bounded-depth tree needs point-query (not full-traversal) semantics in a shading language without recursion or dynamic-size local arrays.

**Core idea (translate before coding, then implement in GLSL):**
1. Maintain `x, y, n = 0, 0, 64u` and a running `leavesBefore = 0u` (count of leaves fully to the left of the target in traversal order) plus a `splitBitCursor` that tracks how many split-map bits have been consumed so far (mirrors `SGFP4SplitMapReader.bit` in the CPU reader).
2. Loop `for (uint level = 0u; level < 4u; ++level)`: if `n < 8u`, this node is a forced leaf (4×4 case) — stop descending. Otherwise read one split bit at `splitBitCursor` (`(map[splitBitCursor>>5] >> (splitBitCursor&31)) & 1u`, exactly mirroring the CPU reader's `words[bit>>5] >> (bit&31)` [VERIFIED: include/MNN/SGFP4DequantUtils.hpp:158], which is also safe under the GLSL ES shift-UB rule since `bit&31` is always in `[0,31)` — see Common Pitfalls), advance `splitBitCursor`. If the bit is 0 (leaf, no split) — stop descending, this is the target's leaf. If the bit is 1 (split) — the target leaf is under exactly one of the 4 children; you must determine *which* child and update `splitBitCursor` past the sibling subtrees' bits/leaves that come before it in TL/TR/BL/BR order, which requires knowing each sibling subtree's leaf/bit count — i.e., a bounded **subtree-size** computation, not a free traversal.
3. This means "point descent" alone (without subtree sizing) is **not** sufficient to skip past sibling subtrees' split bits and locate the correct child's sub-bitmap start — the encoding is a linear pre-order stream where a node's children's bit-and-leaf extents are only knowable by actually walking those subtrees. **Recommendation for the planner:** because subtree-size-skipping essentially reintroduces the same complexity as a bounded stack walk, the pragmatic GLSL-legal approach is a **bounded-size fixed local array standing in for the stack**, sized to the true worst case, which is small and static: since only 4 levels exist and pre-order DFS pop order is deterministic, the reachable "frontier" at any moment during a bounded walk needs at most 4 pending siblings (one per level) rather than the CPU's full 85-entry array. A `QuadNode frontier[4]` (a GLSL array of fixed compile-time size 4, indexed only by the loop counter, no arbitrary push/pop) is not a "dynamic stack" in the sense D-03 forbids (unbounded growth / recursion) — it is a compile-time-fixed 4-element local array walked with the same bounded `for` loop, and is the natural GLSL expression of "iterative, fixed-size-stack walker" that Phase-2 D-01 already named as the CPU pattern designed for this port. Confirm this reading against Phase-2 D-01's own wording ("iterative fixed-size-stack walker designed for GLSL portability") before over-engineering a stack-free variant — a `[4]`-sized array with a bounded loop satisfies D-03's actual constraint (no *unbounded* loop, no *dynamic* stack) while being a much smaller and more direct GLSL port of the existing, spec-designed CPU algorithm than a from-scratch subtree-sizing formula.

**Example (existing GLSL idioms this phase reuses verbatim):**
```glsl
// Source: source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp:55-64
uint read_u32_le(uint byteAddr) {
    uint wordIndex = byteAddr >> 2u;
    uint off = byteAddr & 3u;
    uint lo = Container[wordIndex];
    if (off == 0u) {
        return lo;
    }
    uint hi = Container[wordIndex + 1u];
    return (lo >> (8u * off)) | (hi << (32u - 8u * off));
}
```
```cpp
// Source: include/MNN/SGFP4DequantUtils.hpp:154-161 (the exact bit-read
// pattern to mirror in GLSL for the split-map — bit&31 keeps shift amount
// in [0,31), which is required for defined behavior per the GLSL ES spec)
bool next(bool& out) {
    if (bit >= kSGFP4MaxQuadTreeBits) {
        return false;
    }
    out = ((words[bit >> 5] >> (bit & 31)) & 1u) != 0;
    ++bit;
    return true;
}
```

### Pattern 2: MIXED branch shares the uniform branch's payload-cursor tail

**What:** In the current `locateElement`, once `(N, n)` are known for a uniform record, the code computes `leaf = local / (n*n)`, reads that leaf's header, then walks a **sequential per-leaf payload cursor** from `payloadsStart` up to the target leaf, summing each prior leaf's word count (`elems / kNibblesPerWord` or `elems / kSymbolsPerWord`) plus 16-byte alignment padding [VERIFIED: source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp:140-149]. This exact mechanism works unchanged for MIXED leaves — the only difference is that a MIXED leaf's `n` **varies per leaf** (from the split-map descent) instead of being the record-wide constant. The CPU decoder's own leaf-payload loop treats uniform and MIXED leaves identically past leaf-header unpack: `int n = isMixed ? leaves[leaf].n : leafEdge;` [VERIFIED: include/MNN/SGFP4DequantUtils.hpp:435] — this is the exact seam to mirror in GLSL: generalize the payload-cursor loop to read each prior leaf's `n` (from the same bounded descent, applied leaf-by-leaf) rather than assuming a constant `n`.

**When to use:** Whenever adding the MIXED branch — do not write a parallel/duplicate payload-walking loop; parametrize the existing one by per-leaf `n`.

**Example:**
```cpp
// Source: include/MNN/SGFP4DequantUtils.hpp:427-451 (CPU reference —
// this loop structure, generalized for per-leaf n, is what the GLSL
// payload-cursor code already does for uniform layouts and should
// continue to do for MIXED, just with `n` read per-leaf instead of fixed)
for (int leaf = 0; leaf < leafCount; ++leaf) {
    uint32_t header = sgfp4_read_u32_le(blockHeaders + leaf * 4);
    unpack_leaf_header(header, S, bias, mode);
    int n = isMixed ? leaves[leaf].n : leafEdge;
    int elementCount = n * n;
    int wordsPerLeaf = (mode == 0) ? (elementCount / kSGFP4NibblesPerWord)
                                    : (elementCount / kSGFP4SymbolsPerWord);
    // ... decode, then payloadCursor += align16(wordsPerLeaf * 4)
}
```

### Pattern 3: `layout` is a reserved GLSL keyword — already worked around

**What:** Phase 3 discovered mid-port that `layout` cannot be used as a local variable name in GLSL (it's a reserved keyword for the `layout(...)` qualifier syntax) and renamed the local to `layoutEnum` [VERIFIED: 03-02-SUMMARY.md deviation 2; source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp:104]. This phase's new code must keep using `layoutEnum` (already in scope in `locateElement`) and must not reintroduce `layout` as an identifier anywhere in the new branch or helper function.

**When to use:** Immediately — any new local variables in the MIXED branch should avoid other GLSL reserved words too (e.g. `in`, `out`, `flat`, `smooth`, `sample`, `patch`, `centroid`, `precise`, `invariant`, `coherent`, `buffer` are all reserved; the existing code already avoids these).

### Pattern 4: D-07 sweep is a one-line deletion + rename, not a rewrite

**What:** `SGFP4VulkanDequantTest.cpp`'s fixture loop already iterates `sgfp4_fixtures::kFixtures[0..kFixtureCount)` and only skips MIXED via one `if` [VERIFIED: test/op/SGFP4VulkanDequantTest.cpp:133-135]. The 13-fixture count logged in Phase 3 (`"%d uniform fixtures matched..."`) plus the single `mixed_asymmetric` (layout enum 4, the only fixture where `layout == kSGFP4LayoutMixed`) sum to exactly the 14 total fixtures [VERIFIED: test/op/SGFP4DequantFixtures.h:123-138, grep of `kFixtures[]` array]. No new fixtures are needed.

**When to use:** Implementing D-07. Delete the skip block (lines 128-135 of the current file), update the `MNN_PRINT` summary message and (per Claude's Discretion) optionally rename the `MNNTestSuiteRegister` string away from `"vulkan_uniform_parity"` to something reflecting the full sweep (e.g. `"op/sgfp4/vulkan_parity"` or keep the name and just widen the docstring — either is acceptable per CONTEXT.md discretion).

### Anti-Patterns to Avoid
- **Writing a separate quadtree-only shader file:** Rejected by D-05 — doubles variant count, breaks the single-`locateElement`-dispatch mental model the shader's own header comment sets up.
- **Porting `sgfp4_walk_quadtree`'s `stack[85]` literally as a GLSL array:** An 85-entry per-thread local array is legal GLSL but grossly oversized relative to the true bound (max 4 pending frontier entries) and works against D-03's spirit (hard-cap divergence/resource cost); use the bounded 4-level frontier instead (Pattern 1).
- **Adding a leaf-index/aux SSBO "to make the walk simpler":** Explicitly forbidden by D-10 this phase — even though host pre-validation already computes the full leaf table for validation, do not upload it. That's SGV2-18 backlog territory.
- **Left/right-shifting a `uint` by an amount computed without masking to `[0,31]`:** GLSL ES spec: "the result is undefined if... [the shift amount is] greater than or equal to the number of bits in the left expression's base type" [CITED: GLSL ES specification, via wiki.sei.cmu.edu CERT C/GLSL secure-coding reference]. The existing `read_u32_le`'s `32u - 8u*off` (off∈{1,2,3}, so shift∈{8,16,24}) and the CPU split-map reader's `bit & 31` are both already safe; any new shift expression in the MIXED branch must be similarly bounded — never shift a 32-bit value by a raw bit-index without `& 31u` first.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FP16 leaf-header unpack | A hand-rolled bit-twiddling half→float decoder in the new MIXED path | The existing `unpackLeafHeader()` GLSL function (built on `unpackHalf2x16`) | Already correct and shared by the uniform branch; the MIXED leaf header format is byte-identical (spec §6.2 Eq. 6) — there is nothing MIXED-specific about header unpack, only about *how you locate* the header bytes |
| Dual-mode symbol→float decode | A new decode function for MIXED payloads | `codeMode0()` / `codeMode1()`, already parametrized only by the extracted code/symbol value, agnostic to layout | Same math for every layout per spec §4.3; only the payload *addressing* differs by layout, never the decode arithmetic |
| Split-map bit reading | Any GLSL-side reimplementation of bit extraction that doesn't mirror `SGFP4SplitMapReader::next()` bit-for-bit | The identical `(words[bit>>5] >> (bit&31)) & 1u` formula, applied to `Container[]` words read via `read_u32_le` at the split-map's byte offset | A subtly different bit-order or word-order in the GLSL walk vs. the CPU walk is exactly the kind of silent mismatch that would only surface as sporadic decode divergence on `mixed_asymmetric`, not a compile error — must be pixel-for-pixel identical to the spec-verified CPU reference |
| Tree-tiling validation (leaf areas sum to 4096) | Any GPU-side validation of the split map | None — this is intentionally host-only (D-06's "shader stays defensive-only," Phase-2's `dequant_sgfp4_container_cpu` already does the full tiling check before any GPU dispatch happens) | Re-validating on GPU would be pure waste: an invalid container never reaches the shader, since `VulkanSGFP4Dequant`'s creator returns `nullptr` (blocking Execution construction, and therefore dispatch) on any `dequant_sgfp4_container_cpu` failure [VERIFIED: source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp:188-197] |

**Key insight:** Every primitive this phase needs below the tree-navigation logic itself (FP16 unpack, dual-mode decode, container bounds/validation) was already built and is already shared infrastructure. The only genuinely new code is the bounded split-map descent that resolves a flat output index to a leaf's `(offset, n)` — keep that function small, isolated, and directly comparable line-by-line against `sgfp4_walk_quadtree`/`SGFP4SplitMapReader` so a reviewer can verify the port by inspection rather than only by test pass/fail.

## Common Pitfalls

### Pitfall 1: GLSL has no character literals — reuse the assembled-hex pattern
**What goes wrong:** Writing `uint('S')` or similar to construct constants fails GLSL compilation.
**Why it happens:** GLSL (unlike C/C++) has no character-literal syntax.
**How to avoid:** This phase shouldn't need new magic/ASCII constants (kMagic etc. already exist), but if any new named constant is added, follow the existing pattern: an assembled numeric literal with a derivation comment, exactly as `kMagic = 0x34464753u // 'S'|'G'<<8|'F'<<16|'4'<<24` already does [VERIFIED: sgfp4_dequant.comp:30; 03-02-SUMMARY.md deviation 1].
**Warning signs:** glslangValidator compile errors mentioning character/string literals during `makeshader.py` regeneration.

### Pitfall 2: `layout` is a reserved keyword (already fixed once — don't reintroduce)
**What goes wrong:** Using `layout` as a local/parameter name fails compilation with a syntax error at the `layout(...)` qualifier grammar.
**Why it happens:** GLSL reserves `layout` for its own qualifier syntax; it is not merely discouraged, it is a hard parse error.
**How to avoid:** Continue using the existing `layoutEnum` identifier; extend, don't shadow, it in the new branch.
**Warning signs:** glslangValidator compile errors near the enum-dispatch `if/else` chain.

### Pitfall 3: `makeshader.py` exits 0 even when glslang fails — never trust the exit code
**What goes wrong:** A broken shader can be silently "regenerated" with stale/garbage SPIR-V embedded, and the regeneration script still reports success.
**Why it happens:** `makeshader.py` catches glslang exceptions internally, prints a traceback, and continues rather than propagating a non-zero exit [VERIFIED: 03-01-SUMMARY.md deviation 4].
**How to avoid:** After every `python3 makeshader.py` run, grep its captured stdout/stderr log for `error` (case-sensitive per Phase 3's convention) rather than checking `$?`. Phase 3 did this successfully — carry the same verification step forward.
**Warning signs:** `AllShader.cpp`/`.h` diffs that look suspiciously small/unchanged despite a real `.comp` edit, or a shader key that fails to bind at runtime with a cryptic pipeline-creation error.

### Pitfall 4: `test/CMakeLists.txt`'s `GLOB_RECURSE` is configure-time, not build-time
**What goes wrong:** Editing an existing, already-globbed test `.cpp` file needs no reconfigure, but if a *new* test file were added, a stale build would silently report `passed:0` for it (test never registered) rather than failing loudly.
**Why it happens:** CMake's `GLOB_RECURSE` snapshots the file list at `cmake configure` time; adding a file requires a fresh `cmake` configure step before the next build picks it up [VERIFIED: 03-04-SUMMARY.md deviation 1].
**How to avoid:** This phase most likely only *edits* `SGFP4VulkanDequantTest.cpp` (no new file, per D-07's "extend/rename" framing) so this may not apply — but if the planner chooses to rename to a new file rather than edit in place, re-run `cmake` configure before building/testing.
**Warning signs:** `run_test.out` runs clean but the target test name never appears in its output.

### Pitfall 5: WSL `find` ordering churns the regenerated artifacts (harmless but large diffs)
**What goes wrong:** Regenerating `AllShader.cpp`/`AllShader.h`/`VulkanShaderMap.cpp` from WSL reorders ~139k lines of pre-existing, unrelated shader entries because WSL's filesystem enumeration order differs from whatever environment originally produced the committed files.
**Why it happens:** `makeshader.py`/`VulkanCodeGen.py` iterate the shader directory via OS-level directory listing, which is not guaranteed stable across platforms/filesystems.
**How to avoid:** Expect this churn; it is content-equivalent (same shader bodies, different array order) and was already accepted once via checkpoint in Phase 3 [VERIFIED: 03-01-SUMMARY.md deviation 3, 03-02-SUMMARY.md deviation 3]. The planner should schedule a `checkpoint:human-verify` (or reuse the precedent as justification to skip re-asking) before committing the regenerated artifacts, and should diff-review that only the sgfp4-related entries carry semantic changes.
**Warning signs:** A three-file diff of tens of thousands of lines when only one `.comp` file was hand-edited — this is expected, not a bug, but must be sanity-checked (grep counts, not full diff review) per the Phase-3 precedent (`AllShader.cpp`=N, `AllShader.h`=N, `VulkanShaderMap.cpp`=N occurrences of the shader key).

### Pitfall 6: Unmasked bit-shift amounts are undefined behavior in GLSL, not just "implementation-defined"
**What goes wrong:** Computing a shift amount that can reach ≥32 for a `uint` (e.g. via an off-by-one in the split-map bit-index arithmetic) produces undefined results — potentially different per GPU vendor/driver, making a bug that "happens to work" on the dev RTX 4070 Ti Super but breaks elsewhere.
**Why it happens:** Per the GLSL ES specification, shift results are undefined "if the right operand is negative, or greater than or equal to the number of bits in the left expression's base type" [CITED: GLSL ES specification, per wiki.sei.cmu.edu secure-coding reference for GLSL shift operators].
**How to avoid:** Every new shift in the MIXED branch must mask the shift amount into `[0,31]` first — mirror the CPU's `bit & 31` and the existing `read_u32_le`'s bounded `8u*off` (off∈[0,3]) patterns exactly; never shift by a raw, unmasked bit-index or level counter.
**Warning signs:** Vulkan validation-layer warnings are unlikely to catch this (it's a value-correctness issue, not an API-misuse one) — the only practical detection is the parity test itself (`mixed_asymmetric` decoding to wrong values on some runs/devices) or careful code review against the masking pattern above.

### Pitfall 7: The pre-existing `test/op/FP4ModelTest.cpp` build blocker still applies
**What goes wrong:** A from-scratch `run_test.out` build fails to compile because of dead/malformed code in an unrelated, pre-existing test file from the `milestone` workstream.
**Why it happens:** Documented and unfixed since Phase 1 (`deferred-items.md`); still open per STATE.md's pending todos as of this session.
**How to avoid:** Use the same Phase-1/2/3 workaround: temporarily stub the file's contents, build + run the filtered test suite (e.g. `op/sgfp4/`, `op/vulkan/`), restore the file byte-for-byte via `git diff --exit-code` before any commit, and never commit the stub. This is Claude's-Discretion territory per CONTEXT.md — plan a verification step that follows this exact recipe.
**Warning signs:** MSVC errors `C2065`/`C3536`/`C2059` referencing undeclared identifiers `pi`, `sc`, `refVec`, `outSz` inside `FP4ModelConversionTest::run` when attempting a clean build.

## Code Examples

### Current `locateElement` enum dispatch (the exact insertion point)
```glsl
// Source: source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp:105-123
uint N;
uint n;
if (layoutEnum == 0u) {
    N = 1u; n = 64u;
} else if (layoutEnum == 1u) {
    N = 4u; n = 32u;
} else if (layoutEnum == 2u) {
    N = 16u; n = 16u;
} else if (layoutEnum == 3u) {
    N = 64u; n = 8u;
} else if (layoutEnum == 5u) {
    N = 256u; n = 4u;
} else {
    // enum 4 = LAYOUT_MIXED (quadtree — later phase) and >= 6 are
    // invalid; host pre-validation rejects them before dispatch.
    return false;
}
```
This is the exact `else` this phase turns into `else if (layoutEnum == 4u) { ... bounded descent ... }`, keeping the final `else { return false; }` for enum ≥6 (still invalid, still defensive-only per D-06).

### CPU split-map bit reader — the bit-for-bit reference for the GLSL descent
```cpp
// Source: include/MNN/SGFP4DequantUtils.hpp:148-162
struct SGFP4SplitMapReader {
    const uint32_t* words;
    int bit;
    explicit SGFP4SplitMapReader(const uint32_t* w) : words(w), bit(0) {}
    bool next(bool& out) {
        if (bit >= kSGFP4MaxQuadTreeBits) {
            return false;
        }
        out = ((words[bit >> 5] >> (bit & 31)) & 1u) != 0;
        ++bit;
        return true;
    }
};
```

### CPU quadtree walker — algorithm this phase's GLSL descent must be equivalent to
```cpp
// Source: include/MNN/SGFP4DequantUtils.hpp:188-222 (full function already
// read into context above — key structural facts for the GLSL port:
// root is {x:0, y:0, n:64}; push order is BR,BL,TR,TL so pop order is
// TL,TR,BL,BR (pre-order DFS); nodes with n < 8 never read a bit and are
// always leaves; nodes with n >= 8 read exactly one bit before deciding
// to split or become a leaf.
```

### GLSL uniform-branch payload-cursor tail — the code the MIXED branch falls into
```glsl
// Source: source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp:138-161
uint payloadCursor = payloadsStart;
for (uint l = 0u; l < leaf; ++l) {
    uint prevHeader = read_u32_le(blockHeadersStart + 4u * l);
    float prevS, prevBias;
    uint prevMode;
    unpackLeafHeader(prevHeader, prevS, prevBias, prevMode);
    uint elems = n * n;  // <-- for MIXED, this must become per-leaf n, not record n
    uint words = (prevMode == 0u) ? (elems / kNibblesPerWord) : (elems / kSymbolsPerWord);
    payloadCursor += (4u * words + kAlign16 - 1u) & ~(kAlign16 - 1u);
}
```

### Test loop skip to delete for D-07
```cpp
// Source: test/op/SGFP4VulkanDequantTest.cpp:126-135 — DELETE this filter
if (fixture.layout == MNN::kSGFP4LayoutMixed) {
    continue;
}
```

## State of the Art

| Old Approach (Phase 3, this codebase) | Current/Target Approach (Phase 4) | When Changed | Impact |
|--------------|------------------|---------------|--------|
| `locateElement` handles layout enums {0,1,2,3,5} only; enum 4 returns `false` (defensive reject, never reached because host pre-validation already excludes MIXED containers from GPU dispatch entirely — Phase 3 never even builds a MIXED sidecar for the GPU path) | `locateElement` also resolves enum 4 via a bounded split-map descent, producing the same `(payloadWordByte, inWordLane, laneBitWidth, S, bias)` tuple the uniform branches already produce | This phase (04) | Completes GPU/CPU feature parity — the shader can now decode 100% of what the CPU decoder (Phase 2) already handles, closing SGV2-15/16 |
| Parity test (`op/sgfp4/vulkan_uniform_parity`) explicitly skips the one MIXED fixture — 13/14 fixtures exercised on GPU | Same test (or a renamed variant) exercises all 14 fixtures, no skip | This phase (04) | SC-3 ("complete feature set decodes consistently on CPU and Vulkan") becomes literally true, not "true modulo MIXED" |

**Deprecated/outdated:** None — this phase is purely additive to the shader and test; nothing built in Phases 1-3 needs to change or be removed (confirmed by D-06's "no `VulkanSGFP4Dequant` C++ changes expected").

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The recommended GLSL translation strategy (a fixed 4-element "frontier" array walked in a bounded loop, rather than a literal 85-entry stack port) is *a* valid way to satisfy D-03's "no dynamic stack, no unbounded loop" constraint, but the exact array size/shape is this researcher's proposed synthesis, not something read verbatim from an existing GLSL file (no prior GLSL quadtree-descent code exists in this codebase to cite) | Architecture Patterns → Pattern 1 | If the planner/executor implements a naively different bound (e.g. genuinely needs more than 4 frontier slots because "point descent" turns out to require sibling-subtree-size bookkeeping the frontier can't hold), the loop bound or array size may need revisiting; validate the concrete design against `sgfp4_walk_quadtree`'s actual push/pop pattern during planning, and treat this section as a starting hypothesis to prototype against the `mixed_asymmetric` fixture early, not as a pre-verified algorithm |
| A2 | No additional GLSL bit-manipulation built-ins (e.g. `bitfieldExtract`, `findLSB`) beyond what's already used (`read_u32_le`'s shift/mask, `unpackHalf2x16`) are necessary or available-and-unused in this shader's GLSL version target | Standard Stack, Architecture Patterns | Low risk — the existing uniform branch already proves plain shift/mask suffices for this container format; if a cleaner built-in existed and mattered, Phase 3 would likely have used it already |

## Open Questions

1. **Exact shape of the bounded split-map descent algorithm**
   - What we know: the CPU reference (`sgfp4_walk_quadtree`) is authoritative and spec-correct; the GLSL port must produce bit-identical leaf resolution; D-03 mandates a bounded-loop, no-stack shape.
   - What's unclear: whether a 4-element "frontier" array (this research's proposed synthesis, A1) is sufficient without also tracking per-sibling bit-count offsets, or whether the cleanest correct GLSL expression instead computes, for each level, both "does the target's ancestor at this level split" AND "how many split-map bits/leaves are consumed by earlier same-level siblings" via a small closed-form recursion unrolled 4 times (since depth is fixed, a fully-unrolled 4-level `if` chain — not a `for` loop at all — may in fact be simpler and avoids any array).
   - Recommendation: the planner should prototype this against the single `mixed_asymmetric` fixture (already committed, with known expected weights) as the very first task of the phase, before investing in the full test-sweep plumbing — treat correct decode of that one fixture's every leaf as the go/no-go gate for the chosen algorithm shape, exactly as D-09 already implies (weight parity transitively proves walk correctness).

2. **Whether a fully-unrolled 4-level `if`/`else if` chain (no loop at all) is preferable to a `for (level<4)` loop**
   - What we know: D-03 requires "bounded loop, no stack... fixed loop bound of 4 tree levels" — the CONTEXT.md wording says "loop," implying a `for` construct is expected, but doesn't forbid full unrolling (which is loop-bound-1 in the limit).
   - What's unclear: which is more readable/maintainable in GLSL for this specific 4-level, non-recomputed-per-level structure.
   - Recommendation: this is Claude's Discretion per CONTEXT.md ("Loop/branch arrangement within the bounded ≤4-level walk") — either shape satisfies D-03's letter; prefer whichever the executor finds clearer to verify against the CPU reference during code review.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Vulkan-capable GPU + driver | Live GPU parity test execution | Yes — RTX 4070 Ti SUPER confirmed working in Phase 3 (`op/vulkan/fp4_dequant_correctness` passed) | Driver version not re-probed this session (unchanged since Phase 3) | Test has a built-in graceful skip (`MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN) == nullptr` → `return true`) — no fallback needed, test degrades gracefully |
| `.build` CMake configuration (`MNN_VULKAN=ON`, `MNN_VULKAN_IMAGE=OFF`) | Building `run_test.out` with the Vulkan buffer backend linked | Yes — confirmed present in `.build/CMakeCache.txt` this session | N/A (build config, not versioned) | None needed — already configured from Phase 3 |
| glslangValidator (WSL interop via `thirdparty/build/Windows/Release/shaderc/bin` symlinks) | `makeshader.py` regeneration | Yes — confirmed reachable this session (`glslangValidator --version` → `11:14.3.0` from WSL) | 11:14.3.0 | None needed — Phase 3's provisioning persists |
| `MNN_SUPPORT_TRANSFORMER_FUSE` build flag | `SGFP4VulkanDequantTest.cpp` is gated behind `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE` | Not re-probed this session; was enabled for Phase 1-3 builds per their SUMMARY files | — | If disabled, the whole test file compiles to nothing (not a failure, just excluded) — the planner should confirm this flag is still enabled in the build used for verification |

**Missing dependencies with no fallback:** None identified.

**Missing dependencies with fallback:** None currently missing — all previously-provisioned Phase 3 infrastructure was re-confirmed live this session.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | MNN's own `MNNTestSuite` (self-registering `MNNTestCase` subclasses, single monolithic `run_test.out` binary) |
| Config file | none — test files are auto-discovered via `test/CMakeLists.txt`'s `GLOB_RECURSE` (configure-time; see Pitfall 4) |
| Quick run command | `run_test.out.exe op/sgfp4/` (runs all `op/sgfp4/*` suites — CPU `uniform_decode`, `mixed_decode`, plus the Vulkan parity test) |
| Full suite command | `run_test.out.exe` (blocked by the pre-existing `FP4ModelTest.cpp` issue — requires the temp-stub workaround per Pitfall 7) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|--------------------|--------------|
| SGV2-15 | GLSL shader walks LAYOUT_MIXED split-map and decodes variable per-leaf-size records on GPU | unit (numeric, via parity) | `run_test.out.exe op/sgfp4/vulkan_uniform_parity` (or renamed test name) | ✅ (edit existing `test/op/SGFP4VulkanDequantTest.cpp`) |
| SGV2-16 | CPU/Vulkan decode-parity for mixed/adaptive containers within float tolerance | unit | same command, `mixed_asymmetric` fixture no longer skipped | ✅ (same file; fixture already committed in `SGFP4DequantFixtures.h`) |

### Sampling Rate
- **Per task commit:** `run_test.out.exe op/sgfp4/` (fast — 3 registered suites, seconds on GPU hardware)
- **Per wave merge:** same, plus `run_test.out.exe op/fp4` and `run_test.out.exe op/vulkan/fp4_dequant_correctness` (E2M1 additivity regression guard, unchanged from Phase 3's pattern)
- **Phase gate:** full `op/sgfp4/` green + shader-embedding artifact grep counts sane (per Pitfall 5) before `/gsd-verify-work`

### Wave 0 Gaps
None — existing test infrastructure (fixtures, CPU oracle, Vulkan parity harness, graceful-skip convention) fully covers this phase's requirements. No new test files, fixtures, or framework setup needed; the phase is a shader edit + a one-line test-loop change + mechanical shader-embedding regeneration.

### What parity alone does NOT cover
Per the research-focus's explicit ask, two things are outside what D-07's single full-sweep parity test can validate:
1. **The GLSL bounded-loop bound itself (D-03's "fixed loop bound of 4 tree levels").** The `mixed_asymmetric` fixture's actual tree depth is whatever the Phase-2 encoder produced for it (likely not exercising the full depth-4 worst case on every branch) — passing parity on this one fixture does not prove the bounded-loop implementation is correct for trees that use the full 4-level depth on every branch, or that it fails safely (rather than reading out-of-bounds / looping incorrectly) on a maximally-subdivided 85-bit tree. **Mitigation already in place:** the CPU-side golden-traversal test (Phase-2 D-05, independent enumerator) already validates full-depth traversal correctness on CPU with dedicated fixtures; nothing in this phase's D-09 reasoning claims the GPU walk was tested at full depth, only that *if* the GPU walk produces correct values for the fixtures it's given, the walk logic is presumed correct by extension of the already-verified CPU algorithm it mirrors. If the planner wants stronger coverage, consider requesting (as a phase task, not necessarily required by SGV2-15/16's literal text) that the Python encoder emit one additional fixture that forces full 4-level subdivision (a genuine 4×4-leaf-everywhere `mixed_allsplit`-style tree at the deepest level) — note `mixed_allsplit` already exists but per its fixture definition (layout enum 5, "FULL_4X4 collapse") it is a *uniform-collapsed* fixture, not a true MIXED-enum deep tree, so it does NOT exercise the GLSL MIXED branch at all.
2. **Malformed/adversarial split-maps beyond host pre-validation.** D-06 confirms the shader is intentionally defensive-only; host-side `dequant_sgfp4_container_cpu` (already fully exercises Phase 2's negative tests: split bit on a 4×4 node, >85-bit maps, non-tiling leaves, truncated payloads) is the sole gate against malformed containers ever reaching the GPU. This phase's parity test, by construction, only ever feeds the shader containers that already passed that CPU gate — so it provides zero coverage of "what does the shader do if somehow given a malformed MIXED container" (answer: undefined/unspecified, and per Roadmap Note 3, out of scope — attestation/adversarial-input GPU-side hardening is not this workstream's concern).

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V5 Input Validation | Yes — but entirely satisfied by existing, unchanged infrastructure | Host-side `dequant_sgfp4_container_cpu` (Phase 1/2, unchanged) remains the sole untrusted-input validation gate; the GPU shader continues to be defensive-only (`idx >= outElementCount` guard) per D-06 — this phase adds no new untrusted-input surface, since the shader only ever receives containers that already passed host validation |
| V6 Cryptography | No | Not applicable — no crypto in this phase |
| V2/V3/V4 (Auth/Session/Access Control) | No | Not applicable — this is an internal inference-engine decode path, not a network-facing service |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| Out-of-bounds SSBO read from a buggy split-map descent (e.g. computing a byte offset past `containerBytes`) | Tampering / Information Disclosure (reads adjacent GPU buffer memory) | Not newly mitigated by this phase's shader code (D-06 explicitly keeps the shader defensive-only) — the mitigation is structural: host pre-validation (`dequant_sgfp4_container_cpu`, unchanged) guarantees any container the shader receives is well-formed and fully tiles, so a *correct* GLSL port of the validated algorithm cannot read out of bounds. This makes correctness of the GLSL port itself (matching the CPU reference bit-for-bit) the operative security property for this phase — get the port right, rather than adding shader-side bounds checks (which D-06/D-10 both discourage as scope creep into SGV2-18 territory) |
| Undefined-behavior bit shifts producing platform-dependent wrong values (not a memory-safety issue, but a correctness/consistency one) | Tampering (in the sense of silent data corruption across GPU vendors) | Mask every shift amount into `[0,31]` before use, per Pitfall 6 |

## Sources

### Primary (HIGH confidence — direct codebase reads this session)
- `include/MNN/SGFP4DequantUtils.hpp` — CPU decode core, split-map constants, quadtree walker, leaf decode (full file read)
- `source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp` — current shader state (full file read)
- `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.{hpp,cpp}` — Execution class, host pre-validation gate (full files read)
- `test/op/SGFP4VulkanDequantTest.cpp` — Phase 3 parity test, full structure (full file read)
- `test/op/SGFP4DequantFixtures.h` — fixture struct layout and the 14-entry array (targeted reads/greps around the byte-array-free portions)
- `.planning/workstreams/sgfp4-pivot/phases/04-vulkan-decode-adaptive-quadtree-layout-mixed/04-CONTEXT.md` — locked decisions D-01..D-10
- `.planning/workstreams/sgfp4-pivot/phases/03-vulkan-decode-uniform-layouts/03-{01,02,04}-SUMMARY.md` — prior-phase pitfalls, tooling recipe, GPU hardware confirmation
- `.planning/workstreams/sgfp4-pivot/phases/02-adaptive-quadtree-layout-cpu-layout-mixed/02-CONTEXT.md` — CPU walker design intent (D-01: "designed for GLSL portability")
- `.planning/workstreams/sgfp4-pivot/phases/01-affine-dual-mode-decode-core-cpu-uniform-layouts/deferred-items.md` — FP4ModelTest.cpp build blocker
- `.planning/workstreams/sgfp4-pivot/{REQUIREMENTS,STATE,ROADMAP}.md` — requirement text, decision history, phase success criteria
- Live WSL probe this session — `glslangValidator --version` → `Glslang Version: 11:14.3.0`; `.build/CMakeCache.txt` → `MNN_VULKAN:BOOL=ON`, `MNN_VULKAN_IMAGE:BOOL=OFF`

### Secondary (MEDIUM confidence)
- GLSL ES specification shift-operator undefined-behavior rule — [CITED: wiki.sei.cmu.edu CERT secure-coding reference summarizing the GLSL ES spec]

### Tertiary (LOW confidence)
None — no unverified web-search-only claims were relied upon for the technical port strategy; the one open design question (exact frontier-array shape) is flagged explicitly in Assumptions Log / Open Questions rather than asserted as fact.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — everything is first-party code already read in full this session, no external packages
- Architecture (uniform-branch reuse, host/shader split): HIGH — read directly from the current shader and Execution class
- Architecture (bounded-descent algorithm shape for the new MIXED branch): MEDIUM — this is a researched synthesis/recommendation, not a pre-existing verified implementation; flagged explicitly in Assumptions Log A1 and Open Questions
- Pitfalls: HIGH — all sourced from Phase 3's own documented deviations on this exact shader/pipeline, not generic GLSL folklore
- Validation architecture: HIGH — test harness, fixture set, and gap analysis all directly read from committed files

**Research date:** 2026-08-25
**Valid until:** Effectively unbounded for the structural facts (internal, stable codebase) — but the GPU/toolchain environment probe (glslang reachability, `.build` config) should be re-verified if more than a few days elapse before planning executes, per this workstream's own precedent of environment drift between phases (WSL `find` ordering, CMake reconfigure requirements).
