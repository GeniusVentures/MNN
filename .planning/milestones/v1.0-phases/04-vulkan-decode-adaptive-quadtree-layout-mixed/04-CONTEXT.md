# Phase 4: Vulkan Decode — Adaptive Quadtree (LAYOUT_MIXED) - Context

**Gathered:** 2026-08-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend the Phase-3 Vulkan shader to walk the LAYOUT_MIXED quadtree split-map (pre-order DFS, TL/TR/BL/BR) and decode variable per-leaf-size records on GPU, completing GPU parity with the CPU reference across the complete SGFP4 v2 feature set. Requirements: SGV2-15, SGV2-16.

**Out of scope (locked by ROADMAP/REQUIREMENTS):** no GPU performance tuning / indexing machinery (SGV2-18 backlog — thread coalescing, workgroup shaping, aux index buffers), no E2M1 `VulkanFP4Dequant` changes (additive), no CPU decode changes, no FlatBuffers schema changes, no other backends (Metal/CUDA/OpenCL), no end-to-end model integration (SGV2-17 backlog).

</domain>

<decisions>
## Implementation Decisions

### GPU parallel strategy (SGV2-15)
- **D-01:** Thread-per-weight walk — extends Phase-3 D-03/D-04 directly. Each thread re-walks the quadtree split-map (≤85 nodes, ≤12-byte/3-word bitmap) to locate its leaf, then decodes one weight via the existing shift-mask-FMA path. Stateless: no shared memory, no inter-thread sync, no separate kernel phases.
- **D-02:** SGV2-15's parenthetical "one workgroup per macroblock" is **descriptive, not normative** — treat it as an implementation hint from the requirement draft. The binding success criteria are (a) correct pre-order-DFS split-map walk with variable per-leaf-size decode on GPU, and (b) CPU/Vulkan parity within float tolerance. The verifier checks function, not dispatch shape. (Recorded explicitly so the planner/verifier don't fight over requirement wording.)
- **D-03:** GLSL walk structure: **bounded loop, no stack** — fixed loop bound of 4 tree levels (64→32→16→8→4; nodes ≥8 carry split bits, 4×4 always leaf per spec §6.2). No dynamic stack arrays, no unbounded loops; hard-caps divergence cost. (Carries Phase-2 D-01's no-recursion discipline.)
  - **Resolved during planning (Plan 04-01):** the literal "4 tree levels" bound describes tree *depth*, not the loop trip count needed to locate a leaf. `SGFP4DequantUtils.hpp:297-303` confirms LAYOUT_MIXED output is leaf-major (leaves concatenated in pre-order-DFS traversal order), not a spatial (x,y) grid — so a GPU thread cannot resolve "which leaf owns flat index K" via a constant 4-step spatial descent; it must walk the full split-map in traversal order, accumulating consumed leaf area, until the target index falls in range. Plan 04-01 Task 1 implements this as a single bounded walk using a **compile-time-fixed 16-slot array** (true worst-case pending-sibling depth is 13) and a **compile-time-bounded 341-iteration loop cap** (85 max internal-node bit reads + 256 max leaves) — still a fixed, non-recursive, non-data-resized bound, satisfying D-03's underlying intent (no recursion, no dynamically-growable structures, hard-capped divergence cost) even though the concrete bound is larger than the literal "4." Independently verified by both the planner and gsd-plan-checker against the cited source lines.
- **D-04:** Multi-record containers (B>1, e.g. the committed b3 fixture): keep the existing per-thread linear record scan of the offset table, accumulating consumed output elements — the same structure the Phase-3 shader already uses for uniform records and that the CPU decode uses. Walk the quadtree only when the owning record's layout enum is 4.

### Shader organization
- **D-05:** **Extend the single existing `sgfp4_dequant.comp`** — `locateElement` gains the MIXED branch (enum 4) instead of returning false. One shader code path handles all layouts; the shader's own header comment anticipated this ("a later phase replaces locateElement"). FP16/FP32 variants remain generated from the single `.comp` via the `FLOAT` macro / `macro.json` (no new variant count). Regenerate and commit `AllShader.cpp` / `AllShader.h` / `VulkanShaderMap.cpp` via `makeshader.py` (locked roadmap note 5).
- **D-06:** `SGFP4DequantConst` stays **unchanged** (`outElementCount`, `containerBytes`). The shader derives everything else from container bytes via `read_u32_le`; the container is self-describing per spec §3. No `VulkanSGFP4Dequant` C++ changes are expected beyond what the shader extension requires (none anticipated — the Execution already dispatches `UP_DIV(elementCount, 256)` and pre-validates containers host-side per Phase-3 D-05, which covers MIXED validation via `dequant_sgfp4_container_cpu`).

### Parity test + final sweep (SGV2-16, Success Criteria 1–3)
- **D-07:** **One full-sweep parity test**: extend/rename the Phase-3 `op/sgfp4/vulkan_uniform_parity` test to iterate over ALL 14 committed fixtures in `test/op/SGFP4DequantFixtures.h` — both code modes × all 5 uniform layouts, `mixed_allsplit`, `uniform_collapse`, `mixed_asymmetric`, and the b3 multi-record fixture — each decoded via CPU reference AND Vulkan, compared with `checkVectorByRelativeError`. This single test satisfies Success Criteria 1–3 at once: SC-3's "complete feature set" sweep falls out because the uniform fixtures re-run through the now-complete shader.
- **D-08:** Comparison uses `checkVectorByRelativeError` rtol 1e-4 with the Phase-3 graceful-skip convention when no Vulkan device is available (skip + clear message). Consistent with roadmap note 3: float-tolerance parity, never byte-exactness.
- **D-09:** SC-1 walk correctness is **proven transitively by weight parity** — the golden pre-order-DFS traversal order is already independently verified on CPU (Phase-2 D-05 enumerator + golden-traversal test), and the GPU walk is value-driven per weight with no separate traversal state to inspect. Correct expected weights at every coordinate imply correct walk order/geometry. No separate GPU structural traversal test.

### Host-side aux data
- **D-10:** **No aux data — stay stateless.** Threads re-walk per D-01; no leaf-index SSBO, no host-emitted per-record leaf table, no ConstBuffer additions (D-06). All indexing machinery (leaf tables, record base tables for binary search, workgroup shaping) is deferred to SGV2-18 GPU-perf backlog alongside coalescing/shared-memory tuning. Phase 4 is deliberately the simplest correct GPU quadtree: host pre-validation already fully decodes the container for validation, so an aux table would be cheap, but nothing in this phase's criteria justifies the extra upload + shader-consumption plumbing.

### Claude's Discretion
- GLSL helper decomposition inside `sgfp4_dequant.comp` for the MIXED branch (helper function vs. inline in `locateElement`).
- Loop/branch arrangement within the bounded (≤4-level) walk, provided no unbounded loops and no stacks.
- Exact test-file naming and registration (reuse the Phase-3 test file vs. rename it), within the existing `op/sgfp4/...` namespace.
- Handling of the known `test/op/FP4ModelTest.cpp` build blocker during verification (Phase-1 `deferred-items.md` temporary-local-stub workaround, never committed).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### SGFP4 v2 specification
- `.planning/sgfp4-arxiv-v2.txt` §6.2 — split-map serialization (12-byte/3-word LE bitmap, pre-order DFS, TL/TR/BL/BR, nodes ≥8 carry bits, 4×4 always leaf, ≤85 bits), per-leaf headers, per-leaf payload word counts, 16-byte padding — the exact structure the GLSL walk must reproduce
- `.planning/sgfp4-arxiv-v2.txt` §6.1 — uniform layouts (already ported in Phase 3; the MIXED branch coexists in the same walk)
- `.planning/sgfp4-arxiv-v2.txt` §4.3–4.4 — dual-mode payload packing and affine reconstruction (already in the shader; unchanged)

### Workstream planning
- `.planning/workstreams/sgfp4-pivot/ROADMAP.md` — Phase 4 goal, success criteria 1–3, locked roadmap notes 1–6 (note 5: buffer backend + makeshader.py regeneration)
- `.planning/workstreams/sgfp4-pivot/REQUIREMENTS.md` — SGV2-15, SGV2-16 normative text; SGV2-17..21 explicitly v2 backlog (SGV2-18 = GPU perf tuning, where D-10's deferred machinery lives)
- `.planning/quick/260821-p1q-evaluate-current-fp4-ultra-fp4-implement/SGFP4-PIVOT-ANALYSIS.md` — full gap analysis and decision history

### Prior phase artifacts (contracts this phase builds on)
- `.planning/workstreams/sgfp4-pivot/phases/03-vulkan-decode-uniform-layouts/03-CONTEXT.md` — Phase-3 decisions D-01..D-07 this phase extends (esp. D-03/D-04 thread-per-weight re-walk, D-05 host pre-validation, D-06 FP16/FP32 variants)
- `.planning/workstreams/sgfp4-pivot/phases/02-adaptive-quadtree-layout-cpu-layout-mixed/02-CONTEXT.md` — Phase-2 decisions D-01 (iterative fixed-size-stack walker, designed for GLSL portability) and D-05 (independent golden-traversal enumerator)
- `.planning/workstreams/sgfp4-pivot/phases/01-affine-dual-mode-decode-core-cpu-uniform-layouts/deferred-items.md` — known `test/op/FP4ModelTest.cpp` build blocker (affects verification strategy)

### Key implementation references (scouted)
- `include/MNN/SGFP4DequantUtils.hpp` — normative CPU decode incl. the LAYOUT_MIXED walker and split-map constants; the GLSL porting reference
- `source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp` — the shader being extended; `locateElement` currently rejects enum 4; framing constants and `read_u32_le` already 1:1 with the header
- `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.{hpp,cpp}` — Execution with host sidecar load, pre-validation, SSBO upload; expected unchanged by this phase
- `test/op/SGFP4DequantFixtures.h` — all 14 fixtures incl. `mixed_allsplit` / `uniform_collapse` / `mixed_asymmetric` (with expected weights) the full-sweep test consumes
- `CLAUDE.md` (repo root) — makeshader.py regeneration contract for buffer-backend GLSL edits

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `sgfp4_dequant.comp` — the uniform shader already reads layout enums and walks the offset table per thread; MIXED slots into the existing `locateElement` enum dispatch (the `else` branch currently reserved for enum 4 / ≥6).
- `include/MNN/SGFP4DequantUtils.hpp` — the CPU LAYOUT_MIXED walker is the reference algorithm; its bounded traversal translates directly to the D-03 bounded-4-level GLSL loop.
- `test/op/SGFP4DequantFixtures.h` — 14 fixtures with embedded expected weights; no new fixture generation needed for parity (Phase 2 committed the mixed ones).
- `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp` — creator, host pre-validation (`dequant_sgfp4_container_cpu` into scratch), and dispatch already handle arbitrary containers; no changes anticipated (D-06).

### Established Patterns
- Shader pipeline: GLSL edit → `python3 source/backend/vulkan/buffer/compiler/makeshader.py` → commit regenerated `AllShader.cpp`, `AllShader.h`, `VulkanShaderMap.cpp` (locked roadmap note 5; `kSgfp4WorkgroupSize = 256` is named on both C++ and GLSL sides — keep them equal).
- Dual-backend parity test pattern from Phase 3 (`op/sgfp4/vulkan_uniform_parity`): fixtures → CPU reference vs Vulkan module → `checkVectorByRelativeError`, skip-with-message without a Vulkan device.
- Header-only decode core with named framing constants mirrored 1:1 in GLSL (`kMagic`, `kVersion`, `kOffsetTableStart`, etc.).

### Integration Points
- `locateElement` in `sgfp4_dequant.comp` — the single place the MIXED branch is added.
- Phase-3 parity test file (in `test/op/`, picked up via `test/CMakeLists.txt` GLOB_RECURSE) — extended to the full 14-fixture sweep (D-07).
- `makeshader.py` outputs — regenerated and committed with the shader change.

</code_context>

<specifics>
## Specific Ideas

- The shader's own header comment ("a later phase replaces locateElement") is the intended seam — the extension should read as completing that TODO, not as a rewrite.
- Walk bound: max tree depth is 4 (64→4); split bits only exist on nodes with edge ≥8; a leaf of edge n consumes n²/8 (mode 0) or n²/16 (mode 1) words, 16-byte aligned (SC, Phase-2 criterion 4).
- Keep the known build blocker in mind: `test/op/FP4ModelTest.cpp` prevents a from-scratch `run_test.out` build — verification steps use the Phase-1 temporary-local-stub workaround (never committed).
- Parity comparison uses float tolerance only (rtol 1e-4) — roadmap note 3 excludes byte-exactness infrastructure.
- `uniform_collapse` in the sweep doubles as a regression that the encoder's normative uniform-collapse (spec §6.3) still round-trips through the GPU path.

</specifics>

<deferred>
## Deferred Ideas

None new — discussion stayed within phase scope. Standing deferrals carried for visibility:
- GPU perf / indexing machinery (leaf-table SSBO, record base tables, workgroup-per-macroblock shape, coalescing) — SGV2-18 v2 backlog (D-10).
- E2E model integration (SGV2-17), encoder-side/benchmark work, and the `test/op/FP4ModelTest.cpp` fix — owned elsewhere (`milestone` workstream Phase 4 plan 04-02).

</deferred>

---

*Phase: 4-vulkan-decode-adaptive-quadtree-layout-mixed*
*Context gathered: 2026-08-25*
