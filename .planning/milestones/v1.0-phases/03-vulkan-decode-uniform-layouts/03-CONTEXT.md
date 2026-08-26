# Phase 3: Vulkan Decode — Uniform Layouts - Context

**Gathered:** 2026-08-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Port the CPU-validated uniform-layout SGFP4 v2 decode to the Vulkan buffer backend: a new GLSL compute shader (FP4_AFFINE + T158_AFFINE affine reconstruction via shift-mask-FMA) plus a new Vulkan Execution class reading the same `{magic, offset, size}` external-sidecar descriptor as the CPU path, with Vulkan output matching the CPU reference decode within float tolerance. Requirements: SGV2-12, SGV2-13, SGV2-14.

**Out of scope (locked by ROADMAP/REQUIREMENTS):** no quadtree/LAYOUT_MIXED GPU work (Phase 4 — though nothing in this phase's design may preclude it), no E2M1 `VulkanFP4Dequant` changes (additive), no CPU decode changes, no FlatBuffers schema changes, no GPU performance tuning beyond functional correctness (SGV2-18 is v2 backlog), no other backends (Metal/CUDA/OpenCL).

</domain>

<decisions>
## Implementation Decisions

### Sidecar / container IO (SGV2-13)
- **D-01:** Host-side upload of the whole container. The Vulkan Execution's `onResize` reads the container bytes from the external sidecar exactly like the CPU path (`FileLoader` + real `std::ifstream` file-size probe per Phase 1 T-01-04), then copies the bytes into a `VulkanBuffer` bound as an SSBO. The shader reads raw container bytes from that storage buffer. No sparse-memory/file-backed extensions, no CPU-side frame parsing split.
- **D-02:** The parity test drives the identical container fixtures through both backends — same sidecar file, same descriptor — so the GPU path is exercised through the same external-container plumbing as production, not through a synthetic in-graph tensor.

### Shader structure (SGV2-12)
- **D-03:** Thread-per-weight, stateless. One linear thread per output weight (mirrors `fp4_dequant.comp`'s element-parallel mapping). No workgroup-per-record staging, no inter-workgroup sync, no prefix/index kernel. Workgroup sizing (local_size_x, dispatch arithmetic) follows the existing `VulkanFP4Dequant` convention and is planner's discretion.
- **D-04:** Each thread performs the full framing re-walk (magic/version → offset table → its record → leaf header → payload word) redundantly to locate its weight. The re-walk is a few hundred cached uint32 reads per thread — negligible for uniform layouts, and it keeps the shader stateless. This is a deliberate trade: simplicity now over indexing machinery that Phase 4 would have to rebuild for the quadtree anyway.

### Validation posture (ASVS V5 carried to GPU)
- **D-05:** Host pre-validates. `onResize` validates the full container structure ONCE using the existing header-only checks in `SGFP4DequantUtils.hpp` (magic `'SGF4'`, version `0x02`, record-offset table bounds, record/leaf geometry, payload word counts) before any dispatch. Only a validated container is uploaded and dispatched. Malformed containers are rejected with the same error semantics as the CPU path (error return, no partial output writes). The shader does NOT duplicate bounds-checking — no defined-OOB-output convention is needed.

### Output precision
- **D-06:** FP16 default + FP32 variant, mirroring `VulkanFP4Dequant` D-04: `vkBn->useFP16()` selects the FP16-output shader variant, otherwise the FP32 variant. Shader naming follows the backend convention (`glsl_sgfp4_dequant_FP16_comp` / `glsl_sgfp4_dequant_comp`), and both variants are embedded via `makeshader.py` with regenerated `AllShader.cpp` / `AllShader.h` / `VulkanShaderMap.cpp` committed (locked roadmap note 5).

### Parity verification (SGV2-14)
- **D-07:** One dual-backend C++ test registered in `MNNTestSuite`: each uniform-layout fixture is decoded via `dequant_sgfp4_container_cpu()` AND via a Vulkan session through the new Execution, then compared with `checkVectorByRelativeError` within float tolerance. The test degrades gracefully (skip with clear message) when no Vulkan device is available; the Windows build/test machine has one.

### Claude's Discretion
- GLSL helper decomposition inside the `.comp` file(s) (single file with variants vs. shared include structure, subject to `makeshader.py` pipeline constraints).
- Workgroup size and dispatch arithmetic (constant, named — no magic numbers).
- C++ code organization of the Execution: whether container validation helpers are reused directly from `SGFP4DequantUtils.hpp` or wrapped in small host-side helpers in the Execution.
- Test registration naming within the existing `op/sgfp4/...` namespace.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### SGFP4 v2 specification
- `.planning/sgfp4-arxiv-v2.txt` §3 — v2 self-framed stream: magic `'SGF4'`, version `0x02`, record-offset table, per-record `sb_header`
- `.planning/sgfp4-arxiv-v2.txt` §4.3–4.4 — dual-mode payload packing and affine reconstruction math (`w = S·c + bias`) the shader must reproduce
- `.planning/sgfp4-arxiv-v2.txt` §6.1 — uniform layouts (LAYOUT_UNIFORM_64/32/16/8, LAYOUT_FULL_4x4), raster leaf order, per-leaf payload word counts

### Workstream planning
- `.planning/workstreams/sgfp4-pivot/ROADMAP.md` — Phase 3 goal, success criteria, locked roadmap notes 1–6 (note 5: Vulkan buffer backend + makeshader.py regeneration)
- `.planning/workstreams/sgfp4-pivot/REQUIREMENTS.md` — SGV2-12..14 normative text; SGV2-17..21 explicitly v2 backlog (no perf tuning here)
- `.planning/quick/260821-p1q-evaluate-current-fp4-ultra-fp4-implement/SGFP4-PIVOT-ANALYSIS.md` — full gap analysis and decision history

### Phase 1/2 artifacts (code and contracts this phase ports)
- `.planning/workstreams/sgfp4-pivot/phases/01-affine-dual-mode-decode-core-cpu-uniform-layouts/01-01-PLAN.md` — threat model (T-01-04 DoS bound, file-size probe), schema/plumbing decisions the Vulkan Execution must mirror
- `.planning/workstreams/sgfp4-pivot/phases/01-affine-dual-mode-decode-core-cpu-uniform-layouts/deferred-items.md` — known `test/op/FP4ModelTest.cpp` build blocker (affects full-suite verification strategy)
- `include/MNN/SGFP4DequantUtils.hpp` — normative CPU decode implementation being ported; all constants, FP16 header unpack, and container checks live here
- `CLAUDE.md` (repo root) — makeshader.py regeneration contract for buffer-backend GLSL edits

### Key implementation references (scouted)
- `source/backend/vulkan/buffer/execution/VulkanFP4Dequant.{hpp,cpp}` — execution-class pattern: const-buffer push, descriptor set layout, pipeline selection, FP16/FP32 variant choice (D-06 mirror), dispatch + barrierSource
- `source/backend/vulkan/buffer/execution/glsl/fp4_dequant.comp` — element-parallel shader pattern this phase mirrors
- `source/backend/cpu/CPUSGFP4Dequant.cpp` — external-sidecar loading pattern (FileLoader + ifstream size probe) to replicate in Vulkan `onResize`

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `include/MNN/SGFP4DequantUtils.hpp` — the CPU decode this phase ports. All framing/leaf checks reusable as-is on the host for D-05 pre-validation; the constants and FP16 unpack math are the GLSL porting reference.
- `source/backend/vulkan/buffer/execution/VulkanFP4Dequant.{hpp,cpp}` — the structural template for the new Execution: `VulkanBasicExecution` subclass, `FP4DequantConst`-style uniform const buffer, descriptor types (SSBO/SSBO/UBO), `getPipeline` variant selection, `vkCmdDispatch` with `UP_DIV`, and `barrierSource`.
- `source/backend/vulkan/buffer/execution/glsl/fp4_dequant.comp` — simplest existing buffer-backend compute shader; establishes the GLSL style (local_size_x, bindings, half conversion) to follow.
- `test/op/SGFP4DequantFixtures.h` — the committed uniform-layout fixtures (both modes × all 5 uniform layouts) the parity test reuses; expected weights are already embedded, enabling CPU-reference decode at test time.

### Established Patterns
- Buffer-backend Execution registration: `VulkanBackend::addCreator(OpType, creator)` via a `static bool gResistor = []{...}()` lambda (seen in `VulkanFP4Dequant.cpp`). The new Execution registers on `OpType_SGFP4Dequant`.
- External-sidecar loading: `{magic, offset, size}` descriptor + `mOp->externalPath()` gate (`USE_EXTERNAL_DATA`) — CPU pattern transfers directly to the Vulkan host side.
- Shader pipeline: GLSL under `source/backend/vulkan/buffer/execution/glsl/` → `python3 source/backend/vulkan/buffer/compiler/makeshader.py` → commit regenerated `AllShader.cpp`, `AllShader.h`, `VulkanShaderMap.cpp`.

### Integration Points
- `source/backend/vulkan/buffer/execution/` — new `VulkanSGFP4Dequant.{hpp,cpp}` + `glsl/sgfp4_dequant.comp` (+ FP16 variant); creator registered for `OpType_SGFP4Dequant`.
- `source/backend/vulkan/buffer/compiler/makeshader.py` outputs — regenerated and committed with the shader change.
- `test/op/SGFP4DequantTest.cpp` or a sibling `test/op/SGFP4VulkanTest.cpp` — the dual-backend parity test (SGV2-14), picked up automatically via `test/CMakeLists.txt` GLOB_RECURSE.
- `dequant_sgfp4_container_cpu()` — the reference implementation the parity test calls for the CPU side of the comparison.

</code_context>

<specifics>
## Specific Ideas

- Parity comparison uses `checkVectorByRelativeError` (float tolerance), never byte-exactness — roadmap note 3 explicitly excludes attestation/byte-exact infrastructure.
- The test must handle absent Vulkan devices gracefully (skip + clear message), since the suite may also run on machines without one.
- Keep the known build blocker in mind: `test/op/FP4ModelTest.cpp` (pre-existing, `milestone` workstream) prevents a from-scratch `run_test.out` build — see Phase 1 `deferred-items.md` and its temporary-local-stub workaround (never committed).
- Container allocation/DoS bound (Phase 1 T-01-04): the Vulkan host path must probe real file size BEFORE allocating the upload buffer — same untrusted-sidecar posture.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (GPU workgroup/coalescing/shared-memory tuning is SGV2-18 backlog; quadtree GPU walk is Phase 4 by design.)

</deferred>

---

*Phase: 3-vulkan-decode-uniform-layouts*
*Context gathered: 2026-08-24*
