# Roadmap: MNN — SGFP4 v2 Quadtree-Adaptive Quantization

## Milestones

- ✅ **v1.0 SGFP4 v2 Decode (Vulkan-parity)** — Phases 1-4 (shipped 2026-08-26)
- 🚧 **v2.0 SGFP4 v2 Model-Artifact Injection Tool** — Phases 5-7 (planning)
- 📋 **v3.0 SGFP4 v2 Converter Integration** — Phases 8-12 (planned; re-scoped at plan time)

## Phases

<details>
<summary>✅ v1.0 SGFP4 v2 Decode (Vulkan-parity) (Phases 1-4) — SHIPPED 2026-08-26</summary>

Full detail archived to `.planning/milestones/v1.0-ROADMAP.md`; phase docs (PLAN/SUMMARY/VERIFICATION/etc.) archived to `.planning/milestones/v1.0-phases/`.

- [x] Phase 1: Affine Dual-Mode Decode Core (CPU, Uniform Layouts) (2/2 plans) — completed 2026-08-24
- [x] Phase 2: Adaptive Quadtree Layout (CPU, LAYOUT_MIXED) (2/2 plans) — completed 2026-08-24
- [x] Phase 3: Vulkan Decode — Uniform Layouts (4/4 plans) — completed 2026-08-25
- [x] Phase 4: Vulkan Decode — Adaptive Quadtree (LAYOUT_MIXED) (2/2 plans) — completed 2026-08-25

</details>

### 🚧 v2.0 SGFP4 v2 Model-Artifact Injection Tool (Planning)

**Milestone Goal:** A standalone tool takes a normally-converted `.mnn` plus one or more real SGFP4 v2 container files and produces a final `.mnn` + external sidecar where target weight tensors are produced by `OpType_SGFP4Dequant` nodes — verified loadable and runnable via the **classic** Interpreter/Session API (the downstream `SGProcessingManager` consumption path).

**Why this milestone (2026-08-26 restructuring):** v1.0 shipped the consume side (decode) only — nothing anywhere produces a real, loadable `.mnn` that actually uses `OpType_SGFP4Dequant` on real weights. The previously-roadmapped Converter Integration milestone (now v3.0) reaches a real artifact only at the end of 5 phases; this milestone front-loads one via post-hoc graph surgery on an already-converted `.mnn`, using externally-encoded containers from gnus-poc's `fp4_exporter.py` (the canonical real-weight SGFP4 v2 encoder — independently verified byte-for-byte against `SGFP4DequantUtils.hpp`; MNN's own `tools/fp4/encode_sgfp4.py` is test-oracle-only). Restructured while the converter milestone was at 0% with no plans written — zero renumbering cost. Learnings (sidecar conventions, dims/padding, classic-API load behavior) flow forward into v3.0 planning.

**Terminology:** the format is called **"SGFP4 v2"** — never "Ultra FP4" (that is the sibling `milestone` workstream's unrelated E2M1/`FP4_ULTRA` format, despite gnus-poc manifest labels suggesting otherwise).

- [ ] **Phase 5: Injection Core — Artifact Construction & Graph Splicing** — Build the tool: load `.mnn`, splice `SGFP4Dequant` nodes with sidecar byte ranges, rewire consumers, serialize to disk
- [ ] **Phase 6: Classic-API Load & Run Validation** — Prove injected artifacts run through `Interpreter`/`Session` (not just Express), matching FP32 baseline
- [ ] **Phase 7: Multi-Tensor Hardening & Structured-Data Coverage** — Multiple containers/tensors, MIXED/quadtree artifact coverage, clean failure on malformed input

## Phase Details

### Phase 5: Injection Core — Artifact Construction & Graph Splicing

**Goal**: Given a normally-converted `.mnn` and one or more SGFP4 v2 container files, the tool produces a new `.mnn` + external sidecar in which target weight tensors are replaced by `OpType_SGFP4Dequant` nodes — correct at the Express/`Module::load` level first.
**Depends on**: Phase 4 (v1.0 — existing CPU decode Execution is the ground-truth consumer)
**Requirements**: SGINJ-01, SGINJ-02, SGINJ-03, SGINJ-04
**Success Criteria** (what must be TRUE):

1. The tool accepts (a) a `.mnn` from the unmodified mnnconvert/llmexport path and (b) one or more SGFP4 v2 container files (gnus-poc `fp4_exporter.py --adaptive` output), rejecting legacy v1 containers via version check rather than silently misdecoding.
2. Per target weight tensor: an `Op` with `type = OpType_SGFP4Dequant`, `main.type = OpParameter_SGFP4DequantParam`, `SGFP4DequantParamT{magic = kSGFP4Magic, external = {offset, size}, dims = {dimO, dimI}}` — with `op->externalPath` set literally on the op itself (this op is NOT covered by `OpCommonUtils::createExecutionWithExternal` auto-injection; documented gotcha from the existing test).
3. Byte ranges written into a single merged sidecar are non-overlapping and match each op's `{offset, size}`; the spliced graph's downstream consumers read the new node's output instead of the original constant.
4. Serialized via `Variable::save(vars, fileName)` — the direct-to-file overload (not the in-memory `std::vector<int8_t>` variant) — and the artifact reloads via Express `Module::load` (with `rtmgr->setExternalFile()` before load) decoding weights through the existing CPU Execution within oracle tolerance.

**Plans**: 2 plans

Plans:
**Wave 1**

- [ ] 05-01-PLAN.md — Runtime-level injection recipe test (A1 spike) + byte-level version-gate helper (SGINJ-01..04)

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 05-02-PLAN.md — `sgfp4_inject` tool: manifest-driven pairing, graph surgery, sidecar merge, save + in-tool verify

### Phase 6: Classic-API Load & Run Validation

**Goal**: The injected artifact loads and runs through the classic Interpreter/Session API — `Interpreter::createFromFile`/`createFromBuffer` → `createSession` → `runSession` — the exact path `SGProcessingManager::MNN_Tensor::Process()` uses downstream; never previously verified end-to-end.
**Depends on**: Phase 5
**Requirements**: SGINJ-05, SGINJ-06
**Success Criteria** (what must be TRUE):

1. `Interpreter::createFromFile` → `createSession` → `runSession` succeeds on an injected artifact, including correct session input/output tensor identification (expected friction: the only prior proof-of-concept graph had zero inputs).
2. End-to-end inference with at least one injected weight tensor matches an FP32/reference baseline within defined tolerance on CPU, via the existing decode Execution.
3. External-sidecar resolution works under the classic API path (external path arrives via the op itself, not a session-level `setExternalFile`).

**Plans**: 1/2 plans executed

- [x] 06-01-PLAN.md
- [ ] 06-02-PLAN.md

### Phase 7: Multi-Tensor Hardening & Structured-Data Coverage

**Goal**: The tool handles realistic multi-weight models and the full SGFP4 v2 format surface — not just the single uniform-random demo artifact — and fails cleanly on malformed input.
**Depends on**: Phase 6
**Requirements**: SGINJ-07, SGINJ-08
**Success Criteria** (what must be TRUE):

1. Multiple target weight tensors (and/or multiple containers) inject into a single artifact with independent, collision-free sidecar byte ranges, loading and running correctly.
2. At least one structured (non-uniform) container exercises the LAYOUT_MIXED/quadtree decode path end-to-end — the uniform-random demo container produces all `UNIFORM_64` and does NOT count as quadtree coverage (handoff caveat; a structured second artifact must be obtained or generated).
3. Weight shapes/dims convention (`dims = {dimO, dimI}` on the matrix) is documented and applied; malformed/empty inputs fail cleanly in the tool rather than emitting a corrupt artifact (relevant because a corrupt artifact would crash the downstream unchecked-nullptr path in `SGProcessingManager`, not fail gracefully).

**Plans**: TBD

### 📋 v3.0 SGFP4 v2 Converter Integration (Planned)

**Milestone Goal:** A pytorch → onnx → mnnconvert pipeline can produce a `.mnn` file that runs SGFP4 v2 (quadtree-adaptive) FP4 inference, via a native mnnconvert CLI flag.

Formerly the v2.0 milestone (roadmapped 2026-08-25 at 0% execution; moved 2026-08-26 when the Injection Tool milestone was inserted ahead of it). Phase content below carries forward; **note for v3.0 planning:** the SGFP4-pivot handoff (2026-08-26) designates gnus-poc's `fp4_exporter.py` as the canonical real-weight encoder — Phase 9 (C++ encoder port) must be re-evaluated at plan time against direct consumption of exporter output (as the injection tool does), and Phase 11 should absorb the injection tool's sidecar/rewiring learnings.

- [ ] **Phase 8: Schema + Sidecar Wiring** - Add `buffer:[byte]` to `SGFP4DequantParam` and wire `RemoveParams.cpp` to externalize it through the existing shared sidecar mechanism
- [ ] **Phase 9: Real-Weight C++ Encoder Port** - Port the Python quadtree/dual-mode encoder to C++, operating on real (non-64-aligned) weight tensor shapes
- [ ] **Phase 10: Real-Weight Validation Against Actual Model Statistics** - Validate/revise encoder parameters against a real model's weight distributions before graph-rewrite integration
- [ ] **Phase 11: Graph-Rewrite PostConverter Pass + CLI Flag** - New `PostConverter` pass inserts `SGFP4Dequant` nodes; new CLI flag triggers it; `WeightQuantAndCoding.cpp` skip-guard prevents double-processing
- [ ] **Phase 12: End-to-End Validation** - A real model converts and runs correct inference on CPU and Vulkan via the new flag

**(Phase details for 8-12 preserved as previously roadmapped — goals, dependencies, and success criteria transfer with renumbering: Phase 8 was Phase 5, 9←6, 10←7, 11←8, 12←9. Dependencies: Phase 8 ← Phase 4 (v1.0) + v2.0 learnings; Phase 9 ← 8; Phase 10 ← 9; Phase 11 ← 8, 10; Phase 12 ← 11.)**

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 1. Affine Dual-Mode Decode Core (CPU, Uniform) | v1.0 | 2/2 | Complete | 2026-08-24 |
| 2. Adaptive Quadtree Layout (CPU, LAYOUT_MIXED) | v1.0 | 2/2 | Complete | 2026-08-24 |
| 3. Vulkan Decode — Uniform Layouts | v1.0 | 4/4 | Complete | 2026-08-25 |
| 4. Vulkan Decode — Adaptive Quadtree | v1.0 | 2/2 | Complete | 2026-08-25 |
| 5. Injection Core — Artifact Construction & Graph Splicing | v2.0 | 0/2 | Not started | - |
| 6. Classic-API Load & Run Validation | v2.0 | 1/2 | In Progress|  |
| 7. Multi-Tensor Hardening & Structured-Data Coverage | v2.0 | 0/TBD | Not started | - |
| 8. Schema + Sidecar Wiring | v3.0 | 0/TBD | Not started | - |
| 9. Real-Weight C++ Encoder Port | v3.0 | 0/TBD | Not started | - |
| 10. Real-Weight Validation Against Actual Model Statistics | v3.0 | 0/TBD | Not started | - |
| 11. Graph-Rewrite PostConverter Pass + CLI Flag | v3.0 | 0/TBD | Not started | - |
| 12. End-to-End Validation | v3.0 | 0/TBD | Not started | - |
