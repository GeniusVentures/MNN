# Requirements: MNN SGFP4 v2 Model-Artifact Injection Tool

**Defined:** 2026-08-26 (restructured from Converter Integration per 2026-08-26 handoff)
**Core Value:** A standalone tool takes a normally-converted `.mnn` plus one or more real SGFP4 v2 container files and produces a final `.mnn` + external sidecar where target weight tensors are produced by `OpType_SGFP4Dequant` nodes — verified loadable and runnable via the classic Interpreter/Session API, the exact path downstream `SGProcessingManager` consumption uses.

> **Builds on sgfp4-pivot v1.0** (archived — see `.planning/milestones/v1.0-REQUIREMENTS.md`), which shipped the CONSUME side only (decode). Nothing anywhere currently produces a real, loadable `.mnn` file that actually uses `OpType_SGFP4Dequant` on real weights — that is the gap this milestone closes.
>
> **Data source:** the canonical SGFP4 v2 encoder for real weights is **gnus-poc's `quantize/fp4_exporter.py`** (`GeniusCognitiveSystem/GNUS-NEO-SWARM/gnus-poc`) — independently implemented, verified byte-for-byte spec-compliant against `include/MNN/SGFP4DequantUtils.hpp`. MNN's own `tools/fp4/encode_sgfp4.py` is test-oracle-only, NOT for real weights. A starter artifact already exists: `gnus-poc/models/specialists_mlx/demo/fp4/demo.sgfp4` (132,368 bytes, 512×512 matrix, byte-verified) — but it is uniform random noise (all `UNIFORM_64`), so quadtree/MIXED coverage needs a structured second artifact.
>
> **Terminology lock:** call the format **"SGFP4 v2"** — never "Ultra FP4". gnus-poc's `manifest.json`/`PROJECT.md` label their SGFP4 output "Ultra FP4"/`fp4_ultra_v0.2`, which is a *different, unrelated format* from SuperGenius/MNN's actual `FP4_ULTRA` (E2M1, sibling `milestone` workstream).
>
> **Formerly-v2.0 Converter Integration is now v3.0** — its requirements (SGV2-22..32, below in Future Requirements) were re-mapped to Phases 8-12 on 2026-08-26 at 0% execution. v3.0 planning must re-evaluate the C++ encoder port against direct consumption of gnus-poc exporter output (as this tool does) and absorb this milestone's sidecar/rewiring learnings.

## v1 Requirements

Requirements for this milestone (v2.0). Each maps to roadmap phases.

### Injection Core — Artifact Construction & Graph Splicing

- [ ] **SGINJ-01**: The tool accepts a normally-converted `.mnn` (unmodified mnnconvert/llmexport output) plus one or more SGFP4 v2 container files (gnus-poc `fp4_exporter.py --adaptive` output), rejecting legacy v1 containers via version check rather than silently misdecoding
- [ ] **SGINJ-02**: Per target weight tensor, the tool constructs an `Op` with `type = OpType_SGFP4Dequant`, `main.type = OpParameter_SGFP4DequantParam`, `SGFP4DequantParamT{magic = kSGFP4Magic, external = {offset, size}, dims = {dimO, dimI}}`, with `op->externalPath` set literally on the op itself — this op is NOT one of the types `OpCommonUtils::createExecutionWithExternal` auto-rewrites (unlike Convolution2D/Scale/LayerNorm); a documented gotcha that is easy to miss
- [ ] **SGINJ-03**: Container bytes are written into a single merged external sidecar with non-overlapping `{offset, size}` ranges per op, and the graph is rewired so downstream consumers (e.g. the MatMul/conv that consumed the original constant weight) read the new `SGFP4Dequant` node's output
- [ ] **SGINJ-04**: The artifact serializes via `Variable::save(vars, fileName)` — the **direct-to-file** overload (`include/MNN/expr/Expr.hpp:157`), not the in-memory `std::vector<int8_t>` variant the existing test uses — and reloads via Express `Module::load`, decoding weights through the existing CPU Execution within oracle tolerance

### Classic-API Load & Run Validation

- [ ] **SGINJ-05**: The injected artifact loads and runs through the **classic** API — `Interpreter::createFromFile`/`createFromBuffer` → `createSession` → `runSession` (matching `demo/exec/pictureRecognition.cpp` and downstream `SGProcessingManager::MNN_Tensor::Process()`) — NOT just `Module::load`/Express; never previously verified end-to-end, expect friction around session input/output tensor identification (the only existing proof-of-concept graph had zero inputs)
- [ ] **SGINJ-06**: End-to-end inference with injected weight tensors matches an FP32/reference baseline within defined tolerance on CPU, with external-sidecar resolution working under the classic path (external path arrives via the op itself, not a session-level `setExternalFile`)

### Multi-Tensor Hardening & Structured-Data Coverage

- [ ] **SGINJ-07**: Multiple target weight tensors (and/or multiple containers) inject into a single artifact with independent, collision-free sidecar byte ranges, loading and running correctly; weight `dims = {dimO, dimI}` convention is documented and applied
- [ ] **SGINJ-08**: At least one structured (non-uniform) container exercises the LAYOUT_MIXED/quadtree decode path end-to-end (the uniform-random demo container is all `UNIFORM_64` and does not count — a structured artifact must be obtained or generated); malformed/empty inputs fail cleanly in the tool rather than emitting a corrupt artifact (a corrupt artifact would crash the downstream unchecked-nullptr path in `SGProcessingManager`, not fail gracefully)

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### v3.0 Converter Integration (SGV2-22..32 — formerly this milestone's v1 requirements)

Mapped to Phases 8-12 as of 2026-08-26; full phase mapping lives in ROADMAP.md's v3.0 section.

- **SGV2-22/23** → Phase 8 (Schema + Sidecar Wiring): `SGFP4DequantParam.buffer:[byte]` + `RemoveParams.cpp` externalization
- **SGV2-24/25** → Phase 9 (Real-Weight C++ Encoder Port): python→C++ port, non-64-multiple tiling policy
- **SGV2-26/27** → Phase 10 (Real-Weight Validation): encoder params vs. real weight statistics
- **SGV2-28/29/30** → Phase 11 (Graph-Rewrite PostConverter Pass + CLI Flag)
- **SGV2-31/32** → Phase 12 (End-to-End Validation): CPU + Vulkan

> **v3.0 planning note:** re-evaluate Phase 9 (C++ encoder port) against direct consumption of gnus-poc `fp4_exporter.py --adaptive` output — the injection tool (this milestone) consumes containers directly from Python; the C++ port may still be justified for a self-contained single-command UX, but that is a plan-time decision, not locked.

### Performance & Coverage

- **SGV2-33**: GPU decode performance tuning — workgroup sizing, coalesced loads, shared-memory fused dequantize→matmul
- **SGV2-34**: Additional macroblock geometries (e.g. 128×32, 32×64, 128×64) and alternate payload sizes/alignments (spec Section 10 design variants)
- **SGV2-35**: Extend SGFP4 v2 decode to other backends (Metal / CUDA / OpenCL)
- **SGV2-36**: Optional Laplacian-pyramid error weighting in the encoder's quadtree cost function
- **SGV2-37**: Per-layer SGFP4 opt-out mechanism (`PostTreatContext::quantInfo`-style), and bias/BN folding parity with `Convolution2D`'s embedded path (v2.0 accepts plain `bias:[float]` left on the rewired conv op)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| MNN-side encoding of real weights | The tool consumes containers produced by gnus-poc's `fp4_exporter.py`; MNN producing SGFP4 v2 bytes is v3.0's (Phase 9) problem, and even there direct consumption of exporter output is on the table |
| Fixing gnus-poc's `pipeline/runner.py` default quantize stage (invokes exporter without `--adaptive`, emitting legacy v1 the decoder doesn't support) | gnus-poc-side fix; this workstream must not build against that default output — only `--adaptive` v2 containers |
| Calibration/activation-data requirements (GPTQ/AWQ/llama.cpp `--imatrix`-style) | Would break MNN's established zero-calibration UX; applies to v3.0's converter path |
| Bespoke per-tensor accuracy reporting tooling | Reuse the FP32/reference baseline comparisons built into Phase 6/7 validation instead of new reporting infrastructure |
| Retrofitting `IDSTEncoder`/`IDSTQuan` for SGFP4 | Flat, fixed-shape format — rejected in the original pivot analysis; unchanged |
| First-class FlatBuffers schema fields for quadtree internals | Locked to opaque external-file blob + minimal `{magic, offset, size}` descriptor (v1.0 decision, still applies) |
| Vulkan decode of injected artifacts in this milestone | v1.0 shipped the Vulkan Execution; classic-API (Phase 6) and multi-tensor (Phase 7) validation target CPU; Vulkan E2E remains v3.0 Phase 12's success criterion |
| Fixing `SGProcessingManager`'s unchecked-`nullptr` `Process()` deref in `StartProcessing()` | Adjacent known issue in SuperGenius, out of MNN scope — but Phase 7 requirement 3 (clean failure on malformed input) exists precisely because a corrupt artifact would crash there rather than fail gracefully |
| Fixing `SGProcessingManager`'s stale `mnn_tensor_fp4_test.cpp` tests | Adjacent known issue, unrelated to SGFP4; separate repo's backlog |
| Ultra FP4 (E2M1) converter integration or hardening | Separate format, separate `milestone` workstream's scope; additive only, must not touch it |
| Attestation / verifiable-execution support | Carried forward from v1.0 — MNN runs AI processing and returns a result, SuperGenius verifies separately |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SGINJ-01 | Phase 5 | Pending |
| SGINJ-02 | Phase 5 | Pending |
| SGINJ-03 | Phase 5 | Pending |
| SGINJ-04 | Phase 5 | Pending |
| SGINJ-05 | Phase 6 | Pending |
| SGINJ-06 | Phase 6 | Pending |
| SGINJ-07 | Phase 7 | Pending |
| SGINJ-08 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 8 total (SGINJ-01..08)
- Mapped to phases: 8/8 ✓
- Unmapped: 0
- v3.0 requirements (SGV2-22..32) mapped to Phases 8-12 — see ROADMAP.md

---
*Requirements defined: 2026-08-26*
*Last updated: 2026-08-26 — v2.0 restructured to Model-Artifact Injection Tool per handoff; Converter Integration (SGV2-22..32) moved to v3.0 / Phases 8-12*
