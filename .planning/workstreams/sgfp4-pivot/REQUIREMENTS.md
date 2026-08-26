# Requirements: MNN SGFP4 v2 Converter Integration

**Defined:** 2026-08-26
**Core Value:** A pytorch → onnx → mnnconvert pipeline can produce a `.mnn` file that runs SGFP4 v2 (quadtree-adaptive) FP4 inference, via a native mnnconvert CLI flag.

> **Builds on sgfp4-pivot v1.0** (archived — see `.planning/milestones/v1.0-REQUIREMENTS.md`), which shipped CPU + Vulkan decode-only Executions for SGFP4 v2, validated only against synthetic Python-generated test fixtures. This milestone closes the gap to a usable end-user pipeline: quantizing real model weights and wiring the result natively into `mnnconvert`.
>
> **Architecture note (from research, see `.planning/research/SUMMARY.md`):** "op-type rewrite" is graph surgery, not in-place mutation. `OpType_SGFP4Dequant` is a 0-input weight-producer node; quantizing a `Convolution2D`/`ConvolutionDepthwise`/`Deconvolution` op means inserting a new `SGFP4Dequant` node whose output feeds the *original, type-unchanged* conv op as `inputs[1]` — MNN's existing multi-input-conv support (CPU + Vulkan) already consumes this, so no new decode-consuming Execution code is needed.

## v1 Requirements

Requirements for this milestone (v2.0). Each maps to roadmap phases.

### Schema + Sidecar Wiring

- [ ] **SGV2-22**: `SGFP4DequantParam` (`schema/default/CaffeOp.fbs`) gains a `buffer:[byte]` field to stage raw encoded container bytes pre-serialization, mirroring `Convolution2D.quanParameter.buffer:[byte]`
- [ ] **SGV2-23**: `tools/converter/source/common/RemoveParams.cpp` handles `OpParameter_SGFP4DequantParam`, reusing the existing shared external-sidecar mechanism (single `ofstream` / threaded `offset`) rather than a separate file or counter

### Real-Weight Encoder

- [ ] **SGV2-24**: A C++ port of `tools/fp4/encode_sgfp4.py`'s quadtree subdivision + dual-mode (FP4_AFFINE/T158_AFFINE) selection logic operates on real weight-tensor data and is validated against the existing Python `--selftest` oracle for correctness parity
- [ ] **SGV2-25**: A defined and implemented tiling/padding policy handles weight tensor shapes that are not multiples of the 64×64 macroblock edge (open gap identified by research — no existing codebase convention found)

### Real-Weight Validation

- [ ] **SGV2-26**: A real-weight validation step (distinct from synthetic fixtures) characterizes quantization error against actual model weight distributions from a roadmap-selected test model, run before the encoder is wired into the graph-rewrite pass
- [ ] **SGV2-27**: Encoder tiling/threshold parameters (macroblock size, subdivision thresholds, mode-selection epsilon) are revised if real-weight validation shows synthetic-fixture-tuned defaults don't hold

### Graph-Rewrite + CLI Integration

- [ ] **SGV2-28**: A new `PostConverter` optimizer pass (modeled on `tools/converter/source/optimizer/postconvert/SplitBlockQuantConvolution.cpp`) inserts an `SGFP4Dequant` node producing decoded weights, consumed by the original conv/deconv op as `inputs[1]`, and sets `externalPath` on the new node so the existing CPU/Vulkan decode Executions succeed (they hard-require it — see `CPUSGFP4Dequant::onResize`)
- [ ] **SGV2-29**: A new native mnnconvert CLI flag (peer to `--weightQuantBits`, distinctly named from the sibling `milestone` workstream's Ultra FP4/E2M1 flag to avoid "two FP4 flags" confusion) triggers SGFP4 v2 quantization during conversion; self-gated via `Global<modelConfig>::Get()` rather than the `expectedPass` short-circuit mechanism (which skips the entire default optimizer pipeline)
- [ ] **SGV2-30**: `WeightQuantAndCoding.cpp` — which runs unconditionally on `Convolution*`/`Deconvolution*` ops today, even at its own default settings — skips ops already rewritten to consume an `SGFP4Dequant` input, preventing double-processing/corruption of the same weight data

### End-to-End Validation

- [ ] **SGV2-31**: A real test model converts via pytorch → onnx → mnnconvert(SGFP4 flag) → `.mnn` and runs correct inference on CPU
- [ ] **SGV2-32**: The same converted model runs correct inference on Vulkan

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap. (Carried forward from v1.0's deferred list — still applicable, plus one new item.)

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
| Calibration/activation-data requirements (GPTQ/AWQ/llama.cpp `--imatrix`-style) | Would break MNN's established zero-calibration, single-command CLI-flag UX that both `--weightQuantBits` and HQQ already guarantee |
| Bespoke per-tensor accuracy reporting tooling | Reuse MNN's existing converter-embedded validator (`--testdir`/`--testconfig` → `Cli::testconvert`) instead of building new reporting infrastructure |
| Retrofitting `IDSTEncoder`/`IDSTQuan` for SGFP4 | Flat, fixed-shape format — already researched and rejected in the original pivot analysis for being unable to represent macroblocks/quadtrees |
| First-class FlatBuffers schema fields for quadtree internals | Locked to opaque external-file blob + minimal `{magic, offset, size}` descriptor (v1.0 decision, still applies) |
| Vendoring a tensor-math library for per-leaf encoder loops | Project prioritizes binary size; RTTI/exceptions disabled |
| Ultra FP4 (E2M1) converter integration or hardening | Separate format, separate `milestone` workstream's scope; this milestone is additive and must not touch it |
| Attestation / verifiable-execution support | Carried forward from v1.0 — MNN runs AI processing and returns a result, SuperGenius verifies separately |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SGV2-22 | Phase 5 | Pending |
| SGV2-23 | Phase 5 | Pending |
| SGV2-24 | Phase 6 | Pending |
| SGV2-25 | Phase 6 | Pending |
| SGV2-26 | Phase 7 | Pending |
| SGV2-27 | Phase 7 | Pending |
| SGV2-28 | Phase 8 | Pending |
| SGV2-29 | Phase 8 | Pending |
| SGV2-30 | Phase 8 | Pending |
| SGV2-31 | Phase 9 | Pending |
| SGV2-32 | Phase 9 | Pending |

**Coverage:**
- v1 requirements: 11 total
- Mapped to phases: 11/11 ✓
- Unmapped: 0

---
*Requirements defined: 2026-08-26*
*Last updated: 2026-08-25 after roadmap creation (Phases 5-9 mapped)*
