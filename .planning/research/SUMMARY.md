# Project Research Summary

**Project:** MNN — SGFP4 v2 Converter Integration (sgfp4-pivot v2.0)
**Domain:** Model-converter weight quantization (C++ compiler-adjacent tooling)
**Researched:** 2026-08-25
**Confidence:** HIGH

## Executive Summary

This milestone ports an already-validated SGFP4 v2 quadtree encoder (`tools/fp4/encode_sgfp4.py`) from a synthetic-fixture Python oracle into a native C++ path inside `mnnconvert`, so real trained convolution weights can be SGFP4-quantized at conversion time. Nothing net-new is required at the algorithm or decode level — CPU/Vulkan decode Executions, the FlatBuffers descriptor, and container constants already shipped in v1.0. The work is entirely encode/convert-side: a real-weight C++ encoder, sidecar/schema wiring, a graph-rewrite pass, and a new CLI flag.

The critical architectural correction from research: the "op-type rewrite" originally assumed (Convolution2D → SGFP4Dequant in place) is wrong. `OpType_SGFP4Dequant` is a 0-input, decode-only weight-producer node with no conv math, so quantizing a `Convolution2D` means **graph surgery** — inserting a new `SGFP4Dequant` node that produces the decoded weight tensor, consumed by the *original, type-unchanged* `Convolution`/`ConvolutionDepthwise`/`Deconvolution` op as its second input. MNN already supports weight-as-second-input convolution end to end (shape inference, CPU multi-input Execution, both Vulkan backends) — no new decode-consuming Execution code is needed. This must run as an early `PostConverter` optimizer pass (precedent: `tools/converter/source/optimizer/postconvert/SplitBlockQuantConvolution.cpp`), not inside the late per-op `writeFb.cpp` loop, which can only mutate existing ops, not insert nodes.

The dominant risk is unvalidated assumptions in the ported algorithm carrying over from synthetic-fixture tuning: 64×64 macroblock tiling, `[ASSUMED]`-tagged quadtree split thresholds, and per-leaf scale search were tuned on smooth synthetic ramp/constant tiles, not real weight distributions or arbitrary conv/depthwise shapes that rarely divide evenly by 64. This exact bug class (a silently wrong scale-calibration constant) already caused one production defect in this codebase (`MAX_E2M1_VALUE`, fixed earlier in this workstream's history) — real-weight validation against actual model statistics must be a first-class, early deliverable, not something deferred to final E2E testing.

## Key Findings

### Recommended Stack

No new external dependencies are needed — everything required already exists in-tree. The task is a **port + wire-up**, not new-capability work: mirror `WeightQuantAndCoding.cpp`/`HQQQuantizer.cpp` for the encoder's C++ shape, `RemoveParams.cpp` for external-sidecar wiring (once a schema gap is closed — see Architecture), and `writeFb.cpp`'s call-site pattern for hookup. Prior-art search (llama.cpp K-quants, ONNX Runtime block quantizer, TensorRT-LLM Model Optimizer) found no engine implementing SGFP4's specific combination (recursive quadtree + per-leaf error-driven dual-mode + ternary veto) natively — llama.cpp's K-quants is the closest analog for "do quantization inside the converter binary, not a Python side-script," validating this milestone's chosen shape over the Ultra FP4 precedent's post-hoc-script pattern.

**Core technologies:**
- C++ port of `tools/fp4/encode_sgfp4.py`'s quadtree/dual-mode logic — no new library; validated against the existing Python `--selftest` oracle as the correctness gate
- `tools/converter/source/common/WeightQuantAndCoding.cpp` pattern (CLI flag → `modelConfig` field → per-op dispatch) — reused, not reinvented, for the new `--sgfp4` (name TBD) flag
- `tools/converter/source/optimizer/postconvert/` `PostConverter` pass pattern — reused for the graph-insertion step

**What NOT to build:** don't retrofit `IDSTEncoder`/`IDSTQuan` (flat, fixed-shape, already researched and rejected in the original pivot analysis for being unable to represent macroblocks/quadtrees); don't add a first-class FlatBuffers schema for quadtree internals (already locked as opaque external-file blob); don't vendor a tensor-math library for small per-leaf loops (project prioritizes binary size, RTTI/exceptions disabled); don't require calibration/activation data (GPTQ/AWQ/llama.cpp `--imatrix`-style) — this would break MNN's established zero-calibration, single-command CLI-flag UX that both `--weightQuantBits` and HQQ already guarantee.

### Expected Features

**Must have (table stakes):**
- Native mnnconvert CLI flag (peer to `--weightQuantBits`) that quantizes real weights into SGFP4 v2 during conversion — no separate post-hoc script
- Targets the same op set as `--weightQuantBits` today: `Convolution2D`/`ConvolutionDepthwise`/`Deconvolution` weights
- Reuses MNN's built-in converter-embedded validator (`--testdir`/`--testconfig`, `Cli::testconvert`) rather than building bespoke per-tensor accuracy reporting
- End-to-end correctness: pytorch → onnx → mnnconvert(flag) → `.mnn` runs correct inference on CPU and/or Vulkan

**Should have (competitive):**
- Real-weight validation step distinct from synthetic-fixture tests — closes the highest-risk gap (assumptions tuned on ramp/constant tiles, not real distributions) before it's baked into the pipeline
- Clear, distinguishable CLI flag naming from the sibling `milestone` workstream's Ultra FP4 (E2M1) flag/tooling, to avoid "two FP4 flags" user confusion

**Defer (v2+):**
- GPU decode performance tuning (already tracked as SGV2-18)
- Additional macroblock geometries / alternate payload sizes (SGV2-19)
- Other-backend ports — Metal/CUDA/OpenCL (SGV2-20)
- Per-layer SGFP4 opt-out via `PostTreatContext::quantInfo`-style mechanism
- Bias/BN folding parity with `Convolution2D`'s embedded path (defer to plain `bias:[float]` left on the rewired conv op for v2.0; revisit if accuracy requires it)

### Architecture Approach

The conversion pipeline is: ONNX/TF/Caffe frontend → intermediate MNN graph → **[NEW] SGFP4 graph-rewrite `PostConverter` pass** (inserts `SGFP4Dequant` nodes, rewires conv `inputs[1]`) → existing `WeightQuantAndCoding.cpp` (must explicitly skip SGFP4-rewritten ops to avoid double-processing/corruption — it runs unconditionally on `Convolution*`/`Deconvolution*` today even at its own default) → **[MODIFIED] `RemoveParams.cpp`** (new case for `SGFP4DequantParam`, reusing the single shared `ofstream`/threaded-offset sidecar mechanism — not a separate file/counter) → final FlatBuffers write-out.

**Major components:**
1. **Schema fix** (`schema/default/CaffeOp.fbs`) — add `buffer:[byte]` to `SGFP4DequantParam` to stage raw encoded container bytes pre-serialization (precedent: `Convolution2D.quanParameter.buffer:[byte]`); this is the true first dependency, blocking everything else
2. **C++ encoder** — new component porting `encode_sgfp4.py`'s quadtree subdivision + dual-mode selection logic, callable per-weight-tensor
3. **Graph-rewrite `PostConverter` pass** — new component, modeled on `SplitBlockQuantConvolution.cpp`, inserts `SGFP4Dequant` nodes and rewires consuming conv ops; must self-gate via `Global<modelConfig>::Get()` (not the `expectedPass` short-circuit mechanism, which skips the entire default optimizer pipeline)
4. **`RemoveParams.cpp` SGFP4DequantParam case** — modified component, external-sidecar write path
5. **CLI flag + `config.hpp` field** — modified component, thin wiring layer

A second, independent runtime-side switch was found and must also be set correctly: `OpCommonUtils.cpp::createExecutionWithExternal` gates the `MNN_LOW_MEMORY`/mmap loading path and only recognizes `Convolution2D`/`Scale`/`LayerNorm` today. Additionally, both `CPUSGFP4Dequant::onResize` and `VulkanSGFP4Dequant` hard-require `mOp->externalPath()` to be non-null — the converter must explicitly set `op->externalPath = config.modelFile + ".weight"` (precedent: `SplitBlockQuantConvolution.cpp:45`) or the already-shipped v1.0 decode path will silently return `NOT_SUPPORT` on every real model.

### Critical Pitfalls

1. **Real-weight-vs-synthetic-fixture surprises** — `MACROBLOCK_EDGE=64` tiling and quadtree split thresholds are `[ASSUMED]`-tagged and tuned on smooth synthetic tiles; real conv/depthwise weight shapes rarely divide evenly by 64, and real weight distributions likely over-split, degrading compression. *Avoid by:* validating against real model weight statistics as its own deliverable (not folded into final E2E testing), and deciding the non-64-multiple tiling/padding policy explicitly before implementation.
2. **`externalPath` not set → silent `NOT_SUPPORT`** — both decode Executions hard-require it; the converter must set it explicitly. *Avoid by:* treating this as a required task in the graph-rewrite phase with an explicit test asserting decode succeeds (not just that conversion completes).
3. **`WeightQuantAndCoding.cpp` double-processing SGFP4-rewritten ops** — it runs unconditionally on the same op types even at default settings. *Avoid by:* adding an explicit skip guard keyed on ops already rewritten to consume an `SGFP4Dequant` input.
4. **Sidecar offset collisions** — the external file uses one shared `ofstream` and threaded `offset` reference across all ops; a separate file/counter for SGFP4 would corrupt multi-layer models. *Avoid by:* reusing the exact existing `saveExternalData` mechanism, not building a parallel one.
5. **`expectedPass` CLI short-circuit landmine** — mechanism used by `--splitQuantBlock` skips the *entire* default optimizer pipeline, not just adds a pass. *Avoid by:* self-gating the new pass via `Global<modelConfig>::Get()`, matching `TransformGroupConvolution.cpp`.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Schema + Sidecar Wiring
**Rationale:** Unblocks every other phase — the missing `buffer:[byte]` field on `SGFP4DequantParam` and the `RemoveParams.cpp` case are hard prerequisites. Mechanical, low-risk, no algorithm work.
**Delivers:** Schema field added + regenerated; `RemoveParams.cpp` handles `SGFP4DequantParam` reusing the shared sidecar mechanism; `externalPath` set correctly on rewritten ops.
**Addresses:** Container-format plumbing needed by all later phases.
**Avoids:** Pitfall 4 (sidecar offset collisions), Pitfall 2 (externalPath not set).

### Phase 2: Real-Weight C++ Encoder Port
**Rationale:** Testable in isolation against the existing Python `--selftest` oracle before any converter integration risk is introduced.
**Delivers:** C++ port of `encode_sgfp4.py`'s quadtree subdivision + dual-mode (FP4_AFFINE/T158_AFFINE) selection logic, operating on real weight-tensor data.
**Uses:** Stack pattern from `WeightQuantAndCoding.cpp`/`HQQQuantizer.cpp` for the C++ encoder shape.

### Phase 3: Real-Weight Validation Against Actual Model Statistics
**Rationale:** Closes the highest-risk gap (algorithm tuned on synthetic fixtures) before it's baked into the graph-rewrite pipeline in Phase 4 — cheaper to catch here than after full integration.
**Delivers:** Validation of tiling/padding policy for non-64-multiple weight shapes, and quantization-error characterization against real (not synthetic-ramp) weight distributions.
**Avoids:** Pitfall 1 (real-weight-vs-synthetic-fixture surprises) — the top-flagged risk across all four research documents.

### Phase 4: Graph-Rewrite PostConverter Pass + CLI Flag
**Rationale:** Where integration risk concentrates — graph node insertion, tensor-index bookkeeping, and coordination with the existing `WeightQuantAndCoding.cpp` pass.
**Delivers:** New `PostConverter` pass (modeled on `SplitBlockQuantConvolution.cpp`) inserting `SGFP4Dequant` nodes and rewiring conv `inputs[1]`; new CLI flag + `config.hpp` field; skip-guard in `WeightQuantAndCoding.cpp`.
**Implements:** Architecture component 3 (graph-rewrite pass) and component 5 (CLI wiring).

### Phase 5: End-to-End Validation
**Rationale:** The milestone's actual acceptance bar — a real model, not synthetic fixtures, run through the full pipeline.
**Delivers:** pytorch → onnx → mnnconvert(flag) → `.mnn` → correct inference on CPU and/or Vulkan for a real test model.

### Phase Ordering Rationale

- Schema/sidecar (Phase 1) must precede everything since `RemoveParams.cpp` and `externalPath` wiring are hard dependencies for both the encoder's output format and the graph-rewrite pass's node construction
- Real-weight validation (Phase 3) is deliberately placed *before* full graph-rewrite integration (Phase 4) rather than folded into final E2E testing, because research flagged unvalidated synthetic-fixture assumptions as the dominant risk — cheaper to discover tiling/threshold problems against an isolated encoder than after it's wired into the converter graph
- Graph-rewrite (Phase 4) comes after the encoder is proven correct in isolation (Phase 2) and validated against real data (Phase 3), since it's the highest-complexity integration surface (node insertion + pass-ordering landmines) and benefits from a trustworthy encoder underneath it

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** No designated real validation model/corpus was identified in research — needs a decision during planning (roadmap can "pick a suitable test model" per prior milestone-scoping conversation)
- **Phase 3:** Non-64-multiple tiling/padding policy is an open design gap, not resolved by research — the reshape convention (row-major `[OC, IC*KH*KW]` vs. alternative flattening) for macroblock tiling wasn't found explicitly documented anywhere in the codebase
- **Phase 4:** Node-insertion/tensor-index bookkeeping for the graph-rewrite pass — review `SplitBlockQuantConvolution.cpp` closely during planning as the concrete template

Phases with standard patterns (skip research-phase):
- **Phase 1:** Direct precedent exists (`Convolution2D.quanParameter.buffer:[byte]`, existing `RemoveParams.cpp` switch cases) — mechanical extension
- **Phase 2:** Direct port of already-correct, already-tested Python source; oracle for correctness already exists
- **Phase 5:** Standard E2E validation pattern, same shape as v1.0's own parity-testing phases

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Converter integration points, existing schema/decode readiness confirmed via direct `file:line` reads across `WeightQuantAndCoding.cpp`, `RemoveParams.cpp`, `writeFb.cpp`, `cli.cpp`, `config.hpp`, `CPUSGFP4Dequant.cpp` |
| Features | MEDIUM-HIGH | Existing `--weightQuantBits` pattern and existing SGFP4 encoder capability read directly (HIGH); ecosystem comparison (llama.cpp, ONNX Runtime, Intel Neural Compressor) via WebSearch, cross-referenced but not independently source-verified (MEDIUM) |
| Architecture | HIGH | Graph-mutation architecture, schema gap, and build order all directly confirmed via decode-op contracts, multi-input conv support across CPU/Vulkan, and actual call sites in `writeFb.cpp`/`PostConverter.cpp` |
| Pitfalls | HIGH | All pitfalls traced to specific lines in the actual codebase; real-weight-vs-synthetic risk is grounded in explicit `[ASSUMED]` tags and hard-coded constants in the actual encoder, though no real model has been run through it yet (this gap is itself the top pitfall, addressed by Phase 3) |

**Overall confidence:** HIGH

### Gaps to Address

- Non-64-multiple tiling/padding policy for real conv weight shapes: unresolved by research — must be decided during Phase 3 planning, not deferred further
- No real validation model/corpus identified: roadmap/planning should select one (matches the milestone-scoping conversation's decision to let the roadmap pick a suitable test model)
- Exact CLI flag name/semantics (peer to `--weightQuantBits`): left as a planning decision; should be visually/textually distinct from the sibling `milestone` workstream's Ultra FP4 flag naming to avoid "two FP4 flags" confusion
- Whether the C++ port should be a full reimplementation vs. a build-time Python subprocess call: PROJECT.md's "native mnnconvert CLI flag" language is interpreted here as requiring an in-process C++ port — confirm this interpretation during requirements/roadmap if there's any ambiguity
- Bias/BN folding parity: deferred to v2.0 as plain `bias:[float]` on the rewired conv op; revisit only if accuracy requires it
- Possible additional hidden op-type-keyed switches beyond the known `OpCommonUtils.cpp` mmap path — worth a final grep sweep during Phase 4 planning

## Sources

### Primary (HIGH confidence)
- Direct source reads: `tools/converter/source/common/{WeightQuantAndCoding,RemoveParams,writeFb}.cpp`, `tools/converter/source/common/cli.cpp`, `tools/converter/include/config.hpp`, `tools/converter/source/optimizer/postconvert/SplitBlockQuantConvolution.cpp`, `tools/converter/source/optimizer/postconvert/TransformGroupConvolution.cpp`, `schema/default/CaffeOp.fbs`, `source/backend/cpu/CPUSGFP4Dequant.cpp`, `source/backend/vulkan/.../VulkanSGFP4Dequant.*`, `source/shape/ShapeConvolution.cpp`, `source/backend/cpu/compute/ConvolutionFloatFactory.cpp`, `source/core/OpCommonUtils.cpp`, `tools/fp4/encode_sgfp4.py`, `tools/fp4/quantize_fp4.py`
- `.planning/PROJECT.md` — Current Milestone section, workstream scope and locked decisions
- Sibling `milestone` workstream's `STATE.md` and Ultra FP4 tool integration strategy (for cross-workstream collision analysis)

### Secondary (MEDIUM confidence)
- WebSearch: llama.cpp K-quants architecture (converter-embedded quantization precedent)
- WebSearch: ONNX Runtime block quantizer, TensorRT-LLM Model Optimizer (Python-side quantization anti-pattern comparison)
- WebSearch: general weight-only-quantization (WOQ) ecosystem patterns (GPTQ/AWQ/HQQ calibration conventions)

### Tertiary (LOW confidence)
- None — all findings traced to either direct source reads or corroborated multi-source web research

---
*Research completed: 2026-08-25*
*Ready for roadmap: yes*
