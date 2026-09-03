# Architecture Research: SGFP4 v2 Converter Integration

**Domain:** MNN model-converter (mnnconvert) graph-mutation and external-weight-serialization pipeline
**Researched:** 2026-08-25
**Confidence:** HIGH — every claim below is grounded in source read directly from this checkout, cited `file:line`. No external/web sources were needed; this is a closed-repository architecture question, not an ecosystem survey.

## Standard Architecture

### System Overview (mnnconvert pipeline, as it exists today)

```
┌────────────────────────────────────────────────────────────────────────┐
│  FRONTEND IMPORT (ONNX/TF/Caffe/TFLite → in-memory MNN::NetT)          │
│  tools/converter/source/{onnx,tensorflow,caffe,tflite}/                │
│  Convolution weights land as OpType_Convolution/-Depthwise/Deconv      │
│  with `Convolution2D.weight:[float]` populated in-line (no external)   │
└───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  OPTIMIZER / GRAPH-MUTATION PASSES  (cli.cpp:769 → optimizeNet())      │
│  tools/converter/source/optimizer/PostConverter.cpp:optimizeNetImpl    │
│  Ordered, named `PostConverter` passes; each may insert/delete ops     │
│  and tensors (NetT::oplists is a vector<unique_ptr<OpT>>, mutated by   │
│  iterator insert/erase — see SplitBlockQuantConvolution.cpp:29-202     │
│  for the established insert-new-node idiom).                          │
│  Late-pipeline passes already fuse INTO Convolution2D here:            │
│    MergeBNToConvolution, MergeScaleToConvolution, MergeReluToConv...,  │
│    TransformGroupConvolution, ConvertMatMulToConv2D                    │
│  → by the end of this stage every conv is in its FINAL Convolution2D/  │
│    ConvolutionDepthwise/Deconvolution op-shape.                        │
│  ★ THIS is where the new SGFP4 op-type-rewrite pass belongs.           │
└───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  writeFb.cpp:postTreat() — per-op FINAL PASS (writeFb.cpp:90-176)      │
│  For every op (main list + subgraphs), in order:                       │
│   1. loadExternalParam()   (RemoveParams.cpp:131) — pulls PRE-EXISTING │
│      external bytes back into memory (re-conversion / already-external │
│      source models only)                                               │
│   2. WeightQuantAndCoding() (WeightQuantAndCoding.cpp:75) — int2-8      │
│      in-place quant of Convolution2D.weight → quanParameter (IDST)     │
│   3. RemoveAndStoreParam()  (RemoveParams.cpp:30) — flushes big byte    │
│      buffers (quanParameter.buffer, weight, bias, Blob data, ...) to   │
│      the `.mnn.weight` sidecar, replacing them with `external:[int64]` │
│      offset/size pairs in the OpT                                      │
│  ★ RemoveAndStoreParam has NO case for OpParameter_SGFP4DequantParam   │
│    today — this is the second thing that must be added.                │
└───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  FlatBuffers serialize → `.mnn` + `.mnn.weight` sidecar (writeFb.cpp:  │
│  269-293)                                                               │
└────────────────────────────────────────────────────────────────────────┘
```

### Runtime decode side (already shipped in sgfp4-pivot v1.0 — unchanged by this milestone)

```
Interpreter loads .mnn + .mnn.weight
    │
    ├─ OpType_SGFP4Dequant node (0 inputs, 1 output)
    │    ShapeSGFP4Dequant (source/shape/ShapeSGFP4Dequant.cpp:19-38)
    │      output shape = SGFP4DequantParam.dims (manifest-resident)
    │    CPUSGFP4Dequant::onResize  (reads external[offset,size] from
    │      the sidecar via mOp->externalPath, once)
    │    CPUSGFP4Dequant::onExecute (decodes container → float tensor,
    │      EVERY forward call — see Pitfalls)
    │    VulkanSGFP4Dequant mirrors this on GPU
    │         │
    │         ▼ (decoded float weight tensor, as inputs[1])
    │  OpType_Convolution / ConvolutionDepthwise / Deconvolution
    │    (UNCHANGED op type — consumes weight as a 2nd graph input,
    │     a pre-existing MNN capability, not new for SGFP4 — see below)
    ▼
  conv output
```

### Component Responsibilities

| Component | Responsibility | Status |
|-----------|----------------|--------|
| `schema/default/CaffeOp.fbs:118-122` `SGFP4DequantParam` | On-disk descriptor `{magic, external:[offset,size], dims}` | Exists (v1.0), **missing a staging-bytes field** — see Gap below |
| `source/shape/ShapeSGFP4Dequant.cpp` | Computes output tensor shape from `dims` | Exists, unmodified |
| `source/backend/cpu/CPUSGFP4Dequant.{hpp,cpp}` / `VulkanSGFP4Dequant.{hpp,cpp}` | Decode container bytes → float tensor at runtime | Exists, unmodified — pure decode, **no conv math** |
| `tools/fp4/encode_sgfp4.py` | Reference/oracle encoder (synthetic fixtures) | Exists — **source of truth to port to C++** |
| `tools/converter/source/optimizer/postconvert/*.cpp` (`PostConverter` passes) | Graph-structural mutation (insert/delete ops+tensors) | Exists as a pattern; **new pass needed** |
| `tools/converter/source/common/WeightQuantAndCoding.cpp` | In-place per-op int2-8 weight quant (no graph mutation) | Exists — **not the right home for SGFP4** (see below) |
| `tools/converter/source/common/RemoveParams.cpp` | Per-OpParameter-type switch: byte-buffer fields → sidecar `external:[offset,size]` | Exists — **needs a new switch case** |
| `tools/converter/source/common/cli.cpp` / `tools/converter/include/config.hpp` | CLI flag parsing → `modelConfig` | Exists — **needs a new flag, mirrors `weightQuantBits`** |
| New: `SGFP4Encoder.{hpp,cpp}` (proposed) | Real-weight port of `encode_sgfp4.py`'s math | **Does not exist yet** |
| New: `SGFP4QuantizeConvolution.cpp` (proposed `PostConverter`) | Graph split: Convolution2D → `[SGFP4Dequant weight-producer node] + [Convolution/Deconv consuming it as inputs[1]]` | **Does not exist yet** |

## Key Architectural Finding: "op-type rewrite" is a graph SPLIT, not an in-place mutation

The milestone context's phrase "op-type rewrite (Convolution2D → OpType_SGFP4Dequant)" cannot mean literally flipping one op's type in place, the way `WeightQuantAndCoding.cpp` flips `Convolution2D.weight` → `Convolution2D.quanParameter` in place (same op, same op count, same tensor graph). Evidence:

- `ShapeSGFP4Dequant.cpp:22-36` and `CPUSGFP4Dequant.cpp:91-112` both confirm `OpType_SGFP4Dequant` is a **0-input, Const-like decode-only op**: it produces a float weight *tensor*, it does not perform convolution/matmul. There is no fused "SGFP4 conv" Execution anywhere in the codebase (confirmed by exhaustive grep across `source/backend/{cpu,vulkan}` for `SGFP4`).
- Therefore quantizing a Convolution2D's weight to SGFP4 necessarily requires **two graph nodes** where there was one: a new `OpType_SGFP4Dequant` node that owns the sidecar-backed weight, and the original conv node (type **unchanged** — still `OpType_Convolution`/`ConvolutionDepthwise`/`OpType_Deconvolution`) rewired to take that decoded tensor as a second input rather than an embedded `weight:[float]`.
- MNN **already has** first-class support for exactly this "weight as a second graph input" conv shape — it is not new plumbing this milestone must invent:
  - Shape inference: `source/shape/ShapeConvolution.cpp:33-38` — `if (inputs.size() > 1 && outputCount == 0) { outputCount = inputs[1]->length(0); kX = inputs[1]->length(3); kY = inputs[1]->length(2); }` (comment: `"From TF's multi input convolution"`). Weight tensor layout expected: `[O, I, KH, KW]`.
  - CPU execution: `source/backend/cpu/compute/ConvolutionFloatFactory.cpp:188-190` (`ConvolutionTiledExecutorMultiInput`), plus `CPUDeconvolution.cpp:371,376`, `CPUDeconvolutionDepthwise.cpp:108,177`, `CPUConvolutionDepthwise.cpp:112,215` all branch on `inputs.size() > 1`.
  - Vulkan execution (both backends): `source/backend/vulkan/image/execution/VulkanConvolutionImpl.cpp:211`, `VulkanConvolution.cpp:178-192`, `VulkanDeconvolution.cpp:151`, `VulkanDeconvolutionDepthwise.cpp:113`; buffer backend: `source/backend/vulkan/buffer/execution/VulkanConvolution.cpp:249-250`, `VulkanDeconvolution.cpp:197,239`.
- **Recommendation:** treat "op-type rewrite" as: *the op that owns the weight bytes* changes from `Convolution2D` (embedded) to `SGFP4Dequant` (external, standalone weight-producer); the conv op itself keeps its existing type and gains a second input, exercising MNN's pre-existing multi-input conv/deconv path on every backend this milestone targets (CPU and Vulkan). No new Execution code is needed for the consuming side.

## Gap Found: schema has no place to stage raw container bytes pre-serialization

`SGFP4DequantParam` (`schema/default/CaffeOp.fbs:118-122`) intentionally carries only `{magic, external:[offset,size], dims}` — by design ("No macroblock/quadtree/leaf/split-map fields belong here"). But `external` is only valid **after** `RemoveAndStoreParam` has run; before that, the just-encoded container bytes need somewhere to live in the in-memory `OpT`, the same way `Convolution2D.quanParameter.buffer:[byte]` (`CaffeOp.fbs:56`, the `IDSTQuan` table) stages encoded weight bytes between `WeightQuantAndCoding` and `RemoveAndStoreParam` (`RemoveParams.cpp:41-51` reads `param->quanParameter->buffer`).

**This field does not currently exist on `SGFP4DequantParam`.** It is a small, additive, schema change required before anything else can work end-to-end:

```
table SGFP4DequantParam {
    magic:uint32;
    buffer:[byte];    // NEW — opaque, pre-encoded v2 container bytes, staged
                       // in-memory until RemoveAndStoreParam flushes them to
                       // the sidecar and replaces this with `external`.
    external:[int64];
    dims:[int];
}
```

This is consistent with the schema comment's intent (no *structured* macroblock/leaf fields — `buffer` is opaque, exactly like `quanParameter.buffer` is for `Convolution2D`), and it is the one change every other piece of this milestone depends on. Adding it requires regenerating `schema/current/*_generated.h` via `schema/generate.sh` (or `generate.ps1` on Windows) — see Build Order below.

## Integration Points

### New components

| Component | File (proposed) | Purpose |
|-----------|------------------|---------|
| SGFP4 real-weight encoder | `tools/converter/source/common/SGFP4Encoder.{hpp,cpp}` (peer to `IDSTEncoder.hpp`, used the same way `WeightQuantAndCoding.cpp:12` uses `IDSTEncoder`) | C++ port of `tools/fp4/encode_sgfp4.py`'s `encode_leaf_fp4`/`encode_leaf_t158`/`select_mode`/`subdivide_macroblock`/`classify_layout`/`pack_leaf_header`/`pack_payload`/`encode_container`/`encode_container_adaptive`. Must reuse the byte-layout constants already centralized in `include/MNN/SGFP4DequantUtils.hpp` (magic, header sizes, nibble/symbol packing) rather than re-deriving them, to guarantee encoder/decoder never drift. |
| Graph-rewrite `PostConverter` pass | `tools/converter/source/optimizer/postconvert/SGFP4QuantizeConvolution.cpp` | For each `Convolution2D`/`ConvolutionDepthwise`/`Deconvolution` op (when the new CLI flag is set): call the encoder on `param->weight`, build a new `OpT` with `type=OpType_SGFP4Dequant`, `main.type=OpParameter_SGFP4DequantParam`, `SGFP4DequantParamT{magic, buffer=<container bytes>, dims=[O,I,KH,KW]}`, insert it into `net->oplists` + a new tensor name, then clear the original conv's `weight`/`bias`(if folded) and extend its `inputIndexes` to reference the new node's output tensor. Registered via `PostConverterRegister<SGFP4QuantizeConvolution> __l("SGFP4QuantizeConvolution");` (same idiom as every file in `postconvert/`). |
| New CLI flag | `tools/converter/include/config.hpp` (new `bool`/`int` field, peer to `weightQuantBits` at `config.hpp:42-45`) + `cli.cpp` (`cxxopts` option block, peer to `weightQuantBits` at `cli.cpp:209-212`, plus a parse-and-assign block peer to `cli.cpp:496-504`) | Native CLI trigger for SGFP4 quantization |

### Modified components

| Component | File:line | Change |
|-----------|-----------|--------|
| Schema | `schema/default/CaffeOp.fbs:118-122` | Add `buffer:[byte]` staging field to `SGFP4DequantParam` (see Gap above); regenerate `schema/current/*.h` |
| `RemoveParams.cpp` write path | `RemoveAndStoreParam`, switch at `RemoveParams.cpp:35-101` | New `case MNN::OpParameter_SGFP4DequantParam:` — `storeWeight<int8_t>(fs, param->buffer, param->external, offset)` (single blob, unlike Convolution2D's 3-4 field case; closest precedent is the single-vector `Scale`/`LayerNorm` cases at `RemoveParams.cpp:53-68`) |
| `RemoveParams.cpp` load path | `loadExternalParam`, switch at `RemoveParams.cpp:138-204` | New `case MNN::OpType_SGFP4Dequant:` mirroring the `Convolution2D`/`Scale`/`LayerNorm` cases — needed for round-trip/re-conversion consistency even though the fresh-encode path (Step 3 below) doesn't strictly require it on first pass |
| Optimizer pass list | `tools/converter/source/optimizer/PostConverter.cpp:265-360ish` (`optimizeNetImpl`'s hardcoded `postConvertPass`/`afterProgramConvert` lists) | Insert `"SGFP4QuantizeConvolution"` **after** `MergeBNToConvolution`, `MergeScaleToConvolution`, `TransformGroupConvolution`, `ConvertMatMulToConv2D` (so it only ever sees final, fully-fused `Convolution2D`/`ConvolutionDepthwise`/`Deconvolution` ops) |
| `cli.cpp` | `cli.cpp:769` (`optimizeNet(netT, ..., expectedPass)` call site) | **Do NOT** add the new pass name to `expectedPass` the way `splitQuantBlock` does at `cli.cpp:763-765` — that is a trap (see Pitfall below). Self-gate the pass internally instead, via `Global<modelConfig>::Get()`, exactly like `TransformGroupConvolution.cpp:177-178` (`if(config->groupConvNative)`) and `ConvertMatMulToConv2D.cpp:121-122` (`if(!config->convertMatmulToConv)`) already do for their own flags |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `SGFP4QuantizeConvolution` pass ↔ `SGFP4Encoder` | Direct C++ call, in-process | Pass owns graph mutation; encoder owns only the math (pure function: `vector<float> weight, dims → container bytes`), independently unit-testable against `encode_sgfp4.py --selftest` as a cross-language oracle, exactly as `test/op/SGFP4DequantFixtures.h` already does for the decode side |
| `SGFP4QuantizeConvolution` pass ↔ `writeFb.cpp` postTreat loop | Implicit, via `NetT::oplists` contents | The pass does **not** need to call `RemoveAndStoreParam` itself (unlike `SplitBlockQuantConvolution.cpp:86,121`, which manages its own external file because it operates on an *already-externalized* re-conversion source). Since this pass runs on freshly-imported, in-memory `Convolution2D.weight` floats, the ordinary `writeFb.cpp:159-167` → `_postTreatOp` → `RemoveAndStoreParam` loop picks up the new `SGFP4Dequant` node automatically once the new switch case (above) exists — avoiding a second, duplicate sidecar-writing code path |
| New `SGFP4Dequant` node ↔ rewired `Convolution`/`Deconvolution` node | Graph tensor edge (`inputIndexes[1]`) | Reuses MNN's existing multi-input conv/deconv Executions (CPU: `ConvolutionTiledExecutorMultiInput`; Vulkan: multi-input branches cited above) — no new Execution code required |

## Recommended Build Order

Dependencies flow: **schema → (encoder ∥ RemoveParams case) → graph-rewrite pass → CLI flag → E2E validation**. Encoder math and the `RemoveParams.cpp` case are mutually independent and can be built in parallel once the schema field exists; the graph-rewrite pass needs both.

1. **Schema field** (`schema/default/CaffeOp.fbs:118-122` — add `buffer:[byte]`) + regenerate (`schema/generate.sh`/`generate.ps1`) → commit regenerated `schema/current/*.h`. Blocking for everything else; do this first and alone.
2. **`RemoveParams.cpp` case** (write path first, load path for symmetry) — buildable and testable in isolation by hand-constructing an `OpT{type=OpType_SGFP4Dequant, main=SGFP4DequantParamT{buffer=<bytes>}}` and calling `RemoveAndStoreParam` directly, the same way `test/op/SGFP4DequantTest.cpp:360-447` already hand-constructs an `SGFP4DequantParamT` for its op-level decode test. No dependency on the encoder or the graph pass.
3. **`SGFP4Encoder.{hpp,cpp}`** (C++ port of `encode_sgfp4.py`) — buildable and testable in isolation against the Python `--selftest` oracle (byte-for-byte container comparison), no converter wiring required. Can proceed in parallel with step 2.
4. **`SGFP4QuantizeConvolution` graph-rewrite pass** — depends on steps 1–3. This is where real integration risk concentrates: node insertion/tensor-index bookkeeping (follow `SplitBlockQuantConvolution.cpp:29-202`'s iterator-insert idiom), weight-layout mapping into `dims=[O,I,KH,KW]` to match `ShapeConvolution.cpp:33-38`'s expectations for `inputs[1]`, and correctly clearing/rewiring the original conv op.
5. **CLI flag** (`config.hpp` field, `cli.cpp` `cxxopts` block + parse/assign, `PostConverter.cpp` pass-list insertion, self-gate check inside the pass) — thin plumbing; the flag has nothing to gate until step 4's pass class exists, but the `config.hpp`/`cli.cpp` boilerplate itself has no dependency and can be drafted in parallel with step 4.
6. **End-to-end validation** — pytorch → onnx → `mnnconvert --sgfp4Quantize` (or whatever the flag is named) → `.mnn`, run on CPU (exercises `ConvolutionTiledExecutorMultiInput`) and/or Vulkan (exercises the multi-input branches in `VulkanConvolutionImpl.cpp`/`VulkanConvolution.cpp`), compared against an unquantized float baseline.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Wiring the new pass through `expectedPass` like `splitQuantBlock`

**What people might do:** Mirror `cli.cpp:763-765` (`if (modelPath.splitQuantBlock) { expectedPass.emplace_back("SplitBlockQuantConvolution"); }`) for the new flag.
**Why it's wrong:** `optimizeNet()` (`PostConverter.cpp:635-638`) short-circuits: `if (!expectPasses.empty()) { RunNetPass(expectPasses, originNet); return std::move(originNet); }`. Any non-empty `expectedPass` list **skips the entire default optimization pipeline** — `RemoveInplace`, `RemoveUnusefulOp`, `TransformInnerProduct`, all the `MergeXToConvolution` passes, `ConvertMatMulToConv2D`, etc. This mechanism exists for re-optimizing an *already-converted* `.mnn` file in isolation (`SplitBlockQuantConvolution`'s actual use case — see its own external-file handling at `SplitBlockQuantConvolution.cpp:24-26,44-52`), not for injecting one extra pass into a normal ONNX/TF→MNN conversion.
**Do this instead:** Insert the new pass's name unconditionally into `optimizeNetImpl`'s hardcoded pass list in `PostConverter.cpp` (after the weight-fusion passes), and self-gate execution inside the pass via `Global<modelConfig>::Get()`, exactly as `TransformGroupConvolution.cpp:177-178` and `ConvertMatMulToConv2D.cpp:121-122` already do.

### Anti-Pattern 2: Extending `WeightQuantAndCoding.cpp` instead of writing a new `PostConverter` pass

**What people might do:** Add an `if (config.sgfp4Quantize)` branch inside `WeightQuantAndCoding()` (`WeightQuantAndCoding.cpp:75`), since it already receives every `Convolution2D`/`ConvolutionDepthwise`/`Deconvolution` op and is the existing "quantize this conv" entry point.
**Why it's wrong:** `WeightQuantAndCoding` is called per-op, in-place, from inside `writeFb.cpp:159-167`'s `for (auto& op : netT->oplists)` loop (by value/reference, not by iterator) — it can only *mutate* the op it's given, it structurally cannot insert a sibling node or grow `netT->tensorName`, both of which the SGFP4 split requires (see Key Architectural Finding above).
**Do this instead:** A `PostConverter` graph pass (see Integration Points), which owns `net->oplists`/`net->tensorName` via an insert-capable iterator, run earlier in the pipeline (during `optimizeNet()`, not during `writeFb.cpp`'s final per-op pass).

## Performance / Correctness Pitfalls Flagged for Later Phases

- **Multi-input convs fall off MNN's fast prepacked-weight kernels.** `ConvolutionFloatFactory.cpp:188-190` routes any conv with `inputs.size() > 1` to `ConvolutionTiledExecutorMultiInput`, bypassing the Winograd/prepacked-weight tiled paths used for single-input (embedded-weight) convs. SGFP4-quantized convs will not get MNN's fastest CPU kernels; same applies on Vulkan (the `inputs.size() > 1` branches cited above are separate, generic code paths from the prepacked fast-conv kernels). This is an inherent cost of "decode via graph op" rather than "decode fused inside the conv Execution" (the pattern MNN's existing int4/int8 weight-quant path uses instead, via `quanParameter`). Given `PROJECT.md`'s Active requirements for this milestone only ask for E2E *correctness*, not benchmarking, this is acceptable for v2.0 but should be flagged as a follow-up roadmap item, not silently assumed away.
- **`CPUSGFP4Dequant::onExecute` re-decodes the full container on every forward call** (`CPUSGFP4Dequant.cpp:91-112` — decode happens in `onExecute`, not cached from `onResize`). This compounds with the multi-input conv's own per-resize weight re-packing. Pre-existing v1.0 behavior, unchanged by this milestone, but worth surfacing since it directly affects how "performance-competitive" any E2E validation in step 6 will look.
- **`dims` must be the 4D conv-weight shape, not `[O, I]`.** The schema comment's example (`schema/default/CaffeOp.fbs:121`, `"e.g. [O, I]"`) undersells what `ShapeConvolution.cpp:33-38` actually needs from `inputs[1]`: `length(0)=O`, `length(2)=KH`, `length(3)=KW` (i.e. `[O, I, KH, KW]`, standard MNN conv-weight layout). The graph-rewrite pass must populate `SGFP4DequantParamT.dims` accordingly, not merely `[outputCount, inputCount]`.

## Sources

- Direct repository inspection (this checkout, branch `MNN_Ultra_v2`) — no external documentation lookup was performed; this is a closed-repository architecture question about MNN's own converter internals.
- `tools/converter/source/common/WeightQuantAndCoding.cpp`, `RemoveParams.cpp`, `writeFb.cpp`
- `schema/default/CaffeOp.fbs` (`Convolution2D`, `IDSTQuan`, `SGFP4DequantParam`)
- `schema/generate.sh`
- `include/MNN/SGFP4DequantUtils.hpp`
- `source/shape/ShapeSGFP4Dequant.cpp`, `source/shape/ShapeConvolution.cpp`
- `source/backend/cpu/CPUSGFP4Dequant.{hpp,cpp}`, `source/backend/cpu/compute/ConvolutionFloatFactory.cpp`, `source/backend/cpu/CPUDeconvolution.cpp`, `CPUDeconvolutionDepthwise.cpp`, `CPUConvolutionDepthwise.cpp`
- `source/backend/vulkan/{image,buffer}/execution/VulkanConvolution*.cpp`, `VulkanDeconvolution*.cpp`
- `tools/fp4/encode_sgfp4.py` (reference encoder, to be ported)
- `tools/converter/source/optimizer/PostConverter.cpp`, `PostTreatUtils.hpp`, `postconvert/SplitBlockQuantConvolution.cpp`, `postconvert/TransformGroupConvolution.cpp`, `merge/ConvertMatMulToConv2D.cpp`
- `tools/converter/include/config.hpp`, `tools/converter/source/common/cli.cpp`
- `test/op/SGFP4DequantTest.cpp` (op-level hand-construction precedent)
- `.planning/PROJECT.md`, `.planning/workstreams/sgfp4-pivot/STATE.md`, `ROADMAP.md`

---
*Architecture research for: MNN SGFP4 v2 converter integration (sgfp4-pivot v2.0)*
*Researched: 2026-08-25*
