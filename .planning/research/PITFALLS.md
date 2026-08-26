# Pitfalls Research

**Domain:** Native converter-side adaptive weight quantization (SGFP4 v2 quadtree container) added to an existing ONNX→MNN conversion pipeline
**Researched:** 2026-08-25
**Confidence:** HIGH (grounded in direct reads of `tools/converter/source/common/{RemoveParams,WeightQuantAndCoding,writeFb,cli}.cpp`, `source/core/{Interpreter,Pipeline,OpCommonUtils,ConvolutionCommon}.cpp`, `source/backend/cpu/CPUSGFP4Dequant.cpp`, `include/MNN/SGFP4DequantUtils.hpp`, `tools/fp4/encode_sgfp4.py`, `schema/default/{MNN,CaffeOp}.fbs`, and `.planning/workstreams/milestone/STATE.md`); MEDIUM on real-weight-distribution predictions since no real model has been run through the pipeline yet (that gap is itself the top pitfall below)

## Critical Pitfalls

### Pitfall 1: Quadtree/macroblock tiling assumptions break on real conv weight shapes

**What goes wrong:**
`tools/fp4/encode_sgfp4.py` hard-codes `MACROBLOCK_EDGE = 64` with the comment "every uniform layout tiles one 64x64 macroblock exactly" — the container format has no documented notion of a ragged/partial macroblock. Synthetic test fixtures were almost certainly authored (or generated) as exact multiples of 64 in each tiled dimension. Real `Convolution2D`/`Deconvolution` weight tensors reshape to `[OC, IC*KH*KW]`, and real `ConvolutionDepthwise` weights reshape to `[channel_multiplier*group, KH*KW]` — neither dimension is generally a multiple of 64 (e.g. a 3x3 depthwise layer has `KH*KW = 9`; a 37-output-channel conv has `OC = 37`). The encoder will either crash on non-tileable shapes, silently truncate the tail, or (if padding is added) leak un-initialized/zero padding into the container that the consuming op must know to slice back off.

**Why it happens:**
The encoder and its policy constants were built and tuned exclusively against Phase 2's synthetic round-trip fixtures (`SGFP4DequantFixtures.h`, emitted via `--emit-cpp-fixture`), which by construction are square, power-of-two-friendly test tiles. Nobody has yet run a real `OC x (IC*KH*KW)` matrix — with `ConvolutionDepthwise`'s especially tiny second dimension — through the tiling logic.

**How to avoid:**
Before writing any converter integration code, explicitly define and implement the padding/remainder policy for macroblocks that don't divide evenly: decide whether to zero-pad each dimension up to the next multiple of 64 (simplest, but wastes container space badly for tiny depthwise kernels) or to fall back to `LAYOUT_UNIFORM_8`/`LAYOUT_FULL_4X4` for undersized dimensions instead of forcing a 64-wide macroblock. Explicitly unit-test all three target op shapes (`Convolution2D` typical `OC x (IC*KH*KW)`, `ConvolutionDepthwise` with `KH*KW` as small as 9, `Deconvolution`'s transposed weight layout) against the *actual* MNN weight-tensor layout, not a square placeholder.

**Warning signs:**
Encoder throws on shape assertions, or silently produces containers whose decoded element count (`outElementCount`, product of `dims`) doesn't match `OC*IC*KH*KW` — this mismatch will only surface as a shape-inference or numerical-mismatch failure much later, at inference time.

**Phase to address:**
The real-weight encoder adaptation phase (turning the synthetic-fixture Python encoder into one that consumes actual `Convolution2D`/`ConvolutionDepthwise`/`Deconvolution` weight tensors) — this must be resolved before any RemoveParams.cpp wiring work starts, since the container's byte layout depends on it.

---

### Pitfall 2: Quadtree split policy is `[ASSUMED]`-tagged and tuned for smooth synthetic content, not i.i.d.-noise-like trained weights

**What goes wrong:**
`encode_sgfp4.py` marks its own adaptive-splitting constants as guesses: `LEVEL_THRESHOLDS` ("[ASSUMED] geometric interpolation... per-element MSE"), `VETO_FACTOR = 3.0` ("[ASSUMED] A3: ternary outlier veto multiplier"), `HYSTERESIS_DELTA = 0.05` ("[ASSUMED] A4"). These were calibrated (if at all) against synthetic ramp/constant tiles, which have strong local spatial smoothness — exactly the case a quadtree encoder is designed to exploit well. Real trained conv weights are much closer to a heavy-tailed, roughly-i.i.d. distribution at fine granularity with no spatial smoothness in the `[OC, IC*KH*KW]` reshape (there's no reason adjacent output channels or adjacent kernel taps should be numerically similar). Against real weights, the encoder is likely to split every macroblock all the way down to 4x4 leaves everywhere, which (a) is far slower than expected (recursive per-level MSE search over real-sized tensors, not toy tiles) and (b) produces a *worse* compression ratio than a flat `LAYOUT_UNIFORM_8`/`UNIFORM_4` layout would, because the 12-byte split map plus extra per-leaf headers add overhead that isn't amortized when nearly everything ends up as a leaf anyway.

**Why it happens:**
Quadtree/wavelet-style adaptive encoders are built around spatial locality assumptions that hold for images/ramps but not for weight matrices reshaped from `[OC, IC, KH, KW]` — there is no "image" in a conv weight tensor, so the encoder's core premise (some macroblocks are locally smooth and can collapse to `LAYOUT_UNIFORM_64`) may simply not apply to real weights the way it applied to synthetic fixtures.

**How to avoid:**
Before wiring this into mnnconvert, run the real-weight encoder against actual weight tensors from at least one real model (see Pitfall 4) and measure: (a) resulting layout-enum distribution (how often does anything collapse above `UNIFORM_8`?), (b) achieved bytes-per-weight vs. plain int8/int4 quant, (c) encode wall-clock time at model scale. If real weights pathologically over-split, either recalibrate the `[ASSUMED]` thresholds against real weight statistics, or add an explicit "pick cheapest of {flat-uniform, quadtree}" comparison per macroblock rather than always defaulting to the adaptive path.

**Warning signs:**
Encoded model is larger than the equivalent int4/int8-quantized model, or conversion of a real model takes dramatically longer than the synthetic-fixture unit tests suggested it should.

**Phase to address:**
The real-weight encoder adaptation phase — validate against real weight statistics, not just round-trip correctness, before declaring the encoder "done."

---

### Pitfall 3: Scale-search granularity and mode selection tuned on smooth data can miscalibrate on real per-channel outliers

**What goes wrong:**
The affine mode-selection search (`S_SEARCH_CANDIDATES = 16` candidates spanning `[0.5, 1.5] * max|w| / 7`, `MODE_SELECT_EPS = 0.10`, ternary outlier veto `VETO_FACTOR = 3.0`) assumes the per-leaf `max|w|` is a reasonable anchor for the whole block. Real conv weights — especially the first/last layers of a network, or depthwise layers where each channel can have a wildly different magnitude — routinely contain per-channel outliers 5-10x the bulk of the distribution. A 16-point linear search around a single outlier-dominated `max|w|` can produce a scale that clips the bulk of "normal" weights into 1-2 effective codes, destroying accuracy even though the round-trip byte-exactness tests (which only check decode(encode(x)) == x for the *specific* value chosen, not perceptual/numerical quality against ideal quantization) pass cleanly.

**Why it happens:**
This project already has a proven precedent for exactly this class of bug: the sibling Ultra FP4 (E2M1) work shipped with `MAX_E2M1_VALUE` set to `6.0` instead of the spec-correct `3.0` (quick task `260821-rql`), a scale-calibration defect that silently corrupted a live downstream consumer (`SuperGenius/SGProcessingManager`) and was only caught by a dedicated regression test, not by the existing round-trip/decode tests. The same failure mode — a scale constant that's fine for a narrow test distribution but wrong in general — is structurally likely to recur here since the search/veto constants are self-admittedly "exemplary," not derived from the spec's own worked numeric example.

**How to avoid:**
Treat scale/mode-selection calibration as a first-class deliverable with its own accuracy regression test (e.g. weight-reconstruction MSE / max-abs-error against real per-channel weight distributions, not just structural round-trip), independent of the container-format round-trip tests that already exist. Explicitly test layers with known outlier-heavy channels (depthwise, first conv layer typical of vision models) before considering the encoder validated.

**Warning signs:**
Per-leaf/per-channel reconstruction error is much higher for real weights than the synthetic-fixture error bars suggested; end-to-end model accuracy (not just per-tensor MSE) degrades noticeably more than the equivalent int4/int8 weight-quant path.

**Phase to address:**
The real-weight encoder adaptation phase, verified again in the end-to-end validation phase (accuracy check, not just "the model runs").

---

### Pitfall 4: `WeightQuantAndCoding` still mutates the same Convolution2D/Deconvolution/ConvolutionDepthwise ops unless explicitly excluded

**What goes wrong:**
`tools/converter/source/common/writeFb.cpp`'s `_postTreatOp()` calls `WeightQuantAndCoding(op, config, &context)` **unconditionally** on every `Convolution`/`ConvolutionDepthwise`/`Deconvolution`/`DeconvolutionDepthwise` op, immediately before `RemoveAndStoreParam(op, ...)`, regardless of whether a new SGFP4 flag was passed. Its only early-out is `if (param->quanParameter.get() != nullptr) return;` — a check keyed on the *legacy* int2-8 quant field, which an SGFP4-converted op will not have set (SGFP4 uses a dedicated `SGFP4DequantParam`/new op, not `quanParameter`). Worse, even at the flag's own default (`weightQuantBits == 0`), the function still calls `CommonCompute::compressFloatWeightToSparse(op.get())` on the op **before** checking whether `param->weight` is even non-empty. If the new SGFP4 flag doesn't explicitly prevent this legacy pass from touching SGFP4-targeted ops (or doesn't clear `param->weight` cleanly when moving weight data out to the new dequant op), the same tensor can be silently sparse-compressed or int8-quantized by the legacy path in addition to being SGFP4-encoded — two competing quantization passes racing on the same weight data.

**Why it happens:**
`WeightQuantAndCoding` predates SGFP4 and was never designed with a second, mutually-exclusive weight-quantization mechanism in mind; its guard clause protects against re-running itself, not against a sibling quantization system operating on the same op types.

**How to avoid:**
Add an explicit skip condition to `WeightQuantAndCoding` (or a guard in `_postTreatOp` before calling it) for any op that has already been claimed by the SGFP4 flag/pass — e.g. check for the new op type, or a sentinel already set on the op, before entering the function body at all. Do not rely on `param->weight.size() == 0` as an implicit guard unless the SGFP4 pass is verified to *always* empty `param->weight` for every op it touches, on every code path (including early-return/error paths).

**Warning signs:**
A model converted with the new SGFP4 flag ends up with `Convolution2D` ops that carry *both* a stray `quanParameter`/`sparseParameter` and SGFP4 external data; converted model size doesn't match the expected SGFP4 container size; decode-time shape/size assertions in `CPUSGFP4Dequant`/`VulkanSGFP4Dequant` fail on a real-model conversion despite passing on synthetic fixtures.

**Phase to address:**
The CLI flag + graph-rewrite integration phase — write an explicit interaction test (`--weightQuantBits N` + new SGFP4 flag together, and SGFP4 flag alone at every `weightQuantBits` default) before declaring the flag done.

---

### Pitfall 5: Op-insertion graph surgery happens too late if done inside the per-op `postTreat` loop

**What goes wrong:**
Per `writeFb.cpp`, `postTreat()` iterates `netT->oplists` (and each subgraph's `nodes`) **once**, calling `_postTreatOp()` per existing op — `loadExternalParam` → `WeightQuantAndCoding` → `RemoveAndStoreParam`, all against the *same, already-existing* op object. This loop has no facility for inserting brand-new ops into the list mid-iteration. SGFP4, per the locked design (`OpType_SGFP4Dequant = 605` / `SGFP4DequantParam`, a *dedicated new op*, not an in-place field mutation like `quanParameter`), requires actual graph surgery — splitting a single `Convolution2D` node into two nodes (a new `SGFP4Dequant` producer feeding the existing conv's weight as a runtime input) with new tensor indices and rewired `inputIndexes`. Attempting this rewrite from inside `_postTreatOp`/`RemoveAndStoreParam` will either corrupt the topology (iterating a container being mutated), silently do nothing (because "add a case to `RemoveAndStoreParam`'s switch" only handles *serializing an existing op's* fields, it cannot conjure a new op node), or bypass earlier passes (shape inference, op fusion) that never see the new op because it didn't exist when they ran.

**Why it happens:**
The existing int2-8 weight-quant precedent (`WeightQuantAndCoding`) sets a mental model of "quantization = mutate this op's fields in place," which this milestone's own design explicitly departs from ("op-type rewrite during conversion"). It's easy to reach for the same `_postTreatOp`/`RemoveAndStoreParam` insertion point out of habit when the actual graph-rewrite must happen as an earlier, separate optimizer pass (comparable to `tools/converter/source/optimizer/merge/` passes, which run and rewrite the graph before `postTreat`/`writeFb` serialize it).

**How to avoid:**
Implement the Convolution→(SGFP4Dequant + Convolution-with-weight-input) rewrite as its own optimizer pass registered in the normal optimizer pipeline (pattern-matched against `tools/converter/source/optimizer/merge/` or `postconvert/` precedent), running strictly before `postTreat()`. Only add the `OpParameter_SGFP4DequantParam` case to `RemoveAndStoreParam`'s switch to serialize the *already-inserted* op's external payload — do not try to make that switch case responsible for creating the op.

**Warning signs:**
`RemoveAndStoreParam`'s new case never actually triggers on a converted model (because the op it expects to see, `OpType_SGFP4Dequant`, was never inserted into `oplists`); shape inference or op-fusion optimizer passes throw on the rewired graph because they ran before the rewrite and hold stale assumptions about `Convolution2D` always owning its own weight.

**Phase to address:**
The CLI flag + graph-rewrite integration phase — the rewrite pass must exist and be verified (dump the graph and inspect node/tensor topology) before wiring `RemoveAndStoreParam`.

---

### Pitfall 6: External sidecar must stay a single, centrally-offset-tracked file — a parallel/independent writer causes real offset collisions

**What goes wrong:**
`saveExternalData()` opens exactly one `std::ofstream` for `<model>.mnn.weight` and threads a single `int64_t offset` by reference through every op's `RemoveAndStoreParam()` call, in file order, so all ops (Convolution2D, Scale, LayerNorm, Blob, and — once added — SGFP4) share one monotonically-increasing offset space with no gaps. If the SGFP4 integration is implemented as anything other than "one more `case` inside the same `RemoveAndStoreParam` switch, using the same `storeWeight`-style pattern against the same `fs`/`offset` reference" — e.g. a separate write pass that opens its own file handle, or that maintains its own local offset counter — two failure modes become likely: (a) a second sidecar file (`.sgfp4.weight` or similar) that breaks the runtime's implicit convention of computing the sidecar path as `<model_path> + ".weight"` (`Interpreter.cpp`: `net->externalFile = std::string(file) + ".weight";` — there is no per-op sidecar-path field read at load time in the primary path), or (b) two writers racing on the same file/offset counter, producing overlapping `[offset, size]` ranges that silently read garbage for one of the two ops sharing the collision.

**Why it happens:**
`storeWeight<T>()`'s templated `std::vector<T>` swap/clear pattern doesn't fit SGFP4's payload, which is a raw pre-encoded byte blob (not a `std::vector<float>`/`std::vector<int8_t>` MNN already owns per-field) — it's tempting to write a bespoke helper for it rather than adapting it to the existing `storeWeight` call convention, and a bespoke helper is where independent-offset bugs creep in.

**How to avoid:**
Write the new case to call the same `storeWeight`-shaped helper (or a trivially-adapted overload taking `const uint8_t*`/`size_t` instead of a `std::vector<T>&`) against the *same* `fs`/`offset` parameters already passed into `RemoveAndStoreParam`. Never open a second output file for SGFP4 payloads, and never introduce a second offset counter.

**Warning signs:**
A converted model has a `.mnn.weight` file whose size doesn't match the sum of all ops' recorded `external[1]` sizes; two SGFP4-quantized layers in the same model produce identical or overlapping decoded content; models with only one SGFP4 layer work but models with several corrupt weights on layers other than the first.

**Phase to address:**
The RemoveParams.cpp/sidecar-wiring phase — add a multi-layer (≥3 SGFP4-quantized ops in one model) test specifically to catch offset accumulation bugs; a single-layer test cannot detect this class of bug.

---

### Pitfall 7: A second, separate runtime-side external-data switch statement also needs an SGFP4 case (MNN_LOW_MEMORY / mmap path)

**What goes wrong:**
`source/core/OpCommonUtils.cpp`'s `createExecutionWithExternal()`/`_RebuildExternalOp()` is a **second**, independent switch statement (only `Convolution2D`, `Scale`, `LayerNorm` today) that gates MNN's low-memory `useCachedMmap` external-loading optimization — a real project build option (`-DMNN_LOW_MEMORY=ON` is in this repo's standard build command per `CLAUDE.md`). It's easy to believe "the converter writes the sidecar, the CPU/Vulkan `Execution` reads `externalPath`/`external`, therefore external-data support is complete" once `RemoveParams.cpp` has a case and `CPUSGFP4Dequant`/`VulkanSGFP4Dequant` already read `mOp->externalPath()` directly — but that second switch is a distinct code path specifically for the mmap/low-memory loading mode, and omitting `SGFP4Dequant` from it means models converted with the new flag either silently skip the memory-mapped fast path (functionally correct but defeats the point of `MNN_LOW_MEMORY`) or, depending on how the fallback behaves for unrecognized op types under `usemmap==true`, may not be exercised at all in that build configuration.

**Why it happens:**
This switch lives in `source/core/` (runtime), physically and mentally far from `tools/converter/source/common/RemoveParams.cpp` (convert-time) — someone focused on "does the flag produce a working `.mnn` file" is very unlikely to think to grep for every other switch statement keyed on `OpParameter_Convolution2D`/`OpParameter_Scale` across the whole runtime.

**How to avoid:**
Before declaring the converter integration complete, grep the whole `source/` tree (not just `tools/converter/`) for every switch/dispatch keyed on `OpParameter_Convolution2D` or `USE_EXTERNAL_DATA(...)` and audit each one for whether it needs an `SGFP4DequantParam`/`OpType_SGFP4Dequant` case. Explicitly test the model with `-DMNN_LOW_MEMORY=ON` and `useCachedMmap > 1` set (the project's own standard build flag), not just the default build.

**Warning signs:**
End-to-end inference works in a normal build but behaves differently (or crashes) only when `MNN_LOW_MEMORY`/`useCachedMmap` is enabled — a configuration this project explicitly ships (`CLAUDE.md`'s documented build command includes `-DMNN_LOW_MEMORY=ON`).

**Phase to address:**
The end-to-end validation phase — include an `MNN_LOW_MEMORY=ON` build/run leg in the validation matrix, not just the default build.

---

### Pitfall 8: `op->externalPath` can leak an absolute build-machine path into the shipped model if the SGFP4 pass ever needs to re-read already-externalized data

**What goes wrong:**
There is an existing precedent in this exact codebase for baking an absolute, machine-specific path into `op->externalPath`: `tools/converter/source/optimizer/postconvert/SplitBlockQuantConvolution.cpp` sets `op->externalPath = config->modelFile + ".weight";` (using whatever path the user passed on the CLI, which is very often an absolute local path) so that pass can re-read already-serialized external data mid-pipeline, then clears it again afterward. If SGFP4's graph-rewrite pass (Pitfall 5) or its RemoveParams wiring ever needs a similar "read back data I already wrote earlier in the pipeline" step and copies this pattern without also copying the corresponding `.clear()`, a Windows-specific absolute path (drive letter, backslashes) — or any developer's local absolute path — gets embedded into the final `.mnn` flatbuffer and shipped. This is both a portability bug (the path is meaningless on a different machine/OS) and a minor information-disclosure issue (leaks local directory structure/usernames into a distributed model artifact).

**Why it happens:**
`RemoveAndStoreParam`/`loadExternalParam`'s own early-out (`if (!op->externalPath.empty()) { return; }` / `.clear()` at the end) makes it look like the field is always transient, but that invariant is only true if every code path that *sets* `externalPath` also *clears* it before final `Net::Pack` — which is easy to miss when adding a new pass that touches the field for the first time on this project's given platform (Windows dev environment per this repo's environment, versus MNN's more common Linux/macOS CI).

**How to avoid:**
If the SGFP4 pass needs to re-read external data mid-pipeline, mirror `SplitBlockQuantConvolution.cpp`'s pattern exactly, including its `.clear()` on every exit path (success and error). Add a converter-output check (or a unit test) that asserts no op in the final serialized `.mnn` has a non-empty `externalPath` string containing a filesystem separator or drive letter.

**Warning signs:**
A `.mnn` file produced on one developer's Windows machine fails to load (or loads but can't find its weight file) when copied to another machine or a Linux CI runner; `strings` on a shipped `.mnn` file reveals a local path.

**Phase to address:**
The RemoveParams.cpp/sidecar-wiring phase — add the "no absolute paths in the final model" assertion as part of that phase's own verification, since it's cheap to check and easy to silently regress later.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|-----------------|------------------|
| Zero-pad every macroblock dimension up to 64 instead of adding a small-tensor fallback layout | Fastest to implement, one code path | Depthwise/small-conv layers bloat massively (e.g. a 9-element `KH*KW` row padded to 64 is 7x overhead) | Only for a first correctness-only spike; must be replaced before real-weight validation (Pitfall 1) |
| Reuse `WeightQuantAndCoding`'s existing `quanParameter != nullptr` guard as the SGFP4 exclusion check instead of adding an explicit SGFP4-aware guard | No new guard code to write/test | Silent double-quantization races (Pitfall 4) that only surface on real models, not synthetic fixtures | Never — the guard checks the wrong field for SGFP4 |
| Skip auditing the `OpCommonUtils.cpp` mmap-path switch (Pitfall 7) because "the decode Execution already reads external data directly" | Ships faster | `MNN_LOW_MEMORY=ON` builds silently lose the mmap optimization or misbehave for SGFP4 models | Never for a production-facing milestone; acceptable only for a throwaway spike explicitly marked "CPU-only, no mmap" |
| Hard-code the `[offset, size]` quadtree container as always-uncompressed/always-full-precision fallback if the adaptive encoder over-splits on real weights (Pitfall 2), rather than fixing the split thresholds | Unblocks end-to-end pipeline quickly | Defeats the entire point of SGFP4 (compression ratio); ships a feature that's technically correct but not valuable | Acceptable as a temporary escape hatch behind a debug flag, not as the shipped default |

## Integration Gotchas

Common mistakes when connecting to existing converter/runtime machinery.

| Integration | Common Mistake | Correct Approach |
|-------------|-----------------|-------------------|
| `RemoveParams.cpp`'s `RemoveAndStoreParam` switch | Treating the new case as "the place graph rewrite happens" | It only serializes an *already-inserted* op's fields; graph rewrite (new op insertion) is a separate, earlier optimizer pass (Pitfall 5) |
| `WeightQuantAndCoding` unconditional per-op call in `_postTreatOp` | Assuming the new SGFP4 flag automatically suppresses the legacy int2-8 pass | Must add an explicit skip condition; the existing `quanParameter != nullptr` guard doesn't cover SGFP4 (Pitfall 4) |
| Sidecar file writing (`saveExternalData`) | Writing SGFP4 payloads via a separate file handle/offset counter | Reuse the single shared `ofstream`/`offset` reference already threaded through every op (Pitfall 6) |
| `--weightQuantBits` CLI flag | Assuming cxxopts / `cli.cpp` already validates mutually-exclusive flag combinations | This codebase has **no** existing cross-flag conflict validation pattern (e.g. `--hqq` just silently forces `weightQuantAsymmetric=true` rather than erroring on conflicting explicit flags) — the new SGFP4 flag needs its own explicit validation code, not an assumed convention |
| Runtime mmap/low-memory external loading (`OpCommonUtils.cpp`) | Assuming "external data support" is a single integration point | It's two independent switch statements in two different subsystems (convert-time `RemoveParams.cpp`, runtime `OpCommonUtils.cpp`) that must both be updated (Pitfall 7) |

## Performance Traps

Patterns that work at small scale (synthetic fixtures / single-layer tests) but fail against real models.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|-----------------|
| Recursive quadtree MSE search re-run per macroblock with no early-exit/caching | Conversion time acceptable on a handful of synthetic 64x64 tiles, but scales poorly across a real model with hundreds of conv layers each reshaping to large `[OC, IC*KH*KW]` matrices | Benchmark encode time against a real model early; add early-exit once a level's error is clearly below threshold rather than always recursing to the floor | Any real model beyond a couple of layers — quadtree recursion cost grows with tensor size, synthetic fixtures never exercised this |
| `_largeModel()`/`_computeOpExternalSizeInMB()` 2GB auto-external-data heuristic missing the new `OpParameter_SGFP4DequantParam` case | A model with many large SGFP4-quantized layers doesn't trigger auto external-data mode when `--saveExternalData` isn't explicitly passed, producing an oversized single `.mnn` file instead of the intended sidecar split | Add `OpParameter_SGFP4DequantParam` to `_computeOpExternalSizeInMB`'s switch alongside `Convolution2D`/`Blob` | Any real model relying on the auto-detection path rather than always passing `--saveExternalData` explicitly |

## Security Mistakes

Domain-specific issues beyond general web security (this is a local file-format/converter, not a networked service, but the sidecar is untrusted input at load time).

| Mistake | Risk | Prevention |
|---------|------|------------|
| Trusting the container's internal `size`/offset fields without bounding against the real on-disk sidecar size | A crafted or corrupted `.mnn.weight` could force an oversized allocation (the exact class of bug already fixed once in `CPUSGFP4Dequant.cpp`'s `queryFileSize`/bounds-check comment, `T-01-04`) | Any new read path added for the converter-side re-load (Pitfall 8's "read back already-written data" case) must apply the same bounds-check discipline already established in `CPUSGFP4Dequant.cpp`, not skip it because "the converter wrote this file itself, so it must be trusted" |
| Embedding absolute developer-machine paths in `externalPath` (Pitfall 8) | Minor info-disclosure (local directory structure, usernames) shipped inside a distributed model artifact | Explicit "no absolute paths in final model" check as part of converter-output verification |

## UX Pitfalls

Common CLI/converter user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-------------------|
| New SGFP4 flag silently no-ops (or silently falls back to float/int8) when the target ops don't exist in the model, or when combined with an incompatible flag | User believes conversion succeeded with SGFP4 quantization applied when it didn't; discovered only at inference time via unexpectedly large model size or wrong op types | Print an explicit summary at the end of conversion (mirroring the existing `_largeModel`/`MNN_PRINT("Save Weight to %s\n", ...)` pattern) stating how many ops were SGFP4-quantized, and hard-error (not silently ignore) on genuinely conflicting flag combinations |
| Two "FP4" flags in `--help` output (existing `tools/fp4/quantize_fp4.py`-driven Ultra FP4 workflow vs. the new native SGFP4 mnnconvert flag) with no clear naming distinction | Users conflate the two unrelated FP4 formats and pass the wrong one, or file bugs against the wrong workstream | Name the new flag unambiguously (e.g. `--sgfp4QuantV2`, not a generic `--fp4Quant`) and cross-reference both formats' docs so the difference is discoverable from `--help` text alone |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **RemoveParams.cpp case added:** Often missing the corresponding graph-rewrite pass that actually inserts the `SGFP4Dequant` op into `oplists` in the first place — verify by dumping the converted graph's op list, not just checking the switch statement compiles.
- [ ] **CLI flag wired:** Often missing explicit mutual-exclusion/interaction handling with `--weightQuantBits`/`--hqq`/`--weightQuantAsymmetric` — verify by running the new flag together with each existing weight-quant flag and confirming a clear error or well-defined precedence, not undefined double-processing.
- [ ] **Decode path "already works" (v1.0 shipped):** Often mistaken for "converter integration is therefore low-risk" — verify by actually running a real model through pytorch→onnx→mnnconvert(SGFP4 flag)→`.mnn`→inference, since v1.0 was validated only against directly-constructed synthetic containers fed straight to the `Execution`, never through the converter.
- [ ] **External sidecar wiring:** Often missing a multi-layer test (≥3 SGFP4-quantized ops sharing one sidecar file) — a single-layer test cannot catch offset-accumulation bugs (Pitfall 6).
- [ ] **MNN_LOW_MEMORY compatibility:** Often missing entirely — verify by building and running with `-DMNN_LOW_MEMORY=ON` and `useCachedMmap > 1`, the project's own documented standard build flag (Pitfall 7).
- [ ] **No absolute paths in shipped model:** Often missing — verify by inspecting `externalPath` on every op in the final serialized `.mnn` (Pitfall 8).

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|-----------------|------------------|
| Macroblock tiling breaks on real shapes (Pitfall 1) | MEDIUM | Add explicit padding/fallback-layout logic; re-run the container round-trip self-test against the newly-added real-shape test cases; no format/schema change needed since `dims` already carries true geometry |
| Quadtree over-splits on real weights (Pitfall 2) | MEDIUM | Add a "compare total bytes across candidate layouts, pick cheapest" step to the encoder; no decode-side change needed since decode already handles all layout enums |
| Legacy `WeightQuantAndCoding` double-processes SGFP4-targeted ops (Pitfall 4) | LOW | Add the missing skip condition; re-convert affected models — no schema change, purely a converter-side ordering fix |
| Sidecar offset collision from a rogue independent writer (Pitfall 6) | HIGH | Requires re-converting every model produced with the buggy code path (silently corrupted weight data is not detectable after the fact without a checksum); add a post-write validation pass (sum of recorded external sizes == sidecar file size) to catch this class of bug going forward |
| Absolute path leaked into shipped model (Pitfall 8) | LOW | Re-convert affected models with the `.clear()` fix; add the "no absolute paths" converter-output assertion so it can't regress silently again |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|--------------------|-----------------|
| 1. Macroblock tiling breaks on real shapes | Real-weight encoder adaptation | Unit tests against actual `OC x (IC*KH*KW)` / `ConvolutionDepthwise` / `Deconvolution` shapes, not square placeholders |
| 2. Quadtree split policy mistuned for real distributions | Real-weight encoder adaptation | Measure layout-enum distribution + compression ratio against a real model's weights, not just round-trip byte-exactness |
| 3. Scale/mode-selection miscalibration on outliers | Real-weight encoder adaptation, re-checked in end-to-end validation | Weight-reconstruction MSE regression test on outlier-heavy real layers (depthwise, first conv) |
| 4. `WeightQuantAndCoding` double-processes SGFP4 ops | CLI flag + graph-rewrite integration | Explicit test: SGFP4 flag combined with every `--weightQuantBits` value including default (0) |
| 5. Graph surgery done too late (inside postTreat loop) | CLI flag + graph-rewrite integration | Dump and inspect converted graph topology (op list, tensor indices) before/after the rewrite pass |
| 6. Sidecar offset collisions from independent writer | RemoveParams.cpp/sidecar-wiring | Multi-layer (≥3 SGFP4 ops) conversion + sidecar-size-sum validation |
| 7. Runtime mmap-path switch missing SGFP4 case | End-to-end validation | Build and run with `-DMNN_LOW_MEMORY=ON`, `useCachedMmap > 1` |
| 8. Absolute path leaks into shipped model | RemoveParams.cpp/sidecar-wiring | Assert no `externalPath` in final `.mnn` contains a path separator/drive letter |

## Sources

- `tools/converter/source/common/RemoveParams.cpp` (direct read — sidecar write/read mechanics, existing fragile `external.size() != 3` arity guard)
- `tools/converter/source/common/WeightQuantAndCoding.cpp` (direct read — legacy quant pass, `weightQuantBits`/HQQ interaction, unconditional per-op mutation)
- `tools/converter/source/common/writeFb.cpp` (direct read — `postTreat`/`_postTreatOp` per-op loop ordering, `_largeModel`/`_computeOpExternalSizeInMB` 2GB heuristic)
- `tools/converter/source/common/cli.cpp` (direct read — `--weightQuantBits`/`--weightQuantAsymmetric`/`--hqq`/`--weightQuantBlock` flag definitions and lack of cross-flag validation)
- `tools/converter/source/optimizer/postconvert/SplitBlockQuantConvolution.cpp` (direct read — existing precedent for `op->externalPath` absolute-path assignment/clear pattern)
- `source/core/Interpreter.cpp`, `source/core/Pipeline.cpp` (direct read — runtime sidecar path convention: `<model_path> + ".weight"`, computed at load time)
- `source/core/OpCommonUtils.cpp` (direct read — second, independent `createExecutionWithExternal`/`_RebuildExternalOp` switch gating `MNN_LOW_MEMORY`/mmap loading)
- `source/backend/cpu/CPUSGFP4Dequant.cpp`, `include/MNN/SGFP4DequantUtils.hpp` (direct read — shipped v1.0 decode path, existing bounds-check discipline for external size)
- `tools/fp4/encode_sgfp4.py` (direct read — synthetic-fixture reference encoder, `[ASSUMED]`-tagged split-policy constants, `MACROBLOCK_EDGE = 64` tiling assumption)
- `schema/default/MNN.fbs`, `schema/default/CaffeOp.fbs` (direct read — `OpType_SGFP4Dequant = 605`, `SGFP4DequantParam` table definition)
- `.planning/workstreams/milestone/STATE.md` (direct read — confirms sibling Ultra FP4/E2M1 workstream's Phase 4 conversion pipeline currently integrates via an external Python script driving `MNNConvert -f MNN/-f JSON` round-trips against `OpType_Dequantize`, not native `RemoveParams.cpp`/`WeightQuantAndCoding.cpp`/`cli.cpp` code — low direct code-collision risk today, but shared `schema/default/*.fbs` enum ID space and CLI-flag-naming confusion remain real)
- `.planning/PROJECT.md` (project context — locked decisions on target op set, external-sidecar container format, and the sibling workstream's non-conflicting scope)
- Precedent bug: quick task `260821-rql` (`MAX_E2M1_VALUE` scale-calibration defect in the sibling Ultra FP4 work, `tools/fp4/quantize_fp4.py`) — cited as direct evidence that scale/calibration-constant bugs of exactly the kind flagged in Pitfall 3 have already occurred once in this codebase and silently corrupted a live downstream consumer

---
*Pitfalls research for: SGFP4 v2 real-weight converter integration (mnnconvert)*
*Researched: 2026-08-25*
