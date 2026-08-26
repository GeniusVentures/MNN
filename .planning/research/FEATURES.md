# Feature Research

**Domain:** Converter-embedded, adaptive-block weight-only quantization (native `mnnconvert` CLI flag for SGFP4 v2)
**Researched:** 2026-08-25
**Confidence:** MEDIUM (codebase-derived findings HIGH — read directly from `tools/converter/source/common/`; ecosystem-pattern findings MEDIUM — cross-referenced web sources on llama.cpp/GGUF and general WOQ literature, no single authoritative spec for "the" converter-embedded pattern)

## Feature Landscape

### Table Stakes (Users Expect These)

Features a native SGFP4 flag must have to be usable and consistent with MNN's existing `--weightQuantBits` quantization UX. Missing these = the flag feels like a bolted-on experiment, not a first-class quant mode.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| CLI flag peer to `--weightQuantBits` (e.g. `--weightQuantSGFP4` / `--sgfp4Quant`), parsed via `cxxopts` in `cli.cpp` and stored on `modelConfig` (`config.hpp`) | Every existing MNN quant mode (`--weightQuantBits`, `--weightQuantAsymmetric`, `--weightQuantBlock`, `--hqq`) follows this exact registration pattern; users and downstream tooling (`auto_quant.py`) expect new quant flags to look identical | LOW | Mechanical: add `bool`/flag field to `modelConfig` (config.hpp:42-49) + `cxxopts` option block (cli.cpp:208-228) + a `result.count(...)` handler (cli.cpp:477-503) |
| Op-type dispatch limited to `Convolution2D` / `ConvolutionDepthwise` / `Deconvolution` (the three types the milestone explicitly names) | Mirrors exactly which op types `WeightQuantAndCoding.cpp:75-84` already switches on for float-weight paths; matching this set keeps SGFP4 a drop-in alternative branch rather than a parallel dispatch mechanism | LOW-MEDIUM | Add SGFP4 branch inside (or beside) the existing `WeightQuantAndCoding` dispatch function, gated by the new flag, mutually exclusive with `weightQuantBits`/HQQ paths on the same op |
| Real-weight encoder invocation at convert time (not just synthetic fixtures) | `tools/fp4/encode_sgfp4.py` is currently a standalone Python test-oracle that only consumes hand-built fixture arrays; the milestone's whole point is feeding real `Convolution2D.weight` float arrays through the same encode math during conversion | MEDIUM-HIGH | `mnnconvert` is a C++ binary; the Python encoder cannot be `import`-ed into it. Requires either (a) porting the encoder's core math to C++ inside the converter, or (b) shelling out to Python as a conversion sub-step. Porting to C++ is the only option consistent with "native CLI flag" (no external Python dependency at convert time) |
| Op-type rewrite / discriminator so the runtime picks the SGFP4 decode `Execution` instead of the IDST int2-8 dequant path | The existing quant path signals "use int2-8 dequant" purely by `param->quanParameter` being non-null (`WeightQuantAndCoding.cpp:87`); SGFP4 needs an equivalent unambiguous signal (new discriminator field or op-type value) so `source/core/` execution selection routes to the already-built SGFP4 CPU/Vulkan `Execution`s | MEDIUM | Must not collide with existing `quanParameter`-based dispatch; the v1.0 decode ops already assume a specific container/param shape, so this is wiring, not new decode logic |
| External sidecar emission for `SGFP4DequantParam` via `RemoveParams.cpp` | `RemoveParams.cpp:36-52` already has the exact `{magic, offset, size}` sidecar precedent for `Convolution2D.quanParameter`/`weight`/`bias`; PROJECT.md explicitly calls out following this precedent | LOW-MEDIUM | Additive `case` in the `RemoveAndStoreParam` switch (store) and `loadExternalParam` switch (load) — same `storeWeight<T>`/`loadExternalData<T>` template helpers, no new sidecar mechanism needed |
| Bail-out / passthrough for layers that don't fit SGFP4's geometry (e.g. not tileable into 64x64 macroblocks, 1x1 kernels, tiny channel counts) | `WeightQuantAndCoding.cpp:172-179` already has this exact pattern for `--weightQuantBlock` (`MNN_PRINT("...don't use block-quant for the layer...")` and falls through to unquantized/default path); users expect quant flags to degrade gracefully per-layer, not hard-fail the whole conversion | MEDIUM | SGFP4's macroblock/quadtree geometry (64x64 base, recursion floor at 4x4) is stricter than `--weightQuantBlock`'s simple divisibility check, so more layers will need fallback |
| End-to-end validation via the **existing** `--testdir` / `--testconfig` / `--thredhold` mechanism (`Cli::testconvert`, cli.cpp:785-787) | `mnnconvert` already ships a built-in numeric-diff validator against reference input/output `.npy`-style fixtures at a configurable threshold — this is MNN's established "does the converted model still work" check, used for every conversion, not something to reinvent | LOW (harness exists) / MEDIUM (producing good real-model reference fixtures) | This is the validation dependency the milestone should reuse rather than build bespoke tooling — directly answers "what validation steps are expected" from the research question |

### Differentiators (Competitive Advantage)

Features that make SGFP4 more than "yet another `--weightQuantBits` mode" — this is where the quadtree-adaptive design actually earns its complexity budget.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Adaptive quadtree layout selection (LAYOUT_MIXED) reusing the encoder's per-level MSE-threshold + hysteresis policy, instead of a single fixed block size | This is the actual "quadtree-adaptive" value proposition named in the milestone/workstream title — a fixed block size (like `--weightQuantBlock`) can't locally adapt precision to weight-distribution hot spots the way recursive quadtree splitting can | HIGH | Policy already fully implemented and validated in `tools/fp4/encode_sgfp4.py` (`LEVEL_THRESHOLDS`, `VETO_FACTOR`, `HYSTERESIS_DELTA`, spec section 6.3) — the work is porting/wiring this logic into the C++ convert-time path, not re-deriving it |
| Dual-mode per-block selection (FP4_AFFINE vs T158_AFFINE, Eq. 5 + ternary outlier veto) | Lets each leaf block pick the code mode that best fits its local value distribution rather than a single global mode — directly improves accuracy vs. a fixed-format quant like plain int4 | HIGH (but pre-solved) | Also already implemented in the Python encoder; decode-side already supports both modes (v1.0 shipped both FP4_AFFINE and T158_AFFINE CPU+Vulkan). Risk is confined to the port, not the algorithm |
| Per-tensor / per-model compression-ratio and quant-error reporting, extending the existing `quantMutableInfo`/`compressionParamsFile` proto plumbing | `WeightQuantAndCoding.cpp:114-132` already emits per-layer bits/asymmetric/block-size info into a compression-params proto for `--weightQuantBits`; extending the same proto with SGFP4-specific fields (layout mix %, per-tensor MSE) gives users visibility without new tooling | MEDIUM | Nice-to-have parity feature; reuses infra rather than building a new report format |
| Validated on both CPU and Vulkan decode paths (not just CPU) | v1.0 already shipped CPU+Vulkan parity across 14 fixtures — an end-to-end real-model validation that exercises both backends is a stronger proof point than CPU-only and directly serves the sibling GeniusCogntiveSystem Vulkan-inference goal in PROJECT.md's Core Value | MEDIUM | Decode-side work is done; this is validation-harness scope (run `--testdir` style comparison against both CPU and Vulkan backend) |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|------------------|-------------|
| Calibration / activation-data requirement (GPTQ/AWQ-style: run forward passes over a calibration dataset to compute a Hessian or activation-importance matrix) | Common in the broader WOQ ecosystem (GPTQ, AWQ, llama.cpp's `--imatrix`) and often assumed to improve low-bit accuracy | Neither MNN's existing `--weightQuantBits`/`--hqq` path nor the SGFP4 v2 spec's adaptive quadtree policy requires activation data — both are purely weight-intrinsic (HQQ is explicitly calibration-free; the quadtree MSE thresholds operate on weight values only). Adding a calibration requirement breaks CLI-flag parity with `--weightQuantBits` (single-command, no dataset needed) and adds a data-dependency the milestone never asked for | Keep SGFP4 quantization purely weight-intrinsic, exactly as specced. If accuracy proves insufficient later, treat calibration as an explicit future differentiator, not baseline |
| Bespoke per-tensor accuracy-report CLI/dashboard (à la Intel Neural Compressor's WOQ reporting) | Appears in some ecosystem tools and looks like good "professional tooling" | Building new reporting infrastructure is scope creep for this milestone — the milestone's actual success criterion is "a real test model runs correct inference" (PROJECT.md), which the existing `--testdir`/`--thredhold` pass/fail check already answers | Reuse `Cli::testconvert`; only extend the `compressionParamsFile` proto (see Differentiators) if/when there's a concrete downstream consumer for richer reports |
| Supporting every conv-family op type, including `DeconvolutionDepthwise` and requantization from already-int8 sources (`ConvInt8`/`DepthwiseConvInt8`) | `--weightQuantBits` supports all of these (`WeightQuantAndCoding.cpp:80-84`), so "full parity" is tempting | The milestone explicitly scopes real-weight SGFP4 to `Convolution2D`/`ConvolutionDepthwise`/`Deconvolution` only (PROJECT.md Target Features); extending further multiplies the op-dispatch and fallback-geometry surface for op types nobody asked for in this milestone | Match the milestone's named 3 op types exactly; log the rest as explicit future work if a real model needs them |
| User-tunable adaptive-policy flags exposed directly on `mnnconvert` (first-class `--sgfp4Eps`, `--sgfp4LevelThresholds`, `--sgfp4VetoFactor`, `--sgfp4HysteresisDelta`) | The Python encoder already exposes `--eps`/`--level-thresholds`/`--veto-factor`/`--hysteresis-delta` as CLI knobs, so surfacing them on `mnnconvert` too seems like natural parity | Prematurely expanding the CLI surface before real-model validation shows the spec's exemplary defaults (section 6.3) don't generalize is speculative complexity; every additional knob is another thing the end-to-end validation step must cover | Ship v1 with spec defaults baked in (single on/off flag, no policy tuning); promote specific knobs to real flags only if real-model accuracy data demands it |
| Attestation / byte-exactness / verifiable-execution guarantees on the quantized output | Sounds like a natural "correctness" feature for a novel quant format | Already explicitly ruled out in PROJECT.md's Key Decisions log ("attestation/byte-exactness out of scope — MNN's job is inference, not verifiable-execution; SuperGenius verifies results separately") | Rely on the existing numeric-threshold `--testdir` check; leave attestation to the downstream SuperGenius/SGProcessingManager consumer |
| Restricting the converter output to a single backend (CPU-only or Vulkan-only) | Might seem simpler to validate one backend first | The SGFP4 v2 decode `Execution`s already exist and are parity-verified on **both** CPU and Vulkan (v1.0, 14-fixture sweep) — artificially restricting the converter output would waste already-built capability and contradict PROJECT.md's Vulkan-first Core Value | Converter output should be backend-agnostic (same `.mnn` + sidecar runs on either); validate end-to-end on at least one backend for MVP, both as a differentiator |

## Feature Dependencies

```
CLI flag (config.hpp field + cxxopts option)
    └──requires──> WeightQuantAndCoding-style dispatch branch (op-type switch, gated by flag)
                       └──requires──> Real-weight encoder callable from C++ convert path
                       └──requires──> Op-type rewrite / discriminator (runtime Execution selection)
                       └──requires──> RemoveParams.cpp external-sidecar case (SGFP4DequantParam)

Real-weight encoder (C++ port of encode_sgfp4.py math)
    └──requires──> Existing spec-derived policy constants (LEVEL_THRESHOLDS, VETO_FACTOR,
                    HYSTERESIS_DELTA, S_SEARCH_* ) ported unchanged from the Python reference

Adaptive quadtree (LAYOUT_MIXED) selection ──enhances──> base SGFP4 CLI flag
    (base flag could ship with a single fixed uniform layout; quadtree is the
     differentiator, but per PROJECT.md's "quadtree-adaptive" framing it is
     treated as MVP-core here, not deferred — see MVP Definition)

End-to-end validation (--testdir/--testconfig/--thredhold)
    └──requires──> CLI flag + encoder + op-rewrite + sidecar ALL functioning together
                    (integration test of the other four — not independently parallelizable)

Per-tensor/per-model reporting ──enhances──> base SGFP4 CLI flag
    (extends existing compressionParamsFile proto; not required for correctness)

Calibration/activation-data requirement ──conflicts──> CLI-flag parity with --weightQuantBits
    (adding a data dependency breaks the single-command, no-dataset UX every
     existing MNN quant flag guarantees)
```

### Dependency Notes

- **CLI flag requires dispatch branch:** Exactly mirrors how `--weightQuantBits` requires `WeightQuantAndCoding.cpp`'s op-type switch (cli.cpp:209-212 → config.hpp:42 → WeightQuantAndCoding.cpp:75-84) — same three-file wiring pattern, new SGFP4 branch instead of (or alongside) the int2-8 IDST branch.
- **Dispatch branch requires real-weight encoder + op-rewrite + sidecar case:** None of these three can be dropped — without the encoder there's nothing to quantize with; without the op-rewrite the runtime falls back to the wrong (or no) `Execution`; without the sidecar case the quantized payload has nowhere to be written (MNN's external-data convention, `RemoveParams.cpp`).
- **Adaptive quadtree enhances but doesn't block the base flag:** Decode already supports all uniform layouts *and* `LAYOUT_MIXED` (v1.0 shipped both), so technically a v1 could hardcode a single uniform layout (e.g. `LAYOUT_UNIFORM_16`) as a lower-risk fallback if the quadtree port proves too costly for the milestone window — flagged here as a de-risking option even though PROJECT.md frames quadtree-adaptive as the target.
- **Calibration conflicts with CLI parity:** Call out explicitly per the downstream consumer's request — this is the sharpest anti-feature because it's the most likely "obviously good idea" from ecosystem exposure (GPTQ/AWQ/llama.cpp `--imatrix`) that would actually regress the UX MNN has standardized on.

## MVP Definition

### Launch With (v1)

Minimum viable product — a real pytorch → onnx → mnnconvert(SGFP4 flag) → `.mnn` pipeline that runs correct inference.

- [ ] Native CLI flag peer to `--weightQuantBits`, wired through `cli.cpp` + `config.hpp` — essential, this *is* the milestone's stated deliverable
- [ ] Op-type dispatch for `Convolution2D`/`ConvolutionDepthwise`/`Deconvolution` only — essential, matches milestone scope exactly, avoids the "support everything" anti-feature
- [ ] Real-weight encoder (C++ port of `encode_sgfp4.py`'s affine dual-mode + adaptive quadtree logic) — essential, without it there is no real-weight quantization, only the existing synthetic-fixture path
- [ ] Op-type rewrite / discriminator so runtime selects the already-built SGFP4 decode `Execution` — essential, otherwise the quantized weights are unreadable at inference time
- [ ] `RemoveParams.cpp` external sidecar wiring (`SGFP4DequantParam {magic, offset, size}`) — essential, follows the explicit precedent PROJECT.md calls out
- [ ] Graceful per-layer fallback for geometry that doesn't fit SGFP4's macroblock/quadtree constraints — essential, prevents hard conversion failures on real (non-synthetic) models with irregular channel counts
- [ ] End-to-end validation via existing `--testdir`/`--testconfig`/`--thredhold` on one real test model, at least one backend (CPU or Vulkan) — essential, this is the milestone's explicit acceptance bar

### Add After Validation (v1.x)

- [ ] Validation on the second backend (whichever of CPU/Vulkan wasn't covered in v1) — trigger: v1 end-to-end passes on one backend and there's time/need to prove backend-agnosticism
- [ ] Per-tensor/per-model compression-ratio and quant-error reporting via extended `compressionParamsFile` proto — trigger: a concrete downstream consumer (e.g. GeniusCogntiveSystem) asks for visibility into quant quality per layer
- [ ] User-tunable adaptive-policy CLI knobs (`--sgfp4Eps` etc.) — trigger: real-model validation shows spec defaults need per-model tuning

### Future Consideration (v2+)

- [ ] Additional op-type coverage (`DeconvolutionDepthwise`, requantization from `ConvInt8`/`DepthwiseConvInt8`) — defer until a real model actually needs SGFP4 on these op types
- [ ] Calibration/activation-aware refinement of adaptive thresholds — defer indefinitely unless weight-intrinsic quantization proves insufficient; treat as a deliberate architecture change, not an incremental add
- [ ] Attestation/byte-exactness guarantees — permanently out of scope per PROJECT.md's Key Decisions log, not a "someday" item

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Native CLI flag (config.hpp + cli.cpp) | HIGH | LOW | P1 |
| Op-type dispatch (3 named types) | HIGH | LOW-MEDIUM | P1 |
| Real-weight encoder (C++ port) | HIGH | HIGH | P1 |
| Op-type rewrite / runtime discriminator | HIGH | MEDIUM | P1 |
| RemoveParams.cpp sidecar case | HIGH | LOW-MEDIUM | P1 |
| Per-layer geometry fallback | MEDIUM | MEDIUM | P1 |
| Adaptive quadtree (LAYOUT_MIXED) selection | HIGH | HIGH | P1 (milestone-defining, but see de-risking note above) |
| Dual-mode per-block selection | MEDIUM-HIGH | HIGH (pre-solved algorithm) | P1 (comes for free with the encoder port) |
| E2E validation via `--testdir` | HIGH | LOW-MEDIUM | P1 |
| Second-backend validation | MEDIUM | LOW-MEDIUM | P2 |
| Per-tensor/per-model reporting | MEDIUM | MEDIUM | P2 |
| User-tunable policy CLI knobs | LOW | LOW-MEDIUM | P3 |
| Additional op-type coverage | LOW (no current model needs it) | MEDIUM | P3 |
| Calibration/activation-data support | LOW (not requested, breaks UX parity) | HIGH | Not planned |
| Attestation/byte-exactness | N/A (explicitly out of scope) | N/A | Not planned |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

"Competitors" here = other converter-embedded / weight-only quantization tools whose CLI UX and validation patterns are relevant precedent, plus MNN's own existing `--weightQuantBits` path as the internal baseline.

| Feature | MNN `--weightQuantBits` (internal baseline) | llama.cpp GGUF (`llama-quantize`) | Intel Neural Compressor WOQ | Our SGFP4 Approach |
|---------|----------------------------------------------|-------------------------------------|-------------------------------|---------------------|
| Block/group sizing | Fixed `--weightQuantBlock N` (channel-wise if unset) | Fixed per-format super-block (e.g. Q4_K = 256-element blocks with per-block fp16 scale+min) | Configurable group size, fixed per run | Adaptive per-block via quadtree recursion (64→32→16→8→4), not a single fixed size — this is the actual differentiator vs. both peers |
| Calibration data requirement | None (weight-intrinsic; HQQ variant also calibration-free) | Optional `--imatrix` (importance matrix from calibration corpus) improves low-bit accuracy but isn't required for base quant | Some WOQ recipes (GPTQ/AWQ-style) require calibration; others (RTN) don't | None — stays calibration-free like MNN's baseline (see Anti-Features: calibration conflicts with CLI parity) |
| Mode/format selection | Single format per run (bits 2-8, symmetric/asymmetric) | Multiple named formats chosen at CLI time (Q4_0, Q4_K_M, Q6_K, ...), fixed once selected | Multiple algorithms selectable (RTN, GPTQ, AWQ, ...) | Per-block dual-mode (FP4_AFFINE / T158_AFFINE) selected automatically per leaf, not chosen globally by the user |
| Validation mechanism | External (user runs their own accuracy checks) | External (user runs perplexity/benchmark tools separately) | Built-in eval harness with per-tensor/per-layer metrics | Reuse MNN's existing built-in `--testdir`/`--thredhold` numeric-diff harness (already in `mnnconvert`) — no new tooling needed |
| CLI flag shape | `--weightQuantBits N --weightQuantAsymmetric --weightQuantBlock N --hqq` | Single format positional arg + optional `--imatrix path` | Python API / recipe config file, not a single CLI flag | New flag peer to `--weightQuantBits` (on/off + spec defaults baked in for v1), matching MNN's existing flag-per-mode convention rather than llama.cpp's format-enum or INC's config-file style |

## Sources

- Direct codebase reads (HIGH confidence, primary source): `tools/converter/source/common/cli.cpp` (flag registration, lines 195-228, 477-503, 785-787), `tools/converter/include/config.hpp` (`modelConfig` struct), `tools/converter/source/common/WeightQuantAndCoding.cpp` (existing op-type dispatch, block-wise quant loop, HQQ branch, compression-proto reporting), `tools/converter/source/common/RemoveParams.cpp` (external sidecar store/load precedent), `tools/fp4/encode_sgfp4.py` (existing SGFP4 v2 synthetic-fixture reference encoder, adaptive-policy constants), `tools/fp4/quantize_fp4.py` (existing post-hoc E2M1 tool, for contrast — shells out to `mnnconvert` rather than integrating into it), `.planning/PROJECT.md` (milestone scope, Key Decisions log)
- [GGUF File Format Explained (llama.cpp)](https://apxml.com/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/gguf-format) (MEDIUM confidence — general web source, cross-referenced)
- [Quantizing Models - llama.cpp](https://mintlify.com/ggml-org/llama.cpp/models/quantizing-models) (MEDIUM confidence)
- [llama.cpp tools/quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) (MEDIUM confidence — GitHub-hosted primary docs)
- [Difference in different quantization methods (ggml-org/llama.cpp Discussion #2094)](https://github.com/ggml-org/llama.cpp/discussions/2094) (MEDIUM confidence — community discussion)
- [Weight Only Quantization (WOQ) — Intel Neural Compressor docs](https://intel.github.io/neural-compressor/latest/docs/source/quantization_weight_only.html) (MEDIUM confidence — vendor docs)

---
*Feature research for: Converter-embedded adaptive-block weight-only quantization (SGFP4 v2, mnnconvert integration)*
*Researched: 2026-08-25*
