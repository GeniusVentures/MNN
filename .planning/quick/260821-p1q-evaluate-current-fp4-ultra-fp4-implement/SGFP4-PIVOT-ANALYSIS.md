# SGFP4 Pivot Analysis

**Quick task:** 260821-p1q
**Date:** 2026-08-21
**Scope:** Documentation-only analysis. No files under `source/`, `test/`, or `tools/` were modified by this task.

## Executive Summary

The current Ultra FP4 implementation is a minimal E2M1 floating-point microformat reusing MNN's native `symmetricQuan` per-channel scale/bias container, while the new SGFP4 spec (`.planning/sgfp4-arxiv-v2.pdf` / `.planning/sgfp4-arxiv-v2.txt`) defines a fundamentally different affine-integer, dual-mode (FP4_AFFINE + T158_AFFINE ternary), macroblock-addressed container with verifiability as a first-class design goal. Top-line recommendation: do not retrofit the current work; treat SGFP4 as new, additive roadmap work (candidate Phase 6+), and fix a concrete pre-existing scale-calibration defect in the current pipeline independent of the pivot decision.

## Locked Decisions (2026-08-22, user-directed — supersedes the open questions below where they conflict)

1. **Target v2 only.** No v1 work, staged or otherwise — v1 fixed-payload is dropped from scope entirely. Resolves P3.
2. **GNUS Execution Integrity / attestation is explicitly OUT of scope for MNN.** MNN's job is purely running AI processing and returning a result; SuperGenius is responsible for checking/verifying that result. No conformance-vector tests, no byte-exactness test infrastructure, no integer-exact-only kernel variants are required in MNN for this reason. Resolves P5 (as: not in scope, full stop — stronger than the analysis's original "defer" framing).
3. **MNN-only scope.** This workstream plans and implements SGFP4 v2 entirely within MNN. Any `SGProcessingManager`/SuperGenius-side integration (consuming whatever new decode entry point MNN ends up exposing) is explicitly deferred to a separate GSD plan in that repo — not part of this workstream's phases. The cross-repo API-contract risk noted in Section 6 still applies (a future SuperGenius-side plan will need to update its call site), but designing or scheduling that update is out of scope here.
4. **Container adoption depth (P4) is still open** — the user does not yet have enough information to decide, and v2's own shape narrows the real choice: v2's variable-size, quadtree-subdivided macroblock records cannot be represented in MNN's flat per-channel `symmetricQuan` arrays at all (unlike v1, which could plausibly fit a "math-only" per-channel/per-block adoption). Choosing v2 therefore effectively eliminates P4 option (b) "math-only adoption" — the real choice narrows to P4 (a) full container replacement vs. (c) hybrid opaque-blob-per-op. See Section 7 for the architecture research needed before this can be answered concretely — since that research, a concrete recommendation now exists (external-file + opaque descriptor); confirm with the user before locking it.

## 1. Current Implementation Summary (Ultra FP4 — Phases 2 and 4)

The current format is E2M1: 1 sign bit, 2 exponent bits (bias = 1), 1 mantissa bit. Encodable values are 0, ±0.5, ±1, ±1.5, ±2, ±3, plus ±Inf and NaN as special codes (`include/MNN/FP4DequantUtils.hpp`, `dequant_e2m1_cpu`). Two values are packed per byte: the low nibble holds the even-index element, the high nibble the odd-index element (`pack_fp4_byte` in both `include/MNN/FP4DequantUtils.hpp` and `tools/fp4/quantize_fp4.py`).

Scale is a per-output-channel FP32 value stored in MNN's native `symmetricQuan.scale` array; `symmetricQuan.bias` is hardcoded to `[0.0] * oc` in `tools/fp4/quantize_fp4.py` (`quantize_model`, `main["symmetricQuan"] = {... "bias": [0.0] * oc ...}`). There is no custom binary container of any kind — no macroblocks, no header/offset arrays, no embedded flags. The entire "container" is three flat parallel arrays (`weight`, `scale`, `bias`) inside the existing `symmetricQuan` FlatBuffers structure that MNN already uses for INT4/INT8 quantization, with `nbits=4` selecting the FP4 codepath at dispatch time.

Both runtime paths apply no scale or bias inside the dequant op itself — they are pure code-to-float lookups, with per-channel scale applied downstream by MNN's existing `symmetricQuan` consumer:

- **`CPUFP4Dequant::onExecute`** (`source/backend/cpu/CPUFP4Dequant.cpp`) calls `dequant_fp4_packed_cpu`, a pure per-nibble table lookup with no scale/bias applied in-op. `CPUFP4DequantCreator::onCreate` detects FP4 by matching the input packed-byte count against `(outputElementCount + 1) / 2`, deferring to the existing `CPUDequantizeCreator` fallback otherwise.
- **`VulkanFP4Dequant`** (`source/backend/vulkan/buffer/execution/VulkanFP4Dequant.cpp`) dispatches `glsl_fp4_dequant_FP16_comp` or `glsl_fp4_dequant_comp` depending on `useFP16()` / an explicit FP32 flag, using 256-thread workgroups (`vkCmdDispatch(cmdBuffer->get(), UP_DIV(elementCount, 256), 1, 1)`), also with no in-shader scale/bias.

**Git authorship note:** all four FP4 source files (`include/MNN/FP4DequantUtils.hpp`, `tools/fp4/quantize_fp4.py`, `source/backend/cpu/CPUFP4Dequant.cpp`, `source/backend/vulkan/buffer/execution/VulkanFP4Dequant.cpp`) are authored by `Super Genius <ken+git@gnus.ai>` — the same author line as the SGFP4 paper (Kenneth Hurley, GNUS.AI / Super Genius). This means the current Ultra FP4 implementation is very likely the predecessor design by the same author later formalized as SGFP4.

**Current status:**
- Phase 2 (shader + pipeline integration): complete.
- Phase 4 plan 04-01 (`quantize_fp4.py` + `CPUFP4Dequant`): complete.
- Phase 4 plan 04-02 (E2E CPU+Vulkan model test): planned but NOT yet executed.
- Phase 5 (model-level regression tests): not yet scoped, depends on Phase 4.

### Verified defect: MAX_E2M1_VALUE scale-calibration bug

`tools/fp4/quantize_fp4.py` defines `MAX_E2M1_VALUE = 6.0` and computes `scale = max_abs / MAX_E2M1_VALUE` per channel (`quantize_channel_weights`), then normalizes each weight as `channel_weights[i] / scale`.

But the largest finite magnitude E2M1 can represent is **3.0** (nibble `0x5`/`0xD`, `biased_e=2, m=1` — see the E2M1 test-vector table in `FP4DequantUtils.hpp`). `encode_fp4_e2m1` saturates any value with `biased_e >= 3` (i.e., magnitude >= 4.0) to ±Inf:

```python
e = int(np.floor(np.log2(val)))
biased_e = e + 1  # bias = 1
if biased_e >= 3:
    return 0x06 | (s << 3)  # saturate to max (6.0)
```

Since the per-channel max-magnitude weight normalizes to exactly `max_abs / (max_abs / 6.0) = 6.0`, and `6.0` has `biased_e = floor(log2(6)) + 1 = 3`, that weight saturates to ±Inf on every channel, every time — not to the intended max finite code ±3.0. The correct divisor is **3.0**, not 6.0.

This wastes roughly one exponent level of dynamic range for the rest of the channel's weights and injects a guaranteed Inf into every quantized channel's largest-magnitude weight. This is independent of any SGFP4 pivot decision and should be fixed regardless, because Phase 4 plan 04-02's own acceptance criteria checks "packed FP4 weights ... match original float weights within E2M1 precision (max error <= 0.5)" — a criterion this defect can directly violate for any channel whose max-magnitude weight participates in the compared output.

## 2. SGFP4 Spec Summary

SGFP4 (Kenneth Hurley, GNUS.AI / Super Genius) defines an affine reconstruction rule `w_hat = S * c + bias` (Eq. 2, Section 3.2), with S and bias as FP16 (IEEE 754 binary16) values packed into a single uint32 in packHalf2x16 order (S in the upper 16 bits, bias in the lower 16 bits).

Two code modes share this reconstruction rule:
- **Mode 0, FP4_AFFINE:** a 4-bit two's-complement signed integer, codes `c` in `[-8, 7]`.
- **Mode 1, T158_AFFINE:** ternary, codes `c` in `{-1, 0, +1}`, 2 bits/code, in the BitNet b1.58 class (Section 3.2, citing Ma et al. 2024).

Mode is selected per-block by round-trip error comparison: "choose T158 iff e_T158 <= (1 + epsilon) * e_FP4", with default epsilon = 0.10, range [0.05, 0.20] (Eq. 5, Section 4.4).

**v1 fixed-payload profile (Section 4):** 64x64-weight macroblocks, addressed via three parallel arrays:
- `headers[B]` — packed S/bias (uint32 per macroblock) per macroblock.
- `offsets[B]` — a 16-byte-aligned base offset into the codes blob, with the mode flag in bit 0 and an error-hint in bits 2-3 of the structurally-zero low 4 bits (since payloads are 2048 bytes and 16-byte aligned, offsets are always multiples of 16).
- a codes blob of constant 2048-byte payloads per macroblock regardless of mode — mode 1 (ternary) payloads pad to the same 2048-byte size: 1024 bytes of 2-bit ternary codes (words 0-255) + 1024 bytes reserved-zero (words 256-511). Mode 0 (FP4) uses all 512 words as 4-bit nibbles.

**v2 quadtree-adaptive profile (Section 6):** a self-framed stream (magic `'SGF4'`, version `0x02`, a record-offset table), with variable-size per-macroblock records. Each record has a layout enum (`LAYOUT_UNIFORM_64/32/16/8`, `LAYOUT_MIXED` via an explicit pre-order-DFS quadtree split bitmap with quadrant order TL/TR/BL/BR, `LAYOUT_FULL_4x4`), followed by per-leaf headers packing S plus a truncated (12-bit) bias with flags in the low 4 bits (bit 0 = mode; the truncated 4 low mantissa bits of bias are repurposed, bounding relative bias error to ~0.8% of |bias|).

The paper's own status statement (Section 4, "Status" paragraph) is explicit: "At the time of writing, no artifacts of either profile have been issued; the reference pipeline targets v2. v1 is retained ... as the uniform-stride baseline ... and as the simplest conformance target."

**Section 8's verifiability goal:** the spec is normatively closed (byte order, code bit order, rounding, reserved-bit rules) so independent CPU/GPU implementations decode bit-identically, supporting the "GNUS Execution Integrity System" attestation use case — teacher-forced replay + checkpoint-tolerance verification across untrusted nodes in a decentralized inference network — with the ternary mode's integer-exact matmul registering in the cheapest determinism class (bit-exact reference semantics, output-hash comparison), while FP4-affine paths with floating-point accumulation register in a bounded-drift class with checkpoint-band comparison.

The paper is explicitly spec-only: "empirical results are left to a companion report" (Abstract, Section 9), and Section 9 references an "open reference pipeline" not yet public at time of writing.

## 3. Gap Analysis

| Dimension | Current MNN Impl | SGFP4 Spec | Gap/Impact |
|---|---|---|---|
| Numeric reconstruction | Floating-point microformat table lookup (E2M1); bias hardcoded to 0 | Affine integer `w = S*c + bias` with a real FP16 bias term | Current impl has no bias correction at all; SGFP4's affine rule requires a real per-block bias parameter and a different decode operation (FMA vs. table lookup). |
| Code space | E2M1's non-uniform float codeset including Inf/NaN traps | FP4_AFFINE's uniform signed-integer codes `[-8, 7]`, no special values | E2M1 wastes code space on Inf/NaN and has non-uniform step sizes; SGFP4's integer codes are uniform and trap-free, simplifying both encode and verification. |
| Granularity/container | MNN's per-op `symmetricQuan` flat scale/bias arrays (channel-granularity, no binary container) | SGFP4's macroblock-addressed container with packed headers/offsets and embedded flags (64x64 tiles) | **This is the single biggest architectural gap.** SGFP4's container model does not map onto MNN's per-op `symmetricQuan` schema at all — there is no macroblock concept, no offset table, no flag-in-low-bits mechanism anywhere in MNN's quantization path today. |
| Dual-mode/ternary | Entirely absent today — FP4-only, no per-block mode selection | SGFP4's core contribution: per-block FP4/ternary mixing selected by round-trip error | Would require a new second code path (ternary decode, mode-selection encoder heuristic) with zero current analog in MNN's FP4 pipeline. |
| v1 vs v2 profile | Current impl matches neither | v1 = fixed-payload uniform addressing; v2 = quadtree-adaptive, the paper's own reference-pipeline target | v1 is architecturally the smaller lift given today's flat-array integration points; v2 is what the paper's own reference pipeline targets but is substantially more complex (self-framed stream, quadtree). |
| Verifiability/bit-exact decode | No current test coverage strategy exists for cross-device bit-exactness | SGFP4 treats this as a first-class, normatively-specified requirement (byte order, bit order, rounding, reserved bits all normative) | Adopting this would require conformance-vector ("golden container") tests and explicit CPU/Vulkan decode-parity tests — nothing like this exists in the current Ultra FP4 test plan. |
| Scale/bias precision | Per-channel scale is FP32; bias is always 0.0 | Packed FP16 scale+bias pair (packHalf2x16), truncated further in v2 leaf headers | SGFP4 uses lower-precision (FP16) parameters but actually uses the bias term, unlike the current impl's unused FP32-precision-but-always-zero bias. |
| Pending defect | `MAX_E2M1_VALUE=6.0` bug (Section 1) causes guaranteed Inf on every channel's max-magnitude weight | N/A (SGFP4 has no equivalent saturation trap in its integer code space) | Orthogonal to the pivot decision but must be fixed either way before Phase 4 plan 04-02 can pass its own acceptance criteria. |

## 4. Recommended Pivots and Decisions (prioritized)

**P1 (do now, no pivot required — elevated priority, see Section 6):** Fix the `MAX_E2M1_VALUE` scale-calibration bug in `tools/fp4/quantize_fp4.py` (change the divisor from 6.0 to 3.0) before executing Phase 4 plan 04-02. Add an explicit acceptance check to 04-02 that each channel's max-magnitude weight round-trips to a finite value, not Inf. **This is no longer purely an internal test-criteria concern**: Section 6 confirms `FP4DequantUtils.hpp`'s `dequant_fp4_packed_cpu()` is already called live from SuperGenius's `SGProcessingManager` distributed processing pipeline, so this bug corrupts the max-magnitude element of every FP4_ULTRA tensor actually processed there today, not just MNN's own test fixtures.

**P2 (user decision):** Whether to still execute Phase 4 plan 04-02 (validating/closing out the current E2M1 pipeline as a shipped baseline, with the P1 fix folded in) before starting SGFP4 work, or abandon it in favor of jumping straight to SGFP4. Recommendation: execute it — it validates already-built infrastructure (Phase 2 + Phase 4-01) cheaply and does not block a future SGFP4 phase, since SGFP4 work is additive (new op/format handling), not a modification of the existing `symmetricQuan` `nbits=4` path.

**P3 (user decision, explicitly flagged as open — do not decide unilaterally):** v1-first vs. v2-first vs. staged v1-then-v2 adoption of SGFP4 in MNN.
- v1 pros: smaller lift given current flat-array integration surface; simpler conformance target; validates the affine-dual-mode decode math independent of the harder container work.
- v1 cons: not what the paper's reference pipeline targets; neither profile has a published reference implementation yet per the paper's own status note.
- v2 pros: matches the paper's reference-pipeline target and its accuracy/compression story.
- v2 cons: substantially larger scope — self-framed stream format, quadtree split-map, per-leaf headers, variable-size records, recursive error-driven encoder.
- Non-binding sketch: a staged approach — affine dual-mode math first (v1-shaped), full container/quadtree later.

**P4 (user decision, explicitly flagged as open):** Container adoption depth, three named sub-options:
- (a) **Full container adoption** — replace `symmetricQuan`-based encoding with a true SGFP4-conformant binary blob (`headers[B]`/`offsets[B]`/codes blob), sacrificing MNN's native per-channel array introspection but gaining spec conformance for the attestation use case.
- (b) **Math-only adoption** — keep MNN's flat per-channel arrays (widened to carry a real bias value) but switch the reconstruction formula to SGFP4's affine rule and add ternary as a second per-channel/per-block code path. Smaller lift, but no bit-exact conformance to the SGFP4 wire format, only the numerics.
- (c) **Hybrid** — store SGFP4-format containers as an opaque per-op blob decoded via a dedicated new dequant creator branch, leaving all non-FP4 MNN ops untouched.
- Lean recommendation (Claude's suggestion, not a locked decision): (b) first, (c) as a natural follow-up if verifiability becomes required, (a) only if wire-format conformance is explicitly required.
- **Cross-repo constraint (see Section 6):** `FP4DequantUtils.hpp`'s `dequant_fp4_packed_cpu()` is a live API contract consumed directly by SuperGenius's `SGProcessingManager` submodule. Any of (a)/(b)/(c) that changes this function's signature, packing layout, or E2M1 semantics is a breaking change for that downstream repo and must either preserve a compatibility shim or be coordinated with a corresponding `SGProcessingManager` update (and a submodule-pointer bump in SuperGenius) landed together. This raises the practical cost of option (a) specifically, since full container adoption most directly obsoletes the current pass-through call.

**P5 (defer, flag for user):** Whether the GNUS Execution Integrity System's attestation use case (teacher-forced replay across untrusted nodes) is actually in scope for this MNN fork. It is the SGFP4 paper's stated primary motivation but has zero footprint anywhere in the current Ultra FP4 roadmap (Phases 1-5 never mention attestation, execution verification, or conformance vectors). If in scope, a future phase would need a conformance-vector ("golden container") test format, explicit byte-order unit tests for CPU vs. Vulkan decode parity, and possibly integer-exact-only kernel variants for the ternary path.

**P6 (sizing note, not scoped here):** Ternary (T158_AFFINE) CPU decode, ternary Vulkan/GLSL decode + `makeshader.py`-pipeline shader registration, and an encoder mode-selection heuristic (round-trip MSE compare, default epsilon=0.10) are entirely new work not present in any current roadmap phase. Rough-order-of-magnitude estimate: comparable combined scope to Phase 2 (shader + pipeline) plus Phase 4 (tool + CPU dequant), i.e., realistically 2 full phases (one for affine-dual-mode math on CPU+Vulkan, one for the container/addressing layer) — consistent with the staged approach suggested in P3.

## 5. Suggested Phase 6+ Scope Sketch (non-binding)

These are sketches for the user's next roadmap-editing session, not additions to ROADMAP.md:

- **Phase 6 candidate — "SGFP4 Affine Dual-Mode Reconstruction (v1 math, math-only container per P4-b)":** CPU+Vulkan decode for FP4_AFFINE + T158_AFFINE with per-channel/per-block FP16 scale+bias, encoder mode-selection heuristic, correctness tests against the paper's Eq. 2/3/4.
- **Phase 7 candidate (conditional on Phase 6 outcome and the user's P3 decision):** either SGFP4 v1 fixed-payload container conformance, or skip straight to v2 quadtree-adaptive if matching the paper's reference pipeline is prioritized over shipping sooner.
- **Phase 8 candidate (conditional on the P5 decision):** verifiable-execution/bit-exact conformance testing, only if the GNUS attestation use case is confirmed in scope.

## 6. Cross-Repo Integration: SuperGenius `SGProcessingManager` (verified live consumer)

Added after initial publication, in response to a user pointer to `W:\gnus\GeniusNetwork\SuperGenius\SGProcessingManager\test\processors\` — this section corrects and sharpens Sections 4/5's framing of MNN's FP4 work as self-contained.

**FP4_ULTRA decode is already wired live, not pending.** `SGProcessingManager/src/processors/processing_processor_mnn_tensor.cpp` (submodule of `SuperGenius`, currently pinned at commit `e1f28d73` on branch `dev_cognitive`, dated 2026-08-20) includes `<MNN/FP4DequantUtils.hpp>` directly and, for `InputFormat::FP4_ULTRA` tensors, calls:

```cpp
// Line 274-279:
else if ( format == sgns::InputFormat::FP4_ULTRA )
{
    // Pass-through to MNN's own E2M1 decode (D-09) -- no dequant math duplicated here.
    const auto *src = reinterpret_cast<const uint8_t *>( tensorData.data() );
    MNN::dequant_fp4_packed_cpu( src, signalValues.data(), expectedElements );
}
```

`FP4_ULTRA` is accepted in the tensor-format allow-list (lines 217-223) alongside FLOAT32/FLOAT16/INT32/INT16/INT8, and its expected-byte-count calculation (lines 226-234, referencing decision "D-13") mirrors the 2-nibbles-per-byte packing in `FP4DequantUtils.hpp`. The outer `SuperGenius` superproject has already bumped its submodule pointer to this exact commit (commit `afc17e52`, "chore: bump SGProcessingManager for DataType::LLM fix + FP4_ULTRA decode wiring") — this is live in both repos' current HEAD, not orphaned or speculative work.

**Consequence for this analysis:** `dequant_fp4_packed_cpu()` in `include/MNN/FP4DequantUtils.hpp` is a **live cross-repo API contract**, not an MNN-internal implementation detail. This sharpens P1 and P4 above (see inline notes added to each) and adds a new, concrete constraint to any SGFP4 pivot: a change to this function's signature, packing layout, or E2M1 semantics is a breaking change for `SGProcessingManager`'s distributed tensor-processing pipeline and requires coordinated landing across both repos (MNN + a `SGProcessingManager` update + a `SuperGenius` submodule-pointer bump), not just an MNN-side decision.

**Side-finding — a stale test in the sibling repo:** `SGProcessingManager/test/processors/mnn_tensor_fp4_test.cpp`'s `Fp4UltraRecognizedButDecodeUnavailable` test still asserts the *old* stub behavior (`ProcessingErrorStage::FORMAT_UNSUPPORTED`, message containing "MNN_Ultra") from commit `b5471e0` ("give FP4_ULTRA a structured failure path"). That behavior was replaced one commit later by `e1f28d7` (the real-decode wiring above) without updating the test. Running that test suite against `SGProcessingManager`'s current HEAD would fail this assertion. This is out of MNN's scope to fix directly (different repo, different git history) but is worth flagging to whoever owns `SGProcessingManager`'s CI.

**Documentation gap:** decisions "D-09" and "D-13" (cited by code comments above) and the earlier "D-04" (cited in commit `b5471e0`'s message) have no surviving committed design doc in either repo — `SuperGenius/.planning` has no phase covering this integration, and the `SGProcessingManager` submodule has no `.planning/` directory at all on any branch checked (`main`, `dev_cognitive`, `dev_rendering`, `feat/add-llm-fp4ultra-processors`). The only record of these decisions is code comments and commit messages.

**No attestation/ZK tie-in exists yet.** SuperGenius's actual proof system (`src/proof/GeniusProver.cpp`, `ProofSystem` submodule) is scoped to confidential token-transfer proofs via Pedersen commitments — unrelated to model-execution verification, checkpoint replay, or quantization. This means SGFP4's Section 8 attestation motivation (P5 above) currently has no concrete anchor anywhere in this codebase family; it would be new work in both MNN and SuperGenius if pursued.

## 7. Container Integration Architecture Research (added 2026-08-22, resolves P4)

Researched MNN's actual FlatBuffers Op/schema architecture to ground the P4 container-adoption-depth decision rather than guess. Findings:

**Existing quant schema is flat and fixed-shape, no recursion capability.** `schema/default/CaffeOp.fbs:55-72` (`IDSTQuan`) and `:80-100` (`QuantizedFloatParam`, what `symmetricQuan` actually is) are both flat typed arrays (`weight:[byte]`, `scale:[float]`, etc.) — neither can represent SGFP4 v2's variable-size, recursively-subdivided quadtree records natively.

**MNN already has two working precedents for exactly this shape of problem** (large, variable-size, per-tensor binary payload):
1. **Opaque in-graph blob:** `Extra.info:[byte]` (`schema/default/MNN.fbs:218-225`, `OpType_Extra=512`) — MNN's actual custom-op escape hatch; the FlatBuffers schema carries raw bytes it does not interpret, dispatch is by string `type`/`engine`, and a registered creator parses the blob itself. `ExtraInfo.fbs:11-15` (`buffer:[int8]`) is the same pattern in miniature.
2. **External sidecar file:** `Convolution2D.external:[int64]` (`CaffeOp.fbs:109`, an `[offset, weight_bytes_size, bias_bytes_size]` tuple) points into a `<model>.mnn.weight` file loaded via `FileLoader` (`ConvolutionCommon.cpp:567-598`, `Interpreter.cpp:96`) — the FlatBuffers op only carries the offset/size, not the payload. This is the same generic mechanism MNN's LLM export already reuses for weight files (`transformers/llm/export/llmexport.py:267`), not a bespoke LLM-only path.

**FlatBuffers loading is zero-copy** (`Interpreter.cpp:62-125`: the whole `.mnn` file is read into one buffer and `GetNet()` is a raw pointer cast, no eager deserialization) — so an opaque `[ubyte]`/external-file blob costs nothing extra to hold; all the real parsing work (walking SGFP4's own offset table, quadtree recursion) happens inside whatever Execution class reads it, exactly as `ConvolutionCommon.cpp:552-658` already does for existing quant formats.

**Recommendation (research-grounded, not a guess): the external-file pattern augmented with a minimal opaque descriptor, not a first-class FlatBuffers schema for SGFP4's internal structure.** Concretely: store the entire SGFP4 v2 container (magic, record-offset table, macroblock records) as-is in a `.mnn.weight`-style external file; the op itself carries only a small descriptor (e.g. `magic:uint32; offset:int64; size:int64` — modeled on `Convolution2D.external`'s existing `[offset, size]` shape, or reusing `Extra.info` for the in-graph case); a new dedicated Execution class (CPU + Vulkan) does SGFP4's own byte-level offset-table/quadtree parsing internally, unconstrained by FlatBuffers. This is option (c) "hybrid opaque blob" from Section 4/P4, refined to match MNN's actual established idiom rather than inventing a new one — a genuine schema-first approach (P4 option (a): typed FlatBuffers fields modeling headers/offsets/quadtree structure) would require the FlatBuffers compiler and every schema consumer (converter, all backends, doc generators) to understand quadtree recursion for a format that's inherently variable-size and self-describing, fighting the grain of a system that already routes this exact class of problem through opaque-blob + external-file, twice.

Sources: `schema/default/CaffeOp.fbs:55-110`, `schema/default/MNN.fbs:199,213-225,453`, `schema/default/ExtraInfo.fbs:11-15`, `source/core/Interpreter.cpp:62-125`, `source/core/ConvolutionCommon.cpp:552-658`, `transformers/llm/export/llmexport.py:267`.

## 8. Open Questions for the User

1. ~~v1 vs. v2 target, or staged v1-then-v2?~~ **RESOLVED 2026-08-22: v2 only** (see Locked Decisions above).
2. Container adoption depth — narrowed to (a) full spec-conformant container vs. (c) hybrid opaque-blob (option (b) math-only is not viable for v2's variable-granularity quadtree records — see Locked Decisions #4). **Section 7's architecture research now gives a concrete, grounded recommendation** (external `.mnn.weight`-style file + minimal offset/size descriptor in the op, mirroring `Convolution2D.external`) — pending user confirmation before this is locked as a decision rather than a recommendation.
3. ~~Is the GNUS Execution Integrity System attestation use case actually in scope for this MNN fork...~~ **RESOLVED 2026-08-22: not in scope.** MNN only runs AI processing and returns a result; SuperGenius checks it. No attestation/conformance-vector work belongs in this workstream.
4. Should Phase 4 plan 04-02 be executed as a "close out the E2M1 baseline" step (recommended per P2), or abandoned in favor of jumping straight to SGFP4 work? **Still open.**
5. ~~Should any future SGFP4 planning be run as a single cross-repo workstream spanning MNN and SuperGenius/SGProcessingManager...~~ **RESOLVED 2026-08-22: no — MNN-only scope.** SuperGenius-side integration (consuming MNN's new decode entry point, fixing the stale `Fp4UltraRecognizedButDecodeUnavailable` test, backfilling D-04/D-09/D-13 documentation) is deferred to a separate GSD plan in that repo, out of scope for this workstream.

## Appendix: Sources Consulted

Note: paths below reflect the post-2026-08-21 workstream migration (existing Phase 1-5 roadmap/state/phases relocated to `.planning/workstreams/milestone/` when the `sgfp4-pivot` workstream was created). At the time this analysis was written, these files lived directly under `.planning/`.

- .planning/sgfp4-arxiv-v2.txt
- include/MNN/FP4DequantUtils.hpp
- tools/fp4/quantize_fp4.py
- source/backend/cpu/CPUFP4Dequant.cpp
- source/backend/vulkan/buffer/execution/VulkanFP4Dequant.cpp
- .planning/workstreams/milestone/phases/02-ultra-fp4-quantization/02-CONTEXT.md
- .planning/workstreams/milestone/phases/04-convert-test-models-mnn-or-onnx-into-ultra-fp4-quantization-/04-01-SUMMARY.md
- .planning/workstreams/milestone/phases/04-convert-test-models-mnn-or-onnx-into-ultra-fp4-quantization-/04-02-PLAN.md
