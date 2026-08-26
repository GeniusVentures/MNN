# Stack Research

**Domain:** Native model-converter weight quantization (C++, FlatBuffers-schema inference engine)
**Researched:** 2026-08-25
**Confidence:** HIGH (codebase-grounded — every recommendation below traces to a file already read in this repo) / MEDIUM for the external prior-art comparison (WebSearch corroboration, not independently verified against source)

## Summary Up Front

This milestone needs **zero new external dependencies**. Everything required to encode real weights into SGFP4 v2 containers and wire them into mnnconvert already exists in-tree: the decode side (`include/MNN/SGFP4DequantUtils.hpp`, `CPUSGFP4Dequant`, `VulkanSGFP4Dequant`), the FlatBuffers descriptor (`SGFP4DequantParam` in `schema/default/CaffeOp.fbs`, already generated into `schema/current/`), the reference encode algorithm (`tools/fp4/encode_sgfp4.py`), and the converter-side pattern to follow (`WeightQuantAndCoding.cpp` + `RemoveParams.cpp` + `writeFb.cpp`). The work is a **port + wire-up**, not a build-a-new-capability task. Confirmed by reading `tools/fp4/encode_sgfp4.py`, `include/MNN/SGFP4DequantUtils.hpp`, `tools/converter/source/common/{WeightQuantAndCoding,RemoveParams,writeFb,cli}.cpp`, `tools/converter/include/config.hpp`, `tools/converter/source/common/CommonUtils.hpp`, `schema/default/CaffeOp.fbs`, `source/backend/cpu/CPUSGFP4Dequant.cpp`.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| C++11 (project baseline; converter has no CXX_STANDARD override) | n/a | Language for the new encoder | `tools/converter/CMakeLists.txt` sets no standard override, so it inherits the project's C++11 default. `WeightQuantAndCoding.cpp` and `HQQQuantizer.hpp` are both plain C++11 (`std::vector`, `std::unique_ptr`, no `<optional>`/structured bindings) — match that, don't introduce C++17-only syntax into converter-common code. |
| `half_float::half` (vendored `3rd_party/half/half.hpp`, v1.12.0) | 1.12.0 | FP16 pack/unpack of each leaf's `(S, bias)` header | The decode side (`SGFP4DequantUtils.hpp::unpack_leaf_header`) already uses this exact library via `std::memcpy` into `half_float::half`. The encoder MUST use the same library for the inverse (`float -> half` truncation) — anything else (hand-rolled FP16, `_Float16`, `__fp16`) risks producing S/bias bit patterns that don't bit-match what `dequant_sgfp4_container_cpu` will read back, silently breaking round-trip fidelity. No new vendoring needed. |
| FlatBuffers (vendored `3rd_party/flatbuffers`, v1.10.0) | 1.10.0 | Reads/writes `SGFP4DequantParam` (`magic`, `external:[offset,size]`, `dims`) already generated at `schema/current/CaffeOp_generated.h` | The schema table is **already defined and generated** (`schema/default/CaffeOp.fbs:118-122`, `OpType_SGFP4Dequant = 605` in `MNN.fbs:211`). This milestone's converter work does not need `flatc` regeneration — it only needs to *populate* the existing table (`SGFP4DequantParamT` with `magic = kSGFP4Magic`, `external`, `dims`) when rewriting a `Convolution2D` op into an `OpType_SGFP4Dequant` op. Only touch the `.fbs` if a genuinely new field is discovered mid-implementation (not expected). |
| `MNN::SGFP4DequantUtils.hpp` (existing, `include/MNN/`) | n/a (in-tree) | Shared container-format constants and helpers | Reuse `kSGFP4Magic`, `kSGFP4Version`, `kSGFP4FixedHeaderSize`, the `kSGFP4Layout*` enum, `kSGFP4QuadTreeMinSplitSize`, `kSGFP4MaxQuadTreeBits`, `kSGFP4LeafHeader*` masks, and `sgfp4_align16()` directly from this header rather than re-declaring them in converter code. This is the single source of truth the CPU/Vulkan decoders already depend on — importing it into the new encoder guarantees the encoder and decoder can never drift on a magic number, mask, or alignment constant. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| cxxopts (vendored `tools/converter/include/cxxopts.hpp`) | 2.1.0 | New mnnconvert CLI flag registration | Add the new flag (e.g. `sgfp4Quant`, peer to `weightQuantBits`/`weightQuantAsymmetric`/`weightQuantBlock`/`hqq`) in `cli.cpp` exactly where those four are declared (`cli.cpp:208-228`) and parsed (`cli.cpp:477-503`), storing into a new `modelConfig` field (`config.hpp:42-45` pattern). No new CLI library needed. |
| `MNN::Compression` protobuf (`MNN_compression.pb.h`, existing) | n/a (in-tree) | Optional: per-layer quant-method override via `PostTreatContext::quantInfo`, mirroring how `WeightQuantAndCoding` reads `context->quantInfo` (`WeightQuantAndCoding.cpp:95-109`) to let a `.json` compression-params file override CLI defaults per op name | Only needed if this milestone wants per-layer SGFP4 opt-out/override (e.g. "quantize all conv layers except X"). Not required for the MVP CLI-flag path; flag as a stretch goal, not a blocker. |
| `MNN::FileLoader` (`source/core/FileLoader.hpp`, existing) | n/a (in-tree) | Reads the SGFP4 sidecar back for `mnn2json`/re-conversion round-trips | Already used identically by `RemoveParams.cpp::loadExternalParam` for `Convolution2D`; the new `OpParameter_SGFP4DequantParam` case in `loadExternalParam` should follow the exact same `FileLoader` + `fl->offset()` + `fl->read()` pattern (see Integration Points below). |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `python tools/fp4/encode_sgfp4.py --selftest` | Ground truth / port source | This is the file to port line-for-line into C++, not a tool to keep calling at conversion time. Every function in it (`encode_leaf_fp4`, `encode_leaf_t158`, `select_mode`, `apply_ternary_veto`, `subdivide_macroblock`, `build_split_map`, `classify_layout`, `pack_leaf_header`, `pack_payload`, `encode_macroblock_mixed`, `encode_macroblock_adaptive`, `encode_container_adaptive`) has a direct 1:1 C++ counterpart to write. Keep the Python file as the cross-language oracle for a differential test (encode the same weight tensor in Python and C++, diff the container bytes) rather than deleting/replacing it. |
| `clang-format -i -style=file <file>` | Style conformance | Per project CLAUDE.md — run on the new encoder source before commit. |
| `run_test.out` (existing `test/op/SGFP4DequantTest.cpp`, `SGFP4VulkanDequantTest.cpp`) | Regression guard for the decode side while the encoder is built | These tests currently only exercise the Python-encoder-produced fixtures (`SGFP4DequantFixtures.h`). Once the C++ encoder exists, add a companion test that round-trips real (or realistic-shaped) weight tensors through the new C++ encoder and the existing `dequant_sgfp4_container_cpu` — this is the missing "real weights" validation called out in the milestone goal. |

## Where New Code Goes (converter-side file plan)

Modeled directly on the existing `WeightQuantAndCoding.cpp` / `RemoveParams.cpp` / `writeFb.cpp` triangle — do not invent a new integration shape:

| New/changed file | Role | Precedent it mirrors |
|---|---|---|
| `tools/converter/source/common/SGFP4Quantizer.hpp` + `.cpp` (new) | Port of `encode_sgfp4.py`'s leaf-encode, quadtree-subdivide, classify, and byte-pack functions, operating on `std::vector<float>`/raw arrays | `HQQQuantizer.hpp`/`.cpp` — a self-contained quantizer class with a `QuantizationConfig` struct and a `quantize()` entry point, called from `WeightQuantAndCoding.cpp`'s `_HQQQuant()` helper |
| `void SGFP4QuantAndCoding(std::unique_ptr<MNN::OpT>& op, const modelConfig& config, const PostTreatContext* context);` — new function, declared in `CommonUtils.hpp` next to `WeightQuantAndCoding`'s declaration (line 42), defined in `SGFP4Quantizer.cpp` or a new `SGFP4QuantAndCoding.cpp` | Per-op entry point: filters on `OpType_Convolution`/`ConvolutionDepthwise`/`Deconvolution` (same gate `WeightQuantAndCoding.cpp:80-84` uses, minus the Int8 variants), invokes the encoder, then **rewrites `op->type` to `OpType_SGFP4Dequant`** and replaces `op->main` with a populated `SGFP4DequantParamT` (`magic = kSGFP4Magic`, `dims = {oc, ic*kh*kw}` or equivalent, `external` left empty until `RemoveParams.cpp` fills it) | `WeightQuantAndCoding()` itself — same function signature, same call site |
| `writeFb.cpp::_postTreatOp` (existing, line ~30-44) | Add the `SGFP4QuantAndCoding(op, config, &context);` call, gated by the new CLI flag, positioned **before** `WeightQuantAndCoding(op, config, &context)` so the two are mutually exclusive per-op (an op already rewritten to `OpType_SGFP4Dequant` must not also fall through the `OpType_Convolution` branch of `WeightQuantAndCoding`) | Existing call ordering in the same function |
| `RemoveParams.cpp::RemoveAndStoreParam` (existing) | Add `case MNN::OpParameter_SGFP4DequantParam:` — write the encoder's raw container byte blob via `storeWeight<uint8_t>(fs, containerBytes, param->external, offset, false)`, matching the `Convolution2D` case's `storeWeight<int8_t>(fs, param->quanParameter->buffer, param->external, offset, false)` (line 43) | `Convolution2D`/`Scale`/`LayerNorm` cases in the same switch (lines 36-98) |
| `RemoveParams.cpp::loadExternalParam` (existing) | Add the matching read-back case, mirroring the `Convolution2D` branch's `fl->offset(param->external[0]); loadExternalData<uint8_t>(fl, ..., param->external[1]);` pattern (lines 147-156) | Same function, `Convolution2D` case |
| Somewhere in the new write path (either `SGFP4QuantAndCoding` or `writeFb.cpp`) | **Must explicitly set `op->externalPath = config.modelFile + ".weight"`** on the rewritten op | `SplitBlockQuantConvolution.cpp:45` — this is the one easy-to-miss step: `CPUSGFP4Dequant::onResize` and `VulkanSGFP4Dequant` both hard-require `mOp->externalPath()` to be non-null (`CPUSGFP4Dequant.cpp:50`); without this assignment the already-shipped decode Executions will return `NOT_SUPPORT` on every real-model run even if the container bytes are written correctly |
| `config.hpp` | New field(s): `bool sgfp4Quant = false;` (peer to `weightQuantBits` etc., line 42-45); optionally expose the Python encoder's tunable knobs (`eps`, per-level MSE thresholds, veto factor, hysteresis delta) as advanced CLI overrides, mirroring `encode_sgfp4.py`'s `--eps`/`--level-thresholds`/`--veto-factor`/`--hysteresis-delta` flags | Existing `weightQuantBits`/`weightQuantBlock` fields |
| `cli.cpp` | New cxxopts option block + parse block for `sgfp4Quant` (and optional tuning flags) | Lines 208-228 (declaration), 477-503 (parsing) |

## Installation / Build Wiring

This is a CMake C++ project, not an npm project — "installation" means CMakeLists wiring, not package installs:

```cmake
# tools/converter/CMakeLists.txt (or wherever WeightQuantAndCoding.cpp is listed)
# add the new files next to the existing ones:
#   source/common/SGFP4Quantizer.cpp
#   source/common/SGFP4Quantizer.hpp
# (WeightQuantAndCoding.cpp, RemoveParams.cpp, HQQQuantizer.cpp already compiled
#  into the same MNNConvertDeps target per .build/tools/converter/MNNConvertDeps.vcxproj)
```

No new `find_package`/`FetchContent`/vendoring step. No schema regeneration (`schema/generate.sh`/`generate.ps1`) unless the `.fbs` table needs a field it doesn't already have. No `makeshader.py` step — that's Vulkan GLSL-only and this milestone is CPU-side converter work (the Vulkan *decode* path is already shipped and untouched by this milestone).

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|--------------------------|
| Plain `std::vector<float>` loops for the leaf encoder (mean, S-candidate search, MSE) | Eigen / xtensor for vectorized leaf math | Never for this milestone. Leaves are at most 64x64 = 4096 elements; MNN explicitly disables RTTI/exceptions and prioritizes binary size (project CLAUDE.md). `WeightQuantAndCoding.cpp` itself uses raw loops (`findAbsMax`/`findMinMax`) for the same class of problem — match that, don't add a tensor-math dependency to the converter binary for a handful of small nested loops. |
| Native C++ port living inside `tools/converter/source/common/` | Keep `encode_sgfp4.py` as the only encoder and shell out to Python from mnnconvert | Never — this is explicitly what the milestone is fixing. `tools/fp4/quantize_fp4.py` (Ultra FP4/E2M1) already demonstrates the failure mode of this approach: it's a standalone post-hoc tool that round-trips an already-converted `.mnn` file through JSON, not integrated into the conversion pipeline. The milestone context explicitly calls this out as the pattern to NOT repeat for SGFP4 v2. |
| Reuse `include/MNN/SGFP4DequantUtils.hpp` constants/helpers from the encoder | Re-derive constants independently in converter code (as the Python reference encoder does, by design, for its "independent oracle" role) | The Python reference encoder is intentionally independent (it exists to catch decoder bugs via an independent implementation) — that rationale does NOT apply to the production C++ encoder, which should share the single C++ header with the decoder to eliminate an entire class of "encoder/decoder constant drift" bugs. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|--------------|
| Extending `IDSTEncoder::encode()` / `quanParameter` (`IDSTQuan` FlatBuffers table) to carry SGFP4 data | `IDSTQuan` is a flat, fixed-shape `weight/alpha/scale` container built for the int2-8 symmetric/asymmetric block-quant scheme; it has no concept of macroblocks, quadtree split maps, dual-mode per-leaf headers, or an FP16-packed scale+bias word. Retrofitting it would fight the schema (this exact "flat schema can't represent SGFP4" gap was already identified and resolved in `.planning/quick/260821-p1q.../SGFP4-PIVOT-ANALYSIS.md` Section 7). | The already-locked, already-schema'd `SGFP4DequantParam` + external-sidecar pattern (`{magic, offset, size}` descriptor), which mirrors `Convolution2D.external`. |
| A first-class FlatBuffers schema modeling the quadtree/macroblock/leaf structure itself | Would require `flatc` regeneration and force every schema consumer (all backends, `mnn2json`, doc generators) to understand quadtree recursion for a format that's inherently variable-size and self-describing — fighting the grain of a system that already solves "large variable-size per-tensor binary payload" twice via opaque-blob + external-file (`Extra.info`, `Convolution2D.external`). Already researched and rejected in the pivot analysis (Section 7). | The existing minimal `SGFP4DequantParam` descriptor; all container-internal parsing stays inside the Execution classes (already true for `CPUSGFP4Dequant`/`VulkanSGFP4Dequant`) and should be equally true for the new encoder — it parses/emits raw bytes per `SGFP4DequantUtils.hpp`, no FlatBuffers involvement in the container's internal layout. |
| Calling `tools/fp4/quantize_fp4.py`'s post-hoc JSON round-trip pattern as a template | It's explicitly the anti-pattern this milestone exists to avoid for SGFP4 (see milestone context: "not integrating into mnnconvert itself"). It also has a live, unrelated bug (`MAX_E2M1_VALUE=6.0` scale-calibration defect, documented in the pivot analysis) that should not be copied into new code by association. | `WeightQuantAndCoding.cpp`'s native, in-pipeline, per-op transform pattern. |
| A generic tensor/quantization framework (e.g., vendoring GGML, or a slimmed llama.cpp `ggml-quants.c`) to get "adaptive block quant for free" | No existing engine in this problem space (see Prior Art below) implements SGFP4's specific quadtree/dual-mode/ternary-veto scheme; vendoring a whole third-party quantization library to reuse ~10% of it (the general "iterate blocks, minimize round-trip error" idea) adds a large, license-bearing dependency for no real leverage, and directly conflicts with the project's binary-size priority. | Direct 1:1 port of `encode_sgfp4.py`'s already-spec-correct, already-tested logic. |

## Prior Art: Adaptive Block Quantization Embedded in a Converter (Comparison)

Researched to answer "has anyone shipped error-driven adaptive-block quantization natively inside a model converter/engine-builder, in C++, the way this milestone wants for SGFP4?" — confidence MEDIUM (WebSearch-sourced, not independently code-verified against these repos the way the MNN-internal findings above were).

| Engine | Where quantization runs | Adaptive block **size**? | Error-driven **mode/type** selection? | Relevance to SGFP4 v2 |
|---|---|---|---|---|
| **llama.cpp** (`ggml-quants.c`, `tools/quantize` / `llama-quantize`) | **Native C, inside the converter/quantizer binary itself** — no Python round-trip | No — K-quants (`Q2_K`...`Q6_K`) use a fixed hierarchical structure: 256-weight superblocks split into fixed-size sub-blocks with their own (often quantized) scale/min. Block size is not chosen per-region by an error threshold. | Partially — the `--imatrix` importance-matrix mechanism weights each element's contribution to quantization error using calibration-data activation magnitudes, influencing *which quant type* a tensor is assigned, but this is a global/per-tensor choice, not SGFP4's per-64x64-macroblock, per-leaf FP4_AFFINE-vs-T158_AFFINE choice via Eq. 5 round-trip MSE comparison. | **Closest architectural analog** for "do this natively in C++ inside the converter tool, not as a Python post-hoc script" — validates the integration *shape* this milestone is choosing (mirrors `WeightQuantAndCoding.cpp`'s existing native-C++ pattern). SGFP4's recursive quadtree down to 4x4 leaves with per-leaf mode selection is more granular than anything in K-quants' fixed hierarchy — MNN is not just replicating prior art here, it's extending past it. |
| **ONNX Runtime** (`onnxruntime.quantization` block-wise/`MatMulNBits` quantizer; RTN/GPTQ/AWQ/HQQ) | Python tool (`quantize_dynamic`/`matmul_4bits_quantizer.py`), operating on an already-exported `.onnx` file, producing a new `.onnx` — structurally the same "post-hoc external tool" pattern as MNN's own `tools/fp4/quantize_fp4.py` | No — block size (32/64/128) is a fixed user-specified parameter, uniform across the whole tensor. | No adaptive per-block algorithm selection; RTN/GPTQ/AWQ/HQQ are chosen once, globally, by the user/config. | Confirms the anti-pattern to avoid: ORT's quantizer is exactly the shape of `quantize_fp4.py` (external Python tool, not converter-native). MNN's existing int2-8 `WeightQuantAndCoding.cpp` is already architecturally ahead of ORT here; SGFP4 should extend that native-C++ advantage, not regress to ORT's pattern. |
| **TensorRT-LLM / NVIDIA Model Optimizer** (`modelopt.torch.quantization`, AWQ/GPTQ/FP4 NVFP4-MXFP4) | Python calibration library, producing a pre-quantized checkpoint later consumed by the separate TensorRT-LLM C++ engine builder | No — block size (e.g. `--awq_block_size 128`) is fixed and user-specified, uniform per tensor. | No error-driven per-block mode switch; algorithm (AWQ vs GPTQ vs FP4) is chosen once per model/run via calibration, not per-block at encode time. | Same conclusion as ONNX Runtime: calibration/quantization lives in a separate Python layer outside the C++ engine builder. Not a precedent for "adaptive per-block scheme embedded directly in the converter binary." |

**Bottom line:** no comparable inference engine researched implements SGFP4 v2's specific combination (recursive quadtree block-size adaptation + per-leaf error-driven dual-mode selection + ternary-outlier veto) natively in a C++ converter. llama.cpp's K-quants is the nearest architectural precedent for *doing native block quant in the converter binary at all* (as opposed to a Python side-tool) — which validates this milestone's chosen integration shape (`WeightQuantAndCoding.cpp`-style native C++) over the alternative (a `quantize_fp4.py`-style Python post-hoc tool). Beyond that shape-level validation, the actual quadtree/dual-mode algorithm has no external prior art to lean on — `tools/fp4/encode_sgfp4.py` (already spec-correct and already tested via `--selftest`) is the real source of truth to port, not any external codebase.

## Stack Patterns by Variant

**If the weight tensor's flattened `[O, I*kh*kw]` shape is not a multiple of 64 in both dimensions** (the common case — most conv layers won't tile evenly into 64x64 macroblocks):
- This is an open question the Python reference encoder does not answer (it only encodes pre-tiled 64x64 `tiles` passed in by the caller/test harness — see `encode_container_adaptive(tiles, ...)` in `encode_sgfp4.py`).
- The new C++ `SGFP4QuantAndCoding` entry point is the first real caller that must decide a padding/tiling policy (e.g., zero-pad the flattened weight matrix up to the next 64x64 multiple, track the true `dims` separately for the FlatBuffers `dims` field vs. the padded encode grid). This is a genuine open design gap, not something to silently improvise — flag it for the phase-planning/discuss step rather than resolving it unilaterally in STACK research.

**If a per-layer opt-out or accuracy validation is wanted before full-scale rollout:**
- Reuse the `PostTreatContext::quantInfo` per-op override mechanism (`WeightQuantAndCoding.cpp:95-109`) rather than inventing a second config-override mechanism — it already supports exactly this pattern (bits/asymmetric/block_size per `(subgraph, op_name)` pair) and could be extended with an SGFP4-specific field with minimal new protobuf surface.

## Version Compatibility

| Component | Compatible With | Notes |
|-----------|------------------|-------|
| New encoder (`half_float::half` v1.12.0) | Existing decoder (`SGFP4DequantUtils.hpp`, same vendored `half.hpp`) | Both must use the identical vendored copy at `3rd_party/half/half.hpp` — do not add a second half-float library or hand-roll conversion, since the FP16 truncation behavior (round-to-nearest-even) must match exactly for `pack_leaf_header`'s bit-for-bit outputs to match what `unpack_leaf_header` in C++ decode already expects (this is proven correct for the Python encoder via `--selftest`'s wire-format-fidelity check; the C++ port must preserve that same rounding behavior). |
| `SGFP4DequantParam` schema (already generated, flatbuffers 1.10.0) | New encoder's output | No version bump needed — the encoder only *populates* existing generated types (`SGFP4DequantParamT`), it doesn't touch `.fbs`/regenerate. If a genuinely new field turns out to be needed, regenerate via `schema/generate.sh` (Linux/macOS) or `schema/generate.ps1` (Windows, relevant on this Windows dev machine) and commit both `schema/default/*.fbs` and `schema/current/*_generated.h` together. |
| New `sgfp4Quant` CLI flag (cxxopts 2.1.0) | Existing `weightQuantBits`/`hqq` flags | Should be mutually exclusive with `weightQuantBits`/`hqq` on the same op (an op can't be both int-quantized and SGFP4-quantized) — enforce this in `_postTreatOp`'s call ordering (SGFP4 check first, `return`/skip `WeightQuantAndCoding` if the op was already rewritten to `OpType_SGFP4Dequant`), not via cxxopts-level flag conflict validation. |

## Sources

- `tools/fp4/encode_sgfp4.py` (in-tree, HIGH confidence — the algorithm to port) — full read
- `include/MNN/SGFP4DequantUtils.hpp` (in-tree, HIGH confidence — decode-side contract the encoder must match) — full read
- `tools/converter/source/common/WeightQuantAndCoding.cpp` (in-tree, HIGH confidence — converter integration pattern) — full read
- `tools/converter/source/common/RemoveParams.cpp` (in-tree, HIGH confidence — external-sidecar wiring pattern) — full read
- `tools/converter/source/common/writeFb.cpp` (in-tree, HIGH confidence — per-op post-treat call site) — read lines 1-90
- `tools/converter/source/common/CommonUtils.hpp`, `tools/converter/include/config.hpp`, `tools/converter/source/common/cli.cpp` (in-tree, HIGH confidence — CLI flag / config plumbing) — read relevant sections
- `tools/converter/source/common/HQQQuantizer.hpp` (in-tree, HIGH confidence — precedent for a self-contained quantizer class) — full read
- `schema/default/CaffeOp.fbs`, `schema/default/MNN.fbs`, `schema/current/CaffeOp_generated.h`, `schema/current/MNN_generated.h` (in-tree, HIGH confidence — confirms `SGFP4DequantParam`/`OpType_SGFP4Dequant` already exist) — grep + targeted read
- `source/backend/cpu/CPUSGFP4Dequant.cpp`, `source/shape/ShapeSGFP4Dequant.cpp` (in-tree, HIGH confidence — confirms `externalPath` requirement and decode contract) — full read
- `source/core/OpCommonUtils.hpp`/`.cpp`, `tools/converter/source/optimizer/postconvert/SplitBlockQuantConvolution.cpp` (in-tree, HIGH confidence — confirms `op->externalPath` assignment precedent) — grep + targeted read
- `.planning/quick/260821-p1q-evaluate-current-fp4-ultra-fp4-implement/SGFP4-PIVOT-ANALYSIS.md` (in-tree, HIGH confidence — locked P4 container-architecture decision, already-identified schema gaps) — full read
- `3rd_party/half/half.hpp` + `ChangeLog.txt`, `3rd_party/flatbuffers/include/flatbuffers/base.h`, `tools/converter/include/cxxopts.hpp` (in-tree, HIGH confidence — exact vendored versions) — grep for version constants
- WebSearch: "llama.cpp k-quants adaptive block quantization importance matrix source code superblock" (MEDIUM confidence, corroborated by multiple results including `ggml-org/llama.cpp` README)
- WebSearch: "ONNX Runtime quantization tool block-wise weight quantization error driven mode selection C++" (MEDIUM confidence, corroborated by `onnxruntime.ai` official docs)
- WebSearch: "TensorRT Model Optimizer INT4 AWQ block quantization engine build weight compression C++" (MEDIUM confidence, corroborated by NVIDIA TensorRT-LLM/Model-Optimizer docs)

---
*Stack research for: SGFP4 v2 real-weight encoder + native mnnconvert integration*
*Researched: 2026-08-25*
