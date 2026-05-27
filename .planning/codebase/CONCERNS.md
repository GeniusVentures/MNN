# Codebase Concerns

**Analysis Date:** 2026-05-27

## Tech Debt

### Accumulated TODO/FIXME/HACK Comments

**Scope:** Over 150 TODO/FIXME/HACK comments across `source/` (137 in `.cpp`, 20 in `.h`), plus more in `transformers/`.

**By Backend (TODOs):**
| Backend | Count | Representative |
|---------|-------|----------------|
| CPU | 37 | `CPUSoftmax.cpp:201` — "Fix x86 compute error"; `CommonOptFunction.cpp:4630` — "MNNPackTranspose and MNNUnpackTranspose is reverted" |
| Vulkan | 13 | `VulkanPipeline.cpp:371` — "Set blend Info"; `VulkanImage.cpp:30` — "FIXME: find better method" |
| OpenCL | 12 | `reduction_buf_mnn_cl.cpp:5-8` — Four embedded TODO comments in shader string for missing features |
| Neuropilot | 7 | `neuron_usdk_executor.cpp:380` — "FIXME: Remove workaround once shared weights 0 size bug is resolved" |
| CUDA | 6 | `CUDABackend.cpp:50` — "Search CUDA Device info and use best one" |
| TensorRT | 6 | `TRTBackend.cpp:207` — "FIXME: Compute input info" |
| QNN | 5 | `QNNBackend.cpp:1916` — "TODO: Fix bug and complete"; Chinese-language TODO in `QNNStridedSlice.cpp:223` |

**Core (37 TODOs):**
- `Session.cpp:301` — "Separate Schedule and Malloc"
- `Schedule.cpp:113` — "FIXME: Support Auto determine"
- `Pipeline.cpp:491` — "FIXME: Remove onMaskOpReady in future"; `:843` — "Recompute release mask"
- `GeometryComputerUtils.cpp:204` — "FIXME: Find better way to may compability for old model"

**Years-Old Issues:** Many TODOs reference specific people or un-patched bugs (e.g., `ShapeStridedSlice.cpp:268` — "there is one in linfeng's faster"), suggesting some have been open for extended periods.

### Deprecated APIs

**`MNN_SUPPORT_DEPRECATED_OP` guard:** 18 files in `source/` wrap entire implementations in this preprocessor guard:
- **CPU backend:** 11 files for quantized ops (`CPUQuantizedMaxPool.cpp`, `CPUQuantizedSoftmax.cpp`, `CPUQuanConvolutionDepthwise.cpp`, `CPUEltwiseInt8.cpp`, `CPUMoments.cpp`, `CPUInstanceNorm.cpp`, etc.)
- **Shape inference:** `ShapeQuantizedMaxPool.cpp`, `ShapeQuantizedAvgPool.cpp`, `ShapeMoments.cpp`
- **Geometry:** `GeometryConv3D.cpp` uses `MNN_SUPPORT_DEPRECATED_OPV2`

**Impact:** These ops compile to empty stubs unless explicitly enabled. The old quantized op formats persist in exported models but receive no active maintenance or optimization. The geometry decomposer (`GeometryComputerUtils.cpp:204`) has FIXME comments explicitly about maintaining compatibility with old models.

### Conditional Compilation Debt

**`MNN_SUPPORT_TRANSFORMER_FUSE` guard:** Entire Vulkan attention pipeline (`VulkanAttention.cpp`, 1505 lines; `VulkanLinearAttention.cpp`, 243 lines) and their CUDA/Metal/OpenCL equivalents are wrapped in this macro. If disabled at build time, all LLM inference silently falls back to decomposed ops with dramatically worse performance and no warning. Multiple build configurations diverge the codebase's effective behavior.

**`MNN_LOW_MEMORY` guard:** Used in LLM engine and some backends to select different execution paths. Testing both configurations requires separate builds.

### Hardcoded Magic Numbers

**Strassen penalty constants:**
- `source/backend/cpu/compute/StrassenMatmulComputor.cpp:277` — `const float penalty = core->penalty; // FIXME: Find better way to set it`
- `source/backend/opencl/execution/buffer/StrassenMatmulOpenCLComputor.cpp:250` — `const float penalty = 30.0; // FIXME: Find better way to set it`

These are cross-backend duplicated constants with inconsistent derivation (one uses a member, one hardcoded). No documentation on how these values were calibrated or when they need adjustment.

### KVCache Resize Inefficiency

`VulkanAttention.cpp:174-230` — When KVCache needs to grow, the resize allocates new buffers and copies old content via per-row `VkBufferCopy` regions (up to `kvHeadNum * d4Size` regions for key, `kvHeadNum` for value). For large models this generates thousands of copy commands. A ring buffer or virtual addressing approach would eliminate the repack cost.

## Known Bugs

### x86 Softmax Compute Error

**Symptoms:** Numerical error on x86 CPU backend during softmax computation.
**Files:** `source/backend/cpu/CPUSoftmax.cpp:201`
**Trigger:** Certain input shapes on x86.
**Workaround:** x86 path uses a separate implementation from ARM; ARM path works correctly.
**Code:**
```cpp
//TODO: Fix x86 compute error and use the same function
```

### QNN Backend Incomplete Implementation

**Symptoms:** `QNNBackend.cpp:1916` marked as "TODO: Fix bug and complete" on a critical path. `QNNStridedSlice.cpp:223` has incomplete handling of `newAxisMask` and `ellipsisMask` parameters. `QNNConvDepthwise.cpp:265` lacks support for asymmetric quantization.
**Files:** `source/backend/qnn/backend/QNNBackend.cpp`, `source/backend/qnn/execution/QNNStridedSlice.cpp`, `source/backend/qnn/execution/QNNConvDepthwise.cpp`
**Risk:** QNN backend used for Qualcomm NPU inference on Snapdragon devices. Incomplete ops cause silent fallback or incorrect results.

### Neuropilot Shared Weights Bug

**Symptoms:** Neuron Adapter has a "shared weights 0 size bug" requiring workaround code.
**Files:** `source/backend/neuropilot/mtk/executor/neuron_usdk_executor.cpp:380`
**Trigger:** Certain model weight configurations on MediaTek Neuron runtime.
**Fix approach:** Awaiting upstream fix in Neuron Adapter; workaround in place but fragile.

### OpenCL Reduction Known Issues

**Symptoms:** Multiple documented gaps in reduction implementation.
**Files:** `source/backend/opencl/execution/cl/reduction_buf_mnn_cl.cpp:5-8`, `source/backend/opencl/execution/cl/reduction_mnn_cl.cpp:4-7`
**Known gaps:**
- Does not support reduce across batch dimension
- Does not support `keep_dim=False`
- Channel reduce result re-pack problem (output corruption)

## Security Considerations

### Unsafe String Formatting

**Risk:** Buffer overflow via `sprintf` with potentially attacker-controlled input.
**Files:**
- `source/core/TensorUtils.cpp:877` — `sprintf(info, "size: %d, %d, %d; src: %d, %d, %d, %d; dst: %d, %d, %d, %d", ...)` — fixed-size buffer, variable content
- `source/core/Pipeline.cpp:134` — `sprintf(buffer, "%d", index)` — controlled by op count, but no bounds check on buffer

**Current mitigation:** Callers use fixed-size stack buffers. No validation that output fits.
**Recommendations:** Replace with `snprintf` and bounds checking on all paths.

### Unchecked memcpy Operations

53 `memcpy`/`memset` sites in `source/core/`. Most operate on buffer sizes derived from model metadata (FlatBuffers). Malformed model files could trigger out-of-bounds writes:
- `source/core/ConvolutionCommon.cpp:490-851` — Multiple `memcpy` calls using sizes from `quan->weightSize()`, `quan->alpha()->size()` — these are model-provided values
- `source/core/FileLoader.hpp:63` — Direct `memcpy` without bounds validation

**Current mitigation:** Model converter validates inputs. No runtime validation of FlatBuffer-provided sizes.

### Vulkan Memory Barrier Correctness

**Risk:** Incorrect Vulkan synchronization leading to data races or undefined behavior.
**Files:** `source/backend/vulkan/buffer/execution/VulkanAttention.cpp:665-674`
**Description:** KV cache update after `vkCmdDispatch` uses a full `VkMemoryBarrier` instead of `VkBufferMemoryBarrier` because "KV cache buffers may be reallocated ... descriptor set updated there, so we must not record a VkBufferMemoryBarrier with a stale VkBuffer handle." This is correct but fragile — any future code change that removes that allocation pattern could introduce a subtle synchronization bug.

### No Input Validation for Model Files

**Risk:** The Interpreter (`source/core/Interpreter.cpp:111`) copies raw buffer directly via `memcpy`. There is no structural validation of the FlatBuffer schema before processing. A malformed model could trigger crashes, undefined behavior, or potentially exploitable memory access patterns through unchecked size fields propagated to `memcpy` calls.

## Performance Bottlenecks

### CPU-Side Causal Mask Generation

**Files:** `source/backend/vulkan/buffer/execution/VulkanAttention.cpp:1222-1238`
**Problem:** When using KV cache with a lower-triangular causal mask (scalar mask placeholder), the implementation allocates a CPU-side tensor, fills it with a double-nested loop `for(q) { for(k) { ... }}`, then copies to GPU. This is O(queryLen * totalLen) on the CPU.
**Impact:** For large context windows (e.g., 128K tokens), this becomes a latency bottleneck.
**Improvement path:** Generate the mask directly on GPU via a compute shader.

### Vulkan Prefill Multi-Pass Dispatch Overhead

**Files:** `source/backend/vulkan/buffer/execution/VulkanAttention.cpp:862-915`
**Problem:** K-block prefill loops over `kStart` in steps of 512, dispatching 4 compute shaders per block (QK, Softmax, QKV, barriers). For a 128K context this is ~250 iterations × 4 dispatches = 1000 pipeline dispatches.
**Impact:** Pipeline dispatch overhead dominates for large sequences, even though the O(qLen*totalLen) memory is avoided.
**Improvement path:** Coalesce the QK+Softmax passes into a single shader, or use indirect dispatch.

### Vulkan KVCache Expand Full Repack

**Files:** `source/backend/vulkan/buffer/execution/VulkanAttention.cpp:174-230`
**Problem:** When KVCache grows, the entire cache must be repacked because the row stride (maxLen) changes. This generates `kvHeadNum * d4Size` copy regions for key data and `kvHeadNum` for value data. For a model with 8 KV heads and 128-dim heads, d4Size=32, producing 256 individual `VkBufferCopy` commands just for the key repack.
**Impact:** Context window expansion causes a latency spike that grows linearly with number of KV heads and head dimension.

### Vulkan Attention Slow Paths

**Files:** `source/backend/vulkan/buffer/execution/VulkanAttention.cpp:172-186`
**Fallback decode path:** When subgroup operations are not supported or TurboQuantK is enabled, decode falls back to `glsl_attention_fused_packed` which uses a generic tiled dispatch (x=headNum/8, y=queryLen/8) instead of the optimized subgroup decode. Subgroup operations require both hardware support and a specific VkSubgroupFeatureFlags combination (`BASIC | ARITHMETIC`). Availability varies widely across mobile GPUs.

### Strassen Matrix Multiply Heuristic

**Files:**
- `source/backend/cpu/compute/StrassenMatmulComputor.cpp:277`
- `source/backend/opencl/execution/buffer/StrassenMatmulOpenCLComputor.cpp:250`

**Problem:** Hardcoded penalty constant determines when to use Strassen vs. standard matmul. This constant (30.0 in OpenCL) is not validated against modern hardware and is explicitly marked FIXME in both locations. Wrong thresholds cause suboptimal algorithm selection.

## Fragile Areas

### Vulkan Image Backend (Partially Abandoned)

**Files:** `source/backend/vulkan/image/`
**Why fragile:** The image backend is missing 10 critical ops that exist in the buffer backend:
- No `OpType_Attention` — LLM inference impossible
- No `OpType_LinearAttention` — Linear attention models impossible
- No `OpType_LayerNorm` — Most transformer architectures fail
- No `OpType_MatMul` — Matrix multiplication (critical for transformer FFN)
- No `OpType_PReLU`, `OpType_OneHot`, `OpType_Range`, `OpType_Select`, `OpType_Extra`

**Additionally:** `glslbackup/` directory at `source/backend/vulkan/image/execution/glslbackup/` contains 7 abandoned uint8 format shaders (370 lines total) that are no longer compiled into the shader map. The image backend appears to be kept for basic CNN inference only.

**Safe modification:** All modern LLM/diffusion work targets the buffer backend. Image backend changes should be limited to CNN ops (conv, pool, binary, softmax).

### Vulkan Shader Autogeneration Pipeline

**Files:**
- `source/backend/vulkan/buffer/compiler/makeshader.py` — 23K Python script
- `source/backend/vulkan/buffer/compiler/AllShader.cpp` — **138,702 lines** (largest file in repo)
- `source/backend/vulkan/buffer/shaders/AllShader.h` — Generated declarations
- `source/backend/vulkan/buffer/compiler/VulkanShaderMap.cpp` — Generated map

**Why fragile:**
1. 140 GLSL `.comp` shader source files feed into `makeshader.py`
2. Each `.comp` is compiled twice (FP32 and FP16 variants) via `glslangValidator` → SPIR-V, then optionally optimized via `spirv-opt`
3. Output is embedded as C byte arrays in `AllShader.cpp`
4. **If you edit a shader, you MUST run `makeshader.py`** to regenerate `AllShader.cpp`, `AllShader.h`, and `VulkanShaderMap.cpp` — the build won't pick up `.comp` changes otherwise

**Risk:** Developers not familiar with this pipeline edit `.comp` files, build succeeds (since only the auto-generated `.cpp` is compiled), and wonder why their changes don't take effect.

**Duplicate pipeline:** Both `buffer/compiler/` and `image/compiler/` have independent `makeshader.py` scripts and generated outputs.

### QNN Backend Platform Dependency

**Files:** `source/backend/qnn/`
**Why fragile:** The QNN backend requires Qualcomm's proprietary SDK headers (`QnnTypeMacros.hpp` has a TODO about SNPE build compatibility). It produces separate `.so` library that must match the device firmware version. The backend has bilingual (Chinese/English) TODO comments and incomplete implementations. Build failures common when QNN SDK version mismatches.

### No RTTI / No Exceptions Codebase

**Build flags:** `-fno-rtti -fno-exceptions` in `CMakeLists.txt:613`
**Impact:**
- All error handling uses `ErrorCode` enum returns (e.g., `OUT_OF_MEMORY`, `NO_ERROR`) — checked manually at every call site
- Dynamic dispatch uses creator function tables (`addCreator(OpType_X, new Creator)`) instead of RTTI
- No `dynamic_cast` possible — all downcasts are `static_cast`, requiring the developer to guarantee type correctness
- Any third-party library that throws exceptions becomes incompatible
- OOM conditions cannot be caught — they terminate the process

### Manual GPU Memory Management in Vulkan

**Files:** `source/backend/vulkan/buffer/execution/VulkanAttention.cpp`
**Pattern:** The `VulkanAttention` constructor calls `allocUniform`, `getPipeline`, `createSet`. The destructor calls `onReleaseBuffer` × 9, `recycleUniform` × 2. Temporary buffers in `onEncode` are acquired/released via explicit `onAcquireBuffer`/`onReleaseBuffer` calls. A premature return or exception (simulated via error code) must manually release all acquired buffers.
**Risk:** Resource leaks on error paths. The decode path (`onEncode` at line 924-954) has a manual cleanup block that releases all 7 prefill temporaries — if a new temporary is added but the decode cleanup block is not updated, it leaks.

## Scaling Limits

### Vulkan KCache Single Buffer Allocation

**Files:** `source/backend/vulkan/buffer/execution/VulkanAttention.cpp:146-297`
**Current capacity:** KVCache allocates as a single contiguous `VulkanBuffer` per KV type (key, value, packedKey, packedValue) sized `maxLen * kvHeadNum * headDim * sizeof(element)`.
**Limit:** `maxBufferSize` of the Vulkan device (typically 256MB-4GB). For a 70B-parameter model with 128K context: 128000 × 8 × 128 × 2 bytes ≈ 262 MB for key cache alone (with FP16). Large models with long contexts may approach or exceed device limits.
**Scaling path:** PagedAttention-style block-based allocation, or offload to host memory with streaming.

### Prefill Intermediate Memory

**Files:** `source/backend/vulkan/buffer/execution/VulkanAttention.cpp:718-788`
**Current allocation:** K-block prefill allocates per-head-row QK block buffer (queryLen/4 × kBlock4 elements), weight block buffer (same size), plus state buffers (m, l, alpha). For very large models with many heads × large query lengths, this can exceed available VRAM.
**Limit:** QK block buffer = `rowCount * kBlock4 * sizeof(float)` = `queryLen * headNum * 512 * 4`. For 32 heads × 4096 query tokens = 32 × 4096 × 512 × 4 = 256 MB. Multiple such buffers exist simultaneously (QK + W blocks).

## Backend Maturity Comparison

| Backend | Files | Ops (approx) | Attention | INT4 Conv | Test Coverage | Status |
|---------|-------|-------------|-----------|-----------|---------------|--------|
| CPU | 333 | ~120 | Yes (2 impls) | 15 files | Moderate (unit tests) | **Reference** |
| OpenCL | 220 | ~90 | Yes (3 impls) | 37 files | Low | Active |
| CUDA | 172 | ~80 | Yes (FlashAttn) | 4 files | Low | Active (LLM) |
| Vulkan (buffer) | ~90 | ~30 | Yes (prefill+decode) | 7 files | **None** | Active (LLM) |
| Vulkan (image) | ~55 | ~22 | **No** | 0 files | **None** | Maintenance |
| TensorRT | 91 | ~60 | Partial | 0 files | Low | Active |
| Metal | 41 | ~35 | Yes | 7 files | Low | Active |
| HIAI | 96 | ~50 | No | 0 files | Low | Vendor |
| CoreML | 40 | ~30 | No | 0 files | Low | Vendor |
| Neuropilot | 62 | ~40 | Converter only | 1 file | Low | Vendor |
| QNN | 62 | ~45 | Yes | 0 files | Low | Vendor |
| NNAPI | 35 | ~20 | No | 0 files | Low | Maintenance |
| OpenGL | 50 | ~15 | No | 0 files | Low | Legacy |
| ARM82 | 18 | Helper funcs | No | 1 file | Low | Specialized |

### Vulkan-Specific Maturity Notes

**Buffer backend (primary):**
- LLM support: Attention (prefill with K-blocking + online softmax, decode with subgroup opt, fused packed fallback), LinearAttention (Gated Delta Rule), LayerNorm, MatMul
- CNN support: Convolution (standard, depthwise, 1x1 coop, 1x1 general), Deconvolution, Pooling, Raster
- Quantization: INT4 weight support (dequant on-the-fly during conv, GemV)
- Missing ops: No Embedding, no RotaryPositionEmbedding
- **Zero automated tests**

**Image backend (secondary):**
- CNN-only: Convolution, Deconvolution, Pooling, Raster, Binary, Unary, Softmax, Interp
- Missing all LLM ops (Attention, LinearAttention, LayerNorm, MatMul)
- **Zero automated tests**
- Contains abandoned `glslbackup/` directory with unused uint8 shaders

**Vulkan subgroup operations:**
- Used for optimized decode attention (`attention_decode_q1_subgroup`)
- Requires `VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT`
- Falls back silently to generic path if unavailable — performance cliff on devices without subgroup support (older Mali GPUs)

## FP4 / INT4 / Quantization State

### What Exists

**INT4 (4-bit integer) quantization:** Well-supported across CPU, OpenCL, Vulkan buffer, Metal, and CUDA backends for convolution weight compression. Weights are stored as 4-bit integers and dequantized on-the-fly during inference.

**Vulkan buffer INT4 shaders:**
- `glsl_convolutionint4_comp` (with RELU/RELU6/FP16 variants)
- `glsl_conv1x1_int4_weight_prepare_comp`
- `glsl_int4_weight_to_pack_comp` (with FP16 variant)
- `glsl_int4_weight_to_coop_comp` (with FP16 variant)
- `glsl_gemv_dequant_int4_comp` (with FP16 variant)
- Files: Built into autogenerated `AllShader.cpp` via `makeshader.py`

**LLM quantization (Python export-time):**
- `transformers/llm/export/utils/hqq_quantizer.py` — HQQ (Half-Quadratic Quantization) for weight compression during model export. Implements chunked quantization with automatic memory estimation to avoid OOM on large models.
- `transformers/llm/export/utils/awq_quantizer.py` — AWQ (Activation-aware Weight Quantization)
- `transformers/llm/export/utils/smooth_quantizer.py` — SmoothQuant for activation quantization
- `transformers/llm/export/utils/omni_quantizer.py` — OmniQuant pipeline combining smooth quantization + weight optimization
- CLI: `--hqq`, `--quant_bit`, `--quant_block`, `--embed_bit` [16, 8, 4] flags on `llmexport.py`

**TurboQuant K/V cache compression (Vulkan):**
- `VulkanAttention.cpp` supports runtime KV cache compression using turboquant format
- Block size: 32 bytes (kTurboQuantKBlockSize = 32, kTurboQuantKPackedWordCount = 4)
- Enabled via `KVMeta` flags from the runtime
- Only block size 32 is supported; format must be 0

### What Does NOT Exist

**FP4 (4-bit float / NF4):** No FP4 support detected in the inference engine. The codebase uses INT4 for sub-byte quantization, not float4. The `NF4` / `float4` references found are Metal shader vector types (`float4` = `vec4`), not 4-bit floating-point formats. Actual 4-bit float quantization (NF4 as used in QLoRA) is not implemented.

**FP4 runtimes across backends:** None. INT4 is the only sub-8-bit format supported at inference time.

**Missing INT4 in backends:** CoreML, HIAI, NNAPI, OpenGL, QNN, TensorRT have zero INT4 quantization files — models with INT4 weights will either fail or fall back to CPU.

**Dynamic quantization:** All quantization is static (applied at export/conversion time). There is no runtime dynamic quantization or calibration.

## Code Duplication Patterns

### Vulkan Buffer vs Image Backend Duplication

**18 ops duplicated** across `source/backend/vulkan/buffer/execution/` and `source/backend/vulkan/image/execution/`:
`VulkanArgMax`, `VulkanBasicExecution`, `VulkanBinary`, `VulkanConvolution`, `VulkanConvolutionImpl`, `VulkanDeconvolution`, `VulkanGridSample`, `VulkanInterp`, `VulkanLoop`, `VulkanMatMul`, `VulkanPool`, `VulkanROIPooling`, `VulkanRaster`, `VulkanReduce`, `VulkanResize`, `VulkanScale`, `VulkanSoftmax`, `VulkanUnary`

Each pair represents independent implementations that must be maintained separately. Changes to op semantics (e.g., new binary op types, new convolution features) must be ported to both.

### Attention Across Backends

Attention is implemented in 7+ backends with independent codebases:
- **CPU:** `CPUAttention.cpp`, `CPULinearAttention.cpp`
- **CUDA:** `AttentionExecution.cu`, `LinearAttentionExecution.cu` + FlashAttention plugins (FmhaV2, Fmhca)
- **Metal:** `MetalAttention.mm`, `MetalLinearAttention.mm` with embedded Metal Shading Language in headers
- **OpenCL:** `AttentionBufExecution.cpp`, `LinearAttentionBufExecution.cpp`, `SelfAttentionBufExecution.cpp` + `.cl` kernels
- **Vulkan:** 14 GLSL compute shaders + `VulkanAttention.cpp` (1505 lines), `VulkanLinearAttention.cpp` (243 lines)
- **QNN:** `QNNAttention.cpp`
- **Neuropilot:** `AttentionConverter.cpp` (converter only)

Each backend independently implements masking, KV cache management, GQA (grouped query attention), and scale factors. A bug fix or feature addition (e.g., sliding window attention, ALiBi) requires N parallel implementations.

### Softmax Across Backends

Softmax is duplicated across 11 backends (CPU, CUDA, HIAI, Metal, OpenCL, OpenGL, QNN, TensorRT, Vulkan buffer, Vulkan image, RISC-V RVV), each with independent implementations. The Vulkan prefill attention also embeds an online softmax variant that differs from the standalone VulkanSoftmax implementation.

### Strassen Penalty Duplication

The Strassen-vs-standard matmul penalty constant is independently defined in:
- `source/backend/cpu/compute/StrassenMatmulComputor.cpp:277`
- `source/backend/opencl/execution/buffer/StrassenMatmulOpenCLComputor.cpp:250`

Different derivation (one from `core->penalty`, one hardcoded `30.0f`), both marked FIXME.

## Test Coverage Gaps

### ZERO Vulkan Tests

**What's not tested:** The entire Vulkan backend (buffer + image) has no dedicated test files. Searching `test/` for `*vulkan*` or `*Vulkan*` returns zero results.
**Files:** N/A (no tests exist)
**Risk:** Every Vulkan op implementation ships to production with zero automated validation. Bugs manifest as incorrect model outputs with no regression protection.
**Priority:** **Critical**

### OpenCL Reduction Tests

**What's not tested:** Reduction operations with batch dimension, `keep_dim=False` mode, and channel reduction re-pack — all known to have issues per embedded TODO comments.
**Files:** `source/backend/opencl/execution/cl/reduction_buf_mnn_cl.cpp`, `source/backend/opencl/execution/cl/reduction_mnn_cl.cpp`
**Risk:** Silent incorrect outputs for unsupported configurations.
**Priority:** High

### QNN / Neuropilot / CoreML / NNAPI Backends

**What's not tested:** Vendor-specific backends have no dedicated tests in `test/`. They rely entirely on integration-level model tests that may not exercise all op code paths.
**Risk:** Vendor SDK version updates silently break functionality.
**Priority:** Medium

### LLM Attention Edge Cases

**What's not tested:** Multi-query attention scenarios with KV cache > maxLen reallocation, TurboQuantK/V cache compression with all block size combinations, prefill/decode transition when queryLen changes.
**Files:** `transformers/llm/engine/` has no dedicated attention unit tests.
**Risk:** LLM inference correctness degrades after cache expansion or quantization format changes.
**Priority:** Medium

## Missing Critical Features

### No Vulkan Embedding / Rotary Embedding Ops

**Problem:** Embedding lookup and RoPE (Rotary Position Embedding) are not implemented as Vulkan compute ops. These fall back to CPU, requiring CPU-GPU transfers for every token in every layer.
**Files affected:** Any transformer model using RoPE (LLaMA, Qwen, Mistral, etc.)
**Performance impact:** For decode (auto-regressive generation), each token requires CPU-side RoPE computation + GPU upload, adding latency per layer.
**Workaround:** None — always falls back.

### No FP4 (4-bit Float) Quantization

**Problem:** MNN supports INT4 (4-bit integer) but not FP4/NF4 (4-bit floating point as used in QLoRA and modern quantization-aware training). Models exported with NF4 weights cannot be loaded.
**Blocks:** Deployment of QLoRA-fine-tuned models; compatibility with the growing ecosystem of 4-bit float formats.

### No Vulkan Attention for Image Backend

**Problem:** The Vulkan image backend has no Attention op at all. While the buffer backend is the focus for LLM, the image backend is the default for many CNN models on Android. Any model combining CNN feature extractors with attention layers (e.g., vision transformers) will incur a backend switch cost.
**Files:** `source/backend/vulkan/image/execution/` — no `*Attention*` files

### Incomplete QNN Backend

**Problem:** `QNNBackend.cpp:1916` marked as incomplete with a TODO to "fix bug and complete." Operations like StridedSlice miss edge case handling (newAxisMask, ellipsisMask). ConvDepthwise lacks asymmetric quantization support.
**Blocks:** Full model coverage on Qualcomm NPU.

---

*Concerns audit: 2026-05-27*
