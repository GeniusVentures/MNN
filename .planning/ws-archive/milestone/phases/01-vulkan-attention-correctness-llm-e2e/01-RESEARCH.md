# Research: Phase 1 — Vulkan Attention Correctness & LLM E2E

**Researched:** 2026-05-27
**Status:** Complete

## Executive Summary

Plan 01 (VULK-06, VULK-07 — buffer barriers and GPU mask generation) is **fully implemented** in the current codebase. Plans 02 (tests) and 03 (E2E LLM validation) remain to be executed. The Vulkan Attention implementation (VulkanAttention.cpp, 1567 lines) is already production-quality with proper buffer barrier synchronization, GPU-side causal mask generation, flash-attention-style prefill tiling with online softmax, and subgroup-optimized decode.

## Implementation Status Audit

### Plan 01: COMPLETE ✓

| Requirement | Status | Evidence |
|-------------|--------|----------|
| VULK-06 (buffer barriers) | DONE | `VkBufferMemoryBarrier` at lines 683-735 of VulkanAttention.cpp; zero `VkMemoryBarrier` instances found (0 grep hits) |
| VULK-07 (GPU mask gen) | DONE | `attention_mask_gen.comp` exists (23 lines); `mMaskGenPipeline`/`mMaskGenSet` created in constructor (lines 356-368); dispatch in `onBeforeExecute` (lines 1269-1300) |
| Shader autogen | DONE | AllShader.cpp (4 refs), AllShader.h (4 refs), VulkanShaderMap.cpp (2 refs) all contain `attention_mask_gen` |

**VULK-06 detail — Buffer barrier implementation:**
The `onEncode` method at lines 683-735 dynamically accumulates `VkBufferMemoryBarrier` entries for KV cache buffers. The `std::vector<VkBufferMemoryBarrier>` includes:
- `mKVCache->key` buffer (always)
- `mKVCache->value` buffer (always)
- `mKVCache->packedKey` (conditional on `useTurboQuantK`)
- `mKVCache->packedValue` (conditional on `useTurboQuantV`)

Each barrier specifies `srcAccessMask = SHADER_WRITE | TRANSFER_WRITE`, `dstAccessMask = SHADER_READ | TRANSFER_READ`, `offset = 0`, `size = VK_WHOLE_SIZE`. The barrier is issued via `vkCmdPipelineBarrier` with source/destination stage masks set to `COMPUTE_SHADER | TRANSFER`.

**VULK-07 detail — GPU mask generation:**
- Shader (`attention_mask_gen.comp`): `local_size_x = 256`, computes `causalLimit = pastLen + q`, outputs `0.0f` or `-FLT_MAX` per element
- C++ wiring: `mMaskGenUniform` (4 ints), `mMaskGenPipeline` (created with `{UNIFORM_BUFFER, STORAGE_BUFFER}` descriptor types), `mMaskGenSet`
- Dispatch path: creates `mSyntheticMask` Tensor if dimensions changed, writes uniform with `{queryLen, totalLen, pastLen, 0}`, dispatches `UP_DIV(queryLen * totalLen, 256)` workgroups, submits and waits before proceeding to attention shaders
- After dispatch, sets `hasMask=1`, `maskQlen=queryLen`, `maskKvlen=totalLenForCompute`, `lowerTriangularMask=0` — downstream shaders consume the synthetic mask buffer identically to a user-provided mask

### Plan 02: NOT STARTED

No test files exist in `test/op/` for Vulkan Attention or LinearAttention. Required deliverables:
- `test/op/VulkanAttentionTest.cpp` (VULK-01, VULK-03, VULK-04, VULK-05)
- `test/op/VulkanLinearAttentionTest.cpp` (VULK-02, VULK-05)

### Plan 03: NOT STARTED

E2E LLM validation via `llm_demo` has not been performed.

## Codebase Patterns for Remaining Work

### Pattern: Vulkan Backend Test via RuntimeManager

Source: `test/op/AttentionTest.cpp:65-91` (`_makeAttentionModule`)

```cpp
// Key: use RuntimeManager with ScheduleConfig to force Vulkan backend
MNN::ScheduleConfig config;
config.type = MNN_FORWARD_VULKAN;  // force Vulkan
MNN::BackendConfig bnConfig;
bnConfig.precision = BackendConfig::Precision_High;
bnConfig.memory = BackendConfig::Memory_High;
config.backendConfig = &bnConfig;
config.numThread = 1;
std::shared_ptr<Executor::RuntimeManager> rtmgr(
    Executor::RuntimeManager::createRuntimeManager(config));
rtmgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, &gMeta);
rtmgr->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
std::shared_ptr<Module> m(Module::load({}, {}, buffer.data(), buffer.size(), rtmgr));
```

**Critical:** `Module::load` with `RuntimeManager` is the ONLY way to force a specific backend for an op defined via Express `Variable::create(Expr::create(...))`. Direct `Interpreter::createSession` with a Vulkan config also works, but the Module pattern is used in existing attention tests.

### Pattern: Test Registration

```cpp
// Register: MNNTestSuiteRegister(ClassName, "category/subcategory/name")
MNNTestSuiteRegister(VulkanAttentionCorrectnessTest, "op/vulkan/attention_correctness");
```

The `MNNTestSuiteRegister` macro creates a static `MNNTestRegister<ClassName>` that auto-registers on program startup. Test names follow hierarchical string convention: `op/vulkan/attention_correctness`.

### Pattern: Vulkan Backend Availability Check

Source: `test/core/BackendTest.cpp:272-294`

```cpp
// Check if Vulkan runtime is available before running tests
auto creator = MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN);
if (nullptr == creator) {
    // Vulkan not available — skip test gracefully (return true, not false)
    return true;
}
```

### Pattern: CPU Reference via Express API

Source: `test/op/AttentionTest.cpp:110-157` (`_computeAttentionExpr`)

The reference implementation uses MNN's Express API to construct the same computation with CPU backend:
- `_Reshape`, `_Transpose`, `_MatMul`, `_Softmax` for standard attention
- All variables are `VARP` (variable pointer) types
- The Express API automatically runs on CPU (default backend)

### Pattern: Naive CPU Reference for Complex Ops

Source: `test/op/LinearAttentionTest.cpp:28-178` (`NaiveLinearAttention`)

For ops with complex multi-step logic (like LinearAttention's Conv1D + SiLU + Split + GQA + L2Norm + Gated Delta Rule), the reference is a plain C++ implementation:
- `std::vector<float>` for all state
- `forward()` method takes raw float pointers
- Steps: Conv1D depthwise → SiLU activation → Split QKV → GQA broadcast → L2Norm → Gated Delta Rule recurrence
- This is the ONLY reliable reference — no Express API equivalent exists for the full LinearAttention pipeline

### Pattern: Float Comparison

Source: `test/TestUtils.h`

```cpp
// Absolute error check
bool checkVector(const T* result, const T* rightData, int size, T threshold);

// Relative error check (PREFERRED for attention due to varying magnitudes)
bool checkVectorByRelativeError(const T* result, const T* rightData, int size, T rtol);
```

**Threshold guidance for attention:**
- Standard attention: `rtol = 0.01` (single MatMul + Softmax, low accumulation)
- LinearAttention (gated delta rule): `rtol = 0.02` (multi-step: Conv1D + SiLU + L2Norm + recurrence, higher accumulation error)

### Pattern: Guard Macros

All Vulkan attention code is gated behind `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE`. Test files must use the same guard. The CMake flag `-DMNN_SUPPORT_TRANSFORMER_FUSE=ON` must be set for build.

### Pattern: Tensor Creation In Tests

```cpp
// Create random input tensors
auto Q = _Input({1, NumHead, queryLen, HeadDim}, NCHW);
auto K = _Input({1, KvNumHead, keyLen, HeadDim}, NCHW);
auto V = _Input({1, KvNumHead, keyLen, HeadDim}, NCHW);

// Fill with random data using writeMap
auto qPtr = Q->writeMap<float>();
for (int i = 0; i < Q->getInfo()->size; ++i) {
    qPtr[i] = randomGen();  // varies per invocation
}
```

Note: existing AttentionTest uses `#define GENERATE_TOKENS 128` but `rand()` for variability. Tests should NOT use fixed seeds to catch deterministic bugs across dimensions.

### Pattern: Build and Run

```bash
# Build
mkdir -p build && cd build
cmake .. -DMNN_BUILD_TEST=ON -DMNN_VULKAN=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
make -j$(nproc) run_test.out

# Run Vulkan-specific tests
./run_test.out 'op/vulkan/'
```

`run_test.out` auto-globs all `.cpp` under `test/` via CMake — no manual CMakeLists.txt edits needed.

## VulkanAttention Architecture (Critical for Test Understanding)

### Class Hierarchy
```
VulkanBasicExecution (base)
  └── VulkanAttention (1567-line implementation)
```

### Key Member Variables
| Member | Purpose |
|--------|---------|
| `mKVCache` | `shared_ptr<KVCache>` — holds key/value/packedKey/packedValue VulkanBuffers |
| `mMeta` | `KVMeta*` — page table metadata (previous, remove, add, reserve[]) |
| `mParam` | Uniform buffer for `GpuParam` struct (s0-s2 ivec4, f0 vec4) |
| `mMaskGenPipeline`/`mMaskGenSet` | GPU mask generation compute pipeline |
| `mUpdatePipeline`/`mUpdateSet` | KV cache update compute pipeline |
| Prefill pipelines | `mInitStatePipeline`, `mQKBlockFullPipeline`, `mSoftmaxOnlinePipeline`, etc. |
| Decode pipelines | `mAttentionPipeline`, `mDecodeQ1SubgroupPipeline`, `mDecodeQ1SubgroupHD128Pipeline` |

### GpuParam Layout
```cpp
struct GpuParam {
    ivec4 s0; // qLen, kLen, headNum, kvHeadNum
    ivec4 s1; // headDim, group, pastLen, totalLen
    ivec4 s2; // maskQlen, maskKvlen, hasMask, cacheMaxLen
    vec4 f0;  // scale, sparseVTau, lowerTriangularMask, turboQuantKBlockSize
};
```

### Execution Flow
1. `onEncode()` — records command buffer: KV update dispatch + barrier + prefill or decode dispatch
2. `onBeforeExecute()` — compacts KVCache (if remove/add), generates GPU mask (if causal), writes uniforms, binds descriptor sets — then GPU executes the pre-recorded commands

### Attention Configurations
| Mode | queryLen | Path | Pipeline Used |
|------|----------|------|---------------|
| Decode (q=1) | 1 | Decode | `mAttentionPipeline` or subgroup variants |
| Prefill (q>1) | >1 | Prefill k-block | `mInitStatePipeline` → `mQKBlock*` → `mSoftmaxOnlinePipeline` → `mQKVAcc*` → `mFinalizePipeline` |
| No KVCache | any | Legacy fused | `mAttentionLegacyPipeline` |

## VulkanLinearAttention Architecture

### Pipeline Flow (3 compute shaders)
1. `linear_attn_conv_silu.comp` — Conv1D depthwise + SiLU activation
2. `linear_attn_conv_state_update.comp` — Update persistent conv state
3. `linear_attn_gated_delta_rule.comp` — Split QKV, GQA broadcast, L2Norm, Gated Delta Rule recurrence

### Key Parameters
- `mNumKHeads`, `mNumVHeads`, `mHeadKDim`, `mHeadVDim`, `mUseQKL2Norm`
- `mAttentionType` = "gated_delta_rule" (the only supported type in Vulkan)
- Persistent state: `mConvState` [B, D, convStateSize], `mRecurrentState` [B, H, dk, dv]

## LLM Engine Integration (Plan 03)

### Backend Selection
`llm.cpp:48-63` (`backend_type_convert`): `"vulkan"` → `MNN_FORWARD_VULKAN` (line 59-60). Already present — no code changes needed.

### Build Command
```bash
cmake .. -DMNN_BUILD_LLM=ON -DMNN_VULKAN=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
make -j$(nproc) llm_demo
```

### Runtime Invocation
```bash
./build/llm_demo /path/to/model/config.json vulkan
```
Where `config.json` contains model path, tokenizer config, and optionally `"backend_type":"vulkan"`.

### Model Requirement
The LLM must be an MNN-format model (`.mnn` file) converted from a HuggingFace model. Recommended test model: Qwen2-0.5B or similar small quantized model. The model config JSON must reference the `.mnn` file path and tokenizer.

## Shader Autogeneration Pipeline

```
glsl/*.comp (GLSL source)
    ↓ makeshader.py
AllShader.cpp (SPIR-V byte arrays)
AllShader.h (extern declarations)
VulkanShaderMap.cpp (name → byte array mapping)
```

**All shader files are already generated** — `makeshader.py` has been run and `attention_mask_gen` entries exist in all three autogen files. No re-run needed unless new shaders are added (not in current scope).

## Dependencies Between Plans

```
Plan 01 (VULK-06, VULK-07): COMPLETE — no action needed
    ↓
Plan 02 (VULK-01 through VULK-05): Tests — depends on Plan 01 correctness fixes being in place
    ↓
Plan 03 (VULK-08): E2E LLM validation — depends on Plan 02 passing (proves the ops are correct)
```

## Gotchas and Risk Areas

1. **Vulkan runtime availability:** Tests must handle systems without Vulkan. Use `MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN)` check and return `true` (skip) if null.

2. **KVCache compaction:** The `onBeforeExecute` KVCache compaction logic (lines 1058-1263) is complex with reserve[] page table manipulation. Tests using multi-turn KVCache exercise this code path heavily.

3. **FP16 vs FP32:** VulkanAttention supports both FP16 and FP32 modes. The existing AttentionTest uses `MNNTestSuite::get()->pStaus.precision` which is set from the `run_test.out` command line. Vulkan-specific tests should similarly respect `pStaus.precision` or explicitly test both.

4. **Attention mask shape assumptions:** The prefill path at `_supportAttentionPrefill()` (lines 61-78) returns `false` when a causal mask tensor is provided (instead of null/scalar), forcing the legacy path. GPU mask gen only activates for `lowerTriangularMask != 0`. Tests must provide masks matching these assumptions.

5. **Shader compilation failures:** If `vkBn->getPipeline()` returns null for any shader, the constructor asserts and crashes. Tests should be designed to be robust against initialization failures.

6. **Numerical tolerance:** GPU float precision varies by vendor. The `rtol=0.01` threshold for attention and `rtol=0.02` for LinearAttention have been chosen to accommodate typical GPU rounding differences while catching correctness bugs. If tests fail on specific GPUs with borderline values, the tolerance may need adjustment.

7. **`MNN_SUPPORT_TRANSFORMER_FUSE` build flag:** Both the source code AND tests are gated behind this flag. Without it, the Vulkan attention path is completely compiled out. Build must include `-DMNN_SUPPORT_TRANSFORMER_FUSE=ON`.

## Standard Stack

- **Language:** C++17 (forced for `MNN_SUPPORT_TRANSFORMER_FUSE` path), GLSL (Vulkan compute shaders)
- **Build:** CMake with Ninja, `-DMNN_VULKAN=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_TEST=ON`
- **Test framework:** MNN custom framework (`MNNTestCase`, `MNNTestSuiteRegister`)
- **Shader pipeline:** `makeshader.py` (Python) — already executed for current shaders
- **RTTI/Exceptions:** Both disabled (`-fno-rtti -fno-exceptions`)

## Architecture Patterns

- **Two-phase execution:** `onEncode` (record commands) → `onBeforeExecute` (update uniforms, write descriptor sets, manage KVCache lifecycle). Commands are pre-recorded in onEncode and executed by the GPU after onBeforeExecute returns.
- **Pipeline + DescriptorSet pattern:** Each compute shader gets one `VulkanPipeline*` and one `shared_ptr<VulkanLayout::DescriptorSet>`. Descriptor sets are written in onBeforeExecute with current buffer handles.
- **Uniform buffer staging:** `vkBn->allocUniform()` creates host-visible uniform buffers. Data is written via `map()`/`unmap()` and bound to descriptor sets.
- **Validation via `MNN_ASSERT`:** Extensive assertions validate dimensions, buffer states, and pointer validity throughout. These are debug-only (compile out in release).
- **Backend creator registration:** Static initialization via lambda: `static bool gResistor = []() { VulkanBackend::addCreator(OpType_Attention, ...); return true; }();`

## Common Pitfalls

1. **Don't hand-roll Vulkan synchronization:** The `VulkanCommandPool::Buffer` provides `barrierSource()` for simple cases. For cross-buffer synchronization, use `VkBufferMemoryBarrier` with `vkCmdPipelineBarrier` as already implemented.

2. **Don't modify FlatBuffers schemas:** Model format changes require schema regeneration. Attention op uses `AttentionParam` and `LinearAttentionParam` from existing schemas.

3. **Don't modify makeshader.py:** The shader embedding pipeline is standard across all Vulkan shaders. Changes to it would affect all 60+ shaders.

4. **Don't use `using namespace std;`:** Although observed in test files, new code should not introduce this pattern.

5. **Don't allocate GPU memory in hot paths:** KVCache expansion and mask buffer creation use `Backend::DYNAMIC` with proper `onAcquireBuffer`/`onReleaseBuffer` lifecycle.

## Validation Architecture

For plan correctness: Vulkan tests pass → E2E llm_demo produces coherent output. No Nyquist validation strategy is needed (the existing plans are structurally complete).

## Package Legitimacy Audit

No new package installations required. All dependencies are vendored in the MNN source tree or provided by the system Vulkan SDK.
