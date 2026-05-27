# Phase 1: Vulkan Attention Correctness & LLM E2E - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers a proven-correct Vulkan attention pipeline — buffer barriers fixed (VkBufferMemoryBarrier), GPU mask generation active (attention_mask_gen.comp), correctness tests passing for Attention and LinearAttention ops, and `llm_demo` producing coherent output matching CPU reference quality. Covers requirements VULK-01 through VULK-08.
</domain>

<decisions>
## Implementation Decisions

### Test Registration & Naming
- **D-01:** Vulkan attention tests register under `op/vulkan/` prefix. Test names: `op/vulkan/attention_correctness` and `op/vulkan/linear_attention_correctness`.

### Vulkan Runtime Unavailability
- **D-02:** When Vulkan runtime is not available (headless CI, systems without Vulkan drivers), tests log a warning via `MNN_ERROR` or `MNN_PRINT`, then return `true` (pass/skip). Do not fail silently.

### LLM Model for E2E Validation
- **D-03:** Qwen2-0.5B is the preferred model for Plan 03 E2E validation. A `.mnn` format export of this model is not currently available — this is a known blocker for Plan 03 execution. The model must be exported via MNN converter before Plan 03 Task 2 can proceed.

### Test Data Determinism
- **D-04:** Default to a fixed random seed (`TEST_RANDOM_SEED`) for reproducibility in unit test suites. Variable (unseeded) data is acceptable as a secondary coverage mode but not the default. This aligns with existing MNN test conventions.

### Pre-existing (from PROJECT.md)
- **D-05:** Focus on Vulkan buffer backend (not image backend) for Phase 1.
- **D-06:** Correctness testing takes priority over performance optimization. Perf tuning deferred.
- **D-07:** `MNN_SUPPORT_TRANSFORMER_FUSE` build flag gates all Vulkan attention code — tests and impl must be guarded by it.

### the agent's Discretion
None — all gray areas were decided by the user.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Planning Artifacts
- `.planning/ROADMAP.md` — Phase 1 scope, success criteria, requirement mapping, plan index
- `.planning/REQUIREMENTS.md` — Full VULK-01 through VULK-08 requirements and traceability
- `.planning/PROJECT.md` — Project constraints, key decisions, tech stack, build flags
- `.planning/STATE.md` — Current position, blockers/concerns, session continuity

### Existing Plans
- `.planning/phases/01-vulkan-attention-correctness-llm-e2e/01-01-PLAN.md` — Plan 01: Barrier fix + GPU mask (implemented)
- `.planning/phases/01-vulkan-attention-correctness-llm-e2e/01-02-PLAN.md` — Plan 02: Test suite (pending)
- `.planning/phases/01-vulkan-attention-correctness-llm-e2e/01-03-PLAN.md` — Plan 03: E2E validation (pending)

### Research
- `.planning/phases/01-vulkan-attention-correctness-llm-e2e/01-RESEARCH.md` — Vulkan backend patterns, shader pipeline, test infrastructure findings; confirms Plan 01 implemented in source

### Vulkan Backend Source
- `source/backend/vulkan/buffer/execution/VulkanAttention.cpp` — The 1505-line attention implementation; onEncode barrier + onBeforeExecute mask generation
- `source/backend/vulkan/buffer/execution/VulkanAttention.hpp` — GpuParam struct, KVCache struct, pipeline/descriptor set member declarations
- `source/backend/vulkan/buffer/execution/VulkanLinearAttention.cpp` — 243-line LinearAttention implementation
- `source/backend/vulkan/buffer/execution/glsl/attention_mask_gen.comp` — GPU compute shader for causal mask generation
- `source/backend/vulkan/buffer/compiler/makeshader.py` — Shader autogeneration pipeline; must be re-run after .comp edits

### Test Infrastructure
- `test/MNNTestSuite.h` — Test framework: MNNTestCase, MNNTestSuiteRegister macro, MNNTEST_ASSERT
- `test/TestUtils.h` — Comparison helpers: checkVectorByRelativeError, FP32Converter
- `.planning/codebase/TESTING.md` — Test patterns, registration conventions, run_test.out invocation, CI configuration
- `test/op/AttentionTest.cpp` — Existing CPU Attention test (1041 lines); reference for _makeAttentionModule pattern, KVMeta struct usage, test class structure
- `test/op/LinearAttentionTest.cpp` — Existing CPU LinearAttention test (472 lines); NaiveLinearAttention reference implementation

### LLM Engine
- `transformer/llm/engine/src/llm.cpp` — backend_type_convert('vulkan'), initRuntime ScheduleConfig
- `transformer/llm/engine/src/llmconfig.hpp` — backend_type() method reading JSON config
- `transformer/llm/engine/demo/llm_demo.cpp` — CLI entry point, benchmark function, response API
- `transformer/llm/engine/include/llm/llm.hpp` — Llm class API: createLLM, response, generate, set_config
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`_makeAttentionModule` pattern** (`test/op/AttentionTest.cpp:65-87`): Creates OpT with OpType_Attention and generates KVCache-configured Module. Plans 02 will clone this pattern with Vulkan backend forcing.
- **`NaiveLinearAttention` struct** (`test/op/LinearAttentionTest.cpp:28-178`): Full CPU reference implementation (Conv1D + SiLU + QKV split + GQA + L2Norm + GatedDeltaRule). Copied directly into Vulkan test for reference comparison.
- **`checkVectorByRelativeError`** (`test/TestUtils.h:58`): Standard float tolerance comparison. Plans 02 use rtol=0.01 (Attention) and rtol=0.02 (LinearAttention).
- **`MNNGetExtraRuntimeCreator`** (`test/core/BackendTest.cpp:272-294`): Pattern for checking backend availability at runtime; used for Vulkan skip logic.
- **`backend_type_convert`** (`transformer/llm/engine/src/llm.cpp:48-64`): Already maps 'vulkan' string to MNN_FORWARD_VULKAN — no LLM engine code changes needed for Plan 03.

### Established Patterns
- **Test registration:** `MNNTestSuiteRegister(ClassName, 'namespace/test/name')` — used by Plans 02
- **Test CMake glob:** All `.cpp` under `test/` auto-collected into `run_test.out` — no CMake edits needed
- **Reference comparison:** Tests compute CPU reference inline, run Vulkan, compare with relative error
- **Shader autogeneration:** GLSL `.comp` edits require running `makeshader.py` to regenerate `AllShader.cpp`/`AllShader.h`/`VulkanShaderMap.cpp`
- **Build flag gating:** `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE` guards all attention code

### Integration Points
- **VulkanAttention.cpp constructor** (~line 299): Pipeline creation point — tests must match the binding indices set here
- **VulkanAttention.cpp onBeforeExecute** (~line 1208): Mask generation dispatch — replaced by GPU shader in Plan 01
- **llm_demo** → **VulkanAttention** via `backend_type=vulkan` → RuntimeManager → ScheduleConfig.type = MNN_FORWARD_VULKAN
- **Shader pipeline** (`makeshader.py` → `AllShader.cpp`/`.h`/`VulkanShaderMap.cpp`): Already complete for all current shaders including `attention_mask_gen`
</code_context>

<specifics>
## Specific Ideas

- Qwen2-0.5B is the preferred model for E2E validation, but `.mnn` format not available — Plan 03 Task 2 is blocked until a model is exported via MNN converter.
- Vulkan unavailability should produce a visible warning (not silent skip) so CI operators know why tests aren't running.
- Fixed-seed determinism preferred for CI-grade reproducibility; variable data as secondary for catching dimension-specific bugs.
</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.
</deferred>

---

*Phase: 01-vulkan-attention-correctness-llm-e2e*
*Context gathered: 2026-05-27*
