# Roadmap: MNN — Vulkan Backend LLM Enablement

## Overview

Multi-phase journey from zero Vulkan LLM capability to a complete inference pipeline with Ultra FP4 quantization, TurboQuant documentation, FP4 model conversion, and model-level regression testing. Phase 1 establishes correctness — making the existing but untested Vulkan Attention implementation proven and integrated, culminating in an end-to-end `llm_demo` run producing coherent output. Phase 2 adds Ultra FP4, a new 4-bit floating-point quantization op, as a Vulkan shader operation per the GeniusCogntiveSystem architecture design. Phase 3 completes TurboQuant documentation (config contract, CPU fallback). Phase 4 builds the FP4 model conversion pipeline to produce quantized .mnn models from test or ONNX inputs. Phase 5 adds model-level regression tests for Vulkan + TurboQuant + sparse-V, which depends on having quantized models from Phase 4.

## Phases

- [x] **Phase 1: Vulkan Attention Correctness & LLM E2E** — Vulkan backend runs `llm_demo` producing coherent output; correctness over performance (completed 2026-05-27)
- [x] **Phase 2: Ultra FP4 Quantization** — FP4 quantized models run on Vulkan backend with acceptable precision (completed 2026-05-28)
- [x] **Phase 3: TurboQuant Documentation** — Document TurboQuant config contract and CPU fallback behavior (issues #8, #9) (completed 2026-05-28)
- [ ] **Phase 4: FP4 Model Conversion Pipeline** — Convert test .mnn or ONNX models into Ultra FP4 quantization format using the MNN converter
- [ ] **Phase 5: Model-Level Regression Tests** — Add LLM/ELM regression tests for Vulkan + TurboQuant + sparse-V (issue #7)

## Phase Details

### Phase 1: Vulkan Attention Correctness & LLM E2E
**Goal**: Vulkan backend can run `llm_demo` producing coherent text output matching CPU reference quality
**Depends on**: Nothing (first phase)
**Requirements**: VULK-01, VULK-02, VULK-03, VULK-04, VULK-05, VULK-06, VULK-07, VULK-08
**Success Criteria** (what must be TRUE):
  1. Running `llm_demo` with Vulkan backend produces coherent text output matching CPU reference quality across single-turn and multi-turn conversations
  2. Vulkan Attention and LinearAttention ops pass correctness tests for all supported configurations (GQA, MHA, MQA) matching CPU/Metal reference within float tolerance
  3. Vulkan test suite (Attention, LinearAttention, KVCache, mask generation) is integrated and passes via `./run_test.out`
  4. Vulkan Attention synchronization uses proper buffer barriers (`vkCmdPipelineBarrier` with `VkBufferMemoryBarrier`), eliminating the global memory barrier fragility
   5. Causal mask generation executes GPU-side with no CPU O(N²) bottleneck for large context windows
**Plans**: 3 plans
Plans:
- [x] 01-01-PLAN.md — Attention Synchronization & GPU Mask (VULK-06, VULK-07) ✅ Verified in source (2026-05-27)
- [x] 01-02-PLAN.md — Attention & LinearAttention Test Suite (VULK-01 through VULK-05)
- [x] 01-03-PLAN.md — LLM E2E Validation (VULK-08)

### Phase 2: Ultra FP4 Quantization
**Goal**: FP4 quantized models run on Vulkan backend with acceptable precision
**Depends on**: Phase 1
**Requirements**: FP4-01, FP4-02, FP4-03, FP4-04, FP4-05, FP4-06
**Success Criteria** (what must be TRUE):
  1. FP4 dequantization GLSL shader is built, embedded via `makeshader.py` pipeline, and registered in the Vulkan backend execution table
  2. FP4 quantized model can be loaded from model files and run through Vulkan backend without errors
  3. FP4 dequantization output matches FP16/FP32 reference within acceptable precision tolerance
  4. FP4-enabled model inference on Vulkan backend produces correct end-to-end results, matching INT4-quality or better
**Plans**: 2 plans
Plans:
- [x] 02-01-PLAN.md — FP4 Dequant GLSL Shader + Vulkan Pipeline Integration (FP4-01, FP4-02, FP4-03, FP4-04)
- [x] 02-02-PLAN.md — FP4 Correctness Test + Precision Verification (FP4-05, FP4-06)

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Vulkan Attention Correctness & LLM E2E | 3/3 | Complete    | 2026-05-27 |
| 2. Ultra FP4 Quantization | 2/2 | Complete    | 2026-05-28 |
| 3. TurboQuant Documentation | 1/1 | Complete    | 2026-05-28 |
| 4. FP4 Model Conversion Pipeline | 0/2 | Planned | — |
| 5. Model-Level Regression Tests | 0/0 | Not Started | — |

### Phase 3: TurboQuant Documentation (issues #8, #9)

**Goal:** Document TurboQuant config contract and CPU fallback behavior to close remaining documentation gaps
**Requirements**: DOC-01, DOC-02
**Depends on:** Phase 2
**Plans:** 1 plan

Plans:
- [x] 03-01-PLAN.md — TURBOQUANT.md config contract + CPU fallback docs

### Phase 4: Convert test models (.mnn or ONNX) into Ultra FP4 quantization formats using the MNN converter

**Goal:** A Python tool converts float .mnn model weights to E2M1 FP4 packed format, producing valid .mnn models that the Vulkan backend loads and executes via the existing VulkanFP4Dequant (registered under OpType_Dequantize). A CPU-side FP4 dequant execution class enables validation.
**Requirements**: PH4-CONV-01, PH4-CONV-02, PH4-TEST-01, PH4-TEST-02
**Depends on:** Phase 2
**Plans:** 2 plans

Plans:
- [ ] 04-01-PLAN.md — Python FP4 quantization tool (quantize_fp4.py) + CPU FP4 dequant runtime (CPUFP4Dequant)
- [ ] 04-02-PLAN.md — End-to-end FP4 model conversion test (FP4ModelTest) validating CPU and Vulkan backends

### Phase 5: Add model-level regression tests for Vulkan TurboQuant and sparse-V

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 4
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd-plan-phase 5 to break down)
