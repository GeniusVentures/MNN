# Roadmap: MNN — Vulkan Backend LLM Enablement

## Overview

Two-phase journey from zero Vulkan LLM capability to a complete inference pipeline with Ultra FP4 quantization. Phase 1 establishes correctness — making the existing but untested Vulkan Attention implementation proven and integrated, culminating in an end-to-end `llm_demo` run producing coherent output. Phase 2 adds Ultra FP4, a new 4-bit floating-point quantization op, as a Vulkan shader operation per the GeniusCogntiveSystem architecture design.

## Phases

- [x] **Phase 1: Vulkan Attention Correctness & LLM E2E** — Vulkan backend runs `llm_demo` producing coherent output; correctness over performance (completed 2026-05-27)
- [ ] **Phase 2: Ultra FP4 Quantization** — FP4 quantized models run on Vulkan backend with acceptable precision

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
- [ ] 02-02-PLAN.md — FP4 Correctness Test + Precision Verification (FP4-05, FP4-06)

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Vulkan Attention Correctness & LLM E2E | 3/3 | Complete    | 2026-05-27 |
| 2. Ultra FP4 Quantization | 0/2 | Planned | - |
