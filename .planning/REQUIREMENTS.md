# Requirements: MNN Vulkan Backend LLM Enablement

**Defined:** 2026-05-27
**Core Value:** A complete Vulkan LLM inference pipeline with Ultra FP4 quantization

## v1 Requirements

### Vulkan Attention — Correctness & Testing

- [x] **VULK-01**: Vulkan Attention op produces output matching CPU reference within float tolerance for all supported attention configurations (GQA, MHA, MQA)
- [x] **VULK-02**: Vulkan LinearAttention op is functional and produces correct output
- [x] **VULK-03**: Vulkan Attention KVCache page table logic is correct across single-turn and multi-turn inference
- [x] **VULK-04**: Vulkan Attention handles variable sequence lengths and batch sizes correctly
- [x] **VULK-05**: Vulkan-specific test cases are added to the test suite covering Attention and LinearAttention ops

### Vulkan Attention — Robustness & LLM Integration

- [x] **VULK-06**: Vulkan Attention uses proper buffer barriers (vkCmdPipelineBarrier with buffer memory barrier) instead of global memory barrier for KVCache synchronization
- [x] **VULK-07**: Causal mask generation is moved to GPU-side compute (remove O(N²) CPU bottleneck in `setArg`/mask upload path)
- [x] **VULK-08**: Vulkan backend successfully runs an end-to-end LLM via `llm_demo` producing coherent text output

### Ultra FP4 Quantization

- [ ] **FP4-01**: FP4 dequantization GLSL shader is implemented per GeniusCogntiveSystem architecture design
- [ ] **FP4-02**: FP4 quantized weight tensor layout is defined and compatible with model loading pipeline
- [ ] **FP4-03**: FP4 op is registered in the Vulkan backend (buffer) execution table with proper shape inference
- [ ] **FP4-04**: FP4 shaders are embedded via `makeshader.py` pipeline and AllShader files are regenerated
- [ ] **FP4-05**: FP4 dequantization produces output matching FP16/FP32 reference within acceptable precision
- [ ] **FP4-06**: FP4-enabled model can be loaded and run through the Vulkan backend producing correct inference results

## v2 Requirements

- **VULK-09**: Vulkan image backend gets Attention, LinearAttention, LayerNorm, MatMul op implementations (10 missing critical ops)
- **VULK-10**: Vulkan Attention performance optimization — tune workgroup sizes, memory access patterns, subgroup utilization
- **FP4-07**: FP4 quantization for weights in CPU and Metal backends
- **FP4-08**: Runtime FP4 weight compression (currently HQQ is export-time only)
- **FP4-09**: BF16 accumulation option for FP4 dequant (improved precision)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Metal/CUDA backend changes | Vulkan-only focus |
| Training or fine-tuning | Inference engine only |
| Non-Vulkan FP4 (CPU, Metal) | Vulkan-first, others v2 |
| Vulkan image backend beyond 10 critical ops | Scope creep; buffer backend is priority |
| Mobile-specific Vulkan optimizations | Deferred until correctness is established |
| Python export pipeline changes | HQQ export already works; FP4 runtime is the gap |
| Upstream MNN changes outside Vulkan backend | Submodule constraints |

## Traceability

| Requirement | Category | Phase | Status |
|-------------|----------|-------|--------|
| VULK-01 | Attention correctness | Phase 1 | Pending |
| VULK-02 | LinearAttention | Phase 1 | Pending |
| VULK-03 | KVCache management | Phase 1 | Pending |
| VULK-04 | Variable lengths | Phase 1 | Pending |
| VULK-05 | Test coverage | Phase 1 | Pending |
| VULK-06 | Sync barriers | Phase 1 | Pending |
| VULK-07 | GPU mask | Phase 1 | Pending |
| VULK-08 | E2E LLM run | Phase 1 | Pending |
| FP4-01 | Shader impl | Phase 2 | Pending |
| FP4-02 | Weight layout | Phase 2 | Pending |
| FP4-03 | Op registration | Phase 2 | Pending |
| FP4-04 | Shader embedding | Phase 2 | Pending |
| FP4-05 | Precision | Phase 2 | Pending |
| FP4-06 | E2E FP4 inference | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14 (100%)
- Phase 1: 8 requirements (VULK-01 through VULK-08)
- Phase 2: 6 requirements (FP4-01 through FP4-06)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-27*
*Last updated: 2026-05-27 after roadmap creation (traceability populated)*
