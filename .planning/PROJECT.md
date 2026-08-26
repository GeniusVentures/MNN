# MNN — Vulkan Backend LLM Enablement

## What This Is

MNN is a lightweight deep learning inference engine targeting mobile and server platforms, supporting CNN, Transformer, LLM, and Diffusion models. This project focuses on completing the Vulkan backend to run LLMs end-to-end and adding Ultra FP4 quantization as a new Vulkan shader operation.

MNN lives as a submodule under `GeniusNetwork/thirdparty/` — upstream is external; this project adds Vulkan-specific capabilities that may require upstream contribution. Ultra FP4 design originates from the sibling `GeniusCogntiveSystem/docs/architecture/` workspace.

## Core Value

**A complete Vulkan LLM inference pipeline** — the existing Vulkan Attention implementation (1505 lines) must graduate from compile-gated experimental code to a tested, production-ready op that can run LLMs on Vulkan-capable devices, followed by Ultra FP4 quantization for 4-bit float inference.

## Current Milestone: sgfp4-pivot v2.0 — SGFP4 v2 Converter Integration

**Workstream:** sgfp4-pivot (`.planning/workstreams/sgfp4-pivot/`)

**Goal:** A pytorch → onnx → mnnconvert pipeline can produce a `.mnn` file that runs SGFP4 v2 (quadtree-adaptive) FP4 inference, via a native mnnconvert CLI flag.

**Target features:**
- Real-weight SGFP4 v2 encoder — adapts the existing synthetic-fixture Python encoder (built for Phase 2 round-trip tests) to quantize actual `Convolution2D`/`ConvolutionDepthwise`/`Deconvolution` weight tensors from a converted model
- `tools/converter/source/common/RemoveParams.cpp` wired to emit the `SGFP4DequantParam` external sidecar (`{magic, offset, size}`), following the `Convolution2D.external` precedent already in that file
- New native mnnconvert CLI flag (peer to `--weightQuantBits`) that triggers SGFP4 v2 quantization + op-type rewrite during conversion
- End-to-end validation: a real test model runs correct inference after pytorch → onnx → mnnconvert(SGFP4 flag) → `.mnn`, on CPU and/or Vulkan

Context: sgfp4-pivot v1.0 (SGV2-01..16) shipped the decode-only path — CPU + Vulkan Executions for both uniform and LAYOUT_MIXED quadtree containers, validated only against synthetic test fixtures. This milestone closes the gap to a usable end-user pipeline.

## Requirements

### Validated

- ✓ Graph optimization + heterogeneous backend scheduling architecture — existing
- ✓ 13 hardware backends (CPU, Metal, CUDA, OpenCL, Vulkan, ARM, TensorRT, QNN, NN, OpenVINO, CoreML, TRT, HIAI) — existing
- ✓ Session API (Interpreter → createSession → runSession) — existing
- ✓ Module/Express API (Module::load → onForward) — existing
- ✓ FlatBuffers schema-based op definitions — existing
- ✓ Shape inference pipeline (source/shape/) — existing
- ✓ Geometry decomposition pipeline (source/geometry/) — existing
- ✓ NC4HW4 Tensor memory layout — existing
- ✓ LLM export pipeline (transformers/llm/export/) — existing
- ✓ LLM text inference engine (transformers/llm/engine/) — existing
- ✓ KVCache management and sampling strategies — existing
- ✓ Vulkan buffer backend with 100+ op implementations — existing
- ✓ Vulkan image backend (partial) — existing
- ✓ SGFP4 v2 (quadtree-adaptive, affine dual-mode) decode — CPU + Vulkan Executions, both code modes (FP4_AFFINE, T158_AFFINE), all uniform layouts + LAYOUT_MIXED, CPU/Vulkan parity verified across 14 fixtures — sgfp4-pivot v1.0 (additive to existing E2M1 Ultra FP4 path)

### Active

- [ ] **VULK-01**: Vulkan Attention op passes correctness tests (matching CPU/Metal reference outputs)
- [ ] **VULK-02**: Vulkan LinearAttention op is functional and tested
- [ ] **VULK-03**: Vulkan image backend has Attention, LinearAttention, LayerNorm, MatMul ops (currently missing 10 critical ops)
- [ ] **VULK-04**: Vulkan Attention KVCache management is correct across multi-turn inference
- [ ] **VULK-05**: Causal mask generation is GPU-side (remove O(N²) CPU bottleneck)
- [ ] **VULK-06**: Vulkan synchronization in Attention uses proper buffer barriers (not global memory barrier)
- [ ] **VULK-07**: Vulkan backend can run an end-to-end LLM (llm_demo) on Vulkan devices
- [ ] **VULK-08**: Vulkan tests are added to test suite (currently zero Vulkan-specific tests)
- [ ] **FP4-01**: Ultra FP4 GLSL shader op is implemented per GeniusCogntiveSystem design docs
- [ ] **FP4-02**: FP4 dequantization path works in Vulkan buffer backend
- [ ] **FP4-03**: FP4 quantized weights can be loaded from model files
- [ ] **FP4-04**: Ultra FP4 op is registered in the Vulkan backend execution table
- [ ] **FP4-05**: FP4 shaders are embedded via makeshader.py pipeline
- [ ] **FP4-06**: FP4 op passes correctness tests against reference FP16/FP32 output

### Out of Scope

- Metal/CUDA backend changes — Vulkan-only focus
- Training or fine-tuning support — inference only
- Upstream MNN changes outside Vulkan backend
- Vulkan image backend beyond the 10 missing critical ops
- Non-Vulkan backends for FP4 (CPU FP4 deferred)
- Python export pipeline changes (HQQ is already export-time)
- Mobile-specific Vulkan optimizations (Android GPU workgroup tuning deferred)

## Context

- **Submodule:** MNN is `GeniusNetwork/thirdparty/MNN` — GSD planning files live inside this repo but commit scope is limited to the submodule
- **Ultra FP4 design source:** `../../GeniusCogntiveSystem/docs/architecture/` — not yet started, needs to be reviewed before implementation
- **Current state:** Vulkan Attention exists but is gated behind `MNN_SUPPORT_TRANSFORMER_FUSE` build flag, has zero tests, and uses manual synchronization that is known fragile
- **Downstream consumer:** GeniusCogntiveSystem leverages MNN inference — Vulkan LLM support unblocks mobile/desktop GPU inference for that project
- **Build:** `cmake .. -DMNN_BUILD_LLM=ON -DMNN_VULKAN=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON`
- **Shader regeneration:** Editing GLSL under `source/backend/vulkan/buffer/execution/glsl/` requires running `python3 source/backend/vulkan/buffer/compiler/makeshader.py` to regenerate embedded shader files
- **Test runner:** `cd build && ./run_test.out`

## Constraints

- **Tech stack:** C++11 (default) with C++17 optional; GLSL for Vulkan shaders; Python for export pipeline
- **Build:** CMake-based; RTTI and exceptions disabled (`-fno-rtti -fno-exceptions`)
- **Compatibility:** Must not break non-Vulkan backends; `MNN_SUPPORT_TRANSFORMER_FUSE` gating should eventually be removed
- **Code style:** Google Style variant — 4-space indent, 120-char lines, PascalCase classes, camelCase functions, mCamelCase members
- **Shader pipeline:** All GLSL edits must go through `makeshader.py` → regenerate `AllShader.cpp`, `AllShader.h`, `VulkanShaderMap.cpp`
- **Performance:** Vulkan ops must match or exceed CPU performance for LLM inference (target: sub-100ms per token on mid-range GPU)
- **Upstream:** Changes should be structured to be upstreamable to MNN mainline

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Focus on Vulkan buffer backend first, image backend second | Buffer backend has 100+ existing ops and the Attention impl; image backend is missing 10 critical ops | — Pending |
| Start with Vulkan Attention correctness (tests) before performance | Current impl has zero tests; correctness must precede optimization | — Pending |
| Ultra FP4 as new Vulkan shader op rather than modifying existing quantization | FP4 is a new format (not INT4/INT8); requires new GLSL kernel and op registration | — Pending |
| Codebase map committed before starting | Needed to understand existing architecture, conventions, and concerns | ✓ Good |
| DeepSeek V4 for planning agents | Local runtime preference | ✓ Good |
| Execute Phase 4 plan 04-02 (E2E FP4 model test) before starting any SGFP4 pivot work, folding in the MAX_E2M1_VALUE scale-calibration bugfix | 04-02 validates already-built E2M1 infrastructure (Phase 2 + Phase 4-01) cheaply; SGFP4 work is additive (new format/op handling), not a modification of the existing symmetricQuan nbits=4 path, so finishing 04-02 first does not block it | — Pending |
| sgfp4-pivot v1.0: SGFP4 v2 targets v2-only, external-sidecar container (`.mnn.weight`-style, `{magic, offset, size}` descriptor), attestation/byte-exactness out of scope | MNN's job is inference, not verifiable-execution; SuperGenius verifies results separately | ✓ Good |
| sgfp4-pivot v1.0: proceeded to archive with Phase 2's formal `02-VERIFICATION.md` deferred (checkbox flip only happens as part of verify, which never ran for Phase 2) | Work is evidenced by 02-01/02-02-SUMMARY.md and built upon by Phases 3-4 without issue; re-running verify retroactively adds no new information | ⚠️ Revisit — run `/gsd-verify-work 2` retroactively if a formal report is ever needed |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-08-25 after archiving sgfp4-pivot v1.0 and starting sgfp4-pivot milestone v2.0*
