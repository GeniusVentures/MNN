---
phase: 01-vulkan-attention-correctness-llm-e2e
plan: 03
type: execute
tasks: 3
completed: 3
files_modified:
  - transformers/llm/engine/src/llm.cpp
  - build/llm_demo
  - /tmp/qwen2-model/mnn_export/
---

# Plan 03 Summary: LLM E2E Validation

## What was built

**Task 1:** Built `llm_demo` with Vulkan backend linked statically against MoltenVK (`libMoltenVK.a`) from the thirdparty build system. Configured with `MNN_USE_SYSTEM_LIB=ON` to bypass dlopen wrapping. Linked against Apple Metal frameworks (CoreFoundation, CoreGraphics, Metal, QuartzCore, IOSurface, Foundation, AppKit).

**Task 2:** Exported Qwen2-0.5B-Instruct to MNN format via `llmexport.py --export mnn`. Converted ONNX model from Hugging Face (`onnx-community/Qwen2-0.5B-ONNX` → `llmexport.py`). Ran E2E inference with `./llm_demo config.json` using Vulkan backend. Produced coherent output: "Hello! How can I assist you today?"

**Task 3:** Documented results (this file).

## Key decisions

- **MoltenVK linking:** Used `MNN_USE_SYSTEM_LIB=ON` + `Vulkan_LIBRARY=libMoltenVK.a` to link MoltenVK statically, avoiding the dlopen/LIB_WRAPPER path. Apple framework linker flags required for MoltenVK's Metal dependencies.
- **Model export:** Qwen2-0.5B-Instruct selected per CONTEXT.md D-03. Exported via `llmexport.py` which handles LLM-specific model structure (embedding separation, tokenizer export, MNN weight quantization).

## E2E result

| Criterion | Result |
|-----------|--------|
| llm_demo runs with Vulkan backend | Pass |
| Produces coherent English text | Pass — "Hello! How can I assist you today?" |
| No INTERNAL_ERROR / crash | Pass |
| Multi-turn conversation | Partial (stdin pipe limitation in test, but single-turn verified) |

## Known issues

| Issue | Severity | Notes |
|-------|----------|-------|
| **"Vulkan don't support 128, Raster"** | Medium | Many Raster ops (tensor reshaping/transpose) fall back to CPU. Vulkan backend lacks GPU-accelerated Raster for some tensor shapes. |
| **"Vulkan don't support 600, While"** | Medium | While loop op (used in self-attention blocks) falls back to CPU. Vulkan backend needs While op GPU implementation. |
| **Generation speed** | Low (Phase 1) | Slow due to CPU fallbacks. Correctness over performance per Phase 1 goal. |

## Build configuration (for reproducibility)

```bash
cmake .. \
  -DMNN_BUILD_LLM=ON \
  -DMNN_VULKAN=ON \
  -DMNN_VULKAN_IMAGE=OFF \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DMNN_USE_SYSTEM_LIB=ON \
  -DVulkan_INCLUDE_DIR=<thirdparty>/build/OSX/Debug/moltenvk/build/include \
  -DVulkan_LIBRARY=<thirdparty>/build/OSX/Debug/moltenvk/build/lib/MoltenVK.xcframework/macos-arm64_x86_64/libMoltenVK.a \
  -DCMAKE_EXE_LINKER_FLAGS="-framework CoreFoundation -framework CoreGraphics -framework CoreServices -framework IOKit -framework IOSurface -framework Metal -framework QuartzCore -framework Foundation -framework AppKit" \
  -G Ninja
```

## Self-Check: PASSED

All 3 tasks completed. VULK-08 verified: Vulkan backend runs E2E LLM producing coherent output.
