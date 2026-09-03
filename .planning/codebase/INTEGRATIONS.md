# External Integrations

**Analysis Date:** 2026-05-27

## Hardware Backends

MNN's core design is a heterogeneous backend scheduler. Each backend provides per-op implementations for its target hardware.

### GPU Backends

| Backend | CMake Option | Default | Platforms | Source |
|---------|-------------|---------|-----------|--------|
| **Vulkan** | `MNN_VULKAN` | ON | Android, Linux, Windows, macOS | `source/backend/vulkan/` |
| **Metal** | `MNN_METAL` | OFF | Apple (iOS, macOS) | `source/backend/metal/` |
| **OpenCL** | `MNN_OPENCL` | OFF | Android, Linux, macOS | `source/backend/opencl/` |
| **OpenGL** | `MNN_OPENGL` | OFF | Android (GLES), Linux | `source/backend/opengl/` |
| **CUDA** | `MNN_CUDA` | OFF | Linux (NVIDIA GPU) | `source/backend/cuda/` |

**Vulkan** (`source/backend/vulkan/`):
- Two memory backends: buffer mode (`source/backend/vulkan/buffer/`) and image mode (`source/backend/vulkan/image/`)
- Select via `MNN_VULKAN_IMAGE` (default ON)
- Uses GLSL shaders compiled via `makeshader.py` into `AllShader.cpp`/`AllShader.h`
- Runtime Vulkan library loaded via dlopen (not linked directly) when `MNN_USE_SYSTEM_LIB=OFF`
- GPU tuning: `MNN_GPU_TUNING_WIDE` (default), `MNN_GPU_TUNING_HEAVY`, `MNN_GPU_TUNING_NONE`
- Profiling: `MNN_GPU_TIME_PROFILE=ON`

**Metal** (`source/backend/metal/`):
- Apple-only (iOS 8.0+, macOS)
- Metal Shading Language (`.metal`) shader files compiled to `mnn.metallib`
- Metal4 tensor instructions supported via `MNN_METAL_TENSOR` (default ON)
- Objective-C++ implementation (`*.mm` files)
- Links: `Foundation.framework`, `Metal.framework`, `CoreGraphics.framework`
- Specialized kernels: Winograd convolution, 1x1 convolution, SIMD group operations, KVCache for LLM

**CUDA** (`source/backend/cuda/`):
- NVIDIA GPU support (Linux primarily, limited Windows)
- Subdirectories: `core/`, `execution/`
- CUDA profiling: `MNN_CUDA_PROFILE=OFF`
- Requires libcudart, CUDA toolkit

### NPU / AI Accelerator Backends

| Backend | CMake Option | Platform | Source |
|---------|-------------|----------|--------|
| **CoreML** | `MNN_COREML` | Apple (iOS, macOS) | `source/backend/coreml/` |
| **NNAPI** | `MNN_NNAPI` | Android 8.1+ | `source/backend/nnapi/` |
| **QNN** (Qualcomm) | `MNN_QNN` | Android (Snapdragon) | `source/backend/qnn/` |
| **TensorRT** | `MNN_TENSORRT` | Linux (NVIDIA) | `source/backend/tensorrt/` |
| **NeuroPilot** | `MNN_NEUROPILOT` | MediaTek | `source/backend/neuropilot/` |
| **HiAI** (Huawei) | `MNN_NPU` | Android / HarmonyOS | `source/backend/hiai/` |

**CoreML** (`source/backend/coreml/`):
- Links Apple frameworks: `CoreML.framework`, `Foundation.framework`, `Metal.framework`, `CoreVideo.framework`
- Uses CoreML model format internally via `mlmodel/`

**QNN** (`source/backend/qnn/`):
- Qualcomm Neural Network SDK integration
- Online finalize support via `MNN_QNN_ONLINE_FINALIZE` (default ON)
- Python conversion helper: `source/backend/qnn/npu_convert.py`
- Dependency script: `prepare_qnn_deps.sh`

**TensorRT** (`source/backend/tensorrt/`):
- NVIDIA TensorRT inference optimization
- Subdirectories: `backend/`, `execution/`

### CPU Backend (always built)

**CPU ISA Optimizations** (`source/backend/cpu/`):

| Architecture | Subdirectory | ISA Extensions |
|-------------|-------------|----------------|
| **ARM v7 (AArch32)** | `source/backend/cpu/arm/arm32/` | NEON, FP16 |
| **ARM v8 (AArch64)** | `source/backend/cpu/arm/arm64/` | NEON, ASIMD, dotprod, i8mm, BF16 |
| **ARM v8.2** | `source/backend/arm82/` | FP16 compute (half-precision Ops) |
| **ARM v8.6+** | `source/backend/cpu/arm/arm64/sme2_asm/` | SME2 (Scalable Matrix Extension 2) |
| **x86/x86_64 SSE** | `source/backend/cpu/x86_x64/sse/` | SSE4.1 |
| **x86_64 AVX** | `source/backend/cpu/x86_x64/avx/` | AVX2 |
| **x86_64 AVX+FMA** | `source/backend/cpu/x86_x64/avxfma/` | AVX2 + FMA |
| **x86_64 AVX-512** | `source/backend/cpu/x86_x64/avx512/` | AVX-512F/DQ/VL/BW + VNNI |
| **RISC-V** | `source/backend/cpu/riscv/` | RVV (RISC-V Vector Extension) |
| **BF16** | `source/backend/cpu/bf16/` | BFloat16 SIMD routines |

**KleidiAI Integration** (`cmake/KleidiAI.cmake`):
- Arm's KleidiAI library v1.14.0 — downloaded automatically at build time
- Provides optimized matmul kernels for dotprod, i8mm, and SME2 instruction sets
- Wrapper files: `source/backend/cpu/arm/mnn_kleidiai.cpp`, `mnn_kleidiai_util.cpp`
- Used in convolution implementations: `source/backend/cpu/compute/KleidiAI*.cpp`

**oneDNN Integration** (`cmake/oneDNN.cmake`):
- oneDNN v1.7 — Intel's Deep Neural Network Library, downloaded at build time
- Used for x86 convolution optimizations via `source/backend/cpu/OneDNN*.cpp`
- Only available on x86_64 Linux

**CPU Compute Library** (`source/backend/cpu/compute/`):
- Optimized convolution implementations: Winograd, tiled, Strassen, depthwise, INT8, sparse
- Image processing: `ImageProcessFunction.cpp`

## Model Serialization: FlatBuffers

**Schema framework:**
- Op and model format defined in FlatBuffers schema files: `schema/default/*.fbs`
- Generated C++ headers: `schema/current/*_generated.h`
- Schema files:
  - `MNN.fbs` — Primary operator schema (Net, Op, Extra)
  - `Tensor.fbs` — Tensor descriptor
  - `Type.fbs` — Data types and parameter types
  - `CaffeOp.fbs` — Caffe-specific operator parameters
  - `TensorflowOp.fbs` — TensorFlow-specific operator parameters
  - `TFQuantizeOp.fbs` — TFLite quantization parameters
  - `TrainInfo.fbs` — Training-related metadata
  - `ExtraInfo.fbs` — Extended model information
  - `UserDefine.fbs` — User-defined extensions

**FlatBuffers library:** Vendored at `3rd_party/flatbuffers/` (full library, includes gRPC, reflection support).

## Model Converter

**Location:** `tools/converter/`

**Input formats supported:**
- **ONNX** → MNN (primary path for PyTorch, TensorFlow 2.x, etc.)
- **TensorFlow** (frozen graph .pb) → MNN
- **Caffe** → MNN
- **TFLite** → MNN
- **TorchScript** → MNN (via libtorch dependency)

**Python export pipeline for LLMs:** `transformers/llm/export/`
- Entry point: `llmexport.py`
- Uses HuggingFace `transformers` library to load models
- Converts to MNN format via ONNX intermediate representation
- Supports quantization: `--hqq` flag

**Dependencies:** Protocol Buffers (`3rd_party/protobuf/`) for ONNX/TF/Caffe parsing.

**CMake:** `MNN_BUILD_CONVERTER=ON` (default OFF)

## Python Bindings (PyMNN)

**Location:** `pymnn/`

**Package name:** `mnn` (or `mnn_internal`, `mnn_cuda`, `mnn_trt`, `mnn_vulkan`, `mnn_opencl`, `mnn_render`)

**Architecture:**
- C++ extension modules: `_mnncengine` (core API), `_tools` (converter, quantizer)
- Source: `pymnn/src/MNN.cc`, `pymnn/src/MNNTools.cc`
- Built via `setuptools` with CMake integration
- Entry points: `mnnconvert`, `mnnquant`, `mnn` CLI tools
- Requires NumPy for array interchange

**Setup:** `pymnn/pip_package/setup.py` (488 lines)

## LLM Inference Engine

**Location:** `transformers/llm/`

**Python Export** (`transformers/llm/export/`):
- `llmexport.py` — Main export entry point
- `utils/model_mapper.py` — Model field mapping
- `utils/model.py` — Unified LlmModel class
- `utils/transformers.py` — Attention, Decoder, RoPE export logic
- Requires: torch, transformers, onnx, onnxslim, onnxruntime, peft, sentencepiece, Pillow

**C++ Engine** (`transformers/llm/engine/`):
- `llm.cpp` — Text-only inference
- `omni.cpp` — Multimodal inference (vision + audio)
- Features: KVCache management, sampling strategies
- Headers: `transformers/llm/engine/include/llm/llm.hpp`, `llm/reranker.hpp`
- CMake: `MNN_BUILD_LLM=ON` (default), `MNN_BUILD_LLM_OMNI=ON` (default)

**LLM Demos:**
- `./llm_demo` — Text inference
- `./llm_bench` — Benchmarking

## Diffusion Support

**Location:** `transformers/diffusion/`

- C++ engine: `transformers/diffusion/engine/`
- Headers: `diffusion/diffusion.hpp`, `diffusion/sana_llm.hpp`
- Requires: `MNN_BUILD_DIFFUSION=ON`, `MNN_BUILD_OPENCV=ON`, `MNN_IMGCODECS=ON`

## Image Processing API

**Location:** `tools/cv/`

- Custom OpenCV-like C++ API (no external OpenCV dependency)
- Includes: `tools/cv/include/cv/` (headers), `tools/cv/source/` (implementation)
- Submodules: `imgproc`, `calib3d`
- CMake: `MNN_BUILD_OPENCV=ON`
- Required by LLM omni (image input), Diffusion

## Audio Processing API

**Location:** `tools/audio/`

- Audio processing C++ API
- Headers: `tools/audio/include/audio/*.hpp`
- CMake: `MNN_BUILD_AUDIO=ON`
- Required by LLM omni (audio input)

## Code Generation

**Location:** `codegen/`

- On-device code generation for backend kernels
- Targets: CPU (`codegen/cpu/`), CUDA (`codegen/cuda/`), Metal (`codegen/metal/`), OpenCL (`codegen/opencl/`)
- Op fusion pass: `codegen/OpFuse.cpp`
- Source module generation: `codegen/SourceModule.cpp`
- CMake: `MNN_BUILD_CODEGEN=OFF` (default)

## Internal / Authentication

**Conditional feature** (requires `MNN_INTERNAL=ON` and proprietary `schema/private/`):

- Model authentication and metrics logging
- Internal logging subsystem: `source/internal/logging/`
- On Linux: links `libcurl`, `libssl`, `libcrypto` for HTTPS logging
- On iOS: uses Aliyun Log C SDK (included in build per `MNN.podspec`)

## Threading Model

**Two threading implementations:**

1. **MNN Thread Pool** (default, `MNN_USE_THREAD_POOL=ON`):
   - Implementation: `source/backend/cpu/ThreadPool.cpp`
   - Platform-agnostic, does not require OpenMP
   - Compatible with iOS (where OpenMP is unavailable)

2. **OpenMP** (`MNN_OPENMP=ON`):
   - Alternative thread pool using system OpenMP
   - Not available on Apple platforms
   - When `MNN_USE_THREAD_POOL=ON`, `MNN_OPENMP` is forced OFF

**Mutual exclusion:** These two options cannot both be enabled.

## Platform-Specific Integrations

### Android
- **Logging:** Android logcat via `<android/log.h>` (when `MNN_USE_LOGCAT=ON`)
- **NDK build:** `project/android/build_64.sh`
- **JNI interface:** `MNN_JNI=ON` builds `source/jni/` for Java usage
- **Shared memory:** `MNN_MEMORY_AHARDWAREBUFFER` for zero-copy tensor sharing

### iOS / macOS
- **Framework:** `MNN_AAPL_FMWK=ON` produces `MNN.framework`
- **Metal:** GPU acceleration via Metal Shading Language
- **CoreML:** NPU delegation to Apple Neural Engine
- **CocoaPods:** `MNN.podspec`, `MNN_Render.podspec`

### Windows
- **MSVC** — Primary Windows compiler with `/MT` or `/MD` runtime
- **Clang-cl** — Alternative via `lld-link`

### HarmonyOS (OpenHarmony)
- **Logging:** HiLog via `<hilog/log.h>`
- **NPU:** Huawei HiAI backend

## Third-Party Code Licenses

Vendored third-party license notices: `3rd_party/lic/`
- `flatbuffer_license` — FlatBuffers (Apache 2.0)
- `skia_license` — Skia (BSD-style)
- `tensorflow_license` — TensorFlow (Apache 2.0)

**Project License:** Apache License 2.0 (`LICENSE.txt`, `MNN.podspec`)

## Network / External APIs

MNN is an **offline inference engine** — it does not connect to external network services at runtime.

**Exceptions (conditional, require `MNN_INTERNAL=ON`):**
- Internal logging to Aliyun Log Service (SLS) via HTTPS (`libcurl + libssl + libcrypto`)

## Environment Variables

**Build time:**
- `MNN_ASSEMBLER` — Path to GNU assembler for Windows ASM compilation
- `PROJECT_ROOT` — Override project root for PyMNN builds
- `NATIVE_LIBRARY_OUTPUT` / `NATIVE_INCLUDE_OUTPUT` — Android NDK output paths

**Runtime:**
- No mandatory environment variables for inference
- Backend selection is done programmatically (not via environment)

---

*Integration audit: 2026-05-27*
