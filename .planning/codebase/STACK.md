# Technology Stack

**Analysis Date:** 2026-05-27

## Languages

**Primary:**
- C++ (C++11 default, optionally C++17) — Core engine, all backends, Express API, LLM/diffusion engines, tools, converter (`source/`, `express/`, `transformers/`, `tools/`)
- C (C99 / gnu99) — Selected backend kernels and third-party integrations (`3rd_party/`, Vulkan buffer shaders)

**Secondary:**
- Python 2.7+ / 3.5+ — PyMNN bindings (`pymnn/`), LLM export pipeline (`transformers/llm/export/`), model compression (`tools/mnncompress/`), Sherpa-MNN wrapper (`apps/frameworks/sherpa-mnn/`)
- Objective-C++ — Metal backend on Apple platforms (`source/backend/metal/*.mm`), CoreML backend
- Assembly — ARM NEON (aarch32/aarch64), ARM SME2, x86 SSE/AVX/AVX2/AVX512/VNNI kernels

## Version

**MNN Version:** 3.4.1 (defined in `include/MNN/MNNDefine.h` lines 75-78)

## Runtime

**Environment:**
- Native C/C++ runtime — No managed runtime; the library is a native shared/static library
- RTTI and exceptions **disabled** (`-fno-rtti -fno-exceptions` in CMakeLists.txt lines 613-614)
- Default symbol visibility: hidden in release builds (`-fvisibility=hidden`)

**Package Manager:**
- None (CMake-based, no conan/vcpkg required)
- Vendored third-party libraries under `3rd_party/`

## Build System

**Core:**
- **CMake** 3.6+ — Main build system (`CMakeLists.txt`, 1104 lines)
- Build modes:
  - `MNN_BUILD_SHARED_LIBS` (default ON) — Shared library (`libMNN.so`, `MNN.dll`, `libMNN.dylib`)
  - `MNN_BUILD_SHARED_LIBS=OFF` — Static library
  - `MNN_AAPL_FMWK` — Apple framework bundle (`MNN.framework`)
  - `MNN_SEP_BUILD` (default ON) — Build backends as separate shared libs, loaded via dlopen
- Compiler requirements: GCC, Clang, AppleClang, or MSVC

**Build Script:**
- `build_lib.sh` — Unified build script for Android, iOS, HarmonyOS, and Python builds (807 lines)

**Key CMake Options (50+):**
| Option | Default | Purpose |
|--------|---------|---------|
| `MNN_BUILD_SHARED_LIBS` | ON | Shared vs static library |
| `MNN_BUILD_LLM` | ON | LLM inference engine |
| `MNN_BUILD_LLM_OMNI` | ON | Multimodal (image/audio) in LLM |
| `MNN_BUILD_DIFFUSION` | OFF | Diffusion model demo |
| `MNN_BUILD_TRAIN` | OFF | Training framework |
| `MNN_BUILD_CONVERTER` | OFF | Model converter tools |
| `MNN_BUILD_TEST` | ON | Unit tests |
| `MNN_BUILD_PROTOBUFFER` | ON | Bundle protobuf for converter |
| `MNN_BUILD_OPENCV` | OFF | OpenCV-like image processing API |
| `MNN_BUILD_AUDIO` | OFF | Audio processing API |
| `MNN_USE_THREAD_POOL` | ON | MNN's own thread pool (vs OpenMP) |
| `MNN_OPENMP` | OFF | OpenMP threading (not on Apple) |
| `MNN_SUPPORT_BF16` | OFF | BFloat16 compute |
| `MNN_LOW_MEMORY` | OFF | Low memory mode (forced ON for LLM) |

**CPU ISA Targeting:**
| Option | Default | Architecture |
|--------|---------|--------------|
| `MNN_USE_SSE` | ON | x86 SSE4.1 |
| `MNN_AVX2` | ON | x86 AVX2/FMA |
| `MNN_AVX512` | OFF | x86 AVX-512 (with VNNI optional) |
| `MNN_ARM82` | ON | ARMv8.2 (FP16, dotprod, i8mm) |
| `MNN_SME2` | ON | ARM Scalable Matrix Extension 2 |
| `MNN_KLEIDIAI` | ON | Arm KleidiAI library integration |
| `MNN_USE_RVV` | OFF | RISC-V Vector Extension |

## Frameworks

**Core Inference:**
- Session API (low-level): `Interpreter → createSession → runSession` — operates on raw Tensors
- Module API (high-level, recommended): `Module::load → onForward(VARP)` — Express-based dynamic graph

**Testing:**
- Custom test runner (`run_test.out`) — built from `test/` directory
- Test framework: `MNN_BUILD_TEST=ON` builds tests via `test/CMakeLists.txt`

**Build/Dev:**
- **clang-format** — Code formatting (`.clang-format`: Google-based, 4-space indent, 120 col, C++11 standard)
- **pre-commit** hooks — Incremental clang-format check, commit message format, large file check (`.pre-commit-config.yaml`)

## Key Dependencies

### Vendored (in `3rd_party/`)

| Library | Purpose | Key Files |
|---------|---------|-----------|
| **FlatBuffers** | Model file serialization format; op schema definitions | `3rd_party/flatbuffers/`, `schema/default/*.fbs`, `schema/current/*_generated.h` |
| **Protocol Buffers** | Converter model import (ONNX, TF, Caffe, TFLite) | `3rd_party/protobuf/` |
| **half** | IEEE 754 half-precision float (float16) in C++ | `3rd_party/half/` |
| **imageHelper** | Image processing utilities | `3rd_party/imageHelper/` |
| **OpenCLHeaders** | OpenCL API headers for GPU backend | `3rd_party/OpenCLHeaders/` |
| **rapidjson** | JSON parsing (internal logging, model config) | `3rd_party/rapidjson/` |

### Downloaded at Build Time

| Library | Version | Source | Purpose |
|---------|---------|--------|---------|
| **KleidiAI** | v1.14.0 | GitHub: ARM-software/kleidiai | ARM optimized matmul, depthwise conv kernels (NEON dotprod/i8mm + SME2) |
| **oneDNN** | v1.7 | GitHub: oneapi-src/oneDNN | Intel x86 optimized DNN primitives |

### System Libraries by Platform

| Platform | System Libraries |
|----------|-----------------|
| **Linux** | `pthread`, `dl`, `stdc++` |
| **Android** | `log`, `m`, `android` |
| **Apple (Metal)** | `Foundation`, `Metal`, `CoreGraphics` |
| **Apple (CoreML)** | `CoreML`, `Foundation`, `Metal`, `CoreVideo` |
| **OpenGL** | `GLESv3`, `EGL` |
| **Windows** | `msvcrt` |
| **OHOS** | `libhilog_ndk.z.so` |

## SDLC Chain

**CI/CD:**
- **GitHub Actions** — 16 workflow files:
  - `linux.yml` — Linux x86 build + test (SSE/non-SSE/AVX512 variants)
  - `macos.yml` — macOS build + test
  - `windows.yml` — Windows MSVC build
  - `android.yml` — Android NDK cross-compile (ARM32/ARM64)
  - `ios.yml` — iOS framework build
  - `pymnn_linux.yml`, `pymnn_macos.yml`, `pymnn_windows.yml`, `pymnn_release.yml` — PyMNN wheels
  - `code-format.yml` — Code formatting check
  - `llm-pr-review.yml` — LLM-specific PR checks
  - `fastlane.yml` — Fastlane-based deployment
  - `stale.yml` — Stale issue management
  - `wiki.yml` — Documentation publish

**iOS/macOS Distribution:**
- CocoaPods: `MNN.podspec` (v2.2.0 listed), `MNN_Render.podspec`
- Framework builds via Xcode toolchain (`cmake/ios.toolchain.cmake`)

## Key Python Dependencies

**LLM Export (`transformers/llm/export/requirements.txt`):**
`torch`, `transformers`, `onnx`, `onnxslim`, `onnxruntime`, `peft`, `sentencepiece`, `Pillow`, `Requests`, `tqdm`, `yaspin`, `numpy`, `datasets`

**PyMNN (`pymnn/pip_package/setup.py`):**
Core dependency: `numpy`. Optional: OpenCV, CUDA, TensorRT, Vulkan backends.

**Model Compression (`tools/mnncompress/setup.py`):**
`tensorly==0.4.5`, `aliyun-log-python-sdk`

**Documentation (`docs/requirements.txt`):**
Sphinx 5.0.0, recommonmark, sphinx_markdown_tables, sphinx_rtd_theme

## Configuration

**Build Configuration:**
- All configuration via CMake options (no separate config file)
- `cmake/MNNConfig.cmake.in` — CMake package export template
- `cmake/macros.cmake` — Platform library prefix/extension definitions
- `cmake/ios.toolchain.cmake` — iOS cross-compilation toolchain

**Runtime Configuration:**
- Backend registration at compile time via CMake options
- GPU tuning modes configured per-session (defined in `include/MNN/MNNForwardType.h`)
- Thread count configurable at runtime

---

*Stack analysis: 2026-05-27*
