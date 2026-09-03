# Codebase Structure

**Analysis Date:** 2026-05-27

## Directory Layout

```
MNN/
├── include/MNN/              # Public C++ headers (stable API surface)
│   ├── expr/                 # Express/Module API headers
│   └── plugin/               # Plugin system API
├── source/                   # Core inference engine implementation
│   ├── core/                 # Session, Pipeline, Interpreter, Tensor, Backend
│   ├── backend/              # Hardware backend implementations (13 backends)
│   ├── shape/                # Shape inference (SizeComputer per op type)
│   ├── geometry/             # Geometry decomposition (ops → primitives)
│   ├── cv/                   # Computer vision utilities (image processing)
│   ├── math/                 # Math utilities
│   ├── utils/                # General utilities (InitNet, etc.)
│   ├── jni/                  # JNI bindings (Android)
│   └── plugin/               # Plugin system implementation
├── express/                  # Express API (dynamic graph, high-level)
│   └── module/               # Module implementations (PipelineModule, StaticModule, etc.)
├── tools/                    # Build tools and utilities
│   ├── converter/            # Model converter (ONNX/TF/Caffe/TFLite/Torch → MNN)
│   │   ├── source/           # Converter source (MNNConverter.cpp, plugins per format)
│   │   ├── include/          # Converter headers
│   │   └── tools/            # Converter auxiliary tools
│   ├── quantization/         # Quantization tools
│   ├── mnncompress/          # Model compression tools
│   ├── evaluation/           # Model evaluation/benchmark tools
│   ├── train/                # Training utilities (limited support)
│   ├── cpp/                  # C++ utility programs
│   ├── cv/                   # CV-related tools
│   ├── audio/                # Audio processing tools
│   └── script/               # Build/CI scripts
├── transformers/             # Model-specific subsystems
│   ├── llm/                  # LLM support
│   │   ├── export/           # Python export: HuggingFace → MNN
│   │   └── engine/           # C++ inference: llm.cpp, omni.cpp, KVCache, sampler
│   └── diffusion/            # Diffusion model support
├── schema/                   # FlatBuffers schema definitions
│   └── default/              # Main schema: MNN.fbs, CaffeOp.fbs, TensorflowOp.fbs, etc.
├── pymnn/                    # Python bindings (PyMNN)
├── test/                     # Test cases
├── demo/                     # Demo applications
├── 3rd_party/                # Third-party dependencies
├── cmake/                    # CMake modules
├── project/                  # IDE project files (Xcode, VS)
├── resource/                 # Model resources
├── docs/                     # Documentation sources
├── doc/                      # Generated documentation
├── codegen/                  # Code generation scripts/templates
├── apps/                     # Application examples
├── benchmark/                # Benchmarking tools
├── ciscripts/                # CI/CD scripts
├── package_scripts/          # Packaging scripts
├── backupcode/               # Backup/deprecated code
├── skills/                   # AI agent skills (for automated workflows)
├── .planning/                # Planning artifacts (this doc)
├── CMakeLists.txt            # Root CMake build
├── build_lib.sh              # One-shot library build script
├── release.sh                # Release build script
├── test.sh                   # Test runner script
├── CLAUDE.md                 # AI coding assistant context
├── README.md                 # Project README (English)
├── README_CN.md              # Project README (Chinese)
└── README_JP.md              # Project README (Japanese)
```

## Directory Purposes

### `include/MNN/` — Public API Headers
- **Purpose:** Stable, versioned public interface for all consumers. These headers are what external projects include.
- **Contains:** C++ header files defining the API surface: `Interpreter.hpp`, `Tensor.hpp`, `MNNDefine.h`, `MNNForwardType.h`, `ErrorCode.hpp`, `ImageProcess.hpp`, `Matrix.h`, `Rect.h`, `HalideRuntime.h`, `AutoTime.hpp`, `MNNSharedContext.h`
- **Subdirectories:**
  - `expr/` — Express API: `Module.hpp`, `Executor.hpp`, `Expr.hpp`, `ExprCreator.hpp`, `MathOp.hpp`, `NeuralNetWorkOp.hpp`, `Optimizer.hpp`, `Scope.hpp`, `ExecutorScope.hpp`
  - `plugin/` — Plugin system API
- **Key files:** `Interpreter.hpp` (545 lines, main entry point), `Tensor.hpp` (320 lines, data container)

### `source/core/` — Inference Core
- **Purpose:** Core inference engine: model lifecycle, scheduling, pipeline execution, backend abstraction, tensor management.
- **Contains:** 45 files (headers + implementations)
- **Key files:**
  - `Interpreter.cpp` (671 lines) — Model loading, session creation, inference dispatch
  - `Session.hpp` / `Session.cpp` (573 lines) — Session: resize + run for one subgraph
  - `Pipeline.hpp` / `Pipeline.cpp` (1250 lines) — Per-backend pipeline: encode → alloc → execute
  - `Schedule.hpp` / `Schedule.cpp` (437 lines) — Graph scheduling: map ops to pipelines
  - `Backend.hpp` (458 lines) — `Backend`, `Runtime`, `RuntimeCreator` abstractions
  - `Execution.hpp` (139 lines) — `Execution` abstract class (per-op compute)
  - `Command.hpp` (39 lines) — `Command` and `CommandBuffer` structs
  - `Tensor.cpp`, `TensorUtils.hpp` (228 lines) — Tensor implementation and utilities
  - `Backend.cpp`, `BufferAllocator.hpp`, `RuntimeFactory.hpp` — Backend infrastructure
  - `AutoStorage.h`, `Concurrency.h`, `Macro.h`, `NonCopyable.hpp` — Utility primitives
  - `FileLoader.hpp` / `FileLoader.cpp` — MNN model file I/O
  - `OpCommonUtils.hpp` / `OpCommonUtils.cpp` — Op utility functions
  - `WrapExecution.hpp` / `WrapExecution.cpp` — Execution wrappers (e.g., for geometry raster ops)
  - `KVCacheManager.hpp` / `KVCacheManager.cpp` — LLM key-value cache management
  - `MemoryFormater.h` — BF16/FP32 conversion, tensor format printing
  - `SimdHeader.h` — SIMD detection macros
  - `WinogradInt8Attr.hpp` — Winograd Int8 convolution attributes

### `source/backend/` — Hardware Backend Implementations
- **Purpose:** Per-hardware op implementations. Each subdirectory is a complete backend.
- **Contains:** 13 backend subdirectories
- **Backends:**
  - `cpu/` (164 files) — CPU backend: ARM, x86_64, RISC-V, ThreadPool, oneDNN, KleidiAI, BF16, Int8
    - CPU op implementations: `CPUConvolution.cpp`, `CPUBinary.cpp`, `CPUUnary.cpp`, `CPUPool.cpp`, `CPUSoftmax.cpp`, `CPURaster.cpp`, `CPUReduction.cpp`, `CPUMatMul.cpp`, `CPUAttention.cpp`, etc.
    - Sub-arch: `arm/`, `x86_x64/`, `riscv/`, `bf16/`, `compute/`, `render/`
    - Special: `ThreadPool.cpp`, `CPUOPRegister.cpp` (op registration), `CPUResizeCache.cpp`
  - `metal/` — Apple Metal GPU
  - `cuda/` — NVIDIA CUDA GPU
  - `opencl/` — OpenCL GPU (Android/desktop)
  - `vulkan/` — Vulkan GPU (with GLSL shaders)
  - `opengl/` — OpenGL ES GPU
  - `arm82/` — ARM v8.2+ FP16/SIMD extensions
  - `tensorrt/` — NVIDIA TensorRT backend
  - `coreml/` — Apple CoreML delegate
  - `nnapi/` — Android NNAPI delegate
  - `qnn/` — Qualcomm QNN backend
  - `neuropilot/` — MediaTek NeuroPilot backend
  - `hiai/` — Huawei HIAI backend

### `source/shape/` — Shape Inference
- **Purpose:** Computes output tensor shapes (and FLOPs estimates) for each op type.
- **Contains:** 80 files — `Shape[OpName].cpp` per op, plus `SizeComputer.hpp` (189 lines), `ShapeRegister.cpp`
- **Pattern:** Each `Shape*.cpp` file implements `SizeComputer::onComputeSize()` for one or more op types.
- **Key files:**
  - `SizeComputer.hpp` — Abstract `SizeComputer` class; `SizeComputerSuite` registry singleton
  - `ShapeRegister.cpp` — Maps all OpType values to SizeComputer instances
  - `ShapeConvolution.cpp`, `ShapeBinaryOp.cpp`, `ShapeMatMul.cpp`, `ShapeReshape.cpp`, etc.

### `source/geometry/` — Geometry Decomposition
- **Purpose:** Decomposes complex ops into simpler backend-agnostic primitives (Raster ops + compute ops).
- **Contains:** 49 files — `Geometry[OpName].cpp` per op, plus `GeometryComputer.hpp` (92 lines), `GeometryOPRegister.cpp`
- **Key files:**
  - `GeometryComputer.hpp` — Abstract `GeometryComputer` class; `DefaultGeometryComputer` fallback
  - `GeometryComputer.cpp` — Implementation, registration infrastructure
  - `GeometryComputerUtils.hpp` / `.cpp` — `shapeComputeAndGeometryTransform()` entry point
  - `ConvertUtils.hpp` / `.cpp` — Data format conversion utilities (NCHW ↔ NC4HW4 ↔ NHWC)
  - `GeometryConv2D.cpp` — Convolution decomposition (Im2Col + MatMul pattern)
  - `GeometryBinary.cpp`, `GeometryUnary.cpp`, `GeometryReduce.cpp`, etc.

### `express/` — Express (Dynamic Graph) API
- **Purpose:** High-level dynamic graph API. Module system for modern workloads (LLM, Diffusion).
- **Contains:** 11 top-level files + `module/` subdirectory (14 files)
- **Key files:**
  - `Executor.cpp`, `Expr.cpp`, `MathOp.cpp`, `NeuralNetWorkOp.cpp`, `Utils.cpp`
  - `module/Module.cpp` — Base Module class implementation
  - `module/PipelineModule.hpp` / `PipelineModule.cpp` — Express graph → Module conversion
  - `module/StaticModule.hpp` / `StaticModule.cpp` (66 lines header) — Wraps Session for Module API
  - `module/IfModule.cpp`, `WhileModule.cpp`, `MoEModule.cpp`, `NMSModule.cpp` — Special modules
  - `CMakeLists.txt` — Express build config

### `schema/` — FlatBuffers Schema
- **Purpose:** Defines the MNN model format (`.mnn` files are FlatBuffers binaries).
- **Contains:**
  - `default/MNN.fbs` (538 lines) — Master schema: `Net`, `Op`, `OpType` enum, `TensorDescribe`
  - `default/CaffeOp.fbs` (376 lines) — Caffe-origin op parameters (Convolution2D, Pool, etc.)
  - `default/TensorflowOp.fbs` — TensorFlow-origin op parameters
  - `default/TFQuantizeOp.fbs` — TF quantization op parameters
  - `default/ExtraInfo.fbs` — Extra metadata
  - `default/Tensor.fbs` — Tensor data type definitions
  - `default/Type.fbs` — Data type definitions
  - `default/UserDefine.fbs` — User-defined op extension
  - `default/TrainInfo.fbs` — Training metadata
  - `current/` — Current active schema copy
  - `generate.sh`, `generate.ps1` — Schema code generation scripts

### `tools/converter/` — Model Converter
- **Purpose:** Converts external model formats (ONNX, TensorFlow, Caffe, TFLite, Torch) to MNN format.
- **Contains:**
  - `source/MNNConverter.cpp` — Main converter entry point
  - `source/onnx/` — ONNX → MNN converter
  - `source/tensorflow/` — TensorFlow → MNN converter
  - `source/caffe/` — Caffe → MNN converter
  - `source/tflite/` — TFLite → MNN converter
  - `source/torch/` — TorchScript → MNN converter
  - `source/common/` — Shared converter utilities (writeFb.cpp, quantization, JSON, UUID, optimization)
  - `source/optimizer/` — Graph optimization passes (post-conversion)
  - `source/MNN/` — MNN model post-processing (addBizCode.cpp)
  - `source/compression/` — Model compression (sparsity, pruning)
  - `include/` — Converter public headers
  - `tools/` — Auxiliary converter tools, `forward.json`, `user_provide_quant_params.json`

### `transformers/llm/` — LLM Subsystem
- **Purpose:** End-to-end LLM export (Python) and inference (C++).
- **Contains:**
  - `export/` — Python: `llmexport.py` + `utils/` (model_mapper, model, transformers)
  - `engine/` — C++ inference engine:
    - `src/llm.cpp` — Text LLM inference
    - `src/omni.cpp` / `omni.hpp` — Multimodal (vision/audio) inference
    - `src/sampler.cpp` / `sampler.hpp` — Token sampling strategies
    - `src/llmconfig.cpp` / `llmconfig.hpp` — LLM configuration
    - `src/diskembedding.cpp` / `diskembedding.hpp` — Disk-based embeddings for large vocab
    - `src/kvmeta.hpp` — KVCache metadata
    - `src/speculative_decoding/` — Speculative decoding support
    - `src/tokenizer/` — Tokenizer implementations
    - `include/llm/llm.hpp` (259 lines) — Public LLM API: `Llm`, `ChatMessage`, `MultimodalPrompt`, etc.
    - `include/llm/reranker.hpp` — Re-ranker API
    - `demo/` — Demo applications
    - `test/` — LLM-specific tests
    - `tools/` — LLM utilities
    - `app/` — Production application examples
    - `benchmark/` — LLM benchmarking
    - `collect/` — Data collection tools
    - `config.json` — LLM engine configuration
  - `diffusion/` — Diffusion model support (Stable Diffusion, etc.)

### `tools/` — Build Tools and Utilities
- **Purpose:** Utilities for model conversion, quantization, compression, evaluation, training.
- **Subdirectories:** `converter/`, `quantization/`, `mnncompress/`, `evaluation/`, `train/`, `cpp/`, `cv/`, `audio/`, `script/`

### `pymnn/` — Python Bindings
- **Purpose:** Python API wrapping the C++ inference engine.
- **Pattern:** C++ → Python via PyBind or similar. Provides `mnn` Python package.

### `test/` — Test Suite
- **Purpose:** Unit tests, model tests, conversion tests, quantization tests, LLM tests.
- **Runner:** `test.sh` or `./run_test.out` from build directory.

### `3rd_party/` — Third-Party Dependencies
- **Purpose:** Vendored or referenced third-party libraries (FlatBuffers, half precision, etc.).

## Key File Locations

### Entry Points
- `include/MNN/Interpreter.hpp` — Session API entry point (model loading, session creation, inference)
- `include/MNN/expr/Module.hpp` — Module API entry point (high-level model loading + forward)
- `include/MNN/expr/Executor.hpp` — Express executor entry point
- `tools/converter/source/MNNConverter.cpp` — Converter CLI entry point
- `transformers/llm/engine/include/llm/llm.hpp` — LLM C++ API entry point
- `source/core/Interpreter.cpp` — Interpreter implementation (671 lines)
- `source/core/Session.cpp` — Session implementation (573 lines)

### Configuration
- `CMakeLists.txt` — Root build configuration with all `option()` flags
- `schema/default/MNN.fbs` — Model format schema (538 lines)
- `.clang-format` — C++ formatting rules (Google-style variant, 4-space indent, 120-char width)
- `.pre-commit-config.yaml` — Pre-commit hooks
- `MNN.podspec`, `MNN_Render.podspec` — iOS CocoaPods specs
- `transformers/llm/engine/config.json` — LLM engine default config

### Core Logic
- `source/core/Schedule.cpp` — Graph scheduling algorithm (437 lines)
- `source/core/Pipeline.cpp` — Pipeline encode/alloc/execute (1250 lines)
- `source/core/Backend.hpp` — Backend/Runtime abstraction (458 lines)
- `source/core/Execution.hpp` — Execution abstraction (139 lines)
- `source/core/TensorUtils.hpp` — Tensor internals and utilities (228 lines)
- `source/shape/SizeComputer.hpp` — Shape inference abstraction (189 lines)
- `source/shape/ShapeRegister.cpp` — Op type → SizeComputer mapping
- `source/geometry/GeometryComputer.hpp` — Geometry decomposition abstraction (92 lines)
- `source/geometry/GeometryOPRegister.cpp` — Op type → GeometryComputer mapping
- `source/geometry/GeometryComputerUtils.cpp` — shapeComputeAndGeometryTransform entry

### Backend-specific
- `source/backend/cpu/CPUBackend.hpp` (251 lines) — CPU Runtime + Backend
- `source/backend/cpu/CPUOPRegister.cpp` — CPU-specific op registration
- `source/backend/cpu/ThreadPool.cpp` / `ThreadPool.hpp` — CPU thread pool
- `source/backend/cpu/CPUConvolution.cpp` — Convolution (largest CPU op)

### Express / Module
- `express/module/PipelineModule.cpp` — Express→Module transformation
- `express/module/StaticModule.cpp` — Session-wrapping Module
- `express/Executor.cpp` — Express executor implementation

### LLM
- `transformers/llm/engine/src/llm.cpp` — LLM text inference
- `transformers/llm/engine/src/sampler.cpp` — Token sampling
- `transformers/llm/engine/include/llm/llm.hpp` — LLM public API
- `transformers/llm/export/llmexport.py` — Model export entry point

### Converter
- `tools/converter/source/MNNConverter.cpp` — Main converter
- `tools/converter/source/common/writeFb.cpp` — FlatBuffers serialization
- `tools/converter/source/onnx/` — ONNX import
- `tools/converter/source/common/WeightQuantAndCoding.cpp` — Weight quantization

### Testing
- `test/` — All test files (unit tests, model tests)
- `test.sh` — Test runner script

## Naming Conventions

### Files
- **C++ headers:** `PascalCase.hpp` for classes, `PascalCase.h` for C-compatible headers (e.g., `Interpreter.hpp`, `Tensor.hpp`, `MNNDefine.h`, `MNNForwardType.h`)
- **C++ sources:** `PascalCase.cpp` matching header name (e.g., `Interpreter.cpp`, `Schedule.cpp`)
- **Shape inference:** `Shape` prefix + OpName (e.g., `ShapeConvolution.cpp`, `ShapeBinaryOp.cpp`)
- **Geometry decomposition:** `Geometry` prefix + OpName (e.g., `GeometryConv2D.cpp`, `GeometryBinary.cpp`)
- **Backend op implementations:** `BackendPrefix` + OpName (e.g., `CPUConvolution.cpp`, `CPUBinary.cpp`)
- **Test files:** Various patterns under `test/`
- **FlatBuffers schema:** `PascalCase.fbs` (e.g., `MNN.fbs`, `CaffeOp.fbs`)
- **Python:** `snake_case.py` (e.g., `llmexport.py`, `model_mapper.py`)
- **Generated code:** `MNN_generated.h` (from FlatBuffers schema)

### Directories
- `snake_case` or single word lowercase (e.g., `source/core/`, `source/shape/`, `tools/converter/`)
- Backend subdirectories use lowercase abbreviations: `cpu/`, `cuda/`, `opencl/`, `vulkan/`, `arm82/`

### Symbols
- **Classes:** `PascalCase` (e.g., `Interpreter`, `Session`, `Backend`, `Execution`, `SizeComputer`, `GeometryComputer`)
- **Functions:** `camelCase` (e.g., `createSession`, `runSession`, `onCreate`, `onExecute`, `onComputeSize`)
- **Member variables:** `mCamelCase` (e.g., `mNet`, `mValid`, `mNeedResize`, `mPipelines`)
- **Macros:** `UPPER_SNAKE_CASE` (e.g., `MNN_PUBLIC`, `MNN_ASSERT`, `MNN_VERSION`, `MNN_MAX_TENSOR_DIM`)
- **Enums:** `PascalCase` or `UPPER_SNAKE_CASE` depending on context (e.g., `MNNForwardType` enum with `MNN_FORWARD_CPU`, `SessionMode` with `Session_Debug`)
- **Namespaces:** `MNN` (core), `MNN::Express` (Express API), `MNN::Transformer` (LLM)

## Where to Add New Code

### New Feature / New Op
- **Schema definition:** Add op table in `schema/default/CaffeOp.fbs` (or new `.fbs` file), add `OpType` enum entry in `schema/default/MNN.fbs`
- **Shape inference:** Add `source/shape/Shape[OpName].cpp`, register in `source/shape/ShapeRegister.cpp`
- **Geometry decomposition:** Add `source/geometry/Geometry[OpName].cpp`, register in `source/geometry/GeometryOPRegister.cpp`
- **CPU implementation:** Add `source/backend/cpu/CPU[OpName].cpp` + `.hpp`, register in `source/backend/cpu/CPUOPRegister.cpp`
- **GPU implementations:** Add to respective backend directories (`cuda/`, `metal/`, `opencl/`, `vulkan/`)
- **Tests:** Add under `test/`

### New Backend
- Create directory `source/backend/[name]/`
- Implement `Runtime` subclass (inherit `MNN::Runtime`)
- Implement `Backend` subclass (inherit `MNN::Backend`) — `onCreate()` returns `Execution` for each supported op
- Register via `MNNInsertExtraRuntimeCreator()` in `source/core/RuntimeFactory.cpp`

### New Module (Express API)
- Add `express/module/[Name]Module.cpp` + `.hpp`
- Inherit from `MNN::Express::Module`, implement `onForward(VARP) → VARP`

### Utilities
- Shared helpers: `source/core/` (if core), `source/utils/` (if auxiliary)
- Converter utilities: `tools/converter/source/common/`

### LLM Model Support
- Python export: `transformers/llm/export/` — follow skill: `skills/support-new-llm/SKILL.md`
- C++ inference: usually handled by generic engine; add config in `transformers/llm/engine/config.json` if needed

## Special Directories

### `.planning/`
- **Purpose:** AI-assisted planning artifacts (codebase maps, implementation plans)
- **Generated:** Yes (by GSD tools)
- **Committed:** Yes (shared planning context)

### `schema/current/`
- **Purpose:** Active copy of schema used for code generation
- **Generated:** Yes (from `schema/default/`)
- **Committed:** Yes

### `3rd_party/`
- **Purpose:** Vendored third-party code
- **Generated:** No
- **Committed:** Yes

### `build/` or `cmake-build-*/`
- **Purpose:** Build output directories (not in repo root structure, created by CMake)
- **Generated:** Yes
- **Committed:** No (gitignored)

### `project/`
- **Purpose:** IDE project files (Xcode, Visual Studio)
- **Generated:** Partially
- **Committed:** Yes (convenience)

### `resource/`
- **Purpose:** Test model resources
- **Generated:** No
- **Committed:** Yes (LFS or external)

### `backupcode/`
- **Purpose:** Deprecated/backup code not in active use
- **Generated:** No
- **Committed:** Yes

### Restricted Directories (do NOT read/modify without authorization)
- `schema/private/` — Internal proprietary schema
- `source/internal/` — Internal proprietary code

---

*Structure analysis: 2026-05-27*
