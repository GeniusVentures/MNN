<!-- GSD:project-start source:PROJECT.md -->
## Project

**MNN — Vulkan Backend LLM Enablement**

MNN is a lightweight deep learning inference engine targeting mobile and server platforms, supporting CNN, Transformer, LLM, and Diffusion models. This project focuses on completing the Vulkan backend to run LLMs end-to-end and adding Ultra FP4 quantization as a new Vulkan shader operation.

MNN lives as a submodule under `GeniusNetwork/thirdparty/` — upstream is external; this project adds Vulkan-specific capabilities that may require upstream contribution. Ultra FP4 design originates from the sibling `GeniusCogntiveSystem/docs/architecture/` workspace.

**Core Value:** **A complete Vulkan LLM inference pipeline** — the existing Vulkan Attention implementation (1505 lines) must graduate from compile-gated experimental code to a tested, production-ready op that can run LLMs on Vulkan-capable devices, followed by Ultra FP4 quantization for 4-bit float inference.

### Constraints

- **Tech stack:** C++11 (default) with C++17 optional; GLSL for Vulkan shaders; Python for export pipeline
- **Build:** CMake-based; RTTI and exceptions disabled (`-fno-rtti -fno-exceptions`)
- **Compatibility:** Must not break non-Vulkan backends; `MNN_SUPPORT_TRANSFORMER_FUSE` gating should eventually be removed
- **Code style:** Google Style variant — 4-space indent, 120-char lines, PascalCase classes, camelCase functions, mCamelCase members
- **Shader pipeline:** All GLSL edits must go through `makeshader.py` → regenerate `AllShader.cpp`, `AllShader.h`, `VulkanShaderMap.cpp`
- **Performance:** Vulkan ops must match or exceed CPU performance for LLM inference (target: sub-100ms per token on mid-range GPU)
- **Upstream:** Changes should be structured to be upstreamable to MNN mainline
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- C++ (C++11 default, optionally C++17) — Core engine, all backends, Express API, LLM/diffusion engines, tools, converter (`source/`, `express/`, `transformers/`, `tools/`)
- C (C99 / gnu99) — Selected backend kernels and third-party integrations (`3rd_party/`, Vulkan buffer shaders)
- Python 2.7+ / 3.5+ — PyMNN bindings (`pymnn/`), LLM export pipeline (`transformers/llm/export/`), model compression (`tools/mnncompress/`), Sherpa-MNN wrapper (`apps/frameworks/sherpa-mnn/`)
- Objective-C++ — Metal backend on Apple platforms (`source/backend/metal/*.mm`), CoreML backend
- Assembly — ARM NEON (aarch32/aarch64), ARM SME2, x86 SSE/AVX/AVX2/AVX512/VNNI kernels
## Version
## Runtime
- Native C/C++ runtime — No managed runtime; the library is a native shared/static library
- RTTI and exceptions **disabled** (`-fno-rtti -fno-exceptions` in CMakeLists.txt lines 613-614)
- Default symbol visibility: hidden in release builds (`-fvisibility=hidden`)
- None (CMake-based, no conan/vcpkg required)
- Vendored third-party libraries under `3rd_party/`
## Build System
- **CMake** 3.6+ — Main build system (`CMakeLists.txt`, 1104 lines)
- Build modes:
- Compiler requirements: GCC, Clang, AppleClang, or MSVC
- `build_lib.sh` — Unified build script for Android, iOS, HarmonyOS, and Python builds (807 lines)
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
- Session API (low-level): `Interpreter → createSession → runSession` — operates on raw Tensors
- Module API (high-level, recommended): `Module::load → onForward(VARP)` — Express-based dynamic graph
- Custom test runner (`run_test.out`) — built from `test/` directory
- Test framework: `MNN_BUILD_TEST=ON` builds tests via `test/CMakeLists.txt`
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
- **GitHub Actions** — 16 workflow files:
- CocoaPods: `MNN.podspec` (v2.2.0 listed), `MNN_Render.podspec`
- Framework builds via Xcode toolchain (`cmake/ios.toolchain.cmake`)
## Key Python Dependencies
## Configuration
- All configuration via CMake options (no separate config file)
- `cmake/MNNConfig.cmake.in` — CMake package export template
- `cmake/macros.cmake` — Platform library prefix/extension definitions
- `cmake/ios.toolchain.cmake` — iOS cross-compilation toolchain
- Backend registration at compile time via CMake options
- GPU tuning modes configured per-session (defined in `include/MNN/MNNForwardType.h`)
- Thread count configurable at runtime
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Code Style
- Indentation: 4 spaces (no tabs)
- Line width: 120 characters (`ColumnLimit: 120`)
- Braces: Attached (`BreakBeforeBraces: Attach`)
- Pointer alignment: Left (`int* ptr` not `int *ptr`)
- Single-line functions: allowed via `AllowShortFunctionsOnASingleLine: Inline`
- Single-line if/loops/blocks: disallowed
- No namespace indentation (`NamespaceIndentation: None`)
- Fix namespace comments: enabled (`FixNamespaceComments: true`)
- Include sorting: disabled (`SortIncludes: Never`)
- Max consecutive empty lines: 1
- Trailing comments aligned
- Pre-commit hook (`.pre-commit-config.yaml`): `git-clang-format` on changed lines only
- GitHub Actions CI (`code-format.yml`): validates format on PR using `git-clang-format --diff`
- Commit message format also enforced: `[Module:Type] Description` (Types: Feature, Bugfix, Perf, Refact, Style, Doc, Test, Chore)
## Naming Conventions
- PascalCase: `Interpreter`, `Session`, `Backend`, `Execution`, `AutoStorage`, `Tensor`, `MNNTestCase`, `MNNTestSuite`
- CPU backend ops: `CPU` prefix + PascalCase: `CPUConvolution`, `CPUBinary`, `CPUSoftmax`, `CPUAttention`
- Files: Class name with `.hpp`/`.cpp` (some older files use `.h` only)
- camelCase: `createSession`, `onExecute`, `onResize`, `onAcquireBuffer`, `readMap`, `writeMap`, `checkVector`, `getVersion`
- Template helpers in tests may use `_` prefix convention: `_Conv`, `_Input`, `_Const`
- Static factory functions: `create*` pattern: `Interpreter::createFromFile`, `Tensor::create`, `Tensor::createDevice`
- `mCamelCase`: `mData`, `mBackEnd`, `mValid`, `mSize`, `mTests`, `mSparse`, `mProc`, `mNeedAllocIO`
- Public member structs: may omit `m` prefix for POD-style: `Session::ModeGroup` fields use camelCase
- UPPER_SNAKE_CASE: `MNN_ERROR`, `MNN_PRINT`, `MNN_ASSERT`, `MNN_CHECK`, `ALIMIN`, `ALIMAX`, `UP_DIV`, `ROUND_UP`
- Debug-only macros: `FUNC_PRINT`, `FUNC_PRINT_ALL`, `AUTOTIME`
- Header guards: `#ifndef FileName_h` (no leading `_`), e.g., `#ifndef Backend_hpp`, `#ifndef MNNDefine_h`
- Test class name ends with `Test`: `ConvolutionTestOnCPU`, `BackendCopyBufferFloatTest`, `BinaryOPTest`
- Test registration string: `category/subcategory/name`: `"op/convolution/conv2d"`, `"speed/convolution/conv2d"`, `"engine/backend/copy_buffer_float"`
## C++ Standards
- C++17 used when `MNN_CUDA` with `MNN_SUPPORT_TRANSFORMER_FUSE` enabled, or when `CMAKE_CXX_STANDARD` is forced to 17
- C++0x fallback when `MNN_USE_CPP11` is OFF
- RTTI disabled via `-fno-rtti`
- Exceptions disabled via `-fno-exceptions`
- The codebase uses error codes and null returns instead of throw/catch
## Error Handling
- Core inference methods return `ErrorCode`: `Session::run()`, `Execution::onExecute()`, `Execution::onResize()`, `Backend::onResizeEnd()`
- Factory functions return `nullptr` on failure: `Interpreter::createFromFile()`, `MNNGetExtraRuntimeCreator()`
- Individual ops report via `ErrorCode`; framework aggregates and returns first non-zero code
- Test functions return `bool`: `true` for pass, `false` for failure
- `MNN_ERROR(format, ...)` — logs error message (platform-dependent: printf, android log, syslog)
- `MNN_PRINT(format, ...)` — logs informational message
- `MNN_CHECK(success, log)` — conditional error log
- `MNN_ASSERT(x)` — debug-only assertion (expands to nothing in release builds)
- `MNNTEST_ASSERT(x)` — test assertion, returns `false` from test function on failure
## Memory Management
- `AutoStorage<T>` — owns aligned heap buffer, reallocates on `reset()`, frees on destruction
- `AutoRelease<T>` — RAII for `new`/`delete` (non-copyable, reset overwrites)
- `BufferStorage` — owns `uint8_t*` buffer with `allocated_size` and `offset` tracking
- `RefCount` base class in `AutoStorage.h` with `addRef()`/`decRef()` — delete-on-zero-refs
- `SharedPtr<T>` — custom intrusive reference-counted pointer (pre-C++11 compatible)
- Macros: `SAFE_REF(x)`, `SAFE_UNREF(x)`, `SAFE_ASSIGN(dst, src)`
- Base class that deletes copy constructor, move constructor, copy assignment, move assignment
- Used by `Backend`, `Execution`, `NonCopyable`-derived creators, and other polymorphic classes
- `StorageType` enum controls buffer lifecycle: `STATIC`, `DYNAMIC`, `DYNAMIC_SEPERATE`, `DYNAMIC_IN_EXECUTION`
- `BufferAllocator` (`source/core/BufferAllocator.hpp`) manages GPU/device memory pools
- CPU backend uses `EagerBufferAllocator` / dynamic allocator with resize cache
- `std::shared_ptr` used extensively for higher-level objects: `Expression`, `Execution::Creator`, `Runtime`, `Tensor`
- `std::unique_ptr` used for `OpT` (FlatBuffers op descriptors), `FileLoader`, session containers
- Raw pointers (`Tensor*`, `Backend*`) in performance-critical execution paths
## Common Code Patterns
### 1. Op Registration (Schema → Shape → Backend)
### 2. Test Case Pattern
- `checkVector<T>(result, expected, size, threshold)` — absolute error check
- `checkVectorByRelativeError<T>(result, expected, size, rtol)` — relative error check
- `checkVectorByRelativeError<T>(result, expected1, expected2, size, rtol)` — check against two possible references
- `dispatch(std::function<void(MNNForwardType)> payload)` — iterate over available backends
- `FP32Converter` — array of precision-conversion functors (fp32, bf16, fp16)
### 3. Backend Implementation Pattern
### 4. Runtime / Backend Creation Pattern
### 5. Module/Session Pattern
### 6. Tensor Data Access Pattern
### 7. Variable/Express API Pattern
## File Header Convention
## Directory Convention
- Header files: `.hpp` for C++, `.h` for C-compatible headers
- Implementation: `.cpp` for C++, `.mm` for Objective-C++
- Test files: `*Test.cpp` for unit tests, `*SpeedTest.cpp` for benchmarks
- One class per header file (generally), with template helpers in same file
- Associated headers and sources share the same directory
## Logging
| Platform | MNN_PRINT | MNN_ERROR |
|----------|-----------|-----------|
| Android | `__android_log_print` (INFO) | `__android_log_print` (ERROR) |
| OHOS | hilog (DEBUG) | hilog (ERROR) |
| iOS | syslog + fprintf stderr | syslog + fprintf stderr |
| Default | printf | printf |
## Comments
## Function Design
## Module Design
## Include/Import Organization
## Anti-Patterns
### 1. `using namespace std;` (observed in test files)
### 2. Raw `new`/`delete` for test case objects
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## System Overview
```text
```
## Component Responsibilities
| Component | Responsibility | Key Files |
|-----------|----------------|-----------|
| **Interpreter** | Model loading, session creation/management, lifecycle | `include/MNN/Interpreter.hpp`, `source/core/Interpreter.cpp` |
| **Session** | Inference session: resize + run for a subgraph on one or more pipelines | `source/core/Session.hpp`, `source/core/Session.cpp` |
| **Pipeline** | Encode (shape+geometry), alloc memory, execute for a single backend config | `source/core/Pipeline.hpp`, `source/core/Pipeline.cpp` |
| **Schedule** | Maps model ops → PipelineInfo per ScheduleConfig (backend routing) | `source/core/Schedule.hpp`, `source/core/Schedule.cpp` |
| **Backend** | Abstract hardware backend: creates Execution instances, manages tensor memory | `source/core/Backend.hpp`, `source/core/Backend.cpp` |
| **Runtime** | Manages Backend lifecycle and shared resources per forward type | `source/core/Backend.hpp` (Runtime class) |
| **Execution** | Per-op compute implementation on a specific backend | `source/core/Execution.hpp`, `source/core/Execution.cpp` |
| **Tensor** | Data container: host/device memory, NC4HW4 format, regions for slicing | `include/MNN/Tensor.hpp`, `source/core/Tensor.cpp` |
| **SizeComputer** | Calculates output tensor shapes for each op type | `source/shape/SizeComputer.hpp`, `source/shape/ShapeRegister.cpp` |
| **GeometryComputer** | Decomposes ops into simpler primitives (Raster + compute ops) | `source/geometry/GeometryComputer.hpp`, `source/geometry/GeometryOPRegister.cpp` |
| **Executor** | High-level Express graph execution: lazy eval, runtime management | `include/MNN/expr/Executor.hpp`, `express/Executor.cpp` |
| **Module** | High-level module abstraction: `load`/`onForward` for dynamic graphs | `include/MNN/expr/Module.hpp`, `express/module/` |
| **Converter** | ONNX/TF/Caffe/TFLite/Torch → MNN FlatBuffers format | `tools/converter/source/MNNConverter.cpp`, `tools/converter/source/onnx/`, etc. |
## Pattern Overview
- **Two API levels:** Low-level Session API (Interpreter → Session) for maximum control; high-level Module API (Express dynamic graph) for LLM/Diffusion/modern workloads
- **FlatBuffers-based model format:** Schema definitions in `schema/default/*.fbs` → generated C++ types (`MNN_generated.h`). Models are serialized `.mnn` files.
- **NC4HW4 internal tensor format:** Channels packed in groups of 4 for SIMD efficiency across ARM/x86/GPU backends
- **Heterogeneous scheduling:** Each `ScheduleConfig` maps to a pipeline with a specific backend; multiple configs = multi-path session
- **Geometry decomposition:** Complex ops (Conv, Pool, etc.) are decomposed into Raster (data movement) + simpler compute ops
- **Op registration chain:** Schema (FlatBuffers) → Shape inference (`source/shape/`) → Geometry decomposition (`source/geometry/`) → Backend Execution (`source/backend/*/`)
- **Lazy evaluation in Express:** The Module API builds a dynamic compute graph; Executor resolves it lazily
## Layers
### Public API Layer (`include/MNN/`)
- **Purpose:** Stable public interface for all consumers
- **Location:** `include/MNN/`
- **Key headers:**
- **Depends on:** Nothing (self-contained public interface)
- **Used by:** All external consumers, apps, LLM/diffusion integrations
### Interpreter / Session Layer (`source/core/`)
- **Purpose:** Model lifecycle management (load → create session → resize → run → release)
- **Location:** `source/core/`
- **Contains:** `Interpreter.cpp`, `Session.cpp`, `Schedule.cpp`, `Pipeline.cpp`
- **Depends on:** Shape inference, Geometry, Backend, Tensor
- **Used by:** Both Session API and Module API
### Scheduling Layer (`source/core/Schedule.cpp`)
- **Purpose:** Routes model ops to pipelines based on ScheduleConfig (backend type, num threads, save tensors)
- **Key logic:**
- **Static models** (`Usage_INFERENCE_STATIC`): skip shape/geometry recompute
### Pipeline Layer (`source/core/Pipeline.cpp`)
- **Purpose:** Single-backend inference pipeline: **encode → allocMemory → execute**
- **Three-phase lifecycle:**
- **Geometry context:** `GeometryComputer::Context` manages region fusion, const tensor cache, raster operations
- **Geometry mask flags** control: region fuse, multi-region fuse, use loop (instead of raster+compute), cache
### Shape Inference Layer (`source/shape/`)
- **Purpose:** Computes output tensor shapes for each op type given input shapes
- **Location:** `source/shape/`
- **Key files:** `SizeComputer.hpp`, `ShapeRegister.cpp`, `ShapeConvolution.cpp`, `ShapeBinaryOp.cpp`, etc. (80 files)
- **Pattern:** Each op type has a `Shape[OpName].cpp` implementing `SizeComputer::onComputeSize()`
- **Registration:** `ShapeRegister.cpp` maps OpType → SizeComputer via `SizeComputerSuite::insert()`
- **Also computes:** FLOPs estimation via `onComputeFlops()`
### Geometry Decomposition Layer (`source/geometry/`)
- **Purpose:** Decomposes high-level ops into backend-agnostic primitives for heterogenous scheduling
- **Location:** `source/geometry/`
- **Key files:** `GeometryComputer.hpp`, `GeometryOPRegister.cpp` (49 files total)
- **Pattern:** Each op type has a `Geometry[OpName].cpp` implementing `GeometryComputer::onCompute()`
- **Output:** Fills a `CommandBuffer` with Raster ops (data movement) + simpler compute ops
- **Registration:** Via `REGISTER_GEOMETRY` macro, linked through `GeometryOPRegister.cpp`
- **Context features:** Region fuse optimization, const tensor caching, raster cache
- **Disable via:** CMake `MNN_SKIPBUILD_GEOMETRY` or `geometryMask = 0` when backend uses `Compiler_Origin`
### Heterogeneous Backend Layer (`source/backend/`)
- **Purpose:** Per-hardware op implementations
- **Location:** `source/backend/`
- **Backend hierarchy:**
- **Backend list:**
- **Backend selection:**
- **Runtime pluggability:** `RuntimeCreator` registration via `MNNInsertExtraRuntimeCreator()` / `MNNGetExtraRuntimeCreator()`
### Tensor / Data Layer
- **Purpose:** Data container abstracting host and device memory
- **Location:** `include/MNN/Tensor.hpp`, `source/core/Tensor.cpp`, `source/core/TensorUtils.hpp`
- **Internal format:** `MNN_DATA_FORMAT_NC4HW4` — channels packed by 4 for SIMD; `channel_pack_num = 4` (default)
- **Dimension types:** `TENSORFLOW` (NHWC), `CAFFE` (NCHW), `CAFFE_C4` (NC4HW4)
- **Memory types:** `MEMORY_BACKEND` (device), `MEMORY_HOST`, `MEMORY_VIRTUAL`, `MEMORY_OUTSIDE`
- **Usage:** INPUT, OUTPUT, CONSTANT, NORMAL, TRAINABLE
- **Regions:** Tensor slicing via `InsideDescribe::Region` (src/dst view + origin pointer)
- **Maximum dimensions:** 9 (`MNN_MAX_TENSOR_DIM`)
## Data Flow
### Primary Inference Path (Session API)
### Module API Path (Express)
### Converter Pipeline
### LLM Subsystem
## Key Abstractions
### Interpreter (`include/MNN/Interpreter.hpp`, `source/core/Interpreter.cpp`)
- **Purpose:** Model loading and session factory. Multiple sessions share one Interpreter (and model buffer).
- **Creation:** `createFromFile(path)` or `createFromBuffer(data, size)`
- **Session management:** `createSession(config)` / `createMultiPathSession(configs)` / `releaseSession(session)`
- **Inference:** `runSession(session)` / `runSessionWithCallBack(session, before, after)`
- **Model release:** `releaseModel()` frees model buffer to save memory after sessions created
### Session (`source/core/Session.hpp`, `source/core/Session.cpp`)
- **Purpose:** Inference session bound to specific pipelines and backends.
- **Lifecycle:** resize() → run() (repeatable); resize triggered by input shape changes
- **Modes:** Input_Inside/User, Output_Inside/User, Backend_Fix/Auto, Resize_Direct/Defer, Memory_Collect/Cache
- **Internals:** Vector of `Pipeline` shared_ptrs; `RuntimeInfo` (first=per-backend runtimes, second=CPU runtime)
### Pipeline (`source/core/Pipeline.hpp`, `source/core/Pipeline.cpp`)
- **Purpose:** Single-backend pipeline with encode → allocMemory → execute cycle.
- **Encode:** Shape compute + geometry transform → fills `CommandBuffer` (list of `Command` structs)
- **Command:** `{const Op*; Tensor* inputs/outputs; shared_ptr<Execution>; OperatorInfo}`
- **Tuning:** GPU backends can auto-tune op selection (`TuningAttr`)
### Schedule (`source/core/Schedule.hpp`, `source/core/Schedule.cpp`)
- **Purpose:** Static scheduling: maps model ops to `PipelineInfo` entries per `ScheduleConfig`
- **ScheduleInfo:** `{PipelineInfo[]; inputTensors; outputTensor; allTensors; defaultBackend}`
- **PipelineInfo:** `{BackendCache, OpCacheInfo[]}` — one per ScheduleConfig
- **Backend auto-detection:** Priority order for `MNN_FORWARD_AUTO`: HIAI → CoreML → TensorRT → CUDA → OpenCL → Metal → Vulkan → CPU
### Backend / Runtime / Execution (`source/core/Backend.hpp`, `source/core/Execution.hpp`)
- **Runtime:** Per-forward-type singleton. Creates Backends. `CompilerType` determines geometry usage. Manages thread pool (CPU) or device context (GPU).
- **Backend:** Per-pipeline instance. `onCreate(inputs, outputs, op) → Execution*`. Four storage types: STATIC, DYNAMIC, DYNAMIC_SEPERATE, DYNAMIC_IN_EXECUTION.
- **Execution:** Per-op compute. `onResize() → onExecute()`. Registered via `Execution::Creator` with `MNNForwardType` key.
- **Pluggability:** `MNNInsertExtraRuntimeCreator(type, creator)` / `MNNGetExtraRuntimeCreator(type)` for user backends.
### Tensor (`include/MNN/Tensor.hpp`, `source/core/TensorUtils.hpp`)
- **Purpose:** Universal data container. `host` pointer for CPU, `deviceId` for GPU/device.
- **Internal format:** NC4HW4 (channel_pack_num=4), extendable to pack16.
- **Regions:** Slice/view system via `InsideDescribe::Region` — describes memory reuse without copying.
- **Quantization:** `QuantAttr` attached to tensor describes scale/zero/min/max.
- **TensorArray:** Dynamic-sized array of tensors via `TensorArrayAttr`.
### SizeComputer (`source/shape/SizeComputer.hpp`)
- **Purpose:** Per-op shape inference. Registered in `SizeComputerSuite` singleton.
- **Interface:** `onComputeSize(op, inputs, outputs) → bool`, `onComputeFlops(op, inputs, outputs) → float`
- **Content-dependent shapes:** `mNeedContentInputIndex` tracks inputs whose content (not just shape) is needed.
### GeometryComputer (`source/geometry/GeometryComputer.hpp`)
- **Purpose:** Op decomposition for heterogeneous execution. Registered globally.
- **Interface:** `onCompute(op, inputs, outputs, context, cmd) → bool`
- **Context:** Manages const tensor caching, raster cache, region fusion.
- **DefaultGeometryComputer:** Fallback — wraps op in a single non-decomposed Command.
- **Registration:** `REGISTER_GEOMETRY` macro + `registerGeometryOps()`
## Entry Points
### Session API Entry Point
- **Location:** `include/MNN/Interpreter.hpp`
- **Triggers:** User creates Interpreter from file/buffer, creates Session, calls runSession
- **Responsibilities:** Model loading, session lifecycle, inference execution
### Module API Entry Point
- **Location:** `include/MNN/expr/Module.hpp`
- **Triggers:** `Module::load()` or `Module::extract()`; then `module->onForward(inputs)`
- **Responsibilities:** High-level model loading, dynamic graph execution
### Express API Entry Point
- **Location:** `include/MNN/expr/Executor.hpp`, `include/MNN/expr/Expr.hpp`
- **Triggers:** `Executor::getGlobalExecutor()`, VARP creation, `Expr` graph construction
- **Responsibilities:** Lazy evaluation, runtime management, memory GC
### Converter Entry Point
- **Location:** `tools/converter/source/MNNConverter.cpp`
- **Triggers:** CLI tool invocation: `MNNConvert -f ONNX --modelFile model.onnx --MNNModel output.mnn`
- **Responsibilities:** Format conversion, graph optimization, quantization
### LLM Entry Point
- **Location:** `transformers/llm/engine/src/llm.cpp`, `transformers/llm/engine/include/llm/llm.hpp`
- **Triggers:** `llm_demo` or `llm_bench` binaries; programmatic via `MNN::Transformer::Llm` class
- **Responsibilities:** Text generation, chat, KVCache management, sampling
## Architectural Constraints
- **Threading:** Single-threaded event loop per session by default. CPU backend uses `ThreadPool` for inner-op parallelism. GPU backends use async command submission.
- **RTTI and exceptions:** Both disabled (`-fno-rtti -fno-exceptions`). Errors returned via `ErrorCode`.
- **Global state:** `SizeComputerSuite::get()` (singleton), `MNNGetExtraRuntimeCreator()` (global creator map), `Executor::getGlobalExecutor()` (singleton executor). Backend creators registered via static initialization.
- **C++ standard:** C++11 default.
- **Maximum tensor dimensions:** 9 (`MNN_MAX_TENSOR_DIM` in `source/core/TensorUtils.hpp`)
- **Model format:** FlatBuffers-based `.mnn` binary. Schema in `schema/default/MNN.fbs` → generated `MNN_generated.h`.
## Op Registration Pattern
## Error Handling
- **Strategy:** Error codes throughout. `ErrorCode` enum defines `NO_ERROR`, `COMPUTE_SIZE_ERROR`, `NOT_SUPPORT`, etc.
- **Session level:** `Session::resize()` and `Session::run()` return `ErrorCode`. `Session::valid()` checks state.
- **Pipeline level:** `Pipeline::encode()` / `allocMemory()` / `execute()` return `ErrorCode`.
- **Execution level:** `Execution::onResize()` / `onExecute()` return `ErrorCode`. Failed Execution sets `mValid = false`.
- **Backend level:** `Runtime::pCurrentStatus` and `pExecutionStatus` track last error.
## Cross-Cutting Concerns
- **Logging:** Platform-adaptive: Android logcat (`__android_log_print`), iOS syslog + stderr, or `printf`. Macros: `MNN_PRINT`, `MNN_ERROR`, `MNN_ASSERT` (debug-only).
- **Validation:** `MNN_CHECK(success, log)` macro. `OpCommonUtils::checkNet()` validates model buffer integrity.
- **Authentication:** Model-level auth via `Interpreter::createFromBufferInternal(net, enforceAuth)` — auth enforcement is optional/build-config dependent.
- **Cache:** GPU backends support persistent tuning cache via `Interpreter::setCacheFile()` / `updateCacheFile()`. KVCache for LLM supports disk offload via `KVCACHE_SIZE_LIMIT` hint.
- **Memory management:** Backend storage types control buffer reuse (STATIC vs DYNAMIC vs DYNAMIC_SEPERATE). `Session_Memory_Collect` vs `Session_Memory_Cache` controls whether static memory is recycled or cached between resizes.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
