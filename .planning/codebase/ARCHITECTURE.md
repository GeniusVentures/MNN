# Architecture

**Analysis Date:** 2026-05-27

## System Overview

MNN is a lightweight deep learning **inference engine** (not a training framework) for mobile and server platforms. It uses a **graph optimization + heterogeneous backend scheduling** architecture. The design prioritizes performance and binary size.

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                         Public API Layer                                  │
│  ┌────────────────────────────┬──────────────────────────────────────┐   │
│  │  Session API (low-level)   │  Module API (high-level, Express)    │   │
│  │  `include/MNN/Interpreter  │  `include/MNN/expr/Module.hpp`      │   │
│  │   .hpp`                    │  `include/MNN/expr/Executor.hpp`     │   │
│  │  Interpreter → createSession│  Module::load → onForward(VARP)     │   │
│  │  → runSession               │                                      │   │
│  └──────────────┬─────────────┴───────────────────┬──────────────────┘   │
│                 │                                  │                     │
│                 ▼                                  ▼                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Session / Pipeline Layer                      │   │
│  │  `source/core/Session.cpp`  `source/core/Pipeline.cpp`           │   │
│  │  Session manages one or more Pipelines.                          │   │
│  │  Pipeline: encode → allocMemory → execute                        │   │
│  │  (encode = shape compute + geometry transform)                    │   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             │                                            │
│                 ┌───────────┼──────────────┐                             │
│                 ▼           ▼              ▼                             │
│  ┌───────────────────┐ ┌───────────┐ ┌──────────────┐                    │
│  │  Shape Inference  │ │ Geometry  │ │  Schedule    │                    │
│  │  `source/shape/`   │ │`source/   │ │ `source/core/│                    │
│  │  SizeComputer per │ │ geometry/`│ │ Schedule.cpp`│                    │
│  │  op type           │ │ decomposes│ │ maps ops to  │                    │
│  │                    │ │ complex   │ │ pipelines &  │                    │
│  │                    │ │ ops       │ │ backends     │                    │
│  └───────────────────┘ └───────────┘ └──────────────┘                    │
│                                                                            │
│                             ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │               Heterogeneous Backend Layer                          │    │
│  │  Runtime → Backend → Execution (per-op implementation)             │    │
│  │  `source/backend/cpu/`  `source/backend/cuda/`                     │    │
│  │  `source/backend/metal/`  `source/backend/opencl/`                 │    │
│  │  `source/backend/vulkan/`  `source/backend/arm82/`                 │    │
│  │  `source/backend/tensorrt/`  `source/backend/nnapi/`               │    │
│  │  `source/backend/qnn/`  `source/backend/neuropilot/`               │    │
│  │  `source/backend/coreml/`  `source/backend/hiai/`                  │    │
│  │  `source/backend/opengl/`                                            │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                             │                                              │
│                             ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                     Tensor (Data Layer)                            │    │
│  │  `include/MNN/Tensor.hpp`  `source/core/Tensor.cpp`               │    │
│  │  Data format: NC4HW4 (channels packed by 4 for SIMD)              │    │
│  │  Host memory (`host`) vs Device memory (`deviceId`)               │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
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

**Overall:** Graph optimization with heterogeneous backend dispatch — model ops are scheduled into one or more pipelines, each pipeline bound to a hardware backend. Ops are first decomposed via the Geometry layer into backend-agnostic primitives, then executed by backend-specific Execution objects.

**Key Characteristics:**
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
  - `Interpreter.hpp` — Session API: model loading, session creation, session run, tensor access
  - `Tensor.hpp` — Data container with host/device memory and NC4HW4 format
  - `MNNForwardType.h` — Backend type enum (CPU=0, METAL=1, CUDA=2, OPENCL=3, AUTO=4, VULKAN=7, etc.), BackendConfig
  - `MNNDefine.h` — Platform macros, version info (v3.4.1)
  - `ErrorCode.hpp` — Error code enum
  - `expr/Module.hpp` — High-level Module abstraction (load/forward)
  - `expr/Executor.hpp` — Express graph executor with RuntimeManager
  - `expr/Expr.hpp` — Expression/VARP types for dynamic graph
  - `ImageProcess.hpp` — Image preprocessing
  - `Matrix.h` — Matrix utilities
  - `plugin/` — Plugin system API
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
  1. Resolve backend type (auto-detection priority: HIAI → CoreML → TensorRT → CUDA → OpenCL → Metal → Vulkan → CPU)
  2. Generate op graph per config (linear order or subgraph from path inputs/outputs)
  3. Create `ScheduleInfo` with `PipelineInfo` entries: `{BackendCache, OpCacheInfo[]}`
  4. Mark input/output/constant tensors by usage
- **Static models** (`Usage_INFERENCE_STATIC`): skip shape/geometry recompute

### Pipeline Layer (`source/core/Pipeline.cpp`)
- **Purpose:** Single-backend inference pipeline: **encode → allocMemory → execute**
- **Three-phase lifecycle:**
  1. **encode()** — Shape compute (SizeComputer per op) + Geometry transform (decompose ops to primitives) → fill executeBuffer with Commands
  2. **allocMemory()** — Create Execution objects per Command, acquire tensor buffers from Backend
  3. **execute()** — Call each Command's Execution::onExecute() in order
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
  - `Runtime` — singleton per forward type; creates Backend instances; manages threads, allocators, cache
  - `Backend` — per-pipeline instance; creates Execution objects; manages tensor memory (STATIC/DYNAMIC/DYNAMIC_SEPERATE storage)
  - `Execution` — per-op instance; `onResize()` + `onExecute()`
- **Backend list:**
  - `cpu/` — CPU (ARM/x86_64/RISC-V), includes ThreadPool, oneDNN, KleidiAI, BFloat16, Int8
  - `arm82/` — ARM v8.2+ SIMD extensions
  - `metal/` — Apple Metal GPU
  - `cuda/` — NVIDIA CUDA GPU
  - `opencl/` — OpenCL GPU (Android/desktop)
  - `vulkan/` — Vulkan GPU
  - `opengl/` — OpenGL ES GPU
  - `tensorrt/` — NVIDIA TensorRT
  - `coreml/` — Apple CoreML (via NN forward type)
  - `nnapi/` — Android NNAPI (via NN forward type)
  - `qnn/` — Qualcomm QNN (offline convert)
  - `neuropilot/` — MediaTek NeuroPilot (offline convert)
  - `hiai/` — Huawei HIAI (user backend slot)
- **Backend selection:**
  - `Compiler_Loop` (default for CPU): Use Geometry decomposition → run on loop-style backend
  - `Compiler_Geometry`: Decompose, then dispatch sub-ops
  - `Compiler_Origin`: No decomposition; backend handles op directly
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

1. **Model Loading** (`source/core/Interpreter.cpp:90-98`):
   - `Interpreter::createFromFile()` → `loadModelFile()` reads file via `FileLoader` → `createFromBufferInternal()` validates with `OpCommonUtils::checkNet()` → creates `Interpreter(Content*)`
   - FlatBuffers `GetNet(buffer)` deserializes model graph

2. **Session Creation** (`source/core/Interpreter.cpp:255-359`):
   - `createMultiPathSession(configs, runtime)` → `Schedule::schedule(info, net, configs, runtime)`
   - Schedule: for each config → resolve backend type → `_scheduleUnit()` builds op list → create `ScheduleInfo` with `PipelineInfo[]`
   - `new Session(std::move(info), modes, std::move(rt))` → session constructor creates `Pipeline` objects
   - If `validForResize` and `Session_Resize_Direct`: auto-call `session->resize()`

3. **Resize** (`source/core/Session.cpp:272-299`):
   - `Session::resize()` → for each pipeline: `pipeline->encode(debug, permitCodegen)` then `pipeline->allocMemory(firstMalloc, permitCodegen)`

4. **Pipeline Encode** (`source/core/Pipeline.cpp:201-238`):
   - If static model: copy op info directly to Command buffer
   - If dynamic model: `GeometryComputerUtils::shapeComputeAndGeometryTransform()` — runs shape inference + geometry decomposition → fills `executeBuffer`

5. **Pipeline Alloc Memory** (`source/core/Pipeline.cpp` ~`allocMemory`):
   - For each Command: create Execution via `backend->onCreate(inputs, outputs, op)` → `execution->onResize()` → acquire tensor buffers via `backend->onAcquireBuffer()`

6. **Run** (`source/core/Session.cpp:243-255`):
   - `Session::run()` → for each pipeline: `pipeline->execute()`
   - Pipeline execute: iterate Commands → `execution->onExecute(inputs, outputs)`

7. **Output Retrieval** (`source/core/Interpreter.cpp` `getSessionOutput`):
   - `Interpreter::getSessionOutput(session, name)` → `Session::getOutput(name)` → returns Tensor pointer
   - For device tensors: `Tensor::copyToHostTensor()` to read results

### Module API Path (Express)

1. **Module loading** (`include/MNN/expr/Module.hpp:72-76`):
   - `Module::load(inputs, outputs, file, config)` → internally creates `PipelineModule` from FlatBuffers model
   - `PipelineModule` (`express/module/PipelineModule.cpp`) wraps:
     - `ExprModule` for simple single-op modules
     - `StaticModule` for full inference with Session underneath

2. **Forward** (`express/module/StaticModule.hpp`):
   - `StaticModule::onForward(inputs)` → `_resize()` + `_execute()` → delegates to internal `Session` object

3. **Executor** (`include/MNN/expr/Executor.hpp`):
   - `Executor::RuntimeManager` manages Runtime instances
   - Lazy evaluation: `lazyEval = true` with modes `LAZY_FULL`, `LAZY_CONTENT`, `LAZY_COMPUTE_ONCE`
   - `Executor::makeCache()` pre-computes expression graph

### Converter Pipeline

1. **Source models** (ONNX, TF, Caffe, TFLite, Torch) → read by converter plugins in `tools/converter/source/`
2. **Common processing** (`tools/converter/source/common/`): `writeFb.cpp` serializes to MNN FlatBuffers
3. **Optimization** (`tools/converter/source/optimizer/`): graph optimization passes
4. **Output:** `.mnn` binary file (FlatBuffers format) with optional `external weight` file
5. **Post-processing:** `tools/converter/source/MNN/addBizCode.cpp` adds business identifier

### LLM Subsystem

1. **Python export** (`transformers/llm/export/`):
   - `llmexport.py` entry point: HuggingFace model → MNN format
   - `utils/model_mapper.py` — model field mapping
   - `utils/model.py` — unified `LlmModel` class
   - `utils/transformers.py` — Attention/Decoder/RoPE export

2. **C++ inference** (`transformers/llm/engine/`):
   - `src/llm.cpp` — text LLM inference with KVCache management and sampling
   - `src/omni.cpp` — multimodal: vision/audio
   - `include/llm/llm.hpp` — public API: `ChatMessage`, `Tokenizer`, `LlmConfig`, `Sampler`, `Pipeline`
   - Uses Module API (Express) internally for model execution
   - Supports speculative decoding (`speculative_decoding/`), disk embedding (`diskembedding.cpp`)

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

Every op goes through a registration chain:

1. **Schema definition** — FlatBuffers table in `schema/default/*.fbs` (e.g., `CaffeOp.fbs` defines `Convolution2DCommon`)
2. **OpType enum entry** — In `schema/default/MNN.fbs` (e.g., `Convolution = 29`)
3. **Shape inference** — `source/shape/Shape[OpName].cpp` registered via `ShapeRegister.cpp` → `SizeComputerSuite`
4. **Geometry decomposition (optional)** — `source/geometry/Geometry[OpName].cpp` registered via `GeometryOPRegister.cpp`
5. **Backend Execution** — Per-backend implementation. E.g., CPU: `source/backend/cpu/CPUConvolution.cpp` creates `Execution` subclass. GPU backends similarly.

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

---

*Architecture analysis: 2026-05-27*
