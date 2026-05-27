# Coding Conventions

**Analysis Date:** 2026-05-27

## Code Style

**Formatting Tool:** clang-format (via `git-clang-format` on changed lines only)
**Config:** `.clang-format` — BasedOnStyle: Google with project overrides

**Key Settings:**
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

**Format Enforcement:**
- Pre-commit hook (`.pre-commit-config.yaml`): `git-clang-format` on changed lines only
- GitHub Actions CI (`code-format.yml`): validates format on PR using `git-clang-format --diff`
- Commit message format also enforced: `[Module:Type] Description` (Types: Feature, Bugfix, Perf, Refact, Style, Doc, Test, Chore)

## Naming Conventions

**Classes:**
- PascalCase: `Interpreter`, `Session`, `Backend`, `Execution`, `AutoStorage`, `Tensor`, `MNNTestCase`, `MNNTestSuite`
- CPU backend ops: `CPU` prefix + PascalCase: `CPUConvolution`, `CPUBinary`, `CPUSoftmax`, `CPUAttention`
- Files: Class name with `.hpp`/`.cpp` (some older files use `.h` only)

**Functions:**
- camelCase: `createSession`, `onExecute`, `onResize`, `onAcquireBuffer`, `readMap`, `writeMap`, `checkVector`, `getVersion`
- Template helpers in tests may use `_` prefix convention: `_Conv`, `_Input`, `_Const`
- Static factory functions: `create*` pattern: `Interpreter::createFromFile`, `Tensor::create`, `Tensor::createDevice`

**Member Variables:**
- `mCamelCase`: `mData`, `mBackEnd`, `mValid`, `mSize`, `mTests`, `mSparse`, `mProc`, `mNeedAllocIO`
- Public member structs: may omit `m` prefix for POD-style: `Session::ModeGroup` fields use camelCase

**Constants and Macros:**
- UPPER_SNAKE_CASE: `MNN_ERROR`, `MNN_PRINT`, `MNN_ASSERT`, `MNN_CHECK`, `ALIMIN`, `ALIMAX`, `UP_DIV`, `ROUND_UP`
- Debug-only macros: `FUNC_PRINT`, `FUNC_PRINT_ALL`, `AUTOTIME`
- Header guards: `#ifndef FileName_h` (no leading `_`), e.g., `#ifndef Backend_hpp`, `#ifndef MNNDefine_h`

**Test Classes:**
- Test class name ends with `Test`: `ConvolutionTestOnCPU`, `BackendCopyBufferFloatTest`, `BinaryOPTest`
- Test registration string: `category/subcategory/name`: `"op/convolution/conv2d"`, `"speed/convolution/conv2d"`, `"engine/backend/copy_buffer_float"`

## C++ Standards

**Primary Standard:** C++11 (default, set via `CMAKE_CXX_STANDARD 11`)
**C Standard:** C99/gnu99 (`CMAKE_C_STANDARD 99`)

**Exceptions:**
- C++17 used when `MNN_CUDA` with `MNN_SUPPORT_TRANSFORMER_FUSE` enabled, or when `CMAKE_CXX_STANDARD` is forced to 17
- C++0x fallback when `MNN_USE_CPP11` is OFF

**Disabled Features:**
- RTTI disabled via `-fno-rtti`
- Exceptions disabled via `-fno-exceptions`
- The codebase uses error codes and null returns instead of throw/catch

## Error Handling

**No Exceptions:** The codebase compiles with `-fno-exceptions`. All error handling uses return values.

**Error Codes:** `include/MNN/ErrorCode.hpp` defines `enum ErrorCode`:
```cpp
NO_ERROR           = 0,    // Success
OUT_OF_MEMORY      = 1,    // Memory allocation failure
NOT_SUPPORT        = 2,    // Op/feature not supported
COMPUTE_SIZE_ERROR = 3,    // Shape computation failure
NO_EXECUTION       = 4,    // No backend execution found
INVALID_VALUE      = 5,    // Invalid parameter

// User errors
INPUT_DATA_ERROR = 10,
CALL_BACK_STOP   = 11,

// Op Resize errors
TENSOR_NOT_SUPPORT = 20,
TENSOR_NEED_DIVIDE = 21,

// File errors
FILE_CREATE_FAILED = 30,
FILE_REMOVE_FAILED = 31,
FILE_OPEN_FAILED   = 32,
FILE_CLOSE_FAILED  = 33,
FILE_RESIZE_FAILED = 34,
FILE_SEEK_FAILED   = 35,
FILE_NOT_EXIST     = 36,
FILE_UNMAP_FAILED  = 37
```

**Return Value Patterns:**
- Core inference methods return `ErrorCode`: `Session::run()`, `Execution::onExecute()`, `Execution::onResize()`, `Backend::onResizeEnd()`
- Factory functions return `nullptr` on failure: `Interpreter::createFromFile()`, `MNNGetExtraRuntimeCreator()`
- Individual ops report via `ErrorCode`; framework aggregates and returns first non-zero code
- Test functions return `bool`: `true` for pass, `false` for failure

**Error Logging:**
- `MNN_ERROR(format, ...)` — logs error message (platform-dependent: printf, android log, syslog)
- `MNN_PRINT(format, ...)` — logs informational message
- `MNN_CHECK(success, log)` — conditional error log
- `MNN_ASSERT(x)` — debug-only assertion (expands to nothing in release builds)
- `MNNTEST_ASSERT(x)` — test assertion, returns `false` from test function on failure

**Example from `source/core/Interpreter.cpp`:**
```cpp
Interpreter* Interpreter::createFromFile(const char* file) {
    Content* net = loadModelFile(file);
    if (nullptr == net) {
        return nullptr;
    }
    // ...
}
static void writeCacheFile(const Content* net, std::pair<const void*, size_t> buffer) {
    bool res = FileLoader::write(net->cacheFile.c_str(), buffer);
    if (!res) {
        MNN_ERROR("Write Cache File error!\n");
        return;
    }
}
```

## Memory Management

**Aligned Allocator:** Custom aligned allocation via `source/core/MNNMemoryUtils.h`:
```cpp
void* MNNMemoryAllocAlign(size_t size, size_t align);  // default align = 64
void  MNNMemoryFreeAlign(void* mem);
void* MNNMemoryCallocAlign(size_t size, size_t align);
```

**RAII Wrappers** (`source/core/AutoStorage.h`):
- `AutoStorage<T>` — owns aligned heap buffer, reallocates on `reset()`, frees on destruction
- `AutoRelease<T>` — RAII for `new`/`delete` (non-copyable, reset overwrites)
- `BufferStorage` — owns `uint8_t*` buffer with `allocated_size` and `offset` tracking

**Reference Counting:**
- `RefCount` base class in `AutoStorage.h` with `addRef()`/`decRef()` — delete-on-zero-refs
- `SharedPtr<T>` — custom intrusive reference-counted pointer (pre-C++11 compatible)
- Macros: `SAFE_REF(x)`, `SAFE_UNREF(x)`, `SAFE_ASSIGN(dst, src)`

**NonCopyable** (`source/core/NonCopyable.hpp`):
- Base class that deletes copy constructor, move constructor, copy assignment, move assignment
- Used by `Backend`, `Execution`, `NonCopyable`-derived creators, and other polymorphic classes

**Backend Memory:** (`source/core/Backend.hpp`)
- `StorageType` enum controls buffer lifecycle: `STATIC`, `DYNAMIC`, `DYNAMIC_SEPERATE`, `DYNAMIC_IN_EXECUTION`
- `BufferAllocator` (`source/core/BufferAllocator.hpp`) manages GPU/device memory pools
- CPU backend uses `EagerBufferAllocator` / dynamic allocator with resize cache

**Standard Smart Pointers:**
- `std::shared_ptr` used extensively for higher-level objects: `Expression`, `Execution::Creator`, `Runtime`, `Tensor`
- `std::unique_ptr` used for `OpT` (FlatBuffers op descriptors), `FileLoader`, session containers
- Raw pointers (`Tensor*`, `Backend*`) in performance-critical execution paths

## Common Code Patterns

### 1. Op Registration (Schema → Shape → Backend)

Three-layer op implementation:
1. **Schema definition** — FlatBuffers in `schema/default/*.fbs`
2. **Shape inference** — `source/shape/Shape<OpName>.cpp` implements `SizeComputer::onComputeSize()`
3. **Backend execution** — `source/backend/<backend>/CPU<OpName>.hpp` implements `Execution::onExecute()`

### 2. Test Case Pattern

Tests use a custom framework (not gtest/catch2). File: `test/MNNTestSuite.h`.

```cpp
// test/op/BinaryOPTest.cpp
class BinaryTestCommon : public MNNTestCase {
protected:
    template<typename Tin, typename Tout>
    bool test(VARP (*opFunc)(VARP, VARP), string name, float threshold, ...) {
        // Set up inputs, run op, compare output to reference
        auto output = opFunc(input_x, input_y);
        auto gotOutput = output->template readMap<Tout>();
        if (!checkVectorByRelativeError<Tout>(gotOutput, data_out.data(), size_out, threshold)) {
            MNN_ERROR("%s test failed!\n", name.c_str());
            return false;
        }
        return true;
    }
};

class AddTest : public BinaryTestCommon {
    virtual bool run(int precision) { ... }
};
MNNTestSuiteRegister(AddTest, "op/binary/add");
```

**Key test utilities** (`test/TestUtils.h`):
- `checkVector<T>(result, expected, size, threshold)` — absolute error check
- `checkVectorByRelativeError<T>(result, expected, size, rtol)` — relative error check
- `checkVectorByRelativeError<T>(result, expected1, expected2, size, rtol)` — check against two possible references
- `dispatch(std::function<void(MNNForwardType)> payload)` — iterate over available backends
- `FP32Converter` — array of precision-conversion functors (fp32, bf16, fp16)

### 3. Backend Implementation Pattern

```cpp
// source/backend/cpu/CPUBinary.hpp
class CPUBinary : public Execution {
public:
    CPUBinary(Backend *b, MNNBinaryExecute proc, int activationType) : Execution(b) {
        mProc = proc;
    }
    virtual ~CPUBinary() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) override;
private:
    MNNBinaryExecute mProc;
    int mNeedBroadcastIndex = -1;
    int mTotalSize;
    int mThreadNum;
};
```

### 4. Runtime / Backend Creation Pattern

```cpp
// source/backend/cpu/CPUBackend.hpp
class CPURuntime : public Runtime {
    virtual Backend* onCreate(const BackendConfig* config, Backend* origin) const override;
    virtual void onReset(int numberThread, const BackendConfig* config, bool full) override;
    virtual void onGabageCollect(int level) override;
    // ...
};
```

### 5. Module/Session Pattern

```cpp
// source/core/Session.hpp
class Session {
    ErrorCode run() const;
    ErrorCode runWithCallBack(const TensorCallBackWithInfo& before,
                               const TensorCallBackWithInfo& after,
                               bool sync = false) const;
    // ...
};
```

### 6. Tensor Data Access Pattern

```cpp
auto hostTensor = Tensor::create<float>({batch, channel, height, width}, nullptr, Tensor::CAFFE);
auto hostData = hostTensor->host<float>();
hostData[i] = value;

auto deviceTensor = Tensor::createDevice<float>({batch, height, width, channel});
bn->onAcquireBuffer(deviceTensor.get(), Backend::DYNAMIC);
bn->onCopyBuffer(hostTensor.get(), deviceTensor.get());

// Read back
auto outputPtr = deviceTensor->host<float>();
// Or via Express API:
float* data = var->writeMap<float>();
float* data = var->readMap<float>();
```

### 7. Variable/Express API Pattern

```cpp
using namespace MNN::Express;
auto input = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
::memcpy(input->writeMap<float>(), data, size);
auto weight = _Const(weightData.data(), {oc, ic, kh, kw}, NCHW, halide_type_of<float>());
auto output = _Conv(weight, bias, _Convert(input, NC4HW4), pad, stride, dilate, group, pads);
auto outputData = output->readMap<float>();
```

## File Header Convention

All source files follow this pattern:
```cpp
//
//  FileName.cpp
//  MNN
//
//  Created by MNN on YYYY/MM/DD.
//  Copyright © 2018, Alibaba Group Holding Limited
//
```

## Directory Convention

- Header files: `.hpp` for C++, `.h` for C-compatible headers
- Implementation: `.cpp` for C++, `.mm` for Objective-C++
- Test files: `*Test.cpp` for unit tests, `*SpeedTest.cpp` for benchmarks
- One class per header file (generally), with template helpers in same file
- Associated headers and sources share the same directory

## Logging

**Framework:** Platform-dependent printf/logcat/syslog via macros in `include/MNN/MNNDefine.h`

**Platform dispatch:**
| Platform | MNN_PRINT | MNN_ERROR |
|----------|-----------|-----------|
| Android | `__android_log_print` (INFO) | `__android_log_print` (ERROR) |
| OHOS | hilog (DEBUG) | hilog (ERROR) |
| iOS | syslog + fprintf stderr | syslog + fprintf stderr |
| Default | printf | printf |

**Pattern:** Log on error conditions only. Performance-critical paths should not log.

## Comments

**When to comment:** Public API headers use Doxygen-style `@brief`/`@param`/`@return`/`@warning` blocks. Implementation files use minimal comments, relying on clear naming.

**JSDoc/TSDoc:** Not applicable (C++ codebase).

## Function Design

**Size:** Functions tend to be moderate (20-60 lines). Large parameter-sweep test functions exist in test files.

**Parameters:** When > 3 parameters, consider grouping into a struct (e.g., `Backend::Info`, `Session::ModeGroup`).

**Return Values:** `ErrorCode` for operations that can fail. `bool` for tests and shape computations. `nullptr` for factory failures. `float`/`void` for computations that cannot fail.

## Module Design

**Exports:** Use `MNN_PUBLIC` macro for public API visibility. Public headers under `include/MNN/`.

**Internal headers:** Under `source/core/`, `source/backend/<name>/`, etc. Not exported.

**Historical copy artifacts (#3883):** Some source files exist under both `source/core/` and `source/backend/cpu/` with near-identical content. This is an artifact of a historical copy. Use `source/core/` as the canonical location.

## Include/Import Organization

**Standard order observed:**
1. Corresponding header file (for `.cpp` files)
2. MNN public headers (`<MNN/...>`)
3. Internal core headers (`"core/..."`, `"backend/..."`)
4. Standard library headers (`<vector>`, `<string>`, `<map>`, etc.)

**Path Aliases:** CMake `target_include_directories` ensures `source/` and `include/` are on include path. Internal headers included with relative paths like `"core/Backend.hpp"`.

## Anti-Patterns

### 1. `using namespace std;` (observed in test files)
**What happens:** `using namespace std;` at top of some test files (`test/op/BinaryOPTest.cpp`)
**Why it's wrong:** Pollutes namespace, can cause ambiguous compilation when combined with MNN names
**Do this instead:** Qualify `std::` explicitly or use targeted `using std::string;` declarations

### 2. Raw `new`/`delete` for test case objects
**What happens:** `MNNTestSuite` stores raw `MNNTestCase*` pointers allocated with `new` and cleaned in destructor (`test/MNNTestSuite.cpp`)
**Why it's wrong:** Potential for leaks if suite is not properly destroyed
**Do this instead:** This is acceptable for test framework simplicity, but new tests should consider `std::unique_ptr`.

---

*Convention analysis: 2026-05-27*
