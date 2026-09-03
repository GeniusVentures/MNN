# Testing Patterns

**Analysis Date:** 2026-05-27

## Test Framework

**Runner:** Custom framework (not gtest/catch2) — implemented in `test/MNNTestSuite.h` and `test/MNNTestSuite.cpp`
**Binary:** `run_test.out` (built from all `.cpp` files under `test/`, glob-recursed by CMake)

**Test Class Hierarchy:**
- `MNNTestCase` — pure abstract base class requiring `virtual bool run(int precision) = 0` (`test/MNNTestSuite.h:34`)
- `MNNTestSuite` — singleton container that holds and runs all registered tests
- `MNNTestRegister<Case>` — static registration template (instantiated via `MNNTestSuiteRegister` macro)

### Registration Macro

```cpp
// test/MNNTestSuite.h:121
#define MNNTestSuiteRegister(Case, name) static MNNTestRegister<Case> __r##Case(name)

// Usage:
MNNTestSuiteRegister(ConvolutionTestOnCPU, "op/convolution/conv2d");
MNNTestSuiteRegister(BackendCopyBufferFloatTest, "engine/backend/copy_buffer_float");
```

**Convention:** Test names are hierarchical strings:
| Prefix | Category | Example |
|--------|----------|---------|
| `op/` | Operator unit tests | `op/convolution/conv2d`, `op/binary/add` |
| `speed/` | Performance benchmarks | `speed/convolution/conv2d`, `speed/MatMul` |
| `engine/` | Engine/infrastructure tests | `engine/backend/copy_buffer_float` |
| `model/` | Full model tests | `model/MobileNet`, `model/Transformer` |
| `expr/` | Express API tests | `expr/ModuleTest`, `expr/MatMulTest` |
| `cv/` | Computer vision tests | `cv/ImageProcessTest` |
| `nn/` | Neural network training tests | (only built when `MNN_BUILD_TRAIN=ON`) |
| `grad/` | Gradient/training tests | (only built when `MNN_BUILD_TRAIN=ON`) |

### Assertion Macros

```cpp
// test/MNNTestSuite.h:122-129
#define MNNTEST_ASSERT(x)                                        \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            MNN_ERROR("Error for %s, %d\n", __func__, __LINE__); \
            return false;                                        \
        }                                                        \
    }
```

Tests return `bool` — `true` for pass, `false` for failure. No exception-based assertion.

### Test Output Format

```cpp
// test/MNNTestSuite.cpp:34-37
static void printTestResult(int wrong, int right, const char* flag) {
    MNN_PRINT("TEST_NAME_UNIT%s: 单元测试%s\nTEST_CASE_AMOUNT_UNIT%s: ", flag, flag, flag);
    MNN_PRINT("{\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n", wrong, right);
    MNN_PRINT("TEST_CASE={\"name\":\"单元测试%s\",\"failed\":%d,\"passed\":%d}\n", flag, wrong, right);
}
```

## Test File Organization

**Location:** `test/` directory at project root. Tests are co-located with the test framework, not with the source code.

**Directory Structure:**
```
test/
├── CMakeLists.txt          # Build config: globs all *.cpp, links MNN_DEPS
├── main.cpp                # Entry point: parses CLI args, sets up executor
├── MNNTestSuite.h          # Test framework header (MNNTestCase, MNNTestSuite, MNNTestRegister)
├── MNNTestSuite.cpp        # Framework implementation (run/runAll logic, result printing)
├── TestUtils.h             # Test utilities (checkVector, FP32Converter, dispatch helpers)
├── TestUtils.cpp           # Test utilities implementation (dispatch, getCurrentType)
├── CommonOpCreator.hpp     # Shared op-construction helpers for tests (_Conv, _HybridConv, etc.)
├── op/                     # Operator unit tests (100+ files, one per op type)
│   ├── ConvolutionTest.cpp
│   ├── BinaryOPTest.cpp
│   ├── MatMulTest.cpp
│   └── ...
├── core/                   # Core engine tests
│   ├── BackendTest.cpp     # Backend buffer copy tests
│   ├── TensorTest.cpp      # Tensor shape/data tests
│   ├── BufferAllocatorTest.cpp
│   └── ...
├── expr/                   # Express API tests
│   ├── ModuleTest.cpp
│   ├── MatMulTest.cpp
│   └── ...
├── model/                  # Full model integration tests
│   ├── MobileNetTest.cpp
│   ├── SqueezeNetTest.cpp
│   └── TransformerTest.cpp
├── speed/                  # Performance benchmark tests
│   ├── BinarySpeedTest.cpp
│   ├── ConvSpeedInt8Test.cpp
│   ├── MatMulSpeed.cpp
│   └── ...
├── cv/                     # Image processing tests
│   ├── ImageProcessTest.cpp
│   └── MatrixTest.cpp
├── clone/                  # Model cloning tests
│   └── CloneNetTest.cpp
├── grad/                   # Gradient tests (MNN_BUILD_TRAIN only)
├── nn/                     # Neural network training tests (MNN_BUILD_TRAIN only)
├── backend/                # Backend-specific tests
├── plugin/                 # Plugin system tests
├── kleidiai/               # KleidiAI tests
└── sharedmem/              # Shared memory tests
```

**Naming Convention:**
- Files: `*Test.cpp` for correctness tests, `*SpeedTest.cpp` or `*Speed.cpp` for benchmarks
- One test file per op or feature group

## Build Configuration

**CMake options** (`test/CMakeLists.txt`):
```cmake
# All .cpp files under test/ are globbed into run_test.out
file(GLOB_RECURSE Files ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_executable(run_test.out ${Files})
target_link_libraries(run_test.out ${MNN_DEPS})
```

**Conditional compilation:**
- `grad/` and `nn/` test files are excluded when `MNN_BUILD_TRAIN=OFF`
- On Apple platforms, `.mm` (Objective-C++) files are also included for Metal dispatch
- `MNN_WITH_PLUGIN` adds `test/plugin/` and links `plugin_matmul`
- Android adds `android` library dependency
- `MNN_SUPPORT_BF16` adds compile definition

## How to Run Tests

**Build:**
```bash
mkdir build && cd build
cmake .. -DMNN_BUILD_TEST=ON   # other flags as needed
make -j$(nproc)
```

**Run:**
```bash
# Run all tests (skips "speed" and "model" tests automatically)
./run_test.out

# Run tests matching a name prefix
./run_test.out "op/convolution"     # runs all convolution tests

# With backend/precision/thread options
./run_test.out all [backend] [precision] [thread] [flag] [memory] [dynamicOption] [kleidiAI] [divisionRatio]

# Help
./run_test.out --help
```

**Command-line arguments** (`test/main.cpp`):
| Position | Argument | Default | Description |
|----------|----------|---------|-------------|
| 1 | test_name | all | Name prefix filter; "all" runs everything |
| 2 | backend | 0 (CPU) | 0=CPU, 3=OpenCL |
| 3 | precision | 1 (High) | 0=Normal, 1=High, 2=Low, 3=Low_BF16 |
| 4 | thread/mode | 1 | Thread count or GPU mode |
| 5 | flag | "" | Test result label |
| 6 | memory | 0 (Normal) | Memory mode |
| 7 | dynamicOption | 0 | Dynamic quantization option |
| 8 | kleidiAI | 0 | Enable KleidiAI |
| 9 | divisionRatio | 1 | SME/NEON division ratio |

**Test filtering in `runAll()`** (`test/MNNTestSuite.cpp:77-113`):
- Tests with "speed" in name are automatically skipped (benchmark only)
- Tests with "model" in name are automatically skipped (require model resources)
- Run specific tests by passing a name prefix: `./run_test.out "op/convolution"`

## Test Patterns

### Pattern 1: Standard Operator Test

```cpp
// test/op/BinaryOPTest.cpp

class BinaryTestCommon : public MNNTestCase {
protected:
    template<typename Tin, typename Tout>
    bool test(VARP (*opFunc)(VARP, VARP), string name, float threshold,
              const vector<Tin>& data_x, const vector<Tin>& data_y,
              const vector<Tout>& data_out, ...) {
        // 1. Create input tensors
        auto input_x = _Input(shape_x, format, halide_type_of<Tin>());
        auto input_y = _Input(shape_y, format, halide_type_of<Tin>());
        // 2. Set input data
        auto ptr_x = input_x->template writeMap<Tin>();
        memcpy(ptr_x, data_x.data(), size_x * sizeof(Tin));
        input_x->unMap();
        // 3. Run the op
        auto output = opFunc(input_x, input_y);
        // 4. Read output and compare
        auto gotOutput = output->template readMap<Tout>();
        if (!checkVectorByRelativeError<Tout>(gotOutput, data_out.data(), size_out, threshold)) {
            MNN_ERROR("%s test failed!\n", name.c_str());
            return false;
        }
        return true;
    }
};

// Concrete test class
class AddTest : public BinaryTestCommon {
public:
    virtual ~AddTest() = default;
    virtual bool run(int precision) {
        // Run specific test cases
        return test<float, float>(_Add, "Add", 0.01, data_x, data_y, data_out,
                                   {1, 3, 4, 5}, {1, 3, 4, 5}, {1, 3, 4, 5});
    }
};
MNNTestSuiteRegister(AddTest, "op/binary/add");
```

### Pattern 2: Parameter Sweep Test

```cpp
// test/op/ConvolutionTest.cpp — ConvolutionTest template class

template <typename ConvolutionType>
class ConvolutionTest : public ConvolutionType {
protected:
    static bool test(MNNForwardType type, const std::string& device_name,
                     int precision, MNN::SparseAlgo sparseAlgo,
                     std::vector<int> blocks, bool checkSpectial = false) {
        // Sweep over batch sizes, channels, spatial dims, kernels, etc.
        for (int b = 1; b <= 2; b++) {
            for (auto oc : ocSize) {
                for (auto ic : icSize) {
                    for (auto is : isSize) {
                        for (int kw = 1; kw <= 3; kw+=2) {
                            for (int kh = 1; kh <= 3; kh+=3) {
                                for (int d = 1; d <= 2; d++) {
                                    for (int s = 1; s <= 2; s++) {
                                        for (int p = 0; p <= 1; p++) {
                                            bool succ = ConvolutionType().test(
                                                type, device_name, "Conv2D", b, ic, oc, is, is,
                                                PadMode_CAFFE, p, p, kh, kw, s, d, 1, precision, ...);
                                            if (!succ) {
                                                MNN_ERROR("Error for conv ...\n");
                                                return false;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
};
```

### Pattern 3: Multi-Backend Dispatch

```cpp
// test/core/BackendTest.cpp

class BackendCopyBufferFloatTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        for (int i = 0; i < MNN_FORWARD_ALL; ++i) {
            auto type = (MNNForwardType)i;
            auto creator = MNNGetExtraRuntimeCreator(type);
            if (nullptr == creator) continue;

            for (int p = 0; p < 3; ++p) {
                MNN::Backend::Info info;
                info.type = type;
                BackendConfig user;
                user.precision = (MNN::BackendConfig::PrecisionMode)p;
                info.user = &user;
                std::shared_ptr<Runtime> runtime(creator->onCreate(info));
                std::shared_ptr<Backend> bn(runtime->onCreate(&user));

                // Run backend-specific tests
                bool res = NC4HW4_2_NC4HW4_float(bn);
                if (!res) return false;
            }
        }
        return true;
    }
};
```

### Pattern 4: Reference Implementation Comparison

Tests compute expected values using a CPU reference implementation, then compare against MNN's output:

```cpp
// test/op/ConvolutionTest.cpp — ConvolutionCommonTest::test()
// 1. Compute reference output
reference_conv2d(inputData, weightData, biasData, outputData, ...);

// 2. Run MNN
auto input = _Input({batch, ic, ih, iw}, NCHW, halide_type_of<float>());
auto output = _Conv(weightVar, biasVar, _Convert(input, NC4HW4), ...);
auto outputPtr = output->readMap<float>();

// 3. Compare
if (!checkVectorByRelativeError<float>(outputPtr, outputData.data(),
                                        outputData.size(), 0.05)) {
    MNN_ERROR("test failed\n");
    return false;
}
```

### Pattern 5: Precision-Aware Testing

Tests account for precision mode when setting error thresholds:

```cpp
float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 100;
if (!checkVectorByRelativeError<float>(outputPtr, expected, size, 0.001 * errorScale)) {
    // ...
}
```

**Precision conversion functions** (test/TestUtils.h:110-121):
```cpp
using ConvertFP32 = float(*)(float fp32Value);
const static std::vector<ConvertFP32> FP32Converter = {
    keepFP32Precision,    // Index 0,1: fp32
    convertFP32ToBF16,    // Index 2: bf16 (if MNN_SUPPORT_BF16)
    convertFP32ToFP16     // Index 3: fp16
};
```

## Validation Helpers

**Absolute error** (`test/TestUtils.h:44`):
```cpp
template <typename T>
bool checkVector(const T* result, const T* rightData, size_t size, T threshold);
```

**Relative error** (`test/TestUtils.h:58`):
```cpp
template <typename T>
bool checkVectorByRelativeError(const T* result, const T* rightData, int size, float rtol);
```

**Relative error with dual references** (`test/TestUtils.h:78`):
```cpp
template <typename T>
bool checkVectorByRelativeError(const T* result, const T* rightData, const T* alterRightData,
                                 int size, float rtol);
```
Used when there are two valid outputs (e.g., pre-fused bias vs. post-fused bias in convolution).

## Mocking and Stubbing

**This codebase does not use mocking frameworks.** Since MNN is a compute engine with deterministic outputs:
- Unit tests compute reference values using CPU reference implementations written directly in test code
- Backend tests create real Runtimes and Backends, not mocks
- No gmock, FakeIt, or similar libraries are used
- Test isolation relies on creating fresh Executor/Runtime instances per test

**What NOT to Mock:**
- Tensor data — use actual data buffers
- Backend/Execution — create real instances through the Runtime factory
- Ops — construct via Express API with real compute

## Fixtures and Test Data

**Test Data Generation:** Random but deterministic data using fixed seeds:
```cpp
#define TEST_RANDOM_SEED 100
srand(TEST_RANDOM_SEED);
```

**Common patterns:**
- Identity-like but non-trivial weights: `((i / kw) % 1317) * ((i / kh) % 1317) + ...`
- Small random input values: `(data % 255) / 255.0f` range
- Bias values with varied magnitude

**Location:** Test data is generated inline within test functions. No external fixture files are used.

## Coverage

**Coverage flag:** `MNN_ENABLE_COVERAGE` (`CMakeLists.txt:70`)
```cmake
option(MNN_ENABLE_COVERAGE "Build with coverage enable" OFF)
# When enabled:
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
```

**No enforced threshold.** Coverage is opt-in and used for manual analysis.

## Test Types

### Unit Tests (`test/op/`, `test/core/`)
- Scope: Individual operators, tensor operations, buffer management
- Run via: `./run_test.out "op/"` or `./run_test.out "core/"`
- Self-contained: no external model files needed
- Parameterized: sweep over input sizes, data types (float/int8/int4), pad modes

### Integration Tests (`test/model/`, `test/expr/`)
- Scope: Full model inference, Express API workflows
- Run via: model tests are skipped in `runAll()` (require external `.mnn` model files)
- Examples: MobileNet, SqueezeNet, Transformer inference

### Performance Benchmarks (`test/speed/`)
- Scope: Measure execution time of specific operations
- Run via: individual test names (skipped in `runAll()`)
- Pattern: Use `AUTOTIME` macro or `MNN::Timer` for wall-clock measurement
- Emphasis: Many iterations over large tensors

### Gradient Tests (`test/grad/`)
- Scope: Verify gradient computation correctness (training mode)
- Only built when `MNN_BUILD_TRAIN=ON`

### Conversion Tests (`test.sh`)
- TIFF/TFLite/ONNX/Torch model conversion tests
- Run via shell script `test.sh` (not `run_test.out`)
- Validate round-trip conversion accuracy

## CI Configuration

### Primary CI: GitHub Actions (`.github/workflows/`)
| Workflow | File | Triggers | What it runs |
|----------|------|----------|-------------|
| linux | `linux.yml` | push to master/feature/**, PR to master | Build + test (CPU, non-SSE, AVX512 variants) |
| macos | `macos.yml` | push to master/feature/**, PR to master | Build + test (macOS, CPU+OpenCL+Vulkan, LLM) |
| android | `android.yml` | push to master/feature/**, PR to master | Build only: arm64 + arm32 |
| code-format | `code-format.yml` | PR to master | Commit message format + clang-format diff check |
| windows | `windows.yml` | push to master/feature/** | Build + test (MSVC) |
| ios | `ios.yml` | push/PR to master | Build (Xcode) |
| stale | `stale.yml` | scheduled | Mark stale issues/PRs |

**Secondary CI: Travis CI** (`.travis.yml`):
- macOS: CPU+Metal, CPU only, iOS Xcode/CMake
- Linux: CPU+OpenCL+ThreadPool+Vulkan, CPU+OpenCL+OMP+Vulkan
- Android: AArch32/AArch64 with ThreadPool/OMP + Vulkan
- Windows: x64/x86 CPU

### CI Test Commands (Linux/macOS):
```bash
cd build && ./run_test.out           # Run all unit tests
```

### CI Matrix Coverage:
- **Platforms:** Ubuntu, macOS, Windows, Android (build only)
- **Compilers:** GCC, Clang, MSVC, Apple Clang
- **Backends:** CPU, OpenCL, Vulkan
- **Optimizations:** SSE ON/OFF, AVX512 ON/OFF, BF16
- **Frameworks:** LLM with vision/audio support

## Benchmark Patterns

**Benchmark tool:** `benchmark/benchmark.cpp` and `benchmark/benchmarkExprModels.cpp`
- Measures model inference speed using Express API
- Reports per-layer timing

**Speed tests** (`test/speed/*.cpp`):
- Micro-benchmarks for individual ops
- Use repeated iterations (`TIME 100`) over large tensors (`WIDTH 5001 x HEIGHT 1001`)
- Report wall-clock time via `AUTOTIME` macro or `MNN::Timer`
- Skipped during `runAll()` — must be run explicitly by name

**LLM Benchmark tool:**
```bash
cd build
./llm_bench -m /path/to/MODEL/config.json
```

**Pattern:**
```cpp
#include <MNN/AutoTime.hpp>
// ...
{
    AUTOTIME;  // Prints elapsed time when scope exits
    for (int i = 0; i < TIME; ++i) {
        input0->writeMap<float>();
        output->readMap<float>();
    }
}
```

---

*Testing analysis: 2026-05-27*
