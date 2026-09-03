# Phase 5: Injection Core — Artifact Construction & Graph Splicing - Pattern Map

**Mapped:** 2026-08-26
**Files analyzed:** 5 (4 new, 1 modified)
**Analogs found:** 4 / 5 (only `sha256.h` lacks a functional in-repo analog)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `tools/fp4/sgfp4_inject.cpp` | controller (CLI tool main) | file-I/O + transform (graph splicing) | `tools/quantization/calibration.cpp` (surgery) + `tools/quantization/quantized.cpp` (main) + `test/op/SGFP4DequantTest.cpp:384-401` (op build) | exact (multi-part) |
| `tools/fp4/sha256.h` | utility (vendored single-header) | transform (hash) | `3rd_party/half/half.hpp` (vendored-header convention only) | partial (no SHA-256 exists in repo) |
| `tools/fp4/CMakeLists.txt` | config | n/a | `tools/quantization/CMakeLists.txt` | exact |
| `test/op/SGFP4InjectTest.cpp` | test | file-I/O + transform (build→save→reload→verify) | `test/op/SGFP4DequantTest.cpp` | exact |
| `CMakeLists.txt` (root, modified) | config | n/a | `MNN_BUILD_QUANTOOLS` option + `include(...)` block | exact |

> Optional (agent discretion, not required): a helper header `tools/fp4/sgfp4_inject.hpp`. If created, mirror the `.hpp`/`.cpp` split of `tools/quantization/calibration.hpp` + `calibration.cpp` (declaration header + implementation in a separate TU).

---

## Pattern Assignments

### `tools/fp4/sgfp4_inject.cpp` (controller, file-I/O + transform)

Three separate in-repo precedents combine here. Copy each.

**Analog A — CLI entry + arg parsing:** `tools/quantization/quantized.cpp` + `tools/quantization/calibration.cpp:1715-1731`

`tools/quantization/quantized.cpp:8-16` (thin main delegating to a `*_main` fn):
```cpp
#include <fstream>
#include <sstream>
#include <string>
#include "calibration.hpp"

int main(int argc, const char* argv[]) {
    return quant_main(argc, argv);
}
```

`tools/quantization/calibration.cpp:1715-1724` (arg-count guard + positional parsing):
```cpp
int quant_main(int argc, const char* argv[]) {
    if (argc < 4) {
        DLOG(INFO) << "Usage: ./quantized.out src.mnn dst.mnn preTreatConfig.json\n";
        return 0;
    }
    const char* modelFile      = argv[1];
    const char* preTreatConfig = argv[3];
    const char* dstFile        = argv[2];
    DLOG(INFO) << ">>> modelFile: " << modelFile;
```
> **Apply to Phase 5:** replace positional args with `--model`, `--niche-dir` (repeatable), `--output` (D-10). Keep the same structure: `int main(int argc, const char* argv[])` → parse → delegate to a worker function returning nonzero on error. Use `MNN_PRINT`/`MNN_ERROR` (not `DLOG`) for user-facing diagnostics; no exceptions (project convention).

**Analog B — Graph load / surgery / save:** `tools/quantization/calibration.cpp:1320-1330, 1419`

```cpp
auto varMap = Variable::loadMap(_originalModelFile.c_str());
if (varMap.empty()) {
    MNN_ERROR("Can not load model %s\n", _originalModelFile.c_str());
    return;
}

auto inputOutputs = Variable::getInputAndOutput(varMap);
auto varInputs       = Variable::mapToSequence(inputOutputs.first);
auto varOutputs      = Variable::mapToSequence(inputOutputs.second);
```
```cpp
Variable::save(predicts, _destModelFile.c_str());   // direct-to-file overload
```
> **Apply to Phase 5:** `loadMap` → `getInputAndOutput` → enumerate weight VARP by `getInfo()->dim` for D-02 shape match → build dequant node → `Variable::replace(weightVar, dequantVar)` → recompute `varOutputs` AFTER rewiring → `Variable::save(varOutputs, outMnnPath.c_str())` (D-06/D-07/SGINJ-04).

**Rewiring primitive — the correct API is `Variable::replace` (NOT `replaceInput`):** `include/MNN/expr/Expr.hpp:168`
```cpp
// include/MNN/expr/Expr.hpp:168 (Variable class, static)
static void replace(VARP dst, VARP src);
```
> CONTEXT.md D-06 names `Variable::replaceInput` — that symbol does **not exist** (see RESEARCH.md Pitfall 1). `replace(dst, src)` mutates the const Expr in place; consumers keep their `mTo` back-refs. After `replace`, `weightVar` (not `dequantVar`) is the live node — save only the recomputed outputs (RESEARCH.md Pitfall 4).

**Analog C — SGFP4Dequant op construction:** `test/op/SGFP4DequantTest.cpp:384-401`

```cpp
std::shared_ptr<MNN::OpT> op(new MNN::OpT);
op->type = MNN::OpType_SGFP4Dequant;
op->main.type = MNN::OpParameter_SGFP4DequantParam;
auto* param = new MNN::SGFP4DequantParamT;
param->magic = MNN::kSGFP4Magic;
param->external = {0, static_cast<int64_t>(fixture.containerSize)};
param->dims = {fixture.dimO, fixture.dimI};
op->main.value = param;
// ... externalPath must be set directly on the Op — this op is NOT one of
// the types OpCommonUtils::createExecutionWithExternal rewrites ...
op->externalPath = sidecarPath;

auto output = Variable::create(Expr::create(op.get(), {}));   // 0-input source op
```
> **Apply to Phase 5 (SGINJ-02):** `param->external = {offset, size}` with the per-container byte offset in the merged sidecar (not hardcoded 0); `param->dims = {dimO, dimI}` from manifest `fp4_binary.stats.shape` (D-05); `op->externalPath = sidecarPath` literal (Pitfall 2 — without it, `CPUSGFP4Dequant::onResize` returns `NOT_SUPPORT`). Set `dequantVar->setName(weightVar->name() + "_sgfp4")` after `Variable::create` (D-08; `Variable::setName` is at `Expr.hpp:118`).

**Error handling pattern** (whole tool): return codes + `MNN_ERROR` logs, no exceptions.
```cpp
// Source: tools/quantization/calibration.cpp:1321-1324
if (varMap.empty()) {
    MNN_ERROR("Can not load model %s\n", _originalModelFile.c_str());
    return;          // (or: return 1; in the tool's main)
}
```
> D-02/D-03/SGINJ-01 hard errors = `MNN_ERROR("...")` + `return 1;`. D-12 verify failure = `MNN_ERROR("decode mismatch")` + nonzero exit.

---

### `tools/fp4/CMakeLists.txt` (config)

**Analog:** `tools/quantization/CMakeLists.txt` (entire file, 19 lines)

```cmake
set(MNN_QUAN_TOOLS "")

file(GLOB QUANFILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
add_executable(quantized.out ${QUANFILES})
list(APPEND MNN_QUAN_TOOLS quantized.out)

foreach(TARGET ${MNN_QUAN_TOOLS})
    target_link_libraries(${TARGET} ${MNN_DEPS})
    if (MSVC)
        target_compile_definitions(${TARGET} PRIVATE "_CRT_SECURE_NO_WARNINGS")
        if (NOT MNN_BUILD_SHARED_LIBS)
            foreach (DEPEND ${MNN_DEPS})
                target_link_options(${TARGET} PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:${DEPEND}>)
            endforeach ()
        endif()
    endif()
endforeach()
```
> **Apply to Phase 5:** name the tool `sgfp4_inject.out`, glob `tools/fp4/*.cpp *.hpp` (this will pick up `sgfp4_inject.cpp` and any helper), link `${MNN_DEPS}` (gives Express + core MNN). Note: `encode_sgfp4.py`/`quantize_fp4.py` in the same dir are `.py`, unaffected by the `*.cpp` glob. The vendored `sha256.h` is a header — globbing `*.hpp` won't pull a `.h`, so either name it `sha256.hpp` or add `*.h` to the glob (see no-analog note below).

---

### `test/op/SGFP4InjectTest.cpp` (test)

**Analog:** `test/op/SGFP4DequantTest.cpp` — full structural template.

**Includes + guard** (lines 9-27):
```cpp
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include "half.hpp"
#include "MNN_generated.h"
#include "MNN/SGFP4DequantUtils.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/expr/Executor.hpp"
#include "MNN/expr/Module.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "SGFP4DequantFixtures.h"

using namespace MNN::Express;
```
> The `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE` gate is mandatory (RESEARCH.md A5): the SGFP4 tests only execute when the flag is ON; the build must pass it.

**Class + run signature + registration** (lines 42-46, 565):
```cpp
class SGFP4DequantTest : public MNNTestCase {
public:
    SGFP4DequantTest()  = default;
    virtual ~SGFP4DequantTest() = default;

    virtual bool run(int precision) {
        ...
        return true;
    }
};
MNNTestSuiteRegister(SGFP4DequantTest, "op/sgfp4/uniform_decode");
```
> **Apply to Phase 5:** register as `op/sgfp4/inject` (SGINJ-02/03/04 end-to-end) and `op/sgfp4/inject_v1_reject` (SGINJ-01). `run(int precision)` returns `bool`; use `checkVectorByRelativeError<float>` for the oracle comparison (same as `SGFP4DequantTest.cpp:442-446`).

**Numeric comparison pattern** (`test/op/SGFP4DequantTest.cpp:436-446`):
```cpp
size_t outCount = static_cast<size_t>(outInfo->size);
if (outCount != fixture.expectedCount) {
    MNN_ERROR("SGFP4DequantTest: op-level output element count %zu != expected %zu\n", outCount,
               fixture.expectedCount);
    return false;
}
if (!checkVectorByRelativeError<float>(outPtr, fixture.expected, static_cast<int>(outCount),
                                        kFixtureRelativeTolerance)) {
    MNN_ERROR("SGFP4DequantTest: op-level decode mismatch\n");
    return false;
}
```
> The fixture data lives in `test/op/SGFP4DequantFixtures.h` (`sgfp4_fixtures::kFixtures[]`, `name`/`container`/`containerSize`/`dimO`/`dimI`/`expected`/`expectedCount`). For the inject test, embed or point at the demo container (`demo.sgfp4`, 132,368 bytes, 512×512 — RESEARCH.md Pitfall 6: uniform-only, so it proves the uniform path only).

---

### `CMakeLists.txt` (root, modified) (config)

**Analog:** the existing `MNN_BUILD_QUANTOOLS` option + tool-include block.

Option declaration (`CMakeLists.txt:49`):
```cmake
option(MNN_BUILD_QUANTOOLS "Build Quantized Tools or not" OFF)
```

Tool include block (`CMakeLists.txt:966-968`):
```cmake
IF(MNN_BUILD_QUANTOOLS)
include(${CMAKE_CURRENT_LIST_DIR}/tools/quantization/CMakeLists.txt)
ENDIF()
```
> **Apply to Phase 5:** add `option(MNN_BUILD_SGFP4_TOOLS "Build SGFP4 injection tools or not" OFF)` near line 49, and add an `IF(MNN_BUILD_SGFP4_TOOLS) include(${CMAKE_CURRENT_LIST_DIR}/tools/fp4/CMakeLists.txt) ENDIF()` block adjacent to the quantization include. Follow the `include(...)` style (not `add_subdirectory`).

---

## Shared Patterns

### SGFP4Dequant op construction (magic + external + dims + literal externalPath)
**Source:** `test/op/SGFP4DequantTest.cpp:384-401` (also `test/op/SGFP4VulkanDequantTest.cpp:47-56`)
**Apply to:** `tools/fp4/sgfp4_inject.cpp`, `test/op/SGFP4InjectTest.cpp`
```cpp
std::shared_ptr<MNN::OpT> op(new MNN::OpT);
op->type = MNN::OpType_SGFP4Dequant;
op->main.type = MNN::OpParameter_SGFP4DequantParam;
auto* param = new MNN::SGFP4DequantParamT;
param->magic = MNN::kSGFP4Magic;
param->external = {offset, static_cast<int64_t>(size)};
param->dims = {dimO, dimI};
op->main.value = param;
op->externalPath = sidecarPath;   // literal — REQUIRED (Pitfall 2)
auto output = Variable::create(Expr::create(op.get(), {}));
```
Struct layout (`schema/current/CaffeOp_generated.h:1441-1448`): `SGFP4DequantParamT{uint32_t magic; std::vector<int64_t> external; std::vector<int32_t> dims;}`.

### Reload via Module::load — setExternalFile MUST precede load
**Source:** `test/op/SGFP4DequantTest.cpp:412-420`
**Apply to:** `tools/fp4/sgfp4_inject.cpp` (D-12 verify), `test/op/SGFP4InjectTest.cpp`
```cpp
MNN::ScheduleConfig config;
config.type = MNN_FORWARD_CPU;
std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
rtmgr->setExternalFile(sidecarPath);                      // BEFORE Module::load (Pitfall 5)

std::shared_ptr<Module> m(Module::load({}, {}, reinterpret_cast<const uint8_t*>(buffer.data()),
                                        buffer.size(), rtmgr));
if (nullptr == m) { MNN_ERROR("... Module::load returned null\n"); return false; }
auto outputs = m->onForward({});
```
> For the tool's D-12, load from file via the file-path `Module::load({}, {}, outMnnPath.c_str(), rtmgr)` overload after `Variable::save(varOutputs, outMnnPath.c_str())`.

### Version gate + container validation (SGINJ-01)
**Source:** `include/MNN/SGFP4DequantUtils.hpp:29-39` (constants) + `source/backend/cpu/CPUSGFP4Dequant.cpp:45-52` (consume-side gate)
**Apply to:** `tools/fp4/sgfp4_inject.cpp`
```cpp
// include/MNN/SGFP4DequantUtils.hpp:29-39
constexpr uint32_t kSGFP4Magic = ('S') | ('G' << 8) | ('F' << 16) | ('4' << 24);
constexpr uint8_t kSGFP4Version = 0x02;
constexpr size_t kSGFP4FixedHeaderSize = 16;
constexpr size_t kSGFP4VersionByteOffset = 4;
```
Probe = `n >= kSGFP4FixedHeaderSize && sgfp4_read_u32_le(p) == kSGFP4Magic && p[kSGFP4VersionByteOffset] == kSGFP4Version`. v1 fixed-payload files have no `SGF4` magic, so the probe rejects them (RESEARCH.md: don't trust manifest `fp4_binary.format`).
```cpp
// source/backend/cpu/CPUSGFP4Dequant.cpp:45-48 (consume side — the gate the artifact must pass)
if (!USE_EXTERNAL_DATA(param) || nullptr == mOp->externalPath()) {
    return NOT_SUPPORT;
}
auto external = param->external()->data();
int64_t offset = external[0];
int64_t size   = external[1];
```

### Error handling (no exceptions; error codes + MNN_ERROR/MNN_PRINT)
**Source:** `tools/quantization/calibration.cpp:1321-1324`, `test/op/SGFP4DequantTest.cpp` throughout
**Apply to:** all new C++ files
- Tool: `MNN_ERROR("...")` + `return 1;` for hard errors (D-02/D-03/SGINJ-01/D-12); `MNN_PRINT` for progress.
- Test: `MNN_ERROR("...")` + `return false;` from `run(int precision)`.
- No `throw`/`catch` (RTTI and exceptions disabled).

### Manifest JSON parsing
**Source:** vendored `3rd_party/rapidjson` (D-09 locked). No in-repo analog file in this repo's tools (converter uses protobuf); use rapidjson DOM API directly.
**Apply to:** `tools/fp4/sgfp4_inject.cpp`
- Read `fp4_binary.sha256` (D-03), `fp4_binary.stats.shape` (D-01/D-05), `fp4_binary.path` (basename only — Pitfall 3).
- Treat manifest fields as untrusted: validate `stats.shape` is exactly 2 positive ints before use.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `tools/fp4/sha256.h` | utility | transform (hash) | No SHA-256 implementation exists anywhere in the MNN repo (grep for `sha256/SHA-256/BCrypt` hits only `.build` MSVC `BCRYPT.H` tlog paths). Planner should pin a specific public-domain single-header (e.g. RFC 6234 / WjCryptLib style) and review its license header before committing. |

**Vendored-header convention to copy** (partial match): `3rd_party/half/half.hpp` — license header at top (lines 3-14), include guard `#ifndef HALF_HALF_HPP` (line 22), single self-contained namespace (line 210). If the SHA-256 file is named `sha256.h`, it will NOT be picked up by `tools/fp4/CMakeLists.txt`'s `file(GLOB ... *.cpp *.hpp)` — either name it `sha256.hpp` or add `*.h` to the glob.

---

## Metadata

**Analog search scope:**
- `tools/quantization/` (CLI tool, CMake, surgery, serialization precedent)
- `tools/fp4/` (existing Python encoder dir — confirms `.cpp`/`CMakeLists.txt` are new)
- `test/op/SGFP4DequantTest.cpp`, `test/op/SGFP4VulkanDequantTest.cpp` (op-construction + reload + test registration)
- `include/MNN/expr/Expr.hpp` (Variable/Expr API signatures)
- `include/MNN/SGFP4DequantUtils.hpp` (format constants + oracle)
- `schema/current/CaffeOp_generated.h` (SGFP4DequantParamT layout)
- `source/backend/cpu/CPUSGFP4Dequant.cpp`, `source/core/OpCommonUtils.cpp` (consume side + externalPath gotcha)
- `test/CMakeLists.txt` (glob-recursive test registration)
- root `CMakeLists.txt` (tool option gating)
- `3rd_party/half/half.hpp` (vendored-header convention)

**Files scanned:** 14
**Pattern extraction date:** 2026-08-26
