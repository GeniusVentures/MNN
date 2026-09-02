# Phase 9: Real-Weight C++ Encoder Port - Pattern Map

**Mapped:** 2026-08-28
**Files analyzed:** 10 (6 new + 2 modified + 2 conditional)
**Analogs found:** 10 / 10

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `tools/fp4/sgfp4_encode.hpp` | utility (header-pair lib, public API) | transform | `tools/fp4/sgfp4_inject_core.hpp` (header/namespace/guard pattern) + `include/MNN/SGFP4DequantUtils.hpp` (framing constants) | exact |
| `tools/fp4/sgfp4_encode.cpp` | utility (implementation) | transform | `include/MNN/SGFP4DequantUtils.hpp` (decode-side inverse) + `test/op/SGFP4TestUtil.hpp` (write-side helpers) | role-match |
| `tools/fp4/author_real_shape_fixture.py` | utility (golden generator) | transform / file-I/O | `tools/fp4/author_structured_fixture.py` | exact |
| `test/op/SGFP4RealShapeFixtures.h` | model (committed fixture data) | static data | `test/op/SGFP4DequantFixtures.h` + `test/op/SGFP4StructuredFixtures.h` | exact |
| `test/op/SGFP4EncodeTest.cpp` | test | request-response (decode-vs-decode) | `test/op/SGFP4DequantTest.cpp` (round-trip layer) + `test/op/SGFP4InjectTest.cpp` (rtol 1e-4) | exact |
| `test/op/SGFP4VulkanEncodeParityTest.cpp` (or extend `SGFP4VulkanDequantTest.cpp`) | test (integration) | request-response | `test/op/SGFP4VulkanDequantTest.cpp` | exact |
| `tools/fp4/CMakeLists.txt` | config/build | — | `tools/fp4/CMakeLists.txt` (current) | exact |
| `test/CMakeLists.txt` | config/build | — | `test/CMakeLists.txt` (current) | exact |
| *(conditional F1)* `include/MNN/SGFP4DequantUtils.hpp` + `source/backend/cpu/CPUSGFP4Dequant.cpp` + `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp` + `source/shape/ShapeSGFP4Dequant.cpp` | model/controller (decoder crop path) | transform | existing decoders (unchanged-read contract, crop path is the only permitted touch) | exact |
| *(conditional F1)* schema `SGFP4DequantParam` (padded dims) | model (schema) | static data | `schema/default/*.fbs` + Phase 8 `schema/generate.ps1` flow | role-match |

---

## Pattern Assignments

### `tools/fp4/sgfp4_encode.hpp` (utility, transform — new)

**Analog:** `tools/fp4/sgfp4_inject_core.hpp` (header-pair namespace/guard pattern) + `include/MNN/SGFP4DequantUtils.hpp` (constants to reuse).

**Include-guard + namespace pattern** (`sgfp4_inject_core.hpp` lines 22-46):
```cpp
#ifndef TOOLS_FP4_SGFP4_INJECT_CORE_HPP
#define TOOLS_FP4_SGFP4_INJECT_CORE_HPP

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "MNN/SGFP4DequantUtils.hpp"

namespace sgfp4_inject {   // -> use namespace sgfp4_encode
```

**Public one-shot API (from RESEARCH.md Pattern 1 — the D-10 contract, to be finalized against this namespace convention):**
```cpp
// sgfp4_encode.hpp
#include <cstdint>
#include <vector>
namespace sgfp4_encode {
// Encode FP32 row-major [dimO][dimI] weights to an SGFP4 v2 container.
// Mirrors fp4_exporter.py --adaptive (DEFAULT_V2_THRESHOLDS, ternary_delta=0.10).
// Returns empty vector on malformed input (non-finite, non-positive dims).
std::vector<uint8_t> encode(const float* weights, int dimO, int dimI);
}
```

**Framing constants to reuse, never redefine** (`include/MNN/SGFP4DequantUtils.hpp` lines 20-90):
```cpp
constexpr uint32_t kSGFP4Magic = (static_cast<uint32_t>('S')) | (static_cast<uint32_t>('G') << 8)
                               | (static_cast<uint32_t>('F') << 16) | (static_cast<uint32_t>('4') << 24);
constexpr uint8_t  kSGFP4Version = 0x02;
constexpr size_t   kSGFP4FixedHeaderSize   = 16;
constexpr size_t   kSGFP4VersionByteOffset = 4;
constexpr size_t   kSGFP4RecordCountOffset = 5;
constexpr size_t   kSGFP4RecordOffsetTableStart = kSGFP4FixedHeaderSize;
constexpr size_t   kSGFP4RecordOffsetEntrySize  = 4;
constexpr size_t   kSGFP4Alignment              = 16;
constexpr uint32_t kSGFP4LeafHeaderScaleShift = 16;
constexpr uint32_t kSGFP4LeafHeaderBiasMask   = 0xFFF0u;
constexpr uint32_t kSGFP4LeafHeaderFlagsMask  = 0xFu;
constexpr uint32_t kSGFP4LeafHeaderModeBit    = 0x1u;
constexpr int kSGFP4NibblesPerWord = 8;  constexpr int kSGFP4SymbolsPerWord = 16;
constexpr int kSGFP4LayoutEnumMask  = 0x7u;
constexpr size_t kSGFP4SplitMapWords = 3;  constexpr size_t kSGFP4SplitMapBytes = 12;
enum SGFP4UniformLayout : uint32_t {
    kSGFP4LayoutUniform64 = 0, kSGFP4LayoutUniform32 = 1, kSGFP4LayoutUniform16 = 2,
    kSGFP4LayoutUniform8 = 3,  kSGFP4LayoutMixed = 4,    kSGFP4LayoutFull4x4 = 5,
    kSGFP4LayoutEnumCount = 6,
};
```
The encoder writes `magic[4] | version(0x02) | B(u32) | pad0[7]` then `record_offsets[B]` at byte 16 — exactly the layout `dequant_sgfp4_container_cpu` reads (lines 305-330).

---

### `tools/fp4/sgfp4_encode.cpp` (utility, transform — new)

**Analog:** `include/MNN/SGFP4DequantUtils.hpp` (decode-side inverse the encoder mirrors) + `test/op/SGFP4TestUtil.hpp` (the existing write-side LE helpers).

**Write-side LE u32 helper already exists** (`test/op/SGFP4TestUtil.hpp` lines 92-98 — copy this into an anonymous namespace in the .cpp, or reuse the pattern):
```cpp
inline void writeU32Le(std::vector<uint8_t>& out, size_t offset, uint32_t value) {
    out[offset]     = static_cast<uint8_t>(value & 0xFFu);
    out[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    out[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xFFu);
    out[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xFFu);
}
```

**FP16 pack via vendored half** (mirror of `unpack_leaf_header` at `SGFP4DequantUtils.hpp` lines 245-256; the encode-side inverse from RESEARCH.md Pattern 2 / Pitfall 4):
```cpp
#include "MNN/SGFP4DequantUtils.hpp"
#include "half.hpp"
namespace {
inline uint16_t float_to_half_bits(float v) {   // mirrors fp4_exporter._float_to_half
    half_float::half h(std::max(-65504.0f, std::min(65504.0f, v)));
    uint16_t bits; std::memcpy(&bits, &h, sizeof(bits)); return bits;
}
inline uint32_t pack_leaf_header(uint16_t sBits, uint16_t bBits, int mode) {
    return (static_cast<uint32_t>(sBits) << MNN::kSGFP4LeafHeaderScaleShift)
         | (static_cast<uint32_t>(bBits) & ~0xFu)          // HEADER_CLEAR_FLAGS_MASK (Pitfall 4)
         | static_cast<uint32_t>(mode & MNN::kSGFP4LeafHeaderModeBit);
}
}
```

**Round-half-to-even (Pitfall 2):** use `std::rint`/`std::nearbyint`, NOT `std::round`, for code quantization (`codes = np.clip(np.round(...), -8, 7)`).

**Float64 fit internals (Pitfall 3):** `_fit_affine`/`_fit_ternary`/`_combined_gate_error` run in `double`, cast to FP16 only at pack time.

**Alignment padding** (`SGFP4DequantUtils.hpp` line ~91): every record region, block-header block, and leaf payload is padded via `sgfp4_align16(x)` to a 16-byte multiple — the encoder must emit the same.

---

### `tools/fp4/author_real_shape_fixture.py` (utility, transform/file-I/O — new)

**Analog:** `tools/fp4/author_structured_fixture.py` (clone verbatim, swap the weight recipe + emit target).

**gnus-poc import block** (`author_structured_fixture.py` lines 1-30):
```python
#!/usr/bin/env python3
"""Author the <shape> SGFP4 v2 test fixture header. ..."""
import hashlib
import sys
from pathlib import Path

GNUST_POC_ROOT = Path("W:/gnus/GeniusCognitiveSystem/GNUS-NEO-SWARM/gnus-poc")
sys.path.insert(0, str(GNUST_POC_ROOT))

import numpy as np  # noqa: E402
from quantize.fp4_exporter import FP4Exporter  # noqa: E402
```

**Deterministic recipe + seeded RNG** (`author_structured_fixture.py` lines 38-46):
```python
weights = np.full((DIM_O, DIM_I), 0.002, dtype=np.float32)
rng = np.random.default_rng(20260828)   # seeded -> byte-identical regeneration
weights += (rng.random((DIM_O, DIM_I)).astype(np.float32) - 0.5) * 0.0005
```

**Encode + framing validation + sha256 provenance** (`author_structured_fixture.py` lines 74-94):
```python
exporter = FP4Exporter()
binary, stats = exporter.export_weights(weights, "phase9_real_shape", adaptive=True)
if len(binary) < 5 or binary[0:4] != b"SGF4" or binary[4] != 0x02:
    print(f"FATAL: bad framing (magic/version) on {len(binary)} bytes", file=sys.stderr)
    return 1
digest = hashlib.sha256(binary).hexdigest()
```

**D-05 extension:** the generator also emits `{input weights, container bytes, decoded reference}` C arrays (not just container bytes as in the structured fixture) so `SGFP4EncodeTest` can feed the weights into the C++ encoder AND compare C++-encode→decode vs Python-encode→decode. Emit all three arrays per shape with the same no-timestamp/no-unseeded-RNG determinism contract (lines 6-17).

---

### `test/op/SGFP4RealShapeFixtures.h` (model, static data — new)

**Analog:** `test/op/SGFP4DequantFixtures.h` (struct + array + count) + `test/op/SGFP4StructuredFixtures.h` (constexpr dims / sha256 provenance comment).

**Auto-gen header preamble** (`SGFP4StructuredFixtures.h` lines 1-18):
```cpp
// Auto-generated by tools/fp4/author_real_shape_fixture.py.
// DO NOT EDIT BY HAND -- regenerate via:
//   python tools/fp4/author_real_shape_fixture.py test/op/SGFP4RealShapeFixtures.h
#ifndef SGFP4RealShapeFixtures_h
#define SGFP4RealShapeFixtures_h
#include <cstddef>
```

**Fixture struct + array + count** (`SGFP4DequantFixtures.h` lines 11-21, 123-139):
```cpp
namespace sgfp4_fixtures {   // new: namespace sgfp4_real_shape_fixtures
struct Fixture {
    const char* name;
    const unsigned char* container;
    size_t containerSize;
    int dimO;
    int dimI;
    int mode;
    int layout;
    const float* expected;
    size_t expectedCount;
};
static const Fixture kFixtures[] = {
    {"mode0_uniform64", kFixture_mode0_uniform64_data, sizeof(kFixture_mode0_uniform64_data), 64, 64, 0, 0,
     kFixture_mode0_uniform64_expected, sizeof(kFixture_mode0_uniform64_expected) / sizeof(float)},
    // ...
};
static const size_t kFixtureCount = sizeof(kFixtures) / sizeof(kFixtures[0]);
} // namespace sgfp4_fixtures
```
**D-05 addition:** extend the struct with `const float* inputWeights;` (the source FP32 plane) so the test can call the C++ encoder on it. Real-shape fixtures carry non-64-multiple `dimO`/`dimI` (e.g. `100, 36`, `250, 128`, `37, 91`, `64, 36`, `5, 5`, `1, 1`).

---

### `test/op/SGFP4EncodeTest.cpp` (test, request-response — new)

**Analog:** `test/op/SGFP4DequantTest.cpp` (suite skeleton + round-trip layer) + `test/op/SGFP4InjectTest.cpp` (the rtol 1e-4 decode-vs-decode tolerance bar, D-04).

**Suite skeleton + includes** (`SGFP4DequantTest.cpp` lines 1-55):
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
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "SGFP4TestUtil.hpp"
#include "SGFP4RealShapeFixtures.h"
using namespace MNN::Express;

namespace {
constexpr float kFixtureRelativeTolerance = 1e-4f;   // D-04 decode-vs-decode bar
} // namespace

class SGFP4EncodeTest : public MNNTestCase {
public:
    SGFP4EncodeTest()  = default;
    virtual ~SGFP4EncodeTest() = default;
    virtual bool run(int precision) { /* ... */ return true; }
};
```

**Decode-vs-decode assertion pattern** (`SGFP4DequantTest.cpp` lines 65-85 — the exact `checkVectorByRelativeError` call the planner copies):
```cpp
bool testFixtureRoundTrip() {
    for (size_t i = 0; i < sgfp4_fixtures::kFixtureCount; ++i) {
        const auto& fixture = sgfp4_fixtures::kFixtures[i];
        std::vector<float> out(fixture.expectedCount, 0.0f);
        bool ok = MNN::dequant_sgfp4_container_cpu(fixture.container, fixture.containerSize,
                                                    out.data(), fixture.expectedCount);
        if (!ok) { MNN_ERROR("... decode returned false\n"); return false; }
        if (!checkVectorByRelativeError<float>(out.data(), fixture.expected,
                                                static_cast<int>(fixture.expectedCount),
                                                kFixtureRelativeTolerance)) {
            MNN_ERROR("... round-trip mismatch\n"); return false;
        }
    }
    return true;
}
```
Phase 9 re-shapes this: (1) call `sgfp4_encode::encode(fixture.inputWeights, dimO, dimI)`, (2) decode the C++ container via the oracle, (3) compare against the Python `expected` (not byte-vs-byte) at rtol 1e-4.

**Suite registration macro** (all SGFP4 tests; `SGFP4DequantTest.cpp` line 565):
```cpp
MNNTestSuiteRegister(SGFP4EncodeTest, "op/sgfp4/encode");
```

**`#endif // MNN_SUPPORT_TRANSFORMER_FUSE`** closes every SGFP4 test file — the encoder's `#ifdef` gate must match.

---

### `test/op/SGFP4VulkanEncodeParityTest.cpp` (test, integration — new or extend)

**Analog:** `test/op/SGFP4VulkanDequantTest.cpp` (the D-08 Vulkan-leg parity pattern).

**Op-level sidecar + Vulkan config construction** (`SGFP4VulkanDequantTest.cpp` lines 42-72):
```cpp
std::shared_ptr<MNN::OpT> op(new MNN::OpT);
op->type      = MNN::OpType_SGFP4Dequant;
op->main.type = MNN::OpParameter_SGFP4DequantParam;
auto* param   = new MNN::SGFP4DequantParamT;
param->magic   = MNN::kSGFP4Magic;
param->external = {0, static_cast<int64_t>(fixture.containerSize)};
param->dims     = {fixture.dimO, fixture.dimI};
op->main.value  = param;
op->externalPath = sidecarPath;
auto output = Variable::create(Expr::create(op.get(), {}));
auto buffer = Variable::save({output});
MNN::ScheduleConfig config; config.type = MNN_FORWARD_VULKAN;
```

**Graceful no-GPU skip** (`SGFP4VulkanDequantTest.cpp` lines 121-127):
```cpp
auto vulkanCreator = MNN::MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN);
if (nullptr == vulkanCreator) {
    MNN_PRINT("Vulkan backend not available — skipping SGFP4 Vulkan parity test\n");
    return true;
}
```

**GPU/CPU parity check** (`SGFP4VulkanDequantTest.cpp` lines 96-100):
```cpp
if (!checkVectorByRelativeError<float>(outPtr, cpuRef, static_cast<int>(outCount), rtol)) { ... }
```
For D-08, `cpuRef` is the Python-encoded→decoded golden; the C++-encoded container is the sidecar. Register via `MNNTestSuiteRegister(SGFP4VulkanEncodeParityTest, "op/sgfp4/vulkan_encode_parity");`.

---

### `tools/fp4/CMakeLists.txt` (config/build — modify)

**Analog:** `tools/fp4/CMakeLists.txt` (current, 19 lines — the glob double-compiles, RESEARCH Q3/Anti-pattern).

**Current wiring to modify** (full file):
```cmake
set(MNN_SGFP4_TOOLS "")
file(GLOB SGFP4FILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
add_executable(sgfp4_inject.out ${SGFP4FILES})
list(APPEND MNN_SGFP4_TOOLS sgfp4_inject.out)

foreach(TARGET ${MNN_SGFP4_TOOLS})
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
**Required change (RESEARCH Q3 / Anti-pattern):** add a dedicated static lib and narrow the glob so `sgfp4_encode.cpp` compiles once:
```cmake
add_library(sgfp4_encode STATIC ${CMAKE_CURRENT_LIST_DIR}/sgfp4_encode.cpp)
target_include_directories(sgfp4_encode PUBLIC ${CMAKE_CURRENT_LIST_DIR})
# narrow the existing glob to exclude sgfp4_encode.cpp, then:
target_link_libraries(sgfp4_inject.out sgfp4_encode)
```

---

### `test/CMakeLists.txt` (config/build — modify)

**Analog:** `test/CMakeLists.txt` (current, lines 1-45).

**Current wiring** (`test/CMakeLists.txt` lines 1-24):
```cmake
set(TEST_DEPS ${MNN_DEPS})
if(APPLE)
  file(GLOB_RECURSE Files ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.mm)
else()
  file(GLOB_RECURSE Files ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
endif()
add_executable(run_test.out ${Files})
target_link_libraries(run_test.out ${MNN_DEPS})
...
target_include_directories(run_test.out PRIVATE ${CMAKE_CURRENT_LIST_DIR}/)
```
**Required change (RESEARCH Q3):** link the encoder lib into `run_test.out` so `SGFP4EncodeTest.cpp` (a `test/` glob member) can call `sgfp4_encode::encode`:
```cmake
target_link_libraries(run_test.out sgfp4_encode)
```
Because `sgfp4_encode` is defined in `tools/fp4/CMakeLists.txt`, that file must be `include()`d from the test context (or the lib target hoisted into a location reachable from both — e.g. add a `if(MNN_BUILD_SGFP4_TOOLS OR MNN_BUILD_TEST)` gate in `tools/fp4/CMakeLists.txt` and ensure `test/` adds the subdirectory first). Planner must resolve the exact add_subdirectory ordering.

---

### *(conditional F1)* Decoder padded-crop path (model/controller, transform — modify only if D-07/D-08 land)

**Analog:** the existing decoders themselves (read-only contract; only the crop path may be touched).

**Exact-match contract that must be relaxed** (`include/MNN/SGFP4DequantUtils.hpp` lines 452-464 — Finding F1 evidence):
```cpp
if (static_cast<size_t>(elementCount) > outElementCount - outCursor) { return false; }
// ...
return outCursor == outElementCount;   // <- rejects padded-plane decode
```

**CPU decode call sites** (`source/backend/cpu/CPUSGFP4Dequant.cpp`):
- `onResize` eager oracle (lines 59-67): `dequant_sgfp4_container_cpu(..., outputs[0]->elementSize())`
- `onExecute` (lines 91-103): `dequant_sgfp4_container_cpu(mContainer.data(), mContainer.size(), dest, elementCount)`

**Vulkan element-count handling** (`source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp`):
- `SGFP4DequantConst { uint32_t outElementCount; uint32_t containerBytes; }` (lines 24-27)
- `onEncode`: `uint32_t elementCount = static_cast<uint32_t>(output->elementSize());` → `vkCmdDispatch(UP_DIV(elementCount, kSgfp4WorkgroupSize), ...)` (lines 78-110). Shader-side `idx >= outElementCount` guard in `glsl/sgfp4_dequant.comp`.

**Shape inference** (`source/shape/ShapeSGFP4Dequant.cpp` lines 17-29): output dims come from `param->dims()`; a padded-crop path either (a) extends `SGFP4DequantParam` with optional `paddedDims` (schema regen via `schema/generate.ps1`, Phase 8 precedent), or (b) decodes the padded plane to scratch and crops `out[r*dimI+c] = padded[r*paddedCols+c]` (Pitfall 5: NOT a flat prefix). Flag the boundary contradiction (phase excludes "any changes to the decoders") to the planner before implementing.

---

## Shared Patterns

### Framing constants — single source of truth
**Source:** `include/MNN/SGFP4DequantUtils.hpp` (lines 20-90)
**Apply to:** `sgfp4_encode.hpp`, `sgfp4_encode.cpp`, and any F1 decoder touch.
The encoder MUST reuse `MNN::kSGFP4Magic`, `MNN::kSGFP4Version`, `MNN::kSGFP4Alignment`, `MNN::sgfp4_align16`, layout enums, leaf-header masks, split-map constants — never redefine (the W-1 offset-convention bug class).

### Decode-vs-decode tolerance — rtol 1e-4 (D-04)
**Source:** `test/op/SGFP4DequantTest.cpp` line 29, `test/op/SGFP4InjectTest.cpp` line ~41, `test/op/SGFP4VulkanDequantTest.cpp` line 37
**Apply to:** `SGFP4EncodeTest.cpp`, Vulkan parity leg.
```cpp
constexpr float kFixtureRelativeTolerance = 1e-4f;
```
Assertion via `checkVectorByRelativeError<float>(got, ref, count, rtol)` — never byte-exact container comparison.

### Test registration — `op/sgfp4/*` family
**Source:** all `test/op/SGFP4*Test.cpp` bottom lines
**Apply to:** `SGFP4EncodeTest.cpp` (+ Vulkan parity).
```cpp
MNNTestSuiteRegister(SGFP4EncodeTest, "op/sgfp4/encode");
```
Filtered runs: `.build/run_test.out op/sgfp4/encode`; full family: `.build/run_test.out op/sgfp4`.

### Error handling — `false`/`ErrorCode` return, no exceptions
**Source:** `dequant_sgfp4_container_cpu` (returns `false` on malformed), `CPUSGFP4Dequant::onResize/onExecute` (returns `INVALID_VALUE`/`NOT_SUPPORT`)
**Apply to:** `sgfp4_encode.cpp` (return empty vector on malformed input per D-10) and any decoder change.
RTTI/exceptions are disabled; encoder validates finite FP32 + positive dims (ASVS V5) and bounds dims at entry (integer-overflow guard).

### Golden-fixture determinism contract
**Source:** `tools/fp4/author_structured_fixture.py` lines 6-17
**Apply to:** `author_real_shape_fixture.py`.
No timestamp in provenance, no unseeded RNG; sha256 provenance block; regeneration must be byte-identical.

---

## No Analog Found

No close in-tree match exists for the **scipy Gaussian-filter Laplacian weighting** (RESEARCH Pitfall 1) — the C++ encoder must hand-roll a separable 2D Gaussian with kernel radius `int(4*sigma + 0.5)`, `mode='reflect'`, sum-to-1 normalization. This is the one genuinely hand-rolled numeric piece; validate C++ vs scipy on random patches BEFORE wiring into split decisions (D-04 tolerates residual flips).

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `sgfp4_encode.cpp` (Laplacian-weighted-error module only) | utility | transform | scipy is Python-only; no in-tree C++ Gaussian filter exists — necessary hand-roll per RESEARCH "Don't Hand-Roll" table |

---

## Metadata

**Analog search scope:** `tools/fp4/`, `test/op/SGFP4*`, `test/CMakeLists.txt`, `include/MNN/SGFP4DequantUtils.hpp`, `source/backend/cpu/`, `source/backend/vulkan/buffer/execution/`, `source/shape/`
**Files scanned:** 12 analog files
**Pattern extraction date:** 2026-08-28
