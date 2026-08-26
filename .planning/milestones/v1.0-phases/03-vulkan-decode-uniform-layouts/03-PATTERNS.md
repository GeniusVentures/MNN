# Phase 3: Vulkan Decode — Uniform Layouts - Pattern Map

**Mapped:** 2026-08-24
**Files analyzed:** 8 (4 new source files, 1 config edit, 3 regenerated artifacts)
**Analogs found:** 8 / 8 (7 exact/close, 1 partial — the GLSL framing-walk itself has no in-tree analog)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.hpp` | execution header (class decl) | transform | `source/backend/vulkan/buffer/execution/VulkanFP4Dequant.hpp` | exact |
| `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp` | execution impl (Vulkan op + host file-I/O) | file-I/O → GPU transform | `VulkanFP4Dequant.cpp` (structure) + `source/backend/cpu/CPUSGFP4Dequant.cpp` (host sidecar load) | exact (both halves) |
| `source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp` | GLSL compute shader | transform (decode) | `source/backend/vulkan/buffer/execution/glsl/fp4_dequant.comp` | partial — GLSL conventions exact; framing walk is NEW (no in-tree analog) |
| `source/backend/vulkan/buffer/execution/glsl/macro.json` | build config (shader variant registry) | config | existing entries in same file (`blitregion.comp` entry, lines 8-9) | exact |
| `source/backend/vulkan/buffer/compiler/AllShader.cpp` | generated artifact | generated | n/a — REGENERATED via `makeshader.py`, never hand-edited | n/a |
| `source/backend/vulkan/buffer/shaders/AllShader.h` | generated artifact | generated | n/a — REGENERATED | n/a |
| `source/backend/vulkan/buffer/compiler/VulkanShaderMap.cpp` | generated artifact | generated | n/a — REGENERATED | n/a |
| `test/op/SGFP4VulkanDequantTest.cpp` | test (dual-backend parity) | file-I/O + request-response (module onForward) | `test/op/VulkanFP4DequantTest.cpp` (harness) + `test/op/SGFP4DequantTest.cpp::runSgfp4Module` (0-input module) | exact |

**No CMake edits anywhere** — `source/backend/vulkan/CMakeLists.txt` and `test/CMakeLists.txt` both use `GLOB_RECURSE` (verified in RESEARCH.md).

## Pattern Assignments

### `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.hpp` (execution header)

**Analog:** `VulkanFP4Dequant.hpp` (whole file, 33 lines — clone structure verbatim)

```cpp
// Source: source/backend/vulkan/buffer/execution/VulkanFP4Dequant.hpp lines 13-31
#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanFP4Dequant : public VulkanBasicExecution {
public:
    VulkanFP4Dequant(Backend* bn, bool useFP32Output);
    virtual ~VulkanFP4Dequant();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mDequantPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    bool mUseFP32Output;
};
} // namespace MNN
```

SGFP4 additions to the private section: `std::shared_ptr<VulkanBuffer> mContainerBuffer;` (SSBO with raw container bytes per D-01) and `std::vector<uint8_t> mContainer;` (host copy for upload ctor arg). Header guard `#ifndef VulkanSGFP4Dequant_hpp` per convention.

**Critical timing constraint:** `VulkanBasicExecution` has NO `onResize` hook — only `onEncode`. D-01's sidecar read + D-05's pre-validation must run in the **creator's `onCreate` → constructor** (the `MNN::Op*` with `externalPath` is available there). See the `.cpp` pattern below.

---

### `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp` (execution impl)

**Analog 1:** `VulkanFP4Dequant.cpp` — class structure, pipeline selection, dispatch. Three excerpts:

**Const-buffer struct + constructor pipeline selection** (lines 14-49):
```cpp
// Source: VulkanFP4Dequant.cpp lines 14-17
struct FP4DequantConst {
    uint32_t elementCount;
    uint32_t srcBytes;
};

// lines 19-23: ctor — const UBO + descriptor types
VulkanFP4Dequant::VulkanFP4Dequant(Backend* bn, bool useFP32Output) : VulkanBasicExecution(bn) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    mConstBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, sizeof(FP4DequantConst), nullptr,
                                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,   // binding 0
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,   // binding 1
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER    // binding 2
    };

    // lines 37-48: D-06 FP16/FP32 variant selection — clone exactly, rename shaders
    std::string shaderName;
    if (mUseFP32Output) {
        shaderName = "glsl_fp4_dequant_comp";          // SGFP4: "glsl_sgfp4_dequant_comp"
    } else if (vkBn->useFP16()) {
        shaderName = "glsl_fp4_dequant_FP16_comp";     // SGFP4: "glsl_sgfp4_dequant_FP16_comp"
    } else {
        shaderName = "glsl_fp4_dequant_comp";
    }
    mDequantPipeline = vkBn->getPipeline(shaderName, types);
    mDescriptorSet.reset(mDequantPipeline->createSet());
```
SGFP4 difference: binding 0 is NOT an input tensor — it is the container SSBO (`mContainerBuffer`). Binding 1 = output tensor buffer, binding 2 = const. Const struct fields: `containerWords` (or byte size) + `outElementCount`.

**onEncode: const write, descriptor writes, dispatch, barrier** (lines 54-90):
```cpp
// Source: VulkanFP4Dequant.cpp lines 61-87
{
    auto dequantConst = reinterpret_cast<FP4DequantConst*>(mConstBuffer->map());
    dequantConst->elementCount = elementCount;
    mConstBuffer->unmap();
}
auto outputBuffer = extra->getTensorBuffer(output);
auto outputSize   = extra->getTensorSize(output);
mDescriptorSet->writeBuffer(mContainerBuffer->buffer(), 0, mContainerBuffer->size(), 0); // container SSBO
mDescriptorSet->writeBuffer(outputBuffer.first->buffer(), 1, outputSize, outputBuffer.second);
mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());
mDequantPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
vkCmdDispatch(cmdBuffer->get(), UP_DIV(elementCount, 256), 1, 1);
cmdBuffer->barrierSource(outputBuffer.first->buffer(), outputBuffer.second, outputSize);
return NO_ERROR;
```
SGFP4: NO input tensor reads (0-input Const-like op); elementCount = `outputs[0]->elementSize()`.

**Creator + static registration** (lines 93-109):
```cpp
// Source: VulkanFP4Dequant.cpp lines 93-109
class VulkanFP4DequantCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs,
                                            const std::vector<Tensor*>& outputs,
                                            const MNN::Op* op,
                                            Backend* backend) const override {
        bool useFP32Output = false;
        return new VulkanFP4Dequant(backend, useFP32Output);
    }
};
static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Dequantize, new VulkanFP4DequantCreator);
    return true;
}();
```
SGFP4 changes: register on **`OpType_SGFP4Dequant`**, and `onCreate` reads the op BEFORE constructing: `op->main_as_SGFP4DequantParam()` + `op->externalPath()` — this is where D-01 sidecar load + D-05 pre-validation happen (constructor receives the loaded bytes / returns nullptr-equivalent error path before any pipeline is built).

**Analog 2:** `CPUSGFP4Dequant.cpp` — host-side external-sidecar loading. Clone lines 21-96 into the Vulkan creator/constructor path:

**File-size probe (T-01-04 DoS bound)** (lines 21-37, copy the helper verbatim):
```cpp
// Source: CPUSGFP4Dequant.cpp lines 25-36
bool queryFileSize(const std::string& path, size_t& outSize) {
    std::ifstream probe(path, std::ios::binary | std::ios::ate);
    if (!probe.is_open()) {
        return false;
    }
    auto pos = probe.tellg();
    if (pos < 0) {
        return false;
    }
    outSize = static_cast<size_t>(pos);
    return true;
}
```

**Descriptor gate + bounded read** (lines 40-96):
```cpp
// Source: CPUSGFP4Dequant.cpp lines 43-51, 62-95 (excerpt)
auto param = mOp->main_as_SGFP4DequantParam();
if (nullptr == param) { return INVALID_VALUE; }
if (!USE_EXTERNAL_DATA(param) || nullptr == mOp->externalPath()) { return NOT_SUPPORT; }
auto external = param->external()->data();
int64_t offset = external[0];
int64_t size   = external[1];
// ... queryFileSize bound check (offsetSize > fileSize || readSize > fileSize - offsetSize -> INVALID_VALUE)
FileLoader loader(mOp->externalPath()->c_str(), true);
if (!loader.valid()) { return NOT_SUPPORT; }
mContainer.resize(readSize);
loader.offset(offset);
if (!loader.read(reinterpret_cast<char*>(mContainer.data()), size)) { return INVALID_VALUE; }
```

**D-05 pre-validation (scratch-decode form, recommended in RESEARCH Pattern 3):**
```cpp
// Reuses the fully-bounds-checked Phase-1-tested walk verbatim:
std::vector<float> scratch(outElementCount);
if (!dequant_sgfp4_container_cpu(mContainer.data(), mContainer.size(),
                                 scratch.data(), outElementCount)) {
    return INVALID_VALUE;  // malformed — no upload, no dispatch (D-05)
}
```

**D-01 SSBO upload (VulkanBuffer hostData ctor — simplest form, RESEARCH Pattern 2a):**
```cpp
mContainerBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), /*separate=*/false,
                        containerBytes, mContainer.data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
```

---

### `source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp` (GLSL compute shader)

**Analog:** `fp4_dequant.comp` — conventions only (structure/layout/FLOAT macro). The framing walk + affine decode body is NEW (see "No Analog Found").
**Normative math reference:** `include/MNN/SGFP4DequantUtils.hpp` constants + `unpack_leaf_header` + `sgfp4_decode_leaf_payload` (C++ → GLSL 1:1 port).

**Bindings + workgroup + bounds guard** (`fp4_dequant.comp` lines 12-22, 66-69):
```glsl
// Source: fp4_dequant.comp
layout(binding = 0) readonly buffer SrcBuffer { uint SrcRaw[]; };   // SGFP4: Container[]
layout(binding = 1) writeonly buffer DstBuffer { FLOAT Dst[]; };    // unchanged
layout(binding = 2) uniform ConstBuffer { uint elementCount; };     // SGFP4: + containerSize
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= elementCount) {
        return;   // ONLY in-shader guard — host pre-validated (D-05)
    }
```

**Header comment discipline** (`fp4_dequant.comp` lines 1-5 — mandatory):
```glsl
// FP4 Dequantization (E2M1) GLSL compute shader
// FLOAT macro is provided by makeshader.py (float for FP32, float16_t for FP16).
// Do NOT redefine FLOAT/FLOAT2/FLOAT4 — makeshader.py prepends the header.
```
Also: **no `#version` line** — makeshader.py prepends `FP32_HEADER`/`FP16_HEADER`.

**New GLSL body (no in-tree analog; derived from format constants — from RESEARCH Code Examples, planner should treat as reference implementation):** `read_u32_le(byteAddr)` unaligned word composition (mandatory — record count `B` sits at byte 5), `unpackLeafHeader` via `unpackHalf2x16` (never hand-roll FP16 bit math), `codeMode0` (`(nib ^ 0x8u) - 0x8u` sign-extend) / `codeMode1` (ternary 01→+1, 10→−1, 00/11→0), per-thread `locateElement` framing re-walk (D-04), `Dst[idx] = FLOAT(S) * FLOAT(c) + FLOAT(bias);`. Full skeleton in RESEARCH.md "Code Examples".

---

### `source/backend/vulkan/buffer/execution/glsl/macro.json` (config edit)

**Analog:** the `blitregion.comp` entry, lines 8-9 — the minimal `useFP16` form (no macros needed for SGFP4):
```json
// Source: source/backend/vulkan/buffer/execution/glsl/macro.json lines 8-9
"blitregion.comp": {
    "useFP16": true
},
```
Add an identical entry for `"sgfp4_dequant.comp"` — **mandatory for D-06**: `makeshader.py` generates the `_FP16_comp` variant ONLY if this entry exists with `"useFP16": true` (`genShaderFileObjs`, makeshader.py:489-512).

---

### Regenerated artifacts: `AllShader.cpp`, `AllShader.h`, `VulkanShaderMap.cpp`

**No analog-pattern applies — these are outputs, never hand-edited** (`/*Auto Generated File, Don't Modified.*/` header). Regenerate:
```bash
cd source/backend/vulkan/buffer/compiler && python3 makeshader.py   # WSL or Git Bash ONLY (POSIX find)
```
Verify embedding: `grep -c 'sgfp4_dequant' AllShader.cpp` (≥4), `../shaders/AllShader.h` (4), `VulkanShaderMap.cpp` (2). **Wave-0 blocker:** `glslangValidator` is not installed anywhere on this machine (WSL `glslang-tools` install or Windows Vulkan SDK — see RESEARCH Open Question 1).

---

### `test/op/SGFP4VulkanDequantTest.cpp` (test, dual-backend parity)

**Analog 1:** `VulkanFP4DequantTest.cpp` — Vulkan harness (guard, config, includes).

**Vulkan availability guard + schedule config** (lines 66-81):
```cpp
// Source: test/op/VulkanFP4DequantTest.cpp lines 66-81
auto vulkanCreator = MNN::MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN);
if (nullptr == vulkanCreator) {
    MNN_PRINT("Vulkan backend not available — skipping FP4 dequant test\n");
    return true;                                          // D-07 graceful skip
}
MNN::ScheduleConfig config;
config.type = MNN_FORWARD_VULKAN;
MNN::BackendConfig backendConfig;
backendConfig.precision = MNN::BackendConfig::Precision_High;   // forces FP32 tensors + FP32 variant (Pitfall 2)
backendConfig.memory    = MNN::BackendConfig::Memory_High;
config.backendConfig    = &backendConfig;
std::shared_ptr<Executor::RuntimeManager> rtmgr(
    Executor::RuntimeManager::createRuntimeManager(config));
```

**Includes pattern** (`VulkanFP4DequantTest.cpp` lines 10-20): `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE` gate, `MNN_generated.h`, `MNN/expr/{Expr,ExprCreator,Module}.hpp`, `MNNTestSuite.h`, `TestUtils.h`.

**Analog 2:** `SGFP4DequantTest.cpp::runSgfp4Module` (lines 384-470) — the 0-input module pattern that IS the Vulkan test's core (swap `MNN_FORWARD_CPU` → `MNN_FORWARD_VULKAN`; everything else is identical):

```cpp
// Source: test/op/SGFP4DequantTest.cpp lines 386-404 + 445-460 (excerpt)
std::shared_ptr<MNN::OpT> op(new MNN::OpT);
op->type = MNN::OpType_SGFP4Dequant;
op->main.type = MNN::OpParameter_SGFP4DequantParam;
auto* param = new MNN::SGFP4DequantParamT;
param->magic = MNN::kSGFP4Magic;
param->external = {0, static_cast<int64_t>(fixture.containerSize)};
param->dims = {fixture.dimO, fixture.dimI};
op->main.value = param;
op->externalPath = sidecarPath;   // MUST be set directly on OpT (Pitfall 4 —
                                  // createExecutionWithExternal does not rewrite this op type)

auto output = Variable::create(Expr::create(op.get(), {}));   // 0-input Const-like source op
auto buffer = Variable::save({output});
std::shared_ptr<Module> m(Module::load({}, {}, (const uint8_t*)buffer.data(), buffer.size(), rtmgr));
auto outputs = m->onForward({});
auto* outPtr = outputs[0]->readMap<float>();
```

**Sidecar write + tolerance + assertion** (`SGFP4DequantTest.cpp` lines 363-381 + 28-31 + 458-462):
```cpp
constexpr float kFixtureRelativeTolerance = 1e-4f;   // lines 30-31

std::ofstream ofs(sidecarPath, std::ios::binary | std::ios::trunc);       // temp sidecar, lines 364-372
ofs.write(reinterpret_cast<const char*>(fixture->container), fixture->containerSize);
// ...
if (!checkVectorByRelativeError<float>(outPtr, fixture.expected,
        static_cast<int>(fixture.expectedCount), kFixtureRelativeTolerance)) {
    MNN_ERROR("...: parity mismatch\n");
    return false;
}
```

**Fixture loop + CPU reference:** clone the loop from `SGFP4DequantTest.cpp::testFixtureRoundTrip` (lines 78-95) — CPU side is `dequant_sgfp4_container_cpu(fixture.container, fixture.containerSize, out.data(), fixture.expectedCount)`; filter to uniform-layout fixtures only (explicit named filter; mixed fixtures are Phase 4). Fixtures: `#include "SGFP4DequantFixtures.h"` (`sgfp4_fixtures::kFixtures` / `kFixtureCount`).

**Test-class + registration pattern** (both analogs): `class X : public MNNTestCase { virtual bool run(int precision) override }`, end with `REGISTER_TEST(SGFP4VulkanDequant, op/sgfp4/vulkan_uniform_parity)` (naming is discretion; keep the `op/sgfp4/` namespace from Phase 1).

---

## Shared Patterns

### External-sidecar loading with DoS bound (T-01-04)
**Source:** `CPUSGFP4Dequant.cpp` lines 21-96 (excerpts above)
**Apply to:** `VulkanSGFP4Dequant.cpp` creator/constructor — probe real file size BEFORE any allocation (`mContainer` AND the `VulkanBuffer` SSBO).

### Host pre-validation before dispatch (D-05 / ASVS V5)
**Source:** `SGFP4DequantUtils.hpp::dequant_sgfp4_container_cpu` (scratch-decode reuse)
**Apply to:** `VulkanSGFP4Dequant.cpp` — validate ONCE, upload ONLY validated bytes, reject malformed with error return (CPU-path semantics), never dispatch on failure.

### FP16/FP32 shader-variant selection (D-06)
**Source:** `VulkanFP4Dequant.cpp` lines 37-48 + `macro.json` entry
**Apply to:** Execution constructor (3-way `useFP32Output` / `vkBn->useFP16()` / fallback) + mandatory `macro.json` `"useFP16": true` entry.

### Buffer-backend creator registration
**Source:** `VulkanFP4Dequant.cpp` lines 93-109
**Apply to:** `VulkanSGFP4Dequant.cpp` — `static bool gResistor = [](){ VulkanBackend::addCreator(OpType_SGFP4Dequant, ...); return true; }();`

### Error handling
**Source:** `CPUSGFP4Dequant.cpp` (returns `INVALID_VALUE` / `NOT_SUPPORT`, no exceptions — engine-wide convention, `-fno-exceptions`)
**Apply to:** all new C++ host code. Tests: `MNN_ERROR(...)` + `return false`; graceful Vulkan skip: `MNN_PRINT` + `return true`.

### Precision-aware test tolerance
**Source:** `VulkanFP4DequantTest.cpp` line 84 (`rtol = (precision == 3) ? 0.02f : 0.01f`) and `SGFP4DequantTest.cpp` line 30 (`kFixtureRelativeTolerance = 1e-4f`)
**Apply to:** parity test — `Precision_High` config → FP32 path → tight rtol 1e-4 for the primary assertion; optional relaxed pass (~2e-3) with default precision to exercise the FP16 variant.

## No Analog Found

| File / Component | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `sgfp4_dequant.comp` framing walk + affine decode body | shader logic | transform | No in-tree GLSL reads structured framing from an SSBO (existing shaders consume flat packed data). `fp4_dequant.comp` supplies conventions only. Use RESEARCH.md "Code Examples" (verified against `SGFP4DequantUtils.hpp` constants) as the reference implementation; normative math is `include/MNN/SGFP4DequantUtils.hpp`. |
| `read_u32_le` unaligned byte-composition helper | shader utility | transform | Novel — forced by `B` at byte 5 straddling u32 words. RESEARCH.md Code Examples has the verified implementation. |

## Metadata

**Analog search scope:** `source/backend/vulkan/buffer/execution/` (+`glsl/`), `source/backend/cpu/`, `test/op/`, `include/MNN/`, `source/backend/vulkan/buffer/compiler/makeshader.py` (via RESEARCH verified reads)
**Primary analogs read in full:** `VulkanFP4Dequant.{hpp,cpp}`, `fp4_dequant.comp`, `CPUSGFP4Dequant.cpp`, `macro.json`, `VulkanFP4DequantTest.cpp`, `SGFP4DequantTest.cpp` (targeted: fixture loop + `runSgfp4Module`)
**Pattern extraction date:** 2026-08-24
