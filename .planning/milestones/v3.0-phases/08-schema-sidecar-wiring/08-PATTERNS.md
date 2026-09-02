# Phase 8: Schema + Sidecar Wiring - Pattern Map

**Mapped:** 2026-08-28
**Files analyzed:** 16 (new + modified; 2 read-only ground-truth analogs counted separately)
**Analogs found:** 15 / 16 (14 exact, 1 role-match; 0 no-analog)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `schema/default/CaffeOp.fbs` | schema | transform (serialization format) | existing `SGFP4DequantParam` table + `Convolution2D.external` | exact |
| `schema/current/CaffeOp_generated.h` + `schema/current/MNN_generated.h` | schema (generated) | mechanical regen | `CaffeOp_generated.h:1440-1515` (`SGFP4DequantParamT`) | exact |
| `tools/converter/source/common/RemoveParams.cpp` | converter service | file-I/O (batch write) | `storeWeight<T>` + `RemoveAndStoreParam` Blob case + `loadExternalParam` Blob read-back | exact |
| `tools/converter/source/common/CommonUtils.hpp` | config/declaration | n/a (decl only) | `CommonUtils.hpp:36-37` existing declarations | exact (likely no edit) |
| `tools/converter/source/common/writeFb.cpp` | config (read-only) | file-I/O (flag gating) | itself — `postTreat` `needExternalWeight` | exact (no edit) |
| `source/backend/cpu/CPUSGFP4Dequant.cpp` | backend execution | transform (decode) | same file's existing sidecar gate + `VulkanSGFP4Dequant` creator host-pre-validation | exact |
| `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp` | backend execution | transform (decode) | `CPUSGFP4Dequant::onResize` + same file's creator gate | exact |
| `include/MNN/SGFP4DequantUtils.hpp` | utility (read-only) | n/a (validation helpers) | itself — `sgfp4_is_v2_container` / `sgfp4_align16` | exact (no edit) |
| `source/core/OpCommonUtils.cpp` | runtime core | request-response (op dispatch) | `createExecutionWithExternal` switch at `:665` | exact (comment only) |
| `test/op/SGFP4TestUtil.hpp` | test utility (create) | n/a (shared helpers) | duplicated helpers in `SGFP4MultiTensorTest.cpp:105-215` (region-relative builder) | exact |
| `test/op/SGFP4DequantTest.cpp` | test | transform (parity) | `SGFP4VulkanDequantTest.cpp:41-105` op-construction + own `testOpLevelExternalSidecar` | exact |
| `test/op/SGFP4VulkanDequantTest.cpp` | test | transform (parity) | own `runSgfp4VulkanModule` + skip guard (`:117-123`) | exact |
| new converter round-trip test (`tools/converter/source/`) | test | file-I/O (batch) | `TestPassManager.cpp` / `TestConvertResult.cpp` + `saveExternalData` | role-match (placement open, RESEARCH Q1) |
| `test/op/SGFP4ClassicAPITest.cpp` | test (retrofit) | transform | `SGFP4TestUtil.hpp` (to be created) | exact |
| `test/op/SGFP4MultiTensorTest.cpp` | test (retrofit) | transform | `SGFP4TestUtil.hpp` (region-relative builder is the reference) | exact |
| `test/op/SGFP4InjectTest.cpp` | test (retrofit) | transform | `SGFP4TestUtil.hpp` | exact |

---

## Pattern Assignments

### `schema/default/CaffeOp.fbs` (schema, transform)

**Analog:** `schema/default/CaffeOp.fbs:113-124` (existing `SGFP4DequantParam` table) + `Convolution2D` table at `:102-110` (external vector field precedent)

**Append pattern** (current table, lines 113-124 — append `buffer` LAST, FlatBuffers "Addition" evolution rule):
```fbs
// SGFP4 v2 container descriptor: ... (existing locked comment retained)
table SGFP4DequantParam {
    magic:uint32;     // 'SGF4' little-endian sanity value
    external:[int64]; // [offset, size] into the .mnn.weight sidecar
    dims:[int];       // output tensor geometry, e.g. [O, I]
    buffer:[byte];    // D-01: live serialized decode source; empty => sidecar path
}
```

**Inline-data field precedent** (`Convolution2D.external`, lines 102-110):
```fbs
table Convolution2D {
    common:Convolution2DCommon;
    weight:[float];
    bias:[float];
    ...
    external:[int64]; // [offset, weight_bytes_size, bias_bytes_size]
}
```

**Contract:** the locked design comment ("No macroblock/quadtree/leaf/split-map fields belong here") must be preserved; the new `buffer` field follows `Blob.float32s`/`uint8s` inline-data precedent (op params that serialize inline). Regen via `schema/generate.ps1`; commit `.fbs` + all regenerated `schema/current/*.h`.

---

### `schema/current/CaffeOp_generated.h` + `schema/current/MNN_generated.h` (generated, mechanical)

**Analog:** `schema/current/CaffeOp_generated.h:1440-1515` (existing `SGFP4DequantParamT` / table / builder)

**Generated result to expect** (verified against current file — field slots 4/6/8; `buffer` lands at slot 10):
```cpp
// SGFP4DequantParamT gains:
std::vector<uint8_t> buffer;

// table accessor:
const flatbuffers::Vector<uint8_t> *buffer() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(10);
}
// Verify gains: VerifyOffset(verifier, 10) && verifier.VerifyVector(buffer())
// Builder gains add_buffer(...); CreateSGFP4DequantParam gains a buffer parameter.
```

**`MNN_generated.h`:** union `OpParameter_SGFP4DequantParam = 102` and `AsSGFP4DequantParam()` are unchanged — file touched but content-stable. Per RESEARCH Pitfall 1: `git status schema/current/` and commit **every** regenerated header; do NOT commit `AllShader.*` / `VulkanShaderMap.cpp` (no GLSL edit this phase).

---

### `tools/converter/source/common/RemoveParams.cpp` (converter service, file-I/O batch)

**Analog:** `storeWeight<T>` (`:14-29`), `RemoveAndStoreParam` Blob case (`:62-104`), `saveExternalData` (`:106-124`), `loadExternalParam` Blob read-back (`:167-190`)

**Core primitive** (`storeWeight<T>`, lines 14-29):
```cpp
template <typename T>
static void storeWeight(std::ofstream* fs, std::vector<T>& weight, std::vector<int64_t>& external, int64_t& offset, bool check = true) {
    if (weight.empty() && check) {
        return;
    }
    if (external.empty()) {
        external.push_back(offset);
    }
    int64_t size = weight.size() * sizeof(T);
    fs->write(reinterpret_cast<const char*>(weight.data()), size);
    weight.clear();
    std::vector<T> empty;
    weight.swap(empty);
    external.push_back(size);
    offset += size;
}
```

**New aligned SGFP4 case** (D-05/D-06 — add to the `switch (opType)` in `RemoveAndStoreParam`, after `OpParameter_Blob`; mirrors RESEARCH Code Example):
```cpp
case MNN::OpParameter_SGFP4DequantParam: {
    auto param = op->main.AsSGFP4DequantParam();
    if (param->buffer.empty()) { break; }
    if (param->external.empty()) { param->external.push_back(offset); }
    const size_t trueSize = param->buffer.size();
    fs->write(reinterpret_cast<const char*>(param->buffer.data()),
              static_cast<std::streamsize>(trueSize));
    const size_t aligned = MNN::sgfp4_align16(trueSize);   // D-05 16-byte pad
    const size_t pad = aligned - trueSize;
    static const char kZero = '\0';
    for (size_t p = 0; p < pad; ++p) { fs->put(kZero); }
    param->external.push_back(static_cast<int64_t>(trueSize)); // D-06 true size
    param->buffer.clear();                                    // no dual-source
    std::vector<uint8_t> empty; param->buffer.swap(empty);
    offset += static_cast<int64_t>(aligned);                  // pad in offset advance only
    break;
}
```

**Blob read-back precedent** (`loadExternalParam`, `OpType_TrainableParam`/`OpType_Const` case, `:167-190`) — for the optional `loadExternalParam` symmetry case (RESEARCH Q3):
```cpp
case MNN::OpType_TrainableParam:
case MNN::OpType_Const: {
    auto param = op->main.AsBlob();
    if (param->external.size() != 2) { return; }
    ...
    fl->offset(param->external[0]);
    switch (param->dataType) {
        case MNN::DataType_DT_UINT8:
            loadExternalData<uint8_t>(fl, param->uint8s, param->external[1]);
            break;
        ...
    }
    param->external.clear();
    break;
}
```

**Error handling:** none thrown (exceptions disabled) — `storeWeight` skips empty vectors; `saveExternalData` returns `false` on file-open failure only.

---

### `tools/converter/source/common/CommonUtils.hpp` (config/declaration)

**Analog:** `CommonUtils.hpp:36-37` — `RemoveAndStoreParam` / `loadExternalParam` already declared; no new declaration needed unless a named aligned-store helper is exposed (then add next to line 37):
```cpp
void RemoveAndStoreParam(std::unique_ptr<MNN::OpT>& op, std::ofstream* fs, int64_t& offset);
void loadExternalParam(std::unique_ptr<MNN::OpT>& op, MNN::FileLoader* fl);
```

---

### `tools/converter/source/common/writeFb.cpp` (config, read-only — no edit)

**Analog:** itself. Flag gating (D-07: no new flag) at `postTreat` lines 89-176; `needExternalWeight` resolution at `:108-118`:
```cpp
// Check If need external weight
bool needExternalWeight = config.saveExternalData;
if ((!needExternalWeight) && config.model != modelConfig::MNN) {
    needExternalWeight = _largeModel(netT.get());
}
std::ofstream externalWeightOs;
if (needExternalWeight) {
    auto weightName = config.MNNModel + ".weight";
    MNN_PRINT("Save Weight to %s\n", weightName.c_str());
    externalWeightOs.open(weightName.c_str(), ios::binary);
    ...
}
```
`_postTreatOp` (`:29-45`) calls `loadExternalParam` then, `if (needExternalWeight) RemoveAndStoreParam(op, &weightPath, offset)` — the SGFP4 case rides this automatically. **No edit.**

---

### `source/backend/cpu/CPUSGFP4Dequant.cpp` (backend execution, transform)

**Analog:** same file's `onResize` gate (`:42-58`) + `VulkanSGFP4Dequant` creator host-pre-validation (`VulkanSGFP4Dequant.cpp:193-202`)

**Gate to replace** (lines 42-53):
```cpp
ErrorCode CPUSGFP4Dequant::onResize(...) {
    auto param = mOp->main_as_SGFP4DequantParam();
    if (nullptr == param) {
        return INVALID_VALUE;
    }
    // Mirrors ConvolutionCommon.cpp's USE_EXTERNAL_DATA(param) + externalPath
    // gate: this op only supports the external-sidecar container form.
    if (!USE_EXTERNAL_DATA(param) || nullptr == mOp->externalPath()) {
        return NOT_SUPPORT;
    }
    ...
}
```

**Buffer-first dispatch to insert** (D-01/D-02 — before the sidecar block; note `.data()/.size()` accessor form, Pitfall 6):
```cpp
auto param = mOp->main_as_SGFP4DequantParam();
if (nullptr == param) {
    return INVALID_VALUE;
}
const auto* buf = param->buffer();                    // const Vector<uint8_t>*
if (buf != nullptr && buf->size() > 0) {              // D-01 buffer-first
    mContainer.assign(buf->data(), buf->data() + buf->size());
    if (!sgfp4_is_v2_container(mContainer.data(), mContainer.size())) {
        return INVALID_VALUE;                          // magic/version entry check (D-02)
    }
    return NO_ERROR;                                   // dims-consistency at onExecute (oracle)
}
// ... existing sidecar path (offset/size sanity + T-01-04 queryFileSize + FileLoader) unchanged
```

**`onExecute` decode oracle is unchanged** (`:80-108`): `dequant_sgfp4_container_cpu(mContainer.data(), mContainer.size(), dest, elementCount)` returns `false` → `INVALID_VALUE`, never partial output. The `queryFileSize` helper (`:24-41`) stays for sidecar mode only.

---

### `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp` (backend execution, transform)

**Analog:** `CPUSGFP4Dequant::onResize` dispatch above + this file's creator gate (`:128-158`) and host pre-validation (`:193-202`)

**Gate to replace** (creator, lines 128-158):
```cpp
auto param = op->main_as_SGFP4DequantParam();
if (nullptr == param) { MNN_ERROR(...); return nullptr; }
// Same external-sidecar gate as CPUSGFP4Dequant ...
if (!USE_EXTERNAL_DATA(param) || nullptr == op->externalPath()) {
    MNN_ERROR("VulkanSGFP4Dequant: op requires external sidecar data\n");
    return nullptr;
}
...offset/size sanity, T-03-02 queryFileSize, FileLoader read...
```

**Buffer-first branch to insert** (before the gate; creator already does host pre-validation, so buffer mode mirrors the validation it already performs on sidecar bytes — RESEARCH Q2 recommendation):
```cpp
const auto* buf = param->buffer();
std::vector<uint8_t> container;
if (buf != nullptr && buf->size() > 0) {
    container.assign(buf->data(), buf->data() + buf->size());
    if (!sgfp4_is_v2_container(container.data(), container.size())) {
        MNN_ERROR("VulkanSGFP4Dequant: inline buffer failed v2 version gate\n");
        return nullptr;
    }
    // fall through to the shared host pre-validation below (dims-consistency)
}
// else: existing sidecar read fills `container`
```

**Host pre-validation to reuse** (lines 193-202):
```cpp
std::vector<float> scratch(static_cast<size_t>(elementCount));
if (!dequant_sgfp4_container_cpu(container.data(), container.size(), scratch.data(),
                                 static_cast<size_t>(elementCount))) {
    MNN_ERROR("VulkanSGFP4Dequant: container failed host pre-validation\n");
    return nullptr;
}
scratch.clear();
scratch.shrink_to_fit();
```
**Shared constants:** `kSgfp4WorkgroupSize = 256` (`:24`); the `VulkanSGFP4Dequant` ctor takes `std::vector<uint8_t> container` — unchanged (both modes funnel into the same ctor).

---

### `include/MNN/SGFP4DequantUtils.hpp` (utility, read-only — no edit)

**Analog:** itself. Entry-validation + alignment helpers the Phase 8 code calls:
```cpp
// sgfp4_align16 (~line 97-100):
inline size_t sgfp4_align16(size_t x) {
    return (x + (kSGFP4Alignment - 1)) & ~(kSGFP4Alignment - 1);
}

// sgfp4_is_v2_container (~line 119-126) — the D-02 entry gate:
inline bool sgfp4_is_v2_container(const uint8_t* data, size_t size) {
    return data != nullptr && size >= kSGFP4FixedHeaderSize &&
           sgfp4_read_u32_le(data) == kSGFP4Magic &&
           data[kSGFP4VersionByteOffset] == kSGFP4Version;
}
```
`kSGFP4Magic` (`:22-28`), `kSGFP4Version = 0x02`, `kSGFP4Alignment = 16`, `dequant_sgfp4_container_cpu` oracle — all already present. **No edit.**

---

### `source/core/OpCommonUtils.cpp` (runtime core — D-12 comment only)

**Analog:** `createExecutionWithExternal` switch at `:665-680`:
```cpp
bool hasExternal = false;
switch (op->main_type()) {
    case OpParameter_Convolution2D:
        hasExternal = USE_EXTERNAL_DATA(op->main_as_Convolution2D());
        break;
    case OpParameter_Scale:
        hasExternal = USE_EXTERNAL_DATA(op->main_as_Scale());
        break;
    case OpParameter_LayerNorm:
        hasExternal = USE_EXTERNAL_DATA(op->main_as_LayerNorm());
        break;
    default:
        break;
}
if (!hasExternal) {
    return backend->onCreate(inputs, outputs, op);
}
```
**D-12 edit:** add a comment (not a case) documenting that `SGFP4Dequant` is intentionally absent — the decoder owns dispatch, so ops flow to `backend->onCreate` unmodified. No code change.

---

### `test/op/SGFP4TestUtil.hpp` (test utility, create — D-10)

**Analog:** the duplicated helpers. **Use the region-relative builder from `SGFP4MultiTensorTest.cpp:169-215` as the reference** (NOT the absolute-offset copy in `SGFP4ClassicAPITest.cpp:168-210` — Pitfall 5 / W-1).

**Helpers to extract (verbatim from `SGFP4MultiTensorTest.cpp:105-165`):** `tempPath`, `cwdPath`, `makeDir`, `removeDir`, `fileExists`, `writeU32Le`, `writeBytes`, `readBytes`:
```cpp
std::string tempPath(const char* prefix, const char* suffix) {
    std::ostringstream oss;
    oss << prefix << static_cast<unsigned long>(std::time(nullptr)) << "_"
        << static_cast<unsigned long>(std::rand()) << suffix;
    return oss.str();
}

void writeU32Le(std::vector<uint8_t>& out, size_t offset, uint32_t value) {
    out[offset]     = static_cast<uint8_t>(value & 0xFFu);
    out[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    out[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xFFu);
    out[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xFFu);
}
```

**Region-relative offset-table entry** (the W-1-correct convention, `SGFP4MultiTensorTest.cpp:191-199`):
```cpp
for (int b = 0; b < recordCount; ++b) {
    writeU32Le(out, MNN::kSGFP4RecordOffsetTableStart + b * MNN::kSGFP4RecordOffsetEntrySize,
               static_cast<uint32_t>(b * kRecordSize));   // region-RELATIVE, not absolute
}
```
(The buggy `SGFP4ClassicAPITest.cpp:168-210` writes `kRecordRegionStart + b * kRecordSize` — absolute; do NOT propagate.)

**Header guard + includes convention** (from any existing SGFP4 test): `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE`, `#include "MNN/SGFP4DequantUtils.hpp"`, `#include "MNN_generated.h"`; note `SGFP4MultiTensorTest.cpp` also includes `"fp4/sgfp4_inject_core.hpp"` for `sgfp4::sha256_hex`.

---

### `test/op/SGFP4DequantTest.cpp` (test, parity — D-08 buffer-mode variant)

**Analog:** `SGFP4VulkanDequantTest.cpp:41-105` op-construction + own `testOpLevelExternalSidecar` (registered `op/sgfp4/uniform_decode` at `:565`, `op/sgfp4/mixed_decode` at `:864`)

**Op-construction precedent** (`SGFP4VulkanDequantTest.cpp:41-60` — clone for buffer mode, drop `external`/`externalPath`, set `param->buffer` instead):
```cpp
std::shared_ptr<MNN::OpT> op(new MNN::OpT);
op->type      = MNN::OpType_SGFP4Dequant;
op->main.type = MNN::OpParameter_SGFP4DequantParam;
auto* param   = new MNN::SGFP4DequantParamT;
param->magic   = MNN::kSGFP4Magic;
param->external = {0, static_cast<int64_t>(fixture.containerSize)};  // sidecar mode
param->dims     = {fixture.dimO, fixture.dimI};
op->main.value  = param;
op->externalPath = sidecarPath;                                     // buffer mode: omit
// buffer mode: param->buffer = std::vector<uint8_t>(fixture.container, fixture.container + fixture.containerSize);
```
**Tolerance:** `kFixtureRelativeTolerance = 1e-4f`; parity assert via `checkVectorByRelativeError<float>` against `dequant_sgfp4_container_cpu` oracle.

---

### `test/op/SGFP4VulkanDequantTest.cpp` (test, parity — D-08 buffer-mode variant)

**Analog:** own `runSgfp4VulkanModule` (`:41-105`) + no-device skip guard (`:117-123`):
```cpp
auto vulkanCreator = MNN::MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN);
if (nullptr == vulkanCreator) {
    MNN_PRINT("Vulkan backend not available — skipping SGFP4 Vulkan parity test\n");
    return true;
}
```
Buffer-mode variant reuses `runSgfp4VulkanModule` but builds the `OpT` with `param->buffer` populated and no `externalPath`; same `Precision_High` FP32 variant (`kFixtureRelativeTolerance = 1e-4f`) and FP16 relaxed pass (`kFp16RelativeTolerance = 2e-3f`). Registered `op/sgfp4/vulkan_uniform_parity` (`:188`).

---

### new converter round-trip test (test, file-I/O batch — D-09, placement open)

**Analog:** `tools/converter/source/TestPassManager.cpp` / `TestConvertResult.cpp` (converter-side executables) + `saveExternalData`/`RemoveAndStoreParam`.

**Placement decision (RESEARCH Q1, Pitfall 3):** `run_test.out` links only `MNN_DEPS` (`test/CMakeLists.txt:1,18`), NOT `MNNConvertDeps`; this workspace is static (`MNN_BUILD_SHARED_LIBS=OFF`), so the `TestPassManager`/`TestConvertResult` shared-libs branch is not built. **Recommendation (a):** a small `tools/converter/source/<name>.cpp` executable linking `MNNConvertDeps` (peer to `TestPassManager`), added **unconditionally** in `tools/converter/CMakeLists.txt` (not in the shared-libs branch).

**Fixture construction** (FlatBuffers object API, not hand-rolled bytes — RESEARCH "Don't Hand-Roll"):
```cpp
std::unique_ptr<MNN::NetT> netT(new MNN::NetT);
auto op = std::make_unique<MNN::OpT>();
op->type = MNN::OpType_SGFP4Dequant;
op->main.type = MNN::OpParameter_SGFP4DequantParam;
auto* param = new MNN::SGFP4DequantParamT;
param->magic = MNN::kSGFP4Magic;
param->dims = {dimO, dimI};
param->buffer = containerBytes;          // populated buffer
op->main.value = param;
netT->oplists.push_back(std::move(op));
saveExternalData(netT, outPath + ".weight");
```
**Assertions (D-09):** 16-byte-aligned `external[0]`; monotonic/non-overlapping across ≥2 SGFP4 ops + a trailing `Convolution2D` (Pitfall 2); `external == {offset, true-size}`; `buffer` cleared in the serialized `OpT`; reload+decode parity via `Interpreter`/`Module`.

---

### `test/op/SGFP4ClassicAPITest.cpp`, `SGFP4MultiTensorTest.cpp`, `SGFP4InjectTest.cpp` (retrofit — D-10)

**Analog:** `SGFP4TestUtil.hpp` (to be created above).

**Retrofit contract:**
- Delete the local duplicated helpers (`tempPath`, `cwdPath`, `makeDir`, `removeDir`, `fileExists`, `writeU32Le`, `writeBytes`, `readBytes`, `buildContainerUniform64`) and `#include "SGFP4TestUtil.hpp"`.
- **`SGFP4ClassicAPITest.cpp` MUST switch its `buildContainerUniform64` to the shared region-relative builder** (`SGFP4MultiTensorTest.cpp:169-215`) — this is the W-1 bug-class fix pulled forward per D-10/D-13. Its current absolute-offset version (`:168-210`, `kRecordRegionStart + b * kRecordSize`) is the buggy copy.
- Keep each file's own suite registrations unchanged: `op/sgfp4/classic_api` / `classic_api_missing_sidecar` (`:442`, `:508`), `op/sgfp4/multi_tensor` / `malformed_inputs` (`:634`, `:981`), `op/sgfp4/inject` / `inject_v1_reject` (`:309`, `:387`).

---

## Shared Patterns

### Test registration
**Source:** `MNNTestSuiteRegister(ClassName, "op/sgfp4/<name>")` — verified across all five SGFP4 test files.
**Apply to:** all new/modified SGFP4 test files.
```cpp
MNNTestSuiteRegister(SGFP4DequantTest, "op/sgfp4/uniform_decode");
MNNTestSuiteRegister(SGFP4VulkanDequantTest, "op/sgfp4/vulkan_uniform_parity");
```
Filtered run: `run_test.out op/sgfp4/<name>` (filter is `test->name.find(prefix) == 0`).

### Feature gate
**Source:** all SGFP4 test files open with `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE` (required for the op type to even compile).
**Apply to:** every `test/op/SGFP4*.cpp` (including new `SGFP4TestUtil.hpp`).

### Error handling (no exceptions)
**Source:** runtime code returns `ErrorCode` (`INVALID_VALUE` / `NOT_SUPPORT` / `NO_ERROR`); converter returns `bool`/`nullptr`; tests return `bool`. Never throw. `MNN_ERROR(...)` for diagnostics in creators; `MNN_PRINT(...)` for test pass logs.
**Apply to:** `CPUSGFP4Dequant.cpp`, `VulkanSGFP4Dequant.cpp`, converter test, all SGFP4 tests.

### Decode validation (V5 input-validation control)
**Source:** `sgfp4_is_v2_container` (entry gate) + `dequant_sgfp4_container_cpu` (full bounds-checked decode, returns `false` → no partial output).
**Apply to:** both backends' buffer-mode dispatch (D-02); the sidecar path retains its existing T-01-04/T-03-02 `queryFileSize` bound.

### 16-byte alignment
**Source:** `MNN::sgfp4_align16(size_t)` from `SGFP4DequantUtils.hpp` (single source of truth).
**Apply to:** the new `RemoveAndStoreParam` SGFP4 case (D-05). Pad advances `offset` only; `external` records true size (D-06). Emission must match `tools/fp4/sgfp4_inject_core.hpp:377-389`:
```cpp
node.sidecarOffset = offsetCursor;
ofs.write(reinterpret_cast<const char*>(node.containerBytes.data()), node.containerBytes.size());
const size_t aligned = MNN::sgfp4_align16(node.containerBytes.size());
const size_t pad     = aligned - node.containerBytes.size();
static const char kZero = '\0';
for (size_t p = 0; p < pad; ++p) { ofs.put(kZero); }
offsetCursor += aligned;   // pad lives in the offset advance only
```

### Accessor form split (Pitfall 6)
**Source:** runtime uses table accessors (`param->buffer()`, `.data()/.size()`); converter uses object API (`param->buffer`, `.size()/.clear()`).
**Apply to:** `CPUSGFP4Dequant.cpp` / `VulkanSGFP4Dequant.cpp` (runtime) vs `RemoveParams.cpp` (converter).

---

## No Analog Found

None — every file has an in-repo analog. The single genuine open question is **placement** of the D-09 converter round-trip test (no unconditionally-built converter-side test executable exists yet; the `TestPassManager`/`TestConvertResult` precedents are gated behind `MNN_BUILD_SHARED_LIBS=ON`). Planner should follow RESEARCH Q1 recommendation (a) and may consult `RESEARCH.md` for the exact CMake wiring.

---

## Metadata

**Analog search scope:** `schema/default/`, `schema/current/`, `tools/converter/source/common/`, `tools/converter/CMakeLists.txt`, `tools/fp4/`, `source/backend/cpu/`, `source/backend/vulkan/buffer/execution/`, `source/core/`, `include/MNN/`, `test/op/`, `test/CMakeLists.txt`.
**Files scanned:** 16 source/test/schema files + `tools/converter/CMakeLists.txt` + `.build` link logs (for `MNNConvertDeps` static-vs-shared confirmation).
**Pattern extraction date:** 2026-08-28
**Restricted dirs avoided:** `schema/private/`, `source/internal/` (per `CLAUDE.md`).
