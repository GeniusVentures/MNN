# Phase 11: Graph-Rewrite PostConverter Pass + CLI Flag - Pattern Map

**Mapped:** 2026-09-01
**Files analyzed:** 15 (1 new, 12 modified, 1 verify-only, 1 optional doc)
**Analogs found:** 14 / 15 (the new pass file has an exact in-tree analog; only the D-13 smoke doc step has no code analog — it is a documentation task)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `tools/converter/source/optimizer/postconvert/InsertSGFP4Dequant.cpp` (NEW) | converter pass (transform) | graph rewrite (batch transform) | `tools/converter/source/optimizer/postconvert/SplitBlockQuantConvolution.cpp` | exact |
| `tools/converter/source/optimizer/PostConverter.cpp` | config/orchestration | batch | itself (final pass batch tail) | exact |
| `tools/converter/include/PostConverter.hpp` | interface declaration | n/a | existing `optimizeNet` declaration | exact |
| `tools/converter/source/common/WeightQuantAndCoding.cpp` | per-op hook (transform) | batch | its own `quanParameter` guard + `RemoveParams.cpp:70-73` `inputs>1` break | exact |
| `tools/converter/source/common/cli.cpp` | CLI parser (config) | request-response (argv → config) | `--hqq` / `--fp16` / `--saveExternalData` table+parse entries | exact |
| `tools/converter/include/config.hpp` | config model | n/a | `useHQQ` / `saveHalfFloat` / `dumpPass` fields | exact |
| `tools/converter/source/MNNConverter.cpp` | entrypoint | request-response | its own `if (!res)` block | exact |
| `CMakeLists.txt` (root) | build config | n/a | `tools/fp4/CMakeLists.txt:7-8` lib definition (being hoisted) | exact |
| `tools/converter/CMakeLists.txt` | build config | n/a | existing `MNNConvertDeps` link blocks (both branches) | exact |
| `tools/converter/source/TestSGFP4Converter.cpp` | test | batch (synthetic NetT → pass → assert) | itself (PHASE A/B scaffolding) | exact |
| `tools/fp4/sgfp4_inject_core.hpp` | tool core (transform) | file-I/O | its own `failCleanup` lambda + later failure sites | exact |
| `tools/fp4/author_structured_fixture.py` | authoring script | file-I/O | `validate_real_weights.py` `--gnus-poc-root` override | role-match |
| `tools/fp4/author_real_shape_fixture.py` | authoring script | file-I/O | same as above | role-match |
| `tools/fp4/README.md` (optional, D-13) | documentation | n/a | existing README sections (`## Usage`, `## Failure behavior`) | role-match |
| `test/op/SGFP4ClassicAPITest.cpp` | test (VERIFY-ONLY) | n/a | n/a — W-1 already fixed at `1df51b7e` | n/a |

## Pattern Assignments

### `tools/converter/source/optimizer/postconvert/InsertSGFP4Dequant.cpp` (NEW — converter pass, graph rewrite)

**Analog:** `tools/converter/source/optimizer/postconvert/SplitBlockQuantConvolution.cpp` (structure, registration, config access, tensorName growth) + `tools/converter/source/common/RemoveParams.cpp` (net+subgraph walk, weight reload) + `tools/fp4/sgfp4_inject_core.hpp` (node construction/splice conventions).

**Imports pattern** (`SplitBlockQuantConvolution.cpp:7-17`):
```cpp
#include <fstream>
#include <MNN/MNNDefine.h>
#include "../PostTreatUtils.hpp"
#include "config.hpp"
#include "../Global.hpp"
#include "core/FileLoader.hpp"
#include "core/ConvolutionCommon.hpp"
```
The new pass needs additionally: `#include "MNN/SGFP4DequantUtils.hpp"` (constants) and the Phase 9 encoder header (`tools/fp4/sgfp4_encode.hpp` — include path arrives via the `sgfp4_encode` target's PUBLIC include dirs, `tools/fp4/CMakeLists.txt:8`).

**Pass skeleton + registration pattern** (`SplitBlockQuantConvolution.cpp:19-26` + `:206`):
```cpp
class SplitBlockQuantConvolution : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        auto& mNet = net;
        auto config = Global<modelConfig>::Get();
        // ...
    }
};
static PostConverterRegister<SplitBlockQuantConvolution> __l("SplitBlockQuantConvolution");
```
Base class + registration macro live in `PostTreatUtils.hpp:20-41`:
```cpp
class PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const = 0;
    static PostConverter* get(std::string key);
    static void add(std::shared_ptr<PostConverter> converter, std::string key);
};
template <class T>
class PostConverterRegister {
public:
    PostConverterRegister(const char* claim) {
        T* instance = new T;
        PostConverter::add(std::shared_ptr<PostConverter>(instance), claim);
    }
};
```

**Flag gate pattern (D-14 dead-code-when-off)** — guard at the top of `onExecute`, following `SplitBlockQuantConvolution.cpp:26` config access:
```cpp
auto config = Global<modelConfig>::Get();
if (nullptr == config || !config->useSGFP4) {
    return true; // D-14: zero behavior change when flag absent
}
```

**Idempotency / rewrite condition (D-02 fingerprint)** — `RemoveParams.cpp:68-73` is the codebase's own `inputs>1` precedent (the Convolution2D case of `RemoveAndStoreParam`):
```cpp
case MNN::OpParameter_Convolution2D:
{
    if (op->inputIndexes.size() > 1) {
        break;
    }
```
The pass's rewrite condition is the same check inverted: rewrite only conv-family ops with `inputIndexes.size() == 1` (and `quanParameter == nullptr`).

**New-tensor-index allocation pattern** (`SplitBlockQuantConvolution.cpp:95-97`):
```cpp
subOp->outputIndexes[0] = (int)net->tensorName.size();
net->tensorName.emplace_back(originOutputName + "_" + std::to_string(i));
```
For the subgraph branch, `subgraph->tensors` is the namespace equivalent — append there, use `subgraph->tensors.size()` (pre-push) as the new index. Never renumber existing indices (a subgraph node may reference outer indices).

**Net + subgraph walk pattern (D-03)** (`RemoveParams.cpp:145-157`, `saveExternalData`):
```cpp
int64_t offset = 0;
for (auto& op : netT->oplists) {
    RemoveAndStoreParam(op, &extraFile, offset);
}
for (auto& subgraph : netT->subgraphs) {
    for (auto& op : subgraph->nodes) {
        RemoveAndStoreParam(op, &extraFile, offset);
    }
}
```

**Op-type gate (D-06)** — copy the conv-family list from `WeightQuantAndCoding.cpp:59-62` minus the Int8 types:
```cpp
if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise &&
    opType != MNN::OpType_Deconvolution && opType != MNN::OpType_DeconvolutionDepthwise) {
    return;
}
```

**Dims arithmetic (flatten {oc, ic*kx*ky} → [out, in])** — mirror `WeightQuantAndCoding.cpp:126-131` exactly: `oc = common->outputCount; kernelSize = weightSize / oc;` → `dimO = oc; dimI = kernelSize`. Encoder zero-pads non-64-multiples internally.

**External-spilled weight reload pattern (KEY Q3)** — `RemoveParams.cpp:180-215` (`loadExternalParam`) shows the exact `{offset, weightSize, biasSize}` read semantics:
```cpp
fl->offset(param->external[0]);
// quan path omitted; float path:
loadExternalData<float>(fl, param->weight, param->external[1]);
loadExternalData<float>(fl, param->bias, param->external[2]);
```
with the helper (`RemoveParams.cpp:217-222`):
```cpp
template <typename T>
static void loadExternalData(MNN::FileLoader* fl, std::vector<T>& data, int64_t size) {
    if (0 == size) {
        return;
    }
    data.resize(size / sizeof(T));
    fl->read(reinterpret_cast<char*>(data.data()), size);
}
```
CRITICAL: flush `config->externalFile` (null-check first — it is nulled when open failed, `PostConverter.cpp:645-647`) before any `FileLoader` read. Spill producer for reference: `tools/converter/source/optimizer/merge/ConvertMatMulToConv2D.cpp:270-279`.

**Weight-clear idiom after encode** (`RemoveParams.cpp:26-29`, `storeWeight`):
```cpp
weight.clear();
std::vector<T> empty;
weight.swap(empty);
```
Keep `param->bias` (restore it from the spill when `external.size() == 3`); clear `param->external`.

**Buffer-staged OpT construction (Phase 8 D-11 contract)** — `tools/converter/source/TestSGFP4Converter.cpp:57-74` (`makeSgfp4Op`) is the reference builder:
```cpp
op->type      = MNN::OpType_SGFP4Dequant;
op->main.type = MNN::OpParameter_SGFP4DequantParam;
auto* param   = new MNN::SGFP4DequantParamT;
param->magic  = MNN::kSGFP4Magic;
param->dims   = {dimO, dimI};
param->buffer.resize(container.size());
if (!container.empty()) {
    std::memcpy(param->buffer.data(), container.data(), container.size());
}
op->main.value = param;
```
`external` stays `{}`, `externalPath` stays empty — externalization rides `storeSGFP4Container` (`RemoveParams.cpp:39-72`) untouched.

**Naming/splice conventions** — mirror the injection tool (`tools/fp4/sgfp4_inject_core.hpp:399-405`) so the two artifact producers stay structurally comparable: tensor name `<weightName>_sgfp4`, new node output becomes conv `inputs[1]`, conv op type unchanged, bias untouched.

**Encoder call (D-08)** — `tools/fp4/sgfp4_encode.hpp:52-67`:
```cpp
extern const EncodeConfig kDefaultEncodeConfig;
std::vector<uint8_t> encode(const float* weights, int dimO, int dimI);
std::vector<uint8_t> encode(const float* weights, int dimO, int dimI, const EncodeConfig& config);
```
Use the config-carrying overload with a named converter-side constant aliasing `kDefaultEncodeConfig` (alias as const-ref in the pass .cpp ONLY — never redefine, never in a header; MSVC C2086 pitfall). Comment documents `tools/fp4/real_weight_validation_report.json`.

---

### `tools/converter/source/optimizer/PostConverter.cpp` (orchestration — final pass batch)

**Analog:** the file's own tail, `PostConverter.cpp:393-394`:
```cpp
    RunNetPass({"ReIndexTensor"}, newNet);
    RunNetPass({"ReIndexOnnxIfAlias"}, newNet);
```
Change to run the new pass BEFORE `ReIndexTensor` (order locked by research KEY Q2 — `ReIndexTensor` then compacts/dedups the pass's `tensorName` additions for free):
```cpp
    RunNetPass({"InsertSGFP4Dequant", "ReIndexTensor"}, newNet);
    RunNetPass({"ReIndexOnnxIfAlias"}, newNet);
```
`RunNetPass` mechanics (`PostConverter.cpp:144-170`): looks up `PostConverter::get(pass)`, calls `onExecute`, and under `config->dumpPass` prints `[DumpPass] PostConvert::<name>: ops N -> M, tensors ...` — free observability for D-13.

---

### `tools/converter/include/PostConverter.hpp` (interface declaration)

**Analog:** the file's own single declaration (`PostConverter.hpp:23-25`):
```cpp
std::unique_ptr<MNN::NetT> optimizeNet(std::unique_ptr<MNN::NetT>& netT, bool forTraining, modelConfig& config, const std::vector<std::string>& expectPasses);
```
Add alongside it (D-12 needs to call it from the test):
```cpp
void RunNetPass(const std::vector<std::string>& passes, std::unique_ptr<MNN::NetT>& originNet);
```
(The symbol already has external linkage in `PostConverter.cpp:144`; only the declaration is missing.)

---

### `tools/converter/source/common/WeightQuantAndCoding.cpp` (per-op hook — D-02 skip-guard)

**Analog:** the hook's own guard block, `WeightQuantAndCoding.cpp:58-63`:
```cpp
    auto param = op->main.AsConvolution2D();
    auto& common = param->common;
    if (param->quanParameter.get() != nullptr) {
        return;
    }
```
Insert the topology guard immediately after (before the `useHqq` reads at `:64`):
```cpp
    // NEW (D-02): SGFP4-rewritten convs carry their weight as a second
    // input tensor; an original converter conv has only its activation index.
    if (op->inputIndexes.size() > 1) {
        return;
    }
```

---

### `tools/converter/source/common/cli.cpp` (CLI parser — D-04 flag + D-05 mutex)

**Analog:** the exact `--hqq`/`--fp16`/`--saveExternalData` precedents.

Option-table entries (boolean flags take no `cxxopts::value`): `cli.cpp:196-197` (`fp16`), `:243-245` (`hqq`), `:299-301` (`saveExternalData`):
```cpp
    (
     "fp16",
     "save Conv's weight/bias in half_float data type")
```
```cpp
    (
     "hqq",
     "using hqq quant method to improve accuracy, default: false, if use hqq, weightQuantAsymmetric is set as true"
     )
```
Place `"sgfp4"` near `hqq`. Help text must say **"SGFP4 v2"**, never "Ultra FP4" (locked terminology).

Parse-block precedents: `cli.cpp:466-468` (`fp16`), `:505-512` (`hqq`):
```cpp
    // half float
    if (result.count("fp16")) {
        modelPath.saveHalfFloat = true;
    }
```
```cpp
    if (result.count("hqq")) {
        if(modelPath.weightQuantAsymmetric) {
            modelPath.useHQQ = true;
        } else {
            std::cout << "Warning, MNN Convert only support Hqq with weight asymmetric quant! Disable Hqq currently" <<  std::endl;
        }
    }
```
D-04 parse line: `if (result.count("sgfp4")) { modelPath.useSGFP4 = true; }`.

**D-05 mutex** — NO hard-error precedent exists (the `hqq` case above is a soft downgrade D-05 rejects). Insert at the END of `initializeMNNConvertArgs`, just before `return true;` (`cli.cpp:561-564`, after the `dumpPass` block):
```cpp
    if (result.count("dumpPass")) {
        modelPath.dumpPass = true;
    }
    if (modelPath.useSGFP4 && (modelPath.weightQuantBits != 0 || modelPath.useHQQ || modelPath.saveHalfFloat)) {
        MNN_ERROR("--sgfp4 cannot be combined with --weightQuantBits, --hqq, or --fp16 "
                  "(conflicting weight transforms on the same tensors)\n");
        return false;
    }
    return true;
```
(`weightQuantBits` defaults to 0 = unset, `config.hpp:40`. `MNN_ERROR` already used in this file, e.g. `cli.cpp:463`.)

---

### `tools/converter/include/config.hpp` (config model — useSGFP4 field)

**Analog:** the `useHQQ`/`saveHalfFloat` block, `config.hpp:39-41`:
```cpp
    int weightQuantBits = 0;// If weightQuantBits > 0, it means the bit
    bool weightQuantAsymmetric = true;
    int weightQuantBlock = -1;
    bool useHQQ = false;
```
Add `bool useSGFP4 = false;` adjacent to `useHQQ`. Note the constructor init-list (`config.hpp:16-22`) only initializes a subset — in-place `= false` default (like `useHQQ`) is the convention; no ctor edit needed.

---

### `tools/converter/source/MNNConverter.cpp` (entrypoint — OQ1 exit code)

**Analog:** its own parse-failure block, `MNNConverter.cpp:15-18`:
```cpp
    auto res = MNN::Cli::initializeMNNConvertArgs(modelPath, argc, argv);
    if (!res) {
        return 0;
    }
```
Change `return 0;` → `return 1;` so D-05's mutex exits non-zero. (`pymnn/src/MNNTools.cc:34` calls the function directly and checks `res` itself — unaffected.)

---

### `CMakeLists.txt` (root — hoist sgfp4_encode above converter)

**Analog:** the lib definition being hoisted, `tools/fp4/CMakeLists.txt:7-8`:
```cmake
add_library(sgfp4_encode STATIC ${CMAKE_CURRENT_LIST_DIR}/sgfp4_encode.cpp)
target_include_directories(sgfp4_encode PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/3rd_party/half)
```
Move these two lines (adjusted paths) into the root `CMakeLists.txt` just above the converter include at `:913-916`:
```cmake
if (NOT MNN_SKIPBUILD_GEOMETRY)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tools/converter)
endif()
```
Also add `set_target_properties(sgfp4_encode PROPERTIES POSITION_INDEPENDENT_CODE ON)` for Linux SHARED builds (harmless under MSVC). Gate choice (always-with-converter vs `MNN_BUILD_SGFP4_TOOLS`) is planner OQ2 — research recommends Option A (always with `MNN_BUILD_CONVERTER`). Leave the `tools/fp4` executables where they are (`CMakeLists.txt:960-962`); de-duplicate so the lib is defined exactly once.

---

### `tools/converter/CMakeLists.txt` (link encoder into MNNConvertDeps)

**Analog:** the existing link blocks, `tools/converter/CMakeLists.txt:52-58`:
```cmake
  IF(MNN_BUILD_SHARED_LIBS)
     add_library(MNNConvertDeps SHARED ${COMMON_SRC} ${MNN_CONVERTER_BACKENDS_OBJECTS} ${CMAKE_CURRENT_LIST_DIR}/../../3rd_party/flatbuffers/src/util.cpp $<TARGET_OBJECTS:MNNUtils>)
     add_dependencies(MNNConvertDeps MNN)
  ELSE()
    add_library(MNNConvertDeps STATIC ${COMMON_SRC} ${MNN_CONVERTER_BACKENDS_OBJECTS} ${CMAKE_CURRENT_LIST_DIR}/../../3rd_party/flatbuffers/src/util.cpp)
  ENDIF()
```
Add `target_link_libraries(MNNConvertDeps PUBLIC sgfp4_encode)` in both branches (after the hoist, the target exists at configure time). `TestSGFP4Converter` (`:63-78`, `/WHOLEARCHIVE` on `MNNConvertDeps`) pulls the pass registrar automatically; `sgfp4_encode` symbols arrive by reference from pass code. The pass .cpp itself is auto-globbed by `tools/converter/source/optimizer/CMakeLists.txt:1-3` (`file(GLOB_RECURSE OPTIMIZER_SRC ...)`) — no listing needed, but note the GLOB re-configure caveat (this file's edit re-triggers configure anyway).

---

### `tools/converter/source/TestSGFP4Converter.cpp` (test — D-12 pass mechanics)

**Analog:** the file's own scaffolding. `CHECK` macro (`:34-41`), `makeSgfp4Op` builder (`:57-74`), `serializeNet` (`:77-87`), PHASE A synthetic-NetT + sidecar assertions (`:97-140`), PHASE B classic-API reload + oracle parity (`:143-180`).

New PHASE (pass mechanics) drives the registered pass directly:
```cpp
#include "PostConverter.hpp"   // gains RunNetPass declaration
modelConfig config;
config.useSGFP4 = true;
MNN::Express::Global<modelConfig>::Reset(&config);
MNN::Express::RunNetPass({"InsertSGFP4Dequant"}, net);
```
(`Global<modelConfig>::Reset` precedent: `PostConverter.cpp:634`.) Assertions per D-12: SGFP4Dequant node count, conv `inputs[1]` == new node's output index, `param->weight` empty, dequant `buffer` non-empty / `external == {}` / `externalPath` empty, light-tier conv untouched (`elements < 4096` or `dimI == 1`), subgraph `nodes`+`tensors` growth, idempotency under double invocation, and the `external == {offset, wsize, biasize}` spilled-weight path (synthetic temp bin, `FileLoader` reload, bias restored into `param->bias`). Flag-OFF variant (`config.useSGFP4 = false` → zero mutation) is the D-14 unit-level check. Container bytes for assertions: `sgfp4_test::buildContainerUniform64` from `test/op/SGFP4TestUtil.hpp` (already on the include path, `tools/converter/CMakeLists.txt:65`).

---

### `tools/fp4/sgfp4_inject_core.hpp` (tool core — W-2 failCleanup hoist)

**Analog:** the lambda's own definition and call sites. Current structure (`:278-310`): declarations + arg parse loop (`:278-288`), `usage(); return 1;` on unknown/missing args (`:288-294`), `sidecarPath` derived (`:296`), then the lambda (`:304-310`):
```cpp
    const auto failCleanup = [&outputPath, &sidecarPath]() {
        std::remove(outputPath.c_str());
        std::remove(sidecarPath.c_str());
    };
```
Hoist above the arg parse loop, restructured to be safe while `outputPath` is still empty (never `std::remove` a bare `.weight` in the CWD):
```cpp
    const auto failCleanup = [&outputPath]() {
        if (!outputPath.empty()) {
            std::remove(outputPath.c_str());
            std::remove((outputPath + ".weight").c_str());
        }
    };
```
All 12 later failure sites already call `failCleanup()` — unchanged. `sidecarPath` stays derived at `:296` for the success path.

---

### `tools/fp4/author_structured_fixture.py` + `tools/fp4/author_real_shape_fixture.py` (authoring scripts — W-3 env var)

**Analog:** `validate_real_weights.py:49` already carries the override pattern (as argparse); these two scripts hard-code instead — `author_structured_fixture.py:24-26`:
```python
# Locate the gnus-poc repo (override only if your checkout lives elsewhere).
GNUST_POC_ROOT = Path("W:/gnus/GeniusCognitiveSystem/GNUS-NEO-SWARM/gnus-poc")
sys.path.insert(0, str(GNUST_POC_ROOT))
```
(identical at `author_real_shape_fixture.py:27-29`). Fix shape (needs `import os`):
```python
GNUST_POC_ROOT = Path(os.environ.get("SGFP4_GNUS_POC_ROOT",
                                     "W:/gnus/GeniusCognitiveSystem/GNUS-NEO-SWARM/gnus-poc"))
```

---

### `tools/fp4/README.md` (optional — D-13 smoke doc)

**Analog:** the file's existing section shapes (`## Usage` at `:18`, `## Failure behavior` at `:77`). Add a short `mnnconvert --sgfp4` smoke section: the exact CLI invocation on `W:\gnus\models\alexnet_Opset16.onnx`, expected node-count assertion (8 candidates minus light-tier skips), mutex rejection shape, and corpus provenance note. No code analog — documentation task.

### `test/op/SGFP4ClassicAPITest.cpp` (VERIFY-ONLY — W-1)

Already retrofitted at commit `1df51b7e` — the file's comment block (`:84-95`) documents the swap to the shared region-relative builder `sgfp4_test::buildContainerUniform64`. **Do not re-touch.** Verify-only: `run_test.out op/sgfp4/classic_api` green, then annotate the milestone-audit item as retired.

## Shared Patterns

### PostConverter pass registration + invocation
**Source:** `tools/converter/source/optimizer/PostTreatUtils.hpp:20-41` + `PostConverter.cpp:144-170`
**Apply to:** the new pass file; the `PostConverter.cpp` batch edit; the D-12 test (via the new `RunNetPass` declaration).
Registration is a file-static `PostConverterRegister<T> __l("name");` at the bottom of the pass .cpp; invocation is `RunNetPass({"name", ...}, net)`; `--dumpPass` size-diff logging is automatic.

### Config threading (flag → modelConfig → Global<modelConfig>::Get())
**Source:** `cli.cpp:466-468` (parse) → `config.hpp:39-41` (field) → `SplitBlockQuantConvolution.cpp:26` (read)
**Apply to:** `--sgfp4` end-to-end: cli.cpp table+parse, config.hpp field, pass guard, test `Global<modelConfig>::Reset`.

### The `inputs > 1` topology fingerprint
**Source:** `tools/converter/source/common/RemoveParams.cpp:70-73`
**Apply to:** BOTH the pass's rewrite condition (`== 1` → rewrite; idempotency under double `RunOptimize`, `PostConverter.cpp:649` + `:685`) AND the D-02 `WeightQuantAndCoding` skip (`> 1` → return). The two checks are the same fingerprint by design.

### Buffer-first artifact contract (Phase 8 D-11)
**Source:** `tools/converter/source/TestSGFP4Converter.cpp:57-74` (builder) + `RemoveParams.cpp:39-72` (`storeSGFP4Container`)
**Apply to:** the pass's OpT construction — `buffer` populated, `external = {}`, `externalPath` empty. Never write `externalPath`/`external` directly in the pass (the `SplitBlockQuantConvolution` external-store style is the anti-pattern Phase 8 rejected).

### net->oplists + subgraphs iteration
**Source:** `RemoveParams.cpp:145-157` (`saveExternalData`); with per-subgraph context: `writeFb.cpp:159-168` (`context.subgraph = subgraph->name;`)
**Apply to:** the pass's D-03 walk and any D-12 subgraph test fixture.

### Error reporting
**Source:** `MNN_ERROR` usage in `cli.cpp:463`, `sgfp4_inject_core.hpp` throughout
**Apply to:** D-05 mutex message, pass internal failures (log + `return true` to avoid breaking unrelated conversions — match `RunNetPass`'s soft handling of `!valid`).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| D-05 hard mutex | CLI validation | request-response | No parse-time hard-error-on-flag-combination precedent exists in `cli.cpp` (nearest is the soft `--hqq` downgrade at `:505-512`). Use the research-specified end-of-function check. |
| D-13 smoke script/doc | documentation | n/a | No existing converter smoke-script convention in-tree; README section or small script per planner discretion. |

## Metadata

**Analog search scope:** `tools/converter/source/optimizer/**`, `tools/converter/source/common/**`, `tools/converter/include/`, `tools/converter/CMakeLists.txt`, root `CMakeLists.txt`, `tools/fp4/**`, `test/op/SGFP4*`
**Files scanned:** ~20 (passes, hooks, CLI, config, CMake, encoder, inject tool, tests, fixtures)
**Pattern extraction date:** 2026-09-01
