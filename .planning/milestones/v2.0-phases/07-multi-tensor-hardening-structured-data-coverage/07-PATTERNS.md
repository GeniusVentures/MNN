# Phase 7: Multi-Tensor Hardening & Structured-Data Coverage - Pattern Map

**Mapped:** 2026-08-27
**Files analyzed:** 5 (3 new, 2 modified)
**Analogs found:** 5 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `tools/fp4/sgfp4_inject_core.hpp` (modify) | utility (tool core) | file-I/O + transform | itself — write ordering in `run()` | exact (same file) |
| `test/op/SGFP4MultiTensorTest.cpp` (new) | test | file-I/O + request-response (session run) | `test/op/SGFP4ClassicAPITest.cpp` | exact |
| `test/op/SGFP4StructuredFixtures.h` (new) | fixture (generated data header) | data (static arrays) | `test/op/SGFP4DequantFixtures.h` | exact |
| `tools/fp4/README.md` (new) | docs | n/a | `tools/cv/README.md` | role-match |
| `tools/fp4/sgfp4_inject.cpp` (modify only if diag/args change) | CLI shim | file-I/O | itself + `sgfp4_inject_core.hpp::usage()` | exact (same file, expected unchanged) |

## Pattern Assignments

### `tools/fp4/sgfp4_inject_core.hpp` (utility, file-I/O + transform) — D-11 atomicity edit

**Analog:** the file itself (Phase 5/6 code). The edit is confined to `run()` write ordering; everything else stays byte-identical.

**Insertion point for the failure-cleanup helper** — the sidecar path is known right after arg validation (`tools/fp4/sgfp4_inject_core.hpp:296`), before any file write:

```cpp
    if (modelPath.empty() || outputPath.empty() || nicheDirs.empty()) {
        usage();
        return 1;
    }
    const std::string sidecarPath = outputPath + ".weight";
```

Every post-arg failure site that must gain cleanup (`return 1` at lines 313, 341, 360, 377, 408, 424, 429, 435, 441, 447, 461 — pre-write sites at 289/294/304 are already clean but can call the same helper harmlessly once `outputPath` is non-empty):

```cpp
    // ---- Model load + exact-shape pairing (D-02, D-04, T-05-05) --------
    auto varMap = Variable::loadMap(modelPath.c_str());
    if (varMap.empty()) {
        MNN_ERROR("sgfp4_inject: '%s' loaded as an empty variable map\n", modelPath.c_str());
        return 1;
    }
```

**First-write site (sidecar open, line 353-360) — where partial artifacts begin:**

```cpp
    // ---- Sidecar merge (D-11, SGINJ-03): write all containers into one
    // stream, non-overlapping, 16-byte-aligned offsets. Offsets are known
    // before Op construction so the spliced ops carry final {offset, size}.
    {
        std::ofstream ofs(sidecarPath, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            MNN_ERROR("sgfp4_inject: cannot open sidecar '%s' for write\n", sidecarPath.c_str());
            return 1;
        }
```

**Output save + verify-stage failures (the D-11 target: these `return 1`s leave both files on disk today)** (`tools/fp4/sgfp4_inject_core.hpp:394-461`):

```cpp
    auto outputs = Variable::mapToSequence(Variable::getInputAndOutput(varMap).second);
    Variable::save(outputs, outputPath.c_str());
    MNN_PRINT("sgfp4_inject: wrote '%s' + '%s' (%zu node(s) injected)\n", outputPath.c_str(), sidecarPath.c_str(),
              injected.size());
    ...
        if (nullptr == full) {
            MNN_ERROR("sgfp4_inject: verification reload of '%s' returned null module\n", outputPath.c_str());
            return 1;
        }
```

**Cleanup idiom to copy** — the same file already uses `std::remove` for its verify temp files (line 422), and the exit-code convention is `return 1` with `MNN_ERROR` diagnostic (never abort/throw — exceptions disabled repo-wide):

```cpp
        std::remove(tempPath.c_str());
```

Option A sketch (RESEARCH.md Q1 recommendation; lambda pattern mirrors the test's `cleanupGuard` below):

```cpp
    // After sidecarPath is known:
    auto failCleanup = [&]() {
        std::remove(outputPath.c_str());
        std::remove(sidecarPath.c_str());
    };
    // every post-write `return 1` becomes { failCleanup(); return 1; }
```

**Constraints:** success path must remain byte-identical (Pitfall 1 — `op/sgfp4/classic_api*` must stay green); `sgfp4_inject.cpp` shim keeps building unchanged (`tools/fp4/sgfp4_inject.cpp` includes only this header + calls `sgfp4_inject::run(argc, argv)`).

---

### `test/op/SGFP4MultiTensorTest.cpp` (test, file-I/O + classic-API session) — NEW

**Analog:** `test/op/SGFP4ClassicAPITest.cpp` (Phase 6, direct template). Secondary: `test/op/SGFP4InjectTest.cpp` for the graph-structure assertion.

**Header/namespace pattern** (`test/op/SGFP4ClassicAPITest.cpp:18,21-46` — copy verbatim, including the MSVC `using namespace MNN;` auto-fix and the `fp4/...` include that resolves because `tools/` is on `run_test.out`'s include path):

```cpp
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <cstdio>
#include <cstdint>
...
#include "MNN_generated.h"
#include "MNN/SGFP4DequantUtils.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "fp4/sgfp4_inject_core.hpp"

using namespace MNN;
using namespace MNN::Express;
```

**Temp-path / dir helpers** (`test/op/SGFP4ClassicAPITest.cpp:103-101,145-166` — `time+rand` naming satisfies the fresh-paths-per-probe requirement, Pitfall 6):

```cpp
std::string tempPath(const char* prefix, const char* suffix) {
    std::ostringstream oss;
    oss << prefix << static_cast<unsigned long>(std::time(nullptr)) << "_"
        << static_cast<unsigned long>(std::rand()) << suffix;
    return oss.str();
}

bool makeDir(const std::string& path) {
#if defined(_WIN32)
    return 0 == _mkdir(path.c_str());
#else
    return 0 == mkdir(path.c_str(), 0755);
#endif
}
```

**Uniform container builder — generalize to (dimO, dimI)** (`test/op/SGFP4ClassicAPITest.cpp:155-190`; note the constexpr-vs-inline workaround comment at :76-83):

```cpp
bool buildContainerUniform64(std::vector<uint8_t>& out) {
    // Arithmetic kRecordRegionStart must agree with the format's own helper.
    if (kRecordRegionStart != MNN::sgfp4_align16(kOffsetTableBytes)) {
        return false;
    }
    out.assign(kRecordRegionStart + kRecordCount * kRecordSize, 0);
    ...
```

Parameterize `recordCount = (dimO/64)*(dimI/64)`, derive `kOffsetTableBytes`/`kRecordRegionStart` from it (arithmetic expression, NOT `constexpr` calling `sgfp4_align16` — MSVC C2131, Phase 6 auto-fix).

**Synthetic niche-dir writer — parameterize for name/shape** (`test/op/SGFP4ClassicAPITest.cpp:193-210`; the manifest `path` field must basename-match the container filename — Pitfall 5):

```cpp
bool writeNicheDir(const std::vector<uint8_t>& containerBytes, const std::string& dir) {
    if (!makeDir(dir)) {
        return false;
    }
    const std::string containerPath = dir + "/phase6_fixture.sgfp4";
    if (!writeBytes(containerPath, containerBytes.data(), containerBytes.size())) {
        return false;
    }
    const std::string digest = sgfp4::sha256_hex(containerBytes.data(), containerBytes.size());
    std::ostringstream oss;
    oss << "{\"fp4_binary\":{\"path\":\"phase6_fixture.sgfp4\",\"sha256\":\"" << digest
        << "\",\"stats\":{\"shape\":[" << kMatrixDim << "," << kMatrixDim << "]}}}";
    const std::string manifest = oss.str();
    return writeBytes(dir + "/manifest.json", reinterpret_cast<const uint8_t*>(manifest.data()), manifest.size());
}
```

For the structured niche dir, feed the fixture array (`SGFP4StructuredFixtures.h`) in place of `containerBytes`; same minimal manifest JSON, swapped shape.

**In-process injection argv pattern** (`test/op/SGFP4ClassicAPITest.cpp:322-331`; multi-tensor = add a second `--niche-dir` pair, argc 7→9):

```cpp
    const char* argv[] = {"sgfp4_inject",                          // 0
                          "--model",     fx.basePath.c_str(),       // 1..2
                          "--niche-dir", fx.nicheDir.c_str(),       // 3..4
                          "--output",    fx.outPath.c_str()};       // 5..6
    if (0 != sgfp4_inject::run(7, argv)) {
        MNN_ERROR("SGFP4ClassicAPITest: sgfp4_inject::run failed\n");
        return false;
    }
```

**Classic-API session + named-I/O + parity flow** (`test/op/SGFP4ClassicAPITest.cpp:346-433` — the happy-path suite body; notes the two gotchas inline):

```cpp
            const auto& inAll  = net->getSessionInputAll(session);
            const auto& outAll = net->getSessionOutputAll(session);
            if (1 != inAll.count("input")) { ... }
            ...
            // resizeSession returns VOID (Pitfall 1): resize errors surface
            // at runSession below.
            net->resizeSession(session);
            ::memcpy(inputTensor->host<float>(), inputVals.data(), kMatrixDim * sizeof(float));
            const ErrorCode code = net->runSession(session);
            ...
            if (!runClassicSession(fx.basePath, inputVals, baseline)) {
                break;
            }
            if (!checkVectorByRelativeError<float>(got, baseline.data(), kMatrixDim, kParityRelativeTolerance)) {
```

`runClassicSession` helper at :234-281 is the FP32-baseline session (copy as-is; for chained 2-MatMul the output width becomes 64 not 512).

**Graph-structure assertion (offset-collision proof)** — secondary analog `test/op/SGFP4InjectTest.cpp:112-120,268-306`. The `collectExprs` walker:

```cpp
// Collect every expr reachable (through inputs) from the given roots.
void collectExprs(const EXPRP& expr, std::set<EXPRP>& visited) {
    if (nullptr == expr || 0 != visited.count(expr)) {
        return;
    }
    visited.insert(expr);
    for (const auto& input : expr->inputs()) {
        collectExprs(input->expr().first, visited);
    }
}
```

And its use (extend: expect 2 dequant ops, read each `SGFP4DequantParam.external{offset,size}` + `dims` per op — op params accessible via `expr->get()->main_as_SGFP4DequantParam()`; assert disjoint/aligned ranges, then `memcmp` each sidecar range against the source container bytes):

```cpp
                auto reloaded = Variable::loadMap(outPath.c_str());
                ...
                std::set<EXPRP> exprs;
                for (const auto& nameVar : reloaded) {
                    collectExprs(nameVar.second->expr().first, exprs);
                }
                int dequantCount = 0;
                int constCount   = 0;
                for (const auto& expr : exprs) {
                    if (nullptr != expr->get() && MNN::OpType_SGFP4Dequant == expr->get()->type()) {
                        ++dequantCount;
                    }
```

**Malformed-probe skeleton** — analog `SGFP4ClassicAPIMissingSidecarTest` (`test/op/SGFP4ClassicAPITest.cpp:444-508`, the failure-probe suite shape): build fixture → mutate → assert exit ≠ 0 → assert absence; one looping class per RESEARCH Q6 recommendation, per-probe diagnostics naming the probe index. File-existence check idiom:

```cpp
bool fileExists(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    return ifs.good();
}
```

**Registration pattern** (`test/op/SGFP4ClassicAPITest.cpp:442,508` — two suites in one file, `#endif // MNN_SUPPORT_TRANSFORMER_FUSE` at end):

```cpp
MNNTestSuiteRegister(SGFP4ClassicAPITest, "op/sgfp4/classic_api");
MNNTestSuiteRegister(SGFP4ClassicAPIMissingSidecarTest, "op/sgfp4/classic_api_missing_sidecar");
```

Phase 7 equivalents: `MNNTestSuiteRegister(SGFP4MultiTensorTest, "op/sgfp4/multi_tensor");` and e.g. `MNNTestSuiteRegister(SGFP4MalformedInputsTest, "op/sgfp4/malformed_inputs");`.

**Fixture-cleanup guard idiom** (`test/op/SGFP4InjectTest.cpp:141-145` / `SGFP4ClassicAPITest.cpp:334-342`) — lambda over `std::remove` + `removeDir`, invoked on both pass and fail paths.

---

### `test/op/SGFP4StructuredFixtures.h` (generated fixture header) — NEW

**Analog:** `test/op/SGFP4DequantFixtures.h` (exact precedent for generated C-array form).

**Header + provenance-comment convention** (`test/op/SGFP4DequantFixtures.h:1-11` — swap the regeneration note to the gnus-poc authoring command per D-01..D-03; record `layout_distribution`, sha256, and weight recipe in the comment per RESEARCH Pitfall 2):

```cpp
// Auto-generated by tools/fp4/encode_sgfp4.py --emit-cpp-fixture.
// DO NOT EDIT BY HAND -- regenerate via:
//   python tools/fp4/encode_sgfp4.py --emit-cpp-fixture test/op/SGFP4DequantFixtures.h
#ifndef SGFP4DequantFixtures_h
#define SGFP4DequantFixtures_h

#include <cstddef>
```

**Array + metadata-constant form** (lines 30-32, 123-141): `static const unsigned char kX_data[] = { 0x.., ... };` with dims as named constants. Phase 7 shape (per RESEARCH "Code Examples"):

```cpp
static const unsigned char kStructuredMixedData[] = { 0x53, 0x47, ... };
constexpr int kStructuredDimO = 512, kStructuredDimI = 512;
constexpr size_t kStructuredSize = sizeof(kStructuredMixedData);
// plus comment-recorded kStructuredLayoutMixedCount (provenance only — the
// C++ test cannot re-derive MIXED-ness; oracle equality does not prove it)
```

Namespace (`sgfp4_fixtures`) is optional for a single-fixture header; a bare `static const` at file scope in an include-once header is the existing pattern. **No expected-decode float array needed** — the test's oracle is `dequant_sgfp4_container_cpu` at runtime (Phase 6 D-06 zero-by-construction pattern), unlike the round-trip fixtures which carry `_expected[]`.

---

### `tools/fp4/README.md` (docs) — NEW (D-13)

**Analog:** `tools/cv/README.md` (tool-family README: short intro → build/usage → parameter/macro sections). Nearest content analog for the four required sections is the tool's own `usage()` text plus header comments:

**Usage block source** (`tools/fp4/sgfp4_inject_core.hpp:147-151`):

```cpp
inline void usage() {
    MNN_PRINT("Usage: sgfp4_inject --model <path> --niche-dir <dir> [--niche-dir <dir>...] --output <path>\n");
    MNN_PRINT("  Each --niche-dir is an unmodified fp4_exporter.py --adaptive output dir\n");
    MNN_PRINT("  (manifest.json + <niche>.sgfp4). Emits <output> plus <output>.weight.\n");
}
```

Technical content for all four areas (dims convention, niche-dir/manifest contract, CLI, sidecar layout) is enumerated in RESEARCH.md Q7 with exact code anchors (`makeDequantOp` for the param shape, `loadNicheDir` for the manifest fields, the sidecar-merge block for 16-byte alignment, `kOracleRelativeTolerance`/verify chain for the in-tool verify). Structure the README as: intro → CLI usage → niche-dir/manifest input contract → dims convention → sidecar layout → failure behavior (post-D-11). Any `tools/*/README.md` style is acceptable; `tools/cv/README.md` is the shortest reference.

---

### `tools/fp4/sgfp4_inject.cpp` (CLI shim) — likely UNCHANGED

**Analog:** itself. Only touched if diagnostics/args change (discretion). Constraint from phase canonical files: `sgfp4_inject.out` must keep building standalone (`cmake --build . --target sgfp4_inject.out` smoke). Any diag-only change stays in the core header; the shim keeps its include-and-forward shape.

## Shared Patterns

### Error handling / exit codes
**Source:** `tools/fp4/sgfp4_inject_core.hpp` throughout; `test/op/SGFP4ClassicAPITest.cpp`
**Apply to:** all core-header edits and all test failure branches.
`MNN_ERROR(...)` diagnostic + `return 1` (tool) / `return false` after `MNN_ERROR` (test, usually inside the `do { ... } while(false);` + `break` pattern with a trailing `cleanupGuard()`). Never throw/abort.

### Absolute cwd-anchored temp paths
**Source:** `SGFP4ClassicAPITest.cpp:109-121,307-321` (`cwdPath()` + `tempPath()`)
**Apply to:** every model/niche/out path in the new suites (op `externalPath` is baked literally; Phase 6 Pitfall 3).

### sha256 + manifest emission
**Source:** `sgfp4::sha256_hex` (`tools/fp4/sha256.hpp`) + `writeNicheDir` ostringstream JSON (`SGFP4ClassicAPITest.cpp:193-210`)
**Apply to:** both niche dirs in the positive test and every probe variant. Recompute sha after ANY byte mutation or the probe silently degrades into the bad-sha probe.

### Fixture cleanup guard
**Source:** `SGFP4InjectTest.cpp:141-145` lambda + `cleanupFixture` (`SGFP4ClassicAPITest.cpp:334-342`)
**Apply to:** all new suite classes; extend the removal list with the second niche dir's files.

## No Analog Found

None — every deliverable has a strong in-repo analog. The only non-analog work is the one-time Python fixture authoring in the external gnus-poc repo (D-01..D-03), covered by RESEARCH.md Q3's recipe (`export_weights(..., adaptive=True)` + `stats["layout_distribution"][4] > 0` self-assert), not by a codebase pattern.

## Metadata

**Analog search scope:** `tools/fp4/`, `test/op/`, `tools/*/README.md`
**Files read in full/targeted:** `tools/fp4/sgfp4_inject_core.hpp`, `test/op/SGFP4ClassicAPITest.cpp`, `test/op/SGFP4InjectTest.cpp` (structure-assertion section), `test/op/SGFP4DequantFixtures.h` (header + registry), `tools/cv/README.md`
**Pattern extraction date:** 2026-08-27
