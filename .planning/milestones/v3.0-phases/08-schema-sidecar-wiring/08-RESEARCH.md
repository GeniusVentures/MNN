# Phase 8: Schema + Sidecar Wiring - Research

**Researched:** 2026-08-28
**Domain:** FlatBuffers schema evolution + converter external-sidecar wiring (native C++ inference engine)
**Confidence:** HIGH (every code-level finding below was read directly from this repo; the one schema-evolution rule was verified against official FlatBuffers docs)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 (buffer-first decoder fallback):** `SGFP4DequantParam.buffer:[byte]` is a **live serialized decode source**, not converter-transient staging. If `buffer` is non-empty, the Execution decodes the container directly from it; if empty, the existing external-sidecar path (`external = {offset, size}` + `op->externalPath`) runs. This mirrors `Blob`'s inline-data pattern (`float32s`/`uint8s` serialized inline in the op param) and unlocks single-file `.mnn` artifacts.
- **D-02:** The hard `NOT_SUPPORT` gate in `CPUSGFP4Dequant::onResize` (currently `USE_EXTERNAL_DATA(param) && mOp->externalPath()` or bail, lines ~48–53) is replaced with buffer-first dispatch: `param->buffer()` non-empty → skip FileLoader entirely; empty → current sidecar path with all existing validation (offset/size sanity, T-01-04 file-size bounds check, loader validity). Buffer-mode must retain equivalent input validation (magic check at decode entry, size-vs-dims consistency) since there is no file to bound against.
- **D-03:** The Vulkan Execution gets the same buffer-first dispatch (its onResize currently mirrors the CPU sidecar gate). Both backends ship in this phase — no split commitment.
- **D-04:** Sidecar-mode remains the original supported path; buffer-mode is additive. Existing tests and artifacts (injection-tool output) must remain green unchanged.
- **D-05 (16-byte-aligned storeWeight):** `RemoveAndStoreParam` gains an `OpParameter_SGFP4DequantParam` case that appends the container bytes to the sidecar via the `storeWeight<uint8_t>` pattern, but pads the region to a **16-byte multiple** (zero-filled pad bytes) before advancing the shared `offset` — matching `sgfp4_inject`'s emission convention exactly. One ecosystem-wide alignment rule; the decoder reads by exact `{offset, size}` so padding is inert to it.
- **D-06:** After the sidecar write, `param->buffer` is cleared (standard `storeWeight` behavior) — no keep-buffer duplication (rejected: doubles artifact size, creates silent dual-source ambiguity). `external` is set to the region's `{offset, size}` (inclusive of pad? no — `size` is the **true container size**; the pad lives only in the offset advance).
- **D-07:** Externalization remains conditional on the existing converter flags (`config.saveExternalData` / `_largeModel` auto-trigger in `writeFb.cpp:108-118`); when OFF, the buffer simply serializes inline in the `.mnn`. No new converter flag in this phase.
- **D-08 (decode parity):** New tests asserting buffer-mode decode == sidecar-mode decode == existing oracle (`SGFP4DequantFixtures` / `dequant_sgfp4_container_cpu`), on both CPU and Vulkan, using identical container bytes across the two placements.
- **D-09 (converter round-trip):** A converter-path test driving `RemoveAndStoreParam`/`saveExternalData` on a synthetic `NetT` containing an SGFP4 op with a populated buffer; asserts the emitted sidecar layout is 16-byte aligned, monotonic, non-overlapping (mirroring the v2.0 audit's sidecar assertions), `external == {offset, true-size}`, buffer cleared in the serialized op, and reload+decode parity.
- **D-10 (test-util dedup, pulled forward from Phase 11):** Extract `SGFP4TestUtil.hpp` (tempPath, container builders, sidecar/offset helpers) from the duplicated helpers across `SGFP4ClassicAPITest.cpp` / `SGFP4MultiTensorTest.cpp` / `SGFP4InjectTest.cpp`, and retrofit those three files onto it. The new Phase 8 tests are born on the shared helpers. This retires the v2.0 audit's test-helper-dedup debt early; using the correct region-relative offset convention from day one prevents another W-1-class divergence.
- **D-11 (buffer-staging contract, for Phase 11):** Phase 11's PostConverter pass writes `OpT` with `buffer = [container bytes]`, `external = {}`, **no** `externalPath` — a pure graph rewrite with zero byte I/O in the pass. Externalization then happens (or not) via the existing `postTreat`/`saveExternalData` flag machinery using D-05; runtime dispatch (D-01) handles whichever form ships. Sidecar artifact when externalization ON; single-file `.mnn` when OFF.
- **D-12 (documented non-interception):** Phase 8 must document (comment at the `createExecutionWithExternal` switch in `source/core/OpCommonUtils.cpp:665` and/or in `SGFP4DequantParam`'s schema comment) that `SGFP4Dequant` is **not** and need not be one of the auto-rewritten types (`Convolution2D`/`Scale`/`LayerNorm`) — the decoder owns dispatch, so ops flow to `backend->onCreate` unmodified. This stops the Phase 11 planner from hunting for a `createExecutionWithExternal` case that intentionally doesn't exist.
- **D-13 (tech-debt timing unchanged):** W-1 (classic_api offset-convention retrofit) and W-2 (arg-stage failCleanup) stay with the injection tool / Phase 11 per the v2.0 audit's original placement; only the test-helper dedup (part of W-1's bug-class root cause) moves into Phase 8 via D-10.

### the agent's Discretion

- Exact schema comment wording in `CaffeOp.fbs`; whether `buffer` is generated via the same flatc regeneration flow (`schema/generate.ps1` / `generate.sh` → regenerate `MNN.generated.h`, commit both `.fbs` and generated headers).
- Internal structure of the `storeWeight` SGFP4 case (dedicated aligned-store helper vs inline pad logic).
- Test file naming and placement within `test/op/` family conventions.
- Whether decode-parity tests share one parameterized fixture across CPU/Vulkan or two files (existing precedent: separate `SGFP4DequantTest.cpp` / `SGFP4VulkanDequantTest.cpp`).
- Validation-strength details of buffer-mode entry checks (beyond magic + dims consistency) short of reimplementing full T-01-04 (which is file-bounded by nature).

### Deferred Ideas (OUT OF SCOPE)

- Non-64-multiple weight shapes / tiling-padding conventions — Phase 10 (unchanged placement)
- CLI flag design for triggering SGFP4 conversion — Phase 11
- W-1 classic_api offset-convention retrofit and W-2 arg-stage failCleanup — Phase 11 per v2.0 audit placement (only the helper dedup moved to Phase 8, D-10/D-13)
- A potential future inline-threshold convenience (small containers staying inline automatically) — unnecessary complexity now; externalization stays flag-driven (D-07)
- Extending `createExecutionWithExternal` interception to SGFP4 — explicitly rejected (D-12); decoder owns dispatch
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SGV2-22 | `SGFP4DequantParam.buffer:[byte]` schema field | Pattern 1 (FlatBuffers table append — verified official docs); `schema/default/CaffeOp.fbs` + regen flow (`generate.ps1`) |
| SGV2-23 | `RemoveParams.cpp` externalization of that buffer through the shared sidecar mechanism | Patterns 2-3 (`storeWeight` + aligned emission); `RemoveAndStoreParam` switch; `writeFb.cpp::postTreat` flag gating |
</phase_requirements>

## Summary

Phase 8 is a **schema-evolution + wiring** phase, not a build-a-new-capability phase. All four consumers/producers already exist: the `SGFP4DequantParam` FlatBuffers table (`schema/default/CaffeOp.fbs:118-124`, generated into `schema/current/CaffeOp_generated.h`), the converter's shared sidecar mechanism (`tools/converter/source/common/RemoveParams.cpp`, `writeFb.cpp`), the CPU decode Execution (`source/backend/cpu/CPUSGFP4Dequant.cpp`), and the Vulkan decode Execution (`source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp`). The work is: (1) append one field `buffer:[byte]` to the table and regenerate committed headers; (2) add one `case` to the `RemoveAndStoreParam` switch using an aligned `storeWeight` variant; (3) add buffer-first dispatch in front of the two decoders' existing sidecar path; (4) write parity + round-trip tests; (5) extract `SGFP4TestUtil.hpp` and document the Phase 11 hand-off contract.

The single most important architectural finding: **`storeWeight<T>` raw-concatenates with no alignment (`offset += size`)**, so the SGFP4 case must be a dedicated aligned variant that pads its own region to a 16-byte multiple *and advances the shared `offset` by the padded size* — while reporting `external = {offset, true-size}` (pad lives only in the offset advance, matching `sgfp4_inject_core.hpp:377-389`). The second finding: **the D-09 converter round-trip test cannot live in `test/op/` (`run_test.out` links only `MNN_DEPS`, not the converter)** — `RemoveAndStoreParam`/`saveExternalData` live in `MNNConvertDeps`, which is built only under `MNN_BUILD_CONVERTER=ON`, and the workspace builds static (`MNN_BUILD_SHARED_LIBS=OFF`), so the `TestPassManager`/`TestConvertResult` precedents (which are gated behind `MNN_BUILD_SHARED_LIBS=ON`) are **not** available. This is the one genuinely open placement question (see Open Questions).

**Primary recommendation:** Append `buffer:[byte]` as the last field of `SGFP4DequantParam` (FlatBuffers `Addition` rule — verified against official docs), regenerate via `schema/generate.ps1` (flatc is built on-demand from `3rd_party/flatbuffers` with Ninja), commit both `CaffeOp_generated.h` and `MNN_generated.h`, add an aligned `storeWeight` SGFP4 case to `RemoveAndStoreParam`, and gate D-09's converter round-trip test behind `MNN_BUILD_CONVERTER` with its own small converter-side executable (or a `test/op/` target conditionally linked against `MNNConvertDeps`).

## Architectural Responsibility Map

Tiers here are the project's real layers (adapted from the generic web tiers — MNN is a native C++ engine, not a client/server app):

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `buffer:[byte]` schema field + regen | Schema/FlatBuffers (`schema/default/*.fbs` → `schema/current/*.h`) | — | The field is a serialization-format concern; regen is `flatc`-mechanical |
| Runtime decode of `buffer` (buffer-first dispatch) | Backend Execution (`source/backend/cpu`, `source/backend/vulkan`) | Schema (generated accessor) | The decoder owns dispatch (D-01/D-12); it reads `param->buffer()` |
| Sidecar externalization (aligned store) | Converter (`tools/converter/source/common/RemoveParams.cpp`) | — | `RemoveAndStoreParam`/`saveExternalData` already own the shared offset + `.weight` file |
| `saveExternalData` flag gating | Converter (`writeFb.cpp::postTreat`) | — | `needExternalWeight` resolution already exists; no new flag (D-07) |
| Decode-parity tests (buffer == sidecar == oracle) | Test harness (`test/op/`) | Backend Execution | Tests drive the runtime through `Module`/`Interpreter` |
| Converter round-trip test (synthetic `NetT`) | Converter test target (`tools/converter/source/`) | Test harness | Needs `MNNConvertDeps` linkage — see Open Questions |
| `SGFP4TestUtil.hpp` extraction + retrofit | Test harness (`test/op/`) | — | Dedup of duplicated helpers in 3 test files (D-10) |
| Non-interception documentation (D-12) | Runtime core (`source/core/OpCommonUtils.cpp:665`) + Schema (fbs comment) | — | Doc-only: `SGFP4Dequant` stays out of `createExecutionWithExternal` |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| C++11 (project baseline) | n/a | All edited files | `CMAKE_CXX_STANDARD 11` (`CMakeLists.txt:27-29`); converter-common code is plain C++11; do NOT introduce C++17-only syntax |
| FlatBuffers (vendored `3rd_party/flatbuffers`) | 1.10.0 | `SGFP4DequantParam` table + object API | Already generates `CaffeOp_generated.h`; `buffer:[byte]` → `std::vector<uint8_t> buffer` in `SGFP4DequantParamT` + `buffer()` accessor on the table. [VERIFIED: `schema/current/CaffeOp_generated.h:1440-1515`; version from `.planning/research/STACK.md`] |
| `MNN::SGFP4DequantUtils.hpp` (in-tree, `include/MNN/`) | n/a | Framing constants + `dequant_sgfp4_container_cpu` oracle + `sgfp4_align16` | Single source of truth for magic/version/alignment/decode; decoders already depend on it |
| `MNN::FileLoader` (`source/core/FileLoader.hpp`) | n/a | Sidecar-mode read | Existing sidecar path in both Executions; unchanged |
| `std::ofstream` + shared `int64_t& offset` | n/a | Sidecar write | Existing `storeWeight`/`saveExternalData` primitive |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `MNN_generated.h` / `CaffeOp_generated.h` (regenerated) | n/a | Generated types | Regenerated after `.fbs` edit; commit both |
| `tools/fp4/sgfp4_inject_core.hpp` | n/a | 16-byte-aligned sidecar emission reference | The D-05 convention to match — read `:377-389` |
| `MNNTestSuite` / `MNNTestCase` | n/a | Test registration | `run_test.out` filtered runs (`op/sgfp4/*`) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Dedicated aligned `storeWeight` helper | Reuse stock `storeWeight<uint8_t>` raw-concat | Raw-concat diverges from the injection tool's 16-byte convention (the whole point of D-05); rejected by locked decision R2 |
| Buffer serialized inline only when externalization OFF (option C) | Decoder-fallback `buffer` (option B) | B is locked — `buffer` is a live decode source, not converter-transient |
| Converter test in `test/op/` (run_test.out) | Converter-side test executable | `run_test.out` doesn't link `MNNConvertDeps`; see Open Questions |

**Installation:** No new external packages. Build/regeneration steps are CMake/`flatc`, not package installs.

**Version verification:** No new packages to verify. The only "external" tool is `flatc`, built from the vendored `3rd_party/flatbuffers` on demand (`schema/generate.ps1:14-27`).

## Package Legitimacy Audit

**Not applicable** — this phase installs zero external packages. All dependencies are in-tree (vendored FlatBuffers, `half.hpp`, `SGFP4DequantUtils.hpp`, `FileLoader`). No `slopcheck`/registry gate needed.

## Architecture Patterns

### System Architecture Diagram

```mermaid
flowchart TD
    subgraph Convert-time ["Converter (MNN_BUILD_CONVERTER=ON)"]
        P11[Phase 11 PostConverter pass<br/>writes OpT with buffer=bytes, external={}] --> PT[writeFb.cpp::postTreat]
        PT -->|needExternalWeight?| FLAG{config.saveExternalData<br/>or _largeModel}
        FLAG -->|NO| INLINE[Serialize op with buffer inline<br/>single-file .mnn]
        FLAG -->|YES| RP[RemoveAndStoreParam<br/>new SGFP4DequantParam case]
        RP --> AL[aligned storeWeight:<br/>write bytes, zero-pad to 16-mult,<br/>offset += padded, external={offset,true-size},<br/>buffer cleared]
        AL --> WC[.mnn.weight sidecar<br/>single shared offset]
    end

    subgraph Runtime ["Runtime (CPU + Vulkan)"]
        M[Op with SGFP4DequantParam] --> D{buffer() non-empty?}
        D -->|YES| BUF[decode directly from inline bytes<br/>magic + dims-consistency entry checks]
        D -->|NO| EXT[sidecar path: external={offset,size}<br/>+ op->externalPath + T-01-04 bounds]
        BUF --> ORACLE[dequant_sgfp4_container_cpu<br/>or Vulkan host pre-validation + SSBO upload]
        EXT --> ORACLE
        ORACLE --> OUT[float weight tensor]
    end
```

### Recommended Project Structure
```
schema/default/CaffeOp.fbs          # + buffer:[byte] appended to SGFP4DequantParam
schema/current/CaffeOp_generated.h  # regenerated (committed)
schema/current/MNN_generated.h      # regenerated (committed — union refs unchanged but file touched)
tools/converter/source/common/
├── RemoveParams.cpp                # + OpParameter_SGFP4DequantParam case (aligned store)
└── CommonUtils.hpp                 # (no change — RemoveAndStoreParam already declared)
source/backend/cpu/CPUSGFP4Dequant.cpp   # buffer-first dispatch in onResize
source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp  # buffer-first dispatch in creator
source/core/OpCommonUtils.cpp       # D-12 comment only (no code change)
test/op/
├── SGFP4TestUtil.hpp               # NEW: extracted shared helpers (D-10)
├── SGFP4ClassicAPITest.cpp         # retrofitted onto SGFP4TestUtil.hpp
├── SGFP4MultiTensorTest.cpp        # retrofitted (region-relative builder becomes the shared one)
├── SGFP4InjectTest.cpp             # retrofitted
├── SGFP4DequantTest.cpp            # + buffer-mode parity test (D-08)
└── SGFP4VulkanDequantTest.cpp      # + buffer-mode parity test (D-08)
tools/converter/source/<new-test>.cpp  # D-09 converter round-trip (placement TBD — see Open Questions)
```

### Pattern 1: FlatBuffers table field append (schema evolution)
**What:** Add `buffer:[byte]` as the **last** field of the existing `SGFP4DequantParam` table.
**When to use:** Any `.fbs` table extension. Per official FlatBuffers `Evolution` rules, new fields MUST go at the end of the table; old data reads fine (absent field → default value), old code ignores the new field.
**Example:**
```fbs
// schema/default/CaffeOp.fbs (current, lines 113-124)
table SGFP4DequantParam {
    magic:uint32;     // 'SGF4' little-endian sanity value
    external:[int64]; // [offset, size] into the .mnn.weight sidecar
    dims:[int];       // output tensor geometry, e.g. [O, I]
    buffer:[byte];    // NEW: live serialized decode source (D-01); empty => sidecar path
}
```
**Generated result (verified against current `CaffeOp_generated.h:1440-1515`):** `SGFP4DequantParamT` gains `std::vector<uint8_t> buffer;`; the table struct gains `const flatbuffers::Vector<uint8_t> *buffer() const { return GetPointer<const flatbuffers::Vector<uint8_t> *>(10); }` (next vtable slot after `dims` at 8); `Verify` gains `VerifyOffset(verifier, 10) && verifier.VerifyVector(buffer())`; `SGFP4DequantParamBuilder` gains `add_buffer(...)`; `CreateSGFP4DequantParam` gains a `buffer` parameter.
- Source: [VERIFIED: flatbuffers.dev/evolution — "Addition: New fields MUST be added to the end of the table definition. This allows older data to still be read correctly (giving you the default value of the added field if accessed)."]

### Pattern 2: `storeWeight<T>` and the shared sidecar offset
**What:** `RemoveAndStoreParam` writes each op's payload to one `std::ofstream` while threading one `int64_t& offset` through every op; `storeWeight<T>` does `fs->write(...)`, `weight.clear()`, `external.push_back(size)`, `offset += size`.
**When to use:** Any op payload externalization. The SGFP4 case is a dedicated aligned variant.
**Example (current primitive, `RemoveParams.cpp:14-29`):**
```cpp
template <typename T>
static void storeWeight(std::ofstream* fs, std::vector<T>& weight, std::vector<int64_t>& external, int64_t& offset, bool check = true) {
    if (weight.empty() && check) { return; }
    if (external.empty()) { external.push_back(offset); }
    int64_t size = weight.size() * sizeof(T);
    fs->write(reinterpret_cast<const char*>(weight.data()), size);
    weight.clear();
    std::vector<T> empty; weight.swap(empty);
    external.push_back(size);
    offset += size;
}
```

### Pattern 3: Aligned sidecar emission (the D-05 reference)
**What:** Pad each SGFP4 region to a 16-byte multiple with zero bytes before advancing the offset; record the true size, not the padded size.
**When to use:** The new `OpParameter_SGFP4DequantParam` case — must match `sgfp4_inject_core.hpp:377-389` exactly.
**Example (injection-tool reference, `tools/fp4/sgfp4_inject_core.hpp:377-389`):**
```cpp
size_t offsetCursor = 0;
for (auto& node : injected) {
    node.sidecarOffset = offsetCursor;
    ofs.write(reinterpret_cast<const char*>(node.containerBytes.data()), node.containerBytes.size());
    const size_t aligned = MNN::sgfp4_align16(node.containerBytes.size());
    const size_t pad     = aligned - node.containerBytes.size();
    for (size_t p = 0; p < pad; ++p) { ofs.put('\0'); }
    offsetCursor += aligned;          // pad lives in the offset advance only
    // node.sidecarSize = true container size (set earlier), NOT aligned
}
```

### Pattern 4: Buffer-first decoder dispatch (D-01/D-02/D-03)
**What:** In front of the existing sidecar gate, test `param->buffer()`; non-empty → decode from inline bytes; empty → existing sidecar path unchanged.
**When to use:** Both `CPUSGFP4Dequant::onResize` and `VulkanSGFP4Dequant`'s creator. `buffer()` returns a `const flatbuffers::Vector<uint8_t>*` (empty/null when absent) — `.data()`/`.size()` are stable for the session lifetime because `mOp` points into the model buffer.
**Example (CPU dispatch shape — current gate at `CPUSGFP4Dequant.cpp:48-53`):**
```cpp
// current hard gate (to be replaced):
if (!USE_EXTERNAL_DATA(param) || nullptr == mOp->externalPath()) { return NOT_SUPPORT; }

// target dispatch:
const auto* buf = param->buffer();
if (buf != nullptr && buf->size() > 0) {
    mContainer.assign(buf->data(), buf->data() + buf->size()); // copy for unified onExecute
    // entry validation: sgfp4_is_v2_container(mContainer.data(), mContainer.size()) + dims-consistency
    return mContainer.empty() ? INVALID_VALUE : NO_ERROR;
}
// ... else existing sidecar path (offset/size sanity, T-01-04 file-size probe, FileLoader) unchanged
```

### Anti-Patterns to Avoid
- **Opening a second sidecar file/counter for SGFP4:** `saveExternalData` already threads one `offset` through one `ofstream`; a parallel writer causes offset collisions (`PITFALLS.md` Pitfall 6). Use the existing `fs`/`offset`.
- **Reporting `size = padded`:** locked decision D-06 — `external = {offset, true-size}`; pad advances `offset` only. The decoder reads exact `{offset,size}` so pad is inert.
- **Reimplementing `dequant_sgfp4_container_cpu` or magic/version checks in buffer mode:** the oracle already does full bounds/magic/version validation; reuse it (and `sgfp4_is_v2_container` for the entry gate).
- **Touching `schema/private/` or `source/internal/`:** off-limits per `CLAUDE.md`/`AGENTS.md`.
- **GLSL edits:** none needed this phase — do NOT run `makeshader.py`; `AllShader.cpp`/`AllShader.h`/`VulkanShaderMap.cpp` stay unchanged.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| External sidecar write + offset tracking | A new write pass/file | `RemoveAndStoreParam` + `saveExternalData` (existing `storeWeight`-shaped helper against the same `fs`/`offset`) | Single shared offset space prevents collisions; `Interpreter.cpp:96` computes sidecar path as `<file> + ".weight"` — no per-op sidecar file |
| 16-byte alignment | A new `align_up` helper | `MNN::sgfp4_align16()` from `SGFP4DequantUtils.hpp` | Single source of truth; the injection tool and decoders already use it |
| Magic/version/dims validation in buffer mode | A new validator | `sgfp4_is_v2_container()` + `dequant_sgfp4_container_cpu()` | Already byte-verified against the gnus-poc exporter; no file to bound against, so the decoder's own full bounds checks are the bound |
| Converter round-trip fixture construction | Hand-rolled FlatBuffers byte assembly | FlatBuffers object API (`SGFP4DequantParamT`, `OpT`, `NetT` + `Op::Pack`) | The generated `CreateSGFP4DequantParam`/`Op::Pack` are the supported path |
| Test-helper dedup | Keep 3 copies of `tempPath`/`writeU32Le`/`buildContainerUniform64`/… | `SGFP4TestUtil.hpp` (D-10) | Duplication caused the W-1 offset-convention divergence |

**Key insight:** The entire external-data mechanism (shared offset, `.weight` naming, flag gating, `FileLoader` read-back) already exists and is battle-tested by `Convolution2D`/`Scale`/`LayerNorm`/`Blob`. Phase 8 is strictly additive — one schema field, one switch case, one dispatch branch. Any "new mechanism" is a smell.

## Common Pitfalls

### Pitfall 1: Generated-header drift / committing the wrong files
**What goes wrong:** After editing `CaffeOp.fbs` and regenerating, only `CaffeOp_generated.h` is committed; `MNN_generated.h` (which references the union) is left stale, or the committed headers don't match the `.fbs`.
**Why it happens:** `schema/generate.ps1` regenerates **all** `.fbs` in the dir (`Get-ChildItem ..\$DIR\*.fbs`), so multiple headers change in one run; it's easy to stage only the visually-changed one.
**How to avoid:** After regen, `git status` the `schema/current/` dir and commit **every** regenerated header, plus the `.fbs` edit. `MNN_generated.h` will regenerate identically in content for the SGFP4 union (enum value `OpParameter_SGFP4DequantParam = 102` and `AsSGFP4DequantParam()` don't change) but the file is touched. Do NOT commit `AllShader.*`/`VulkanShaderMap.cpp` (unchanged — no GLSL edit this phase).
**Warning signs:** `MNN_generated.h` references a type/field the generated `CaffeOp_generated.h` doesn't define (or vice versa); CI format check complains.

### Pitfall 2: Offset/alignment interaction with other sidecar users
**What goes wrong:** The aligned SGFP4 case either (a) fails to advance `offset` by the padded size, so the next op's region overlaps the pad, or (b) advances correctly but reports `size = padded`, so the decoder reads pad bytes as container data.
**Why it happens:** `storeWeight` has no alignment concept; all existing cases are exact-size. The SGFP4 case is the first padded region in the shared offset space.
**How to avoid:** Dedicated aligned helper: write `trueSize` bytes, write `pad` zero bytes, `offset += aligned`, `external.push_back(trueSize)` (after `external.push_back(offset)`). Add a multi-op test (≥2 SGFP4 ops + a Convolution2D after them) to prove non-overlapping monotonic offsets across mixed types.
**Warning signs:** Second SGFP4 op's `offset % 16 != 0`; a Convolution2D following an SGFP4 op reads garbage; decoded content for layer 2 overlaps layer 1.

### Pitfall 3: Converter round-trip test cannot link the converter from `run_test.out`
**What goes wrong:** D-09's test is placed in `test/op/` and fails to compile/link because `RemoveAndStoreParam`/`saveExternalData` live in `MNNConvertDeps`, which `run_test.out` (links `MNN_DEPS` only, `test/CMakeLists.txt:1,18`) does not link.
**Why it happens:** `RemoveParams.cpp` is `tools/converter/source/common/*.cpp` → `MNNConvertDeps` (`tools/converter/CMakeLists.txt:36-40`), built only under `MNN_BUILD_CONVERTER=ON`. The converter-side test precedents (`TestPassManager`, `TestConvertResult`) are gated behind `MNN_BUILD_SHARED_LIBS=ON`, but this workspace builds **static** (`MNN_BUILD_SHARED_LIBS=OFF`, `.build/CMakeCache.txt:298`) — so those precedents are not even built here.
**How to avoid:** Give D-09 its own home that links `MNNConvertDeps` (see Open Questions Q1). Do not assume the `test/op/` + `run_test.out` convention covers it.
**Warning signs:** "undefined reference to `RemoveAndStoreParam` / `saveExternalData`" at test link time.

### Pitfall 4: Buffer-mode skips the file-bounded T-01-04 check without a replacement bound
**What goes wrong:** The buffer path has no file to probe; a naive port drops validation entirely, so a truncated/garbage inline buffer reaches the decode with no early rejection.
**Why it happens:** T-01-04 (`queryFileSize` + offset/size-vs-file check) is file-bounded by nature; `dequant_sgfp4_container_cpu`'s internal bounds checks are the real replacement but only fire at `onExecute`, not at `onResize`.
**How to avoid:** In buffer-mode `onResize`, add explicit entry checks: `sgfp4_is_v2_container(data, size)` (magic + version, the SGINJ-01 gate) and a size-vs-dims consistency check (the decode oracle already verifies "container decodes to exactly `elementCount` elements" — either call it at `onResize` as the Vulkan creator already does, or rely on it at `onExecute` and document that the `onResize`-time checks are magic/version + non-empty). D-02 explicitly allows "magic check at decode entry, size-vs-dims consistency" — do not reimplement full file-bounds.
**Warning signs:** A malformed inline buffer crashes or writes partial output instead of returning `INVALID_VALUE`.

### Pitfall 5: Region-relative vs absolute offset-table entries (the W-1 bug class)
**What goes wrong:** `SGFP4TestUtil.hpp` extraction copies the **absolute**-offset builder from `SGFP4ClassicAPITest.cpp:167-171` instead of the **region-relative** builder from `SGFP4MultiTensorTest.cpp:190-199`, re-introducing the exact divergence the audit flagged.
**Why it happens:** Both files define `buildContainerUniform64`; the classic-API one is the buggy copy (decode-vs-decode-valid but not encoder-conformant).
**How to avoid:** The shared helper in `SGFP4TestUtil.hpp` must be the **generalized region-relative** builder (`SGFP4MultiTensorTest.cpp`'s `buildContainerUniform64(dimO, dimI, out)`). Retrofit `SGFP4ClassicAPITest.cpp` onto it (this is the W-1 fix pulled forward per D-10).
**Warning signs:** Offset-table entries that start at the container's absolute byte offset (`16 + b*4`) rather than record-region-relative (`b * kRecordSize`).

### Pitfall 6: `main_as_SGFP4DequantParam()` vs `AsSGFP4DequantParam()` accessor mismatch
**What goes wrong:** Runtime code uses `op->main_as_SGFP4DequantParam()` (returns `const SGFP4DequantParam*`, the flatbuffers root) and reads `param->buffer()` (a `const Vector<uint8_t>*`); converter code uses `op->main.AsSGFP4DequantParam()` (returns `SGFP4DequantParamT*`, the object API) and reads `param->buffer` (a `std::vector<uint8_t>`). Mixing the two (e.g. calling `param->buffer.size()` in runtime code) fails to compile.
**Why it happens:** FlatBuffers generates two parallel APIs; the naming is close enough to confuse.
**How to avoid:** Runtime dispatch uses `buffer()` + `.data()/.size()`; converter `RemoveAndStoreParam` uses `buffer` + `.size()/.clear()`. Keep them clearly separated in the two files.
**Warning signs:** Compile errors on `.size()` vs `->size()`.

## Code Examples

### [Schema field addition]
```fbs
// schema/default/CaffeOp.fbs:118-124 (append buffer last)
table SGFP4DequantParam {
    magic:uint32;     // 'SGF4' little-endian sanity value
    external:[int64]; // [offset, size] into the .mnn.weight sidecar
    dims:[int];       // output tensor geometry, e.g. [O, I]
    buffer:[byte];    // D-01: live serialized decode source; empty => sidecar path
}
```

### [Aligned storeWeight case in RemoveAndStoreParam]
```cpp
// tools/converter/source/common/RemoveParams.cpp — new case in the switch
case MNN::OpParameter_SGFP4DequantParam: {
    auto param = op->main.AsSGFP4DequantParam();
    if (param->buffer.empty()) { break; }               // nothing to externalize
    if (param->external.empty()) { param->external.push_back(offset); }
    const size_t trueSize = param->buffer.size();
    fs->write(reinterpret_cast<const char*>(param->buffer.data()),
              static_cast<std::streamsize>(trueSize));
    const size_t aligned = MNN::sgfp4_align16(trueSize);   // 16-byte pad (D-05)
    const size_t pad = aligned - trueSize;
    if (pad > 0) { fs->write(kZeroPad, static_cast<std::streamsize>(pad)); }
    param->external.push_back(static_cast<int64_t>(trueSize)); // true size (D-06)
    param->buffer.clear();                                    // no dual-source
    std::vector<uint8_t> empty; param->buffer.swap(empty);
    offset += static_cast<int64_t>(aligned);                  // pad in offset advance only
    break;
}
```
(Note: `sgfp4_align16` takes `size_t`; a static `const char kZeroPad[16] = {}` or a loop of `fs->put('\0')` supplies pad bytes. This is a `storeWeight`-shaped aligned variant, not a new mechanism.)

### [Buffer-first dispatch — CPU onResize]
```cpp
// source/backend/cpu/CPUSGFP4Dequant.cpp::onResize (replacing lines 48-53)
auto param = mOp->main_as_SGFP4DequantParam();
if (nullptr == param) { return INVALID_VALUE; }
const auto* buf = param->buffer();
if (buf != nullptr && buf->size() > 0) {          // D-01 buffer-first
    mContainer.assign(buf->data(), buf->data() + buf->size());
    if (!sgfp4_is_v2_container(mContainer.data(), mContainer.size())) {
        return INVALID_VALUE;                      // magic/version entry check
    }
    return NO_ERROR;                               // decode + dims-consistency in onExecute (oracle)
}
// ... existing sidecar path (USE_EXTERNAL_DATA + externalPath + offset/size + T-01-04 + FileLoader) unchanged
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `SGFP4DequantParam` = `{magic, external, dims}` only; sidecar is the only placement | + `buffer:[byte]` as a live inline decode source (FlatBuffers table append) | Phase 8 | Single-file `.mnn` artifacts become possible; sidecar stays supported |
| `storeWeight<T>` raw-concat (no alignment) | Aligned `storeWeight` variant for SGFP4 (16-byte pad, true-size reported) | Phase 8 | Matches the injection tool's merged-sidecar convention exactly |
| CPU/Vulkan decoders hard-gate on sidecar | Buffer-first dispatch with sidecar fallback | Phase 8 | Both data placements, one unified convention |

**Deprecated/outdated:**
- The hard `NOT_SUPPORT` sidecar-only gate (`CPUSGFP4Dequant.cpp:48-53`, `VulkanSGFP4Dequant.cpp` creator gate) — replaced by dispatch, not removed (sidecar path retained verbatim).

## Runtime State Inventory

**Omitted** — this is not a rename/refactor/migration phase. It is a schema-evolution + wiring phase with no stored data, live-service config, OS-registered state, secret/env keys, or build-artifact rename concerns. (The one build-artifact note — regenerated `schema/current/*.h` must be re-committed — is covered in Pitfall 1, not as a "stale artifact carrying an old name".)

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| CMake | flatc on-demand build + full build | ✓ | (present — `.build/CMakeCache.txt` exists) | — |
| Ninja | `schema/generate.ps1` (`cmake -G "Ninja"` at `:20`) | ? (verify at exec time) | — | Use `schema/generate.sh` under WSL (already used for glslang in Phase 4) |
| `flatc` | Schema regen | built on-demand from `3rd_party/flatbuffers` (`generate.ps1:14-27`) | 1.10.0 (vendored) | none needed |
| Protobuf | `MNNConvertDeps` (via `CommonUtils.hpp` → `MNN_compression.pb.h`) | ✓ (build succeeded with `MNN_BUILD_PROTOBUFFER=OFF`, `.build/CMakeCache.txt:289` → system protobuf found) | system | `MNN_BUILD_PROTOBUFFER=ON` to bundle |
| Vulkan device | Vulkan parity tests | build ✓ (`MNN_VULKAN=ON`, `.build/CMakeCache.txt:456`); runtime device — guarded skip | — | `MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN)==nullptr` → tests pass-skip |
| `MNN_BUILD_CONVERTER` | D-09 converter round-trip | ✓ ON in `.build` (`:261`) | — | — |

**Missing dependencies with no fallback:**
- None blocking. (Protobuf is present; Vulkan device absence is already gracefully handled by the test guard.)

**Missing dependencies with fallback:**
- Ninja on Windows (if absent): use `schema/generate.sh` via WSL, or `cmake -G` with a different generator in `generate.ps1`.

**Key build-config fact (from `.build/CMakeCache.txt`):** the workspace builds with `MNN_BUILD_SHARED_LIBS=OFF`, `MNN_BUILD_CONVERTER=ON`, `MNN_BUILD_PROTOBUFFER=OFF`, `MNN_BUILD_SGFP4_TOOLS=ON`, `MNN_BUILD_TEST=ON`, `MNN_SUPPORT_TRANSFORMER_FUSE=ON`, `MNN_VULKAN=ON`. **Implication:** all `test/op/SGFP4*.cpp` are `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE`-gated (so that flag is required to even compile them), and the converter-side `TestPassManager`/`TestConvertResult` targets are **not** built under static libs.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | MNN's custom runner (`run_test.out`, `MNNTestSuite`/`MNNTestCase` registration via `MNNTestSuite::add`) |
| Config file | none — `test/CMakeLists.txt` auto-globs `test/**/*.cpp` |
| Quick run command | `run_test.out op/sgfp4/<name>` (filter = `test->name.find(prefix) == 0`, `test/main.cpp` + `MNNTestSuite.cpp:43`) |
| Full suite command | `run_test.out` — **blocked** by dead `test/op/FP4ModelTest.cpp` (pre-existing, out of scope; use filtered `op/sgfp4/` runs) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SGV2-22 | `buffer:[byte]` in `SGFP4DequantParam`; regen committed; existing artifacts + suites green | compile + regression | `run_test.out op/sgfp4/dequant` and `op/sgfp4/vulkan_uniform_parity` etc. | ✅ (existing suites) — regen adds no new test |
| SGV2-23 | `RemoveAndStoreParam` externalizes buffer via aligned store | converter round-trip | new converter-side test (Q1) | ❌ Wave 0 (placement TBD) |
| D-08 | buffer-mode == sidecar-mode == oracle (CPU + Vulkan) | unit/parity | `run_test.out op/sgfp4/<new>` | ❌ Wave 0 |
| D-10 | `SGFP4TestUtil.hpp` extracted; 3 files retrofitted | unit (compile + existing suites) | `run_test.out op/sgfp4/classic_api`, `multi_tensor`, `inject_*` | ✅ retrofit existing |
| D-12 | non-interception documented (comment only) | n/a (doc) | — | ❌ comment add |

### Sampling Rate
- **Per task commit:** `run_test.out op/sgfp4/<touched-suite>` (CPU suites) — fast filtered run.
- **Per wave merge:** `run_test.out op/sgfp4/` (all SGFP4 CPU suites) + Vulkan parity suite (skips if no device).
- **Phase gate:** all `op/sgfp4/` suites green + converter round-trip green + an existing v2.0 injected artifact still loads (D-04 regression).

### Wave 0 Gaps
- [ ] `test/op/SGFP4TestUtil.hpp` — shared helpers (tempPath, cwdPath, makeDir/removeDir, fileExists, writeU32Le, writeBytes/readBytes, generalized region-relative `buildContainerUniform64`, niche-dir writer).
- [ ] Converter round-trip test target (D-09) — placement + CMake wiring (Q1).
- [ ] Buffer-mode parity tests (D-08) — CPU + Vulkan files (or one parameterized fixture, planner discretion).
- [ ] `loadExternalParam` SGFP4 read-back case — if the planner decides converter-reload symmetry is in-scope (optional; see Open Questions Q3).

## Security Domain

### Applicable ASVS Categories (level 1)
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | n/a (no auth surface in a schema/wiring phase) |
| V3 Session Management | no | n/a |
| V4 Access Control | no | n/a |
| V5 Input Validation | **yes** | Container decode validation: `sgfp4_is_v2_container` (magic/version), T-01-04 file-size bounds (sidecar), `dequant_sgfp4_container_cpu` full bounds checks (both paths) |
| V6 Cryptography | no | n/a (SHA-256 in the injection tool is out of scope for Phase 8) |

### Known Threat Patterns for C++ inference-engine decode
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Attacker-controlled `external()[1]` forces oversized allocation | DoS | T-01-04: probe real on-disk file size before `mContainer.resize` (already present, sidecar path) |
| Truncated/garbage inline `buffer` reaches decode | Tampering / DoS | `sgfp4_is_v2_container` entry gate + `dequant_sgfp4_container_cpu` bounds-checked decode (returns false, no partial output) |
| Off-by-one/overlap in sidecar `{offset,size}` across mixed op types | Tampering | Aligned monotonic store + multi-op non-overlap test (Pitfall 2) |
| Buffer-mode `onResize` writes partial output on malformed input | Tampering | Return `INVALID_VALUE` before `onExecute`; oracle returns false → no partial write |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `buffer` appended last gets vtable slot 10 and `GetPointer<...>(10)` (auto field-id assignment, no explicit `id` attributes in `CaffeOp.fbs`) | Pattern 1 | Low — `flatc` assigns slots deterministically; planner verifies by inspecting regenerated header |
| A2 | `param->buffer()` remains valid for the session lifetime (no copy strictly needed; copying into `mContainer` is safest) | Pattern 4 | Low — `mOp` is the session-owned model buffer; copying removes all doubt |
| A3 | Buffer-mode entry validation = `sgfp4_is_v2_container` (magic+version) + dims-consistency via the decode oracle, NOT a full reimplementation of T-01-04 | Pitfall 4 | Medium — D-02 explicitly permits this scope; planner confirms exact check set |
| A4 | `loadExternalParam` (converter read-back) needs a matching SGFP4 case for symmetry | Open Questions Q3 | Low-Medium — only matters if the converter reloads its own output; may be deferrable |

**Note:** The FlatBuffers forward/backward compatibility of the field append (the D-04 regression guarantee) is **verified** from official docs, not assumed — see Sources.

## Open Questions (RESOLVED at plan time 2026-08-28 — resolutions recorded in the Phase 8 PLAN.md set)

1. **Where does the D-09 converter round-trip test live and link?** — **RESOLVED → 08-06:** option (a), a new `TestSGFP4Converter` executable under `tools/converter/source/` linked in `tools/converter/CMakeLists.txt` against `MNNConvertDeps` + `${MNN_DEPS}` (mirroring MNNConvert's static-branch whole-archive linking), with `target_include_directories(.../test/op/)` for `SGFP4TestUtil.hpp`. Keeps `run_test.out` free of the converter dependency.
   - What we know: `RemoveAndStoreParam`/`saveExternalData` are in `MNNConvertDeps` (converter lib), built only with `MNN_BUILD_CONVERTER=ON`; `run_test.out` links only `MNN_DEPS`; workspace builds static (`MNN_BUILD_SHARED_LIBS=OFF`) so the `TestPassManager`/`TestConvertResult` precedents are not built.
   - What's unclear: whether to (a) add a new converter-side test executable under `tools/converter/source/` linking `MNNConvertDeps` (add unconditionally, not in the shared-libs branch), (b) conditionally link `run_test.out` against `MNNConvertDeps` when `MNN_BUILD_CONVERTER=ON`, or (c) drive `MNNConvert` end-to-end (not viable yet — no importer emits an SGFP4 op until Phase 11).
   - Recommendation: **(a)** — a small `tools/converter/source/<name>.cpp` executable (peer to `TestPassManager` but added unconditionally) that constructs a synthetic `NetT` with a populated `SGFP4DequantParamT.buffer`, calls `saveExternalData`, and asserts 16-aligned/monotonic/non-overlapping layout + `external == {offset,true-size}` + buffer cleared + runtime reload parity. This keeps `run_test.out` clean of a converter dependency. (Falls under Claude's Discretion "test naming/placement"; flagging because it may need to live outside `test/op/`.)

2. **Exact buffer-mode validation strength (beyond magic + dims)** — **RESOLVED → 08-03:** mirror the Vulkan creator's existing host-pre-validation for both backends in buffer mode (consistent, early failure) — `sgfp4_is_v2_container` magic/version gate plus an eager `dequant_sgfp4_container_cpu` dims-consistency check at setup; no reimplementation of file-bounded T-01-04.

3. **Is a `loadExternalParam` SGFP4 read-back case in scope?** — **RESOLVED → 08-04:** yes, included for symmetry (required for `_postTreatOp` re-convert paths).

4. **One parameterized parity fixture vs two test files (CPU/Vulkan)** — **RESOLVED → 08-05:** follow precedent — two files, `SGFP4DequantTest.cpp` (CPU) + `SGFP4VulkanDequantTest.cpp` (Vulkan), each gaining a buffer-mode variant; suites `op/sgfp4/dequant_buffer` + `op/sgfp4/vulkan_buffer_parity` (Vulkan pass-skips with no device).

## Sources

### Primary (HIGH confidence — codebase reads)
- `schema/default/CaffeOp.fbs:118-124` — current `SGFP4DequantParam` table.
- `schema/current/CaffeOp_generated.h:1440-1515` — generated `SGFP4DequantParamT` / table / builder (field slots 4/6/8).
- `schema/current/MNN_generated.h:1219,1857,2700-2705,4236` — `OpParameter_SGFP4DequantParam = 102`, traits, `AsSGFP4DequantParam()`.
- `schema/default/MNN.fbs:211,445` — `SGFP4Dequant = 605`, `SGFP4DequantParam` in `OpParameter` union.
- `schema/generate.ps1` / `generate.sh` — flatc on-demand build + regen flow.
- `tools/converter/source/common/RemoveParams.cpp` — `storeWeight` (14-29), `RemoveAndStoreParam` (31-104, Blob case 62-104), `saveExternalData` (106-124), `loadExternalParam` (118+).
- `tools/converter/source/common/writeFb.cpp` — `_postTreatOp` (29-45), `_largeModel` (61-86), `postTreat` needExternalWeight + `.weight` naming (89-176).
- `tools/converter/source/common/CommonUtils.hpp` — `RemoveAndStoreParam`/`loadExternalParam` declarations.
- `source/backend/cpu/CPUSGFP4Dequant.cpp` — `onResize` gate (48-88), `queryFileSize` (24-41), `onExecute` decode.
- `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp` — creator gate + host pre-validation (126-210).
- `include/MNN/SGFP4DequantUtils.hpp` — framing constants, `sgfp4_is_v2_container`, `sgfp4_align16`, `dequant_sgfp4_container_cpu`.
- `source/core/OpCommonUtils.cpp:543-726` — `_RebuildExternalOp` + `createExecutionWithExternal` (the D-12 non-interception switch).
- `source/core/OpCommonUtils.hpp:58` — `USE_EXTERNAL_DATA` macro.
- `tools/fp4/sgfp4_inject_core.hpp:261-271,377-389` — `makeDequantOp` + aligned sidecar emission (D-05 reference).
- `test/op/SGFP4DequantTest.cpp`, `SGFP4VulkanDequantTest.cpp`, `SGFP4ClassicAPITest.cpp`, `SGFP4MultiTensorTest.cpp`, `SGFP4InjectTest.cpp` — test patterns, duplicated helpers, region-relative builder (190-199).
- `test/main.cpp`, `test/MNNTestSuite.cpp` — filtered-run mechanism.
- `test/CMakeLists.txt` — auto-glob + `MNN_DEPS`-only linking.
- `.build/CMakeCache.txt:261,289,295,298,301,432,456` — actual build flags.
- `.planning/research/{STACK,FEATURES,PITFALLS,ARCHITECTURE,SUMMARY}.md` — prior verified research (FlatBuffers 1.10.0, converter triangle, Pitfalls 1-7).
- `.planning/workstreams/sgfp4-pivot/v2.0-MILESTONE-AUDIT.md` — W-1/W-2/W-3 tech debt.

### Secondary (MEDIUM confidence — official docs, single authoritative source)
- FlatBuffers `Evolution` rules: `https://flatbuffers.dev/evolution/` — "Addition: New fields MUST be added to the end of the table definition. This allows older data to still be read correctly (giving you the default value of the added field if accessed). Older code will simply ignore the new field." (This is the authoritative backing for D-04's regression guarantee.)

### Tertiary (LOW confidence)
- None — no unverified WebSearch findings; all ecosystem claims were either code-verified or sourced from the prior in-repo research.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all deps in-tree, versions read from source/prior verified research.
- Architecture: HIGH — every integration point (schema, converter, CPU/Vulkan, tests) read directly.
- Pitfalls: HIGH — grounded in actual code + prior audit; one placement question (converter test) flagged as open.

**Research date:** 2026-08-28
**Valid until:** 2026-09-11 (30 days — stable C++/FlatBuffers domain; the only fast-moving element is the build env, which is fixed in `.build/CMakeCache.txt`).

## Project Constraints (from copilot-instructions.md)

The workspace `copilot-instructions.md` (GSD config) and `CLAUDE.md`/`AGENTS.md` (project) directives that bind the planner:

- `schema/private/` and `source/internal/` are **off-limits** — do not read/modify/reference (`CLAUDE.md` Restricted Access).
- Vulkan GLSL edits require `makeshader.py` regen + committing `AllShader.cpp`/`AllShader.h`/`VulkanShaderMap.cpp` — **not needed this phase** (no GLSL change).
- Schema edits regenerate via `schema/generate.ps1`; commit the regenerated `schema/current/` headers alongside the `.fbs`.
- 4-space indent, 120-col lines, attached braces, PascalCase classes, camelCase functions, `mCamelCase` members; RTTI/exceptions disabled (`-fno-rtti -fno-exceptions`); C++11 default.
- Commit message format: `[Module:Type] Description`.
- GSD workflow enforcement: do not make direct repo edits outside a GSD workflow (this research artifact is the GSD-produced deliverable).
