---
phase: 08-schema-sidecar-wiring
workstream: sgfp4-pivot
verified: 2026-08-28T00:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 8: Schema + Sidecar Wiring — Verification Report

**Phase Goal:** Add `buffer:[byte]` to `SGFP4DequantParam` (SGV2-22), wire buffer-first decode dispatch into the CPU + Vulkan Executions (SGV2-22), and add the converter externalization path (aligned storeWeight in RemoveAndStoreParam, SGV2-23) with decode-parity and round-trip tests — establishing single-file `.mnn` (inline buffer) as a supported data placement alongside the shipped external-sidecar mode, with zero regression to the v2.0 sidecar path (D-04).

**Verified:** 2026-08-28
**Status:** passed
**Re-verification:** No — initial verification (no previous `08-VERIFICATION.md` existed)

## Goal Achievement

### Observable Truths

Merged from ROADMAP.md Phase 8 Success Criteria (SC1–SC5, with SC4 split into runtime-parity and converter-round-trip) plus PLAN frontmatter must_haves (which add no scope beyond the roadmap contract).

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Schema carries `buffer:[byte]` as LAST field; headers regenerated/committed; existing artifacts + all `op/sgfp4/` suites green (SC1, 08-01) | ✓ VERIFIED | `schema/default/CaffeOp.fbs:125-131` — table with `buffer:[byte];` at line 131, last of 4 fields, with D-11/D-12 comments. `schema/current/CaffeOp_generated.h:1440-1529` — `SGFP4DequantParamT::buffer`, `buffer()` (vtable slot 10, after dims slot 8), `add_buffer`, `CreateSGFP4DequantParam(..., buffer = 0)`. `schema/current/MNN_generated.h:1219-1221` — `OpParameter_SGFP4DequantParam = 102` as union MAX; union/traits intact. live: 11/11 suites pass incl. D-04 regressions (`classic_api`, `multi_tensor`, `inject`) |
| 2 | Shared `SGFP4TestUtil.hpp` with region-relative builder retrofits all three test files (SC5, 08-02) | ✓ VERIFIED | `test/op/SGFP4TestUtil.hpp:42` (`namespace sgfp4_test`), `:145` `buildContainerUniform64(dimO, dimI, out)`, `:167` offset entries `static_cast<uint32_t>(b * kRecordSize)` — region-relative, NOT `kRecordRegionStart + b`. All three files include it (`SGFP4ClassicAPITest.cpp:51`, `SGFP4MultiTensorTest.cpp:52`, `SGFP4InjectTest.cpp:37`); zero local definitions of the moved helpers remain (definition-pattern grep across all three files: empty) |
| 3 | Buffer-first dispatch in CPU + Vulkan decoders; empty buffer → unchanged sidecar path; entry validation retained; D-12 comment (SC2, 08-03) | ✓ VERIFIED | `source/backend/cpu/CPUSGFP4Dequant.cpp:49-70` — `param->buffer()` branch BEFORE the `USE_EXTERNAL_DATA` gate (:75), `sgfp4_is_v2_container` magic/version gate (:56), eager `dequant_sgfp4_container_cpu` dims-consistency check (:65), `INVALID_VALUE` on failure; sidecar block (T-01-04 file-size bounds, FileLoader) unchanged below. `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp:139-152` — buffer-first with sidecar as explicit `else` (:153-183 incl. T-03-02 bounds). `source/core/OpCommonUtils.cpp:671-679` — D-12 non-interception comment above the `createExecutionWithExternal` switch (comment-only, no case added) |
| 4 | Converter externalization: aligned storeWeight, true-size external, buffer cleared, no new flag; symmetric read-back (SC3, 08-04) | ✓ VERIFIED | `tools/converter/source/common/RemoveParams.cpp:29-61` — `storeSGFP4Container`: writes trueSize, zero-pads to `MNN::sgfp4_align16(trueSize)`, `external = {offset, trueSize}` (pad inert), buffer cleared (swap idiom), offset advances by aligned size; `:133-137` — `case OpParameter_SGFP4DequantParam` in `RemoveAndStoreParam` via `AsSGFP4DequantParam()`; `:241-252` — `case OpType_SGFP4Dequant` in `loadExternalParam` (external.size()==2 guard, `loadExternalData<int8_t>`, external cleared). Gated only by the existing `saveExternalData`/`_largeModel` machinery — no new converter flag introduced |
| 5 | Decode-parity suites prove buffer == sidecar == oracle on CPU and Vulkan; malformed buffer rejected (SC4a, 08-05) | ✓ VERIFIED | `test/op/SGFP4DequantTest.cpp:695` registers `op/sgfp4/dequant_buffer` (all `sgfp4_fixtures::kFixtures` via CPU Module from inline buffer, parity 1e-4; malformed probe :609/:666 — truncated 8-byte buffer must fail). `test/op/SGFP4VulkanDequantTest.cpp:305` registers `op/sgfp4/vulkan_buffer_parity` (no-device skip guard :205-210, `dequant_sgfp4_container_cpu` oracle per fixture, tight FP32 + relaxed default-precision passes). Live: both suites PASS, vulkan_buffer_parity ran ON DEVICE (1671.9 ms) |
| 6 | Converter round-trip test drives saveExternalData/RemoveAndStoreParam; layout 16-aligned/monotonic/non-overlapping; external == {offset, true-size}; buffer cleared; reload+decode parity (SC4b, 08-06) | ✓ VERIFIED | `tools/converter/source/TestSGFP4Converter.cpp:103-198` — builds two containers via `sgfp4_test::buildContainerUniform64`, drives `saveExternalData` (:130, :168), asserts `p0->external == {0, trueSize0}` (:135), `p1->external == {aligned0, trueSize1}` (:137), conv at `aligned0+aligned1` (:140), buffers cleared (:142), sidecar length == aligned total + conv bytes (:147), region memcmp integrity (:149-150), serialize + reload via classic Interpreter → `dequant_sgfp4_container_cpu` oracle parity (:175-198). `tools/converter/CMakeLists.txt:55-69` (static branch, MSVC/GNU/Clang/Apple whole-archive link) and `:99-101` (shared branch). Live: exit 0, "PASS (layout + reload parity)" |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `schema/default/CaffeOp.fbs` | `buffer:[byte]` last field of SGFP4DequantParam | ✓ VERIFIED | :125-131, with D-11/D-12 comments |
| `schema/current/CaffeOp_generated.h` | regenerated buffer accessor/add_buffer/Create | ✓ VERIFIED | :1440-1529; flatc maps `[byte]` → `int8_t` (PLAN's `uint8_t` wording is imprecise, semantics byte-correct) |
| `schema/current/MNN_generated.h` | content-stable union header | ✓ VERIFIED | :1219-1221 OpParameter_MAX = SGFP4DequantParam |
| `test/op/SGFP4TestUtil.hpp` | shared namespace-scoped helpers + region-relative builder | ✓ VERIFIED | 205 lines, `namespace sgfp4_test`, 10 inline helpers |
| `test/op/SGFP4ClassicAPITest.cpp` | retrofitted, absolute-offset builder gone | ✓ VERIFIED | includes header; :171 calls shared builder; 0 local helper definitions |
| `test/op/SGFP4MultiTensorTest.cpp` | retrofitted onto shared builder | ✓ VERIFIED | :234, :534 call `sgfp4_test::buildContainerUniform64` |
| `test/op/SGFP4InjectTest.cpp` | retrofitted | ✓ VERIFIED | includes header; 0 local helper definitions |
| `source/backend/cpu/CPUSGFP4Dequant.cpp` | buffer-first dispatch | ✓ VERIFIED | :49-70 before sidecar gate :75 |
| `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp` | buffer-first in creator | ✓ VERIFIED | :139-152, sidecar as else :153+ |
| `source/core/OpCommonUtils.cpp` | D-12 non-interception comment | ✓ VERIFIED | :671-679, comment-only |
| `tools/converter/source/common/RemoveParams.cpp` | SGFP4 store + load-back cases | ✓ VERIFIED | :133, :241 + `storeSGFP4Container` :29 |
| `test/op/SGFP4DequantTest.cpp` | `op/sgfp4/dequant_buffer` suite | ✓ VERIFIED | registered :695; live PASS |
| `test/op/SGFP4VulkanDequantTest.cpp` | `op/sgfp4/vulkan_buffer_parity` suite w/ skip guard | ✓ VERIFIED | registered :305; guard :205; live PASS on device |
| `tools/converter/source/TestSGFP4Converter.cpp` | round-trip executable | ✓ VERIFIED | 200 lines; live exit 0 |
| `tools/converter/CMakeLists.txt` | target in both build branches | ✓ VERIFIED | :55-69 static, :99-101 shared |

All artifacts: exists ✓, substantive ✓, wired ✓ (used by live-passing test suites / linked targets / dispatch paths exercised by `dequant_buffer` and `TestSGFP4Converter`).

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| 3 test .cpp files | `SGFP4TestUtil.hpp` | `#include "SGFP4TestUtil.hpp"` | ✓ WIRED | includes at :51/:52/:37; call sites qualified |
| `SGFP4TestUtil.hpp` | `MNN/SGFP4DequantUtils.hpp` | `sgfp4_align16` runtime agreement assert | ✓ WIRED | `SGFP4TestUtil.hpp:157` |
| CPU/Vulkan decoders | `param->buffer()` | generated accessor | ✓ WIRED | CPUSGFP4Dequant.cpp:52; VulkanSGFP4Dequant.cpp:144 |
| `RemoveAndStoreParam` | `param->buffer` | `AsSGFP4DequantParam()` object API | ✓ WIRED | RemoveParams.cpp:134 |
| `storeSGFP4Container` | `MNN::sgfp4_align16` | 16-byte pad | ✓ WIRED | RemoveParams.cpp:45 |
| `dequant_buffer` suite | `SGFP4DequantFixtures.h` | identical container bytes both placements | ✓ WIRED | SGFP4DequantTest.cpp:580 |
| vulkan_buffer_parity suite | `dequant_sgfp4_container_cpu` | CPU oracle | ✓ WIRED | SGFP4VulkanDequantTest.cpp:219 |
| `TestSGFP4Converter.cpp` | `saveExternalData` / `dequant_sgfp4_container_cpu` | CommonUtils.hpp include / oracle parity | ✓ WIRED | :130, :168 / :175 |
| `.fbs` → generated headers | regen pipeline | buffer() accessor present | ✓ WIRED | CaffeOp_generated.h:973→(SGFP4 table at :1465), :1496 add_buffer, :1511 Create |

### Data-Flow Trace (Level 4)

Not applicable — systems C++ code (schema, decoders, converter utilities), no dynamic-data-rendering components. Substitute behavioral evidence: the buffer bytes flow op-param → `mContainer`/`container` → `dequant_sgfp4_container_cpu`/Vulkan pipeline end-to-end, proven live by `dequant_buffer` (0.9 ms), `vulkan_buffer_parity` (1671.9 ms, on device), and `TestSGFP4Converter` Phase-B reload parity.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All op/sgfp4 suites (D-04 sidecar regression + buffer mode + Vulkan parity) | `.build\Release\run_test.out op/sgfp4/` | `passed:11, failed:0, blocked:0`, exit 0 | ✓ PASS |
| Converter round-trip (layout + reload parity) | `.build\Release\TestSGFP4Converter.exe` | `PASS (layout + reload parity)`, exit 0 | ✓ PASS |
| Zero diagnostics in phase-modified files | IDE error check on all 10 source artifacts | No errors found | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|---------------------|----------|
| SGV2-22 | 08-01, 08-02, 08-03, 08-05 | `SGFP4DequantParam.buffer:[byte]` schema + buffer-first decode wiring (CPU+Vulkan) + parity tests | ✓ SATISFIED | Truths 1, 2, 3, 5; defined at `.planning/milestones/v2.0-REQUIREMENTS.md:43` (v3.0 Converter Integration, mapped → Phase 8) |
| SGV2-23 | 08-04, 08-06 | `RemoveParams.cpp` externalization (aligned storeWeight) with round-trip test | ✓ SATISFIED | Truths 4, 6; same requirements file mapping |

Orphaned requirements: none — both phase-mapped IDs (SGV2-22, SGV2-23) are claimed by plans and satisfied. (SGV2-24..32 are mapped by the ROADMAP to Phases 9-12, outside this phase's scope.)

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | Zero `TBD`/`FIXME`/`XXX`/`HACK`/`PLACEHOLDER` markers across all 10 phase-modified source files | — | — |

ℹ️ **Note (not a gap):** `test/op/FP4ModelTest.cpp` is pre-existing dead code from the unrelated `milestone` workstream. It blocked the `run_test.out` build and was temp-stubbed then restored byte-identical (git-verified), per 08-06-SUMMARY. Ownership: milestone WS Phase 4 plan 04-02. Not counted against Phase 8.

### Human Verification Required

None. All phase must-haves are verifiable programmatically (schema inspection, code reading, test execution); all tests were run live during this verification with exit 0.

### Gaps Summary

No gaps. All 6 merged must-have truths (roadmap SC1-SC5) verified at all four levels against the actual codebase, with both behavioral spot-checks executed fresh during this verification (11/11 suites, converter round-trip exit 0). The PLAN 08-01 artifact spec's `std::vector<uint8_t> buffer` wording differs cosmetically from the generated `std::vector<int8_t>` (flatc's `[byte]` mapping) — a plan-side imprecision, byte-correct in implementation and consistently handled (`loadExternalData<int8_t>`, `storeSGFP4Container` int8_t signature).

---

_Verified: 2026-08-28_
_Verifier: the agent (gsd-verifier)_
