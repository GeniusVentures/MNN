---
phase: 01-affine-dual-mode-decode-core-cpu-uniform-layouts
verified: 2026-08-24T19:29:43Z
status: passed
score: 5/5 must-haves verified (ROADMAP success criteria) / 7/7 requirement IDs satisfied
behavior_unverified: 0
overrides_applied: 0
---

# Phase 1: Affine Dual-Mode Decode Core (CPU, Uniform Layouts) Verification Report

**Phase Goal:** A new dedicated CPU Execution class decodes SGFP4 v2 uniform-layout containers to float weights via the affine dual-mode rule `w = S·c + bias`, loading the container from an external `.mnn.weight`-style sidecar through a minimal `{magic, offset, size}` op descriptor. Establishes decode correctness for both code modes and the container plumbing, with no quadtree complexity.

**Verified:** 2026-08-24T19:29:43Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Encoder-produced SGFP4 v2 uniform-layout container round-trips through the new CPU Execution and reconstructs weights via `w = S·c + bias` for both FP4_AFFINE and T158_AFFINE within error bound | ✓ VERIFIED | Independently re-ran `./.build/Release/run_test.out.exe op/sgfp4` — all 5 layers PASSED (fixture round-trip 11 cases, ternary reserved, FP16 precision, malformed-container negatives, op-level end-to-end). Independently re-ran `python tools/fp4/encode_sgfp4.py --selftest` — 12/12 PASSED (all layouts x both modes, B=3, automatic mode selection). Decode math confirmed in `include/MNN/SGFP4DequantUtils.hpp:163-183` (`sgfp4_decode_leaf_payload`, `w = S*c + bias`), and encoder math confirmed in `tools/fp4/encode_sgfp4.py:124-163` (independent inverse implementation) |
| 2 | Op loads container from external `.mnn.weight`-style sidecar using only `{magic, offset, size}` descriptor — no macroblock/quadtree fields anywhere in FlatBuffers schema | ✓ VERIFIED | `schema/default/CaffeOp.fbs:118-122` — `SGFP4DequantParam{magic:uint32; external:[int64]; dims:[int];}` only. Grepped schema/current/*.h and schema/default/*.fbs for macroblock/quadtree/leaf/split terms in the new param — none found. `CPUSGFP4Dequant::onResize` (`source/backend/cpu/CPUSGFP4Dequant.cpp:43-89`) reads via `FileLoader(op->externalPath())` + `external()->data()[0/1]`, mirroring `ConvolutionCommon.cpp`'s idiom |
| 3 | All five uniform layouts (UNIFORM_64/32/16/8, FULL_4x4) decode with correct leaf count, row-major raster order, normative payload word counts (n²/8 mode 0, n²/16 mode 1), verified via `./run_test.out` | ✓ VERIFIED | `sgfp4_resolve_uniform_layout` (`SGFP4DequantUtils.hpp:95-123`) maps all 5 Table-3 layouts correctly (N=1/64, N=4/32, N=16/16, N=64/8, N=256/4); word counts computed as `elementCount/8` (mode 0) and `elementCount/16` (mode 1) at `SGFP4DequantUtils.hpp:306-307`. Test fixtures (`test/op/SGFP4DequantFixtures.h`) cover all 5 layouts x both modes (10 cases) + a B=3 alignment case (11 total) — confirmed via direct re-run of `run_test.out op/sgfp4` (fixture round-trip 11 cases PASSED) |
| 4 | FP16 scale+bias unpack (packHalf2x16 order; 12-bit truncated-bias recovery `S=half(h>>16)`, `bias=half(h&0xFFF0)`, `flags=h&0xF`) matches reference half→float within FP16 precision; ternary reserved symbol `11` decodes to 0 | ✓ VERIFIED | `unpack_leaf_header` (`SGFP4DequantUtils.hpp:133-145`) implements exactly this via vendored `half_float::half`. `SGFP4DequantTest::testFp16LeafHeaderPrecision` independently re-derives expected S/bias via `half_float::half` + manual 0xFFF0 masking and asserts match within 1e-3 — PASSED on independent re-run. `SGFP4DequantTest::testTernaryReservedSymbol` hand-builds an all-`0b11`-symbol payload and asserts every element decodes to `bias` (code 0) — PASSED on independent re-run |
| 5 | Existing E2M1 `CPUFP4Dequant`/`dequant_fp4_packed_cpu` path and tests are unchanged (additive, not replacement) | ✓ VERIFIED | `git diff abec332c HEAD -- include/MNN/FP4DequantUtils.hpp source/backend/cpu/CPUFP4Dequant.hpp source/backend/cpu/CPUFP4Dequant.cpp tools/fp4/quantize_fp4.py` produces zero lines of diff across the phase's entire commit range. `git log` confirms these files were last touched by pre-phase commits (`9d0de72a`, `cffaf4bd`), not by any phase-1 commit. Full suite run confirms `op/vulkan/fp4_dequant_correctness` (the E2M1 regression test) still passes as part of the 375/375 green full-suite run |

**Score:** 5/5 ROADMAP success criteria verified, 0 present-but-behavior-unverified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `include/MNN/SGFP4DequantUtils.hpp` | Header-only v2 decode: framing, uniform record walk, FP16 unpack, dual-mode decode, affine reconstruct, bounds validation | ✓ VERIFIED | 329 lines, exports `dequant_sgfp4_container_cpu` + `unpack_leaf_header` at `namespace MNN` scope (not `detail::`); all literals are named constants; full bounds-checking present at every read (lines 219-324) |
| `source/backend/cpu/CPUSGFP4Dequant.hpp/.cpp` | Execution: creator + onResize (FileLoader external read) + onExecute (decode) + REGISTER_CPU_OP_CREATOR | ✓ VERIFIED | `onResize` reads sidecar via `FileLoader` with a real on-disk size probe (bugfixed in 01-02 from a broken always-0 `FileLoader::size()` check — confirmed fix present at lines 22-39); `onExecute` calls `dequant_sgfp4_container_cpu` and returns `INVALID_VALUE` on failure (no silent partial writes); registered via `REGISTER_CPU_OP_CREATOR(CPUSGFP4DequantCreator, OpType_SGFP4Dequant)` |
| `source/shape/ShapeSGFP4Dequant.cpp` | Shape computer setting output dims from `param->dims()`, float output, tolerates 0 inputs | ✓ VERIFIED | `onComputeSize` reads `param->dims()`, sets `halide_type_of<float>()`, does not touch `inputs` at all (Const-like); `REGISTER_SHAPE(ShapeSGFP4Dequant, OpType_SGFP4Dequant)` present |
| `schema/default/MNN.fbs` | `OpType_SGFP4Dequant` enum tail entry + `SGFP4DequantParam` union entry | ✓ VERIFIED | `SGFP4Dequant = 605` appended after prior tail (`GridSample=604`); `SGFP4DequantParam` appended to `OpParameter` union tail |
| `schema/default/CaffeOp.fbs` | `table SGFP4DequantParam{magic; external:[int64]; dims:[int];}` | ✓ VERIFIED | Exact 3-field table present, no macroblock/quadtree fields |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `CPUSGFP4Dequant.cpp` | CPUBackend creator map | `REGISTER_CPU_OP_CREATOR(CPUSGFP4DequantCreator, OpType_SGFP4Dequant)` | ✓ WIRED | Present in source and confirmed propagated into `source/backend/cpu/CPUOPRegister.cpp` (manual +2-line append, verified via `git diff --stat` showing exactly `+2` insertions, no reordering of pre-existing entries) |
| `CPUSGFP4Dequant.cpp` | external `.mnn.weight` sidecar | `FileLoader(op->externalPath()) + offset(external()[0]) + read(size=external()[1])` | ✓ WIRED | Confirmed at `CPUSGFP4Dequant.cpp:66-87`; op-level test (`testOpLevelExternalSidecar`/`runSgfp4Module`) writes a real sidecar file and drives the op through `Module::load` → `onForward`, independently re-run and PASSED |
| `ShapeSGFP4Dequant.cpp` | SizeComputer registry | `REGISTER_SHAPE(ShapeSGFP4Dequant, OpType_SGFP4Dequant)` | ✓ WIRED | Present in source and in `source/shape/ShapeRegister.cpp` (manual +2-line append, verified) |

### Data-Flow Trace (Level 4)

The op-level test (`SGFP4DequantTest::runSgfp4Module`) is itself an end-to-end data-flow proof: a real fixture container is written to a temp `.mnn.weight` file on disk, an `OpT` with `SGFP4DequantParamT{magic, external=[0,size], dims}` and `externalPath` set to that file is built, run through `Module::load` → `CPUBackend` → `CPUSGFP4Dequant::onResize` (real `FileLoader` disk read) → `onExecute` (real `dequant_sgfp4_container_cpu` decode), and the resulting output tensor is compared element-by-element against the fixture's independently-computed `expected` weights. This was independently re-run (not just trusted from SUMMARY) and passed — confirming the data path is a real disk-file → decode → tensor flow, not a hardcoded/static return.

### Behavioral Spot-Checks / Full Test Execution

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| SGFP4 op/sgfp4 suite (all 5 test layers) | `./.build/Release/run_test.out.exe op/sgfp4` (independently re-run by verifier) | fixture round-trip (11 cases), ternary-reserved, FP16-precision, malformed-container negatives, op-level end-to-end — all PASSED | ✓ PASS |
| Python encoder self-test | `python tools/fp4/encode_sgfp4.py --selftest` (independently re-run by verifier) | 12/12 PASSED (all layouts x both modes, B=3, automatic mode selection), all "max wire-format diff=0" | ✓ PASS |
| Full regression suite | `./.build/Release/run_test.out.exe` (independently re-run by verifier) | `{"blocked":0,"failed":0,"passed":375,"skipped":0}` | ✓ PASS |
| E2M1 path byte-for-byte unchanged | `git diff abec332c HEAD -- include/MNN/FP4DequantUtils.hpp source/backend/cpu/CPUFP4Dequant.{hpp,cpp} tools/fp4/quantize_fp4.py` | 0 lines of diff | ✓ PASS |
| `schema/private/`, `source/internal/` untouched | `git log --oneline abec332c..HEAD -- schema/private source/internal` | no commits | ✓ PASS |
| `test/op/FP4ModelTest.cpp` not modified by this phase | `git diff abec332c HEAD -- test/op/FP4ModelTest.cpp` | 0 lines of diff (pre-existing breakage from unrelated `milestone` workstream commit `cffaf4bd`, documented in `deferred-items.md`, correctly out of scope) | ✓ PASS |

Note: the build binary at `.build/Release/run_test.out.exe` was verified to reflect the current committed tree — `git status` shows a fully clean working tree (only an untracked `.build/` directory, no modified/staged files) — so this is not a stale-binary false positive.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| SGV2-01 | 01-01 | Affine dual-mode decode, both code modes | ✓ SATISFIED | `sgfp4_decode_leaf_payload` (two's-complement mode 0, ternary mode 1 w/ 11→0), tested in fixture round-trip + ternary-reserved test |
| SGV2-02 | 01-01 | FP16 scale+bias unpack, packHalf2x16 order, 12-bit truncated bias | ✓ SATISFIED | `unpack_leaf_header`, tested in `testFp16LeafHeaderPrecision` |
| SGV2-03 | 01-01 | v2 self-framed stream parsing (magic/version/B/offset table, `sb_header` layout bits) | ✓ SATISFIED | `dequant_sgfp4_container_cpu` framing logic, tested via B=3 fixture + malformed-input tests |
| SGV2-04 | 01-01 | Uniform-layout record walk, deterministic leaf geometry, raster order, payload word counts | ✓ SATISFIED | `sgfp4_resolve_uniform_layout` + word-count arithmetic, tested across all 5 layouts x both modes |
| SGV2-05 | 01-01 | New FlatBuffers descriptor `{magic,offset,size}`, external sidecar via `FileLoader`, no macroblock/quadtree fields | ✓ SATISFIED | `SGFP4DequantParam` schema + `CPUSGFP4Dequant::onResize` |
| SGV2-06 | 01-01 | New dedicated CPU Execution class, additive to E2M1 | ✓ SATISFIED | `CPUSGFP4Dequant` registered under its own `OpType_SGFP4Dequant`; E2M1 confirmed byte-for-byte unchanged |
| SGV2-07 | 01-02 | Minimal Python encoder + CPU unit tests validating round-trip for both modes across all uniform layouts | ✓ SATISFIED | `tools/fp4/encode_sgfp4.py` (`--selftest`/`--emit-cpp-fixture`) + `test/op/SGFP4DequantTest.cpp` (`op/sgfp4/uniform_decode`), both independently re-run and passing |

No orphaned requirements: REQUIREMENTS.md's Phase 1 section lists exactly SGV2-01 through SGV2-07, and both PLAN frontmatters (01-01: SGV2-01..06, 01-02: SGV2-07) together cover all seven with no gaps or duplicates.

### Anti-Patterns Found

None. Grepped all phase-created/modified source files (`SGFP4DequantUtils.hpp`, `CPUSGFP4Dequant.{hpp,cpp}`, `ShapeSGFP4Dequant.cpp`, `encode_sgfp4.py`, `SGFP4DequantTest.cpp`) for `TBD|FIXME|XXX|TODO|HACK|PLACEHOLDER|not yet implemented|coming soon` — zero matches.

### Human Verification Required

None. All must-haves were verified programmatically, and the behavioral test suite (both the C++ `run_test.out op/sgfp4` and the Python encoder `--selftest`) was independently re-executed by the verifier (not merely trusted from SUMMARY.md) with fully passing results.

### Gaps Summary

No gaps. All 5 ROADMAP success criteria and all 7 requirement IDs (SGV2-01 through SGV2-07) are verified against the actual codebase: the schema is a clean tail-append with no macroblock/quadtree fields, the decode core correctly implements both affine code modes with proper bounds validation, the CPU Execution class correctly loads from an external sidecar (including a real bug found and fixed by Plan 01-02 in the DoS-bound file-size check), and the reference encoder + C++ test suite prove round-trip correctness for both modes across all five uniform layouts plus a non-multiple-of-4 alignment edge case. The existing E2M1 path is confirmed completely untouched across the full commit range. The one known pre-existing issue (`test/op/FP4ModelTest.cpp` compile breakage) is correctly out of this phase's scope, was not committed to, and is properly deferred to the `milestone` workstream's Phase 4 plan 04-02 per `deferred-items.md`.

---

_Verified: 2026-08-24T19:29:43Z_
_Verifier: Claude (gsd-verifier)_
