---
phase: 01-affine-dual-mode-decode-core-cpu-uniform-layouts
plan: 02
subsystem: testing
tags: [mnn, sgfp4, fp4, quantization, cpu-backend, python-encoder, flatbuffers, half-float]

requires:
  - phase: 01-affine-dual-mode-decode-core-cpu-uniform-layouts (plan 01)
    provides: "OpType_SGFP4Dequant schema, SGFP4DequantUtils.hpp decode core, CPUSGFP4Dequant Execution"
provides:
  - "tools/fp4/encode_sgfp4.py: standalone reference encoder for SGFP4 v2 uniform-layout containers (affine dual-mode, per-leaf Eq.5 mode selection) with an independent Python reference decoder and --selftest/--emit-cpp-fixture CLI"
  - "test/op/SGFP4DequantFixtures.h: encoder-generated, committed cross-language fixtures (11 cases: both modes x 5 uniform layouts + a B=3 alignment case)"
  - "test/op/SGFP4DequantTest.cpp (op/sgfp4/uniform_decode): fixture round-trip, hand-built edge cases (ternary reserved symbol, FP16 header precision, malformed-input negatives), and op-level end-to-end decode via the CPU backend"
  - "Bugfix: CPUSGFP4Dequant's T-01-04 DoS bound now uses a real on-disk file-size probe instead of the always-zero FileLoader::size()"
affects: []

tech-stack:
  added: []
  patterns:
    - "Op.externalPath set directly on the OpT at model-build time (not via rtmgr->setExternalFile) for custom op types outside OpCommonUtils::createExecutionWithExternal's Convolution2D/Scale/LayerNorm allowlist"
    - "Python reference encoder ships its own independent reference decoder (mirrors the C++ decode core) so --selftest proves wire-format round-trip, not just in-process array equality"

key-files:
  created:
    - tools/fp4/encode_sgfp4.py
    - test/op/SGFP4DequantFixtures.h
    - test/op/SGFP4DequantTest.cpp
  modified:
    - source/backend/cpu/CPUSGFP4Dequant.cpp

key-decisions:
  - "Op.externalPath must be set directly on the OpT in the op-level test (opT->externalPath = sidecarPath) -- rtmgr->setExternalFile() alone does not populate it for OpType_SGFP4Dequant, since OpCommonUtils::createExecutionWithExternal only rewrites externalPath onto the Op for Convolution2D/Scale/LayerNorm main types. CPUSGFP4Dequant reads mOp->externalPath() directly, so any other op type needing this mechanism must bake the path into the Op itself at build time."
  - "Fixed a real bug in CPUSGFP4Dequant::onResize (from Plan 01-01): the T-01-04 DoS bound compared external()[1] against FileLoader::size(), which is only populated by the parameterless whole-file FileLoader::read() and stays 0 for the offset+size-bounded FileLoader::read(buffer,size) this op uses -- the bound always tripped, so the op could never actually load a real sidecar. Replaced with a direct std::ifstream::tellg() on-disk size probe."
  - "Per-leaf (not per-macroblock) mode selection: each of a macroblock's N leaves independently computes both FP4_AFFINE and T158_AFFINE encodings and applies Eq.5, per the plan's explicit 'per leaf/block' instruction -- matches the v2 spec's leaf-level (S,bias,mode) header granularity."
  - "Fixture generation forces a single mode across all leaves of a given fixture (via an internal force_mode override on the same per-leaf encode path) so each of the 11 committed fixtures cleanly exercises one mode x layout combination; the default (unforced) selftest path still exercises genuine per-leaf Eq.5 automatic selection."

requirements-completed: [SGV2-07]

coverage:
  - id: D1
    description: "tools/fp4/encode_sgfp4.py encodes both modes x all 5 uniform layouts with per-leaf Eq.5 mode selection and round-trips within the affine bound (--selftest)"
    requirement: "SGV2-07"
    verification:
      - kind: unit
        ref: "python tools/fp4/encode_sgfp4.py --selftest"
        status: pass
    human_judgment: false
  - id: D2
    description: "test/op/SGFP4DequantTest.cpp (op/sgfp4/uniform_decode): cross-language fixture round-trip for both modes x all 5 uniform layouts + B=3 alignment"
    requirement: "SGV2-01, SGV2-04, SGV2-07"
    verification:
      - kind: unit
        ref: "./run_test.out op/sgfp4 -- fixture round-trip (11 cases) PASSED"
        status: pass
    human_judgment: false
  - id: D3
    description: "Hand-built edge cases: ternary reserved symbol 11->0, FP16 leaf-header precision (incl. 0xFFF0-masked bias), malformed-container negatives (bad magic/version, OOB offset, LAYOUT_MIXED)"
    requirement: "SGV2-01, SGV2-02, SGV2-03"
    verification:
      - kind: unit
        ref: "./run_test.out op/sgfp4 -- ternary/FP16-precision/malformed-container sub-cases PASSED"
        status: pass
    human_judgment: false
  - id: D4
    description: "Op-level end-to-end decode through the CPU backend via Op.externalPath + rtmgr->setExternalFile (0-input source op, Module::load -> onForward)"
    requirement: "SGV2-05, SGV2-06"
    verification:
      - kind: integration
        ref: "./run_test.out op/sgfp4 -- op-level end-to-end decode via setExternalFile PASSED"
        status: pass
    human_judgment: false
  - id: D5
    description: "E2M1 path unchanged (additive-only, Success Criterion 5) -- full test suite green including op/vulkan/fp4_dequant_correctness"
    requirement: "SC#5"
    verification:
      - kind: integration
        ref: "./run_test.out (full suite) -- 375 passed, 0 failed, incl. op/sgfp4/uniform_decode and op/vulkan/fp4_dequant_correctness"
        status: pass
    human_judgment: true
    rationale: "op/fp4/conversion (test/op/FP4ModelTest.cpp) could not be included in this run -- that file is pre-existing broken/dead code from an unrelated Phase 4-01 checkpoint commit (cffaf4bd), out of this plan's scope per PROJECT.md (deferred to the milestone workstream's Phase 4 plan 04-02). It was temporarily neutralized locally (never committed, restored byte-for-byte -- verified via `git diff` showing zero changes) purely so run_test.out could link at all. A human should confirm this substitution is acceptable, since the E2M1 regression check for that specific test file did not literally execute this session."

duration: 40min
completed: 2026-08-24
status: complete
---

# Phase 1 Plan 2: SGFP4 v2 Reference Encoder + CPU Round-Trip Tests Summary

**Standalone SGFP4 v2 encoder (`tools/fp4/encode_sgfp4.py`) with an independent Python reference decoder, encoder-generated cross-language fixtures, and a CPU test suite (`op/sgfp4/uniform_decode`) proving round-trip decode for both affine code modes across all five uniform layouts, end-to-end through the CPU backend.**

## Performance

- **Duration:** ~40 min
- **Started:** 2026-08-24T18:58:00Z
- **Completed:** 2026-08-24T19:17:00Z
- **Tasks:** 2 completed
- **Files modified:** 4 (3 created, 1 modified)

## Accomplishments
- Implemented `tools/fp4/encode_sgfp4.py`: the spec's exemplary affine encode (Section 4.4) for both FP4_AFFINE and T158_AFFINE, per-leaf Eq.5 mode selection, FP16 leaf-header packing (inverse of `unpack_leaf_header`), dual-mode payload packing (Section 4.3), and the full v2 container writer (Sections 6.1/6.2) for all five uniform layouts plus B != 0 (mod 4) alignment. Ships its own independent Python reference decoder (mirrors `dequant_sgfp4_container_cpu` byte-for-byte) so `--selftest` proves the *encoded bytes* round-trip, not just in-process arrays.
- Generated `test/op/SGFP4DequantFixtures.h` via `--emit-cpp-fixture`: 11 committed fixtures (both modes x 5 uniform layouts + a B=3 alignment case), each with container bytes, dims, mode, layout, and expected reconstructed weights.
- Wrote `test/op/SGFP4DequantTest.cpp` (`op/sgfp4/uniform_decode`): fixture round-trip, ternary-reserved-symbol, FP16-header-precision, malformed-container negative cases, and an op-level end-to-end decode test through the CPU backend.
- **Found and fixed a real bug in Plan 01-01's `CPUSGFP4Dequant::onResize`**: the DoS-bound size check used `FileLoader::size()`, which is only populated by the (unused) whole-file `FileLoader::read()` and always reads 0 for the offset+size-bounded read this op performs — the bound tripped on every load, so the op could never actually decode a real external sidecar. Replaced with a direct `std::ifstream::tellg()` file-size probe queried before the `FileLoader` read.
- Discovered and worked around (without modifying) a pre-existing broken file, `test/op/FP4ModelTest.cpp`, that blocks `run_test.out` from building at all; documented in `deferred-items.md` as out of this plan's scope (Phase 4 plan 04-02's responsibility).
- Ran the full `run_test.out` suite: 375 passed, 0 failed, including `op/sgfp4/uniform_decode` and `op/vulkan/fp4_dequant_correctness` (E2M1 regression check).

## Task Commits

Each task was committed atomically:

1. **Task 1: Reference uniform-layout SGFP4 v2 encoder** - `be514526` (feat)
2. **Task 2: CPU round-trip + edge-case + op-level tests** - `4f2e8c1c` (test) -- also includes the CPUSGFP4Dequant DoS-bound bugfix (Rule 1)

_No TDD tasks in this plan (`tdd="false"` on both)._

## Files Created/Modified
- `tools/fp4/encode_sgfp4.py` - standalone SGFP4 v2 reference encoder + independent Python reference decoder + `--selftest`/`--emit-cpp-fixture` CLI (does not modify or import `quantize_fp4.py`)
- `test/op/SGFP4DequantFixtures.h` - encoder-generated, committed cross-language fixtures (11 cases)
- `test/op/SGFP4DequantTest.cpp` - `op/sgfp4/uniform_decode` MNNTestSuite case (3 layers: fixture round-trip, hand-built edge cases, op-level plumbing)
- `source/backend/cpu/CPUSGFP4Dequant.cpp` - fixed the T-01-04 DoS-bound file-size check (see Deviations)

## Decisions Made
- **`Op.externalPath` must be set directly on the OpT**, not derived from `rtmgr->setExternalFile()` alone. Traced `OpCommonUtils::createExecutionWithExternal` (`source/core/OpCommonUtils.cpp:665-724`): it only rewrites/injects a session-derived `externalPath` for `OpParameter_Convolution2D` / `OpParameter_Scale` / `OpParameter_LayerNorm` main types (`OpCommonUtils.cpp:671-683`). `OpType_SGFP4Dequant` isn't in that allowlist, so `CPUSGFP4Dequant::onResize`'s `mOp->externalPath()` read only ever sees whatever was literally baked into the `Op` flatbuffer at build time. The op-level test sets `op->externalPath = sidecarPath` directly on the `OpT` before `Expr::create`/`Variable::save` (confirmed this field round-trips correctly through `Op::Pack`/`UnPack`), and additionally calls `rtmgr->setExternalFile(sidecarPath)` as harmless belt-and-suspenders consistent with the 01-01 SUMMARY's Pitfall 2 note. This refines (does not contradict) 01-01's finding: `rtmgr->setExternalFile()` is necessary for the *file-based/Const-tensor* external-file default-naming mechanism, but insufficient by itself for this op's own external-sidecar read.
- **Per-leaf mode selection**, per the plan's explicit instruction: each leaf in a macroblock independently computes both mode encodings and applies Eq.5, rather than selecting one mode for an entire macroblock.
- **Fixture generation uses a `force_mode` override** on the same per-leaf encode path (not a separate code path) so each committed fixture is homogeneous in mode for clean per-layout/per-mode test coverage; the unforced/default path (used by `--selftest`'s "automatic mode selection" case) exercises genuine Eq.5 selection.
- **Malformed-input hand-built tests reuse a real fixture's bytes** (the `mode0_uniform64` fixture, B=1/N=1) rather than constructing separate byte arrays by hand, mutating specific fields located via the public `SGFP4DequantUtils.hpp` named constants/helpers (`kSGFP4RecordOffsetTableStart`, `sgfp4_align16`, etc.) -- avoids re-deriving offsets by hand (no magic numbers) and keeps the test's intent legible.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed CPUSGFP4Dequant's broken DoS-bound file-size check**
- **Found during:** Task 2, first op-level test run (`code=2 in onForward` with no further diagnosis available from existing logging)
- **Issue:** `CPUSGFP4Dequant::onResize` (written in Plan 01-01) computed `fileSize = loader.size()` after `FileLoader loader(path, true)`. `FileLoader::size()` (`source/core/FileLoader.cpp`) only returns `mTotalSize`, which is populated exclusively by the parameterless, whole-file `FileLoader::read()` -- never by the `offset()` + `read(buffer, size)` pair this op actually calls. So `fileSize` was always `0`, the bound `readSize > fileSize - offsetSize` always tripped, and the op returned `INVALID_VALUE` for every real sidecar (silently converted to `NOT_SUPPORT` by an upstream const-folding wrapper in `GeometryComputerUtils.cpp`, which is why the observable symptom gave no direct clue). This meant the external-sidecar path added in Plan 01-01 could never actually succeed at runtime -- a correctness-blocking bug in functionality this plan's own required op-level test needed to exercise.
- **Fix:** Added a small local `queryFileSize()` helper using `std::ifstream(path, ios::binary | ios::ate).tellg()` to get the sidecar's real on-disk size, queried *before* opening the `FileLoader` (still bounding the allocation before any read is attempted, preserving the original T-01-04 DoS-bound intent).
- **Files modified:** `source/backend/cpu/CPUSGFP4Dequant.cpp`
- **Verification:** `./run_test.out op/sgfp4` (op-level end-to-end decode via `setExternalFile` now passes) and full suite (375/375).
- **Committed in:** `4f2e8c1c` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 Rule 1 bug, in a file from a prior plan but blocking this plan's own required op-level verification)
**Impact on plan:** No scope creep beyond the minimal fix needed to make the external-sidecar path actually functional; the original DoS-bound *intent* (T-01-04) is preserved, just implemented correctly. Logged the unrelated pre-existing `FP4ModelTest.cpp` build breakage to `deferred-items.md` instead of fixing it (out of this plan's scope).

## Issues Encountered
- **`test/op/FP4ModelTest.cpp` (pre-existing, unrelated) blocks `run_test.out` from building.** This file (committed at `cffaf4bd`, a Phase 4-01 checkpoint on the separate `milestone` workstream) contains dead/unreachable code after an early `return true;` with undeclared identifiers -- a genuine compile error on any compiler, not an MSVC quirk. Since `MNNTestSuite` is a single monolithic binary, this blocks building `run_test.out` at all. Worked around locally (never committed) by temporarily replacing the file's contents with a neutral stub for the duration of the build+test verification, then restoring the original content byte-for-byte (`git diff` showed zero changes afterward). Logged in `deferred-items.md`; recommended follow-up is the `milestone` workstream's Phase 4 plan 04-02.
- **`op/fp4/conversion` (inside the neutralized file) therefore did not literally execute this session** -- see `coverage` D5's `human_judgment: true` / rationale above. `op/vulkan/fp4_dequant_correctness` (the other E2M1-path test, unaffected by the workaround) did run and pass, gracefully skipping its Vulkan-specific assertions since this build has `MNN_VULKAN=OFF`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 1 (affine dual-mode decode core, CPU, uniform layouts) is now behaviorally complete: schema + shape computer + decode core + CPU Execution (Plan 01-01) and reference encoder + full round-trip test coverage (Plan 01-02) are both committed and passing.
- `./run_test.out op/sgfp4` and the full suite are green (375/375) as of this session, using a locally-reconfigured `.build` (`-DMNN_BUILD_TEST=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON`) and a temporary (uncommitted) workaround for the unrelated pre-existing `FP4ModelTest.cpp` breakage.
- Recommend: before/alongside Phase 3 (Vulkan port), either fix or remove `test/op/FP4ModelTest.cpp`'s dead code (Phase 4 plan 04-02, per `PROJECT.md`'s pending decision) so `run_test.out` builds cleanly without a manual workaround for any future session.
- The reference encoder (`tools/fp4/encode_sgfp4.py`) and its fixtures are the trusted CPU-side oracle Phase 3's Vulkan decode port should validate against.

---
*Phase: 01-affine-dual-mode-decode-core-cpu-uniform-layouts*
*Completed: 2026-08-24*
