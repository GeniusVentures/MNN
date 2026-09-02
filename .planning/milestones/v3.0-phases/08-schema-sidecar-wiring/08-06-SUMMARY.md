---
phase: 08-schema-sidecar-wiring
plan: 06
subsystem: converter
tags: [converter, round-trip, sgfp4, testing]

requires:
  - phase: 08-04 converter externalization
    provides: storeSGFP4Container + loadExternalParam SGFP4 cases under test
  - phase: 08-02 test util
    provides: sgfp4_test::buildContainerUniform64/readBytes/tempPath/cwdPath
provides:
  - tools/converter/source/TestSGFP4Converter.cpp — standalone converter round-trip executable
  - TestSGFP4Converter CMake target (static + shared branches) linking MNNConvertDeps + MNN_DEPS
affects: [Phase 11 PostConverter externalization verification]

tech-stack:
  added: []
  patterns: [converter-side standalone test executable; classic-API reload assertion of externalized artifacts]

key-files:
  created: [tools/converter/source/TestSGFP4Converter.cpp]
  modified: [tools/converter/CMakeLists.txt]

key-decisions:
  - "RESEARCH Q1 resolved as planned: standalone executable under tools/converter/source/ — RemoveAndStoreParam/saveExternalData live in MNNConvertDeps (MNN_BUILD_CONVERTER only), unreachable from run_test.out (MNN_DEPS only); TestPassManager/TestConvertResult precedents are shared-libs-gated and absent in this static workspace"
  - "MSVC static link mirrors MNNConvert's /WHOLEARCHIVE chain; GNU/Clang and Apple branches mirrored for upstreamability"
  - "Phase B sets op->externalPath LITERALLY (D-12 non-interception — the only way SGFP4 resolves the sidecar) and reloads via classic Interpreter::createFromFile→createSession→runSession"
  - "Asserted zone-byte integrity (memcmp of each SGFP4 region vs source bytes) on top of the planned offset assertions"

patterns-established:
  - "Converter round-trip test pattern: NetT -> saveExternalData -> layout asserts -> serialize + literal externalPath -> classic-API reload -> oracle parity"

requirements-completed: [SGV2-23]

coverage:
  - id: D1
    description: "Converter round-trip: aligned/monotonic/non-overlapping sidecar; external == {offset, true-size}; buffers cleared; reload decode == oracle"
    requirement: "SGV2-23"
    verification:
      - kind: unit
        ref: ".build/Release/TestSGFP4Converter.exe — 'PASS (layout + reload parity)', exit 0"
        status: pass
      - kind: unit
        ref: "run_test.out op/sgfp4/ still exit 0 (11 suites; converter test does not pollute run_test.out)"
        status: pass
    human_judgment: false

duration: 45min
completed: 2026-08-28
status: complete
---

# Phase 08 Plan 06: converter round-trip test

**TestSGFP4Converter proves the 08-04 externalization end-to-end: aligned mixed-type sidecar layout + externalized-artifact reload decoding identically to the oracle.**

## Performance

- **Duration:** ~45 min
- **Tasks:** 2/2
- **Files modified:** 2 (1 created)

## Accomplishments
- Phase A (layout): NetT{SGFP4 64x64, SGFP4 64x128, Convolution2D(16 floats)} → `saveExternalData` → asserts op0 external=={0,trueSize0}, op1 external=={align16(trueSize0),trueSize1}, Conv offset==aligned0+aligned1, both buffers empty, sidecar length == aligned total + 64 bytes, and per-region byte-exact zone integrity
- Phase B (reload parity): single-op NetT externalized, serialized with literal `op->externalPath`, reloaded via classic Interpreter/Session, runSession NO_ERROR, output count + decode == oracle (rtol 1e-6)
- CMake: `TestSGFP4Converter` target added to BOTH the static branch (mirroring MNNConvert's MSVC /WHOLEARCHIVE / GNU whole-archive / Apple all_load link chains) and the shared branch; include dir wired to `test/op/` for SGFP4TestUtil.hpp
- Binary lands at `.build/Release/TestSGFP4Converter.exe` (CMAKE_RUNTIME_OUTPUT_DIRECTORY)

## Deviations
- Two compile-fix iterations (vector<int8_t> copy in makeSgfp4Op; saveExternalData takes `unique_ptr<NetT>&` not `NetT`) — caught by the plan's own build gate and fixed before commit; also required adding the missing `saveExternalData` declaration to CommonUtils.hpp (folded into 08-04's commit).
- Workspace builds Release (`.build`, `CMAKE_BUILD_TYPE=Release`); a Debug-config attempt hit the pre-existing libprotobuf Debug/Release link mismatch — used the established Release convention instead.

## Issues
- `test/op/FP4ModelTest.cpp` (pre-existing dead code, `milestone` WS, documented blocker) had to be temp-stubbed to build run_test.out; restored byte-identical post-build (`git diff --exit-code` = 0). Unchanged known debt, owner: milestone WS Phase 4 plan 04-02.
