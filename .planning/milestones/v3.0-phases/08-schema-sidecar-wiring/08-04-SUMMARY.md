---
phase: 08-schema-sidecar-wiring
plan: 04
subsystem: converter
tags: [converter, external-data, sgfp4, alignment]

requires:
  - phase: 08-01 buffer field
    provides: SGFP4DequantParamT::buffer object-API field externalized here
provides:
  - storeSGFP4Container aligned sidecar emission in RemoveAndStoreParam (SGV2-23)
  - OpType_SGFP4Dequant read-back case in loadExternalParam (Q3 symmetry)
  - saveExternalData declaration in CommonUtils.hpp
affects: [08-06 converter round-trip test, Phase 11 PostConverter externalization flag machinery]

tech-stack:
  added: []
  patterns: [aligned (16-byte) external store with true-size descriptor; symmetric load-back]

key-files:
  created: []
  modified: [tools/converter/source/common/RemoveParams.cpp, tools/converter/source/common/CommonUtils.hpp]

key-decisions:
  "0": "Helper storeSGFP4Container defined ABOVE RemoveAndStoreParam (first draft placed it after — MSVC C3861)"
  1: "param->buffer is std::vector<int8_t> (flatc [byte]); helper signature matches — byte-correct via reinterpret_cast on write"
  2: "external records TRUE size; pad only advances the shared offset (D-06) — matches sgfp4_inject_core.hpp:377-389 exactly"
  3: "Read-back uses loadExternalData<int8_t> and clears external — mirrors Blob; guards external.size() != 2 with return"

patterns-established:
  - "Aligned external store pattern: trueSize write + zero pad to sgfp4_align16 + offset advance by aligned"

requirements-completed: [SGV2-23]

coverage:
  - id: D1
    description: "Converter externalizes SGFP4 buffer bytes to the shared sidecar aligned/monotonic/non-overlapping; symmetric read-back restores buffer"
    requirement: "SGV2-23"
    verification:
      - kind: unit
        ref: "TestSGFP4Converter.exe Phase A (layout: offsets, true-size, buffer cleared, zone bytes) + Phase B (reload decode == oracle) — exit 0"
        status: pass
      - kind: unit
        ref: "cmake --build .build --config Release (MNNConvertDeps compiles) — exit 0"
        status: pass
    human_judgment: false

duration: 30min
completed: 2026-08-28
status: complete
---

# Phase 08 Plan 04: converter SGFP4 externalization

**SGFP4 container bytes now externalize through the converter's shared sidecar machinery with the injection tool's exact alignment convention, plus symmetric read-back.**

## Performance

- **Duration:** ~30 min
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments
- `case OpParameter_SGFP4DequantParam` in `RemoveAndStoreParam` → `storeSGFP4Container`: writes trueSize bytes, zero-pads to `sgfp4_align16(trueSize)`, records `external = {offset, trueSize}`, clears buffer (swap idiom), advances offset by aligned size — same `fs`/`offset` thread as `saveExternalData` (no second sidecar/counter)
- `case OpType_SGFP4Dequant` in `loadExternalParam`: external.size()==2 guard, `fl->offset`, `loadExternalData<int8_t>` into buffer, external cleared (Q3 symmetry for `_postTreatOp`'s load-before-store ordering)
- Added the missing `saveExternalData` declaration to `CommonUtils.hpp` (function was defined but never declared — required by 08-06)
- No new converter flag (D-07): gating stays in writeFb.cpp postTreat's `config.saveExternalData`/`_largeModel`

## Deviations
- Plan asked for a storeWeight "variant"; implemented as a dedicated `storeSGFP4Container` static helper (storeWeight's template shape can't express pad-to-aligned + true-size descriptor) — same call-site position and semantics.

## Issues
- Compile-fix iteration: helper ordering (C3861) and vector type (uint8_t→int8_t) caught by the converter build gate and fixed before commit.
