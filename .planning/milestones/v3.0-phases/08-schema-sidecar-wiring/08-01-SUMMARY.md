---
phase: 08-schema-sidecar-wiring
plan: 01
subsystem: schema
tags: [flatbuffers, sgfp4, schema-evolution]

requires:
  - phase: 07-04 structured fixtures
    provides: op/sgfp4 suites that serve as the D-04 regression gate
provides:
  - SGFP4DequantParam.buffer ([byte] FlatBuffers field, last position — table-append evolution rule)
  - Regenerated schema/current/CaffeOp_generated.h with buffer accessor/Builder/Object-API field
  - D-11/D-12 design comments on the SGFP4DequantParam table
affects: [08-03 buffer dispatch, 08-05 buffer tests, 08-06 converter externalization, Phase 11 PostConverter]

tech-stack:
  added: []
  patterns: [flatc table-append backward-compatible field addition]

key-files:
  created: []
  modified: [schema/default/CaffeOp.fbs, schema/current/CaffeOp_generated.h]

key-decisions:
  - "buffer appended as the LAST field (FlatBuffers Addition evolution rule) so existing sidecar-mode serialization stays byte-stable"
  - "D-11 buffer-staging and D-12 non-interception contracts documented on the table itself, not just in planning docs"
  - "MNN_generated.h regenerated content-stable (union/traits untouched by this field)"

patterns-established:
  - "Schema-change flow: edit .fbs -> powershell -File schema/generate.ps1 -> commit .fbs + regenerated current/*.h"

requirements-completed: [SGV2-22]

coverage:
  - id: D1
    description: "SGFP4DequantParam carries buffer:[byte] as last field; headers regenerated and committed"
    requirement: "SGV2-22"
    verification:
      - kind: unit
        ref: "run_test.out op/sgfp4/ (all 11 suites incl. classic_api D-04 regression) — exit 0"
        status: pass
    human_judgment: false

duration: 75min
completed: 2026-08-28
status: complete
---

# Phase 08 Plan 01: SGFP4DequantParam buffer field

**Inline decode source `buffer:[byte]` added to the SGFP4 schema; regenerated headers; all 11 op/sgfp4 suites green (D-04 sidecar regression confirmed).**

## Performance

- **Duration:** ~75 min (schema edit + flatc regen + full rebuild + suite run)
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments
- Appended `buffer:[byte]` as the LAST field of `table SGFP4DequantParam` (table-append backward compatibility)
- Regenerated `schema/current/CaffeOp_generated.h` via `schema/generate.ps1` (flatc built on demand from 3rd_party); `MNN_generated.h` regenerated content-stable
- Verified generated surface: `SGFP4DequantParamT::buffer` (std::vector<int8_t> — flatc [byte] → int8_t), `buffer()` accessor at vtable offset 10, `add_buffer`, `CreateSGFP4DequantParam(..., buffer)`
- Updated the locked table comment with the D-11 buffer-staging contract and D-12 non-interception note
- D-04 regression: full `op/sgfp4/` family (11 suites) green after rebuild — existing v2.0 injected artifact still loads/decodes via classic API (`op/sgfp4/classic_api`)

## Deviations
- None material. Plan's acceptance cited `std::vector<uint8_t> buffer`; flatc generates `std::vector<int8_t>` for `[byte]` (repo-wide convention, e.g. Blob.uint8s uses [ubyte]) — semantics identical, all consumers use byte-correct code (see 08-04/08-06 notes).

## Issues
- Build-tree PDB contention (MSVC C1041) from an interrupted parallel build required killing stale MSBuild node processes and deleting a stale PDB — environment issue, not a code issue.
