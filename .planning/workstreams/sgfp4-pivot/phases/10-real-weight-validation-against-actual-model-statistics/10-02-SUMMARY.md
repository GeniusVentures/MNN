---
phase: 10-real-weight-validation-against-actual-model-statistics
plan: "02"
subsystem: testing
tags: [sgfp4, fp4, encoder, cmake, parity, msvc]

requires:
  - phase: 09-real-weight-c-encoder-port
    provides: shipped sgfp4_encode static lib + byte-exactness baseline vs gnus-poc
provides:
  - sgfp4_encode_dump.out dump-driven C++ encode harness (D-11) under MNN_BUILD_SGFP4_TOOLS, linking sgfp4_encode
  - Standalone proof that dump→shipped-encoder→container is byte-exact vs fp4_exporter --adaptive on aligned + padded-path planes
affects: [10-03, phase-11-postconverter]

tech-stack:
  added: []
  patterns:
    - dump→container transducer harness mirroring sgfp4_inject.out target pattern but linking the encoder lib

key-files:
  created:
    - tools/fp4/sgfp4_encode_dump.cpp
  modified:
    - tools/fp4/CMakeLists.txt
    - tools/fp4/README.md

key-decisions:
  - "Harness writes the output file only after the full encode succeeds (failed/exit-2 runs never leave a container behind) — the tooling loop never sees a stale artifact"
  - "Windows subprocess invocation requires the .exe extension in the Python subprocess argv (plan smoke hit FileNotFoundError without it) — recorded for 10-03's driver wiring"
  - "Exit 2 proven via a 1×4 NaN dump: encoder-rejected input leaves no output file (Test-Path False)"

patterns-established:
  - Phase 9 seeded-plane parity smoke pattern (default_rng(seed), aligned + non-aligned shapes, filecmp.cmp shallow=False)

requirements-completed: [SGV2-27]

coverage:
  - id: D1
    description: "sgfp4_encode_dump.out target exists, builds clean, honors --weights/--dimO/--dimI/--out with exit contract 0/1/2"
    requirement: SGV2-27
    verification:
      - kind: other
        ref: "cmake --build .build --config Release --target sgfp4_encode_dump.out (clean build)"
        status: pass
      - kind: other
        ref: "usage → exit 1; NaN dump → exit 2 with no output file; success → one-line summary"
        status: pass
    human_judgment: false
  - id: D2
    description: "Byte-exactness smoke vs gnus-poc exporter on aligned (128×192) and non-64-aligned (100×300) deterministic planes"
    requirement: SGV2-27
    verification:
      - kind: other
        ref: "tmp throwaway smoke: filecmp.cmp byte-exact=True both shapes (12432B / 20704B)"
        status: pass
    human_judgment: false
---

# Phase 10 Plan 02: C++ Encode-Parity Harness Summary

One-liner: `sgfp4_encode_dump.out` — a dump-driven transducer that runs the shipped Phase 9 encoder on raw FP32 dumps and writes SGFP4 v2 containers, proven byte-exact against the gnus-poc exporter on both aligned and padded-path synthetic planes, de-risking 10-03 to pure wiring.

**Duration:** ~15 min | **Tasks:** 2 | **Files:** 3

## Accomplishments

- `tools/fp4/sgfp4_encode_dump.cpp`: minimal CLI harness (`--weights/--dimO/--dimI/--out`), exact dump-size check (`dimO*dimI*4`), `sgfp4_encode::encode` invocation, empty-vector → exit 2 with no output file, success → container write + one machine-parseable summary line (`dimO=%d dimI=%d container_bytes=%zu`).
- `tools/fp4/CMakeLists.txt`: target added to the `MNN_SGFP4_TOOLS` foreach (inherits `MNN_DEPS` + MSVC `/WHOLEARCHIVE`), with explicit `target_link_libraries(sgfp4_encode_dump.out sgfp4_encode)` — the parity-relevant distinction from `sgfp4_inject.out` noted in comments.
- Standalone smoke (throwaway, under tmp/): seeded planes 128×192 and 100×300 encoded via harness subprocess vs `export_weights(adaptive=True)` — **byte-identical containers both shapes**; transients cleaned afterward.
- README note documenting the harness's Phase 10 parity role.

## Verification Log

- Build: `cmake --build .build --config Release --target sgfp4_encode_dump.out` → clean (alongside untouched `sgfp4_encode` / `sgfp4_inject.out`).
- Exit contract: missing args → 1 with usage; NaN dump → 2, `Test-Path` on output → False; both synthetic encodes → 0.
- Byte-exactness: aligned 12432B == 12432B, non-aligned 20704B == 20704B, `filecmp.cmp(shallow=False)` True both.

## Deviations from Plan

**[Rule 2 – Build path] Phase 9 build directory is `.build`, not `build`** — Found during: Task 1 | Issue: plan's verify command uses `build`; the existing MSVC build tree lives at `.build/` (repo-root listing confirms `sgfp4_encode.vcxproj` etc. there). | Fix: built via `cmake --build .build`. 10-03's driver default must use the same path. | Files: none | Verification: clean build output above. | Commit: ec5e1557

**Total deviations:** 1 auto-fixed (environment path). **Impact:** none — same toolchain, same targets.

## Self-Check: PASSED

- [x] Target in `MNN_SGFP4_TOOLS` foreach; compiles warning-clean under Phase 9 MSVC settings
- [x] Exit-code contract 0/1/2 implemented; no output file after exit-2 (proven)
- [x] Both shapes byte-identical to exporter (no rtol fallback needed — recorded as expected clean case)
- [x] `sgfp4_encode` and `sgfp4_inject.out` untouched and still building; run_test.out suites untouched; zero decoder/gnus-poc changes (git diff scope verified)
- [x] Machine-parseable summary line confirmed for 10-03's subprocess loop

## Issues Encountered

None blocking.

## Next Phase Readiness

Ready for 10-03: harness path `.build/Release/sgfp4_encode_dump.out.exe` (`.exe` suffix required for Python subprocess on Windows).
