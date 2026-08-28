---
phase: 08-schema-sidecar-wiring
plan: 05
subsystem: testing
tags: [sgfp4, buffer-mode, parity, vulkan]

requires:
  - phase: 08-01 buffer field
    provides: schema surface for inline buffer
  - phase: 08-03 buffer dispatch
    provides: the buffer-first decode paths under test
provides:
  - op/sgfp4/dequant_buffer — CPU buffer-mode parity (all fixtures) + malformed-buffer rejection
  - op/sgfp4/vulkan_buffer_parity — Vulkan inline-buffer GPU parity vs CPU oracle
affects: [Phase 11 PostConverter single-file artifact validation]

tech-stack:
  added: []
  patterns: [buffer-mode op construction: buffer.assign, external empty, externalPath unset]

key-files:
  created: []
  modified: [test/op/SGFP4DequantTest.cpp, test/op/SGFP4VulkanDequantTest.cpp]

key-decisions:
  - "Buffer-mode op form: param->buffer.assign(container), external left EMPTY, externalPath UNSET — forces the buffer-first branch; no sidecar file written anywhere in either suite"
  - "Malformed probe: first 8 bytes of a valid container; acceptance = no usable decoded output (null module / empty forward / no data) — never partial output"
  - "Vulkan suite reuses D-07 pass-skip guard and the tight-then-relaxed two-pass structure; on this workspace a Vulkan device IS present and both passes ran"

patterns-established:
  - "Buffer-mode == sidecar-mode == oracle triangle asserted from identical container bytes"

requirements-completed: [SGV2-22]

coverage:
  - id: D1
    description: "CPU buffer-mode decode parity for all fixtures + malformed-buffer rejection"
    requirement: "SGV2-22"
    verification:
      - kind: unit
        ref: "run_test.out op/sgfp4/dequant_buffer — exit 0 (0.788 ms)"
        status: pass
  - id: D2
    description: "Vulkan buffer-mode GPU-vs-CPU-oracle parity (tight FP32 + relaxed default precision), no-device pass-skip"
    requirement: "SGV2-22"
    verification:
      - kind: unit
        ref: "run_test.out op/sgfp4/vulkan_buffer_parity — exit 0 (on-device, 14 fixtures, 1574 ms)"
        status: pass
    human_judgment: false

duration: 30min
completed: 2026-08-28
status: complete
---

# Phase 08 Plan 05: buffer-mode parity suites

**Buffer-mode decode == sidecar-mode decode == oracle proven on CPU and on-device Vulkan; malformed inline buffers rejected without partial output.**

## Performance

- **Duration:** ~30 min
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments
- `SGFP4DequantBufferTest` (`op/sgfp4/dequant_buffer`): iterates all 14 fixtures through CPU Module sessions built from inline buffers; parity vs fixture.expected within 1e-4; malformed probe (truncated 8-byte buffer) rejected cleanly
- `SGFP4VulkanBufferParityTest` (`op/sgfp4/vulkan_buffer_parity`): D-07 no-device guard + per-fixture CPU oracle + tight (Precision_High, 1e-4) and relaxed (default precision, 2e-3) GPU passes; ran ON DEVICE — 14 fixtures matched
- Both suites avoid any sidecar file I/O (pure single-file artifact form)

## Deviations
- None.

## Issues
- None.
