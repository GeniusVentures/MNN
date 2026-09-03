---
phase: 08-schema-sidecar-wiring
plan: 03
subsystem: sgfp4
tags: [runtime, vulkan, cpu-backend, buffer-dispatch]

requires:
  - phase: 08-01 buffer field
    provides: param->buffer() accessor consumed by both decoders
provides:
  - Buffer-first dispatch in CPUSGFP4Dequant::onResize (magic gate + eager oracle dims check)
  - Buffer-first dispatch in VulkanSGFP4DequantCreator::onCreate (magic gate, shared host pre-validation)
  - D-12 non-interception comment at OpCommonUtils::createExecutionWithExternal
affects: [08-05 buffer parity tests, Phase 11 PostConverter (single-file .mnn)]

tech-stack:
  added: []
  patterns: [buffer-first dispatch with external-sidecar fallback, single shared host pre-validation path]

key-files:
  created: []
  modified: [source/backend/cpu/CPUSGFP4Dequant.cpp, source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp, source/core/OpCommonUtils.cpp]

key-decisions:
  - "CPU buffer branch copies buf into mContainer (FlatBuffers buffer may point into the model buffer — never treated as owned storage, RESEARCH A2)"
  - "Q2 decision honored: eager dequant_sgfp4_container_cpu into scratch doubles as the dims-consistency check AND the buffer-mode replacement for T-01-04's file-size DoS bound (buffer already fully materialized)"
  - "Vulkan: container declared before dispatch; both branches funnel into the EXISTING host pre-validation — no duplicate validation code"
  - "Sidecar paths preserved verbatim as the empty-buffer else/fallthrough (D-04)"

patterns-established:
  - "SGFP4 data-placement dispatch: buffer-first, sidecar-fallback, decoder-owned (not OpCommonUtils-intercepted)"

requirements-completed: [SGV2-22]

coverage:
  - id: D1
    description: "CPU + Vulkan decoders dispatch buffer-first with magic/version + dims-consistency entry checks; empty buffer keeps the exact sidecar path"
    requirement: "SGV2-22"
    verification:
      - kind: unit
        ref: "run_test.out op/sgfp4/uniform_decode, op/sgfp4/mixed_decode (CPU sidecar unchanged) — exit 0"
        status: pass
      - kind: unit
        ref: "run_test.out op/sgfp4/vulkan_uniform_parity (sidecar path on device) + op/sgfp4/dequant_buffer + op/sgfp4/vulkan_buffer_parity (buffer path exercised by 08-05) — exit 0"
        status: pass
    human_judgment: false

duration: 35min
completed: 2026-08-28
status: complete
---

# Phase 08 Plan 03: buffer-first decoder dispatch

**Both SGFP4 decoders decode inline param->buffer directly (gated + oracle-checked); sidecar path unchanged; D-12 documented.**

## Performance

- **Duration:** ~35 min
- **Tasks:** 3/3
- **Files modified:** 3

## Accomplishments
- `CPUSGFP4Dequant::onResize`: buffer branch before the USE_EXTERNAL_DATA gate — `mContainer.assign`, `sgfp4_is_v2_container` entry gate, eager `dequant_sgfp4_container_cpu` scratch decode (INVALID_VALUE on failure, mContainer cleared); sidecar block verbatim below
- `VulkanSGFP4DequantCreator::onCreate`: restructured with `container` hoisted before dispatch; buffer branch assigns + gates, sidecar branch unchanged as the else; both funnel into the existing host pre-validation; `return new VulkanSGFP4Dequant(...)` untouched
- `OpCommonUtils.cpp`: D-12 comment above the `switch (op->main_type())` — SGFP4 intentionally absent from the auto-rewrite set (comment only, verified no executable change)
- Existing suites green: uniform_decode, mixed_decode, vulkan_uniform_parity (on-device FP32 tight + default-precision passes), plus the full op/sgfp4/ family

## Deviations
- None.

## Issues
- None.
