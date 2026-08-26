---
phase: 03-vulkan-decode-uniform-layouts
plan: 02
subsystem: vulkan
tags: [glsl, compute-shader, sgfp4, dequant, spirv, makeshader]

requires:
  - phase: 03-vulkan-decode-uniform-layouts/03-01
    provides: "glslang toolchain in WSL + Vulkan build configuration + regeneration recipe"
provides:
  - "sgfp4_dequant.comp: uniform-layout SGFP4 v2 GLSL decode (both code modes, all 5 uniform layouts)"
  - "Embedded shader keys glsl_sgfp4_dequant_comp and glsl_sgfp4_dequant_FP16_comp in VulkanShaderMap"
affects: [03-03, 03-04]

tech-stack:
  added: []
  patterns: ["per-thread framing re-walk in GLSL (D-04): read_u32_le + sequential leaf payload cursor", "drop-in locateElement() seam reserved for the Phase-4 quadtree extension"]

key-files:
  created:
    - source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp
  modified:
    - source/backend/vulkan/buffer/execution/glsl/macro.json
    - source/backend/vulkan/buffer/compiler/AllShader.cpp
    - source/backend/vulkan/buffer/shaders/AllShader.h
    - source/backend/vulkan/buffer/compiler/VulkanShaderMap.cpp

key-decisions:
  - "kMagic written as assembled literal 0x34464753u with ASCII comment — GLSL has no character literals (deviation from the plan's 'S'|'G'<<8 form, value identical)"
  - "read_u32_le composes from at most two aligned words with an off==0 fast path; used for every container read"
  - "FP16 math done via unpackHalf2x16 only; affine reconstruction in FP32 with the FLOAT cast only at the store"

patterns-established:
  - "Thread-per-weight decode: guard is only idx >= outElementCount (D-05); magic/version checks stay host-side"

requirements-completed: [SGV2-12]

coverage:
  - id: D1
    description: "sgfp4_dequant.comp decodes uniform-layout SGFP4 v2 containers (5 layouts x 2 code modes) via shift-mask-FMA"
    requirement: SGV2-12
    verification:
      - kind: other
        ref: "glslangValidator prelude-copy compile: VALIDATE_EXIT=0 (10KB SPIR-V)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Both FP32 and FP16 variants embedded through the makeshader.py pipeline"
    requirement: SGV2-12
    verification:
      - kind: other
        ref: "grep counts: AllShader.cpp=4, AllShader.h=4, VulkanShaderMap.cpp=2 (glsl_sgfp4_dequant_comp + glsl_sgfp4_dequant_FP16_comp)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Numeric parity with the CPU reference decode"
    requirement: SGV2-12
    verification: []
    human_judgment: true
    rationale: "Numeric parity requires a live GPU dispatch through the Vulkan Execution class (plans 03-03/03-04); first proven by op/vulkan/sgfp4_dequant_parity in plan 03-04"

duration: 45min
completed: 2026-08-24
status: complete
---

# Phase 03 Plan 02: SGFP4 v2 Uniform-Layout GLSL Decode Shader Summary

Uniform-layout SGFP4 v2 decode shader ported to GLSL with all framing constants 1:1 from SGFP4DequantUtils.hpp, FP16 variant registered, and both variants embedded via the locked makeshader.py pipeline.

## Performance

- **Duration:** ~45 min
- **Tasks:** 2/2
- **Files:** 5 (1 created, 4 modified)

## Accomplishments

- `sgfp4_dequant.comp` (194 lines): `read_u32_le` for all container reads (B at byte 5 straddles words), `unpackLeafHeader` via `unpackHalf2x16`, dual-mode decode (`codeMode0` 4-bit two's-complement; `codeMode1` T158 ternary), per-thread framing re-walk with sequential leaf payload cursor (D-04), affine reconstruct `w = S*c + bias` with the FLOAT cast only at the store, sole guard `idx >= outElementCount` (D-05).
- All five uniform layout enums handled (0/1/2/3/5); enum 4 (MIXED) and >= 6 return false — matching host pre-validation.
- `macro.json` gained `"sgfp4_dequant.comp": { "useFP16": true }`; artifacts regenerated from WSL (`python3 makeshader.py`, glslang via shaderc interop).
- Verified embeddings: `glsl_sgfp4_dequant_comp` + `glsl_sgfp4_dequant_FP16_comp` in VulkanShaderMap (2 insertions), AllShader.h (4 declarations), AllShader.cpp (array + length symbols).

## Deviations from Plan

1. **[Rule 1 - portability] GLSL has no character literals** — `uint('S')` fails compilation. Fix: assembled literal `0x34464753u` with the `'S' | 'G'<<8 | 'F'<<16 | '4'<<24` derivation in the comment. Value identical to the plan's intent.
2. **[Rule 1 - portability] `layout` is a reserved keyword** — the walk's local variable was renamed `layoutEnum`. Found via glslang compile; fixed immediately.
3. **[Rule 3 - environment] regeneration entry-order churn** — as flagged in 03-01, the WSL regeneration reorders the ~139k lines of existing entries in the three artifacts (content-equivalent; only sgfp4 payloads + the 03-01 fp4_dequant fix bytes are semantic changes). Approved by user in the 03-01 checkpoint.

## Authentication Gates

None.

## Issues Encountered

None. `makeshader.py` exits 0 even on glslang failure — its log was grepped for `error` (0 hits) rather than trusting the exit code (recorded as deviation 4 in 03-01-SUMMARY).

## Next Plan Readiness

Ready for 03-03: shader keys `glsl_sgfp4_dequant_comp` / `glsl_sgfp4_dequant_FP16_comp` resolve at runtime via `vkBn->getPipeline(shaderName, types)`; descriptor layout is bindings 0=container SSBO (read), 1=dst SSBO (write), 2=uniform u32 pair {outElementCount, containerBytes}; dispatch is `UP_DIV(outElementCount, 256)`.

## Self-Check: PASSED

- No `#version` line, no FLOAT definition in the real file ✓
- Prelude copy compiles clean (VALIDATE_EXIT=0, SPIR-V emitted) ✓
- Every container read goes through `read_u32_le` ✓
- Constants match SGFP4DequantUtils.hpp exactly (kRecordCountOffset=5, kOffsetTableStart=16, kAlign16=16, kLayoutEnumMask=0x7, kLeafBiasMask=0xFFF0, kLeafModeBit=0x1, kNibblesPerWord=8, kSymbolsPerWord=16) ✓
- All 5 uniform enums handled; 4 and >=6 return false ✓
- local_size_x = 256 ✓
- macro.json entry with useFP16: true ✓; grep counts 4/4/2 ✓; Auto-Generated headers intact ✓
- Commit `71fc8518` contains exactly the 5 contract files ✓
