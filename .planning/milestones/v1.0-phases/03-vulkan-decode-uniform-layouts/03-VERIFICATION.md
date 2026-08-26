---
status: passed
phase: 03-vulkan-decode-uniform-layouts
verified: 2026-08-24
must_haves_score: 3/3
requirements: [SGV2-12, SGV2-13, SGV2-14]
---

# Phase 03 VERIFICATION — Vulkan Decode — Uniform Layouts

Goal-backward verification: does the codebase actually deliver what the phase promised?

## Goal

A Vulkan buffer-backend GLSL Execution decodes uniform-layout SGFP4 v2 containers on GPU (FP4_AFFINE + T158_AFFINE, shift-mask-FMA), reading the same external-sidecar descriptor as the CPU path, with output matching the CPU reference decode within float tolerance.

## Must-Haves Verified

### 1. GLSL shader embedded via makeshader.py with regenerated artifacts committed — VERIFIED

- `source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp` (194 lines): `read_u32_le` for all container reads, `unpackHalf2x16` leaf headers, `codeMode0` (4-bit two's-complement) + `codeMode1` (T158 ternary), all five uniform layout enums (0/1/2/3/5; 4 and ≥6 rejected).
- Constants 1:1 with `include/MNN/SGFP4DequantUtils.hpp` (audited: kRecordCountOffset=5, kOffsetTableStart=16, kAlign16=16, kLayoutEnumMask=0x7, kLeafBiasMask=0xFFF0, kLeafModeBit=0x1, kNibblesPerWord=8, kSymbolsPerWord=16).
- Embedding (commit `71fc8518`): `glsl_sgfp4_dequant_comp` + `glsl_sgfp4_dequant_FP16_comp` present — grep counts AllShader.cpp=4, AllShader.h=4, VulkanShaderMap.cpp=2; Auto-Generated headers intact.
- Glslang compile proof: prelude-copy validation exit 0 (10,144-byte SPIR-V).

### 2. Execution registered, same sidecar descriptor as CPU path — VERIFIED

- `VulkanBackend::addCreator(OpType_SGFP4Dequant, new VulkanSGFP4DequantCreator)` via `static bool gResistor` (commit `d95cf8f8`).
- Descriptor path identical to `CPUSGFP4Dequant.cpp`: `op->main_as_SGFP4DequantParam()` + `USE_EXTERNAL_DATA(param)` + `op->externalPath()` + ifstream size probe BEFORE allocation + bounded `FileLoader` read.
- D-05 host pre-validation: `dequant_sgfp4_container_cpu` scratch pass; false → `nullptr` from creator (no upload/dispatch).
- Both code modes produce weights: parity test's mode0/mode1 fixtures cover both.
- On-the-GPU proof: fixtures execute through a real Vulkan module session (13 containers × 2 precision passes).

### 3. CPU/Vulkan parity within float tolerance via run_test.out — VERIFIED

- `op/sgfp4/vulkan_uniform_parity`: **passed:3 across the suite; 13/13 uniform fixtures** match `dequant_sgfp4_container_cpu` at rtol 1e-4 (Precision_High) + 2e-3 (default precision) on the RTX 4070 Ti SUPER (2026-08-24).
- Dual-direction truth: fixtures↔CPU (drift guard) and CPU↔GPU both asserted.

## Requirement Traceability

| Req | Plan(s) | Evidence | Status |
|-----|---------|----------|--------|
| SGV2-12 | 03-01, 03-02 | shader + embedding commits `71fc8518`; glslang proof; smoke test green | ✓ complete |
| SGV2-13 | 03-03 | registration + sidecar plumbing commit `d95cf8f8`; build green | ✓ complete |
| SGV2-14 | 03-04 | parity test commit `37f02b3b`; 13/13 pass on GPU | ✓ complete |

## Additivity Contract (E2M1 unchanged)

- `git diff --exit-code` clean on `VulkanFP4Dequant.{hpp,cpp}` after 03-03.
- `op/fp4` and `op/vulkan/fp4_dequant_correctness` green post-phase.
- EXCEPTION (user-approved, documented in 03-01-SUMMARY): two latent BUG FIXES in pre-existing `fp4_dequant.comp` (byte-vs-word indexing; e=3/m=1 NaN) were required to satisfy the smoke-test prerequisite — net behavior improvement, not a contract change; regenerated artifacts committed through the locked pipeline.

## Verification Debt / Warnings

1. Full-suite `run_test.out` run is blocked by the pre-existing `FP4ModelTest.cpp` dead code (milestone workstream Phase 4 plan 04-02's responsibility). All phase-scoped filtered suites (`op/sgfp4/`, `op/fp4`, `op/vulkan/fp4_dequant_correctness`) are green. The temp-stub workaround was used for builds and restored byte-for-byte every time (verified).
2. Graceful-skip branch (no Vulkan device) is code-audited but not live-exercised (no Vulkan-less machine available).
3. Regeneration entry-order churn: the three generated artifacts reordered ~139k lines (WSL find ordering ≠ original generator environment). Content-equivalent; future regenerations stable. Flagged for reviewer awareness in the `71fc8518` commit message.

## Verdict

**PASSED** — 3/3 must-haves verified against the live codebase; all three phase requirements delivered with executable evidence on physical GPU hardware.
