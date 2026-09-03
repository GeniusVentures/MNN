---
plan: "09-02"
type: summary
requirements: [SGV2-25]
commit: 89a903e8 (initial) / ff822e7c (spatial decode extension)
---

# Plan 09-02 Summary — Padded-Crop Decode Path (D-11a)

## What Was Built

- `dequant_sgfp4_container_cpu_crop()` in `include/MNN/SGFP4DequantUtils.hpp` (additive; the legacy
  `dequant_sgfp4_container_cpu` signature is UNCHANGED): decodes the padded plane then crops row-major
  `out[r*dimI+c] = scratch[r*paddedDimI+c]` — a stride crop, never a flat prefix (Pitfall 5)
- `CPUSGFP4Dequant` derives `mPaddedDimO/mPaddedDimI/mIsPadded` from `param->dims()` (`((d+63)/64)*64`)
  and dispatches buffer-mode validation + onExecute decode through the crop overload for non-64-aligned
  shapes; aligned path byte-identical to before
- `SGFP4DequantConst` (Vulkan) extended to 5 fields: `paddedOutElementCount`, `containerBytes`,
  `outDimO`, `outDimI`, `paddedDimI`; dispatch bounds over the padded plane
  (`UP_DIV(paddedOutElementCount, 256)`); creator pre-validates padded containers via the crop overload;
  Execution now carries true `{dimO, dimI}` (replacing the bare outElementCount)
- GLSL `sgfp4_dequant.comp`: pad-region invocations locate then early-return; write
  `Dst[row*outDimI+col]` for `row<outDimO && col<outDimI` only; backwards compatible (aligned dims
  collapse to `idx`). `makeshader.py` regen committed (`AllShader.cpp`; `AllShader.h` and
  `VulkanShaderMap.cpp` unchanged for this edit — the sgfp4 symbols were already registered and the
  header declarations byte-stable)

## Deviations

- `dequant_sgfp4_container_cpu_crop` was later rewired (commit ff822e7c) to call a new
  `dequant_sgfp4_container_cpu_plane()` spatial decoder rather than the legacy stream oracle — the
  stream append order diverges from the normative padded-plane convention for multi-superblock-wide
  grids (tiles_x ≥ 2). See 09-04-SUMMARY.md Deviations; legacy oracle untouched for Phase 1–8 fixtures.
- `locateElement` (shader) received the same spatial mapping during Plan 09-05 verification — see
  09-05-SUMMARY.md.

Verification: `run_test.out op/sgfp4` 11/11 green at commit time (13/13 after Wave 3); padded path
exercised end-to-end by Plans 09-04/09-05 suites.
