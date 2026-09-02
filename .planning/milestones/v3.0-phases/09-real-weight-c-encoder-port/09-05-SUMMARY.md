---
plan: "09-05"
type: summary
requirements: [SGV2-24, SGV2-25]
commit: ff822e7c
---

# Plan 09-05 Summary — Vulkan Encode-Parity Suite (D-08 complete)

## What Was Built

`test/op/SGFP4VulkanEncodeParityTest.cpp` — suite `op/sgfp4/vulkan_encode_parity`:
- **Aligned leg** (128×64): C++ encode → sidecar → Vulkan Session (Precision_High FP32 variant) → decoded
  output matches the Python reference at rtol 1e-4
- **Padded-crop leg** (100×36, 37×91, 64×36, 5×5, 1×1): all decode to TRUE dims (e.g. 3600 / 3367
  elements, never the padded count)
- **Crop-correctness probe** (`shape_100x36`): row-boundary check — `out[dimI]` must equal
  `expected[dimI]`, NOT `expected[paddedI]` (flat-prefix contamination guard, Pitfall 5)
- **Graceful no-GPU skip** via `MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN)` null check

Result: 7/7 fixtures + crop probe pass; full `op/sgfp4` family 13/13 green.

## Deviations (carried from the 09-04 diagnosis)

- **Vulkan shader spatial record mapping**: `locateElement` in `sgfp4_dequant.comp` previously walked
  records as a leaf-major stream (record 0 → out[0..4096), ...) — identical to the CPU oracle bug. Rewritten
  to map each padded-plane element SPATIALLY: `b = (row/64)*tilesX + col/64`, uniform-layout leaf via
  raster tile coords `((row%64)/n, (col%64)/n)`, MIXED via `local = (row%64)*64 + col%64` into the existing
  `locateMixedLeaf` walk. Mixes-derived `tilesX` from `uConst.paddedDimI` (5-field struct from Plan 09-02)
  with `B % tilesX` and `b < B` guards. `makeshader.py` regenerated `AllShader.cpp` (SPIR-V cache cleared
  for the changed shader; regen ran without spirv-opt after an opt failure fallback unlink).
- **Test-helper dangling pointer** (found the hard way — manifested as `0xdddddddd` canary reads on the
  WHOLE output): `runVulkanSession` returned `readMap<float>()` of a VARP destroyed at function exit
  (Express tensor memory returns to the allocator pool). Fixed by copying into caller-owned
  `outStorage.assign(outPtr, outPtr + outCount)` before return. Root-caused via a control experiment
  replaying the (passing) `mode0_uniform16_b3` fixture through this helper — same canary → helper bug,
  not session/shape.
- Diagnostic printouts from the debugging session (sidecar path print, divergence histogram, control
  experiment, `.bin`→`.mnn.weight` rename probe) were removed after root-causing.

Verification: `run_test.out op/sgfp4/vulkan_encode_parity` → passed 1/1; family run 13/13.
