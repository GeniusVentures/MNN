---
plan: "09-04"
type: summary
requirements: [SGV2-24, SGV2-25]
commit: ff822e7c
fixtures: 7/7 pass
---

# Plan 09-04 Summary — CPU Decode-Parity Suite (with encoder byte-exactness fixes)

## What Was Built

`test/op/SGFP4EncodeTest.cpp` — suite `op/sgfp4/encode`:
- **Container framing**: magic + version gate on every encoded container
- **Fixture parity**: all 7 real-shape fixtures (100×36, 250×128, 37×91, 64×36, 5×5, 1×1, 128×64) — C++
  encode → CPU decode matches the Python `decode_v2` reference at rtol 1e-4; padded shapes via the `_crop`
  overload, aligned shapes direct
- **Security gates**: NaN / ±Inf / zero-dim / negative-dim / null → empty vector
- **All-zero input**: valid container decoding to zeros
- **LAYOUT_FULL_4X4 coverage**: deterministic LCG uniform-noise plane forces all-split to 4×4 leaves; hard
  assertion on the sb_header low 3 bits == 5 (verified against the Python encoder's layout distribution)

## Deviations

Three encoder defects surfaced during verification — fixed in this commit (also fixes the committed
Plan 09-01 code):
1. **`packNibbles` double allocation**: `out` was pre-sized to `wordCount*4` then `appendU32Le` PUSHED four
   more bytes per word → payload 2× oversized (4096 vs 2048 bytes) with a zero-run prefix. Fix: reserve +
   append. Containers now byte-identical to Python for every fixture (verified 100×36 and 250×128:
   0 diff bytes).
2. **FP16 conversion truncation vs RNE**: `struct.pack('<e')` rounds-to-nearest-even; half.hpp's default
   `HALF_ROUND_STYLE=-1` truncates → 1-ulp scale drift (SB1 of 100×36: 0x317a vs 0x317b). Subtlety: defining
   `HALF_ROUND_STYLE 1` before includes is NOT sufficient in a multi-TU binary — half.hpp is header-only,
   so the plain `half(float)` constructor is an inline function COMDAT-folded by the MSVC linker with
   truncating instantiations from other TUs (ODR violation; the macros compiled in the encoder TU were
   silently discarded at link time). Fix: `half_cast<half, std::round_to_nearest>(v)` — the explicit
   template argument instantiates a distinct template immune to folding.
3. **Stream vs spatial decode-convention mismatch** (`test/CMakeLists.txt` scope exceeded — also required
   shader-side change, see 09-05-SUMMARY): the legacy CPU oracle / Vulkan shader appended records as a
   leaf-major linear stream (`rec0 → out[0..4096)`), which equals the padded row-major plane ONLY for
   one-superblock-wide grids (tiles_x == 1). `shape_250x128` (first tiles_x=2 fixture, also shape_128x64)
   exposed it: Python's normative `decode_v2` reconstructs the padded plane SPATIALLY and crops.
   Fix: new `dequant_sgfp4_container_cpu_plane()` in SGFP4DequantUtils.hpp (additive; legacy stream oracle
   untouched for Phase 1–8 fixtures); `dequant_sgfp4_container_cpu_crop` now calls the plane decoder.
   Encode-suite aligned fixtures now use `_crop` semantics consistent with the plane convention.

Verification: `run_test.out op/sgfp4/encode` → passed 1/1; full `op/sgfp4` family 13/13 green.
