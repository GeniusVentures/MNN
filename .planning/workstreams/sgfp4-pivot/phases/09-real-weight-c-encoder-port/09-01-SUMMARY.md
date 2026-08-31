---
plan: "09-01"
type: summary
requirements: [SGV2-24, SGV2-25]
commit: 510fad98 (initial) / ff822e7c (byte-exactness fixes)
---

# Plan 09-01 Summary — SGFP4 v2 C++ Encoder Port (sgfp4_encode lib)

## What Was Built

- `tools/fp4/sgfp4_encode.hpp` — one-shot public API: `sgfp4_encode::encode(const float*, int, int) →
  std::vector<uint8_t>`; empty-vector contract for invalid input (non-finite weights, non-positive dims,
  dims > 65536)
- `tools/fp4/sgfp4_encode.cpp` — functional port of gnus-poc `fp4_exporter.py --adaptive` +
  `quadtree.py` + `laplacian.py`:
  - Separable Gaussian Laplacian pyramid (sigma=2·2^level, reflect boundary, truncate 4.0), hand-rolled
    per RESEARCH Pitfall 1 exception (no scipy in the C++ toolchain)
  - `_fit_affine` 16-candidate logspace scale search + `_fit_ternary`, all accumulation in double;
    quantization rounding via `std::rint` (numpy half-to-even parity)
  - Quadtree `_try_block` with DEFAULT_V2_THRESHOLDS, combined gate error, ternary preference + outlier
    veto, hysteresis slack 1.1 (dead 0.8 improvement code deliberately absent)
  - `_classify_layout` incl. LAYOUT_FULL_4X4 (D-11b), 3-word pre-order DFS split map, 16-byte-aligned
    record + container assembly, internal zero-padding of non-64-aligned planes
  - Framing constants REUSED from `MNN::kSGFP4*` (never redefined)
- CMake: `add_library(sgfp4_encode STATIC)` in tools/fp4 (explicit inject sources — glob double-compile
  anti-pattern removed); root include order fixed (SGFP4_TOOLS before TEST); `run_test.out` links
  `sgfp4_encode` under a `if(TARGET ...)` guard

## Deviations

- `test/op/FP4ModelTest.cpp` (documented pre-existing broken dead code from the `milestone` workstream,
  STATE.md blocker) had to be repaired to link `run_test.out`: the file contained an orphaned second
  `run()`-body fragment (unreferenced locals `outSz`/`refVec`/`sc`/`pk` at file scope after the first
  method's closing brace). Minimal fix: deleted the orphaned fragment; the intact
  `FP4ModelConversionTest` (op/fp4/conversion) is preserved and registered as before. Recommend the
  `milestone` workstream ratify or supersede.
- Byte-exactness fixes (packNibbles double-alloc, RNE FP16 half_cast with explicit-rounding template
  argument defeating MSVC COMDAT folding of inline conversions) landed in commit ff822e7c — see
  09-04-SUMMARY.md Deviations for the full diagnosis. Post-fix, C++ containers are byte-identical to the
  gnus-poc exporter for both spot-checked shapes (100×36, 250×128: 0 diff bytes).

Verification: builds clean (MSVC Debug, MNN_BUILD_SGFP4_TOOLS=ON + MNN_BUILD_TEST=ON); parity proven by
the Plan 09-04 suite (7/7 fixtures).
