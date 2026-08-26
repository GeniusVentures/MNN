---
phase: 02-adaptive-quadtree-layout-cpu-layout-mixed
plan: 02
status: complete
started: 2026-08-24
completed: 2026-08-24
duration_minutes: 35
requirements: [SGV2-10, SGV2-11]
---

# Plan 02-02 Summary: Adaptive quadtree encoder + mixed fixtures + mixed_decode tests

## What Was Done

### Task 1 — Encoder subdivision (`tools/fp4/encode_sgfp4.py`)

- Constants: `SPLIT_MAP_WORDS=3`, `SPLIT_MAP_BYTES=12`, `MAX_QUADTREE_BITS=85`,
  `QUADTREE_MIN_SPLIT_SIZE=8`, `LEVEL_THRESHOLDS` (geometric interpolation
  0.01@64 → 0.0005@4, tagged `[ASSUMED]` A1/A2), `VETO_FACTOR=3.0` (A3),
  `HYSTERESIS_DELTA=0.05` (A4).
- `apply_ternary_veto()`: blocks T158 when
  `max|w-mean| > veto_factor * mean|w-mean|`, else Eq. 5.
- `subdivide_macroblock()`: recursive error-driven subdivision — accept when
  per-element MSE ≤ per-level threshold, else split TL/TR/BL/BR subject to
  hysteresis (`sum(child_errs) < (1-δ)*parent_err`), recursion floor at 4x4.
  Returns `(x, y, n, mode, S, bias, codes)` leaves in DFS order.
- `build_split_map()`: recursive serializer — one bit per node ≥ 8 in
  pre-order DFS, bit k = word[k//32] bit k%32, `struct.pack('<3I', ...)`,
  asserts ≤ 85 bits.
- `classify_layout()`: all-same-size leaf sets with exact Table 3 count and
  area tiling collapse to the uniform enum (leaves reordered to raster);
  else `LAYOUT_MIXED` (DFS order kept).
- `encode_macroblock_mixed()` / `encode_macroblock_adaptive()` /
  `encode_container_adaptive()`: MIXED record emission
  (`sb_header + 12B map + headers + pad-to-16 + payloads`), uniform collapse
  routed through the unchanged uniform encoder (byte-identical), container
  framing with the `cursor % 16 == 0` assertion intact.

### Task 2 — Reference decoder + selftest + fixtures

- `_walk_split_map_ref()`: independent iterative DFS enumeration of
  `(x, y, n)` leaves (deliberately separate from `build_split_map`), with
  bit-exhaustion assertion.
- `decode_container_ref()`: MIXED branch — map at `rec_start+4`, 12B, walk,
  N headers + variable-size payloads in traversal order (`n²/8` / `n²/16`
  words, 16B padding); uniform path byte-identical.
- `selftest()` adaptive cases (all PASS, layout asserted):
  all-split ramp (amp 60) → `LAYOUT_FULL_4X4` collapse; constant →
  `LAYOUT_UNIFORM_64` collapse (NOT MIXED); asymmetric TL-ramp →
  `LAYOUT_MIXED` (4/8/32 leaf sizes); depth-mixed quadrants → `LAYOUT_MIXED`;
  multi-record (MIXED + uniform) container.
- `build_fixture_cases()` + regeneration: `mixed_allsplit` (layout 5
  collapse), `uniform_collapse` (layout 0 — encoder emits uniform, not
  MIXED), `mixed_asymmetric` (layout 4) — seeded rng 20260825,
  `mode = -1` (per-leaf adaptive).
- CLI knobs `--eps`, `--level-thresholds T64,T32,T16,T8,T4`,
  `--veto-factor`, `--hysteresis-delta` with validation, threaded through
  `emit_cpp_fixture`/`encode_container_adaptive`; docstring updated.

### Task 3 — C++ `op/sgfp4/mixed_decode` (`test/op/SGFP4DequantTest.cpp`)

- `LeafExpect` + `enumerateExpected()`: independent golden enumerator (own
  bit accounting, no shared code with `MNN::sgfp4_walk_quadtree`, D-05).
- `SGFP4MixedDecodeTest` registered `op/sgfp4/mixed_decode`:
  - `testGoldenTraversal()`: decodes `mixed_asymmetric`, enumerates expected
    leaves from the raw split map, verifies geometry tiling + block-for-block
    agreement with the encoder's DFS stream (64 leaves, mixed 4/8/32 sizes —
    also exercises 4x4-no-split-bit since the subtree bottoms out at 4).
  - `testMixedRoundTrip()`: `mixed_allsplit` / `uniform_collapse` /
    `mixed_asymmetric` decode == `expected` at 1e-4 relative tolerance.
  - `testMixedNegativeCases()` (D-09): (a) all-85-split-bits map → reader
    exhaustion rejects; (b) bit 0=1, bit 1=0, bits 2-4=1 → mid-tree
    exhaustion rejects; (c) container cut mid-payload rejects; (d)
    truncation right after the split map (lying leaf sizes) rejects.

## Files Modified

- `tools/fp4/encode_sgfp4.py` (encoder + ref decoder + selftest + fixtures + CLI)
- `test/op/SGFP4DequantFixtures.h` (regenerated: +3 adaptive fixtures)
- `test/op/SGFP4DequantTest.cpp` (+ `SGFP4MixedDecodeTest`, enumerator, negatives)

## Commits

- `b2a83969` feat(02-02): add adaptive quadtree encoder, mixed fixtures, and mixed_decode tests (SGV2-10/11)

## Key Decisions / Deviations

- **Constructive ramp tiles over scaled noise (deviation in fixture design,
  within discretion):** scaling random noise does NOT force splits — the
  affine fit's relative error is scale-invariant and hysteresis rejects
  marginal gains. Selftest/fixture tiles use linear ramps (which cannot be
  fit by any single affine grid) with tuned amplitudes: full ramp amp 60 for
  all-split (lower amps stall at mixed 4/8 sizes), TL-quadrant ramp amp 12
  for asymmetric MIXED (amp 6 fits at the root and collapses to uniform64).
  These amplitudes are deterministic given the seeded rng.
- Non-tiling-area negative (D-09b) is realized as a mid-tree bit-exhaustion
  map: a complete split bitmap always tiles exactly by construction, so the
  `area != 4096` defense-in-depth check is unreachable from pure bitmaps —
  the walk-failure rejection path is the observable behavior instead
  (documented in the test comment).
- Selftest's all-split case asserts the **normative collapse** to
  `LAYOUT_FULL_4X4` (spec §6.3: all-same-size leaf sets MUST use the uniform
  layout) — a MIXED all-split emission would be non-conformant.

## Issues Encountered

- Two `SyntaxError: name used prior to global declaration` (Python requires
  the `global` statement before any use of the names, including in
  `add_argument(default=...)`) — fixed by moving the declaration to the top
  of `main()`.
- Initial selftest design (scaled noise) produced uniform collapses instead
  of the intended MIXED trees — diagnosed via amplitude probe, fixed with
  ramp tiles (see Key Decisions).

## Verification Results

- `python tools/fp4/encode_sgfp4.py --selftest` → `SELFTEST PASSED`
  (uniform no-regression + 5 adaptive cases incl. collapse/MIXED assertions).
- `python tools/fp4/encode_sgfp4.py --emit-cpp-fixture
  test/op/SGFP4DequantFixtures.h` → exit 0; grep count for the three mixed
  fixture names = 9 (≥ 3 required); `kFixtures` entries carry
  `layout == 5/0/4` respectively.
- `./run_test.out op/sgfp4` → `SGFP4DequantTest: all layers PASSED` AND
  `SGFP4MixedDecodeTest: all layers PASSED`; suite result
  `{"passed":2,"failed":0}`.
- FP4ModelTest stub workaround applied for local builds and restored
  byte-for-byte (hash-verified; working tree clean for that file).

## Self-Check: PASSED

All task acceptance criteria re-verified: constants/symbols present;
`build_split_map` packs `<3I` with ≤85-bit assert; `classify_layout`
uniform-collapse works (collapse fixture emits layout 0, all-split emits
layout 5); ref decoder MIXED branch independent of `build_split_map`;
`select_mode` Eq. 5 unchanged (eps 0.10); CLI knobs validated; committed
fixtures include all three names; independent enumerator does not call
`sgfp4_walk_quadtree`; both test suites green.
