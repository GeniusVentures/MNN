---
phase: 02-adaptive-quadtree-layout-cpu-layout-mixed
plan: 01
status: complete
started: 2026-08-24
completed: 2026-08-24
duration_minutes: 25
requirements: [SGV2-08, SGV2-09]
---

# Plan 02-01 Summary: LAYOUT_MIXED quadtree split-map CPU decode

## What Was Done

Taught the header-only CPU decode core `dequant_sgfp4_container_cpu()` (in
`include/MNN/SGFP4DequantUtils.hpp`) to decode the previously-rejected
`LAYOUT_MIXED` (enum 4) record:

1. **Split-map constants** (spec §6.2): `kSGFP4SplitMapWords = 3`,
   `kSGFP4SplitMapBytes = 12`, `kSGFP4MaxQuadTreeBits = 85` (1+4+16+64),
   `kSGFP4QuadTreeMinSplitSize = 8`.
2. **`SGFP4SplitMapReader`** — bounds-checked bit reader (bit k = bit k%32 of
   LE word k/32); reads past 85 bits fail closed (T-02-01).
3. **`QuadNode {x, y, n}`** + **`sgfp4_walk_quadtree()`** — iterative pre-order
   DFS walk with an explicit fixed-size `QuadNode stack[85]`; TL pops first
   (BR/BL/TR/TL push order); nodes of size 4 never read a bit; `top + 4 > 85`
   stack guard (T-02-04) and `leafCount >= maxLeaves` guard.
4. **MIXED branch** in `dequant_sgfp4_container_cpu`: map span bounds-checked
   against `containerSize`, walk → `QuadNode leaves[256]`, strict tiling check
   `sum(n_i^2) == 4096` (T-02-02), `blockHeadersStart = recStart + 4 + 12`,
   per-leaf edge `n = leaves[leaf].n` feeding the unchanged
   `sgfp4_decode_leaf_payload()` (all existing per-leaf payload/output bounds
   checks retained). Uniform path (enums 0-3, 5) byte-identical;
   `sgfp4_resolve_uniform_layout` still rejects enum ≥ 6. Doc comment updated.

Tests (`test/op/SGFP4DequantTest.cpp`):
- Case (d) in `testMalformedContainers` replaced: now a deterministic
  malformed-MIXED negative (enum → MIXED + truncate at
  `recStart + 4 + kSGFP4SplitMapBytes`; rejected because
  `blockHeadersStart == containerSize`).
- `buildSingleLeafContainer` inserts the 12-byte all-zero split map for MIXED.
- New `testMixedDegenerateSmoke()`: all-zero map = single 64×64 leaf, decodes
  to `bias` everywhere.
- New `testMixedTraversalGolden()`: hand-built map `{0x00000001, 0, 0}` → four
  32×32 leaves with per-leaf bias markers 100/101/102/103 proves TL/TR/BL/BR
  pre-order DFS without the encoder (D-05 independence respected — the golden
  is hand-built, not derived from the walker).
- Refactor: shared `packLeafHeaderWord()` helper extracted; new
  `buildSplitFourLeafContainer()` helper.

## Files Modified

- `include/MNN/SGFP4DequantUtils.hpp` (+161 net lines: constants, reader,
  `QuadNode`, walker, MIXED branch, doc updates)
- `test/op/SGFP4DequantTest.cpp` (+~140 net lines: case (d) replacement, split
  map in fixture builder, 2 new test layers, 2 helpers)

## Commits

- `1c9e5633` feat(02-01): add LAYOUT_MIXED quadtree split-map decode (CPU, SGV2-08/09)

## Key Decisions

- Walker pops TL first by pushing BR, BL, TR, TL (reverse push) — a single
  generic rule, no quadrant-unrolling.
- MIXED validation order: map span → walk → area sum → block-header span →
  per-leaf payload/output bounds; every check before any dereference.
- `buildSplitFourLeafContainer` emits zero-filled padded payloads (0 codes →
  `w = bias_k`) so the traversal golden doubles as a payload-ordering check.

## Deviations

None.

## Issues Encountered

- Transient compile error from an accidental comment/brace gluing edit in
  `testMalformedContainers` — fixed immediately; final build clean.

## Verification Results

- `cmake --build .build --target run_test.out` — compiles and links cleanly.
- `./run_test.out op/sgfp4` — `SGFP4DequantTest: all layers PASSED`
  (11 fixture round-trips, degenerate MIXED, traversal golden, ternary
  reserved, FP16 header precision, malformed negatives incl. new MIXED case,
  op-level sidecar end-to-end). `"passed":1,"failed":0`.
- FP4ModelTest stub workaround applied for the local build and restored
  byte-for-byte (SHA-256 hash verified identical to backup; `git status` clean
  for that file).

## Self-Check: PASSED

All task acceptance criteria re-verified post-commit: constants present;
reader/walker signatures match; MIXED branch with `recStart + 4 +
kSGFP4SplitMapBytes`; uniform resolver unchanged; no recursion; C++11-clean
(aggregate `QuadNode` without default member initializers); test suite green.
