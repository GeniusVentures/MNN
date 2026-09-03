---
plan: "09-03"
type: summary
requirements: [SGV2-24, SGV2-25]
commit: f67f9631
---

# Plan 09-03 Summary — Real-Shape Golden Fixture Generator (D-05)

## What Was Built

- `tools/fp4/author_real_shape_fixture.py`: deterministic generator cloning the
  `author_structured_fixture.py` pattern — per-shape seeded RNG
  (`default_rng(20260828 + dimO*1000 + dimI)`), `FP4Exporter.export_weights(..., adaptive=True)`
  (2-D array directly; no shape kwarg — verified against the exporter's calling convention), expected
  reference via the gnus-poc `decode_v2` oracle (crops to true dims itself),
  sha256 provenance comments, framing gate (exit non-zero unless `SGF4` + 0x02)
- `test/op/SGFP4RealShapeFixtures.h` (committed generated output): `namespace sgfp4_real_shape_fixtures`,
  `struct RealShapeFixture {name, container, containerSize, dimO, dimI, inputWeights, inputCount,
  expected, expectedCount}`, 7 fixtures: 100×36, 250×128, 37×91, 64×36 (one-dim-aligned), 5×5, 1×1
  (tiny/mostly-pad), 128×64 (fully aligned — Plan 09-05 aligned leg)

## Notes

- Regeneration verified byte-identical (second run diffed clean; no timestamps, no unseeded RNG).
- Fixture containers double as Python-oracle goldens for the Plan 09-04 byte-exactness work: post
  encoder fixes the C++ containers match these bytes exactly (spot-verified 100×36 and 250×128).
- The committed `expected` arrays encode the SPATIAL decode convention (from `decode_v2`) — the data
  that surfaced the stream-vs-plane decoder divergence fixed in 09-04/09-05.

Verification: generator exit 0; byte-identical regen; header compiles into `run_test.out` (family build).
