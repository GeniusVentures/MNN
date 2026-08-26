---
phase: 2
slug: adaptive-quadtree-layout-cpu-layout-mixed
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-24
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source: 02-RESEARCH.md `## Validation Architecture`.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | MNNTestSuite (`run_test.out`, monolithic binary) + Python `--selftest` oracle |
| **Config file** | none (tests auto-globbed via `test/CMakeLists.txt:12` `GLOB_RECURSE`) |
| **Quick run command** | `./run_test.out op/sgfp4` |
| **Full suite command** | `./run_test.out` (requires FP4ModelTest temporary-local-stub workaround to build) |
| **Estimated runtime** | ~seconds for `op/sgfp4` filter; minutes for full suite |

---

## Sampling Rate

- **After every task commit:** `python tools/fp4/encode_sgfp4.py --selftest` (encoder tasks) / `./run_test.out op/sgfp4` (decoder tasks)
- **Per wave merge:** `./run_test.out op/sgfp4` + `./run_test.out op/fp4` (E2M1 regression)
- **Phase gate:** full `./run_test.out` green before `/gsd-verify-work`

---

## Wave 0 Gaps (from RESEARCH.md)

- [ ] `test/op/SGFP4DequantTest.cpp` — add `op/sgfp4/mixed_decode` (golden traversal, mixed round-trip, negative split-map cases)
- [ ] `test/op/SGFP4DequantFixtures.h` — regenerate with mixed fixtures (all-split, uniform-collapse, asymmetric)
- [ ] `tools/fp4/encode_sgfp4.py` — extend `--selftest` to cover mixed/adaptive round-trip
