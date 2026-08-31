---
status: complete
phase: 09-real-weight-c-encoder-port
source: [09-01-SUMMARY.md, 09-02-SUMMARY.md, 09-03-SUMMARY.md, 09-04-SUMMARY.md, 09-05-SUMMARY.md]
started: 2026-08-31T00:00:00Z
updated: 2026-08-31T00:00:00Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

[testing complete]

## Tests

### 1. C++ Encoder → Python-Oracle Parity (byte-exact containers)
expected: Run `.build/run_test.out op/sgfp4/encode` — passes 1/1; all 7 real-shape fixtures decode back to the Python `decode_v2` reference at rtol 1e-4; every container carries `SGF4` magic + version 0x02.
result: pass

### 2. Full SGFP4 Test Family (regression gate)
expected: Run `.build/run_test.out op/sgfp4` — all 13/13 tests in the family pass (Phase 1–8 suites unaffected by the encoder/decoder changes).
result: pass
verified: automated (2026-08-31, family run: "all <op/sgfp4> tests passed" — 13 passed / 0 failed / 0 skipped / 0 blocked)

### 3. Vulkan Encode-Parity incl. Padded-Crop Shapes
expected: Run `.build/run_test.out op/sgfp4/vulkan_encode_parity` — 7/7 fixtures pass through a real Vulkan session (Precision_High FP32); padded shapes (100×36, 37×91, 64×36, 5×5, 1×1) decode to TRUE dims (3600 / 3367 / ... elements, never padded counts); the 100×36 crop probe confirms the row-boundary check (`out[dimI] == expected[dimI]`, no flat-prefix contamination). On a machine without Vulkan, the suite skips gracefully.
result: pass
verified: automated (2026-08-31, family run: "7 fixtures (aligned + padded-crop) matched Python reference on Vulkan (rtol 1e-4)" + "'shape_100x36' crop probe PASSED" — real Vulkan device present, no skip)

### 4. Golden Fixture Regenerability
expected: Re-run `python tools/fp4/author_real_shape_fixture.py` — exits 0, framing gate (`SGF4` + 0x02) enforced, and the regenerated `test/op/SGFP4RealShapeFixtures.h` is byte-identical to the committed header (deterministic seeded RNG, no timestamps).
result: pass
verified: automated (2026-08-31: generator exit 0, all 7 fixtures emitted with sha256 provenance; regenerated header byte-identical to committed — Compare-Object clean, git status clean)

### 5. Encoder Input-Validation Gates
expected: Covered by the encode suite's security gates (NaN / ±Inf / zero-dim / negative-dim / null inputs → empty vector) and all-zero input (valid container decoding to zeros). Confirm suite green in Test 1, or spot-run the security-gate cases explicitly.
result: pass
verified: automated (2026-08-31, encode suite green in family run — security gates + all-zero case are assertions inside op/sgfp4/encode)

## Summary

total: 5
passed: 5
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none yet]
