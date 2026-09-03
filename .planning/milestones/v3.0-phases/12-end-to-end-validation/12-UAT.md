---
status: complete
phase: 12-end-to-end-validation
source: [12-01-SUMMARY.md, 12-02-SUMMARY.md]
started: 2026-09-02T12:00:00.000Z
updated: 2026-09-02T14:30:00.000Z
---

## Current Test

[testing complete]

## Tests

### 1. [automated] optimizeNetImpl aborts --sgfp4 conversions + cli.cpp null-guard (12-01 D2)
expected: MNNConvert corrupt.onnx --sgfp4 -> exit 1, no 'Converted Success!'; corpus --sgfp4 --dumpPass -> exit 0, 'ops 74 -> 82', 'Converted Success!'
result: pass
source: automated
coverage_id: 12-01/D2

### 2. [automated] One committed script invocation gates CPU + Vulkan vs FP32 baseline (12-02 D1)
expected: pwsh tools/fp4/e2e_validation.ps1 -Corpus W:\gnus\models\alexnet_Opset16.onnx -> PASS: cpu / PASS: vulkan / E2E VALIDATION: PASS, exit 0
result: pass
source: automated
coverage_id: 12-02/D1

### 3. [automated] Vulkan leg genuinely on Vulkan (12-02 D2)
expected: script stdout 'vulkan backend confirmed: backendType is 7'; vulkaninfo pre-check; exit 2 path on missing device
result: pass
source: automated
coverage_id: 12-02/D2

### 4. [automated] D-11 negative leg green inside gate invocation (12-02 D4)
expected: PASS: D-11 negative leg (corrupt + --sgfp4 -> exit 1, no 'Converted Success!')
result: pass
source: automated
coverage_id: 12-02/D4

### 5. RunNetPass bool contract + T10 sibling assertion (12-01 D1)
expected: .build\Release\TestSGFP4Converter.exe exits 0 with a PASS line; T10 asserts RunNetPass returns false on the NaN/Inf encode-failure net
result: pass
note: user-run output reviewed; 'conv_c empty container -- skipping' lines are the intentional T7/T10 NaN-fixture encode-failure contract firing (MNN_ERROR-level expected noise); final 'PASS (layout + reload parity + pass mechanics)' line present

### 6. D-12 flag-off byte-identical behavior (12-01 D3)
expected: MNNConvert flag-off corrupt model -> exit 0 (same messages as pre-change); flag-off corpus -> exit 0 + 'Converted Success!'
result: pass
note: rerun by Claude 2026-09-02: corrupt flag-off exit 0, corpus flag-off exit 0 with .mnn artifact produced (command syntax: --modelFile/--MNNModel)

### 7. No regression in SGFP4 suites (12-01 D4)
expected: .build\Release\run_test.out.exe op/sgfp4 -> passed:13 failed:0; TestSGFP4Converter green
result: pass
note: rerun by Claude 2026-09-02: passed:13 failed:0 skipped:0, exit 0; TestSGFP4Converter PASS (test 5)

### 8. Tolerances locked with recorded derivation (12-02 D3)
expected: e2e_validation.ps1 header LOCKED block shows TolAbs=10.14433 / TolRel=948.601606 with per-backend measured values, date, commit 54bbeaf8; README documents methodology (Phase 10 citation, text-dump floor caveat)
result: pass
note: verified by Claude 2026-09-02: locked literals + full derivation block (per-backend measured, 2026-09-01, commit 54bbeaf8, Phase 10 anchor, 1e-5 text-dump caveat); README has zero 'Ultra FP4' occurrences

### 9. D-10 diagnostics on every verdict line (12-02 D5)
expected: Gate output prints per-backend max-abs/max-rel with argmax indices, e.g. 'PASS: cpu max-abs=5.07216500E+000 (idx 533), max-rel=2.37302592E+002 (idx 573)'
result: pass
note: full gate rerun by Claude 2026-09-02: exact PASS lines with indices for cpu (idx 533/573) and vulkan (idx 533/638), D-11 negative leg PASS, 'E2E VALIDATION: PASS', exit 0

### 10. Codec correctness fixes: spatial decode + encoder split maps (12-02 D6)
expected: python decode_v2 on a C++ encoder container gives maxAbs=0.367087 / ratio=0.1648 (identical to python exporter's own roundtrip); run_test.out op/sgfp4 13/13; TestSGFP4Converter PASS
result: pass
note: independently reproduced by Claude 2026-09-02 on a FRESH 250x128 multi-tile MIXED weight (the defect-exposing tiles_x>=2 shape, seed 42): sgfp4_encode_dump.out.exe -> python decode_v2 maxAbs=0.087659 ratio=0.0116 (FP4-noise level) — PASS; suites green via tests 5/7

## Summary

total: 10
passed: 10
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none yet]
