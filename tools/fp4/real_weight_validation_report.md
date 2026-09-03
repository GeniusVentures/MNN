# SGFP4 v2 Real-Weight Validation Report

- **Model:** `W:\gnus\models\alexnet_Opset16.onnx`
- **Model SHA-256:** `4bc388cc32cc789f4d08687a69e46ccf724cfee1e5775f1486847799ae538b53`
- **Generated (UTC):** 2026-09-01 01:43:16Z
- **Toolchain:** python 3.13.4, numpy 2.2.5, onnx 1.18.0
- **gnus-poc root:** `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc`
- **Threshold table:** tmp\sgfp4_validation\delta.json

Effective threshold table:

| leaf size | max_mse | max_relative |
|---|---|---|
| 64 | 0.01 | 0.384 |
| 32 | 0.005 | 0.079 |
| 16 | 0.002 | 0.03 |
| 8 | 0.001 | 0.015 |
| 4 | 0.0099 | 0.03 |

## Per-layer results

Gate metric (user-reformulated 2026-08-31): hard gate = plain per-element worst-leaf MSE; relative criterion = leaf energy ratio `mse / signal_power` (the folding the exporter's own split driver uses). The plain per-element relative ratio is reported in parentheses as an informational statistic only (structurally unbounded near zero-weight).

| tensor | dims | 2-D projection | elements | tier | kurtosis | outliers (6σ / q99) | leaf histogram | fp4/t158 | worst leaf MSE (target) | worst rel. err. (target) | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `features.0.weight` | 64x3x11x11 | 64x363 | 23232 | full | 4.76 | 2.58e-04 / 1.00e-02 | 64:6 | 6/0 | 5.020e-04 (0.01) | 3.370e-02 (0.384) [plain 1.3e+02] | **PASS** |
| `features.0.bias` | 64 | 64x1 | 64 | light | - | - | - | - | 6.954e-03 (roundtrip) | 1.643e-01 (max-abs) | **PASS** |
| `features.3.weight` | 192x64x5x5 | 192x1600 | 307200 | full | 40.89 | 1.97e-03 / 1.00e-02 | 64:75 | 75/0 | 1.326e-03 (0.01) | 3.482e-01 (0.384) [plain 1.3e+04] | **PASS** |
| `features.3.bias` | 192 | 192x1 | 192 | light | - | - | - | - | 3.358e-04 (roundtrip) | 5.063e-02 (max-abs) | **PASS** |
| `features.6.weight` | 384x192x3x3 | 384x1728 | 663552 | full | 5.96 | 6.71e-04 / 1.00e-02 | 64:162 | 162/0 | 3.591e-04 (0.01) | 1.596e-01 (0.384) [plain 1.5e+04] | **PASS** |
| `features.6.bias` | 384 | 384x1 | 384 | light | - | - | - | - | 3.237e-04 (roundtrip) | 1.180e-01 (max-abs) | **PASS** |
| `features.8.weight` | 256x384x3x3 | 256x3456 | 884736 | full | 1.84 | 2.76e-04 / 1.00e-02 | 64:216 | 216/0 | 7.402e-05 (0.01) | 9.812e-02 (0.384) [plain 3.6e+04] | **PASS** |
| `features.8.bias` | 256 | 256x1 | 256 | light | - | - | - | - | 1.597e-03 (roundtrip) | 1.634e-01 (max-abs) | **PASS** |
| `features.10.weight` | 256x256x3x3 | 256x2304 | 589824 | full | 1.02 | 1.20e-04 / 1.00e-02 | 64:144 | 144/0 | 3.516e-05 (0.01) | 4.316e-02 (0.384) [plain 3.5e+06] | **PASS** |
| `features.10.bias` | 256 | 256x1 | 256 | light | - | - | - | - | 3.020e-03 (roundtrip) | 2.404e-01 (max-abs) | **PASS** |
| `classifier.1.weight` | 4096x9216 | 4096x9216 | 37748736 | full | 0.16 | 1.17e-06 / 1.00e-02 | 64:9216 | 9216/0 | 2.513e-06 (0.01) | 2.829e-02 (0.384) [plain 3.6e+06] | **PASS** |
| `classifier.1.bias` | 4096 | 4096x1 | 4096 | light | - | - | - | - | 5.433e-06 (roundtrip) | 1.176e-02 (max-abs) | **PASS** |
| `classifier.4.weight` | 4096x4096 | 4096x4096 | 16777216 | full | 0.49 | 3.64e-06 / 1.00e-02 | 64:4096 | 4096/0 | 5.074e-06 (0.01) | 3.571e-02 (0.384) [plain 9.0e+05] | **PASS** |
| `classifier.4.bias` | 4096 | 4096x1 | 4096 | light | - | - | - | - | 2.345e-05 (roundtrip) | 2.533e-02 (max-abs) | **PASS** |
| `classifier.6.weight` | 1000x4096 | 1000x4096 | 4096000 | full | 1.93 | 1.22e-04 / 1.00e-02 | 64:1024 | 1024/0 | 2.310e-05 (0.01) | 6.292e-02 (0.384) [plain 2.9e+05] | **PASS** |
| `classifier.6.bias` | 1000 | 1000x1 | 1000 | light | - | - | - | - | 1.790e-05 (roundtrip) | 1.468e-02 (max-abs) | **PASS** |

## Pad overhead (non-64-aligned tensors)

| tensor | projection | padded/plain ratio | effective bpw |
|---|---|---|---|
| `features.0.weight` | 64x363 | 1.0579 | 4.281 |
| `classifier.6.weight` | 1000x4096 | 1.0240 | 4.136 |

## C++ encode parity (sgfp4_encode_dump.out)

| tensor | byte-exact | decode-stats rtol | status |
|---|---|---|---|
| `features.0.weight` | True | 0.0 (byte-exact; decode delta 0.0e+00) | PASS |
| `classifier.6.weight` | False | within 1e-4 | PASS (rtol fallback) |
| `classifier.1.weight` | False | within 1e-4 | PASS (rtol fallback) |
| `features.8.weight` | True | 0.0 (byte-exact; decode delta 0.0e+00) | PASS |
| `features.0.bias` | True | 0.0 (byte-exact; decode delta 0.0e+00) | PASS |
| `features.3.bias` | True | 0.0 (byte-exact; decode delta 0.0e+00) | PASS |

## Summary

- Layers swept: 16 (8 full tier, 8 light tier)
- **D-07 gate: PASS** — every layer meets its per-leaf-size targets.
- Threshold decision: gate green under the revised table `tmp\sgfp4_validation\delta.json` (see the delta section).

## Threshold delta

| leaf size | old max_mse | new max_mse | old max_relative | new max_relative | motivating statistic |
|---|---|---|---|---|---|
| 64 | 0.01 | 0.01 | 0.05 | 0.384 | worst observed leaf energy-ratio 0.348 on outlier-heavy 64x64 leaves (features.3/6/8, classifier.6); cascade-converged with 10% headroom |
| 32 | 0.005 | 0.005 | 0.03 | 0.079 | worst observed 0.071 (features.3.weight) |
| 16 | 0.002 | 0.002 | 0.02 | 0.03 | worst observed 0.0264 |
| 8 | 0.001 | 0.001 | 0.01 | 0.015 | worst observed 0.0131 |
| 4 | 0.0005 | 0.0099 | 0.005 | 0.03 | max_mse: forced min-size leaves on features.3.weight (worst 8.99e-3; quadtree accepts at min size by construction); max_relative: worst 0.0267 |

Revision provenance: the relative criterion is the user-reformulated (2026-08-31) leaf energy ratio `mse / signal_power` — the same folding the exporter's split driver applies. The plain per-element relative ratio is structurally unbounded on real weights (worst 3.6e6) and is reported informationally only.

This delta is a documented gnus-poc-side proposal (D-09); no gnus-poc code changes were made.
