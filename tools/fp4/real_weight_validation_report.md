# SGFP4 v2 Real-Weight Validation Report

- **Model:** `W:\gnus\models\alexnet_Opset16.onnx`
- **Model SHA-256:** `4bc388cc32cc789f4d08687a69e46ccf724cfee1e5775f1486847799ae538b53`
- **Generated (UTC):** 2026-08-31 23:47:30Z
- **Toolchain:** python 3.13.4, numpy 2.2.5, onnx 1.18.0
- **gnus-poc root:** `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc`
- **Threshold table:** DEFAULT_V2_THRESHOLDS (exporter defaults)

Effective threshold table:

| leaf size | max_mse | max_relative |
|---|---|---|
| 64 | 0.01 | 0.05 |
| 32 | 0.005 | 0.03 |
| 16 | 0.002 | 0.02 |
| 8 | 0.001 | 0.01 |
| 4 | 0.0005 | 0.005 |

## Per-layer results

| tensor | dims | 2-D projection | elements | tier | kurtosis | outliers (6σ / q99) | leaf histogram | fp4/t158 | worst leaf MSE (target) | worst rel. err. (target) | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `features.0.weight` | 64x3x11x11 | 64x363 | 23232 | full | 4.76 | 2.58e-04 / 1.00e-02 | 64:6 | 6/0 | 5.020e-04 (0.01) | 1.337e+02 (0.05) | **FAIL** |
| `features.0.bias` | 64 | 64x1 | 64 | light | - | - | - | - | 6.954e-03 (roundtrip) | 1.643e-01 (max-abs) | **PASS** |
| `features.3.weight` | 192x64x5x5 | 192x1600 | 307200 | full | 40.89 | 1.97e-03 / 1.00e-02 | 64:68 32:12 16:9 8:57 4:652 | 798/0 | 8.993e-03 (0.0005) | 1.343e+04 (0.05) | **FAIL** |
| `features.3.bias` | 192 | 192x1 | 192 | light | - | - | - | - | 3.358e-04 (roundtrip) | 5.063e-02 (max-abs) | **PASS** |
| `features.6.weight` | 384x192x3x3 | 384x1728 | 663552 | full | 5.96 | 6.71e-04 / 1.00e-02 | 64:159 32:8 16:10 8:7 4:68 | 252/0 | 1.048e-03 (0.0005) | 1.464e+04 (0.05) | **FAIL** |
| `features.6.bias` | 384 | 384x1 | 384 | light | - | - | - | - | 3.237e-04 (roundtrip) | 1.180e-01 (max-abs) | **PASS** |
| `features.8.weight` | 256x384x3x3 | 256x3456 | 884736 | full | 1.84 | 2.76e-04 / 1.00e-02 | 64:216 | 216/0 | 7.402e-05 (0.01) | 3.582e+04 (0.05) | **FAIL** |
| `features.8.bias` | 256 | 256x1 | 256 | light | - | - | - | - | 1.597e-03 (roundtrip) | 1.634e-01 (max-abs) | **PASS** |
| `features.10.weight` | 256x256x3x3 | 256x2304 | 589824 | full | 1.02 | 1.20e-04 / 1.00e-02 | 64:144 | 144/0 | 3.516e-05 (0.01) | 3.503e+06 (0.05) | **FAIL** |
| `features.10.bias` | 256 | 256x1 | 256 | light | - | - | - | - | 3.020e-03 (roundtrip) | 2.404e-01 (max-abs) | **PASS** |
| `classifier.1.weight` | 4096x9216 | 4096x9216 | 37748736 | full | 0.16 | 1.17e-06 / 1.00e-02 | 64:9216 | 9216/0 | 2.513e-06 (0.01) | 3.585e+06 (0.05) | **FAIL** |
| `classifier.1.bias` | 4096 | 4096x1 | 4096 | light | - | - | - | - | 5.433e-06 (roundtrip) | 1.176e-02 (max-abs) | **PASS** |
| `classifier.4.weight` | 4096x4096 | 4096x4096 | 16777216 | full | 0.49 | 3.64e-06 / 1.00e-02 | 64:4096 | 4096/0 | 5.074e-06 (0.01) | 8.971e+05 (0.05) | **FAIL** |
| `classifier.4.bias` | 4096 | 4096x1 | 4096 | light | - | - | - | - | 2.345e-05 (roundtrip) | 2.533e-02 (max-abs) | **PASS** |
| `classifier.6.weight` | 1000x4096 | 1000x4096 | 4096000 | full | 1.93 | 1.22e-04 / 1.00e-02 | 64:1024 | 1024/0 | 2.310e-05 (0.01) | 2.861e+05 (0.05) | **FAIL** |
| `classifier.6.bias` | 1000 | 1000x1 | 1000 | light | - | - | - | - | 1.790e-05 (roundtrip) | 1.468e-02 (max-abs) | **PASS** |

## Pad overhead (non-64-aligned tensors)

| tensor | projection | padded/plain ratio | effective bpw |
|---|---|---|---|
| `features.0.weight` | 64x363 | 1.0579 | 4.281 |
| `classifier.6.weight` | 1000x4096 | 1.0240 | 4.136 |

## C++ encode parity (sgfp4_encode_dump.out)

SKIPPED — run with `--encode-dump <path>` to activate the C++ parity leg (wired in plan 10-03).

## Summary

- Layers swept: 16 (8 full tier, 8 light tier)
- **D-07 gate: FAIL** — layers failing their targets: `features.0.weight`, `features.3.weight`, `features.6.weight`, `features.8.weight`, `features.10.weight`, `classifier.1.weight`, `classifier.4.weight`, `classifier.6.weight`
- Failing leaf tuples are recorded in the JSON sidecar for the threshold-delta loop.

## Threshold delta

No data-justified revision (D-09): the effective table equals DEFAULT_V2_THRESHOLDS.
