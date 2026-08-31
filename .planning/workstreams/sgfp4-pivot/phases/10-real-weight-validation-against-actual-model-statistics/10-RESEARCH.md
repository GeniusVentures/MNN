# Phase 10: Real-Weight Validation Against Actual Model Statistics - Research

**Researched:** 2026-08-31
**Domain:** SGFP4 v2 encoder statistical validation on real model weights (Python driver + C++ encode-parity sampling)
**Confidence:** HIGH (codebase facts verified by direct read; corpus facts verified by on-disk inspection + live ONNX dumps + timed exporter probes)

## Summary

Phase 10 validates the Phase 9 C++ encoder's locked parameter set (`DEFAULT_V2_THRESHOLDS` split policy, dual code modes, layout emission, zero-pad-64) against a real model's full float32 weight set, and turns the thresholds into an explicit tunable via a config struct (D-08) if — and only if — the data demands revision. All twelve CONTEXT decisions (D-01..D-12) were treated as locked; this research *confirms* mechanics rather than re-deciding them.

The corpus search produced a concrete finding: **the gnus-poc repo itself contains no full-model float32 weights** — only one 512×512 synthetic demo artifact (`models/specialists_mlx/demo/fp4/demo.sgfp4`, provenance `base_model: ""`, generated 2026-08-26) and no `.safetensors`/`.onnx`/`.pt`/`.npz` files anywhere under gnus-poc. The specialist training pipeline (`training/train_specialists_mlx.py`) targets 30B Qwen3 bases via HF/MLX that were never checked out locally. The genuine real-model pool on this machine lives in `W:\gnus\models\`; two candidates qualify cleanly (AlexNet ONNX, super-resolution ONNX) and both were inspected live with `onnx` 1.18 — shapes dumped, element counts computed, and the gnus-poc exporter timed on representative tensors (full 61.1M-element AlexNet sweep ≈ 85–120 s single-threaded).

**Primary recommendation:** Approve **Candidate A: `W:\gnus\models\alexnet_Opset16.onnx`** (primary corpus, 16 FP32 tensors incl. 5 large 2-D FC planes and 5 4-D conv kernels, ~61.1M elements / ~233 MB) — with **Candidate B: `W:\gnus\models\super-resolution-10.onnx`** (~240 KB, 8 tensors, all sub-64 or 1-D dims) as the recommended *complementary* add-on rather than an alternative: it is the natural D-04 synthetic-non-aligned-covered-by-real-weights substitute, because every one of its weight tensors already exercises the padded/tiny tiers that AlexNet's aligned conv kernels miss.

## Validation Corpus Recommendation (D-02 — USER APPROVAL REQUIRED)

### Search scope (what was actually checked)

| Location | Result | Evidence |
|----------|--------|----------|
| `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\models\` | ❌ no full-model weights | recursive listing: only `specialists_mlx/demo/fp4/{demo.sgfp4, demo_stats.json, manifest.json}` (132,368 B container; shape 512×512; layout_distribution all-UNIFORM_64; manifest `base_model: ""`, `trained_at: null`) |
| gnus-poc repo-wide weight-file scan (`*.safetensors/*.onnx/*.pt/*.pth/*.bin/*.npz/*.npy/*.mnn`) | ❌ zero hits | recursive `Get-ChildItem -Include` returned empty |
| `W:\gnus\models\` | ✅ real models | `alexnet_Opset16.onnx` (244,407,335 B), `super-resolution-10.onnx` (240,078 B), `age_net.caffemodel` (45.7 MB), plus converted `.mnn` variants and a `.tflite` |
| MNN `benchmark/models/` | (present but already-converted `.mnn`, CNN-only, no ONNX source) | 8 `.mnn` files, 5.8–126 KB |

**Verdict on the CONTEXT's "gnus-poc specialist model" phrasing (D-01/D-02):** the *format* is canonical gnus-poc (the exporter/decoder under test), but no gnus-poc-trained specialist checkpoints exist on disk — the training outputs were never materialized locally. The demo 512×512 plane is synthetic program output, not trained weights; it fails D-01's "real model" bar. The corpus must therefore come from the local real-model pool; both candidates below are classic vision models with unambiguous public provenance, which satisfies D-02's provenance note requirement.

### Candidate A — AlexNet ONNX (PRIMARY) `[VERIFIED: onnx live dump]`

`W:\gnus\models\alexnet_Opset16.onnx` — 244,407,335 bytes, opset 16, all 16 initializers are `FLOAT` (FP32), totaling **61,100,840 elements (~233 MB FP32)**:

| Tensor | Dims | Flattened | 2-D projection (dimO×dimI) | 64-aligned? |
|--------|------|-----------|------------------------------|-------------|
| features.0.weight | [64, 3, 11, 11] | 23,232 | 64×363 | dimI 363 % 64 = 43 → **non-aligned** |
| features.0.bias | [64] | 64 | 64×1 | tiny tier |
| features.3.weight | [192, 64, 5, 5] | 307,200 | 192×1600 | dimO 192 = 3×64 ✓, dimI ✓ (as 2-D); per-[64,64] view fully aligned |
| features.3.bias | [192] | 192 | 192×1 | tiny tier |
| features.6.weight | [384, 192, 3, 3] | 663,552 | 384×1728 | aligned |
| features.6.bias | [384] | 384 | 384×1 | tiny tier |
| features.8.weight | [256, 384, 3, 3] | 884,736 | 256×3456 | aligned |
| features.8.bias | [256] | 256 | 256×1 | tiny tier |
| features.10.weight | [256, 256, 3, 3] | 589,824 | 256×2304 | aligned |
| features.10.bias | [256] | 256 | 256×1 | tiny tier |
| classifier.1.weight | [4096, 9216] | 37,748,736 | native 2-D, aligned | aligned |
| classifier.1.bias | [4096] | 4,096 | 4096×1 | 1-D tier |
| classifier.4.weight | [4096, 4096] | 16,777,216 | native 2-D, aligned | aligned |
| classifier.4.bias | [4096] | 4,096 | 1-D tier | |
| classifier.6.weight | [1000, 4096] | 4,096,000 | native 2-D | dimO 1000 % 64 = 40 → **non-aligned** |
| classifier.6.bias | [1000] | 1,000 | 1-D tier | |

- **Layer variety:** 5 conv kernels (4-D), 5 FC planes (native 2-D, incl. 37.7M-element), 6 bias vectors (tiny/1-D tier) — hits every D-03 tier in one model.
- **Padded-path coverage (D-04):** two genuinely non-64-aligned tensors (`features.0.weight` 64×363, `classifier.6.weight` 1000×4096) — the real-weight padded path is exercised without any synthetic fallback.
- **Scale range:** 64 to 37.7M elements — three orders of magnitude; kurtosis/outlier character will differ strongly between conv3×3 and FC planes, exactly the distribution variety SGV2-26 wants.
- **Extraction:** `onnx.load()` → `graph.initializer` raw_data / packed_field → `numpy.frombuffer` FP32; zero new deps (verified live: `onnx 1.18.0`, `numpy 2.2.5`, `python 3.13.4` all installed).
- **Provenance:** public torchvision AlexNet exported to ONNX opset 16; standard, unambiguous, license-clean for a statistical report (no weights committed — only derived statistics, per D-05).
- **Runtime cost (measured):** gnus-poc exporter probe on the two largest shapes: classifier.1-class (64×9216 slab) 0.82 s, 256×256 0.09 s, 64×64 0.01 s → full 61.1M-element sweep ≈ **85–120 s** single-threaded, well within phase budget. C++ encoder is byte-identical-and-faster; parity sampling adds minutes at most.

### Candidate B — Super-Resolution ONNX (complementary add-on) `[VERIFIED: onnx live dump]`

`W:\gnus\models\super-resolution-10.onnx` — 240,078 bytes, 8 FP32 initializers:

| Tensor | Dims |
|--------|------|
| conv1.weight | [64, 1, 5, 5] → 64×25 (non-aligned: 25 % 64 = 25) |
| conv1.bias | [64] |
| conv2.weight | [64, 64, 3, 3] → 64×576 (aligned) |
| conv2.bias | [64] |
| conv3.weight | [32, 64, 3, 3] → 32×576 (dimO 32 < 64 → single partial superblock row) |
| conv3.bias | [32] (tiny) |
| conv4.weight | [9, 32, 3, 3] → 9×288 (both dims non-aligned, tiny) |
| conv4.bias | [9] (tiny) |

- **Value:** every weight tensor is either non-64-aligned or tiny — it is the densest per-tensor padded/tiny coverage available; whole-model sweep runs in seconds. Using it alongside A means D-04's synthetic non-aligned fallback is likely unnecessary (real-weight non-aligned coverage already exceeds what a synthetic fixture would add).
- **Limitation as sole corpus:** only ~5 weight tensors above the tiny floor, narrow scale range, no native 2-D FC planes → fails D-01's "varied layer types/scales" spirit alone.

### Recommendation

**Approve Candidate A (AlexNet ONNX) as the primary validation corpus; strongly recommend also sweeping Candidate B (super-resolution) as a cheap complementary pass.** Together they cover: native 2-D FC + 4-D conv flattening, aligned + non-aligned + both-dims-non-aligned + sub-64 dims, 9-element to 37.7M-element tensors, and six 1-D bias vectors. If the user wants strictly one model (D-01's letter), A alone still covers every tier with real weights including two non-aligned tensors.

## Technical Approach

### 1. Weight extraction mechanics (confirming D-11; mechanics are Claude's discretion)

Least-friction path (verified live against both candidates):

```python
import onnx, numpy as np
m = onnx.load(r"W:\gnus\models\alexnet_Opset16.onnx")
for init in m.graph.initializer:
    arr = onnx.numpy_helper.to_array(init)        # FP32 ndarray, dims verbatim
    # 4-D [O, I, kh, kw] -> 2-D [O, I*kh*kw] via reshape (row-major, matches
    # tools/fp4 README dims convention dimO=shape[0], dimI=shape[1])
    # 1-D [O] -> [O, 1] degenerate plane (tiny tier)
```

- `onnx.numpy_helper.to_array` handles raw_data and external-data-free packed fields; both candidate files load with zero issues (verified: 16/8 initializers enumerated with correct dims).
- **Flattening rule (dims convention):** 4-D conv `[O, I, kh, kw]` → `[O, I·kh·kw]` — this is the same row-major flattening the MNN converter uses for Conv/InnerProduct weights and matches `tools/fp4/README.md`'s `dimO = shape[0]` convention. `[VERIFIED: README lines "dimO = shape[0] (output rows), dimI = shape[1]"]`
- **Python deps** (all already assumed/installed — no new dependencies): `onnx`, `numpy` (used by every `tools/fp4` script via gnus-poc imports), `pathlib/json/hashlib/argparse` stdlib. `torch 2.7.1+cpu` also present but NOT needed — avoid it; ONNX direct read is the cleanest provenance chain (file → initializer bytes → ndarray, no framework inference).
- Timing/storage: extraction holds one tensor at a time in memory (max 151 MB for `classifier.1.weight`); the raw-dump staging file for C++ parity sampling (see §5) is the only disk artifact (~tens of MB for sampled layers only).

### 2. Statistics design (confirming D-06; metric details are Claude's discretion)

Per-layer statistics computed in the Python driver on the **original FP32 weights** and on **decoded-vs-original error**:

| Metric | Definition (recommended) |
|--------|--------------------------|
| Weight histogram | 64 log-spaced bins over the tensor's value range; per-tensor (not global) edges |
| Kurtosis | Standardized Fisher kurtosis `E[((x-μ)/σ)⁴] − 3`, computed in float64 (`np.kurtosis(x, fisher=True)`); >10 flags outlier-heavy → T158/leaf-split pressure |
| Outlier share | fraction of \|x\| > μ + 6σ AND fraction of \|x\| > 0.99-quantile (both reported; the first is the quadtree-relevant "sparse spikes" indicator) |
| Per-layer MSE | `mean((orig − decoded)²)` over the **true (cropped) region**, float64 |
| Per-layer max relative error | `max(|orig − decoded| / (|orig| + 1e-12))` — elementwise, worst value |
| Leaf-size histogram | counts of emitted leaves per size ∈ {64, 32, 16, 8, 4} — taken from `QuadtreeEncoder.encode()`'s returned block list (`block["size"]`), called exactly as `fp4_exporter._export_v2_adaptive` calls it (per-64×64 superblock, same thresholds/ternary_delta/fit functions) `[VERIFIED: fp4_exporter.py lines ~294-321; quadtree.py _try_block returns dicts with "size"]`. The exporter's own `stats["layout_distribution"]` (superblock-level layout enum counts) is reported alongside per-superblock; note it is NOT the leaf histogram. |
| Code-mode mix | fp4 vs t158 leaf counts (from the same block lists; `block["mode"]`) |
| Pad overhead | `(ceil(dimO/64)·64 · ceil(dimI/64)·64) / (dimO·dimI)` as a ratio and the byte delta `effective_bpw(target − source)`; also `stats["effective_bpw"]` from the exporter. Non-64-aligned tensors only. `[VERIFIED: padding done inside both encoders: fp4_exporter.py _export_v2_adaptive `padded = np.zeros((tiles_y*64, tiles_x*64))`; same in sgfp4_encode.cpp]` |

Tiny-tensor tier (D-03, < 64 in a dim or single partial superblock): shape/framing check + roundtrip decode + max-abs error only — no distribution fit (statistics on <4K elements are noise). Reuse the Phase 9 fixture convention for the check but with fresh real extracts (1-D bias vectors land here naturally).

### 3. Per-layer gate mechanics (confirming D-07)

The exact gate the exporter itself applies `[VERIFIED: quadtree.py _try_block + _combined_gate_error, read in full]`:

1. For every leaf the quadtree emits, `selected_error` is the **Laplacian-weighted** error (`laplacian.compute(region, reconstructed, block_size)`), and the accept test is `combined_gate_error(region, selected_error, max_mse, max_relative) <= effective_max_mse` (with hysteresis slack 1.1 on non-accepted parents, forced accept at min size 4).
2. `_combined_gate_error` = `max(selected_error, max_mse · ((selected_error / signal_power) / max_relative))` where `signal_power = mean(region²)` in float64 — i.e. the relative gate is folded onto the MSE scale.
3. **Consequence for the driver:** the D-07 acceptance metric must be the **plain (unweighted) per-element MSE and relative error of the final decoded tensor vs the original weights**, compared against the per-leaf-size target of the leaves covering each element. This is deliberately *stricter and more honest* than re-deriving the Laplacian gate (the encode-side weighting is policy, not the shipped quality contract). Per-layer worst-MSE = max over leaves in that layer of `mean((orig−decoded)²) restricted to the leaf's true (cropped) footprint`; worst-relative likewise. Leaf footprints come from the same `QuadtreeEncoder.encode()` block lists (x, y, size per superblock), so no container re-parsing is needed.
4. **Threshold table locations (both repos):**
   - Python: `W:\...\gnus-poc\quantize\fp4_exporter.py` module-level `DEFAULT_V2_THRESHOLDS` dict `{64: {max_mse 0.01, max_relative 0.05}, 32: {0.005, 0.03}, 16: {0.002, 0.02}, 8: {0.001, 0.01}, 4: {0.0005, 0.005}}` `[VERIFIED: read directly]`
   - C++: `tools/fp4/sgfp4_encode.cpp` `constexpr Threshold kDefaultV2Thresholds[5]` — values identical `[VERIFIED: lines 444-450, read directly]`
5. **D-08 delta expression:** the driver writes any proposed table as a JSON blob keyed exactly like `DEFAULT_V2_THRESHOLDS` (`{"64": {"max_mse": ...}}`), and the report renders a three-column diff (size / old / new / motivating statistic). The C++ config struct consumes the same numbers (see §7).

### 4. C++ encode-parity sampling (confirming D-11)

What Phase 9 actually shipped (verified against `tools/fp4/CMakeLists.txt` and summaries):

- `tools/fp4/CMakeLists.txt` contains exactly two targets: `sgfp4_encode` (STATIC lib) and `sgfp4_inject.out` (explicitly does NOT link the encoder). **There is no existing dump-driven C++ harness** — Phase 9 parity ran via `run_test.out` suites (`test/op/SGFP4EncodeTest.cpp`, suite `op/sgfp4/encode`) consuming *committed* fixtures.
- The context therefore tasks research with the harness question: a small executable under `MNN_BUILD_SGFP4_TOOLS` that reads a raw FP32 weight dump + dims and writes the C++ container — mirroring `sgfp4_inject.out`'s existing target pattern (`add_executable ... sgfp4_inject.out; list(APPEND MNN_SGFP4_TOOLS ...)`; the foreach adds `MNN_DEPS` linking and MSVC `/WHOLEARCHIVE` handling). Gating: option `MNN_BUILD_SGFP4_TOOLS` at root `CMakeLists.txt:50`, included at line 961. `[VERIFIED]`

**Recommended harness design (fits existing pattern, no new app surface beyond what D-11 sanctions):**

```
tools/fp4/sgfp4_encode_dump.cpp   + entry in tools/fp4/CMakeLists.txt:
  add_executable(sgfp4_encode_dump.out ${CMAKE_CURRENT_LIST_DIR}/sgfp4_encode_dump.cpp)
  list(APPEND MNN_SGFP4_TOOLS sgfp4_encode_dump.out)   # inherits foreach linking
CLI: sgfp4_encode_dump.out --weights <f32 dump path> --dimO N --dimI M --out <container path>
Contract: raw little-endian FP32, dimO*dimI values, row-major; exit 0 + file written, or
non-zero + no file (empty-vector from encode() → exit 2 with MNN_ERROR). No model parsing.
```

Driver-side sampling loop (per sampled layer): write dump → invoke harness (subprocess, like `quantize_fp4.py` already shells out to MNN binaries `[VERIFIED: quantize_fp4.py imports subprocess]`) → byte-compare container vs `fp4_exporter.export_weights(..., adaptive=True)` output (**assert byte-exact — Phase 9 established byte-exactness as achieved, "0 diff bytes" on 100×36 and 250×128 `[VERIFIED: 09-01/09-04 summaries]`; fall back to the contractual rtol-1e-4 decode-vs-decode only if a platform rounding case surfaces, and then record it**) → decode both via `decode_v2` / `dequant_sgfp4_container_cpu_crop` and check C++ decode-error stats match Python-computed reference within rtol 1e-4 (`SGFP4EncodeTest.cpp` pattern, `kEncodeRelTol = 1e-4f` `[VERIFIED: test/op/SGFP4EncodeTest.cpp:32]`).

**Sampling plan (Claude's discretion per CONTEXT):** parity-check (a) every non-64-aligned tensor (full coverage of the risky path — in candidate A that's `features.0.weight`, `classifier.6.weight`; in B all four convs), (b) the largest aligned 2-D plane (`classifier.1.weight` — 37.7M elements, minutes not hours), (c) one aligned conv kernel, (d) 2–3 tiny/bias tensors. ≈ 8–10 sampled layers total; every layer still passes D-07 via the Python reference statistics regardless of sampling.

### 5. Report artifact + driver interface (confirming D-05/D-12)

**Driver:** `tools/fp4/validate_real_weights.py` (name per CONTEXT suggestion; final naming planner's call).

```
CLI:
  python tools/fp4/validate_real_weights.py \
      --model <path.onnx> \
      --report <out.md> \            # default tools/fp4/real_weight_validation_report.md
      --encode-dump <path> \         # sgfp4_encode_dump.out; omit → skip C++ parity leg
      --sample <tensor-name> ... \   # parity sample list; default: auto (rule above)
      --gnus-poc-root <path>         # default W:/gnus/GeniusCognitiveSystem/GNUS-NEO-SWARM/gnus-poc
Exit codes: 0 = all layers pass gate + parity OK
            1 = usage/IO error          2 = NaN/Inf or unreadable tensor (see Risks)
            3 = D-07 per-layer gate FAILURE (report still written, failing layers listed)
            4 = C++ parity mismatch
Inputs : ONNX file, gnus-poc repo (import path), optional dump-harness binary
Outputs: committed markdown report + sidecar JSON (machine-readable gate results,
         proposed-threshold delta if any) under tools/fp4/
```

**Report shape (markdown):** header (model, date, tool hashes, exporter/C++ encoder commit) → per-layer table (name, dims→2-D projection, elements, tier, kurtosis, outlier share, leaf-size histogram, code-mode mix, worst leaf MSE vs target, worst relative vs target, PASS/FAIL) → pad-overhead sub-table (non-aligned layers) → C++ parity sampling table (layer, byte-exact yes/no, decode rtol, PASS/FAIL) → summary (gate result, any D-08 proposed delta with motivating stats, D-09 documented-delta block for gnus-poc).

### 6. Config struct (confirming D-08; structure is Claude's discretion)

Grow `tools/fp4/sgfp4_encode.hpp` additively — Phase 9 D-10's deferral resolves here:

```cpp
namespace sgfp4_encode {
struct EncodeConfig {
    // Per-leaf-size gates, indexed by leaf size; defaults identical to
    // gnus-poc DEFAULT_V2_THRESHOLDS (fp4_exporter.py).
    struct Gate { double maxMse; double maxRelative; };
    Gate leafGates[5];        // [i] ↔ leaf size 64/32/16/8/4 (kDefaultV2Thresholds order)
    // Future knobs deliberately absent (D-10: thresholds only).
};
static const EncodeConfig kDefaultEncodeConfig;   // = kDefaultV2Thresholds values

// Existing one-shot overload stays EXACTLY as shipped (Phase 9 D-10 constraint:
// knob-less overload unchanged; defaults Python-identical):
std::vector<uint8_t> encode(const float* weights, int dimO, int dimI);
// New:
std::vector<uint8_t> encode(const float* weights, int dimO, int dimI, const EncodeConfig& config);
}
```

- Internally: the one-shot overload forwards to the config overload with `kDefaultEncodeConfig`; the quadtree context's hard-coded `kDefaultV2Thresholds[5]` lookup (`sgfp4_encode.cpp` ~line 491) becomes a config-carried array. No behavior change when defaults are used — **the 13/13 green suites (incl. `op/sgfp4/encode` byte-exactness fixtures) must stay green without modification**, which is the overload-compatibility proof.
- Overload (not default-argument parameter): keeps the existing signature link-compatible and makes "which call site uses tuned values" greppable.
- If no layer fails the gate, the struct still lands (it is cheap, unblocks Phase 11 tuning, and closes the Phase 9 D-10 deferral) — but with defaults only and an explicit report note "no data-justified revision".

### 7. Risks

1. **NaN/Inf in real weights** — encoder contract returns empty vector (`sgfp4_encode.cpp` ~line 752-755, ASVS V5 gate `[VERIFIED]`); Python exporter raises in `QuadtreeEncoder.encode` (`raise ValueError("superblock contains NaN or Inf values")` `[VERIFIED: quadtree.py]`). Driver policy: pre-scan each tensor with `np.isfinite().all()`; any failure → hard exit 2 with the tensor named. Real ONNX CNN checkpoints essentially never carry non-finite weights; if one does, that's a corpus problem (switch tensor/model), not an encoder problem.
2. **First-run gate failures** (expected to be rare but possible): the expected failure mode is 4×4-forced leaves on outlier-heavy conv planes exceeding `max_relative 0.005` — at min leaf size the quadtree force-accepts regardless of error (`if size <= min_block_size: accept = True` `[VERIFIED: quadtree.py]`), so a "fails the thresholds' own targets" layer is possible by construction. Revision flow (already locked D-08/D-09): driver collects failing (layer, size, error) tuples → proposes minimal delta table → re-run driver with `--thresholds <delta.json>` → report records before/after. Budget one re-run iteration; the driver must accept an override thresholds file from day one (it just forwards to `export_weights(thresholds=...)`, a first-class exporter parameter `[VERIFIED: export_weights signature]`).
3. **Scale sensitivity of the relative gate:** `combined_gate_error` divides by `signal_power = mean(region²)`; near-zero-mean planes with tiny magnitudes (bias-like) make the relative term explode. This is encoder-inherent behavior, not a driver bug — the report should annotate such layers rather than the gate silently passing on the `signal_power <= 1e-12` ε-escape `[VERIFIED: _kRelativeEpsilon = 1e-12]`.
4. **Time/disk:** full AlexNet sweep ≈ 2 min Python (measured extrapolation), C++ parity sampling adds ≈ 3–5 min incl. subprocess round-trips; sampled raw dumps ≈ 160 MB max transient (one layer at a time, deleted after use); report + sidecar ≈ tens of KB committed. No risk.
5. **Tiny-tensor floor choice:** 1-D bias vectors `[N]` mapped to `[N,1]` planes are >99% padding — valid inputs (encoder zero-pads) but statistically meaningless. The D-03 floor must place them in the light tier (`dimI == 1` or `elements < 4096`); floor rule is planner's to lock from this recommendation.
6. **Corpus provenance drift:** `W:\gnus\models\alexnet_Opset16.onnx` is an untracked local file. Mitigation: the report records sha256 of the ONNX file + per-tensor element counts; D-05 already forbids committing weights. Flag for user: if they prefer a corpus file that is tracked or re-procurable from a public source, say so at approval time.
7. **ONNX external-data edge:** both candidates use inline initializers (verified). If a future corpus file uses external data, `onnx.load_model(..., load_external_data=True)` handles it — one-line fallback, no design impact.

## Recommended Plan Structure

**10-01 — Corpus approval + extraction & statistics driver (Wave 1)**
Extract ONNX initializers; per-layer distribution stats (histograms, kurtosis, outlier share); leaf-size histogram + code-mode mix via `QuadtreeEncoder`; pad-overhead computation; report skeleton; D-07 gate evaluation on ALL layers via Python reference (SGV2-26). Produces the descriptive half of the committed report. *Blocked on user corpus approval (D-02) — the only human gate in the phase.*

**10-02 — C++ encode-parity harness `sgfp4_encode_dump.out` + sampling integration (Wave 1, independent of 10-01's stats work; needs 10-01's extraction utilities at integration time)**
New dump-driven target under `MNN_BUILD_SGFP4_TOOLS` mirroring `sgfp4_inject.out` pattern; driver `--encode-dump` leg: sampled layers byte-exact vs exporter + decode rtol 1e-4 both directions (SGV2-27 partial). Adds the parity tables to the report.

**10-03 — Config struct + (conditional) threshold delta (Wave 2, after 10-01 gate results and 10-02 parity)**
`EncodeConfig` + overload in `sgfp4_encode.hpp/.cpp`; defaults Python-identical; existing suites green unchanged. If gates failed: minimal data-justified delta + `--thresholds` re-run loop + D-09 documented-delta block in the report. Finalizes the committed report artifact; closes Phase 9's D-10 deferral (SGV2-27).

Dependencies: 10-01 ∥ 10-02 (Wave 1) → 10-03 (Wave 2). Three plans; each self-verifying (driver exit codes / suite green).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `W:\gnus\models\alexnet_Opset16.onnx` provenance is public torchvision AlexNet (opset-16 export) | Corpus | Low — statistics don't depend on lineage; user confirms at approval |
| A2 | 4-D conv `[O,I,kh,kw]` → `[O, I·kh·kw]` flattening matches what Phase 11's PostConverter will feed the encoder | Extraction | Low-Med — if Phase 11 chooses a different flatten order, per-leaf statistics shift slightly; convention revisit at Phase 11 plan time |
| A3 | Byte-exactness (not just rtol 1e-4) can be asserted for sampled real layers on this MSVC toolchain | Parity | Low — harness falls back to contractual rtol decode-vs-decode and records the divergence |

All other claims above are marked `[VERIFIED: ...]` with file/line evidence read in this session.

## Open Questions (user decisions only)

1. **Corpus approval (D-02 — blocking 10-01):** Approve Candidate A (AlexNet ONNX) as primary? Add Candidate B (super-resolution ONNX) as complementary sweep, or strictly one model? (Recommendation: A + B; B costs minutes and likely obviates the D-04 synthetic fallback.)
2. **Tiny-floor rule (locks from recommendation):** `elements < 4096` OR `dimI == 1` → light tier — confirm or adjust.
3. **Harness naming (cosmetic):** `sgfp4_encode_dump.out` acceptable, or different name preference for the dump-driven parity target?

## Sources

### Primary (HIGH confidence — read directly this session)
- `tools/fp4/sgfp4_encode.hpp`, `tools/fp4/sgfp4_encode.cpp` (kDefaultV2Thresholds lines 444-450, gate ~536-540, NaN gate ~752-755)
- `tools/fp4/CMakeLists.txt` (both existing targets, foreach pattern), root `CMakeLists.txt:50,961-962`
- gnus-poc `quantize/fp4_exporter.py` (DEFAULT_V2_THRESHOLDS, `_export_v2_adaptive`, `export_weights(thresholds=...)` signature), `quantize/quadtree.py` (full gate logic incl. force-accept and `_combined_gate_error`), `quantize/sgfp4_decoder.py` (decode_v2), `quantize/sgfp4_format.py`
- `tools/fp4/author_real_shape_fixture.py` (Phase 9 generator, GNUST_POC_ROOT pattern), `tools/fp4/quantize_fp4.py` (subprocess pattern), `tools/fp4/README.md` (dims convention, 64-multiple limitation context)
- `test/op/SGFP4EncodeTest.cpp` (kEncodeRelTol 1e-4, decode-vs-decode pattern), `include/MNN/SGFP4DequantUtils.hpp` (oracle + `_crop`/`_plane` overloads), `test/op/SGFP4InjectTest.cpp`
- 09-CONTEXT.md (D-10 deferral), 09-01/09-03/09-04-SUMMARY.md (byte-exactness record, fixture generator, decode-convention fix)
- Live probes: onnx initializer dumps (both candidates), exporter timing (3 shapes), dependency versions (numpy 2.2.5 / onnx 1.18.0 / python 3.13.4 / torch 2.7.1+cpu present)

### Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| python + numpy + onnx | driver | ✓ | 3.13.4 / 2.2.5 / 1.18.0 | — |
| onnx runtime weights (both candidate files) | corpus | ✓ | — | — |
| gnus-poc repo (import path) | reference encoder/decoder/quadtree | ✓ | W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc | — |
| `sgfp4_inject`-family build (MSVC + MNN_BUILD_SGFP4_TOOLS) | dump harness | ✓ (Phase 9 built clean) | — | — |
| torch | not required | ✓ (2.7.1+cpu) | — | unused by design |

No missing blocking dependencies.

## RESEARCH COMPLETE
