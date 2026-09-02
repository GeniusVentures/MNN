---
plan: "11-05"
status: complete
started: 2026-09-01
completed: 2026-09-01
commits: [f2b00b08]
---

# Plan 11-05 Summary: D-13 real-corpus smoke + D-14 flag-OFF gate + README

## D-13 Evidence (all legs green)

| Leg | Result |
|---|---|
| Flag-ON conversion | `Converted Success!`, exit 0, `--dumpPass`: `InsertSGFP4Dequant: ops 74 -> 82` then `no change` (idempotent in-pipeline) |
| Node count | **8 SGFP4Dequant ops** + 8 two-input (rewired) convolutions in the artifact — exactly the expected 8 candidates (5 feature convs + 3 classifier FC→convs), zero light-tier skips (matches the Phase 10 tier table) |
| Decode leg (classic API) | `createFromFile` → `createSession` → `resizeTensor({1,3,224,224})` → `resizeSession` → `runSession` = **NO_ERROR** on the real converted AlexNet — full flag→pass→buffer→artifact→runtime chain proven |
| Mutex leg | `--sgfp4 --fp16` / `--hqq` / `--weightQuantBits` → MNN_ERROR mutex text, exit 1, no output (verified in 11-03, re-confirmed) |

## D-14 Evidence

- Flag-OFF corpus conversion: `Converted Success!`, exit 0.
- `run_test.out op/sgfp4`: **13/13** green.
- `TestSGFP4Converter.exe`: PASS (layout + reload parity + pass mechanics) — including the new 4-D dims contract tests.
- `git status --porcelain test/`: **empty** (zero test-file edits across the phase).

## The Deviation (D-13): 4-D conv-weight dims

**Found:** the decode probe crashed (`0xC0000005`) in `runSession`. Root cause — exactly the pitfall the v2.0 architecture research flagged: `ShapeConvolution.cpp:33-38` (and `ConvolutionTiledExecutorMultiInput::onExecute`, which reads `inputs[1]->channel()/batch()/stride(1)`) require a multi-input conv's weight input to be **4-D `[O, I, kH, kW]`** with CAFFE-side channel semantics. The pass (and, notably, the injection tool before it) wrote `dims = {dimO, dimI}` — a 2-D tensor — so shape inference read garbage extents. The injection tool never hit this because its classic_api test consumes the weight via **MatMul**, not a conv; the converter path is the first true conv consumer.

**Fix (backward-compatible, 3 files):**
1. `InsertSGFP4Dequant.cpp` — emits `dims = {O, kernelSize/(kx*ky), ky, kx}` when `common->kernelX/kernelY` divide `kernelSize` (schema defaults make unset kernels 1×1 → `{dimO, dimI, 1, 1}`); flat `{dimO, dimI}` fallback otherwise. Encode plane unchanged (`dimO × dimI`).
2. `CPUSGFP4Dequant.cpp` — new `readDecodeDims` helper: `dims[0]` = dimO, **product of remaining dims** = dimI. 2-D artifacts decode bit-identically to before.
3. `ShapeSGFP4Dequant.cpp` — rank ≥ 3 → `MNN_DATA_FORMAT_NCHW` (CAFFE-equivalent, so `Tensor::channel()` = dim[1] = I); rank-2 stays NHWC (MatMul artifacts unchanged).

**Proof of compatibility:** all 13 `op/sgfp4` suites (which exercise the injection tool's 2-D artifacts end-to-end) pass unchanged; PHASE C T1 updated to the 4-D contract and a new **T6b** asserts `{64,32,2,2}` for explicit 2×2 kernels + flat-plane decode.

**Scope note:** this touched `source/` runtime files beyond the plan's declared `files_modified` (README only) and beyond the pre-authorized OQ3 contingency (which covered Express round-trip node loss — not hit; node count was exactly 8). Recorded here and in STATE as an explicit D-13 deviation per the plan's own "document, don't hide" instruction. OQ3 was NOT triggered.

## Deviations

1. **4-D dims runtime change** (above) — the deviation of record for this plan.
2. Decode-probe mechanics: the classic-API leg needs `resizeTensor` on the dynamic input `x` before `resizeSession` (AlexNet has no static input shape); the probe is a scratch tool (`tmp/p13_decode_probe.cpp`, untracked) documented in README step 3 by its API sequence rather than committed.

## Self-Check

- [x] All D-13 legs green; all D-14 legs green
- [x] README `## mnnconvert --sgfp4 smoke` section with corpus sha256 `4bc388cc`, expected counts, deviation note, flag-OFF statement
- [x] No OQ3 fallback; primary D-01 placement confirmed (74→82 ops)
- [x] Committed: f2b00b08

## Key Files

### modified
- `tools/fp4/README.md`
- `tools/converter/source/optimizer/postconvert/InsertSGFP4Dequant.cpp`
- `tools/converter/source/TestSGFP4Converter.cpp` (T1 update + T6b)
- `source/backend/cpu/CPUSGFP4Dequant.cpp`
- `source/shape/ShapeSGFP4Dequant.cpp`

## Open Items

- `tmp/p13_decode_probe.cpp` / `.exe` + `tmp/t4_diag.*` scratch tools — untracked, disposable.
- Phase 12 (E2E CPU/Vulkan validation + accuracy gates) owns formal output-accuracy comparison; this phase proved load/run only, per plan wording.
