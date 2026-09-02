---
status: passed
phase: 10-real-weight-validation-against-actual-model-statistics
verified: 2026-09-01
verifier: inline (Copilot sequential executor)
method: goal-backward analysis vs ROADMAP success criteria + REQUIREMENTS traceability + live artifact/run inspection
---

# Phase 10 VERIFICATION

Phase goal: *Validate/revise encoder parameters against a real model's weight distributions before graph-rewrite integration (synthetic-fixture-tuned assumptions are the top-flagged risk).*

## Success Criteria Check (ROADMAP §Phase 10, derived at plan time)

### 1. All-layer gate green (D-07) — PASS

- Final driver run under the converged delta: `exit=0`, 16/16 layers PASS (verified live: `swept 16 layers; gate: PASS`).
- Revision is a documented, data-justified, minimal-arithmetic delta (clear+10% per failing size) re-run green via `--thresholds` — loop evidence: 15,765 (plain metric) → metric reformulated (user-approved 2026-08-31, recorded as a Rule 4 decision + STATE session entry) → 581 → 10 → 2 → 0.
- The plain per-element relative metric's structural unsuitability is documented in the report (gate-metric note) rather than silently satisfied.
- Zero tolerated failing tail in the final state.

### 2. Statistics + distribution evidence (D-06) — PASS

- `tools/fp4/real_weight_validation_report.md` + `.json` committed: per-layer shape/projection/tier, kurtosis, dual outlier shares, leaf-size histograms, code-mode mix, pad-overhead rows (features.0.weight 1.0579, classifier.6.weight 1.0240), corpus provenance (model sha256 `4bc388cc…`), toolchain versions, effective threshold table.
- No real weights committed (D-05); no new `test/op/` regression suites (git history: only tools/fp4 + .planning files touched).

### 3. C++ encode-parity sampled (D-11) — PASS

- `sgfp4_encode_dump.out` exists under `MNN_BUILD_SGFP4_TOOLS`, links `sgfp4_encode`, exit contract 0/1/2 verified.
- Sampling rule honored: both non-64-aligned tensors (features.0.weight, classifier.6.weight), largest aligned plane (classifier.1.weight), one aligned conv (features.8.weight), 2 light-tier biases — 6 layers.
- 4/6 byte-exact; 2 took the contractual rtol-1e-4 decode-vs-decode fallback (recorded in the report as expected behavior); decode-stats matched within 1e-4 on all.

### 4. Config struct shipped (D-08/D-09) — PASS

- `EncodeConfig{Gate leafGates[5]}` in `sgfp4_encode.hpp` + config-carrying overload; knob-less overload signature unchanged (forwards to `kDefaultEncodeConfig`).
- 13/13 `op/sgfp4` suites green WITHOUT test modification (compatibility proof; `git status test/` clean throughout).
- Divergence probe: config(defaults) byte-identical; different config measurably changes output (20576B vs 15280B).
- Divergence from gnus-poc defaults rendered as the D-09 documented-delta block (5 rows with motivating statistics); no gnus-poc code changes. C++ default promotion explicitly declined with parity rationale (cross-repo default parity outranks one-sided promotion) — the plan's own "fixtures/parity outrank promotion" rule applied; promo-probe evidence recorded.

## Requirements Traceability

- **SGV2-26** (encoder params vs real weight statistics): closed by criteria 1+2 — all-layer gate green on the approved corpus with distribution evidence. Carry-forward note format (workstream REQUIREMENTS.md) updated via plan summaries.
- **SGV2-27** (validation/revision loop closure): closed by criteria 3+4 — parity machinery (harness + wired leg) shipped and passing; revision loop executed to a documented conclusion (delta derived, converged, recorded; promotion decision made with rationale).

## Notes / Carry-forwards

1. The validated table lives in `real_weight_validation_report.json` + the `sgfp4_encode.cpp` comment — Phase 11's PostConverter should accept `EncodeConfig` explicitly rather than relying on defaults until gnus-poc adopts the delta upstream.
2. The two rtol-fallback parity rows (classifier.1/classifier.6) stem from threshold-sensitive split decisions at real-weight boundaries — consistent with the byte-exact bar holding on all other sampled layers.
3. `run_test.out` full-family gate (13/13) re-verified after the final revert commit.

## Verdict

**PASSED** — all four success criteria verified against live artifacts, runs, and git history. Phase 10 goal achieved: the encoder's parameter set is validated against real model statistics, with a documented revision proposal and parity-sampled C++ verification.
