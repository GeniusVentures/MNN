---
phase: 10-real-weight-validation-against-actual-model-statistics
plan: "01"
subsystem: testing
tags: [sgfp4, fp4, quantization, validation, onnx, statistics]

requires:
  - phase: 09-real-weight-c-encoder-port
    provides: shipped C++ encoder + gnus-poc reference byte-exactness baseline
provides:
  - Reusable Python validation driver tools/fp4/validate_real_weights.py (extraction, tiering, stats, gate, report)
  - Committed real-weight validation report (descriptive half) + JSON sidecar with failing tuples
affects: [10-03, phase-11-postconverter]

tech-stack:
  added: []
  patterns:
    - dump-driven exit-code contract validation driver (0/1/2/3/4)

key-files:
  created:
    - tools/fp4/validate_real_weights.py
    - tools/fp4/real_weight_validation_report.md
    - tools/fp4/real_weight_validation_report.json
  modified: []

key-decisions:
  - "D-07 metric is the plain per-element worst-leaf MSE/relative (not the Laplacian-weighted internal gate) — spotlighting relative-error explosion on near-zero weights (worst 3.6e6) that the weighted internal gate never surfaces"
  - "Corpus reality vs plan truth text: AlexNet has 8 full-tier weights + 8 light-tier biases, not 10/6 — the plan's 10/6 claim miscounted the RESEARCH table (which itself lists 16 rows incl. 6 biases of 8); tier rule implemented as the D-03 expression so the partition follows the data, not the prose"
  - "Leaf geometry driven by instantiating QuadtreeEncoder per superblock exactly as fp4_exporter._export_v2_adaptive does (same fit functions/ternary_delta/laplacian), with the container always produced by export_weights so the normative framing path is the exporter's own"

patterns-established:
  - "No-op threshold delta equivalence check: run --thresholds with identical values and diff failing tuples at rtol 1e-9"

requirements-completed: [SGV2-26]

coverage:
  - id: D1
    description: "Validation driver enumerating all 16 FP32 tensors with tier classification and finite gate"
    requirement: SGV2-26
    verification:
      - kind: other
        ref: "python tools/fp4/validate_real_weights.py --list-only (16 tensors, 8 full / 8 light)"
        status: pass
      - kind: other
        ref: "finite_gate NaN probe exits 2 (throwaway check)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Committed real-weight statistics report + JSON sidecar covering the corpus"
    requirement: SGV2-26
    verification:
      - kind: other
        ref: "python tools/fp4/validate_real_weights.py --model W:\\gnus\\models\\alexnet_Opset16.onnx (exit 3 = legitimate D-07 gate failure feeding 10-03)"
        status: pass
      - kind: other
        ref: "no-op --thresholds run produced identical failing tuples (rtol 1e-9) proving override plumbing"
        status: pass
    human_judgment: false
---

# Phase 10 Plan 01: Real-Weight Validation Driver Summary

One-liner: Python driver sweeping all 16 AlexNet FP32 tensors (8 full / 8 light tier) with distribution stats, leaf histograms, and the plain-metric D-07 gate — producing the committed descriptive half of the phase's acceptance evidence; gate legitimately fails (exit 3) with 15,765 failing leaf tuples recorded for 10-03's delta loop.

**Duration:** ~25 min | **Tasks:** 2 | **Files:** 3

## Accomplishments

- `tools/fp4/validate_real_weights.py` — argparse CLI (`--model/--report/--gnus-poc-root/--thresholds/--encode-dump/--sample/--workdir/--list-only`), ONNX initializer extraction, D-03 tier rule (`elements < 4096 OR dimI == 1`), finite pre-scan (exit 2), 64-bin log-spaced histogram / Fisher kurtosis / dual outlier-share stats, leaf-size histogram + code-mode mix via reference quadtree drive, plain per-leaf-footprint MSE/relative gate vs per-size targets, pad-overhead ratio rows, markdown + JSON sidecar writers, exit-code contract 0/1/2/3(/4 reserved).
- Committed report artifacts: per-layer table (16 rows, 8 full tier each with explicit PASS/FAIL + worst-leaf numbers vs targets), pad-overhead sub-table for `features.0.weight` (1.0579) and `classifier.6.weight` (1.0240), parity table stubbed SKIPPED for 10-03, sidecar recording every failing `(layer, size, kind, value, target)` tuple.
- `--thresholds` override plumbing proven end-to-end with a no-op delta (identical failing layers + tuples at rtol 1e-9).

## Gate Result (exit 3 — legitimate, feeds 10-03 Task 3 Branch B)

- All 8 full-tier layers FAIL the relative target; worst relative errors 1.3e2–3.6e6, driven by near-zero original weights making `|o-d|/(|o|+1e-12)` explode.
- 13 MSE failures, all forced-4×4 leaves on `features.3.weight` (worst 8.99e-3 vs 5e-4 target) — the quadtree's forced-accept-at-min-size construction.
- 0 epsilon escapes (no silent signal_power ≤ 1e-12 passes).
- Failing-tuple distribution: 14,929 × size-64, 720 × size-4, 64 × size-8, 20 × size-32, 19 × size-16 (relative kind); 13 × size-4 (mse kind).

## Deviations from Plan

**[Rule 2 – Missing critical info] Tier partition differs from plan truth text** — Found during: Task 2 | Issue: plan's must-have says "all six bias vectors light / all ten conv/FC weight tensors full"; the actual opset-16 AlexNet corpus has 8 biases and 8 conv/FC weights (the RESEARCH table itself lists exactly these 16). | Fix: implemented the D-03 rule as an expression (never hardcoded), driver enumerates the corpus as-is: 8 light / 8 full. | Files: none beyond planned | Verification: `--list-only` output matches the RESEARCH §Candidate A table row-for-row. | Commit: n/a (folded into task 2 design)

**Total deviations:** 1 auto-fixed (info mismatch, rule-code-1-class). **Impact:** none — rule-based tiering is stricter than name-based partitioning.

## Self-Check: PASSED

- [x] `--help` and `--list-only` succeed; 16 tensors enumerated with dims table matching 10-RESEARCH §Candidate A
- [x] Full run completes in budget (~10 min incl. 37.7M-element layer) with exit 3 (contract-conformant gate failure)
- [x] All 16 tensors in report with explicit PASS/FAIL; light tier carries roundtrip framing + max-abs; full tier carries worst-leaf MSE/relative vs per-size targets
- [x] Both non-64-aligned tensors have pad-overhead rows
- [x] Exit-code contract 0/1/2/3 implemented and exercised (0 not applicable — gate failed; 2 via NaN probe; 1 observed via malformed UTF-8 thresholds file during plumbing check); 4 reserved for 10-03
- [x] `--thresholds` path exercised with no-op delta producing unchanged gate result
- [x] Report + sidecar committed under tools/fp4/; no real weights committed; no test/op/ changes; no gnus-poc changes (git diff scope verified)

## Issues Encountered

None blocking. Observation for 10-03: the relative target is systematically unattainable on real weights under the plain metric (any near-zero weight with nonzero decode error yields a huge ratio). The data-driven delta loop must decide whether this constitutes a threshold revision or a documented metric formulation gap.

## Next Phase Readiness

Ready for 10-02 (independent) and 10-03 (consumes sidecar failing tuples + report tables).
