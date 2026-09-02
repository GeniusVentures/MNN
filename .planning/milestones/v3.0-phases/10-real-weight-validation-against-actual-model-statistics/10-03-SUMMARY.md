---
phase: 10-real-weight-validation-against-actual-model-statistics
plan: "03"
subsystem: testing
tags: [sgfp4, fp4, encoder, config, parity, thresholds, validation]

requires:
  - phase: 10-real-weight-validation-against-actual-model-statistics
    provides: plan 10-01 driver + failing tuples sidecar; plan 10-02 parity harness
provides:
  - EncodeConfig struct + config-carrying encode overload (Phase 9 D-10 deferral resolved, D-08)
  - Wired C++ parity-sampling leg in validate_real_weights.py (6 real layers sampled)
  - Converged threshold-revision delta (4 iterations) + final committed acceptance report; C++ defaults kept Python-identical with documented promotion-decline rationale
affects: [phase-11-postconverter, gnus-poc-upstream]

tech-stack:
  added: []
  patterns:
    - overload-based (not default-argument) config threading — greppable call-site distinction
    - threshold-delta convergence loop via --thresholds override files (cascade: relaxing targets shifts split decisions, residual failures shrink monotonically)

key-files:
  created: []
  modified:
    - tools/fp4/sgfp4_encode.hpp
    - tools/fp4/sgfp4_encode.cpp
    - tools/fp4/validate_real_weights.py
    - tools/fp4/real_weight_validation_report.md
    - tools/fp4/real_weight_validation_report.json

key-decisions:
  - "D-07 relative gate reformulated (user-approved 2026-08-31): hard gate = plain per-element worst-leaf MSE; relative criterion = leaf energy ratio mse/signal_power (the exporter's own folding). The plain per-element ratio is structurally unbounded on real weights (worst 3.6e6 vs 0.05 target) — near-zero weights explode the denominator under any scale-based FP4 quantization; no finite threshold can satisfy it"
  - "C++ default promotion DECLINED with rationale: cross-repo default parity outranks promotion — gnus-poc's exporter still defaults to DEFAULT_V2_THRESHOLDS, so a one-sided promotion makes default-vs-default encodes diverge (directly observed: classifier.6/classifier.1 parity rows diverged when only the Python side used the delta). D-09's upstream-proposal path is the correct route; consumers wanting the validated table pass EncodeConfig explicitly (values documented in the cpp comment)"
  - "Promotion probe still run for evidence: 13/13 suites green even under promoted values (fixture planes' split decisions unaffected), but parity semantics — not fixture survival — is the binding constraint"
  - "MSVC brace-elision: EncodeConfig aggregate init requires explicit Gate{} per element (C2440 otherwise); kDefaultEncodeConfig must be extern in header + defined once in cpp (static-in-header collided with the definition, C2086)"

patterns-established:
  - Convergence-loop revision: apply clear+10% to failing sizes, re-run, repeat; residuals shrank 581 → 10 → 2 → 0 in 3 iterations after the metric reformulation

requirements-completed: [SGV2-26, SGV2-27]

coverage:
  - id: D1
    description: "EncodeConfig struct + config-carrying overload, behavior-preserving (13/13 suites unmodified)"
    requirement: SGV2-27
    verification:
      - kind: other
        ref: "run_test.out op/sgfp4 → 13/13 passed, zero test-file changes (git status test/ clean)"
        status: pass
      - kind: other
        ref: "divergence probe: config(defaults) byte-identical; strict-config container differs (20576B vs 15280B)"
        status: pass
    human_judgment: false
  - id: D2
    description: "C++ parity leg wired and run on 6 sampled real layers"
    requirement: SGV2-27
    verification:
      - kind: other
        ref: "driver --encode-dump run: 4/6 byte-exact PASS; classifier.6/classifier.1 rtol-1e-4 fallback PASS (documented divergence)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Threshold revision loop to documented conclusion + final committed report"
    requirement: SGV2-26
    verification:
      - kind: other
        ref: "final driver run under delta.json: exit 0, 16/16 layers PASS; report contains D-09 delta table with motivating statistics"
        status: pass
      - kind: other
        ref: "promotion probe: suites green under promoted values; promotion declined with parity rationale recorded in cpp comment"
        status: pass
    human_judgment: false
---

# Phase 10 Plan 03: EncodeConfig + Parity Leg + Threshold Decision Summary

One-liner: shipped `EncodeConfig` with Python-identical defaults (suites 13/13 green unmodified, divergence probe proves the knob threads through), wired the C++ parity leg (6 real layers — 4 byte-exact, 2 contractual rtol fallback), reformulated the structurally-unsatisfiable relative gate per user decision, converged the threshold delta in 4 iterations to all-green exit 0, and declined C++ default promotion with a documented parity rationale — the report is the phase's acceptance evidence.

**Duration:** ~50 min | **Tasks:** 3 | **Files:** 5

## Accomplishments

- **Task 1 — EncodeConfig (D-08):** `sgfp4_encode.hpp` grows `EncodeConfig{Gate leafGates[5]}` (thresholds only, per D-10) + `encode(w, dimO, dimI, const EncodeConfig&)` overload; knob-less overload unchanged and now a one-line forward to `kDefaultEncodeConfig`. Quadtree gate lookup reads `ctx.thresholds` threaded from config. Compatibility proof: 13/13 `op/sgfp4` suites green with zero test-file modifications. Divergence probe: strict-config container differs (20576B vs 15280B) while config(defaults) is byte-identical.
- **Task 2 — parity leg:** `run_parity_leg` drives the harness per sampled layer (dump → subprocess → byte-compare vs `export_weights --adaptive` → decode-stats rtol 1e-4 → transient cleanup). Sampled: `features.0.weight` (byte-exact), `classifier.6.weight` (rtol fallback), `classifier.1.weight` (rtol fallback, largest plane), `features.8.weight` (byte-exact aligned conv), 2 light-tier biases (byte-exact). Exit 4 wired for mismatch (untriggered).
- **Task 3 — threshold decision (Branch B):** see Key Decisions. Loop: initial reformulated-gate run exposed 581 energy-ratio failures → delta iterations at clear+10% → 10 → 2 → 0. Final table: `max_relative` 64:0.384, 32:0.079, 16:0.03, 8:0.015, 4:0.03; `max_mse(4)` 0.0099. Final run: **exit 0, 16/16 PASS**. Report renders the D-09 delta block (size / old / new / motivating statistic) as the gnus-poc upstream proposal.

## Verification Log

- `run_test.out op/sgfp4`: 13/13 passed (post-EncodeConfig, and during the promotion probe)
- Divergence probe: `defaults==config(defaults): 1 ; strict differs: 1`
- Final driver: `--thresholds delta.json --encode-dump ...` → `swept 16 layers; gate: PASS; exit=0`
- `git status test/` clean throughout — zero test modifications (D-05 honored)

## Deviations from Plan

**[Rule 4 – Architectural] D-07 relative-gate reformulation (user-approved)** — Found during: Task 3 | Issue: the plain per-element relative metric is structurally unsatisfiable on real weights (15,752 failures, worst 3.6e6 vs 0.05 — near-zero denominators); the plan's literal Branch B ("raise to the smallest value that clears the observed worst error") would require a meaningless ~8-order relaxation that also coarsens the split policy. | Resolution: presented options via user checkpoint; user chose "reformulate relative gate" (energy-ratio folding, matching the exporter's split driver) + "land the size-4 MSE delta". Session recorded in STATE before the checkpoint. | Files: `validate_real_weights.py` | Verification: converged green run, exit 0. | Commit: f4e3223d

**[Rule 4 – Architectural] C++ default promotion declined (plan-consistent)** — Found during: Task 3 step 4 | Issue: plan permits promoting revised values to `kDefaultEncodeConfig` if suites stay green — they do, but a one-sided promotion breaks Python↔C++ default parity (gnus-poc exporter defaults unchanged; observed directly when the driver's Python side encoded under the delta while the harness used defaults). | Resolution: defaults kept Python-identical; validated values documented in the cpp comment as the explicit-config path; report states the D-09 upstream-proposal status. This follows the plan's own "byte-exactness fixtures outrank promotion / D-09 proposal path exists precisely for this" rule, applied to cross-repo default parity. | Files: `sgfp4_encode.cpp` | Verification: 13/13 after revert. | Commit: f4e3223d

**[Rule 3 – Environment] Revision loop needed 4 iterations, not 1** — Found during: Task 3 | Issue: plan budgeted one re-run iteration; relaxing targets shifts quadtree split decisions, so new (smaller) residuals appear — the cascade converged in 3 additional iterations (581→10→2→0), all using the same clear+10% arithmetic. | Impact: none — monotone convergence, loop stayed within the plan's arithmetic definition.

**Total deviations:** 4 (2 user-approved architectural, 1 plan-consistent declination, 1 budget arithmetic). **Impact:** gate green; all constraints (D-05/D-09/D-10) honored.

## Self-Check: PASSED

- [x] `git diff` zero changes under `test/`; 13/13 suites green unmodified
- [x] Full driver run with parity: exit 0; report has 16-tensor stats table, pad-overhead rows, 6-layer parity table, gate-metric note, threshold-decision + D-09 delta blocks
- [x] `delta.json` retained under workdir; re-run under delta green; C++ defaults promotion consistent with fixture outcome (declined, with rationale)
- [x] No gnus-poc-side changes; no test/op/ changes; no real weights committed

## Issues Encountered

None blocking. Carry-forward for Phase 11: the validated threshold table lives in `tools/fp4/real_weight_validation_report.json` + the cpp comment — the PostConverter pass should accept an EncodeConfig and default to Python-identical values until gnus-poc adopts the delta upstream.

## Next Phase Readiness

Phase 11 unblocked: encoder callable with defaults or explicit config; report is the hand-off artifact; gnus-poc delta documented for upstream (D-09). Phase complete.
