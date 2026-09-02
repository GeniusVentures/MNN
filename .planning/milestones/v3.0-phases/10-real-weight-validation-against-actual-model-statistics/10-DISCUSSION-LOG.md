# Phase 10: Real-Weight Validation Against Actual Model Statistics - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-31
**Phase:** 10-Real-Weight Validation Against Actual Model Statistics
**Areas discussed:** Validation corpus, Acceptance bar, Revision policy, Tooling form

---

## Gray Areas Selected

| Area | Description | Selected |
|------|-------------|----------|
| Validation corpus | Which real weights to validate against | ✓ |
| Acceptance bar | What "validated" means and how evidence is captured | ✓ |
| Revision policy | What happens if real weights expose DEFAULT_V2_THRESHOLDS as wrong | ✓ |
| Tooling form | Python/C++ harness split, driver and report placement, C++ verification depth | ✓ |

---

## Validation corpus

**Q1: What is the primary validation corpus?**

| Option | Description | Selected |
|--------|-------------|----------|
| Root .mnn artifacts | The sgfp4_mi_* .mnn artifacts (base/base2/out + niche manifest) at repo root — smallest effort, ~3-4 tensors, questionable provenance | |
| One real model, full | A representative gnus-poc specialist model whose full weight set (all conv/fc layers, varied shapes incl. non-64-aligned) feeds the validation | ✓ |
| Multi-model corpus | Several models / layer families — strongest statistics, multiplies harness + provenance work | |

**Q2: How is the specific validation model chosen?**

| Option | Description | Selected |
|--------|-------------|----------|
| I'll designate it | User designates the canonical model at planning time | |
| Agent proposes, I approve | Researcher proposes 1-2 candidates with varied layer shapes + provenance; user approves before planning locks the corpus | ✓ |
| Agent decides | Researcher picks best-coverage local model and proceeds | |

**Q3: Within the chosen model, which tensors get validated?**

| Option | Description | Selected |
|--------|-------------|----------|
| All layers, tiered depth | Every float32 weight tensor above a small size floor gets full stats + encode; tiny tensors (<64 in a dim) get a lighter edge-case check | ✓ |
| Phase-11 targets only | Only layer types the PostConverter pass will target (large Conv/FC) | |
| Stratified sample | A stratified sample across layer types/sizes — cheaper, weaker claims | |

**Q4: How is non-64-aligned (padded-path) coverage guaranteed?**

| Option | Description | Selected |
|--------|-------------|----------|
| Real + synth fallback | Real model tensors naturally; one synthetic non-aligned add-on only if real coverage is scarce | ✓ |
| Real only, no synth | Strictly real tensors — risk of silently skipping the padded path on an all-aligned model | |
| Require non-aligned | Model must have ≥N non-64-aligned tensors as an approval criterion | |

**User's choices:** One real model full sweep; agent proposes / user approves; all layers tiered; real + synthetic fallback.
**Notes:** STATE.md had explicitly deferred "real-validation model/corpus selection" to Phase 10 — resolved here.

---

## Acceptance bar

**Q1: What is the minimum evidence "validated" requires?**

| Option | Description | Selected |
|--------|-------------|----------|
| Weight-space stats | Per-layer error stats (MSE, relative error, outlier handling) vs thresholds' own criteria | |
| Stats + distribution fit | Weight-space stats PLUS distribution characterization (histograms, kurtosis/outlier share, emitted leaf-size statistics) — enough to justify threshold revisions with data | ✓ |
| Add output sensitivity | Additionally require end-to-end model-output sensitivity — creeps toward Phase 12 E2E territory | |

**Q2: Is there a hard per-layer pass/fail gate, and how strict?**

| Option | Description | Selected |
|--------|-------------|----------|
| All layers must pass | Every layer meets the thresholds' own targets (max_mse / max_relative per leaf size); a single failing layer blocks phase completion and forces parameter revision | ✓ |
| Allow small tail | Tolerate a small failing tail (≤10% of tensors, none over 2x threshold) documented for later attention | |
| Report-only | No per-layer bar; deliver report + revision decisions; correctness gates live in Phase 12 | |

**Q3: How is the acceptance evidence captured?**

| Option | Description | Selected |
|--------|-------------|----------|
| Report artifact | Committed statistical report (per-layer table: shape, distribution summary, leaf-size histogram, worst MSE/rel-err vs threshold) — reviewable artifact, not a brittle numeric CI gate | ✓ |
| Test-suite gate | New automated op/sgfp4/ suite with committed real-weight fixtures — repeatable but has size/licensing/provenance concerns | |
| Report + test gate | Both — heaviest | |

**User's choices:** Stats + distribution fit; all layers must pass; report artifact.
**Notes:** The hard D-07 gate pairs with the D-08 revision policy: a failing layer triggers threshold revision rather than dead-ending the phase.

---

## Revision policy

**Q1: If the validation data demands it, how are thresholds revised?**

| Option | Description | Selected |
|--------|-------------|----------|
| Retune in C++ | Retune the threshold table inside MNN's encoder only; accept temporary divergence from gnus-poc | |
| Config struct + tuned set | Threshold tables become an explicit tunable (Phase 9 D-10 deferred config struct); defaults stay Python-identical; revisions are deliberate, data-justified deltas | ✓ |
| Report-only, no revision | Record findings and stop for replan if gates fail | |

**Q2: If thresholds diverge from gnus-poc defaults, how is cross-repo consistency handled?**

| Option | Description | Selected |
|--------|-------------|----------|
| MNN diverges | MNN-only tuning; gnus-poc stays on defaults; behavioral divergence risk on borderline weights | |
| Doc delta, upstream later | Revised table documented as a gnus-poc-side proposal in the report; upstream adoption handled in the sibling repo | ✓ |
| Strict simultaneous parity | Revisions valid only if gnus-poc adopts them simultaneously — blocks on cross-repo coordination | |

**Q3: Which revision scope applies if validation demands changes?**

| Option | Description | Selected |
|--------|-------------|----------|
| Thresholds only | Zero-pad-64 and row-major-crop stay locked as shipped in Phase 9; pad-region overhead is reported, not re-engineered | ✓ |
| Thresholds + padding | Also plan a native partial-superblock path if pad overhead is significant | |

*(Note: the tool duplicated an option row in the presented Q3; the selection "Thresholds only" was unambiguous.)*

**User's choices:** Config struct + tuned set; documented delta / upstream later; thresholds only.
**Notes:** These resolve the three carry-forward deferrals from Phase 9 (D-02 thresholds, D-06 padding, D-10 config struct).

---

## Tooling form

**Q1: What form does the validation harness take?**

| Option | Description | Selected |
|--------|-------------|----------|
| C++ harness (tools/fp4) | A C++ tool under MNN_BUILD_SGFP4_TOOLS that reads converted .mnn weights, runs the real encoder, emits stats — validates exactly what ships | |
| Python analysis | Python-side analysis using fp4_exporter.py + MNN's encode_sgfp4.py oracle — measures Python, not the shipped C++ encoder | |
| Hybrid: Py drives, C++ verified | Python drives corpus extraction/statistics/reporting; the shipped C++ encoder is verified at sampling points (container-byte + decode comparisons vs exporter output) | ✓ |

**Q2: Where do the validation driver and report live?**

| Option | Description | Selected |
|--------|-------------|----------|
| Planning artifacts dir | Report under .planning/workstreams/sgfp4-pivot/phases/10-.../ only | |
| tools/fp4 + report | Driver script + generated report both under tools/fp4 — reusable post-phase, discoverable next to the encoder it validates | ✓ |
| Split: script + phase report | Driver in tools/fp4, report artifact in the phase planning dir | |

**Q3: How is the shipped C++ encoder itself verified inside the hybrid flow?**

| Option | Description | Selected |
|--------|-------------|----------|
| Encode parity sampling | Per-layer container comparison: C++ encode vs fp4_exporter.py --adaptive (byte/decode-parity per Phase 9 rtol 1e-4 pattern) on sampled layers, plus C++ decode-error stats matching Python-computed reference | ✓ |
| Smoke only | Run C++ encoder over every layer, check contract isn't hit + container parses | |
| Trust Phase 9 parity | No explicit cross-check; rely wholly on the Phase 9 byte-exactness claim | |

**User's choices:** Hybrid (Python drives, C++ verified); tools/fp4 placement for driver + report; encode-parity sampling.

---

## Claude's Discretion

- Driver/report file naming within tools/fp4 conventions.
- Statistic metric definitions beyond the D-06 list (histogram bins, outlier definitions, kurtosis estimator).
- Parity sampling strategy (which/how many layers), provided every layer still passes the D-07 gate via Python reference statistics.
- Corpus extraction mechanics (ONNX/`.mnn`/checkpoint) — researcher recommends, planner locks.
- Config-struct structure from D-08 (fields, defaults wiring).
- Whether the tiny-tensor tiered check reuses Phase 9 generated-golden tiny shapes or fresh extracts.

## Deferred Ideas

- Real-weight regression test suite in `test/op/` — possible Phase 11+ addition (D-5 rejected for this phase).
- Output-sensitivity validation — Phase 12 E2E territory.
- Native partial-superblock traversal — stays rejected; revisit only if measured pad overhead proves costly, in a future phase.
- gnus-poc exporter adoption of revised thresholds — sibling-repo work triggered by D-09's documented delta.
- Multi-model corpus expansion — driver is reusable if ever wanted.
