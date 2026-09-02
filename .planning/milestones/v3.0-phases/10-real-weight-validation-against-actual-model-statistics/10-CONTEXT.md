# Phase 10: Real-Weight Validation Against Actual Model Statistics - Context

**Gathered:** 2026-08-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Validate the Phase 9 C++ encoder (`tools/fp4/sgfp4_encode.hpp/.cpp`) — its locked `DEFAULT_V2_THRESHOLDS` split policy, dual code modes, layout emission, and zero-pad-64 padding — against a real model's weight distributions, and revise encoder parameters only where the data demands it. This is the risk-retirement phase between the encoder port (Phase 9) and the graph-rewrite PostConverter pass (Phase 11): synthetic-fixture-tuned assumptions are the top-flagged risk and must be confirmed (or corrected) on real weights before converter integration builds on them.

Scope anchor (requirements SGV2-26/27, from `.planning/workstreams/sgfp4-pivot/REQUIREMENTS.md`):
- SGV2-26: encoder params vs. real weight statistics
- SGV2-27: (validation/revision loop closure before graph-rewrite integration)

Includes: selecting and approving one real validation model; a full weight-set statistical sweep (per-layer error stats + distribution characterization, tiered depth for tiny tensors); encode-parity sampling of the shipped C++ encoder against gnus-poc `fp4_exporter.py --adaptive`; a committed validation report artifact + reusable driver under `tools/fp4/`; optionally growing the encoder API with a config struct carrying data-justified threshold revisions (defaulted to Python-identical values).

Excludes: Phase 11's PostConverter/graph-rewrite pass and CLI flag; Phase 12's end-to-end CPU/Vulkan model inference validation; any decoder changes; any changes to `sgfp4_inject`; calibration/activation-data methods (locked out-of-scope in REQUIREMENTS.md); padding-policy redesign (zero-pad-64 + row-major crop stay as shipped — overhead is reported, not re-engineered); gnus-poc-side code changes (threshold-revision deltas are documented proposals only); committing real-weight regression test suites to `test/op/` (report artifact is the acceptance evidence, per D-05).

</domain>

<decisions>
## Implementation Decisions

### Validation corpus (SGV2-26)
- **D-01 (one real model, full weight set):** The validation corpus is a single real model's complete weight set — not the repo-root `sgfp4_mi_*` injection artifacts alone (too few tensors, questionable provenance) and not a multi-model corpus (multiplies harness/provenance work beyond a clean phase). A real model with varied layer types/scales and non-64-aligned tensors is exactly the risk SGV2-26 flags.
- **D-02 (agent proposes, user approves):** The researcher proposes 1–2 candidate models (a real gnus-poc specialist model with varied layer shapes, including non-64-aligned tensors where possible); the user approves the specific model before planning locks the corpus. The proposal must note each candidate's layer-shape variety and provenance.
- **D-03 (all layers, tiered depth):** Every float32 weight tensor above a small size floor gets full statistics + encode; tiny tensors (< 64 in a dim, single partial superblock) get a lighter edge-case check. Full sweep is what makes the statistics real.
- **D-04 (real + synthetic fallback for non-aligned coverage):) ** Non-64-aligned (padded-path) coverage comes from the real model's tensors naturally; if the approved model turns out to have scarce/no non-64-aligned tensors, one synthetic non-aligned add-on keeps padded-path coverage rather than gambling on luck. Pure-synthetic-only or hard model-approval requirements on non-aligned count are both rejected.

### Acceptance bar (what "validated" means)
- **D-05 (report artifact, not test gate):) ** Acceptance evidence is a committed statistical report: per-layer table (shape, distribution summary, leaf-size histogram the quadtree emits, worst MSE/relative-error vs. the corresponding `DEFAULT_V2_THRESHOLDS` target, pad-overhead where relevant), plus encode-parity sampling results (D-11). Statistics are inherently descriptive — a rigid numeric CI gate on every metric would be brittle. No new `op/sgfp4/` regression suite with committed real-weight fixtures this phase (size/licensing/provenance concerns); that stays discretionary for Phase 11+ if a regression gate is ever wanted.
- **D-06 (stats + distribution fit as minimum evidence):) ** "Validated" requires weight-space error statistics AND distribution characterization (histograms, kurtosis/outlier share, emitted leaf-size statistics per layer) — enough to justify any threshold revision with data. End-to-end model-output sensitivity (layer-output cosine similarity etc.) is NOT required this phase — that is Phase 12's E2E territory.
- **D-07 (all layers must pass):) ** Hard per-layer gate: every layer must meet the thresholds' own targets (max_mse / max_relative per leaf size) after encoding. A single failing layer blocks phase completion and triggers the revision policy (D-08/D-09). No tolerated failing tail, no report-only exit.

### Revision policy (what happens when data demands change)
- **D-08 (config struct + tuned set; defaults stay Python-identical):) ** Threshold tables become an explicit tunable: the encoder API grows a config struct (the Phase 9 D-10 deferred extension) whose defaults remain the Python-identical `DEFAULT_V2_THRESHOLDS`. Any revision lands as a deliberate, data-justified delta from those defaults — not as an in-place silent retune of the shipped values. The knob-less one-shot `encode(w, dimO, dimI)` overload stays available and unchanged.
- **D-09 (documented delta, upstream later):) ** If thresholds diverge from gnus-poc defaults, the revised table is recorded in the validation report as a gnus-poc-side proposal (documented delta with the motivating statistics). Upstream adoption happens in the sibling repo — no cross-repo code changes here, and no strict simultaneous-parity blocking.
- **D-10 (thresholds only):) ** Revision scope is thresholds/config-struct only. Zero-pad-64 and row-major-crop stay locked exactly as shipped in Phase 9 (D-06/D-07/D-11a there); pad-region overhead on real shapes is measured and reported but not re-engineered. Native partial-superblock traversal remains rejected (Phase 9 D-06) with the same revisit condition: only if pad overhead proves costly — and even then it is a future phase, not this one.

### Tooling form (how validation physically runs)
- **D-11 (hybrid: Python drives, C++ verified):) ** A Python driver (under `tools/fp4/`) extracts the corpus weights, runs the gnus-poc reference encoder and Python-side statistics, and produces the report. The shipped C++ encoder is verified at Python sampling points: per-layer (or sampled-layer) container comparison vs `fp4_exporter.py --adaptive` output using the Phase 9 decode-parity pattern (rtol 1e-4 decode-vs-decode), plus C++ decode-error stats matching the Python-computed reference. No new standalone C++ application surface; no Python-only measurement of the shipped encoder.
- **D-12 (tools/fp4 + report placement):) ** Both the reusable driver script and the generated report live under `tools/fp4/` (e.g. `validate_real_weights.py`-style driver + generated report artifact) — discoverable next to the encoder it validates, reusable post-phase by anyone retuning. A copy/reference in the phase planning dir is at planner's discretion; the canonical home is `tools/fp4/`.

### Claude's Discretion
- Exact driver script and report file naming within the `tools/fp4/` conventions (`validate_real_weights.py` suggested but not locked).
- Statistic metric definitions beyond D-06's list (which histogram bins, outlier definition thresholds, kurtosis estimator choice).
- Sampling strategy for D-11's C++-vs-Python parity points (which/how many layers per model) as long as every layer still passes the D-07 gate via the Python reference statistics.
- Corpus extraction mechanics (ONNX → weights, `.mnn` tensor dump, or direct checkpoint read) — researcher recommends, planner locks.
- Structure of the config struct from D-08 (fields, defaults wiring, overload vs. parameter behavior).
- Whether the tiny-tensor tiered check (D-03) reuses the Phase 9 generated-golden tiny shapes or fresh extracts.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The encoder under validation (Phase 9 output)
- `tools/fp4/sgfp4_encode.hpp` — one-shot `encode(const float*, dimO, dimI)` API, v2-adaptive scope summary, invalid-input contract; grows the D-08 config struct here
- `tools/fp4/sgfp4_encode.cpp` — the quadtree/dual-mode implementation whose `DEFAULT_V2_THRESHOLDS` split policy, code modes, and layout emission are the validation targets
- `tools/fp4/CMakeLists.txt` — `MNN_BUILD_SGFP4_TOOLS` wiring (any C++ harness/test target slots in here)

### Canonical Python reference (gnus-poc, external repo)
- `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\quantize\fp4_exporter.py` — `FP4Exporter`, `DEFAULT_V2_THRESHOLDS` (the values any D-08 delta is measured against), `--adaptive` path the C++ encoder is parity-sampled against
- `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\quantize\sgfp4_format.py` — framing constants and enums shared with the MNN side
- gnus-poc specialist models (e.g. under `gnus-poc/models/`) — D-02 candidate pool the researcher draws from

### MNN-side encode/statistics tooling
- `tools/fp4/encode_sgfp4.py` — MNN's test-oracle Python encoder (locked test-oracle-only per STATE.md 2026-08-26); reusable for D-11 Python-side statistics
- `tools/fp4/quantize_fp4.py`, `tools/fp4/test_quantize_fp4.py` — existing FP4 quantization metrics/tests the driver can pattern-match
- `tools/fp4/author_structured_fixture.py`, `tools/fp4/author_real_shape_fixture.py` (Phase 9) — deterministic corpus-generation / cross-language comparison patterns for the driver
- `tools/fp4/README.md` — tools conventions, dims convention (`dimO = shape[0]`, `dimI = shape[1]`), documented "64-multiple dims only" limitation context

### Decode-oracle and parity patterns
- `include/MNN/SGFP4DequantUtils.hpp` — `dequant_sgfp4_container_cpu` oracle (decode-parity sampling uses this), framing constants
- `test/op/SGFP4InjectTest.cpp` — the rtol 1e-4 decode-vs-decode cross-language tolerance pattern D-11 reuses
- `test/op/SGFP4TestUtil.hpp` — shared test helpers
- `test/op/SGFP4RealShapeFixtures.h` (Phase 9) — real-shape golden fixtures incl. tiny-tensor coverage the D-03 tiered check can reuse

### Workstream planning
- `.planning/workstreams/sgfp4-pivot/ROADMAP.md` §Phase 10 — goal line ("Success Criteria: TBD at plan time" — this context supplies them)
- `.planning/workstreams/sgfp4-pivot/REQUIREMENTS.md` — SGV2-26/27 mapping
- `.planning/workstreams/sgfp4-pivot/phases/09-real-weight-c-encoder-port/09-CONTEXT.md` — D-02 (thresholds deferred to Phase 10), D-06/D-07 (padding locked), D-10 (config-struct deferral) — the carry-forwards this phase resolves
- `.planning/workstreams/sgfp4-pivot/phases/09-real-weight-c-encoder-port/09-SUMMARY.md` set — as-built encoder state, byte-exactness claim D-11 leans on

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/fp4/sgfp4_encode.hpp/.cpp` — the shipped, byte-exact-vs-gnus-poc encoder (Phase 9, 13/13 suites green) — the system under validation; already validates inputs and returns empty-vector on NaN/Inf/bad dims
- `tools/fp4/encode_sgfp4.py` + `quantize_fp4.py` — Python encode/statistics machinery the D-11 driver builds on
- `author_real_shape_fixture.py` — the Phase 9 generator that already drives `fp4_exporter.py --adaptive` on deterministic weights and emits C arrays — the cross-language comparison skeleton for the driver
- `dequant_sgfp4_container_cpu` in `SGFP4DequantUtils.hpp` — deterministic decode oracle for error statistics and parity sampling
- `sgfp4_mi_*.mnn` / `.mnn.weight` artifacts + `sgfp4_mi_niche_*.d/manifest.json` at repo root — existing real-model injection outputs; useful as smoke inputs / provenance reference, but NOT the primary corpus (D-01)

### Established Patterns
- Cross-language settle-at decode-parity rtol 1e-4 (not byte-exactness for error claims) — though Phase 9 achieved byte-exactness vs the exporter; D-11 sampling may assert either, rtol 1e-4 is the contractual floor
- `tools/fp4/` is the home for encoder-adjacent Python tooling and reports (D-12)
- `MNN_BUILD_SGFP4_TOOLS=ON` gates all `tools/fp4` C++ targets
- Committed-fixture pipeline: deterministic generator → C-array header (D-04's synthetic fall back follows this if needed)

### Integration Points
- **Downstream Phase 11** consumes this phase's outcome: a validated (possibly config-struct-carrying) encoder whose defaults may embed data-justified revised thresholds — the PostConverter pass links and calls it
- **gnus-poc upstream** — D-09's documented delta is the hand-off artifact for any exporter-side threshold adoption
- **Phase 12 E2E** inherits the guarantee that every layer of at least one real model passed the per-layer gate (D-07)

</code_context>

<specifics>
## Specific Ideas

- User explicitly approved "agent proposes, I approve" for the model pick — researcher must present 1–2 candidates with layer-shape variety and provenance before planning locks the corpus (D-02).
- Report artifact (not test suite) as acceptance evidence was a deliberate choice over the "Report + test gate" option — revisit only if Phase 11 changes threaten quality regressions.

</specifics>

<deferred>
## Deferred Ideas
- **Real-weight regression test suite in `test/op/`:** deferred — report artifact is this phase's evidence (D-05); a committed-fixture regression gate can be added in Phase 11+ if needed.
- **Output-sensitivity validation (layer-output cosine similarity on random inputs):** Phase 12 E2E territory (D-06) — not this phase.
- **Native partial-superblock traversal (padding re-engineering):** stays rejected (D-10); pad overhead is reported; revisit only in a future phase if the measured overhead on real models proves costly.
- **gnus-poc exporter adoption of revised thresholds:** documented-delta proposal only (D-09); actual upstream change belongs to the sibling repo.
- **Multi-model corpus expansion:** rejected for this phase (D-01); the driver (D-12) is reusable if a broader corpus is ever wanted.

</deferred>

---

*Phase: 10-Real-Weight Validation Against Actual Model Statistics*
*Context gathered: 2026-08-31*
