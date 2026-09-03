# Phase 7: Multi-Tensor Hardening & Structured-Data Coverage - Context

**Gathered:** 2026-08-27
**Status:** Ready for planning

<domain>
## Phase Boundary

The `sgfp4_inject` tool handles realistic multi-weight models and the full SGFP4 v2 format surface: multiple target weight tensors / containers inject into a single artifact with independent collision-free sidecar byte ranges and load/run correctly via the classic API; at least one structured (non-uniform, LAYOUT_MIXED/quadtree) container exercises the quadtree decode path end-to-end through injection; and malformed/empty inputs fail cleanly in the tool (no partial output artifact, non-zero exit, diagnostic) rather than emitting a corrupt artifact — SGINJ-07/SGINJ-08.

Out of scope: same-shape weight disambiguation / manifest tensor-name keying (v3.0 Phase 11), non-64-multiple shape padding conventions (v3.0 Phase 10), Vulkan E2E (v3.0 Phase 12), decode-domain robustness changes to `CPUSGFP4Dequant` (v1.0 territory), fixing gnus-poc's `runner.py` default quantize output.

</domain>

<decisions>
## Implementation Decisions

### Structured Container Source & Fixture Form
- **D-01:** The structured LAYOUT_MIXED/quadtree container is produced by the REAL gnus-poc encoder (`fp4_exporter.py --adaptive` run on structured weights) — NOT MNN's test-oracle-only `tools/fp4/encode_sgfp4.py`. This satisfies the STATE.md pending todo and exercises the actual encoder the tool's input contract targets.
- **D-02:** The container bytes are baked into the test as a generated C-array header fixture (following the `SGFP4DequantFixtures.h` precedent), with a correct synthetic `manifest.json` (computed sha256 over the fixture bytes) written to a temp dir at runtime — fully self-contained, no runtime Python, no committed binaries, no env-var conditional skips (Phase 6 D-10 precedent).
- **D-03:** The one-time export run happens at fixture-authoring time (quick-task or manual step in the gnus-poc repo). The cross-repo dependency exists only at authoring time, never in the test harness or CI.
- **D-04:** The structured container runs the same classic-API inject→load→run chain as the uniform container (inside the multi-tensor test) — LAYOUT_MIXED op-level decode correctness is already proven (v1.0 Phases 2/4); this phase proves the injection path with it, not new decode coverage. No separate dedicated suite for the structured container.

### Multi-Tensor Setup
- **D-05:** Base model topology: `Input[512]` with two distinguishable-shape weights — e.g. `w1[512,512]` (paired with the structured container) plus a second smaller weight (e.g. `[512,64]` or `[64,64]`, paired with the uniform demo-lineage container) — direct extension of the Phase 6 D-02 `Input[512] → MatMul[512,512]` precedent, with each container pairing to its own niche dir.
- **D-06:** Container mix: one structured (gnus-poc exported) + one uniform (existing demo-container lineage) — a single injected artifact proves BOTH multi-tensor collision-free offsets AND quadtree/structured coverage in one run.
- **D-07:** Validation path is the **classic API** (`Interpreter::createFromFile` → `createSession` → `runSession`) per SGINJ-07's "loading and running correctly" wording and the downstream SGProcessingManager path. Express `Module::load` coverage remains the tool's unconditional in-tool verify (Phase 5 D-12) — not duplicated in the test.
- **D-08:** Same-shape weight collision handling is **deferred**: exact-shape pairing keeps hard-failing on 2+ matches (Phase 5 D-02). Manifest tensor-name keying (needed for real LLM models with repeated weight shapes) goes to the deferred list for v3.0 Phase 11 (converter graph-rewrite pass).

### Malformed-Input Probe Depth & Clean Failure
- **D-09:** Full input-failure matrix probed at the `sgfp4_inject::run()` level (in-process via `sgfp4_inject_core.hpp`): truncated/empty container, bad sha256, legacy v1 container, manifest missing fields, dims disagreeing with the matched tensor, zero-match and multi-match shapes — every case must exit non-zero with a diagnostic.
- **D-10:** Corrupted-payload-byte probes are **included** (picked up from Phase 6 D-14 deferral): a container that passes magic/version/sha256 but has garbage body bytes — the tool must never crash; either a clean structural success (decode-level tolerance is not the injector's concern) or a clean failure.
- **D-11:** **Atomicity requirement:** a failed run must leave NO partial `out.mnn`/`out.mnn.weight` behind — all validation completes before any output file is written, or output goes to temp files promoted by rename. This exists because a corrupt artifact would crash the downstream unchecked-nullptr path in `SGProcessingManager` rather than fail gracefully. (Note: the current core validates all niche dirs up front but writes the sidecar before graph surgery/serialization — the planner should verify/adjust write ordering or temp-file strategy.)
- **D-12:** One new `run_test.out` suite (e.g. `test/op/SGFP4MultiTensorTest.cpp`, registered under `op/sgfp4/multi_tensor`) carries BOTH the positive multi-tensor/structured tests AND the malformed-input probes — single file, Phase 6 D-09 pattern, filtered-suite workaround for the `FP4ModelTest.cpp` full-build blocker.

### dims Convention Documentation
- **D-13:** SGINJ-07's "documented" requirement = a new `tools/fp4/README.md` covering: the `dims = {dimO, dimI}` matrix convention (2-D `[out, in]` row-major weights only — transposed pairing was rejected in Phase 5), the niche-dir/manifest input contract, CLI usage, and sidecar layout. The natural entry point for anyone picking up the tool.
- **D-14:** The "applied" requirement is satisfied by the EXISTING cross-checks (manifest `stats.shape` vs matched `.mnn` tensor dims, param `dims` set from the same source) — tests assert these fire; no new mechanism.

### Claude's Discretion
- Exact second-weight shape (`[512,64]` vs `[64,64]`) and how the two MatMuls compose in the test graph (chained vs parallel), provided both weights are 2-D with exact 64-multiple shapes and distinct from each other.
- The structured fixture's source weights (what structured matrix to feed the exporter at authoring time — e.g. block-constant + ramp + noise mixture guaranteeing MIXED/non-uniform macroblocks).
- Exact malformed-probe test-case organization within the single suite (one test case vs multiple), and probe-specific byte mutations.
- The atomicity implementation choice (full pre-validation vs temp-file+rename), provided a failed run leaves no partial output.
- README length/structure beyond the four required content areas (D-13); error-message wording; logging verbosity.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Injection tool (subject of hardening)
- `tools/fp4/sgfp4_inject_core.hpp` — the entire core: `loadNicheDir` validation (write-ordering relevant to D-11), exact-shape pairing, sidecar merge (16-byte-aligned monotonic cursor), `makeDequantOp`, in-tool verify
- `tools/fp4/sgfp4_inject.cpp` — thin CLI shim; Phase 7 must keep `sgfp4_inject.out` building standalone
- `tools/fp4/CMakeLists.txt` — tool build wiring
- `tools/fp4/sha256.hpp` — vendored SHA-256 used for synthetic manifests (D-02/D-09)

### Format / decode ground truth (in-repo)
- `include/MNN/SGFP4DequantUtils.hpp` — container framing, `sgfp4_is_v2_container` version gate, `sgfp4_align16`, `dequant_sgfp4_container_cpu` (oracle)
- `source/backend/cpu/CPUSGFP4Dequant.cpp` — the CPU Execution the classic path drives; its behavior on garbage payload bytes is relevant to D-10 expectations
- `tools/fp4/encode_sgfp4.py` — container-format constants documentation (test-oracle-only encoder; NOT the structured fixture source per D-01)
- `schema/current/MNN_generated.h` — `SGFP4DequantParamT{magic, external{offset,size}, dims{dimO,dimI}}` shape

### Test precedents (in-repo)
- `test/op/SGFP4DequantFixtures.h` — generated C-array fixture precedent (D-02)
- `test/op/SGFP4ClassicAPITest.cpp` — Phase 6 suite: in-process injection via core header, synthetic niche-dir temp-dir construction, classic-API session flow, missing-sidecar probe — the direct template for D-12
- `test/TestUtils.h` — `checkVectorByRelativeError` assertion helpers
- `test/CMakeLists.txt` — suite build (glob); filtered-suite workaround for the `FP4ModelTest.cpp` blocker (STATE.md pending todo)
- `.planning/codebase/TESTING.md` — test framework conventions (`MNNTestSuiteRegister`, hierarchical suite strings)

### Classic-API reference flow
- `demo/exec/pictureRecognition.cpp` — canonical classic-API demo (`Interpreter::createFromFile` → `createSession` → named I/O → `runSession`)
- `include/MNN/Interpreter.hpp` — public classic-API surface

### Exporter artifacts (external, fixture-authoring time only)
- `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\quantize\fp4_exporter.py` — canonical encoder; run with `--adaptive` (v2) on structured weights to produce the D-01 container
- `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\quantize\manifest.py` — manifest schema for the synthetic manifest (D-02)
- `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\models\specialists_mlx\demo\fp4\demo.sgfp4` — uniform demo container lineage (D-06 second weight source)

### Workstream planning
- `.planning/workstreams/sgfp4-pivot/REQUIREMENTS.md` — SGINJ-07, SGINJ-08; Out of Scope table
- `.planning/workstreams/sgfp4-pivot/ROADMAP.md` §"Phase 7" — success criteria 1–3
- `.planning/workstreams/sgfp4-pivot/STATE.md` — locked decisions (structured-container caveat, externalPath gotcha, FP4ModelTest blocker, terminology)
- `.planning/workstreams/sgfp4-pivot/phases/05-injection-core-artifact-construction-graph-splicing/05-CONTEXT.md` — tool contract decisions D-01..D-13 this phase hardens
- `.planning/workstreams/sgfp4-pivot/phases/06-classic-api-load-run-validation/06-CONTEXT.md` — D-09..D-16 (test-form, fixture, probe-depth precedents); D-14/D-15 deferrals picked up here

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `sgfp4_inject::run(argc, argv)` from `tools/fp4/sgfp4_inject_core.hpp` — in-process injection entry the new suite drives (no subprocess; Phase 6 D-12 pattern)
- `test/op/SGFP4ClassicAPITest.cpp` — near-complete template: synthetic niche-dir construction (manifest + sha256 + container to temp dir), classic-API load/run, failure probing
- `dequant_sgfp4_container_cpu` — FP32-decode oracle for baseline weights (Phase 6 D-06 zero-by-construction baseline pattern carries over)
- `Variable::loadMap` / `Variable::replace` / `Variable::save` Express surgery chain — already handles N nodes in a loop; multi-tensor is exercising existing code, not adding surgery

### Established Patterns
- Multi-niche-dir CLI (`--niche-dir <dir>...`) and merged-sidecar cursor loop already exist in the core — Phase 7 proves them with N>1 rather than building them
- Generated fixtures + runtime-written temp niche dirs (no committed binaries, no env-var skips)
- Filtered-suite runs (`op/sgfp4/...`) to dodge the `FP4ModelTest.cpp` full-build blocker
- `op->externalPath` set literally on the op; classic path resolves it without session-level `setExternalFile`

### Integration Points
- New suite `test/op/SGFP4MultiTensorTest.cpp` registers into `run_test.out` under `op/sgfp4/multi_tensor`
- `tools/fp4/README.md` is a new file referenced by the tool family (v3.0 converter work inherits it)
- Atomicity fix (D-11), if it changes `sgfp4_inject_core.hpp` write ordering, must keep the Phase 6 classic-API test's assumptions intact (same core, same behavior on success)
- Downstream consumer: `SGProcessingManager` (separate repo) — clean-failure behavior (D-09..D-11) is what protects its unchecked-nullptr path

</code_context>

<specifics>
## Specific Ideas

- User emphasized the structured container must come from the REAL gnus-poc encoder (not `encode_sgfp4.py`) but be baked to a self-contained C-array fixture — the cross-repo dependency exists only at authoring time (D-01..D-03).
- User chose the single-artifact proof strategy: one multi-tensor injection with one structured + one uniform container covers both SGINJ-07 and the quadtree half of SGINJ-08 in one run (D-05/D-06).
- User defined "fail cleanly" strictly: no partial output files may survive a failed run (D-11) — stronger than just non-zero exit.

</specifics>

<deferred>
## Deferred Ideas

- Same-shape weight disambiguation via manifest tensor-name keying — needed for real LLM models with repeated weight shapes; v3.0 Phase 11 (converter graph-rewrite pass design).
- Out-of-bounds-offset probing via hand-tampered artifacts — still considered unnecessary (the injector provably can't produce bad offsets: monotonic cursor + in-tool verify); revisit only if the artifact format gains external editors.
- Non-64-multiple weight shapes / tiling-padding conventions — v3.0 Phase 10.
- Real quantization-error tolerance calibration — v3.0 Phase 10.
- Vulkan E2E with injected artifacts — v3.0 Phase 12.
- gnus-poc `pipeline/runner.py` default-quantize fix (emits legacy v1 without `--adaptive`) — gnus-poc side, out of scope.

</deferred>

---

*Phase: 7-Multi-Tensor Hardening & Structured-Data Coverage*
*Context gathered: 2026-08-27*
