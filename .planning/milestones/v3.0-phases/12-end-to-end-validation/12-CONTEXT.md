# Phase 12: End-to-End Validation - Context

**Gathered:** 2026-09-01
**Status:** Ready for planning

<domain>
## Phase Boundary

A real model (the Phase 10 approved corpus, `W:\gnus\models\alexnet_Opset16.onnx`) converts via `mnnconvert --sgfp4` and runs **numerically correct inference on both CPU and Vulkan** — closing the v3.0 milestone (SGV2-31 CPU, SGV2-32 Vulkan). Phase 11 handed off an artifact proven to convert, load, and execute (`runSession` = NO_ERROR); this phase adds the formal **output-accuracy gates** on both backends, delivered as one committed validation script, plus the one flagged converter fix: RunNetPass log-only failure escalation (`--sgfp4` + pass failure must exit non-zero).

Scope anchor (requirements SGV2-31/32, from `.planning/milestones/v2.0-REQUIREMENTS.md` §"v3.0 Converter Integration"):
- SGV2-31: CPU end-to-end inference correctness via the new flag
- SGV2-32: Vulkan end-to-end inference correctness via the new flag

Includes: FP32-baseline comparison methodology (deterministic synthetic input, numeric max-abs + relative-error gates with Phase 10-anchored tolerances), the Vulkan classic-API execution leg, the committed E2E validation script with failure diagnostics, and the SGFP4-scoped RunNetPass error-escalation fix in mnnconvert.

Excludes: encoder/decoder/pass changes beyond the exit-code fix (Phases 8-11 outputs consumed as-shipped); per-layer error-tracing tooling; exotic-op corpus expansion beyond AlexNet; performance benchmarking (SGV2-33 territory); other backends (Metal/CUDA/OpenCL, SGV2-35); MatMul/LLM-path rewriting (deferred); any gnus-poc-side change.

</domain>

<decisions>
## Implementation Decisions

### Accuracy gate definition (SGV2-31 core)
- **D-01 (numeric gate, not classification):** The formal gate is a numeric comparison of the final output tensor(s) against the FP32 baseline — max-abs + relative error, matching MNN test conventions (`checkVectorByRelativeError` / `checkVector` style). Top-1/top-5 class match is NOT the gate (AlexNet head classification semantics add nothing to quantization-error measurement).
- **D-02 (Phase 10-anchored tolerances):** Tolerance numbers are anchored to Phase 10's validated threshold/error data (`tools/fp4/real_weight_validation_report.json` territory) — consistent with measured FP4 error characteristics on this exact corpus; no invented numbers. Exact per-gate values locked at plan time from that data.
- **D-03 (deterministic synthetic input):** Input is a seeded deterministic synthetic tensor (SGFP4TestUtil fixture style) — fully reproducible, no image asset or preprocessing dependency.
- **D-04 (FP32-baseline = same path):** Baseline is the SAME ONNX converted WITHOUT `--sgfp4`, run through the identical session/input path — apples-to-apples within MNN, isolating quantization error. No external-framework (ORT/PyTorch) ground truth in the gate.

### Vulkan validation approach (SGV2-32 core)
- **D-05 (classic API, VULKAN forward type):** The Vulkan leg runs the SAME converted artifact through a classic-API session (`Interpreter::createFromFile` → `createSession` with `MNN_FORWARD_VULKAN` → `runSession`) — mirroring the CPU leg exactly and the downstream `SGProcessingManager` consumption shape. Express/Module path is not the gate.
- **D-06 (both backends vs the same FP32 baseline):** Vulkan output is compared against the SAME FP32 baseline with the SAME tolerance as CPU — one baseline, two independent backend gates (implicitly covering cross-backend parity without a second threshold to maintain).
- **D-07 (Vulkan is a hard requirement):** No SKIP semantics — an environment without Vulkan capability FAILS the phase. The validation machine must have a working Vulkan device/driver.

### Validation artifact form
- **D-08 (one committed validation script):** The deliverable is a single committed script (PowerShell, matching the repo's Windows validation precedent) driving the full E2E: FP32 convert → `--sgfp4` convert → CPU session run → Vulkan session run → output comparison. Corpus path is a script parameter (Phase 11 D-13 precedent — corpus stays a test-time dependency, not a committed asset). No new `run_test.out` suite, no synthetic in-repo net.
- **D-09 (script drives native tools):** Comparison logic lives in the script, shelling out to existing MNN tools/binaries that execute sessions and dump outputs — no new dedicated C++ validator build target. (Planner picks the exact existing tool/dump mechanism.)
- **D-10 (final-output diagnostics):** On failure, the gate emits per-backend final-output diagnostics: max-abs error, relative error, failing index — sufficient to distinguish tolerance issue from real bug. No per-layer tracing tooling (that is follow-up territory if ever needed).

### Carry-forward: RunNetPass error escalation (Phase 11 verification flag)
- **D-11 (fix it here):** When `--sgfp4` is set and `InsertSGFP4Dequant` fails or transactionally skips, mnnconvert must exit non-zero with a clear `MNN_ERROR` — never print "Converted Success!" over a silently-FP32 artifact. Phase 12 depends on the flag's exit-code honesty for its own gating, so the fix belongs here.
- **D-12 (SGFP4-scoped only):** The escalation touches only the SGFP4 path — zero behavior change for any other pass or for flag-off conversions. Generalized RunNetPass error semantics are explicitly out of scope.

### Claude's Discretion
- Exact tolerance values derived from the Phase 10 report data (structure is locked: max-abs + relative).
- Which existing MNN tool/binary the script shells out to for session execution and output dumping (D-09), and the exact dump/comparison mechanics.
- Script location and naming (e.g. `tools/fp4/` alongside the W-2 probe precedent vs. a scripts dir), parameter spelling, and README/documentation placement.
- The exact escalation mechanism in the converter (pass-result propagation point), constrained to D-12's zero-flag-off-impact guarantee.
- Whether the script additionally asserts SGFP4 node presence in the artifact (cheap sanity, complements D-11).
- Structure of the synthetic input generator (seed handling, value range) as long as it is deterministic and documented.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Workstream planning (phase contracts this phase closes)
- `.planning/workstreams/sgfp4-pivot/ROADMAP.md` §Phase 12 — goal ("a real model converts and runs correct inference on CPU and Vulkan via the new flag"), SGV2-31/32 mapping, success criteria TBD-at-plan-time note
- `.planning/milestones/v2.0-REQUIREMENTS.md` §"v3.0 Converter Integration" — SGV2-31/32 requirement text; §"Performance & Coverage" SGV2-33/35 (explicitly out of scope)
- `.planning/workstreams/sgfp4-pivot/phases/11-graph-rewrite-postconverter-pass-cli-flag/11-VERIFICATION.md` §Notes — the two Phase 12 hand-offs: 4-D dims learning (conv consumers require `{O,I,kH,kW}` geometry) and the RunNetPass log-only failure flag D-11 resolves
- `.planning/workstreams/sgfp4-pivot/phases/11-graph-rewrite-postconverter-pass-cli-flag/11-CONTEXT.md` — D-13 (CLI smoke precedent: corpus as test-time dependency), D-14 (Phase 12 owns formal accuracy gates), buffer contract
- `.planning/workstreams/sgfp4-pivot/phases/10-real-weight-validation-against-actual-model-statistics/10-CONTEXT.md` — D-01/D-02 corpus lock (AlexNet, sha256 provenance), D-03 light-tier rule
- `tools/fp4/real_weight_validation_report.json` — the validated error/threshold data D-02 anchors tolerances to

### Converter pipeline (D-11/D-12 fix sites)
- `tools/converter/source/optimizer/PostConverter.cpp` — `RunNetPass` mechanics (~:393 final pass batch), the log-only failure behavior D-11 escalates
- `tools/converter/source/optimizer/postconvert/InsertSGFP4Dequant.cpp` — the pass whose failure must propagate; transactional-skip semantics
- `tools/converter/source/common/cli.cpp` — `--sgfp4` parse (`:230`/`:493`) and mutex (`:577`) — where flag state is available for scoped escalation
- `tools/converter/include/config.hpp` — `modelConfig::useSGFP4` — the config field gating the escalation

### Runtime consumers (execution legs the script drives)
- `source/backend/cpu/CPUSGFP4Dequant.cpp` — buffer-first CPU decode (read-only ground truth)
- `source/backend/vulkan/` SGFP4 Vulkan Execution — the Vulkan decode leg (Phase 3/4 output)
- `demo/exec/pictureRecognition.cpp` — classic-API session execution precedent (createFromFile → createSession → runSession, input/output tensor identification)
- `include/MNN/SGFP4DequantUtils.hpp` — format spec ground truth (tolerance context)

### Test assets (script precedents and fixture style)
- `tools/fp4/README.md` — tool documentation conventions where the validation script usage is documented
- `tools/fp4/w2_failcleanup_probe.ps1` — committed PowerShell probe precedent (Phase 11 W-2) for D-08's script form
- `test/op/SGFP4TestUtil.hpp` — deterministic fixture generation style (D-03)
- `test/op/SGFP4ClassicAPITest.cpp` — classic-API load/run test precedent including external-sidecar handling

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `w2_failcleanup_probe.ps1` (Phase 11) — committed PowerShell probe pattern: build, execute, assert exit code/artifacts — the direct template for D-08's script
- Phase 11's PHASE B pattern (11-05) — classic-API load/run of the converted corpus artifact (`Interpreter::createFromFile` → `createSession` → `resizeSession` → `runSession`), already proven on `tmp\p11_smoke.mnn`; D-05 extends it to `MNN_FORWARD_VULKAN`
- `TestSGFP4Converter.exe` + the 13 `op/sgfp4` suites — the no-regression baseline the phase must keep green
- Phase 10's report tooling — the error-statistics methodology behind D-02's anchored tolerances

### Established Patterns
- One FP32 baseline, two backend gates (D-06) — mirrors the v1.0 CPU↔Vulkan parity suite philosophy but anchored to FP32 instead of cross-backend
- Corpus-as-parameter scripted validation (Phase 11 D-13) — D-08 follows it exactly; no corpus bytes ever enter the repo
- Flag-state availability through `Global<modelConfig>` — D-11's escalation reads `useSGFP4` exactly where the mutex already does
- Deterministic seeded fixtures (`SGFP4TestUtil`) — D-03's input style

### Integration Points
- New validation script (location at planner's discretion, `tools/fp4/` likely) — drives existing `MNNConvert`/test binaries; no new build target
- Converter exit-code change — `PostConverter.cpp`/`MNNConverter.cpp` return path, gated on `useSGFP4` + pass result
- Documentation — README section describing the E2E validation procedure and its hard Vulkan requirement

</code_context>

<specifics>
## Specific Ideas

- The E2E flow must be a single script invocation: `script.ps1 -Corpus <alexnet.onnx>` producing a clear PASS/FAIL per backend with D-10 diagnostics — the milestone's demonstrable artifact.
- Terminology lock applies everywhere user-visible: "SGFP4 v2", never "Ultra FP4".
- Corpus continuity: reuse `W:\gnus\models\alexnet_Opset16.onnx` (Phase 10 approved, sha256 `4bc388cc…`) — no new corpus approval.

</specifics>

<deferred>
## Deferred Ideas

- Per-layer error tracing / intermediate-tensor extraction tooling — follow-up if a gate failure ever needs localization beyond final-output diagnostics (D-10)
- Generalized RunNetPass failure semantics (all passes escalate) — converter-wide correctness change, out of this workstream's mandate (D-12)
- Performance benchmarking of SGFP4 inference (SGV2-33 GPU decode tuning) — future requirement
- Additional backends (Metal/CUDA/OpenCL, SGV2-35) — future requirement
- Corpus expansion / exotic-op robustness sweep beyond AlexNet — possible follow-up validation phase
- MatMul/LLM-export path rewriting — deferred since Phase 11, remains post-v3.0 territory
</deferred>
