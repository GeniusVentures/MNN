# Phase 6: Classic-API Load & Run Validation - Context

**Gathered:** 2026-08-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Prove Phase 5's injected artifacts load and run through the **classic** API — `Interpreter::createFromFile` → `createSession` → `runSession` (the exact `SGProcessingManager::MNN_Tensor::Process()` downstream path) — with session input/output identification working and end-to-end inference matching an FP32/reference baseline within defined tolerance on CPU, via the existing decode Execution. External-sidecar resolution must work under the classic path (external path arrives via the op itself, not a session-level `setExternalFile`) — SGINJ-05/SGINJ-06.

Out of scope: multi-tensor/structured-data (LAYOUT_MIXED) coverage, malformed-input hardening beyond the missing-sidecar probe, runtime decode changes (Phase 7 and v3.0 Phases 8-12 territory).

Express/`Module::load` artifact validity is already proven (Phase 5 in-tool verify, D-12) — this phase adds the classic-API proof only.

</domain>

<decisions>
## Implementation Decisions

### Test Model & Artifact Scope
- **D-01:** Inject in-test — the test generates the base model at test time, runs the injection, then loads the injected result via the classic API. No committed `.mnn`/sidecar fixtures; fully self-contained and immune to tool-output drift.
- **D-02:** Base model topology is `Input[512] → MatMul(weight[512,512]) → output` — direct extension of Phase 5's `minimal_512.mnn` precedent, with exactly one real session input/output so the input-identification friction flagged in the success criteria is actually exercised (the prior PoC graph had zero inputs).
- **D-03:** Classic-API entry point is `createFromFile` only (`createFromBuffer` is mentioned by SGINJ-05 but the file path is the documented SGProcessingManager reference flow; buffer coverage is not worth doubling the harness surface this phase).
- **D-04:** The injected weight source uses the demo-container chain established in Phase 5 (512×512, byte-verified, all `UNIFORM_64`) — structured/quadtree containers remain Phase 7 by design.

### FP32 Baseline & Tolerance
- **D-05:** The FP32 baseline is the SAME base model (pre-injection) loaded via classic API and run with the identical input — a direct end-to-end injected-vs-FP32 comparison as SGINJ-06 wording requires.
- **D-06:** The base model's FP32 weight is the **decoded container** (`dequant_sgfp4_container_cpu` of the same container bytes). Injected-vs-FP32 output difference is then zero-by-construction on weights — isolating classic-API plumbing correctness from quantization error entirely (decode-vs-decode is already proven in Phase 5).
- **D-07:** Tolerance is a tight pair-relative check (`checkVectorByRelativeError`-style, rtol ~1e-4), consistent with Phase 5's in-tool verify tolerance; bit-exactness is NOT required (legitimate FP reassociation across sessions may differ).
- **D-08:** Session input tensor is filled deterministically in-code (fixed values / LCG) — reproducible failures, self-contained, follows the `SGFP4DequantFixtures.h` precedent. No golden-output vector, no per-run randomness.

### Validation Harness Form
- **D-09:** The harness is a new `run_test.out` suite (e.g. `test/op/SGFP4ClassicAPITest.cpp`, registered under `op/sgfp4/classic_api`), using the established filtered-suite workaround for the known `FP4ModelTest.cpp` full-build blocker (STATE.md).
- **D-10:** The container fixture is **generated** (small `.sgfp4` or C-array header under `test/op/`, following the `SGFP4DequantFixtures.h` precedent) — NOT the 132,368-byte `demo.sgfp4` committed to the repo, and NOT an env-var-dependent external path that silently skips.
- **D-11:** The test exercises the tool's **real input contract**: it writes a synthetic niche dir (manifest.json with the correct sha256 computed over the generated container + the container file) to a temp dir at runtime, then injects from it — covering manifest parsing, sha256 verification, and the version gate on the way to the classic-API load.
- **D-12:** Injection invocation is via a **shared core header**: refactor `sgfp4_inject.cpp`'s core into `tools/fp4/sgfp4_inject_core.hpp` (function: model path + niche dirs + output path → exit int), with the tool's `main()` and the test both linking it. No subprocess launch, no re-implementation in test code.

### Failure-Mode Probing Depth
- **D-13:** **Probe missing sidecar**: with the `.weight` sidecar absent, classic-API load/run must fail gracefully (non-zero ErrorCode / nullptr) rather than crash — documents actual behavior for the downstream `SGProcessingManager` team (their code has an unchecked-nullptr path).
- **D-14:** **Skip** corrupted-payload-byte probing — decode-domain robustness is Phase 7 (SGINJ-08) territory.
- **D-15:** **Skip** out-of-bounds-offset probing via hand-tampered artifacts — the injector provably can't produce bad offsets (monotonic cursor + in-tool verify); Phase 7's clean-failure criteria can adopt it if ever needed.
- **D-16:** The test **explicitly asserts named session input/output identification** (`getSessionInputAll`/`getSessionOutputAll` returning the names the base model defined) — if names differ after injection, the test fails and surfaces exactly the friction the success criteria warn about (no "grab first tensor" minimalism).

### Claude's Discretion
- Exact fixture generation method (C-array header vs. generated `.sgfp4` file written to temp), fixture size (smallest sufficient), temp-dir mechanism (portable, no `<filesystem>` per Phase 5 precedent).
- Test class/file naming beyond following the `SGFP4*Test.cpp` pattern, and thesuite registration string within the `op/sgfp4/` namespace.
- The shared core header's exact signature/structure (single function vs. small set), provided both tool `main()` and the test use it without duplication.
- Whether the missing-sidecar probe lives in the same test case or a sibling test case in the same file.
- Error-diagnostics wording and logging verbosity.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Classic-API reference flow (in-repo)
- `demo/exec/pictureRecognition.cpp` — canonical classic-API demo SGINJ-05 cites: `Interpreter::createFromFile` → `createSession` → `getSessionInput`/`getSessionOutput` → `runSession`
- `include/MNN/Interpreter.hpp` — public classic-API surface (createFromFile, createSession, getSessionInputAll/getSessionOutputAll)
- `source/core/Session.cpp` / `Pipeline.cpp` — classic-path execution internals (op scheduling, error propagation relevant to D-13)

### Injection tool (Phase 5 output, refactored this phase)
- `tools/fp4/sgfp4_inject.cpp` — the tool whose core becomes `sgfp4_inject_core.hpp` (D-12); CLI contract manifest-driven niche dirs, sidecar `<output>.weight`
- `tools/fp4/sha256.hpp` — vendored SHA-256 used for the synthetic niche-dir manifest (D-11)
- `tools/fp4/CMakeLists.txt` — tool build wiring; the new shared header changes what's compiled here (header-only refactor must keep the `sgfp4_inject.out` target building)

### Test precedents (in-repo)
- `test/op/SGFP4DequantTest.cpp` — op-level construction + round-trip test; pattern for the new classic-API test file
- `test/op/SGFP4DequantFixtures.h` — generated-fixture precedent for D-10
- `test/TestUtils.h` — `checkVectorByRelativeError` assertion helpers (D-07)
- `test/CMakeLists.txt` — suite build (glob `*.cpp`); known `FP4ModelTest.cpp` blocker requires filtered-suite verification (STATE.md pending todo)

### Decode ground truth (in-repo)
- `include/MNN/SGFP4DequantUtils.hpp` — `dequant_sgfp4_container_cpu` (D-06 FP32-weight source), container framing constants
- `source/backend/cpu/CPUSGFP4Dequant.cpp` — the Execution that must fire under the classic path, incl. its external-file failure behavior (D-13)

### Workstream planning
- `.planning/workstreams/sgfp4-pivot/REQUIREMENTS.md` — SGINJ-05, SGINJ-06
- `.planning/workstreams/sgfp4-pivot/ROADMAP.md` §"Phase 6" — success criteria 1–3
- `.planning/workstreams/sgfp4-pivot/STATE.md` — locked decisions (externalPath gotcha, terminology, FP4ModelTest blocker, structured-container Phase 7 caveat)
- `.planning/workstreams/sgfp4-pivot/phases/05-injection-core-artifact-construction-graph-splicing/05-CONTEXT.md` — Phase 5 decisions D-01..D-13 this phase builds on
- `.planning/workstreams/sgfp4-pivot/phases/05-injection-core-artifact-construction-graph-splicing/05-02-SUMMARY.md` — tool implementation details (offset cursor, naming, verify flow) the refactor must preserve

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/fp4/sgfp4_inject.cpp` — entire tool core: manifest parsing, sha256, pairing, OpT construction, sidecar merge, `Variable::save`, in-tool Module::load verify (D-12 refactor source)
- `dequant_sgfp4_container_cpu` (`SGFP4DequantUtils.hpp`) — produces both the fixture bytes' FP32-decoded weight (D-06) and, indirectly, the fixture sha256 source
- `test/TestUtils.h` `checkVectorByRelativeError` — D-07 tolerance check
- `Expression`/`Variable` Express APIs as used in Phase 5 tests — base-model construction (`_Input`, `MatMul` op) in-test

### Established Patterns
- Classic-API session flow as in `demo/exec/pictureRecognition.cpp`: file → session → named I/O tensors → resize → run → read (D-03/D-16)
- Manifest-driven niche-dir input contract with sha256 integrity (Phase 5 D-01/D-03) — test must satisfy it exactly (D-11)
- `op->externalPath` set literally on the op; classic path resolves it without session-level `setExternalFile` (SGINJ-06; the thing being proven)
- Generated C-array fixtures (`SGFP4DequantFixtures.h`) instead of committed binaries (D-10)
- Filtered-suite run (`op/sgfp4/...`) to dodge the `FP4ModelTest.cpp` full-build blocker (D-09)

### Integration Points
- Test registers into `run_test.out` under `op/sgfp4/classic_api` (or similar) — CI/aggregation uses suite strings
- `tools/fp4/CMakeLists.txt` gains the core header include; `sgfp4_inject.out` target must still build standalone
- Downstream consumer of the proof is `SGProcessingManager::MNN_Tensor::Process()` (separate repo/workstream) — D-13's graceful-failure documentation is written for that team
- Learnings flow forward to Phase 7 (multi-tensor, structured data) and v3.0 Phases 8–12 (converter integration inherits sidecar/classic-load conventions)

</code_context>

<specifics>
## Specific Ideas

- User emphasized (via selections) maximizing reuse of the exact Phase 5 chain: same topology (`Input[512] → MatMul[512,512]`), same demo-container lineage, but generated fixtures instead of committed binaries.
- User chose the zero-by-construction baseline (D-06): this phase validates classic-API **plumbing**, deliberately deferring real quantization-error tolerance to when real FP32-vs-quantized comparisons exist (Phase 7+/v3.0).
- No other specific references — standard approaches accepted elsewhere.

</specifics>

<deferred>
## Deferred Ideas

- `createFromBuffer` classic-API coverage — mentioned in SGINJ-05 text; only worth adding if SGProcessingManager actually loads from memory buffers (revisit when that integration work starts).
- Corrupted-payload-byte and out-of-bounds-offset failure probing — Phase 7 (SGINJ-08 malformed-input hardening).
- Real quantization-error tolerance calibration (random FP32 weight → encode → compare with FP4-scale rtol) — deferred until a real FP32-vs-quantized comparison exists; candidate for v3.0 Phase 10 (real-weight validation).
- Conv2D-weight injection under classic API (4-D weights vs the 2-D `{dimO, dimI}` pairing convention) — v3.0 territory.
- Multi-tensor / LAYOUT_MIXED structured-container classic-API runs — Phase 7 by design (structured container must still be obtained or generated from gnus-poc first; STATE.md pending todo).

</deferred>

---

*Phase: 6-Classic-API Load & Run Validation*
*Context gathered: 2026-08-27*
