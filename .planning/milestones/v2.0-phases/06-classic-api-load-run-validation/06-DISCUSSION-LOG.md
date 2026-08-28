# Phase 6: Classic-API Load & Run Validation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-27
**Phase:** 6-Classic-API Load & Run Validation
**Areas discussed:** Test model & artifact scope, FP32 baseline & tolerance, Validation harness form, Failure-mode probing depth

---

## Test Model & Artifact Scope

### Q1: How should the classic-API test obtain its injected artifact?

| Option | Description | Selected |
|--------|-------------|----------|
| Inject in-test | Generate the base model at test time, run sgfp4_inject on it, then load the injected result via classic API. Self-contained, no committed fixtures, exercises the whole chain. | ✓ |
| Committed fixture | Commit the injected out.mnn + out.mnn.weight from Phase 5 under resource/ — simpler test, but fixtures drift from the tool and bloat the repo (132 KB sidecar). | |
| Base fixture + in-test inject | Commit only the small base .mnn; injection runs at test time. Middle ground if the container must live outside the repo. | |

**User's choice:** Inject in-test

### Q2: What should the test model's graph look like?

| Option | Description | Selected |
|--------|-------------|----------|
| Input → MatMul | Input[512] → MatMul(weight[512,512]) → output. Extends minimal_512 precedent, one real session input/output, hits the input-identification friction. | ✓ |
| Mixed two-MatMul graph | Input → MatMul(injected) → MatMul(FP32) → output. Proves coexistence of injected and normal weights but adds baseline-math complexity. | |
| Conv2D graph | Exercises the conv-weight use case, but Conv weights are 4-D vs the 2-D {dimO,dimI} pairing convention — risks dragging in Phase 7 scope. | |

**User's choice:** Input → MatMul

### Q3: Which Interpreter entry points should the classic-API test cover?

| Option | Description | Selected |
|--------|-------------|----------|
| createFromFile only | The exact SGProcessingManager downstream path; keep the test tight. | ✓ |
| Both file and buffer | Run the full flow twice — covers both SGINJ-05 entry points but doubles harness surface. | |
| createFromBuffer only | Presumes file path works without testing it — riskier assumption. | |

**User's choice:** createFromFile only

### Q4: Which SGFP4 v2 container feeds the injected weight?

| Option | Description | Selected |
|--------|-------------|----------|
| Demo container | Reuse gnus-poc demo.sgfp4 (512×512, byte-verified) — same input as Phase 5; structured data stays in Phase 7. | ✓ |
| Synthetic container | Generate in-test via encode_sgfp4.py oracle logic — self-contained but diverges from the real-artifact chain. | |
| Both | Extra coverage, likely redundant since decode correctness is proven elsewhere. | |

**User's choice:** Demo container

---

## FP32 Baseline & Tolerance

### Q1: What is the FP32/reference baseline the injected run is compared against?

| Option | Description | Selected |
|--------|-------------|----------|
| Original FP32 model | Load the SAME base model (pre-injection) via classic API, identical input; direct end-to-end comparison as SGINJ-06 wording requires. | ✓ |
| Manual decode+MatMul oracle | Compute expected output in-test from dequant_sgfp4_container_cpu + manual MatMul — re-proves decode, already proven in Phase 5. | |
| Both FP32 and oracle | Strongest but heaviest; oracle part duplicates Phase 5 in-tool verify. | |

**User's choice:** Original FP32 model

### Q2: How should the FP32 baseline weights relate to the container?

| Option | Description | Selected |
|--------|-------------|----------|
| Decoded container | Base model's FP32 weight = dequant_sgfp4_container_cpu(container) — injected-vs-FP32 difference zero by construction on weights; isolates classic-API plumbing from quantization error. | ✓ |
| Random weight + encode | Generate random FP32 weight, encode+inject its container — real quantization error requiring real tolerance calibration; adds encoder dependency. | |
| Unrelated weights | Keep minimal_512's original weight vs demo.sgfp4 — comparison would be meaningless noise. | |

**User's choice:** Decoded container

### Q3: What numeric tolerance for the injected-vs-FP32 output comparison?

| Option | Description | Selected |
|--------|-------------|----------|
| Tight rtol ~1e-4 | Pair-relative check consistent with Phase 5 in-tool verify; allows legitimate FP reassociation. | ✓ |
| Bit-exact | Strongest claim but classic-API session may reorder ops/use different gemm paths — flaky risk. | |
| Loose FP4-scale tolerance | Future-proofs for Phase 7 real comparisons; unnecessary now with identical weights. | |

**User's choice:** Tight rtol ~1e-4

### Q4: How should the session input tensor be filled?

| Option | Description | Selected |
|--------|-------------|----------|
| Deterministic in-code | Fixed LCG or hardcoded values — reproducible, self-contained, follows SGFP4DequantTest fixture precedent. | ✓ |
| Golden output vector | Hardcoded offline-computed output — brittle across backend paths. | |
| Random per-run | Varied coverage but non-reproducible failures. | |

**User's choice:** Deterministic in-code

---

## Validation Harness Form

### Q1: What form should the classic-API validation harness take?

| Option | Description | Selected |
|--------|-------------|----------|
| run_test.out suite | New test file registered as 'op/sgfp4/classic_api' — established pattern; filtered-suite workaround for FP4ModelTest blocker is known. | ✓ |
| Standalone tool binary | Avoids the broken full-suite build but lives outside the test framework and CI gates. | |
| Both suite and tool | Widest coverage but duplicates harness logic. | |

**User's choice:** run_test.out suite

### Q2: How does the test source the container?

| Option | Description | Selected |
|--------|-------------|----------|
| Generated fixture | Small generated .sgfp4 fixture or C-array header under test/op/ following SGFP4DequantFixtures.h precedent — not the 132 KB demo binary. | ✓ |
| Env-var path + skip | Try known locations, SKIP if not found — keeps real-artifact chain but tests silently skip on machines without gnus-poc. | |
| Commit demo.sgfp4 | Guarantees the exact real artifact but bloats the repo with a large binary. | |

**User's choice:** Generated fixture

### Q3: How should the injected artifact be produced in-test, given sgfp4_inject's input contract?

| Option | Description | Selected |
|--------|-------------|----------|
| Synthetic niche dir | Test writes manifest.json (correct sha256) + generated container to a temp dir, then injects from it — exercises the tool's real input contract (manifest parsing, sha256, version gate). | ✓ |
| Direct 05-01 recipe in-test | Bypasses the tool; Phase 6 then never validates actual tool output through classic API. | |
| Raw FlatBuffers surgery | Most control, most code, highest drift risk. | |

**User's choice:** Synthetic niche dir

### Q4: How does the test invoke the injection?

| Option | Description | Selected |
|--------|-------------|----------|
| Shared core header | Refactor sgfp4_inject core into tools/fp4/sgfp4_inject_core.hpp; tool main() and test both link it. No subprocess, cross-platform. | ✓ |
| Subprocess launch | Tests the shipped binary end-to-end but adds subprocess/portability complexity in the suite. | |
| Re-implement in test | No tool dependency but drifts from what the tool actually emits. | |

**User's choice:** Shared core header

---

## Failure-Mode Probing Depth

### Q1: Test the missing-sidecar failure mode under the classic path?

| Option | Description | Selected |
|--------|-------------|----------|
| Probe gracefully | Assert non-zero ErrorCode / nullptr rather than crash; documents behavior for the SGProcessingManager team (unchecked-nullptr path). | ✓ |
| Skip — Phase 7 scope | Happy path only; SGINJ-05/06 don't require negatives. | |
| Probe + fix runtime | Include runtime fixes if it crashes — scope-creep risk for this phase. | |

**User's choice:** Probe gracefully

### Q2: Probe corrupted container payload bytes?

| Option | Description | Selected |
|--------|-------------|----------|
| Skip — Phase 7 | Payload-level corruption is decode-domain, SGINJ-08 territory. | ✓ |
| Include garbage-payload test | Asserts graceful error or bounded garbage output; overlaps Phase 7 criteria. | |

**User's choice:** Skip — Phase 7

### Q3: Probe out-of-bounds sidecar offsets via a hand-tampered artifact?

| Option | Description | Selected |
|--------|-------------|----------|
| Skip | Injector provably can't produce bad offsets (monotonic cursor + in-tool verify); hand-tampering adds maintenance burden. | ✓ |
| Include tamper test | Patch external.offset out-of-bounds, re-serialize, assert graceful failure — maps to SGProcessingManager crash concern. | |

**User's choice:** Skip

### Q4: Explicitly assert session input/output tensor identification for the injected graph?

| Option | Description | Selected |
|--------|-------------|----------|
| Assert named I/O | Test asserts getSessionInputAll/getSessionOutputAll return the names the base model defined — surfaces the friction the success criteria warn about. | ✓ |
| Any-tensor minimal check | Grab whatever tensors are returned without asserting names — wouldn't catch silent I/O mismatch. | |

**User's choice:** Assert named I/O

---

## Claude's Discretion

- Exact fixture generation method (C-array header vs. generated .sgfp4 file written to temp), fixture size, temp-dir mechanism (portable, no `<filesystem>`).
- Test class/file naming within the SGFP4*Test.cpp pattern; suite registration string within `op/sgfp4/`.
- Shared core header exact signature/structure, provided both tool main() and the test use it without duplication.
- Whether the missing-sidecar probe is the same test case or a sibling case in the same file.
- Error-diagnostics wording and logging verbosity.

## Deferred Ideas

- `createFromBuffer` classic-API coverage — revisit when SGProcessingManager integration confirms buffer loads.
- Corrupted-payload and out-of-bounds-offset probing — Phase 7 (SGINJ-08).
- Real quantization-error tolerance calibration — v3.0 Phase 10 territory.
- Conv2D-weight (4-D) injection under classic API — v3.0.
- Multi-tensor / LAYOUT_MIXED structured-container classic-API runs — Phase 7 (structured container still to be obtained/generated from gnus-poc).
