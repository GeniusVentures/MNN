# Phase 12: End-to-End Validation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-01
**Phase:** 12-end-to-end-validation
**Areas discussed:** Accuracy gate definition, Vulkan validation approach, Validation artifact form, RunNetPass error escalation

---

## Accuracy gate definition

### Gate metric

| Option | Description | Selected |
|--------|-------------|----------|
| Numeric vs FP32 baseline | Final output tensor(s) vs FP32-converted baseline: max-abs + relative error thresholds, matching MNN checkVectorByRelativeError conventions | ✓ |
| Top-1/top-5 class match | Argmax over final logits must match FP32 baseline | |
| Both (gate + informational) | Hard numeric gate plus informational top-1 match | |

**User's choice:** Numeric vs FP32 baseline
**Notes:** Directly quantifies FP4 quantization damage; classification semantics add nothing.

### Tolerance

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 10-anchored thresholds | Reuse the tolerance framework Phase 10 established (real_weight_validation_report.json territory) | ✓ |
| Spec-derived conservative | Planner derives from FP4 spec error bounds with conservative multipliers | |
| Calibrate at plan time | Structure locked now, numbers from a calibration run | |

**User's choice:** Phase 10-anchored thresholds
**Notes:** Consistent with validated FP4 error characteristics on this exact corpus; no invented numbers.

### Input

| Option | Description | Selected |
|--------|-------------|----------|
| Deterministic synthetic input | Seeded pseudo-random/fixed tensor, SGFP4TestUtil fixture style | ✓ |
| Real image asset | Real image through AlexNet preprocessing | |
| Synthetic gate + image info | Gate on synthetic; informational real-image run | |

**User's choice:** Deterministic synthetic input
**Notes:** Reproducible anywhere, no asset dependency.

### Baseline

| Option | Description | Selected |
|--------|-------------|----------|
| FP32 .mnn same path | Same ONNX converted WITHOUT --sgfp4, identical session/input path | ✓ |
| External framework output | ONNX Runtime / PyTorch ground truth | |
| FP32 gate + external sanity | Gate vs FP32 .mnn; spot-check one tensor externally | |

**User's choice:** FP32 .mnn same path
**Notes:** Isolates quantization error within the same engine and graph.

---

## Vulkan validation approach

### Vulkan path

| Option | Description | Selected |
|--------|-------------|----------|
| Classic API, VULKAN type | Same artifact through classic-API session with MNN_FORWARD_VULKAN | ✓ |
| Express/Module path | Express/Module with Vulkan Executor config | |
| Classic gate + Module info | Primary classic API; supplementary Module run | |

**User's choice:** Classic API, VULKAN type
**Notes:** Mirrors the CPU leg and the downstream SGProcessingManager consumption shape.

### Vulkan compare

| Option | Description | Selected |
|--------|-------------|----------|
| Both vs FP32 baseline | Vulkan and CPU each independently gated vs the same FP32 baseline, same tolerance | ✓ |
| Vulkan vs CPU-SGFP4 | Backend parity gate only | |
| FP32 gate + parity assert | Hard FP32 gate plus explicit cross-backend parity threshold | |

**User's choice:** Both vs FP32 baseline
**Notes:** One baseline, two gates; implicitly covers cross-backend parity.

### No-Vulkan env

| Option | Description | Selected |
|--------|-------------|----------|
| SKIP when unavailable | Vulkan leg reports SKIP; CPU gate mandatory | |
| Hard requirement | No Vulkan capability fails the phase | ✓ |
| Follow repo convention | Match existing Vulkan test handling | |

**User's choice:** Hard requirement
**Notes:** No SKIP semantics — validation machine must have working Vulkan.

---

## Validation artifact form

### Artifact

| Option | Description | Selected |
|--------|-------------|----------|
| Committed validation script | PowerShell script drives full E2E; corpus path as parameter (Phase 11 D-13 precedent) | ✓ |
| run_test.out suite | Committed suite that skips when corpus env var absent | |
| Synthetic suite + corpus script | Small always-on synthetic E2E plus real-corpus script | |

**User's choice:** Committed validation script
**Notes:** Also avoids the tension with D-07 (skip-when-absent suite would contradict the hard Vulkan requirement).

### Compare host

| Option | Description | Selected |
|--------|-------------|----------|
| Script drives native tools | Script shells out to existing MNN tools; comparison logic in script | ✓ |
| Dedicated C++ validator | New tools/fp4 executable doing convert→run→compare internally | |
| You decide | Planner picks the least-surface pattern | |

**User's choice:** Script drives native tools
**Notes:** Least new native code; no new build target.

### Failure detail

| Option | Description | Selected |
|--------|-------------|----------|
| Final-output diagnostics | Per-tensor max-abs, rel-error, failing index on failure | ✓ |
| Per-layer tracing | Walk intermediates to localize error explosion | |
| Minimal output | Pass/fail with max-error print only | |

**User's choice:** Final-output diagnostics
**Notes:** Enough to diagnose tolerance-vs-bug without new tooling surface.

---

## RunNetPass error escalation

### Escalation

| Option | Description | Selected |
|--------|-------------|----------|
| Fix it in Phase 12 | --sgfp4 + pass failure → non-zero exit with clear MNN_ERROR | ✓ |
| Defer, script asserts instead | Phase 12 stays pure validation; E2E script asserts node count | |
| Fix + script assertion | Fix exit code AND script-side node-presence assertion | |

**User's choice:** Fix it in Phase 12
**Notes:** Phase 12 depends on the flag's exit-code honesty for its own gating.

### Fix scope

| Option | Description | Selected |
|--------|-------------|----------|
| SGFP4-scoped only | Only InsertSGFP4Dequant failure escalates; zero impact elsewhere | ✓ |
| General RunNetPass errors | All pass failures become converter errors | |
| You decide mechanism | Planner picks, constrained to flag-on-only impact | |

**User's choice:** SGFP4-scoped only
**Notes:** Generalizing touches every conversion path — regression risk beyond this workstream's mandate.

---

## Claude's Discretion

- Exact tolerance values derived from the Phase 10 report data
- Which existing MNN tool/binary the script shells out to for execution/dumping
- Script location/naming, parameter spelling, README placement
- Exact escalation mechanism in the converter (constrained to D-12)
- Whether the script additionally asserts SGFP4 node presence
- Synthetic input generator structure (seed handling, value range) — deterministic and documented

## Deferred Ideas

- Per-layer error tracing tooling
- Generalized RunNetPass failure semantics
- Performance benchmarking (SGV2-33)
- Additional backends (SGV2-35)
- Corpus expansion / exotic-op sweep beyond AlexNet
- MatMul/LLM-export path rewriting
