# Phase 1: Vulkan Attention Correctness & LLM E2E - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 01-vulkan-attention-correctness-llm-e2e
**Areas discussed:** Test registration naming, Vulkan unavailability behavior, LLM model for E2E, Test determinism

---

## Test Registration Naming

| Option | Description | Selected |
|--------|-------------|----------|
| `op/vulkan/` | Prefix mirroring existing `op/` convention | ✓ |
| `engine/vulkan/` | Prefix for backend-level tests | |

**User's choice:** Stick with `op/vulkan/` — aligns with existing MNN test naming conventions under `test/op/`.
**Notes:** Test names: `op/vulkan/attention_correctness` and `op/vulkan/linear_attention_correctness`.

---

## Vulkan Unavailability Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Skip silently | Return true, no output | |
| Log warning then skip | Print warning, return true | ✓ |
| Fail loudly | Return false | |

**User's choice:** Log a warning when Vulkan is unavailable, then skip (return true).
**Notes:** Visible log output so CI operators or developers know why the tests aren't running. Not a hard failure since not all environments have Vulkan.

---

## LLM Model for E2E

| Option | Description | Selected |
|--------|-------------|----------|
| Qwen2-0.5B | Small but capable, good for testing | ✓ |
| TinyLlama | Alternative small model | |
| Phi-1.5 | Microsoft small model | |

**User's choice:** Qwen2-0.5B is preferred.
**Notes:** Model not currently available in `.mnn` format. This is a known blocker for Plan 03 Task 2. Model must be exported via MNN converter before E2E validation can proceed.

---

## Test Determinism

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed seed only | Reproducible, CI-friendly | |
| Variable data only | Catches dimension-specific bugs | |
| Both, default fixed | Fixed seed default, variable as secondary | ✓ |

**User's choice:** Default to fixed seed (matching existing MNN `TEST_RANDOM_SEED 100` convention), with variable/unseeded data acceptable as secondary coverage.
**Notes:** Aligns with existing MNN test patterns. Fixed seed preferred for CI reproducibility.

---

## the agent's Discretion

None — all gray areas were decided by the user.

## Deferred Ideas

None — discussion stayed within phase scope.
