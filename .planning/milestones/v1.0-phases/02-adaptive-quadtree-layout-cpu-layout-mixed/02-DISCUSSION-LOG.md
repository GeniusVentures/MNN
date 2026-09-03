# Phase 2: Adaptive Quadtree Layout (CPU, LAYOUT_MIXED) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-24
**Phase:** 2-adaptive-quadtree-layout-cpu-layout-mixed
**Areas discussed:** Split-map bit layout & traversal, Encoder decision policy, Decode structure, Test strategy & golden split-map

**Process note:** The interactive question tool returned only the last answer of each multi-question batch, causing the orchestrator to re-ask earlier questions. User flagged the repetition; remaining questions were locked at their recommended defaults with user consent ("If any of those are wrong, say so" — no corrections given). Selections below marked * were explicitly confirmed by the user; unmarked ones are recommended defaults accepted implicitly.

---

## Split-map bit layout & traversal

| Option | Description | Selected |
|--------|-------------|----------|
| Iterative explicit stack | Fixed-size stack (depth ≤4, ≤85 nodes); no recursion; ports to Phase 4 GLSL | ✓ |
| Recursive C++ walk | Cleaner CPU code, but Phase 4 shader needs a separate iterative version | |
| You decide | Researcher/planner picks | |

**User's choice:** Iterative explicit stack (recommended default, accepted)
**Notes:** Phase 4's GLSL shader cannot recurse — one traversal algorithm for CPU and GPU.

| Option | Description | Selected |
|--------|-------------|----------|
| Validate strictly | Bounds-checked bits, tile-exact verification, malformed = failure (ASVS V5 posture) | ✓ * |
| Trust after header check | Validate header only, walk assuming well-formed | |

**User's choice:** Validate strictly *

| Option | Description | Selected |
|--------|-------------|----------|
| Independent enumerator | Separate helper enumerates expected leaf coords; catches decoder bugs | ✓ * |
| Single shared walk | One traversal shared by decoder and tests — bugs pass silently | |
| You decide | Planner picks | |

**User's choice:** Independent enumerator *

| Option | Description | Selected |
|--------|-------------|----------|
| Same header, new section | Split-map constants in SGFP4DequantUtils.hpp — wire format in one place | ✓ * |
| Separate quadtree header | New SGFP4Quadtree.hpp — splits format definition across headers | |
| You decide | Planner picks | |

**User's choice:** Same header, new section *

---

## Encoder decision policy

| Option | Description | Selected |
|--------|-------------|----------|
| Lock spec defaults, expose CLI flags | ε=0.10, per-level MSE thresholds, veto, hysteresis, floor — overridable via CLI | ✓ * |
| Hard-code defaults only | No knobs — simplest, matches "exemplary reference" scope | |
| You decide | Planner picks | |

**User's choice:** Lock spec defaults, expose CLI flags *

| Option | Description | Selected |
|--------|-------------|----------|
| Extend encode_sgfp4.py | One reference encoder for the whole v2 format | ✓ |
| Separate quadtree script | New encode_sgfp4_quadtree.py — duplicated container code | |
| You decide | Planner picks | |

**User's choice:** Extend encode_sgfp4.py (recommended default, accepted)

---

## Decode structure

| Option | Description | Selected |
|--------|-------------|----------|
| Extend existing function | LAYOUT_MIXED branch inside dequant_sgfp4_container_cpu — single entry point | ✓ * |
| Separate mixed decoder | New dequant_sgfp4_mixed_record() called by the existing function | |
| You decide | Planner picks | |

**User's choice:** Extend existing function *

---

## Test strategy & golden split-map

| Option | Description | Selected |
|--------|-------------|----------|
| Encoder-generated, committed | Deterministic synthetic split-map fixtures (all-split, uniform-collapse, asymmetric) | ✓ |
| Hand-built fixtures | Byte-by-hand construction — duplicates wire-format knowledge | |
| You decide | Planner picks | |

**User's choice:** Encoder-generated, committed (recommended default, accepted)

| Option | Description | Selected |
|--------|-------------|----------|
| Split-map + size abuse | Malformed split-maps, truncated payloads, lying leaf sizes + Phase 1 negatives | ✓ * |
| Minimal round-trip only | Round-trip + golden traversal; skip adversarial cases | |
| You decide | Planner picks | |

**User's choice:** Split-map + size abuse *

---

## Claude's Discretion

- Internal encoder code organization within encode_sgfp4.py (function decomposition, CLI flag naming)
- Exact stack representation in the decoder (fixed-size, recursion-free is the constraint)

## Deferred Ideas

None — discussion stayed within phase scope.
