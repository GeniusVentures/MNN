# Phase 2: Adaptive Quadtree Layout (CPU, LAYOUT_MIXED) - Context

**Gathered:** 2026-08-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend the Phase 1 CPU SGFP4 v2 decode path to handle the variable-size `LAYOUT_MIXED` record via the pre-order-DFS quadtree split-map, and add the error-driven quadtree encoder that produces adaptive layouts (per-level thresholds, per-region mode selection with ε=0.10, ternary outlier veto, uniform-layout collapse). Completes the full SGFP4 v2 feature set on CPU. Requirements: SGV2-08, SGV2-09, SGV2-10, SGV2-11.

**Out of scope (locked by ROADMAP/REQUIREMENTS):** no Vulkan work (Phase 3/4), no FlatBuffers schema changes (quadtree lives only in container bytes), no E2M1 path changes, no encoder accuracy/perplexity benchmarking, no attestation/byte-exactness infrastructure.

</domain>

<decisions>
## Implementation Decisions

### Split-map decode (SGV2-08, SGV2-09)
- **D-01:** Decoder walks the quadtree **iteratively with an explicit fixed-size stack** (max depth 4 for 64→32→16→8→4; ≤85 nodes total). No recursion — the identical algorithm must port to the Phase 4 GLSL shader, where recursion is impossible. One traversal algorithm for CPU and GPU.
- **D-02:** Split-map is **validated strictly** during the walk: every bit read bounds-checked against the 12-byte/3-word map; leaf sizes/positions verified to tile the macroblock exactly; total leaf payload words cross-checked against record bounds; any malformed split-map = decode failure. Continues Phase 1's ASVS V5 untrusted-container posture.
- **D-03:** Split-map constants (3 words, 85 bits max) and the bit-reader helper go in a **new section of the existing `include/MNN/SGFP4DequantUtils.hpp`** — the wire-format definition stays in one header, consistent with Phase 1.

### Decode structure (SGV2-08, SGV2-09)
- **D-04:** LAYOUT_MIXED decode is a **branch inside the existing `dequant_sgfp4_container_cpu()`** (which currently rejects enum 4). Single public decode entry point, stays header-only. `CPUSGFP4Dequant` needs no new routing — it already dispatches through this function.

### Golden traversal test (SGV2-11)
- **D-05:** The golden pre-order DFS traversal-order check uses an **independent enumerator** — a separate small helper that enumerates expected leaf (x, y, n) coordinates for a given split-map. Decoder and enumerator are independent implementations of the same spec rule (Section 6.2), so a traversal bug in one is caught by the other. Do NOT share one walk between decoder and test.

### Encoder (SGV2-10)
- **D-06:** Encoder policy knobs are **locked to spec defaults, overridable via CLI flags**: ε=0.10 (T158 chosen iff `e_T158 ≤ (1+ε)·e_FP4`), per-level MSE thresholds tightening with depth (0.01 @64 → 0.0005 @4), ternary outlier veto, hysteresis against oscillation, recursion floor at 4×4. Uniform-layout collapse when all leaves share one size is normative (spec Section 6.3), not optional.
- **D-07:** The quadtree encoder **extends the existing `tools/fp4/encode_sgfp4.py`** — one reference encoder for the whole v2 format; `--selftest` and `--emit-cpp-fixture` stay unified (Phase 1 pattern). No separate quadtree script.

### Test strategy (SGV2-11)
- **D-08:** Mixed-layout fixtures are **encoder-generated and committed** (same pattern as Phase 1's 11 fixtures in `test/op/SGFP4DequantFixtures.h`): deterministic synthetic split-maps covering at minimum all-split, uniform-collapse (encoder must emit the uniform layout, not MIXED), and asymmetric mixed trees.
- **D-09:** Negative tests cover **split-map + size abuse** on top of Phase 1's malformed-container negatives: split bit on a 4×4 node, maps implying >85 nodes, leaves that don't tile the macroblock, truncated variable-size payloads, mixed records lying about leaf sizes.

### Claude's Discretion
- Internal encoder code organization within encode_sgfp4.py (function decomposition, CLI flag naming).
- Exact stack representation in the decoder (array + depth counter vs. small struct), provided it remains fixed-size and recursion-free.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### SGFP4 v2 specification
- `.planning/sgfp4-arxiv-v2.txt` §6.2 — split-map serialization (12-byte/3-word LE bitmap, pre-order DFS, TL/TR/BL/BR, ≥8 nodes carry bits, 4×4 always leaf, ≤85 bits), per-leaf headers (Eq. 6), per-leaf payloads (n²/8 or n²/16 words, 16-byte padding)
- `.planning/sgfp4-arxiv-v2.txt` §6.3 — exemplary encoder policy (per-level thresholds, ε mode selection, hysteresis, outlier veto, uniform-collapse rule)
- `.planning/sgfp4-arxiv-v2.txt` §4.3–4.4 — dual-mode payload packing and affine encode math (shared with Phase 1)

### Workstream planning
- `.planning/workstreams/sgfp4-pivot/ROADMAP.md` — Phase 2 goal, success criteria, locked roadmap notes 1–6
- `.planning/workstreams/sgfp4-pivot/REQUIREMENTS.md` — SGV2-08..11 normative text
- `.planning/quick/260821-p1q-evaluate-current-fp4-ultra-fp4-implement/SGFP4-PIVOT-ANALYSIS.md` — full gap analysis and decision history

### Phase 1 artifacts (decisions and code this phase builds on)
- `.planning/workstreams/sgfp4-pivot/phases/01-affine-dual-mode-decode-core-cpu-uniform-layouts/01-01-PLAN.md` — threat model T-01-03 (layout enum acceptance), schema/plumbing decisions
- `.planning/workstreams/sgfp4-pivot/phases/01-affine-dual-mode-decode-core-cpu-uniform-layouts/01-02-SUMMARY.md` — per-leaf mode selection decision, fixture generation pattern, externalPath plumbing, DoS-bound bugfix

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `include/MNN/SGFP4DequantUtils.hpp` — all Phase 1 constants and inline helpers (`sgfp4_align16`, `sgfp4_read_u32_le`, `unpack_leaf_header`, `sgfp4_decode_leaf_payload`, `dequant_sgfp4_container_cpu`). `kSGFP4LayoutMixed = 4` already defined and currently rejected; the enum table and bounds-check scaffolding are ready to extend.
- `source/backend/cpu/CPUSGFP4Dequant.cpp` — Execution class with external-sidecar loading (FileLoader + `std::ifstream` size probe). No changes expected for Phase 2; it routes through `dequant_sgfp4_container_cpu`.
- `tools/fp4/encode_sgfp4.py` — reference encoder with per-leaf Eq.5 mode selection, FP16 header packing, dual-mode payload packing, container writer, independent Python reference decoder, `--selftest` / `--emit-cpp-fixture` CLI. The quadtree encoder extends this file.
- `test/op/SGFP4DequantTest.cpp` + `test/op/SGFP4DequantFixtures.h` — fixture round-trip, edge-case, and malformed-negative test pattern to mirror for mixed layouts (`op/sgfp4/uniform_decode` → add e.g. `op/sgfp4/mixed_decode`).

### Established Patterns
- Header-only decode core with named constants (no magic numbers); every read bounds-checked against container size before it happens.
- Encoder-generated committed C++ fixtures keep Python encoder and C++ decoder in lockstep; `force_mode`-style overrides allowed for homogeneous test coverage.
- Sequential/linear decode order (locked in Phase 1) extends naturally: MIXED records append leaves in traversal order to the same linear output stream.

### Integration Points
- `dequant_sgfp4_container_cpu()` in SGFP4DequantUtils.hpp — the single place LAYOUT_MIXED acceptance is added (currently returns false for enum 4).
- `tools/fp4/encode_sgfp4.py` — gains the recursive error-driven subdivision and split-map emission.
- `test/op/` — new mixed-decode test case(s) registered in MNNTestSuite.

</code_context>

<specifics>
## Specific Ideas

- Golden traversal check: verify leaves are visited in pre-order DFS with quadrant order TL/TR/BL/BR, and that 4×4 nodes contribute no split bit (Success Criterion 1, SGV2-08).
- Uniform-collapse verification: an input whose leaves all share one size must cause the encoder to emit the corresponding uniform layout enum — not LAYOUT_MIXED (Success Criterion 2).
- Keep the known build blocker in mind: `test/op/FP4ModelTest.cpp` (pre-existing, from the `milestone` workstream) prevents a from-scratch `run_test.out` build; see Phase 1 `deferred-items.md`. Plan verification steps accordingly (same temporary-local-stub workaround, never committed).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (Phase 1's deferred `test/op/FP4ModelTest.cpp` build blocker remains owned by the `milestone` workstream's Phase 4 plan 04-02 and is unchanged by this phase.)

</deferred>

---

*Phase: 2-adaptive-quadtree-layout-cpu-layout-mixed*
*Context gathered: 2026-08-24*
