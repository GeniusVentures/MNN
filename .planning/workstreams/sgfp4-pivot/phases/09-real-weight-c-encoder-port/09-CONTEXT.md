# Phase 9: Real-Weight C++ Encoder Port - Context

**Gathered:** 2026-08-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Port the Python quadtree/dual-mode SGFP4 v2 encoder (gnus-poc `fp4_exporter.py --adaptive`) to C++, operating on real (non-64-aligned) weight tensor shapes. The C++ encoder produces container bytes that are byte-compatible with the existing unchanged decoders (CPU `SGFP4DequantUtils` oracle + Vulkan Execution) and land in `SGFP4DequantParam.buffer` per Phase 8's D-11 staging contract — consumed downstream by Phase 11's graph-rewrite PostConverter pass.

Scope anchor (requirements SGV2-24/25, from `.planning/workstreams/sgfp4-pivot/REQUIREMENTS.md`):
- SGV2-24: python→C++ encoder port
- SGV2-25: non-64-multiple tiling/padding policy

The roadmap-mandated plan-time re-evaluation (C++ port vs. direct consumption of exporter output) has been resolved: **the port goes ahead, v2-vcore scope only** (see D-01).

Includes: v2 adaptive quadtree encode core in C++ (FP4/T158 affine code modes, MIXED + uniform layouts the adaptive path emits), non-64-multiple zero-pad policy, real-shape generated golden fixtures, decode-parity tests against `fp4_exporter.py` output on both CPU and Vulkan decoders.

Excludes: v1 fixed-payload format, FULL_4X4 and other layouts the adaptive path never emits, Phase 11's PostConverter/graph-rewrite pass and CLI flag, Phase 10's encoder-parameter validation against real model weight statistics, any changes to `sgfp4_inject`, any changes to the decoders (existing decode math must consume the encoder's output unchanged).

</domain>

<decisions>
## Implementation Decisions

### Port vs. consume (roadmap-mandated re-evaluation, resolved)
- **D-01 (v2-core port):** Phase 9 IS a C++ encoder port — NOT direct export-dir consumption. Scope is the **v2 adaptive quadtree encode core only**: the `--adaptive` path's code modes (FP4_AFFINE / T158_AFFINE) and the layouts that path emits (LAYOUT_MIXED plus its uniform fallbacks). v1 fixed-payload, and layouts the adaptive path never emits (FULL_4X4, etc.) are NOT ported. Justification: self-contained single-command `mnnconvert --sgfp4` UX for the Phase 11 converter path with no Python dependency.
- **D-02 (Python-identical policy):** The adaptive quadtree decision policy must mirror `fp4_exporter.py --adaptive` exactly — same superblock scanning order, same split decisions, same `DEFAULT_V2_THRESHOLDS` MSE/relative-error values per leaf size (64→4: max_mse 0.01/0.005/0.002/0.001/0.0005, max_relative 0.05/0.03/0.02/0.01/0.005). Thresholds are not tunable knobs in this phase — Phase 10 revises them against real weight statistics if validation demands.
- **D-03 (header-pair lib under tools/fp4):** The encoder ships as `sgfp4_encode.hpp`/`sgfp4_encode.cpp` under `tools/fp4/` (building alongside `sgfp4_inject_core.hpp` under the existing `MNN_BUILD_SGFP4_TOOLS` CMake wiring), exercised via test infra in Phase 9. Phase 11's PostConverter pass links/includes it later — no converter-tree placement now.

### Parity bar
- **D-04 (decode-parity, not byte-exact):** Acceptance bar is **decode-parity**: C++-encode → decode must match Python-encode → decode within the existing cross-language tolerance (the rtol 1e-4 decode-vs-decode pattern from `SGFP4InjectTest`). Byte-exact container output is explicitly NOT required — near-tie threshold decisions may flip between encoders, bounded by decode tolerance. This avoids forcing bit-exact FP16/accumulation-order reproduction of NumPy.
- **D-05 (real-shape generated goldens):** Fixture strategy is a generator mirroring `tools/fp4/author_structured_fixture.py`: run `fp4_exporter.py --adaptive` on deterministic pseudo-random FP32 weights of non-64-aligned shapes (e.g. 100×36, 250×128, plus tiny <64 tensors), emitting `{input weights, container bytes, decoded reference}` C arrays into a committed, regenerable fixture header.

### Non-64-multiple tiling (SGV2-25)
- **D-06 (zero-pad to 64):** Non-64-multiple shapes are handled by **internal zero-padding to 64-multiples**: the encoder pads the weight plane with zeros, encodes the padded plane, and records the true `{dimO, dimI}` in the container/spec — no native partial-superblock traversal.
- **D-07 (row-major crop):** Crop semantics are **row-major**: the injected/rewritten op keeps `dims = {dimO, dimI}` (Phase 5 contract); the pad region is encoded but only the true-dims region is consumed — the first `dimO*dimI` elements row-major from the decoded padded plane. The researcher/planner must verify the existing decoders' `elementCount` handling actually supports this (decode-plane-larger-than-elementCount consistency).
- **D-08 (verify in Phase 9):** Padded non-aligned decode is verified **in Phase 9** against both real decoders — the CPU oracle (`dequant_sgfp4_container_cpu`) and the Vulkan Execution — as a correctness prerequisite, before Phase 10/11 build on it.

### Encoder placement & API
- **D-09 (tools/fp4, converter links later):** Build placement is `tools/fp4/` under the existing `MNN_BUILD_SGFP4_TOOLS` CMake structure (single lib home both `sgfp4_inject` and — in Phase 11 — the converter target depend on). No shared-lib target under `source/` and no header-only-only compromise; `.hpp` + `.cpp` pair compiles once into the tools lib.
- **D-10 (one-shot encode API):** Public API is a single encode function: raw FP32 weights + `{dimO, dimI}` in → container bytes (`std::vector<uint8_t>`) out. Layout/thresholds/mode are fixed at v2-adaptive defaults — no config knobs. If Phase 10's parameter-revision work requires them, the API grows a config struct THEN, not speculatively now.

### Claude's Discretion
- Exact function/file naming within the `sgfp4_*` conventions (`sgfp4_encode.hpp` suggested but not locked).
- Internal structure of the encoder (quadtree builder class vs. free functions; MSE accumulation details short of the D-04 parity bar).
- Which specific non-64-aligned shapes the golden generator covers beyond the D-05 examples (small/tiny/one-dim-aligned variety).
- Test suite naming/placement within the `op/sgfp4/` family conventions and the `tools/fp4/` test wiring.
- Whether tiny tensors (< 64 in a dim, single partial superblock) get dedicated hand-built edge cases in addition to the generated goldens (D-05 covers them statistically; explicit edge cases are at planner's judgment).
- Whether `encode_sgfp4.py`'s role comments need updating to note the C++ encoder as the new converter-path encoder while the Python script stays test-oracle — documentation-level detail.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Canonical Python encoder (the port source)
- `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\quantize\fp4_exporter.py` — the export-ledger source: `FP4Exporter` class, `DEFAULT_V2_THRESHOLDS`, v2 container layout (`magic[4] | version[1] | B[4] | pad0[7] | record_offsets[B] | pad1 | records`), `_zero_pad` 16-byte alignment, `--adaptive` quadtree path
- `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\quantize\sgfp4_format.py` — framing constants (ALIGNMENT, MACROBLOCK_SIZE, MIN_LEAF_SIZE, LEAF_MODE_MASK, SPLIT_MAP_WORDS, SGFP4_MAGIC, SGFP4_VERSION_V2, CodeMode, Layout enums) — the C++ port must mirror these (MNN side already has counterparts in `SGFP4DequantUtils.hpp`)

### Decode side (must consume encoder output UNCHANGED)
- `include/MNN/SGFP4DequantUtils.hpp` — MNN-side framing constants + `dequant_sgfp4_container_cpu` oracle; elementCount/dims handling that D-07's row-major crop relies on
- `source/backend/cpu/CPUSGFP4Dequant.cpp` — CPU Execution (buffer-first dispatch from Phase 8)
- `source/backend/vulkan/buffer/execution/` SGFP4 files — Vulkan Execution counterpart (D-08 verification target)

### Container consumption precedent (v2.0 injection tool)
- `tools/fp4/sgfp4_inject_core.hpp` — container framing, `sgfp4_is_v2_container` byte-level gate, 16-byte alignment emission convention; the structural pattern the encoder's output must satisfy
- `tools/fp4/README.md` — sidecar/alignment conventions, dims convention (`dimO = shape[0]`, `dimI = shape[1]`), "64-multiple dims only" limitation this phase lifts

### Existing test-oracle encoder & fixtures
- `tools/fp4/encode_sgfp4.py` — MNN's Python encoder, locked as **test-oracle-only** (not the canonical encoder — STATE.md 2026-08-26)
- `tools/fp4/author_structured_fixture.py` — the fixture-generator pattern D-05 mirrors (deterministic generation → committed C-array header)
- `test/op/SGFP4DequantFixtures.h`, `test/op/SGFP4StructuredFixtures.h` — existing committed fixture headers (naming/layout precedents)
- `test/op/SGFP4TestUtil.hpp` — shared test helpers (Phase 8 D-10 extraction) — new tests must be born on these
- `test/op/SGFP4InjectTest.cpp` — the rtol 1e-4 decode-vs-decode cross-language tolerance pattern (D-04's bar)

### Phase 8 hand-off contract (the consumer of this phase's output)
- `.planning/workstreams/sgfp4-pivot/phases/08-schema-sidecar-wiring/08-CONTEXT.md` — D-11 buffer-staging contract (`buffer = [container bytes]`, `external = {}`, no `externalPath`) this encoder's output flows into at Phase 11

### Workstream planning
- `.planning/workstreams/sgfp4-pivot/ROADMAP.md` §Phase 9 — goal line + plan-time re-evaluation note (resolved by D-01)
- `.planning/workstreams/sgfp4-pivot/REQUIREMENTS.md` — SGV2-24/25 requirement text
- `.planning/milestones/v2.0-MILESTONE-AUDIT.md` — tech-debt context (W-1/W-2/W-3) the roadmap assigns to Phase 11, not here

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/fp4/sgfp4_inject_core.hpp` — header-pair lib pattern in tools/fp4; SHA-256/framing helpers alongside (`sha256.hpp`)
- `tools/fp4/CMakeLists.txt` — `MNN_BUILD_SGFP4_TOOLS` wiring; new encode sources slot in here (D-03/D-09)
- `include/MNN/SGFP4DequantUtils.hpp` — framing constants already mirrored from gnus-poc (`kSGFP4Magic`, layout enums, leaf-header shifts/masks) — the encoder reuses these rather than redefining
- `dequant_sgfp4_container_cpu` — deterministic decode oracle for D-04 parity assertions (both as golden-reference decoder and as the CPU-side consumer in D-08)
- `author_structured_fixture.py` — regenerable golden-fixture pipeline to clone for D-05
- `test/op/SGFP4TestUtil.hpp` — container builders / sidecar / temp-path helpers for the new suites
- `3rd_party/half/` — FP16 bit-conversion precedent for C++-side scale/bias encoding (`fp16_bits` parity with the exporter)

### Established Patterns
- Decoders decode by exact `{offset, size}` with 16-byte-aligned, zero-padded regions — encoder output must pad every record and leaf payload to 16-byte multiples (exporter `_zero_pad` / `ALIGNMENT = 16`)
- Byte-level v2 gate: first 4 bytes `SGF4` magic + version byte `0x02` — encoder emits this exactly
- `dims = {dimO, dimI}` (out-rows, in-cols) on the op; exact-shape pairing downstream — encoder records true dims per D-06 (padding is encode-internal)
- Cross-language testing settles at decode-vs-decode rtol 1e-4 rather than byte-exactness (D-04 follows suit)
- `op/sgfp4/` test-suite registration + filtered runs via `run_test.out` (full run still blocked by unrelated dead `test/op/FP4ModelTest.cpp` — pre-existing, out of scope)

### Integration Points
- **Downstream Phase 11** invokes the encoder from its PostConverter pass via the D-03/D-09 lib link; output bytes stage into `SGFP4DequantParam.buffer` per Phase 8 D-11 (zero byte I/O in the pass)
- **Downstream Phase 10** validates/revises encoder parameters (D-02's thresholds) against real model weight distributions — may grow the API a config struct (D-10)
- **Existing decoders are integration-verification targets, not modification targets** (D-08): padded-plane encode must flow through CPU + Vulkan decode unchanged
- **`sgfp4_inject` unaffected** — it consumes pre-made exporter output; the encoder does not touch it

</code_context>

<specifics>
## Specific Ideas

No new specifics beyond the recorded decisions — all four gray areas resolved via the structured options as presented (Port v2 core only / decode-parity rtol + real-shape goldens / zero-pad to 64 + row-major crop + verify in Phase 9 / tools-fp4 lib + one-shot API).

</specifics>

<deferred>
## Deferred Ideas

- **Configurable encoder API (config struct with threshold table, layout/code-mode overrides):** deferred to Phase 10 — only if real-weight validation demands parameter revision (D-10).
- **Native partial-superblock quadtree traversal (no padding):** rejected for this phase (D-06); revisit only if pad-region overhead proves costly on real models (Phase 10's territory).
- **Same-shape disambiguation via tensor-name keying** and other injection-tool limitations: stay with Phase 11 / injection-tool work per the v2.0 milestone audit placement.
- **v1 fixed-payload and non-adaptive layouts port:** rejected outright (D-01) — v2-only milestone.

</deferred>

---

*Phase: 9-Real-Weight C++ Encoder Port*
*Context gathered: 2026-08-28*
