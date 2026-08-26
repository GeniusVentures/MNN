# Phase 5: Injection Core - Context

**Gathered:** 2026-08-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the standalone SGFP4 v2 injection tool: given a normally-converted `.mnn` (unmodified mnnconvert/llmexport output) plus one or more gnus-poc `fp4_exporter.py --adaptive` output directories, produce a new `.mnn` + external sidecar in which target weight tensors are produced by `OpType_SGFP4Dequant` nodes — correct at the Express/`Module::load` level (classic Interpreter/Session validation is Phase 6; multi-tensor/structured-data hardening is Phase 7).

Requirements SGINJ-01..04 (REQUIREMENTS.md) lock: v1-container rejection via version check, exact op/param construction with `op->externalPath` set literally on the op, single merged sidecar with non-overlapping `{offset, size}` ranges, serialization via `Variable::save(vars, fileName)` direct-to-file overload, and `Module::load` reload (with `rtmgr->setExternalFile()` before load) decoding through the existing CPU Execution within oracle tolerance.

</domain>

<decisions>
## Implementation Decisions

### Container→Tensor Pairing
- **D-01:** Pairing is manifest-driven. The tool consumes exporter output directories (each containing `manifest.json` + `<niche>.sgfp4`); it reads `fp4_binary.path` and `fp4_binary.stats.shape` from the manifest itself — no container path or tensor name duplicated on the CLI.
- **D-02:** Target-tensor selection by exact shape match: find `.mnn` weight tensors whose shape exactly equals `{dimO=shape[0], dimI=shape[1]}`. Exactly one match → inject. Zero or multiple matches → hard error listing candidate tensor names/shapes.
- **D-03:** Integrity check at inject time: compute sha256 of the container file and compare against manifest `fp4_binary.sha256`; mismatch → hard error. (Catches wrong-niche/corrupt containers before they become garbage weights.)
- **D-04:** Exact match only — non-64-multiple / padded shapes are rejected. The tiling/padding convention gap is a known v3.0 Phase 10 item; do not invent padding rules in this tool.
- **D-05:** `SGFP4DequantParam.dims` comes from the manifest's `fp4_binary.stats.shape` (same source of truth as pairing); the tool cross-checks it against the matched `.mnn` tensor shape and errors on disagreement.

### Graph Surgery Mechanism
- **D-06:** Express VARP-level surgery: load the `.mnn` as a VARP graph, construct the `SGFP4Dequant` op from a hand-built `OpT` (as `test/op/SGFP4DequantTest.cpp` and `test/op/SGFP4VulkanDequantTest.cpp` already demonstrate), then rewire each consumer from the original constant weight to the dequant node via `Variable::replaceInput` (quantization-tool precedent). No manual FlatBuffers `NetT` oplists index bookkeeping.
- **D-07:** The original constant weight tensor is detached and dead-dropped: leave it in the loaded VARP graph untouched; after consumer rewiring it becomes dead code and `Variable::save` drops unreachable constants naturally. No forced removal code.
- **D-08:** Injected node keeps the original weight tensor's name with an `_sgfp4` suffix (e.g. `weight` → `weight_sgfp4`) for graph debugging and Phase 6 baseline correlation.

### Tool Form & CLI
- **D-09:** The tool is a new C++ binary under `tools/fp4/` (e.g. `tools/fp4/sgfp4_inject.cpp`) with its own CMakeLists gated behind a CMake option, linked against core MNN Express. Manifest JSON parsing uses the vendored `3rd_party/rapidjson`. (Tool-family precedent: `tools/fp4/encode_sgfp4.py` already lives there.)
- **D-10:** CLI surface: `sgfp4_inject --model input.mnn --niche-dir <dir> [--niche-dir <dir>...] --output out.mnn`. Each `--niche-dir` is an unmodified `fp4_exporter.py` output directory (manifest.json + .sgfp4). Phase 7 multi-tensor injection = multiple `--niche-dir` args.
- **D-11:** The sidecar is emitted alongside the output model as `<output>.weight` (e.g. `out.mnn` → `out.mnn.weight`), mirroring the `Convolution2D.external` convention already locked in STATE.md.

### Verification Depth
- **D-12:** In-tool post-serialization verification is unconditional: after `Variable::save`, the tool reloads the artifact via Express `Module::load` (calling `rtmgr->setExternalFile(sidecar)` before load), runs each `SGFP4Dequant` node, and compares outputs against a direct CPU decode of the same container bytes (via `dequant_sgfp4_container_cpu`). Mismatch → nonzero exit with diagnostic. This satisfies SGINJ-04 inside the tool — every produced artifact is proven good at creation time.
- **D-13:** The in-tool numeric check is a decode-oracle comparison (reloaded-artifact decode vs. fresh decode of the same container — deterministically identical). FP32-tolerance comparison against the original model stays in the Phase 5 test suite and Phase 6 end-to-end, where the FP32 baseline must be loaded anyway.

### Claude's Discretion
- Internal structure of the binary (single TU vs. helper headers under `tools/fp4/`), CMake option naming, exact error-message wording, logging verbosity.
- Weight-tensor enumeration order and candidate-listing format in ambiguity errors.
- Whether the sha256 implementation uses a small vendored/public-domain header or platform API (no OpenSSL dependency introduced).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Format / decode ground truth (in-repo)
- `include/MNN/SGFP4DequantUtils.hpp` — container framing constants, decode helpers; byte-verified against gnus-poc exporter
- `tools/fp4/encode_sgfp4.py` — test-oracle-only encoder; documents container format constants and packing conventions (spec §4.3/6.1/6.2)
- `source/backend/cpu/CPUSGFP4Dequant.cpp` / `.hpp` — the ground-truth consumer Execution this artifact must satisfy
- `source/shape/ShapeSGFP4Dequant.cpp` — output shape comes from `SGFP4DequantParam::dims` (manifest-resident)

### Op construction precedents (in-repo)
- `test/op/SGFP4DequantTest.cpp` — existing op-level construction: builds `OpT` with `OpType_SGFP4Dequant` + `SGFP4DequantParamT{magic, external{offset,size}, dims}`, writes a sidecar, sets `op->externalPath` literally on the op, and round-trips through `Module::load`
- `test/op/SGFP4VulkanDequantTest.cpp` — same pattern; documents the externalPath gotcha in comments
- `test/op/SGFP4DequantFixtures.h` — generated fixtures + expected decode outputs (oracle data for tests)

### Serialization / surgery precedents (in-repo)
- `include/MNN/expr/Expr.hpp:157` — `Variable::save` direct-to-file overload (mandated by SGINJ-04)
- `tools/quantization/calibration.cpp` — `Variable::save(predicts, path)` + VARP graph manipulation precedent
- `tools/converter/source/optimizer/Program.cpp` / `Pass.cpp` — converter-side Express serialization patterns

### Exporter artifacts + manifest (external)
- `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\quantize\fp4_exporter.py` — canonical real-weight encoder (`--adaptive`); writes `<niche>.sgfp4`, `<niche>_stats.json`, `manifest.json` per niche dir
- `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\quantize\manifest.py` — `ManifestBuilder` (manifest schema)
- `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\quantize\sgfp4_format.py` — container format constants (exporter side)
- `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\models\specialists_mlx\demo\fp4\manifest.json` — live example manifest: `fp4_binary.path`, `fp4_binary.sha256`, `fp4_binary.format`, `fp4_binary.stats.shape=[512,512]` (+ macroblock/layout stats)
- `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\models\specialists_mlx\demo\fp4\demo.sgfp4` — starter container (132,368 bytes, 512×512, all `UNIFORM_64`)

### Workstream planning
- `.planning/workstreams/sgfp4-pivot/REQUIREMENTS.md` — SGINJ-01..04 for this phase
- `.planning/workstreams/sgfp4-pivot/ROADMAP.md` §"Phase 5" — success criteria 1–4
- `.planning/workstreams/sgfp4-pivot/STATE.md` — locked decisions (sidecar convention, externalPath gotcha, terminology)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `test/op/SGFP4DequantTest.cpp` op-construction block — near-copy ready for the injector's OpT building (magic/external/dims/externalPath)
- `dequant_sgfp4_container_cpu` (SGFP4DequantUtils.hpp) — deterministic decode oracle for D-12/D-13 verification
- `3rd_party/rapidjson` — vendored JSON parser for `manifest.json` reading
- `Variable::replaceInput` (Express) — consumer rewiring primitive used by quantization tools

### Established Patterns
- Sidecar + op-resident `externalPath` with `{offset, size}` on the sidecar file per op (Convolution2D.external analog) — the two CPU/Vulkan test files and the existing Executions already implement the load side
- `rtmgr->setExternalFile()` must be called before `Module::load` for buffer-based loads (documented Phase 1 pitfall — buffer loads don't auto-set externalPath)
- `fp4_exporter.py` niche-dir layout (`manifest.json`, `<niche>.sgfp4`, `<niche>_stats.json`) is the tool's input contract (D-01/D-10)

### Integration Points
- Output artifact consumed downstream by: Phase 6 classic `Interpreter::createFromFile → createSession → runSession` (SGProcessingManager path) — tool must not depend on Express-only serialization side effects
- Output artifact decoded by existing `CPUSGFP4Dequant` Execution + `ShapeSGFP4Dequant` shape computer — no runtime code changes needed in this phase
- v3.0 Converter Integration (Phases 8–12) expects to inherit this tool's sidecar/rewiring conventions

</code_context>

<specifics>
## Specific Ideas

- User explicitly directed matching the tool's input contract to GNUS-NEO-SWARM's exporter output ("it generates an sgfp4 and json file... We need to match up with that") — hence manifest-driven pairing (D-01) rather than CLI pair lists.
- No other specific references — standard approaches accepted elsewhere.

</specifics>

<deferred>
## Deferred Ideas

- Non-64-multiple weight shapes / tiling-padding conventions — belongs to v3.0 Phase 10 (real-weight validation), noted in STATE.md pending todos.
- Transposed shape matching (dimO×dimI vs dimI×dimO tolerance) — rejected for now (risk of silently pairing wrong semantics); revisit only if a real model needs it.
- `--no-verify` style skip flag for bulk injection runs — only add when a bulk use case actually appears (Phase 7 or later).
- Structured (non-uniform / LAYOUT_MIXED) container coverage — Phase 7 (SGINJ-08), requires obtaining or generating a second, structured container from gnus-poc.

</deferred>

---

*Phase: 5-Injection Core — Artifact Construction & Graph Splicing*
*Context gathered: 2026-08-26*
