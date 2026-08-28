# Phase 5: Injection Core - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-26
**Phase:** 5-Injection Core — Artifact Construction & Graph Splicing
**Areas discussed:** Container→tensor pairing, Graph surgery mechanism, Tool form & CLI, Verification depth

---

## Container→Tensor Pairing

| Option | Description | Selected |
|--------|-------------|----------|
| CLI pair list | Explicit `--inject container.sgfp4:tensor.name` pairs; fully explicit, natural Phase 7 extension | |
| Dims auto-match | Tool auto-matches containers to weight tensors by dims equality; zero config but needs tie-breakers | |
| Single-pair only | Single container flag for Phase 5; pairing logic deferred to Phase 7 | |
| Manifest + shape | Tool consumes (container, manifest.json) dirs; shape match from manifest; sha256 integrity | ✓ |

**User's choice:** Free-text redirect → match the tool's contract to GNUS-NEO-SWARM exporter output ("I would reference W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM for what information it is going to avail you. I think it generates an sgfp4 and json file, I do not fully know what they contain. We need to match up with that.")
**Notes:** Investigated the exporter: `fp4_exporter.py --adaptive` writes per-niche dirs with `<niche>.sgfp4` + `manifest.json` (carrying `fp4_binary.path`, `.sha256`, `.format`, `.stats.shape=[512,512]`, macroblock/stats info) + `<niche>_stats.json`. Presented manifest-informed options; user chose **Manifest + shape** (exact-shape matching, unique-match-or-error, sha256 check, manifest as dims source).

### Follow-up questions (pairing)
- Integrity: sha256 check ✓ (vs header-only, vs full decode+stats verify)
- Shape strictness: Exact match only ✓ — non-64-multiple shapes rejected; padding convention is a v3.0 Phase 10 item (vs exact+diagnostic, vs allow-transpose)

## Graph Surgery Mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Express VARP | Load as VARP graph, build dequant op from OpT, `Variable::replaceInput` rewiring; quantization-tool precedent; satisfies SGINJ-04 naturally | ✓ |
| FlatBuffers NetT | Mutate NetT oplists + input indices manually, then rebuild Express graph for save; more control, error-prone index bookkeeping | |
| Research both | Researcher prototypes both paths empirically; slower but de-risks Express friction | |

**User's choice:** Express VARP
**Notes:** None

### Follow-up questions (surgery)
- Original constant fate: Detach + dead-drop ✓ (Variable::save drops unreachable constants; FP32 baseline stays available via separate load) (vs force removal, vs verify-in-tests)
- dims source: Manifest dims ✓ (`fp4_binary.stats.shape`, cross-checked against matched tensor) (vs tensor dims, vs container header)

## Tool Form & CLI

| Option | Description | Selected |
|--------|-------------|----------|
| C++ binary tools/fp4 | New sgfp4_inject.cpp + own CMakeLists behind an option; links core Express + rapidjson; mirrors sgfp4 tool-family precedent | ✓ |
| Python driver | Python driver calling a C++ helper via subprocess; exporter-workflow-familiar but packaging friction | |
| MNNConvert flag | Fold into existing MNNConvert CLI; couples Phase 5 to converter build, blurs milestone boundary | |

**User's choice:** C++ binary tools/fp4
**Notes:** None

### Follow-up questions (CLI)
- CLI surface: Niche-dir args ✓ (`--model X --niche-dir D [--niche-dir D2...] --output out.mnn`; sidecar `out.mnn.weight`) (vs file-pair args, vs config file)
- Node naming: Name + suffix ✓ (`weight` → `weight_sgfp4`) (vs auto names, vs niche-prefix flag)

## Verification Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Reload in-tool | Unconditional post-save `Module::load` reload + per-node decode check; artifact proven at creation time | ✓ |
| Tests only | Tool validates inputs structurally; Module::load oracle check lives in the test suite | |
| Flag-gated | Same reload check behind `--verify` (default on); one more flag to carry | |

**User's choice:** Reload in-tool
**Notes:** None

### Follow-up questions (verify)
- Comparison baseline: Decode oracle ✓ — reloaded decode vs. fresh `dequant_sgfp4_container_cpu` of same bytes; FP32 tolerance stays in tests/Phase 6 (vs FP32 tolerance in-tool, vs structural-only)

---

## Claude's Discretion

- Binary internal structure (TU layout), CMake option naming, error wording, logging verbosity
- Weight-tensor enumeration order / candidate-list formatting in ambiguity errors
- sha256 implementation choice (vendored small header vs platform API; no OpenSSL)

## Deferred Ideas

- Non-64-multiple shapes / tiling-padding conventions → v3.0 Phase 10
- Transposed shape matching tolerance → revisit only if a real model needs it
- `--no-verify` bulk-run skip flag → only when a bulk use case appears (Phase 7+)
- Structured (LAYOUT_MIXED) container coverage → Phase 7 (SGINJ-08), needs a structured gnus-poc artifact
