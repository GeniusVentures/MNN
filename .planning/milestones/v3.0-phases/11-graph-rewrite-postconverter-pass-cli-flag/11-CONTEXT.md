# Phase 11: Graph-Rewrite PostConverter Pass + CLI Flag - Context

**Gathered:** 2026-09-01
**Status:** Ready for planning

<domain>
## Phase Boundary

A new `PostConverter` pass inserts `OpType_SGFP4Dequant` nodes ahead of target conv-family weight tensors, driven by a new boolean mnnconvert CLI flag (`--sgfp4`); `WeightQuantAndCoding` gains a topology-based skip-guard so already-rewritten convs are never double-processed. Container bytes come from the Phase 9 C++ encoder (`sgfp4_encode::encode`) staged per Phase 8's D-11 buffer contract (`buffer` populated, `external = {}`, no `externalPath`); externalization rides the existing `saveExternalData`/`_largeModel` machinery untouched. The phase also retires the v2.0 milestone-audit tech debt assigned to it: W-1 (classic_api region-relative offset retrofit) and W-2 (arg-stage failCleanup hoist), plus the small adjacent W-3 env-var portability fix.

Scope anchor (requirements SGV2-28/29/30, from `.planning/milestones/v2.0-REQUIREMENTS.md` §"v3.0 Converter Integration"):
- SGV2-28: the graph-rewrite PostConverter pass
- SGV2-29: the CLI flag trigger
- SGV2-30: the WeightQuantAndCoding skip-guard (double-processing prevention)

Includes: the pass itself (oplists + control-flow subgraphs), flag parsing + hard mutex against conflicting weight-transform flags, the skip-guard, threshold sourcing (named Python-identical config constant), pass-mechanics tests extending `TestSGFP4Converter`, a real mnnconvert CLI smoke run on the approved AlexNet corpus, flag-OFF no-regression gate, and the W-1/W-2/W-3 debt retirement.

Excludes: Phase 12's end-to-end CPU/Vulkan inference validation and output-accuracy gates; encoder/decoder changes (Phases 8-10 outputs are consumed as-shipped); MatMul/`OpParameter_MatMul` weight rewriting (only conv-family; MatMul-derived convs ride along only where `TransformInnerProduct` already converted them); threshold-file CLI overrides (Python-identical defaults only); gnus-poc-side changes of any kind; per-layer opt-out mechanisms (SGV2-37 territory); the injection tool's UX beyond the W-2 cleanup fix.

</domain>

<decisions>
## Implementation Decisions

### Pass placement & wiring
- **D-01 (registered PostConverter pass):** The rewrite ships as a named, registered `PostConverter` pass (e.g. `"InsertSGFP4Dequant"` via `PostConverterRegister`), appended to the final `RunNetPass` batch in `PostConverter.cpp` — after the Merge* passes have finalized conv weights, in the company of `ReIndexTensor`. Rationale: standard mechanism, free `--dumpPass` observability, directly drivable from `TestSGFP4Converter` via `RunNetPass`, and net-level topology access (which the per-op `WeightQuantAndCoding` hook cannot provide). Rejected: a dedicated sweep inside `postTreat()` (writeFb.cpp) — workable but non-discoverable and outside the pass tooling.
- **D-02 (topology-based skip-guard):** `WeightQuantAndCoding` skips any conv op whose `inputIndexes.size() > 1` — an original converter conv carries only its input-activation index (weights live in the op param), so a second input is the visible fingerprint of an SGFP4-rewritten conv. Pure topology check: no schema change, no config coupling, no marker field. (Existing `quanParameter != nullptr` early-return stays as-is.)
- **D-03 (full net coverage, subgraphs included):** The pass walks `netT->oplists` AND every `subgraph->nodes` iteratively — mirroring how `RemoveUnusefulOp`/`saveExternalData` already iterate — so control-flow weights inside If/While subgraphs are rewritten too. Rejected: main-oplist-only with deferred subgraph coverage.

### CLI flag & op scope
- **D-04 (boolean `--sgfp4`):** One boolean flag `--sgfp4` in `cli.cpp` mapping to a new `modelConfig::useSGFP4` field — the exact `--hqq`/`--fp16` precedent shape. No value arguments, no threshold path.
- **D-05 (hard mutex on conflicting flags):** `--sgfp4` combined with `--weightQuantBits`, `--hqq`, or `--fp16` is a hard parse-time error (clear `MNN_ERROR`, non-zero exit). Conflicting weight transforms on the same tensors are nonsensical; fail fast rather than silently pick a winner. (This is CLI-level UX; D-02's topology guard remains the structural double-processing defense.)
- **D-06 (conv-family op scope):** The pass rewrites exactly the 4 types `WeightQuantAndCoding` handles — `Convolution`, `ConvolutionDepthwise`, `Deconvolution`, `DeconvolutionDepthwise` — flattening weights `{oc, ic*kx*ky}` to the 2-D `[out, in]` plane per the locked dims convention. MatMul-derived weights are covered only where `TransformInnerProduct` already produced a `Convolution` op before the pass runs (a placement side-effect, not a dedicated MatMul case — that is out of scope).
- **D-07 (Phase 10 D-03 light-tier floor adopted):** The pass leaves weights FP32 when `elements < 4096` OR `dimI == 1` — the validated corpus tiering rule; tiny tensors are pad-overhead-dominated and were light-tier in the Phase 10 report. Everything else encodes.

### Threshold sourcing
- **D-08 (named Python-identical constant, not the validated delta):** The pass threads an explicit, greppably-named converter-side config constant equal to `sgfp4_encode::kDefaultEncodeConfig` — satisfying Phase 10's "accept EncodeConfig explicitly rather than defaults" carry-forward via a named swap-point rather than a silent knob-less call. It is deliberately NOT the Phase 10 validated delta: cross-repo default parity outranks one-sided promotion (gnus-poc still defaults to `DEFAULT_V2_THRESHOLDS`; Phase 10's D-09 rationale). When the delta is adopted upstream, the constant is the single edit point. A comment at the constant documents where the validated table lives (`tools/fp4/real_weight_validation_report.json`). No `--sgfp4Thresholds`-style CLI override.

### Tech-debt retirement (v2.0 milestone audit)
- **D-09 (W-1 retrofit):** `SGFP4ClassicAPITest.cpp:167-171` is retrofitted to write REGION-RELATIVE offset-table entries (encoder-conformant convention; conformant builder reference `SGFP4MultiTensorTest.cpp:190-199`, now in `SGFP4TestUtil.hpp`). The audit finding closes; no annotate-only compromise.
- **D-10 (W-2 fix):** The `failCleanup` lambda in `sgfp4_inject_core.hpp` is hoisted above the two arg-validation returns (`:278-284`) so usage()-exit paths also remove stale output artifacts — making behavior match the README's full-removal promise.
- **D-11 (W-3 included):** The hard-coded gnus-poc root (`author_structured_fixture.py:25`, and the same hardcode pattern in the sibling scripts where present) gains an env-var override (e.g. `SGFP4_GNUS_POC_ROOT`, falling back to the current path). Small portability fix; retires the last actionable audit item alongside this phase's tool work.

### Test strategy
- **D-12 (extend TestSGFP4Converter):** Pass-mechanics tests extend the Phase 8 standalone `TestSGFP4Converter.cpp` executable (already linking `MNNConvertDeps` + `MNN_DEPS`): a synthetic `NetT` with conv ops → pass ON → assert `SGFP4Dequant` nodes inserted, consumers rewired to the new node's output (conv `inputs[1]`), FP32 weights cleared from the conv param, container bytes in `buffer`, `external = {}`/no `externalPath` (D-11 Phase 8 contract), light-tier skip honored, subgraph coverage exercised. No new CMake surface.
- **D-13 (real CLI smoke on the corpus):** The flag→artifact path is proven by a real `mnnconvert --sgfp4` invocation on the approved corpus (`W:\gnus\models\alexnet_Opset16.onnx`) — output `.mnn` asserted to contain `SGFP4Dequant` nodes and decode via the classic API. Corpus file is a test-time dependency (documented scripted/manual step, not a committed always-on gate — corpus provenance per Phase 10 D-01/D-02).
- **D-14 (flag-OFF no-regression gate):** Flag OFF → zero behavior change: the pass is dead code when `--sgfp4` is absent; all 13 `op/sgfp4` suites + existing converter tests green with no test-file edits. Flag-ON exotic-op behavior beyond the corpus is NOT gated this phase (Phase 12's E2E territory).

### Claude's Discretion
- Exact pass registration string (suggested `"InsertSGFP4Dequant"`) and file name within `tools/converter/source/optimizer/postconvert/` conventions.
- Exact pass ordering within the final `RunNetPass` batch (before/after `ReIndexTensor`) — planner verifies tensor-index bookkeeping (`tensorName` growth, new output index allocation, `ReIndexTensor` interplay) and locks it.
- The named constant's exact name/placement (converter-side vs. reusing `kDefaultEncodeConfig` directly with a comment) as long as it stays greppable and Python-identical.
- Whether D-05's mutex error message enumerates each conflicting flag individually or collectively.
- Structure of the synthetic nets in D-12 (how many convs, subgraph shape, light-tier inclusion) beyond the listed assertions.
- How D-13's smoke run is scripted/documented (README section vs. test script) and its tolerance wording for decode-vs-FP32 comparison (Phase 12 owns formal accuracy gates).
- Whether the D-11 env-var override name aligns with any existing gnus-poc-side convention discovered at plan time.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Workstream planning (phase contracts this phase builds on)
- `.planning/workstreams/sgfp4-pivot/ROADMAP.md` §Phase 11 — goal, W-1/W-2 mandate, SGV2-28/29/30 mapping
- `.planning/milestones/v2.0-REQUIREMENTS.md` §"v3.0 Converter Integration" — SGV2-28/29/30 requirement text; §"Performance & Coverage" SGV2-37 (explicitly out of scope here)
- `.planning/milestones/v2.0-MILESTONE-AUDIT.md` §tech_debt — W-1 (`SGFP4ClassicAPITest.cpp:167-171` absolute offsets), W-2 (`sgfp4_inject_core.hpp:278-284` failCleanup gap), W-3 (`author_structured_fixture.py:25` hard-coded root)
- `.planning/workstreams/sgfp4-pivot/phases/08-schema-sidecar-wiring/08-CONTEXT.md` — D-11 buffer-staging contract (pass writes `buffer`, `external = {}`, no `externalPath`), D-05/D-06 aligned sidecar store, D-12 createExecutionWithExternal non-interception
- `.planning/workstreams/sgfp4-pivot/phases/10-real-weight-validation-against-actual-model-statistics/10-CONTEXT.md` — D-03 light-tier rule; D-08/D-09 threshold-delta rationale
- `.planning/workstreams/sgfp4-pivot/phases/10-real-weight-validation-against-actual-model-statistics/10-VERIFICATION.md` §Notes — the "pass EncodeConfig explicitly" carry-forward D-08 here resolves

### Converter pipeline (where the pass hooks in — to be modified)
- `tools/converter/source/optimizer/PostTreatUtils.hpp` — `PostConverter` base + `PostConverterRegister<T>` macro
- `tools/converter/source/optimizer/PostConverter.cpp` — `RunNetPass` mechanics, the final pass batch (`ReIndexTensor`/`ReIndexOnnxIfAlias` at ~:393), pass-chain sequencing the new pass joins
- `tools/converter/source/common/writeFb.cpp` — `_postTreatOp` call order (`WeightQuantAndCoding` → `RemoveAndStoreParam`), `postTreat` external-weight flag machinery (unchanged this phase)
- `tools/converter/source/common/WeightQuantAndCoding.cpp` — the per-op hook gaining D-02's `inputIndexes.size() > 1` skip; its op-type gate defines D-06's scope
- `tools/converter/source/common/cli.cpp` — `--hqq`/`--weightQuantBits`/`--fp16` parse precedents (~:460-520), option-table entries (~:180-290)
- `tools/converter/include/config.hpp` — `modelConfig` fields (`useHQQ`, `weightQuantBits`, `saveHalfFloat`) — `useSGFP4` slots in here
- `tools/converter/source/optimizer/postconvert/GenerateSubGraph.cpp:581-589` + `saveExternalData` — net + subgraph iteration precedent for D-03
- `tools/converter/source/optimizer/postconvert/SplitBlockQuantConvolution.cpp:44-52` — precedent for a post-convert pass writing `externalPath`/`external` (the pattern Phase 8 D-11 deliberately does NOT use — buffer staging instead)

### Encoder + externalization (consumed as-shipped)
- `tools/fp4/sgfp4_encode.hpp` — `encode(weights, dimO, dimI[, EncodeConfig])` API, `kDefaultEncodeConfig`, `EncodeConfig` struct
- `tools/fp4/sgfp4_encode.cpp` — `kDefaultEncodeConfig` definition + the validated-delta table comment D-08's constant points at
- `tools/converter/source/common/RemoveParams.cpp` — `storeSGFP4Container` (16-byte-aligned sidecar store, Phase 8), `RemoveAndStoreParam` SGFP4 case
- `tools/fp4/real_weight_validation_report.json` — the validated (unused-by-default) threshold table D-08 documents

### Injection-tool rewiring precedent + W-2 site
- `tools/fp4/sgfp4_inject_core.hpp` — consumer-rewiring pattern (`SGFP4Dequant` node output → consumer `inputs[1]`, op type unchanged, bias stays `[float]`), node naming `weight → weight_sgfp4`, in-tool decode verification; `:278-284` + `:304` are the W-2 fix sites
- `tools/fp4/README.md` — sidecar/alignment conventions, dims convention, failure-behavior promise W-2 must honor

### Test assets
- `tools/converter/source/TestSGFP4Converter.cpp` — D-12's extension target; synthetic-NetT construction + assertion scaffolding from Phase 8 (08-06)
- `tools/converter/CMakeLists.txt:55-100` — `TestSGFP4Converter` link chain (static + shared branches) — already wired, no new surface
- `test/op/SGFP4TestUtil.hpp` — shared helpers incl. the region-relative offset builder (W-1 reference fix)
- `test/op/SGFP4ClassicAPITest.cpp:167-171` — W-1 retrofit site
- `test/op/SGFP4MultiTensorTest.cpp:190-199` — the encoder-conformant builder W-1 retrofits onto
- `tools/fp4/author_structured_fixture.py:25` — W-3 env-var site (check `author_real_shape_fixture.py`/`validate_real_weights.py` for the same hardcode)

### Runtime consumers (read-only ground truth for what the artifact must satisfy)
- `source/backend/cpu/CPUSGFP4Dequant.cpp` — buffer-first dispatch the pass's output feeds
- `source/core/OpCommonUtils.cpp:665` — `createExecutionWithExternal` switch SGFP4 intentionally stays out of (Phase 8 D-12 comment)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `PostConverterRegister<T>` + `RunNetPass` — the registration + invocation machinery D-01 plugs into; `--dumpPass` size-diff logging comes free
- `TestSGFP4Converter.cpp` — synthetic `NetT` builder, sidecar/layout assertion macros, and the `MNNConvertDeps` link chain; D-12 extends rather than duplicates it
- `sgfp4_inject_core.hpp`'s rewiring block — the proven node-insertion + consumer-splice + naming pattern, transplantable to a converter pass operating on `OpT`/`netT` instead of loaded vars
- `sgfp4_encode::encode` + `EncodeConfig` — the encoding primitive; already MSVC-proven (`EncodeConfig` aggregate-init brace-elision pitfall documented in Phase 10 learnings)
- `SGFP4TestUtil.hpp` region-relative offset builder — W-1's retrofit target reference

### Established Patterns
- Per-op hooks (`WeightQuantAndCoding`) vs net-level passes (`RunNetPass`) — the structural split that forces D-01's placement; per-op hooks see no topology
- `storeWeight` post-hoc externalization (`_postTreatOp`: quant → store) — the pass must finish before `RemoveAndStoreParam` encounters SGFP4 ops, which the final-pass placement + existing call order already guarantee
- Flag→config→hook threading (`cli.cpp` → `modelConfig` → converter code via `Global<modelConfig>::Get()`) — D-04/D-05 follow it exactly
- Buffer-first decoder dispatch (Phase 8) — the artifact contract: whatever placement ships (`buffer` inline or sidecar-externalized), runtime decodes it

### Integration Points
- New pass file under `tools/converter/source/optimizer/postconvert/` + one line in `PostConverter.cpp`'s final pass batch
- `WeightQuantAndCoding.cpp` early-return guard extension (D-02)
- `cli.cpp` option table + parse block, `config.hpp` field (D-04/D-05)
- Converter build must link/expose `sgfp4_encode` (currently a `tools/fp4` static lib under `MNN_BUILD_SGFP4_TOOLS`) — planner resolves the exact CMake wiring (likely: converter target gains the encode lib as a dependency when SGFP4 support is compiled in)

</code_context>

<specifics>
## Specific Ideas

- The pass's node-naming and consumer-rewiring should mirror the injection tool's conventions (`weight → weight_sgfp4`, splice at `inputs[1]`, conv op type unchanged, bias untouched) so the two artifact producers stay structurally comparable.
- The flag's help text should name the format correctly: "SGFP4 v2" — never "Ultra FP4" (locked terminology).
- D-13's smoke proof should reuse the Phase 10 approved corpus (AlexNet ONNX, sha256 `4bc388cc…`) for continuity of provenance — no new corpus approval needed.

</specifics>

<deferred>
## Deferred Ideas

- MatMul/`OpParameter_MatMul` weight rewriting (LLM-export path) — potentially its own phase after Phase 12; rides along this phase only via `TransformInnerProduct` side-effect (D-06)
- `--sgfp4Thresholds` CLI file override — rejected this phase (D-08); revisit only if a real consumer need emerges after upstream delta adoption
- Per-layer SGFP4 opt-out (`PostTreatContext::quantInfo`-style, SGV2-37) — future requirement, explicitly out of scope
- Flag-ON converter corpus sweep beyond AlexNet (exotic-op robustness) — Phase 12 E2E territory (D-14)
- gnus-poc upstream adoption of the validated threshold delta — sibling-repo proposal flow (Phase 10 D-09), not blocked by this phase

</deferred>
