# Phase 8: Schema + Sidecar Wiring - Context

**Gathered:** 2026-08-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Add `buffer:[byte]` to `SGFP4DequantParam` (schema evolution) and wire `RemoveParams.cpp` so SGFP4 container bytes carried in the op param can be externalized through the converter's existing shared `.mnn.weight` sidecar mechanism (`saveExternalData` / `--saveExternalData` / `_largeModel` auto-trigger). Make the runtime decoders accept **both** data placements — inline buffer and external sidecar — with one unified convention. This is the serialization foundation Phases 9–11 (encoder port → real-weight validation → graph-rewrite PostConverter pass) build on.

Scope anchor (requirements SGV2-22/23, from `.planning/milestones/v2.0-REQUIREMENTS.md` §"v3.0 Converter Integration"):
- SGV2-22: `SGFP4DequantParam.buffer:[byte]` schema field
- SGV2-23: `RemoveParams.cpp` externalization of that buffer through the shared sidecar mechanism

Includes (per decisions below): CPU + Vulkan Execution buffer-first dispatch, 16-byte-aligned sidecar writes in `RemoveAndStoreParam`, decode-parity + converter-round-trip tests, and `SGFP4TestUtil.hpp` extraction with retrofit of the three existing SGFP4 test files.

Excludes: the PostConverter graph-rewrite pass itself (Phase 11), any encoder work (Phase 9/10), CLI flags (Phase 11), sgfp4_inject changes.

</domain>

<decisions>
## Implementation Decisions

### buffer field semantics
- **D-01 (buffer-first decoder fallback):** `SGFP4DequantParam.buffer:[byte]` is a **live serialized decode source**, not converter-transient staging. If `buffer` is non-empty, the Execution decodes the container directly from it; if empty, the existing external-sidecar path (`external = {offset, size}` + `op->externalPath`) runs. This mirrors `Blob`'s inline-data pattern (`float32s`/`uint8s` serialized inline in the op param) and unlocks single-file `.mnn` artifacts.
- **D-02:** The hard `NOT_SUPPORT` gate in `CPUSGFP4Dequant::onResize` (currently `USE_EXTERNAL_DATA(param) && mOp->externalPath()` or bail, lines ~48–53) is replaced with buffer-first dispatch: `param->buffer()` non-empty → skip FileLoader entirely; empty → current sidecar path with all existing validation (offset/size sanity, T-01-04 file-size bounds check, loader validity). Buffer-mode must retain equivalent input validation (magic check at decode entry, size-vs-dims consistency) since there is no file to bound against.
- **D-03:** The Vulkan Execution gets the same buffer-first dispatch (its onResize currently mirrors the CPU sidecar gate). Both backends ship in this phase — no split commitment.
- **D-04:** Sidecar-mode remains the original supported path; buffer-mode is additive. Existing tests and artifacts (injection-tool output) must remain green unchanged.

### sidecar write convention
- **D-05 (16-byte-aligned storeWeight):** `RemoveAndStoreParam` gains an `OpParameter_SGFP4DequantParam` case that appends the container bytes to the sidecar via the `storeWeight<uint8_t>` pattern, but pads the region to a **16-byte multiple** (zero-filled pad bytes) before advancing the shared `offset` — matching `sgfp4_inject`'s emission convention exactly. One ecosystem-wide alignment rule; the decoder reads by exact `{offset, size}` so padding is inert to it.
- **D-06:** After the sidecar write, `param->buffer` is cleared (standard `storeWeight` behavior) — no keep-buffer duplication (rejected: doubles artifact size, creates silent dual-source ambiguity). `external` is set to the region's `{offset, size}` (inclusive of pad? no — `size` is the **true container size**; the pad lives only in the offset advance).
- **D-07:** Externalization remains conditional on the existing converter flags (`config.saveExternalData` / `_largeModel` auto-trigger in `writeFb.cpp:108-118`); when OFF, the buffer simply serializes inline in the `.mnn`. No new converter flag in this phase.

### Phase 8 test scope (T3 — full)
- **D-08 (decode parity):** New tests asserting buffer-mode decode == sidecar-mode decode == existing oracle (`SGFP4DequantFixtures` / `dequant_sgfp4_container_cpu`), on both CPU and Vulkan, using identical container bytes across the two placements.
- **D-09 (converter round-trip):** A converter-path test driving `RemoveAndStoreParam`/`saveExternalData` on a synthetic `NetT` containing an SGFP4 op with a populated buffer; asserts the emitted sidecar layout is 16-byte aligned, monotonic, non-overlapping (mirroring the v2.0 audit's sidecar assertions), `external == {offset, true-size}`, buffer cleared in the serialized op, and reload+decode parity.
- **D-10 (test-util dedup, pulled forward from Phase 11):** Extract `SGFP4TestUtil.hpp` (tempPath, container builders, sidecar/offset helpers) from the duplicated helpers across `SGFP4ClassicAPITest.cpp` / `SGFP4MultiTensorTest.cpp` / `SGFP4InjectTest.cpp`, and retrofit those three files onto it. The new Phase 8 tests are born on the shared helpers. This retires the v2.0 audit's test-helper-dedup debt early; using the correct region-relative offset convention from day one prevents another W-1-class divergence.

### converter-pass hand-off contract
- **D-11 (buffer-staging contract, for Phase 11):** Phase 11's PostConverter pass writes `OpT` with `buffer = [container bytes]`, `external = {}`, **no** `externalPath` — a pure graph rewrite with zero byte I/O in the pass. Externalization then happens (or not) via the existing `postTreat`/`saveExternalData` flag machinery using D-05; runtime dispatch (D-01) handles whichever form ships. Sidecar artifact when externalization ON; single-file `.mnn` when OFF.
- **D-12 (documented non-interception):** Phase 8 must document (comment at the `createExecutionWithExternal` switch in `source/core/OpCommonUtils.cpp:665` and/or in `SGFP4DequantParam`'s schema comment) that `SGFP4Dequant` is **not** and need not be one of the auto-rewritten types (`Convolution2D`/`Scale`/`LayerNorm`) — the decoder owns dispatch, so ops flow to `backend->onCreate` unmodified. This stops the Phase 11 planner from hunting for a `createExecutionWithExternal` case that intentionally doesn't exist.
- **D-13 (tech-debt timing unchanged):** W-1 (classic_api offset-convention retrofit) and W-2 (arg-stage failCleanup) stay with the injection tool / Phase 11 per the v2.0 audit's original placement; only the test-helper dedup (part of W-1's bug-class root cause) moves into Phase 8 via D-10.

### Claude's Discretion
- Exact schema comment wording in `CaffeOp.fbs`; whether `buffer` is generated via the same flatc regeneration flow (`schema/generate.ps1` / `generate.sh` → regenerate `MNN.generated.h`, commit both `.fbs` and generated headers).
- Internal structure of the `storeWeight` SGFP4 case (dedicated aligned-store helper vs inline pad logic).
- Test file naming and placement within `test/op/` family conventions.
- Whether decode-parity tests share one parameterized fixture across CPU/Vulkan or two files (existing precedent: separate `SGFP4DequantTest.cpp` / `SGFP4VulkanDequantTest.cpp`).
- Validation-strength details of buffer-mode entry checks (beyond magic + dims consistency) short of reimplementing full T-01-04 (which is file-bounded by nature).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Schema (to be modified)
- `schema/default/CaffeOp.fbs:118` — current `SGFP4DequantParam { magic:uint32; external:[int64]; dims:[int] }` table + its locked design comment ("No macroblock/quadtree/leaf/split-map fields belong here")
- `schema/current/MNN_generated.h` — generated flatbuffers types (must be regenerated after `.fbs` edit: `schema/generate.ps1` on Windows using `3rd_party/flatbuffers/tmp/flatc.exe`)
- `schema/current/CaffeOp_generated.h` — where the SGFP4 param type actually generates (verify after regeneration)

### Converter externalization path (to be modified)
- `tools/converter/source/common/RemoveParams.cpp` — `storeWeight<T>` pattern (write bytes → set `external = {offset, size}` → clear vector), `RemoveAndStoreParam` (per-OpParameter dispatch, the `Blob` case at ~62-105 is the closest analog for byte-vector externalization), `saveExternalData` (oplists + subgraphs loop), `loadExternalParam` (inverse path, needed if converter reloads its own output)
- `tools/converter/source/common/writeFb.cpp:89-176` — `postTreat`: `needExternalWeight` flag resolution (`config.saveExternalData` / `_largeModel`), `.weight` file naming (`config.MNNModel + ".weight"`), `__convert_external_data.bin` staging and cleanup
- `tools/converter/source/common/CommonUtils.hpp` — `RemoveAndStoreParam` / `loadExternalParam` declarations

### Runtime decode path (to be modified)
- `source/backend/cpu/CPUSGFP4Dequant.cpp` — `onResize` gate to be replaced with buffer-first dispatch; T-01-04 file-size bounds check (preserve for sidecar mode); `onExecute` decode via `SGFP4DequantUtils`
- `include/MNN/SGFP4DequantUtils.hpp` — container framing constants, `dequant_sgfp4_container_cpu` oracle decode, byte-verified against gnus-poc exporter
- Vulkan Execution counterpart — locate via `source/backend/vulkan/buffer/execution/` SGFP4 files (mirrors CPU gate; same dispatch change)

### Existing external-mechanism precedent (read-only ground truth)
- `source/core/OpCommonUtils.cpp:543-726` — `_RebuildExternalOp` (Scale/LayerNorm/Convolution2D re-materialization) and `createExecutionWithExternal` at `:665` (the switch SGFP4 intentionally stays out of — see D-12); `useCachedMmap > 1` path passes original op through
- `tools/converter/source/optimizer/postconvert/SplitBlockQuantConvolution.cpp:44-52` — precedent for setting `op->externalPath` + `external` in a post-convert pass

### Test assets (to be modified / created)
- `test/op/SGFP4DequantTest.cpp`, `test/op/SGFP4VulkanDequantTest.cpp` — existing op-construction + sidecar round-trip patterns; both get buffer-mode variants (D-08)
- `test/op/SGFP4DequantFixtures.h` — generated fixtures + expected oracle outputs
- `test/op/SGFP4ClassicAPITest.cpp`, `test/op/SGFP4MultiTensorTest.cpp`, `test/op/SGFP4InjectTest.cpp` — the three files with duplicated helpers to extract into `SGFP4TestUtil.hpp` (D-10); `SGFP4MultiTensorTest.cpp:190-199` holds the correct region-relative offset-convention builder (the W-1 reference fix)
- `tools/fp4/sgfp4_inject*` — the 16-byte-aligned sidecar emission convention D-05 must match

### Workstream planning
- `.planning/workstreams/sgfp4-pivot/ROADMAP.md` §Phase 8 — goal line
- `.planning/milestones/v2.0-REQUIREMENTS.md` §"v3.0 Converter Integration" — SGV2-22/23 requirement text
- `.planning/milestones/v2.0-MILESTONE-AUDIT.md` §tech_debt — W-1/W-2/W-3 items and test-helper duplication finding (context for D-10/D-13)
- `.planning/milestones/v2.0-phases/05-injection-core-artifact-construction-graph-splicing/05-CONTEXT.md` — locked injection-tool contracts (manifest pairing, literal externalPath, merged aligned sidecar) that Phase 8's convention must not break

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `storeWeight<T>` template (`RemoveParams.cpp:14-29`) — the externalization primitive; SGFP4 case needs the aligned variant (D-05/D-06)
- `Blob` case in `RemoveAndStoreParam` — direct precedent for byte-vector (`uint8s`) externalization including the `totalSize <= 1024` inline-threshold idea (SGFP4 case: no such threshold — container bytes are always large and always eligible)
- `dequant_sgfp4_container_cpu` (`SGFP4DequantUtils.hpp`) — deterministic decode oracle for D-08 parity assertions
- Existing sidecar-mode test scaffolding (`SGFP4DequantTest.cpp` op-construction block) — clone-and-vary for buffer-mode tests
- `sgfp4_inject`'s aligned-append emission — the alignment convention reference implementation for D-05

### Established Patterns
- Sidecar + op-resident `externalPath` with per-op `{offset, size}` — the load side (`FileLoader::offset/read`) is fully implemented in both Executions; buffer-mode is purely additive dispatch in front of it
- Op parameters that carry inline data serialize directly (`Blob.float32s` etc.) — `buffer:[byte]` follows the same flatbuffers object-API flow (`SGFP4DequantParamT::buffer` `std::vector<uint8_t>`)
- `MNN_TEST` suite registration `op/sgfp4/*` naming and filtered runs (full `run_test.out` still blocked by unrelated dead `test/op/FP4ModelTest.cpp` — pre-existing, out of scope, tracked in STATE.md)
- Schema edits regenerate via `schema/generate.ps1` (builds `flatc` from `3rd_party/flatbuffers` on demand) — generated headers are committed

### Integration Points
- **Downstream Phase 11** consumes D-05/D-11: its PostConverter pass emits buffer-staged ops and relies on `postTreat` externalization — no byte I/O in the pass
- **Downstream Phases 9/10** (C++ encoder port / validation) produce container bytes that land in `buffer` via the same staging contract
- **Injection tool (`sgfp4_inject`)**: unaffected functionally (it writes sidecars directly), but shares the alignment convention; its emitted artifacts must keep loading (D-04 regression guard)
- **`createExecutionWithExternal`** (`OpCommonUtils.cpp:665`): intentionally NOT extended — documented per D-12

</code_context>

<specifics>
## Specific Ideas

No new user specifics beyond the four recorded decision areas — the discussion was fully structured around the gray-area tables, and each selected option (B / R2 / T3 / H1) was chosen as presented without modification.

</specifics>

<deferred>
## Deferred Ideas

- Non-64-multiple weight shapes / tiling-padding conventions — Phase 10 (unchanged placement)
- CLI flag design for triggering SGFP4 conversion — Phase 11
- W-1 classic_api offset-convention retrofit and W-2 arg-stage failCleanup — Phase 11 per v2.0 audit placement (only the helper dedup moved to Phase 8, D-10/D-13)
- A potential future inline-threshold convenience (small containers staying inline automatically) — unnecessary complexity now; externalization stays flag-driven (D-07)
- Extending `createExecutionWithExternal` interception to SGFP4 — explicitly rejected (D-12); decoder owns dispatch

</deferred>

---

*Phase: 8-Schema + Sidecar Wiring*
*Context gathered: 2026-08-28*
