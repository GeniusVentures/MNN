# Phase 7: Multi-Tensor Hardening & Structured-Data Coverage - Research

**Researched:** 2026-08-27
**Domain:** C++ MNN injection tool hardening + test engineering + one-time Python fixture authoring
**Confidence:** HIGH (nearly all claims verified by direct code reading in this session; only the structured-weight recipe and gnus-poc Python env carry assumptions)

## Summary

Phase 7 is a **proving + hardening** phase, not a build-from-scratch phase. Direct reading of `tools/fp4/sgfp4_inject_core.hpp` confirms the multi-niche-dir machinery already exists and is structurally correct: `run()` accepts repeated `--niche-dir` args (repeatable `std::vector` pushback), validates ALL niche dirs up front in a loop, pairs each manifest shape to exactly one 2-D model weight (hard-failing on zero or 2+ matches), merges all containers into ONE sidecar via a 16-byte-aligned monotonic cursor (`offsetCursor += sgfp4_align16(bytes.size())`), and splices N dequant nodes in a loop. What Phase 7 must add: (a) the **D-11 atomicity fix** — the sidecar is currently written at `sgfp4_inject_core.hpp:340-375` BEFORE graph surgery and the output model is saved BEFORE the in-tool verify, so an error after those points leaves partial `out.mnn`/`out.mnn.weight` on disk; (b) a **structured (LAYOUT_MIXED) fixture** from the real gnus-poc encoder baked as a C-array header; (c) a **malformed-input matrix** — today every listed probe (truncated/empty, bad sha256, v1/magic, manifest missing fields, zero/multi-match) ALREADY exits non-zero with a diagnostic because they are all pre-write validation failures; the only probes requiring new behavior are garbage-body-bytes (write-ordering) and the atomicity assertion itself (no partial files); (d) the **`tools/fp4/README.md`** dims/README deliverable (D-13).

The structured-fixture authoring path is fully understood: `FP4Exporter.export_weights(w, niche, adaptive=True)` → returns `(binary, stats)` where `stats["layout_distribution"]` is a 6-entry dict indexed by layout enum (index 4 = MIXED) — the authoring script can self-assert MIXED presence. `_classify_layout` emits `Layout.MIXED` whenever the quadtree leaf set is NOT a uniform cover of same-size tiles, so a 64×64 superblock whose TL quadrant needs deeper splits than its siblings guarantees MIXED. The gnus-poc `__main__` CLI only exports dummy `randn(512,512)*0.01` noise (all-UNIFORM_64 lineage — do NOT use it for the structured fixture); the authoring script must call `export_weights`/`export_to_file` programmatically with constructed weights.

**Primary recommendation:** Two plans. Plan 1: D-11 atomicity fix (temp-promote or failure-cleanup, planner's choice per discretion) + structured fixture quick-task + `test/op/SGFP4MultiTensorTest.cpp` multi-tensor positive suite (one structured + one uniform container, collision-free offset assertions via reloading the artifact, classic-API run with FP32 parity). Plan 2: the malformed-input probe suite (same file, additional registered suites) + `tools/fp4/README.md`. Both plans verify via filtered `run_test.out op/sgfp4/...` suites plus `sgfp4_inject.out` build smoke, per the Phase 5/6 workaround for the `FP4ModelTest.cpp` blocker.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Structured Container Source & Fixture Form**
- **D-01:** The structured LAYOUT_MIXED/quadtree container is produced by the REAL gnus-poc encoder (`fp4_exporter.py --adaptive` run on structured weights) — NOT MNN's test-oracle-only `tools/fp4/encode_sgfp4.py`.
- **D-02:** The container bytes are baked into the test as a generated C-array header fixture (following the `SGFP4DequantFixtures.h` precedent), with a correct synthetic `manifest.json` (computed sha256 over the fixture bytes) written to a temp dir at runtime — fully self-contained, no runtime Python, no committed binaries, no env-var conditional skips.
- **D-03:** The one-time export run happens at fixture-authoring time (quick-task or manual step in the gnus-poc repo). The cross-repo dependency exists only at authoring time, never in the test harness or CI.
- **D-04:** The structured container runs the same classic-API inject→load→run chain as the uniform container (inside the multi-tensor test) — no separate dedicated suite for the structured container.

**Multi-Tensor Setup**
- **D-05:** Base model topology: `Input[512]` with two distinguishable-shape weights — e.g. `w1[512,512]` (paired with the structured container) plus a second smaller weight (e.g. `[512,64]` or `[64,64]`, paired with the uniform demo-lineage container) — each container pairing to its own niche dir.
- **D-06:** Container mix: one structured (gnus-poc exported) + one uniform (existing demo-container lineage) — a single injected artifact proves BOTH multi-tensor collision-free offsets AND quadtree/structured coverage in one run.
- **D-07:** Validation path is the **classic API** (`Interpreter::createFromFile` → `createSession` → `runSession`). Express `Module::load` coverage remains the tool's unconditional in-tool verify (Phase 5 D-12) — not duplicated in the test.
- **D-08:** Same-shape weight collision handling is **deferred**: exact-shape pairing keeps hard-failing on 2+ matches. Manifest tensor-name keying goes to v3.0 Phase 11.

**Malformed-Input Probe Depth & Clean Failure**
- **D-09:** Full input-failure matrix probed at the `sgfp4_inject::run()` level (in-process): truncated/empty container, bad sha256, legacy v1 container, manifest missing fields, dims disagreeing with the matched tensor, zero-match and multi-match shapes — every case must exit non-zero with a diagnostic.
- **D-10:** Corrupted-payload-byte probes are **included** (from Phase 6 D-14 deferral): a container that passes magic/version/sha256 but has garbage body bytes — the tool must never crash; either clean structural success or clean failure.
- **D-11:** **Atomicity requirement:** a failed run must leave NO partial `out.mnn`/`out.mnn.weight` behind — all validation completes before any output file is written, or output goes to temp files promoted by rename. (Current core validates all niche dirs up front but writes the sidecar before graph surgery/serialization — the planner should verify/adjust write ordering or temp-file strategy.)
- **D-12:** One new `run_test.out` suite file (e.g. `test/op/SGFP4MultiTensorTest.cpp`, registered under `op/sgfp4/multi_tensor`) carries BOTH the positive multi-tensor/structured tests AND the malformed-input probes — single file, Phase 6 D-09 pattern, filtered-suite workaround for the `FP4ModelTest.cpp` full-build blocker.

**dims Convention Documentation**
- **D-13:** SGINJ-07's "documented" requirement = a new `tools/fp4/README.md` covering: the `dims = {dimO, dimI}` matrix convention (2-D `[out, in]` row-major weights only — transposed pairing was rejected in Phase 5), the niche-dir/manifest input contract, CLI usage, and sidecar layout.
- **D-14:** The "applied" requirement is satisfied by the EXISTING cross-checks (manifest `stats.shape` vs matched `.mnn` tensor dims, param `dims` set from the same source) — tests assert these fire; no new mechanism.

### Claude's Discretion
- Exact second-weight shape (`[512,64]` vs `[64,64]`) and how the two MatMuls compose in the test graph (chained vs parallel), provided both weights are 2-D, exact 64-multiple shapes, and distinct from each other.
- The structured fixture's source weights (what structured matrix to feed the exporter — e.g. block-constant + ramp + noise mixture guaranteeing MIXED/non-uniform macroblocks).
- Exact malformed-probe test-case organization within the single suite (one test case vs multiple), and probe-specific byte mutations.
- The atomicity implementation choice (full pre-validation vs temp-file+rename), provided a failed run leaves no partial output.
- README length/structure beyond the four required content areas (D-13); error-message wording; logging verbosity.

### Deferred Ideas (OUT OF SCOPE)
- Same-shape weight disambiguation via manifest tensor-name keying — v3.0 Phase 11.
- Out-of-bounds-offset probing via hand-tampered artifacts — provably unnecessary (monotonic cursor + in-tool verify).
- Non-64-multiple weight shapes / tiling-padding conventions — v3.0 Phase 10.
- Real quantization-error tolerance calibration — v3.0 Phase 10.
- Vulkan E2E with injected artifacts — v3.0 Phase 12.
- gnus-poc `pipeline/runner.py` default-quantize fix — gnus-poc side.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SGINJ-07 | Multiple target weight tensors (and/or multiple containers) inject into a single artifact with independent, collision-free sidecar byte ranges, loading and running correctly; weight `dims = {dimO, dimI}` convention documented and applied | Multi-niche-dir loop + monotonic aligned sidecar cursor exist and are verified correct in `sgfp4_inject_core.hpp` (Q1 below); multi-tensor test graph composition analyzed (Q5); offset-collision assertion technique identified (reload artifact, walk ops, check `{offset,size}` ranges — Phase 5 graph-structure-assertion precedent); README content areas enumerated (Q7) |
| SGINJ-08 | At least one structured (non-uniform) container exercises the LAYOUT_MIXED/quadtree decode path end-to-end; malformed/empty inputs fail cleanly rather than emitting a corrupt artifact | Structured-fixture authoring recipe resolved via `_classify_layout` + `stats["layout_distribution"]` self-assert (Q3); in-tool verify + classic run exercise quadtree decode end-to-end; malformed matrix enumerated with current-code behavior per probe (Q4); D-11 atomicity gap located precisely with two fix options (Q1) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Structured container production (one-time) | Authoring-time Python (gnus-poc repo) | — | D-01/D-03: the REAL encoder is the input contract; dependency exists only at fixture-authoring time |
| Fixture baking (container → C-array header) | Authoring-time codegen (committed header under `test/op/`) | — | D-02: self-contained tests, no runtime Python |
| Synthetic manifest + sha256 at runtime | Test harness (C++) | `tools/fp4/sha256.hpp` | D-02: manifest computed over fixture bytes in temp dir |
| Multi-tensor injection + atomicity | Tool core (`sgfp4_inject_core.hpp`) | CLI shim (unchanged) | D-11 changes write ordering only in the core header |
| Failure-matrix enforcement | Tool core (existing validation) asserted by tests | — | D-09: all probes already fail pre-write except verify-stage |
| Quadtree/structured end-to-end proof | Test suite via classic API (D-07) | In-tool Express verify (Phase 5 D-12, unchanged) | Classic API is the downstream SGProcessingManager path |
| dims convention documentation | `tools/fp4/README.md` (new file) | — | D-13 |
| Decode-domain robustness on garbage bytes | Existing `dequant_sgfp4_container_cpu` (bounds-checked) — read-only | — | D-10: decode robustness is v1.0 territory; the tool just must not crash |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| MNN core (Express + classic API) | in-repo | Graph surgery, `Variable::save/loadMap`, `Interpreter/Session` | The subject under test [VERIFIED: codebase] |
| vendored rapidjson | `3rd_party/rapidjson` | Manifest JSON parsing | Already used by the core header [VERIFIED: codebase] |
| `tools/fp4/sha256.hpp` | in-repo | sha256 for synthetic manifests | Phase 6 D-11 precedent; `sgfp4::sha256_hex(data, size)` API confirmed [VERIFIED: codebase] |
| MNNTestSuite framework | in-repo (`test/MNNTestSuite.h`) | Test registration/execution | `MNNTestSuiteRegister(Class, "op/sgfp4/...")` pattern [VERIFIED: codebase] |
| gnus-poc `fp4_exporter.py` + `numpy` | external (authoring time only) | Structured fixture production | D-01; `FP4Exporter.export_weights(..., adaptive=True)` API confirmed [VERIFIED: external repo read] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `checkVectorByRelativeError<float>` (`test/TestUtils.h`) | in-repo | FP32 parity assertions (rtol 1e-4) | Multi-tensor classic-run parity (Phase 6 D-07 pattern) |
| `dequant_sgfp4_container_cpu` | in-repo | FP32-weight oracle (zero-by-construction baseline, Phase 6 D-06) | Baseline weights for BOTH containers; also MIXED decode oracle |

**Installation:** None — zero new packages. Everything is in-repo or already-vendored.

## Package Legitimacy Audit

Not applicable — this phase installs no external packages (no npm/PyPI/cargo dependencies added; the gnus-poc exporter is an existing sibling-repo script used read-only at authoring time). [VERIFIED: this session's plan — no package installs]

## Answers to the Planner's Key Research Questions

### Q1: Current multi-niche-dir behavior + D-11 atomicity gap

**Multi-niche-dir is already correct for N>1.** [VERIFIED: codebase, `tools/fp4/sgfp4_inject_core.hpp`]

The flow inside `sgfp4_inject::run()`:
1. Arg parse: `--niche-dir` is repeatable (`nicheDirs.push_back`), requires ≥1.
2. **All niche dirs validated up front** (`loadNicheDir` loop, lines ~300-310): unique `*.sgfp4` discovery, manifest parse (`fp4_binary.sha256/path/stats.shape`), basename cross-check, sha256 integrity, `sgfp4_is_v2_container` gate. Any failure → `return 1` BEFORE any file write.
3. Model load (`Variable::loadMap`) + **exact-shape pairing per niche**: enumerates non-input 2-D vars with dims exactly `{dimO, dimI}`; `candidates.size() != 1` → hard error listing candidates (the "hard-fail on 2+ matches" behavior Phase 7 must confirm stays — it will, unchanged).
4. **Sidecar merge** (write begins HERE): one `ofstream` on `<output>.weight`, monotonic `offsetCursor`, each container written then zero-padded to `sgfp4_align16(size)`. Offsets are known before Op construction — final `{offset, size}` baked into ops.
5. Surgery loop: `makeDequantOp` → `Variable::create(Expr::create(...))` → `Variable::replace(weightVar, dequantVar)` per node.
6. `Variable::save(outputs, outputPath)` — output model write.
7. Unconditional in-tool verify: full-module `Module::load` reload + per-node isolated 0-input sub-modules (`<output>.verify_N.mnn` temp files, deleted after) vs `dequant_sgfp4_container_cpu` oracle; mismatch → `return 1`.

**The D-11 gap (precise):** failure paths that occur AFTER step 4 begins leave partial artifacts:
- Sidecar write failure mid-stream → partial `.weight`.
- `Variable::save` failure → sidecar complete, no/partial `.mnn`.
- Any verify failure (step 7) → **both files fully on disk despite the run exiting 1** — this is the concrete case D-11 targets (a corrupt/garbage container reaching verify leaves a full-looking artifact behind, which would crash SGProcessingManager's unchecked-nullptr path).

**Minimal surgical change options (planner's choice per discretion):**

- **Option A — failure-cleanup (smallest diff):** wrap steps 4-7; on any `return 1` after the sidecar open, `std::remove` both `outputPath` and `sidecarPath` before returning. Keep one shared ` fail(path1, path2)` cleanup helper or a small RAII guard. Note: this also removes PRE-EXISTING outputs from an earlier successful run if a new run fails — decide whether that is acceptable (arguably yes: the run was asked to produce this output and failed; leaving a stale artifact that does not correspond to the failed inputs is the safer contract for downstream consumers, and it satisfies D-11's "no partial files" literally because none remain).
- **Option B — temp+rename (stronger):** write sidecar and model to `<output>.tmp.weight` / `<output>.tmp`, run verify against them, then rename both to final names. **Subtlety verified in code:** the ops' `externalPath` is baked as the FINAL sidecar path inside the saved model; during verify of the temp model, `CPUSGFP4Dequant::onResize` reads `mOp->externalPath()` (the final name — Phase 1 documented that `rtmgr->setExternalFile` does NOT populate this op type's path). So under Option B the sidecar must exist at its FINAL path during verify — meaning the sidecar cannot itself be temp-renamed unless verify is performed against a byte-buffer reload (`Module::load` from buffer with `setExternalFile(tempSidecar)` — but the sub-module verify already constructs fresh ops, so it can pass the temp path). The one verifiable clean composition: sidecar written straight to final path; MODEL saved to temp path, verified, then renamed; on failure remove temp model + final sidecar. Existing pre-existing outputs are preserved (rename is atomic overwrite on Windows via `MoveFileEx` or `std::rename` semantics — on Windows `std::rename` fails if destination exists, so use `std::remove(final)` then `std::rename`, or `_wrename`-family; **flag: cross-platform rename-over-existing is not portable in pure C++11** — Option A avoids this entirely).
- **Recommendation: Option A** (failure-cleanup). It is the minimal change, fully testable (probes assert file absence after failed runs), avoids Windows rename-over-existing portability issues, and satisfies D-11's actual wording ("a failed run must leave NO partial out.mnn/out.mnn.weight behind").

Also confirm: **Phase 6 test compatibility** — on the success path Option A changes nothing, so `op/sgfp4/classic_api*` suites stay green without modification.

### Q2: How `SGFP4ClassicAPITest.cpp` builds fixtures — the direct template

[VERIFIED: codebase, `test/op/SGFP4ClassicAPITest.cpp`]

The Phase 6 fixture pipeline (`buildInjectedArtifact`):
1. **In-test uniform container** — `buildContainerUniform64()` writes a 512×512 all-`UNIFORM_64` v2 container byte-by-byte from `kSGFP4*` constants (132,368 bytes, framing identical to the real demo container). Constants: `kRecordCount = 64`, `kRecordSize = 2064` (8B headers + 8B pad + 2048B nibble payload), `kRecordRegionStart = 272 = align16(16 + 64*4)`.
2. Oracle decode → FP32 baseline weight (zero-by-construction parity, D-06).
3. `buildNamedBaseModel(weight, path)` — `_Input({1,512}) 'input'` → `_Const weight 'weight'` → `_MatMul 'output'` → `Variable::save`.
4. `writeNicheDir(bytes, dir)` — creates dir, writes `<dir>/phase6_fixture.sgfp4` + `manifest.json` whose JSON is exactly `{"fp4_binary":{"path":"phase6_fixture.sgfp4","sha256":"<computed>","stats":{"shape":[512,512]}}}` — computed via `sgfp4::sha256_hex`. This minimal manifest satisfies the tool's parser (it requires only `path`(basename cross-check), `sha256`, `stats.shape`; ignores `format`/`container`/etc.).
5. In-process injection: argv array → `sgfp4_inject::run(7, argv)`.
6. Cleanup removes all temp files + dir.

**What a second weight/container pairing requires:** duplicate the niche-dir step with its own temp dir and container name; the base model gains a second `_Const`. For the uniform second container at a non-512×512 shape, `buildContainerUniform64` generalizes directly: `B = (dimO/64)*(dimI/64)` superblocks; all its size constants derive from `kMatrixDim` — parameterize the helper (e.g. `buildContainerUniform64(dimO, dimI, out)`) and assert `sgfp4_is_v2_container` + oracle round-trip as the builder already does for 512×512. For `[512,64]`: B = 8, offset table = 16 + 32 = 48 → `align16` = 48. [VERIFIED: framing arithmetic from `SGFP4DequantUtils.hpp` decode path]

**How the demo container is consumed today:** NOT read from gnus-poc — Phase 6 D-10 explicitly rejected committed binaries and env-var paths; the container is *regenerated in-test* from format constants (byte-framing equivalence with the real demo container asserted: same 132,368 total). Phase 7 follows the same pattern: uniform second weight = generalized in-test builder; structured container = committed C-array header (because a real encoder produced it and it cannot be regenerated from constants).

### Q3: Structured-fixture authoring recipe

[VERIFIED: external repo read — `fp4_exporter.py`, `quadtree.py`, `manifest.py`; plus `sgfp4_format.py` presence confirmed by imports]

**Do NOT use the CLI `__main__`** — it exports `np.random.randn(512,512)*0.01` dummy noise (the all-UNIFORM_64 lineage). Call the API programmatically:

```python
# In gnus-poc repo root (PYTHONPATH=repo root so `quantize.*` imports resolve)
import numpy as np
from quantize.fp4_exporter import FP4Exporter
exporter = FP4Exporter()               # project_root defaults alongside quantize/
binary, stats = exporter.export_weights(weights, "phase7_structured", adaptive=True)
```

**What guarantees MIXED:** `_classify_layout(blocks)` returns `Layout.MIXED` unless ALL quadtree leaves share one size AND count == `(64/size)²`. So any superblock with asymmetric leaf sizes → MIXED. The quadtree (`QuadtreeEncoder._try_block`) accepts a region when its Laplacian-weighted error ≤ `max_mse` threshold for that size (`DEFAULT_V2_THRESHOLDS`: 64→0.01, 32→0.005, 16→0.002, 8→0.001, 4→0.0005), else splits into quadrants. Recipe shape (Claude's discretion area, recommended):
- Most 64×64 tiles: constant or very smooth → pass at size 64/32 (uniform leaves).
- One or a few tiles: **structured asymmetry** — e.g. TL quadrant carries a ramp/gradient with amplitude large enough to fail the 64 and 32 gates but pass at 16/8, while TR/BL/BR quadrants are near-constant — forcing mixed leaf sizes within that superblock. The proven in-repo analog is the v1.0 Phase 2 fixture finding in STATE.md: "TL-quadrant ramp amp 12 = asymmetric MIXED" (that was for `encode_sgfp4.py`'s thresholds; magnitudes need re-tuning for gnus-poc's `DEFAULT_V2_THRESHOLDS`).
- **Caveat on constants:** quadrant-constant ±equal-amplitude blocks can be EXACTLY represented by FP4 affine (codes −8..7 × scale + bias) and may collapse to uniform — vary amplitudes/ramps rather than symmetric constants.
- **Self-assert at authoring time:** `stats["layout_distribution"]` is `{0..5: counts}` — assert `distribution[4] > 0` (≥1 MIXED superblock; index 4 = `Layout.MIXED`, enums confirmed in both `sgfp4_format.py` imports and `SGFP4DequantUtils.hpp`). If zero MIXED, escalate structure amplitudes and re-run. Prefer a distribution with BOTH some uniform and some MIXED superblocks (single container covers both decode paths — matches D-06's one-artifact-proves-both spirit).
- Weight shape must be 64-multiple 2-D (D-05 pairs the structured container with `w1[512,512]` — the e.g. shape; a smaller 64-multiple like 128×128 or 256×256 is permitted by the discretion clause and keeps the C-array header small; flag for the planner: 512×512 structured container ≈ 130-300 KB raw → ~0.7-1.6 MB as `0xNN, ` text; `SGFP4DequantFixtures.h` already carries ~150 KB of arrays, so either size works, smaller is friendlier).

**Manifest schema (for the synthetic manifest, D-02):** the real `ManifestBuilder.build()` emits `manifest_version/niche/base_model/created/fp4_binary{path, sha256, format, container{...}, stats{...}}/training/...`, where `stats` is the exporter's stats dict verbatim (includes `shape`, `layout_distribution`, `total_bytes`, ...). The tool's parser (verified in `loadNicheDir`) reads ONLY: `fp4_binary.sha256` (string), `fp4_binary.path` (basename cross-checked case-insensitively against the discovered `*.sgfp4` file), `fp4_binary.stats.shape` (exactly 2 positive ints). Phase 6's minimal synthetic manifest is the proven pattern; Phase 7 reuses it byte-for-byte for both niche dirs (structured + uniform), only swapping container bytes/names/shapes. Optionally include `format`/`layout_distribution` fields for realism — the tool ignores them either way.

**C-array conversion (precedent `SGFP4DequantFixtures.h`):** emit a header like `test/op/SGFP4StructuredFixtures.h` with `static const unsigned char kStructured_mixed_data[] = {...};` plus metadata constants (`kStructuredDimO/kStructuredDimI`, maybe `kStructuredLayoutMixedCount` captured from stats for a header-comment/documentation of fixture provenance — regeneration command, export parameters, sha256, layout_distribution). The exporting script itself can live in the commit message / README comment; a committed one-liner regeneration note in the header comment matches the `SGFP4DequantFixtures.h` "DO NOT EDIT — regenerate via" convention.

### Q4: Malformed-input matrix — current behavior per probe

All probes run through `sgfp4_inject::run(argc, argv)` in-process (D-09). Current-code behavior, verified by reading `loadNicheDir`/`run()` and `dequant_sgfp4_container_cpu` / `CPUSGFP4Dequant`:

| # | Probe | How to construct | Current behavior | Gap for D-09/D-10/D-11 |
|---|-------|------------------|------------------|------------------------|
| 1 | Empty container | Write 0-byte `.sgfp4`; manifest sha256 computed over the empty bytes (so sha passes) | `sgfp4_is_v2_container` rejects (size < fixed header) → exit 1, diagnostic, no files written | None — assert as-is |
| 2 | Truncated container | First N < 16 bytes (e.g. 15) of a valid container; sha256 recomputed over the truncation | Version gate rejects → exit 1, no files | None |
| 3 | Bad sha256 | Valid container; manifest sha256 string with one hex digit flipped | sha mismatch error → exit 1, no files | None |
| 4 | Legacy v1 container | v1 framing = headers[B]\|offsets[B]\|codes blob (no `SGF4` magic). Cheapest constructions: (a) 32+ bytes of a v1-style header block (Phase 5 `SGFP4InjectV1RejectTest` precedent used a magic-less 32-byte buffer); (b) valid v2 container with magic byte corrupted (`[0] ^= 0xFF`), sha recomputed; (c) valid magic but version byte = 0x01 | Version gate (`magic + version` byte check) rejects → exit 1, no files | None — (b)/(c) additionally prove the gate checks bytes, not `fp4_binary.format` |
| 5 | Manifest missing fields | Omit `sha256`; omit `path`; omit `stats.shape`; `shape` with rank 3 or non-positive ints | Each has a dedicated `kMissing`-style error → exit 1, no files | None (all four parser branches verified present) |
| 6 | Dims disagreement / zero-match | Niche with `stats.shape` that matches no model weight (e.g. `[256,256]` vs model with `[512,512]`+`[512,64]`) | `expected exactly 1 weight match, found 0` → exit 1, no files | None. NOTE: because `dims` and the pairing key come from the same manifest `stats.shape`, "dims disagreeing with the matched tensor" is UNREACHABLE by construction (D-14's existing cross-check); it collapses into the zero-match probe. Planner should state this rather than invent an unreachable case |
| 7 | Multi-match shapes | Base model with TWO `[512,512]` weights + one `[512,512]` niche | `found 2` hard error listing candidates → exit 1, no files | None — confirms D-08 (2+ matches stay a hard fail) |
| 8 | **Garbage body bytes (D-10)** | Valid container whose PAYLOAD-region byte(s) are flipped, **sha256 recomputed and manifest rewritten** (otherwise probe 3 fires instead). Target: nibble-payload byte of record 0 (safe, structural framing intact) | Framing/sha/version all pass → injection proceeds → decode of garbage bytes happens in in-tool verify: either (a) `dequant_sgfp4_container_cpu` returns false (structural bounds-check trips — e.g. if layout enum byte corrupted instead) → verify sub-module fails cleanly, or (b) decode "succeeds" with garbage VALUES → oracle comparison mismatch → `return 1` with diagnostic. **No crash** in either branch: every decode read is bounds-checked (`SGFP4DequantUtils.hpp` documents full ASVS-V5 bounds discipline); `CPUSGFP4Dequant::onExecute` returns `INVALID_VALUE` on decode-false rather than writing garbage | **BOTH files already on disk when run exits 1 → violates D-11 until the atomicity fix lands.** This probe is the atomicity regression test |
| 9 | Atomicity assertion (per D-11) | Run probe 8 (and optionally a `Variable::save`-failure simulation is NOT practically injectable — stick with probe 8) | Today: `out.mnn` + `out.mnn.weight` remain after exit 1 | **The core gap.** After the fix: test asserts `!exists(out.mnn) && !exists(out.mnn.weight)` after every failing probe (cheap: check after probes 1-8 uniformly) |

Additional verified facts feeding this matrix:
- The version gate is byte-level (`sgfp4_read_u32_le(data) == kSGFP4Magic && data[4] == 0x02`, null/min-size guards) and never reads `fp4_binary.format` (Phase 5 decision — the manifest's `"format": "fp4_ultra_v0.2"` label is a terminology trap; never switch to trusting it).
- Verify-stage failures all `return 1` with `MNN_ERROR` diagnostics — never `abort`/throw (exceptions disabled repo-wide). The "never crash" bar is about decode内存-safety, which the bounds-checked oracle + `INVALID_VALUE` Execution path already guarantee. [VERIFIED: `CPUSGFP4Dequant.cpp` + decode oracle]

### Q5: Multi-tensor test graph — recommended composition

**Recommended (Claude's discretion, both options valid):**

- **Chained:** `input[1,512] 'input'` → `MM1(input, w1[512,512]) → [1,512]` → `MM2(h, w2[512,64]) → [1,64] 'output'`. Single named input/output preserved (Phase 6 D-16 named-I/O asserts carry over verbatim); pairing shapes `{512,512}` and `{512,64}` are distinct, both 64-multiples; MatMul validity: `[1,512]×[512,512] = [1,512]`, `[1,512]×[512,64] = [1,64]`. Baseline = same model pre-injection with weights = oracle decodes of the two containers (zero-by-construction parity, D-06 pattern extended to two weights).
- Parallel alternative (`MM1`, `MM2` both off `input`, two outputs) — requires multi-output session handling and breaks the single-`'output'` named pattern; no advantage. Prefer chained.

**Pairing mechanics (verified in core):** the pairing loop runs over the ORIGINAL `varMap` for ALL niches BEFORE any surgery — each niche independently finds its exactly-one 2-D non-input var. `w1`/`w2` names are free; injected nodes become `w1_sgfp4`/`w2_sgfp4` (D-08 suffix convention). The "2+ matches hard-fail" behavior is probe 7's target and is unchanged by multi-tensor.

**Collision-free offset assertion technique (SGINJ-07's core claim):** after injection, reload `out.mnn` via `Variable::loadMap` and walk every reachable expr (Phase 5 `SGFP4InjectTest` A1 graph-structure-assertion precedent): assert exactly 2 `OpType_SGFP4Dequant` ops; read each op's `SGFP4DequantParam.external = {offset, size}` and `dims`; assert (a) `dims == {512,512}` / `{512,64}` respectively, (b) ranges `[offset, offset+size)` disjoint, (c) `offset % 16 == 0` for both, (d) sizes equal the respective container byte sizes, and (e) — the strongest byte-level check — read the sidecar file and `memcmp` each range against the corresponding container bytes (structured container's range vs its fixture array, uniform's range vs the in-test-built bytes). This proves the merged-sidecar cursor end-to-end without trusting the tool's log output.

### Q6: Suite registration + filtered-suite verification

[VERIFIED: `test/MNNTestSuite.h` + Phase 5/6 summaries]

- Pattern: one file `test/op/SGFP4MultiTensorTest.cpp` behind `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE` (all SGFP4 tests gate on it), containing MULTIPLE registered classes, e.g.:
  - `MNNTestSuiteRegister(SGFP4MultiTensorTest, "op/sgfp4/multi_tensor")` — positive: multi-tensor injection + collision-free offsets + classic-API named-I/O run + FP32 parity (covers SGINJ-07 + structured/quadtree half of SGINJ-08 via the structured container in the same injection, D-04/D-06).
  - Malformed probes: D-12 requires them in the same FILE; organization is discretion. Recommended: one class `SGFP4MalformedInputsTest` registered as e.g. `"op/sgfp4/malformed_inputs"` that loops the probe table (each probe = fresh temp dirs + fresh `run()` invocation + assert exit ≠ 0 + assert no partial output files), reporting which probe index failed in its `MNN_ERROR` message. A single suite keeps the family count tidy and matches "one file with positive + negative tests" (Phase 6 D-09 pattern: 2 suites in 1 file).
  - Namespace gotcha (Phase 6 auto-fix, repeat it): include BOTH `using namespace MNN;` and `using namespace MNN::Express;` (MSVC).
  - Include gotcha: `#include "fp4/sgfp4_inject_core.hpp"` resolves because `tools/` is already on `run_test.out`'s include path (confirmed in 06-02 summary; no CMake change needed). Fixture header lives beside the test (`#include "SGFP4StructuredFixtures.h"`).
- **Filtered-suite workaround (exact procedure from 05-01/06-02 summaries):** after every `cmake` configure in `.build`, filter the pre-existing unrelated broken `test/op/FP4ModelTest.cpp` out of the untracked generated `.build/run_test.out.vcxproj` (local unblock only — no tracked file touched; permanent fix owned by the `milestone` workstream). Then:
  - Per task commit: `cd .build; .\run_test.out op/sgfp4/multi_tensor` (and/or `op/sgfp4/malformed_inputs`).
  - Per wave: `.\run_test.out op/sgfp4` (family regression; currently 7 suites → grows), plus `cmake --build . --target sgfp4_inject.out` (standalone-tool build smoke, required by the phase's canonical files note) and ideally one real CLI E2E run.
- PowerShell is the execution shell (Phases 5/6 ran `.build\Release\run_test.out.exe` paths — mirror 06-02's commands).

### Q7: README (D-13) content areas mapped to current code

[VERIFIED: all from `sgfp4_inject_core.hpp` + `SGFP4DequantUtils.hpp`]

1. **dims convention:** `SGFP4DequantParam.dims = {dimO, dimI}` where `dimO = shape[0]` (output rows), `dimI = shape[1]` (input cols); weights are 2-D `[out, in]` row-major ONLY; exact shape match required, 64-multiple dims only, no transposed-pairing tolerance (Phase 5 rejection), no same-shape disambiguation yet (2+ matches hard-fail; v3.0 Phase 11 pointer).
2. **Niche-dir/manifest input contract:** each `--niche-dir` = unmodified `fp4_exporter.py --adaptive` output dir; must contain exactly ONE `*.sgfp4` plus `manifest.json`; required manifest fields `fp4_binary.{sha256, path, stats.shape}` (`path` basename cross-checked, never resolved literally — it is repo-root-relative with backslashes); integrity = sha256 over container bytes; v2-only byte-level version gate (never trusts `fp4_binary.format`, whose `"fp4_ultra_v0.2"` label is a known terminology trap — format name is "SGFP4 v2").
3. **CLI usage:** `sgfp4_inject --model <path> --niche-dir <dir> [--niche-dir <dir>...] --output <path>` (repeat `--niche-dir` per tensor); emits `<output>` + `<output>.weight`; nonzero exit + diagnostic on any failure; no partial outputs left on failure (state after D-11 fix).
4. **Sidecar layout:** single merged `<output>.weight`; per-op ranges non-overlapping; each container placed at a 16-byte-aligned offset (`sgfp4_align16`), zero-padded to the next multiple; per-op descriptor `SGFP4DequantParamT{magic = kSGFP4Magic, external = {offset, size}, dims = {dimO, dimI}}` with `op->externalPath` set literally on the op (documented gotcha: `createExecutionWithExternal` does NOT cover this op type); injected node naming `weight → weight_sgfp4`; unconditional in-tool decode-vs-oracle verify on every run.

## Architecture Patterns

### Recommended Project Structure (phase deliverables)
```
tools/fp4/
├── sgfp4_inject_core.hpp   # MODIFIED: D-11 atomicity (failure-cleanup or temp-promote)
├── sgfp4_inject.cpp        # UNCHANGED (thin shim keeps building)
├── sha256.hpp              # UNCHANGED
└── README.md               # NEW (D-13, four content areas)
test/op/
├── SGFP4MultiTensorTest.cpp     # NEW (D-12): multi_tensor + malformed_inputs suites
└── SGFP4StructuredFixtures.h    # NEW (D-02): gnus-poc-generated structured C-array fixture
(external, authoring time only: gnus-poc export run producing the fixture bytes)
```

### Pattern 1: In-process tool invocation from tests
**What:** argv-array call into `sgfp4_inject::run(argc, argv)` — no subprocess (Phase 6 D-12).
**When to use:** every probe and the positive suite.
**Example:** `SGFP4ClassicAPITest.cpp` `buildInjectedArtifact` — `const char* argv[] = {"sgfp4_inject", "--model", ..., "--niche-dir", ..., "--output", ...}; sgfp4_inject::run(7, argv);` [VERIFIED: codebase]

### Pattern 2: Synthetic niche dir
**What:** temp dir + container file + minimal manifest with runtime-computed sha256.
**When to use:** every niche the tests feed the tool (both positive containers, every probe variant).
**Example:** `writeNicheDir()` in `SGFP4ClassicAPITest.cpp` — parameterize for name/shape/dims. [VERIFIED: codebase]

### Pattern 3: Artifact-structure assertion via reload
**What:** `Variable::loadMap(outPath)` + walk exprs to assert op types/params; plus direct sidecar byte-range `memcmp`.
**When to use:** SGINJ-07 collision-free offset proof.
**Example:** Phase 5 `SGFP4InjectTest` A1 structure assertion (exactly 1 `SGFP4Dequant`, 0 `CONSTANT` exprs). [VERIFIED: 05-01 summary]

### Pattern 4: Failure-cleanup / temp-promote atomicity
**What:** no output file survives a failed run.
**When to use:** D-11. Recommended implementation: shared cleanup helper invoked on every failure `return 1` after the sidecar `ofstream` opens; tests assert absence.
**Sketch** (Option A, planner to finalize):
```cpp
// After sidecarPath is known:
auto failCleanup = [&]() {
    std::remove(outputPath.c_str());
    std::remove(sidecarPath.c_str());
};
// ... every `return 1` from sidecar-write onward becomes { failCleanup(); return 1; }
```

### Anti-Patterns to Avoid
- **Trusting `fp4_binary.format`** for the version gate — byte-level gate only (Phase 5 lock).
- **Relative niche/model/output paths in tests** — Phase 6 Pitfall 3: absolute cwd-anchored temp paths everywhere (`op->externalPath` is baked literally into the artifact).
- **Mutating container bytes without recomputing the manifest sha256** — the probe silently degrades into the bad-sha probe (probe 3 vs probe 8 confusion).
- **Symmetric quadrant constants for the structured weights** — FP4 affine can represent them exactly and the quadtree may not split; use ramps/amplitude asymmetry (Q3).
- **`constexpr` calling `sgfp4_align16`** — MSVC C2131 (inline, not constexpr); compute arithmetically + runtime assert (Phase 6 auto-fix precedent).
- **`using namespace MNN::Express;` alone** — MSVC misses `Interpreter`/`Tensor`/`ErrorCode` in the parent namespace; add `using namespace MNN;` (Phase 6 auto-fix).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| sha256 | custom digest | `tools/fp4/sha256.hpp` (`sgfp4::sha256_hex`) | vendored, already proven in Phase 6 tests |
| Container framing constants | re-derived magic numbers | `MNN::kSGFP4*` constants from `SGFP4DequantUtils.hpp` | Phase 6 pattern; keeps framing tied to the format header |
| JSON manifest emission | string-building helper libraries | plain `std::ostringstream` (Phase 6 `writeNicheDir` pattern) | tool consumes rapidjson; emission is a 4-field literal — nothing more needed |
| FP32-decode oracle | test-side decoder | `dequant_sgfp4_container_cpu` | deterministic oracle already proven identical across CPU Execution/Vulkan |
| Structured container | `encode_sgfp4.py` re-derivation | the REAL gnus-poc exporter (D-01) | input-contract fidelity is the point of the phase |
| Directory ops | `<filesystem>` | `_mkdir`/`mkdir` + `FindFirstFileA`/`dirent` helpers (copy Phase 6/C++11 tool precedent) | C++11 default; phase 5/6 code already solved this portably enough |

**Key insight:** everything except the four deliverables (atomicity fix, fixture, test file, README) already exists and is proven — the phase's risk is concentrated in the fixture-authoring one-timer and the atomicity fix, not in new machinery.

## Runtime State Inventory

Not a rename/refactor/migration phase — SKIPPED per protocol. (No stored-data, service-config, OS-registered, secrets, or build-artifact renames involved; the one adjacent concern — stale generated `.build/run_test.out.vcxproj` picking up `FP4ModelTest.cpp` on reconfigure — is documented under Q6.)

## Common Pitfalls

### Pitfall 1: Atomicity fix breaks Phase 6 suites
**What goes wrong:** restructuring write ordering changes success-path behavior (e.g. renamed intermediates left behind, externalPath pointing at a renamed sidecar).
**Why:** verify reads ops' `externalPath` (final name) — temp-rename schemes interact badly (see Q1 Option B analysis).
**How to avoid:** prefer Option A (failure-cleanup — success path untouched); run `op/sgfp4/` family regression after the change.
**Warning signs:** classic_api suites failing on artifact load after the core edit.

### Pitfall 2: Structured fixture silently all-uniform
**What goes wrong:** authoring weights too "easy" → every superblock passes 64-gate → `layout_distribution[4] == 0` → the phase ships quadtree coverage in name only.
**Why:** FP4 affine fits constants/smooth regions exactly; thresholds (0.01 MSE @ 64) are lenient.
**How to avoid:** authoring script MUST assert `stats["layout_distribution"][Layout.MIXED] > 0` (and record the distribution in the fixture header comment); the C++ test can additionally decode-with-oracle (already does) — but oracle equality does NOT prove MIXED was traversed; keep the stats-based proof at authoring time and optionally bake `kStructuredMixedSuperblocks` into the header as documentation.
**Warning signs:** fixture header's layout_distribution comment shows all-uniform.

### Pitfall 3: vcxproj glob cycle (recurring environment issue)
**What goes wrong:** adding `SGFP4MultiTensorTest.cpp` requires a `cmake` reconfigure in `.build`, which re-adds the broken `FP4ModelTest.cpp` to the generated vcxproj.
**How to avoid:** reconfigure → re-apply the one-line vcxproj filter → build (documented, repeatable; 05-01/06-02 precedent).
**Warning signs:** MSVC errors in `FP4ModelTest.cpp` after configure.

### Pitfall 4: MSVC-specific compile traps in new test file
`constexpr` on inline helpers (C2131), missing `using namespace MNN;`, NOMINMAX interactions — all three auto-fixed in Phase 6; copy the fixed idioms from `SGFP4ClassicAPITest.cpp` verbatim.

### Pitfall 5: Manifest `path` basename cross-check
The tool requires `basename(fp4_binary.path) ==` discovered `*.sgfp4` filename (case-insensitive). When tests write probe niche dirs with custom container names, the manifest `path` field must match — otherwise the probe fails on the WRONG check. (Learned as T-05-03; encoded in `writeNicheDir`.)

### Pitfall 6: Partial-output assertion needs fresh paths
The D-11 test assertions (`!exists(out) && !exists(out.weight)` after failed runs) are only meaningful on fresh temp paths per probe — a leftover file from a previous probe/run would false-fail or false-pass. Reuse Phase 6's `time+rand` temp naming per invocation and clean up after each probe.

## Code Examples

### Generalized uniform container builder (sketch — from Phase 6's builder, parameterized)
```cpp
// Source: test/op/SGFP4ClassicAPITest.cpp (buildContainerUniform64), generalized
bool buildContainerUniform64(int dimO, int dimI, std::vector<uint8_t>& out) {
    const int kBlocksPerRow = dimI / 64, kBlocksPerCol = dimO / 64;
    const size_t recordCount = static_cast<size_t>(kBlocksPerRow) * kBlocksPerCol;
    const size_t tableBytes = 16 + recordCount * 4;
    const size_t regionStart = (tableBytes + 15) & ~size_t(15); // runtime-check vs sgfp4_align16
    // ... same record-emission loop as Phase 6, over recordCount records
}
```

### Collision-free sidecar assertion (sketch — SGINJ-07 core claim)
```cpp
// Source pattern: Phase 5 SGFP4InjectTest structure assertion + sgfp4_inject_core.hpp offsets
auto varMap = Variable::loadMap(outPath.c_str());
// walk exprs: collect {offset, size, dimO, dimI} from each OpType_SGFP4Dequant
// assert: ranges disjoint (offsetA+sizeA <= offsetB || offsetB+sizeB <= offsetA)
// assert: all offsets % 16 == 0; sizes == container byte lengths
// read sidecar file; memcmp(sidecar+offset, containerBytes) == 0 per node
```

### Fixture header (shape — from SGFP4DequantFixtures.h convention)
```cpp
// Auto-generated from gnus-poc FP4Exporter (fp4_exporter.py --adaptive path), DO NOT EDIT.
// Weights: <recipe>; layout_distribution: {uniform64: X, ..., mixed: Y}; sha256: ...
static const unsigned char kStructuredMixedData[] = { 0x53, 0x47, ... };
constexpr int kStructuredDimO = 512, kStructuredDimI = 512;
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single-niche injection assumed but untested (N=1 only) | N>1 loop exists in core since Phase 5 | Phase 5 | Phase 7 proves, not builds |
| Malformed input → non-zero exit (pre-write failures) | Same + atomic no-partial-output (D-11) | This phase | Downstream SGProcessingManager crash-safety |
| Test containers regenerated in-test from constants | Same for uniform; C-array for structured (real encoder) | This phase | Encoder fidelity for quadtree coverage |

**Deprecated/outdated:** none new. (Standing repo note: gnus-poc `pipeline/runner.py` default output is legacy v1 — never consumed here; `fp4_binary.format` label "fp4_ultra_v0.2" is the known terminology trap.)

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | gnus-poc Python env (numpy + `quantize.*` imports) runs as-is on this machine | Q3 / Environment | Fixture authoring blocked → fallback: fix env or hand as manual step to user; no code impact |
| A2 | The recommended structured-weight recipe (asymmetric quadrant structure + ramp) yields ≥1 MIXED superblock under `DEFAULT_V2_THRESHOLDS` | Q3 / Pitfall 2 | Iteration needed at authoring time; stats self-assert makes failure loud, not silent |
| A3 | Generalizing `buildContainerUniform64` to `[512,64]` works (B=8, region start 48) | Q2 / Q5 | Minor arithmetic adjustment in the builder; framing verified against decode path |
| A4 | Option A (failure-cleanup) is acceptable for D-11 (vs strict temp+rename) | Q1 | D-11 wording allows it ("all validation... or temp files promoted by rename" — cleanup achieves the no-partial-files outcome); planner should confirm interpretation in PLAN.md |
| A5 | MSVC `std::remove` early-deletes open files reliably enough for the cleanup path (close streams before remove; ofstreams are scope-closed by then in current code shape) | Q1 | If a stream is still open at cleanup, Windows remove fails → ensure cleanup runs after stream scopes close (planner detail) |

## Open Questions (RESOLVED)

1. **Structured fixture dimensions (512×512 vs smaller)**
   - What we know: D-05 suggests `w1[512,512]` for the structured pair; discretion clause allows other 64-multiples; header size scales ~0.7-1.6 MB as text at 512².
   - What's unclear: user's tolerance for header size vs fidelity to the suggested topology.
   - Recommendation: default to 512×512 (mirrors D-05's e.g. exactly, strongest claim); planner may drop to 256×256 if the header feels heavy — assert MIXED presence either way.
   - **Decision:** 512×512 adopted (Plan 07-02 — mirrors D-05 exactly).
2. **Exact malformed-probe suite organization** (one looping class vs a few targeted classes) — pure discretion; recommendation is one looping class (`op/sgfp4/malformed_inputs`) with per-probe diagnostics, keeping the family count at +2 suites total.
   - **Decision:** one looping probe class with per-probe index/name diagnostics (Plan 07-03 Task 2).
3. **Fix windows for cleanup vs. pre-existing outputs** (Option A removing a PREVIOUS run's good artifact when a new run over the same output path fails) — decide and document in PLAN.md; either behavior satisfies D-11's literal text.
   - **Decision:** failed runs remove stale artifacts from previous runs at the same paths; semantics documented in the `failCleanup` comment (Plan 07-01 Task 1).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| MSVC build tree (`.build`) + `run_test.out` | all test verification | ✓ | Phase 5/6-proven | — |
| `sgfp4_inject.out` target build | tool standalone smoke | ✓ (existing target) | — | — |
| gnus-poc repo (`W:\gnus\GeniusCognitiveSystem\...\gnus-poc`) | fixture authoring only | ✓ (files read this session) | — | manual export by user |
| Python 3 + numpy in gnus-poc env | fixture authoring only | ✓ [ASSUMED] (repo previously produced demo artifact; not probed this session) | — | probe at quick-task start; fix env or hand off |
| `tools/` on `run_test.out` include path | `#include "fp4/sgfp4_inject_core.hpp"` | ✓ (confirmed in 06-02 summary) | — | — |
| PowerShell | command conventions | ✓ | — | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** numpy env assumption (A1) — quick-task probes and recovers.

## Validation Architecture

> Nyquist validation enabled (`.planning/config.json`: `workflow.nyquist_validation: true`).

### Test Framework
| Property | Value |
|----------|-------|
| Framework | MNN native test framework (`MNNTestSuite`, `run_test.out`) — in-process tool invocation via shared core header |
| Config file | `test/CMakeLists.txt` (existing glob — new files picked up on reconfigure) + `tools/fp4/CMakeLists.txt` (existing target) |
| Quick run command | `cd .build; .\run_test.out op/sgfp4/multi_tensor` (PowerShell; same for `op/sgfp4/malformed_inputs`) |
| Full suite command | `cd .build; .\run_test.out op/sgfp4` (family; full binary still blocked by unrelated `FP4ModelTest.cpp` — filtered-workaround per STATE.md) + `cmake --build . --target sgfp4_inject.out` |

### Critical Failure Modes of This Phase's Implementation
1. **Atomicity regression/bug** — failed run leaves partial artifact (downstream = SGProcessingManager nullptr crash). Highest severity.
2. **Sidecar offset collision** between the two containers (would corrupt weights silently if verify tolerance passed — it can't; but the assert must exist).
3. **Structured fixture not actually MIXED** — quadtree coverage claimed but all-uniform bytes shipped (Pitfall 2).
4. **Tool crash on garbage payload** (D-10) — must be clean failure/structural success.
5. **Pairing ambiguity mishandled** — 2+ matches must stay a hard fail (D-08).
6. **Standalone tool build break** from core-header edits (must keep `sgfp4_inject.out` + Phase 5/6 suites green).

### Validation Dimensions (unit/integration/e2e boundaries for the injection tool)
| Dimension | Boundary | Coverage in this phase |
|-----------|----------|------------------------|
| Unit (probe-level) | `sgfp4_inject::run()` exit codes + on-disk file absence, per malformed probe | `op/sgfp4/malformed_inputs` (probes 1-8 of Q4 table) |
| Integration (tool pipeline) | niche dirs → pairing → merged sidecar → surgery → save → in-tool verify | `op/sgfp4/multi_tensor` structure assertions (reload + offset/memcmp) |
| E2E (consumer path) | classic `Interpreter::createFromFile → createSession → runSession`, FP32 parity, named I/O | `op/sgfp4/multi_tensor` classic run (structured + uniform weights in one artifact) |
| Build | `sgfp4_inject.out` standalone build + CLI smoke; `op/sgfp4/` family regression | per-wave commands |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SGINJ-07 (multi-tensor + offsets) | 2 containers → 1 artifact, disjoint aligned ranges, byte-identical ranges, classic run parity | integration | `cd .build; .\run_test.out op/sgfp4/multi_tensor` | ❌ Wave 0 (`test/op/SGFP4MultiTensorTest.cpp`) |
| SGINJ-08 (structured/quadtree E2E) | same suite: structured container decoded through classic path, parity vs oracle baseline | integration | `cd .build; .\run_test.out op/sgfp4/multi_tensor` | ❌ Wave 0 (depends on fixture authoring) |
| SGINJ-08 (malformed → clean fail) | all probes exit ≠ 0 with diagnostic, no partial outputs, no crash | unit | `cd .build; .\run_test.out op/sgfp4/malformed_inputs` | ❌ Wave 0 |
| D-11 atomicity | garbage-body probe leaves no `out.mnn`/`out.mnn.weight` | unit | (same suite, probe 8 + absence assert) | ❌ Wave 0 |
| D-13 README | four content areas present | source assertion (grep/manual) | `git show` review + verify-work | ❌ (new `tools/fp4/README.md`) |

### Sampling Rate
- **Per task commit:** the specific new suite (`op/sgfp4/multi_tensor` or `op/sgfp4/malformed_inputs`) — target < 60s (both suites well under; fixture work is Python-side, asserted at authoring time).
- **Per wave merge:** `.\run_test.out op/sgfp4` family (+ expected count), `cmake --build . --target sgfp4_inject.out`, plus one full CLI E2E (Phase 5 demo lineage or test-style niche dirs) asserting artifact files appear on success.
- **Phase gate:** all `op/sgfp4/*` green, standalone build OK, no regressions in `classic_api*` after the core-header edit (Pitfall 1), README reviewed.
- **Max feedback latency:** ~2-3 minutes (rebuild + family run).

### Reference Datasets / Fixtures
| Fixture | Source | Role |
|---------|--------|------|
| Uniform `[512,64]` (or `[64,64]`) container | in-test generalized builder (Phase 6 pattern) | second weight, demo-lineage uniform coverage |
| Structured container C-array | gnus-poc `FP4Exporter`, `--adaptive` path, MIXED-asserted (Q3 recipe) | quadtree/LAYOUT_MIXED end-to-end proof |
| Malformed probe bytes | in-test mutations of the uniform builder output (+ manifest edits; sha recomputed where the probe targets a later gate) | failure matrix |
| Oracle decodes | `dequant_sgfp4_container_cpu` over both containers | FP32 baseline weights (zero-by-construction parity) |

### Wave 0 Gaps
- [ ] `test/op/SGFP4MultiTensorTest.cpp` — both suites (REQ: SGINJ-07, SGINJ-08)
- [ ] `test/op/SGFP4StructuredFixtures.h` — generated fixture (blocks structured suite; authoring quick-task)
- [ ] `tools/fp4/README.md` — D-13
- [ ] No framework install/config needed — infrastructure complete.

## Security Domain

> `security_enforcement` absent in config → enabled. Phase handles untrusted input bytes at authoring boundaries already hardened in prior phases; no new attack surface is introduced by Phase 7 (it ADDS assertions about existing guards).

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | Existing: byte-level version gate, sha256 integrity, bounds-checked decode (`dequant_sgfp4_container_cpu` ASVS-V5 discipline), `CPUSGFP4Dequant` DoS bound (T-01-04) — Phase 7 asserts these fire (D-09/D-10), adds none |
| V6 Cryptography | no (sha256 used for integrity detection, not secrecy) | vendored `sha256.hpp` — not hand-rolled this phase |
| V2/V3/V4/V10+ | no | no auth/session/network/API surface in this tool |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious/tampered container → garbage weights or crash | Tampering | sha256 manifest check + version gate + DoS-bounded, bounds-checked decode — all existing; D-10 probe proves no-crash |
| Corrupt artifact reaches downstream consumer (crash the unchecked-nullptr path) | Denial of Service | D-11 atomicity — the phase's core hardening contribution |
| Oversized-allocation DoS via crafted `external.size` | DoS | existing on-disk size probe in `CPUSGFP4Dequant::onResize` (T-01-04) — untouched |

## Sources

### Primary (HIGH confidence — direct code reading this session)
- `tools/fp4/sgfp4_inject_core.hpp` — full multi-niche flow, write ordering, verify chain, D-11 gap location
- `tools/fp4/sgfp4_inject.cpp`, `tools/fp4/CMakeLists.txt` — shim + build wiring
- `test/op/SGFP4ClassicAPITest.cpp` — complete fixture/niche-dir/classic-API template
- `test/op/SGFP4DequantFixtures.h` — C-array fixture precedent (incl. existing MIXED fixtures from the oracle encoder — NOT usable for D-01, but structural precedent)
- `include/MNN/SGFP4DequantUtils.hpp` — framing constants, version gate, bounds-checked decode (garbage-byte behavior)
- `source/backend/cpu/CPUSGFP4Dequant.cpp` — onResize/onExecute failure semantics (error codes, no crash)
- `test/CMakeLists.txt`, `.planning/codebase/TESTING.md` — suite build/registration conventions
- `.planning/workstreams/sgfp4-pivot/STATE.md`, `REQUIREMENTS.md`, `ROADMAP.md` — locked decisions, blockers, phase criteria
- Phase 5/6 CONTEXT.md + 05-01/06-02 SUMMARIES + 06-VALIDATION.md — precedents, auto-fixes, filtered-suite workaround
- External (authoring-time): `W:\gnus\GeniusCognitiveSystem\GNUS-NEO-SWARM\gnus-poc\quantize\{fp4_exporter.py, quadtree.py, manifest.py}` — exporter API, `_classify_layout`/`_build_split_map`, thresholds, `layout_distribution` stats, manifest schema

### Secondary (MEDIUM)
- None required — domain is fully in-repo/in-sibling-repo.

### Tertiary (LOW / marked for validation)
- numpy availability in the gnus-poc Python env (A1) — probe at quick-task start.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — everything in-repo, read directly
- Architecture/multi-tensor mechanics: HIGH — core header read line-by-line; N>1 path verified structurally correct
- Malformed matrix: HIGH — every probe traced to a concrete code branch; only garbage-bytes-through-verify empirically untested (it IS the test)
- Structured fixture recipe: MEDIUM — exporter internals verified; the exact weight values yielding MIXED need one authoring-time iteration (asserted loudly, not silently)
- Atomicity fix: HIGH on gap identification; MEDIUM on Option-B portability details (documented; Option A recommended avoids them)

**Research date:** 2026-08-27
**Valid until:** 2026-09-27 (stable in-repo domain; gnus-poc exporter is a moving sibling repo — re-verify its API if the fixture quick-task runs > 30 days out)
