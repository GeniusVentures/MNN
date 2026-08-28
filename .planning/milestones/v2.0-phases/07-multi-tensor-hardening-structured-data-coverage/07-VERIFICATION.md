---
phase: 07-multi-tensor-hardening-structured-data-coverage
verified: 2026-08-27T00:00:00Z
status: passed
score: 11/11 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: none
  previous_score: n/a
  gaps_closed: []
  gaps_remaining: []
  regressions: []
---

# Phase 7: Multi-Tensor Hardening & Structured-Data Coverage Verification Report

**Phase Goal (ROADMAP §Phase 7):** The tool handles realistic multi-weight models and the full SGFP4 v2 format surface — not just the single uniform-random demo artifact — and fails cleanly on malformed input.
**Verified:** 2026-08-27 (verifier session; phase artifacts dated 2026-08-28 local)
**Status:** passed
**Re-verification:** No — initial verification (no prior `*-VERIFICATION.md` in phase dir)

## Goal Achievement

### Observable Truths

Roadmap success criteria (the contract) merged with PLAN must_haves truths.

| #   | Truth | Source | Status | Evidence |
| --- | ----- | ------ | ------ | -------- |
| 1 | Multiple target weight tensors (and/or multiple containers) inject into a single artifact with independent, collision-free sidecar byte ranges, loading and running correctly | ROADMAP SC-1 / SGINJ-07 | ✓ VERIFIED | `run_test.out op/sgfp4/multi_tensor` → passed:1 failed:0 (re-run by verifier). Suite asserts exactly 2 `OpType_SGFP4Dequant` ops, disjoint ranges, `offset % 16 == 0` (~L473), per-range sidecar `memcmp` byte-identity (~L548), classic `Interpreter/Session` run with named `input`/`output` I/O (~L585-586) and FP32 parity via `checkVectorByRelativeError<float>` (~L617) |
| 2 | At least one structured (non-uniform) container exercises the LAYOUT_MIXED/quadtree decode path end-to-end (fixture from the REAL gnus-poc encoder; committed as C-array header) | ROADMAP SC-2 / SGINJ-08 | ✓ VERIFIED | `test/op/SGFP4StructuredFixtures.h`: `kStructuredMixedData[]` (140,240 bytes), provenance records `layout_distribution: {0: 52, 4: 12}` (MIXED=12), sha256 `9ebb8c1f...257`, regeneration command. Verifier re-ran `python tools/fp4/author_structured_fixture.py` to a temp path → exit 0, identical sha256, `fc.exe /b` vs committed header → **no differences** (deterministic round-trip). MIXED end-to-end: the mixed container IS `w1` in the multi_tensor chained MatMul and decodes through `dequant_sgfp4_container_cpu` + inject→classic-run (suite green). `kStructuredMixedCount = 12` guard re-asserted in-test (~L339) |
| 3 | Weight dims convention (`dims = {dimO, dimI}`) documented and applied; malformed/empty inputs fail cleanly (no partial output artifact, non-zero exit, diagnostic) | ROADMAP SC-3 / SGINJ-07+08 | ✓ VERIFIED | README `tools/fp4/README.md` §"dims convention (dims = {dimO, dimI})" documents `dimO = shape[0]`/`dimI = shape[1]`, 2-D exact-shape pairing, 2+-match hard-fail. `run_test.out op/sgfp4/malformed_inputs` → passed:1 failed:0 (verifier re-run): all 13 probes (empty, truncated, bad-sha, bad-magic, version-1, 5 manifest-field variants, zero-match, multi-match, garbage-body) exit ≠ 0 with per-probe `MNN_ERROR`/`MNN_PRINT` diagnostics and zero output files — run log shows each "failed cleanly, no partial output PASSED" |
| 4 | A failed sgfp4_inject run leaves NO out.mnn / out.mnn.weight behind (D-11) | 07-01 must_have | ✓ VERIFIED | `sgfp4_inject_core.hpp` L304: `failCleanup` lambda defined immediately after `sidecarPath` (L296); grep shows exactly 12 `failCleanup()`-preceded `return 1` sites (L315/325/354/374/392/424/441/447/454/461/468/483) and only the 2 pre-sidecarPath arg-validation returns (L289/294) without it — matches plan intent exactly. Behavioral proof: malformed_inputs suite seeds stale artifacts per probe then asserts both files absent — all 13 pass |
| 5 | A successful run produces byte-identical output to the pre-change tool (Phase 6 classic_api suites stay green) | 07-01 must_have | ✓ VERIFIED | `run_test.out op/sgfp4` full family → passed:9 failed:0, including `classic_api` and `classic_api_missing_sidecar` (verifier re-run) |
| 6 | The dims = {dimO, dimI} convention, niche-dir/manifest contract, CLI usage, and sidecar layout are documented in tools/fp4/README.md (D-13) | 07-01 must_have | ✓ VERIFIED | README verified on disk (74 lines): CLI synopsis (`sgfp4_inject --model ... --niche-dir ...`), niche-dir/manifest contract (path/sha256/stats.shape, byte-level v2 gate, `fp4_ultra_v0.2` terminology-trap warning), dims convention section, sidecar layout (16-byte-aligned, `externalPath` gotcha, `weight_sgfp4` naming), failure-behavior section (non-zero exit + no partial outputs) |
| 7 | A structured (LAYOUT_MIXED) SGFP4 v2 container produced by the REAL gnus-poc encoder exists as a committed C-array fixture header | 07-02 must_have | ✓ VERIFIED | `test/op/SGFP4StructuredFixtures.h` committed (8,793 lines); produced by `author_structured_fixture.py` calling `FP4Exporter.export_weights(weights, "phase7_structured", adaptive=True)` (script L82) — programmatic API, not the dummy-noise `__main__`. Commit `c72f1d1b` on `MNN_Ultra_v2` |
| 8 | The fixture's header comment records provenance (regeneration command, weight recipe, layout_distribution, sha256) proving MIXED presence at authoring time | 07-02 must_have | ✓ VERIFIED | Header lines 1-14: `DO NOT EDIT BY HAND` + regeneration command, recipe description, full `layout_distribution`, MIXED count 12, sha256, byte length; no timestamps. Verifier's own regeneration reproduced identical sha256 — provenance is honest |
| 9 | Two containers inject into a single artifact whose dequant ops carry disjoint 16-byte-aligned sidecar ranges, byte-identical to the source containers | 07-03 must_have | ✓ VERIFIED | Source assertions at L473 (`offset % 16`), L548 (`memcmp` per range); suite executed green with disjoint ranges ([0,140240) vs [140240,156800) per SUMMARY, re-confirmed by passing run) |
| 10 | The multi-tensor artifact loads and runs via classic Interpreter/Session with named I/O and FP32-parity against oracle-decoded weights (structured/quadtree coverage inside the multi-tensor suite) | 07-03 must_have | ✓ VERIFIED | L585/586 `getSessionInputAll`/`getSessionOutputAll`, L617 `checkVectorByRelativeError<float>(..., 64, rtol)` vs pre-injection FP32 baseline; suite green. Structured container runs the same inject→load→run chain (D-04) |
| 11 | Every malformed-input probe exits non-zero with a diagnostic and leaves NO output files behind (incl. stale artifacts) | 07-03 must_have | ✓ VERIFIED | 13-probe table L653-674; per-probe: fresh time+rand paths, stale-artifact seeding (L911 area), `rc != 0` assert (L941), file-absence assert; all 13 "failed cleanly, no partial output PASSED" in verifier's own run. Multi-match probe ("found 2" hard-fail) locks D-08; dims-disagreement anti-case documented as unreachable-by-construction (L970-973) |

**Score:** 11/11 truths verified

### Behavioral Spot-Checks (verifier-executed ground truth)

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Full sgfp4 family | `.\.build\Debug\run_test.out.exe op/sgfp4` | passed:9 failed:0 skipped:0 | ✓ PASS |
| Multi-tensor suite | `.\.build\Debug\run_test.out.exe op/sgfp4/multi_tensor` | passed:1 failed:0 | ✓ PASS |
| Malformed-input suite | `.\.build\Debug\run_test.out.exe op/sgfp4/malformed_inputs` | passed:1 failed:0 (all 13 probes) | ✓ PASS |
| Standalone tool build | `cmake --build .build --target sgfp4_inject.out --config Debug` | sgfp4_inject.out.exe linked, exit 0 | ✓ PASS |
| Fixture determinism | `python tools/fp4/author_structured_fixture.py <temp>` + `fc.exe /b` vs committed | exit 0, identical sha256 `9ebb8c1f...257`, "FC: no differences encountered" | ✓ PASS |

### Probe Execution

| Probe | Command | Result | Status |
| ----- | ------- | ------ | ------ |
| op/sgfp4 family (incl. both new suites) | `run_test.out.exe op/sgfp4` | 9/9 passed | PASS |
| op/sgfp4/multi_tensor | `run_test.out.exe op/sgfp4/multi_tensor` | 1/1 passed | PASS |
| op/sgfp4/malformed_inputs | `run_test.out.exe op/sgfp4/malformed_inputs` | 1/1 passed (13/13 probes) | PASS |

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `tools/fp4/sgfp4_inject_core.hpp` | `failCleanup` at 12 post-sidecarPath failure sites | ✓ VERIFIED | L304 lambda; 12 call sites confirmed by grep — count exact |
| `tools/fp4/README.md` | four D-13 areas + failure semantics | ✓ VERIFIED | all sections present, facts match code (sha256 gate, byte-level v2 gate, align16 naming) |
| `tools/fp4/author_structured_fixture.py` | re-runnable authoring, `layout_distribution`, `adaptive=True`, MIXED assert | ✓ VERIFIED | L25 GNUST_POC_ROOT, L82 export_weights(adaptive=True), L86 literal assert, L91-92 framing self-check |
| `test/op/SGFP4StructuredFixtures.h` | `kStructuredMixedData`, dims 512/512, size, MixedCount=12, provenance | ✓ VERIFIED | all symbols present (L8788-8793); sha256 matches verifier regeneration |
| `test/op/SGFP4MultiTensorTest.cpp` | two MNNTestSuiteRegister strings, multi-tensor + malformed suites | ✓ VERIFIED | L634 `op/sgfp4/multi_tensor`, L981 `op/sgfp4/malformed_inputs` (exact strings) |

All artifacts: exists ✓, substantive ✓ (no stubs — real assertions, builders, probe machinery), wired ✓ (fixtures consumed by the test; core header compiled into both run_test.out and sgfp4_inject.out).

### Key Link Verification

| From | To | Via | Status |
| ---- | -- | --- | ------ |
| `SGFP4MultiTensorTest.cpp` | `SGFP4StructuredFixtures.h` | `kStructuredMixedData` bytes into synthetic niche dir | ✓ WIRED (L412-413, L535, L860-861) |
| `SGFP4MultiTensorTest.cpp` | `sgfp4_inject_core.hpp` | in-process `sgfp4_inject::run(argc, argv)` | ✓ WIRED (L429 two-niche argc=9; L937 single-niche argc=7) |
| `sgfp4_inject_core.hpp::run()` | `std::remove(outputPath/sidecarPath)` | `failCleanup(); return 1;` pattern | ✓ WIRED (12 sites) |
| `author_structured_fixture.py` | `SGFP4StructuredFixtures.h` | C-array emission | ✓ WIRED (emits `kStructuredMixedData`; byte-identical round-trip proven) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| SGINJ-07 | 07-01 (truth-3/doc half), 07-03 (behavioral half) | Multi-tensor/container injection, collision-free sidecar ranges, load+run correctly; dims convention documented and applied | ✓ SATISFIED | multi_tensor suite green (disjoint/aligned/byte-identical + classic run parity); README dims section + applied in `makeDequantOp` (parities proven by run) |
| SGINJ-08 | 07-01 (cleanup half), 07-02 (fixture), 07-03 (E2E + failure matrix) | Structured MIXED container decodes end-to-end; malformed/empty inputs fail cleanly, no corrupt artifact | ✓ SATISFIED | MIXED fixture from real encoder (12 MIXED superblocks, determinism proven); structured w1 through inject→classic run; 13/13 clean-fail probes with zero partial outputs |

Orphaned requirements: none — REQUIREMENTS.md maps exactly SGINJ-07 and SGINJ-08 to Phase 7, both claimed by plans and satisfied. No later-phase deferral needed.

### Git Evidence

All seven documented commits verified on `MNN_Ultra_v2`: `025d96b2` (failCleanup), `2e3c9385` (README), `5330a673` (07-01 summary), `c72f1d1b` (fixture script+header), `6c32b140` (07-02 summary), `f2f7b76e` (multi-tensor+malformed suites), `05736d32` (07-03 summary).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| (none) | — | No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER, no stub returns, no console-only implementations in any phase-modified file | — | — |

Note (informational, not a gap): the `kGarbageBody` probe enum comment says "payload byte flipped" while the implementation corrupts a framing byte (record-0 sb_header layout enum → 7). This is the documented 07-03 SUMMARY decision — affine decode is total over payload nibbles so payload garbage is structural success by design; the framing variant is the one that deterministically reaches the D-11 cleanup path. Intent (clean failure + no partial output on structurally-garbage input) fully preserved; probe passes.

### Data-Flow Trace (Level 4)

Not applicable in the rendering sense — this phase produces a CLI tool and C++ unit tests, no UI. Equivalent check performed: fixture header (data source) → test binary decode/inject (real bytes, 140,240) → assertions over sidecar `memcmp`, named-I/O session runs, and FP32 parity — real data flows end-to-end, confirmed by green runs. The fixture is not hardcoded-empty; its sha256 was independently reproduced by the verifier's regeneration run.

### Human Verification Required

None. All phase outcomes are command-line observable and were re-executed by the verifier (test suites, tool build, fixture regeneration determinism). No UI, external services, or subjective quality judgments in scope.

### Gaps Summary

No gaps. All three roadmap success criteria are observably true with verifier-executed evidence (not SUMMARY claims): (1) multi-container injection with disjoint byte-identical sidecar ranges runs through the classic API, (2) a real-encoder LAYOUT_MIXED fixture (12 MIXED superblocks, byte-reproducible) exercises the quadtree path end-to-end, (3) the dims convention is documented and a 13-probe malformed-input matrix fails cleanly with zero partial or stale artifacts. SGINJ-07 and SGINJ-08 are both satisfied; family regression 9/9 green; standalone tool builds; all documented commits present on `MNN_Ultra_v2`.

---

_Verified: 2026-08-27_
_Verifier: the agent (gsd-verifier)_
