---
phase: 07-multi-tensor-hardening-structured-data-coverage
plan: 03
subsystem: testing
tags: [sgfp4, fp4, multi-tensor, malformed-inputs, atomicity, classic-api]

requires:
  - phase: 07-multi-tensor-hardening-structured-data-coverage
    provides: "07-01 failCleanup (D-11) in sgfp4_inject::run; 07-02 structured MIXED fixture header"
provides:
  - op/sgfp4/multi_tensor suite — 2 containers → 1 artifact, disjoint 16-byte-aligned byte-identical ranges, classic-API load/run with FP32 parity (SGINJ-07; structured half of SGINJ-08)
  - op/sgfp4/malformed_inputs suite — 13-probe failure matrix, every probe exit ≠ 0 + no partial output (SGINJ-08; D-09/D-10/D-11; D-08 multi-match lock)
affects: [v3.0 converter integration (reuses suite patterns), SGProcessingManager downstream safety proof]

tech-stack:
  added: []
  patterns: [probe-table looping test over mutation kinds, stale-artifact seeding to prove D-11 stale removal]

key-files:
  created: [test/op/SGFP4MultiTensorTest.cpp]
  modified: []

key-decisions:
  - "Garbage-body probe corrupts a FRAMING byte (record-0 sb_header layout enum -> 7) rather than a payload nibble: the affine decode is total over payload nibbles, so payload garbage is a structural SUCCESS by design; framing corruption is the D-10 probe that reaches the D-11 cleanup path"
  - "Every probe seeds stale out.mnn/out.mnn.weight before running, so the no-partial-output assertion also proves stale-artifact removal (07-01's documented semantics)"
  - "Fix to generalized buildContainerUniform64: offset-table entries are REGION-RELATIVE (encoder convention: cursor from 0; decoder recomputes regionStart+rel). The Phase-6 copy wrote absolute offsets — decode silently misread headers"
  - "For-loop assertion failures use flag+post-loop break (raw break inside a range-for does NOT exit the enclosing do-while — caught when a byte-identity error printed but the suite still passed)"

patterns-established:
  - "Diagnostics-in-noise: a suite that prints MNN_ERROR lines but reports PASSED is an assertion-propagation bug, not success — always grep the log, not just the exit code"

requirements-completed: [SGINJ-07, SGINJ-08]

coverage:
  - id: D1
    description: "SGINJ-07: two containers (structured MIXED fixture + in-test uniform) inject into a single artifact with disjoint 16-byte-aligned byte-identical sidecar ranges; artifact loads/runs via classic Interpreter/Session with named input/output and FP32 parity (rtol 1e-4) over the 64-float output"
    requirement: "SGINJ-07"
    verification:
      - kind: unit
        ref: "run_test.out op/sgfp4/multi_tensor → passed:1 (2 dequant ops, disjoint ranges [0,140240) vs [140240,156800), memcmp byte-identity per range, input/output named I/O, parity vs pre-injection base)"
        status: pass
    human_judgment: false
  - id: D2
    description: "SGINJ-08: structured LAYOUT_MIXED container decoded end-to-end through inject→classic run (inside multi_tensor suite per D-04)"
    requirement: "SGINJ-08"
    verification:
      - kind: unit
        ref: "multi_tensor w1 = oracle decode of kStructuredMixedData; MIXED provenance guard kStructuredMixedCount(12) > 0 re-asserted"
        status: pass
    human_judgment: false
  - id: D3
    description: "SGINJ-08 clean-failure: full malformed-input matrix (empty, truncated, bad-sha, magic-flip, version-1, 5 manifest-field variants, zero-match, multi-match, framing-garbage) — exit ≠ 0 + zero output files including seeded stale artifacts"
    requirement: "SGINJ-08"
    verification:
      - kind: unit
        ref: "run_test.out op/sgfp4/malformed_inputs → all 13 probes PASS with per-probe index/name diagnostics and D-11 file-absence assertions"
        status: pass
  - id: D4
    description: "Family regression: op/sgfp4 full family green at 9 suites (7 pre-existing + 2 new); classic_api* byte-identity after 07-01 core-header edit confirmed"
    requirement: "SGINJ-07"
    verification:
      - kind: unit
        ref: "run_test.out op/sgfp4 → passed:9 failed:0"
        status: pass
      - kind: unit
        ref: "cmake --build --target sgfp4_inject.out → exit 0 (standalone tool still builds)"
        status: pass
    human_judgment: false

duration: 80min
completed: 2026-08-28
status: complete
---

# Plan 07-03: Multi-tensor + malformed-input suites Summary

**One artifact, two containers, everything proven: 2-container injection with disjoint byte-identical sidecar ranges + classic-API FP32 parity, and a 13-probe malformed-input matrix where every failure leaves zero output files (including seeded stale artifacts).**

## Performance

- **Duration:** ~80 min (incl. diagnostics)
- **Completed:** 2026-08-28
- **Tasks:** 2
- **Files modified:** 1 (created)

## Accomplishments
- `op/sgfp4/multi_tensor`: chained `input[1,512]→MatMul(w1[512,512])→MatMul(w2[512,64])→output` base model; 2-niche in-process injection (argc=9); asserts exactly 2 `OpType_SGFP4Dequant` ops, dims `{512,512}`+`{512,64}`, sizes equal to sources, `offset % 16 == 0`, disjoint ranges, per-range sidecar `memcmp` equality; classic-API run with named I/O and FP32 parity (rtol 1e-4) over 64 outputs.
- `op/sgfp4/malformed_inputs`: one looping test case over a 13-entry probe table (fresh time+rand paths per probe), each asserting `run() != 0` AND absence of both output files after seeding stale artifacts (D-11 regression against 07-01's `failCleanup`); multi-match probe locks the "found 2" hard-fail (D-8); comment documents the dims-disagreement anti-case (unreachable by construction).
- Family regression: `op/sgfp4` 9/9 green.

## Task Commits

1. **Tasks 1+2 (one new file, both suites)** - `f2f7b76e`

## Files Created/Modified
- `test/op/SGFP4MultiTensorTest.cpp` — new (983 lines)

## Decisions Made
- Garbage-body probe = framing corruption (sb_header layout enum → 7): payload-nibble garbage is a total-function decode (structural success by design); framing corruption deterministically fails the decode → reaches the D-11 cleanup path. Documented inline.
- Probe-7 (multi-match) uses the structured fixture bytes as the [512,512] single niche against a dedicated 2× [512,512]-weight model (parallel MatMul + Add).
- Offset-table fix confined to this file's generalized builder (the Phase-6 512×512 file-bytes were self-consistent under their own absolute-offset convention ONLY by accident of the 272-byte region start producing a uniform-looking misread; new file uses the encoder's region-relative convention).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Byte-identity/dims assertion failures did not fail the suite**
- **Found during:** Task 1 (first green run printed "sidecar bytes ... differ" yet PASSED)
- **Issue:** Two compounding bugs: (a) source matching keyed on `dimO` alone — both nodes have dimO=512, so the structured source was compared against the uniform node's range; (b) `break` inside a range-for does not exit the enclosing `do{...}while(false)` — errors logged, suite still passed.
- **Fix:** dims matched as pairs; for-loop failures set a flag checked after the loop. Re-run: honest PASS with zero error lines.
- **Files modified:** test/op/SGFP4MultiTensorTest.cpp
- **Verification:** multi_tensor green with clean log.
- **Committed in:** f2f7b76e

**2. [Rule 1 - Bug] Garbage-payload probe degenerated to structural success**
- **Found during:** Task 2 (probe 12: "run unexpectedly returned 0")
- **Issue:** Plan's probe 8 recipe (flip a payload nibble byte + recompute sha) assumes decode fails on garbage payload — but the affine decode is total over nibbles; garbage payloads decode to different values. Additionally, the diagnostic hunt exposed that the copied builder wrote ABSOLUTE offset-table entries while the format (encoder/decoder) uses REGION-RELATIVE offsets — the decode was misreading headers off-by-regionStart all along (masked because payload-pattern bytes parsed as valid uniform layouts).
- **Fix:** (a) builder switched to region-relative offsets (encoder parity); (b) probe now corrupts record-0's sb_header layout enum to 7 (invalid ≥ 6) — passes gate+sha, deterministically fails structural decode → non-zero exit + failCleanup regression.
- **Files modified:** test/op/SGFP4MultiTensorTest.cpp
- **Verification:** all 13 probes pass; family 9/9; standalone tool still builds.
- **Committed in:** f2f7b76e

---

**Total deviations:** 2 auto-fixed (2 bug-class)
**Impact on plan:** Both were correctness fixes to the test itself; no scope change. The offset-table discovery is recorded as a decision — the Phase-6 file's variant remains green under its own convention but the new file follows the encoder's.

## Issues Encountered
- The FP4ModelTest.cpp vcxproj filter must be re-applied after every `cmake ..` reconfigure (known blocker, owner: milestone WS 04-02). Applied twice this plan (initial re-glob + post-diagnostic-cleanup re-glob).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All Phase 7 success criteria observably true: multi-container disjoint ranges + classic run (Criterion 1), structured MIXED end-to-end via real-encoder fixture (Criterion 2), clean malformed failures with no partial artifacts (Criterion 3).
- v3.0 inherits: probe-table pattern, region-relative container-building reference, family gate at 9 suites.

*Plan: 07-03*
*Completed: 2026-08-28*
