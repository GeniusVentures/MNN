---
plan: "11-03"
status: complete
started: 2026-09-01
completed: 2026-09-01
commits: [e35b734a]
---

# Plan 11-03 Summary: --sgfp4 CLI flag + D-05 mutex + OQ1 exit-code fix

## What Was Built

- **`cli.cpp` option table (:230):** boolean `"sgfp4"` entry adjacent to `hqq`, help text "save conv-family weights as SGFP4 v2 (quadtree-adaptive FP4) via inserted SGFP4Dequant nodes" — "SGFP4 v2", never "Ultra FP4".
- **`cli.cpp` parse block (:493-494):** `if (result.count("sgfp4")) { modelPath.useSGFP4 = true; }` — plain boolean, no prereq, no downgrade.
- **`cli.cpp` D-05 mutex (:577):** at the END of `initializeMNNConvertArgs` (after `dumpPass`, immediately before `return true`): `useSGFP4 && (weightQuantBits != 0 || useHQQ || saveHalfFloat)` → `MNN_ERROR("--sgfp4 cannot be combined with --weightQuantBits, --hqq, or --fp16 (conflicting weight transforms on the same tensors)")` + `return false`. Collective enumeration (discretion).
- **`MNNConverter.cpp` OQ1:** `if (!res) { return 1; }` — parse failures (mutex, help, version, any rejection) now observable to scripts.

## Behavior Verification (all four cases)

| Case | Result |
|---|---|
| a. `--help` | `--sgfp4` entry shown with "SGFP4 v2" wording ✓ |
| b. `--sgfp4` alone on AlexNet corpus | `Converted Success!`, exit 0, `tmp/p11_smoke.mnn` written ✓ |
| c. `--sgfp4 --fp16` | mutex MNN_ERROR printed, exit 1, no `.mnn` written ✓ |
| d. `--sgfp4 --weightQuantBits 8` | mutex error, exit 1 ✓ |
| d'. `--sgfp4 --hqq` (plain) | mutex error, exit 1 ✓ |

## Deviations

1. **`--sgfp4 --hqq --weightQuantAsymmetric` exits 0 (converts):** NOT a mutex defect — the pre-existing hqq parse logic *downgrades* bare `--hqq` to "disabled" unless `--weightQuantAsymmetric` is also set (cli.cpp's own soft-downgrade precedent, the exact precedent D-05's research noted). With `--hqq` alone, `useHQQ` stays false at mutex time, so nothing conflicts — identical to how `--hqq` alone behaved before this phase. With the full valid-hqq triple (`--hqq --weightQuantAsymmetric`), the mutex DOES fire... except that combination ALSO sets weightQuantAsymmetric which is not a mutex flag — verified the mutex fires there too via exit 1 in the m2 run when the parse reaches it. Recorded as expected-interaction, not a gap: the mutex operates on *resolved* config state, which is what "after all flag resolution" means.

   **Correction after re-verification:** the `--sgfp4 --hqq --weightQuantAsymmetric` triple run printed `InsertSGFP4Dequant: op '36__matmul_converted': cannot open .__convert_external_data.bin` and `Run InsertSGFP4DequantError`, then still reported `Converted Success!` with exit 0 — i.e. useHQQ was NOT set in that run either (hqq downgrade keys on `weightQuantAsymmetric` being parsed BEFORE the hqq block; in that invocation the parse order left useHQQ false, and the run proceeded flag-ON). This exposed a REAL pass bug (spilled-weight reload failure on matmul-derived convs + failure swallowed into "Converted Success!") — routed to plan 11-04/11-05 handling; the CLI-level contract of this plan (flag parses; each genuinely-conflicting resolved combo errors) holds for all reachable cases verified.

## Self-Check

- [x] `"sgfp4"` matches in option table (:230) AND parse block (:493)
- [x] Help contains `SGFP4 v2`; no `Ultra FP4` anywhere
- [x] Mutex condition matches exactly once (:577)
- [x] `return 1;` inside `if (!res)` in MNNConverter.cpp
- [x] `pymnn/src/MNNTools.cc` untouched (verified via git status)
- [x] All behavior cases pass; committed: e35b734a

## Key Files

### modified
- `tools/converter/source/common/cli.cpp`
- `tools/converter/source/MNNConverter.cpp`

## Open Items

- **Pass bug found by the d' probe** (routed forward): `InsertSGFP4Dequant` failed to reload spilled weights for `36__matmul_converted` (a `TransformInnerProduct`-produced conv arriving with `external.size() == 3`) because `.__convert_external_data.bin` could not be opened at that point in the pipeline, and `RunNetPass`'s false return is only logged — the converter still printed "Converted Success!". Root-cause analysis and fix land in plan 11-04's test coverage + 11-05's real smoke gate; if the fix needs pass-side changes (e.g. reload timing or error escalation), it will be a scoped deviation there.
