---
plan: "11-02"
status: complete
started: 2026-09-01
completed: 2026-09-01
commits: [19add75a]
---

# Plan 11-02 Summary: W-1/W-2/W-3 audit-debt retirement

## What Was Built

**Task 1 — W-2 failCleanup hoist (`sgfp4_inject_core.hpp`):**
- Lambda moved ABOVE the arg parse loop; captures only `&outputPath`; body guards on `if (!outputPath.empty())` and computes `outputPath + ".weight"` internally (no `sidecarPath` capture — never removes a bare `.weight` in CWD while `--output` is unparsed).
- Both arg-validation returns (`usage(); failCleanup(); return 1;` on unknown arg and on missing required args) now clean up.
- All later failure sites unchanged (lambda still in scope; `sidecarPath` still derived for the success path).
- Comment block updated to note the W-2 hoist covers ALL failure paths.
- `sgfp4_inject.out` rebuilt clean under MSVC (no C2065 — hoist correctness proof).
- **Probe:** committed as `tools/fp4/w2_failcleanup_probe.ps1`; run → `W-2 probe: PASS (arg-stage failure removed both stale artifacts)`, exit 1 observed, both stale files removed.

**Task 2 — W-3 env-var root override:**
- `author_structured_fixture.py` + `author_real_shape_fixture.py`: `GNUST_POC_ROOT = Path(os.environ.get("SGFP4_GNUS_POC_ROOT", "W:/gnus/GeniusCognitiveSystem/GNUS-NEO-SWARM/gnus-poc"))` (+ `import os` added to both).
- `validate_real_weights.py`: `DEFAULT_GNUS_POC_ROOT` now env-derived with the same literal fallback; `--gnus-poc-root` stays authoritative when passed explicitly.
- All three parse (`ast.parse`) clean; env resolution demonstrated both ways (override `D:/override/root` honored; unset → original default; real gnus-poc root via env imports `FP4Exporter` cleanly).

**Task 3 — W-1 verify-and-close (no code edit):**
- `run_test.out op/sgfp4/classic_api` → 2/2 passed, exit 0.
- Source inspection: `SGFP4ClassicAPITest.cpp:171` routes through `sgfp4_test::buildContainerUniform64`; the :81-84 comment documents the swap from the former local ABSOLUTE-offset builder. No local absolute-offset builder remains.
- **W-1 retired by commit `1df51b7e`** ("[Test:Refact] dedup SGFP4 test helpers into SGFP4TestUtil.hpp (D-10)" — Phase 8 plan 08-02 D-10 pull-forward; stat shows the 151-line reduction in SGFP4ClassicAPITest.cpp). Verified green this phase; no edit made.
- `git status --porcelain test/op/` clean.

## Deviations

1. **Probe argument order:** first probe draft passed `--bogus-flag` before `--output` — at that point `outputPath` is legitimately still empty and cleanup is a no-op BY DESIGN (nothing has claimed an output path yet). Fixed the probe to pass `--output` first, then the bad flag — the W-2 contract is "once --output is known, ANY failure path removes stale artifacts at that path." Probe comment documents this.
2. **`pwsh` unavailable** on this machine; the committed probe script runs under Windows PowerShell (`powershell -File`). The script itself is version-agnostic (no pwsh-only syntax).

## Self-Check

- [x] `failCleanup` definition line < arg parse loop line; both validation returns have adjacent `failCleanup();`
- [x] Lambda body: `if (!outputPath.empty())`, internal `.weight` computation
- [x] W-2 probe PASS transcript captured (exit 1, both files gone)
- [x] `sgfp4_inject.out` builds clean under MSVC
- [x] All 3 scripts contain `os.environ.get("SGFP4_GNUS_POC_ROOT"` with W:/ fallback
- [x] All 3 scripts `ast.parse` clean
- [x] Env override + fallback + real-root import demonstrated
- [x] `run_test.out op/sgfp4/classic_api` 2/2 green; `test/op/` untouched
- [x] Committed: 19add75a

## Key Files

### created
- `tools/fp4/w2_failcleanup_probe.ps1`

### modified
- `tools/fp4/sgfp4_inject_core.hpp`
- `tools/fp4/author_structured_fixture.py`
- `tools/fp4/author_real_shape_fixture.py`
- `tools/fp4/validate_real_weights.py`

## Open Items

None — all three audit items closed with evidence.
