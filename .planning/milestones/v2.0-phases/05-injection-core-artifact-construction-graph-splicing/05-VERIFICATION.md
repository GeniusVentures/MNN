---
phase: 05-injection-core-artifact-construction-graph-splicing
verified: 2026-08-28T00:00:00Z
status: passed
score: 4/4 roadmap truths verified
overrides_applied: 0
re_verification:
  previous_status: none
  note: "Consolidated verification — generated retroactively from 05-UAT.md (5/5 automated pass, 2026-08-28), 05-01/05-02 SUMMARYs, and the v2.0 milestone audit's integration-checker code-level findings. Mirrors the v1.0 Phase 2 doc-debt precedent."
human_verification: []
evidence_sources:
  - 05-UAT.md (/gsd-verify-work 5, 2026-08-28, 5/5 passed)
  - 05-01-SUMMARY.md / 05-02-SUMMARY.md (commits 67fc8947, 3b4c43b4)
  - v2.0-MILESTONE-AUDIT.md integration map (gsd-integration-checker, 2026-08-28)
---

# Phase 5: Injection Core — Artifact Construction & Graph Splicing — Verification Report

**Phase Goal (ROADMAP §Phase 5):** Given a normally-converted `.mnn` and one or more SGFP4 v2 container files, the tool produces a new `.mnn` + external sidecar in which target weight tensors are replaced by `OpType_SGFP4Dequant` nodes — correct at the Express/`Module::load` level first.
**Verified:** 2026-08-28 (consolidated from UAT + summaries + integration audit)
**Status:** passed
**Re-verification:** N/A — initial VERIFICATION.md, written retrospectively after `/gsd-verify-work 5` produced 05-UAT.md

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Tool accepts a normally-converted `.mnn` + SGFP4 v2 containers (gnus-poc `fp4_exporter.py --adaptive` output), rejecting legacy v1 via version check rather than silently misdecoding | ✓ VERIFIED | Byte-level gate `sgfp4_is_v2_container` (`include/MNN/SGFP4DequantUtils.hpp:110-114`, magic + version-byte, null/min-size guards) — never consults manifest `fp4_binary.format`. UAT test 4 PASS: corrupted-magic-with-fixed-sha256 → version-gate diagnostic, exit 1; unmodified-manifest + corrupted magic → sha256-mismatch diagnostic, exit 1. Suite `op/sgfp4/inject_v1_reject` PASS (bad-magic, bad-version, v1-layout magic-less buffer, nullptr, 15-byte truncation all rejected — `test/op/SGFP4InjectTest.cpp:313-380`). Integration checker: gate→suite wiring confirmed; malformed-suite probes `bad_magic`/`version_one` also exercise it |
| 2 | Per target weight tensor: `OpType_SGFP4Dequant` + `SGFP4DequantParamT{magic=kSGFP4Magic, external{offset,size}, dims{dimO,dimI}}` with `op->externalPath` set literally on the op (not one of the `createExecutionWithExternal` auto-rewrite types) | ✓ VERIFIED | `makeDequantOp` (`tools/fp4/sgfp4_inject_core.hpp:259-272`) fills all fields incl. literal `op->externalPath`; schema wiring confirmed `schema/default/CaffeOp.fbs:118` → generated `CaffeOp_generated.h:1440+`. Proven first in the 05-01 spike (`SGFP4InjectTest.cpp:180-206`, Pitfall-2 comment) then reproduced in the tool. Structurally asserted downstream: `multi_tensor` suite asserts exactly-N ops with valid `SGFP4DequantParam` (`SGFP4MultiTensorTest.cpp:480-500`). UAT test 3 exercises the resulting op end-to-end |
| 3 | Byte ranges in single merged sidecar non-overlapping, matching each op's `{offset,size}`; downstream consumers read the new node's output instead of the original constant | ✓ VERIFIED | Sidecar merge: single `<output>.weight` stream, offsets via `sgfp4_align16` cursor, monotonic non-overlapping (core `:338-371`); offsets computed before op construction so ops carry final `{offset,size}`. UAT test 5 PASS: 132,368 B = 16×8,273 exactly — aligned/monotonic/non-overlapping. Rewiring: `Variable::replace(weightVar, dequantVar)`; 05-01 spike asserted graph structure post-save — exactly 1 `OpType_SGFP4Dequant` expr, 0 `VARP::CONSTANT` weight exprs (original Const dead-dropped). Later phases re-prove at strength: `multi_tensor` disjoint 16-aligned ranges [0,140240)/[140240,156800) with per-range sidecar `memcmp` byte-identity |
| 4 | Serialized via `Variable::save(vars, fileName)` direct-to-file overload; reloads via Express `Module::load` (with `rtmgr->setExternalFile()` before load) decoding through the existing CPU Execution within oracle tolerance | ✓ VERIFIED | Direct-to-file save at core `:366` (no in-memory variant). UAT tests 2+3 PASS: exit-0 E2E on the pristine gnus-poc demo niche dir + `minimal_512.mnn` → `out.mnn` + `out.mnn.weight` pair emitted; in-tool unconditional verify prints `node … {512,512} offset=0 size=132368 verified (decode==oracle)` (rtol 1e-4f vs `dequant_sgfp4_container_cpu`). 05-01 suite independently reloaded the saved artifact via `Module::load` + pre-load `setExternalFile` and passed oracle parity (`SGFP4InjectTest.cpp:216-275`) |

**Score:** 4/4 truths verified

### Plan-Level Must-Haves

| Plan | Must-Have | Status | Evidence |
|------|-----------|--------|----------|
| 05-01 | A1 graph-surgery recipe proven at runtime level before tool exists | ✓ VERIFIED | Spike suite `op/sgfp4/inject` PASS — build/save/splice/`Variable::replace`/re-save/reload/run with oracle parity + graph-structure assertion (1 dequant expr, 0 CONSTANT exprs). Assumption A1 confirmed in SUMMARY |
| 05-01 | Byte-level v2 version-gate primitive `sgfp4_is_v2_container` (SGINJ-01) | ✓ VERIFIED | Header-only inline in `SGFP4DequantUtils.hpp`; `op/sgfp4/inject_v1_reject` PASS with 5 rejection cases |
| 05-02 | Standalone tool E2E on real demo inputs | ✓ VERIFIED | UAT test 2: exit 0, artifact pair; test 3: decode==oracle; tests 4-5: 4 negative-path classes all exit 1 with diagnostics (incl. bonus basename-mismatch hard-error, 0/>1 `*.sgfp4` in niche dir) |
| 05-02 | sha256 KAT + manifest-integrity gate (D-03) | ✓ VERIFIED | 05-02 SUMMARY: empty-string KAT digest matches FIPS 180-4 vector; demo-container digest matches Python `hashlib.sha256`; mismatch → hard error (UAT test 4 case B) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Inject spike + reject suites | `run_test.out op/sgfp4/inject`, `op/sgfp4/inject_v1_reject` | PASS (18.1 ms / 0.05 ms, 05-01 SUMMARY) | ✓ PASS |
| Family regression incl. Phase 5 suites | `run_test.out op/sgfp4/` | passed:7 failed:0 (UAT test 1, 2026-08-28; passed:9 in Phase 7 verify) | ✓ PASS |
| Tool E2E (demo inputs) | `sgfp4_inject.out --model minimal_512.mnn --niche-dir <demo fp4 dir> --output out.mnn` | exit 0; out.mnn (536 B) + out.mnn.weight (132,368 B) — weight named `Const2` in current model, size delta vs 480 B summary is naming-only (UAT test 2 note) | ✓ PASS |
| In-tool decode==oracle | (stdout of E2E above) | `node 'Const2' {512,512} offset=0 size=132368 verified (decode==oracle)` | ✓ PASS |
| Negative paths | corrupted-magic ×2 manifests, 2-`*.sgfp4` niche dir, shape-[128,64] mismatch | All exit 1 with diagnostics; no artifacts written | ✓ PASS |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `test/op/SGFP4InjectTest.cpp` | Suites `op/sgfp4/inject` + `op/sgfp4/inject_v1_reject`; spike recipe with literal externalPath | ✓ VERIFIED | Registration `:309`, `:387`; recipe `:180-206`; reload+parity `:216-275` — integration checker W10 confirms recipe→tool correspondence |
| `tools/fp4/sgfp4_inject.cpp` (05-02; now thin shim after 06-01 refactor) | CLI `--model/--niche-dir/--output` → out.mnn + out.mnn.weight | ✓ VERIFIED | Built and exercised at UAT; 06-01 refactor preserved the contract byte-identically (see 06-VERIFICATION) |
| `tools/fp4/sha256.hpp` | Self-contained SHA-256, KAT-verified | ✓ VERIFIED | Empty-string KAT + multi-block real-container digest match (05-02 SUMMARY) |
| `tools/fp4/CMakeLists.txt` + root option | `MNN_BUILD_SGFP4_TOOLS` option, `sgfp4_inject.out` target | ✓ VERIFIED | `CMakeLists.txt:50`, `tools/fp4/CMakeLists.txt:3-5`; target built for UAT runs |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| `SGFP4InjectTest.cpp` | `SGFP4DequantUtils.hpp` | `sgfp4_is_v2_container` + `dequant_sgfp4_container_cpu` oracle | ✓ WIRED |
| Spike recipe (05-01) | Tool surgery loop (05-02) | `makeDequantOp` + `Variable::replace` correspondence | ✓ WIRED (integration checker W10) |
| Tool op construction | Schema → CPU op | `CaffeOp.fbs:118` → `CPUSGFP4Dequant.cpp:44-88` (`USE_EXTERNAL_DATA`, literal `mOp->externalPath()`) | ✓ WIRED (integration checker W6/W7) |
| Tool serialization | Express reload | `Variable::save` direct-to-file → `Module::load` + pre-load `setExternalFile` | ✓ WIRED (UAT tests 2-3) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | Zero TODO/FIXME/XXX/HACK/placeholder/stub hits in the phase's files (integration checker I-2 scan) | — | — |

### Deviations (informational)

- Build unblock: pre-existing out-of-scope `test/op/FP4ModelTest.cpp` blocker bypassed by filtering the untracked generated `.vcxproj` only — no tracked file touched (05-01 SUMMARY; tracked in STATE.md, owned by the `milestone` workstream).
- Runtime is Windows/MSVC rather than the plan's MSYS2 shell notes — identical exit-code/PASS semantics.
- `windows.h` min/max clash guarded via `NOMINMAX`; directory listing via `FindFirstFileA`/`dirent.h` (C++11, no `<filesystem>`).

### Human Verification Required

None. All success criteria are machine-verified (suites, E2E exit codes, oracle parity, diagnostics) — evidence re-executed during UAT (2026-08-28) rather than SUMMARY claims.

### Gaps Summary

No gaps. All four roadmap success criteria are verified with executable evidence: v1 rejection via byte-level gate (suite + tool negatives), schema-conformant `SGFP4Dequant` ops with literal `externalPath`, merged non-overlapping 16-aligned sidecar with consumer rewiring (structural + byte-identity proof), and direct-to-file serialization reloading through `Module::load` within oracle tolerance (rtol 1e-4). Runtime evidence chain: 05-01/05-02 suites → UAT re-run (5/5, 2026-08-28) → Phases 6/7 family regressions (7/7, 9/9) → integration-checker code audit (2026-08-28).

---

_Verified: 2026-08-28 (consolidated)_
_Verifier: the agent (gsd-verifier evidence consolidated during /gsd-audit-milestone)_
