---
status: complete
phase: 05-injection-core-artifact-construction-graph-splicing
source: [05-01-SUMMARY.md, 05-02-SUMMARY.md]
started: 2026-08-28T12:00:00.000Z
updated: 2026-08-28T12:30:00.000Z
---

## Current Test

[testing complete]

## Tests

### 1. Inject spike + version-gate suites green (no regression)
expected: Filtered `op/sgfp4/` family: all 5 suites pass, 0 failed — including new `op/sgfp4/inject` and `op/sgfp4/inject_v1_reject`; no regression to uniform_decode, mixed_decode, vulkan_uniform_parity.
result: pass
source: automated
note: Run by Claude 2026-08-28 — `.build\Release\run_test.out.exe op/sgfp4/` → 7/7 suites passed (passed:7, failed:0, blocked:0, skipped:0); family now also includes later-phase classic_api + classic_api_missing_sidecar; `inject` and `inject_v1_reject` both PASSED.

### 2. sgfp4_inject end-to-end on demo inputs → exit 0 + artifact pair
expected: `sgfp4_inject.out --model minimal_512.mnn --niche-dir <demo fp4 dir> --output out.mnn` exits 0 and emits `out.mnn` (~480 B) + `out.mnn.weight` (~132,368 B), with the sidecar laid alongside `<output>.weight`.
result: pass
source: automated
note: Run by Claude 2026-08-28 against pristine gnus-poc demo fp4 dir (specialists_mlx/demo/fp4) + current tmp/minimal_512.mnn → exit 0; wrote out.mnn (536 B) + out.mnn.weight (132,368 B). Sizes differ slightly from summary (480 B) because the current minimal_512.mnn names its weight Const2 — behavior equivalent.

### 3. In-tool verify: injected node decodes == oracle
expected: Tool's unconditional full-artifact `Module::load` verify prints per-node confirmation (e.g. `node 'weight' {512,512} offset=0 size=132368 verified (decode==oracle)`) — splice decodes match `dequant_sgfp4_container_cpu` at rtol 1e-4f.
result: pass
source: automated
note: Run by Claude 2026-08-28 — `node 'Const2' {512,512} offset=0 size=132368 verified (decode==oracle)`.

### 4. Negative paths hard-error with exit 1
expected: Corrupted magic byte with sha256 fixed to match → version-gate diagnostic, exit 1. Unmodified manifest + corrupted magic → sha256-mismatch diagnostic, exit 1. No artifact written as valid on failure paths.
result: pass
source: automated
note: Run by Claude 2026-08-28 — Case A (sha updated): version-gate diagnostic, exit 1. Case B (orig manifest): sha256-mismatch diagnostic, exit 1. Bonus: container basename != manifest fp4_binary.path basename also hard-errors (exit 1). No out*.mnn artifacts present after any failure path.

### 5. Shape pairing + sidecar merge diagnostics (ambiguous / multiple niches)
expected: Each `--niche-dir` resolves a unique `*.sgfp4` (0 or >1 → hard error listing candidates); pairing by exact `{dimO, dimI}` among non-input 2-D vars errs listing candidates on zero/multiple matches; sidecar offsets are 16-byte-aligned, monotonic, non-overlapping in the merged `.weight` stream.
result: pass
source: automated
note: Run by Claude 2026-08-28 — Case C: two *.sgfp4 in niche dir → `must contain exactly one *.sgfp4 file, found 2`, exit 1. Case D: manifest shape [128,64] → `expected exactly 1 weight match, found 0:`, exit 1 (source inspection confirms the candidate listing prints per-candidate on the >1 branch; zero-match has nothing to list). Sidecar: 132,368 B = 16×8,273 exactly — aligned, offset 0, monotonic, non-overlapping.

## Summary

total: 5
passed: 5
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none yet]
