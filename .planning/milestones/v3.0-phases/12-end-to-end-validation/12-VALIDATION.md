---
phase: 12
slug: end-to-end-validation
status: planned
nyquist_compliant: true
wave_0_complete: false
created: 2026-09-01
---

# Phase 12 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | MNN custom test suite (`run_test.out`, `MNNTestSuiteRegister`) + `TestSGFP4Converter.exe` + the phase's committed E2E script |
| **Config file** | `test/CMakeLists.txt` (existing; no changes expected) |
| **Quick run command** | `.build\Release\run_test.out.exe op/sgfp4` (13 suites, filtered — full build broken by unrelated `FP4ModelTest.cpp`) |
| **Full suite command** | `.build\Release\run_test.out.exe op/sgfp4` + `.build\Release\TestSGFP4Converter.exe` + `tools\fp4\<e2e-script>.ps1 -Corpus W:\gnus\models\alexnet_Opset16.onnx` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `.build\Release\run_test.out.exe op/sgfp4`
- **After every plan wave:** quick gate + `TestSGFP4Converter.exe` + one full E2E script run (once the script exists)
- **Before `/gsd-verify-work`:** Full E2E script PASS on both backends + D-11 negative leg + all regression suites green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 12-01-T1 | 12-01 | 1 | D-11 (library leg) | T-12-01 / T-12-02 | `RunNetPass` returns false on missing pass or `onExecute` false; `optimizeNetImpl` returns nullptr + MNN_ERROR when `useSGFP4` && failed | unit/integration | `cmake --build .build --config Release --target TestSGFP4Converter MNNConvert; if ($LASTEXITCODE -ne 0) { exit 1 }; & .build\Release\TestSGFP4Converter.exe; if ($LASTEXITCODE -ne 0) { exit 1 }; & .build\Release\run_test.out.exe op/sgfp4; if ($LASTEXITCODE -ne 0) { exit 1 }` | ✅ (existing suites) | ⬜ pending |
| 12-01-T2 | 12-01 | 1 | D-11 (CLI leg) + D-12 | T-12-01 / — | `--sgfp4` + corrupt model → exit ≠ 0, no "Converted Success!"; flag-off exit codes byte-identical (0) | integration | `cmake --build .build --config Release --target MNNConvert; if ($LASTEXITCODE -ne 0) { exit 1 }` then the three MNNConvert invocations (positive `--sgfp4` corpus → $LASTEXITCODE 0; negative `--sgfp4` corrupt → non-zero; flag-off corrupt → 0) per 12-01 Task 2 `<verify>`, plus `run_test.out op/sgfp4` + `TestSGFP4Converter.exe` with $LASTEXITCODE checks | ✅ | ⬜ pending |
| 12-02-T1 | 12-02 | 2 | SGV2-31 + SGV2-32 (script + measure) | T-12-03 / T-12-05 | vulkaninfo pre-check → exit 2 when no device (no SKIP); `backendType is 7` assert on Vulkan leg; Test-Path pre-checks exit 2 | e2e (script, -MeasureOnly) | `pwsh -File tools/fp4/e2e_validation.ps1 -Corpus W:\gnus\models\alexnet_Opset16.onnx -WorkRoot tmp/p12_measure -MeasureOnly; if ($LASTEXITCODE -ne 0) { exit 1 }` (expect exit 0, measured max-abs/max-rel for both backends printed) | ❌ created in-task (self-providing Wave 0) | ⬜ pending |
| 12-02-T2 | 12-02 | 2 | SGV2-31 + SGV2-32 (locked gate) + D-11 consumer | T-12-04 / T-12-01 | full-gate run: CPU + Vulkan PASS vs locked tolerances; D-11 negative leg inside same invocation; temp-dir cleanup on pass | e2e (script, full gate) | `pwsh -File tools/fp4/e2e_validation.ps1 -Corpus W:\gnus\models\alexnet_Opset16.onnx; if ($LASTEXITCODE -ne 0) { exit 1 }` (expect exit 0, `PASS: cpu` + `PASS: vulkan`) then `run_test.out op/sgfp4` + `TestSGFP4Converter.exe` with $LASTEXITCODE checks | ✅ after 12-02-T1 | ⬜ pending |

*Task IDs and waves back-filled from 12-01-PLAN.md / 12-02-PLAN.md frontmatter (2026-09-01 revision).*

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tools/fp4/e2e_validation.ps1` — the phase's central artifact (covers SGV2-31, SGV2-32, D-11 positive/negative legs) — **created by 12-02-T1 itself (self-providing); each task's `<verify>` only runs files its own task (or an earlier wave/task) produces**
- [ ] README section in `tools/fp4/README.md` documenting usage + hard Vulkan requirement — created by 12-02-T2

*Existing infrastructure covers everything else — no framework installs or fixtures needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Vulkan device availability (RTX 4070 Ti SUPER, Vulkan 1.4.321) | SGV2-32 | Hardware-dependent; probe-verified this session but can drift across driver updates | Run `vulkaninfo --summary` and confirm a Vulkan 1.x device enumerates before the Vulkan E2E leg |

---

## Security Domain

Local, offline validation tooling; no network, no secrets, no user input beyond a file path. Applicable ASVS categories: none materially (V5 Input Validation — the script validates `-Corpus` path existence and converter exit codes before proceeding; that is the entire surface).

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed/corrupt corpus path fed to tools | Tampering (local) | Script `Test-Path` pre-check + exit-nonzero-on-any-tool-failure (D-11 chain enforces converter honesty) |
| Temp-dir leftovers with model data | Information disclosure (local dev box) | Script cleans its temp dirs (`Remove-Item -Recurse -Force`, W-2 probe precedent) |

Note: the SGFP4 op's own input hardening (DoS bounds, magic/version gates, host pre-validation) shipped in Phases 1/3/8 and is regression-gated — not re-litigated here.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (e2e script is self-provided by 12-02-T1)
- [x] No watch-mode flags
- [x] Feedback latency < 120s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** planner (back-filled from PLAN frontmatter, 2026-09-01 revision)
