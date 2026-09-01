---
phase: 12
slug: end-to-end-validation
status: draft
nyquist_compliant: false
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
| TBD-at-plan-time | — | — | SGV2-31 | — / — | N/A | e2e (script) | `tools\fp4\<e2e-script>.ps1 -Corpus <alexnet>` (CPU leg PASS) | ❌ W0 | ⬜ pending |
| TBD-at-plan-time | — | — | SGV2-32 | — / — | N/A | e2e (script) | same script (Vulkan leg PASS + backendType=7 assert) | ❌ W0 | ⬜ pending |
| TBD-at-plan-time | — | — | D-11 | Tampering (local) | converter exit ≠ 0 on pass failure, no "Converted Success!" | integration | script negative-path leg (forced pass failure → assert exit ≠ 0) | ❌ W0 | ⬜ pending |
| TBD-at-plan-time | — | — | D-12 | — / — | flag-off conversions byte-identical behavior/exit codes | regression | converter re-run flag-off + `run_test.out op/sgfp4` 13/13 + `TestSGFP4Converter.exe` | ✅ | ⬜ pending |
| TBD-at-plan-time | — | — | No-regression | — / — | all prior SGFP4 behavior intact | unit/suite | `run_test.out op/sgfp4`; `TestSGFP4Converter.exe` | ✅ | ⬜ pending |

*Task IDs and wave assignments are filled in by the planner from PLAN.md frontmatter.*

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tools/fp4/<e2e-script>.ps1` — the phase's central artifact (covers SGV2-31, SGV2-32, D-11 positive/negative legs)
- [ ] README section in `tools/fp4/README.md` documenting usage + hard Vulkan requirement

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

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
