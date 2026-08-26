---
phase: 3
slug: vulkan-decode-uniform-layouts
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-24
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | MNNTestSuite (in-tree; static self-registration via `REGISTER_TEST`) |
| **Config file** | `test/CMakeLists.txt` — `GLOB_RECURSE` auto-discovers new test files (no edit needed) |
| **Quick run command** | `./run_test.out "op/sgfp4/"` |
| **Full suite command** | `./run_test.out` (note: full from-scratch build needs the `FP4ModelTest.cpp` temp-stub workaround — see Phase 1 `deferred-items.md`) |
| **Estimated runtime** | ~10–60 seconds for the filtered sgfp4 suite (GPU init dominates) |

---

## Sampling Rate

- **After every task commit:** Run `./run_test.out "op/sgfp4/"`
- **After every plan wave:** Full sgfp4 suite + build-integrity greps on regenerated shader artifacts (`AllShader.cpp`, `shaders/AllShader.h`, `VulkanShaderMap.cpp`)
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~120 seconds (includes incremental rebuild of the Vulkan backend)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 0 | SGV2-12 | — | N/A (toolchain provisioning) | build | `glslangValidator --version` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | SGV2-12 | T-03-01 | Shader reads only the host-validated container SSBO; no bounds re-check needed (host gates dispatch) | build + grep | `python3 source/backend/vulkan/buffer/compiler/makeshader.py` then grep `sgfp4_dequant` in regenerated artifacts | ❌ | ⬜ pending |
| 03-01-03 | 01 | 1 | SGV2-13 | T-03-02 | File-size probe BEFORE upload-buffer allocation (untrusted-sidecar DoS posture, mirrors Phase 1 T-01-04) | unit/build | `./run_test.out "op/sgfp4/"` | ❌ | ⬜ pending |
| 03-02-01 | 02 | 2 | SGV2-14 | — | Parity test degrades gracefully when no Vulkan device | integration (GPU) | `./run_test.out "op/sgfp4/vulkan"` | ❌ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Note: task IDs are provisional pre-planning; the planner's final task breakdown supersedes this map — keep the requirement/threat/verify-command columns aligned.*

---

## Wave 0 Requirements

- [ ] Toolchain provisioning: `glslangValidator` reachable from the environment used to run `makeshader.py` (WSL `sudo apt-get install glslang-tools` — interactive sudo — or Windows Vulkan SDK + POSIX `find` via Git Bash). Verify with `glslangValidator --version`.
- [ ] Build reconfiguration: `.build` (or a fresh build dir) configured with `-DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF` (current cache has `MNN_VULKAN=OFF`), then smoke the existing `VulkanFP4DequantTest` to prove the Vulkan device path works end-to-end on this machine.
- [ ] `test/op/SGFP4VulkanDequantTest.cpp` (or sibling) stub registered under `op/sgfp4/...` — auto-globbed by `test/CMakeLists.txt`.

*Existing infrastructure otherwise covers all phase requirements (fixtures, CPU reference, test harness patterns are all committed from Phases 1–2).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| WSL sudo install of `glslang-tools` | SGV2-12 | sudo is interactive (password) | Run `sudo apt-get install -y glslang-tools` in WSL once, confirm `glslangValidator --version` |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
