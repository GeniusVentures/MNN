---
phase: 4
slug: vulkan-decode-adaptive-quadtree-layout-mixed
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-25
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | MNNTestSuite (in-tree; static self-registration via `REGISTER_TEST`) |
| **Config file** | `test/CMakeLists.txt` — `GLOB_RECURSE` auto-discovers test files (no edit needed; configure-time, see RESEARCH.md Pitfall 4) |
| **Quick run command** | `run_test.out.exe "op/sgfp4/"` |
| **Full suite command** | `run_test.out.exe` (blocked by the pre-existing `FP4ModelTest.cpp` issue on a from-scratch build — use the Phase 1 temp-stub workaround per `deferred-items.md`) |
| **Estimated runtime** | ~10–60 seconds for the filtered sgfp4 suite (GPU init dominates) |

---

## Sampling Rate

- **After every task commit:** Run `run_test.out.exe "op/sgfp4/"`
- **After every plan wave:** Full `op/sgfp4/` suite + `op/fp4` and `op/vulkan/fp4_dequant_correctness` (E2M1 additivity regression guard, unchanged from Phase 3) + shader-embedding artifact grep on regenerated `AllShader.cpp` / `shaders/AllShader.h` / `VulkanShaderMap.cpp`
- **Before `/gsd-verify-work`:** Full `op/sgfp4/` suite must be green on Vulkan-capable hardware (graceful skip otherwise)
- **Max feedback latency:** ~120 seconds (includes incremental rebuild of the Vulkan backend)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | SGV2-15 | T-04-01 | GLSL MIXED branch only reads within the host-pre-validated container SSBO; every new bit-shift masked to `[0,31]` (RESEARCH.md Pitfall 6) — no shader-side bounds re-check added (D-06/D-10 keep shader defensive-only, correctness of the port is the operative control) | build | `python3 source/backend/vulkan/buffer/compiler/makeshader.py` (grep log for "error" — exits 0 even on glslang failure, Pitfall 2) then `run_test.out.exe "op/sgfp4/"` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | SGV2-15, SGV2-16 | — | N/A (regenerated-artifact commit, not new logic) | build + grep | grep `sgfp4_dequant` present in regenerated `AllShader.cpp`/`AllShader.h`/`VulkanShaderMap.cpp` | ❌ | ⬜ pending |
| 04-01-03 | 01 | 1 | SGV2-16 | — | Parity test degrades gracefully (skip + message) when no Vulkan device, per Phase 3 convention | integration (GPU) | `run_test.out.exe "op/sgfp4/vulkan_uniform_parity"` (all 14 fixtures, `mixed_asymmetric` skip line removed) | ❌ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Note: task IDs are provisional pre-planning; the planner's final task breakdown supersedes this map — keep the requirement/threat/verify-command columns aligned.*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.* Fixtures (`test/op/SGFP4DequantFixtures.h`, all 14 committed), CPU oracle (`SGFP4DequantUtils.hpp`), the Vulkan parity harness and its no-device skip convention, and the toolchain (glslang via WSL, `.build` already `MNN_VULKAN=ON`) all carry forward unchanged from Phase 3 — reconfirmed live during Phase 4 research. No new test files, fixtures, or framework setup needed.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.* Unlike Phase 3, no interactive/sudo provisioning step is required — the WSL glslang toolchain and Vulkan build configuration were already provisioned and are re-confirmed reachable this session.

---

## Coverage Gaps (parity alone does not close these — see RESEARCH.md "What parity alone does NOT cover")

1. **Bounded-loop depth (D-03's fixed 4-level descent).** The committed `mixed_asymmetric` fixture's tree depth is whatever Phase 2's encoder produced for it and is not guaranteed to exercise the full depth-4 worst case on every branch. Mitigated by relying on the already-verified CPU golden-traversal test (Phase-2 D-05) as the structural-correctness reference the GPU port must match, per D-09 — not by a dedicated GPU-side deep-tree fixture in this phase.
2. **Malformed/adversarial split-maps beyond host pre-validation.** Out of scope per Roadmap Note 3 and D-06 — the shader only ever receives containers that already passed `dequant_sgfp4_container_cpu`.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
