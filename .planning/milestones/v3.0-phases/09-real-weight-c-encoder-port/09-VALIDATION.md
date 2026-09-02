---
phase: 9
slug: real-weight-c-encoder-port
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-28
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | MNN custom `MNNTestSuite` (`run_test.out`), suites registered under `op/sgfp4/*`; Python oracle: gnus-poc `fp4_exporter.py` + `dequant_sgfp4_container_cpu` |
| **Config file** | none — suites registered in `test/op/*.cpp`, built via `test/CMakeLists.txt` |
| **Quick run command** | `.build/run_test.out op/sgfp4/encode` |
| **Full suite command** | `.build/run_test.out op/sgfp4` (full `run_test.out` still blocked by unrelated dead `test/op/FP4ModelTest.cpp` — pre-existing, out of scope) |
| **Estimated runtime** | ~10 seconds (encode suite), ~60 seconds (sgfp4 family) |

---

## Sampling Rate

- **After every task commit:** Run `.build/run_test.out op/sgfp4/encode`
- **After every plan wave:** Run `.build/run_test.out op/sgfp4`
- **Before `/gsd-verify-work`:** Full suite must be green + golden regenerability check (`python tools/fp4/author_real_shape_fixture.py` → byte-identical fixture header)
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

> Task IDs to be finalized by PLAN.md; rows below map requirements to the verification methods research identified.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD (encoder core) | TBD | TBD | SGV2-24 | V5 | Reject NaN/Inf weights; bound dims; size_t products | unit | `.build/run_test.out op/sgfp4/encode` | ❌ W0 | ⬜ pending |
| TBD (CPU decode parity) | TBD | TBD | SGV2-24 | — | N/A | unit (decode-vs-decode rtol 1e-4) | `.build/run_test.out op/sgfp4/encode` | ❌ W0 | ⬜ pending |
| TBD (Vulkan decode parity, D-08) | TBD | TBD | SGV2-24 | — | N/A | integration | `.build/run_test.out op/sgfp4/vulkan_encode_parity` | ❌ W0 | ⬜ pending |
| TBD (padded non-aligned shapes, SGV2-25) | TBD | TBD | SGV2-25 | — | N/A | unit/integration | `.build/run_test.out op/sgfp4/encode` | ❌ W0 — gated on Finding F1 decision | ⬜ pending |
| TBD (tiny <64 tensors) | TBD | TBD | SGV2-25 | — | N/A | unit | `.build/run_test.out op/sgfp4/encode` | ❌ W0 | ⬜ pending |
| TBD (threshold-flip robustness) | TBD | TBD | SGV2-24 | — | N/A | unit (constructed near-tie) | `.build/run_test.out op/sgfp4/encode` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tools/fp4/sgfp4_encode.hpp` / `sgfp4_encode.cpp` — encoder sources
- [ ] `tools/fp4/author_real_shape_fixture.py` + committed regenerable fixture header (D-05)
- [ ] `test/op/SGFP4EncodeTest.cpp` — CPU decode-parity + layout distribution + edge cases
- [ ] Vulkan encode-parity leg (extend `SGFP4VulkanDequantTest.cpp` pattern)
- [ ] CMake: `sgfp4_encode` lib target reachable from both `run_test.out` and `tools/fp4` (Research Q3)
- [ ] Padded-crop path resolution (Research Finding F1 / Q1) — gated on planner + user decision

*Existing `test/op/SGFP4TestUtil.hpp` covers shared container/sidecar helpers; no new framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Vulkan leg on machines without a Vulkan device | SGV2-24 (D-08) | Requires GPU-capable runtime | Run `.build/run_test.out op/sgfp4/vulkan_encode_parity` on a Vulkan-capable device (as exercised in Phases 3–4); CPU-oracle parity is the always-on primary gate |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
