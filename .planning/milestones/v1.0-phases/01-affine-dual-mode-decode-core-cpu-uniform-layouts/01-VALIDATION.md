---
phase: 1
slug: affine-dual-mode-decode-core-cpu-uniform-layouts
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-08-21
source: derived from 01-RESEARCH.md "Validation Architecture" section (nyquist_validation: true in .planning/config.json)
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Materialized from the `## Validation Architecture` section of `01-RESEARCH.md`; no new research performed.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | MNN's own `MNNTestSuite` (C++), run via `run_test.out` |
| **Config file** | none — tests self-register via `MNNTestSuiteRegister` |
| **Quick run command** | `cd build && ./run_test.out op/sgfp4` |
| **Full suite command** | `cd build && ./run_test.out` |
| **Estimated runtime** | quick run ~5–10s; full suite ~minutes (whole MNN op suite) |

The behavioral suite lives in `test/op/SGFP4DequantTest.cpp`, registered `MNNTestSuiteRegister(SGFP4DequantTest, "op/sgfp4/uniform_decode")` and gated by `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE` (parity with the sibling FP4 tests). Build with `-DMNN_BUILD_TEST=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON`.

---

## Sampling Rate

- **After every task commit:** Run `cd build && ./run_test.out op/sgfp4` (add `op/fp4` when touching the shared build, to confirm the E2M1 regression stays green — Success Criterion 5).
- **After every plan wave:** Run `cd build && ./run_test.out` (full suite green).
- **Before `/gsd-verify-work`:** Full suite must be green.
- **Max feedback latency:** ~10s (quick run).

**Wave-1 caveat:** the `op/sgfp4` behavioral test file does not exist until Plan 01-02 (wave 2) creates it. Wave-1 tasks (Plan 01-01) are therefore sampled by their **inline** compile / behavioral-smoke / full-build gates (see the Per-Task Verification Map). `./run_test.out op/sgfp4` becomes the sampling command once 01-02 Task 2 lands the suite.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | SGV2-05 (schema half) | — | Tail-append only; no macroblock/quadtree fields (SC#2) | build / structural | `sh schema/generate.sh` + grep gates on `MNN_generated.h` / `ShapeRegister.cpp` | ✅ inline | ⬜ pending |
| 01-01-02 | 01 | 1 | SGV2-01, SGV2-02, SGV2-03, SGV2-04 | T-01-01 / T-01-02 / T-01-03 | Malformed container rejected without OOB (ASVS V5); FP16 header unpacked via vendored `half` | unit (compile + behavioral smoke) | scratch compile+link+run smoke (FP16 unpack + malformed-input rejection) + grep gates | ✅ inline smoke (full behavioral → 01-02) | ⬜ pending |
| 01-01-03 | 01 | 1 | SGV2-05, SGV2-06 | T-01-04 | `FileLoader::valid()` + size-vs-file bound before read | integration (build) | grep gates on registration + full `cmake … && make` link | ✅ inline | ⬜ pending |
| 01-02-01 | 02 | 2 | SGV2-07 | T-0102-01 | Encoder byte-layout is exact inverse of decoder; round-trip within affine bound | unit | `python tools/fp4/encode_sgfp4.py --selftest` | ✅ encoder self-test | ⬜ pending |
| 01-02-02 | 02 | 2 | SGV2-01…07 | T-0102-01 / T-0102-02 | Malformed-input negative cases (bad magic/version, OOB offset, LAYOUT_MIXED) return false | unit + integration | `./run_test.out op/sgfp4` (this task **creates** the suite) | ❌ W0 → created here | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Requirements → Test Map (from 01-RESEARCH.md)

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SGV2-01 | Affine `w=S·c+bias`, both modes, ternary `11`→0 | unit | `./run_test.out op/sgfp4` | ❌ Wave 0 (`test/op/SGFP4DequantTest.cpp`) |
| SGV2-02 | FP16 (S, bias) unpack incl. 12-bit truncated bias | unit | `./run_test.out op/sgfp4` | ❌ Wave 0 |
| SGV2-03 | v2 framing parse (magic/ver/B/offset table) | unit | `./run_test.out op/sgfp4` | ❌ Wave 0 |
| SGV2-04 | All 5 uniform layouts, raster order, word counts | unit | `./run_test.out op/sgfp4` | ❌ Wave 0 |
| SGV2-05 | External `{magic,offset,size}` sidecar load | integration | `./run_test.out op/sgfp4` | ❌ Wave 0 |
| SGV2-06 | New CPU Execution produces float tensor | integration | `./run_test.out op/sgfp4` | ❌ Wave 0 |
| SGV2-07 | Encoder + round-trip both modes × all layouts | unit+integration | `./run_test.out op/sgfp4` | ❌ Wave 0 |
| SC#5 | E2M1 path unchanged | regression | `./run_test.out op/fp4 op/vulkan/fp4_dequant_correctness` | ✅ exists |

---

## Wave 0 Requirements

- [ ] `test/op/SGFP4DequantTest.cpp` — behavioral suite covering SGV2-01…07 (new file, created in **Plan 01-02, wave 2**; picked up automatically by `test/CMakeLists.txt` `GLOB_RECURSE` — no CMake edit needed).
- [ ] `test/op/SGFP4DequantFixtures.h` — encoder-generated cross-language fixtures (container bytes + dims + expected weights for both modes × all 5 uniform layouts + a B≢0 (mod 4) case), created in **Plan 01-02, wave 2**.
- No new test-framework install needed — `MNNTestSuite` already builds with `MNN_BUILD_TEST=ON`.

**Interim wave-1 coverage:** Plan 01-01 Task 2 carries an inline behavioral smoke gate (compile+link+run of a scratch driver asserting FP16 leaf-header unpack and malformed-container rejection) so the decode core gets *some* behavioral signal in wave 1, before the full round-trip suite arrives in 01-02.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.* (The end-of-phase human verify — `human_verify_mode: end-of-phase` — is a phase-level UAT gate, not a per-behavior manual validation.)

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (`test/op/SGFP4DequantTest.cpp`, `SGFP4DequantFixtures.h`)
- [x] No watch-mode flags
- [x] Feedback latency < ~10s (quick `op/sgfp4` run)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-08-21 (derived from `01-RESEARCH.md` Validation Architecture; `wave_0_complete` flips true when Plan 01-02 lands `test/op/SGFP4DequantTest.cpp`).
