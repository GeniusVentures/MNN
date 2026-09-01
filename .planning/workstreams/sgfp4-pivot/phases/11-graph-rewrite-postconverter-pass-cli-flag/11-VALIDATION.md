---
phase: 11
slug: graph-rewrite-postconverter-pass-cli-flag
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-01
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `11-RESEARCH.md` §Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Standalone assert-macro executable (`TestSGFP4Converter.cpp` `CHECK` macro) + MNN test suite (`run_test.out`) |
| **Config file** | `tools/converter/CMakeLists.txt` (test target wiring, already present) |
| **Quick run command** | `TestSGFP4Converter.exe` (build target) |
| **Full suite command** | `run_test.out op/sgfp4` (13 suites) + `TestSGFP4Converter.exe` |
| **Estimated runtime** | ~60–120 seconds (build excluded) |

---

## Sampling Rate

- **After every task commit:** Run `TestSGFP4Converter.exe`; for pass/CLI tasks also `run_test.out op/sgfp4` quick subset touched by the change.
- **After every plan wave:** Run `run_test.out op/sgfp4` (13/13) + full `TestSGFP4Converter.exe` + `git status test/` clean (D-14).
- **Before `/gsd-verify-work`:** Full suite green + D-13 smoke executed and documented (nodes-present + decode + mutex behavior).
- **Max feedback latency:** ~120 seconds.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD-01-01 | 01 | 1 | SGV2-28 | T-11-05 / — | Encode failures propagate as pass failure (MNN_ERROR + `return false`), never encode garbage | unit | `TestSGFP4Converter.exe` | ❌ W0 | ⬜ pending |
| TBD-01-02 | 01 | 1 | SGV2-28 | — | Pass inserts SGFP4Dequant nodes, rewires `inputs[1]`, clears FP32 weights, stages `buffer`/`external=={}` | unit (synthetic NetT → `RunNetPass`) | `TestSGFP4Converter.exe` | ❌ W0 | ⬜ pending |
| TBD-01-03 | 01 | 1 | SGV2-28 | — | Light-tier floor (`<4096` elems or `dimI==1`) skipped | unit (synthetic tiny conv) | `TestSGFP4Converter.exe` | ❌ W0 | ⬜ pending |
| TBD-01-04 | 01 | 1 | SGV2-28 | — | Subgraph coverage (`subgraph->nodes` + `tensors` growth) | unit (synthetic NetT w/ subgraph) | `TestSGFP4Converter.exe` | ❌ W0 | ⬜ pending |
| TBD-01-05 | 01 | 1 | SGV2-28 | — | Idempotency (double `RunNetPass` / full `optimizeNet` round-trip → no doubling) | unit | `TestSGFP4Converter.exe` | ❌ W0 | ⬜ pending |
| TBD-01-06 | 01 | 1 | SGV2-28 | T-11-01 | External-spilled weight path (`external==3` reload incl. bias restore) | unit (synthetic spilled conv + temp bin) | `TestSGFP4Converter.exe` | ❌ W0 | ⬜ pending |
| TBD-02-01 | 02 | 2 | SGV2-29 | — | `--sgfp4` parses → `useSGFP4=true`; mutex rejects conflicting combos (`--fp16`/`--hqq`/`--weightQuantBits`) | CLI smoke (scripted) | `MNNConvert ... --sgfp4 [--fp16]` + output/exit assertions | ❌ new script/doc step | ⬜ pending |
| TBD-02-02 | 02 | 2 | SGV2-29+28 | — | Real corpus end-to-end (nodes present + decode) | CLI smoke (D-13, documented — corpus is test-time dependency) | see D-13 | ❌ new doc/script | ⬜ pending |
| TBD-03-01 | 03 | 3 | SGV2-30 | — | `WeightQuantAndCoding` skips `inputs>1` convs | unit (synthetic rewritten conv → hook no-op) | `TestSGFP4Converter.exe` | ❌ W0 | ⬜ pending |
| TBD-03-02 | 03 | 3 | SGV2-30/D-14 | — | Flag OFF → zero mutation; 13 `op/sgfp4` suites green, zero test-file edits | regression | `run_test.out op/sgfp4` + `git status test/` clean | ✅ suites exist | ⬜ pending |
| TBD-04-01 | 04 | 4 | W-2 | — | Arg-stage failCleanup removes stale artifacts | manual/scripted probe | stale-file + bad-arg run | ❌ tiny script step | ⬜ pending |
| TBD-04-02 | 04 | 4 | W-3 | — | Env-var root override works | manual (authoring-time scripts) | `SGFP4_GNUS_POC_ROOT=… python author_…` | ❌ manual | ⬜ pending |
| TBD-04-03 | 04 | 4 | W-1 | — | (Already fixed `1df51b7e`) suites still green | regression | `run_test.out op/sgfp4/classic_api` | ✅ exists | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `TestSGFP4Converter.cpp` pass-mechanics section (synthetic conv/subgraph/light-tier/spilled/idempotency cases) — covers SGV2-28/30 unit legs
- [ ] `RunNetPass` declaration in `tools/converter/include/PostConverter.hpp` (currently only `optimizeNet` is declared; the test needs the symbol)
- [ ] D-13 smoke script/doc (mutex + corpus run + assertions) — manual gate, corpus present (`W:\gnus\models\alexnet_Opset16.onnx`)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| D-13 corpus smoke | SGV2-29+28 | Corpus is a test-time dependency, not a committed always-on gate (provenance per Phase 10 D-01/D-02) | Run `MNNConvert -f ONNX --modelFile W:\gnus\models\alexnet_Opset16.onnx --MNNModel out.mnn --sgfp4`; assert output contains `SGFP4Dequant` nodes; decode via classic API |
| W-2 failCleanup probe | W-2 | Requires stale-file + bad-arg invocation against the CLI tool | Create stale output file, invoke `sgfp4_inject` with bad args, assert stale file removed |
| W-3 env-var override | W-3 | Authoring-time Python scripts, not CI | `SGFP4_GNUS_POC_ROOT=<alt> python author_structured_fixture.py …` and confirm the alternate root is used |

---

## Security Domain

`security_enforcement: true`, ASVS L1 (`.planning/config.json`). This phase processes trusted converter inputs (model files the user supplies for conversion) and adds no network, auth, or crypto surface.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (weakly) | Encoder's existing NaN/Inf + dims guards (`sgfp4_encode.cpp:768-780`); pass propagates `encode`'s empty-vector failure as a pass failure (MNN_ERROR + `return false`), never encodes garbage |
| V6 Cryptography | no | — |

### Threat Patterns (converter C++ pass)

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed model → OOB reads in pass | Tampering | Bounds from `param->external` sizes guarded by file-existence + `FileLoader` failure checks; `encode` input-count derived from `dimO*dimI` matches weight vector size (assert before call) |
| Huge dims → integer overflow | DoS | `encode` rejects dims > 65536 (`sgfp4_encode.hpp` contract); `dimO*dimI` computed in `size_t` |
| Temp-bin path collision (`.__convert_external_data.bin` in CWD) | Tampering | Pre-existing converter behavior, unchanged this phase (deleted at `writeFb.cpp:170`) |
