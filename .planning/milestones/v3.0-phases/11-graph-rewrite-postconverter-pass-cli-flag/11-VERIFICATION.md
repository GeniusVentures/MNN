---
status: passed
phase: 11-graph-rewrite-postconverter-pass-cli-flag
verified: 2026-09-01
requirements: [SGV2-28, SGV2-29, SGV2-30]
score: 6/6
---

# Phase 11 Verification Report

**Method**: goal-backward check of every ROADMAP success criterion against the committed codebase and executed evidence (inline verification per Copilot sequential mode; no separate verifier agent).

## Success Criteria vs. Evidence

### 1. Pass shipped & invoked (SGV2-28) — VERIFIED
- Registered pass `PostConverterRegister<InsertSGFP4Dequant> __l("InsertSGFP4Dequant")` (`InsertSGFP4Dequant.cpp:294`); invoked `RunNetPass({"InsertSGFP4Dequant", "ReIndexTensor"}, newNet)` (`PostConverter.cpp:393`) — before ReIndexTensor per KEY Q2.
- 4 conv types gated (`isSgfp4TargetOpType`); oplists + subgraphs walked (D-03, saveExternalData walk shape).
- Light-tier floor `elements < 4096 || dimI == 1` (line 219).
- `kSGFP4ConverterEncodeConfig` alias with validated-delta documentation (D-08).
- Buffer contract (Phase 8 D-11): buffer populated, `external == {}`, empty `externalPath` — asserted in PHASE C T1 and T6/T6b.
- In-param + spilled weights handled (flush + reload + bias restore) — PHASE C T7; **deviation noted**: reload via `std::ifstream`, not FileLoader (MSVC `fopen_s` exclusive sharing vs. the converter's own open ofstream; probe-verified in-test).
- Idempotency: `inputIndexes.size() == 1` + `quanParameter == nullptr` condition — PHASE C T9 and the real pipeline's second `no change` dump.
- Encode failure transactional — PHASE C T10 (NaN/Inf variants: pass false, conv byte-identical, no node).

### 2. CLI flag + mutex (SGV2-29) — VERIFIED
- `--sgfp4` boolean option ("SGFP4 v2" wording, no "Ultra FP4") → `modelPath.useSGFP4 = true` (`cli.cpp:230/:493`).
- Mutex `useSGFP4 && (weightQuantBits != 0 || useHQQ || saveHalfFloat)` → MNN_ERROR + return false (`cli.cpp:577`); exit 1 via OQ1 fix (`MNNConverter.cpp` `return 1;`).
- Behavior-verified on the corpus: all three conflicting combos exit 1 with no output; `--sgfp4` alone converts.

### 3. Skip-guard (SGV2-30) — VERIFIED
- `WeightQuantAndCoding.cpp:95`: `inputIndexes.size() > 1` early return, positioned between the quanParameter return (:87) and the `weightQuantBits == 0` sparse path (:142) — line-number-verified.
- PHASE C T8 drives the hook directly on a rewritten conv: no-op.

### 4. Tests + smoke (D-12/D-13) — VERIFIED
- PHASE C (12 tests incl. T6b) green: insertion, rewiring, clearing, buffer contract, light-tier (both variants), subgraph w/ tensor-stability, spill reload, flag-off zero-mutation, skip-guard no-op, idempotency, encode-failure propagation, round-trip survival, 4-D dims.
- Real corpus smoke: `ops 74 -> 82` (K=8, matching the Phase 10 tier table), artifact loads and `runSession` returns NO_ERROR via the classic API.
- OQ3 fallback **not triggered** (no node loss; primary D-01 placement confirmed).
- **Recorded deviation (D-13, documented in 11-05-SUMMARY)**: 4-D conv-weight dims `{O, I, kH, kW}` written by the pass + decoder generalization (`dims[0]` × product(rest)) + NCHW format for rank ≥ 3 in `ShapeSGFP4Dequant`. Root cause: 2-D weight tensors crash conv shape inference/`ConvolutionTiledExecutorMultiInput` (the v2.0 architecture research had flagged exactly this). Backward compatibility proven: all 13 suites (injection-tool 2-D artifacts) pass unchanged.

### 5. No-regression (D-14) — VERIFIED
- Flag OFF: corpus conversion succeeds, pass never mutates (PHASE C T3).
- `run_test.out op/sgfp4` 13/13; `TestSGFP4Converter.exe` green.
- `git status --porcelain test/` empty — zero test-file edits across the phase.

### 6. Tech debt retired (D-09/D-10/D-11) — VERIFIED
- W-1: verified-and-closed at `1df51b7e` (classic_api routes through `sgfp4_test::buildContainerUniform64`; 2/2 green this phase; no edit).
- W-2: failCleanup hoisted with empty-outputPath guard; committed probe `w2_failcleanup_probe.ps1` PASSES (arg-stage failure removes both stale artifacts).
- W-3: `SGFP4_GNUS_POC_ROOT` env override with original fallback in all three scripts; parse + resolution verified both ways.

## Requirement Traceability

| Req | Status | Evidence |
|---|---|---|
| SGV2-28 | Met | Criteria 1, 4; commits dc6b1d62, ee24e0e5, f2b00b08 |
| SGV2-29 | Met | Criterion 2; commit e35b734a |
| SGV2-30 | Met | Criteria 3, 6; commits dc6b1d62, 19add75a |

## Notes & Carried Items

- The 4-D dims deviation is the significant design learning of this phase: **the injection tool's 2-D dims convention is only valid for MatMul consumers** — any conv consumer requires 4-D conv-weight geometry. Phase 12 (E2E CPU/Vulkan + accuracy gates) inherits a proven-loadable artifact and should fold the output-accuracy comparison on top.
- `RunNetPass` failure is log-only by design; the converter still prints `Converted Success!` when the pass returns false (transactional skip keeps the artifact safe, but the success message can overstate). Flagged for Phase 12 consideration (error escalation), not a phase-11 gate.
- Scratch probes (`tmp/p13_decode_probe.*`, `tmp/t4_diag.*`) are untracked/disposable.

**Verdict: PASSED — 6/6 must-haves verified, all requirements met, deviations documented not hidden.**
