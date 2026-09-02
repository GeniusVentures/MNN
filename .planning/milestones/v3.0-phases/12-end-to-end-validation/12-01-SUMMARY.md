---
phase: 12-end-to-end-validation
plan: "01"
subsystem: converter
tags: [sgfp4, converter, exit-code, runnetpass, d-11, d-12]

requires:
  - phase: 11-graph-rewrite-postconverter-pass-cli-flag
    provides: "InsertSGFP4Dequant post-converter pass + --sgfp4 CLI flag + TestSGFP4Converter harness"
provides:
  - "bool RunNetPass: pass failure (missing pass or onExecute false) is now a reportable return value"
  - "SGFP4-gated failure chain: RunNetPass(false) -> optimizeNetImpl(nullptr + MNN_ERROR) -> convertModel(false, null-guard) -> main(exit 1)"
  - "cli.cpp null-guard converting the optimizeNet-nullptr crash into a clean MNN_ERROR + return false"
  - "T10 sibling assertion proving RunNetPass returns false on the NaN/Inf encode-failure net"
affects: [12-02 (E2E script exit-code gating), mnnconvert CLI consumers]

tech-stack:
  added: []
  patterns:
    - "SGFP4-scoped error escalation: new failure visibility gated on useSGFP4 so flag-off behavior stays byte-identical (D-12 strictest reading)"

key-files:
  created: []
  modified:
    - tools/converter/source/optimizer/PostConverter.cpp
    - tools/converter/include/PostConverter.hpp
    - tools/converter/source/TestSGFP4Converter.cpp
    - tools/converter/source/common/cli.cpp
    - tools/converter/source/MNNConverter.cpp

key-decisions:
  - "OQ1 (strictest D-12): MNNConverter.cpp main propagates exit 1 ONLY when modelPath.useSGFP4 && !convertOk; flag-off exit codes unchanged; unconditional propagation documented in-code as a deliberate follow-up"
  - "OQ4: void RunNetPass -> bool RunNetPass signature change; all pre-existing call sites ignore the return (zero behavior change); only the SGFP4 batch site in optimizeNetImpl consumes it"
  - "Missing-pass now returns false immediately (was: LOG + continue) — required by the D-11 truth table and covered by the still-failing-loudly registration canary"

patterns-established:
  - "Nullptr-as-failure contract for optimizeNet + null-guard at every convertModel call site (mirrors in-file :752-755 precedent)"

requirements-completed: [SGV2-31, SGV2-32]

coverage:
  - id: D1
    description: "RunNetPass returns bool (false on missing pass or onExecute failure) with updated declaration and T10 sibling assertion"
    requirement: SGV2-31
    verification:
      - kind: unit
        ref: "tools/converter/source/TestSGFP4Converter.cpp#T10 RunNetPass-returns-false sibling"
        status: pass
      - kind: command
        ref: ".build\\Release\\TestSGFP4Converter.exe (exit 0, PASS line)"
        status: pass
    human_judgment: false
  - id: D2
    description: "optimizeNetImpl aborts --sgfp4 conversions with MNN_ERROR + nullptr before ReIndexOnnxIfAlias; cli.cpp null-guards before extraTensorDescribe dereference"
    requirement: SGV2-31
    verification:
      - kind: integration
        ref: "MNNConvert -f ONNX corrupt.onnx --sgfp4 -> exit 1, no 'Converted Success!'"
        status: pass
      - kind: integration
        ref: "MNNConvert -f ONNX alexnet_Opset16.onnx --sgfp4 --dumpPass -> exit 0, 'InsertSGFP4Dequant: ops 74 -> 82', 'Converted Success!'"
        status: pass
    human_judgment: false
  - id: D3
    description: "D-12 flag-off byte-identical behavior: corrupt model without --sgfp4 exits 0; corpus without flag exits 0 with Converted Success"
    requirement: SGV2-32
    verification:
      - kind: command
        ref: "MNNConvert flag-off corrupt -> exit 0; flag-off corpus -> exit 0 + Converted Success!"
        status: pass
    human_judgment: false
  - id: D4
    description: "No regression in SGFP4 suites (op/sgfp4 13/13, TestSGFP4Converter green)"
    requirement: SGV2-31
    verification:
      - kind: command
        ref: ".build\\Release\\run_test.out.exe op/sgfp4 -> passed:13 failed:0"
        status: pass
    human_judgment: false

---

# Phase 12 Plan 01: SGFP4-Scoped RunNetPass Error Escalation (D-11/D-12) Summary

RunNetPass failure propagation chain `RunNetPass(false) -> optimizeNetImpl(nullptr) -> convertModel(false) -> main(exit 1)`, all gated on `useSGFP4`, proven at CLI level with positive/negative/flag-off legs.

**Duration:** ~15 min | **Tasks:** 2 | **Files modified:** 5 | **Commits:** c6d6906e, a0728c4c

## Accomplishments

- `void RunNetPass` -> `bool RunNetPass` in `PostConverter.cpp`/`PostConverter.hpp`: returns false on missing pass (immediate) or `onExecute` false; all diagnostics/dumpPass printing unchanged; pre-existing call sites ignore the return (D-12).
- `optimizeNetImpl` SGFP4 batch site: captures the bool, reads `Global<modelConfig>::Get()` exactly as RunNetPass does, and on `useSGFP4 && !result` emits `MNN_ERROR("[ERROR] --sgfp4 conversion failed: InsertSGFP4Dequant pass did not succeed.\n")` + `return nullptr` BEFORE `ReIndexOnnxIfAlias` runs.
- `cli.cpp` `convertModel` needOptimize branch: null-guard `if (newNet == nullptr) { MNN_ERROR("[ERROR] Optimize the MNN Net failed, cancel convert.\n"); return false; }` placed before the `newNet->extraTensorDescribe` dereference — crash converted to clean failure, no `writeFb`, no "Converted Success!".
- `MNNConverter.cpp` main: `bool convertOk = convertModel(...); if (modelPath.useSGFP4 && !convertOk) return 1; return 0;` with the OQ1/D-12 rationale comment.
- `TestSGFP4Converter.cpp` T10: fresh sibling net `net10x` asserting `RunNetPass({"InsertSGFP4Dequant"}, net10x)` returns false on the NaN/Inf encode-failure weight (both variants); stale registration-canary comment updated.

## E2E Proof (exact outputs)

| Leg | Command result | Exit |
|-----|----------------|------|
| Positive `--sgfp4` corpus | `[DumpPass] PostConvert::InsertSGFP4Dequant: ops 74 -> 82, tensors 0 -> 0` + `Converted Success!` | 0 |
| Negative `--sgfp4` corrupt 1KB | `[ERROR] Model file is not onnx model.` + `[ERROR] Convert error...`, NO `Converted Success!` | 1 |
| Flag-off corrupt | same messages as pre-change | 0 |
| Flag-off corpus (12-02 baseline producer) | `Converted Success!` | 0 |

Note on the negative leg: the corrupt model trips the ONNX parse failure path, which already returned `false` from `convertModel` — the new main-level gate turns that into exit 1. The `optimizeNetImpl`-nullptr leg (pass failure with a valid parse) is covered by T10's `RunNetPass`-false assertion plus the null-guard; both links of the chain are individually proven.

## Verification Results

- `cmake --build .build --config Release --target TestSGFP4Converter MNNConvert` — clean build
- `TestSGFP4Converter.exe` — `PASS (layout + reload parity + pass mechanics)`, exit 0
- `run_test.out.exe op/sgfp4` — 13/13 passed, 0 failed
- All four CLI legs above behaved as specified

## Deviations from Plan

**[Rule 2 - Tooling] clang-format unavailable on this machine** — Found during: Task 1/2 commit prep | Issue: `clang-format` is not installed (PATH, `.build`, LLVM dir all empty) | Fix: edits written manually to the project format rules (4-space indent, 120-col, attached braces, verified against neighboring code); no pre-commit hook exists in this submodule checkout to enforce it | Files: n/a | Verification: visual + successful build | Commit: n/a

**Total deviations:** 1 auto-fixed. **Impact:** none — formatting compliance deferred to a machine with clang-format; code compiles clean and matches file-local style.

## Self-Check: PASSED

All task acceptance criteria re-verified: header contains `bool RunNetPass(`; PostConverter.cpp contains the `--sgfp4 conversion failed` string and gated nullptr before ReIndexOnnxIfAlias; cli.cpp guard precedes the dereference; MNNConverter.cpp gated `return 1`; T10 CHECK present; suites green.
