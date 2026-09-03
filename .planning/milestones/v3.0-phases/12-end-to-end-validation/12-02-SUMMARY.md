---
phase: 12-end-to-end-validation
plan: "02"
subsystem: testing
tags: [sgfp4, e2e, validation, cpu, vulkan, tolerances, codec-fix]

requires:
  - phase: 12-end-to-end-validation
    provides: "Plan 12-01 D-11 exit-code chain (the script's negative leg consumes it)"
provides:
  - "tools/fp4/e2e_validation.ps1 - committed multi-leg E2E gate (convert, run CPU+Vulkan, tolerance-gate, D-11 negative leg, -MeasureOnly derivation mode)"
  - "Locked tolerances TolAbs=10.14433 / TolRel=948.601606 with recorded derivation"
  - "tools/fp4/README.md E2E validation section (full usage + methodology documentation)"
  - "SGFP4 v2 codec fixes (Rule 3 blockers surfaced by the accuracy gate): spatial decode convention in CPU/Vulkan runtime decoders; encoder split-map origin + child-order fixes; spatially-ordered test fixtures + oracles"
affects: [v3.0 milestone sign-off, all SGFP4 consumers (correctness-critical)]

tech-stack:
  added: []
  patterns:
    - "Measure-then-lock tolerances: -MeasureOnly prints measured worst x2.0 suggestion; locked values carry per-backend measured provenance in the script header"

key-files:
  created:
    - tools/fp4/e2e_validation.ps1
  modified:
    - tools/fp4/README.md
    - source/backend/cpu/CPUSGFP4Dequant.cpp
    - source/backend/cpu/CPUSGFP4Dequant.hpp
    - source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp
    - tools/fp4/sgfp4_encode.cpp
    - tools/fp4/encode_sgfp4.py
    - tools/fp4/sgfp4_inject_core.hpp
    - test/op/SGFP4DequantFixtures.h
    - test/op/SGFP4DequantTest.cpp
    - test/op/SGFP4VulkanDequantTest.cpp
    - test/op/SGFP4MultiTensorTest.cpp

key-decisions:
  - "OQ2 measure-then-lock: measured (post-codec-fix) cpu max-abs 5.072165/max-rel 237.302592, vulkan max-abs 3.926990/max-rel 474.300803; locked 2.0x worst = TolAbs 10.14433 / TolRel 948.601606, same gate both backends (D-06)"
  - "OQ3 guarded relative error relErr = absErr/max(|base|, 1e-3); max-abs primary, relative secondary"
  - "Node-presence assert: InsertSGFP4Dequant ops N->M with N != M (complements D-11)"
  - "Rule 3 codec fixes (auto-fixed blockers, deviation-documented): (1) runtime decoders moved to the normative SPATIAL padded-plane convention (gnus-poc decode_v2 = the Vulkan shader's existing convention; the flat stream decoder is only plane-correct for one-superblock-wide grids); (2) C++ encoder buildSplitMapBits fixed for global-vs-local coordinates AND TL/TR/BL/BR child order; C++ containers now pass strict python decode_v2 with errors identical to the python exporter's own roundtrip"

patterns-established:
  - "Cross-repo codec validation: encode with the C++ encoder, decode with the strict python decode_v2 (and vice versa) to prove convention parity"

requirements-completed: [SGV2-31, SGV2-32]

coverage:
  - id: D1
    description: "One committed script invocation converts, runs CPU + Vulkan (classic API), and gates both backends against the FP32 baseline"
    requirement: SGV2-31
    verification:
      - kind: e2e
        ref: "pwsh tools/fp4/e2e_validation.ps1 -Corpus W:\\gnus\\models\\alexnet_Opset16.onnx -> PASS: cpu / PASS: vulkan / E2E VALIDATION: PASS, exit 0"
        status: pass
    human_judgment: false
  - id: D2
    description: "Vulkan leg genuinely on Vulkan: backendType is 7 asserted from stdout, vulkaninfo pre-check, no SKIP (D-07)"
    requirement: SGV2-32
    verification:
      - kind: e2e
        ref: "script stdout 'vulkan backend confirmed: backendType is 7'; exit 2 path on missing device"
        status: pass
    human_judgment: false
  - id: D3
    description: "Tolerances locked with recorded derivation (measured-worst x 2.0, both backends, Phase 10 citation, text-dump floor caveat)"
    requirement: SGV2-31
    verification:
      - kind: command
        ref: "e2e_validation.ps1 header LOCKED block (per-backend measured values, date, commit 54bbeaf8) + README methodology"
        status: pass
    human_judgment: false
  - id: D4
    description: "D-11 negative leg green inside the gate invocation"
    requirement: SGV2-31
    verification:
      - kind: e2e
        ref: "PASS: D-11 negative leg (corrupt + --sgfp4 -> exit 1, no 'Converted Success!')"
        status: pass
    human_judgment: false
  - id: D5
    description: "D-10 diagnostics: per-backend max-abs/max-rel with argmax indices printed on every verdict line"
    requirement: SGV2-31
    verification:
      - kind: command
        ref: "PASS: cpu max-abs=5.07216500E+000 (idx 533), max-rel=2.37302592E+002 (idx 573)"
        status: pass
    human_judgment: false
  - id: D6
    description: "Codec correctness fixes (spatial decode + encoder split maps) with cross-repo python decode_v2 validation and full regression green"
    requirement: SGV2-32
    verification:
      - kind: command
        ref: "python decode_v2 on C++ container: maxAbs=0.367087 ratio=0.1648 (== python exporter's own roundtrip); run_test.out op/sgfp4 13/13; TestSGFP4Converter PASS"
        status: pass
    human_judgment: false

---

# Phase 12 Plan 02: Committed E2E Validation Script (SGV2-31/SGV2-32) Summary

One committed invocation gates SGFP4 v2 CPU and Vulkan inference against a same-path FP32 baseline with measure-then-locked tolerances — and, on the way, surfaced and fixed two latent SGFP4 codec defects that made every real converted model produce garbage.

**Duration:** ~2.5 h | **Tasks:** 2 | **Files:** script + README + 9 codec/test files | **Commits:** 54bbeaf8 (codec), 6292e25f (script), 7f5ed0e0 (lock+README)

## Accomplishments

- `tools/fp4/e2e_validation.ps1`: param block (`-Corpus` mandatory; `-MnnConvert`/`-Driver`/`-WorkRoot` defaulted; `-MeasureOnly` switch), vulkaninfo D-07 fail-fast, deterministic input (seed 20260901, [-1,1), 150,528 floats), baseline+SGFP4 conversions with `InsertSGFP4Dequant: ops N -> M` node-presence assert, three isolated driver legs (forward 0/0/7, precision mask 1, per-leg WorkingDirectory, `backendType is 7` assert), guarded comparison core with D-10 diagnostics, D-11 negative leg, pass-only cleanup.
- Full gate green: `PASS: cpu max-abs=5.07216500E+000 (idx 533), max-rel=2.37302592E+002 (idx 573)`; `PASS: vulkan max-abs=3.92699000E+000 (idx 533), max-rel=4.74300803E+002 (idx 638)`; `PASS: D-11 negative leg`; `E2E VALIDATION: PASS (cpu + vulkan + D-11 negative)`, exit 0.
- Tolerances locked: `TolAbs = 10.14433`, `TolRel = 948.601606` (2.0x measured worst across both backends; per-backend measured values, date, and codec-fix provenance in the script header and README).
- README §"SGFP4 v2 end-to-end validation (Phase 12)": prerequisites (hard Vulkan), invocation block with `# expect:` lines, parameter table, tolerance methodology with Phase 10 citation and text-dump floor caveat, seed/range spec, exit semantics, D-11 leg description, codec-fix note. No "Ultra FP4" occurrences.
- **Codec fixes (Rule 3 blockers)**: see Deviations.

## Verification Results

- Full gate: exit 0 with per-backend PASS lines (exact lines above).
- `-MeasureOnly`: exit 0, both backends' measured values printed (pre-lock evidence).
- `run_test.out op/sgfp4`: 13/13 passed, 0 failed (after fixture/oracle updates).
- `TestSGFP4Converter.exe`: PASS.
- Cross-repo: C++ encoder container → strict python `decode_v2` = maxAbs 0.367087 / ratio 0.1648, identical to the python exporter's own roundtrip on the same weight; python container → C++ spatial decoder = same error (convention parity proven both directions).

## Deviations from Plan

**[Rule 3 - Blocker bug] SGFP4 v2 codec produced garbage for every real multi-tile / MIXED weight** — Found during: Task 1 `-MeasureOnly` (CPU output all-zero, Vulkan noise; weight-level decode showed blocks off by ~5 orders of magnitude) | Issue (two defects): (1) `CPUSGFP4Dequant::onResize/onExecute` used `dequant_sgfp4_container_cpu` (flat leaf-concat stream order), which equals the row-major plane only for one-superblock-wide grids — the normative convention (gnus-poc `decode_v2`, already implemented by the Vulkan shader and `dequant_sgfp4_container_cpu_plane`) places record (by,bx) spatially; (2) the C++ encoder's `buildSplitMapBits` compared GLOBAL leaf coordinates against a LOCAL (0,0)-rooted walk (corrupting every superblock except (0,0)) and pushed children so the walk popped BL before TR while `tryBlock` and the decoder both traverse TL,TR,BL,BR (corrupting every deep LAYOUT_MIXED tree) | Fix: CPU + Vulkan-prevalidation decoders now always call `dequant_sgfp4_container_cpu_crop` (spatial; aligned case = no-op stride crop); `buildSplitMapBits` takes the superblock origin and the corrected push order; `mIsPadded` member removed (no longer needed) | Files: `source/backend/cpu/CPUSGFP4Dequant.{cpp,hpp}`, `source/backend/vulkan/buffer/execution/VulkanSGFP4Dequant.cpp`, `tools/fp4/sgfp4_encode.cpp` | Verification: python `decode_v2` cross-check both directions (identical errors to the python exporter's own roundtrip); weight-level artifact check all 8 weights at FP4 noise; 13/13 suites + converter tests green | Commit: 54bbeaf8

**[Rule 2 - Consequential test updates] Fixtures and oracles regenerated to the spatial convention** — Found during: the codec fix flipped the output order for all multi-leaf fixtures | Issue: `test/op/SGFP4DequantFixtures.h` expected arrays were leaf-concat (stream) order — a test-generator convention, not the runtime norm | Fix: `encode_sgfp4.py` gained `decode_container_ref_spatial` and `--emit-cpp-fixture` now emits spatially-ordered expected vectors (container bytes unchanged); direct-decode oracles in `SGFP4DequantTest.cpp` (fixture round-trip, hand-built traversal golden, mixed golden + round-trip), `SGFP4VulkanDequantTest.cpp` (both parity suites' CPU references), `SGFP4MultiTensorTest.cpp` (structured/uniform/pristine oracle decodes), and `sgfp4_inject_core.hpp` (verification oracle) switched to `_crop`; the hand-built TL/TR/BL/BR golden now asserts quadrant-tile placement | Files: listed in key-files | Verification: 13/13 `op/sgfp4` suites green | Commit: 54bbeaf8

**[Rule 2 - Tooling] Environment friction (PS 5.1 encoding, stale driver)** — Found during: Task 1 bring-up | Issue: (a) em-dash characters broke PS 5.1 parsing of BOM-less files — script saved ASCII + UTF-8 BOM; (b) `vulkaninfo` stderr under `$ErrorActionPreference=Stop` threw — captured via `cmd /c ... 2>&1`; (c) the repo's `MNNV2Basic.out.exe` predated the Phase 11 D-13 runtime fix — rebuilt | Files: `tools/fp4/e2e_validation.ps1` | Verification: script runs end-to-end | Commit: 6292e25f

**Total deviations:** 1 Rule-3 auto-fixed (codec), 2 Rule-2 consequential. **Impact:** the codec fixes are correctness-critical for every SGFP4 v2 consumer (both backends, the injection tool, the converter) — Phase 11's smoke (load/run only, no output comparison) could not see them; they are the reason this E2E gate exists.

## Self-Check: PASSED

Script acceptance criteria re-verified: param block with `$Corpus`/`$MeasureOnly`/`$TolAbs`/`$TolRel`/`$Eps = 1e-3`/`$Seed = 20260901` (locked numeric literals, no placeholders); `no Vulkan device` exit-2 pre-check; all legs precision mask 1 + per-leg `-WorkingDirectory`; Vulkan `backendType is 7` assert; guarded comparison with failing indices; `InsertSGFP4Dequant: ops` assert; negative leg asserts non-zero exit AND absence of `Converted Success!`; `-MeasureOnly` and full-gate runs both exit 0; README section complete with no `Ultra FP4`.
