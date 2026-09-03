---
plan: "11-01"
status: complete
started: 2026-09-01
completed: 2026-09-01
commits: [dc6b1d62]
---

# Plan 11-01 Summary: InsertSGFP4Dequant PostConverter pass

## What Was Built

**Task 1 — encoder hoist + config field + declaration (OQ2 Option A):**
- Root `CMakeLists.txt`: `sgfp4_encode` static lib defined inside the `if (NOT MNN_SKIPBUILD_GEOMETRY)` block, immediately ABOVE `add_subdirectory(tools/converter)` (line 922 vs 929), guarded by `IF(MNN_BUILD_CONVERTER)`; PIC ON for Linux SHARED MNNConvertDeps.
- `tools/fp4/CMakeLists.txt`: existing definition wrapped in `if(NOT TARGET sgfp4_encode)` — exactly-once across option combos (both converter and SGFP4 tools ON verified by this build).
- `tools/converter/CMakeLists.txt`: single `target_link_libraries(MNNConvertDeps PUBLIC sgfp4_encode)` after the SHARED/STATIC IF/ELSE — covers both branches.
- `tools/converter/include/config.hpp`: `bool useSGFP4 = false;` adjacent to `useHQQ` (D-04 field; CLI parse lands in 11-03).
- `tools/converter/include/PostConverter.hpp`: `MNN::Express::RunNetPass` declaration (namespace-qualified to match the PostConverter.cpp definition — global-scope declaration caused C2668 ambiguity).

**Task 2 — the pass + registration + skip-guard:**
- NEW `tools/converter/source/optimizer/postconvert/InsertSGFP4Dequant.cpp` (277 lines): D-14 flag-off dead-code guard; D-08 `kSGFP4ConverterEncodeConfig` file-scope alias; D-06 4-type gate; idempotency (`inputIndexes.size() == 1` AND `quanParameter == nullptr`); KEY Q3 spilled-weight reload (externalFile flush → FileLoader on `.__convert_external_data.bin` → bias restore → external clear); Pitfall 8 dims arithmetic; T-11-02 size_t pre-encode assertion; D-07 light-tier floor (`< 4096 || dimI == 1`); transactional mutation (clear/push only after successful encode); Phase 8 D-11 buffer contract (buffer populated, external `{}`, no externalPath); D-03 both oplist + subgraph->nodes walks with per-scope tensor-namespace appenders; `_sgfp4` node naming per injection-tool convention; producer inserted immediately before consumer.
- `PostConverter.cpp:393`: `RunNetPass({"InsertSGFP4Dequant", "ReIndexTensor"}, newNet)` — BEFORE ReIndexTensor (KEY Q2 order lock).
- `WeightQuantAndCoding.cpp:95`: D-02 `inputIndexes.size() > 1` early return, positioned after quanParameter return (:87) and before `weightQuantBits == 0` (:142) — verified by line numbers.

## Deviations

1. **`RunNetPass` declaration namespaced** (`MNN::Express::`): plan asked for a global declaration matching the definition "character-for-character" — but the definition lives inside `namespace MNN::Express`, so a global-scope declaration is a *different* function to MSVC (C2668 ambiguity at PostConverter.cpp:636). The declaration is now inside the same namespace; signature otherwise identical.
2. **`tools/converter/source/optimizer/CMakeLists.txt` edited** (not in plan's files_modified): `MNNConverterOpt` is an OBJECT library consumed via `$<TARGET_OBJECTS:>` — sgfp4_encode's PUBLIC include dirs do not reliably propagate through that consumption pattern on this generator; added explicit `target_include_directories` for tools/fp4 + include + 3rd_party/half. (First attempt used `../../../` — one level short, resolved to `tools/tools/fp4`; fixed to `../../../../`.)
3. **Plan's verify step `cmake --build . --target MNNConvertDeps TestSGFP4Converter`**: default (Debug) config fails to LINK MNNConvert in this tree — `libprotobuf.lib` was built Release (MT vs MTd LNK2038 mismatch, pre-existing). Built with `--config Release` instead, matching how prior phases produced MNNConvert.exe/TestSGFP4Converter.exe in `.build/Release/`.

## Self-Check

- [x] All source assertions from acceptance criteria verified via Select-String (line numbers captured above)
- [x] `MNNConvert.exe` + `TestSGFP4Converter.exe` build clean (Release)
- [x] `TestSGFP4Converter.exe` → "PASS (layout + reload parity)", exit 0
- [x] `run_test.out op/sgfp4` → 13/13 passed (flag-off invariance, D-14)
- [x] Configure with BOTH `MNN_BUILD_SGFP4_TOOLS=ON` and `MNN_BUILD_CONVERTER=ON` — no duplicate-target error (this .build cache has both ON; re-configure succeeded)
- [x] Committed atomically: dc6b1d62

## Key Files

### created
- `tools/converter/source/optimizer/postconvert/InsertSGFP4Dequant.cpp`

### modified
- `CMakeLists.txt`
- `tools/fp4/CMakeLists.txt`
- `tools/converter/CMakeLists.txt`
- `tools/converter/source/optimizer/CMakeLists.txt` (deviation 2)
- `tools/converter/include/config.hpp`
- `tools/converter/include/PostConverter.hpp`
- `tools/converter/source/optimizer/PostConverter.cpp`
- `tools/converter/source/common/WeightQuantAndCoding.cpp`

## Open Items

- Flag-off end-to-end MNNConvert run (no `--sgfp4`) — deferred to plan 11-05's D-14 gate which runs it properly on the corpus; unit-level flag-off invariance (pass never runs body when config false) is proven by the 13/13 suite + PHASE A/B above.
