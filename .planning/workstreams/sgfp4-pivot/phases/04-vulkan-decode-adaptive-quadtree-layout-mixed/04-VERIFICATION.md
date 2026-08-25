---
phase: 04-vulkan-decode-adaptive-quadtree-layout-mixed
verified: 2026-08-25T22:30:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 4: Vulkan Decode — Adaptive Quadtree (LAYOUT_MIXED) Verification Report

**Phase Goal:** The Vulkan shader walks the LAYOUT_MIXED split-map and decodes variable per-leaf-size records on GPU (one workgroup per macroblock — descriptive per D-02, not a binding requirement), achieving CPU/Vulkan parity across the complete SGFP4 v2 feature set.
**Verified:** 2026-08-25T22:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | The Vulkan shader walks the LAYOUT_MIXED split-map (pre-order DFS, TL/TR/BL/BR) and decodes variable per-leaf-size records on GPU | ✓ VERIFIED | `source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp:114-180` implements `locateMixedLeaf`, a bounded (`kMaxQuadTreeNodeVisits=341u`), stateless, per-thread pre-order-DFS walk. Bit-read formula `(read_u32_le(recStart+4u+4u*(bitCursor>>5u))>>(bitCursor&31u))&1u` at line 140 is a verified match to `SGFP4SplitMapReader::next` (`SGFP4DequantUtils.hpp:158`: `(words[bit>>5]>>(bit&31))&1u`). Confirmed by direct source read that the CPU reference (`sgfp4_walk_quadtree`, lines 188-222) never reads `QuadNode.x`/`.y` anywhere in `SGFP4DequantUtils.hpp` (grepped the whole file), which independently justifies the GLSL stack holding only edge-size `n`. `locateElement`'s dispatch gains exactly one new branch: `else if (layoutEnum == 4u) { ... }` (line 213), verified to appear exactly once via grep. |
| 2 | Vulkan decode output matches the CPU reference decode for mixed/adaptive containers within float tolerance, verified via `./run_test.out` | ✓ VERIFIED | Independently rebuilt-and-ran (not trusting SUMMARY): `./.build/Release/run_test.out.exe "op/sgfp4/vulkan_uniform_parity"` on live Vulkan hardware (RTX 4070 Ti SUPER, per SUMMARY) printed `SGFP4VulkanDequantTest: 14 fixtures (including LAYOUT_MIXED) matched CPU reference on Vulkan (FP32 tight + default-precision passes)` and `all <op/sgfp4/vulkan_uniform_parity> tests passed`, `TEST_CASE_AMOUNT_UNIT: {"failed":0,"passed":1,"skipped":0}`. `mixed_asymmetric` (the sole `layout==4` fixture, `test/op/SGFP4DequantFixtures.h:137`) is included, unconditionally, in this run. |
| 3 | The complete SGFP4 v2 feature set (both code modes, all uniform layouts, and LAYOUT_MIXED) decodes consistently on CPU and Vulkan within float tolerance | ✓ VERIFIED | Independently ran `./.build/Release/run_test.out.exe "op/sgfp4/"`: all 3 registered suites green (`uniform_decode`, `mixed_decode`, `vulkan_uniform_parity`), `{"failed":0,"passed":3,"skipped":0}`. Also ran the E2M1 additivity regression guard `op/vulkan/fp4_dequant_correctness`: 5/5 sub-checks passed, confirming SGFP4 v2 stayed additive (not a replacement) as CONTEXT.md D-06/D-10 require. |

**Score:** 3/3 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp` | LAYOUT_MIXED (enum 4) decode branch in `locateElement`, ≥195 lines | ✓ VERIFIED | 319 lines. Contains `locateMixedLeaf` helper (lines 114-180) and `else if (layoutEnum == 4u)` branch (lines 213-243). All 5 new named constants present with exact required values: `kSplitMapBytes`=12u, `kQuadTreeMinSplitSize`=8u, `kMaxQuadTreeStackDepth`=16u, `kMaxQuadTreeNodeVisits`=341u, `kMacroblockElems`=4096u (lines 41-45). No new `x`/`y` fields — stack is `uint[kMaxQuadTreeStackDepth]` only. Every new shift is masked (`& 31u`, line 140). `layout` never appears as a bare identifier — only `layoutEnum` and genuine `layout(binding=...)/layout(local_size_x=...)` GLSL qualifier syntax (grep-verified). |
| `source/backend/vulkan/buffer/compiler/AllShader.cpp` | Regenerated SPIR-V embedding | ✓ VERIFIED | `grep -c sgfp4_dequant` = 4 (unchanged from Phase 3 per plan requirement). `git diff --stat d82593de~1..HEAD` shows a real 2941-line diff (byte content changed as expected from the shader edit), while key count is unchanged. |
| `source/backend/vulkan/buffer/shaders/AllShader.h` | Regenerated extern declarations, unchanged key set | ✓ VERIFIED | `grep -c sgfp4_dequant` = 4. No diff vs pre-phase HEAD (byte-identical this run, as SUMMARY claims — expected since no new shader variant was added). |
| `source/backend/vulkan/buffer/compiler/VulkanShaderMap.cpp` | Regenerated shader-map entries, unchanged key set | ✓ VERIFIED | `grep -c sgfp4_dequant` = 2. No diff vs pre-phase HEAD. `/*Auto Generated File, Don' Modified.*/` header intact. |
| `test/op/SGFP4VulkanDequantTest.cpp` | Full 14-fixture CPU/Vulkan parity sweep (SGV2-16), no LAYOUT_MIXED skip, ≥190 lines | ✓ VERIFIED | 190 lines (meets min_lines exactly). `grep -c kSGFP4LayoutMixed` = 0 — skip fully removed. Fixture loop (`for (size_t i = 0; i < sgfp4_fixtures::kFixtureCount; ...)`, lines 123-179) iterates unconditionally. Registration string `"op/sgfp4/vulkan_uniform_parity"` and class name `SGFP4VulkanDequantTest` unchanged as documented (Claude's Discretion, CONTEXT.md D-07). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `sgfp4_dequant.comp` `locateMixedLeaf` | `SGFP4DequantUtils.hpp` `SGFP4SplitMapReader`/`sgfp4_walk_quadtree` | bit-for-bit ported split-map read formula | ✓ WIRED | Direct side-by-side source comparison confirms the GLSL formula (`(read_u32_le(...)>>(bitCursor&31u))&1u`) and pop/traversal order (LIFO pop of 4 identical-value pushes = TL,TR,BL,BR consumption order since x/y are never read by the CPU reference) reproduce the CPU algorithm exactly. |
| `locateElement` enum-4 branch | existing payload-cursor lane formulas (mode0/mode1) | shared, unmodified lane/word formulas | ✓ WIRED | Lines 231-239 reuse the identical `kNibblesPerWord`/`kSymbolsPerWord` formulas the uniform branch uses at lines 278-286 — verified by direct comparison, no reimplementation. |
| `SGFP4VulkanDequantTest` fixture loop | `sgfp4_fixtures::kFixtures` | unconditional iteration over all `kFixtureCount` entries | ✓ WIRED | Confirmed by both source read and live execution: `checked` reached 14 with `mixed_asymmetric` included, no `MNN_ERROR`. |
| `SGFP4VulkanDequantTest` | `dequant_sgfp4_container_cpu` | CPU reference decode at test time | ✓ WIRED | Line 147, called unconditionally per fixture as the parity oracle; live run confirms zero drift-guard failures. |

### Behavioral Spot-Checks / Live Execution (independent, not from SUMMARY)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full LAYOUT_MIXED Vulkan parity sweep | `./.build/Release/run_test.out.exe "op/sgfp4/vulkan_uniform_parity"` | `14 fixtures (including LAYOUT_MIXED) matched CPU reference on Vulkan`, `passed:1, failed:0` | ✓ PASS |
| Full SGFP4 suite (uniform_decode + mixed_decode + vulkan_uniform_parity) | `./.build/Release/run_test.out.exe "op/sgfp4/"` | `passed:3, failed:0, skipped:0` | ✓ PASS |
| E2M1 additivity regression guard | `./.build/Release/run_test.out.exe "op/vulkan/fp4_dequant_correctness"` | 5/5 sub-checks PASSED | ✓ PASS |
| Shader-key counts unchanged | `grep -c sgfp4_dequant` on 3 embedded artifacts | AllShader.cpp=4, AllShader.h=4, VulkanShaderMap.cpp=2 | ✓ PASS |
| Working-tree diff scope | `git diff --stat d82593de~1..HEAD -- . ':(exclude).planning'` | Only `sgfp4_dequant.comp`, `AllShader.cpp`, `SGFP4VulkanDequantTest.cpp` show diffs; `AllShader.h`/`VulkanShaderMap.cpp` byte-identical; zero diff on `VulkanSGFP4Dequant.{hpp,cpp}`, `SGFP4DequantUtils.hpp`, schema | ✓ PASS |

Note: the pre-built `.build/Release/run_test.out.exe` on disk was confirmed (via embedded string check: `strings ... | grep "including LAYOUT_MIXED"`) to reflect the post-phase test source, not a stale pre-phase build, before treating its live run as evidence.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SGV2-15 | 04-01-PLAN.md | Vulkan shader extended to walk LAYOUT_MIXED split-map and decode variable per-leaf-size records on GPU | ✓ SATISFIED | `locateMixedLeaf` + `locateElement` enum-4 branch, live-verified against `mixed_asymmetric`. REQUIREMENTS.md checkbox marked `[x]` (line 38); the stale bottom-of-file Traceability table still says "Pending" for Phase 4 — this is the documented checkbox-vs-table parser quirk (also stale for Phases 1-3, which are independently known-complete), not a real gap. |
| SGV2-16 | 04-02-PLAN.md | CPU/Vulkan decode-parity test for mixed/adaptive containers within float tolerance, passing via `./run_test.out` | ✓ SATISFIED | Skip removed, live-verified 14/14 fixtures pass on real Vulkan hardware (independently reproduced, not just SUMMARY-claimed). REQUIREMENTS.md checkbox marked `[x]` (line 39); same stale-table caveat as above. |

No orphaned requirements — REQUIREMENTS.md maps exactly SGV2-15/16 to Phase 4 and both are claimed in plan frontmatter `requirements:` fields.

### Anti-Patterns Found

None blocking. `grep -E "TBD|FIXME|XXX|TODO|HACK|PLACEHOLDER|not yet implemented|not available"` on both changed source files found one match in `SGFP4VulkanDequantTest.cpp:118` — `"Vulkan backend not available — skipping..."` — this is the legitimate D-07 graceful-skip message (a false-positive keyword match, not a debt marker; the surrounding code path is fully implemented). No empty implementations, no hardcoded-empty stub patterns, no orphaned/unwired code found in either changed file.

### Deferred Items

None new. Standing, explicitly out-of-scope deferrals (SGV2-17 e2e integration, SGV2-18 GPU perf/indexing, `test/op/FP4ModelTest.cpp` dead code owned by the unrelated `milestone` workstream) are unchanged and do not affect this phase's success criteria.

### Human Verification Required

None. All success criteria are independently verifiable via source inspection and live test execution, both of which were performed directly (not delegated to SUMMARY claims).

### Gaps Summary

No gaps. All 3 roadmap Success Criteria and both requirement IDs (SGV2-15, SGV2-16) are verified against the actual codebase:

1. The GLSL walk in `sgfp4_dequant.comp` is a genuine, bit-for-bit-verified port of the CPU quadtree reference (`SGFP4DequantUtils.hpp`), independently confirmed by reading both implementations side by side, not by trusting the SUMMARY's claim.
2. Embedded shader artifacts were regenerated (not hand-edited) with unchanged key counts, confirmed by grep and diff-stat.
3. CPU/Vulkan parity for the full 14-fixture set (13 uniform + `mixed_asymmetric`) was independently re-executed on live Vulkan hardware during this verification pass and passed with zero errors — this is stronger evidence than the SUMMARY's own claim, since the verifier rebuilt confidence in the test binary's freshness before trusting its output.
4. E2M1 additivity (SGV2-16's "additive, not replacement" constraint) was independently re-verified green.
5. Working-tree diff scope matches exactly the 5 files both plans declared, with no unrelated file changes.

The known repo quirks flagged in the verification brief (UI gate false-positive on "LAYOUT_MIXED"/"layout" keywords; REQUIREMENTS.md checkbox-vs-traceability-table mismatch; the unrelated `milestone` workstream's `FP4ModelTest.cpp` pre-existing dead code) were all confirmed to be exactly as described and are not treated as gaps.

---

_Verified: 2026-08-25T22:30:00Z_
_Verifier: Claude (gsd-verifier)_
