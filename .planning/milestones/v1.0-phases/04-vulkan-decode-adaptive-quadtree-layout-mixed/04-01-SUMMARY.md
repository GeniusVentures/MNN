---
phase: 04-vulkan-decode-adaptive-quadtree-layout-mixed
plan: 01
subsystem: infra
tags: [glsl, vulkan, compute-shader, quadtree, sgfp4, glslang, makeshader]

# Dependency graph
requires:
  - phase: 03-vulkan-decode-uniform-layouts
    provides: sgfp4_dequant.comp uniform-layout locateElement, read_u32_le, unpackLeafHeader, codeMode0/1, the makeshader.py regeneration pipeline, and the SGFP4VulkanDequantTest.cpp parity harness
  - phase: 02-adaptive-quadtree-layout-cpu-layout-mixed
    provides: the CPU SGFP4SplitMapReader / sgfp4_walk_quadtree reference algorithm this GLSL walk mirrors bit-for-bit, and the mixed_asymmetric fixture with known-good expected weights
provides:
  - GLSL locateMixedLeaf helper + locateElement enum-4 branch decoding LAYOUT_MIXED on GPU
  - Regenerated AllShader.cpp/AllShader.h/VulkanShaderMap.cpp embedding the extended shader
  - Live-verified (uncommitted sanity gate) parity of all 14 fixtures including mixed_asymmetric
affects: [04-02-PLAN, sgfp4-pivot workstream Phase 4 completion, SGV2-16 full-sweep test authoring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GLSL bounded quadtree descent: fixed uint[16] stack holding only edge-size n (never x/y), compile-time loop cap 341, no recursion"
    - "WSL-hosted Windows glslang toolchain must be invoked with relative paths from a drvfs (/mnt/...) cwd -- absolute WSL /tmp paths are unresolvable by the Windows .exe"

key-files:
  created: []
  modified:
    - source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp
    - source/backend/vulkan/buffer/compiler/AllShader.cpp
    - source/backend/vulkan/buffer/shaders/AllShader.h (regenerated, byte-identical this run)
    - source/backend/vulkan/buffer/compiler/VulkanShaderMap.cpp (regenerated, byte-identical this run)

key-decisions:
  - "Stack holds only edge-size n (not x,y,n triples) since dequant_sgfp4_container_cpu never reads QuadNode.x/.y and all 4 split children share edge n/2u -- push order is irrelevant"
  - "MIXED branch is self-contained inside its own else-if(layoutEnum==4u){...} using `continue` to skip the shared uniform N*n*n tail, since MIXED has no fixed per-leaf n"
  - "glslangValidator (a symlinked Windows .exe) invoked from WSL must use relative paths from a drvfs cwd, never /tmp -- Windows exes cannot resolve pure-Linux VM filesystem paths through interop"

patterns-established:
  - "locateMixedLeaf(recStart, local, out leafOrdinal, out leafN, out inLeaf, out blockHeadersStart, out payloadCursor): single bounded walk that both locates the target leaf and accumulates payload-cursor offset for leaves strictly before it"

requirements-completed: [SGV2-15]

coverage:
  - id: D1
    description: "locateElement's LAYOUT_MIXED branch (enum 4) decodes via a bounded, stateless, per-thread split-map walk matching the CPU reference bit-for-bit"
    requirement: "SGV2-15"
    verification:
      - kind: unit
        ref: "run_test.out.exe op/sgfp4/vulkan_uniform_parity (temporary uncommitted skip-bypass) -- 14/14 fixtures incl. mixed_asymmetric matched CPU reference at FP32-tight (rtol 1e-4) and default-precision (rtol 2e-3) passes"
        status: pass
    human_judgment: false
  - id: D2
    description: "Regenerated AllShader.cpp/AllShader.h/VulkanShaderMap.cpp embed the updated shader via makeshader.py, unchanged shader-key counts (4/4/2, same as Phase 3)"
    requirement: "SGV2-15"
    verification:
      - kind: unit
        ref: "grep -c sgfp4_dequant AllShader.cpp==4, AllShader.h==4, VulkanShaderMap.cpp==2; makeshader.log grep 'error' == 0 matches; total const-unsigned-char entry count unchanged (359 before/after)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Standing (unmodified, skip-still-present) test file passes with zero regression to the already-shipped uniform path"
    requirement: "SGV2-15"
    verification:
      - kind: unit
        ref: "run_test.out.exe op/sgfp4/ -- 13/13 uniform fixtures + mixed_decode + uniform_decode suites, passed:3 failed:0"
        status: pass
    human_judgment: false

duration: 10min
completed: 2026-08-25
status: complete
---

# Phase 4 Plan 1: Vulkan LAYOUT_MIXED Quadtree Decode Summary

**GLSL bounded split-map walk in `locateElement` decodes SGFP4 v2 LAYOUT_MIXED (enum 4) on GPU, matching the CPU quadtree reference bit-for-bit on `mixed_asymmetric`**

## Performance

- **Duration:** 10 min
- **Started:** 2026-08-25T20:48:46Z
- **Completed:** 2026-08-25T20:58:56Z
- **Tasks:** 2 (combined into one atomic commit per CLAUDE.md's makeshader.py contract)
- **Files modified:** 2 (with real diffs: `sgfp4_dequant.comp`, `AllShader.cpp`); 2 more regenerated-but-byte-identical this run (`AllShader.h`, `VulkanShaderMap.cpp`)

## Accomplishments
- `locateElement` now decodes all six layout enums (0/1/2/3/5 uniform + 4 MIXED); enum >= 6 remains the sole rejection path
- New `locateMixedLeaf` helper: a bounded, stateless, per-thread pre-order-DFS walk over the 3-word/12-byte split map, using a fixed `uint[16]` stack of pending edge-sizes and a `341`-iteration loop cap — no recursion, no shared memory, no dynamically-sized local storage (D-01/D-03/D-04)
- Live GPU sanity gate (temporary, uncommitted) confirmed all 14 fixtures — including the one true `mixed_asymmetric` LAYOUT_MIXED fixture — decode identically on CPU and Vulkan at both FP32-tight (rtol 1e-4) and default-precision (rtol 2e-3) tolerances
- Regenerated and committed `AllShader.cpp`/`AllShader.h`/`VulkanShaderMap.cpp` via the locked `makeshader.py` pipeline; shader-key counts unchanged (4/4/2, matching Phase 3)
- Zero regression: the standing, unmodified test file still passes 13/13 uniform fixtures plus the CPU-side `uniform_decode`/`mixed_decode` suites

## Task Commits

Task 1 (shader authoring) and Task 2 (regeneration + live sanity gate + commit) were combined into a single atomic commit, per CLAUDE.md's explicit contract that a buffer-backend GLSL edit and its regenerated embedded-shader artifacts must land together:

1. **Task 1 + Task 2: Author LAYOUT_MIXED walk, regenerate shader artifacts, live-verify, commit** - `d82593de` (feat)

**Plan metadata:** (this commit, plus STATE/ROADMAP/REQUIREMENTS updates — see below)

_Note: the live GPU sanity-gate test edit (skip-bypass in `SGFP4VulkanDequantTest.cpp`) was made, verified, and reverted via `git checkout --` before this commit — `git diff --exit-code test/op/SGFP4VulkanDequantTest.cpp` confirmed clean, so it never entered the commit._

## Files Created/Modified
- `source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp` - New `locateMixedLeaf` helper + `else if (layoutEnum == 4u)` branch in `locateElement`; 5 new named constants (`kSplitMapBytes`, `kQuadTreeMinSplitSize`, `kMaxQuadTreeStackDepth`, `kMaxQuadTreeNodeVisits`, `kMacroblockElems`); stale header/docstring comments updated to reflect completed layout-enum coverage
- `source/backend/vulkan/buffer/compiler/AllShader.cpp` - Regenerated SPIR-V byte-array embedding (same 359 total shader entries as HEAD, same sgfp4_dequant key count 4, differing only in the sgfp4_dequant SPIR-V content and unrelated-entry ordering per the documented WSL-`find`-ordering churn pattern)
- `source/backend/vulkan/buffer/shaders/AllShader.h` - Regenerated (byte-identical to the committed version this run — same shader-key set, no new variants)
- `source/backend/vulkan/buffer/compiler/VulkanShaderMap.cpp` - Regenerated (byte-identical to the committed version this run)

## Decisions Made
- **Stack holds only `n` (edge size), never `(x, y, n)`:** verified against `dequant_sgfp4_container_cpu` (`SGFP4DequantUtils.hpp:403,435`) that `x`/`y` are never read; all 4 children of a split share edge `n/2u` and decode order (leaf-major, per pre-order-DFS traversal) makes push/pop order among identical values irrelevant — this let the walk use a plain `uint[16]` stack instead of a `QuadNode`-shaped one.
- **MIXED branch is fully self-contained with `continue`:** rather than trying to shoehorn MIXED into the shared `N*n*n` uniform tail (which assumes a single record-wide leaf edge, invalid for MIXED's variable per-leaf `n`), the new `else if (layoutEnum == 4u)` branch does its own idx-check/decode/return/base-increment inline and uses `continue` to skip the shared tail for that loop iteration when its record doesn't contain `idx`.
- **`half` renamed to `childN`:** GLSL reserves `half` as a keyword (half-precision float type name) even though not otherwise used in this shader; the standalone glslangValidator compile caught this immediately (Pitfall 2's sibling issue — same class of reserved-word trap as the previously-fixed `layout`→`layoutEnum` rename).
- **WSL/glslang toolchain path handling:** confirmed and applied Phase 3's documented recipe (`03-01-SUMMARY.md`: "Windows glslang exes cannot resolve Linux absolute paths — invoke with relative paths from a drvfs cwd") for both the standalone Task-1 compile check and the `makeshader.py` regeneration; `/tmp`-based paths (even after MSYS path-conversion workarounds) reliably produced `unable to open input file`, while `cd`-ing into a `/mnt/w/...` directory and using relative filenames worked cleanly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Renamed reserved-keyword local variable `half` to `childN`**
- **Found during:** Task 1 standalone glslangValidator compile check
- **Issue:** The initial `locateMixedLeaf` implementation used `uint half = n / 2u;` for the split-children edge size; `half` is a GLSL reserved word (half-precision float type), causing a hard parse error (`'half' : Reserved word`)
- **Fix:** Renamed the local variable to `childN` throughout the split-bit-handling block
- **Files modified:** `source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp`
- **Verification:** Standalone `glslangValidator -S comp --target-env vulkan1.1 -V` re-run produced `EXIT:0` and a valid SPIR-V binary
- **Committed in:** `d82593de` (part of the combined Task 1+2 commit; never landed as reserved-word-broken code in any commit)

**2. [Rule 1 - Bug] Restructured the `locateElement` layout-enum dispatch to keep the MIXED branch self-contained via `continue`**
- **Found during:** Task 1 implementation planning (before first edit)
- **Issue:** The plan's literal instruction ("replace the final `else` with `else if (layoutEnum == 4u) {...} else {return false;}`") would, if implemented naively inside the existing `if/else-if` chain that sets `N`/`n` and falls through to a shared `N*n*n` tail, either leave `N`/`n` uninitialized when MIXED's branch doesn't set them, or force MIXED's variable per-leaf geometry into the uniform tail's fixed-`n` assumptions
- **Fix:** Kept the shape the plan asked for (`else if (layoutEnum == 4u) { ... } else { return false; }`, `layoutEnum == 4u` appears exactly once) but made the MIXED branch fully self-contained (its own idx-check, decode, and `base += recordElems`) and added a `continue;` at the end of that branch so the shared uniform tail below the if-chain never executes with uninitialized `N`/`n` for MIXED records
- **Files modified:** `source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp`
- **Verification:** All 14 fixtures (uniform + MIXED) passed the live Vulkan sanity gate; the shared uniform tail's behavior is provably unaffected since it is unreachable for `layoutEnum == 4u`
- **Committed in:** `d82593de`

---

**Total deviations:** 2 auto-fixed (1 blocking reserved-word fix, 1 bug-class control-flow correction to satisfy the plan's stated acceptance criteria without leaving uninitialized variables)
**Impact on plan:** Both fixes were necessary to make the plan's literal instructions compile and behave correctly; neither changes the plan's algorithmic intent (stateless bounded walk, `layoutEnum == 4u` appearing exactly once, no shared state). No scope creep.

## Issues Encountered
- **glslangValidator (WSL-symlinked Windows .exe) could not resolve any `/tmp`-based Linux path**, even after `wslpath -w` conversion to a full Windows path passed as an argument — every invocation reported `unable to open input file`. Root-caused via a minimal-repro (`min.comp` with trivial content still failed identically) and cross-referenced against Phase 3's own documented `03-01-SUMMARY.md` finding: the toolchain requires the shell's **working directory** to be a drvfs (`/mnt/<drive>/...`) mount and the input/output filenames to be **relative** — WSL's interop path translation applies to the process's cwd but not reliably to individual Linux-absolute-path arguments passed to the Windows binary. Once both the Task-1 standalone check and the `makeshader.py` regeneration were run with `cd /mnt/w/gnus/.../compiler` and relative filenames, both worked cleanly (`EXIT:0`).
- The Bash tool's output capture when piping `wsl -e bash -c '...'` through git-bash occasionally interleaved/garbled stdout across the wsl.exe process boundary (observed as truncated or reordered lines in returned tool output). Worked around by having the WSL-side command redirect all output to a log file under the Windows-visible scratchpad directory (`/mnt/c/Users/.../scratchpad/*.log`) and reading that file directly with the `Read` tool instead of relying on the piped stdout — this is a Bash-tool/WSL-interop quirk of this environment, not a defect in `makeshader.py` or `glslangValidator` themselves, and does not affect correctness of the regenerated artifacts (verified independently via `git diff --stat` and grep-based key-count checks on disk).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- SGV2-15's shader-side implementation is complete and live-GPU-validated (uncommitted sanity gate) against `mixed_asymmetric`; the `locateElement`/`locateMixedLeaf` code is ready for Plan 04-02 to make the skip-removal official in `SGFP4VulkanDequantTest.cpp` and complete the SGV2-16 full-14-fixture-sweep test (D-07).
- No blockers. The known `test/op/FP4ModelTest.cpp` pre-existing build blocker (unrelated `milestone`-workstream dead code) was not encountered this session since the standing `.build` tree already had `run_test.out.exe` built and did not require a from-scratch reconfigure; Plan 04-02 should still budget for the Phase-1 temp-stub workaround if a clean rebuild becomes necessary.
- `AllShader.h`/`VulkanShaderMap.cpp` came out byte-identical to the already-committed versions this run (no diff) — this is expected and benign (no shader-key/variant changes were made), not a sign the regeneration silently no-op'd; verified via the grep-count and total-entry-count checks documented above.

## Self-Check: PASSED

- FOUND: source/backend/vulkan/buffer/execution/glsl/sgfp4_dequant.comp
- FOUND: source/backend/vulkan/buffer/compiler/AllShader.cpp
- FOUND: source/backend/vulkan/buffer/shaders/AllShader.h
- FOUND: source/backend/vulkan/buffer/compiler/VulkanShaderMap.cpp
- FOUND: .planning/workstreams/sgfp4-pivot/phases/04-vulkan-decode-adaptive-quadtree-layout-mixed/04-01-SUMMARY.md
- FOUND commit: d82593de

---
*Phase: 04-vulkan-decode-adaptive-quadtree-layout-mixed*
*Completed: 2026-08-25*
