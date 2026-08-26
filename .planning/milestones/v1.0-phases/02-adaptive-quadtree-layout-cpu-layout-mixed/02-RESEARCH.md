# Phase 2: Adaptive Quadtree Layout (CPU, LAYOUT_MIXED) - Research

**Researched:** 2026-08-24
**Domain:** MNN CPU decode of the SGFP4 v2 variable-size quadtree record (`LAYOUT_MIXED`) + error-driven quadtree encoder — pre-order-DFS split-map walking, per-leaf variable payload decode, adaptive subdivision with per-level thresholds / ε mode selection / ternary outlier veto / uniform collapse
**Confidence:** HIGH — all wire-format claims cite the in-repo spec (§6.2/§6.3/§4.3); all codebase claims verified against `include/MNN/SGFP4DequantUtils.hpp`, `tools/fp4/encode_sgfp4.py`, and Phase 1 artifacts at file level. The only `[ASSUMED]` items are the **exemplary** encoder-policy heuristics the spec deliberately leaves open (veto/hysteresis/intermediate thresholds).

## Summary

Phase 2 has two distinct halves that must stay in lockstep: **(1) a CPU decode extension** that teaches `dequant_sgfp4_container_cpu()` to accept the currently-rejected `LAYOUT_MIXED` (enum 4) record by walking a 12-byte/3-word pre-order-DFS split map with an iterative fixed-size stack, and **(2) an encoder extension** inside `tools/fp4/encode_sgfp4.py` that *produces* those records via error-driven quadtree subdivision with per-level thresholds, per-region ε mode selection, a ternary outlier veto, hysteresis, and normative uniform-layout collapse.

Everything needed already exists and is verified. The decode core is a single header-only function (`dequant_sgfp4_container_cpu`) whose uniform path already does 95% of the work — framing, offset-table bounds-checking, leaf-header unpack (`unpack_leaf_header`), dual-mode payload decode (`sgfp4_decode_leaf_payload`), and 16-byte alignment (`sgfp4_align16`). The MIXED branch differs only in three ways: (a) a 12-byte split map sits between `sb_header` and the block headers; (b) leaf count `N` and each leaf's edge size `n` come from the split-map walk, not Table 3; (c) leaves are visited in pre-order DFS (TL/TR/BL/BR), not row-major raster order. The encoder (`encode_sgfp4.py`) already has `encode_leaf`, `pack_leaf_header`, `pack_payload`, and the v2 container writer; it gains a recursive subdivision driver plus split-map emission and the uniform-collapse rule.

The single most consequential correctness trap is **leaf ordering**: uniform layouts store leaves in row-major raster order, but `LAYOUT_MIXED` stores them in pre-order DFS order — and these two orderings *differ* for the same geometric tiling (e.g., 16 leaves of 16×16). The decoder, the encoder's "expected" fixture generation, and the golden enumerator (D-05) must each use the *correct* ordering for the layout in play. A second trap is **output layout is leaf-major (tiled), not row-major**: Phase 1 locked "sequential/linear" output (each leaf's n×n block contiguous), and Phase 2 appends MIXED leaves to that same linear stream in DFS order — so the decoded element ordering for MIXED is DFS-leaf-major, matching neither uniform raster order nor the 64×64 row-major spatial order. This is fine for the phase's success criteria (which test traversal order and payload sizing, not spatial scatter), but it must be understood by every implementer or the round-trip fixtures will silently disagree.

**Primary recommendation:** Add a `sgfp4_walk_quadtree()` split-map walker (fixed-size stack, bounds-checked bit reader, `≤85` bits) plus the split-map constants and a `kSGFP4SplitMapSize = 12` / `kSGFP4SplitMapWords = 3` / `kSGFP4MaxQuadTreeBits = 85` block to `SGFP4DequantUtils.hpp`; branch the existing `dequant_sgfp4_container_cpu()` on `layoutEnum == kSGFP4LayoutMixed` to read the split map, walk it to collect per-leaf `(x, y, n)` and count `N`, then decode block headers and variable-size payloads in traversal order. Extend `encode_sgfp4.py` with `subdivide_macroblock()` (recursive error test → split or accept), `build_split_map()`, and a `classify_layout()` that collapses all-same-size leaf sets to the Table 3 uniform enum. Split the phase into ~2 plans (decode core, then encoder + tests), mirroring Phase 1's structure.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Split-map bit layout / constants / bit-reader | `include/MNN/SGFP4DequantUtils.hpp` (header-only) | — | D-03: wire format stays in one header; header-only keeps Phase 4 GLSL port feasible |
| LAYOUT_MIXED record walk (iterative, fixed stack) | `dequant_sgfp4_container_cpu()` (same header) | — | D-01/D-04: single decode entry point; identical algorithm must port to Phase 4 shader |
| Per-leaf variable payload decode | `sgfp4_decode_leaf_payload()` (reused as-is) | — | Leaf decode is size-parameterized already; only `n` varies per leaf now |
| Error-driven quadtree subdivision + split-map emission | `tools/fp4/encode_sgfp4.py` (extended) | — | D-07: one reference encoder for the whole v2 format |
| Golden traversal verification | independent C++ enumerator (new test helper) | Python reference decoder | D-05: decoder and enumerator are independent implementations of §6.2 |
| Container production / fixture generation | `encode_sgfp4.py --emit-cpp-fixture` (extended) | — | Phase 1 pattern: encoder-generated committed fixtures |
| Malformed split-map rejection (ASVS V5) | `dequant_sgfp4_container_cpu()` bounds checks | — | D-02: strict validation, decode failure on malformed maps |
| Op-level routing / sidecar loading | `CPUSGFP4Dequant` | — | **No change** — already dispatches through `dequant_sgfp4_container_cpu()` `[VERIFIED: source/backend/cpu/CPUSGFP4Dequant.cpp]` |

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SGV2-08 | LAYOUT_MIXED split-map parsing — 12B/3-word LE bitmap, pre-order DFS, TL/TR/BL/BR, nodes ≥8 carry a bit, 4×4 always leaf | §6.2 split-map paragraph + Table 3; bit-reader + iterative walk in Code Examples |
| SGV2-09 | Variable per-leaf decode — headers/payloads in traversal order, edge size n drives payload word count, 16-byte alignment | §6.2 per-leaf headers/payloads; MIXED record decode in Code Examples |
| SGV2-10 | Error-driven quadtree encoder — recursive subdivision, per-level thresholds, ε=0.10 mode selection, outlier veto, uniform collapse | §6.3 + §4.4 (Eq. 5); encoder subdivision in Code Examples |
| SGV2-11 | CPU round-trip tests incl. golden split-map traversal check via `./run_test.out` | Test strategy below; golden enumerator pattern |

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Decoder walks the quadtree **iteratively with an explicit fixed-size stack** (max depth 4 for 64→32→16→8→4; ≤85 nodes total). No recursion — the identical algorithm must port to the Phase 4 GLSL shader, where recursion is impossible. One traversal algorithm for CPU and GPU.
- **D-02:** Split-map is **validated strictly** during the walk: every bit read bounds-checked against the 12-byte/3-word map; leaf sizes/positions verified to tile the macroblock exactly; total leaf payload words cross-checked against record bounds; any malformed split-map = decode failure. Continues Phase 1's ASVS V5 untrusted-container posture.
- **D-03:** Split-map constants (3 words, 85 bits max) and the bit-reader helper go in a **new section of the existing `include/MNN/SGFP4DequantUtils.hpp`** — the wire-format definition stays in one header, consistent with Phase 1.
- **D-04:** LAYOUT_MIXED decode is a **branch inside the existing `dequant_sgfp4_container_cpu()`** (which currently rejects enum 4). Single public decode entry point, stays header-only. `CPUSGFP4Dequant` needs no new routing — it already dispatches through this function.
- **D-05:** The golden pre-order DFS traversal-order check uses an **independent enumerator** — a separate small helper that enumerates expected leaf (x, y, n) coordinates for a given split-map. Decoder and enumerator are independent implementations of the same spec rule (Section 6.2), so a traversal bug in one is caught by the other. Do NOT share one walk between decoder and test.
- **D-06:** Encoder policy knobs are **locked to spec defaults, overridable via CLI flags**: ε=0.10 (T158 chosen iff `e_T158 ≤ (1+ε)·e_FP4`), per-level MSE thresholds tightening with depth (0.01 @64 → 0.0005 @4), ternary outlier veto, hysteresis against oscillation, recursion floor at 4×4. Uniform-layout collapse when all leaves share one size is normative (spec Section 6.3), not optional.
- **D-07:** The quadtree encoder **extends the existing `tools/fp4/encode_sgfp4.py`** — one reference encoder for the whole v2 format; `--selftest` and `--emit-cpp-fixture` stay unified (Phase 1 pattern). No separate quadtree script.
- **D-08:** Mixed-layout fixtures are **encoder-generated and committed** (same pattern as Phase 1's 11 fixtures in `test/op/SGFP4DequantFixtures.h`): deterministic synthetic split-maps covering at minimum all-split, uniform-collapse (encoder must emit the uniform layout, not MIXED), and asymmetric mixed trees.
- **D-09:** Negative tests cover **split-map + size abuse** on top of Phase 1's malformed-container negatives: split bit on a 4×4 node, maps implying >85 nodes, leaves that don't tile the macroblock, truncated variable-size payloads, mixed records lying about leaf sizes.

### the agent's Discretion

- Internal encoder code organization within encode_sgfp4.py (function decomposition, CLI flag naming).
- Exact stack representation in the decoder (array + depth counter vs. small struct), provided it remains fixed-size and recursion-free.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope. (Phase 1's deferred `test/op/FP4ModelTest.cpp` build blocker remains owned by the `milestone` workstream's Phase 4 plan 04-02 and is unchanged by this phase.)
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `include/MNN/SGFP4DequantUtils.hpp` (header-only, `half_float::half`) | in-repo (Phase 1) | Split-map walk + MIXED decode; FP16 scale/bias unpack | Already the single decode entry point `dequant_sgfp4_container_cpu` `[VERIFIED: codebase]`; header-only keeps Phase 4 shader port feasible |
| `tools/fp4/encode_sgfp4.py` (Python 3 + numpy) | in-repo; numpy 2.2.5 `[VERIFIED: shell probe]` | Reference quadtree encoder + fixtures | D-07; numpy already a dependency of the existing encoder — no new package |
| `MNNTestSuite` / `checkVectorByRelativeError` / `FP32Converter` | in-repo (`test/MNNTestSuite.h`, `test/TestUtils.h`) | Test harness + FP16-tolerant comparison | Phase 1's exact pattern `[VERIFIED: test/op/SGFP4DequantTest.cpp]` |
| CMake test glob | in-repo (`test/CMakeLists.txt:12` `GLOB_RECURSE *.cpp`) | Auto-picks up the new `test/op/*Test.cpp` | No CMake edit needed `[VERIFIED: test/CMakeLists.txt]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `sgfp4_decode_leaf_payload()` / `unpack_leaf_header()` / `sgfp4_align16()` / `sgfp4_read_u32_le()` | in-repo | Reused verbatim by the MIXED branch | Per-leaf decode is already size-parameterized |
| `encode_leaf()` / `pack_leaf_header()` / `pack_payload()` / `encode_container()` | in-repo (`encode_sgfp4.py`) | Reused by the subdivision driver | The encoder only adds subdivision + split-map + layout classification |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Iterative fixed-stack walk (chosen) | Recursive C++ DFS | Cleaner CPU code, but Phase 4 GLSL cannot recurse — D-01 forbids it; would force a second divergent GPU implementation |
| Branch inside `dequant_sgfp4_container_cpu` (chosen) | Separate `dequant_sgfp4_mixed_record()` | D-04 locks single entry point; a separate function would still need the same framing checks duplicated |
| Extend `encode_sgfp4.py` (chosen) | New `encode_sgfp4_quadtree.py` | D-07 forbids duplication; container writer / packing / mode-selection would be copy-pasted |
| Independent enumerator (chosen) | Share one walk between decoder & test | D-05 forbids — a shared walk lets the same traversal bug pass silently in both |

**Installation:** No new external packages. numpy (2.2.5) is already present `[VERIFIED: shell probe]`; `half.hpp` and flatc are vendored under `3rd_party/`.

**Version verification:**
```
python 3.13.4   [VERIFIED: shell probe]
numpy  2.2.5    [VERIFIED: shell probe]
cmake  3.29.2   [VERIFIED: shell probe]
```

## Package Legitimacy Audit

Phase 2 installs **no new external packages**. The only third-party runtime dependency (numpy, used by the Python encoder) is already present (2.2.5) and is the same dependency the existing `tools/fp4/quantize_fp4.py` and Phase 1's `encode_sgfp4.py` use. `half_float::half` (`3rd_party/half/half.hpp`) is vendored in-repo. No registry lookup or slopcheck run is applicable.

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Data-Flow Diagram

```
  encode_sgfp4.py (encoder, extended)                        dequant_sgfp4_container_cpu (decoder, extended)
  ─────────────────────────────────────                      ───────────────────────────────────────────
  macroblock float tile [64×64]                                     container bytes (untrusted)
        │                                                                    │
        ▼                                                                    ▼
  subdivide_macroblock(region):                                   parse framing: magic 'SGF4', ver 0x02, B,
     encode both modes (Eq.4.4), ε mode-select                       16B-aligned record_offsets[B]
     compute MSE vs per-level threshold                                │
        │  pass? ── accept as leaf                                    per record: sb_header → layout enum (bits 0–2)
        │  fail?  ── split into TL/TR/BL/BR (floor at 4×4)            │
        │  veto/hysteresis gate the split                            enum 4 (MIXED)? ── read 12B split map (3×u32 LE)
        ▼                                                            │
  classify_layout(leaves):                                           walk: pre-order DFS, TL/TR/BL/BR,
     all leaves one size & tile exactly?                             nodes ≥8 read one bit (1=split, 0=leaf),
       YES → emit Table 3 uniform enum (raster order)                nodes of 4 contribute NO bit (≤85 bits total)
       NO  → emit LAYOUT_MIXED (DFS order) + split map               │
        │                                                            collect leaves: (x, y, n)  → count N
        ▼                                                            │
  build_split_map(leaves): 3 LE words, bit k = word[k/32] bit k%32    ▼
        │                                                  block_headers[N] (4B each, traversal order)
        ▼                                                  pad → payloads[N] (each padded to 16B)
  v2 container bytes ──(--emit-cpp-fixture)──► committed C++ fixture  │
                                              test/op/SGFP4DequantFixtures.h
                                              │
                                              ▼
                                          round-trip: out (DFS leaf-major)
```

### Recommended File Layout (Phase 2 deltas only — Phase 1 files already exist)
```
include/MNN/SGFP4DequantUtils.hpp     # (edit) + split-map constants, bit-reader,
                                      #        sgfp4_walk_quadtree(), MIXED branch in
                                      #        dequant_sgfp4_container_cpu()
tools/fp4/encode_sgfp4.py             # (edit) + subdivide_macroblock(), build_split_map(),
                                      #        classify_layout(), per-level thresholds,
                                      #        veto/hysteresis, uniform collapse;
                                      #        --emit-cpp-fixture gains MIXED cases
test/op/SGFP4DequantFixtures.h        # (regenerated) mixed fixtures appended
test/op/SGFP4DequantTest.cpp          # (edit) + op/sgfp4/mixed_decode: golden traversal,
                                      #        round-trip, negative split-map cases
```

### Pattern 1: Iterative pre-order DFS quadtree walk (fixed-size stack)
**What:** Serialize the split map by visiting nodes in pre-order DFS, quadrant order TL/TR/BL/BR. A node of size ≥8 reads one bit (1 = split into 4, 0 = leaf); a node of size 4 is always a leaf and reads no bit. Max bits = 1 + 4 + 16 + 64 = 85, stored in 3 little-endian uint32 words (bit k = bit `k mod 32` of word `k/32`).
**When to use:** The ONLY way to enumerate MIXED leaves; also the algorithm the Phase 4 GLSL shader must replicate (D-01).
**Example (C++, header-only, no recursion):**
```cpp
// Source: .planning/sgfp4-arxiv-v2.txt §6.2  [CITED]
// Constants (add to SGFP4DequantUtils.hpp):
constexpr size_t  kSGFP4SplitMapWords = 3;    // 3 x uint32 = 12 bytes
constexpr size_t  kSGFP4SplitMapBytes = 12;
constexpr int     kSGFP4MaxQuadTreeBits = 85; // 1+4+16+64
constexpr int     kSGFP4QuadTreeMinSplitSize = 8; // nodes >= 8 carry a bit

// Bit reader over the 3-word LE map; returns false if idx >= 85.
struct SGFP4SplitMapReader {
    const uint32_t* words;  // 3 words, bounds-checked by caller
    int bit = 0;
    bool next(bool& out) {
        if (bit >= kSGFP4MaxQuadTreeBits) return false;
        int w = bit >> 5, b = bit & 31;
        out = ((words[w] >> b) & 1u) != 0;
        ++bit;
        return true;
    }
};

// Fixed-size stack walk. Collects leaves into `leaves` (up to 256).
struct QuadNode { int x, y, n; };
inline bool sgfp4_walk_quadtree(const uint32_t* map, int& leafCount, QuadNode* leaves, int maxLeaves) {
    QuadNode stack[85]; int top = 0;
    stack[top++] = {0, 0, 64};
    SGFP4SplitMapReader r{map};
    leafCount = 0;
    while (top > 0) {
        QuadNode node = stack[--top];
        if (node.n >= kSGFP4QuadTreeMinSplitSize) {
            bool split = false;
            if (!r.next(split)) return false;          // reads past 85 bits -> malformed
            if (split) {
                int h = node.n / 2;
                if (top + 4 > 85) return false;        // defensive bound (can't exceed ~13 in practice)
                stack[top++] = {node.x + h, node.y + h, h}; // BR (pushed last, popped last)
                stack[top++] = {node.x,     node.y + h, h}; // BL
                stack[top++] = {node.x + h, node.y,     h}; // TR
                stack[top++] = {node.x,     node.y,     h}; // TL (popped first)
                continue;
            }
            // fall through: leaf
        }
        if (leafCount >= maxLeaves) return false;
        leaves[leafCount++] = node;                    // 4x4 always lands here, no bit consumed
    }
    return true;
}
```
**Key insight:** max pending stack entries is 13 (push-4-at-once scheme), so an 85-entry array is trivially safe and documents the ≤85-node invariant. Pushing in reverse (BR, BL, TR, TL) makes TL pop first, which is the spec's quadrant order.

### Pattern 2: LAYOUT_MIXED record decode (branch in `dequant_sgfp4_container_cpu`)
**What:** The MIXED branch differs from the uniform branch in three places: the 12-byte split map follows `sb_header`; leaf count/sizes come from the walk; leaves are read in traversal (DFS) order.
**When to use:** Replace the `sgfp4_resolve_uniform_layout` rejection path for enum 4.
**Example (restructured per-record loop):**
```cpp
// Source: spec §6.2 + existing uniform path in SGFP4DequantUtils.hpp  [CITED/VERIFIED]
uint32_t layoutEnum = sbHeader & kSGFP4LayoutEnumMask;
int leafCount = 0; int leafEdge = 0;
size_t blockHeadersStart;
bool isMixed = (layoutEnum == kSGFP4LayoutMixed);
QuadNode leaves[256];
if (isMixed) {
    // split map sits between sb_header and block headers
    if (recStart + 4 + kSGFP4SplitMapBytes > containerSize) return false;
    const uint32_t* map = reinterpret_cast<const uint32_t*>(container + recStart + 4);
    if (!sgfp4_walk_quadtree(map, leafCount, leaves, 256)) return false;
    // D-02 tiling check: sum of leaf areas must be exactly 64*64
    int area = 0; for (int i = 0; i < leafCount; ++i) area += leaves[i].n * leaves[i].n;
    if (area != 4096) return false;
    blockHeadersStart = recStart + 4 + kSGFP4SplitMapBytes;
} else {
    if (!sgfp4_resolve_uniform_layout(layoutEnum, leafCount, leafEdge)) return false;
    blockHeadersStart = recStart + 4;
}
size_t blockHeadersBytes = (size_t)leafCount * 4;
if (blockHeadersBytes > containerSize - blockHeadersStart) return false;
const uint8_t* blockHeaders = container + blockHeadersStart;
size_t payloadsStart = sgfp4_align16(blockHeadersStart + blockHeadersBytes);
size_t payloadCursor = payloadsStart;
for (int leaf = 0; leaf < leafCount; ++leaf) {
    int n = isMixed ? leaves[leaf].n : leafEdge;
    uint32_t header = sgfp4_read_u32_le(blockHeaders + leaf * 4);
    float S, bias; int mode; unpack_leaf_header(header, S, bias, mode);
    int elementCount = n * n;
    int wordsPerLeaf = (mode == 0) ? elementCount / kSGFP4NibblesPerWord
                                   : elementCount / kSGFP4SymbolsPerWord;
    size_t payloadBytes = (size_t)wordsPerLeaf * 4;
    if (payloadBytes > containerSize - payloadCursor) return false;   // truncated payload
    if ((size_t)elementCount > outElementCount - outCursor) return false;
    sgfp4_decode_leaf_payload(reinterpret_cast<const uint32_t*>(container + payloadCursor),
                              n, S, bias, mode, out + outCursor);      // leaf-major append
    outCursor += elementCount;
    payloadCursor += sgfp4_align16(payloadBytes);
}
```
**Note:** `out + outCursor` appends each leaf's n×n block contiguously in DFS order — the canonical MIXED output order (see Pitfall 1).

### Pattern 3: Error-driven encoder subdivision + uniform collapse
**What:** Recursively test a region against a per-level MSE threshold; accept if it passes, else split into quadrants (floor at 4×4), with hysteresis and the ternary outlier veto gating the split. Classify the leaf set; emit uniform (raster order) vs MIXED (DFS order).
**Example (Python, extends `encode_sgfp4.py`):**
```python
# Source: .planning/sgfp4-arxiv-v2.txt §6.3 (exemplary policy)  [CITED]
LEVEL_THRESHOLDS = {64: 0.01, 32: 0.0047, 16: 0.0022, 8: 0.0011, 4: 0.0005}  # [ASSUMED interpolation]

def subdivide_macroblock(tile64, x=0, y=0, n=64):
    """Return list of (x, y, n, mode, S, bias, codes). Recursion floor at n==4."""
    region = tile64[y:y+n, x:x+n]
    mode_fp4, S_fp4, b_fp4, c_fp4, e_fp4 = encode_leaf_with_err(region, MODE_FP4_AFFINE)
    mode_t158, S_t158, b_t158, c_t158, e_t158 = encode_leaf_with_err(region, MODE_T158_AFFINE)
    if n == 4:
        return [(x, y, n) + select_best(...)]          # floor
    mse = min(e_fp4, e_t158) / (n * n)                 # per-element MSE
    if mse <= LEVEL_THRESHOLDS[n]:
        return [(x, y, n) + select_best(...)]          # accept
    h = n // 2
    kids = []
    for (cx, cy) in [(x, y), (x+h, y), (x, y+h), (x+h, y+h)]:   # TL, TR, BL, BR
        kids += subdivide_macroblock(tile64, cx, cy, h)
    return kids
```
The ternary outlier veto (block T158 when the region's extremes exceed what `{bias-S, bias, bias+S}` can represent) and the hysteresis (avoid splitting on marginal improvement) are **exemplary** and not formula-specified — tag `[ASSUMED]` and expose as CLI knobs (D-06).

### Anti-Patterns to Avoid
- **Recursion in the C++ walker:** D-01 forbids it; the same algorithm must port to GLSL.
- **Re-deriving leaf size from a header field:** leaf edge `n` comes *only* from the split-map walk; the per-leaf header contains `(S, bias, mode)` and no size (a "lying leaf size" is exactly the abuse D-09 targets).
- **A single shared walk between decoder and test:** D-05 forbids; the golden enumerator must be an independent implementation.
- **Hard-coded split-map offsets:** use `kSGFP4SplitMapBytes` / `kSGFP4SplitMapWords` / `kSGFP4MaxQuadTreeBits` constants (CLAUDE.md named-constant discipline).
- **Emitting MIXED for a uniform-collapse leaf set:** the uniform-collapse rule is normative (§6.3), not optional (Success Criterion 2).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FP16 (S, bias) → float | hand-rolled half decode | `half_float::half` via `unpack_leaf_header()` | Vendored, tested, already on include path; the reserved-bit/subnormal cases are subtle `[VERIFIED: 3rd_party/half]` |
| 16-byte alignment | manual masking per site | `sgfp4_align16()` (existing) | Single named helper already used by the uniform path |
| Little-endian u32 reads | manual byte shifts per site | `sgfp4_read_u32_le()` (existing) | Already bounds-checked-by-convention; keeps the MIXED branch consistent |
| Leaf payload decode | new per-size decoder | `sgfp4_decode_leaf_payload()` (existing) | Already parameterized by `leafEdge`; only the caller's `n` varies |

**Key insight:** The MIXED branch is a *driver* change, not a *codec* change. Every primitive (header unpack, payload decode, alignment, framing) already exists and is Phase-1-tested; hand-rolling any of them would both duplicate code and risk divergence from the uniform path.

## Common Pitfalls

### Pitfall 1: Uniform raster order ≠ MIXED DFS order (the highest-risk bug)
**What goes wrong:** The decoder (or encoder "expected") walks MIXED leaves in DFS order but the fixture comparison assumes raster order — or vice versa — producing a silent full-tensor mismatch that still "round-trips" internally.
**Why it happens:** §6.2 uses two different orderings: "row-major raster order" for uniform layouts and "pre-order DFS" for MIXED. For a 4×4 tile grid of 16×16 leaves, raster = (0,0)(0,1)(0,2)(0,3)(1,0)… while DFS = TLTL,TLTR,TLBL,TLBR,TRTL… — clearly different.
**How to avoid:** The golden enumerator (D-05) must be the single source of truth for DFS order; the encoder's `classify_layout()` must emit raster order when collapsing to uniform and DFS order when emitting MIXED; the "expected" fixture values must be generated in the *same* order the decoder emits. Test both a uniform-collapse fixture (encoder emits uniform enum, decoder takes the uniform raster path) and a true MIXED fixture.
**Warning signs:** Round-trip passes but golden-traversal check fails; or a uniform-collapse fixture decodes to a permuted result.

### Pitfall 2: Output is leaf-major (tiled), not 64×64 row-major
**What goes wrong:** Implementers assume `out[0..4095]` is the row-major macroblock; actually each leaf's n×n block is contiguous (Phase 1's "sequential" order), and for MIXED the leaf sequence is DFS.
**Why it happens:** Phase 1 locked sequential/linear output; the container is a decode-spec, and the spatial scatter back to `[O,I]` row-major is deferred (SGV2-17, out of this workstream).
**How to avoid:** Treat `out` strictly as "concatenation of leaf reconstructions in traversal order". The golden enumerator's `(x, y, n)` coordinates are for tiling *validation*, not for scattering output. Don't introduce a scatter in this phase.

### Pitfall 3: Split-map bit indexing / 12-byte placement
**What goes wrong:** Off-by-one in `bit k = word[k/32] bit k%32`; or placing the split map at the wrong offset (it follows `sb_header`, so block headers start at `recStart + 4 + 12`).
**Why it happens:** Three LE words + the "≥8 nodes carry bits, 4×4 don't" rule + the extra 12 bytes shift every subsequent offset.
**How to avoid:** Use the `SGFP4SplitMapReader` pattern with a hard 85-bit cap; keep `blockHeadersStart = recStart + 4 + kSGFP4SplitMapBytes` for MIXED (vs `recStart + 4` for uniform). Negative tests: a bit set past 85, and a 4×4 node that a naive walker would try to read a bit for.

### Pitfall 4: 16-byte alignment after the variable-size split-map region
**What goes wrong:** The pad between block headers and payloads is computed wrong because the block-header start moved by +12 bytes, so `payloadsStart` is misaligned and every payload read lands 4 bytes off.
**Why it happens:** `payloadsStart = sgfp4_align16(blockHeadersStart + blockHeadersBytes)` is correct only if `blockHeadersStart` is recomputed for the MIXED layout. `recStart` is 16-aligned (record offsets are 16-byte multiples), so `sgfp4_align16` on absolute offsets stays consistent.
**How to avoid:** Reuse the existing `sgfp4_align16` on absolute offsets (it works because `recStart` is 16-aligned); add a fixture with an odd leaf count (e.g., a 64→32 split with one 32×32 leaf and three 16×16 leaves → 4 leaves, no pad; vs a tree with 2 leaves → 8-byte headers, needs 8-byte pad) to exercise padding.

### Pitfall 5: Leaf count / area validation (D-02 "tile the macroblock exactly")
**What goes wrong:** A malformed map that produces leaves whose areas don't sum to 4096, or whose block-header count exceeds the record, is accepted and reads out of bounds.
**Why it happens:** The walk alone doesn't guarantee the block headers/payloads are present and correctly sized.
**How to avoid:** After the walk: (1) sum of `n²` must equal 4096; (2) `blockHeadersBytes = leafCount*4` bounds-checked; (3) each payload `wordsPerLeaf*4` bounds-checked against the record; (4) total decoded elements bounded against `outElementCount` (already in the uniform path).

### Pitfall 6: The pre-existing `test/op/FP4ModelTest.cpp` build blocker
**What goes wrong:** `run_test.out` cannot build from scratch because `test/op/FP4ModelTest.cpp` (dead code after an early `return true;`, undeclared identifiers, mismatched braces) fails to compile in the monolithic MNNTestSuite binary.
**Why it happens:** Committed at `cffaf4bd` on the `milestone` workstream; out of this workstream's scope (owned by `milestone` Phase 4 plan 04-02).
**How to avoid (Phase 1's verified workaround):** For local build/test verification only, temporarily replace the file with a neutral stub, build + run the full suite, then restore byte-for-byte (`git diff` shows zero changes) before any commit. **Never commit the stub.** `[VERIFIED: 01-affine-.../deferred-items.md]`

## Code Examples

### Golden traversal enumerator (D-05 — independent of the decoder walk)
```cpp
// Source: spec §6.2; INDEPENDENT implementation from sgfp4_walk_quadtree  [CITED]
// Enumerate expected leaf (x, y, n) in pre-order DFS. A separate recursive-free
// formulation (e.g., explicit depth-indexed arrays) so a bug in the decoder's
// stack push/pop cannot be shared with this oracle.
struct LeafExpect { int x, y, n; };
void enumerateExpected(const uint32_t* map, LeafExpect* out, int& count) {
    // depth-first by quadrant expansion, reading the SAME bit stream with its own
    // independent bit accounting; asserts 4x4 nodes consume no bit.
    // ... independent from SGFP4SplitMapReader ...
}
```
The test builds a MIXED container whose leaf k's `(S, bias)` encodes a unique marker `f(k)`, decodes, and asserts each n×n block equals the marker of the leaf the enumerator says should occupy that traversal position.

### Encoder split-map emission (Python)
```python
# Source: spec §6.2; bit k = word[k//32] bit k%32, LE  [CITED]
def build_split_map(leaves):   # leaves in DFS order with (x, y, n)
    bits = []
    def emit(node):            # recursive over the tree, emitting 1/0 per >=8 node
        if node.n >= 8:
            is_split = any(child of node is subdivided)
            bits.append(1 if is_split else 0)
            if is_split:
                for q in ("TL", "TR", "BL", "BR"): emit(child)
        # n==4: no bit
    words = [0, 0, 0]
    for k, b in enumerate(bits):
        words[k // 32] |= b << (k % 32)
    assert len(bits) <= 85
    return struct.pack("<3I", *words)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Uniform fixed-leaf layouts only (Phase 1) | Quadtree-adaptive variable-size `LAYOUT_MIXED` | Phase 2 (this phase) | Decode must derive leaf geometry from a split map; encoder gains subdivision |
| E2M1 "Ultra FP4" float microcode | SGFP4 v2 affine integer dual-mode | Phase 1 | Unchanged here — Phase 2 only adds the adaptive layout on top |

**Deprecated/outdated:**
- The `sgfp4_resolve_uniform_layout()` rejection of enum 4 is retired *only* for `kSGFP4LayoutMixed` — the uniform enums and the `>= 6` malformed rejection stay. The error-driven quadtree is imported from image/video codec practice (JPEG2000/HEVC/AV1), per spec §2 — not a novel format, a novel *container* application `[CITED: spec §2]`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Intermediate per-level MSE thresholds (32/16/8) interpolate geometrically 0.01 → 0.0005 | Standard Stack / Pattern 3 | Encoder splits more/less aggressively; round-trip still passes but fixture leaf sets differ. Mitigated by making thresholds CLI-overridable (D-06) |
| A2 | Per-level thresholds are per-element MSE (total MSE / n²), matching Eq.5's per-block L2 error convention | Pattern 3 | A total-MSE reading would split almost everything; a wrong reading changes output layouts. Flag for user confirmation |
| A3 | Ternary outlier veto = block T158 when `max|w − bias|` exceeds a small multiple (e.g. 2–4×) of `mean|w − bias|` | Pattern 3 | Spec §6.3 gives no formula ("outlier veto" only). Any reasonable rule conforms; exact factor is discretionary |
| A4 | Hysteresis = only split when child total error improves the parent error by a margin (e.g. < (1−δ)·parent, δ≈0.05) | Pattern 3 | Spec gives no formula. Oscillation is self-correcting in a deterministic encoder, so risk is low |
| A5 | The walk's max pending stack is 13, so an 85-entry fixed array is safe | Pattern 1 | Derived, but a defensive `top+4 > 85` bound (shown) removes any risk |
| A6 | MIXED canonical output order = DFS leaf-major (append in traversal order), matching Phase 1's "sequential" stream | Pitfall 2 | The CONTEXT explicitly asserts this ("MIXED records append leaves in traversal order to the same linear output stream"), so confidence is high; if a future consumer needs spatial row-major, a separate scatter is required (SGV2-17, out of scope) |

## Open Questions (RESOLVED)

1. **Exact per-level threshold table and MSE interpretation (total vs per-element)** — RESOLVED: adopt per-element MSE with `[ASSUMED]` geometric interpolation of the 32/16/8 values (A1/A2), exposed as the `--level-thresholds` CLI table; the plans implement this.
   - What we know: spec §6.3 says "MSE ≤ 0.01 at 64 down to ≤ 0.0005 at 4" with "e.g." (exemplary); D-06 locks these endpoints and CLI overridability.
   - What's unclear: the 32/16/8 values and whether MSE is normalized by leaf area.
   - Recommendation: adopt per-element MSE with geometric interpolation (A1/A2); expose the full table as `--level-thresholds` CLI; flag in the plan for a one-line user confirmation.

2. **Ternary outlier veto and hysteresis formulas** — RESOLVED: implement documented `[ASSUMED]` heuristics (A3/A4) as CLI knobs (`--veto-factor`, `--hysteresis-delta`); the plans implement this as exemplary-not-normative.
   - What we know: spec names both but specifies neither.
   - What's unclear: exact constants.
   - Recommendation: implement simple, documented heuristics (A3/A4) as CLI knobs; note in the plan that they are exemplary, not normative.

3. **Golden traversal test mechanics** — RESOLVED: use marker-encoding (option a) — per-leaf marker S/bias compared block-by-block, no extra API surface; the enumerator drives which marker each output block should hold. 02-01 Task 2 adds a hand-built traversal golden; 02-02 Task 3 adds the independent-enumerator golden.
   - What we know: D-05 mandates an independent enumerator; the decoder only exposes the concatenated float stream.
   - What's unclear: whether to (a) encode per-leaf marker S/bias and compare block-by-block, or (b) also expose a debug/trace helper returning `(x,y,n)`.
   - Recommendation: use marker-encoding (a) — no extra API surface; the enumerator drives which marker each output block should hold.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3 | `encode_sgfp4.py --selftest` / `--emit-cpp-fixture` | ✓ | 3.13.4 | — |
| numpy | encoder | ✓ | 2.2.5 | — |
| cmake | `run_test.out` build | ✓ | 3.29.2 | — |
| C++17 toolchain (MSVC 19.44) | `run_test.out` build | ✓ | (Phase 1 built it) | — |
| `half_float::half` | decoder | ✓ (vendored) | — | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none.

> Note: the `test/op/FP4ModelTest.cpp` compile blocker is a *source* issue, not a tool availability issue — see Pitfall 6. Build verification requires the Phase 1 temporary-local-stub workaround (never committed).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | MNNTestSuite (`run_test.out`, monolithic binary) + Python `--selftest` oracle |
| Config file | none (tests auto-globbed via `test/CMakeLists.txt:12` `GLOB_RECURSE`) |
| Quick run command | `./run_test.out op/sgfp4` (mixed_decode) after build |
| Full suite command | `./run_test.out` (expect 375 + new cases green; requires the FP4ModelTest stub workaround to build) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SGV2-08 | Split-map walk: DFS order, TL/TR/BL/BR, 4×4 no bit, ≤85 bits | unit | `./run_test.out op/sgfp4` (golden traversal) | ❌ Wave 0 (add to `SGFP4DequantTest.cpp`) |
| SGV2-08/09 | Variable per-leaf decode + 16-byte alignment (n²/8 or n²/16 words) | unit | `./run_test.out op/sgfp4` (mixed round-trip) | ❌ Wave 0 |
| SGV2-09 | Malformed split-maps rejected (split-on-4×4, >85 bits, non-tiling, truncated, lying sizes) | unit | `./run_test.out op/sgfp4` (negatives) | ❌ Wave 0 |
| SGV2-10 | Encoder: ε mode selection, veto, uniform collapse | unit (Python) | `python tools/fp4/encode_sgfp4.py --selftest` | ✅ exists (extend) |
| SGV2-11 | Mixed/adaptive round-trip within per-level thresholds | integration | `./run_test.out op/sgfp4` + `--selftest` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `python tools/fp4/encode_sgfp4.py --selftest` (encoder) / `./run_test.out op/sgfp4` (decoder)
- **Per wave merge:** `./run_test.out op/sgfp4` + `./run_test.out op/fp4` (E2M1 regression, SC#5)
- **Phase gate:** full `./run_test.out` green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `test/op/SGFP4DequantTest.cpp` — add `op/sgfp4/mixed_decode` (golden traversal, mixed round-trip, negative split-map cases)
- [ ] `test/op/SGFP4DequantFixtures.h` — regenerate with mixed fixtures (all-split, uniform-collapse, asymmetric)
- [ ] `tools/fp4/encode_sgfp4.py` — extend `--selftest` to cover mixed/adaptive round-trip

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Bounds-checked split-map bit reads (85-bit cap), leaf-count/area validation, per-payload bounds checks, decode-failure on malformed input (D-02) |
| V6 Cryptography | no | — |

### Known Threat Patterns for SGFP4 v2 LAYOUT_MIXED decode

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed split map forces reads past the 3-word map (bit index ≥85) | Denial of Service / Tampering | `SGFP4SplitMapReader` hard-caps at 85 bits; return false |
| Split map implies leaf set that does not tile the 64×64 macroblock (area ≠4096, or leaf count >256) | Tampering | Post-walk area-sum and leaf-count validation (D-02) |
| Truncated variable-size payload (record too short for the implied leaf sizes) | Denial of Service | Per-payload `wordsPerLeaf*4` bounds check against `containerSize` before each read |
| Mixed record "lying about leaf sizes" (headers/payloads inconsistent with the map) | Tampering | Leaf `n` derived only from the split map; payload word count recomputed from that `n`; cross-checked against record bounds |
| Oversized `outElementCount` from attacker-controlled `B` | Denial of Service | Existing Phase 1 bound: `elementCount > outElementCount - outCursor` → reject |

**Note:** Security posture continues Phase 1's untrusted-container model (the container is attacker-influenced bytes). No new attack surface beyond the split map itself, which D-02 handles.

## Sources

### Primary (HIGH confidence)
- `.planning/sgfp4-arxiv-v2.txt` §6.2 (split-map serialization, block order, per-leaf headers Eq. 6, per-leaf payloads) — normative wire format
- `.planning/sgfp4-arxiv-v2.txt` §6.3 (exemplary v2 encoder policy) — per-level thresholds, ε mode selection, veto, hysteresis, uniform collapse
- `.planning/sgfp4-arxiv-v2.txt` §4.3/§4.4 (payload packing, affine encode math, Eq. 5 mode selection)
- `include/MNN/SGFP4DequantUtils.hpp` — existing decode core, constants, `dequant_sgfp4_container_cpu`, `sgfp4_resolve_uniform_layout` rejection point
- `tools/fp4/encode_sgfp4.py` — existing encoder structure to extend (`encode_leaf`, `pack_payload`, `encode_container`, `--selftest`, `--emit-cpp-fixture`)
- `source/backend/cpu/CPUSGFP4Dequant.cpp` — confirms no routing change needed (D-04)
- `test/CMakeLists.txt:12` — `GLOB_RECURSE` test pickup confirmed
- `.planning/workstreams/sgfp4-pivot/phases/01-affine-dual-mode-decode-core-cpu-uniform-layouts/01-02-PLAN.md` + `01-02-SUMMARY.md` + `deferred-items.md` — Phase 1 decisions, fixture pattern, build-blocker workaround

### Secondary (MEDIUM confidence)
- `.planning/workstreams/sgfp4-pivot/REQUIREMENTS.md` — SGV2-08..11 normative text
- `.planning/workstreams/sgfp4-pivot/ROADMAP.md` — locked roadmap notes 1–6

### Tertiary (LOW confidence)
- Training knowledge of quadtree/hysteresis/veto heuristics (image-codec subdivision practice) — tagged `[ASSUMED]` in the Assumptions Log

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; all reuse targets verified at file level
- Architecture: HIGH — MIXED decode is a driver change over Phase-1-tested primitives; walk algorithm directly from §6.2
- Pitfalls: HIGH — the two ordering pitfalls (raster vs DFS, leaf-major output) are verified against spec text and Phase 1 code

**Research date:** 2026-08-24
**Valid until:** 2026-09-24 (stable — spec is in-repo and frozen for this workstream)
