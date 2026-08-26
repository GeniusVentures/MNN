# Phase 2: Adaptive Quadtree Layout (CPU, LAYOUT_MIXED) - Pattern Map

**Mapped:** 2026-08-24
**Files analyzed:** 4 (all extend/regenerate existing Phase 1 artifacts)
**Analogs found:** 4 / 4

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `include/MNN/SGFP4DequantUtils.hpp` (extend) | utility (header-only decode core) | transform (bytes → floats, sequential) | itself — Phase 1 constants + `dequant_sgfp4_container_cpu` | exact |
| `tools/fp4/encode_sgfp4.py` (extend) | utility/tool (reference encoder + oracle) | transform (floats → bytes) + file I/O | itself — Phase 1 `encode_leaf`/`pack_payload`/`encode_container` | exact |
| `test/op/SGFP4DequantTest.cpp` (extend) | test | request-response (call decode → assert) | itself — Phase 1 layered test class | exact |
| `test/op/SGFP4DequantFixtures.h` (regenerate) | test data (static) | static data | itself — generated fixture table + `emit_cpp_fixture` generator | exact |

**Notes on classification:** All four files are *extensions* of Phase 1 artifacts, so the "closest analog" is the file's own current content — the pattern to copy is the established structure *inside* each file. The one genuinely new code shape is the **golden traversal enumerator** (D-05), whose analog is the *independent implementation* pattern already exemplified by the Python reference decoder (`decode_container_ref`); see Shared Patterns.

---

## Pattern Assignments

### `include/MNN/SGFP4DequantUtils.hpp` (utility, transform)

**Analog:** itself — Phase 1 header-only decode core.

The MIXED extension adds exactly three things per D-03/D-04, each with an existing in-file template:

**A. Named-constant block pattern** (lines 20–62) — new split-map constants go in a new section next to the existing layout enum:

```cpp
// sb_header (spec section 6.2): layout enum lives in bits 0-2.
constexpr uint32_t kSGFP4LayoutEnumMask = 0x7u;

// Table 3 uniform-layout map (Phase 1 subset -- LAYOUT_MIXED is Phase 2).
enum SGFP4UniformLayout : uint32_t {
    kSGFP4LayoutUniform64 = 0, // N=1,   leaf edge 64
    ...
    kSGFP4LayoutMixed     = 4, // Phase 2 -- rejected here
    ...
};
```

Copy this discipline verbatim: `constexpr` with a spec-section comment, PascalCase `kSGFP4*` names. Add `kSGFP4SplitMapBytes = 12`, `kSGFP4SplitMapWords = 3`, `kSGFP4MaxQuadTreeBits = 85`, `kSGFP4QuadTreeMinSplitSize = 8` (CLAUDE.md named-constant rule; no magic numbers).

**B. Little-endian read + alignment helper pattern** (lines 84–97) — the bit reader reuses this:

```cpp
inline size_t sgfp4_align16(size_t x) {
    return (x + (kSGFP4Alignment - 1)) & ~(kSGFP4Alignment - 1);
}

inline uint32_t sgfp4_read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}
```

**C. Bounds-checked per-record decode loop** (lines 270–326) — the MIXED branch slots in between the `layoutEnum` extraction and `blockHeadersStart`:

```cpp
uint32_t sbHeader   = sgfp4_read_u32_le(container + recStart);
uint32_t layoutEnum = sbHeader & kSGFP4LayoutEnumMask;
int leafCount = 0;
int leafEdge  = 0;
if (!sgfp4_resolve_uniform_layout(layoutEnum, leafCount, leafEdge)) {
    return false;
}

size_t blockHeadersStart = recStart + 4;
size_t blockHeadersBytes = static_cast<size_t>(leafCount) * 4;
if (blockHeadersBytes > containerSize - blockHeadersStart) {
    return false;
}
...
for (int leaf = 0; leaf < leafCount; ++leaf) {
    uint32_t header = sgfp4_read_u32_le(blockHeaders + leaf * 4);
    float S = 0.0f, bias = 0.0f;
    int mode = 0;
    unpack_leaf_header(header, S, bias, mode);

    int elementCount = leafEdge * leafEdge;
    int wordsPerLeaf = (mode == 0) ? (elementCount / kSGFP4NibblesPerWord)
                                    : (elementCount / kSGFP4SymbolsPerWord);
    size_t payloadBytes = static_cast<size_t>(wordsPerLeaf) * 4;
    if (payloadBytes > containerSize - payloadCursor) {
        return false;
    }
    if (static_cast<size_t>(elementCount) > outElementCount - outCursor) {
        return false;
    }
    sgfp4_decode_leaf_payload(reinterpret_cast<const uint32_t*>(container + payloadCursor), leafEdge, S,
                              bias, mode, out + outCursor);
    outCursor += static_cast<size_t>(elementCount);
    payloadCursor += sgfp4_align16(payloadBytes);
}
```

**The Phase 2 change (per RESEARCH Pattern 1/2):** branch on `layoutEnum == kSGFP4LayoutMixed` before the `sgfp4_resolve_uniform_layout` call; when mixed, read the 12-byte map at `recStart + 4`, run `sgfp4_walk_quadtree()` to fill a `QuadNode leaves[256]`, validate area sum == 4096 (D-02), set `blockHeadersStart = recStart + 4 + kSGFP4SplitMapBytes`, and drive the existing loop with per-leaf `n = leaves[leaf].n`. Every leaf primitive (`unpack_leaf_header`, `sgfp4_decode_leaf_payload`, `sgfp4_align16`) is reused verbatim — this is a *driver* change, not a *codec* change.

**Error handling pattern:** return `false` on every malformed/out-of-bounds condition before any dereference (ASVS V5 posture already in place — every `return false` above).

---

### `tools/fp4/encode_sgfp4.py` (utility/tool, transform + file I/O)

**Analog:** itself — Phase 1 reference encoder. The quadtree extension reuses every existing primitive and adds only subdivision + split-map + layout classification (D-07).

**A. Constants block pattern** (lines 33–95) — add `LEVEL_THRESHOLDS`, split-map constants, and the ε/veto/hysteresis knobs here:

```python
LAYOUT_MIXED = 4       # quadtree -- Phase 2, not emitted by this encoder
LAYOUT_FULL_4X4 = 5
LAYOUT_TABLE = {
    LAYOUT_UNIFORM_64: (1, 64),
    ...
}
MACROBLOCK_EDGE = 64
MODE_FP4_AFFINE = 0
MODE_T158_AFFINE = 1
MODE_SELECT_EPS = 0.10        # default eps in [0.05, 0.20] (Eq. 5)
```

New: `SPLIT_MAP_WORDS = 3`, `SPLIT_MAP_BYTES = 12`, `MAX_QUADTREE_BITS = 85`, `LEVEL_THRESHOLDS = {64: 0.01, ...}`, and CLI-exposed veto/hysteresis defaults (D-06; tag interpolation as `[ASSUMED]` per RESEARCH A1/A2).

**B. Per-leaf encode + mode-select pattern** (lines 124–184) — `subdivide_macroblock()` reuses `encode_leaf_fp4`/`encode_leaf_t158`/`select_mode`:

```python
def select_mode(err_fp4, err_t158, eps=MODE_SELECT_EPS):
    """Eq. 5: choose T158 iff e_T158 <= (1 + eps) * e_FP4."""
    return MODE_T158_AFFINE if err_t158 <= (1.0 + eps) * err_fp4 else MODE_FP4_AFFINE

def encode_leaf(w, force_mode=None):
    s_fp4, b_fp4, c_fp4, e_fp4 = encode_leaf_fp4(w)
    s_t158, b_t158, c_t158, e_t158 = encode_leaf_t158(w)
    mode = force_mode if force_mode is not None else select_mode(e_fp4, e_t158)
    ...
```

The subdivision driver calls `encode_leaf` (or its two components to get both errors) per region, compares per-element MSE `min(e_fp4, e_t158) / (n*n)` against `LEVEL_THRESHOLDS[n]`, and recurses into TL/TR/BL/BR (floor at n==4).

**C. Packing + container writer pattern** (lines 188–295) — `build_split_map()` and `classify_layout()` feed the existing `encode_macroblock`/`encode_container`:

```python
def pack_leaf_header(mode, S, bias):
    S_bits = float_to_half_bits(clip_fp16_range(S))
    bias_bits_masked = float_to_half_bits(clip_fp16_range(bias)) & LEAF_HEADER_BIAS_MASK
    header = ((S_bits << LEAF_HEADER_SCALE_SHIFT) | bias_bits_masked | (mode & LEAF_HEADER_MODE_BIT))
    return header & 0xFFFFFFFF, S_bits, bias_bits_masked

def encode_macroblock(leaves_weights, layout_enum, force_mode=None):
    N, n = LAYOUT_TABLE[layout_enum]
    ...
    sb_header = layout_enum & LAYOUT_ENUM_MASK
    block_headers_bytes = b''.join(struct.pack('<I', h) for h in header_words)
    pre_payload_len = 4 + len(block_headers_bytes)
    pad_len = align16(pre_payload_len) - pre_payload_len
    record = struct.pack('<I', sb_header) + block_headers_bytes + b'\x00' * pad_len
```

**Phase 2 change:** `encode_macroblock` gains a MIXED path — when leaves are non-uniform, prepend the 12-byte split map after `sb_header` (i.e. `record = struct.pack('<I', sb_header) + split_map + block_headers_bytes + pad`), and emit leaves in **pre-order DFS order** (Pitfall 1: uniform = raster order, MIXED = DFS order). `classify_layout()` collapses all-same-size leaf sets to the Table 3 uniform enum (normative, Success Criterion 2).

**D. Selftest + fixture-gen + CLI pattern** (lines 394–618) — the new `--level-thresholds` / veto / hysteresis flags follow the existing `argparse` shape; `build_fixture_cases()` gains all-split / uniform-collapse / asymmetric MIXED cases; `emit_cpp_fixture()` is reused unchanged:

```python
def main():
    parser = argparse.ArgumentParser(description="SGFP4 v2 ... reference encoder")
    parser.add_argument("--selftest", action="store_true", help="...")
    parser.add_argument("--emit-cpp-fixture", metavar="PATH", help="...")
    args = parser.parse_args()
    if not args.selftest and not args.emit_cpp_fixture:
        parser.print_help()
        sys.exit(1)
```

---

### `test/op/SGFP4DequantTest.cpp` (test, request-response)

**Analog:** itself — Phase 1 layered test class. Add an `op/sgfp4/mixed_decode` case (or a new `mixed` layer inside the same class) mirroring the existing structure.

**A. File header + tolerance + class dispatch pattern** (lines 1–48):

```cpp
#include "MNN/SGFP4DequantUtils.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "SGFP4DequantFixtures.h"

namespace {
constexpr float kFixtureRelativeTolerance = 1e-4f;
} // namespace

class SGFP4DequantTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        if (!testFixtureRoundTrip()) { return false; }
        if (!testTernaryReservedSymbol()) { return false; }
        ...
        MNN_PRINT("SGFP4DequantTest: all layers PASSED\n");
        return true;
    }
private:
    ...
};
```

**B. Round-trip layer pattern** (lines 56–79) — mixed round-trip copies this exact shape:

```cpp
bool testFixtureRoundTrip() {
    for (size_t i = 0; i < sgfp4_fixtures::kFixtureCount; ++i) {
        const auto& fixture = sgfp4_fixtures::kFixtures[i];
        std::vector<float> out(fixture.expectedCount, 0.0f);
        bool ok = MNN::dequant_sgfp4_container_cpu(fixture.container, fixture.containerSize, out.data(),
                                                    fixture.expectedCount);
        if (!ok) {
            MNN_ERROR("... decode returned false\n");
            return false;
        }
        if (!checkVectorByRelativeError<float>(out.data(), fixture.expected,
                                                static_cast<int>(fixture.expectedCount),
                                                kFixtureRelativeTolerance)) {
            MNN_ERROR("... round-trip mismatch\n");
            return false;
        }
    }
    return true;
}
```

**C. Negative-test layer pattern** (lines 178–266) — new split-map/size abuse negatives (D-09) copy this mutation-by-located-byte style. Note the existing LAYOUT_MIXED rejection case (lines ~262–274) **must be inverted**: Phase 2 now *accepts* a well-formed MIXED record; the "reject" cases become *malformed* split-maps (split-on-4×4, >85 bits, non-tiling, truncated payload, lying sizes):

```cpp
// (d) LAYOUT_MIXED (enum 4) -- Phase 2 layout, must be rejected here.
{
    std::vector<uint8_t> bad = good;
    ...
    bad[recStart] = static_cast<uint8_t>((bad[recStart] & ~MNN::kSGFP4LayoutEnumMask) |
                                          MNN::kSGFP4LayoutMixed);
    if (MNN::dequant_sgfp4_container_cpu(bad.data(), bad.size(), scratch.data(), scratch.size())) {
        MNN_ERROR("SGFP4DequantTest: LAYOUT_MIXED (Phase 2) container was accepted\n");
        return false;
    }
}
```

This existing case (`bad[recStart]` mutated to enum 4 but *no valid split map following*) becomes the first malformed-split-map negative; the new tests add valid-framing-with-bad-map variants.

**D. Test registration** (line 420) — append a new registration string (don't rename the existing one):

```cpp
MNNTestSuiteRegister(SGFP4DequantTest, "op/sgfp4/uniform_decode");
// Phase 2 adds: MNNTestSuiteRegister(SGFP4MixedDecodeTest, "op/sgfp4/mixed_decode");
```

**Golden traversal enumerator (D-05):** a *new, separate* small helper — do NOT reuse `sgfp4_walk_quadtree`. Its closest analog is the independence pattern in the Python reference decoder (see Shared Patterns below); the test encodes a unique `f(k)` marker in leaf `k`'s `(S, bias)` and asserts each output n×n block holds the marker of the leaf the enumerator says occupies that DFS position.

---

### `test/op/SGFP4DequantFixtures.h` (test data, static)

**Analog:** itself — generated fixture header + its generator `emit_cpp_fixture` (Python, lines 520–578).

**Fixture struct + table pattern** (header lines 12–22 and the trailing table):

```cpp
struct Fixture {
    const char* name;
    const unsigned char* container;
    size_t containerSize;
    int dimO;
    int dimI;
    int mode;
    int layout;
    const float* expected;
    size_t expectedCount;
};

static const Fixture kFixtures[] = {
    {"mode0_uniform64", kFixture_mode0_uniform64_data, sizeof(...), 64, 64, 0, 0, ..., ...},
    ...
};
static const size_t kFixtureCount = sizeof(kFixtures) / sizeof(kFixtures[0]);
```

**Generator pattern** (`emit_cpp_fixture`, Python lines 520–578) — mixed fixtures are appended by adding cases in `build_fixture_cases()` with `layout = LAYOUT_MIXED (4)`; the `struct.pack`/hex emission code is reused unchanged. New fixture names per D-08: e.g. `mixed_allsplit`, `uniform_collapse` (encoder must emit the uniform enum, not MIXED — Success Criterion 2), `mixed_asymmetric`.

---

## Shared Patterns

### 1. Named-constants discipline (no magic numbers)
**Source:** `include/MNN/SGFP4DequantUtils.hpp` lines 20–62 and `tools/fp4/encode_sgfp4.py` lines 33–95.
**Apply to:** both halves of Phase 2.
Every wire-format literal is a named `constexpr` (C++) / module-level constant (Python) with a spec-section comment. New Phase 2 literals (`kSGFP4SplitMapBytes`, `kSGFP4MaxQuadTreeBits`, `LEVEL_THRESHOLDS`, ε) must follow this. CLAUDE.md explicitly forbids magic numbers.

### 2. Bounds-check-before-read (ASVS V5 untrusted-container posture)
**Source:** `include/MNN/SGFP4DequantUtils.hpp` — every `return false` in `dequant_sgfp4_container_cpu` (lines 214–326).
**Apply to:** the MIXED decode branch and the bit reader.
Every split-map bit read, block-header read, and payload read is bounds-checked against `containerSize` before it happens. D-02 adds: 85-bit cap, area-sum == 4096, per-payload word-count bounds check, leaf `n` derived *only* from the map.

### 3. Independent-implementation oracle (D-05, and the existing Python↔C++ lockstep)
**Source:** `tools/fp4/encode_sgfp4.py` `decode_container_ref` (lines 350–388) — an independent Python reference decode used by `--selftest` to prove the *bytes* round-trip, not just in-process arrays.
**Apply to:** the golden traversal enumerator in `SGFP4DequantTest.cpp` (D-05) and the Python `--selftest` extension (SGV2-10/11).
The enumerator is a separate, recursion-free formulation of §6.2 that shares no walk with `sgfp4_walk_quadtree`, so a traversal bug in one is caught by the other.

### 4. FP16 handling via vendored `half` / numpy float16
**Source:** `unpack_leaf_header` (header lines 139–154) uses `half_float::half` + `std::memcpy`; Python `float_to_half_bits`/`half_bits_to_float` (lines 108–117).
**Apply to:** both halves — no hand-rolled FP16 decode/encode anywhere (RESEARCH "Don't Hand-Roll").

### 5. 16-byte alignment via named helper, applied to *absolute* offsets
**Source:** `sgfp4_align16` (header lines 84–89) / `align16` (Python lines 98–100).
**Apply to:** MIXED `payloadsStart = sgfp4_align16(blockHeadersStart + blockHeadersBytes)` — correct only because `blockHeadersStart` is recomputed for MIXED (`recStart + 4 + 12`) and `recStart` is 16-aligned (Pitfall 4).

### 6. FP16-tolerant test comparison
**Source:** `checkVectorByRelativeError<float>` + `kFixtureRelativeTolerance = 1e-4f` (`SGFP4DequantTest.cpp` lines 26, 65).
**Apply to:** the new mixed round-trip and golden traversal assertions.

### 7. Encoder-generated committed fixtures keep Python and C++ in lockstep
**Source:** `build_fixture_cases()` + `emit_cpp_fixture()` (Python lines 476–578) → `test/op/SGFP4DequantFixtures.h`.
**Apply to:** all MIXED fixtures (D-08) — never hand-edit the header; always regenerate via `--emit-cpp-fixture`.

---

## No Analog Found

No files lack an analog — every target file is a Phase 1 artifact being extended, and its own current content is an exact-match template. Two *new code shapes* have no direct C++ copy source, but each has a defined pattern anchor:

| New code shape | Role | Reason | Pattern anchor |
|----------------|------|--------|----------------|
| `sgfp4_walk_quadtree()` + `SGFP4SplitMapReader` (C++) | utility, transform | No existing iterative fixed-stack walker in repo | spec §6.2 + RESEARCH Pattern 1; `sgfp4_read_u32_le` bit-math style |
| `subdivide_macroblock()`/`build_split_map()`/`classify_layout()` (Python) | utility, transform | No existing recursive encoder in `encode_sgfp4.py` | spec §6.3 + RESEARCH Pattern 3; existing `encode_leaf`/`pack_payload` decomposition style |
| Golden traversal enumerator (C++ test helper) | test | D-05 mandates independent implementation | independence pattern of `decode_container_ref` (Shared Pattern 3) |

## Metadata

**Analog search scope:** `include/MNN/SGFP4DequantUtils.hpp`, `tools/fp4/encode_sgfp4.py`, `test/op/SGFP4DequantTest.cpp`, `test/op/SGFP4DequantFixtures.h`
**Files scanned:** 4 (all read in full except fixture data arrays, which are generated byte dumps)
**Pattern extraction date:** 2026-08-24
