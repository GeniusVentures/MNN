# Phase 1: Affine Dual-Mode Decode Core (CPU, Uniform Layouts) - Research

**Researched:** 2026-08-21
**Domain:** MNN inference-engine op integration — new CPU Execution class for a self-framed binary weight container (SGFP4 v2), external-sidecar loading, FP16 affine dequant
**Confidence:** HIGH (all integration claims verified against source at file:line; all format claims cited to the spec)

## Summary

Phase 1 adds a **new, dedicated CPU decode op** for the SGFP4 v2 container restricted to uniform layouts. Everything the phase needs already has a working, idiomatic precedent in this repo: MNN's `Convolution2D.external:[int64]` + top-level `Op.externalPath` + `FileLoader` triple is exactly the external-sidecar mechanism the locked container decision mirrors, and the existing E2M1 `CPUFP4Dequant` is the pattern for a small dequant-style `Execution`. The FP16 half→float conversion the affine rule needs is already vendored and unconditionally on the include path (`3rd_party/half/half.hpp`, `half_float::half`), so no hand-rolled converter is required.

The single most consequential decision is op/schema integration. The locked decision ("external file + minimal `{magic, offset, size}` descriptor, no macroblock/quadtree typed fields") is fully satisfiable with a **new `OpType_SGFP4Dequant` + a small new FlatBuffers param table** — this is the path MNN's own `add-new-op` skill mandates and the only path that gets a working *runtime CPU Execution*. The tempting shortcut of reusing `OpType_Extra` to skip the flatc round-trip **does not work for CPU runtime execution**: no CPU backend creator dispatches `Extra` by string (only `OpType_Plugin` has a CPU creator; `Extra` is a converter/geometry-stage escape hatch, not a backend-execution one). Recommend the new-OpType path.

**Primary recommendation:** Add `OpType_SGFP4Dequant` + `table SGFP4DequantParam { magic:uint32; external:[int64]; dims:[int]; }` to the schema; write a `CPUSGFP4Dequant` Execution that reads the container once at setup via `FileLoader` (mirroring `ConvolutionCommon.cpp:590-598`), parses v2 framing + uniform records internally, and reconstructs `w = S·c + bias` using `half_float::half` for FP16 unpack. Split Phase 1 into ~5 tasks across 2 plans (decode-core+plumbing, then encoder+tests).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Container byte layout (magic/offset-table/records) | New CPU Execution (`CPUSGFP4Dequant`) | — | Locked decision: FlatBuffers never models the internal structure; the Execution parses SGFP4's own bytes |
| `{magic, offset, size}` op descriptor | FlatBuffers schema (`SGFP4DequantParam`) | — | Minimal typed descriptor only; mirrors `Convolution2D.external` |
| Sidecar `.mnn.weight` file read | `FileLoader` + `Op.externalPath` (core load path) | Execution setup (`onResize`/Creator) | Existing generic external-weight mechanism; read once at setup, not per-inference |
| FP16 (S, bias) → float | `3rd_party/half/half.hpp` (`half_float::half`) | — | Vendored, unconditionally on include path, already used by CPU code |
| Affine reconstruct `w = S·c + bias` | New CPU Execution | — | Core numeric contribution; pure shift/mask/FMA |
| Output tensor shape (O, I) | Shape computer (`ShapeSGFP4Dequant`) from param `dims` | Model manifest | Spec §6.1: tensor geometry lives in the manifest, not the container |
| Container production | Python encoder (`tools/fp4/`, new file) | — | Mirrors `quantize_fp4.py` role for the new format |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `3rd_party/half/half.hpp` (`half_float::half`) | vendored (in-repo) | IEEE-754 binary16 → float for scale/bias unpack | Already vendored and on the include path unconditionally (`CMakeLists.txt:446,459`) `[VERIFIED: codebase]`; header-only, C++11-compatible, RTTI/exception-free — compatible with MNN's `-fno-rtti -fno-exceptions` |
| `FileLoader` (`source/core/FileLoader.hpp`) | in-repo | Seek + read container bytes from the `.mnn.weight` sidecar | The exact mechanism `ConvolutionCommon` uses for external quantized weights `[VERIFIED: codebase]` |
| flatc (`3rd_party/flatbuffers`) | vendored | Regenerate `MNN_generated.h` after schema edit | Bootstrapped automatically by `schema/generate.sh` if not built `[VERIFIED: codebase]` |
| Python 3 + numpy | 3.13.4 / numpy 2.2.5 (present) | Reference encoder | Same stack `tools/fp4/quantize_fp4.py` already uses `[VERIFIED: shell probe]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Express / Module API (`MNN/expr/*`) | in-repo | Build the op via `OpT` → `Variable::save` → `Module::load` in tests | Op-level plumbing test (see `FP4ModelTest.cpp`, `VulkanFP4DequantTest.cpp`) |
| `MNNTestSuite` (`test/MNNTestSuite.h`) | in-repo | Test registration/harness | All `test/op/*Test.cpp` |
| `checkVectorByRelativeError`, `FP32Converter` (`test/TestUtils.h`) | in-repo | FP16-tolerant comparison | Matches how the Vulkan FP4 test validates FP16 decode |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New `OpType_SGFP4Dequant` | `OpType_Extra` (`Extra.info:[byte]`) | Avoids flatc round-trip, **but no CPU runtime creator dispatches `Extra`** (`source/backend/cpu/CPUOPRegister.cpp` registers only `OpType_Plugin`, not `Extra`) `[VERIFIED: codebase]`. Would require building the missing string-dispatch layer — more work than a clean OpType. Rejected. |
| `half_float::half` (half.hpp) | Hand-rolled scalar half→float | Reinvents a solved, tested, vendored utility; adds bug surface for the reserved-bit / subnormal cases. Rejected — see Don't Hand-Roll. |
| Explicit `offset:int64; size:int64;` fields | Reuse `external:[int64]` = `[offset, size]` | Reusing `external:[int64]` lets the op reuse the `USE_EXTERNAL_DATA(param)` macro (`OpCommonUtils.hpp:58`) and the established load idiom verbatim; recommended (see Q1). |

**Installation:** No new external packages. numpy is already a dependency of the existing encoder.

**Version verification:** `python -c "import numpy"` → numpy 2.2.5 present `[VERIFIED: shell probe]`. flatc is not on PATH but `schema/generate.sh` builds it from `3rd_party/flatbuffers/tmp/` on first run `[VERIFIED: codebase]`.

## Package Legitimacy Audit

Phase 1 installs **no new external packages**. The only third-party runtime dependency (numpy, used by the Python encoder) is already a dependency of the existing `tools/fp4/quantize_fp4.py`. The FP16 library (`half.hpp`) and flatc are vendored in-repo under `3rd_party/`. No registry audit is applicable.

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SGV2-01 | Affine `w = S·c + bias` for FP4_AFFINE (codes [-8,7]) + T158_AFFINE (codes {-1,0,+1}, reserved `11`→0) | Decode math from spec §3.2 / §4.3 (Eq. 2/3/4); reconstruct in new Execution — see CPU Decode Implementation Shape |
| SGV2-02 | FP16 scale+bias unpack (packHalf2x16 order; v2 12-bit truncated bias `S=half(h>>16)`, `bias=half(h&0xFFF0)`, `flags=h&0xF`) | `half_float::half` reuse (Eq. 6, §6.2); see Q3 |
| SGV2-03 | v2 stream framing (magic `'SGF4'`, ver `0x02`, `B`, 16-byte-aligned offset table, per-record `sb_header` layout enum bits 0–2) | Spec §6.1/§6.2 byte-level layout documented in Code Examples |
| SGV2-04 | Uniform record walk (UNIFORM_64/32/16/8, FULL_4x4), row-major raster leaf order, n²/8 (mode 0) / n²/16 (mode 1) payload words | Table 3 leaf counts + §4.3 packing; documented below |
| SGV2-05 | New `{magic, offset, size}` descriptor + `.mnn.weight` sidecar via `FileLoader`; no macroblock/quadtree schema fields | Q1 (schema) + Q2 (loading) — mirrors `Convolution2D.external` at `ConvolutionCommon.cpp:590-598` |
| SGV2-06 | New CPU Execution parsing container internally, additive to E2M1 `CPUFP4Dequant` | New files only (`CPUSGFP4Dequant.*`, `SGFP4DequantUtils.hpp`); E2M1 path untouched — SC#5 |
| SGV2-07 | Python encoder for uniform v2 containers + CPU round-trip tests via `./run_test.out` | Q4 (test scaffolding) + encoder mirrors `quantize_fp4.py` |
</phase_requirements>

## Architecture Patterns

### System Data-Flow Diagram

```
                       model load (Interpreter / Module)
                                    │
             Op.externalPath = "<model>.mnn.weight"  (Interpreter.cpp:96,278)
                                    │
                                    ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  CPUSGFP4Dequant  (new Execution)                                  │
   │                                                                    │
   │  onResize()  ── run ONCE at setup, not per-inference ─────────┐    │
   │    read descriptor: magic, external[0]=offset, external[1]=size│   │
   │    FileLoader(op->externalPath()).offset(offset).read(size) ──┼──► container buffer (held on the Execution)
   │                                                               │    │
   │  parse v2 framing (§6.1): magic 'SGF4', ver 0x02, B,          │    │
   │      16B-aligned record_offsets[B]                            │    │
   │                                                               ▼    │
   │  per record: sb_header → layout enum (Table 3)  ──► N, leaf size   │
   │      (uniform only — MIXED/quadtree is Phase 2)                    │
   │                                                                    │
   │  per leaf (row-major raster):                                      │
   │      header h (u32): S=half(h>>16), bias=half(h&0xFFF0),           │
   │                      mode = (h & 0xF) & 1                          │
   │      payload: mode 0 → 4-bit two's-complement nibbles (n²/8 words) │
   │               mode 1 → 2-bit ternary symbols   (n²/16 words)       │
   │      reconstruct w = S·c + bias  ────────────────────────────►  output float tensor (O × I)
   └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    downstream consumer / test comparison
```

### Recommended File Layout (all NEW files — additive)
```
include/MNN/SGFP4DequantUtils.hpp     # header-only decode helpers (mirrors FP4DequantUtils.hpp): framing parse, leaf-header unpack, dual-mode payload decode, affine reconstruct
source/backend/cpu/CPUSGFP4Dequant.hpp   # Execution declaration (mirrors CPUFP4Dequant.hpp)
source/backend/cpu/CPUSGFP4Dequant.cpp   # Creator + onResize (file read) + onExecute (decode)
source/shape/ShapeSGFP4Dequant.cpp       # REGISTER_SHAPE — output dims from param
schema/default/MNN.fbs                    # (edit) OpType_SGFP4Dequant + OpParameter union entry
schema/default/CaffeOp.fbs  (or MNN.fbs)  # (edit) table SGFP4DequantParam
tools/fp4/encode_sgfp4.py                 # NEW encoder (do NOT modify quantize_fp4.py)
test/op/SGFP4DequantTest.cpp              # MNNTestSuiteRegister(..., "op/sgfp4/uniform_decode")
```

### Pattern 1: Read external container once at setup (not per-inference)
**What:** Open `FileLoader`, seek to `offset`, read `size` bytes into a buffer owned by the Execution, in `onResize()` (or the Creator). Decode in `onExecute()` from the cached buffer.
**When to use:** Always — file I/O per `onExecute` call would be a severe performance regression and violates MNN's setup/execute split.
**Example:** verbatim idiom from `ConvolutionCommon.cpp:590-598` (see Code Examples).

### Pattern 2: Dequant-style Execution with same-shape passthrough
**What:** `CPUFP4Dequant` is a minimal `Execution` that only implements `onExecute` and produces a float tensor. `CPUSGFP4Dequant` follows this but adds `onResize` for the one-time file read and holds the container buffer.
**When to use:** Copy `CPUFP4Dequant.hpp/.cpp` structure; add a member `AutoStorage<uint8_t>`/`std::vector<uint8_t>` for the container.

### Anti-Patterns to Avoid
- **Reusing `OpType_Extra` for CPU runtime decode:** no CPU creator dispatches it → op will not execute. Use a real OpType.
- **Editing `FP4DequantUtils.hpp` / `CPUFP4Dequant.cpp` / `quantize_fp4.py`:** breaks SC#5 (additive-only) and the live `dequant_fp4_packed_cpu` cross-repo contract. All new work goes in new files.
- **Modeling macroblocks/leaves/quadtree as FlatBuffers fields:** violates the locked container decision. The schema carries only `{magic, offset, size}` (+ output dims).
- **Hand-rolling half→float:** use `half_float::half`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FP16 → float32 | Custom bit-twiddling half decoder | `half_float::half` (`3rd_party/half/half.hpp`) | Subnormals, inf/nan, rounding all handled; already vendored and on include path; used by CPU code (`CPURasterAndInterpolate.cpp:8,382`) `[VERIFIED: codebase]` |
| Seek+read from sidecar | `fopen`/`fseek`/`fread` by hand | `FileLoader` | Handles the `Op.externalPath` wiring, offset/read API MNN already standardizes on `[VERIFIED: codebase]` |
| Schema codegen | Hand-writing `MNN_generated.h` structs | `schema/generate.sh` (flatc) | Enum ordinals + object-API must stay consistent; skill step 1 mandates this `[CITED: skills/add-new-op/step1-schema.md]` |
| Creator registration list | Hand-editing `CPUOPRegister.cpp` / `ShapeRegister.cpp` | `tools/script/register.py` | These lists are auto-generated from `REGISTER_*` macros `[VERIFIED: codebase]` |

**Key insight:** Every mechanism Phase 1 needs already exists in-repo with a working precedent. The phase is *composition of existing idioms* plus new decode math, not new infrastructure.

## Common Pitfalls

### Pitfall 1: Output shape has no natural source (op is Const-like)
**What goes wrong:** A new OpType with no shape computer fails size inference; and unlike the E2M1 path (which piggybacked on `OpType_Dequantize`'s "copy input shape"), SGFP4Dequant has no meaningful float input tensor to copy from — it *produces* weights from the sidecar.
**Why it happens:** Spec §6.1 explicitly keeps tensor geometry (O, I) in the enclosing manifest, not in the container. `[CITED: sgfp4-arxiv-v2.txt §6.1]`
**How to avoid:** Add an output-dims field to the param table (`dims:[int]`, e.g. `[O, I]`) and write `ShapeSGFP4Dequant` to set the output shape+type from it. This is NOT a macroblock/quadtree field, so it stays compliant with the locked decision. `REGISTER_SHAPE(ShapeSGFP4Dequant, OpType_SGFP4Dequant)`, output type `halide_type_of<float>()` (mirrors `ShapeDequantize.cpp:18-20`).
**Warning signs:** `SizeComputer` assertion/failure; zero-size output tensor.

### Pitfall 2: `externalPath` not set for buffer/Express-built test models
**What goes wrong:** `Interpreter.cpp:96` sets `externalFile = <file> + ".weight"` only on the **file-based** load path. A test that builds the op via `OpT` → `Variable::save` → `Module::load(buffer, rtmgr)` may have `op->externalPath() == nullptr`, so the `FileLoader` open fails.
**Why it happens:** The sidecar filename is derived from the model file path; there is no file path for a buffer/Express model.
**How to avoid:** Confirm and use the RuntimeManager/Interpreter external-file setter (`Interpreter::setExternalFile`, `Interpreter.cpp:192`) or write the model to a temp `.mnn` on disk with its `.mnn.weight` sidecar so the default naming applies. **Flag for planner:** verify the exact `RuntimeManager`/`Module::load` path for setting the external file — this is the one plumbing detail not fully traced (see Open Questions).
**Warning signs:** null `externalPath`, `FileLoader::valid()` false.

### Pitfall 3: 16-byte alignment / padding arithmetic
**What goes wrong:** Records or payloads read at the wrong offset because `pad0` (7B after the 9-byte fixed header → 16B), `pad1` (so the record region starts at `align16(16 + 4B)`), and per-payload 16-byte padding are mishandled.
**Why it happens:** The spec's alignment guarantees only hold when the pads are honored; `pad1` specifically exists to fix misalignment when `B ≢ 0 (mod 4)`. `[CITED: sgfp4-arxiv-v2.txt §6.1]`
**How to avoid:** Compute region start as `align16(16 + 4*B)`; treat each `record_offsets[b]` as relative to that region start; pad each leaf payload up to a 16-byte multiple when advancing. Add a unit test with `B` not a multiple of 4.
**Warning signs:** decode drifts after the first record; works for B=1 but fails for B=3.

### Pitfall 4: Two's-complement nibble vs. E2M1 nibble confusion
**What goes wrong:** Reusing E2M1 `dequant_e2m1_cpu` (sign/exp/mantissa) instead of SGFP4's **plain 4-bit two's-complement integer** (0..7→0..7, 8..15→-8..-1).
**Why it happens:** Both are "FP4" and pack 8 nibbles/word, but the code semantics are completely different.
**How to avoid:** SGFP4 mode 0 codes are integers; sign-extend the nibble: `c = (n ^ 0x8) - 0x8` for a 4-bit value. Keep this in the NEW `SGFP4DequantUtils.hpp`, never touch `FP4DequantUtils.hpp`.
**Warning signs:** decoded values look exponentially spaced instead of linear.

### Pitfall 5: Ternary reserved symbol and word-count
**What goes wrong:** Symbol `11` (0x3) not decoded as 0; or mode-1 word count computed as n²/8 instead of n²/16.
**How to avoid:** Mapping `00→0, 01→+1, 10→-1, 11→0(reserved)` `[CITED: §4.3 Eq.4]`; mode-1 payload is `n²/16` little-endian uint32 words (16 symbols/word). For the 64×64 leaf this is 256 words = 1024B of codes.
**Warning signs:** ternary decode passes for some symbols but injects spurious values.

## Code Examples

### v2 file framing parse (from spec §6.1) — new `SGFP4DequantUtils.hpp`
```cpp
// Source: sgfp4-arxiv-v2.txt §6.1 (little-endian throughout)
// Fixed header is 16 bytes: magic[4]='SGF4', version(u8)=0x02, B(u32), pad0[7].
struct SGFP4Header { uint32_t magic; uint8_t version; uint32_t B; };
// record region begins at align16(16 + 4*B); record_offsets[b] is relative to it.
static inline size_t align16(size_t x) { return (x + 15u) & ~size_t(15u); }
// const uint8_t* base = container;
// uint32_t B = read_u32_le(base + 5);                 // after magic(4)+version(1)
// const uint32_t* recOff = (const uint32_t*)(base + 16);
// size_t region = align16(16 + 4*(size_t)B);
// const uint8_t* rec_b = base + region + recOff[b];
```

### Per-leaf header unpack (spec §6.2, Eq. 6) using vendored half
```cpp
// Source: sgfp4-arxiv-v2.txt §6.2 Eq.(6); half_float from 3rd_party/half/half.hpp
#include "half.hpp"              // include dir added unconditionally (CMakeLists.txt:446)
static inline void unpack_leaf_header(uint32_t h, float& S, float& bias, int& mode) {
    uint16_t sBits    = (uint16_t)(h >> 16);
    uint16_t biasBits = (uint16_t)(h & 0xFFF0);   // 4 low mantissa bits repurposed as flags
    half_float::half hs, hb;
    std::memcpy(&hs, &sBits, 2);
    std::memcpy(&hb, &biasBits, 2);
    S = (float)hs; bias = (float)hb;
    mode = (int)(h & 0x1);                          // bit 0 = mode; bit1/2-3 reserved 0
}
```

### Dual-mode payload decode + affine reconstruct (spec §3.2, §4.3)
```cpp
// Source: sgfp4-arxiv-v2.txt §3.2 Eq.(2), §4.3 Eq.(3)/(4). n = leaf edge size.
// Mode 0 (FP4_AFFINE): n*n codes, 8 nibbles per little-endian uint32 word, n*n/8 words.
for (int i = 0; i < n*n; ++i) {
    uint32_t w = words[i >> 3];
    int nib = (w >> (4 * (i & 7))) & 0xF;
    int c   = (nib ^ 0x8) - 0x8;          // 4-bit two's complement -> [-8,7]
    out[i]  = S * (float)c + bias;
}
// Mode 1 (T158_AFFINE): n*n codes, 16 symbols per word, n*n/16 words.
for (int i = 0; i < n*n; ++i) {
    uint32_t w  = words[i >> 4];
    int sym     = (w >> (2 * (i & 15))) & 0x3;
    int c       = (sym == 1) ? +1 : (sym == 2) ? -1 : 0;   // 00->0,01->+1,10->-1,11->0
    out[i]      = S * (float)c + bias;
}
```

### External-file read idiom to mirror (from `ConvolutionCommon.cpp:590-598`)
```cpp
// Source: source/core/ConvolutionCommon.cpp:590-598  [VERIFIED: codebase]
std::unique_ptr<FileLoader> external_file(new FileLoader(op->externalPath()->c_str()));
external_file->offset(external_info[0]);                 // external_info = external()->data()
external_file->read((char*)buffer_ptr, buffer_size);     // buffer_size = external_info[1]
```

### Uniform layout table (spec Table 3) — Phase 1 subset
```
enum  name               N      leaf n   mode0 words (n²/8)   mode1 words (n²/16)
 0    LAYOUT_UNIFORM_64    1     64        512                  256
 1    LAYOUT_UNIFORM_32    4     32        128                   64
 2    LAYOUT_UNIFORM_16   16     16         32                   16
 3    LAYOUT_UNIFORM_8    64      8          8                    4
 4    LAYOUT_MIXED       var.   mixed      -- Phase 2 --        -- Phase 2 --
 5    LAYOUT_FULL_4x4    256      4          2                    1
```
Leaves stored in row-major raster order of the tile grid; each payload padded to a 16-byte multiple. `[CITED: sgfp4-arxiv-v2.txt §6.2 Table 3, §4.3]`

## Answers to Focus Questions

### Q1 — Op/schema integration strategy (concrete recommendation)
**Recommendation: brand-new `OpType_SGFP4Dequant` + a small new param table. Do NOT reuse `OpType_Extra`.**

Rationale (verified):
- `OpType_Extra` (`MNN.fbs:512`) carries `info:[byte]` and is MNN's custom-op *escape hatch*, but on the CPU backend there is **no `Extra` creator** — `source/backend/cpu/CPUOPRegister.cpp` registers `___CPUPluginCreator__OpType_Plugin__` (only `OpType_Plugin`), and `CPUPlugin.cpp:76` asserts `op->type() == OpType_Plugin`. `[VERIFIED: codebase]` So an `Extra` op would not dispatch to any runtime Execution on CPU. Reusing it would mean building the missing string-dispatch layer — strictly more work than a clean OpType.
- The `add-new-op` skill step 1 mandates: append the OpType to the `OpType` enum **tail only** (ordinals are immutable), define a param table, append it to the `OpParameter` union tail, then regenerate with flatc. `[CITED: skills/add-new-op/step1-schema.md §1.3–1.5]`

Concrete schema edit:
```fbs
// schema/default/MNN.fbs — append to OpType enum tail (after GridSample = 604)
SGFP4Dequant = 605,

// param table (place in CaffeOp.fbs near QuantizedFloatParam, or MNN.fbs)
table SGFP4DequantParam {
    magic:uint32;        // 'SGF4' little-endian sanity check
    external:[int64];    // [offset, size] into the .mnn.weight sidecar (reuses USE_EXTERNAL_DATA)
    dims:[int];          // output tensor geometry (O, I) — from manifest, per spec §6.1
}

// append to OpParameter union tail
SGFP4DequantParam
```
`external:[int64] = [offset, size]` is preferred over separate `offset:int64; size:int64;` because it lets the op reuse the `USE_EXTERNAL_DATA(param)` macro (`OpCommonUtils.hpp:58` = `param->external() && param->external()->size() > 1`) and the exact `ConvolutionCommon` read idiom verbatim `[VERIFIED: codebase]`. `magic` remains a distinct field to satisfy the locked "`{magic, offset, size}`" descriptor shape. `dims` is tensor geometry (explicitly manifest-resident per spec §6.1), **not** a macroblock/quadtree field, so SC#2 ("no macroblock/quadtree fields in schema") holds. `[CITED: sgfp4-arxiv-v2.txt §6.1]`

Post-edit steps (mandated): `schema/generate.sh` regenerates `MNN_generated.h` into `schema/current/` (note: skill says `schema/default/generate.sh`, but the actual script is `schema/generate.sh` and it outputs to `schema/current/`, which is on the include path — `CMakeLists.txt:443`) `[VERIFIED: codebase]`; then `REGISTER_SHAPE(...)`, `REGISTER_CPU_OP_CREATOR(CPUSGFP4DequantCreator, OpType_SGFP4Dequant)`, and run `tools/script/register.py` to regenerate `CPUOPRegister.cpp` / `ShapeRegister.cpp`.

### Q2 — External-file loading mechanics (traced)
The mechanism is three cooperating pieces, all verified:
1. **Sidecar path (top-level `Op.externalPath`, a `string` — `MNN.fbs:453`)** `[VERIFIED: codebase]`. On file load, `Interpreter.cpp:96` sets `net->externalFile = std::string(file) + ".weight"`, and `Interpreter.cpp:278` propagates it as `info.externalWeightPath`. `Interpreter::setExternalFile` (`:192`) overrides. `OpCommonUtils.cpp:658` writes `externalPath` onto each op during model prep.
2. **Descriptor (`param->external():[int64]`)** gated by `USE_EXTERNAL_DATA(param)` (`OpCommonUtils.hpp:58`). For `Convolution2D` it is `[offset, weight_bytes_size, bias_bytes_size]` (`CaffeOp.fbs:109`); SGFP4 uses `[offset, size]`.
3. **Read (`FileLoader`)**: `ConvolutionCommon.cpp:590-598` opens `FileLoader(op->externalPath()->c_str())`, calls `->offset(external_info[0])` then `->read((char*)buf, buffer_size)`. `[VERIFIED: codebase]` `FileLoader` API (`FileLoader.hpp`): ctor(path), `offset(int64_t)`, `read(char*, int64_t)`, `valid()`, `size()`.

**What Phase 1 adds:** in `CPUSGFP4DequantCreator::onCreate` (or the Execution's `onResize`, which runs at setup — not per-inference), read `external()->data()[0]` (offset) and `[1]` (size), open a `FileLoader` on `op->externalPath()`, seek+read `size` bytes into a buffer owned by the Execution. Decode from that buffer in `onExecute`. This is a direct transcription of the `ConvolutionCommon` idiom into the new op — **no new loading infrastructure required**.

### Q3 — CPU decode implementation shape
Byte-level walk (uniform-only), verified against spec §6.1/§6.2/§4.3 and captured in Code Examples above:
1. **Framing (§6.1):** parse `magic 'SGF4'`, `version 0x02`, `B`; fixed header is 16 bytes (`magic4 + ver1 + B4 + pad0[7]`); `record_offsets[B]` (u32, LE) begin at byte 16; record region starts at `align16(16 + 4B)`; `record_offsets[b]` is relative to that region.
2. **Record (§6.2):** `sb_header` u32 → layout enum in bits 0–2 (Table 3); **no split map** for uniform layouts (skip — that is Phase 2); then `block_headers[N]` (u32 each, N from Table 3), pad to 16B, then `payloads[N]`.
3. **Leaf header (§6.2 Eq. 6):** `S = half(h>>16)`, `bias = half(h & 0xFFF0)`, `flags = h & 0xF`, `mode = flags & 1`.
4. **Payload (§4.3):** mode 0 → `n²/8` LE u32 words, 8 two's-complement nibbles/word; mode 1 → `n²/16` words, 16 ternary symbols/word (`11`→0); each payload padded to 16B.
5. **Reconstruct:** `w = S·c + bias`, row-major within the leaf; leaves in row-major raster order.

**FP16 utility — reuse, do not write:** `3rd_party/half/half.hpp` provides `half_float::half` with `operator float()` / `detail::half2float`. It is **unconditionally** on the include path (`CMakeLists.txt:446,459` — not gated on any backend flag) `[VERIFIED: codebase]`, is header-only and C++11/RTTI-free (compatible with MNN's `-fno-rtti -fno-exceptions`), and is already used by CPU code (`source/backend/cpu/render/CPURasterAndInterpolate.cpp:8` `#include "half.hpp"`, `:382` casts `(half_float::half*)`) `[VERIFIED: codebase]`. The `HalfToFloat` seen earlier in SuperGenius's `processing_processor_mnn_tensor.cpp` is a downstream-repo helper and out of scope; the MNN-native answer is `half.hpp`. No hand-rolled converter needed.

### Q4 — Test scaffolding pattern
- **File:** `test/op/SGFP4DequantTest.cpp`; register `MNNTestSuiteRegister(SGFP4DequantTest, "op/sgfp4/uniform_decode");` — mirrors the sibling `op/fp4/conversion` (`FP4ModelTest.cpp:108`) and `op/vulkan/fp4_dequant_correctness` (`VulkanFP4DequantTest.cpp:338`) naming, so `op/sgfp4/...`. `[VERIFIED: codebase]`
- **Compile gate:** wrap in `#ifdef MNN_SUPPORT_TRANSFORMER_FUSE` for parity with both existing FP4 tests. `[VERIFIED: codebase]`
- **Comparison:** use `checkVectorByRelativeError` and, for FP16-precision reference, `FP32Converter[3]` (from `test/TestUtils.h`), exactly as `VulkanFP4DequantTest.cpp:206-222` does.
- **Structure (recommend two layers):**
  1. *Direct decode unit test* — hand-build (or load a committed fixture) container bytes in-memory, call the new `SGFP4DequantUtils` decode function, compare to a straightforward reference `w = S·c + bias`. Fast, proves framing/parse/dual-mode/FP16/ternary-reserved. Cover: both modes × all 5 uniform layouts, plus a `B ≢ 0 (mod 4)` case, plus the ternary `11`→0 case.
  2. *Op-level plumbing test* — build the op via `OpT` → `Variable::save` → `Module::load(..., rtmgr)` (pattern from `VulkanFP4DequantTest.cpp:33-56` / `FP4ModelTest.cpp:58-71`), pointing at a temp `.mnn.weight` sidecar, run through the CPU backend, and confirm the external-file path decodes end-to-end. This is what exercises SGV2-05/06.
- **Container source for tests:** simplest is a small C++ container-builder helper in the test (mirrors the encoder's packing); optionally also decode a container produced by the new Python encoder (committed fixture or offline step) to prove cross-language round-trip (SGV2-07). Flag the choice for the planner.

### Q5 — Task/plan sizing (recommendation for planner)
Given project convention (1–3 tasks/plan, phases 1–2 plans) and the 7 requirements, ~5 natural tasks across **2 plans** — consistent with ROADMAP's "~2 plans" estimate and the `add-new-op` skill's step ordering (schema → shape → CPU backend → tests):

**Plan 01-01 — Decode core + container plumbing** (SKILL steps 1→3; SGV2-01,02,03,04,05,06)
- Task A: Schema + op registration — `OpType_SGFP4Dequant`, `SGFP4DequantParam`, flatc regen, `ShapeSGFP4Dequant`, empty `CPUSGFP4Dequant` + Creator registration, `register.py`.
- Task B: External-file loading — descriptor parse + `FileLoader` read at setup (mirror `ConvolutionCommon.cpp:590-598`), buffer held on the Execution.
- Task C: CPU decode core (the heavy task) — `SGFP4DequantUtils.hpp` v2 framing parse, uniform record walk, leaf-header FP16 unpack via `half.hpp`, both-mode payload decode, affine reconstruct. *If this task grows, split into C1 (framing + uniform walk + FP16 header) and C2 (dual-mode payload decode + affine).* 

**Plan 01-02 — Encoder + round-trip tests** (SKILL step 4; SGV2-07)
- Task D: Python encoder `tools/fp4/encode_sgfp4.py` — round-to-nearest affine encode + per-block mode selection, emit uniform-layout v2 containers + sidecar. Do NOT modify `quantize_fp4.py`.
- Task E: CPU tests `test/op/SGFP4DequantTest.cpp` — direct decode + op-level plumbing, both modes × 5 uniform layouts, FP16 precision, ternary reserved, `B ≢ 0 (mod 4)` alignment.

## Project Constraints (from CLAUDE.md)
- MNN is an **inference engine** — prioritize performance and binary size; the decode op must read the sidecar once at setup, not per-inference.
- **Do NOT read/modify** `schema/private/` or `source/internal/` (also restated in the add-new-op skill).
- **C++11**, Google-style variant (`.clang-format`), 4-space indent, 120-col, `PascalCase` classes / `camelCase` funcs / `mCamelCase` members; **RTTI and exceptions disabled** (`-fno-rtti -fno-exceptions`) — `half.hpp` is compatible.
- Named constants over magic numbers (the many `0xFFF0`, `>>16`, `n²/8` literals should be short named constants/helpers in `SGFP4DequantUtils.hpp`).
- Format touched files with `clang-format -i -style=file`.
- This phase is effectively "add a new op" → follow `skills/add-new-op/SKILL.md` step by step (each step passes its test before proceeding).
- No Vulkan/GLSL work this phase, so the `makeshader.py` regeneration rule does not apply until Phase 3.

## Runtime State Inventory

Not applicable — Phase 1 is a **greenfield additive** feature (new op, new files), not a rename/refactor/migration. No existing stored data, service config, OS-registered state, secrets, or build artifacts carry a string this phase renames.
- Stored data: None — new container format, no existing datastore keys.
- Live service config: None.
- OS-registered state: None.
- Secrets/env vars: None.
- Build artifacts: One-time only — after the schema edit, `schema/current/MNN_generated.h` must be regenerated and the CPU/shape register lists refreshed via `register.py`; these are build-time regenerations, not stale runtime state.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| E2M1 float microformat, per-channel `symmetricQuan`, bias≡0, `OpType_Dequantize` piggyback | SGFP4 v2 affine-integer `w=S·c+bias`, dual-mode (FP4/ternary), self-framed container, dedicated new op | This workstream (2026-08) | New op is fully separate; E2M1 path and its `dequant_fp4_packed_cpu` cross-repo contract stay unchanged (SC#5) |

**Deprecated/outdated in this context:** none — SGFP4 v2 is additive. The E2M1 `MAX_E2M1_VALUE=6.0→3.0` calibration bug noted in the analysis is already fixed in `quantize_fp4.py:36` `[VERIFIED: codebase]` and is orthogonal to this phase.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Output shape should be carried as a param `dims:[int]` and resolved by a new shape computer | Q1 / Pitfall 1 | If MNN expects shape from a graph input instead, the op wiring changes; but spec §6.1 explicitly keeps geometry in the manifest, so param-carried dims is the natural fit. Low risk. |
| A2 | Adding `dims` to the descriptor does not violate the locked "`{magic, offset, size}`" decision | Q1 | If the planner/user reads the lock strictly, they may prefer geometry come from the consuming op instead. `dims` is tensor geometry, not macroblock/quadtree structure, so SC#2 still holds — but confirm in discuss/planning. |
| A3 | Reuse the `MNN_SUPPORT_TRANSFORMER_FUSE` compile gate for the new test | Q4 | If SGFP4 should be gated by a distinct flag, the gate changes; reusing it matches sibling FP4 tests. Low risk. |
| A4 | Buffer/Express-built test models can be pointed at a sidecar via `Interpreter::setExternalFile`/RuntimeManager | Pitfall 2 / Open Q1 | If no such path exists for `Module::load`, tests must write a real `.mnn` + `.mnn.weight` to disk. Medium risk — verify early. |
| A5 | `half_float::half` FP16 semantics match the spec's `half()` exactly (round-to-nearest-even, IEEE binary16) | Q3 | Spec §6.2 truncates 4 mantissa bits into flags; as long as decode reads `bias = half(h & 0xFFF0)`, half.hpp's decode is standard IEEE and matches. Low risk. |

## Open Questions

1. **How is `Op.externalPath` set for a buffer-loaded or Express-built model?**
   - What we know: file-based `Interpreter` load defaults it to `<file>.weight` (`Interpreter.cpp:96`); `Interpreter::setExternalFile` exists (`:192`).
   - What's unclear: the exact call to set it when a test uses `Module::load(buffer, rtmgr)` (no file path).
   - Recommendation: in the op-level test, either write a temp `.mnn` + `.mnn.weight` pair to disk (so default naming applies) or locate the `RuntimeManager`/`Interpreter` external-file setter. Resolve in the first plan's plumbing task.
2. **Container test fixtures: C++ builder vs. Python-encoder output vs. both?**
   - Recommendation: C++ builder for the fast decode unit test; additionally decode one Python-encoder-produced container to prove cross-language round-trip (SGV2-07). Planner to decide whether the Python fixture is generated at test time or committed.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3 | Reference encoder (Task D) | ✓ | 3.13.4 | — |
| numpy | Reference encoder | ✓ | 2.2.5 | — |
| flatc | Schema regen after OpType edit | ✗ (not on PATH) | — | `schema/generate.sh` bootstraps it from `3rd_party/flatbuffers/tmp/` on first run `[VERIFIED: codebase]` |
| C++ toolchain + CMake | Build + `run_test.out` | assumed (repo builds today) | — | — |
| `half.hpp` include | FP16 unpack | ✓ | vendored | on include path unconditionally (`CMakeLists.txt:446`) |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** flatc — bootstrapped by `schema/generate.sh`.

## Validation Architecture

> `.planning/config.json` for this workstream was not located during research; treating `nyquist_validation` as enabled (default). If explicitly false, skip.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | MNN's own `MNNTestSuite` (C++), run via `run_test.out` |
| Config file | none — tests self-register via `MNNTestSuiteRegister` |
| Quick run command | `cd build && ./run_test.out op/sgfp4` |
| Full suite command | `cd build && ./run_test.out` |

### Phase Requirements → Test Map
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

### Sampling Rate
- **Per task commit:** `./run_test.out op/sgfp4` (+ `op/fp4` when touching shared build).
- **Per plan/wave merge:** `./run_test.out` (full suite green).
- **Phase gate:** Full suite green before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `test/op/SGFP4DequantTest.cpp` — covers SGV2-01..07 (new file; add to `test/CMakeLists.txt`/test glob).
- [ ] Optional container fixture (committed bytes or encoder-generated) for cross-language round-trip.
- [ ] No new framework install needed — `MNNTestSuite` already builds with `MNN_BUILD_TEST=ON`.

## Security Domain

> `security_enforcement` config not located; SGFP4 v2 attestation/verifiable-execution is explicitly OUT of scope for this workstream (locked decision #2 / ROADMAP note 3). This section is limited to the input-validation surface of parsing an untrusted binary container.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | The decoder parses an external, potentially malformed binary blob. Validate: `magic == 'SGF4'`, `version == 0x02`, every `record_offsets[b]` + computed record/payload extent stays within `size`, `B` and per-layout `N` bound loop counts, and layout enum ∈ {0,1,2,3,5} for Phase 1 (reject 4/MIXED and ≥6). Bounds-check before every read from the container buffer. |
| V6 Cryptography | no | No crypto in scope (attestation explicitly out of scope). |
| V2/V3/V4 (auth/session/access) | no | Not an auth-bearing surface. |

### Known Threat Patterns
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed/oversized offsets → OOB read | Tampering / DoS | Validate `offset+size ≤ file size` and every record/payload extent against the buffer length before dereferencing (mirror the defensive extent checks the size computer + read path already imply) |
| Huge `B`/`N` → excessive allocation | DoS | Bound against the declared output `dims` (O×I); a uniform layout implies fixed `N`, so cross-check `N` and total decoded element count against `dims`. |

## Sources

### Primary (HIGH confidence — verified at file:line in this session)
- `source/core/ConvolutionCommon.cpp:515-610` — external-file load + `FileLoader` idiom the new op mirrors
- `source/core/Interpreter.cpp:96,192,278` — `.weight` sidecar naming + `setExternalFile` + propagation
- `source/core/OpCommonUtils.hpp:58` (`USE_EXTERNAL_DATA`), `OpCommonUtils.cpp:50-64,658` — external gate + `externalPath` wiring
- `source/core/FileLoader.hpp` — `FileLoader` API (offset/read/valid/size)
- `source/backend/cpu/CPUFP4Dequant.{hpp,cpp}`, `CPUDequantize.cpp:118-152` — Execution + Creator pattern to copy; E2M1 route
- `source/backend/cpu/CPUBackend.hpp:218-247` (`REGISTER_CPU_OP_CREATOR`), `CPUOPRegister.cpp`, `CPUPlugin.cpp:76,98` — registration; Extra has no CPU creator
- `source/shape/ShapeDequantize.cpp`, `ShapeRegister.cpp:67,191` — shape computer + `REGISTER_SHAPE` pattern
- `schema/default/MNN.fbs:17,218-225,446-453,512,605-region` + `OpType`/`OpParameter`/`Op.externalPath`; `CaffeOp.fbs:80-109` (`QuantizedFloatParam`, `Convolution2D.external`)
- `CMakeLists.txt:438-462` — `3rd_party/half` unconditionally on include path
- `3rd_party/half/half.hpp:1023-1053` (`half_float::half`, `operator float`); `source/backend/cpu/render/CPURasterAndInterpolate.cpp:8,382` — existing CPU usage
- `schema/generate.sh` — flatc bootstrap + `schema/current/` output
- `test/op/FP4ModelTest.cpp`, `test/op/VulkanFP4DequantTest.cpp` — test structure + `MNNTestSuiteRegister` naming
- `tools/fp4/quantize_fp4.py` — encoder pattern to mirror (and the `MAX_E2M1_VALUE=3.0` fix, line 36)
- `skills/add-new-op/SKILL.md`, `step1-schema.md` — mandated op-add process

### Secondary (spec — CITED)
- `.planning/sgfp4-arxiv-v2.txt` §3.2 (Eq. 2, code modes), §4.2 (aligned-offset flags), §4.3 (normative payload packing, Eq. 3/4), §6.1 (v2 file framing), §6.2 (macroblock record, Table 3, Eq. 6 leaf-header unpack)

### Tertiary (LOW confidence)
- none — every integration claim was verified against source this session; format claims are cited to the spec.

## Metadata

**Confidence breakdown:**
- Op/schema integration (Q1): HIGH — verified Extra has no CPU creator; new-OpType path matches the mandated skill.
- External-file loading (Q2): HIGH — traced end-to-end at file:line.
- CPU decode shape (Q3): HIGH — spec-normative + half.hpp reuse verified on include path and in CPU code.
- Test scaffolding (Q4): HIGH — direct pattern match to sibling FP4 tests.
- Task sizing (Q5): MEDIUM — reasoned against project convention + skill ordering; exact split is the planner's call.
- Output-shape design (A1/A2): MEDIUM — recommended `dims` param needs discuss/planning confirmation against the locked descriptor wording.

**Research date:** 2026-08-21
**Valid until:** ~2026-09-20 (stable — in-repo integration points, spec is frozen)
