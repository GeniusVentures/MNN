# Phase 9: Real-Weight C++ Encoder Port - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-28
**Phase:** 9-Real-Weight C++ Encoder Port
**Areas discussed:** Port vs. consume, Parity bar, Non-aligned tiling, Encoder placement

---

## Port vs. consume

| Option | Description | Selected |
|--------|-------------|----------|
| Port v2 core only | Port ONLY the v2 adaptive quadtree encoder (FP4/T158 affine, MIXED + uniform fallbacks). No v1, no fixed-payload, no unused layouts. Single-command UX without dead code. | ✓ |
| Consume, don't port | No C++ encode logic; mnnconvert consumes fp4_exporter.py output dirs as sgfp4_inject does. Phase 9 shrinks to pipeline hardening. | |
| Full-feature port | Full parity port: v1 + v2, all layouts, both code modes. Maximum independence, most sync surface. | |
| Port + consume | Hybrid: C++ port for converter path + keep export-dir consumption for pre-quantized weights. | |

**User's choice:** Port v2 core only
**Notes:** Resolves the roadmap-mandated plan-time re-evaluation (handoff 2026-08-26).

Follow-up — quadtree policy faithfulness:

| Option | Description | Selected |
|--------|-------------|----------|
| Python-identical | Same superblock scan, same split decisions, same DEFAULT_V2_THRESHOLDS as fp4_exporter.py --adaptive. | ✓ |
| Tunable constants | Same traversal but thresholds become C++-side named constants for Phase 10 tuning. | |
| Simplified policy | Fixed split policy now, refine in Phase 10. Faster, risks divergence. | |

**User's choice:** Python-identical

Follow-up — delivery shape:

| Option | Description | Selected |
|--------|-------------|----------|
| Header lib + tests | sgfp4_encode.hpp/.cpp under tools/fp4, linked by Phase 11's pass later. | ✓ |
| CLI tool now | Standalone sgfp4_encode.out mirroring sgfp4_inject.out. | |
| Converter-embedded | Encoder in tools/converter/source from day one. | |

**User's choice:** Header lib + tests

## Parity bar

| Option | Description | Selected |
|--------|-------------|----------|
| Decode-parity rtol | C++-encode→decode matches Python-encode→decode within rtol 1e-4 (existing pattern). Byte-exactness NOT required. | ✓ |
| Byte-exact parity | Byte-identical container output; forces bit-exact FP behavior. | |
| Structural + decode | Byte-exact structure (offsets, flags, split maps) + decode-parity payloads. | |

**User's choice:** Decode-parity rtol

Follow-up — fixtures:

| Option | Description | Selected |
|--------|-------------|----------|
| Real-shape goldens | Generator (mirroring author_structured_fixture.py) runs fp4_exporter.py on deterministic pseudo-random non-64-aligned shapes; commits {weights, container, decoded ref} C arrays. | ✓ |
| Hand-built cases | Hand-constructed containers (border leaves, exact-fit, tiny tensors). | |
| Goldens + edges | Both generated goldens and hand-built edge cases. Most coverage, most effort. | |

**User's choice:** Real-shape goldens

## Non-aligned tiling

| Option | Description | Selected |
|--------|-------------|----------|
| Zero-pad to 64 | Encoder pads to 64-multiples internally, records true {dimO,dimI}; decode padded plane, consume true-dims region. | ✓ |
| Native edge leaves | Quadtree handles partial superblocks natively; requires verifying decoder traversal supports it. | |
| Follow Python exporter | Adopt whatever fp4_exporter.py does for non-64 shapes today. | |

**User's choice:** Zero-pad to 64

Follow-up — pad representation/crop:

| Option | Description | Selected |
|--------|-------------|----------|
| Row-major crop | dims={dimO,dimI} on op; first dimO*dimI elements row-major from decoded padded plane; verify decoders' elementCount handling. | ✓ |
| Explicit pad blocks | Pad region gets all-zero superblocks for visible/cheap wasted decode. | |
| Skip pad leaves | Don't encode pad rows/cols; dense record layout. | |

**User's choice:** Row-major crop

Follow-up — verification timing:

| Option | Description | Selected |
|--------|-------------|----------|
| Verify in Phase 9 | Padded non-aligned decode proven through BOTH real decoders (CPU oracle + Vulkan Execution) now — correctness prerequisite. | ✓ |
| Defer to Phase 10/11 | Encode-side unit tests only; end-to-end padded decode later. | |

**User's choice:** Verify in Phase 9

## Encoder placement

| Option | Description | Selected |
|--------|-------------|----------|
| tools/fp4 + link | sgfp4_encode.hpp/.cpp in tools/fp4 under MNN_BUILD_SGFP4_TOOLS; converter target links it in Phase 11. | ✓ |
| Shared lib target | Encoder in shared location both tools/fp4 and converter consume from day one. | |
| Header-only | Zero CMake target changes; Phase 11 just #includes. Compile-time cost per consumer. | |

**User's choice:** tools/fp4 + link

Follow-up — API surface:

| Option | Description | Selected |
|--------|-------------|----------|
| One-shot encode fn | FP32 weights + {dimO,dimI} → std::vector<uint8_t> container bytes; v2-adaptive defaults, no knobs. | ✓ |
| Config-struct API | Encoder config (code mode, layout override, threshold table) with defaults. | |
| Streaming API | Incremental rows/chunks for huge tensors. | |

**User's choice:** One-shot encode fn

---

## Claude's Discretion

- Exact function/file naming within sgfp4_* conventions (sgfp4_encode.hpp suggested).
- Internal encoder structure (quadtree builder class vs free functions; MSE accumulation details within the D-04 parity bar).
- Specific non-64-aligned shapes in the golden generator beyond the D-05 examples.
- Test suite naming/placement (op/sgfp4/ family + tools/fp4 wiring).
- Whether tiny tensors get dedicated hand-built edge cases in addition to generated goldens.
- encode_sgfp4.py role-comment documentation updates (stays test-oracle; C++ encoder becomes converter-path encoder).

## Deferred Ideas

- Configurable encoder API (config struct with threshold table) — Phase 10, only if real-weight validation demands it.
- Native partial-superboard traversal (no padding) — rejected this phase; revisit if pad overhead costs on real models (Phase 10 territory).
- Same-shape disambiguation via tensor-name keying — Phase 11 / injection tool per v2.0 audit.
- v1 fixed-payload and non-adaptive layouts — rejected outright.
