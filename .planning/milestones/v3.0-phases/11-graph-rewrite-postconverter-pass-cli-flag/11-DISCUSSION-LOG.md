# Phase 11: Graph-Rewrite PostConverter Pass + CLI Flag - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-01
**Phase:** 11-graph-rewrite-postconverter-pass-cli-flag
**Areas discussed:** Pass placement & wiring, CLI flag & op scope, Threshold sourcing, Tech-debt scope, Test strategy

---

## Pass placement & wiring

| Option | Description | Selected |
|--------|-------------|----------|
| PostConverter pass | Named registered pass appended to the final RunNetPass batch (with ReIndexTensor); standard mechanism, free --dumpPass logging, drivable from TestSGFP4Converter | ✓ |
| postTreat sweep | Dedicated sweep in postTreat() before the _postTreatOp loop; explicit ordering, no registration | |
| You decide | Claude picks after research | |

**User's choice:** PostConverter pass
**Notes:** Net-level topology access (consumer rewiring) forces a net-level pass; per-op WeightQuantAndCoding hook cannot see topology.

| Option | Description | Selected |
|--------|-------------|----------|
| inputs>1 guard (rec) | Skip convs with inputIndexes.size() > 1 — an original conv has one input (weights in param); a second input fingerprints an SGFP4-rewritten conv. Pure topology, no schema change | ✓ |
| Guard + flag mutex | Additionally force weightQuantBits=0 at CLI parse when SGFP4 on — defense in depth | |
| You decide | Claude picks guard mechanism | |

**User's choice:** inputs>1 guard (rec)
**Notes:** Flag-level mutex later chosen as a hard parse error under CLI flag & op scope (D-05) — the structural guard remains the topology-level defense.

| Option | Description | Selected |
|--------|-------------|----------|
| Both, iteratively (rec) | Walk netT->oplists AND all subgraph->nodes, mirroring RemoveUnusefulOp/saveExternalData iteration | ✓ |
| Main oplist only | Subgraph coverage deferred; AlexNet corpus has no control flow so nothing observable lost | |
| You decide | Claude decides post-research | |

**User's choice:** Both, iteratively (rec)

## CLI flag & op scope

| Option | Description | Selected |
|--------|-------------|----------|
| --sgfp4 bool (rec) | Boolean flag mirroring --hqq/--fp16 precedent; thresholds from pass-internal config | ✓ |
| Flag + thresholds file | --sgfp4Thresholds <file.json> mirroring --compressionParamsFile; inject validated delta without rebuild | |
| You decide | Claude decides name/shape post-research | |

**User's choice:** --sgfp4 bool (rec)

| Option | Description | Selected |
|--------|-------------|----------|
| Hard error (rec) | --sgfp4 + (--weightQuantBits | --hqq | --fp16) → MNN_ERROR, non-zero exit at parse | ✓ |
| Warn + sgfp4 wins | Quant flags ignored with warning print | |
| You decide | Claude picks post-research | |

**User's choice:** Hard error (rec)

| Option | Description | Selected |
|--------|-------------|----------|
| 4 conv types (rec) | Convolution, ConvolutionDepthwise, Deconvolution, DeconvolutionDepthwise; {oc, ic*kx*ky} flattened 2-D; MatMul-derived weights ride along only if TransformInnerProduct already converted them | ✓ |
| Convs + MatMul | Dedicated MatMul/InnerProduct case (attrs-based access, LLM path) — Phase 12+ territory | |
| You decide | Claude scopes post-research on final graph contents | |

**User's choice:** 4 conv types (rec)

| Option | Description | Selected |
|--------|-------------|----------|
| D-03 floor (rec) | Skip when elements < 4096 OR dimI == 1 (Phase 10 validated tiering); tiny tensors are pad-overhead-dominated | ✓ |
| No floor | Encode every conv weight regardless of size | |
| You decide | Claude applies Phase 10 rule, may tune constant with evidence | |

**User's choice:** D-03 floor (rec)

## Threshold sourcing

| Option | Description | Selected |
|--------|-------------|----------|
| Python defaults (rec) | Pass uses a dedicated converter config = kDefaultEncodeConfig (Python-identical); cross-repo default parity outranks promotion (Phase 10 rationale); delta documented in report | ✓ |
| Validated delta | Use the Phase 10 validated table (16/16 layers green) — best accuracy, diverges from gnus-poc defaults | |
| Defaults + override | Python defaults + --sgfp4Thresholds file override — extra CLI surface | |
| You decide | Claude decides post-research | |

**User's choice:** Python defaults (rec)
**Notes:** The Phase 10 carry-forward ("accept EncodeConfig explicitly rather than defaults") is satisfied by a greppable named constant (single swap-point for future upstream delta adoption), not a silent knob-less call.

## Tech-debt scope

| Option | Description | Selected |
|--------|-------------|----------|
| Retrofit now (rec) | Fix SGFP4ClassicAPITest.cpp:167-171 to region-relative offsets per the SGFP4TestUtil.hpp builder; ROADMAP-assigned | ✓ |
| Annotate only | Comment the known divergence; finding stays open as documented debt | |

**User's choice:** Retrofit now (rec)

| Option | Description | Selected |
|--------|-------------|----------|
| Hoist lambda (rec) | Hoist failCleanup above the two arg-validation returns so usage()-exits remove stale artifacts; matches README promise | ✓ |
| Scope README | Weaken the README claim to exclude arg errors | |

**User's choice:** Hoist lambda (rec)

| Option | Description | Selected |
|--------|-------------|----------|
| Include (env-var) (rec) | Env-var override (e.g. SGFP4_GNUS_POC_ROOT) in author_structured_fixture.py + siblings sharing the hardcode | ✓ |
| Defer W-3 | Leave for a future housekeeping pass; keep phase boundary at ROADMAP's W-1/W-2 | |

**User's choice:** Include (env-var) (rec)

## Test strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Extend TSGFP4C (rec) | Extend Phase 8's TestSGFP4Converter.cpp: synthetic NetT → pass ON → node insertion, rewiring, weights cleared, buffer populated, light-tier skip, subgraph coverage | ✓ |
| New executable | Separate TestSGFP4PostConverter.cpp — cleaner but duplicates solved CMake chain | |
| You decide | Claude decides post-research | |

**User's choice:** Extend TSGFP4C (rec)

| Option | Description | Selected |
|--------|-------------|----------|
| Real mnnconvert (rec) | Drive the real binary on approved AlexNet corpus; assert SGFP4Dequant nodes + classic-API decode; documented scripted step (corpus is a test-time dependency) | ✓ |
| Programmatic only | C++ converter entry in a test executable — CI-friendly but doesn't prove flag wiring | |
| Both | Programmatic gates + one manual smoke; Phase 12 formalizes E2E | |

**User's choice:** Real mnnconvert (rec)

| Option | Description | Selected |
|--------|-------------|----------|
| Flag-OFF green (rec) | Flag OFF → zero behavior change; 13 op/sgfp4 suites + converter tests green, no test edits; pass is dead code when flag absent | ✓ |
| + flag-ON corpus | Additionally sweep converter test corpus flag-ON for exotic-op crash-proofing | |
| You decide | Claude scopes post-research | |

**User's choice:** Flag-OFF green (rec)

---

## Claude's Discretion

- Pass registration string and file naming within postconvert/ conventions
- Exact pass ordering within the final RunNetPass batch (tensor-index bookkeeping vs ReIndexTensor interplay — planner verifies and locks)
- Named threshold constant's exact name/placement (converter-side vs. kDefaultEncodeConfig + comment) given greppable + Python-identical constraints
- D-05 mutex error message wording (enumerate flags individually vs collectively)
- Synthetic-net structure for D-12 beyond the listed assertions
- D-13 smoke scripting/documentation form (README vs test script) + decode-vs-FP32 tolerance wording
- D-11 env-var name alignment with any discovered gnus-poc convention

## Deferred Ideas

- MatMul/OpParameter_MatMul weight rewriting (LLM-export path) — future phase after 12
- --sgfp4Thresholds CLI file override — rejected this phase; revisit post upstream delta adoption
- Per-layer SGFP4 opt-out (SGV2-37 quantInfo-style) — future requirement
- Flag-ON converter corpus sweep beyond AlexNet — Phase 12
- gnus-poc upstream adoption of validated threshold delta — sibling-repo proposal flow
