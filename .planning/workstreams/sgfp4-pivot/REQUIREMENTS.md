# Requirements: MNN SGFP4 v2 Converter Integration

**Defined:** pending `/gsd-new-milestone` for v3.0 (placeholder after v2.0 close on 2026-08-28)
**Core Value (draft):** A pytorch → onnx → mnnconvert pipeline can produce a `.mnn` file that runs SGFP4 v2 (quadtree-adaptive) FP4 inference, via a native mnnconvert CLI flag.

> **v2.0 shipped 2026-08-28** — the Model-Artifact Injection Tool (`tools/fp4/sgfp4_inject`) closed the "nothing produces a real loadable `.mnn` using `OpType_SGFP4Dequant` on real weights" gap. Its requirements (SGINJ-01..08) are archived complete in `.planning/milestones/v2.0-REQUIREMENTS.md`, with the milestone audit (passed 8/8) at `.planning/milestones/v2.0-MILESTONE-AUDIT.md`.
>
> **Carry-forward requirements (SGV2-22..32)** for Phases 8-12 live in `.planning/milestones/v2.0-REQUIREMENTS.md` §"v2 Requirements / v3.0 Converter Integration" until re-mapped by `/gsd-new-milestone`:
>
> - **SGV2-22/23** → Phase 8 (Schema + Sidecar Wiring): `SGFP4DequantParam.buffer:[byte]` + `RemoveParams.cpp` externalization
> - **SGV2-24/25** → Phase 9 (Real-Weight C++ Encoder Port): python→C++ port, non-64-multiple tiling policy — **re-evaluate at plan time** vs. direct consumption of gnus-poc `fp4_exporter.py --adaptive` output
> - **SGV2-26/27** → Phase 10 (Real-Weight Validation): encoder params vs. real weight statistics
> - **SGV2-28/29/30** → Phase 11 (Graph-Rewrite PostConverter Pass + CLI Flag) — absorb v2.0 sidecar/rewiring learnings; retire v2.0 tech debt (W-1 classic_api offset-convention retrofit, W-2 arg-stage failCleanup, W-3 portable gnus-poc root, `SGFP4TestUtil.hpp` dedup)
> - **SGV2-31/32** → Phase 12 (End-to-End Validation): CPU + Vulkan

## v1 Requirements

To be defined by `/gsd-new-milestone` (requirements questioning → research → mapping).

## Out of Scope

Carried forward from v2.0 (still applies):

- Calibration/activation-data requirements (GPTQ/AWQ/`--imatrix`-style) — would break MNN's zero-calibration UX
- Retrofitting `IDSTEncoder`/`IDSTQuan` for SGFP4 — rejected in the pivot analysis
- First-class FlatBuffers schema fields for quadtree internals — locked to opaque external-file blob + minimal `{magic, offset, size}` descriptor
- Ultra FP4 (E2M1) converter integration — separate `milestone` workstream, additive only
- Attestation / verifiable-execution support — SuperGenius verifies separately

## Traceability

To be built during v3.0 milestone creation.

---
*Created 2026-08-28 at v2.0 close (v2.0 requirements archived to `.planning/milestones/v2.0-REQUIREMENTS.md`)*

