# Requirements: MNN SGFP4 v2 — next milestone

**Defined:** pending `/gsd-new-milestone --ws sgfp4-pivot` (placeholder after v3.0 close on 2026-09-02)

> **v3.0 shipped 2026-09-02** — Converter Integration closed the full pytorch → onnx → `mnnconvert --sgfp4` → SGFP4 v2 inference pipeline (CPU + Vulkan, E2E accuracy-gated vs FP32 baseline). Its requirements (SGV2-22..32, 11/11) are archived complete in `.planning/milestones/v3.0-REQUIREMENTS.md`; milestone summary at `.planning/milestones/v3.0-ROADMAP.md`.
>
> **Deferred candidates for v4.0** (never scoped; carried from v2.0-REQUIREMENTS.md §"Performance & Coverage"):
> - SGV2-33: GPU decode performance tuning (workgroup sizing, fused dequantize→matmul)
> - SGV2-34: Additional macroblock geometries / payload alignments
> - SGV2-35: SGFP4 v2 decode on Metal / CUDA / OpenCL
> - SGV2-36: Optional Laplacian-pyramid error weighting in encoder cost function
> - SGV2-37: Per-layer SGFP4 opt-out + bias/BN folding parity

## v1 Requirements

To be defined by `/gsd-new-milestone` (requirements questioning → research → mapping).

## Out of Scope

Carried forward from v3.0 (still applies):

- Calibration/activation-data requirements (GPTQ/AWQ/`--imatrix`-style) — would break MNN's zero-calibration UX
- Retrofitting `IDSTEncoder`/`IDSTQuan` for SGFP4 — rejected in the pivot analysis
- First-class FlatBuffers schema fields for quadtree internals — locked to opaque blob + minimal descriptor
- Ultra FP4 (E2M1) converter integration — separate `milestone` workstream, additive only
- Attestation / verifiable-execution support — SuperGenius verifies separately
- D-09 threshold-table promotion in gnus-poc defaults — upstream proposal route; consumers pass `EncodeConfig` explicitly

## Traceability

To be built during next milestone creation.

---
*Created 2026-09-02 at v3.0 close (v3.0 requirements archived to `.planning/milestones/v3.0-REQUIREMENTS.md`)*
