# Phase 8: Schema + Sidecar Wiring - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-28
**Phase:** 8-Schema + Sidecar Wiring
**Areas discussed:** buffer field semantics, sidecar write convention, Phase 8 test scope, converter-pass hand-off contract

---

## Gray Area Selection

All four presented areas were selected for discussion (multiSelect): buffer field semantics, sidecar write convention, Phase 8 test scope, converter-pass hand-off contract.

Prior decisions carried forward (not re-asked): minimal `{magic, offset, size}` descriptor lock (2026-08-22), literal `op->externalPath` on the op / not in `createExecutionWithExternal`'s rewrite set (SGINJ-02), injection tool's 16-byte-aligned merged sidecar convention (v2.0), terminology lock ("SGFP4 v2").

---

## Buffer Field Semantics

Presented with code grounding: `CPUSGFP4Dequant::onResize` lines ~48-53 hard-gates on `USE_EXTERNAL_DATA(param) && mOp->externalPath()` → `NOT_SUPPORT`.

| Option | Description | Selected |
|--------|-------------|----------|
| A — Converter-transient | buffer never serialized; exists only in converter memory between pass and storeWeight; no runtime change | |
| B — Decoder fallback | live schema field: non-empty buffer → decode from it; empty → sidecar path; mirrors Blob; enables single-file .mnn; CPU+Vulkan Execution update | ✓ |
| C — Conditional dual-mode | serialize buffer inline when needExternalWeight OFF; storeWeight moves to sidecar when ON; runtime stays sidecar-gated | |

**User's choice:** B — decoder fallback
**Notes:** No additional rationale provided beyond the option descriptions.

---

## Sidecar Write Convention

Presented with code grounding: `storeWeight` raw-concatenates (`offset += size`, no padding) while `sgfp4_inject` pads every region to 16-byte alignment (v2.0 audit evidence: 132,368 B = 16×8,273).

| Option | Description | Selected |
|--------|-------------|----------|
| R1 — Plain storeWeight | stock `storeWeight<uint8_t>` like Blob's uint8s; unpadded; convention diverges from injection tool | |
| R2 — 16B aligned | pad region to 16-byte multiple (zero-filled) before advancing offset; matches sgfp4_inject; padding inert to decoder | ✓ |
| R3 — Keep buffer | write to sidecar but leave param->buffer intact; robust to sidecar loss but doubles artifact size, silent dual-source ambiguity | |

**User's choice:** R2 — 16B aligned
**Notes:** No additional rationale provided beyond the option descriptions.

---

## Phase 8 Test Scope

Presented with audit context: v2.0 flagged duplicated helpers across three SGFP4 test files; W-1 offset-convention divergence born from that duplication; audit originally placed helper dedup in Phase 11.

| Option | Description | Selected |
|--------|-------------|----------|
| T1 — Decode parity only | buffer-mode vs sidecar-mode vs oracle on CPU+Vulkan; no converter-path test; alignment decision ships unverified until Phase 11 | |
| T2 — + Converter round-trip | T1 plus drive RemoveAndStoreParam/saveExternalData asserting 16B-aligned monotonic non-overlapping layout + reload parity; dedup deferred to Phase 11 | |
| T3 — Full incl. dedup | T2 plus SGFP4TestUtil.hpp extraction now, retrofitting the three existing test files; Phase 11 starts on clean helpers | ✓ |

**User's choice:** T3 — full incl. dedup
**Notes:** No additional rationale provided beyond the option descriptions.

---

## Converter-Pass Hand-Off Contract

Presented with code grounding: `createExecutionWithExternal` (OpCommonUtils.cpp:665) only intercepts Convolution2D/Scale/LayerNorm; SGFP4 ops pass unmodified to `backend->onCreate` — decoder owns dispatch.

| Option | Description | Selected |
|--------|-------------|----------|
| H1 — Buffer staging | Phase 11 pass writes OpT with buffer=[bytes], external={}, no externalPath; pure graph rewrite; existing converter flags decide externalization via D-2 storeWeight case | ✓ |
| H2 — Dual-emit documented | H1 plus Phase 8 explicitly tests both converter outcomes (externalization ON and OFF) | |
| H3 — Inject-tool parity | Phase 11 links sgfp4_inject core, emits sidecar + externalPath directly; drags W-1/W-2 into Phase 11's critical path | |

**User's choice:** H1 — buffer staging
**Notes:** No additional rationale provided beyond the option descriptions.

---

## Claude's Discretion

- Schema comment wording in `CaffeOp.fbs`; flatc regeneration flow details (committed generated headers)
- Internal structure of the aligned storeWeight case (helper vs inline pad logic)
- Test file naming/placement; parameterized vs separate CPU/Vulkan parity tests
- Buffer-mode entry validation strength (beyond magic + dims consistency)

## Deferred Ideas

- Non-64-multiple tiling/padding conventions → Phase 10
- CLI flag design → Phase 11
- W-1 classic_api offset retrofit, W-2 arg-stage failCleanup → Phase 11 (audit placement; only helper dedup moved to Phase 8)
- Inline-threshold convenience (auto-inline small containers) → rejected for now; externalization stays flag-driven
- Extending `createExecutionWithExternal` to SGFP4 → explicitly rejected
