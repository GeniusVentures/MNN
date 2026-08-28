# Milestones

## v2.0 SGFP4 v2 Model-Artifact Injection Tool (Shipped: 2026-08-28)

**Phases completed:** 3 phases, 7 plans (Phases 5-7, workstream `sgfp4-pivot`)
**Closeout:** verified — milestone audit passed 8/8 (`.planning/milestones/v2.0-MILESTONE-AUDIT.md`)

**Key accomplishments:**

- Express graph-surgery spike proving `Variable::replace` consumer-rewiring (spliced `OpType_SGFP4Dequant` node verifiably replaces the original weight Const) + byte-level v2 version gate `sgfp4_is_v2_container` (Phase 5)
- Standalone `sgfp4_inject` tool (`MNN_BUILD_SGFP4_TOOLS=ON`): manifest-driven shape pairing, SHA-256 integrity gate, merged 16-byte-aligned non-overlapping sidecar, direct-to-file `Variable::save`, unconditional in-tool decode==oracle verification (Phase 5)
- Classic Interpreter/Session API validation: injected artifacts load/run via `createFromFile → createSession → runSession` with named I/O surviving injection verbatim, FP32 parity rtol 1e-4 on CPU, sidecar resolving via the op's literal `externalPath` — the exact downstream `SGProcessingManager` path (Phase 6)
- Structured LAYOUT_MIXED fixture from the real gnus-poc encoder (140,240 B, 12 MIXED superblocks, byte-deterministic regeneration) exercising the quadtree decode path end-to-end (Phase 7)
- Multi-tensor injection (2 containers, disjoint collision-free ranges, byte-identity) + 13-probe malformed-input clean-failure matrix guaranteeing zero partial or stale artifacts (Phase 7)

**Known tech debt (non-blocking, detailed in v2.0-MILESTONE-AUDIT.md):**

- W-1: `classic_api` test container uses absolute vs. region-relative offset-table convention (parity still valid decode-vs-decode; `multi_tensor` covers the encoder-conformant path)
- W-2: arg-validation failures skip failCleanup (README over-promises stale removal on arg errors)
- W-3: `author_structured_fixture.py` hard-codes the gnus-poc absolute path (env-var override suggested)
- Duplicated test helpers across the three SGFP4 test files (invited the W-1 drift) — extract `SGFP4TestUtil.hpp`

## v1.0 SGFP4 v2 Decode (Vulkan-parity) (Shipped: 2026-08-26)

**Phases completed:** 4 phases, 10 plans, 20 tasks

**Key accomplishments:**

- CPU affine dual-mode decode (`w = S·c + bias`) for FP4_AFFINE + T158_AFFINE, v2 stream framing, and external-sidecar container loading (Phase 1)
- Standalone SGFP4 v2 encoder (`tools/fp4/encode_sgfp4.py`) with an independent Python reference decoder and a CPU test suite (`op/sgfp4/uniform_decode`) proving round-trip decode for both affine code modes across all five uniform layouts (Phase 1)
- CPU LAYOUT_MIXED adaptive quadtree decode (pre-order DFS, TL/TR/BL/BR) plus an error-driven encoder with per-level MSE thresholds and ternary outlier veto (Phase 2)
- Vulkan GLSL uniform-layout decode shader with CPU/Vulkan parity within float tolerance (Phase 3)
- GLSL bounded split-map walk in `locateElement` decodes SGFP4 v2 LAYOUT_MIXED on GPU — deleted the one-fixture skip so all 14 committed fixtures (uniform + mixed) run through CPU oracle and real Vulkan GPU dispatch, closing SGV2-16 (Phase 4)
- Fully additive to the existing E2M1 Ultra FP4 path — zero changes to the live cross-repo `dequant_fp4_packed_cpu()` contract

**Known gaps:**

- Phase 2 (`02-VERIFICATION.md`) never had its formal verification report generated — the checkbox-flip step never ran. Work is evidenced by 02-01/02-02-SUMMARY.md and built upon without issue by Phases 3-4. Acknowledged and deferred; run `/gsd-verify-work 2` retroactively if a formal report is needed.

---
