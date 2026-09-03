# Milestones

## v3.0 SGFP4 v2 Converter Integration (Shipped: 2026-09-02)

**Phases completed:** 5 phases, 21 plans (Phases 8-12, workstream `sgfp4-pivot`)
**Closeout:** verified — 11/11 requirements (SGV2-22..32); 09/12-VERIFICATION.md consolidated from UAT evidence at close (v2.0 Phase-5 precedent); no standalone milestone audit (per-phase VERIFICATION reports + 10/10 Phase-12 UAT instead)

**Key accomplishments:**

- Schema `SGFP4DequantParam.buffer:[byte]` + buffer-first decode dispatch (CPU+Vulkan) + aligned 16-byte sidecar externalization in `RemoveParams.cpp`; `SGFP4TestUtil.hpp` dedup (Phase 8)
- Byte-exact C++ port of gnus-poc `fp4_exporter.py --adaptive` (`tools/fp4/sgfp4_encode`) — 0 diff bytes vs Python; padded-crop decode path on CPU + Vulkan for non-64-aligned real shapes; deterministic committed fixtures (Phase 9)
- Real-weight validation on the AlexNet corpus: D-07 gate reformulated (plain worst-leaf MSE), threshold convergence 581→0 in 3 iterations, committed acceptance report; `EncodeConfig` with Python-identical defaults (promotion declined to preserve cross-repo parity) (Phase 10)
- `InsertSGFP4Dequant` PostConverter pass (buffer-staged, idempotent, transactional failures, 4-D conv dims) + `--sgfp4` CLI flag with hard mutex + `WeightQuantAndCoding` skip-guard; v2.0 tech debt W-1/W-2/W-3 retired (Phase 11)
- E2E gate `tools/fp4/e2e_validation.ps1`: `mnnconvert --sgfp4` artifact runs correct inference on CPU + Vulkan vs FP32 baseline (locked tolerances, D-10 diagnostics, exit-code honesty) — surfacing and fixing 2 correctness-critical codec defects: spatial decode convention + encoder split-map coordinates/order (Phase 12)

**Known gaps / deferred:**

- D-09: threshold-table promotion in gnus-poc defaults — upstream proposal route; consumers pass `EncodeConfig` explicitly
- E2E corpus (AlexNet ONNX) is a test-time dependency, documented in `tools/fp4/README.md` — not an always-on CI gate
- Vulkan leg requires a real device by design (hard FAIL, never SKIP)
- `MNN_BUILD_SGFP4_TOOLS` defaults OFF — release-packaging note for CLI binaries
- SGV2-33..37 (Performance & Coverage) never scoped — deferred candidates for v4.0

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
