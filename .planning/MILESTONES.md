# Milestones

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
