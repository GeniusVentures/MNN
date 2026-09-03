# Workstream Archive

Inactive/dormant workstreams preserved here. Not deletable git history — kept browsable.

## milestone

**Project:** Vulkan Backend LLM Enablement (v1.0, 87% complete at archive time)
**Archived:** 2026-09-02 (files had been deleted from disk uncommitted; restored from HEAD `58ee88e6` and moved here)

**Status at archive:**
- Phases 1–3 complete (Vulkan Attention correctness, Ultra FP4 E2M1 quantization, TurboQuant docs + model-level regression)
- Phase 4: plan 04-01 done (TurboQuant-V + mask gen shader), **plan 04-02 (E2E model test) pending**
- Phase 5 (Vulkan TurboQuant regression tests) not started

**Cross-workstream notes (from sgfp4-pivot):**
- sgfp4-pivot Phase 9 minimally repaired this workstream's dead code in `test/op/FP4ModelTest.cpp` (deleted orphaned fragment; `FP4ModelConversionTest` preserved) — ratify or supersede if resumed
- Ultra FP4 (E2M1) and SGFP4 v2 are **different formats**: `milestone` owns FP4_ULTRA/E2M1, `sgfp4-pivot` owns SGFP4 v2 quadtree-adaptive — terminology lock documented in sgfp4-pivot ROADMAP
- Its sibling `sgfp4-pivot` shipped all 3 milestones (v1.0/v2.0/v3.0, tags exist) on 2026-09-02

**To resume:** move this directory back to `.planning/workstreams/milestone/` and read `STATE.md` (stopped_at: "Phase 4 plan 04-01 complete — next: 04-02").
