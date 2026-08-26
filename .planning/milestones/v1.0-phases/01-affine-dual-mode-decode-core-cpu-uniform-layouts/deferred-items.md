# Deferred Items — Phase 01 (affine-dual-mode-decode-core-cpu-uniform-layouts)

Out-of-scope discoveries logged per the executor's scope-boundary rule
("only auto-fix issues directly caused by the current task's changes").

## 1. `test/op/FP4ModelTest.cpp` does not compile (pre-existing, unrelated to sgfp4-pivot)

- **Found during:** Plan 01-02, Task 2 verification (`cmake -DMNN_BUILD_TEST=ON
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON && cmake --build`).
- **Origin:** committed at `cffaf4bd` ("[Vulkan:Feature] TurboQuant-V + mask
  gen shader + FP4 tooling checkpoint; Phase 4 plan 04-01 complete"), on the
  `milestone` workstream, well before this `sgfp4-pivot` workstream existed.
- **Symptom:** the file has unreachable/dead code after an early `return
  true;` inside `FP4ModelConversionTest::run` (undeclared identifiers `pi`,
  `sc`, `refVec`, `outSz`; mismatched braces) — a genuine, compiler-agnostic
  syntax/semantic error, not an MSVC-specific quirk. It fails to compile with
  MSVC 19.44 (`error C2065`, `C3536`, `C2059`, etc.) and would fail with any
  standards-conformant C++ compiler.
- **Impact:** `run_test.out` cannot be built at all while this file is part
  of `test/op/`, because MNNTestSuite is a single monolithic binary (static
  self-registration) — there is no way to build a partial test binary.
- **Why deferred, not fixed:** `PROJECT.md`'s Key Decisions table records
  "Execute Phase 4 plan 04-02 (E2E FP4 model test) before starting any
  SGFP4 pivot work" as **Pending** (not yet done) — this file is explicitly
  Phase 4 plan 04-02's responsibility to complete/fix, not this phase's.
  Fixing it here would be out-of-scope architectural work on an unrelated
  subsystem (E2M1 Conv2D dequant model test), risking scope creep into
  another workstream's plan.
- **How this phase verified anyway:** for local build/test verification
  only (Plan 01-02, Task 2), the file's contents were temporarily replaced
  with a trivial neutral stub, `run_test.out` was built and the full suite
  run (375 passed / 0 failed, including `op/sgfp4/uniform_decode` and
  `op/fp4`), then the file was restored byte-for-byte via `git diff`
  verification (zero diff) before any commit. **No change to this file was
  committed.**
- **Recommended follow-up:** the `milestone` workstream's Phase 4 plan
  04-02 (E2E FP4 model test) should either finish or remove
  `FP4ModelTest.cpp`'s dead code so the full test suite is buildable again
  without a manual workaround.
