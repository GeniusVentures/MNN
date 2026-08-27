---
phase: 05-injection-core
plan: 02
subsystem: sgfp4-injection
tags: [sgfp4, tool, graph-surgery, sidecar, express]
key-files:
  - tools/fp4/sgfp4_inject.cpp
  - tools/fp4/sha256.hpp
  - tools/fp4/CMakeLists.txt
  - CMakeLists.txt
metrics:
  end_to_end_exit_code: 0
  artifact_model_bytes: 480
  artifact_sidecar_bytes: 132368
  negative_paths_exit_codes: [1, 1]
---

# Plan 05-02 Summary: sgfp4_inject Standalone Injection Tool

## Objective

Build the standalone `sgfp4_inject` tool: given a normally-converted `.mnn`
plus gnus-poc `fp4_exporter.py --adaptive` output directories, produce a new
`.mnn` + merged external sidecar where each target weight tensor is produced
by an `OpType_SGFP4Dequant` node — using the exact recipe proven by 05-01.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 + 2 | 3b4c43b4 | `tools/fp4/sha256.hpp` (vendored SHA-256), `tools/fp4/CMakeLists.txt`, root `CMakeLists.txt` option/include, `tools/fp4/sgfp4_inject.cpp` (single-TU tool) |

## What Was Done

- **`tools/fp4/sha256.hpp`** — self-contained public-domain SHA-256 (FIPS
  180-4), include guard `TOOLS_FP4_SHA256_HPP`, single public fn
  `sgfp4::sha256_hex(const uint8_t*, size_t) -> std::string` (64-char
  lowercase hex). No OpenSSL, no `<filesystem>` (C++11).
- **CMake wiring** — `option(MNN_BUILD_SGFP4_TOOLS ... OFF)` right after
  `MNN_BUILD_QUANTOOLS`; include block after the QUANTOOLS block;
  `tools/fp4/CMakeLists.txt` mirrors `tools/quantization/CMakeLists.txt`
  (glob `*.cpp`+`*.hpp`, `sgfp4_inject.out` target, MSVC `_CRT_SECURE_NO_WARNINGS`/
  `/WHOLEARCHIVE` block verbatim).
- **`tools/fp4/sgfp4_inject.cpp`** — single TU, CLI `--model/--niche-dir
  (repeatable)/--output`, sidecar `=<output>.weight`:
  - Per niche dir: discovers the unique `*.sgfp4` (case-insensitive; 0 or >1
    → hard error), parses `manifest.json` via vendored rapidjson DOM, reads
    `fp4_binary.sha256` + `fp4_binary.path` (basename cross-check only,
    never resolved literally) + `fp4_binary.stats.shape` (exactly 2 positive
    ints, else hard error), verifies sha256 of container bytes against the
    manifest (mismatch → hard error, D-03), byte-level version gate
    `sgfp4_is_v2_container` (never consults `fp4_binary.format`, SGINJ-01).
  - Pairing by exact `{dimO, dimI}` shape among non-input 2-D vars (D-02/
    D-04); zero-or-multiple matches → hard error listing candidates.
  - Op construction near-copies 05-01: `OpType_SGFP4Dequant` +
    `SGFP4DequantParamT{magic=kSGFP4Magic, external{offset,size}, dims}` with
    `op->externalPath` set literally (Pitfall 2); node named
    `<weight>_sgfp4` (D-08); `Variable::replace(weightVar, dequantVar)`
    (D-06) — original Const dead-dropped naturally (D-07).
  - Sidecar merge: single `<output>.weight` stream, offsets cursor advancing
    by `MNN::sgfp4_align16(containerSize)`, padding written, non-overlapping
    monotonic (D-11, SGINJ-03). Offsets computed BEFORE op construction so
    spliced ops carry final `{offset, size}`.
  - Serialize: outputs recomputed AFTER all rewiring;
    `Variable::save(outputs, outputPath.c_str())` direct-to-file (SGINJ-04).
  - In-tool verify (D-12/D-13): unconditional full-artifact
    `Module::load` (rtmgr with `setExternalFile(sidecar)` before load), then
    per-node isolated 0-input sub-modules saved to a temp path and reloaded
    under the same rtmgr, compared element-wise against
    `dequant_sgfp4_container_cpu` at rtol 1e-4f (local equivalent of
    checkVectorByRelativeError). Any mismatch/non-null failure → nonzero exit.

## Verification Results

- Build: `cmake --build . --target sgfp4_inject.out` (MSVC, config Release,
  with `-DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_SGFP4_TOOLS=ON`) — OK.
- **End-to-end** (real demo niche dir, real `minimal_512.mnn` with a
  [512,512] MatMul weight): exit 0, emitted `out.mnn` (480 B) +
  `out.mnn.weight` (132,368 B); in-tool verify printed
  `node 'weight' {512,512} offset=0 size=132368 verified (decode==oracle)`.
- **Negative paths**: corrupted magic byte with sha256 fixed to match →
  version-gate diagnostic, exit 1 (SGINJ-01 in-tool); unmodified manifest +
  corrupted magic → sha256-mismatch diagnostic, exit 1 (D-03).
- **sha256 KAT**: empty-string digest =
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  (KAT-PASS); demo container digest matches Python `hashlib.sha256`
  (`ce226cb4...4f974b`) — multi-block + padding exercised for real.
- Prior SGFP4 suites were green after 05-01 (5/5); no runtime code touched
  by this plan (tool + CMake + additive header only).

## Deviations

- Runtime is Windows/MSVC (`.build`, PowerShell), not MSYS2/MinGW as the
  plan's shell notes assumed — same commands via `cmake --build`, identical
  exit-code/PASS semantics.
- `windows.h` `min`/`max` macro clash fixed via `NOMINMAX` guard before
  `<windows.h>` in the directory-listing helper (Win32) with a `dirent.h`
  fallback for POSIX.
- Directory listing uses `FindFirstFileA`/`dirent.h` (no `<filesystem>`,
  keeping the C++11 default).
- The FP4ModelTest.cpp full-suite build blocker (STATE.md known issue) was
  bypassed by filtering the **untracked generated** `.vcxproj` only (see
  05-01 summary); no tracked file modified.

## Self-Check

PASSED — all acceptance criteria hold: required tokens present; no
`replaceInput` and no `fp4_binary.format` version-gate consultation;
`op->externalPath` literal; `sgfp4_align16` single-stream non-overlapping
sidecar; build + end-to-end + negative smokes + KAT all verified.
