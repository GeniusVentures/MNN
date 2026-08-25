# Phase 3: Vulkan Decode — Uniform Layouts - Research

**Researched:** 2026-08-24
**Domain:** Vulkan buffer-backend GLSL compute shader + Vulkan Execution class porting the CPU-validated SGFP4 v2 uniform-layout decode (affine dual-mode, external-sidecar container)
**Confidence:** HIGH (code-level facts verified by direct file reads; environment facts verified by live terminal probes)

## Summary

Phase 3 ports `dequant_sgfp4_container_cpu()` (`include/MNN/SGFP4DequantUtils.hpp`, 559 lines, read in full) to the Vulkan buffer backend as a new GLSL compute shader + `VulkanBasicExecution` subclass. Every structural ingredient is already proven in-tree: `VulkanFP4Dequant.{hpp,cpp}` is a working end-to-end template (constructor pipeline selection with FP16/FP32 variants, uniform const buffer at binding 2, `vkCmdDispatch` with `UP_DIV(…, 256)`, `barrierSource`), `fp4_dequant.comp` establishes the GLSL conventions (no `#version`, `FLOAT` macro from makeshader.py, `layout(local_size_x = 256)`), `CPUSGFP4Dequant.cpp` is the exact host-side external-sidecar loading pattern (FileLoader + real `std::ifstream` size probe per T-01-04), and `test/op/VulkanFP4DequantTest.cpp` is the exact dual-backend test harness pattern (Vulkan-availability guard + `MNN_FORWARD_VULKAN` schedule config). No CMake edits are needed anywhere: `source/backend/vulkan/CMakeLists.txt` and `test/CMakeLists.txt` both use `GLOB_RECURSE`.

The two genuine technical novelties vs `VulkanFP4Dequant` are (1) the shader must read **arbitrary byte offsets** of the container framing (e.g. record count `B` lives at byte 5, straddling aligned u32 words — GLSL SSBO `uint[]` indexing only addresses aligned words, so a little-endian byte-composition helper is required), and (2) the Execution is **Const-like with 0 input tensors** and gets its bytes from the external sidecar on the host side, uploaded once into a `VulkanBuffer` SSBO — unlike `VulkanFP4Dequant` which consumes an in-graph input tensor. One critical build-infrastructure finding: `glslangValidator` is currently installed **nowhere** (WSL ✓ has `python3` + `xxd`, ✗ no glslang/spirv-opt; Windows ✗ no Vulkan SDK on PATH), so makeshader.py regeneration requires a one-time toolchain install (WSL `glslang-tools` needs a sudo password, or Vulkan SDK + Git Bash on Windows because makeshader.py uses POSIX `find`/`os.popen` that Windows `find.exe` does not satisfy). Also: the current `.build` directory is configured with `MNN_VULKAN:BOOL=OFF` and `MNN_VULKAN_IMAGE` defaults ON — the buffer backend requires reconfiguring with `-DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF`.

**Primary recommendation:** Build the Execution by direct structural cloning of `VulkanFP4Dequant` with host-side container loading cloned from `CPUSGFP4Dequant::onResize` (moved into the creator/constructor, since `VulkanBasicExecution` has no `onResize` hook), a stateless thread-per-output-weight shader (D-03/D-04) that re-walks the framing per thread via a GLSL `read_u32_le(byteAddr)` helper, host pre-validation by one scratch-buffer pass of the already fully-bounds-checked `dequant_sgfp4_container_cpu` (D-05), and a sibling parity test `test/op/SGFP4VulkanDequantTest.cpp` cloned from `VulkanFP4DequantTest.cpp` + Phase 1's `runSgfp4Module` (0-input op module with `op->externalPath` set directly).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Host-side upload of the whole container. The Vulkan Execution's `onResize` reads the container bytes from the external sidecar exactly like the CPU path (`FileLoader` + real `std::ifstream` file-size probe per Phase 1 T-01-04), then copies the bytes into a `VulkanBuffer` bound as an SSBO. The shader reads raw container bytes from that storage buffer. No sparse-memory/file-backed extensions, no CPU-side frame parsing split.
- **D-02:** The parity test drives the identical container fixtures through both backends — same sidecar file, same descriptor — so the GPU path is exercised through the same external-container plumbing as production, not through a synthetic in-graph tensor.
- **D-03:** Thread-per-weight, stateless. One linear thread per output weight (mirrors `fp4_dequant.comp`'s element-parallel mapping). No workgroup-per-record staging, no inter-workgroup sync, no prefix/index kernel. Workgroup sizing follows the existing `VulkanFP4Dequant` convention and is planner's discretion.
- **D-04:** Each thread performs the full framing re-walk (magic/version → offset table → its record → leaf header → payload word) redundantly to locate its weight. Deliberate trade: simplicity now over indexing machinery Phase 4 would have to rebuild.
- **D-05:** Host pre-validates. `onResize`-equivalent validates the full container structure ONCE using the existing header-only checks in `SGFP4DequantUtils.hpp` before any dispatch. Only a validated container is uploaded and dispatched. Malformed containers rejected with CPU-path error semantics (error return, no partial output writes). The shader does NOT duplicate bounds-checking.
- **D-06:** FP16 default + FP32 variant, mirroring `VulkanFP4Dequant` D-04: `vkBn->useFP16()` selects the FP16-output shader variant, otherwise the FP32 variant. Shader naming `glsl_sgfp4_dequant_FP16_comp` / `glsl_sgfp4_dequant_comp`; both variants embedded via `makeshader.py` with regenerated `AllShader.cpp` / `AllShader.h` / `VulkanShaderMap.cpp` committed (locked roadmap note 5).
- **D-07:** One dual-backend C++ test registered in `MNNTestSuite`: each uniform-layout fixture decoded via `dequant_sgfp4_container_cpu()` AND via a Vulkan session through the new Execution, compared with `checkVectorByRelativeError` within float tolerance. Test degrades gracefully (skip + clear message) when no Vulkan device; the Windows build/test machine has one.

### Claude's Discretion
- GLSL helper decomposition inside the `.comp` file(s) (single file with variants vs. shared include structure, subject to `makeshader.py` pipeline constraints).
- Workgroup size and dispatch arithmetic (constant, named — no magic numbers).
- C++ code organization of the Execution: whether container validation helpers are reused directly from `SGFP4DequantUtils.hpp` or wrapped in small host-side helpers in the Execution.
- Test registration naming within the existing `op/sgfp4/...` namespace.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope. (GPU workgroup/coalescing/shared-memory tuning is SGV2-18 backlog; quadtree GPU walk is Phase 4 by design.)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SGV2-12 | GLSL compute shader decodes uniform-layout SGFP4 v2 containers on the Vulkan buffer backend (FP4_AFFINE + T158_AFFINE affine reconstruction via shift-mask-FMA), embedded via `makeshader.py` with regenerated `AllShader.cpp` / `AllShader.h` / `VulkanShaderMap.cpp` | Shader conventions, FP16 variant generation mechanism (`macro.json` `useFP16`), naming scheme (`glsl_sgfp4_dequant[_FP16]_comp`), and the byte-level GLSL decode math (incl. the unaligned `read_u32_le` problem and `unpackHalf2x16` header recovery) all documented below with verified code examples |
| SGV2-13 | New Vulkan Execution class registered in the buffer-backend execution table, reading the same `{magic, offset, size}` external-sidecar descriptor as the CPU path | `VulkanFP4Dequant` creator/registration pattern, `CPUSGFP4Dequant` FileLoader + ifstream-probe host loading pattern, `VulkanBuffer` hostData/map upload mechanics, and the "no `onResize` on `VulkanBasicExecution`" timing constraint all verified |
| SGV2-14 | CPU/Vulkan decode-parity test for uniform-layout containers within float tolerance, passing via `./run_test.out` | `VulkanFP4DequantTest` harness pattern (availability guard, `MNN_FORWARD_VULKAN` config, `Precision_High` → FP32 output), Phase 1 `runSgfp4Module` 0-input-module pattern (`op->externalPath` set directly), fixtures in `test/op/SGFP4DequantFixtures.h`, `test/CMakeLists.txt` auto-glob; known `FP4ModelTest.cpp` build blocker + workaround documented |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Container bytes → GPU-visible SSBO | Vulkan Execution host code (C++) | — | D-01: host reads sidecar via FileLoader; only the host has filesystem access; `VulkanBuffer` (HOST_VISIBLE pool memory) is the standard upload vehicle |
| Container structural validation (ASVS V5) | Host, pre-dispatch (C++) | — | D-05: one-time validation using `SGFP4DequantUtils.hpp` checks; shader never sees unvalidated bytes, so no OOB convention needed |
| Framing walk (magic → records → leaves) | GLSL compute shader per thread | — | D-03/D-04: stateless redundant re-walk per thread keeps the shader index-free |
| Affine reconstruction `w = S·c + bias` | GLSL compute shader | — | Pure arithmetic; shift-mask-FMA per element |
| FP16 leaf-header unpack (S/bias/mode) | GLSL compute shader | — | `unpackHalf2x16` covers both halves; flags from masked low nibble |
| Op schema / shape inference | Already done (Phase 1) | — | `OpType_SGFP4Dequant`, `SGFP4DequantParam {magic, external{offset,size}, dims}`, `ShapeSGFP4Dequant` exist and are registered; this phase touches none of it |
| Execution registration | `VulkanBackend::addCreator` static lambda | — | Established buffer-backend pattern (`VulkanFP4Dequant.cpp:91-96`) |
| Dual-backend parity verification | `MNNTestSuite` C++ test | — | D-07; same fixtures, same sidecar file, `checkVectorByRelativeError` |
| SPIR-V embedding | `makeshader.py` pipeline (offline) | — | Locked CLAUDE.md contract; regenerated artifacts committed |

## Standard Stack

No external packages are installed in this phase. The stack is entirely MNN-internal (verified by direct code reads).

### Core
| Component | Location | Purpose | Why Standard |
|-----------|----------|---------|--------------|
| `VulkanBasicExecution` | `source/backend/vulkan/buffer/execution/VulkanBasicExecution.hpp` | Base class for the new Execution (`onEncode` interface) | Every buffer-backend op derives from it; `VulkanBasicExecutionDirect::onResize` calls `onEncode` during resize, wrapping commands in a command buffer |
| `VulkanBuffer` | `source/backend/vulkan/component/VulkanBuffer.hpp` | Container upload SSBO + const UBO | Pool-backed; ctor accepts `hostData` for direct upload, `map()/unmap()` for host writes (59 map() usages across the backend) |
| `VulkanBackend::getPipeline / addCreator / useFP16` | `source/backend/vulkan/buffer/backend/VulkanBackend.cpp` | Pipeline lookup by shader key, op-type registration, FP16 selection | `getPipeline(key, descriptorTypes)` (line 152); `useFP16()` = precision != `Precision_High` && device FP16 support (line 102) |
| `makeshader.py` | `source/backend/vulkan/buffer/compiler/makeshader.py` | GLSL → SPIR-V → C arrays embedding | Locked CLAUDE.md contract; discovers `glsl/*.comp` via POSIX `find`, prepends FP32/FP16 headers (defines `FLOAT`), compiles via `glslangValidator`, embeds via `xxd -i` |
| `dequant_sgfp4_container_cpu` + constants | `include/MNN/SGFP4DequantUtils.hpp` | Normative decode ported to GLSL; host pre-validation | Phase 1/2-validated; all framing constants, leaf-header unpack math, payload packing normative here |
| `MNNTestSuite` | `test/` | Parity test registration + `run_test.out` runner | `test/CMakeLists.txt:12` GLOB_RECURSE picks up new test files automatically |

### Supporting
| Component | Location | When to Use |
|-----------|----------|-------------|
| `test/op/SGFP4DequantFixtures.h` | committed | Parity test fixtures (both modes × all 5 uniform layouts + b≠0 alignment + mixed); expected weights embedded → CPU reference at test time |
| `test/op/VulkanFP4DequantTest.cpp` | committed | Template for the Vulkan availability guard + schedule config + tolerance handling |
| `test/op/SGFP4DequantTest.cpp::runSgfp4Module` | committed | Template for the 0-input Const-like op module build (`op->externalPath` set directly on OpT — Phase 1 pitfall) |
| `FileLoader` + `std::ifstream` size probe | `source/core/FileLoader.hpp`, pattern in `CPUSGFP4Dequant.cpp:22-49` | D-01/T-01-04 host sidecar read; probe BEFORE allocation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Whole-container SSBO upload (D-01) | Host-side parse → leaf table UBO | Locked OUT by D-01; would also be rebuilt by Phase 4 quadtree anyway |
| Per-thread framing re-walk (D-04) | Separate index kernel / workgroup-per-record staging | Locked OUT by D-03/D-04; deliberately simple, uniform-layout-negligible |
| Scratch-decode host validation (see D-05 section) | Hand-rolled validate-only walker | Scratch-decode reuses the fully-bounds-checked Phase-1-tested walk at zero new code; recommended (planner's call) |

**Installation (toolchain prerequisite, not packages):**
```bash
# WSL option (python3 + xxd verified present; glslangValidator MISSING - sudo needs password):
sudo apt-get install glslang-tools spirv-tools   # spirv-tools optional; makeshader.py skips spirv-opt if absent
```

## Package Legitimacy Audit

No external packages are installed by this phase (C++ stdlib + MNN-internal code only; the only tooling additions are system packages `glslang-tools`/Vulkan SDK for the offline shader compiler). Not applicable — no registry packages introduced.

## Architecture Patterns

### System Architecture Diagram

```mermaid
flowchart TD
    subgraph Host["Host (C++) — VulkanSGFP4Dequant"]
        A[OpT: OpType_SGFP4Dequant<br/>external = {offset, size}<br/>op->externalPath = sidecar] --> B[ifstream size probe<br/>T-01-04 DoS bound]
        B --> C[FileLoader offset+size read<br/>→ mContainer bytes]
        C --> D[Host pre-validation ONCE<br/>dequant_sgfp4_container_cpu<br/>into scratch (D-05)]
        D -->|valid| E[VulkanBuffer SSBO upload<br/>container bytes]
        D -->|malformed| F[return error<br/>no dispatch, no partial writes]
        E --> G[VulkanBuffer UBO const<br/>containerWords, outElementCount]
    end
    subgraph GPU["GPU — glsl_sgfp4_dequant[_FP16]_comp"]
        H[thread idx = gl_GlobalInvocationID.x] --> I{idx < outElementCount?}
        I -->|yes| J[Framing re-walk (D-04):<br/>magic/version → B → record_offsets<br/>→ sb_header layout → block headers<br/>→ locate leaf containing idx]
        J --> K[read_u32_le byte-composition<br/>of leaf header + payload word]
        K --> L[unpackHalf2x16 → S, bias<br/>flags → mode]
        L --> M{mode}
        M -->|0 FP4_AFFINE| N[nibble sign-extend<br/>c in -8..7]
        M -->|1 T158_AFFINE| O[2-bit ternary<br/>00→0 01→+1 10→−1 11→0]
        N --> P[w = S·c + bias → Dst&#91;idx&#93;]
        O --> P
        I -->|no| Q[return]
    end
    E -.descriptor binding 0.-> H
    G -.binding 2.-> H
    P --> R[barrierSource<br/>output visible downstream]
    subgraph Test["MNNTestSuite — SGV2-14"]
        S1[Fixture bytes → temp .mnn.weight sidecar] --> S2[CPU: dequant_sgfp4_container_cpu]
        S1 --> S3[Vulkan: Module via op->externalPath<br/>MNN_FORWARD_VULKAN]
        S2 --> S4[checkVectorByRelativeError]
        S3 --> S4
    end
```

### Recommended Project Structure
```
source/backend/vulkan/buffer/execution/
├── VulkanSGFP4Dequant.hpp        # NEW — VulkanBasicExecution subclass
├── VulkanSGFP4Dequant.cpp        # NEW — creator registration on OpType_SGFP4Dequant
└── glsl/
    └── sgfp4_dequant.comp        # NEW — uniform-layout decode (FP32/FP16 via macro.json)
source/backend/vulkan/buffer/compiler/
├── makeshader.py                 # unchanged — run, don't edit
├── AllShader.cpp                 # REGENERATED + committed
├── VulkanShaderMap.cpp           # REGENERATED + committed
└── glsl/macro.json               # EDIT — add {"sgfp4_dequant.comp": {"useFP16": true}}
source/backend/vulkan/buffer/shaders/
└── AllShader.h                   # REGENERATED + committed
test/op/
└── SGFP4VulkanDequantTest.cpp    # NEW — dual-backend parity (auto-globbed)
```

### Pattern 1: Buffer-Backend Execution Class (verified from `VulkanFP4Dequant.cpp`)
**What:** `VulkanBasicExecution` subclass; pipeline + descriptor set built in constructor; per-dispatch state written in `onEncode`.
**When to use:** Always for buffer-backend compute ops.
**Timing constraint (critical):** `VulkanBasicExecution` has NO `onResize` hook — only `onEncode(inputs, outputs, cmdBuffer)` (called by the `VulkanBasicExecutionDirect::onResize` wrapper, `VulkanBasicExecution.cpp:30-52`). So D-01's host sidecar read + validation must happen either in the **creator's `onCreate`/constructor** (the `MNN::Op*` is available there — `VulkanFP4DequantCreator::onCreate` receives it, and `CPUSGFP4Dequant` shows `mOp` carries `externalPath`) or lazily on first `onEncode`. Recommended: constructor (fail-fast, error return before any pipeline dispatch, matches D-05 "before any dispatch").
```cpp
// Source: source/backend/vulkan/buffer/execution/VulkanFP4Dequant.cpp (verified)
struct FP4DequantConst { uint32_t elementCount; uint32_t srcBytes; };

VulkanFP4Dequant::VulkanFP4Dequant(Backend* bn, bool useFP32Output) : VulkanBasicExecution(bn) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    mConstBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false,
                    sizeof(FP4DequantConst), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,   // binding 0: Src
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,   // binding 1: Dst
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER    // binding 2: Const
    };
    std::string shaderName = mUseFP32Output ? "glsl_fp4_dequant_comp"
                          : vkBn->useFP16()  ? "glsl_fp4_dequant_FP16_comp"
                                             : "glsl_fp4_dequant_comp";
    mDequantPipeline = vkBn->getPipeline(shaderName, types);
    mDescriptorSet.reset(mDequantPipeline->createSet());
}

ErrorCode VulkanFP4Dequant::onEncode(...) {
    // ... map/unmap const buffer, writeBuffer ×3, bind, dispatch:
    mDequantPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(elementCount, 256), 1, 1);
    cmdBuffer->barrierSource(outputBuffer.first->buffer(), outputBuffer.second, outputSize);
    return NO_ERROR;
}

class VulkanFP4DequantCreator : public VulkanBackend::Creator {
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs,
                                            const std::vector<Tensor*>& outputs,
                                            const MNN::Op* op, Backend* backend) const override { ... }
};
static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Dequantize, new VulkanFP4DequantCreator);
    return true;
}();
```
For SGFP4: register on **`OpType_SGFP4Dequant`**, and the creator receives `op` → read `op->main_as_SGFP4DequantParam()->external()->data()` (`{offset, size}`) + `op->externalPath()->str()` exactly like `CPUSGFP4Dequant::onResize` (`CPUSGFP4Dequant.cpp:51-100`, verified).

### Pattern 2: Container Upload as SSBO (verified from `VulkanBuffer.hpp` + backend usage)
**What:** Two equivalent mechanisms for D-01's host→SSBO copy:
```cpp
// (a) direct in constructor — VulkanBuffer ctor copies hostData before returning:
mContainerBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), /*separate=*/false,
                        containerBytes, mContainer.data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
// (b) map/memcpy (59 in-tree map() call sites; e.g., VulkanBackend.cpp:443 ::memcpy(mHostBuffer->map(), src, size)):
::memcpy(mContainerBuffer->map(), mContainer.data(), containerBytes);
mContainerBuffer->unmap();
```
Both are pool-backed HOST_VISIBLE memory; (a) is simpler and keeps `onEncode` free of host copies.

### Pattern 3: Host Pre-Validation (D-05, recommended form)
**What:** Validate once, host-side, before upload/dispatch. Recommended: run the existing decode into a scratch buffer — it is already fully bounds-checked (every read guarded against `containerSize`, `T-01-02` overflow guards, out-count must equal exactly `outElementCount`, verified in `SGFP4DequantUtils.hpp:424-558`):
```cpp
std::vector<float> scratch(outElementCount);
if (!dequant_sgfp4_container_cpu(mContainer.data(), mContainer.size(),
                                 scratch.data(), outElementCount)) {
    return INVALID_VALUE;  // malformed — same error semantics as CPU path, no dispatch
}
```
This reuses the Phase-1/2-tested walk verbatim (zero new validation code to review) at the cost of one CPU decode per session — acceptable for a correctness-focused phase and within the "reuse vs wrap" discretion. (A leaner `validate_only` walker extracting the loop from `dequant_sgfp4_container_cpu` is the alternative if the planner wants to avoid the scratch decode; strictly a code-organization choice.)

### Pattern 4: GLSL Shader Conventions (verified from `fp4_dequant.comp` + `makeshader.py`)
- **Do NOT** write `#version` and **do NOT** define `FLOAT/FLOAT2/FLOAT4` — makeshader.py prepends `FP32_HEADER`/`FP16_HEADER` (`FLOAT` = `float` or `float16_t` with `GL_EXT_shader_explicit_arithmetic_types_float16`).
- FP16 variant is generated ONLY if `glsl/macro.json` contains an entry with `"useFP16": true` (`genShaderFileObjs`, makeshader.py:489-512 — verified) → **`macro.json` edit is mandatory** for D-06.
- Shader keys: file `glsl/sgfp4_dequant.comp` → names `glsl_sgfp4_dequant_comp` (FP32) + `glsl_sgfp4_dequant_FP16_comp` (FP16) via `getFileName()` path normalization (parts from `glsl/` onward, separators→`_`).
- Workgroup: `layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;` + host `UP_DIV(elementCount, 256)` (existing convention; exact size is discretion but 256 matches every neighboring compute shader and the dispatch arithmetic).
- `if (gl_GlobalInvocationID.x >= elementCount) return;` bounds guard at main() top (`fp4_dequant.comp` lines 66-69).

### Anti-Patterns to Avoid
- **Reading container bytes as `uint[]` at unaligned byte offsets:** SSBO `uint Dst[i]` addresses word `i` only; record count `B` sits at byte 5 (straddles words 1 and 2). Must use a byte-composition helper (see Code Examples). Sign of a violation: values like `0x00SGF402` garbage or shader compile/refactor failures on strict drivers.
- **In-shader bounds checking / OOB-output conventions:** locked out by D-05; the host guarantees a valid container, so a pure `if (idx >= outElementCount) return;` suffices.
- **Building framing-index machinery (leaf-offset table, prefix sums):** locked out by D-03/D-04; Phase 4 owns quadtree indexing.
- **Editing generated files by hand:** `AllShader.cpp`/`AllShader.h`/`VulkanShaderMap.cpp` are regenerated outputs (`/*Auto Generated File, Don't Modified.*/`); any manual edit is overwritten and pollutes the diff.
- **Running makeshader.py from Windows PowerShell directly:** `findAllShader()` uses `os.popen("find <path> -name \"*.comp\"")` — Windows resolves `find` to `find.exe` (a completely different tool) → silent empty/incorrect file list. Run under WSL or Git Bash.
- **Redefining push constants:** follow the backend convention — uniform buffer at binding 2 (`VulkanSoftmax`/`VulkanArgMax` pattern), not `push_constant`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FP16 bit→float conversion in GLSL | Manual mantissa/exponent assembly | `unpackHalf2x16(uint)` (core GLSL 4.5) | One instruction; matches `unpack_leaf_header` semantics (S = high half, bias = `h & 0xFFF0` low half) |
| Sidecar file loading + DoS bound | New file I/O / stat code | `FileLoader` + `queryFileSize`-style `std::ifstream` probe (clone from `CPUSGFP4Dequant.cpp:23-49`) | Phase 1 T-01-04 lesson: `FileLoader::size()` is NOT a filesystem stat — stays 0 for offset+size reads; the direct ifstream probe is the validated fix |
| Container structural validation | New validator | `dequant_sgfp4_container_cpu` scratch pass (or minimal refactor of its walk) | Every read already bounds-checked and Phase-1-tested; new validation code = new attack surface |
| SPIR-V embedding | Custom codegen | `makeshader.py` pipeline (locked by CLAUDE.md) | Regenerating 3 output files is the committed contract; hand-embedding breaks the map lookup (`VulkanShaderMap::init`) |
| Vulkan device-availability detection in test | Device enumeration | `MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN)` null check (`VulkanFP4DequantTest.cpp:66-70`) | Established graceful-skip pattern (D-07) |
| 0-input Const-like op module construction | Session API manual wiring | Phase 1 `runSgfp4Module` pattern: `Expr::create(op.get(), {})` → `Variable::save` → `Module::load(buffer,…, rtmgr)` | Verified working for CPU; Vulkan reuses it with `MNN_FORWARD_VULKAN` config; encodes the `op->externalPath`-must-be-set-directly pitfall |

**Key insight:** every non-GLSL component of this phase exists in-tree and is tested; the only genuinely new logic is ~100 lines of shader arithmetic (byte-composition read + framing walk + affine decode). Resist the pull to generalize any of the C++ scaffolding.

## Common Pitfalls

### Pitfall 1: Unaligned container byte accesses in GLSL
**What goes wrong:** The framing has u32 fields at non-4-byte-aligned offsets — `B` at byte 5 (`kSGFP4RecordCountOffset = 5`, straddling header words), and record/leaf fields at `recStart + …` offsets that are 4-aligned only relative to the record (which itself starts at `regionStart + recOffRel`, and offsets are 16-aligned in practice but the format only guarantees the documented alignments). Direct `Container[byteAddr / 4]` indexing is only correct when `byteAddr % 4 == 0`.
**Why it happens:** SSBO `uint[]` indexing is word-granular by definition.
**How to avoid:** Byte-composition helper reading (at most) two words (see Code Examples). Practically, `sb_header`, block headers, split map and payloads land on 4-byte-aligned addresses given the 16-byte record/payload alignment rules — but `B` at byte 5 unconditionally requires the general helper, so use it for every framing read for uniformity.
**Warning signs:** Decode works for `B` read via a lucky aligned path but record offsets come out as garbage; or worse, works on one driver and not another (UB).

### Pitfall 2: FP16 backend output vs. test tolerance (two independent meanings of "FP16")
**What goes wrong:** Two different precision switches exist: (a) the **shader output variant** (`glsl_..._FP16_comp` writes `float16_t` Dst) and (b) the **backend tensor format** (`VulkanBackend.cpp:102`: `mUseFP16 = precision != Precision_High && device FP16 support` — makes ALL float tensors on the backend FP16, and host copy-back converts, lines 471/490/529). A test comparing against a CPU float reference with tight rtol will spuriously fail when the backend stores FP16.
**How to avoid:** Parity test sets `backendConfig.precision = BackendConfig::Precision_High` (the exact `VulkanFP4DequantTest.cpp:73` pattern) to force the FP32 path for the primary assertion (rtol 1e-4 like Phase 1's `kFixtureRelativeTolerance`); optionally a secondary relaxed pass (rtol ≈ 2e-3, FP16 epsilon) with default precision to exercise the FP16 variant. `VulkanFP4DequantTest` derives `rtol = (precision == 3) ? 0.02f : 0.01f` from the suite's `precision` arg for exactly this reason.
**Warning signs:** Failures clustered where |w| is large (relative error from half-ulp rounding) but passes everywhere else.

### Pitfall 3: makeshader.py environment (verified broken right now)
**What goes wrong:** Regeneration silently produces empty/incorrect output if `glslangValidator` or `xxd` are missing (glslang failures print to stdout but the script continues; `xxd` failure leaves empty temp file), or if run under Windows PowerShell (POSIX `find` missing).
**Why it happens:** `glslangValidator` is currently installed NOWHERE on this machine (verified: WSL has `python3` 3.10 + `xxd` but no glslang/spirv-opt; Windows has Python but no Vulkan SDK on PATH; no vendored binary in-repo).
**How to avoid:** Plan a Wave-0 toolchain task: either `sudo apt-get install glslang-tools` in WSL (needs the user's sudo password — interactive step) or install the Vulkan SDK on Windows and run via Git Bash. `spirv-opt` is optional — `_spirv_opt_tag()` degrades gracefully (`MNN_VULKAN_DISABLE_SPIRV_OPT=1` also force-disables). Then `cd source/backend/vulkan/buffer/compiler && python3 makeshader.py`. The `.cache/` dir is absent (not committed), so ALL ~70 shaders recompile — expected slow first run, harmless (deterministic `xxd` output → diffs limited to the new shader).
**Warning signs:** `AllShader.cpp` diff that touches other shaders' byte arrays; or compilation errors referencing missing `glsl_sgfp4_dequant_comp` symbol at link.

### Pitfall 4: Zero-input op scheduling on the Vulkan backend
**What goes wrong:** SGFP4Dequant is Const-like (no input tensors); wiring assumptions from input-consuming ops (like `VulkanFP4Dequant` reading `extra->getTensorBuffer(input)`) don't apply; and, per Phase 1 (STATE.md), `rtmgr->setExternalFile()` alone does NOT populate `op->externalPath` for this op type — `createExecutionWithExternal` only rewrites Convolution2D/Scale/LayerNorm.
**How to avoid:** Mirror Phase 1's `runSgfp4Module` exactly: set `op->externalPath = sidecarPath` on the OpT directly; build output via `Expr::create(op.get(), {})`; only swap `config.type = MNN_FORWARD_VULKAN` (+ Precision_High) in the RuntimeManager. The Execution itself never touches an input tensor; output buffer via `extra->getTensorBuffer(outputs[0])`.
**Warning signs:** `NOT_SUPPORT` returned from the creator (externalPath null at Vulkan-execution creation time).

### Pitfall 5: `FP4ModelTest.cpp` blocks a from-scratch `run_test.out` build
**What goes wrong:** `test/op/FP4ModelTest.cpp` (pre-existing dead code from `milestone` commit `cffaf4bd`) fails to compile under MSVC → the monolithic `run_test.out` cannot build at all (documented in Phase 1 `deferred-items.md`).
**How to avoid:** Same Phase 1 workaround for local verification only: temporarily stub the file, build + run the suite (Phase 1 got 375 passed incl. `op/sgfp4/*`), restore byte-for-byte, never commit the stub. Or land the `milestone` 04-02 fix first (open, non-blocking per STATE.md).
**Warning signs:** `error C2065 'pi'`-family errors during build; unrelated to this phase's changes.

### Pitfall 6: Build configuration — buffer backend not enabled
**What goes wrong:** Current `.build/CMakeCache.txt` has `MNN_VULKAN:BOOL=OFF` (verified) and `MNN_VULKAN_IMAGE` defaults ON — building as-is compiles neither the buffer backend nor the new Execution; with `MNN_VULKAN_IMAGE=ON`, `buffer/*` sources are excluded entirely (CMakeLists line 4).
**How to avoid:** Configure with `-DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF -DMNN_BUILD_TEST=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON` (the exact flags the `milestone` workstream used for its Vulkan builds; `MNN_SUPPORT_TRANSFORMER_FUSE` gates the test files' `#ifdef`, and SGFP4 test code lives behind it). Default `MNN_USE_SYSTEM_LIB=OFF` uses the dlopen wrapper (`vulkan_wrapper.cpp` loads `vulkan-1.dll`/`libvulkan.so.1` at runtime — verified present at `C:\Windows\System32\vulkan-1.dll`).
**Warning signs:** Linker errors on `VulkanBackend` symbols; test silently skips Vulkan (`MNNGetExtraRuntimeCreator` returns null).

## Code Examples

Verified patterns from the tree, plus the new GLSL math derived 1:1 from `SGFP4DequantUtils.hpp` constants (`kSGFP4*`) — every constant below is normative from that header.

### GLSL: unaligned little-endian u32 read (new, derived from format constants)
```glsl
// Container SSBO: binding 0, raw bytes viewed as little-endian u32 words.
layout(binding = 0) readonly buffer ContainerBuffer { uint Container[]; };

// Word loads are native-endian; the container is little-endian. On all
// MNN Vulkan targets (x86/ARM consumer devices) host == little-endian, and
// the host uploads raw bytes, so word i reads back the same u32 the CPU's
// sgfp4_read_u32_le() would read at byte 4*i. Compose arbitrary byte
// addresses from (at most) two aligned words.
uint read_u32_le(uint byteAddr) {
    uint word = byteAddr >> 2u;
    uint off  = byteAddr & 3u;      // 0..3 (off==0 is the common aligned case)
    uint lo   = Container[word];
    if (off == 0u) return lo;
    uint hi = Container[word + 1u]; // neighboring word always exists: framing
                                    // u32 fields never end past the container
    return (lo >> (8u * off)) | (hi << (32u - 8u * off));
}
```
Framing constants to mirror (names named, no magic numbers, per CLAUDE.md):
```glsl
const uint  kMagic              = 0x46464753u; // 'SGF4' little-endian ('S'|'G'<<8|'F'<<16|'4'<<24)
const uint  kVersion            = 2u;
const uint  kFixedHeaderSize    = 16u;
const uint  kVersionByteOffset  = 4u;
const uint  kRecordCountOffset  = 5u;
const uint  kOffsetTableStart   = 16u;
const uint  kAlign16            = 16u;
const uint  kLeafScaleShift     = 16u;
const uint  kLeafBiasMask       = 0xFFF0u;
const uint  kLeafModeBit        = 0x1u;
const uint  kLayoutEnumMask     = 0x7u;
const uint  kNibblesPerWord     = 8u;   // mode 0
const uint  kSymbolsPerWord     = 16u;  // mode 1
```

### GLSL: leaf header unpack + dual-mode affine decode (port of `unpack_leaf_header` / `sgfp4_decode_leaf_payload`)
```glsl
// S = half(h>>16); bias = half(h & 0xFFF0); mode = h & 0x1  (spec Eq. 6)
void unpackLeafHeader(uint h, out float S, out float bias, out uint mode) {
    S    = unpackHalf2x16(h).y;                       // high 16 bits = scale bits
    bias = unpackHalf2x16((h & kLeafBiasMask)).x;     // low half, low 4 bits zeroed
    mode = h & kLeafModeBit;
}

// mode 0: 4-bit two's complement, c in [-8,7] — (nib ^ 0x8) - 0x8 sign-extend:
float codeMode0(uint nib) { return float((nib ^ 0x8u) - 0x8u); }
// mode 1: ternary 00->0, 01->+1, 10->-1, 11->0 (reserved):
float codeMode1(uint sym) { return (sym == 1u) ? 1.0 : (sym == 2u) ? -1.0 : 0.0; }

// Per-thread framing re-walk (D-04): find the record/leaf holding global
// output element `idx`; mirrors dequant_sgfp4_container_cpu's sequential
// record→leaf walk, aborting early once the leaf is found. Uniform layouts
// only (Phase 4 adds the mixed walk). Returns payload byte address of the
// element's word + intra-word index; out S/bias fill from the leaf header.
bool locateElement(uint idx, out uint payloadWordByte, out uint intra,
                   out float S, out float bias, out uint mode) {
    uint B = read_u32_le(kRecordCountOffset);          // unaligned — must compose
    uint offsetTableEnd = kOffsetTableStart + 4u * B;
    uint regionStart = (offsetTableEnd + kAlign16 - 1u) & ~(kAlign16 - 1u);
    uint outBase = 0u;
    for (uint b = 0u; b < B; ++b) {
        uint recStart = regionStart + read_u32_le(kOffsetTableStart + 4u * b);
        uint sbHeader = read_u32_le(recStart);
        uint layout   = sbHeader & kLayoutEnumMask;
        // resolve_uniform_layout (Table 3): enum -> {leafCount N, leafEdge n}
        uint N; uint n;
        if      (layout == 0u) { N = 1u;   n = 64u; }
        else if (layout == 1u) { N = 4u;   n = 32u; }
        else if (layout == 2u) { N = 16u;  n = 16u; }
        else if (layout == 3u) { N = 64u;  n = 8u;  }
        else if (layout == 5u) { N = 256u; n = 4u;  }
        else return false;                              // MIXED (4) / invalid — host
                                                        // pre-validation rejects these anyway
        uint recElements = N * n * n;
        if (idx < outBase + recElements) {
            uint local = idx - outBase;
            uint leaf  = local / (n * n);
            uint inLeaf = local - leaf * n * n;
            uint headersStart = recStart + 4u;
            uint payloadsStart = (headersStart + 4u * N + kAlign16 - 1u) & ~(kAlign16 - 1u);
            uint h;
            {
                // sequential payload cursor walk up to this leaf (D-04 redundant):
                // per-leaf payload bytes = align16(n*n/8 or n*n/16 * 4 per its mode)
                uint cursor = payloadsStart;
                for (uint l = 0u; l < leaf; ++l) {
                    uint lh = read_u32_le(headersStart + 4u * l);
                    uint words = ((lh & kLeafModeBit) == 0u) ? (n * n / kNibblesPerWord)
                                                             : (n * n / kSymbolsPerWord);
                    cursor += ((4u * words) + kAlign16 - 1u) & ~(kAlign16 - 1u);
                }
                h = read_u32_le(headersStart + 4u * leaf);
                payloadWordByte = cursor;
            }
            unpackLeafHeader(h, S, bias, mode);
            if (mode == 0u) { payloadWordByte += 4u * (inLeaf / kNibblesPerWord); intra = inLeaf % kNibblesPerWord; }
            else            { payloadWordByte += 4u * (inLeaf / kSymbolsPerWord); intra = inLeaf % kSymbolsPerWord; }
            return true;
        }
        outBase += recElements;
    }
    return false;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outElementCount) return;   // only guard needed — host pre-validated (D-05)
    uint wordByte; uint intra; float S; float bias; uint mode;
    if (!locateElement(idx, wordByte, intra, S, bias, mode)) return; // unreachable on validated input
    uint w = read_u32_le(wordByte);
    float c = (mode == 0u) ? codeMode0((w >> (4u * intra)) & 0xFu)
                           : codeMode1((w >> (2u * intra)) & 0x3u);
    Dst[idx] = FLOAT(S) * FLOAT(c) + FLOAT(bias);   // w = S·c + bias (FMA)
}
```
(The exact helper decomposition is discretion D-3; `makeshader.py` accepts any single-file structure as long as `FLOAT`/`#version` are untouched.)

### Parity test skeleton (blend of `VulkanFP4DequantTest.cpp:61-80` + Phase 1 `runSgfp4Module`)
```cpp
// Source patterns: test/op/VulkanFP4DequantTest.cpp (verified) + test/op/SGFP4DequantTest.cpp::runSgfp4Module (verified)
virtual bool run(int precision) override {
    auto vulkanCreator = MNN::MNNGetExtraRuntimeCreator(MNN_FORWARD_VULKAN);
    if (nullptr == vulkanCreator) {
        MNN_PRINT("Vulkan backend not available — skipping SGFP4 Vulkan parity test\n");
        return true;                                    // D-07 graceful skip
    }
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_VULKAN;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High; // -> FP32 tensors + FP32 shader variant
    config.backendConfig = &backendConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(
        Executor::RuntimeManager::createRuntimeManager(config));

    for (size_t i = 0; i < sgfp4_fixtures::kFixtureCount; ++i) {
        const auto& fx = sgfp4_fixtures::kFixtures[i];
        if (fx.layout == /* skip kFixture_mixed_* / uniform_collapse not in uniform set as needed */)
            continue; // Phase 3 scope: uniform layouts only (mixed_* fixtures are Phase 4/parity-with-CPU anyway — planner picks set)
        // 1) write fixture bytes to temp sidecar (Phase 1 testOpLevelExternalSidecar pattern)
        // 2) CPU reference: dequant_sgfp4_container_cpu(fixture.container, ..., expectedCount)
        // 3) Vulkan module: OpT with externalPath set DIRECTLY, dims from fixture -> Module::load(buffer,...,rtmgr)
        // 4) outputs[0]->readMap<float>() vs fixture->expected:
        if (!checkVectorByRelativeError<float>(gpuOut, cpuRef, count, 1e-4f)) return false;
    }
    return true;
}
```
Registration (naming discretion): `REGISTER_TEST(SGFP4VulkanDequant, op/sgfp4/vulkan_uniform_parity)` — following the `op/sgfp4/...` namespace from Phase 1.

### Toolchain commands
```bash
# 1) WSL toolchain (one-time; sudo password required — interactive):
sudo apt-get install glslang-tools            # spirv-tools optional; auto-skipped if absent
# 2) Regenerate shaders (POSIX env; .cache absent -> full recompile, expected):
cd source/backend/vulkan/buffer/compiler && python3 makeshader.py
# 3) Verify embedding (milestone 02-01 verification pattern):
grep -c 'sgfp4_dequant' AllShader.cpp        # >= 4 (name + _len, FP32 + FP16)
grep -c 'sgfp4_dequant' ../shaders/AllShader.h  # 4 extern declarations
grep -c 'sgfp4_dequant' VulkanShaderMap.cpp     # 2 map insertions
# 4) Reconfigure build + run:
cmake -S . -B .build -DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF -DMNN_BUILD_TEST=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
cmake --build .build --target run_test.out   # (mind the FP4ModelTest.cpp stub workaround, never committed)
./.build/run_test.out.exe "op/sgfp4/"
```

## Runtime State Inventory

Not applicable — this is a feature-add phase, not a rename/refactor/migration phase. (No stored data, service config, OS-registered state, secrets, or stale build artifacts carry renamed identifiers. The `.build/` CMakeCache reconfiguration in Pitfall 6 is an ordinary build-config change, not runtime state migration.)

## Common Pitfalls (checklist form for plan verification)

1. Unaligned GLSL byte reads → `read_u32_le` helper used for ALL framing reads (esp. `B` at byte 5).
2. FP16 backend tensor format vs shader variant → test uses `Precision_High` for the tight-tolerance pass.
3. makeshader.py env → Wave-0 toolchain task (glslang install) + run under WSL/Git Bash; verify grep counts.
4. Zero-input op module → `op->externalPath` set directly on OpT; no input-tensor wiring in the Execution.
5. `FP4ModelTest.cpp` build blocker → temp-stub verify pattern (never committed) or milestone 04-02 first.
6. Build config → `-DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF`; dlopen wrapper finds `vulkan-1.dll`.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| (milestone) E2M1 `VulkanFP4Dequant`: in-graph packed-nibble input tensor, uniform 1-u32 const | (this phase) SGFP4 v2: external-sidecar container SSBO, framing walk in shader, affine dual-mode | Phase 3 (now) | New additive op; E2M1 path untouched (locked REQUIREMENTS "Additive, not a replacement") |
| Shader FP16 variant hand-written | `macro.json` `useFP16: true` auto-generates the FP16 variant via header prepend | Existing makeshader.py design | One `.comp` file yields both pipeline keys |
| CPU-only SGFP4 decode (Phases 1-2) | GPU parity (this phase); quadtree GPU (Phase 4) | — | Shader struct choice (D-03/D-04 stateless, recursive-walk-compatible per Phase 2's recursion-free `sgfp4_walk_quadtree`) must not preclude Phase 4 |

**Deprecated/outdated:** none in-tree relevant here.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | GLSL storage `uint` loads of host-uploaded bytes decode as little-endian u32s on all target devices (host LE ⇔ GPU LE for the SSBO byte-copy path) | Code Examples / Pitfall 1 | Medium — no target MNN deploys to is big-endian GPU; if ever, `read_u32_le` is the single place to add a byteswap. Validated-by-construction since bytes are copied verbatim and word endianness matches host memory layout exposed via the buffer. |
| A2 | Per-thread redundant re-walk cost is negligible for uniform layouts (D-04 rationale; a few hundred cached uint reads per thread ≤ 256 leaves × few words) | D-03/D-04 discussion | Low — locked decision anyway; correctness phase, perf is SGV2-18 backlog. |
| A3 | One CPU scratch-decode pass per resize for D-05 validation is acceptable overhead for weight-decode ops (once per session, not per token) | Pattern 3 | Low — weight tensors decode once; if the planner prefers, a validate-only walker is a drop-in alternative (same discretion bucket). |
| A4 | Prior milestone Vulkan builds/tests were produced in an environment that has since changed (no glslang now, `.build` at VULKAN=OFF) — regeneration must be re-provisioned this phase | Environment Availability | Medium — if a working toolchain location exists that I failed to find, the Wave-0 install task is simply redundant, not harmful. |
| A5 | Windows `Precision_High` Vulkan session drives NVIDIA RTX 4070 Ti SUPER through `vulkan-1.dll` via the dlopen wrapper successfully (driver present, verified;удш runtime exercise not yet executed this session) | Pitfalls 2/6 | Low-Medium — planned Wave-0/Wave-1 verification step covers it; graceful-skip guard means a failure degrades to "test skipped" not "suite fails" (but Phase success criterion requires an actual run — see Open Questions). |

## Open Questions

1. **Where is (or: how do we provision) a working `glslangValidator`?**
   - What we know: not on Windows PATH, not in WSL, not vendored in-repo (all probed). WSL has python3+xxd; WSL sudo requires a password (interactive upload step).
   - Recommendation: Wave-0 task with two options — (a) user runs `sudo apt-get install glslang-tools` in WSL (one interactive command), or (b) Vulkan SDK install on Windows + run makeshader.py from Git Bash. Non-blocking for coding (shader authoring/execution-class/test all proceed; only the regeneration+commit task gates on it).
2. **Was the previous `run_test.out` Vulkan execution performed on this machine or another (CI/Linux)?**
   - What we know: `.build/` currently has MNN_VULKAN=OFF; `run_test.out.exe` exists; no Vulkan test run logs found for the milestone phase on this box; WSL has libvulkan + mesa ICDs but no NVIDIA WSL ICD visible in `/usr/share/vulkan/icd.d` (lvp/d3d12 fallback uncertain, `vulkaninfo` absent).
   - Recommendation: Window: reconfigure `.build` per Pitfall 6 and smoke-test `VulkanFP4DequantTest` first (it already exists) to establish the Vulkan-runtime path on Windows before building the new test. If the Windows path fails at runtime, fall back to WSL Lavapipe (lvp ICD present) for shader-toolchain + parity execution — needs verification.
3. **Fixture set for the parity test — include the Phase 2 mixed/uniform-collapse fixtures?**
   - What we know: `kFixtures` contains mixed + uniform-collapse entries alongside the 5 uniform × 2 mode + b3 alignment fixtures; the new shader (uniform-only) will reject mixed layouts, which host pre-validation (D-05) turns into a clean error — not a parity target this phase.
   - Recommendation: parity-loop over uniform fixtures only (filter `fixture.layout != kSGFP4LayoutMixed` and skip `uniform_collapse` if it emits MIXED — its enum is uniform per Phase 2 uniform-collapse rule, so it likely qualifies; verify at implementation). Discretion-level; keep the filter explicit and named.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| NVIDIA GPU + Vulkan driver (RTX 4070 Ti SUPER, `C:\Windows\System32\vulkan-1.dll`) | Parity test runtime (D-07) | ✓ (driver present, verified via `Get-CimInstance`) | Driver 32.0.15.9186 | WSL Lavapipe (`vulkan_lvp.so` present; untested for MNN) |
| `python3` (WSL) | makeshader.py | ✓ | 3.10.12 | Windows Python 3.13 (but see POSIX-`find` constraint) |
| `xxd` (WSL) | makeshader.py embedding | ✓ | /usr/bin/xxd | Git Bash xxd |
| `glslangValidator` | makeshader.py compilation | **✗** | — | **Install needed** (WSL `glslang-tools` w/ sudo password, or Windows Vulkan SDK + Git Bash) |
| `spirv-opt` | makeshader.py optimization (optional) | ✗ | — | Auto-skipped by `_spirv_opt_tag()`; also `MNN_VULKAN_DISABLE_SPIRV_OPT=1` |
| CMake + MSVC 17 2022 (`.build`) | Build | ✓ | VS 17 2022, existing `.build` | — |
| `.build` configured for buffer Vulkan backend | Compilation of new Execution | **✗** (`MNN_VULKAN=OFF`, `MNN_VULKAN_IMAGE` unset→ON) | — | Reconfigure (Pitfall 6 command; no data migration needed) |
| `dequant_sgfp4_container_cpu` / fixtures / `OpType_SGFP4Dequant` schema+shape | Everything | ✓ (Phase 1/2, committed & tested) | — | — |

**Missing dependencies with no fallback:**
- `glslangValidator` — blocks the makeshader.py regeneration task (SGV2-12 success criterion 1). Install is one interactive sudo command / SDK install; plan it as Wave-0 `checkpoint:human-verify`.

**Missing dependencies with fallback:**
- `spirv-opt` — optional optimization; script degrades automatically.
- Windows Vulkan runtime for tests — driver present; WSL Lavapipe is the untested fallback if a Windows runtime issue surfaces.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | MNNTestSuite (in-tree; static self-registration via `REGISTER_TEST`) |
| Config file | `test/CMakeLists.txt` — `GLOB_RECURSE` auto-discovers new test files (line 12); no edit needed |
| Quick run command | `./run_test.out "op/sgfp4/"` (Windows: `run_test.out.exe`) |
| Full suite command | `./run_test.out` (note Pitfall 5: full build needs the FP4ModelTest temp-stub workaround) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SGV2-12 | Shader decodes both modes × all uniform layouts; embedded via makeshader.py | integration (GPU) + build-integrity grep | `./run_test.out "op/sgfp4/"` + `grep -c sgfp4_dequant AllShader.cpp ../shaders/AllShader.h VulkanShaderMap.cpp` | ❌ Wave 0 (`test/op/SGFP4VulkanDequantTest.cpp`) |
| SGV2-13 | Execution loads same sidecar descriptor as CPU (real file, real externalPath) | integration (GPU, module-level) | covered by the same parity test (module built from OpT + sidecar file, D-02) | ❌ Wave 0 (same file) |
| SGV2-14 | GPU output == CPU reference within float tolerance | integration (dual-backend) | `./run_test.out "op/sgfp4/vulkan_uniform_parity"` (naming discretion) | ❌ Wave 0 (same file) |

### Sampling Rate
- **Per task commit:** `cmake --build .build --target run_test.out && ./.build/run_test.out.exe "op/sgfp4/"` (with FP4ModelTest stub workaround as needed)
- **Per wave merge:** same, plus `./run_test.out "op/fp4"` (E2M1 regression guard — additive-not-replacement contract) and a `grep -c` of the three autogenerated files
- **Phase gate:** full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `test/op/SGFP4VulkanDequantTest.cpp` — covers SGV2-12/13/14 (clone `VulkanFP4DequantTest` harness + `runSgfp4Module` module pattern)
- [ ] Toolchain install (glslang) — `sudo apt-get install glslang-tools` (WSL) or Vulkan SDK (Windows) — checkpoint:human-verify
- [ ] `.build` reconfiguration (`-DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=OFF`) + pre-existing `VulkanFP4DequantTest` smoke run to validate the machine's Vulkan execution path before authoring

*(No shared fixtures needed — `SGFP4DequantFixtures.h` already provides everything.)*

## Security Domain

`security_enforcement` enabled (absent in `.planning/config.json` → default). Phase 1's threat model carries forward; this phase adds one trust boundary crossing (**untrusted sidecar bytes now flow onto the GPU**).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface (offline inference op) |
| V3 Session Management | no | N/A |
| V4 Access Control | no | N/A |
| V5 Input Validation | **yes** | Untrusted-container posture: (a) real-file-size probe BEFORE allocation (T-01-04 pattern, `queryFileSize` clone — an attacker-controlled `external()[1]` must not force oversized `VulkanBuffer` allocations); (b) host pre-validation ONCE via the fully-bounds-checked `dequant_sgfp4_container_cpu` walk (D-05) — record-offset table bounded before trusting `B` (T-01-02 overflow guards already in the Phase-1 code), out-count must equal exactly `outElementCount` (bounds per-record work); only validated bytes reach the SSBO; shader performs no partial writes on malformed input (never dispatched) |
| V6 Cryptography | no | No secrets, no crypto |

### Known Threat Patterns for Vulkan/GLSL decode ops

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| DoS via declared-size vs. real-file mismatch (huge allocation) | DoS | ifstream size probe before `mContainer`/`VulkanBuffer` allocation (Phase 1 T-01-04, replication mandatory) |
| DoS via attacker-controlled record count B / offsets | DoS | Offset-table bounds guard + exact out-count equality in the reused validator (already in `dequant_sgfp4_container_cpu`) |
| OOB shader reads/writes from malformed geometry | Tampering/DoS | Host pre-validation (D-05) — shader reads only a validated container; `idx >= outElementCount` early-return is the sole in-shader guard; no defined-OOB-output convention needed |
| Shader injection | Tampering | Not applicable — GLSL is developer-authored and compiled offline; SPIR-V embedded at build time (attacker controls only data buffers, never code) |
| Unvalidated path handling (`externalPath`) | Info disclosure | Path comes from the signed model file via FlatBuffers, same as Convolution2D external weights (locked design); no new surface |

## Sources

### Primary (HIGH confidence — direct code/file reads this session)
- `include/MNN/SGFP4DequantUtils.hpp` — full container format constants, decode walk, all `kSGFP4*` values (the porting reference)
- `source/backend/vulkan/buffer/execution/VulkanFP4Dequant.{hpp,cpp}` — Execution class template, pipeline selection, creator registration, dispatch/barrier
- `source/backend/vulkan/buffer/execution/glsl/fp4_dequant.comp` — GLSL conventions (FLOAT macro, local_size, bounds guard)
- `source/backend/vulkan/buffer/compiler/makeshader.py` — FP32/FP16 headers, `macro.json` `useFP16` variant generation, naming (`getFileName`), POSIX `find`/`glslangValidator`/`xxd` dependencies, spirv-opt degradation, cache design
- `source/backend/vulkan/buffer/execution/glsl/macro.json` — variant config format
- `source/backend/vulkan/buffer/execution/VulkanBasicExecution.{hpp,cpp}` — no-onResize constraint; Direct wrapper encodes during resize
- `source/backend/vulkan/component/VulkanBuffer.hpp` — ctor hostData upload + map()/unmap()
- `source/backend/vulkan/buffer/backend/VulkanBackend.cpp` — `useFP16()` (Precision_High gate, line 102), `getPipeline` (152), FP16 copy-back conversion (471-529)
- `source/backend/vulkan/CMakeLists.txt` + root `CMakeLists.txt` + `test/CMakeLists.txt` — GLOB_RECURSE auto-discovery (no CMake edits), `MNN_VULKAN_IMAGE` buffer/image switch
- `source/backend/cpu/CPUSGFP4Dequant.cpp` — FileLoader + `queryFileSize` probe + USE_EXTERNAL_DATA/externalPath gate (host-side template)
- `test/op/VulkanFP4DequantTest.cpp` — availability guard, `MNN_FORWARD_VULKAN` config, `Precision_High`, tolerance-vs-precision handling
- `test/op/SGFP4DequantTest.cpp` + `test/op/SGFP4DequantFixtures.h` — `runSgfp4Module` 0-input module pattern, fixture inventory
- `test/op/FP4ModelTest.cpp` blocker: `.planning/workstreams/sgfp4-pivot/phases/01-.../deferred-items.md`
- Workstream docs: `03-CONTEXT.md`, `REQUIREMENTS.md`, `STATE.md`, `ROADMAP.md` (locked notes 1-6)

### Secondary (MEDIUM confidence — verified live environment probes this session)
- Windows: `vulkan-1.dll` present; RTX 4070 Ti SUPER driver 32.0.15.9186; Python 3.12/3.13; no glslang on PATH; `.build` cache `MNN_VULKAN=OFF`
- WSL: python3 3.10.12, xxd, libvulkan 1.3.204 + mesa ICDs (no NVIDIA ICD in icd.d); glslangValidator/spirv-opt absent (all standard locations probed); sudo password-required
- Git history: `73e155ea` (VulkanFP4Dequant), `76b31a33` (its test), milestone phase plans confirm prior `-DMNN_VULKAN=ON` builds

### Tertiary (LOW confidence)
- None used for any normative claim.

## Metadata

**Confidence breakdown:**
- Architecture/patterns: HIGH — everything is cloned from verified in-tree code; no speculative APIs
- GLSL math: HIGH — direct 1:1 port of Phase-1-tested constants/semantics; only the `read_u32_le` helper is new derivation, from the verified format layout
- Toolchain/environment: MEDIUM — verified as-of-today state, but provisioning path (A4/A5) needs one interactive step and a smoke run
- Pitfalls: HIGH — all six traced to concrete code lines or live probes

**Research date:** 2026-08-24
**Valid until:** 2026-09-24 (stable internal codebase; environment facts valid until the machine state changes)
