# Phase 12: End-to-End Validation - Pattern Map

**Mapped:** 2026-09-01
**Files analyzed:** 6 (1 new script, 4 modified C++/docs, 1 unmodified driver documented by contract)
**Analogs found:** 6 / 6

> Read-only constraint honored: this is the only file written. All excerpts below were verified by direct reads this session.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `tools/fp4/<e2e-script>.ps1` (new, name at planner discretion — suggest `e2e_validation.ps1`) | test / orchestration script | batch (multi-leg process orchestration) | `tools/fp4/w2_failcleanup_probe.ps1` | exact (repo's committed-PS-validation precedent) |
| `tools/converter/source/optimizer/PostConverter.cpp` (modify) | service (transform pipeline) | transform | itself: `optimizeNetImpl` nullptr return `:279`; `RunNetPass` `:144-170`; SGFP4 batch call `:393` | exact (in-file precedent) |
| `tools/converter/source/common/cli.cpp` (modify) | controller (CLI) | request-response | itself: `convertModel` error paths `:690-810` (MNN_ERROR + `return false` precedents) | exact (in-file precedent) |
| `tools/converter/source/MNNConverter.cpp` (modify) | controller (entry point) | request-response | itself: Phase 11 OQ1 `return 1` block `:15-20` | exact (in-file precedent) |
| `tools/converter/include/PostConverter.hpp` (conditionally modify — only if `RunNetPass` signature changes) | interface header | — | itself: existing `RunNetPass` declaration `:27-30` | exact |
| `tools/fp4/README.md` (modify) | config / documentation | — | itself: "mnnconvert --sgfp4 smoke (Phase 11)" section | exact |
| `tools/cpp/MNNV2Basic.cpp` (**NO modification** — D-09 driver, consumed as-shipped) | service (session executor) | file-I/O | n/a — invocation contract documented below | n/a |

## Pattern Assignments

### `tools/fp4/<e2e-script>.ps1` (test/orchestration script, batch)

**Analog:** `tools/fp4/w2_failcleanup_probe.ps1` (entire file, 56 lines — the Phase 11 W-2 committed-script precedent)

Copy this skeleton verbatim and extend to multi-leg orchestration. Key excerpts:

**Header comment + param + fail-fast pre-checks** (lines 1-23):
```powershell
# W-2 stale-artifact probe (Phase 11, Plan 11-02, D-10): proves that a
# usage()-exit (bad argument) ALSO removes stale output artifacts ...
# Usage (from the MNN repo root, after building sgfp4_inject.out):
#   pwsh tools/fp4/w2_failcleanup_probe.ps1 [-Exe <path-to-sgfp4_inject.out.exe>]
# Exit 0 = probe PASSED ...
param(
    [string]$Exe = ".build/Release/sgfp4_inject.out.exe"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Exe)) {
    Write-Host "FAIL: sgfp4_inject executable not found at '$Exe' (pass -Exe)"
    exit 2
}
```
For Phase 12: `param([string]$Corpus, [string]$MnnConvert = ".build/Release/MNNConvert.exe", [string]$Driver = ".build/Release/MNNV2Basic.out.exe")`; pre-check all three paths (`Test-Path`) **plus** the step-0 `vulkaninfo --summary` D-07 fail-fast (no Vulkan device ⇒ `exit 2`, never SKIP).

**Temp-dir setup + cleanup discipline** (lines 25-26, 45):
```powershell
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("sgfp4_w2_probe_" + [System.Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $tmp | Out-Null
...
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
```
For Phase 12: **one temp dir per leg** (`tmp\p12_e2e\<leg>\` for baseline-cpu / sgfp4-cpu / sgfp4-vulkan) — Pitfall 3: `input_0.txt`, `output.txt`, `.order`, `.tempcache` are all CWD-relative and collide otherwise.

**Process execution + exit-code assertion** (lines 31-35):
```powershell
$proc = Start-Process -FilePath $Exe `
    -ArgumentList("--output", $outMnn, "--bogus-flag") `
    -NoNewWindow -PassThru -Wait
$code = $proc.ExitCode
```
For the run legs, add `-WorkingDirectory <per-leg temp dir>` (Pattern 1 of RESEARCH — CWD isolation). For the conversions, the `& MNNConvert.exe ... 2>&1` capture form from `tools/fp4/README.md`'s Phase 11 smoke is the precedent for stdout assertion (see below).

**PASS/FAIL verdict + exit semantics** (lines 47-55):
```powershell
Write-Host ("exit code          : " + $code)
...
if ($code -eq 1 -and $mnnGone -and $weightGone) {
    Write-Host "W-2 probe: PASS (arg-stage failure removed both stale artifacts)"
    exit 0
} else {
    Write-Host "W-2 probe: FAIL"
    exit 1
}
```
Phase 12 verdict shape: per-backend PASS/FAIL lines (D-10 diagnostics: max-abs, max-rel, failing index), overall `exit 0` only if CPU leg AND Vulkan leg AND the D-11 negative-path leg all pass.

**Conversion + node-presence assertion** — analog: `tools/fp4/README.md` Phase 11 smoke, steps 1 and 2 (lines ~91-110):
```powershell
.build\Release\MNNConvert.exe -f ONNX `
    --modelFile W:\gnus\models\alexnet_Opset16.onnx `
    --MNNModel tmp\p11_smoke.mnn --sgfp4 --dumpPass
# expect: "[DumpPass] PostConvert::InsertSGFP4Dequant: ops 74 -> 82"
```
Script-side: capture `2>&1`, assert `$LASTEXITCODE -eq 0` and `-match 'InsertSGFP4Dequant: ops (\d+) -> (\d+)'` with `$Matches[1] -ne $Matches[2]` (pass rewrote ops) — the cheap SGFP4-node-presence sanity at planner discretion.

**Driver invocation contract** — analog: `tools/cpp/MNNV2Basic.cpp` (read-only; do NOT edit — Pitfall 2/A3):
- Argv order (line ~193 usage + parse block :196-235): `model runLoops runMask forwardType numberThread precisionMask inputDims [cpuIds]`
- Forward type: `argv[4]` — `0` = CPU, `7` = `MNN_FORWARD_VULKAN` (`include/MNN/MNNForwardType.h`)
- **Precision mask `argv[6]` MUST be `1`** (`Precision_High`): `MNNV2Basic.cpp:209-215` defaults `Precision_Low` when `argc <= 6`, and `VulkanBackend.cpp:102` enables FP16 storage whenever precision ≠ High on FP16-capable devices — Pitfall 1 (FP16 error source poisons the single-tolerance D-06 gate)
- Input dims `argv[7]` = `1x3x224x224` — Pitfall 6 (AlexNet has no static shape; resize is mandatory; the tool checks `RESIZE_STATUS` and errors cleanly at :313-320)
- Input: `_loadInputFromFile(inputTensor, pwd, "input_0.txt")` (:97-130, called at :343) — element-wise `input >> inputData[i]` for float tensors; any whitespace separation works; 150,528 floats for 1×3×224×224
- Output: `output.txt` at `pwd + "output.txt"` (:422-431) via `dumpTensor2File` — tab-separated, ~6 significant digits (`std::ofstream <<` default; Pitfall 2: document the 1e-5-ish floor, do not edit the driver)
- stdout assert: `MNN_PRINT("Session Info: ... backendType is %d\n", ..., backendType[0])` (:328) — the Vulkan leg must assert `backendType is 7` (Pitfall 7)
- Hard-fail-not-fallback: `config.backupType = type` with comment "If type not fount, let it failed" (:255-257) — keep, never weaken (D-07)
- `.tempcache` + `.order` also land in CWD — per-leg dirs handle both
- `pwd` stays `"./"` on Windows (`rfind("/")` never matches backslash, :196-199) — another reason to use `-WorkingDirectory`, not path-prefixing

**Synthetic input generation** (D-03) — analog: `test/op/SGFP4TestUtil.hpp` fixture style (seeded `std::rand`, `tempPath` uses `std::time/std::rand` at :59-63). PowerShell equivalent: documented fixed seed (`[Random]::new(<seed>)`), uniform `[-1, 1)` range (activations scale, both signs), written once and byte-identically fed to all three legs (D-04). Determinism note A4: cheap repeat-run consistency check optional.

**Comparison core** (D-10) — no file analog (script-side new logic); use RESEARCH's verified form:
```powershell
# per index i over the parsed output.txt rows:
#   absErr_i = [math]::Abs($sgfp4[$i] - $base[$i])
#   relErr_i = absErr_i / [math]::Max([math]::Abs($base[$i]), $eps)   # denominator guard, Phase 10 D-07
# Gate: (max absErr -le $TolAbs) -and (max relErr -le $TolRel)
# Fail: print per-backend max-abs, max-rel, and the argmax index for both metrics
```
Tolerance derivation (D-02/Pitfall 9): form + sanity bounds from `tools/fp4/real_weight_validation_report.json` (`context.thresholds."64"` = `max_mse 0.01`, `max_relative 0.384` — weight-level, NOT directly transcribable); lock at plan/execute time as measured-worst × documented margin from actual baseline-vs-sgfp4 runs on the locked corpus.

---

### `tools/converter/source/optimizer/PostConverter.cpp` (service/transform — D-11 core)

**Analog:** the same file's own error-propagation and config-gating patterns.

**The D-11 target — log-only failure** (`RunNetPass`, lines 144-170; excerpt 158-170):
```cpp
void RunNetPass(const std::vector<std::string>& passes, std::unique_ptr<MNN::NetT>& originNet) {
    auto config = Global<modelConfig>::Get();
    bool dumpPass = config != nullptr && config->dumpPass;
    for (auto pass : passes) {
        auto convert = PostConverter::get(pass);
        if (nullptr == convert) {
            LOG(INFO) << "Can't find pass of " << pass << "\n";
            continue;
        }
        ...
        bool valid = convert->onExecute(originNet);
        ...
        if (!valid) {
            LOG(INFO) << "Run " << pass << "Error\n";   // <- log-only today (D-11 target)
        }
    }
}
```

**The SGFP4 call site to escalate** (line 393, inside `optimizeNetImpl`):
```cpp
    RunNetPass({"InsertSGFP4Dequant", "ReIndexTensor"}, newNet);
    RunNetPass({"ReIndexOnnxIfAlias"}, newNet);

    return std::move(newNet);
```

**The in-file nullptr-failure precedent to replicate** (`optimizeNetImpl`, lines 277-280):
```cpp
    if (originNet->oplists.size() <= 0) {
        return nullptr;
    }
```
D-12-safe shape: aggregate failure ONLY when `config->useSGFP4` (read `Global<modelConfig>::Get()` exactly as `RunNetPass` already does at :145), thread it out of `optimizeNetImpl` as `return nullptr` + `MNN_ERROR(...)` — the established failure convention of this function.

**Config-gating precedent** (also mirrors `InsertSGFP4Dequant.cpp:294`):
```cpp
if (nullptr == config || !config->useSGFP4) { ... }
```

**Header note (A5):** `RunNetPass` IS declared in `tools/converter/include/PostConverter.hpp:27-30` (added Phase 11 for `TestSGFP4Converter`). If the signature changes (e.g., returns bool), update the declaration there AND check `tools/converter/source/TestSGFP4Converter.cpp` call sites (`RunNetPass({"InsertSGFP4Dequant"}, net)` pattern, :284+). Alternative avoiding the header touch: keep `void` signature, communicate failure via `originNet = nullptr` or an out-param — planner picks; the `TestSGFP4Converter` regression suite must stay green either way (D-12 gate).

**`optimizeNet` wrapper context** (line 631-634 — where `Global<modelConfig>` is already Reset, i.e., flag state is reliably available downstream):
```cpp
std::unique_ptr<MNN::NetT> optimizeNet(std::unique_ptr<MNN::NetT>& originNet, bool forTraining, modelConfig& config, const std::vector<std::string>& expectPasses) {
    BackendConfig bnConfig;
    auto exe = ExecutorScope::Current();
    Global<modelConfig>::Reset(&config);
```

---

### `tools/converter/source/common/cli.cpp` (controller — null-guard + honest return)

**Analog:** the same file's `convertModel` error paths.

**The unguarded dereference to fix** (lines ~785-790, inside `convertModel`'s `needOptimize` branch):
```cpp
    if (needOptimize) {
        std::cout << "Start to Optimize the MNN Net..." << std::endl;
        std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT, modelPath.forTraining, modelPath, expectedPass);
        if (newNet->extraTensorDescribe.size()>0 && expectedPass.empty()) {
```
Add the null-guard before the dereference: `if (newNet == nullptr) { MNN_ERROR("..."); return false; }` — otherwise the D-11 `optimizeNetImpl` nullptr is a crash, not an error (Pitfall 5c).

**The in-file MNN_ERROR + return false precedent to copy** (parse-failure path, lines ~752-755):
```cpp
    if (netT.get() == nullptr || parseRes) {
        MNN_ERROR("[ERROR] Convert error, please check your file format.\n");
        return false;
    }
```
Also the mnn2json/json2mnn branches (lines ~706-721) use the identical `MNN_ERROR(...); return false;` shape.

**The "Converted Success!" lie to prevent** (lines ~798-806):
```cpp
    if (0 == error) {
        std::cout << "Converted Success!" << std::endl;
    } else {
        std::cout << "Converted Failed!" << std::endl;
    }
    ...
    return true;
```
Note `convertModel` returns `true` even on `writeFb` failure today — the D-11 chain requires the SGFP4-failure path to `return false` BEFORE `writeFb`/success print. Scope per D-12 (OQ1 of RESEARCH): strictest reading gates the new failure propagation on `modelPath.useSGFP4`; flag-off behavior byte-identical.

---

### `tools/converter/source/MNNConverter.cpp` (controller/entry — honest exit code)

**Analog:** the same file's Phase 11 OQ1 fix — the exact comment-plus-code pattern to extend.

Current file in full (24 lines; relevant part, lines 11-24):
```cpp
int main(int argc, char *argv[]) {
    modelConfig modelPath;

    // parser command line arg
    auto res = MNN::Cli::initializeMNNConvertArgs(modelPath, argc, argv);
    if (!res) {
        // OQ1 (Phase 11): parse failure must be observable to scripts --
        // exit 1, not 0. Covers the D-05 --sgfp4 mutex, help/version paths,
        // and every other parse rejection.
        return 1;
    }
    // Convert
    MNN::Cli::convertModel(modelPath);
    return 0;
}
```
The fix: capture `convertModel`'s ignored bool and return non-zero on false — either unconditionally (`if (!MNN::Cli::convertModel(modelPath)) { return 1; }`) or gated per the D-12 scope decision (RESEARCH OQ1: strictest = only exit non-zero for useSGFP4 conversions; planner must pick explicitly and surface at checkpoint). Keep the commented rationale style of the OQ1 block — it is the established convention in this file.

---

### `tools/fp4/README.md` (documentation — usage section)

**Analog:** the file's own "## `mnnconvert --sgfp4` smoke (Phase 11, D-13/D-14 — manual gate)" section (lines ~89-140).

Conventions to replicate for the new E2E section:
- `##` heading naming the tool, phase, and gate status ("test-time manual gate — the corpus is a developer-machine dependency, NOT an always-on CI gate")
- Corpus provenance line: `W:\gnus\models\alexnet_Opset16.onnx` (sha256 `4bc388cc…`, Phase 10 D-01/D-02 approval)
- Numbered PowerShell code blocks with `# expect:` comment lines asserting observable behavior (exit codes, `[DumpPass]` lines)
- Explicit statement of the hard requirements: working Vulkan device/driver (D-07 — no SKIP), built `.build\Release` binaries (`MNNConvert.exe`, `MNNV2Basic.out.exe`, `run_test.out.exe`)
- Document: script parameters (`-Corpus` required), the tolerance formula + Phase 10 anchor citation + locked values, the synthetic-input seed/range, per-backend PASS/FAIL + exit-code semantics, and the D-11 negative-path leg
- Terminology lock: **"SGFP4 v2"** everywhere; never "Ultra FP4"

---

## Shared Patterns

### Exit-code discipline (applies to: script + all three C++ files)
**Sources:** `w2_failcleanup_probe.ps1` (`exit 0/1/2`), `MNNConverter.cpp` OQ1 block (`return 1`), `cli.cpp` (`return false` + `MNN_ERROR`)
Every failure surfaces as a non-zero process exit with a diagnostic. The E2E script's own gating depends on this chain (D-11 exists so the script can trust `MNNConvert`'s exit code): pass failure → `optimizeNetImpl` nullptr + `MNN_ERROR` → `convertModel` null-guard `return false` → main `return 1` → script `$LASTEXITCODE` assertion.

### Error reporting (applies to: all C++ edits)
**Source:** `cli.cpp:752-755` and throughout the converter
`MNN_ERROR("...\n"); return false;` — never exceptions (disabled), never LOG-only for user-visible failure. Reserve `LOG(INFO)` for pass-internal diagnostics.

### Config gating on `Global<modelConfig>` (applies to: PostConverter.cpp escalation)
**Source:** `PostConverter.cpp:145` (`config != nullptr && config->dumpPass`), `InsertSGFP4Dequant.cpp:294` (`nullptr == config || !config->useSGFP4`)
Always null-check the Global before reading; gate all NEW behavior on `useSGFP4` so flag-off is byte-identical (D-12).

### Per-leg CWD isolation + cleanup (applies to: script run legs)
**Source:** `w2_failcleanup_probe.ps1:25-26,45`
Guid-based temp dir per leg; `Remove-Item -Recurse -Force -ErrorAction SilentlyContinue` cleanup; `.tempcache`/`.order`/`output.txt`/`input_0.txt` are CWD-relative in the driver.

### Deterministic seeded fixtures (applies to: script input generation)
**Source:** `test/op/SGFP4TestUtil.hpp` (`std::rand`-seeded style, :59-63)
Fixed documented seed; uniform `[-1, 1)`; identical bytes to all legs; document seed + range in script header and README.

### Regression gates (applies to: every task commit)
**Source:** workstream convention (RESEARCH Validation Architecture)
`.build\Release\run_test.out.exe op/sgfp4` (13 suites) + `.build\Release\TestSGFP4Converter.exe` — the D-12 no-flag-off-change proof. Note `TestSGFP4Converter` drives `RunNetPass` directly (see header note above): any signature change must keep it compiling and passing.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (script comparison core) | test logic | batch | No existing PowerShell numeric-comparison code in the repo — use the RESEARCH-verified formula (max-abs primary + guarded-denominator relative, failing-index diagnostics) |

Everything else has an in-repo analog (script skeleton, all C++ fix sites, README section).

## Metadata

**Analog search scope:** `tools/fp4/`, `tools/converter/source/` + `include/`, `tools/cpp/`, `test/op/`
**Files read:** `w2_failcleanup_probe.ps1`, `PostConverter.cpp` (:100-200, :260-420, :625-680), `PostConverter.hpp`, `cli.cpp` (:680-830), `MNNConverter.cpp` (full), `MNNV2Basic.cpp` (:1-460), `tools/fp4/README.md`, `real_weight_validation_report.json`, `SGFP4TestUtil.hpp` + workspace-wide `useSGFP4` grep
**Pattern extraction date:** 2026-09-01
