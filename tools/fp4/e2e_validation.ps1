# SGFP4 v2 end-to-end validation (Phase 12, Plans 12-01/12-02 - SGV2-31/SGV2-32)
#
# Purpose: one committed invocation that converts the approved AlexNet corpus
# to FP32 (baseline) and SGFP4 artifacts, runs BOTH artifacts on CPU and
# Vulkan (classic API), and gates the SGFP4 outputs against the FP32
# baseline with locked tolerances. Closes SGV2-31 (CPU) and SGV2-32
# (Vulkan) for the v3.0 milestone.
#
# Usage (from the MNN repo root, after building .build\Release):
#   pwsh tools/fp4/e2e_validation.ps1 -Corpus W:\gnus\models\alexnet_Opset16.onnx
#   pwsh tools/fp4/e2e_validation.ps1 -Corpus <onnx> -MeasureOnly   # tolerance derivation
#
# Exit semantics:
#   0 = all gates PASS (tolerances + assertions + D-11 negative leg)
#   1 = tolerance or assertion FAIL (per-backend diagnostics printed, D-10)
#   2 = infra: missing binary/corpus or no Vulkan device
#
# Tolerance methodology (measure-then-lock, D-02 anchored):
#   Gate is max-abs (primary) AND guarded relative error (secondary):
#     relErr_i = absErr_i / max(|baseline_i|, Eps),  Eps = 1e-3
#   Form/sanity anchor: tools/fp4/real_weight_validation_report.json
#   (context.thresholds."64": max_mse 0.01, max_relative 0.384 - weight-level
#   metrics, cited for provenance only, never transcribed as output gates).
#   Locked values below = 2.0x measured worst across BOTH backends, <date>.
#   CAVEAT: the driver's output.txt text dump carries ~1e-5 print precision
#   (std::ofstream default ~6 significant digits) - tolerances below that
#   floor are meaningless (Pitfall 2/A3).
#
# Synthetic input (D-03/D-04): seed 20260901, uniform [-1, 1),
# 150,528 floats (1x3x224x224), written ONCE and byte-identically fed to
# all three legs.
#
# Cleanup: on PASS the work root is removed; on FAIL it is KEPT for
# debugging (per-leg dirs contain input_0.txt / output.txt / .order /
# .tempcache).

param(
    [Parameter(Mandatory=$true)][string]$Corpus,
    [string]$MnnConvert = ".build/Release/MNNConvert.exe",
    [string]$Driver = ".build/Release/MNNV2Basic.out.exe",
    [string]$WorkRoot = "tmp/p12_e2e",
    [switch]$MeasureOnly
)

$ErrorActionPreference = "Stop"

# ----------------------------------------------------------------------------
# Locked tolerances (Task 2 fills these from -MeasureOnly data).
# MUST-LOCK: placeholders until measured; do not ship a gate run on 0.0.
# ----------------------------------------------------------------------------
$TolAbs = 0.0   # MUST-LOCK: 2.0 x measured worst max-abs  (both backends)
$TolRel = 0.0   # MUST-LOCK: 2.0 x measured worst max-rel  (both backends)
$Eps    = 1e-3  # guarded relative-error denominator floor (Phase 10 D-07)
$Seed   = 20260901

# ----------------------------------------------------------------------------
# Step 0: infra pre-checks (D-07 hard Vulkan requirement - no SKIP semantics)
# ----------------------------------------------------------------------------
if (-not (Test-Path $Corpus)) { Write-Host "FAIL: corpus not found at '$Corpus'"; exit 2 }
if (-not (Test-Path $MnnConvert)) { Write-Host "FAIL: MNNConvert not found at '$MnnConvert' (pass -MnnConvert)"; exit 2 }
if (-not (Test-Path $Driver)) { Write-Host "FAIL: driver not found at '$Driver' (pass -Driver)"; exit 2 }

# NOTE: capture via cmd /c so vulkaninfo's stderr loader warnings merge as
# plain text — under $ErrorActionPreference=Stop a native 2>&1 redirect in
# PS 5.1 turns stderr lines into ErrorRecords and can throw.
$vulkanInfo = $null
try {
    $vulkanInfo = (& cmd /c "vulkaninfo --summary 2>&1") | Out-String
} catch {
    $vulkanInfo = $null
}
if (-not $vulkanInfo -or ($vulkanInfo -notmatch "deviceName\s*=")) {
    Write-Host "FAIL: no Vulkan device (D-07 hard requirement - no SKIP)"
    exit 2
}

# ----------------------------------------------------------------------------
# Work root
# ----------------------------------------------------------------------------
if (Test-Path $WorkRoot) { Remove-Item -Recurse -Force $WorkRoot }
New-Item -ItemType Directory -Path $WorkRoot | Out-Null
$WorkRoot = (Resolve-Path $WorkRoot).Path
$overallPass = $true
$cleanupWork = $true   # set $false on any failure so artifacts are kept

function Fail-Step([string]$msg) {
    Write-Host "FAIL: $msg"
    $script:cleanupWork = $false
    exit 1
}

try {

# ----------------------------------------------------------------------------
# Step 1: deterministic synthetic input (D-03/D-04)
# ----------------------------------------------------------------------------
$rng = [Random]::new($Seed)
$sb = New-Object System.Text.StringBuilder
for ($i = 0; $i -lt 150528; ++$i) {
    [void]$sb.Append(($rng.NextDouble() * 2.0 - 1.0).ToString("R9", [Globalization.CultureInfo]::InvariantCulture))
    [void]$sb.Append("`n")
}
$inputFile = Join-Path $WorkRoot "input_0.txt"
[IO.File]::WriteAllText($inputFile, $sb.ToString())

# ----------------------------------------------------------------------------
# Step 2: conversions (baseline FP32 + SGFP4 with node-presence assert)
# ----------------------------------------------------------------------------
$absCorpus  = (Resolve-Path $Corpus).Path
$baselineMnn = Join-Path $WorkRoot "baseline.mnn"
$sgfp4Mnn    = Join-Path $WorkRoot "sgfp4.mnn"

$null = & $MnnConvert -f ONNX --modelFile $absCorpus --MNNModel $baselineMnn 2>&1
if ($LASTEXITCODE -ne 0) { Fail-Step "baseline (flag-off) conversion exited $LASTEXITCODE" }

$sgfp4Out = & $MnnConvert -f ONNX --modelFile $absCorpus --MNNModel $sgfp4Mnn --sgfp4 --dumpPass 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host ($sgfp4Out | Out-String)
    Fail-Step "--sgfp4 conversion exited $LASTEXITCODE"
}
$sgfp4Text = $sgfp4Out | Out-String
if ($sgfp4Text -match "InsertSGFP4Dequant: ops (\d+) -> (\d+)") {
    if ($Matches[1] -eq $Matches[2]) {
        Write-Host ($sgfp4Out | Out-String)
        Fail-Step "node-presence: InsertSGFP4Dequant rewrote nothing (ops $($Matches[1]) -> $($Matches[2]))"
    }
    Write-Host "node-presence: InsertSGFP4Dequant ops $($Matches[1]) -> $($Matches[2])"
} else {
    Write-Host ($sgfp4Out | Out-String)
    Fail-Step "node-presence: no 'InsertSGFP4Dequant: ops N -> M' line in --dumpPass stdout"
}

# ----------------------------------------------------------------------------
# Step 3: three isolated run legs (Pattern 1 / Pitfall 3)
#   argv: model runLoops runMask forwardType numberThread precisionMask inputDims
#   precision mask MUST be 1 (Precision_High) on EVERY leg (Pitfall 1).
# ----------------------------------------------------------------------------
$legs = @(
    @{ Name = "baseline-cpu";   Model = $baselineMnn; Forward = "0" },
    @{ Name = "sgfp4-cpu";      Model = $sgfp4Mnn;    Forward = "0" },
    @{ Name = "sgfp4-vulkan";   Model = $sgfp4Mnn;    Forward = "7" }
)
foreach ($leg in $legs) {
    $legDir = Join-Path $WorkRoot ("leg-" + $leg.Name)
    New-Item -ItemType Directory -Path $legDir | Out-Null
    Copy-Item $inputFile (Join-Path $legDir "input_0.txt")
    $absModel = [IO.Path]::GetFullPath($leg.Model)  # model path is resolved by the process, not CWD
    $stdoutFile = Join-Path $legDir "driver_stdout.txt"
    $proc = Start-Process -FilePath $Driver `
        -ArgumentList($absModel, "1", "0", $leg.Forward, "4", "1", "1x3x224x224") `
        -NoNewWindow -PassThru -Wait -WorkingDirectory $legDir `
        -RedirectStandardOutput $stdoutFile
    $legStdout = Get-Content $stdoutFile -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) {
        Write-Host "leg $($leg.Name): driver exited $($proc.ExitCode)"
        $cleanupWork = $false
        Fail-Step "run leg $($leg.Name) failed"
    }
    # Vulkan leg must prove it did not silently fall back (Pitfall 7).
    if ($leg.Forward -eq "7") {
        $stdoutText = $legStdout | Out-String
        if ($stdoutText -notmatch "backendType is 7") {
            Write-Host $stdoutText
            $cleanupWork = $false
            Fail-Step "Vulkan leg did not report 'backendType is 7' (silent CPU fallback?)"
        }
        Write-Host "vulkan backend confirmed: backendType is 7"
    }
    $leg.OutFile = Join-Path $legDir "output.txt"
    if (-not (Test-Path $leg.OutFile)) {
        $cleanupWork = $false
        Fail-Step "leg $($leg.Name): output.txt missing"
    }
}

# ----------------------------------------------------------------------------
# Step 4: comparison core (D-01/D-06/D-10)
# ----------------------------------------------------------------------------
function Parse-Output([string]$path) {
    $text = [IO.File]::ReadAllText($path)
    $tokens = $text -split '\s+' | Where-Object { $_ -ne "" }
    $vals = New-Object System.Collections.Generic.List[double]
    $ci = [Globalization.CultureInfo]::InvariantCulture
    foreach ($t in $tokens) {
        $v = 0.0
        if ([double]::TryParse($t, [Globalization.NumberStyles]::Float, $ci, [ref]$v)) {
            $vals.Add($v)
        }
    }
    return $vals
}

$baseVals = Parse-Output $legs[0].OutFile
if ($baseVals.Count -eq 0) { $cleanupWork = $false; Fail-Step "baseline output parsed to 0 floats" }

$results = @()
foreach ($leg in $legs[1..2]) {
    $sgfp4Vals = Parse-Output $leg.OutFile
    if ($sgfp4Vals.Count -ne $baseVals.Count) {
        Write-Host ("FAIL: " + $leg.Name + " output count " + $sgfp4Vals.Count + " != baseline " + $baseVals.Count)
        $cleanupWork = $false
        exit 1
    }
    $maxAbs = -1.0; $maxAbsIdx = -1
    $maxRel = -1.0; $maxRelIdx = -1
    for ($i = 0; $i -lt $baseVals.Count; ++$i) {
        $absErr = [math]::Abs($sgfp4Vals[$i] - $baseVals[$i])
        $relErr = $absErr / [math]::Max([math]::Abs($baseVals[$i]), $Eps)
        if ($absErr -gt $maxAbs) { $maxAbs = $absErr; $maxAbsIdx = $i }
        if ($relErr -gt $maxRel) { $maxRel = $relErr; $maxRelIdx = $i }
    }
    $backend = if ($leg.Name -eq "sgfp4-cpu") { "cpu" } else { "vulkan" }
    $results += @{ Backend = $backend; MaxAbs = $maxAbs; MaxAbsIdx = $maxAbsIdx; MaxRel = $maxRel; MaxRelIdx = $maxRelIdx }
    if ($MeasureOnly) {
        Write-Host ("MEASURE: {0}: max-abs={1:E8} (idx {2}), max-rel={3:E8} (idx {4})" -f $backend, $maxAbs, $maxAbsIdx, $maxRel, $maxRelIdx)
    } else {
        $pass = ($maxAbs -le $TolAbs) -and ($maxRel -le $TolRel)
        if ($pass) {
            Write-Host ("PASS: {0} max-abs={1:E8} (idx {2}), max-rel={3:E8} (idx {4})" -f $backend, $maxAbs, $maxAbsIdx, $maxRel, $maxRelIdx)
        } else {
            Write-Host ("FAIL: {0} max-abs={1:E8} (idx {2}), max-rel={3:E8} (idx {4})" -f $backend, $maxAbs, $maxAbsIdx, $maxRel, $maxRelIdx)
            $overallPass = $false
            $cleanupWork = $false
        }
    }
}

if ($MeasureOnly) {
    $worstAbs = 0.0; $worstRel = 0.0
    foreach ($r in $results) {
        if ($r["MaxAbs"] -gt $worstAbs) { $worstAbs = $r["MaxAbs"] }
        if ($r["MaxRel"] -gt $worstRel) { $worstRel = $r["MaxRel"] }
    }
    Write-Host ("MEASURE: worst across backends: max-abs={0:E8}, max-rel={1:E8}" -f $worstAbs, $worstRel)
    Write-Host ("MEASURE: suggested lock (2.0x worst): TolAbs={0:E8}, TolRel={1:E8}" -f (2.0 * $worstAbs), (2.0 * $worstRel))
    Write-Host "MEASURE: -MeasureOnly mode - no gating applied"
    exit 0
}

# ----------------------------------------------------------------------------
# Step 5: D-11 negative leg (run BEFORE the overall verdict so earlier
# tolerance diagnostics have already printed; it does not short-circuit).
# ----------------------------------------------------------------------------
$corruptOnnx = Join-Path $WorkRoot "corrupt.onnx"
$garbage = New-Object byte[] 1024
(New-Object Random 1234).NextBytes($garbage)
[IO.File]::WriteAllBytes($corruptOnnx, $garbage)
$negOut = & $MnnConvert -f ONNX --modelFile $corruptOnnx --MNNModel (Join-Path $WorkRoot "neg.mnn") --sgfp4 2>&1
$negCode = $LASTEXITCODE
$negText = $negOut | Out-String
if (($negCode -eq 0) -or ($negText -match "Converted Success!")) {
    Write-Host $negText
    Write-Host "FAIL: D-11 regression: converter lied (exit=$negCode)"
    $overallPass = $false
    $cleanupWork = $false
} else {
    Write-Host "PASS: D-11 negative leg (corrupt + --sgfp4 -> exit $negCode, no 'Converted Success!')"
}

# ----------------------------------------------------------------------------
# Overall verdict
# ----------------------------------------------------------------------------
if ($overallPass) {
    Write-Host "E2E VALIDATION: PASS (cpu + vulkan + D-11 negative)"
    exit 0
} else {
    Write-Host "E2E VALIDATION: FAIL (see per-backend diagnostics above)"
    exit 1
}

} finally {
    if ($cleanupWork -and (Test-Path $WorkRoot) -and -not $MeasureOnly) {
        Remove-Item -Recurse -Force $WorkRoot -ErrorAction SilentlyContinue
    }
}
