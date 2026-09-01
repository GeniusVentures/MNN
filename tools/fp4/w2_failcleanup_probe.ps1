# W-2 stale-artifact probe (Phase 11, Plan 11-02, D-10): proves that a
# usage()-exit (bad argument) ALSO removes stale output artifacts at the
# targeted --output path, per tools/fp4/README.md's Failure behavior
# promise. Manual verification step -- not a CI gate.
#
# Usage (from the MNN repo root, after building sgfp4_inject.out):
#   pwsh tools/fp4/w2_failcleanup_probe.ps1 [-Exe <path-to-sgfp4_inject.out.exe>]
#
# Exit 0 = probe PASSED (both stale files removed, exit code 1 observed).

param(
    [string]$Exe = ".build/Release/sgfp4_inject.out.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Exe)) {
    Write-Host "FAIL: sgfp4_inject executable not found at '$Exe' (pass -Exe)"
    exit 2
}

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("sgfp4_w2_probe_" + [System.Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $tmp | Out-Null

$outMnn    = Join-Path $tmp "out.mnn"
$outWeight = Join-Path $tmp "out.mnn.weight"
"stale-model"  | Set-Content $outMnn
"stale-sidecar"| Set-Content $outWeight

# NOTE: --output is passed BEFORE the bad flag so the tool has parsed the
# output path when arg validation fails (the W-2 contract: once --output is
# known, ANY failure path removes stale artifacts at that path).
$proc = Start-Process -FilePath $Exe `
    -ArgumentList("--output", $outMnn, "--bogus-flag") `
    -NoNewWindow -PassThru -Wait
$code = $proc.ExitCode

$mnnGone    = -not (Test-Path $outMnn)
$weightGone = -not (Test-Path $outWeight)

Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue

Write-Host ("exit code          : " + $code)
Write-Host ("stale out.mnn gone : " + $mnnGone)
Write-Host ("stale .weight gone : " + $weightGone)

if ($code -eq 1 -and $mnnGone -and $weightGone) {
    Write-Host "W-2 probe: PASS (arg-stage failure removed both stale artifacts)"
    exit 0
} else {
    Write-Host "W-2 probe: FAIL"
    exit 1
}
