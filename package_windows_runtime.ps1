param(
    [string]$ProjectRoot = $PSScriptRoot,
    [string]$OutputRoot = "",
    [string]$BundleName = "",
    [switch]$IncludeCheckpoints
)

$ErrorActionPreference = "Stop"

if (-not $OutputRoot) {
    $OutputRoot = Join-Path $ProjectRoot "dist"
}

$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not $BundleName) {
    $BundleName = "zsad_windows_runtime_$timestamp"
}

$bundleDir = Join-Path $OutputRoot $BundleName
$zipPath = "$bundleDir.zip"
$manifestDir = Join-Path $bundleDir "environment"
$projectDir = Join-Path $bundleDir "project"

New-Item -ItemType Directory -Force -Path $bundleDir | Out-Null
New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null
New-Item -ItemType Directory -Force -Path $projectDir | Out-Null

Write-Host "[1/4] Exporting environment metadata..."

$pythonInfo = @{
    python_executable = (Get-Command python).Source
    python_version = (python --version) 2>&1
    generated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    conda_default_env = $env:CONDA_DEFAULT_ENV
}
$pythonInfo | ConvertTo-Json -Depth 3 | Set-Content -Path (Join-Path $manifestDir "python_info.json") -Encoding UTF8

python -m pip freeze | Set-Content -Path (Join-Path $manifestDir "pip_freeze.txt") -Encoding UTF8

if (Get-Command conda -ErrorAction SilentlyContinue) {
    conda env export --no-builds | Set-Content -Path (Join-Path $manifestDir "conda_env.yml") -Encoding UTF8
} else {
    "conda command not found; skipped conda env export." | Set-Content -Path (Join-Path $manifestDir "conda_env.txt") -Encoding UTF8
}

if (Test-Path (Join-Path $ProjectRoot "requirements.txt")) {
    Copy-Item (Join-Path $ProjectRoot "requirements.txt") (Join-Path $manifestDir "requirements.txt") -Force
}

$packageReadme = @"
This bundle was generated from the zsad project on Windows.

Contents:
- environment/: exported Python and package metadata
- project/: project source snapshot

Notes:
- Checkpoint files are excluded by default.
- Download model weights separately before running inference.
"@
$packageReadme | Set-Content -Path (Join-Path $bundleDir "README_bundle.txt") -Encoding UTF8

Write-Host "[2/4] Copying project files..."

$excludeDirs = @(".git", "__pycache__", ".venv", "dist", "benchmark_results", "results")
if (-not $IncludeCheckpoints) {
    $excludeDirs += @("checkpoint", "checkpoints")
}

$excludePatterns = @("*.pyc", "*.pyo", "*.zip")

Get-ChildItem -Path $ProjectRoot -Force | ForEach-Object {
    $name = $_.Name
    if ($excludeDirs -contains $name) {
        return
    }

    $destination = Join-Path $projectDir $name
    if ($_.PSIsContainer) {
        Copy-Item $_.FullName $destination -Recurse -Force
    } else {
        $skip = $false
        foreach ($pattern in $excludePatterns) {
            if ($name -like $pattern) {
                $skip = $true
                break
            }
        }
        if (-not $skip) {
            Copy-Item $_.FullName $destination -Force
        }
    }
}

Write-Host "[3/4] Cleaning copied cache files..."
Get-ChildItem -Path $projectDir -Recurse -Force -Directory | Where-Object { $_.Name -eq "__pycache__" } | Remove-Item -Recurse -Force
Get-ChildItem -Path $projectDir -Recurse -Force -Include *.pyc,*.pyo | Remove-Item -Force

Write-Host "[4/4] Creating zip archive..."
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}
Compress-Archive -Path $bundleDir -DestinationPath $zipPath -Force

Write-Host ""
Write-Host "Bundle directory: $bundleDir"
Write-Host "Bundle zip:       $zipPath"
