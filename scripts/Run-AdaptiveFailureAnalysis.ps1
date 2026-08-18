[CmdletBinding()]
param(
    [string]$DataDir = "data/research/fx_research",
    [string]$JsonPath = "data/research/fx_adaptive_failure_analysis.json",
    [string]$MarkdownPath = "data/research/fx_adaptive_failure_analysis.md"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
Push-Location $repoRoot
try {
    & $python -m app.research_failure_analysis --data-dir $DataDir --json-path $JsonPath --markdown-path $MarkdownPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
