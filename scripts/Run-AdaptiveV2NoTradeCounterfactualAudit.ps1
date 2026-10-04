[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

# Research-only runner for Adaptive v2 Diagnostic #4 (GitHub Issue #14).
# It never syncs history, never opens the paper database, and never sends orders.
# Output paths are fixed repo-relative defaults inside the Python module, so
# the artifacts land under data/research/ of this worktree.

# Resolve the repo root from this script's own location (not the caller's
# current directory) so the runner is safe to invoke from any git worktree.
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
    Write-Host "No .venv interpreter under $repoRoot; using 'python' from PATH."
}

# Resolve a relative -DataDir against the caller's directory before changing
# location, so the path means the same thing it did when typed.
$dataDirFull = [System.IO.Path]::GetFullPath(
    [System.IO.Path]::Combine((Get-Location).ProviderPath, $DataDir)
)

$moduleFile = Join-Path $repoRoot "app\research_no_trade_counterfactual_audit.py"
if (-not (Test-Path -LiteralPath $moduleFile)) {
    throw "Audit module not found in this worktree: $moduleFile"
}

$exitCode = 0
Push-Location -LiteralPath $repoRoot
try {
    if (-not (Test-Path -LiteralPath $dataDirFull)) {
        throw "DataDir not found: $dataDirFull. Pass -DataDir pointing at an already-synced FX research history directory (for example the main checkout's data/fx_research). This research-only runner never syncs MT5 history, never touches the paper database, and never sends orders."
    }

    if ($ValidateOnly) {
        Write-Host "VALIDATE_ONLY_OK repo_root=$repoRoot data_dir=$dataDirFull python=$python module=$moduleFile"
    }
    else {
        & $python -m app.research_no_trade_counterfactual_audit run --data-dir $dataDirFull
        $exitCode = $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

exit $exitCode
