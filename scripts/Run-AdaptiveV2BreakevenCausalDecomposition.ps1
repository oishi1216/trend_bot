[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [string]$JsonPath = "data/research/adaptive_v2_breakeven_causal_decomposition.json",
    [string]$MarkdownPath = "data/research/adaptive_v2_breakeven_causal_decomposition.md"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $commonGitDir = & git -C $repoRoot rev-parse --path-format=absolute --git-common-dir 2>$null
    if ($LASTEXITCODE -eq 0 -and $commonGitDir) {
        $mainCheckoutRoot = Split-Path -Parent ($commonGitDir | Select-Object -First 1)
        $sharedPython = Join-Path $mainCheckoutRoot ".venv\Scripts\python.exe"
        if (Test-Path -LiteralPath $sharedPython) {
            $python = $sharedPython
        }
    }
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found in this checkout or the repository's main checkout. Create .venv before running the research diagnostic."
}

$dataDirFull = [System.IO.Path]::GetFullPath(
    [System.IO.Path]::Combine((Get-Location).ProviderPath, $DataDir)
)

Push-Location -LiteralPath $repoRoot
try {
    if (-not (Test-Path -LiteralPath $dataDirFull)) {
        throw "DataDir not found: $dataDirFull. Pass -DataDir pointing at already-synced FX research history. This research-only diagnostic never syncs MT5 history, never touches the paper SQLite database, and never sends orders."
    }

    & $python -m app.research_breakeven_causal_decomposition run --data-dir $dataDirFull --json-path $JsonPath --markdown-path $MarkdownPath
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

exit 0
