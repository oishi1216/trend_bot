[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [Parameter(Mandatory = $true)]
    [string]$Diagnostic6JsonPath,
    [string]$JsonPath = "data/research/adaptive_v2_drawdown_recovery_contract_audit.json",
    [string]$MarkdownPath = "data/research/adaptive_v2_drawdown_recovery_contract_audit.md"
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
    throw "Python virtual environment not found in this checkout or the repository's main checkout."
}

$dataDirFull = [System.IO.Path]::GetFullPath(
    [System.IO.Path]::Combine((Get-Location).ProviderPath, $DataDir)
)
$diagnostic6Full = [System.IO.Path]::GetFullPath(
    [System.IO.Path]::Combine((Get-Location).ProviderPath, $Diagnostic6JsonPath)
)

Push-Location -LiteralPath $repoRoot
try {
    if (-not (Test-Path -LiteralPath $dataDirFull)) {
        throw "DataDir not found: $dataDirFull"
    }
    if (-not (Test-Path -LiteralPath $diagnostic6Full)) {
        throw "Diagnostic6JsonPath not found: $diagnostic6Full"
    }

    & $python -m app.research_drawdown_recovery_contract_audit run --data-dir $dataDirFull --diagnostic6-json $diagnostic6Full --json-path $JsonPath --markdown-path $MarkdownPath
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

exit 0
