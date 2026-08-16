$ErrorActionPreference = "Stop"

$projectPath = "C:\Users\abdaq\trend_bot"
$pythonPath = Join-Path $projectPath ".venv\Scripts\python.exe"
$mt5Path = "C:\Program Files\OANDA MetaTrader 5\terminal64.exe"
$logDirectory = Join-Path $projectPath "data\task_logs"
$dataDir = Join-Path $projectPath "data\fx_research"
$cachePath = Join-Path $projectPath "data\research\fx_adaptive_backtest.json"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDirectory "adaptive_research_$timestamp.log"
$latestLogPath = Join-Path $logDirectory "adaptive_research_latest.log"
$exitCode = 1
$mutex = $null
$hasLock = $false

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

function Write-TaskLog {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Message
    )

    $Message | Out-File -FilePath $logPath -Encoding utf8 -Append
}

Write-TaskLog "=== FX ADAPTIVE RESEARCH ==="
Write-TaskLog "started_at=$((Get-Date).ToString('o'))"
Write-TaskLog "project=$projectPath"
Write-TaskLog "log=$logPath"
Write-TaskLog ""

try {
    $mutex = New-Object System.Threading.Mutex(
        $false,
        "TrendBotAdaptiveResearch"
    )
    $hasLock = $mutex.WaitOne(0)
    if (-not $hasLock) {
        throw "Another adaptive research run is already active."
    }

    if (-not (Test-Path -LiteralPath $projectPath)) {
        throw "Project directory not found: $projectPath"
    }
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw "Python executable not found: $pythonPath"
    }
    if (-not (Test-Path -LiteralPath $mt5Path)) {
        throw "MT5 terminal not found: $mt5Path"
    }

    Set-Location -LiteralPath $projectPath

    $env:PYTHONPATH = "."
    $env:MT5_TERMINAL_PATH = $mt5Path
    $env:MARKET_DATA_CANDLE_COUNT = "3200"

    Write-TaskLog "step=verify"
    $verifyOutput = & $pythonPath -m app.fx_research_data verify 2>&1
    $verifyExitCode = $LASTEXITCODE
    $verifyOutput | ForEach-Object { $_.ToString() } | Out-File -FilePath $logPath -Encoding utf8 -Append
    Write-TaskLog "verify_exit_code=$verifyExitCode"

    if ($verifyExitCode -ne 0) {
        Write-TaskLog ""
        Write-TaskLog "HUMAN_BOUNDARY=MT5 connection/login is unavailable in this environment."
        Write-TaskLog "HUMAN_BOUNDARY_COMMAND=$pythonPath -m app.fx_research_data verify"
        Write-TaskLog "HUMAN_BOUNDARY_EXPECTED=MT5_CONNECTION=OK: {...}"
        $exitCode = $verifyExitCode
        throw "MT5 connection verification failed; see HUMAN_BOUNDARY lines in the log."
    }

    Write-TaskLog ""
    Write-TaskLog "step=sync"
    $syncOutput = & $pythonPath -m app.fx_research_data sync --data-dir $dataDir --count 3200 2>&1
    $syncExitCode = $LASTEXITCODE
    $syncOutput | ForEach-Object { $_.ToString() } | Out-File -FilePath $logPath -Encoding utf8 -Append
    Write-TaskLog "sync_exit_code=$syncExitCode"
    if ($syncExitCode -ne 0) {
        $exitCode = $syncExitCode
        throw "FX history sync failed for one or more instruments."
    }

    Write-TaskLog ""
    Write-TaskLog "step=run"
    $runOutput = & $pythonPath -m app.research_backtest run --data-dir $dataDir --cache-path $cachePath 2>&1
    $runExitCode = $LASTEXITCODE
    $runOutput | ForEach-Object { $_.ToString() } | Out-File -FilePath $logPath -Encoding utf8 -Append
    Write-TaskLog "run_exit_code=$runExitCode"
    if ($runExitCode -ne 0) {
        $exitCode = $runExitCode
        throw "Research backtest run failed."
    }

    Write-TaskLog ""
    Write-TaskLog "NEXT_STEP=Open the dashboard Research tab (GET /api/research/status) to review refreshed results."
    $exitCode = 0
}
catch {
    if ($exitCode -eq 1 -and $LASTEXITCODE) {
        $exitCode = $LASTEXITCODE
    }
    Write-TaskLog ""
    Write-TaskLog "TASK_ERROR=$($_.Exception.Message)"
    Write-TaskLog "TASK_ERROR_TYPE=$($_.Exception.GetType().FullName)"
}
finally {
    Write-TaskLog "finished_at=$((Get-Date).ToString('o'))"
    Write-TaskLog "exit_code=$exitCode"

    Copy-Item -LiteralPath $logPath -Destination $latestLogPath -Force

    Get-ChildItem -LiteralPath $logDirectory -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like "adaptive_research_20*" -and
            $_.LastWriteTime -lt (Get-Date).AddDays(-90)
        } |
        Remove-Item -Force -ErrorAction SilentlyContinue

    if ($hasLock -and $null -ne $mutex) {
        $mutex.ReleaseMutex()
    }
    if ($null -ne $mutex) {
        $mutex.Dispose()
    }
}

exit $exitCode
