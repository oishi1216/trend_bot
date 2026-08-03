$ErrorActionPreference = "Stop"

$projectPath = "C:\Users\abdaq\trend_bot"
$pythonPath = Join-Path $projectPath ".venv\Scripts\python.exe"
$runnerPath = Join-Path $projectPath "scripts\run_adaptive_paper_daily.py"
$mt5Path = "C:\Program Files\OANDA MetaTrader 5\terminal64.exe"
$dotenvPath = Join-Path $projectPath ".env"
$logDirectory = Join-Path $projectPath "data\task_logs"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDirectory "adaptive_paper_daily_$timestamp.log"
$jsonPath = Join-Path $logDirectory "adaptive_paper_daily_$timestamp.json"
$latestLogPath = Join-Path $logDirectory "adaptive_paper_daily_latest.log"
$latestJsonPath = Join-Path $logDirectory "adaptive_paper_daily_latest.json"
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

function Get-DotEnvValue {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    $pattern = "^\s*" + [regex]::Escape($Name) + "\s*=\s*(.*)$"
    $value = $null
    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        if ($line -match $pattern) {
            $value = $Matches[1].Trim()
        }
    }

    if ($null -eq $value) {
        return $null
    }

    $doubleQuote = [string][char]34
    $singleQuote = [string][char]39
    if (
        ($value.StartsWith($doubleQuote) -and $value.EndsWith($doubleQuote)) -or
        ($value.StartsWith($singleQuote) -and $value.EndsWith($singleQuote))
    ) {
        $value = $value.Substring(1, $value.Length - 2)
    }

    return $value
}

Write-TaskLog "=== ADAPTIVE PAPER WINDOWS TASK ==="
Write-TaskLog "started_at=$((Get-Date).ToString('o'))"
Write-TaskLog "project=$projectPath"
Write-TaskLog "log=$logPath"
Write-TaskLog ""

try {
    $mutex = New-Object System.Threading.Mutex(
        $false,
        "TrendBotAdaptivePaperDaily"
    )
    $hasLock = $mutex.WaitOne(0)
    if (-not $hasLock) {
        throw "Another adaptive paper daily run is already active."
    }

    if (-not (Test-Path -LiteralPath $projectPath)) {
        throw "Project directory not found: $projectPath"
    }
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw "Python executable not found: $pythonPath"
    }
    if (-not (Test-Path -LiteralPath $runnerPath)) {
        throw "Daily runner not found: $runnerPath"
    }
    if (-not (Test-Path -LiteralPath $mt5Path)) {
        throw "MT5 terminal not found: $mt5Path"
    }

    Set-Location -LiteralPath $projectPath

    $branchOutput = & git branch --show-current 2>&1
    $gitExitCode = $LASTEXITCODE
    if ($gitExitCode -ne 0) {
        throw "Failed to read Git branch: $($branchOutput -join ' ')"
    }

    $branch = ($branchOutput | Out-String).Trim()
    $allowedBranches = @(
        "feature/adaptive-dual-regime-v1",
        "master"
    )
    if ($branch -notin $allowedBranches) {
        throw "Unexpected Git branch: $branch"
    }

    Write-TaskLog "branch=$branch"

    $blockedCurrencies = Get-DotEnvValue `
        -Path $dotenvPath `
        -Name "ENTRY_BLOCKED_CURRENCIES"
    if ($null -eq $blockedCurrencies) {
        $blockedCurrencies = ""
    }

    $env:PYTHONPATH = "."
    $env:DB_PATH = "data/adaptive_paper.sqlite3"
    $env:BROKER_MODE = "mt5_paper"
    $env:TRADING_ARMED = "true"
    $env:STRATEGY_PROFILE = "adaptive_dual_regime_v1"
    $env:MARKET_DATA_CANDLE_COUNT = "3200"
    $env:MT5_INSTRUMENTS = "USDJPY,EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,EURJPY,GBPJPY,AUDJPY"
    $env:MT5_TERMINAL_PATH = $mt5Path
    $env:ENTRY_BLOCKED_CURRENCIES = $blockedCurrencies
    $env:OPENAI_FEEDBACK_ENABLED = "false"
    $env:PAPER_INITIAL_BALANCE = "1000000"
    $env:ADAPTIVE_DAILY_JSON_PATH = $jsonPath

    Write-TaskLog "mode=$env:BROKER_MODE"
    Write-TaskLog "profile=$env:STRATEGY_PROFILE"
    Write-TaskLog "database=$env:DB_PATH"
    Write-TaskLog "entry_blocked_currencies=$env:ENTRY_BLOCKED_CURRENCIES"
    Write-TaskLog "trading_armed_for_paper_run=$env:TRADING_ARMED"
    Write-TaskLog ""

    $pythonOutput = & $pythonPath $runnerPath 2>&1
    $pythonExitCode = $LASTEXITCODE

    $pythonOutput |
        ForEach-Object { $_.ToString() } |
        Out-File -FilePath $logPath -Encoding utf8 -Append

    $exitCode = $pythonExitCode
    Write-TaskLog ""
    Write-TaskLog "python_exit_code=$pythonExitCode"
}
catch {
    $exitCode = 1
    Write-TaskLog ""
    Write-TaskLog "TASK_ERROR=$($_.Exception.Message)"
    Write-TaskLog "TASK_ERROR_TYPE=$($_.Exception.GetType().FullName)"
}
finally {
    $env:TRADING_ARMED = "false"
    Write-TaskLog "trading_armed_reset=false"
    Write-TaskLog "finished_at=$((Get-Date).ToString('o'))"
    Write-TaskLog "exit_code=$exitCode"

    Copy-Item -LiteralPath $logPath -Destination $latestLogPath -Force
    if (Test-Path -LiteralPath $jsonPath) {
        Copy-Item -LiteralPath $jsonPath -Destination $latestJsonPath -Force
    }

    Get-ChildItem -LiteralPath $logDirectory -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like "adaptive_paper_daily_20*" -and
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
