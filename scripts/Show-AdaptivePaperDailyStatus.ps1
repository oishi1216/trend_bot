param(
    [int]$Tail = 120,
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$taskName = "FX Adaptive Paper Daily"
$projectPath = "C:\Users\abdaq\trend_bot"
$logDirectory = Join-Path $projectPath "data\task_logs"
$latestLogPath = Join-Path $logDirectory "adaptive_paper_daily_latest.log"
$latestJsonPath = Join-Path $logDirectory "adaptive_paper_daily_latest.json"

Write-Host "=== ADAPTIVE PAPER TASK STATUS ==="

$task = Get-ScheduledTask `
    -TaskName $taskName `
    -ErrorAction SilentlyContinue

if ($null -eq $task) {
    Write-Host "TaskRegistered=False"
}
else {
    $info = Get-ScheduledTaskInfo -TaskName $taskName
    Write-Host "TaskRegistered=True"
    Write-Host "TaskState=$($task.State)"
    Write-Host "NextRunTime=$($info.NextRunTime)"
    Write-Host "LastRunTime=$($info.LastRunTime)"
    Write-Host "LastTaskResult=$($info.LastTaskResult)"

    switch ($info.LastTaskResult) {
        0 { Write-Host "LastResultMeaning=Success" }
        1 { Write-Host "LastResultMeaning=Runtime or instrument error" }
        2 { Write-Host "LastResultMeaning=Safety violation: multiple entries" }
        3 { Write-Host "LastResultMeaning=Configuration safety check failed" }
        default {
            Write-Host "LastResultMeaning=Windows task result or not yet run"
        }
    }
}

Write-Host ""
Write-Host "=== OPEN PAPER POSITIONS ==="

$env:PYTHONPATH = "."
$env:DB_PATH = "data/adaptive_paper.sqlite3"
$env:BROKER_MODE = "mt5_paper"
$env:TRADING_ARMED = "false"
$env:STRATEGY_PROFILE = "adaptive_dual_regime_v1"
$env:MARKET_DATA_CANDLE_COUNT = "3200"
$env:MT5_INSTRUMENTS = "USDJPY,EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,EURJPY,GBPJPY,AUDJPY"
$env:PAPER_INITIAL_BALANCE = "1000000"

Set-Location -LiteralPath $projectPath

@'
from app.config import Settings
from app.storage import Storage

settings = Settings.from_env()
storage = Storage(settings.db_path, settings.paper_initial_balance)
positions = storage.get_positions()

print(f"PositionsOpen={len(positions)}")
for position in positions:
    print(position.to_dict())
'@ | & (Join-Path $projectPath ".venv\Scripts\python.exe") -

Write-Host ""
Write-Host "=== LATEST LOG ==="

if (-not (Test-Path -LiteralPath $latestLogPath)) {
    Write-Host "LatestLogFound=False"
    Write-Host "ExpectedPath=$latestLogPath"
    exit 0
}

$logItem = Get-Item -LiteralPath $latestLogPath
Write-Host "LatestLogFound=True"
Write-Host "LatestLog=$($logItem.FullName)"
Write-Host "LatestLogUpdated=$($logItem.LastWriteTime)"
Write-Host "LatestJson=$latestJsonPath"
Write-Host ""

Get-Content -LiteralPath $latestLogPath -Tail $Tail

if ($Open) {
    Start-Process -FilePath "notepad.exe" -ArgumentList "`"$latestLogPath`""
}
