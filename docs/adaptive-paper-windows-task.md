# Adaptive Paper Windows Task

This task runs `adaptive_dual_regime_v1` once on weekdays at 22:15 local time.
It uses MT5 market data and records paper positions only in:

```text
data/adaptive_paper.sqlite3
```

It does not submit MT5 orders.

## Register and verify

Run from Windows PowerShell 5.1:

```powershell
cd C:\Users\abdaq\trend_bot
git pull --ff-only
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Register-AdaptivePaperDailyTask.ps1 -RunNow
```

The registration script performs these checks before registering the task:

- tracked PowerShell source files pass Parser validation without being modified
- a scheduler-only runtime copy is generated under `data/task_runtime`
- the generated runtime `.ps1` is saved with UTF-8 BOM
- the generated runtime `.ps1` passes PowerShell Parser validation
- the Python daily runner compiles
- required project, Python, runner, and MT5 paths exist

The scheduled task points to:

```text
data/task_runtime/Run-AdaptivePaperDaily.ps1
```

This avoids changing Git-tracked source files during task registration.

The verification run is safe because the strategy is restricted to `mt5_paper` and uses the dedicated paper database.

## Important-event currency blocks

The scheduled runner reads `ENTRY_BLOCKED_CURRENCIES` from the project `.env` file on every run.
Set currencies before major CPI, employment, GDP, or central-bank events:

```dotenv
ENTRY_BLOCKED_CURRENCIES=USD,JPY
```

Clear the block after the event:

```dotenv
ENTRY_BLOCKED_CURRENCIES=
```

The value used by each run is written into the task log.

## View status and latest log

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Show-AdaptivePaperDailyStatus.ps1
```

Open the latest log in Notepad:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Show-AdaptivePaperDailyStatus.ps1 -Open
```

Logs are saved under:

```text
data/task_logs/adaptive_paper_daily_YYYYMMDD_HHMMSS.log
data/task_logs/adaptive_paper_daily_YYYYMMDD_HHMMSS.json
```

The latest copies are:

```text
data/task_logs/adaptive_paper_daily_latest.log
data/task_logs/adaptive_paper_daily_latest.json
```

Timestamped logs older than 90 days are deleted automatically.

## Result codes

| Code | Meaning |
|---:|---|
| 0 | Successful run, including no-entry or already-processed results |
| 1 | Runtime or instrument error |
| 2 | Safety violation: more than one position opened in one run |
| 3 | Configuration safety check failed |

## Remove the task

```powershell
Unregister-ScheduledTask -TaskName "FX Adaptive Paper Daily" -Confirm:$false
```

Removing the task does not delete the paper database, runtime copy, or logs.
