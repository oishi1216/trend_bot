param(
    [switch]$RunNow
)

$ErrorActionPreference = "Stop"

$taskName = "FX Adaptive Paper Daily"
$projectPath = "C:\Users\abdaq\trend_bot"
$sourceRunScript = Join-Path $projectPath "scripts\Run-AdaptivePaperDaily.ps1"
$showScript = Join-Path $projectPath "scripts\Show-AdaptivePaperDailyStatus.ps1"
$registerScript = $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectPath ".venv\Scripts\python.exe"
$pythonRunner = Join-Path $projectPath "scripts\run_adaptive_paper_daily.py"
$runtimeDirectory = Join-Path $projectPath "data\task_runtime"
$runtimeRunScript = Join-Path $runtimeDirectory "Run-AdaptivePaperDaily.ps1"

function Assert-Utf8Bom {
    param([Parameter(Mandatory = $true)][string]$Path)

    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $hasBom = (
        $bytes.Length -ge 3 -and
        $bytes[0] -eq 0xEF -and
        $bytes[1] -eq 0xBB -and
        $bytes[2] -eq 0xBF
    )

    if (-not $hasBom) {
        throw "UTF-8 BOM check failed: $Path"
    }
}

function Assert-PowerShellParserPass {
    param([Parameter(Mandatory = $true)][string]$Path)

    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $Path,
        [ref]$tokens,
        [ref]$errors
    ) | Out-Null

    if ($errors.Count -gt 0) {
        $details = $errors |
            ForEach-Object {
                "line=$($_.Extent.StartLineNumber) message=$($_.Message)"
            }
        throw "PowerShell parser check failed for $Path`: $($details -join '; ')"
    }
}

foreach ($path in @(
    $sourceRunScript,
    $showScript,
    $registerScript,
    $pythonPath,
    $pythonRunner
)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required file not found: $path"
    }
}

# Validate tracked source scripts without modifying them.
foreach ($path in @($sourceRunScript, $showScript, $registerScript)) {
    Assert-PowerShellParserPass -Path $path
}

# Generate a scheduler-only runtime copy with UTF-8 BOM.
New-Item `
    -ItemType Directory `
    -Path $runtimeDirectory `
    -Force | Out-Null

$sourceText = [System.IO.File]::ReadAllText($sourceRunScript)
[System.IO.File]::WriteAllText(
    $runtimeRunScript,
    $sourceText,
    (New-Object System.Text.UTF8Encoding($true))
)

Assert-Utf8Bom -Path $runtimeRunScript
Assert-PowerShellParserPass -Path $runtimeRunScript

& $pythonPath -m py_compile $pythonRunner
if ($LASTEXITCODE -ne 0) {
    throw "Python compile check failed: $pythonRunner"
}

$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runtimeRunScript`""

$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -WeeksInterval 1 `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At (Get-Date -Hour 22 -Minute 15 -Second 0)

$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew

Unregister-ScheduledTask `
    -TaskName $taskName `
    -Confirm:$false `
    -ErrorAction SilentlyContinue

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Runs adaptive_dual_regime_v1 once on weekdays using MT5 paper data and a dedicated SQLite database." |
    Out-Null

Write-Host ""
Write-Host "TASK_REGISTERED=PASS"
Write-Host "TRACKED_FILES_UNCHANGED=PASS"
Write-Host "RUNTIME_BOM_CHECK=PASS"
Write-Host "PARSER_CHECK=PASS"
Write-Host "PYTHON_COMPILE_CHECK=PASS"
Write-Host "TaskName=$taskName"
Write-Host "Schedule=Monday-Friday 22:15 local time"
Write-Host "SourceRunScript=$sourceRunScript"
Write-Host "RuntimeRunScript=$runtimeRunScript"
Write-Host "StatusScript=$showScript"
Write-Host ""

Get-ScheduledTask -TaskName $taskName |
    Select-Object TaskName, State

Get-ScheduledTaskInfo -TaskName $taskName |
    Select-Object NextRunTime, LastRunTime, LastTaskResult

if ($RunNow) {
    Write-Host ""
    Write-Host "Starting one safe verification run..."
    Start-ScheduledTask -TaskName $taskName

    $deadline = (Get-Date).AddMinutes(10)
    do {
        Start-Sleep -Seconds 5
        $task = Get-ScheduledTask -TaskName $taskName
        Write-Host "TaskState=$($task.State)"
    } while ($task.State -eq "Running" -and (Get-Date) -lt $deadline)

    if ($task.State -eq "Running") {
        throw "Verification run did not finish within 10 minutes."
    }

    $info = Get-ScheduledTaskInfo -TaskName $taskName
    Write-Host "VerificationLastRunTime=$($info.LastRunTime)"
    Write-Host "VerificationLastTaskResult=$($info.LastTaskResult)"

    if ($info.LastTaskResult -ne 0) {
        throw "Verification run failed with LastTaskResult=$($info.LastTaskResult)"
    }

    Write-Host "TASK_VERIFICATION=PASS"
}
