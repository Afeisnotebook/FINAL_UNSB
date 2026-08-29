[CmdletBinding()]
param(
    [string]$MainRepo = "E:\UNSB_Expl\FINAL_UNSB",
    [Parameter(Mandatory = $true)][string]$AuditRepo,
    [string]$TrainingRepo = "E:\UNSB_Expl\FINAL_UNSB_EXECUTOR_0da2a37",
    [string]$RunRoot = "E:\UNSB_Expl\runs\FINAL_UNSB_LOCAL_ROUTE1_E200",
    [string]$TrainView = "E:\UNSB_Expl\FOUR_METHOD_MOTIVATION_20260813\frozen\data_views_v2\allinone_100",
    [string]$DataRoot = "E:\UNSB_abl\full_dataset",
    [string]$Manifest = "E:\UNSB_Expl\runs\FINAL_UNSB_LOCAL_ROUTE1_E200\operations\frozen_inputs\legacy_split_manifest.csv",
    [string]$Python = "E:\conda\python.exe",
    [string]$TaskName = "FINAL_UNSB_ROUTE1_AUDITOR",
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"
$MainRepo = (Resolve-Path -LiteralPath $MainRepo).Path
$AuditRepo = (Resolve-Path -LiteralPath $AuditRepo).Path
$TrainingRepo = (Resolve-Path -LiteralPath $TrainingRepo).Path
$RunRoot = (Resolve-Path -LiteralPath $RunRoot).Path
$TrainView = (Resolve-Path -LiteralPath $TrainView).Path
$DataRoot = (Resolve-Path -LiteralPath $DataRoot).Path
$Manifest = (Resolve-Path -LiteralPath $Manifest).Path
$Python = (Resolve-Path -LiteralPath $Python).Path

$Supervisor = Join-Path $MainRepo "operations\local_route1_audit_executor.py"
$Contract = Join-Path $RunRoot "operations\AUDIT_EXECUTOR_CONTRACT.json"
& $Python $Supervisor `
    --init-contract `
    --contract $Contract `
    --main-repo $MainRepo `
    --audit-repo $AuditRepo `
    --training-repo $TrainingRepo `
    --run-root $RunRoot `
    --train-view $TrainView `
    --data-root $DataRoot `
    --manifest $Manifest `
    --python $Python
if ($LASTEXITCODE -ne 0) {
    throw "Audit executor contract initialization failed with exit code $LASTEXITCODE"
}

$TaskPython = Join-Path (Split-Path -Parent $Python) "pythonw.exe"
if (-not (Test-Path -LiteralPath $TaskPython)) {
    $TaskPython = $Python
}
$Arguments = '"{0}" --contract "{1}"' -f $Supervisor, $Contract
$Action = New-ScheduledTaskAction -Execute $TaskPython -Argument $Arguments -WorkingDirectory $MainRepo
$UserId = "{0}\{1}" -f $env:USERDOMAIN, $env:USERNAME
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $UserId
$Principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -ExecutionTimeLimit (New-TimeSpan -Days 7) `
    -MultipleInstances IgnoreNew

$Task = New-ScheduledTask -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings
Register-ScheduledTask -TaskName $TaskName -InputObject $Task -Force | Out-Null
if ($StartNow) {
    Start-ScheduledTask -TaskName $TaskName
}
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State, TaskPath
