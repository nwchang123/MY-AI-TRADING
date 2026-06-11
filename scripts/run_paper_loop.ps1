# Launched by the "TradingAgent-PaperLoop" scheduled task before the U.S. open,
# and by start_all.ps1 (one-touch). The loop skips ticks while the market is
# closed (and on U.S. holidays), trades the session autonomously, and exits
# after MaxIterations ticks -- 18 x 30min covers the full session in both EDT
# (21:30-04:00 MY) and EST (22:30-05:00 MY). Telegram pings on
# start/stop/fills/breakers; output is logged via cmd.exe redirection because
# PowerShell 5.1's *>> mangles native stderr into UTF-16 error records.
param(
    [int]$MaxIterations = 18,
    [int]$IntervalSeconds = 1800
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "runtime\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("loop-{0:yyyy-MM-dd}.log" -f (Get-Date))

# Single-instance guard: the scheduled task and a one-touch start must not both
# run -- two run-loops would fight over the per-cycle lock every tick.
$existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "trading_agent\s+run-loop" }
if ($existing) {
    Add-Content -Path $log -Value "=== loop start skipped $(Get-Date -Format o): already running (pid $($existing.ProcessId)) ==="
    exit 0
}

# Revive the Telegram control bot if sleep/reboot killed it (its own
# single-instance guard makes this a no-op when it is already running).
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "run_bot.ps1")
)

Add-Content -Path $log -Value "=== loop start $(Get-Date -Format o) (max $MaxIterations ticks @ ${IntervalSeconds}s) ==="

cmd /c "python -m trading_agent run-loop --auto-universe --interval-seconds $IntervalSeconds --max-iterations $MaxIterations >> `"$log`" 2>&1"

Add-Content -Path $log -Value "=== loop end $(Get-Date -Format o) exit=$LASTEXITCODE ==="
