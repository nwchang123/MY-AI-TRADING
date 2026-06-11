# Launched by the "TradingAgent-PaperLoop" scheduled task before the U.S. open,
# and by start_all.ps1 (one-touch). The run-loop process skips ticks while the
# market is closed, trades the session autonomously, and exits after
# MaxIterations ticks. This wrapper SELF-HEALS: if run-loop dies mid-session
# (e.g. an accidental taskkill, a sleep/resume, an SDK abort) while the U.S.
# market is still open, it restarts it -- so one killed process no longer ends
# the trading day. Output is logged via cmd.exe redirection (PowerShell 5.1's
# *>> mangles native stderr into UTF-16 error records).
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

while ($true) {
    Add-Content -Path $log -Value "=== loop start $(Get-Date -Format o) (max $MaxIterations ticks @ ${IntervalSeconds}s) ==="

    cmd /c "python -m trading_agent run-loop --auto-universe --interval-seconds $IntervalSeconds --max-iterations $MaxIterations >> `"$log`" 2>&1"
    $code = $LASTEXITCODE

    Add-Content -Path $log -Value "=== loop exited $(Get-Date -Format o) code=$code ==="

    # Self-heal only while the market is still open and the exit was abnormal.
    # A clean finish (code 0, ran all ticks -- normally post-close) ends the
    # session, as does any exit once the market has closed.
    $openRaw = & python -c "from datetime import datetime, timezone; from trading_agent.domain.calendar import is_market_hours; print(int(is_market_hours(datetime.now(timezone.utc))))"
    $marketOpen = ("$openRaw".Trim() -eq "1")

    if ($code -eq 0 -or -not $marketOpen) {
        Add-Content -Path $log -Value "=== session done (code=$code, market_open=$marketOpen) ==="
        break
    }

    Add-Content -Path $log -Value "=== run-loop died mid-session; restarting in 15s ==="
    Start-Sleep -Seconds 15
}
