# Launched by the "TradingAgent-PaperLoop" Windows scheduled task on trading
# weekdays before the U.S. open. The loop itself skips ticks while the market
# is closed (and on U.S. holidays), trades the session autonomously, and exits
# after MaxIterations ticks -- so 18 x 30min covers the full session in both
# EDT (21:30-04:00 MY) and EST (22:30-05:00 MY) without DST handling here.
# Telegram pings on start/stop/fills/breakers; output is also logged to file.
# Logging goes through cmd.exe redirection: PowerShell 5.1's own *>> mangles
# native stderr into UTF-16 error records.
param(
    [int]$MaxIterations = 18,
    [int]$IntervalSeconds = 1800
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "runtime\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("loop-{0:yyyy-MM-dd}.log" -f (Get-Date))

Add-Content -Path $log -Value "=== loop start $(Get-Date -Format o) (max $MaxIterations ticks @ ${IntervalSeconds}s) ==="

cmd /c "python -m trading_agent run-loop --auto-universe --interval-seconds $IntervalSeconds --max-iterations $MaxIterations >> `"$log`" 2>&1"

Add-Content -Path $log -Value "=== loop end $(Get-Date -Format o) exit=$LASTEXITCODE ==="
