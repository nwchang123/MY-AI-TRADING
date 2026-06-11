# Launched by the "TradingAgent-Bot" scheduled task at logon. Keeps the
# Telegram control bot alive: if the listener crashes (network blip, machine
# resume), it restarts after 10s. Only the operator's chat id is answered.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "runtime\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "bot.log"

while ($true) {
    Add-Content -Path $log -Value "=== bot start $(Get-Date -Format o) ==="
    cmd /c "python -m trading_agent bot >> `"$log`" 2>&1"
    Add-Content -Path $log -Value "=== bot exited $(Get-Date -Format o), restarting in 10s ==="
    Start-Sleep -Seconds 10
}
