# Launched by the Startup-folder TradingAgent-Bot.vbs at logon, and revived
# by run_paper_loop.ps1 every trading evening. Keeps the Telegram control bot
# alive: if the listener crashes, it restarts after 10s. A single-instance
# guard makes multiple launchers safe (duplicate getUpdates consumers would
# 409 against each other).
$existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "trading_agent\s+bot" }
if ($existing) {
    exit 0
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "runtime\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "bot.log"

while ($true) {
    Add-Content -Path $log -Value "=== bot start $(Get-Date -Format o) ==="
    cmd /c "python -m trading_agent bot >> `"$log`" 2>&1"
    $code = $LASTEXITCODE
    Add-Content -Path $log -Value "=== bot exited $(Get-Date -Format o), code=$code ==="
    if ($code -eq 0) {
        break
    }
    Add-Content -Path $log -Value "=== restarting in 10s ==="
    Start-Sleep -Seconds 10
}
