# Live monitor window: tails the current trading-loop log in real time.
# Closing this window does NOT stop anything -- it is a read-only viewer.
$root = Split-Path -Parent $PSScriptRoot
try { $Host.UI.RawUI.WindowTitle = "交易代理 实时监控" } catch {}
Write-Host "============  交易代理 实时监控  ============" -ForegroundColor Cyan
Write-Host "（这是只读观察窗;关闭它不影响后台运行）" -ForegroundColor DarkGray
Write-Host ""
while ($true) {
    $latest = Get-ChildItem (Join-Path $root "runtime\logs\loop-*.log") -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) { break }
    Write-Host "还没有交易日志（循环可能在等开盘）。5 秒后重试...  Ctrl+C 退出" -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
Write-Host ("正在实时跟踪: " + $latest.Name) -ForegroundColor Green
Write-Host "------------------------------------------------------------"
Get-Content $latest.FullName -Wait -Tail 40