# Live monitor: a human-readable dashboard refreshed every few seconds.
# Reads heartbeat + the audit log and renders status / progress / recent
# activity in plain Chinese. Read-only: closing this window stops nothing.
param([switch]$Once)
$root = Split-Path -Parent $PSScriptRoot
try { $Host.UI.RawUI.WindowTitle = "交易代理 实时监控" } catch {}

function Get-MarketOpen {
    try {
        $r = & python -c "from datetime import datetime,timezone; from trading_agent.domain.calendar import is_market_hours; print(int(is_market_hours(datetime.now(timezone.utc))))" 2>$null
        return ("$r".Trim() -eq "1")
    } catch { return $null }
}

function Summarize-Reason($text) {
    if (-not $text) { return "" }
    if ($text -match "dilution")              { return "增发摊薄红旗（确定性拦截）" }
    if ($text -match "insider")               { return "内部人抛售红旗" }
    if ($text -match "out-of-the-money|OTM")  { return "深度虚值彩票" }
    if ($text -match "win probability")       { return "盈利概率不足门槛" }
    if ($text -match "Monte Carlo|POP")       { return "蒙特卡洛基线过低" }
    if ($text -match "VETO|veto")             { return "委员会否决" }
    if ($text -match "red flag")              { return "确定性红旗" }
    $s = ($text -replace "`n"," ").Trim()
    if ($s.Length -gt 46) { return $s.Substring(0,46) + "…" }
    return $s
}

while ($true) {
    $now = Get-Date
    $loop = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "trading_agent\s+run-loop" })
    $loopRunning = [bool]$loop.Count

    $hbAge = "?"; $hbNote = ""
    $hbFile = Join-Path $root "runtime\heartbeat.json"
    if (Test-Path $hbFile) {
        try { $hb = Get-Content $hbFile -Raw | ConvertFrom-Json
              $hbAge = [math]::Round(((Get-Date).ToUniversalTime() - ([datetimeoffset]$hb.at).UtcDateTime).TotalMinutes,1)
              $hbNote = $hb.note } catch {}
    }
    $marketOpen = Get-MarketOpen

    # audit recorded_at is UTC, and a U.S. session (13:30-20:00 UTC) sits inside
    # one UTC date -- but spans two LOCAL dates here (MY = UTC+8, so the session
    # crosses local midnight). Filtering by the LOCAL date silently hid the whole
    # post-midnight half of the session. Use the UTC date to match recorded_at.
    $today = [DateTime]::UtcNow.ToString("yyyy-MM-dd")
    $auditFile = Join-Path $root "runtime\audit.jsonl"
    $events = @()
    if (Test-Path $auditFile) {
        $events = Get-Content $auditFile | ForEach-Object { try { $_ | ConvertFrom-Json } catch {} } |
            Where-Object { $_.recorded_at -like "$today*" }
    }
    $cycles = @($events | Where-Object { $_.event_type -eq "cycle_completed" }).Count
    $scanned = @{}
    $events | Where-Object { $_.event_type -eq "universe_selected" } | ForEach-Object { $_.payload.tickers | ForEach-Object { $scanned[$_]=$true } }
    $committee = @($events | Where-Object { $_.event_type -in @("committee_run","committee_cache_hit") })
    $opens = @($committee | Where-Object { $_.payload.decision -eq "open_position" }).Count
    $rejects = @($committee | Where-Object { $_.payload.decision -eq "reject" }).Count
    $llmCalls = ($events | Where-Object { $_.event_type -eq "llm_usage" } | ForEach-Object { [int]$_.payload.calls } | Measure-Object -Sum).Sum
    $pnl = ($events | Where-Object { $_.event_type -eq "position_closed" } | ForEach-Object { [double]$_.payload.realized_pnl_usd } | Measure-Object -Sum).Sum
    if (-not $pnl) { $pnl = 0 }

    Clear-Host
    Write-Host ""
    Write-Host ("   交易代理 · 实时监控          " + $now.ToString("HH:mm:ss")) -ForegroundColor Cyan
    Write-Host "   ============================================"
    Write-Host "   交易循环 : " -NoNewline
    if ($loopRunning) { Write-Host ("运行中  (pid " + $loop[0].ProcessId + ")") -ForegroundColor Green }
    else { Write-Host "已停止！(双击 Start-TradingAgent 重启)" -ForegroundColor Red }
    Write-Host ("   心跳     : " + $hbAge + " 分钟前   " + $hbNote)
    Write-Host "   美股市场 : " -NoNewline
    if ($marketOpen -eq $true) { Write-Host "开盘中" -ForegroundColor Green }
    elseif ($marketOpen -eq $false) { Write-Host "闭市（循环等待开盘）" -ForegroundColor DarkGray }
    else { Write-Host "未知" -ForegroundColor DarkGray }
    Write-Host ""
    Write-Host "   --- 今日进度 ---" -ForegroundColor Yellow
    Write-Host ("   完成周期    : " + $cycles)
    Write-Host ("   分析股票    : " + $scanned.Count + " 只")
    Write-Host ("   委员会研究  : " + $committee.Count + " 次")
    Write-Host ("   开仓 / 拒绝 : " + $opens + " / " + $rejects)
    Write-Host ("   LLM 调用    : " + $llmCalls + " 次")
    Write-Host "   今日已实现盈亏 : " -NoNewline
    $pnlStr = "$" + ('{0:N2}' -f [double]$pnl)
    if ($pnl -gt 0) { Write-Host $pnlStr -ForegroundColor Green }
    elseif ($pnl -lt 0) { Write-Host $pnlStr -ForegroundColor Red }
    else { Write-Host $pnlStr -ForegroundColor Gray }
    Write-Host ""
    Write-Host "   --- 最近动态 ---" -ForegroundColor Yellow
    $recent = @($events | Where-Object { $_.event_type -in @("universe_selected","committee_run","committee_cache_hit","order_filled","position_closed","circuit_breaker_tripped","kill_switch_activated","cycle_crashed","entries_skipped") } | Select-Object -Last 8)
    if ($recent.Count -eq 0) { Write-Host "   （今天还没有事件）" -ForegroundColor DarkGray }
    foreach ($e in $recent) {
        $t = ([datetimeoffset]$e.recorded_at).LocalDateTime.ToString("HH:mm")
        $p = $e.payload
        switch ($e.event_type) {
            "universe_selected"       { Write-Host ("   " + $t + "  选股 " + @($p.tickers).Count + " 只") }
            "committee_run"           { Write-Host ("   " + $t + "  委员会[" + $p.ticker + "] " + $p.decision + " — " + (Summarize-Reason $p.output.rationale)) }
            "committee_cache_hit"     { Write-Host ("   " + $t + "  委员会[" + $p.ticker + "] " + $p.decision + "（缓存复用）") -ForegroundColor DarkGray }
            "order_filled"            { Write-Host ("   " + $t + "  成交 " + $p.option_code + " @ " + $p.price) -ForegroundColor Green }
            "position_closed"         { Write-Host ("   " + $t + "  平仓 " + $p.option_code + " 盈亏 $" + $p.realized_pnl_usd) }
            "circuit_breaker_tripped" { Write-Host ("   " + $t + "  [熔断] " + $p.reason) -ForegroundColor Red }
            "kill_switch_activated"   { Write-Host ("   " + $t + "  [停机] " + $p.reason) -ForegroundColor Red }
            "cycle_crashed"           { Write-Host ("   " + $t + "  [崩溃] " + $p.error) -ForegroundColor Red }
            "entries_skipped"         { Write-Host ("   " + $t + "  跳过开仓: " + $p.reason) -ForegroundColor DarkYellow }
        }
    }
    Write-Host ""
    Write-Host "   (每 10 秒刷新 · 关闭窗口不影响后台运行 · Ctrl+C 退出)" -ForegroundColor DarkGray
    if ($Once) { break }
    Start-Sleep -Seconds 10
}