# One-touch starter for the whole trading-agent stack. Safe to run anytime:
# every component has a single-instance guard, so this never double-starts
# what is already running. Brings up, in order: the OpenD gateway (GUI, manual
# login), the Telegram control bot, and the autonomous trading loop.
$ErrorActionPreference = "Continue"
$scripts = $PSScriptRoot
$root = Split-Path -Parent $scripts
Set-Location $root

function Write-Step($tag, $msg) { Write-Host ("[{0}] {1}" -f $tag, $msg) }

Write-Host ""
Write-Host "==================  交易代理 一键启动  =================="
Write-Host ""

# --- 1. OpenD gateway --------------------------------------------------------
$port = (Test-NetConnection -ComputerName 127.0.0.1 -Port 11111 -WarningAction SilentlyContinue).TcpTestSucceeded
if ($port) {
    Write-Step "OpenD" "网关已在线 (127.0.0.1:11111) ✓"
} else {
    $candidates = @(
        "C:\Users\HP\Downloads\MoomooOpenDInstaller\extracted_10.6.6608\moomoo_OpenD_10.6.6608_Windows\moomoo_OpenD-GUI_10.6.6608_Windows\moomoo_OpenD-GUI_10.6.6608_Windows.exe"
    )
    $exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $exe) {
        $exe = Get-ChildItem "C:\Users\HP\Downloads\MoomooOpenDInstaller", "$env:APPDATA\moomoo_OpenD" `
            -Recurse -Filter "*OpenD-GUI*.exe" -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty FullName
    }
    if ($exe) {
        Start-Process $exe
        Write-Step "OpenD" "已启动 GUI -- 请在窗口中登录。循环会在登录后自动恢复。"
    } else {
        Write-Step "OpenD" "找不到 GUI,请手动启动并登录 OpenD (端口 11111)。"
    }
}

# --- 2. Telegram control bot (single-instance guard inside run_bot.ps1) -------
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $scripts "run_bot.ps1")
)
Write-Step "Bot " "控制机器人已确保运行 (24 小时,可在 Telegram 用命令/问答)"

# --- 3. Autonomous trading loop (single-instance guard inside the script) -----
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $scripts "run_paper_loop.ps1")
)
Write-Step "Loop" "交易循环已确保运行 (闭市自动跳过,盘中每 30 分钟一轮)"

Start-Sleep -Seconds 3
Write-Host ""
Write-Host "------------------------  当前状态  ------------------------"
$bot = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "trading_agent\s+bot" }
$loop = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "trading_agent\s+run-loop" }
Write-Host ("  OpenD 端口 11111 : " + $(if ((Test-NetConnection 127.0.0.1 -Port 11111 -WarningAction SilentlyContinue).TcpTestSucceeded) { "在线 ✓" } else { "未登录 (请登录 OpenD)" }))
Write-Host ("  Telegram 机器人  : " + $(if ($bot) { "运行中 (pid $($bot.ProcessId)) ✓" } else { "启动中..." }))
Write-Host ("  交易循环         : " + $(if ($loop) { "运行中 (pid $($loop.ProcessId)) ✓" } else { "启动中..." }))
Write-Host ""
Write-Host "全部就绪。手机 Telegram 会收到上线通知;有交易动作会推送给你。"
Write-Host "==========================================================="
Write-Host ""
Start-Sleep -Seconds 4
