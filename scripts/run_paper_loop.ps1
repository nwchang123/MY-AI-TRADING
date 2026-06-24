# Launched by the "TradingAgent-PaperLoop" scheduled task before the U.S. open,
# and by start_all.ps1 (one-touch). The run-loop process skips ticks while the
# market is closed, trades the session autonomously, and (with --stop-after-close)
# exits cleanly once the market closes. This wrapper SELF-HEALS: if run-loop dies
# mid-session (e.g. an accidental taskkill, a sleep/resume, an SDK abort) while
# the U.S. market is still open, it restarts it -- so one killed process no
# longer ends the trading day.
#
# CONTINUOUS SELECTION (2026-06-17, operator "一直选"): interval is 60s, so as soon
# as one selection round finishes the next begins (near-continuous), catching new
# catalysts sooner. A per-ticker decision cache keeps re-evaluating UNCHANGED names
# cheap (cache hit, no LLM call); only fresh evidence / newly-eligible contracts
# cost tokens, and the daily token budget still hard-caps spend. --stop-after-close
# ends the session cleanly at the close; MaxIterations is just a high backstop.
#
# Output logging uses .NET Process with file streams (robust with Unicode paths
# like the full-width ！ in the project folder name; avoids cmd.exe and
# PowerShell 5.1 stderr-mangling issues).
param(
    [int]$MaxIterations = 500,
    [int]$IntervalSeconds = 60,
    [int]$WebPort = 8765,
    [bool]$PublishWebPublic = $true
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "runtime\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("loop-{0:yyyy-MM-dd}.log" -f (Get-Date))

function Test-MarketOpenNow {
    $openRaw = & python -c "from datetime import datetime, timezone; from trading_agent.domain.calendar import is_market_hours; print(int(is_market_hours(datetime.now(timezone.utc))))"
    return ("$openRaw".Trim() -eq "1")
}

function Notify-WebDashboardLink {
    param([string]$Url, [string]$LogFile)
    $notifyOutput = & python -m trading_agent notify-web-dashboard --url $Url 2>&1
    if ($notifyOutput) {
        Add-Content -Path $LogFile -Value $notifyOutput -Encoding UTF8
    }
}

function Get-WebToken {
    $tokenPath = Join-Path $root "runtime\web_dashboard.token"
    if (Test-Path $tokenPath) {
        $existing = (Get-Content $tokenPath -Raw -ErrorAction SilentlyContinue).Trim()
        if ($existing) { return $existing }
    }
    $token = & python -c "import secrets; print(secrets.token_urlsafe(24))"
    Set-Content -Path $tokenPath -Value $token -Encoding UTF8
    return "$token".Trim()
}

function Ensure-Cloudflared {
    param([string]$LogFile)
    $toolDir = Join-Path $root "runtime\tools"
    New-Item -ItemType Directory -Force -Path $toolDir | Out-Null
    $exe = Join-Path $toolDir "cloudflared.exe"
    if (Test-Path $exe) { return $exe }

    Add-Content -Path $LogFile -Value "=== downloading cloudflared $(Get-Date -Format o) ==="
    Invoke-WebRequest `
        -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
        -OutFile $exe -UseBasicParsing
    return $exe
}

function Ensure-PublicTunnel {
    param([int]$Port, [string]$Token, [string]$LogFile)

    $publicFile = Join-Path $root "runtime\web_dashboard.public_url.txt"
    $existingTunnel = @(Get-CimInstance Win32_Process -Filter "Name = 'cloudflared.exe'" -ErrorAction SilentlyContinue)
    if ($existingTunnel -and (Test-Path $publicFile)) {
        $existingUrl = (Get-Content $publicFile -Raw -ErrorAction SilentlyContinue).Trim()
        if ($existingUrl) {
            Add-Content -Path $LogFile -Value "=== web tunnel already running $(Get-Date -Format o): $existingUrl (pid $($existingTunnel[0].ProcessId)) ==="
            return $existingUrl
        }
    }

    $exe = Ensure-Cloudflared -LogFile $LogFile
    $out = Join-Path $logDir "cloudflared.out.log"
    $err = Join-Path $logDir "cloudflared.err.log"
    Remove-Item $out, $err -ErrorAction SilentlyContinue

    Start-Process $exe -WindowStyle Hidden -WorkingDirectory $root `
        -ArgumentList @("tunnel", "--url", "http://127.0.0.1:$Port", "--no-autoupdate") `
        -RedirectStandardOutput $out -RedirectStandardError $err

    $baseUrl = $null
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        $text = ""
        if (Test-Path $out) { $text += (Get-Content $out -Raw -ErrorAction SilentlyContinue) }
        if (Test-Path $err) { $text += "`n" + (Get-Content $err -Raw -ErrorAction SilentlyContinue) }
        if ($text -match "https://[a-zA-Z0-9-]+\.trycloudflare\.com") {
            $baseUrl = $Matches[0]
            break
        }
    }

    if (-not $baseUrl) {
        Add-Content -Path $LogFile -Value "=== web tunnel failed to publish URL $(Get-Date -Format o) ==="
        return $null
    }

    $publicUrl = "$baseUrl/?token=$Token"
    Set-Content -Path $publicFile -Value $publicUrl -Encoding UTF8
    Add-Content -Path $LogFile -Value "=== web tunnel published $(Get-Date -Format o): $publicUrl ==="
    return $publicUrl
}

function Ensure-WebDashboard {
    param([int]$Port, [string]$LogFile)

    $token = Get-WebToken
    $url = "http://127.0.0.1:$Port/?token=$Token"
    $existingWeb = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match "trading_agent\s+web-dashboard" })

    if ($existingWeb) {
        Add-Content -Path $LogFile -Value "=== web dashboard already running $(Get-Date -Format o): $url (pid $($existingWeb[0].ProcessId)) ==="
        $notifyUrl = $url
        if ($PublishWebPublic) {
            $published = Ensure-PublicTunnel -Port $Port -Token $token -LogFile $LogFile
            if ($published) { $notifyUrl = $published }
        }
        Notify-WebDashboardLink -Url $notifyUrl -LogFile $LogFile
        return
    }

    $webOut = Join-Path $logDir "web-dashboard.out.log"
    $webErr = Join-Path $logDir "web-dashboard.err.log"
    Start-Process python -WindowStyle Hidden -WorkingDirectory $root `
        -ArgumentList @("-m", "trading_agent", "web-dashboard", "--host", "127.0.0.1", "--port", "$Port", "--token", $token) `
        -RedirectStandardOutput $webOut -RedirectStandardError $webErr

    $online = $false
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 500
        $online = (Test-NetConnection 127.0.0.1 -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded
        if ($online) { break }
    }

    if ($online) {
        Add-Content -Path $LogFile -Value "=== web dashboard started $(Get-Date -Format o): $url ==="
        $notifyUrl = $url
        if ($PublishWebPublic) {
            $published = Ensure-PublicTunnel -Port $Port -Token $token -LogFile $LogFile
            if ($published) { $notifyUrl = $published }
        }
        Notify-WebDashboardLink -Url $notifyUrl -LogFile $LogFile
    } else {
        Add-Content -Path $LogFile -Value "=== web dashboard start attempted but port $Port is not reachable $(Get-Date -Format o) ==="
    }
}

function Stop-WebDashboard {
    param([string]$LogFile)

    $webProcesses = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match "trading_agent\s+web-dashboard" }
    foreach ($p in $webProcesses) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        Add-Content -Path $LogFile -Value "=== web dashboard stopped $(Get-Date -Format o): pid $($p.ProcessId) ==="
    }
    $tunnels = Get-CimInstance Win32_Process -Filter "Name = 'cloudflared.exe'" -ErrorAction SilentlyContinue
    foreach ($p in $tunnels) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        Add-Content -Path $LogFile -Value "=== web tunnel stopped $(Get-Date -Format o): pid $($p.ProcessId) ==="
    }
}

# Revive the Telegram control bot if sleep/reboot killed it (its own
# single-instance guard makes this a no-op when it is already running).
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", ('"' + (Join-Path $PSScriptRoot "run_bot.ps1") + '"')
)

if (Test-MarketOpenNow) {
    Ensure-WebDashboard -Port $WebPort -LogFile $log
}

# Single-instance guard: the scheduled task and a one-touch start must not both
# run -- two run-loops would fight over the per-cycle lock every tick. This runs
# AFTER the web check so the exact market-open task can publish the dashboard
# link even when the pre-open wrapper already started the run-loop.
$existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "trading_agent\s+run-loop" }
if ($existing) {
    Add-Content -Path $log -Value "=== loop start skipped $(Get-Date -Format o): already running (pid $($existing.ProcessId)) ==="
    exit 0
}

# Helper: run python and merge stdout+stderr into the log file using .NET
# Process API. This avoids cmd.exe (which can mishandle Unicode paths in
# hidden-window contexts) and PowerShell 5.1's stderr-to-ErrorRecord mangling.
function Wait-OpenDReady {
    # The moomoo OpenD gateway must be listening on 127.0.0.1:11111 before the
    # run-loop opens quote/trade contexts. When the scheduled task fires before
    # OpenD has finished launching (e.g. both starting at login), the first
    # cycles abort with "OpenD gateway is not reachable" -- 9 such crashed
    # cycles on 2026-06-23. Poll briefly so the common concurrent-startup race
    # is absorbed; if OpenD is still down after the bound, proceed anyway (the
    # run-loop already survives a per-cycle gateway outage and self-recovers).
    param([string]$LogFile, [int]$TimeoutSeconds = 120)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $ok = (Test-NetConnection 127.0.0.1 -Port 11111 -WarningAction SilentlyContinue).TcpTestSucceeded
        if ($ok) {
            Add-Content -Path $LogFile -Value "=== OpenD gateway ready on 127.0.0.1:11111 $(Get-Date -Format o) ==="
            return $true
        }
        Start-Sleep -Seconds 2
    }
    Add-Content -Path $LogFile -Value "=== OpenD gateway not reachable after ${TimeoutSeconds}s; starting loop anyway $(Get-Date -Format o) ==="
    return $false
}

function Run-PythonLogged {
    param([string]$Arguments, [string]$LogFile)

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "python"
    $psi.Arguments = $Arguments
    $psi.WorkingDirectory = $root
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi

    # Collect output asynchronously to avoid deadlocks
    $outBuf = [System.Text.StringBuilder]::new()
    $errBuf = [System.Text.StringBuilder]::new()

    $outHandler = { if ($EventArgs.Data -ne $null) { $Event.MessageData.AppendLine($EventArgs.Data) } }
    $errHandler = { if ($EventArgs.Data -ne $null) { $Event.MessageData.AppendLine($EventArgs.Data) } }

    $outEvent = Register-ObjectEvent -InputObject $p -EventName OutputDataReceived -Action $outHandler -MessageData $outBuf
    $errEvent = Register-ObjectEvent -InputObject $p -EventName ErrorDataReceived -Action $errHandler -MessageData $errBuf

    $p.Start() | Out-Null
    $p.BeginOutputReadLine()
    $p.BeginErrorReadLine()

    # Wait for process to exit — this blocks until python finishes all iterations
    $p.WaitForExit()

    # Ensure all async output is flushed
    Start-Sleep -Milliseconds 500
    Unregister-Event -SourceIdentifier $outEvent.Name
    Unregister-Event -SourceIdentifier $errEvent.Name

    # Append captured output to log
    $combined = ($errBuf.ToString() + $outBuf.ToString()).TrimEnd()
    if ($combined) {
        Add-Content -Path $LogFile -Value $combined -Encoding UTF8
    }

    return $p.ExitCode
}

# Absorb the OpenD startup race before the first run-loop tick so the session
# does not open with a burst of "gateway not reachable" crashed cycles.
Wait-OpenDReady -LogFile $log | Out-Null

while ($true) {
    Add-Content -Path $log -Value "=== loop start $(Get-Date -Format o) (max $MaxIterations ticks @ ${IntervalSeconds}s) ==="

    $code = Run-PythonLogged `
        -Arguments "-m trading_agent run-loop --auto-universe --interval-seconds $IntervalSeconds --max-iterations $MaxIterations --stop-after-close" `
        -LogFile $log

    Add-Content -Path $log -Value "=== loop exited $(Get-Date -Format o) code=$code ==="

    # Self-heal only while the market is still open and the exit was abnormal.
    # A clean finish (code 0, ran all ticks -- normally post-close) ends the
    # session, as does any exit once the market has closed.
    $openRaw = & python -c "from datetime import datetime, timezone; from trading_agent.domain.calendar import is_market_hours; print(int(is_market_hours(datetime.now(timezone.utc))))"
    $marketOpen = ("$openRaw".Trim() -eq "1")

    if ($code -eq 0 -or -not $marketOpen) {
        Add-Content -Path $log -Value "=== session done (code=$code, market_open=$marketOpen) ==="
        Stop-WebDashboard -LogFile $log
        break
    }

    Add-Content -Path $log -Value "=== run-loop died mid-session; restarting in 15s ==="
    Start-Sleep -Seconds 15
}
