# Launched by the "TradingAgent-PaperLoop" scheduled task before the U.S. open,
# and by start_all.ps1 (one-touch). The run-loop process skips ticks while the
# market is closed, trades the session autonomously, and exits after
# MaxIterations ticks. This wrapper SELF-HEALS: if run-loop dies mid-session
# (e.g. an accidental taskkill, a sleep/resume, an SDK abort) while the U.S.
# market is still open, it restarts it -- so one killed process no longer ends
# the trading day.
#
# Output logging uses .NET Process with file streams (robust with Unicode paths
# like the full-width ！ in the project folder name; avoids cmd.exe and
# PowerShell 5.1 stderr-mangling issues).
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
    "-File", ('"' + (Join-Path $PSScriptRoot "run_bot.ps1") + '"')
)

# Helper: run python and merge stdout+stderr into the log file using .NET
# Process API. This avoids cmd.exe (which can mishandle Unicode paths in
# hidden-window contexts) and PowerShell 5.1's stderr-to-ErrorRecord mangling.
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

while ($true) {
    Add-Content -Path $log -Value "=== loop start $(Get-Date -Format o) (max $MaxIterations ticks @ ${IntervalSeconds}s) ==="

    $code = Run-PythonLogged `
        -Arguments "-m trading_agent run-loop --auto-universe --interval-seconds $IntervalSeconds --max-iterations $MaxIterations" `
        -LogFile $log

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
