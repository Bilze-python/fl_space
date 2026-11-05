param(
    [int]$Port = 8720,
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Continue"
$deployDir = $PSScriptRoot
$logs = Join-Path $deployDir "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$logFile = Join-Path $logs "keepalive.log"

function Write-Log([string]$message) {
    $line = ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $message)
    Add-Content -Path $logFile -Value $line -Encoding UTF8
}

function Test-Health([string]$url) {
    $code = & curl.exe -s -o NUL -w "%{http_code}" --max-time 8 --connect-timeout 5 $url 2>$null
    return ($code -eq "200")
}

$lock = Join-Path $deployDir "keepalive.lock"
if (Test-Path $lock) {
    $age = (Get-Date) - (Get-Item $lock).LastWriteTime
    if ($age.TotalSeconds -lt 120) { Write-Log "skip: previous run active"; exit 0 }
}
Set-Content -Path $lock -Value (Get-Date -Format o)

try {
    $python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
    if (-not (Test-Path $python)) { Write-Log "ERROR: venv python missing"; exit 1 }

    $env:SPACEFL_PUBLIC_DEMO = "1"
    $env:SPACEFL_OFFLINE_ONLY = "1"
    $localHealth = "http://127.0.0.1:$Port/api/health"

    if (-not (Test-Health $localHealth)) {
        Write-Log "starting demo server on $Port"
        Start-Process -FilePath $python -ArgumentList @(
            "-m", "uvicorn", "--factory", "web.server:create_app",
            "--host", "127.0.0.1", "--port", "$Port", "--log-level", "warning"
        ) -WorkingDirectory $ProjectRoot -WindowStyle Hidden `
          -RedirectStandardOutput (Join-Path $logs "server.out.log") `
          -RedirectStandardError (Join-Path $logs "server.err.log") | Out-Null
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Milliseconds 1000
            if (Test-Health $localHealth) { break }
        }
    }
    if (-not (Test-Health $localHealth)) { Write-Log "ERROR: demo server not healthy"; exit 1 }

    $urlFile = Join-Path $deployDir "public_url.txt"
    if (Test-Path $urlFile) {
        $currentUrl = (Get-Content $urlFile -TotalCount 1 -ErrorAction SilentlyContinue)
        if ($currentUrl -and (Test-Health ("$currentUrl/api/health"))) {
            Write-Log "reuse existing public URL: $currentUrl"
            exit 0
        }
    }

    $ssh = "C:\Windows\System32\OpenSSH\ssh.exe"
    $sshOut = Join-Path $logs "ssh-tunnel.out.log"
    $sshErr = Join-Path $logs "ssh-tunnel.err.log"

    Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine -match "localhost.run" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Remove-Item $sshOut, $sshErr -Force -ErrorAction SilentlyContinue

    Write-Log "opening SSH tunnel via localhost.run"
    Start-Process -FilePath $ssh -ArgumentList @(
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=NUL",
        "-o", "ServerAliveInterval=30",
        "-o", "ExitOnForwardFailure=yes",
        "-R", "80:127.0.0.1:$Port",
        "nokey@localhost.run"
    ) -RedirectStandardOutput $sshOut -RedirectStandardError $sshErr -WindowStyle Hidden | Out-Null

    $publicUrl = $null
    for ($i = 0; $i -lt 100; $i++) {
        Start-Sleep -Milliseconds 1000
        $content = ""
        foreach ($file in @($sshOut, $sshErr)) {
            try { $content += (Get-Content $file -Raw -ErrorAction SilentlyContinue) } catch { }
        }
        $matches = [regex]::Matches($content, "https://[a-zA-Z0-9]+\.lhr\.life")
        if ($matches.Count -gt 0) {
            $candidate = $matches[$matches.Count - 1].Value
            if (Test-Health ("$candidate/api/health")) { $publicUrl = $candidate; break }
        }
    }

    if (-not $publicUrl) { Write-Log "ERROR: could not establish public tunnel"; exit 1 }

    $previousUrl = ""
    if (Test-Path $urlFile) { $previousUrl = (Get-Content $urlFile -TotalCount 1) }
    $record = @(
        $publicUrl,
        ("updated: " + (Get-Date -Format o)),
        ("port: " + $Port),
        ("local: http://127.0.0.1:$Port/")
    ) -join "`n"
    [System.IO.File]::WriteAllText($urlFile, $record, (New-Object System.Text.UTF8Encoding($false)))
    Write-Log "public URL ready: $publicUrl"

    # 隧道地址变化时，同步更新 HF 免费静态入口（可选，需要 cookie.txt）
    if ($previousUrl -ne $publicUrl) {
        $updater = Join-Path $ProjectRoot "deploy\huggingface\update_static_space.py"
        $cookieFile = Join-Path $ProjectRoot "deploy\huggingface\cookie.txt"
        if ((Test-Path $updater) -and (Test-Path $cookieFile)) {
            $proxy = "http://127.0.0.1:10808"
            $proxyFile = Join-Path $ProjectRoot "deploy\huggingface\proxy.txt"
            if (Test-Path $proxyFile) { $proxy = (Get-Content $proxyFile -TotalCount 1).Trim() }
            $env:HTTPS_PROXY = $proxy
            $env:HTTP_PROXY = $proxy
            try {
                $output = & $python $updater 2>&1
                Write-Log ("static HF entry updated: " + ($output -join " | "))
            } catch {
                Write-Log ("static HF entry update failed: " + $_.Exception.Message)
            }
        }
    }
} finally {
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
}
