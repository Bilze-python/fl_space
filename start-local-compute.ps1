param(
    [string]$Server = "https://spacefl-research-agent.onrender.com",
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [ValidateRange(1, 64)][int]$Threads = 2,
    [ValidateRange(10, 86400)][int]$MaxSeconds = 3600
)

$ErrorActionPreference = "Stop"
$spaceflPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $spaceflPython)) {
    throw "请先按 docs/LOCAL_COMPUTE.md 创建 venv 并安装本地依赖。"
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $spaceflPython -m web.local_worker connect --server $Server --device $Device --threads $Threads --max-seconds $MaxSeconds
    if ($LASTEXITCODE -ne 0) { throw "本地连接器已停止，请查看上方提示。" }
} finally {
    Pop-Location
}
