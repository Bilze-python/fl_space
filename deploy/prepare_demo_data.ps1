param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$demoExperiments = Join-Path $PSScriptRoot 'demo_data/experiments'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Write-JsonNoBom([string]$path, $value) {
    $json = $value | ConvertTo-Json -Depth 30
    [System.IO.File]::WriteAllText($path, $json, $utf8NoBom)
}

$sources = @(
    @{ name = 'fedleo_offload_seed42'; path = 'fedleo_web_validation_20261015_171358_42' },
    @{ name = 'fedleo_offload_seed20260807'; path = 'fedleo_web_validation_20260915_114213_20260807' },
    @{ name = 'fedleo_offload_seed2'; path = 'experiments/validation_results/fedleo_local_validation_20260810_seed2' }
)

foreach ($source in $sources) {
    $sourceDir = Join-Path $ProjectRoot $source.path
    $target = Join-Path $demoExperiments $source.name
    if (-not (Test-Path -LiteralPath (Join-Path $sourceDir 'validation_summary.json'))) {
        Write-Host "Skipping missing experiment: $sourceDir"
        continue
    }
    New-Item -ItemType Directory -Path (Join-Path $target 'offload_on'), (Join-Path $target 'offload_off') -Force | Out-Null

    $summary = Get-Content -LiteralPath (Join-Path $sourceDir 'validation_summary.json') -Raw | ConvertFrom-Json
    foreach ($key in @('data_dir', 'device', 'verbose')) {
        $summary.config.PSObject.Properties.Remove($key)
    }
    Write-JsonNoBom (Join-Path $target 'validation_summary.json') $summary
    foreach ($variant in @(
        @{ dir = 'offload_on'; file = 'fedleo_offload.json' },
        @{ dir = 'offload_off'; file = 'fedleo_no_offload.json' }
    )) {
        $variantPath = Join-Path (Join-Path $sourceDir $variant.dir) $variant.file
        if (-not (Test-Path -LiteralPath $variantPath)) { continue }
        $raw = Get-Content -LiteralPath $variantPath -Raw | ConvertFrom-Json
        $public = [ordered]@{
            config = $raw.config
            history = $raw.history
        }
        Write-JsonNoBom (Join-Path (Join-Path $target $variant.dir) $variant.file) $public
    }
    Write-Host "Prepared public demo experiment: $target"
}

Write-Host "Public demo data ready in $demoExperiments"
