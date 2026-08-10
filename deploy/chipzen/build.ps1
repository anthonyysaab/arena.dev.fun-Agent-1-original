# Build, smoke-test, and export the Chipzen upload tarball (PowerShell 5.1+).
# Gzips via .NET instead of a shell pipe: PowerShell pipes corrupt binary
# streams, which is the documented first-time failure on Windows.
param(
    [string]$Tag = "playground-bot:v1"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$Out = Join-Path $ScriptDir "playground-bot.tar.gz"
$TmpTar = Join-Path $ScriptDir "playground-bot.tar"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "docker not found on PATH. Install Docker Desktop first, then re-run."
}

$Weights = Join-Path $RepoRoot "artifacts\tiny-policy-pure.json"
if (-not (Test-Path $Weights)) {
    Write-Error ("{0} missing. Export it first (needs a Python with torch):`n  python tools\export_pure_weights.py" -f $Weights)
}

Write-Host "==> docker build ($Tag)"
docker build --platform linux/amd64 -f (Join-Path $ScriptDir "Dockerfile") -t $Tag $RepoRoot
if ($LASTEXITCODE -ne 0) { Write-Error "docker build failed" }

Write-Host "==> smoke test: construct the bot inside the image"
docker run --rm --entrypoint python $Tag -u -c "from bot import PlaygroundChipzenBot; PlaygroundChipzenBot(); print('bot constructs and weights load OK')"
if ($LASTEXITCODE -ne 0) { Write-Error "in-image smoke test failed" }

$ImageBytes = [long](docker image inspect $Tag --format '{{.Size}}')
$ImageMB = [math]::Round($ImageBytes / 1MB)
Write-Host "==> image size: $ImageMB MB (platform cap: 200 MB)"
if ($ImageMB -gt 200) { Write-Error "image exceeds the 200 MB platform cap" }

Write-Host "==> docker save + gzip -> $Out"
if (Test-Path $TmpTar) { Remove-Item $TmpTar -Force }
docker save -o $TmpTar $Tag
if ($LASTEXITCODE -ne 0) { Write-Error "docker save failed" }
try {
    $inStream = [System.IO.File]::OpenRead($TmpTar)
    $outStream = [System.IO.File]::Create($Out)
    $gzip = New-Object System.IO.Compression.GzipStream($outStream, [System.IO.Compression.CompressionLevel]::Optimal)
    $inStream.CopyTo($gzip)
    $gzip.Dispose(); $outStream.Dispose(); $inStream.Dispose()
} finally {
    if (Test-Path $TmpTar) { Remove-Item $TmpTar -Force }
}

$ArchiveMB = [math]::Round((Get-Item $Out).Length / 1MB)
Write-Host "==> upload artifact: $Out ($ArchiveMB MB compressed, cap: 250 MB)"
if ($ArchiveMB -gt 250) { Write-Error "archive exceeds the 250 MB upload cap" }

Write-Host "Done. Upload $Out through the Chipzen developer UI."
