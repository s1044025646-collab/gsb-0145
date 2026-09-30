# Start the backend bound to 127.0.0.1 on an OS-assigned port.
# Manages only this project's node process. Port/PID are written to tmp/server.json.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path tmp, data | Out-Null

if (Test-Path tmp/server.json) {
  $old = Get-Content tmp/server.json | ConvertFrom-Json
  if ($old.pid -and (Get-Process -Id $old.pid -ErrorAction SilentlyContinue)) {
    Write-Output "already running: http://127.0.0.1:$($old.port) (pid $($old.pid))"
    exit 0
  }
}

if (-not $env:CF_DB_PATH) { $env:CF_DB_PATH = 'data/app.db' }
$proc = Start-Process node -ArgumentList '--experimental-strip-types','--no-warnings','src/api/server.ts' -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 2
$info = Get-Content tmp/server.json | ConvertFrom-Json
Write-Output "listening on http://127.0.0.1:$($info.port) (pid $($proc.Id))"
