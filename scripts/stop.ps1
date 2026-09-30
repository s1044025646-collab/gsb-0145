# Stop the backend previously started by serve.ps1 (this project's process only).
$root = Split-Path -Parent $PSScriptRoot
$infoPath = Join-Path $root 'tmp/server.json'
if (-not (Test-Path $infoPath)) {
  Write-Output 'no tmp/server.json found; nothing to stop'
  exit 0
}
$info = Get-Content $infoPath | ConvertFrom-Json
if ($info.pid -and (Get-Process -Id $info.pid -ErrorAction SilentlyContinue)) {
  Stop-Process -Id $info.pid
  Write-Output "stopped pid $($info.pid)"
} else {
  Write-Output "pid $($info.pid) was not running"
}
Remove-Item $infoPath -Force
