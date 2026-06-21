param(
    [int]$Port = 8000,
    [string]$HostAddress = "0.0.0.0"
)

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1

if ($listener) {
    Write-Host "Backend already running: http://127.0.0.1:$Port (PID $($listener.OwningProcess))"
    Write-Host "If you need a clean restart, run: powershell -ExecutionPolicy Bypass -File .\scripts\stop_dev_ports.ps1"
    exit 0
}

Write-Host "Starting backend on http://127.0.0.1:$Port ..."
py -m uvicorn app.main:app --reload --host $HostAddress --port $Port
