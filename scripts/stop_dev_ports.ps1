param(
    [int[]]$Ports = @(8000, 5173)
)

foreach ($port in $Ports) {
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue

    if (-not $listeners) {
        Write-Host "No listener on port $port"
        continue
    }

    foreach ($listener in $listeners) {
        $processId = $listener.OwningProcess
        Write-Host "Stopping PID $processId on port $port ..."
        Stop-Process -Id $processId -Force
    }
}
