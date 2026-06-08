param(
    [int]$Port = 8888
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

& "$Root\.venv\Scripts\python.exe" -m notebook `
    --notebook-dir "$Root" `
    --ip 127.0.0.1 `
    --port $Port `
    --no-browser
