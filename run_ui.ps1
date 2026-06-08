param(
    [int]$Port = 5000
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$env:FLASK_APP = "web_app.py"
& "$Root\.venv\Scripts\python.exe" -m flask run `
    --host 127.0.0.1 `
    --port $Port
