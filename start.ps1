# Starts Ollama with the GPU settings the bot needs, then the bot.
#   powershell -ExecutionPolicy Bypass -File start.ps1
# Flash attention + 8-bit KV cache shrink the LLM's VRAM so qwen2.5:7b fits completely on a 6 GB GPU
# (no layers on the CPU). Ollama must be (re)started with these variables; a tray Ollama started at login won't have them.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$cfg = @{}
Get-Content .env -Encoding UTF8 | Where-Object { $_ -match '^\s*([A-Z_]+)\s*=\s*(.*)$' } | ForEach-Object {
    $cfg[$Matches[1]] = $Matches[2].Trim()
}

$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
if ($cfg["OLLAMA_MODELS"]) { $env:OLLAMA_MODELS = $cfg["OLLAMA_MODELS"] }

$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) { $ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" }

$running = Get-Process ollama -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "Restarting Ollama with GPU settings (flash attention, q8_0 KV cache)…"
    Get-Process "ollama app" -ErrorAction SilentlyContinue | Stop-Process -Force
    $running | Stop-Process -Force
    Start-Sleep 2
}
New-Item -ItemType Directory -Force data | Out-Null
Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden `
    -RedirectStandardOutput data\ollama.out.log -RedirectStandardError data\ollama.log
for ($i = 0; $i -lt 30; $i++) {
    try { Invoke-RestMethod http://127.0.0.1:11434/api/version -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep 1 }
}

Write-Host "Starting the bot (log: data\bot.log)…"
& .venv\Scripts\python.exe -X utf8 run.py *>> data\bot.log
