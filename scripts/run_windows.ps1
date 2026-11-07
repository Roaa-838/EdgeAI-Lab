# Run from an extracted D:\EdgeAI-Lab folder. Adjust Python/paths as needed.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = '.\.venv\Scripts\python.exe'
& $py -m edgeai status
& $py -m edgeai smoke-test
Write-Host 'Download and train Imagenette only when you are ready:'
Write-Host '& $py -m edgeai download-imagenette'
Write-Host '& $py -m edgeai train --epochs 8 --batch-size 32'
Write-Host '& $py -m edgeai export --check-parity'
Write-Host '& $py -m edgeai quantize'
Write-Host '& $py -m edgeai benchmark'
Write-Host '& $py -m edgeai report'
