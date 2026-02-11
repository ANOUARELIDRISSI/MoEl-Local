#############################################################################
# MoEl - High-Performance Local LLM Platform
# Windows PowerShell Automation Script
#############################################################################

param(
    [Parameter(Position=0)]
    [ValidateSet("setup", "start", "stop", "restart", "status", "logs")]
    [string]$Command = "start"
)

$ErrorActionPreference = "Stop"

# Project paths
$PROJECT_ROOT = $PSScriptRoot
$VENV_DIR = Join-Path $PROJECT_ROOT "venv_win"
$BACKEND_DIR = Join-Path $PROJECT_ROOT "backend"
$FRONTEND_DIR = Join-Path $PROJECT_ROOT "frontend"
$LOGS_DIR = Join-Path $PROJECT_ROOT "logs"
$CONFIG_FILE = Join-Path $PROJECT_ROOT "config\hardware.json"

# Colors
function Write-Success { param($msg) Write-Host "[MoEl] $msg" -ForegroundColor Green }
function Write-Error { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Write-Warning { param($msg) Write-Host "[WARNING] $msg" -ForegroundColor Yellow }
function Write-Info { param($msg) Write-Host "[INFO] $msg" -ForegroundColor Cyan }

function Show-Banner {
    Write-Host ""
    Write-Host "=======================================================" -ForegroundColor Blue
    Write-Host "         MoEl - Local LLM Platform (Windows)           " -ForegroundColor Blue
    Write-Host "       High-Performance AI Inference System            " -ForegroundColor Blue
    Write-Host "=======================================================" -ForegroundColor Blue
    Write-Host ""
}

function Test-PythonInstalled {
    try {
        $pythonVersion = python --version 2>&1
        Write-Info "Python found: $pythonVersion"
        return $true
    } catch {
        Write-Error "Python not found. Please install Python 3.8+"
        return $false
    }
}

function Initialize-Setup {
    Show-Banner
    Write-Success "Starting MoEl Setup..."
    
    if (-not (Test-PythonInstalled)) { exit 1 }
    
    # Create directories
    $dirs = @($LOGS_DIR, (Join-Path $PROJECT_ROOT "uploads"), (Join-Path $PROJECT_ROOT "outputs"), (Join-Path $PROJECT_ROOT "models"))
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Info "Created directory: $dir"
        }
    }
    
    # Create virtual environment
    if (-not (Test-Path $VENV_DIR)) {
        Write-Info "Creating virtual environment..."
        python -m venv $VENV_DIR
        Write-Success "Virtual environment created"
    } else {
        Write-Info "Virtual environment already exists"
    }
    
    # Install dependencies
    Write-Info "Installing dependencies..."
    $pipPath = Join-Path $VENV_DIR "Scripts\pip.exe"
    
    # Core dependencies
    & $pipPath install --quiet --upgrade pip
    & $pipPath install --quiet `
        fastapi uvicorn flask flask-cors python-multipart `
        pydantic pydantic-settings requests aiofiles httpx `
        torch transformers accelerate tqdm colorlog numpy
    
    Write-Success "Dependencies installed successfully"
    
    # Download model
    Write-Info "Pre-downloading model (Qwen2.5-0.5B-Instruct - fast CPU inference)..."
    $pythonPath = Join-Path $VENV_DIR "Scripts\python.exe"
    & $pythonPath -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)"
    
    Write-Success "Setup completed! Run '.\run.ps1 start' to launch MoEl"
}

function Get-ProcessOnPort {
    param([int]$Port)
    try {
        $conn = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($conn) { return $conn.OwningProcess }
    } catch {}
    return $null
}

function Stop-PortProcess {
    param([int]$Port)
    $procId = Get-ProcessOnPort -Port $Port
    if ($procId) {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 500
    }
}

function Start-MoEl {
    Show-Banner
    Write-Success "Starting MoEl..."
    
    # Check setup
    if (-not (Test-Path $VENV_DIR)) {
        Write-Error "Setup not completed. Run '.\run.ps1 setup' first"
        exit 1
    }
    
    $pythonPath = Join-Path $VENV_DIR "Scripts\python.exe"
    
    # Stop existing processes
    Write-Info "Checking for existing processes..."
    Stop-PortProcess -Port 8000
    Stop-PortProcess -Port 5000
    
    # Start Backend
    Write-Info "Starting Backend API (port 8000)..."
    $backendScript = @"
import sys
sys.path.insert(0, r'$BACKEND_DIR')
import os
os.chdir(r'$BACKEND_DIR')
import uvicorn
from app import app
uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
"@
    $backendJob = Start-Process -FilePath $pythonPath -ArgumentList "-c", "`"$backendScript`"" -WorkingDirectory $BACKEND_DIR -PassThru -WindowStyle Minimized
    Write-Info "Backend PID: $($backendJob.Id)"
    
    # Wait for backend to initialize
    Write-Info "Waiting for backend to initialize..."
    $maxWait = 60
    $waited = 0
    while ($waited -lt $maxWait) {
        Start-Sleep -Seconds 2
        $waited += 2
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Success "Backend is ready!"
                break
            }
        } catch {}
        Write-Host "." -NoNewline
    }
    Write-Host ""
    
    if ($waited -ge $maxWait) {
        Write-Warning "Backend took longer than expected. It may still be loading the model..."
    }
    
    # Start Frontend
    Write-Info "Starting Frontend Web UI (port 5000)..."
    $frontendJob = Start-Process -FilePath $pythonPath -ArgumentList "app.py" -WorkingDirectory $FRONTEND_DIR -PassThru -WindowStyle Minimized
    Write-Info "Frontend PID: $($frontendJob.Id)"
    
    Start-Sleep -Seconds 3
    
    Write-Host ""
    Write-Success "============================================"
    Write-Success "  MoEl is running!"
    Write-Success "============================================"
    Write-Host ""
    Write-Info "Backend API:    http://localhost:8000"
    Write-Info "Web Interface:  http://localhost:5000"
    Write-Host ""
    Write-Info "Model: Qwen2.5-0.5B-Instruct (CPU-optimized, fast & accurate)"
    Write-Host ""
    Write-Success "Opening browser..."
    Start-Process "http://localhost:5000"
}

function Stop-MoEl {
    Show-Banner
    Write-Success "Stopping MoEl..."
    
    Stop-PortProcess -Port 8000
    Stop-PortProcess -Port 5000
    
    # Kill any remaining Python processes for MoEl
    Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -like "*$VENV_DIR*"
    } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    Write-Success "MoEl stopped"
}

function Get-MoElStatus {
    Show-Banner
    Write-Info "Checking MoEl status..."
    
    $backendPid = Get-ProcessOnPort -Port 8000
    $frontendPid = Get-ProcessOnPort -Port 5000
    
    if ($backendPid) {
        Write-Success "Backend:  RUNNING (PID: $backendPid)"
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
            Write-Info "  Health: OK"
        } catch {
            Write-Warning "  Health: Not responding"
        }
    } else {
        Write-Error "Backend:  STOPPED"
    }
    
    if ($frontendPid) {
        Write-Success "Frontend: RUNNING (PID: $frontendPid)"
    } else {
        Write-Error "Frontend: STOPPED"
    }
}

function Show-Logs {
    Write-Info "Recent logs:"
    $logFiles = Get-ChildItem -Path $LOGS_DIR -Filter "*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 2
    foreach ($log in $logFiles) {
        Write-Host "`n=== $($log.Name) ===" -ForegroundColor Yellow
        Get-Content $log.FullName -Tail 30
    }
}

# Main execution
switch ($Command) {
    "setup"   { Initialize-Setup }
    "start"   { Start-MoEl }
    "stop"    { Stop-MoEl }
    "restart" { Stop-MoEl; Start-Sleep -Seconds 2; Start-MoEl }
    "status"  { Get-MoElStatus }
    "logs"    { Show-Logs }
    default   { Start-MoEl }
}
