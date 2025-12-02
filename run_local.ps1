# Run this script in Powershell with: .\run_local.ps1
param(
    [string]$BackendDir = "backend",
    [string]$FrontendDir = "frontend",
    [string]$BackendApp = "app/app.py",
    [string]$PythonExe = "python"   # or "python3" if needed
)

function Log($msg) {
    Write-Host ""
    Write-Host "===> $msg" -ForegroundColor Cyan
    Write-Host ""
}


# Setup & run Flask backend
Log "Setting up and starting Flask backend (SQLite)..."

Set-Location $BackendDir

# Create venv if missing
if (-not (Test-Path ".venv")) {
    Log "Creating virtual environment..."
    & $PythonExe -m venv .venv
}

# Activate venv
$venvActivate = Join-Path ".venv" "Scripts\Activate.ps1"
. $venvActivate

# Install dependencies
if (Test-Path "requirements.txt") {
    Log "Installing backend dependencies..."
    pip install -r requirements.txt
}

# Start Flask backend in background
Log "Starting Flask API..."
$backendProc = Start-Process $PythonExe -ArgumentList $BackendApp -PassThru

Set-Location ..

# Setup & run frontend
Log "Setting up and starting React/Vite frontend..."

Set-Location $FrontendDir

if (Test-Path "package.json") {
    Log "Installing frontend dependencies (npm install)..."
    npm install
}

Log "Starting Vite dev server (React frontend)..."
npm run dev

# When npm run dev exits (Ctrl+C), stop backend
Log "Stopping Flask backend (PID: $($backendProc.Id))..."
try {
    Stop-Process -Id $backendProc.Id -ErrorAction SilentlyContinue
} catch {}

Set-Location ..

Log "Backend and frontend stopped."
# leave .venv environment with 'deactivate'
