#!/usr/bin/env bash
set -e

BACKEND_DIR="backend"
FRONTEND_DIR="frontend"
BACKEND_APP="app/app.py"
PYTHON_BIN="python"   # or "python3" 

# Helper
log() {
  echo ""
  echo "===> $1"
  echo ""
}

# Setup & run Flask backend
log "Setting up and starting Flask backend"

cd "$BACKEND_DIR"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
  log "Creating virtual environment..."
  $PYTHON_BIN -m venv .venv
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Install dependencies
if [ -f "requirements.txt" ]; then
  log "Installing backend dependencies..."
  pip install -r requirements.txt
fi

# Run Flask backend in background
log "Starting Flask API..."
$PYTHON_BIN "$BACKEND_APP" &
BACKEND_PID=$!

cd ..

# Setup & run frontend
log "Setting up and starting React/Vite frontend..."

cd "$FRONTEND_DIR"

if [ -f "package.json" ]; then
  log "Installing frontend dependencies (npm install)..."
  npm install
fi

log "Starting Vite dev server (React frontend)..."
npm run dev

# When Vite stops (Ctrl+C), kill backend
log "Stopping Flask backend (PID: $BACKEND_PID)..."
kill "$BACKEND_PID" || true

cd ..

log "Backend & frontend stopped."
