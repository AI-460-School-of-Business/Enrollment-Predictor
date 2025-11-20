#!/bin/bash
# Startup script to ensure proper initialization order

echo "Container starting up..."

# Set up signal handlers for graceful shutdown
trap 'echo "Received shutdown signal, exporting data..."; python /app/server/database/export_data.py; exit 0' SIGTERM SIGINT

# Check if this is the db-init service (check for db-init.py in server/database)
if [ -f "/app/server/database/db-init.py" ]; then
    echo "Running database initialization..."
    python /app/server/database/db-init.py
else
    echo "Running web application..."
    python app.py &
    
    # Wait for the background process
    wait $!
fi