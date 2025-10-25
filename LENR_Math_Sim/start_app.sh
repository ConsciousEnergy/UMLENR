#!/bin/bash
# Start both backend and frontend for LENR Simulation Framework
set -euo pipefail

echo "Starting LENR Simulation Framework..."
echo "=================================="

# Start backend
echo "Starting backend API..."
cd backend || { echo "ERROR: Failed to cd into backend directory. Ensure the script is run from LENR_Math_Sim directory."; exit 1; }
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"
# Wait for backend to be ready
for i in {1..30}; do
  if nc -z localhost 8000 2>/dev/null || curl -s http://localhost:8000/docs > /dev/null 2>&1; then
    echo "Backend is ready"
    break
  fi
  echo "Waiting for backend... ($i/30)"
  sleep 1
done
sleep 3

# Start frontend
echo "Starting frontend..."
cd ../frontend || { echo "ERROR: Failed to cd into frontend directory. Directory may not exist."; exit 1; }
npm start &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

echo "=================================="
echo "Application running!"
echo "API: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for interrupt
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
