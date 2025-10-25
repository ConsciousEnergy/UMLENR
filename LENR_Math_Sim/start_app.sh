#!/bin/bash
# Start both backend and frontend for LENR Simulation Framework

echo "Starting LENR Simulation Framework..."
echo "=================================="

# Start backend
echo "Starting backend API..."
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait for backend to be ready
sleep 3

# Start frontend
echo "Starting frontend..."
cd ../frontend
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
