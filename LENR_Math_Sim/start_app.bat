@echo off
echo Starting LENR Simulation Framework...
echo ==================================

echo Starting backend API...
start "LENR Backend" cmd /k "cd backend && uvicorn main:app --reload --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak > NUL

echo Starting frontend...
start "LENR Frontend" cmd /k "cd frontend && npm start"

echo ==================================
echo Application running!
echo API: http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Close both command windows to stop the services
pause
