🚀 LENR Simulation Framework - Setup Instructions

## Prerequisites

- Python 3.12 or later
- Node.js 14+ and npm
- Git

## Quick Setup

### 1. Backend Setup
```bash
cd LENR_Math_Sim/backend
pip install -r requirements.txt
```

**Note**: If you encounter package compatibility issues, some optional packages can be installed separately or skipped (like fenics, mpi4py).

### 2. Frontend Setup
```bash
cd LENR_Math_Sim/frontend
npm install
# or: yarn install
```

### 3. Start Everything

**Windows (Command Prompt or PowerShell):**
```bash
# From LENR_Math_Sim/LENR_Math_Sim directory
start_app.bat
```

**Windows (Git Bash) / Linux / Mac:**
```bash
# From LENR_Math_Sim/LENR_Math_Sim directory
chmod +x start_app.sh
./start_app.sh
```

**Important**: If using Git Bash on Windows, use `./start_app.sh` instead of `start_app.bat`

**Manual Start (Recommended for Git Bash users):**
```bash
# Terminal 1 - Backend
cd LENR_Math_Sim/backend
uvicorn main:app --reload

# Terminal 2 - Frontend  
cd LENR_Math_Sim/frontend
npm start
```

## Access Points

- **Frontend UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **API Base**: http://localhost:8000

## Features Available

### Frontend (React + Three.js)
- Interactive parameter controls
- 3D visualization of enhancement fields
- Real-time results display
- Chart visualizations
- Preset configurations

### Backend (FastAPI)
- REST API endpoints
- Simulation execution
- Parameter optimization
- ML-powered suggestions
- WebSocket support

## Test the System

```bash
# Run comprehensive tests
python run_full_test.py

# Test API
python test_api.py

# Demo simulation
python demo_simulation.py
```

## Troubleshooting

1. **Port already in use**: Change ports in:
   - Backend: `uvicorn main:app --port 8001`
   - Frontend: `PORT=3001 npm start`

2. **Module not found**: Ensure you're in the correct directory
   - Backend modules need: `cd backend`
   - Tests run from: `LENR_Math_Sim/`

3. **CORS issues**: Backend is configured for localhost:3000
   - Update `main.py` if using different ports

## Environment Variables

Create `.env` files:

**backend/.env:**
```
API_HOST=0.0.0.0
API_PORT=8000
```

**frontend/.env:**
```
REACT_APP_API_URL=http://localhost:8000
```

## Docker (Optional)

```bash
docker-compose up
```

## Ready to Use!

The system is fully operational with:
- ✅ All physics modules
- ✅ API endpoints  
- ✅ ML optimization
- ✅ 3D visualization
- ✅ 100% tests passing

Start exploring LENR simulations! 🎉
