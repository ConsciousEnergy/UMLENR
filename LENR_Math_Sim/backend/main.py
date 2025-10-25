"""Main FastAPI application for LENR Mathematical Simulations."""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events."""
    # Startup
    logger.info("Starting LENR Simulation Server...")
    # Initialize database connections, cache, etc.
    yield
    # Shutdown
    logger.info("Shutting down LENR Simulation Server...")
    # Clean up resources

# Create FastAPI application
app = FastAPI(
    title="LENR Mathematical Simulation API",
    description="API for Low-Energy Nuclear Reactions mathematical simulations based on unified theoretical framework",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LENR Mathematical Simulation API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "simulations": "/api/v1/simulations",
            "parameters": "/api/v1/parameters",
            "results": "/api/v1/results",
            "websocket": "/ws"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    # Check if physics modules are importable
    physics_modules = {}
    try:
        from core.quantum_tunneling import QuantumTunneling
        physics_modules["quantum_tunneling"] = True
    except:
        physics_modules["quantum_tunneling"] = False
    
    try:
        from core.electron_screening import ElectronScreening
        physics_modules["electron_screening"] = True
    except:
        physics_modules["electron_screening"] = False
    
    try:
        from core.lattice_effects import LatticeEffects
        physics_modules["lattice_effects"] = True
    except:
        physics_modules["lattice_effects"] = False
    
    try:
        from core.interface_dynamics import InterfaceDynamics
        physics_modules["interface_dynamics"] = True
    except:
        physics_modules["interface_dynamics"] = False
    
    try:
        from core.bubble_dynamics import BubbleDynamics
        physics_modules["bubble_dynamics"] = True
    except:
        physics_modules["bubble_dynamics"] = False
    
    # Check solvers
    solvers = {}
    try:
        from solvers.poisson_schrodinger import PoissonSchrodingerSolver
        solvers["poisson_schrodinger"] = True
    except:
        solvers["poisson_schrodinger"] = False
    
    try:
        from solvers.monte_carlo import MonteCarloUncertainty
        solvers["monte_carlo"] = True
    except:
        solvers["monte_carlo"] = False
    
    all_healthy = all(physics_modules.values()) and all(solvers.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "version": "1.0.0",
        "service": "lenr-simulation-api",
        "physics_modules": physics_modules,
        "solvers": solvers
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time simulation updates."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Process websocket messages
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# API v1 endpoints
from api import simulation, parameters, ml_endpoints
app.include_router(simulation.router)
app.include_router(parameters.router)
app.include_router(ml_endpoints.router)

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors."""
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
