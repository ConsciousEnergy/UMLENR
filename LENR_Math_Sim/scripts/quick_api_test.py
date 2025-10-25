#!/usr/bin/env python
"""Quick API functionality test."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

# Test API imports
try:
    from main import app  # type: ignore
    from api.models import SimulationParameters, SimulationRequest  # type: ignore
    from api.simulation import router as sim_router  # type: ignore
    from api.parameters import router as param_router  # type: ignore
    from api.ml_endpoints import router as ml_router  # type: ignore
    
    print("[OK] All API modules imported successfully")
    
    # Test creating a simulation request
    params = SimulationParameters(
        material="Pd",
        temperature=300.0,
        loading_ratio=0.95,
        electric_field=1e10
    )
    
    request = SimulationRequest(
        parameters=params,
        energy=10.0,
        calculate_rate=True
    )
    
    print(f"[OK] Created simulation request for {params.material} at {params.temperature}K")
    print(f"[OK] Loading ratio: {params.loading_ratio}")
    print(f"[OK] Electric field: {params.electric_field:.1e} V/m")
    
    # Check routers
    print(f"\n[OK] Simulation router has {len(sim_router.routes)} endpoints")
    print(f"[OK] Parameters router has {len(param_router.routes)} endpoints")  
    print(f"[OK] ML router has {len(ml_router.routes)} endpoints")
    
    print("\n[SUCCESS] API is ready for use!")
    print("\nTo start the API server, run:")
    print("  cd backend")
    print("  uvicorn main:app --reload")
    print("\nThen access the API at:")
    print("  http://localhost:8000/docs")
    
except Exception as e:
    print(f"[FAIL] API test failed: {e}")
    sys.exit(1)
