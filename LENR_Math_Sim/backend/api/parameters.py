"""API endpoints for parameter management and scanning."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any
from datetime import datetime
import uuid
import logging

from api.models import (
    ParameterScanRequest,
    ParameterScanResult,
    SimulationParameters
)

from core.integrated_simulation import IntegratedLENRSimulation, IntegratedParameters

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/parameters", tags=["parameters"])

# In-memory storage for parameter scans
scans_db: Dict[str, ParameterScanResult] = {}


@router.get("/defaults", response_model=SimulationParameters)
async def get_default_parameters() -> SimulationParameters:
    """
    Get default simulation parameters.
    
    Returns the default parameter values used in simulations.
    """
    return SimulationParameters()


@router.get("/ranges")
async def get_parameter_ranges() -> Dict[str, Dict[str, Any]]:
    """
    Get valid parameter ranges.
    
    Returns the minimum, maximum, and recommended values for each parameter.
    """
    return {
        "temperature": {
            "min": 200.0,
            "max": 500.0,
            "default": 300.0,
            "unit": "K",
            "description": "System temperature"
        },
        "loading_ratio": {
            "min": 0.0,
            "max": 1.0,
            "default": 0.9,
            "critical": 0.85,
            "unit": "dimensionless",
            "description": "D/Pd or H/Ni loading ratio"
        },
        "electric_field": {
            "min": 1e6,
            "max": 1e12,
            "default": 1e9,
            "unit": "V/m",
            "description": "Applied electric field"
        },
        "surface_potential": {
            "min": 0.0,
            "max": 5.0,
            "default": 0.5,
            "unit": "V",
            "description": "Surface potential"
        },
        "defect_density": {
            "min": 1e18,
            "max": 1e23,
            "default": 1e20,
            "unit": "defects/m³",
            "description": "Defect density in material"
        },
        "coherence_domain_size": {
            "min": 10,
            "max": 100000,
            "default": 1000,
            "unit": "atoms",
            "description": "Size of coherent domain"
        },
        "bubble_radius": {
            "min": 1e-9,
            "max": 1e-3,
            "default": 10e-6,
            "unit": "m",
            "description": "Initial bubble radius"
        },
        "driving_pressure": {
            "min": 0,
            "max": 1e7,
            "default": 1.5e5,
            "unit": "Pa",
            "description": "Acoustic driving pressure"
        },
        "driving_frequency": {
            "min": 0,
            "max": 1e6,
            "default": 20e3,
            "unit": "Hz",
            "description": "Acoustic driving frequency"
        }
    }


@router.post("/scan", response_model=ParameterScanResult)
async def parameter_scan(request: ParameterScanRequest) -> ParameterScanResult:
    """
    Perform a parameter scan.
    
    Scans a single parameter through a range of values while keeping
    other parameters fixed, and returns the enhancement factors.
    """
    # Validate parameter name
    valid_params = [
        "temperature", "loading_ratio", "electric_field",
        "surface_potential", "defect_density", "coherence_domain_size",
        "bubble_radius", "driving_pressure", "driving_frequency"
    ]
    
    if request.parameter_name not in valid_params:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameter: {request.parameter_name}"
        )
    
    # Use base parameters or defaults
    base_params = request.base_parameters or SimulationParameters()
    
    # Convert to IntegratedParameters
    integrated_params = IntegratedParameters()
    integrated_params.material = base_params.material.value
    integrated_params.temperature = base_params.temperature
    integrated_params.loading_ratio = base_params.loading_ratio
    integrated_params.electric_field = base_params.electric_field
    integrated_params.surface_potential = base_params.surface_potential
    integrated_params.defect_density = base_params.defect_density
    integrated_params.coherence_domain_size = base_params.coherence_domain_size
    
    # Run scan
    results_enhancement = []
    results_probability = []
    results_energy = []
    
    for value in request.values:
        # Set parameter value
        setattr(integrated_params, request.parameter_name, value)
        
        # Run simulation
        try:
            sim = IntegratedLENRSimulation(integrated_params)
            results = sim.calculate_total_enhancement(energy=request.energy)
            
            results_enhancement.append(results["total_combined_enhancement"])
            results_probability.append(results["final_reaction_probability"])
            results_energy.append(results["total_energy_concentration"])
        except Exception as e:
            logger.warning(f"Scan failed for {request.parameter_name}={value}: {e}")
            results_enhancement.append(0.0)
            results_probability.append(0.0)
            results_energy.append(0.0)
    
    # Create result
    scan_id = f"scan_{uuid.uuid4().hex[:12]}"
    result = ParameterScanResult(
        scan_id=scan_id,
        parameter_name=request.parameter_name,
        parameter_values=request.values,
        total_enhancement=results_enhancement,
        tunneling_probability=results_probability,
        energy_concentration=results_energy,
        created_at=datetime.now()
    )
    
    # Store result
    scans_db[scan_id] = result
    
    return result


@router.get("/scan/{scan_id}", response_model=ParameterScanResult)
async def get_scan_result(scan_id: str) -> ParameterScanResult:
    """
    Get parameter scan results by ID.
    
    Retrieves the results of a previously performed parameter scan.
    """
    if scan_id not in scans_db:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    return scans_db[scan_id]


@router.get("/optimal")
async def get_optimal_parameters() -> Dict[str, Any]:
    """
    Get optimal parameter combinations.
    
    Returns parameter combinations that yield the highest enhancement factors
    based on theoretical predictions and previous simulations.
    """
    return {
        "high_enhancement": {
            "description": "Parameters for maximum enhancement",
            "parameters": {
                "material": "Pd",
                "temperature": 300.0,
                "loading_ratio": 0.95,
                "electric_field": 1e10,
                "surface_potential": 0.5,
                "defect_density": 1e21,
                "coherence_domain_size": 1000,
                "bubble_radius": 10e-6,
                "driving_pressure": 2e5,
                "driving_frequency": 20e3
            },
            "expected_enhancement": "10^7 - 10^8",
            "notes": "Optimized for maximum tunneling enhancement"
        },
        "stable_operation": {
            "description": "Parameters for stable, reproducible results",
            "parameters": {
                "material": "Pd",
                "temperature": 300.0,
                "loading_ratio": 0.90,
                "electric_field": 5e9,
                "surface_potential": 0.4,
                "defect_density": 5e20,
                "coherence_domain_size": 500,
                "bubble_radius": 5e-6,
                "driving_pressure": 1e5,
                "driving_frequency": 25e3
            },
            "expected_enhancement": "10^5 - 10^6",
            "notes": "More conservative parameters for reproducibility"
        },
        "nickel_hydrogen": {
            "description": "Optimized for Ni-H systems",
            "parameters": {
                "material": "Ni",
                "temperature": 350.0,
                "loading_ratio": 0.8,
                "electric_field": 2e10,
                "surface_potential": 0.6,
                "defect_density": 2e21,
                "coherence_domain_size": 2000,
                "bubble_radius": 20e-6,
                "driving_pressure": 3e5,
                "driving_frequency": 15e3
            },
            "expected_enhancement": "10^6 - 10^7",
            "notes": "Adapted for nickel-hydrogen reactions"
        }
    }


@router.post("/validate")
async def validate_parameters(params: SimulationParameters) -> Dict[str, Any]:
    """
    Validate a set of parameters.
    
    Checks if the provided parameters are within valid ranges and
    likely to produce meaningful results.
    """
    issues = []
    warnings = []
    
    # Check critical loading ratio
    if params.loading_ratio < 0.85:
        warnings.append("Loading ratio below critical threshold (0.85)")
    
    # Check field strength
    if params.electric_field < 1e8:
        warnings.append("Electric field may be too low for significant enhancement")
    elif params.electric_field > 1e11:
        warnings.append("Electric field may cause material breakdown")
    
    # Check defect density
    if params.defect_density < 1e19:
        warnings.append("Low defect density may reduce reaction sites")
    
    # Check coherence domain
    if params.coherence_domain_size < 100:
        warnings.append("Small coherence domain may limit collective effects")
    
    # Check bubble parameters
    if params.driving_pressure > 0 and params.driving_frequency == 0:
        issues.append("Driving pressure specified but frequency is zero")
    
    # Temperature check
    if params.temperature < 250:
        warnings.append("Low temperature may reduce thermal activation")
    elif params.temperature > 400:
        warnings.append("High temperature may destabilize loading")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "parameter_quality": {
            "loading": "good" if params.loading_ratio >= 0.9 else "fair" if params.loading_ratio >= 0.85 else "poor",
            "field": "good" if 1e9 <= params.electric_field <= 1e10 else "fair",
            "defects": "good" if params.defect_density >= 1e20 else "fair"
        }
    }
