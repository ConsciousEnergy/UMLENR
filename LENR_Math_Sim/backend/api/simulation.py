"""API endpoints for LENR simulations."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
from datetime import datetime
import uuid
import asyncio
import logging

from api.models import (
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
    SimulationParameters,
    ValidationResult,
    ErrorResponse
)

from core.integrated_simulation import IntegratedLENRSimulation, IntegratedParameters

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/simulations", tags=["simulations"])

# In-memory storage (replace with database in production)
simulations_db: Dict[str, SimulationResult] = {}


def convert_params_to_integrated(params: SimulationParameters) -> IntegratedParameters:
    """Convert API parameters to IntegratedParameters."""
    integrated_params = IntegratedParameters()
    
    # Map fields
    integrated_params.material = params.material.value
    integrated_params.temperature = params.temperature
    integrated_params.loading_ratio = params.loading_ratio
    integrated_params.electric_field = params.electric_field
    integrated_params.surface_potential = params.surface_potential
    integrated_params.defect_density = params.defect_density
    integrated_params.coherence_domain_size = params.coherence_domain_size
    integrated_params.bubble_radius = params.bubble_radius
    integrated_params.driving_pressure = params.driving_pressure
    integrated_params.driving_frequency = params.driving_frequency
    
    return integrated_params


async def run_simulation_async(simulation_id: str, request: SimulationRequest):
    """Run simulation asynchronously."""
    try:
        # Update status
        simulations_db[simulation_id].status = SimulationStatus.RUNNING
        
        # Convert parameters
        integrated_params = convert_params_to_integrated(request.parameters)
        
        # Run simulation
        sim = IntegratedLENRSimulation(integrated_params)
        results = sim.calculate_total_enhancement(energy=request.energy)
        
        # Prepare output
        output = {
            "total_enhancement": results["total_combined_enhancement"],
            "tunneling_probability": results["final_reaction_probability"],
            "energy_concentration": results["total_energy_concentration"],
            "screening_energy": results["screening_energy"],
            "lattice_enhancement": results["lattice_enhancement"],
            "interface_enhancement": results["interface_enhancement"],
            "max_interface_field": results["max_interface_field"],
            "coherent_energy": results["coherent_energy"]
        }
        
        # Add reaction rate if requested
        if request.calculate_rate:
            rate_results = sim.reaction_rate_estimate(energy=request.energy)
            output["reaction_rate"] = rate_results["enhanced_rate"]
            output["power_density"] = rate_results["power_density"]
            output["total_power"] = rate_results["total_power"]
        
        # Add validation if requested
        if request.include_validation:
            validation = sim.validate_against_paper()
            output["validation"] = validation
        
        # Update simulation result
        simulations_db[simulation_id].results = output
        simulations_db[simulation_id].status = SimulationStatus.COMPLETED
        simulations_db[simulation_id].completed_at = datetime.now()
        
    except Exception as e:
        logger.error(f"Simulation {simulation_id} failed: {e}")
        simulations_db[simulation_id].status = SimulationStatus.FAILED
        simulations_db[simulation_id].error = str(e)
        simulations_db[simulation_id].completed_at = datetime.now()


@router.post("/", response_model=SimulationResult)
async def create_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
) -> SimulationResult:
    """
    Create and run a new LENR simulation.
    
    This endpoint creates a new simulation with the specified parameters
    and runs it asynchronously. The simulation ID is returned immediately,
    and the results can be retrieved later.
    """
    # Generate simulation ID
    simulation_id = f"sim_{uuid.uuid4().hex[:12]}"
    
    # Create simulation result
    result = SimulationResult(
        simulation_id=simulation_id,
        status=SimulationStatus.PENDING,
        parameters=request.parameters,
        energy=request.energy,
        created_at=datetime.now()
    )
    
    # Store in database
    simulations_db[simulation_id] = result
    
    # Run simulation in background
    background_tasks.add_task(run_simulation_async, simulation_id, request)
    
    return result


@router.get("/{simulation_id}", response_model=SimulationResult)
async def get_simulation(simulation_id: str) -> SimulationResult:
    """
    Get simulation results by ID.
    
    Retrieves the status and results of a previously created simulation.
    """
    if simulation_id not in simulations_db:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    return simulations_db[simulation_id]


@router.get("/", response_model=List[SimulationResult])
async def list_simulations(
    status: Optional[SimulationStatus] = None,
    limit: int = 100,
    offset: int = 0
) -> List[SimulationResult]:
    """
    List all simulations.
    
    Returns a list of all simulations, optionally filtered by status.
    """
    # Filter by status if provided
    results = list(simulations_db.values())
    if status:
        results = [r for r in results if r.status == status]
    
    # Apply pagination
    return results[offset:offset + limit]


@router.delete("/{simulation_id}")
async def delete_simulation(simulation_id: str) -> Dict[str, str]:
    """
    Delete a simulation.
    
    Removes a simulation from the database.
    """
    if simulation_id not in simulations_db:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    del simulations_db[simulation_id]
    return {"message": f"Simulation {simulation_id} deleted"}


@router.post("/{simulation_id}/validate", response_model=ValidationResult)
async def validate_simulation(simulation_id: str) -> ValidationResult:
    """
    Validate simulation results against paper predictions.
    
    Checks if the simulation results match the theoretical predictions
    from the LENR paper.
    """
    if simulation_id not in simulations_db:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim_result = simulations_db[simulation_id]
    if sim_result.status != SimulationStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Simulation not completed")
    
    if not sim_result.results:
        raise HTTPException(status_code=400, detail="No results available")
    
    results = sim_result.results
    
    # Perform validation checks
    validation = ValidationResult(
        enhancement_in_range=1e3 <= results.get("total_enhancement", 0) <= 1e5,
        screening_in_range=10 <= results.get("screening_energy", 0) <= 100,
        field_in_range=1e9 <= results.get("max_interface_field", 0) <= 1e11,
        energy_concentration_in_range=10 <= results.get("energy_concentration", 0) <= 100,
        all_checks_passed=False,
        details={
            "total_enhancement": results.get("total_enhancement", 0),
            "screening_energy": results.get("screening_energy", 0),
            "max_interface_field": results.get("max_interface_field", 0),
            "energy_concentration": results.get("energy_concentration", 0)
        }
    )
    
    # Check if all passed
    validation.all_checks_passed = (
        validation.enhancement_in_range and
        validation.screening_in_range and
        validation.field_in_range and
        validation.energy_concentration_in_range
    )
    
    return validation


@router.post("/batch", response_model=List[SimulationResult])
async def create_batch_simulations(
    requests: List[SimulationRequest],
    background_tasks: BackgroundTasks
) -> List[SimulationResult]:
    """
    Create multiple simulations in batch.
    
    Runs multiple simulations with different parameters simultaneously.
    """
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 simulations per batch")
    
    results = []
    for request in requests:
        # Generate simulation ID
        simulation_id = f"sim_{uuid.uuid4().hex[:12]}"
        
        # Create simulation result
        result = SimulationResult(
            simulation_id=simulation_id,
            status=SimulationStatus.PENDING,
            parameters=request.parameters,
            energy=request.energy,
            created_at=datetime.now()
        )
        
        # Store in database
        simulations_db[simulation_id] = result
        results.append(result)
        
        # Run simulation in background
        background_tasks.add_task(run_simulation_async, simulation_id, request)
    
    return results
