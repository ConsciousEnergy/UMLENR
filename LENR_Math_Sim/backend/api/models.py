"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
from enum import Enum


class SimulationStatus(str, Enum):
    """Simulation status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MaterialType(str, Enum):
    """Material type enumeration."""
    PD = "Pd"
    NI = "Ni"
    TI = "Ti"


class SimulationParameters(BaseModel):
    """Parameters for LENR simulation."""
    
    material: MaterialType = Field(default=MaterialType.PD, description="Material type")
    temperature: float = Field(default=300.0, ge=200.0, le=500.0, description="Temperature in Kelvin")
    loading_ratio: float = Field(default=0.9, ge=0.0, le=1.0, description="D/Pd or H/Ni loading ratio")
    electric_field: float = Field(default=1e9, ge=1e6, le=1e12, description="Electric field in V/m")
    surface_potential: float = Field(default=0.5, ge=0.0, le=5.0, description="Surface potential in V")
    defect_density: float = Field(default=1e20, ge=1e18, le=1e23, description="Defect density in defects/m³")
    coherence_domain_size: int = Field(default=1000, ge=10, le=100000, description="Coherence domain size in atoms")
    bubble_radius: float = Field(default=10e-6, ge=1e-9, le=1e-3, description="Bubble radius in meters")
    driving_pressure: float = Field(default=1.5e5, ge=0, le=1e7, description="Driving pressure in Pa")
    driving_frequency: float = Field(default=20e3, ge=0, le=1e6, description="Driving frequency in Hz")
    
    class Config:
        schema_extra = {
            "example": {
                "material": "Pd",
                "temperature": 300.0,
                "loading_ratio": 0.95,
                "electric_field": 1e10,
                "surface_potential": 0.5,
                "defect_density": 1e21,
                "coherence_domain_size": 1000
            }
        }


class SimulationRequest(BaseModel):
    """Request model for running a simulation."""
    
    parameters: SimulationParameters
    energy: float = Field(default=10.0, ge=0.1, le=10000.0, description="Incident energy in eV")
    calculate_rate: bool = Field(default=True, description="Calculate reaction rate")
    include_validation: bool = Field(default=True, description="Include validation against paper")
    
    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "material": "Pd",
                    "temperature": 300.0,
                    "loading_ratio": 0.95
                },
                "energy": 10.0,
                "calculate_rate": True
            }
        }


class SimulationResult(BaseModel):
    """Result model for simulation output."""
    
    simulation_id: str
    status: SimulationStatus
    parameters: SimulationParameters
    energy: float
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "simulation_id": "sim_123456",
                "status": "completed",
                "parameters": {"material": "Pd", "temperature": 300.0},
                "energy": 10.0,
                "results": {
                    "total_enhancement": 1.5e7,
                    "tunneling_probability": 3.2e8,
                    "energy_concentration": 42.5
                }
            }
        }


class MonteCarloRequest(BaseModel):
    """Request model for Monte Carlo analysis."""
    
    n_samples: int = Field(default=1000, ge=10, le=100000, description="Number of MC samples")
    n_bootstrap: int = Field(default=1000, ge=100, le=10000, description="Bootstrap samples for CI")
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.999, description="Confidence level")
    parallel: bool = Field(default=True, description="Use parallel processing")
    parameter_ranges: Optional[Dict[str, Dict[str, float]]] = None
    output_quantities: Optional[List[str]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "n_samples": 1000,
                "n_bootstrap": 1000,
                "confidence_level": 0.95,
                "parallel": True,
                "output_quantities": ["total_enhancement", "energy_concentration"]
            }
        }


class MonteCarloResult(BaseModel):
    """Result model for Monte Carlo analysis."""
    
    analysis_id: str
    status: SimulationStatus
    n_samples: int
    n_successful: int
    n_failed: int
    statistics: Optional[Dict[str, Dict[str, float]]] = None
    confidence_intervals: Optional[Dict[str, Dict[str, Any]]] = None
    convergence: Optional[Dict[str, Dict[str, Any]]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class ParameterScanRequest(BaseModel):
    """Request model for parameter scanning."""
    
    parameter_name: str = Field(description="Name of parameter to scan")
    values: List[float] = Field(description="Values to test")
    base_parameters: Optional[SimulationParameters] = None
    energy: float = Field(default=10.0, description="Energy for calculations")
    
    class Config:
        schema_extra = {
            "example": {
                "parameter_name": "loading_ratio",
                "values": [0.8, 0.85, 0.9, 0.95, 0.99],
                "energy": 10.0
            }
        }


class ParameterScanResult(BaseModel):
    """Result model for parameter scan."""
    
    scan_id: str
    parameter_name: str
    parameter_values: List[float]
    total_enhancement: List[float]
    tunneling_probability: List[float]
    energy_concentration: List[float]
    created_at: datetime


class PoissonSchrodingerRequest(BaseModel):
    """Request for Poisson-Schrödinger calculation."""
    
    grid_size: int = Field(default=1000, ge=100, le=10000, description="Number of grid points")
    grid_min: float = Field(default=-10e-9, description="Minimum position in meters")
    grid_max: float = Field(default=10e-9, description="Maximum position in meters")
    temperature: float = Field(default=300.0, description="Temperature in K")
    phi_left: float = Field(default=0.0, description="Left boundary potential in V")
    phi_right: float = Field(default=0.5, description="Right boundary potential in V")
    n_electrons: int = Field(default=10, ge=1, le=100, description="Number of electrons")
    n_states: int = Field(default=5, ge=1, le=20, description="Number of states to compute")


class PoissonSchrodingerResult(BaseModel):
    """Result for Poisson-Schrödinger calculation."""
    
    calculation_id: str
    converged: bool
    iterations: int
    energies: List[float]  # in eV
    max_field: float  # V/m
    max_potential: float  # V


class ValidationResult(BaseModel):
    """Validation result against paper predictions."""
    
    enhancement_in_range: bool
    screening_in_range: bool
    field_in_range: bool
    energy_concentration_in_range: bool
    all_checks_passed: bool
    details: Dict[str, Any]


class HealthCheck(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    physics_modules: Dict[str, bool]
    solvers: Dict[str, bool]
    timestamp: datetime


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    
    type: str  # "status", "progress", "result", "error"
    simulation_id: Optional[str] = None
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str
    detail: Optional[str] = None
    status_code: int
