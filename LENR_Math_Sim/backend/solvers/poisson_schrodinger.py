"""Coupled Poisson-Schrödinger solver for LENR simulations.

This module implements the coupled PDE solver described in Section 3.1
of the theoretical framework, solving for electrostatic potentials and
quantum wavefunctions near metal-hydrogen interfaces.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass
import logging
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy import linalg
from scipy.interpolate import interp1d

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    HBAR, E_CHARGE, M_ELECTRON, M_DEUTERON, EPSILON_0,
    EV_TO_JOULE, JOULE_TO_EV, ANGSTROM, NANOMETER,
    PD_WORK_FUNCTION, PD_FERMI_ENERGY
)

logger = logging.getLogger(__name__)


@dataclass
class SolverParameters:
    """Parameters for Poisson-Schrödinger solver."""
    
    # Grid parameters
    grid_size: int  # Number of grid points
    grid_min: float  # Minimum position (m)
    grid_max: float  # Maximum position (m)
    
    # Physical parameters
    temperature: float  # Temperature (K)
    particle_mass: float  # Particle mass (kg)
    dielectric_constant: float  # Relative dielectric constant
    
    # Boundary conditions
    phi_left: float  # Potential at left boundary (V)
    phi_right: float  # Potential at right boundary (V)
    psi_left: complex  # Wavefunction at left boundary
    psi_right: complex  # Wavefunction at right boundary
    
    # Numerical parameters
    max_iterations: int  # Maximum self-consistent iterations
    convergence_tolerance: float  # Convergence criterion
    mixing_parameter: float  # Mixing for self-consistency (0-1)
    adaptive_mesh: bool  # Use adaptive mesh refinement
    
    # Time evolution (optional)
    time_dependent: bool  # Solve time-dependent equations
    time_step: float  # Time step (s)
    total_time: float  # Total simulation time (s)


class PoissonSchrodingerSolver:
    """Coupled Poisson-Schrödinger solver for interface calculations."""
    
    def __init__(self, parameters: Optional[SolverParameters] = None):
        """Initialize solver with parameters."""
        self.params = parameters or self._default_parameters()
        self._setup_grid()
        self._setup_operators()
        
    @staticmethod
    def _default_parameters() -> SolverParameters:
        """Get default solver parameters for 1D interface."""
        return SolverParameters(
            grid_size=1000,
            grid_min=-10e-9,  # -10 nm
            grid_max=10e-9,   # +10 nm
            temperature=300.0,
            particle_mass=M_ELECTRON,
            dielectric_constant=80.0,  # Water
            phi_left=0.0,
            phi_right=0.5,
            psi_left=0.0,
            psi_right=0.0,
            max_iterations=100,
            convergence_tolerance=1e-6,
            mixing_parameter=0.3,
            adaptive_mesh=False,
            time_dependent=False,
            time_step=1e-18,  # 1 attosecond
            total_time=1e-15  # 1 femtosecond
        )
    
    def _setup_grid(self):
        """Set up spatial grid with optional adaptive refinement."""
        if self.params.adaptive_mesh:
            # Create adaptive grid with higher resolution near interface
            interface_pos = 0.0
            interface_width = 1e-9  # 1 nm
            
            # Create three regions: left, interface, right
            n_interface = self.params.grid_size // 2
            n_bulk = self.params.grid_size // 4
            
            # High resolution near interface
            x_interface = np.linspace(
                interface_pos - interface_width,
                interface_pos + interface_width,
                n_interface
            )
            
            # Lower resolution in bulk
            x_left = np.linspace(
                self.params.grid_min,
                interface_pos - interface_width,
                n_bulk,
                endpoint=False
            )
            x_right = np.linspace(
                interface_pos + interface_width,
                self.params.grid_max,
                n_bulk,
                endpoint=False
            )[1:]
            
            self.x = np.concatenate([x_left, x_interface, x_right])
            self.x = np.sort(np.unique(self.x))
            self.N = len(self.x)
        else:
            # Uniform grid
            self.x = np.linspace(
                self.params.grid_min,
                self.params.grid_max,
                self.params.grid_size
            )
            self.N = self.params.grid_size
        
        # Calculate grid spacing
        self.dx = np.diff(self.x)
        self.dx_mean = np.mean(self.dx)
    
    def _setup_operators(self):
        """Set up differential operators using finite differences."""
        # Laplacian operator for Poisson equation
        self.laplacian_poisson = self._create_laplacian_matrix()
        
        # Hamiltonian operator for Schrödinger equation
        self.kinetic_operator = self._create_kinetic_matrix()
    
    def _create_laplacian_matrix(self) -> sparse.csr_matrix:
        """Create sparse matrix for Laplacian operator."""
        # Second-order finite difference for non-uniform grid
        diagonals = []
        offsets = []
        
        # Main diagonal
        main_diag = np.zeros(self.N)
        for i in range(1, self.N - 1):
            dx_left = self.x[i] - self.x[i-1]
            dx_right = self.x[i+1] - self.x[i]
            main_diag[i] = -2.0 / (dx_left * dx_right)
        
        # Boundary conditions
        main_diag[0] = 1.0
        main_diag[-1] = 1.0
        
        diagonals.append(main_diag)
        offsets.append(0)
        
        # Upper diagonal
        upper_diag = np.zeros(self.N - 1)
        for i in range(1, self.N - 1):
            dx_right = self.x[i+1] - self.x[i]
            dx_avg = (self.x[i+1] - self.x[i-1]) / 2.0
            upper_diag[i] = 1.0 / (dx_right * dx_avg)
        
        diagonals.append(upper_diag)
        offsets.append(1)
        
        # Lower diagonal
        lower_diag = np.zeros(self.N - 1)
        for i in range(0, self.N - 2):
            dx_left = self.x[i+1] - self.x[i]
            dx_avg = (self.x[i+2] - self.x[i]) / 2.0 if i < self.N - 2 else dx_left
            lower_diag[i] = 1.0 / (dx_left * dx_avg)
        
        diagonals.append(lower_diag)
        offsets.append(-1)
        
        return sparse.diags(diagonals, offsets, shape=(self.N, self.N), format='csr')
    
    def _create_kinetic_matrix(self) -> sparse.csr_matrix:
        """Create kinetic energy operator matrix."""
        # -ℏ²/(2m) ∇²
        prefactor = -HBAR**2 / (2 * self.params.particle_mass)
        return prefactor * self.laplacian_poisson
    
    def solve_poisson(self, charge_density: np.ndarray) -> np.ndarray:
        """Solve Poisson equation for given charge density.
        
        ∇²φ = -ρ/ε
        
        Args:
            charge_density: Charge density at grid points (C/m³)
            
        Returns:
            Electrostatic potential φ (V)
        """
        # Right-hand side
        eps = self.params.dielectric_constant * EPSILON_0
        rhs = -charge_density / eps
        
        # Apply boundary conditions
        rhs[0] = self.params.phi_left
        rhs[-1] = self.params.phi_right
        
        # Solve linear system
        phi = sparse_linalg.spsolve(self.laplacian_poisson, rhs)
        
        return phi
    
    def solve_schrodinger(self, potential: np.ndarray, 
                         n_states: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Solve time-independent Schrödinger equation.
        
        Hψ = Eψ where H = -ℏ²/(2m)∇² + V
        
        Args:
            potential: Potential energy at grid points (J)
            n_states: Number of eigenstates to compute
            
        Returns:
            Tuple of (energies, wavefunctions)
        """
        # Build Hamiltonian matrix
        V_matrix = sparse.diags(potential, 0, shape=(self.N, self.N))
        hamiltonian = self.kinetic_operator + V_matrix
        
        # Convert to dense for eigenvalue solver (for small systems)
        if self.N < 2000:
            H_dense = hamiltonian.toarray()
            
            # Apply boundary conditions (zero at boundaries)
            H_dense[0, :] = 0
            H_dense[0, 0] = 1e10  # Large energy penalty
            H_dense[-1, :] = 0
            H_dense[-1, -1] = 1e10
            
            # Solve eigenvalue problem
            energies, wavefunctions = linalg.eigh(H_dense)
            
            # Select lowest n_states
            energies = energies[:n_states]
            wavefunctions = wavefunctions[:, :n_states]
        else:
            # Use sparse solver for large systems
            energies, wavefunctions = sparse_linalg.eigsh(
                hamiltonian, k=n_states, which='SA'
            )
        
        # Normalize wavefunctions
        for i in range(n_states):
            norm = np.trapz(np.abs(wavefunctions[:, i])**2, self.x)
            wavefunctions[:, i] /= np.sqrt(norm)
        
        return energies, wavefunctions
    
    def calculate_charge_density(self, wavefunctions: np.ndarray, 
                                occupations: np.ndarray,
                                background_charge: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate charge density from wavefunctions.
        
        Args:
            wavefunctions: Electronic wavefunctions
            occupations: Occupation numbers for each state
            background_charge: Background ionic charge density
            
        Returns:
            Total charge density (C/m³)
        """
        # Electronic charge density
        rho_electron = np.zeros(self.N)
        
        n_states = wavefunctions.shape[1]
        for i in range(n_states):
            rho_electron += occupations[i] * np.abs(wavefunctions[:, i])**2
        
        # Convert to charge density
        rho_electron *= -E_CHARGE
        
        # Add background charge if provided
        if background_charge is not None:
            rho_total = rho_electron + background_charge
        else:
            rho_total = rho_electron
        
        return rho_total
    
    def solve_self_consistent(self, initial_potential: Optional[np.ndarray] = None,
                            n_electrons: int = 10) -> Dict:
        """Solve coupled Poisson-Schrödinger equations self-consistently.
        
        Args:
            initial_potential: Initial guess for potential (V)
            n_electrons: Number of electrons in system
            
        Returns:
            Dictionary with results
        """
        # Initialize potential
        if initial_potential is None:
            # Linear potential as initial guess
            phi = np.linspace(self.params.phi_left, self.params.phi_right, self.N)
        else:
            phi = initial_potential.copy()
        
        # Convert to energy units
        V = phi * E_CHARGE  # Joules
        
        # Self-consistent loop
        converged = False
        iteration = 0
        
        while not converged and iteration < self.params.max_iterations:
            # Store old potential
            V_old = V.copy()
            
            # Solve Schrödinger equation
            energies, wavefunctions = self.solve_schrodinger(V, n_states=n_electrons)
            
            # Calculate occupations (simple filling at T=0)
            occupations = np.ones(min(n_electrons, len(energies)))
            if len(occupations) < n_electrons:
                occupations[-1] = n_electrons - len(occupations) + 1
            
            # Calculate charge density
            rho = self.calculate_charge_density(wavefunctions, occupations)
            
            # Solve Poisson equation
            phi_new = self.solve_poisson(rho)
            V_new = phi_new * E_CHARGE
            
            # Mix old and new potentials
            alpha = self.params.mixing_parameter
            V = alpha * V_new + (1 - alpha) * V_old
            
            # Check convergence
            error = np.max(np.abs(V - V_old))
            converged = error < self.params.convergence_tolerance * E_CHARGE
            
            iteration += 1
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: error = {error/E_CHARGE:.2e} eV")
        
        if converged:
            logger.info(f"Converged after {iteration} iterations")
        else:
            logger.warning(f"Did not converge after {iteration} iterations")
        
        # Calculate final quantities
        electric_field = -np.gradient(phi, self.x)
        
        return {
            "converged": converged,
            "iterations": iteration,
            "potential": phi,
            "electric_field": electric_field,
            "charge_density": rho,
            "energies": energies * JOULE_TO_EV,  # Convert to eV
            "wavefunctions": wavefunctions,
            "x_grid": self.x
        }
    
    def solve_time_dependent(self, initial_psi: np.ndarray,
                            potential_func: Callable,
                            output_times: Optional[np.ndarray] = None) -> Dict:
        """Solve time-dependent Schrödinger equation.
        
        iℏ ∂ψ/∂t = Hψ
        
        Args:
            initial_psi: Initial wavefunction
            potential_func: Function V(x, t) returning potential
            output_times: Times at which to save wavefunction
            
        Returns:
            Dictionary with time evolution results
        """
        if not self.params.time_dependent:
            raise ValueError("Solver not configured for time-dependent problems")
        
        # Time grid
        dt = self.params.time_step
        n_steps = int(self.params.total_time / dt)
        
        if output_times is None:
            output_times = np.linspace(0, self.params.total_time, 11)
        
        # Initialize wavefunction
        psi = initial_psi.copy()
        psi_history = []
        t_history = []
        
        # Time evolution using Crank-Nicolson method
        for step in range(n_steps):
            t = step * dt
            
            # Get potential at current time
            V = potential_func(self.x, t)
            
            # Build Hamiltonian
            H = self.kinetic_operator + sparse.diags(V, 0)
            
            # Crank-Nicolson: (1 + iHdt/2ℏ)ψ(t+dt) = (1 - iHdt/2ℏ)ψ(t)
            factor = 1j * dt / (2 * HBAR)
            A = sparse.identity(self.N) + factor * H
            B = sparse.identity(self.N) - factor * H
            
            # Evolve one time step
            psi_new = sparse_linalg.spsolve(A, B @ psi)
            
            # Normalize
            norm = np.trapz(np.abs(psi_new)**2, self.x)
            psi_new /= np.sqrt(norm)
            
            psi = psi_new
            
            # Save if at output time
            if any(np.abs(output_times - t) < dt/2):
                psi_history.append(psi.copy())
                t_history.append(t)
        
        return {
            "times": np.array(t_history),
            "wavefunctions": np.array(psi_history),
            "x_grid": self.x
        }
    
    def calculate_tunneling_current(self, barrier_potential: np.ndarray,
                                   energy: float) -> float:
        """Calculate tunneling current through potential barrier.
        
        Uses WKB approximation for transmission coefficient.
        
        Args:
            barrier_potential: Potential barrier (J)
            energy: Incident particle energy (J)
            
        Returns:
            Transmission coefficient
        """
        # Find classical turning points
        turning_points = []
        for i in range(len(barrier_potential) - 1):
            if (barrier_potential[i] - energy) * (barrier_potential[i+1] - energy) < 0:
                turning_points.append(i)
        
        if len(turning_points) < 2:
            # No barrier or particle above barrier
            return 1.0
        
        # WKB integral
        x1_idx = turning_points[0]
        x2_idx = turning_points[1]
        
        # Integrate sqrt(2m(V-E)) from x1 to x2
        integrand = np.sqrt(2 * self.params.particle_mass * 
                          np.abs(barrier_potential[x1_idx:x2_idx] - energy))
        
        wkb_integral = np.trapz(integrand, self.x[x1_idx:x2_idx])
        
        # Transmission coefficient
        T = np.exp(-2 * wkb_integral / HBAR)
        
        return T
    
    def interface_field_enhancement(self, interface_position: float = 0.0,
                                   roughness: float = 1e-9) -> Dict:
        """Calculate field enhancement at rough interface.
        
        Args:
            interface_position: Position of interface (m)
            roughness: RMS roughness (m)
            
        Returns:
            Field enhancement results
        """
        # Create rough interface profile
        interface_profile = np.exp(-(self.x - interface_position)**2 / (2 * roughness**2))
        
        # Dielectric function with interface
        epsilon = 1.0 + (self.params.dielectric_constant - 1) * interface_profile
        
        # Modified Poisson equation with varying dielectric
        # ∇·(ε∇φ) = -ρ
        
        # Solve for field distribution
        test_charge = 1e-18  # Small test charge
        rho = np.zeros(self.N)
        rho[self.N//2] = test_charge / self.dx_mean  # Point charge at center
        
        # Simplified solution (assuming slow variation of ε)
        phi = self.solve_poisson(rho / epsilon)
        E_field = -np.gradient(phi, self.x)
        
        # Enhancement factor
        E_uniform = test_charge / (4 * np.pi * EPSILON_0 * roughness**2)
        enhancement = np.max(np.abs(E_field)) / E_uniform
        
        return {
            "interface_profile": interface_profile,
            "dielectric_profile": epsilon,
            "electric_field": E_field,
            "enhancement_factor": enhancement,
            "max_field": np.max(np.abs(E_field))
        }


# Example usage and testing
if __name__ == "__main__":
    print("Poisson-Schrödinger Solver for LENR Simulations")
    print("=" * 60)
    
    # Create solver with default parameters
    solver = PoissonSchrodingerSolver()
    
    print(f"\nGrid Configuration:")
    print(f"  Grid points: {solver.N}")
    print(f"  Grid range: [{solver.params.grid_min*1e9:.2f}, {solver.params.grid_max*1e9:.2f}] nm")
    print(f"  Average spacing: {solver.dx_mean*1e9:.3f} nm")
    
    # Test Poisson solver
    print(f"\nTesting Poisson Solver...")
    
    # Create test charge distribution (Gaussian)
    sigma = 1e-9  # 1 nm width
    rho_test = 1e-18 * np.exp(-solver.x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    
    phi = solver.solve_poisson(rho_test)
    E_field = -np.gradient(phi, solver.x)
    
    print(f"  Maximum potential: {np.max(phi):.3f} V")
    print(f"  Maximum field: {np.max(np.abs(E_field)):.2e} V/m")
    
    # Test Schrödinger solver
    print(f"\nTesting Schrödinger Solver...")
    
    # Create potential well
    V_well = np.zeros(solver.N)
    well_width = 2e-9  # 2 nm
    well_depth = 1.0 * EV_TO_JOULE  # 1 eV deep
    V_well[np.abs(solver.x) < well_width/2] = -well_depth
    
    energies, wavefunctions = solver.solve_schrodinger(V_well, n_states=3)
    
    print(f"  Lowest 3 energy levels (eV):")
    for i, E in enumerate(energies[:3]):
        print(f"    E_{i}: {E*JOULE_TO_EV:.4f} eV")
    
    # Test self-consistent solution
    print(f"\nTesting Self-Consistent Solution...")
    
    results = solver.solve_self_consistent(n_electrons=5)
    
    if results["converged"]:
        print(f"  Converged in {results['iterations']} iterations")
        print(f"  Ground state energy: {results['energies'][0]:.4f} eV")
        print(f"  Maximum field: {np.max(np.abs(results['electric_field'])):.2e} V/m")
    else:
        print(f"  Did not converge after {results['iterations']} iterations")
    
    # Test interface field enhancement
    print(f"\nTesting Interface Field Enhancement...")
    
    enhancement = solver.interface_field_enhancement(roughness=2e-9)
    
    print(f"  Field enhancement factor: {enhancement['enhancement_factor']:.2f}x")
    print(f"  Maximum field: {enhancement['max_field']:.2e} V/m")
    
    # Test tunneling current
    print(f"\nTesting Tunneling Current Calculation...")
    
    # Create barrier
    barrier_height = 0.5 * EV_TO_JOULE  # 0.5 eV
    barrier_width = 1e-9  # 1 nm
    V_barrier = np.zeros(solver.N)
    V_barrier[np.abs(solver.x) < barrier_width/2] = barrier_height
    
    # Calculate transmission for different energies
    print(f"  Transmission coefficient vs energy:")
    for E_eV in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        E = E_eV * EV_TO_JOULE
        T = solver.calculate_tunneling_current(V_barrier, E)
        print(f"    E = {E_eV:.1f} eV: T = {T:.2e}")
    
    print(f"\n[SUCCESS] Poisson-Schrödinger solver is operational!")
