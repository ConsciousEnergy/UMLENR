"""Quantum tunneling calculations for LENR simulations.

This module implements the quantum tunneling models described in Section 2.4
of the theoretical framework, including Gamow factors, barrier penetration,
and coherent tunneling effects.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    HBAR, C, E_CHARGE, M_PROTON, M_DEUTERON, 
    EV_TO_JOULE, JOULE_TO_EV, FINE_STRUCTURE,
    D_D_COULOMB_BARRIER, D_D_NUCLEAR_RADIUS
)

logger = logging.getLogger(__name__)


@dataclass
class TunnelingParameters:
    """Parameters for tunneling calculations."""
    
    particle1_mass: float  # Mass of first particle (kg)
    particle2_mass: float  # Mass of second particle (kg)
    charge1: float  # Charge of first particle (C)
    charge2: float  # Charge of second particle (C)
    temperature: float  # Temperature (K)
    electric_field: float  # Applied electric field (V/m)
    screening_energy: float  # Electron screening energy (eV)
    coherence_factor: float  # Coherent enhancement factor
    loading_ratio: float  # D/Pd or H/Ni loading ratio


class QuantumTunneling:
    """Quantum tunneling calculator for nuclear reactions in condensed matter."""
    
    def __init__(self, parameters: Optional[TunnelingParameters] = None):
        """Initialize tunneling calculator with parameters."""
        self.params = parameters or self._default_dd_parameters()
        
    @staticmethod
    def _default_dd_parameters() -> TunnelingParameters:
        """Get default parameters for D-D fusion."""
        return TunnelingParameters(
            particle1_mass=M_DEUTERON,
            particle2_mass=M_DEUTERON,
            charge1=E_CHARGE,
            charge2=E_CHARGE,
            temperature=300.0,
            electric_field=1e9,
            screening_energy=10.0,
            coherence_factor=1.0,
            loading_ratio=0.9
        )
    
    def calculate_reduced_mass(self) -> float:
        """Calculate reduced mass of the two-particle system."""
        m1, m2 = self.params.particle1_mass, self.params.particle2_mass
        return (m1 * m2) / (m1 + m2)
    
    def calculate_gamow_factor(self, energy: float) -> float:
        """Calculate Gamow penetration factor.
        
        Args:
            energy: Relative kinetic energy (eV)
            
        Returns:
            Gamow factor G (dimensionless)
        """
        # Convert energy to Joules
        E = energy * EV_TO_JOULE
        
        # Calculate reduced mass
        mu = self.calculate_reduced_mass()
        
        # Sommerfeld parameter
        eta = (self.params.charge1 * self.params.charge2 * 
               np.sqrt(mu / (2 * E))) / (4 * np.pi * 8.854e-12 * HBAR)
        
        # Gamow factor
        G = 2 * np.pi * eta
        
        return G
    
    def tunneling_probability_wkb(self, energy: float, 
                                 barrier_height: Optional[float] = None,
                                 barrier_width: Optional[float] = None) -> float:
        """Calculate tunneling probability using WKB approximation.
        
        Args:
            energy: Incident particle energy (eV)
            barrier_height: Coulomb barrier height (eV)
            barrier_width: Effective barrier width (m)
            
        Returns:
            Tunneling probability (0-1)
        """
        if barrier_height is None:
            barrier_height = self.calculate_effective_barrier()
        
        if barrier_width is None:
            barrier_width = self.calculate_barrier_width(energy, barrier_height)
        
        if energy >= barrier_height:
            return 1.0
        
        # Convert to SI units
        E = energy * EV_TO_JOULE
        V = barrier_height * EV_TO_JOULE
        
        # Reduced mass
        mu = self.calculate_reduced_mass()
        
        # WKB integral
        kappa = np.sqrt(2 * mu * (V - E)) / HBAR
        wkb_integral = kappa * barrier_width
        
        # Tunneling probability
        P = np.exp(-2 * wkb_integral)
        
        return P
    
    def calculate_effective_barrier(self) -> float:
        """Calculate effective Coulomb barrier with screening.
        
        Returns:
            Effective barrier height (eV)
        """
        # Base Coulomb barrier
        barrier = D_D_COULOMB_BARRIER
        
        # Apply electron screening reduction
        barrier -= self.params.screening_energy
        
        # Apply electric field reduction (simplified)
        field_reduction = self.params.electric_field * D_D_NUCLEAR_RADIUS * JOULE_TO_EV
        barrier -= field_reduction
        
        # Loading ratio effect (empirical)
        if self.params.loading_ratio > 0.85:
            loading_factor = 1.0 - 0.1 * (self.params.loading_ratio - 0.85)
            barrier *= loading_factor
        
        return max(barrier, 0.0)
    
    def calculate_barrier_width(self, energy: float, barrier_height: float) -> float:
        """Calculate effective barrier width.
        
        Args:
            energy: Incident energy (eV)
            barrier_height: Barrier height (eV)
            
        Returns:
            Barrier width (m)
        """
        if energy >= barrier_height:
            return 0.0
        
        # Classical turning points
        r_inner = D_D_NUCLEAR_RADIUS
        
        # Outer turning point from Coulomb potential
        Z1Z2 = (self.params.charge1 * self.params.charge2) / (E_CHARGE * E_CHARGE)
        r_outer = (Z1Z2 * 1.44e-9 * EV_TO_JOULE) / (energy * EV_TO_JOULE)
        
        return max(r_outer - r_inner, 0.0)
    
    def coherent_tunneling_enhancement(self, n_particles: int = 4) -> float:
        """Calculate enhancement from coherent multi-body tunneling.
        
        Based on Takahashi's tetrahedral symmetric condensate model.
        
        Args:
            n_particles: Number of coherently tunneling particles
            
        Returns:
            Enhancement factor
        """
        if n_particles <= 1:
            return 1.0
        
        # Base coherence enhancement
        base_enhancement = self.params.coherence_factor
        
        # Multi-body correlation factor (simplified model)
        correlation = np.sqrt(n_particles)
        
        # Phase space enhancement
        phase_factor = np.power(n_particles, 0.25)
        
        # Total enhancement
        enhancement = base_enhancement * correlation * phase_factor
        
        return enhancement
    
    def calculate_total_enhancement(self, energy: float) -> Dict[str, float]:
        """Calculate total tunneling enhancement with all effects.
        
        Args:
            energy: Incident energy (eV)
            
        Returns:
            Dictionary with enhancement factors
        """
        # Base tunneling probability
        base_prob = self.tunneling_probability_wkb(energy)
        
        # Screening enhancement
        screened_barrier = self.calculate_effective_barrier()
        screened_prob = self.tunneling_probability_wkb(energy, screened_barrier)
        screening_enhancement = screened_prob / base_prob if base_prob > 0 else 1.0
        
        # Coherent enhancement
        coherent_enhancement = self.coherent_tunneling_enhancement()
        
        # Field enhancement (simplified)
        field_enhancement = 1.0 + 0.1 * np.log10(self.params.electric_field / 1e9)
        
        # Total enhancement
        total_enhancement = screening_enhancement * coherent_enhancement * field_enhancement
        
        return {
            "base_probability": base_prob,
            "screened_probability": screened_prob,
            "screening_enhancement": screening_enhancement,
            "coherent_enhancement": coherent_enhancement,
            "field_enhancement": field_enhancement,
            "total_enhancement": total_enhancement,
            "final_probability": base_prob * total_enhancement
        }
    
    def reaction_rate(self, energy: float, density: float, 
                      cross_section: Optional[float] = None) -> float:
        """Calculate reaction rate per unit volume.
        
        Args:
            energy: Relative energy (eV)
            density: Number density of reactants (m^-3)
            cross_section: Reaction cross section (m^2)
            
        Returns:
            Reaction rate (reactions/m^3/s)
        """
        # Get tunneling probability
        enhancement = self.calculate_total_enhancement(energy)
        P = enhancement["final_probability"]
        
        # Estimate cross section if not provided
        if cross_section is None:
            # Simplified fusion cross section model
            cross_section = 1e-31 * P  # m^2
        
        # Relative velocity
        mu = self.calculate_reduced_mass()
        v = np.sqrt(2 * energy * EV_TO_JOULE / mu)
        
        # Reaction rate
        rate = density * density * cross_section * v * P
        
        return rate
    
    def temperature_averaged_rate(self, temperature: float, 
                                 density: float) -> Tuple[float, float]:
        """Calculate Maxwell-averaged reaction rate.
        
        Args:
            temperature: Temperature (K)
            density: Number density (m^-3)
            
        Returns:
            Tuple of (rate, uncertainty)
        """
        from scipy.integrate import quad
        from scipy.constants import k as KB
        
        kT = KB * temperature * JOULE_TO_EV
        
        def maxwell_weighted_rate(E):
            # Maxwell-Boltzmann distribution
            weight = np.sqrt(E / kT) * np.exp(-E / kT)
            # Reaction rate at this energy
            rate = self.reaction_rate(E, density)
            return weight * rate
        
        # Integrate over energy distribution
        result, error = quad(maxwell_weighted_rate, 0.1, 100 * kT)
        
        # Normalize
        result *= 2 * np.sqrt(2 / (np.pi * kT))
        error *= 2 * np.sqrt(2 / (np.pi * kT))
        
        return result, error
    
    def simulate_tunneling_events(self, n_samples: int = 10000,
                                 energy_range: Tuple[float, float] = (0.1, 100.0)) -> Dict:
        """Monte Carlo simulation of tunneling events.
        
        Args:
            n_samples: Number of Monte Carlo samples
            energy_range: Range of energies to sample (eV)
            
        Returns:
            Dictionary with simulation results
        """
        # Generate random energies
        energies = np.random.uniform(energy_range[0], energy_range[1], n_samples)
        
        # Calculate tunneling for each energy
        probabilities = []
        enhancements = []
        
        for E in energies:
            result = self.calculate_total_enhancement(E)
            probabilities.append(result["final_probability"])
            enhancements.append(result["total_enhancement"])
        
        probabilities = np.array(probabilities)
        enhancements = np.array(enhancements)
        
        # Statistics
        results = {
            "mean_probability": np.mean(probabilities),
            "std_probability": np.std(probabilities),
            "max_probability": np.max(probabilities),
            "min_probability": np.min(probabilities),
            "mean_enhancement": np.mean(enhancements),
            "std_enhancement": np.std(enhancements),
            "percentiles": {
                "p50": np.percentile(probabilities, 50),
                "p90": np.percentile(probabilities, 90),
                "p95": np.percentile(probabilities, 95),
                "p99": np.percentile(probabilities, 99)
            },
            "energies": energies.tolist(),
            "probabilities": probabilities.tolist()
        }
        
        logger.info(f"Tunneling simulation complete: mean P = {results['mean_probability']:.2e}")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Create tunneling calculator with default D-D parameters
    tunneling = QuantumTunneling()
    
    # Calculate tunneling at different energies
    energies = [1.0, 10.0, 100.0, 1000.0]  # eV
    
    print("Quantum Tunneling Calculations for D-D Fusion")
    print("=" * 50)
    
    for E in energies:
        result = tunneling.calculate_total_enhancement(E)
        print(f"\nEnergy: {E} eV")
        print(f"  Base probability: {result['base_probability']:.2e}")
        print(f"  With screening: {result['screened_probability']:.2e}")
        print(f"  Total enhancement: {result['total_enhancement']:.2e}")
        print(f"  Final probability: {result['final_probability']:.2e}")
    
    # Run Monte Carlo simulation
    print("\n" + "=" * 50)
    print("Monte Carlo Simulation (10000 samples)")
    print("=" * 50)
    
    sim_results = tunneling.simulate_tunneling_events(n_samples=10000)
    print(f"Mean probability: {sim_results['mean_probability']:.2e} ± {sim_results['std_probability']:.2e}")
    print(f"Mean enhancement: {sim_results['mean_enhancement']:.2e}")
    print(f"95th percentile: {sim_results['percentiles']['p95']:.2e}")
    print(f"99th percentile: {sim_results['percentiles']['p99']:.2e}")
