"""Electron screening calculations for LENR simulations.

This module implements the electron screening models described in Section 2.1
of the theoretical framework, including Thomas-Fermi screening, Debye screening,
and interface electron dynamics.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass
import logging
from scipy import special, integrate
from scipy.optimize import fsolve

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    HBAR, C, E_CHARGE, M_ELECTRON, KB, EPSILON_0,
    EV_TO_JOULE, JOULE_TO_EV, FINE_STRUCTURE,
    PD_FERMI_ENERGY, NI_FERMI_ENERGY, AVOGADRO,
    R_BOHR, RYDBERG, ANGSTROM, PICOMETER
)

logger = logging.getLogger(__name__)


@dataclass
class ScreeningParameters:
    """Parameters for electron screening calculations."""
    
    material: str  # "Pd", "Ni", "Ti", etc.
    temperature: float  # Temperature (K)
    electron_density: float  # Electron density (electrons/m^3)
    loading_ratio: float  # D/Pd or H/Ni loading ratio
    lattice_constant: float  # Lattice constant (m)
    fermi_energy: float  # Fermi energy (eV)
    surface_roughness: float  # Surface roughness (m)
    defect_density: float  # Defect density (defects/m^3)


class ElectronScreening:
    """Calculator for electron screening effects in LENR systems."""
    
    def __init__(self, parameters: Optional[ScreeningParameters] = None):
        """Initialize screening calculator with parameters."""
        self.params = parameters or self._default_pd_parameters()
        
    @staticmethod
    def _default_pd_parameters() -> ScreeningParameters:
        """Get default parameters for Pd-D system."""
        # Pd has ~0.36 electrons per atom in conduction band
        electron_density = 0.36 * (1.0 / (3.89e-10)**3)  # electrons/m^3
        
        return ScreeningParameters(
            material="Pd",
            temperature=300.0,
            electron_density=electron_density,
            loading_ratio=0.9,
            lattice_constant=3.89e-10,
            fermi_energy=PD_FERMI_ENERGY,
            surface_roughness=10e-9,
            defect_density=1e20
        )
    
    def thomas_fermi_screening_length(self) -> float:
        """Calculate Thomas-Fermi screening length.
        
        Returns:
            Screening length (m)
        """
        # Fermi energy in Joules
        E_f = self.params.fermi_energy * EV_TO_JOULE
        
        # Fermi wavevector
        k_f = np.sqrt(2 * M_ELECTRON * E_f) / HBAR
        
        # Thomas-Fermi screening wavevector
        # k_TF = sqrt(4*k_f/(pi*a_0)) for 3D
        k_TF = np.sqrt(4 * k_f / (np.pi * R_BOHR))
        
        # Screening length
        lambda_TF = 1.0 / k_TF
        
        return lambda_TF
    
    def debye_screening_length(self) -> float:
        """Calculate Debye screening length for thermal electrons.
        
        Returns:
            Debye length (m)
        """
        # Thermal energy
        kT = KB * self.params.temperature
        
        # Debye length
        lambda_D = np.sqrt(EPSILON_0 * kT / 
                          (self.params.electron_density * E_CHARGE**2))
        
        return lambda_D
    
    def combined_screening_length(self) -> float:
        """Calculate combined screening length from Thomas-Fermi and Debye.
        
        Returns:
            Effective screening length (m)
        """
        lambda_TF = self.thomas_fermi_screening_length()
        lambda_D = self.debye_screening_length()
        
        # Combined screening (inverse quadrature)
        lambda_eff = 1.0 / np.sqrt(1.0/lambda_TF**2 + 1.0/lambda_D**2)
        
        return lambda_eff
    
    def yukawa_potential(self, r: float, Z1: float = 1, Z2: float = 1) -> float:
        """Calculate screened Coulomb (Yukawa) potential.
        
        Args:
            r: Distance (m)
            Z1, Z2: Atomic numbers
            
        Returns:
            Screened potential (eV)
        """
        lambda_s = self.combined_screening_length()
        
        # Yukawa potential: V(r) = (Z1*Z2*e^2/(4*pi*eps_0*r)) * exp(-r/lambda_s)
        V_coulomb = Z1 * Z2 * E_CHARGE**2 / (4 * np.pi * EPSILON_0 * r)
        V_screened = V_coulomb * np.exp(-r / lambda_s) * JOULE_TO_EV
        
        return V_screened
    
    def screening_energy(self, nuclear_separation: float = 1e-10) -> float:
        """Calculate screening energy reduction at given separation.
        
        Args:
            nuclear_separation: Distance between nuclei (m)
            
        Returns:
            Screening energy (eV)
        """
        # Unscreened Coulomb potential
        V_unscreened = E_CHARGE**2 / (4 * np.pi * EPSILON_0 * nuclear_separation)
        V_unscreened *= JOULE_TO_EV
        
        # Screened potential
        V_screened = self.yukawa_potential(nuclear_separation)
        
        # Screening energy is the difference
        U_e = V_unscreened - V_screened
        
        # Add loading ratio enhancement
        if self.params.loading_ratio > 0.85:
            enhancement = 1.0 + 2.0 * (self.params.loading_ratio - 0.85)
            U_e *= enhancement
        
        # Add defect enhancement
        defect_enhancement = 1.0 + np.log10(self.params.defect_density / 1e18) * 0.1
        U_e *= max(defect_enhancement, 1.0)
        
        return U_e
    
    def interface_electron_overlap(self, distance_from_surface: float) -> float:
        """Calculate electron wavefunction overlap at interface.
        
        Based on Section 2.2 - electron wavefunctions extending into near-field.
        
        Args:
            distance_from_surface: Distance from metal surface (m)
            
        Returns:
            Overlap probability (dimensionless)
        """
        # Electron decay length (1-5 pm as per paper)
        decay_length = 2.5e-12  # 2.5 pm average
        
        # Work function barrier
        work_function = 5.12 if self.params.material == "Pd" else 5.15  # eV
        
        # Decay constant
        kappa = np.sqrt(2 * M_ELECTRON * work_function * EV_TO_JOULE) / HBAR
        
        # Wavefunction amplitude
        psi = np.exp(-kappa * distance_from_surface)
        
        # Overlap probability
        overlap = psi**2
        
        # Surface roughness enhancement
        if distance_from_surface < self.params.surface_roughness:
            roughness_factor = 1.0 + self.params.surface_roughness / (10e-9)
            overlap *= roughness_factor
        
        return overlap
    
    def modified_rydberg_enhancement(self, n_principal: int = 10) -> float:
        """Calculate enhancement from modified Rydberg matter states.
        
        Args:
            n_principal: Principal quantum number
            
        Returns:
            Enhancement factor
        """
        # Rydberg radius scales as n^2
        r_rydberg = R_BOHR * n_principal**2
        
        # Binding energy scales as 1/n^2
        binding_energy = RYDBERG / n_principal**2
        
        # Enhancement from increased spatial extent
        spatial_enhancement = (r_rydberg / R_BOHR)**0.5
        
        # Reduced binding increases screening
        binding_enhancement = np.sqrt(RYDBERG / binding_energy)
        
        # Combined enhancement (limited to realistic values)
        enhancement = min(spatial_enhancement * binding_enhancement, 10.0)
        
        return enhancement
    
    def plasmon_enhanced_screening(self, frequency: Optional[float] = None) -> float:
        """Calculate surface plasmon enhancement of screening.
        
        Args:
            frequency: Plasmon frequency (Hz), if None use bulk plasmon
            
        Returns:
            Enhancement factor
        """
        if frequency is None:
            # Bulk plasmon frequency
            omega_p = np.sqrt(self.params.electron_density * E_CHARGE**2 / 
                            (EPSILON_0 * M_ELECTRON))
        else:
            omega_p = frequency
        
        # Plasmon energy
        E_plasmon = HBAR * omega_p * JOULE_TO_EV
        
        # Enhancement from field concentration
        # Surface plasmons can enhance local fields by 10-100x
        field_enhancement = min(10.0 * np.sqrt(E_plasmon / self.params.fermi_energy), 100.0)
        
        # Screening enhancement is roughly sqrt of field enhancement
        screening_enhancement = np.sqrt(field_enhancement)
        
        return screening_enhancement
    
    def lattice_compression_effect(self, strain: float = 0.0) -> float:
        """Calculate screening changes due to lattice compression.
        
        Args:
            strain: Lattice strain (fractional)
            
        Returns:
            Screening modification factor
        """
        # Compression increases electron density
        density_factor = (1 + strain)**3
        
        # Modified screening length
        lambda_0 = self.combined_screening_length()
        lambda_compressed = lambda_0 / np.sqrt(density_factor)
        
        # Enhancement factor
        enhancement = lambda_0 / lambda_compressed
        
        return enhancement
    
    def total_screening_energy(self, nuclear_separation: float = 1e-10,
                              include_enhancements: bool = True) -> Dict[str, float]:
        """Calculate total screening energy with all effects.
        
        Args:
            nuclear_separation: Distance between nuclei (m)
            include_enhancements: Include enhancement mechanisms
            
        Returns:
            Dictionary with screening contributions
        """
        # Base screening energy
        base_screening = self.screening_energy(nuclear_separation)
        
        results = {
            "base_screening": base_screening,
            "thomas_fermi_length": self.thomas_fermi_screening_length(),
            "debye_length": self.debye_screening_length(),
            "combined_length": self.combined_screening_length()
        }
        
        if include_enhancements:
            # Interface enhancement (at 5 pm from surface)
            interface_factor = self.interface_electron_overlap(5e-12)
            
            # Rydberg enhancement
            rydberg_factor = self.modified_rydberg_enhancement()
            
            # Plasmon enhancement
            plasmon_factor = self.plasmon_enhanced_screening()
            
            # Lattice strain (assume 1% compression near defects)
            strain_factor = self.lattice_compression_effect(0.01)
            
            # Total enhancement
            total_enhancement = (1.0 + 
                               0.1 * interface_factor + 
                               0.05 * (rydberg_factor - 1.0) + 
                               0.2 * (plasmon_factor - 1.0) + 
                               0.15 * (strain_factor - 1.0))
            
            total_screening = base_screening * total_enhancement
            
            results.update({
                "interface_enhancement": interface_factor,
                "rydberg_enhancement": rydberg_factor,
                "plasmon_enhancement": plasmon_factor,
                "strain_enhancement": strain_factor,
                "total_enhancement": total_enhancement,
                "total_screening_energy": total_screening
            })
        else:
            results["total_screening_energy"] = base_screening
        
        return results
    
    def screening_factor_for_tunneling(self, energy: float) -> float:
        """Calculate screening factor for use in tunneling calculations.
        
        Args:
            energy: Incident particle energy (eV)
            
        Returns:
            Screening factor F_screen
        """
        # Get total screening energy
        screening = self.total_screening_energy()
        U_e = screening["total_screening_energy"]
        
        # Screening factor (Eq. 11.14 in Storms)
        # F_screen = exp(pi * eta * U_e / E)
        # where eta is Sommerfeld parameter
        
        # Simplified approximation
        F_screen = np.exp(np.pi * np.sqrt(U_e / energy))
        
        return F_screen
    
    def monte_carlo_screening_distribution(self, n_samples: int = 1000) -> Dict:
        """Monte Carlo sampling of screening energy distribution.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Dictionary with statistical results
        """
        screening_energies = []
        
        for _ in range(n_samples):
            # Vary parameters within reasonable ranges
            temp_variation = np.random.normal(1.0, 0.1)
            density_variation = np.random.normal(1.0, 0.15)
            loading_variation = np.random.uniform(0.85, 1.0)
            
            # Temporary modified parameters
            original_temp = self.params.temperature
            original_density = self.params.electron_density
            original_loading = self.params.loading_ratio
            
            self.params.temperature *= temp_variation
            self.params.electron_density *= density_variation
            self.params.loading_ratio = loading_variation
            
            # Calculate screening
            result = self.total_screening_energy()
            screening_energies.append(result["total_screening_energy"])
            
            # Restore parameters
            self.params.temperature = original_temp
            self.params.electron_density = original_density
            self.params.loading_ratio = original_loading
        
        screening_energies = np.array(screening_energies)
        
        return {
            "mean": np.mean(screening_energies),
            "std": np.std(screening_energies),
            "min": np.min(screening_energies),
            "max": np.max(screening_energies),
            "median": np.median(screening_energies),
            "percentiles": {
                "p10": np.percentile(screening_energies, 10),
                "p25": np.percentile(screening_energies, 25),
                "p75": np.percentile(screening_energies, 75),
                "p90": np.percentile(screening_energies, 90)
            },
            "samples": screening_energies.tolist()
        }


# Example usage and testing
if __name__ == "__main__":
    # Create screening calculator for Pd-D system
    screening = ElectronScreening()
    
    print("Electron Screening Calculations for Pd-D System")
    print("=" * 60)
    
    # Calculate screening lengths
    lambda_TF = screening.thomas_fermi_screening_length()
    lambda_D = screening.debye_screening_length()
    lambda_eff = screening.combined_screening_length()
    
    print(f"\nScreening Lengths:")
    print(f"  Thomas-Fermi length: {lambda_TF*1e10:.3f} Å")
    print(f"  Debye length: {lambda_D*1e10:.3f} Å")
    print(f"  Combined effective: {lambda_eff*1e10:.3f} Å")
    
    # Calculate screening energies at different separations
    print(f"\nScreening Energy vs Nuclear Separation:")
    print(f"  {'Separation (Å)':<15} {'Screening (eV)':<15} {'Enhancement':<15}")
    print("  " + "-" * 45)
    
    separations = [0.5, 1.0, 2.0, 5.0, 10.0]  # Angstroms
    for sep_A in separations:
        sep_m = sep_A * ANGSTROM
        result = screening.total_screening_energy(sep_m)
        base = result["base_screening"]
        total = result["total_screening_energy"]
        enhancement = result.get("total_enhancement", 1.0)
        print(f"  {sep_A:<15.1f} {total:<15.2f} {enhancement:<15.3f}")
    
    # Interface electron overlap
    print(f"\nInterface Electron Overlap:")
    distances = [1, 2, 5, 10, 20]  # picometers
    for d_pm in distances:
        d_m = d_pm * PICOMETER
        overlap = screening.interface_electron_overlap(d_m)
        print(f"  {d_pm} pm: {overlap:.2e}")
    
    # Monte Carlo analysis
    print(f"\nMonte Carlo Analysis (1000 samples):")
    mc_results = screening.monte_carlo_screening_distribution(1000)
    print(f"  Mean screening energy: {mc_results['mean']:.2f} ± {mc_results['std']:.2f} eV")
    print(f"  Range: {mc_results['min']:.2f} - {mc_results['max']:.2f} eV")
    print(f"  90% confidence: {mc_results['percentiles']['p10']:.2f} - {mc_results['percentiles']['p90']:.2f} eV")
    
    # Enhancement factors
    print(f"\nEnhancement Mechanisms:")
    result = screening.total_screening_energy()
    print(f"  Plasmon enhancement: {result['plasmon_enhancement']:.2f}x")
    print(f"  Rydberg enhancement: {result['rydberg_enhancement']:.2f}x")
    print(f"  Strain enhancement: {result['strain_enhancement']:.2f}x")
    print(f"  Total enhancement: {result['total_enhancement']:.2f}x")
