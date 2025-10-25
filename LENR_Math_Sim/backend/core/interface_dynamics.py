"""Interface dynamics and double-layer field calculations for LENR simulations.

This module implements the interface electron dynamics and double-layer effects
described in Sections 2.2 and 2.3 of the theoretical framework, including
charge separation, field enhancement, and nanostructure effects.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
import logging
from scipy import special, integrate, optimize
from scipy.constants import epsilon_0

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    HBAR, E_CHARGE, M_ELECTRON, M_DEUTERON, KB, EPSILON_0,
    EV_TO_JOULE, JOULE_TO_EV, ANGSTROM, NANOMETER,
    PD_WORK_FUNCTION, NI_WORK_FUNCTION
)

logger = logging.getLogger(__name__)


@dataclass
class InterfaceParameters:
    """Parameters for interface dynamics calculations."""
    
    material: str  # "Pd", "Ni", etc.
    temperature: float  # Temperature (K)
    surface_potential: float  # Surface potential (V)
    double_layer_thickness: float  # Double layer thickness (m)
    surface_charge_density: float  # Surface charge density (C/m^2)
    dielectric_constant: float  # Relative dielectric constant
    ionic_strength: float  # Ionic strength of electrolyte (mol/L)
    ph: float  # pH of solution
    nanostructure_radius: float  # Radius of nanostructures (m)
    surface_roughness: float  # RMS surface roughness (m)


class InterfaceDynamics:
    """Calculator for interface dynamics and electric field effects in LENR."""
    
    def __init__(self, parameters: Optional[InterfaceParameters] = None):
        """Initialize interface dynamics calculator."""
        self.params = parameters or self._default_parameters()
        
    @staticmethod
    def _default_parameters() -> InterfaceParameters:
        """Get default parameters for Pd-electrolyte interface."""
        return InterfaceParameters(
            material="Pd",
            temperature=300.0,
            surface_potential=0.5,  # V
            double_layer_thickness=1e-9,  # 1 nm
            surface_charge_density=0.1,  # C/m^2
            dielectric_constant=80.0,  # Water
            ionic_strength=1.0,  # mol/L
            ph=7.0,
            nanostructure_radius=10e-9,  # 10 nm
            surface_roughness=5e-9  # 5 nm
        )
    
    def debye_length(self) -> float:
        """Calculate Debye screening length in electrolyte.
        
        Returns:
            Debye length (m)
        """
        # Convert ionic strength to number density
        # For 1:1 electrolyte: n = 2 * I * N_A * 1000
        n_ions = 2 * self.params.ionic_strength * 6.022e23 * 1000  # ions/m^3
        
        # Debye length
        kT = KB * self.params.temperature
        lambda_D = np.sqrt(self.params.dielectric_constant * EPSILON_0 * kT / 
                          (n_ions * E_CHARGE**2))
        
        return lambda_D
    
    def double_layer_field(self, distance: float) -> float:
        """Calculate electric field in double layer (Gouy-Chapman model).
        
        Args:
            distance: Distance from electrode surface (m)
            
        Returns:
            Electric field strength (V/m)
        """
        # Debye length
        lambda_D = self.debye_length()
        
        # Surface potential
        phi_0 = self.params.surface_potential
        
        # Field at distance x: E = (phi_0 / lambda_D) * exp(-x/lambda_D)
        if distance < 0:
            return 0.0
        
        E_field = (phi_0 / lambda_D) * np.exp(-distance / lambda_D)
        
        # Enhancement at rough surfaces
        if distance < self.params.surface_roughness:
            roughness_factor = 1.0 + self.params.surface_roughness / (5e-9)
            E_field *= roughness_factor
        
        return E_field
    
    def stern_layer_field(self) -> float:
        """Calculate field in Stern layer (compact double layer).
        
        Returns:
            Stern layer field (V/m)
        """
        # Stern layer thickness (typically 0.3-0.5 nm)
        d_stern = 0.4e-9  # m
        
        # Field from potential drop across Stern layer
        E_stern = self.params.surface_potential / d_stern
        
        return E_stern
    
    def nanostructure_field_enhancement(self, position: np.ndarray) -> float:
        """Calculate field enhancement near nanostructures.
        
        Args:
            position: 3D position relative to nanostructure center (m)
            
        Returns:
            Field enhancement factor
        """
        r = np.linalg.norm(position)
        R = self.params.nanostructure_radius
        
        if r < R:
            # Inside nanostructure
            return 1.0
        
        # Field enhancement for spherical nanoparticle
        # E_local/E_0 = 1 + 2*(R/r)^3 for conducting sphere
        enhancement = 1.0 + 2.0 * (R / r)**3
        
        # Maximum enhancement cap
        max_enhancement = 100.0
        enhancement = min(enhancement, max_enhancement)
        
        return enhancement
    
    def tip_field_enhancement(self, tip_radius: float, gap: float) -> float:
        """Calculate field enhancement at sharp tips/protrusions.
        
        Args:
            tip_radius: Radius of curvature at tip (m)
            gap: Distance from tip (m)
            
        Returns:
            Field enhancement factor
        """
        # Lightning rod effect: E ~ 1/r for sharp tips
        base_enhancement = self.params.nanostructure_radius / tip_radius
        
        # Distance-dependent decay
        decay_factor = np.exp(-gap / tip_radius)
        
        # Total enhancement
        enhancement = 1.0 + (base_enhancement - 1.0) * decay_factor
        
        # Cap at realistic values
        return min(enhancement, 1000.0)
    
    def interface_capacitance(self) -> float:
        """Calculate interface capacitance (double layer + Stern).
        
        Returns:
            Capacitance per unit area (F/m^2)
        """
        # Stern layer capacitance
        d_stern = 0.4e-9
        C_stern = self.params.dielectric_constant * EPSILON_0 / d_stern
        
        # Diffuse layer capacitance (Gouy-Chapman)
        lambda_D = self.debye_length()
        kT = KB * self.params.temperature
        
        # For small potentials: C_GC = eps*eps_0/lambda_D * cosh(z*e*phi_0/(2*kT))
        z = 1  # Charge number
        sinh_factor = np.sinh(z * E_CHARGE * self.params.surface_potential / (2 * kT))
        C_GC = (self.params.dielectric_constant * EPSILON_0 / lambda_D) * \
               np.sqrt(2) * sinh_factor
        
        # Total capacitance (series combination)
        C_total = 1.0 / (1.0/C_stern + 1.0/C_GC)
        
        return C_total
    
    def surface_plasmon_field(self, wavelength: float = 500e-9) -> float:
        """Calculate surface plasmon enhanced field.
        
        Args:
            wavelength: Excitation wavelength (m)
            
        Returns:
            Field enhancement from surface plasmons
        """
        # Simplified surface plasmon resonance model
        # Peak enhancement when wavelength matches plasmon resonance
        
        # Plasmon wavelength for Pd (approximate)
        lambda_plasmon = 200e-9  # UV range for Pd
        
        # Lorentzian resonance profile
        gamma = 50e-9  # Linewidth
        detuning = wavelength - lambda_plasmon
        
        # Enhancement factor
        enhancement = 1.0 + 10.0 / (1.0 + (detuning / gamma)**2)
        
        # Roughness increases plasmon coupling
        if self.params.surface_roughness > 1e-9:
            roughness_factor = 1.0 + np.log10(self.params.surface_roughness / 1e-9)
            enhancement *= roughness_factor
        
        return enhancement
    
    def electrostriction_pressure(self, field: float) -> float:
        """Calculate electrostrictive pressure from high fields.
        
        Args:
            field: Electric field strength (V/m)
            
        Returns:
            Pressure (Pa)
        """
        # Electrostriction coefficient for water (typical)
        gamma_e = 5e-10  # m^2/V^2
        
        # Pressure: P = 0.5 * eps * eps_0 * gamma_e * E^2
        pressure = 0.5 * self.params.dielectric_constant * EPSILON_0 * \
                  gamma_e * field**2
        
        return pressure
    
    def charge_injection_rate(self, field: float) -> float:
        """Calculate charge injection rate at interface (Fowler-Nordheim).
        
        Args:
            field: Electric field at interface (V/m)
            
        Returns:
            Current density (A/m^2)
        """
        # Work function
        phi = PD_WORK_FUNCTION if self.params.material == "Pd" else NI_WORK_FUNCTION
        phi_J = phi * EV_TO_JOULE
        
        # Fowler-Nordheim tunneling
        # J = A * F^2 * exp(-B * phi^(3/2) / F)
        # where F is field, phi is work function
        
        A_FN = E_CHARGE**3 / (8 * np.pi * HBAR * phi_J)
        B_FN = 8 * np.pi * np.sqrt(2 * M_ELECTRON) / (3 * HBAR * E_CHARGE)
        
        if field < 1e6:  # Below threshold
            return 0.0
        
        J = A_FN * field**2 * np.exp(-B_FN * phi_J**(3/2) / field)
        
        return J
    
    def interface_energy_density(self) -> float:
        """Calculate total energy density at interface.
        
        Returns:
            Energy density (eV/atom)
        """
        # Electric field energy
        E_field = self.double_layer_field(0)  # At surface
        field_energy = 0.5 * self.params.dielectric_constant * EPSILON_0 * E_field**2
        
        # Convert to eV per atom
        # Assume atomic density ~ 10^29 atoms/m^3
        atomic_density = 1e29
        field_energy_per_atom = field_energy * JOULE_TO_EV / atomic_density
        
        # Chemical potential contribution
        # mu = -e * phi for charged species
        chemical_energy = abs(self.params.surface_potential)  # eV
        
        # Surface energy contribution
        # Typical surface energy ~ 1-2 J/m^2
        surface_energy = 1.5  # J/m^2
        # Convert to eV/atom (surface atoms ~ 10^19 /m^2)
        surface_atoms = 1e19
        surface_energy_per_atom = surface_energy * JOULE_TO_EV / surface_atoms
        
        # Total interface energy
        total_energy = field_energy_per_atom + chemical_energy + surface_energy_per_atom
        
        return total_energy
    
    def field_assisted_tunneling_factor(self, field: float) -> float:
        """Calculate field-assisted tunneling enhancement.
        
        Args:
            field: Local electric field (V/m)
            
        Returns:
            Tunneling enhancement factor
        """
        # Field lowers effective barrier
        # Delta_V = e * F * d, where d is tunneling distance
        
        d_tunnel = 1e-10  # 1 Angstrom
        barrier_reduction = E_CHARGE * field * d_tunnel * JOULE_TO_EV
        
        # Enhancement factor ~ exp(Delta_V / kT)
        kT = KB * self.params.temperature * JOULE_TO_EV
        
        if barrier_reduction > 0:
            enhancement = np.exp(min(barrier_reduction / kT, 10.0))  # Cap at e^10
        else:
            enhancement = 1.0
        
        return enhancement
    
    def total_interface_enhancement(self) -> Dict[str, float]:
        """Calculate total enhancement from interface effects.
        
        Returns:
            Dictionary with all interface contributions
        """
        # Calculate various fields
        stern_field = self.stern_layer_field()
        double_layer_field_0 = self.double_layer_field(0)
        
        # Maximum field (at surface with nanostructure enhancement)
        nano_enhancement = self.nanostructure_field_enhancement(
            np.array([self.params.nanostructure_radius * 1.1, 0, 0])
        )
        max_field = stern_field * nano_enhancement
        
        # Field enhancement at sharp features
        tip_enhancement = self.tip_field_enhancement(1e-9, 1e-9)
        
        # Plasmon enhancement
        plasmon_enhancement = self.surface_plasmon_field()
        
        # Tunneling enhancement
        tunneling_enhancement = self.field_assisted_tunneling_factor(max_field)
        
        # Energy density
        interface_energy = self.interface_energy_density()
        
        # Capacitance
        capacitance = self.interface_capacitance()
        
        # Electrostriction
        pressure = self.electrostriction_pressure(max_field)
        
        # Current injection
        current = self.charge_injection_rate(max_field)
        
        results = {
            "stern_field": stern_field,
            "double_layer_field": double_layer_field_0,
            "max_field": max_field,
            "nano_enhancement": nano_enhancement,
            "tip_enhancement": tip_enhancement,
            "plasmon_enhancement": plasmon_enhancement,
            "tunneling_enhancement": tunneling_enhancement,
            "interface_energy": interface_energy,
            "capacitance": capacitance,
            "electrostriction_pressure": pressure,
            "injection_current": current,
            "debye_length": self.debye_length()
        }
        
        # Total enhancement factor
        total_enhancement = (tunneling_enhancement * 
                           np.sqrt(nano_enhancement) * 
                           np.power(plasmon_enhancement, 0.25))
        
        results["total_enhancement"] = total_enhancement
        
        return results
    
    def pH_effect_on_field(self, ph: float) -> float:
        """Calculate pH effect on interface field.
        
        Args:
            ph: Solution pH
            
        Returns:
            Field modification factor
        """
        # Nernstian shift: 59 mV per pH unit at 25°C
        delta_ph = ph - self.params.ph
        potential_shift = 0.059 * delta_ph  # V
        
        # Modified surface potential
        modified_potential = self.params.surface_potential + potential_shift
        
        # Field modification
        field_factor = abs(modified_potential / self.params.surface_potential)
        
        return field_factor
    
    def monte_carlo_interface_sampling(self, n_samples: int = 1000) -> Dict:
        """Monte Carlo sampling of interface enhancement distribution.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Statistical results
        """
        enhancements = []
        max_fields = []
        
        for _ in range(n_samples):
            # Vary parameters
            potential_var = np.random.uniform(0.1, 1.0)
            roughness_var = np.random.uniform(1e-9, 20e-9)
            nano_radius_var = np.random.uniform(5e-9, 50e-9)
            
            # Store originals
            original_potential = self.params.surface_potential
            original_roughness = self.params.surface_roughness
            original_radius = self.params.nanostructure_radius
            
            # Apply variations
            self.params.surface_potential = potential_var
            self.params.surface_roughness = roughness_var
            self.params.nanostructure_radius = nano_radius_var
            
            # Calculate enhancement
            results = self.total_interface_enhancement()
            enhancements.append(results["total_enhancement"])
            max_fields.append(results["max_field"])
            
            # Restore
            self.params.surface_potential = original_potential
            self.params.surface_roughness = original_roughness
            self.params.nanostructure_radius = original_radius
        
        enhancements = np.array(enhancements)
        max_fields = np.array(max_fields)
        
        return {
            "enhancement": {
                "mean": np.mean(enhancements),
                "std": np.std(enhancements),
                "max": np.max(enhancements),
                "p90": np.percentile(enhancements, 90)
            },
            "field": {
                "mean": np.mean(max_fields),
                "std": np.std(max_fields),
                "max": np.max(max_fields),
                "p90": np.percentile(max_fields, 90)
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Create interface dynamics calculator
    interface = InterfaceDynamics()
    
    print("Interface Dynamics and Double-Layer Field Calculations")
    print("=" * 70)
    
    # Basic parameters
    print(f"\nInterface Parameters:")
    print(f"  Material: {interface.params.material}")
    print(f"  Surface potential: {interface.params.surface_potential} V")
    print(f"  Nanostructure radius: {interface.params.nanostructure_radius*1e9:.1f} nm")
    print(f"  Surface roughness: {interface.params.surface_roughness*1e9:.1f} nm")
    print(f"  Ionic strength: {interface.params.ionic_strength} mol/L")
    
    # Field calculations
    print(f"\nElectric Fields:")
    print(f"  Stern layer field: {interface.stern_layer_field():.2e} V/m")
    print(f"  Double layer field (surface): {interface.double_layer_field(0):.2e} V/m")
    print(f"  Debye length: {interface.debye_length()*1e9:.2f} nm")
    
    # Field profile
    print(f"\nDouble Layer Field Profile:")
    distances = [0, 0.5, 1, 2, 5, 10]  # nm
    for d_nm in distances:
        d_m = d_nm * 1e-9
        field = interface.double_layer_field(d_m)
        print(f"  {d_nm:2.1f} nm: {field:.2e} V/m")
    
    # Enhancement factors
    print(f"\nField Enhancement Factors:")
    results = interface.total_interface_enhancement()
    print(f"  Nanostructure: {results['nano_enhancement']:.1f}x")
    print(f"  Tip enhancement: {results['tip_enhancement']:.1f}x")
    print(f"  Plasmon: {results['plasmon_enhancement']:.1f}x")
    print(f"  Maximum field: {results['max_field']:.2e} V/m")
    
    # Energy and effects
    print(f"\nInterface Energy and Effects:")
    print(f"  Interface energy density: {results['interface_energy']:.3f} eV/atom")
    print(f"  Capacitance: {results['capacitance']:.3f} F/m²")
    print(f"  Electrostriction pressure: {results['electrostriction_pressure']:.2e} Pa")
    print(f"  Tunneling enhancement: {results['tunneling_enhancement']:.2f}x")
    print(f"  Total enhancement: {results['total_enhancement']:.2f}x")
    
    # pH effect
    print(f"\npH Effects on Field:")
    ph_values = [5, 6, 7, 8, 9]
    for ph in ph_values:
        factor = interface.pH_effect_on_field(ph)
        print(f"  pH {ph}: {factor:.2f}x field modification")
    
    # Monte Carlo analysis
    print(f"\nMonte Carlo Analysis (1000 samples):")
    mc_results = interface.monte_carlo_interface_sampling(1000)
    print(f"  Enhancement: {mc_results['enhancement']['mean']:.2f} ± {mc_results['enhancement']['std']:.2f}")
    print(f"  90th percentile: {mc_results['enhancement']['p90']:.2f}x")
    print(f"  Maximum field: {mc_results['field']['max']:.2e} V/m")
