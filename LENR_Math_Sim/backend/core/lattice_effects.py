"""Lattice effects and quantum coherence calculations for LENR simulations.

This module implements the lattice coherence models described in Section 2.5
of the theoretical framework, including Preparata's QED coherent domains,
Hagelstein's phonon-nuclear coupling, and collective oscillation effects.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
import logging
from scipy import special, integrate, linalg
from scipy.fft import fft, fftfreq

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    HBAR, C, E_CHARGE, M_ELECTRON, M_DEUTERON, M_PROTON, KB,
    EV_TO_JOULE, JOULE_TO_EV, EPSILON_0, MU_0,
    PD_DEBYE_TEMP, NI_DEBYE_TEMP, PD_LATTICE_CONSTANT,
    NI_LATTICE_CONSTANT, AVOGADRO
)

logger = logging.getLogger(__name__)


@dataclass
class LatticeParameters:
    """Parameters for lattice effects calculations."""
    
    material: str  # "Pd", "Ni", etc.
    temperature: float  # Temperature (K)
    lattice_constant: float  # Lattice constant (m)
    debye_temperature: float  # Debye temperature (K)
    loading_ratio: float  # D/Pd or H/Ni loading ratio
    coherence_length: float  # Coherence domain size (m)
    phonon_frequency: float  # Dominant phonon frequency (Hz)
    coupling_strength: float  # Phonon-nuclear coupling strength
    defect_density: float  # Defect density (defects/m^3)
    domain_size: int  # Number of atoms in coherent domain


class LatticeEffects:
    """Calculator for lattice coherence and collective effects in LENR."""
    
    def __init__(self, parameters: Optional[LatticeParameters] = None):
        """Initialize lattice effects calculator."""
        self.params = parameters or self._default_pd_parameters()
        
    @staticmethod
    def _default_pd_parameters() -> LatticeParameters:
        """Get default parameters for Pd-D system."""
        debye_freq = KB * PD_DEBYE_TEMP / HBAR
        
        return LatticeParameters(
            material="Pd",
            temperature=300.0,
            lattice_constant=PD_LATTICE_CONSTANT,
            debye_temperature=PD_DEBYE_TEMP,
            loading_ratio=0.9,
            coherence_length=10e-9,  # 10 nm coherent domains
            phonon_frequency=debye_freq,
            coupling_strength=0.01,  # Dimensionless coupling
            defect_density=1e20,  # defects/m^3
            domain_size=1000  # atoms
        )
    
    def debye_frequency(self) -> float:
        """Calculate Debye cutoff frequency.
        
        Returns:
            Debye frequency (Hz)
        """
        return KB * self.params.debye_temperature / HBAR
    
    def phonon_energy(self, frequency: Optional[float] = None) -> float:
        """Calculate phonon energy.
        
        Args:
            frequency: Phonon frequency (Hz), if None use Debye frequency
            
        Returns:
            Phonon energy (eV)
        """
        if frequency is None:
            frequency = self.debye_frequency()
        
        return HBAR * frequency * JOULE_TO_EV
    
    def coherent_domain_energy(self) -> float:
        """Calculate energy in a coherent domain (Preparata model).
        
        Based on QED coherent domain theory where electromagnetic
        field couples to matter oscillations.
        
        Returns:
            Coherent domain energy (eV/atom)
        """
        # Number of atoms in coherent domain
        n_atoms = self.params.domain_size
        
        # Coherent oscillation frequency (typically THz range)
        omega_coherent = 2 * np.pi * 1e12  # 1 THz
        
        # Zero-point energy of coherent oscillation
        E_zp = 0.5 * HBAR * omega_coherent * JOULE_TO_EV
        
        # Coherent energy scales with sqrt(N) for bosonic modes
        E_coherent = E_zp * np.sqrt(n_atoms) / n_atoms
        
        # Loading ratio enhancement
        if self.params.loading_ratio > 0.85:
            loading_factor = 1.0 + 2.0 * (self.params.loading_ratio - 0.85)
            E_coherent *= loading_factor
        
        return E_coherent
    
    def phonon_nuclear_coupling(self, phonon_mode: int = 0) -> float:
        """Calculate phonon-nuclear coupling strength (Hagelstein model).
        
        Args:
            phonon_mode: Phonon mode index
            
        Returns:
            Coupling energy (eV)
        """
        # Phonon frequency for given mode
        omega = self.params.phonon_frequency * (1 + 0.1 * phonon_mode)
        
        # Nuclear transition matrix element (simplified)
        # Based on E1 transition strength
        nuclear_element = 0.1 * E_CHARGE * 1e-15  # fm scale
        
        # Phonon displacement amplitude
        # <x^2> = hbar/(2*m*omega) for quantum harmonic oscillator
        mass = M_DEUTERON
        displacement = np.sqrt(HBAR / (2 * mass * omega))
        
        # Coupling energy
        coupling = self.params.coupling_strength * nuclear_element * displacement
        coupling_eV = coupling * JOULE_TO_EV
        
        return coupling_eV
    
    def frohlich_coherence_condition(self) -> bool:
        """Check if Fröhlich coherence condition is satisfied.
        
        Coherence emerges when energy pumping rate exceeds dissipation.
        
        Returns:
            True if coherence condition is met
        """
        # Energy pumping rate (from loading/electric field)
        pump_rate = self.params.loading_ratio * 1e10  # Hz (simplified)
        
        # Dissipation rate (phonon scattering)
        T_ratio = self.params.temperature / self.params.debye_temperature
        dissipation_rate = 1e12 * T_ratio**3  # Hz (Debye model)
        
        # Coherence condition
        return pump_rate > dissipation_rate
    
    def tetrahedral_cluster_energy(self) -> float:
        """Calculate energy for tetrahedral D4 cluster (Takahashi model).
        
        Returns:
            Cluster formation energy (eV)
        """
        # Tetrahedral configuration of 4 deuterons
        n_deuterons = 4
        
        # Binding energy in lattice cage
        lattice_binding = 0.5 * n_deuterons  # eV per deuteron
        
        # Exchange interaction energy
        exchange_energy = 0.1 * n_deuterons * (n_deuterons - 1) / 2
        
        # Zero-point energy reduction from correlation
        zp_reduction = 0.2 * np.sqrt(n_deuterons)
        
        # Total cluster energy
        E_cluster = lattice_binding - exchange_energy - zp_reduction
        
        return E_cluster
    
    def plasmon_polariton_coupling(self) -> float:
        """Calculate plasmon-polariton coupling in lattice.
        
        Mixed modes of collective electronic and lattice oscillations.
        
        Returns:
            Coupling strength (eV)
        """
        # Plasma frequency for conduction electrons
        n_electrons = 1e28  # electrons/m^3 (typical for metals)
        omega_p = np.sqrt(n_electrons * E_CHARGE**2 / (EPSILON_0 * M_ELECTRON))
        
        # Optical phonon frequency (approximate)
        omega_phonon = self.debye_frequency() * 0.8
        
        # Anti-crossing gap at resonance
        coupling_strength = 0.1 * HBAR * np.sqrt(omega_p * omega_phonon) * JOULE_TO_EV
        
        return coupling_strength
    
    def bose_einstein_condensation_check(self) -> Dict[str, Union[bool, float]]:
        """Check for possible BEC-like behavior in loaded lattice.
        
        Returns:
            Dictionary with BEC parameters
        """
        # De Broglie wavelength for deuterons
        thermal_momentum = np.sqrt(2 * M_DEUTERON * KB * self.params.temperature)
        lambda_db = HBAR / thermal_momentum
        
        # Inter-particle spacing
        n_density = self.params.loading_ratio / self.params.lattice_constant**3
        spacing = n_density**(-1/3)
        
        # BEC condition: lambda_db > spacing
        bec_possible = lambda_db > spacing
        
        # Critical temperature estimate
        T_c = (HBAR**2 / (2 * M_DEUTERON * KB)) * (n_density**(2/3))
        
        return {
            "bec_possible": bec_possible,
            "de_broglie_wavelength": lambda_db,
            "particle_spacing": spacing,
            "critical_temperature": T_c,
            "overlap_ratio": lambda_db / spacing
        }
    
    def lattice_strain_field(self, position: np.ndarray) -> np.ndarray:
        """Calculate strain field around defects.
        
        Args:
            position: 3D position vector (m)
            
        Returns:
            Strain tensor components
        """
        # Simplified strain field around point defect
        r = np.linalg.norm(position)
        
        if r < 1e-12:  # Avoid singularity
            r = 1e-12
        
        # Strain decays as 1/r^3 for point defects
        strain_magnitude = 0.01 * (self.params.lattice_constant / r)**3
        
        # Strain tensor (simplified isotropic)
        strain = strain_magnitude * np.eye(3)
        
        # Add shear components
        strain[0, 1] = strain[1, 0] = 0.5 * strain_magnitude
        
        return strain
    
    def phonon_dos(self, energy: float) -> float:
        """Calculate phonon density of states (Debye model).
        
        Args:
            energy: Energy (eV)
            
        Returns:
            Density of states (states/eV)
        """
        # Debye cutoff energy
        E_debye = self.phonon_energy()
        
        if energy > E_debye or energy <= 0:
            return 0.0
        
        # Debye DOS: g(E) = 3 * E^2 / E_D^3
        dos = 3 * energy**2 / E_debye**3
        
        return dos
    
    def coherent_energy_transfer(self, energy_nuclear: float) -> float:
        """Calculate coherent energy transfer from nuclear to lattice.
        
        Based on Hagelstein's model of coherent energy exchange.
        
        Args:
            energy_nuclear: Nuclear energy to be transferred (eV)
            
        Returns:
            Fraction of energy coherently transferred
        """
        # Number of phonons needed
        E_phonon = self.phonon_energy()
        n_phonons = energy_nuclear / E_phonon
        
        # Coherent transfer probability decreases with number of phonons
        # P ~ exp(-n_phonons / n_critical)
        n_critical = 100  # Critical number for breakdown
        
        if n_phonons > n_critical:
            transfer_prob = np.exp(-n_phonons / n_critical)
        else:
            # For small n, use perturbation theory result
            transfer_prob = 1.0 / (1.0 + (n_phonons / 10)**2)
        
        # Enhancement from coherent domain
        if self.params.domain_size > 100:
            coherence_enhancement = np.sqrt(self.params.domain_size / 100)
            transfer_prob *= min(coherence_enhancement, 10.0)
        
        return transfer_prob
    
    def polaron_formation_energy(self) -> float:
        """Calculate polaron formation energy in loaded lattice.
        
        Returns:
            Polaron binding energy (eV)
        """
        # Electron-phonon coupling constant
        alpha = 0.1  # Weak to intermediate coupling
        
        # Phonon energy
        E_phonon = self.phonon_energy()
        
        # Polaron binding energy (Fröhlich model)
        E_polaron = alpha * E_phonon
        
        # Enhancement in highly loaded lattice
        if self.params.loading_ratio > 0.85:
            E_polaron *= (1 + self.params.loading_ratio)
        
        return E_polaron
    
    def total_lattice_enhancement(self) -> Dict[str, float]:
        """Calculate total enhancement from all lattice effects.
        
        Returns:
            Dictionary with all contributions
        """
        results = {
            "coherent_domain_energy": self.coherent_domain_energy(),
            "phonon_nuclear_coupling": self.phonon_nuclear_coupling(),
            "tetrahedral_cluster": self.tetrahedral_cluster_energy(),
            "plasmon_polariton": self.plasmon_polariton_coupling(),
            "polaron_energy": self.polaron_formation_energy(),
            "frohlich_coherent": self.frohlich_coherence_condition()
        }
        
        # BEC check
        bec_results = self.bose_einstein_condensation_check()
        results["bec_possible"] = bec_results["bec_possible"]
        results["bec_overlap_ratio"] = bec_results["overlap_ratio"]
        
        # Energy concentration factor
        total_energy = sum([
            results["coherent_domain_energy"],
            results["phonon_nuclear_coupling"],
            abs(results["tetrahedral_cluster"]),
            results["plasmon_polariton"],
            results["polaron_energy"]
        ])
        
        # Enhancement factor for tunneling
        # Energy concentration can reduce effective barrier
        enhancement_factor = 1.0 + total_energy / 10.0  # Normalized by 10 eV
        
        # Coherence bonus
        if results["frohlich_coherent"]:
            enhancement_factor *= 2.0
        
        # BEC bonus
        if results["bec_possible"]:
            enhancement_factor *= 1.5
        
        results["total_energy_concentration"] = total_energy
        results["enhancement_factor"] = enhancement_factor
        
        return results
    
    def monte_carlo_lattice_sampling(self, n_samples: int = 1000) -> Dict:
        """Monte Carlo sampling of lattice enhancement distribution.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Statistical results
        """
        enhancement_factors = []
        coherent_energies = []
        
        for _ in range(n_samples):
            # Vary parameters
            temp_var = np.random.normal(1.0, 0.1)
            loading_var = np.random.uniform(0.85, 1.0)
            domain_var = np.random.randint(100, 10000)
            
            # Store original values
            original_temp = self.params.temperature
            original_loading = self.params.loading_ratio
            original_domain = self.params.domain_size
            
            # Apply variations
            self.params.temperature *= temp_var
            self.params.loading_ratio = loading_var
            self.params.domain_size = domain_var
            
            # Calculate enhancement
            results = self.total_lattice_enhancement()
            enhancement_factors.append(results["enhancement_factor"])
            coherent_energies.append(results["coherent_domain_energy"])
            
            # Restore parameters
            self.params.temperature = original_temp
            self.params.loading_ratio = original_loading
            self.params.domain_size = original_domain
        
        enhancement_factors = np.array(enhancement_factors)
        coherent_energies = np.array(coherent_energies)
        
        return {
            "enhancement": {
                "mean": np.mean(enhancement_factors),
                "std": np.std(enhancement_factors),
                "max": np.max(enhancement_factors),
                "p90": np.percentile(enhancement_factors, 90)
            },
            "coherent_energy": {
                "mean": np.mean(coherent_energies),
                "std": np.std(coherent_energies),
                "max": np.max(coherent_energies)
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Create lattice effects calculator
    lattice = LatticeEffects()
    
    print("Lattice Effects and Coherence Calculations")
    print("=" * 60)
    
    # Basic parameters
    print(f"\nLattice Parameters:")
    print(f"  Material: {lattice.params.material}")
    print(f"  Temperature: {lattice.params.temperature} K")
    print(f"  Loading ratio: {lattice.params.loading_ratio}")
    print(f"  Coherence length: {lattice.params.coherence_length*1e9:.1f} nm")
    print(f"  Domain size: {lattice.params.domain_size} atoms")
    
    # Energy scales
    print(f"\nEnergy Scales:")
    print(f"  Debye energy: {lattice.phonon_energy():.3f} eV")
    print(f"  Coherent domain: {lattice.coherent_domain_energy():.3f} eV/atom")
    print(f"  Phonon-nuclear coupling: {lattice.phonon_nuclear_coupling():.3f} eV")
    print(f"  Tetrahedral cluster: {lattice.tetrahedral_cluster_energy():.3f} eV")
    print(f"  Plasmon-polariton: {lattice.plasmon_polariton_coupling():.3f} eV")
    print(f"  Polaron binding: {lattice.polaron_formation_energy():.3f} eV")
    
    # Coherence checks
    print(f"\nCoherence Conditions:")
    print(f"  Fröhlich coherence: {lattice.frohlich_coherence_condition()}")
    
    bec_check = lattice.bose_einstein_condensation_check()
    print(f"  BEC possible: {bec_check['bec_possible']}")
    print(f"  De Broglie wavelength: {bec_check['de_broglie_wavelength']*1e10:.3f} Å")
    print(f"  Particle spacing: {bec_check['particle_spacing']*1e10:.3f} Å")
    print(f"  Overlap ratio: {bec_check['overlap_ratio']:.3f}")
    
    # Total enhancement
    print(f"\nTotal Lattice Enhancement:")
    results = lattice.total_lattice_enhancement()
    print(f"  Energy concentration: {results['total_energy_concentration']:.3f} eV")
    print(f"  Enhancement factor: {results['enhancement_factor']:.2f}x")
    
    # Coherent energy transfer
    print(f"\nCoherent Energy Transfer Efficiency:")
    energies = [1, 10, 100, 1000, 10000]  # eV
    for E in energies:
        efficiency = lattice.coherent_energy_transfer(E)
        print(f"  {E:5d} eV: {efficiency:.3f}")
    
    # Monte Carlo analysis
    print(f"\nMonte Carlo Analysis (1000 samples):")
    mc_results = lattice.monte_carlo_lattice_sampling(1000)
    print(f"  Enhancement factor: {mc_results['enhancement']['mean']:.2f} ± {mc_results['enhancement']['std']:.2f}")
    print(f"  90th percentile: {mc_results['enhancement']['p90']:.2f}x")
    print(f"  Coherent energy: {mc_results['coherent_energy']['mean']:.3f} ± {mc_results['coherent_energy']['std']:.3f} eV/atom")
