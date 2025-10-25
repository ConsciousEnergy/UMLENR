"""Physical constants and standard parameters for LENR simulations."""

import numpy as np

# Fundamental Physical Constants (SI units)
HBAR = 1.054571817e-34  # Reduced Planck's constant (J·s)
C = 299792458.0  # Speed of light (m/s)
KB = 1.380649e-23  # Boltzmann constant (J/K)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
M_ELECTRON = 9.1093837015e-31  # Electron mass (kg)
M_PROTON = 1.67262192369e-27  # Proton mass (kg)
M_DEUTERON = 3.343583719e-27  # Deuteron mass (kg)
M_NEUTRON = 1.67492749804e-27  # Neutron mass (kg)
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
MU_0 = 1.25663706212e-6  # Vacuum permeability (H/m)
AVOGADRO = 6.02214076e23  # Avogadro's number (mol^-1)
R_BOHR = 5.29177210903e-11  # Bohr radius (m)
RYDBERG = 13.605693122994  # Rydberg energy (eV)
FINE_STRUCTURE = 7.2973525693e-3  # Fine structure constant (dimensionless)

# Energy Conversion Factors
EV_TO_JOULE = 1.602176634e-19  # eV to Joule
JOULE_TO_EV = 1.0 / EV_TO_JOULE  # Joule to eV
KEV_TO_JOULE = 1.602176634e-16  # keV to Joule
MEV_TO_JOULE = 1.602176634e-13  # MeV to Joule

# Length Scales
ANGSTROM = 1e-10  # Angstrom to meter
NANOMETER = 1e-9  # Nanometer to meter
PICOMETER = 1e-12  # Picometer to meter
FEMTOMETER = 1e-15  # Femtometer to meter

# Time Scales
FEMTOSECOND = 1e-15  # Femtosecond to second
ATTOSECOND = 1e-18  # Attosecond to second

# Material Properties (Palladium)
PD_LATTICE_CONSTANT = 3.89e-10  # Pd lattice constant (m)
PD_DEBYE_TEMP = 274.0  # Pd Debye temperature (K)
PD_FERMI_ENERGY = 7.1  # Pd Fermi energy (eV)
PD_WORK_FUNCTION = 5.12  # Pd work function (eV)
PD_DENSITY = 12023.0  # Pd density (kg/m^3)

# Material Properties (Nickel)
NI_LATTICE_CONSTANT = 3.52e-10  # Ni lattice constant (m)
NI_DEBYE_TEMP = 450.0  # Ni Debye temperature (K)
NI_FERMI_ENERGY = 11.67  # Ni Fermi energy (eV)
NI_WORK_FUNCTION = 5.15  # Ni work function (eV)
NI_DENSITY = 8908.0  # Ni density (kg/m^3)

# LENR Specific Parameters
D_D_COULOMB_BARRIER = 400e3  # D-D Coulomb barrier height (eV)
D_D_NUCLEAR_RADIUS = 2.1e-15  # D-D nuclear interaction radius (m)
SCREENING_LENGTH = 1e-10  # Typical electron screening length (m)
MAX_LOADING_RATIO = 1.0  # Maximum D/Pd or H/Ni loading ratio
CRITICAL_LOADING_RATIO = 0.85  # Critical loading ratio for LENR effects

# Simulation Parameters
DEFAULT_TEMPERATURE = 300.0  # Default temperature (K)
DEFAULT_PRESSURE = 101325.0  # Default pressure (Pa)
DEFAULT_ELECTRIC_FIELD = 1e9  # Default interface field (V/m)

# Numerical Parameters
GRID_RESOLUTION_MIN = 0.01e-9  # Minimum grid resolution (m)
GRID_RESOLUTION_MAX = 1e-12  # Maximum grid resolution near interfaces (m)
TIME_STEP_MIN = 0.1e-15  # Minimum time step (s)
TIME_STEP_MAX = 10e-18  # Maximum time step for critical regions (s)
CONVERGENCE_TOLERANCE = 1e-12  # Numerical convergence tolerance
MAX_ITERATIONS = 10000  # Maximum iterations for solvers

# Monte Carlo Parameters
DEFAULT_MC_SAMPLES = 10000  # Default number of Monte Carlo samples
MIN_MC_SAMPLES = 1000  # Minimum samples for statistical validity
MAX_MC_SAMPLES = 10000000  # Maximum samples for deep exploration

# Energy Thresholds
MIN_OBSERVABLE_ENERGY = 0.001  # Minimum observable energy (eV)
MAX_FIELD_ENERGY = 100.0  # Maximum local field energy (eV/atom)
FUSION_THRESHOLD_ENERGY = 10e3  # Minimum energy for fusion (eV)

# Statistical Thresholds
SIGNIFICANCE_LEVEL = 0.05  # Statistical significance level (p-value)
CONFIDENCE_LEVEL = 0.95  # Confidence level for intervals
MIN_SIGNAL_TO_NOISE = 3.0  # Minimum signal-to-noise ratio

# Casimir Effect Parameters
CASIMIR_PLATE_SEPARATION = 10e-9  # Typical plate separation (m)
CASIMIR_FORCE_SCALE = 1e-7  # Typical Casimir force scale (N)

# Bubble Dynamics Parameters
BUBBLE_RADIUS_MIN = 1e-9  # Minimum bubble radius (m)
BUBBLE_RADIUS_MAX = 1e-6  # Maximum bubble radius (m)
COLLAPSE_PRESSURE_MAX = 1e9  # Maximum collapse pressure (Pa)
COLLAPSE_TEMPERATURE_MAX = 10000.0  # Maximum collapse temperature (K)

# Field Enhancement Factors
NANOSTRUCTURE_ENHANCEMENT = 100.0  # Field enhancement at nanostructures
DEFECT_ENHANCEMENT = 10.0  # Field enhancement at defects
INTERFACE_ENHANCEMENT = 5.0  # Field enhancement at interfaces

# Experimental Validation Thresholds
MIN_EXCESS_HEAT = 0.01  # Minimum detectable excess heat (W)
MIN_HELIUM_DETECTION = 1e11  # Minimum He-4 atoms for detection
MIN_TRITIUM_DETECTION = 1e10  # Minimum tritium atoms for detection
MIN_NEUTRON_FLUX = 1e-3  # Minimum neutron flux above background (n/s)

def get_thermal_energy(temperature: float = DEFAULT_TEMPERATURE) -> float:
    """Calculate thermal energy kT at given temperature."""
    return KB * temperature * JOULE_TO_EV

def get_debye_frequency(material: str = "Pd") -> float:
    """Get Debye frequency for specified material."""
    if material.upper() == "PD":
        return KB * PD_DEBYE_TEMP / HBAR
    elif material.upper() == "NI":
        return KB * NI_DEBYE_TEMP / HBAR
    else:
        raise ValueError(f"Unknown material: {material}")

def get_fermi_velocity(material: str = "Pd") -> float:
    """Calculate Fermi velocity for specified material."""
    if material.upper() == "PD":
        fermi_energy_j = PD_FERMI_ENERGY * EV_TO_JOULE
    elif material.upper() == "NI":
        fermi_energy_j = NI_FERMI_ENERGY * EV_TO_JOULE
    else:
        raise ValueError(f"Unknown material: {material}")
    
    return np.sqrt(2 * fermi_energy_j / M_ELECTRON)

def get_plasma_frequency(electron_density: float) -> float:
    """Calculate plasma frequency for given electron density."""
    return np.sqrt(electron_density * E_CHARGE**2 / (EPSILON_0 * M_ELECTRON))

def get_screening_length(electron_density: float, temperature: float) -> float:
    """Calculate Thomas-Fermi screening length."""
    thermal_energy = KB * temperature
    debye_length = np.sqrt(EPSILON_0 * thermal_energy / (electron_density * E_CHARGE**2))
    fermi_energy = (HBAR**2 / (2 * M_ELECTRON)) * (3 * np.pi**2 * electron_density)**(2/3)
    thomas_fermi = np.sqrt(EPSILON_0 * fermi_energy / (electron_density * E_CHARGE**2))
    
    # Use combined screening length
    return 1.0 / np.sqrt(1.0/debye_length**2 + 1.0/thomas_fermi**2)
