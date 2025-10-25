"""Bubble dynamics and cavitation calculations for LENR simulations.

This module implements the bubble collapse dynamics described in Section 2.7
of the theoretical framework, including Rayleigh-Plesset, Gilmore, and 
Keller-Miksis equations, as well as Electro-Nuclear Collapse (ENC) coupling.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass
import logging
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import fsolve

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    KB, E_CHARGE, EPSILON_0, C,
    EV_TO_JOULE, JOULE_TO_EV, 
    DEFAULT_TEMPERATURE, DEFAULT_PRESSURE
)

logger = logging.getLogger(__name__)


@dataclass
class BubbleParameters:
    """Parameters for bubble dynamics calculations."""
    
    R0: float  # Initial bubble radius (m)
    P0: float  # Ambient pressure (Pa)
    P_vapor: float  # Vapor pressure (Pa)
    rho: float  # Liquid density (kg/m^3)
    sigma: float  # Surface tension (N/m)
    viscosity: float  # Dynamic viscosity (Pa·s)
    gamma: float  # Specific heat ratio
    c_sound: float  # Sound speed in liquid (m/s)
    temperature: float  # Temperature (K)
    driving_pressure: float  # External driving pressure amplitude (Pa)
    driving_frequency: float  # Driving frequency (Hz)
    electric_field: float  # Applied electric field (V/m)


class BubbleDynamics:
    """Calculator for bubble dynamics and cavitation effects in LENR."""
    
    def __init__(self, parameters: Optional[BubbleParameters] = None):
        """Initialize bubble dynamics calculator."""
        self.params = parameters or self._default_water_parameters()
        
    @staticmethod
    def _default_water_parameters() -> BubbleParameters:
        """Get default parameters for water at room temperature."""
        return BubbleParameters(
            R0=10e-6,  # 10 micron initial radius
            P0=101325.0,  # 1 atm
            P_vapor=2338.0,  # Water vapor pressure at 20°C
            rho=998.0,  # Water density
            sigma=0.0728,  # Surface tension water/air
            viscosity=0.001,  # Water viscosity
            gamma=1.33,  # Specific heat ratio for vapor
            c_sound=1480.0,  # Sound speed in water
            temperature=293.15,  # 20°C
            driving_pressure=1.5e5,  # 1.5 atm driving
            driving_frequency=20e3,  # 20 kHz ultrasound
            electric_field=0.0  # No field initially
        )
    
    def rayleigh_plesset(self, t: float, y: np.ndarray, 
                        driving_func: Optional[Callable] = None) -> np.ndarray:
        """Rayleigh-Plesset equation for bubble dynamics.
        
        R*R'' + (3/2)*R'^2 = (1/rho)*(P_B - P_inf - P_drive)
        
        Args:
            t: Time (s)
            y: State vector [R, R']
            driving_func: External pressure driving function
            
        Returns:
            Derivatives [R', R'']
        """
        R, Rdot = y
        
        # Prevent negative radius
        if R <= 0:
            R = 1e-10
        
        # Bubble pressure (polytropic gas law)
        P_gas = (self.params.P0 + 2*self.params.sigma/self.params.R0) * \
                (self.params.R0/R)**(3*self.params.gamma)
        
        # Vapor pressure
        P_vapor = self.params.P_vapor
        
        # Total bubble pressure
        P_B = P_gas + P_vapor - 2*self.params.sigma/R - \
              4*self.params.viscosity*Rdot/R
        
        # External pressure
        if driving_func is None:
            # Default sinusoidal driving
            P_drive = self.params.driving_pressure * \
                     np.sin(2*np.pi*self.params.driving_frequency*t)
        else:
            P_drive = driving_func(t)
        
        P_inf = self.params.P0 + P_drive
        
        # Rayleigh-Plesset acceleration
        Rddot = (1/R) * ((P_B - P_inf)/self.params.rho - 1.5*Rdot**2)
        
        return np.array([Rdot, Rddot])
    
    def gilmore_equation(self, t: float, y: np.ndarray) -> np.ndarray:
        """Gilmore equation for high-amplitude oscillations.
        
        Accounts for liquid compressibility effects.
        
        Args:
            t: Time (s)
            y: State vector [R, U] where U = R'
            
        Returns:
            Derivatives [R', U']
        """
        R, U = y
        
        if R <= 0:
            R = 1e-10
        
        # Enthalpy at bubble wall
        n = 7.0  # Tait equation parameter for water
        B = 3000e5  # Tait parameter (Pa)
        
        # Pressure at bubble wall
        P_gas = (self.params.P0 + 2*self.params.sigma/self.params.R0) * \
                (self.params.R0/R)**(3*self.params.gamma)
        P_B = P_gas + self.params.P_vapor - 2*self.params.sigma/R
        
        # Liquid enthalpy
        H = (n/(n-1)) * (self.params.P0/self.params.rho) * \
            (((P_B + B)/(self.params.P0 + B))**((n-1)/n) - 1)
        
        # Speed of sound at bubble wall
        C = self.params.c_sound * np.sqrt(1 + ((n-1)/n)*H*self.params.rho/self.params.P0)
        
        # Modified pressure term
        P_drive = self.params.driving_pressure * \
                 np.sin(2*np.pi*self.params.driving_frequency*t)
        P_inf = self.params.P0 + P_drive
        
        # Gilmore acceleration
        dH_dt = (n/(n-1)) * (self.params.P0/self.params.rho) * \
                ((n-1)/n) * ((P_B + B)/(self.params.P0 + B))**(-1/n) * \
                (1/(self.params.P0 + B))
        
        # Time derivative of enthalpy
        dP_B = -3*self.params.gamma*P_gas*U/R + 2*self.params.sigma*U/R**2
        dH = dH_dt * dP_B
        
        # Acceleration
        Udot = ((1 + U/C)*H + R*dH/(2*C) - 1.5*U**2*(1 - U/(3*C)))/(R*(1 - U/C))
        
        return np.array([U, Udot])
    
    def keller_miksis(self, t: float, y: np.ndarray) -> np.ndarray:
        """Keller-Miksis equation for acoustic cavitation.
        
        Includes acoustic radiation damping.
        
        Args:
            t: Time (s)
            y: State vector [R, R']
            
        Returns:
            Derivatives [R', R'']
        """
        R, Rdot = y
        
        if R <= 0:
            R = 1e-10
        
        c = self.params.c_sound
        
        # Bubble pressure
        P_gas = (self.params.P0 + 2*self.params.sigma/self.params.R0) * \
                (self.params.R0/R)**(3*self.params.gamma)
        P_B = P_gas + self.params.P_vapor - 2*self.params.sigma/R - \
              4*self.params.viscosity*Rdot/R
        
        # Driving pressure
        omega = 2*np.pi*self.params.driving_frequency
        P_drive = self.params.driving_pressure * np.sin(omega*t)
        P_inf = self.params.P0 + P_drive
        
        # Pressure derivative
        dP_drive = self.params.driving_pressure * omega * np.cos(omega*t)
        
        # Keller-Miksis equation
        factor1 = (1 - Rdot/c)
        factor2 = (1 + Rdot/c)
        factor3 = (1 - Rdot/(3*c))
        
        numerator = (P_B - P_inf)/self.params.rho + R*dP_drive/(self.params.rho*c) - \
                   1.5*Rdot**2*factor3
        
        Rddot = numerator / (R*factor1 + R*Rdot/(c*factor2))
        
        return np.array([Rdot, Rddot])
    
    def solve_bubble_dynamics(self, t_span: Tuple[float, float], 
                            model: str = "rayleigh_plesset",
                            y0: Optional[np.ndarray] = None) -> Dict:
        """Solve bubble dynamics equations.
        
        Args:
            t_span: Time interval (t0, tf)
            model: Model to use ("rayleigh_plesset", "gilmore", "keller_miksis")
            y0: Initial conditions [R0, R'0]
            
        Returns:
            Solution dictionary
        """
        if y0 is None:
            y0 = [self.params.R0, 0.0]
        
        # Select model
        if model == "rayleigh_plesset":
            func = self.rayleigh_plesset
        elif model == "gilmore":
            func = self.gilmore_equation
        elif model == "keller_miksis":
            func = self.keller_miksis
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Solve ODE
        sol = solve_ivp(func, t_span, y0, 
                       dense_output=True, 
                       method='LSODA',
                       rtol=1e-8, atol=1e-10)
        
        return {
            "t": sol.t,
            "R": sol.y[0],
            "Rdot": sol.y[1],
            "success": sol.success,
            "model": model
        }
    
    def collapse_temperature(self, R: float, Rdot: float) -> float:
        """Calculate temperature during bubble collapse.
        
        Uses adiabatic compression model.
        
        Args:
            R: Current bubble radius (m)
            Rdot: Bubble wall velocity (m/s)
            
        Returns:
            Temperature (K)
        """
        # Compression ratio
        compression = (self.params.R0 / R)**3
        
        # Adiabatic heating
        T_adiabatic = self.params.temperature * \
                     compression**(self.params.gamma - 1)
        
        # Kinetic heating from collapse
        # T_kinetic ~ rho * Rdot^2 / (3 * k_B * n)
        # where n is number density
        
        n_gas = self.params.P0 / (KB * self.params.temperature)  # molecules/m^3
        T_kinetic = self.params.rho * Rdot**2 / (3 * KB * n_gas)
        
        # Total temperature
        T_total = T_adiabatic + T_kinetic
        
        # Cap at realistic values
        return min(T_total, 50000.0)  # Max 50,000 K
    
    def collapse_pressure(self, R: float, Rdot: float) -> float:
        """Calculate pressure during bubble collapse.
        
        Args:
            R: Current bubble radius (m)
            Rdot: Bubble wall velocity (m/s)
            
        Returns:
            Pressure (Pa)
        """
        # Gas pressure from compression
        P_gas = (self.params.P0 + 2*self.params.sigma/self.params.R0) * \
                (self.params.R0/R)**(3*self.params.gamma)
        
        # Dynamic pressure from wall motion
        P_dynamic = 0.5 * self.params.rho * Rdot**2
        
        # Shock pressure (simplified)
        if abs(Rdot) > self.params.c_sound:
            # Supersonic collapse
            P_shock = self.params.rho * self.params.c_sound * abs(Rdot)
        else:
            P_shock = 0
        
        # Total pressure
        P_total = P_gas + P_dynamic + P_shock
        
        return P_total
    
    def energy_concentration(self, R: float, Rdot: float) -> Dict[str, float]:
        """Calculate energy concentration during collapse.
        
        Args:
            R: Bubble radius (m)
            Rdot: Wall velocity (m/s)
            
        Returns:
            Energy densities and concentrations
        """
        # Volume
        V = (4/3) * np.pi * R**3
        V0 = (4/3) * np.pi * self.params.R0**3
        
        # Potential energy from compression
        E_potential = self.params.P0 * V0 * np.log(V0/V)
        
        # Kinetic energy of liquid
        E_kinetic = 2 * np.pi * self.params.rho * R**3 * Rdot**2
        
        # Thermal energy
        T = self.collapse_temperature(R, Rdot)
        n_molecules = self.params.P0 * V / (KB * self.params.temperature)
        E_thermal = 1.5 * n_molecules * KB * T
        
        # Energy density (J/m^3)
        energy_density = (E_potential + E_kinetic + E_thermal) / V
        
        # Convert to eV/atom
        # Assume ~10^29 atoms/m^3 in compressed region
        atoms_per_m3 = 1e29
        energy_per_atom = energy_density * JOULE_TO_EV / atoms_per_m3
        
        return {
            "potential_energy": E_potential,
            "kinetic_energy": E_kinetic,
            "thermal_energy": E_thermal,
            "total_energy": E_potential + E_kinetic + E_thermal,
            "energy_density": energy_density,
            "energy_per_atom": energy_per_atom,
            "temperature": T,
            "pressure": self.collapse_pressure(R, Rdot)
        }
    
    def electro_nuclear_collapse(self, R: float, field: float) -> Dict[str, float]:
        """Calculate Electro-Nuclear Collapse (ENC) parameters.
        
        Based on Matsumoto's observations of micro-plasmoid formation.
        
        Args:
            R: Bubble radius (m)
            field: Electric field strength (V/m)
            
        Returns:
            ENC parameters
        """
        # Field ionization threshold
        E_ionization = 3e9  # V/m for water
        
        if field < E_ionization:
            ionization_degree = (field / E_ionization)**2
        else:
            ionization_degree = 1.0
        
        # Plasma density
        n_plasma = ionization_degree * 1e28  # electrons/m^3
        
        # Plasma frequency
        omega_p = np.sqrt(n_plasma * E_CHARGE**2 / (EPSILON_0 * 9.109e-31))
        
        # Magnetic field from current (simplified)
        # B ~ mu_0 * J * R, where J is current density
        J = n_plasma * E_CHARGE * 1e6  # Assume 1 km/s drift velocity
        B_field = 1.256e-6 * J * R
        
        # Pinch pressure
        P_pinch = B_field**2 / (2 * 1.256e-6)
        
        # Compression factor from pinch
        compression_factor = 1.0 + P_pinch / self.params.P0
        
        # ENC energy concentration
        enc_energy = 0.5 * EPSILON_0 * field**2 + B_field**2 / (2 * 1.256e-6)
        enc_energy_eV = enc_energy * R**3 * JOULE_TO_EV / n_plasma
        
        return {
            "ionization_degree": ionization_degree,
            "plasma_density": n_plasma,
            "plasma_frequency": omega_p,
            "magnetic_field": B_field,
            "pinch_pressure": P_pinch,
            "compression_factor": compression_factor,
            "enc_energy_per_particle": enc_energy_eV,
            "is_enc_regime": field > E_ionization
        }
    
    def sonoluminescence_conditions(self, R: float, Rdot: float) -> bool:
        """Check if conditions for sonoluminescence are met.
        
        Args:
            R: Bubble radius (m)
            Rdot: Wall velocity (m/s)
            
        Returns:
            True if SL conditions are satisfied
        """
        # Temperature threshold for light emission
        T = self.collapse_temperature(R, Rdot)
        T_threshold = 10000.0  # K
        
        # Compression ratio threshold
        compression = (self.params.R0 / R)
        compression_threshold = 10.0
        
        # Wall velocity threshold (approach sound speed)
        velocity_threshold = 0.1 * self.params.c_sound
        
        return (T > T_threshold and 
                compression > compression_threshold and 
                abs(Rdot) > velocity_threshold)
    
    def shock_wave_parameters(self, R: float, Rdot: float) -> Dict[str, float]:
        """Calculate shock wave parameters from bubble collapse.
        
        Args:
            R: Bubble radius (m)
            Rdot: Wall velocity (m/s)
            
        Returns:
            Shock wave properties
        """
        # Mach number
        mach = abs(Rdot) / self.params.c_sound
        
        if mach < 1.0:
            # No shock
            return {
                "mach_number": mach,
                "shock_pressure": 0.0,
                "shock_temperature": self.params.temperature,
                "shock_velocity": 0.0
            }
        
        # Rankine-Hugoniot relations for strong shock
        # P2/P1 = (2*gamma*M^2 - (gamma-1))/(gamma+1)
        gamma = self.params.gamma
        P_shock = self.params.P0 * (2*gamma*mach**2 - (gamma-1))/(gamma+1)
        
        # Temperature jump
        T_shock = self.params.temperature * \
                 ((2*gamma*mach**2 - (gamma-1)) * 
                  ((gamma-1)*mach**2 + 2)) / ((gamma+1)**2 * mach**2)
        
        # Shock velocity
        v_shock = abs(Rdot) * (gamma+1)*mach**2 / ((gamma-1)*mach**2 + 2)
        
        return {
            "mach_number": mach,
            "shock_pressure": P_shock,
            "shock_temperature": T_shock,
            "shock_velocity": v_shock
        }
    
    def simulate_collapse(self, n_cycles: int = 5) -> Dict:
        """Simulate multiple acoustic cycles with collapse analysis.
        
        Args:
            n_cycles: Number of acoustic cycles to simulate
            
        Returns:
            Simulation results
        """
        # Time span for simulation
        period = 1.0 / self.params.driving_frequency
        t_span = (0, n_cycles * period)
        
        # Solve dynamics
        solution = self.solve_bubble_dynamics(t_span)
        
        if not solution["success"]:
            logger.warning("Bubble dynamics solution failed")
            return solution
        
        # Analyze collapse events
        R = solution["R"]
        Rdot = solution["Rdot"]
        t = solution["t"]
        
        # Find minimum radii (collapse points)
        min_indices = []
        for i in range(1, len(R)-1):
            if R[i] < R[i-1] and R[i] < R[i+1]:
                min_indices.append(i)
        
        # Analyze each collapse
        collapses = []
        for idx in min_indices:
            R_min = R[idx]
            Rdot_collapse = Rdot[idx]
            
            energy = self.energy_concentration(R_min, Rdot_collapse)
            shock = self.shock_wave_parameters(R_min, Rdot_collapse)
            sl_conditions = self.sonoluminescence_conditions(R_min, Rdot_collapse)
            
            if self.params.electric_field > 0:
                enc = self.electro_nuclear_collapse(R_min, self.params.electric_field)
            else:
                enc = None
            
            collapses.append({
                "time": t[idx],
                "radius": R_min,
                "velocity": Rdot_collapse,
                "compression_ratio": self.params.R0 / R_min,
                "temperature": energy["temperature"],
                "pressure": energy["pressure"],
                "energy_per_atom": energy["energy_per_atom"],
                "mach_number": shock["mach_number"],
                "sonoluminescence": sl_conditions,
                "enc_parameters": enc
            })
        
        # Statistics
        if collapses:
            max_temp = max(c["temperature"] for c in collapses)
            max_pressure = max(c["pressure"] for c in collapses)
            max_energy = max(c["energy_per_atom"] for c in collapses)
        else:
            max_temp = max_pressure = max_energy = 0
        
        return {
            "solution": solution,
            "collapses": collapses,
            "n_collapses": len(collapses),
            "max_temperature": max_temp,
            "max_pressure": max_pressure,
            "max_energy_per_atom": max_energy
        }


# Example usage and testing
if __name__ == "__main__":
    # Create bubble dynamics calculator
    bubble = BubbleDynamics()
    
    print("Bubble Dynamics and Cavitation Calculations")
    print("=" * 70)
    
    # Parameters
    print(f"\nBubble Parameters:")
    print(f"  Initial radius: {bubble.params.R0*1e6:.1f} µm")
    print(f"  Ambient pressure: {bubble.params.P0/1e5:.1f} bar")
    print(f"  Driving pressure: {bubble.params.driving_pressure/1e5:.1f} bar")
    print(f"  Driving frequency: {bubble.params.driving_frequency/1e3:.1f} kHz")
    print(f"  Sound speed: {bubble.params.c_sound:.0f} m/s")
    
    # Simulate collapse
    print(f"\nSimulating {5} acoustic cycles...")
    results = bubble.simulate_collapse(n_cycles=5)
    
    print(f"\nCollapse Analysis:")
    print(f"  Number of collapses: {results['n_collapses']}")
    print(f"  Maximum temperature: {results['max_temperature']:.0f} K")
    print(f"  Maximum pressure: {results['max_pressure']/1e9:.2f} GPa")
    print(f"  Maximum energy concentration: {results['max_energy_per_atom']:.2f} eV/atom")
    
    # Detail first few collapses
    print(f"\nDetailed Collapse Events:")
    for i, collapse in enumerate(results['collapses'][:3]):
        print(f"\n  Collapse #{i+1} at t = {collapse['time']*1e6:.2f} µs:")
        print(f"    Minimum radius: {collapse['radius']*1e9:.1f} nm")
        print(f"    Compression ratio: {collapse['compression_ratio']:.1f}x")
        print(f"    Wall velocity: {abs(collapse['velocity']):.0f} m/s")
        print(f"    Temperature: {collapse['temperature']:.0f} K")
        print(f"    Pressure: {collapse['pressure']/1e9:.3f} GPa")
        print(f"    Energy/atom: {collapse['energy_per_atom']:.3f} eV")
        print(f"    Mach number: {collapse['mach_number']:.2f}")
        print(f"    Sonoluminescence: {collapse['sonoluminescence']}")
    
    # Test with electric field for ENC
    print(f"\n" + "="*70)
    print("Testing Electro-Nuclear Collapse (ENC) Regime")
    print("="*70)
    
    bubble.params.electric_field = 5e9  # 5 GV/m
    enc_params = bubble.electro_nuclear_collapse(1e-9, bubble.params.electric_field)
    
    print(f"\nENC Parameters at E = {bubble.params.electric_field/1e9:.1f} GV/m:")
    print(f"  Ionization degree: {enc_params['ionization_degree']:.3f}")
    print(f"  Plasma density: {enc_params['plasma_density']:.2e} /m³")
    print(f"  Plasma frequency: {enc_params['plasma_frequency']:.2e} rad/s")
    print(f"  Magnetic field: {enc_params['magnetic_field']:.3f} T")
    print(f"  Pinch pressure: {enc_params['pinch_pressure']/1e9:.3f} GPa")
    print(f"  Compression factor: {enc_params['compression_factor']:.2f}x")
    print(f"  ENC energy/particle: {enc_params['enc_energy_per_particle']:.3f} eV")
    print(f"  In ENC regime: {enc_params['is_enc_regime']}")
    
    # Test different models
    print(f"\n" + "="*70)
    print("Comparing Bubble Models")
    print("="*70)
    
    models = ["rayleigh_plesset", "keller_miksis", "gilmore"]
    t_span = (0, 100e-6)  # 100 microseconds
    
    for model in models:
        try:
            sol = bubble.solve_bubble_dynamics(t_span, model=model)
            R_min = np.min(sol["R"])
            R_max = np.max(sol["R"])
            print(f"\n{model.replace('_', '-').title()}:")
            print(f"  R_min/R0: {R_min/bubble.params.R0:.3f}")
            print(f"  R_max/R0: {R_max/bubble.params.R0:.3f}")
            print(f"  Compression: {bubble.params.R0/R_min:.1f}x")
        except Exception as e:
            print(f"\n{model}: Failed - {e}")
