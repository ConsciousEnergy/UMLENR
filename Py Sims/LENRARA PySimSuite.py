##LENRARA Python Simulation Suite (WORK IN PROGRESS)
#The following is  a set of computational function variables to help us simulate Hydrogen Fusion Processes in Condensed Matter Nuclar Science
#License GNU 3.0 Conscious Energy 2024

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint, solve_ivp

# Constants
k_B = 1.380649e-23  # Boltzmann constant, J/K
hbar = 1.0545718e-34  # Reduced Planck constant, J·s
e = 1.60217662e-19  # Elementary charge, C
c = 3.0e8  # Speed of light, m/s
Z_1 = 1  # Atomic number for hydrogen
Z_2 = 1  # Atomic number for hydrogen
a = 1.0  # Lattice parameter
n = 3  # Number of unit cells along each axis

# Thermodynamic properties
delta_H = -40e3  # Enthalpy of formation, J/mol H2
delta_S = 130  # Entropy, J/mol·K H2

def create_fcc_lattice(a, n):
    """Create FCC lattice."""
    positions = []
    basis_vectors = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for vec in basis_vectors:
                    pos = (i + vec[0], j + vec[1], k + vec[2])
                    positions.append(pos)
    return np.array(positions) * a

def vant_hoff(delta_H, delta_S, T):
    """Calculate equilibrium pressure using Van't Hoff equation."""
    R = 8.314  # J/(mol·K)
    return np.exp((delta_H - T * delta_S) / (R * T))

def fusion_cross_section(T):
    """Calculate fusion cross-section."""
    term1 = (e**2 / (4 * np.pi * hbar * c))**2
    term2 = (1 / (k_B * T))**2
    term3 = np.exp(-3 * np.pi / (4 * np.sqrt(2)) * (Z_1 * Z_2 * e**2 / (hbar * c))**2 / (k_B * T))
    return 1e-24 * term1 * term2 * term3

def fusion_rate(n1, n2, sigma, v_rel):
    """Calculate fusion rate."""
    return n1 * n2 * sigma * v_rel

def simulate_fusion(T, n1, n2, v_rel):
    """Simulate LENR fusion event."""
    sigma = fusion_cross_section(T)
    rate = fusion_rate(n1, n2, sigma, v_rel)
    return 1 / rate

def excess_heat_generation(time, P_excess):
    """Calculate excess heat generation."""
    return np.trapz(P_excess, time)

def transmutation_model(y, t, k):
    """Transmutation pathways model."""
    return -k * y

def glow_discharge_power(I, V):
    """Calculate glow discharge power."""
    return I * V

def neutron_production_rate(sigma, rho_e, rho_p, E_e):
    """Calculate neutron production rate."""
    return hbar * sigma * rho_e * rho_p * E_e

def tsc_fusion_rate(S, G, T):
    """Calculate TSC fusion rate."""
    return S * G * T

# Initialize lattice and parameters
fcc_lattice = create_fcc_lattice(a, n)
T = 300  # Temperature in Kelvin
n1, n2 = 1e28, 1e28  # Number densities in m^-3
v_rel = 1e6  # Relative velocity in m/s

# Calculate equilibrium pressure using van't Hoff equation
equilibrium_pressure = vant_hoff(delta_H, delta_S, T)
print(f'Equilibrium Pressure: {equilibrium_pressure:.2e} Pa')

# Simulate LENR event
time_to_fusion = simulate_fusion(T, n1, n2, v_rel)
decay_positions = fcc_lattice
decay_times = np.random.exponential(scale=time_to_fusion, size=decay_positions.shape[0])

# Example excess power data (arbitrary for illustration)
time = np.linspace(0, 100, 1000)
P_excess = np.sin(time / 10) + 1  # Arbitrary excess power pattern
Q_excess = excess_heat_generation(time, P_excess)
print(f'Excess Heat Generated: {Q_excess:.2e} J')

# Example transmutation calculation
t = np.linspace(0, 100, 1000)
k = 0.1  # Arbitrary rate constant
y0 = 1e23  # Initial concentration
transmutation_result = odeint(transmutation_model, y0, t, args=(k,))
print(f'Transmutation result at final time: {transmutation_result[-1][0]:.2e}')

# Example glow discharge calculation
I = 1.0  # Current in Amperes
V = 100  # Voltage in Volts
P_glow = glow_discharge_power(I, V)
print(f'Glow Discharge Power: {P_glow:.2e} W')

# Example neutron production rate calculation
sigma = 1e-28  # Cross-section for the electron-proton reaction, m^2
rho_e = 1e28  # Electron density, m^-3
rho_p = 1e28  # Proton density, m^-3
E_e = 1e6  # Electron energy, eV
R_n = neutron_production_rate(sigma, rho_e, rho_p, E_e)
print(f'Neutron Production Rate: {R_n:.2e} neutrons/s')

# Example TSC fusion rate calculation
S = 1.0  # Symmetry factor
G = 1.0  # Gibbs free energy for TSC formation
T = 300  # System's temperature in Kelvin
R_TSC = tsc_fusion_rate(S, G, T)
print(f'TSC Fusion Rate: {R_TSC:.2e} reactions/s')

# Sonication simulation using FDTD
def sonication_simulation(nx, nt, dx, dt, source_position):
    """Simulate sonication using FDTD."""
    p = np.zeros(nx)  # Pressure field
    v = np.zeros(nx)  # Velocity field
    p[source_position] = 1  # Initial disturbance

    for t in range(nt):
        v[1:] += (dt / dx) * (p[1:] - p[:-1])
        p[:-1] += (dt / dx) * (v[1:] - v[:-1])
    return p

nx = 200  # Number of spatial steps
nt = 750  # Number of time steps
dx = 0.01  # Spatial step (m)
dt = dx / (2 * c)  # Time step (s)
source_position = nx // 2

p_final = sonication_simulation(nx, nt, dx, dt, source_position)

# Plotting initial lattice
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter([], [], [], c='r', s=100)
ax.set_xlim([0, n])
ax.set_ylim([0, n])
ax.set_zlim([0, n])
ax.set_title('LENR Event Simulation')

def update(frame, scat, decay_positions, decay_times):
    """Update function for animation."""
    decay_times -= frame
    active_indices = decay_times > 0
    scat._offsets3d = (decay_positions[active_indices, 0],
                       decay_positions[active_indices, 1],
                       decay_positions[active_indices, 2])
    return scat,

ani = FuncAnimation(fig, update, frames=np.arange(0, 20, 0.1),
                    fargs=(scat, decay_positions, decay_times), interval=50)

plt.show()

# Modeling transmutation decay channels
def neutron_capture(y, t, n_density, capture_rate):
    """Modeling neutron capture and decay."""
    N, D, He, gamma = y
    dNdt = -capture_rate * n_density * N
    dDdt = capture_rate * n_density * N - capture_rate * D
    dHedt = capture_rate * D
    dGammadt = capture_rate * D
    return [dNdt, dDdt, dHedt, dGammadt]

# Initial conditions for transmutation decay channels
y0 = [1e23, 0, 0, 0]  # Initial concentrations
t = np.linspace(0, 100, 1000)  # Time array
n_density = 1e28  # Neutron density
capture_rate = 1e-3  # Capture rate

# Solve ODE for neutron capture
sol = odeint(neutron_capture, y0, t, args=(n_density, capture_rate))

# Plot results for neutron capture and decay
plt.figure()
plt.plot(t, sol[:, 0], label='Neutron')
plt.plot(t, sol[:, 1], label='Deuterium')
plt.plot(t, sol[:, 2], label='Helium-3')
plt.plot(t, sol[:, 3], label='Gamma photons')
plt.xlabel('Time [s]')
plt.ylabel('Concentration')
plt.legend()
plt.title('Neutron Capture and Decay')
plt.show()

# Simulating cavitation using Rayleigh-Plesset equation
def rayleigh_plesset(t, R, P0, Pv, rho, gamma, mu, sigma):
    """Modeling bubble dynamics under sonication."""
    R, R_dot = R
    P = P0 - (2 * sigma / R) - (4 * mu * R_dot / R)
    P_bubble = Pv + (2 * sigma / R)
    R_ddot = (P - P_bubble) / (rho * R) - (3 / 2) * (R_dot ** 2 / R)
    return [R_dot, R_ddot]

# Initial conditions for Rayleigh-Plesset equation
R0 = [1e-6, 0]  # Initial radius and velocity
t_span = (0, 0.01)
t_eval = np.linspace(*t_span, 1000)
P0 = 101325  # Ambient pressure (Pa)
Pv = 2338  # Vapor pressure of water at 20°C (Pa)
rho = 1000  # Density of water (kg/m^3)
gamma = 1.4  # Polytropic index
mu = 0.001  # Viscosity of water (Pa·s)
sigma = 0.0728  # Surface tension of water (N/m)

# Solve the Rayleigh-Plesset equation
sol = solve_ivp(rayleigh_plesset, t_span, R0, args=(P0, Pv, rho, gamma, mu, sigma), t_eval=t_eval)

# Set up the figure and axis for animation
fig, ax = plt.subplots()
ax.set_xlim(t_span)
ax.set_ylim(0, np.max(sol.y[0]) * 1.1)
line, = ax.plot([], [], lw=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Bubble Radius [m]')
ax.set_title('Bubble Dynamics under Sonication')

# Initialization function for animation
def init():
    line.set_data([], [])
    return line,

# Update function for animation
def update(frame):
    line.set_data(sol.t[:frame], sol.y[0][:frame])
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(sol.t), init_func=init, blit=True)

# Save the animation (optional)
# ani.save('bubble_dynamics.mp4', writer='ffmpeg')

plt.show()
