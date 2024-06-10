import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Constants
k = 8.99e9  # Coulomb's constant in N m^2 C^-2
q = -1.6e-19  # Charge of an electron in Coulombs
n = 10  # Number of electrons per side of the cube
spacing = 1e-15  # Spacing in meters (1 fm)
total_energy_mev = 0  # Initialize total energy in MeV

# Generate coordinates for electrons in a cubic array
positions = [(x, y, z) for x in range(n) for y in range(n) for z in range(n)]

# Calculate the total energy
total_energy = 0
distances = []
energies = []

for i in range(len(positions)):
    for j in range(i + 1, len(positions)):
        r_ij = np.sqrt(np.sum((np.array(positions[i]) - np.array(positions[j])) ** 2)) * spacing
        energy_ij = k * (q * q) / r_ij
        total_energy += energy_ij
        distances.append(r_ij)
        energies.append(energy_ij)

# Convert energy from Joules to MeV
total_energy_mev = total_energy / 1.60218e-13

# Setup the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, n-1)
ax.set_ylim(0, n-1)
ax.set_zlim(0, n-1)

# Plot initial positions
scat = ax.scatter(*zip(*positions), c='blue', s=20)

# Decay constants and flight times
half_life_tritium = 12.32 * 365.25 * 24 * 3600  # 12.32 years in seconds
decay_constant_tritium = np.log(2) / half_life_tritium
flight_time_H4 = 1e-9
flight_time_H5 = 1e-12
flight_time_H6 = 1e-12
flight_time_H7 = 1e-12
flight_time_H8 = 1e-15

# Initial concentrations for isotopes
initial_concentration = {'H3': 1e23, 'H4': 0, 'H5': 0, 'H6': 0, 'H7': 0, 'H8': 0}

# Time array for tritium decay simulation
time_array = np.linspace(0, 10 * half_life_tritium, 1000)

# Function for tritium decay
def decay_tritium(y, t):
    H3, He3, e, nu = y
    dH3dt = -decay_constant_tritium * H3
    dHe3dt = decay_constant_tritium * H3
    de_dt = decay_constant_tritium * H3  # Beta particles
    dnu_dt = decay_constant_tritium * H3  # Electron antineutrinos
    return [dH3dt, dHe3dt, de_dt, dnu_dt]

# Initial conditions for tritium decay
initial_conditions_tritium = [initial_concentration['H3'], 0, 0, 0]

# Solve ODE for tritium decay
solution_tritium = odeint(decay_tritium, initial_conditions_tritium, time_array)

# Function to simulate short-lived isotopes
def simulate_unstable_isotopes(t, initial_concentration):
    concentrations = {
        'H4': initial_concentration['H4'] * np.exp(-t / flight_time_H4),
        'H5': initial_concentration['H5'] * np.exp(-t / flight_time_H5),
        'H6': initial_concentration['H6'] * np.exp(-t / flight_time_H6),
        'H7': initial_concentration['H7'] * np.exp(-t / flight_time_H7),
        'H8': initial_concentration['H8'] * np.exp(-t / flight_time_H8)
    }
    return concentrations

# Time array for short-lived isotopes
time_short = np.linspace(0, 1e-8, 100)  # Short time frame for unstable isotopes

# Simulate unstable isotopes
unstable_isotopes_concentration = simulate_unstable_isotopes(time_short, initial_concentration)

# Animation function
def update(frame):
    ax.cla()
    ax.set_xlim(0, n-1)
    ax.set_ylim(0, n-1)
    ax.set_zlim(0, n-1)
    ax.scatter(*zip(*positions), c='blue', s=20)
    
    if frame < len(distances):
        i, j = np.unravel_index(frame, (n*n*n, n*n*n))
        if i != j and i < j:
            x_values = [positions[i][0], positions[j][0]]
            y_values = [positions[i][1], positions[j][1]]
            z_values = [positions[i][2], positions[j][2]]
            ax.plot(x_values, y_values, z_values, 'r-', alpha=0.5)
            ax.text((x_values[0] + x_values[1]) / 2, (y_values[0] + y_values[1]) / 2, (z_values[0] + z_values[1]) / 2, f'{energies[frame]:.2e} J', fontsize=8)

# Create animation
ani = FuncAnimation(fig, update, frames=len(distances), interval=1, repeat=False)

# Plot results for tritium decay
plt.figure()
plt.plot(time_array, solution_tritium[:, 0], label='Tritium')
plt.plot(time_array, solution_tritium[:, 1], label='Helium-3')
plt.xlabel('Time (s)')
plt.ylabel('Concentration')
plt.legend()
plt.title('Tritium Decay')
plt.show()

# Plot results for unstable isotopes
plt.figure()
for isotope in ['H4', 'H5', 'H6', 'H7', 'H8']:
    plt.plot(time_short, unstable_isotopes_concentration[isotope], label=isotope)
plt.xlabel('Time (s)')
plt.ylabel('Concentration')
plt.legend()
plt.title('Decay of Unstable Isotopes')
plt.show()

# Show plot
plt.show()
