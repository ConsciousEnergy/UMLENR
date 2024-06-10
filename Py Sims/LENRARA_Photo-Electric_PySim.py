#Photo-Electric Effects in LENR
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, e, epsilon_0, m_e, hbar

def photoelectric_effect(wavelength, intensity, scale='quantum'):
    """
    Simulate electron densities and momentum in the photoelectric effect in hydrogen
    from the Planck scale up to the molecular scale.
    
    Parameters:
    wavelength (float): The wavelength of the incident photon in meters.
    intensity (float): The intensity of the incident light in W/m^2.
    scale (str): The scale of the simulation ('quantum' or 'classical').

    Returns:
    electron_density (float): The density of ejected electrons.
    electron_momentum (float): The momentum of ejected electrons.
    """
    # Constants
    phi = 13.6 * e  # Work function of hydrogen in Joules
    h_nu = h * c / wavelength  # Energy of the incident photon

    if scale == 'quantum':
        if h_nu >= phi:
            # Quantum photoelectric effect
            electron_energy = h_nu - phi
            electron_momentum = np.sqrt(2 * m_e * electron_energy)
            electron_density = intensity / (h_nu * e)
        else:
            # Below threshold, no electrons ejected
            electron_energy = 0
            electron_momentum = 0
            electron_density = 0

    elif scale == 'classical':
        # Classical approximation (Drude model)
        electron_density = intensity / (h_nu * e)
        electron_momentum = np.sqrt(2 * m_e * h_nu)
    
    return electron_density, electron_momentum

def plot_results(wavelengths, intensities, scale='quantum'):
    """
    Plot the electron densities and momentum for a range of wavelengths and intensities.
    """
    electron_densities = []
    electron_momenta = []

    for wavelength in wavelengths:
        for intensity in intensities:
            density, momentum = photoelectric_effect(wavelength, intensity, scale)
            electron_densities.append(density)
            electron_momenta.append(momentum)

    electron_densities = np.array(electron_densities).reshape(len(wavelengths), len(intensities))
    electron_momenta = np.array(electron_momenta).reshape(len(wavelengths), len(intensities))

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot electron densities
    im1 = ax[0].imshow(electron_densities, extent=[min(intensities), max(intensities), min(wavelengths), max(wavelengths)], aspect='auto', origin='lower')
    ax[0].set_title('Electron Densities')
    ax[0].set_xlabel('Intensity (W/m^2)')
    ax[0].set_ylabel('Wavelength (m)')
    fig.colorbar(im1, ax=ax[0], orientation='vertical')

    # Plot electron momenta
    im2 = ax[1].imshow(electron_momenta, extent=[min(intensities), max(intensities), min(wavelengths), max(wavelengths)], aspect='auto', origin='lower')
    ax[1].set_title('Electron Momenta')
    ax[1].set_xlabel('Intensity (W/m^2)')
    ax[1].set_ylabel('Wavelength (m)')
    fig.colorbar(im2, ax=ax[1], orientation='vertical')

    plt.tight_layout()
    plt.show()

# Define a range of wavelengths and intensities
wavelengths = np.linspace(1e-10, 1e-6, 100)  # From Planck scale to molecular scale
intensities = np.linspace(1e1, 1e5, 100)     # Range of light intensities

# Plot results for quantum scale
plot_results(wavelengths, intensities, scale='quantum')

# Plot results for classical scale
plot_results(wavelengths, intensities, scale='classical')
