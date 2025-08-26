"""
OpenWave Equations
from: https://energywavetheory.com/equations/

Core Energy Wave Theory equations converted to Python.

This module exposes:
- energy_wave_equation: The fundamental Energy Wave Equation E = ρV(c/λ_l * A)²
"""

from constants import QWAVE_DENSITY as rho, QWAVE_SPEED as c, QWAVE_LENGTH as lambda_l, QWAVE_AMPLITUDE as A_l


def energy_wave_equation(volume, amplitude=None, wavelength=None):
    """
    Energy Wave Equation: E = ρV(c/λ_l * A)²
    
    The fundamental equation from which all EWT equations are derived.
    
    Args:
        volume (float): Volume V in m³
        amplitude (float, optional): Amplitude A in m. Defaults to QWAVE_AMPLITUDE
        wavelength (float, optional): Wavelength λ_l in m. Defaults to QWAVE_LENGTH
    
    Returns:
        float: Energy E in Joules
    """
    if amplitude is None:
        amplitude = A_l
    if wavelength is None:
        wavelength = lambda_l
    
    return rho * volume * (c / wavelength * amplitude) ** 2


def longitudinal_energy_equation(K):
    """
    Longitudinal Energy Equation (Particles): E_l(K) = (4πρK⁵A_l⁶c²/3λ_l³) * Σ(n=1 to K)[n³-(n-1)³]/n⁴
    
    Used to calculate the rest energy of particles.
    
    Args:
        K (int): Particle wave center count (dimensionless)
    
    Returns:
        float: Particle energy E_l in Joules
    """
    import numpy as np
    
    # Calculate the summation term
    n_values = np.arange(1, K + 1)
    summation = np.sum((n_values**3 - (n_values - 1)**3) / n_values**4)
    
    # Calculate the energy
    coefficient = (4 * np.pi * rho * (K**5) * (A_l**6) * (c**2)) / (3 * (lambda_l**3))
    energy = coefficient * summation
    
    return energy


print(energy_wave_equation(1.0))  # Example usage with 1 m³ volume
print(longitudinal_energy_equation(1))  # Example usage with K=1
print(longitudinal_energy_equation(10))  # Example usage with K=10
