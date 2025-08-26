"""
OpenWave Equations
from: https://energywavetheory.com/equations/

Core Energy Wave Theory equations converted to Python.

This module exposes:
- energy_wave_equation: The fundamental Energy Wave Equation E = ρV(c/λ_l * A)²
"""

import config
import quantum_wave

print(quantum_wave.LENGTH)
#print(constants.QuantumWave.LENGTH)


# TODO: create Neutrino class
NEUTRINO_ENERGY = Ev = 3.8280e-19                 # J, neutrino "seed" energy used by EWT (~ 2.39 eV)

# TODO: create Electron class
ELECTRON_ENERGY = Ee = 8.1871e-14                 # J, electron rest energy (~ 0.511 MeV)
ELECTRON_K = Ke = 10                         # electron wave center count (dimensionless)
ELECTRON_RADIUS = re = 2.8179403262e-15           # m, electron classical radius
ELECTRON_OUTER_SHELL = Oe = 2.138743820                # electron outer shell multiplier (dimensionless)
ELECTRON_ORBITAL_G = g_lambda = 0.9873318320         # electron orbital g-factor (dimensionless)
ELECTRON_SPIN_G = g_A = 0.9826905018              # electron spin g-factor (dimensionless)

# TODO: create Proton class
PROTON_ENERGY = Ep = 1.5033e-10                 # J, CODATA proton rest energy (~ 938.272 MeV)
PROTON_K = Kp = 44               # proton wave center count (dimensionless)
PROTON_ORBITAL_G = g_p = 0.9898125300              # proton orbital g-factor (dimensionless)





def longitudinal_energy_equation(K):
    """
    Longitudinal Energy Equation (Particles): E_l(K) = (4πρK⁵A_l⁶c²/3λ_l³) * Σ(n=1 to K)[n³-(n-1)³]/n⁴
    
    Used to calculate the rest energy of particles.
    
    Args:
        K (int): Particle wave center count (dimensionless)
    
    Returns:
        float: Particle energy E_l in Joules
    
    Raises:
        ValueError: If K is not a positive integer
        TypeError: If K is not an integer
    """
    import numpy as np
    
    if not isinstance(K, int):
        raise TypeError("K must be an integer")
    if K <= 0:
        raise ValueError("K must be a positive integer")
    
    # Calculate the summation term
    n_values = np.arange(1, K + 1)
    summation = np.sum((n_values**3 - (n_values - 1)**3) / n_values**4)
    
    # Calculate the energy
    coefficient = (4 * np.pi * quantum_wave.DENSITY * (K**5) * (quantum_wave.AMPLITUDE**6) * (quantum_wave.SPEED**2)) / (3 * (quantum_wave.LENGTH**3))
    energy = coefficient * summation
    
    return energy


if __name__ == "__main__":
    # Example usage
    print(f"Longitudinal energy for K=1: {longitudinal_energy_equation(1):.2e} J")
    print(f"Longitudinal energy for K=10: {longitudinal_energy_equation(10):.2e} J")
    print(f"Longitudinal energy for K=44: {longitudinal_energy_equation(44):.2e} J")
