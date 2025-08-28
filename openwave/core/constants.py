"""
OpenWave Constants
from: https://energywavetheory.com/equations/

This module provides fundamental constants for Energy Wave Theory (EWT) simulations:

- Classical physics constants (Planck, electromagnetic, atomic)
- Mathematical constants (pi, e, phi)
- Energy conversion constants and utility functions
- All values use SI units (kg, m, s) for consistency

Constants are organized into logical groups with descriptive comments
and alternative variable names for different naming conventions.
"""

import numpy as np

import openwave.core.config as config


# =====================
# QUANTUM SPACE (AKA: AKASHA @yoga, WUJI @taoism, AETHER @ancient)
# =====================
QSPACE_DENSITY = 3.859764540e22  # kg / m^3, quantum-space density (aether medium, rho)


# =====================
# QUANTUM WAVE (AKA: PRANA @yoga, QI @taoism, THE FORCE @starwars)
# =====================
QWAVE_LENGTH = 2.854096501e-17  # m, quantum-wave length
QWAVE_AMPLITUDE = 9.215405708e-19  # m, quantum-wave amplitude (equilibrium-to-peak)
QWAVE_SPEED = 299792458  # m / s, quantum-wave velocity (speed of light, c)


# =====================
# Neutrino particle
# =====================
NEUTRINO_ENERGY = 3.8280e-19  # J, neutrino "seed" energy used by EWT (~ 2.39 eV)


# =====================
# Electron particle
# =====================
ELECTRON_ENERGY = 8.1871e-14  # J, electron rest energy (~ 0.511 MeV)
ELECTRON_K = 10  # electron wave center count (dimensionless)
ELECTRON_RADIUS = 2.8179403262e-15  # m, electron classical radius
ELECTRON_OUTER_SHELL = 2.138743820  # electron outer shell multiplier
ELECTRON_ORBITAL_G = 0.9873318320  # electron orbital g-factor
ELECTRON_SPIN_G = 0.9826905018  # electron spin g-factor (dimensionless)


# =====================
#  Proton particle
# =====================
PROTON_ENERGY = 1.5033e-10  # J, CODATA proton rest energy (~ 938.272 MeV)
PROTON_K = 44  # proton wave center count (dimensionless)
PROTON_ORBITAL_G = 0.9898125300  # proton orbital g-factor (dimensionless)


# =====================
# Classical constants
# =====================
PLANCK_LENGTH = 1.616255e-35  # m, Planck length
PLANCK_TIME = 5.391247e-44  # s, Planck time
PLANCK_MASS = 2.176434e-8  # kg, Planck mass
PLANCK_CHARGE = 1.875545956e-18  # m, Planck charge
PLANCK_CONSTANT = 6.62607015e-34  # J·s, Planck constant — exact definition

FINE_STRUCTURE = 7.2973525693e-3  # fine-structure constant, alpha
ELECTRIC_CONSTANT = 8.8541878128e-12  # F/m, vacuum permittivity, epsilon_0
MAGNETIC_CONSTANT = 1.25663706212e-6  # N, vacuum permeability, mu_0
BOHR_RADIUS = 5.29177210903e-11  # m, Bohr radius, a_0
ELEMENTARY_CHARGE = 1.6022e-19  # m, The elementary charge from CODATA values
COULOMB_CONSTANT = 8.9875517923e9  # N·m^2 / C^2, Coulomb's constant, k

# DELTA : amplitude factor (dimensionless) — user-supplied per system/transition
# K     : particle wave center count (dimensionless)
# Q     : particle count in a group (dimensionless)


# =====================
# Conversion constants
# =====================
EV2J = 1.602176634e-19  # J, per electron-volt, eV
KWH2J = 3.6e6  # J, per kilowatt-hour, kWh
CAL2J = 4.184  # J, per thermochemical calorie, cal


# =====================
# Unit converters
# =====================
def J_to_eV(energy_J: float) -> float:
    """Convert joules to electron-volts."""
    return energy_J / EV2J


def eV_to_J(energy_eV: float) -> float:
    """Convert electron-volts to joules."""
    return energy_eV * EV2J


def J_to_kWh(energy_J: float) -> float:
    """Convert joules to kilowatt-hours."""
    return energy_J / KWH2J


def kWh_to_J(energy_kWh: float) -> float:
    """Convert kilowatt-hours to joules."""
    return energy_kWh * KWH2J


# =====================
# ENERGY WAVE EQUATION
# =====================
def energy_wave_equation(volume):
    """
    Energy Wave Equation: E = ρV(c/λl * A)²
    The fundamental equation from which all EWT equations are derived.
    Args:
        volume (float): Volume V in m³
    Returns:
        float: Energy E in Joules
    """
    return QSPACE_DENSITY * volume * (QWAVE_SPEED / QWAVE_LENGTH * QWAVE_AMPLITUDE) ** 2


# =====================
# Derivations
# =====================
def density_derivation():
    """
    Wave Constant - Density Derivation
    Density is set to the well-measured Planck constant and using wavelength
    calculated from wavelength_derivation.
    From the EWT documentation:
    ρ = {h} * (9λl^3) / (32π * K^11e * A^7l * c * Oe) * g_λ^-1
    Returns:
        float: Density (aether) in kg/m³
    """
    # Direct calculation based on the documented formula
    numerator = PLANCK_CONSTANT * 9 * QWAVE_LENGTH**3 * (1 / ELECTRON_ORBITAL_G)
    denominator = (
        32
        * np.pi
        * ELECTRON_K**11
        * QWAVE_AMPLITUDE**7
        * QWAVE_SPEED
        * ELECTRON_OUTER_SHELL
    )
    calculated_density = numerator / denominator
    # Return the calculated value to show the relationship
    return calculated_density


def wavelength_derivation():
    """
    Wave Constant - Wavelength Derivation
    Wavelength (longitudinal) is set to the well-measured classical electron radius.
    λl = {re} * (1/K²e) * g_λ^-1
    Returns:
        float: Wavelength (longitudinal) in meters
    """
    return ELECTRON_RADIUS * (1 / ELECTRON_K**2) * (1 / ELECTRON_ORBITAL_G)


def amplitude_derivation():
    """
    Wave Constant - Amplitude Derivation
    Amplitude (longitudinal) is set to the well-measured fine structure constant
    and using wavelength calculated from wavelength_derivation.
    Al = {αe^-1} * (3πλl) / (4K^4e)
    Returns:
        float: Amplitude (longitudinal) in meters
    """
    return (1 / FINE_STRUCTURE) * (3 * np.pi * QWAVE_LENGTH) / (4 * ELECTRON_K**4)


def particle_energy(K):
    """
    Longitudinal Energy Equation (Particles): E_l(K) = (4πρK⁵A_l⁶c²/3λ_l³) * Σ(n=1 to K)[n³-(n-1)³]/n⁴
    Used to calculate the rest energy of particles.
    Args:
        K (int): Particle wave center count (dimensionless)
    Returns:
        float: Particle energy E_l in Joules
    """
    # Calculate the summation term
    n_values = np.arange(1, K + 1)
    summation = np.sum((n_values**3 - (n_values - 1) ** 3) / n_values**4)
    # Calculate the energy
    coefficient = (
        4 * np.pi * QSPACE_DENSITY * (K**5) * (QWAVE_AMPLITUDE**6) * (QWAVE_SPEED**2)
    ) / (3 * (QWAVE_LENGTH**3))
    energy = coefficient * summation
    return energy


if __name__ == "__main__":
    # Smoke-tests
    print("\n===============================")
    print("SMOKE-TESTS")
    print("===============================")

    print("SCREEN CONFIGURATION")
    print(f"Width {config.SCREEN_WIDTH}px, Height {config.SCREEN_HEIGHT}px")

    print("\n_______________________________")
    print("ENERGY WAVE EQUATION")
    print(f"1 m³ of vacuum: {energy_wave_equation(1):.2e} J")

    print("\n_______________________________")
    print("WAVE CONSTANTS DERIVATIONS")
    derived_density = density_derivation()
    print("\nQUANTUM SPACE DENSITY")
    print(f"Derived: {derived_density:.9e} kg/m³")
    print(f"Stored : {QSPACE_DENSITY:.9e} kg/m³")

    derived_wavelength = wavelength_derivation()
    print("\nQUANTUM WAVE LENGTH")
    print(f"Derived: {derived_wavelength:.9e} m")
    print(f"Stored : {QWAVE_LENGTH:.9e} m")

    derived_amplitude = amplitude_derivation()
    print("\nQUANTUM WAVE AMPLITUDE")
    print(f"Derived: {derived_amplitude:.9e} m")
    print(f"Stored : {QWAVE_AMPLITUDE:.9e} m")

    print("\n_______________________________")
    print("PARTICLE ENERGY")
    print(f"NEUTRINO (K=1): {particle_energy(1):.2e} J")
    print(f"ELECTRON (K=10): {particle_energy(10):.2e} J")
    print(f"PROTON (K=44): {particle_energy(44):.2e} J")
    print("_______________________________")
