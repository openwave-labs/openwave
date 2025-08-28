import numpy as np
import constants
import quantum_space
import matter

LENGTH = 2.854096501e-17  # m, quantum-wave length
AMPLITUDE = 9.215405708e-19  # m, quantum-wave amplitude (equilibrium-to-peak)
SPEED = 299792458  # m / s, quantum-wave velocity (speed of light, c)


def energy_wave_equation(volume):
    """
    Energy Wave Equation: E = ρV(c/λl * A)²
    The fundamental equation from which all EWT equations are derived.
    Args:
        volume (float): Volume V in m³
    Returns:
        float: Energy E in Joules
    """

    return quantum_space.DENSITY * volume * (SPEED / LENGTH * AMPLITUDE) ** 2


def wavelength_derivation():
    """
    Wave Constant - Wavelength Derivation
    Wavelength (longitudinal) is set to the well-measured classical electron radius.
    λl = {re} * (1/K²e) * g_λ^-1
    Returns:
        float: Wavelength (longitudinal) in meters
    """
    return (
        matter.Electron.RADIUS
        * (1 / matter.Electron.K**2)
        * (1 / matter.Electron.ORBITAL_G)
    )


def amplitude_derivation():
    """
    Wave Constant - Amplitude Derivation
    Amplitude (longitudinal) is set to the well-measured fine structure constant
    and using wavelength calculated from wavelength_derivation.
    Al = {αe^-1} * (3πλl) / (4K^4e)
    Returns:
        float: Amplitude (longitudinal) in meters
    """

    return (
        (1 / constants.FINE_STRUCTURE)
        * (3 * np.pi * LENGTH)
        / (4 * matter.Electron.K**4)
    )


if __name__ == "__main__":
    # Example usage
    print(f"Energy for 1 m³ volume: {energy_wave_equation(1):.2e} J")

    # Wave constants derivations smoke-test
    print("\nWave Constants Derivations smoke-test:")

    derived_wavelength = wavelength_derivation()
    print(f"Derived Wavelength: {derived_wavelength:.9e} m")
    print(f"Stored LENGTH constant: {LENGTH:.9e} m")

    derived_amplitude = amplitude_derivation()
    print(f"\nDerived Amplitude: {derived_amplitude:.9e} m")
    print(f"Stored AMPLITUDE constant: {AMPLITUDE:.9e} m")
