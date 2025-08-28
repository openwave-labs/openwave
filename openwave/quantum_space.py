import numpy as np
import config
import constants
import quantum_wave
import matter


DENSITY = 3.859764540e22  # kg / m^3, quantum-space density (aether medium, rho)


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
    numerator = (
        constants.PLANCK_CONSTANT
        * 9
        * quantum_wave.LENGTH**3
        * (1 / matter.Electron.ORBITAL_G)
    )

    denominator = (
        32
        * np.pi
        * matter.Electron.K**11
        * quantum_wave.AMPLITUDE**7
        * quantum_wave.SPEED
        * matter.Electron.OUTER_SHELL
    )

    calculated_density = numerator / denominator

    # Return the calculated value to show the relationship
    return calculated_density


if __name__ == "__main__":
    # Example usage
    print(f"Width {config.screen_width}, Height {config.screen_height}")

    # Wave constants derivations smoke-test
    print("\nWave Constants Derivations smoke-test:")

    derived_density = density_derivation()
    print(f"Derived Density: {derived_density:.9e} kg/m³")
    print(f"Stored DENSITY constant: {DENSITY:.9e} kg/m³")
