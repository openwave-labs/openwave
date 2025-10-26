"""
Energy Wave Theory (EWT) constants module.

This module provides validations of fundamental constants for Energy Wave Theory simulations,
sourced from https://energywavetheory.com/equations/

Derivation Functions:
- Wave constant derivations (density, wavelength, amplitude)

All values use SI units (kg, m, s) for consistency.
"""

import numpy as np

import openwave.common.constants as constants


# ================================================================
# Derivations Wave Constants
# ================================================================


def density_derivation_wave():
    """
    Wave Constant - Density Derivation
    Density is set to the well-measured Planck constant and using wavelength
    calculated from wavelength_derivation.

    ρ = {h} * (9λl^3) / (32π * K^11e * A^7l * c * Oe) * gλ^-1

    Returns:
        float: Medium Density in kg/m³
    """
    # Direct calculation based on the documented formula
    numerator = (
        constants.PLANCK_CONSTANT
        * 9
        * constants.EWAVE_LENGTH**3
        * (1 / constants.ELECTRON_ORBITAL_G)
    )
    denominator = (
        32
        * np.pi
        * constants.ELECTRON_K**11
        * constants.EWAVE_AMPLITUDE**7
        * constants.EWAVE_SPEED
        * constants.ELECTRON_OUTER_SHELL
    )
    calculated_density = numerator / denominator
    # Return the calculated value to show the relationship
    return calculated_density


def density_derivation_hydrogen():
    """
    Wave Constant - Density Derivation (Hydrogen-based)
    Based on Eq. 1.1 from "Relationship of the Speed of Light to Aether Density"

    ρ = mₚ / (4/3 πrₕ³)

    Where:
        mₚ = Planck mass
        rₕ = Hydrogen 1s radius (Bohr radius)

    Returns:
        float: Medium Density in kg/m³
    """
    # Direct calculation based on Eq. 1.1
    calculated_density = constants.PLANCK_MASS / ((4 / 3) * np.pi * constants.BOHR_RADIUS**3)

    # Return the calculated value to show the relationship
    return calculated_density


def density_derivation_electron():
    """
    Wave Constant - Density Derivation (Electron-based)
    Based on Eq. 1.2 from "Relationship of the Speed of Light to Aether Density"

    ρ = μ₀ / (4/3 πrₑ²)

    Where:
        μ₀ = Magnetic constant
        rₑ = Electron classical radius

    Note: rₑ² appears in the denominator instead of rₑ³ due to linear density considerations

    Returns:
        float: Medium Density in kg/m³
    """
    # Direct calculation based on Eq. 1.2
    calculated_density = constants.MAGNETIC_CONSTANT / (
        (4 / 3) * np.pi * constants.ELECTRON_RADIUS**2
    )

    # Return the calculated value to show the relationship
    return calculated_density


def density_derivation_planck():
    """
    Wave Constant - Density Derivation (Planck/Photon-based)
    Based on Eq. 1.3 from "Relationship of the Speed of Light to Aether Density"

    ρ = (2h / qₚ²c) × (1 / (4/3 πrₑ²))

    Where:
        h = Planck constant
        qₚ = Planck charge
        c = Speed of light (wave velocity)
        rₑ = Electron classical radius

    Returns:
        float: Medium Density in kg/m³
    """
    # Direct calculation based on Eq. 1.3
    calculated_density = (
        (2 * constants.PLANCK_CONSTANT) / (constants.PLANCK_CHARGE**2 * constants.EWAVE_SPEED)
    ) * (1 / ((4 / 3) * np.pi * constants.ELECTRON_RADIUS**2))

    # Return the calculated value to show the relationship
    return calculated_density


def wavelength_derivation():
    """
    Wave Constant - Wavelength Derivation
    Wavelength (longitudinal) is set to the well-measured classical electron radius.

    λl = {re} * (1/Ke²) * gλ^-1

    Returns:
        float: Wavelength (longitudinal) in meters
    """
    return (
        constants.ELECTRON_RADIUS
        * (1 / constants.ELECTRON_K**2)
        * (1 / constants.ELECTRON_ORBITAL_G)
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
        * (3 * np.pi * constants.EWAVE_LENGTH)
        / (4 * constants.ELECTRON_K**4)
    )


if __name__ == "__main__":
    # Constants smoke-tests
    print("\n===============================")
    print("CONSTANTS SMOKE-TESTS")
    print("===============================")

    print("WAVE CONSTANTS DERIVATIONS")

    print("\nSPACETIME DENSITY")
    print(f"Derived Wave: {density_derivation_wave():.9e} kg/m³")
    print(f"Derived Hydrogen: {density_derivation_hydrogen():.9e} kg/m³")
    print(f"Derived Electron: {density_derivation_electron():.9e} kg/m³")
    print(f"Derived Planck: {density_derivation_planck():.9e} kg/m³")
    print(f"Stored : {constants.MEDIUM_DENSITY:.9e} kg/m³")

    print("\nENERGY-WAVE LENGTH")
    print(f"Derived: {wavelength_derivation():.9e} m")
    print(f"Stored : {constants.EWAVE_LENGTH:.9e} m")

    print("\nENERGY-WAVE AMPLITUDE")
    print(f"Derived: {amplitude_derivation():.9e} m")
    print(f"Stored : {constants.EWAVE_AMPLITUDE:.9e} m")
    print("_______________________________")
