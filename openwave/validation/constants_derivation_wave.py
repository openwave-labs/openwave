"""
Energy Wave Theory (EWT) constants module.

This module provides validations of fundamental constants for Energy Wave Theory simulations,
sourced from https://energywavetheory.com/equations/

Derivation Functions:
- Wave constant derivations (density, wavelength, amplitude)

All values use SI units (kg, m, s) for consistency.
"""

import numpy as np

import openwave.core.constants as constants


# ================================================================
# Derivations Wave Constants
# ================================================================


def density_derivation():
    """
    Wave Constant - Density Derivation
    Density is set to the well-measured Planck constant and using wavelength
    calculated from wavelength_derivation.

    ρ = {h} * (9λl^3) / (32π * K^11e * A^7l * c * Oe) * g_λ^-1

    Returns:
        float: Density (aether) in kg/m³
    """
    # Direct calculation based on the documented formula
    numerator = (
        constants.PLANCK_CONSTANT
        * 9
        * constants.QWAVE_LENGTH**3
        * (1 / constants.ELECTRON_ORBITAL_G)
    )
    denominator = (
        32
        * np.pi
        * constants.ELECTRON_K**11
        * constants.QWAVE_AMPLITUDE**7
        * constants.QWAVE_SPEED
        * constants.ELECTRON_OUTER_SHELL
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
        * (3 * np.pi * constants.QWAVE_LENGTH)
        / (4 * constants.ELECTRON_K**4)
    )


if __name__ == "__main__":
    # Constants smoke-tests
    print("\n===============================")
    print("CONSTANTS SMOKE-TESTS")
    print("===============================")

    print("WAVE CONSTANTS DERIVATIONS")

    print("\nQUANTUM SPACE DENSITY")
    print(f"Derived: {density_derivation():.9e} kg/m³")
    print(f"Stored : {constants.QSPACE_DENSITY:.9e} kg/m³")

    print("\nQUANTUM WAVE LENGTH")
    print(f"Derived: {wavelength_derivation():.9e} m")
    print(f"Stored : {constants.QWAVE_LENGTH:.9e} m")

    print("\nQUANTUM WAVE AMPLITUDE")
    print(f"Derived: {amplitude_derivation():.9e} m")
    print(f"Stored : {constants.QWAVE_AMPLITUDE:.9e} m")
    print("_______________________________")
