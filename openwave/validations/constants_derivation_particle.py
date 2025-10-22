"""
Energy Wave Theory (EWT) constants module.

This module provides validations of fundamental constants for Energy Wave Theory simulations,
sourced from https://energywavetheory.com/equations/

Derivation Functions:
- Particle constant derivations (outer shell, g-factors)

All values use SI units (kg, m, s) for consistency.
"""

import numpy as np

import openwave.common.constants as constants


# ================================================================
# Derivations Particle Constants
# ================================================================


def electron_outer_shell_derivation():
    """
    Particle Constant - Electron Outer Shell Multiplier Derivation
    Electron outer shell multiplier is a constant for readability replacing the summation
    in the electron's particle energy.

    Oe = Σ(n=1 to Ke) [n³ - (n-1)³] / n⁴

    Returns:
        float: Electron outer shell multiplier (dimensionless)
    """
    outer_shell = 0
    for n in range(1, constants.ELECTRON_K + 1):
        outer_shell += (n**3 - (n - 1) ** 3) / n**4
    return outer_shell


def electron_orbital_g_derivation():
    """
    Particle Constant - Electron Orbital G-Factor Derivation
    Electron orbital g-factor is set to the well-measured classical electron radius.

    gλ = {re} * (1 / (Ke² * λl))

    Note: The derivation of this constant and the wavelength constant is circular.
    The final value was determined through iteration until all constants resolved correctly.

    Returns:
        float: Electron orbital g-factor (dimensionless)
    """
    return constants.ELECTRON_RADIUS * (1 / (constants.ELECTRON_K**2 * constants.EWAVE_LENGTH))


def electron_spin_g_derivation():
    """
    Particle Constant - Electron Spin G-Factor Derivation
    Electron spin g-factor is set to the Planck charge.

    gA = {qP^-1} * (2 * Al)

    Where qP is the Planck charge.

    Returns:
        float: Electron spin g-factor (dimensionless)
    """
    # From the document: gA = {qP^-1} * 2 * Al
    # This means: gA = (2 * Al) / qP
    return (2 * constants.EWAVE_AMPLITUDE) / constants.PLANCK_CHARGE


def proton_orbital_g_derivation():
    """
    Particle Constant - Proton Orbital G-Factor Derivation
    Proton orbital g-factor is set to proton's mass.

    gp = {mp^-1} * (4πρ * Ke^8 * Al^6 * Oe) / (9 * λl^3) * √(λl / Al)

    Where mp is the proton mass.

    Note: As stated in the EWT documentation, the derivation of g-factors involves
    iterative refinement until all constants resolve correctly.

    Returns:
        float: Proton orbital g-factor (dimensionless)
    """
    # Calculate based on the formula
    numerator = (
        4
        * np.pi
        * constants.MEDIUM_DENSITY
        * (constants.ELECTRON_K**8)
        * (constants.EWAVE_AMPLITUDE**6)
        * constants.ELECTRON_OUTER_SHELL
    )
    denominator = 9 * (constants.EWAVE_LENGTH**3)

    # The full formula with the square root term
    g_factor = (
        (1 / constants.PROTON_MASS)
        * (numerator / denominator)
        * np.sqrt(constants.EWAVE_LENGTH / constants.EWAVE_AMPLITUDE)
    )

    # TODO: This shows the mathematical relationship. The exact value requires
    # iterative refinement as mentioned in the EWT documentation
    return g_factor


if __name__ == "__main__":
    # Constants smoke-tests
    print("\n===============================")
    print("CONSTANTS SMOKE-TESTS")
    print("===============================")

    print("PARTICLE CONSTANTS DERIVATIONS")

    print("\nELECTRON OUTER SHELL MULTIPLIER")
    print(f"Derived: {electron_outer_shell_derivation():.9f}")
    print(f"Stored : {constants.ELECTRON_OUTER_SHELL:.9f}")

    print("\nELECTRON ORBITAL G-FACTOR")
    print(f"Derived: {electron_orbital_g_derivation():.10f}")
    print(f"Stored : {constants.ELECTRON_ORBITAL_G:.10f}")

    print("\nELECTRON SPIN G-FACTOR")
    print(f"Derived: {electron_spin_g_derivation():.10f}")
    print(f"Stored : {constants.ELECTRON_SPIN_G:.10f}")

    print("\nPROTON ORBITAL G-FACTOR")
    print(f"Derived: {proton_orbital_g_derivation():.10f}")
    print(f"Stored : {constants.PROTON_ORBITAL_G:.10f}")
    print("_______________________________")
