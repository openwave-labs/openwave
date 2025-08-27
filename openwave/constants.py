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

# =====================
# Classical constants
# =====================
PLANCK_LENGTH = l_p = 1.616255e-35  # m, Planck length
PLANCK_TIME = t_p = 5.391247e-44  # s, Planck time
PLANCK_MASS = m_p = 2.176434e-8  # kg, Planck mass
PLANCK_CHARGE = q_p = 1.875545956e-18  # m, Planck charge
PLANCK_CONSTANT = h = 6.62607015e-34  # J·s, Planck constant — exact definition

FINE_STRUCTURE = alpha = 7.2973525693e-3  # fine-structure constant (dimensionless)
ELECTRIC_CONSTANT = epsilon_0 = 8.8541878128e-12  # F/m, vacuum permittivity
MAGNETIC_CONSTANT = mu_0 = 1.25663706212e-6  # N, vacuum permeability
BOHR_RADIUS = a_0 = 5.29177210903e-11  # m, Bohr radius
ELEMENTARY_CHARGE = 1.6022e-19  # m, The elementary charge from CODATA values
COULOMB_CONSTANT = k = 8.9875517923e9  # N·m^2 / C^2, Coulomb's constant

PI = pi = 3.14159265358979323846  # pi (dimensionless)
E = e = 2.71828182845904523536  # euler's number (dimensionless)
PHI = phi = (1 + 5**0.5) / 2  # golden ratio (dimensionless)

# DELTA : amplitude factor (dimensionless) — user-supplied per system/transition
# K     : particle wave center count (dimensionless)
# Q     : particle count in a group (dimensionless)


# =====================
# Conversion constants
# =====================
J_ELECTRON_VOLT = eV = 1.602176634e-19  # J, per electron-volt
J_KILOWATT_HOUR = kWh = 3.6e6  # J, per kilowatt-hour
J_CALORIE = cal = 4.184  # J, per thermochemical calorie


# =====================
# Unit converters
# =====================
def J_to_eV(E_J: float) -> float:
    """Convert joules to electron-volts."""
    return E_J / eV


def eV_to_J(E_eV: float) -> float:
    """Convert electron-volts to joules."""
    return E_eV * eV


def J_to_kWh(E_J: float) -> float:
    """Convert joules to kilowatt-hours."""
    return E_J / kWh


def kWh_to_J(E_kWh: float) -> float:
    """Convert kilowatt-hours to joules."""
    return E_kWh * kWh
