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

PI = 3.14159265358979323846  # pi (dimensionless)
E = 2.71828182845904523536  # euler's number (dimensionless)
PHI = (1 + 5**0.5) / 2  # golden ratio (dimensionless)

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
