"""Common utility functions and unit conversion constants for OpenWave.

This module provides:
- Energy unit conversion constants (eV, J, kWh, cal)
- Mathematical utility functions (rounding operations)

Unit Conversion Constants:
    Energy conversions between joules (J), electronvolts (eV),
    kilowatt-hours (kWh), and calories (cal).

Functions:
    round_to_nearest_odd: Convert float to nearest odd integer.

Examples:
    >>> from openwave.common import utils
    >>> energy_j = 1.0 * utils.EV2J  # Convert 1 eV to joules
    >>> energy_ev = energy_j * utils.J2EV  # Convert back to eV
    >>> utils.round_to_nearest_odd(4.7)  # Returns 5
    5
"""

# ================================================================
# Unit Conversion - ENERGY
# ================================================================
# For energy-frequency relation: E = h * f
# Example: 1 eV photon → f = E/h ≈ 2.417989 × 10^14 Hz
EV2J = 1.602176634e-19  # J, 1 eV in joules (exact, same as elementary charge)
J2EV = 1 / EV2J  # eV, 1 J in electronvolts
KWH2J = 3.6e6  # J, per kilowatt-hour, kWh
J2KWH = 1 / KWH2J  # kWh, per joule
CAL2J = 4.184  # J, per thermochemical calorie, cal
J2CAL = 1 / CAL2J  # cal, per joule


# ================================================================
# Math Utilities
# ================================================================
# Round to nearest odd integer
def round_to_nearest_odd(value):
    """Convert float to nearest odd integer."""
    rounded = round(value)
    return rounded if rounded % 2 == 1 else (rounded + 1 if value >= rounded else rounded - 1)
