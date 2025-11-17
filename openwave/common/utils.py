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
