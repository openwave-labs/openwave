import numpy as np

import openwave.core.constants as constants

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
    return (
        constants.QSPACE_DENSITY
        * volume
        * (constants.QWAVE_SPEED / constants.QWAVE_LENGTH * constants.QWAVE_AMPLITUDE)
        ** 2
    )


# =====================
# Particle Energy
# =====================
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
        4
        * np.pi
        * constants.QSPACE_DENSITY
        * (K**5)
        * (constants.QWAVE_AMPLITUDE**6)
        * (constants.QWAVE_SPEED**2)
    ) / (3 * (constants.QWAVE_LENGTH**3))
    energy = coefficient * summation
    return energy


if __name__ == "__main__":
    print("\n_______________________________")
    print("ENERGY WAVE EQUATION")
    print(f"1 m³ of vacuum: {energy_wave_equation(1):.2e} J")

    print("\n_______________________________")
    print("PARTICLE ENERGY")
    print(f"NEUTRINO (K=1): {particle_energy(1):.2e} J")
    print(f"ELECTRON (K=10): {particle_energy(10):.2e} J")
    print(f"PROTON (K=44): {particle_energy(44):.2e} J")
    print("_______________________________")
