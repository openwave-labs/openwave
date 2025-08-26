"""
OpenWave Constants
from: https://energywavetheory.com/equations/

This module exposes:
- Wave constants (values and units per the EWT "wave constants" set)
- Particle/electron-specific constants used throughout EWT
- Common classical constants used alongside wave constants
"""

from math import pi, e
from typing import Tuple

# =====================
# Quantum Wave constants (EWT)
# =====================
# Units are SI (kg, m, s). Values follow the EWT wave-constants view.
QWAVE_LENGTH = lambda_l = 2.854096501e-17      # m, quantum-wave longitudinal wave-length
QWAVE_AMPLITUDE = A_l = 9.215405708e-19           # m, quantum-wave longitudinal wave-amplitude (equilibrium-to-peak)
QWAVE_SPEED = c = 299792458                   # m / s, quantum-wave velocity (speed of light)
QWAVE_DENSITY = rho = 3.859764540e22            # kg / m^3, quantum-wave medium density (quantum-space or aether)

# =====================
# Variables (dimensionless) and particle constants (EWT)
# =====================
# DELTA : amplitude factor (dimensionless) — user-supplied per system/transition
# K     : particle wave center count (dimensionless)
# Q     : particle count in a group (dimensionless)

# Electron-specific constants / g-factors:
ELECTRON_K = Ke = 10                         # electron wave center count (dimensionless)
ELECTRON_OUTER_SHELL = Oe = 2.138743820                # electron outer shell multiplier (dimensionless)
ELECTRON_ORBITAL_G = g_lambda = 0.9873318320         # electron orbital g-factor (dimensionless)
ELECTRON_SPIN_G = g_A = 0.9826905018              # electron spin g-factor (dimensionless)
PROTON_ORBITAL_G = g_p = 0.9898125300              # proton orbital g-factor (dimensionless)

# =====================
# Classical constants used in the same context
# =====================
PLANCK_LENGTH = l_p = 1.616255e-35              # m, Planck length
PLANCK_TIME = t_p = 5.391247e-44              # s, Planck time
PLANCK_MASS = m_p = 2.176434e-8               # kg, Planck mass
PLANCK_CHARGE = q_p = 1.875545956e-18           # m, Planck charge

ELECTRON_RADIUS = re = 2.8179403262e-15           # m, electron classical radius
ALPHA = alpha = 7.2973525693e-3         # fine-structure constant (dimensionless)
PI = pi = 3.14159265358979323846     # pi (dimensionless)
E = e = 2.71828182845904523536      # euler's number (dimensionless)
PHI = phi = (1 + 5 ** 0.5) / 2        # golden ratio (dimensionless)

PLANCK_CONSTANT = h = 6.62607015e-34              # J·s, Planck constant — exact definition
ELECTRON_VOLT = eV = 1.602176634e-19            # J, exact definition
ELECTRIC_CONSTANT = epsilon_0 = 8.8541878128e-12    # F/m (C^2 / N·m^2), electric constant (vacuum permittivity)
MAGNETIC_CONSTANT = mu_0 = 1.25663706212e-6         # N / A^2 (H/m), magnetic constant (vacuum permeability)
BOHR_RADIUS = a_0 = 5.29177210903e-11         # m, Bohr radius
COULOMB_CONSTANT = k = 8.9875517923e9             # N·m^2 / C^2, Coulomb's constant

# Electron & neutrino reference energies used with wave constants
NEUTRINO_ENERGY = Ev = 3.8280e-19                 # J, neutrino "seed" energy used by EWT (~ 2.39 eV)
ELECTRON_ENERGY = Ee = 8.1871e-14                 # J, electron rest energy (~ 0.511 MeV)

# =====================
# Unit helpers
# =====================
def J_to_eV(E_J: float) -> float:
    """Convert joules to electron-volts."""
    return E_J / eV

def eV_to_J(E_eV: float) -> float:
    """Convert electron-volts to joules."""
    return E_eV * eV






# =====================
# Constants Smoke-Test
# =====================
def qwave_length_check():
    """Check that lambda_l = c / f, where f = Ee / h."""
    f = Ee / h
    lambda_calc = c / f
    assert abs(lambda_calc - lambda_l) < 1e-30, f"lambda_l check failed: {lambda_calc} vs {lambda_l}"





if __name__ == "__main__":
    # Run smoke-test
    qwave_length_check()
    print("All constant checks passed.")

