"""
OpenWave constants module.

This module provides fundamental constants for OpenWave simulations,
sourced from EWT at https://energywavetheory.com/equations/

All values use SI units (kg, m, s) for consistency.
"""

# ================================================================
# WAVE-MEDIUM
# ================================================================
MEDIUM_DENSITY = 3.506335701e22  # kg / m^3, wave-medium density (ρ)
ATTOMETER = 1e-18  # m, attometer length scale (for memory efficiency in simulations)

# ================================================================
# ENERGY-WAVE
# ================================================================
EWAVE_LENGTH = 2.854096501e-17  # m, energy-wave length (λ)
EWAVE_AMPLITUDE = 9.215405708e-19  # m, energy-wave amplitude (A, equilibrium-to-peak)
EWAVE_SPEED = 299792458  # m / s, energy-wave velocity (c, speed of light)

EWAVE_FREQUENCY = 1.050393558e25  # Hz, energy-wave frequency (f = EWAVE_SPEED/EWAVE_LENGTH)
EWAVE_PERIOD = 9.520241169e-26  # s, energy-wave period (T = 1 / EWAVE_FREQUENCY)

# ================================================================
# Neutrino particle (seed particle)
# ================================================================
NEUTRINO_ENERGY = 3.8280e-19  # J, neutrino "seed" energy used by EWT (~ 2.39 eV)

# ================================================================
# Electron particle
# ================================================================
ELECTRON_ENERGY = 8.1871e-14  # J, electron rest energy (~ 0.511 MeV)
ELECTRON_K = 10  # electron wave center count (dimensionless)
ELECTRON_RADIUS = 2.8179403262e-15  # m, electron classical radius
ELECTRON_OUTER_SHELL = 2.138743820  # electron outer shell multiplier
ELECTRON_ORBITAL_G = 0.9873318320  # electron orbital g-factor (gλ, dimensionless)
ELECTRON_SPIN_G = 0.9826905018  # electron spin g-factor (gA, dimensionless)
# In Energy Wave Equations: Correction Factors (https://vixra.org/abs/1803.0243),
# a potential explanation for the values of these g-factors is presented as
# a relation of Earth’s outward velocity and spin velocity against a rest frame for the universe.

# ================================================================
#  Proton particle
# ================================================================
PROTON_ENERGY = 1.5033e-10  # J, CODATA proton rest energy (~ 938.272 MeV)
PROTON_K = 44  # proton wave center count (dimensionless)
PROTON_ORBITAL_G = 0.9898125300  # proton orbital g-factor (gp, dimensionless)
PROTON_MASS = 1.67262192369e-27  # kg, proton mass from CODATA

# ================================================================
# Classical constants
# ================================================================
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
COULOMB_CONSTANT = 8.9875517923e9  # N·m^2/C^2 (N when charge C is distance), k

# DELTA : amplitude factor (dimensionless) — user-supplied per system/transition
# K     : particle wave center count (dimensionless)
# Q     : particle count in a group (dimensionless)
