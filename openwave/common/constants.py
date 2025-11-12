"""
OpenWave constants module.

This module provides fundamental constants for OpenWave simulations,
sourced from EWT at https://energywavetheory.com/equations/

All values use SI units (kg, m, s) for consistency.
"""

import numpy as np

# ================================================================
# Scaled SI Units for Numerical Precision
# ================================================================
# OpenWave engines use scaled units to maintain f32 precision:
# Spatial:  ATTOMETER = 1e-18 m
#   - Wavelength: ~28.5 am (vs 2.85e-17 m)
#   - Grid spacing: ~1.25 am (vs 1.25e-18 m)
#   - Naming: variables/fields with suffix '_am'
#
# Temporal: RONTOSECOND = 1e-27 s
#   - Timestep: ~2.4 rs (vs 2.4e-27 s)
#   - Period: ~95.2 rs (vs 9.52e-26 s)
#   - Naming: variables/fields with suffix '_rs'
#
# Benefits:
#   - Solution for floating-point precision
#   - Prevents catastrophic cancellation in derivatives/gradients
#   - Maintains 6-7 significant digits with f32
#   - Scaled values near 1.0 (optimal for floating point)
#   - Reduces memory usage (f32 vs f64)
#   - Improves computational performance (f32 vs f64)
ATTOMETER = 1e-18  # m, attometer length scale
RONTOSECOND = 1e-27  # s, rontosecond time scale

# ================================================================
# WAVE-MEDIUM
# ================================================================
MEDIUM_DENSITY = 3.859764604e22  # kg / m^3, wave-medium density (ρ)
EWAVE_SPEED = 299792458  # m / s, energy-wave velocity (c, speed of light)

# ================================================================
# ENERGY-WAVE
# ================================================================
EWAVE_AMPLITUDE = 9.215405708e-19  # m, energy-wave amplitude (A, equilibrium-to-peak)
EWAVE_FREQUENCY = 1.050393558e25  # Hz, energy-wave frequency (f)

EWAVE_LENGTH = 2.854096501e-17  # m, energy-wave length (λ = EWAVE_SPEED / EWAVE_FREQUENCY)
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
PLANCK_CONSTANT = 6.62607015e-34  # J·s, h = Planck constant
PLANCK_CONSTANT_REDUCED = PLANCK_CONSTANT / (2 * np.pi)  # J·s, ħ = reduced Planck constant

FINE_STRUCTURE = 7.2973525693e-3  # fine-structure constant, alpha
ELECTRIC_CONSTANT = 8.8541878128e-12  # F/m, vacuum permittivity, epsilon_0
MAGNETIC_CONSTANT = 1.25663706212e-6  # kg/m, vacuum permeability, 2π.10-7, mu_0

BOHR_RADIUS = 5.29177210903e-11  # m, rₕ = Hydrogen 1s radius (Bohr Radius)
HYDROGEN_LINE = 1.420405751e9  # Hz, Hydrogen 21cm line frequency, spin-flip transition
HYDROGEN_LYMAN_ALPHA = 2.4660677e15  # Hz, Hydrogen Lyman-alpha frequency

ELEMENTARY_CHARGE = 1.6022e-19  # m, The elementary charge from CODATA values
COULOMB_CONSTANT = 8.9875517923e9  # N·m^2/C^2 (N when charge C is distance), k

GOLDEN_RATIO = 1.6180339887  # φ = (1+sqrt(5))/2 (dimensionless)

# DELTA : amplitude factor (dimensionless) — user-supplied per system/transition
# K     : particle wave center count (dimensionless)
# Q     : particle count in a group (dimensionless)
