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
# Spatial:  ATTOMETER = 1e-18 m (var suffix '_am')
#   - Wavelength: ~28.5 am (vs 2.85e-17 m)
#   - Amplitude: ~0.92 am (vs 9.2e-19 m)
#
# Temporal: RONTOSECOND = 1e-27 s (var suffix '_rs, _amrs, _rHz')
#   - Frequency: ~0.01 rHz (vs 1.05e25 Hz)
#   - Wave Speed: ~0.3 am/rs (vs 2.99e8 m/s)
#   - Timestep: ~4.15 rs (vs 4.15e-27 s)
#   - Period: ~95.2 rs (vs 9.52e-26 s)
#
# Benefits:
#   - Solution for floating-point precision
#   - Prevents catastrophic cancellation in derivatives/gradients
#   - Maintains 6-7 significant digits with f32
#   - Scaled values near 1.0 (optimal for floating point)
#   - Improves memory usage & computational performance (f32 vs f64)
ATTOMETER = 1e-18  # m/am, attometer length scale
RONTOSECOND = 1e-27  # s/rs, rontosecond time scale

# ================================================================
# WAVE-FIELD MEDIUM [EWT Constants]
# ================================================================
MEDIUM_DENSITY = 3.859764604e22  # kg / m^3, wave-medium density (ρ)
EWAVE_SPEED = 299792458  # m / s, speed of light (c), c² = elasticity / density of medium

# ================================================================
# ENERGY-WAVE RHYTHM & SIZE [EWT Constants]
# ================================================================
EWAVE_FREQUENCY = 1.050393558e25  # Hz, energy-wave frequency (f = EWAVE_SPEED / EWAVE_LENGTH)
EWAVE_AMPLITUDE = 9.215405708e-19  # m, energy-wave amplitude (A, equilibrium-to-peak)
EWAVE_LENGTH = 2.854096501e-17  # m, energy-wave length (λ = EWAVE_SPEED / EWAVE_FREQUENCY)
EWAVE_PERIOD = 9.520241169e-26  # s, energy-wave period (T = 1 / EWAVE_FREQUENCY)

# ================================================================
# NEUTRINO Particle (seed particle)
# ================================================================
NEUTRINO_K = 1  # neutrino wave center count (dimensionless)
NEUTRINO_RADIUS = EWAVE_LENGTH  # 2.85e-17m, neutrino radius = 1 λ
NEUTRINO_ENERGY = 3.8280e-19  # J, neutrino "seed" energy used by EWT (~ 2.39 eV)

# ================================================================
# ELECTRON Particle
# ================================================================
ELECTRON_K = 10  # electron wave center count (dimensionless)
ELECTRON_RADIUS = 2.8179403262e-15  # m, electron classical radius
ELECTRON_MASS = 9.1093837139e-31  # kg, electron mass from CODATA
ELECTRON_ENERGY = 8.1871e-14  # J, electron rest energy (~ 0.511 MeV)
ELECTRON_OUTER_SHELL = 2.138743820  # electron outer shell multiplier
ELECTRON_ORBITAL_G = 0.9873318320  # electron orbital g-factor (gλ, dimensionless)
ELECTRON_SPIN_G = 0.9826905018  # electron spin g-factor (gA, dimensionless)
# In Energy Wave Equations: Correction Factors (https://vixra.org/abs/1803.0243),
# a potential explanation for the values of these g-factors is presented as
# a relation of Earth’s outward velocity and spin velocity against a rest frame for the universe.

# ================================================================
#  PROTON & NEUTRON Particle
# ================================================================
PROTON_K = 44  # proton wave center count (dimensionless)
PROTON_RADIUS = 8.414e-16  # m, proton radius
PROTON_MASS = 1.67262192595e-27  # kg, proton mass from CODATA
PROTON_ENERGY = 1.5033e-10  # J, CODATA proton rest energy (~ 938.272 MeV)
PROTON_ORBITAL_G = 0.9898125300  # proton orbital g-factor (gp, dimensionless)

NEUTRON_MASS = 1.67492749804e-27  # kg, neutron mass from CODATA 2022

# ================================================================
# Classical Constants
# ================================================================
PLANCK_LENGTH = 1.616255e-35  # m, Planck length
PLANCK_TIME = 5.391247e-44  # s, Planck time
PLANCK_MASS = 2.176434e-8  # kg, Planck mass
PLANCK_CHARGE = 1.875459e-18  # m, Planck charge
PLANCK_CONSTANT = 6.62607015e-34  # J·s, h = Planck constant
PLANCK_CONSTANT_REDUCED = PLANCK_CONSTANT / (2 * np.pi)  # J·s, ħ = reduced Planck constant

FINE_STRUCTURE = 7.2973525643e-3  # fine-structure constant, alpha
ELECTRIC_CONSTANT = 8.8541878188e-12  # F/m, vacuum permittivity, epsilon_0
MAGNETIC_CONSTANT = 1.25663706127e-6  # kg/m, vacuum permeability, 4π.10-7, mu_0
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3/kg/s^2, G, Newtonian constant of gravitation

BOHR_RADIUS = 5.29177210544e-11  # m, rₕ = Hydrogen 1s radius (Bohr Radius)
HYDROGEN_LINE = 1.420405751e9  # Hz, Hydrogen 21cm line frequency, spin-flip transition
HYDROGEN_LYMAN_ALPHA = 2.4660677e15  # Hz, Hydrogen Lyman-alpha frequency

ELEMENTARY_CHARGE = 1.602176634e-19  # m, The elementary charge from CODATA values
COULOMB_CONSTANT = 8.9875517923e9  # N·m^2/C^2 (N when charge C is distance), k
BOHR_MAGNETON = 9.2740100657e-24  # J/T, μ_B, Bohr magneton (~ 5.788 e-5 eV/T)
AVOGADRO_NUMBER = 6.02214076e23  # 1/mol, N_A, Avogadro's number

# ================================================================
# De Broglie / Matter Wave Constants
# ================================================================
# De Broglie wavelength: λ_dB = h / p = h / (m*v)
# At rest energy (Compton wavelength): λ_C = h / (m*c)
COMPTON_WAVELENGTH_ELECTRON = 2.42631023538e-12  # m, λ_C, Compton wavelength of electron
COMPTON_WAVELENGTH_PROTON = 1.32140985539e-15  # m, λ_C,p = h / (m_p * c), CODATA 2022
COMPTON_WAVELENGTH_NEUTRON = 1.31959090581e-15  # m, λ_C,n = h / (m_n * c), CODATA 2022

# Reduced Compton wavelengths: λ_bar = λ / (2π) = ℏ / (m * c)
COMPTON_WAVELENGTH_ELECTRON_REDUCED = 3.8615926796e-13  # m, ℏ / (m_e * c), CODATA 2022
COMPTON_WAVELENGTH_PROTON_REDUCED = 2.10308910336e-16  # m, ℏ / (m_p * c), CODATA 2022
COMPTON_WAVELENGTH_NEUTRON_REDUCED = 2.10019415255e-16  # m, ℏ / (m_n * c), CODATA 2022

# Rydberg constant & energy (fundamental atomic energy scale)
RYDBERG_CONSTANT = 10973731.568157  # m^-1, R_∞, Rydberg constant, CODATA 2022
RYDBERG_ENERGY = 2.1798723611035e-18  # J, E_∞ = R_∞ * h * c (~ 13.605693 eV)

# ================================================================
# Electromagnetic Wave Constants
# ================================================================
# Impedance of free space: Z_0 = √(μ_0 / ε_0) = μ_0 * c
IMPEDANCE_VACUUM = 376.730313412  # Ω, Z_0, characteristic impedance of vacuum, CODATA 2022
# Critical for EM wave energy density: u = (ε_0/2)*E^2 + (1/2μ_0)*B^2
# and Poynting vector: S = (1/μ_0) * E × B = E^2 / Z_0

# ================================================================
# Additional Constants
# ================================================================

PI = np.pi  # π = 3.1415926535... (dimensionless)
EULER_NUMBER = np.e  # e = 2.7182818284... (dimensionless)
GOLDEN_RATIO = 1.6180339887  # φ = (1+sqrt(5))/2 (dimensionless)

# DELTA : amplitude factor (dimensionless) — user-supplied per system/transition
# K     : particle wave center count (dimensionless)
# Q     : particle count in a group (dimensionless)
