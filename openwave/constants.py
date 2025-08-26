"""
OpenWave Constants
from: https://energywavetheory.com/equations/

This module exposes:
- Wave constants (values and units per the EWT "wave constants" set)
- Particle/electron-specific constants used throughout EWT
- Common classical constants used alongside wave constants
- Equations implemented as Python functions:
  * Particle (longitudinal) rest energy via E(K) = Ev * K**5
  * Photon (transverse) energy/frequency/wavelength relations — via E = h * f and f = c / λ
  * Electric force & potential energy (classical Coulomb; EWT "charge-as-length" form)
  * Unit helpers (J <-> eV)

Notes:
- In EWT, "charge" is modeled as displacement (meters). Where appropriate, this module
  provides both the classical formula and an "EWT interpretation" variant.
- The PDF also lists Magnetic, Gravitational, Strong, Orbital, and Relativistic forms.
  Their exact closed forms in wave constants are image-based in the PDF; if you want me to
  add those explicit forms, point me to the textual definitions or include the formula specs,
  and I will extend this module accordingly.
"""

from math import pi, e
from typing import Tuple

# =====================
# Wave constants (EWT)
# =====================
# Units are SI (kg, m, s). Values follow the EWT wave-constants view.
wave_length = lambda_l = 2.854096501e-17      # m, quantum-wave longitudinal wave-length
wave_amplitude = A_l = 9.215405708e-19           # m, quantum-wave longitudinal wave-amplitude (equilibrium-to-peak)
wave_speed = c = 299792458                   # m / s, quantum-wave velocity (speed of light)
wave_density = rho = 3.859764540e22            # kg / m^3, quantum-wave medium density (quantum-space or aether)

# =====================
# Variables (dimensionless) and particle constants (EWT)
# =====================
# delta : amplitude factor (dimensionless) — user-supplied per system/transition
# K : particle wave center count (dimensionless)
# Q : particle count in a group (dimensionless)

# Electron-specific constants / g-factors:
Ke = 10                         # electron wave center count (dimensionless)
Oe = 2.138743820                # electron outer shell multiplier (dimensionless)
g_lambda = 0.9873318320         # electron orbital g-factor (dimensionless)
g_A = 0.9826905018              # electron spin g-factor (dimensionless)
g_p = 0.9898125300              # proton orbital g-factor (dimensionless)

# =====================
# Classical constants used in the same context
# =====================
l_p = 1.616255e-35              # m, Planck length
t_p = 5.391247e-44              # s, Planck time
m_p = 2.176434e-8               # kg, Planck mass
q_p = 1.875545956e-18           # m, Planck charge

re = 2.8179403262e-15           # m, electron classical radius
alpha = 7.2973525693e-3         # fine-structure constant (dimensionless)
pi = 3.14159265358979323846     # pi (dimensionless)
e = 2.71828182845904523536      # euler's number (dimensionless)
phi = (1 + 5 ** 0.5) / 2        # golden ratio (dimensionless)

h = 6.62607015e-34              # J·s, Planck constant — exact definition
eV = 1.602176634e-19            # J, exact definition
epsilon_0 = 8.8541878128e-12    # F/m (C^2 / N·m^2), electric constant (vacuum permittivity)
mu_0 = 1.25663706212e-6         # N / A^2 (H/m), magnetic constant (vacuum permeability)
a_0 = 5.29177210903e-11         # m, Bohr radius
k = 8.9875517923e9              # N·m^2 / C^2, Coulomb's constant

# Electron & neutrino reference energies used with wave constants
Ev = 3.8280e-19                 # J, neutrino "seed" energy used by EWT (~ 2.39 eV)
Ee = 8.1871e-14                 # J, electron rest energy (~ 0.511 MeV)

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
# Constants Smoke-Check
# =====================
def wavelength_check():
    """Check that lambda_l = c / f, where f = Ee / h."""
    f = Ee / h
    lambda_calc = c / f
    assert abs(lambda_calc - lambda_l) < 1e-30, f"lambda_l check failed: {lambda_calc} vs {lambda_l}"




# =====================
# Misc helpers
# =====================
def summarize_constants() -> str:
    """
    Return a human-readable multi-line summary of key constants and their units.
    Uses an f-string to inject current module variable values.
    """
    return f"""
Wave constants:
  lambda_l   = {lambda_l:.6e} m
  A_l        = {A_l:.6e} m
  rho        = {rho:.6e} kg/m^3
  c          = {c:.6f} m/s

Particle constants (electron & g-factors):
  Ke         = {Ke} (dimensionless)
  Oe         = {Oe}
  g_lambda   = {g_lambda}
  g_A        = {g_A}
  g_p        = {g_p}

Classical constants (used alongside the wave set):
  l_p        = {l_p:.6e} m
  t_p        = {t_p:.6e} s
  m_p        = {m_p:.6e} kg
  q_p        = {q_p:.6e} m

  h          = {h:.6e} J·s
  eV         = {eV:.6e} J
  epsilon_0  = {epsilon_0:.6e} F/m
  mu_0       = {mu_0:.6e} H/m
  k          = {k:.6e} N·m^2/C^2
  alpha      = {alpha:.6e}
  phi        = {phi:.6e}

Reference energies:
  Ev         = {Ev:.6e} J  (~{J_to_eV(Ev):.3f} eV)
  Ee         = {Ee:.6e} J  (~{J_to_eV(Ee):.3f} eV)
""".strip()


if __name__ == "__main__":
    # Show constants
    print(summarize_constants())
