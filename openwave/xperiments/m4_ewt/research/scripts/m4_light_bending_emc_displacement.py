#!/usr/bin/env python3
"""
M4/EWT - Solar light bending from EMC displacement field.

OpenWave criterion:
    Gravity: metric phenomena
    Test: light bending (time dilation and Lambda intentionally omitted).

Mechanism:
    The EMC density around the Sun is lowered by the energy density
    of the solitons. The surrounding EMCs move toward the density
    deficit, creating a displacement field

        u(r) = - chi * grad( N_nu(r) ).

    This displacement changes the local EMC spacing, which in turn
    modifies the local propagation speed. The effective refractive
    index n(r) is only a scalar encoding of that displacement:

        n(r) = 1 / sqrt(1 - 2*r_s/r)

    The bending is computed from grad n(r), but the physical origin
    is the EMC displacement, not a generic optical slow-down.

    This script computes the solar-limb bending angle and compares
    it with the observed 1.75 arcsec.
"""

import math

# ----------------------------------------------------------------------
# 1. Physical constants and EWT density levels
# ----------------------------------------------------------------------
print("[1/4] Loading physical constants and EWT density levels...")

G      = 6.67430e-11          # m^3 kg^-1 s^-2
c      = 299792458.0          # m/s
M_sun  = 1.989e30             # kg
R_sun  = 6.957e8              # m

r_s = 2.0 * G * M_sun / (c * c)

N_stat = 3.298651882390107e52  # statutory EMC density
N_eff  = 6.252517621935487e48  # effective EMC density inside matter
N_max  = 5.300415534439117e54  # absolute maximum EMC density

print(f"    G          = {G:.6e} m^3 kg^-1 s^-2")
print(f"    c          = {c:.3f} m/s")
print(f"    M_sun      = {M_sun:.3e} kg")
print(f"    R_sun      = {R_sun:.3e} m")
print(f"    r_s        = {r_s:.3f} m")
print(f"    N_stat     = {N_stat:.6e}")
print(f"    N_eff      = {N_eff:.6e}")
print(f"    N_max      = {N_max:.6e}")

# ----------------------------------------------------------------------
# 2. EMC density profile and scalar encoding n(r)
# ----------------------------------------------------------------------
print("[2/4] Building EMC density profile and scalar encoding n(r)...")

def N_nu(r):
    """
    Local EMC density outside the Sun.
    N_nu(r) = N_stat * (1 - 2*r_s/r)
    """
    if r <= 0:
        raise ValueError("r must be positive")
    deficit = 2.0 * r_s / r
    if deficit >= 1.0:
        raise ValueError("Deficit >= 1: invalid in this static profile")
    return N_stat * (1.0 - deficit)

def n_index(r):
    """
    Scalar encoding of the EMC displacement.

    n(r) = 1 / sqrt(1 - 2*r_s/r)

    This is not an independent optical assumption: it follows from
    the same EMC density that defines the displacement field u(r).
    """
    return 1.0 / math.sqrt(1.0 - 2.0 * r_s / r)

print(f"    N_nu(R_sun) / N_stat = {N_nu(R_sun)/N_stat:.12f}")
print(f"    n(R_sun)              = {n_index(R_sun):.12f}")

# ----------------------------------------------------------------------
# 3. Light bending integral (u-substitution to avoid singularity)
# ----------------------------------------------------------------------
print("[3/4] Integrating the light bending angle using u = R/r ...")

# We integrate from r = R to infinity using u = R/r.
# The integral becomes:
#   Integral = r_s / R^2 * ∫_{0}^{1} (1 - 2*r_s*u/R)^(-3/2) / sqrt(1-u^2) du
# and the bending angle is:
#   theta_rad = 2 * R * Integral
#
# This form is smooth at both limits (u=0 and u=1).

def integrand_u(u):
    if u <= 0.0 or u >= 1.0:
        return 0.0
    factor = (1.0 - 2.0 * r_s * u / R_sun) ** (-1.5)
    return u * factor / math.sqrt(1.0 - u * u)

try:
    from scipy.integrate import quad
    USE_SCIPY = True
except ImportError:
    USE_SCIPY = False
    print("    scipy not available; using simple trapezoidal fallback.")

if USE_SCIPY:
    integral_u, err = quad(integrand_u, 0.0, 1.0, epsabs=1e-14, epsrel=1e-10)
    integral = (r_s / (R_sun * R_sun)) * integral_u
else:
    N = 20000
    du = 1.0 / N
    total = 0.0
    for i in range(N):
        u = (i + 0.5) * du
        total += integrand_u(u) * du
    integral_u = total
    integral = (r_s / (R_sun * R_sun)) * integral_u

theta_rad = 2.0 * R_sun * integral

theta_arcsec = theta_rad * 180.0 / math.pi * 3600.0

theta_expected_rad = 2.0 * r_s / R_sun
theta_expected_arcsec = theta_expected_rad * 180.0 / math.pi * 3600.0

print(f"    Integral over u     = {integral_u:.6e}")
print(f"    Integral over r     = {integral:.6e}")
print(f"    theta numerical     = {theta_arcsec:.6f} arcsec")
print(f"    theta analytic      = {theta_expected_arcsec:.6f} arcsec")

# ----------------------------------------------------------------------
# 4. Comparison with observation
# ----------------------------------------------------------------------
print("[4/4] Comparing with observed solar-limb deflection...")

target_arcsec = 1.75
rel_diff = abs(theta_arcsec - target_arcsec) / target_arcsec * 100.0

print(f"    Observed target      = {target_arcsec:.2f} arcsec")
print(f"    EWT numeric result   = {theta_arcsec:.6f} arcsec")
print(f"    Relative difference  = {rel_diff:.4f}%")

if rel_diff < 1.0:
    print("    RESULT: PASS (within 1%)")
else:
    print("    RESULT: FAIL (check assumptions)")

print("\nDone.")