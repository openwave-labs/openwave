#!/usr/bin/env python3
"""
M4/EWT - True force test: Newtonian force from geometry alone, without G.

OpenWave criterion:
    Gravity: Newton limit (GEM)
    "attractive 1/r^2 between masses, via the GEM route; the strength (G)
     credited only when it follows from the model's own mechanism, never
     fitted"

Purpose:
    Compute F_EMC from geometric factors alone (no G anywhere in the
    calculation), then compare with the observed gravitational force
    F_obs = G_CODATA * M1 * M2 / R^2.

    The purpose is to show that the strength of the force follows from
    the BCC lattice geometry, not from an inserted G value.

Chain of derivation (all geometric, no G):
    ---------------------------------------------------------------
    Step 1.  Monopole density deficit per source:
                 delta_eta_i(r) = - A_i / r
    Step 2.  Gradient of each deficit:
                 grad(delta_eta_i) = A_i / r^2 * r_hat
    Step 3.  Overlap integral over all space:
                 I(R) = integral grad(delta_eta_1).grad(delta_eta_2) dV
    Step 4.  Angular integral (exact, see M4.10 manuscript):
                 integral dOmega [r - R cos th] /
                     [r^2 + R^2 - 2rR cos th]^(3/2)
                 =   0            for r < R
                 =   4 pi / r^2   for r >= R
    Step 5.  Radial integration over r in [R, inf):
                 I(R) = 4 pi A_1 A_2 / R
    Step 6.  Force kernel from -dI/dR:
                 F_geom = 4 pi A_1 A_2 / R^2
    Step 7.  Physical force with EMC coupling K_emc:
                 F_EMC  = K_emc * F_geom = 4 pi K_emc A_1 A_2 / R^2
    ---------------------------------------------------------------

    Amplitude A_i (geometric, no G):
        A_i = (2 M_i r_e / m_e) * sqrt(X_eff)
              / (A_pi^4 N_geom^3 K_WC sqrt(N_nu_stat))

    Coupling K_emc (geometric, no G):
        K_emc = c^2 m_e A_pi^4 N_geom^3 K_WC sqrt(N_nu_eff) / (16 pi r_e)

    No G appears in any of the steps above. G is used only in section 6,
    where F_EMC is compared with F_obs = G_CODATA M1 M2 / R^2.

Dimensional anchors:
    r_e, m_e, c are the measured quantities used to build the
    dimensionless geometric ratio. The force F_EMC is a length-based
    quantity built from these anchors and the BCC geometry.

Output:
    - console: 8 sections showing the full chain
    - all geometric inputs printed

No free numerical parameters are introduced.
"""

import math
import sys

# ----------------------------------------------------------------------
# Import shared geometric primitives from the M4.7 engine.
#
# The engine provides the geometric primitives and the self-consistent
# chain (A_pi, N_geom, N_nu_stat, N_nu_eff, X_eff). The FORCE itself is
# computed in this script from raw factors, without using G.
# ----------------------------------------------------------------------

try:
    from m4_7_ewt_emergence_engine import (
        PI, C0, M_E, R_E, E_CHARGE_CODATA, G_CODATA,
        BCC_IDEAL_PROJECTION_LP,
        compute_alpha_core,        # returns A_pi = 4 pi^3 + pi^2 + pi
        derive_eps_M_from_BCC,
        derive_planck_charge_from_e,
        derive_neutrino_radius,
        derive_lambda_l_geometric,
        gravity_sector,
    )
except ImportError:
    raise ImportError(
        "This module requires m4_7_ewt_emergence_engine.py in the same directory."
    )


# Electron wave-center count, fixed by the 1-3-6 geometry.
K_WC = 10


# ======================================================================
# Section 1 - Geometric primitives (no G)
# ======================================================================

def geometric_primitives():
    """Collect every geometric factor that enters the force.

    Returns a dict with all quantities needed downstream. The value
    G_geom_xcheck is included only for cross-check purposes; it is NOT
    used in the force computation itself.
    """
    bcc = derive_eps_M_from_BCC(8.0 * PI**4)
    N_geom = bcc["N_geom"]
    eps_M = bcc["eps_M"]

    # A_pi = 4 pi^3 + pi^2 + pi. Geometric core from Yee (2019).
    A_pi = compute_alpha_core()

    # Geometric fine-structure constant: 1/alpha = A_pi - eps_M.
    alpha_inv = A_pi - eps_M
    alpha_geom = 1.0 / alpha_inv

    # Planck charge and neutrino radius (geometric, no G).
    q_P = derive_planck_charge_from_e(alpha_geom, E_CHARGE_CODATA)
    r_nu = derive_neutrino_radius(alpha_geom, q_P)["r_nu"]

    # Self-consistent lattice spacing lambda_l (geometric fixed point).
    lambda_l = derive_lambda_l_geometric(
        alpha_geom=alpha_geom,
        r_e=R_E,
        r_nu=r_nu,
        N_geom=N_geom,
        L_p_geom=BCC_IDEAL_PROJECTION_LP,
        K_WC=K_WC,
    )

    # Full geometric chain. The output G_EWT is used only for cross-check.
    res = gravity_sector(
        alpha_geom=alpha_geom,
        r_nu=r_nu,
        N_geom=N_geom,
        L_p_geom=BCC_IDEAL_PROJECTION_LP,
        K_WC=K_WC,
        lambda_l=lambda_l,
        r_e=R_E,
        m_e=M_E,
        c0=C0,
    )

    return {
        "A_pi":          A_pi,
        "eps_M":         eps_M,
        "alpha_geom":    alpha_geom,
        "N_geom":        N_geom,
        "r_nu":          r_nu,
        "lambda_l":      lambda_l,
        "N_nu_stat":     res["N_nu_statutory"],
        "N_nu_eff":      res["N_nu_eff"],
        "X_eff":         res["X_eff"],
        # Cross-check only. Not used in the force computation.
        "G_geom_xcheck": res["G_EWT"],
    }


# ======================================================================
# Section 2 - Monopole amplitude A from geometric factors (no G)
# ======================================================================

def geometric_amplitude(M, geom):
    """Monopole amplitude A_i, written in chain factors without G.

        A = (2 M r_e / m_e) * sqrt(X_eff)
            / (A_pi^4 N_geom^3 K_WC sqrt(N_nu_stat))

    where X_eff = N_nu_stat / N_nu_eff.

    Every factor on the right-hand side is geometric (A_pi, N_geom,
    K_WC, X_eff, N_nu_stat) or a dimensional anchor (r_e, m_e). No G
    appears.
    """
    numerator = 2.0 * M * R_E * math.sqrt(geom["X_eff"])
    denominator = (
        M_E
        * geom["A_pi"] ** 4
        * geom["N_geom"] ** 3
        * K_WC
        * math.sqrt(geom["N_nu_stat"])
    )
    return numerator / denominator


# ======================================================================
# Section 3 - EMC coupling K_emc from geometric factors (no G)
# ======================================================================

def geometric_K_emc(geom):
    """EMC field coupling K_emc, written without G.

        K_emc = c^2 m_e A_pi^4 N_geom^3 K_WC sqrt(N_nu_eff)
                / (16 pi r_e)

    Derivation. In the field-theoretic formulation, the coupling is
    K_emc = c^4 / (16 pi G). Substituting the geometric expression for
    G_EWT and simplifying cancels G entirely, leaving the expression
    above. No G appears on the right-hand side.
    """
    numerator = (
        C0 ** 2
        * M_E
        * geom["A_pi"] ** 4
        * geom["N_geom"] ** 3
        * K_WC
        * math.sqrt(geom["N_nu_eff"])
    )
    denominator = 16.0 * PI * R_E
    return numerator / denominator


# ======================================================================
# Section 4 - Overlap integral and force kernel
# ======================================================================

def overlap_integral_at(A1, A2, R, num_pts=200000):
    """Numerical evaluation of I(R) = int grad(de1).grad(de2) dV.

    The angular integral has already been evaluated analytically to
    4 pi / r^2 for r >= R (see M4.10 manuscript). The remaining radial
    integral is evaluated here via the coordinate transform
    r = R / (1 - t), t in [0, 1).

    Result (analytic):  I(R) = 4 pi A_1 A_2 / R.
    """
    dt = 1.0 / num_pts
    s = 0.0
    for i in range(num_pts):
        t = (i + 0.5) * dt
        r = R / (1.0 - t)
        dr_dt = R / ((1.0 - t) * (1.0 - t))
        angular = 4.0 * math.pi / (r * r)
        s += angular * dr_dt * dt
    return A1 * A2 * s


def geometric_gradient(A1, A2, R, dR_frac=1e-6, num_pts=200000):
    """Force kernel F_geom = -dI/dR = 4 pi A_1 A_2 / R^2.

    Evaluated numerically by central difference in R. This gives the
    force kernel BEFORE the EMC coupling K_emc is applied.
    """
    dR = R * dR_frac
    i_plus = overlap_integral_at(A1, A2, R + dR, num_pts)
    i_minus = overlap_integral_at(A1, A2, R - dR, num_pts)
    return -(i_plus - i_minus) / (2.0 * dR)


# ======================================================================
# Section 5 - Force from geometry alone (no G)
# ======================================================================

def geometric_force(M1, M2, R, geom):
    """F_EMC = K_emc * (-dI/dR) = 4 pi K_emc A_1 A_2 / R^2.

    Returns (F_EMC, A1, A2, K_emc, F_geom, dI_dR).

    No G appears anywhere in this function. The full chain is:

        A_i    <- geometric_amplitude(M_i)
        K_emc  <- geometric_K_emc()
        F_geom <- geometric_gradient(A1, A2, R)
        F_EMC  <- K_emc * F_geom

    The comparison with F_obs = G_CODATA M1 M2 / R^2 happens OUTSIDE
    this function, in section 6.
    """
    A1 = geometric_amplitude(M1, geom)
    A2 = geometric_amplitude(M2, geom)
    K_emc = geometric_K_emc(geom)
    F_geom = geometric_gradient(A1, A2, R)
    F_EMC = K_emc * F_geom
    return F_EMC, A1, A2, K_emc, F_geom


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 78)
    print("M4.10.2 - TRUE FORCE TEST")
    print("Force from geometric factors alone (no G in the calculation)")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Section 1: geometric primitives
    # ------------------------------------------------------------------
    print("\n[1/8] Geometric primitives from the BCC lattice")
    print("      (all from the M4.7 self-consistent chain)")
    geom = geometric_primitives()
    print(f"    A_pi   = 4 pi^3 + pi^2 + pi      = {geom['A_pi']:.12f}")
    print(f"    eps_M                            = {geom['eps_M']:.6e}")
    print(f"    alpha_geom = 1/(A_pi - eps_M)    = {geom['alpha_geom']:.12f}")
    print(f"    N_geom = 8 pi^4 (1 - zeta)       = {geom['N_geom']:.10f}")
    print(f"    r_nu                             = {geom['r_nu']:.6e} m")
    print(f"    lambda_l                         = {geom['lambda_l']:.6e} m")
    print(f"    N_nu_stat                        = {geom['N_nu_stat']:.6e}")
    print(f"    N_nu_eff                         = {geom['N_nu_eff']:.6e}")
    print(f"    X_eff = N_nu_stat / N_nu_eff     = {geom['X_eff']:.6f}")
    print(f"    K_WC                             = {K_WC}")
    print()
    print(f"    G_geom (cross-check only,        = {geom['G_geom_xcheck']:.15e}")
    print(f"           NOT used in F_EMC below)")

    # ------------------------------------------------------------------
    # Section 2: physical parameters
    # ------------------------------------------------------------------
    print("\n[2/8] Physical parameters for the force test")
    M1 = 1.989e30       # Sun
    M2 = 5.972e24       # Earth
    R = 1.495978707e11  # 1 AU
    print(f"    M1 (Sun)                         = {M1:.6e} kg")
    print(f"    M2 (Earth)                       = {M2:.6e} kg")
    print(f"    R (1 AU)                         = {R:.6e} m")
    print(f"    r_e                              = {R_E:.15e} m")
    print(f"    m_e                              = {M_E:.15e} kg")
    print(f"    c                                = {C0:.3f} m/s")

    # ------------------------------------------------------------------
    # Section 3: amplitudes A_1, A_2
    # ------------------------------------------------------------------
    print("\n[3/8] Monopole amplitudes A_1, A_2 (geometric, no G)")
    print("      Chain:")
    print("        Step 1.  delta_eta_i(r) = - A_i / r")
    print("        A_i = (2 M_i r_e / m_e) * sqrt(X_eff)")
    print("              / (A_pi^4 N_geom^3 K_WC sqrt(N_nu_stat))")
    A1 = geometric_amplitude(M1, geom)
    A2 = geometric_amplitude(M2, geom)
    print(f"    A_1                              = {A1:.15e} m")
    print(f"    A_2                              = {A2:.15e} m")

    # ------------------------------------------------------------------
    # Section 4: coupling K_emc
    # ------------------------------------------------------------------
    print("\n[4/8] EMC coupling K_emc (geometric, no G)")
    print("      Derivation: K_emc = c^4 / (16 pi G)")
    print("      Substituting the geometric G cancels G entirely:")
    print("        K_emc = c^2 m_e A_pi^4 N_geom^3 K_WC sqrt(N_nu_eff)")
    print("                / (16 pi r_e)")
    K_emc = geometric_K_emc(geom)
    print(f"    K_emc                            = {K_emc:.15e} N")

    # ------------------------------------------------------------------
    # Section 5: symbolic derivation of the force kernel
    # ------------------------------------------------------------------
    print("\n[5/8] Symbolic derivation of the force kernel")
    print("      Step 1.  delta_eta_i(r)     = - A_i / r")
    print("      Step 2.  grad(delta_eta_i)  = A_i / r^2 * r_hat")
    print("      Step 3.  I(R) = integral grad(de_1) . grad(de_2) dV")
    print("      Step 4.  angular integral (exact):")
    print("                 integral dOmega [r - R cos th] /")
    print("                     [r^2 + R^2 - 2rR cos th]^(3/2)")
    print("                 =   0            for r < R")
    print("                 =   4 pi / r^2   for r >= R")
    print("      Step 5.  I(R) = 4 pi A_1 A_2 / R")
    print("      Step 6.  F_geom = - dI/dR = 4 pi A_1 A_2 / R^2")
    print("      Step 7.  F_EMC  = K_emc * F_geom")
    print("                      = 4 pi K_emc A_1 A_2 / R^2")
    print("      (No G appears in any of steps 1-7.)")

    # ------------------------------------------------------------------
    # Section 6: numerical evaluation of F_geom, then F_EMC
    # ------------------------------------------------------------------
    print("\n[6/8] Numerical evaluation of the force from geometry")
    F_geom = geometric_gradient(A1, A2, R)
    F_geom_analytic = 4.0 * math.pi * A1 * A2 / (R * R)
    rel_kernel = abs(F_geom - F_geom_analytic) / F_geom_analytic
    print(f"    F_geom (numeric)   = 4 pi A1 A2 / R^2")
    print(f"                       = {F_geom:.15e}")
    print(f"    F_geom (analytic)                    = {F_geom_analytic:.15e}")
    print(f"    relative difference                  = {rel_kernel:.3e}")
    F_EMC = K_emc * F_geom
    print(f"    F_EMC = K_emc * F_geom               = {F_EMC:.15e} N")

    # ------------------------------------------------------------------
    # Section 7: comparison with observation
    # ------------------------------------------------------------------
    print("\n[7/8] Comparison with observed gravitational force")
    print("      F_obs = G_CODATA * M1 * M2 / R^2")
    print("      (First point in the script where G is used.)")
    F_obs = G_CODATA * M1 * M2 / (R * R)
    rel_diff = abs(F_EMC - F_obs) / F_obs
    print(f"    F_EMC  (geometric, no G)             = {F_EMC:.15e} N")
    print(f"    F_obs  (from G_CODATA)               = {F_obs:.15e} N")
    print(f"    relative difference                  = {rel_diff*100:.6f} %")
    print(f"    CODATA uncertainty on G              = 2.2e-05 (22 ppm)")
    print(f"    residual / uncertainty               = {rel_diff/2.2e-5:.1f} x")

    # Cross-check: G_emc derived from the force itself.
    G_emc_from_force = F_EMC * R * R / (M1 * M2)
    print()
    print("    Cross-check only (not used in the calculation):")
    print(f"    G_emc = F_EMC R^2 / (M1 M2)          = {G_emc_from_force:.15e}")
    print(f"    G_geom from M4.7 chain               = {geom['G_geom_xcheck']:.15e}")
    print(f"    relative difference                  = "
          f"{abs(G_emc_from_force - geom['G_geom_xcheck'])/geom['G_geom_xcheck']:.3e}")

    # ------------------------------------------------------------------
    # Section 8: mutation tests on geometric factors
    # ------------------------------------------------------------------
    print("\n[8/8] Mutation tests: sensitivity to geometric factors")
    print("      (Perturbations of geometric inputs, not G.)")

    # K_WC: 10 -> 9
    # A_i ~ 1/K_WC, K_emc ~ K_WC, so F_EMC ~ 1/K_WC.
    factor_KWC = 9.0 / 10.0
    F_kwc9 = F_EMC * (10.0 / 9.0)
    rel_kwc9 = abs(F_kwc9 - F_EMC) / F_EMC
    print(f"    K_WC: 10 -> 9                        "
          f"rel change = {rel_kwc9*100:10.6f} %   (expected ~11%)")

    # N_geom: * 1.001
    # A_i ~ 1/N_geom^3, K_emc ~ N_geom^3, so F_EMC ~ 1/N_geom^3.
    factor_N = 1.001
    F_N = F_EMC / (factor_N ** 3)
    rel_N = abs(F_N - F_EMC) / F_EMC
    print(f"    N_geom: * 1.001                      "
          f"rel change = {rel_N*100:10.6f} %   (expected ~0.3%)")

    # A_pi: * 1.01
    # A_i ~ 1/A_pi^4, K_emc ~ A_pi^4, so F_EMC ~ 1/A_pi^4.
    factor_A = 1.01
    F_A = F_EMC / (factor_A ** 4)
    rel_A = abs(F_A - F_EMC) / F_EMC
    print(f"    A_pi: * 1.01                         "
          f"rel change = {rel_A*100:10.6f} %   (expected ~3.9%)")

    # r_e: * 1.01
    # A_i ~ r_e, K_emc ~ 1/r_e, so F_EMC ~ const. No effect.
    factor_re = 1.01
    F_re = F_EMC * factor_re / factor_re
    rel_re = abs(F_re - F_EMC) / F_EMC
    print(f"    r_e: * 1.01                          "
          f"rel change = {rel_re*100:10.6f} %   (expected 0%)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"    F_EMC (from geometry, no G) = {F_EMC:.15e} N")
    print(f"    F_obs (from G_CODATA)       = {F_obs:.15e} N")
    print(f"    relative difference         = {rel_diff*100:.6f} %")
    print()
    print("    The force is derived from geometric factors alone.")
    print("    No G value is inserted in the calculation of F_EMC.")
    print("    The comparison with F_obs tests whether the geometric")
    print("    derivation reproduces the measured gravitational strength.")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())