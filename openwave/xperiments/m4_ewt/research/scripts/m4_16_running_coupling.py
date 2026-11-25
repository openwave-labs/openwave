#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From manuscript version: not in manuscript yet

M4.16 - Running gravitational coupling (scale dependence).

OpenWave criterion:
    Running coupling
    Test: coupling varies with scale (onset at the core radius).

Mechanism:
    In the BCC vacuum framework, the asymptotic dimensionless gravitational
    coupling alpha_G emerges purely from lattice geometry:

        alpha_G_geom = 1 / [ A_pi^5 * (1 - eps_M/A_pi) * N_geom^3 * K_WC * sqrt(N_eff) ]

    which yields alpha_G_geom = 1.752633e-45 (~0.047% from CODATA 2022).
    At macroscopic scales (r >> r_e), the effective vacuum density settles
    at N_eff and both alpha_G(r) and G(r) remain constant. Near and inside
    the soliton core radius (r <= r_e), local lattice compression / core
    structure modifies the effective local density N(r), causing alpha_G(r)
    and G(r) to run with scale (onset at r ~ r_e).

ASCII only. No BOM.
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from m4_7_ewt_emergence_engine import (
        PI,
        C0,
        M_E,
        ALPHA_G_CODATA,
        G_CODATA,
        BCC_IDEAL_PROJECTION_LP,
        compute_alpha_core,
        compute_alpha_geometric,
        derive_eps_M_from_BCC,
        derive_lambda_l_geometric,
        derive_planck_charge_from_e,
        derive_neutrino_radius,
        derive_alpha_G_pure_geom,
        gravity_sector,
    )
except ImportError:
    raise ImportError(
        "This script requires m4_7_ewt_emergence_engine.py in the same directory."
    )


def main():
    print("=" * 76)
    print("M4.16 - Running Gravitational Coupling (Scale Dependence)")
    print("=" * 76)
    print()

    # --- 1. Geometric Decomposition of Asymptotic alpha_G ---
    K_WC = 10
    bcc = derive_eps_M_from_BCC(8.0 * PI**4)
    N_geom = bcc["N_geom"]
    eps_M = bcc["eps_M"]

    alpha_inv = compute_alpha_geometric(eps_M)
    alpha_geom = 1.0 / alpha_inv
    A_pi = compute_alpha_core()

    q_P = derive_planck_charge_from_e(alpha_geom, 1.602176634e-19)
    r_nu = derive_neutrino_radius(alpha_geom, q_P)["r_nu"]
    r_e = 100.0 * r_nu

    lambda_l = derive_lambda_l_geometric(
        alpha_geom, r_e, r_nu, N_geom, BCC_IDEAL_PROJECTION_LP, K_WC
    )

    gravity_res = gravity_sector(
        alpha_geom=alpha_geom,
        r_nu=r_nu,
        N_geom=N_geom,
        L_p_geom=BCC_IDEAL_PROJECTION_LP,
        K_WC=K_WC,
        lambda_l=lambda_l,
        r_e=r_e,
        m_e=M_E,
        c0=C0,
    )

    N_stat = gravity_res["N_nu_statutory"]
    N_eff = gravity_res["N_nu_eff"]
    X_eff = N_stat / N_eff
    G_Base = gravity_res["G_Base"]
    G_EWT_asymp = gravity_res["G_EWT"]

    alpha_G_asymp = derive_alpha_G_pure_geom(eps_M, A_pi, N_geom, K_WC, N_eff)
    rel_err_asymp = abs(alpha_G_asymp - ALPHA_G_CODATA) / ALPHA_G_CODATA * 100.0

    print("[1/3] Geometric Origin of Asymptotic alpha_G (~1.753e-45):")
    print("  Formula: alpha_G = 1 / [ A_pi^5 * (1 - eps_M/A_pi) * N_geom^3 * K_WC * sqrt(N_eff) ]")
    print(f"    A_pi                 = {A_pi:.12f}")
    print(f"    A_pi^5               = {A_pi**5:.6e}  (5D geometric ladder)")
    print(f"    (1 - eps_M / A_pi)   = {1.0 - eps_M / A_pi:.12f}  (vacuum correction)")
    print(f"    N_geom^3             = {N_geom**3:.6e}  (BCC lattice stiffness)")
    print(f"    K_WC                 = {K_WC}             (wave-center count)")
    print(f"    N_stat               = {N_stat:.6e} (dimensionless)")
    print(f"    X_eff = N_stat/N_eff = {X_eff:.6f}      (volume deficit factor)")
    print(f"    N_eff                = {N_eff:.6e} (dimensionless)")
    print(f"    sqrt(N_eff)          = {math.sqrt(N_eff):.6e} (dimensionless)")
    print("  ------------------------------------------------------------------------")
    print(f"    alpha_G (EWT geom)   = {alpha_G_asymp:.6e}")
    print(f"    alpha_G (CODATA)     = {ALPHA_G_CODATA:.6e}")
    print(f"    Relative error       = {rel_err_asymp:.6f} %")
    print(f"    G_EWT (asymptotic)   = {G_EWT_asymp:.6e} m^3 kg^-1 s^-2")
    print(f"    Core radius r_e      = {r_e:.6e} m")
    print()

    # --- 2. Scale-Dependent Effective Density Profile N(r) ---
    # For r >> r_e, N(r) -> N_eff (recovering exact asymptotic alpha_G and G_EWT).
    # For r <= r_e, core lattice compression increases local density toward N_stat,
    # producing a smooth running of alpha_G(r) and G(r) with onset at r ~ r_e.
    def N_local(r):
        core_envelope = np.exp(-((r / r_e) ** 2))
        return N_eff + (N_stat - N_eff) * core_envelope

    def alpha_G_running(r):
        N_r = N_local(r)
        return 1.0 / (
            (A_pi**5) * (1.0 - eps_M / A_pi) * (N_geom**3) * K_WC * math.sqrt(N_r)
        )

    def G_running(r):
        # Since alpha_G = alpha_geom * (G / G_Base), G(r) scales directly with alpha_G(r)
        return G_Base * (alpha_G_running(r) / alpha_geom)

    # --- 3. Evaluate Across Scales ---
    scales = np.logspace(-1, 2, 61)  # 0.1 r_e to 100 r_e
    alpha_G_vals = [alpha_G_running(s * r_e) for s in scales]
    G_vals = [G_running(s * r_e) for s in scales]

    print("[2/3] Scale Dependence of alpha_G(r) and G(r):")
    print(f"  {'r / r_e':>10} {'N(r) [-]':>16} {'alpha_G(r)':>16} {'G(r) [SI]':>16}")
    print("  " + "-" * 64)

    sample_indices = [0, 10, 15, 20, 25, 30, 40, 60]
    for idx in sample_indices:
        s = scales[idx]
        r = s * r_e
        print(
            f"  {s:>10.2f} {N_local(r):>16.6e} {alpha_G_vals[idx]:>16.6e} {G_vals[idx]:>16.6e}"
        )

    alpha_G_far = alpha_G_vals[-1]
    err_far = abs(alpha_G_far - ALPHA_G_CODATA) / ALPHA_G_CODATA * 100.0
    ratio_core_to_far = alpha_G_vals[0] / alpha_G_far

    print("  " + "-" * 64)
    print(f"  Asymptotic alpha_G (r = 100 r_e) = {alpha_G_far:.6e}")
    print(f"  CODATA 2022 reference            = {ALPHA_G_CODATA:.6e}")
    print(f"  Asymptotic relative error        = {err_far:.6f} %")
    print(f"  Core-to-asymptotic ratio (0.1r_e)= {ratio_core_to_far:.6e}")
    print("  By construction: the profile's width is r_e and its r = 0 value is N_stat.")
    print()

    # --- 4. Plotting ---
    print("[3/3] Generating plot...")
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))

    color1 = "tab:blue"
    ax1.plot(scales, alpha_G_vals, "-", color=color1, linewidth=2.0, label=r"$\alpha_G(r)$ (EWT)")
    ax1.axhline(
        ALPHA_G_CODATA,
        color="tab:red",
        linestyle="--",
        linewidth=1.4,
        label=r"$\alpha_G$ CODATA 2022",
    )
    ax1.axvline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=r"Core Radius ($r = r_e$)",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Normalized Scale $r / r_e$")
    ax1.set_ylabel(r"Gravitational Coupling $\alpha_G(r)$", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, which="both", alpha=0.25)

    # Secondary y-axis showing G(r) directly
    ax2 = ax1.twinx()
    color2 = "tab:green"
    ax2.plot(scales, G_vals, "--", color=color2, linewidth=1.2, alpha=0.75, label=r"$G(r)$")
    ax2.axhline(G_CODATA, color="gray", linestyle="-.", linewidth=1.0, alpha=0.6)
    ax2.set_ylabel(r"Effective Newton Constant $G(r)$ [$\mathrm{m^3\,kg^{-1}\,s^{-2}}$]", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combine legends cleanly
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.title(r"Running Gravitational Coupling $\alpha_G(r)$ and $G(r)$ (Onset at $r_e$)")
    fig.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(script_dir, "..", "plots"))
    plot_path = os.path.join(out_dir, "m4_16_running_coupling.png")
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
