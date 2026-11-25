#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From manuscript version: not in manuscript yet

M4.17 - Parameter / field dependence G(B) and induced alpha_G(B).

Mechanism:
    Starting from the asymptotic EWT gravitational constant G_EWT derived
    from the BCC vacuum geometry in m4_7_ewt_emergence_engine.py, we
    model an effective coupling G(B) controlled by a local medium /
    field parameter B normalized to a characteristic scale B_0
    (x_B = B / B_0):

        G(B) = G_EWT * f_G(B / B_0),   with   f_G(0) = 1

    and the corresponding dimensionless gravitational coupling:

        alpha_G(B) = alpha_geom * (G(B) / G_Base) = alpha_G_asymp * f_G(B / B_0).

    Physically, B is treated here as a local medium/field state parameter
    in the core neighborhood rather than an arbitrary modification of an
    electron's intrinsic properties. For B << B_0 (weak local fields or
    standard macroscopic conditions), f_G -> 1 and G(B) recovers the
    baseline G_EWT; departures appear only when B approaches or exceeds
    the characteristic scale B_0.

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
        G_CODATA,
        ALPHA_G_CODATA,
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


def f_G_profile(b_ratio: np.ndarray | float, mode: str = "saturation", exponent: float = 2.0):
    """
    Dimensionless scaling factor f_G(B / B_0) such that f_G(0) = 1.

    Supported modes:
      - 'saturation': 1.0 / (1.0 + (B / B_0)**p)  (smooth suppression above B_0)
      - 'gaussian'  : np.exp(-((B / B_0)**p))     (sharp onset near B ~ B_0)
    """
    x = np.asarray(b_ratio, dtype=float)
    if mode == "saturation":
        return 1.0 / (1.0 + np.power(x, exponent))
    if mode == "gaussian":
        return np.exp(-np.power(x, exponent))
    raise ValueError(f"Unsupported mode: {mode}")


def main():
    print("=" * 76)
    print("M4.17 - Parametric / Field Dependence G(B) and alpha_G(B)")
    print("=" * 76)
    print()

    # --- 1. Baseline Geometry from Emergence Engine ---
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

    N_eff = gravity_res["N_nu_eff"]
    G_Base = gravity_res["G_Base"]
    G_EWT = gravity_res["G_EWT"]
    alpha_G_base = derive_alpha_G_pure_geom(eps_M, A_pi, N_geom, K_WC, N_eff)

    # Characteristic reference scale B_0 (normalized units; adjust B_0_phys if needed)
    B_0 = 1.0
    mode = "saturation"
    exponent = 2.0

    print("[1/3] Baseline Parameters from m4_7_ewt_emergence_engine:")
    print(f"    G_Base (c^2 r_e / m_e) = {G_Base:.6e} m^3 kg^-1 s^-2")
    print(f"    G_EWT (B -> 0 limit)   = {G_EWT:.6e} m^3 kg^-1 s^-2")
    print(f"    G_CODATA 2022          = {G_CODATA:.6e} m^3 kg^-1 s^-2")
    print(f"    alpha_G (B -> 0 limit) = {alpha_G_base:.6e}")
    print(f"    alpha_G (CODATA 2022)  = {ALPHA_G_CODATA:.6e}")
    print(f"    Profile f_G(B / B_0)   = {mode} (power p = {exponent}, B_0 = {B_0})")
    print()

    # --- 2. Evaluate G(B) and alpha_G(B) Across B / B_0 ---
    b_ratios = np.logspace(-3, 2, 81)  # B / B_0 from 1e-3 to 1e2
    f_vals = f_G_profile(b_ratios, mode=mode, exponent=exponent)
    G_B_vals = G_EWT * f_vals
    alpha_G_B_vals = alpha_G_base * f_vals

    print("[2/3] Scan over Normalized Parameter B / B_0:")
    print(f"  {'B / B_0':>12} {'f_G(B/B_0)':>14} {'G(B) [SI]':>18} {'alpha_G(B)':>18}")
    print("  " + "-" * 66)

    sample_indices = [0, 16, 32, 40, 48, 64, 80]
    for idx in sample_indices:
        print(
            f"  {b_ratios[idx]:>12.4e} {f_vals[idx]:>14.6e} {G_B_vals[idx]:>18.6e} {alpha_G_B_vals[idx]:>18.6e}"
        )

    low_field_diff = abs(G_B_vals[0] - G_EWT) / G_EWT * 100.0
    at_b0_ratio = f_vals[40]  # B / B_0 = 1.0 at index 40 (10^(-3 + 40*(5/80)) = 10^-0.5? No: 81 pts -> step 5/80 = 0.0625)
    idx_b0 = int(np.argmin(np.abs(b_ratios - 1.0)))

    print("  " + "-" * 66)
    print(f"  Low-field G(1e-3 B_0)  = {G_B_vals[0]:.6e} (rel. diff vs G_EWT: {low_field_diff:.3e} %)")
    print(f"  At B / B_0 = {b_ratios[idx_b0]:.4f}   : f_G = {f_vals[idx_b0]:.6f}, G(B) = {G_B_vals[idx_b0]:.6e}")
    print(f"  High-B G(100 B_0)      = {G_B_vals[-1]:.6e}")
    print()

    # --- 3. Plotting ---
    print("[3/3] Generating plot...")
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))

    color1 = "tab:green"
    ax1.plot(b_ratios, G_B_vals, "-", color=color1, linewidth=2.0, label=r"$G(B)$")
    ax1.axhline(
        G_EWT,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"$G_{\mathrm{EWT}}$ ($B \to 0$)",
    )
    ax1.axvline(
        1.0,
        color="tab:orange",
        linestyle=":",
        linewidth=1.5,
        label=r"Scale $B = B_0$",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Normalized Parameter $B / B_0$")
    ax1.set_ylabel(r"Effective Constant $G(B)$ [$\mathrm{m^3\,kg^{-1}\,s^{-2}}$]", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, which="both", alpha=0.25)

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.plot(
        b_ratios,
        alpha_G_B_vals,
        "-.",
        color=color2,
        linewidth=1.4,
        alpha=0.85,
        label=r"$\alpha_G(B)$",
    )
    ax2.set_ylabel(r"Gravitational Coupling $\alpha_G(B)$", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    plt.title(r"Parametric Dependence of $G(B)$ and $\alpha_G(B)$")
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(out_dir, "m4_17_g_of_b.png")
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()