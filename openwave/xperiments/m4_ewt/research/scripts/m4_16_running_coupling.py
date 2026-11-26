#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From manuscript version: 5.4.4

M4.16 - Running gravitational coupling (all tested variants & eta_loc).

Compares working transition profiles f(x) (x = r / r_e) in:
    N(r) = N_eff + (N_stat - N_eff) * f(r / r_e)
across monotonic profiles, EMC-wall profiles centered at r = r_e (x = 1.0),
and regularized local metric / deficit variants using eta_loc(x).

Exact internal radial density dynamics inside the soliton are not yet derived
from first-principles BCC lattice dynamics; all profiles are working ansatze
showing that the N_eff -> N_stat transition drives scale running.

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
        E_CHARGE_CODATA,
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


def _clamp01(val):
    return max(0.0, min(1.0, float(val)))


def _woods_saxon_normalized(x, x0=1.0, width=0.25):
    """Smooth step around x0 = 1.0, normalized so f(0) = 0."""
    raw = 1.0 / (1.0 + math.exp(-(x - x0) / width))
    raw_0 = 1.0 / (1.0 + math.exp(-(-x0) / width))
    raw_inf = 1.0
    if abs(raw_inf - raw_0) < 1e-15:
        return 0.0
    return (raw - raw_0) / (raw_inf - raw_0)


def eta_loc_profile(x, delta_core_scale=0.5, p=2.0, wall_amp=0.0, wall_width=0.15):
    """
    Regularized local metric / deficit factor eta_loc(x) around core scale r_e,
    with optional wall perturbation at x = 1.0.
    """
    base_deficit = delta_core_scale / (1.0 + (x**p))
    wall_term = wall_amp * math.exp(-(((x - 1.0) / wall_width) ** 2))
    eta = 1.0 - base_deficit + wall_term
    return max(1e-6, min(1.0 + abs(wall_amp) + 1.0, float(eta)))


def _weight_from_eta_loc(x, delta_core_scale=0.5, p=2.0, wall_amp=0.0, wall_width=0.15):
    """Map eta_loc(x) to normalized transition weight f(x) in [0, 1] with f(0) ~ 0, f(inf) -> 1."""
    eta_x = eta_loc_profile(
        x,
        delta_core_scale=delta_core_scale,
        p=p,
        wall_amp=wall_amp,
        wall_width=wall_width,
    )
    eta_0 = eta_loc_profile(
        0.0,
        delta_core_scale=delta_core_scale,
        p=p,
        wall_amp=wall_amp,
        wall_width=wall_width,
    )
    eta_inf = 1.0
    denom = eta_inf - eta_0
    if abs(denom) < 1e-15:
        return _clamp01((x * x) / (1.0 + x * x))
    return _clamp01((eta_x - eta_0) / denom)


def get_all_profiles():
    """Return all tested transition profiles f(x) for x = r / r_e."""
    return {
        # Monotonic profiles
        "gaussian": lambda x: _clamp01(1.0 - math.exp(-(x * x))),
        "exponential": lambda x: _clamp01(1.0 - math.exp(-x)),
        "rational_x2": lambda x: _clamp01((x * x) / (1.0 + x * x)),
        "woods_saxon": lambda x: _clamp01(_woods_saxon_normalized(x, x0=1.0, width=0.25)),
        # EMC-wall profiles centered at r = r_e (x = 1.0)
        "wall_gaussian_base": lambda x: _clamp01(
            (1.0 - math.exp(-(x * x))) + 0.35 * math.exp(-(((x - 1.0) / 0.18) ** 2))
        ),
        "wall_rational_base": lambda x: _clamp01(
            ((x * x) / (1.0 + x * x)) + 0.40 * math.exp(-(((x - 1.0) / 0.18) ** 2))
        ),
        "wall_narrow_peak": lambda x: _clamp01(
            ((x * x) / (1.0 + x * x)) + 0.45 * math.exp(-(((x - 1.0) / 0.08) ** 2))
        ),
        "wall_wide_shoulder": lambda x: _clamp01(
            ((x * x) / (1.0 + x * x)) + 0.35 * math.exp(-(((x - 1.0) / 0.35) ** 2))
        ),
        "wall_dip_on_re": lambda x: _clamp01(
            ((x * x) / (1.0 + x * x)) - 0.35 * math.exp(-(((x - 1.0) / 0.15) ** 2))
        ),
        # Regularized local metric / deficit variants (eta_loc)
        "eta_loc_p2": lambda x: _weight_from_eta_loc(x, delta_core_scale=0.5, p=2.0),
        "eta_loc_p1": lambda x: _weight_from_eta_loc(x, delta_core_scale=0.5, p=1.0),
        "eta_loc_with_wall": lambda x: _weight_from_eta_loc(
            x, delta_core_scale=0.5, p=2.0, wall_amp=0.15, wall_width=0.15
        ),
    }


def main():
    print("=" * 82)
    print("M4.16 - Running Gravitational Coupling (All Variants & eta_loc)")
    print("=" * 82)
    print()

    # --- 1. Baseline EWT Parameters ---
    K_WC = 10
    bcc = derive_eps_M_from_BCC(8.0 * PI**4)
    N_geom = bcc["N_geom"]
    eps_M = bcc["eps_M"]

    alpha_inv = compute_alpha_geometric(eps_M)
    alpha_geom = 1.0 / alpha_inv
    A_pi = compute_alpha_core()

    q_P = derive_planck_charge_from_e(alpha_geom, E_CHARGE_CODATA)
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
    G_Base = gravity_res["G_Base"]
    alpha_G_core = derive_alpha_G_pure_geom(eps_M, A_pi, N_geom, K_WC, N_eff)
    alpha_G_far_lim = derive_alpha_G_pure_geom(eps_M, A_pi, N_geom, K_WC, N_stat)
    G_far_lim = G_Base * (alpha_G_far_lim / alpha_geom)

    print("[1/3] Reference Endpoints (N_eff -> N_stat transition):")
    print(f"  Core radius r_e              = {r_e:.6e} m")
    print(f"  N_eff (core dilution)        = {N_eff:.6e}")
    print(f"  N_stat (unperturbed lattice) = {N_stat:.6e}")
    print(f"  alpha_G (r -> 0, N_eff)      = {alpha_G_core:.6e} (CODATA: {ALPHA_G_CODATA:.6e})")
    print(f"  alpha_G (r -> inf, N_stat)   = {alpha_G_far_lim:.6e}")
    print(f"  G(r -> inf, N_stat)         = {G_far_lim:.6e} m^3 kg^-1 s^-2 (CODATA G: {G_CODATA:.6e})")
    print()

    # --- 2. Evaluate All Profiles Across Scales ---
    x_pre = np.geomspace(0.01, 0.8, 15)
    x_wall_zone = np.linspace(0.82, 1.25, 31)
    x_post = np.geomspace(1.3, 100.0, 25)
    scales = np.unique(np.concatenate([x_pre, x_wall_zone, x_post]))

    profiles = get_all_profiles()
    results = {}

    for name, f_func in profiles.items():
        N_vals = []
        aG_vals = []
        G_vals = []
        eta_vals = []
        for s in scales:
            x_val = float(s)
            fx = f_func(x_val)
            N_r = N_eff + (N_stat - N_eff) * fx
            aG_r = 1.0 / (
                (A_pi**5) * (1.0 - eps_M / A_pi) * (N_geom**3) * K_WC * math.sqrt(N_r)
            )
            G_r = G_Base * (aG_r / alpha_geom)
            N_vals.append(N_r)
            aG_vals.append(aG_r)
            G_vals.append(G_r)
            eta_vals.append(eta_loc_profile(x_val))
        results[name] = {
            "N": np.array(N_vals),
            "alpha_G": np.array(aG_vals),
            "G": np.array(G_vals),
            "eta_loc": np.array(eta_vals),
        }

    print("[2/3] Comparison Across All Variants at Representative Radii (x = r / r_e):")
    check_xs = [0.01, 0.1, 1.0, 10.0, 100.0]
    for name in profiles:
        print(f"  Variant: {name}")
        print(f"    {'x=r/r_e':>10} {'f(x)':>12} {'N(r)':>15} {'alpha_G(r)':>15} {'G(r) [SI]':>15}")
        for cx in check_xs:
            idx = int(np.argmin(np.abs(scales - cx)))
            s = scales[idx]
            fx = profiles[name](float(s))
            nr = results[name]["N"][idx]
            agr = results[name]["alpha_G"][idx]
            gr = results[name]["G"][idx]
            print(f"    {s:>10.2f} {fx:>12.6e} {nr:>15.6e} {agr:>15.6e} {gr:>15.6e}")
    print()

    # --- 3. Plotting ---
    print("[3/3] Generating multi-variant plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for name in profiles:
        ax1.plot(scales, results[name]["alpha_G"], linewidth=1.6, label=name)
        ax2.plot(scales, results[name]["G"], linewidth=1.6, label=name)

    ax1.axvline(1.0, color="black", linestyle=":", linewidth=1.3, label="r = r_e")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(0.01, 100.0)
    ax1.set_xlabel(r"Normalized Scale $r / r_e$")
    ax1.set_ylabel(r"$\alpha_G(r)$")
    ax1.set_title(r"Running $\alpha_G(r)$ Across All Variants ($N_{\mathrm{eff}} \to N_{\mathrm{stat}}$)")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=7, ncol=2)

    ax2.axvline(1.0, color="black", linestyle=":", linewidth=1.3, label="r = r_e")
    ax2.axhline(G_CODATA, color="gray", linestyle="--", linewidth=1.1, label="CODATA G")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(0.01, 100.0)
    ax2.set_xlabel(r"Normalized Scale $r / r_e$")
    ax2.set_ylabel(r"$G(r)$ [$\mathrm{m^3\,kg^{-1}\,s^{-2}}$]")
    ax2.set_title(r"Running $G(r)$ Across All Variants ($N_{\mathrm{eff}} \to N_{\mathrm{stat}}$)")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(out_dir, "m4_16_running_coupling_all_variants.png")
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()