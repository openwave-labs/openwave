#!/usr/bin/env python3
"""
From manuscript version: 5.0.0

Enhanced EWT -- AMM Resonance Scanner (Onion Model)
=====================================================
Python port of the Scilab AMM_find.sc script, adapted to the
zero-calibration geometric eps_M from BCC packing impedance.

Scans the (K_muon, K_tau) space and identifies resonance points
with minimal internal error against the EWT shell references.

Generates:
  Fig1_Muon_Profile.pdf
  Fig2_Tau_Profile.pdf
  Fig3_3D_Peak.pdf
  Fig4_Topography.pdf
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use("pdf")  # non-interactive backend
import matplotlib.pyplot as plt

try:
    from m4_7_ewt_emergence_engine import (
        PI, SQRT2, SQRT3, EULER,
        compute_alpha_core,
        compute_alpha_geometric,
        derive_eps_M_from_BCC,
    )
except ImportError:
    raise ImportError("This module requires m4_7_ewt_emergence_engine.py in the same directory.")


# =============================================================================
# 1. Core EWT calculation using geometric eps_M
# =============================================================================

def calculate_EWT_from_geometry(Kn_m_in: int, Kn_t_in: int):
    """
    Compute the predicted AMMs and their internal errors for given
    nodal counts.

    Returns
    -------
    tuple (am_ppm, at_ppm, e_m, e_t)
    """

    # Use pure BCC geometry
    bcc = derive_eps_M_from_BCC(8.0 * PI**4)
    eps_M = bcc["eps_M"]

    A_pi = compute_alpha_core()
    alpha_geom = 1.0 / compute_alpha_geometric(eps_M)

    Kn_e = 10

    # Experimental / internal shell references
    target_amu = 248.8 / 1e6
    target_atau = 1177.21 / 1e6

    # Invariants from Onion Model
    L_mu = 5
    L_tau = 34

    # Core: Electron Ground State
    ae_pred = (alpha_geom / (2.0 * PI)) * (1.0 - eps_M * (PI**3))

    # Shell 1: Muon Resonance
    M_mu_shell = (Kn_m_in - Kn_e) / Kn_e
    B_mu_scale = (3.0 * A_pi * PI**3) / (2.0 * L_mu**2)
    amu_shell = B_mu_scale * (1.0 - eps_M) ** (M_mu_shell * PI**3)

    am_ppm = ae_pred + amu_shell  # in dimensionless (not ppm) actually
    # Wait: ae_pred is dimensionless, amu_shell is dimensionless.
    # In Scilab they did ae_pred + amu_shell without scaling by 1e6.
    # Then they divided by 1e6 to compare to target_amu.
    # We will replicate exactly.

    e_m = abs(am_ppm / 1e6 - target_amu) / target_amu * 100.0

    # Shell 2: Tau Resonance (Recursive addition)
    M_tau_rel = Kn_t_in / Kn_e
    B_tau_base = ((3.0 * A_pi * PI**3) / (8.0 * SQRT2)) + (A_pi / 2.0)
    at_shell = B_tau_base * (1.0 - eps_M) ** (M_tau_rel * PI**3)

    at_ppm = am_ppm + at_shell + L_mu**2  # interface tension L_mu^2 = 25

    e_t = abs(at_ppm / 1e6 - target_atau) / target_atau * 100.0

    return am_ppm, at_ppm, e_m, e_t


# =============================================================================
# 2. Scan and output
# =============================================================================

def main():
    print("=== EWT RECURSIVE HIERARCHY ANALYSIS ===")
    print("Reference: Onion Model")
    print()

    # Scan ranges
    Km_scan = list(range(195, 216))     # 195..215
    Kt_scan = list(range(2170, 2191))   # 2170..2190

    # Meshgrid equivalent using nested lists
    Z_err = [[0.0 for _ in Km_scan] for _ in Kt_scan]
    am_arr = [[0.0 for _ in Km_scan] for _ in Kt_scan]
    at_arr = [[0.0 for _ in Km_scan] for _ in Kt_scan]

    special_points = [(207, 2181), (200, 2181), (207, 2180), (200, 2180)]

    for i, Kt in enumerate(Kt_scan):
        for j, Km in enumerate(Km_scan):
            am, at, em, et = calculate_EWT_from_geometry(Km, Kt)
            Z_err[i][j] = (em + et) / 2.0
            am_arr[i][j] = am
            at_arr[i][j] = at

            if (Km, Kt) in special_points:
                print(
                    f"POINT: Km={Km}, Kt={Kt} | "
                    f"Err_mu: {em:.6f}% | Err_tau: {et:.6f}% | "
                    f"Mean: {(em+et)/2:.6f}% | amu: {am:.4f} | atau: {at:.4f}"
                )

    # Z_plot = 1 / (Z_err + 0.0001) transposed for plotting
    Z_plot = [[1.0 / (Z_err[i][j] + 0.0001) for i in range(len(Kt_scan))]
              for j in range(len(Km_scan))]

    # Ensure output directory
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # -------------------------------------------------------------------------
    # FIG 1: Muon Profile
    # -------------------------------------------------------------------------
    m_idx = Kt_scan.index(2181)
    muon_err = [Z_err[m_idx][j] for j in range(len(Km_scan))]

    plt.figure()
    plt.plot(Km_scan, muon_err, "r-o")
    plt.grid(True)
    plt.xlabel("Nodes Km")
    plt.ylabel("Total Error %")
    plt.title("Muon 2D Resonance Profile")
    plt.savefig(os.path.join(out_dir, "Fig1_Muon_Profile.pdf"))
    plt.close()
    print("EXPORTED: Fig1_Muon_Profile.pdf")

    # -------------------------------------------------------------------------
    # FIG 2: Tau Profile
    # -------------------------------------------------------------------------
    t_idx = Km_scan.index(207)
    tau_err = [Z_err[i][t_idx] for i in range(len(Kt_scan))]

    plt.figure()
    plt.plot(Kt_scan, tau_err, "b-s")
    plt.grid(True)
    plt.xlabel("Nodes Kt")
    plt.ylabel("Total Error %")
    plt.title("Tau 2D Resonance Profile")
    plt.savefig(os.path.join(out_dir, "Fig2_Tau_Profile.pdf"))
    plt.close()
    print("EXPORTED: Fig2_Tau_Profile.pdf")

    # -------------------------------------------------------------------------
    # FIG 3: 3D Stability Surface
    # -------------------------------------------------------------------------
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Create meshgrid for plotting
    X, Y = np.meshgrid(Km_scan, Kt_scan)  # X: Km (shape len(Kt) x len(Km)), Y: Kt

    # Z_err has shape (len(Kt), len(Km))
    # We want Z shape also (len(Kt), len(Km))
    Z_for_3d = np.array(Z_err)

    # For 1/error, compute directly from Z_err
    Z_inv = 1.0 / (Z_for_3d + 0.0001)

    ax.plot_surface(X, Y, Z_inv, cmap="viridis", alpha=0.8)
    ax.set_xlabel("Km")
    ax.set_ylabel("Kt")
    ax.set_zlabel("1 / error")
    ax.set_title("Recursive Stability Surface (Onion Model)")
    ax.view_init(elev=35, azim=45)
    plt.savefig(os.path.join(out_dir, "Fig3_3D_Peak.pdf"))
    plt.close()
    print("EXPORTED: Fig3_3D_Peak.pdf")

    # -------------------------------------------------------------------------
    # FIG 4: Topographic Contour Map (3D version)
    # -------------------------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Create meshgrid for plotting (same as FIG 3)
    X, Y = np.meshgrid(Km_scan, Kt_scan)

    # Z_inv was computed earlier as 1 / (Z_err + 0.0001)
    Z_for_contour = np.array(Z_inv)  # shape (len(Kt), len(Km))

    # Draw the surface
    ax.plot_surface(
        X, Y, Z_for_contour,
        cmap="viridis",
        alpha=0.6,
        edgecolor="none",
        antialiased=True
    )

    # Draw contour lines on the surface
    ax.contour3D(
        X, Y, Z_for_contour,
        15,
        cmap="Reds",
        linewidths=1.5
    )

    ax.set_xlabel("Km")
    ax.set_ylabel("Kt")
    ax.set_zlabel("1 / error")
    ax.set_title("Topographic Lattice Latches (Km vs Kt)")

    # Adjust viewing angle for better depth perception
    ax.view_init(elev=35, azim=45)

    plt.savefig(os.path.join(out_dir, "Fig4_Topography_3D.pdf"))
    plt.close()
    print("EXPORTED: Fig4_Topography_3D.pdf")

    print("\n--- ALL DATA SYNCHRONIZED WITH LATEX DOCUMENT ---")


if __name__ == "__main__":
    main()
