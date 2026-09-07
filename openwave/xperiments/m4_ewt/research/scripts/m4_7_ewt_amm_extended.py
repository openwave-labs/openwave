#!/usr/bin/env python3
"""
From manuscript version: 5.0.0

Enhanced EWT -- Extended Lepton AMM Module
============================================
Provides the extended lepton anomalous magnetic moments with internal
shell references and consistency tests between the geometric AMM
sector and the orbital mass sector.

This module complements ewt_emergence_engine.py, which contains the
basic full AMM predictions and rigidity tests.
"""

import math

try:
    from m4_7_ewt_emergence_engine import (
        PI, SQRT2, SQRT3, EULER,
        A_MU_EXP,
        A_TAU_EXP,
        A_E_CODATA,
        compute_alpha_core,
        derive_eps_M_from_BCC,
        get_AMMi_K,
    )
except ImportError:
    raise ImportError("This module requires m4_7_ewt_emergence_engine.py in the same directory.")


def compute_lepton_amms_extended(alpha_geom: float, eps_M: float) -> dict:
    """
    Compute extended lepton AMMs with internal shell references.

    Parameters
    ----------
    alpha_geom : float
        Geometric fine-structure constant.
    eps_M : float
        Magnetic deficit from BCC geometry.

    Returns
    -------
    dict with:
        a_e_ppm
        a_mu_shell_ppm
        a_mu_full_ppm
        a_tau_shell_total_ppm
        a_tau_full_ppm
        ref_a_mu_shell_ppm
        ref_a_tau_shell_ppm
        err_a_mu_shell_pct
        err_a_tau_shell_pct
    """
    A_pi = compute_alpha_core()
    L_mu_dim = 5
    L_tau_dim = 34
    K_e = 10

    # Electron
    a_e_ppm = (alpha_geom / (2.0 * PI)) * (1.0 - eps_M * (PI**3)) * 1e6

    # Muon shell
    K_mu_total = get_AMMi_K(2)
    K_mu_delta = K_mu_total - K_e
    M_mu_shell = K_mu_delta / K_e

    B_mu_scale = (3.0 * A_pi * PI**3) / (2.0 * L_mu_dim**2)
    a_mu_shell_ppm = B_mu_scale * (1.0 - eps_M) ** (M_mu_shell * PI**3)

    # Projection operator for muon (2D -> 1D)
    O_mu = 1.0 / (4.0 * PI**2)

    a_mu_full_ppm = a_e_ppm + a_mu_shell_ppm * O_mu

    # Tau shell
    K_tau_total = get_AMMi_K(3)
    M_tau_rel = K_tau_total / K_e

    B_tau_base = ((3.0 * A_pi * PI**3) / (8.0 * SQRT2)) + (A_pi / 2.0)
    a_tau_shell_raw_ppm = B_tau_base * (1.0 - eps_M) ** (M_tau_rel * PI**3)

    # Recursive accumulation with interface tension L_mu^2
    a_tau_shell_total_ppm = a_mu_shell_ppm + a_tau_shell_raw_ppm + L_mu_dim**2

    # Projection operator for tau (3D -> 3D, unity)
    a_tau_full_ppm = a_e_ppm + (a_tau_shell_total_ppm - a_e_ppm)

    # Internal references (from orbital mass relations)
    ref_a_mu_shell_ppm = 248.8
    ref_a_tau_shell_ppm = 1177.21

    err_a_mu_shell_pct = abs(a_mu_shell_ppm - ref_a_mu_shell_ppm) / ref_a_mu_shell_ppm * 100.0
    err_a_tau_shell_pct = abs(a_tau_shell_total_ppm - ref_a_tau_shell_ppm) / ref_a_tau_shell_ppm * 100.0

    return {
        "a_e_ppm": a_e_ppm,
        "a_mu_shell_ppm": a_mu_shell_ppm,
        "a_mu_full_ppm": a_mu_full_ppm,
        "a_tau_shell_total_ppm": a_tau_shell_total_ppm,
        "a_tau_full_ppm": a_tau_full_ppm,
        "ref_a_mu_shell_ppm": ref_a_mu_shell_ppm,
        "ref_a_tau_shell_ppm": ref_a_tau_shell_ppm,
        "err_a_mu_shell_pct": err_a_mu_shell_pct,
        "err_a_tau_shell_pct": err_a_tau_shell_pct,
    }


def report_extended_amms(res: dict) -> None:
    """Print the extended AMM report in Scilab style."""
    print("\n--- EXTENDED LEPTON ANOMALOUS MAGNETIC MOMENTS ---")
    print(f"  Electron full a_e       : {res['a_e_ppm']:.6f} ppm")
    print(f"    CODATA reference      : {A_E_CODATA*1e6:.6f} ppm")
    print(f"    Full rel. error       : {abs(res['a_e_ppm'] - A_E_CODATA*1e6)/(A_E_CODATA*1e6)*100:.6f} %")
    print()
    print(f"  Muon shell (internal)   : {res['a_mu_shell_ppm']:.6f} ppm")
    print(f"    EWT shell reference   : {res['ref_a_mu_shell_ppm']:.6f} ppm")
    print(f"    Internal rel. error   : {res['err_a_mu_shell_pct']:.6f} %")
    print(f"  Muon full a_mu          : {res['a_mu_full_ppm']:.6f} ppm")
    print(f"    Experimental target   : {A_MU_EXP*1e6:.6f} ppm")
    print(f"    Full rel. error       : {abs(res['a_mu_full_ppm'] - A_MU_EXP*1e6)/(A_MU_EXP*1e6)*100:.6f} %")
    print()
    print(f"  Tau shell total         : {res['a_tau_shell_total_ppm']:.6f} ppm")
    print(f"    EWT shell reference   : {res['ref_a_tau_shell_ppm']:.6f} ppm")
    print(f"    Internal rel. error   : {res['err_a_tau_shell_pct']:.6f} %")
    print(f"  Tau full a_tau          : {res['a_tau_full_ppm']:.6f} ppm")
    print(f"    Experimental target   : {A_TAU_EXP*1e6:.6f} ppm")
    print(f"    Full rel. error       : {abs(res['a_tau_full_ppm'] - A_TAU_EXP*1e6)/(A_TAU_EXP*1e6)*100:.6f} %")


def main():
    print("EWT EXTENDED LEPTON AMM MODULE")

    # Use pure BCC geometry from the main engine
    bcc = derive_eps_M_from_BCC(8.0 * PI**4)
    eps_M = bcc["eps_M"]
    alpha_geom = 1.0 / (compute_alpha_core() - eps_M)

    res = compute_lepton_amms_extended(alpha_geom, eps_M)
    report_extended_amms(res)


if __name__ == "__main__":
    main()
