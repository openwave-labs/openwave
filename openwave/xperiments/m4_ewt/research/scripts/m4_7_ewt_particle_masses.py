#!/usr/bin/env python3
"""
From manuscript version: 5.0.0

Enhanced EWT -- Particle Mass Module
======================================
Computes particle masses using the three EWT modes:
  - spherical mode (fundamental cores)
  - orbital mode (muon, tau)
  - meson mode (K^5 scaling)

Performs a full scan of PDG/CODATA particles and identifies natural
integer-K resonances.

This module should be used together with ewt_emergence_engine.py.
"""

import math

# --- Import shared constants ---
try:
    from m4_7_ewt_emergence_engine import PI, EULER, SQRT2, SQRT3
except ImportError:
    raise ImportError("This module requires m4_7_ewt_emergence_engine.py in the same directory.")

# -----------------------------------------------------------------------------
# 1. Fundamental wave constants (from Jeff Yee's original EWT)
# -----------------------------------------------------------------------------
RHO_A     = 3.8597645397410479e22    # aether density [kg/m^3]
A_LONG    = 9.2154057079234868e-19   # longitudinal wave amplitude [m]
L_LONG    = 2.8540965006585549e-17   # longitudinal wavelength [m]
C_LIGHT   = 299792458.0              # speed of light [m/s]
J_TO_GEV  = 6.24150934e9             # Joule to GeV conversion


def get_Ol(K: int) -> float:
    """
    Shell energy summation for the spherical mode.

    O_l = sum_{n=1}^{K} (n^3 - (n-1)^3) / n^4
    """
    return sum((n**3 - (n - 1) ** 3) / (n**4) for n in range(1, K + 1))


def mass_spherical(K: int) -> float:
    """
    Longitudinal energy equation for spherical mode.

    Returns mass in GeV.
    """
    E_j = (RHO_A * (4.0 / 3.0) * PI * K**5 * A_LONG**6 * C_LIGHT**2) / L_LONG**3
    return E_j * get_Ol(K) * J_TO_GEV


def mass_orbital(K: int) -> float:
    """
    Orbital mode for muon (K=20) and tau (K=50).
    """
    E_e = mass_spherical(10)
    if K == 20:
        return E_e * 185.68543
    elif K == 50:
        return E_e * 3436.795
    else:
        raise ValueError("Orbital mode only supports K=20 (muon) or K=50 (tau).")


def mass_meson_style(K: int) -> float:
    """
    Meson mode: m(K) = m_e * (K / K_e)^5.
    """
    m_e_GeV = 0.00051099895
    K_e = 10
    return m_e_GeV * (K / K_e) ** 5


def K_from_mass(m_target: float) -> float:
    """
    Inverse meson mode: find K from a target mass.
    """
    m_e_GeV = 0.00051099895
    K_e = 10
    return K_e * (m_target / m_e_GeV) ** (1.0 / 5.0)


# -----------------------------------------------------------------------------
# 2. PARTICLE DATA TABLE (full scan from Scilab Part XIII)
# -----------------------------------------------------------------------------
PARTICLE_DATA = [
    # Leptons
    ("Neutrino",        "PDG 2022",    0.00000000238),
    ("Electron",        "CODATA 2022", 0.00051099895),
    ("Muon",            "PDG 2022",    0.10565837),
    ("Tau",             "PDG 2022",    1.77686),

    # Quarks
    ("Quark u",         "PDG 2022",    0.002162),
    ("Quark d",         "PDG 2022",    0.004692),
    ("Quark s",         "PDG 2022",    0.094954),
    ("Quark c",         "PDG 2022",    1.2730),
    ("Quark b",         "PDG 2022",    4.1830),
    ("Quark t",         "PDG 2022",    172.690),

    # Gauge bosons
    ("W boson",         "PDG 2022",    80.3770),
    ("W boson",         "CDF II 2022", 80.4335),
    ("Z boson",         "PDG 2022",    91.1876),
    ("Higgs",           "PDG 2022",    125.25),

    # Baryons
    ("Proton",          "CODATA 2022", 0.93827208816),
    ("Neutron",         "CODATA 2022", 0.93956542052),
    ("Lambda",          "PDG 2022",    1.11568),
    ("Sigma+",          "PDG 2022",    1.18937),
    ("Sigma0",          "PDG 2022",    1.19264),
    ("Sigma-",          "PDG 2022",    1.19745),
    ("Xi0",             "PDG 2022",    1.31486),
    ("Xi-",             "PDG 2022",    1.32171),
    ("Omega-",          "PDG 2022",    1.67245),

    # Charmed baryons
    ("Lambda_c+",       "PDG 2022",    2.28646),
    ("Sigma_c++",       "PDG 2022",    2.45397),
    ("Xi_c+",           "PDG 2022",    2.46771),
    ("Xi_c0",           "PDG 2022",    2.47044),
    ("Omega_c0",        "PDG 2022",    2.69530),
    ("Xi_cc++",         "PDG 2022",    3.62155),
    ("Xi_cc+",          "LHCb 2026",   3.61997),

    # Mesons
    ("Pion+-",          "PDG 2022",    0.13957039),
    ("Pion0",           "PDG 2022",    0.13497770),
    ("Kaon+-",          "PDG 2022",    0.49367700),
    ("Kaon0",           "PDG 2022",    0.49761700),
    ("Eta",             "PDG 2022",    0.54753),
    ("Rho770",          "PDG 2022",    0.77526),
    ("Omega782",        "PDG 2022",    0.78265),
    ("Phi1020",         "PDG 2022",    1.01946),
    ("D0 meson",        "PDG 2022",    1.86484),
    ("D+ meson",        "PDG 2022",    1.86966),
    ("D_s+",            "PDG 2022",    1.96835),
    ("J/psi",           "PDG 2022",    3.09690),
    ("B+ meson",        "PDG 2022",    5.27934),
    ("B0 meson",        "PDG 2022",    5.27965),
    ("B_s0",            "PDG 2022",    5.36688),
    ("B_c*+",           "ATLAS 2026",  6.3390),
    ("Upsilon(1S)",     "PDG 2022",    9.46030),
    ("Upsilon(2S)",     "PDG 2022",    10.02326),
    ("Upsilon(3S)",     "PDG 2022",    10.35520),
    ("Z_c(3900)",       "PDG 2022",    3.8884),
    ("X(3872)",         "PDG 2022",    3.87165),
    ("Omega_cc*",       "CERN 2026",   3.7259),
]


def run_mass_scan(particle_data: list) -> list:
    """
    Scan particles, compute K_exact, K_int, meson mass, and error.

    Returns a list of tuples for near-integer resonances.
    """
    near_integer = []
    print(f"{'Particle':<16} | {'Source':<12} | {'Target [GeV]':>14} | {'K_exact':>8} | {'K_int':>8} | {'m_int [GeV]':>12} | {'err_int %':>10}")
    print("-" * 110)

    for name, source, m_t in particle_data:
        K_ex = K_from_mass(m_t)
        K_in = round(K_ex)
        m_int = mass_meson_style(K_in)
        err = abs(m_int - m_t) / m_t * 100

        print(f"{name:<16} | {source:<12} | {m_t:14.8f} | {K_ex:8.4f} | {K_in:8d} | {m_int:12.6f} | {err:10.4f}")

        if abs(K_ex - K_in) < 0.15:
            near_integer.append((name, source, K_ex, K_in, m_t, m_int, err))

    return near_integer


def report_near_integer(near_integer: list) -> None:
    """Print natural integer-K resonances."""
    print("\n--- NEAR-INTEGER K RESONANCES (|K - round(K)| < 0.15) ---")
    print("Natural EWT lattice alignment without parameter adjustment.")
    print("-" * 100)
    for name, source, K_ex, K_in, m_t, m_int, err in near_integer:
        print(f"*** {name:<16} [{source:<12}]  K={K_ex:.6f} -> K_int={K_in:3d}  m_int={m_int:.8f} GeV  err={err:.4f}%")
    print("-" * 100)


def report_spherical_mass_table() -> None:
    """
    Print the spherical/fermion mass table as in Scilab Part VI.
    """
    print("\n" + "-" * 80)
    print("    ENERGY WAVE THEORY: SUBATOMIC MASS PREDICTION ENGINE")
    print("    Validated against: Particle-Forces-Calculations-v7.1.xlsx")
    print("-" * 80)
    print(f"{'Particle':<12} | {'K':>3} | {'Calculated [GeV]':>18} | {'Error':>8}")
    print("-" * 80)

    # Tuples: (name, K, target, mode)
    mass_entries = [
        ("Neutrino",    1,  0.00000000238, "sph"),
        ("Quark u",     13, 0.002162,      "sph"),
        ("Electron",    10, 0.00051099,    "sph"),
        ("Quark d",     15, 0.004692,      "sph"),
        ("Muon",        20, 0.09488543,    "orb"),
        ("Quark s",     28, 0.094954,      "sph"),
        ("Tau",         50, 1.75619909,    "orb"),
        ("Omega_cc*",   58, 3.7259,        "sph"),
        ("W Boson",    109, 80.387,        "sph"),
        ("Z Boson",    110, 91.182,        "sph"),
        ("Higgs",      117, 124.9613,      "sph"),
    ]

    for name, K_val, target, mode in mass_entries:
        if mode == "sph":
            res = mass_spherical(K_val)
        else:
            res = mass_orbital(K_val)
        err = abs(res - target) / target * 100
        print(f"{name:<12} | {K_val:3d} | {res:18.12f} | {err:.4f}%")

    print("-" * 80)


def main():
    print("EWT PARTICLE MASS MODULE")

    # Full meson-mode scan
    near = run_mass_scan(PARTICLE_DATA)
    report_near_integer(near)

    # Spherical/orbital mass table
    report_spherical_mass_table()


if __name__ == "__main__":
    main()
