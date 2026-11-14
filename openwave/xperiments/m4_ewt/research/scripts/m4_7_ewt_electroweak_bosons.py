#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From manuscript version: 5.3.2 (not published yet) or leter

Enhanced EWT -- Electroweak and Heavy Boson Module
====================================================
Extends the main emergence engine with:
  - Mixed W-Z vertex operator (harmonic pi^6 - pi^7 combination)
  - Weinberg sector (W mass via mixed operator, default)
  - Pure pi^6 W mass (for comparison / neutral vertices)
  - Higgs-vector boson mixing angles (ZH pure, WH mixed)
  - Cabibbo angle (surface mixing)
  - heavy boson geometric radii (Z, H)
  - comparison with experimental W-mass benchmarks (PDG 2024, ATLAS, CMS)

This module supersedes the standalone ewt_mixed_interaction_operator.py,
which is now integrated here.

All functions that return a computed quantity return a dict
(matching the convention of ewt_emergence_engine.py).

This module should be used together with ewt_emergence_engine.py
and ewt_particle_masses.py.
"""

import math

# --- Import shared constants and functions ---
try:
    from m4_7_ewt_emergence_engine import (
        PI, SQRT2, SQRT3, EULER,
        M_Z_CODATA, M_H_CODATA,
        BCC_PACKING_FRACTION,
        E_CHARGE_CODATA,
        compute_alpha_core,
        compute_alpha_geometric,
        derive_eps_M_from_BCC,
        derive_planck_charge_from_e,
        derive_neutrino_radius,
        build_geometric_ladder,
        lattice_impedance,
    )
    from m4_7_ewt_particle_masses import mass_spherical
except ImportError:
    raise ImportError(
        "This module requires m4_7_ewt_emergence_engine.py and m4_7_ewt_particle_masses.py."
    )


# ============================================================
# Physical constants (CODATA / PDG) and experimental benchmarks
# ============================================================

SIN2_THETA_W_MSBAR = 0.23122   # MS-bar at M_Z (CODATA)
# Flavor deficit is computed at runtime from the lattice impedance zeta:
# eps_flavor = 1 / (8 * pi^5 * (1 - zeta))
# analogous to eps_M = 1 / (8 * pi^7 * (1 - zeta)) for charged weak.
# The ideal value (zeta = 0) is 1/(8*pi^5) ~ 4.085e-4; with zeta ~ 6e-4,
# the correction shifts eps_flavor by ~0.06%.

# Standard Model reference values
M_W_SM_TREE = M_Z_CODATA * math.sqrt(1.0 - SIN2_THETA_W_MSBAR)
M_W_SM_FULL = 80.360           # radiative-corrected SM reference

# Experimental W-mass benchmarks: (mass [GeV], 1-sigma uncertainty [GeV])
EXP_BENCHMARKS = {
    "CDF II 2022":      (80.4335, 0.0094),
    "PDG 2022 average": (80.377,  0.012),
    "ATLAS 2024":       (80.360,  0.016),
    "CMS 2024":         (80.3602, 0.0099),
    "PDG 2024 average": (80.3692, 0.0133),
}
REFERENCE_BENCHMARK = "PDG 2024 average"


# ============================================================
# Mixed interaction operator (harmonic mean of pi^6 and pi^7)
# ============================================================

def mixed_operator(pi6: float, pi7: float) -> float:
    """
    Harmonic combination of the neutral (pi^6) and charged (pi^7) rungs.

    O_mixed = pi^6 * pi^7 / (pi^6 + pi^7)

    Limiting behavior:
      - pi^7 -> infinity : O_mixed -> pi^6  (neutral channel dominant)
      - pi^6 -> infinity : O_mixed -> pi^7  (charged channel dominant)

    This is the unique simple parameter-free combination that reduces
    to the correct single-rung limit in either decoupling case.
    Arithmetic and geometric means both diverge in these limits.
    """
    return (pi6 * pi7) / (pi6 + pi7)


def c_gap_mixed(c_local: float, pi6: float, pi7: float) -> dict:
    """
    Return the mixed C_gap operator and the underlying O_mixed value.

    Returns
    -------
    dict with keys:
        C_gap   : 1 + O_mixed * C_local
        O_mixed : harmonic mean of pi^6 and pi^7
    """
    o_mixed = mixed_operator(pi6, pi7)
    return {"C_gap": 1.0 + o_mixed * c_local, "O_mixed": o_mixed}


# ============================================================
# Weinberg sector (W mass)
# ============================================================

def weinberg_sector(C_gap: float, M_Z: float, sin2_W: float) -> dict:
    """
    Compute W boson mass for a given C_gap operator.

        M_W = M_Z * sqrt( (1 - sin^2(theta_W)) * C_gap )

    Parameters
    ----------
    C_gap : float
        Either the pure pi^6 operator or the mixed operator; the caller
        chooses which one based on the physics (neutral vertex vs mixed
        W-Z vertex).
    M_Z : float
        Z-boson mass [GeV].
    sin2_W : float
        MS-bar Weinberg angle.

    Returns
    -------
    dict with keys:
        M_W_pred : predicted W mass [GeV]
        C_gap    : the operator value used
        sin2_W   : the Weinberg angle input
    """
    M_W_pred = M_Z * math.sqrt((1.0 - sin2_W) * C_gap)
    return {"M_W_pred": M_W_pred, "C_gap": C_gap, "sin2_W": sin2_W}


def weinberg_sector_mixed(
    c_local: float, pi6: float, pi7: float,
    M_Z: float, sin2_W: float,
) -> dict:
    """
    Convenience wrapper: compute W mass via the mixed W-Z operator.

    Returns
    -------
    dict with keys:
        M_W_pred : predicted W mass [GeV]
        C_gap    : the mixed operator value used
        O_mixed  : the harmonic mean of pi^6 and pi^7
        sin2_W   : the Weinberg angle input
    """
    cg = c_gap_mixed(c_local, pi6, pi7)
    result = weinberg_sector(cg["C_gap"], M_Z, sin2_W)
    result["O_mixed"] = cg["O_mixed"]
    return result


# ============================================================
# Higgs-vector boson mixing angles
# ============================================================

def compute_higgs_mixing_angles(
    C_gap_ZH: float,
    C_gap_WH: float,
    M_Z_EWT: float,
    M_H_EWT: float,
    M_W_EWT: float,
) -> dict:
    """
    Compute Higgs-vector mixing angles.

    Z-H uses the pure pi^6 operator (both solitons neutral);
    W-H uses the mixed operator (one charged, one neutral soliton).

    Returns
    -------
    dict with keys:
        sin2_ZH : sin^2(theta_ZH) using C_gap_ZH (pure pi^6)
        sin2_WH : sin^2(theta_WH) using C_gap_WH (mixed)
    """
    sin2_ZH = 1.0 - ((M_Z_EWT / M_H_EWT) ** 2) * (1.0 / C_gap_ZH)
    sin2_WH = 1.0 - ((M_W_EWT / M_H_EWT) ** 2) * (1.0 / C_gap_WH)
    return {"sin2_ZH": sin2_ZH, "sin2_WH": sin2_WH}


# ============================================================
# Cabibbo sector (surface mixing)
# ============================================================

def cabibbo_sector(
    M_d_EWT: float,
    M_s_EWT: float,
    M_d_PDG: float,
    M_s_PDG: float,
    eps_flavor: float = None,
    target: float = 0.22430,
) -> dict:
    """
    Compute the Cabibbo angle with the flavor deficit correction.

    Formula:
        sin(theta_C) = sqrt(M_d/M_s) * (1 + eps_flavor)

    where eps_flavor = 1/(8*pi^5*(1-zeta)) is the flavor deficit at
    the pi^5 rung, analogous to eps_M = 1/(8*pi^7*(1-zeta)) at the
    charged weak rung. If eps_flavor is None, the ideal limit
    (zeta = 0) value 1/(8*pi^5) is used.

    Parameters
    ----------
    M_d_EWT, M_s_EWT : float
        Spherical-mode quark masses [GeV] (Variant A).
    M_d_PDG, M_s_PDG : float
        PDG light-quark masses [GeV] (Variant B/C).
    eps_flavor : float, optional
        Flavor deficit. Default None -> ideal 1/(8*pi^5).
    target : float
        Reference sin(theta_C), default PDG 0.22430.

    Returns
    -------
    dict with keys:
        eps_flavor   : the flavor deficit value
        correction   : (1 + eps_flavor)
        sinC_A       : prediction with EWT masses
        sinC_B       : prediction with PDG masses
        err_A_pct    : relative error of Variant A [%]
        err_B_pct    : relative error of Variant B [%]
        target       : reference value
    """
    if eps_flavor is None:
        eps_flavor = 1.0 / (8.0 * math.pi**5)  # ideal limit (zeta = 0)
    correction = 1.0 + eps_flavor
    sinC_A = math.sqrt(M_d_EWT / M_s_EWT) * correction
    sinC_B = math.sqrt(M_d_PDG / M_s_PDG) * correction
    return {
        "eps_flavor": eps_flavor,
        "correction": correction,
        "sinC_A": sinC_A,
        "sinC_B": sinC_B,
        "err_A_pct": abs(sinC_A - target) / target * 100.0,
        "err_B_pct": abs(sinC_B - target) / target * 100.0,
        "target": target,
        "M_d_EWT": M_d_EWT,
        "M_s_EWT": M_s_EWT,
        "M_d_PDG": M_d_PDG,
        "M_s_PDG": M_s_PDG,
    }


# ============================================================
# Heavy boson radii
# ============================================================

def compute_boson_radii(r_e_geom: float) -> dict:
    """
    Predict radii of heavy neutral bosons from spherical mode masses
    using r = r_e * (m / m_e)^(1/5).

    Returns
    -------
    dict with keys:
        r_Z, r_H : predicted radii [m]
        m_Z, m_H : spherical-mode masses [GeV]
    """
    m_e_GeV = mass_spherical(10)
    m_Z = mass_spherical(110)
    m_H = mass_spherical(117)

    r_Z = r_e_geom * (m_Z / m_e_GeV) ** (1.0 / 5.0)
    r_H = r_e_geom * (m_H / m_e_GeV) ** (1.0 / 5.0)

    return {"r_Z": r_Z, "r_H": r_H, "m_Z": m_Z, "m_H": m_H}


# ============================================================
# Weinberg angle consistency test
# ============================================================

def test_weinberg_angle(
    M_Z: float, M_W: float, C_gap: float,
    label: str, target_sin2_W: float = SIN2_THETA_W_MSBAR,
) -> dict:
    """
    Compute sin^2(theta_W) from given boson masses and C_gap.

        sin^2(theta_W) = 1 - (M_W/M_Z)^2 * (1/C_gap)

    Returns
    -------
    dict with keys:
        sin2_W_pred  : reconstructed Weinberg angle
        target       : reference value
        rel_err_pct  : relative error [%]
        label        : input label
    """
    sin2_W_pred = 1.0 - ((M_W / M_Z) ** 2) * (1.0 / C_gap)
    rel_err_pct = abs(sin2_W_pred - target_sin2_W) / target_sin2_W * 100.0
    return {
        "sin2_W_pred": sin2_W_pred,
        "target": target_sin2_W,
        "rel_err_pct": rel_err_pct,
        "label": label,
    }


# ============================================================
# Benchmark comparison
# ============================================================

def compare_w_mass_benchmarks(
    M_W_EWT: float,
    benchmarks: dict = None,
    reference_name: str = REFERENCE_BENCHMARK,
) -> dict:
    """
    Compare an EWT W-mass prediction with experimental benchmarks.

    Returns
    -------
    dict with keys:
        M_W_EWT    : the EWT prediction [GeV]
        reference  : name of the reference benchmark
        deviations : {benchmark_name: {M_W, sigma, delta_mev, sigma_dev}}
    """
    if benchmarks is None:
        benchmarks = EXP_BENCHMARKS
    ref_M, _ = benchmarks[reference_name]
    deviations = {}
    for name, (M, sigma) in benchmarks.items():
        deviations[name] = {
            "M_W": M,
            "sigma": sigma,
            "delta_mev": (M_W_EWT - M) * 1000.0,
            "delta_vs_ref_mev": (M - ref_M) * 1000.0,
            "sigma_dev": (M_W_EWT - M) / sigma,
        }
    return {
        "M_W_EWT": M_W_EWT,
        "reference": reference_name,
        "deviations": deviations,
    }


# ============================================================
# Sector runners (encapsulate geometry + computation + print)
# ============================================================

def run_geometry_summary(geom: dict) -> None:
    print("--- GEOMETRY ---")
    print(f"  C_local             = {geom['C_local']:.6e}")
    print(f"  pi^6                = {geom['pi6']:.6f}")
    print(f"  pi^7                = {geom['pi7']:.6f}")
    print(f"  C_gap (pure pi^6)   = {geom['C_gap_pure']:.10f}")
    print(f"  C_gap (mixed)       = {geom['C_gap_mixed']:.10f}")
    print(f"  O_mixed             = {geom['O_mixed']:.4f}")
    print()


def run_mixed_w_sector(geom: dict) -> dict:
    result = weinberg_sector_mixed(
        geom["C_local"], geom["pi6"], geom["pi7"],
        M_Z_CODATA, SIN2_THETA_W_MSBAR,
    )
    print("--- MIXED W-Z VERTEX (main prediction) ---")
    print(f"  O_mixed        = {result['O_mixed']:.4f}")
    print(f"  C_gap (mixed)  = {result['C_gap']:.10f}")
    print(f"  M_W_EWT        = {result['M_W_pred']:.4f} GeV")
    print()
    return result


def run_pure_w_sector(geom: dict) -> dict:
    result = weinberg_sector(geom["C_gap_pure"], M_Z_CODATA, SIN2_THETA_W_MSBAR)
    print("--- PURE pi^6 W MASS (for comparison) ---")
    print(f"  C_gap (pure pi^6)  = {result['C_gap']:.10f}")
    print(f"  M_W (pure pi^6)    = {result['M_W_pred']:.4f} GeV")
    print()
    return result


def run_benchmark_comparison(M_W_EWT: float) -> dict:
    result = compare_w_mass_benchmarks(M_W_EWT)
    print("--- BENCHMARK COMPARISON (EWT mixed vs experiments) ---")
    print(f"  Reference: {result['reference']}")
    print(f"  {'Benchmark':<20} {'M_W [GeV]':<22} {'EWT-bench [MeV]':>16} {'sigma':>8}")
    print("  " + "-" * 66)
    for name, d in result["deviations"].items():
        print(f"  {name:<20} {d['M_W']:.4f} +/- {d['sigma']:.4f}   "
              f"{d['delta_mev']:>+10.1f} {d['sigma_dev']:>+7.1f} sigma")
    print()
    return result


def run_higgs_sector(geom: dict, M_W_EWT: float) -> dict:
    M_Z_EWT = mass_spherical(110)
    M_H_EWT = mass_spherical(117)
    mixing = compute_higgs_mixing_angles(
        C_gap_ZH=geom["C_gap_pure"],
        C_gap_WH=geom["C_gap_mixed"],
        M_Z_EWT=M_Z_EWT,
        M_H_EWT=M_H_EWT,
        M_W_EWT=M_W_EWT,
    )
    print("--- HIGGS-VECTOR MIXING ---")
    print(f"  Z (K=110)   = {M_Z_EWT:.10f} GeV  (spherical mode)")
    print(f"  H (K=117)   = {M_H_EWT:.10f} GeV  (spherical mode)")
    print(f"  sin^2(theta_ZH) [pure pi^6] = {mixing['sin2_ZH']:.10f}")
    print(f"  sin^2(theta_WH) [mixed]     = {mixing['sin2_WH']:.10f}")
    print()
    mixing["M_Z_EWT"] = M_Z_EWT
    mixing["M_H_EWT"] = M_H_EWT
    return mixing


def run_weinberg_tests(geom: dict, M_W_EWT: float, M_Z_EWT: float) -> dict:
    print("--- WEINBERG ANGLE CONSISTENCY TESTS ---")
    results = {}

    r1 = test_weinberg_angle(M_Z_EWT, M_W_EWT, geom["C_gap_mixed"],
                             "EWT spherical masses + mixed")
    print(f"  [{r1['label']}]")
    print(f"    M_Z = {M_Z_EWT:.6f}, M_W = {M_W_EWT:.6f}")
    print(f"    sin^2(theta_W) = {r1['sin2_W_pred']:.10f}  "
          f"(rel. err. {r1['rel_err_pct']:.4f}%)")
    results["ewt_spherical"] = r1

    r2 = test_weinberg_angle(M_Z_CODATA, M_W_SM_FULL, geom["C_gap_mixed"],
                             "empirical masses (PDG 2024) + mixed")
    print(f"  [{r2['label']}]")
    print(f"    M_Z = {M_Z_CODATA:.6f}, M_W = {M_W_SM_FULL:.6f}")
    print(f"    sin^2(theta_W) = {r2['sin2_W_pred']:.10f}  "
          f"(rel. err. {r2['rel_err_pct']:.4f}%)")
    results["empirical"] = r2

    print()
    return results


def run_radii_sector(r_e_geom: float) -> dict:
    radii = compute_boson_radii(r_e_geom)
    print("--- HEAVY BOSON RADII (spherical mode, r ~ m^(1/5)) ---")
    print(f"  Z (K=110): r = {radii['r_Z']:.4e} m")
    print(f"  H (K=117): r = {radii['r_H']:.4e} m")
    print("  Order of magnitude matches nuclear mass-equivalents.")
    print()
    return radii


def run_cabibbo_sector(geom: dict) -> dict:
    M_d_EWT = mass_spherical(15)
    M_s_EWT = mass_spherical(28)
    
    zeta = geom["zeta"]
    eps_flavor = 1.0 / (8.0 * math.pi**5 * (1.0 - zeta))

    # PDG light-quark masses at mu = 2 GeV
    M_D_PDG_2022 = 0.00467
    M_S_PDG_2022 = 0.09340
    M_D_PDG_2024 = 0.00470
    M_S_PDG_2024 = 0.09350
    target = 0.22430

    # Variants with eps_flavor correction
    result_2024 = cabibbo_sector(
        M_d_EWT=M_d_EWT, M_s_EWT=M_s_EWT,
        M_d_PDG=M_D_PDG_2024, M_s_PDG=M_S_PDG_2024,
        eps_flavor=eps_flavor,
        target=target,
    )
    result_2022 = cabibbo_sector(
        M_d_EWT=M_d_EWT, M_s_EWT=M_s_EWT,
        M_d_PDG=M_D_PDG_2022, M_s_PDG=M_S_PDG_2022,
        eps_flavor=eps_flavor,
        target=target,
    )

    # Bare GST reference
    bare_2024 = math.sqrt(M_D_PDG_2024 / M_S_PDG_2024)
    bare_2022 = math.sqrt(M_D_PDG_2022 / M_S_PDG_2022)

    print("--- CABIBBO SECTOR (pi^5 flavor mixing) ---")
    print(f"  zeta (lattice impedance) = {zeta:.10e}")
    print(f"  eps_flavor = 1/(8 pi^5 (1-zeta)) = {eps_flavor:.10e}")
    print(f"  Correction factor        = {result_2024['correction']:.10f}")
    print(f"  Target                   = {target:.5f}  (PDG |V_us|)")
    
    print()
    print(f"  Variant A (EWT spherical masses, K=15 & K=28):")
    print(f"    M_d = {M_d_EWT:.6f} GeV, M_s = {M_s_EWT:.6f} GeV")
    print(f"    sin(theta_C) = {result_2024['sinC_A']:.6f}  "
          f"(err {result_2024['err_A_pct']:.3f}%)")
    print()
    print(f"  Variant B (PDG 2024 masses + eps_flavor):")
    print(f"    M_d = {M_D_PDG_2024:.5f} GeV, M_s = {M_S_PDG_2024:.4f} GeV")
    print(f"    sin(theta_C) = {result_2024['sinC_B']:.6f}  "
          f"(err {result_2024['err_B_pct']:.4f}%)")
    print()
    print(f"  Variant C (PDG 2022 masses + eps_flavor):")
    print(f"    M_d = {M_D_PDG_2022:.5f} GeV, M_s = {M_S_PDG_2022:.4f} GeV")
    print(f"    sin(theta_C) = {result_2022['sinC_B']:.6f}  "
          f"(err {result_2022['err_B_pct']:.3f}%)")
    print()
    print(f"  Bare GST reference (no correction):")
    print(f"    PDG 2024: sin(theta_C) = {bare_2024:.6f}  "
          f"(err {abs(bare_2024-target)/target*100:.3f}%)")
    print(f"    PDG 2022: sin(theta_C) = {bare_2022:.6f}  "
          f"(err {abs(bare_2022-target)/target*100:.3f}%)")
    print()

    return {
        "eps_flavor": result_2024["eps_flavor"],
        "correction": result_2024["correction"],
        "variant_A": {"sinC": result_2024["sinC_A"], "err": result_2024["err_A_pct"]},
        "variant_B_2024": {"sinC": result_2024["sinC_B"], "err": result_2024["err_B_pct"]},
        "variant_C_2022": {"sinC": result_2022["sinC_B"], "err": result_2022["err_B_pct"]},
        "bare_gst_2024": bare_2024,
        "bare_gst_2022": bare_2022,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("EWT ELECTROWEAK & BOSON MODULE")
    print()

    # --- Geometry (once) ---
    bcc = derive_eps_M_from_BCC(8.0 * PI**4)
    eps_M = bcc["eps_M"]
    zeta = lattice_impedance(8.0 * PI**4, BCC_PACKING_FRACTION)  # explicit
    ladder = build_geometric_ladder(eps_M, "BCC packing derived")
    C_local = ladder["C_local"]
    C_gap_pure = ladder["C_gap"]
    cg_mixed = c_gap_mixed(C_local, PI**6, PI**7)

    geom = {
        "eps_M":       eps_M,
        "zeta":        zeta,
        "C_local":     C_local,
        "C_gap_pure":  C_gap_pure,
        "C_gap_mixed": cg_mixed["C_gap"],
        "O_mixed":     cg_mixed["O_mixed"],
        "pi6":         PI**6,
        "pi7":         PI**7,
    }

    run_geometry_summary(geom)

    # --- Mixed W (default) and benchmark comparison ---
    mixed_result = run_mixed_w_sector(geom)
    run_benchmark_comparison(mixed_result["M_W_pred"])

    # --- Pure pi^6 W (for comparison / neutral vertices) ---
    run_pure_w_sector(geom)

    # --- Higgs mixing (ZH pure, WH mixed) ---
    higgs = run_higgs_sector(geom, mixed_result["M_W_pred"])

    # --- Weinberg consistency tests ---
    run_weinberg_tests(geom, mixed_result["M_W_pred"], higgs["M_Z_EWT"])

    # --- Heavy boson radii ---
    alpha_inv = compute_alpha_geometric(eps_M)
    alpha_geom = 1.0 / alpha_inv
    q_P = derive_planck_charge_from_e(alpha_geom, E_CHARGE_CODATA)
    r_nu = derive_neutrino_radius(alpha_geom, q_P)["r_nu"]
    r_e_geom = 100.0 * r_nu
    run_radii_sector(r_e_geom)

    # --- Cabibbo sector ---
    run_cabibbo_sector(geom)


if __name__ == "__main__":
    main()