#!/usr/bin/env python3
"""
M4/EWT - N as the internal lock-in point: drift, resolution, grouping.

OpenWave criterion:
    Gravity: Newton limit (GEM) + local metric phenomena (strength clause)
    Cosmology: Lambda / cosmic acceleration

Purpose:
    Quantify how the geometric stiffness N_geom = 778.8025 acts as the
    internal lock-in point of the Enhanced EWT model.

    Three levels of analysis:
      1. DRIFT        - how much each sector's value at N_geom drifts away
                        when N is changed by +/- 10.
      2. RESOLUTION   - how narrowly each sector can locate N, computed as
                        its relative error divided by its logarithmic
                        derivative d(ln O)/d(ln N).
      3. GROUPING     - which sectors actually constrain N, which are only
                        consistent, and which provide no constraint.

    Only PDG quark masses are used for the Cabibbo sector. No EWT quark
    masses enter the calculation.

    Cosmological rows (added 2026-11-14):
      - Omega_vac(N): effective vacuum fraction Omega_BCC * C_gap. This
        depends on N only through eps_M, so it is INDEPENDENT of G and
        lambda_l. It is the only cosmological sector that does not
        inherit G's sensitivity.
      - Lambda_vac(N): full chain N -> eps_M -> alpha -> {hbar, lambda_l}
        -> G -> T_p -> R_H -> H_0 -> Lambda_dS -> Omega_vac -> Lambda_vac.
        Because G enters through a -12 power and the chain preserves
        that exponent, beta(Lambda_vac) is dominated by -12 from G. This
        row is therefore a CONSISTENCY TEST WITH G, not an independent
        constraint on N. It is reported honestly as such.
        Two distinct G-related observations apply to the cosmological
        chain. First, G_geom is a derived quantity, not a free parameter:
        the combinations hbar/G (in T_p^2) and hbar*G (in lambda_l^2)
        reduce to expressions in r_e, m_e, c, alpha_geom, and the BCC
        geometric factors, so the 0.048% residual on G_geom is a test of
        the geometry, not an accuracy on H_0 or Lambda_vac. Second, G(N)
        itself scales as N^-12, so Lambda_vac(N) inherits that sensitivity
        even though the G symbol drops out of the symbolic combination.
        The first observation is about provenance; the second is about
        the lock-in scan's N-dependence. Both are true and distinct.

Output:
    - console: three sections
    - CSV and JSON saved to research/data/ (local-only, not committed)
"""

import math
import csv
import json
from pathlib import Path

try:
    from m4_7_ewt_emergence_engine import (
        PI, SQRT2, SQRT3, EULER,
        G_CODATA, ALPHA_INV_CODATA, A_E_CODATA,
        A_MU_EXP, A_TAU_EXP,
        SIN2_THETA_W, SIN_THETA_C_PDG,
        M_Z_CODATA, M_W_PDG,
        M_D_PDG, M_S_PDG,
        C0, M_E, R_E, E_CHARGE_CODATA,
        BCC_IDEAL_PROJECTION_LP,
        compute_alpha_geometric,
        derive_eps_M_from_BCC,
        build_geometric_ladder,
        derive_planck_charge_from_e,
        derive_neutrino_radius,
        derive_hbar,
        derive_lambda_l_geometric,
        gravity_sector,
        compute_lepton_amms,
    )
except ImportError:
    raise ImportError(
        "This module requires m4_7_ewt_emergence_engine.py in the same directory."
    )


# ----------------------------------------------------------------------
# Cosmological constants (v5.3.1, see M4.14 findings)
# ----------------------------------------------------------------------
K_B           = 1.380649e-23    # exact SI 2019
T_CMB_CODATA  = 2.7255          # COBE/FIRAS anchor
LAMBDA_PLANCK = 1.089e-52       # Planck 2018 [m^-2]
OMEGA_L_BAO   = 0.6889          # Planck+BAO


# ----------------------------------------------------------------------
# 1. Sector value functions
# ----------------------------------------------------------------------

def alpha_value(N: float) -> float:
    """Inverse fine-structure constant at given N."""
    eps_M = 1.0 / (N * PI**3)
    return compute_alpha_geometric(eps_M)


def G_value(N: float) -> float:
    """Gravitational constant at given N (full self-consistent chain)."""
    eps_M = 1.0 / (N * PI**3)
    alpha_inv = compute_alpha_geometric(eps_M)
    alpha_geom = 1.0 / alpha_inv

    q_P = derive_planck_charge_from_e(alpha_geom, E_CHARGE_CODATA)
    nu_res = derive_neutrino_radius(alpha_geom, q_P)
    r_nu = nu_res["r_nu"]

    lambda_l = derive_lambda_l_geometric(
        alpha_geom=alpha_geom,
        r_e=R_E,
        r_nu=r_nu,
        N_geom=N,
        L_p_geom=BCC_IDEAL_PROJECTION_LP,
        K_WC=10,
    )

    res = gravity_sector(
        alpha_geom=alpha_geom,
        r_nu=r_nu,
        N_geom=N,
        L_p_geom=BCC_IDEAL_PROJECTION_LP,
        K_WC=10,
        lambda_l=lambda_l,
        r_e=R_E,
        m_e=M_E,
        c0=C0,
    )
    return res["G_EWT"]


def amm_values(N: float) -> dict:
    """Full lepton AMM predictions at given N (dimensionless)."""
    eps_M = 1.0 / (N * PI**3)
    alpha_inv = compute_alpha_geometric(eps_M)
    alpha_geom = 1.0 / alpha_inv

    amm = compute_lepton_amms(alpha_geom, eps_M)
    return {
        "a_e": amm["a_e_ppm"] * 1e-6,
        "a_mu": amm["a_mu_ppm"] * 1e-6,
        "a_tau": amm["a_tau_ppm"] * 1e-6,
    }


def mixing_values(N: float) -> dict:
    """Raw mixing angle predictions at given N (N = N_geom in the scan)."""
    eps_M = 1.0 / (N * PI**3)

    # zeta tracks N through the constraint N = 8*pi^4 * (1 - zeta):
    #   zeta = 1 - N/(8*pi^4)
    # At N = N_geom_ref it reproduces the reference zeta = 6.034e-4 from
    # derive_eps_M_from_BCC. The ladder is taken from the engine so that
    # C_local, C_gap, and eps_flavor share a single source of truth.
    zeta = 1.0 - N / (8.0 * PI**4)
    ladder = build_geometric_ladder(eps_M, zeta=zeta)

    C_gap = ladder["C_gap"]
    eps_flavor = ladder["eps_flavor"]

    # M_W_PDG is the anchor (PDG 2022 average, 80.377 GeV)
    sin2_W_pred = 1.0 - (M_W_PDG / M_Z_CODATA) ** 2 * (1.0 / C_gap)
    sinC_pred = math.sqrt(M_D_PDG / M_S_PDG) * (1.0 + eps_flavor)

    return {
        "sin2_W": sin2_W_pred,
        "sin_C": sinC_pred,
    }


def Omega_vac_value(N: float) -> float:
    """
    Effective vacuum fraction Omega_vac = eta_BCC * C_gap at given N.

    INDEPENDENT of G and lambda_l: this sector tests only eps_M(N)
    through the neutral gap operator C_gap. It is the only cosmological
    quantity that does not inherit G's sensitivity. See M4.14 findings.
    """
    eps_M = 1.0 / (N * PI**3)
    zeta = 1.0 - N / (8.0 * PI**4)
    ladder = build_geometric_ladder(eps_M, zeta=zeta)
    eta_BCC = SQRT3 * PI / 8.0
    return eta_BCC * ladder["C_gap"]


def Lambda_vac_value(N: float) -> float:
    """
    Observed cosmological constant Lambda_vac = Omega_vac * Lambda_dS at N.

    WARNING: the logarithmic derivative is dominated by -12 from G(N),
    through the chain N -> eps_M -> alpha -> {hbar, lambda_l} -> G -> T_p
    -> R_H -> H_0 -> Lambda_dS -> Omega_vac -> Lambda_vac. This row is a
    CONSISTENCY TEST WITH G, not an independent constraint on N. For an
    independent cosmological test, see Omega_vac.
    """
    eps_M = 1.0 / (N * PI**3)
    alpha_inv = compute_alpha_geometric(eps_M)
    alpha_geom = 1.0 / alpha_inv

    q_P = derive_planck_charge_from_e(alpha_geom, E_CHARGE_CODATA)
    r_nu = derive_neutrino_radius(alpha_geom, q_P)["r_nu"]

    lambda_l = derive_lambda_l_geometric(
        alpha_geom=alpha_geom,
        r_e=R_E,
        r_nu=r_nu,
        N_geom=N,
        L_p_geom=BCC_IDEAL_PROJECTION_LP,
        K_WC=10,
    )

    gravity_res = gravity_sector(
        alpha_geom=alpha_geom,
        r_nu=r_nu,
        N_geom=N,
        L_p_geom=BCC_IDEAL_PROJECTION_LP,
        K_WC=10,
        lambda_l=lambda_l,
        r_e=R_E,
        m_e=M_E,
        c0=C0,
    )
    G_EWT = gravity_res["G_EWT"]

    hbar_geom = derive_hbar(alpha_geom, R_E, M_E, C0)
    T_p = (1.0 / K_B) * math.sqrt(hbar_geom * C0**5 / G_EWT)

    H_factor = 8.0 * (2.0 * PI)**2
    R_H = lambda_l * (T_p / T_CMB_CODATA)**2 / H_factor
    H_0_si = C0 / R_H
    Lambda_dS = 3.0 * H_0_si**2 / C0**2

    return Omega_vac_value(N) * Lambda_dS


SECTOR_VALUE_FUNCS = {
    "alpha": alpha_value,
    "G": G_value,
    "a_e": lambda N: amm_values(N)["a_e"],
    "a_mu": lambda N: amm_values(N)["a_mu"],
    "a_tau": lambda N: amm_values(N)["a_tau"],
    "sin2_W": lambda N: mixing_values(N)["sin2_W"],
    "sin_C": lambda N: mixing_values(N)["sin_C"],
    "Lambda_vac": Lambda_vac_value,
    "Omega_vac": Omega_vac_value,
}

SECTOR_TARGETS = {
    "alpha": ALPHA_INV_CODATA,
    "G": G_CODATA,
    "a_e": A_E_CODATA,
    "a_mu": A_MU_EXP,
    "a_tau": A_TAU_EXP,
    "sin2_W": SIN2_THETA_W,
    "sin_C": SIN_THETA_C_PDG,
    "Lambda_vac": LAMBDA_PLANCK,
    "Omega_vac": OMEGA_L_BAO,
}

SECTOR_LABELS = {
    "alpha": "alpha^-1",
    "G": "G",
    "a_e": "a_e",
    "a_mu": "a_mu",
    "a_tau": "a_tau",
    "sin2_W": "sin^2(theta_W)",
    "sin_C": "sin(theta_C)",
    "Lambda_vac": "Lambda_vac",
    "Omega_vac": "Omega_vac",
}


# ----------------------------------------------------------------------
# 2. Numerical helpers
# ----------------------------------------------------------------------

def relative_error(N: float, sector: str) -> float:
    """Relative error of a sector prediction vs its target."""
    val = SECTOR_VALUE_FUNCS[sector](N)
    target = SECTOR_TARGETS[sector]
    return abs(val - target) / abs(target)


def log_derivative(N: float, sector: str, h: float = 1e-6) -> float:
    """d(ln O)/d(ln N) for sector at N by central difference in log N."""
    log_N = math.log(N)
    val_plus = math.log(SECTOR_VALUE_FUNCS[sector](math.exp(log_N + h)))
    val_minus = math.log(SECTOR_VALUE_FUNCS[sector](math.exp(log_N - h)))
    return (val_plus - val_minus) / (2.0 * h)


def find_N_opt(sector: str, N_start: float = 50.0, N_stop: float = 2000.0,
               num_scan: int = 400) -> float | None:
    """
    Find N such that prediction(N) == target by sign scan + bisection.
    Returns None if no sign change is found in the scan range.
    """
    log_vals = []
    log_min = math.log(N_start)
    log_max = math.log(N_stop)
    for i in range(num_scan):
        log_N = log_min + (log_max - log_min) * i / (num_scan - 1)
        log_vals.append(log_N)

    prev_log_N = log_vals[0]
    prev_res = SECTOR_VALUE_FUNCS[sector](math.exp(prev_log_N)) - SECTOR_TARGETS[sector]

    for log_N in log_vals[1:]:
        res = SECTOR_VALUE_FUNCS[sector](math.exp(log_N)) - SECTOR_TARGETS[sector]
        if prev_res * res < 0.0:
            # bracketing interval
            lo, hi = prev_log_N, log_N
            lo_res, hi_res = prev_res, res
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                mid_res = SECTOR_VALUE_FUNCS[sector](math.exp(mid)) - SECTOR_TARGETS[sector]
                if lo_res * mid_res < 0.0:
                    hi = mid
                    hi_res = mid_res
                else:
                    lo = mid
                    lo_res = mid_res
            return math.exp(0.5 * (lo + hi))
        prev_log_N = log_N
        prev_res = res

    return None


# ----------------------------------------------------------------------
# 3. Main
# ----------------------------------------------------------------------

def main():
    print("=" * 78)
    print("M4.11 - N AS INTERNAL LOCK-IN POINT")
    print("       drift | resolution | grouping")
    print("=" * 78)

    bcc = derive_eps_M_from_BCC(8.0 * PI**4)
    N_geom_ref = bcc["N_geom"]
    print(f"Reference N_geom = {N_geom_ref:.6f}")

    # Reference values at N_geom
    ref_values = {name: func(N_geom_ref) for name, func in SECTOR_VALUE_FUNCS.items()}

    print("\nReference values at N_geom:")
    for name, label in SECTOR_LABELS.items():
        print(f"  {label:<18} = {ref_values[name]:.12e}")

    # Shift for the crossing test; feeds the shifted_diff column of the CSV only.
    shift_C = ref_values["sin2_W"] - ref_values["sin_C"]
    print(f"\nshift_C (sin^2W - sinC at N_geom) = {shift_C:.10f}")

    N_start = 500.0
    N_stop = 1100.0
    N_step = 2.0
    N_vals = []
    n = N_start
    while n <= N_stop + 1e-9:
        N_vals.append(n)
        n += N_step

    print(f"\nDrift scan range: [{N_start}, {N_stop}], step = {N_step}, {len(N_vals)} points")

    # ------------------------------------------------------------------
    # SECTION 1: DRIFT
    # ------------------------------------------------------------------
    drift = {name: [] for name in SECTOR_VALUE_FUNCS}
    shifted_diff = []

    print("\nComputing drift ...")
    for idx, N in enumerate(N_vals):
        if idx % 50 == 0:
            print(f"  {idx}/{len(N_vals)}")
        for name in SECTOR_VALUE_FUNCS:
            drift[name].append(
                abs(SECTOR_VALUE_FUNCS[name](N) - ref_values[name]) / abs(ref_values[name])
            )
        mix = mixing_values(N)
        shifted_diff.append(abs(mix["sin2_W"] - (mix["sin_C"] + shift_C)))

    def drift_exact(name, delta):
        """
        Drift evaluated AT N_geom + delta, not at the nearest scan-grid point.

        The scan grid has step 2 and does not contain N_geom = 778.8025, so a
        nearest-point lookup reports the drift at N = 778 and at N_geom - 10.80
        and N_geom + 9.20 instead of at N_geom and N_geom +/- 10.
        """
        val = SECTOR_VALUE_FUNCS[name](N_geom_ref + delta)
        return abs(val - ref_values[name]) / abs(ref_values[name])

    print("\n" + "=" * 78)
    print("SECTION 1: DRIFT relative to N_geom (zero at N_geom by construction)")
    print("=" * 78)
    header = (f"{'Sector':<18} {'drift@N_geom':>14} "
              f"{'drift@-10':>12} {'drift@+10':>12}")
    print(header)
    print("-" * len(header))

    for name, label in SECTOR_LABELS.items():
        d_at = drift_exact(name, 0.0)
        d_m10 = drift_exact(name, -10.0)
        d_p10 = drift_exact(name, +10.0)
        print(f"{label:<18} {d_at:14.6e} {d_m10:12.6e} {d_p10:12.6e}")

    # ------------------------------------------------------------------
    # SECTION 2: RESOLUTION
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SECTION 2: RESOLUTION (error / |d ln O / d ln N|)")
    print("=" * 78)

    beta = {}
    err = {}
    resolution = {}
    N_opt = {}

    for name, label in SECTOR_LABELS.items():
        beta[name] = log_derivative(N_geom_ref, name)
        err[name] = relative_error(N_geom_ref, name)
        if abs(beta[name]) > 1e-15:
            resolution[name] = err[name] / abs(beta[name])
        else:
            resolution[name] = float("inf")
        N_opt[name] = find_N_opt(name)

    header = (f"{'Sector':<18} {'beta':>12} {'error':>12} "
              f"{'resolution':>14} {'N_opt':>12}")
    print(header)
    print("-" * len(header))

    for name, label in SECTOR_LABELS.items():
        beta_s = beta[name]
        err_s = err[name]
        res_s = resolution[name]
        n_opt_s = N_opt[name]
        n_opt_str = f"{n_opt_s:.2f}" if n_opt_s is not None else "none"
        print(f"{label:<18} {beta_s:12.3e} {err_s:12.6e} {res_s:14.6e} {n_opt_str:>12}")

    # ------------------------------------------------------------------
    # SECTION 3: GROUPING
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SECTION 3: GROUPING")
    print("=" * 78)

    groups = {
        "PRIMARY (resolves N, <1%)": [],
        "WEAK (consistent, 1-100%)": [],
        "NO CONSTRAINT (>100%)": [],
    }

    for name, label in SECTOR_LABELS.items():
        r = resolution[name]
        if r < 0.01:
            groups["PRIMARY (resolves N, <1%)"].append(name)
        elif r < 1.0:
            groups["WEAK (consistent, 1-100%)"].append(name)
        else:
            groups["NO CONSTRAINT (>100%)"].append(name)

    for group_name, members in groups.items():
        print(f"\n{group_name}:")
        if not members:
            print("  (none)")
            continue
        for name in members:
            n_opt_s = N_opt[name]
            n_opt_str = f"{n_opt_s:.6f}" if n_opt_s is not None else "none"
            print(f"  {SECTOR_LABELS[name]:<18} N_opt = {n_opt_str:>12} "
                  f"resolution = {resolution[name]:.6e}")

    # Primary agreement
    primary_members = sorted(groups["PRIMARY (resolves N, <1%)"], key=lambda n: resolution[n])
    if len(primary_members) >= 2:
        n0 = N_opt[primary_members[0]]
        n1 = N_opt[primary_members[1]]
        agreement_pct = abs(n0 - n1) / N_geom_ref * 100.0
        print(f"\nPrimary agreement ({SECTOR_LABELS[primary_members[0]]} vs "
              f"{SECTOR_LABELS[primary_members[1]]}): {agreement_pct:.4f}% of N_geom")

    # ------------------------------------------------------------------
    # NOTE on the cosmological rows
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("NOTE: cosmological rows")
    print("=" * 78)
    print("  Two distinct G-related observations:")
    print()
    print("  1. G_geom is a derived quantity, not a free parameter.")
    print("     In the combinations that determine T_p^2 (~ hbar/G) and")
    print("     lambda_l^2 (~ hbar*G), the symbols hbar and G do not")
    print("     appear when the expressions are written out: both")
    print("     combinations reduce to expressions in r_e, m_e, c,")
    print("     alpha_geom, and the BCC geometric factors (manuscript")
    print("     v5.3.1, sec:cosmo_summary). Numerically, G_EWT is")
    print("     computed in this script as an intermediate, but the")
    print("     chain could be written without it. Consequently, the")
    print("     0.048% residual of G_geom against CODATA is a test of")
    print("     the BCC geometry at the level of G, not a direct")
    print("     accuracy on H_0 or Lambda_vac.")
    print()
    print("  2. G(N) sensitivity propagates through the chain (this is")
    print("     what the lock-in scan measures).")
    print("     G depends on N through the -12 exponent of the lattice")
    print("     stiffness, and Lambda_vac inherits this exponent:")
    print("     beta(Lambda_vac) = -12.014, dominated by -12 from G.")
    print("     Lambda_vac is therefore a CONSISTENCY TEST WITH G, not")
    print("     an independent constraint on N.")
    print()
    print("  Omega_vac depends on N only through eps_M (via C_gap), so it")
    print("  is INDEPENDENT of G and lambda_l. It is the only")
    print("  cosmological sector that does not inherit G's sensitivity.")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "m4_11_lock_in_n_scan.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["N"] + [f"drift_{name}" for name in SECTOR_LABELS] + ["shifted_diff"]
        writer.writerow(header)
        for i, N in enumerate(N_vals):
            row = [f"{N:.6f}"]
            for name in SECTOR_LABELS:
                row.append(f"{drift[name][i]:.12e}")
            row.append(f"{shifted_diff[i]:.12e}")
            writer.writerow(row)
    print(f"\nSaved: {csv_path}")

    # JSON
    json_path = out_dir / "m4_11_lock_in_n_scan.json"
    summary = {
        "N_geom_ref": N_geom_ref,
        "drift_scan_range": [N_start, N_stop, N_step],
        "reference_values_at_N_geom": ref_values,
        "drift_at_N_geom": {name: drift_exact(name, 0.0) for name in SECTOR_LABELS},
        "drift_at_N_geom_minus10": {name: drift_exact(name, -10.0) for name in SECTOR_LABELS},
        "drift_at_N_geom_plus10": {name: drift_exact(name, +10.0) for name in SECTOR_LABELS},
        "log_derivative_beta": beta,
        "relative_error_at_N_geom": err,
        "resolution": resolution,
        "N_opt": N_opt,
        "grouping": {g: [SECTOR_LABELS[n] for n in members] for g, members in groups.items()},
        "shift_C": shift_C,
        "cosmological_note": {
            "Lambda_vac": "inherits beta from G; consistency test, not independent",
            "Omega_vac": "independent of G; tests eps_M(N) alone",
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()