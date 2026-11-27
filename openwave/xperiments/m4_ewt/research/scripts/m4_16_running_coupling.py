"""
M4.16: Boundary Onset and Scale Emergence of Gravitational Coupling
From manuscript version: not in manuscript yet

Models the boundary emergence of G and alpha_G across x = r / r_e = 1.
For x < 1 (interior push-out zone), outward EMC push-out from K_WC wave
centers compounded by restoring medium equilibrium dominates -- analogous to
asking how a uniform statutory background N_stat self-gravitates -- so no
local interior G(r) is defined (NaN).

Epistemic status in this script:
- Engine-derived invariants (G_EWT, alpha_G_EWT, N_nu_eff, N_nu_statutory,
  X_eff) come from m4_7_ewt_emergence_engine.py.
- All radial onset profiles Phi_onset(x) for x >= 1 are working hypotheses /
  ansatzes for boundary emergence and are NOT derived from BCC lattice
  mechanics or a first-principles interior field equation.
"""

from pathlib import Path
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from m4_7_ewt_emergence_engine import (
        PI,
        G_CODATA,
        ALPHA_G_CODATA,
        C0,
        M_E,
        R_E,
        E_CHARGE_CODATA,
        BCC_IDEAL_PROJECTION_LP,
        compute_alpha_core,
        compute_alpha_geometric,
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


# ---------------------------------------------------------------------------
# 1. Reference invariants computed from m4_7_ewt_emergence_engine
# ---------------------------------------------------------------------------
K_WC = 10

_A_PI = compute_alpha_core()
_BCC = derive_eps_M_from_BCC(8.0 * PI**4)
_EPS_M = _BCC["eps_M"]
_N_GEOM = _BCC["N_geom"]
_ALPHA_INV = compute_alpha_geometric(_EPS_M)
_ALPHA_GEOM = 1.0 / _ALPHA_INV
_Q_PLANCK = derive_planck_charge_from_e(_ALPHA_GEOM, E_CHARGE_CODATA)
_R_NU = derive_neutrino_radius(_ALPHA_GEOM, _Q_PLANCK)["r_nu"]
_LAMBDA_L = derive_lambda_l_geometric(
    alpha_geom=_ALPHA_GEOM,
    r_e=R_E,
    r_nu=_R_NU,
    N_geom=_N_GEOM,
    L_p_geom=BCC_IDEAL_PROJECTION_LP,
    K_WC=K_WC,
)
_GRAV = gravity_sector(
    alpha_geom=_ALPHA_GEOM,
    r_nu=_R_NU,
    N_geom=_N_GEOM,
    L_p_geom=BCC_IDEAL_PROJECTION_LP,
    K_WC=K_WC,
    lambda_l=_LAMBDA_L,
    r_e=R_E,
    m_e=M_E,
    c0=C0,
)

G_EWT = _GRAV["G_EWT"]
N_NU_EFF = _GRAV["N_nu_eff"]
N_NU_STAT = _GRAV["N_nu_statutory"]
X_EFF = _GRAV["X_eff"]
ALPHA_G_EWT = _ALPHA_GEOM / (
    (_A_PI**4) * (_N_GEOM**3) * K_WC * np.sqrt(N_NU_EFF)
)


# ---------------------------------------------------------------------------
# 2. Boundary onset ansatzes Phi_onset(x) for x = r / r_e
#    WORKING HYPOTHESES / ANSATZES (NOT derived from BCC lattice dynamics):
#    - For x < 1: NaN (push-out zone; no local interior G(r) defined).
#    - For x >= 1: phenomenological / working onset profiles that recover
#      Phi_onset -> 1 asymptotically (G -> G_EWT, alpha_G -> ALPHA_G_EWT).
# ---------------------------------------------------------------------------
def phi_step_boundary(x: np.ndarray) -> np.ndarray:
    """Working ansatz: ideal sharp boundary onset at x = 1 (not derived from BCC)."""
    out = np.full_like(x, np.nan, dtype=float)
    out[x >= 1.0] = 1.0
    return out


def phi_emc_wall_step(x: np.ndarray, wall_jump: float = 1.0) -> np.ndarray:
    """
    Working hypothesis: step/abrupt Degraded EMC Wall onset at x >= 1.
    Treats the boundary wall as producing an immediate jump at x = 1 rather
    than a gradual radial ramp; NOT derived from BCC lattice equations.
    """
    out = np.full_like(x, np.nan, dtype=float)
    out[x >= 1.0] = float(wall_jump)
    return out


def phi_monotonic_shell(x: np.ndarray) -> np.ndarray:
    """
    Working ansatz (Scenario A closure shape): monotonic onset across x >= 1
    using 1 - exp(-K_WC * (x - 1)). The exponential form and K_WC decay scale
    are a modeling ansatz, NOT a derivation from BCC lattice mechanics.
    """
    out = np.full_like(x, np.nan, dtype=float)
    mask = x >= 1.0
    dx = x[mask] - 1.0
    out[mask] = 1.0 - np.exp(-K_WC * dx)
    return out


def phi_emc_wall_peak(x: np.ndarray, wall_excess: float = 0.12) -> np.ndarray:
    """
    Working ansatz (Scenario B shell profile): adds a localized bump
    wall_excess * (K_WC * dx) * exp(-K_WC * dx) on top of the monotonic onset.
    Both the functional form and wall_excess are working assumptions, NOT
    derived from BCC lattice dynamics.
    """
    out = np.full_like(x, np.nan, dtype=float)
    mask = x >= 1.0
    dx = x[mask] - 1.0
    base_closure = 1.0 - np.exp(-K_WC * dx)
    shell_bump = wall_excess * (K_WC * dx) * np.exp(-K_WC * dx)
    out[mask] = base_closure + shell_bump
    return out


VARIANTS = {
    "step_boundary": phi_step_boundary,
    "emc_wall_step": phi_emc_wall_step,
    "monotonic_shell": phi_monotonic_shell,
    "emc_wall_peak": phi_emc_wall_peak,
}


def compute_profiles(x: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate Phi_onset(x), alpha_G(x), and G(x) across radial grid x = r / r_e."""
    results: dict[str, dict[str, np.ndarray]] = {}
    for name, fn in VARIANTS.items():
        phi = fn(x)
        results[name] = {
            "phi": phi,
            "alpha_g": ALPHA_G_EWT * phi,
            "g": G_EWT * phi,
        }
    return results


def print_summary() -> None:
    """Print engine invariants and representative values at key radii."""
    rel_err_g = abs(G_EWT - G_CODATA) / G_CODATA * 100.0
    rel_err_ag = abs(ALPHA_G_EWT - ALPHA_G_CODATA) / ALPHA_G_CODATA * 100.0

    print("=== M4.16 Boundary Onset & Scale Emergence Summary ===")
    print("Note: Phi_onset(x) variants for x >= 1 are working ansatzes, NOT BCC derivations.")
    print(f"X_eff           = {X_EFF:.10f}")
    print(f"N_nu_stat       = {N_NU_STAT:.10e}")
    print(f"N_nu_eff        = {N_NU_EFF:.10e}")
    print(
        f"G_EWT (x >= 1)  = {G_EWT:.10e} m^3 kg^-1 s^-2 "
        f"(rel. diff vs CODATA: {rel_err_g:.6f}%)"
    )
    print(
        f"alpha_G_EWT     = {ALPHA_G_EWT:.10e} "
        f"(rel. diff vs CODATA: {rel_err_ag:.6f}%)"
    )
    print("Interior x < 1  : push-out dominated (no local interior G(r) defined)\n")

    sample_x = np.array([0.5, 1.0, 1.1, 1.5, 10.0, 100.0])
    profiles = compute_profiles(sample_x)

    header = f"{'Variant':<18} | " + " | ".join([f"x={val:<5g}" for val in sample_x])
    print(header)
    print("-" * len(header))
    for name, data in profiles.items():
        row_vals = [
            f"{'push-out':<9}" if np.isnan(g_val) else f"{g_val:.3e}"
            for g_val in data["g"]
        ]
        print(f"{name:<18} | " + " | ".join(row_vals))


def generate_plot(output_path: Path) -> None:
    """Generate two-panel plot of boundary onset and far-field convergence."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.logspace(-1, 2, 1200)
    profiles = compute_profiles(x)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True)

    for ax in (ax1, ax2):
        ax.axvspan(
            x.min(),
            1.0,
            color="#f2e6e6",
            alpha=0.65,
            label="Interior push-out zone (r < r_e)",
        )
        ax.axvline(
            1.0,
            color="#333333",
            linestyle="--",
            linewidth=1.1,
            label="Boundary onset (r = r_e)",
        )

    styles = {
        "step_boundary": ("-", "#1f77b4", "Step boundary onset (ansatz, not BCC)"),
        "emc_wall_step": (":", "#9467bd", "Degraded EMC Wall jump (hypothesis, not BCC)"),
        "monotonic_shell": ("--", "#2ca02c", "Monotonic exp(-K_WC*(x-1)) (ansatz, not BCC)"),
        "emc_wall_peak": ("-.", "#d62728", "EMC Wall bump profile (ansatz, not BCC)"),
    }

    for name, (ls, color, label) in styles.items():
        ax1.plot(
            x,
            profiles[name]["g"],
            linestyle=ls,
            color=color,
            linewidth=1.8,
            label=label,
        )
        ax2.plot(
            x,
            profiles[name]["alpha_g"],
            linestyle=ls,
            color=color,
            linewidth=1.8,
            label=label,
        )

    ax1.axhline(
        G_CODATA,
        color="black",
        linestyle="-.",
        linewidth=1.0,
        alpha=0.75,
        label="G_CODATA",
    )
    ax1.set_xscale("log")
    ax1.set_ylabel(r"Emergent $G(r)$ [$\mathrm{m^3\,kg^{-1}\,s^{-2}}$]")
    ax1.set_title("M4.16: Boundary Emergence of Gravitational Coupling at $r = r_e$")
    ax1.set_ylim(0.0, G_EWT * 1.18)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="lower right", fontsize=8.0)

    ax2.axhline(
        ALPHA_G_CODATA,
        color="black",
        linestyle="-.",
        linewidth=1.0,
        alpha=0.75,
        label=r"$\alpha_G^{\mathrm{CODATA}}$",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Normalized radius $x = r / r_e$")
    ax2.set_ylabel(r"Dimensionless $\alpha_G(r)$")
    ax2.set_ylim(0.0, ALPHA_G_EWT * 1.18)
    ax2.grid(True, which="both", alpha=0.25)

    plt.tight_layout()
    with open(output_path, "wb") as fh:
        fig.savefig(fh, format="png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    print_summary()
    plot_file = (
        Path(__file__).resolve().parent.parent / "plots" / "m4_16_running_coupling.png"
    )
    generate_plot(plot_file)
