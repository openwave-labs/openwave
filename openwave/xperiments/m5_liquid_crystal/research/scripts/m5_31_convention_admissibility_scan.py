"""Admissibility-leg convention analysis for the M5.31 curvature scale curve.

This is an INDEPENDENT companion to
``m5_31_coupling_curvature_scan.py``.  It imports no M5.31 physics module: it
reads the already-tracked, already-audited data file
``data/m5_31_coupling_curvature_scan.json`` and rebuilds the two preregistered
inverse-coupling readings from scratch, cross-checking its reimplementation
against the shipped arrays.

Purpose.  M5.31 reports two conditional readings of the measured curvature
form factor ``C(rho)`` side by side and states (note section 1.3, Discussion
#438) that the ``C -> g_R`` dictionary "must be settled by the field/action
dictionary rather than by agreement with a target".  This script applies a
principled, non-target-fitting selection procedure to those two readings --
physical admissibility legs, the method an adjacent, independently-reviewed
public induced-gravity result uses to keep or drop a coupling scheme
(Substrate Framework, accepted claims C-IGR-004 and C-GRV-002, v0.162.0,
Apache-2.0, https://github.com/vantasnerdan/substrate-framework).  The
mathematics below is rederived here so the contribution stands on its own; the
public claims are cited as provenance, not as load-bearing evidence.

Two leg groups.

  Group A -- admissibility of a running curve (positivity, strict monotone
  running in log mu, finite non-zero plateau).  A reading that fails any of
  these is not an admissible running curve.

  Group B -- is the object already a renormalized coupling?  A one-loop
  coupling has a scale-constant log-slope (d(1/g^2)/d log mu ~ const = -b0).
  A classical form factor does not.  This is the leg that COULD separate a
  genuine coupling from a form factor.

Result reported, not asserted.  If a leg admits exactly one reading it selects
it; if the legs treat both readings identically the discrimination is a
declared null and the honest residual is quantified as the exact convention
spread ``R_conv(rho) = (amplitude reading)/(energy reading) = C_ref/C`` -- the
same "report the spread, do not pick" discipline the cited result uses for its
own scheme residual.  The dictionary itself stays author-gated on the M5
action.

Run from the repository root:

    python3 openwave/xperiments/m5_liquid_crystal/research/scripts/\
        m5_31_convention_admissibility_scan.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
IN_JSON = RESEARCH / "data" / "m5_31_coupling_curvature_scan.json"
OUT_JSON = RESEARCH / "data" / "m5_31_convention_admissibility_scan.json"
OUT_PLOT = RESEARCH / "plots" / "m5_31_convention_admissibility.png"

# Group B tolerance: the coefficient of variation of the interior log-slope
# below which a curve is called scale-constant (one-loop-like).  A genuine
# one-loop coupling sits far below this; the M5.31 form factor sits far above.
ONE_LOOP_CV_MAX = 0.05
# Group A plateau tolerance on the far inverse-coupling and its slope.
PLATEAU_SLOPE_MAX = 0.10


def inverse_coupling(c_values: np.ndarray, power: float) -> np.ndarray:
    """Rebuilt from scratch: energy power=1, amplitude power=2, ref=far shell."""
    c_values = np.asarray(c_values, dtype=float)
    c_ref = float(c_values[-1])
    return (c_ref / c_values) ** power


def local_cubic_slope(log_mu: np.ndarray, y: np.ndarray, width: int = 5) -> np.ndarray:
    """Independent overlapping local-cubic derivative dy/d(log mu)."""
    log_mu = np.asarray(log_mu, dtype=float)
    y = np.asarray(y, dtype=float)
    half = width // 2
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, min(i - half, len(y) - width))
        hi = lo + width
        coeff = np.polynomial.polynomial.polyfit(log_mu[lo:hi] - log_mu[i], y[lo:hi], 3)
        out[i] = coeff[1]
    return out


def leg_positivity(inv_g2: np.ndarray) -> bool:
    return bool(np.all(inv_g2 > 0.0))


def leg_strict_monotone(rho: np.ndarray, inv_g2: np.ndarray) -> bool:
    """1/g^2 strictly decreasing in rho (strictly increasing in mu=1/rho)."""
    order = np.argsort(rho)
    return bool(np.all(np.diff(inv_g2[order]) < 0.0))


def leg_finite_plateau(inv_g2: np.ndarray, slope: np.ndarray) -> bool:
    """Far inverse-coupling is finite, positive, and its running has died off."""
    far_ok = np.isfinite(inv_g2[-1]) and inv_g2[-1] > 0.0
    slope_ok = abs(float(slope[-1])) < PLATEAU_SLOPE_MAX
    return bool(far_ok and slope_ok)


def leg_one_loop_constant(slope: np.ndarray, interior: slice) -> bool:
    """True only if the interior log-slope is scale-constant (a coupling)."""
    s = slope[interior]
    mean = float(np.mean(s))
    if mean == 0.0:
        return False
    cv = float(np.std(s) / abs(mean))
    return cv < ONE_LOOP_CV_MAX


def analyse(rho: np.ndarray, c_values: np.ndarray, shipped: dict | None) -> dict:
    rho = np.asarray(rho, dtype=float)
    log_mu = -np.log(rho)
    interior = slice(2, -2)

    readings = {}
    for name, power in (("energy", 1.0), ("amplitude", 2.0)):
        inv_g2 = inverse_coupling(c_values, power)
        slope = local_cubic_slope(log_mu, inv_g2)
        legs = {
            "A_positivity": leg_positivity(inv_g2),
            "A_strict_monotone_running": leg_strict_monotone(rho, inv_g2),
            "A_finite_nonzero_plateau": leg_finite_plateau(inv_g2, slope),
            "B_one_loop_scale_constant": leg_one_loop_constant(slope, interior),
        }
        readings[name] = {
            "power": power,
            "inverse_g2": inv_g2.tolist(),
            "slope_dlogmu": slope.tolist(),
            "interior_slope_cv": float(
                np.std(slope[interior]) / abs(np.mean(slope[interior]))
            ),
            "legs": legs,
        }

    # Independence check: rebuilt readings must match the shipped arrays.
    reimpl_max_abs_diff = None
    if shipped is not None:
        diffs = []
        for name in ("energy", "amplitude"):
            shipped_inv = np.asarray(shipped[name]["inverse_g2"], dtype=float)
            diffs.append(
                float(np.max(np.abs(shipped_inv - np.asarray(readings[name]["inverse_g2"]))))
            )
        reimpl_max_abs_diff = max(diffs)

    # The exact convention residual: amplitude/energy = C_ref/C.
    inv_e = np.asarray(readings["energy"]["inverse_g2"])
    inv_a = np.asarray(readings["amplitude"]["inverse_g2"])
    r_conv = inv_a / inv_e
    r_conv_closed = float(c_values[-1]) / np.asarray(c_values, dtype=float)

    # Does any leg separate the two readings?  (Reported, not assumed.)
    discriminating_legs = [
        leg
        for leg in readings["energy"]["legs"]
        if readings["energy"]["legs"][leg] != readings["amplitude"]["legs"][leg]
    ]

    both_admissible = all(
        readings[n]["legs"][leg]
        for n in ("energy", "amplitude")
        for leg in ("A_positivity", "A_strict_monotone_running", "A_finite_nonzero_plateau")
    )
    neither_is_coupling = not any(
        readings[n]["legs"]["B_one_loop_scale_constant"] for n in ("energy", "amplitude")
    )

    # Mutation: dent one interior C downward -> monotone running must break.
    mutated = np.asarray(c_values, dtype=float).copy()
    k = len(mutated) // 2
    mutated[k] = mutated[k] * 0.5
    mutated_inv_e = inverse_coupling(mutated, 1.0)
    mutation_breaks_monotone = not leg_strict_monotone(rho, mutated_inv_e)

    gates = {
        "reimplementation_matches_shipped_lt_1e-9": (
            reimpl_max_abs_diff is None or reimpl_max_abs_diff < 1e-9
        ),
        "both_readings_admissible_running_curves": both_admissible,
        "discrimination_is_null_no_leg_separates": len(discriminating_legs) == 0,
        "neither_reading_is_a_one_loop_coupling": neither_is_coupling,
        "residual_equals_C_ref_over_C_lt_1e-12": float(
            np.max(np.abs(r_conv - r_conv_closed))
        )
        < 1e-12,
        "mutation_breaks_monotone_leg": mutation_breaks_monotone,
    }

    return {
        "rho": rho.tolist(),
        "log_mu_over_mu0": log_mu.tolist(),
        "C_used": np.asarray(c_values, dtype=float).tolist(),
        "C_ref": float(c_values[-1]),
        "readings": readings,
        "reimplementation_max_abs_diff_vs_shipped": reimpl_max_abs_diff,
        "convention_residual_R_conv": r_conv.tolist(),
        "R_conv_range": [float(np.min(r_conv)), float(np.max(r_conv))],
        "discriminating_legs": discriminating_legs,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def plot_result(rho, readings, r_conv) -> None:
    rho = np.asarray(rho)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for name in ("energy", "amplitude"):
        axes[0].plot(rho, readings[name]["inverse_g2"], "o-", ms=3, label=f"{name}: 1/g_R^2")
    axes[0].set(xscale="log", xlabel="rho = r/r0", ylabel="1/g_R^2 (conditional)")
    axes[0].legend(frameon=False)
    axes[1].plot(rho, r_conv, "o-", ms=3, color="crimson")
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[1].set(
        xscale="log",
        xlabel="rho = r/r0",
        ylabel="R_conv = amplitude / energy = C_ref / C",
        title="convention residual (collapses to 1 at the Coulomb plateau)",
    )
    fig.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=180)
    plt.close(fig)


def main() -> int:
    source = json.loads(IN_JSON.read_text())
    rho = np.asarray(source["rho"], dtype=float)
    finest = source["spatial_refinement"][-1]
    c_measured = np.asarray(finest["C"], dtype=float)
    shipped = source.get("measured_interpretations", {}).get("schemes")

    result = analyse(rho, c_measured, shipped)
    result["source"] = {
        "data_file": "data/m5_31_coupling_curvature_scan.json",
        "C_series": f"spatial_refinement finest (n={finest['n']}, h/r0={finest['h_over_r0']:.4f})",
        "provenance_method": (
            "admissibility-leg selection rederived from Substrate Framework "
            "accepted claims C-IGR-004, C-GRV-002 (v0.162.0, Apache-2.0)"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    plot_result(rho, result["readings"], result["convention_residual_R_conv"])

    print(json.dumps({"gates": result["gates"], "all_gates_pass": result["all_gates_pass"]}, indent=2))
    print(f"R_conv range: {result['R_conv_range']}")
    print(f"reimpl vs shipped max|diff|: {result['reimplementation_max_abs_diff_vs_shipped']}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_PLOT}")
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
