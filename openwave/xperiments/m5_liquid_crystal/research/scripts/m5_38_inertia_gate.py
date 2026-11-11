"""M5.38 R1: the isospin-inertia gate on the stored M5.22 census states.

The read (tasks/m5_38_task_details.md § 2), with M(x) the stored 3x3 order tensor and T_a the three
rotation generators acting by commutator at fixed position:

    iota_ab(x) = tr([T_a, M(x)]^T [T_b, M(x)]),      (T_a)_bc = -eps_abc
    I_ab(R)    = h^3 sum_{|x| < R, free cells} iota_ab(x)
    sum_a iota_aa(x) = 2 sum_{i<j} (lam_i(x) - lam_j(x))^2          (lam = the eigenvalues of M(x))
    vacuum (1, delta, 0): sum_a iota_aa = 2 [(1 - delta)^2 + 1 + delta^2]
    Theta_ab(shell) = mean of iota_ab over r1 <= |x| < r2, off the column (cylindrical radius > 6)

Labels (pre-registered): an eigen-direction of Theta is UNBROKEN if its eigenvalue is below 1e-3 of the
scale AND the integral projected on it moves by less than 1 percent from R = 14 to R = 22; otherwise
BROKEN. The state's word is the count of unbroken directions (3 = FLAT, 0 = EXTENSIVE). The word is
withheld if the summed shell density departs from the closed form by more than 2 percent.

Controls run first; no state is labeled unless all four pass.

Run from this directory: python3 m5_38_inertia_gate.py [--write]   ->  ../data/m5_38_inertia_gate.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

EPS = np.zeros((3, 3, 3))
for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    EPS[a, b, c] = 1.0
    EPS[a, c, b] = -1.0
T = -EPS                                   # (T_a)_bc = -eps_abc

RADII = np.arange(4.0, 22.0 + 1e-9, 2.0)   # centered balls
WINDOWS = ((18.0, 22.0), (16.0, 20.0))     # the outer shell and its second window
RHO_COLUMN = 6.0                           # shell statistics are taken at cylindrical radius above this
R_IN, R_OUT = 14.0, 22.0                   # the growth read
UNBROKEN_EIG, UNBROKEN_GROWTH, CLOSED_FORM_TOL = 1e-3, 0.01, 0.02


def iota(M):
    """iota_ab(x) = tr([T_a, M]^T [T_b, M]) on a field M[..., 3, 3]."""
    C = np.einsum("aij,...jk->...aik", T, M) - np.einsum("...ij,ajk->...aik", M, T)
    return np.einsum("...aij,...bij->...ab", C, C)


def vac_sum(delta):
    return 2.0 * ((1.0 - delta) ** 2 + 1.0 + delta ** 2)


def grid(n, h):
    x = (np.arange(n) - (n - 1) / 2.0) * h
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return X, Y, Z


def pin_shell(n, h, depth=1.6):
    """the census instrument's pinned shell (m5_21_2b_a_instrument.pin_shell), reproduced."""
    wc = max(1, int(np.ceil(depth / h)))
    P = np.zeros((n, n, n), dtype=bool)
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = slice(0, wc); P[tuple(sl)] = True
        sl[ax] = slice(n - wc, n); P[tuple(sl)] = True
    return P


def read_state(M, h, delta, free=None):
    """the full read on one field; returns a JSON-ready dict."""
    n = M.shape[0]
    X, Y, Z = grid(n, h)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    rho = np.sqrt(X ** 2 + Y ** 2)
    if free is None:
        free = ~pin_shell(n, h)
    io = iota(M.astype(np.float64))
    scale = vac_sum(delta)
    # the integrals
    I = []
    cells = []
    for R in RADII:
        m = free & (r < R)
        I.append(io[m].sum(axis=0) * h ** 3)
        cells.append(int(m.sum()))
    I = np.array(I)
    i_in, i_out = int(np.argmin(abs(RADII - R_IN))), int(np.argmin(abs(RADII - R_OUT)))
    out = {"n": int(n), "h": float(h), "delta": float(delta), "radii": RADII.tolist(), "cells_in_ball": cells,
           "I_trace": np.trace(I, axis1=1, axis2=2).tolist(), "I_diag": np.diagonal(I, axis1=1, axis2=2).tolist(),
           "cell_count_growth_14_to_22": cells[i_out] / cells[i_in], "pure_volume_growth_14_to_22": (R_OUT / R_IN) ** 3,
           "closed_form_vacuum_sum": scale, "windows": {}}
    lam = np.linalg.eigvalsh(M.astype(np.float64))
    # the radial profile of the summed density over the closed form (added at the run: where the exterior departs)
    s = np.trace(io, axis1=-2, axis2=-1) / scale
    prof = []
    for r1 in np.arange(0.0, 24.0, 2.0):
        sh = free & (r >= r1) & (r < r1 + 2.0)
        if not sh.any():
            continue
        off, on = sh & (rho > RHO_COLUMN), sh & (rho <= RHO_COLUMN)
        prof.append({"r1": float(r1), "r2": float(r1 + 2.0), "all": float(s[sh].mean()), "min": float(s[sh].min()),
                     "off_column": float(s[off].mean()) if off.any() else None, "on_column": float(s[on].mean()) if on.any() else None})
    out["profile_summed_density_over_closed_form"] = prof
    out["min_summed_density_over_closed_form_free_cells"] = float(s[free].min())
    out["pinned_shell_summed_density_over_closed_form"] = float(s[~free].mean())
    out["I_trace_22_over_vacuum_volume"] = float(np.trace(I[i_out]) / (scale * cells[i_out] * h ** 3))
    # the alternative reading raised by the audit: the inertia of the SPECTRAL deviation N = M - sum_k lam_vac_k v_k v_k^T
    # (the eigenvectors of M itself, the vacuum eigenvalues in the same order); its summed density is the eigenvalue form on mu = lam - lam_vac
    mu = lam - np.array([0.0, delta, 1.0])
    dev = 2.0 * ((mu[..., 0] - mu[..., 1]) ** 2 + (mu[..., 0] - mu[..., 2]) ** 2 + (mu[..., 1] - mu[..., 2]) ** 2)
    Idev = [float(dev[free & (r < R)].sum() * h ** 3) for R in RADII]
    g1422 = (Idev[i_out] / Idev[i_in] - 1.0) if Idev[i_in] > 0 else None
    out["spectral_deviation_reading"] = {"I_trace": Idev, "growth_14_to_22": g1422,
                                         "last_step_20_to_22": (Idev[-1] / Idev[-2] - 1.0) if Idev[-2] > 0 else None,
                                         "flat_within_1_percent": bool(g1422 is not None and abs(g1422) < UNBROKEN_GROWTH),
                                         "pinned_shell_mean_density": float(dev[~free].mean())}
    words = []
    for (r1, r2) in WINDOWS:
        sh = free & (r >= r1) & (r < r2)
        off, on = sh & (rho > RHO_COLUMN), sh & (rho <= RHO_COLUMN)
        Th = io[off].mean(axis=0)
        w, V = np.linalg.eigh(Th)
        bar = UNBROKEN_EIG * max(float(np.trace(Th)), scale)
        dirs = []
        for k in range(3):
            v = V[:, k]
            p_in, p_out = float(v @ I[i_in] @ v), float(v @ I[i_out] @ v)
            tot = float(np.trace(I[i_out]))
            moved = abs(p_out - p_in) / abs(p_in) if abs(p_in) > 1e-12 * max(tot, 1.0) else (0.0 if abs(p_out - p_in) <= 1e-12 * max(tot, 1.0) else float("inf"))
            unbroken = bool(w[k] <= bar and moved < UNBROKEN_GROWTH)
            dirs.append({"eigenvalue": float(w[k]), "eigenvalue_over_trace": float(w[k] / np.trace(Th)) if np.trace(Th) > 0 else 0.0,
                         "direction": v.tolist(), "I_proj_14": p_in, "I_proj_22": p_out,
                         "growth_14_to_22": (p_out / p_in) if abs(p_in) > 0 else None, "label": "UNBROKEN" if unbroken else "BROKEN"})
        count = sum(d["label"] == "UNBROKEN" for d in dirs)
        dens = float(np.trace(Th))
        ok = abs(dens / scale - 1.0) <= CLOSED_FORM_TOL
        word = {3: "FLAT", 0: "EXTENSIVE"}.get(count, f"{count} UNBROKEN")
        words.append(word if ok else "WITHHELD")
        out["windows"][f"{r1:g}_{r2:g}"] = {
            "cells_off_column": int(off.sum()), "cells_on_column": int(on.sum()),
            "summed_density_off_column": dens, "summed_density_over_closed_form": dens / scale,
            "summed_density_on_column": float(np.trace(io[on].mean(axis=0))) if on.any() else None,
            "per_generator_density_off_column": np.diag(Th).tolist(),
            "mean_eigenvalues_of_M_off_column": lam[off].mean(axis=0).tolist(),
            "theta_eigen": dirs, "unbroken_count": count, "closed_form_within_2_percent": bool(ok), "word": words[-1]}
    out["word"] = words[0] if words[0] == words[1] else f"WINDOWS DISAGREE ({words[0]} / {words[1]})"
    return out


def controls():
    n, h, delta = 32, 1.5, 0.3
    X, Y, Z = grid(n, h)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    res = {}
    # (a) a uniform vacuum tensor: the summed density is the closed form
    Mv = np.broadcast_to(np.diag([1.0, delta, 0.0]), (n, n, n, 3, 3)).copy()
    a = read_state(Mv, h, delta)
    dev = abs(a["windows"]["18_22"]["summed_density_over_closed_form"] - 1.0)
    res["a_uniform_vacuum"] = {"summed_density_over_closed_form_minus_1": dev, "word": a["word"],
                               "pass": bool(dev < 1e-12 and a["word"] == "EXTENSIVE"),
                               "fails_if": "the summed density differs from 2[(1-d)^2 + 1 + d^2] beyond round-off, or the uniform biaxial vacuum is not read EXTENSIVE"}
    # (b) a random symmetric field: the summed density equals the eigenvalue form
    rng = np.random.default_rng(38)
    A = rng.normal(size=(6, 6, 6, 3, 3)); Ms = A + np.swapaxes(A, -1, -2)
    lam = np.linalg.eigvalsh(Ms)
    eig_form = 2.0 * ((lam[..., 0] - lam[..., 1]) ** 2 + (lam[..., 0] - lam[..., 2]) ** 2 + (lam[..., 1] - lam[..., 2]) ** 2)
    devb = float(np.max(np.abs(np.trace(iota(Ms), axis1=-2, axis2=-1) - eig_form) / eig_form))
    res["b_eigenvalue_identity"] = {"max_relative_deviation": devb, "pass": bool(devb < 1e-12),
                                    "fails_if": "sum_a iota_aa differs from 2 sum_{i<j} (lam_i - lam_j)^2 on a random symmetric field"}
    # (c) a planted compact record: the vacuum tensor blended to a multiple of the identity outside r = 8
    wgt = np.clip((8.0 - r) / 2.0, 0.0, 1.0); wgt = 0.5 - 0.5 * np.cos(np.pi * wgt)
    iso = np.broadcast_to(np.eye(3) * np.trace(np.diag([1.0, delta, 0.0])) / 3.0, Mv.shape)
    Mc = wgt[..., None, None] * Mv + (1.0 - wgt[..., None, None]) * iso
    c = read_state(Mc, h, delta)
    res["c_planted_compact_record"] = {"word": c["word"], "I_trace_at_10": c["I_trace"][3], "I_trace_at_22": c["I_trace"][-1],
                                       "unbroken_count": c["windows"]["18_22"]["unbroken_count"],
                                       "pass": bool(c["windows"]["18_22"]["unbroken_count"] == 3 and abs(c["I_trace"][-1] - c["I_trace"][3]) < 1e-9),
                                       "fails_if": "the reader does not return three unbroken directions and a flat integral on a record that vanishes outside r = 8",
                                       "note": "the word prints WITHHELD by the closed-form clause (the exterior is not the vacuum), as it must; the count is the control"}
    # (d) a uniform uniaxial tensor: exactly one direction unbroken
    Mu = np.broadcast_to(np.diag([1.0, 0.0, 0.0]), (n, n, n, 3, 3)).copy()
    d = read_state(Mu, h, 0.0)
    res["d_uniform_uniaxial"] = {"unbroken_count": d["windows"]["18_22"]["unbroken_count"],
                                 "eigenvalues": [x["eigenvalue"] for x in d["windows"]["18_22"]["theta_eigen"]],
                                 "pass": bool(d["windows"]["18_22"]["unbroken_count"] == 1),
                                 "fails_if": "a uniform uniaxial tensor does not read exactly one unbroken direction"}
    return res


STATES = [
    ("proton-analog", "P-0.5_plane_sc6_n32_pinned_d0.3"), ("proton-analog", "P-0.5_plane_sc6_n48_pinned_d0.3"),
    ("proton-analog", "P-0.5_plane_sc6_n48_pinned_d0.3_ext"), ("proton-analog", "P-0.5_plane_sc4_n32_pinned_d0.3"),
    ("proton-analog", "P-0.5_plane_sc6_n32_pinned_d0.2"), ("proton-analog", "P-0.5_plane_sc6_n32_pinned_d0.1"),
    ("neutron-analog", "P-1_plane_sc6_n32_pinned_d0.3"), ("neutron-analog", "P-1_plane_sc6_n48_pinned_d0.3"),
    ("neutron-analog", "P-1_plane_sc6_n32_pinned_d0.2"), ("neutron-analog", "P-1_plane_sc6_n32_pinned_d0.1"),
    ("lepton reference", "E+0.5_plane_sc6_n32_pinned_d0.3"),
]


def main():
    out = {"versions": {"python": sys.version.split()[0], "numpy": np.__version__},
           "thresholds": {"unbroken_eigenvalue_over_scale": UNBROKEN_EIG, "unbroken_growth": UNBROKEN_GROWTH,
                          "closed_form_tolerance": CLOSED_FORM_TOL, "column_cylindrical_radius": RHO_COLUMN, "windows": list(WINDOWS)},
           "controls": controls(), "states": {}}
    ok = all(v["pass"] for v in out["controls"].values())
    out["controls_pass"] = bool(ok)
    for k, v in out["controls"].items():
        print(f"control {k}: {'PASS' if v['pass'] else 'FAIL'}  {json.dumps({a: b for a, b in v.items() if a not in ('fails_if', 'note', 'pass')})}")
    if not ok:
        print("controls FAILED: no state is read"); sys.exit(1)
    for role, tag in STATES:
        z = np.load(os.path.join(DATA, f"m5_22_end_{tag}.npz"))
        res = read_state(z["M"], float(z["h"]), float(z["delta"]))
        res["role"] = role
        out["states"][tag] = res
        w = res["windows"]["18_22"]
        print(f"{role:17s} {tag:38s} word {res['word']:10s} dens/closed {w['summed_density_over_closed_form']:.4f}  "
              f"eig/trace {[round(d['eigenvalue_over_trace'], 4) for d in w['theta_eigen']]}  "
              f"growth {[round(d['growth_14_to_22'], 3) for d in w['theta_eigen']]} (cells {res['cell_count_growth_14_to_22']:.3f})  "
              f"min dens/closed {res['min_summed_density_over_closed_form_free_cells']:.3f}  pinned {res['pinned_shell_summed_density_over_closed_form']:.4f}  "
              f"I(22)/vac {res['I_trace_22_over_vacuum_volume']:.3f}  deviation reading: I(14) {res['spectral_deviation_reading']['I_trace'][5]:.0f} I(22) {res['spectral_deviation_reading']['I_trace'][-1]:.0f} "
              f"growth {100 * res['spectral_deviation_reading']['growth_14_to_22']:+.0f} % last step {100 * res['spectral_deviation_reading']['last_step_20_to_22']:+.1f} %")
    out["fails_if"] = ("the reading of M5.37 § 2 (at most one unbroken generator on the census proton-analog) fails if two or more "
                       "directions read UNBROKEN on the n 48 endpoint in both windows")
    p48 = out["states"]["P-0.5_plane_sc6_n48_pinned_d0.3"]["windows"]
    out["m5_37_reading_holds"] = bool(all(p48[k]["unbroken_count"] <= 1 for k in p48))
    print("M5.37 § 2 reading holds:", out["m5_37_reading_holds"])
    if "--write" in sys.argv:
        with open(os.path.join(DATA, "m5_38_inertia_gate.json"), "w") as f:
            json.dump(out, f, indent=1)
        print("wrote ../data/m5_38_inertia_gate.json")


if __name__ == "__main__":
    main()
