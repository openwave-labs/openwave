"""M5.32 R13-W, step W1: the static degenerate wall as a 1D profile (the frozen
packet, ledger 6.2, verbatim).

FUNCTIONAL   E_stat[M] on a profile M(z), vacuum at both ends (pinned shell depth
             1.6 at the z ends only; x, y uniform, 4 x 4 columns), the degeneracy
             d2 = d3 CONSTRAINED at z = 0 (the (2,3) block projected onto its trace
             part after every FIRE step), then RELEASED.
OBSERVABLES  sigma_0 = (E_stat[wall] - E_stat[vacuum]) / area, its h-convergence
             order over h in {1.5, 1.0, 0.75} at L = 48, its L-dependence over
             L in {48, 72, 96} at h = 1.5; whether the released profile keeps
             d2 = d3 at z = 0 (gap23 at the wall cell after release).
VERDICTS     pass: ESTABLISHED_KINEMATIC (finite, h-converged sigma_0).
             fail (released profile melts, sigma_0 -> 0): W2 decides. W1 never refutes.
W0 PREDICTION (m5_32_r13w_w0.py S3): E_u = 0 for every planar profile, so
             sigma_0 = h V4_deg exactly for a one-cell layer (V4_deg = 1.80e-6 per
             volume at the projected vacuum), order-1 convergence TO ZERO; released,
             the layer returns to the vacuum spectrum (V4 is the only force and its
             minimum is the vacuum orbit).
CONTROL      the ORIENTATION wall (the vacuum rotated by Delta q in the (2,3) plane
             for z > 0, no degeneracy): sigma = 0 exactly at every angle.
Numerical representation: FIRE dt0 0.01, dt_max 0.1, 12000 iterations or fmax < 1e-6.

Out: ../data/m5_32_r13w_w1.json, ../plots/m5_32_r13w_w1.png
Run: /opt/anaconda3/envs/master312/bin/python3 m5_32_r13w_w1.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m5_32_r13w_common", os.path.join(HERE, "m5_32_r13w_common.py"))
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
INS4 = C.INS4
OUT = os.path.join(C.DATA, "m5_32_r13w_w1.json")
PNG = os.path.join(C.PLOTS, "m5_32_r13w_w1.png")
NXY = 4
MAXIT, FTOL = 12000, 1e-6


def slab_cfg(n, L):
    cfg = C.cfg_of(n, L)
    return cfg


def wall_case(n, L, release=True):
    cfg = slab_cfg(n, L)
    h = cfg["h"]
    vac = INS4.vac4(cfg)
    M = np.broadcast_to(vac, (NXY, NXY, n, 4, 4)).copy()
    kz = n // 2
    wall = np.zeros((NXY, NXY, n), bool); wall[:, :, kz] = True
    wc = max(1, int(np.ceil(1.6 / h)))
    free = np.ones((NXY, NXY, n), bool); free[:, :, :wc] = False; free[:, :, n - wc:] = False
    area = NXY * NXY * h * h

    def proj(Mx):
        return C.degenerate_project(Mx, wall)
    t0 = time.time()
    Mc, info_c = C.fire_proj(M, cfg, free, MAXIT, project=proj, tag=f"w1_n{n}_L{L:g}_con", log_every=1000, f_tol=FTOL)
    e_u, e_v = INS4.e_parts(Mc, cfg)
    e_vac = sum(INS4.e_parts(np.broadcast_to(vac, M.shape).copy(), cfg))
    sigma = (e_u + e_v - e_vac) / area
    gap_c = float(C.gap23(Mc)[0, 0, kz])
    prof_c = C.gap23(Mc)[0, 0].tolist()
    rec = {"n": n, "L": L, "h": h, "area": area, "constrained": {
        "stop": info_c["stop"], "iters": info_c["iters"], "wall_s": info_c["wall_s"],
        "E_u": float(e_u), "V4": float(e_v), "E_vac": float(e_vac), "sigma_0": float(sigma),
        "gap_at_wall": gap_c, "eigs_at_wall": C.spatial_eigs(Mc)[0, 0, kz].tolist(),
        "M00_at_wall": float(Mc[0, 0, kz, 0, 0]), "gap_profile": prof_c}}
    C.log(f"W1 n{n} L{L:g} h{h:.3g} constrained: sigma_0 {sigma:.6e} (E_u {e_u:.3e}, V4 {e_v:.6e}), "
          f"gap at wall {gap_c:.3e}, stop {info_c['stop']} @ {info_c['iters']}")
    if release:
        Mr, info_r = C.fire_proj(Mc, cfg, free, MAXIT, project=None, tag=f"w1_n{n}_L{L:g}_rel", log_every=1000, f_tol=FTOL)
        e_u2, e_v2 = INS4.e_parts(Mr, cfg)
        gap_r = float(C.gap23(Mr)[0, 0, kz])
        rec["released"] = {"stop": info_r["stop"], "iters": info_r["iters"], "wall_s": info_r["wall_s"],
                           "E_u": float(e_u2), "V4": float(e_v2), "sigma": float((e_u2 + e_v2 - e_vac) / area),
                           "gap_at_wall": gap_r, "eigs_at_wall": C.spatial_eigs(Mr)[0, 0, kz].tolist(),
                           "max_dev_from_vacuum": float(np.max(np.abs(Mr - vac))),
                           "gap_profile": C.gap23(Mr)[0, 0].tolist(), "trace": info_r["trace"]}
        C.log(f"W1 n{n} L{L:g} released: sigma {rec['released']['sigma']:.3e}, gap at wall {gap_r:.4f} "
              f"(vacuum 0.3), max |M - vac| {rec['released']['max_dev_from_vacuum']:.2e}, stop {info_r['stop']} @ {info_r['iters']}")
    rec["constrained"]["trace"] = info_c["trace"]
    return rec


def release_stability(n=32, L=48.0, eps=1e-3):
    """DEVIATION LOGGED AT EXECUTE (not pre-registered): the exact descent keeps
    d2 = d3 on the released cell BY SYMMETRY (V4 depends on the spectrum only, so
    its gradient at a degenerate pair is isotropic in that block and cannot split
    it), which says nothing about stability.  Test: split the released cell's
    (2,3) block by eps diag(+1, -1) in its eigenframe and descend; record the gap.
    Closure: the V4 curvature along the splitting direction at the released cell
    (central difference), negative = the degenerate point is a maximum of V4 along
    the split = unstable."""
    cfg = slab_cfg(n, L)
    h = cfg["h"]
    vac = INS4.vac4(cfg)
    M = np.broadcast_to(vac, (NXY, NXY, n, 4, 4)).copy()
    kz = n // 2
    wall = np.zeros((NXY, NXY, n), bool); wall[:, :, kz] = True
    wc = max(1, int(np.ceil(1.6 / h)))
    free = np.ones((NXY, NXY, n), bool); free[:, :, :wc] = False; free[:, :, n - wc:] = False
    Mc, _ = C.fire_proj(M, cfg, free, MAXIT, project=lambda X: C.degenerate_project(X, wall), tag="w1_stab_con", log_every=1000, f_tol=FTOL)
    Mr, _ = C.fire_proj(Mc, cfg, free, MAXIT, project=None, tag="w1_stab_rel", log_every=1000, f_tol=FTOL)
    gap_rel = float(C.gap23(Mr)[0, 0, kz])
    # V4 curvature along the split at the released cell
    cell = Mr[0, 0, kz]
    w, V = np.linalg.eigh(cell[1:, 1:])
    split = np.zeros((4, 4)); split[1:, 1:] = np.outer(V[:, 1], V[:, 1]) - np.outer(V[:, 0], V[:, 0])
    p = C.LAG.default_params(s=C.S, g=cfg["g"], delta=cfg["delta"])

    def v4(Mcell):
        return float(C.LAG.v4_density_np(None, Mcell[None], p)[0])
    s = 1e-3
    curv = (v4(cell + s * split) - 2 * v4(cell) + v4(cell - s * split)) / (s * s)
    # perturb and descend
    Mp = Mr.copy()
    Mp[:, :, kz] = cell + eps * split
    gaps = []

    def diag(X):
        gaps.append(float(C.gap23(X)[0, 0, kz]))
        return {"gap_wall": gaps[-1]}
    Mq, info = C.fire_proj(Mp, cfg, free, MAXIT, project=None, tag="w1_stab_pert", log_every=1000, f_tol=FTOL, diag=diag)
    gap_end = float(C.gap23(Mq)[0, 0, kz])
    gap_start = float(C.gap23(Mp)[0, 0, kz])          # = 2 eps for the diag(+1, -1) split (audit G6)
    # per-cell descent of the split cell alone (the slab is per-cell once E_u = 0): where does the split go?
    from scipy.optimize import minimize
    x0 = (cell + eps * split)
    iu = np.triu_indices(4)

    def unpack(x):
        Mx = np.zeros((4, 4)); Mx[iu] = x; return Mx + Mx.T - np.diag(np.diag(Mx))
    rr = minimize(lambda x: v4(unpack(x)), x0[iu], method="BFGS", options={"gtol": 1e-14, "maxiter": 5000})
    eigs_cell_min = C.spatial_eigs(unpack(rr.x)[None])[0].tolist()
    rec = {"n": n, "L": L, "h": h, "eps": eps, "gap_released_exact_descent": gap_rel, "gap_start": gap_start,
           "per_cell_descent_from_split": {"V4_end": float(rr.fun), "eigs_end": eigs_cell_min, "gap_end": float(eigs_cell_min[1] - eigs_cell_min[2])},
           "V4_curvature_along_split": curv, "gap_trace_after_perturbation": [(r["it"], r["gap_wall"]) for r in info["trace"]],
           "gap_end": gap_end, "eigs_end": C.spatial_eigs(Mq)[0, 0, kz].tolist(), "stop": info["stop"], "iters": info["iters"],
           "gap_monotone_growth": bool(all(b >= a for a, b in zip(gaps, gaps[1:]))),
           "unstable": bool(curv < 0 and gap_end > gap_start * (1 + 1e-3) and all(b >= a for a, b in zip(gaps, gaps[1:]))
                            and eigs_cell_min[1] - eigs_cell_min[2] > 0.29)}
    C.log(f"release stability n{n} L{L:g}: exact-descent gap {gap_rel:.2e}; V4 curvature along the split {curv:+.3e}; "
          f"after the eps = {eps:g} split the gap goes {gap_start:.4g} -> {gap_end:.6g} in {info['iters']} FIRE it "
          f"(slow: the cell's stiffness ratio collapses FIRE's dt); the per-cell descent from the split point ends at "
          f"spectrum {[round(v, 4) for v in eigs_cell_min]} (gap {eigs_cell_min[1] - eigs_cell_min[2]:.4f}); unstable = {rec['unstable']}")
    return rec


def orientation_wall(n=32, L=48.0):
    cfg = slab_cfg(n, L)
    vac = INS4.vac4(cfg)
    kz = n // 2
    rows = []
    for dq in np.linspace(0.0, 2 * np.pi, 13):
        M = np.broadcast_to(vac, (NXY, NXY, n, 4, 4)).copy()
        Rq = C.rot(C.G1, dq)
        M[:, :, kz:] = Rq @ vac @ Rq.T
        e_u, e_v = INS4.e_parts(M, cfg)
        rows.append({"dq": float(dq), "E_u": float(e_u), "V4": float(e_v)})
    return rows


if __name__ == "__main__":
    res = {"h_ladder": [], "L_ladder": [], "orientation_wall": orientation_wall()}
    ow = res["orientation_wall"]
    C.log(f"orientation wall control: max E_u {max(r['E_u'] for r in ow):.3e}, max V4 {max(r['V4'] for r in ow):.3e} over 13 angles")
    for n, L in ((32, 48.0), (48, 48.0), (64, 48.0)):
        res["h_ladder"].append(wall_case(n, L))
        json.dump(res, open(OUT, "w"), indent=1)
    for n, L in ((48, 72.0), (64, 96.0)):
        res["L_ladder"].append(wall_case(n, L))
        json.dump(res, open(OUT, "w"), indent=1)
    res["release_stability"] = release_stability()
    json.dump(res, open(OUT, "w"), indent=1)
    # convergence order of sigma_0 in h (three refinements at L = 48)
    hs = np.array([r["h"] for r in res["h_ladder"]])
    sg = np.array([r["constrained"]["sigma_0"] for r in res["h_ladder"]])
    order = float(np.polyfit(np.log(hs), np.log(np.abs(sg) + 1e-300), 1)[0])
    w0 = json.load(open(os.path.join(C.DATA, "m5_32_r13w_w0.json")))
    v4deg = w0.get("S3_numbers", {}).get("V4_deg_per_volume")
    v4min = w0.get("S6_numbers", {}).get("V4_deg_constrained_min")
    res["summary"] = {
        "sigma_0_vs_h": {str(r["h"]): r["constrained"]["sigma_0"] for r in res["h_ladder"]},
        "sigma_0_vs_L_at_h1.5": {str(r["L"]): r["constrained"]["sigma_0"] for r in res["h_ladder"][:1] + res["L_ladder"]},
        "h_convergence_order_of_sigma_0": order,
        "W0_prediction_sigma_0_eq_h_V4deg_projected_point": {str(r["h"]): (r["h"] * v4deg if v4deg else None) for r in res["h_ladder"]},
        "converged_reference_h_V4deg_min": {str(r["h"]): r["h"] * v4min for r in res["h_ladder"]} if v4min else None,
        "sigma_0_is_unconverged_snapshot": True,
        "fmax_at_stop": {str(r["h"]): r["constrained"]["trace"][-1]["fmax"] for r in res["h_ladder"]},
        "released_gap_at_wall": {f"n{r['n']}_L{r['L']:g}": r["released"]["gap_at_wall"] for r in res["h_ladder"] + res["L_ladder"]},
        "released_keeps_degeneracy": bool(all(abs(r["released"]["gap_at_wall"]) < 1e-3 for r in res["h_ladder"] + res["L_ladder"])),
        "orientation_wall_max_energy": max(r["E_u"] + r["V4"] for r in ow),
        "release_stability": {k: v for k, v in res["release_stability"].items() if k != "gap_trace_after_perturbation"},
    }
    s = res["summary"]
    finite_converged = (abs(order) < 0.5) and (abs(sg[-1]) > 1e-12)
    st = res["release_stability"]
    s["verdict"] = ("ESTABLISHED_KINEMATIC (finite h-converged sigma_0)" if finite_converged
                    else "W1 FAIL as pre-registered: sigma_0 ~ h^%.2f -> 0; the released layer keeps d2 = d3 under the exact "
                    "descent by symmetry only (V4 curvature along the split %+.2e) and %s under an eps = %g split; W2 decides"
                    % (order, st["V4_curvature_along_split"],
                       "is UNSTABLE (negative curvature; the gap grows monotonically %.4g -> %.6g under FIRE, slowly because the cell's stiffness ratio collapses dt; the per-cell descent from the split point reaches the vacuum spectrum, gap %.4f)" % (st["gap_start"], st["gap_end"], st["per_cell_descent_from_split"]["gap_end"])
                       if st["unstable"] else "holds (gap %.2e)" % st["gap_end"], st["eps"]))
    C.log("SUMMARY " + json.dumps(s, indent=None))
    json.dump(res, open(OUT, "w"), indent=1)
    # plot
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].loglog(hs, np.abs(sg), "o-", label="sigma_0 (constrained wall)")
    if v4deg:
        ax[0].loglog(hs, hs * v4deg, "k--", label="W0: h V4_deg")
    ax[0].set_xlabel("h"); ax[0].set_ylabel("sigma_0"); ax[0].legend(); ax[0].set_title(f"W1 tension vs h (order {order:.2f})")
    r0 = res["h_ladder"][0]
    z = (np.arange(r0["n"]) - (r0["n"] - 1) / 2) * r0["h"]
    ax[1].plot(z, r0["constrained"]["gap_profile"], "o-", label="constrained")
    ax[1].plot(z, r0["released"]["gap_profile"], "s-", label="released")
    ax[1].set_xlabel("z"); ax[1].set_ylabel("d2 - d3"); ax[1].legend(); ax[1].set_title("gap profile, n32 L48")
    ax[2].plot([r["dq"] for r in ow], [r["E_u"] + r["V4"] for r in ow], "o-")
    ax[2].set_xlabel("Delta q"); ax[2].set_ylabel("E_stat"); ax[2].set_title("orientation wall (control)")
    fig.tight_layout(); fig.savefig(PNG, dpi=110)
    C.log(f"written {OUT} and {PNG}")
