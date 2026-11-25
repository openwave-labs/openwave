"""M5.56: one chain (or one re-relaxation of K5) per process, single thread, on m5_51_action.py and m5_56_turn.py.

The rows of the plan that run (section 9 at delta = 0; sections 11 and 12 at delta = 0.05), each a descent of U over
the six components of the free nodes (faces, and in rows 11 to 18 the ball rho <= R1, held at the start's values):

  1 N1 (null chain of H1), 2 B2, 3 B3 (the references of H2 and H3), 6 to 10 (free core, T[2pi]; 8 and 9 by nonlinear
  CG), 11, 13, 16 (held core, T[2pi]), 17, 18 (held core, T[4pi]); 22 B4a, 23 B4b, 24 B4-H2 (references at
  delta = 0.05), 25 to 27 (T[2pi]), 28, 29 (W[2pi]), 30 to 33 (the pair ladder d0 = 14, 10, 6, 0), 34, 35 (T[-2pi]
  about ring 1 of a two-ring end), 36, 37 (W[2pi] about ring 1)

Run rules (plan, sections 4 and 14): L-BFGS-B, maxcor 10, ftol 0, gtol 1e-7 on the gradient with its factor h^3,
maxfun 1.25 times the cap; rows 8 and 9 scipy CG, gtol 1e-7 in the largest-component norm; caps 30000 iterations a
chain and 10000 a K5, 5 hours a process; up to three restarts after an abnormal line-search end (L-BFGS-B) or a loss
of precision (CG); route stops at 0 (the start), 10, 20, 50, 100, 200, 300, 500, 700, 1000, then every 500, and the
end; major stops at 1000, 3000, 10000, 20000, 30000 and the end. Noise: 1e-4 normal draws on each component of each
free node, numpy.random.default_rng(56000 + row) (rows 8, 9 the draws of rows 6, 7); K5 seeds 56100 + row.

  CONVERGED (delta = 0)  gate (max |dU/dm| / h^3 over the free nodes) <= 1e-3 at the end, and the end by gtol or two
                         major stops >= 5000 iterations apart within 1e-6 relative in U and 0.002 in every R_L
  CONVERGED (delta > 0)  gate <= 1e-3 and minor torque <= 1e-5 at the end, and the end by gtol or two major stops
                         >= 5000 apart within 1e-8 in U, 0.002 in every R_L and mu_free <= 0.003 between the two
  CAPPED                 any other end

Usage:  python m5_56_relax.py --row 6 | --k5 6 | --list | --spec 6
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_51_action as A  # noqa: E402
import m5_51_reads as RD  # noqa: E402
import m5_56_turn as T  # noqa: E402

CK = T.CK
WORK = Path(os.environ.get("M556_WORK", str(CK / "chains")))
SUPP = Path(os.environ.get("M556_SUPP", str(CK / "supp")))
MAJOR = (1000, 3000, 10000, 20000, 30000)
WALL_HOURS = float(os.environ.get("M556_WALL_HOURS", "5.0"))
TWO_PI = 2 * np.pi

ROWS = {
    1: dict(name="N1", grid="H1", delta=0.0, ref="G6", start="ref", noise=True, held="faces", turn=("T", 0.0, "normal", 11.0), rung="null"),
    2: dict(name="B2", grid="H2", delta=0.0, ref=None, start="interp_G6", noise=False, held="faces", turn=None, rung="reference"),
    3: dict(name="B3", grid="H3", delta=0.0, ref=None, start="embed_G6", noise=False, held="faces", turn=None, rung="reference"),
    6: dict(name="F-H1-2n", grid="H1", delta=0.0, ref="G6", start="turn", noise=True, held="faces", turn=("T", TWO_PI, "normal", 11.0), rung=1),
    7: dict(name="F-H1-2i", grid="H1", delta=0.0, ref="G6", start="turn", noise=True, held="faces", turn=("T", TWO_PI, "inplane", 11.0), rung=1),
    8: dict(name="F-H1-2n-CG", grid="H1", delta=0.0, ref="G6", start="turn", noise=True, noise_row=6, held="faces", turn=("T", TWO_PI, "normal", 11.0),
            rung=1, opt="CG"),
    9: dict(name="F-H1-2i-CG", grid="H1", delta=0.0, ref="G6", start="turn", noise=True, noise_row=7, held="faces", turn=("T", TWO_PI, "inplane", 11.0),
            rung=1, opt="CG"),
    10: dict(name="F-H2-2n", grid="H2", delta=0.0, ref=2, start="turn", noise=True, held="faces", turn=("T", TWO_PI, "normal", 11.0), rung=1),
    11: dict(name="K-H1-2n", grid="H1", delta=0.0, ref="G6", start="turn", noise=True, held="ball", turn=("T", TWO_PI, "normal", 11.0), rung=2),
    13: dict(name="K-H2-2n", grid="H2", delta=0.0, ref=2, start="turn", noise=True, held="ball", turn=("T", TWO_PI, "normal", 11.0), rung=2),
    16: dict(name="K-H3w-2n", grid="H3", delta=0.0, ref=3, start="turn", noise=True, held="ball", turn=("T", TWO_PI, "normal", 14.0), rung=2),
    17: dict(name="K-H2-4n", grid="H2", delta=0.0, ref=2, start="turn", noise=True, held="ball", turn=("T", 2 * TWO_PI, "normal", 11.0), rung=2),
    18: dict(name="K-H3w-4n", grid="H3", delta=0.0, ref=3, start="turn", noise=True, held="ball", turn=("T", 2 * TWO_PI, "normal", 14.0), rung=2),
    22: dict(name="B4a", grid="H1", delta=T.DELTA, ref=None, start="seed_bi", noise=True, held="faces", turn=None, rung="reference"),
    23: dict(name="B4b", grid="H1", delta=T.DELTA, ref=None, start="G6_bi_faces", noise=False, held="faces", turn=None, rung="reference"),
    24: dict(name="B4-H2", grid="H2", delta=T.DELTA, ref=None, start="interp_H1d", noise=False, held="faces", turn=None, rung="reference"),
    25: dict(name="F4-H1-Tn", grid="H1", delta=T.DELTA, ref="H1d", start="turn", noise=True, held="faces", turn=("T", TWO_PI, "normal", 11.0), rung=4),
    26: dict(name="F4-H1-Ti", grid="H1", delta=T.DELTA, ref="H1d", start="turn", noise=True, held="faces", turn=("T", TWO_PI, "inplane", 11.0), rung=4),
    27: dict(name="F4-H2-Tn", grid="H2", delta=T.DELTA, ref=24, start="turn", noise=True, held="faces", turn=("T", TWO_PI, "normal", 11.0), rung=4),
    28: dict(name="F4-H1-W", grid="H1", delta=T.DELTA, ref="H1d", start="turn", noise=True, held="faces", turn=("W", TWO_PI, None, 11.0), rung=4),
    29: dict(name="F4-H2-W", grid="H2", delta=T.DELTA, ref=24, start="turn", noise=True, held="faces", turn=("W", TWO_PI, None, 11.0), rung=4),
    30: dict(name="P4-14", grid="P", delta=T.DELTA, ref=None, start="pair", d0=14.0, noise=True, held="faces", turn=None, rung="pair"),
    31: dict(name="P4-10", grid="P", delta=T.DELTA, ref=None, start="pair", d0=10.0, noise=True, held="faces", turn=None, rung="pair"),
    32: dict(name="P4-6", grid="P", delta=T.DELTA, ref=None, start="pair", d0=6.0, noise=True, held="faces", turn=None, rung="pair"),
    33: dict(name="P4-0", grid="P", delta=T.DELTA, ref=None, start="pair", d0=0.0, noise=True, held="faces", turn=None, rung="pair"),
    34: dict(name="P4-T-wide", grid="P", delta=T.DELTA, ref="Pwide", start="turn", noise=True, held="faces", turn=("T", -TWO_PI, "z", "pair"), rung=5),
    35: dict(name="P4-T-narrow", grid="P", delta=T.DELTA, ref="Pnarrow", start="turn", noise=True, held="faces", turn=("T", -TWO_PI, "z", "pair"), rung=5),
    36: dict(name="P4-W-wide", grid="P", delta=T.DELTA, ref="Pwide", start="turn", noise=True, held="faces", turn=("W", TWO_PI, None, "pair"), rung=5),
    37: dict(name="P4-W-narrow", grid="P", delta=T.DELTA, ref="Pnarrow", start="turn", noise=True, held="faces", turn=("W", TWO_PI, None, "pair"), rung=5),
}


def cname(row):
    return f"r{row:02d}_{ROWS[row]['name']}"


def load_rec(name):
    f = WORK / f"{name}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d if d.get("finished") else None


def load_end(name):
    rec = load_rec(name)
    z = np.load(WORK / name / rec["end_file"], allow_pickle=False)
    return np.asarray(z["m6"], float), np.asarray(z["x"], float), rec


def qualifies(row):
    """A reference enters a chain only if CONVERGED, K5 met, one ring of degree 1 in regime (TWO RINGS on P)."""
    rec = load_rec(cname(row))
    k5 = load_rec("k5_" + cname(row))
    if rec is None or rec.get("end_state") != "CONVERGED" or k5 is None or not k5.get("k5", {}).get("passed"):
        return False
    e = rec["end"]
    if ROWS[row]["grid"] == "P":
        return e.get("pair", {}).get("word") == "TWO RINGS"
    return bool(e.get("one_ring") and e.get("line_regime") and e.get("degree", {}).get("reads_one"))


def h1d_ref_row():
    if qualifies(22):
        return 22
    if load_rec("k5_" + cname(22)) is not None or (load_rec(cname(22)) is not None and load_rec(cname(22))["end_state"] != "CONVERGED"):
        if qualifies(23):
            return 23
    return None


def pair_refs():
    """The two-ring ends of rows 30 to 33 (CONVERGED, K5 met): the widest and the narrowest separation."""
    ends = []
    for r in (30, 31, 32, 33):
        if qualifies(r):
            ends.append((load_rec(cname(r))["end"]["pair"]["separation"], r))
    if not ends:
        return None, None
    ends.sort()
    wide = ends[-1][1]
    narrow = ends[0][1] if len(ends) > 1 else None
    return wide, narrow


def reference(row):
    """(field, x, ring list, U_ref, name) of a row's reference."""
    spec = ROWS[row]
    ref = spec["ref"]
    if ref == "G6":
        m6, x, e = T.load51("G6_mid_a")
        return m6, x, [T.ring_of(e)], "G6_mid_a", e
    if ref == "H1d":
        r = h1d_ref_row()
        if r is None:
            raise RuntimeError("not run: no reference")
        ref = r
    if ref in ("Pwide", "Pnarrow"):
        w, n = pair_refs()
        ref = w if ref == "Pwide" else n
        if ref is None:
            raise RuntimeError("not run: no two-ring state" if w is None else "not run: one two-ring end")
    if not qualifies(ref):
        raise RuntimeError("not run: no reference")
    m6, x, rec = load_end(cname(ref))
    e = rec["end"]
    if ROWS[ref]["grid"] == "P":
        rings = [dict(c=np.asarray(q["c"]), nu=T.unit(q["nu"]), RL=q["RL"]) for q in e["pair"]["rings"]]
    else:
        rings = [T.ring_of(e)]
    return m6, x, rings, cname(ref), e


# ----------------------------------------------------------------------------- the job

class Job:
    pass


def build(row, k5=False):
    spec = ROWS[row]
    j = Job()
    j.row, j.spec, j.k5 = row, spec, k5
    j.out = WORK
    j.name = ("k5_" if k5 else "") + cname(row)
    j.grid, j.delta = spec["grid"], spec["delta"]
    hb, h, N = T.GRIDS[j.grid]
    j.x = T.axis_of(j.grid)
    j.h, j.N = h, N
    X = T.mesh(j.x)
    j.info = {}
    j.ref6 = None
    j.rings = []
    j.collar = None
    face = T.faces_mask(N)
    # the field the faces hold
    if j.delta == 0.0:
        hold = A.to6(A.Model(N, hb).M0)
    elif j.grid == "P":
        hold = T.pair_bi(j.x, 14.0)
    else:
        hold = T.seed_bi(j.x)
    if spec["ref"] is not None:
        j.ref6, xr, j.rings, j.ref_name, j.ref_end = reference(row)
        assert np.allclose(xr, j.x)
    st = spec["start"]
    if st == "ref":
        m6 = j.ref6.copy()
    elif st == "interp_G6":
        g6, xs, _ = T.load51("G6_mid_a")
        m6 = T.interp_to(g6, xs, j.x)
        j.info["face_deviation_before_reset"] = float(np.abs(m6 - hold)[face].max())
    elif st == "embed_G6":
        g6, xs, _ = T.load51("G6_mid_a")
        m6 = hold.copy()
        i0 = int(round((xs[0] - j.x[0]) / h))
        m6[i0:i0 + len(xs), i0:i0 + len(xs), i0:i0 + len(xs)] = g6
    elif st == "seed_bi":
        m6 = hold.copy()
    elif st == "G6_bi_faces":
        g6, xs, _ = T.load51("G6_mid_a")
        m6 = np.where(face[..., None], hold, g6)
    elif st == "interp_H1d":
        r = h1d_ref_row()
        if r is None:
            raise RuntimeError("not run: no reference")
        src, xs, _ = load_end(cname(r))
        m6 = T.interp_to(src, xs, j.x)
        j.info["source"] = cname(r)
        j.info["face_deviation_before_reset"] = float(np.abs(m6 - hold)[face].max())
    elif st == "pair":
        m6 = T.pair_bi(j.x, spec["d0"])
    elif st == "turn":
        kind, Th, axname, R2 = spec["turn"]
        rg = j.rings[0]
        pair = R2 == "pair"
        if pair:
            c1, c2 = j.rings[0], j.rings[1]
            sep = float(np.linalg.norm(c1["c"] - c2["c"]))
            R1 = c1["RL"] + 1.5
            R2 = min(sep - c2["RL"] - 1.5, min(hb - abs(c1["c"][a]) for a in range(3)) - 2 * h)
            c = c1["c"]
            outer = R2 + h
            exclude = np.linalg.norm(X - c2["c"], axis=-1) <= c2["RL"] + 2.0
        else:
            R1 = rg["RL"] + 1.5
            c = rg["c"]
            outer = None
            exclude = None
        axis = dict(normal=rg["nu"], inplane=T.inplane_axis(rg["nu"]), z=np.array([0, 0, 1.0])).get(axname)
        if kind == "T":
            m6 = T.T_turn(j.ref6, j.x, Th, axis, c, R1, R2)
        else:
            m6 = T.W_turn(j.ref6, j.ref6, j.x, Th, c, R1, R2, fill=pair)
        j.collar = dict(kind=kind, Th=Th, axis=None if axis is None else np.asarray(axis), axname=axname, c=np.asarray(c), R1=R1, R2=R2, outer=outer,
                        exclude=exclude, pair=pair)
        j.info.update(collar=dict(kind=kind, Theta=Th, axis=None if axis is None else list(axis), c=list(c), R1=R1, R2=R2, outer=outer, pair=pair))
    else:
        raise ValueError(st)
    if spec["start"] in ("ref",) and spec["rung"] == "null":
        rg = j.rings[0]
        j.collar = dict(kind="T", Th=0.0, axis=rg["nu"], axname="normal", c=rg["c"], R1=rg["RL"] + 1.5, R2=11.0, outer=None, exclude=None, pair=False)
    # held nodes
    held = np.zeros((N, N, N), bool)
    if spec["held"] == "ball":
        rho = np.linalg.norm(X - j.collar["c"], axis=-1)
        held = rho <= j.collar["R1"]
        j.info["held_ball_nodes"] = int(held.sum())
    j.held = held
    hold = np.where(held[..., None], m6, hold)
    j.mod = T.make_model(j.grid, j.delta, hold6=hold, held=held)
    m6 = j.mod.hold_faces(m6)
    j.free = j.mod.mask.copy()
    j.info["faces_vs_hold"] = float(np.abs(m6 - hold)[face].max())
    if j.grid == "P":
        q = T._cube_q(m6, j.x, np.zeros(3), hb - h)
        j.info["box_surface"] = dict(liftable=q["liftable"], abs_q=q["abs_q"])
    j.start_clean = m6.copy()
    # the start rule (turned starts, before the noise)
    j.start_rule = None
    if st == "turn":
        cl = j.collar
        rule = T.start_rule(m6, j.x, j.delta, cl["Th"], cl["R1"], cl["R2"], j.rings, minor=j.delta > 0)
        if cl["kind"] == "W":
            if cl["pair"]:
                _, col, reached, ncol, least = T.e1_fill(j.ref6, j.x, cl["c"], cl["R1"], cl["R2"])
                w, _ = T.eigs(j.ref6)
                g12c = float((w[..., 0] - w[..., 1])[col].min())
                wreg = dict(reached=reached, collar_nodes=ncol, least_link_dot=least, least_g12=g12c,
                            regime=bool(reached == ncol and least >= np.cos(np.radians(45)) and g12c >= 0.5))
            else:
                wreg = T.W_regime(j.ref6, j.x, cl["c"], cl["R1"], cl["R2"])
            rule["W_regime"] = wreg
            rule["passes"] = bool(rule["passes"] and wreg["regime"])
        j.start_rule = rule
    # the noise
    if spec["noise"] or k5:
        if k5:
            sd = 56100 + row
        else:
            sd = 56000 + spec.get("noise_row", row)
        j.noise_seed = sd
    else:
        j.noise_seed = None
    j.tube = T.tubes(j.x, j.rings) if j.rings else np.zeros((N, N, N), bool)
    j.rho_c = None if j.collar is None else np.linalg.norm(X - j.collar["c"], axis=-1)
    if j.collar is not None and spec["rung"] == 1:
        j.interior = j.free & (j.rho_c <= j.collar["R1"]) & ~j.tube
    else:
        j.interior = None
    if j.ref6 is not None:
        j.mod.evaluate(j.ref6, need_grad=False)
        j.ref_dens = j.mod.edens.copy()
        j.U_ref = float(j.mod.evaluate(j.ref6, need_grad=False)["U"])
    if j.collar is not None:
        cl = j.collar
        j.col_mask = (j.rho_c > cl["R1"]) & (j.rho_c < cl["R2"]) & (True if cl["exclude"] is None else ~cl["exclude"])
        if cl["kind"] == "W" and cl["pair"]:
            j.e1_fill = T.e1_pair_profile(j.ref6, j.x, cl["c"], cl["R1"], cl["R2"])
        else:
            j.e1_fill = None
    return j


def add_noise(j, m6):
    if j.noise_seed is None:
        return m6
    rng = np.random.default_rng(j.noise_seed)
    out = m6.copy()
    out[j.free] += 1e-4 * rng.standard_normal((int(j.free.sum()), 6))
    return out


# ----------------------------------------------------------------------------- reads at the stops

def route_read(j, z, it, major):
    m6 = np.asarray(z, float).reshape(j.N, j.N, j.N, 6)
    out = dict(iteration=it, major=bool(major))
    tq = T.minor_torque(j.mod, m6) if j.delta > 0 else None
    if tq is not None:
        out.update(U=tq["U"], U4=tq["U4"], UV=tq["UV"], ULV=tq["ULV"], gate=tq["gate"], torque=tq["torque"])
        dens = j.mod.edens.copy()
    else:
        p = j.mod.evaluate(m6, need_grad=True)
        dens = j.mod.edens.copy()
        out.update(U=p["U"], U4=p["U4"], UV=p["UV"], ULV=p["ULV"], gate=float(np.abs(p["grad"]).max() / j.h ** 3))
    if j.ref6 is not None:
        out["dU"] = out["U"] - j.U_ref
    cl = j.collar
    if cl is not None:
        tr = T.turn_reader(m6, j.ref6, j.x, cl["c"], cl["R1"], outer=cl["outer"], exclude=cl["exclude"], axis=cl["axis"], dens=(dens, j.ref_dens))
        out["turn"] = dict(sigma=tr["sigma"], q0_sign_shell=tr["q0_sign_shell"], sign_shell=tr["sign_shell"], chi_max=tr["chi_max"],
                           rho_t_cross=tr["rho_t_cross"], rho_least_q0=tr["rho_least_q0"], r_max=tr["r_max"], step_max=tr["step_max"],
                           q0_outer=tr["q0_outer"], regime=tr["regime"], in_regime=tr["in_regime"], dU_inside_sign_shell=tr.get("dU_inside_sign_shell"),
                           shells=[dict(rho=r["rho"], q=r["q"], r=r["r"], chi=r["chi"], angle=r.get("angle"), dU=r.get("dU_shell"), n=r["n"]) for r in tr["rows"]])
        if j.interior is not None:
            last = tr["rows"][-1]
            out["interior"] = T.interior_fit(m6, j.ref6, j.x, j.interior, last["q"], cl["axis"], last.get("angle", 0.0))
    if j.ref6 is not None or j.spec["rung"] in (1, 2, 4, 5):
        if j.spec["held"] == "ball":
            out["tear"] = T.tear_reads(m6, j.x, j.delta, None, j.free)
        else:
            out["tear"] = T.tear_reads(m6, j.x, j.delta, j.tube if j.rings else None, j.free)
    else:
        w, _ = T.eigs(m6)
        g = w[..., 0] - w[..., 1]
        out["least_g12"] = float(g[j.free].min())
    if j.delta > 0 and j.ref6 is not None:
        reg = j.free & ~j.tube
        out["mu_free"] = T.mu(m6, j.ref6, reg)
        if cl is not None:
            out["mu_col"] = T.mu(m6, j.ref6, j.col_mask)
            if cl["kind"] == "W":
                mp = T.minor_profile(m6, j.ref6, j.x, cl["c"], cl["R1"], outer=cl["outer"], exclude=cl["exclude"], e1_ref=j.e1_fill)
                out["minor"] = dict(alpha_s=mp["alpha_s"], sign_shell=mp["sign_shell"], in_regime=mp["in_regime"], regime=mp["regime"],
                                    min_coherence=mp["min_coherence"], min_align=mp["min_align"],
                                    shells=[[r["rho"], r["alpha"], r["coherence"], r["align"]] for r in mp["rows"]])
    if major:
        out.update(major_read(j, m6))
    return out


def major_read(j, m6):
    out = {}
    if j.grid == "P":
        out["pair"] = T.two_ring_reader(m6, j.x)
        if j.delta > 0:
            out["lines"] = T.lines_read(m6, j.x, np.zeros(3), [0, 0, 1.0], (12.0, 14.0, 16.0), j.delta, tube_mask=j.tube if j.rings else None)
    else:
        hb = T.GRIDS[j.grid][0]
        r = RD.read_state(m6, j.x, hb, "mid", with_grad=False)
        keep = ("RL", "center", "normal", "line_regime", "line_gmax", "halfturn_all", "cls", "angle_axis", "angle_diag", "one_ring", "all_crossings",
                "consecutive_ok", "far_nodes", "label", "largest_share", "degree", "RA", "RA_nodes", "first_guess", "eigenvalue_min", "eigenvalue_max",
                "least_gap_node", "N_over_U", "loophole", "out_of_plane_spread", "m2_harmonic_rho", "P")
        out.update({k: r.get(k) for k in keep})
        if j.delta > 0:
            c = np.asarray(r["center"]) if r.get("center") is not None else np.zeros(3)
            nu = np.asarray(r["normal"]) if r.get("normal") is not None else np.array([0, 0, 1.0])
            ring_t = T.tubes(j.x, [dict(c=c, nu=nu, RL=r["RL"])]) if r.get("RL") is not None else None
            out["lines"] = T.lines_read(m6, j.x, c, nu, (6.0, 8.0, 10.0), j.delta, tube_mask=ring_t)
    return out


# ----------------------------------------------------------------------------- the descent

def run(j, m6_0, cap, work, rec, log, guard, deadline):
    state_f = work / "state.json"
    st = json.loads(state_f.read_text()) if state_f.exists() else dict(iteration=0, nfev=0, stops=[], restarts=[], resumes=[], saved=[])
    x = m6_0.reshape(-1).copy()
    if st["iteration"] > 0:
        last = st["saved"][-1]
        x = np.load(work / last["file"], allow_pickle=False)["m6"].reshape(-1)
        st["resumes"].append(st["iteration"])
        log(f"resumes at it {st['iteration']} from {last['file']} with a fresh optimizer memory")
    box = dict(it=st["iteration"], nfev=0, cut=None)
    opt = j.spec.get("opt", "LBFGSB")
    majors = tuple(m for m in MAJOR if m <= cap)

    def is_route(it):
        return it in (10, 20, 50, 100, 200, 300, 500, 700, 1000) or (it > 1000 and it % 500 == 0)

    def save_state():
        state_f.write_text(json.dumps(dict(st, nfev=st["nfev"] + box["nfev"]), default=T._jsonable))

    def stop(z, it, kind):
        major = it in majors or kind == "end"
        f = f"s{it:05d}.npz" if kind != "end" else "end.npz"
        np.savez(work / f, m6=np.asarray(z).reshape(j.N, j.N, j.N, 6), x=j.x)
        r = route_read(j, z, it, major)
        r.update(kind=kind, file=f)
        st["stops"].append(r)
        st["saved"].append(dict(iteration=it, file=f, major=bool(major), keep=bool(it <= 1000 or major or kind == "end")))
        # prune: keep route stops to 1000, majors, the last three route stops
        route_only = [s for s in st["saved"] if not s["keep"]]
        for s in route_only[:-3]:
            (work / s["file"]).unlink(missing_ok=True)
            s["deleted"] = True
            s["keep"] = True
        st["iteration"] = it
        save_state()
        tline = ""
        if "turn" in r:
            t = r["turn"]
            tline = f" sigma {t['sigma']} q0s {t['q0_sign_shell']:.3f} chi {t['chi_max']:.3f} reg {t['in_regime']}"
        if "interior" in r:
            tline += f" r_int {r['interior']['r_int']:.3f} psi_int {r['interior']['psi_int']:.3f}"
        if "tear" in r:
            tline += f" tear {r['tear']['tear']} gmin {r['tear']['least_g12_outside']:.3f}"
        if "mu_free" in r:
            tline += f" mu {r['mu_free']['mu']:.4f}"
        if "minor" in r:
            tline += f" alpha_s {r['minor']['alpha_s']:.3f}"
        if "pair" in r:
            tline += f" pair {r['pair']['word']} comps {r['pair']['components']} sep {r['pair'].get('separation')}"
        if "RL" in r and major:
            tline += f" RL {r['RL']} cls {r.get('cls')} one_ring {r.get('one_ring')}"
        log(f"{kind} it {it} U {r['U']:.10f} gate {r['gate']:.3e}{tline}")
        return r

    if st["iteration"] == 0 and not st["stops"]:
        stop(x, 0, "start")

    def fun(z):
        p = j.mod.evaluate(z)
        box["nfev"] += 1
        return p["U"], p["grad"].reshape(-1)

    def cb(z, *args):
        box["it"] += 1
        it = box["it"]
        if is_route(it) or it in majors:
            stop(z.copy() if not hasattr(z, "x") else z.x.copy(), it, "stop")
            (work / "progress.json").write_text(json.dumps(dict(chain=j.name, iteration=it, time=time.strftime("%Y-%m-%d %H:%M:%S"))))
            if guard is not None:
                try:
                    if guard.wrap_up():
                        box["cut"] = "power"
                        raise StopIteration
                except StopIteration:
                    raise
                except Exception:  # noqa: BLE001
                    pass
        if it % 25 == 0 and time.time() > deadline:
            box["cut"] = "wall"
            box["x"] = np.asarray(z).copy()
            raise StopIteration
        if it >= cap:
            box["cut"] = "cap"
            box["x"] = np.asarray(z).copy()
            raise StopIteration

    message = ""
    by_tol = False
    while True:
        left = cap - box["it"]
        if left <= 0:
            message = "CAP"
            break
        if opt == "LBFGSB":
            maxfun = int(1.25 * cap) - (st["nfev"] + box["nfev"])
            if maxfun <= 0:
                message = "MAXFUN"
                break
            res = minimize(fun, x, jac=True, method="L-BFGS-B", callback=cb,
                           options=dict(maxiter=left, maxfun=maxfun, ftol=0.0, gtol=1e-7, maxcor=10))
        else:
            res = minimize(fun, x, jac=True, method="CG", callback=cb, options=dict(maxiter=left, gtol=1e-7, norm=np.inf))
        x = res.x
        message = str(res.message)
        if box["cut"] == "power":
            np.savez(work / f"s{box['it']:05d}.npz", m6=x.reshape(j.N, j.N, j.N, 6), x=j.x)
            st["saved"].append(dict(iteration=box["it"], file=f"s{box['it']:05d}.npz", major=False, keep=True))
            st["iteration"] = box["it"]
            save_state()
            log("power warning: state saved, exit 75")
            sys.exit(75)
        if box["cut"] == "wall":
            x = box["x"]
            message = "CUT BY THE WALL-TIME CAP"
            break
        if box["cut"] == "cap":
            x = box["x"]
            message = "CUT BY THE ITERATION CAP (the callback count)"
            break
        mu_ = message.upper()
        by_tol = ("PROJECTED GRADIENT" in mu_) if opt == "LBFGSB" else bool(res.success)
        if by_tol:
            break
        abnormal = ("ABNORMAL" in mu_) if opt == "LBFGSB" else ("PRECISION" in mu_)
        stalled = bool(st["restarts"]) and st["restarts"][-1]["iteration"] == box["it"]
        if abnormal and len(st["restarts"]) < 3 and not stalled:
            st["restarts"].append(dict(iteration=box["it"], message=message))
            log(f"restart {len(st['restarts'])} at it {box['it']}: {message}")
            continue
        break
    it = box["it"]
    st["nfev"] += box["nfev"]
    box["nfev"] = 0
    if st["stops"][-1]["iteration"] == it and st["stops"][-1]["kind"] != "start":
        st["stops"].pop()
        st["saved"] = [s for s in st["saved"] if s["iteration"] != it or s["file"].startswith("s0000") and it == 0]
    end = stop(x, it, "end")
    rec.update(iterations=it, evaluations=st["nfev"], message=message, ended_by_gtol=bool(by_tol), cut_by_wall=bool(box["cut"] == "wall"),
               ended_at_cap=bool(it >= cap), restarts=st["restarts"], resumes=st["resumes"], end=end, end_file="end.npz", stops=st["stops"],
               saved=st["saved"])
    # the earlier state of the second step
    cands = [s for s in st["saved"] if not s.get("deleted") and s["file"] != "end.npz" and (work / s["file"]).exists()]
    earlier = None
    for gap in (500, 100):
        c = [s for s in cands if s["iteration"] <= it - gap]
        if c:
            earlier = max(c, key=lambda q: q["iteration"])
            break
    if earlier is not None:
        ze = np.load(work / earlier["file"], allow_pickle=False)["m6"]
        er = route_read(j, ze.reshape(-1), earlier["iteration"], True)
        er.update(kind=f"saved route stop {earlier['iteration']}", file=earlier["file"])
        rec["earlier"] = er
    else:
        rec["earlier"] = None
    rec["converged_check"] = converged(j, rec, st["stops"], majors)
    rec["end_state"] = "CONVERGED" if rec["converged_check"]["converged"] else "CAPPED"
    return rec


def ring_list(r):
    if r.get("pair") is not None:
        p = r["pair"]
        if p["word"] == "TWO RINGS":
            return [q["RL"] for q in p["rings"]]
        return None
    return [r["RL"]] if r.get("RL") is not None else None


def converged(j, rec, stops, majors):
    e = rec["end"]
    out = dict(gate=e["gate"], gate_ok=bool(e["gate"] <= 1e-3), by_gtol=rec["ended_by_gtol"])
    if j.delta > 0:
        out["torque"] = e["torque"]
        out["torque_ok"] = bool(e["torque"] <= 1e-5)
    maj = [s for s in stops if s.get("major") and s["kind"] in ("stop", "end")]
    pair = None
    drop_RL = j.grid == "P" and e.get("pair", {}).get("word") in ("ONE OBJECT", "neither")
    if len(maj) >= 2:
        best = None
        for a in range(len(maj) - 1):
            p, q = maj[-2 - a], maj[-1]
            if q["iteration"] - p["iteration"] >= 5000:
                best = (p, q)
                break
        if best is not None:
            p, q = best
            dU = abs(q["U"] / p["U"] - 1)
            rp, rq = ring_list(p), ring_list(q)
            if drop_RL:
                dR = 0.0
            elif rp is None or rq is None or len(rp) != len(rq):
                dR = None
            else:
                dR = max(abs(u - v) for u, v in zip(rp, rq))
            pair = dict(iterations=[p["iteration"], q["iteration"]], dU_rel=dU, dRL=dR)
            if j.delta > 0:
                zp = np.load(j.out / j.name / p["file"], allow_pickle=False)["m6"] if (j.out / j.name / p["file"]).exists() else None
                zq = np.load(j.out / j.name / q["file"], allow_pickle=False)["m6"]
                pair["mu_free"] = None if zp is None else T.mu(zq, zp, j.free & ~j.tube)["mu"]
                ok = dU <= 1e-8 and dR is not None and dR <= 0.002 and pair["mu_free"] is not None and pair["mu_free"] <= 0.003
            else:
                ok = dU <= 1e-6 and dR is not None and dR <= 0.002
            pair["ok"] = bool(ok)
    out["stops_pair"] = pair
    second = bool(pair and pair["ok"])
    conv = out["gate_ok"] and (out["by_gtol"] or second) and (out.get("torque_ok", True))
    out["converged"] = bool(conv)
    return out


def k5_compare(j, rec, src):
    e, s = rec["end"], src["end"]
    out = dict(dU_rel=abs(e["U"] / s["U"] - 1))
    ok = out["dU_rel"] <= 1e-6
    if j.grid == "P":
        out["pair_word"] = [s["pair"]["word"], e["pair"]["word"]]
        ok &= e["pair"]["word"] == s["pair"]["word"]
        if s["pair"]["word"] == "TWO RINGS" and e["pair"]["word"] == "TWO RINGS":
            out["dRL"] = max(abs(a["RL"] - b["RL"]) for a, b in zip(e["pair"]["rings"], s["pair"]["rings"]))
            ok &= out["dRL"] <= 0.002
    else:
        if e.get("RL") is None or s.get("RL") is None:
            out["dRL"] = None
            ok = False
        else:
            out["dRL"] = abs(e["RL"] - s["RL"])
            ok &= out["dRL"] <= 0.002
        out["class"] = [s.get("cls"), e.get("cls")]
        out["degree"] = [s["degree"]["values"], e["degree"]["values"]]
        ok &= s.get("cls") == e.get("cls") and s["degree"]["values"] == e["degree"]["values"]
    if "turn" in e:
        out["sigma"] = [s["turn"]["sigma"], e["turn"]["sigma"]]
        out["chi_max"] = [s["turn"]["chi_max"], e["turn"]["chi_max"]]
        ok &= s["turn"]["sigma"] == e["turn"]["sigma"]
        for b in (0.05, np.pi / 2):
            ok &= (s["turn"]["chi_max"] <= b) == (e["turn"]["chi_max"] <= b)
    if j.delta > 0:
        zs = np.load(j.out / src["name"] / "end.npz", allow_pickle=False)["m6"]
        ze = np.load(j.out / j.name / "end.npz", allow_pickle=False)["m6"]
        out["mu_free"] = T.mu(ze, zs, j.free & ~j.tube)["mu"]
        ok &= out["mu_free"] <= 0.01
        if "minor" in e:
            a0, a1 = s["minor"]["alpha_s"], e["minor"]["alpha_s"]
            out["alpha_s"] = [a0, a1]
            for b in (0.1, np.pi):
                ok &= (abs(a0) <= b) == (abs(a1) <= b)
    out["passed"] = bool(ok)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", type=int)
    ap.add_argument("--k5", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--spec", type=int, help="build the start of a row and print its start rule (no descent)")
    ap.add_argument("--cap", type=int, default=None, help="smoke tests only")
    ap.add_argument("--supp", type=int, help="the supplement outside the frozen plan: the row with start-rule clause (iv) not gating")
    ap.add_argument("--supp-k5", type=int, help="K5 of a supplement row")
    a = ap.parse_args()
    if a.list:
        for r in ROWS:
            print(r, cname(r))
        return
    WORK.mkdir(parents=True, exist_ok=True)
    if a.spec:
        j = build(a.spec)
        print(json.dumps(dict(info=j.info, start_rule=j.start_rule), default=T._jsonable, indent=1))
        return
    supp = a.supp is not None or a.supp_k5 is not None
    row = a.row or a.k5 or a.supp or a.supp_k5
    k5 = a.k5 is not None or a.supp_k5 is not None
    base = f"s{row:02d}_{ROWS[row]['name']}" if supp else cname(row)
    name = ("k5_" if k5 else "") + base
    OUT = SUPP if supp else WORK
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / name
    work.mkdir(exist_ok=True)
    jf = OUT / f"{name}.json"
    if jf.exists() and json.loads(jf.read_text()).get("finished"):
        print("already finished")
        return
    logf = open(work / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {name} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    try:
        j = build(row, k5=k5)
        j.out = OUT
        j.name = name
    except RuntimeError as err:
        rec = dict(name=name, row=row, finished=True, end_state="NOT RUN", reason=str(err))
        jf.write_text(json.dumps(rec, indent=1) + "\n")
        log(f"NOT RUN: {err}")
        return
    guard = None
    try:
        sys.path.insert(0, os.path.expanduser("~/Library/Application Support/UPSSentinel"))
        import ups_guard
        ups_guard.register()
        guard = ups_guard
    except Exception as e:  # noqa: BLE001
        log(f"no power guard: {e}")
    rec = dict(name=name, row=row, supplement=bool(supp), chain=ROWS[row]["name"], grid=j.grid, delta=j.delta, rung=ROWS[row]["rung"], optimizer=ROWS[row].get("opt", "LBFGSB"),
               held=ROWS[row]["held"], noise_seed=j.noise_seed, info=j.info, start_rule=j.start_rule, finished=False)
    if j.ref6 is not None:
        rec.update(reference=j.ref_name, U_ref=j.U_ref, rings=[dict(c=r["c"].tolist(), nu=r["nu"].tolist(), RL=r["RL"]) for r in j.rings])
    if k5:
        src = json.loads((OUT / f"{base}.json").read_text())
        m6e = np.asarray(np.load(OUT / base / src["end_file"], allow_pickle=False)["m6"], float)
        m6_0 = add_noise(j, m6e)
        cap = 10000
        rec["k5_of"] = base
    else:
        if supp:
            sr = j.start_rule
            wreg = sr.get("W_regime")
            cl = j.collar
            split = T.minor_jump_split(j.start_clean, j.ref6, j.x, j.delta, j.rings, cl["c"], cl["R1"], cl["R2"])
            rec["supplement_clause_iv"] = split
            ok = bool(sr["clause_i"] and sr["clause_ii"] and (wreg is None or wreg["regime"]))
            if not ok:
                rec.update(finished=True, end_state="NOT RUN", reason="not run (supplement): a clause other than (iv) fails")
            elif split["new_in_collar"] > 0:
                rec.update(finished=True, end_state="NOT RUN", reason="not run (supplement): the start holds a minor jump link in the collar that the reference does not")
            if rec.get("end_state") == "NOT RUN":
                jf.write_text(json.dumps(rec, indent=1, default=T._jsonable) + "\n")
                log(rec["reason"])
                return
            log(f"supplement: clause (iv) not gating; split {json.dumps({k: v for k, v in split.items() if k in ('outside_collar_equal', 'new_in_collar')})} "
                f"start {split['start']['counts']} reference {split['reference']['counts']}")
        elif j.start_rule is not None and not j.start_rule["passes"]:
            rec.update(finished=True, end_state="NOT RUN", reason="not run: start rule")
            jf.write_text(json.dumps(rec, indent=1, default=T._jsonable) + "\n")
            log(f"NOT RUN: start rule {json.dumps(j.start_rule, default=T._jsonable)}")
            return
        m6_0 = add_noise(j, j.start_clean)
        cap = 30000
    if a.cap:
        cap = a.cap
    rec["cap"] = cap
    log(f"starts: {json.dumps({k: v for k, v in rec.items() if k != 'finished'}, default=T._jsonable)}")
    t0 = time.time()
    prev = json.loads((work / "wall.json").read_text())["seconds"] if (work / "wall.json").exists() else 0.0
    deadline = t0 + WALL_HOURS * 3600 - prev
    try:
        rec = run(j, m6_0, cap, work, rec, log, guard, deadline)
    finally:
        (work / "wall.json").write_text(json.dumps(dict(seconds=prev + time.time() - t0)))
    if k5:
        rec["k5"] = k5_compare(j, rec, src)
        log(f"K5 {'MET' if rec['k5']['passed'] else 'NOT MET'} {json.dumps(rec['k5'], default=T._jsonable)}")
    rec["finished"] = True
    jf.write_text(json.dumps(rec, indent=1, default=T._jsonable) + "\n")
    log(f"END it {rec['iterations']} {rec['message']} {rec['end_state']} gate {rec['end']['gate']:.3e} U {rec['end']['U']:.10f}")


if __name__ == "__main__":
    main()
