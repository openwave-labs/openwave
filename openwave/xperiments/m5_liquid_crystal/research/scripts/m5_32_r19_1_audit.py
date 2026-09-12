"""M5.32 R19-1 AUDIT: an independent recomputation of the static Newton
read on the R3 boost-dressed pair (the producer: m5_32_r19_1_pair.py, its
JSON m5_32_r19_1_pair.json, its saved end fields m5_32_r19_1/<tag>.npz).
Every density below is written from the definitions, not imported; the
producer's entrants module is imported for ONE cross-check line only.

EQUATIONS FIRST
---------------
Field M(x) real symmetric 4x4 per cell on a 32^3 lattice (L = 48, h = 1.5),
eta = diag(-1, 1, 1, 1), vacuum M_vac = diag(g, 1, 0.3, 0) at g = 8 and
g = 32. Jets A_i = d_i M (i = 1, 2, 3) on the certified sym stencil (two
branches fwd and bwd of weight 1/2; the stack's d1 and branches are used
for the jets, everything after them is this file's).
    <X, Y>_eta = tr(eta X eta Y^T) = sum_ab eta_a eta_b X_ab Y_ab
    P_ij = A_i eta A_j,  F_ij = P_ij - P_ji,  G_ij = P_ij + P_ji
    f_I1 = (1/2) sum_ij <F_ij, F_ij>_eta = sum_{i<j} <F_ij, F_ij>_eta
u the timelike unit eigenvector of M eta (u^T eta u = -1; this file's own
eigensolve: the eigenvector of M eta with negative eta-norm),
    Pi_u = -u u^T eta,  Pi_s = 1 - Pi_u
    A^t_i = Pi_u A_i Pi_s^T + Pi_s A_i Pi_u^T   (symmetric, since A_i is)
    P^tt_ij = A^t_i eta A^t_j,  (P^tt_ij)^T = P^tt_ji
    X^Gam_ij = F_ij + 2 (P^tt_ij)^T = F_ij + 2 P^tt_ji     (Gamma)
    f_Gam = (1/2) sum_ij <X^Gam_ij, X^Gam_ij>_eta   (all nine (i, j), the
            diagonal X_ii = 2 P^tt_ii included)
    X^Gam_tl = F - (P^tt - P^tt^T) + T_tl(Pi_s (P^tt + P^tt^T) Pi_s^T),
            T_tl(Y) = Y - (1/3) tr(eta Y) (eta + u u^T)  (Gamma, traceless)
    X^Bu = Pi_s F Pi_s^T + Pi_u G Pi_s^T + Pi_s G Pi_u^T + Pi_u G Pi_u^T
    X^GG = G
The sector pieces (this file's split, same contraction):
    F_t = Pi_u F Pi_s^T + Pi_s F Pi_u^T   (the time row of F in the frame u)
    I(F_t) = (1/2) sum_ij <F_t, F_t>,  rest = (1/2) sum_ij <F - F_t, F - F_t>
    X_s = P^tt + P^tt^T,  I(X_s) = (1/2) sum_ij <X_s, X_s>
    time-row identity: Pi_u X^Gam Pi_s^T + Pi_s X^Gam Pi_u^T = F_t, because
    Pi_s^T eta Pi_u = 0 makes P^tt block-diagonal in the frame u
Potential V4 = h^3 sum_cells w sum_{p=1..4} (tr((M eta)^p) - c_p)^2 with
c_p the vacuum traces (the registry's v4_density_np, cross-checked against
the stack's e_parts). Energies:
    E_X[M] = 4 h^3 sum_br wt sum_cells f_X + V4
    E_int(d) = E(pair) - 2 E(single);  dressing part = E_int(dr) - E_int(un)
    force: attraction iff E_int increases with d (F = -dE/dd < 0)
    fits A + B / d^p (p = 1, 3, 5) and A + B / d + C ln(d) / d, own lstsq
Pre-registered outcome rules: NEWTON_SIGN_REVERSED = attractive and best
exponent 1; ATTRACTIVE_SHORT_RANGE = attractive with best exponent 3 or 5;
CANDIDATE_REFUTED = repulsive, or no object.

CLAIMS AND LINES (each can fail):
 1  energies: my E_X on every saved end field vs the row's E (rel 1e-9),
    the pieces table's I1 and Gam (rel 1e-9), the g 32 E_int tables and the
    dressing parts (abs 1e-6 relative to |E_int|)
 2  attribution on the killed g 8 fields: (a) under I1, Gam, Gam_tl the
    time-row piece carries the dive (I(F_t) < 0, I(F_t) / I1 in [0.8, 1.2],
    rest in (0, 2000); the actual range is reported); (b) I(X_s) end > seed there; (c) the time-row
    identity on every saved field (1e-10 of max(|F|, |P^tt|), the scale of
    the two terms whose difference is tested); (d) on the Bu and GG
    kills the time-row piece is NOT the dive (|I(F_t)| < 1 percent of |E|,
    I(X_s) > 100 |I(F_t)|), the producer's own wording for those rows
 3  instrument: the certified dressed rows at g 8 all RUNAWAY (5 of 5), no
    kill at g 32 (18 of 18 rows at the budget); RUNAWAY rows show max |M_0i|
    end / seed > 3 on the saved field, OK rows < 3; the g 8 undressed
    E_int string 17.25 / 18.37 / 19.58 / 20.10 rising
 4  the Coulomb identity on the 10 undressed end fields: my f_Gam, f_Gam_tl,
    f_Bu vs my f_I1 (rel 1e-12)
 5  the g 32 read: Gam 5 of 5 at the budget, E_int decreasing (repulsive),
    own fits and best exponent, the label CANDIDATE_REFUTED (repulsive)
    follows; my I1 dressed rows equal R3's record (rel 1e-6); Gam's dressing
    part exceeds I1's at every d; the producer's "overlap reproduces E_int
    within 2 percent at every d" (max relative gap reported)
 6  convergence: the outer-window slope |E_int(18) - E_int(24)| exceeds the
    summed last-quarter drifts of the two pairs; a geometric-tail
    extrapolation of the quarter drops (reported, not a line); the g 32
    amplitude trends by my own ball read (held = ratio in [0.9, 1.1], 13 rows)
 7  pin shell equal to the seed on every saved field (exact); the eigenframe
    defined on every cell (my n_bad = 0), the g 32 min gap > 30; the g 32
    far-field spectrum of M eta within 0.1 of the vacuum spectrum (the
    spectrum is boost-invariant, so this reads the vacuum basin through the
    dressing); the same box for the single and the pair
Out: ../data/m5_32_r19_1_audit.json. Runtime printed at the end.
Single process, energy evaluations only (no relaxation, nothing at n 48).
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
IN_JSON = os.path.join(DATA, "m5_32_r19_1_pair.json")
R3_JSON = os.path.join(DATA, "m5_32_r3_pair.json")
NPZ = os.path.join(DATA, "m5_32_r19_1")
OUT_JSON = os.path.join(DATA, "m5_32_r19_1_audit.json")
T0 = time.time()


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


LAG = _load("m5_32_lagrangian", "m5_32_lagrangian.py")      # v4_density_np, default_params
B3 = LAG.B3                                                  # d1, branches, pin_shell, coords, base_cfg, e_parts
RB = _load("m5_32_r2_b_bounded", "m5_32_r2_b_bounded.py")    # tl_eig (cross-check of my frame only)
R3 = _load("m5_32_r3_ii_pair", "m5_32_r3_ii_pair.py")        # seed_field, centers_of
EN = _load("m5_32_r19_entrants", "m5_32_r19_entrants.py")    # ONE cross-check line

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
ETA_D = np.array([-1.0, 1.0, 1.0, 1.0])
I4 = np.eye(4)
DELTA = 0.3


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def cfg_of(n, L, g):
    return B3.base_cfg(s=-1.0, g=g, n=n, L=float(L), delta=DELTA)


# ================= my own frame, brackets and densities =================
def frame(M):
    """u (N, 4) with u^T eta u = -1: the eigenvector of M eta of negative eta-norm.
    Returns u, gap (min distance of its eigenvalue to the others), n_bad (cells
    where the count of negative-norm eigenvectors is not one or the spectrum
    is not real), and the sorted real spectrum (N, 4)."""
    Me = M @ ETA
    lam, V = np.linalg.eig(Me)
    scale = np.maximum(np.max(np.abs(lam), axis=-1), 1.0)
    real_ok = np.max(np.abs(lam.imag), axis=-1) <= 1e-8 * scale
    lam = lam.real
    V = V.real
    nrm = np.einsum("nak,a,nak->nk", V, ETA_D, V)          # eta-norm of each column
    neg = nrm < 0
    n_neg = neg.sum(axis=-1)
    ok = real_ok & (n_neg == 1)
    k = np.argmin(nrm, axis=-1)
    uk = np.take_along_axis(V, k[:, None, None], axis=-1)[:, :, 0]
    nk = np.take_along_axis(nrm, k[:, None], axis=-1)[:, 0]
    u = uk / np.sqrt(np.maximum(-nk, 1e-300))[:, None]
    lk = np.take_along_axis(lam, k[:, None], axis=-1)[:, 0]
    dist = np.abs(lam - lk[:, None])
    np.put_along_axis(dist, k[:, None], np.inf, axis=-1)
    gap = dist.min(axis=-1)
    return u, gap, int(np.sum(~ok)), np.sort(lam, axis=-1)


def ip(X, Y):
    """<X, Y>_eta = tr(eta X eta Y^T) per cell, X, Y (..., 4, 4)."""
    return np.trace(ETA @ X @ ETA @ np.swapaxes(Y, -1, -2), axis1=-2, axis2=-1)


def half_sum(X):
    """(1/2) sum_ij <X_ij, X_ij>_eta over a (3, 3, N, 4, 4) family."""
    return 0.5 * np.sum(ip(X, X), axis=(0, 1))


def bracket_family(A):
    """P_ij = A_i eta A_j for A (3, N, 4, 4) -> (3, 3, N, 4, 4)."""
    AE = A @ ETA
    return np.stack([np.stack([AE[i] @ A[j] for j in range(3)]) for i in range(3)])


def transpose_ij(P):
    return np.swapaxes(P, 0, 1)


def sandwich(Lm, X, Rm):
    """Lm X Rm^T on the internal indices for X (3, 3, N, 4, 4) or (3, N, 4, 4)."""
    return Lm @ X @ np.swapaxes(Rm, -1, -2)


def densities(A, u):
    """all densities per cell on one stencil branch: A (3, N, 4, 4), u (N, 4)."""
    N = A.shape[1]
    P = bracket_family(A)
    F = P - transpose_ij(P)
    G = P + transpose_ij(P)
    Pu = -np.einsum("na,nb->nab", u, u) @ ETA           # -u u^T eta
    Ps = I4[None] - Pu
    At = sandwich(Pu, A, Ps) + sandwich(Ps, A, Pu)
    Ptt = bracket_family(At)
    PttT = np.swapaxes(Ptt, -1, -2)                     # the internal transpose
    transpose_check = float(np.max(np.abs(PttT - transpose_ij(Ptt))))
    Xg = F + 2.0 * transpose_ij(Ptt)                    # F + 2 (P^tt)^T, via P^tt_ji
    Xs = Ptt + PttT
    Xa = Ptt - PttT
    Y = sandwich(Ps, Xs, Ps)
    trY = np.einsum("a,ijnaa->ijn", ETA_D, Y)
    gs = ETA[None] + np.einsum("na,nb->nab", u, u)
    Xtl = F - Xa + Y - (trY / 3.0)[..., None, None] * gs[None, None]
    Xb = sandwich(Ps, F, Ps) + sandwich(Pu, G, Ps) + sandwich(Ps, G, Pu) + sandwich(Pu, G, Pu)
    Ft = sandwich(Pu, F, Ps) + sandwich(Ps, F, Pu)
    TX = sandwich(Pu, Xg, Ps) + sandwich(Ps, Xg, Pu)
    out = {"I1": half_sum(F), "Gam": half_sum(Xg), "Gam_tl": half_sum(Xtl), "Bu": half_sum(Xb), "GG": half_sum(G),
           "I_Ft": half_sum(Ft), "I_rest": half_sum(F - Ft), "I_Xs": half_sum(Xs), "I_Xa": half_sum(Xa)}
    scal = {"timerow_identity_abs": float(np.max(np.abs(TX - Ft))), "F_max": float(np.max(np.abs(F))),
            "Ptt_max": float(np.max(np.abs(Ptt))), "Ptt_transpose_check_abs": transpose_check}
    return out, scal


def read_field(M, cfg, want_spec=False):
    """E_X for every object, the sector pieces, the frame numbers, V4."""
    h = cfg["h"]
    h3 = h ** 3
    n = cfg["n"]
    Mf = M.reshape(-1, 4, 4)
    u, gap, n_bad, spec = frame(Mf)
    acc = {}
    scal = {"timerow_identity_abs": 0.0, "F_max": 0.0, "Ptt_max": 0.0, "Ptt_transpose_check_abs": 0.0}
    for br, wt in B3.branches(cfg["stencil"]):
        A = np.stack([B3.d1(M, ax, h, br) for ax in range(3)]).reshape(3, -1, 4, 4)
        dens, sc = densities(A, u)
        for k, v in dens.items():
            acc[k] = acc.get(k, 0.0) + wt * float(np.sum(v))
        for k in scal:
            scal[k] = max(scal[k], sc[k])
    p = LAG.default_params(s=-1.0, g=cfg["g"])
    V4 = h3 * float(np.sum(LAG.v4_density_np(None, M, p)))
    V4_stack = float(B3.e_parts(M, cfg)[1])
    out = {"curv": {k: 4.0 * h3 * v for k, v in acc.items()}, "V4": V4, "V4_stack_rel": abs(V4 - V4_stack) / max(abs(V4), 1e-300)}
    out["E"] = {k: out["curv"][k] + V4 for k in ("I1", "Gam", "Gam_tl", "Bu", "GG")}
    out["frame"] = {"min_gap": float(gap.min()), "n_bad": n_bad, "max_abs_M0i": float(np.max(np.abs(M[..., 0, 1:])))}
    out["checks"] = scal
    if want_spec:
        out["spec"] = spec.reshape(n, n, n, 4)
    return out


def ball_max(M, cfg, kind, d, r=5.0):
    X, Y, Z = B3.coords(cfg["n"], cfg["h"])
    m0i = np.sqrt(np.sum(M[..., 0, 1:] ** 2, axis=-1))
    out = {}
    for lab, zc in zip(("top", "bot"), R3.centers_of(kind, d)):
        ball = np.sqrt(X * X + Y * Y + (Z - zc) ** 2) <= r
        out[lab] = float(np.max(m0i[ball]))
    return out


def far_spectrum_dev(spec, cfg, kind, d, g, r_far=8.0):
    """max deviation of the sorted spectrum of M eta from the vacuum spectrum
    sorted(-g, 1, 0.3, 0) on cells farther than r_far from every core."""
    X, Y, Z = B3.coords(cfg["n"], cfg["h"])
    far = np.ones(X.shape, dtype=bool)
    for zc in R3.centers_of(kind, d):
        far &= np.sqrt(X * X + Y * Y + (Z - zc) ** 2) > r_far
    vac = np.sort(np.array([-g, 1.0, DELTA, 0.0]))
    dev = np.max(np.abs(spec - vac[None, None, None, :]), axis=-1)
    return {"far_max_dev": float(dev[far].max()), "far_cells": int(far.sum()), "all_max_dev": float(dev.max()),
            "frac_cells_dev_gt_0.5": float(np.mean(dev > 0.5))}


def fit_pow(ds, es, pw):
    ds, es = np.asarray(ds, float), np.asarray(es, float)
    X = np.stack([np.ones_like(ds), ds ** (-pw)], axis=1)
    c, *_ = np.linalg.lstsq(X, es, rcond=None)
    pred = X @ c
    ss = float(np.sum((es - es.mean()) ** 2))
    return {"A": float(c[0]), "B": float(c[1]), "R2": float(1.0 - np.sum((es - pred) ** 2) / ss), "p": pw}


def fit_log(ds, es):
    ds, es = np.asarray(ds, float), np.asarray(es, float)
    X = np.stack([np.ones_like(ds), 1.0 / ds, np.log(ds) / ds], axis=1)
    c, *_ = np.linalg.lstsq(X, es, rcond=None)
    pred = X @ c
    ss = float(np.sum((es - es.mean()) ** 2))
    return {"A": float(c[0]), "B": float(c[1]), "C": float(c[2]), "R2": float(1.0 - np.sum((es - pred) ** 2) / ss)}


def sign_of(ds, es):
    o = np.argsort(ds)
    es = np.asarray(es)[o]
    dE = es[-1] - es[-2]
    return {"sign": "ATTRACTIVE" if dE > 0 else "REPULSIVE" if dE < 0 else "flat",
            "monotone_decreasing": bool(np.all(np.diff(es) < 0)), "monotone_increasing": bool(np.all(np.diff(es) > 0))}


def outcome(force, fits, no_object):
    if no_object:
        return "CANDIDATE_REFUTED (no object)"
    if force["sign"] == "REPULSIVE":
        return "CANDIDATE_REFUTED (repulsive)"
    if force["sign"] != "ATTRACTIVE":
        return "UNDECIDED (flat)"
    best = max((k for k in fits if k.startswith("pow")), key=lambda k: fits[k]["R2"])
    return "NEWTON_SIGN_REVERSED" if fits[best]["p"] == 1 else f"ATTRACTIVE_SHORT_RANGE (best exponent {fits[best]['p']})"


def geometric_tail(trace, n_acc, E0, nq=4):
    """the energy drops over the last nq quarters of the accepted steps; a
    geometric extrapolation of the tail beyond the budget (ratio from the
    last two quarters; nan when the drops do not decay)."""
    if not trace:
        return None
    acc = np.array([0.0] + [r["acc"] for r in trace], float)
    E = np.array([E0] + [r["E"] for r in trace], float)
    qs = np.linspace(0, n_acc, nq + 1)
    Eq = np.interp(qs, acc, E, left=np.nan)
    drops = -np.diff(Eq)
    out = {"quarter_drops": [float(x) for x in drops], "last_quarter_dE": float(-drops[-1]) if np.isfinite(drops[-1]) else None}
    if np.all(np.isfinite(drops[-2:])) and drops[-2] > 0 and 0 < drops[-1] < drops[-2]:
        r = drops[-1] / drops[-2]
        out["ratio"] = float(r)
        out["tail_extrapolated"] = float(drops[-1] * r / (1.0 - r))
    else:
        out["ratio"] = None
        out["tail_extrapolated"] = None
    return out


# ================= the audit =================
def main():
    J = json.load(open(IN_JSON))
    rows = {r["tag"]: r for r in J["rows"]}
    pieces = J.get("pieces", {})
    lines = {}
    A = {"rows": {}, "lines": {}}

    # ---- claim 1 and the per-field reads ----
    worst_E = worst_piece = worst_v4 = worst_tr = worst_ptt = 0.0
    for tag in sorted(rows):
        r = rows[tag]
        f = os.path.join(NPZ, f"{tag}.npz")
        if not os.path.exists(f):
            A["rows"][tag] = {"status": r.get("status"), "saved": False}
            continue
        cfg = cfg_of(r["n"], r["L"], r["g"])
        M = np.load(f)["M"]
        M0, _ = R3.seed_field(cfg, r["kind"], r["d"], r["scale"])
        end = read_field(M, cfg, want_spec=True)
        seed = read_field(M0, cfg)
        obj = r["obj"]
        rel_E = abs(end["E"][obj] - r["E"]) / max(abs(r["E"]), 1e-300)
        pc = pieces.get(tag, {}).get("end", {})
        rel_piece = max(abs(end["curv"]["I1"] - pc.get("I1", np.nan)) / max(abs(pc.get("I1", 1.0)), 1.0),
                        abs(end["curv"]["Gam"] - pc.get("Gam", np.nan)) / max(abs(pc.get("Gam", 1.0)), 1.0)) if pc else np.nan
        pin = B3.pin_shell(cfg["n"], cfg["h"], 1.6)
        pin_dev = float(np.max(np.abs(M[pin] - M0[pin])))
        m0i_seed = float(np.max(np.abs(M0[..., 0, 1:])))
        m0i_end = float(np.max(np.abs(M[..., 0, 1:])))
        spec = far_spectrum_dev(end.pop("spec"), cfg, r["kind"], r["d"], r["g"])
        amp = {"seed": ball_max(M0, cfg, r["kind"], r["d"]), "end": ball_max(M, cfg, r["kind"], r["d"])}
        amp["ratio"] = {k: amp["end"][k] / amp["seed"][k] if amp["seed"][k] > 0 else None for k in amp["end"]}
        des = r.get("descent", {})
        A["rows"][tag] = {"status": r.get("status"), "stop": des.get("stop"), "obj": obj, "kind": r["kind"], "d": r["d"], "g": r["g"],
                          "dressed": r["dressed"], "saved": True, "E_producer": r["E"], "E_mine": end["E"][obj], "rel_E": rel_E,
                          "E_all_mine": end["E"], "curv_end": end["curv"], "curv_seed": seed["curv"], "V4": end["V4"],
                          "V4_vs_stack_rel": end["V4_stack_rel"], "rel_pieces_I1_Gam": rel_piece, "frame": end["frame"],
                          "checks": end["checks"], "pin_shell_max_dev": pin_dev, "m0i_seed": m0i_seed, "m0i_end": m0i_end,
                          "m0i_ratio": m0i_end / max(m0i_seed, 1e-300) if m0i_seed > 0 else None, "spectrum": spec, "amp": amp,
                          "trace_tail": geometric_tail(des.get("trace", []), des.get("accepted", 0), des.get("E0", np.nan)), "shape": list(M.shape),
                          "box": {"n": r["n"], "L": r["L"], "h": cfg["h"]}}
        worst_E = max(worst_E, rel_E)
        worst_piece = max(worst_piece, rel_piece) if np.isfinite(rel_piece) else worst_piece
        worst_v4 = max(worst_v4, end["V4_stack_rel"])
        worst_tr = max(worst_tr, end["checks"]["timerow_identity_abs"] / max(end["checks"]["F_max"], end["checks"]["Ptt_max"], 1e-300))
        worst_ptt = max(worst_ptt, end["checks"]["Ptt_transpose_check_abs"])
        log(f"{tag:30s} {r.get('status'):8s} E prod {r['E']:16.4f} mine {end['E'][obj]:16.4f} rel {rel_E:.1e} pieces rel {rel_piece:.1e} "
            f"Ft {end['curv']['I_Ft']:12.1f} rest {end['curv']['I_rest']:9.1f} Xs {end['curv']['I_Xs']:12.1f} gap {end['frame']['min_gap']:.3f} "
            f"bad {end['frame']['n_bad']} pin {pin_dev:.1e} far {spec['far_max_dev']:.3f}")
    R = A["rows"]
    saved = {t: v for t, v in R.items() if v["saved"]}
    lines["L1a_E_total_matches_row_E_rel_1e-9"] = worst_E < 1e-9
    lines["L1b_pieces_I1_Gam_match_rel_1e-9"] = worst_piece < 1e-9
    A["claim1"] = {"n_saved_fields": len(saved), "worst_rel_E": worst_E, "worst_rel_pieces": worst_piece, "worst_V4_vs_stack_rel": worst_v4,
                   "worst_Ptt_transpose_check_abs": worst_ptt}

    # ---- E_int tables at g 32 (mine), dressing parts ----
    def eint_table(obj, dressed, g, n=32):
        pref = f"{obj}_{'dr' if dressed else 'un'}"
        single = saved.get(f"{pref}_single_d0_n{n}_g{g:g}")
        if single is None or single["status"] != "OK":
            return None
        tab = {}
        for t, v in saved.items():
            if t.startswith(pref + "_same_") and v["g"] == g and v["status"] == "OK":
                tab[v["d"]] = v["E_all_mine"][obj] - 2.0 * single["E_all_mine"][obj]
        return {"E_single": single["E_all_mine"][obj], "E_int": dict(sorted(tab.items()))}

    E32 = {k: eint_table(*k) for k in (("Gam", True, 32.0), ("Gam_tl", True, 32.0), ("I1", True, 32.0), ("I1", False, 32.0), ("I1", False, 8.0), ("Gam", True, 8.0))}
    res = J.get("results", {})
    worst_eint = 0.0
    cmp = {}
    for (obj, dr, g), tab in E32.items():
        if tab is None:
            continue
        key = f"{obj}_{'dr' if dr else 'un'}_n32_g{g:g}"
        prod = res.get(key, {}).get("rows", {})
        for d, e in tab["E_int"].items():
            pe = prod.get(f"d{d:g}", {}).get("E_int")
            if pe is not None:
                dev = abs(e - pe) / max(abs(pe), 1.0)
                worst_eint = max(worst_eint, dev)
                cmp[f"{key}_d{d:g}"] = {"mine": e, "producer": pe, "rel": dev}
    un32 = E32[("I1", False, 32.0)]["E_int"]
    dress = {}
    for obj in ("Gam", "Gam_tl", "I1"):
        tab = E32[(obj, True, 32.0)]
        if tab:
            dress[obj] = {d: e - un32[d] for d, e in tab["E_int"].items() if d in un32}
    for obj, dp in dress.items():
        prod = res.get(f"{obj}_dr_n32_g32", {}).get("dressing_part", {})
        for d, e in dp.items():
            pe = prod.get(f"d{d:g}")
            if pe is not None:
                worst_eint = max(worst_eint, abs(e - pe) / max(abs(pe), 1.0))
    lines["L1c_g32_Eint_and_dressing_parts_match_rel_1e-6"] = worst_eint < 1e-6
    A["claim1"]["E_int_mine"] = {f"{o}_{'dr' if dr else 'un'}_g{g:g}": t for (o, dr, g), t in E32.items()}
    A["claim1"]["dressing_part_mine_g32"] = dress
    A["claim1"]["E_int_vs_producer"] = cmp
    A["claim1"]["worst_rel_E_int"] = worst_eint

    # ---- claim 2: attribution on the killed g 8 fields ----
    killed = {t: v for t, v in saved.items() if v["g"] == 8.0 and v["dressed"] and v["status"] != "OK"}
    tr_group = {t: v for t, v in killed.items() if v["obj"] in ("I1", "Gam", "Gam_tl")}
    corner_group = {t: v for t, v in killed.items() if v["obj"] in ("Bu", "GG")}
    c2 = {}
    ok_a = ok_b = ok_d = True
    for t, v in killed.items():
        c = v["curv_end"]
        Ft, I1, rest, Xs = c["I_Ft"], c["I1"], c["I_rest"], c["I_Xs"]
        rec = {"obj": v["obj"], "I1": I1, "I_Ft": Ft, "rest": rest, "I_Xs_end": Xs, "I_Xs_seed": v["curv_seed"]["I_Xs"], "E_end": v["E_mine"],
               "Ft_over_I1": Ft / I1 if I1 != 0 else None, "Ft_over_E": Ft / v["E_mine"]}
        if t in tr_group:
            a = (Ft < 0) and (0.8 <= Ft / I1 <= 1.2) and (0.0 < rest < 2000.0)
            b = Xs > v["curv_seed"]["I_Xs"]
            rec.update(dive_by_timerow=bool(a), Xs_rose=bool(b))
            ok_a &= a
            ok_b &= b
        else:
            dd = (abs(Ft) < 0.01 * abs(v["E_mine"])) and (Xs > 100.0 * abs(Ft))
            rec.update(timerow_not_the_dive=bool(dd))
            ok_d &= dd
        c2[t] = rec
    lines["L2a_killed_I1_Gam_Gamtl_dive_is_timerow_piece_Ft_over_I1_in_0.8_1.2_rest_lt_2000"] = ok_a and len(tr_group) == 12
    lines["L2b_killed_I1_Gam_Gamtl_IXs_rises"] = ok_b and len(tr_group) == 12
    lines["L2c_timerow_identity_X_vs_F_all_saved_fields_1e-10"] = worst_tr < 1e-10 and len(saved) == 44
    lines["L2d_Bu_GG_kills_timerow_not_the_dive"] = ok_d and len(corner_group) == 8
    A["claim2"] = {"killed": c2, "n_killed": len(killed), "n_timerow_group": len(tr_group), "n_corner_group": len(corner_group),
                   "rest_range_timerow_group": [min(x["rest"] for x in c2.values() if x["obj"] in ("I1", "Gam", "Gam_tl")),
                                                max(x["rest"] for x in c2.values() if x["obj"] in ("I1", "Gam", "Gam_tl"))],
                   "Ft_over_I1_range_timerow_group": [min(x["Ft_over_I1"] for x in c2.values() if x["obj"] in ("I1", "Gam", "Gam_tl")),
                                                      max(x["Ft_over_I1"] for x in c2.values() if x["obj"] in ("I1", "Gam", "Gam_tl"))],
                   "worst_timerow_identity_rel": worst_tr}

    # ---- claim 3: the instrument ----
    i1dr8 = [t for t in rows if t.startswith("I1_dr_") and rows[t]["g"] == 8.0]
    stops8 = {t: rows[t]["descent"]["stop"] for t in i1dr8}
    g32dr = [t for t in rows if rows[t]["g"] == 32.0]
    stops32 = {t: rows[t]["descent"]["stop"] for t in g32dr}
    lines["L3a_certified_dressed_g8_5of5_RUNAWAY"] = len(i1dr8) == 5 and all(s == "RUNAWAY" for s in stops8.values())
    lines["L3b_g32_no_kill_18of18_budget"] = len(g32dr) == 18 and all(s == "budget" for s in stops32.values())
    ok_ratio = True
    ratio_rec = {}
    for t, v in saved.items():
        if not v["dressed"]:
            continue
        rr = v["m0i_ratio"]
        ratio_rec[t] = {"stop": v["stop"], "ratio": rr, "E": v["E_mine"]}
        if v["stop"] == "RUNAWAY":
            ok_ratio &= rr > 3.0
        elif v["stop"].startswith("DIVERGED"):
            ok_ratio &= v["E_mine"] < -1e6
        elif v["stop"] == "budget":
            ok_ratio &= (rr < 3.0) and (v["E_mine"] > -1e6)
    lines["L3c_kill_rules_verified_on_saved_fields"] = ok_ratio
    un8 = E32[("I1", False, 8.0)]["E_int"]
    string = [17.25, 18.37, 19.58, 20.10]
    mine8 = [un8[d] for d in (12.0, 18.0, 24.0, 30.0)]
    lines["L3d_g8_undressed_string_17.25_18.37_19.58_20.10_rising"] = all(abs(a - b) < 0.01 for a, b in zip(mine8, string)) and all(np.diff(mine8) > 0)
    A["claim3"] = {"I1_dr_g8_stops": stops8, "g32_stops": stops32, "m0i_ratios": ratio_rec, "g8_undressed_E_int_mine": mine8}

    # ---- claim 4: the Coulomb identity ----
    worst_c = {"Gam": 0.0, "Gam_tl": 0.0, "Bu": 0.0}
    n_un = 0
    for t, v in saved.items():
        if v["dressed"]:
            continue
        n_un += 1
        for o in worst_c:
            worst_c[o] = max(worst_c[o], abs(v["E_all_mine"][o] - v["E_all_mine"]["I1"]) / max(abs(v["E_all_mine"]["I1"]), 1e-300))
    lines["L4_coulomb_identity_10_undressed_end_fields_rel_1e-12"] = n_un == 10 and max(worst_c.values()) < 1e-12
    A["claim4"] = {"n_undressed_fields": n_un, "max_rel_dev": worst_c}

    # ---- claim 5: the g 32 read ----
    gam = E32[("Gam", True, 32.0)]
    ds = sorted(gam["E_int"])
    es = [gam["E_int"][d] for d in ds]
    fits = {f"pow{p}": fit_pow(ds, es, p) for p in (1, 3, 5)}
    fits["log"] = fit_log(ds, es)
    best = max((k for k in fits if k.startswith("pow")), key=lambda k: fits[k]["R2"])
    force = sign_of(ds, es)
    gam_stops = [rows[t]["descent"]["stop"] for t in rows if t.startswith("Gam_dr_") and rows[t]["g"] == 32.0]
    label = outcome(force, fits, not (len(gam_stops) == 5 and all(s == "budget" for s in gam_stops)))
    prod_label = res.get("Gam_dr_n32_g32", {}).get("outcome")
    lines["L5a_Gam_g32_repulsive_label_CANDIDATE_REFUTED_repulsive_matches_producer"] = (
        label == "CANDIDATE_REFUTED (repulsive)" and label == prod_label and force["monotone_decreasing"] and ds == [10.0, 14.0, 18.0, 24.0])
    # R3's record
    R3J = json.load(open(R3_JSON))
    r3 = {r["tag"]: r for r in R3J["rows"] if r.get("lam") == 0.0 and r.get("n") == 32}
    r3_dr = {r["d"]: r["E"] - 2.0 * r3["lam0_dr1_single_d0_n32"]["E"] for r in r3.values() if r["kind"] == "same" and r.get("scale") == 1.0}
    r3_un = {r["d"]: r["E"] - 2.0 * r3["lam0_un0_single_d0_n32"]["E"] for r in r3.values() if r["kind"] == "same" and r.get("scale") == 0.0}
    r3_dress = {d: r3_dr[d] - r3_un[d] for d in r3_dr if d in r3_un}
    i1 = E32[("I1", True, 32.0)]
    worst_r3 = max(abs(i1["E_int"][d] - r3_dr[d]) / max(abs(r3_dr[d]), 1.0) for d in r3_dr)
    r3_force = sign_of(sorted(r3_dr), [r3_dr[d] for d in sorted(r3_dr)])
    same_cfg = RB.cfg_of(32, 48.0)
    cfg32 = cfg_of(32, 48.0, 32.0)
    cfg_same = all(same_cfg.get(k) == cfg32.get(k) for k in ("s", "g", "n", "L", "delta", "stencil", "h"))
    lines["L5b_I1_dressed_g32_equals_R3_record_rel_1e-6_and_R3_repulsive"] = worst_r3 < 1e-6 and r3_force["sign"] == "REPULSIVE" and cfg_same
    ratio_dress = {d: dress["Gam"][d] / dress["I1"][d] for d in dress["Gam"] if d in dress["I1"]}
    lines["L5c_Gam_dressing_part_exceeds_I1_dressing_part_every_d"] = all(v > 1.0 for v in ratio_dress.values()) and len(ratio_dress) == 4
    # the producer's "overlap reproduces E_int within 2 percent at every d"
    xs_single = saved["Gam_dr_single_d0_n32_g32"]["curv_end"]["I_Xs"]
    overlap = {}
    for t, v in saved.items():
        if t.startswith("Gam_dr_same_") and v["g"] == 32.0:
            overlap[v["d"]] = {"overlap_IXs": v["curv_end"]["I_Xs"] - 2.0 * xs_single, "E_int": gam["E_int"][v["d"]]}
            overlap[v["d"]]["rel_gap"] = abs(overlap[v["d"]]["overlap_IXs"] - gam["E_int"][v["d"]]) / abs(gam["E_int"][v["d"]])
    lines["L5d_producer_overlap_IXs_reproduces_Eint_within_2_percent_every_d"] = all(o["rel_gap"] <= 0.02 for o in overlap.values())
    A["claim5"] = {"Gam_g32": {"ds": ds, "E_int": es, "fits": fits, "best_exponent": fits[best]["p"], "force": force, "label_mine": label,
                               "label_producer": prod_label, "stops": gam_stops},
                   "R3_record": {"E_int_dr": r3_dr, "E_int_un": r3_un, "dressing_part": r3_dress, "force": r3_force, "worst_rel_vs_mine": worst_r3,
                                 "same_cfg_as_R3": cfg_same},
                   "dressing_ratio_Gam_over_I1": ratio_dress, "overlap_IXs_vs_E_int": overlap,
                   "max_rel_gap_overlap": max(o["rel_gap"] for o in overlap.values())}
    for o in ("Gam_tl", "I1"):
        tab = E32[(o, True, 32.0)]
        dd = sorted(tab["E_int"])
        A["claim5"][f"{o}_g32"] = {"ds": dd, "E_int": [tab["E_int"][d] for d in dd], "force": sign_of(dd, [tab["E_int"][d] for d in dd])}
        if len(dd) >= 3:
            ff = {f"pow{p}": fit_pow(dd, [tab["E_int"][d] for d in dd], p) for p in (1, 3, 5)}
            ff["log"] = fit_log(dd, [tab["E_int"][d] for d in dd])
            A["claim5"][f"{o}_g32"]["fits"] = ff

    # ---- claim 6: convergence and amplitude ----
    lq = {t: rows[t]["descent"].get("last_quarter_dE") for t in rows if t.startswith("Gam_dr_") and rows[t]["g"] == 32.0}
    single_lq = lq["Gam_dr_single_d0_n32_g32"]
    bound_value = {d: abs(lq[f"Gam_dr_same_d{d:g}_n32_g32"]) + 2.0 * abs(single_lq) for d in ds}
    slope_18_24 = gam["E_int"][18.0] - gam["E_int"][24.0]
    bound_slope = abs(lq["Gam_dr_same_d18_n32_g32"]) + abs(lq["Gam_dr_same_d24_n32_g32"])
    lines["L6a_outer_slope_exceeds_summed_last_quarter_drifts"] = slope_18_24 > bound_slope
    tails = {t: saved[t]["trace_tail"] for t in saved if t.startswith("Gam_dr_") and saved[t]["g"] == 32.0}
    tail_eint = {}
    ts = tails["Gam_dr_single_d0_n32_g32"]["tail_extrapolated"]
    for d in ds:
        tp = tails[f"Gam_dr_same_d{d:g}_n32_g32"]["tail_extrapolated"]
        tail_eint[d] = None if (tp is None or ts is None) else -(tp - 2.0 * ts)
    amp_ok = True
    amp_rec = {}
    for t, v in saved.items():
        if v["g"] == 32.0 and v["dressed"]:
            rat = [x for x in v["amp"]["ratio"].values() if x is not None]
            held = all(0.9 <= x <= 1.1 for x in rat)
            amp_rec[t] = {"ratio": v["amp"]["ratio"], "held": held, "producer": rows[t].get("amp_trend")}
            amp_ok &= held and all(rows[t]["amp_trend"].get(k) == "held" for k in ("top", "bot") if k in rows[t]["amp_trend"])
    lines["L6b_g32_dressing_held_by_own_ball_read_all_13_rows"] = amp_ok and len(amp_rec) == 13
    A["claim6"] = {"last_quarter_dE": lq, "E_int_drift_bound_value": bound_value, "E_int_values": gam["E_int"],
                   "value_sign_robust": {d: gam["E_int"][d] > bound_value[d] for d in ds},
                   "outer_slope_18_24": slope_18_24, "slope_drift_bound": bound_slope, "geometric_tails": tails,
                   "E_int_tail_extrapolated": tail_eint, "E_int_extrapolated": {d: gam["E_int"][d] + (tail_eint[d] or 0.0) for d in ds}, "amp": amp_rec}

    # ---- claim 7: pin shell, frame, basin, box ----
    pin_worst = max(v["pin_shell_max_dev"] for v in saved.values())
    lines["L7a_pin_shell_at_seed_values_exact_all_saved_fields"] = pin_worst == 0.0
    nbad = {t: v["frame"]["n_bad"] for t in saved for v in [saved[t]]}
    gap32 = min(v["frame"]["min_gap"] for v in saved.values() if v["g"] == 32.0)
    gap8 = {t: v["frame"]["min_gap"] for t, v in saved.items() if v["g"] == 8.0}
    lines["L7b_eigenframe_defined_everywhere_and_g32_gap_gt_30"] = max(nbad.values()) == 0 and gap32 > 30.0
    far32 = {t: v["spectrum"] for t, v in saved.items() if v["g"] == 32.0}
    far8 = {t: v["spectrum"] for t, v in saved.items() if v["g"] == 8.0}
    lines["L7c_g32_far_field_spectrum_within_0.1_of_vacuum"] = max(v["far_max_dev"] for v in far32.values()) < 0.1
    boxes = {(v["box"]["n"], v["box"]["L"], v["box"]["h"], tuple(v["shape"])) for v in saved.values()}
    lines["L7d_same_box_single_and_pair_32_48_h1.5"] = boxes == {(32, 48.0, 1.5, (32, 32, 32, 4, 4))}
    A["claim7"] = {"pin_shell_worst_dev": pin_worst, "n_bad": nbad, "min_gap_g32": gap32, "min_gap_g8": gap8,
                   "spectrum_g32": far32, "spectrum_g8": far8, "boxes": [list(map(str, b)) for b in boxes]}

    # ---- the ONE cross-check against the producer's entrants module ----
    xt = "Gam_dr_same_d24_n32_g32"
    cfg = cfg_of(32, 48.0, 32.0)
    M = np.load(os.path.join(NPZ, f"{xt}.npz"))["M"]
    xc = {}
    for o in ("I1", "Gam", "Gam_tl", "Bu", "GG"):
        br = EN.block_reads(M, cfg, o)
        xc[o] = abs(br["E_total"] - saved[xt]["E_all_mine"][o]) / abs(br["E_total"])
    lines["X_cross_check_entrants_module_5_objects_rel_1e-10"] = max(xc.values()) < 1e-10
    A["cross_check_entrants"] = {"field": xt, "rel": xc}

    A["lines"] = {k: bool(v) for k, v in lines.items()}
    A["runtime_s"] = round(time.time() - T0, 1)
    A["audited_utc"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    with open(OUT_JSON, "w") as f:
        json.dump(A, f, indent=1, default=float)
    print()
    for k, v in A["lines"].items():
        print(f"{'PASS' if v else 'FAIL'} {k}")
    print(f"\nGam g32 E_int {[round(e, 1) for e in es]} fits R2 {{{', '.join(f'{k}: {v['R2']:.4f}' for k, v in fits.items())}}} best {fits[best]['p']} label {label}")
    print(f"dressing ratio Gam/I1 {{{', '.join(f'{d:g}: {v:.2f}' for d, v in ratio_dress.items())}}}")
    print(f"overlap gap max {A['claim5']['max_rel_gap_overlap'] * 100:.2f} percent; drift bounds {bound_value}; slope 18-24 {slope_18_24:.0f} vs {bound_slope:.0f}")
    print(f"tails {tail_eint}")
    print(f"runtime {A['runtime_s']} s")


if __name__ == "__main__":
    main()
