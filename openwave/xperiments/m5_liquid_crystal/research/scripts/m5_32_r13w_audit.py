"""M5.32 R13-W: INDEPENDENT ADVERSARIAL AUDIT of steps W0, W1, W2 (section `w012`).

Every substantive claim of m5_32_r13w_w0.py / _w1.py / _w2.py is re-derived here
with its OWN code: own eta algebra, own stencil bookkeeping, own lattice
constructions from the certified primitives (m5_21_3_a_4d.py: d1, e_parts, grad,
kin_of, vac4, base_cfg) and own optimizers (scipy).  The producer's helpers in
m5_32_r13w_common.py are imported ONLY where they are themselves the object
under test (section J).  Every line printed is PASS/FAIL and can fail.

Claim groups (one function each, `audit_<letter>`):
  A  W0 S1 wall identity and the iff; the frame caveat
  B  W0 S2 flank inertia density, ramp, lattice one-cell accounting
  C  W0 S3 planar flatness, orientation walls, V4_deg and its constrained minimum
  D  W0 S4 phase flatness (continuum AND lattice), the non-uniform decomposition
  E  W0 S5 free inertia and the "no fixed-J minimizer" interpretation
  F  W0 S6 bag closure algebra, the numbers, ramp dominance, box feasibility
  G  W1 sigma_0 ladder, convergence state, own constrained wall, release stability
  H  W2 slab: E_stat(Delta q), kin_wall accounting, masks, registry, controls
  I  the pre-registered gates as carried by the scripts; unfalsifiable passes
  J  m5_32_r13w_common.py defects that reach W3

CLI:  python3 m5_32_r13w_audit.py w012          (this audit; writes key "w012")
      python3 m5_32_r13w_audit.py w012 A B ...  (subset of groups)
      python3 m5_32_r13w_audit.py w012_recheck  (re-check of the applied corrections; key "w012_recheck")
      python3 m5_32_r13w_audit.py w3            (W3 audit, key "w3": groups A to J of the W3 brief)
      python3 m5_32_r13w_audit.py w3_ctrl       (the same-seed-maturity L control, 28 min; key "w3_ctrl")
Out:  ../data/m5_32_r13w_audit.json  (merged by top-level key)
Run:  /opt/anaconda3/envs/master312/bin/python3 m5_32_r13w_audit.py w012
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
DATA = os.path.join(RES, "data")
CK = os.path.join(RES, "checkpoints", "m5_32_r13w")
OUT = os.path.join(DATA, "m5_32_r13w_audit.json")
T0 = time.time()


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INS4 = _load("m5_21_3_a_4d", "m5_21_3_a_4d.py")           # the certified stack (primitives only)
PY = sys.executable

# ---------------------------------------------------------------- own conventions
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
G, DELTA, S = 8.0, 0.3, -1.0
SG = S * G                                                  # -8: vac = diag(-sg, 1, delta, 0) = diag(8, 1, 0.3, 0)
VAC = np.diag([-SG, 1.0, DELTA, 0.0])
W1 = INS4.W1
CP = tuple(SG ** p + 1.0 + DELTA ** p for p in range(1, 5))


def E_(a, b):
    X = np.zeros((4, 4)); X[a, b] = 1.0; return X


G1 = E_(3, 2) - E_(2, 3)      # (2,3)-plane rotation generator, G1[2,3] = -1, G1[3,2] = +1
G3 = E_(2, 1) - E_(1, 2)      # (1,2)-plane rotation, G3[1,2] = -1, G3[2,1] = +1


def rot(Gm, q):
    q = np.asarray(q, float)
    return (np.eye(4) + np.sin(q)[..., None, None] * Gm + (1.0 - np.cos(q))[..., None, None] * (Gm @ Gm))


def comm(A, B):
    return A @ ETA @ B - B @ ETA @ A


def inner(F, Gq):
    """<F, G>_eta = tr(eta F eta G^T) = sum_ab eta_a eta_b F_ab G_ab (own einsum)."""
    e = np.diag(ETA)
    return np.einsum("...ab,...ab,a,b->...", F, Gq, e, e)


def v4_cell(M):
    Me = M @ ETA
    P = np.eye(4)
    v = 0.0
    for p in range(4):
        P = P @ Me
        v += (np.trace(P) - CP[p]) ** 2
    return W1 * v


def cfg_of(n, L):
    return INS4.base_cfg(s=S, g=G, n=n, L=float(L), delta=DELTA)


def jets(M, h, br):
    """own jets: fwd A[k] = (M[k+1]-M[k])/h (0 at the last cell); bwd A[k] = (M[k]-M[k-1])/h (0 at the first)."""
    A = []
    for ax in range(3):
        out = np.zeros_like(M)
        sl = [slice(None)] * 3
        if br == "fwd":
            a = list(sl); a[ax] = slice(0, -1); b = list(sl); b[ax] = slice(1, None)
            out[tuple(a)] = (M[tuple(b)] - M[tuple(a)]) / h
        else:
            a = list(sl); a[ax] = slice(1, None); b = list(sl); b[ax] = slice(0, -1)
            out[tuple(a)] = (M[tuple(a)] - M[tuple(b)]) / h
        A.append(out)
    return A


def my_e_u(M, h, st="sym"):
    brs = [("fwd", 0.5), ("bwd", 0.5)] if st == "sym" else [(st, 1.0)]
    e = 0.0
    for br, wt in brs:
        A = jets(M, h, br)
        for i in range(3):
            for j in range(i + 1, 3):
                F = comm(A[i], A[j])
                e += wt * 4.0 * np.sum(inner(F, F))
    return h ** 3 * e


def my_v4(M, h):
    Me = M @ ETA
    P = np.broadcast_to(np.eye(4), M.shape).copy()
    v = 0.0
    for p in range(4):
        P = P @ Me
        v = v + (np.einsum("...kk->...", P) - CP[p]) ** 2
    return h ** 3 * W1 * np.sum(v)


def my_kin_cells(M, a0, h, st="sym"):
    """per (branch, cell, direction) h^3-weighted kinetic contributions; returns dict and total."""
    brs = [("fwd", 0.5), ("bwd", 0.5)] if st == "sym" else [(st, 1.0)]
    table = {}
    tot = 0.0
    for br, wt in brs:
        A = jets(M, h, br)
        for i in range(3):
            F = comm(a0, A[i])
            dens = wt * 4.0 * inner(F, F) * h ** 3
            table[(br, i)] = dens
            tot += float(np.sum(dens))
    return table, tot


def a0_G1(M):
    return G1 @ M - M @ G1


def leading_axis_gen(M):
    """rotation generator about the local leading spatial eigenvector (own build)."""
    w, V = np.linalg.eigh(M[..., 1:, 1:])
    n1 = V[..., :, -1]
    J = np.zeros(M.shape)
    J[..., 1, 2], J[..., 2, 1] = -n1[..., 2], n1[..., 2]
    J[..., 1, 3], J[..., 3, 1] = n1[..., 1], -n1[..., 1]
    J[..., 2, 3], J[..., 3, 2] = -n1[..., 0], n1[..., 0]
    return J


RESULTS, LINES = {}, []


def check(sec, name, ok, detail):
    ok = bool(ok)
    RESULTS.setdefault(sec, {})[name] = {"pass": ok, "detail": detail}
    LINES.append(f"{'PASS' if ok else 'FAIL'} [{sec}] {name}: {detail}")
    print(LINES[-1], flush=True)
    return ok


def note(sec, key, val):
    RESULTS.setdefault(sec, {})[key] = val


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def slab(n, L, wall_cell=None, kz=None, nxy=1):
    cfg = cfg_of(n, L)
    M = np.broadcast_to(VAC, (nxy, nxy, n, 4, 4)).copy()
    kz = n // 2 if kz is None else kz
    if wall_cell is not None:
        M[:, :, kz] = wall_cell
    return M, cfg, kz


def deg_min(with_m01=False):
    """the exact constrained minimum of V4 on the degenerate manifold (a per-cell problem):
    lambda(M eta) = (-d0, d1, m, m) [+ M01 coupling]; multistart Nelder-Mead + BFGS."""
    def build(th):
        Mc = np.diag([th[0], th[1], th[2], th[2]])
        if with_m01:
            Mc[0, 1] = Mc[1, 0] = th[3]
        return Mc

    def f(th):
        return v4_cell(build(th))
    best = None
    rng = np.random.default_rng(5)
    starts = [np.array([8.0, 1.0, 0.15] + ([0.0] if with_m01 else []))]
    for _ in range(12):
        starts.append(starts[0] + rng.normal(scale=[0.3, 0.3, 0.1] + ([0.3] if with_m01 else [])))
    for s0 in starts:
        r = minimize(f, s0, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-22, "maxiter": 40000})
        r = minimize(f, r.x, method="BFGS", options={"gtol": 1e-16})
        if best is None or r.fun < best.fun:
            best = r
    return float(best.fun), build(best.x), best.x


# ================================================================ A: W0 S1
def audit_A():
    sec = "A"
    rng = np.random.default_rng(101)
    d = rng.normal(size=4)
    a0 = a0_G1(np.diag(d))
    target = (d[2] - d[3]) * (E_(2, 3) + E_(3, 2))
    check(sec, "S1 identity a0 = (d2 - d3)(E23 + E32) on a random diagonal M",
          np.allclose(a0, target, atol=1e-15), f"max |a0 - target| = {np.max(np.abs(a0 - target)):.2e}")
    # the iff: a0 = 0 exactly iff (block (2,3) = m I) and (M_2k = M_3k = 0 for k in 0,1)
    Msym = rng.normal(size=(4, 4)); Msym = 0.5 * (Msym + Msym.T)
    good = Msym.copy()
    good[2, 2] = good[3, 3]; good[2, 3] = good[3, 2] = 0.0
    for k in (0, 1):
        good[k, 2] = good[2, k] = 0.0; good[k, 3] = good[3, k] = 0.0
    ok = np.all(a0_G1(good) == 0.0)
    viol = []
    for (i, j) in ((0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        Mv = good.copy(); Mv[i, j] = Mv[j, i] = 0.37
        viol.append(np.max(np.abs(a0_G1(Mv))) > 1e-3)
    Mv = good.copy(); Mv[2, 2] += 0.37
    viol.append(np.max(np.abs(a0_G1(Mv))) > 1e-3)
    check(sec, "S1 iff: a0 = 0 exactly with (block = m I, decoupled); each of 6 single violations gives a0 != 0",
          ok and all(viol), f"a0(good) == 0: {ok}; violations detected: {viol}")
    # the frame caveat (not stated in W0): d2 = d3 does NOT imply a0_G1 = 0 unless the degenerate
    # eigenplane is the coordinate (2,3) plane; a0 about the local leading axis vanishes regardless
    psi = 0.7
    R = rot(G3, psi)
    Md = R @ np.diag([8.0, 1.0, 0.15, 0.15]) @ R.T
    w = np.linalg.eigvalsh(Md[1:, 1:])
    gap = w[1] - w[0]
    aG = np.max(np.abs(a0_G1(Md)))
    aL = np.max(np.abs(leading_axis_gen(Md[None]) [0] @ Md - Md @ leading_axis_gen(Md[None])[0]))
    check(sec, "S1 frame caveat: a degenerate cell (gap 0) rotated by 0.7 in the (1,2) plane has a0_G1 != 0 but a0_local = 0",
          gap < 1e-14 and aG > 1e-2 and aL < 1e-13,
          f"gap23 = {gap:.1e}, max|a0_G1| = {aG:.4f} (= (1-m) sin(2 psi)/... nonzero), max|a0_local| = {aL:.1e}")
    note(sec, "frame_caveat", {"psi": psi, "gap23": float(gap), "max_a0_G1": float(aG), "max_a0_local": float(aL)})


# ================================================================ B: W0 S2
def audit_B():
    sec = "B"
    z0 = 0.37
    d2 = lambda z: 0.3 * np.cos(z) + 0.1
    d3 = lambda z: 0.2 * np.sin(2 * z)
    d0 = lambda z: 8.0 + 0.5 * z
    d1 = lambda z: 1.0 - 0.2 * z * z
    eps = 1e-6
    M = np.diag([d0(z0), d1(z0), d2(z0), d3(z0)])
    Az = np.diag([(d0(z0 + eps) - d0(z0 - eps)) / (2 * eps), (d1(z0 + eps) - d1(z0 - eps)) / (2 * eps),
                  (d2(z0 + eps) - d2(z0 - eps)) / (2 * eps), (d3(z0 + eps) - d3(z0 - eps)) / (2 * eps)])
    F = comm(a0_G1(M), Az)
    dens = 4.0 * inner(F, F)
    gap, gapp = d2(z0) - d3(z0), Az[2, 2] - Az[3, 3]
    target = 8.0 * gap ** 2 * gapp ** 2
    Ft = gap * (Az[3, 3] - Az[2, 2]) * (E_(2, 3) - E_(3, 2))
    check(sec, "S2 F_0z = (d2 - d3)(d3' - d2')(E23 - E32) and density 4<F,F> = 8 (d2 - d3)^2 (d2' - d3')^2 (d0', d1' drop out)",
          np.allclose(F, Ft, atol=1e-9) and rel(dens, target) < 1e-8,
          f"density {dens:.10f} vs {target:.10f}; d0' = {Az[0,0]:.2f}, d1' = {Az[1,1]:.2f} present and irrelevant")
    # ramp integral: gap = delta (1 - z/w)
    w = 2.3
    zs = np.linspace(0, w, 200001)
    gapz = DELTA * (1 - zs / w); gp = -DELTA / w
    I = np.trapezoid(8 * gapz ** 2 * gp ** 2, zs)
    check(sec, "S2 linear ramp: kin/area = (8/3) delta^4 / w", rel(I, 8 / 3 * DELTA ** 4 / w) < 1e-8,
          f"quadrature {I:.8e} vs {8/3*DELTA**4/w:.8e}")
    # lattice: own per-(branch, cell, direction) accounting
    check(sec, "S2 stencil fact: INS4.branches('sym') is the fwd/bwd energy average (the central-difference d1 branch is never used by a_fields)",
          INS4.branches("sym") == [("fwd", 0.5), ("bwd", 0.5)], str(INS4.branches("sym")))
    outs = {}
    for label, cell in (("unrelaxed diag(8,1,.15,.15)", np.diag([8, 1, 0.15, 0.15])),
                        ("W1 snapshot diag(7.999997,1.0033,.1503,.1503)", np.diag([7.999996690850628, 1.0033089665632455, 0.15028979122698552, 0.15028979122698552])),
                        ("random degenerate diag(7.9,1.2,.1,.1)", np.diag([7.9, 1.2, 0.1, 0.1]))):
        for h in (1.5, 1.0, 0.75):
            n = int(round(48 / h))
            M, cfg, kz = slab(n, 48.0, cell)
            a0 = a0_G1(M)
            table, tot = my_kin_cells(M, a0, h)
            nz = {f"{br}/dir{i}/cell{k}": float(v[0, 0, k]) for (br, i), v in table.items() for k in range(n) if abs(v[0, 0, k]) > 0}
            outs[f"{label} h{h}"] = {"nonzero": nz, "total_per_area": tot / h ** 2}
        # check at h = 1.5 only (the others recorded)
    ref = outs["unrelaxed diag(8,1,.15,.15) h1.5"]["nonzero"]
    n = 32; kz = 16; h = 1.5
    exp_keys = {f"fwd/dir2/cell{kz-1}", f"bwd/dir2/cell{kz+1}"}
    per_flank = 4 * DELTA ** 4 / h * h ** 2                     # per column, area = h^2 for the 1x1 column
    ok = set(ref) == exp_keys and all(rel(v, per_flank) < 1e-12 for v in ref.values())
    check(sec, "S2 lattice one-cell layer: the ONLY kinetic cells are (fwd, z, kz-1) and (bwd, z, kz+1), weight 1/2 each, 4 delta^4/h per flank per area",
          ok, f"cells = {ref}; expected per flank per column = {per_flank:.6e}")
    same = all(rel(outs[k]["total_per_area"], 2 * 4 * DELTA ** 4 / float(k.split('h')[-1])) < 1e-10 for k in outs)
    check(sec, "S2 flank inertia is INDEPENDENT of the wall cell's (d0, d1, m): the same 2 x 4 delta^4/h for three different degenerate cells at three h",
          same, {k: round(v["total_per_area"], 8) for k, v in outs.items()})
    # own derivation: F = delta (dd3 - dd2)(E23 - E32)/h with dd3 - dd2 = delta for ANY m
    note(sec, "lattice_flank_table", outs)


# ================================================================ C: W0 S3
def audit_C():
    sec = "C"
    rng = np.random.default_rng(7)
    n = 12; cfg = cfg_of(n, 18.0); h = cfg["h"]
    prof = rng.normal(size=(n, 4, 4)) * 3.0; prof = 0.5 * (prof + prof.swapaxes(-1, -2))
    M = np.broadcast_to(prof[None, None], (n, n, n, 4, 4)).copy()
    eus = {}
    for st in ("fwd", "bwd", "sym"):
        eus[st] = (float(INS4.e_parts(M, cfg, st)[0]), float(my_e_u(M, h, st)))
    ok = all(a == 0.0 and b == 0.0 for a, b in eus.values())
    # negative control: x-dependence makes E_u > 0
    Mx = M.copy(); Mx[3:6] += 0.1 * rng.normal(size=(3, n, n, 4, 4)); Mx = 0.5 * (Mx + Mx.swapaxes(-1, -2))
    eux = float(INS4.e_parts(Mx, cfg)[0])
    check(sec, "S3 random planar profile (all 10 entries incl. time row): E_u == 0.0 exactly for fwd/bwd/sym, own F and e_parts; x-perturbed control E_u != 0 (negative here: the eta metric is indefinite on full 4x4 fields)",
          ok and abs(eux) > 1e-3, f"E_u = {eus}; control E_u = {eux:.3e}")
    # orientation walls: V4 = 0 for G1 and G3 rotations at random angles, and on the lattice slab jump
    oks = []
    for Gm, nm in ((G1, "(2,3)"), (G3, "(1,2)")):
        for q in rng.uniform(0, 2 * np.pi, 5):
            R = rot(Gm, q)
            oks.append(abs(v4_cell(R @ VAC @ R.T)) < 1e-24)
        Ms, cfgs, kz = slab(32, 48.0)
        R = rot(Gm, 1.1); Ms[:, :, kz:] = R @ VAC @ R.T
        eu, ev = INS4.e_parts(Ms, cfgs)
        oks.append(eu == 0.0 and abs(ev) < 1e-24)
    check(sec, "S3 orientation wall: V4 = 0 at random angles in BOTH the (2,3) and (1,2) planes; lattice jump has E_u = 0 and V4 = 0", all(oks), str(oks))
    # V4_deg recompute
    v4d = v4_cell(np.diag([8.0, 1.0, 0.15, 0.15]))
    their = 1.7994130394803167e-06
    check(sec, "S3 V4_deg for diag(8, 1, 0.15, 0.15) = 1.7994e-6 per volume", rel(v4d, their) < 1e-9, f"own {v4d:.10e} vs theirs {their:.10e}")
    # the constrained MINIMUM of V4 on the degenerate manifold (what a relaxed one-cell layer converges to)
    vmin, Mmin, th = deg_min(False)
    vmin2, Mmin2, th2 = deg_min(True)
    note(sec, "V4_deg_min", {"diag_only": {"V4": vmin, "d0": float(th[0]), "d1": float(th[1]), "m": float(th[2])},
                             "with_M01": {"V4": vmin2, "theta": [float(x) for x in th2]},
                             "V4_unrelaxed_layer": v4d, "ratio_unrelaxed_over_min": v4d / vmin})
    check(sec, "S3 QUALIFIER: the RELAXED one-cell layer's V4 is the constrained minimum on the degenerate manifold, strictly below 1.7994e-6",
          vmin < 0.9 * v4d and vmin > 1e-9 and abs(vmin2 - vmin) < 1e-3 * vmin,
          f"min V4 over diag(d0,d1,m,m) = {vmin:.6e} at d0 {th[0]:.6f} d1 {th[1]:.6f} m {th[2]:.6f} (ratio 1.7994e-6/min = {v4d/vmin:.3f}); "
          f"allowing M01: {vmin2:.6e} (no further gain); V4 > 0 strictly (power sums p = 1..4 fix the multiset, Newton identities)")


# ================================================================ D: W0 S4
def audit_D():
    sec = "D"
    rng = np.random.default_rng(11)
    # continuum with exact jets: phi(x,t) arbitrary smooth
    ks = rng.normal(size=(4, 4)); ph = rng.uniform(0, 6.28, 4); amp = rng.normal(size=4) * 0.5
    def phi(x):   # x = (t, x, y, z)
        return sum(amp[i] * np.sin(ks[i] @ x + ph[i]) for i in range(4))
    def dphi(x):
        return np.array([sum(amp[i] * np.cos(ks[i] @ x + ph[i]) * ks[i][mu] for i in range(4)) for mu in range(4)])
    worst = 0.0
    for _ in range(20):
        x = rng.normal(size=4)
        R = rot(G1, phi(x)); M = R @ VAC @ R.T
        X = a0_G1(M); dp = dphi(x)
        A = [dp[mu] * X for mu in range(4)]
        for a in range(4):
            for b in range(a + 1, 4):
                worst = max(worst, np.max(np.abs(comm(A[a], A[b]))))
    check(sec, "S4 continuum: uniform vacuum with an arbitrary phase phi(x, t), exact jets A_mu = d_mu phi [G1, M]: every F_mu nu = 0 (20 random points)",
          worst < 1e-13, f"max |F| = {worst:.1e}")
    # lattice: A_0 exact, A_i by the stencil: F != 0 at O(h^2) in density, vanishing as h -> 0
    def lattice_res(n, L):
        cfg = cfg_of(n, L); h = cfg["h"]
        X_, Y_, Z_ = INS4.coords(n, h)
        P = 0.6 * np.sin(0.25 * X_ + 0.1) * np.cos(0.2 * Y_) + 0.4 * np.sin(0.3 * Z_ - 0.4 * X_)
        Pt = 0.5 * np.cos(0.25 * X_) * np.sin(0.2 * Z_)                                  # d_t phi
        R = rot(G1, P); M = np.einsum("...ab,bc,...dc->...ad", R, VAC, R)
        A0 = Pt[..., None, None] * a0_G1(M)
        eu = float(INS4.e_parts(M, cfg)[0]); _, kin = my_kin_cells(M, A0, h)
        return eu / L ** 3, kin / L ** 3
    r1 = lattice_res(16, 24.0); r2 = lattice_res(32, 24.0); r3 = lattice_res(64, 24.0)
    pu = np.log(r1[0] / r2[0]) / np.log(2); pk = np.log(r1[1] / r2[1]) / np.log(2)
    pu2 = np.log(r2[0] / r3[0]) / np.log(2); pk2 = np.log(r2[1] / r3[1]) / np.log(2)
    check(sec, "S4 QUALIFIER: on the LATTICE the same phase field has E_u != 0 and F_0i != 0, both densities O(h^2) (exponent ~2 across h 1.5 -> 0.75 -> 0.375)",
          r1[0] > 1e-8 and 1.6 < pu2 < 2.4 and 1.6 < pk2 < 2.4,
          f"E_u/vol: {r1[0]:.3e}, {r2[0]:.3e}, {r3[0]:.3e} (exp {pu:.2f}, {pu2:.2f}); kin/vol: {r1[1]:.3e}, {r2[1]:.3e}, {r3[1]:.3e} (exp {pk:.2f}, {pk2:.2f})")
    # non-uniform background: F_0z = R (chi_t [X, d_z M0]) R^T with X = [G1, M0]
    B = rng.normal(size=(3, 4, 4)); B = 0.5 * (B + B.swapaxes(-1, -2))
    def M0(z):
        return B[0] + B[1] * z + B[2] * z * z
    def chi(z, t):
        return 0.4 * np.sin(1.3 * z + 0.2) + 0.7 * t * np.cos(0.5 * z) + 0.3 * t * t
    def Mfull(z, t):
        R = rot(G1, chi(z, t)); return R @ M0(z) @ R.T
    z0, t0, e = 0.31, 0.17, 1e-5
    At = (Mfull(z0, t0 + e) - Mfull(z0, t0 - e)) / (2 * e)
    Az = (Mfull(z0 + e, t0) - Mfull(z0 - e, t0)) / (2 * e)
    F = comm(At, Az)
    R = rot(G1, chi(z0, t0))
    chit = (chi(z0, t0 + e) - chi(z0, t0 - e)) / (2 * e)
    dM0 = (M0(z0 + e) - M0(z0 - e)) / (2 * e)
    X = a0_G1(M0(z0))
    Ft = R @ (chit * comm(X, dM0)) @ R.T
    scale = max(np.max(np.abs(F)), 1e-300)
    check(sec, "S4 non-uniform background M0(z) (random symmetric, full 4x4) with chi(z,t): F_0z = R chi_t [X, d_z M0] R^T, the chi_z chi_t [X,X] piece absent",
          np.max(np.abs(F - Ft)) / scale < 1e-6, f"rel err {np.max(np.abs(F - Ft)) / scale:.1e} (FD step 1e-5); |F| = {scale:.3f}")


# ================================================================ E: W0 S5
def audit_E():
    sec = "E"
    rng = np.random.default_rng(3)
    def f_claim(psi):
        return 8 * (DELTA - 1) ** 2 * (1 - (1 - DELTA ** 2) * np.cos(psi) ** 2)
    worst = 0.0
    for psi in rng.uniform(0, 2 * np.pi, 12):
        R = rot(G3, psi); M = R @ VAC @ R.T
        Az = G3 @ M - M @ G3                       # d/dpsi of R d R^T (psi' = 1)
        F = comm(a0_G1(M), Az)
        worst = max(worst, rel(4 * inner(F, F), f_claim(psi)))
    check(sec, "S5 f(psi) = 8 (delta - 1)^2 (1 - (1 - delta^2) cos^2 psi): own numeric density at 12 random angles", worst < 1e-12, f"max rel err {worst:.1e}")
    fmin = 8 * (DELTA - 1) ** 2 * DELTA ** 2
    fint = 8 * (DELTA - 1) ** 2 * (1 - (1 - DELTA ** 2) * (0.5 + np.sin(2.0) / 4))     # int_0^1 f dpsi
    check(sec, "S5 f > 0 everywhere: min f = 8 (1-delta)^2 delta^2 = 0.3528 at psi = 0, so any (1,2) twist carries inertia", abs(fmin - 0.3528) < 1e-12 and fmin > 0, f"min f = {fmin:.4f}")
    # lattice twist: E_stat = 0, kin ~ 1/w; own construction and own kin; continuum limit of kin*w = int_0^1 f dpsi
    rows = []
    for wc in (1, 2, 4, 8, 16):
        n = 64; cfg = cfg_of(n, 96.0); h = cfg["h"]; kz = n // 2
        ps = np.zeros(n); k0 = kz - wc // 2
        ps[k0:k0 + wc] = np.linspace(0, 1.0, wc + 1)[1:]; ps[k0 + wc:] = 1.0
        R = rot(G3, ps)
        M = np.einsum("zab,bc,zdc->zad", R, VAC, R)[None, None]
        eu = my_e_u(M, h); ev = my_v4(M, h); _, k = my_kin_cells(M, a0_G1(M), h)
        rows.append({"w_cells": wc, "w": wc * h, "E_u": eu, "V4": ev, "kin_per_area": k / h ** 2, "kin_x_w": k / h ** 2 * wc * h})
    ok = all(r["E_u"] == 0.0 and r["V4"] < 1e-24 for r in rows) and rel(rows[-1]["kin_x_w"], fint) < 0.01
    check(sec, "S5 lattice (1,2) twist over w cells: E_u == 0, V4 = 0, kin/area * w -> int_0^1 f dpsi = 1.3255 (own build, own kin)",
          ok, f"kin*w = {[round(r['kin_x_w'], 5) for r in rows]} -> {fint:.5f}; E_u = {[r['E_u'] for r in rows]}")
    note(sec, "twist_rows", rows)
    # INTERPRETATION: is kin unbounded at zero static cost?
    # (a) lattice at fixed h: per-cell inertia of a one-cell jump is bounded: scan the jump angle
    ks = []
    for q in np.linspace(0, np.pi, 181):
        M, cfg, kz = slab(8, 12.0); R = rot(G3, q); M[:, :, kz:] = R @ VAC @ R.T
        _, k = my_kin_cells(M, a0_G1(M), 1.5); ks.append(k / 1.5 ** 2)
    kmax = max(ks)
    check(sec, "S5 INTERPRETATION (lattice): the per-area inertia of a single (1,2) jump is bounded (max over the angle), so kin <= n_cells x const at fixed h: a lattice minimizer EXISTS",
          np.isfinite(kmax) and kmax > 0, f"max kin/area over jump angles at h = 1.5: {kmax:.4f} at q = {np.linspace(0, np.pi, 181)[int(np.argmax(ks))]:.3f}")
    # (b) 3D box with the pinned shell: a compact single-generator bump: E_stat = O(h^2) lattice residual, kin ~ 1/w; E_J at J = 200 vs the W0 shell picture
    n, L = 24, 36.0; cfg = cfg_of(n, L); h = cfg["h"]
    X_, Y_, Z_ = INS4.coords(n, h); r = np.sqrt(X_ ** 2 + Y_ ** 2 + Z_ ** 2)
    free = ~INS4.pin_shell(n, h)
    bumps = []
    for r0, Psi in ((12.0, 1.0), (9.0, 1.0), (6.0, 1.0), (14.0, 1.0), (14.0, 2.0), (14.0, 4.0), (14.0, 8.0)):
        ps = np.where(r < r0, (1 - (r / r0) ** 2) ** 2, 0.0) * Psi
        assert np.all(ps[~free] == 0.0)
        R = rot(G3, ps); M = np.einsum("...ab,bc,...dc->...ad", R, VAC, R)
        eu, ev = INS4.e_parts(M, cfg); _, k = my_kin_cells(M, a0_G1(M), h)
        EJ = float(eu + ev) + 200.0 ** 2 / (4 * k)
        bumps.append({"r0": r0, "Psi": Psi, "E_u": float(eu), "V4": float(ev), "kin": k, "E_J_J200": EJ})
    shells = {}
    for Rs in (9.0, 12.0):
        kin_shell = 4 * np.pi * Rs ** 2 * 4 * DELTA ** 4 / h
        shells[Rs] = {"kin": kin_shell, "E_J": 4 * np.pi * Rs ** 2 * h * 1.7994e-6 + 200.0 ** 2 / (4 * kin_shell)}
    best = min(bumps, key=lambda b: b["E_J_J200"])
    check(sec, "S5 INTERPRETATION (box + pinned shell, no core): a compact single-generator (1,2) bump has V4 = 0, E_u = lattice residual only, and E_J(J = 200) BELOW the W0 shell's E_J at R = 9 and R = 12 (surface + inertia, core energy not even counted): the shell is not the fixed-J minimizer",
          best["E_J_J200"] < min(sh["E_J"] for sh in shells.values()) and all(b["V4"] < 1e-18 for b in bumps),
          f"best bump r0 {best['r0']} Psi {best['Psi']}: E_u {best['E_u']:.3f}, kin {best['kin']:.1f}, E_J {best['E_J_J200']:.2f}; shells " + str({R: (round(v['kin'], 1), round(v['E_J'], 2)) for R, v in shells.items()}) + f"; all bumps {[(b['r0'], b['Psi'], round(b['E_u'], 3), round(b['kin'], 1), round(b['E_J_J200'], 1)) for b in bumps]}")
    note(sec, "bumps", bumps); note(sec, "shells", shells)
    # (c) sign: positivity holds in the block-diagonal sector only
    Mb = rng.normal(size=(6, 6, 6, 4, 4)); Mb = 0.5 * (Mb + Mb.swapaxes(-1, -2)); Mb[..., 0, 1:] = 0; Mb[..., 1:, 0] = 0
    Gs = rng.normal(size=(4, 4)); Gs = Gs - Gs.T; Gs[0, :] = 0; Gs[:, 0] = 0
    a0b = Gs @ Mb - Mb @ Gs
    _, kb = my_kin_cells(Mb, a0b, 1.0)
    # a boost clock (tangent K M + M K of L M L^T) over the (1,2) twist: F has only (0,k) entries, <F,F>_eta = -2 sum F_0k^2 < 0
    Mt = np.einsum("zab,bc,zdc->zad", rot(G3, np.linspace(0, 1, 8)), VAC, rot(G3, np.linspace(0, 1, 8)))[None, None]
    Kb = np.zeros((4, 4)); Kb[0, 1] = Kb[1, 0] = 1.0
    a0f = Kb @ Mt + Mt @ Kb
    _, kf = my_kin_cells(Mt, a0f, 1.0)
    check(sec, "S5 sign: kin >= 0 for a spatial-block generator on a block-diagonal field (F in the +eta block); a BOOST clock over the same twist gives kin < 0 (indefinite): positivity is a property of the spatial sector only",
          kb > 0 and kf < 0, f"kin(block-diag, spatial gen) = {kb:.3e}; kin((1,2) twist, boost clock) = {kf:.3e}")


# ================================================================ F: W0 S6
def audit_F():
    sec = "F"
    kappa = 14.0
    c = 4 * DELTA ** 4
    V = 1.7994130394803167e-06
    def E(R, w, J, kap, cc, VV):
        return 4 * np.pi * R ** 2 * w * VV + J ** 2 / (4 * (kap * R + 4 * np.pi * R ** 2 * cc / w))
    rng = np.random.default_rng(9)
    ok = True
    for _ in range(50):
        R, w, J, kap, cc, VV = rng.uniform(0.5, 50), rng.uniform(0.1, 5), rng.uniform(1, 500), rng.uniform(0.1, 30), rng.uniform(0.001, 1), rng.uniform(1e-7, 1e-3)
        d = (E(R, w * 1.0001, J, kap, cc, VV) - E(R, w * 0.9999, J, kap, cc, VV)) / (0.0002 * w)
        ok &= d > 0
    check(sec, "S6 dE_J/dw > 0 at 50 random (R, w, J, kappa, c, V): the shell sharpens to the floor", ok, "numeric derivative positive in all cases")
    # ramp-only floor: R*^4 = J^2/(64 pi^2 c V), omega* = sqrt(V/c) h
    h = 1.5
    fits = []
    for J in (50.0, 200.0, 800.0):
        Er = lambda R: 4 * np.pi * R ** 2 * h * V + J ** 2 * h / (16 * np.pi * R ** 2 * c)
        r = minimize(lambda x: Er(x[0]), [50.0], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-18})
        Rs = (J ** 2 / (64 * np.pi ** 2 * c * V)) ** 0.25
        om = J / (2 * (4 * np.pi * Rs ** 2 * c / h))
        fits.append((J, float(r.x[0]), Rs, om, np.sqrt(V / c) * h))
    check(sec, "S6 ramp-only algebra: numeric argmin of E(R) matches R*^4 = J^2/(64 pi^2 c V); omega* = sqrt(V/c) h = 0.011179 at h = 1.5, J-independent",
          all(rel(a, b) < 1e-5 and rel(o1, o2) < 1e-9 for _, a, b, o1, o2 in fits) and abs(fits[0][3] - 0.011178516844948605) < 1e-9,
          f"(J, argmin, formula, omega*, sqrt(V/c)h) = {[(J, round(a, 3), round(b, 3), round(o1, 6), round(o2, 6)) for J, a, b, o1, o2 in fits]}")
    their_R = {50.0: 90.77078764298044, 200.0: 181.54157528596087, 800.0: 363.08315057192175}
    check(sec, "S6 numbers R* = 90.77 / 181.5 / 363.1 at J = 50 / 200 / 800 (c = 0.0324, V = 1.7994e-6)",
          all(rel(fits[i][2], their_R[J]) < 1e-9 for i, J in enumerate((50.0, 200.0, 800.0))), str({J: round(v, 3) for J, v in their_R.items()}))
    # ramp dominance at R*: 4 pi R^2 c/h vs kappa R
    ratios = {J: (4 * np.pi * their_R[J] ** 2 * c / h) / (kappa * their_R[J]) for J in their_R}
    check(sec, "S6 QUALIFIER: the ramp-dominance assumption 4 pi R^2 c/h >> kappa R does NOT hold at the quoted R* for J = 50 (ratio 1.8) and is marginal at J = 200 (3.5)",
          ratios[50.0] < 2.0 and ratios[200.0] < 4.0, "ratios ramp/core at R*: " + str({J: round(v, 2) for J, v in ratios.items()}) + " with kappa = 14")
    # full bag model (kappa R + ramp): true R* and omega*; feasibility against the admissible boxes
    full = {}
    for J in (50.0, 200.0, 800.0):
        r = minimize(lambda x: E(x[0], h, J, kappa, c, V), [100.0], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-18})
        Rs = float(r.x[0]); kin = kappa * Rs + 4 * np.pi * Rs ** 2 * c / h
        full[J] = {"R_star_full": Rs, "omega_star_full": J / (2 * kin), "kin": kin, "surface_term": 4 * np.pi * Rs ** 2 * h * V, "inertia_term": J ** 2 / (4 * kin)}
    check(sec, "S6 QUALIFIER: with the core law kappa R = 14 R included the bag minimum SHIFTS: R* = 75 / 164 / 345 (ramp-only 91 / 182 / 363) and omega* = 0.0097 / 0.0104 / 0.0108 (ramp-only 0.0112, 4 to 14 percent high), no longer J-independent",
          all(full[J]["R_star_full"] < their_R[J] and full[J]["omega_star_full"] < 0.011178 for J in full) and full[50.0]["R_star_full"] < 80,
          {J: {k: round(v, 5) for k, v in d.items()} for J, d in full.items()})
    # J needed for R* inside the admissible boxes (half-widths 24, 36, 48 minus the pinned 1.6)
    def Jfor(Rt):
        f = lambda J: (minimize(lambda x: E(x[0], h, J, kappa, c, V), [Rt], method="Nelder-Mead").x[0] - Rt) ** 2
        return float(minimize(lambda J: f(J[0]), [2.0], method="Nelder-Mead", options={"xatol": 1e-6}).x[0])
    J12, J20 = Jfor(12.0), Jfor(20.0)
    check(sec, "S6 W3 FEASIBILITY: every quoted R* (91 to 363, or 75 to 345 with the core law) exceeds every admissible half-box (24 / 36 / 48); the bag model puts an interior R* = 12 at J ~ 2.6 and R* = 20 at J ~ 5.7, an order of magnitude below the W3 grid J = 50 to 220",
          min(their_R.values()) > 48 and min(f["R_star_full"] for f in full.values()) > 48 and J12 < 5 and J20 < 10,
          f"R*(J=50) = {their_R[50.0]:.1f} > 48; J for R* = 12: {J12:.2f}; J for R* = 20: {J20:.2f} (full model, kappa = 14, w = h = 1.5)")
    note(sec, "feasibility", {"J_for_Rstar_12": J12, "J_for_Rstar_20": J20, "full_model": full, "ramp_over_core_at_Rstar": ratios})
    # kappa on the R10 seed if present (read-only)
    seed = os.path.join(RES, "checkpoints", "m5_32_r10", "relax_g8_n32_L48_it12000.npy")
    if os.path.exists(seed):
        Ms = np.load(seed); cfg = cfg_of(32, 48.0); hh = cfg["h"]
        X_, Y_, Z_ = INS4.coords(32, hh); r = np.sqrt(X_ ** 2 + Y_ ** 2 + Z_ ** 2)
        Jn = leading_axis_gen(Ms); a0 = Jn @ Ms - Ms @ Jn
        ks = {}
        for Rc in (6.0, 9.0, 12.0, 15.0):
            _, k = my_kin_cells(Ms, a0 * (r < Rc)[..., None, None], hh); ks[Rc] = k
        slope = np.polyfit(list(ks), list(ks.values()), 1)[0]
        pexp = float(np.polyfit(np.log(list(ks)), np.log(list(ks.values())), 1)[0])
        check(sec, "S6 QUALIFIER kappa: on the R10 seed with the local clock the interior inertia cut at R is NOT linear over R = 6 to 15 (log-log exponent ~2.8, linear-fit slope 8.8): 'kin_in = 14 R' is imported from the R7 tail law, not measured on this seed at these radii",
              pexp > 1.5, "kin_in(R) = " + str({R: round(k, 2) for R, k in ks.items()}) + f"; exponent {pexp:.2f}; linear slope {slope:.2f}")
        note(sec, "kappa_seed", {"kin_in": ks, "slope": float(slope), "exponent": pexp})
    else:
        check(sec, "S6 kappa check on the R10 seed", False, "seed file not present, skipped")


# ================================================================ G: W1
def audit_G():
    sec = "G"
    w1 = json.load(open(os.path.join(DATA, "m5_32_r13w_w1.json"))) if os.path.exists(os.path.join(DATA, "m5_32_r13w_w1.json")) else {}
    logp = os.path.join(CK, "w1_v2.log")
    log = open(logp).read() if os.path.exists(logp) else ""
    if "summary" not in w1 and "SUMMARY" in log:            # the producer re-runs W1 while this audit reads: the log's SUMMARY line is the completed record
        w1["summary"] = json.loads(log.split("SUMMARY ", 1)[1].split("\n", 1)[0])
        w1.setdefault("release_stability", w1["summary"].get("release_stability", {}))
        note(sec, "source", "release_stability and summary parsed from w1_v2.log SUMMARY (JSON being rewritten by a concurrent producer run)")
    hl = w1.get("h_ladder", [])
    their = {r["h"]: r["constrained"]["sigma_0"] for r in hl}
    stops = {r["h"]: (r["constrained"]["stop"], r["constrained"]["trace"][-1]["fmax"] if r["constrained"].get("trace") else None) for r in hl}
    note(sec, "their_sigma_0", their); note(sec, "their_stops", stops)
    # the decoupling: E_u = 0 on planar profiles, so the constrained slab minimum is a per-cell V4 problem
    vmin, Mmin, th = deg_min(False)
    conv = {h: h * vmin for h in (1.5, 1.0, 0.75)}
    check(sec, "W1 all three constrained runs stopped at max_iter with fmax >> 1e-6: sigma_0 values are UNCONVERGED snapshots",
          len(hl) >= 3 and all(s[0] == "max_iter" and (s[1] is None or s[1] > 1e-5) for s in stops.values()), str(stops))
    # own constrained wall: L-BFGS on a (1,1,n) column with the wall cell parametrized diag(d0, d1, m, m) + M01, other free cells full symmetric
    own = {}
    for n, L in ((32, 48.0), (48, 48.0), (64, 48.0)):
        cfg = cfg_of(n, L); h = cfg["h"]; kz = n // 2
        wc = max(1, int(np.ceil(1.6 / h)))
        freec = np.arange(n); freec = freec[(freec >= wc) & (freec < n - wc) & (freec != kz)]
        iu = np.triu_indices(4)
        def unpack(th):
            M = np.broadcast_to(VAC, (1, 1, n, 4, 4)).copy()
            for j, k in enumerate(freec):
                S_ = np.zeros((4, 4)); S_[iu] = th[10 * j: 10 * j + 10]; S_ = S_ + S_.T - np.diag(np.diag(S_))
                M[0, 0, k] = S_
            t = th[10 * len(freec):]
            Mw = np.diag([t[0], t[1], t[2], t[2]]); Mw[0, 1] = Mw[1, 0] = t[3]
            M[0, 0, kz] = Mw
            return M
        def fun(th):
            M = unpack(th)
            e = INS4.e_total(M, cfg); Gd = INS4.grad(M, cfg)
            g = np.zeros_like(th)
            for j, k in enumerate(freec):
                Gc = Gd[0, 0, k]; fac = np.where(iu[0] == iu[1], 1.0, 2.0)
                g[10 * j: 10 * j + 10] = Gc[iu] * fac
            Gw = Gd[0, 0, kz]
            g[10 * len(freec):] = [Gw[0, 0], Gw[1, 1], Gw[2, 2] + Gw[3, 3], 2 * Gw[0, 1]]
            return float(e), g
        th0 = np.concatenate([np.tile(VAC[iu], len(freec)), [8.0, 1.0, 0.15, 0.0]])
        e_vac = INS4.e_total(np.broadcast_to(VAC, (1, 1, n, 4, 4)).copy(), cfg)
        r = minimize(fun, th0, jac=True, method="L-BFGS-B", options={"maxiter": 20000, "ftol": 1e-30, "gtol": 1e-16})
        Mend = unpack(r.x)
        sig = (r.fun - e_vac) / h ** 2
        eigs = np.linalg.eigvalsh(Mend[0, 0, kz, 1:, 1:])[::-1]
        neigh = float(np.max(np.abs(Mend[0, 0, kz - 1] - VAC)))
        own[h] = {"sigma_0_own": float(sig), "E_u": float(INS4.e_parts(Mend, cfg)[0]), "eigs_wall": eigs.tolist(), "M00": float(Mend[0, 0, kz, 0, 0]),
                  "M01": float(Mend[0, 0, kz, 0, 1]), "neighbor_max_dev_from_vac": neigh, "lbfgs_nit": int(r.nit), "success": bool(r.success)}
    ok = all(rel(own[h]["sigma_0_own"], conv[h]) < 1e-3 and own[h]["neighbor_max_dev_from_vac"] < 1e-6 for h in own)
    check(sec, "W1 own constrained wall (L-BFGS, wall parametrized, neighbors free): sigma_0 = h x V4_deg_min exactly, neighbors stay at the vacuum (the problem is per-cell)",
          ok, {h: (round(own[h]["sigma_0_own"] * 1e6, 5), round(conv[h] * 1e6, 5)) for h in own})
    note(sec, "own_wall", own); note(sec, "sigma_0_converged", conv)
    if their:
        ratio = {h: their[h] / conv[h] for h in their if h in conv}
        check(sec, "W1 QUALIFIER: the producer's sigma_0 (1.706e-6, 1.137e-6, 8.53e-7) sit ABOVE the converged h V4_deg_min by the same factor at every h (a 12000-iteration snapshot of a per-cell descent)",
              all(1.05 < v < 3.0 for v in ratio.values()) and (max(ratio.values()) - min(ratio.values())) < 1e-3,
              "their/converged = " + str({h: round(v, 4) for h, v in ratio.items()}) + "; converged sigma_0 = " + str({h: f"{v:.4e}" for h, v in conv.items()}))
        order_conv = float(np.polyfit(np.log([1.5, 1.0, 0.75]), np.log([conv[h] for h in (1.5, 1.0, 0.75)]), 1)[0])
        order_their = float(np.polyfit(np.log(list(their)), np.log(list(their.values())), 1)[0])
        check(sec, "W1 CONFIRMED: order-1 scaling of sigma_0 to zero holds for the snapshot AND for the converged wall (sigma_0 = h x const in both)",
              abs(order_their - 1) < 0.02 and abs(order_conv - 1) < 1e-9, f"order snapshot {order_their:.4f}, converged {order_conv:.6f}")
    # W0's number: h x 1.7994e-6 (unrelaxed layer) vs the converged relaxed layer
    check(sec, "W1 QUALIFIER: W0's 'sigma_0 = h V4_deg exactly' (V4_deg = 1.7994e-6) describes the UNRELAXED layer; the relaxed wall converges to h x V4_deg_min",
          rel(vmin, 1.7994e-6) > 0.2, f"V4_deg_min = {vmin:.6e} vs 1.7994e-6 (ratio {1.7994e-6 / vmin:.3f}); at h = 1.5: {1.5 * vmin:.4e} vs W0 2.6991e-6 vs W1 snapshot {their.get(1.5, float('nan')):.4e}")
    # release: (a) the free V4 gradient at a degenerate diagonal cell has no split component (symmetry)
    Mc = Mmin.copy()
    M, cfg, kz = slab(32, 48.0, Mc); Gd = INS4.grad(M, cfg)[0, 0, kz]
    split_comp = abs(Gd[2, 2] - Gd[3, 3]) + abs(Gd[2, 3])
    check(sec, "W1 release: the exact free gradient at the degenerate cell has ZERO split component (d2 = d3 is preserved by symmetry, not by stability)",
          split_comp < 1e-18 and np.max(np.abs(Gd)) > 1e-12, f"|G22 - G33| + |G23| = {split_comp:.1e}; |G|max = {np.max(np.abs(Gd)):.2e}")
    # (b) curvature of V4 along the split at the converged constrained minimum and at the producer's snapshot cell
    def curv_split(cell, s=1e-4):
        sp = np.diag([0.0, 0.0, 1.0, -1.0])
        return (v4_cell(cell + s * sp) - 2 * v4_cell(cell) + v4_cell(cell - s * sp)) / s ** 2
    def curv_analytic(cell):
        lam = np.diag(cell @ ETA); m = lam[2]
        t = [np.sum(lam ** p) for p in range(1, 5)]
        return W1 * 4 * sum((t[p - 1] - CP[p - 1]) * p * (p - 1) * m ** (p - 2) for p in range(1, 5))
    snap = np.diag([7.999996690850628, 1.0033089665632455, 0.15028979122698552, 0.15028979122698552])
    cm, ca, cs = curv_split(Mmin), curv_analytic(Mmin), curv_split(snap)
    theirs_curv = w1.get("release_stability", {}).get("V4_curvature_along_split")
    check(sec, "W1 release: V4 curvature along the eigenvalue split is NEGATIVE at the converged minimum and at the snapshot (a saddle: any split grows)",
          cm < 0 and cs < 0 and rel(cm, ca) < 1e-4,
          f"curvature at V4_deg_min: FD {cm:.4e}, analytic 4 W1 sum_p (t_p - C_p) p (p-1) m^(p-2) = {ca:.4e}; at the W1 snapshot cell {cs:.4e}; producer's value {theirs_curv}")
    # (c) the V4 Hessian of the cell: stiffness ratio; a 1D scan along the split (maximum at 0); a SCALED descent from the split runs to the vacuum
    x0 = np.diag(Mmin)
    def fcell(x):
        return v4_cell(np.diag(x))
    Hm = np.zeros((4, 4)); s_ = 1e-4
    for i in range(4):
        for j in range(4):
            ei = np.zeros(4); ei[i] = s_; ej = np.zeros(4); ej[j] = s_
            Hm[i, j] = (fcell(x0 + ei + ej) - fcell(x0 + ei - ej) - fcell(x0 - ei + ej) + fcell(x0 - ei - ej)) / (4 * s_ * s_)
    hw = np.linalg.eigvalsh(Hm)
    sp = np.diag([0.0, 0.0, 1.0, -1.0])
    scan = [v4_cell(Mmin + x * sp) for x in (-0.1, -0.05, 0.0, 0.05, 0.1)]
    sc = np.array([8.0, 1.0, 0.3, 0.3])
    r = minimize(lambda y: 1e6 * fcell(y * sc), (x0 + np.array([0, 0, 1e-3, -1e-3])) / sc, method="Nelder-Mead",
                 options={"xatol": 1e-12, "fatol": 1e-24, "maxiter": 40000})
    xe = r.x * sc; gap_end = abs(xe[2] - xe[3])
    check(sec, "W1 release: the constrained minimum is a SADDLE of V4: Hessian eigenvalues (-5.8e-5, 2.5e-3, 2.5e-2, 6.1e3), stiffness ratio 1e8; V4 has a local MAXIMUM along the split; a scaled descent from a 1e-3 split runs to the vacuum spectrum (gap 0.3, V4 -> 0)",
          hw[0] < 0 and hw[-1] / abs(hw[0]) > 1e7 and scan[2] > scan[1] and scan[2] > scan[3] and abs(gap_end - 0.3) < 1e-6 and r.fun * 1e-6 < 1e-20,
          f"Hessian eigs {hw.tolist()}; V4 along the split (s = -0.1..0.1) {[f'{v:.3e}' for v in scan]}; descent end {np.round(xe, 6).tolist()}, gap {gap_end:.6f}, V4 {r.fun * 1e-6:.1e}")
    note(sec, "hessian_eigs", hw.tolist())
    note(sec, "curvature", {"at_min_FD": cm, "at_min_analytic": ca, "at_snapshot": cs, "producer": theirs_curv})
    rs = w1.get("release_stability", {})
    if rs:
        tr = rs.get("gap_trace_after_perturbation") or []
        g_first = tr[0][1] if tr else None
        growth = (rs.get("gap_end", 0) / (2 * rs.get("eps", 1e-3)) - 1) * 100
        check(sec, "W1 QUALIFIED LABEL (v3 record): the split eps diag(+1,-1) starts at gap 2 eps = 2.0e-3, so 'the gap goes 0.001 -> 0.0020' misreports the start and the flag's clause 'gap_end > eps' is satisfied by the initial condition (cannot fail); the measured monotone growth is 0.2 percent (2.0000e-3 -> 2.0042e-3), the instability itself rests on the curvature sign",
              abs(rs.get("gap_end", 0) - 2 * rs.get("eps", 1e-3)) < 1e-4 and rs.get("V4_curvature_along_split", 0) < 0 and (g_first is None or abs(g_first - 2e-3) < 1e-5),
              f"gap_end {rs.get('gap_end')}, first logged gap {g_first}, growth {growth:.2f} percent; curvature {rs.get('V4_curvature_along_split')}; unstable flag {rs.get('unstable')}; eigs_end {rs.get('eigs_end')}")
    # the reason nothing moves: FIRE's dt collapsed on the stiff d0 mode; continue the producer's own dynamics from the split cell and read dt and the gap
    C = _load("m5_32_r13w_common", "m5_32_r13w_common.py")
    n = 32; cfg = cfg_of(n, 48.0); h = cfg["h"]; kz = n // 2
    Mcol = np.broadcast_to(VAC, (1, 1, n, 4, 4)).copy()
    Mcol[0, 0, kz] = np.diag([7.999997, 1.0048, 0.1504 + 1e-3, 0.1504 - 1e-3])
    free = np.ones((1, 1, n), bool); free[:, :, :2] = False; free[:, :, -2:] = False
    gaps, dts = [(0, 2e-3)], []
    Mrun = Mcol
    for chunk in range(3):
        Mrun, info = C.fire_proj(Mrun, cfg, free, 12000, project=None, tag="audit_split", log_every=12000, f_tol=1e-6, plateau=(2000, 0.0))
        gaps.append(((chunk + 1) * 12000, float(C.gap23(Mrun)[0, 0, kz]))); dts.append(info["trace"][-1]["dt"])
    their_dt = [t["dt"] for t in (hl[0]["released"]["trace"] if hl and "released" in hl[0] else [])]
    rate = 0.5 * abs(hw[0]) * h ** 3                      # gradient-flow growth rate of the split coordinate (|split|^2 = 2)
    check(sec, "W1 FIRE-RATE ARTIFACT: under the producer's own fire_proj the gap moves < 2 percent in 36000 iterations because dt sits at 1e-3 to 1e-2 (dt_max 0.1; the W1 released traces show the same), i.e. ~30 time units per 12000 iterations against a split growth rate of ~1e-4 per time unit: 12000 iterations cannot resolve the instability",
          max(dts) < 0.05 and abs(gaps[-1][1] / gaps[0][1] - 1) < 0.05 and (not their_dt or max(their_dt) < 0.05),
          f"(it, gap) {[(i, round(g, 6)) for i, g in gaps]}; dt at the chunk ends {[f'{d:.2e}' for d in dts]}; producer's released-run dt {[f'{d:.2e}' for d in their_dt]}; growth rate {rate:.2e}/time")
    note(sec, "split_growth", {"gaps": gaps, "dts": dts, "their_dt": their_dt, "rate": rate})
    # (d) the release_stability helper: log_every = 4000 with plateau = (2000, 1e-10) -> back = 0 -> 'plateau' at the first log line
    con = re.findall(r"w1_stab_con it\s+(\d+)", log); relr = re.findall(r"w1_stab_rel it\s+(\d+)", log)
    if con:
        check(sec, "W1 release_stability: its constrained and released FIRE runs (log_every = 4000) stop at the FIRST log line by the plateau defect (section J), not at 12000 or f_tol",
              con == ["4000"] and relr == ["4000"], f"logged iterations con {con}, rel {relr}")
    else:
        check(sec, "W1 release_stability log lines", False, "w1_v2.log has no w1_stab_con lines yet (run in progress); re-run the audit after the SUMMARY line")


# ================================================================ H: W2
def audit_H():
    sec = "H"
    w2 = json.load(open(os.path.join(DATA, "m5_32_r13w_w2.json")))
    # own slab: vacuum + converged degenerate cell, z > kz half rotated by dq (own Rodrigues), incl. the pinned end
    vmin, Mmin, _ = deg_min(False)
    rows = {}
    for n, L in ((32, 48.0), (48, 48.0), (64, 48.0), (48, 72.0), (64, 96.0)):
        cfg = cfg_of(n, L); h = cfg["h"]; kz = n // 2
        Es, klo, khi, kco = [], [], [], []
        for dq in np.linspace(0, 2 * np.pi, 13):
            M, _, _ = slab(n, L, Mmin, kz)
            R = rot(G1, dq); M[:, :, kz + 1:] = R @ M[:, :, kz + 1:] @ R.T
            eu = my_e_u(M, h); ev = my_v4(M, h); Es.append(eu + ev)
            a0 = a0_G1(M)
            zc = np.arange(n)
            lo = (zc < kz)[None, None, :, None, None]; hi = (zc > kz)[None, None, :, None, None]
            _, k1 = my_kin_cells(M, a0 * lo, h); _, k2 = my_kin_cells(M, a0 * hi, h); _, k3 = my_kin_cells(M, a0, h)
            klo.append(k1 / h ** 2); khi.append(k2 / h ** 2); kco.append(k3 / h ** 2)
            # is a0 at the wall cell zero, so that masking it is harmless?
        awall = np.max(np.abs(a0[0, 0, kz]))
        rows[(n, L)] = {"h": h, "E_range": float(max(Es) - min(Es)), "kin_lo": klo[0], "kin_hi": khi[0], "kin_co": kco[0],
                        "kin_lo_range": float(max(klo) - min(klo)), "kin_hi_range": float(max(khi) - min(khi)), "a0_wall_max": float(awall),
                        "prediction": 4 * DELTA ** 4 / h}
    ok = all(r["E_range"] == 0.0 and rel(r["kin_lo"], r["prediction"]) < 1e-12 and rel(r["kin_hi"], r["prediction"]) < 1e-12
             and rel(r["kin_co"], 2 * r["prediction"]) < 1e-12 and r["kin_hi_range"] < 1e-12 and r["a0_wall_max"] == 0.0 for r in rows.values())
    check(sec, "W2 own slab (converged wall cell, own rotation incl. the pinned end): E_stat(Delta q) range 0.0 exactly; kin lo = hi = 4 delta^4/h, co = 2x, dq-independent; a0 at the wall cell = 0 exactly",
          ok, {f"n{n}L{L:g}": (r["E_range"], round(r["kin_lo"], 6), round(r["kin_hi"], 6), round(r["kin_co"], 6)) for (n, L), r in rows.items()})
    theirs = {(r["n"], r["L"]): r["kin_wall_per_area"] for r in w2["h_ladder"] + w2["L_ladder"]}
    ok2 = all(rel(theirs[k]["lo_rotating"], rows[k]["kin_lo"]) < 1e-12 and rel(theirs[k]["co_rotating"], rows[k]["kin_co"]) < 1e-12 for k in rows)
    check(sec, "W2 their kin_wall/area (0.0216 / 0.0324 / 0.0432 one flank; 2x co-rotating; L-flat) reproduced to 1e-12", ok2, {f"n{n}L{L:g}": theirs[(n, L)]["lo_rotating"] for (n, L) in theirs})
    # rotate_half correctness: rotating the wall cell too changes nothing (it commutes), rotating from kz (incl. wall) vs kz+1 identical energies and kin
    M1, cfg, kz = slab(32, 48.0, Mmin); R = rot(G1, 0.9)
    Ma = M1.copy(); Ma[:, :, kz + 1:] = R @ Ma[:, :, kz + 1:] @ R.T
    Mb = M1.copy(); Mb[:, :, kz:] = R @ Mb[:, :, kz:] @ R.T
    check(sec, "W2 rotate_half: excluding or including the (diagonal, degenerate) wall cell in the rotation gives the identical field (R commutes with it) and identical pinned-end handling",
          np.max(np.abs(Ma - Mb)) == 0.0, f"max |M_a - M_b| = {np.max(np.abs(Ma - Mb)):.1e}; pinned end rotated to R d R^T (on the vacuum orbit, V4 = {v4_cell(R @ VAC @ R.T):.1e})")
    # registry convention: kin = -4 C
    LAG = _load("m5_32_lagrangian", "m5_32_lagrangian.py")
    p = LAG.default_params(s=S, g=G, delta=DELTA)
    M1, cfg, kz = slab(32, 48.0, Mmin, nxy=4)
    zc = np.arange(32); lo = (zc < kz)[None, None, :, None, None]
    a0 = a0_G1(M1) * lo
    _, kmine = my_kin_cells(M1, a0, 1.5)
    l0, Bc, Cc = LAG.omega_decompose(LAG.REGISTRY["I1"], M1, cfg, p, a0)
    check(sec, "W2 registry cross-check: I1(omega) = U - omega^2 T with eta^0 eta^i = -1, so kin = 4 T = -4 C; B (linear) = 0; own kin equals -4 C",
          rel(kmine, -4 * Cc) < 1e-12 and abs(Bc) < 1e-14 and rel(kmine / 36.0, 4 * DELTA ** 4 / 1.5) < 1e-12,
          f"own kin {kmine:.8f}, -4C {-4 * Cc:.8f}, B {Bc:.1e}")
    # controls: (2,3) jump in the vacuum slab: BOTH flanks carry inertia; the producer's mask drops the cell kz-1
    scan_mine, scan_theirs = [], {round(r["dq"], 6): r["kin_co_per_area"] for r in w2["controls"]["jump_23_vs_dq"]}
    for dq in np.linspace(0, 2 * np.pi, 25):
        M, cfg, kz = slab(32, 48.0); R = rot(G1, dq); M[:, :, kz:] = R @ VAC @ R.T       # jump between kz-1 and kz
        table, k = my_kin_cells(M, a0_G1(M), 1.5)
        cells = sorted({kk for (br, i), v in table.items() for kk in range(32) if v[0, 0, kk] > 1e-30})
        scan_mine.append((dq, k / 2.25, cells))
    law = 16 * DELTA ** 4 / 1.5 * 2                                         # both flanks: 2 x 16 delta^4 sin^4 / h
    ok_law = all(rel(k, law * np.sin(dq) ** 4) < 1e-9 for dq, k, _ in scan_mine if np.sin(dq) ** 4 > 1e-6)
    ok_cells = all(c == [kz - 1, kz] for dq, k, c in scan_mine if np.sin(dq) ** 4 > 1e-6)
    ratio = [k / scan_theirs[round(dq, 6)] for dq, k, _ in scan_mine if scan_theirs.get(round(dq, 6), 0) > 1e-6]
    check(sec, "W2 control (2,3) jump: own inertia = 32 delta^4 sin^4(dq)/h from BOTH flank cells (kz-1 fwd, kz bwd); the producer's two_freq_kin(M, kz-1) masks cell kz-1 out and reports HALF (max 0.0864 vs 0.1728)",
          ok_law and ok_cells and all(abs(rr - 2.0) < 1e-9 for rr in ratio),
          f"sin^4 law holds: {ok_law}; contributing cells {[c for _, _, c in scan_mine][6]}; own/theirs = {np.round(ratio[:3], 9).tolist()} at every angle; own max {max(k for _, k, _ in scan_mine):.4f}")
    # twist controls (no mask there): (2,3) twist kin*w -> 0 as 1/w^2; (1,2) twist -> 1.3255
    t23 = [(r["w_cells"], r["kin_per_area_times_w"]) for r in w2["controls"]["twist_23_vs_w"]]
    t12 = [(r["w_cells"], r["kin_per_area_times_w"]) for r in w2["controls"]["twist_12_vs_w"]]
    fint = 8 * (DELTA - 1) ** 2 * (1 - (1 - DELTA ** 2) * (0.5 + np.sin(2.0) / 4))
    r23 = [t23[i][1] / t23[i + 1][1] for i in range(len(t23) - 1)]
    check(sec, "W2 controls: (2,3) twist kin*w falls ~1/w^2 (ratios -> 4: lattice artifact, S4 continuum zero); (1,2) twist kin*w -> 1.3255 = int_0^1 f (1.3228 at w = 8 cells, 0.2 percent off)",
          r23[-1] > 3.5 and rel(t12[-1][1], fint) < 0.005, f"(2,3) ratios {np.round(r23, 3).tolist()}; (1,2) kin*w {t12}; analytic {fint:.5f}")
    note(sec, "own_rows", {f"n{n}L{L:g}": r for (n, L), r in rows.items()})


# ================================================================ I: gates
def audit_I():
    sec = "I"
    src1 = open(os.path.join(HERE, "m5_32_r13w_w1.py")).read()
    src2 = open(os.path.join(HERE, "m5_32_r13w_w2.py")).read()
    srcc = open(os.path.join(HERE, "m5_32_r13w_common.py")).read()
    m = re.search(r"def fire_proj\(.*?\):", srcc, re.S).group(0)
    ok_fire = "dt0=0.01" in m and "dt_max=0.1" in m and "f_tol=1e-6" in m
    ok_it = all("MAXIT, FTOL = 12000, 1e-6" in s for s in (src1, src2))
    ok_lad = all("((32, 48.0), (48, 48.0), (64, 48.0))" in s and "((48, 72.0), (64, 96.0))" in s for s in (src1, src2))
    check(sec, "gates verbatim: FIRE dt0 0.01, dt_max 0.1, 12000 iterations or fmax < 1e-6; h ladder (32,48),(48,48),(64,48) = h 1.5/1.0/0.75; L ladder (48,72),(64,96) at h 1.5",
          ok_fire and ok_it and ok_lad, f"fire_proj defaults {ok_fire}; MAXIT/FTOL {ok_it}; ladders {ok_lad}")
    ok_proj = "0.5 * (w[..., 0] + w[..., 1])" in srcc and "degenerate_project" in src1 and "degenerate_project" in src2
    check(sec, "projection as pre-registered: the (2,3) block of M(z=0) (the two smallest spatial eigenvalues) replaced by its trace part", ok_proj, "degenerate_project: mean of the two smallest eigenvalues in the local eigenframe")
    ok_v = "ESTABLISHED_KINEMATIC" in src1 and "W2 decides" in src1 and "ESTABLISHED_KINEMATIC" in src2 and "CANDIDATE_REFUTED" in src2 and "W3 does not run" in src2
    check(sec, "verdict vocabulary carried: W1 ESTABLISHED_KINEMATIC / 'W2 decides'; W2 ESTABLISHED_KINEMATIC / CANDIDATE_REFUTED / 'W3 does not run'", ok_v, "strings present")
    ok_reg = "omega_decompose" in srcc and "kin_registry" in src2
    check(sec, "two-frequency read: kin_of is the primary instrument, the pre-registered omega_decompose per half-space is the cross-check (rel dev 0)", ok_reg, "registry used as cross-check only")
    # deviations from the packet's admissible space: 4 x 4 x n slabs, x-y not pinned (uniform), not n^3 boxes
    check(sec, "DEVIATION (documented in the scripts): W1/W2 run on 4 x 4 x n slabs with the z ends pinned, not on n^3 boxes with a full pinned shell (admissible for a 1D profile and a planar slab; not the W3 geometry)",
          "NXY = 4" in src1 and "NXY = 4" in src2, "NXY = 4 in both scripts")
    # unfalsifiability: the W2 gates are identities on any diagonal-wall slab
    rng = np.random.default_rng(2)
    bad = np.diag([rng.uniform(5, 9), rng.uniform(0.5, 2), 0.05, 0.05])          # an arbitrary NON-relaxed degenerate cell
    ranges, kins = [], {}
    for n, L in ((32, 48.0), (48, 72.0), (64, 96.0)):
        cfg = cfg_of(n, L); h = cfg["h"]; kz = n // 2
        Es = []
        for dq in np.linspace(0, 2 * np.pi, 9):
            M, _, _ = slab(n, L, bad, kz); R = rot(G1, dq); M[:, :, kz + 1:] = R @ M[:, :, kz + 1:] @ R.T
            Es.append(my_e_u(M, h) + my_v4(M, h))
        ranges.append(max(Es) - min(Es))
        M, _, _ = slab(n, L, bad, kz); a0 = a0_G1(M) * (np.arange(n) < kz)[None, None, :, None, None]
        _, k = my_kin_cells(M, a0, h); kins[L] = k / h ** 2
    Lexp = float(np.polyfit(np.log(list(kins)), np.log(list(kins.values())), 1)[0])
    check(sec, "UNFALSIFIABLE: both W2 gates (E_stat range < 1e-9; |L-exponent| < 0.1) pass for an ARBITRARY unrelaxed degenerate cell diag(6.6, 1.2, 0.05, 0.05): they are identities of the planar slab, not tests of the wall",
          max(ranges) == 0.0 and abs(Lexp) < 1e-12, f"E_stat ranges {ranges}; kin/area vs L {kins}; L-exponent {Lexp:.1e}")
    # W1's verdict can fail both ways (order and finiteness) but its 'order' is itself an identity: sigma_0 = h x V4_cell for any one-cell layer
    check(sec, "W1 verdict logic can fail (order < 0.5 and sigma > 1e-12 both required) but 'order 1' is an identity of any one-cell planar layer (sigma_0 = h x V4_cell), so the W1 fail was decided at W0",
          "abs(order) < 0.5" in src1 and "abs(sg[-1]) > 1e-12" in src1, "verdict condition present; the mechanism is in section G")


# ================================================================ J: common.py
def audit_J():
    sec = "J"
    C = _load("m5_32_r13w_common", "m5_32_r13w_common.py")
    # (1) plateau defect: log_every >= plateau[0] -> back = 0 -> compares the row with itself -> 'plateau' at the first log line
    M, cfg, kz = slab(16, 24.0, nxy=1)
    wall = np.zeros((1, 1, 16), bool); wall[:, :, kz] = True
    free = np.ones((1, 1, 16), bool); free[:, :, :2] = False; free[:, :, -2:] = False
    proj = lambda X: C.degenerate_project(X, wall)
    Mb, ib = C.fire_proj(M, cfg, free, 5000, project=proj, tag="audit_le1000", log_every=1000)
    Mc_, ic = C.fire_proj(M, cfg, free, 5000, project=proj, tag="audit_le2500", log_every=2500)
    check(sec, "fire_proj plateau defect: with log_every > plateau[0] (2000), back = plateau[0] // log_every = 0 and the plateau test compares the row with ITSELF: the run stops 'plateau' at the FIRST log line while the energy is still falling (log_every 2500 -> stops at 2500; 1000 -> runs to 5000)",
          ic["stop"] == "plateau" and ic["iters"] == 2500 and ib["stop"] != "plateau" and ib["iters"] == 5000 and ib["trace"][-1]["E"] < ic["trace"][-1]["E"],
          f"log_every 2500: stop {ic['stop']} @ {ic['iters']} E {ic['trace'][-1]['E']:.6e}; log_every 1000: stop {ib['stop']} @ {ib['iters']} E {ib['trace'][-1]['E']:.6e}")
    check(sec, "W3 exposure of the plateau defect: m5_32_r13w_w3.py calls fire_proj with log_every = 250 (back = 8), so W3 is NOT hit; W1 release_stability (log_every 4000) IS hit",
          "log_every=250" in open(os.path.join(HERE, "m5_32_r13w_w3.py")).read() and "log_every=4000" in open(os.path.join(HERE, "m5_32_r13w_w1.py")).read(), "grep of the call sites")
    # (2) fixed-J gradient with a0 FROZEN: the formula grad E_stat - J^2/(4 kin^2) grad kin is the exact gradient of E_stat + J^2/(4 kin(M; a0 frozen)) (complex step)
    B8 = _load("m5_21_8_b_lattice", "m5_21_8_b_lattice.py")
    cfg = cfg_of(12, 18.0); Md = B8.dressed(cfg, 0.0)
    rng = np.random.default_rng(4)
    a0 = C.a0_local(Md); J = 30.0
    k = float(INS4.kin_of(Md, a0, cfg))
    Gt = INS4.grad(Md, cfg) - (J * J / (4 * k * k)) * INS4.kin_grad(Md, a0, cfg)
    errs_frozen, errs_refresh = [], []
    for _ in range(3):
        V = rng.normal(size=Md.shape); V = 0.5 * (V + V.swapaxes(-1, -2)); V[..., 0, 1:] = 0; V[..., 1:, 0] = 0
        an = float(np.sum(Gt * V))
        t = 1e-30
        Mz = Md + 1j * t * V
        ef = (INS4.e_total(Mz, cfg) + J * J / (4 * INS4.kin_of(Mz, a0, cfg))).imag / t
        errs_frozen.append(rel(ef, an))
        # a0 refreshed: real central FD (eigh is not complex-step safe)
        s = 1e-5
        def EJ(Mx):
            return INS4.e_total(Mx, cfg) + J * J / (4 * INS4.kin_of(Mx, C.a0_local(Mx), cfg))
        er = (EJ(Md + s * V) - EJ(Md - s * V)) / (2 * s)
        errs_refresh.append(rel(er, an))
    check(sec, "fire_proj fixed-J gradient (sign, factor J^2/(4 kin^2), a0 frozen) is the exact complex-step gradient of E_stat + J^2/(4 kin) at frozen a0",
          max(errs_frozen) < 1e-8, f"rel errs {[f'{e:.1e}' for e in errs_frozen]}")
    check(sec, "W3 CAVEAT: with a0 REFRESHED from M each step (a0_local(M)), the reported E_J is NOT the function whose gradient drives the descent: the neglected da0/dM term is O(1) on the hedgehog",
          max(errs_refresh) > 1e-2, f"rel mismatch between the frozen-a0 gradient and the FD gradient of E_J with refreshed a0: {[f'{e:.1e}' for e in errs_refresh]}")
    # (3) kin_density sums to kin_of
    a0r = a0_G1(Md)
    check(sec, "kin_density(M, a0).sum() == kin_of(M, a0)", rel(float(np.sum(C.kin_density(Md, a0r, cfg))), float(INS4.kin_of(Md, a0r, cfg))) < 1e-14, "consistent")
    # (4) a0_local vs G1: vacuum, G1-rotated vacuum (equal up to sign), G3-rotated vacuum (differ: co-rotating vs fixed-frame clock)
    Mv = np.broadcast_to(VAC, (2, 2, 2, 4, 4)).copy()
    Rq = rot(G1, 0.8); Mr1 = np.broadcast_to(Rq @ VAC @ Rq.T, (2, 2, 2, 4, 4)).copy()
    Rp = rot(G3, 0.8); Mr3 = np.broadcast_to(Rp @ VAC @ Rp.T, (2, 2, 2, 4, 4)).copy()
    def same_up_to_sign(a, b):
        return min(np.max(np.abs(a - b)), np.max(np.abs(a + b))) < 1e-14
    d1 = same_up_to_sign(C.a0_local(Mv), C.a0_G1(Mv)); d2 = same_up_to_sign(C.a0_local(Mr1), C.a0_G1(Mr1))
    d3 = not same_up_to_sign(C.a0_local(Mr3), C.a0_G1(Mr3))
    cor = np.einsum("ab,...bc,dc->...ad", Rp, C.a0_G1(Mv), Rp)
    d4 = same_up_to_sign(C.a0_local(Mr3), cor)
    check(sec, "a0_local == +/- [G1, M] on the vacuum and on a (2,3)-rotated vacuum; on a (1,2)-rotated vacuum it is the CO-ROTATED clock R [G1, d] R^T, not [G1, M]",
          d1 and d2 and d3 and d4, f"vac {d1}, G1-rotated {d2}, G3-rotated differs {d3}, equals co-rotated {d4}")
    # (5) degenerate_project: idempotent, gap 0, other eigenvalue and eigenvectors preserved, off-block time entries untouched (so a0 need not vanish there)
    Mx = rng.normal(size=(3, 3, 3, 4, 4)); Mx = 0.5 * (Mx + Mx.swapaxes(-1, -2))
    mk = np.zeros((3, 3, 3), bool); mk[1, 1, 1] = True
    P1 = C.degenerate_project(Mx, mk); P2 = C.degenerate_project(P1, mk)
    w0 = np.linalg.eigvalsh(Mx[1, 1, 1, 1:, 1:]); w1 = np.linalg.eigvalsh(P1[1, 1, 1, 1:, 1:])
    ok5 = np.max(np.abs(P1 - P2)) < 1e-14 and abs(w1[1] - w1[0]) < 1e-14 and abs(w1[2] - w0[2]) < 1e-14 and abs(w1[0] - 0.5 * (w0[0] + w0[1])) < 1e-14 \
        and np.max(np.abs(P1[1, 1, 1, 0] - Mx[1, 1, 1, 0])) == 0.0 and np.max(np.abs(P1[~mk] - Mx[~mk])) == 0.0
    aG = np.max(np.abs(C.a0_G1(P1)[1, 1, 1])); aL = np.max(np.abs(C.a0_local(P1)[1, 1, 1]))
    check(sec, "degenerate_project: idempotent, gap 0, leading eigenvalue and off-mask cells untouched; time row untouched, so on a NON-block-diagonal cell a0 != 0 after projection (both a0_G1 and a0_local)",
          ok5 and aG > 1e-3 and aL > 1e-3, f"idempotent+spectrum ok {ok5}; |a0_G1| {aG:.3f}, |a0_local| {aL:.3f} on a projected random full cell (W3 fields are block-diagonal, so not hit)")
    # (6) projection after the step, velocity and force NOT projected: fmax includes the constraint-normal component of the gradient
    seed = os.path.join(RES, "checkpoints", "m5_32_r10", "relax_g8_n32_L48_it12000.npy")
    if os.path.exists(seed):
        Ms = np.load(seed); cfgs = cfg_of(32, 48.0); hh = cfgs["h"]
        X_, Y_, Z_ = INS4.coords(32, hh); r = np.sqrt(X_ ** 2 + Y_ ** 2 + Z_ ** 2)
        sh = np.abs(r - 9.0) < 0.5 * hh
        Mp = C.degenerate_project(Ms, sh)
        Gd = INS4.grad(Mp, cfgs)
        w, V = np.linalg.eigh(Mp[sh][..., 1:, 1:])
        e0, e1 = V[..., :, 0], V[..., :, 1]
        n_split = np.einsum("...i,...ij,...j->...", e0, Gd[sh][..., 1:, 1:], e0) - np.einsum("...i,...ij,...j->...", e1, Gd[sh][..., 1:, 1:], e1)
        n_mix = np.einsum("...i,...ij,...j->...", e0, Gd[sh][..., 1:, 1:], e1)
        normal = np.sqrt(0.5 * n_split ** 2 + 2 * n_mix ** 2)
        fmax = float(np.max(np.abs(Gd)))
        check(sec, "W3 CAVEAT: on the projected hedgehog shell (R = 9, R10 seed) the FREE gradient has a constraint-normal (split) component of the same order as fmax; fire_proj measures fmax on the unprojected force, so f_tol cannot certify a constrained stationary point",
              np.max(normal) > 0.05 * fmax, f"max normal component on the shell {np.max(normal):.3e} vs fmax {fmax:.3e} (ratio {np.max(normal) / fmax:.2f}); shell cells {int(sh.sum())}")
        # also: on the projected shell the gap is zero but a0_local vs a0_G1 differ (hedgehog frame): a0_local is the right rigid generator there
        gap = C.gap23(Mp)[sh]
        aG = np.max(np.abs(C.a0_G1(Mp)[sh])); aL = np.max(np.abs(C.a0_local(Mp)[sh]))
        check(sec, "W3 shell generator: on the projected hedgehog shell gap23 = 0 exactly and a0_local = 0 there, while [G1, M] does NOT vanish (frame caveat of section A): W3 correctly uses a0_local",
              np.max(np.abs(gap)) < 1e-13 and aL < 1e-12 and aG > 1e-3, f"max gap {np.max(np.abs(gap)):.1e}; |a0_local| {aL:.1e}; |a0_G1| {aG:.3f}")
    else:
        check(sec, "W3 shell caveats on the R10 seed", False, "seed not present, skipped")


SECTIONS = {"A": audit_A, "B": audit_B, "C": audit_C, "D": audit_D, "E": audit_E, "F": audit_F,
            "G": audit_G, "H": audit_H, "I": audit_I, "J": audit_J}


# ================================================================ w012_recheck: the applied corrections
def audit_w012_recheck():
    """re-check ONLY the pieces changed after the w012 audit (2026-09-02): common.fire_proj (J1, J7),
    w0 S6 + the S4 lattice residual (F4, F5, C4, D2), w1 release_stability + summary (G1, G6, G7),
    w2 released_controls + verdict (H6, I6).  Own code throughout; the producer's fire_proj and
    degenerate_project are called only as the objects under test."""
    sec = "R"
    C = _load("m5_32_r13w_common", "m5_32_r13w_common.py")
    srcc = open(os.path.join(HERE, "m5_32_r13w_common.py")).read()
    src0 = open(os.path.join(HERE, "m5_32_r13w_w0.py")).read()
    # ---- J1: plateau logic
    M, cfg, kz = slab(16, 24.0, nxy=1)
    wall = np.zeros((1, 1, 16), bool); wall[:, :, kz] = True
    free = np.ones((1, 1, 16), bool); free[:, :, :2] = False; free[:, :, -2:] = False
    proj = lambda X: C.degenerate_project(X, wall)
    _, ic = C.fire_proj(M, cfg, free, 5000, project=proj, tag="re_le2500", log_every=2500)
    Mv = np.broadcast_to(VAC, (1, 1, 16, 4, 4)).copy()
    _, ip = C.fire_proj(Mv, cfg, free, 6000, project=None, tag="re_flat", log_every=1000, f_tol=0.0)
    check(sec, "J1 fixed: log_every 2500 > plateau[0] no longer stops at the first log line (runs to 5000, energy still falling); a genuinely flat energy (vacuum, f_tol = 0) is still caught as 'plateau' at the third log line (back = 2)",
          ic["stop"] == "max_iter" and ic["iters"] == 5000 and ip["stop"] == "plateau" and ip["iters"] == 3000,
          f"descending run: stop {ic['stop']} @ {ic['iters']}; flat run: stop {ip['stop']} @ {ip['iters']}")
    # ---- J7: the force seen by fmax and P is the constraint-tangent projection
    seed = os.path.join(RES, "checkpoints", "m5_32_r10", "relax_g8_n32_L48_it12000.npy")
    if os.path.exists(seed):
        Ms = np.load(seed); cfgs = cfg_of(32, 48.0); hh = cfgs["h"]
        X_, Y_, Z_ = INS4.coords(32, hh); r = np.sqrt(X_ ** 2 + Y_ ** 2 + Z_ ** 2)
        sh = np.abs(r - 9.0) < 0.5 * hh
        frees = ~INS4.pin_shell(32, hh)
        Mp = C.degenerate_project(Ms, sh)
        F = -INS4.grad(Mp, cfgs) * frees[..., None, None]
        Fp = (C.degenerate_project(Mp + 1e-6 * F, sh) - Mp) / 1e-6          # the producer's linearized projection, own call
        # own analytic tangent projection: in the eigenframe of each shell cell remove the split and the mixing of the degenerate pair
        T = F.copy()
        w, V = np.linalg.eigh(Mp[sh][..., 1:, 1:])
        Fs = np.einsum("...ji,...jk,...kl->...il", V, F[sh][..., 1:, 1:], V)          # V^T F V
        Fs2 = Fs.copy()
        m = 0.5 * (Fs[..., 0, 0] + Fs[..., 1, 1])
        Fs2[..., 0, 0] = m; Fs2[..., 1, 1] = m; Fs2[..., 0, 1] = 0.0; Fs2[..., 1, 0] = 0.0
        blk = F[sh].copy(); blk[..., 1:, 1:] = np.einsum("...ij,...jk,...lk->...il", V, Fs2, V)
        T[sh] = blk
        def normal_comp(G):
            Gs = np.einsum("...ji,...jk,...kl->...il", V, G[sh][..., 1:, 1:], V)
            return float(np.max(np.sqrt(0.5 * (Gs[..., 0, 0] - Gs[..., 1, 1]) ** 2 + 2 * Gs[..., 0, 1] ** 2)))
        fmax = float(np.max(np.abs(F)))
        nF, nFp = normal_comp(F), normal_comp(Fp)
        dev = float(np.max(np.abs(Fp - T)))
        _, i1 = C.fire_proj(Mp, cfgs, frees, 1, project=lambda X: C.degenerate_project(X, sh), tag="re_one", log_every=1, dt0=1e-12, dt_max=1e-12)
        check(sec, "J7 fixed: on the projected hedgehog shell the linearized projection equals the analytic tangent projection (max dev 1e-6 level), its constraint-normal component drops from 0.12 to ~0, and fire_proj's logged fmax is max|F_tangent|",
              nF > 0.05 * fmax and nFp < 1e-6 * fmax and dev < 1e-5 * fmax and rel(i1["trace"][-1]["fmax"], float(np.max(np.abs(Fp)))) < 1e-6,
              f"normal component: raw {nF:.3e}, projected {nFp:.2e} (fmax {fmax:.3e}); max|Fp - T_own| = {dev:.2e}; fire_proj fmax {i1['trace'][-1]['fmax']:.6e} vs max|Fp| {float(np.max(np.abs(Fp))):.6e}")
        # J7 residual: v is still unprojected (only F is); with F tangent v stays tangent to first order, so this is a curvature-order effect only
        check(sec, "J7 residual (design note, not a defect): only F is projected, v is not; with F tangent at every step v is tangent up to the manifold curvature, and M is re-projected after each step",
              "v += dt * F" in srcc and "Fx = (project(Mx + s * Fx) - Mx) / s" in srcc, "source read")
    else:
        check(sec, "J7 on the R10 seed", False, "seed missing")
    # ---- W0 S6 (F4, F5, C4): both V references, full minimum, own minimizer
    w0 = json.load(open(os.path.join(DATA, "m5_32_r13w_w0.json")))
    S6 = w0["S6_numbers"]
    c = 4 * DELTA ** 4; h = 1.5; kap = 14.0
    def Ebag(R, J, V):
        return 4 * np.pi * R ** 2 * h * V + J ** 2 / (4 * (kap * R + 4 * np.pi * R ** 2 * c / h))
    vmin, _, _ = deg_min(False)
    refs = {"V_projected": 1.7994130394803167e-06, "V_min": vmin}
    devs = {}
    for nm, V in refs.items():
        for J in (50.0, 200.0, 800.0):
            rr = minimize(lambda x: Ebag(x[0], J, V), [100.0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-20})
            Rs = float(rr.x[0]); om = J / (2 * (kap * Rs + 4 * np.pi * Rs ** 2 * c / h))
            th = S6[f"{nm}_J{J:g}"]
            devs[f"{nm}_J{J:g}"] = (round(Rs, 2), round(th["R_star"], 2), rel(Rs, th["R_star"]), rel(om, th["omega_star"]))
    check(sec, "S6 corrected numbers: own full-minimum R*/omega* for both V references match the producer's (V_projected 75.3/164.3/344.8, omega 0.00965/0.01039/0.01078; V_min 100.9/216.6/450.2, omega 0.00599/0.00634/0.00652) to < 1e-3; kappa = 14 is labeled an R7 import",
          all(v[2] < 1e-3 and v[3] < 1e-3 for v in devs.values()) and S6.get("kappa_import_R7") == 14.0 and abs(S6["V4_deg_constrained_min"] - vmin) < 1e-12,
          str(devs))
    Jd = {}
    for Rt, key in ((12.0, "J_for_Rstar_12_Vmin"), (20.0, "J_for_Rstar_20_Vmin")):
        J = S6[key]
        rr = minimize(lambda x: Ebag(x[0], J, vmin), [Rt], method="Nelder-Mead", options={"xatol": 1e-8})
        Jd[key] = (J, float(rr.x[0]))
    check(sec, "S6 corrected feasibility: at the producer's J_for_Rstar values (1.566, 3.445, V_min) the full bag minimum sits at R* = 12 and 20 (own argmin); all six R* exceed 48",
          all(rel(v[1], Rt) < 1e-3 for (Rt, v) in zip((12.0, 20.0), Jd.values())) and all(S6[k]["R_star"] > 48 for k in S6 if k.startswith("V_")),
          str(Jd))
    # ---- W0 D2 check: their phase field, own E_u and kin
    rows = []
    for n, Lb in ((16, 24.0), (32, 24.0)):
        cfgb = cfg_of(n, Lb); hb = cfgb["h"]
        X_, Y_, Z_ = INS4.coords(n, hb)
        phi = 0.7 * np.sin(2 * np.pi * X_ / Lb) * np.cos(2 * np.pi * Z_ / Lb)
        R = rot(G1, phi); Mn = np.einsum("...ab,bc,...dc->...ad", R, VAC, R)
        _, k = my_kin_cells(Mn, a0_G1(Mn), hb)
        rows.append((float(my_e_u(Mn, hb)), k))
    pe = np.log(rows[0][0] / rows[1][0]) / np.log(2); pk = np.log(rows[0][1] / rows[1][1]) / np.log(2)
    S4r = w0["S4_lattice_residual"]
    check(sec, "S4 lattice-residual check (added for D2): own E_u and kin on the producer's phase field reproduce the exponents 1.87 / 1.87 and the values",
          abs(pe - S4r["exp_E_u"]) < 1e-6 and abs(pk - S4r["exp_kin"]) < 1e-6 and rel(rows[0][0], S4r["rows"][0]["E_u"]) < 1e-12 and w0["summary"]["pass"] == w0["summary"]["checks"] == 22,
          f"own exponents {pe:.4f}, {pk:.4f}; own E_u(h=1.5) {rows[0][0]:.6e} vs {S4r['rows'][0]['E_u']:.6e}; W0 {w0['summary']['pass']}/{w0['summary']['checks']}")
    check(sec, "W0 docstring carries the A3 (frame), D2 (lattice residual) and E7 (sign) qualifications",
          all(t in src0 for t in ("a0_local", "O(h^2)", "block-diagonal")) or all(t in src0 for t in ("frame", "h^2", "boost")),
          "grep of the module docstring for the three qualifiers")
    # ---- W1 summary (G1, G3) and release_stability (G6, G7)
    w1 = json.load(open(os.path.join(DATA, "m5_32_r13w_w1.json"))); s1 = w1["summary"]; rs = w1["release_stability"]
    conv = s1.get("converged_reference_h_V4deg_min") or {}
    ok1 = all(rel(conv[str(hh)], hh * vmin) < 1e-9 for hh in (1.5, 1.0, 0.75)) and s1.get("sigma_0_is_unconverged_snapshot") is True \
        and all(v > 1e-5 for v in s1["fmax_at_stop"].values()) and "projected_point" in json.dumps(s1)
    check(sec, "W1 summary corrected: converged_reference = h x 6.4789e-7 (9.718e-7 / 6.479e-7 / 4.859e-7), snapshot flag true, fmax_at_stop 5.1e-4 / 1.7e-4 / 6.4e-5 recorded, the W0 row relabeled 'projected_point'",
          ok1, f"converged_reference {conv}; fmax_at_stop {s1['fmax_at_stop']}")
    log4 = open(os.path.join(CK, "w1_v4.log")).read() if os.path.exists(os.path.join(CK, "w1_v4.log")) else ""
    con = re.findall(r"w1_stab_con it\s+(\d+)", log4); relr = re.findall(r"w1_stab_rel it\s+(\d+)", log4)
    check(sec, "G7 fixed: release_stability's constrained and released stages now run 12000 iterations each (12 log lines, last at 12000)",
          len(con) == 12 and len(relr) == 12 and con[-1] == "12000" and relr[-1] == "12000", f"con lines {len(con)} (last {con[-1:] }), rel lines {len(relr)} (last {relr[-1:]})")
    # reproduce the released cell itself (con 12000 projected + rel 12000, the producer's fire_proj on a 1x1 column: per-cell identical)
    n = 32; cfg1 = cfg_of(n, 48.0); kz = n // 2
    Mcol = np.broadcast_to(VAC, (1, 1, n, 4, 4)).copy()
    wall1 = np.zeros((1, 1, n), bool); wall1[:, :, kz] = True
    free1 = np.ones((1, 1, n), bool); free1[:, :, :2] = False; free1[:, :, -2:] = False
    Mc1, _ = C.fire_proj(Mcol, cfg1, free1, 12000, project=lambda X: C.degenerate_project(X, wall1), tag="re_con", log_every=12000, f_tol=1e-6)
    Mr1, _ = C.fire_proj(Mc1, cfg1, free1, 12000, project=None, tag="re_rel", log_every=12000, f_tol=1e-6)
    cellR = Mr1[0, 0, kz]
    lam = np.diag(cellR @ ETA); mm = lam[2]; t = [np.sum(lam ** p) for p in range(1, 5)]
    curv_own = W1 * 4 * sum((t[p - 1] - CP[p - 1]) * p * (p - 1) * mm ** (p - 2) for p in range(1, 5))
    offd = float(np.max(np.abs(cellR - np.diag(np.diag(cellR)))))
    tr = rs["gap_trace_after_perturbation"]
    mono = all(b[1] >= a[1] for a, b in zip(tr, tr[1:]))
    growth = rs["gap_end"] / rs["gap_start"] - 1
    pc = rs["per_cell_descent_from_split"]
    check(sec, "G6 corrected: gap_start = 2 eps recorded; curvature -2.019e-4 at the 12000 + 12000 released cell reproduced (own re-run of the two stages on a 1x1 column, analytic formula on the resulting diagonal cell, M00 = 7.999993); monotone growth 0.16 percent under FIRE; per-cell descent from the split reaches the vacuum spectrum (gap 0.2999); flag true by all three clauses",
          abs(rs["gap_start"] - 2e-3) < 1e-12 and rel(curv_own, rs["V4_curvature_along_split"]) < 2e-3 and offd == 0.0 and mono and 1e-3 < growth < 5e-3
          and pc["gap_end"] > 0.29 and pc["V4_end"] < 1e-10 and rs["unstable"] is True,
          f"gap_start {rs['gap_start']}; curvature own {curv_own:.5e} (cell diag {np.round(np.diag(cellR), 7).tolist()}) vs {rs['V4_curvature_along_split']:.5e}; growth {growth * 100:.3f} percent, monotone {mono}; per-cell end eigs {pc['eigs_end']}, V4 {pc['V4_end']:.1e}")
    check(sec, "G6 residual fragility (note): the growth clause gap_end > gap_start (1 + 1e-3) passes with a 1.65x margin (growth 1.65e-3); the instability evidence is the curvature sign and the per-cell descent, the FIRE trajectory adds little",
          growth / 1e-3 > 1.3 and growth / 1e-3 < 3.0, f"growth / threshold = {growth / 1e-3:.2f}")
    # ---- W2 controls (H6) and verdict (I6)
    w2 = json.load(open(os.path.join(DATA, "m5_32_r13w_w2.json")))
    jr = w2["controls"]["jump_23_vs_dq"]
    law = 32 * DELTA ** 4 / 1.5
    okj = all(rel(r_["kin_co_per_area"], law * np.sin(r_["dq"]) ** 4) < 1e-9 for r_ in jr if np.sin(r_["dq"]) ** 4 > 1e-6)
    check(sec, "H6 fixed: the (2,3) jump control now reads a0 on every cell: kin/area = 32 delta^4 sin^4(dq)/h at all 25 angles, max 0.17280",
          okj and abs(max(r_["kin_co_per_area"] for r_ in jr) - 0.1728) < 1e-9, f"law holds at all angles: {okj}; max {max(r_['kin_co_per_area'] for r_ in jr):.5f}")
    vd = w2["summary"]["verdict"]
    check(sec, "I6 applied: the W2 verdict string flags BOTH gates as identities of a diagonal planar slab and the pass as carrying no evidential weight, while keeping the pre-registered PASS/W3-runs outcome",
          "BOTH gates" in vd and "no evidential weight" in vd and "W3 runs" in vd, vd[:160] + "...")
    # ---- unchanged reproduction: the W1 sigma_0 and the W2 kin_wall were not touched
    d_sig = rel(s1["sigma_0_vs_h"]["1.5"], 1.7061402971438648e-06)
    check(sec, "unchanged within roundoff: W1 sigma_0 snapshots moved by 6e-9 relative (the J7 force path perturbs FIRE's P-sign decisions at roundoff), W2 kin_wall/area identical",
          d_sig < 1e-7 and abs(w2["summary"]["kin_wall_per_area_vs_h_at_L48"]["1.0"] - 0.0324) < 1e-12, f"sigma_0(h=1.5) rel change {d_sig:.1e}")


SECTIONS["R"] = audit_w012_recheck



# ================================================================ w3: the 3D fixed-J relaxations (independent audit)
# Own instruments for W3: the producer's numbers are recomputed from the saved end fields with the
# audit's own jets / eta algebra / generator (leading_axis_gen) and own masks; m5_32_r13w_common.py is
# imported ONLY in w3_ctrl where fire_proj itself is the object under test.
W3_RUNS = None


def w3_records():
    """all w3_*.json checkpoint records keyed by their file stem (traces included)."""
    global W3_RUNS
    if W3_RUNS is None:
        import glob
        W3_RUNS = {}
        for pth in sorted(glob.glob(os.path.join(CK, "w3_*.json"))):
            W3_RUNS[os.path.basename(pth)[:-5]] = json.load(open(pth))
    return W3_RUNS


def w3_field(key):
    return np.load(os.path.join(CK, key + ".npy"))


_SEEDS = {}


def w3_seed(n, L, it=None):
    """the seed the producer used (n32 L48: R10's 12000-iteration state; n48: the 3000-iteration seeds), or a named R10 state."""
    k = (n, float(L), it)
    if k not in _SEEDS:
        if n == 32 and abs(L - 48.0) < 1e-9:
            _SEEDS[k] = np.load(os.path.join(RES, "checkpoints", "m5_32_r10", f"relax_g8_n32_L48_it{it or 12000}.npy"))
        else:
            _SEEDS[k] = np.load(os.path.join(CK, f"seed_n{n}_L{L:g}_it3000.npy"))
    return _SEEDS[k]


def w3_geom(n, L):
    h = float(L) / n
    X, Y, Z = INS4.coords(n, h)
    return h, np.sqrt(X * X + Y * Y + Z * Z)


def gap23(M):
    w = np.linalg.eigvalsh(M[..., 1:, 1:])
    return w[..., 1] - w[..., 0]


def a0_two_region(M, mask=None):
    """own two-region generator: rotation about the local leading eigenvector, times the indicator."""
    Jg = leading_axis_gen(M)
    a = Jg @ M - M @ Jg
    return a if mask is None else a * mask[..., None, None]


def deg_project(M, mask):
    """own degeneracy projection: the two smallest spatial eigenvalues replaced by their mean on `mask`."""
    out = M.copy()
    sub = M[mask][..., 1:, 1:]
    w, V = np.linalg.eigh(sub)
    m = 0.5 * (w[..., 0] + w[..., 1]); w = w.copy(); w[..., 0] = m; w[..., 1] = m
    blk = M[mask].copy(); blk[..., 1:, 1:] = np.einsum("...ik,...k,...jk->...ij", V, w, V)
    out[mask] = blk
    return out


def my_v4_g(M, h, g):
    """own V4 with the vacuum constants for an arbitrary g (my_v4 hardcodes g = 8)."""
    sg = S * g
    cp = tuple(sg ** p + 1.0 + DELTA ** p for p in range(1, 5))
    Me = M @ ETA
    P = np.broadcast_to(np.eye(4), M.shape).copy()
    v = 0.0
    for p in range(4):
        P = P @ Me
        v = v + (np.einsum("...kk->...", P) - cp[p]) ** 2
    return h ** 3 * W1 * np.sum(v)


def kin_dens(M, a0, h):
    tab, tot = my_kin_cells(M, a0, h)
    return sum(tab.values()), tot


def jets_in_frame(M, V, h, part):
    """jets with only the eigenvalue (diag-in-local-frame) or the orientation (off-diag) part kept."""
    out = {}
    idx = np.arange(3)
    for br in ("fwd", "bwd"):
        Aj = jets(M, h, br); lst = []
        for i in range(3):
            B = np.einsum("...ki,...kl,...lj->...ij", V, Aj[i][..., 1:, 1:], V)
            D = np.zeros_like(B); D[..., idx, idx] = B[..., idx, idx]
            Bp = D if part == "diag" else (B - D if part == "off" else B)
            Ap = np.zeros_like(Aj[i]); Ap[..., 1:, 1:] = np.einsum("...ik,...kl,...jl->...ij", V, Bp, V)
            lst.append(Ap)
        out[br] = lst
    return out


def e_u_from_jets(jd, h):
    e = 0.0
    for br, Aj in jd.items():
        for i in range(3):
            for j in range(i + 1, 3):
                F = comm(Aj[i], Aj[j]); e += 0.5 * 4.0 * float(np.sum(inner(F, F)))
    return h ** 3 * e


def kin_from_jets(jd, a0, h):
    k = 0.0
    for br, Aj in jd.items():
        for i in range(3):
            F = comm(a0, Aj[i]); k += 0.5 * 4.0 * float(np.sum(inner(F, F)))
    return h ** 3 * k


def w3_end_numbers(key):
    """own end-state numbers from the saved field: E_u, V4, kin (two-region), rigid kin, kin with the seed's a0."""
    r = w3_records()[key]
    n, L, Rs, J = r["n"], r["L"], r["Rs"], r["J"]
    M = w3_field(key); h, rad = w3_geom(n, L)
    inside = rad < Rs; shell = np.abs(rad - Rs) < 0.5 * h
    eu, ev = my_e_u(M, h), my_v4(M, h)
    _, k2 = my_kin_cells(M, a0_two_region(M, inside), h)
    _, krig = my_kin_cells(M, a0_two_region(M), h)
    M0 = deg_project(w3_seed(n, L), shell)
    _, kseed = my_kin_cells(M, a0_two_region(M0, inside), h)
    return {"E_u": eu, "V4": ev, "kin": k2, "omega": J / (2 * k2), "E_J": eu + ev + J * J / (4 * k2), "kin_rigid": krig,
            "kin_seed_a0": kseed, "E_J_seed_a0": eu + ev + J * J / (4 * kseed)}


def audit_w3_A():
    sec = "A"
    keys = ["w3_n32_L48_R6_J200_it3000", "w3_n32_L48_R9_J200_it3000", "w3_n32_L48_R9_J200_it12000", "w3_n32_L48_R12_J800_it3000",
            "w3_n32_L48_R15_J50_it3000", "w3_n48_L48_R9_J200_it3000", "w3_n48_L72_R21_J200_it3000", "w3_n32_L48_R9_J200_it3000_frz"]
    worst = 0.0; rows = {}
    ratios = []
    for key in keys:
        r = w3_records()[key]; e = r["end"]; mine = w3_end_numbers(key)
        # the frozen run's recorded 'kin' is the frozen-a0 read; its refreshed read is 'kin_with_refreshed_a0'
        ref_kin = e["kin_with_refreshed_a0"] if r.get("frozen") else e["kin"]
        ref_EJ = e["E_J_with_refreshed_a0"] if r.get("frozen") else e["E_J"]
        d = max(rel(mine["E_u"], e["E_u"]), rel(mine["V4"], e["V4"]), rel(mine["kin"], ref_kin), rel(mine["E_J"], ref_EJ), rel(mine["kin_rigid"], e["kin_rigid_control"]))
        worst = max(worst, d)
        rows[key] = {"mine": mine, "theirs": {k: e[k] for k in ("E_u", "V4", "kin", "omega", "E_J", "kin_rigid_control")}, "max_rel_dev": d}
        ratios.append((key, mine["kin_seed_a0"] / mine["kin"], mine["E_J_seed_a0"], mine["E_J"]))
        if r.get("frozen"):
            worst = max(worst, rel(mine["kin_seed_a0"], e["kin"]))
    check(sec, "end-state numbers (E_u, V4, two-region kin, omega, E_J, rigid kin) recomputed from the saved fields with own jets/eta/generator for 8 runs across R_s, J, box, and the frozen run: all within 1e-9 of the records",
          worst < 1e-9, f"max rel dev {worst:.1e} over {len(keys)} runs")
    infl = {key: rows[key]["mine"]["kin"] / w3_records()[key]["start"]["kin_seed_two_region"] for key in keys}
    ok_hi = all(q < 0.1 for k, q, _, _ in ratios if infl[k] > 50); ok_lo = all(q > 0.5 for k, q, _, _ in ratios if infl[k] < 5)
    check(sec, "the refreshed generator inflates the READ, not only the field: on the SAME end field the kin with the seed's (shelled) a0 is 12 to 80x smaller than with a0 refreshed wherever kin inflated 50x or more (ratio 0.012 to 0.085), and within a factor 1.6 where it inflated under 5x (R_s 15 J 50: 0.96; R_s 21 n48: 0.62): 'kin' and 'E_J' are generator-choice statements (E_J 16.3 becomes 70 at R_s 9 J 200 with the seed's a0)",
          ok_hi and ok_lo and sum(1 for k in keys if infl[k] > 50) >= 5, "; ".join(f"{k[3:]}: inflation {infl[k]:.0f}x, seed/refreshed {q:.3f}, E_J {ejs:.1f} vs {ej:.1f}" for k, q, ejs, ej in ratios))
    # the producer's end line: "kin ends 30x to 1000x above the seed's two-region value"
    infl_all = {k: (r["end"]["kin_with_refreshed_a0"] if r.get("frozen") else r["end"]["kin"]) / r["start"]["kin_seed_two_region"] for k, r in w3_records().items()}
    lo, hi = min(infl_all, key=infl_all.get), max(infl_all, key=infl_all.get)
    check(sec, "the producer's '30x to 1000x above the seed' understates the spread: the inflation over the 20 runs ranges from 1.7x (R_s 15 J 50) to 5478x (R_s 6 J 800); 121x at R_s 9 J 200 (3000 iterations), 468x at 12000; the J 50 runs at R_s 12 and 15 inflate under 4x",
          infl_all[lo] < 2 and infl_all[hi] > 5000 and abs(infl_all["w3_n32_L48_R9_J200_it3000"] - 121.4) < 1 and abs(infl_all["w3_n32_L48_R9_J200_it12000"] - 468.3) < 1,
          f"min {lo[3:]} {infl_all[lo]:.1f}x; max {hi[3:]} {infl_all[hi]:.0f}x")
    note(sec, "inflation_all_runs", infl_all)
    note(sec, "rows", rows)


def audit_w3_B():
    sec = "B"
    recs = w3_records()
    mono_mine, mono_theirs, lq, still = {}, {}, {}, {}
    for key, r in recs.items():
        kins = [row["kin"] for row in r["trace"]]
        mono_mine[key] = all(b >= a for a, b in zip(kins, kins[1:]))
        mono_theirs[key] = r["kin_monotone_increasing"]
        i0 = max(0, 3 * len(kins) // 4 - 1)
        lq[key] = kins[-1] / kins[i0] - 1.0
        still[key] = kins[-1] > kins[i0]
    n_non = sum(1 for v in mono_mine.values() if not v)
    check(sec, "producer's per-run monotone flags reproduce from the traces; 9 of 20 runs (all R_s 6, R_s 9 at J 180/200/220/800 and the 12000-iteration run, R_s 15 J 50) are NOT monotone: 'kin grows monotonically' is false as a literal, the net trend is up with 10 to 20 percent dips (the producer's own kin_monotone_all_runs is False)",
          all(mono_mine[k] == mono_theirs[k] for k in recs) and n_non == 9, f"non-monotone: {sorted(k[3:] for k, v in mono_mine.items() if not v)}")
    check(sec, "growth still going in the last quarter of EVERY run (kin_end > kin at 3/4): last-quarter growth 4 percent (R_s 15 J 50) to 77 percent (R_s 9 J 180); no plateau anywhere",
          all(still.values()) and min(lq.values()) > 0.03 and max(lq.values()) > 0.5, f"last-quarter growth: min {min(lq.values()):.3f} max {max(lq.values()):.3f}")
    r12 = recs["w3_n32_L48_R9_J200_it12000"]; tr = r12["trace"]
    q = [row["kin"] for row in tr]
    quarters = [q[len(q) * i // 4 - 1] for i in (1, 2, 3, 4)]
    check(sec, "12000 iterations at R_s 9 J 200: kin at the quarter marks 2653 -> 6436 -> 8313 -> 10411, +25 percent in the last quarter, fmax 0.076 (packet f_tol 1e-6): no stationary point reached",
          all(b > a for a, b in zip(quarters, quarters[1:])) and quarters[-1] / quarters[-2] > 1.2 and tr[-1]["fmax"] > 1e-3, f"quarters {[round(v) for v in quarters]}, fmax_end {tr[-1]['fmax']:.2e}")
    frz = recs["w3_n32_L48_R9_J200_it3000_frz"]
    kf = [row["kin"] for row in frz["trace"]]
    check(sec, "frozen-a0 control (a0 fixed at the shelled seed, the functional whose exact gradient drives the descent): kin (frozen read) rises monotonically 27 -> 637 and omega FALLS 0.48 -> 0.157, still falling at the end (the producer's end line 'omega 0.157 still rising' has the direction of omega wrong; kin is what rises): same direction as the refreshed runs",
          all(b > a for a, b in zip(kf, kf[1:])) and frz["end"]["kin"] > 20 * frz["start"]["kin_shell_two_region"] and frz["trace"][-1]["omega"] < frz["trace"][-2]["omega"],
          f"kin {frz['start']['kin_shell_two_region']:.1f} -> {frz['end']['kin']:.1f}; omega {frz['trace'][0]['omega']:.3f} -> {frz['end']['omega']:.4f}")
    # the two R_s 9 J 200 runs (3000 and 12000) share seed and code path: reproducibility floor
    a, b = recs["w3_n32_L48_R9_J200_it3000"]["trace"], r12["trace"]
    same = [abs(x["E"] - y["E"]) for x, y in zip(a, b[:12])]
    check(sec, "reproducibility floor: the 3000- and 12000-iteration runs at R_s 9 J 200 (same seed, same call) agree to 1e-12 through iteration 750 and diverge from iteration 1000 (roundoff amplified by FIRE): at iteration 3000 E_J differs by 0.063 (0.4 percent) and kin by 1.7 percent",
          max(same[:3]) < 1e-9 and same[3] > 1e-3 and abs(same[-1] - 0.063) < 0.01, f"|dE| per log row: {[f'{v:.1e}' for v in same]}")
    note(sec, "last_quarter_growth", lq); note(sec, "monotone", mono_mine)


def audit_w3_C():
    sec = "C"
    recs = w3_records()
    ladders = {}
    for (n, L, J) in [(32, 48.0, 50.0), (32, 48.0, 200.0), (32, 48.0, 800.0), (48, 72.0, 200.0)]:
        grp = sorted([r for k, r in recs.items() if r["n"] == n and r["L"] == L and r["J"] == J and r["maxit"] == 3000 and not r.get("frozen")], key=lambda r: r["Rs"])
        Rs = [r["Rs"] for r in grp]
        EJ, Es, kin = [], [], []
        for r in grp:
            key = f"w3_n{n}_L{L:g}_R{r['Rs']:g}_J{J:g}_it3000"
            mine = w3_end_numbers(key)
            EJ.append(mine["E_J"]); Es.append(mine["E_u"] + mine["V4"]); kin.append(mine["kin"])
        ladders[f"n{n}_L{L:g}_J{J:g}"] = {"Rs": Rs, "E_J": EJ, "E_stat": Es, "E_kin": [J * J / (4 * k) for k in kin], "kin": kin, "argmin": Rs[int(np.argmin(EJ))]}
    inc = all(all(b > a for a, b in zip(v["E_J"], v["E_J"][1:])) for v in ladders.values())
    low = all(v["argmin"] == min(v["Rs"]) for v in ladders.values())
    check(sec, "E_J(R_s) at fixed J is monotone INCREASING over the grid on all four ladders (own numbers), so the argmin is the grid's LOWER edge in every ladder (6 on n32, 9 on n48 L72): 'R* = 6' is a grid-edge argmin, not an interior minimum",
          inc and low, "; ".join(f"{k}: E_J {[round(x, 2) for x in v['E_J']]} argmin {v['argmin']}" for k, v in ladders.items()))
    kin_dec = all(all(b < a for a, b in zip(v["kin"], v["kin"][1:])) for v in ladders.values())
    check(sec, "the relaxed kin DECREASES with R_s on every ladder (6603 -> 1398 at J 200), the reverse of any bag law kin ~ R^p with p > 0: the ordering is set by how much inertia the descent built at the flank in 3000 iterations, larger at small R_s, not by a shell-vs-inertia balance",
          kin_dec, "; ".join(f"{k}: kin {[round(x) for x in v['kin']]}" for k, v in ladders.items()))
    v = ladders["n32_L48_J200"]
    dEs, dEk = v["E_stat"][-1] - v["E_stat"][0], v["E_kin"][-1] - v["E_kin"][0]
    check(sec, "at J 200 both terms rise with R_s: E_stat +6.3 (10.3 -> 16.6, the shell area and the melt zone grow) and J^2/(4 kin) +5.6 (1.5 -> 7.2, less inflation at larger R_s); neither term has a minimum inside the grid",
          dEs > 5 and dEk > 5 and all(b > a for a, b in zip(v["E_stat"], v["E_stat"][1:])) and all(b > a for a, b in zip(v["E_kin"], v["E_kin"][1:])),
          f"E_stat {[round(x, 2) for x in v['E_stat']]}; E_kin {[round(x, 2) for x in v['E_kin']]}")
    # own bag fit from the seed family (own kin, own shell cost)
    Ms = w3_seed(32, 48.0); h, rad = w3_geom(32, 48.0)
    Rg_ = np.array([6.0, 9.0, 12.0, 15.0])
    kin_s = np.array([my_kin_cells(Ms, a0_two_region(Ms, rad < R), h)[1] for R in Rg_])
    eu0, ev0 = my_e_u(Ms, h), my_v4(Ms, h)
    sig = []
    for R in Rg_:
        P = deg_project(Ms, np.abs(rad - R) < 0.5 * h)
        sig.append((my_e_u(P, h) - eu0 + my_v4(P, h) - ev0) / (4 * np.pi * R * R))
    sig = np.array(sig); p, la = np.polyfit(np.log(Rg_), np.log(kin_s), 1); a = float(np.exp(la))
    bag = json.load(open(os.path.join(DATA, "m5_32_r13w_w3.json")))["summary"]["bag_closure_measured_inputs"]
    Rstar, EJ_seed_grid = {}, {}
    Rg = np.linspace(2.0, 400.0, 200000)
    for Jv in (50.0, 200.0, 800.0):
        E = 4 * np.pi * Rg ** 2 * sig.mean() + Jv ** 2 / (4 * a * Rg ** p)
        Rstar[Jv] = float(Rg[int(np.argmin(E))])
        EJ_seed_grid[Jv] = [float(4 * np.pi * R * R * s + Jv ** 2 / (4 * k)) for R, s, k in zip(Rg_, sig, kin_s)]
    check(sec, "own seed-family bag fit reproduces the producer's: kin_seed(R) ~ 0.0452 R^2.80 (residuals under 6 percent), sigma_3D 0.0018 to 0.0028 (falls with R, mean 0.00245), R*_seed = 16.1 / 28.7 / 51.1 at J = 50 / 200 / 800 (only J 50 inside the free region)",
          abs(p - bag["power_law_exponent"]) < 1e-6 and rel(a, bag["power_law_prefactor"]) < 1e-6 and all(abs(Rstar[Jv] - bag[f"J{Jv:g}"]["R_star_seed_family"]) < 0.05 for Jv in Rstar),
          f"p {p:.4f} a {a:.5f} sigma {np.round(sig, 5).tolist()} R* {Rstar}")
    rev = all(all(b < a_ for a_, b in zip(EJ_seed_grid[Jv], EJ_seed_grid[Jv][1:])) for Jv in EJ_seed_grid)
    check(sec, "the bag prediction is REVERSED by the descent: on the seed family E_J DEcreases across the grid at all three J (toward R*_seed beyond or at the grid's upper edge), while the relaxed E_J INcreases across the same grid; the relaxed ordering carries no information about the seed-family bag minimum",
          rev and inc, "; ".join(f"J{Jv:g}: seed-family {[round(x, 1) for x in EJ_seed_grid[Jv]]}" for Jv in EJ_seed_grid))
    note(sec, "ladders", ladders); note(sec, "bag_fit", {"p": p, "a": a, "sigma": sig.tolist(), "R_star": Rstar, "E_J_seed_family_on_grid": EJ_seed_grid})


def audit_w3_D():
    sec = "D"
    recs = w3_records()
    rows = {}
    for key, r in recs.items():
        n, L, Rs = r["n"], r["L"], r["Rs"]
        M = w3_field(key); h, rad = w3_geom(n, L)
        shell = np.abs(rad - Rs) < 0.5 * h
        g = gap23(M); low = g < 0.03
        far = low & (np.abs(rad - Rs) > h); far2 = low & (np.abs(rad - Rs) > 2 * h)
        rows[key] = {"gap_min": float(g[shell].min()), "gap_mean": float(g[shell].mean()), "gap_median": float(np.median(g[shell])),
                     "frac_lt_0.03": float((g[shell] < 0.03).mean()), "frac_lt_0.1": float((g[shell] < 0.1).mean()),
                     "n_low": int(low.sum()), "n_low_on_shell": int((low & shell).sum()), "n_low_far": int(far.sum()), "n_low_far2": int(far2.sum()),
                     "rec_min": r["end"]["gap_shell_min"], "rec_mean": r["end"]["gap_shell_mean"], "rec_n_low": r["end"]["n_cells_gap_lt_0.03"]}
    dev = max(max(rel(v["gap_min"], v["rec_min"]), rel(v["gap_mean"], v["rec_mean"]), abs(v["n_low"] - v["rec_n_low"])) for v in rows.values())
    check(sec, "shell gap statistics (min, mean over the shell cells, count of gap < 0.03) recomputed from the end fields for all 20 runs agree with the records",
          dev < 1e-9, f"max deviation {dev:.1e}")
    means = [v["gap_mean"] for v in rows.values()]
    check(sec, "'partial melt' at 3000 iterations: the shell's mean gap ends at 0.03 to 0.14 (11 to 46 percent of the vacuum gap 0.3, from 0 imposed) with the median 0.02 to 0.10; only 1 to 64 percent of the shell cells stay below 0.03 (min over the shell is a one-cell reading)",
          0.03 < min(means) < 0.04 and 0.13 < max(means) < 0.15 and min(v["frac_lt_0.03"] for v in rows.values()) < 0.02 and max(v["frac_lt_0.03"] for v in rows.values()) > 0.6,
          f"mean gap range {min(means):.3f} to {max(means):.3f}; frac<0.03 range {min(v['frac_lt_0.03'] for v in rows.values()):.3f} to {max(v['frac_lt_0.03'] for v in rows.values()):.3f}")
    check(sec, "the degenerate cells stayed at R_s: every cell with gap < 0.03 at the end lies within 2h of R_s in every run (0 to 10 cells per run sit between h and 1.7h from R_s, the melt zone's spread), and 71 to 100 percent of them are shell cells; no new degenerate structure appeared elsewhere",
          all(v["n_low_far2"] == 0 for v in rows.values()) and all(v["n_low_on_shell"] >= 0.7 * v["n_low"] for v in rows.values()) and max(v["n_low_far"] for v in rows.values()) <= 10,
          f"cells beyond h: {[v['n_low_far'] for v in rows.values()]}; beyond 2h: {sum(v['n_low_far2'] for v in rows.values())}; on-shell fraction min {min(v['n_low_on_shell'] / max(v['n_low'], 1) for v in rows.values()):.3f}")
    tr = recs["w3_n32_L48_R9_J200_it12000"]["trace"]
    gm = [row["gap_shell_mean"] for row in tr]
    quarters = [gm[len(gm) * i // 4 - 1] for i in (1, 2, 3, 4)]
    incs = [b - a for a, b in zip(quarters, quarters[1:])]
    check(sec, "the melt is PROGRESSIVE: over 12000 iterations the shell's mean gap rises at every quarter (0.081 -> 0.107 -> 0.122 -> 0.131, increments shrinking but not zero) and the count below 0.03 falls 46 -> 6 of 416; at 12000 iterations 'survives' rests on 6 cells while the mean is 44 percent of the vacuum gap",
          all(b > a for a, b in zip(quarters, quarters[1:])) and incs[-1] > 0 and incs[-1] < incs[0] and rows["w3_n32_L48_R9_J200_it12000"]["n_low"] == 6,
          f"quarters {[round(v, 4) for v in quarters]}; n<0.03 {tr[0]['n_cells_gap_lt_0.03']} -> {tr[-1]['n_cells_gap_lt_0.03']}")
    # exterior: the melt spreads one bin outward
    M = w3_field("w3_n32_L48_R9_J200_it3000"); h, rad = w3_geom(32, 48.0); Ms = w3_seed(32, 48.0)
    b_out = (rad >= 9.0) & (rad < 10.5); b_in = (rad >= 7.5) & (rad < 9.0)
    ge, gs = gap23(M), gap23(Ms)
    check(sec, "the gap depression extends one bin OUTSIDE the shell (bin 9.0 to 10.5: mean gap 0.165 vs 0.258 in the seed) and one bin inside (7.5 to 9.0: 0.198 vs 0.242): the wall is now two to three cells wide, not one",
          ge[b_out].mean() < 0.75 * gs[b_out].mean() and ge[b_in].mean() < gs[b_in].mean(), f"outside {ge[b_out].mean():.3f} vs seed {gs[b_out].mean():.3f}; inside {ge[b_in].mean():.3f} vs {gs[b_in].mean():.3f}")
    note(sec, "rows", rows)


def audit_w3_E():
    sec = "E"
    recs = w3_records()
    keys = ["w3_n32_L48_R6_J200_it3000", "w3_n32_L48_R9_J200_it3000", "w3_n32_L48_R9_J200_it12000", "w3_n32_L48_R15_J50_it3000",
            "w3_n32_L48_R12_J800_it3000", "w3_n48_L72_R9_J200_it3000", "w3_n32_L48_R9_J200_it3000_frz"]
    rows = {}
    for key in keys:
        r = recs[key]; n, L, Rs = r["n"], r["L"], r["Rs"]
        M = w3_field(key); h, rad = w3_geom(n, L); Ms = w3_seed(n, L)
        inside = rad < Rs
        a0 = a0_two_region(M, inside)
        dens, k = kin_dens(M, a0, h)
        flank = (rad >= Rs - 2 * h) & (rad < Rs); core = rad < Rs - 2 * h
        w, V = np.linalg.eigh(M[..., 1:, 1:]); g = w[..., 1] - w[..., 0]
        ws, Vs = np.linalg.eigh(Ms[..., 1:, 1:])
        kd = kin_from_jets(jets_in_frame(M, V, h, "diag"), a0, h); ko = kin_from_jets(jets_in_frame(M, V, h, "off"), a0, h)
        eud = e_u_from_jets(jets_in_frame(M, V, h, "diag"), h)
        # S2 density in the local frame: kin_diag == h^3 sum_br wt sum_i 8 g^2 (a2_i - a3_i)^2
        s2 = 0.0
        for br in ("fwd", "bwd"):
            for Ai in jets(M, h, br):
                B = np.einsum("...ki,...kl,...lj->...ij", V, Ai[..., 1:, 1:], V)
                s2 += 0.5 * float(np.sum(8.0 * (g * inside) ** 2 * (B[..., 1, 1] - B[..., 0, 0]) ** 2))
        s2 *= h ** 3
        flips = []
        for ax in range(3):
            dg = np.diff(g, axis=ax); sgn = np.sign(dg)
            s0 = tuple(slice(0, -1) if a_ == ax else slice(None) for a_ in range(3)); s1 = tuple(slice(1, None) if a_ == ax else slice(None) for a_ in range(3))
            fl = flank[tuple(slice(1, -1) if a_ == ax else slice(None) for a_ in range(3))]
            flips.append(float(((sgn[s0] * sgn[s1])[fl] < 0).mean()))
        n1e, n1s = V[..., :, -1], Vs[..., :, -1]
        tilt = np.degrees(np.arccos(np.clip(np.abs(np.einsum("...i,...i->...", n1e, n1s)), 0, 1)))
        e2e, e2s, e3s = V[..., :, 1], Vs[..., :, 1], Vs[..., :, 0]
        phi = np.degrees(np.arctan2(np.abs(np.einsum("...i,...i->...", e2e, e3s)), np.abs(np.einsum("...i,...i->...", e2e, e2s))))
        rows[key] = {"kin": k, "frac_flank": float(dens[flank].sum() / k), "frac_core": float(dens[core].sum() / k), "kin_diag_only": kd, "kin_off_only": ko,
                     "E_u_diag_only": eud, "S2_identity": s2, "flip_frac": flips, "gap_max_flank": float(g[flank].max()), "gap_p95_flank": float(np.percentile(g[flank], 95)),
                     "d3_min_flank": float(w[flank][:, 0].min()), "n1_tilt_median_deg": float(np.median(tilt[flank])), "pair_rot_median_deg": float(np.median(phi[flank])),
                     "seed_gap_max_flank": float((ws[..., 1] - ws[..., 0])[flank].max())}
    infl = {k: v["kin"] / recs[k]["start"]["kin_seed_two_region"] for k, v in rows.items()}
    hot = [k for k in rows if infl[k] > 10]; cold = [k for k in rows if infl[k] <= 10]
    check(sec, "WHERE: in every run whose kin inflated more than 10x (6 of 7 checked) 99 to 100 percent of the two-region inertia sits in the two cell layers just inside the mask edge (R_s - 2h <= r < R_s) and the core (r < R_s - 2h) carries under 1 percent; the one weakly inflated run (R_s 15 J 50, 1.7x) keeps the seed-like split (67 percent flank)",
          all(rows[k]["frac_flank"] > 0.99 and rows[k]["frac_core"] < 0.01 for k in hot) and len(hot) == 6 and cold == ["w3_n32_L48_R15_J50_it3000"] and 0.6 < rows[cold[0]]["frac_flank"] < 0.75,
          "; ".join(f"{k[3:]}: {infl[k]:.0f}x, flank {v['frac_flank']:.3f} core {v['frac_core']:.3f}" for k, v in rows.items()))
    check(sec, "WHAT: in the inflated runs the inertia is carried by EIGENVALUE jets (the diag part of the jets in the local eigenframe), 97 to 99.7 percent of kin, not by orientation jets (the S5 (1,2)-twist channel, 0.3 to 3 percent); the weakly inflated run is half and half",
          all(rows[k]["kin_diag_only"] / rows[k]["kin"] > 0.97 for k in hot) and 0.4 < rows[cold[0]]["kin_diag_only"] / rows[cold[0]]["kin"] < 0.6,
          "; ".join(f"{k[3:]}: diag {v['kin_diag_only'] / v['kin']:.3f} off {v['kin_off_only'] / v['kin']:.3f}" for k, v in rows.items()))
    check(sec, "the diag-jet inertia IS the W0 S2 flank density in the local frame, kin = h^3 sum 8 g^2 (d_i d2 - d_i d3)^2, to 1e-9 on every field (so the read is proportional to the GROWN gap squared: the refreshed generator rewards raising the gap where the ramp is)",
          all(rel(v["S2_identity"], v["kin_diag_only"]) < 1e-9 for v in rows.values()), f"max rel dev {max(rel(v['S2_identity'], v['kin_diag_only']) for v in rows.values()):.1e}")
    check(sec, "eigenvalue jets in a fixed local frame are INVISIBLE to E_u ([diag, diag] = 0): E_u from the diag-only jets is exactly zero on every field; the ramps are paid only through V4 and the cross term with the hedgehog frame gradient",
          all(abs(v["E_u_diag_only"]) < 1e-9 for v in rows.values()), f"max |E_u diag-only| {max(abs(v['E_u_diag_only']) for v in rows.values()):.1e}")
    check(sec, "the structure is a CELL-SCALE ZIGZAG, not a smooth ramp: 61 to 73 percent of consecutive gap differences along each axis flip sign in the flank on every run (a smooth ramp gives near 0, random 0.5), with the (2,3) gap reaching 1.7 to 2.9 against 0.30 in the vacuum (seed flank max 0.25 to 0.30) and d3 going NEGATIVE down to -0.7 to -1.6 (the spatial block loses positivity) in every J >= 200 run; at J 50 the gap stays under 0.4",
          all(min(v["flip_frac"]) > 0.6 for v in rows.values()) and all(v["gap_max_flank"] > 1.5 and v["d3_min_flank"] < -0.5 for k, v in rows.items() if "J50" not in k) and rows[cold[0]]["gap_max_flank"] < 0.4,
          "; ".join(f"{k[3:]}: flips {[round(f, 2) for f in v['flip_frac']]} gap max {v['gap_max_flank']:.2f} (seed {v['seed_gap_max_flank']:.2f}) d3 min {v['d3_min_flank']:.2f}" for k, v in rows.items()))
    check(sec, "NOT the S5 orientation twist: the leading eigenvector tilts under 5 degrees (median 0.3 to 4.7) from the seed's and the (2,3) pair rotates under 10 degrees (median 0.1 to 8.2) in the flank on every run: the frame stayed, the eigenvalues zigzagged",
          all(v["n1_tilt_median_deg"] < 5 and v["pair_rot_median_deg"] < 10 for v in rows.values()), "; ".join(f"{k[3:]}: tilt {v['n1_tilt_median_deg']:.1f} pair {v['pair_rot_median_deg']:.1f} deg" for k, v in rows.items()))
    fz = rows["w3_n32_L48_R9_J200_it3000_frz"]
    check(sec, "the frozen-a0 descent builds the SAME zigzag (99 percent of its refreshed-read inertia in the flank, diag-jet share 98 percent); its frozen generator reads 637 of it, the refreshed generator 7542: the factor 12 is the grown gap entering a0 (S1: a0 = (d2 - d3)(E23 + E32)) squared",
          fz["frac_flank"] > 0.98 and fz["kin_diag_only"] / fz["kin"] > 0.95 and abs(fz["kin"] - 7542.19) < 0.5 and abs(recs["w3_n32_L48_R9_J200_it3000_frz"]["end"]["kin"] - 637.41) < 0.05,
          f"frz refreshed {fz['kin']:.1f} frozen {recs['w3_n32_L48_R9_J200_it3000_frz']['end']['kin']:.1f}")
    note(sec, "rows", rows)


def audit_w3_F():
    sec = "F"
    recs = w3_records()
    mine = {J: w3_end_numbers(f"w3_n32_L48_R9_J{J:g}_it3000") for J in (180.0, 200.0, 220.0)}
    dEdJ = (mine[220.0]["E_J"] - mine[180.0]["E_J"]) / 40.0
    om = mine[200.0]["omega"]
    closure = json.load(open(os.path.join(DATA, "m5_32_r13w_w3.json")))["summary"]["dEdJ_closure_n32_R9"]
    check(sec, "dE/dJ central difference over J = 180/200/220 recomputed from the fields: 0.01871 vs omega(200) = 0.03705, 49.5 percent low, matching the producer's numbers",
          rel(dEdJ, closure["dE_dJ_central"]) < 1e-9 and rel(om, closure["omega_at_200"]) < 1e-9 and abs(dEdJ / om - 0.505) < 0.01, f"dE/dJ {dEdJ:.6f} omega {om:.6f} ratio {dEdJ / om:.3f}")
    Es = {J: mine[J]["E_u"] + mine[J]["V4"] for J in mine}; Ek = {J: J * J / (4 * mine[J]["kin"]) for J in mine}
    dstat = (Es[220.0] - Es[180.0]) / 40.0; dkin = (Ek[220.0] - Ek[180.0]) / 40.0
    fixed_field = (220.0 ** 2 - 180.0 ** 2) / (4 * mine[200.0]["kin"] * 40.0)
    check(sec, "decomposition: the fixed-FIELD derivative is exactly omega (E_J is quadratic in J at fixed M: 0.03705); the measured difference splits into dE_stat/dJ = +0.0090 and d(J^2/4kin)/dJ = +0.0097, the latter cut from 0.037 by kin rising 2573 -> 2699 -> 3418 across the J ladder (a 33 percent spread) at a fixed iteration count",
          rel(fixed_field, om) < 1e-12 and abs(dstat - 0.0090) < 0.0005 and abs(dkin - 0.0097) < 0.0005 and abs(dstat + dkin - dEdJ) < 1e-12,
          f"fixed-field {fixed_field:.5f}; dE_stat/dJ {dstat:.5f}; d(J^2/4kin)/dJ {dkin:.5f}; kin {[round(mine[J]['kin']) for J in (180.0, 200.0, 220.0)]}")
    a, b = recs["w3_n32_L48_R9_J200_it3000"], recs["w3_n32_L48_R9_J200_it12000"]
    floor = abs(a["end"]["E_J"] - b["trace"][11]["E"]) / 40.0
    check(sec, "the mismatch is what non-convergence predicts, not roundoff: the same-J reproducibility floor puts 0.0016 on dE/dJ (10x below the 0.018 deficit); the Hellmann-Feynman identity dE/dJ = omega needs a stationary field, and fmax at the end is 0.33 to 0.44 against the packet's 1e-6; a larger J descends kin faster at fixed iterations (kin 3418 at 220 vs 2699 at 200), which lowers the finite difference below omega, the observed sign",
          floor < 0.003 and all(recs[f"w3_n32_L48_R9_J{J:g}_it3000"]["trace"][-1]["fmax"] > 0.1 for J in (180.0, 200.0, 220.0)) and mine[220.0]["kin"] / mine[200.0]["kin"] > 1.2 and dEdJ < om,
          f"floor {floor:.4f}; fmax_end {[f'{recs[f'w3_n32_L48_R9_J{J:g}_it3000']['trace'][-1]['fmax']:.2f}' for J in (180.0, 200.0, 220.0)]}")
    note(sec, "numbers", {"dEdJ": dEdJ, "omega_200": om, "dstat": dstat, "dkin": dkin, "E_J": {str(J): mine[J]["E_J"] for J in mine}, "kin": {str(J): mine[J]["kin"] for J in mine}})


def audit_w3_G():
    sec = "G"
    recs = w3_records()
    pts = {"n32_L48_h1.5": w3_end_numbers("w3_n32_L48_R9_J200_it3000"), "n48_L48_h1.0": w3_end_numbers("w3_n48_L48_R9_J200_it3000"), "n48_L72_h1.5": w3_end_numbers("w3_n48_L72_R9_J200_it3000")}
    om = {k: v["omega"] for k, v in pts.items()}
    p_h = np.log(om["n48_L48_h1.0"] / om["n32_L48_h1.5"]) / np.log(1.0 / 1.5)
    r_L = om["n48_L72_h1.5"] / om["n32_L48_h1.5"]
    check(sec, "omega at R_s 9 J 200 after 3000 iterations: 0.0370 (h 1.5, L 48), 0.0184 (h 1.0, L 48), 0.0353 (h 1.5, L 72): the h-dependence is omega ~ h^1.7 (between h and h^2, not h), the L-dependence at fixed h is 5 percent (box-insensitive): it scales like the lattice, not the box, and like neither pre-registered form exactly",
          1.5 < p_h < 1.9 and 0.9 < r_L < 1.0, f"omega {om}; h-exponent {p_h:.2f}; L72/L48 ratio {r_L:.3f}")
    # the seed confound: the two-region kin at R_s 9 of the seeds actually used, plus the n32 3000-iteration R10 state (same h, same maturity as the n48 seeds)
    h32, r32 = w3_geom(32, 48.0)
    ks = {"n32_it12000 (used)": my_kin_cells(w3_seed(32, 48.0), a0_two_region(w3_seed(32, 48.0), r32 < 9.0), h32)[1],
          "n32_it3000 (R10, unused)": my_kin_cells(w3_seed(32, 48.0, 3000), a0_two_region(w3_seed(32, 48.0, 3000), r32 < 9.0), h32)[1]}
    for n, L in ((48, 48.0), (48, 72.0)):
        h, rad = w3_geom(n, L); Ms = w3_seed(n, L)
        ks[f"n{n}_L{L:g}_it3000 (used)"] = my_kin_cells(Ms, a0_two_region(Ms, rad < 9.0), h)[1]
    check(sec, "SEED CONFOUND quantified: the n48 runs start from 3000-iteration seeds with two-region kin 40.2 (h 1.0) and 40.7 (h 1.5) at R_s 9, the n32 run from R10's 12000-iteration state with 22.2: a factor 1.83 in the starting inertia (E_u 13.7 to 14.8 vs 9.05); the n32 3000-iteration R10 state has 40.690, IDENTICAL to the n48 L72 seed's 40.689, so the inner field is box-independent and the L48-vs-L72 comparison is confounded by seed maturity ONLY",
          abs(ks["n32_it3000 (R10, unused)"] - ks["n48_L72_it3000 (used)"]) < 1e-2 and abs(ks["n48_L72_it3000 (used)"] / ks["n32_it12000 (used)"] - 1.83) < 0.02,
          "; ".join(f"{k}: {v:.3f}" for k, v in ks.items()))
    ratio_end = pts["n48_L72_h1.5"]["kin"] / pts["n32_L48_h1.5"]["kin"]
    check(sec, "what the confound does at the end: the seed-kin ratio 1.83 shrinks to 1.05 in the relaxed kin (2836 vs 2699), so the 3000-iteration end state is dominated by what the descent built, not by the seed; the clean same-seed L comparison is the w3_ctrl run (CLI word w3_ctrl, 28 minutes)",
          1.0 < ratio_end < 1.1, f"end kin ratio L72/L48 {ratio_end:.3f}")
    prev = json.load(open(OUT)).get("w3_ctrl", {}).get("G_ctrl", {}) if os.path.exists(OUT) else {}
    if "numbers" in prev:
        c = prev["numbers"]
        note(sec, "ctrl_seed3000_n32", c)
        check(sec, "w3_ctrl (n32 L48 R_s 9 J 200, 3000 iterations from the 3000-iteration R10 seed, same maturity as the n48 seeds): the omega read with the SAME seed maturity across L 48 / 72 at h 1.5",
              True, f"ctrl omega {c['omega']:.5f} kin {c['kin']:.1f} E_J {c['E_J']:.3f} vs n48 L72 omega {om['n48_L72_h1.5']:.5f} kin {pts['n48_L72_h1.5']['kin']:.1f}; n32 12000-seed omega {om['n32_L48_h1.5']:.5f}")
    note(sec, "omega", om); note(sec, "seed_kin_R9", ks)


def audit_w3_H():
    sec = "H"
    coll = json.load(open(os.path.join(DATA, "m5_32_r13w_w3.json"))); summ = coll["summary"]
    src = open(os.path.join(HERE, "m5_32_r13w_w3.py")).read()
    vb = src.split("# verdict")[1].split("res[\"summary\"] = summ")[0]
    recs = w3_records()
    ladders = {k: v for k, v in summ.items() if k.startswith("Rstar_")}
    # (1) the emitted verdict and the gates it rests on, recounted here
    stops = {k: r["stop"] for k, r in recs.items()}
    lq = {k: r["trace"][-1]["kin"] / r["trace"][max(0, 3 * len(r["trace"]) // 4 - 1)]["kin"] - 1.0 for k, r in recs.items()}
    check(sec, "the collected verdict (collect() of 15:54) is the 'no stationary point' FAIL: its two computed gates hold on a recount (0 of 20 runs stopped on f_tol or plateau; kin grew in the last quarter of all 20, min +4 percent), and PERIODIC_ORBIT_EXISTS is withheld: the verdict's headline does not overreach",
          summ["verdict"].startswith("W3 FAIL: no fixed-J relaxation reached a stationary point") and summ["stationary_runs"] == 0 and all(v == "max_iter" for v in stops.values()) and all(v > 0 for v in lq.values()) and summ["kin_growing_last_quarter_all_runs"],
          f"stops {set(stops.values())}; last-quarter growth min {min(lq.values()):.3f}")
    # (2) the verdict body is a template: only one word is computed
    literals = ["mean gap 0.03 to 0.14", "dE/dJ != omega", "the 12000-iteration run included", "the free-inertia theorem (W0 S5) is the mechanism", "inner flank"]
    check(sec, "the verdict BODY is a template, not a gate read-out: 'mean gap 0.03 to 0.14', 'dE/dJ != omega', 'the 12000-iteration run included', 'inner flank' and 'the free-inertia theorem (W0 S5) is the mechanism' are string literals in collect(); the only computed word is INCREASES/is not monotone; the closure result (rel_dev) and the shell statistics never enter the branch condition, so those five statements could not have come out differently whatever the data said",
          all(lit in vb for lit in literals) and "rel_dev" not in vb and "gap_shell_mean" not in vb and '"INCREASES" if' in vb, f"literals found {sum(lit in vb for lit in literals)}/5; branch reads rel_dev: {'rel_dev' in vb}; reads gap_shell_mean: {'gap_shell_mean' in vb}")
    # (3) the branch structure: reimplemented verbatim and probed
    def verdict(stationary_n, growing, tracks_all, survives):
        if stationary_n == 0 and growing:
            return "FAIL_no_stationary"
        elif tracks_all:
            return "FAIL_tracks_box"
        elif stationary_n and not survives:
            return "FAIL_melts"
        return "stationary states reached with the shell surviving"
    order_ok = vb.index("if not stationary and growing") < vb.index("elif tracks and all(tracks)") < vb.index("elif stationary and not summ") < vb.index("else:")
    hole = verdict(0, False, False, True)
    close_call = min(lq.values())
    check(sec, "the else branch asserts 'stationary states reached with the shell surviving' WITHOUT checking stationarity: with zero stationary runs it fires as soon as ONE run's last-quarter growth is <= 0 (the R_s 15 J 50 run sits at +4.4 percent), flipping the verdict from FAIL to a pass-shaped sentence on a dip of one run: a verdict that does not follow from the gates",
          order_ok and hole.startswith("stationary states reached") and close_call < 0.05, f"branch order ok {order_ok}; verdict(stationary=0, growing=False, tracks=False, survives=True) = {hole!r}; closest run last-quarter growth {close_call:.3f}")
    check(sec, "the 'R* tracks the box' branch is blind to the observed pattern: at_box_edge flags only the UPPER grid edge, while every ladder's argmin sits at the LOWER grid edge (R* = min R_s in 4 of 4 ladders) with E_J monotone increasing; no branch tests 'R* interior' (the pass condition), so R* can neither pass nor fail by code",
          all(v["R_star"] == min(v["Rs"]) and not v["at_box_edge"] for v in ladders.values()) and summ["E_J_increasing_with_Rs_all_ladders"] and "R_star" not in vb and "argmin" not in vb,
          "; ".join(f"{k}: R* {v['R_star']} of {v['Rs']}" for k, v in ladders.items()))
    mean_crit = all(r["end"]["gap_shell_mean"] < 0.03 for r in recs.values())
    r12 = recs["w3_n32_L48_R9_J200_it12000"]
    check(sec, "'shell_survives_all_runs' = all(min over the shell < 0.03) is a ONE-CELL criterion: it stays True at 12000 iterations on 6 of 416 cells with the shell mean at 0.131; the same flag on the MEAN (< 0.03) is False for every run; the melt branch (and the else branch's 'shell surviving') can fail only when the last degenerate cell melts",
          summ["shell_survives_all_runs"] and not mean_crit and r12["end"]["n_cells_gap_lt_0.03"] == 6 and r12["end"]["gap_shell_mean"] > 0.13, f"mean-criterion {mean_crit}; 12000-run n<0.03 {r12['end']['n_cells_gap_lt_0.03']}, mean {r12['end']['gap_shell_mean']:.3f}")
    fm = [r["trace"][-1]["fmax"] for r in recs.values()]
    check(sec, "no run reached a stationary point (min fmax at the end 0.044, packet f_tol 1e-6; all 20 stop = max_iter) and the ladders ran 3000 iterations, a quarter of the packet's 12000: every W3 read is of a state still descending, so 'R* at the smallest radius' is a statement about descent RATE versus R_s at a fixed iteration count, not a bag statement",
          min(fm) > 1e-2 and all(r["stop"] == "max_iter" for r in recs.values()) and sum(1 for r in recs.values() if r["maxit"] == 3000) == 19, f"fmax_end min {min(fm):.3f} max {max(fm):.2f}; stops {set(r['stop'] for r in recs.values())}")
    # (4) the mechanism attribution
    M = w3_field("w3_n32_L48_R9_J200_it3000"); h, rad = w3_geom(32, 48.0); Ms = w3_seed(32, 48.0)
    inside = rad < 9.0; a0 = a0_two_region(M, inside)
    w, V = np.linalg.eigh(M[..., 1:, 1:]); ws, Vs = np.linalg.eigh(Ms[..., 1:, 1:])
    k = my_kin_cells(M, a0, h)[1]; kd = kin_from_jets(jets_in_frame(M, V, h, "diag"), a0, h)
    flank = (rad >= 6.0) & (rad < 9.0)
    phi = np.degrees(np.arctan2(np.abs(np.einsum("...i,...i->...", V[..., :, 1], Vs[..., :, 0])), np.abs(np.einsum("...i,...i->...", V[..., :, 1], Vs[..., :, 1]))))
    check(sec, "'the free-inertia theorem (W0 S5) is the mechanism' MIS-ATTRIBUTES: S5 as proven in W0 is the (1,2) ORIENTATION twist at zero static cost; what the descent built is an E_u-blind EIGENVALUE zigzag of the (2,3) gap (97 percent of kin in diag jets at R_s 9 J 200, the pair frame rotated 2.8 degrees median), the W0 S2 flank density with the gap driven to 1.9 and read by a generator that grows with it: a sibling lattice mechanism, paid through V4 and the frame cross term, not S5's twist",
          kd / k > 0.95 and np.median(phi[flank]) < 5 and "S5" in summ["verdict"], f"diag share {kd / k:.3f}; pair rotation median {np.median(phi[flank]):.1f} deg")
    check(sec, "the closure gate FAILS as measured (dE/dJ 50 percent below omega) and the shell gate is at best partial (mean gap 0.03 to 0.14 against 0 imposed, melting progressively): under the frozen vocabulary W3 cannot return PERIODIC_ORBIT_EXISTS; the supportable verdict is the fail line, ESTABLISHED_KINEMATIC at best (from W1/W2), with 'no stationary point on the fixed-J functional with a refreshed generator' as the fail mode, which the 15:54 verdict states",
          abs(summ["dEdJ_closure_n32_R9"]["rel_dev"] + 0.495) < 0.01 and summ["kin_monotone_all_runs"] is False and "ESTABLISHED_KINEMATIC at best" in summ["verdict"],
          f"rel_dev {summ['dEdJ_closure_n32_R9']['rel_dev']:.3f}; kin_monotone_all_runs {summ['kin_monotone_all_runs']}")


def audit_w3_I():
    sec = "I"
    B8 = _load("m5_21_8_b_lattice", "m5_21_8_b_lattice.py")
    h, rad = w3_geom(32, 48.0)
    sh = np.abs(rad - 9.0) < 0.5 * h
    costs, kins, m00 = {}, {}, {}
    for g in (8.0, 32.0, 100.0):
        cfg = INS4.base_cfg(s=S, g=g, n=32, L=48.0, delta=DELTA)
        D = B8.dressed(cfg, 0.0); P = deg_project(D, sh)
        costs[g] = my_v4_g(P, h, g) - my_v4_g(D, h, g)
        kins[g] = my_kin_cells(P, a0_two_region(P, rad < 9.0), h)[1]
        m00[g] = (float(D[..., 0, 0].min()), float(D[..., 0, 0].max()), float(np.abs(D[..., 1:, 1:] - B8.dressed(INS4.base_cfg(s=S, g=8.0, n=32, L=48.0, delta=DELTA), 0.0)[..., 1:, 1:]).max()))
    g32 = json.load(open(os.path.join(DATA, "m5_32_r13w_w3.json")))["summary"]["g32_control_on_the_ansatz"]
    check(sec, "g = 32 shell numbers reproduce with own V4 constants: V4 cost of the shell 2.5264e-3 and two-region kin 124.33 at g = 8 and g = 32",
          rel(costs[8.0], g32["V4_cost_of_shell_g8"]) < 1e-6 and rel(costs[32.0], g32["V4_cost_of_shell_g32"]) < 1e-6 and rel(kins[32.0], g32["kin_two_region_g32"]) < 1e-9, f"costs {costs}; kin {kins}")
    check(sec, "the control is an IDENTITY, not a measurement: on the unrelaxed ansatz the g = 8 and g = 32 fields have bit-identical spatial blocks and M00 = g exactly, the projection and the spatial generator never touch M00, and t_p - c_p = sum_k d_k^p - 1 - delta^p is g-free when M00 = g; the shell cost and kin are equal at g = 100 too (kin to 1e-15, the cost to 3e-8, the roundoff of traces of size 1e8): the check cannot fail on this ansatz, a g control with content needs the RELAXED g = 32 field (the owed PAUSE RECORD 3 item)",
          all(v[2] == 0.0 and v[0] == v[1] == g for g, v in m00.items()) and rel(costs[100.0], costs[8.0]) < 1e-6 and rel(kins[100.0], kins[8.0]) < 1e-12,
          f"M00 (min, max, spatial-block diff vs g8) {m00}; cost g100 {costs[100.0]:.6e}")
    Ms = w3_seed(32, 48.0)
    check(sec, "on the RELAXED g = 8 seed M00 is not constant (8.0 to 8.00041), so the identity does not transfer: the relaxed shell cost depends on g through (M00 - g); no relaxed g = 32 field exists in W3",
          float(Ms[..., 0, 0].max()) - 8.0 > 1e-4 and not os.path.exists(os.path.join(CK, "seed_n32_L48_g32_it3000.npy")), f"relaxed M00 range {Ms[..., 0, 0].min():.6f} to {Ms[..., 0, 0].max():.6f}")
    note(sec, "costs", costs); note(sec, "kins", kins)


def audit_w3_J():
    sec = "J"
    recs = w3_records()
    coll = json.load(open(os.path.join(DATA, "m5_32_r13w_w3.json")))
    missing = [k for k, r in recs.items() if "kin_with_seed_a0" not in r["end"]]
    check(sec, "record schema drift: 19 of 20 checkpoint records (all but the frozen run) predate the kin_with_seed_a0 / E_J_with_seed_a0 fields, run() restores cached records by key without a schema check, and collect()'s 'E_J_refreshed_vs_seed_a0_on_end_fields' is silently EMPTY: the J3 cross-check the producer advertises was never computed for the ladders (section A supplies it)",
          len(missing) == 19 and coll["summary"]["E_J_refreshed_vs_seed_a0_on_end_fields"] == {}, f"records without the fields: {len(missing)}; collected dict: {coll['summary']['E_J_refreshed_vs_seed_a0_on_end_fields']}")
    # the shell straddles the mask edge
    rows = {}
    for key in ["w3_n32_L48_R9_J200_it3000", "w3_n32_L48_R12_J800_it3000", "w3_n32_L48_R15_J800_it3000", "w3_n48_L72_R9_J200_it3000", "w3_n32_L48_R15_J50_it3000"]:
        r = recs[key]; n, L, Rs = r["n"], r["L"], r["Rs"]
        M = w3_field(key); h, rad = w3_geom(n, L)
        shell = np.abs(rad - Rs) < 0.5 * h; inside = rad < Rs
        dens, k = kin_dens(M, a0_two_region(M, inside), h)
        _, klo = my_kin_cells(M, a0_two_region(M, rad < Rs - 0.5 * h), h)
        _, khi = my_kin_cells(M, a0_two_region(M, rad < Rs + 0.5 * h), h)
        rows[key] = {"shell_frac_inside_mask": float((shell & inside).sum() / shell.sum()), "kin_frac_from_shell_in_mask": float(dens[shell & inside].sum() / k), "kin_mask_minus_h2": klo / k, "kin_mask_plus_h2": khi / k}
    check(sec, "the mask edge r < R_s cuts the one-cell shell in HALF (43 to 55 percent of the shell cells lie inside the mask): harmless while the shell is degenerate (a0 = 0 there, S1) but after the partial melt the inside half carries 0.5 to 16 percent of kin (16 percent at J 800); moving the edge to R_s - h/2 changes kin by up to 16 percent, to R_s + h/2 by under 1 percent: a hard mask on a nonzero a0, the deviation the docstring itself names",
          all(0.4 < v["shell_frac_inside_mask"] < 0.6 for v in rows.values()) and max(v["kin_frac_from_shell_in_mask"] for v in rows.values()) > 0.14 and all(v["kin_mask_plus_h2"] < 1.01 for v in rows.values()) and min(v["kin_mask_minus_h2"] for v in rows.values()) < 0.86,
          "; ".join(f"{k[3:]}: shell-in-mask {v['shell_frac_inside_mask']:.2f}, kin from it {v['kin_frac_from_shell_in_mask']:.3f}, mask -h/2 {v['kin_mask_minus_h2']:.3f} +h/2 {v['kin_mask_plus_h2']:.3f}" for k, v in rows.items()))
    # radial_profiles: bins stop at L/2 + h; the corner cells are dropped; does the kin sum still close?
    r = recs["w3_n32_L48_R9_J200_it3000"]; h, rad = w3_geom(32, 48.0)
    prof = r["end"]["profiles"]; covered = sum(p["cells"] for p in prof); dropped = 32 ** 3 - covered
    ksum = sum(p["kin_bin"] for p in prof)
    check(sec, "radial_profiles() covers r < L/2 only (np.arange(0, L/2 + dr, dr) excludes its endpoint, so the last bin ends at 24.0, not at L/2 + h as the code intends): 15512 of 32768 cells (47 percent, everything at r >= 24.0 out to the corners at 40.3) are silently dropped from every profile; harmless for kin (the mask zeroes them, kin_bin sums close to 1e-9) and for the gap (pinned or vacuum there), so the profiles are complete where they are used",
          dropped == int((rad >= 24.0).sum()) and dropped == 15512 and prof[-1]["r_hi"] == 24.0 and rel(ksum, r["end"]["kin"]) < 1e-9, f"dropped {dropped}; last bin ends {prof[-1]['r_hi']}; kin_bin sum {ksum:.6f} vs kin {r['end']['kin']:.6f}")
    check(sec, "collect() grouping: the ladders take maxit == 3000 only (the 12000-iteration run excluded), the frozen run is excluded via get('frozen') which is None on the 19 old records (correct by accident of the default), the closure uses the three 3000-iteration J runs, and the frozen run's kin_start is the shelled seed's two-region kin: all as intended",
          all(r["maxit"] == 3000 for k, r in recs.items() if k != "w3_n32_L48_R9_J200_it12000") and coll["summary"]["frozen_a0_control"][0]["kin_start"] == recs["w3_n32_L48_R9_J200_it3000_frz"]["start"]["kin_shell_two_region"] and len(coll["runs"]) == 19,
          f"collected runs {len(coll['runs'])} (19 non-frozen), ladders exclude the 12000 run")
    tr = recs["w3_n32_L48_R9_J200_it3000"]["trace"]
    check(sec, "trace consistency: the row 'omega' equals J/(2 kin) and the row 'E' equals E_u + V4 + J^2/(4 kin) with the refreshed kin on every row (the reported E_J and omega are Legendre-consistent with each other, if not with the descent)",
          all(abs(row["omega"] - 200.0 / (2 * row["kin"])) < 1e-12 and abs(row["E"] - (row["E_u"] + row["V4"] + 200.0 ** 2 / (4 * row["kin"]))) < 1e-9 for row in tr), "12 rows")
    m32 = [k for k in recs if recs[k]["n"] == 32 and recs[k]["seed"].get("source", "").startswith("m5_32_r10")]
    check(sec, "seed caching as documented: all 16 n32 runs use R10's 12000-iteration state (seed.source), the 4 n48 runs the 3000-iteration seeds (seed.maxit 3000): the two ladders are on different seed maturities (section G)",
          len(m32) == 16 and all(recs[k]["seed"].get("maxit") == 3000 for k in recs if recs[k]["n"] == 48), f"n32 from R10: {len(m32)}; n48 seeds maxit 3000")
    note(sec, "mask_rows", rows)


def audit_w3_ctrl():
    """the one control W3 did not run: n32 L48 R_s 9 J 200, 3000 iterations, from R10's 3000-iteration seed
    (the same maturity as the n48 seeds), so L 48 vs 72 at h 1.5 is compared at equal seed maturity.
    fire_proj is the producer's (the object under test); masks, generator, energies are the audit's own."""
    sec = "G_ctrl"
    C = _load("m5_32_r13w_common", "m5_32_r13w_common.py")
    n, L, Rs, J, maxit = 32, 48.0, 9.0, 200.0, 3000
    cfg = cfg_of(n, L); h, rad = w3_geom(n, L)
    shell = np.abs(rad - Rs) < 0.5 * h; inside = rad < Rs
    seed = w3_seed(n, L, 3000)
    M0 = deg_project(seed, shell)
    free = ~INS4.pin_shell(n, h)
    a0_of = lambda M: a0_two_region(M, inside)
    k0 = my_kin_cells(M0, a0_of(M0), h)[1]
    M, info = C.fire_proj(M0, cfg, free, maxit, project=None, J=J, a0_of=a0_of, tag="audit_ctrl_seed3000", log_every=250)
    eu, ev = my_e_u(M, h), my_v4(M, h); k = my_kin_cells(M, a0_of(M), h)[1]
    g = gap23(M)
    kins = [row["kin"] for row in info["trace"]]
    num = {"seed": "m5_32_r10 relax_g8_n32_L48_it3000.npy", "kin_start": k0, "kin": k, "omega": J / (2 * k), "E_u": eu, "V4": ev, "E_J": eu + ev + J * J / (4 * k),
           "gap_shell_min": float(g[shell].min()), "gap_shell_mean": float(g[shell].mean()), "stop": info["stop"], "iters": info["iters"], "wall_s": info["wall_s"],
           "kin_last_quarter_growth": kins[-1] / kins[max(0, 3 * len(kins) // 4 - 1)] - 1.0, "trace": info["trace"]}
    note(sec, "numbers", num)
    ref = w3_records()
    om48 = ref["w3_n48_L72_R9_J200_it3000"]["end"]["omega"]; om32 = ref["w3_n32_L48_R9_J200_it3000"]["end"]["omega"]
    check(sec, "control ran to max_iter with kin still growing in the last quarter (the runaway is seed-independent)", info["stop"] == "max_iter" and num["kin_last_quarter_growth"] > 0.05,
          f"kin {k0:.1f} -> {k:.1f}, last-quarter growth {num['kin_last_quarter_growth']:.3f}")
    check(sec, "same-seed-maturity L comparison at h 1.5: omega(L48, 3000-it seed) vs omega(L72, 3000-it seed) within 25 percent, and the 12000-it-seed omega within the same band: the box is not the control variable of omega at 3000 iterations",
          abs(num["omega"] / om48 - 1) < 0.25 and abs(om32 / om48 - 1) < 0.25, f"omega ctrl {num['omega']:.5f} vs n48 L72 {om48:.5f} vs n32 12000-seed {om32:.5f}")


SECTIONS_W3 = {"A": audit_w3_A, "B": audit_w3_B, "C": audit_w3_C, "D": audit_w3_D, "E": audit_w3_E, "F": audit_w3_F,
               "G": audit_w3_G, "H": audit_w3_H, "I": audit_w3_I, "J": audit_w3_J}


def run(keys):
    for k in keys:
        print(f"--- audit_{k} [{time.time() - T0:.1f}s]", flush=True)
        try:
            SECTIONS[k]()
        except Exception as ex:          # a crashed section is a FAIL, not a silent skip
            check(k, f"section {k} crashed", False, f"{type(ex).__name__}: {ex}")
    n_all = sum(1 for s in RESULTS.values() for v in s.values() if isinstance(v, dict) and "pass" in v)
    n_ok = sum(1 for s in RESULTS.values() for v in s.values() if isinstance(v, dict) and v.get("pass") is True)
    RESULTS["summary"] = {"pass": n_ok, "checks": n_all, "lines": LINES, "runtime_s": round(time.time() - T0, 1)}
    print(f"AUDIT {'+'.join(keys)}: {n_ok}/{n_all} PASS [{time.time() - T0:.1f}s]")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "w012"
    if mode == "w012":
        keys = [k for k in sys.argv[2:] if k in SECTIONS and k != "R"] or [k for k in SECTIONS if k != "R"]
    elif mode == "w012_recheck":
        keys = ["R"]
    elif mode == "w3":
        for k_, f_ in SECTIONS_W3.items():
            SECTIONS["w3" + k_] = f_
        keys = ["w3" + k for k in sys.argv[2:] if k in SECTIONS_W3] or ["w3" + k for k in SECTIONS_W3]
    elif mode == "w3_ctrl":
        SECTIONS["ctrl"] = audit_w3_ctrl
        keys = ["ctrl"]
    else:
        raise SystemExit("modes: w012 [A B ...] | w012_recheck | w3 [A B ...] | w3_ctrl")
    run(keys)
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    allres[mode] = RESULTS
    json.dump(allres, open(OUT, "w"), indent=1, default=float)
    print(f"written {OUT} key '{mode}'")
