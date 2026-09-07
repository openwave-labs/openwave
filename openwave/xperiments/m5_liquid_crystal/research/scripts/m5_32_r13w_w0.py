"""M5.32 R13-W, step W0: the symbolic closure of the degenerate-wall clock
convention on the certified action, BEFORE any lattice number.

The candidate (the model author, 2026-08-31): 3D regions of constant clock
frequency omega separated by 2D walls on which the last two eigenvalues of
M equalize (d2 = d3), so that the (2,3)-plane isorotation acts trivially on
the wall and the two phases decouple there.

EQUATIONS FIRST (the certified conventions, m5_21_3_a_4d.py / m5_32_lagrangian.py)
-----------------------------------------------------------------------------
eta = diag(-1, 1, 1, 1);  [A, B]_eta = A eta B - B eta A;
<F, G>_eta = tr(eta F eta G^T) = sum_ab eta_a eta_b F_ab G_ab;
jets A_mu = d_mu M, A_0 = omega a0;  F_mu nu = [A_mu, A_nu]_eta;
E = 4 h^3 sum_x [ sum_{i<j} <F_ij, F_ij>_eta + omega^2 sum_i <F_0i, F_0i>_eta ]
    + h^3 W1 sum_x sum_p (tr((M eta)^p) - C_p)^2,
kin = 4 h^3 sum_x sum_i <[a0, A_i]_eta, [a0, A_i]_eta>_eta,   E_J = E_stat + J^2/(4 kin).
Vacuum (code branch s = -1, the notebook frame): d4 = diag(g, 1, delta, 0), g = 8,
delta = 0.3. Clock generator G1 (m5_21_8_b_lattice.G1): G1[2,3] = -1, G1[3,2] = +1,
the rotation of the (2,3) internal plane, i.e. of the (delta, 0) eigenplane;
a0 = G1 M - M G1 (the unit-omega tangent; G1^T = -G1).

THE SIX STATEMENTS CHECKED HERE (each a PASS line that can fail)
-------------------------------------------------------------------
S1  Wall identity. In the eigenframe of the (2,3) block, a0 = (d2 - d3)(E23 + E32):
    the generator vanishes on a surface where d2 = d3 and nowhere else.
S2  Derivative condition and the flank inertia. For a profile M(z) diagonal in a
    fixed frame, F_0z = [a0, d_z M]_eta = (d2 - d3)(d3' - d2')(E23 - E32), so the
    kinetic density is 8 (d2 - d3)^2 (d2' - d3')^2: it vanishes ON the wall
    (d2 = d3) and lives on the FLANKS where the eigenvalue ramp is steep. A linear
    ramp of width w from (delta, 0) to the degenerate pair carries
    kin/area = (8/3) delta^4 / w per unit omega^2 (continuum); the lattice
    one-cell layer under the sym stencil carries 4 delta^4 / h per rotating flank.
S3  Planar flatness (the static side). Any profile M(z) has F_ij = 0 for all
    spatial pairs (only A_z != 0, and [0, A_z] = 0), so its static energy is V4
    ONLY. Two consequences: an ORIENTATION wall (a rotation of the vacuum across
    the plane, spectrum preserved) has zero tension at any width; a DEGENERATE
    layer of thickness w has tension w V4_deg -> 0 as w -> 0. No planar wall in
    this action has an h-converged finite tension.
S4  Phase flatness (the time side). For M = R(phi) M0 R(phi)^T with M0 uniform
    and phi(x, t) ARBITRARY, every F_mu nu vanishes: A_mu = d_mu phi [G1, M] are
    all parallel and [X, X]_eta = 0. On a non-uniform M0 the phase couples only
    through [X, d_mu M0], X = [G1, M0]: F_mu nu = d_mu phi [X, A_nu^0] - d_nu phi
    [X, A_mu^0] + F^0_mu nu. The inertia density and the twist stiffness are the
    same tensor k_mu nu = <[X, A_mu^0], [X, A_nu^0]>: wherever the phase carries
    inertia it also carries stiffness, and where it carries none (uniform or
    degenerate regions) it is free.
S5  Free inertia. An orientation twist of the vacuum in a plane that does NOT
    commute with G1 (here the (1,2) plane, generator G3 of m5_21_8_b_lattice)
    has zero static cost by S3 and a kinetic density psi'^2 f(psi) > 0 under the
    G1 clock, so a twist of angle Psi over width w carries inertia ~ Psi^2/w at
    zero static cost: at fixed J the infimum of E_J over fields is E_stat, with
    kin unbounded (on the lattice, bounded by h). No fixed-J minimizer exists in
    the full field space; every finite-omega fixed-J state of the record is a
    constrained (family) minimum.
S6  Bag closure with a lattice-thin shell. E_J(R, w) = E_core + 4 pi R^2 w V4_deg
    + J^2 / (4 (kappa R + 4 pi R^2 c / w)) is monotone INCREASING in w for every
    R (dE/dw > 0 termwise), so the shell sharpens to the lattice floor w = h; in
    the ramp-dominated LIMIT (4 pi R^2 c / h >> kappa R) R*^4 = J^2 / (64 pi^2 c
    V4_deg) is h-independent while omega* = J / (2 kin*) = sqrt(V4_deg / c) h -> 0
    linearly in h. The NUMBERS are read from the full one-dimensional minimum of
    E_J(R) at w = h (not the limit; audit F4), with kappa = 14 stated as the R7
    tail-law import (audit F6; the W3 seed's own interior law is measured in W3),
    and with BOTH V4 references: the projected point diag(8, 1, 0.15, 0.15)
    (V4_deg = 1.80e-6) and the constrained minimum of V4 on the degenerate manifold
    (V4_deg_min = 6.48e-7, audit C4). Every R* lies outside every admissible box
    (audit F5): the shell tension of this action cannot hold a bag inside a
    48-to-96 box at any J >= 50. A finite R* with omega* -> 0 is the prediction.

AUDIT NOTES CARRIED (2026-09-02, the independent W0-W2 audit, m5_32_r13w_audit.py):
    S1 in the coordinate frame reads "a0_G1 = 0 iff d2 = d3 AND the degenerate
    eigenplane is the (2,3) coordinate plane"; the frame-free statement is
    "a0_local = 0 iff d2 = d3" (a0_local = the rotation about the local leading
    eigenvector, the generator W3 uses). S4 is continuum-exact; on the lattice a
    smooth (2,3) phase field carries E_u and kin residuals of O(h^2) (checked
    below). The positivity of kin (S5) is a property of spatial-block generators
    on block-diagonal fields; a boost generator over the same twist gives kin < 0.

Out: ../data/m5_32_r13w_w0.json
Run: /opt/anaconda3/envs/master312/bin/python3 m5_32_r13w_w0.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import time

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
DATA = os.path.join(RES, "data")
OUT = os.path.join(DATA, "m5_32_r13w_w0.json")
T0 = time.time()


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INS4 = _load("m5_21_3_a_4d", "m5_21_3_a_4d.py")
B8 = _load("m5_21_8_b_lattice", "m5_21_8_b_lattice.py")
LAG = _load("m5_32_lagrangian", "m5_32_lagrangian.py")

ETA = sp.diag(-1, 1, 1, 1)
G1 = sp.Matrix(B8.G1.astype(int))      # (2,3) rotation, the clock
G3 = sp.Matrix(B8.G3.astype(int))      # (1,2) rotation, the non-commuting twist
RESULTS, LINES = {}, []


def comm_eta(A, B):
    return A * ETA * B - B * ETA * A


def inner_eta(F, G):
    return (ETA * F * ETA * G.T).trace()


def rot(G, q):
    """exp(q G) for a spatial rotation generator (G^3 = -G)."""
    return sp.eye(4) + sp.sin(q) * G + (1 - sp.cos(q)) * G * G


def check(name, ok, detail):
    RESULTS[name] = {"pass": bool(ok), "detail": detail}
    LINES.append(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    print(LINES[-1], flush=True)


# ---------------------------------------------------------------- S1
def s1_wall_identity():
    d0, d1, d2, d3 = sp.symbols("d0 d1 d2 d3", real=True)
    M = sp.diag(d0, d1, d2, d3)
    a0 = G1 * M - M * G1
    E23 = sp.zeros(4, 4); E23[2, 3] = 1
    target = (d2 - d3) * (E23 + E23.T)
    ok = sp.simplify(a0 - target) == sp.zeros(4, 4)
    # nowhere else: a general symmetric M in the block eigenframe still has the
    # off-block entries of a0 set by M_2k, M_3k; a0 = 0 forces them to vanish too
    m = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"m{min(i,j)}{max(i,j)}", real=True))
    a0g = G1 * m - m * G1
    sol = sp.solve(list(a0g), [m[0, 2], m[0, 3], m[1, 2], m[1, 3], m[2, 3], m[2, 2]], dict=True)
    ok2 = len(sol) == 1 and all(sol[0].get(m[0, k], 0) == 0 and sol[0].get(m[1, k], 0) == 0
                                for k in (2, 3)) and sol[0].get(m[2, 3], 0) == 0 \
        and sp.simplify(sol[0].get(m[2, 2]) - m[3, 3]) == 0
    check("S1 wall identity a0 = (d2 - d3)(E23 + E32)", ok, "exact in the block eigenframe")
    check("S1 a0 = 0 iff the (2,3) block is a multiple of the identity and decoupled", ok2,
          f"solve(a0 = 0) -> {sol}")


# ---------------------------------------------------------------- S2
def s2_flank_inertia():
    z, w, delta = sp.symbols("z w delta", positive=True)
    d2f, d3f = sp.Function("d2")(z), sp.Function("d3")(z)
    M = sp.diag(sp.Symbol("d0"), sp.Symbol("d1"), d2f, d3f)
    a0 = G1 * M - M * G1
    Az = M.diff(z)
    F = comm_eta(a0, Az)
    E23 = sp.zeros(4, 4); E23[2, 3] = 1
    target = (d2f - d3f) * (d3f.diff(z) - d2f.diff(z)) * (E23 - E23.T)
    ok = sp.simplify(F - target) == sp.zeros(4, 4)
    dens = sp.simplify(4 * inner_eta(F, F))
    dens_target = 8 * (d2f - d3f) ** 2 * (d2f.diff(z) - d3f.diff(z)) ** 2
    ok2 = sp.simplify(dens - dens_target) == 0
    check("S2 F_0z = (d2 - d3)(d3' - d2')(E23 - E32) on a fixed-frame profile", ok, "exact")
    check("S2 kinetic density 8 (d2 - d3)^2 (d2' - d3')^2, zero ON the wall, flank-borne", ok2,
          str(dens))
    # linear ramp: gap delta (1 - z/w) on 0 < z < w, degenerate pair at z = w
    s = sp.symbols("s", positive=True)
    gap = delta * (1 - s)
    ramp = sp.integrate(8 * gap ** 2 * (delta / w) ** 2, (s, 0, 1)) * w
    ok3 = sp.simplify(ramp - sp.Rational(8, 3) * delta ** 4 / w) == 0
    check("S2 linear ramp of width w: kin/area = (8/3) delta^4 / w per unit omega^2", ok3,
          str(sp.simplify(ramp)))
    # lattice one-cell layer, sym stencil: fwd branch at the rotating flank cell
    # A_z = (M_deg - M_vac)/h, a0 = full gap delta; bwd branch at the layer (a0 = 0)
    h = sp.symbols("h", positive=True)
    Mv = sp.diag(8, 1, delta, 0)
    Md = sp.diag(8, 1, delta / 2, delta / 2)
    a0v = G1 * Mv - Mv * G1
    Fl = comm_eta(a0v, (Md - Mv) / h)
    per_col = 4 * h ** 3 * sp.Rational(1, 2) * inner_eta(Fl, Fl)      # one flank cell, fwd weight 1/2
    per_area = sp.simplify(per_col / h ** 2)
    ok4 = sp.simplify(per_area - 4 * delta ** 4 / h) == 0
    check("S2 lattice one-cell layer: kin/area = 4 delta^4 / h per rotating flank (sym stencil)",
          ok4, str(per_area))
    RESULTS["S2_numbers"] = {"delta": 0.3, "continuum_ramp_coeff_(8/3)delta^4": float(sp.Rational(8, 3) * sp.Rational(3, 10) ** 4),
                             "lattice_flank_coeff_4delta^4": float(4 * sp.Rational(3, 10) ** 4),
                             "lattice_flank_per_area_at_h": {str(hh): float(4 * 0.3 ** 4 / hh) for hh in (1.5, 1.0, 0.75)}}


# ---------------------------------------------------------------- S3
def s3_planar_flatness():
    z = sp.symbols("z", real=True)
    m = sp.Matrix(4, 4, lambda i, j: sp.Function(f"m{min(i,j)}{max(i,j)}")(z))
    Ax, Ay, Az = sp.zeros(4, 4), sp.zeros(4, 4), m.diff(z)
    Fs = [comm_eta(Ax, Ay), comm_eta(Ax, Az), comm_eta(Ay, Az)]
    ok = all(F == sp.zeros(4, 4) for F in Fs)
    check("S3 planar profile: F_xy = F_xz = F_yz = 0 identically (static energy = V4 only)", ok,
          "symbolic, general symmetric M(z)")
    # orientation wall: spectrum preserved -> V4 = 0 exactly
    q, delta, g = sp.symbols("q delta g", real=True)
    d4 = sp.diag(g, 1, delta, 0)
    for nm, G in (("(2,3) plane G1", G1), ("(1,2) plane G3", G3)):
        R = rot(G, q)
        Mq = R * d4 * R.T
        tr = [sp.simplify(((Mq * ETA) ** p).trace() - ((d4 * ETA) ** p).trace()) for p in range(1, 5)]
        check(f"S3 orientation wall in the {nm}: tr((M eta)^p) invariant, V4 = 0 at every angle",
              all(t == 0 for t in tr), str(tr))
    # lattice confirmation: random planar profile -> E_u exactly 0 on the certified stack
    rng = np.random.default_rng(13)
    cfg = INS4.base_cfg(s=-1.0, g=8.0, n=12, L=18.0)
    prof = rng.normal(size=(12, 4, 4)); prof = 0.5 * (prof + prof.swapaxes(-1, -2))
    M = np.broadcast_to(prof[None, None, :, :, :], (12, 12, 12, 4, 4)).copy()
    e_u, e_v = INS4.e_parts(M, cfg)
    ok2 = e_u == 0.0
    check("S3 lattice: a random planar profile has E_u == 0.0 exactly under the sym stencil", ok2,
          f"E_u = {e_u!r}, V4 = {e_v:.6g}")
    # degenerate layer tension on the lattice = h * V4_deg per cell column
    Mv = INS4.vac4(cfg)
    Md = Mv.copy(); Md[2, 2] = Md[3, 3] = 0.5 * cfg["delta"]
    Ml = np.broadcast_to(Mv, (12, 12, 12, 4, 4)).copy(); Ml[:, :, 6] = Md
    e_u, e_v = INS4.e_parts(Ml, cfg)
    tr_d = LAG.v4_traces_np(Md[None])
    v4_deg = float(sum((tr_d[p][0] - INS4.c4_of(cfg)[p]) ** 2 for p in range(4)) * INS4.W1)
    sigma = e_v / (12 * 12 * cfg["h"] ** 2)
    check("S3 lattice: one-cell degenerate layer has E_u == 0.0 and tension sigma_0 = h V4_deg",
          e_u == 0.0 and (v4_deg is None or abs(sigma - cfg["h"] * v4_deg) < 1e-12 * max(1.0, abs(sigma))),
          f"E_u = {e_u!r}, sigma_0 = {sigma:.6e}, h V4_deg = {None if v4_deg is None else cfg['h'] * v4_deg}")
    RESULTS["S3_numbers"] = {"V4_deg_per_volume": v4_deg, "sigma_0_one_cell_at_h1.5": sigma}


# ---------------------------------------------------------------- S4
def s4_phase_flatness():
    x, y, z, t = sp.symbols("x y z t", real=True)
    g, delta = sp.symbols("g delta", real=True)
    phi = sp.Function("phi")(x, y, z, t)
    d4 = sp.diag(g, 1, delta, 0)
    R = rot(G1, phi)
    M = R * d4 * R.T
    A = [M.diff(v) for v in (t, x, y, z)]
    Fs = {f"{a}{b}": sp.simplify(comm_eta(A[a], A[b])) for a in range(4) for b in range(a + 1, 4)}
    ok = all(F == sp.zeros(4, 4) for F in Fs.values())
    check("S4 uniform vacuum with an ARBITRARY phase phi(x,t): every F_mu nu = 0 (no action at all)",
          ok, "symbolic, both the time and the space pairs")
    # decomposition on a non-uniform background: F = dphi_mu [X, A_nu] - dphi_nu [X, A_mu] + F0
    m0 = sp.Matrix(4, 4, lambda i, j: sp.Function(f"n{min(i,j)}{max(i,j)}")(z))
    phz = sp.Function("chi")(z, t)
    Rz = rot(G1, phz)
    Mz = Rz * m0 * Rz.T
    At, Az = Mz.diff(t), Mz.diff(z)
    F = comm_eta(At, Az)
    X = G1 * m0 - m0 * G1
    A0z = m0.diff(z)
    lhs = Rz.T * F * Rz          # back to the rotating frame
    rhs = phz.diff(t) * comm_eta(X, A0z)          # A_t^0 = 0 (static background)
    # F in the lab frame = R (dphi_t [X, A_z^0] + phi_z * (terms that cancel: [X, X] = 0)) R^T
    ok2 = sp.simplify(lhs - rhs) == sp.zeros(4, 4)
    check("S4 non-uniform background: F_0z = R (chi_t [X, d_z M0]) R^T, the phase couples only through [X, dM0]",
          ok2, "symbolic; the chi_z chi_t [X, X] piece is identically zero")


# ---------------------------------------------------------------- S5
def s5_free_inertia():
    z, delta = sp.symbols("z delta", real=True)
    psi = sp.Function("psi")(z)
    d4 = sp.diag(8, 1, delta, 0)
    R = rot(G3, psi)
    M = R * d4 * R.T
    Az = M.diff(z)
    a0 = G1 * M - M * G1
    F = comm_eta(a0, Az)
    dens = sp.simplify(4 * inner_eta(F, F))
    # static cost: planar (S3) -> E_u = 0, and the spectrum is preserved -> V4 = 0
    trs = [sp.simplify(((M * ETA) ** p).trace() - ((d4 * ETA) ** p).trace()) for p in range(1, 5)]
    fpsi = sp.simplify(dens / psi.diff(z) ** 2)
    ok = all(t == 0 for t in trs) and sp.simplify(fpsi.diff(psi.diff(z))) == 0
    # positivity: f(psi) > 0 except at isolated angles
    fnum = sp.lambdify((psi, delta), fpsi, "numpy")
    ang = np.linspace(0, 2 * np.pi, 721)
    vals = fnum(ang, 0.3)
    ok2 = np.min(vals) >= 0 and np.mean(vals) > 0
    check("S5 non-commuting twist psi(z) in the (1,2) plane: V4 = 0, E_u = 0, kin density = psi'^2 f(psi)",
          ok, f"f(psi) = {fpsi}")
    check("S5 f(psi) >= 0 with positive mean: inertia ~ Psi^2/w at ZERO static cost (kin unbounded at fixed J)",
          ok2, f"mean f over a period at delta = 0.3: {float(np.mean(vals)):.6g}, min {float(np.min(vals)):.3g}")
    # lattice confirmation: twist over w cells, E_stat = 0 and kin ~ 1/w
    cfg = INS4.base_cfg(s=-1.0, g=8.0, n=16, L=24.0)
    rows = []
    for wc in (2, 4, 8):
        n = cfg["n"]
        ps = np.zeros(n)
        k0 = n // 2 - wc // 2
        ps[k0:k0 + wc] = np.linspace(0, 1.0, wc, endpoint=False)[:] * (1.0 / wc) * wc
        ps[k0:k0 + wc] = np.linspace(0, 1.0, wc + 1)[1:]          # rises to Psi = 1 over wc cells
        ps[k0 + wc:] = 1.0
        Rn = B8.rot_field(B8.G3, np.broadcast_to(ps[None, None, :], (n, n, n)))
        Mn = np.einsum("...ab,bc,...dc->...ad", Rn, INS4.vac4(cfg), Rn)
        a0 = B8.G1 @ Mn - Mn @ B8.G1
        e_u, e_v = INS4.e_parts(Mn, cfg)
        k = INS4.kin_of(Mn, a0, cfg) / (n * n * cfg["h"] ** 2)
        rows.append({"w_cells": wc, "w": wc * cfg["h"], "E_u": float(e_u), "V4": float(e_v), "kin_per_area": float(k)})
        print(f"   twist over {wc} cells (w = {wc * cfg['h']:.2f}): E_u {e_u!r} V4 {e_v:.3e} kin/area {k:.6f}", flush=True)
    kw = [r["kin_per_area"] * r["w"] for r in rows]
    ok3 = all(r["E_u"] == 0.0 for r in rows) and all(r["V4"] < 1e-24 for r in rows) \
        and max(kw) / min(kw) < 1.6
    check("S5 lattice: twists of width w have E_u == 0.0, V4 = 0, and kin/area * w roughly constant (~1/w)",
          ok3, f"kin/area * w = {[round(v, 6) for v in kw]}")
    RESULTS["S5_numbers"] = rows


# ---------------------------------------------------------------- S6
def s6_bag_closure():
    R, w, J, kappa, c, V, E0 = sp.symbols("R w J kappa c V E0", positive=True)
    E = E0 + 4 * sp.pi * R ** 2 * w * V + J ** 2 / (4 * (kappa * R + 4 * sp.pi * R ** 2 * c / w))
    dEdw = sp.simplify(E.diff(w))
    # both pieces of dE/dw are positive: 4 pi R^2 V and J^2 pi R^2 c / (w^2 kin^2)
    num = sp.simplify(dEdw * (kappa * R + 4 * sp.pi * R ** 2 * c / w) ** 2 * w ** 2)
    ok = sp.simplify(num - (4 * sp.pi * R ** 2 * V * w ** 2 * (kappa * R + 4 * sp.pi * R ** 2 * c / w) ** 2
                            + J ** 2 * sp.pi * R ** 2 * c)) == 0
    check("S6 dE_J/dw > 0 termwise: the shell sharpens to the lattice floor at every R and J", ok, str(dEdw))
    # ramp-dominated floor w = h: R*^4 = J^2 / (64 pi^2 c V), omega* = J h / (8 pi c R*^2)
    h = sp.symbols("h", positive=True)
    Eh = E0 + 4 * sp.pi * R ** 2 * h * V + J ** 2 * h / (16 * sp.pi * R ** 2 * c)
    Rstar = sp.solve(sp.Eq(Eh.diff(R), 0), R)
    Rstar = [r for r in Rstar if r.is_positive is not False]
    R4 = sp.simplify(Rstar[0] ** 4)
    ok2 = sp.simplify(R4 - J ** 2 / (64 * sp.pi ** 2 * c * V)) == 0
    kin_star = 4 * sp.pi * Rstar[0] ** 2 * c / h
    om = sp.simplify(J / (2 * kin_star))
    ok3 = sp.simplify(om - J * h / (8 * sp.pi * c * Rstar[0] ** 2)) == 0 and sp.simplify(om / h).has(h) is False
    check("S6 ramp-dominated floor: R*^4 = J^2/(64 pi^2 c V4_deg) is h-independent", ok2, str(R4))
    check("S6 omega* = J h/(8 pi c R*^2) is LINEAR in h (-> 0 in the continuum) in the ramp-dominated limit", ok3, str(om))
    # the constrained minimum of V4 on the degenerate manifold diag(d0, d1, m, m) (audit C4)
    from scipy.optimize import minimize
    cfg = INS4.base_cfg(s=-1.0, g=8.0, n=4, L=6.0)
    cp = INS4.c4_of(cfg)

    def v4_of(x):
        Mc = np.diag([x[0], x[1], x[2], x[2]])
        t = LAG.v4_traces_np(Mc[None])
        return float(sum((t[k][0] - cp[k]) ** 2 for k in range(4)) * INS4.W1)
    x0 = np.array([8.0, 1.0, 0.15])
    r = minimize(v4_of, x0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-18, "maxiter": 20000})
    V_pt, V_min = v4_of(x0), float(r.fun)
    check("S6 V4 on the degenerate manifold: projected point 1.80e-6, constrained minimum 6.48e-7 (audit C4)",
          abs(V_pt - 1.7994e-6) < 1e-9 and abs(V_min - 6.479e-7) < 2e-9, f"V_pt {V_pt:.4e}, V_min {V_min:.4e} at {r.x}")
    # full one-dimensional minimum of E_J(R) at w = h (audit F4): both V references, kappa = 14 import
    c = 4 * 0.3 ** 4
    kappa_import = 14.0
    vals = {"c_lattice_one_flank": c, "V4_deg_projected_point": V_pt, "V4_deg_constrained_min": V_min,
            "kappa_import_R7": kappa_import, "note": "R* from the full minimum of 4 pi R^2 h V + J^2/(4 (kappa R + 4 pi R^2 c/h)) at h = 1.5"}
    Rg = np.linspace(2.0, 2000.0, 400000)
    hh = 1.5
    for Vname, Vv in (("V_projected", V_pt), ("V_min", V_min)):
        for Jv in (50.0, 200.0, 800.0):
            kin = kappa_import * Rg + 4 * np.pi * Rg ** 2 * c / hh
            E = 4 * np.pi * Rg ** 2 * hh * Vv + Jv ** 2 / (4 * kin)
            i = int(np.argmin(E))
            vals[f"{Vname}_J{Jv:g}"] = {"R_star": float(Rg[i]), "omega_star": float(Jv / (2 * kin[i])),
                                        "ramp_over_core": float(4 * np.pi * Rg[i] ** 2 * c / hh / (kappa_import * Rg[i])),
                                        "surface_term": float(4 * np.pi * Rg[i] ** 2 * hh * Vv), "kinetic_term": float(Jv ** 2 / (4 * kin[i]))}
    # the J that puts R* inside a box (audit F5): R* = 12 and 20 with V_min
    for Rt in (12.0, 20.0):
        # stationarity: 8 pi R h V = J^2 kin'(R) / (4 kin^2)
        kin = kappa_import * Rt + 4 * np.pi * Rt ** 2 * c / hh
        dkin = kappa_import + 8 * np.pi * Rt * c / hh
        vals[f"J_for_Rstar_{Rt:g}_Vmin"] = float(np.sqrt(8 * np.pi * Rt * hh * V_min * 4 * kin ** 2 / dkin))
    ok4 = all(vals[k]["R_star"] > 48.0 for k in vals if k.startswith("V_")) 
    check("S6 numbers: every R* (both V references, J = 50 / 200 / 800) lies beyond every admissible half-box (48)", ok4,
          json.dumps({k: (round(v["R_star"], 1), round(v["omega_star"], 5)) for k, v in vals.items() if k.startswith("V_")}))
    RESULTS["S6_numbers"] = vals
    print("   S6 numbers:", json.dumps(vals, indent=None)[:900], flush=True)
    # audit D2: lattice residual of the (2,3) phase flatness is O(h^2)
    rows = []
    for n, Lb in ((16, 24.0), (32, 24.0)):
        cfgb = INS4.base_cfg(s=-1.0, g=8.0, n=n, L=Lb)
        X, Y, Z = INS4.coords(n, cfgb["h"])
        phi = 0.7 * np.sin(2 * np.pi * X / Lb) * np.cos(2 * np.pi * Z / Lb)
        Rn = B8.rot_field(B8.G1, phi)
        Mn = np.einsum("...ab,bc,...dc->...ad", Rn, INS4.vac4(cfgb), Rn)
        e_u, _ = INS4.e_parts(Mn, cfgb)
        k = INS4.kin_of(Mn, B8.G1 @ Mn - Mn @ B8.G1, cfgb)
        rows.append({"h": cfgb["h"], "E_u": float(e_u), "kin": float(k)})
    pe = np.log(rows[0]["E_u"] / rows[1]["E_u"]) / np.log(2.0)
    pk = np.log(rows[0]["kin"] / rows[1]["kin"]) / np.log(2.0)
    check("S4 lattice residual of the (2,3) phase flatness is O(h^2) in E_u and kin (audit D2)", pe > 1.5 and pk > 1.5,
          f"exponents E_u {pe:.2f}, kin {pk:.2f}; {rows}")
    RESULTS["S4_lattice_residual"] = {"rows": rows, "exp_E_u": float(pe), "exp_kin": float(pk)}


if __name__ == "__main__":
    for f in (s1_wall_identity, s2_flank_inertia, s3_planar_flatness, s4_phase_flatness,
              s5_free_inertia, s6_bag_closure):
        print(f"--- {f.__name__} [{time.time() - T0:.1f}s]", flush=True)
        f()
    n_pass = sum(1 for k, v in RESULTS.items() if isinstance(v, dict) and v.get("pass") is True)
    n_all = sum(1 for k, v in RESULTS.items() if isinstance(v, dict) and "pass" in v)
    RESULTS["summary"] = {"pass": n_pass, "checks": n_all, "lines": LINES,
                          "runtime_s": round(time.time() - T0, 1)}
    json.dump(RESULTS, open(OUT, "w"), indent=1, default=float)
    print(f"W0: {n_pass}/{n_all} PASS, written {OUT} [{time.time() - T0:.1f}s]")
