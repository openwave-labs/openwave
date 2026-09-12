"""M5.32 R19: the mixed-bracket entrants as lattice energies with EXACT
gradients (the registry extension the R19 ladder runs on).

EQUATIONS FIRST
---------------
Field M(x) real symmetric 4x4 per cell, eta = diag(-1, 1, 1, 1), jets
A_i = d_i M on the certified sym stencil (1/2 (fwd + bwd), the density per
branch, exact adjoints d1_adj), A_0 = omega a0 optional (energy reads only;
the gradient is the static one, a0 frozen). u the timelike unit
eigenvector of M eta (u^T eta u = -1), Pi_u = -u u^T eta, Pi_s = 1 - Pi_u.
    P_mu nu = A_mu eta A_nu,  F = P - P^T,  G = P + P^T   (^T internal)
    A^t = Pi_u A Pi_s^T + Pi_s A Pi_u^T,  P^tt = A^t eta A^t
Objects X (m5_32_r19_c9_c12, the same definitions, re-implemented here
with their adjoints and gated against that module to roundoff):
    I1      X = F
    GG      X = G                                     (A)
    T       X = F on spatial (mu, nu), G on (0, i), (i, 0), (0, 0)
    Bu      X = Pi_s F Pi_s^T + Pi_u G Pi_s^T + Pi_s G Pi_u^T + Pi_u G Pi_u^T
    Gam     X = F + 2 (P^tt)^T                        (Gamma)
    Gam_tl  X = F - (P^tt - P^tt^T) + T_tl(Pi_s (P^tt + P^tt^T) Pi_s^T),
            T_tl(Y) = Y - (1/3) tr(eta Y) (eta + u u^T)  (Gamma, traceless)
Density f_X = (1/2) sum_{mu nu} eta^mu eta^nu <X_mu nu, X_mu nu>_eta.
The family (ledger § 6.8):
    L(c) = -4 [(1 - c) I1 + c X] - V4
    E_stat[M] = 4 h^3 sum_br wt sum_cells [(1 - c) f_I1 + c f_X] + V4
Gradient wrt symmetric M: the stencil adjoint of df/dA_i (exact) plus the
eigenframe piece df/du chained through the first-order eigenvector
perturbation of M eta (the R2 chain, m5_32_r2_b_bounded.energy_grad,
verbatim), then sym4; V4 by the registry (v4_grad_np). Backpropagation:
W = df/dX = eta^mu eta^nu eta X eta; every sandwich S1 Z S2^T has the
adjoints S1^T W S2 (to Z), W S2 Z^T (to S1), W^T S1 Z (to S2); the
transpose map has adjoint W^T; T_tl has adjoint W - (1/3) <eta + u u^T, W>
diag(eta) and contributes -(1/3) tr(eta Y) (W + W^T) u to df/du; the
product P = B eta C has adjoints W C eta (to B) and eta B W (to C);
Pi_s = 1 - Pi_u folds into dPi_u, and df/du = -(dPi_u eta u + eta dPi_u^T u).

GATES (python3 m5_32_r19_entrants.py --gate): each a PASS line that can
fail: (1) the densities equal m5_32_r19_c9_c12.objects on random jets and
random u; (2) the I1 energy AND gradient equal the certified stack's
(m5_32_r2_b_bounded.energy_grad at lambda = 0, at g = 32 where that
instrument's V4 gradient lives) to roundoff on a dressed field; (3) every object's gradient passes a central finite-difference test
along 4 random symmetric directions on an n = 8 dressed field (rel 1e-6);
(4) on a Coulomb-sector field (no time row and a constant M_00; Bu
needs the corner jet zero too) Gam, Gam_tl, Bu equal I1 exactly (energy
and gradient); (5) the mutant that drops the eigenframe chain
breaks gate (3) for the u-dependent objects (its FD residual at least
1e3 times the true gradient's).
Out: ../data/m5_32_r19_entrants_gate.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


LAG = _load("m5_32_lagrangian", "m5_32_lagrangian.py")
RB = _load("m5_32_r2_b_bounded", "m5_32_r2_b_bounded.py")
B3 = LAG.B3
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
ETA_D = np.array([-1.0, 1.0, 1.0, 1.0])
W_MK = np.einsum("m,k->mk", ETA_D, ETA_D)
I4 = np.eye(4)
OBJECTS = ("I1", "GG", "T", "Bu", "Gam", "Gam_tl")
MUTANT = {"eta_proj": False, "no_u_chain": False}


def projectors(u):
    e = np.array([1.0, 1.0, 1.0, 1.0]) if MUTANT["eta_proj"] else ETA_D
    Pu = -np.einsum("na,nb,b->nab", u, u, e)
    return Pu, I4[None] - Pu


def sw(L, X, R):
    return np.einsum("nab,mknbc,ndc->mknad", L, X, R, optimize=True)


def swT(L, X, R):
    """L^T X R (the adjoint of sw to its middle argument)."""
    return np.einsum("nba,mknbc,ncd->mknad", L, X, R, optimize=True)


def sw1(L, A, R):
    return np.einsum("nab,mnbc,ndc->mnad", L, A, R, optimize=True)


def P_of(A):
    return np.einsum("mnab,bc,kncd->mknad", A, ETA, A, optimize=True)


def Ifull(X):
    return 0.5 * np.einsum("mk,a,b,mknab,mknab->n", W_MK, ETA_D, ETA_D, X, X, optimize=True)


def tI(X):
    return X.swapaxes(-1, -2)


def dP_to_dA(dP, A):
    """P = A_mu eta A_nu: dA_mu = sum_nu dP_mu nu A_nu eta + sum_k eta A_k dP_k mu."""
    return (np.einsum("mknab,knbc,cd->mnad", dP, A, ETA, optimize=True)
            + np.einsum("ab,knbc,kmncd->mnad", ETA, A, dP, optimize=True))


def fwd_bwd(A, u, obj, need_grad=True):
    """A (4, N, 4, 4) symmetric jets, u (N, 4). Returns f (N,), dA (4,N,4,4), du (N,4).
    Static fast path: with A[0] == 0 only the spatial derivative pairs are
    built (T equals I1 there)."""
    if A.shape[0] == 4 and not np.any(A[0]):
        f, dA, du = _fwd_bwd(A[1:], u, "I1" if obj == "T" else obj, need_grad, W_MK[1:, 1:])
        if need_grad:
            dA = np.concatenate([np.zeros_like(dA[:1]), dA], axis=0)
        return f, dA, du
    return _fwd_bwd(A, u, obj, need_grad, W_MK)


def _fwd_bwd(A, u, obj, need_grad, Wmk):
    N = A.shape[1]
    Pu, Ps = projectors(u)
    P = P_of(A); PT = tI(P)
    F = P - PT; G = P + PT
    At = Ptt = Y = trY = gs = None
    if obj in ("Gam", "Gam_tl"):
        At = sw1(Pu, A, Ps) + sw1(Ps, A, Pu)
        Ptt = P_of(At); PttT = tI(Ptt)
    if obj == "I1":
        X = F
    elif obj == "GG":
        X = G
    elif obj == "T":
        X = F.copy(); X[0] = G[0]; X[:, 0] = G[:, 0]
    elif obj == "Gam":
        X = F + 2.0 * PttT
    elif obj == "Gam_tl":
        Xs = Ptt + PttT; Xa = Ptt - PttT
        Y = sw(Ps, Xs, Ps)
        trY = np.einsum("a,mknaa->mkn", ETA_D, Y)
        gs = ETA[None] + np.einsum("na,nb->nab", u, u)
        X = F - Xa + Y - (trY / 3.0)[..., None, None] * gs[None, None]
    elif obj == "Bu":
        X = sw(Ps, F, Ps) + sw(Pu, G, Ps) + sw(Ps, G, Pu) + sw(Pu, G, Pu)
    else:
        raise ValueError(obj)
    f = 0.5 * np.einsum("mk,a,b,mknab,mknab->n", Wmk, ETA_D, ETA_D, X, X, optimize=True)
    if not need_grad:
        return f, None, None
    W = Wmk[:, :, None, None, None] * np.einsum("a,b,mknab->mknab", ETA_D, ETA_D, X)
    dPu = np.zeros((N, 4, 4)); dPs = np.zeros((N, 4, 4)); du = np.zeros((N, 4))
    dPtt = None
    if obj == "I1":
        dP = W - tI(W)
    elif obj == "GG":
        dP = W + tI(W)
    elif obj == "T":
        dF = W.copy(); dF[0] = 0.0; dF[:, 0] = 0.0
        dG = np.zeros_like(W); dG[0] = W[0]; dG[:, 0] = W[:, 0]
        dP = dF - tI(dF) + dG + tI(dG)
    elif obj == "Gam":
        dP = W - tI(W)
        dPtt = 2.0 * tI(W)
    elif obj == "Gam_tl":
        dP = W - tI(W)
        dXa = -W
        # X_tl = Y - (trY/3) gs
        dY = W - (np.einsum("nab,mknab->mkn", gs, W) / 3.0)[..., None, None] * np.diag(ETA_D)[None, None, None]
        dgs = np.einsum("mkn,mknab->nab", -trY / 3.0, W)
        du += np.einsum("nab,nb->na", dgs + tI(dgs), u)
        # Y = Ps Xs Ps^T
        dXs = swT(Ps, dY, Ps)
        Xs = Ptt + PttT
        dPs += np.einsum("mknab,nbc,mkndc->nad", dY, Ps, Xs, optimize=True) \
            + np.einsum("mknba,nbc,mkncd->nad", dY, Ps, Xs, optimize=True)
        dPtt = dXs + tI(dXs) + dXa - tI(dXa)
    elif obj == "Bu":
        dF = swT(Ps, W, Ps)
        dG = swT(Pu, W, Ps) + swT(Ps, W, Pu) + swT(Pu, W, Pu)
        # sandwich adjoints to the projectors: S1 Z S2^T -> dS1 += W S2 Z^T, dS2 += W^T S1 Z
        def s1(Wm, S2, Z):
            return np.einsum("mknab,nbc,mkndc->nad", Wm, S2, Z, optimize=True)

        def s2(Wm, S1, Z):
            return np.einsum("mknba,nbc,mkncd->nad", Wm, S1, Z, optimize=True)
        dPs += s1(W, Ps, F) + s2(W, Ps, F)
        dPu += s1(W, Ps, G); dPs += s2(W, Pu, G)
        dPs += s1(W, Pu, G); dPu += s2(W, Ps, G)
        dPu += s1(W, Pu, G) + s2(W, Pu, G)
        dP = dF - tI(dF) + dG + tI(dG)
    dA = dP_to_dA(dP, A)
    if dPtt is not None:
        dAt = dP_to_dA(dPtt, At)
        dA += (np.einsum("nba,mnbc,ncd->mnad", Pu, dAt, Ps, optimize=True)
               + np.einsum("nba,mnbc,ncd->mnad", Ps, dAt, Pu, optimize=True))
        # At = Pu A Ps^T + Ps A Pu^T
        dPu += np.einsum("mnab,nbc,mncd->nad", dAt, Ps, A, optimize=True) \
            + np.einsum("mnba,nbc,mncd->nad", dAt, Ps, A, optimize=True)
        dPs += np.einsum("mnba,nbc,mncd->nad", dAt, Pu, A, optimize=True) \
            + np.einsum("mnab,nbc,mncd->nad", dAt, Pu, A, optimize=True)
    dPu_tot = dPu - dPs
    e = np.array([1.0, 1.0, 1.0, 1.0]) if MUTANT["eta_proj"] else ETA_D
    du += -(np.einsum("nab,b,nb->na", dPu_tot, e, u) + e[None, :] * np.einsum("nba,nb->na", dPu_tot, u))
    return f, dA, du


def u_chain(M, v):
    """the R2 eigenvector chain: v = dE/du0 per cell -> dE/dM (unsymmetrized)."""
    u0, V, lamv, sig, k0, ok, gap = RB.tl_eig(M)
    vu = np.einsum("...a,...ak->...k", v, V)
    l0 = np.take_along_axis(lamv, k0[..., None], axis=-1)[..., 0]
    den = l0[..., None] - lamv
    mask = np.arange(4)[None, :] != k0.reshape(-1, 1)
    mask = mask.reshape(den.shape)
    c = np.where(mask, sig * vu / np.where(mask, den, 1.0), 0.0)
    w = np.einsum("...ak,...k->...a", V, c)
    ew = w @ ETA
    eu = u0 @ ETA
    return ew[..., :, None] * eu[..., None, :]


def energy_grad(M, cfg, obj, c=1.0, a0=None, omega=0.0, need_grad=True, p=None):
    """E_stat of L(c) = -4[(1-c) I1 + c X] - V4 and its exact gradient (a0 frozen)."""
    h3, h = cfg["h"] ** 3, cfg["h"]
    shape = M.shape[:-2]
    u0, V, lamv, sig, k0, ok, gap = RB.tl_eig(M)
    info = {"ok": bool(np.all(ok)), "min_gap": float(np.min(gap)), "n_bad": int(np.sum(~ok)),
            "max_abs_M0i": float(np.max(np.abs(M[..., 0, 1:])))}
    if not info["ok"] and obj in ("Bu", "Gam", "Gam_tl") and c != 0.0:
        return np.nan, None, info
    uf = u0.reshape(-1, 4)
    E = 0.0
    G = np.zeros_like(M) if need_grad else None
    vacc = np.zeros(uf.shape) if need_grad else None
    for br, wt in B3.branches(cfg["stencil"]):
        A = np.zeros((4,) + M.shape)
        for ax in range(3):
            A[1 + ax] = B3.d1(M, ax, h, br)
        if a0 is not None and omega != 0.0:
            A[0] = omega * a0
        Af = A.reshape(4, -1, 4, 4)
        pieces = []
        if c != 1.0:
            pieces.append((1.0 - c, "I1"))
        if c != 0.0:
            pieces.append((c, obj))
        for coef, ob in pieces:
            f, dA, du = fwd_bwd(Af, uf, ob, need_grad)
            E += wt * coef * float(f.sum())
            if need_grad:
                dA = dA.reshape(A.shape)
                for ax in range(3):
                    G += wt * coef * B3.d1_adj(dA[1 + ax], ax, h, br)
                vacc += wt * coef * du
    if p is None:
        p = LAG.default_params(s=cfg["s"], g=cfg["g"], delta=cfg["delta"]) if "delta" in LAG.default_params.__code__.co_varnames else LAG.default_params(s=cfg["s"], g=cfg["g"])
    _, ev = B3.e_parts(M, cfg)
    E = 4.0 * h3 * E + float(ev)
    if not need_grad:
        return E, None, info
    if not MUTANT["no_u_chain"]:
        G += u_chain(M, vacc.reshape(shape + (4,)))
    G = 4.0 * h3 * B3.sym4(G) + h3 * B3.sym4(LAG.v4_grad_np(M, p))
    return E, G, info


def block_reads(M, cfg, obj, c=1.0):
    """E_stat split: the certified I1 read, the X read, the time-row (R^eg) piece and the boost-boost pieces of X."""
    h3, h = cfg["h"] ** 3, cfg["h"]
    u0, V, lamv, sig, k0, ok, gap = RB.tl_eig(M)
    uf = u0.reshape(-1, 4)
    acc = {"I1": 0.0, obj: 0.0}
    for br, wt in B3.branches(cfg["stencil"]):
        A = np.zeros((4,) + M.shape)
        for ax in range(3):
            A[1 + ax] = B3.d1(M, ax, h, br)
        Af = A.reshape(4, -1, 4, 4)
        for ob in acc:
            f, _, _ = fwd_bwd(Af, uf, ob, need_grad=False)
            acc[ob] += wt * float(f.sum())
    _, ev = B3.e_parts(M, cfg)
    out = {"V4": float(ev), "E_curv_I1": 4.0 * h3 * acc["I1"], f"E_curv_{obj}": 4.0 * h3 * acc[obj],
           "E_total": 4.0 * h3 * ((1 - c) * acc["I1"] + c * acc[obj]) + float(ev), "c": c,
           "min_gap": float(np.min(gap)), "max_abs_M0i": float(np.max(np.abs(M[..., 0, 1:])))}
    return out


# ================= gates =================
def _rand_dressed(cfg, rng, amp=0.15, boost=0.2):
    n, h = cfg["n"], cfg["h"]
    X, Y, Z = B3.coords(n, h)
    d = np.diag([cfg["g"], 1.0, cfg["delta"], 0.0])
    M = np.tile(d, (n, n, n, 1, 1))
    # a smooth random symmetric perturbation (three long-wavelength modes)
    for _ in range(3):
        k = rng.normal(size=3) * 2 * np.pi / cfg["L"]
        S = rng.normal(size=(4, 4)); S = amp * (S + S.T)
        M = M + np.cos(k[0] * X + k[1] * Y + k[2] * Z)[..., None, None] * S
    # a local boost dressing
    R = np.sqrt(X * X + Y * Y + Z * Z)
    bl = boost * np.exp(-(R / (0.3 * cfg["L"])) ** 2)
    K = np.zeros(X.shape + (4, 4)); nx, ny, nz = X / np.maximum(R, 1e-9), Y / np.maximum(R, 1e-9), Z / np.maximum(R, 1e-9)
    K[..., 0, 1], K[..., 0, 2], K[..., 0, 3] = nx, ny, nz
    K[..., 1, 0], K[..., 2, 0], K[..., 3, 0] = nx, ny, nz
    K2 = np.zeros_like(K); K2[..., 0, 0] = 1.0
    for i, a in enumerate((nx, ny, nz)):
        for j, b in enumerate((nx, ny, nz)):
            K2[..., 1 + i, 1 + j] = a * b
    Q = np.eye(4)[None, None, None] + np.sinh(bl)[..., None, None] * K + (np.cosh(bl) - 1.0)[..., None, None] * K2
    return B3.sym4(np.einsum("...ab,...bc,...dc->...ad", Q, M, Q))


def gate(n=8, L=12.0, g=8.0, seed=5):
    rng = np.random.default_rng(seed)
    C = _load("m5_32_r19_c9_c12", "m5_32_r19_c9_c12.py")
    cfg = B3.base_cfg(s=-1.0, g=g, n=n, L=L, delta=0.3)
    p = LAG.default_params(s=-1.0, g=g)
    out = {"n": n, "L": L, "g": g}
    lines = {}
    # (1) densities vs the C9 module
    Ar = rng.normal(size=(4, 50, 4, 4)); Ar = Ar + Ar.swapaxes(-1, -2)
    Mr = np.stack([np.diag([g, 1.0, 0.3, 0.0]) + 0.2 * (lambda S: S + S.T)(rng.normal(size=(4, 4))) for _ in range(50)])
    ur, ok, _ = C.u_of(Mr)
    ob = C.objects(Ar, ur)
    worst = 0.0
    for o in ("I1", "GG", "Bu", "Gam", "Gam_tl"):
        f, _, _ = fwd_bwd(Ar, ur, o, need_grad=False)
        worst = max(worst, C.rel(f, ob[o]))
    fT, _, _ = fwd_bwd(Ar, ur, "T", need_grad=False)
    PT_ = C.P_of(Ar); FT_ = PT_ - PT_.swapaxes(0, 1); GT_ = PT_ + PT_.swapaxes(0, 1); XT = FT_.copy(); XT[0] = GT_[0]; XT[:, 0] = GT_[:, 0]
    worst = max(worst, C.rel(fT, C.Ifull(XT)))
    out["densities_vs_c9_module_rel"] = worst
    lines["G1_densities_match_c9_module"] = worst < 1e-12
    # (2) I1 energy and gradient vs the certified stack
    cfg32 = B3.base_cfg(s=-1.0, g=32.0, n=n, L=L, delta=0.3)
    M32 = _rand_dressed(cfg32, rng)
    E1, G1, info = energy_grad(M32, cfg32, "I1", c=1.0, p=LAG.default_params(s=-1.0, g=32.0))
    E2, G2, info2 = RB.energy_grad(M32, cfg32, 0.0)
    M = _rand_dressed(cfg, rng)
    out["I1_vs_certified"] = {"E_rel": abs(E1 - E2) / max(abs(E2), 1e-300), "G_rel": C.rel(G1, G2)}
    lines["G2_I1_equals_certified_stack"] = out["I1_vs_certified"]["E_rel"] < 1e-12 and out["I1_vs_certified"]["G_rel"] < 1e-10
    # (3) FD gradient gates
    def fd_gate(obj, c, M, eps=1e-3, ndir=4):
        E0, G0, inf = energy_grad(M, cfg, obj, c=c, p=p)
        worst = 0.0
        for _ in range(ndir):
            dM = rng.normal(size=M.shape); dM = B3.sym4(dM); dM /= np.sqrt(np.sum(dM * dM))
            e = lambda t: energy_grad(M + t * dM, cfg, obj, c=c, need_grad=False, p=p)[0]  # noqa: E731
            fd = (8.0 * (e(eps) - e(-eps)) - (e(2 * eps) - e(-2 * eps))) / (12.0 * eps)
            an = float(np.sum(G0 * dM))
            worst = max(worst, abs(fd - an) / max(np.sqrt(np.sum(G0 * G0)), 1e-12))
        return worst, E0, inf["min_gap"]
    out["fd"] = {}
    for obj in OBJECTS:
        w, E0, gap = fd_gate(obj, 1.0, M)
        out["fd"][obj] = {"worst_rel": w, "E": E0, "min_gap": gap}
        lines[f"G3_fd_gradient_{obj}"] = w < 1e-6
    w, E0, gap = fd_gate("Gam", 0.5, M)
    out["fd"]["Gam_c0.5"] = {"worst_rel": w, "E": E0}
    lines["G3_fd_gradient_Gam_c0.5"] = w < 1e-6
    # (4) Coulomb-sector identity
    Mc = _rand_dressed(cfg, rng, boost=0.0)
    Mc[..., 0, 1:] = 0.0; Mc[..., 1:, 0] = 0.0; Mc[..., 0, 0] = g
    Ec, Gc, _ = energy_grad(Mc, cfg, "I1", p=p)
    out["coulomb"] = {}
    for obj in ("Gam", "Gam_tl", "Bu"):
        Eo, Go, _ = energy_grad(Mc, cfg, obj, p=p)
        out["coulomb"][obj] = {"E_rel": abs(Eo - Ec) / max(abs(Ec), 1e-300), "G_rel": C.rel(Go, Gc)}
        lines[f"G4_coulomb_identity_{obj}"] = out["coulomb"][obj]["E_rel"] < 1e-12 and out["coulomb"][obj]["G_rel"] < 1e-10
    # (5) mutant
    MUTANT["no_u_chain"] = True
    mut = {}
    for obj in ("Gam", "Bu"):
        w, _, _ = fd_gate(obj, 1.0, M)
        mut[obj] = w
    MUTANT["no_u_chain"] = False
    out["mutant_no_u_chain_fd_worst"] = mut
    lines["G5_mutant_breaks_fd_gate"] = max(mut.values()) > 1e3 * max(out["fd"]["Gam"]["worst_rel"], out["fd"]["Bu"]["worst_rel"])
    out["lines"] = {k: bool(v) for k, v in lines.items()}
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "m5_32_r19_entrants_gate.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out["lines"].items():
        print(f"{'PASS' if v else 'FAIL'} {k}")
    print(json.dumps({k: out[k] for k in ("densities_vs_c9_module_rel", "I1_vs_certified", "fd", "coulomb", "mutant_no_u_chain_fd_worst")}, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    if a.gate:
        t0 = time.time()
        gate()
        print(f"gate runtime {time.time() - t0:.1f}s")
