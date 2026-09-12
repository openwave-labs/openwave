"""M5.32 R19-0 adversarial audit: an independent re-derivation of the six
claim groups of the R19-0 rung (the C9 to C12 identities, the form-level
certificate, the entrants' exact gradients, the C12 hedgehog ladder), written
from the definitions in its own conventions (jets stored cell first, its own
eigen-solve of M eta, its own eigenframe lift as a rotation times a Cayley
boost, its own Gram construction of the omega^2 form, its own np.gradient
stencil on the lattice, an h-ladder that never exceeds n = 24), so that
agreement with the producer is a check and not a re-run. This file does NOT
import m5_32_r19_c9_c12.py or m5_32_r19_certificate.py; it imports
m5_32_r19_entrants.py ONLY to call the energy_grad under test.

EQUATIONS FIRST
---------------
Field M(x) real symmetric 4x4, eta = diag(-1, 1, 1, 1), vacuum
M_vac = diag(g, 1, delta, 0) with g = 8, delta = 0.3 (M eta has the spectrum
(-g, 1, delta, 0)). Jets A_mu = d_mu M, and with ^T the internal transpose
    P_mu nu = A_mu eta A_nu,  F = P - P^T,  G = P + P^T,
    <X, Y>_eta = sum_ab eta_a eta_b X_ab Y_ab,
    I(X) = (1/2) sum_{mu nu} eta^mu eta^nu <X_mu nu, X_mu nu>_eta,
    J(X, Y) = (1/2) sum_{mu nu} eta^mu eta^nu <X_mu nu, Y_mu nu>_eta.
u is the timelike unit eigenvector of M eta (u^T eta u = -1), Pi_u = -u u^T eta,
Pi_s = 1 - Pi_u, A^t = Pi_u A Pi_s^T + Pi_s A Pi_u^T, P^tt = A^t eta A^t,
X_a = P^tt - P^tt^T, X_s = P^tt + P^tt^T. The objects:
    I1: X = F;   GG: X = G;   T: F on spatial (mu, nu), G when an index is 0;
    Bu: X = Pi_s F Pi_s^T + Pi_u G Pi_s^T + Pi_s G Pi_u^T + Pi_u G Pi_u^T;
    Gam: X = F + 2 (P^tt)^T = F - X_a + X_s;
    Gam_tl: X = F - X_a + T_tl(Pi_s X_s Pi_s^T),
            T_tl(Y) = Y - (1/3) tr(eta Y) (eta + u u^T).
Basis at the vacuum frame (u = e_0): I1 = (1/2) F.F (all four indices with
eta), I2 = F[m n a b] F[a b m n], I3 = eta_m eta_b F[m n a b] F[m a n b],
R[n a] = sum_m F[m n a m], I4 = eta_n eta_a R[n a]^2, I5 = R[n a] R[a n],
I6 = (sum_n R[n n])^2, C6a = (sum_m eta^m tr(A_m eta A_m eta))^2,
C6b = sum_mk eta^m eta^k tr(A_m eta A_k eta)^2.

Claim 1 (the time-row map). M = O D O^T with O eta O^T = eta and
Omega_mu = O^-1 d_mu O. Then O eta O^T = eta differentiated gives
Omega eta + eta Omega^T = 0 (Omega eta antisymmetric: symmetric time row,
antisymmetric spatial block, zero diagonal) and the product rule gives
    d_mu M = O (Omega_mu D + D Omega_mu^T + d_mu D) O^T,
    A~[0 i] = Omega[0 i] (lambda_i + lambda_0),
    A~[i j] = Omega[i j] (lambda_j - lambda_i) (i != j spatial),
    A~[i i] = d lambda_i,   A~ = O^-1 A O^-T.
Here the lift is O(t) = exp(rot(r(t))) Cayley(boost(a(t))), with
Cayley(K) = (1 + K)(1 - K)^-1 (a Lorentz matrix for K eta antisymmetric),
and every derivative is a Richardson 4-point finite difference in t; u comes
from this file's own eigen-solve of M eta, not from the lift.

Claim 2 (identities). F and X_a are antisymmetric and X_s symmetric under
mu <-> nu, and the weight eta^mu eta^nu is symmetric, so the full
contraction <F - X_a, X_s> = sum eta^mu eta^nu <..> vanishes by the (mu, nu)
parity alone: for any jets (symmetric or not), any u (normalised or not),
and in both F conventions in use (the c9_c12 code swaps the derivative
pair, the entrants code transposes the internal pair; they coincide on
d_mu M and differ on non-symmetric jets). Then Gam = I(F - X_a) + I(X_s)
and, with I(F - X_a) = I1 + I(X_a) - 2 J(F, X_a),
    Gam - I1 = I(X_a) + I(X_s) - 2 J(F, X_a).
Sharper: with F = F^ss + F^mx + X_a (F^ss = Pi_s F Pi_s^T - X_a the
spatial-spatial part, F^mx the two off-diagonal blocks) every cross term
vanishes and Gam = I(F^ss) + I(F^mx) + I(X_s).

Claim 3 (rank). Ranks by the singular-value gap of the column-normalised
density matrix over several jet distributions and restricted families.

Claim 4 (certificate). Static jets x in R^30 (A_1, A_2, A_3 symmetric),
clock A_0 = omega a0, a0 = Gen M_vac + M_vac Gen^T. X_ij depends on
(A_i, A_j) only, X_0i and X_i0 are linear in x at fixed a0, X_00 depends on
a0 only, so every density is exactly
    I_X(omega; x) = A_X(x) + omega^2 x^T Q_X x + omega^4 D_X,
    Q_X = -(1/2) sum_i (L_0i^T W L_0i + L_i0^T W L_i0),   W = diag(eta_a eta_b),
    D_X = (1/2) <X_00(a0, a0), X_00(a0, a0)>_eta,
with L_0i the 16x30 matrix of x -> X_0i (built column by column on the
basis, NOT by polarisation of the quartic). H2(c) = -4 [(1 - c) Q_I1 + c Q_X]
is the rigid-clock kinetic form; the c search maximises over c the minimum
over the six Lorentz channels of lambda_min(H2(c)) (a concave function of
c, so a grid plus a golden-section refinement finds its maximum). The
Legendre transform of L = -4 (A + C omega^2 + D omega^4) is
    H = omega dL/domega - L = 4 A - 4 C omega^2 - 12 D omega^4.
Own closed forms tested as a sharpening: lambda_min(-4 Q_I1) =
-16 (lambda_k + lambda_0)^2 on boost_k and -8 (lambda_i - lambda_j)^2 on
rot_k ({i, j} the two spatial indices other than k).

Claim 5 (gradients). E(M) from m5_32_r19_entrants.energy_grad; the
directional derivative along dM is compared with a Richardson 4-point
finite difference at two step sizes (the smaller residual is kept, so the
truncation and roundoff floors are separated); the relative error is
|fd - <G, dM>| / (|G| |dM|). The object part is isolated as
E_X - E_I1 (both at c = 1, V4 cancels) so that a large V4 gradient cannot
hide a curvature-gradient error. Fields: a smooth random field boost-dressed
with rapidity 0.5 (built here with per-cell matrix exponentials), a field
with a nearly degenerate spatial spectrum, and a field whose timelike gap is
about 0.5 (a spatial eigenvalue of M eta near -g + 0.5).

Claim 6 (C12). The single hedgehog seed (m5_21_4_a_pair.seed_pair single,
isotropic blend at the core) and the rigid director null (the same tensor,
centers = []), embedded by m5_21_3_a_4d.embed34. I1 per cell from this file's
own jets (np.gradient, central in the interior; and the one-sided average as
a second stencil). Because the seed is analytic in x, the core ball r <= 3
is independent of the box, so the ladder runs h = 3, 2, 1.5, 1, 0.75, 0.5,
0.375 with n in {16, 24} and L in {48, 24, 12, 9}. The rigid null's core
ball is read through its local exponent per h pair and fitted as
a/h + b over h <= 1 (a point defect with density 1/r^4 integrated over
h < r < 3 gives a (1/h - 1/3): the local exponent sits above 1 at finite h
and trends to 1); the melted seed's core ball is reported with its max/min
ratio where the ball is resolved (h <= 1.5) and split into the point ball
r <= 1.5, the axis tube (rho <= 1, r > 1.5) and the remainder. The growth partition
(core ball, axis cylinder rho <= 1.5, the rest) is read at fixed L = 24
(h 1.5 -> 1) and L = 12 (h 0.75 -> 0.5).

Every printed line is PASS/FAIL on a number that can go either way.
Out: ../data/m5_32_r19_0_audit.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(DATA, "m5_32_r19_0_audit.json")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


LAG = _load("m5_32_lagrangian", "m5_32_lagrangian.py")
B3 = LAG.B3
PAIR = _load("m5_21_4_a_pair", "m5_21_4_a_pair.py")
ENT = _load("m5_32_r19_entrants", "m5_32_r19_entrants.py")   # energy_grad under test, nothing else

E = np.array([-1.0, 1.0, 1.0, 1.0])
ETA = np.diag(E)
I4 = np.eye(4)
G8, DELTA = 8.0, 0.3
MVAC = np.diag([G8, 1.0, DELTA, 0.0])
U0 = np.array([[1.0, 0.0, 0.0, 0.0]])
OBJECTS = ("I1", "GG", "T", "Bu", "Gam", "Gam_tl")
T0 = time.time()
LINES = {}


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def line(key, ok, detail):
    LINES[key] = {"pass": bool(ok), "detail": detail}
    print(f"{'PASS' if ok else 'FAIL'} {key}: {detail}", flush=True)


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ================= own algebra, cell-first layout =================
# A: (N, 4, 4, 4) = [cell, mu, a, b];  X: (N, 4, 4, 4, 4) = [cell, mu, nu, a, b]
def P_of(A):
    return np.einsum("nmab,bc,nkcd->nmkad", A, ETA, A, optimize=True)


def tI(X):
    return X.swapaxes(-1, -2)


def pair(X, Y):
    return np.einsum("m,k,a,b,nmkab,nmkab->n", E, E, E, E, X, Y, optimize=True)


def dens(X):
    return 0.5 * pair(X, X)


def J(X, Y):
    return 0.5 * pair(X, Y)


def projectors(u):
    Pu = -np.einsum("na,nb,b->nab", u, u, E)
    return Pu, I4[None] - Pu


def sw(L, X, R):
    return np.einsum("nab,nmkbc,ndc->nmkad", L, X, R, optimize=True)


def sw1(L, A, R):
    return np.einsum("nab,nmbc,ndc->nmad", L, A, R, optimize=True)


def timerow(A, u):
    Pu, Ps = projectors(u)
    return sw1(Pu, A, Ps) + sw1(Ps, A, Pu)


def X_of(A, u, name):
    P = P_of(A)
    F = P - tI(P)
    G = P + tI(P)
    if name == "I1":
        return F
    if name == "GG":
        return G
    if name == "T":
        X = F.copy()
        X[:, 0] = G[:, 0]
        X[:, :, 0] = G[:, :, 0]
        return X
    Pu, Ps = projectors(u)
    if name == "Bu":
        return sw(Ps, F, Ps) + sw(Pu, G, Ps) + sw(Ps, G, Pu) + sw(Pu, G, Pu)
    At = sw1(Pu, A, Ps) + sw1(Ps, A, Pu)
    Ptt = P_of(At)
    if name == "Gam":
        return F + 2.0 * tI(Ptt)
    if name == "Gam_tl":
        Xa = Ptt - tI(Ptt)
        Xs = Ptt + tI(Ptt)
        Y = sw(Ps, Xs, Ps)
        trY = np.einsum("a,nmkaa->nmk", E, Y)
        gs = ETA[None] + np.einsum("na,nb->nab", u, u)
        return F - Xa + Y - (trY / 3.0)[..., None, None] * gs[:, None, None]
    raise ValueError(name)


def timelike_u(M):
    """own eigen-solve: the eta-negative-norm eigenvector of M eta, u^T eta u = -1; gap to the rest."""
    lam, V = np.linalg.eig(M @ ETA)
    lam = lam.real
    V = V.real
    n2 = np.einsum("nak,a,nak->nk", V, E, V)
    k0 = np.argmin(n2, axis=-1)
    u = np.take_along_axis(V, k0[:, None, None], axis=-1)[..., 0]
    nu = np.take_along_axis(n2, k0[:, None], axis=-1)[:, 0]
    u = u / np.sqrt(-nu)[:, None]
    u = u * np.where(u[:, :1] < 0, -1.0, 1.0)
    l0 = np.take_along_axis(lam, k0[:, None], axis=-1)[:, 0]
    d = np.abs(l0[:, None] - lam)
    np.put_along_axis(d, k0[:, None], np.inf, axis=-1)
    return u, d.min(axis=-1), (n2 < 0).sum(axis=-1)


def gen_rot(r):
    G = np.zeros((4, 4))
    G[1, 2], G[1, 3], G[2, 3] = -r[2], r[1], -r[0]
    G[2, 1], G[3, 1], G[3, 2] = r[2], -r[1], r[0]
    return G


def gen_boost(a):
    G = np.zeros((4, 4))
    G[0, 1:] = a
    G[1:, 0] = a
    return G


def cayley(K):
    """(1 + K)(1 - K)^-1, Lorentz for K eta antisymmetric."""
    return np.linalg.solve((I4 - K).T, (I4 + K).T).T


def richardson(f, h=1e-3):
    return (8.0 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12.0 * h)


def sym_jets(rng, N, dist="normal", nmu=4):
    if dist == "normal":
        X = rng.normal(size=(N, nmu, 4, 4))
    elif dist == "uniform":
        X = rng.uniform(-1.0, 1.0, size=(N, nmu, 4, 4))
    elif dist == "heavy":
        X = rng.standard_t(2, size=(N, nmu, 4, 4))
    else:
        raise ValueError(dist)
    return X + X.swapaxes(-1, -2)


def random_lorentz(rng, rap=1.0, rot=1.0):
    a = rng.normal(size=3)
    a = a / np.linalg.norm(a) * rap * rng.uniform(0.3, 1.0)
    return expm(gen_boost(a)) @ expm(gen_rot(rot * rng.normal(size=3)))


# ================= claim 1: the time-row map =================
def audit_1(rng, n_trials=48):
    keys = ["map_exact", "timerow_factor", "spatial_factor", "diag_is_dD", "omega_eta_antisym",
            "At_vs_boost_only", "Ass_vs_rot_plus_dDs", "corner_vs_dD0", "u_vs_lift_e0", "melt_kills_rotation_entries"]
    worst = {k: 0.0 for k in keys}
    paper_ctrl = np.inf
    for t in range(n_trials):
        kind = t % 3
        if kind == 0:
            D0 = MVAC + np.diag([0.0, *(0.1 * rng.normal(size=3))])
        elif kind == 1:
            D0 = np.diag([G8, 0.5, 0.5, 0.5])              # fully melted spatial spectrum
        else:
            D0 = np.diag([G8, 1.0, 1.0, 0.0])              # a partial degeneracy
        dD = np.diag(0.3 * rng.normal(size=4))
        r0, r1 = 0.7 * rng.normal(size=3), rng.normal(size=3)
        a0 = rng.normal(size=3)
        a0 = a0 / np.linalg.norm(a0) * rng.uniform(0.1, 0.55)
        a1 = 0.4 * rng.normal(size=3)

        def O_of(s):
            return expm(gen_rot(r0 + s * r1)) @ cayley(gen_boost(a0 + s * a1))

        def M_of(s):
            O = O_of(s)
            return O @ (D0 + s * dD) @ O.T

        O = O_of(0.0)
        assert rel(O @ ETA @ O.T, ETA) < 1e-12
        dO = richardson(O_of)
        dM = richardson(M_of)
        Oi = np.linalg.inv(O)
        Om = Oi @ dO
        OmE = Om @ ETA
        worst["omega_eta_antisym"] = max(worst["omega_eta_antisym"], float(np.abs(OmE + OmE.T).max() / np.abs(OmE).max()))
        OT = O.T @ dO
        OTE = OT @ ETA
        paper_ctrl = min(paper_ctrl, float(np.abs(OTE + OTE.T).max() / np.abs(OTE).max()))
        Af = Oi @ dM @ Oi.T
        scale = np.abs(Af).max()
        worst["map_exact"] = max(worst["map_exact"], rel(Af, Om @ D0 + D0 @ Om.T + dD))
        lam = np.diag(D0)
        for i in (1, 2, 3):
            pred = Om[0, i] * (lam[i] + lam[0])
            worst["timerow_factor"] = max(worst["timerow_factor"], abs(Af[0, i] - pred) / scale, abs(Af[i, 0] - pred) / scale)
            worst["diag_is_dD"] = max(worst["diag_is_dD"], abs(Af[i, i] - dD[i, i]) / scale)
        worst["diag_is_dD"] = max(worst["diag_is_dD"], abs(Af[0, 0] - dD[0, 0]) / scale)
        for i, j in ((1, 2), (1, 3), (2, 3)):
            pred = Om[i, j] * (lam[j] - lam[i])
            worst["spatial_factor"] = max(worst["spatial_factor"], abs(Af[i, j] - pred) / scale, abs(Af[j, i] - pred) / scale)
            if kind == 1:
                worst["melt_kills_rotation_entries"] = max(worst["melt_kills_rotation_entries"], abs(Af[i, j]) / scale)
        # frame-free, with this file's own u
        u, gap, ntime = timelike_u(M_of(0.0)[None])
        assert ntime[0] == 1
        worst["u_vs_lift_e0"] = max(worst["u_vs_lift_e0"], min(rel(u[0], O[:, 0]), rel(u[0], -O[:, 0])))
        Pu, Ps = projectors(u)
        A1 = dM[None, None]
        At = (sw1(Pu, A1, Ps) + sw1(Ps, A1, Pu))[0, 0]
        Ass = sw1(Ps, A1, Ps)[0, 0]
        Ac = sw1(Pu, A1, Pu)[0, 0]
        Omb = np.zeros((4, 4)); Omb[0, 1:] = Om[0, 1:]; Omb[1:, 0] = Om[1:, 0]
        Omr = Om.copy(); Omr[0, :] = 0.0; Omr[:, 0] = 0.0
        dDs = dD.copy(); dDs[0, 0] = 0.0
        dD0 = np.zeros((4, 4)); dD0[0, 0] = dD[0, 0]
        worst["At_vs_boost_only"] = max(worst["At_vs_boost_only"], np.abs(At - O @ (Omb @ D0 + D0 @ Omb.T) @ O.T).max() / scale)
        worst["Ass_vs_rot_plus_dDs"] = max(worst["Ass_vs_rot_plus_dDs"], np.abs(Ass - O @ (Omr @ D0 + D0 @ Omr.T + dDs) @ O.T).max() / scale)
        worst["corner_vs_dD0"] = max(worst["corner_vs_dD0"], np.abs(Ac - O @ dD0 @ O.T).max() / scale)
    out = {"worst_rel": worst, "paper_OT_dO_eta_antisym_control_min": float(paper_ctrl), "n_trials": n_trials}
    line("A1a_time_row_map_exact_on_own_lift", worst["map_exact"] < 1e-8 and worst["timerow_factor"] < 1e-8 and worst["spatial_factor"] < 1e-8 and worst["diag_is_dD"] < 1e-8,
         f"worst rel map {worst['map_exact']:.1e}, time-row factor {worst['timerow_factor']:.1e}, spatial factor {worst['spatial_factor']:.1e}, diag {worst['diag_is_dD']:.1e}")
    line("A1b_omega_eta_antisymmetric_and_paper_OTdO_is_not", worst["omega_eta_antisym"] < 1e-8 and paper_ctrl > 1e-2,
         f"Omega eta asym rel {worst['omega_eta_antisym']:.1e}; O^T dO control min {paper_ctrl:.2e}")
    line("A1c_frame_free_At_is_the_boost_only_jet_with_own_u", max(worst[k] for k in ("At_vs_boost_only", "Ass_vs_rot_plus_dDs", "corner_vs_dD0", "u_vs_lift_e0")) < 1e-8,
         f"At {worst['At_vs_boost_only']:.1e}, Ass {worst['Ass_vs_rot_plus_dDs']:.1e}, corner {worst['corner_vs_dD0']:.1e}, u vs O e0 {worst['u_vs_lift_e0']:.1e}")
    line("A1d_melted_spectrum_hides_rotation_keeps_time_row", worst["melt_kills_rotation_entries"] < 1e-8,
         f"spatial off-diagonal of the frame jet at lambda_1=lambda_2=lambda_3: {worst['melt_kills_rotation_entries']:.1e} (time row intact by A1a)")
    return out


# ================= claim 2: the identities =================
def audit_2(rng, N=1500):
    out = {}
    # random u: boosted, normalised and NOT normalised
    Ls = [random_lorentz(rng, rap=1.5) for _ in range(N)]
    u_n = np.stack([L[:, 0] for L in Ls])
    u_x = u_n * rng.uniform(0.5, 2.0, size=(N, 1))
    A = sym_jets(rng, N)
    res = {}
    for tag, u in (("u_normalised", u_n), ("u_unnormalised", u_x), ("u_e0", np.tile(U0, (N, 1)))):
        P = P_of(A); F = P - tI(P)
        At = timerow(A, u); Ptt = P_of(At); Xa = Ptt - tI(Ptt); Xs = Ptt + tI(Ptt)
        Gam = dens(X_of(A, u, "Gam")); I1 = dens(F)
        sc = float(np.abs(Gam).max())
        res[tag] = {"cross_FminusXa_Xs_rel": float(np.abs(pair(F - Xa, Xs)).max() / sc),
                    "decomp_Gam_minus_I1_rel": float(np.abs(Gam - I1 - dens(Xa) - dens(Xs) + 2.0 * J(F, Xa)).max() / sc)}
        if tag == "u_normalised":
            Pu, Ps = projectors(u)
            Fss = sw(Ps, F, Ps) - Xa
            Fmx = sw(Pu, F, Ps) + sw(Ps, F, Pu)
            Fcorner = sw(Pu, F, Pu)
            res[tag]["fine_decomp_Gam=I(Fss)+I(Fmx)+I(Xs)_rel"] = float(np.abs(Gam - dens(Fss) - dens(Fmx) - dens(Xs)).max() / sc)
            res[tag]["fine_decomp_I1=I(Fss)+I(Fmx)+I(Xa)+2J(Fss,Xa)_rel"] = float(np.abs(I1 - dens(Fss) - dens(Fmx) - dens(Xa) - 2.0 * J(Fss, Xa)).max() / sc)
            res[tag]["Pu_F_Pu_max"] = float(np.abs(Fcorner).max() / np.abs(F).max())
            res[tag]["cross_Fss_Fmx_rel"] = float(np.abs(pair(Fss, Fmx)).max() / sc)
            res[tag]["I_Fmx_max"] = float(dens(Fmx).max() / sc)
            res[tag]["I_Xa_min"] = float(dens(Xa).min() / sc)
    out["symmetric_jets"] = res
    # (a) Gam = I1 when the time row vanishes in the u frame, u boosted
    Af = sym_jets(rng, N)
    Af[:, :, 0, 1:] = 0.0; Af[:, :, 1:, 0] = 0.0
    Ab = np.einsum("nab,nmbc,ndc->nmad", np.stack(Ls), Af, np.stack(Ls))
    Gb = dens(X_of(Ab, u_n, "Gam")); Gtl = dens(X_of(Ab, u_n, "Gam_tl")); Bb = dens(X_of(Ab, u_n, "Bu")); Ib = dens(X_of(Ab, u_n, "I1"))
    sc = float(np.abs(Ib).max())
    # Bu keeps the corner: in the u frame Pi_u G Pi_u^T = -2 c_mu c_nu u u^T with c_mu = u^T eta A_mu eta u,
    # so Bu - I1 = 2 (sum_mu eta^mu c_mu^2)^2 whenever A^t = 0 (own closed form, tested here)
    cmu = np.einsum("na,a,nmab,b,nb->nm", u_n, E, Ab, E, u_n)
    pred = 2.0 * np.einsum("m,nm->n", E, cmu ** 2) ** 2
    # and with the corner jet zero as well (the producer's Coulomb family) Bu = I1
    Ac = Af.copy(); Ac[:, :, 0, 0] = 0.0
    Ac = np.einsum("nab,nmbc,ndc->nmad", np.stack(Ls), Ac, np.stack(Ls))
    Bc = dens(X_of(Ac, u_n, "Bu")); Ic = dens(X_of(Ac, u_n, "I1"))
    out["coulomb_in_boosted_frame"] = {"Gam_minus_I1_rel": float(np.abs(Gb - Ib).max() / sc), "Gam_tl_minus_I1_rel": float(np.abs(Gtl - Ib).max() / sc),
                                       "Bu_minus_I1_rel_corner_free": float(np.abs(Bb - Ib).max() / sc), "At_norm_rel": float(np.abs(timerow(Ab, u_n)).max() / np.abs(Ab).max()),
                                       "Bu_minus_I1_vs_closed_form_rel": float(np.abs(Bb - Ib - pred).max() / np.abs(Bb - Ib).max()),
                                       "Bu_minus_I1_rel_corner_zero": float(np.abs(Bc - Ic).max() / max(np.abs(Ic).max(), 1e-300))}
    # the control: with the producer's F = P - P^(mu<->nu) (m5_32_r19_c9_c12 swaps the DERIVATIVE pair) F is
    # internally antisymmetric only for symmetric jets; on non-symmetric jets the cross term survives.
    # With this file's F = P - P^T (internal transpose, the entrants' convention) it never does.
    An = rng.normal(size=(N, 4, 4, 4))
    P = P_of(An); Fd = P - P.swapaxes(1, 2); Fi = P - tI(P)
    At = timerow(An, u_n); Ptt = P_of(At); Xa_d = Ptt - Ptt.swapaxes(1, 2); Xs_d = Ptt + Ptt.swapaxes(1, 2)
    Xa_i = Ptt - tI(Ptt); Xs_i = Ptt + tI(Ptt)
    scd = float(np.abs(dens(Fd - Xa_d + Xs_d)).max()); sci = float(np.abs(dens(Fi - Xa_i + Xs_i)).max())
    out["nonsymmetric_jets_control"] = {"deriv_swap_F_cross_FminusXa_Xs_rel": float(np.abs(pair(Fd - Xa_d, Xs_d)).max() / scd),
                                        "deriv_swap_F_internal_antisym_defect_rel": float(np.abs(Fd + tI(Fd)).max() / np.abs(Fd).max()),
                                        "internal_transpose_F_cross_rel": float(np.abs(pair(Fi - Xa_i, Xs_i)).max() / sci),
                                        "two_F_conventions_differ_on_nonsymmetric_jets_rel": rel(Fd, Fi)}
    # degenerate spectrum u (from an M with a melted spatial block, boosted)
    Md = np.stack([L @ np.diag([G8, 0.5, 0.5, 0.5]) @ L.T for L in Ls[:200]])
    ud, gapd, nt = timelike_u(Md)
    Ad = sym_jets(rng, 200)
    P = P_of(Ad); F = P - tI(P)
    At = timerow(Ad, ud); Ptt = P_of(At); Xa = Ptt - tI(Ptt); Xs = Ptt + tI(Ptt)
    Gam = dens(X_of(Ad, ud, "Gam")); I1 = dens(F); sc = float(np.abs(Gam).max())
    out["degenerate_spectrum_u"] = {"n_timelike_ok": int(np.all(nt == 1)), "u_vs_L_e0_rel": float(max(min(rel(ud[i], Ls[i][:, 0]), rel(ud[i], -Ls[i][:, 0])) for i in range(200))),
                                    "cross_rel": float(np.abs(pair(F - Xa, Xs)).max() / sc),
                                    "decomp_rel": float(np.abs(Gam - I1 - dens(Xa) - dens(Xs) + 2.0 * J(F, Xa)).max() / sc)}
    c = out["coulomb_in_boosted_frame"]
    line("A2a_Gam_and_Gam_tl_equal_I1_when_u_frame_time_row_vanishes_boosted_u", max(c["Gam_minus_I1_rel"], c["Gam_tl_minus_I1_rel"]) < 1e-12,
         f"rel Gam {c['Gam_minus_I1_rel']:.1e}, Gam_tl {c['Gam_tl_minus_I1_rel']:.1e} at |A^t|/|A| = {c['At_norm_rel']:.1e}")
    line("A2a2_Bu_needs_the_corner_jet_zero_too_closed_form_2_sum_eta_c2_squared", c["Bu_minus_I1_rel_corner_free"] > 1e-3 and c["Bu_minus_I1_vs_closed_form_rel"] < 1e-12 and c["Bu_minus_I1_rel_corner_zero"] < 1e-12,
         f"A^t = 0 only: Bu - I1 rel {c['Bu_minus_I1_rel_corner_free']:.2e}, equals 2 (sum eta^mu c_mu^2)^2 to {c['Bu_minus_I1_vs_closed_form_rel']:.1e}; corner zero too: {c['Bu_minus_I1_rel_corner_zero']:.1e}")
    w = max(res[t]["cross_FminusXa_Xs_rel"] for t in res)
    nc = out["nonsymmetric_jets_control"]
    line("A2b_cross_term_vanishes_unconditionally_mu_nu_parity_both_F_conventions_any_jets_any_u", w < 1e-12 and nc["deriv_swap_F_cross_FminusXa_Xs_rel"] < 1e-12 and nc["internal_transpose_F_cross_rel"] < 1e-12 and nc["two_F_conventions_differ_on_nonsymmetric_jets_rel"] > 1e-2,
         f"symmetric jets, three u families: {w:.1e}; NON-symmetric jets: derivative-swap F (c9_c12 code) {nc['deriv_swap_F_cross_FminusXa_Xs_rel']:.1e}, internal-transpose F (entrants code) {nc['internal_transpose_F_cross_rel']:.1e}, while the two F differ by {nc['two_F_conventions_differ_on_nonsymmetric_jets_rel']:.2f} there")
    w2 = max(max(res[t]["decomp_Gam_minus_I1_rel"] for t in res), out["degenerate_spectrum_u"]["decomp_rel"])
    line("A2c_Gam_minus_I1_decomposition_and_fine_three_term_split", w2 < 1e-12 and res["u_normalised"]["fine_decomp_Gam=I(Fss)+I(Fmx)+I(Xs)_rel"] < 1e-12,
         f"Gam - I1 = I(Xa) + I(Xs) - 2J(F,Xa): {w2:.1e}; Gam = I(Fss) + I(Fmx) + I(Xs): {res['u_normalised']['fine_decomp_Gam=I(Fss)+I(Fmx)+I(Xs)_rel']:.1e}")
    return out


# ================= claim 3: the span =================
def basis8(A):
    P = P_of(A); F = P - tI(P)
    d = {}
    d["I1"] = 0.5 * np.einsum("m,k,a,b,nmkab,nmkab->n", E, E, E, E, F, F, optimize=True)
    d["I2"] = np.einsum("nmkab,nabmk->n", F, F, optimize=True)
    d["I3"] = np.einsum("m,b,nmkab,nmakb->n", E, E, F, F, optimize=True)
    R = np.einsum("nmkam->nka", F)
    d["I4"] = np.einsum("k,a,nka,nka->n", E, E, R, R)
    d["I5"] = np.einsum("nka,nak->n", R, R)
    d["I6"] = np.einsum("nkk->n", R) ** 2
    AE = A @ ETA
    tr2 = np.einsum("nmab,nkba->nmk", AE, AE, optimize=True)
    d["C6a"] = np.einsum("m,nmm->n", E, tr2) ** 2
    d["C6b"] = np.einsum("m,k,nmk,nmk->n", E, E, tr2, tr2)
    return d


def svd_rank(cols, tol=1e-8):
    X = np.stack(cols, 1)
    X = X / np.linalg.norm(X, axis=0)
    s = np.linalg.svd(X, compute_uv=False)
    r = int(np.sum(s > tol * s[0]))
    gap = float(s[r - 1] / s[r]) if r < len(s) else float("inf")
    return r, gap, s


def audit_3(rng, N=600):
    out = {}
    # gate: own basis equals the registry's on the same jets
    A = sym_jets(rng, 200)
    mine = basis8(A)
    Fa = LAG.F_of_A(A.transpose(1, 0, 2, 3))
    w = max(rel(mine[k], LAG.density_from_K(Fa, LAG.REGISTRY[k]._K())) for k in ("I1", "I2", "I3", "I4", "I5", "I6"))
    out["own_basis_vs_registry_rel"] = w
    line("A3a_own_basis_densities_equal_the_registry", w < 1e-10, f"worst rel {w:.1e} over I1..I6 (C6a, C6b are not registry terms)")
    fam = {}
    ladders = {}
    for name in ("normal_full", "uniform_full", "heavy_full", "normal_static", "static_zero_time_row", "static_only_time_row", "full_only_time_row", "static_zero_time_row_and_corner"):
        dist = name.split("_")[0] if name.split("_")[0] in ("normal", "uniform", "heavy") else "normal"
        A = sym_jets(rng, N, dist)
        if "static" in name:
            A[:, 0] = 0.0
        if "zero_time_row" in name:
            A[:, :, 0, 1:] = 0.0; A[:, :, 1:, 0] = 0.0
            if "corner" not in name:
                pass
            else:
                A[:, :, 0, 0] = 0.0
        if "only_time_row" in name:
            B = np.zeros_like(A); B[:, :, 0, 1:] = A[:, :, 0, 1:]; B[:, :, 1:, 0] = A[:, :, 1:, 0]; A = B
        u = np.tile(U0, (N, 1))
        d = basis8(A)
        cols = [d[k] for k in d]
        ob = {k: dens(X_of(A, u, k)) for k in ("GG", "Bu", "Gam", "Gam_tl", "T")}
        lad = {}
        r, gap, s = svd_rank(cols); lad["basis8"] = [r, gap]
        for add in (["GG"], ["GG", "Bu"], ["GG", "Bu", "Gam"], ["GG", "Bu", "Gam", "Gam_tl"]):
            r, gap, s = svd_rank(cols + [ob[k] for k in add]); lad["+" + "+".join(add)] = [r, gap]
        r, gap, s = svd_rank(cols + [ob[k] for k in ("GG", "Bu", "Gam", "Gam_tl", "T")]); lad["+all_five_incl_T"] = [r, gap]
        lad["sv_ratio_12"] = float(s[-1] / s[0]) if len(s) >= 13 else None
        ladders[name] = lad
        fam[name] = {"singular_values_all13": s.tolist()}
    out["ranks"] = ladders
    out["families"] = fam
    ok_main = all(ladders[n]["basis8"][0] == 8 and ladders[n]["+GG"][0] == 9 and ladders[n]["+GG+Bu"][0] == 10 and ladders[n]["+GG+Bu+Gam"][0] == 11 and ladders[n]["+GG+Bu+Gam+Gam_tl"][0] == 12
                  for n in ("normal_full", "uniform_full", "heavy_full"))
    gaps = {n: ladders[n]["+GG+Bu+Gam+Gam_tl"][1] for n in ("normal_full", "uniform_full", "heavy_full")}
    line("A3b_rank_8_9_10_11_12_on_three_distributions_by_svd_gap", ok_main and min(gaps.values()) > 1e3,
         f"ranks {[ladders[n]['+GG+Bu+Gam+Gam_tl'][0] for n in ('normal_full', 'uniform_full', 'heavy_full')]}, sv gap at rank 12 {min(gaps.values()):.1e}")
    # static jets: the basis has an exact relation (found by the null singular vector, then tested as a line)
    A = sym_jets(rng, N); A[:, 0] = 0.0
    d = basis8(A)
    relation = float(np.abs(d["I2"] - 4.0 * d["I5"] + d["I6"]).max() / np.abs(d["I5"]).max())
    Af_ = sym_jets(rng, N); df = basis8(Af_)
    relation_full = float(np.abs(df["I2"] - 4.0 * df["I5"] + df["I6"]).max() / np.abs(df["I5"]).max())
    out["static_relation_I2_minus_4I5_plus_I6_rel"] = {"static": relation, "full_jets_control": relation_full}
    st = ladders["normal_static"]
    line("A3c_static_jets_basis_rank_7_by_I2_minus_4I5_plus_I6_equals_0_then_ladder_8_9_10_11", st["basis8"][0] == 7 and relation < 1e-12 and relation_full > 1e-2
         and [st[k][0] for k in ("+GG", "+GG+Bu", "+GG+Bu+Gam", "+GG+Bu+Gam+Gam_tl")] == [8, 9, 10, 11] and st["+all_five_incl_T"][0] == 11,
         f"static: basis8 {st['basis8'][0]}, relation rel {relation:.1e} (full-jet control {relation_full:.2f}); +GG {st['+GG'][0]}, +Bu {st['+GG+Bu'][0]}, +Gam {st['+GG+Bu+Gam'][0]}, +Gam_tl {st['+GG+Bu+Gam+Gam_tl'][0]}, +T {st['+all_five_incl_T'][0]} (T = I1 there)")
    cz = ladders["static_zero_time_row"]; czc = ladders["static_zero_time_row_and_corner"]
    line("A3d_coulomb_family_collapses_Gam_Gam_tl_always_and_Bu_only_with_the_corner_zero", cz["+GG+Bu+Gam"][0] == cz["+GG+Bu"][0] == cz["+GG+Bu+Gam+Gam_tl"][0] and cz["+GG+Bu"][0] == cz["+GG"][0] + 1
         and czc["+GG+Bu+Gam+Gam_tl"][0] == czc["+GG"][0],
         f"zero time row, corner free: basis8 {cz['basis8'][0]}, +GG {cz['+GG'][0]}, +Bu {cz['+GG+Bu'][0]}, +Gam {cz['+GG+Bu+Gam'][0]}, +Gam_tl {cz['+GG+Bu+Gam+Gam_tl'][0]}; corner zero too: +GG {czc['+GG'][0]}, +all {czc['+GG+Bu+Gam+Gam_tl'][0]}")
    nf = ladders["normal_full"]
    line("A3f_T_is_a_13th_direction_on_full_jets", nf["+all_five_incl_T"][0] == 13, f"full jets: basis8 + GG + Bu + Gam + Gam_tl + T rank {nf['+all_five_incl_T'][0]}")
    tr_ = ladders["static_only_time_row"]
    out["only_time_row_note"] = f"static only-time-row family: basis8 rank {tr_['basis8'][0]}, +GG+Bu+Gam+Gam_tl {tr_['+GG+Bu+Gam+Gam_tl'][0]}"
    line("A3e_only_time_row_family_is_degenerate_for_the_basis", tr_["basis8"][0] < 8, out["only_time_row_note"])
    return out


# ================= claim 4: the certificate =================
IU = np.triu_indices(4)


def static_from_x(x):
    A = np.zeros((4, 4, 4))
    for i in range(3):
        S = np.zeros((4, 4)); S[IU] = x[10 * i:10 * i + 10]
        A[1 + i] = S + S.T - np.diag(np.diag(S))
    return A


def gram_Q(a0, name):
    """Q_X (30x30) by the linear maps x -> X_0i, X_i0 (a0 fixed, omega = 1), and D_X from X_00."""
    cols = []
    for p in range(30):
        x = np.zeros(30); x[p] = 1.0
        A = static_from_x(x); A[0] = a0
        X = X_of(A[None], U0, name)[0]
        cols.append(np.stack([X[0, 1:], X[1:, 0]]))         # (2, 3, 4, 4)
    Lm = np.stack(cols, -1).reshape(2, 3, 16, 30)
    Wab = np.outer(E, E).reshape(16)
    Q = np.zeros((30, 30))
    for s in range(2):
        for i in range(3):
            Q += Lm[s, i].T @ (Wab[:, None] * Lm[s, i])
    Q *= -0.5                                               # eta^0 eta^i = -1, the 1/2 of I(X)
    A = np.zeros((4, 4, 4)); A[0] = a0
    X00 = X_of(A[None], U0, name)[0][0, 0]
    D = 0.5 * float(np.einsum("a,b,ab,ab", E, E, X00, X00))
    return Q, D


def channels():
    ch = {}
    for k in (1, 2, 3):
        e = np.zeros(3); e[k - 1] = 1.0
        G = gen_boost(e); ch[f"boost_{k}"] = G @ MVAC + MVAC @ G.T
        G = gen_rot(e); ch[f"rot_{k}"] = G @ MVAC + MVAC @ G.T
    return ch


def lam_min(H):
    return float(np.linalg.eigvalsh(0.5 * (H + H.T))[0])


def best_c(QI, QX, chans, lo, hi, ngrid=1201):
    def f(c):
        return min(lam_min(-4.0 * ((1 - c) * QI[n] + c * QX[n])) for n in chans)
    cs = np.linspace(lo, hi, ngrid)
    vals = np.array([f(c) for c in cs])
    i = int(np.argmax(vals))
    a, b = cs[max(i - 1, 0)], cs[min(i + 1, ngrid - 1)]
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c1, c2 = b - gr * (b - a), a + gr * (b - a)
    f1, f2 = f(c1), f(c2)
    for _ in range(60):
        if f1 < f2:
            a, c1, f1 = c1, c2, f2; c2 = a + gr * (b - a); f2 = f(c2)
        else:
            b, c2, f2 = c2, c1, f1; c1 = b - gr * (b - a); f1 = f(c1)
    cb = 0.5 * (a + b)
    return float(cb), float(f(cb))


def audit_4(rng):
    out = {}
    ch = channels()
    lor6 = list(ch)
    Q = {n: {} for n in ch}; D = {n: {} for n in ch}
    struct_worst = 0.0
    for n, a0 in ch.items():
        for k in OBJECTS:
            Qk, Dk = gram_Q(a0, k)
            Q[n][k] = Qk; D[n][k] = Dk
            # falsify the structure: full density at random (x, omega) vs A(x) + omega^2 x Q x + omega^4 D
            for _ in range(4):
                x = rng.normal(size=30)
                A = static_from_x(x)
                A0 = A.copy()
                base = float(dens(X_of(A0[None], U0, k))[0])
                for wv in (0.37, -1.3, 2.1):
                    Aw = A.copy(); Aw[0] = wv * a0
                    full = float(dens(X_of(Aw[None], U0, k))[0])
                    pred = base + wv ** 2 * float(x @ Qk @ x) + wv ** 4 * Dk
                    struct_worst = max(struct_worst, abs(full - pred) / max(abs(full), 1.0))
    out["structure_worst_rel"] = struct_worst
    line("A4a_even_quartic_with_gram_Q_and_constant_D_on_random_x_omega", struct_worst < 1e-9, f"worst rel {struct_worst:.1e} over 6 channels x 6 objects x 4 x x 3 omega")
    lam = np.diag(MVAC)
    closed = {f"boost_{k}": -16.0 * (lam[k] + lam[0]) ** 2 for k in (1, 2, 3)}
    for k in (1, 2, 3):
        i, j = [x for x in (1, 2, 3) if x != k]
        closed[f"rot_{k}"] = -8.0 * (lam[i] - lam[j]) ** 2
    me = {n: {k: lam_min(-4.0 * Q[n][k]) for k in OBJECTS} for n in ch}
    out["min_eig_minus4Q"] = me
    out["own_closed_form_min_eig_I1"] = closed
    claimed = {"boost_1": -1296.0, "rot_1": -0.72, "rot_2": -8.0, "rot_3": -3.92}
    w_claim = max(abs(me[n]["I1"] - v) / abs(v) for n, v in claimed.items())
    w_closed = max(abs(me[n]["I1"] - closed[n]) / abs(closed[n]) for n in ch)
    line("A4b_min_eig_of_minus4Q_I1_matches_claimed_values_and_own_closed_form", w_claim < 1e-9 and w_closed < 1e-9,
         f"claimed rel {w_claim:.1e}; closed form -16 (lambda_k + g)^2 boosts, -8 (lambda_i - lambda_j)^2 rotations rel {w_closed:.1e}; boost_2 {me['boost_2']['I1']:.4f}, boost_3 {me['boost_3']['I1']:.4f}")
    w_same = max(abs(me[n][k] - me[n]["I1"]) / abs(me[n]["I1"]) for n in ch for k in OBJECTS)
    line("A4c_min_eig_unchanged_by_every_object_on_six_lorentz_channels", w_same < 1e-9, f"worst rel {w_same:.1e}")
    # why: the minimising vector of Q_I1 sits in the kernel of Q_X - Q_I1
    kern = {}
    for n in ("boost_1", "rot_3"):
        w_, V_ = np.linalg.eigh(-4.0 * Q[n]["I1"])
        v0 = V_[:, 0]
        kern[n] = {k: float(np.linalg.norm((Q[n][k] - Q[n]["I1"]) @ v0) / max(np.linalg.norm(Q[n][k] - Q[n]["I1"]), 1e-300)) for k in OBJECTS if k != "I1"}
        kern[n]["min_eigvec_support"] = [int(i) for i in np.where(np.abs(v0) > 1e-6)[0]]
        kern[n]["diff_spectrum_Gam_minus_I1_range"] = [float(np.linalg.eigvalsh(Q[n]["Gam"] - Q[n]["I1"])[0]), float(np.linalg.eigvalsh(Q[n]["Gam"] - Q[n]["I1"])[-1])]
    out["min_eigvec_in_kernel_of_difference"] = kern
    # rotation identities
    w_rot = max(max(np.abs(Q[f"rot_{k}"][o] - Q[f"rot_{k}"]["I1"]).max(), abs(D[f"rot_{k}"][o])) for k in (1, 2, 3) for o in ("Bu", "Gam", "Gam_tl"))
    line("A4d_rotation_channels_Q_Bu_Gam_Gam_tl_equal_Q_I1_and_D_zero", w_rot < 1e-9, f"worst abs {w_rot:.1e}")
    # boost D closed forms
    out["D"] = D
    wD = 0.0
    for k in (1, 2, 3):
        v4 = (lam[k] + lam[0]) ** 4; n = f"boost_{k}"
        wD = max(wD, abs(D[n]["Gam"] - 4 * v4) / (4 * v4), abs(D[n]["Bu"] - 2 * v4) / (2 * v4), abs(D[n]["Gam_tl"] - 4 * v4 / 3) / (4 * v4 / 3),
                 abs(D[n]["GG"] - D[n]["Gam"]) / (4 * v4), abs(D[n]["T"] - D[n]["Gam"]) / (4 * v4), abs(D[n]["I1"]) / (4 * v4))
    line("A4e_boost_D_closed_forms_4_2_4over3_v4_and_GG_T_equal_Gam", wD < 1e-9, f"worst rel {wD:.1e}; boost_1 D_Gam {D['boost_1']['Gam']:.1f}")
    # the c search
    QI = {n: Q[n]["I1"] for n in ch}
    search = {}
    for k in OBJECTS:
        if k == "I1":
            continue
        QX = {n: Q[n][k] for n in ch}
        c3, v3 = best_c(QI, QX, lor6, -3.0, 3.0)
        cw, vw = best_c(QI, QX, lor6, -200.0, 200.0, ngrid=4001)
        per = {n: best_c(QI, QX, [n], -3.0, 3.0, ngrid=601) for n in ch}
        search[k] = {"joint6_best_c_in_pm3": c3, "joint6_best_min_eig": v3, "joint6_best_c_in_pm200": cw, "joint6_best_min_eig_pm200": vw,
                     "per_channel_best_c_and_min_eig_pm3": per}
    out["c_search"] = search
    worst_best = max(search[k]["joint6_best_min_eig"] for k in search)
    worst_best_w = max(search[k]["joint6_best_min_eig_pm200"] for k in search)
    summary = {k: (round(search[k]["joint6_best_c_in_pm3"], 3), round(search[k]["joint6_best_min_eig"], 2)) for k in search}
    line("A4f_no_c_makes_H2_PSD_on_the_six_channels_best_c_still_negative", worst_best < -1e-6 and worst_best_w < -1e-6,
         f"max over c in [-3,3] of min-channel lambda_min per X (best c, value): {summary}; in [-200,200] worst {worst_best_w:.2f}")
    per_any = max(search[k]["per_channel_best_c_and_min_eig_pm3"][n][1] for k in search for n in ch)
    out["per_channel_any_c_works"] = per_any
    line("A4g_not_even_a_single_channel_admits_a_c_in_pm3", per_any < -1e-6, f"best single-channel lambda_min over X, c: {per_any:.4f}")
    # Legendre
    A_, C_, D_ = 1.7, -0.6, 0.35
    wL = 0.0
    for w in (0.3, 1.1, -2.2):
        Lw = -4.0 * (A_ + C_ * w ** 2 + D_ * w ** 4)
        dL = -4.0 * (2 * C_ * w + 4 * D_ * w ** 3)
        H = w * dL - Lw
        wL = max(wL, abs(H - (4 * A_ - 4 * C_ * w ** 2 - 12 * D_ * w ** 4)))
    line("A4h_legendre_H_equals_4A_minus_4C_w2_minus_12D_w4", wL < 1e-12, f"max abs {wL:.1e}")
    return out


# ================= claim 5: the gradients =================
def own_field(rng, cfg, base, amp, rap, kdir=(0.9, 0.4, 0.2)):
    n, h, L = cfg["n"], cfg["h"], cfg["L"]
    X, Y, Z = B3.coords(n, h)
    M = np.tile(np.diag(base), (n, n, n, 1, 1))
    for _ in range(3):
        k = rng.normal(size=3) * 2 * np.pi / L
        ph = rng.uniform(0, 2 * np.pi)
        S = rng.normal(size=(4, 4)); S = amp * (S + S.T)
        M = M + np.cos(k[0] * X + k[1] * Y + k[2] * Z + ph)[..., None, None] * S
    R = np.sqrt(X * X + Y * Y + Z * Z)
    b = rap * np.exp(-(R / (0.4 * L)) ** 2)
    d = np.stack([np.sin(kdir[0] * Y) + 0.3, np.cos(kdir[1] * X) - 0.2, 0.5 + 0.2 * np.sin(kdir[2] * Z)], -1)
    d = d / np.linalg.norm(d, axis=-1)[..., None]
    Mf = M.reshape(-1, 4, 4); bf = b.ravel(); df = d.reshape(-1, 3)
    out = np.empty_like(Mf)
    for i in range(Mf.shape[0]):
        Qb = expm(bf[i] * gen_boost(df[i]))
        out[i] = Qb @ Mf[i] @ Qb.T
    out = 0.5 * (out + out.swapaxes(-1, -2))
    return out.reshape(M.shape)


def directions(rng, shape, n):
    ds = []
    for _ in range(4):
        d = rng.normal(size=shape); d = 0.5 * (d + d.swapaxes(-1, -2)); ds.append(("random_sym", d))
    d = np.zeros(shape); d[n // 2, n // 2 - 1, n // 3, 0, 2] = d[n // 2, n // 2 - 1, n // 3, 2, 0] = 1.0; ds.append(("single_cell_time_row", d))
    d = np.zeros(shape); d[n // 3, n // 2, n // 2 + 1, 0, 0] = 1.0; ds.append(("single_cell_corner", d))
    d = np.zeros(shape); d[n // 2, n // 2, n // 2, 1, 3] = d[n // 2, n // 2, n // 2, 3, 1] = 1.0; ds.append(("single_cell_spatial", d))
    d = np.zeros(shape); S = rng.normal(size=(4, 4)); d[0, n - 1, 2] = S + S.T; ds.append(("boundary_cell_sym", d))
    d = np.zeros(shape); d[n - 1, 0, 0, 0, 1] = d[n - 1, 0, 0, 1, 0] = 1.0; ds.append(("corner_cell_time_row", d))
    return [(name, d / np.sqrt(np.sum(d * d))) for name, d in ds]


def audit_5(rng, n=8, L=12.0):
    cfg = B3.base_cfg(s=-1.0, g=G8, n=n, L=L, delta=DELTA)
    p = LAG.default_params(s=-1.0, g=G8)
    fields = {"boosted_rap0.5": own_field(rng, cfg, [G8, 1.0, DELTA, 0.0], 0.15, 0.5),
              "near_degenerate_spatial": own_field(rng, cfg, [G8, 0.60, 0.57, 0.55], 0.02, 0.3),
              "small_gap_0.5": own_field(rng, cfg, [G8, -7.5, DELTA, 0.0], 0.03, 0.3)}
    out = {"n": n, "L": L}
    eps_list = (1e-3, 3e-4)

    def energy(M, obj):
        return ENT.energy_grad(M, cfg, obj, c=1.0, need_grad=False, p=p)[0]

    worst_iso = {k: 0.0 for k in OBJECTS}
    worst_full = {k: 0.0 for k in OBJECTS}
    worst_dir = {}
    for fname, M in fields.items():
        u, gap, nt = timelike_u(M.reshape(-1, 4, 4))
        rap = float(np.arccosh(np.abs(u[:, 0])).max())
        rec = {"own_min_gap": float(gap.min()), "own_all_timelike": bool(np.all(nt == 1)), "max_rapidity_of_u": rap, "objects": {}}
        dirs = directions(rng, M.shape, n)
        G = {}; E0 = {}; info = {}
        for obj in OBJECTS:
            E0[obj], G[obj], info[obj] = ENT.energy_grad(M, cfg, obj, c=1.0, p=p)
        rec["ent_min_gap"] = info["Gam"]["min_gap"]
        rec["E"] = {k: float(E0[k]) for k in OBJECTS}
        # cache the I1 energies along every (direction, eps, t)
        cacheI1 = {}
        for di, (dn, dM) in enumerate(dirs):
            for eps in eps_list:
                for t in (eps, -eps, 2 * eps, -2 * eps):
                    cacheI1[(di, eps, t)] = energy(M + t * dM, "I1")
        for obj in OBJECTS:
            ro = {}
            for di, (dn, dM) in enumerate(dirs):
                an_full = float(np.sum(G[obj] * dM))
                an_iso = float(np.sum((G[obj] - G["I1"]) * dM))
                nf = np.sqrt(np.sum(G[obj] ** 2)); ni = np.sqrt(np.sum((G[obj] - G["I1"]) ** 2))
                err_full, err_iso = np.inf, np.inf
                for eps in eps_list:
                    ev = {t: (cacheI1[(di, eps, t)] if obj == "I1" else energy(M + t * dM, obj)) for t in (eps, -eps, 2 * eps, -2 * eps)}
                    fd_full = (8.0 * (ev[eps] - ev[-eps]) - (ev[2 * eps] - ev[-2 * eps])) / (12.0 * eps)
                    err_full = min(err_full, abs(fd_full - an_full) / max(nf, 1e-300))
                    if obj != "I1":
                        dv = {t: ev[t] - cacheI1[(di, eps, t)] for t in ev}
                        fd_iso = (8.0 * (dv[eps] - dv[-eps]) - (dv[2 * eps] - dv[-2 * eps])) / (12.0 * eps)
                        err_iso = min(err_iso, abs(fd_iso - an_iso) / max(ni, 1e-300))
                    else:
                        err_iso = err_full
                ro[dn] = {"rel_err_full": float(err_full), "rel_err_isolated": float(err_iso)}
                worst_full[obj] = max(worst_full[obj], err_full); worst_iso[obj] = max(worst_iso[obj], err_iso)
                worst_dir[dn] = max(worst_dir.get(dn, 0.0), err_iso)
            rec["objects"][obj] = {"per_direction": ro, "worst_full": max(v["rel_err_full"] for v in ro.values()),
                                   "worst_isolated": max(v["rel_err_isolated"] for v in ro.values()),
                                   "grad_norm_full": float(nf), "grad_norm_isolated": float(ni)}
        out[fname] = rec
        log(f"claim 5 field {fname}: min_gap {rec['own_min_gap']:.3f}, max rapidity {rap:.3f}, worst isolated {max(rec['objects'][k]['worst_isolated'] for k in OBJECTS):.2e}")
    out["worst_isolated_per_object"] = worst_iso
    out["worst_full_per_object"] = worst_full
    out["worst_isolated_per_direction_kind"] = worst_dir
    for obj in OBJECTS:
        line(f"A5_{obj}_gradient_exact_fd_rel_lt_1e-6_three_fields_ten_directions", worst_iso[obj] < 1e-6 and worst_full[obj] < 1e-6,
             f"worst rel (object part isolated) {worst_iso[obj]:.2e}, (full E) {worst_full[obj]:.2e}")
    sg = out["small_gap_0.5"]
    line("A5_small_gap_field_really_has_gap_near_0.5_and_stays_exact", 0.3 < sg["own_min_gap"] < 0.7 and max(sg["objects"][k]["worst_isolated"] for k in OBJECTS) < 1e-6,
         f"own min gap {sg['own_min_gap']:.3f}, entrants min gap {sg['ent_min_gap']:.3f}, worst isolated {max(sg['objects'][k]['worst_isolated'] for k in OBJECTS):.2e}")
    bf = out["boosted_rap0.5"]
    line("A5_boosted_field_has_u_far_from_e0_and_boundary_time_row_directions_exact", bf["max_rapidity_of_u"] > 0.4 and max(worst_dir[k] for k in ("boundary_cell_sym", "corner_cell_time_row", "single_cell_time_row")) < 1e-6,
         f"max rapidity of u {bf['max_rapidity_of_u']:.3f}; worst boundary / corner / time-row direction {max(worst_dir[k] for k in ('boundary_cell_sym', 'corner_cell_time_row', 'single_cell_time_row')):.2e}")
    return out


# ================= claim 6: the C12 ladder =================
def I1_density_spatial(A):
    """A (..., 3, 4, 4) spatial jets -> sum_{i<j} <F_ij, F_ij>_eta (= the static I1)."""
    d = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            F = A[..., i, :, :] @ ETA @ A[..., j, :, :] - A[..., j, :, :] @ ETA @ A[..., i, :, :]
            d = d + np.einsum("a,b,...ab,...ab->...", E, E, F, F, optimize=True)
    return d


def jets_gradient(M, h):
    return np.stack([np.gradient(M, h, axis=ax) for ax in range(3)], axis=3)


def jets_onesided(M, h, side):
    A = np.zeros(M.shape[:3] + (3,) + M.shape[3:])
    for ax in range(3):
        sl_hi = [slice(None)] * 3; sl_lo = [slice(None)] * 3
        sl_hi[ax] = slice(1, None); sl_lo[ax] = slice(0, -1)
        diff = (M[tuple(sl_hi)] - M[tuple(sl_lo)]) / h
        tgt = [slice(None)] * 3
        tgt[ax] = slice(0, -1) if side == "fwd" else slice(1, None)
        A[tuple(tgt) + (ax,)] = diff
    return A


def density_two_stencils(M, h):
    dc = I1_density_spatial(jets_gradient(M, h))
    do = 0.5 * (I1_density_spatial(jets_onesided(M, h, "fwd")) + I1_density_spatial(jets_onesided(M, h, "bwd")))
    return dc, do


def hedgehog_pair(n, L):
    cfg = B3.base_cfg(s=-1.0, g=G8, n=n, L=float(L), delta=DELTA)
    h = cfg["h"]
    X, Y, Z = B3.coords(n, h)
    rho = np.sqrt(X * X + Y * Y); r = np.sqrt(rho * rho + Z * Z)
    M3m = PAIR.seed_pair(cfg, "single", 0.0)
    M3r = PAIR._tensor_from_nhat(n, h, DELTA, PAIR._nhat_from_alpha(n, h, np.arctan2(rho, Z)), [])
    return h, B3.embed34(M3m, cfg), B3.embed34(M3r, cfg), r, rho


def audit_6():
    out = {}
    ladder = [(16, 48.0), (24, 48.0), (16, 24.0), (24, 24.0), (16, 12.0), (24, 12.0), (24, 9.0)]
    rows = []
    for n, L in ladder:
        h, Mm, Mr, r, rho = hedgehog_pair(n, L)
        core = r <= 3.0
        point = r <= 1.5                                  # the hedgehog point
        tube = core & ~point & (rho <= 1.0)               # the second director's axis line, inside the ball
        remainder = core & ~point & ~tube
        row = {"n": n, "L": L, "h": h, "n_core_cells": int(core.sum())}
        for tag, M in (("melted", Mm), ("rigid", Mr)):
            dc, do = density_two_stencils(M, h)
            row[tag] = {"core_r3_central": float(h ** 3 * dc[core].sum()), "core_r3_onesided": float(h ** 3 * do[core].sum()),
                        "point_r1p5_central": float(h ** 3 * dc[point].sum()), "axis_tube_central": float(h ** 3 * dc[tube].sum()),
                        "remainder_central": float(h ** 3 * dc[remainder].sum()), "core_max_density_central": float(dc[core].max())}
        rows.append(row)
        log(f"claim 6 h={h:.3f}: melted core {row['melted']['core_r3_central']:.4f} / {row['melted']['core_r3_onesided']:.4f}, rigid core {row['rigid']['core_r3_central']:.3f} / {row['rigid']['core_r3_onesided']:.3f}")
    out["ladder"] = rows
    hs = np.array([r_["h"] for r_ in rows])
    fits = {}
    for tag in ("melted", "rigid"):
        for st in ("central", "onesided"):
            y = np.array([r_[tag][f"core_r3_{st}"] for r_ in rows])
            fine = hs <= 1.5
            sl_pow = float(np.polyfit(np.log(1.0 / hs[fine]), np.log(np.abs(y[fine])), 1)[0])
            sl_log = float(np.polyfit(np.log(1.0 / hs[fine]), y[fine], 1)[0])
            local = [float(np.log(y[i + 1] / y[i]) / np.log(hs[i] / hs[i + 1])) for i in range(len(hs) - 1)]
            fits[f"{tag}_{st}"] = {"values": y.tolist(), "power_of_1_over_h_h_le_1.5": sl_pow, "slope_vs_log_1_over_h_h_le_1.5": sl_log,
                                   "local_exponent_per_pair": local, "max_over_min_h_le_1.5": float(y[fine].max() / y[fine].min()),
                                   "max_over_min_all": float(y.max() / y.min()), "ratio_finest_over_h1": float(y[-1] / y[3])}
    out["fits"] = fits
    # growth partition at fixed L, interior only (r <= L/2 - 1.5 keeps the face stencils out), both stencils
    part = {}
    for L in (24.0, 18.0, 12.0):
        pr = {}
        for n in (16, 24):
            h, Mm, Mr, r, rho = hedgehog_pair(n, L)
            dc, do = density_two_stencils(Mm, h)
            inside = r <= L / 2.0 - 1.5
            core = inside & (r <= 3.0); axis = inside & (rho <= 1.5) & ~core; rest = inside & ~core & ~axis
            pr[str(n)] = {"h": h}
            for st, dd in (("central", dc), ("onesided", do)):
                pr[str(n)][st] = {"interior_total": float(h ** 3 * dd[inside].sum()), "core": float(h ** 3 * dd[core].sum()),
                                  "axis_cyl": float(h ** 3 * dd[axis].sum()), "rest": float(h ** 3 * dd[rest].sum())}
        for st in ("central", "onesided"):
            dtot = pr["24"][st]["interior_total"] - pr["16"][st]["interior_total"]
            pr[f"growth_fraction_{st}"] = {k: (float((pr["24"][st][k] - pr["16"][st][k]) / dtot) if abs(dtot) > 1e-12 else None) for k in ("core", "axis_cyl", "rest")}
            pr[f"growth_fraction_{st}"]["d_total"] = float(dtot)
        part[f"L{int(L)}"] = pr
    out["growth_partition_fixed_L_interior"] = part
    fr = fits["rigid_central"]; fo = fits["rigid_onesided"]
    for st in ("central", "onesided"):
        y = np.array([r_["rigid"][f"core_r3_{st}"] for r_ in rows[3:]]); ih = 1.0 / hs[3:]
        a, b = np.polyfit(ih, y, 1)
        fits[f"rigid_{st}"]["fit_a_over_h_plus_b_h_le_1"] = {"a": float(a), "b": float(b), "b_over_a": float(b / a),
                                                             "rel_resid": float(np.abs(y - (a * ih + b)).max() / y.max())}
    rig = rows[-1]["rigid"]
    out["rigid_finest_axis_tube_share_of_core"] = rig["axis_tube_central"] / rig["core_r3_central"]
    fc, fs = fr["fit_a_over_h_plus_b_h_le_1"], fo["fit_a_over_h_plus_b_h_le_1"]
    line("A6a_rigid_null_core_ball_is_a_over_h_plus_b_with_b_negative_local_exponent_above_1_trending_to_1",
         fc["rel_resid"] < 0.03 and fs["rel_resid"] < 0.03 and fc["b"] < 0 and fs["b"] < 0 and all(e > 1.0 for e in fr["local_exponent_per_pair"][3:]) and fr["local_exponent_per_pair"][-1] < fr["local_exponent_per_pair"][3],
         f"fit y = a/h + b over h<=1: central a {fc['a']:.2f}, b/a {fc['b_over_a']:.2f}, resid {fc['rel_resid']:.3f}; one-sided a {fs['a']:.2f}, b/a {fs['b_over_a']:.2f}, resid {fs['rel_resid']:.3f}; local exponents central {[round(e, 2) for e in fr['local_exponent_per_pair'][2:]]}, one-sided {[round(e, 2) for e in fo['local_exponent_per_pair'][2:]]}; point ball r<=1.5 share of the rigid core at h=0.375 {rows[-1]['rigid']['point_r1p5_central'] / rows[-1]['rigid']['core_r3_central']:.2f}")
    fm = fits["melted_central"]; fmo = fits["melted_onesided"]
    mel = {r_["h"]: (round(r_["melted"]["point_r1p5_central"], 4), round(r_["melted"]["axis_tube_central"], 4), round(r_["melted"]["remainder_central"], 4)) for r_ in rows[2:]}
    out["melted_core_split_point_tube_remainder_central"] = {str(k): v for k, v in mel.items()}
    line("A6b_melted_core_ball_bounded_where_resolved_h_le_1.5_max_over_min_lt_2_both_stencils", fm["max_over_min_h_le_1.5"] < 2.0 and fmo["max_over_min_h_le_1.5"] < 2.0,
         f"h<=1.5 central max/min {fm['max_over_min_h_le_1.5']:.2f} (values {[round(v, 4) for v in fm['values'][2:]]}), one-sided {fmo['max_over_min_h_le_1.5']:.2f}; finest/h=1 {fm['ratio_finest_over_h1']:.3f}; coarse h=3,2 balls hold 8 cells ({[round(v, 4) for v in fm['values'][:2]]})")
    tube = [r_["melted"]["axis_tube_central"] for r_ in rows[3:]]
    line("A6b2_melted_core_growth_lives_in_the_axis_tube_not_the_point", tube[-1] > 1.5 * tube[1] and abs(rows[-1]["melted"]["point_r1p5_central"] - rows[3]["melted"]["point_r1p5_central"]) < 0.02,
         f"axis tube inside the ball at h = 1, 0.75, 0.5, 0.375: {[round(t, 4) for t in tube]}; point ball r<=1.5: {[round(r_['melted']['point_r1p5_central'], 4) for r_ in rows[3:]]}")
    fa = {L: {st: part[L][f"growth_fraction_{st}"]["axis_cyl"] for st in ("central", "onesided")} for L in part}
    out["axis_growth_fractions"] = fa
    fine_ok = all(fa["L12"][st] > 0.5 for st in ("central", "onesided"))
    coarse_ok = all(fa["L24"][st] > 0.5 for st in ("central", "onesided"))
    line("A6c_seed_growth_sits_on_the_axis_cylinder_only_for_the_finer_pairs", fine_ok and not coarse_ok,
         f"axis growth fraction (central, one-sided): L=24 h 1.5->1 {tuple(round(fa['L24'][s], 2) for s in ('central', 'onesided'))}, L=18 h 1.125->0.75 {tuple(round(fa['L18'][s], 2) for s in ('central', 'onesided'))}, L=12 h 0.75->0.5 {tuple(round(fa['L12'][s], 2) for s in ('central', 'onesided'))}")
    return out


# ================= main =================
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


def main():
    rng = np.random.default_rng(1919)
    out = {"vacuum": "M_vac = diag(8, 1, 0.3, 0)", "delta": DELTA}
    log("claim 1")
    out["claim1_time_row_map"] = audit_1(rng)
    log("claim 2")
    out["claim2_identities"] = audit_2(rng)
    log("claim 3")
    out["claim3_rank"] = audit_3(rng)
    log("claim 4")
    out["claim4_certificate"] = audit_4(rng)
    log("claim 5")
    out["claim5_gradients"] = audit_5(rng)
    log("claim 6")
    out["claim6_c12_ladder"] = audit_6()
    out["lines"] = LINES
    out["runtime_s"] = time.time() - T0
    os.makedirs(DATA, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(_jsonable(out), f, indent=1)
    npass = sum(1 for v in LINES.values() if v["pass"])
    print(f"{npass}/{len(LINES)} PASS, runtime {out['runtime_s']:.1f}s, wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
