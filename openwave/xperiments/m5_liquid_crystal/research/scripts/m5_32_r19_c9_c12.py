"""M5.32 R19-0 part 1: the C9 to C12 identities for the (Γ) entrant, the
boost sector at the connection level (the author's rev-294 § 255.4),
algebra and lattice evaluations only, no relaxation.

The proposal (report rev 294, § 255.4): "replace R^gg by the traceless
symmetric bilinear S^gg_mu nu = {Gamma^g_mu, Gamma^g_nu} - trace, keeping
the commutator for the spatial sector", with Gamma_mu the connection of
the field's eigenframe and Gamma^g its boost 3-vector (the 2021 paper's
eqs (6) to (9): R^ee = Gamma x Gamma, R^gg = Gamma^g x Gamma^g, R^eg the
EM-GEM cross curvature, all sitting in the blocks of [Gamma_mu, Gamma_nu]).

EQUATIONS FIRST
---------------
Field M(x) real symmetric 4x4, eta = diag(-1, 1, 1, 1), the vacuum
M_vac = diag(g, 1, delta, 0) (so M eta has the spectrum (-g, 1, delta, 0)),
jets A_mu = d_mu M, P_mu nu = A_mu eta A_nu, F = P - P^T (the certified
commutator), G = P + P^T, <X, Y>_eta = sum_ab eta_a eta_b X_ab Y_ab, and
every density with the full contraction I(X) = (1/2) sum_{mu nu} eta^mu
eta^nu <X_mu nu, X_mu nu>_eta.

The eigenframe. M = O D O^T with O Lorentz (O eta O^T = eta) and D =
diag(lambda_0, lambda_1, lambda_2, lambda_3); the connection
Omega_mu = O^-1 d_mu O is in so(1, 3): Omega eta antisymmetric, i.e.
Omega has a SYMMETRIC time row (the boost 3-vector a_mu = Omega_mu[0, i])
and an ANTISYMMETRIC spatial block (the rotation part). Then exactly
    d_mu M = O (Omega_mu D + D Omega_mu^T + d_mu D) O^T,
and in the frame (A~ = O^-1 A O^-T):
    A~[0, i] = a_i (lambda_i + lambda_0)          the boost part, the time row
    A~[i, j] = Omega_ij (lambda_j - lambda_i)      the rotation part, i != j
    A~[i, i] = d lambda_i                          the diagonal
so the boost content of the connection is the jet's time row in the
field's own frame and nothing else lands there (C9a). The shape factors
at the vacuum: (lambda_i + g) = (1 + g, delta + g, g), never zero, and
(lambda_j - lambda_i) = (1 - delta, 1, delta), which VANISH at a melted
core (the isotropic blend lambda_1 = lambda_2 = lambda_3): the rotation
sector is protected by the melt, the boost sector is not (C12).

Frame-free, with u the timelike unit eigenvector of M eta (u^T eta u = -1),
Pi_u = -u u^T eta, Pi_s = 1 - Pi_u:
    A^t = Pi_u A Pi_s^T + Pi_s A Pi_u^T            (the time row and column)
    P^tt_mu nu = A^t_mu eta A^t_nu
    X_a = P^tt - P^tt^T   (the boost-boost commutator, in F already)
    X_s = P^tt + P^tt^T   (the boost-boost anticommutator)
    X_s_tl = Pi_s X_s Pi_s^T - (1/3) tr(eta Pi_s X_s Pi_s^T) (eta + u u^T)
                          (the author's literal S^gg: spatial, traceless,
                           no (0, 0) corner)
    F^(Gamma)     = F - X_a + X_s        density Gam    = I(F^(Gamma))
    F^(Gamma, tl) = F - X_a + X_s_tl     density Gam_tl = I(F^(Gamma, tl))
    (B-u)         = Pi_s F Pi_s^T + Pi_u G Pi_s^T + Pi_s G Pi_u^T + Pi_u G Pi_u^T
In the frame, with v_mu = (lambda_i + lambda_0) a_mu,i:
    P^tt_mu nu = (v_mu . v_nu) e_0 e_0^T - v_mu v_nu^T
so X_a's spatial block is -(v_mu v_nu^T - v_nu v_mu^T) (the cross product
R^gg with the shape factors, the SO(1, 3) sign of the paper's eq (39)),
and X_s = 2 (v_mu . v_nu) e_0 e_0^T - (v_mu v_nu^T + v_nu v_mu^T): the
symmetric outer product plus a corner carrying the trace. Every F is
odd and every X_s even under the (mu, nu) swap (for symmetric jets the
internal transpose; the parity argument needs neither symmetric jets nor
anything about u), so
    <F - X_a, X_s>_eta = 0 identically: I(F^(Gamma)) = I(F - X_a) + I(X_s),
the EM-GEM cross term of eq (9) has no survivor under the replacement,
at the connection level and at the field level alike (C10).

CHECKS (each a number in the JSON, each falsifiable)
    C9a  the time-row map: A_mu from the lift (a Richardson finite
         difference of M(t) = O_0 e^{t Omega} (D + t dD) e^{t Omega^T} O_0^T)
         equals the exact formula; the frame-free A^t equals the boost-only
         jet O_0 (Omega^b D + D Omega^b^T) O_0^T; the spatial and corner
         parts carry the rotation part and dD; the shape factors
    C9b  (Gamma) built two ways (from Omega^b with the shape factors, and
         from the (M, A) projection) agrees to roundoff; the lift ambiguity
         (O_0 -> O_0 diag(+-1)) and a global Lorentz transform leave Gam,
         Gam_tl and (B-u) unchanged; the fixed-frame (B) is the control
    C9c  the span at the vacuum frame: ranks of {I1..I6, C6a, C6b} + GG +
         B + Gam + Gam_tl; residuals; the Coulomb sector (Gam = I1 exactly with A^t = 0; Bu = I1
         needs the corner jet u^T eta A eta u zero as well, the audit);
         the static sector split of I1 into the R^ee / R^gg / R^eg analogs
         with their signs (the negative piece is the R^eg time row, which
         (Gamma) does not touch)
    C10  eq (9)'s cross term: connection level (SO(4) +2 R^ee.R^gg, SO(1,3)
         -2 R^ee.R^gg, replacement 0) and field level with the shape
         factors (the F^rr / F^bb cross term's sign vs the SO(1, 3) form;
         the (Gamma) cross term 0 to roundoff)
    C11  the omega degree on the rigid clock A_0 = omega a0: Gam, Gam_tl,
         (B-u) quartic; the sign of the omega^4 coefficient on random jets
         and on the six Lorentz channels
    C12  the hedgehog core on the lattice, an h-ladder at L = 48 (n = 32,
         48, 64): the single hedgehog (the R3 seed, isotropic-blended core)
         undressed and b*-dressed (the M5.21.14 record profile, b*(0.15) =
         0.03), and the RIGID-director null (no blend): totals and core
         balls of I1, GG, (B-u), Gam, Gam_tl and the boost-boost pieces;
         the rigid null's core ball diverges like 1/h for every quartic
         density (the winding count), the melted undressed core ball stays
         bounded (the unrelaxed seed's TOTAL still grows with refinement:
         that growth sits on the z-axis cylinder, the second director's
         line singularity of the seed, not at the core; partitioned into
         core ball r <= 3, axis cylinder rho <= 1.5, the rest), and the
         dressed seed's boost-boost pieces grow fast under refinement: the
         record's b* profile is a saw of unit wavelength, so its radial
         derivative is unresolved at the R3 instrument's h = 1.5 and 1.0
         (the same seed R3 relaxed; reported as the instrument's caveat,
         the boost shape factor lambda_i + g does not melt)
Out: ../data/m5_32_r19_c9_c12.json
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
OUT = os.path.join(DATA, "m5_32_r19_c9_c12.json")


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
PAIR = _load("m5_21_4_a_pair", "m5_21_4_a_pair.py")
B3 = LAG.B3

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
ETA_D = np.array([-1.0, 1.0, 1.0, 1.0])
W = np.einsum("m,k->mk", ETA_D, ETA_D)
I4 = np.eye(4)
G_MAIN, DELTA = 8.0, 0.3
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def vac(g):
    return np.diag([g, 1.0, DELTA, 0.0])


# ---------------- generators and lifts ----------------
def so13(rng, scale=1.0, boost=True, rot=True):
    """symmetric time row (boost 3-vector) + antisymmetric spatial block."""
    G = np.zeros((4, 4))
    if boost:
        a = scale * rng.normal(size=3); G[0, 1:] = a; G[1:, 0] = a
    if rot:
        r = scale * rng.normal(size=3)
        G[1, 2], G[1, 3], G[2, 3] = -r[2], r[1], -r[0]
        G[2, 1], G[3, 1], G[3, 2] = r[2], -r[1], r[0]
    return G


def boost_part(G):
    B = np.zeros((4, 4)); B[0, 1:] = G[0, 1:]; B[1:, 0] = G[1:, 0]; return B


def rot_part(G):
    R = G.copy(); R[0, :] = 0.0; R[:, 0] = 0.0; return R


def jet_of(Om, D, dD):
    return Om @ D + D @ Om.T + dD


def M_lift(O0, Om, D, dD, t):
    E = expm(t * Om)
    return O0 @ E @ (D + t * dD) @ E.T @ O0.T


def fd_jet(O0, Om, D, dD, h=1e-3):
    f = lambda t: M_lift(O0, Om, D, dD, t)  # noqa: E731
    return (8.0 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12.0 * h)


# ---------------- frame-free objects ----------------
def projectors(u):
    Pu = -np.einsum("na,nb,b->nab", u, u, ETA_D)
    return Pu, I4[None] - Pu


def sw(L, X, R):
    """L X R^T on the internal indices, X (4,4,N,4,4)."""
    return np.einsum("nab,mknbc,ndc->mknad", L, X, R, optimize=True)


def sw1(L, A, R):
    return np.einsum("nab,mnbc,ndc->mnad", L, A, R, optimize=True)


def P_of(A):
    return np.einsum("mnab,bc,kncd->mknad", A, ETA, A, optimize=True)


def ip(X, Y):
    return np.einsum("a,b,mknab,mknab->mkn", ETA_D, ETA_D, X, Y, optimize=True)


def Ifull(X):
    return 0.5 * np.einsum("mk,mkn->n", W, ip(X, X))


def cross(X, Y):
    return np.einsum("mk,mkn->n", W, ip(X, Y))


def objects(A, u, pieces=False):
    """A (4, N, 4, 4) symmetric jets, u (N, 4) timelike unit eigenvectors."""
    Pu, Ps = projectors(u)
    At = sw1(Pu, A, Ps) + sw1(Ps, A, Pu)
    P = P_of(A); F = P - P.swapaxes(0, 1); G = P + P.swapaxes(0, 1)
    Ptt = P_of(At); Xa = Ptt - Ptt.swapaxes(0, 1); Xs = Ptt + Ptt.swapaxes(0, 1)
    XsS = sw(Ps, Xs, Ps)
    trS = np.einsum("a,mknaa->mkn", ETA_D, XsS)
    gs = ETA[None] + np.einsum("na,nb->nab", u, u)
    Xs_tl = XsS - (trS / 3.0)[..., None, None] * gs[None, None]
    FG = F - Xa + Xs; FGtl = F - Xa + Xs_tl
    Bu = sw(Ps, F, Ps) + sw(Pu, G, Ps) + sw(Ps, G, Pu) + sw(Pu, G, Pu)
    out = {"I1": Ifull(F), "GG": Ifull(G), "Bu": Ifull(Bu), "Gam": Ifull(FG),
           "Gam_tl": Ifull(FGtl), "I_FminusXa": Ifull(F - Xa), "I_Xs": Ifull(Xs),
           "I_Xs_tl": Ifull(Xs_tl), "I_Xa": Ifull(Xa),
           "cross_FminusXa_Xs": cross(F - Xa, Xs), "cross_FminusXa_Xs_tl": cross(F - Xa, Xs_tl)}
    Ft = sw(Pu, F, Ps) + sw(Ps, F, Pu)
    out["I_Ft_timerow"] = Ifull(Ft)
    if pieces:
        return out, {"F": F, "G": G, "Xa": Xa, "Xs": Xs, "Xs_tl": Xs_tl, "At": At, "Bu": Bu, "P": P}
    return out


def u_of(M):
    u0, V, lam, sig, k0, ok, gap = RB.tl_eig(M)
    return u0, ok, gap


def frame_B(A):
    """the fixed-frame (B) of the pre-check (P_t = e_0 e_0^T), on jets A (4,N,4,4)."""
    PT = np.zeros((4, 4)); PT[0, 0] = 1.0; PS = I4 - PT
    P = P_of(A); F = P - P.swapaxes(0, 1); G = P + P.swapaxes(0, 1)
    B = PS @ F @ PS + PT @ G @ PS + PS @ G @ PT + PT @ G @ PT
    return Ifull(B)


def lorentz_transform(L, A):
    """M -> L M L^T on the internal indices, d_mu -> L^-T on the derivative index."""
    return np.einsum("mk,k...ab,ca,db->m...cd", np.linalg.inv(L).T, A, L, L)


def registry_densities(A):
    Fa = LAG.F_of_A(A)
    d = {k: LAG.density_from_K(Fa, LAG.REGISTRY[k]._K()) for k in ["I1", "I2", "I3", "I4", "I5", "I6"]}
    AE = np.einsum("mnab,bc->mnac", A, ETA); tr2 = np.einsum("mnab,knba->mkn", AE, AE)
    d["C6a"] = np.einsum("m,mmn->n", ETA_D, tr2) ** 2
    d["C6b"] = np.einsum("mk,mkn->n", W, tr2 ** 2)
    return d


def in_span(t, cols, names):
    X = np.stack(cols, 1)
    c = np.linalg.lstsq(X, t, rcond=None)[0]
    r = t - X @ c
    return {"rel_resid": float(np.linalg.norm(r) / np.linalg.norm(t)),
            "coef": {n: float(v) for n, v in zip(names, c)}}


def rank_of(cols, scale):
    return int(np.linalg.matrix_rank(np.stack(cols, 1), tol=1e-9 * scale))


def sym_jets(rng, n):
    X = rng.normal(size=(4, n, 4, 4))
    return X + X.swapaxes(-1, -2)


# ================= C9 =================
def c9(rng, g, n_trials=40, n_jets=400):
    D0 = vac(g)
    out = {"g": g}
    # ---- C9a: the time-row map on random lifts
    worst = {"fd_vs_exact": 0.0, "At_vs_boost_jet": 0.0, "Ass_vs_rot_jet": 0.0,
             "corner_vs_dD0": 0.0, "u_vs_O0e0": 0.0}
    for _ in range(n_trials):
        O0 = expm(so13(rng, 0.5))
        D = D0 + np.diag(0.1 * rng.normal(size=4)) * np.diag([1, 1, 1, 0])
        Oms = [so13(rng, 1.0) for _ in range(4)]
        dDs = [np.diag(0.3 * rng.normal(size=4)) for _ in range(4)]
        M0 = O0 @ D @ O0.T
        A_ex = np.stack([O0 @ jet_of(Oms[m], D, dDs[m]) @ O0.T for m in range(4)])
        A_fd = np.stack([fd_jet(O0, Oms[m], D, dDs[m]) for m in range(4)])
        worst["fd_vs_exact"] = max(worst["fd_vs_exact"], rel(A_fd, A_ex))
        u, ok, gap = u_of(M0[None]); u = u[0]
        assert ok[0]
        worst["u_vs_O0e0"] = max(worst["u_vs_O0e0"], min(rel(u, O0[:, 0]), rel(u, -O0[:, 0])))
        Pu, Ps = projectors(u[None])
        Af = A_fd[:, None]
        At = (sw1(Pu, Af, Ps) + sw1(Ps, Af, Pu))[:, 0]
        Ass = sw1(Ps, Af, Ps)[:, 0]
        Ac = sw1(Pu, Af, Pu)[:, 0]
        At_b = np.stack([O0 @ jet_of(boost_part(Oms[m]), D, 0 * D) @ O0.T for m in range(4)])
        dDs_s = [dd * np.diag([0, 1, 1, 1]) for dd in dDs]
        Ass_r = np.stack([O0 @ jet_of(rot_part(Oms[m]), D, dDs_s[m]) @ O0.T for m in range(4)])
        Ac_d = np.stack([O0 @ (dDs[m] * np.diag([1, 0, 0, 0])) @ O0.T for m in range(4)])
        worst["At_vs_boost_jet"] = max(worst["At_vs_boost_jet"], rel(At, At_b))
        worst["Ass_vs_rot_jet"] = max(worst["Ass_vs_rot_jet"], rel(Ass, Ass_r))
        worst["corner_vs_dD0"] = max(worst["corner_vs_dD0"], rel(Ac, Ac_d))
    # the shape factors at the vacuum, frame O_0 = 1
    Om = so13(rng, 1.0); Aj = jet_of(Om, D0, 0 * D0)
    lam = np.diag(D0)
    sf_t = [float(Aj[0, i] / Om[0, i]) for i in (1, 2, 3)]
    sf_r = {f"{i}{j}": float(Aj[i, j] / Om[i, j]) for i, j in ((1, 2), (1, 3), (2, 3))}
    out["C9a"] = {"worst_rel": worst,
                  "shape_factor_time_row": {"measured": sf_t, "lambda_i+lambda_0": [float(lam[i] + lam[0]) for i in (1, 2, 3)]},
                  "shape_factor_spatial": {"measured": sf_r, "lambda_j-lambda_i": {f"{i}{j}": float(lam[j] - lam[i]) for i, j in ((1, 2), (1, 3), (2, 3))}},
                  "note": "Omega = O^-1 dO is the so(1,3) element (Omega eta antisymmetric: symmetric time row, antisymmetric spatial block); the paper's O^T dO equals it for SO(4) and equals eta Omega eta up to the sign of the time row for a Lorentz O; every density here is even in the boost vector"}
    # ---- C9b: two ways, lift ambiguity, covariance
    worst = {"Xs_two_ways": 0.0, "Gam_two_ways": 0.0, "Gam_tl_two_ways": 0.0, "Bu_vs_frameB": 0.0,
             "lift_sign_flip_Gam": 0.0, "lift_sign_flip_Gam_tl": 0.0,
             "cov_rot_Gam": 0.0, "cov_boost_Gam": 0.0, "cov_boost_Gam_tl": 0.0, "cov_boost_Bu": 0.0,
             "cov_boost_I1": 0.0, "cov_boost_GG": 0.0}
    ctrl_B = 0.0
    for _ in range(n_trials):
        O0 = expm(so13(rng, 0.5))
        D = D0 + np.diag(0.1 * rng.normal(size=4)) * np.diag([1, 1, 1, 0])
        lam = np.diag(D)
        Oms = [so13(rng, 1.0) for _ in range(4)]
        dDs = [np.diag(0.3 * rng.normal(size=4)) for _ in range(4)]
        M0 = O0 @ D @ O0.T
        A = np.stack([O0 @ jet_of(Oms[m], D, dDs[m]) @ O0.T for m in range(4)])
        u, ok, _ = u_of(M0[None])
        ob, pc = objects(A[:, None], u, pieces=True)
        # way 1: from the boost vectors with the shape factors, in the frame
        s = lam[1:] + lam[0]
        v = np.stack([s * Oms[m][0, 1:] for m in range(4)])       # (4, 3)
        Xs1 = np.zeros((4, 4, 4, 4))
        for m in range(4):
            for k in range(4):
                vv = v[m] @ v[k]
                Xs1[m, k, 0, 0] = 2.0 * vv
                Xs1[m, k, 1:, 1:] = -(np.outer(v[m], v[k]) + np.outer(v[k], v[m]))
        Xs1 = np.einsum("ab,mkbc,dc->mkad", O0, Xs1, O0)
        worst["Xs_two_ways"] = max(worst["Xs_two_ways"], rel(Xs1, pc["Xs"][:, :, 0]))
        Gam1 = Ifull((pc["F"] - pc["Xa"] + Xs1[:, :, None]))
        worst["Gam_two_ways"] = max(worst["Gam_two_ways"], rel(Gam1, ob["Gam"]))
        # traceless variant two ways
        Ps1 = O0 @ np.diag([0.0, 1, 1, 1]) @ np.linalg.inv(O0)
        XsS1 = np.einsum("ab,mkbc,dc->mkad", Ps1, Xs1, Ps1)
        tr1 = np.einsum("a,mkaa->mk", ETA_D, XsS1)
        gs1 = ETA + np.outer(u[0], u[0])
        Xs_tl1 = XsS1 - (tr1 / 3.0)[..., None, None] * gs1
        Gtl1 = Ifull(pc["F"] - pc["Xa"] + Xs_tl1[:, :, None])
        worst["Gam_tl_two_ways"] = max(worst["Gam_tl_two_ways"], rel(Gtl1, ob["Gam_tl"]))
        # (B-u) equals the fixed-frame (B) on the frame jets
        Oi = np.linalg.inv(O0)
        Af = np.einsum("ab,mbc,dc->mad", Oi, A, Oi)
        worst["Bu_vs_frameB"] = max(worst["Bu_vs_frameB"], rel(frame_B(Af[:, None]), ob["Bu"]))
        # lift ambiguity: O_0 -> O_0 S, S = diag(1, -1, 1, -1), Omega -> S Omega S
        S = np.diag([1.0, -1.0, 1.0, -1.0])
        A2 = np.stack([(O0 @ S) @ jet_of(S @ Oms[m] @ S, D, dDs[m]) @ (O0 @ S).T for m in range(4)])
        assert rel(A2, A) < 1e-12
        ob2 = objects(A2[:, None], u)
        worst["lift_sign_flip_Gam"] = max(worst["lift_sign_flip_Gam"], rel(ob2["Gam"], ob["Gam"]))
        worst["lift_sign_flip_Gam_tl"] = max(worst["lift_sign_flip_Gam_tl"], rel(ob2["Gam_tl"], ob["Gam_tl"]))
        # covariance under a global Lorentz transform (internal and derivative indices)
        for kind, L in (("rot", expm(so13(rng, 0.7, boost=False))), ("boost", expm(so13(rng, 0.3, rot=False)))):
            AL = lorentz_transform(L, A)
            ML = L @ M0 @ L.T
            uL, okL, _ = u_of(ML[None])
            assert okL[0]
            obL = objects(AL[:, None], uL)
            worst[f"cov_{kind}_Gam"] = max(worst[f"cov_{kind}_Gam"], rel(obL["Gam"], ob["Gam"]))
            if kind == "boost":
                for key in ("Gam_tl", "Bu", "I1", "GG"):
                    worst[f"cov_boost_{key}"] = max(worst[f"cov_boost_{key}"], rel(obL[key], ob[key]))
                ctrl_B = max(ctrl_B, rel(frame_B(AL[:, None]), frame_B(A[:, None])))
    out["C9b"] = {"worst_rel": worst, "control_fixed_frame_B_under_boost_rel": ctrl_B}
    # ---- C9c: the span at the vacuum frame (u = e_0)
    A = sym_jets(rng, n_jets)
    u = np.tile(np.array([1.0, 0, 0, 0]), (n_jets, 1))
    ob = objects(A, u)
    d = registry_densities(A)
    names = list(d); cols = [d[k] for k in names]
    scale = np.abs(ob["Gam"]).max()
    r8 = rank_of(cols, scale)
    ranks = {"basis8": r8,
             "+GG": rank_of(cols + [ob["GG"]], scale),
             "+GG+B": rank_of(cols + [ob["GG"], ob["Bu"]], scale),
             "+GG+B+Gam": rank_of(cols + [ob["GG"], ob["Bu"], ob["Gam"]], scale),
             "+GG+B+Gam+Gam_tl": rank_of(cols + [ob["GG"], ob["Bu"], ob["Gam"], ob["Gam_tl"]], scale)}
    span = {"Gam_in_basis8+GG+B": in_span(ob["Gam"], cols + [ob["GG"], ob["Bu"]], names + ["GG", "B"]),
            "Gam_tl_in_basis8+GG+B+Gam": in_span(ob["Gam_tl"], cols + [ob["GG"], ob["Bu"], ob["Gam"]], names + ["GG", "B", "Gam"]),
            "Gam_minus_I1_in_{I_Xa,I_Xs}": in_span(ob["Gam"] - ob["I1"], [ob["I_Xa"], ob["I_Xs"], ob["cross_FminusXa_Xs"] * 0 + cross_FX(A, u)], ["I_Xa", "I_Xs", "cross_F_Xa"])}
    # sectors
    As = A.copy(); As[0] = 0.0
    obs = objects(As, u)
    Ac = As.copy(); Ac[:, :, 0, :] = 0.0; Ac[:, :, :, 0] = 0.0
    obc = objects(Ac, u)
    sect = {"coulomb_Gam_minus_I1_max": float(np.abs(obc["Gam"] - obc["I1"]).max()),
            "coulomb_Gam_tl_minus_I1_max": float(np.abs(obc["Gam_tl"] - obc["I1"]).max()),
            "coulomb_Bu_minus_I1_max": float(np.abs(obc["Bu"] - obc["I1"]).max()),
            "coulomb_I1_scale": float(np.abs(obc["I1"]).max()),
            "static_I_Xa_min": float(obs["I_Xa"].min()), "static_I_Xs_min": float(obs["I_Xs"].min()),
            "static_I_Xs_tl_min": float(obs["I_Xs_tl"].min()),
            "static_I_Ft_timerow_max": float(obs["I_Ft_timerow"].max()),
            "static_Gam_minus_I1_range": [float((obs["Gam"] - obs["I1"]).min()), float((obs["Gam"] - obs["I1"]).max())],
            "static_frac_Gam_negative": float(np.mean(obs["Gam"] < 0)),
            "static_frac_Gam_tl_negative": float(np.mean(obs["Gam_tl"] < 0)),
            "static_frac_I1_negative": float(np.mean(obs["I1"] < 0)),
            "static_frac_Bu_negative": float(np.mean(obs["Bu"] < 0)),
            "static_unit_sphere_inf": unit_sphere_inf(obs, As)}
    decomp = {"max_abs_Gam_minus_(I_FminusXa+I_Xs)": float(np.abs(ob["Gam"] - ob["I_FminusXa"] - ob["I_Xs"]).max()),
              "max_abs_cross_FminusXa_Xs": float(np.abs(ob["cross_FminusXa_Xs"]).max()),
              "max_abs_cross_FminusXa_Xs_tl": float(np.abs(ob["cross_FminusXa_Xs_tl"]).max()),
              "Gam_scale": float(scale)}
    out["C9c"] = {"n_jets": n_jets, "ranks": ranks, "span": span, "sectors": sect, "decomposition": decomp}
    return out


def cross_FX(A, u):
    _, pc = objects(A, u, pieces=True)
    return cross(pc["F"], pc["Xa"])


def unit_sphere_inf(obs, As):
    nrm = np.sqrt(np.einsum("mnab,mnab->n", As, As))
    return {k: float((obs[k] / nrm ** 4).min()) for k in ("I1", "GG", "Bu", "Gam", "Gam_tl")}


# ================= C10 =================
def cross_mat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def c10(rng, g, n_samp=400):
    D0 = vac(g)
    res = {"g": g, "n_samp": n_samp}
    agree_m, agree_p, conn_so4, conn_so13, conn_rep, fld_cross, gam_cross = [], [], [], [], [], [], []
    pieces = {"I_Frr": [], "I_Fbb": [], "I_Frb": [], "cross_rr_bb": [], "cross_rr_rb": [], "cross_bb_rb": []}
    for _ in range(n_samp):
        om = rng.normal(size=(4, 3)); a = rng.normal(size=(4, 3))
        so4 = so13v = repl = 0.0
        for m in range(4):
            for k in range(4):
                Ree = np.cross(om[m], om[k]); Rgg = np.cross(a[m], a[k])
                Sgg = np.outer(a[m], a[k]) + np.outer(a[k], a[m]) - (2.0 / 3.0) * (a[m] @ a[k]) * np.eye(3)
                w = W[m, k]
                so4 += w * 2.0 * (Ree @ Rgg); so13v += -w * 2.0 * (Ree @ Rgg)
                repl += w * 2.0 * np.sum(cross_mat(Ree) * Sgg)
        conn_so4.append(so4); conn_so13.append(so13v); conn_rep.append(repl)
        # field level at the vacuum frame, dD = 0
        Gr = [np.zeros((4, 4)) for _ in range(4)]; Gb = [np.zeros((4, 4)) for _ in range(4)]
        for m in range(4):
            Gb[m][0, 1:] = a[m]; Gb[m][1:, 0] = a[m]
            r = om[m]
            Gr[m][1, 2], Gr[m][1, 3], Gr[m][2, 3] = -r[2], r[1], -r[0]
            Gr[m][2, 1], Gr[m][3, 1], Gr[m][3, 2] = r[2], -r[1], r[0]
        Ar = np.stack([jet_of(Gr[m], D0, 0 * D0) for m in range(4)])[:, None]
        Ab = np.stack([jet_of(Gb[m], D0, 0 * D0) for m in range(4)])[:, None]
        Pr, Pb = P_of(Ar), P_of(Ab)
        Prb = np.einsum("mnab,bc,kncd->mknad", Ar, ETA, Ab) + np.einsum("mnab,bc,kncd->mknad", Ab, ETA, Ar)
        Frr = Pr - Pr.swapaxes(0, 1); Fbb = Pb - Pb.swapaxes(0, 1); Frb = Prb - Prb.swapaxes(0, 1)
        Ftot = P_of(Ar + Ab); Ftot = Ftot - Ftot.swapaxes(0, 1)
        assert rel(Ftot, Frr + Fbb + Frb) < 1e-12
        c_rr_bb = float(2.0 * cross(Frr, Fbb)[0])
        fld_cross.append(c_rr_bb)
        agree_m.append(np.sign(c_rr_bb) == np.sign(so13v)); agree_p.append(np.sign(c_rr_bb) == np.sign(so4))
        for key, val in (("I_Frr", Ifull(Frr)), ("I_Fbb", Ifull(Fbb)), ("I_Frb", Ifull(Frb)),
                         ("cross_rr_bb", 2 * cross(Frr, Fbb)), ("cross_rr_rb", 2 * cross(Frr, Frb)), ("cross_bb_rb", 2 * cross(Fbb, Frb))):
            pieces[key].append(float(val[0]))
        u = np.array([[1.0, 0, 0, 0]])
        ob = objects(Ar + Ab, u)
        gam_cross.append(float(ob["cross_FminusXa_Xs"][0]))
    res["connection_level"] = {"SO4_cross_abs_mean": float(np.mean(np.abs(conn_so4))),
                               "SO13_cross_abs_mean": float(np.mean(np.abs(conn_so13))),
                               "replacement_cross_max_abs": float(np.max(np.abs(conn_rep))),
                               "SO4_vs_SO13_sign_opposite_frac": float(np.mean(np.sign(conn_so4) == -np.sign(conn_so13)))}
    res["field_level"] = {"Frr_Fbb_cross_abs_mean": float(np.mean(np.abs(fld_cross))),
                          "sign_agrees_with_SO13_form_frac": float(np.mean(agree_m)),
                          "sign_agrees_with_SO4_form_frac": float(np.mean(agree_p)),
                          "Gamma_cross_FminusXa_Xs_max_abs": float(np.max(np.abs(gam_cross))),
                          "pieces_range": {k: [float(np.min(v)), float(np.max(v))] for k, v in pieces.items()}}
    # the same on STATIC jets only (Omega_0 = 0): the signs of the three sectors
    st = {"I_Frr": [], "I_Fbb": [], "I_Frb": []}
    for _ in range(n_samp):
        om = rng.normal(size=(4, 3)); a = rng.normal(size=(4, 3)); om[0] = 0; a[0] = 0
        Gr = [np.zeros((4, 4)) for _ in range(4)]; Gb = [np.zeros((4, 4)) for _ in range(4)]
        for m in range(4):
            Gb[m][0, 1:] = a[m]; Gb[m][1:, 0] = a[m]
            r = om[m]
            Gr[m][1, 2], Gr[m][1, 3], Gr[m][2, 3] = -r[2], r[1], -r[0]
            Gr[m][2, 1], Gr[m][3, 1], Gr[m][3, 2] = r[2], -r[1], r[0]
        Ar = np.stack([jet_of(Gr[m], D0, 0 * D0) for m in range(4)])[:, None]
        Ab = np.stack([jet_of(Gb[m], D0, 0 * D0) for m in range(4)])[:, None]
        Pr, Pb = P_of(Ar), P_of(Ab)
        Prb = np.einsum("mnab,bc,kncd->mknad", Ar, ETA, Ab) + np.einsum("mnab,bc,kncd->mknad", Ab, ETA, Ar)
        st["I_Frr"].append(float(Ifull(Pr - Pr.swapaxes(0, 1))[0]))
        st["I_Fbb"].append(float(Ifull(Pb - Pb.swapaxes(0, 1))[0]))
        st["I_Frb"].append(float(Ifull(Prb - Prb.swapaxes(0, 1))[0]))
    res["static_sector_signs"] = {k: [float(np.min(v)), float(np.max(v))] for k, v in st.items()}
    return res


# ================= C11 =================
def c11(rng, g, n_jets=300):
    D0 = vac(g)
    A = sym_jets(rng, n_jets); a0 = sym_jets(rng, n_jets)[0]
    u = np.tile(np.array([1.0, 0, 0, 0]), (n_jets, 1))
    ws = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]); V = np.vander(ws, 5, increasing=True)
    keys = ["I1", "GG", "Bu", "Gam", "Gam_tl"]
    vals = {k: [] for k in keys}
    for wv in ws:
        Aw = A.copy(); Aw[0] = wv * a0
        ob = objects(Aw, u)
        for k in keys:
            vals[k].append(ob[k])
    out = {"g": g, "n_jets": n_jets}
    for k in keys:
        coef = np.linalg.solve(V, np.stack(vals[k], 1).T)
        out[k] = {f"c{i}_max": float(np.abs(coef[i]).max()) for i in range(5)}
        out[k]["frac_c4_negative"] = float(np.mean(coef[4] < -1e-12 * np.abs(coef[4]).max()))
        out[k]["frac_c2_negative"] = float(np.mean(coef[2] < -1e-12 * np.abs(coef[2]).max()))
    # the six Lorentz channels at the vacuum with random static jets
    ch = {}
    for name, G in [("boost_1", None), ("boost_2", None), ("boost_3", None), ("rot_1", None), ("rot_2", None), ("rot_3", None)]:
        Gm = np.zeros((4, 4))
        kk = int(name[-1])
        if name.startswith("boost"):
            Gm[0, kk] = Gm[kk, 0] = 1.0
        else:
            i, j = [x for x in (1, 2, 3) if x != kk]
            s = 1.0 if (kk, i, j) in [(1, 2, 3), (2, 3, 1), (3, 1, 2)] else -1.0
            Gm[i, j] = -s; Gm[j, i] = s
        a0c = np.tile(Gm @ D0 + D0 @ Gm.T, (n_jets, 1, 1))
        As = A.copy(); As[0] = 0.0
        vv = {k: [] for k in keys}
        for wv in ws:
            Aw = As.copy(); Aw[0] = wv * a0c
            ob = objects(Aw, u)
            for k in keys:
                vv[k].append(ob[k])
        ch[name] = {}
        for k in keys:
            coef = np.linalg.solve(V, np.stack(vv[k], 1).T)
            ch[name][k] = {"c2_range": [float(coef[2].min()), float(coef[2].max())],
                           "c4_range": [float(coef[4].min()), float(coef[4].max())]}
    out["channels"] = ch
    return out


# ================= C12 =================
def dressed(M4, cfg, scale=1.0):
    rs, bstar = RB.bstar_record()
    R, K, K2 = RB.boost_geom(cfg)
    bl = scale * np.interp(R.ravel(), rs, bstar).reshape(R.shape)
    Md, _ = RB.dress(M4, None, bl, K, K2)
    return Md


def rigid_single(cfg):
    n, h, delta = cfg["n"], cfg["h"], cfg["delta"]
    X, Y, Z = B3.coords(n, h)
    rho = np.sqrt(X * X + Y * Y)
    return PAIR._tensor_from_nhat(n, h, delta, PAIR._nhat_from_alpha(n, h, np.arctan2(rho, Z)), [])


def lattice_totals(M, cfg, p, chunk=32768):
    """h^3 sum_br wt sum_cells density, for every object; plus core balls."""
    n = cfg["n"]; h3 = cfg["h"] ** 3
    X, Y, Z = B3.coords(n, cfg["h"])
    R = np.sqrt(X * X + Y * Y + Z * Z).ravel()
    RHO = np.sqrt(X * X + Y * Y).ravel()
    u0, ok, gap = u_of(M)
    uf = u0.reshape(-1, 4)
    keys = ["I1", "GG", "Bu", "Gam", "Gam_tl", "I_Xa", "I_Xs", "I_Xs_tl", "I_Ft_timerow", "I_FminusXa"]
    tot = {k: 0.0 for k in keys}; core3 = {k: 0.0 for k in keys}; core6 = {k: 0.0 for k in keys}
    axis = {k: 0.0 for k in keys}; offax = {k: 0.0 for k in keys}
    dens_max = {k: 0.0 for k in keys}; core_max = {k: 0.0 for k in keys}
    shells = np.array([1.5, 3.0, 4.5, 6.0, 9.0, 12.0])
    shell_sum = {k: np.zeros(len(shells)) for k in keys}
    for A, wt in LAG.lattice_jets(M, cfg):
        Af = A.reshape(4, -1, 4, 4)
        N = Af.shape[1]
        for s in range(0, N, chunk):
            ob = objects(Af[:, s:s + chunk], uf[s:s + chunk])
            r = R[s:s + chunk]; rho = RHO[s:s + chunk]
            in_core = r <= 3.0; on_axis = (rho <= 1.5) & ~in_core
            for k in keys:
                dk = ob[k]
                tot[k] += wt * float(dk.sum())
                core3[k] += wt * float(dk[in_core].sum())
                core6[k] += wt * float(dk[r <= 6.0].sum())
                axis[k] += wt * float(dk[on_axis].sum())
                offax[k] += wt * float(dk[~in_core & ~on_axis].sum())
                dens_max[k] = max(dens_max[k], wt * float(np.abs(dk).max()))
                if np.any(in_core):
                    core_max[k] = max(core_max[k], wt * float(np.abs(dk[in_core]).max()))
                lo = 0.0
                for si, hi in enumerate(shells):
                    shell_sum[k][si] += wt * float(dk[(r > lo) & (r <= hi)].sum()); lo = hi
    ref = LAG.term_lagrangian(LAG.REGISTRY["I1"], M, cfg, p)
    out = {"n": n, "h": cfg["h"], "min_gap": float(gap.min()), "n_bad_cells": int(np.sum(~ok)),
           "registry_I1_check_rel": abs(h3 * tot["I1"] - ref) / max(abs(ref), 1e-300),
           "total": {k: h3 * v for k, v in tot.items()},
           "core_r3": {k: h3 * v for k, v in core3.items()},
           "core_r6": {k: h3 * v for k, v in core6.items()},
           "axis_cyl_rho1p5_excl_core": {k: h3 * v for k, v in axis.items()},
           "off_axis_excl_core": {k: h3 * v for k, v in offax.items()},
           "core_r3_max_density": core_max,
           "shells": {k: (h3 * v).tolist() for k, v in shell_sum.items()}, "shell_edges": shells.tolist(),
           "max_density": dens_max}
    return out


def c12(g, ns=(32, 48, 64), L=48.0):
    p = LAG.default_params(s=-1.0, g=g)
    out = {"g": g, "L": L, "ns": list(ns), "bstar_at_r": {}}
    rs, bstar = RB.bstar_record()
    for r in (0.15, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0):
        out["bstar_at_r"][str(r)] = float(np.interp(r, rs, bstar))
    a_melt = (1.0 + DELTA) / 3.0
    out["shape_factors"] = {"vacuum_time_row": [1 + g, DELTA + g, g], "vacuum_spatial": [1 - DELTA, 1.0, DELTA],
                            "melted_core_time_row": [a_melt + g] * 3, "melted_core_spatial": [0.0, 0.0, 0.0]}
    fields = {}
    for n in ns:
        cfg = B3.base_cfg(s=-1.0, g=g, n=n, L=L, delta=DELTA)
        M3m = PAIR.seed_pair(cfg, "single", 0.0)
        M3r = rigid_single(cfg)
        M4m = B3.embed34(M3m, cfg); M4r = B3.embed34(M3r, cfg)
        for name, M in (("melted_undressed", M4m), ("melted_dressed", dressed(M4m, cfg)),
                        ("rigid_undressed", M4r), ("rigid_dressed", dressed(M4r, cfg))):
            log(f"C12 {name} n={n}")
            fields.setdefault(name, {})[str(n)] = lattice_totals(M, cfg, p)
    out["fields"] = fields
    # ladders: E(h) ratios
    lad = {}
    for name, byn in fields.items():
        lad[name] = {}
        keys = byn[str(ns[0])]["total"].keys()
        for k in keys:
            lad[name][k] = {"total": [byn[str(n)]["total"][k] for n in ns],
                            "core_r3": [byn[str(n)]["core_r3"][k] for n in ns],
                            "axis_cyl": [byn[str(n)]["axis_cyl_rho1p5_excl_core"][k] for n in ns],
                            "off_axis": [byn[str(n)]["off_axis_excl_core"][k] for n in ns],
                            "core_r3_max_density": [byn[str(n)]["core_r3_max_density"][k] for n in ns],
                            "ratio_last_first_total": byn[str(ns[-1])]["total"][k] / byn[str(ns[0])]["total"][k] if abs(byn[str(ns[0])]["total"][k]) > 1e-300 else None,
                            "ratio_last_first_core_r3": byn[str(ns[-1])]["core_r3"][k] / byn[str(ns[0])]["core_r3"][k] if abs(byn[str(ns[0])]["core_r3"][k]) > 1e-300 else None}
    out["ladders"] = lad
    return out


# ================= main =================
def main():
    rng = np.random.default_rng(19)
    out = {"vacuum": "M_vac = diag(g, 1, delta, 0), delta = 0.3; M eta spectrum (-g, 1, delta, 0)", "delta": DELTA}
    log("C9 at g = 8 and 32")
    out["C9"] = {"g8": c9(rng, 8.0), "g32": c9(rng, 32.0)}
    log("C10")
    out["C10"] = {"g8": c10(rng, 8.0), "g32": c10(rng, 32.0)}
    log("C11")
    out["C11"] = {"g8": c11(rng, 8.0)}
    log("C12")
    out["C12"] = c12(8.0)
    out["runtime_s"] = time.time() - T0
    c9g = out["C9"]["g8"]; c10g = out["C10"]["g8"]; c11g = out["C11"]["g8"]; c12g = out["C12"]
    v = {}
    v["C9a_timerow_map_exact"] = max(c9g["C9a"]["worst_rel"][k] for k in ("At_vs_boost_jet", "Ass_vs_rot_jet", "corner_vs_dD0")) < 1e-9 and c9g["C9a"]["worst_rel"]["fd_vs_exact"] < 1e-7
    v["C9a_shape_factors"] = rel(np.array(c9g["C9a"]["shape_factor_time_row"]["measured"]), np.array(c9g["C9a"]["shape_factor_time_row"]["lambda_i+lambda_0"])) < 1e-12
    v["C9b_Gamma_two_ways"] = max(c9g["C9b"]["worst_rel"][k] for k in ("Xs_two_ways", "Gam_two_ways", "Gam_tl_two_ways", "Bu_vs_frameB")) < 1e-10
    v["C9b_covariant_and_lift_independent"] = max(c9g["C9b"]["worst_rel"][k] for k in ("lift_sign_flip_Gam", "lift_sign_flip_Gam_tl", "cov_rot_Gam", "cov_boost_Gam", "cov_boost_Gam_tl", "cov_boost_Bu")) < 1e-9 and c9g["C9b"]["control_fixed_frame_B_under_boost_rel"] > 1e-3
    rk = c9g["C9c"]["ranks"]
    v["C9c_Gamma_new_direction"] = rk["+GG+B+Gam"] == rk["+GG+B"] + 1 and c9g["C9c"]["span"]["Gam_in_basis8+GG+B"]["rel_resid"] > 0.05
    v["C9c_Gamma_tl_new_direction"] = rk["+GG+B+Gam+Gam_tl"] == rk["+GG+B+Gam"] + 1
    v["C9c_coulomb_sector_identical"] = c9g["C9c"]["sectors"]["coulomb_Gam_minus_I1_max"] == 0.0 and c9g["C9c"]["sectors"]["coulomb_Gam_tl_minus_I1_max"] == 0.0
    v["C9c_static_boost_boost_pieces_nonnegative"] = c9g["C9c"]["sectors"]["static_I_Xa_min"] >= -1e-12 and c9g["C9c"]["sectors"]["static_I_Xs_min"] >= -1e-12
    v["C9c_static_timerow_piece_nonpositive"] = c9g["C9c"]["sectors"]["static_I_Ft_timerow_max"] <= 1e-12
    v["C10_no_cross_term_under_replacement"] = c10g["connection_level"]["replacement_cross_max_abs"] < 1e-12 and c10g["field_level"]["Gamma_cross_FminusXa_Xs_max_abs"] < 1e-10 and c9g["C9c"]["decomposition"]["max_abs_cross_FminusXa_Xs"] < 1e-10 * c9g["C9c"]["decomposition"]["Gam_scale"]
    v["C10_SO4_SO13_cross_opposite"] = c10g["connection_level"]["SO4_vs_SO13_sign_opposite_frac"] == 1.0
    v["C11_Gamma_quartic_in_omega"] = all(c11g[k]["c4_max"] > 1e-3 * c11g[k]["c2_max"] for k in ("Gam", "Gam_tl", "Bu")) and c11g["I1"]["c4_max"] < 1e-8 * c11g["I1"]["c2_max"]
    lad = c12g["ladders"]
    v["C12_rigid_null_diverges"] = lad["rigid_undressed"]["I1"]["ratio_last_first_core_r3"] > 1.5 and lad["rigid_undressed"]["Gam"]["ratio_last_first_core_r3"] > 1.5
    mu = lad["melted_undressed"]
    v["C12_melted_core_ball_bounded"] = all(max(mu[k]["core_r3"]) / min(mu[k]["core_r3"]) < 2.0 for k in ("I1", "Gam"))
    dtot = mu["I1"]["total"][-1] - mu["I1"]["total"][0]
    dax = mu["I1"]["axis_cyl"][-1] - mu["I1"]["axis_cyl"][0]
    v["C12_seed_growth_sits_on_the_axis_line"] = dax / dtot > 0.5 if abs(dtot) > 1e-12 else True
    v["C12_dressed_boost_pieces_unresolved_at_R3_h"] = lad["melted_dressed"]["I_Xs"]["ratio_last_first_core_r3"] > 2.0
    v["C12_registry_I1_gate"] = all(c12g["fields"][f][str(n)]["registry_I1_check_rel"] < 1e-10 for f in c12g["fields"] for n in c12g["ns"])
    out["verdict"] = {k: bool(x) for k, x in v.items()}
    os.makedirs(DATA, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    for k, x in out["verdict"].items():
        print(f"{'PASS' if x else 'FAIL'} {k}")
    print(json.dumps({"C9c": c9g["C9c"], "C10": c10g, "C11_g8": {k: c11g[k] for k in ("I1", "Bu", "Gam", "Gam_tl")}}, indent=1))
    print(json.dumps({"C12_ladders": {f: {k: lad[f][k] for k in ("I1", "Gam", "Gam_tl", "Bu", "I_Xa", "I_Xs", "I_Ft_timerow")} for f in lad}}, indent=1))
    return out


if __name__ == "__main__":
    main()
