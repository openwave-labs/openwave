"""M5.32 R16 shared instrument (ledger 6.5 as amended by R16-0): the author's local-circle
object v4 on the degenerate vacuum, every term circle-averaged, both H-adjoint completions,
the plateau-weighted projected stiffness with its Frechet derivative, the split-weighted
regulator, the local clock, a FIRE with callables, and the selftest gates.  R15's certified
pieces are consumed read-only through m5_32_r15_common (the patched certified stack, the
projectors P_g / P_1 / P23 with their pair resolvents R_g / R_1, the split term, the FIRE
logic).

EQUATIONS FIRST (E-orientation throughout: static +, inertia + omega^2; this IS the
author's -c K Lagrangian convention)
-------------------------------------------------------------------------------
Conventions: M real symmetric 4x4 per cell, eta = diag(-1, 1, 1, 1), N = M eta (eta-self-
adjoint, real spectrum), code branch s = -1, the degenerate vacuum M = diag(g, 1, delta, delta)
with the N-spectrum (-g, 1, delta, delta), g = 8, delta = 0.3.  Per cell the isolated
eigenvalues lambda_g (most negative) and lambda_1 (largest) with their spectral projectors
P_g, P_1 (matrix polynomials in N, Newton-polished, complex-step exact: R15), the pair
projector P23 = I - P_g - P_1, the pair resolvents R_j = (N - lambda_j)^-1 on the pair block.

The frame (all rational in M except one square root):
    u u^T = -P_g eta,  u = column j / sqrt((u u^T)_jj), j = argmax of the diagonal, u_0 > 0
    n n^T =  P_1 eta,  n likewise, then ORIENTED by a smooth lift: sign(n . n_ref) > 0,
             n_ref the outward radial direction at the seed and the previous step's n after
             (the director is an RP^2 field; T_alpha needs a local orientation, the average
             does not: alpha -> -alpha)
    G = eta + 2 (eta u)(eta u)^T = eta (I - 2 P_g)        (the author's H; positive definite)
    J^a_b = eta^aa eps_abcd u^c n^d                        (the rotation generator about n in
             the rest frame of u; eta-antisymmetric, J u = J n = 0, J^2 = -P23)
    R(beta) = I + sin(beta) J + (1 - cos(beta)) J^2       (a Lorentz map, R^T eta R = eta)
    T_alpha M = R(alpha/2) M R(alpha/2)^T                  (the author's local circle)
    a0 = J M + M J^T                                       (the local clock generator, the
             tangent of the circle; = J M - M J at u = e_0, R15's a0_local up to the lift sign)

The plain (unaveraged) action, densities per cell, h^3-weighted, A_i = d_i M on the certified
sym stencil (forward and backward branches averaged), A_0 = omega a0:
    quartic (both completions):  E_h = 4 sum_{i<j} tr(G F_ij G F_ij^T),  kin_h = 4 sum_i tr(G F_0i G F_0i^T)
        I_norm:    F_mn = A_m eta A_n - A_n eta A_m     (the registry's I1_h, R15's read)
        I_rebuild: F_mn = A_m G   A_n - A_n G   A_m     (the author's form, R16-0 C1 / C3)
    potential:  V4^dd = W1 sum_p (tr N^p - C_p)^2,  C_p = (-g)^p + 1 + 2 delta^p
    split:      U = mu rho^2,  rho^2 = (1/2) tr B^2 = (1/4)(lambda_2 - lambda_3)^2 = (s^2 - 4 p) / 4
    stiffness:  K_P^proj = (1/2)[sum_i tr(Om_i^T eta Om_i eta) + omega^2 tr(Om_0^T eta Om_0 eta)],
                Om_mu = w(N) A_mu eta w(N), the PLATEAU weight w(N) = sum_k w(lambda_k) P_k with
                w = 1 on |lambda - delta| <= 0.5, a cosine taper to 0 at lambda = 1 and at
                lambda = -1, 0 beyond; in the run's spectral domain (the pair inside the plateau,
                the director isolated) w(N) = P23 + w(lambda_1) P_1 + w(lambda_g) P_g exactly,
                w(lambda_g) = 0 at lambda_g = -g; the H-form equals the eta-form for any w with
                w(lambda_g) = 0 (w u = 0, (eta u)^T w = 0: the R15 theorem extended)
    regulator:  reg = c_s rho^2 E2,  E2 = sum_i tr(A_i G A_i G) (+ omega^2 tr(a0 G a0 G))
    E_stat = E_h + V4^dd + U + c_P K_P^proj_stat + reg_stat;  kin_tot = kin_h + c_P kin_KP + kin_reg
The circle-averaged action (the object of R16):  E_v4 = (1/n_s) sum_k E[T_(alpha_k) M, R_k a0 R_k^T],
alpha_k = 2 pi k / n_s.  The trigonometric degree in alpha: 2 in the continuum for the quartic
(A_k = R [A + Theta M + M Theta^T] R^T, Theta of degree 1 in beta = alpha / 2, quartic in A),
but on the LATTICE the finite difference of R(x) M(x) R(x)^T between neighboring cells carries
R(x)^T R(x + h), of degree 2 in beta, so the lattice quartic density has degree 4 in alpha
(the quadratic terms K_P, E2 degree 2): the 2-sample average is off by 1e-5 (random field) to
4e-2 (kin_h on the hedgehog core, where the frame turns by O(1) between cells), the 4-sample
average by 1e-9 (E_stat) to 1e-6 (kin_h), the 8-sample average EXACT (degree <= 7).  So R16-0
C4b's degree-1 statement holds on its point jets (the continuum with one jet) only.  The
instrument: n_s = 4 in the descents (an O(h^2)-level defect, stated), n_s = 8 in every read,
gate and verdict, with the 8 -> 16 doubling test at 1e-12 as the exactness gate.  Fixed K:  E_K = E_stat + K^2 / (4 kin_tot), a0 refreshed each step and frozen
in the gradient (the R15 protocol).

GRADIENTS (analytic adjoints, gated by complex step at 1e-8 and central differences at 1e-6):
    quartic pair density d = tr(G F G F^T), F = X K Y - Y K X (K = eta or G):
        dd/dF = 2 G F G =: Lam;  dd/dX = Lam Y K - K Y Lam;  dd/dY = K X Lam - Lam X K;
        dd/dG = F G F^T + F^T G F (+ X Lam Y - Y Lam X for K = G)
    E2:  d tr(A G A G)/dA = 2 G A G,  /dG = 2 A G A
    K_P: dE = tr(Y dOm), Y = eta Om^T eta, Z = X w Y + Y w X;  dE/dA = (eta w Y w)^T;
         tr(Z dw) = tr(W dN),  W = [P_g Z R_g + R_g Z P_g + P_1 Z R_1 + R_1 Z P_1]
                                  + sum_{j in g,1} [w'(l_j) tr(Z P_j) P_j - w(l_j)(P_j Z S_j + S_j Z P_j)]
         with S_1 = P_g / (l_g - l_1) + R_1, S_g = P_1 / (l_1 - l_g) + R_g (the reduced resolvents;
         dP_j = -(S_j dN P_j + P_j dN S_j), d lambda_j = tr(P_j dN))
    G:   dG = 2 eta (S_g dN P_g + P_g dN S_g)  ->  W_G = 2 (P_g Z_G S_g + S_g Z_G P_g), Z_G = Lam_G^T eta
    V4:  W = 2 W1 sum_p p (tr N^p - C_p) N^(p-1)
    every W (dE = tr(W dN)) becomes dE/dM = (eta W)^T symmetrized (dN = dM eta)
    the circle (the chain rule through R(n(M), u(M))):  with Lam_M = dE/dM_k, Lam_a = dE/da0_k,
        direct: R^T Lam_M R;  Lam_R = 2 (Lam_M R M + Lam_a R a0);
        Lam_J = sin(beta) Lam_R + (1 - cos(beta)) (Lam_R J^T + J^T Lam_R);
        (Lam_u)_c = sum Lam_J_ab Ebar_abcd n_d,  (Lam_n)_d = sum Lam_J_ab Ebar_abcd u_c;
        u = S[:, j] / sqrt(S_jj):  (Lam_S)_aj = (Lam_u)_a / sqrt(S_jj),  (Lam_S)_jj -= (Lam_u . u) / (2 S_jj);
        S_u = -P_g eta: W_u = P_g Z_u S_g + S_g Z_u P_g, Z_u = eta Lam_S^T;
        S_n = +P_1 eta: W_n = -(P_1 Z_n S_1 + S_1 Z_n P_1), Z_n = eta Lam_S^T (with the lift sign)

Selftest (python3 m5_32_r16_common.py): frame identities, the plateau-weight identities
(w = P23 where lambda_1 >= 1; w^2 != w in the taper; H-form == eta-form), rho^2 = s^2 on the
diagonal sheet, vacuum facts, every gradient gate (each part, static and kinetic, plain and
averaged, with the director in the taper), E_norm == the registry I1_h read, K_P^proj ==
K_P^23 where lambda_1 >= 1, a0 == R15's a0_local up to the lift sign, T_2pi = id, the doubling
test 2 -> 4 -> 8 samples, the symmetry-defect gate on the averaged action with the unaveraged
regulator as the control that must FAIL, covariance under global boosts and rotations with
the no-eta control failing, and the wall time per averaged energy + gradient at n32.
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import time

import numpy as np

ARGV = list(sys.argv)
sys.argv = [sys.argv[0]]
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
DATA, PLOTS = os.path.join(RES, "data"), os.path.join(RES, "plots")
CK = os.path.join(RES, "checkpoints", "m5_32_r16")
os.makedirs(CK, exist_ok=True)
T0 = time.time()


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C15 = _load("m5_32_r15_common", "m5_32_r15_common.py")
sys.argv = ARGV
INS4, EXT, C13, B8 = C15.INS4, C15.EXT, C15.C13, C15.B8
ETA, EYE = C15.ETA, C15.EYE
G, DELTA, S, W1 = C15.G, C15.DELTA, C15.S, C15.W1
_tr = C15._tr
KILL = os.path.join(RES, "killswitch")

# the plateau weight (the author's 2026-09-06 definition, R16-0 reading)
PL_LO, PL_HI, TAP_LO, TAP_HI = DELTA - 0.5, DELTA + 0.5, -1.0, 1.0
# Ebar_abcd = eta^aa eps_abcd
_EPS = np.zeros((4, 4, 4, 4))
for _perm, _sg in (((0, 1, 2, 3), 1), ((0, 1, 3, 2), -1), ((0, 2, 1, 3), -1), ((0, 2, 3, 1), 1), ((0, 3, 1, 2), 1), ((0, 3, 2, 1), -1),
                   ((1, 0, 2, 3), -1), ((1, 0, 3, 2), 1), ((1, 2, 0, 3), 1), ((1, 2, 3, 0), -1), ((1, 3, 0, 2), -1), ((1, 3, 2, 0), 1),
                   ((2, 0, 1, 3), 1), ((2, 0, 3, 1), -1), ((2, 1, 0, 3), -1), ((2, 1, 3, 0), 1), ((2, 3, 0, 1), 1), ((2, 3, 1, 0), -1),
                   ((3, 0, 1, 2), -1), ((3, 0, 2, 1), 1), ((3, 1, 0, 2), 1), ((3, 1, 2, 0), -1), ((3, 2, 0, 1), -1), ((3, 2, 1, 0), 1)):
    _EPS[_perm] = _sg
EBAR = np.diag(ETA)[:, None, None, None] * _EPS


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def killed():
    return os.path.exists(KILL)


def cfg_v4(n, L, mu=1e-2, cP=1.0, cs=0.4, completion="rebuild", n_samples=4, g=G, delta=DELTA):
    """the R16 configuration: the degenerate-vacuum certified cfg plus the v4 coefficients.
    mu multiplies rho^2 (the author's normalization; R15's cfg['mu'] multiplied 4 rho^2)."""
    cfg = C15.cfg_dd(n, L, mu=0.0, cP=0.0, g=g, delta=delta)
    cfg["mu"] = float(mu)
    cfg["cP"] = float(cP)
    cfg["cs"] = float(cs)
    cfg["completion"] = completion
    cfg["n_samples"] = int(n_samples)
    return cfg


# ------------------------------------------------ the plateau weight
def w_plateau(lam):
    """w(lambda): 1 on the plateau [delta - 0.5, delta + 0.5], cosine tapers to 0 at 1 and at -1,
    0 beyond.  Complex-step safe (the branch is chosen on the real part)."""
    x = np.real(lam)
    out = np.zeros_like(lam)
    on = (x >= PL_LO) & (x <= PL_HI)
    hi = (x > PL_HI) & (x < TAP_HI)
    lo = (x < PL_LO) & (x > TAP_LO)
    out = np.where(on, 1.0, out)
    out = np.where(hi, 0.5 * (1.0 + np.cos(np.pi * (lam - PL_HI) / (TAP_HI - PL_HI))), out)
    out = np.where(lo, 0.5 * (1.0 + np.cos(np.pi * (PL_LO - lam) / (PL_LO - TAP_LO))), out)
    return out


def dw_plateau(lam):
    x = np.real(lam)
    out = np.zeros_like(lam)
    hi = (x > PL_HI) & (x < TAP_HI)
    lo = (x < PL_LO) & (x > TAP_LO)
    out = np.where(hi, -0.5 * np.pi / (TAP_HI - PL_HI) * np.sin(np.pi * (lam - PL_HI) / (TAP_HI - PL_HI)), out)
    out = np.where(lo, 0.5 * np.pi / (PL_LO - TAP_LO) * np.sin(np.pi * (PL_LO - lam) / (PL_LO - TAP_LO)), out)
    return out


# ------------------------------------------------ the frame
def _col_unit(Smat):
    """v with v v^T = Smat (rank one, PSD on the real part): v = column j / sqrt(S_jj), j = argmax diag."""
    d = np.real(np.einsum("...aa->...a", Smat))
    j = np.argmax(d, axis=-1)
    col = np.take_along_axis(Smat, j[..., None, None], axis=-1)[..., :, 0]
    sjj = np.take_along_axis(np.einsum("...aa->...a", Smat), j[..., None], axis=-1)[..., 0]
    return col / np.sqrt(sjj)[..., None], j


def frame(M, n_ref=None):
    """the per-cell frame: the R15 projectors and resolvents, u, n (lifted), G, J, and the weight
    pieces.  n_ref (..., 4) orients n (n . n_ref > 0); None = the raw column sign."""
    N, Pg, P1, P23, Rg, R1, lg, l1, s, p = C15.projectors(M)
    Su = -Pg @ ETA
    Sn = P1 @ ETA
    u, _ = _col_unit(Su)
    n, _ = _col_unit(Sn)
    if n_ref is not None:
        sg = np.where(np.real(np.einsum("...a,...a->...", n, n_ref)) >= 0.0, 1.0, -1.0)
        n = n * sg[..., None]
    else:
        sg = np.ones(M.shape[:-2])
    I = np.broadcast_to(EYE, N.shape)
    Gm = ETA @ (I - 2.0 * Pg)
    J = np.einsum("abcd,...c,...d->...ab", EBAR, u, n)
    Sg = P1 / (l1 - lg)[..., None, None] + Rg
    S1 = Pg / (lg - l1)[..., None, None] + R1
    fr = {"N": N, "Pg": Pg, "P1": P1, "P23": P23, "Rg": Rg, "R1": R1, "lg": lg, "l1": l1, "s": s, "p": p,
          "u": u, "n": n, "nsign": sg, "G": Gm, "J": J, "Sg": Sg, "S1": S1,
          "w1": w_plateau(l1), "dw1": dw_plateau(l1), "wg": w_plateau(lg), "dwg": dw_plateau(lg)}
    # w = I - (1 - w_g) P_g - (1 - w_1) P_1  (== P23 + w_1 P_1 + w_g P_g; this form multiplies the
    # director's projector by (1 - w_1), exactly 0 wherever the director sits inside the plateau,
    # so a director nearly degenerate with the pair (the hedgehog's isotropic center) is harmless)
    c1 = (1.0 - fr["w1"])[..., None, None]
    cg = (1.0 - fr["wg"])[..., None, None]
    fr["c1"], fr["cg"] = c1, cg
    fr["w"] = I - _safe_mul(cg, Pg) - _safe_mul(c1, P1)
    # the pair outside the plateau (a large split, or the pair shifted): the restricted form above
    # keeps weight 1 on the pair; there the label-free spectral function (a full eigendecomposition
    # in the G-metric, Daleckii-Krein derivative) replaces it, on those cells only (real fields)
    sr, pr = np.real(s), np.real(p)
    disc = np.sqrt(np.maximum(sr * sr - 4.0 * pr, 0.0))
    l2, l3 = (sr + disc) / 2.0, (sr - disc) / 2.0
    out = (l2 > PL_HI) | (l3 < PL_LO)
    fr["pair_out"] = out
    if np.any(out) and not np.iscomplexobj(M):
        gen = general_weight(M[out], u[out])
        fr["w"] = fr["w"].copy()
        fr["w"][out] = gen["w"]
        fr["general"] = gen
    return fr


def boost_of(u):
    """the pure boost B with B e_0 = u (symmetric, B^T eta B = eta); G = B^-2, G^(1/2) = eta B eta."""
    g = u[..., 0]
    bs = u[..., 1:]
    b2 = np.sum(bs * bs, axis=-1)
    B = np.zeros(u.shape + (4,))
    B[..., 0, 0] = g
    B[..., 0, 1:] = bs
    B[..., 1:, 0] = bs
    fac = np.where(b2 > 1e-300, (g - 1.0) / np.maximum(b2, 1e-300), 0.0)
    B[..., 1:, 1:] = np.eye(3) + fac[..., None, None] * bs[..., :, None] * bs[..., None, :]
    return B


def general_weight(M, u):
    """the label-free spectral function w(N) = V w(Lambda) V^-1 with V = G^(-1/2) Q, Q the
    orthonormal eigenvectors of the symmetric G^(1/2) N G^(-1/2), and the pieces of its
    Daleckii-Krein derivative (the divided-difference matrix)."""
    N = M @ ETA
    B = boost_of(u)
    Gh = ETA @ B @ ETA                     # G^(1/2)
    Ghi = B                                # G^(-1/2)
    Ns = Gh @ N @ Ghi
    Ns = 0.5 * (Ns + np.swapaxes(Ns, -1, -2))
    lam, Q = np.linalg.eigh(Ns)
    V = Ghi @ Q
    Vi = np.swapaxes(Q, -1, -2) @ Gh
    wl = w_plateau(lam)
    dl = lam[..., :, None] - lam[..., None, :]
    dw = wl[..., :, None] - wl[..., None, :]
    close = np.abs(dl) < 1e-7
    mid = 0.5 * (lam[..., :, None] + lam[..., None, :])
    Wdd = np.where(close, dw_plateau(mid), dw / np.where(close, 1.0, dl))
    w = V @ (wl[..., :, None] * Vi)
    return {"w": w, "V": V, "Vi": Vi, "Wdd": Wdd, "lam": lam}


def _safe_mul(c, X):
    """c * X with 0 wherever c == 0 exactly (X may be huge or non-finite there)."""
    return np.where(np.real(c) == 0.0, 0.0, c * np.where(np.isfinite(X), X, 0.0))


GAP_MIN = 1e-3


def domain(fr, cfg=None):
    """the spectral-domain statement: pair eigenvalues, their plateau membership, the director's
    isolation and taper admixture.  Escape (d) = the director leaves isolation (its gap to the
    pair below GAP_MIN anywhere: the local circle's axis undefined).  The taper-core radius r_d
    (the largest radius with lambda_1 < 0.8; the hedgehog's isotropic center has one on the
    seed already) is a diagnostic, reported, not a stop."""
    s, p = np.real(fr["s"]), np.real(fr["p"])
    disc = np.sqrt(np.maximum(s * s - 4.0 * p, 0.0))
    l2, l3 = (s + disc) / 2.0, (s - disc) / 2.0
    l1, lg = np.real(fr["l1"]), np.real(fr["lg"])
    out = {"l1_min": float(np.min(l1)), "l1_max": float(np.max(l1)), "lg_max": float(np.max(lg)),
           "l2_max": float(np.max(l2)), "l3_min": float(np.min(l3)),
           "gap_1_2_min": float(np.min(l1 - l2)), "pair_in_plateau": bool(np.min(l3) >= PL_LO and np.max(l2) <= PL_HI),
           "director_isolated": bool(np.min(l1 - l2) > GAP_MIN), "w1_max": float(np.max(np.real(fr["w1"]))),
           "cells_in_taper": int(np.sum(l1 < TAP_HI - 1e-12)), "cells_director_in_plateau": int(np.sum(l1 < PL_HI)),
           "escape_d": bool(np.min(l1 - l2) <= GAP_MIN), "half_split_max": float(np.max(disc) / 2.0),
           "cells_pair_outside_plateau": int(np.sum(fr["pair_out"])) if "pair_out" in fr else 0}
    if cfg is not None and l1.ndim == 3:
        X, Y, Z = INS4.coords(cfg["n"], cfg["h"])
        r = np.sqrt(X * X + Y * Y + Z * Z)
        inpl = l1 < PL_HI
        out["r_d"] = float(np.max(r[inpl])) if np.any(inpl) else 0.0
        intap = l1 < TAP_HI - 1e-9
        out["r_taper"] = float(np.max(r[intap])) if np.any(intap) else 0.0
    return out


def rot_R(J, beta):
    I = np.broadcast_to(EYE, J.shape)
    return I + np.sin(beta) * J + (1.0 - np.cos(beta)) * (J @ J)


def a0_of(M, fr=None, n_ref=None):
    """the local clock generator a0 = J M + M J^T (the circle's tangent at M)."""
    if fr is None:
        fr = frame(M, n_ref)
    J = fr["J"]
    return J @ M + M @ np.swapaxes(J, -1, -2)


def W_to_gradM(W):
    Gm = np.swapaxes(ETA @ W, -1, -2)
    return 0.5 * (Gm + np.swapaxes(Gm, -1, -2))


def sym(X):
    return 0.5 * (X + np.swapaxes(X, -1, -2))


# ------------------------------------------------ per-cell densities with adjoints
def quartic_pair(X, Y, Gm, comp, need_grad=True):
    """d = tr(G F G F^T), F = X K Y - Y K X, K = eta (norm) or G (rebuild).  Returns
    (d, dX, dY, dG) per cell (unsymmetrized dX, dY)."""
    K = Gm if comp == "rebuild" else np.broadcast_to(ETA, Gm.shape)
    F = X @ K @ Y - Y @ K @ X
    GF = Gm @ F
    d = _tr(GF @ Gm @ np.swapaxes(F, -1, -2))
    if not need_grad:
        return d, None, None, None
    Lam = 2.0 * GF @ Gm
    dX = Lam @ Y @ K - K @ Y @ Lam
    dY = K @ X @ Lam - Lam @ X @ K
    FT = np.swapaxes(F, -1, -2)
    dG = F @ Gm @ FT + FT @ Gm @ F
    if comp == "rebuild":
        dG = dG + X @ Lam @ Y - Y @ Lam @ X
    return d, dX, dY, dG


def e2_cells(A, Gm, need_grad=True):
    """tr(A G A G) per cell, dA = 2 G A G, dG = 2 A G A."""
    AG = A @ Gm
    d = _tr(AG @ AG)
    if not need_grad:
        return d, None, None
    return d, 2.0 * Gm @ A @ Gm, 2.0 * A @ Gm @ A


def kp_cells(A_list, fr, need_grad=True):
    """K_P^proj per cell for the jets in A_list (each frozen): E = (1/2) sum tr(Om^T eta Om eta),
    Om = w X w, X = A eta.  Returns (E, W (the dN cotangent through w), [dA])."""
    w = fr["w"]
    E = np.zeros(w.shape[:-2], dtype=w.dtype)
    Zsum = np.zeros_like(w)
    dA_out = []
    for A in A_list:
        X = A @ ETA
        Om = w @ X @ w
        Y = ETA @ np.swapaxes(Om, -1, -2) @ ETA
        E = E + 0.5 * _tr(Y @ Om)
        if need_grad:
            dA_out.append(np.swapaxes(ETA @ w @ Y @ w, -1, -2))
            Zsum = Zsum + X @ w @ Y + Y @ w @ X
    if not need_grad:
        return E, None, None
    return E, W_through_w(Zsum, fr), dA_out


def W_through_w(Z, fr):
    """tr(Z dw) = tr(W dN) for w = P23 + w(l1) P1 + w(lg) Pg."""
    W = np.zeros_like(Z)
    for Pj, Sj, cj, dwj in ((fr["P1"], fr["S1"], fr["c1"], fr["dw1"]), (fr["Pg"], fr["Sg"], fr["cg"], fr["dwg"])):
        W = W + _safe_mul(cj, Pj @ Z @ Sj + Sj @ Z @ Pj) + _safe_mul(dwj[..., None, None], _tr(Z @ Pj)[..., None, None] * Pj)
    if "general" in fr:
        gen = fr["general"]
        out = fr["pair_out"]
        Zt = gen["Vi"] @ Z[out] @ gen["V"]
        W = W.copy()
        W[out] = gen["V"] @ (Zt * gen["Wdd"]) @ gen["Vi"]
    return W


def W_through_G(LamG, fr):
    """tr(LamG^T dG) = tr(W dN) for G = eta (I - 2 P_g)."""
    Pg, Sg = fr["Pg"], fr["Sg"]
    ZG = np.swapaxes(LamG, -1, -2) @ ETA
    return 2.0 * (Pg @ ZG @ Sg + Sg @ ZG @ Pg)


def v4_cells(fr, cfg, need_grad=True):
    N = fr["N"]
    cp = C15.cp_dd(cfg["g"], cfg["delta"])
    P = N
    t, pows = [], [np.broadcast_to(EYE, N.shape)]
    for q in range(4):
        if q:
            P = P @ N
        t.append(_tr(P))
        pows.append(P)
    v = W1 * sum((t[q] - cp[q]) ** 2 for q in range(4))
    if not need_grad:
        return v, None
    W = sum((2.0 * W1 * (q + 1) * (t[q] - cp[q]))[..., None, None] * pows[q] for q in range(4))
    return v, W


# ------------------------------------------------ the plain action
def action(M, cfg, a0=None, need_grad=True, need_a0grad=False, fr=None, n_ref=None):
    """the plain (unaveraged) v4 action on a lattice field: parts (h^3-weighted), the static
    gradient, the kinetic gradient at frozen a0, and optionally d kin / d a0."""
    h = cfg["h"]
    h3 = h ** 3
    comp = cfg["completion"]
    mu, cP, cs = cfg["mu"], cfg["cP"], cfg["cs"]
    if fr is None:
        fr = frame(M, n_ref)
    Gm = fr["G"]
    cplx = np.iscomplexobj(M)
    num = (lambda x: complex(x)) if cplx else (lambda x: float(x))
    spl, dspl = C15.split_cells(M, need_grad)
    rho2 = spl / 4.0
    v4, W_v4 = v4_cells(fr, cfg, need_grad)
    Eh = np.zeros(M.shape[:-2], dtype=M.dtype)
    e2, kp, kh, ke2, kkp = (np.zeros_like(Eh) for _ in range(5))
    adj_stat = np.zeros_like(M) if need_grad else None
    adj_kin = np.zeros_like(M) if (need_grad and a0 is not None) else None
    Wst, Wkin, LamG_st, LamG_kin = (np.zeros_like(M) for _ in range(4))
    ga0 = np.zeros_like(M) if (a0 is not None and need_a0grad) else None
    for br, wt in INS4.branches(cfg["stencil"]):
        A = [INS4.d1(M, ax, h, br) for ax in range(3)]
        gA = [np.zeros_like(M) for _ in range(3)] if need_grad else None
        gAk = [np.zeros_like(M) for _ in range(3)] if (need_grad and a0 is not None) else None
        for i in range(3):
            for j in range(i + 1, 3):
                d, dX, dY, dG = quartic_pair(A[i], A[j], Gm, comp, need_grad)
                Eh = Eh + wt * 4.0 * d
                if need_grad:
                    gA[i] += 4.0 * dX
                    gA[j] += 4.0 * dY
                    LamG_st += wt * 4.0 * dG
        for i in range(3):
            d, dA, dG = e2_cells(A[i], Gm, need_grad)
            e2 = e2 + wt * d
            if need_grad:
                gA[i] += cs * rho2[..., None, None] * dA
                LamG_st += wt * cs * rho2[..., None, None] * dG
        Ek, Wk, dAk = kp_cells(A, fr, need_grad)
        kp = kp + wt * Ek
        if need_grad:
            Wst += wt * cP * Wk
            for i in range(3):
                gA[i] += cP * dAk[i]
                adj_stat += wt * INS4.d1_adj(sym(gA[i]), i, h, br)
        if a0 is not None:
            for i in range(3):
                d, dX, dY, dG = quartic_pair(a0, A[i], Gm, comp, need_grad)
                kh = kh + wt * 4.0 * d
                if need_grad:
                    gAk[i] += 4.0 * dY
                    LamG_kin += wt * 4.0 * dG
                    adj_kin += wt * INS4.d1_adj(sym(gAk[i]), i, h, br)
                    if ga0 is not None:
                        ga0 += wt * 4.0 * dX
    if a0 is not None:
        Ekk, Wkk, dAkk = kp_cells([a0], fr, need_grad)
        kkp = kkp + Ekk
        d, dA, dG = e2_cells(a0, Gm, need_grad)
        ke2 = ke2 + d
        if need_grad:
            Wkin += cP * Wkk
            LamG_kin += cs * rho2[..., None, None] * dG
            if ga0 is not None:
                ga0 += cP * dAkk[0] + cs * rho2[..., None, None] * dA
    parts = {"E_h": num(h3 * np.sum(Eh)), "V4": num(h3 * np.sum(v4)), "U": num(h3 * mu * np.sum(rho2)),
             "KP": num(h3 * cP * np.sum(kp)), "reg": num(h3 * cs * np.sum(rho2 * e2)),
             "E2_raw": num(h3 * np.sum(e2)), "rho2_sum": num(h3 * np.sum(rho2)), "KP_raw": num(h3 * np.sum(kp))}
    parts["E_stat"] = parts["E_h"] + parts["V4"] + parts["U"] + parts["KP"] + parts["reg"]
    if a0 is not None:
        parts["kin_h"] = num(h3 * np.sum(kh))
        parts["kin_KP"] = num(h3 * cP * np.sum(kkp))
        parts["kin_reg"] = num(h3 * cs * np.sum(rho2 * ke2))
        parts["kin_tot"] = parts["kin_h"] + parts["kin_KP"] + parts["kin_reg"]
    out = {"parts": parts, "fr": fr, "rho2": rho2, "e2": e2}
    if a0 is not None:
        out["kin_cells"] = h3 * (kh + cP * kkp + cs * rho2 * ke2)
    if not need_grad:
        return out
    Gst = W_to_gradM(Wst + W_v4 + W_through_G(LamG_st, fr))
    Gst += (mu + cs * e2)[..., None, None] * dspl / 4.0
    out["grad_stat"] = h3 * (Gst + adj_stat)
    if a0 is not None:
        Gk = W_to_gradM(Wkin + W_through_G(LamG_kin, fr))
        Gk += (cs * ke2)[..., None, None] * dspl / 4.0
        out["grad_kin"] = h3 * (Gk + adj_kin)
        if ga0 is not None:
            out["grad_a0"] = h3 * sym(ga0)
    return out


# ------------------------------------------------ the circle
def circle_adjoint(fr, beta, M, a0, LamM, Lama):
    """the pull-back of the cotangents (LamM on M_k = R M R^T, Lama on a0_k = R a0 R^T) to M,
    through R(beta) = R(J(u(M), n(M))); returns the per-cell gradient contribution."""
    J = fr["J"]
    R = rot_R(J, beta)
    RT = np.swapaxes(R, -1, -2)
    JT = np.swapaxes(J, -1, -2)
    grad = RT @ LamM @ R
    LamR = 2.0 * (LamM @ R @ M)
    if Lama is not None:
        LamR = LamR + 2.0 * (Lama @ R @ a0)
    LamJ = np.sin(beta) * LamR + (1.0 - np.cos(beta)) * (LamR @ JT + JT @ LamR)
    Lu = np.einsum("...ab,abcd,...d->...c", LamJ, EBAR, fr["n"])
    Ln = np.einsum("...ab,abcd,...c->...d", LamJ, EBAR, fr["u"])
    W = np.zeros_like(M)
    for vec, Lv, Smat, sign_dP, Pj, Sj in ((fr["u"], Lu, -fr["Pg"] @ ETA, +1.0, fr["Pg"], fr["Sg"]),
                                          (fr["n"] * fr["nsign"][..., None], Ln * fr["nsign"][..., None], fr["P1"] @ ETA, -1.0, fr["P1"], fr["S1"])):
        d = np.real(np.einsum("...aa->...a", Smat))
        j = np.argmax(d, axis=-1)
        sjj = np.take_along_axis(np.einsum("...aa->...a", Smat), j[..., None], axis=-1)[..., 0]
        LamS = np.zeros_like(M)
        np.put_along_axis(LamS, j[..., None, None] * np.ones(M.shape[:-2] + (4, 1), dtype=int), (Lv / np.sqrt(sjj)[..., None])[..., :, None], axis=-1)
        corr = -0.5 * np.einsum("...a,...a->...", Lv, vec) / sjj
        idx = np.arange(4)
        diag_add = np.zeros_like(M)
        onehot = (idx[None, :] == j[..., None]).astype(M.dtype)
        diag_add[..., idx, idx] = corr[..., None] * onehot
        LamS = LamS + diag_add
        Z = ETA @ np.swapaxes(LamS, -1, -2)
        W = W + sign_dP * (Pj @ Z @ Sj + Sj @ Z @ Pj)
    return grad + W_to_gradM(W)


def averaged(M, cfg, a0=None, need_grad=True, need_a0grad=False, n_samples=None, n_ref=None):
    """the circle-averaged action E_v4 = (1/n_s) sum_k E[T_(alpha_k) M, R_k a0 R_k^T]."""
    ns = int(cfg["n_samples"] if n_samples is None else n_samples)
    fr0 = frame(M, n_ref)
    parts = None
    gst = np.zeros_like(M) if need_grad else None
    gkin = np.zeros_like(M) if (need_grad and a0 is not None) else None
    ga0 = np.zeros_like(M) if (need_a0grad and a0 is not None) else None
    dom = domain(fr0, cfg)
    for k in range(ns):
        beta = np.pi * k / ns
        if k == 0:
            Mk, a0k = M, a0
            res = action(M, cfg, a0, need_grad, need_a0grad or (need_grad and a0 is not None), fr=fr0)
        else:
            R = rot_R(fr0["J"], beta)
            RT = np.swapaxes(R, -1, -2)
            Mk = R @ M @ RT
            a0k = (R @ a0 @ RT) if a0 is not None else None
            res = action(Mk, cfg, a0k, need_grad, need_a0grad or (need_grad and a0 is not None), n_ref=fr0["n"])
        pk = res["parts"]
        parts = {kk: (pk[kk] / ns if parts is None else parts[kk] + pk[kk] / ns) for kk in pk}
        if need_grad:
            if k == 0:
                gst += res["grad_stat"] / ns
                if a0 is not None:
                    gkin += res["grad_kin"] / ns
            else:
                gst += circle_adjoint(fr0, beta, M, a0, res["grad_stat"], None) / ns
                if a0 is not None:
                    gkin += circle_adjoint(fr0, beta, M, a0, res["grad_kin"], res["grad_a0"]) / ns
        if ga0 is not None:
            if k == 0:
                ga0 += res["grad_a0"] / ns
            else:
                ga0 += (RT @ res["grad_a0"] @ R) / ns
    out = {"parts": parts, "fr": fr0, "domain": dom}
    if need_grad:
        out["grad_stat"] = gst
        if a0 is not None:
            out["grad_kin"] = gkin
    if ga0 is not None:
        out["grad_a0"] = ga0
    return out


def kin_a0_grad(M, cfg, v, n_ref=None, n_samples=None, fr0=None):
    """d kin_tot / d a0 at a0 = v on the circle-averaged action (kin is a pointwise quadratic form
    in a0: this is 2 T v with T the per-cell 2 x 2 inertia on the doublet, the R16-2 operator).
    Returns (grad_a0, kin_tot, kin_cells)."""
    ns = int(cfg["n_samples"] if n_samples is None else n_samples)
    if fr0 is None:
        fr0 = frame(M, n_ref)
    h3 = cfg["h"] ** 3
    comp, cP, cs = cfg["completion"], cfg["cP"], cfg["cs"]
    g = np.zeros_like(M)
    kin = 0.0
    kc = np.zeros(M.shape[:-2])
    for k in range(ns):
        beta = np.pi * k / ns
        if k == 0:
            Mk, vk, fr = M, v, fr0
        else:
            R = rot_R(fr0["J"], beta)
            RT = np.swapaxes(R, -1, -2)
            Mk, vk = R @ M @ RT, R @ v @ RT
            fr = frame(Mk, fr0["n"])
        Gm = fr["G"]
        spl, _ = C15.split_cells(Mk, need_grad=False)
        rho2 = spl / 4.0
        gk = np.zeros_like(M)
        kh = np.zeros(M.shape[:-2], dtype=M.dtype)
        for br, wt in INS4.branches(cfg["stencil"]):
            A = [INS4.d1(Mk, ax, cfg["h"], br) for ax in range(3)]
            for i in range(3):
                d, dX, _, _ = quartic_pair(vk, A[i], Gm, comp, True)
                kh = kh + wt * 4.0 * d
                gk += wt * 4.0 * dX
        Ekk, _, dAkk = kp_cells([vk], fr, True)
        d2, dA2, _ = e2_cells(vk, Gm, True)
        gk += cP * dAkk[0] + cs * rho2[..., None, None] * dA2
        kcell = h3 * (kh + cP * Ekk + cs * rho2 * d2)
        kc += np.real(kcell) / ns
        kin += float(np.sum(np.real(kcell))) / ns
        gk = h3 * sym(gk)
        if k == 0:
            g += gk / ns
        else:
            g += (RT @ gk @ R) / ns
    return g, kin, kc


# ------------------------------------------------ fixed-K energy and FIRE
def energy_and_grad(M, cfg, K=None, n_ref=None, need_grad=True, a0_frozen=None):
    """E_stat (K None) or E_K = E_stat + K^2 / (4 kin_tot) with a0 = a0_of(M) refreshed here and
    frozen in the gradient (a0_frozen overrides the refresh: the gate and the stationarity read)."""
    fr = frame(M, n_ref)
    a0 = a0_of(M, fr) if (K is not None and a0_frozen is None) else a0_frozen
    res = averaged(M, cfg, a0, need_grad, need_a0grad=need_grad and a0 is not None, n_ref=n_ref)
    pp = res["parts"]
    if K is None:
        E = pp["E_stat"]
        g = res["grad_stat"] if need_grad else None
    else:
        kin = pp["kin_tot"]
        E = pp["E_stat"] + K * K / (4.0 * kin)
        pp["E_K"], pp["omega"] = E, K / (2.0 * kin)
        g = (res["grad_stat"] - (K * K / (4.0 * kin * kin)) * res["grad_kin"]) if need_grad else None
    return E, g, pp, res["domain"], fr


def fire_v4(M0, cfg, free_mask, max_iter, K=None, n_ref=None, log_every=100, tag="", f_tol=1e-6,
            plateau=(2000, 1e-10), dt0=0.01, dt_max=0.1, diag=None, ck_path=None, ck_every=500):
    """C15.fire_lp's FIRE logic on the circle-averaged v4 action (static or fixed K).  The
    director lift n_ref is propagated (n_ref <- the current oriented n every step).  Stops on
    the killswitch file, on escape (d) (the director fully in the clock block), or non-finite."""
    M = M0.copy()
    free = free_mask[..., None, None].astype(float)
    v = np.zeros_like(M)
    dt, alpha, n_up = dt0, 0.1, 0
    hist = []
    nref = n_ref
    E, F, pp, dom, fr = energy_and_grad(M, cfg, K, nref)
    nref = fr["n"]
    F = -F * free
    t0 = time.time()
    stop = "max_iter"
    it = 0
    for it in range(1, max_iter + 1):
        Pw = float(np.sum(F * v))
        if Pw > 0.0:
            n_up += 1
            vn = np.sqrt(np.sum(v * v))
            fn = np.sqrt(np.sum(F * F))
            v = (1 - alpha) * v + alpha * (F / max(fn, 1e-300)) * vn
            if n_up > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            v[:] = 0.0
            dt *= 0.5
            alpha = 0.1
            n_up = 0
        v += dt * F
        M += dt * v
        try:
            E, F, pp, dom, fr = energy_and_grad(M, cfg, K, nref)
        except np.linalg.LinAlgError as e:                                  # a blown-up cell: the eigensolver fails
            print(f"  {tag} it {it}: {e!r}: the field is non-finite or degenerate; stopping", flush=True)
            stop = "non-finite"
            break
        nref = fr["n"]
        F = -F * free
        fmax = float(np.max(np.abs(F)))
        if not np.isfinite(fmax):
            stop = "non-finite"
            break
        if it % log_every == 0 or it == max_iter:
            row = {"it": it, "fmax": fmax, "dt": dt}
            row.update(pp)
            row.update({f"dom_{kk}": vv for kk, vv in dom.items()})
            if diag is not None:
                row.update(diag(M, fr))
            hist.append(row)
            key = "E_K" if K is not None else "E_stat"
            extra = f" kin {pp['kin_tot']:10.4f} om {pp['omega']:.5f}" if K is not None else ""
            print(f"  {tag} it {it:6d} {key} {pp[key]:14.6f} E_h {pp['E_h']:10.5f} V4 {pp['V4']:.4e} U {pp['U']:.4e} KP {pp['KP']:.4e} "
                  f"reg {pp['reg']:.4e} fmax {fmax:.3e}{extra} l1min {dom['l1_min']:.4f} w1max {dom['w1_max']:.3f} split {dom['half_split_max']:.4f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            back = max(1, plateau[0] // max(log_every, 1))
            if len(hist) > back and abs(row[key] - hist[-1 - back][key]) < plateau[1]:
                stop = "plateau"
                break
            if ck_path is not None and it % ck_every == 0:
                np.save(ck_path, M)
            if dom["escape_d"]:
                stop = "escape_d"
                break
            if killed():
                stop = "killswitch"
                break
        if fmax < f_tol:
            stop = "f_tol"
            break
    if ck_path is not None:
        np.save(ck_path, M)
    return M, {"stop": stop, "trace": hist, "wall_s": round(time.time() - t0, 1), "iters": it, "n_ref": nref}


# ------------------------------------------------ seeds and reads
def radial_ref(cfg, kind="radial"):
    """the initial director lift: 'radial' (outward, smooth on hedgehog textures) or 'x' (uniform
    e_1, smooth on the vacuum and on uniform-director fields; the outward lift FLIPS across the
    plane x = 0 on a uniform director and the circle samples then carry a discontinuity plane)."""
    X, Y, Z = INS4.coords(cfg["n"], cfg["h"])
    nref = np.zeros((cfg["n"],) * 3 + (4,))
    if kind == "x":
        nref[..., 1] = 1.0
        return nref
    r = np.sqrt(X * X + Y * Y + Z * Z)
    nref[..., 1], nref[..., 2], nref[..., 3] = X / r, Y / r, Z / r
    return nref


def lift_quality(fr, n_ref):
    """min over cells of |n . n_ref| (eta metric): near 0 the lift is ambiguous there."""
    d = np.abs(np.real(np.einsum("...a,ab,...b->...", fr["n"], ETA, n_ref)))
    return float(np.min(d)), int(np.sum(d < 0.2))


def load_r15_seed(n, L):
    """the R15-M relaxed hedgehog (mu 1e-2, c_P 1) on n{n} L{L} (local checkpoint, gitignored)."""
    p = os.path.join(RES, "checkpoints", "m5_32_r15", "m_hedgehog", f"relax_n{n}_L{int(L)}_mu0.01_cP1.npy")
    return np.load(p), p


def random_spectral_field(rng, n, cfg, dir_noise=(-0.15, 0.05), pair_noise=0.12, tilt=0.3, boost=0.05, smooth=2, pair_offset=(0.0, 0.0)):
    """a smooth random field with CONTROLLED spectrum: M = L D L^T, D = diag(g + n_g, 1 + n_1,
    delta + n_2, delta + n_3), L a smooth random rotation times a small boost.  n_1 in dir_noise
    puts the director in the taper (n_1 < 0) or above 1 (n_1 > 0)."""
    def sm(X):
        for _ in range(smooth):
            for ax in range(3):
                X = 0.5 * (X + np.roll(X, 1, axis=ax))
                X = 0.5 * (X + np.roll(X, -1, axis=ax))
        return X
    shape = (n, n, n)
    ng = 0.3 * sm(rng.normal(size=shape))
    n1 = sm(rng.uniform(dir_noise[0], dir_noise[1], size=shape))
    n2 = pair_noise * sm(rng.normal(size=shape))
    n3 = pair_noise * sm(rng.normal(size=shape))
    D = np.zeros(shape + (4, 4))
    D[..., 0, 0] = G + ng
    D[..., 1, 1] = 1.0 + n1
    D[..., 2, 2] = DELTA + n2 + pair_offset[0]
    D[..., 3, 3] = DELTA + n3 + pair_offset[1]
    Lm = np.broadcast_to(EYE, shape + (4, 4)).copy()
    for (a, b) in ((1, 2), (2, 3), (1, 3)):
        th = tilt * sm(rng.normal(size=shape))
        Rm = np.broadcast_to(EYE, shape + (4, 4)).copy()
        Rm[..., a, a], Rm[..., b, b] = np.cos(th), np.cos(th)
        Rm[..., a, b], Rm[..., b, a] = -np.sin(th), np.sin(th)
        Lm = Lm @ Rm
    for a in (1, 2, 3):
        ch = boost * sm(rng.normal(size=shape))
        Bm = np.broadcast_to(EYE, shape + (4, 4)).copy()
        Bm[..., 0, 0], Bm[..., a, a] = np.cosh(ch), np.cosh(ch)
        Bm[..., 0, a], Bm[..., a, 0] = np.sinh(ch), np.sinh(ch)
        Lm = Lm @ Bm
    return Lm @ D @ np.swapaxes(Lm, -1, -2)


def spectral_reference_w(M):
    """the general spectral function w(N) = V w(Lambda) V^-1 from a full eigendecomposition (a
    label-free reference for the restricted form; not used in the run)."""
    lam, V = np.linalg.eig(M @ ETA)
    lam, V = lam.real, V.real
    w = V @ (w_plateau(lam)[..., :, None] * np.linalg.inv(V))
    return w, lam


# ------------------------------------------------ selftests
def selftest(write=True, n=6, L=9.0):
    res, lines = {}, []
    rng = np.random.default_rng(1606)

    def check(name, ok, val):
        res[name] = {"ok": bool(ok), "value": val}
        lines.append(f"{'PASS' if ok else 'FAIL'} {name}: {val}")
        log(lines[-1])

    def gate(name, fn_e, fn_g, M, D, tol_fd=1e-6, tol_cs=1e-8):
        g = fn_g(M)
        an = float(np.sum(g * D))
        eps = 1e-5
        fd = (float(fn_e(M + eps * D)) - float(fn_e(M - eps * D))) / (2 * eps)
        try:
            cs = float(np.imag(fn_e(M + 1e-20j * D)) / 1e-20)
        except Exception as e:                                                          # noqa: BLE001
            cs = f"complex step unavailable: {e!r}"
        rel_fd = abs(an - fd) / max(abs(fd), 1e-300)
        rel_cs = (abs(an - cs) / max(abs(cs), 1e-300)) if isinstance(cs, float) else None
        ok = rel_fd < tol_fd and (rel_cs is None or rel_cs < tol_cs)
        check(f"gradient gate {name}", ok, {"analytic": an, "fd": fd, "cs": cs, "rel_fd": rel_fd, "rel_cs": rel_cs})

    cfgs = {c: cfg_v4(n, L, completion=c) for c in ("norm", "rebuild")}
    cfg = cfgs["rebuild"]
    M = random_spectral_field(rng, n, cfg)                       # the director in the taper on ~3/4 of the cells
    Mup = random_spectral_field(rng, n, cfg, dir_noise=(0.0, 0.05))   # lambda_1 >= 1 everywhere
    fr = frame(M)
    dom = domain(fr)
    res["domain_random"] = dom
    log(f"random field domain: {dom}")
    # 1. frame identities
    u, nn, J, P23 = fr["u"], fr["n"], fr["J"], fr["P23"]
    check("u^T eta u = -1, n^T eta n = 1", np.max(np.abs(np.einsum("...a,ab,...b->...", u, ETA, u) + 1)) < 1e-10 and np.max(np.abs(np.einsum("...a,ab,...b->...", nn, ETA, nn) - 1)) < 1e-10,
          [float(np.max(np.abs(np.einsum("...a,ab,...b->...", u, ETA, u) + 1))), float(np.max(np.abs(np.einsum("...a,ab,...b->...", nn, ETA, nn) - 1)))])
    check("u, n are eigenvectors of N (-g and 1 eigenvalues)", np.max(np.abs(fr["N"] @ u[..., None] - fr["lg"][..., None, None] * u[..., None])) < 1e-9 and np.max(np.abs(fr["N"] @ nn[..., None] - fr["l1"][..., None, None] * nn[..., None])) < 1e-9,
          [float(np.max(np.abs(fr["N"] @ u[..., None] - fr["lg"][..., None, None] * u[..., None]))), float(np.max(np.abs(fr["N"] @ nn[..., None] - fr["l1"][..., None, None] * nn[..., None])))])
    check("J^2 = -P23", np.max(np.abs(J @ J + P23)) < 1e-9, float(np.max(np.abs(J @ J + P23))))
    check("J eta-antisymmetric (J^T eta + eta J = 0), J u = J n = 0", np.max(np.abs(np.swapaxes(J, -1, -2) @ ETA + ETA @ J)) < 1e-12 and np.max(np.abs(J @ u[..., None])) < 1e-12 and np.max(np.abs(J @ nn[..., None])) < 1e-12,
          float(np.max(np.abs(np.swapaxes(J, -1, -2) @ ETA + ETA @ J))))
    R = rot_R(J, 0.7)
    check("R(beta) Lorentz: R^T eta R = eta", np.max(np.abs(np.swapaxes(R, -1, -2) @ ETA @ R - ETA)) < 1e-12, float(np.max(np.abs(np.swapaxes(R, -1, -2) @ ETA @ R - ETA))))
    check("G == EXT.h_cov_np (the author's H from the timelike eigenvector)", np.max(np.abs(fr["G"] - EXT.h_cov_np(M))) < 1e-9, float(np.max(np.abs(fr["G"] - EXT.h_cov_np(M)))))
    Rpi = rot_R(J, np.pi)
    check("T_2pi = id pointwise (R(pi) M R(pi)^T = M)", np.max(np.abs(Rpi @ M @ np.swapaxes(Rpi, -1, -2) - M)) < 1e-9, float(np.max(np.abs(Rpi @ M @ np.swapaxes(Rpi, -1, -2) - M))))
    # 2. the weight
    wN = fr["w"]
    frup = frame(Mup)
    check("w(N) = P23 where lambda_1 >= 1", np.max(np.abs(frup["w"] - frup["P23"])) < 1e-12, float(np.max(np.abs(frup["w"] - frup["P23"]))))
    tap = np.real(fr["l1"]) < 1.0
    check("w^2 != w in the taper (a weight, not a projector)", np.max(np.abs((wN @ wN - wN)[tap])) > 1e-3, float(np.max(np.abs((wN @ wN - wN)[tap]))))
    wref, lamref = spectral_reference_w(M)
    check("restricted w(N) == the general spectral function (eig-based, label-free)", np.max(np.abs(wN - wref)) < 1e-9, float(np.max(np.abs(wN - wref))))
    A = sym(rng.normal(size=M.shape))
    Om = wN @ (A @ ETA) @ wN
    OmT = np.swapaxes(Om, -1, -2)
    Hm, Hi = fr["G"], ETA + 2.0 * u[..., :, None] * u[..., None, :]
    hf, ef = _tr(OmT @ Hm @ Om @ Hi), _tr(OmT @ ETA @ Om @ ETA)
    check("H-form == eta-form for the plateau weight (w u = 0)", np.max(np.abs(hf - ef)) < 1e-9 * max(1.0, float(np.max(np.abs(ef)))), float(np.max(np.abs(hf - ef))))
    Bu = boost_of(u)
    Gh = ETA @ Bu @ ETA
    check("G^(1/2) = eta B_u eta (the pure boost): G^(1/2) G^(1/2) = G", np.max(np.abs(Gh @ Gh - fr["G"])) < 1e-10, float(np.max(np.abs(Gh @ Gh - fr["G"]))))
    gen = general_weight(M, u)
    check("general (label-free) w(N) == the restricted form where the pair is inside the plateau", np.max(np.abs(gen["w"] - wN)) < 1e-9, float(np.max(np.abs(gen["w"] - wN))))
    Mout = random_spectral_field(rng, n, cfg, dir_noise=(-0.05, 0.05), pair_noise=0.3, pair_offset=(0.4, -0.4))     # the pair partly outside the plateau
    frout = frame(Mout)
    res["domain_pair_out"] = domain(frout)
    check("a field with the pair partly outside the plateau exists for the gate", res["domain_pair_out"]["cells_pair_outside_plateau"] > 0, res["domain_pair_out"]["cells_pair_outside_plateau"])
    wref_out, _ = spectral_reference_w(Mout)
    check("general w(N) == the eig-based reference on the pair-outside field", np.max(np.abs(frout["w"] - wref_out)) < 1e-8, float(np.max(np.abs(frout["w"] - wref_out))))
    # 3. rho^2 = s^2 on the diagonal sheet; vacuum facts
    sv = 0.15
    Md = np.broadcast_to(np.diag([G, 1.0, DELTA + sv, DELTA - sv]), M.shape).copy()
    spl, _ = C15.split_cells(Md, need_grad=False)
    check("rho^2 = s^2 on the diagonal sheet", np.max(np.abs(spl / 4.0 - sv * sv)) < 1e-12, float(np.max(np.abs(spl / 4.0 - sv * sv))))
    Mv = np.broadcast_to(INS4.vac4(cfg), M.shape).copy()
    rv = averaged(Mv, cfg, a0_of(Mv), need_grad=True)
    check("vacuum: every part 0 (roundoff of the degenerate split), a0 = 0, gradient 0", all(abs(v) < 1e-11 for v in rv["parts"].values()) and np.max(np.abs(a0_of(Mv))) < 1e-12 and np.max(np.abs(rv["grad_stat"])) < 1e-12,
          {k: float(v) for k, v in rv["parts"].items()})
    # 4. reads against the registry and R15
    for c in ("norm", "rebuild"):
        r = action(M, cfgs[c], need_grad=False)["parts"]
        if c == "norm":
            ih = C15.i1h_static(M, cfg)
            check("E_h (I_norm) == 4 x the registry I1_h read", abs(r["E_h"] - ih) < 1e-9 * abs(ih), [r["E_h"], ih])
        res[f"parts_random_{c}"] = r
    rup = action(Mup, cfg, need_grad=False)["parts"]
    cfg15 = C15.cfg_dd(n, L, mu=0.0, cP=1.0)
    kp15 = C15.kp23_energy_grad(Mup, cfg15, need_grad=False)[0]
    check("K_P^proj == K_P^23 where lambda_1 >= 1", abs(rup["KP"] - kp15) < 1e-10 * abs(kp15), [rup["KP"], kp15])
    a0 = a0_of(M)
    a0c = C13.a0_local(M)
    Me0 = random_spectral_field(rng, n, cfg, boost=0.0)
    a0e, a0ce = a0_of(Me0), C13.a0_local(Me0)
    check("a0 == R15 a0_local up to the per-cell lift sign (u = e_0 field)", np.max(np.minimum(np.max(np.abs(a0e - a0ce), axis=(-1, -2)), np.max(np.abs(a0e + a0ce), axis=(-1, -2)))) < 1e-9,
          float(np.max(np.minimum(np.max(np.abs(a0e - a0ce), axis=(-1, -2)), np.max(np.abs(a0e + a0ce), axis=(-1, -2))))))
    # 5. gradient gates, plain action, each part (rebuild), then norm total, then kinetic
    D = sym(rng.normal(size=M.shape))
    nref = fr["n"]
    for c in ("rebuild", "norm"):
        cf = cfgs[c]
        gate(f"plain E_stat ({c}, director in the taper)", lambda X: action(X, cf, need_grad=False, n_ref=nref)["parts"]["E_stat"], lambda X: action(X, cf, n_ref=nref)["grad_stat"], M, D)
    zero = dict(cfg); zero["mu"], zero["cP"], zero["cs"] = 0.0, 0.0, 0.0
    gate("plain E_h + V4 alone (rebuild)", lambda X: action(X, zero, need_grad=False, n_ref=nref)["parts"]["E_stat"], lambda X: action(X, zero, n_ref=nref)["grad_stat"], M, D)
    # the parts individually: zero the other coefficients and compare E_stat (E_h and V4 have no coefficient: use the full minus)
    for key in ("U", "KP", "reg"):
        one = dict(cfg)
        one["mu"], one["cP"], one["cs"] = (cfg["mu"] if key == "U" else 0.0), (cfg["cP"] if key == "KP" else 0.0), (cfg["cs"] if key == "reg" else 0.0)
        gate(f"plain part {key} (+ E_h + V4)", lambda X, o=one: action(X, o, need_grad=False, n_ref=nref)["parts"]["E_stat"], lambda X, o=one: action(X, o, n_ref=nref)["grad_stat"], M, D)
    Dout = sym(rng.normal(size=M.shape))
    nrefo = frout["n"]
    go = action(Mout, cfg, n_ref=nrefo)["grad_stat"]
    ano = float(np.sum(go * Dout))
    epso = 1e-5
    fdo = (action(Mout + epso * Dout, cfg, need_grad=False, n_ref=nrefo)["parts"]["E_stat"] - action(Mout - epso * Dout, cfg, need_grad=False, n_ref=nrefo)["parts"]["E_stat"]) / (2 * epso)
    check("gradient gate (central differences) with the pair OUTSIDE the plateau (the Daleckii-Krein adjoint)", abs(ano - fdo) < 1e-6 * abs(fdo), {"analytic": ano, "fd": fdo, "rel": abs(ano - fdo) / abs(fdo)})
    gate("plain kin_tot (frozen a0, rebuild)", lambda X: action(X, cfg, a0, need_grad=False, n_ref=nref)["parts"]["kin_tot"], lambda X: action(X, cfg, a0, n_ref=nref)["grad_kin"], M, D)
    gate("plain kin_tot (frozen a0, norm)", lambda X: action(X, cfgs["norm"], a0, need_grad=False, n_ref=nref)["parts"]["kin_tot"], lambda X: action(X, cfgs["norm"], a0, n_ref=nref)["grad_kin"], M, D)
    Da = sym(rng.normal(size=M.shape))
    ga = action(M, cfg, a0, need_a0grad=True, n_ref=nref)["grad_a0"]
    an = float(np.sum(ga * Da))
    eps = 1e-5
    fd = (action(M, cfg, a0 + eps * Da, need_grad=False, n_ref=nref)["parts"]["kin_tot"] - action(M, cfg, a0 - eps * Da, need_grad=False, n_ref=nref)["parts"]["kin_tot"]) / (2 * eps)
    check("gradient gate d kin / d a0 (plain)", abs(an - fd) < 1e-7 * abs(fd), {"analytic": an, "fd": fd})
    # 6. the averaged action: gates, doubling, symmetry defect, covariance
    for c in ("rebuild", "norm"):
        cf = cfgs[c]
        gate(f"averaged E_stat ({c})", lambda X: averaged(X, cf, need_grad=False, n_ref=nref)["parts"]["E_stat"], lambda X: averaged(X, cf, n_ref=nref)["grad_stat"], M, D)
    gate("averaged kin_tot (frozen a0, rebuild)", lambda X: averaged(X, cfg, a0, need_grad=False, n_ref=nref)["parts"]["kin_tot"], lambda X: averaged(X, cfg, a0, n_ref=nref)["grad_kin"], M, D)
    gate("fixed-K E_K (K = 50, rebuild, a0 frozen: the protocol's gradient)", lambda X: energy_and_grad(X, cfg, 50.0, nref, need_grad=False, a0_frozen=a0)[0], lambda X: energy_and_grad(X, cfg, 50.0, nref, a0_frozen=a0)[1], M, D)
    gT = energy_and_grad(M, cfg, 50.0, nref)[1]
    epsT = 1e-5
    fdT = (energy_and_grad(M + epsT * D, cfg, 50.0, nref, need_grad=False)[0] - energy_and_grad(M - epsT * D, cfg, 50.0, nref, need_grad=False)[0]) / (2 * epsT)
    res["frozen_vs_true_EK_derivative"] = {"frozen_a0_analytic": float(np.sum(gT * D)), "true_fd": fdT}
    log(f"fixed-K: the frozen-a0 directional derivative {float(np.sum(gT * D)):.6f} vs the true E_K derivative (a0 refreshed) {fdT:.6f} (the R15 protocol; the stationarity read uses the true one)")
    dbl = {}
    for c in ("rebuild", "norm"):
        p2 = averaged(M, cfgs[c], a0, need_grad=False, n_samples=2)["parts"]
        p4 = averaged(M, cfgs[c], a0, need_grad=False, n_samples=4)["parts"]
        p8 = averaged(M, cfgs[c], a0, need_grad=False, n_samples=8)["parts"]
        dbl[c] = {k: [abs(p2[k] - p4[k]) / max(abs(p4[k]), 1e-300), abs(p4[k] - p8[k]) / max(abs(p8[k]), 1e-300)] for k in p2}
    worst24 = max(dbl[c][k][0] for c in dbl for k in dbl[c])
    worst48 = max(dbl[c][k][1] for c in dbl for k in dbl[c])
    check("doubling test 4 -> 8 samples on every part (static + kinetic), both completions", worst48 < 1e-11, {"worst_rel_4_8": worst48, "worst_rel_2_4": worst24, "per_part": dbl})
    check("the degree-2 component: the 2-sample average is NOT exact on a general field (recorded, not a failure of the instrument)", worst24 > 1e-9, worst24)
    # the symmetry-defect gate: E_avg(T_beta M) == E_avg(M) for generic beta; the unaveraged regulator must fail
    defects = {}
    for beta in (0.4, 1.1, 2.0):
        Rb = rot_R(fr["J"], beta)
        Mb = Rb @ M @ np.swapaxes(Rb, -1, -2)
        pa, pb = averaged(M, cfg, a0, need_grad=False, n_ref=nref)["parts"], averaged(Mb, cfg, a0_of(Mb, n_ref=nref), need_grad=False, n_ref=nref)["parts"]
        ra, rb = action(M, cfg, a0, need_grad=False, n_ref=nref)["parts"], action(Mb, cfg, a0_of(Mb, n_ref=nref), need_grad=False, n_ref=nref)["parts"]
        defects[str(beta)] = {"averaged": {k: abs(pa[k] - pb[k]) / max(abs(pa[k]), 1e-300) for k in pa},
                              "unaveraged": {k: abs(ra[k] - rb[k]) / max(abs(ra[k]), 1e-300) for k in ra}}
    worst_avg = max(v for b in defects for v in defects[b]["averaged"].values())
    ctrl = min(defects[b]["unaveraged"]["reg"] for b in defects)
    ctrl_h = min(defects[b]["unaveraged"]["E_h"] for b in defects)
    inv_pot = max(max(defects[b]["unaveraged"]["V4"], defects[b]["unaveraged"]["U"]) for b in defects)
    check("symmetry-defect gate: the averaged action invariant under T_beta (every part, 1e-10)", worst_avg < 1e-10, {"worst_rel": worst_avg, "defects": defects})
    check("control: the UNAVERAGED regulator rho^2 E2 FAILS the gate (and the unaveraged quartic too)", ctrl > 1e-4 and ctrl_h > 1e-4, {"reg_min_defect": ctrl, "E_h_min_defect": ctrl_h})
    check("the potential and rho^2 are exactly circle-invariant unaveraged (roundoff)", inv_pot < 1e-10, inv_pot)
    # covariance under global Lorentz maps (the lift reference transformed along)
    def lorentz(kind):
        if kind == "boost":
            Km = np.zeros((4, 4)); Km[0, 2] = Km[2, 0] = 1.0
            b = 0.3
            return EYE + np.sinh(b) * Km + (np.cosh(b) - 1.0) * (Km @ Km)
        Gm = np.zeros((4, 4)); Gm[1, 3], Gm[3, 1] = -1.0, 1.0
        q = 0.7
        return EYE + np.sin(q) * Gm + (1 - np.cos(q)) * (Gm @ Gm)
    for kind in ("boost", "rotation"):
        Lm = lorentz(kind)
        ML = Lm @ M @ Lm.T
        nrefL = np.einsum("ab,...b->...a", Lm, nref)
        pa = averaged(M, cfg, a0_of(M, n_ref=nref), need_grad=False, n_ref=nref)["parts"]
        pb = averaged(ML, cfg, a0_of(ML, n_ref=nrefL), need_grad=False, n_ref=nrefL)["parts"]
        worst_c = max(abs(pa[k] - pb[k]) / max(abs(pa[k]), 1e-300) for k in pa)
        check(f"covariance of the averaged action under a global {kind} (every part)", worst_c < 1e-9, {"worst_rel": worst_c, "parts": pa})
    Lm = lorentz("boost")
    ML = Lm @ M @ Lm.T
    frob0 = float(np.sum(np.array([np.sum(INS4.d1(M, ax, cfg["h"], "fwd") ** 2) for ax in range(3)])))
    frob1 = float(np.sum(np.array([np.sum(INS4.d1(ML, ax, cfg["h"], "fwd") ** 2) for ax in range(3)])))
    check("no-eta control (plain Frobenius gradient norm) FAILS covariance under the boost", abs(frob1 - frob0) > 1e-3 * abs(frob0), [frob0, frob1])
    # 7. the lift: the orientation propagates (n . n_ref > 0 after frame with n_ref)
    check("the lift: n . n_ref > 0 on every cell", bool(np.min(np.einsum("...a,...a->...", frame(M, nref)["n"], nref)) > 0), float(np.min(np.einsum("...a,...a->...", frame(M, nref)["n"], nref))))
    # 8. timing at n32 (one averaged energy + gradient, static and fixed K, rebuild)
    cfg32 = cfg_v4(32, 48.0)
    M32, _ = load_r15_seed(32, 48)
    nref32 = radial_ref(cfg32)
    t = time.time(); r32 = averaged(M32, cfg32, need_grad=True, n_ref=nref32); t_stat = time.time() - t
    t = time.time(); energy_and_grad(M32, cfg32, 200.0, nref32); t_kin = time.time() - t
    p2s = averaged(M32, cfg32, a0_of(M32, n_ref=nref32), need_grad=False, n_ref=nref32, n_samples=2)["parts"]
    p4s = averaged(M32, cfg32, a0_of(M32, n_ref=nref32), need_grad=False, n_ref=nref32, n_samples=4)["parts"]
    p8s = averaged(M32, cfg32, a0_of(M32, n_ref=nref32), need_grad=False, n_ref=nref32, n_samples=8)["parts"]
    dbl_seed = {k: [abs(p2s[k] - p4s[k]) / max(abs(p4s[k]), 1e-300), abs(p4s[k] - p8s[k]) / max(abs(p8s[k]), 1e-300)] for k in p2s}
    res["timing_n32"] = {"averaged_static_s": t_stat, "fixed_K_s": t_kin, "seed_parts": r32["parts"], "seed_domain": r32["domain"], "seed_doubling_2_4_and_4_8": dbl_seed}
    log(f"seed doubling (2->4, 4->8) per part: {dbl_seed}")
    log(f"n32 timing: averaged static energy + gradient {t_stat:.2f} s, fixed-K {t_kin:.2f} s; seed parts {r32['parts']}; seed domain {r32['domain']}")
    res["n_pass"] = sum(1 for v in res.values() if isinstance(v, dict) and v.get("ok"))
    res["n_total"] = sum(1 for v in res.values() if isinstance(v, dict) and "ok" in v)
    log(f"selftest {res['n_pass']}/{res['n_total']}")
    if write:
        json.dump(res, open(os.path.join(DATA, "m5_32_r16_common_selftest.json"), "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    selftest()
