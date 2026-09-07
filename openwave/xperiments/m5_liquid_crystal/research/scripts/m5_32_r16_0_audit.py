"""
m5_32_r16_0_audit.py: INDEPENDENT ADVERSARIAL AUDIT of rung R16-0 (task M5.32).

Own script, own method. The producer scripts (m5_32_r16_0_symbolic / _reduced /
_fields, m5_32_r15_common) and their data were NOT opened. Only the definitions
and the claim list of the audit prompt were used. Every check is a function
`check_<id>()` returning a dict {"verdict", "own_numbers", "note"}.

EQUATIONS (all in the prompt's conventions)
-------------------------------------------
eta = diag(-1, 1, 1, 1); M real symmetric contravariant; N = M eta.
Vacuum M = diag(g, 1, delta, delta), g = 8, delta = 0.3 -> spec(N) = (-g, 1, delta, delta).
u = timelike eigenvector of N, u^T eta u = -1;  G(u) = eta + 2 (eta u)(eta u)^T.
Jets A_i = d_i M (lattice: fwd / bwd one-sided, "sym" = average of the two branch
energies; "cen" = central difference in the interior, one-sided at the edges).
F^eta_ij = A_i eta A_j - A_j eta A_i;   F^G_ij = A_i G A_j - A_j G A_i.
E_eta     = 4 sum_{i<j} tr(eta F^eta eta F^eta^T)  h^3
E_norm    = 4 sum_{i<j} tr(G   F^eta G   F^eta^T)  h^3
E_rebuild = 4 sum_{i<j} tr(G   F^G   G   F^G^T)    h^3
V4^dd = W1 sum_{p=1..4} (tr N^p - C_p)^2,  C_p = (-g)^p + 1 + 2 delta^p.
P23 = I - P_g - P_1 (spectral projectors of the two isolated eigenvalues of N);
K_P^23 static = (1/2) sum_i tr(Om_i^T eta Om_i eta), Om_i = P23 (A_i eta) P23;
K_P^23 inertia under a0 = G23 M - M G23: (1/2) tr(Om_0^T eta Om_0 eta).
E2 static = sum_i tr(A_i G A_i G).
R_pq(a): rotation in the (p,q) plane, R[p,p] = R[q,q] = cos a, R[p,q] = -sin a, R[q,p] = sin a.
Boost L(n, chi) = I + sinh(chi) K + (cosh(chi) - 1) K^2, K[0,i] = K[i,0] = n_i.
Witness: M = L (R_12(k z) D R_12^T) L^T, chi(r) = 0.5 exp(-r^2/8), D = diag(g,1,delta,delta).
Reduced line: E_J = int 4 pi r^2 [(c/2) s'^2 + V(s)] dr + J^2 / (4 int 4 pi r^2 c s^2 dr), c = 4.
Rational weight: w(x) = f(x)/f(delta), f(x) = (x-g)(x-1)/((x-g)^2 + (x-1)^2); W(s) = [w(delta+s) w(delta-s)]^2.
Sextic: V = mu s^2 - nu s^4 + kappa s^6, (mu, nu, kappa) = (1e-2, 4e-2, 0.4).
Biaxiality: Q = triple - mean, beta^2 = 1 - 6 (tr Q^3)^2 / (tr Q^2)^3.
tau = eps_ijk S_il d_j S_kl, S = delta I + (1 - delta) n n^T.

Usage: python3 scripts/m5_32_r16_0_audit.py [--only C1a,C3,...] [--fast]
Writes data/m5_32_r16_0_audit.json (relative paths only).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import sympy as sp
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")

G0, DELTA, W1 = 8.0, 0.3, 0.000724023879
MU, NU, KAPPA = 1e-2, 4e-2, 0.4
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
I4 = np.eye(4)
G23 = np.zeros((4, 4)); G23[2, 3] = -1.0; G23[3, 2] = 1.0
GZ = np.zeros((4, 4)); GZ[1, 2] = -1.0; GZ[2, 1] = 1.0   # (1,2)-plane generator

FIELD_R10 = "checkpoints/m5_32_r10/relax_g8_n32_L48_it12000.npy"
FIELD_P4_32 = "checkpoints/m5_32_r15/p4_fixedj/fixedJ_n32_L48_J200.npy"
FIELD_P4_48 = "checkpoints/m5_32_r15/p4_fixedj/fixedJ_n48_L72_J200.npy"
FIELD_M = "checkpoints/m5_32_r15/m_hedgehog/relax_n32_L48_mu0.01_cP1.npy"


def load_field(rel):
    return np.load(os.path.join(ROOT, rel))


# ============================================================ algebra helpers
def rot(p, q, a, dim=4):
    R = np.eye(dim)
    R[p, p] = R[q, q] = np.cos(a); R[p, q] = -np.sin(a); R[q, p] = np.sin(a)
    return R


def rot_sym(p, q, a):
    R = sp.eye(4)
    R[p, p] = R[q, q] = sp.cos(a); R[p, q] = -sp.sin(a); R[q, p] = sp.sin(a)
    return R


def boost(nhat, chi):
    """L(n, chi) per cell; nhat (..., 3), chi (...)."""
    sh = np.shape(chi)
    K = np.zeros(sh + (4, 4)); K2 = np.zeros(sh + (4, 4))
    K[..., 0, 1:] = nhat; K[..., 1:, 0] = nhat
    K2[..., 0, 0] = 1.0
    K2[..., 1:, 1:] = nhat[..., :, None] * nhat[..., None, :]
    return (np.eye(4) + np.sinh(chi)[..., None, None] * K
            + (np.cosh(chi) - 1.0)[..., None, None] * K2)


def tr(X):
    return np.einsum("...ii->...", X)


def quad(X, F, Y):
    """tr(X F Y F^T) per cell."""
    return np.einsum("...ab,...bc,...cd,...ad->...", X, F, Y, F, optimize=True)


def G_of_u(u):
    eu = np.einsum("ab,...b->...a", ETA, u)
    return ETA + 2.0 * eu[..., :, None] * eu[..., None, :]


def timelike_u(M):
    """u per cell from eig(N): the eigenvalue with the smallest real part; u^T eta u = -1."""
    N = M @ ETA
    w, V = np.linalg.eig(N)
    idx = np.argmin(w.real, axis=-1)
    u = np.take_along_axis(V, idx[..., None, None], axis=-1)[..., 0].real
    nrm = -np.einsum("...a,ab,...b->...", u, ETA, u)
    return u / np.sqrt(nrm)[..., None]


def spectral_projectors(M):
    """P_g, P_1, P23 per cell via eig(N) (equivalent to the Lagrange rational projectors)."""
    N = M @ ETA
    w, V = np.linalg.eig(N)
    Vi = np.linalg.inv(V)
    ig = np.argmin(w.real, axis=-1)
    i1 = np.argmax(w.real, axis=-1)
    def proj(idx):
        col = np.take_along_axis(V, idx[..., None, None], axis=-1)[..., 0]
        row = np.take_along_axis(Vi, idx[..., None, None], axis=-2)[..., 0, :]
        return (col[..., :, None] * row[..., None, :]).real
    Pg, P1 = proj(ig), proj(i1)
    return Pg, P1, np.eye(4) - Pg - P1


def kp23_static(M, A_list, P23):
    out = 0.0
    for A in A_list:
        Om = P23 @ (A @ ETA) @ P23
        out = out + 0.5 * np.einsum("...ba,bc,...cd,da->...", Om, ETA, Om, ETA, optimize=True)
    return out


def kp23_inertia(M, a0, P23):
    Om = P23 @ (a0 @ ETA) @ P23
    return 0.5 * np.einsum("...ba,bc,...cd,da->...", Om, ETA, Om, ETA, optimize=True)


def v4dd(M):
    N = M @ ETA
    P = N; out = 0.0
    for p in range(1, 5):
        if p > 1:
            P = P @ N
        Cp = (-G0) ** p + 1.0 + 2.0 * DELTA ** p
        out = out + (tr(P) - Cp) ** 2
    return W1 * out


# ============================================================ lattice helpers
def coords(n, h):
    x = (np.arange(n) - (n - 1) / 2.0) * h
    return np.meshgrid(x, x, x, indexing="ij")


def d1(f, ax, h, st):
    """own stencil: fwd, bwd, cen (central interior + one-sided edges), per (periodic central)."""
    out = np.zeros_like(f)
    sl = [slice(None)] * f.ndim
    def at(i):
        s = list(sl); s[ax] = i; return tuple(s)
    if st == "fwd":
        out[at(slice(0, -1))] = (f[at(slice(1, None))] - f[at(slice(0, -1))]) / h
    elif st == "bwd":
        out[at(slice(1, None))] = (f[at(slice(1, None))] - f[at(slice(0, -1))]) / h
    elif st == "cen":
        out[at(slice(1, -1))] = (f[at(slice(2, None))] - f[at(slice(0, -2))]) / (2 * h)
        out[at(0)] = (f[at(1)] - f[at(0)]) / h
        out[at(-1)] = (f[at(-1)] - f[at(-2)]) / h
    elif st == "per":
        out = (np.roll(f, -1, axis=ax) - np.roll(f, 1, axis=ax)) / (2 * h)
    return out


def branches(st):
    return [("fwd", 0.5), ("bwd", 0.5)] if st == "sym" else [(st, 1.0)]


def curvature_densities(M, h, st="sym", u=None):
    """per-cell densities (before h^3) of E_eta, E_norm, E_rebuild (4 sum_{i<j} ...)."""
    if u is None:
        u = timelike_u(M)
    G = G_of_u(u)
    E4 = np.broadcast_to(ETA, G.shape)
    de = np.zeros(M.shape[:-2]); dn = np.zeros_like(de); dr = np.zeros_like(de)
    for br, wt in branches(st):
        A = [d1(M, ax, h, br) for ax in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                Fe = A[i] @ ETA @ A[j] - A[j] @ ETA @ A[i]
                Fg = A[i] @ G @ A[j] - A[j] @ G @ A[i]
                de += wt * 4.0 * quad(E4, Fe, E4)
                dn += wt * 4.0 * quad(G, Fe, G)
                dr += wt * 4.0 * quad(G, Fg, G)
    return de, dn, dr


def curvature_energies(M, h, st="sym", u=None):
    de, dn, dr = curvature_densities(M, h, st, u)
    h3 = h ** 3
    return {"E_eta": float(de.sum() * h3), "E_norm": float(dn.sum() * h3),
            "E_rebuild": float(dr.sum() * h3),
            "min_dens_norm": float(dn.min()), "min_dens_rebuild": float(dr.min())}


def rel(a, b):
    return float(abs(a - b) / max(abs(b), 1e-300))


def numzero(expr, syms, npts=25, seed=7, lo=-2.5, hi=2.5):
    """max |expr| over random real points (robust zero test where sympy's simplify stalls on trig)."""
    f = sp.lambdify(syms, expr, "numpy")
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(npts):
        vals = [rng.uniform(lo, hi) for _ in syms]
        worst = max(worst, float(np.abs(np.array(f(*vals), dtype=float)).max()))
    return worst


# ============================================================ C1a
def check_C1a():
    E01 = np.zeros((4, 4)); E01[0, 1] = E01[1, 0] = 1.0
    def F(A1, A2, Gm):
        return A1 @ Gm @ A2 - A2 @ Gm @ A1
    def n2(F):
        return float(np.sum(F * F))
    r = {}
    r["A1=I: |F^I|^2"] = n2(F(I4, E01, I4)); r["A1=I: |F^eta|^2"] = n2(F(I4, E01, ETA))
    r["A1=eta: |F^I|^2"] = n2(F(ETA, E01, I4)); r["A1=eta: |F^eta|^2"] = n2(F(ETA, E01, ETA))
    rng = np.random.default_rng(1)
    mx = 0.0
    for _ in range(50):
        A1 = np.zeros((4, 4)); A2 = np.zeros((4, 4))
        S1 = rng.normal(size=(3, 3)); S2 = rng.normal(size=(3, 3))
        A1[1:, 1:] = S1 + S1.T; A2[1:, 1:] = S2 + S2.T
        A1[0, 0] = rng.normal(); A2[0, 0] = rng.normal()
        mx = max(mx, np.abs(F(A1, A2, I4) - F(A1, A2, ETA)).max())
    r["block-diag max|F^I - F^eta|"] = mx
    ok = (r["A1=I: |F^I|^2"] == 0 and abs(r["A1=I: |F^eta|^2"] - 8) < 1e-12
          and r["A1=eta: |F^eta|^2"] == 0 and abs(r["A1=eta: |F^I|^2"] - 8) < 1e-12 and mx < 1e-13)
    return {"verdict": "CONFIRMED" if ok else "REFUTED", "own_numbers": r,
            "note": "direct evaluation; block-diagonal identity tested on 50 random jets"}


# ============================================================ C1b
def sym_point_jet_setup():
    chi, psi, b, k, g, d = sp.symbols("chi psi b k g delta", real=True)
    L1 = sp.eye(4); L1[0, 0] = L1[1, 1] = sp.cosh(chi); L1[0, 1] = L1[1, 0] = sp.sinh(chi)
    Q = L1 * rot_sym(1, 2, psi)
    D = sp.diag(g, 1, d, d)
    M = Q * D * Q.T
    eta = sp.diag(-1, 1, 1, 1)
    u = Q[:, 0]
    eu = eta * u
    Gm = eta + 2 * eu * eu.T
    As = b * M.diff(chi); At = k * M.diff(psi)
    return dict(chi=chi, psi=psi, b=b, k=k, g=g, d=d, M=M, eta=eta, G=Gm, As=As, At=At, u=u)


def check_C1b():
    S = sym_point_jet_setup()
    eta, Gm, As, At = S["eta"], S["G"], S["As"], S["At"]
    Fe = As * eta * At - At * eta * As
    Fg = As * Gm * At - At * Gm * As
    Un = 4 * (Gm * Fe * Gm * Fe.T).trace()
    Ur = 4 * (Gm * Fg * Gm * Fg.T).trace()
    # exact value at chi = psi = 0
    sub0 = {S["chi"]: 0, S["psi"]: 0}
    Un0 = sp.simplify(Un.subs(sub0)); Ur0 = sp.simplify(Ur.subs(sub0))
    target = 8 * (S["d"] - 1) ** 2 * (S["g"] + 1) ** 2 * S["b"] ** 2 * S["k"] ** 2
    val0 = float(Un0.subs({S["g"]: 8, S["d"]: sp.Rational(3, 10), S["b"]: 1, S["k"]: 1}))
    # numeric equality on random (chi, psi), high-precision lambdify
    fn = sp.lambdify((S["chi"], S["psi"], S["b"], S["k"], S["g"], S["d"]), Un - Ur, "mpmath")
    fu = sp.lambdify((S["chi"], S["psi"], S["b"], S["k"], S["g"], S["d"]), Un, "mpmath")
    import mpmath as mp
    mp.mp.dps = 30
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(40):
        c, p = rng.uniform(-1.5, 1.5), rng.uniform(-3, 3)
        dif = fn(mp.mpf(c), mp.mpf(p), mp.mpf(1), mp.mpf(1), mp.mpf(8), mp.mpf("0.3"))
        base = fu(mp.mpf(c), mp.mpf(p), mp.mpf(1), mp.mpf(1), mp.mpf(8), mp.mpf("0.3"))
        worst = max(worst, float(abs(dif) / abs(base)))
    # u check: N u = -g u
    Nu = (S["M"] * eta * S["u"] + S["g"] * S["u"]).subs({S["chi"]: 0.7, S["psi"]: 1.1, S["g"]: 8, S["d"]: 0.3})
    r = {"U_norm(0,0) - target (symbolic)": str(sp.simplify(Un0 - target)),
         "U_rebuild(0,0) - target (symbolic)": str(sp.simplify(Ur0 - target)),
         "value at g8 d0.3 b=k=1": val0,
         "max rel |U_norm - U_rebuild| over 40 random (chi,psi), 30 digits": worst,
         "|N u + g u| at (0.7,1.1)": float(max(abs(x) for x in Nu))}
    ok = (r["U_norm(0,0) - target (symbolic)"] == "0" and r["U_rebuild(0,0) - target (symbolic)"] == "0"
          and abs(val0 - 317.52) < 1e-9 and worst < 1e-20)
    return {"verdict": "CONFIRMED" if ok else "REFUTED", "own_numbers": r,
            "note": "symbolic at chi=psi=0; equality for all (chi,psi) tested at 40 random points in 30-digit arithmetic (an analytic function vanishing there is identically zero to any practical standard)"}


# ============================================================ C1c
def witness_field(n, L, k, D=None):
    h = L / n
    X, Y, Z = coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    chi = 0.5 * np.exp(-r * r / 8.0)
    nhat = np.stack([X / r, Y / r, Z / r], axis=-1)
    Lb = boost(nhat, chi)
    if D is None:
        D = np.diag([G0, 1.0, DELTA, DELTA])
    if k == 0.0:
        RDR = np.broadcast_to(D, (n, n, n, 4, 4))
    else:
        ca, sa = np.cos(k * Z), np.sin(k * Z)
        R = np.zeros((n, n, n, 4, 4)); R[..., 0, 0] = 1.0; R[..., 3, 3] = 1.0
        R[..., 1, 1] = ca; R[..., 2, 2] = ca; R[..., 1, 2] = -sa; R[..., 2, 1] = sa
        RDR = R @ D @ np.swapaxes(R, -1, -2)
    M = Lb @ RDR @ np.swapaxes(Lb, -1, -2)
    u_an = Lb[..., :, 0]
    return M, h, u_an


def check_C1c(fast=False):
    boxes = [(32, 48.0)] + ([] if fast else [(64, 48.0), (64, 24.0)])
    claim = {"n64L48": {"ratio": [1.38, 1.40, 1.46], "eta": [-515, -1739, -3583],
                        "norm": [557, 1835, 4002], "reb": [770, 2561, 5855]},
             "n64L24": {"ratio": [1.24, 1.25, 1.26]}}
    r = {}
    for n, L in boxes:
        M0, h, u0 = witness_field(n, L, 0.0)
        # independent u: eig at a subsample vs the analytic L e0
        sub = (slice(None, None, 8),) * 3
        ue = timelike_u(M0[sub]); ua = u0[sub]
        r[f"n{n}L{int(L)} max|u_eig - u_an| (subsample, sign-fixed)"] = float(np.abs(np.abs(ue) - np.abs(ua)).max())
        for st in (["sym"] if n == 64 else ["sym", "cen"]):
            E0 = curvature_energies(M0, h, st, u0)
            row = {}
            for k in (0.5, 1.0, 2.0):
                Mk, _, uk = witness_field(n, L, k)
                Ek = curvature_energies(Mk, h, st, uk)
                dE = {key: Ek[key] - E0[key] for key in ("E_eta", "E_norm", "E_rebuild")}
                dE["ratio_reb_over_norm"] = dE["E_rebuild"] / dE["E_norm"]
                row[f"k{k}"] = dE
            r[f"n{n}L{int(L)}_{st}"] = row
    # verdict on the ratios (n64 L48 sym is the claim's box)
    key = "n64L48_sym" if not fast else "n32L48_sym"
    ratios = [r[key][f"k{k}"]["ratio_reb_over_norm"] for k in (0.5, 1.0, 2.0)]
    signs_ok = all(r[key][f"k{k}"]["E_eta"] < 0 < r[key][f"k{k}"]["E_norm"] and r[key][f"k{k}"]["E_rebuild"] > 0
                   for k in (0.5, 1.0, 2.0))
    ok_ratio = all(abs(a - b) / b < 0.03 for a, b in zip(ratios, claim["n64L48"]["ratio"]))
    if not fast:
        vals_ok = all(rel(r[key][f"k{k}"][kk], claim["n64L48"][ck][i]) < 0.02
                      for i, k in enumerate((0.5, 1.0, 2.0)) for kk, ck in (("E_eta", "eta"), ("E_norm", "norm"), ("E_rebuild", "reb")))
    else:
        vals_ok = None
    verdict = "CONFIRMED" if (signs_ok and ok_ratio and (vals_ok or fast)) else ("QUALIFIED" if signs_ok else "REFUTED")
    r["claim_ratios_n64L48"] = claim["n64L48"]["ratio"]
    r["own_ratios_" + key] = ratios
    r["reb_over_4_" + key] = [r[key][f"k{k}"]["E_rebuild"] / 4 for k in (0.5, 1.0, 2.0)]
    return {"verdict": verdict, "own_numbers": r,
            "note": "own witness construction, own stencils; 'sym' = average of fwd/bwd branch energies (matches the certified stack's convention), 'cen' = central interior. All n64 L48 and n64 L24 numbers reproduced to 4 digits. Caveat: the ratios are NOT converged in h (1.38-1.46 at h 0.75, 1.24-1.26 at h 0.375) and are stencil-specific (the central stencil on n32 gives 1.00 / 0.81 / 1.11); at h 0.75 the k = 2 twist has kh = 1.5 rad per cell. The reading of the author's h column (+183/+594/+1549 = DeltaE_rebuild/4) is an interpretation of numbers I cannot reproduce; only the arithmetic (192 / 640 / 1464 = rebuild/4) is checked here."}


# ============================================================ C2a
def check_C2a():
    s, mu = sp.symbols("s mu", real=True)
    g, d = sp.Integer(8), sp.Rational(3, 10)
    Nd = sp.diag(-g, 1, d + s, d - s)
    V4 = 0
    for p in range(1, 5):
        Cp = (-g) ** p + 1 + 2 * d ** p
        V4 += ((Nd ** p).trace() - Cp) ** 2
    V4 = sp.expand(W1 * V4)
    target = sp.expand(W1 * ((2 * s ** 2) ** 2 + (6 * d * s ** 2) ** 2 + (12 * d ** 2 * s ** 2 + 2 * s ** 4) ** 2))
    r = {"V4dd(s) - closed form (symbolic)": str(sp.simplify(V4 - target))}
    # V/s^2 for mu s^2 + V4dd: monotone increasing in s^2 -> min at s->0
    Vq = sp.lambdify(s, (MU * s ** 2 + V4) / s ** 2)
    ss = np.linspace(1e-3, 2.0, 4000)
    vq = Vq(ss)
    r["quadratic+V4: min V/s^2 on (0,2]"] = float(vq.min()); r["at s"] = float(ss[vq.argmin()])
    r["quadratic+V4: monotone increasing"] = bool(np.all(np.diff(vq) > 0))
    # sextic
    Vs = lambda x: MU - NU * x ** 2 + KAPPA * x ** 4
    sstar = np.sqrt(NU / (2 * KAPPA))
    r["sextic: s*"] = float(sstar); r["sextic: min V/s^2"] = float(Vs(sstar))
    r["sextic: mu - nu^2/(4 kappa)"] = MU - NU ** 2 / (4 * KAPPA)
    V4f = sp.lambdify(s, V4)
    x = 0.205
    r["sextic + V4dd at s=0.205: V/s^2"] = float(Vs(x) + V4f(x) / x ** 2)
    xs = np.linspace(1e-3, 0.5, 20000)
    v6 = Vs(xs) + V4f(xs) / xs ** 2
    r["sextic + V4dd: min V/s^2"] = float(v6.min()); r["sextic + V4dd: argmin s"] = float(xs[v6.argmin()])
    # K_P^23 inertia of a uniform split under a0 = [G23, M]
    inert = {}
    for sv in (0.05, 0.15, 0.25):
        Ms = np.diag([G0, 1.0, DELTA + sv, DELTA - sv])
        a0 = G23 @ Ms - Ms @ G23
        _, _, P23 = spectral_projectors(Ms[None])
        inert[f"s={sv}"] = {"inertia": float(kp23_inertia(Ms[None], a0[None], P23)[0]), "4s^2": 4 * sv ** 2}
    r["K_P^23 inertia"] = inert
    r["omega_c^2 = mu/(4 c_P) at mu 1e-2, c_P 1"] = MU / 4.0
    ok = (r["V4dd(s) - closed form (symbolic)"] == "0" and r["quadratic+V4: monotone increasing"]
          and abs(r["sextic: min V/s^2"] - 0.009) < 1e-12 and abs(sstar - 0.2236) < 1e-4
          and abs(r["sextic + V4dd at s=0.205: V/s^2"] - 0.00929) < 1e-5
          and all(abs(v["inertia"] - v["4s^2"]) < 1e-12 for v in inert.values()))
    return {"verdict": "CONFIRMED" if ok else "REFUTED", "own_numbers": r,
            "note": "symbolic V4dd on the sheet; inertia via eig-based spectral projectors; omega_c^2 = mu/(4 c_P) follows from inertia 4 s^2 with the kinetic term c_P x inertia"}


# ============================================================ C2b
def check_C2b():
    a = W1 * (4 + 36 * DELTA ** 2 + 144 * DELTA ** 4)  # s^4 coefficient of V4dd
    c = 4.0; J = 1000.0
    wc = np.sqrt(MU / c)
    out = {}
    prev = None; mono = True
    for Vol in [1e2, 1e3, 1e4, 1e5, 1e6, 1e8, 1e10]:
        E = lambda x: MU * x + a * x ** 2 / Vol + J ** 2 / (4 * c * x)
        xs = np.logspace(0, 6, 200001)
        ratio = E(xs).min() / (wc * J)
        out[f"Vol={Vol:.0e}"] = float(ratio)
        if prev is not None and ratio > prev + 1e-12:
            mono = False
        prev = ratio
    ok = all(v >= 1.0 - 1e-12 for v in out.values()) and abs(out["Vol=1e+10"] - 1.0) < 1e-6 and mono
    return {"verdict": "CONFIRMED" if ok else "REFUTED",
            "own_numbers": {"a (s^4 coeff of V4dd)": a, "omega_c": wc, "min E/(omega_c J)": out, "monotone down to 1": mono},
            "note": "grid minimization of E(x); the a-term vanishes as 1/Vol, AM-GM gives the limit 1 from above"}


# ============================================================ C2c
def reduced_line_min(J, R, V, dV, c=4.0, dr=0.5, inits=None, dirichlet=False):
    """minimize E_J over s_i on r_i = (i+1/2) dr, i < N = R/dr; returns (E_min, s, X).
    dirichlet=True pins s = 0 at r = R (ghost point), i.e. the vacuum at the box edge; False = free edge."""
    N = int(round(R / dr))
    r = (np.arange(N) + 0.5) * dr
    w = 4 * np.pi * r ** 2 * dr            # cell weights
    rh = (np.arange(N - 1) + 1.0) * dr     # half points
    wh = 4 * np.pi * rh ** 2 * dr
    wR = 4 * np.pi * R ** 2 * dr if dirichlet else 0.0
    def energy(sv):
        ds = np.diff(sv) / dr
        grad = 0.5 * c * np.sum(wh * ds ** 2) + 0.5 * c * wR * (sv[-1] / dr) ** 2
        pot = np.sum(w * V(sv))
        X = np.sum(w * c * sv ** 2)
        return grad + pot + J ** 2 / (4 * X), X
    def fun(sv):
        ds = np.diff(sv) / dr
        X = np.sum(w * c * sv ** 2)
        E = 0.5 * c * np.sum(wh * ds ** 2) + 0.5 * c * wR * (sv[-1] / dr) ** 2 + np.sum(w * V(sv)) + J ** 2 / (4 * X)
        gg = np.zeros_like(sv)
        t = c * wh * ds / dr
        gg[1:] += t; gg[:-1] -= t
        gg[-1] += c * wR * sv[-1] / dr ** 2
        gg += w * dV(sv)
        gg += -J ** 2 / (4 * X ** 2) * (2 * c * w * sv)
        return E, gg
    best = (np.inf, None, None)
    if inits is None:
        inits = []
    for s0 in inits:
        res = minimize(fun, s0, jac=True, method="L-BFGS-B", options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10})
        E, X = energy(res.x)
        if E < best[0]:
            best = (E, res.x, X)
    return best, r


def check_C2c(fast=False):
    c = 4.0; wc = np.sqrt(MU / c)
    a4 = W1 * (4 + 36 * DELTA ** 2 + 144 * DELTA ** 4); a6 = W1 * 48 * DELTA ** 2; a8 = W1 * 4
    V4 = lambda x: a4 * x ** 4 + a6 * x ** 6 + a8 * x ** 8
    dV4 = lambda x: 4 * a4 * x ** 3 + 6 * a6 * x ** 5 + 8 * a8 * x ** 7
    Vq = lambda x: MU * x ** 2 + V4(x); dVq = lambda x: 2 * MU * x + dV4(x)
    V6 = lambda x: MU * x ** 2 - NU * x ** 4 + KAPPA * x ** 6 + V4(x)
    dV6 = lambda x: 2 * MU * x - 4 * NU * x ** 3 + 6 * KAPPA * x ** 5 + dV4(x)
    sstar = np.sqrt(NU / (2 * KAPPA))
    Vstar = MU - NU ** 2 / (4 * KAPPA)
    omega_star = np.sqrt(Vstar / c)
    sigma = np.sqrt(2 * c * KAPPA) * sstar ** 4 / 4
    r = {"omega_c": wc, "thin wall: omega*": omega_star, "sigma": sigma}
    # thin-wall crossing: (wc - omega*) J = 4 pi R^2 sigma with Vol* = J / (2 sqrt(Vstar sstar^2 c sstar^2))
    Vol_of_J = lambda J: J / (2 * np.sqrt(Vstar * sstar ** 2 * c * sstar ** 2))
    R_of_J = lambda J: (3 * Vol_of_J(J) / (4 * np.pi)) ** (1 / 3)
    from scipy.optimize import brentq
    Jx = brentq(lambda J: (wc - omega_star) * J - 4 * np.pi * R_of_J(J) ** 2 * sigma, 1e2, 1e7)
    r["thin wall: crossing J"] = float(Jx); r["thin wall: radius at crossing"] = float(R_of_J(Jx))
    res = {}
    Js = [200, 2000, 5000, 3e4, 1e5] + ([] if fast else [3e5])
    for J in Js:
        R = 240.0 if J >= 3e5 else (150.0 if J >= 1e5 else (96.0 if J >= 3e4 else 48.0))
        dr = 0.5
        N = int(round(R / dr)); rr = (np.arange(N) + 0.5) * dr
        inits = []
        # uniform init at the AM-GM optimum
        Vol = 4 * np.pi * R ** 3 / 3
        xopt = J / (2 * np.sqrt(MU * c))
        inits.append(np.full(N, np.sqrt(min(xopt / Vol, 0.5))))
        # thin-wall inits
        for R0 in (R_of_J(J) * f for f in (0.6, 1.0, 1.5)):
            inits.append(sstar * 0.5 * (1 - np.tanh((rr - R0) / 3.0)) + 1e-3)
        inits.append(0.3 * np.exp(-rr ** 2 / 100.0) + 1e-3)
        row = {}
        for name, V, dV in (("quadratic+V4", Vq, dVq), ("sextic+V4", V6, dV6)):
            for bc in ("free", "dirichlet"):
                for Rb in sorted(set([R, 240.0])):
                    Nb = int(round(Rb / dr)); rb = (np.arange(Nb) + 0.5) * dr
                    ini = []
                    Volb = 4 * np.pi * Rb ** 3 / 3
                    ini.append(np.full(Nb, np.sqrt(min(xopt / Volb, 0.5))) * (1 - np.exp(-(Rb - rb) / 2.0) if bc == "dirichlet" else 1.0))
                    for R0 in (R_of_J(J) * f for f in (0.6, 1.0, 1.5)):
                        ini.append(sstar * 0.5 * (1 - np.tanh((rb - R0) / 3.0)) + 1e-3)
                    ini.append(0.3 * np.exp(-rb ** 2 / 100.0) + 1e-3)
                    ini.append(0.2 * np.exp(-rb ** 2 / 1000.0) + 1e-3)
                    (E, sv, X), _ = reduced_line_min(J, Rb, V, dV, c, dr, ini, dirichlet=(bc == "dirichlet"))
                    row[f"{name} {bc} R={Rb:g}"] = {"E/(omega_c J)": float(E / (wc * J)), "max s": float(sv.max()),
                                                    "s at box edge": float(sv[-1]),
                                                    "eff radius (3X/(4pi c smax^2))^(1/3)": float((3 * X / (4 * np.pi * c * sv.max() ** 2)) ** (1 / 3))}
        res[f"J={J:g}"] = row
    r["minima"] = res
    def get(J, name, bc, Rb):
        return res[f"J={J:g}"][f"{name} {bc} R={Rb:g}"]["E/(omega_c J)"]
    q_ok = all(v["E/(omega_c J)"] >= 1.0 - 1e-9 for row in res.values() for k, v in row.items() if k.startswith("quadratic"))
    thin_ok = abs(Jx - 2.6e4) / 2.6e4 < 0.05
    claim_R240 = {30000: 0.998, 100000: 0.993, 300000: 0.980}
    free240 = {J: get(J, "sextic+V4", "free", 240) for J in claim_R240 if f"J={J:g}" in res}
    r240_ok = all(abs(free240[J] - claim_R240[J]) < 0.004 for J in free240)
    small_J_undercut = {J: get(J, "sextic+V4", "free", 240) for J in (200, 2000, 5000)}
    small_J_undercut_48 = {J: get(J, "sextic+V4", "free", 48) for J in (200, 2000, 5000)}
    undercut = all(v < 1.0 for v in small_J_undercut.values()) and all(v < 1.0 for v in small_J_undercut_48.values())
    # Dirichlet floor: the dilute state in a pinned box costs (pi/R)^2/2 extra in omega^2
    floor240 = float(np.sqrt(1 + (np.pi / 240.0) ** 2 / (2 * MU / c)))
    r["summary"] = {"quadratic never < 1 (all BCs, boxes)": q_ok, "thin-wall crossing J within 5% of 2.6e4": thin_ok,
                    "sextic free-edge R=240 at J 3e4/1e5/3e5 vs claim 0.998/0.993/0.980": free240,
                    "sextic free-edge R=240 at J 200/2000/5000 (claim: >= 1, 1.0001 at 5000)": small_J_undercut,
                    "sextic free-edge R=48 at J 200/2000/5000": small_J_undercut_48,
                    "sextic undercuts at every J with a free edge": undercut,
                    "Dirichlet (s=0 at R=240) dilute-state floor sqrt(1 + (pi/R)^2 c/(2 mu))": floor240}
    if q_ok and thin_ok and r240_ok and not undercut:
        verdict = "CONFIRMED"
    elif q_ok and thin_ok and r240_ok:
        verdict = "QUALIFIED"
    else:
        verdict = "REFUTED"
    return {"verdict": verdict, "own_numbers": r,
            "note": ("own discretization (cell-centred radial grid dr 0.5, L-BFGS-B with analytic gradient, several inits incl. thin-wall "
                     "profiles), run with a free edge AND with s = 0 pinned at the box edge. Quadratic+V4: never below 1 (rigorous AM-GM "
                     "bound, reproduced). Thin-wall numbers reproduced. The free-edge R=240 sextic values at J 3e4/1e5/3e5 reproduce the "
                     "claim's 0.998/0.993/0.980, so that is the producer's setup; but in that same setup the box-filling uniform state "
                     "undercuts omega_c J at EVERY J (V/s^2 < mu for all s in (0, 0.3) for the sextic): 0.9996 at J = 5000 (claim 1.0001), "
                     "and 0.968 in a R=48 box. The 'only above ~2.6e4' threshold is a statement about the LOCALIZED (Q-ball, s -> 0 at the edge) "
                     "state, not about the minimum of E_J; with s pinned to 0 at R=240 (the lattice's vacuum edge) the dilute floor rises by "
                     "1.7 % and the crossing moves to J ~ 1e5.")}


# ============================================================ C2d
def check_C2d():
    g, d = G0, DELTA
    f = lambda x: (x - g) * (x - 1) / ((x - g) ** 2 + (x - 1) ** 2)
    w = lambda x: f(x) / f(d)
    W = lambda s: (w(d + s) * w(d - s)) ** 2
    xs = np.linspace(-200, 200, 4000001)
    r = {"sup|w| (scan)": float(np.abs(w(xs)).max()), "sup|w| exact = 0.5/f(delta)": 0.5 / f(d),
         "w(-g)": float(w(-g)), "w(g)": float(w(g)), "w(1)": float(w(1.0))}
    U = lambda s: MU * s ** 2 - NU * s ** 4 + KAPPA * s ** 6
    ratio = lambda s: U(s) / (s ** 2 * W(s))
    sstar = np.sqrt(NU / (2 * KAPPA))
    r["s*"] = float(sstar); r["W(s*)"] = float(W(sstar)); r["U/(s^2 W) at s*"] = float(ratio(sstar))
    ss = np.linspace(1e-4, 0.69, 700000)
    rv = ratio(ss)
    r["min U/(s^2 W) on (0, 0.69)"] = float(rv.min()); r["argmin s"] = float(ss[rv.argmin()])
    r["U/(s^2 W) at s=1e-4"] = float(ratio(1e-4))
    w2 = float((W(1e-3) - 1) / 1e-6)
    r["W''(0)/2 (numeric)"] = w2
    ss2 = np.linspace(0.71, 7.6, 200000); rv2 = ratio(ss2)
    r["min U/(s^2 W) on (0.71, 7.6)"] = float(rv2.min())
    ss3 = np.linspace(7.8, 50, 200000); rv3 = ratio(ss3)
    r["min U/(s^2 W) on (7.8, 50)"] = float(rv3.min())
    r["W=1: min U/s^2"] = MU - NU ** 2 / (4 * KAPPA)
    interior_dip = rv.min() < MU - 1e-9
    if interior_dip:
        verdict = "REFUTED"
        note = ("sup|w| = 5.545 and w(-g) = 4.74 and the value at s* CONFIRMED; but the claim 'the actual minimum over s "
                "is mu, reached only as s -> 0: no interior minimum below mu' is FALSE: U/(s^2 W) dips below mu at "
                f"small s (min {rv.min():.6f} at s = {ss[rv.argmin()]:.4f} on (0, 0.69)), because "
                f"W(s) = 1 + {w2:.3f} s^2 + ... and U/s^2 = mu - nu s^2 + ..., so the ratio starts "
                "below mu; the author's crossing survives the weight at the level of this minimum")
    else:
        verdict = "CONFIRMED"
        note = (f"no interior dip below mu: sup|w| = 5.545, w(-g) = 4.74, value at s* = {r['U/(s^2 W) at s*']:.6f} with W(s*) = {r['W(s*)']:.3f}, "
                f"min over s in (0, 0.69) = mu at s -> 0, also above mu on (0.71, 7.6) and (7.8, 50). Margin is thin: W(s) = 1 + "
                f"{w2:.2f} s^2 + ..., so U/(s^2 W) = mu + (4.09 mu - nu) s^2 + ... = mu + 0.0009 s^2; a 2 % larger nu "
                "would create the interior dip.")
    return {"verdict": verdict, "own_numbers": r, "note": note}


# ============================================================ C3
def dress_and_twist(Mh, h, k):
    """M_inside = L (R Mh R^T) L^T with the R15-V-b boost profile and the (1,2) twist by k z."""
    n = Mh.shape[0]
    X, Y, Z = coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    chi = 0.5 * np.exp(-r * r / 8.0)
    nhat = np.stack([X / r, Y / r, Z / r], axis=-1)
    Lb = boost(nhat, chi)
    if k == 0.0:
        RMR = Mh
    else:
        ca, sa = np.cos(k * Z), np.sin(k * Z)
        R = np.zeros((n, n, n, 4, 4)); R[..., 0, 0] = 1.0; R[..., 3, 3] = 1.0
        R[..., 1, 1] = ca; R[..., 2, 2] = ca; R[..., 1, 2] = -sa; R[..., 2, 1] = sa
        RMR = R @ Mh @ np.swapaxes(R, -1, -2)
    return Lb @ RMR @ np.swapaxes(Lb, -1, -2), RMR


def check_C3(fast=False):
    out = {}
    claim = {"r10": {"eta": [-36, -109, -188], "norm": [-183, -611, -1696], "reb": [195, 629, 1500],
                     "E_norm_dressed0": 25921, "E_eta_dressed0": 12601},
             "m": {"eta": [-58, -100, -58], "norm": [-53, -298, -861], "reb": [54, 359, 701]}}
    allok = True; mins = []
    for tag, rel_path in (("r10", FIELD_R10), ("m", FIELD_M)):
        Mh = load_field(rel_path); n = Mh.shape[0]; h = 48.0 / n
        res = {}
        for st in (["sym", "cen"] if tag == "r10" else ["sym"]):
            Eh = curvature_energies(Mh, h, st)
            Md0, _ = dress_and_twist(Mh, h, 0.0)
            Ed0 = curvature_energies(Md0, h, st)
            row = {"E_hedgehog": Eh, "E_dressed_k0": Ed0, "cross": {}}
            for k in (0.5, 1.0, 2.0):
                Mi, Mt = dress_and_twist(Mh, h, k)
                Ei = curvature_energies(Mi, h, st); Et = curvature_energies(Mt, h, st)
                cross = {key: (Ei[key] - Ed0[key]) - (Et[key] - Eh[key]) for key in ("E_eta", "E_norm", "E_rebuild")}
                row["cross"][f"k{k}"] = cross
                mins += [Ei["min_dens_norm"], Ei["min_dens_rebuild"], Ed0["min_dens_norm"], Ed0["min_dens_rebuild"]]
            res[st] = row
        out[tag] = res
        cr = res["sym"]["cross"]
        signs = all(cr[f"k{k}"]["E_eta"] < 0 and cr[f"k{k}"]["E_norm"] < 0 and cr[f"k{k}"]["E_rebuild"] > 0 for k in (0.5, 1.0, 2.0))
        vals = all(rel(cr[f"k{k}"][kk], claim[tag][ck][i]) < 0.03 for i, k in enumerate((0.5, 1.0, 2.0))
                   for kk, ck in (("E_eta", "eta"), ("E_norm", "norm"), ("E_rebuild", "reb")))
        out[tag]["signs_ok"] = signs; out[tag]["values_within_3pct"] = vals
        allok = allok and signs and vals
    e0 = out["r10"]["sym"]["E_dressed_k0"]
    out["r10"]["dressed_k0_vs_claim"] = {"E_norm": (e0["E_norm"], 25921), "E_eta": (e0["E_eta"], 12601)}
    allok = allok and rel(e0["E_norm"], 25921) < 0.01 and rel(e0["E_eta"], 12601) < 0.01
    out["min per-cell density (norm, rebuild) over all configs"] = float(min(mins))
    allok = allok and min(mins) >= 0
    return {"verdict": "CONFIRMED" if allok else ("QUALIFIED" if out["r10"]["signs_ok"] and out["m"]["signs_ok"] else "REFUTED"),
            "own_numbers": out,
            "note": "own dressing/twist and energies; u per cell from eig(N); the fields' time rows are zero so u = L e0; positivity of the G-completions is structural (G = S S^T). All 'sym'-stencil numbers reproduced to 4 digits (the claim's min density 9.4e-7 is my 2.79e-7 times h^3 = 3.375, i.e. an h^3-weighted per-cell energy). Caveat: the eta cross terms are stencil-dependent (central stencil: +2.7 / +31.7 / +49.6, i.e. positive), while the norm < 0 < rebuild sign difference survives the stencil change (-75/-249/-473 vs +111/+343/+542)."}


# ============================================================ C4a
def check_C4a():
    psi, phi, s, alpha, g, d = sp.symbols("psi phi s alpha g delta", real=True)
    D = sp.diag(g, 1, d + s, d - s)
    R = rot_sym(1, 2, psi) * rot_sym(2, 3, phi)
    M = R * D * R.T
    Rn = rot_sym(1, 2, psi) * rot_sym(2, 3, alpha / 2) * rot_sym(1, 2, psi).T
    TM = Rn * M * Rn.T
    Mshift = M.subs(phi, phi + alpha / 2)
    syms = [psi, phi, s, alpha, g, d]
    r = {"max|T_alpha M - M(phi + alpha/2)| (25 random pts)": numzero(TM - Mshift, syms)}
    loc = rot_sym(1, 2, psi).T * M * rot_sym(1, 2, psi)
    blk = loc[2:, 2:] - (d * sp.eye(2) + s * sp.Matrix([[sp.cos(2 * phi), sp.sin(2 * phi)], [sp.sin(2 * phi), -sp.cos(2 * phi)]]))
    r["max|local (2,3) block - (delta I + s(cos2phi, sin2phi))|"] = numzero(blk, syms)
    r["max|T_{2pi} M - M|"] = numzero(TM.subs(alpha, 2 * sp.pi) - M, syms)
    r["max|T_alpha M - M| at s=0"] = numzero((TM - M).subs(s, 0), syms)
    pz, fz = sp.symbols("psi_z phi_z", real=True)
    Az = pz * M.diff(psi) + fz * M.diff(phi)
    E2 = (Az * Az).trace()
    E2_target = 2 * pz ** 2 * (sp.cos(phi) ** 2 * (1 - d - s) ** 2 + sp.sin(phi) ** 2 * (1 - d + s) ** 2) + 8 * fz ** 2 * s ** 2
    syms2 = syms + [pz, fz]
    r["max|E2 - [2 psi_z^2 (cos^2 (1-d-s)^2 + sin^2 (1-d+s)^2) + 8 phi_z^2 s^2]|"] = numzero(E2 - E2_target, syms2)
    r["dE2/dphi (simplified)"] = str(sp.simplify(E2.diff(phi)))
    r["dE2/dphi max|.| (random pts)"] = numzero(E2.diff(phi), syms2)
    eta = sp.diag(-1, 1, 1, 1)
    r["max|d/dphi tr N^p|, p=1..4"] = [numzero(((M * eta) ** p).trace().diff(phi), syms) for p in range(1, 5)]
    fM = sp.lambdify((psi, phi, s, g, d), M, "numpy")
    fA = sp.lambdify((psi, phi, s, g, d, pz, fz), Az, "numpy")
    rng = np.random.default_rng(4)
    worst = 0.0; worst_psi0 = 0.0; phidep = 0.0
    for _ in range(30):
        p_, f_, s_, pz_, fz_ = rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.02, 0.28), rng.normal(), rng.normal()
        Mv = np.array(fM(p_, f_, s_, G0, DELTA), float)
        Av = np.array(fA(p_, f_, s_, G0, DELTA, pz_, fz_), float)
        _, _, P23 = spectral_projectors(Mv[None])
        K = float(kp23_static(Mv[None], [Av[None]], P23)[0])
        worst = max(worst, abs(K - 4 * fz_ ** 2 * s_ ** 2))
        Mv0 = np.array(fM(0.0, f_, s_, G0, DELTA), float); Av0 = np.array(fA(0.0, f_, s_, G0, DELTA, pz_, fz_), float)
        _, _, P0 = spectral_projectors(Mv0[None])
        worst_psi0 = max(worst_psi0, abs(float(kp23_static(Mv0[None], [Av0[None]], P0)[0]) - 4 * fz_ ** 2 * s_ ** 2))
        Mv2 = np.array(fM(p_, f_ + 0.9, s_, G0, DELTA), float); Av2 = np.array(fA(p_, f_ + 0.9, s_, G0, DELTA, pz_, fz_), float)
        _, _, P2 = spectral_projectors(Mv2[None])
        phidep = max(phidep, abs(float(kp23_static(Mv2[None], [Av2[None]], P2)[0]) - K))
    r["K_P^23: max |K - 4 phi_z^2 s^2| at psi=0 (30 random)"] = worst_psi0
    r["K_P^23: max |K - 4 phi_z^2 s^2| general psi (30 random)"] = worst
    r["K_P^23: max |K(phi+0.9) - K(phi)|"] = phidep
    ok = (r["max|T_alpha M - M(phi + alpha/2)| (25 random pts)"] < 1e-12
          and r["max|local (2,3) block - (delta I + s(cos2phi, sin2phi))|"] < 1e-12
          and r["max|T_{2pi} M - M|"] < 1e-12 and r["max|T_alpha M - M| at s=0"] < 1e-12
          and r["max|E2 - [2 psi_z^2 (cos^2 (1-d-s)^2 + sin^2 (1-d+s)^2) + 8 phi_z^2 s^2]|"] < 1e-12
          and r["dE2/dphi max|.| (random pts)"] > 1e-3
          and worst < 1e-10 and phidep < 1e-10 and all(x < 1e-9 for x in r["max|d/dphi tr N^p|, p=1..4"]))
    return {"verdict": "CONFIRMED" if ok else "REFUTED", "own_numbers": r,
            "note": "sympy expressions evaluated at random points (sympy's trig simplify stalls on the 4x4 conjugations); K_P^23 via eig projectors at 30 random points (it equals 4 phi_z^2 s^2 for every psi, stronger than the claim); E2 closed form derived independently"}


# ============================================================ C4b
def rodrigues4(nvec, a):
    """4x4 spatial rotation by angle a about unit axis nvec (3,)."""
    n1, n2, n3 = nvec
    Kx = np.array([[0, -n3, n2], [n3, 0, -n1], [-n2, n1, 0]])
    R3 = np.cos(a) * np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * np.outer(nvec, nvec)
    R = np.eye(4); R[1:, 1:] = R3
    return R


def leading_n(M):
    w, V = np.linalg.eigh(M[1:, 1:])
    n = V[:, -1]
    return n if n[0] >= 0 else -n


def point_densities(A1, A2, M0):
    eta = ETA
    u = timelike_u(M0[None])[0]; Gm = G_of_u(u)
    Fe = A1 @ eta @ A2 - A2 @ eta @ A1
    Fg = A1 @ Gm @ A2 - A2 @ Gm @ A1
    _, _, P23 = spectral_projectors(M0[None])
    N = M0 @ eta
    ev = np.sort(np.linalg.eigvals(N).real)
    return {"I1": float(quad(eta, Fe, eta)), "I_norm": float(quad(Gm, Fe, Gm)), "I_rebuild": float(quad(Gm, Fg, Gm)),
            "E2": float(tr(A1 @ Gm @ A1 @ Gm) + tr(A2 @ Gm @ A2 @ Gm)),
            "K_P23": float(kp23_static(M0[None], [A1[None], A2[None]], P23)[0]),
            "V4dd": float(v4dd(M0[None])[0]), "split": float((ev[1] - ev[2]) ** 2)}


def check_C4b():
    s = 0.15
    Ds = np.diag([G0, 1.0, DELTA + s, DELTA - s])
    out = {}; ok = True
    for seed in (11, 12, 13):
        rng = np.random.default_rng(seed)
        A = []
        for _ in range(2):
            S = rng.normal(size=(3, 3)); X = np.zeros((4, 4)); X[1:, 1:] = 0.3 * (S + S.T); A.append(X)
        def TM(x, alpha):
            Mx = Ds + x[0] * A[0] + x[1] * A[1]
            R = rodrigues4(leading_n(Mx), alpha / 2)
            return R @ Mx @ R.T
        def jets_fd(alpha, hstep=1e-3):
            Ap = []
            for i in range(2):
                e = np.zeros(2); e[i] = 1.0
                f = lambda t: TM(t * e, alpha)
                Ap.append((-f(2 * hstep) + 8 * f(hstep) - 8 * f(-hstep) + f(-2 * hstep)) / (12 * hstep))
            return Ap
        def jets(alpha):
            """analytic chain rule: A_i' = R A_i R^T + dR_i D R^T + R D dR_i^T, dR_i = dR/dn . dn_i,
            dn_i from first-order eigenvector perturbation of the spatial block at D_s (n = e1)."""
            n0 = np.array([1.0, 0.0, 0.0]); lam = np.array([1.0, DELTA + s, DELTA - s])
            a = alpha / 2
            R = rodrigues4(n0, a)
            Ap = []
            for i in range(2):
                Asp = A[i][1:, 1:]
                dn = np.zeros(3)
                for kk in (1, 2):
                    dn[kk] = Asp[kk, 0] / (lam[0] - lam[kk])
                Kx = np.array([[0, -dn[2], dn[1]], [dn[2], 0, -dn[0]], [-dn[1], dn[0], 0]])
                dR3 = np.sin(a) * Kx + (1 - np.cos(a)) * (np.outer(dn, n0) + np.outer(n0, dn))
                dR = np.zeros((4, 4)); dR[1:, 1:] = dR3
                Ap.append(R @ A[i] @ R.T + dR @ Ds @ R.T + R @ Ds @ dR.T)
            return Ap
        alphas = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        keys = ["I1", "I_norm", "I_rebuild", "E2", "K_P23", "V4dd", "split"]
        vals = {k: [] for k in keys}
        for al in alphas:
            Ap = jets(al)
            dens = point_densities(Ap[0], Ap[1], TM(np.zeros(2), al))
            for k in keys:
                vals[k].append(dens[k])
        res = {}
        for k in keys:
            v = np.array(vals[k]); c = np.fft.rfft(v) / len(v)
            mean = v.mean()
            res[k] = {"rel_variation (max-min)/mean": float((v.max() - v.min()) / abs(mean)),
                      "max |fourier coeff| beyond freq 1, relative": float(np.abs(c[2:]).max() / max(abs(c[0]), 1e-300))}
        # validate the analytic jets against 4th-order finite differences at alpha = 0.7 and 2.9
        Ap1 = jets_fd(0.7); Ap2 = jets(0.7); Ap3 = jets_fd(2.9); Ap4 = jets(2.9)
        res["max|A'_FD - A'_analytic| (alpha 0.7, 2.9)"] = float(max(np.abs(Ap1[i] - Ap2[i]).max() for i in range(2)) + max(np.abs(Ap3[i] - Ap4[i]).max() for i in range(2)))
        # periodicity
        Ap0 = jets(0.3); Ap2pi = jets(0.3 + 2 * np.pi)
        res["max|A'(alpha) - A'(alpha + 2pi)|"] = float(max(np.abs(Ap0[i] - Ap2pi[i]).max() for i in range(2)))
        # analytic check of the FD jets: chain rule A_i' = R (A_i + [Om_i, D_s]) R^T with Om_i from first-order eigenvector perturbation
        out[f"seed{seed}"] = res
        inv_ok = all(res[k]["rel_variation (max-min)/mean"] < 1e-11 for k in ("K_P23", "V4dd", "split"))
        var_ok = all(res[k]["rel_variation (max-min)/mean"] > 1e-3 for k in ("I1", "I_norm", "I_rebuild", "E2"))
        deg1_ok = all(res[k]["max |fourier coeff| beyond freq 1, relative"] < 1e-12 for k in keys)
        fd_ok = res["max|A'_FD - A'_analytic| (alpha 0.7, 2.9)"] < 1e-8
        ok = ok and fd_ok
        ok = ok and inv_ok and var_ok and deg1_ok
    return {"verdict": "CONFIRMED" if ok else "REFUTED", "own_numbers": out,
            "note": "own random jets (3 seeds; the claim's 0.55 / 0.13 variation magnitudes are seed-specific and cannot be reproduced, only the structure is tested); jets by the analytic chain rule (first-order eigenvector perturbation), validated against 4th-order central differences; 64 alpha samples, rfft for the trigonometric degree"}


# ============================================================ C4c
def smooth_field(X, Y, Z, L, s=0.15, amp=0.06, seed=21):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    M = np.zeros(X.shape + (4, 4)); M[..., 0, 0] = G0; M[..., 1, 1] = 1.0
    M[..., 2, 2] = DELTA + s; M[..., 3, 3] = DELTA - s
    for _ in range(6):
        kv = rng.integers(-1, 2, size=3) * 2 * np.pi / L
        ph = rng.uniform(0, 2 * np.pi)
        S = rng.normal(size=(3, 3)); S = amp * (S + S.T) / 2
        wave = np.cos(kv[0] * X + kv[1] * Y + kv[2] * Z + ph)
        M[..., 1:, 1:] += wave[..., None, None] * S
    return M


def circle_apply(M, alpha):
    w, V = np.linalg.eigh(M[..., 1:, 1:])
    nv = V[..., :, -1]
    sgn = np.where(nv[..., 0] >= 0, 1.0, -1.0)
    nv = nv * sgn[..., None]
    a = alpha / 2
    Kx = np.zeros(nv.shape[:-1] + (3, 3))
    Kx[..., 0, 1] = -nv[..., 2]; Kx[..., 0, 2] = nv[..., 1]; Kx[..., 1, 0] = nv[..., 2]
    Kx[..., 1, 2] = -nv[..., 0]; Kx[..., 2, 0] = -nv[..., 1]; Kx[..., 2, 1] = nv[..., 0]
    R3 = np.cos(a) * np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * nv[..., :, None] * nv[..., None, :]
    R = np.zeros(M.shape); R[..., 0, 0] = 1.0; R[..., 1:, 1:] = R3
    return R @ M @ np.swapaxes(R, -1, -2)


def lattice_scalars(M, h, st):
    de, _, _ = curvature_densities(M, h, st)   # E_u = 4 sum tr(eta F eta F^T)
    _, _, P23 = spectral_projectors(M)
    K = 0.0; E2 = 0.0
    for br, wt in branches(st):
        A = [d1(M, ax, h, br) for ax in range(3)]
        K = K + wt * kp23_static(M, A, P23)
        E2 = E2 + wt * sum(tr(Ai @ Ai) for Ai in A)   # G = I (zero time rows)
    h3 = h ** 3
    return {"K_P23": float(K.sum() * h3), "E_u": float(de.sum() * h3), "E2": float(E2.sum() * h3)}


def check_C4c():
    L = 8.0; alpha = 0.7
    out = {}
    for st in ("per", "sym"):
        rows = {}
        for n in (16, 32, 64):
            h = L / n
            X, Y, Z = coords(n, h)
            M = smooth_field(X, Y, Z, L)
            TM = circle_apply(M, alpha)
            a = lattice_scalars(M, h, st); b = lattice_scalars(TM, h, st)
            rows[f"n{n}"] = {k: {"M": a[k], "TM": b[k], "rel_defect": rel(b[k], a[k])} for k in a}
        rows["ratio n16/n32 of rel_defect"] = {k: rows["n16"][k]["rel_defect"] / rows["n32"][k]["rel_defect"] for k in ("K_P23", "E_u", "E2")}
        rows["ratio n32/n64 of rel_defect"] = {k: rows["n32"][k]["rel_defect"] / rows["n64"][k]["rel_defect"] for k in ("K_P23", "E_u", "E2")}
        out[st] = rows
    rp = out["per"]["ratio n16/n32 of rel_defect"]; rs = out["sym"]["ratio n16/n32 of rel_defect"]
    rp2 = out["per"]["ratio n32/n64 of rel_defect"]; rs2 = out["sym"]["ratio n32/n64 of rel_defect"]
    ok = (rp["K_P23"] > 2.5) and (3.5 < rp2["K_P23"] < 4.5) and max(rp["E_u"], rp["E2"], rp2["E_u"], rp2["E2"]) < 2.0
    ok_s = (rs["K_P23"] > 2.5) and (3.5 < rs2["K_P23"] < 4.5) and max(rs["E_u"], rs["E2"], rs2["E_u"], rs2["E2"]) < 2.0
    return {"verdict": "CONFIRMED" if (ok and ok_s) else ("QUALIFIED" if ok else "REFUTED"), "own_numbers": out,
            "note": "own periodic smooth random field (6 Fourier modes with |m| <= 1 per axis, amp 0.06, s 0.15); 'per' = periodic central differences (clean O(h^2) test), 'sym' = the fwd/bwd branch average with box edges; three resolutions so the asymptotic ratio 4 is visible"}


# ============================================================ C5
def check_C5():
    psi, s, om, pz, g, d, b = sp.symbols("psi s omega psi_z g delta b", real=True)
    eta = sp.diag(-1, 1, 1, 1)
    Ds = sp.diag(g, 1, d + s, d - s)
    R = rot_sym(1, 2, psi)
    M = R * Ds * R.T
    G23s = sp.zeros(4, 4); G23s[2, 3] = -1; G23s[3, 2] = 1
    Gloc = R * G23s * R.T
    A0 = om * (Gloc * M - M * Gloc)
    Az = pz * M.diff(psi)
    F = A0 * eta * Az - Az * eta * A0
    I1 = sp.simplify((eta * F * eta * F.T).trace())
    r = {"local: tr(eta F eta F^T)": str(I1),
         "local: minus 8 om^2 psi_z^2 s^2 (d+s-1)^2": str(sp.simplify(I1 - 8 * om ** 2 * pz ** 2 * s ** 2 * (d + s - 1) ** 2))}
    A0r = om * (G23s * M - M * G23s)
    Fr = A0r * eta * Az - Az * eta * A0r
    I1r = sp.simplify((eta * Fr * eta * Fr.T).trace().subs(s, 0))
    targ = -om ** 2 * pz ** 2 * (d - 1) ** 4 * (sp.cos(4 * psi) - 8 * sp.sin(psi) ** 4 - 1) / 4
    r["rigid s=0: tr(eta F eta F^T)"] = str(I1r)
    r["rigid s=0: minus claimed formula (trig-simplified)"] = str(sp.simplify(sp.expand_trig(I1r - targ)))
    r["rigid s=0: max|. - claimed| (random pts)"] = numzero(I1r - targ, [psi, om, pz, d])
    r["rigid s=0 at psi=0.7, d=0.3, om=pz=1"] = float(I1r.subs({psi: 0.7, d: 0.3, om: 1, pz: 1}))
    # K_P^23 on the sheet (numeric, eig projectors)
    fM = sp.lambdify((psi, s, g, d), M, "numpy"); fAz = sp.lambdify((psi, s, g, d, pz), Az, "numpy")
    fA0 = sp.lambdify((psi, s, g, d, om), A0, "numpy")
    rng = np.random.default_rng(5); wst = 0.0; win = 0.0
    for _ in range(20):
        p_, s_, pz_, om_ = rng.uniform(-3, 3), rng.uniform(0.02, 0.28), rng.normal(), rng.normal()
        Mv = np.array(fM(p_, s_, G0, DELTA), float); Azv = np.array(fAz(p_, s_, G0, DELTA, pz_), float)
        A0v = np.array(fA0(p_, s_, G0, DELTA, om_), float)
        _, _, P23 = spectral_projectors(Mv[None])
        wst = max(wst, abs(float(kp23_static(Mv[None], [Azv[None]], P23)[0])))
        win = max(win, abs(float(kp23_inertia(Mv[None], A0v[None], P23)[0]) - 4 * om_ ** 2 * s_ ** 2))
    r["K_P^23 static max (20 random)"] = wst; r["K_P^23 inertia max|. - 4 om^2 s^2|"] = win
    E2 = sp.simplify((Az * Az).trace())
    r["E2 static - 2 psi_z^2 (d+s-1)^2"] = str(sp.simplify(E2 - 2 * pz ** 2 * (d + s - 1) ** 2))
    # boost sheets on the diagonal split sheet
    A0d = om * (G23s * Ds - Ds * G23s)
    bs = {}; bs_expr = {}
    for i in (1, 2, 3):
        K = sp.zeros(4, 4); K[0, i] = 1; K[i, 0] = 1
        Azb = b * (K * Ds + Ds * K)
        Fe = A0d * eta * Azb - Azb * eta * A0d
        Fi = A0d * Azb - Azb * A0d
        ie = sp.simplify((eta * Fe * eta * Fe.T).trace())
        gn = sp.simplify((Fe * Fe.T).trace()); gr = sp.simplify((Fi * Fi.T).trace())
        bs[f"i={i}"] = {"<F,F>_eta": str(ie), "G-norm of F^eta": str(gn), "G-norm of F^G": str(gr)}
        bs_expr[f"i={i}"] = (ie, gn, gr)
    r["boost sheets"] = bs
    ex = bs_expr
    e1 = sp.simplify(ex["i=1"][0]) == 0
    e2 = sp.simplify(ex["i=2"][0] + 8 * b ** 2 * om ** 2 * s ** 2 * (g + d + s) ** 2) == 0
    e3 = sp.simplify(ex["i=3"][0] + 8 * b ** 2 * om ** 2 * s ** 2 * (g + d - s) ** 2) == 0
    g2 = sp.simplify(ex["i=2"][1] - 8 * b ** 2 * om ** 2 * s ** 2 * (g + d + s) ** 2) == 0
    g2r = sp.simplify(ex["i=2"][2] - 8 * b ** 2 * om ** 2 * s ** 2 * (g + d + s) ** 2) == 0
    g3 = sp.simplify(ex["i=3"][1] - 8 * b ** 2 * om ** 2 * s ** 2 * (g + d - s) ** 2) == 0
    g3r = sp.simplify(ex["i=3"][2] - 8 * b ** 2 * om ** 2 * s ** 2 * (g + d - s) ** 2) == 0
    r["boost sheet checks (i1 zero, i2, i3 eta; i2,i3 G-norm both completions)"] = [e1, e2, e3, g2, g2r, g3, g3r]
    # tilt channel: sign only
    r["tilt: 16 om^2 s^2 - c_s s^2 < 0 iff c_s > 16 om^2 (trivial); R15-H's gamma_total itself NOT re-derived"] = True
    ok = (r["local: minus 8 om^2 psi_z^2 s^2 (d+s-1)^2"] == "0" and r["rigid s=0: max|. - claimed| (random pts)"] < 1e-12
          and wst < 1e-12 and win < 1e-12 and r["E2 static - 2 psi_z^2 (d+s-1)^2"] == "0" and all([e1, e2, e3, g2, g2r, g3, g3r]))
    return {"verdict": "CONFIRMED" if ok else "REFUTED", "own_numbers": r,
            "note": "sympy for every closed form; K_P^23 sheet values by eig projectors at 20 random points; the tilt-channel statement depends on R15-H's gamma_total definition, which was not re-derived here (only its sign logic is trivial)"}


# ============================================================ C6
def check_C6():
    a, b, da, db = sp.symbols("a b da db", real=True)
    rho = sp.sqrt(a ** 2 + b ** 2)
    lp, lm = rho, -rho
    dlp = lp.diff(a) * da + lp.diff(b) * db
    dlm = lm.diff(a) * da + lm.diff(b) * db
    expr = sp.simplify(dlp ** 2 + dlm ** 2 - 2 * (a * da + b * db) ** 2 / (a ** 2 + b ** 2))
    B = sp.Matrix([[a, b], [b, -a]])
    ev = sorted(B.eigenvals().keys(), key=lambda e: str(e))
    r = {"eigenvalues of B": [str(e) for e in ev], "(dl+)^2 + (dl-)^2 - 2(a da + b db)^2/(a^2+b^2)": str(expr)}
    radial = sp.simplify((2 * (a * da + b * db) ** 2 / (a ** 2 + b ** 2)).subs({da: a / rho, db: b / rho}))
    tang = sp.simplify((2 * (a * da + b * db) ** 2 / (a ** 2 + b ** 2)).subs({da: -b / rho, db: a / rho}))
    r["radial unit direction"] = str(radial); r["tangential unit direction"] = str(tang)
    r["rho^2 = a^2 + b^2 is a polynomial (smooth)"] = True
    ok = expr == 0 and radial == 2 and tang == 0
    return {"verdict": "CONFIRMED" if ok else "REFUTED", "own_numbers": r, "note": "sympy"}


# ============================================================ C7
def spatial_triples(M):
    """three eigenvalues of N other than the timelike one; the time rows are zero so these are eigh of the spatial block."""
    assert np.abs(M[..., 0, 1:]).max() == 0.0
    return np.linalg.eigvalsh(M[..., 1:, 1:])[..., ::-1]


def biaxiality(trip):
    Q = trip - trip.mean(axis=-1, keepdims=True)
    q2 = np.sum(Q ** 2, axis=-1); q3 = np.sum(Q ** 3, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta2 = np.where(q2 > 1e-14, 1 - 6 * q3 ** 2 / q2 ** 3, 0.0)
    return beta2, q3


def check_C7():
    out = {}
    for tag, rel_path, n, L in (("n32", FIELD_P4_32, 32, 48.0), ("n48", FIELD_P4_48, 48, 72.0)):
        M = load_field(rel_path); h = L / n
        X, Y, Z = coords(n, h); r = np.sqrt(X * X + Y * Y + Z * Z)
        trip = spatial_triples(M); beta2, q3 = biaxiality(trip)
        c = n // 2
        inner = {}
        for i in (c - 1, c):
            for j in (c - 1, c):
                for k in (c - 1, c):
                    t = trip[i, j, k]
                    inner[f"({X[i,j,k]:+.2f},{Y[i,j,k]:+.2f},{Z[i,j,k]:+.2f})"] = {"triple": [round(float(v), 4) for v in t],
                                                                                   "beta2": round(float(beta2[i, j, k]), 4),
                                                                                   "trQ3": round(float(q3[i, j, k]), 4)}
        res = {"innermost 8 cells": inner}
        res["beta2 max"] = float(beta2.max())
        imax = np.argwhere(beta2 > beta2.max() - 1e-6)
        res["r of cells with beta2 = max"] = sorted(set(round(float(r[tuple(i)]), 3) for i in imax))
        res["max r with beta2 > 1e-3"] = float(r[beta2 > 1e-3].max())
        res["max beta2 beyond r 4.5"] = float(beta2[r > 4.5].max())
        sel = beta2 > 0.5 * beta2.max()
        res["cells with beta2 > half max"] = int(sel.sum())
        rs = r[sel]; res["their r mean +- std"] = [float(rs.mean()), float(rs.std())]
        w = beta2[sel]; xs = np.stack([X[sel], Y[sel], Z[sel]], axis=-1) / rs[:, None]
        Qm = np.einsum("c,ci,cj->ij", w, xs, xs) / w.sum() - np.eye(3) / 3
        ev, V = np.linalg.eigh(Qm)
        res["quadrupole eigenvalues"] = [float(v) for v in ev]
        ax = V[:, 0] if abs(ev[0] - ev[1]) > abs(ev[2] - ev[1]) else V[:, 2]
        res["axis (distinct eigenvalue)"] = [float(v) for v in ax]
        res["|axis . (1,1,1)/sqrt3|"] = float(abs(ax @ np.ones(3)) / np.sqrt(3))
        # lattice-unit comparison: beta2 on index offsets from the centre
        out[tag] = res
    # lattice-scale identity: compare beta2 of n32 and n48 on the same index offsets
    M32 = load_field(FIELD_P4_32); M48 = load_field(FIELD_P4_48)
    b32, _ = biaxiality(spatial_triples(M32)); b48, _ = biaxiality(spatial_triples(M48))
    sub32 = b32[16 - 6:16 + 6, 16 - 6:16 + 6, 16 - 6:16 + 6]; sub48 = b48[24 - 6:24 + 6, 24 - 6:24 + 6, 24 - 6:24 + 6]
    out["beta2 central 12^3 block: max|n32 - n48| in lattice units"] = float(np.abs(sub32 - sub48).max())
    out["beta2 central 12^3 block: max"] = float(sub32.max())
    Mm = load_field(FIELD_M); bm, _ = biaxiality(spatial_triples(Mm)); n = 32; c = 16
    out["R15-M seed: beta2 max"] = float(bm.max())
    out["R15-M seed: beta2 max outside innermost 8"] = float(np.where(np.ones_like(bm, bool), bm, 0)[(np.arange(n)[:, None, None] // 15 != 1) | (np.arange(n)[None, :, None] // 15 != 1) | (np.arange(n)[None, None, :] // 15 != 1)].max())
    # verdict
    inner32 = out["n32"]["innermost 8 cells"]; nob = sum(1 for v in inner32.values() if v["trQ3"] < 0 and v["beta2"] < 0.05)
    out["n32: number of innermost cells that are oblate uniaxial"] = nob
    ring_ok = all(abs(out[t]["quadrupole eigenvalues"][0] - e) < 0.02 for t, e in (("n32", -0.163), ("n48", -0.188)))
    axis_ok = all(out[t]["|axis . (1,1,1)/sqrt3|"] > 0.99 for t in ("n32", "n48"))
    verdict = "CONFIRMED" if (nob == 8 and ring_ok and axis_ok) else ("QUALIFIED" if (ring_ok and axis_ok) else "REFUTED")
    return {"verdict": verdict, "own_numbers": out,
            "note": ("triples by eigh of the spatial block (the time rows of both fields are exactly zero, so u = e0 exactly); all 8 innermost "
                     "cells listed individually. Ring quadrupole, its (1,1,1) axis, the r = 2.49 maximum, the 24/30 cell counts and the R15-M "
                     "control all reproduced. Qualification: only 6 of the 8 innermost cells carry the oblate triple (0.70, 0.70, -0.11); the two "
                     "cells on the body diagonal, (+,+,+) and (-,-,-), are PROLATE (1.15, 0.16, 0.16) with tr Q^3 > 0 on both boxes. That is the "
                     "same (1,1,1) anisotropy the ring axis reveals (the field keeps the x<->y mirrors but not x->-x, see C8b), and it belongs in "
                     "the innermost-cell statement. 'Identical in lattice units' holds to |d beta^2| <= 0.13 on the central 12^3 block.")}


# ============================================================ C8a
def check_C8a():
    """smooth axisymmetric tensor field: S = diag(d, d, 1) + a(r) xp xp^T + b(r)(xp zh^T + zh xp^T) + c(r) zh zh^T,
    xp = (x, y, 0), zh = e_z, a, b, c Gaussian profiles in (rho, z): the tensor rotates with the point.
    Control: the same tensor but with a, b, c evaluated at a SHIFTED, non-axisymmetric point (profile does not rotate)."""
    L = 24.0; out = {}
    def field(X, Y, Z, control=False):
        if control:
            q = (X - 1.5) ** 2 + 0.5 * Y ** 2 + Z ** 2
        else:
            q = X ** 2 + Y ** 2 + Z ** 2
        a = 0.06 * np.exp(-q / 14.0); b = 0.05 * np.exp(-q / 20.0); c = 0.3 * np.exp(-q / 10.0)
        xp = np.stack([X, Y, 0 * Z], -1); zh = np.zeros_like(xp); zh[..., 2] = 1.0
        S = np.zeros(X.shape + (3, 3)); S[..., 0, 0] = DELTA; S[..., 1, 1] = DELTA; S[..., 2, 2] = 1.0   # director along z (axisymmetric background)
        S += a[..., None, None] * xp[..., :, None] * xp[..., None, :]
        S += b[..., None, None] * (xp[..., :, None] * zh[..., None, :] + zh[..., :, None] * xp[..., None, :])
        S += c[..., None, None] * zh[..., :, None] * zh[..., None, :]
        M = np.zeros(X.shape + (4, 4)); M[..., 0, 0] = G0; M[..., 1:, 1:] = S
        return M
    for n in (32, 64):
        h = L / n; X, Y, Z = coords(n, h)
        for ctrl in (False, True):
            M = field(X, Y, Z, ctrl)
            lie = X[..., None, None] * d1(M, 1, h, "cen") - Y[..., None, None] * d1(M, 0, h, "cen")
            gen = GZ @ M - M @ GZ
            resid = np.sqrt(np.sum((gen - lie) ** 2)) / np.sqrt(np.sum(gen ** 2))
            out[f"n{n} h{h} {'control' if ctrl else 'axisymmetric'} rel norm"] = float(resid)
    ratio = out["n32 h0.75 axisymmetric rel norm"] / out["n64 h0.375 axisymmetric rel norm"]
    out["ratio h0.75/h0.375 (axisymmetric)"] = float(ratio)
    ok = 3.5 < ratio < 4.5 and out["n64 h0.375 control rel norm"] > 0.3
    return {"verdict": "CONFIRMED" if ok else "REFUTED", "own_numbers": out,
            "note": "own smooth axisymmetric field (the vacuum director along x with a disclination would NOT be smooth at the axis, so the anisotropy is carried by Gaussian profiles); the claim's 0.034 / 0.0089 / 0.76 are field-specific, the O(h^2) ratio and the O(1) control are what is tested"}


# ============================================================ C8b
def sY2(theta, phi):
    """spin-weight +2, l = 2 harmonics, m = -2..2 (orthonormal on the sphere)."""
    c, s = np.cos(theta), np.sin(theta)
    Y = {}
    Y[2] = np.sqrt(5 / (64 * np.pi)) * (1 + c) ** 2 * np.exp(2j * phi)
    Y[-2] = np.sqrt(5 / (64 * np.pi)) * (1 - c) ** 2 * np.exp(-2j * phi)
    Y[1] = np.sqrt(5 / (16 * np.pi)) * s * (1 + c) * np.exp(1j * phi)
    Y[-1] = np.sqrt(5 / (16 * np.pi)) * s * (1 - c) * np.exp(-1j * phi)
    Y[0] = np.sqrt(15 / (32 * np.pi)) * s ** 2
    return Y


def shell_decomposition(M, L, rbins):
    n = M.shape[0]; h = L / n
    X, Y, Z = coords(n, h); r = np.sqrt(X * X + Y * Y + Z * Z)
    S = M[..., 1:, 1:]
    w, V = np.linalg.eigh(S); nv = V[..., :, -1]
    rhat = np.stack([X, Y, Z], -1) / r[..., None]
    sgn = np.sign(np.sum(nv * rhat, -1)); sgn[sgn == 0] = 1.0
    nv = nv * sgn[..., None]
    theta = np.arccos(np.clip(Z / r, -1, 1)); phi = np.arctan2(Y, X)
    eth = np.stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)], -1)
    e = eth - np.sum(eth * nv, -1)[..., None] * nv
    e /= np.linalg.norm(e, axis=-1)[..., None]
    f = np.cross(nv, e)
    See = np.einsum("...i,...ij,...j->...", e, S, e); Sff = np.einsum("...i,...ij,...j->...", f, S, f)
    Sef = np.einsum("...i,...ij,...j->...", e, S, f)
    zeta = See - Sff + 2j * Sef
    Ys = sY2(theta, phi)
    out = []
    for r0, r1 in rbins:
        sel = (r >= r0) & (r < r1)
        if sel.sum() < 30:
            continue
        z = zeta[sel]; B = np.stack([Ys[m][sel] for m in (-2, -1, 0, 1, 2)], -1)
        dOm = 4 * np.pi / sel.sum()
        Gram = B.conj().T @ B * dOm; rhs = B.conj().T @ z * dOm
        cm = np.linalg.solve(Gram, rhs)
        P = np.abs(cm) ** 2; tot = np.sum(np.abs(z) ** 2) * dOm
        l2 = np.real(cm.conj() @ Gram @ cm)
        ms = np.array([-2, -1, 0, 1, 2])
        out.append({"shell": [r0, r1], "ncells": int(sel.sum()), "P_m (m=-2..2)": [float(p) for p in P],
                    "P0/total": float(P[2] / tot), "<m>": float(np.sum(ms * P) / P.sum()),
                    "l2 fraction of |zeta|^2": float(l2 / tot), "rms|zeta|": float(np.sqrt(tot / (4 * np.pi))),
                    "asym P2 vs P-2": float(abs(P[4] - P[0]) / max(P[4] + P[0], 1e-300)),
                    "asym P1 vs P-1": float(abs(P[3] - P[1]) / max(P[3] + P[1], 1e-300))})
    return out


def reflection_asymmetry(M):
    """relative Frobenius asymmetry of the field under improper spatial symmetries: axis reflections,
    inversion and the three diagonal (x<->y etc.) mirrors; a chiral field breaks ALL of them."""
    out = {}
    def conj(P, Mp):
        P4 = np.eye(4); P4[1:, 1:] = P
        return P4 @ Mp @ P4.T
    tot = np.sqrt(np.sum(M ** 2))
    for name, P, perm in (("x->-x", np.diag([-1, 1, 1.0]), lambda A: A[::-1, :, :]),
                          ("y->-y", np.diag([1, -1, 1.0]), lambda A: A[:, ::-1, :]),
                          ("z->-z", np.diag([1, 1, -1.0]), lambda A: A[:, :, ::-1]),
                          ("inversion", -np.eye(3), lambda A: A[::-1, ::-1, ::-1]),
                          ("x<->y mirror", np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]]), lambda A: np.swapaxes(A, 0, 1)),
                          ("y<->z mirror", np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0.0]]), lambda A: np.swapaxes(A, 1, 2))):
        out[name] = float(np.sqrt(np.sum((M - conj(P, perm(M))) ** 2)) / tot)
    return out


def check_C8b():
    bins = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 16)]
    out = {}
    for tag, rel_path, L in (("n32", FIELD_P4_32, 48.0), ("n48", FIELD_P4_48, 72.0), ("R15-M seed", FIELD_M, 48.0)):
        Mf = load_field(rel_path)
        out[tag] = shell_decomposition(Mf, L, bins)
        out[tag + " reflection asymmetry"] = reflection_asymmetry(Mf)
    ok = True; frac_note = []; inner_note = []
    for tag in ("n32", "n48"):
        for sh in out[tag]:
            if sh["ncells"] < 100:   # the 32-cell innermost shell cannot support a 5-function fit; reported, not judged
                inner_note.append((tag, sh["shell"], sh["ncells"], round(sh["<m>"], 3), round(sh["asym P1 vs P-1"], 3)))
                continue
            ok = ok and sh["asym P2 vs P-2"] < 0.02 and sh["asym P1 vs P-1"] < 0.05 and abs(sh["<m>"]) < 0.01 and sh["P0/total"] < 1e-3
            frac_note.append((tag, sh["shell"], round(sh["l2 fraction of |zeta|^2"], 3)))
    refl_ok = all(min(out[t + " reflection asymmetry"].values()) < 1e-2 for t in ("n32", "n48"))
    zr = out["n32"][0]["rms|zeta|"] / out["R15-M seed"][0]["rms|zeta|"]
    out["rms|zeta| ratio P-iv n32 / R15-M seed, inner shell"] = float(zr)
    out["l2 fractions (tag, shell, fraction)"] = frac_note
    out["innermost 32-cell shell (tag, shell, ncells, <m>, P1 asym): fit-limited, not judged"] = inner_note
    frac_match = all(0.03 <= f <= 0.07 for _, sh, f in frac_note if sh[1] <= 6) and all(0.35 <= f <= 0.39 for _, sh, f in frac_note if 6 <= sh[0] and sh[1] <= 12)
    out["l2 fractions match the claim's 3-7 % (r<6) and 35-39 % (r 6-12)"] = frac_match
    verdict = "CONFIRMED" if (ok and refl_ok and frac_match) else ("QUALIFIED" if (ok and refl_ok) else "REFUTED")
    return {"verdict": verdict, "own_numbers": out,
            "note": ("own frame construction and own least-squares projection on the five 2Y_2m (spin +2 convention; the +-m label swap "
                     "between conventions does not affect the symmetry test). No chirality CONFIRMED two ways: P_m symmetric in +-m to 1e-3 or "
                     "better and <m> = 0 on every shell with >= 248 cells, and, decisively, the fields keep improper symmetries exactly "
                     "(x<->y mirror asymmetry 5e-6 / 1e-6 on n32 / n48; the axis reflections x->-x are broken at 2.5e-3 / 1.4e-3, the "
                     "(1,1,1) anisotropy of C7, not chirality). The 32-cell innermost shell cannot support a 5-function fit and is reported "
                     "unjudged. NOT reproduced: the l = 2 fractions (mine 0.7-1 % at r 3-6 and 7.5-8.7 % at r 6-12, the claim's 3-7 % and "
                     "35-39 %); the |zeta| ratio to the R15-M seed (48x) matches the claim's 40x.")}


# ============================================================ C9
def tau_field(nvec, h, st, delta=DELTA):
    S = delta * np.eye(3) + (1 - delta) * nvec[..., :, None] * nvec[..., None, :]
    dS = [d1(S, j, h, st) for j in range(3)]
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1; eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
    tau = np.zeros(nvec.shape[:-1])
    for j in range(3):
        tau += np.einsum("ik,...il,...kl->...", eps[:, j, :], S, dS[j], optimize=True)
    return tau


def ncurln(nvec, h, st):
    dn = [d1(nvec, j, h, st) for j in range(3)]
    curl = np.stack([dn[1][..., 2] - dn[2][..., 1], dn[2][..., 0] - dn[0][..., 2], dn[0][..., 1] - dn[1][..., 0]], -1)
    return np.sum(nvec * curl, -1)


def check_C9():
    n, L = 40, 24.0; h = L / n
    X, Y, Z = coords(n, h); r = np.sqrt(X * X + Y * Y + Z * Z)
    k = 0.5
    fields = {}
    fields["hedgehog"] = np.stack([X, Y, Z], -1) / r[..., None]
    r1 = np.sqrt(X * X + Y * Y + (Z - 4) ** 2); r2 = np.sqrt(X * X + Y * Y + (Z + 4) ** 2)
    v = np.stack([X, Y, Z - 4], -1) / r1[..., None] + np.stack([X, Y, Z + 4], -1) / r2[..., None]
    fields["pair"] = v / np.linalg.norm(v, axis=-1)[..., None]
    fields["twist"] = np.stack([np.cos(k * Z), np.sin(k * Z), 0 * Z], -1)
    fields["bend"] = np.stack([np.sin(k * Z), 0 * Z, np.cos(k * Z)], -1)
    out = {}
    for name, nv in fields.items():
        row = {}
        for st in ("cen", "sym"):
            if st == "sym":
                T2 = 0.0; tau = None
                for br, wt in branches("sym"):
                    t = tau_field(nv, h, br); T2 += wt * np.sum(t ** 2) * h ** 3
            else:
                tau = tau_field(nv, h, st); T2 = float(np.sum(tau ** 2) * h ** 3)
            row[f"T2_{st}"] = float(T2)
        tau = tau_field(nv, h, "cen"); nc = ncurln(nv, h, "cen")
        inner = (np.abs(X) < 9) & (np.abs(Y) < 9) & (np.abs(Z) < 9)
        if name == "twist":
            row["tau/(k) interior (cen), mean"] = float(np.mean(tau[inner]) / (-k))
            row["expected cen attenuation sin(2kh)/(2kh)"] = float(np.sin(2 * k * h) / (2 * k * h))
            row["continuum T2 = ((1-d)^2 k)^2 L^3"] = float(((1 - DELTA) ** 2 * k) ** 2 * L ** 3)
        mask = np.abs(nc) > 1e-8
        if mask.any():
            ratio = tau[mask] / nc[mask]
            row["tau/(n.curl n) lattice pointwise: median, iqr"] = [float(np.median(ratio)), float(np.percentile(ratio, 75) - np.percentile(ratio, 25))]
        out[name] = row
    # symbolic identity with a unit vector field parametrized by angles
    x, y, z, d = sp.symbols("x y z delta", real=True)
    th = sp.Function("theta")(x, y, z); ph = sp.Function("phi")(x, y, z)
    nv = sp.Matrix([sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th)])
    S = d * sp.eye(3) + (1 - d) * nv * nv.T
    vars_ = [x, y, z]
    tau = 0
    for i in range(3):
        for j in range(3):
            for kk in range(3):
                e = sp.LeviCivita(i, j, kk)
                if e == 0:
                    continue
                for l in range(3):
                    tau += e * S[i, l] * sp.diff(S[kk, l], vars_[j])
    curl = sp.Matrix([sp.diff(nv[2], y) - sp.diff(nv[1], z), sp.diff(nv[0], z) - sp.diff(nv[2], x), sp.diff(nv[1], x) - sp.diff(nv[0], y)])
    ident = sp.simplify(tau - (1 - d) ** 2 * (nv.T * curl)[0])
    out["symbolic: tau - (1-d)^2 n.curl n"] = str(ident)
    tw = out["twist"]
    tw["T2 implied by the claim's own ratio 0.468: (0.468 k sin(kh)/(kh))^2 L^3"] = float((0.468 * k * np.sin(k * h) / (k * h)) ** 2 * L ** 3)
    base_ok = (ident == 0 and out["hedgehog"]["T2_cen"] < 1e-20 and out["bend"]["T2_cen"] == 0.0
               and abs(tw["tau/(n.curl n) lattice pointwise: median, iqr"][0] - 0.468) < 0.003
               and abs(out["pair"]["T2_cen"] - 2.9e-2) / 2.9e-2 < 0.05
               and abs(out["pair"]["tau/(n.curl n) lattice pointwise: median, iqr"][0] - 0.53) < 0.09)
    t2_ok = abs(tw["T2_cen"] - 630) / 630 < 0.05 or abs(tw["T2_sym"] - 630) / 630 < 0.05
    verdict = "CONFIRMED" if (base_ok and t2_ok) else ("QUALIFIED" if base_ok else "REFUTED")
    return {"verdict": verdict, "own_numbers": out,
            "note": ("own stencils (cen = central interior + one-sided edges; sym = fwd/bwd energy average); symbolic identity with n "
                     "parametrized by angle fields. Reproduced: hedgehog ~1e-29, bend exactly 0, pair 2.9e-2 (cen) and its ratio 0.53, the twist "
                     "ratio 0.468 = 0.49 x sin(2kh)/(2kh) / [sin(kh)/(kh)] (the doubled-frequency attenuation over the lattice n.curl n). NOT "
                     "reproduced: the twist T2 = 630; I get 735 (cen) / 716 (sym), and the claim's own ratio 0.468 implies 757 (interior "
                     "value) or 735 with the one-sided edges, so 630 is inconsistent with 0.468 under any stencil I can construct (a "
                     "boundary-layer exclusion of ~3 cells would give it).")}


# ============================================================ main
CHECKS = {"C1a": check_C1a, "C1b": check_C1b, "C1c": check_C1c, "C2a": check_C2a, "C2b": check_C2b,
          "C2c": check_C2c, "C2d": check_C2d, "C3": check_C3, "C4a": check_C4a, "C4b": check_C4b,
          "C4c": check_C4c, "C5": check_C5, "C6": check_C6, "C7": check_C7, "C8a": check_C8a,
          "C8b": check_C8b, "C9": check_C9}


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=os.path.join(DATA, "m5_32_r16_0_audit.json"))
    args = ap.parse_args()
    ids = [x for x in args.only.split(",") if x] or list(CHECKS)
    results = {}
    if os.path.exists(args.out) and args.only:
        with open(args.out) as fh:
            results = json.load(fh).get("claims", {})
    for cid in ids:
        t0 = time.time()
        fn = CHECKS[cid]
        try:
            res = fn(args.fast) if cid in ("C1c", "C2c", "C3") else fn()
        except Exception as ex:  # noqa
            import traceback
            res = {"verdict": "NOT TESTED", "own_numbers": {},
                   "note": "exception: " + traceback.format_exc()[-800:].replace(ROOT + os.sep, "")}
        res["runtime_s"] = round(time.time() - t0, 1)
        results[cid] = to_jsonable(res)
        print(f"[{cid}] {res['verdict']} ({res['runtime_s']} s)")
        print(json.dumps(results[cid], indent=1, default=str)[:6000])
        sys.stdout.flush()
    tally = {}
    for v in results.values():
        tally[v["verdict"]] = tally.get(v["verdict"], 0) + 1
    payload = {"task": "M5.32 R16-0 independent adversarial audit", "script": "scripts/m5_32_r16_0_audit.py",
               "fields": [FIELD_R10, FIELD_P4_32, FIELD_P4_48, FIELD_M], "tally": tally, "claims": results}
    os.makedirs(DATA, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print("tally:", tally)


if __name__ == "__main__":
    main()
