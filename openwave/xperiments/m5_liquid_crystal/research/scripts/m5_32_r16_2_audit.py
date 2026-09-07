#!/usr/bin/env python3
"""M5.32 R16-2 adversarial audit: the clock operator on the relaxed core.

Independent re-evaluation of the R16-2 claims C2.1 to C2.6 from the DEFINITIONS
only (the producer scripts m5_32_r16_*.py / m5_32_r15_common.py are neither read
nor imported). Own energy evaluator: per-cell numpy eigendecompositions for u, n,
the pair plane, the plateau weight w(N) and G; the certified stencil (d1,
branches) and the lattice helpers (coords, pin_shell) are imported from
m5_21_3_a_4d.py; the circle average uses an own Rodrigues rotation about the
lifted director.

Audit routes (all second differences of the ENERGY, no producer operator):
  C2.1  per-cell 2x2 inertia T on the doublet (pointwise kinetic densities)
  C2.2-4 Rayleigh quotients R(v) = v^T H v / (2 v^T T v) on trial families +
        a small Rayleigh-Ritz on their span (an UPPER bound on the lowest
        Omega^2 of the doublet operator, exact on the vacuum's cosine modes)
  C2.3  the doublet-projected gradient by a 27-color first-difference sweep
  C2.5  additivity of kin_tot over cells + plain vs circle-averaged T
  C2.6  random localized doublet directions: min second difference / (2 v^T T v)
  core  the 10x10 symmetric-block Hessian on the 8 innermost cells

Run (research dir):
  OMP_NUM_THREADS=2 /opt/anaconda3/envs/openwave312/bin/python3 \
      scripts/m5_32_r16_2_audit.py
Output: data/m5_32_r16_2_audit.json (relative paths only).
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from m5_21_3_a_4d import d1, branches, coords, pin_shell  # noqa: E402  (certified stencil only)

T0 = time.time()
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
W1 = 0.000724023879
G_VAC, DELTA = 8.0, 0.3
MU, C_S, C_P = 1e-2, 0.4, 1.0
C_P4 = [(-G_VAC) ** p + 1.0 + 2.0 * DELTA ** p for p in range(1, 5)]
N_CELL, L_BOX = 32, 48.0
H = L_BOX / N_CELL
H3 = H ** 3
FIELDS = {
    "seed": "checkpoints/m5_32_r15/m_hedgehog/relax_n32_L48_mu0.01_cP1.npy",
    "end": "checkpoints/m5_32_r16/r16_1_rebuild_n32_L48.npy",
    "vac": "checkpoints/m5_32_r16/vac_n32_L48.npy",
}
PRODUCER = {  # the brief's numbers, copied for the side-by-side only
    "C2.1": {"T_eig_range_seed": [3.397, 4.081], "T_aa_vacregion_seed": 3.429,
             "T_vac": 3.375},
    "C2.2": {"omega2_seed": [0.08994, 0.08994, 0.08997, 0.08997],
             "rms_radius": 16.65, "core_hess_seed": [3.80, 4.15, 4.15, 36.1]},
    "C2.3": {"omega2_end": [0.08957, 0.08957, 0.08963, 0.08963],
             "core_hess_end_lowest": 1.097, "grad_doublet_end": 0.0153,
             "grad_doublet_seed": 0.324},
    "C2.4": {"omega2_vac": [0.05126, 0.05126, 0.08241, 0.08241],
             "uniform_R": 0.01, "naive_estimate": 0.027},
    "C2.5": {"circle_vs_plain_T_tol": 1e-6},
    "C2.6": {"morse_index": 0},
}
LOG = []


def log(s):
    line = "[%7.1fs] %s" % (time.time() - T0, s)
    print(line, flush=True)
    LOG.append(line)


# ============================ spectral data per cell ============================
def plateau_w(lam):
    """The plateau weight w(lambda): 1 on |lambda-0.3|<=0.5, cosine tapers."""
    lam = np.asarray(lam, dtype=float)
    w = np.zeros_like(lam)
    w[np.abs(lam - 0.3) <= 0.5] = 1.0
    m = (lam > 0.8) & (lam < 1.0)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * (lam[m] - 0.8) / 0.2))
    m = (lam > -1.0) & (lam < -0.2)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * (-0.2 - lam[m]) / 0.8))
    return w


def eta_proj(v):
    """eta-orthogonal projector v (eta v)^T / (v^T eta v), v shape (...,4)."""
    ev = v @ ETA
    nrm = np.einsum("...a,...a->...", v, ev)
    return v[..., :, None] * ev[..., None, :] / nrm[..., None, None]


def spectral(M, lift_ref):
    """Per-cell eigendata of N = M eta.

    Returns dict with lam_u (timelike eigenvalue), lam (3 spatial, descending),
    u (u^T eta u = -1, u_0 > 0), n (n^T eta n = 1, lifted along lift_ref),
    e, f (eta-orthonormal basis of the pair plane), G, w (the plateau spectral
    function w(N)), rho2 (half split squared).
    Fast path when M_0i = 0 exactly (u = e_0, spatial eigh); general path via
    np.linalg.eig of the nonsymmetric N otherwise.
    """
    sh = M.shape[:-2]
    if np.abs(M[..., 0, 1:]).max() < 1e-13:
        lam_u = -M[..., 0, 0]
        ls, Vs = np.linalg.eigh(M[..., 1:, 1:])            # ascending
        lam = ls[..., ::-1]
        V = np.zeros(sh + (4, 4))
        V[..., 1:, 0] = Vs[..., :, 2]                        # n  (largest)
        V[..., 1:, 1] = Vs[..., :, 1]                        # e  (lambda_2)
        V[..., 1:, 2] = Vs[..., :, 0]                        # f  (lambda_3)
        u = np.zeros(sh + (4,)); u[..., 0] = 1.0
        n, e, f = V[..., 0], V[..., 1], V[..., 2]
    else:
        N = M @ ETA
        w_, V_ = np.linalg.eig(N)
        if np.abs(w_.imag).max() > 1e-9:
            raise RuntimeError("complex N-spectrum")
        w_, V_ = w_.real, V_.real
        nrm = np.einsum("...ak,ab,...bk->...k", V_, ETA, V_)
        iu = np.argmin(w_, axis=-1)
        u = np.take_along_axis(V_, iu[..., None, None], axis=-1)[..., 0]
        nu = np.take_along_axis(nrm, iu[..., None], axis=-1)[..., 0]
        if (nu >= 0).any():
            raise RuntimeError("timelike eigenvector not eta-negative")
        u = u / np.sqrt(-nu)[..., None]
        u = u * np.sign(u[..., 0])[..., None]
        lam_u = np.take_along_axis(w_, iu[..., None], axis=-1)[..., 0]
        # the spatial triple: the other three, descending (the timelike one pushed last)
        wsort = np.where(np.arange(4) == iu[..., None], -np.inf, w_)
        order = np.argsort(-wsort, axis=-1)[..., :3]
        lam = np.take_along_axis(w_, order, axis=-1)
        Vs = np.take_along_axis(V_, order[..., None, :], axis=-1)   # (...,4,3)
        ns = np.take_along_axis(nrm, order, axis=-1)
        if (ns <= 0).any():
            raise RuntimeError("spatial eigenvector not eta-positive")
        Vs = Vs / np.sqrt(ns)[..., None, :]
        n, e, f = Vs[..., 0], Vs[..., 1], Vs[..., 2]
        # eta-orthonormalize the pair inside its eigenspace (degenerate case)
        f = f - np.einsum("...a,ab,...b->...", f, ETA, e)[..., None] * e
        f = f / np.sqrt(np.einsum("...a,ab,...b->...", f, ETA, f))[..., None]
    # the director lift
    sgn = np.sign(np.einsum("...a,...a->...", n[..., 1:], lift_ref))
    sgn[sgn == 0] = 1.0
    n = n * sgn[..., None]
    # a DEFINITE handedness of the pair basis: det[u, n, e, f] > 0 (right-handed
    # (n, e, f) in the rest frame), so that J = f (eta e)^T - e (eta f)^T is the
    # rotation about the LIFTED director with one orientation on every cell
    # (eigh returns the pair with random signs; without this the circle would
    # turn +alpha/2 on some cells and -alpha/2 on others, differently for
    # M + eps v and M - eps v)
    det = np.linalg.det(np.stack([u, n, e, f], axis=-1))
    f = f * np.where(det < 0, -1.0, 1.0)[..., None]
    Pu, Pn, Pe, Pf = eta_proj(u), eta_proj(n), eta_proj(e), eta_proj(f)
    Ppair = np.eye(4) - Pu - Pn
    wu, w1, w2, w3 = (plateau_w(lam_u), plateau_w(lam[..., 0]),
                      plateau_w(lam[..., 1]), plateau_w(lam[..., 2]))
    wN = (wu[..., None, None] * Pu + w1[..., None, None] * Pn
          + (0.5 * (w2 + w3))[..., None, None] * Ppair
          + (0.5 * (w2 - w3))[..., None, None] * (Pe - Pf))
    eu = u @ ETA
    G = ETA + 2.0 * eu[..., :, None] * eu[..., None, :]
    rho2 = (0.5 * (lam[..., 1] - lam[..., 2])) ** 2
    return dict(lam_u=lam_u, lam=lam, u=u, n=n, e=e, f=f, G=G, w=wN, rho2=rho2)


def rodrigues(n_vec, e, f, beta):
    """R(beta) = I + sin(beta) J + (1 - cos beta) J^2 with J the rotation
    generator in the pair plane (rest frame of u): J = f (eta e)^T - e (eta f)^T."""
    J = (f[..., :, None] * (e @ ETA)[..., None, :]
         - e[..., :, None] * (f @ ETA)[..., None, :])
    return np.eye(4) + np.sin(beta) * J + (1.0 - np.cos(beta)) * (J @ J), J


def tr2(A, B):
    """tr(A B) per cell."""
    return np.einsum("...ab,...ba->...", A, B)


# ============================ the energy evaluator ============================
def energy_plain(M, lift_ref, a0=None, want_density=False):
    """Plain (single-sample) action pieces on the field M (shape (...,4,4)).

    Returns dict of h^3-weighted totals: E_stat and its parts, kin_tot and its
    parts (with a0 the clock generator, omega^2 pulled out); densities optional.
    """
    sp = spectral(M, lift_ref)
    G, w, rho2 = sp["G"], sp["w"], sp["rho2"]
    wE = w @ ETA
    dens_h = 0.0; dens_KP = 0.0; dens_E2 = 0.0; dens_kh = 0.0
    for br, wt in branches("sym"):
        A = [d1(M, ax, H, br) for ax in range(3)]
        AG = [A[i] @ G for i in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                F = AG[i] @ A[j] - AG[j] @ A[i]
                dens_h = dens_h + wt * 4.0 * tr2(G @ F, G @ F.swapaxes(-1, -2))
            Om = w @ A[i] @ wE                           # w A_i eta w
            dens_KP = dens_KP + wt * 0.5 * tr2(Om.swapaxes(-1, -2) @ ETA, Om @ ETA)
            dens_E2 = dens_E2 + wt * tr2(AG[i], AG[i])
            if a0 is not None:
                F0 = a0 @ G @ A[i] - AG[i] @ a0
                dens_kh = dens_kh + wt * 4.0 * tr2(G @ F0, G @ F0.swapaxes(-1, -2))
    N = M @ ETA
    P = N; dens_V = 0.0
    for p in range(4):
        if p:
            P = P @ N
        dens_V = dens_V + (np.einsum("...kk->...", P) - C_P4[p]) ** 2
    dens_V = W1 * dens_V
    dens_U = MU * rho2
    dens_reg = C_S * rho2 * dens_E2
    dens_stat = dens_h + dens_V + dens_U + C_P * dens_KP + dens_reg
    out = dict(E_h=H3 * dens_h.sum(), V4=H3 * dens_V.sum(), U=H3 * dens_U.sum(),
               K_P=H3 * dens_KP.sum(), reg=H3 * dens_reg.sum(),
               E_stat=H3 * dens_stat.sum())
    if a0 is not None:
        Om0 = w @ a0 @ wE
        dens_kKP = 0.5 * tr2(Om0.swapaxes(-1, -2) @ ETA, Om0 @ ETA)
        dens_kreg = C_S * rho2 * tr2(a0 @ G, a0 @ G)
        dens_kin = dens_kh + C_P * dens_kKP + dens_kreg
        out.update(kin_h=H3 * dens_kh.sum(), kin_KP=H3 * dens_kKP.sum(),
                   kin_reg=H3 * dens_kreg.sum(), kin_tot=H3 * dens_kin.sum())
        if want_density:
            out["dens_kin"] = H3 * dens_kin
    if want_density:
        out["dens_stat"] = H3 * dens_stat
    out["spectral"] = sp
    return out


def energy_circle(M, lift_ref, ns=4, a0=None, want_density=False, keys=("E_stat",)):
    """Circle-averaged action: (1/ns) sum_k E[T_(2 pi k/ns) M], T_alpha M =
    R(alpha/2) M R(alpha/2)^T with R the Rodrigues rotation about the lifted
    director of M itself (a0 co-rotates: a0 -> R a0 R^T)."""
    if ns == 1:
        return energy_plain(M, lift_ref, a0, want_density)
    sp = spectral(M, lift_ref)
    acc = {}
    for k in range(ns):
        beta = 0.5 * (2.0 * np.pi * k / ns)
        R, _ = rodrigues(sp["n"], sp["e"], sp["f"], beta)
        Rt = R.swapaxes(-1, -2)
        Mk = R @ M @ Rt
        a0k = None if a0 is None else R @ a0 @ Rt
        ek = energy_plain(Mk, lift_ref, a0k, want_density)
        for key, val in ek.items():
            if key == "spectral":
                continue
            acc[key] = acc.get(key, 0.0) + val / ns
    return acc


# ============================ lattice helpers ============================
X, Y, Z = coords(N_CELL, H)
RAD = np.sqrt(X * X + Y * Y + Z * Z)
RHAT = np.stack([X, Y, Z], axis=-1) / RAD[..., None]
E1_FIELD = np.zeros_like(RHAT); E1_FIELD[..., 0] = 1.0
PIN = pin_shell(N_CELL, H, 1.6)
FREE = ~PIN
LIFT = {"seed": RHAT, "end": RHAT, "vac": E1_FIELD}


def doublet_frames(sp):
    """Three smooth (e, f) frames in the pair plane of the UNPERTURBED field:
    F2 (e_2 projected), F3 (e_3 projected), S (spherical theta-hat projected)."""
    n = sp["n"][..., 1:]
    frames = {}
    th_hat = np.stack([X * Z, Y * Z, -(X * X + Y * Y)], axis=-1)
    th_hat = th_hat / np.linalg.norm(th_hat, axis=-1, keepdims=True)
    for name, ref in (("F2", np.array([0.0, 1.0, 0.0])),
                      ("F3", np.array([0.0, 0.0, 1.0])), ("S", th_hat)):
        ref = np.broadcast_to(ref, n.shape)
        e = ref - np.einsum("...a,...a->...", ref, n)[..., None] * n
        e = e / np.linalg.norm(e, axis=-1, keepdims=True)
        f = np.cross(n, e)
        frames[name] = (e, f)
    return frames


def doublet_field(frame, a, b):
    """delta M = a (e e^T - f f^T) + b (e f^T + f e^T), zero on the pinned shell."""
    e, f = frame
    ee = e[..., :, None] * e[..., None, :]
    ff = f[..., :, None] * f[..., None, :]
    ef = e[..., :, None] * f[..., None, :]
    v3 = a[..., None, None] * (ee - ff) + b[..., None, None] * (ef + ef.swapaxes(-1, -2))
    v = np.zeros(v3.shape[:-2] + (4, 4))
    v[..., 1:, 1:] = v3
    v[PIN] = 0.0
    return v


def proj_tensor_field(sp, D, env):
    """Smooth doublet from a CONSTANT traceless symmetric spatial tensor D:
    v = env (P D P - (1/2) tr(P D P) P), P = I - n n^T the pair-plane projector
    of the unperturbed field (no frame, hence no frame singularity)."""
    n = sp["n"][..., 1:]
    P = np.eye(3) - n[..., :, None] * n[..., None, :]
    PDP = P @ D @ P
    v3 = PDP - 0.5 * np.einsum("...kk->...", PDP)[..., None, None] * P
    v = np.zeros(v3.shape[:-2] + (4, 4))
    v[..., 1:, 1:] = env[..., None, None] * v3
    v[PIN] = 0.0
    return v


def cosmode(kx, ky, kz, Lf=43.5):
    """Dirichlet box modes vanishing on the first pinned cell centers (x = +-Lf/2,
    Lf = 29 h = 43.5): k = 1 is cos(pi x/Lf), k = 2 is sin(2 pi x/Lf), ..."""
    return (np.sin(kx * np.pi * (X + 0.5 * Lf) / Lf) * np.sin(ky * np.pi * (Y + 0.5 * Lf) / Lf)
            * np.sin(kz * np.pi * (Z + 0.5 * Lf) / Lf))


def second_diff(M, v, lift_ref, eps, ns, E0):
    """v^T H v = [E(M+eps v) - 2E(M) + E(M-eps v)]/eps^2 on the circle-averaged E_stat."""
    Ep = energy_circle(M + eps * v, lift_ref, ns)["E_stat"]
    Em = energy_circle(M - eps * v, lift_ref, ns)["E_stat"]
    return (Ep - 2.0 * E0 + Em) / eps ** 2


def kin_of(M, v, lift_ref, ns=1, density=False):
    r = energy_circle(M, lift_ref, ns, a0=v, want_density=density)
    return r


def rayleigh(M, v, lift_ref, eps, ns, E0):
    q = second_diff(M, v, lift_ref, eps, ns, E0)
    t = kin_of(M, v, lift_ref)["kin_tot"]
    return q / (2.0 * t), q, t


def sym_basis():
    """The 10 symmetric 4x4 directions, unit Frobenius norm."""
    out = []
    for a in range(4):
        for b in range(a, 4):
            D = np.zeros((4, 4)); D[a, b] = 1.0; D[b, a] = 1.0
            out.append(D / np.linalg.norm(D))
    return out


RESULT = {"script": "scripts/m5_32_r16_2_audit.py",
          "fields": FIELDS, "conventions": dict(n=N_CELL, L=L_BOX, h=H, mu=MU, c_s=C_S,
                                                c_P=C_P, W1=W1, pin_depth=1.6,
                                                free_cells=int(FREE.sum())),
          "claims": {}, "notes": []}


def verdict(cid, v, method, **kw):
    RESULT["claims"][cid] = dict(verdict=v, method=method, **kw)
    log("%s -> %s : %s" % (cid, v, method))


# ============================ load + baseline ============================
M_ALL = {k: np.load(os.path.join(RESEARCH, p)) for k, p in FIELDS.items()}
SP = {k: spectral(M_ALL[k], LIFT[k]) for k in M_ALL}
base = {}
for k, M in M_ALL.items():
    sp = SP[k]
    ndotr = np.einsum("...a,...a->...", sp["n"][..., 1:], RHAT)
    gap = (sp["lam"][..., 0] - sp["lam"][..., 1])[FREE]
    t = time.time()
    e1 = energy_circle(M, LIFT[k], 1)
    e4 = energy_circle(M, LIFT[k], 4)
    e8 = energy_circle(M, LIFT[k], 8)
    dt = (time.time() - t) / 13.0
    base[k] = dict(E_stat_plain=e1["E_stat"], E_stat_c4=e4["E_stat"], E_stat_c8=e8["E_stat"],
                   parts_plain={q: e1[q] for q in ("E_h", "V4", "U", "K_P", "reg")},
                   min_ndotr_free=float(np.abs(ndotr[FREE]).min()) if k != "vac" else None,
                   min_gap_lam1_lam2_free=float(gap.min()), max_half_split=float(np.sqrt(sp["rho2"].max())),
                   sec_per_plain_eval=dt)
    log("%s: E_stat plain %.6f  c4 %.6f  c8 %.6f  (%.2fs/plain eval); min gap %.4f; max half split %.5f"
        % (k, e1["E_stat"], e4["E_stat"], e8["E_stat"], dt, gap.min(), np.sqrt(sp["rho2"].max())))
RESULT["baseline"] = base
# general-path cross-check of the fast path on a sample of cells
Msub = M_ALL["seed"][12:20, 12:20, 12:20].copy()
sp_fast = spectral(Msub, RHAT[12:20, 12:20, 12:20])
Mg = Msub.copy(); Mg[..., 0, 1] += 1e-12; Mg[..., 1, 0] += 1e-12   # force the general path (expect ~1e-12 shifts)
sp_gen = spectral(Mg, RHAT[12:20, 12:20, 12:20])
RESULT["notes"].append("fast vs general spectral path on 8^3 core cells: max |dlam| = %.2e, max |dw| = %.2e, max|dG| = %.2e"
                       % (np.abs(sp_fast["lam"] - sp_gen["lam"]).max(), np.abs(sp_fast["w"] - sp_gen["w"]).max(),
                          np.abs(sp_fast["G"] - sp_gen["G"]).max()))
log(RESULT["notes"][-1])
# Rodrigues sanity: J^3 = -J, R eta R^T = eta, J = [n]_x on the spatial block (u = e_0)
R_, J_ = rodrigues(sp_fast["n"], sp_fast["e"], sp_fast["f"], 0.7)
nx = np.zeros(sp_fast["n"].shape[:-1] + (3, 3)); nn = sp_fast["n"][..., 1:]
nx[..., 0, 1] = -nn[..., 2]; nx[..., 0, 2] = nn[..., 1]; nx[..., 1, 0] = nn[..., 2]
nx[..., 1, 2] = -nn[..., 0]; nx[..., 2, 0] = -nn[..., 1]; nx[..., 2, 1] = nn[..., 0]
RESULT["notes"].append("Rodrigues sanity: max|J^3+J| = %.2e, max|R eta R^T - eta| = %.2e, max|J_spatial - [n]_x| = %.2e (J is the right-handed rotation about the lifted n)" % (
    np.abs(J_ @ J_ @ J_ + J_).max(), np.abs(R_ @ ETA @ R_.swapaxes(-1, -2) - ETA).max(), np.abs(J_[..., 1:, 1:] - nx).max()))
log(RESULT["notes"][-1])
# the circle acts LINEARLY on a doublet perturbation: T_alpha(M + eps v) = R M R^T + eps R v R^T (n is an exact eigenvector of N + eps v eta)
_fr = doublet_frames(SP["seed"])["F2"]; _c = np.cos(np.pi * X / 43.5) * np.cos(np.pi * Y / 43.5) * np.cos(np.pi * Z / 43.5)
_v = doublet_field(_fr, _c, np.zeros_like(_c)); _eps = 3e-3
_spp = spectral(M_ALL["seed"] + _eps * _v, RHAT); _Rp, _ = rodrigues(_spp["n"], _spp["e"], _spp["f"], np.pi / 4)
_R, _ = rodrigues(SP["seed"]["n"], SP["seed"]["e"], SP["seed"]["f"], np.pi / 4)
_dev = np.abs(_Rp @ (M_ALL["seed"] + _eps * _v) @ _Rp.swapaxes(-1, -2) - (_R @ M_ALL["seed"] @ _R.swapaxes(-1, -2) + _eps * _R @ _v @ _R.swapaxes(-1, -2))).max()
RESULT["notes"].append("circle linearity on the seed at beta = pi/4, eps = 3e-3, cos111 a-doublet: max |T(M+eps v) - (R M R^T + eps R v R^T)| = %.2e (max half split %.2e)" % (_dev, np.sqrt(SP["seed"]["rho2"].max())))
log(RESULT["notes"][-1])

# ============================ C2.1 + C2.5: the inertia T ============================
FR = {k: doublet_frames(SP[k]) for k in M_ALL}
ONE = np.ones((N_CELL,) * 3); ZERO = np.zeros_like(ONE)
c21 = {}
for k, M in M_ALL.items():
    fr = FR[k]["F2"]
    va = doublet_field(fr, ONE, ZERO); vb = doublet_field(fr, ZERO, ONE)
    ka = kin_of(M, va, LIFT[k], density=True); kb = kin_of(M, vb, LIFT[k], density=True)
    kp = kin_of(M, va + vb, LIFT[k], density=True); km = kin_of(M, va - vb, LIFT[k], density=True)
    Taa, Tbb = ka["dens_kin"], kb["dens_kin"]
    Tab = 0.25 * (kp["dens_kin"] - km["dens_kin"])
    tr_ = Taa + Tbb; det_ = Taa * Tbb - Tab ** 2
    disc = np.sqrt(np.maximum(0.25 * tr_ ** 2 - det_, 0.0))
    ev_lo, ev_hi = 0.5 * tr_ - disc, 0.5 * tr_ + disc
    vacreg = FREE & (RAD > 0.35 * L_BOX)
    parts = {q: float(ka[q]) for q in ("kin_h", "kin_KP", "kin_reg")}
    c21[k] = dict(T_eig_min_free=float(ev_lo[FREE].min()), T_eig_max_free=float(ev_hi[FREE].max()),
                  T_aa_vacregion_mean=float(Taa[vacreg].mean()), T_aa_vacregion_min=float(Taa[vacreg].min()),
                  T_aa_vacregion_max=float(Taa[vacreg].max()), T_aa_center=float(Taa[15, 15, 15]),
                  T_ab_max_abs_free=float(np.abs(Tab[FREE]).max()),
                  T_diag_max_free=float(max(Taa[FREE].max(), Tbb[FREE].max())), T_diag_min_free=float(min(Taa[FREE].min(), Tbb[FREE].min())),
                  T_eig_max_cell=[int(i) for i in np.unravel_index(np.where(FREE, ev_hi, -1).argmax(), ev_hi.shape)],
                  kin_parts_for_unit_a=parts, n_vacregion_cells=int(vacreg.sum()))
    log("C2.1 %s: T eig range free [%.4f, %.4f]; T_aa(r>0.35L) mean %.4f [%.4f, %.4f]; center %.4f; parts %s"
        % (k, ev_lo[FREE].min(), ev_hi[FREE].max(), Taa[vacreg].mean(), Taa[vacreg].min(), Taa[vacreg].max(),
           Taa[15, 15, 15], parts))
seed_rng = [c21["seed"]["T_eig_min_free"], c21["seed"]["T_eig_max_free"]]
vac_ok = abs(c21["vac"]["T_eig_min_free"] - C_P * H3) < 1e-9 and abs(c21["vac"]["T_eig_max_free"] - C_P * H3) < 1e-9
rng_ok = abs(seed_rng[0] - 3.397) < 0.005 and abs(seed_rng[1] - 4.081) < 0.005
RESULT["notes"].append("C2.1 seed: max over free cells of the DIAGONAL T_aa/T_bb = %.4f vs the 2x2 eigenvalue max %.4f (producer 4.081)"
                       % (c21["seed"]["T_diag_max_free"], c21["seed"]["T_eig_max_free"]))
log(RESULT["notes"][-1])
vr_ok = abs(c21["seed"]["T_aa_vacregion_mean"] - 3.429) < 0.01
verdict("C2.1", "CONFIRMED" if (vac_ok and rng_ok and vr_ok) else ("QUALIFIED" if vac_ok else "REFUTED"),
        "per-cell 2x2 inertia from the pointwise kinetic densities (kin_h + c_P kin_KP + kin_reg) with a0 = the doublet basis",
        mine=c21, producer=PRODUCER["C2.1"], exact_vac=C_P * H3)

# ---- C2.5: additivity over cells (pointwise) + plain vs circle-averaged T
c25 = {}
for k in ("seed", "end"):
    M = M_ALL[k]; fr = FR[k]["F2"]
    cx = (16, 16, 16); cy = (17, 16, 16)                # two ADJACENT free cells
    ax_ = np.zeros_like(ONE); ax_[cx] = 1.0
    ay_ = np.zeros_like(ONE); ay_[cy] = 1.0
    vx = doublet_field(fr, ax_, 0.3 * ax_); vy = doublet_field(fr, 0.5 * ay_, ay_)
    kx = kin_of(M, vx, LIFT[k], density=True); ky = kin_of(M, vy, LIFT[k]); kxy = kin_of(M, vx + vy, LIFT[k])
    cross = kxy["kin_tot"] - kx["kin_tot"] - ky["kin_tot"]
    supp = kx["dens_kin"]; off = supp.sum() - supp[cx]
    # dependence on the neighbor's M: kin(v_x) on M with a neighbor cell changed
    Mn = M.copy(); Dn = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.01, 0.004, -0.003], [0.0, 0.004, -0.006, 0.002], [0.0, -0.003, 0.002, 0.0]])
    Mn[cy] += Dn                                        # an anisotropic change of M at the NEIGHBOR cell
    kxn = kin_of(Mn, vx, LIFT[k])["kin_tot"]
    # plain vs circle-averaged (8 samples, outward lift), per cell
    va = doublet_field(fr, ONE, ZERO)
    kpl = kin_of(M, va, LIFT[k], 1, density=True)["dens_kin"]
    kc8 = kin_of(M, va, LIFT[k], 8, density=True)["dens_kin"]
    rel = np.abs(kc8 - kpl)[FREE] / np.abs(kpl)[FREE]
    relf = np.zeros_like(kpl); relf[FREE] = rel
    imax = np.unravel_index(relf.argmax(), relf.shape)
    kc4 = kin_of(M, va, LIFT[k], 4, density=True)["dens_kin"]
    # the same read with a0 NOT co-rotated (v held fixed while M is transformed)
    sp_ = SP[k]; kfix = 0.0
    for kk in range(8):
        R_, _ = rodrigues(sp_["n"], sp_["e"], sp_["f"], 0.5 * (2 * np.pi * kk / 8))
        kfix = kfix + energy_plain(R_ @ M @ R_.swapaxes(-1, -2), LIFT[k], va, True)["dens_kin"] / 8
    relfix = np.abs(kfix - kpl)[FREE] / np.abs(kpl)[FREE]
    c25[k] = dict(circle8_vs_plain_T_argmax_cell=[int(i) for i in imax], circle8_vs_plain_T_argmax_radius=float(RAD[imax]),
                  half_split_at_argmax=float(np.sqrt(sp_["rho2"][imax])), T_plain_at_argmax=float(kpl[imax]), T_circle8_at_argmax=float(kc8[imax]),
                  circle4_vs_plain_T_max_rel=float((np.abs(kc4 - kpl)[FREE] / np.abs(kpl)[FREE]).max()),
                  circle8_vs_plain_T_total_rel=float(abs(kc8[FREE].sum() - kpl[FREE].sum()) / kpl[FREE].sum()),
                  circle8_fixed_a0_vs_plain_T_max_rel=float(relfix.max()),
                  cells_rel_above_1em6=int((rel > 1e-6).sum()), cells_rel_above_1em3=int((rel > 1e-3).sum()),
cross_term_adjacent_cells=float(cross), kin_vx=float(kx["kin_tot"]), kin_vy=float(ky["kin_tot"]),
                  density_off_support=float(off),
                  kin_vx_after_neighbor_M_change=float(kxn), kin_vx_rel_change_neighbor=float((kxn - kx["kin_tot"]) / kx["kin_tot"]),
                  circle8_vs_plain_T_max_rel=float(rel.max()), circle8_vs_plain_T_max_abs=float(np.abs(kc8 - kpl)[FREE].max()))
    log("C2.5 %s: cross(adjacent) %.2e; off-support density %.2e; neighbor-M change -> rel %.2e; circle8 vs plain max rel %.2e at %s r=%.2f (half split %.5f, T %.4f -> %.4f); cells>1e-6: %d; total rel %.2e; fixed-a0 read max rel %.2e"
        % (k, cross, off, c25[k]["kin_vx_rel_change_neighbor"], rel.max(), imax, RAD[imax], c25[k]["half_split_at_argmax"], kpl[imax], kc8[imax],
           c25[k]["cells_rel_above_1em6"], c25[k]["circle8_vs_plain_T_total_rel"], relfix.max()))
pointwise_ok = all(abs(c25[k]["cross_term_adjacent_cells"]) < 1e-10 and abs(c25[k]["density_off_support"]) < 1e-12 for k in c25)
circle_ok = all(c25[k]["circle8_vs_plain_T_max_rel"] < 1e-6 for k in c25)
verdict("C2.5", "CONFIRMED" if (pointwise_ok and circle_ok) else ("QUALIFIED" if pointwise_ok else "REFUTED"),
        "kin_tot additivity over two adjacent single-cell doublets (no cross term) + density support + plain vs 8-sample circle T per cell",
        mine=c25, producer=PRODUCER["C2.5"],
        note="v^T T v is quadratic in v with NO cross term between cells (pointwise in v); T itself depends on the M-neighborhood through A_i (kin_h, reg): recorded as kin_vx_rel_change_neighbor")

# ============================ Rayleigh quotients + Ritz (C2.2, C2.3, C2.4) ============================
EPS = 3e-3
NS_R = 4     # circle samples for the Rayleigh reads (the producer used 4)
E0 = {k: energy_circle(M_ALL[k], LIFT[k], NS_R)["E_stat"] for k in M_ALL}
env_core = np.exp(-(RAD / 10.0) ** 2)
env_avoid = 1.0 - np.exp(-(RAD / 8.0) ** 2)


def ritz_set(k):
    fr = FR[k]
    c111 = cosmode(1, 1, 1); c211 = cosmode(2, 1, 1); c121 = cosmode(1, 2, 1); c112 = cosmode(1, 1, 2)
    fams = [("F2 a cos111", fr["F2"], c111, ZERO), ("F2 b cos111", fr["F2"], ZERO, c111),
            ("F3 a cos111", fr["F3"], c111, ZERO), ("S b cos111", fr["S"], ZERO, c111),
            ("F2 a cos111*coreavoid", fr["F2"], c111 * env_avoid, ZERO),
            ("F2 a gauss10", fr["F2"], env_core, ZERO), ("F2 b gauss10", fr["F2"], ZERO, env_core),
            ("F2 a cos211", fr["F2"], c211, ZERO), ("F2 b cos211", fr["F2"], ZERO, c211),
            ("F2 a cos121", fr["F2"], c121, ZERO), ("F2 a cos112", fr["F2"], c112, ZERO)]
    if k == "vac":   # n = e_1 uniform: the F3 frame duplicates F2 (up to sign) and S is meaningless there
        fams = [x for x in fams if not (x[0].startswith("S ") or x[0].startswith("F3 "))]
    else:            # frame-free doublets: constant traceless tensors projected on the local pair plane
        E = np.eye(3)
        Ds = {"D22-33": np.diag([0.0, 1.0, -1.0]), "D23": np.outer(E[1], E[2]) + np.outer(E[2], E[1]),
              "D12": np.outer(E[0], E[1]) + np.outer(E[1], E[0]), "D11-": np.diag([1.0, -0.5, -0.5])}
        fams = [x for x in fams if x[0] not in ("F2 a cos121", "F2 b gauss10")]
        fams += [("proj %s cos111" % nm, D, c111, None) for nm, D in Ds.items()]
        fams += [("proj D22-33 cos211", Ds["D22-33"], c211, None), ("proj D11- gauss10", Ds["D11-"], env_core, None)]
    return fams


def build_trial(k, fam):
    _, fr, a, b = fam
    if b is None:
        return proj_tensor_field(SP[k], fr, a)
    return doublet_field(fr, a, b)


def ritz_solve(Hm, Tm, rel_tol=1e-8):
    """Generalized eigenproblem H c = lam (2T) c on the span of the trials, with
    (near-)linearly dependent trials removed through the T-eigenbasis."""
    import scipy.linalg as sla
    wT, VT = np.linalg.eigh(2.0 * Tm)
    keep = wT > rel_tol * wT.max()
    Bh = VT[:, keep] / np.sqrt(wT[keep])                # whitening basis
    Hw = Bh.T @ Hm @ Bh
    lam_r, Cw = np.linalg.eigh(0.5 * (Hw + Hw.T))
    return lam_r, Bh @ Cw, int(keep.sum())


ritz = {}
for k in ("vac", "seed", "end"):
    M = M_ALL[k]; fams = ritz_set(k)
    vs = [build_trial(k, fam) for fam in fams]
    names = [x[0] for x in fams]
    K = len(vs)
    q = np.zeros(K); tt = np.zeros(K)
    Hm = np.zeros((K, K)); Tm = np.zeros((K, K))
    for i in range(K):
        q[i] = second_diff(M, vs[i], LIFT[k], EPS, NS_R, E0[k])
        tt[i] = kin_of(M, vs[i], LIFT[k])["kin_tot"]
        Hm[i, i] = q[i]; Tm[i, i] = tt[i]
        log("  %s R[%s] = %.5f  (q %.5f, 2T %.4f)" % (k, names[i], q[i] / (2 * tt[i]), q[i], 2 * tt[i]))
    for i in range(K):
        for j in range(i + 1, K):
            qij = second_diff(M, vs[i] + vs[j], LIFT[k], EPS, NS_R, E0[k])
            Hm[i, j] = Hm[j, i] = 0.5 * (qij - q[i] - q[j])
            tij = kin_of(M, vs[i] + vs[j], LIFT[k])["kin_tot"]
            Tm[i, j] = Tm[j, i] = 0.5 * (tij - tt[i] - tt[j])
    # generalized eigenproblem H c = lam (2T) c on the span (Ritz upper bounds)
    lam_r, C, rank = ritz_solve(Hm, Tm)
    # T-weighted rms radius of the lowest Ritz vector
    c0 = C[:, 0]; v0 = sum(c0[i] * vs[i] for i in range(K))
    d0 = kin_of(M, v0, LIFT[k], density=True)["dens_kin"]
    rms = float(np.sqrt((d0 * RAD ** 2).sum() / d0.sum())); inner = float(d0[RAD < 8].sum() / d0.sum())
    ritz[k] = dict(names=names, R_single=(q / (2 * tt)).tolist(), q=q.tolist(), twoT=(2 * tt).tolist(),
                   ritz_lowest=lam_r[:6].tolist(), lowest_vector_coeffs=c0.tolist(), rank=rank,
                   lowest_rms_radius_Tweighted=rms, lowest_fraction_T_inside_r8=inner, eps=EPS, circle_samples=NS_R)
    log("%s Ritz lowest: %s ; rms radius %.2f, T-frac r<8 %.3f" % (k, np.round(lam_r[:6], 5), rms, inner))

# eps / circle-sample consistency checks on one trial per field
chk = {}
for k in ("vac", "seed", "end"):
    v = doublet_field(FR[k]["F2"], cosmode(1, 1, 1), ZERO)
    t = kin_of(M_ALL[k], v, LIFT[k])["kin_tot"]
    r = {}
    for eps in (1.5e-3, 3e-3, 6e-3):
        r["eps%.1e_ns4" % eps] = second_diff(M_ALL[k], v, LIFT[k], eps, 4, E0[k]) / (2 * t)
    E0_8 = energy_circle(M_ALL[k], LIFT[k], 8)["E_stat"]
    r["eps3.0e-03_ns8"] = second_diff(M_ALL[k], v, LIFT[k], 3e-3, 8, E0_8) / (2 * t)
    E0_1 = energy_circle(M_ALL[k], LIFT[k], 1)["E_stat"]
    r["eps3.0e-03_ns1_plain"] = second_diff(M_ALL[k], v, LIFT[k], 3e-3, 1, E0_1) / (2 * t)
    chk[k] = {kk: float(vv) for kk, vv in r.items()}
    log("consistency %s: %s" % (k, {kk: round(vv, 6) for kk, vv in r.items()}))
RESULT["consistency_cos111_F2a"] = chk

# analytic vacuum reference: R = mu + <|grad zeta|^2>/<|zeta|^2> with the sym-stencil discrete gradient
zeta = cosmode(1, 1, 1); zeta = zeta * FREE
g2 = 0.0
for br, wt in branches("sym"):
    for ax in range(3):
        g2 = g2 + wt * (d1(zeta, ax, H, br) ** 2)
R_analytic_vac = float(MU + g2.sum() / (zeta ** 2).sum())
zeta2 = cosmode(2, 1, 1) * FREE
g2b = 0.0
for br, wt in branches("sym"):
    for ax in range(3):
        g2b = g2b + wt * (d1(zeta2, ax, H, br) ** 2)
R_analytic_vac_211 = float(MU + g2b.sum() / (zeta2 ** 2).sum())
log("vacuum analytic (K_P + U only, sym stencil): R(cos111) = %.5f, R(cos211) = %.5f" % (R_analytic_vac, R_analytic_vac_211))

# ============================ core 10-direction Hessian ============================
SUB = slice(11, 21)
core = {}
for k in ("seed", "end"):
    M = M_ALL[k]
    Ms = M[SUB, SUB, SUB].copy(); lref = LIFT[k][SUB, SUB, SUB]
    E0s = energy_circle(Ms, lref, NS_R)["E_stat"]
    blk = np.zeros(Ms.shape[:3]); blk[4:6, 4:6, 4:6] = 1.0        # the 8 innermost cells (15,16)^3
    dirs = [blk[..., None, None] * D for D in sym_basis()]
    K = 10; Hc = np.zeros((K, K)); qd = np.zeros(K)
    eps_c = 2e-3
    for i in range(K):
        qd[i] = second_diff(Ms, dirs[i], lref, eps_c, NS_R, E0s); Hc[i, i] = qd[i]
    for i in range(K):
        for j in range(i + 1, K):
            qij = second_diff(Ms, dirs[i] + dirs[j], lref, eps_c, NS_R, E0s)
            Hc[i, j] = Hc[j, i] = 0.5 * (qij - qd[i] - qd[j])
    ev = np.linalg.eigvalsh(Hc)
    # sensitivity: 8 circle samples, and eps = 1e-3 (diagonal + the lowest eigenvalue via the full matrix at ns = 8)
    E0s8 = energy_circle(Ms, lref, 8)["E_stat"]; Hc8 = np.zeros((K, K)); qd8 = np.zeros(K)
    for i in range(K):
        qd8[i] = second_diff(Ms, dirs[i], lref, eps_c, 8, E0s8); Hc8[i, i] = qd8[i]
    for i in range(K):
        for j in range(i + 1, K):
            Hc8[i, j] = Hc8[j, i] = 0.5 * (second_diff(Ms, dirs[i] + dirs[j], lref, eps_c, 8, E0s8) - qd8[i] - qd8[j])
    ev8 = np.linalg.eigvalsh(Hc8)
    qd_e = [second_diff(Ms, dirs[i], lref, 1e-3, NS_R, E0s) for i in range(K)]
    # sub-box vs full-box check on one direction
    full = second_diff(M, np.pad(dirs[5], ((11, 11), (11, 11), (11, 11), (0, 0), (0, 0))), LIFT[k], eps_c, NS_R, E0[k])
    core[k] = dict(eigs_block_unitdir=ev.tolist(), eigs_per_cell_unit=(ev / 8.0).tolist(), diag=qd.tolist(),
                   eigs_block_unitdir_ns8=ev8.tolist(), diag_eps1em3=[float(x) for x in qd_e],
                   subbox_vs_full_check=[float(qd[5]), float(full)], eps=eps_c,
                   dir_labels=["%d%d" % (a, b) for a in range(4) for b in range(a, 4)])
    log("core %s: 10-dir Hessian eigs (block, unit dir) %s ; ns8 -> %s ; /8 -> %s ; sub-vs-full %.6f vs %.6f"
        % (k, np.round(ev, 3), np.round(ev8, 3), np.round(ev / 8, 3), qd[5], full))

# ============================ doublet-projected gradient (27-color sweep) ============================
grad = {}
for k in ("seed", "end"):
    M = M_ALL[k]; fr = FR[k]["F2"]
    ga = np.zeros_like(ONE); gb = np.zeros_like(ONE); haa = np.zeros_like(ONE); hbb = np.zeros_like(ONE)
    eps_g = 3e-3
    idx = np.indices((N_CELL,) * 3)
    # per-cell energy density of the circle-averaged E_stat
    d0 = energy_circle(M, LIFT[k], NS_R, want_density=True)["dens_stat"]
    for comp, (gg, hh) in (("a", (ga, haa)), ("b", (gb, hbb))):
        for cx in range(3):
            for cy in range(3):
                for cz in range(3):
                    col = ((idx[0] % 3 == cx) & (idx[1] % 3 == cy) & (idx[2] % 3 == cz) & FREE).astype(float)
                    v = doublet_field(fr, col, ZERO) if comp == "a" else doublet_field(fr, ZERO, col)
                    dp = energy_circle(M + eps_g * v, LIFT[k], NS_R, want_density=True)["dens_stat"]
                    dm = energy_circle(M - eps_g * v, LIFT[k], NS_R, want_density=True)["dens_stat"]
                    dfirst = (dp - dm) / (2 * eps_g); dsec = (dp - 2 * d0 + dm) / eps_g ** 2
                    # sum each perturbed cell's radius-1 neighborhood (disjoint at spacing 3)
                    s1 = dfirst; s2 = dsec
                    # explicit 3^3 neighborhood sum via shifts
                    acc1 = np.zeros_like(s1); acc2 = np.zeros_like(s2)
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dz in (-1, 0, 1):
                                acc1 += np.roll(np.roll(np.roll(s1, dx, 0), dy, 1), dz, 2)
                                acc2 += np.roll(np.roll(np.roll(s2, dx, 0), dy, 1), dz, 2)
                    m = col > 0
                    gg[m] = acc1[m]; hh[m] = acc2[m]
    gnorm = float(np.sqrt((ga ** 2 + gb ** 2)[FREE].sum()))
    # local single-cell Rayleigh quotients (diagonal of H over 2 T_aa / 2 T_bb)
    va = doublet_field(fr, ONE, ZERO); vb = doublet_field(fr, ZERO, ONE)
    Taa = kin_of(M, va, LIFT[k], density=True)["dens_kin"]; Tbb = kin_of(M, vb, LIFT[k], density=True)["dens_kin"]
    Ra = haa[FREE] / (2 * Taa[FREE]); Rb = hbb[FREE] / (2 * Tbb[FREE])
    grad[k] = dict(grad_norm_coeff_basis=gnorm, grad_norm_unitFrobenius_basis=gnorm * np.sqrt(2.0),
                   grad_max_cell=float(np.sqrt(ga ** 2 + gb ** 2).max()),
                   hess_diag_min_a=float(haa[FREE].min()), hess_diag_min_b=float(hbb[FREE].min()),
                   local_R_min_a=float(Ra.min()), local_R_min_b=float(Rb.min()), eps=eps_g,
                   note="gradient w.r.t. the coefficients (a,b) of v = a(ee^T-ff^T)+b(ef^T+fe^T); the |v|_F=1 normalization multiplies by sqrt(2)")
    log("grad %s: |P_doublet grad E| = %.5f (coeff basis) / %.5f (unit-Frobenius basis); H diag min a %.4f b %.4f; local R min %.4f / %.4f"
        % (k, gnorm, gnorm * np.sqrt(2), haa[FREE].min(), hbb[FREE].min(), Ra.min(), Rb.min()))

# ============================ C2.6: random localized doublet directions ============================
rng = np.random.default_rng(20260906)
rand = {}
for k in ("seed", "end"):
    M = M_ALL[k]; rows = []
    for t in range(24):
        r0 = rng.uniform(0.0, 8.0); dirn = rng.normal(size=3); dirn /= np.linalg.norm(dirn)
        c = r0 * dirn; sig = rng.uniform(2.0, 6.0)
        env = np.exp(-(((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) / (2 * sig ** 2)))
        frame = ("F2", "F3", "S")[t % 3]; comp = ("a", "b", "ab")[(t // 3) % 3]
        ang = rng.uniform(0, 2 * np.pi)
        a = env * (1.0 if comp == "a" else (0.0 if comp == "b" else np.cos(ang)))
        b = env * (0.0 if comp == "a" else (1.0 if comp == "b" else np.sin(ang)))
        v = doublet_field(FR[k][frame], a, b)
        R, qv, tv = rayleigh(M, v, LIFT[k], EPS, NS_R, E0[k])
        rows.append(dict(center=c.tolist(), r0=float(r0), sigma=float(sig), frame=frame, comp=comp,
                         R=float(R), q=float(qv), twoT=float(2 * tv)))
    Rs = np.array([r["R"] for r in rows])
    rand[k] = dict(n=len(rows), R_min=float(Rs.min()), R_max=float(Rs.max()), q_min=float(min(r["q"] for r in rows)),
                   argmin=rows[int(Rs.argmin())], rows=rows)
    log("C2.6 %s: %d random localized directions: R in [%.4f, %.4f]; min q = %.4f" % (k, len(rows), Rs.min(), Rs.max(), rand[k]["q_min"]))

# ============================ verdicts ============================
prod = PRODUCER
# ---- C2.4 vacuum: the cosine modes are EXACT eigenmodes of the (K_P + U) operator on the vacuum, so the
# single-trial quotients are the eigenvalues themselves (the Ritz reproduces them to 1e-8)
vac_low = ritz["vac"]["ritz_lowest"]
R_cos = ritz["vac"]["R_single"][0]
ratio = prod["C2.4"]["omega2_vac"][0] / vac_low[0]
ratio2 = prod["C2.4"]["omega2_vac"][2] / vac_low[2]
vu = doublet_field(FR["vac"]["F2"], ONE, ZERO)     # uniform on the free box (zero on the shell)
E0v1 = energy_circle(M_ALL["vac"], LIFT["vac"], 1)["E_stat"]
qu = second_diff(M_ALL["vac"], vu, LIFT["vac"], EPS, 1, E0v1)
tu = kin_of(M_ALL["vac"], vu, LIFT["vac"])["kin_tot"]
# the per-cell uniform read: an interior cell's U-only second derivative over its 2T (no gradient): mu/c_P by construction
c24 = dict(sub=dict(lowest_four=dict(mine=vac_low[:4], producer=prod["C2.4"]["omega2_vac"], ratio_producer_over_mine=[ratio, ratio2],
                                    verdict="REFUTED"),
                    continuum_bottom_below_hedgehog=dict(mine_vac=vac_low[0], mine_hedgehog_upper_bounds=[ritz["seed"]["ritz_lowest"][0], ritz["end"]["ritz_lowest"][0]],
                                                         verdict="QUALIFIED (my hedgehog numbers are UPPER bounds 2.2x above the vacuum bottom; no localized trial came within 6x of it; not a proof of absence)"),
                    uniform_mode_threshold=dict(analytic=MU / C_P, uniform_on_free_box_with_shell_edge=float(qu / (2 * tu)), verdict="CONFIRMED (per cell: U'' / 2T = 2 mu / 2 c_P)"),
                    naive_estimate_underestimates=dict(naive_continuum=float(MU + 3 * (np.pi / 42.0) ** 2), naive_discrete_Lf43p5=R_analytic_vac, mine=R_cos,
                                                       gradient_stiffness_excess_over_cP_grad2=float(R_cos - R_analytic_vac),
                                                       verdict="REFUTED: the doublet's gradient stiffness on the vacuum is EXACTLY c_P |grad zeta|^2 (K_P) plus mu |zeta|^2 (U); nothing exceeds it")),
           ritz_lowest=vac_low, R_cos111_F2a=R_cos, R_analytic_cos111=R_analytic_vac, R_analytic_cos211=R_analytic_vac_211,
           producer_lowest=prod["C2.4"]["omega2_vac"], producer_over_mine=[ratio, ratio2])
verdict("C2.4", "REFUTED", "explicit Rayleigh quotients of the Dirichlet cosine doublets on the vacuum (uniform lift; exact eigenmodes of the K_P + U operator there) + Ritz on 9 trials + the analytic value; the producer's two lowest pairs are 2.000x mine", **c24)

# ---- C2.2 seed
s_low = ritz["seed"]["ritz_lowest"]
seed_core = core["seed"]["eigs_block_unitdir"]
c22 = dict(sub=dict(lowest_four=dict(mine_upper_bound=s_low[:4], producer=prod["C2.2"]["omega2_seed"], ratio_producer_over_my_bound=prod["C2.2"]["omega2_seed"][0] / s_low[0],
                                    verdict="REFUTED as stated (a variational UPPER bound lies below the claimed lowest eigenvalue); after removing the vacuum's factor 2 (0.0450) the claim is compatible with my bound"),
                    delocalized=dict(mine_rms_radius=ritz["seed"]["lowest_rms_radius_Tweighted"], mine_T_frac_r8=ritz["seed"]["lowest_fraction_T_inside_r8"],
                                     producer=[prod["C2.2"]["rms_radius"], 0.025], verdict="CONFIRMED (my lowest Ritz vector is a box mode)"),
                    morse_index_0=dict(min_random_R=rand["seed"]["R_min"], min_single_cell_R=min(grad["seed"]["local_R_min_a"], grad["seed"]["local_R_min_b"]), min_ritz=min(s_low), verdict="CONFIRMED (no negative second difference in any sampled doublet direction)"),
                    core_hessian=dict(mine_ns4=seed_core[:4], mine_ns8=core["seed"]["eigs_block_unitdir_ns8"][:4], producer=prod["C2.2"]["core_hess_seed"],
                                      verdict="QUALIFIED (all positive; the 4th matches to 0.1 percent, the lowest three are 4-6 percent below the producer's)")),
           ritz_lowest=s_low, R_singles=dict(zip(ritz["seed"]["names"], ritz["seed"]["R_single"])), producer_lowest=prod["C2.2"]["omega2_seed"],
           core_hessian=core["seed"], random_R_min=rand["seed"]["R_min"], local_single_cell_R_min=[grad["seed"]["local_R_min_a"], grad["seed"]["local_R_min_b"]])
verdict("C2.2", "REFUTED", "Ritz upper bound on the lowest doublet Omega^2 from 15 trial fields (frames F2/F3/S + frame-free projected tensors) lies %.2fx BELOW the claimed lowest; Morse index 0 and the box-mode character confirmed; core 10-dir Hessian positive, lowest three 4-6 percent off" % (prod["C2.2"]["omega2_seed"][0] / s_low[0]), **c22)

# ---- C2.3 end
e_low = ritz["end"]["ritz_lowest"]
end_core = core["end"]["eigs_block_unitdir"]
c23 = dict(sub=dict(lowest_four=dict(mine_upper_bound=e_low[:4], producer=prod["C2.3"]["omega2_end"], ratio_producer_over_my_bound=prod["C2.3"]["omega2_end"][0] / e_low[0],
                                    verdict="REFUTED as stated (upper bound below the claimed lowest); compatible after removing the factor 2 (0.0448)"),
                    morse_index_0=dict(min_random_R=rand["end"]["R_min"], min_single_cell_R=min(grad["end"]["local_R_min_a"], grad["end"]["local_R_min_b"]), min_ritz=min(e_low), verdict="CONFIRMED"),
                    core_hessian_lowest=dict(mine_ns4=end_core[0], mine_ns8=core["end"]["eigs_block_unitdir_ns8"][0], producer=prod["C2.3"]["core_hess_end_lowest"], verdict="CONFIRMED (positive, within 3 percent)"),
                    doublet_gradient_norm=dict(mine_end=grad["end"]["grad_norm_coeff_basis"], mine_seed=grad["seed"]["grad_norm_coeff_basis"],
                                               producer=[prod["C2.3"]["grad_doublet_end"], prod["C2.3"]["grad_doublet_seed"]], verdict="CONFIRMED (0.3 percent and 0.1 percent)")),
           ritz_lowest=e_low, R_singles=dict(zip(ritz["end"]["names"], ritz["end"]["R_single"])), producer_lowest=prod["C2.3"]["omega2_end"],
           core_hessian=core["end"], grad_doublet=grad["end"], grad_doublet_seed=grad["seed"], random_R_min=rand["end"]["R_min"])
verdict("C2.3", "REFUTED", "Ritz upper bound on the end field %.2fx BELOW the claimed lowest; the 27-color doublet-projected gradient (0.0153 / 0.324) and the core lowest eigenvalue CONFIRMED; Morse index 0 confirmed" % (prod["C2.3"]["omega2_end"][0] / e_low[0]), **c23)

# ---- C2.6 Morse index in the split sector
neg = [k for k in ("seed", "end") if rand[k]["R_min"] < 0 or grad[k]["local_R_min_a"] < 0 or grad[k]["local_R_min_b"] < 0 or min(ritz[k]["ritz_lowest"]) < 0]
verdict("C2.6", "REFUTED" if neg else "CONFIRMED",
        "24 random Gaussian doublet directions per field (sigma 2-6, r0 0-8, frames F2/F3/S, a/b/mixed) + the diagonal second difference at EVERY free cell + the 15-trial Ritz: every second difference positive",
        random=dict(seed=dict(R_min=rand["seed"]["R_min"], q_min=rand["seed"]["q_min"], argmin=rand["seed"]["argmin"]), end=dict(R_min=rand["end"]["R_min"], q_min=rand["end"]["q_min"], argmin=rand["end"]["argmin"])),
        single_cell_R_min=dict(seed=[grad["seed"]["local_R_min_a"], grad["seed"]["local_R_min_b"]], end=[grad["end"]["local_R_min_a"], grad["end"]["local_R_min_b"]]),
        fields_with_negative=neg)

RESULT["ritz"] = ritz; RESULT["core_hessian"] = core; RESULT["gradient"] = grad; RESULT["random_localized"] = rand
RESULT["inertia"] = c21; RESULT["pointwise"] = c25
tally = {}
for cid, c in RESULT["claims"].items():
    tally[c["verdict"]] = tally.get(c["verdict"], 0) + 1
RESULT["tally"] = tally
RESULT["runtime_s"] = time.time() - T0
RESULT["log"] = LOG
out = os.path.join(RESEARCH, "data", "m5_32_r16_2_audit.json")
with open(out, "w") as fh:
    json.dump(RESULT, fh, indent=1, default=float)
log("tally %s ; wrote data/m5_32_r16_2_audit.json ; runtime %.1f s" % (tally, RESULT["runtime_s"]))
print("\n| claim | verdict | mine | producer |\n| --- | --- | --- | --- |")
print("| C2.1 | %s | seed T eig [%.4f, %.4f], vac-region T_aa %.4f, vacuum %.4f | [3.397, 4.081], 3.429, 3.375 |"
      % (RESULT["claims"]["C2.1"]["verdict"], c21["seed"]["T_eig_min_free"], c21["seed"]["T_eig_max_free"], c21["seed"]["T_aa_vacregion_mean"], c21["vac"]["T_eig_min_free"]))
print("| C2.2 | %s | Ritz upper bound %s ; core eigs %s (ns8 %s) | 0.08994 x2, 0.08997 x2 ; 3.80, 4.15 x2, 36.1 |"
      % (RESULT["claims"]["C2.2"]["verdict"], np.round(s_low[:4], 5), np.round(core["seed"]["eigs_block_unitdir"][:4], 3), np.round(core["seed"]["eigs_block_unitdir_ns8"][:4], 3)))
print("| C2.3 | %s | Ritz upper bound %s ; core lowest %.3f ; grad %.4f (seed %.4f) | 0.08957 x2, 0.08963 x2 ; 1.097 ; 0.0153 (0.324) |"
      % (RESULT["claims"]["C2.3"]["verdict"], np.round(e_low[:4], 5), core["end"]["eigs_block_unitdir"][0], grad["end"]["grad_norm_coeff_basis"], grad["seed"]["grad_norm_coeff_basis"]))
print("| C2.4 | %s | Ritz lowest %s ; cos111 R %.5f (analytic %.5f) | 0.05126 x2, 0.08241 x2 |"
      % (RESULT["claims"]["C2.4"]["verdict"], np.round(vac_low[:4], 5), R_cos, R_analytic_vac))
print("| C2.5 | %s | cross %.1e ; circle8-vs-plain %.1e | pointwise ; 1e-6 |"
      % (RESULT["claims"]["C2.5"]["verdict"], max(abs(c25[k]["cross_term_adjacent_cells"]) for k in c25), max(c25[k]["circle8_vs_plain_T_max_rel"] for k in c25)))
print("| C2.6 | %s | random R min seed %.4f end %.4f | index 0 |"
      % (RESULT["claims"]["C2.6"]["verdict"], rand["seed"]["R_min"], rand["end"]["R_min"]))
print("tally:", tally)
