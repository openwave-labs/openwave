#!/usr/bin/env python3
"""M5.32 R16-4 adversarial audit: the principal symbol of the v4 Lagrangian density.

Independent re-derivation of claims C4.1 to C4.5 from the definitions only
(no producer code is imported or read).  Own pointwise Lagrangian density in the
four jets, own channel directions, own second differences (exact for the
polynomial density), own hyperbolicity tests, own co-rotated 8-sample circle
average about the lifted director (frame det[u, n, e, f] > 0).  Only the
certified stencil helpers d1 / branches / coords are imported from
m5_21_3_a_4d.py.

Run (research dir):
    OMP_NUM_THREADS=2 /opt/anaconda3/envs/openwave312/bin/python3 \
        scripts/m5_32_r16_4_audit.py
Output: data/m5_32_r16_4_audit.json (relative paths only) + a terminal table.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from m5_21_3_a_4d import d1, branches, coords  # noqa: E402  (certified stencil only)

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
I4 = np.eye(4)
W1 = 0.000724023879
G_T = 8.0
DELTA = 0.3
MU = 1e-2
C_P = 1.0
C_S = 0.4
N_S = 8
EPS_FD = 0.05          # any step is exact: the density is a polynomial of degree <= 2 per slot
RNG_K = np.random.default_rng(20260906)

FIELDS = {
    "seed_r15m": ("checkpoints/m5_32_r15/m_hedgehog/relax_n32_L48_mu0.01_cP1.npy", None, None),
    "core_r16_1": ("checkpoints/m5_32_r16/r16_1_rebuild_n32_L48.npy", None, None),
    "spike_k50": ("checkpoints/m5_32_r16/r16_3_rebuild_n32_L48_K50.npy", 50.0, 0.2758),
    "spike_k200": ("checkpoints/m5_32_r16/r16_3_rebuild_n32_L48_K200.npy", 200.0, 0.1519),
}
OMEGAS = {
    "seed_r15m": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
    "core_r16_1": [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25],
}
N, L = 32, 48.0
H = L / N
CHANNELS = ["tilt_ne", "tilt_nf", "doublet", "boost_n", "boost_e"]


# ----------------------------------------------------------------- algebra
def w_plateau(lam):
    """The plateau weight: 1 on |lam - 0.3| <= 0.5, cosine tapers to 0 at 1 and -1."""
    lam = np.asarray(lam, dtype=float)
    w = np.zeros_like(lam)
    w[np.abs(lam - 0.3) <= 0.5] = 1.0
    hi = (lam > 0.8) & (lam < 1.0)
    w[hi] = 0.5 * (1.0 + np.cos(np.pi * (lam[hi] - 0.8) / 0.2))
    lo = (lam > -1.0) & (lam < -0.2)
    w[lo] = 0.5 * (1.0 + np.cos(np.pi * (-0.2 - lam[lo]) / 0.8))
    return w


def spec_N(M):
    """Eigen-decomposition of N = M eta for fields with M_0mu = 0 (N symmetric there).

    Returns lam (..., 4) ascending and the orthonormal eigenvectors (..., 4, 4)."""
    Nm = M @ ETA
    assert np.abs(Nm - np.swapaxes(Nm, -1, -2)).max() < 1e-10, "N not symmetric: M_0mu != 0"
    return np.linalg.eigh(Nm)


def w_of_N(M):
    lam, V = spec_N(M)
    return np.einsum("...k,...ak,...bk->...ab", w_plateau(lam), V, V)


def frames(M, X, Y, Z):
    """Per-cell frame: triple (l1 >= l2 >= l3) of the spatial block, n outward, e, f
    right-handed (det[n, e, f] > 0, i.e. det[u, n, e, f] > 0 with u = e_0), J, R-builder."""
    M3 = M[..., 1:, 1:]
    lam, V = np.linalg.eigh(M3)                  # ascending
    l1, l2, l3 = lam[..., 2], lam[..., 1], lam[..., 0]
    n = V[..., :, 2].copy()
    e = V[..., :, 1].copy()
    f = V[..., :, 0].copy()
    r = np.stack([X, Y, Z], axis=-1)
    sn = np.sign(np.einsum("...a,...a->...", n, r))
    sn[sn == 0] = 1.0
    n = n * sn[..., None]
    det = np.einsum("...a,...a->...", n, np.cross(e, f))
    sf = np.sign(det)
    sf[sf == 0] = 1.0
    f = f * sf[..., None]
    J = np.zeros(M.shape)
    J[..., 1:, 1:] = np.einsum("...a,...b->...ab", f, e) - np.einsum("...a,...b->...ab", e, f)
    return dict(l1=l1, l2=l2, l3=l3, n=n, e=e, f=f, J=J, half=0.5 * (l2 - l3))


def rot(J, beta):
    return I4 + np.sin(beta) * J + (1.0 - np.cos(beta)) * (J @ J)


def emb(v3):
    v = np.zeros(4)
    v[1:] = v3
    return v


def unit(X):
    nrm = np.sqrt(np.sum(X * X))
    return X / nrm, nrm


def channel_dirs(M, fr):
    """The five channels at a cell, each a list of unit-Frobenius symmetric 4x4 directions."""
    n, e, f = emb(fr["n"]), emb(fr["e"]), emb(fr["f"])
    out, notes = {}, {}
    for name, v in (("tilt_ne", e), ("tilt_nf", f)):
        T = np.outer(v, n) - np.outer(n, v)
        xi = T @ M + M @ T.T
        xi_u, nrm = unit(xi)
        if nrm < 1e-9:                      # degenerate eigenvalues: the limit direction
            xi_u, _ = unit(np.outer(n, v) + np.outer(v, n))
            notes[name] = "generator vanished (degenerate pair); limit direction used"
        out[name] = [xi_u]
    out["doublet"] = [unit(np.outer(e, e) - np.outer(f, f))[0],
                      unit(np.outer(e, f) + np.outer(f, e))[0]]
    u0 = np.array([1.0, 0, 0, 0])
    for name, v in (("boost_n", n), ("boost_e", e)):
        B = np.outer(u0, v) + np.outer(v, u0)
        out[name] = [unit(B @ M + M @ B.T)[0]]
    return out, notes


# ----------------------------------------------------------------- density
def comm_G(A, B, G):
    return A @ G @ B - B @ G @ A


def q_term(F, G):
    return np.trace(G @ F @ G @ F.T)


def kp_term(A, w):
    Om = w @ A @ ETA @ w
    return np.trace(Om.T @ ETA @ Om @ ETA)


def reg_term(A, G):
    return np.trace(A @ G @ A @ G)


def lag_parts(A0, A, w, rho2, G):
    """The six jet-carrying terms of l (kinetic +, static -), pointwise."""
    kin_h = 4.0 * sum(q_term(comm_G(A0, A[i], G), G) for i in range(3))
    stat_h = 4.0 * sum(q_term(comm_G(A[i], A[j], G), G) for i in range(3) for j in range(i + 1, 3))
    kin_kp = 0.5 * C_P * kp_term(A0, w)
    stat_kp = 0.5 * C_P * sum(kp_term(A[i], w) for i in range(3))
    kin_rg = C_S * rho2 * reg_term(A0, G)
    stat_rg = C_S * rho2 * sum(reg_term(A[i], G) for i in range(3))
    return dict(kin_h=kin_h, stat_h=stat_h, kin_kp=kin_kp, stat_kp=stat_kp,
                kin_rg=kin_rg, stat_rg=stat_rg)


def lag(A0, A, w, rho2, G):
    p = lag_parts(A0, A, w, rho2, G)
    return (p["kin_h"] + p["kin_kp"] + p["kin_rg"]) - (p["stat_h"] + p["stat_kp"] + p["stat_rg"])


def stat_density(A, w, rho2, G, parts=False):
    p = lag_parts(np.zeros((4, 4)), A, w, rho2, G)
    if parts:
        return p["stat_h"], p["stat_kp"], p["stat_rg"]
    return p["stat_h"] + p["stat_kp"] + p["stat_rg"]


def hessian_jets(A0, A, w, rho2, G, xis, eps=EPS_FD):
    """H[mu, nu, a, b] = d^2 l / dA_mu dA_nu [xi_a, xi_b] by exact central differences."""
    m = len(xis)
    jets0 = [A0] + list(A)

    def f(p):
        jets = [jets0[mu] + sum(p[mu * m + a] * xis[a] for a in range(m)) for mu in range(4)]
        return lag(jets[0], jets[1:], w, rho2, G)

    dim = 4 * m
    f0 = f(np.zeros(dim))
    Hf = np.zeros((dim, dim))
    for p in range(dim):
        ep = np.zeros(dim)
        ep[p] = eps
        Hf[p, p] = (f(ep) - 2 * f0 + f(-ep)) / eps ** 2
        for q in range(p + 1, dim):
            eq = np.zeros(dim)
            eq[q] = eps
            Hf[p, q] = (f(ep + eq) - f(ep - eq) - f(-ep + eq) + f(-ep - eq)) / (4 * eps ** 2)
            Hf[q, p] = Hf[p, q]
    return Hf.reshape(4, m, 4, m).transpose(0, 2, 1, 3)


# ----------------------------------------------------------------- hyperbolicity
K_DIRS = None


def k_dirs(nk=96):
    global K_DIRS
    if K_DIRS is None:
        ks = RNG_K.normal(size=(nk, 3))
        ks /= np.linalg.norm(ks, axis=1)[:, None]
        K_DIRS = np.vstack([np.eye(3), ks])
    return K_DIRS


def pencil_roots(H00, B, C):
    """Roots Om of det(Om^2 H00 + 2 Om B + C) = 0 (companion linearization)."""
    m = H00.shape[0]
    Hi = np.linalg.inv(H00)
    Amat = np.block([[np.zeros((m, m)), np.eye(m)], [-Hi @ C, -2.0 * Hi @ B]])
    return np.linalg.eigvals(Amat)


def analyze(Hm, tol=1e-9):
    """Hyperbolicity of the symbol sigma(Om, k) = Om^2 H00 + 2 Om B(k) + C(k, k)."""
    m = Hm.shape[2]
    H00 = 0.5 * (Hm[0, 0] + Hm[0, 0].T)
    scale = np.abs(Hm).max()
    # 3 x 3 (m = 1) or 6 x 6 (m = 2) static stiffness -H_ij, symmetric
    S = -np.block([[Hm[i, j] for j in range(1, 4)] for i in range(1, 4)])
    S = 0.5 * (S + S.T)
    stiff = np.linalg.eigvalsh(S)
    h00_eigs = np.linalg.eigvalsh(H00)
    out = dict(H00=h00_eigs.tolist(), stiffness=stiff.tolist(), scale=float(scale))
    if m == 1:
        H0 = np.array([Hm[0, i][0, 0] for i in range(1, 4)])
        Hij = np.array([[Hm[i, j][0, 0] for j in range(1, 4)] for i in range(1, 4)])
        Hij = 0.5 * (Hij + Hij.T)
        Q = np.outer(H0, H0) - H00[0, 0] * Hij
        qe = np.linalg.eigvalsh(Q)
        out["Q"] = qe.tolist()
        out["H0i"] = H0.tolist()
        out["stiff_min_vec"] = np.linalg.eigh(S)[1][:, 0].tolist()
    max_rel_im, n_complex = 0.0, 0
    if h00_eigs.min() > 0:
        for k in k_dirs():
            B = sum(k[i] * 0.5 * (Hm[0, i + 1] + Hm[i + 1, 0]) for i in range(3))
            C = sum(k[i] * k[j] * Hm[i + 1, j + 1] for i in range(3) for j in range(3))
            C = 0.5 * (C + C.T)
            roots = pencil_roots(H00, B, C)
            rel = np.abs(roots.imag) / max(np.abs(roots).max(), 1e-300)
            if rel.max() > 1e-6:
                n_complex += 1
            max_rel_im = max(max_rel_im, float(rel.max()))
    out["roots_max_rel_imag"] = max_rel_im
    out["n_k_complex"] = n_complex
    if m == 1:
        hyp = bool(h00_eigs.min() > 0 and qe.min() >= -tol * max(scale ** 2, 1e-300))
        hyp_roots = bool(h00_eigs.min() > 0 and n_complex == 0)
        out["hyperbolic_roots"] = hyp_roots
    else:
        hyp = bool(h00_eigs.min() > 0 and n_complex == 0)
    out["hyperbolic"] = hyp
    return out


# ----------------------------------------------------------------- lattice
def lattice_samples(M, X, Y, Z, cells):
    """For each circle sample alpha: the transformed field at the cells, its sym / fwd / bwd
    jets, its rotation R, plus the lattice-wide kinetic sums (omega = 1) for omega recovery."""
    fr = frames(M, X, Y, Z)
    J = fr["J"]
    JJ = J @ J
    out = {c: [] for c in cells}
    kin_sum = dict(sym=0.0, branch=0.0)
    for k in range(N_S):
        beta = 2 * np.pi * k / N_S / 2.0
        R = I4 + np.sin(beta) * J + (1 - np.cos(beta)) * JJ
        Ma = np.einsum("...ab,...bc,...dc->...ad", R, M, R)
        jets = {}
        for br, _wt in branches("sym"):
            jets[br] = np.stack([d1(Ma, ax, H, br) for ax in range(3)], axis=-3)
        jets["sym"] = 0.5 * (jets["fwd"] + jets["bwd"])
        a0 = J @ Ma - Ma @ J                     # J M + M J^T, J antisymmetric
        w = w_of_N(Ma)
        fra = frames(Ma, X, Y, Z)
        rho2 = fra["half"] ** 2
        # lattice kinetic sums at omega = 1 (G = I on M_0mu = 0 fields)
        for reading in ("sym", "branch"):
            brs = [("sym", 1.0)] if reading == "sym" else branches("sym")
            tot = 0.0
            for br, wt in brs:
                A = jets[br]
                kin_h = 0.0
                for i in range(3):
                    F = a0 @ A[..., i, :, :] - A[..., i, :, :] @ a0
                    kin_h = kin_h + 4.0 * np.sum(F * F, axis=(-1, -2))
                tot = tot + wt * kin_h
            Om = w @ a0 @ ETA @ w
            kin_kp = 0.5 * C_P * np.einsum("...ab,...ab->...", np.swapaxes(Om, -1, -2) @ ETA, (Om @ ETA).swapaxes(-1, -2))
            kin_rg = C_S * rho2 * np.einsum("...ab,...ba->...", a0, a0)
            kin_sum[reading] += H ** 3 * float(np.sum(tot + kin_kp + kin_rg)) / N_S
        for c in cells:
            out[c].append(dict(M=Ma[c], R=R[c], a0=a0[c], w=w[c], rho2=float(rho2[c]),
                               A_sym=jets["sym"][c], A_fwd=jets["fwd"][c], A_bwd=jets["bwd"][c]))
    return fr, out, kin_sum


def symbol_at_cell(samples, xis0, omega, reading="sym"):
    """Circle-averaged Hessian blocks in the co-rotated channel basis."""
    m = len(xis0)
    Hacc = np.zeros((4, 4, m, m))
    per_sample = []
    for s in samples:
        R = s["R"]
        xis = [R @ xi @ R.T for xi in xis0]
        A0 = omega * s["a0"]
        if reading == "sym":
            Hs = hessian_jets(A0, s["A_sym"], s["w"], s["rho2"], I4, xis)
        else:
            Hs = 0.5 * (hessian_jets(A0, s["A_fwd"], s["w"], s["rho2"], I4, xis)
                        + hessian_jets(A0, s["A_bwd"], s["w"], s["rho2"], I4, xis))
        per_sample.append(Hs)
        Hacc += Hs
    return Hacc / len(samples), per_sample


def rnd(x, sig=5):
    if isinstance(x, dict):
        return {k: rnd(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, sig) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{x:.{sig}g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return rnd(x.tolist(), sig)
    return x


# ----------------------------------------------------------------- main
def main():
    t_start = time.time()
    X, Y, Z = coords(N, H)
    Rr = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    # the sampled cells: innermost, "x axis" at r 3.3 / 5.8 / 11.9 / 18.0 (two readings)
    inner = (16, 16, 16)
    exact_r = {"r3.3": (17, 17, 16), "r5.8": (19, 17, 16), "r11.9": (23, 18, 16), "r18.0": (27, 19, 16)}
    on_axis = {}
    for lab, tgt in (("r3.3", 3.3), ("r5.8", 5.8), ("r11.9", 11.9), ("r18.0", 18.0)):
        i = int(np.argmin(np.abs(Rr[:, 16, 16] - tgt)))
        on_axis[lab] = (i, 16, 16)
    cells_common = {"inner": inner}
    for lab in exact_r:
        cells_common["x_" + lab] = exact_r[lab]
        cells_common["axis_" + lab] = on_axis[lab]

    report = dict(meta=dict(script="scripts/m5_32_r16_4_audit.py",
                            python=sys.version.split()[0], omp=os.environ.get("OMP_NUM_THREADS"),
                            n=N, L=L, h=H, n_s=N_S, eps_fd=EPS_FD, c_P=C_P, c_s=C_S, mu=MU,
                            n_k_dirs=int(k_dirs().shape[0]),
                            cell_note="x_r*: cells whose r matches the brief's 3.3/5.8/11.9/18.0 "
                                      "(most on-axis representative); axis_r*: cells (i,16,16) on the "
                                      "x line nearest those radii"),
                  fields={}, results={}, omega_dependence={}, c45={}, verdicts=[], notes=[])

    # exactness of the polynomial finite difference (one spot check)
    exact_check = None

    all_results = {}
    for fname, (rel, Kfix, omega_prod) in FIELDS.items():
        t0 = time.time()
        M = np.load(os.path.join(RESEARCH, rel))
        assert M.shape == (N, N, N, 4, 4)
        fr0 = frames(M, X, Y, Z)
        half = fr0["half"]
        spike = tuple(int(v) for v in np.unravel_index(np.argmax(half), half.shape))
        cells = dict(cells_common)
        cells["spike"] = spike
        cells_list = list(dict.fromkeys(cells.values()))
        _fr, samples, kin_sum = lattice_samples(M, X, Y, Z, cells_list)
        finfo = dict(path=rel, M00_min=float(M[..., 0, 0].min()), M00_max=float(M[..., 0, 0].max()),
                     max_abs_M0mu=float(np.abs(M[..., 0, 1:]).max()),
                     max_half=float(half.max()), spike_cell=list(spike),
                     spike_r=float(Rr[spike]), spike_xyz=[float(X[spike]), float(Y[spike]), float(Z[spike])],
                     spike_triple=[float(fr0["l1"][spike]), float(fr0["l2"][spike]), float(fr0["l3"][spike])],
                     spike_gap12=float(fr0["l1"][spike] - fr0["l2"][spike]),
                     kin_tot_omega1_sym=kin_sum["sym"], kin_tot_omega1_branch=kin_sum["branch"],
                     cells={lab: dict(idx=list(c), r=float(Rr[c]),
                                      triple=[float(fr0["l1"][c]), float(fr0["l2"][c]), float(fr0["l3"][c])],
                                      half=float(half[c]),
                                      a0_norm_omega1=float(np.sqrt(np.sum(samples[c][0]["a0"] ** 2))),
                                      A_norm=float(np.sqrt(np.sum(samples[c][0]["A_sym"] ** 2))))
                            for lab, c in cells.items()})
        if Kfix is not None:
            finfo["K"] = Kfix
            finfo["omega_producer"] = omega_prod
            finfo["omega_recomputed_sym"] = Kfix / (2 * kin_sum["sym"])
            finfo["omega_recomputed_branch"] = Kfix / (2 * kin_sum["branch"])
            omegas = [0.0, omega_prod, finfo["omega_recomputed_sym"]]
        else:
            omegas = OMEGAS[fname]
        report["fields"][fname] = finfo

        res = {}
        xis_by_lab = {}
        for lab, c in cells.items():
            s0 = samples[c][0]
            xis, notes = channel_dirs(s0["M"], {k: v[c] for k, v in fr0.items()})
            xis_by_lab[lab] = xis
            if notes:
                report["notes"].append(f"{fname} {lab} {c}: {notes}")
            res[lab] = {}
            for om in omegas:
                res[lab][f"{om:.6g}"] = {}
                for ch in CHANNELS:
                    Hm, per = symbol_at_cell(samples[c], xis[ch], om, "sym")
                    an = analyze(Hm)
                    an["samples_hyperbolic"] = int(sum(analyze(Hs)["hyperbolic"] for Hs in per))
                    if lab in ("inner", "spike") or om == omegas[-1]:
                        Hb, _ = symbol_at_cell(samples[c], xis[ch], om, "branch")
                        anb = analyze(Hb)
                        an["branch_reading"] = dict(H00=anb["H00"], stiffness=anb["stiffness"],
                                                    hyperbolic=anb["hyperbolic"],
                                                    roots_max_rel_imag=anb["roots_max_rel_imag"])
                    an["_H"] = Hm
                    res[lab][f"{om:.6g}"][ch] = an
                    if exact_check is None:
                        s = samples[c][0]
                        Ha = hessian_jets(om * s["a0"], s["A_sym"], s["w"], s["rho2"], I4,
                                          [s["R"] @ x @ s["R"].T for x in xis[ch]], eps=0.05)
                        Hb2 = hessian_jets(om * s["a0"], s["A_sym"], s["w"], s["rho2"], I4,
                                           [s["R"] @ x @ s["R"].T for x in xis[ch]], eps=0.005)
                        exact_check = float(np.abs(Ha - Hb2).max() / max(np.abs(Ha).max(), 1e-300))
        all_results[fname] = res

        # omega dependence (max relative Frobenius deviation of the averaged blocks vs omega = 0)
        dep = {}
        for lab in res:
            dep[lab] = {}
            for ch in CHANNELS:
                H0 = res[lab][f"{omegas[0]:.6g}"][ch]["_H"]
                dev = 0.0
                for om in omegas[1:]:
                    Hw = res[lab][f"{om:.6g}"][ch]["_H"]
                    dev = max(dev, float(np.abs(Hw - H0).max() / max(np.abs(H0).max(), 1e-300)))
                dep[lab][ch] = dev
        report["omega_dependence"][fname] = rnd(dep)

        # C4.5: direct second difference of the STATIC density in the jets at the spike cell
        if Kfix is not None:
            c45 = {}
            c = spike
            key = f"{omega_prod:.6g}"
            xis = xis_by_lab["spike"]
            for ch in ("tilt_ne", "tilt_nf", "doublet", "boost_n", "boost_e"):
                an = res["spike"][key][ch]
                m = len(xis[ch])
                Sfull = -np.block([[an["_H"][i, j] for j in range(1, 4)] for i in range(1, 4)])
                Sfull = 0.5 * (Sfull + Sfull.T)
                evals, evecs = np.linalg.eigh(Sfull)
                v = evecs[:, 0].reshape(3, m)          # v[i, a]: k_i-weight of direction a
                d2 = dict(stat_h=0.0, stat_kp=0.0, stat_rg=0.0)
                for s in samples[c]:
                    R = s["R"]
                    xis_r = [R @ x @ R.T for x in xis[ch]]
                    dA = np.stack([sum(v[i, a] * xis_r[a] for a in range(m)) for i in range(3)])
                    step = 0.05
                    p_plus = stat_density(s["A_sym"] + step * dA, s["w"], s["rho2"], I4, parts=True)
                    p_zero = stat_density(s["A_sym"], s["w"], s["rho2"], I4, parts=True)
                    p_minus = stat_density(s["A_sym"] - step * dA, s["w"], s["rho2"], I4, parts=True)
                    for kk, name in enumerate(("stat_h", "stat_kp", "stat_rg")):
                        d2[name] += (p_plus[kk] - 2 * p_zero[kk] + p_minus[kk]) / step ** 2 / len(samples[c])
                tot = sum(d2.values())
                c45[ch] = dict(direction_is="min-eigenvector of the 3m x 3m static stiffness",
                               d2_static_total=tot, d2_by_term=d2, sign="negative" if tot < 0 else "positive",
                               stiffness_min=float(evals[0]))
            # the mechanism: the kinetic quartic 4 sum_i ||[A_0, A_i]||^2 contributes
            # +8 omega^2 ||[a0, xi]||^2 delta_ij to H_ij, i.e. it LOWERS the stiffness -H_ij by that
            # amount (32 omega^2 rho^2 for the tilt / boost-e channels, 0 for the boost along n)
            mech = {}
            om0 = f"{0.0:.6g}"
            for ch in ("tilt_ne", "tilt_nf", "boost_n", "boost_e"):
                comm2 = 0.0
                for s in samples[c]:
                    xr = s["R"] @ xis[ch][0] @ s["R"].T
                    Fc = s["a0"] @ xr - xr @ s["a0"]
                    comm2 += float(np.sum(Fc * Fc)) / len(samples[c])
                st0 = res["spike"][om0][ch]["stiffness"][0]
                stw = res["spike"][key][ch]["stiffness"][0]
                mech[ch] = dict(comm2_a0_xi=comm2, comm2_over_rho2=comm2 / max(s["rho2"], 1e-300),
                                drop_predicted=8 * omega_prod ** 2 * comm2, drop_observed=st0 - stw,
                                stiffness_omega0=st0, stiffness_omega=stw,
                                omega_star=float(np.sqrt(max(st0, 0) / (8 * comm2))) if comm2 > 1e-12 else None)
            c45["mechanism"] = mech
            c45["omega0_all_hyperbolic"] = all(res["spike"][om0][ch]["hyperbolic"] for ch in CHANNELS)
            # lattice-level supplementary check: second difference of the circle-averaged E_stat
            c45["lattice"] = lattice_static_check(M, X, Y, Z, spike, xis_by_lab, res["spike"][key],
                                                  res["inner"][key], inner)
            report["c45"][fname] = rnd(c45)

        # strip the raw Hessians from the JSON copy
        clean = {}
        for lab in res:
            clean[lab] = {}
            for om in res[lab]:
                clean[lab][om] = {ch: {k: v for k, v in res[lab][om][ch].items() if k != "_H"}
                                  for ch in res[lab][om]}
        report["results"][fname] = rnd(clean)
        print(f"[{fname}] done in {time.time() - t0:.1f} s", flush=True)

    report["meta"]["fd_exactness_rel"] = exact_check
    verdicts, tally = verdicts_of(report, all_results)
    report["verdicts"] = verdicts
    report["tally"] = tally
    report["meta"]["runtime_s"] = round(time.time() - t_start, 1)
    out = os.path.join(RESEARCH, "data", "m5_32_r16_4_audit.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print_table(report)
    print(f"wrote data/m5_32_r16_4_audit.json  runtime {report['meta']['runtime_s']} s")


def lattice_static_check(M, X, Y, Z, spike, xis_by_lab, an_spike, an_inner, inner):
    """Supplementary: second difference of the circle-averaged lattice E_stat along a
    localized plane wave in the min-stiffness channel/k direction at the spike cell, and
    the same construction at the innermost cell (reference).  Branch-averaged energy."""
    def e_stat_lattice(Mf):
        frf = frames(Mf, X, Y, Z)
        J = frf["J"]
        JJ = J @ J
        tot = 0.0
        for k in range(N_S):
            beta = 2 * np.pi * k / N_S / 2.0
            R = I4 + np.sin(beta) * J + (1 - np.cos(beta)) * JJ
            Ma = np.einsum("...ab,...bc,...dc->...ad", R, Mf, R)
            w = w_of_N(Ma)
            rho2 = frames(Ma, X, Y, Z)["half"] ** 2
            e = 0.0
            for br, wt in branches("sym"):
                A = [d1(Ma, ax, H, br) for ax in range(3)]
                eh = 0.0
                for i in range(3):
                    for j in range(i + 1, 3):
                        F = A[i] @ A[j] - A[j] @ A[i]
                        eh = eh + 4.0 * np.sum(F * F, axis=(-1, -2))
                    Om = w @ A[i] @ ETA @ w
                    eh = eh + 0.5 * C_P * np.einsum("...ab,...ab->...", np.swapaxes(Om, -1, -2) @ ETA, (Om @ ETA).swapaxes(-1, -2))
                    eh = eh + C_S * rho2 * np.einsum("...ab,...ba->...", A[i], A[i])
                e = e + wt * eh
            # jet-free terms
            Nm = Ma @ ETA
            P = Nm.copy()
            vd = 0.0
            for p in range(1, 5):
                if p > 1:
                    P = P @ Nm
                cp = (-G_T) ** p + 1.0 + 2 * DELTA ** p
                vd = vd + (np.einsum("...kk->...", P) - cp) ** 2
            e = e + W1 * vd + MU * rho2
            tot += H ** 3 * float(np.sum(e)) / N_S
        return tot

    out = {}
    sig = 2 * H
    kmag = np.pi / (2 * H)
    for lab, c, ch, an_all, xis in (("spike_min", spike, None, an_spike, xis_by_lab["spike"]),
                                    ("inner_tilt", inner, "tilt_ne", an_inner, xis_by_lab["inner"])):
        if ch is None:
            # the most negative-stiffness TILT channel at the spike (spatial one-component channels
            # only: a boost perturbation sets M_0mu != 0, moving u and G, outside this lattice check)
            ch = min(("tilt_ne", "tilt_nf"), key=lambda q: an_all[q]["stiffness"][0])
        an = an_all[ch]
        kv = np.array(an["stiff_min_vec"])
        xi = xis[ch][0]
        xc = np.array([X[c], Y[c], Z[c]])
        dx = np.stack([X - xc[0], Y - xc[1], Z - xc[2]], axis=-1)
        env = np.exp(-np.sum(dx ** 2, axis=-1) / (2 * sig ** 2)) * np.cos(kmag * dx @ kv)
        phi = env[..., None, None] * xi
        s = 1e-3
        e0 = e_stat_lattice(M)
        ep = e_stat_lattice(M + s * phi)
        em = e_stat_lattice(M - s * phi)
        d2 = (ep - 2 * e0 + em) / s ** 2
        norm2 = H ** 3 * float(np.sum(env ** 2))
        out[lab] = dict(cell=list(c), channel=ch, k_dir=kv.tolist(), k_mag=kmag, sigma=sig,
                        E_stat=e0, d2_lattice=d2, d2_per_unit_norm=d2 / norm2,
                        local_stiffness_min=an["stiffness"][0], sign="negative" if d2 < 0 else "positive")
    return out


# ----------------------------------------------------------------- verdicts
def verdicts_of(rep, allres):
    V = []
    R = rep["results"]
    F = rep["fields"]
    dep = rep["omega_dependence"]

    def hyp_all(fname, labs=None, omegas=None, chans=CHANNELS):
        bad = []
        for lab, d in R[fname].items():
            if labs is not None and lab not in labs:
                continue
            for om, dd in d.items():
                if omegas is not None and om not in omegas:
                    continue
                for ch in chans:
                    if not dd[ch]["hyperbolic"]:
                        bad.append((lab, om, ch))
        return bad

    # C4.1
    bad = hyp_all("seed_r15m") + hyp_all("core_r16_1")
    maxdep = max(max(d.values()) for f in ("seed_r15m", "core_r16_1") for d in dep[f].values())
    ci = R["core_r16_1"]["inner"]["0"]
    te = ci["tilt_ne"]
    db = ci["doublet"]
    ext = {lab: R["core_r16_1"][lab]["0"]["tilt_ne"] for lab in ("x_r18.0", "axis_r18.0")}
    prod = dict(H00=1.054, stiff=[1.011, 1.05, 1.05], Qmin=1.065, dbl_H00=1.022, dbl_stiff=0.974,
                ext_H00=1.18e-2, ext_stiff=5.9e-3)
    mine = dict(H00=te["H00"][0], stiff=te["stiffness"], Qmin=te["Q"][0], dbl_H00=db["H00"],
                dbl_stiff=db["stiffness"][0],
                ext=[(lab, F["core_r16_1"]["cells"][lab]["r"], e["H00"][0], e["stiffness"][0]) for lab, e in ext.items()])
    close = (abs(mine["H00"] / prod["H00"] - 1) < 0.05 and abs(mine["stiff"][0] / prod["stiff"][0] - 1) < 0.05
             and abs(mine["Qmin"] / prod["Qmin"] - 1) < 0.1 and abs(mine["dbl_stiff"] / prod["dbl_stiff"] - 1) < 0.05)
    ext_ok = any(abs(e[2] / prod["ext_H00"] - 1) < 0.3 and abs(e[3] / prod["ext_stiff"] - 1) < 0.3 for e in mine["ext"])
    if bad:
        v = "REFUTED"
    elif close and ext_ok and maxdep < 1e-3:
        v = "CONFIRMED"
    else:
        v = "QUALIFIED"
    V.append(dict(claim="C4.1", verdict=v, producer=prod, auditor=rnd(mine),
                  non_hyperbolic=bad, max_omega_rel_dev=rnd(maxdep),
                  method="own FD Hessian of the pointwise density, 8-sample co-rotated average, every "
                         "channel/cell/omega; H00 > 0 and Q PSD (1-comp) or real pencil roots on 99 k's (doublet)"))

    # C4.2
    bad2 = [b for f in ("seed_r15m", "core_r16_1") for b in hyp_all(f, omegas=("0.2", "0.25"), chans=("tilt_ne", "tilt_nf"))]
    V.append(dict(claim="C4.2", verdict="REFUTED" if bad2 else "CONFIRMED", non_hyperbolic=bad2,
                  producer="tilt hyperbolic at omega 0.2 and 0.25 on both cores",
                  auditor=f"{len(bad2)} non-hyperbolic tilt (cell, omega) entries",
                  method="tilt channels at omega 0.2 / 0.25 at all 10 cells + spike cell on both cores"))

    # C4.3
    f = "spike_k50"
    omk = f"{F[f]['omega_producer']:.6g}"
    sp = R[f]["spike"][omk]
    others = hyp_all(f, labs=[l for l in R[f] if l != "spike"], omegas=(omk,))
    prod = dict(tilt_H00=0.45, tilt_ne_stiff=[-0.33, -0.24, -0.18], dbl_stiff=[-1.85, -1.65], boost_e_stiff=-0.38,
                tilt_hyp=False, dbl_hyp=False, boost_e_hyp=False, boost_n_hyp=True, others_hyp=True)
    mine = {ch: dict(H00=sp[ch]["H00"], stiff=sp[ch]["stiffness"][:3], hyp=sp[ch]["hyperbolic"],
                     rel_imag=sp[ch]["roots_max_rel_imag"]) for ch in CHANNELS}
    mine["others_non_hyperbolic"] = others
    qual = (mine["tilt_ne"]["hyp"] is False and mine["tilt_nf"]["hyp"] is False and mine["doublet"]["hyp"] is False
            and mine["boost_e"]["hyp"] is False and mine["boost_n"]["hyp"] is True and not others)
    num = (abs(mine["tilt_ne"]["H00"][0] / 0.45 - 1) < 0.1 and abs(mine["tilt_ne"]["stiff"][0] / -0.33 - 1) < 0.15
           and abs(mine["doublet"]["stiff"][0] / -1.85 - 1) < 0.15 and abs(mine["boost_e"]["stiff"][0] / -0.38 - 1) < 0.15)
    V.append(dict(claim="C4.3", verdict="CONFIRMED" if (qual and num) else ("QUALIFIED" if qual else "REFUTED"),
                  producer=prod, auditor=rnd(mine), omega_used=omk,
                  omega_recomputed=rnd(F[f]["omega_recomputed_sym"]),
                  method="K=50 spike cell at the producer's omega (and mine); same tests as C4.1"))

    # C4.4
    f = "spike_k200"
    omk = f"{F[f]['omega_producer']:.6g}"
    sp = R[f]["spike"][omk]
    prod = dict(dbl_stiff=[-0.38, -0.33, -0.31], dbl_rel_imag=0.63, tilt_H00=0.40, tilt_stiff=[0.068, 0.11],
                dbl_hyp=False, tilt_hyp=True, boosts_hyp=True)
    mine = {ch: dict(H00=sp[ch]["H00"], stiff=sp[ch]["stiffness"][:3], hyp=sp[ch]["hyperbolic"],
                     rel_imag=sp[ch]["roots_max_rel_imag"]) for ch in CHANNELS}
    qual = (mine["doublet"]["hyp"] is False and mine["tilt_ne"]["hyp"] and mine["tilt_nf"]["hyp"]
            and mine["boost_n"]["hyp"] and mine["boost_e"]["hyp"])
    num = (abs(mine["doublet"]["stiff"][0] / -0.38 - 1) < 0.15 and abs(mine["tilt_ne"]["H00"][0] / 0.40 - 1) < 0.1
           and 0.05 < mine["tilt_ne"]["stiff"][0] < 0.13 and abs(mine["doublet"]["rel_imag"] / 0.63 - 1) < 0.2)
    V.append(dict(claim="C4.4", verdict="CONFIRMED" if (qual and num) else ("QUALIFIED" if qual else "REFUTED"),
                  producer=prod, auditor=rnd(mine), omega_used=omk,
                  omega_recomputed=rnd(F[f]["omega_recomputed_sym"]),
                  method="K=200 spike cell at the producer's omega; same tests"))

    # C4.5
    c45 = rep["c45"]
    neg = {}
    for f in ("spike_k50", "spike_k200"):
        for ch, d in c45[f].items():
            if ch not in CHANNELS:
                continue
            if d["stiffness_min"] < 0:
                neg[f + ":" + ch] = dict(d2=d["d2_static_total"], by_term=d["d2_by_term"], sign=d["sign"])
    mech = {f: dict(omega0_all_hyperbolic=c45[f]["omega0_all_hyperbolic"],
                    tilt_drop_pred=c45[f]["mechanism"]["tilt_ne"]["drop_predicted"],
                    tilt_drop_obs=c45[f]["mechanism"]["tilt_ne"]["drop_observed"],
                    tilt_comm2_over_rho2=c45[f]["mechanism"]["tilt_ne"]["comm2_over_rho2"],
                    omega_star={ch: c45[f]["mechanism"][ch]["omega_star"] for ch in c45[f]["mechanism"]})
            for f in ("spike_k50", "spike_k200")}
    all_neg = all(v["sign"] == "negative" for v in neg.values())
    quartic_driven = all(v["by_term"]["stat_h"] < 0 and v["by_term"]["stat_h"] <= min(v["by_term"]["stat_kp"], v["by_term"]["stat_rg"])
                         for v in neg.values())
    latt = {f: {k: (d["sign"], rnd(d["d2_per_unit_norm"])) for k, d in c45[f]["lattice"].items()} for f in c45}
    V.append(dict(claim="C4.5", verdict="CONFIRMED" if (all_neg and quartic_driven and neg) else ("QUALIFIED" if all_neg else "REFUTED"),
                  producer="negative stiffness = negative second derivative of the static density in the jets (quartic)",
                  auditor=rnd(neg), mechanism=rnd(mech), lattice_supplementary=latt,
                  method="direct second difference of the STATIC density (own function, no time slot) along the "
                         "min-stiffness direction, split by term; plus a lattice E_stat second difference"))

    tally = {k: sum(1 for v in V if v["verdict"] == k) for k in ("CONFIRMED", "QUALIFIED", "REFUTED")}
    return V, tally


def print_table(rep):
    print("\n| claim | verdict | producer | auditor | method |")
    print("| --- | --- | --- | --- | --- |")
    for v in rep["verdicts"]:
        print(f"| {v['claim']} | {v['verdict']} | {json.dumps(v['producer'])[:160]} | "
              f"{json.dumps(v['auditor'])[:220]} | {v['method'][:90]} |")
    print(f"\nTally: {rep['tally']}")
    print(f"omega dependence (max rel): { {f: max(max(d.values()) for d in rep['omega_dependence'][f].values()) for f in rep['omega_dependence']} }")
    for f, info in rep["fields"].items():
        line = f"{f}: max half {info['max_half']:.4g} at {info['spike_cell']} r {info['spike_r']:.3g} triple {rnd(info['spike_triple'], 4)}"
        if "K" in info:
            line += f"  omega producer {info['omega_producer']} mine(sym) {info['omega_recomputed_sym']:.4g} mine(branch) {info['omega_recomputed_branch']:.4g}"
        print(line)
    if rep["notes"]:
        print("notes:", *rep["notes"], sep="\n  ")


if __name__ == "__main__":
    main()
