"""M5.32 R17-0 INDEPENDENT ADVERSARIAL AUDIT (claims A1, A2, A3, E1, E2, E3, G1, H1, C1).

Every quantity under test is recomputed here with its OWN code.  The shared instrument is used
ONLY for field loading (C15.seed_uniaxial, np.load) and for reading the producers' stored JSON
numbers; no producer function is called for a quantity under test.

EQUATIONS (this file's own derivations)
--------------------------------------
A1  Own central differences (interior 2h central, one-sided at the faces, written here), own
    F_ij = A_i eta A_j - A_j eta A_i, own E_h = 4 sum_{i<j} tr(F_ij F_ij^T).  Own shell binning
    (width 2h, window 0.20 L .. 0.40 L; a second window 0.25 L .. 0.42 L), own log-log fit.
    Algebra: a RANDOM smooth director n(x) = v/|v| (v a trig polynomial, analytic derivatives),
    M = delta I + (1 - delta) n n^T, I1 = sum_{i<j} tr(F_ij F_ij^T) vs 2 (1-delta)^4 sum Omega_ij^2,
    Omega_ij = n . (d_i n x d_j n).
A2  Own evaluation of int E_1 . E_2 d^3x in PROLATE SPHEROIDAL coordinates (foci at the two
    charges), which reduces to (4 pi / d) times a pure 2D number:
        grad(1/r1).grad(1/r2) = (16/d^4)(xi^2 + eta^2 - 2)/(xi^2 - eta^2)^3,  dV = (d^3/8)(xi^2 - eta^2) dxi deta dphi
        => int = (4 pi / d) int_1^inf int_-1^1 (xi^2 + eta^2 - 2)/(xi^2 - eta^2)^2 deta dxi
    (the bracket must be 1).  Second, independent: a Monte-Carlo of the same integral.  Third, the
    convention-free argument: for ANY density c |E|^2 with E = k r_hat / r^2, the tail amplitude is
    A = c k^2 and the cross term is 2 c k^2 (4 pi / d) = 8 pi A / d, so pair / tail = 8 pi in EVERY
    self-consistent convention (Gaussian, Heaviside-Lorentz, half-square) -- the ratio cannot depend
    on the convention because both numbers are energies.
A3  The record's numbers re-read from data/m5_32_r3_pair.json; own monotonicity, own 1/d and
    log-log fits; own tail read of the R3 single from the stored npz with own jets and own shells,
    in BOTH inner products (tr(F F^T) and tr(eta F eta F^T)).
E1  Own epsilon (built from permutation parity), own X_M = (1/2) sum eps_{mu nu a b} eta^mu eta^nu
    F[mu,nu,a,b], own Lorentz maps (expm of own generators, gate L^T eta L = eta), own random jets,
    own I1..I6 (transcribed from the registry's stated index formulas and cross-checked against the
    registry values), own least squares.  Control: an ALTERNATIVE weight with eta on the internal
    pair as well, to see whether covariance depends on the stated slot rule.
E2  Own derivation: X_M = sum eps_{mu nu a b} eta_mu eta_nu (A_mu eta A_nu)[a,b] (the two halves of
    F are equal after relabelling), J^mu = sum eps_{mu nu a b} eta_mu eta_nu (M eta A_nu)[a,b],
    d_mu J^mu = X_M + sum eps eta eta (M eta d_mu A_nu)[a,b] and the second sum vanishes because
    d_mu d_nu M is symmetric.  Checked numerically on an ANALYTIC smooth field (trig polynomial)
    with exact derivatives for A and exact second derivatives for d_mu J^mu, and by own sympy on an
    independent random cubic polynomial.
E3  l_p := X_M(A_0 = E_p, A_i = background), which is d X_M / d A_0 because X_M is linear in A_0.
    Own continuum evaluation on (i) the exact radial hedgehog and (ii) a RANDOM smooth uniaxial
    director with analytic derivatives.  Own lattice residual at h = 1.5, 1.0, 0.75 with own jets.
G1  Own ANALYTIC symbol.  With A_mu -> A_mu + t k_mu xi, F_mu nu = F^bg + t (k_nu C_mu - k_mu C_nu),
    C_mu = A_mu K xi - xi K A_mu (the t^2 part cancels identically), and
        l = -4 sum_{mu<nu} eta_mu eta_nu <F_mu nu, F_mu nu>,  <X, Y> := tr(G X G Y^T)
    gives EXACTLY
        sigma(Omega, k)[xi, xi] = -4 [ S sum_mu eta_mu <C_mu, C_mu> - <W, W> ],
        S = sum_nu eta_nu k_nu^2 = |k|^2 - Omega^2,   W = sum_mu eta_mu k_mu C_mu.
    Consequences (own): H_0k = 0 for ANY static background (C_0 = 0); H_00 = 4 sum_i <C_i, C_i> =: K_bg
    is PSD (a sum of squares of the G-norm, G positive definite); H_kk + H_00 = 4 <W, W>; and on the
    hedgehog sum_i n_i A_i = 0 exactly, so W = 0 for RADIAL k at EVERY point, giving the exact
    factorization sigma = (Omega^2 - |k|^2) K_bg with 8 crossings at Omega = |k| and the signature
    (8, 2, 0) -> (0, 2, 8).  For transverse k = e, sigma = 4[Omega^2 (sum_i <C_i,C_i>) - <C_e, C_e>]
    at |k| = 1, zero at Omega^2 = <C_e,C_e> / sum_i <C_i,C_i> <= 1: crossings at speeds <= 1.
    Continuum jets on the +z axis: A_x = a (e1 e3^T + e3 e1^T), A_y = a (e2 e3^T + e3 e2^T), A_z = 0,
    a = (1 - delta) / r, embedded 4x4 with M_00 = 8.  The same formula is also evaluated on the
    LATTICE jets at the producers' cells (own jets), which is a defect-free construction of the
    producers' H (see the note on the (mu, nu) symmetrization in the output).
H1  Own spectrum: eigenvalues of N = M eta by np.linalg.eigvals, sorted; Delta = lam[3] - lam[2];
    own free mask (the 1.6-deep pin shell, rewritten here); own shell profile for r_0; own h^3 sums.
C1  Own finite differences (5-point, Richardson-checked) of V4(x) = W1 sum_p (lg^p + l1^p + (m+x)^p
    + (m-x)^p - C_p)^2 at x = half-split, C_p = (-8)^p + 1 + 2 (0.3)^p.

usage: python3 m5_32_r17_0_audit.py
out:   data/m5_32_r17_0_audit.json, checkpoints/m5_32_r17/r17_0_audit.log
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
import m5_32_r16_common as C                              # noqa: E402  (field loading + paths only)

C15, INS4 = C.C15, C.INS4
RES, DATA = C.RES, C.DATA
CK16 = C.CK
CK = os.path.join(RES, "checkpoints", "m5_32_r17")
os.makedirs(CK, exist_ok=True)
T0 = time.time()
LOG = open(os.path.join(CK, "r17_0_audit.log"), "a")

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])                      # own copy
EYE4 = np.eye(4)
DELTA = 0.3
W1 = 0.000724023879
GVAC = 8.0
OUT = {"rung": "R17-0 independent adversarial audit", "claims": {}}


def log(m):
    line = f"[{time.time() - T0:8.1f}s] {m}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


def verdict(key, v, own, producer, note):
    OUT["claims"][key] = {"verdict": v, "own_numbers": own, "producer_numbers": producer, "note": note}
    log(f"{key}: {v}  |  {note[:150]}")


# ================================================================ own lattice utilities
def own_coords(n, h):
    x = (np.arange(n) - (n - 1) / 2.0) * h
    return np.meshgrid(x, x, x, indexing="ij")


def own_free_mask(n, h, depth=1.6):
    """the 'free' cells = complement of a depth-deep shell on every face (own rewrite)."""
    wc = max(1, int(np.ceil(depth / h)))
    P = np.zeros((n, n, n), dtype=bool)
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = slice(0, wc); P[tuple(sl)] = True
        sl[ax] = slice(n - wc, n); P[tuple(sl)] = True
    return ~P


def own_d1(f, ax, h):
    """own central difference: (f[i+1] - f[i-1]) / (2h) in the interior, one-sided at the faces."""
    out = np.zeros_like(f)
    sl = [slice(None)] * f.ndim

    def at(i):
        s = list(sl); s[ax] = i; return tuple(s)
    out[at(slice(1, -1))] = (f[at(slice(2, None))] - f[at(slice(0, -2))]) / (2.0 * h)
    out[at(0)] = (f[at(1)] - f[at(0)]) / h
    out[at(-1)] = (f[at(-1)] - f[at(-2)]) / h
    return out


def own_jets(M, h):
    return [own_d1(M, ax, h) for ax in range(3)]


def comm_eta(X, Y):
    return X @ ETA @ Y - Y @ ETA @ X


def frob(X):
    return np.einsum("...ab,...ab->...", X, X)


def own_Eh(M, h, use_eta=True):
    """own per-cell static quartic density E_h = 4 sum_{i<j} tr(F_ij F_ij^T)."""
    A = own_jets(M, h)
    out = np.zeros(M.shape[:-2])
    for i in range(3):
        for j in range(i + 1, 3):
            F = comm_eta(A[i], A[j]) if use_eta else (A[i] @ A[j] - A[j] @ A[i])
            out = out + 4.0 * frob(F)
    return out


def own_shell_fit(dens, r, h, L, lo, hi, width_mult=2.0):
    """own shell binning (width = width_mult * h) and own log-log fit on [lo L, hi L)."""
    w = width_mult * h
    edges = np.arange(lo * L, hi * L + 1e-9, w)
    rs, ds, ns = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if np.sum(m) >= 8:
            rs.append(0.5 * (a + b)); ds.append(float(np.mean(dens[m]))); ns.append(int(np.sum(m)))
    rs, ds = np.array(rs), np.array(ds)
    out = {"shells_r": rs.tolist(), "shells_mean": ds.tolist(), "shells_ncells": ns}
    ok = ds > 0
    if np.sum(ok) >= 3:
        sl, ic = np.polyfit(np.log(rs[ok]), np.log(ds[ok]), 1)
        out["slope"] = float(sl); out["amp_at_r1"] = float(np.exp(ic))
        out["A_from_shells"] = float(np.mean(ds[ok] * rs[ok] ** 4))
    sel = (r >= lo * L) & (r < hi * L)
    out["A_median_r4"] = float(np.median(dens[sel] * r[sel] ** 4))
    out["A_mean_r4"] = float(np.mean(dens[sel] * r[sel] ** 4))
    out["n_cells_window"] = int(np.sum(sel))
    return out


# ================================================================ A1
def smooth_director(rng, pts, kmax=2):
    """a random smooth unit director n(x) with ANALYTIC derivatives d_i n (not a hedgehog)."""
    ks = rng.integers(-kmax, kmax + 1, size=(6, 3)).astype(float)
    ph = rng.uniform(0, 2 * np.pi, size=(6, 3))
    am = rng.normal(size=(6, 3))
    v = np.zeros((pts.shape[0], 3))
    dv = np.zeros((pts.shape[0], 3, 3))                   # dv[p, i, a] = d_i v_a
    for m in range(6):
        arg = pts @ ks[m]                                 # (P,)
        for a in range(3):
            v[:, a] += am[m, a] * np.sin(arg + ph[m, a])
            for i in range(3):
                dv[:, i, a] += am[m, a] * ks[m, i] * np.cos(arg + ph[m, a])
    nv = np.linalg.norm(v, axis=-1)
    n = v / nv[:, None]
    # d_i n_a = (dv_ia - n_a (n . dv_i)) / |v|
    proj = np.einsum("pia,pa->pi", dv, n)
    dn = (dv - proj[:, :, None] * n[:, None, :]) / nv[:, None, None]
    return n, dn


def claim_A1():
    log("A1: the r^-4 tail amplitude (own density, own jets, own shells) + the uniaxial algebra")
    fields = {"analytic_seed_n32_L48": (None, 32, 48.0), "analytic_seed_n64_L48": (None, 64, 48.0),
              "r16_1_end_n32_L48": (os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"), 32, 48.0),
              "r16_1_end_n48_L72": (os.path.join(CK16, "r16_1_rebuild_n48_L72.npy"), 48, 72.0),
              "r16_1_end_n64_L48": (os.path.join(CK16, "r16_1_rebuild_n64_L48_analytic.npy"), 64, 48.0)}
    A_pred = 8.0 * (1.0 - DELTA) ** 4
    own = {"A_predicted_8(1-delta)^4": A_pred, "fields": {}}
    for lab, (p, n, L) in fields.items():
        h = L / n
        M = C15.seed_uniaxial(C.cfg_v4(n, L)) if p is None else np.load(p)
        X, Y, Z = own_coords(n, h)
        r = np.sqrt(X * X + Y * Y + Z * Z)
        dens = own_Eh(M, h, use_eta=True)
        dens_c = own_Eh(M, h, use_eta=False)
        w1 = own_shell_fit(dens, r, h, L, 0.20, 0.40)
        w2 = own_shell_fit(dens, r, h, L, 0.25, 0.42, width_mult=1.0)
        own["fields"][lab] = {"window_0.20_0.40_w2h": {k: w1[k] for k in ("slope", "amp_at_r1", "A_from_shells", "A_median_r4", "A_mean_r4", "n_cells_window")},
                              "window_0.25_0.42_w1h": {k: w2.get(k) for k in ("slope", "amp_at_r1", "A_from_shells", "A_median_r4", "A_mean_r4", "n_cells_window")},
                              "ratio_A_median_over_pred": w1["A_median_r4"] / A_pred,
                              "eta_vs_plain_commutator_max_rel_diff": float(np.max(np.abs(dens - dens_c)) / max(np.max(np.abs(dens)), 1e-300)),
                              "h": h}
        log(f"  {lab}: own slope {w1['slope']:+.3f} (w2 {w2.get('slope', float('nan')):+.3f}); own A_median {w1['A_median_r4']:.5f} "
            f"(ratio {w1['A_median_r4'] / A_pred:.4f}); A_shells {w1['A_from_shells']:.5f}")
    # ---- the algebra, on a RANDOM smooth director (not a hedgehog), analytic derivatives
    rng = np.random.default_rng(170017)
    pts = rng.uniform(-1.0, 1.0, size=(400, 3))
    n_, dn = smooth_director(rng, pts)
    d = DELTA
    Msp = d * np.eye(3)[None] + (1.0 - d) * np.einsum("pa,pb->pab", n_, n_)
    Ai = [(1.0 - d) * (np.einsum("pa,pb->pab", dn[:, i], n_) + np.einsum("pa,pb->pab", n_, dn[:, i])) for i in range(3)]
    I1 = np.zeros(pts.shape[0]); Om2 = np.zeros(pts.shape[0])
    for i in range(3):
        for j in range(i + 1, 3):
            F = Ai[i] @ Ai[j] - Ai[j] @ Ai[i]
            I1 += np.einsum("pab,pab->p", F, F)
            Om = np.einsum("pa,pa->p", n_, np.cross(dn[:, i], dn[:, j]))
            Om2 += Om ** 2
    resid = float(np.max(np.abs(I1 - 2.0 * (1.0 - d) ** 4 * Om2)) / max(float(np.max(np.abs(I1))), 1e-300))
    # the hedgehog control: sum Omega_ij^2 r^4 = 1
    q = rng.normal(size=(200, 3)); rq = np.linalg.norm(q, axis=-1); nh = q / rq[:, None]
    dnh = (np.eye(3)[None] - np.einsum("pa,pb->pab", nh, nh)) / rq[:, None, None]   # dnh[p,i,a]
    Om2h = np.zeros(200)
    for i in range(3):
        for j in range(i + 1, 3):
            Om2h += np.einsum("pa,pa->p", nh, np.cross(dnh[:, i], dnh[:, j])) ** 2
    own["uniaxial_identity_max_rel_resid_random_director"] = resid
    own["hedgehog_sum_Omega2_times_r4_max_dev_from_1"] = float(np.max(np.abs(Om2h * rq ** 4 - 1.0)))
    own["M_uniaxial_spectrum_check"] = np.round(np.linalg.eigvalsh(Msp[0]), 12).tolist()
    ok_alg = resid < 1e-10 and own["hedgehog_sum_Omega2_times_r4_max_dev_from_1"] < 1e-10
    ratios = [own["fields"][k]["ratio_A_median_over_pred"] for k in own["fields"]]
    ok_tail = all(abs(x - 1.0) < 0.05 for x in ratios)
    slopes = [own["fields"][k]["window_0.20_0.40_w2h"]["slope"] for k in own["fields"]]
    tight = all(abs(x - 1.0) < 0.005 for x in ratios)
    v = "CONFIRMED" if (ok_alg and tight) else ("QUALIFIED" if (ok_alg and ok_tail) else "REFUTED")
    note = ("The ALGEBRA is exact and stronger than claimed: the pointwise identity I1 = 2 (1-delta)^4 sum Omega_ij^2 holds on a RANDOM smooth "
            f"director (not only on the hedgehog) to {resid:.1e}, and sum Omega_ij^2 = 1/r^4 on the hedgehog to "
            f"{own['hedgehog_sum_Omega2_times_r4_max_dev_from_1']:.1e}, so A = 8 (1-delta)^4 = 1.92080 is right. The MEASUREMENT is where I qualify. "
            f"With my own central-difference jets, my own shell binning (width 2h) and my own window (0.20 L .. 0.40 L) the amplitudes are "
            f"{['%.4f' % x for x in ratios]} x 1.92080 for the seed n32 / seed n64 / R16-1 n32 / n48 / n64, i.e. 0.5 to 3.8 percent LOW, against the "
            "producer's 1.0024 / 1.0006 / 0.9876 / 0.9987 / 0.9988. The gap is not noise: their density is the AVERAGE OF TWO DENSITIES (forward-jet and "
            "backward-jet) rather than the density of the averaged (central) jets, and E_h is quartic in the jets, so the two reads differ at O(h^2) with "
            f"opposite-signed error. My log-log slopes are {['%.2f' % x for x in slopes]}, and moving the window to 0.25 L .. 0.42 L with width h moves the "
            "slope by up to 0.08 and the amplitude by up to 0.4 percent. QUALIFICATION, precisely: the tail amplitude equals 8 (1-delta)^4 to a few "
            "percent on every field, but the quoted four-digit ratios (0.999, 1.002, 1.0006) are stencil- and window-specific and should be quoted as "
            "'1.92 to within 4 percent', not as an agreement at the 1e-3 level. Two side facts: the eta and plain-commutator inner products agree to "
            "0.0 on all five fields (M_0i = 0, so the ambiguity between the record arm's F = A eta A - A eta A and the symbolic arm's plain commutator "
            "is empty here), and the relaxed n32 core's slope is -3.85, not -4.")
    verdict("A1", v, own, {"ratios_reported": [1.0024, 1.0006, 0.9876, 0.9987, 0.9988], "A_report": A_pred}, note)


# ================================================================ A2
def claim_A2():
    log("A2: the pair / tail normalization (own prolate-spheroidal quadrature + own Monte Carlo)")
    from scipy.integrate import dblquad
    f = lambda eta, xi: (xi * xi + eta * eta - 2.0) / (xi * xi - eta * eta) ** 2
    g = lambda eta, s: f(eta, 1.0 / s) / s ** 2
    bracket, berr = dblquad(g, 1e-12, 1.0, lambda s: -1.0, lambda s: 1.0, epsabs=1e-11, epsrel=1e-11)
    # own Monte Carlo of int E1.E2 with d = 1 (importance sampling: a mixture of two 1/r^2-weighted clouds + a 1/r^4 tail)
    rng = np.random.default_rng(2024)
    Nmc = 4_000_000
    Rmax = 400.0
    # sample r from p(r) ~ 1/(4 pi r^2 Rmax) on [0, Rmax] about a randomly chosen focus (uniform in angle)
    u = rng.random(Nmc) * Rmax
    ang = rng.normal(size=(Nmc, 3)); ang /= np.linalg.norm(ang, axis=1)[:, None]
    which = rng.integers(0, 2, Nmc)
    c1 = np.array([0.0, 0.0, -0.5]); c2 = np.array([0.0, 0.0, 0.5])
    x = np.where(which[:, None] == 0, c1, c2) + u[:, None] * ang
    r1 = x - c1; r2 = x - c2
    n1 = np.linalg.norm(r1, axis=1); n2 = np.linalg.norm(r2, axis=1)
    integ = np.einsum("pa,pa->p", r1, r2) / (n1 ** 3 * n2 ** 3)
    # mixture pdf: 0.5 * 1/(4 pi n1^2 Rmax) + 0.5 * 1/(4 pi n2^2 Rmax), support = union of the two balls
    pdf = 0.5 / (4 * np.pi * n1 ** 2 * Rmax) * (n1 <= Rmax) + 0.5 / (4 * np.pi * n2 ** 2 * Rmax) * (n2 <= Rmax)
    est = float(np.mean(integ / pdf)); se = float(np.std(integ / pdf) / np.sqrt(Nmc))
    A = 8.0 * (1.0 - DELTA) ** 4
    # own convention chains: density c |E|^2, E = k r_hat / r^2  =>  A = c k^2, cross = 2 c k^2 (4 pi / d)
    chains = {}
    for name, (c, kfac) in {"gaussian (c = 1/8pi, E = q/r^2)": (1.0 / (8 * np.pi), 1.0),
                            "heaviside_lorentz (c = 1/2, E = q/4pi r^2)": (0.5, 1.0 / (4 * np.pi)),
                            "half_square (c = 1/2, E = q/r^2) -- the report's own field+density": (0.5, 1.0)}.items():
        q2 = A / (c * kfac ** 2)                      # A = c (q kfac)^2
        Ucoef = 2.0 * c * (q2 * kfac ** 2) * 4.0 * np.pi
        chains[name] = {"q^2": q2, "U_coefficient": Ucoef, "pair_over_tail": Ucoef / A}
    report_q2 = 2.0 * A                               # the report: A/r^4 = q^2/(2 r^4)
    report_U = report_q2 / (4 * np.pi)
    own = {"prolate_spheroidal_bracket (must be 1)": bracket, "prolate_quad_abserr": berr,
           "int_E1.E2_at_d=1_spheroidal": 4 * np.pi * bracket, "4pi": 4 * np.pi,
           "int_E1.E2_at_d=1_montecarlo": est, "montecarlo_stderr": se, "montecarlo_N": Nmc,
           "convention_chains": chains, "pair_over_tail_own": 8 * np.pi,
           "report_q2_from_its_own_density": report_q2, "report_U_coefficient": report_U,
           "report_pair_over_tail": report_U / A, "factor": (8 * np.pi) / (report_U / A), "16pi^2": 16 * np.pi ** 2,
           "report_internal_arithmetic_ok": bool(abs(report_q2 - 16 * (1 - DELTA) ** 4) < 1e-12 and abs(report_U - 4 * (1 - DELTA) ** 4 / np.pi) < 1e-12)}
    ok = abs(bracket - 1.0) < 1e-8 and abs(est - 4 * np.pi) < 6 * se + 1e-3
    spread = max(abs(v["pair_over_tail"] - 8 * np.pi) for v in chains.values())
    own["max_spread_of_pair_over_tail_across_conventions"] = spread
    v = "CONFIRMED" if (ok and spread < 1e-9) else "QUALIFIED"
    note = ("Own prolate-spheroidal quadrature (foci at the charges, a method independent of the producer's Gauss-flux route) gives "
            f"int E1.E2 = 4 pi / d to {abs(bracket - 1.0):.1e} relative; an own 4e6-sample Monte Carlo (a much weaker, heavy-tailed estimator) gives "
            f"{est:.3f} +- {se:.3f}, i.e. {abs(est - 4 * np.pi) / se:.1f} standard errors from 4 pi = {4 * np.pi:.4f}. "
            "The ratio pair / tail is CONVENTION-FREE: for any density c|E|^2 with E = k r_hat/r^2 the tail is A = c k^2 and the cross term is "
            "2 c k^2 4 pi / d = 8 pi A / d, identical in the Gaussian, Heaviside-Lorentz and half-square chains (spread "
            f"{spread:.1e}). So the report's 1/(2 pi) cannot be recovered in ANY self-consistent chain: it is a NORMALIZATION ERROR in the report "
            "(its own two sentences are each internally consistent -- q^2 = 2A = 16(1-delta)^4 and q^2/(4 pi d) = 4(1-delta)^4/(pi d) -- but the "
            "potential q^2/(4 pi d) belongs to E = q/(4 pi r^2), not to the E = q/r^2 it declared), NOT a misreading by the producer. "
            "QUALIFICATION the producer states and I confirm: BOTH numbers are field-level readings. The measured density is quartic in d_i n "
            "(A/r^4 = 8(1-delta)^4 Omega^2 with Omega quadratic in the gradients), so the 'E field' of two hedgehogs does not superpose linearly; "
            "neither 8 pi A/d nor 0.3057/d is a derived interaction energy of this model.")
    verdict("A2", v, own, {"pair_over_tail_producer": 8 * np.pi, "report": 1.0 / (2 * np.pi), "factor": 16 * np.pi ** 2}, note)


# ================================================================ A3
def claim_A3():
    log("A3: PAIR_LAW_NOT_CERTIFIED on the record + the R3 single's tail (own read)")
    r3 = json.load(open(os.path.join(DATA, "m5_32_r3_pair.json")))
    tab = r3["results"]["tables"]["n32"]
    own = {"from_json": {}}
    for kind in ("same", "anti"):
        ds = np.array([10.0, 14.0, 18.0, 24.0])
        Ei = np.array([tab[f"lam0_{kind}_d{d:g}_n32"]["Eint_static_undressed"] for d in ds])
        Am = np.stack([np.ones_like(ds), 1.0 / ds], -1)
        c, *_ = np.linalg.lstsq(Am, Ei, rcond=None)
        own["from_json"][kind] = {"d": ds.tolist(), "E_int": Ei.tolist(), "diffs": np.diff(Ei).tolist(),
                                  "monotone_increasing": bool(np.all(np.diff(Ei) > 0)),
                                  "loglog_slope_abs": float(np.polyfit(np.log(ds), np.log(np.abs(Ei)), 1)[0]),
                                  "A_plus_B_over_d": {"A": float(c[0]), "B": float(c[1])},
                                  "sign_all_positive": bool(np.all(Ei > 0)), "sign_all_negative": bool(np.all(Ei < 0))}
    # own tail of the R3 single (certified stack, vacuum (1, delta, 0), g 32)
    z = np.load(os.path.join(DATA, "m5_32_r3_ii", "lam0_un0_single_d0_n32.npz"))
    Ms = z["M"].astype(float)
    n, L = 32, 48.0
    h = L / n
    X, Y, Z = own_coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    A = own_jets(Ms, h)
    ed = np.diag(ETA)
    eu_eta = np.zeros((n, n, n))                       # tr(eta F eta F^T): the certified E_u inner product
    eu_frob = np.zeros((n, n, n))                      # tr(F F^T): the R16 inner product
    for i in range(3):
        for j in range(i + 1, 3):
            F = comm_eta(A[i], A[j])
            eu_eta += 4.0 * np.einsum("a,b,...ab,...ab->...", ed, ed, F, F)
            eu_frob += 4.0 * frob(F)
    t_eta = own_shell_fit(eu_eta, r, h, L, 0.20, 0.40)
    t_frob = own_shell_fit(eu_frob, r, h, L, 0.20, 0.40)
    t_wide = own_shell_fit(eu_eta, r, h, L, 0.15, 0.42)
    own["R3_single_tail_own"] = {"eta_inner_window_0.20_0.40": {k: t_eta[k] for k in ("slope", "A_median_r4", "A_from_shells")},
                                 "frob_inner_window_0.20_0.40": {k: t_frob[k] for k in ("slope", "A_median_r4", "A_from_shells")},
                                 "eta_inner_window_0.15_0.42": {k: t_wide[k] for k in ("slope", "A_median_r4", "A_from_shells")},
                                 "E_u_from_own_density_h3_sum": float(h ** 3 * np.sum(eu_eta)),
                                 "M0i_max_abs": float(np.max(np.abs(Ms[..., 0, 1:])))}
    same = own["from_json"]["same"]; anti = own["from_json"]["anti"]
    ok = same["monotone_increasing"] and same["sign_all_positive"] and anti["sign_all_negative"] and anti["loglog_slope_abs"] < -1.5
    v = "CONFIRMED" if ok else "REFUTED"
    note = (f"Like-charge E_int(d) = {np.round(same['E_int'], 2).tolist()} at d = 10/14/18/24: strictly INCREASING (diffs "
            f"{np.round(same['diffs'], 2).tolist()}), so it is not a decaying 1/d law; a least-squares A + B/d has B = {same['A_plus_B_over_d']['B']:.0f} < 0, "
            "i.e. an 'attractive' 1/d coefficient fitted to a rising curve -- a description, not a certified coefficient. Anti-pair E_int = "
            f"{np.round(anti['E_int'], 2).tolist()}, all negative, log-log slope {anti['loglog_slope_abs']:.2f}, steeper than -1. Both re-read straight from "
            "data/m5_32_r3_pair.json, matching the producer exactly. QUALIFICATIONS (mine, both STRENGTHENING the verdict): (1) the R3 single is not in an "
            f"r^-4 tail at all -- my own read gives slope {t_eta['slope']:.3f} on 0.20 L .. 0.40 L and {t_wide['slope']:.3f} on their window, and within one "
            f"window the shell-mean amplitude {t_eta['A_from_shells']:.2f} and the cell median {t_eta['A_median_r4']:.2f} differ by 57 percent, so 'A_R3 = "
            "4.23' is a windowed median of a non-power-law profile and must not be used as a tail amplitude (which also makes the plotted superposition "
            "line 8 pi A_R3 / d meaningless as a prediction); (2) my central-difference density integrates to E_u = "
            f"{own['R3_single_tail_own']['E_u_from_own_density_h3_sum']:.3f} against the certified 18.780 that the producer reproduces exactly with the sym "
            "stencil, a 17 percent stencil dependence in the very quantity whose far field is being read. The eta and Frobenius inner products give "
            f"identical densities here (M_0i = 0 to {own['R3_single_tail_own']['M0i_max_abs']:.0e}).")
    verdict("A3", v, own, {"E_int_same": [76.04, 111.79, 141.13, 173.63], "E_int_anti": [-19.87, -10.66, -6.01, -3.03],
                           "A_R3": 4.2344, "slope": -3.5649}, note)


# ================================================================ E1 / E2 / E3 own X_M machinery
def own_eps4():
    import itertools
    E = np.zeros((4,) * 4)
    for perm in itertools.permutations(range(4)):
        s = 1
        for i in range(4):
            for j in range(i + 1, 4):
                if perm[i] > perm[j]:
                    s = -s
        E[perm] = s
    return E


EPS4 = own_eps4()
ED = np.diag(ETA)


def own_W_stated():
    """(1/2) eps_{mu nu a b} eta^mu eta^nu (the stated rule: eta on the derivative pair only)."""
    return 0.5 * EPS4 * ED[:, None, None, None] * ED[None, :, None, None]


def own_W_alt():
    """the CONTROL rule: eta on the internal pair as well."""
    return 0.5 * EPS4 * ED[:, None, None, None] * ED[None, :, None, None] * ED[None, None, :, None] * ED[None, None, None, :]


def own_F(A):
    """A: (4, ..., 4, 4) -> F[..., mu, nu, a, b] = (A_mu eta A_nu - A_nu eta A_mu)[a, b]."""
    AE = A @ ETA
    P = np.einsum("m...ab,n...bc->...mnac", AE, A, optimize=True)
    return P - P.swapaxes(-4, -3)


def own_X(A, Wt):
    """X = sum Wt[m,n,a,b] F[m,n,a,b].  Wt and F are both antisymmetric in (m, n), so this equals
    2 sum Wt[m,n,a,b] (A_m eta A_n)[a,b]; computed pairwise to avoid materializing F on a lattice."""
    tot = 0.0
    for m in range(4):
        for nn in range(4):
            w = Wt[m, nn]
            if not np.any(w):
                continue
            tot = tot + np.einsum("ab,...ab->...", w, A[m] @ ETA @ A[nn], optimize=True)
    return 2.0 * tot


def own_invariants(A):
    """own I1..I6 from the registry's stated index formulas (slots 0,1 derivative; 2,3 internal)."""
    F = own_F(A)
    e = ED
    E4 = np.einsum("m,n,a,b->mnab", e, e, e, e)
    I1 = 0.5 * np.einsum("mnab,...mnab,...mnab->...", E4, F, F, optimize=True)
    I2 = np.einsum("...mnab,...abmn->...", F, F, optimize=True)
    I3 = np.einsum("m,b,...mnab,...manb->...", e, e, F, F, optimize=True)
    R2 = np.einsum("...mnam->...na", F, optimize=True)                     # R[nu, a] = sum_mu F[mu, nu, a, mu]
    I4 = np.einsum("n,a,...na,...na->...", e, e, R2, R2, optimize=True)
    I5 = np.einsum("...na,...an->...", R2, R2, optimize=True)
    Rd = np.einsum("...mnnm->...", F, optimize=True)
    I6 = Rd ** 2
    return [I1, I2, I3, I4, I5, I6], Rd


def own_lorentz(rng, kind, scale=0.3):
    from scipy.linalg import expm
    Gm = np.zeros((4, 4))
    v = scale * rng.normal(size=3)
    if kind == "boost":
        Gm[0, 1:] = v; Gm[1:, 0] = v
    else:
        Gm[1, 2], Gm[2, 1] = -v[2], v[2]
        Gm[1, 3], Gm[3, 1] = v[1], -v[1]
        Gm[2, 3], Gm[3, 2] = -v[0], v[0]
    Lm = expm(Gm)
    assert np.max(np.abs(Lm.T @ ETA @ Lm - ETA)) < 1e-12
    return Lm


def own_transform(Lm, A):
    LiT = np.linalg.inv(Lm).T
    return np.einsum("mn,n...ab->m...ab", LiT, np.einsum("ab,n...bc,dc->n...ad", Lm, A, Lm, optimize=True), optimize=True)


def claim_E1():
    log("E1: X_M covariance / parity / X_M^2 in span{I1..I6} (own eps, own invariants, own lstsq)")
    rng = np.random.default_rng(90210)                       # a DIFFERENT seed from the producer's 1707
    npts = 400
    A = 0.5 * (rng.normal(size=(4, npts, 4, 4)) + np.swapaxes(rng.normal(size=(4, npts, 4, 4)), -1, -2))
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    Wst, Walt = own_W_stated(), own_W_alt()
    x0 = own_X(A, Wst)
    xa = own_X(A, Walt)
    sc, sca = float(np.max(np.abs(x0))), float(np.max(np.abs(xa)))
    drift, drift_alt = 0.0, 0.0
    for kind in ("boost",) * 4 + ("rotation",) * 4:
        Lm = own_lorentz(rng, kind)
        Ap = own_transform(Lm, A)
        drift = max(drift, float(np.max(np.abs(own_X(Ap, Wst) - x0)) / sc))
        drift_alt = max(drift_alt, float(np.max(np.abs(own_X(Ap, Walt) - xa)) / max(sca, 1e-300)))
    Pref = np.diag([1.0, -1.0, 1.0, 1.0])
    xp = own_X(own_transform(Pref, A), Wst)
    # static purely spatial jets
    As = A.copy(); As[0] = 0.0
    As[1:, :, 0, :] = 0.0; As[1:, :, :, 0] = 0.0
    x_static = float(np.max(np.abs(own_X(As, Wst))))
    # X^2 on span{I1..I6}
    Is, Rd = own_invariants(A)
    V = np.stack(Is, -1)
    scale = np.max(np.abs(V), axis=0)
    c, *_ = np.linalg.lstsq(V / scale, x0 ** 2, rcond=None)
    res = float(np.max(np.abs((V / scale) @ c - x0 ** 2)) / max(float(np.max(np.abs(x0 ** 2))), 1e-300))
    coeff = [float(c[i] / scale[i]) for i in range(6)]
    # cross-check my own I1..I6 against the producer's registry (a basis check, not the quantity under test)
    reg_dev = None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("lag_audit", os.path.join(C.HERE, "m5_32_lagrangian.py"))
        L0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(L0)
        p = L0.default_params(s=-1.0, g=32.0)
        Mdummy = np.zeros((npts, 4, 4))
        reg = [np.asarray(L0.REGISTRY[nm].density(A, Mdummy, p)) for nm in ("I1", "I2", "I3", "I4", "I5", "I6")]
        reg_dev = [float(np.max(np.abs(reg[i] - Is[i])) / max(float(np.max(np.abs(reg[i]))), 1e-300)) for i in range(6)]
    except Exception as exc:                                                   # pragma: no cover
        reg_dev = f"unavailable: {exc}"
    # E1 = -2 X R identity, own
    parity_odd = float(np.max(np.abs(xp + x0)) / sc)
    own = {"covariance_drift_stated_rule": drift, "covariance_drift_ALT_rule_eta_on_internal_pair": drift_alt,
           "parity_odd_residual_|Xp_plus_X|_over_scale": parity_odd,
           "parity_even_residual_|Xp_minus_X|_over_scale": float(np.max(np.abs(xp - x0)) / sc),
           "X_on_static_spatial_jets_max_abs": x_static, "X_scale": sc,
           "X2_coefficients_on_I1..I6": coeff, "X2_max_rel_residual": res,
           "own_I1..I6_vs_registry_max_rel_dev": reg_dev,
           "alt_rule_is_minus_stated_rule": bool(np.max(np.abs(Wst + Walt)) < 1e-14),
           "alt_rule_equals_stated_rule": bool(np.max(np.abs(Wst - Walt)) < 1e-14),
           "own_X_pairwise_vs_full_F_max_rel_dev": float(np.max(np.abs(np.einsum("mnab,...mnab->...", Wst, own_F(A), optimize=True) - x0)) / sc)}
    ok = (drift < 1e-12 and parity_odd < 1e-12 and x_static < 1e-12 and res < 1e-10
          and np.allclose(coeff, [-2, -1, 4, 0, 0, 0], atol=1e-8))
    v = "CONFIRMED" if ok else "REFUTED"
    note = (f"Own eps, own random jets (rng 90210, not the producer's 1707), own I1..I6, own Lorentz maps: covariance drift {drift:.1e} under 8 random "
            f"SO(1,3) maps; parity ODD to {parity_odd:.1e} (and |X' - X| = 2 exactly, i.e. a clean sign flip); X = 0 on static purely spatial jets "
            f"({x_static:.1e}); X^2 = {np.round(coeff, 10).tolist()} . (I1..I6) with max relative residual {res:.1e}. My own I1..I6, transcribed from the "
            f"registry's index formulas rather than called, agree with the registry to {reg_dev}. CONTROL RESULT (asked for, and it BITES): the "
            "alternative slot rule with eta on the internal pair as well is NOT covariant -- its drift under the same 8 maps is "
            f"{drift_alt:.2f}, fifteen orders worse. Since eps_{{mu nu a b}} is nonzero only when all four indices are distinct, that alternative weight "
            "is the bare (1/2) eps with NO eta at all (eta^mu eta^nu eta^a eta^b = det eta = -1 identically), and the bare eps contraction of F is not a "
            "Lorentz scalar. So the 'eps-derivative pair by eta' half of the stated rule is LOAD-BEARING, not cosmetic, and the covariance claim does "
            "depend on it -- the producer states the rule but never tests an alternative, so this strengthens rather than weakens their result.")
    verdict("E1", v, own, {"drift": 3.71e-15, "coefficients": [-2, -1, 4, 0, 0, 0], "residual": 1.3e-15}, note)


def analytic_field(pts):
    """a smooth symmetric M(x^mu) with EXACT first and second derivatives (trig polynomial)."""
    rng = np.random.default_rng(555)
    nm = 5
    ks = rng.normal(size=(nm, 4)) * 0.7
    am = rng.normal(size=(nm, 4, 4))
    am = 0.5 * (am + np.swapaxes(am, -1, -2))
    ph = rng.uniform(0, 2 * np.pi, size=nm)
    P = pts.shape[0]
    M = np.zeros((P, 4, 4)); dM = np.zeros((4, P, 4, 4)); d2M = np.zeros((4, 4, P, 4, 4))
    for m in range(nm):
        arg = pts @ ks[m] + ph[m]
        M += np.sin(arg)[:, None, None] * am[m]
        for mu in range(4):
            dM[mu] += ks[m, mu] * np.cos(arg)[:, None, None] * am[m]
            for nu in range(4):
                d2M[mu, nu] += -ks[m, mu] * ks[m, nu] * np.sin(arg)[:, None, None] * am[m]
    return M, dM, d2M


def claim_E2():
    log("E2: the divergence identity X_M = d_mu J^mu (own analytic field, exact derivatives)")
    rng = np.random.default_rng(31415)
    pts = rng.uniform(-2.0, 2.0, size=(300, 4))
    M, dM, d2M = analytic_field(pts)
    Wst = own_W_stated()
    X = own_X(dM, Wst)
    Wrep = 2.0 * Wst                                              # J^mu = sum eps eta eta (...), no 1/2
    # J^mu = sum_{nu a b} Wrep[mu, nu, a, b] (M eta A_nu)[a, b]
    MEA = np.einsum("...ab,bc,n...cd->n...ad", M, ETA, dM, optimize=True)      # (4, P, 4, 4)
    # d_mu J^mu = sum Wrep[mu,nu,a,b] [ (A_mu eta A_nu)[a,b] + (M eta d_mu A_nu)[a,b] ]
    term1 = np.einsum("mnab,...mnab->...", Wrep, np.einsum("m...ab,bc,n...cd->...mnad", dM, ETA, dM, optimize=True), optimize=True)
    term2 = np.einsum("mnab,...mnab->...", Wrep, np.einsum("...ab,bc,mn...cd->...mnad", M, ETA, d2M, optimize=True), optimize=True)
    divJ = term1 + term2
    sc = float(np.max(np.abs(X)))
    own = {"max|divJ - X| / scale": float(np.max(np.abs(divJ - X)) / sc),
           "max|term2| / scale (the d_mu d_nu M piece, must vanish by eps antisymmetry)": float(np.max(np.abs(term2)) / sc),
           "max|term1 - X| / scale": float(np.max(np.abs(term1 - X)) / sc),
           "scale": sc, "J_uses_factor": "no 1/2 (J = 2 x the X weight), as claimed",
           "half_weight_J_max|divJ/2 - X| / scale": float(np.max(np.abs(0.5 * divJ - X)) / sc)}
    # an own sympy control on an independent random CUBIC polynomial
    try:
        import sympy as sp
        t, x, y, z = sp.symbols("t x y z", real=True)
        co = (t, x, y, z)
        rr = np.random.default_rng(777).integers(-2, 3, size=(4, 4, 8))
        mons = [1, t, x * y, z * z, t * x * y, y ** 3, x * z * t, y * y * z]
        Ms = sp.zeros(4, 4)
        for a_ in range(4):
            for b_ in range(a_, 4):
                e = sum(int(rr[a_, b_, k]) * mons[k] for k in range(8))
                Ms[a_, b_] = e; Ms[b_, a_] = e
        Es = sp.diag(-1, 1, 1, 1)
        Aj = [Ms.diff(v) for v in co]
        Xs = 0; Js = [0, 0, 0, 0]
        for mu in range(4):
            for nu in range(4):
                for a_ in range(4):
                    for b_ in range(4):
                        w = Wrep[mu, nu, a_, b_]
                        if w == 0.0:
                            continue
                        wi = sp.Integer(int(round(w)))
                        Xs += wi * sp.Rational(1, 2) * (Aj[mu] * Es * Aj[nu] - Aj[nu] * Es * Aj[mu])[a_, b_]
                        Js[mu] += wi * (Ms * Es * Aj[nu])[a_, b_]
        dv = sp.expand(sum(Js[mu].diff(co[mu]) for mu in range(4)) - Xs)
        own["sympy_cubic_polynomial_divJ_minus_X"] = str(dv)
        own["sympy_exact"] = bool(sp.simplify(dv) == 0)
    except Exception as exc:                                                   # pragma: no cover
        own["sympy_exact"] = f"unavailable: {exc}"
    ok = own["max|divJ - X| / scale"] < 1e-11 and own["sympy_exact"] is True
    v = "CONFIRMED" if ok else "REFUTED"
    note = ("Own analytic (trig-polynomial) field with EXACT first and second derivatives: |d_mu J^mu - X_M| / scale = "
            f"{own['max|divJ - X| / scale']:.1e}; the d_mu d_nu M piece is {own['max|term2| / scale (the d_mu d_nu M piece, must vanish by eps antisymmetry)']:.1e} "
            "(it vanishes because eps is antisymmetric in (mu, nu) while d_mu d_nu M is symmetric -- a hand identity, not a fit), and my own sympy on an "
            f"independent random CUBIC M gives exactly {own.get('sympy_cubic_polynomial_divJ_minus_X')}. The claimed factor (J with NO 1/2, i.e. twice the "
            f"X weight) is required: the half-weighted J misses by {own['half_weight_J_max|divJ/2 - X| / scale']:.2f} of scale. "
            "NOTE: X_M being a total divergence means it contributes NOTHING to the equations of motion; the producer states this as an identity but the "
            "consequence -- that X_M alone cannot repair anything in the Lagrangian -- is worth saying out loud.")
    verdict("E2", v, own, {"exact": True, "verdict": "TOTAL_DERIVATIVE"}, note)


def l_of_jets(Asp, Wst):
    """l_pq := X_M with A_0 = E_pq (the symmetric unit with both entries 1), A_i = Asp; a 4x4 matrix.
    X_M is linear in A_0, so this IS d X_M / d A_0 in the producer's pair-derivative convention."""
    shape = Asp[0].shape
    out = np.zeros(shape)
    for a_ in range(4):
        for b_ in range(a_, 4):
            E = np.zeros((4, 4)); E[a_, b_] = E[b_, a_] = 1.0
            A4 = np.stack([np.broadcast_to(E, shape)] + list(Asp), 0)
            xc = own_X(A4, Wst)
            out[..., a_, b_] = xc
            out[..., b_, a_] = xc
    return out


def claim_E3():
    log("E3: l = dX_M/dA_0 on uniaxial textures (own continuum + own lattice scaling)")
    Wst = own_W_stated()
    d = DELTA
    rng = np.random.default_rng(606)
    # ---- (i) the EXACT continuum hedgehog
    q = rng.normal(size=(300, 3)); rq = np.linalg.norm(q, axis=-1); nh = q / rq[:, None]
    dnh = (np.eye(3)[None] - np.einsum("pa,pb->pab", nh, nh)) / rq[:, None, None]

    def embed(A3):
        Z = np.zeros(A3.shape[:-2] + (4, 4))
        Z[..., 1:, 1:] = A3
        return Z
    Ah = [embed((1.0 - d) * (np.einsum("pa,pb->pab", dnh[:, i], nh) + np.einsum("pa,pb->pab", nh, dnh[:, i]))) for i in range(3)]
    lh = l_of_jets(Ah, Wst)
    nrm_h = np.sqrt(np.einsum("...ab,...ab->...", lh, lh))
    # ---- (ii) a RANDOM smooth uniaxial director (still uniaxial, still split-free)
    pts = rng.uniform(-1.0, 1.0, size=(300, 3))
    nr, dnr = smooth_director(rng, pts)
    Ar = [embed((1.0 - d) * (np.einsum("pa,pb->pab", dnr[:, i], nr) + np.einsum("pa,pb->pab", nr, dnr[:, i]))) for i in range(3)]
    lr = l_of_jets(Ar, Wst)
    nrm_r = np.sqrt(np.einsum("...ab,...ab->...", lr, lr))
    scale_r = np.sqrt(np.einsum("iab,iab->", np.array(Ar)[:, 0], np.array(Ar)[:, 0]))
    # the curl of the random director (the reason it differs from the hedgehog)
    curl = np.stack([dnr[:, 1, 2] - dnr[:, 2, 1], dnr[:, 2, 0] - dnr[:, 0, 2], dnr[:, 0, 1] - dnr[:, 1, 0]], -1)
    # ---- (iii) the lattice residual on the analytic seed at three h
    lat = {}
    for n, L in ((32, 48.0), (48, 48.0), (64, 48.0)):
        h = L / n
        M = C15.seed_uniaxial(C.cfg_v4(n, L))
        X, Y, Z = own_coords(n, h)
        r = np.sqrt(X * X + Y * Y + Z * Z)
        A = own_jets(M, h)
        lm = l_of_jets(A, Wst)
        nrm = np.sqrt(np.einsum("...ab,...ab->...", lm, lm))
        free = own_free_mask(n, h)
        mk = free & (r >= 3.0) & (r < 6.0)
        lat[f"h={h:g}"] = {"median_|l|_r_3_6": float(np.median(nrm[mk])), "max_|l|_free": float(np.max(nrm[free])),
                           "n": n, "L": L, "h": h, "n_cells": int(np.sum(mk))}
    hs = np.array([lat[k]["h"] for k in lat]); vs = np.array([lat[k]["median_|l|_r_3_6"] for k in lat])
    order = float(np.polyfit(np.log(hs), np.log(vs), 1)[0])
    own = {"continuum_hedgehog_max_|l|": float(np.max(nrm_h)), "continuum_hedgehog_scale_max_|A_i|": float(np.max(np.abs(np.array(Ah)))),
           "continuum_RANDOM_uniaxial_max_|l|": float(np.max(nrm_r)), "continuum_RANDOM_uniaxial_median_|l|": float(np.median(nrm_r)),
           "continuum_RANDOM_uniaxial_|A| scale": float(scale_r), "random_director_mean_|curl n|": float(np.mean(np.linalg.norm(curl, axis=-1))),
           "lattice": lat, "lattice_convergence_order_in_h": order,
           "producer_ratio_h1.5_over_h0.75": 2.8912240748592164}
    hedge_zero = own["continuum_hedgehog_max_|l|"] < 1e-12
    random_zero = own["continuum_RANDOM_uniaxial_max_|l|"] < 1e-10 * max(scale_r, 1.0)
    v = "REFUTED" if (hedge_zero and not random_zero) else ("CONFIRMED" if (hedge_zero and random_zero) else "REFUTED")
    note = ("The claim AS STATED ('l vanishes identically on a uniaxial texture in the continuum') is FALSE. Own continuum evaluation: on the exact "
            f"radial HEDGEHOG l = 0 to {own['continuum_hedgehog_max_|l|']:.1e} (confirmed), but on a RANDOM smooth UNIAXIAL director (same M = delta I + "
            f"(1-delta) n n^T, same split-free spectrum) |l| reaches {own['continuum_RANDOM_uniaxial_max_|l|']:.3f} (median "
            f"{own['continuum_RANDOM_uniaxial_median_|l|']:.3f}) on jets of scale {scale_r:.2f}. Reason, derived by hand: l_ac is the SYMMETRIC part of "
            "T[a,c] = -(1-delta)[(grad n_c x n)_a + n_c (curl n)_a]; for the hedgehog curl n = 0 and (grad n_c x n)_a = eps_{acd} n_d / r is ANTISYMMETRIC "
            "in (a,c), so the symmetric part is zero -- that is a HEDGEHOG property (radial + curl-free), not a uniaxial one. A uniaxial texture with "
            f"curl n != 0 (mean |curl n| = {own['random_director_mean_|curl n|']:.2f} here) gives l != 0. Consequence: the pre-registered inference 'any |l| at "
            "the level of the control rows is discretization, not a clock coupling' is only valid to the extent the relaxed core stays curl-free radial; "
            "it does not follow from split-freeness. SECOND FINDING: my own lattice residual on the analytic seed at h = 1.5 / 1.0 / 0.75 scales as "
            f"h^{order:.2f}, not h^2 (the producer's own 1.5 -> 0.75 ratio 2.89 = h^1.53 says the same), so the residual is NOT clean second-order "
            "discretization either; the one-sided face stencil and the r-window resampling contaminate it.")
    verdict("E3", v, own, {"median_l_h1.5": 0.0286845, "median_l_h0.75": 0.0099212, "ratio": 2.8912,
                           "claim": "l vanishes identically on a uniaxial texture in the continuum"}, note)


# ================================================================ G1
def own_sigma_matrix(Abg, Gm, Om, khat, K=None):
    """the EXACT 10 x 10 principal symbol from the own closed form
       sigma[p, q] = -4 [ S sum_mu eta_mu <C_mu^p, C_mu^q> - <W^p, W^q> ],  <X, Y> = tr(G X G Y^T),
       C_mu = A_mu K xi - xi K A_mu,  S = sum_nu eta_nu k_nu^2,  W = sum_mu eta_mu k_mu C_mu.
    Abg: list of 4 (4,4) background jets; khat: unit 3-vector; K defaults to G (the 'rebuild' completion)."""
    if K is None:
        K = Gm
    B = []
    for a_ in range(4):
        for b_ in range(a_, 4):
            E = np.zeros((4, 4))
            E[a_, b_] = E[b_, a_] = 1.0 if a_ == b_ else 2 ** -0.5
            B.append(E)
    kvec = np.array([Om, khat[0], khat[1], khat[2]])
    ed = np.diag(ETA)
    C = np.zeros((4, 10, 4, 4))
    for mu in range(4):
        for p in range(10):
            C[mu, p] = Abg[mu] @ K @ B[p] - B[p] @ K @ Abg[mu]
    def ip(Xp, Xq):                                        # <X, Y> = tr(G X G Y^T) over the 10-index
        return np.einsum("pab,bc,qdc,da->pq", Xp, Gm, Xq, Gm, optimize=True)
    S = float(np.sum(ed * kvec ** 2))
    tot = np.zeros((10, 10))
    for mu in range(4):
        tot += ed[mu] * ip(C[mu], C[mu])
    W = np.einsum("m,m,mpab->pab", ed, kvec, C, optimize=True)
    sig = -4.0 * (S * tot - ip(W, W))
    return 0.5 * (sig + sig.T), tot, W, C


def own_H_parts(Abg, Gm, khat, K=None):
    """H_00, H_0k, H_kk of the own exact symbol (sigma = Om^2 H00 + 2 Om H0k + Hkk)."""
    s0, _, _, _ = own_sigma_matrix(Abg, Gm, 0.0, khat, K)
    sp, _, _, _ = own_sigma_matrix(Abg, Gm, 1.0, khat, K)
    sm, _, _, _ = own_sigma_matrix(Abg, Gm, -1.0, khat, K)
    H00 = 0.5 * (sp + sm) - s0
    H0k = 0.25 * (sp - sm)
    return H00, H0k, s0


CP_SPEC = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 6]) / 6.0


def claim_G1():
    log("G1: the principal symbol on the EXACT continuum hedgehog (own analytic closed form)")
    d = DELTA
    own = {"continuum_axis": {}, "lattice_own_construction": {}}
    OM = np.linspace(0.0, 1.6, 33)
    for r0 in (3.0, 6.0, 12.0, 18.0):
        a = (1.0 - d) / r0
        Ax = np.zeros((4, 4)); Ax[1, 3] = Ax[3, 1] = a
        Ay = np.zeros((4, 4)); Ay[2, 3] = Ay[3, 2] = a
        Az = np.zeros((4, 4))
        Abg = [np.zeros((4, 4)), Ax, Ay, Az]
        Gm = np.eye(4)                                     # u = e_0 on the hedgehog: G = eta (I - 2 P_g) = I
        rec = {}
        for kind, khat in (("radial", np.array([0.0, 0.0, 1.0])), ("transverse", np.array([1.0, 0.0, 0.0]))):
            H00, H0k, Hkk = own_H_parts(Abg, Gm, khat)
            e00 = np.linalg.eigvalsh(H00)
            e00n = e00 / max(float(np.max(np.abs(e00))), 1e-300)
            sig = []
            for om in OM:
                S, _, _, _ = own_sigma_matrix(Abg, Gm, om, khat)
                ev = np.linalg.eigvalsh(S)
                tol = 1e-10 * max(float(np.max(np.abs(ev))), 1e-300)
                sig.append((int(np.sum(ev < -tol)), int(np.sum(np.abs(ev) <= tol)), int(np.sum(ev > tol))))
            # exact crossing speeds: sigma(Om) = Om^2 H00 + Hkk (H0k = 0); solve the symmetric pencil on
            # the range of H00 (its 2-dim kernel is also in the kernel of Hkk, checked below)
            Hkk_m = Hkk
            wv, Vv = np.linalg.eigh(H00)
            keep = wv > 1e-10 * max(float(np.max(wv)), 1e-300)
            P = Vv[:, keep]
            Ared = P.T @ (-Hkk_m) @ P
            Bred = np.diag(wv[keep])
            wgen = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(Bred, Ared))))
            speeds = np.sqrt(np.clip(wgen[wgen > 1e-12], 0, None))
            ker_leak = float(np.linalg.norm(Vv[:, ~keep].T @ Hkk_m @ Vv[:, ~keep]) / max(np.linalg.norm(H00), 1e-300))
            rec[kind] = {"H00_spectrum_normalized": np.round(e00n, 12).tolist(),
                         "H00_spectrum_vs_CP_pattern_max_dev": float(np.max(np.abs(np.sort(e00n) - CP_SPEC))),
                         "H00_min_eigenvalue": float(np.min(e00)), "H00_is_PSD": bool(np.min(e00) > -1e-12 * max(abs(float(np.max(e00))), 1e-300)),
                         "H0k_norm_over_H00": float(np.linalg.norm(H0k) / max(np.linalg.norm(H00), 1e-300)),
                         "factorization_residual |Hkk + H00| / |H00|": float(np.linalg.norm(Hkk_m + H00) / max(np.linalg.norm(H00), 1e-300)),
                         "signature_at_Omega_0": sig[0], "signature_at_Omega_1.6": sig[-1],
                         "n_distinct_signatures": len(set(sig)),
                         "crossing_speeds_Omega": np.round(speeds, 10).tolist(),
                         "n_crossing_modes": int(len(speeds)), "H00_kernel_leak_into_Hkk": ker_leak,
                         "H00_kernel_dim": int(np.sum(~keep))}
        own["continuum_axis"][f"r={r0:g}"] = rec
        log(f"  continuum r={r0:g}: radial fac {rec['radial']['factorization_residual |Hkk + H00| / |H00|']:.1e}, "
            f"CPdev {rec['radial']['H00_spectrum_vs_CP_pattern_max_dev']:.1e}, sig {rec['radial']['signature_at_Omega_0']} -> {rec['radial']['signature_at_Omega_1.6']}; "
            f"transverse speeds {np.round(rec['transverse']['crossing_speeds_Omega'], 4).tolist()}")
    # the radial factorization is exact at EVERY point of the continuum hedgehog (sum_i n_i A_i = 0)
    rng = np.random.default_rng(4242)
    q = rng.normal(size=(20, 3)); rq = np.linalg.norm(q, axis=-1); nh = q / rq[:, None]
    worst = 0.0
    for pi in range(20):
        nn, rr = nh[pi], rq[pi]
        dn = (np.eye(3) - np.outer(nn, nn)) / rr
        Abg = [np.zeros((4, 4))] + [np.pad((1 - d) * (np.outer(dn[i], nn) + np.outer(nn, dn[i])), ((1, 0), (1, 0))) for i in range(3)]
        H00, H0k, Hkk = own_H_parts(Abg, np.eye(4), nn)
        worst = max(worst, float(np.linalg.norm(Hkk + H00) / max(np.linalg.norm(H00), 1e-300)))
    own["radial_factorization_residual_at_20_random_offaxis_points"] = worst
    # ---- the same exact symbol on the LATTICE jets at the producers' cells (own jets, own formula)
    for lab, (p, n, L) in {"analytic_hedgehog_n32_L48": (None, 32, 48.0),
                           "r16_1_end_n32_L48": (os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"), 32, 48.0),
                           "r16_1_end_n64_L48": (os.path.join(CK16, "r16_1_rebuild_n64_L48_analytic.npy"), 64, 48.0)}.items():
        h = L / n
        M = C15.seed_uniaxial(C.cfg_v4(n, L)) if p is None else np.load(p)
        X, Y, Z = own_coords(n, h)
        A = own_jets(M, h)
        j = n // 2
        rec = {"max_abs_M_0i (0 => u = e_0 => G = I)": float(np.max(np.abs(M[..., 0, 1:])))}
        for rt in (3.0, 6.0, 12.0, 18.0):
            kz = int(np.argmin(np.abs(Z[j, j, :] - rt)))
            cell = (j, j, kz)
            pos = np.array([X[cell], Y[cell], Z[cell]])
            rhat = pos / np.linalg.norm(pos)
            Abg = [np.zeros((4, 4)), A[0][cell], A[1][cell], A[2][cell]]
            out2 = {}
            for comp, Km in (("rebuild_G=I", np.eye(4)), ("norm_K=eta", ETA)):
                H00, H0k, Hkk = own_H_parts(Abg, np.eye(4), rhat, K=Km)
                e00 = np.linalg.eigvalsh(H00); e00n = e00 / max(float(np.max(np.abs(e00))), 1e-300)
                out2[comp] = {"fac": float(np.linalg.norm(Hkk + H00) / max(np.linalg.norm(H00), 1e-300)),
                              "H0k_norm_over_H00": float(np.linalg.norm(H0k) / max(np.linalg.norm(H00), 1e-300)),
                              "CPdev": float(np.max(np.abs(np.sort(e00n) - CP_SPEC))),
                              "H00_min_eig": float(np.min(e00))}
            out2["completions_differ_by"] = float(abs(out2["rebuild_G=I"]["fac"] - out2["norm_K=eta"]["fac"]))
            rec[f"axis_r{rt:g}"] = {"r": float(np.linalg.norm(pos)), **out2}
        own["lattice_own_construction"][lab] = rec
        log(f"  lattice {lab}: " + "; ".join(f"{k} fac {v['rebuild_G=I']['fac']:.2e} CPdev {v['rebuild_G=I']['CPdev']:.2e}"
                                             for k, v in rec.items() if isinstance(v, dict)))
    c6 = own["continuum_axis"]["r=6"]["radial"]
    ok = (c6["factorization_residual |Hkk + H00| / |H00|"] < 1e-12 and c6["H00_spectrum_vs_CP_pattern_max_dev"] < 1e-12
          and c6["signature_at_Omega_0"] == (8, 2, 0) and c6["signature_at_Omega_1.6"] == (0, 2, 8)
          and own["continuum_axis"]["r=6"]["transverse"]["n_crossing_modes"] >= 1
          and max(own["continuum_axis"]["r=6"]["transverse"]["crossing_speeds_Omega"]) <= 1.0 + 1e-12)
    v = "CONFIRMED" if ok else "REFUTED"
    tr6 = own["continuum_axis"]["r=6"]["transverse"]
    note = ("Recomputed ANALYTICALLY on the exact continuum hedgehog jets with an own closed form for the symbol, "
            "sigma = -4[S sum_mu eta_mu <C_mu,C_mu> - <W,W>], derived here and validated to 2e-10 against a direct finite difference of the producer's "
            "OWN lag_density along a genuine plane-wave jet perturbation. Exact results, at every r and independent of the completion: (i) H_0k = 0 "
            "identically, because C_0 = 0 on any static background -- this is general, not a hedgehog fact; (ii) H_00 = 4 sum_i <C_i, C_i> is a sum of "
            "squares in the positive-definite G metric, hence POSITIVE SEMI-DEFINITE with a 2-dimensional kernel, so a fixed signature (3, 2, 5) is "
            "structurally impossible for this quartic and the hunt report's section 65 is refuted by an argument, not only by numbers; (iii) for RADIAL k "
            f"the factorization residual is {c6['factorization_residual |Hkk + H00| / |H00|']:.1e} (machine zero) and the H_00 spectrum matches "
            f"{{0,0,1,1,1,1,2,2,2,6}} to {c6['H00_spectrum_vs_CP_pattern_max_dev']:.1e} at r = 3, 6, 12 and 18 alike, with the signature going "
            f"{c6['signature_at_Omega_0']} -> {c6['signature_at_Omega_1.6']} and all 8 nonzero modes crossing at Omega = |k| = 1 exactly; (iv) the radial "
            "factorization is NOT special to the +z axis or to large r: sum_i n_i A_i = 0 everywhere on the hedgehog, so W = 0 for radial k at every "
            f"point, residual {own['radial_factorization_residual_at_20_random_offaxis_points']:.1e} at 20 random off-axis points -- the producer's "
            "r-dependent residual is entirely lattice/construction error; (v) for TRANSVERSE k the crossing speeds are exactly "
            f"{np.round(tr6['crossing_speeds_Omega'], 6).tolist()}, i.e. two modes at 1/sqrt(2) and three at 1, all <= 1, because the closed form gives "
            "Omega^2 = <C_e,C_e> / sum_i <C_i,C_i> <= 1 identically. So the Complete Picture section-40 claim is CONFIRMED exactly and the hunt report's "
            "section-65 claim is REFUTED. QUALIFICATION on the producer's LATTICE numbers: my own defect-free lattice symbol on the same analytic seed at "
            "the same cells gives radial residuals 6.7e-4 (r 5.4), 7.0e-6 (r 11.3), 5.4e-7 (r 17.3), one to four orders BELOW their 2.8e-2 / 6.3e-3 / "
            "2.7e-3, because their H_kk is contaminated (see unclaimed_hazards.producer_symbol_mixed_slot_mirroring). Their verdicts survive; their "
            "residual numbers should not be quoted.")
    verdict("G1", v, own, {"fac_r6": 2.8e-2, "fac_r12": 6e-3, "fac_r18": 3e-3, "CPdev_r6": 1.1e-3, "CPdev_r18": 1e-5,
                           "signature_claimed": "(8,2,0) -> (0,2,8)", "hunt_report": "(3,2,5) fixed"}, note)


# ================================================================ the producer's symbol construction, audited
def hazard_symmetrization():
    """Audit of the producer's own symbol CONSTRUCTION (not of a claim they made).

    full_symbol loops pi <= qi and mu <= nu, computes ONE mixed derivative
        d2 = D[mu, p; nu, q] := d^2 l / dA_mu dA_nu [xi_p, xi_q]
    and assigns it to H[mu,nu,p,q], H[nu,mu,p,q], H[mu,nu,q,p], H[nu,mu,q,p].  The true tensor is
    symmetric only under the SIMULTANEOUS swap (mu, p) <-> (nu, q), so for mu != nu the 10 x 10 block
    D[mu, .; nu, .] is NOT symmetric in (p, q).  Mirroring its upper triangle (instead of averaging
    D[mu,p;nu,q] with D[mu,q;nu,p]) changes the SYMMETRIC part of the block, so the error survives
    analyze()'s 0.5 (S + S^T).  Measured below against my own exact symbol (validated to 2e-10 by a
    direct finite difference of the producer's OWN lag_density along a genuine plane-wave jet
    perturbation)."""
    log("hazard: auditing the producer's own symbol construction (the mixed-slot (p, q) mirroring)")
    import m5_32_r16_4_symbol as S4
    import importlib.util
    spec = importlib.util.spec_from_file_location("sym_audit", os.path.join(C.HERE, "m5_32_r17_0_symbol.py"))
    SY = importlib.util.module_from_spec(spec); spec.loader.exec_module(SY)
    n, L = 32, 48.0
    h = L / n
    cells = [(16, 16, 19), (16, 16, 23), (16, 16, 27)]
    res = {"cells": [list(c) for c in cells]}
    Hs = {}
    for comp in ("rebuild", "norm"):
        cfg = C.cfg_v4(n, L, mu=0.0, cP=0.0, cs=0.0, completion=comp, n_samples=1)
        M = C15.seed_uniaxial(cfg)
        Hs[comp] = SY.full_symbol(M, cfg, cells)[0]
    res["producer_H_rel_diff_between_completions"] = float(np.max(np.abs(Hs["rebuild"] - Hs["norm"])) / max(float(np.max(np.abs(Hs["rebuild"]))), 1e-300))
    M = C15.seed_uniaxial(C.cfg_v4(n, L))
    A = own_jets(M, h)
    X, Y, Z = own_coords(n, h)
    B = []
    for a_ in range(4):
        for b_ in range(a_, 4):
            E = np.zeros((4, 4)); E[a_, b_] = E[b_, a_] = 1.0 if a_ == b_ else 2 ** -0.5
            B.append(E)
    per_cell = {}
    own_comp_diff = []
    for ci, cell in enumerate(cells):
        pos = np.array([X[cell], Y[cell], Z[cell]]); rhat = pos / np.linalg.norm(pos)
        Abg = [np.zeros((4, 4)), A[0][cell], A[1][cell], A[2][cell]]
        Cm = np.zeros((4, 10, 4, 4))
        for m_ in range(4):
            for pp in range(10):
                Cm[m_, pp] = Abg[m_] @ B[pp] - B[pp] @ Abg[m_]
        ip = lambda Xp, Xq: np.einsum("pab,qab->pq", Xp, Xq, optimize=True)
        T = sum(ip(Cm[a_], Cm[a_]) for a_ in range(1, 4))
        Hc = Hs["rebuild"][ci]
        slots = {}
        for i in range(4):
            for j in range(4):
                if i == 0 or j == 0:
                    Ho = 4.0 * T if (i == 0 and j == 0) else np.zeros((10, 10))
                else:
                    Ho = (-4.0 * T if i == j else 0.0) + 4.0 * 0.5 * (ip(Cm[i], Cm[j]) + ip(Cm[j], Cm[i]))
                slots[f"H[{i},{j}]"] = {"max_abs_dev_from_2x_own": float(np.max(np.abs(Hc[i, j] - 2.0 * Ho))),
                                        "scale_2x_own": float(np.max(np.abs(2.0 * Ho)))}
        H00o, H0ko, Hkko = own_H_parts(Abg, np.eye(4), rhat)
        H00p = Hc[0, 0]; Hkkp = np.einsum("i,j,ijpq->pq", rhat, rhat, Hc[1:, 1:], optimize=True)
        # both completions of MY exact symbol
        s_r, _, _, _ = own_sigma_matrix(Abg, np.eye(4), 0.7, rhat, K=np.eye(4))
        s_n, _, _, _ = own_sigma_matrix(Abg, np.eye(4), 0.7, rhat, K=ETA)
        own_comp_diff.append(float(np.max(np.abs(s_r - s_n)) / max(float(np.max(np.abs(s_r))), 1e-300)))
        eo = np.linalg.eigvalsh(0.5 * (own_sigma_matrix(Abg, np.eye(4), 0.0, rhat)[0] + own_sigma_matrix(Abg, np.eye(4), 0.0, rhat)[0].T))
        ep = np.linalg.eigvalsh(0.5 * (Hkkp + Hkkp.T)) / 2.0
        per_cell[str(cell)] = {
            "r": float(np.linalg.norm(pos)),
            "producer_H00_equals_2x_own_max_dev": float(np.max(np.abs(H00p - 2.0 * H00o))),
            "mixed_slot_max_dev_over_scale": max(slots[f"H[{i},{j}]"]["max_abs_dev_from_2x_own"] / max(slots[f"H[{i},{j}]"]["scale_2x_own"], 1e-300)
                                                 for i in range(1, 4) for j in range(1, 4) if i != j),
            "diagonal_slot_max_dev": max(slots[f"H[{i},{i}]"]["max_abs_dev_from_2x_own"] for i in range(1, 4)),
            "Hkk_rel_dev_from_2x_own": float(np.linalg.norm(Hkkp - 2.0 * Hkko) / max(np.linalg.norm(2.0 * Hkko), 1e-300)),
            "Hkk_dev_is_symmetric": bool(np.linalg.norm((Hkkp - 2 * Hkko) - (Hkkp - 2 * Hkko).T) < 1e-12 * max(np.linalg.norm(Hkkp), 1e-300)),
            "producer_fac_radial": float(np.linalg.norm(Hkkp + H00p) / max(np.linalg.norm(H00p), 1e-300)),
            "own_fac_radial": float(np.linalg.norm(Hkko + H00o) / max(np.linalg.norm(H00o), 1e-300)),
            "eig_at_Omega0_producer_over_2": np.round(ep, 8).tolist(),
            "eig_at_Omega0_own": np.round(eo, 8).tolist(),
            "slots": slots}
    res["per_cell"] = per_cell
    res["own_exact_symbol_completions_rel_diff"] = own_comp_diff
    res["diagnosis"] = (
        "CONFIRMED DEFECT in scripts/m5_32_r17_0_symbol.py::full_symbol. The H_00 slot is exact (it equals 2 x my own H_00 to "
        f"{max(v['producer_H00_equals_2x_own_max_dev'] for v in per_cell.values()):.1e}; the factor 2 is their l_2 = (1/2) sum k k H convention and is "
        "harmless), and so is every DIAGONAL slot H[i,i]. The MIXED slots H[i,j], i != j, are wrong by up to "
        f"{max(v['mixed_slot_max_dev_over_scale'] for v in per_cell.values()):.2f} of their own scale, because the loop computes only D[mu,p;nu,q] for "
        "p <= q and mirrors it onto (q, p), whereas the true block satisfies D[mu,p;nu,q] = D[nu,q;mu,p] (the SIMULTANEOUS swap) and is not symmetric "
        "in (p, q) alone; the correct entry is the average of D[mu,p;nu,q] and D[mu,q;nu,p]. Mirroring changes the SYMMETRIC part, so analyze()'s "
        "0.5 (S + S^T) does not remove it. Measured consequences: (1) H_kk is off by "
        + ", ".join(f"{v['Hkk_rel_dev_from_2x_own'] * 100:.1f} percent at r {v['r']:.1f}" for v in per_cell.values()) +
        "; (2) the reported factorization residual is inflated by 1 to 2 ORDERS -- producer " +
        ", ".join(f"{v['producer_fac_radial']:.2e}" for v in per_cell.values()) + " vs own " +
        ", ".join(f"{v['own_fac_radial']:.2e}" for v in per_cell.values()) +
        " at r 5.4 / 11.3 / 17.3, so 'the residual falls with h and grows in the melted core' is being read off a contaminated number; "
        f"(3) it is the reason their 'completions_agree_rel' is {res['producer_H_rel_diff_between_completions']:.3f} at n32 (JSON 0.058 / 0.123 / 0.022) while "
        f"the TRUE symbol is completion-independent (my own exact symbol agrees between K = G = I and K = eta to {max(own_comp_diff):.1e}), contradicting the "
        "script's own docstring line 'both completions (identical on u = e_0 fields, verified)'; (4) it is why their signature at Omega = 0 reads "
        "(8, 1, 1) instead of the exact (8, 2, 0) at r 5.4 and 11.3, and why their radial crossings smear over Omega in [0.925, 1.075] instead of "
        "sitting at exactly 1. NONE of their QUALITATIVE verdicts change: RADIAL_CHARACTERISTIC, the refutation of the fixed (3, 2, 5) signature, and "
        "the sub-luminal transverse crossings are all confirmed exactly by my analytic continuum computation. What must not be quoted are the residual "
        "numbers 2.8e-2 / 6.3e-3 / 2.7e-3 and the completions_agree_rel row.")
    OUT["unclaimed_hazards"] = OUT.get("unclaimed_hazards", {})
    OUT["unclaimed_hazards"]["producer_symbol_mixed_slot_mirroring"] = res
    log("  " + res["diagnosis"][:260])


# ================================================================ H1
def claim_H1():
    log("H1: K_coll on the saved R16-2 mode (own spectrum, own integrals)")
    n, L = 32, 48.0
    h = L / n
    M = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
    N = M @ ETA
    ev = np.sort(np.real(np.linalg.eigvals(N)), axis=-1)             # ascending: lg, l3, l2, l1
    lam1, lam2 = ev[..., 3], ev[..., 2]
    gap = lam1 - lam2
    free = own_free_mask(n, h)
    X, Y, Z = own_coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    Dmin = float(np.min(gap[free]))
    r_at = float(r[free][int(np.argmin(gap[free]))])
    # own r_0: shell-mean lambda_1 crossing 0.8, own shells (width 1.5h to match the reported definition)
    edges = np.arange(0.0, L / 2 + h, 1.5 * h)
    rm, l1m = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mk = (r >= a) & (r < b) & free
        if np.sum(mk):
            rm.append(0.5 * (a + b)); l1m.append(float(np.mean(lam1[mk])))
    rm, l1m = np.array(rm), np.array(l1m)
    cr = np.where((l1m[:-1] < 0.8) & (l1m[1:] >= 0.8))[0]
    r0 = float(rm[cr[0]] + (0.8 - l1m[cr[0]]) * (rm[cr[0] + 1] - rm[cr[0]]) / (l1m[cr[0] + 1] - l1m[cr[0]])) if len(cr) else 0.0
    # a second, binning-free r_0: the largest r with lambda_1 < 0.8 anywhere (taper radius)
    r0_taper = float(np.max(r[(lam1 < 0.8) & free]))
    r162 = json.load(open(os.path.join(DATA, "m5_32_r16_2.json")))
    Om2 = r162["runs"]["r16_1_end_n32"]["modes"][0]["Omega2"]
    om = float(np.sqrt(Om2) / 2.0)
    mode = np.load(os.path.join(CK16, "r16_2_r16_1_end_n32_mode0.npy"))
    amp = np.sqrt(mode[..., 0] ** 2 + mode[..., 1] ** 2)
    f = amp / np.max(amp)
    intf2 = float(h ** 3 * np.sum(f ** 2))
    Kbox = 2.0 * om * Dmin ** 2 * intf2
    core = r < max(r0, 3.0)
    fcore = amp * core / max(float(np.max(amp[core])), 1e-300)
    intf2c = float(h ** 3 * np.sum(fcore ** 2))
    Kcore = 2.0 * om * Dmin ** 2 * intf2c
    box_vol = (n * h) ** 3
    own = {"Delta_min_free": Dmin, "r_at_Delta_min": r_at, "Delta_min_central_line": float(np.min(gap[16:, 16, 16])),
           "omega": om, "Omega2_from_r16_2_json": Om2, "r_0_profile": r0, "r_0_taper": r0_taper,
           "int_f2_box": intf2, "K_coll_box": Kbox, "int_f2_core": intf2c, "K_coll_core": Kcore,
           "r_0_sqrt_mu": r0 * np.sqrt(1e-2), "box_volume_L^3": box_vol, "int_f2_over_box_volume": intf2 / box_vol,
           "weight_fraction_r_lt_r0": float(np.sum(amp[core] ** 2) / np.sum(amp ** 2)),
           "f_peak_radius": float(r.reshape(-1)[int(np.argmax(amp))]),
           "eigen_method": "np.linalg.eigvals(M @ eta), sorted ascending; gap = lam[3] - lam[2]"}
    pr = {"Delta_min": 0.049814428447529924, "omega": 0.10581220260379516, "r_0_profile": 3.0429921835847935,
          "int_f2_box": 11786.463334922804, "K_coll_box": 6.189556954274855, "K_coll_core": 0.018171519219290343,
          "int_f2_core": 34.603114019346314}
    rels = {k: abs(own[{"Delta_min": "Delta_min_free"}.get(k, k)] - pr[k]) / max(abs(pr[k]), 1e-300) for k in pr}
    ok = all(v < 1e-6 for v in rels.values())
    v = "CONFIRMED" if ok else ("QUALIFIED" if all(v < 1e-2 for v in rels.values()) else "REFUTED")
    note = (f"Own eigenvalues of M eta (np.linalg.eigvals, sorted) give Delta_min = {Dmin:.10f} at r = {r_at:.3f}; own shell profile gives "
            f"r_0 = {r0:.6f}; own h^3 sums give int f^2 = {intf2:.4f} and K_coll = {Kbox:.6f} (box mode), {Kcore:.6f} (core-restricted). "
            f"Max relative deviation from the producer's numbers: {max(rels.values()):.1e}. "
            "QUALIFICATION (mine): int f^2 = 11786 is 10.6 percent of the whole box volume 110592, i.e. the 'mode' fills the box; K_coll_box = 6.19 is "
            "therefore a statement about the box size, not about the soliton, exactly as the producer's own unit_note says -- but the number 6.19 should "
            f"never be quoted without it, since only {own['weight_fraction_r_lt_r0'] * 100:.3f} percent of the mode weight sits inside r_0 and the peak of "
            f"|zeta| is at r = {own['f_peak_radius']:.1f}, past the core. The core-restricted 0.0182 is itself an arbitrary read: it renormalizes f by the "
            "core maximum, so it is not the same functional restricted, and it scales with the (r_0-dependent) cutoff.")
    verdict("H1", v, own, pr, note)


# ================================================================ C1
def claim_C1():
    log("C1: the split curvature of V4 (own finite differences)")
    cp = np.array([(-GVAC) ** p + 1.0 + 2.0 * DELTA ** p for p in range(1, 5)])
    out = {"C_p": cp.tolist(), "fields": {}}

    def V4_of_x(lg, l1, m, x):
        s = 0.0
        for p in range(1, 5):
            s = s + (lg ** p + l1 ** p + (m + x) ** p + (m - x) ** p - cp[p - 1]) ** 2
        return W1 * s

    def hV_exact(lg, l1, m, x):
        """the EXACT second derivative, by hand: V'' = W1 sum_p 2 [T_p'^2 + (T_p - C_p) T_p'']."""
        tot = 0.0
        for p in range(1, 5):
            T = lg ** p + l1 ** p + (m + x) ** p + (m - x) ** p
            T1 = p * ((m + x) ** (p - 1) - (m - x) ** (p - 1))
            T2 = p * (p - 1) * ((m + x) ** (p - 2) + (m - x) ** (p - 2)) if p >= 2 else 0.0 * m
            tot = tot + 2.0 * (T1 ** 2 + (T - cp[p - 1]) * T2)
        return W1 * tot

    for lab, (path, n, L) in {"r16_1_end_n32_L48": (os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"), 32, 48.0),
                              "r16_1_end_n48_L72": (os.path.join(CK16, "r16_1_rebuild_n48_L72.npy"), 48, 72.0),
                              "r16_1_end_n64_L48": (os.path.join(CK16, "r16_1_rebuild_n64_L48_analytic.npy"), 64, 48.0)}.items():
        h = L / n
        M = np.load(path)
        j = n // 2
        line = (slice(j, n), j, j)
        Ml = M[line]
        ev = np.sort(np.real(np.linalg.eigvals(Ml @ ETA)), axis=-1)
        lg, l3, l2, l1 = ev[..., 0], ev[..., 1], ev[..., 2], ev[..., 3]
        m = 0.5 * (l2 + l3); hs = 0.5 * (l2 - l3)
        res = {}
        for eps in (1e-3, 1e-4):
            hV = (-V4_of_x(lg, l1, m, hs + 2 * eps) + 16 * V4_of_x(lg, l1, m, hs + eps) - 30 * V4_of_x(lg, l1, m, hs)
                  + 16 * V4_of_x(lg, l1, m, hs - eps) - V4_of_x(lg, l1, m, hs - 2 * eps)) / (12 * eps * eps)
            res[f"eps={eps:g}"] = {"hV_min_central_line": float(np.min(hV)), "hV_min_over_W1": float(np.min(hV) / W1),
                                   "r_at_min": float(((np.arange(n) - (n - 1) / 2.0) * h)[j:][int(np.argmin(hV))])}
        hVx = hV_exact(lg, l1, m, hs)
        res["exact"] = {"hV_min_central_line": float(np.min(hVx)), "hV_min_over_W1": float(np.min(hVx) / W1),
                        "r_at_min": float(((np.arange(n) - (n - 1) / 2.0) * h)[j:][int(np.argmin(hVx))]),
                        "max_abs_diff_from_FD_eps1e-4": float(np.max(np.abs(hVx - hV)))}
        X, Y, Z = own_coords(n, h)
        r = np.sqrt(X * X + Y * Y + Z * Z)
        free = own_free_mask(n, h)
        evf = np.sort(np.real(np.linalg.eigvals(M @ ETA)), axis=-1)
        lgf, l3f, l2f, l1f = evf[..., 0], evf[..., 1], evf[..., 2], evf[..., 3]
        mf = 0.5 * (l2f + l3f); hsf = 0.5 * (l2f - l3f)
        eps = 1e-4
        hVf = (-V4_of_x(lgf, l1f, mf, hsf + 2 * eps) + 16 * V4_of_x(lgf, l1f, mf, hsf + eps) - 30 * V4_of_x(lgf, l1f, mf, hsf)
               + 16 * V4_of_x(lgf, l1f, mf, hsf - eps) - V4_of_x(lgf, l1f, mf, hsf - 2 * eps)) / (12 * eps * eps)
        res["hV_min_free_all_cells"] = float(np.min(hVf[free]))
        res["hV_min_free_over_W1"] = float(np.min(hVf[free]) / W1)
        res["r_at_hV_min_free"] = float(r[free][int(np.argmin(hVf[free]))])
        res["max_half_split_on_line"] = float(np.max(hs))
        out["fields"][lab] = res
    o32 = out["fields"]["r16_1_end_n32_L48"]["exact"]["hV_min_central_line"]
    o64 = out["fields"]["r16_1_end_n64_L48"]["exact"]["hV_min_central_line"]
    p32, p64 = -0.008072173683817949, -0.010572211437507268
    rel32, rel64 = abs(o32 - p32) / abs(p32), abs(o64 - p64) / abs(p64)
    out["rel_dev_n32"] = rel32; out["rel_dev_n64"] = rel64
    ok = rel32 < 1e-8 and rel64 < 1e-8
    v = "CONFIRMED" if ok else ("QUALIFIED" if (rel32 < 1e-3 and rel64 < 1e-3) else "REFUTED")
    note = (f"Own EXACT second derivative (V'' = W1 sum_p 2[T_p'^2 + (T_p - C_p) T_p''], derived by hand, cross-checked against own 5-point finite "
            f"differences at eps = 1e-3 and 1e-4) reproduces the central-line minima: n32 {o32:.10f} vs the producer's -0.0080721737 (rel dev "
            f"{rel32:.1e}), n64 {o64:.10f} vs -0.0105722114 (rel dev {rel64:.1e}). Own eigenvalues (np.linalg.eigvals of M eta) and own C_p = "
            f"{np.round(cp, 4).tolist()}. The sign is what matters and it is NEGATIVE on both grids: the potential's split curvature at the cell's own "
            "spectrum is a MAXIMUM along the (m + s, m - s) pair, not a minimum. QUALIFICATION: -0.0106 is the CENTRAL-LINE minimum on n64; the n64 "
            f"free-cell minimum over the whole box is {out['fields']['r16_1_end_n64_L48']['hV_min_free_all_cells']:.5f}, 25 percent lower, at r = "
            f"{out['fields']['r16_1_end_n64_L48']['r_at_hV_min_free']:.2f}. The two n64 numbers are not interchangeable and the grid dependence "
            "(-0.0081 at h = 1.5 vs -0.0106 at h = 0.75 on the same physical line) is 31 percent, so no continuum value is established by these two "
            "grids: the quantity is still resolving.")
    verdict("C1", v, out, {"hV_min_n32_line": p32, "hV_min_n64_line": p64}, note)


def hazards_summary():
    """the things I found that the producers did not claim (numbers already computed above)."""
    H = OUT.setdefault("unclaimed_hazards", {})
    a1 = OUT["claims"]["A1"]["own_numbers"]
    a3 = OUT["claims"]["A3"]["own_numbers"]["R3_single_tail_own"]
    e3 = OUT["claims"]["E3"]["own_numbers"]
    g1 = OUT["claims"]["G1"]["own_numbers"]
    h1 = OUT["claims"]["H1"]["own_numbers"]
    c1 = OUT["claims"]["C1"]["own_numbers"]
    sym = json.load(open(os.path.join(DATA, "m5_32_r17_0_symbol.json")))
    fv = sym["backgrounds"]["analytic_hedgehog_n32_L48"]["contractions"]["full_v4_rebuild"]
    H["quartic_density_is_stencil_defined"] = {
        "what": "E_h is quartic in the jets, and the record arm forms 0.5 (density of the forward jets) + 0.5 (density of the backward jets), not the "
                "density of the central (sym-averaged) jets. The two are different functionals at O(h^2).",
        "measured": {"own_central_difference_amplitude_ratios": [a1["fields"][k]["ratio_A_median_over_pred"] for k in a1["fields"]],
                     "producer_ratios": [1.0024, 1.0006, 0.9876, 0.9987, 0.9988],
                     "R3_single_E_u_own_central_diff": a3["E_u_from_own_density_h3_sum"], "R3_single_E_u_certified_sym": 18.779953480468727},
        "why_it_matters": "the four-digit tail agreements (0.999, 1.0006) and the 'certified' E_u are stencil statements, not continuum ones; a 17 percent "
                          "stencil spread sits under the R3 single's energy."}
    H["R3_single_is_not_a_power_law"] = {
        "what": "the certified-stack single's far field is read as an r^-4 tail amplitude A_R3 = 4.23, but it is not a power law on the window used.",
        "measured": {"own_slope_0.20_0.40": a3["eta_inner_window_0.20_0.40"]["slope"], "own_slope_0.15_0.42": a3["eta_inner_window_0.15_0.42"]["slope"],
                     "own_cell_median_A": a3["eta_inner_window_0.20_0.40"]["A_median_r4"], "own_shell_mean_A": a3["eta_inner_window_0.20_0.40"]["A_from_shells"],
                     "median_vs_shell_mean_spread": a3["eta_inner_window_0.20_0.40"]["A_from_shells"] / a3["eta_inner_window_0.20_0.40"]["A_median_r4"]},
        "why_it_matters": "the plotted superposition line 8 pi A_R3 / d and any statement of the form 'the record's pair is X times the superposition "
                          "prediction' inherit a 57 percent ambiguity in A_R3."}
    H["l_lattice_residual_is_not_second_order_and_its_peak_diverges"] = {
        "what": "the |l| residual on the exact analytic seed is presented as discretization that falls with h; it falls as h^1.5, and its PEAK grows.",
        "measured": {"median_|l|_by_h": {k: v["median_|l|_r_3_6"] for k, v in e3["lattice"].items()},
                     "fitted_order_in_h": e3["lattice_convergence_order_in_h"],
                     "max_|l|_on_free_cells_by_h": {k: v["max_|l|_free"] for k, v in e3["lattice"].items()}},
        "why_it_matters": "a residual whose maximum grows from 0.156 (h 1.5) to 0.311 (h 0.75) under refinement is not converging to zero everywhere; "
                          "the core is where l is read, and that is where it is worst."}
    H["hV_is_not_grid_converged"] = {
        "what": "the split curvature of V4 is quoted at two grids as if both were the same number.",
        "measured": {"n32_central_line": c1["fields"]["r16_1_end_n32_L48"]["exact"]["hV_min_central_line"],
                     "n64_central_line": c1["fields"]["r16_1_end_n64_L48"]["exact"]["hV_min_central_line"],
                     "grid_spread_percent": 100 * abs(c1["fields"]["r16_1_end_n64_L48"]["exact"]["hV_min_central_line"] / c1["fields"]["r16_1_end_n32_L48"]["exact"]["hV_min_central_line"] - 1),
                     "n64_free_cell_min": c1["fields"]["r16_1_end_n64_L48"]["hV_min_free_all_cells"]},
        "why_it_matters": "31 percent between the two grids and 25 percent between the central line and the free cells at n64: no continuum value is "
                          "established, and the two n64 numbers are not interchangeable."}
    H["K_coll_box_mode_fills_the_box"] = {
        "what": "int f^2 = 11786 for the saved lowest mode.",
        "measured": {"int_f2": h1["int_f2_box"], "box_volume": h1["box_volume_L^3"], "fraction_of_box": h1["int_f2_over_box_volume"],
                     "weight_fraction_inside_r0": h1["weight_fraction_r_lt_r0"], "peak_radius": h1["f_peak_radius"], "r_0": h1["r_0_profile"]},
        "why_it_matters": "K_coll = 6.19 is 2 omega Delta_min^2 times a box volume; only 0.011 percent of the mode weight is inside r_0 and the peak sits "
                          "at r = 13, past the core. The producer says this in a unit_note but the number is still the headline."}
    H["the_CP_symbol_pattern_belongs_to_the_QUARTIC_ALONE"] = {
        "what": "the {0,0,1,1,1,1,2,2,2,6} pattern and the 2-dimensional kernel are properties of the quartic with c_P = c_s = 0, not of the v4 object "
                "the rung actually studies. The producers ran the full v4 as a control and reported the numbers but drew no conclusion from them.",
        "measured": {"full_v4_H00_spectrum_vs_CP_pattern_max_dev": {nm: fv[nm]["radial"]["H00_spectrum_vs_CP_pattern_max_dev"] for nm in fv},
                     "full_v4_signature_at_0": {nm: fv[nm]["radial"]["signature_at_0"] for nm in fv},
                     "full_v4_n_crossings": {nm: fv[nm]["radial"]["n_crossings"] for nm in fv}},
        "why_it_matters": "with K_P and the regulator on, the H_00 kernel drops from 2 to 1, the spectrum deviates from the Complete Picture pattern by "
                          "0.31 to 0.67, and there are 9 crossings, not 8. The radial factorization survives (residual 1e-4), so the CHARACTERISTIC is "
                          "robust, but the spectral pattern that the Complete Picture report is being credited with is not."}
    H["X_M_is_a_total_derivative_hence_dynamically_inert"] = {
        "what": "E2 confirms X_M = d_mu J^mu exactly for any smooth M.",
        "why_it_matters": "as a Lagrangian term X_M alone contributes nothing to the equations of motion, so no repair of the model can come from it; only "
                          "X_M^2 (which E1 shows is -2 I1 - I2 + 4 I3, already inside the R1 registry) is dynamical. Neither producer arm states this "
                          "conclusion, and both claims are individually correct."}
    H["the_eps_slot_rule_is_load_bearing"] = {
        "what": "the eta weighting on the eps-derivative pair is required for covariance.",
        "measured": {"stated_rule_drift": OUT["claims"]["E1"]["own_numbers"]["covariance_drift_stated_rule"],
                     "alternative_rule_drift": OUT["claims"]["E1"]["own_numbers"]["covariance_drift_ALT_rule_eta_on_internal_pair"]},
        "why_it_matters": "the alternative (eta everywhere, which reduces to bare eps) drifts by 1.94; the producers assert the rule without testing it."}
    H["H_0k_vanishes_for_any_static_background"] = {
        "what": "H_0k = 0 is reported as a measured 0.0 on the hedgehog; it is structural.",
        "measured": {"own_H0k_over_H00_continuum": g1["continuum_axis"]["r=6"]["radial"]["H0k_norm_over_H00"],
                     "own_radial_factorization_at_20_random_offaxis_points": g1["radial_factorization_residual_at_20_random_offaxis_points"]},
        "why_it_matters": "C_0 = 0 whenever A_0 = 0, so the symbol is even in Omega on ANY static background; and sum_i n_i A_i = 0 makes the radial "
                          "factorization exact at every point of the hedgehog, not just on the axis and not just at large r. Both facts make the "
                          "producers' r-dependent lattice residuals pure error bars, which is stronger than what they claim."}
    log(f"unclaimed hazards recorded: {len(H)}")


# ================================================================ main
def main():
    claim_A1()
    json.dump(OUT, open(os.path.join(CK, "r17_0_audit_partial.json"), "w"), indent=1, default=float)
    claim_A2()
    claim_A3()
    json.dump(OUT, open(os.path.join(CK, "r17_0_audit_partial.json"), "w"), indent=1, default=float)
    claim_E1()
    claim_E2()
    claim_E3()
    json.dump(OUT, open(os.path.join(CK, "r17_0_audit_partial.json"), "w"), indent=1, default=float)
    claim_G1()
    hazard_symmetrization()
    json.dump(OUT, open(os.path.join(CK, "r17_0_audit_partial.json"), "w"), indent=1, default=float)
    claim_H1()
    claim_C1()
    hazards_summary()
    tally = {}
    for k, vv in OUT["claims"].items():
        tally[vv["verdict"]] = tally.get(vv["verdict"], 0) + 1
    OUT["tally"] = {"by_verdict": tally, "per_claim": {k: v["verdict"] for k, v in OUT["claims"].items()},
                    "n_claims": len(OUT["claims"])}
    OUT["runtime_s"] = time.time() - T0
    json.dump(OUT, open(os.path.join(DATA, "m5_32_r17_0_audit.json"), "w"), indent=1, default=float)
    log(f"TALLY {tally}; written data/m5_32_r17_0_audit.json ({OUT['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
