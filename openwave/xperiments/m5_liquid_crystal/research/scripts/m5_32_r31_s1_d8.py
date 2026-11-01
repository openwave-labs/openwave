"""M5.32 R31 S1, D8 row: the vacuum conditions READ AT A PARAMETER POINT (not on a relaxed state).

EQUATIONS FIRST
---------------
M real symmetric 4x4, eta = diag(-1,1,1,1), N = M eta, roots q = (-g, 1, delta, delta) (degenerate vacuum),
M_vac = diag(g, 1, delta, delta), w = W1 * w1s.
    V4 = w sum_{p=1..4} (S_p - C_p)^2,  S_p = tr N^p,  C_p = sum_i q_i^p
    dS_p/dX = p tr(N^(p-1) X eta);  at a vacuum S_p = C_p, so with v_p[a] = dS_p/dX_a on the 10-basis
    Hess V4 = 2 w sum_p v_p v_p^T         (rank <= 4: the six O(1,3) orbit directions are exactly flat)
Kinetic cell matrix. The certified weight K(D) = 4 h^3 sum <[D,A_i]_eta,[D,A_i]_eta>_eta is built from the jets
A_i = d_i M, which vanish on the uniform vacuum, so K = 0 there (all 10 directions null). The quadratic
kinetic forms of the S2 list that survive at the vacuum are therefore also read (R29-0 section c):
    Q = lam_sel q_T + lam_s q_spec,  q_T(X) = tr(hXhX) - tr(eta X eta X),  h = eta + 2 (eta u)(eta u)^T,
    q_spec(X) = sum_p (1/2)(p tr(N^(p-1) X eta))^2     (kappa_p = 1)
Plane-wave pencil along khat: (k^2 Q + H) X = omega^2 Q X; norm sign = sign(X^T Q X). Q-null directions are
static (omega^2 -> infinity at any k, gapped by H on that subspace).
c6 multiplies a sextic in the deviatoric spatial block, I2^3 - 6 I3^2, which is quartic-or-higher in the
deviation and has exactly zero Hessian at the vacuum (its deviatoric spatial block vanishes for delta, delta
degenerate); it is stated, not computed.

The R29-0 module runs its whole analysis at import, so its small functions (sym_basis, M_vac_of, h_of_vac,
quad_matrix, q_T, q_spec, v4_of, hess_v4, hess_v4_fd_check) are mirrored here verbatim in convention.
cfg_of, params_of, roots_of and W1 are loaded by file path from R21 / R20 / R23-1.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    argv = sys.argv
    sys.argv = [argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


C23 = _load("m5_32_r23_1_cscan", "m5_32_r23_1_cscan.py")
R21, R20, R0, W1 = C23.R21, C23.R20, C23.R0, C23.W1


def sym_basis():
    out = []
    for a in range(4):
        X = np.zeros((4, 4)); X[a, a] = 1.0; out.append(X)
    for j in range(1, 4):
        X = np.zeros((4, 4)); X[0, j] = X[j, 0] = 1.0; out.append(X)
    for i, j in [(1, 2), (1, 3), (2, 3)]:
        X = np.zeros((4, 4)); X[i, j] = X[j, i] = 1.0; out.append(X)
    return np.array(out)


BX = sym_basis()


def h_of_vac(M):
    w, V = np.linalg.eig(M @ ETA)
    best = None
    for k in range(4):
        v = np.real(V[:, k]); n = v @ ETA @ v
        if n < 0 and (best is None or n < best[0]):
            best = (n, v / np.sqrt(-n))
    eu = ETA @ best[1]
    return ETA + 2.0 * np.outer(eu, eu)


def q_T(X, h):
    return float(np.trace(h @ X @ h @ X) - np.trace(ETA @ X @ ETA @ X))


def q_spec(X, M):
    N, tot, P = M @ ETA, 0.0, np.eye(4)
    for p in range(1, 5):
        tot += 0.5 * (p * np.trace(P @ X @ ETA)) ** 2
        P = P @ N
    return float(tot)


def quad_matrix(q):
    Q = np.zeros((10, 10))
    for a in range(10):
        for b in range(a, 10):
            Q[a, b] = Q[b, a] = 0.5 * (q(BX[a] + BX[b]) - q(BX[a]) - q(BX[b]))
    return Q


def v4_of(M, roots, w):
    N = M @ ETA
    tot, P = 0.0, np.eye(4)
    for p in range(1, 5):
        P = P @ N
        tot += (np.trace(P) - sum(q ** p for q in roots)) ** 2
    return float(w * tot)


def grad_sp(M):
    N = M @ ETA
    V, P = np.zeros((4, 10)), np.eye(4)
    for p in range(1, 5):
        for a in range(10):
            V[p - 1, a] = p * np.trace(P @ BX[a] @ ETA)
        P = P @ N
    return V


def grad_v4(M, roots, w):
    """10-vector dV4/dX_a at any M (not only a vacuum)."""
    N, P, g = M @ ETA, np.eye(4), np.zeros(10)
    V = grad_sp(M)
    for p in range(1, 5):
        P = P @ N
        g += 2.0 * w * (np.trace(P) - sum(q ** p for q in roots)) * V[p - 1]
    return g


def hess_v4(M, roots, w):
    V = grad_sp(M)
    return 2.0 * w * (V.T @ V)


def hess_v4_fd_check(M, roots, w, X, eps=1e-3):
    f = lambda e: v4_of(M + e * X, roots, w)
    d2 = lambda e: (-f(2 * e) + 16 * f(e) - 30 * f(0.0) + 16 * f(-e) - f(-2 * e)) / (12 * e * e)
    return float((16 * d2(eps / 2) - d2(eps)) / 15.0)


def _counts(ev, tol):
    return {"pos": int(np.sum(ev > tol)), "neg": int(np.sum(ev < -tol)), "zero": int(np.sum(np.abs(ev) <= tol))}


def hess_fd_check(M, roots, w, eps=1e-5):
    """max relative error of H against a central difference of the analytic gradient, plus the energy-level check."""
    H = hess_v4(M, roots, w)
    Hfd = np.zeros((10, 10))
    for b in range(10):
        Hfd[:, b] = (grad_v4(M + eps * BX[b], roots, w) - grad_v4(M - eps * BX[b], roots, w)) / (2 * eps)
    rel_grad = float(np.abs(H - Hfd).max() / max(np.abs(H).max(), 1e-300))
    rng = np.random.default_rng(31)
    rel_e = 0.0
    for _ in range(3):
        c = rng.normal(size=10); X = np.tensordot(c, BX, axes=1)
        exact, fd = float(c @ H @ c), hess_v4_fd_check(M, roots, w, X)
        rel_e = max(rel_e, abs(exact - fd) / max(abs(exact), 1e-300))
    return {"max_rel_err_vs_central_diff_of_gradient": rel_grad, "max_rel_err_vs_5pt_energy_richardson": rel_e}


def vacuum_d8_read(g, delta, w1s, c6=0.0, lam_s=0.0, lam_sel=0.0):
    cfg = R21.cfg_of(24, 36.0, g, delta)
    roots = R0.roots_of(cfg, degenerate=True)
    w = W1 * w1s
    M = np.diag([abs(roots[0]), roots[1], roots[2], roots[3]])
    H = hess_v4(M, roots, w)
    hev = np.linalg.eigvalsh(H)
    htol = 1e-9 * max(np.abs(hev).max(), 1e-300)
    res = {"g": g, "delta": delta, "w1s": w1s, "c6": c6, "lam_s": lam_s, "lam_sel": lam_sel, "roots": list(roots),
           "w": w, "vacuum_residual_max": float(max(abs(np.trace(np.linalg.matrix_power(M @ ETA, p)) - sum(q ** p for q in roots)) for p in range(1, 5)))}
    res["hessian"] = {"eigenvalues": hev.tolist(), "zero_modes": _counts(hev, htol)["zero"], "negative_modes": _counts(hev, htol)["neg"],
                      "positive_modes": _counts(hev, htol)["pos"], "fd_check": hess_fd_check(M, roots, w)}
    # certified K: jets vanish on the uniform vacuum
    res["K_cell_certified"] = {"eigenvalues": [0.0] * 10, "signature": {"pos": 0, "neg": 0, "zero": 10},
                               "reason": "K is quadratic in the jets A_i = d_i M, which are zero on a uniform vacuum"}
    h = h_of_vac(M)
    QT, QS = quad_matrix(lambda X: q_T(X, h)), quad_matrix(lambda X: q_spec(X, M))
    Q = lam_sel * QT + lam_s * QS
    qev = np.linalg.eigvalsh(Q)
    qtol = 1e-9 * max(np.abs(qev).max(), 1e-300)
    res["K_quadratic_forms"] = {"Q_T_signature": _counts(np.linalg.eigvalsh(QT), 1e-9), "Q_spec_signature": _counts(np.linalg.eigvalsh(QS), 1e-9),
                                "Q_total_signature": _counts(qev, qtol) if np.abs(qev).max() > 0 else {"pos": 0, "neg": 0, "zero": 10}}
    nc = ["c6 sextic: exactly zero Hessian at the vacuum (stated, not computed)"]
    if np.abs(qev).max() == 0:
        nc.append("plane-wave pencil: lam_s = lam_sel = 0 leaves no kinetic form (Q = 0), no propagating modes defined")
        res["pencil"] = None
    else:
        # pencil along khat = z, kappa = 1 on Q (R29-0 pencil_rows convention)
        w_, U = np.linalg.eigh(Q)
        N0 = U[:, np.abs(w_) <= qtol]
        gap = np.linalg.eigvalsh(N0.T @ H @ N0).tolist() if N0.shape[1] else []
        rows = []
        for kk in (0.0, 0.25, 0.5, 1.0):
            ww, V = sla.eig(kk * kk * Q + H, Q)
            for i in range(10):
                if not np.isfinite(ww[i]) or abs(ww[i]) > 1e12:
                    continue
                x = V[:, i]; nrm = float(np.real(np.conj(x) @ Q @ x))
                rows.append({"k": kk, "omega2": float(ww[i].real), "norm_sign": int(np.sign(nrm)) if abs(nrm) > 1e-12 else 0})
        prop = [r for r in rows if r["omega2"] > 1e-9 * max(abs(H).max(), 1e-300) and r["norm_sign"] != 0]
        res["pencil"] = {"khat": [0, 0, 1], "n_static_Q_null": int(N0.shape[1]), "H_on_Q_null_eigs": gap,
                         "gapped_stay_gapped": bool(all(e > htol for e in gap)) if gap else True,
                         "lowest_propagating_omega2_by_k": {str(k): sorted([r["omega2"] for r in prop if r["k"] == k])[:3] for k in (0.0, 0.25, 0.5, 1.0)},
                         "any_negative_norm_propagating": bool(any(r["norm_sign"] < 0 for r in prop)),
                         "n_negative_omega2_or_unstable": int(sum(r["omega2"] < -1e-9 for r in rows)), "modes": rows}
        nc.append("kappa_p = 1 and lam_sel on q_T, lam_s on q_spec is our reading of the two weights; certified K itself is zero at the vacuum")
    res["not_computed"] = nc
    return res


if __name__ == "__main__":
    for pt in [(8, 0.3, 25), (16, 0.3, 25), (8, 0.1, 25)]:
        r = vacuum_d8_read(*pt, lam_s=1.0, lam_sel=1.0)
        r0 = vacuum_d8_read(*pt)
        r["pencil"] = {k: v for k, v in (r["pencil"] or {}).items() if k != "modes"}
        print(json.dumps({"point": pt, "read_lam1": r, "read_lam0_hessian_only": {k: r0[k] for k in ("hessian", "not_computed")}}, indent=1))
