"""Independent NUMERIC check (double precision quadrature over SU(2)) of items 13, 14, 18, 21.

Nothing here uses the Peter-Weyl/Schur derivation: A, B, the projected cubic N(u), and the
psi psi^T quartic are integrated directly over a product Gauss grid on SU(2) (Euler angles).
Right-Gamma invariance means integrating over SU(2) = integrating over X.
"""
import json
import pathlib
from fractions import Fraction as Fr
from math import comb
import numpy as np
from audit_su2lib import cg
from audit_field import Q1, Q2

HERE = pathlib.Path(__file__).parent
res = {}
s5 = 5 ** 0.5
Pj = json.loads((HERE / "audit_projectors_level6.json").read_text())


def kval(t):
    a, b, c, d = [float(Fr(x)) for x in t]
    return complex(a + b * s5, c + d * s5)


sq = np.array([comb(6, k) ** 0.5 for k in range(7)])


def to_on(Pm):
    # T_on[k,l] = T_mon[k,l] s_l / s_k
    return Pm * sq[None, :] / sq[:, None]


P = {d: to_on(np.array([[kval(x) for x in row] for row in Pj[f"d{d}"]])) for d in (3, 4)}
for d, Pm in P.items():
    assert np.allclose(Pm @ Pm, Pm) and np.allclose(Pm, Pm.conj().T) and abs(np.trace(Pm) - d) < 1e-12

# ---- Euler grid ----
na, ng, nb = 28, 28, 40
al = 2 * np.pi * np.arange(na) / na
ga = 4 * np.pi * np.arange(ng) / ng
xb, wb = np.polynomial.legendre.leggauss(nb)  # x = cos(beta), weight dx = sin(beta) dbeta
be = np.arccos(xb)
A_, B_, G_ = np.meshgrid(al, be, ga, indexing="ij")
W = np.broadcast_to(wb[None, :, None], A_.shape) / (2.0 * na * ng)
W = W.ravel()
assert abs(W.sum() - 1) < 1e-13
A_, B_, G_ = A_.ravel(), B_.ravel(), G_.ravel()
c, s = np.cos(B_ / 2), np.sin(B_ / 2)
ea, eg = np.exp(-0.5j * A_), np.exp(-0.5j * G_)
# U = Rz(a) Ry(b) Rz(g), Rz = diag(e^{-i a/2}, e^{i a/2}), Ry = [[c,-s],[s,c]]
Ua = ea * c * eg
Ub = -ea * s * np.conj(eg)
Uc = np.conj(ea) * s * eg
Ud = np.conj(ea) * c * np.conj(eg)


def D3(a, b, cc, d):
    """orthonormal-basis spin-3 matrices, shape (npts,7,7), x -> a x + c y, y -> b x + d y."""
    npts = a.shape[0]
    Dm = np.zeros((npts, 7, 7), dtype=complex)
    for k in range(7):
        for i in range(k + 1):
            for l in range(7 - k):
                kp = i + l
                Dm[:, kp, k] += comb(k, i) * a ** i * cc ** (k - i) * comb(6 - k, l) * b ** l * d ** (6 - k - l)
    return Dm * sq[None, None, :] / sq[None, :, None]


D = D3(Ua, Ub, Uc, Ud)
# check unitarity on a sample
assert np.allclose(np.einsum("pij,pkj->pik", D[:50], D[:50].conj()), np.eye(7)[None], atol=1e-12)


def D3q(q):
    (a, b), (cc, d) = q.su2()
    return D3(np.array([a.tocomplex()]), np.array([b.tocomplex()]), np.array([cc.tocomplex()]), np.array([d.tocomplex()]))[0]


Dq1, Dq2 = D3q(Q1), D3q(Q2)

# ---- numeric CG tables and maps ----
CG = {}


def cgf(j1, m1, j2, m2, J, M):
    key = (j1, m1, j2, m2, J, M)
    if key not in CG:
        CG[key] = float(cg(j1, m1, j2, m2, J, M))
    return CG[key]


def theta(u):
    return np.array([(-1) ** abs(m) * np.conj(u[-m + 3]) for m in range(-3, 4)])


def couple(xv, yv, j1, j2, J):
    out = np.zeros(2 * J + 1, dtype=complex)
    for iN, N in enumerate(range(-J, J + 1)):
        for m1 in range(-j1, j1 + 1):
            m2 = N - m1
            if abs(m2) <= j2:
                out[iN] += cgf(j1, m1, j2, m2, J, N) * xv[m1 + j1] * yv[m2 + j2]
    return out


def rho(u, K):
    return couple(u, theta(u), 3, 3, K)


def Mmap(u, K):
    return couple(rho(u, K), u, K, 3, 3)


def Nmap(u, J):
    B = couple(u, u, 3, 3, J)
    out = np.zeros(7, dtype=complex)
    for n_ in range(-3, 4):
        for iN, N in enumerate(range(-J, J + 1)):
            if abs(N - n_) <= 3:
                out[n_ + 3] += cgf(3, n_, 3, N - n_, J, N) * np.conj(u[N - n_ + 3]) * B[iN]
    return out


def MK_of_P(Pm, K):
    out = np.zeros(2 * K + 1, dtype=complex)
    for iN, N in enumerate(range(-K, K + 1)):
        for n_ in range(-3, 4):
            npr = N - n_
            if abs(npr) <= 3:
                out[iN] += cgf(3, n_, 3, npr, K, N) * (-1) ** abs(npr) * Pm[n_ + 3, -npr + 3]
    return out


rng = np.random.default_rng(2026)
out = {}
for d in (3, 4):
    Pm = P[d]
    ev, V = np.linalg.eigh(Pm)
    eta = V[:, ev > 0.5]
    assert eta.shape[1] == d and np.allclose(eta.conj().T @ eta, np.eye(d))
    # right-Gamma equivariance of eta: D(h) eta = eta sigma(h)
    for Dh in (Dq1, Dq2):
        sig = eta.conj().T @ Dh @ eta
        assert np.allclose(Dh @ eta, eta @ sig, atol=1e-12)
    R2 = {K: float(np.sum(abs(MK_of_P(Pm, K)) ** 2)) for K in range(7)}
    rec = {"||R_K||^2_numeric": R2}
    Qs, fitsN = [], []
    for trial in range(4):
        u = rng.normal(size=7) + 1j * rng.normal(size=7)
        psi = np.einsum("m,pmn,na->pa", u, D, eta)
        dens = np.sum(abs(psi) ** 2, axis=1)
        Bq = np.sum(W * dens)
        Aq = np.sum(W * dens ** 2)
        nu2 = np.vdot(u, u).real
        r6 = np.sum(abs(rho(u, 6)) ** 2) / nu2 ** 2
        Qs.append({"B_quad": Bq, "B_formula d/7|u|^2": d / 7 * nu2, "Q_quad": Aq / Bq ** 2, "rhat6": r6})
        # projected cubic: (d/7) u'_m = int sum_a conj(psi_{v_m,a}) |psi|^2 psi_a
        f = dens[:, None] * psi
        psis_basis = np.einsum("pmn,na->pma", D, eta)  # psi_{v_m}
        up = (7 / d) * np.einsum("p,pma,pa->m", W, psis_basis.conj(), f)
        Mat = np.stack([Mmap(u, 0), Mmap(u, 6)], axis=1)
        coef, *_ = np.linalg.lstsq(Mat, up, rcond=None)
        resid = np.linalg.norm(Mat @ coef - up) / np.linalg.norm(up)
        fitsN.append({"coef_M0": [coef[0].real, coef[0].imag], "coef_M6": [coef[1].real, coef[1].imag], "rel_resid": resid,
                      "beta_check_<u,N>/|u|^2_over_(A/B)": (np.vdot(u, up).real / nu2) / (Aq / Bq)})
    # linear regression of Q on rhat6 across trials
    X = np.array([[1, q["rhat6"]] for q in Qs])
    yv = np.array([q["Q_quad"] for q in Qs])
    fit = np.linalg.lstsq(X, yv, rcond=None)[0]
    rec["Q_trials"] = Qs
    rec["Q_fit_intercept_slope"] = list(fit)
    rec["N_fits"] = fitsN
    # ---------- item 21: Theta-real eta, psi psi^T quartic ----------
    cands = []
    for a in range(d):
        e = eta[:, a]
        cands.append(e + theta(e))
        cands.append(1j * (e - theta(e)))
    C = np.array(cands).T
    Gm = (C.conj().T @ C).real
    w_, Vv = np.linalg.eigh(Gm)
    keep = Vv[:, w_ > 1e-9]
    Eb = C @ keep
    # orthonormalise within the real span (Gram of Theta-fixed vectors is real)
    Gm2 = (Eb.conj().T @ Eb)
    assert np.allclose(Gm2.imag, 0, atol=1e-10)
    L = np.linalg.cholesky(Gm2.real)
    etaR = Eb @ np.linalg.inv(L).T
    assert etaR.shape[1] == d and np.allclose(etaR.conj().T @ etaR, np.eye(d), atol=1e-10)
    assert all(np.allclose(theta(etaR[:, a]), etaR[:, a], atol=1e-10) for a in range(d))
    assert np.allclose(etaR @ etaR.conj().T, Pm, atol=1e-10)
    for Dh in (Dq1, Dq2):
        sig = etaR.conj().T @ Dh @ etaR
        assert np.allclose(sig.imag, 0, atol=1e-10) and np.allclose(sig @ sig.T, np.eye(d), atol=1e-10)
    # p' = eta eta^T as a V(x)V vector vs w(P) = sum (-1)^{n'} P_{n,-n'} v_n (x) v_{n'}
    Pp = etaR @ etaR.T
    wP = np.array([[(-1) ** abs(npr) * Pm[n_ + 3, -npr + 3] for npr in range(-3, 4)] for n_ in range(-3, 4)])
    rec["item21_pprime_equals_wP"] = bool(np.allclose(Pp, wP, atol=1e-10))
    # with a non-Theta-real eta (generic phase) sigma is not orthogonal: check the invariance fails
    etaX = eta * np.exp(0.3j)  # still gives the same P; sigma unchanged, test instead a complex-rotated basis
    Uc_ = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))[0]
    etaC = etaR @ Uc_
    sigC = etaC.conj().T @ Dq1 @ etaC
    rec["item21_generic_unitary_basis_sigma_sigmaT_is_I"] = bool(np.allclose(sigC @ sigC.T, np.eye(d), atol=1e-8))
    T21 = []
    for trial in range(3):
        u = rng.normal(size=7) + 1j * rng.normal(size=7)
        psi = np.einsum("m,pmn,na->pa", u, D, etaR)
        pt = np.sum(psi * psi, axis=1)  # psi psi^T
        Eq = np.sum(W * abs(pt) ** 2)
        # invariance of psi psi^T under right translation by generators at a few points
        g0 = D[:5]
        inv_ok = all(np.allclose(np.einsum("m,pmn,na->pa", u, g0 @ Dh, etaR).__pow__(2).sum(axis=1),
                                 np.einsum("m,pmn,na->pa", u, g0, etaR).__pow__(2).sum(axis=1)) for Dh in (Dq1, Dq2))
        form = sum(np.sum(abs(couple(u, u, 3, 3, J)) ** 2) * R2[J] / (2 * J + 1) for J in range(7))
        f = pt[:, None] * psi.conj()
        psis_basis = np.einsum("pmn,na->pma", D, etaR)
        up = (7 / d) * np.einsum("p,pma,pa->m", W, psis_basis.conj(), f)
        Mat = np.stack([Nmap(u, 0), Nmap(u, 6)], axis=1)
        coef, *_ = np.linalg.lstsq(Mat, up, rcond=None)
        resid = np.linalg.norm(Mat @ coef - up) / np.linalg.norm(up)
        T21.append({"invariance_psipsiT": inv_ok, "E_quad": Eq, "E_formula": form,
                    "coef_N0": [coef[0].real, coef[0].imag], "coef_N6": [coef[1].real, coef[1].imag], "rel_resid": resid})
    rec["item21"] = T21
    out[f"d{d}"] = rec
res["quadrature"] = out
(HERE / "audit_res_quadrature.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
