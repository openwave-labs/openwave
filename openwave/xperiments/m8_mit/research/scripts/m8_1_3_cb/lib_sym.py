"""Numerical search (double precision) for elements of Gamma fixing a vector of E_0.

Gamma acts on w by D(g) (the set of maps {conj D(g)} of left translation equals {D(g)}: conj U =
eps U eps^-1 with eps in SU(2)), by phases, and by the fibre group:
  R4, R5: charge conjugation w -> c Theta w (s02), so maps D(g) e^{ia} Theta^eps, eps in {0, 1};
  R2: Sp(1) = {alpha + beta Theta}, so maps D(g) q;  D(g) q u = u  iff  D(g) u in H u = span_C(u, Theta u).
Residual functions vanish exactly on the stabilizer; minima are polished and clustered.
"""
import numpy as np
from math import comb, factorial
from scipy.optimize import minimize, least_squares
from lib_rep import poly_D_entry_coeffs
from lib_cg import mrange, w as wfac


def D_np(j, q):
    """D^j for unit quaternion q = (w, x, y, z), this room's convention (lib_exact.quat_to_ab)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    a = complex(q[0], -q[3])
    b = complex(q[2], -q[1])
    ms = mrange(j)
    n = len(ms)
    D = np.zeros((n, n), complex)
    ca, cb = a.conjugate(), b.conjugate()
    for r, m in enumerate(ms):
        for c, mp_ in enumerate(ms):
            s = 0j
            for co, (ea, eb, eca, ecb) in poly_D_entry_coeffs(j, m, mp_):
                s += co * a ** ea * b ** eb * ca ** eca * cb ** ecb
            D[r, c] = s * (wfac(j, m) / wfac(j, mp_)) ** 0.5
    return D


def theta_np(j, u):
    ms = mrange(j)
    n = len(ms)
    return np.array([(-1) ** int(j - ms[r]) * np.conj(u[n - 1 - r]) for r in range(n)])


def residual(j, u, quaternionic, eps):
    u = np.asarray(u, complex)
    u = u / np.linalg.norm(u)
    if quaternionic:
        tu = theta_np(j, u)
        B = np.stack([u, tu], 1)  # orthonormal (<u, Theta u> = 0 for half-integer j)
        def f(q):
            v = D_np(j, q) @ u
            return np.linalg.norm(v - B @ (B.conj().T @ v)) ** 2
    else:
        src = theta_np(j, u) if eps else u
        def f(q):
            v = D_np(j, q) @ src
            return 1.0 - abs(np.vdot(u, v)) ** 2
    return f


def quat_to_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def D_batch(j, Q):
    """D^j for an (N, 4) array of unit quaternions, vectorised."""
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    a = Q[:, 0] - 1j * Q[:, 3]
    b = Q[:, 2] - 1j * Q[:, 1]
    ca, cb = a.conj(), b.conj()
    ms = mrange(j)
    n = len(ms)
    D = np.zeros((Q.shape[0], n, n), complex)
    for r, m in enumerate(ms):
        for c, mp_ in enumerate(ms):
            s = 0
            for co, (ea, eb, eca, ecb) in poly_D_entry_coeffs(j, m, mp_):
                s = s + co * a ** ea * b ** eb * ca ** eca * cb ** ecb
            D[:, r, c] = s * (wfac(j, m) / wfac(j, mp_)) ** 0.5
    return D


def residual_batch(j, u, quaternionic, eps, Q):
    u = np.asarray(u, complex)
    u = u / np.linalg.norm(u)
    D = D_batch(j, Q)
    if quaternionic:
        tu = theta_np(j, u)
        B = np.stack([u, tu], 1)
        v = D @ u
        return np.linalg.norm(v - (v @ B.conj()) @ B.T, axis=1) ** 2
    src = theta_np(j, u) if eps else u
    v = D @ src
    return 1.0 - np.abs(v @ u.conj()) ** 2


def stabilizer_search(j, u, quaternionic, eps=0, ngrid=200000, nkeep=400, seed=0, tol=1e-12):
    """grid of random unit quaternions -> keep the nkeep lowest residuals -> BFGS polish -> cluster
    the zeros (residual < tol) as rotations (g and -g give the same map)."""
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(ngrid, 4))
    vals = np.concatenate([residual_batch(j, u, quaternionic, eps, Q[k:k + 20000]) for k in range(0, ngrid, 20000)])
    order = np.argsort(vals)[:nkeep]
    uu = np.asarray(u, complex) / np.linalg.norm(u)
    if quaternionic:
        tu = theta_np(j, uu)
        B = np.stack([uu, tu], 1)
        src = uu
    else:
        B = uu[:, None]
        src = theta_np(j, uu) if eps else uu

    def vec(q):  # component of D(q) src orthogonal to the target line/plane; zero iff q stabilizes
        v = D_np(j, q) @ src
        r_ = v - B @ (B.conj().T @ v)
        return np.concatenate([r_.real, r_.imag])
    found = []
    for i in order:
        r = least_squares(vec, Q[i] / np.linalg.norm(Q[i]), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if np.linalg.norm(r.fun) < tol:
            q = r.x / np.linalg.norm(r.x)
            R = quat_to_rot(q)
            if not any(np.linalg.norm(R - R2) < 1e-6 for R2, _ in found):
                found.append((R, q))
    return found


def is_group(rots, tol=1e-6):
    """closure of a finite set of rotation matrices under products."""
    for A, _ in rots:
        for B, _ in rots:
            C = A @ B
            if not any(np.linalg.norm(C - R) < tol for R, _ in rots):
                return False
    return True
