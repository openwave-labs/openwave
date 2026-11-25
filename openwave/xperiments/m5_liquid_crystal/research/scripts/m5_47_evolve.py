"""M5.47: the action of M5.42 evolved in time, and the controls K0 to K5 of the time-stepper.

The field is a real symmetric 3 x 3 matrix M(x) on an N^3 grid of spacing h, six components per node in the
order (00, 01, 02, 11, 12, 22). The static part is the action of M5.42 at J = 0 (m5_42_eq42.py), on the
mirror-averaged stencil:

    U = h^3 sum_n [ L1 sum_i |d_i M|^2 + kappa sum_{i<j} |[d_i M, d_j M]|^2 + v sum_k (tr M^k - c_k)^2 ]

The time part is the form whose value on the twist is the inertia of M5.42 (R is the rate of M at the node,
the differences and the weights wn, wd are those of the kernel of m5_42_eq42.py):

    T = h^3 sum_n [ L1 wn |R|^2 + kappa sum_{i, s} wd |[R, d_i^s M]|^2 ]        T(R = Omega [G, M]) = I Omega^2 / 2

    p = dT/dR = h^3 A(M) R        A = 2 L1 wn g + 2 kappa sum wd ( . D^2 + D^2 . - 2 D . D )   (6 x 6 per node)
    dM/dt = R = A^-1 p / h^3      dp/dt = dT/dM - dU/dM            E = T + U

with g = diag(1, 2, 2, 1, 2, 1) the metric of the six components. The nodes of the outer shell are held (R = 0
there). The integrator is the classical fourth-order Runge-Kutta scheme on (M, p).

One pass over the nodes per stage (`stage`): the differences, A and its solve, the force of U and the gradient
of T are built together at each node.

    K0  the U and the force of `stage` against m5_42_eq42.Model.evaluate                (added; not in the plan)
    K1  T = I Omega^2 / 2 on the twist, the I of m5_42_eq42.kernel (the seed and B0a)
    K2  dT/dR and dT/dM against central differences of T at 200 random interior entries (B0a)
    K3  A symmetric and positive at every node; A A^-1 p = p (B0a and the seed)
    K4  E1: the drift of E at steps 0.015 and 0.03
    K5  E1 run back with p reversed: the return at steps 0.015 and 0.03

Usage:  python m5_47_evolve.py --controls            (K0 to K3, then E1 at the two steps in two processes)
        python m5_47_evolve.py --e1 0.015            (one E1 run; writes its summary)
        python m5_47_evolve.py --time                (seconds per step on the four grids)
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from numba import njit

import m5_42_eq42 as Q

DELTA, L1, KAPPA, V = 0.05, 0.03, 1.0, 1.0
PA = np.array([0, 0, 0, 1, 1, 2])
PB = np.array([0, 1, 2, 1, 2, 2])
METRIC = np.array([1.0, 2.0, 2.0, 1.0, 2.0, 1.0])
SOLVE, RATE, STATIC = 0, 1, 2
BASE = dict(B0a=(8.0, 33), B0b=(8.0, 65), B0c=(10.0, 41), B0d=(10.0, 81))


@njit(cache=True, inline="always")
def _mm(A, B, out):
    for a in range(3):
        for b in range(3):
            out[a, b] = A[a, 0] * B[0, b] + A[a, 1] * B[1, b] + A[a, 2] * B[2, b]


@njit(cache=True, inline="always")
def _comm(A, B, tmp, out):
    _mm(A, B, out)
    _mm(B, A, tmp)
    for a in range(3):
        for b in range(3):
            out[a, b] -= tmp[a, b]


@njit(cache=True, inline="always")
def _weights(i, j, k, N, wts, val, wdv):
    """The validity of the six one-sided differences at the node, the node weight wn, and the weights wd."""
    val[0, 0] = i > 0
    val[0, 1] = i < N - 1
    val[1, 0] = j > 0
    val[1, 1] = j < N - 1
    val[2, 0] = k > 0
    val[2, 1] = k < N - 1
    wn = 0.0
    for a in range(2):
        if val[0, a]:
            for b in range(2):
                if val[1, b]:
                    for c in range(2):
                        if val[2, c]:
                            wn += wts[a, b, c]
    for ax in range(3):
        o1 = (ax + 1) % 3
        o2 = (ax + 2) % 3
        for s in range(2):
            wd = 0.0
            if val[ax, s]:
                for p in range(2):
                    if val[o1, p]:
                        for q in range(2):
                            if val[o2, q]:
                                if ax == 0:
                                    wd += wts[s, p, q]
                                elif ax == 1:
                                    wd += wts[q, s, p]
                                else:
                                    wd += wts[p, q, s]
            wdv[ax, s] = wd
    return wn


@njit(cache=True, inline="always")
def _differences(m6, i, j, k, h, val, IDX, Mn, D):
    for a in range(3):
        for b in range(3):
            Mn[a, b] = m6[i, j, k, IDX[a, b]]
    for ax in range(3):
        for s in range(2):
            if val[ax, s]:
                ii = i
                jj = j
                kk = k
                step = 2 * s - 1
                if ax == 0:
                    ii += step
                elif ax == 1:
                    jj += step
                else:
                    kk += step
                for a in range(3):
                    for b in range(3):
                        D[ax, s, a, b] = (m6[ii, jj, kk, IDX[a, b]] - Mn[a, b]) / h


@njit(cache=True, inline="always")
def _build_a(D, wdv, wn, L1, kap, PA, PB, S2, T4, At):
    """At (6 x 6): the Hessian of  L1 wn |R|^2 + kappa sum wd |[R, D]|^2  over the six components of R."""
    for a in range(3):
        for b in range(3):
            S2[a, b] = 0.0
            for c in range(3):
                for d in range(3):
                    T4[a, b, c, d] = 0.0
    for ax in range(3):
        for s in range(2):
            wd = wdv[ax, s]
            if wd > 0.0:
                for a in range(3):
                    for b in range(3):
                        s2 = 0.0
                        for c in range(3):
                            s2 += D[ax, s, a, c] * D[ax, s, c, b]
                        S2[a, b] += wd * s2
                        dab = wd * D[ax, s, a, b]
                        for c in range(3):
                            for d in range(3):
                                T4[a, b, c, d] += dab * D[ax, s, c, d]
    for m in range(6):
        a = PA[m]
        b = PB[m]
        for n in range(6):
            c = PA[n]
            d = PB[n]
            # X = E S2 + S2 E - 2 sum wd D E D for E the unit of the pair (a, b); its entries (c, d) and (d, c)
            x1 = -2.0 * T4[c, a, d, b]
            x2 = -2.0 * T4[d, a, c, b]
            if c == a:
                x1 += S2[b, d]
            if d == b:
                x1 += S2[c, a]
            if d == a:
                x2 += S2[b, c]
            if c == b:
                x2 += S2[d, a]
            if a != b:
                x1 += -2.0 * T4[c, b, d, a]
                x2 += -2.0 * T4[d, b, c, a]
                if c == b:
                    x1 += S2[a, d]
                if d == a:
                    x1 += S2[c, b]
                if d == b:
                    x2 += S2[a, c]
                if c == a:
                    x2 += S2[d, b]
            if c == d:
                At[n, m] = 2.0 * kap * x1
            else:
                At[n, m] = 2.0 * kap * (x1 + x2)
        At[m, m] += 2.0 * L1 * wn * (1.0 if a == b else 2.0)


@njit(cache=True, inline="always")
def _solve6(A, b, x, Lw):
    """x = A^-1 b by Cholesky; False when A is not positive."""
    for i in range(6):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= Lw[i, k] * Lw[j, k]
            if i == j:
                if s <= 0.0:
                    return False
                Lw[i, i] = np.sqrt(s)
            else:
                Lw[i, j] = s / Lw[j, j]
    for i in range(6):
        s = b[i]
        for k in range(i):
            s -= Lw[i, k] * x[k]
        x[i] = s / Lw[i, i]
    for i in range(5, -1, -1):
        s = x[i]
        for k in range(i + 1, 6):
            s -= Lw[k, i] * x[k]
        x[i] = s / Lw[i, i]
    return True


@njit(cache=True)
def stage(m6, p6, r6, f6, wts, h, L1, kap, v, c1, c2, c3, mode, with_U, IDX, PA, PB):
    """One pass over the nodes. Returns (U2, U4, UV, T, bad).

    mode 0 (solve)   p6 is read; r6 receives the rate A^-1 p / h^3 at the interior nodes and zero on the shell
    mode 1 (rate)    r6 is read at every node; p6 receives h^3 A r
    mode 2 (static)  no rate: T = 0, p6 and r6 are not touched

    f6 receives dT/dM - dU/dM over the six components (without the U part when with_U is False), set to zero
    on the shell in mode 0. bad counts the nodes where A was not positive."""
    N = m6.shape[0]
    h3 = h * h * h
    U2 = 0.0
    U4 = 0.0
    UV = 0.0
    T = 0.0
    bad = 0
    D = np.zeros((3, 2, 3, 3))
    gD = np.zeros((3, 2, 3, 3))
    val = np.zeros((3, 2), dtype=np.bool_)
    wdv = np.zeros((3, 2))
    Mn = np.zeros((3, 3))
    M2 = np.zeros((3, 3))
    Rn = np.zeros((3, 3))
    C = np.zeros((3, 3))
    Tm = np.zeros((3, 3))
    T2 = np.zeros((3, 3))
    gM = np.zeros((3, 3))
    S2 = np.zeros((3, 3))
    T4 = np.zeros((3, 3, 3, 3))
    At = np.zeros((6, 6))
    Lw = np.zeros((6, 6))
    rv = np.zeros(6)
    pv = np.zeros(6)
    f6[:] = 0.0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                wn = _weights(i, j, k, N, wts, val, wdv)
                _differences(m6, i, j, k, h, val, IDX, Mn, D)
                shell = i == 0 or j == 0 or k == 0 or i == N - 1 or j == N - 1 or k == N - 1
                active = False
                if mode == 1 or (mode == 0 and not shell):
                    _build_a(D, wdv, wn, L1, kap, PA, PB, S2, T4, At)
                    if mode == 1:
                        for c in range(6):
                            rv[c] = r6[i, j, k, c]
                        for c in range(6):
                            s = 0.0
                            for d in range(6):
                                s += At[c, d] * rv[d]
                            pv[c] = h3 * s
                            p6[i, j, k, c] = pv[c]
                    else:
                        for c in range(6):
                            pv[c] = p6[i, j, k, c]
                            rv[c] = pv[c] / h3
                        if not _solve6(At, rv, rv, Lw):
                            bad += 1
                            for c in range(6):
                                rv[c] = 0.0
                        for c in range(6):
                            r6[i, j, k, c] = rv[c]
                    s = 0.0
                    for c in range(6):
                        s += rv[c] * pv[c]
                    T += 0.5 * s
                    for a in range(3):
                        for b in range(3):
                            Rn[a, b] = rv[IDX[a, b]]
                    active = True
                elif mode == 0:
                    for c in range(6):
                        r6[i, j, k, c] = 0.0
                for ax in range(3):
                    for s in range(2):
                        for a in range(3):
                            for b in range(3):
                                gD[ax, s, a, b] = 0.0
                for a in range(3):
                    for b in range(3):
                        gM[a, b] = 0.0
                if with_U and wn > 0.0:
                    _mm(Mn, Mn, M2)
                    t1 = Mn[0, 0] + Mn[1, 1] + Mn[2, 2] - c1
                    t2 = M2[0, 0] + M2[1, 1] + M2[2, 2] - c2
                    t3 = -c3
                    for a in range(3):
                        for b in range(3):
                            t3 += M2[a, b] * Mn[b, a]
                    UV += wn * (t1 * t1 + t2 * t2 + t3 * t3)
                    for a in range(3):
                        for b in range(3):
                            gM[a, b] += wn * v * (4.0 * t2 * Mn[a, b] + 6.0 * t3 * M2[a, b])
                        gM[a, a] += wn * v * 2.0 * t1
                # the terms with one difference
                for ax in range(3):
                    for s in range(2):
                        wd = wdv[ax, s]
                        if wd <= 0.0:
                            continue
                        Dn = D[ax, s]
                        if with_U:
                            s2 = 0.0
                            for a in range(3):
                                for b in range(3):
                                    s2 += Dn[a, b] * Dn[a, b]
                                    gD[ax, s, a, b] += wd * 2.0 * L1 * Dn[a, b]
                            U2 += wd * s2
                        if active:
                            _comm(Rn, Dn, Tm, C)
                            _comm(Rn, C, Tm, T2)
                            for a in range(3):
                                for b in range(3):
                                    gD[ax, s, a, b] -= wd * 2.0 * kap * T2[a, b]
                # the terms with two differences
                if with_U:
                    for ax in range(3):
                        bx = (ax + 1) % 3
                        cx = (ax + 2) % 3
                        for s in range(2):
                            if not val[ax, s]:
                                continue
                            for t in range(2):
                                if not val[bx, t]:
                                    continue
                                wc = 0.0
                                for q in range(2):
                                    if val[cx, q]:
                                        if ax == 0:
                                            wc += wts[s, t, q]
                                        elif ax == 1:
                                            wc += wts[q, s, t]
                                        else:
                                            wc += wts[t, q, s]
                                if wc <= 0.0:
                                    continue
                                A = D[ax, s]
                                B = D[bx, t]
                                _comm(A, B, Tm, C)
                                s2 = 0.0
                                for a in range(3):
                                    for b in range(3):
                                        s2 += C[a, b] * C[a, b]
                                U4 += wc * s2
                                _comm(C, B, Tm, T2)
                                for a in range(3):
                                    for b in range(3):
                                        gD[ax, s, a, b] += wc * 2.0 * kap * T2[a, b]
                                _comm(A, C, Tm, T2)
                                for a in range(3):
                                    for b in range(3):
                                        gD[bx, t, a, b] += wc * 2.0 * kap * T2[a, b]
                # f = -(dU/dM - dT/dM), over the six components
                for c in range(6):
                    a = PA[c]
                    b = PB[c]
                    g = gM[a, b] if a == b else gM[a, b] + gM[b, a]
                    f6[i, j, k, c] -= h3 * g
                for ax in range(3):
                    for s in range(2):
                        if val[ax, s]:
                            ii = i
                            jj = j
                            kk = k
                            step = 2 * s - 1
                            if ax == 0:
                                ii += step
                            elif ax == 1:
                                jj += step
                            else:
                                kk += step
                            for c in range(6):
                                a = PA[c]
                                b = PB[c]
                                g = gD[ax, s, a, b] if a == b else gD[ax, s, a, b] + gD[ax, s, b, a]
                                g *= h3 / h
                                f6[ii, jj, kk, c] -= g
                                f6[i, j, k, c] += g
    if mode == 0:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if i == 0 or j == 0 or k == 0 or i == N - 1 or j == N - 1 or k == N - 1:
                        for c in range(6):
                            f6[i, j, k, c] = 0.0
    return h3 * L1 * U2, h3 * kap * U4, h3 * v * UV, T, bad


@njit(cache=True)
def a_field(m6, wts, h, L1, kap, IDX, PA, PB, out):
    """out (N, N, N, 6, 6) receives A at every node (p = h^3 A R)."""
    N = m6.shape[0]
    D = np.zeros((3, 2, 3, 3))
    val = np.zeros((3, 2), dtype=np.bool_)
    wdv = np.zeros((3, 2))
    Mn = np.zeros((3, 3))
    S2 = np.zeros((3, 3))
    T4 = np.zeros((3, 3, 3, 3))
    At = np.zeros((6, 6))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                wn = _weights(i, j, k, N, wts, val, wdv)
                _differences(m6, i, j, k, h, val, IDX, Mn, D)
                _build_a(D, wdv, wn, L1, kap, PA, PB, S2, T4, At)
                for a in range(6):
                    for b in range(6):
                        out[i, j, k, a, b] = At[a, b]


class Field:
    """The grid, the couplings and the stage of the equations of motion."""

    def __init__(self, N, half_box, delta=DELTA, L1=L1, kappa=KAPPA, v=V):
        self.N, self.half_box, self.delta, self.L1, self.kappa, self.v = N, half_box, delta, L1, kappa, v
        self.x = np.linspace(-half_box, half_box, N)
        self.h = self.x[1] - self.x[0]
        self.wts = Q.AVERAGED
        self.ck = (1 + delta, 1 + delta ** 2, 1 + delta ** 3)
        self.shape6 = (N, N, N, 6)
        self.interior = np.zeros((N, N, N), bool)
        self.interior[1:-1, 1:-1, 1:-1] = True
        self._scratch = np.zeros(self.shape6)

    def _stage(self, m6, p6, r6, f6, mode, with_U):
        U2, U4, UV, T, bad = stage(m6, p6, r6, f6, self.wts, self.h, self.L1, self.kappa, self.v, self.ck[0],
                                   self.ck[1], self.ck[2], mode, with_U, Q.IDX, PA, PB)
        return dict(U2=U2, U4=U4, UV=UV, U=U2 + U4 + UV, T=T, E=U2 + U4 + UV + T, bad=int(bad))

    def rhs(self, m6, p6, r6, f6):
        """The rate (zero on the shell) and the force from (M, p); the energies of the state."""
        return self._stage(m6, p6, r6, f6, SOLVE, True)

    def momentum(self, m6, r6, p6, f6=None, with_U=False):
        """p = h^3 A r for a rate given at every node; T; with f6, also dT/dM (and -dU/dM with with_U)."""
        return self._stage(m6, p6, r6, self._scratch if f6 is None else f6, RATE, with_U)

    def static(self, m6, f6):
        """U and -dU/dM over the six components (every node, the shell included)."""
        return self._stage(m6, self._scratch, self._scratch, f6, STATIC, True)

    def a_matrices(self, m6):
        out = np.zeros((self.N,) * 3 + (6, 6))
        a_field(m6, self.wts, self.h, self.L1, self.kappa, Q.IDX, PA, PB, out)
        return out


class RK4:
    """The classical fourth-order Runge-Kutta scheme on (M, p), in place."""

    def __init__(self, field):
        self.F = field
        z = lambda: np.zeros(field.shape6)
        self.r, self.f, self.am, self.ap, self.mt, self.pt = z(), z(), z(), z(), z(), z()

    def step(self, m, p, dt):
        """Advances (m, p) by dt; returns the energies of the state before the step."""
        F, r, f, am, ap, mt, pt = self.F, self.r, self.f, self.am, self.ap, self.mt, self.pt
        e = F.rhs(m, p, r, f)
        np.copyto(am, r)
        np.copyto(ap, f)
        for w, c in ((2.0, 0.5 * dt), (2.0, 0.5 * dt), (1.0, dt)):
            np.multiply(r, c, out=mt)
            mt += m
            np.multiply(f, c, out=pt)
            pt += p
            e2 = F.rhs(mt, pt, r, f)
            e["bad"] += e2["bad"]
            np.multiply(r, w, out=mt)
            am += mt
            np.multiply(f, w, out=pt)
            ap += pt
        am *= dt / 6.0
        ap *= dt / 6.0
        m += am
        p += ap
        return e


# ----------------------------------------------------------------------------- states

def load_base(name, data):
    hb, N = BASE[name]
    z = np.load(data / f"m5_46_{name}_end.npz", allow_pickle=False)
    m6 = np.ascontiguousarray(z["m6"])
    assert m6.shape == (N, N, N, 6) and abs(z["x"][-1] - hb) < 1e-12
    return m6, hb, N


def twist_rate(model, m6):
    """K = [G, M] over the six components, at every node."""
    M = Q.from6(m6)
    K = model.G @ M - M @ model.G
    return Q.to6(K)


def smooth_noise(rng, shape, sigma_cells, interior):
    """White Gaussian noise smoothed by a Gaussian, unit root mean square over the interior, zero on the shell."""
    from scipy.ndimage import gaussian_filter

    w = gaussian_filter(rng.standard_normal(shape), sigma_cells, mode="reflect")
    w *= interior
    return w / np.sqrt((w[interior] ** 2).mean())


# ----------------------------------------------------------------------------- the vacuum and the polarization

def potential(M, delta=DELTA):
    """sum_k (tr M^k - c_k)^2 for 3 x 3 matrices M[..., 3, 3]."""
    M2 = M @ M
    M3 = M2 @ M
    tr = lambda A: np.trace(A, axis1=-2, axis2=-1)
    return (tr(M) - (1 + delta)) ** 2 + (tr(M2) - (1 + delta ** 2)) ** 2 + (tr(M3) - (1 + delta ** 3)) ** 2


def vacuum_hessian(M0, delta=DELTA, eps=1e-4):
    """The Hessian of the coded potential over the six components at the matrix M0 (3 x 3), by central
    differences of the potential itself, and its eigen-pairs with the metric of the six components:
    H e = lam g e, e^T g e = 1. The masses of the vacuum branches are v lam / (2 L1)."""
    from scipy.linalg import eigh

    E = np.zeros((6, 3, 3))
    for c in range(6):
        E[c, PA[c], PB[c]] = 1.0
        E[c, PB[c], PA[c]] = 1.0
    H = np.zeros((6, 6))
    for a in range(6):
        for b in range(6):
            H[a, b] = (potential(M0 + eps * (E[a] + E[b]), delta) - potential(M0 + eps * (E[a] - E[b]), delta)
                       - potential(M0 - eps * (E[a] - E[b]), delta) + potential(M0 - eps * (E[a] + E[b]), delta)) / (4 * eps * eps)
    lam, vec = eigh(H, np.diag(METRIC))
    return H, lam, vec


def mid_weights(delta=DELTA):
    """The lower eigen-direction of the vacuum Hessian in the plane of E1 = n n^T and E2 = (1 - n n^T)/sqrt 2,
    as its two weights (w1, w2), and the two eigenvalues of that 2 x 2 problem."""
    lam = np.array([1.0, delta, 0.0])
    J = np.array([k * lam ** (k - 1) for k in (1, 2, 3)])
    H = 2 * J.T @ J
    B = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]).T / np.array([1.0, np.sqrt(2.0)])
    e, w = np.linalg.eigh(B.T @ H @ B)
    w = w[:, 0] * np.sign(w[0, 0])
    return w, e


def polarization(m6, delta=DELTA):
    """W(x) = w1 n n^T + w2 (1 - n n^T)/sqrt 2 (3 x 3 per node), n the eigenvector of the largest eigenvalue of
    the background; also the eigenvalues (ascending) and n."""
    w, _ = mid_weights(delta)
    lam, vec = np.linalg.eigh(Q.from6(m6))
    n = vec[..., 2]
    P = np.einsum("...i,...j->...ij", n, n)
    return w[0] * P + w[1] * (np.eye(3) - P) / np.sqrt(2.0), lam, n


# ----------------------------------------------------------------------------- controls

def control_k0(F, model, m6, name):
    f = np.zeros(F.shape6)
    e = F.static(m6, f)
    ref = model.evaluate(m6, 0.0, need_grad=True, need_I=False)
    full = Q.grad_to6(model.gU)
    dU = abs(e["U"] - ref["U"]) / abs(ref["U"])
    dg = float(np.abs(f + full).max() / np.abs(full).max())
    return dict(state=name, U=e["U"], U_ref=ref["U"], rel_U=dU, rel_force=dg, passed=bool(dU < 1e-12 and dg < 1e-12))


def control_k1(F, model, m6, name):
    I = model.evaluate(m6, 0.0, need_grad=False, need_I=True)["I"]
    om = 0.37
    r = om * twist_rate(model, m6)
    p = np.zeros(F.shape6)
    T = F.momentum(m6, r, p)["T"]
    rel = abs(T - 0.5 * I * om * om) / (0.5 * I * om * om)
    return dict(state=name, I=I, Omega=om, T=T, half_I_Omega2=0.5 * I * om * om, rel=rel, passed=bool(rel < 1e-12))


def control_k2(F, m6, rng, n=200):
    N = F.N
    r = np.stack([smooth_noise(rng, (N, N, N), 1.5, F.interior) for _ in range(6)], -1)
    p = np.zeros(F.shape6)
    f = np.zeros(F.shape6)
    F.momentum(m6, r, p, f)
    scratch = np.zeros(F.shape6)
    eps_r, eps_m = 1e-4, 1e-5
    idx = [tuple(rng.integers(1, N - 1, 3)) + (int(rng.integers(0, 6)),) for _ in range(n)]
    errp, errf = 0.0, 0.0
    for ix in idx:
        v0 = r[ix]
        r[ix] = v0 + eps_r
        tp = F.momentum(m6, r, scratch)["T"]
        r[ix] = v0 - eps_r
        tm = F.momentum(m6, r, scratch)["T"]
        r[ix] = v0
        errp = max(errp, abs((tp - tm) / (2 * eps_r) - p[ix]))
        v0 = m6[ix]
        m6[ix] = v0 + eps_m
        tp = F.momentum(m6, r, scratch)["T"]
        m6[ix] = v0 - eps_m
        tm = F.momentum(m6, r, scratch)["T"]
        m6[ix] = v0
        errf = max(errf, abs((tp - tm) / (2 * eps_m) - f[ix]))
    pm, fm = float(np.abs(p).max()), float(np.abs(f).max())
    return dict(entries=n, max_err_dT_dR=errp, largest_dT_dR=pm, rel_dT_dR=errp / pm, max_err_dT_dM=errf,
                largest_dT_dM=fm, rel_dT_dM=errf / fm, passed=bool(errp / pm < 1e-6 and errf / fm < 1e-6))


def control_k3(F, m6, rng, name):
    A = F.a_matrices(m6)[1:-1, 1:-1, 1:-1]
    asym = float(np.abs(A - np.swapaxes(A, -1, -2)).max() / np.abs(A).max())
    ev = np.linalg.eigvalsh(A)
    p = rng.standard_normal(F.shape6) * F.interior[..., None]
    r, f, p2 = np.zeros(F.shape6), np.zeros(F.shape6), np.zeros(F.shape6)
    e = F.rhs(m6, p, r, f)
    F.momentum(m6, r, p2)
    res = float(np.abs(p2 - p)[F.interior].max() / np.abs(p).max())
    return dict(state=name, asymmetry=asym, least_eigenvalue=float(ev.min()), largest_eigenvalue=float(ev.max()),
                vacuum_value=2 * F.L1, not_positive=e["bad"], solve_residual=res,
                passed=bool(asym < 1e-12 and ev.min() > 0 and e["bad"] == 0 and res < 1e-12))


def run_e1(dt, data, work, length=10.0, share=0.01):
    """B0a at rest with the rate Omega [G, M] (zero on the shell), T = share of U; forward, then back."""
    m0, hb, N = load_base("B0a", data)
    F = Field(N, hb)
    model = Q.Model(N, hb, delta=DELTA, L1=L1)
    K = twist_rate(model, m0) * F.interior[..., None]
    p = np.zeros(F.shape6)
    f = np.zeros(F.shape6)
    T1 = F.momentum(m0, K, p)["T"]
    U0 = F.static(m0, f)["U"]
    om = np.sqrt(share * U0 / T1)
    p *= om
    m = m0.copy()
    rk = RK4(F)
    n = int(round(length / dt))
    series, t0 = [], time.time()
    for s in range(n):
        e = rk.step(m, p, dt)
        series.append((s * dt, e["E"], e["T"], e["U"]))
    e = F.rhs(m, p, rk.r, rk.f)
    series.append((n * dt, e["E"], e["T"], e["U"]))
    turn = float(np.abs(m - m0).max())
    p *= -1.0
    bad = 0
    for s in range(n):
        bad += rk.step(m, p, dt)["bad"]
    back = float(np.abs(m - m0).max())
    S = np.array(series)
    out = dict(step=dt, steps=n, length=length, Omega=float(om), U0=U0, T0=float(S[0, 2]), E0=float(S[0, 1]),
               drift=float(np.abs(S[:, 1] - S[0, 1]).max()), drift_over_T0=float(np.abs(S[:, 1] - S[0, 1]).max() / S[0, 2]),
               E_end=float(S[-1, 1]), T_end=float(S[-1, 2]), displacement_at_the_turn=turn, error_after_the_return=back,
               return_over_turn=back / turn, not_positive=bad, seconds=time.time() - t0,
               series=dict(t=S[::10, 0].tolist(), E=S[::10, 1].tolist(), T=S[::10, 2].tolist(), U=S[::10, 3].tolist()))
    work.mkdir(parents=True, exist_ok=True)
    (work / f"e1_step_{dt:g}.json").write_text(json.dumps(out) + "\n")
    return out


def controls(a):
    rng = np.random.default_rng(4700)
    res, t0 = {}, time.time()
    m6, hb, N = load_base("B0a", a.data)
    F = Field(N, hb)
    model = Q.Model(N, hb, delta=DELTA, L1=L1)
    seed = model.seed6()
    res["K0"] = [control_k0(F, model, seed, "seed"), control_k0(F, model, m6, "B0a")]
    res["K1"] = [control_k1(F, model, seed, "seed"), control_k1(F, model, m6, "B0a")]
    res["K2"] = control_k2(F, m6.copy(), rng)
    res["K3"] = [control_k3(F, seed, rng, "seed"), control_k3(F, m6, rng, "B0a")]
    for k in ("K0", "K1", "K2", "K3"):
        print(k, json.dumps(res[k]), flush=True)
    procs = [subprocess.Popen([sys.executable, "-I", str(HERE / "m5_47_evolve.py"), "--e1", str(dt), "--data", str(a.data),
                               "--work", str(a.work)]) for dt in (0.03, 0.015)]
    rc = [p.wait() for p in procs]
    assert rc == [0, 0], rc
    c, f = (json.loads((a.work / f"e1_step_{dt:g}.json").read_text()) for dt in (0.03, 0.015))
    ratio = c["drift"] / f["drift"]
    res["K4"] = dict(drift_over_T0_at_0015=f["drift_over_T0"], drift_over_T0_at_003=c["drift_over_T0"], ratio=ratio,
                     passed=bool(f["drift_over_T0"] < 1e-4 and 12 <= ratio <= 48))
    gain = c["error_after_the_return"] / f["error_after_the_return"]
    res["K5"] = dict(turn_at_0015=f["displacement_at_the_turn"], return_at_0015=f["error_after_the_return"],
                     return_over_turn_at_0015=f["return_over_turn"], return_at_003=c["error_after_the_return"], gain=gain,
                     passed=bool(f["return_over_turn"] <= 0.01 and gain >= 8))
    res["E1"] = {"0.03": c, "0.015": f}
    for k in ("K4", "K5"):
        print(k, json.dumps(res[k]), flush=True)

    def ok(v):
        return all(x["passed"] for x in v) if isinstance(v, list) else v["passed"]

    res["passed"] = {k: ok(res[k]) for k in ("K0", "K1", "K2", "K3", "K4", "K5")}
    res["all_passed"] = all(res["passed"].values())
    res["seconds"] = time.time() - t0
    (a.data / "m5_47_controls_stepper.json").write_text(json.dumps(res, indent=1) + "\n")
    print("passed:", res["passed"], "seconds", round(res["seconds"], 1))
    sys.exit(0 if res["all_passed"] else 1)


def timing(a):
    for name, (hb, N) in BASE.items():
        m, _, _ = load_base(name, a.data)
        F = Field(N, hb)
        rk = RK4(F)
        p = np.zeros(F.shape6)
        rk.step(m, p, 0.03)
        t = time.time()
        for _ in range(3):
            rk.step(m, p, 0.03)
        print(name, N, "seconds per step", round((time.time() - t) / 3, 3), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--e1", type=float)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    ap.add_argument("--work", type=Path, default=HERE.parent / "checkpoints" / "m5_47")
    a = ap.parse_args()
    if a.controls:
        controls(a)
    elif a.e1:
        run_e1(a.e1, a.data, a.work)
    elif a.time:
        timing(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
