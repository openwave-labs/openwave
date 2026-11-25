"""M5.51: the author's new action in their discretization (the stencil `ma`), with the link weight at the
midpoint or at the end sites, its analytic gradient, the twist inertia, and the controls K1 to K3.

The field is a real symmetric 3 x 3 matrix M(n) at the nodes of an N^3 grid of spacing h in [-L, L]^3, stored as
six components (xx, xy, xz, yy, yz, zz). With D_a^s M(n) = (M(n + s e_a) - M(n)) / (s h) where that neighbor
exists, and s(M) = sum_k (tr M^k - c_k)^2, c_k = 1 + delta^k:

    U = h^3 sum_n [ kappa sum_{a<b} sum_{s,t} 1/4 |[D_a^s M, D_b^t M]|^2 + v s(M(n)) ]
        + L_V h^3 sum_a sum_{forward links n -> n + e_a} w_a(n) |D_a^+ M(n)|^2

    w_end = (s(M(n)) + s(M(n + e_a))) / 2,        w_mid = s((M(n) + M(n + e_a)) / 2)

    I = 2 h^3 sum_n [ L_V s(M) |K|^2 + kappa sum_a sum_s 1/2 |[K, D_a^s M]|^2 ],   K = [G, M],  G_ab = -eps_abc rhat_c

The seed (faces held at it): M_0 = 1/3 + tanh(r / 1.2) (rhat rhat^T - 1/3), r = |x| + 1e-12.

Gradients (symmetric A, B, C = [A, B]): d|C|^2/dA = 2 [C, B], d|C|^2/dB = 2 [A, C];
s'(M) = 2 t1 1 + 4 t2 M + 6 t3 M^2. One link i -> j, D = (M_j - M_i)/h, Mb = (M_i + M_j)/2:
    mid:  d/dM_i = 1/2 s'(Mb) |D|^2 - (2/h) s(Mb) D,          d/dM_j = 1/2 s'(Mb) |D|^2 + (2/h) s(Mb) D
    end:  d/dM_i = 1/2 s'(M_i) |D|^2 - (1/h)(s_i + s_j) D,     d/dM_j = 1/2 s'(M_j) |D|^2 + (1/h)(s_i + s_j) D

Usage:  python m5_51_action.py --selftest [--out FILE.json]       (the controls K1, K2, K3 of the plan, section 2)
"""

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

HERE = Path(__file__).resolve().parent
CK = HERE.parent / "checkpoints" / "m5_51"
AUTHOR = CK / "author"
IU0, IU1 = np.triu_indices(3)
IDX = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])
OFF = IU0 != IU1

# the four stored arrays of the author: grid tag -> (file, half-box)
STORED = dict(G1=("milestone_action_halfbox6_h0p5.npz", 6.0), G2=("milestone_action_halfbox6_h0p375.npz", 6.0),
              G4=("milestone_action_halfbox9_h0p5.npz", 9.0), G6=("milestone_action_halfbox12_h0p5_final.npz", 12.0))


def from6(m6):
    return np.ascontiguousarray(np.asarray(m6)[..., IDX])


def to6(M):
    return np.ascontiguousarray(M[..., IU0, IU1])


def grad_to6(g):
    """The gradient over the six components from the gradient over the nine entries."""
    return g[..., IU0, IU1] + np.where(OFF, g[..., IU1, IU0], 0.0)


def grid(N, half_box):
    x = np.linspace(-half_box, half_box, N)
    return x, x[1] - x[0]


def seed(N, half_box):
    """M_0 (N, N, N, 3, 3), rhat and r of the grid."""
    x, h = grid(N, half_box)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z) + 1e-12
    n = np.stack([X, Y, Z], -1) / r[..., None]
    nn = np.einsum("...i,...j->...ij", n, n)
    M0 = np.eye(3) / 3.0 + np.tanh(r / 1.2)[..., None, None] * (nn - np.eye(3) / 3.0)
    return M0, n, r


def generator(n):
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
    return np.ascontiguousarray(-np.einsum("ijk,...k->...ij", eps, n))


def sfun(M, c=(1.0, 1.0, 1.0)):
    """s(M) for a stack of matrices (NumPy)."""
    M2 = M @ M
    tr = lambda A: np.trace(A, axis1=-2, axis2=-1)  # noqa: E731
    return (tr(M) - c[0]) ** 2 + (tr(M2) - c[1]) ** 2 + (tr(M2 @ M) - c[2]) ** 2


# ----------------------------------------------------------------------------- numba kernels

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
def _s_and_ds(A, c1, c2, c3, A2, dS, want):
    """s(A) and, with want, s'(A) into dS."""
    _mm(A, A, A2)
    t1 = A[0, 0] + A[1, 1] + A[2, 2] - c1
    t2 = A2[0, 0] + A2[1, 1] + A2[2, 2] - c2
    t3 = -c3
    for a in range(3):
        for b in range(3):
            t3 += A2[a, b] * A[b, a]
    if want:
        for a in range(3):
            for b in range(3):
                dS[a, b] = 4.0 * t2 * A[a, b] + 6.0 * t3 * A2[a, b]
            dS[a, a] += 2.0 * t1
    return t1 * t1 + t2 * t2 + t3 * t3


@njit(cache=True)
def kernel(M, h, kap, v, LV, c1, c2, c3, mid, need_grad, gU, sN, dsN, edens):
    """One pass for the node terms and the forward links. Returns (U4, UV, ULV) with their factors; with
    need_grad, gU receives dU/dM over the nine entries of every node. edens receives each node's share of U
    (its potential, its quartic sign pairs and its forward links), U = sum edens."""
    N = M.shape[0]
    h3 = h * h * h
    U4 = 0.0
    UV = 0.0
    ULV = 0.0
    A2 = np.zeros((3, 3))
    dS = np.zeros((3, 3))
    T = np.zeros((3, 3))
    T2 = np.zeros((3, 3))
    C = np.zeros((3, 3))
    Mb = np.zeros((3, 3))
    D = np.zeros((3, 2, 3, 3))
    gD = np.zeros((3, 2, 3, 3))
    val = np.zeros((3, 2), dtype=np.bool_)
    idxn = np.zeros(3, dtype=np.int64)
    # the potential density and its derivative at every node
    for i in range(N):
        for j in range(N):
            for k in range(N):
                sN[i, j, k] = _s_and_ds(M[i, j, k], c1, c2, c3, A2, dS, need_grad)
                if need_grad:
                    for a in range(3):
                        for b in range(3):
                            dsN[i, j, k, a, b] = dS[a, b]
    if need_grad:
        gU[:] = 0.0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                idxn[0] = i
                idxn[1] = j
                idxn[2] = k
                for ax in range(3):
                    val[ax, 0] = idxn[ax] > 0
                    val[ax, 1] = idxn[ax] < N - 1
                sn = sN[i, j, k]
                UV += sn
                en = v * sn
                if need_grad:
                    for a in range(3):
                        for b in range(3):
                            gU[i, j, k, a, b] += h3 * v * dsN[i, j, k, a, b]
                for ax in range(3):
                    for s in range(2):
                        for a in range(3):
                            for b in range(3):
                                gD[ax, s, a, b] = 0.0
                        if val[ax, s]:
                            ii = i
                            jj = j
                            kk = k
                            st = 2 * s - 1
                            if ax == 0:
                                ii += st
                            elif ax == 1:
                                jj += st
                            else:
                                kk += st
                            for a in range(3):
                                for b in range(3):
                                    D[ax, s, a, b] = (M[ii, jj, kk, a, b] - M[i, j, k, a, b]) / h
                # the quartic: each plane over its four sign pairs, 1/4 each, where both neighbors exist
                for pa in range(3):
                    if pa == 0:
                        a0, b0 = 0, 1
                    elif pa == 1:
                        a0, b0 = 0, 2
                    else:
                        a0, b0 = 1, 2
                    for s in range(2):
                        if not val[a0, s]:
                            continue
                        for t in range(2):
                            if not val[b0, t]:
                                continue
                            _comm(D[a0, s], D[b0, t], T, C)
                            q = 0.0
                            for a in range(3):
                                for b in range(3):
                                    q += C[a, b] * C[a, b]
                            U4 += 0.25 * q
                            en += kap * 0.25 * q
                            if need_grad:
                                _comm(C, D[b0, t], T, T2)
                                for a in range(3):
                                    for b in range(3):
                                        gD[a0, s, a, b] += 0.5 * kap * T2[a, b]
                                _comm(D[a0, s], C, T, T2)
                                for a in range(3):
                                    for b in range(3):
                                        gD[b0, t, a, b] += 0.5 * kap * T2[a, b]
                # the L_V term on the forward links
                if LV != 0.0:
                    for ax in range(3):
                        if not val[ax, 1]:
                            continue
                        ii = i
                        jj = j
                        kk = k
                        if ax == 0:
                            ii += 1
                        elif ax == 1:
                            jj += 1
                        else:
                            kk += 1
                        Dn = D[ax, 1]
                        d2 = 0.0
                        for a in range(3):
                            for b in range(3):
                                d2 += Dn[a, b] * Dn[a, b]
                        if mid:
                            for a in range(3):
                                for b in range(3):
                                    Mb[a, b] = 0.5 * (M[i, j, k, a, b] + M[ii, jj, kk, a, b])
                            w = _s_and_ds(Mb, c1, c2, c3, A2, dS, need_grad)
                        else:
                            w = 0.5 * (sn + sN[ii, jj, kk])
                        ULV += w * d2
                        en += LV * w * d2
                        if need_grad:
                            f = h3 * LV
                            for a in range(3):
                                for b in range(3):
                                    if mid:
                                        gi = 0.5 * dS[a, b] * d2
                                        gj = gi
                                    else:
                                        gi = 0.5 * dsN[i, j, k, a, b] * d2
                                        gj = 0.5 * dsN[ii, jj, kk, a, b] * d2
                                    gU[i, j, k, a, b] += f * gi
                                    gU[ii, jj, kk, a, b] += f * gj
                                    gD[ax, 1, a, b] += LV * 2.0 * w * Dn[a, b]
                edens[i, j, k] = h3 * en
                if need_grad:
                    for ax in range(3):
                        for s in range(2):
                            if val[ax, s]:
                                ii = i
                                jj = j
                                kk = k
                                st = 2 * s - 1
                                if ax == 0:
                                    ii += st
                                elif ax == 1:
                                    jj += st
                                else:
                                    kk += st
                                for a in range(3):
                                    for b in range(3):
                                        # the L_V part of gD already carries its LV factor; the quartic part kap
                                        gu = h3 * gD[ax, s, a, b] / h
                                        gU[ii, jj, kk, a, b] += gu
                                        gU[i, j, k, a, b] -= gu
    return h3 * kap * U4, h3 * v * UV, h3 * LV * ULV


@njit(cache=True)
def inertia_kernel(M, G, h, kap, LV, c1, c2, c3, idn):
    """I and its density per node (I = h^3 sum idn), parts (I_LV, I_4)."""
    N = M.shape[0]
    h3 = h * h * h
    A2 = np.zeros((3, 3))
    dS = np.zeros((3, 3))
    K = np.zeros((3, 3))
    T = np.zeros((3, 3))
    Q = np.zeros((3, 3))
    Dn = np.zeros((3, 3))
    IL = 0.0
    I4 = 0.0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                _comm(G[i, j, k], M[i, j, k], T, K)
                sn = _s_and_ds(M[i, j, k], c1, c2, c3, A2, dS, False)
                k2 = 0.0
                for a in range(3):
                    for b in range(3):
                        k2 += K[a, b] * K[a, b]
                dl = 2.0 * LV * sn * k2
                d4 = 0.0
                for ax in range(3):
                    for s in range(2):
                        ii = i
                        jj = j
                        kk = k
                        st = 2 * s - 1
                        if ax == 0:
                            ii += st
                        elif ax == 1:
                            jj += st
                        else:
                            kk += st
                        if ii < 0 or jj < 0 or kk < 0 or ii >= N or jj >= N or kk >= N:
                            continue
                        for a in range(3):
                            for b in range(3):
                                Dn[a, b] = (M[ii, jj, kk, a, b] - M[i, j, k, a, b]) / h
                        _comm(K, Dn, T, Q)
                        q = 0.0
                        for a in range(3):
                            for b in range(3):
                                q += Q[a, b] * Q[a, b]
                        d4 += 0.5 * q
                d4 *= 2.0 * kap
                idn[i, j, k] = dl + d4
                IL += dl
                I4 += d4
    return h3 * (IL + I4), h3 * IL, h3 * I4


class Model:
    """One grid, one weight. Faces held at M_0 unless free_faces."""

    def __init__(self, N, half_box, weight="mid", delta=0.0, kappa=1.0, v=1.0, LV=1.0, free_faces=False):
        assert weight in ("mid", "end")
        self.N, self.half_box, self.weight, self.delta = N, half_box, weight, delta
        self.kappa, self.v, self.LV, self.free_faces = kappa, v, LV, free_faces
        self.x, self.h = grid(N, half_box)
        self.M0, self.n, self.r = seed(N, half_box)
        self.G = generator(self.n)
        self.ck = (1.0 + delta, 1.0 + delta ** 2, 1.0 + delta ** 3)
        self.mask = np.ones((N, N, N), bool)
        if not free_faces:
            self.mask[:] = False
            self.mask[1:-1, 1:-1, 1:-1] = True
        self.shape6 = (N, N, N, 6)
        self.gU = np.zeros((N, N, N, 3, 3))
        self.sN = np.zeros((N, N, N))
        self.dsN = np.zeros((N, N, N, 3, 3))
        self.idn = np.zeros((N, N, N))
        self.edens = np.zeros((N, N, N))

    def seed6(self):
        return to6(self.M0)

    def hold_faces(self, m6):
        """The field with the faces reset to M_0 (no change with free faces)."""
        m6 = np.asarray(m6).reshape(self.shape6)
        if self.free_faces:
            return m6.copy()
        return np.where(self.mask[..., None], m6, to6(self.M0))

    def evaluate(self, m6, need_grad=True):
        M = from6(np.asarray(m6).reshape(self.shape6))
        U4, UV, ULV = kernel(M, self.h, self.kappa, self.v, self.LV, self.ck[0], self.ck[1], self.ck[2], self.weight == "mid",
                             need_grad, self.gU, self.sN, self.dsN, self.edens)
        out = dict(U4=U4, UV=UV, ULV=ULV, U=U4 + UV + ULV)
        if need_grad:
            out["grad"] = grad_to6(self.gU) * self.mask[..., None]
        return out

    def inertia(self, m6):
        M = from6(np.asarray(m6).reshape(self.shape6))
        I, IL, I4 = inertia_kernel(M, self.G, self.h, self.kappa, self.LV, self.ck[0], self.ck[1], self.ck[2], self.idn)
        return dict(I=I, I_LV=IL, I_4=I4, density=self.idn.copy())


def load_stored(tag):
    """The author's stored array of a grid (faces as stored), its axis and half-box."""
    f, hb = STORED[tag]
    z = np.load(AUTHOR / f, allow_pickle=False)
    return np.asarray(z["m6"], float), np.asarray(z["x"], float), hb


# ----------------------------------------------------------------------------- dense NumPy references for the controls

def _shift(A, a, s):
    out = np.zeros_like(A)
    src = [slice(None)] * A.ndim
    dst = [slice(None)] * A.ndim
    if s == 1:
        src[a], dst[a] = slice(1, None), slice(0, -1)
    else:
        src[a], dst[a] = slice(0, -1), slice(1, None)
    out[tuple(dst)] = A[tuple(src)]
    return out


def _chi(N, a, s):
    i = np.arange(N)
    v = (i <= N - 2) if s == 1 else (i >= 1)
    sh = [1, 1, 1]
    sh[a] = N
    return v.astype(float).reshape(sh)


def _nrm(A):
    return (A * A).sum((-1, -2))


def _cm(A, B):
    return A @ B - B @ A


def inertia_dense(M, G, h, kap=1.0, LV=1.0, c=(1.0, 1.0, 1.0)):
    """I from the equation of section 1, dense NumPy (K1 (d))."""
    N = M.shape[0]
    K = _cm(G, M)
    dens = LV * sfun(M, c) * _nrm(K)
    for a in range(3):
        for s in (1, -1):
            D = (_shift(M, a, s) - M) / (s * h)
            dens = dens + kap * 0.5 * _chi(N, a, s) * _nrm(_cm(K, D))
    return float(2.0 * h ** 3 * dens.sum())


def face_terms(M, h, kap=1.0, v=1.0, c=(1.0, 1.0, 1.0)):
    """The terms in which the stencil `ma` and the stencil `averaged` of m5_42_eq42.py differ, evaluated alone:
    sum over nodes of (w_ma - w_avg) per quartic sign pair and (1 - wn_avg) per potential. With L_V = 0, L1 = 0.
    Returns (U_ma - U_avg from these terms, True when every such term couples face nodes only)."""
    N = M.shape[0]
    chi = {(a, s): _chi(N, a, s) * np.ones((N, N, N)) for a in range(3) for s in (1, -1)}
    nb = {a: chi[(a, 1)] + chi[(a, -1)] for a in range(3)}          # valid signs per axis (1 or 2)
    wn = nb[0] * nb[1] * nb[2] / 8.0
    total = v * ((1.0 - wn) * sfun(M, c)).sum()
    face = np.ones((N, N, N), bool)
    face[1:-1, 1:-1, 1:-1] = False
    only_faces = bool(np.all((np.abs(1.0 - wn) < 1e-15) | face))
    D = {(a, s): (_shift(M, a, s) - M) / (s * h) for a in range(3) for s in (1, -1)}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        cx = 3 - a - b
        for s in (1, -1):
            for t in (1, -1):
                w_ma = 0.25 * chi[(a, s)] * chi[(b, t)]
                w_avg = chi[(a, s)] * chi[(b, t)] * nb[cx] / 8.0
                dw = w_ma - w_avg
                nz = np.abs(dw) > 1e-15
                if nz.any():
                    # the term couples n, n + s e_a, n + t e_b: all three must be face nodes
                    fa = _shift(face.astype(float), a, s) > 0.5
                    fb = _shift(face.astype(float), b, t) > 0.5
                    only_faces &= bool(np.all(~nz | (face & fa & fb)))
                total += kap * (dw * _nrm(_cm(D[(a, s)], D[(b, t)]))).sum()
    return float(h ** 3 * total), only_faces


# ----------------------------------------------------------------------------- the controls K1 to K3

def _difference(mod, zp, zm):
    """U(zp) - U(zm) summed node by node (math.fsum of the differences of the node shares), so that the
    round-off of the two totals does not enter the central difference."""
    import math
    mod.evaluate(zp, False)
    a = mod.edens.copy()
    mod.evaluate(zm, False)
    return math.fsum((a - mod.edens).ravel())


def k1a():
    N, hb = 9, 2.0
    rng = np.random.default_rng(51000)
    x, h = grid(N, hb)
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    M0, _, _ = seed(N, hb)
    S, Kv, ph = [], [], []
    for _ in range(4):
        s6 = rng.standard_normal(6)
        Sm = s6[IDX]
        S.append(Sm / np.sqrt((Sm * Sm).sum()))
    for _ in range(4):
        Kv.append(rng.uniform(-np.pi / 2, np.pi / 2, 3))
    for _ in range(4):
        ph.append(rng.uniform(0.0, 2 * np.pi))
    M = M0.copy()
    for j in range(4):
        M = M + 0.1 * S[j] * np.sin(X @ Kv[j] + ph[j])[..., None, None]
    m6 = to6(M)
    rows = []
    ok_all = True
    for weight in ("end", "mid"):
        for delta in (0.0, 0.05):
            for free in (False, True):
                mod = Model(N, hb, weight=weight, delta=delta, free_faces=free)
                z = m6.reshape(-1).copy()
                g = mod.evaluate(z)["grad"].reshape(-1)
                free_idx = np.flatnonzero(np.broadcast_to(mod.mask[..., None], mod.shape6).reshape(-1))
                pick = rng.choice(free_idx, 200, replace=False)
                dirn = np.zeros_like(z)
                dv = rng.standard_normal(len(free_idx))
                dirn[free_idx] = dv / np.linalg.norm(dv)
                fails = 0
                for p in pick:
                    e = 1e-5
                    zp = z.copy()
                    zp[p] += e
                    zm = z.copy()
                    zm[p] -= e
                    fd = _difference(mod, zp, zm) / (2 * e)
                    ref = g[p] * (zp[p] - zm[p]) / (2 * e)
                    ad = abs(fd - ref)
                    rel = ad / max(abs(fd), 1e-300)
                    if not (rel <= 1e-5 or ad <= 1e-9):
                        fails += 1
                gd = float(g @ dirn)
                errs = []
                for e in (1e-2, 1e-3, 1e-4):
                    zp = z + e * dirn
                    zm = z - e * dirn
                    fd = _difference(mod, zp, zm) / (2 * e)
                    errs.append(abs(fd - float(g @ (zp - zm)) / (2 * e)))
                ratios = [errs[0] / errs[1], errs[1] / errs[2]]
                ok = fails == 0 and all(80.0 <= q <= 120.0 for q in ratios)
                ok_all &= ok
                rows.append(dict(weight=weight, delta=delta, free_faces=free, component_fails=fails, directional_errors=errs, ratios_per_decade=ratios,
                                 directional_derivative=gd, passed=bool(ok)))
    return dict(control="K1 (a) the gradient", passed=bool(ok_all), rows=rows), m6


def k1b(m6):
    sys.path.insert(0, str(HERE))
    import m5_42_eq42 as E
    N, hb = 9, 2.0
    mine = Model(N, hb, weight="end", delta=0.0, LV=0.0)
    ref = E.Model(N, hb, delta=0.0, L1=0.0, kappa=1.0, v=1.0, stencil="averaged", fixed_chart=True, free_faces=False)
    a = mine.evaluate(m6)
    b = ref.evaluate(m6.copy(), 0.0, need_grad=True, need_I=False)
    ga = a["grad"]
    gb = b["grad"]
    gnorm = float(np.abs(ga - gb).max() / np.abs(gb).max())
    ft, only_faces = face_terms(from6(m6), mine.h)
    dU = a["U"] - b["U"]
    erel = abs(dU - ft) / abs(ft)
    ok = gnorm <= 1e-12 and erel <= 1e-12 and only_faces
    return dict(control="K1 (b) against the controlled kernel", passed=bool(ok), grad_norm=gnorm, U_ma=a["U"], U_avg=b["U"], difference=dU, face_terms=ft,
                rel_error_vs_face_terms=erel, differing_terms_couple_face_nodes_only=only_faces)


def k1cd():
    sys.path.insert(0, str(HERE))
    import m5_42_eq42 as E
    rows = []
    ok_all = True
    for tag, (f, hb) in STORED.items():
        m6, x, _ = load_stored(tag)
        N = m6.shape[0]
        mod = Model(N, hb, LV=0.0)
        m6 = mod.hold_faces(m6)
        Ic = mod.inertia(m6)["I"]
        ref = E.Model(N, hb, delta=0.0, L1=0.0, kappa=1.0, v=1.0, stencil="averaged")
        Ir = ref.evaluate(m6.copy(), 0.0, need_grad=False, need_I=True)["I"]
        rc = abs(Ic / Ir - 1)
        mod1 = Model(N, hb, LV=1.0)
        Id = mod1.inertia(m6)["I"]
        Idd = inertia_dense(from6(m6), mod1.G, mod1.h)
        rd = abs(Id / Idd - 1)
        ok = rc <= 1e-12 and rd <= 1e-12
        ok_all &= ok
        rows.append(dict(grid=tag, I_quartic=Ic, I_m5_42=Ir, rel_c=rc, I_full=Id, I_dense=Idd, rel_d=rd, passed=bool(ok)))
    return dict(control="K1 (c) and (d) the inertia", passed=bool(ok_all), rows=rows)


AUDIT = {"G1": (20.280477484861102, 20.49765488374363), "G2": (20.423144527162876, 20.532569748664272),
         "G4": (21.28749129056605, 21.50214627273697)}
LAUE_PARTS = {"G1": ("2.4592", "13.4883", "4.33299"), "G4": ("2.4699", "14.374", "4.44365")}


def _printed_ok(val, txt):
    d = len(txt.split(".")[1]) if "." in txt else 0
    return abs(val - float(txt)) <= 0.5 * 10 ** (-d) + 1e-15


def k2():
    rows = []
    ok_all = True
    for tag, (f, hb) in STORED.items():
        m6, x, _ = load_stored(tag)
        N = m6.shape[0]
        me = Model(N, hb, weight="end")
        assert np.allclose(me.x, x)
        m6h = me.hold_faces(m6)
        face_diff = float(np.abs(m6 - m6h).max())
        e = me.evaluate(m6h, False)
        mm = Model(N, hb, weight="mid").evaluate(m6h, False)
        row = dict(grid=tag, stored_face_minus_seed=face_diff, U_end=e["U"], U_mid=mm["U"], U4=e["U4"], UV=e["UV"], ULV_end=e["ULV"], ULV_mid=mm["ULV"],
                   rise_pct=100 * (mm["U"] / e["U"] - 1))
        ok = True
        if tag in AUDIT:
            re = abs(e["U"] / AUDIT[tag][0] - 1)
            rm = abs(mm["U"] / AUDIT[tag][1] - 1)
            row.update(rel_end=re, rel_mid=rm)
            ok &= re <= 1e-9 and rm <= 1e-9
        else:
            row.update(abs_vs_posted=abs(e["U"] - 21.805))
            ok &= abs(e["U"] - 21.805) <= 0.0005
        if tag in LAUE_PARTS:
            p = LAUE_PARTS[tag]
            pk = [_printed_ok(e["ULV"], p[0]), _printed_ok(e["U4"], p[1]), _printed_ok(e["UV"], p[2])]
            row.update(parts_printed=list(p), parts_match=pk)
            ok &= all(pk)
        row["passed"] = bool(ok)
        ok_all &= ok
        rows.append(row)
    return dict(control="K2 the author's states at fixed field", passed=bool(ok_all), rows=rows)


def k3():
    rows = []
    ok_all = True
    h = 0.5
    for d in (0.0, 0.05):
        if d == 0.0:
            A, B = np.diag([1.0, 0.0, 0.0]), np.diag([0.0, 1.0, 0.0])
        else:
            A, B = np.diag([1.0, d, 0.0]), np.diag([1.0, 0.0, d])
        for N in (9, 10):
            hb = (N - 1) * h / 2
            for ax in range(3):
                i = np.arange(N).reshape([N if q == ax else 1 for q in range(3)])
                M = np.where((i < N // 2)[..., None, None], A, B) * np.ones((N, N, N, 1, 1))
                m6 = to6(M)
                e = Model(N, hb, weight="end", delta=d, free_faces=True).evaluate(m6, False)
                mm = Model(N, hb, weight="mid", delta=d, free_faces=True).evaluate(m6, False)
                cf = 13 / 8 * N * N * h if d == 0.0 else N * N * h * (d ** 6 / 2 + 9 * d ** 8 / 8)
                rel = abs(mm["U"] / cf - 1)
                tol = 1e-12 if d == 0.0 else 1e-10
                ok = abs(e["U4"]) <= 1e-14 and abs(e["UV"]) <= 1e-14 and abs(e["U"]) <= 1e-14 and abs(mm["U4"]) <= 1e-14 and abs(mm["UV"]) <= 1e-14 and rel <= tol
                ok_all &= ok
                rows.append(dict(delta=d, N=N, normal_axis=ax, U4=e["U4"], UV=e["UV"], U_end=e["U"], U_mid=mm["U"], closed_form=cf, rel=rel, passed=bool(ok)))
    return dict(control="K3 the loophole in closed form", passed=bool(ok_all), rows=rows)


def selftest(out):
    res = []
    r, m6 = k1a()
    res.append(r)
    print(r["control"], "PASS" if r["passed"] else "FAIL", [(q["weight"], q["delta"], q["free_faces"], q["component_fails"], [round(t, 2) for t in q["ratios_per_decade"]]) for q in r["rows"]], flush=True)
    r = k1b(m6)
    res.append(r)
    print(r["control"], "PASS" if r["passed"] else "FAIL", f"grad norm {r['grad_norm']:.3e} energy rel {r['rel_error_vs_face_terms']:.3e} faces only {r['differing_terms_couple_face_nodes_only']}", flush=True)
    r = k1cd()
    res.append(r)
    print(r["control"], "PASS" if r["passed"] else "FAIL", [(q["grid"], f"{q['rel_c']:.2e}", f"{q['rel_d']:.2e}") for q in r["rows"]], flush=True)
    r = k2()
    res.append(r)
    print(r["control"], "PASS" if r["passed"] else "FAIL", flush=True)
    for q in r["rows"]:
        print("   ", json.dumps(q), flush=True)
    r = k3()
    res.append(r)
    print(r["control"], "PASS" if r["passed"] else "FAIL", f"worst rel {max(q['rel'] for q in r['rows']):.2e}", flush=True)
    passed = all(q["passed"] for q in res)
    if out:
        Path(out).write_text(json.dumps(dict(passed=passed, controls=res), indent=1, default=float) + "\n")
    print("CONTROLS K1 to K3:", "ALL PASS" if passed else "FAIL", flush=True)
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.selftest:
        ap.print_help()
        return
    sys.exit(0 if selftest(a.out) else 1)


if __name__ == "__main__":
    main()
