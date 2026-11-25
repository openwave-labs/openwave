"""M5.42: the founding quartic of arXiv:2108.07896 (eq. 42) with the twist clock at fixed J.

The field is a real symmetric 3 x 3 matrix M(x) on an N^3 grid of spacing h (six components per node).

    U   = h^3 sum_n [ L1 sum_i |d_i M|^2 + kappa sum_{i<j} |[d_i M, d_j M]|^2 + v sum_k (tr M^k - c_k)^2 ]
    I   = 2 h^3 sum_n [ L1 |K|^2 + kappa sum_i |[K, d_i M]|^2 ],    K = [G, M],    G_ab = -eps_abc rhat_c
    E_J = U + J^2 / (2 I),        Omega = J / I,        2 J Omega / E = 4 E_rot / (U + E_rot),  E_rot = J^2 / 2I

with c_k = 1 + delta^k (the vacuum eigenvalues are 1, delta, 0) and J_e = 8 pi (1 - delta)^2 / alpha at kappa = 1.

The stencil. Every term is written at a node n with one-sided differences d_i^s M = (M[n + s e_i] - M[n]) / h,
s = +1 or -1 (the terms are quadratic in each difference, so its sign does not enter). An orientation is a
choice of the three signs; it contributes at the nodes where its three neighbors exist. A stencil is a set of
weights w[sx, sy, sz] over the eight orientations:

    forward          w = 1 on (+, +, +)       the stencil of the author's clock3d.py (cells 0 .. N-2 per axis)
    mirror-averaged  w = 1/8 on all eight     the mean of the forward functional over the eight reflections
                                              of the grid axes (M and G reflected together)

The band term (a change of the model, used only where a run says so):

    U_band = w_band h^3 sum_n wn sum_k [ max(0, lam_k - hi)^2 + max(0, lo - lam_k)^2 ],  lam_k the eigenvalues of M[n]

Gradients are analytic. For symmetric A, B and C = [A, B]:  d|C|^2/dA = 2 [C, B],  d|C|^2/dB = 2 [A, C];
for K = [G, M] and P = dI/dK:  dI/dM = [P, G];  d(tr M^k - c_k)^2 summed = 2 t1 1 + 4 t2 M + 6 t3 M^2.
"""

import numpy as np
from numba import njit

ALPHA = 1.0 / 137.035999
IU0, IU1 = np.triu_indices(3)
IDX = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])
OFF = IU0 != IU1
FORWARD = np.zeros((2, 2, 2))
FORWARD[1, 1, 1] = 1.0
AVERAGED = np.full((2, 2, 2), 0.125)
STENCILS = dict(forward=FORWARD, averaged=AVERAGED)


def j_electron(delta):
    return 8.0 * np.pi * (1.0 - delta) ** 2 / ALPHA


def from6(m6):
    return np.ascontiguousarray(m6[..., IDX])


def to6(M):
    return np.ascontiguousarray(M[..., IU0, IU1])


def grad_to6(g):
    """The gradient with respect to the six components from the gradient with respect to the nine entries."""
    return g[..., IU0, IU1] + np.where(OFF, g[..., IU1, IU0], 0.0)


def vacuum_frame(X, Y, Z, delta, fixed_chart):
    """The author's boundary frame: long axis rhat, minor axis psi = arg(w^4 - 1)/2 in the chart
    w = (x + i y)/(r - z), with four +1/2 punctures at +-x, +-y. With fixed_chart the minor axis on the
    +z axis is the limit of the frame as w goes to infinity (the x direction); as deposited it is y there."""
    r = np.sqrt(X * X + Y * Y + Z * Z) + 1e-12
    n = np.stack([X, Y, Z], -1) / r[..., None]
    w = (X + 1j * Y) / (r - Z + 1e-9)
    psi = 0.5 * np.angle(w ** 4 - 1)
    ex = np.zeros(X.shape + (3,))
    ex[..., 0] = 1.0
    ey = np.zeros(X.shape + (3,))
    ey[..., 1] = 1.0
    k = np.zeros(X.shape + (3,))
    k[..., 2] = 1.0
    m = n - k
    m2 = (m * m).sum(-1, keepdims=True) + 1e-12

    def refl(v):
        return v - 2 * (v * m).sum(-1, keepdims=True) * m / m2

    t1, t2 = refl(ex), refl(ey)
    d2 = np.cos(psi)[..., None] * t1 + np.sin(psi)[..., None] * t2
    d2 = d2 - (d2 * n).sum(-1, keepdims=True) * n
    d2 /= np.linalg.norm(d2, axis=-1, keepdims=True) + 1e-12
    if fixed_chart:
        pole = (X == 0) & (Y == 0) & (Z > 0)
        d2[pole] = np.array([1.0, 0.0, 0.0])
    M = np.einsum("...i,...j->...ij", n, n) + delta * np.einsum("...i,...j->...ij", d2, d2)
    return M, n, r


@njit(cache=True, inline="always")
def _mm(A, B, out):
    for a in range(3):
        for b in range(3):
            out[a, b] = A[a, 0] * B[0, b] + A[a, 1] * B[1, b] + A[a, 2] * B[2, b]


@njit(cache=True, inline="always")
def _comm(A, B, tmp, out):
    """out = A B - B A (tmp is scratch)."""
    _mm(A, B, out)
    _mm(B, A, tmp)
    for a in range(3):
        for b in range(3):
            out[a, b] -= tmp[a, b]


@njit(cache=True)
def kernel(M, G, wts, h, L1, kap, v, c1, c2, c3, need_grad, need_I, band_w, band_lo, band_hi, gU, gI, idn):
    """One pass over the nodes. Returns (U2, U4, UV, UB, I). With need_grad, gU and gI receive dU/dM and
    dI/dM over the nine entries of every node (the band term inside gU). idn receives the inertia density
    per node (I = h^3 sum idn)."""
    N = M.shape[0]
    h3 = h * h * h
    U2 = 0.0
    U4 = 0.0
    UV = 0.0
    UB = 0.0
    I2 = 0.0
    I4 = 0.0
    D = np.zeros((3, 2, 3, 3))
    gDU = np.zeros((3, 2, 3, 3))
    gDI = np.zeros((3, 2, 3, 3))
    K = np.zeros((3, 3))
    PK = np.zeros((3, 3))
    Q = np.zeros((3, 3))
    C = np.zeros((3, 3))
    T = np.zeros((3, 3))
    T2 = np.zeros((3, 3))
    M2 = np.zeros((3, 3))
    Mn = np.zeros((3, 3))
    gMU = np.zeros((3, 3))
    val = np.zeros((3, 2), dtype=np.bool_)
    idxn = np.zeros(3, dtype=np.int64)
    if need_grad:
        gU[:] = 0.0
        gI[:] = 0.0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                idxn[0] = i
                idxn[1] = j
                idxn[2] = k
                for ax in range(3):
                    val[ax, 0] = idxn[ax] > 0
                    val[ax, 1] = idxn[ax] < N - 1
                wn = 0.0
                for a in range(2):
                    if val[0, a]:
                        for b in range(2):
                            if val[1, b]:
                                for c in range(2):
                                    if val[2, c]:
                                        wn += wts[a, b, c]
                for a in range(3):
                    for b in range(3):
                        Mn[a, b] = M[i, j, k, a, b]
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
                                    D[ax, s, a, b] = (M[ii, jj, kk, a, b] - Mn[a, b]) / h
                                    gDU[ax, s, a, b] = 0.0
                                    gDI[ax, s, a, b] = 0.0
                for a in range(3):
                    for b in range(3):
                        gMU[a, b] = 0.0
                        PK[a, b] = 0.0
                dens = 0.0
                Gn = G[i, j, k]
                if need_I:
                    _comm(Gn, Mn, T, K)
                if wn > 0.0:
                    _mm(Mn, Mn, M2)
                    t1 = Mn[0, 0] + Mn[1, 1] + Mn[2, 2] - c1
                    t2 = M2[0, 0] + M2[1, 1] + M2[2, 2] - c2
                    t3 = -c3
                    for a in range(3):
                        for b in range(3):
                            t3 += M2[a, b] * Mn[b, a]
                    UV += wn * (t1 * t1 + t2 * t2 + t3 * t3)
                    if need_grad:
                        for a in range(3):
                            for b in range(3):
                                gMU[a, b] += wn * v * (4.0 * t2 * Mn[a, b] + 6.0 * t3 * M2[a, b])
                            gMU[a, a] += wn * v * 2.0 * t1
                    if band_w > 0.0:
                        # Gershgorin: skip the eigen-decomposition where every disc lies inside the band
                        inside = True
                        for a in range(3):
                            rad = 0.0
                            for b in range(3):
                                if b != a:
                                    rad += abs(Mn[a, b])
                            if Mn[a, a] + rad > band_hi or Mn[a, a] - rad < band_lo:
                                inside = False
                        if not inside:
                            lam, vec = np.linalg.eigh(Mn)
                            for q in range(3):
                                ex = 0.0
                                if lam[q] > band_hi:
                                    ex = lam[q] - band_hi
                                elif lam[q] < band_lo:
                                    ex = lam[q] - band_lo
                                if ex != 0.0:
                                    UB += wn * ex * ex
                                    if need_grad:
                                        for a in range(3):
                                            for b in range(3):
                                                gMU[a, b] += wn * band_w * 2.0 * ex * vec[a, q] * vec[b, q]
                    if need_I:
                        s2 = 0.0
                        for a in range(3):
                            for b in range(3):
                                s2 += K[a, b] * K[a, b]
                                PK[a, b] += wn * 4.0 * L1 * K[a, b]
                        I2 += wn * s2
                        dens += 2.0 * L1 * wn * s2
                # the terms with one difference
                for ax in range(3):
                    o1 = (ax + 1) % 3
                    o2 = (ax + 2) % 3
                    for s in range(2):
                        if not val[ax, s]:
                            continue
                        wd = 0.0
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
                        if wd <= 0.0:
                            continue
                        Dn = D[ax, s]
                        s2 = 0.0
                        for a in range(3):
                            for b in range(3):
                                s2 += Dn[a, b] * Dn[a, b]
                                gDU[ax, s, a, b] += wd * 2.0 * L1 * Dn[a, b]
                        U2 += wd * s2
                        if need_I:
                            _comm(K, Dn, T, Q)
                            s2 = 0.0
                            for a in range(3):
                                for b in range(3):
                                    s2 += Q[a, b] * Q[a, b]
                            I4 += wd * s2
                            dens += 2.0 * kap * wd * s2
                            if need_grad:
                                _comm(Q, Dn, T, T2)
                                for a in range(3):
                                    for b in range(3):
                                        PK[a, b] += wd * 4.0 * kap * T2[a, b]
                                _comm(K, Q, T, T2)
                                for a in range(3):
                                    for b in range(3):
                                        gDI[ax, s, a, b] += wd * 4.0 * kap * T2[a, b]
                # the terms with two differences
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
                            _comm(A, B, T, C)
                            s2 = 0.0
                            for a in range(3):
                                for b in range(3):
                                    s2 += C[a, b] * C[a, b]
                            U4 += wc * s2
                            if need_grad:
                                _comm(C, B, T, T2)
                                for a in range(3):
                                    for b in range(3):
                                        gDU[ax, s, a, b] += wc * 2.0 * kap * T2[a, b]
                                _comm(A, C, T, T2)
                                for a in range(3):
                                    for b in range(3):
                                        gDU[bx, t, a, b] += wc * 2.0 * kap * T2[a, b]
                idn[i, j, k] = dens
                if need_grad:
                    for a in range(3):
                        for b in range(3):
                            gU[i, j, k, a, b] += h3 * gMU[a, b]
                    if need_I:
                        _comm(PK, Gn, T, T2)
                        for a in range(3):
                            for b in range(3):
                                gI[i, j, k, a, b] += h3 * T2[a, b]
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
                                        gu = h3 * gDU[ax, s, a, b] / h
                                        gU[ii, jj, kk, a, b] += gu
                                        gU[i, j, k, a, b] -= gu
                                        if need_I:
                                            gi = h3 * gDI[ax, s, a, b] / h
                                            gI[ii, jj, kk, a, b] += gi
                                            gI[i, j, k, a, b] -= gi
    return h3 * L1 * U2, h3 * kap * U4, h3 * v * UV, h3 * band_w * UB, 2.0 * h3 * (L1 * I2 + kap * I4)


class Model:
    """A box, its frame and seed, and the energy at fixed J with its gradient over the six components."""

    def __init__(self, N, half_box, delta=0.3, L1=0.3, kappa=1.0, v=1.0, stencil="averaged", fixed_chart=True,
                 free_faces=False, band=None):
        self.N, self.half_box, self.delta, self.L1, self.kappa, self.v = N, half_box, delta, L1, kappa, v
        self.stencil, self.fixed_chart, self.free_faces, self.band = stencil, fixed_chart, free_faces, band
        self.wts = STENCILS[stencil]
        x = np.linspace(-half_box, half_box, N)
        self.x, self.h = x, x[1] - x[0]
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        Mv, n, r = vacuum_frame(X, Y, Z, delta, fixed_chart)
        mbar = (1 + delta) / 3
        s = np.tanh(r / 1.2)[..., None, None]
        self.seed = mbar * np.eye(3) + s * (Mv - mbar * np.eye(3))
        self.frame, self.n, self.r = Mv, n, r
        eps = np.zeros((3, 3, 3))
        eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
        eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
        self.G = np.ascontiguousarray(-np.einsum("ijk,...k->...ij", eps, n))
        self.ck = (1 + delta, 1 + delta ** 2, 1 + delta ** 3)
        self.mask = np.ones(X.shape, bool)
        if not free_faces:
            self.mask[:] = False
            self.mask[1:-1, 1:-1, 1:-1] = True
        self.shape6 = X.shape + (6,)
        self.gU = np.zeros(X.shape + (3, 3))
        self.gI = np.zeros(X.shape + (3, 3))
        self.idn = np.zeros(X.shape)
        self.band_w, self.band_lo, self.band_hi = (band["w"], band["lo"], band["hi"]) if band else (0.0, 0.0, 0.0)
        self.J_e = j_electron(delta)

    def seed6(self):
        return to6(self.seed)

    def evaluate(self, m6, J, need_grad=True, need_I=True):
        """Returns a dict with U2, U4, UV, UB, U, I, E and, with need_grad, the gradient of E over the six
        components (zero on the fixed faces). self.idn holds the inertia density of this call."""
        M = from6(m6.reshape(self.shape6))
        need_I = need_I or J != 0.0
        U2, U4, UV, UB, I = kernel(M, self.G, self.wts, self.h, self.L1, self.kappa, self.v, self.ck[0], self.ck[1],
                                   self.ck[2], need_grad, need_I, self.band_w, self.band_lo, self.band_hi,
                                   self.gU, self.gI, self.idn)
        U = U2 + U4 + UV + UB
        out = dict(U2=U2, U4=U4, UV=UV, UB=UB, U=U, I=I, E=U + (J * J / (2.0 * I) if J != 0.0 else 0.0))
        if need_grad:
            g = self.gU if J == 0.0 else self.gU - (J * J / (2.0 * I * I)) * self.gI
            out["grad"] = grad_to6(g) * self.mask[..., None]
        return out


def clip_eigenvalues(m6, lo, hi):
    """The field with the eigenvalues of every node brought into [lo, hi]."""
    lam, vec = np.linalg.eigh(from6(m6))
    lam = np.clip(lam, lo, hi)
    return to6(np.einsum("...ak,...k,...bk->...ab", vec, lam, vec))
