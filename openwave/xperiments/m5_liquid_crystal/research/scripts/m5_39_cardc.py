"""M5.39 R1: an independent evaluator of the Card C lattice energy and its gradient.

Card C is the author's frozen hadronic card (capstone v3.4, frozen_packets/, the code digests of the
Card C payload manifest). This module is written from the card's definition, not from the archived
routines: every term is an explicit loop over nodes, edges or cells, and every gradient is the adjoint
of that loop. It is compared with the archived evaluation by m5_39_r1_gates.py (1e-10 per term and on
the gradient) before anything is relaxed with it.

Unknowns, 13 reals per interior node: p in R^4 (the winding field is q = p / |p|) and the 3x3 factor
f. Derived fields per node:

    tau(q)   the 4x3 triad whose columns are q times the three imaginary units (right multiplication)
    a        = tau f                                  (4x3)
    N        = - q q^T + lam a a^T / G_TIME           (4x4; lam = 1 on the card, 0 in the ablation)
    s        = f^T f                                  (3x3, the record)
    b        = delta (1 - q0) / 2,   c = (1 - q0) / 2
    x1       = s^3 - (1 + b) s^2 + b s,   x2 = s x1
    u        = (s - s^2) / (delta (1 - delta))

The seven energy terms, in the order of the archived `evaluate` (h the spacing; w_e the edge weight
of the two-sided difference rule: 2 for an edge whose two transverse indices are interior, 1 on a
face, 0 or 1 on the box edges, see `_edge_weight`):

    0 tensor quadratic      (h / 8)  sum_e w_e |N_hi - N_lo|^2
    1 quaternion quartic    (1 / 192 h) sum over cells, the 24 simplices of the four Kuhn
                            triangulations (one per body diagonal) and the four points of the
                            degree-2 rule: the second elementary symmetric function of the Gram
                            matrix of the tangent derivative of the normalized affine interpolant
    2 record quartic        (1 / 16 h) sum over nodes, forward and backward stencils, i < j:
                            |[D_i s, D_j s]|^2 with D the plain differences
    3 composite quadratics  (kx h / 4) sum_e w_e (|x1_hi - x1_lo|^2 + |x2_hi - x2_lo|^2)
    4 record potential      (h^3 / 4) sum_nodes ((tr s - (1 + delta) c)^2
                            + |s^3 - (1 + delta) c s^2 + delta c^2 s|^2)
    5 pion mass             BETA2 h^3 sum_nodes (1 - q0)
    6 orientation           (ko h / 2) sum over all edges |[u_lo, u_hi]|^2

The boundary nodes are held (q = (1, 0, 0, 0), f = 0 on Card C); the gradient is returned on the
interior nodes only, in the order of the unknowns.
"""

import itertools
import math

import numba as nb
import numpy as np
from scipy.ndimage import map_coordinates

G_TIME = 230.0
BETA2 = (138.0 / (4.259 * 92.4)) ** 2
RECORD4 = 0.125
POTENTIAL = 0.25
TENSOR = 0.25
QUARTIC = 0.5
SMALL = (5.0 - math.sqrt(5.0)) / 20.0
LARGE = 1.0 - 3.0 * SMALL
TERM_NAMES = ["tensor_quadratic", "quaternion_quartic", "record_quartic", "composite_quadratics",
              "record_potential", "pion_mass", "orientation"]
ISO_VALUE = 0.396  # the isotropic minimum of the record potential that the text names


def _simplex_offsets():
    """Vertex paths of the 24 simplices of a cell: four Kuhn triangulations, one per body diagonal.

    Triangulation 0 walks from the corner (0,0,0) to (1,1,1) one axis at a time, in each of the six
    axis orders. Triangulation r = 1, 2, 3 is its mirror image in the axis r - 1 (the walk starts at
    the corner with that coordinate at 1 and steps down along it).
    """
    off = np.zeros((4, 6, 4, 3), dtype=np.int64)
    for r in range(4):
        for o, perm in enumerate(itertools.permutations(range(3))):
            cur = [0, 0, 0]
            path = [tuple(cur)]
            for axis in perm:
                cur[axis] += 1
                path.append(tuple(cur))
            for m in range(4):
                v = list(path[m])
                if r > 0:
                    v[r - 1] = 1 - v[r - 1]
                off[r, o, m] = v
    return off


OFFSETS = _simplex_offsets()


# ----------------------------------------------------------------------------------------------
# term 1: the quaternion quartic
# ----------------------------------------------------------------------------------------------
@nb.njit(cache=True)
def _quartic_slab(q, g, i, off):
    """Cells of the slab i: energy (unscaled) returned, adjoint added to g on the planes i, i + 1."""
    n = q.shape[0]
    e_slab = 0.0
    v = np.zeros((4, 4))
    gv = np.zeros((4, 4))
    D = np.zeros((3, 4))
    gD = np.zeros((3, 4))
    S = np.zeros(4)
    Spp = np.zeros(4)
    qh = np.zeros(4)
    y = np.zeros(4)
    c = np.zeros(3)
    gc = np.zeros(3)
    M = np.zeros((3, 3))
    A = np.zeros((3, 3))
    W = np.zeros((3, 3))
    Wt = np.zeros((3, 3))
    for j in range(n - 1):
        for k in range(n - 1):
            for r in range(4):
                for o in range(6):
                    for m in range(4):
                        ii = i + off[r, o, m, 0]
                        jj = j + off[r, o, m, 1]
                        kk = k + off[r, o, m, 2]
                        for t in range(4):
                            v[m, t] = q[ii, jj, kk, t]
                            gv[m, t] = 0.0
                    for t in range(4):
                        S[t] = v[0, t] + v[1, t] + v[2, t] + v[3, t]
                        Spp[t] = 0.0
                    for m in range(3):
                        for t in range(4):
                            D[m, t] = v[m + 1, t] - v[m, t]
                            gD[m, t] = 0.0
                    for m in range(3):
                        for mm in range(m, 3):
                            M[m, mm] = D[m, 0] * D[mm, 0] + D[m, 1] * D[mm, 1] + D[m, 2] * D[mm, 2] + D[m, 3] * D[mm, 3]
                            M[mm, m] = M[m, mm]
                            Wt[m, mm] = 0.0
                            Wt[mm, m] = 0.0
                    for iv in range(4):
                        mag2 = 0.0
                        for t in range(4):
                            qh[t] = SMALL * S[t] + (LARGE - SMALL) * v[iv, t]
                            mag2 += qh[t] * qh[t]
                        inv = 1.0 / math.sqrt(mag2)
                        for t in range(4):
                            qh[t] *= inv
                        for m in range(3):
                            c[m] = D[m, 0] * qh[0] + D[m, 1] * qh[1] + D[m, 2] * qh[2] + D[m, 3] * qh[3]
                        for m in range(3):
                            for mm in range(3):
                                A[m, mm] = M[m, mm] - c[m] * c[mm]
                        im4 = inv * inv * inv * inv
                        e = (A[0, 0] * A[1, 1] + A[0, 0] * A[2, 2] + A[1, 1] * A[2, 2]
                             - A[0, 1] * A[0, 1] - A[0, 2] * A[0, 2] - A[1, 2] * A[1, 2]) * im4
                        e_slab += e
                        trA = A[0, 0] + A[1, 1] + A[2, 2]
                        for m in range(3):
                            for mm in range(3):
                                W[m, mm] = -A[m, mm] * im4
                            W[m, m] += trA * im4
                        for m in range(3):
                            gc[m] = -2.0 * (W[m, 0] * c[0] + W[m, 1] * c[1] + W[m, 2] * c[2])
                            for mm in range(3):
                                Wt[m, mm] += W[m, mm]
                        qy = 0.0
                        for t in range(4):
                            y[t] = gc[0] * D[0, t] + gc[1] * D[1, t] + gc[2] * D[2, t]
                            qy += qh[t] * y[t]
                            for m in range(3):
                                gD[m, t] += gc[m] * qh[t]
                        gmag = -4.0 * e * inv
                        for t in range(4):
                            gpp = (y[t] - qh[t] * qy) * inv + gmag * qh[t]
                            Spp[t] += gpp
                            gv[iv, t] += (LARGE - SMALL) * gpp
                    for m in range(3):
                        for t in range(4):
                            gD[m, t] += 2.0 * (Wt[m, 0] * D[0, t] + Wt[m, 1] * D[1, t] + Wt[m, 2] * D[2, t])
                    for m in range(3):
                        for t in range(4):
                            gv[m + 1, t] += gD[m, t]
                            gv[m, t] -= gD[m, t]
                    for m in range(4):
                        ii = i + off[r, o, m, 0]
                        jj = j + off[r, o, m, 1]
                        kk = k + off[r, o, m, 2]
                        for t in range(4):
                            g[ii, jj, kk, t] += gv[m, t] + SMALL * Spp[t]
    return e_slab


@nb.njit(parallel=True, cache=True)
def _quartic(q, off, h):
    n = q.shape[0]
    g = np.zeros_like(q)
    es = np.zeros(n - 1)
    for parity in range(2):
        count = (n - 1 - parity + 1) // 2
        for t in nb.prange(count):
            i = parity + 2 * t
            es[i] = _quartic_slab(q, g, i, off)
    scale = QUARTIC / (96.0 * h)  # (coefficient h^3 / 24) / 4 reflections / h^4
    return scale * es.sum(), scale * g


@nb.njit(parallel=True, cache=True)
def _hemisphere(q, off, nr):
    """Smallest v . (sum of the simplex's vertices): triangulation 0 alone, and the first nr."""
    n = q.shape[0]
    m0 = np.full(n - 1, 4.0)
    m4 = np.full(n - 1, 4.0)
    for i in nb.prange(n - 1):
        a0 = 4.0
        a4 = 4.0
        for j in range(n - 1):
            for k in range(n - 1):
                for r in range(nr):
                    for o in range(6):
                        s0 = 0.0
                        s1 = 0.0
                        s2 = 0.0
                        s3 = 0.0
                        for m in range(4):
                            ii = i + off[r, o, m, 0]
                            jj = j + off[r, o, m, 1]
                            kk = k + off[r, o, m, 2]
                            s0 += q[ii, jj, kk, 0]
                            s1 += q[ii, jj, kk, 1]
                            s2 += q[ii, jj, kk, 2]
                            s3 += q[ii, jj, kk, 3]
                        for m in range(4):
                            ii = i + off[r, o, m, 0]
                            jj = j + off[r, o, m, 1]
                            kk = k + off[r, o, m, 2]
                            d = q[ii, jj, kk, 0] * s0 + q[ii, jj, kk, 1] * s1 + q[ii, jj, kk, 2] * s2 + q[ii, jj, kk, 3] * s3
                            if d < a4:
                                a4 = d
                            if r == 0 and d < a0:
                                a0 = d
        m0[i] = a0
        m4[i] = a4
    return m0.min(), m4.min()


# ----------------------------------------------------------------------------------------------
# node fields
# ----------------------------------------------------------------------------------------------
@nb.njit(cache=True)
def _mm(a, b, out):
    for i in range(3):
        for j in range(3):
            out[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j]


@nb.njit(cache=True)
def _triad(q, tau):
    tau[0, 0] = -q[1]
    tau[1, 0] = q[0]
    tau[2, 0] = q[3]
    tau[3, 0] = -q[2]
    tau[0, 1] = -q[2]
    tau[1, 1] = -q[3]
    tau[2, 1] = q[0]
    tau[3, 1] = q[1]
    tau[0, 2] = -q[3]
    tau[1, 2] = q[2]
    tau[2, 2] = -q[1]
    tau[3, 2] = q[0]


@nb.njit(parallel=True, cache=True)
def _node_fields(state, delta, lam, q, a, N, s, x1, x2, u):
    n = state.shape[0]
    scale = delta * (1.0 - delta)
    for i in nb.prange(n):
        tau = np.zeros((4, 3))
        f = np.zeros((3, 3))
        s2 = np.zeros((3, 3))
        s3 = np.zeros((3, 3))
        for j in range(n):
            for k in range(n):
                p0 = state[i, j, k, 0]
                p1 = state[i, j, k, 1]
                p2 = state[i, j, k, 2]
                p3 = state[i, j, k, 3]
                inv = 1.0 / math.sqrt(p0 * p0 + p1 * p1 + p2 * p2 + p3 * p3)
                qq = q[i, j, k]
                qq[0] = p0 * inv
                qq[1] = p1 * inv
                qq[2] = p2 * inv
                qq[3] = p3 * inv
                for r in range(3):
                    for c in range(3):
                        f[r, c] = state[i, j, k, 4 + 3 * r + c]
                _triad(qq, tau)
                aa = a[i, j, k]
                for r in range(4):
                    for c in range(3):
                        aa[r, c] = tau[r, 0] * f[0, c] + tau[r, 1] * f[1, c] + tau[r, 2] * f[2, c]
                NN = N[i, j, k]
                for r in range(4):
                    for c in range(4):
                        NN[r, c] = -qq[r] * qq[c] + lam * (aa[r, 0] * aa[c, 0] + aa[r, 1] * aa[c, 1] + aa[r, 2] * aa[c, 2]) / G_TIME
                ss = s[i, j, k]
                for r in range(3):
                    for c in range(3):
                        ss[r, c] = f[0, r] * f[0, c] + f[1, r] * f[1, c] + f[2, r] * f[2, c]
                _mm(ss, ss, s2)
                _mm(s2, ss, s3)
                b = 0.5 * delta * (1.0 - qq[0])
                xx1 = x1[i, j, k]
                uu = u[i, j, k]
                for r in range(3):
                    for c in range(3):
                        xx1[r, c] = s3[r, c] - (1.0 + b) * s2[r, c] + b * ss[r, c]
                        uu[r, c] = (ss[r, c] - s2[r, c]) / scale
                _mm(ss, xx1, x2[i, j, k])


# ----------------------------------------------------------------------------------------------
# terms 0, 3, 6: the edge terms (gather form: every node reads its six neighbors)
# ----------------------------------------------------------------------------------------------
@nb.njit(cache=True)
def _edge_weight(n, t1, t2):
    """Weight of an edge in the two-sided difference rule, from its two transverse indices."""
    w = 0.0
    if t1 <= n - 2 and t2 <= n - 2:
        w += 1.0
    if t1 >= 1 and t2 >= 1:
        w += 1.0
    return w


@nb.njit(parallel=True, cache=True)
def _edge_terms(N, x1, x2, u, dN, d1, d2, dU):
    """Unscaled energies (tensor, x1, x2, orientation) and unscaled adjoints of the four fields."""
    n = N.shape[0]
    eN = np.zeros(n)
    e1 = np.zeros(n)
    e2 = np.zeros(n)
    eO = np.zeros(n)
    for i in nb.prange(n):
        idx = np.zeros(3, dtype=np.int64)
        nbr = np.zeros(3, dtype=np.int64)
        cm = np.zeros((3, 3))
        t = np.zeros((3, 3))
        aN = 0.0
        a1 = 0.0
        a2 = 0.0
        aO = 0.0
        for j in range(n):
            for k in range(n):
                idx[0] = i
                idx[1] = j
                idx[2] = k
                gN = dN[i, j, k]
                g1 = d1[i, j, k]
                g2 = d2[i, j, k]
                gU = dU[i, j, k]
                gN[:, :] = 0.0
                g1[:, :] = 0.0
                g2[:, :] = 0.0
                gU[:, :] = 0.0
                Nx = N[i, j, k]
                X1 = x1[i, j, k]
                X2 = x2[i, j, k]
                ux = u[i, j, k]
                for axis in range(3):
                    w = _edge_weight(n, idx[(axis + 1) % 3], idx[(axis + 2) % 3])
                    for sign in range(2):
                        nbr[0] = i
                        nbr[1] = j
                        nbr[2] = k
                        nbr[axis] += 1 - 2 * sign
                        if nbr[axis] < 0 or nbr[axis] > n - 1:
                            continue
                        Ny = N[nbr[0], nbr[1], nbr[2]]
                        Y1 = x1[nbr[0], nbr[1], nbr[2]]
                        Y2 = x2[nbr[0], nbr[1], nbr[2]]
                        uy = u[nbr[0], nbr[1], nbr[2]]
                        for r in range(4):
                            for c in range(4):
                                d = Nx[r, c] - Ny[r, c]
                                gN[r, c] += w * d
                                if sign == 0:
                                    aN += w * d * d
                        for r in range(3):
                            for c in range(3):
                                d = X1[r, c] - Y1[r, c]
                                g1[r, c] += w * d
                                if sign == 0:
                                    a1 += w * d * d
                                d = X2[r, c] - Y2[r, c]
                                g2[r, c] += w * d
                                if sign == 0:
                                    a2 += w * d * d
                        # cm = [ux, uy]; the adjoint of |cm|^2 with respect to ux is 2 [cm, uy]
                        _mm(ux, uy, cm)
                        _mm(uy, ux, t)
                        for r in range(3):
                            for c in range(3):
                                cm[r, c] -= t[r, c]
                                if sign == 0:
                                    aO += cm[r, c] * cm[r, c]
                        _mm(cm, uy, t)
                        for r in range(3):
                            for c in range(3):
                                gU[r, c] += t[r, c]
                        _mm(uy, cm, t)
                        for r in range(3):
                            for c in range(3):
                                gU[r, c] -= t[r, c]
        eN[i] = aN
        e1[i] = a1
        e2[i] = a2
        eO[i] = aO
    return eN.sum(), e1.sum(), e2.sum(), eO.sum()


# ----------------------------------------------------------------------------------------------
# term 2: the record quartic
# ----------------------------------------------------------------------------------------------
@nb.njit(cache=True)
def _record_stencil(s, dS, i, j, k, step):
    """One three-leg stencil at the node (i, j, k), legs at +step or -step: energy, adjoint to dS."""
    A = np.zeros((3, 3, 3))
    gA = np.zeros((3, 3, 3))
    C = np.zeros((3, 3))
    t = np.zeros((3, 3))
    base = s[i, j, k]
    for axis in range(3):
        ii = i + (step if axis == 0 else 0)
        jj = j + (step if axis == 1 else 0)
        kk = k + (step if axis == 2 else 0)
        for r in range(3):
            for c in range(3):
                A[axis, r, c] = s[ii, jj, kk, r, c] - base[r, c]
    e = 0.0
    for a1 in range(3):
        for a2 in range(a1 + 1, 3):
            _mm(A[a1], A[a2], C)
            _mm(A[a2], A[a1], t)
            for r in range(3):
                for c in range(3):
                    C[r, c] -= t[r, c]
                    e += C[r, c] * C[r, c]
            # adjoint of |C|^2: 2 [C, A2] on A1, 2 [A1, C] on A2
            _mm(C, A[a2], t)
            for r in range(3):
                for c in range(3):
                    gA[a1, r, c] += 2.0 * t[r, c]
            _mm(A[a2], C, t)
            for r in range(3):
                for c in range(3):
                    gA[a1, r, c] -= 2.0 * t[r, c]
            _mm(A[a1], C, t)
            for r in range(3):
                for c in range(3):
                    gA[a2, r, c] += 2.0 * t[r, c]
            _mm(C, A[a1], t)
            for r in range(3):
                for c in range(3):
                    gA[a2, r, c] -= 2.0 * t[r, c]
    for axis in range(3):
        ii = i + (step if axis == 0 else 0)
        jj = j + (step if axis == 1 else 0)
        kk = k + (step if axis == 2 else 0)
        for r in range(3):
            for c in range(3):
                dS[ii, jj, kk, r, c] += gA[axis, r, c]
                dS[i, j, k, r, c] -= gA[axis, r, c]
    return e


@nb.njit(parallel=True, cache=True)
def _record_quartic(s, dS):
    """Unscaled energy; the unscaled adjoint is added to dS. Slab i touches the planes i and i + 1."""
    n = s.shape[0]
    es = np.zeros(n - 1)
    for parity in range(2):
        count = (n - 1 - parity + 1) // 2
        for t in nb.prange(count):
            i = parity + 2 * t
            e = 0.0
            for j in range(n - 1):
                for k in range(n - 1):
                    e += _record_stencil(s, dS, i, j, k, 1)
                    e += _record_stencil(s, dS, i + 1, j + 1, k + 1, -1)
            es[i] = e
    return es.sum()


# ----------------------------------------------------------------------------------------------
# terms 4, 5 and the chain back to (p, f)
# ----------------------------------------------------------------------------------------------
@nb.njit(cache=True)
def _poly_adjoint(D, s, s2, c2, c1, out, t, t2):
    """Adjoint through X = s^3 + c2 s^2 + c1 s of a matrix D: added to out (s symmetric)."""
    _mm(D, s2, t)
    for r in range(3):
        for c in range(3):
            out[r, c] += t[r, c]
    _mm(s, D, t)
    _mm(t, s, t2)
    for r in range(3):
        for c in range(3):
            out[r, c] += t2[r, c] + c2 * t[r, c]
    _mm(s2, D, t)
    for r in range(3):
        for c in range(3):
            out[r, c] += t[r, c]
    _mm(D, s, t)
    for r in range(3):
        for c in range(3):
            out[r, c] += c2 * t[r, c] + c1 * D[r, c]


@nb.njit(parallel=True, cache=True)
def _chain(state, q, a, s, x1, dN, d1, d2, dU, dS, gq, delta, lam, cN, cX, cR, cO, h, grad):
    """Potential and mass energies (unscaled), and the gradient on (p, f) for every node."""
    n = state.shape[0]
    scale = delta * (1.0 - delta)
    coeff = POTENTIAL * h ** 3
    ev = np.zeros(n)
    em = np.zeros(n)
    for i in nb.prange(n):
        f = np.zeros((3, 3))
        tau = np.zeros((4, 3))
        s2 = np.zeros((3, 3))
        s3 = np.zeros((3, 3))
        rec = np.zeros((3, 3))
        R = np.zeros((3, 3))
        D1 = np.zeros((3, 3))
        D2 = np.zeros((3, 3))
        T = np.zeros((3, 3))
        t = np.zeros((3, 3))
        t2 = np.zeros((3, 3))
        dA = np.zeros((4, 3))
        B = np.zeros((4, 3))
        dq = np.zeros(4)
        av = 0.0
        am = 0.0
        for j in range(n):
            for k in range(n):
                qq = q[i, j, k]
                ss = s[i, j, k]
                for r in range(3):
                    for c in range(3):
                        f[r, c] = state[i, j, k, 4 + 3 * r + c]
                _mm(ss, ss, s2)
                _mm(s2, ss, s3)
                cc = 0.5 * (1.0 - qq[0])
                b = delta * cc
                cp = (1.0 + delta) * cc
                ct = delta * cc * cc
                tr_s = ss[0, 0] + ss[1, 1] + ss[2, 2]
                rr = tr_s - cp
                rec2 = 0.0
                for r in range(3):
                    for c in range(3):
                        rec[r, c] = s3[r, c] - cp * s2[r, c] + ct * ss[r, c]
                        rec2 += rec[r, c] * rec[r, c]
                        R[r, c] = 2.0 * coeff * rec[r, c]
                av += rr * rr + rec2
                am += 1.0 - qq[0]
                # total adjoint of the record s
                for r in range(3):
                    for c in range(3):
                        T[r, c] = cR * dS[i, j, k, r, c]
                    T[r, r] += 2.0 * coeff * rr
                _poly_adjoint(R, ss, s2, -cp, ct, T, t, t2)
                dc = -2.0 * coeff * rr * (1.0 + delta)
                for r in range(3):
                    for c in range(3):
                        dc += R[r, c] * (-(1.0 + delta) * s2[r, c] + 2.0 * delta * cc * ss[r, c])
                # orientation: u = (s - s^2) / scale
                for r in range(3):
                    for c in range(3):
                        D1[r, c] = cO * dU[i, j, k, r, c] / scale
                        T[r, c] += D1[r, c]
                _mm(D1, ss, t)
                _mm(ss, D1, t2)
                for r in range(3):
                    for c in range(3):
                        T[r, c] -= t[r, c] + t2[r, c]
                # composites: x2 = s x1, x1 = s^3 - (1 + b) s^2 + b s
                for r in range(3):
                    for c in range(3):
                        D2[r, c] = cX * d2[i, j, k, r, c]
                _mm(D2, x1[i, j, k], t)
                _mm(ss, D2, t2)
                for r in range(3):
                    for c in range(3):
                        T[r, c] += t[r, c]
                        D1[r, c] = cX * d1[i, j, k, r, c] + t2[r, c]
                _poly_adjoint(D1, ss, s2, -(1.0 + b), b, T, t, t2)
                db = 0.0
                for r in range(3):
                    for c in range(3):
                        db += D1[r, c] * (ss[r, c] - s2[r, c])
                # tensor: N = - q q^T + lam a a^T / G
                aa = a[i, j, k]
                for r in range(4):
                    dq[r] = 0.0
                    for c in range(4):
                        dq[r] -= cN * (dN[i, j, k, r, c] + dN[i, j, k, c, r]) * qq[c]
                    for c in range(3):
                        acc = 0.0
                        for m in range(4):
                            acc += (dN[i, j, k, r, m] + dN[i, j, k, m, r]) * aa[m, c]
                        dA[r, c] = lam * cN * acc / G_TIME
                for r in range(4):
                    for c in range(3):
                        B[r, c] = dA[r, 0] * f[c, 0] + dA[r, 1] * f[c, 1] + dA[r, 2] * f[c, 2]
                dq[0] += B[1, 0] + B[2, 1] + B[3, 2]
                dq[1] += -B[0, 0] + B[3, 1] - B[2, 2]
                dq[2] += -B[3, 0] - B[0, 1] + B[1, 2]
                dq[3] += B[2, 0] - B[1, 1] - B[0, 2]
                dq[0] -= 0.5 * delta * db + 0.5 * dc + BETA2 * h ** 3
                for r in range(4):
                    dq[r] += gq[i, j, k, r]
                # q = p / |p|
                p0 = state[i, j, k, 0]
                p1 = state[i, j, k, 1]
                p2 = state[i, j, k, 2]
                p3 = state[i, j, k, 3]
                inv = 1.0 / math.sqrt(p0 * p0 + p1 * p1 + p2 * p2 + p3 * p3)
                qd = qq[0] * dq[0] + qq[1] * dq[1] + qq[2] * dq[2] + qq[3] * dq[3]
                for r in range(4):
                    grad[i, j, k, r] = (dq[r] - qq[r] * qd) * inv
                # f: s = f^T f gives f (T + T^T); a = tau f gives tau^T dA
                _triad(qq, tau)
                for r in range(3):
                    for c in range(3):
                        acc = 0.0
                        for m in range(3):
                            acc += f[r, m] * (T[m, c] + T[c, m])
                        for m in range(4):
                            acc += tau[m, r] * dA[m, c]
                        grad[i, j, k, 4 + 3 * r + c] = acc
        ev[i] = av
        em[i] = am
    return ev.sum(), em.sum()


# ----------------------------------------------------------------------------------------------
# the evaluator
# ----------------------------------------------------------------------------------------------
class CardC:
    """Card C on one grid: `terms, gradient = card(state)`; the boundary of `state` is held."""

    def __init__(self, grid, box, delta=0.2, kx=0.25, ko=0.02, lam=1.0):
        self.n = int(grid)
        self.box = float(box)
        self.h = 2.0 * self.box / (self.n - 1)
        self.delta, self.kx, self.ko, self.lam = float(delta), float(kx), float(ko), float(lam)
        n = self.n
        self.q = np.zeros((n, n, n, 4))
        self.a = np.zeros((n, n, n, 4, 3))
        self.N = np.zeros((n, n, n, 4, 4))
        self.s = np.zeros((n, n, n, 3, 3))
        self.x1 = np.zeros((n, n, n, 3, 3))
        self.x2 = np.zeros((n, n, n, 3, 3))
        self.u = np.zeros((n, n, n, 3, 3))
        self.dN = np.zeros((n, n, n, 4, 4))
        self.d1 = np.zeros((n, n, n, 3, 3))
        self.d2 = np.zeros((n, n, n, 3, 3))
        self.dU = np.zeros((n, n, n, 3, 3))
        self.dS = np.zeros((n, n, n, 3, 3))
        self.grad = np.zeros((n, n, n, 13))

    def fields(self, state):
        _node_fields(state, self.delta, self.lam, self.q, self.a, self.N, self.s, self.x1, self.x2, self.u)
        return self.q, self.s

    def __call__(self, state, fields_done=False):
        """The seven terms and the gradient on the full grid (use `interior` on it)."""
        h = self.h
        if not fields_done:
            self.fields(state)
        eP, gq = _quartic(self.q, OFFSETS, h)
        sN, s1, s2, sO = _edge_terms(self.N, self.x1, self.x2, self.u, self.dN, self.d1, self.d2, self.dU)
        self.dS[:] = 0.0
        sR = _record_quartic(self.s, self.dS)
        cN = TENSOR * h           # adjoint factor of (TENSOR h / 2) sum w |dN|^2
        cX = 0.5 * self.kx * h
        cR = RECORD4 / (2.0 * h)  # energy (RECORD4 / 2h) sum |C|^2, adjoint already carries the 2
        cO = self.ko * h          # the gather adjoint is [[ux, uy], uy], half of the full derivative
        sV, sM = _chain(state, self.q, self.a, self.s, self.x1, self.dN, self.d1, self.d2, self.dU, self.dS, gq,
                        self.delta, self.lam, cN, cX, cR, cO, h, self.grad)
        terms = np.array([0.5 * TENSOR * h * sN, eP, 0.5 * RECORD4 / h * sR, 0.25 * self.kx * h * (s1 + s2),
                          POTENTIAL * h ** 3 * sV, BETA2 * h ** 3 * sM, 0.5 * self.ko * h * sO])
        self.composites = (0.25 * self.kx * h * s1, 0.25 * self.kx * h * s2)
        return terms, self.grad

    def margins(self, state=None, all_four=True):
        """The hemisphere margins (triangulation 0 alone, the wall of the card's driver; and all
        four when asked). Positive when every simplex lies in an open hemisphere."""
        if state is not None:
            self.fields(state)
        return _hemisphere(self.q, OFFSETS, 4 if all_four else 1)


def interior(field):
    return field[1:-1, 1:-1, 1:-1]


# ----------------------------------------------------------------------------------------------
# the seeds and the maps between grids (the card's own starts, written from their formulas)
# ----------------------------------------------------------------------------------------------
def coordinates(grid, box):
    x = np.linspace(-box, box, grid)
    return np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1), x[1] - x[0]


def winding_profile(coords, box):
    """The seed's winding field: a hedgehog whose angle falls linearly from pi at the center to a
    cubic foot over the last 15 percent of the radius 0.96 box."""
    radius = np.linalg.norm(coords, axis=-1)
    t = np.clip(1.0 - radius / (0.96 * box), 0.0, 1.0)
    edge = 0.15
    z = t / edge
    phase = np.pi * np.where(t < edge, edge * (2.0 * z * z - z * z * z), t)
    q = np.concatenate((np.cos(phase)[..., None],
                        np.sin(phase)[..., None] * coords / np.maximum(radius[..., None], 1e-15)), axis=-1)
    q[radius < 1e-14, 0] = -1.0
    return q


def frame_of(q):
    """The rotation matrix R(q) of the adjoint action of the unit quaternion q."""
    a, b, c, d = np.moveaxis(q, -1, 0)
    R = np.stack((1 - 2 * (c * c + d * d), 2 * (b * c - a * d), 2 * (b * d + a * c),
                  2 * (b * c + a * d), 1 - 2 * (b * b + d * d), 2 * (c * d - a * b),
                  2 * (b * d - a * c), 2 * (c * d + a * b), 1 - 2 * (b * b + c * c)), axis=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def seed(grid, box, kind, delta=0.2):
    """The card's `linked` or `smooth` seed, or `iso` (ours): the linked seed with its record blended
    to sqrt(ISO_VALUE) times its own frame inside a Gaussian core of one card length."""
    coords, h = coordinates(grid, box)
    radius = np.linalg.norm(coords, axis=-1)
    taper = np.clip((0.96 * box - radius) / (0.24 * box), 0.0, 1.0)
    taper = taper * taper * (3.0 - 2.0 * taper)
    q = winding_profile(coords, box)
    if kind in ("linked", "iso"):
        R = frame_of(q)
    elif kind == "smooth":
        R = np.broadcast_to(np.eye(3), q.shape[:3] + (3, 3))
    else:
        raise ValueError(kind)
    core = 0.08 * np.exp(-radius * radius / 2.0) * taper
    diag = np.zeros(q.shape[:3] + (3, 3))
    diag[..., 0, 0] = core
    diag[..., 1, 1] = 0.65 * core
    diag[..., 2, 2] = 1.0
    f = diag @ np.swapaxes(R, -1, -2)
    boundary = np.ones(q.shape[:3], dtype=bool)
    boundary[1:-1, 1:-1, 1:-1] = False
    f[boundary] = 0.0
    # the middle row is stretched to the length sqrt(b + len^2), b the card's middle eigenvalue target
    b = 0.5 * delta * (1.0 - q[..., 0])
    length = np.linalg.norm(f[..., 1, :], axis=-1)
    mask = length > 0
    f[..., 1, :][mask] *= (np.sqrt(b[mask] + length[mask] ** 2) / length[mask])[:, None]
    if kind == "iso":
        w = np.exp(-radius * radius / 2.0)[..., None, None]
        blend = (1.0 - w) * f + w * math.sqrt(ISO_VALUE) * np.swapaxes(R, -1, -2)
        f[1:-1, 1:-1, 1:-1] = blend[1:-1, 1:-1, 1:-1]
    state = np.concatenate((q, f.reshape(q.shape[:3] + (9,))), axis=-1)
    return state, coords, h


def vacuum(grid):
    state = np.zeros((grid, grid, grid, 13))
    state[..., 0] = 1.0
    return state


def refine(state, coords_old, grid, box):
    """The card's refinement path: cubic interpolation of the 13 components onto a finer grid of the
    same box, the winding field renormalized, the boundary reset to the card's boundary values."""
    coords, h = coordinates(grid, box)
    u = coords_old[:, 0, 0, 0]
    if abs(coords[0, 0, 0, 0] - u[0]) > 1e-12 or abs(coords[-1, 0, 0, 0] - u[-1]) > 1e-12:
        raise ValueError("refinement requires the same physical box")
    z = np.moveaxis((coords - u[0]) / (u[1] - u[0]), -1, 0)
    fine = np.stack([map_coordinates(c, z, order=3, mode="nearest") for c in np.moveaxis(state, -1, 0)], axis=-1)
    out = vacuum(grid)
    inner = fine[1:-1, 1:-1, 1:-1].copy()
    inner[..., :4] /= np.linalg.norm(inner[..., :4], axis=-1, keepdims=True)
    out[1:-1, 1:-1, 1:-1] = inner
    return out, coords, h


def pad(state, grid):
    """Box continuation: the state set in a larger box at the same spacing, the new shell at the
    card's boundary values (q = (1, 0, 0, 0), f = 0)."""
    n0 = state.shape[0]
    width = grid - n0
    if width < 0 or width % 2:
        raise ValueError("the larger grid must add the same number of nodes on both sides")
    out = vacuum(grid)
    w = width // 2
    out[w:w + n0, w:w + n0, w:w + n0] = state
    return out


def reset_frame(state, delta=0.2):
    """The card's frame reset: the factor f rewritten as a lab-frame diagonal set by q0 alone."""
    out = state.copy()
    q0 = out[..., 0]
    b = 0.5 * delta * (1.0 - q0)
    f = np.zeros(out.shape[:3] + (3, 3))
    f[..., 0, 0] = 0.025 * b
    f[..., 1, 1] = np.sqrt(b)
    f[..., 2, 2] = np.sqrt(np.maximum(1.0 - q0, 0.0) / 2.0)
    out[..., 4:] = f.reshape(out.shape[:3] + (9,))
    return out


def mirror(state, axis):
    """The mirror image of a state in the plane x_axis = 0.

    The reflection sigma acts on the coordinates and on the vector part of the winding field,
    q'(x) = (q0, sigma q_vec)(sigma x). That map is q -> e q-bar e^-1 with e the unit of the axis, an
    anti-automorphism, so it turns the right triad tau(q) into the left triad of q' times sigma; the
    left triad is the right one times the rotation v -> q'-bar v q'. The factor that carries the
    tensor N to its mirror image is therefore f'(x) = Rot(q'-bar) sigma f(sigma x), and the record
    is carried along with its node, s'(x) = s(sigma x).
    """
    out = np.flip(state, axis=axis).copy()
    out[..., 1 + axis] *= -1.0
    p = out[..., :4]
    q = p / np.linalg.norm(p, axis=-1, keepdims=True)
    rot = frame_of(q * np.array([1.0, -1.0, -1.0, -1.0]))
    f = out[..., 4:].reshape(out.shape[:3] + (3, 3)).copy()
    f[..., axis, :] *= -1.0
    out[..., 4:] = (rot @ f).reshape(out.shape[:3] + (9,))
    return out


def smooth_test_state(grid, box, amplitude=0.05):
    """A smooth state with no mirror symmetry and the card's boundary values: the linked seed plus
    low-order trigonometric fields under the seed's taper. The same function on every grid."""
    state, coords, h = seed(grid, box, "linked")
    radius = np.linalg.norm(coords, axis=-1)
    taper = np.clip((0.96 * box - radius) / (0.24 * box), 0.0, 1.0)
    taper = taper * taper * (3.0 - 2.0 * taper)
    x, y, z = np.moveaxis(coords, -1, 0)
    for c in range(13):
        k = np.array([0.7 + 0.13 * c, 0.5 - 0.07 * c, 0.9 + 0.05 * c])
        phase = 0.4 + 0.9 * c
        state[..., c] += amplitude * taper * np.sin(k[0] * x + k[1] * y + k[2] * z + phase) * np.cos(0.3 * (c + 1) * x - 0.2 * y)
    return state, coords, h


# ----------------------------------------------------------------------------------------------
# reads that label a state
# ----------------------------------------------------------------------------------------------
@nb.njit(cache=True)
def _det4(m):
    d = 0.0
    for c in range(4):
        cols = [x for x in range(4) if x != c]
        minor = (m[1, cols[0]] * (m[2, cols[1]] * m[3, cols[2]] - m[2, cols[2]] * m[3, cols[1]])
                 - m[1, cols[1]] * (m[2, cols[0]] * m[3, cols[2]] - m[2, cols[2]] * m[3, cols[0]])
                 + m[1, cols[2]] * (m[2, cols[0]] * m[3, cols[1]] - m[2, cols[1]] * m[3, cols[0]]))
        d += (-1.0) ** c * m[0, c] * minor
    return d


@nb.njit(cache=True)
def _degree(q, off, target, perm_sign):
    """Signed count of the simplices of triangulation 0 whose positive cone holds `target`."""
    n = q.shape[0]
    total = 0
    hits = 0
    m = np.zeros((4, 4))
    mc = np.zeros((4, 4))
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                for o in range(6):
                    for v in range(4):
                        for t in range(4):
                            m[t, v] = q[i + off[0, o, v, 0], j + off[0, o, v, 1], k + off[0, o, v, 2], t]
                    det = _det4(m)
                    if abs(det) <= 1e-13:
                        continue
                    inside = True
                    for v in range(4):
                        for r in range(4):
                            for c in range(4):
                                mc[r, c] = m[r, c]
                            mc[r, v] = target[r]
                        if _det4(mc) / det <= 1e-12:
                            inside = False
                            break
                    if inside:
                        hits += 1
                        total += perm_sign[o] * (1 if det > 0 else -1)
    return -total, hits


def degree(q):
    """The winding number of q, by the count of preimages of one generic point of the sphere."""
    target = np.array([0.234, 0.374, -0.557, 0.699])
    target /= np.linalg.norm(target)
    signs = np.array([round(np.linalg.det(np.eye(3)[list(p)])) for p in itertools.permutations(range(3))], dtype=np.int64)
    return _degree(np.ascontiguousarray(q), OFFSETS, target, signs)


def center_eigenvalues(state):
    """Eigenvalues of the record s = f^T f at the center node (odd grids)."""
    n = state.shape[0]
    f = state[n // 2, n // 2, n // 2, 4:].reshape(3, 3)
    return np.linalg.eigvalsh(f.T @ f)


def branch(eigs):
    """The declared branch label, from the center eigenvalues of the record."""
    lo, hi = float(np.min(eigs)), float(np.max(eigs))
    if lo > 0.2 and hi - lo < 0.1:
        return "ISOTROPIC CENTER"
    if lo < 0.1:
        return "SPLIT CENTER"
    return "MIXED"


def save_state(path, state, coords, compressed=True, **extra):
    """A state in the layout the card's own audits read: state, record, q, coords."""
    p = state[..., :4]
    q = p / np.linalg.norm(p, axis=-1, keepdims=True)
    f = state[..., 4:].reshape(state.shape[:3] + (3, 3))
    record = np.swapaxes(f, -1, -2) @ f
    (np.savez_compressed if compressed else np.savez)(path, state=state, record=record, q=q, coords=coords, **extra)
