"""Exact machinery on V_j: CG tables, channel fractions, channel coefficients from the
isotypic projector, and the quartic h with its exact gradient and Hessian in the real
coordinates (Re w_m, Im w_m) of the orthonormal Condon-Shortley components.

Index convention: k = j + m, k = 0..n (n = 2j); real coordinate vector = (Re w_0..Re w_n, Im w_0..Im w_n).
"""
import sympy as sp
from functools import lru_cache
from lib import cg, Nm


def S(x):
    return sp.nsimplify(x)


@lru_cache(maxsize=None)
def cg_table(n):
    """{J: {M: [(k1, k2, C)]}} for V_j x V_j -> V_J, j = n/2."""
    j = sp.Rational(n, 2)
    tab = {}
    for J in range(0, n + 1):
        tab[J] = {}
        for M in range(-J, J + 1):
            lst = []
            for k1 in range(n + 1):
                k2 = M + n - k1  # m1 + m2 = M  <=> k1 + k2 = M + n
                if 0 <= k2 <= n:
                    c = cg(j, k1 - j, j, k2 - j, J, M)
                    if c != 0:
                        lst.append((k1, k2, c))
            tab[J][M] = lst
    return tab


def sym_channels(n):
    """J values occurring in Sym^2 V_j: J = n, n-2, ..."""
    return list(range(n, -1, -2))


def G_JM(n, w, J, M):
    return sum((c * w[k1] * w[k2] for k1, k2, c in cg_table(n)[J][M]), sp.Integer(0))


def fJ(n, w, simplify=True):
    """{J: ||[w x w]_J||^2 / ||w||^4} for J = n, n-2, ..."""
    nw2 = sum(sp.Abs(x) ** 2 for x in w)
    out = {}
    for J in sym_channels(n):
        tot = sum((sp.Abs(sp.expand(G_JM(n, w, J, M))) ** 2 for M in range(-J, J + 1)), sp.Integer(0))
        val = tot / nw2 ** 2
        out[J] = sp.radsimp(sp.expand(val)) if simplify else val
    return out


def qJ_from_projector(n, P, dim_rho):
    """q_J = tr(Pi_J (P x P)) / ((2J+1) s^2), s = dim rho / (2j+1); P the orthogonal
    projector on Condon-Shortley components (sympy Matrix)."""
    s = sp.Rational(dim_rho, n + 1)
    tab = cg_table(n)
    out = {}
    for J in sym_channels(n):
        tr = sp.Integer(0)
        for M in range(-J, J + 1):
            L = tab[J][M]
            for a1, a2, c in L:
                for b1, b2, d in L:
                    tr += c * d * P[a1, b1] * P[a2, b2]
        tr = sp.radsimp(sp.expand(tr))
        out[J] = sp.radsimp(tr / ((2 * J + 1) * s ** 2))
    return out


# ------------------------------------------------------------------ the quartic h
def real_basis(n):
    """complex direction vectors for the real coordinates (Re w_0..Re w_n, Im w_0..Im w_n)."""
    B = []
    for k in range(n + 1):
        v = [sp.Integer(0)] * (n + 1); v[k] = sp.Integer(1); B.append(v)
    for k in range(n + 1):
        v = [sp.Integer(0)] * (n + 1); v[k] = sp.I; B.append(v)
    return B


def to_real(w):
    return [sp.re(x) for x in w] + [sp.im(x) for x in w]


def from_real(x):
    n1 = len(x) // 2
    return [x[k] + sp.I * x[n1 + k] for k in range(n1)]


def simp(e):
    return sp.radsimp(sp.expand(e))


def h_data(n, q, u):
    """Exact h(u), real gradient and real Hessian of h = sum_J q_J sum_M |G_JM(w)|^2 at u."""
    tab = cg_table(n)
    B = real_basis(n)
    dimR = 2 * (n + 1)
    h = sp.Integer(0)
    grad = [sp.Integer(0)] * dimR
    H = [[sp.Integer(0)] * dimR for _ in range(dimR)]
    for J, qJ in q.items():
        for M in range(-J, J + 1):
            L = tab[J][M]
            if not L:
                continue
            G = sum((c * u[k1] * u[k2] for k1, k2, c in L), sp.Integer(0))
            Gc = sp.conjugate(G)
            dG = []
            for d in B:
                dG.append(sum((c * (u[k1] * d[k2] + d[k1] * u[k2]) for k1, k2, c in L), sp.Integer(0)))
            h += qJ * G * Gc
            for a in range(dimR):
                grad[a] += qJ * 2 * sp.re(Gc * dG[a])
            for a in range(dimR):
                for b in range(a, dimR):
                    da, db = B[a], B[b]
                    d2G = sum((c * (da[k1] * db[k2] + db[k1] * da[k2]) for k1, k2, c in L), sp.Integer(0))
                    val = qJ * 2 * sp.re(sp.conjugate(dG[b]) * dG[a] + Gc * d2G)
                    H[a][b] += val
    h = simp(h)
    grad = [simp(g) for g in grad]
    for a in range(dimR):
        for b in range(a, dimR):
            H[a][b] = simp(H[a][b]); H[b][a] = H[a][b]
    return h, grad, sp.Matrix(H)


# ------------------------------------------------------------------ real-linear maps
def mat_i(n):
    n1 = n + 1
    A = sp.zeros(2 * n1, 2 * n1)
    for k in range(n1):
        A[n1 + k, k] = 1    # Re -> Im
        A[k, n1 + k] = -1   # Im -> -Re
    return A


def mat_theta(n):
    """(Theta u)_m = (-1)^(j-m) conj(u_{-m}); index k = j+m, -m <-> n-k, j-m = n-k."""
    n1 = n + 1
    A = sp.zeros(2 * n1, 2 * n1)
    for k in range(n1):
        sgn = (-1) ** (n - k)
        A[k, n - k] = sgn             # Re part
        A[n1 + k, n1 + (n - k)] = -sgn  # Im part (conjugation)
    return A


def complex_to_real_matrix(Cm):
    """real 2N x 2N matrix of the C-linear map w -> Cm w."""
    N = Cm.shape[0]
    A = sp.zeros(2 * N, 2 * N)
    for r in range(N):
        for c in range(N):
            z = Cm[r, c]
            A[r, c] = sp.re(z); A[r, N + c] = -sp.im(z)
            A[N + r, c] = sp.im(z); A[N + r, N + c] = sp.re(z)
    return A


def J_ops(n):
    """Condon-Shortley Jz, J+, J- on index k."""
    j = sp.Rational(n, 2)
    n1 = n + 1
    Jz = sp.zeros(n1, n1); Jp = sp.zeros(n1, n1); Jm = sp.zeros(n1, n1)
    for k in range(n1):
        m = k - j
        Jz[k, k] = m
        if k + 1 < n1:
            Jp[k + 1, k] = sp.sqrt((j - m) * (j + m + 1))
        if k - 1 >= 0:
            Jm[k - 1, k] = sp.sqrt((j + m) * (j - m + 1))
    return Jz, Jp, Jm


def rotation_generators_real(n):
    Jz, Jp, Jm = J_ops(n)
    X1 = sp.I * Jz
    X2 = sp.I * (Jp + Jm) / 2
    X3 = (Jp - Jm) / 2
    return [complex_to_real_matrix(X) for X in (X1, X2, X3)]
