"""Route B (high precision, mpmath): the Peter-Weyl / Clebsch-Gordan form of the slot integrals.

psi_y(g)_a = s^-1/2 sum y_m D^j_{mm'}(g) eta_{m'a};  conj psi = s^-1/2 (Theta y)^T D (Theta eta)
(conj D_{mm'} = (-1)^{m-m'} D_{-m,-m'}, checked in s01).  Coupling D^j D^j = sum CG CG D^{J1} and
D^{J1} D^j = sum CG CG D^J gives
  N(psi_y)_a = s^-3/2 sum_{J1,J} sum_{M,M'} L^{J1 J}_M(y) R^{J1 J}_{M' a} D^J_{MM'}(g),
  L^{J1 J}(y) = [[y (x) y]_{J1} (x) Theta y]_J,  R^{J1 J}_a = sum_b [[eta_a (x) eta_b]_{J1} (x) Theta eta_b]_J,
and with mean D^J_{MM'} conj D^J'_{NN'} = delta/(2J+1):
  ||Pi_{2J} N||^2 = s^-3/(2J+1) sum_{J1,J1'} <L^{J1J}, L^{J1'J}> G^J_{J1 J1'},
  G^J_{J1J1'} = sum_{M',a} R^{J1J}_{M'a} conj R^{J1'J}_{M'a},
  h(y) = mean|psi_y|^4 = s^-2/(2j+1) sum_{J1} <L^{J1 j}(y), y> H_{J1},  H_{J1} = sum_{M'a} R^{J1j}_{M'a} conj eta_{M'a}.
All means are Haar means = means over X with total volume 1 (V = 1).
"""
import json, os
from fractions import Fraction as Fr
import mpmath as mp
import sympy as sp
from lib_cg import cg_table, mrange, sgnsq_to_mp
from lib_rep import theta_apply, Jmats

HERE = os.path.dirname(os.path.abspath(__file__))
SLOT_J = {"R2": Fr(7, 2), "R4": Fr(3), "R5": Fr(3)}


def load_eta(slot):
    E = json.load(open(os.path.join(HERE, "out", "eta_exact.json")))[slot]
    return mp.matrix([[mp.mpf(sp.N(sp.sympify(x), mp.mp.dps + 10)) for x in row] for row in E])


_cgnum = {}


def cgnum(j1, j2, J):
    key = (j1, j2, J, mp.mp.dps)
    if key not in _cgnum:
        t = cg_table(Fr(j1), Fr(j2), Fr(J))
        ms1, ms2, msJ = mrange(j1), mrange(j2), mrange(J)
        _cgnum[key] = [(ms1.index(m1), ms2.index(m2), msJ.index(M), sgnsq_to_mp(sq, mp)) for (m1, m2, M), sq in t.items()]
    return _cgnum[key]


def couple(j1, j2, J, u, v):
    out = [mp.mpc(0)] * int(2 * J + 1)
    for i1, i2, iM, c in cgnum(j1, j2, J):
        out[iM] += c * u[i1] * v[i2]
    return out


def theta(j, u):
    n = len(u)
    ms = mrange(j)
    return [(-1) ** int(j - ms[r]) * mp.conj(u[n - 1 - r]) for r in range(n)]


def dot(u, v):  # sum u conj v
    return mp.fsum(a * mp.conj(b) for a, b in zip(u, v))


class Slot:
    def __init__(self, slot, sN=None):
        self.name = slot
        self.j = j = SLOT_J[slot]
        self.n = int(2 * j + 1)
        self.eta = load_eta(slot)
        self.d = self.eta.cols
        self.sN = mp.mpf(self.d) / self.n if sN is None else sN
        self.lam0 = int(2 * j) * (int(2 * j) + 2)
        self.J1s = [J1 for J1 in range(0, int(2 * j) + 1) if (int(2 * j) - J1) % 2 == 0]  # [y(x)y]_J1 != 0
        self.Js = {}
        for J1 in self.J1s:
            J = abs(Fr(J1) - j)
            while J <= J1 + j:
                self.Js.setdefault(J, []).append(J1)
                J += 1
        etac = [[self.eta[r, a] for r in range(self.n)] for a in range(self.d)]
        teta = [theta(j, c) for c in etac]
        R = {}
        for J1 in self.J1s:
            for J in [J for J in self.Js if J1 in self.Js[J]]:
                Ra = []
                for a in range(self.d):
                    acc = [mp.mpc(0)] * int(2 * J + 1)
                    for b in range(self.d):
                        t = couple(Fr(J1), j, J, couple(j, j, Fr(J1), etac[a], etac[b]), teta[b])
                        acc = [x + y for x, y in zip(acc, t)]
                    Ra.append(acc)
                R[(J1, J)] = Ra
        self.G = {}
        for J, J1list in self.Js.items():
            self.G[J] = {(A, B): mp.fsum(dot(R[(A, J)][a], R[(B, J)][a]) for a in range(self.d)) for A in J1list for B in J1list}
        self.H = {J1: mp.fsum(dot(R[(J1, j)][a], etac[a]) for a in range(self.d)) for J1 in self.Js[j]}

    def L(self, y):
        j = self.j
        ty = theta(j, y)
        Y1 = {J1: couple(j, j, Fr(J1), y, y) for J1 in self.J1s}
        return {(J1, J): couple(Fr(J1), j, J, Y1[J1], ty) for J, J1list in self.Js.items() for J1 in J1list}

    def level_norms(self, y, L=None):
        L = self.L(y) if L is None else L
        out = {}
        for J, J1list in self.Js.items():
            s = mp.fsum(dot(L[(A, J)], L[(B, J)]) * self.G[J][(A, B)] for A in J1list for B in J1list)
            out[J] = mp.re(s) / (self.sN ** 3 * (2 * J + 1))
        return out

    def h(self, y, L=None):
        if L is None:
            j = self.j
            ty = theta(j, y)
            L = {(J1, j): couple(Fr(J1), j, j, couple(j, j, Fr(J1), y, y), ty) for J1 in self.Js[j]}
        s = mp.fsum(dot(L[(J1, self.j)], y) * self.H[J1] for J1 in self.Js[self.j])
        return mp.re(s) / (self.sN ** 2 * self.n)

    def Rt0(self, y):
        """R~_0(y) = - sum_{L != n} ||Pi_L N||^2 / (lambda_L - lambda_0), V = R = 1."""
        ln = self.level_norms(y)
        tot = mp.mpf(0)
        for J, v in ln.items():
            L = int(2 * J)
            if J == self.j:
                continue
            tot -= v / (L * (L + 2) - self.lam0)
        return tot

    # ----- real coordinates x = (Re y, Im y), orthonormal CS components -----
    def to_y(self, x):
        n = self.n
        return [mp.mpc(x[k], x[n + k]) for k in range(n)]

    def hx(self, x):
        return self.h(self.to_y(x))

    def Rx(self, x):
        return self.Rt0(self.to_y(x))


# ---------------- polynomial calculus by exact-degree interpolation ----------------
def _coeffs_along(f, x, v, deg):
    ts = list(range(-(deg // 2), deg - deg // 2 + 1))
    vals = [f([xi + t * vi for xi, vi in zip(x, v)]) for t in ts]
    V = mp.matrix([[mp.mpf(t) ** k for k in range(deg + 1)] for t in ts])
    return mp.lu_solve(V, mp.matrix(vals))


def grad(f, x, deg):
    n = len(x)
    g = []
    for k in range(n):
        e = [0] * n
        e[k] = 1
        g.append(_coeffs_along(f, x, e, deg)[1])
    return mp.matrix(g)


def hess(f, x, deg):
    n = len(x)
    c2 = {}
    for k in range(n):
        e = [0] * n
        e[k] = 1
        c2[k] = _coeffs_along(f, x, e, deg)[2]
    H = mp.matrix(n, n)
    for k in range(n):
        H[k, k] = 2 * c2[k]
        for l in range(k + 1, n):
            e = [0] * n
            e[k] = 1
            e[l] = 1
            H[k, l] = H[l, k] = _coeffs_along(f, x, e, deg)[2] - c2[k] - c2[l]
    return H


def dir_deriv(f, v, deg):
    """x -> Df(x)[v] for a polynomial f of degree deg (result has degree deg-1)."""
    return lambda x: _coeffs_along(f, x, v, deg)[1]


def Jcx(n):
    """complex structure i on real coordinates (Re, Im): i(a + ib) = -b + ia."""
    Jm = mp.matrix(2 * n, 2 * n)
    for k in range(n):
        Jm[k, n + k] = -1
        Jm[n + k, k] = 1
    return Jm


def to_real(y):
    return mp.matrix([mp.re(c) for c in y] + [mp.im(c) for c in y])


def orbit_tangents(slot, u):
    """real tangent vectors of the orbit Gamma u at u: phase i u, rotations i J_a u (a = x, y, z),
    and in R2 also the quaternionic directions Theta u, i Theta u."""
    j = slot.j
    n = slot.n
    Jz, Jp, Jm, Jx, Jy = Jmats(j, mp)
    uc = mp.matrix(u)
    I = mp.mpc(0, 1)
    vecs = [[I * c for c in u]]
    for Ja in (Jx, Jy, Jz):
        t = Ja * uc
        vecs.append([I * t[k] for k in range(n)])
    if slot.name == "R2":
        tu = theta(j, u)
        vecs.append(tu)
        vecs.append([I * c for c in tu])
    return [to_real(v) for v in vecs]


def orth_basis(vecs, tol):
    """orthonormal basis (columns) of the span of real vectors, by modified Gram-Schmidt with
    re-orthogonalisation; returns (basis list, singular values)."""
    A = mp.matrix(len(vecs[0]), len(vecs))
    for c, v in enumerate(vecs):
        for r in range(len(v)):
            A[r, c] = v[r]
    U, S, V = mp.svd_r(A)
    basis = [U[:, k] for k in range(len(vecs)) if S[k] > tol]
    return basis, [S[k] for k in range(min(A.rows, A.cols))]
