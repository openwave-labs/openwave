"""Interval code for Block D (mpmath.iv), offline.

Slice system (19 equations, unknowns z = (x[14], lam, c1..c4)):
  F1 = grad h(x) - lam G x - sum_i c_i G k_i,   F2 = x^T G x - 1,   F3_i = (G k_i)^T (x - x0),
  k_i = X_i x0 (X_i the bench generators of inputs.json).
Krawczyk: K(Z) = zt - Y F(zt) + (I - Y J(Z)) (Z - zt); K(Z) in int(Z) => unique zero in Z."""
import json, os
from fractions import Fraction as Fr
import mpmath as mpm
from mpmath import iv
from lib_quartic import load, pdiff, Compiled, NV

HERE = os.path.dirname(os.path.abspath(__file__))
DPS = 220
GDIAG = [720, 120, 48, 36, 48, 120, 720] * 2


def set_prec(d=DPS):
    iv.dps = d
    mpm.mp.dps = d


def ivq(fr):
    fr = Fr(fr)
    return iv.mpf(fr.numerator) / fr.denominator


def lo(I):
    return mpm.mp.make_mpf(I._mpi_[0])


def hi(I):
    return mpm.mp.make_mpf(I._mpi_[1])


def midf(I):
    return (lo(I) + hi(I)) / 2


def width(I):
    return hi(I) - lo(I)


def excludes_zero(I):
    return lo(I) > 0 or hi(I) < 0


def sign_of(I):
    if lo(I) > 0:
        return 1
    if hi(I) < 0:
        return -1
    return 0


def load_inputs():
    with open(os.path.join(HERE, "inputs.json")) as f:
        return json.load(f)


def generators_exact():
    inp = load_inputs()
    return [[[Fr(v) for v in row] for row in M] for M in inp["generators"]["matrices"]]


class Slot:
    def __init__(self, name, poly=None):
        self.name = name
        if poly is None:
            poly, meta = load(os.path.join(HERE, f"quartic_{name}.json"))
        self.poly = poly
        self.grad_p = [pdiff(poly, a) for a in range(NV)]
        self.hess_p = {}
        for a in range(NV):
            for b in range(a, NV):
                self.hess_p[(a, b)] = pdiff(self.grad_p[a], b)
        self.compile_iv()

    def compile_iv(self):
        self.h_iv = Compiled(self.poly, ivq)
        self.g_iv = [Compiled(p, ivq) for p in self.grad_p]
        self.H_iv = {k: Compiled(p, ivq) for k, p in self.hess_p.items()}
        mpq = lambda c: mpm.mpf(c.numerator) / c.denominator
        self.h_mp = Compiled(self.poly, mpq)
        self.g_mp = [Compiled(p, mpq) for p in self.grad_p]
        self.H_mp = {k: Compiled(p, mpq) for k, p in self.hess_p.items()}

    def h(self, x, interval=True):
        return (self.h_iv if interval else self.h_mp).eval(x, iv.mpf(0) if interval else mpm.mpf(0))

    def grad(self, x, interval=True):
        z = iv.mpf(0) if interval else mpm.mpf(0)
        return [(self.g_iv if interval else self.g_mp)[a].eval(x, z) for a in range(NV)]

    def hess(self, x, interval=True):
        z = iv.mpf(0) if interval else mpm.mpf(0)
        Hs = self.H_iv if interval else self.H_mp
        H = [[None] * NV for _ in range(NV)]
        for (a, b), p in Hs.items():
            v = p.eval(x, z)
            H[a][b] = v
            H[b][a] = v
        return H


# ------------------------------------------------------------------ small interval linear algebra
def matvec(M, v):
    out = []
    for row in M:
        acc = iv.mpf(0)
        for a, b in zip(row, v):
            acc += a * b
        out.append(acc)
    return out


def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    Bt = [[B[k][j] for k in range(m)] for j in range(p)]
    return [[sum((A[i][k] * Bt[j][k] for k in range(m)), iv.mpf(0)) for j in range(p)] for i in range(n)]


def transpose(A):
    return [list(r) for r in zip(*A)]


def gauss_jordan_inverse(A):
    """naive interval Gauss-Jordan; valid enclosure of A^{-1} for every point matrix in A
    provided no pivot interval contains 0 (checked)."""
    n = len(A)
    M = [list(A[i]) + [iv.mpf(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = M[c][c]
        if not excludes_zero(piv):
            raise ArithmeticError("pivot contains 0")
        inv = 1 / piv
        M[c] = [v * inv for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


# ------------------------------------------------------------------ the slice system
class SliceSystem:
    def __init__(self, slot, x0_strings, gens=None):
        self.slot = slot
        self.gens = gens if gens is not None else generators_exact()
        self.Xiv = [[[ivq(v) for v in row] for row in X] for X in self.gens]
        self.x0 = [iv.mpf(s) for s in x0_strings]  # encloses the exact decimal
        self.k = [matvec(X, self.x0) for X in self.Xiv]
        self.Gk = [[GDIAG[a] * k[a] for a in range(NV)] for k in self.k]

    def F(self, z):
        x, lam, c = z[:NV], z[NV], z[NV + 1:]
        g = self.slot.grad(x)
        F1 = []
        for a in range(NV):
            v = g[a] - lam * GDIAG[a] * x[a]
            for i in range(4):
                v -= c[i] * self.Gk[i][a]
            F1.append(v)
        F2 = sum((GDIAG[a] * x[a] * x[a] for a in range(NV)), iv.mpf(0)) - 1
        F3 = [sum((self.Gk[i][a] * (x[a] - self.x0[a]) for a in range(NV)), iv.mpf(0)) for i in range(4)]
        return F1 + [F2] + F3

    def J(self, z):
        x, lam = z[:NV], z[NV]
        H = self.slot.hess(x)
        n = NV + 5
        Jm = [[iv.mpf(0)] * n for _ in range(n)]
        for a in range(NV):
            for b in range(NV):
                Jm[a][b] = H[a][b] - (lam * GDIAG[a] if a == b else 0)
            Jm[a][NV] = -GDIAG[a] * x[a]
            for i in range(4):
                Jm[a][NV + 1 + i] = -self.Gk[i][a]
        for b in range(NV):
            Jm[NV][b] = 2 * GDIAG[b] * x[b]
            for i in range(4):
                Jm[NV + 1 + i][b] = self.Gk[i][b]
        return Jm

    def krawczyk(self, Z, zt=None):
        """returns (passed, K, info).  Z: list of 19 intervals; zt: list of 19 mpf in Z."""
        n = len(Z)
        if zt is None:
            zt = [midf(I) for I in Z]
        zti = [iv.mpf(v) for v in zt]
        Fz = self.F(zti)
        Jmid = self.J(zti)
        Jm = mpm.matrix([[midf(v) for v in row] for row in Jmid])
        Y = Jm ** -1
        Yiv = [[iv.mpf(Y[i, j]) for j in range(n)] for i in range(n)]
        JZ = self.J(Z)
        YJ = matmul(Yiv, JZ)
        IminusYJ = [[(1 if i == j else 0) - YJ[i][j] for j in range(n)] for i in range(n)]
        d = [Z[i] - zti[i] for i in range(n)]
        YF = matvec(Yiv, Fz)
        corr = matvec(IminusYJ, d)
        K = [zti[i] - YF[i] + corr[i] for i in range(n)]
        inside = all(lo(K[i]) > lo(Z[i]) and hi(K[i]) < hi(Z[i]) for i in range(n))
        normIYJ = max(sum(max(abs(lo(v)), abs(hi(v))) for v in row) for row in IminusYJ)
        resid = max(max(abs(lo(v)), abs(hi(v))) for v in Fz)
        return inside, K, {"max|F(zt)|": resid, "||I - Y J(Z)||_inf": normIYJ,
                           "max width K": max(width(v) for v in K), "max width Z": max(width(v) for v in Z)}


def make_box(sys_, centre_x_strings, radius="1e-60"):
    """the box of Section 3: [t - r, t + r] over all 19 unknowns, t = (x_c, lam_c, 0),
    lam_c = 4 h(x_c)/|x_c|_G^2 from our own h (computed at DPS digits).  Built with outward
    rounding, so it contains the exact box; also returns the centre as mpf."""
    xc = [mpm.mpf(s) for s in centre_x_strings]
    hval = sys_.slot.h(xc, interval=False)
    nrm = mpm.fsum(GDIAG[a] * xc[a] ** 2 for a in range(NV))
    lam = 4 * hval / nrm
    r = iv.mpf(radius)
    rr = iv.mpf([-hi(r), hi(r)])
    Z = [iv.mpf(s) + rr for s in centre_x_strings] + [iv.mpf(lam) + rr] + [rr for _ in range(4)]
    centre = xc + [lam] + [mpm.mpf(0)] * 4
    return Z, centre, lam


def refine(sys_, Z, target_exp=-150, nsteps=6):
    """Repeat Krawczyk on Z' = K(Z) cap Z until the width is below 10^target_exp.  Every box
    returned has itself passed the Krawczyk test (K(Z') in int Z'), so it is a certified box
    that contains the unique zero of the original box."""
    hist = []
    for s in range(nsteps):
        ok, K, info = sys_.krawczyk(Z)
        hist.append((ok, info))
        if not ok:
            return Z, hist, False
        newZ = [iv.mpf([max(lo(a), lo(b)), min(hi(a), hi(b))]) for a, b in zip(K, Z)]
        if max(width(v) for v in newZ) < mpm.mpf(10) ** target_exp:
            # certify the final box itself
            ok2, K2, info2 = sys_.krawczyk(newZ)
            hist.append((ok2, info2))
            if ok2:
                return newZ, hist, True
            return Z, hist, True
        Z = newZ
    return Z, hist, True
