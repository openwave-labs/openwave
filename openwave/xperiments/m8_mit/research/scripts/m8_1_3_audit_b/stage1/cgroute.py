"""Second route for the level norms of N(psi_y) and for R0: Clebsch-Gordan coupling of V_j^{(x)3}.
N_a(g) = s^{-3/2} (y (x) Theta y (x) y)^T D^{(x)3}(g) Z_a,  Z_a = sum_b eta_b (x) Theta eta_b (x) eta_a, and with the
coupled orthonormal real basis e_{J M k} (k = intermediate J12):
  ||Pi_{2J} N||^2 = s^{-3}/(2J+1) sum_{k,l} (A A^dag)_{kl} G_{kl},  A_{kM} = e_{JMk} . Y,
  G_{kl} = sum_{M'} g_{kM'}^T P conj(g_{lM'}),  g_{kM'}(m3) = sum_{m1 m2} e_{JM'k}(m1,m2,m3) T2(m1,m2),
  T2(m1, mu) = (Theta-twisted P) = (-1)^{j-mu} P(m1, -mu).
Works with any mpmath context (mp or iv): complex numbers are carried as (re, im) pairs."""
from fractions import Fraction as Fr
import functools
from core import mlist, Fm, k_to_complex
from reps import SLOTS, projector_mono, s_N
from hpoly import cg_exact


class CP:
    __slots__ = ('r', 'i')

    def __init__(s, r, i):
        s.r = r; s.i = i

    def __add__(s, o):
        return CP(s.r + o.r, s.i + o.i)

    def __sub__(s, o):
        return CP(s.r - o.r, s.i - o.i)

    def __mul__(s, o):
        if isinstance(o, CP):
            return CP(s.r * o.r - s.i * o.i, s.r * o.i + s.i * o.r)
        return CP(s.r * o, s.i * o)

    def conj(s):
        return CP(s.r, -s.i)


def P_ortho(name, ctx):
    d, n = SLOTS[name]
    ms = mlist(n)
    Pm = projector_mono(name)
    N = n + 1
    out = [[None] * N for _ in range(N)]
    for a in range(N):
        for b in range(N):
            re, im = k_to_complex(Pm[a][b], ctx)
            f = ctx.sqrt(ctx.mpf(Fm(n, ms[a])) / Fm(n, ms[b]))
            out[a][b] = CP(re * f, im * f)
    return out


@functools.lru_cache(None)
def coupled_basis(n):
    """list over (J, k=J12): dict M -> list of ((i1,i2,i3), sign, square) exact."""
    j = Fr(n, 2)
    ms = mlist(n)
    out = {}
    J12s = [Fr(t) for t in range(0, n + 1)]          # j(x)j -> 0..2j (integers)
    Jmax = 3 * j
    Js = [Jmax - t for t in range(int(Jmax) + 1 + (0 if n % 2 else 1)) if Jmax - t >= 0]
    for J in Js:
        for J12 in J12s:
            if not (abs(J12 - j) <= J <= J12 + j):
                continue
            blk = {}
            for M in [J - t for t in range(int(2 * J) + 1)]:
                lst = []
                for i1, m1 in enumerate(ms):
                    for i2, m2 in enumerate(ms):
                        m3 = M - m1 - m2
                        if m3 not in ms:
                            continue
                        s1, q1 = cg_exact(j, m1, j, m2, J12, m1 + m2)
                        if s1 == 0:
                            continue
                        s2, q2 = cg_exact(J12, m1 + m2, j, m3, J, M)
                        if s2 == 0:
                            continue
                        lst.append(((i1, i2, ms.index(m3)), s1 * s2, q1 * q2))
                blk[M] = lst
            out[(J, J12)] = blk
    return out


class CGRoute:
    def __init__(self, name, ctx):
        self.name = name; self.ctx = ctx
        d, n = SLOTS[name]
        self.n = n; self.N = n + 1
        self.j = Fr(n, 2)
        self.ms = mlist(n)
        self.P = P_ortho(name, ctx)
        cb = coupled_basis(n)
        self.cb = {}
        for key, blk in cb.items():
            self.cb[key] = {M: [(idx, ctx.sqrt(ctx.mpf(q.numerator) / q.denominator) * sg) for idx, sg, q in lst]
                            for M, lst in blk.items()}
        self.Js = sorted(set(J for J, k in cb))
        N = self.N; ms = self.ms; j = self.j
        z = CP(ctx.mpf(0), ctx.mpf(0))
        self.T2 = [[None] * N for _ in range(N)]
        for a in range(N):
            for mu in range(N):
                sgn = (-1) ** int(j - ms[mu])
                self.T2[a][mu] = self.P[a][ms.index(-ms[mu])] * sgn
        # G matrices per J
        self.G = {}
        for J in self.Js:
            ks = [k for (JJ, k) in self.cb if JJ == J]
            gvec = {}
            for k in ks:
                for M, lst in self.cb[(J, k)].items():
                    v = [z] * N
                    for (i1, i2, i3), c in lst:
                        v[i3] = v[i3] + self.T2[i1][i2] * c
                    gvec[(k, M)] = v
            G = {}
            Pg = {}
            for l in ks:
                for M in self.cb[(J, l)]:
                    gl = gvec[(l, M)]
                    Pg[(l, M)] = [sum((self.P[a][b] * gl[b].conj() for b in range(N)), z) for a in range(N)]
            for k in ks:
                for l in ks:
                    acc = z
                    for M in self.cb[(J, k)]:
                        gk = gvec[(k, M)]; pl = Pg[(l, M)]
                        for a in range(N):
                            acc = acc + gk[a] * pl[a]
                    G[(k, l)] = acc
            self.G[J] = (ks, G)
        s = s_N(name)
        self.s3 = (ctx.mpf(s.numerator) / s.denominator) ** 3

    def theta(self, y):
        j = self.j; ms = self.ms
        return [y[self.ms.index(-m)].conj() * ((-1) ** int(j - m)) for m in ms]

    def Aarr(self, y1, y2, y3):
        """A_{J,k,M} for Y = y1 (x) Theta y2 (x) y3"""
        t2 = self.theta(y2)
        z = CP(self.ctx.mpf(0), self.ctx.mpf(0))
        A = {}
        for (J, k), blk in self.cb.items():
            for M, lst in blk.items():
                acc = z
                for (i1, i2, i3), c in lst:
                    acc = acc + y1[i1] * t2[i2] * y3[i3] * c
                A[(J, k, M)] = acc
        return A

    def form(self, A1, A2, J):
        """(1/(2J+1)) sum_{kl} (sum_M A1_{kM} conj A2_{lM}) G_{kl}  (complex)"""
        ks, G = self.G[J]
        z = CP(self.ctx.mpf(0), self.ctx.mpf(0))
        acc = z
        Ms = [J - t for t in range(int(2 * J) + 1)]
        for k in ks:
            for l in ks:
                t = z
                for M in Ms:
                    t = t + A1[(J, k, M)] * A2[(J, l, M)].conj()
                acc = acc + t * G[(k, l)]
        return acc * (1 / (2 * self.ctx.mpf(J.numerator) / J.denominator + 1))

    def levels(self, y):
        A = self.Aarr(y, y, y)
        return {int(2 * J): self.form(A, A, J).r / self.s3 for J in self.Js}

    def R0(self, y):
        lv = self.levels(y)
        n = self.n
        acc = self.ctx.mpf(0)
        for L, v in lv.items():
            if L > n:
                acc = acc - v / (L * (L + 2) - n * (n + 2))
        return acc

    def R0_grad_hess(self, u, basis):
        """gradient and Hessian of R0 at u along the given real directions (list of complex vectors)."""
        ctx = self.ctx; n = self.n
        wts = {J: ctx.mpf(1) / (int(2 * J) * (int(2 * J) + 2) - n * (n + 2)) for J in self.Js if int(2 * J) > n}

        def add(A, B):
            return {k: A[k] + B[k] for k in A}
        N0 = self.Aarr(u, u, u)
        N1 = [add(add(self.Aarr(v, u, u), self.Aarr(u, v, u)), self.Aarr(u, u, v)) for v in basis]
        g = []
        for a in range(len(basis)):
            acc = ctx.mpf(0)
            for J, w in wts.items():
                acc = acc - w * 2 * self.form(N0, N1[a], J).r
            g.append(acc / self.s3)
        H = [[None] * len(basis) for _ in basis]
        for a in range(len(basis)):
            for b in range(a, len(basis)):
                va, vb = basis[a], basis[b]
                Nst = self.Aarr(va, vb, u)
                for t in (self.Aarr(vb, va, u), self.Aarr(va, u, vb), self.Aarr(vb, u, va), self.Aarr(u, va, vb), self.Aarr(u, vb, va)):
                    Nst = add(Nst, t)
                acc = ctx.mpf(0)
                for J, w in wts.items():
                    acc = acc - w * 2 * (self.form(N1[a], N1[b], J).r + self.form(N0, Nst, J).r)
                H[a][b] = H[b][a] = acc / self.s3
        return g, H
