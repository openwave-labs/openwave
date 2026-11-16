"""Exact local data of h at a point: criticality, M-hat, S-hat, k, r_u, charpolys (chart x)."""
import sympy as sp
from sympy import QQ
from sympy.polys.matrices import DomainMatrix
from quartic import *

JC = sp.zeros(14)
for a in range(7):
    JC[a, a + 7] = -1     # new Re = -Im
    JC[a + 7, a] = 1      # new Im = Re
GM = sp.diag(*GDIAG)
GMI = sp.diag(*[sp.Rational(1, g) for g in GDIAG])

def simp(e):
    return sp.radsimp(sp.expand(e))

def field_of(entries):
    gens = set()
    for e in entries:
        for p in sp.preorder_traversal(sp.sympify(e)):
            if isinstance(p, sp.Pow) and p.exp == sp.Rational(1, 2) and p.base.is_Integer:
                gens.add(p)
    return sorted(gens, key=lambda t: t.base)

def charpoly_exact(M, var):
    ent = list(M)
    gens = field_of(ent)
    K = QQ.algebraic_field(*gens) if gens else QQ
    dM = DomainMatrix.from_Matrix(M).convert_to(K)
    cs = dM.charpoly()
    poly = sum(K.to_sympy(c) * var ** (len(cs) - 1 - i) for i, c in enumerate(cs))
    return sp.expand(poly), gens

class PointData:
    def __init__(self, hpoly, w, gens_mats):
        self.w = [sp.sympify(t) for t in w]
        self.x = x_of_w(self.w)
        x = self.x
        self.n2 = simp(sum(GDIAG[a] * x[a] ** 2 for a in range(14)))          # ||w||^2
        self.hx = simp(to_sym(hpoly, x))
        self.hu = simp(self.hx / self.n2 ** 2)                                 # h(u) = Q(u)
        g = [simp(to_sym(p, x)) for p in grad(hpoly)]
        lam = 4 * self.hx / self.n2
        self.crit_resid = [simp(g[a] - lam * GDIAG[a] * x[a]) for a in range(14)]
        self.critical = all(r == 0 for r in self.crit_resid)
        Hp = hess_cached(hpoly)
        H = sp.Matrix(14, 14, lambda a, b: simp(to_sym(Hp[a][b], x)))
        # M-hat as a G-self-adjoint operator in the chart, at the unit vector u = x/|x|_G
        self.Mop = (GMI * H / self.n2 - 4 * self.hu * sp.eye(14)).applyfunc(simp)
        self.Sop = (JC * self.Mop).applyfunc(simp)
        xv = sp.Matrix(x)
        self.K = sp.Matrix.hstack(*[(Xg * xv).applyfunc(simp) for Xg in gens_mats])
        self.k = self.K.rank(simplify=True)
        self.MK_zero = (self.Mop * self.K).applyfunc(simp) == sp.zeros(14, len(gens_mats))
        sig = (self.K.T * GM * JC * self.K).applyfunc(simp)
        self.sigma_gram = sig
        self.r = self.k - sig.rank(simplify=True)
        # u is an eigenvector of M-hat with eigenvalue 8 h(u) at a critical point
        self.Mu = (self.Mop * xv - 8 * self.hu * xv).applyfunc(simp)

def roots_summary(poly_nu, nu):
    """factor over Q; for each factor: degree, multiplicity, #neg real, #pos real, #nonreal, #zero."""
    P = sp.Poly(poly_nu, nu)
    c, facs = sp.factor_list(P)
    out = []
    for f, e in facs:
        f = sp.Poly(f, nu)
        nneg = f.count_roots(-sp.oo, 0) - (1 if f.eval(0) == 0 else 0)
        npos = f.count_roots(0, sp.oo) - (1 if f.eval(0) == 0 else 0)
        nz = 1 if f.eval(0) == 0 else 0
        out.append({"factor": f.as_expr(), "mult": e, "deg": f.degree(), "neg": nneg, "pos": npos,
                    "zero": nz, "nonreal": f.degree() - f.count_roots()})
    return out

def words(rs):
    neg = sum(r["neg"] * r["mult"] for r in rs)
    pos = sum(r["pos"] * r["mult"] for r in rs)
    nonreal = sum(r["nonreal"] * r["mult"] for r in rs)
    zero = sum(r["zero"] * r["mult"] for r in rs)
    if zero:
        return "slow polynomial has a zero root (should not happen by construction)"
    if pos == 0 and nonreal == 0:
        mults = []
        for r in rs:
            mults += [r["mult"]] * r["neg"]
        if all(m == 1 for m in mults): return "elliptic, simple"
        if sorted(mults)[-1] == 2 and sum(1 for m in mults if m == 2) == 1 and all(m <= 2 for m in mults):
            return "elliptic, one double frequency"
        return "elliptic (multiplicities %s: neither 'simple' nor 'one double frequency')" % mults
    w = ["hyperbolic"]
    if nonreal == 0:
        posm = []
        for r in rs: posm += [r["mult"]] * r["pos"]
        if len(posm) == 1 and posm[0] == 1: w.append("one real pair")
        elif sum(posm) >= 2: w.append("real pairs")
    else:
        # exactly one conjugate pair, simple, and no positive root
        nr = [(r["nonreal"], r["mult"]) for r in rs if r["nonreal"]]
        if pos == 0 and len(nr) == 1 and nr[0] == (2, 1): w.append("a complex quartet")
        else: w.append("(no listed sub-word fits: pos=%d, nonreal=%s)" % (pos, nr))
    return ", ".join(w)

_HESS_CACHE = []
def hess_cached(hpoly):
    for hp, H in _HESS_CACHE:
        if hp == hpoly: return H
    H = hess(hpoly); _HESS_CACHE.append((hpoly, H)); return H
