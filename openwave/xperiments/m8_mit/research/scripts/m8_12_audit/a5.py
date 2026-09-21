"""Audit item 5: exact characteristic polynomials of H_u on N_u at the ten orbits, plus a 50-digit route.

Exact route: rational 'weighted' coordinates a_m = sqrt(C(6,3+m)) c_m (a real-linear change of variables;
both the Hessian form and the metric Re<,> are transported, so det(lam*G - H)/det G is unchanged).
The polynomial identity N_a(a(c)) = N(c) is checked exactly first.
Numerical route: c-coordinates, mpmath 50 digits, eigenvalues of the form in an orthonormal basis of N_u.
"""
import sys, pickle, time
import sympy as sp, mpmath as mp
from sympy.polys.matrices import DomainMatrix
from aud_core import *

T0 = time.time()
P = sp.symbols('p3 p2 p1 p0 pm1 pm2 pm3', real=True)
Qs = sp.symbols('q3 q2 q1 q0 qm1 qm2 qm3', real=True)
A = [P[k] + I * Qs[k] for k in range(7)]
Cb = {m: sp.binomial(6, 3 + m) for m in MS}
def Na_poly():
    tot = 0
    for Q in range(-6, 7):
        s = 0
        for m1 in MS:
            m2 = Q - m1
            if abs(m2) <= 3:
                s += (-1) ** (3 - m2) * A[idx(m1)] * sp.conjugate(A[idx(-m2)])
        s = sp.expand(s)
        tot += sp.expand(s * sp.conjugate(s)) / sp.binomial(12, 6 + Q)
    return sp.expand(tot)
Na = Na_poly()
VARS = list(P) + list(Qs)
# identity check against my CG route: N(c) with c_m = a_m / sqrt(C_m)
cvec = [A[idx(m)] / sp.sqrt(Cb[m]) for m in MS]
diffpoly = sp.expand(N_exact(cvec) - Na)
print('identity N_a(a) - N(c(a)) == 0 :', diffpoly == 0, ' (%.0fs)' % (time.time() - T0)); sys.stdout.flush()
G0 = sp.diag(*([1 / Cb[m] for m in MS] * 2))
HessNa = sp.hessian(Na, VARS)

# J in a-coordinates (derived: J+ a_m -> (3-m) a_{m+1}, J- a_m -> (3+m) a_{m-1})
def Jp_a(a):
    out = [0] * 7
    for m in MS:
        if m + 1 <= 3: out[idx(m + 1)] += (3 - m) * a[idx(m)]
    return out
def Jm_a(a):
    out = [0] * 7
    for m in MS:
        if m - 1 >= -3: out[idx(m - 1)] += (3 + m) * a[idx(m)]
    return out
def Ogens_a(a):
    jp, jm = Jp_a(a), Jm_a(a)
    jx = [(p + q) / 2 for p, q in zip(jp, jm)]
    jy = [(p - q) / (2 * I) for p, q in zip(jp, jm)]
    jz = [m * a[idx(m)] for m in MS]
    return [[I * z for z in a], [-I * z for z in jx], [-I * z for z in jy], [-I * z for z in jz]]
# check the a-coordinate J against the c-coordinate J (exact, symbolic vector)
cc = sp.symbols('z0:7')
chk = True
for fa, fc in ((Jp_a, Jp_vec), (Jm_a, Jm_vec)):
    lhs = fa([cc[idx(m)] * sp.sqrt(Cb[m]) for m in MS])
    rhs = [z * sp.sqrt(Cb[m]) for z, m in zip(fc(list(cc)), MS)]
    chk &= all(sp.expand(l - r) == 0 for l, r in zip(lhs, rhs))
print('a-coordinate J+- agree with c-coordinate J+- :', chk)

def realv(a): return sp.Matrix([sp.re(sp.expand(z)) for z in a] + [sp.im(sp.expand(z)) for z in a])

def c_to_a(c): return [sp.sqrt(Cb[m]) * c[idx(m)] for m in MS]

r2, r3, r5 = sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)
def vc(d):
    c = [sp.Integer(0)] * 7
    for m, z in d.items(): c[idx(m)] = sp.sympify(z)
    return c
ORBITS = {   # unnormalised c-representatives
    'v3': vc({3: 1}), 'v2': vc({2: 1}), 'v1': vc({1: 1}), 'v0': vc({0: 1}),
    'xyz': vc({2: 1, -2: -1}), 'cat': vc({3: 1, -3: 1}),
    'A*': vc({1: 2, -2: 1}),
    'D*': vc({2: sp.sqrt(13), -3: 2 * r3}),
    'F*': vc({0: sp.sqrt(6), 2: I * r5, -2: I * r5}),
    'G*': vc({0: sp.sqrt(23), 3: I * sp.sqrt(10), -3: -I * sp.sqrt(10)}),
}
lam = sp.Symbol('lam')
results = {}
for name, c in ORBITS.items():
    a = c_to_a(c)
    # remove a common radical so that the coordinates live in a small field (projective point unchanged)
    nz = [z for z in a if z != 0]
    a = [sp.radsimp(sp.expand(z / nz[-1])) for z in a]
    x = realv(a)
    n2 = sp.expand((x.T * G0 * x)[0])
    sub = dict(zip(VARS, list(x)))
    Nval = sp.expand(Na.subs(sub))
    Hs = HessNa.subs(sub).applyfunc(sp.expand)
    Mf = (Hs / n2 - 4 * Nval / n2 ** 2 * G0).applyfunc(sp.expand)
    O = [realv(o) for o in Ogens_a(a)]
    # annihilation of O_u (as full vectors of the form matrix)
    ann = [max([abs(sp.N(v, 30)) for v in (Mf * o).applyfunc(sp.expand)]) for o in O]
    annex = all(all(sp.expand(v) == 0 for v in (Mf * o)) for o in O)
    rows = sp.Matrix.hstack(G0 * x, *[G0 * o for o in O]).T
    dimO = sp.Matrix.hstack(*O).rank(simplify=True)
    Bn = sp.Matrix.hstack(*rows.nullspace(simplify=True))
    Bn = Bn.applyfunc(sp.radsimp)
    dimN = Bn.shape[1]
    Gm = (Bn.T * G0 * Bn).applyfunc(sp.expand)
    Hm = (Bn.T * Mf * Bn).applyfunc(sp.expand)
    ext = sorted({s for s in (list(Gm) + list(Hm)) for s in sp.sympify(s).atoms(sp.Pow) if s.exp == sp.Rational(1, 2)}, key=str)
    K = sp.QQ.algebraic_field(*ext) if ext else sp.QQ
    Kx = K[lam] if False else None
    dG = DomainMatrix.from_Matrix(Gm).convert_to(K)
    dH = DomainMatrix.from_Matrix(Hm).convert_to(K)
    # charpoly of G^{-1} H over K  ==  det(lam G - H)/det G
    GinvH = dG.inv() * dH
    cp = GinvH.charpoly()
    cpoly = sp.Poly([K.to_sympy(v) for v in cp], lam)
    cpexpr = sp.expand(cpoly.as_expr())
    fac = sp.factor_list(cpexpr)
    # signature: count real roots with sign via exact root isolation on each rational factor
    nneg = nzero = npos = 0
    for f_, mult in fac[1]:
        fp = sp.Poly(f_, lam)
        if not all(cf.is_Rational for cf in fp.all_coeffs()):
            raise RuntimeError('non-rational charpoly factor at ' + name)
        for r in fp.intervals():
            (lo, hi), k = r
            if lo == hi == 0: nzero += k * mult
            elif hi <= 0: nneg += k * mult
            elif lo >= 0: npos += k * mult
            else:
                # refine until sign is decided
                rr = fp.refine_root(lo, hi, eps=sp.Rational(1, 10 ** 12))
                if rr[1] < 0: nneg += k * mult
                elif rr[0] > 0: npos += k * mult
                else: nzero += k * mult
        nreal = sum(k for _, k in fp.intervals())
        assert nreal == fp.degree(), ('non-real root', name)
    results[name] = dict(dimO=dimO, dimN=dimN, sig=(nneg, nzero, npos), charpoly=str(sp.factor(cpexpr)),
                         annih_exact=annex, rhat=str(sp.simplify(Nval / n2 ** 2)), field=[str(e) for e in ext])
    print('%-4s rhat=%s dimO=%d dimN=%d sig=%s O-annihilated exactly=%s  field=%s (%.0fs)' % (
        name, results[name]['rhat'], dimO, dimN, (nneg, nzero, npos), annex, ext, time.time() - T0))
    print('     charpoly =', sp.factor(cpexpr)); sys.stdout.flush()
    # planted defect: drop the -4N term -> annihilation must fail
    Mbad = Hs / n2
    bad = any(any(sp.expand(v) != 0 for v in (Mbad * o)) for o in O[:1])
    results[name]['planted_no4N_fires'] = bad
pickle.dump(results, open('out_a5.pkl', 'wb'))
print('planted defect (no -4N): annihilation check fails at', sum(r['planted_no4N_fires'] for r in results.values()), 'of', len(results))
