"""Audit item 8, maximum: my own attempt at an exact certificate
    lam*|u|^6 - N(u)*|u|^2 = <psi1, A1 psi1> + <psi2, A2 psi2>,  psi1 = u(x)u(x)u,  psi2 = u(x)u(x)Theta(u),
with A1, A2 SO(3)-equivariant PSD operators, parametrised per isotypic block by a real PSD matrix Y:
    A = sum_M (1/kappa_M) E_M Y E_M^dagger,   E_M = [J-^{L-M} h_1, ..., J-^{L-M} h_r].
Soundness needs only: (i) the polynomial identity, exactly; (ii) every Y PSD, exactly.
Coordinates b_m = c_m / sqrt(C(6,3+m)); tensor basis f_m = v_m/sqrt(C(6,3+m)) in which J- f_m = (3+m) f_{m-1},
J+ f_m = (3-m) f_{m+1} (rational).  Written by the auditor from the worklist definitions.
"""
import sys, time, itertools, json
from fractions import Fraction as Fr
import sympy as sp
import numpy as np
from scipy.optimize import minimize
from aud_core import MS, idx

T0 = time.time()
LAM = sp.Rational(463, 924) if len(sys.argv) < 2 else sp.Rational(sys.argv[1])
Cb = {m: sp.binomial(6, 3 + m) for m in MS}
b = sp.symbols('b3 b2 b1 b0 bm1 bm2 bm3')
B = sp.symbols('B3 B2 B1 B0 Bm1 Bm2 Bm3')
GENS = list(b) + list(B)
def P_(e): return sp.Poly(e, *GENS, domain='QQ')
def conjP(p):   # swap b <-> B (real coefficients)
    d = {}
    for mon, cf in p.terms():
        d[mon[7:] + mon[:7]] = cf
    return sp.Poly.from_dict(d, *GENS, domain='QQ')
# target
def rho_b(Q):
    s = 0
    for m1 in MS:
        m2 = Q - m1
        if abs(m2) <= 3:
            s += Cb[m1] * Cb[m2] * (-1) ** (3 - m2) * b[idx(m1)] * B[idx(-m2)]
    return P_(s)
N = sum((rho_b(Q) * conjP(rho_b(Q))) * sp.Rational(1, sp.binomial(12, 6 + Q)) for Q in range(-6, 7))
n2 = P_(sum(Cb[m] * b[idx(m)] * B[idx(m)] for m in MS))
F = n2 ** 3 * LAM - N * n2
print('target built: %d terms (%.0fs)' % (len(F.terms()), time.time() - T0)); sys.stdout.flush()

# ---------------------------------------------------------------- tensors in the f-basis
def jop(state, plus):
    out = {}
    for key, x in state.items():
        for pos in range(3):
            m = key[pos]
            if plus and m + 1 <= 3:
                k2 = key[:pos] + (m + 1,) + key[pos + 1:]; out[k2] = out.get(k2, 0) + x * (3 - m)
            if not plus and m - 1 >= -3:
                k2 = key[:pos] + (m - 1,) + key[pos + 1:]; out[k2] = out.get(k2, 0) + x * (3 + m)
    return {k: v for k, v in out.items() if v != 0}
def norm2(state):   # ||e||^2 in the v-basis:  f_m = t_m v_m, t_m^2 = 1/C_m
    tot = Fr(0)
    for key, x in state.items():
        w = Fr(1)
        for m in key: w /= int(Cb[m])
        tot += x * x * w
    return tot
def weight_basis(L, kind):
    out = []
    if kind == 1:
        for ms in itertools.combinations_with_replacement(MS, 3):
            if sum(ms) == L:
                out.append({p: Fr(1) for p in set(itertools.permutations(ms))})
    else:
        for ms in itertools.combinations_with_replacement(MS, 2):
            for m3 in MS:
                if sum(ms) + m3 == L:
                    out.append({p + (m3,): Fr(1) for p in set(itertools.permutations(ms))})
    return out
def hw_vectors(L, kind):
    Bs = weight_basis(L, kind)
    if not Bs: return []
    imgs = [jop(v, True) for v in Bs]
    keys = sorted({k for im in imgs for k in im})
    if not keys:
        ns = [sp.Matrix([1 if i == j else 0 for i in range(len(Bs))]) for j in range(len(Bs))]
    else:
        Mt = sp.Matrix([[sp.Rational(im.get(k, 0).numerator, im.get(k, 0).denominator) if k in im else 0 for im in imgs] for k in keys])
        ns = Mt.nullspace()
    hs = []
    for vec in ns:
        h = {}
        for cf, v in zip(vec, Bs):
            cfr = Fr(int(sp.numer(cf)), int(sp.denom(cf)))
            for k, x in v.items(): h[k] = h.get(k, 0) + cfr * x
        hs.append({k: x for k, x in h.items() if x != 0})
    return hs
def gpoly(state, kind):
    """<e, psi(u)> as a polynomial in (b, B)."""
    tot = 0
    for key, x in state.items():
        m1, m2, m3 = key
        t = b[idx(m1)] * b[idx(m2)]
        if kind == 1: t = t * b[idx(m3)]
        else: t = t * (-1) ** (3 - m3) * B[idx(-m3)]
        tot += sp.Rational(x.numerator, x.denominator) * t
    return P_(tot)

blocks = []
dimcount = {1: 0, 2: 0}
for kind in (1, 2):
    for L in range(0, 10):
        hs = hw_vectors(L, kind)
        if not hs: continue
        dimcount[kind] += len(hs) * (2 * L + 1)
        fams = []
        for h in hs:
            fam = [h]
            for _ in range(2 * L): fam.append(jop(fam[-1], False))
            assert not jop(fam[-1], False), 'lowering did not terminate'
            fams.append(fam)
        # kappa_M = ||J-^{L-M} h||^2 / ||h||^2, must not depend on h
        kap = []
        for j in range(2 * L + 1):
            ks = {norm2(f[j]) / norm2(f[0]) for f in fams}
            assert len(ks) == 1, 'kappa depends on h'
            kap.append(ks.pop())
        G = [[gpoly(f[j], kind) for j in range(2 * L + 1)] for f in fams]
        blocks.append(dict(kind=kind, L=L, r=len(hs), G=G, kap=kap))
print('dimension check Sym^3 V = 84:', dimcount[1] == 84, '  Sym^2 V (x) V = 196:', dimcount[2] == 196)
print('multiplicities:', [(bl['kind'], bl['L'], bl['r']) for bl in blocks], '(%.0fs)' % (time.time() - T0)); sys.stdout.flush()

# ---------------------------------------------------------------- facial reduction at cat = v3 + v-3 (b = c here)
catpt = {g: 0 for g in GENS}
for s_ in (b[idx(3)], b[idx(-3)], B[idx(3)], B[idx(-3)]): catpt[s_] = 1
print('F(cat) =', F.as_expr().subs(catpt))
params = []; polys = []
for bi, bl in enumerate(blocks):
    r = bl['r']; L = bl['L']
    K = sp.Matrix([[bl['G'][a][j].as_expr().subs(catpt) for a in range(r)] for j in range(2 * L + 1)])
    Pb = K.nullspace() if K.rank() < r else []
    if K.rank() == 0: Pb = [sp.Matrix([1 if i == j else 0 for i in range(r)]) for j in range(r)]
    bl['P'] = Pb
    q = len(Pb)
    if q == 0: continue
    # reduced coefficient polynomials  gt_{al,M} = sum_a P[al][a] g_{a,M}
    gt = [[sum((bl['G'][a][j] * Pb[al][a] for a in range(r) if Pb[al][a] != 0), P_(0)) for j in range(2 * L + 1)] for al in range(q)]
    cgt = [[conjP(p) for p in row] for row in gt]
    for al in range(q):
        for be in range(al, q):
            S = P_(0)
            for j in range(2 * L + 1):
                term = gt[al][j] * cgt[be][j]
                if al != be: term = term + gt[be][j] * cgt[al][j]
                S = S + term * sp.Rational(bl['kap'][j].denominator, bl['kap'][j].numerator)
            params.append((bi, al, be)); polys.append(S)
print('reduced block sizes:', [(bl['kind'], bl['L'], bl['r'], len(bl['P'])) for bl in blocks])
print('unknowns:', len(params), '(%.0fs)' % (time.time() - T0)); sys.stdout.flush()

# ---------------------------------------------------------------- exact linear system
monos = sorted(set(m for p in polys for m in p.monoms()) | set(F.monoms()))
dicts = [dict(p.terms()) for p in polys]; Fd = dict(F.terms())
Amat = sp.Matrix([[d.get(m, 0) for d in dicts] for m in monos])
rhs = sp.Matrix([Fd.get(m, 0) for m in monos])
print('linear system: %d monomials x %d unknowns (%.0fs)' % (len(monos), len(params), time.time() - T0)); sys.stdout.flush()
from sympy.polys.matrices import DomainMatrix
Aug = DomainMatrix.from_Matrix(Amat.row_join(rhs)).convert_to(sp.QQ)
R, piv = Aug.rref()
R = R.to_Matrix()
rank = len([p for p in piv if p < len(params)])
consistent = len(params) not in piv
print('rank %d, consistent: %s' % (rank, consistent))
if not consistent:
    print('NO certificate of this form at lambda =', LAM); sys.exit(0)
pivc = [p for p in piv if p < len(params)]
free = [i for i in range(len(params)) if i not in pivc]
def y_of(tv):
    y = [sp.Integer(0)] * len(params)
    for i, f in enumerate(free): y[f] = tv[i]
    for row, pc in enumerate(pivc):
        y[pc] = R[row, len(params)] - sum(R[row, f] * y[f] for f in free)
    return y
def mats(y):
    out = {}
    for (bi, al, be), v in zip(params, y):
        q = len(blocks[bi]['P'])
        out.setdefault(bi, sp.zeros(q, q))
        out[bi][al, be] = v; out[bi][be, al] = v
    return out
# numeric: maximise the smallest eigenvalue (each block scaled by its trace-free size) over free parameters
Y0 = {bi: np.array(M.evalf(), dtype=float) for bi, M in mats(y_of([0] * len(free))).items()}
Yi = []
for i in range(len(free)):
    e = [0] * len(free); e[i] = 1
    Yi.append({bi: np.array(M.evalf(), dtype=float) - Y0[bi] for bi, M in mats(y_of(e)).items()})
def minev(t):
    return min(np.linalg.eigvalsh(Y0[bi] + sum(t[i] * Yi[i][bi] for i in range(len(free)))).min() for bi in Y0)
best = None
rng = np.random.default_rng(1)
for trial in range(6):
    x0 = rng.normal(size=len(free)) * (0 if trial == 0 else 1)
    r1 = minimize(lambda t: -minev(t), x0, method='Nelder-Mead', options=dict(maxiter=40000, maxfev=40000, xatol=1e-10, fatol=1e-12))
    r2 = minimize(lambda t: -minev(t), r1.x, method='Powell', options=dict(maxiter=40000, xtol=1e-10, ftol=1e-12))
    if best is None or r2.fun < best.fun: best = r2
    print('  trial %d: min eigenvalue margin %.6f' % (trial, -r2.fun)); sys.stdout.flush()
print('best numeric margin:', -best.fun, '(%.0fs)' % (time.time() - T0))
if -best.fun <= 0:
    print('no strictly feasible point found; no certificate'); sys.exit(0)
tr = [sp.Rational(Fr(float(v)).limit_denominator(1000)) for v in best.x]
y = y_of(tr)
if 'perturb' in sys.argv[2:]:
    y[0] = y[0] + sp.Rational(1, 10 ** 6)
    print('PLANTED: first Gram entry perturbed by 1e-6')
Ms = mats(y)
allpd = all(M.is_positive_definite for M in Ms.values())
S = P_(0)
for v, p in zip(y, polys): S = S + p * v
D = F - S
print('EXACT: every reduced block positive definite:', allpd)
print('EXACT: identity lam|u|^6 - N|u|^2 - sum = 0 :', D.is_zero, ' (lambda = %s)' % LAM)
json.dump(dict(lam=str(LAM), blocks=[(bl['kind'], bl['L'], bl['r'], len(bl['P'])) for bl in blocks],
               Y={str((blocks[bi]['kind'], blocks[bi]['L'])): [[str(x) for x in M.row(i)] for i in range(M.rows)] for bi, M in Ms.items()},
               identity=bool(D.is_zero), pd=bool(allpd), margin=float(-best.fun)), open('out_a8max.json', 'w'), indent=1)
