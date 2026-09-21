"""Exact sextic Hermitian-sum-of-squares certificate machinery (item 8, maximum).

Coordinates: a_m = sqrt(C(6,3+m)) c_m (hess.py).  Polynomials in the 14 independent symbols
(a_3..a_-3, abar_3..abar_-3) are dicts  exponent-tuple -> Fraction.  Real-coefficient polynomials
in (a, abar) have  conj(P)(a) = swap(P)(a)  where swap exchanges a and abar.

Soundness does not depend on how the certificate was found: if
    F(a) = sum_b sum_M (1/kappa_{b,M}) * c_{b,M}(a)^T  Yt_b  conj(c_{b,M}(a))
holds as an identity of polynomials and every Yt_b is (real symmetric) positive semidefinite,
then F >= 0 everywhere, because each term is a Hermitian form of a PSD matrix.
"""
from fractions import Fraction as Fr
import itertools
import common as C

MS = C.MS
idx = C.idx
B6 = {m: Fr(C.sp.binomial(6, 3 + m)) for m in MS}
NV = 14   # a_i at position i (i = idx(m)), abar_i at 7 + i

# ------------------------------------------------------------------ polynomial dict arithmetic
def padd(P, Q, s=1):
    R = dict(P)
    for k, v in Q.items():
        R[k] = R.get(k, 0) + s * v
        if R[k] == 0: del R[k]
    return R

def pscale(P, s):
    return {k: v * s for k, v in P.items()} if s != 0 else {}

def pmul(P, Q):
    R = {}
    for k1, v1 in P.items():
        for k2, v2 in Q.items():
            k = tuple(x + y for x, y in zip(k1, k2))
            R[k] = R.get(k, 0) + v1 * v2
    return {k: v for k, v in R.items() if v != 0}

def swap(P):
    return {k[7:] + k[:7]: v for k, v in P.items()}

def mono(i, conj=False):
    e = [0] * NV; e[(7 if conj else 0) + i] = 1
    return {tuple(e): Fr(1)}

def evalc(P, a):
    """evaluate at complex a (list of 7 python/mpmath numbers)."""
    ab = [x.conjugate() for x in a]
    tot = 0
    for k, v in P.items():
        t = float(v) if not hasattr(a[0], 'ae') else v.numerator / C.mp.mpf(v.denominator)
        for i in range(7):
            if k[i]: t = t * a[i] ** k[i]
            if k[7 + i]: t = t * ab[i] ** k[7 + i]
        tot += t
    return tot

# ------------------------------------------------------------------ target polynomial
def norm2_poly():
    P = {}
    for m in MS:
        P = padd(P, pscale(pmul(mono(idx(m)), mono(idx(m), True)), 1 / B6[m]))
    return P

def Nt_poly():
    """Nt(a) = sum_Q |sum_m1 (-1)^(3-Q+m1) a_m1 abar_{m1-Q}|^2 / C(12,6+Q)."""
    tot = {}
    for Q in range(-6, 7):
        S = {}
        for m1 in MS:
            k = m1 - Q
            if abs(k) <= 3:
                S = padd(S, pscale(pmul(mono(idx(m1)), mono(idx(k), True)), (-1) ** (3 - (Q - m1))))
        tot = padd(tot, pscale(pmul(S, swap(S)), Fr(1, int(C.sp.binomial(12, 6 + Q)))))
    return tot

# ------------------------------------------------------------------ tensor-space representation theory (exact)
# J+, J- in a-coordinates: J+ a_m -> (3-m) a_{m+1},  J- a_m -> (3+m) a_{m-1}; metric w_m = 1/C(6,3+m)
def jplus_vec(v, r):
    out = {}
    for key, x in v.items():
        for pos in range(r):
            m = key[pos]
            if m + 1 <= 3:
                k2 = key[:pos] + (m + 1,) + key[pos + 1:]
                out[k2] = out.get(k2, 0) + x * (3 - m)
    return {k: x for k, x in out.items() if x != 0}

def jminus_vec(v, r):
    out = {}
    for key, x in v.items():
        for pos in range(r):
            m = key[pos]
            if m - 1 >= -3:
                k2 = key[:pos] + (m - 1,) + key[pos + 1:]
                out[k2] = out.get(k2, 0) + x * (3 + m)
    return {k: x for k, x in out.items() if x != 0}

def metric(key):
    w = Fr(1)
    for m in key: w /= B6[m]
    return w

def ip(u, v):
    return sum(x * v.get(k, 0) * metric(k) for k, x in u.items())

def sym_basis(L, kind):
    """basis of the weight-L subspace of Sym^3 V (kind 1) or Sym^2 V (x) V (kind 2), as dicts."""
    out = []
    if kind == 1:
        for ms in itertools.combinations_with_replacement(MS, 3):
            if sum(ms) == L:
                v = {}
                for p in set(itertools.permutations(ms)): v[p] = Fr(1)
                out.append(v)
    else:
        for ms in itertools.combinations_with_replacement(MS, 2):
            for m3 in MS:
                if sum(ms) + m3 == L:
                    v = {}
                    for p in set(itertools.permutations(ms)): v[p + (m3,)] = Fr(1)
                    out.append(v)
    return out

def nullspace_fr(rows, n):
    """exact nullspace of a list of rows (lists of Fractions, length n)."""
    M = [list(r) for r in rows]
    piv = []; r = 0
    for c in range(n):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]
        pv = M[r][c]; M[r] = [x / pv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]; M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        piv.append(c); r += 1
        if r == len(M): break
    free = [c for c in range(n) if c not in piv]
    basis = []
    for fc in free:
        v = [Fr(0)] * n; v[fc] = Fr(1)
        for i, pc in enumerate(piv): v[pc] = -M[i][fc]
        basis.append(v)
    return basis

def highest_weight_families(kind):
    """{L: [family]} with family = [e_L, e_{L-1}, ..., e_{-L}] (e_M = J-^{L-M} h), exact."""
    out = {}
    for L in range(0, 10):
        B = sym_basis(L, kind)
        if not B: continue
        # J+ of a general combination must vanish
        imgs = [jplus_vec(b, 3) for b in B]
        keys = sorted({k for im in imgs for k in im})
        rows = [[im.get(k, Fr(0)) for im in imgs] for k in keys]
        ns = nullspace_fr(rows, len(B)) if rows else [[Fr(int(i == j)) for i in range(len(B))] for j in range(len(B))]
        fams = []
        for coeffs in ns:
            h = {}
            for cf, b in zip(coeffs, B):
                for k, x in b.items(): h[k] = h.get(k, 0) + cf * x
            h = {k: x for k, x in h.items() if x != 0}
            fam = [h]
            for M in range(L, -L, -1): fam.append(jminus_vec(fam[-1], 3))
            fams.append(fam)
        if fams: out[L] = fams
    return out

def kappa(L, M):
    """||J-^{L-M} h||^2 / ||h||^2 for a highest-weight h of spin L."""
    k = Fr(1)
    for i in range(L - M):
        Mi = L - i
        k *= L * (L + 1) - Mi * (Mi - 1)
    return k

def coeff_poly(e, kind):
    """c(a) = <e, psi>, psi = a(x)a(x)a (kind 1) or a(x)a(x)theta(a) (kind 2), theta(a)_m = (-1)^(3-m) abar_{-m}."""
    P = {}
    for key, x in e.items():
        w = x * metric(key)
        t = pmul(mono(idx(key[0])), mono(idx(key[1])))
        if kind == 1:
            t = pmul(t, mono(idx(key[2])))
        else:
            m3 = key[2]
            t = pscale(pmul(t, mono(idx(-m3), True)), (-1) ** (3 - m3))
        P = padd(P, pscale(t, w))
    return P


# ------------------------------------------------------------------ the certificate pipeline
from fractions import Fraction as Fr
import numpy as np
def certificate(LAM, perturb=None, log=print):
    """Build and exactly verify the sextic certificate for lambda = LAM.
    perturb: optional callable applied to the rational solution vector y before verification (defect demos).
    Returns a dict with consistency, positivity, identity, data."""
    import time, numpy as np
    T0 = time.time()
    chk = {}
    fams = {1: highest_weight_families(1), 2: highest_weight_families(2)}
    mults = {k: {L: len(f) for L, f in v.items()} for k, v in fams.items()}
    log('multiplicities Sym^3 V:', mults[1], ' Sym^2 V (x) V:', mults[2])
    chk['dimension count Sym^3 V = 84'] = (sum(m * (2 * L + 1) for L, m in mults[1].items()) == 84)
    chk['dimension count Sym^2 V (x) V = 196'] = (sum(m * (2 * L + 1) for L, m in mults[2].items()) == 196)
    # highest-weight property checked exactly
    ok_hw = all(not jplus_vec(f[0], 3) and all(sum(k) == L for k in f[0]) for kind in (1, 2) for L, fl in fams[kind].items() for f in fl)
    chk['all highest-weight vectors satisfy J+ h = 0 and Jz h = L h exactly'] = ok_hw
    ok_low = all(not jminus_vec(f[-1], 3) for kind in (1, 2) for L, fl in fams[kind].items() for f in fl)
    chk['lowering terminates: J- e_{-L} = 0 for every family'] = ok_low
    cat = {C.idx(3): Fr(1), C.idx(-3): Fr(1)}     # a-coordinates of v3 + v-3 (real)
    def eval_real(Pd, vals):
        tot = Fr(0)
        for k, v in Pd.items():
            t = v
            for i in range(14):
                if k[i]:
                    t *= vals.get(i % 7, Fr(0)) ** k[i]
                    if t == 0: break
            tot += t
        return tot
    blocks = []
    for kind in (1, 2):
        for L, fl in sorted(fams[kind].items()):
            Cs = [[coeff_poly(f[L - M], kind) for M in range(L, -L - 1, -1)] for f in fl]   # [a][M]
            # forced kernel from the cat orbit: vectors (C_{a,M}(cat))_a, M = -L..L
            Kvecs = [[eval_real(Cs[a][j], cat) for a in range(len(fl))] for j in range(2 * L + 1)]
            Pb = nullspace_fr(Kvecs, len(fl))        # basis of the orthogonal complement (columns)
            blocks.append(dict(kind=kind, L=L, mult=len(fl), Cs=Cs, P=Pb, kdim=len(fl) - len(Pb)))
    log('reduced block sizes:', [(b['kind'], b['L'], b['mult'], len(b['P'])) for b in blocks])
    # reduced coefficient polynomials and Gram polynomials
    params = []     # (block index, alpha, beta)
    coefpolys = []
    for bi, b in enumerate(blocks):
        r = len(b['P'])
        if r == 0: continue
        L = b['L']
        ct = [[{} for _ in range(2 * L + 1)] for _ in range(r)]
        for al in range(r):
            for j in range(2 * L + 1):
                Pd = {}
                for a in range(b['mult']):
                    if b['P'][al][a] != 0: Pd = padd(Pd, pscale(b['Cs'][a][j], b['P'][al][a]))
                ct[al][j] = Pd
        Ht = {}
        for al in range(r):
            for be in range(r):
                S = {}
                for j in range(2 * L + 1):
                    M = L - j
                    S = padd(S, pscale(pmul(ct[al][j], swap(ct[be][j])), 1 / kappa(L, M)))
                Ht[(al, be)] = S
        for al in range(r):
            for be in range(al, r):
                params.append((bi, al, be))
                coefpolys.append(Ht[(al, al)] if al == be else padd(Ht[(al, be)], Ht[(be, al)]))
    log('unknowns:', len(params), ' (%.0fs)' % (time.time() - T0))
    n2p = norm2_poly(); Ntp = Nt_poly()
    F = padd(pscale(pmul(pmul(n2p, n2p), n2p), LAM), pmul(Ntp, n2p), -1)
    monos = sorted(set(F) | {k for cp in coefpolys for k in cp})
    # exact linear system  sum_k y_k coefpoly_k = F  (monomial by monomial), echelon form streamed
    nun = len(params)
    ech = []     # list of (pivot col, row) with row = list of Fr of length nun+1 (last = rhs)
    def reduce_row(row):
        for pc, prow in ech:
            if row[pc] != 0:
                f = row[pc]; row = [x - f * y for x, y in zip(row, prow)]
        return row
    inconsistent = False
    for mo in monos:
        row = [cp.get(mo, Fr(0)) for cp in coefpolys] + [F.get(mo, Fr(0))]
        row = reduce_row(row)
        pc = next((i for i in range(nun) if row[i] != 0), None)
        if pc is None:
            if row[-1] != 0: inconsistent = True
            continue
        pv = row[pc]; row = [x / pv for x in row]
        ech = [(c_, [x - r_[pc] * y for x, y in zip(r_, row)]) for c_, r_ in ech]
        ech.append((pc, row))
    chk['exact linear system (identity of sextic polynomials) is consistent at lambda = %s' % LAM] = not inconsistent
    if inconsistent:
        return dict(chk=chk, consistent=False)
    rank = len(ech)
    log('equations: %d monomials, rank %d, free %d  (%.0fs)' % (len(monos), rank, nun - rank, time.time() - T0))
    pivots = [pc for pc, _ in ech]
    free = [i for i in range(nun) if i not in pivots]
    def y_from_t(t):
        y = [Fr(0)] * nun
        for i, fi in enumerate(free): y[fi] = t[i]
        for pc, row in ech:
            y[pc] = row[-1] - sum(row[fi] * y[fi] for fi in free)
        return y
    def Ymats(y, as_float=False):
        mats = {}
        for (bi, al, be), v in zip(params, y):
            r = len(blocks[bi]['P'])
            if bi not in mats: mats[bi] = [[Fr(0)] * r for _ in range(r)]
            mats[bi][al][be] = v; mats[bi][be][al] = v
        if as_float: return {bi: np.array([[float(x) for x in row] for row in M]) for bi, M in mats.items()}
        return mats
    # numeric: maximize the minimum eigenvalue over the affine family (barrier Newton in t)
    nf = len(free)
    Y0 = Ymats(y_from_t([Fr(0)] * nf), True)
    Yi = []
    for i in range(nf):
        e = [Fr(0)] * nf; e[i] = Fr(1)
        Ye = Ymats(y_from_t(e), True)
        Yi.append({bi: Ye[bi] - Y0[bi] for bi in Y0})
    bis = sorted(Y0)
    def Ys(t): return {bi: Y0[bi] + sum(t[i] * Yi[i][bi] for i in range(nf)) for bi in bis}
    x = np.zeros(nf + 1); x[nf] = min(np.linalg.eigvalsh(Y).min() for Y in Ys(x[:nf]).values()) - 1.0
    for mu in [1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3]:
        for it in range(100):
            Yl = Ys(x[:nf]); sv = x[nf]
            g = np.zeros(nf + 1); g[nf] = -1.0; Hm = np.zeros((nf + 1, nf + 1))
            for bi in bis:
                n = Yl[bi].shape[0]
                Yinv = np.linalg.inv(Yl[bi] - sv * np.eye(n))
                Pm = [Yinv @ Yi[i][bi] for i in range(nf)] + [-Yinv]
                g -= mu * np.array([np.trace(p_) for p_ in Pm])
                for i in range(nf + 1):
                    for j in range(i, nf + 1):
                        h = mu * np.trace(Pm[i] @ Pm[j]); Hm[i, j] += h
                        if i != j: Hm[j, i] += h
            step = -np.linalg.solve(Hm + 1e-15 * np.eye(nf + 1), g); dec = -g @ step
            def fobj(xx):
                tot = -xx[nf]
                for Y in Ys(xx[:nf]).values():
                    w = np.linalg.eigvalsh(Y - xx[nf] * np.eye(Y.shape[0]))
                    if w.min() <= 0: return np.inf
                    tot -= mu * np.sum(np.log(w))
                return tot
            tt = 1.0; f0 = fobj(x)
            while fobj(x + tt * step) > f0 - 0.25 * tt * dec and tt > 1e-14: tt *= 0.5
            if tt <= 1e-14: break
            x = x + tt * step
            if dec < 1e-12: break
    log('numeric interior point: min eigenvalue margin %.4f  (%.0fs)' % (x[nf], time.time() - T0))
    # round to rationals and verify exactly
    t_rat = [Fr(float(v)).limit_denominator(10 ** 6) for v in x[:nf]]
    y = y_from_t(t_rat)
    if perturb is not None: y = perturb(y)
    mats = Ymats(y)
    def ldl_pivots(M):
        A = [row[:] for row in M]; n = len(A); piv = []
        for k in range(n):
            p = A[k][k]; piv.append(p)
            if p == 0: return piv
            for i in range(k + 1, n):
                f = A[i][k] / p
                for j in range(k, n): A[i][j] -= f * A[k][j]
        return piv
    allpos = True
    pivrep = {}
    for bi, M in mats.items():
        pv = ldl_pivots(M)
        pivrep['block kind=%d L=%d' % (blocks[bi]['kind'], blocks[bi]['L'])] = [str(p) for p in pv]
        if not all(p > 0 for p in pv): allpos = False
    chk['every reduced Gram block is positive definite (exact LDL pivots all > 0)'] = allpos
    # exact identity check by direct expansion
    S = {}
    for (bi, al, be), v, cp in zip(params, y, coefpolys):
        S = padd(S, pscale(cp, v))
    D = padd(F, S, -1)
    chk['exact identity %s |u|^6 - N |u|^2 = sum of Hermitian squares (difference polynomial is 0)' % LAM] = (len(D) == 0)

    return dict(chk=chk, consistent=True, y=y, coefpolys=coefpolys, params=params, blocks=blocks, mats=mats, pivrep=pivrep, rank=rank, nun=nun, margin=float(x[nf]), D=D)
