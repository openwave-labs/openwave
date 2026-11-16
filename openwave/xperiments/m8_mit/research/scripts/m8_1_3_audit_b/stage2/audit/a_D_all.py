import sys, os; sys.dont_write_bytecode = True
if len(sys.argv) > 1:
    os.environ["PLANT"] = sys.argv[1]
"""Block D for the boxes 'O3 R4' and 'O3 R5', then B.4 at those two certified points (clauses 2, 3, 4).
Own interval code on mpmath.iv (krawczyk.py).  h: the exact rational quartic of hpoly.py.  Generators: inputs.json (exact).
Rigour notes:
 * the tested box is the INNER box [t_hi - r_lo, t_lo + r_lo] (t_lo/t_hi: enclosure of the decimal centre, r_lo: lower end of
   the enclosure of 1e-60), contained in the Section-5 box; x0 is enclosed from its exact decimal.
 * after certification, all B.4/D quantities are enclosed over the Krawczyk image K(Z) intersected with Z, a box that contains
   the certified point (clause 2).
 * structural facts used (exact at the true point): K_u is in ker Mhat; Shat u = 8h J0 u with J0 u in K_u; the characteristic
   polynomial of Shat is even (Shat = J0 Mhat, Mhat G-selfadjoint); sigma(J0 u, X u) = 0 for G-skew X (checked exactly).
"""
import os, json, time
from fractions import Fraction as Fr
from mpmath import mp, iv
import sympy as sp
from core import mlist, Fm
from reps import SLOTS
from hpoly import h_poly, r6_poly, PolyEval
from exact_pt import group_generators
from krawczyk import *
from ivutil import ends, outward, width, contains_zero
from chk import check, save_json, planted

here = os.path.dirname(os.path.abspath(__file__))
inp = json.load(open(os.path.join(here, '..', 'inputs.json')))
iv.dps = 130
mp.dps = 130
n = 6; N = 7; D = 14
ms = mlist(n)
Gd = [Fm(n, m) for m in ms] * 2
GdI = [iv.mpf(g) for g in Gd]

# ---- generators from inputs.json, exact; G-skewness and span equality with my own generators
gens_in = [sp.Matrix([[sp.Rational(Fr(e).numerator, Fr(e).denominator) for e in row] for row in M]) for M in inp['generators']['matrices']]
Gs = sp.diag(*Gd)
check('D: each given generator is G-skew (exact)', all((Gs * X + (Gs * X).T) == sp.zeros(D, D) for X in gens_in))
mine = group_generators('R4')
stack = lambda L: sp.Matrix.hstack(*[sp.Matrix(X.reshape(D * D, 1)) for X in L])
r_in = stack(gens_in).rank(); r_all = stack(gens_in + mine).rank()
if planted('Dspan'):
    # planted defect: replace the z-rotation by a G-skew map outside the span (a Theta-free 'rotation' with a wrong weight)
    bad = sp.zeros(D, D); bad[N + 1, 1] = 1; bad[1, N + 1] = -1
    gens_chk = gens_in[:3] + [bad]
    r_in = stack(gens_chk).rank(); r_all = stack(gens_chk + mine).rank()
ok_span = r_in == 4 and r_all == 4
check('D: the given generators span the same 4-dim space as my phase + rotation generators (exact ranks)', ok_span, f'{r_in}, {r_all}')
Xi = [[[iv.mpf(int(X[i, k].p)) / int(X[i, k].q) for k in range(D)] for i in range(D)] for X in gens_in]
J0 = [[iv.mpf(0)] * D for _ in range(D)]
for i in range(N):
    J0[N + i][i] = iv.mpf(1); J0[i][N + i] = iv.mpf(-1)


def conv(c):
    return iv.mpf(c.numerator) / c.denominator if isinstance(c, Fr) else iv.mpf(c)


out = {'D': {'R4': {}, 'R5': {}}, 'B.4': {'R4': {}, 'R5': {}}}
cert = {}
for key in ['O1 R4','O1 R5','O2 R4','O2 R5','O3 R4','O3 R5']:
    slot = key.split()[1]
    box = inp['boxes']['boxes'][key]
    he = PolyEval(h_poly(slot), D)
    t0 = time.time()
    xc = box['centre, x-part']; x0s = box['x0']
    x0 = [iv.mpf(s) for s in x0s]
    k_vec = [mat_vec(X, x0) for X in Xi]
    Gk = [[GdI[a] * kv[a] for a in range(D)] for kv in k_vec]
    # centre, lambda part from my own h (point value at 130 digits)
    xcm = [mp.mpf(s) for s in xc]
    hc = he.value(xcm, lambda c: mp.mpf(c.numerator) / c.denominator if isinstance(c, Fr) else mp.mpf(c))
    nc = sum(Gd[a] * xcm[a] ** 2 for a in range(D))
    lam_c = 4 * hc / nc
    centre = xcm + [lam_c] + [mp.mpf(0)] * 4
    rlo, _ = ends(iv.mpf('1e-60'))
    Z = []
    for i in range(19):
        tl, th = ends(iv.mpf(xc[i])) if i < D else ends(iv.mpf(centre[i]))
        Z.append(iv.mpf([th - rlo, tl + rlo]))
    m = [iv.mpf(xc[i]) if i < D else iv.mpf(centre[i]) for i in range(19)]

    def F(z):
        x = z[:D]; lam = z[D]; c = z[D + 1:]
        g = he.grad(x, conv)
        res = []
        for a in range(D):
            v = g[a] - lam * GdI[a] * x[a]
            for i in range(4):
                v = v - c[i] * Gk[i][a]
            res.append(v)
        res.append(sum((GdI[a] * x[a] * x[a] for a in range(D)), iv.mpf(0)) - 1)
        for i in range(4):
            res.append(sum((Gk[i][a] * (x[a] - x0[a]) for a in range(D)), iv.mpf(0)))
        return res

    def Jac(z):
        x = z[:D]; lam = z[D]
        H = he.hess(x, conv)
        Jm = [[iv.mpf(0)] * 19 for _ in range(19)]
        for a in range(D):
            for b in range(D):
                Jm[a][b] = H[a][b] - (lam * GdI[a] if a == b else iv.mpf(0))
            Jm[a][D] = -GdI[a] * x[a]
            for i in range(4):
                Jm[a][D + 1 + i] = -Gk[i][a]
        for b in range(D):
            Jm[D][b] = 2 * GdI[b] * x[b]
        for i in range(4):
            for b in range(D):
                Jm[D + 1 + i][b] = Gk[i][b]
        return Jm

    if planted('Dbox'):
        Z = [z + iv.mpf('1e-40') for z in Z]
    ok, K, contr = krawczyk(F, Jac, Z, m)
    check(f'D {key}: Krawczyk K(Z) lies in the interior of Z (unique zero of the slice system in the box)', ok,
          f'||I - Y J(Z)||_inf = {mp.nstr(contr, 3)}')
    Kc = [intersect(K[i], Z[i]) for i in range(19)]
    wK = max(width(v) for v in Kc)
    print(f'{key}: Krawczyk done in {time.time() - t0:.1f}s; width of K(Z) cap Z: {mp.nstr(wK, 3)}; '
          f'c-part of K(Z): {[outward(Kc[D + 1 + i], 6) for i in range(4)]}')
    # second Krawczyk iteration on the image (tighter certified box)
    ok2, K2, contr2 = krawczyk(F, Jac, Kc)
    if ok2:
        Kc = [intersect(K2[i], Kc[i]) for i in range(19)]
    print(f'   second iteration on K(Z): {ok2}; width now {mp.nstr(max(width(v) for v in Kc), 3)}')
    cert[key] = Kc
    X = Kc[:D]; lamI = Kc[D]
    # ---------------- quantities over the certified box X
    hX = he.value(X, conv)
    nX = sum((GdI[a] * X[a] * X[a] for a in range(D)), iv.mpf(0))
    HX = he.hess(X, conv)
    Mop = [[HX[a][b] / GdI[a] - (4 * hX if a == b else iv.mpf(0)) for b in range(D)] for a in range(D)]
    Sop = mat_mul(J0, Mop)
    # r6
    r6 = PolyEval(r6_poly(), D).value(X, conv)
    r6hat = 924 * r6 / (nX * nX)
    # k: Gram determinant of the four tangent vectors X_i u (G metric)
    tv = [mat_vec(Xm, X) for Xm in Xi]
    Gram = [[sum((tv[i][a] * GdI[a] * tv[j][a] for a in range(D)), iv.mpf(0)) for j in range(4)] for i in range(4)]
    GramM = iv.matrix(Gram)
    gdet = iv.det(GramM)
    kk = 4 if not contains_zero(gdet) else None
    check(f'D {key}: Gram determinant of the 4 tangent vectors excludes 0 -> k = 4', kk == 4, str(outward(gdet, 8)))
    # r_u: sigma on K_u, sigma(a,b) = a^T G J0 b
    J0tv = [mat_vec(J0, t) for t in tv]
    Sig = [[sum((tv[i][a] * GdI[a] * J0tv[j][a] for a in range(D)), iv.mpf(0)) for j in range(4)] for i in range(4)]
    rot_nonzero = any(not contains_zero(Sig[i][j]) for i in range(1, 4) for j in range(1, 4) if i != j)
    phase_row_small = all(contains_zero(Sig[0][j]) for j in range(4))
    check(f'D {key}: sigma on K_u: phase row encloses 0 (structurally exactly 0) and a rotation entry excludes 0', rot_nonzero and phase_row_small)
    r_u = 4 - 2 if rot_nonzero else None
    # charpoly of Mhat: mu^4 * r(mu); coefficients c_0..c_3 structurally 0, c_4 must exclude 0
    cM = charpoly_leverrier(Mop)
    low_ok = all(contains_zero(cM[i]) for i in range(4))
    c4_ok = not contains_zero(cM[4])
    check(f'D {key}: charpoly(Mhat): c_0..c_3 enclose 0 (structural zeros from K_u) and c_4 excludes 0 (ker Mhat = K_u, n_0 = 0)', low_ok and c4_ok)
    rco = cM[4:]
    signs = []
    for c in rco:
        lo, hi = ends(c)
        signs.append(1 if lo > 0 else (-1 if hi < 0 else 0))
    all_signed = all(s != 0 for s in signs)
    check(f'D {key}: every coefficient of the reduced charpoly of Mhat has a certified sign', all_signed)
    changes = lambda s: sum(1 for a, b in zip(s, s[1:]) if a != b)
    npos = changes(signs)
    nneg = changes([s * (-1) ** i for i, s in enumerate(signs)])
    check(f'D {key}: Descartes counts close (all roots of the symmetric reduced charpoly real and nonzero)', npos + nneg == len(rco) - 1)
    # u carries eigenvalue 8h > 0: remove it from the positive count
    h_pos = ends(hX)[0] > 0
    sig = [nneg, 0, npos - 1] if h_pos else None
    # zero multiplicity of Shat
    cS = charpoly_leverrier(Sop)
    odd_ok = all(contains_zero(cS[i]) for i in range(1, 15, 2))
    low_ok = all(contains_zero(cS[i]) for i in range(0, 6))
    c6_ok = not contains_zero(cS[6])
    check(f'D {key}: charpoly(Shat): odd coefficients and c_0..c_5 enclose 0; c_6 excludes 0 -> zero multiplicity 6', odd_ok and low_ok and c6_ok)
    z = 6 if c6_ok else None
    # slow polynomial in nu: q(nu) = sum_i cS[6+2i] nu^i
    q = [cS[6 + 2 * i] for i in range(5)]
    qmid = [mid(c) for c in q]
    roots = mp.polyroots(list(reversed(qmid)), maxsteps=200, extraprec=200)
    rinfo = []
    for r in roots:
        if abs(mp.im(r)) > mp.mpf(10) ** -40:
            rinfo.append(('complex', r, None, False))
            continue
        rr = mp.re(r)
        okr, Kr = root_krawczyk_real(q, rr, mp.mpf(10) ** -52)
        rinfo.append(('real', rr, Kr, okr))
    all_real = all(t[0] == 'real' and t[3] for t in rinfo)
    check(f'D {key}: all {len(rinfo)} slow roots certified real and simple (Krawczyk, radius 1e-52)', all_real and len(rinfo) == 4)
    rinfo.sort(key=lambda t: mp.re(t[1]))
    widths_ok = all(width(t[2]) <= mp.mpf(10) ** -50 for t in rinfo if t[2] is not None) and width(r6hat) <= mp.mpf(10) ** -50
    check(f'D {key}: every reported enclosure has width <= 1e-50', widths_ok)
    pos = [t for t in rinfo if ends(t[2])[0] > 0]
    neg = [t for t in rinfo if ends(t[2])[1] < 0]
    if not pos and len(neg) == len(rinfo):
        words = 'elliptic, simple'
    elif len(pos) == 1:
        words = 'hyperbolic, one real pair'
    elif len(pos) >= 2:
        words = 'hyperbolic, real pairs'
    else:
        words = 'undetermined'
    print(f'{key}: k = {kk}, r_u = {r_u}, signature {sig}, zero multiplicity {z}')
    print(f'   924 r6hat in {outward(r6hat, 60)}  (width {mp.nstr(width(r6hat), 3)})')
    for t in rinfo:
        print(f'   slow root nu in {outward(t[2], 60)} (width {mp.nstr(width(t[2]), 3)}), sign {"+" if ends(t[2])[0] > 0 else "-"}, simple: {t[3]}')
    print(f'   slow-flow words: {words}')
    # ---- the same enclosures over the Section-5 box itself (inner box Z), for comparison
    XZ = Z[:D]
    hZ = he.value(XZ, conv); nZ = sum((GdI[a] * XZ[a] * XZ[a] for a in range(D)), iv.mpf(0))
    HZ = he.hess(XZ, conv)
    MZ = [[HZ[a][b] / GdI[a] - (4 * hZ if a == b else iv.mpf(0)) for b in range(D)] for a in range(D)]
    cSZ = charpoly_leverrier(mat_mul(J0, MZ))
    qZ = [cSZ[6 + 2 * i] for i in range(5)]
    r6Z = 924 * PolyEval(r6_poly(), D).value(XZ, conv) / (nZ * nZ)
    rootsZ = []
    for t in rinfo:
        okz, Kz = root_krawczyk_real(qZ, mid(t[2]), mp.mpf(10) ** -45)
        rootsZ.append((okz, Kz))
    print(f'   over the Section-5 box: 924 r6hat in {outward(r6Z, 25)} (width {mp.nstr(width(r6Z), 3)}); c_6 of charpoly(Shat) excludes 0: {not contains_zero(cSZ[6])}')
    for okz, Kz in rootsZ:
        print(f'      slow root certified {okz}: width {mp.nstr(width(Kz), 3)}  {outward(Kz, 25)}')
    secZ = {'924 r6hat': outward(r6Z, 60), '924 r6hat width': mp.nstr(width(r6Z), 5),
            'slow roots': [{'certified': okz, 'enclosure': outward(Kz, 60), 'width': mp.nstr(width(Kz), 5)} for okz, Kz in rootsZ]}
    rec = {'enclosures over the Section-5 box (inner box), for comparison': secZ,'box certified (Krawczyk)': ok, 'Krawczyk contraction ||I - Y J(Z)||': mp.nstr(contr, 5),
           'certified box used for the enclosures': 'K(Z) cap Z after a second Krawczyk step (a box that contains the certified point)',
           'certified box max width': mp.nstr(max(width(v) for v in Kc), 5),
           'lambda enclosure': outward(Kc[D], 60), 'c enclosures': [outward(Kc[D + 1 + i], 8) for i in range(4)],
           '924 r6hat': outward(r6hat, 60), '924 r6hat width': mp.nstr(width(r6hat), 5),
           'signature': sig, 'k': kk, 'r_u': r_u, 'zero multiplicity': z,
           'slow roots (nu)': [{'enclosure': outward(t[2], 60), 'width': mp.nstr(width(t[2]), 5),
                                'sign': '+' if ends(t[2])[0] > 0 else '-', 'real and simple': t[3]} for t in rinfo]}
    out['D'][slot][key] = rec
    out['B.4'][slot][key] = dict(rec, **{'slow-flow words': words})
    cert[key] = (Kc, rinfo)

# ---------------- B.4: eigenvalue ratios R5 / R4 (clause 3 pairing), all three points
ratios_all = {}
for o in ['O1','O2','O3']:
    r4 = cert[o+' R4'][1]; r5 = cert[o+' R5'][1]
    lst = []
    for a, (t4, t5) in enumerate(zip(r4, r5)):
        n4 = t4[2]; n5 = t5[2]
        s4 = ends(n4)[0] > 0; s5 = ends(n5)[0] > 0
        if s4 == s5:
            rat = iv.sqrt(n5 / n4); kind = 'real'
        else:
            rat = iv.sqrt(abs(n5) / abs(n4)); kind = 'mixed'
        lst.append({'pair': a + 1, 'ratio': outward(rat, 55), 'type': kind, 'width': mp.nstr(width(rat), 5)})
        print(f'B.4 {o} pair {a + 1}: xi_R5/xi_R4 = {outward(rat, 55)} {kind} (width {mp.nstr(width(rat), 3)})')
        check(f'B.4 {o} pair {a+1}: ratio width <= 1e-50', width(rat) <= mp.mpf(10) ** -50)
    ratios_all[o] = lst
out['B.4']['ratios'] = ratios_all
save_json('a_D_all.json', out)
