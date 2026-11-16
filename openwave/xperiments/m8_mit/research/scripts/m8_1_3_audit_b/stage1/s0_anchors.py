"""Item 0(a)-(e).  Group and character table DERIVED here (Burnside-Dixon on exact class constants, values identified in
Z[phi] and then verified exactly by the orthogonality relations); decompositions exact from the V_j characters U_{2j}(Re q)."""
import random
from fractions import Fraction as Fr
from mpmath import mp, matrix
from core import *
from reps import *
from numrep import *
from hpoly import cg_exact
from s3poly import Phi_polys, norm_inner_vec, kfrac, wnorm2
from chk import check, save_json, planted

mp.dps = 40
out = {}

# ---------------- 0(a)
nq1 = sum((c * c for c in Q1), Q5(0)); nq2 = sum((c * c for c in Q2), Q5(0))
print('||q1||^2 =', nq1, ' ||q2||^2 =', nq2)
check('0(a) |q1| = |q2| = 1 exactly', nq1 == Q5(1) and nq2 == Q5(1))
G, classes, cls_of, chars = group_data()
print('|Gamma0| =', len(G))
cls_info = []
for r, cl in enumerate(classes):
    g = cl[0]
    cls_info.append({'size': len(cl), 'order': element_order(g), 'Re q': str(g[0])})
    print(f'  class {r}: size {len(cl):3d}, element order {element_order(g):2d}, real part {g[0]}')
check('0(a) class sizes sum to |G|', sum(len(c) for c in classes) == len(G))
# exact orthogonality of the identified characters (can fail: identification is numerical)
ok = True
for a in range(len(chars)):
    for b in range(len(chars)):
        v = inner_classfn(chars[a], chars[b])
        ok &= (v == Q5(1 if a == b else 0))
check('0(a) identified characters satisfy exact row orthogonality', ok)
ok = True
for r in range(len(classes)):
    for t in range(len(classes)):
        v = sum((chars[a][r] * chars[a][t] for a in range(len(chars))), Q5(0))
        ok &= (v == (Q5(Fr(len(G), len(classes[r]))) if r == t else Q5(0)))
check('0(a) exact column orthogonality', ok)
# independent check: characters of V_j decompose with nonnegative integer multiplicities for all j <= 20
ok = True
for l in range(0, 21):
    tot = 0
    for i in range(len(chars)):
        m = inner_classfn(Vj_character(l), chars[i])
        ok &= (m.b == 0 and m.a.denominator == 1 and m.a >= 0)
        tot += int(m.a) * int(chars[i][0].a) if m.b == 0 and m.a.denominator == 1 else 0
    ok &= (tot == l + 1)
check('0(a)/(b) every V_l (l<=20) decomposes with nonnegative integer multiplicities, dims add up', ok)
char_out = []
for i, c in enumerate(chars):
    print(f'  chi_{i}: dim {c[0]}:', [str(v) for v in c])
    char_out.append([str(v) for v in c])
out['0(a)'] = {'|q1|': '1', '|q2|': '1', 'order': len(G), 'classes': cls_info,
               'characters': char_out, 'notation': 'a+b*sqrt5 written (a+b*sqrt5); classes in the order listed',
               'status': 'DERIVED in this room'}

# ---------------- 0(b)
lev = {}
for i, c in enumerate(chars):
    lv = irrep_levels(i, 20)
    lev[i] = lv
    print(f'  irrep {i} (dim {c[0]}): levels<=20 with multiplicity', lv)
names = {}
for nm in SLOTS:
    names[nm] = slot_irrep(nm)
    print(f'  slot {nm} -> irrep {names[nm]} (dim {chars[names[nm]][0]}, lowest level {lev[names[nm]][0][0]})')
check('0(b) the three slot names select distinct irreps', len(set(names.values())) == 3)
out['0(b)'] = {'irreps': [{'index': i, 'dim': str(chars[i][0]), 'levels<=20': [[l, m] for l, m in lev[i]]} for i in lev],
               'slots': {nm: names[nm] for nm in names}}

# ---------------- 0(c)
out['0(c)'] = {}
rng = random.Random(3)
for nm in ['R2', 'R4', 'R5']:
    d, n = SLOTS[nm]
    mult = multiplicity(n, names[nm])
    Pm = projector_mono(nm)
    P = P_ortho_num(Pm, n)
    eta = orthonormal_range(P)
    # checks: eta^dag eta = I, P = eta eta^dag, D(h) eta = eta rho(h) for the generators and a few elements
    e1 = mp.mnorm(eta.H * eta - mp.eye(d), 1)
    e2 = mp.mnorm(eta * eta.H - P, 1)
    errs = []
    for q in (Q1, Q2, G[17], G[77]):
        Dq = D_of_q(q, n)
        rho = eta.H * Dq * eta
        errs.append(mp.mnorm(Dq * eta - eta * rho, 1))
        # character of rho equals the slot's character
        tr = sum(rho[i, i] for i in range(d))
        errs.append(abs(tr - chars[names[nm]][cls_of[qkey(q)]].mp(mp)))
    check(f'0(c) {nm}: eta^dag eta = I, eta eta^dag = P, D eta = eta rho, tr rho = chi', max([e1, e2] + errs) < mp.mpf(10) ** -30,
          f'max err {mp.nstr(max([e1, e2] + errs), 3)}')
    # s_N two routes: Schur (dim rho/(2j+1)) and exact S^3 moments at a random rational w
    sN = s_N(nm)
    p = [kfrac(Fr(rng.randint(-4, 4), rng.randint(1, 4)), Fr(rng.randint(-4, 4), rng.randint(1, 4))) for _ in range(n + 1)]
    if planted('sN'):
        p = p[:-1] + [kfrac(0)]
        sN = sN * Fr(101, 100)
    Ph = Phi_polys(nm, p)
    tot = norm_inner_vec(nm, Ph, Ph, levels=False)[None]
    ratio = tot[0] / wnorm2(nm, p)
    check(f'0(c) {nm}: s_N by Schur equals exact S^3 integral at a random w', ratio == sN and tot[1:] == (0, 0, 0), f'{ratio} vs {sN}')
    ms = mlist(n)
    print(f'  {nm}: multiplicity at lowest level n={n}: {mult};  s_N = {sN}')
    print(f'  {nm}: eta (rows m = {[str(m) for m in ms]}):')
    for i in range(n + 1):
        print('     ', [mp.nstr(eta[i, a], 20) for a in range(d)])
    out['0(c)'][nm] = {'multiplicity': mult, 's_N': str(sN),
                       'eta': [[f'{mp.nstr(mp.re(eta[i, a]), 30)}{"+" if mp.im(eta[i, a]) >= 0 else "-"}{mp.nstr(abs(mp.im(eta[i, a])), 30)}*I' for a in range(d)] for i in range(n + 1)],
                       'eta rows': [str(m) for m in ms],
                       'eta construction': 'Gram-Schmidt of the columns of the exact projector P = (dim/|G|) sum chi(h) D(h), in column order m = j..-j'}

# ---------------- 0(d)
sg, sq = cg_exact(3, 3, 3, -3, 6, 0)
print('<3 3; 3 -3 | 6 0> = ', '+' if sg > 0 else '-', f'sqrt({sq})')
# independent: build |6 0> by lowering |3 3>|3 3> six times in the CS orthonormal basis and read the v_3 (x) v_-3 component
from mpmath import sqrt as msqrt
j = 3
ms3 = mlist(6)
vec = {(3, 3): mp.mpf(1)}
J = 6; M = 6
for step in range(6):
    new = {}
    for (m1, m2), c in vec.items():
        if m1 > -3:
            f = msqrt((j + m1) * (j - m1 + 1)); new[(m1 - 1, m2)] = new.get((m1 - 1, m2), 0) + c * f
        if m2 > -3:
            f = msqrt((j + m2) * (j - m2 + 1)); new[(m1, m2 - 1)] = new.get((m1, m2 - 1), 0) + c * f
    nn = msqrt(sum(v * v for v in new.values()))
    vec = {k: v / nn for k, v in new.items()}
lowered = vec[(3, -3)]
check('0(d) Racah CG <33;3-3|60> equals the lowering-operator construction', abs(lowered - sg * mp.sqrt(mp.mpf(sq.numerator) / sq.denominator)) < 1e-30,
      mp.nstr(lowered, 20))
th2 = {}
for j2 in (6, 7):
    W = W_theta(j2)
    T2m = W * conjm(W)            # Theta^2 u = W conj(W conj u) = W conj(W) u
    val = T2m[0, 0]
    th2[j2] = val
    check(f'0(d) Theta^2 is scalar on V_{j2}/2', mp.mnorm(T2m - val * mp.eye(j2 + 1), 1) < 1e-30)
    print(f'  Theta^2 on V_{j2}/2 = {mp.nstr(val, 5)}')
out['0(d)'] = {'sign <3 3;3 -3|6 0>': '+' if sg > 0 else '-', 'value': f'sqrt({sq})',
               'Theta^2 on V_3': str(int(mp.re(th2[6]))), 'Theta^2 on V_7/2': str(int(mp.re(th2[7])))}

# ---------------- 0(e)
nm = 'R2'; d, n = SLOTS[nm]
P = P_ortho_num(projector_mono(nm), n)
eta = orthonormal_range(P)
rho = {}
for q in (Q1, Q2):
    rho[qkey(q)] = eta.H * D_of_q(q, n) * eta
# solve conj(rho(h)) C = C rho(h) for h = q1, q2 (4 unknowns)
rows = []
for q in (Q1, Q2):
    R = rho[qkey(q)]
    for a in range(2):
        for b in range(2):
            row = [mp.mpc(0)] * 4
            for c in range(2):
                row[c * 2 + b] += mp.conj(R[a, c])      # (conj R C)_{ab} = sum_c conj R_{ac} C_{cb}
                row[a * 2 + c] -= R[c, b]               # (C R)_{ab} = sum_c C_{ac} R_{cb}
            rows.append(row)
A = matrix(rows)
U_, S_, V_ = mp.svd_c(A)
cvec = V_.H[:, 3]
C = matrix(2, 2)
for a in range(2):
    for b in range(2):
        C[a, b] = cvec[a * 2 + b]
# normalise to unitary
nrm = mp.sqrt(mp.re((C.H * C)[0, 0]))
C = C / nrm
check('0(e) C is unitary and intertwines conj(rho) with rho on both generators',
      mp.mnorm(C.H * C - mp.eye(2), 1) < 1e-30 and max(mp.mnorm(conjm(rho[qkey(q)]) * C - C * rho[qkey(q)], 1) for q in (Q1, Q2)) < 1e-30,
      f'smallest singular values {mp.nstr(S_[3], 3)}, next {mp.nstr(S_[2], 3)}')
CC = C * conjm(C)
print('  C =', [[mp.nstr(C[a, b], 15) for b in range(2)] for a in range(2)])
print('  C conj(C) =', [[mp.nstr(CC[a, b], 15) for b in range(2)] for a in range(2)])
# the map psi -> conj(psi) C on w: w' = c0 W^T conj(w) with c0 from W^dag conj(eta) C = c0 eta ; compare with Theta w = W conj(w)
W = W_theta(n)
X = W.H * conjm(eta) * C
c0m = eta.H * X
check('0(e) W^dag conj(eta) C is a multiple of eta (multiplicity one)', mp.mnorm(X - eta * c0m, 1) < 1e-30 and abs(c0m[0, 1]) < 1e-30 and abs(c0m[0, 0] - c0m[1, 1]) < 1e-30)
c0 = c0m[0, 0]
WT = W.T
rel = WT * mp.inverse(W)     # W^T = rel * W
check('0(e) W^T = -W on V_7/2', mp.mnorm(WT + W, 1) < 1e-30)
# direct functional check at random group points: conj(phi_w(g)) C == phi_{w'}(g) with w' = -c0 Theta w
ok = True
w = matrix([mp.mpc(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(n + 1)])
wp = -c0 * (W * conjm(w))
for t in range(3):
    a, b = random_su2(rng)
    Dg = D_ortho_num(a, b, n)
    lhs = conjm((w.T * Dg * eta)) * C
    rhs = wp.T * Dg * eta
    ok &= mp.mnorm(lhs - rhs, 1) < 1e-28
check('0(e) psi_w -> conj(psi_w) C equals psi at w\' = (-c0) Theta w at random g', ok)
print('  the map is w -> (-c0) Theta w with -c0 =', mp.nstr(-c0, 20), '|c0| =', mp.nstr(abs(c0), 20))
out['0(e)'] = {'C (in the eta basis of 0(c))': [[mp.nstr(C[a, b], 25) for b in range(2)] for a in range(2)],
               'C conj(C)': '-1' if mp.mnorm(CC + mp.eye(2), 1) < 1e-30 else 'not -1 (see log)',
               'map on w': 'w -> e^{i alpha} Theta w, e^{i alpha} = -c0 = ' + mp.nstr(-c0, 20) +
                           '; the phase is the phase freedom of C (C -> e^{-i alpha} C makes the map exactly Theta)'}
check('0(e) C conj(C) = -1', mp.mnorm(CC + mp.eye(2), 1) < 1e-30)
save_json('out_0.json', out)
