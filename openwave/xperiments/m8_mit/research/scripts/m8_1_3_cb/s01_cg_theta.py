"""Item 0(d) and the conventions underneath everything else:
CG coefficients built from the CS convention (lib_cg), compared with Racah's formula (exact)
and with sympy's library CG (a lookup, reported as a comparison only); D^j checks;
Theta checks."""
import os, sys, random
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp
from lib_check import check, summary, DEFECT
from lib_cg import cg_table, cg, racah, mrange, sgnsq_to_mp, w as wfac
from lib_rep import D_mp, theta_apply, Jmats, group_elements
from lib_exact import qmul, quat_to_ab

mp.mp.dps = 40
random.seed(1)
J7, J3 = Fr(7, 2), Fr(3)

# --- 0(d) -------------------------------------------------------------------
s, r = cg(3, 3, 3, -3, 6, 0)
print(f"0(d) <3 3; 3 -3 | 6 0> = {'+' if s > 0 else '-'}sqrt({r})  = {float(s * r ** 0.5):.15f}")
check("0(d): <3 3;3 -3|6 0> > 0 as the convention requires", s > 0)
for j in (Fr(1, 2), Fr(1), J3, J7):
    ss, rr = cg(j, j, j, -j, 2 * j, 0)
    check(f"<j j; j -j | 2j 0> > 0 for j={j}", ss > 0)

# --- CG: Racah comparison for every table this room uses ---------------------
tables = []
for j in (J3, J7):
    for J1 in range(0, int(2 * j) + 1):
        tables.append((j, j, Fr(J1)))
        J = abs(Fr(J1) - j)
        while J <= J1 + j:
            tables.append((Fr(J1), j, J))
            J += 1
nmis, ncoef = 0, 0
for (j1, j2, J) in tables:
    t = cg_table(j1, j2, J)
    if DEFECT == "cg_sign" and (j1, j2, J) == (J3, J3, Fr(6)):
        k = (Fr(3), Fr(-3), Fr(0))
        t = dict(t); t[k] = (-t[k][0], t[k][1])
    for m1 in mrange(j1):
        for m2 in mrange(j2):
            M = m1 + m2
            if abs(M) > J:
                continue
            ncoef += 1
            if t.get((m1, m2, M), (0, Fr(0))) != racah(j1, m1, j2, m2, J, M):
                nmis += 1
print(f"CG built from the convention vs Racah's formula: {ncoef} coefficients in {len(tables)} tables, {nmis} mismatches")
check("built CG == Racah formula exactly, all tables used", nmis == 0)

# orthonormality of the built CG (exact)
bad = 0
for (j1, j2, J) in [(J3, J3, Fr(J1)) for J1 in range(7)] + [(J7, J7, Fr(J1)) for J1 in range(8)]:
    t = cg_table(j1, j2, J)
    for M in mrange(J):
        s2 = sum(r for (a, b, MM), (sg, r) in t.items() if MM == M)
        bad += (s2 != 1)
check("each coupled state has unit norm (exact sum of squares)", bad == 0)

# sympy library comparison on a sample (lookup, not derivation)
from sympy.physics.quantum.cg import CG as SCG
import sympy as sp
samp, smis = 0, 0
for (j1, j2, J) in random.sample(tables, 25):
    t = cg_table(j1, j2, J)
    for (m1, m2, M), (sg, r) in random.sample(sorted(t.items()), min(6, len(t))):
        v = sp.nsimplify(SCG(sp.Rational(j1), sp.Rational(m1), sp.Rational(j2), sp.Rational(m2), sp.Rational(J), sp.Rational(M)).doit())
        mine = sg * sp.sqrt(sp.Rational(r.numerator, r.denominator))
        samp += 1
        smis += (sp.simplify(v - mine) != 0)
print(f"sympy.physics.quantum.cg.CG comparison: {samp} sampled coefficients, {smis} mismatches")
check("built CG == sympy library CG on the sample", smis == 0)

# --- D^j: homomorphism, unitarity, R_z, CS ladder, coupling covariance -------
G, q1, q2 = group_elements()
def ab_mp(q):
    a, b = quat_to_ab(q)
    return a.mp(mp), b.mp(mp)
for j in (Fr(1, 2), J3, J7):
    p, q = G[17], G[55]
    Dp, Dq, Dpq = D_mp(j, *ab_mp(p), mp), D_mp(j, *ab_mp(q), mp), D_mp(j, *ab_mp(qmul(p, q)), mp)
    check(f"D^{j}(p)D^{j}(q) = D^{j}(pq) (quaternion map is a homomorphism)", mp.mnorm(Dp * Dq - Dpq) < mp.mpf(10) ** -35)
    n = int(2 * j + 1)
    check(f"D^{j} unitary", mp.mnorm(Dp * Dp.H - mp.eye(n)) < mp.mpf(10) ** -35)
    th = mp.mpf('0.7')
    Dz = D_mp(j, mp.expj(-th / 2), mp.mpc(0), mp)
    want = mp.diag([mp.expj(-m * th) for m in mrange(j)])
    check(f"D^{j}(R_z(theta)) = diag(e^(-i m theta))", mp.mnorm(Dz - want) < mp.mpf(10) ** -35)
    # generators: D(exp(-i t sigma_a/2)) ~ 1 - i t J_a
    Jz, Jp, Jm, Jx, Jy = Jmats(j, mp)
    t = mp.mpf(10) ** -12
    Dx = D_mp(j, mp.cos(t / 2), mp.mpc(0, -1) * mp.sin(t / 2), mp)   # exp(-i t sigma_x/2): a = cos, b = -i sin
    Dy = D_mp(j, mp.cos(t / 2), mp.sin(t / 2), mp)                     # exp(-i t sigma_y/2): a = cos, b = sin
    check(f"D^{j}: exp(-i t sigma_x/2) -> 1 - i t J_x (CS J_x)", mp.mnorm((Dx - mp.eye(n)) / t + mp.mpc(0, 1) * Jx) < mp.mpf(10) ** -10)
    check(f"D^{j}: exp(-i t sigma_y/2) -> 1 - i t J_y (CS J_y)", mp.mnorm((Dy - mp.eye(n)) / t + mp.mpc(0, 1) * Jy) < mp.mpf(10) ** -10)

# coupling covariance: [D u (x) D v]_J = D^J [u (x) v]_J
def couple(j1, j2, J, u, v):
    t = cg_table(Fr(j1), Fr(j2), Fr(J))
    ms1, ms2, msJ = mrange(j1), mrange(j2), mrange(J)
    out = mp.matrix(len(msJ), 1)
    for (m1, m2, M), sq in t.items():
        out[msJ.index(M)] += sgnsq_to_mp(sq, mp) * u[ms1.index(m1)] * v[ms2.index(m2)]
    return out
def rvec(n):
    return mp.matrix([mp.mpc(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)])
g = G[40]
for (j1, j2, J) in [(J7, J7, 5), (J3, J3, 6), (J3, J3, 2), (5, J7, Fr(9, 2)), (Fr(6), J3, Fr(9))]:
    u, v = rvec(int(2 * j1 + 1)), rvec(int(2 * j2 + 1))
    lhs = couple(j1, j2, J, D_mp(Fr(j1), *ab_mp(g), mp) * u, D_mp(Fr(j2), *ab_mp(g), mp) * v)
    rhs = D_mp(Fr(J), *ab_mp(g), mp) * couple(j1, j2, J, u, v)
    check(f"coupling covariant under D: ({j1})x({j2})->{J}", mp.norm(lhs - rhs) < mp.mpf(10) ** -30)

# --- Theta -----------------------------------------------------------------
for j in (J3, J7):
    n = int(2 * j + 1)
    u = rvec(n)
    tt = theta_apply(j, theta_apply(j, u, mp), mp)
    sign = +1 if mp.norm(tt - u) < 1e-30 else (-1 if mp.norm(tt + u) < 1e-30 else 0)
    print(f"0(d) Theta^2 on V_{j} = {sign:+d} * identity")
    check(f"Theta^2 = +-1 on V_{j}", sign != 0)
    Dg = D_mp(j, *ab_mp(g), mp)
    check(f"Theta commutes with D^{j}(g)", mp.norm(theta_apply(j, Dg * u, mp) - Dg * theta_apply(j, u, mp)) < 1e-30)
    # conj D_{m m'} = (-1)^{m-m'} D_{-m,-m'}
    ms = mrange(j)
    err = max(abs(mp.conj(Dg[r, c]) - (-1) ** int(ms[r] - ms[c]) * Dg[n - 1 - r, n - 1 - c]) for r in range(n) for c in range(n))
    check(f"conj D^{j}_(m m') = (-1)^(m-m') D^{j}_(-m,-m')", err < 1e-30)
summary()
