"""Items A.1, A.2, A.6 in R2 (j = 7/2).
Exact route: Q = <|phi|^4>_Haar / (s^2 ||w||^4), and Schur orthogonality on D x D gives
q_J = tr(Pi_J (P x P)) / ((2J+1) s^2).
High-precision route: 50-digit quadrature of the integral over SU(2) (quad.py), at test points,
solved for q_J and compared pointwise.
Usage: ./py sA1.py [--plant=NAME]  planted defects: qJ (perturb a q_J by 1e-6), hsval (perturb c_J), beta (perturb beta)
"""
import sys, json, random
import mpmath as mpm
from lib import *
from quartic import fJ, qJ_from_projector, cg_table, sym_channels
import quad

PLANT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--plant=")), None)
chk = Checker()
n = 7; n1 = 8; j = sp.Rational(n, 2)
d = json.load(open(rel("out_slot_R2.json")))
P = sp.Matrix(n1, n1, lambda r, c: sp.sympify(d["P"][r][c]))
eta = sp.Matrix(n1, 2, lambda r, c: sp.sympify(d["eta"][r][c]))
s = sp.sympify(d["s"])

q = qJ_from_projector(n, P, 2)
print("A.1 exact channel coefficients q_J (R2):", {J: str(v) for J, v in q.items()})
chk("q_J rational", all(v.is_Rational for v in q.values()))
if PLANT == "qJ":
    q[3] = q[3] + sp.Rational(1, 10 ** 6)

# test points (index k = j + m)
random.seed(7)
tests = []
for _ in range(6):
    tests.append([sp.Rational(random.randint(-9, 9), random.randint(1, 4)) + sp.I * sp.Rational(random.randint(-9, 9), random.randint(1, 4)) for _ in range(n1)])
tests.append([0] * 7 + [1])  # top weight v_{7/2}
F = [fJ(n, [sp.nsimplify(x) for x in t]) for t in tests]
for f in F:
    chk("sum_J f_J = 1 (exact)", sp.simplify(sum(f.values()) - 1) == 0)

mpm.mp.dps = 50
nodes = quad.haar_D(n, 32, 16)  # |phi|^4: phases up to 28 in each angle -> 32 > 28; degree <= 14 in t -> 16 nodes
eta_mp = [[mpm.mpmathify(sp.N(eta[k, a], 70)) for a in range(2)] for k in range(n1)]
ws = [[mpm.mpmathify(sp.N(x, 70)) for x in t] for t in tests]
avs = quad.averages(nodes, eta_mp, ws)
s_mp = mpm.mpf(s.p) / s.q
Qnum = []
for w, (a2, a4) in zip(ws, avs):
    nw2 = mpm.fsum(abs(x) ** 2 for x in w)
    Qnum.append(a4 / (s_mp ** 2 * nw2 ** 2))
maxdiff = mpm.mpf(0)
for t, f, Qn in zip(tests, F, Qnum):
    Qe = sum(q[J] * f[J] for J in q)
    dif = abs(Qn - mpm.mpmathify(sp.N(Qe, 60)))
    maxdiff = max(maxdiff, dif)
    print(f"   Q quadrature = {mpm.nstr(Qn, 40)}   sum q_J f_J = {sp.N(Qe, 40)}   |diff| = {mpm.nstr(dif, 3)}")
chk("A.1 exact q_J reproduce the 50-digit quadrature of Q at all 7 test points to 1e-40", maxdiff < mpm.mpf(10) ** -40, mpm.nstr(maxdiff, 3))
# solve for q_J from the first four quadrature values (high-precision route), then compare
Js = sym_channels(n)
A = mpm.matrix([[mpm.mpmathify(sp.N(F[i][J], 60)) for J in Js] for i in range(4)])
bvec = mpm.matrix([Qnum[i] for i in range(4)])
qs = mpm.lu_solve(A, bvec)
for J, v in zip(Js, qs):
    print(f"   q_{J} from quadrature = {mpm.nstr(v, 40)}   exact {q[J]}")
dq = max(abs(v - mpm.mpf(q[J].p) / q[J].q) for J, v in zip(Js, qs))
chk("A.1 q_J solved from quadrature agree with exact q_J to 1e-35", dq < mpm.mpf(10) ** -35, mpm.nstr(dq, 3))

# ---------------------------------------------------------------- A.2: Hilbert-Schmidt spin-6 factor
Kspin = 6
def TKM(K, M):
    T = sp.zeros(n1, n1)
    for a in range(n1):
        for b in range(n1):
            m, mp_ = a - j, b - j
            T[a, b] = (-1) ** (j - mp_) * cg(j, m, j, -mp_, K, M)
    return T
Ts = {M: TKM(Kspin, M) for M in range(-Kspin, Kspin + 1)}
# orthonormality of T^K_M (HS)
on = all(sp.simplify((Ts[M].H * Ts[Mp_]).trace() - (1 if M == Mp_ else 0)) == 0 for M in Ts for Mp_ in Ts)
chk(f"T^{Kspin}_M orthonormal in the Hilbert-Schmidt product", on)
# Exact route. ||(ww^H)_K||^2 = sum_M |w^H T_M^H w|^2 = (w x w)^H A (w x w), A = sum_M T_M^H (x) T_M.
# If its symmetrization is sum_J c_J Pi_J, then c_J = tr(Pi_J A) / (2J+1).  Every term of the trace is a
# product of four CG coefficients, i.e. +-sqrt(rational); sympy collects like surds exactly.
def sq_sign(c):
    return (sp.Rational(sp.expand(c ** 2)), 0 if c == 0 else (1 if c > 0 else -1))
Tsq = {M: [[sq_sign(Ts[M][a, b]) for b in range(n1)] for a in range(n1)] for M in Ts}
tab = cg_table(n)
Js = sym_channels(n)
cJ = []
for J in Js:
    tot = sp.Integer(0)
    for Mp_ in range(-J, J + 1):
        L = [(k1, k2, sq_sign(c)) for k1, k2, c in tab[J][Mp_]]
        for m1, m2, (c1, s1) in L:
            for n1_, n2_, (c2, s2) in L:
                for M in Ts:
                    t1, u1 = Tsq[M][n1_][m1]   # (T_M^H)_{m1 n1} = T_M[n1, m1] (T real)
                    t2, u2 = Tsq[M][m2][n2_]
                    sg = s1 * s2 * u1 * u2
                    if sg:
                        tot += sg * sp.sqrt(c1 * c2 * t1 * t2)
    tot = sp.expand(tot)
    cJ.append(tot / (2 * J + 1) + (sp.Rational(1, 10 ** 4) if PLANT == "hsval" and J == 5 else 0))
print("   HS spin-6 factor = sum_J c_J f_J, exact traces: c_J =", dict(zip(Js, map(str, cJ))))
chk("c_J exact traces are rational", all(sp.sympify(c).is_Rational for c in cJ))
# high-precision route: evaluate the HS factor and the f_J at all test points to 50 digits
mpm.mp.dps = 50
def HS_num(w):
    wv = [mpm.mpmathify(sp.N(x, 70)) for x in w]
    tot = mpm.mpf(0)
    for M in Ts:
        z = mpm.fsum(mpm.conj(wv[a]) * mpm.mpmathify(sp.N(Ts[M][b, a], 70)) * wv[b] for a in range(n1) for b in range(n1) if Ts[M][b, a] != 0)
        tot += abs(z) ** 2
    nw2 = mpm.fsum(abs(x) ** 2 for x in wv)
    return tot / nw2 ** 2
dHS = mpm.mpf(0)
for i, t in enumerate(tests):
    hsn = HS_num(t)
    comb = mpm.fsum(mpm.mpmathify(sp.N(c * F[i][J], 70)) for c, J in zip(cJ, Js))
    dHS = max(dHS, abs(hsn - comb))
print("   max |HS(w) - sum c_J f_J(w)| over the 7 test points (50 digits):", mpm.nstr(dHS, 3))
okHS = dHS < mpm.mpf(10) ** -45
chk("HS factor equals sum_J c_J f_J at 7 test points to 1e-45", okHS)
# Q - 1 = sum (q_J - 1) f_J ; proportional to HS iff vectors (q_J - 1) and c_J proportional
v1 = [q[J] - 1 for J in Js]
ratios = [sp.nsimplify(a / b) if b != 0 else (sp.oo if a != 0 else sp.nan) for a, b in zip(v1, cJ)]
print("   (q_J - 1) / c_J by channel:", dict(zip(Js, map(str, ratios))))
prop = all(sp.simplify(a * cJ[0] - b * v1[0]) == 0 for a, b in zip(v1, cJ))
print("A.2 Q - 1 proportional to the HS spin-6 factor in R2:", prop)
beta = None
if prop:
    beta = ratios[0]
    print("A.2 beta_R2 =", beta)
if prop:
    bt = beta + (sp.Rational(1, 1000) if PLANT == "beta" else 0)
    dB = mpm.mpf(0)
    for i, t in enumerate(tests):
        Qe_ = sum(q[J] * F[i][J] for J in q)
        dB = max(dB, abs(mpm.mpmathify(sp.N(Qe_ - 1, 60)) - mpm.mpmathify(sp.N(bt, 60)) * HS_num(t)))
    chk("A.2 Q - 1 = beta * HS spin-6 factor at the 7 test points (50 digits, HS evaluated directly)", dB < mpm.mpf(10) ** -45, mpm.nstr(dB, 3))

# ---------------------------------------------------------------- A.6
fv = F[-1]
qtop = sum(q[J] * fv[J] for J in q)
qtop = sp.radsimp(sp.expand(qtop))
print("A.6 q = Q(v_7/2) =", qtop, "=", sp.N(qtop, 30), "; quadrature:", mpm.nstr(Qnum[-1], 30))
chk("A.6 Q(v_7/2) exact vs quadrature agree to 1e-40", abs(Qnum[-1] - mpm.mpmathify(sp.N(qtop, 60))) < mpm.mpf(10) ** -40)
x1 = sp.sqrt((qtop - 1) / 2) * 100
x2 = (qtop - 1) / 2 * 100
def round_up_2sig(x):
    x = sp.nsimplify(x)
    if x <= 0:
        return None
    e = int(sp.floor(sp.log(x, 10))) - 1
    return sp.ceiling(x / sp.Integer(10) ** e) * sp.Integer(10) ** e
def round_near_2sig(x):
    x = sp.nsimplify(x)
    e = int(sp.floor(sp.log(x, 10))) - 1
    return sp.floor(x / sp.Integer(10) ** e + sp.Rational(1, 2)) * sp.Integer(10) ** e
print("A.6 sqrt((q-1)/2) in % =", sp.N(x1, 20), "-> rounded up (ceiling) to 2 sig:", sp.N(round_up_2sig(x1)) if qtop > 1 else None,
      "; nearest:", sp.N(round_near_2sig(x1)) if qtop > 1 else None)
print("A.6 (q-1)/2 in %       =", sp.N(x2, 20), "-> rounded up (ceiling) to 2 sig:", sp.N(round_up_2sig(x2)) if qtop > 1 else None,
      "; nearest:", sp.N(round_near_2sig(x2)) if qtop > 1 else None)
chk("A.6 q > 1 so that the square root is real", qtop > 1)

out = dict(qJ={str(J): str(v) for J, v in q.items()}, HS6_cJ={str(J): str(c) for J, c in zip(Js, cJ)},
           proportional=bool(prop), beta=str(beta) if beta is not None else None,
           q_top=str(qtop), sqrt_pct=str(sp.N(x1, 30)), half_pct=str(sp.N(x2, 30)),
           sqrt_pct_up=str(sp.N(round_up_2sig(x1))) if qtop > 1 else None, half_pct_up=str(sp.N(round_up_2sig(x2))) if qtop > 1 else None,
           sqrt_pct_near=str(sp.N(round_near_2sig(x1))) if qtop > 1 else None, half_pct_near=str(sp.N(round_near_2sig(x2))) if qtop > 1 else None,
           quad_vs_exact_maxdiff=mpm.nstr(maxdiff, 3), quad_qJ=[mpm.nstr(v, 45) for v in qs])
if PLANT is None:
    json.dump(out, open(rel("out_A1_R2.json"), "w"), indent=1)
chk.summary()
