"""Block A in R4 and R5: A.1 (channel coefficients), A.2 (beta), A.3, A.6.

Exact route: Q(w) = <|phi_w|^4>_Haar / <|phi_w|^2>_Haar^2 (the V's and s_N cancel).  With
|phi_w(g)|^2 = <x_g, P x_g>, x_g = D(g^-1) conj(w), the Haar average of (D x (x) D x)(D x (x) D x)^+ is
sum_J Pi_J (.) Pi_J / (2J+1), so  <|phi|^4> = sum_J tr(Pi_J (P (x) P))/(2J+1) ||[w (x) w]_J||^2 and
q_J = tr(Pi_J (P (x) P)) / ((2J+1) s_N^2), s_N = dim/7 (item 0(c)).
Numeric route: Q from its definition by exact-for-the-integrand quadrature at 50 digits (and 40)
at random points, then q_J by solving the linear system Q = sum q_J f_J."""
import os, sys, json, random
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import *
from quad import avg_moments
import mpmath as mp

chk = Checker()
OUT = os.path.join(HERE, "out")
itw = json.load(open(os.path.join(OUT, "intertwiners.json")))
j = 3
MS = list(range(3, -4, -1))          # index a <-> m = 3 - a
JS = [6, 4, 2, 0]
CG = {}
for J in range(0, 7):
    for m1 in MS:
        for m2 in MS:
            M = m1 + m2
            if abs(M) <= J:
                CG[(m1, m2, J)] = cg(3, m1, 3, m2, J, M)

def couple(a, b, J):
    """[a (x) b]_J components, M = J..-J; a, b indexed by m via MS order."""
    out = []
    for M in range(J, -J - 1, -1):
        t = 0
        for i1, m1 in enumerate(MS):
            m2 = M - m1
            if abs(m2) <= 3:
                t += CG[(m1, m2, J)] * a[i1] * b[MS.index(m2)]
        out.append(t)
    return out

def nrm2(v): return sp.expand(sum(sp.expand(x * sp.conjugate(x)) for x in v))

def fJ_exact(w):
    n2 = nrm2(w)
    return {J: sp.radsimp(sp.expand(nrm2(couple(w, w, J))) / n2 ** 2) for J in JS}

def theta(u):
    return [(-1) ** (3 - MS[a]) * sp.conjugate(u[6 - a]) for a in range(7)]

def hs6_exact(w):
    """||(w w^+)_6||^2 / ||w||^4 with (T^6_M)_{mm'} = (-1)^{j-m'} <j m; j -m' | 6 M>."""
    tot = 0
    for M in range(-6, 7):
        c = 0
        for i, m in enumerate(MS):
            for k, mp_ in enumerate(MS):
                if m - mp_ == M:
                    T = (-1) ** (3 - mp_) * CG[(m, -mp_, 6)]
                    c += sp.conjugate(T) * w[i] * sp.conjugate(w[k])     # tr(T^+ w w^+)
        tot += sp.expand(c * sp.conjugate(c))
    return sp.radsimp(sp.expand(tot) / nrm2(w) ** 2)

def r6_exact(w):
    return sp.radsimp(sp.expand(nrm2(couple(w, theta(w), 6))) / nrm2(w) ** 2)

def qJ_exact(P, d):
    sN = sp.Rational(d, 7)
    out = {}
    for J in range(0, 7):
        t = 0
        for M in range(-J, J + 1):
            for m1 in MS:
                m2 = M - m1
                if abs(m2) > 3: continue
                c1 = CG[(m1, m2, J)]
                if c1 == 0: continue
                for n1 in MS:
                    n2 = M - n1
                    if abs(n2) > 3: continue
                    c2 = CG[(n1, n2, J)]
                    if c2 == 0: continue
                    t += c1 * c2 * P[MS.index(m1), MS.index(n1)] * P[MS.index(m2), MS.index(n2)]
        out[J] = sp.radsimp(sp.expand(t)) / ((2 * J + 1) * sN ** 2)
        assert out[J].is_Rational, out[J]
    return out

results = {"A.1": {}, "A.2": {}, "A.6": {}, "A.3": {}}
random.seed(20261116)
# test points with Gaussian-integer weighted coordinates p_m, i.e. w_m = sqrt(f_m) p_m, so f_J are rational
pts = [[sp.sqrt(fm(6, Fr(m))) * (sp.Integer(random.randint(-3, 3)) + sp.I * random.randint(-3, 3)) for m in MS] for _ in range(6)]
pts.append([sp.Integer(1)] + [0] * 6)            # v_3
fpts = [fJ_exact(w) for w in pts]
chk("f_J are rational at all 7 test points", all(f[J].is_Rational for f in fpts for J in JS))
chk("f_J sum to 1 at all 7 test points", all(sum(f.values()) == 1 for f in fpts))
chk("f_J at v_3 is (1,0,0,0)", fpts[-1] == {6: 1, 4: 0, 2: 0, 0: 0})
Fm = sp.Matrix([[f[J] for J in JS] for f in fpts])
chk("the four f_J are linearly independent functions (rank 4 at the test points)", Fm.rank() == 4)

# Hilbert-Schmidt spin-6 factor in the f_J basis: exact solve on 4 points, verified on the rest
hs = [hs6_exact(w) for w in pts]
aJ = Fm[:4, :].LUsolve(sp.Matrix(hs[:4]))
aJ = {J: sp.radsimp(aJ[i]) for i, J in enumerate(JS)}
assert all(v.is_Rational for v in aJ.values())
print("HS spin-6 factor = sum_J a_J f_J with a_J =", aJ)
pred = [sum(aJ[J] * f[J] for J in JS) for f in fpts]
chk("HS6 = sum a_J f_J verified at the 3 held-out points (exact)", all(sp.simplify(pred[i] - hs[i]) == 0 for i in range(4, 7)))
r6 = [r6_exact(w) for w in pts]
chk("r6hat = HS6 at all 7 test points (exact) [see B.5]", all(sp.simplify(r6[i] - hs[i]) == 0 for i in range(7)))

betas = {}
for name in ("R4", "R5"):
    d = itw[name]["dim"]
    P = sp.Matrix([[sp.sympify(x) for x in r] for r in itw[name]["P"]])
    qall = qJ_exact(P, d)
    print(f"\n== A.1 {name}: exact channel coefficients (all J incl. odd, which never enter Q):")
    for J in range(6, -1, -1): print(f"   J={J}: tr(Pi_J P(x)P)/((2J+1) s_N^2) = {qall[J]}")
    qe = {J: qall[J] for J in JS}
    # numeric: Q from its definition by quadrature at 50 digits, at the 7 points
    mp.mp.dps = 50
    Pn = mp.matrix([[mp.mpmathify(sp.N(P[a, b], 70)) for b in range(7)] for a in range(7)])
    Qnum = []
    for w in pts:
        wn = [mp.mpc(*[mp.mpf(str(t)) for t in sp.N(x, 70).as_real_imag()]) for x in w]
        s2, s4 = avg_moments(Pn, wn)
        Qnum.append(s4 / s2 ** 2)
    for i, (w, Qv) in enumerate(zip(pts, Qnum)):
        ex = sum(qe[J] * fpts[i][J] for J in JS)
        print(f"   point {i}: Q quadrature(50 dig) = {mp.nstr(Qv, 40)}   exact sum q_J f_J = {ex}")
    # fit q_J from the first 4 quadrature values
    A = mp.matrix([[mp.mpf(sp.Rational(fpts[i][J]).p) / sp.Rational(fpts[i][J]).q for J in JS] for i in range(4)])
    sol = mp.lu_solve(A, mp.matrix(Qnum[:4]))
    print("   q_J fitted from quadrature at 4 points:", [mp.nstr(sol[k], 40) for k in range(4)])
    for k, J in enumerate(JS):
        ident = mp.identify(sol[k], tol=mp.mpf(10) ** -40)
        print(f"     J={J}: fitted {mp.nstr(sol[k], 45)}  exact {qe[J]}  diff {mp.nstr(abs(sol[k] - mp.mpf(qe[J].p) / qe[J].q), 3)}")
    maxdiff = max(abs(sol[k] - mp.mpf(qe[J].p) / qe[J].q) for k, J in enumerate(JS))
    chk(f"{name}: quadrature-fitted q_J agree with exact q_J to 1e-40", maxdiff < mp.mpf(10) ** -40)
    hold = max(abs(Qnum[i] - mp.mpf(sp.Rational(sum(qe[J] * fpts[i][J] for J in JS)).p) / sp.Rational(sum(qe[J] * fpts[i][J] for J in JS)).q) for i in range(4, 7))
    chk(f"{name}: exact q_J predict quadrature Q at the 3 held-out points to 1e-40", hold < mp.mpf(10) ** -40)
    planted(f"{name}: held-out test with q_6 off by 1e-30",
            lambda: max(abs(Qnum[i] - (mp.mpf(sp.Rational(sum(qe[J] * fpts[i][J] for J in JS)).p) / sp.Rational(sum(qe[J] * fpts[i][J] for J in JS)).q + mp.mpf(10) ** -30 * mp.mpf(sp.Rational(fpts[i][6]).p) / sp.Rational(fpts[i][6]).q)) for i in range(4, 7)) < mp.mpf(10) ** -40)
    # second precision: one point at 40 digits
    mp.mp.dps = 40
    Pn40 = mp.matrix([[mp.mpmathify(sp.N(P[a, b], 60)) for b in range(7)] for a in range(7)])
    s2, s4 = avg_moments(Pn40, [mp.mpc(*[mp.mpf(str(t)) for t in sp.N(x, 60).as_real_imag()]) for x in pts[0]])
    chk(f"{name}: Q at point 0 at 40 digits agrees with exact to 1e-35",
        abs(s4 / s2 ** 2 - mp.mpf(sp.Rational(sum(qe[J] * fpts[0][J] for J in JS)).p) / sp.Rational(sum(qe[J] * fpts[0][J] for J in JS)).q) < mp.mpf(10) ** -35)
    mp.mp.dps = 50
    results["A.1"][name] = {f"q_{J}": str(qe[J]) for J in JS}

    # A.2
    ratios = {J: (qe[J] - 1) / aJ[J] if aJ[J] != 0 else None for J in JS}
    print(f"== A.2 {name}: (q_J - 1)/a_J =", ratios)
    vals = set(ratios.values())
    prop = (None not in vals) and len(vals) == 1
    if not prop:
        # handle a_J = 0 entries: proportional iff q_J - 1 = 0 there
        prop = all((aJ[J] == 0 and qe[J] == 1) or aJ[J] != 0 for J in JS) and len(set(v for v in vals if v is not None)) == 1
    chk(f"{name}: Q - 1 is a constant multiple of HS6 (exact, all J)", prop)
    qbad = dict(qe); qbad[4] = qbad[4] + sp.Rational(1, 10**6)
    planted(f"{name}: proportionality test with q_4 shifted by 1e-6",
            lambda: len(set((qbad[J] - 1) / aJ[J] for J in JS)) == 1)
    beta = [v for v in vals if v is not None][0] if prop else None
    print(f"   beta_{name} = {beta}")
    betas[name] = beta
    results["A.2"][name] = {"proportional": prop, "beta": str(beta)}
    # A.6
    q = qe[6]
    Qv3 = Qnum[-1]
    chk(f"{name}: quadrature Q(v_3) = q_6 to 1e-40", abs(Qv3 - mp.mpf(q.p) / q.q) < mp.mpf(10) ** -40)
    a = (q - 1) / 2
    sq = sp.sqrt(a)
    def ceil2(x):
        x = mp.mpf(x)
        e = int(mp.floor(mp.log10(x)))
        scale = mp.mpf(10) ** (e - 1)
        return mp.ceil(x / scale) * scale
    pct_sqrt = ceil2(sp.N(100 * sq, 50))
    pct_lin = ceil2(sp.N(100 * a, 50))
    print(f"== A.6 {name}: q = Q(v_3) = {q} = {sp.N(q, 30)}")
    print(f"   sqrt((q-1)/2) = {sp.nsimplify(sq)} = {sp.N(100*sq, 20)} %  -> rounded up to 2 s.f.: {mp.nstr(pct_sqrt, 3)} %")
    print(f"   (q-1)/2 = {a} = {sp.N(100*a, 20)} %  -> rounded up to 2 s.f.: {mp.nstr(pct_lin, 3)} %")
    results["A.6"][name] = {"q": str(q), "sqrt((q-1)/2)": str(sp.radsimp(sq)), "(q-1)/2": str(a),
                            "sqrt((q-1)/2) percent, rounded up 2 s.f.": mp.nstr(pct_sqrt, 3),
                            "(q-1)/2 percent, rounded up 2 s.f.": mp.nstr(pct_lin, 3)}
    results["A.1"][name]["_all_J_incl_odd"] = {str(J): str(qall[J]) for J in range(7)}

print("\n== A.3")
r = betas["R5"] / betas["R4"]; dlt = betas["R4"] - betas["R5"]
print("   beta_R5/beta_R4 =", r, "   beta_R4 - beta_R5 =", dlt)
results["A.3"] = {"beta_R5/beta_R4": str(r), "beta_R4 - beta_R5": str(dlt)}
results["_aJ_HS6"] = {str(J): str(aJ[J]) for J in JS}
json.dump(results, open(os.path.join(OUT, "blockA.json"), "w"), indent=1)
chk.summary()
