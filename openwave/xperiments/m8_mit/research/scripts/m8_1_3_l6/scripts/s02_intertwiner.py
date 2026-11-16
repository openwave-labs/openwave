"""Item 0(c) for R4 and R5 (and 0(d)).

Exact: the isotypic projector P_rho = (dim rho/|G|) sum_h conj(chi_rho(h)) D^3(h), computed on
weighted coordinates with entries in Q(sqrt5, i), then moved to Condon-Shortley components.
eta: orthonormal basis of range(P) (Gram-Schmidt on columns of P, exact in sympy), so eta eta^+ = P,
eta^+ eta = I, and rho(h) := eta^+ D(h) eta.
s_N: exact by Schur orthogonality, s_N = dim rho/(2j+1); checked by an exact-for-this-integrand
quadrature on SU(2) (Euler angles) at 50 digits, independent of Schur."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import *
import mpmath as mp

chk = Checker()
OUT = os.path.join(HERE, "out")
grp = json.load(open(os.path.join(OUT, "group.json")))
order = grp["order"]
G = generate_group([q1, q2])
classes = conj_classes(G)
retr_of = {}
for c in classes:
    pass
def q5s(t): return Q5(Fr(t[0]), Fr(t[1]))
class_retr = [q5s(t) for t in grp["class_retr"]]
chars = [[q5s(t) for t in c] for c in grp["chars"]]

def char_of(g, idx):
    # characters are class functions; in this group the class is determined by Re g (checked in s01)
    k = class_retr.index(g[0])
    return chars[idx][k]

def projector_p(idx, j2):
    d = chars[idx][0]
    n = j2 + 1
    P = [[C5(0) for _ in range(n)] for _ in range(n)]
    for g in G:
        M = mono_matrix(su2(g), j2, C5(0), C5(1))
        c = char_of(g, idx)            # real
        for a in range(n):
            for b in range(n):
                P[a][b] = P[a][b] + M[a][b] * c
    fac = d / order
    return [[P[a][b] * C5(fac) for b in range(n)] for a in range(n)]

def to_cs(Pp, j2):
    ms = mvals(j2)
    n = j2 + 1
    return sp.Matrix(n, n, lambda a, b: sp.nsimplify(Pp[a][b].sym()) * sp.sqrt(sp.Rational(fm(j2, ms[a]), fm(j2, ms[b]))))

def gram_schmidt_range(P, d):
    cols = []
    for k in range(P.shape[1]):
        v = P[:, k]
        for u in cols:
            v = v - u * (u.H * v)[0]
        v = v.applyfunc(sp.radsimp)
        nn = sp.radsimp(sp.sqrt(sp.nsimplify((v.H * v)[0].expand())))
        if nn != 0:
            cols.append((v / nn).applyfunc(sp.radsimp))
        if len(cols) == d: break
    return sp.Matrix.hstack(*cols)

res = {}
j2 = 6
gens_cs = []
for g in (q1, q2):
    Mg = mono_matrix(su2(g), j2, C5(0), C5(1))
    ms = mvals(j2)
    gens_cs.append(sp.Matrix(7, 7, lambda a, b: Mg[a][b].sym() * sp.sqrt(sp.Rational(fm(j2, ms[a]), fm(j2, ms[b])))))
for name in ("R4", "R5"):
    idx = grp["slots"][name]
    d = int(grp["dims"][idx])
    print(f"== 0(c) {name}: irreducible #{idx}, dim {d}, in V_3")
    Pp = projector_p(idx, j2)
    P = to_cs(Pp, j2).applyfunc(sp.radsimp)
    chk(f"{name}: P^2 = P (exact)", (P * P - P).applyfunc(lambda t: sp.radsimp(sp.expand(t))) == sp.zeros(7))
    chk(f"{name}: P Hermitian (exact)", (P - P.H).applyfunc(lambda t: sp.radsimp(sp.expand(t))) == sp.zeros(7))
    trP = sp.radsimp(P.trace())
    print("  multiplicity at level 6 = tr(P)/dim =", trP / d)
    chk(f"{name}: multiplicity one at level 6 (tr P = dim rho)", trP == d)
    for gi, Dg in enumerate(gens_cs):
        chk(f"{name}: P commutes with D^3(q{gi+1}) (exact)",
            (Dg * P - P * Dg).applyfunc(lambda t: sp.radsimp(sp.expand(t))) == sp.zeros(7))
    eta = gram_schmidt_range(P, d)
    EtE = (eta.H * eta).applyfunc(lambda t: sp.radsimp(sp.expand(t)))
    chk(f"{name}: eta^+ eta = I (exact)", EtE == sp.eye(d))
    EEt = (eta * eta.H - P).applyfunc(lambda t: sp.radsimp(sp.expand(t)))
    chk(f"{name}: eta eta^+ = P (exact)", EEt == sp.zeros(7))
    rhos = []
    for gi, Dg in enumerate(gens_cs):
        rho = (eta.H * Dg * eta).applyfunc(lambda t: sp.radsimp(sp.expand(t)))
        rhos.append(rho)
        lhs = (Dg * eta - eta * rho).applyfunc(lambda t: sp.radsimp(sp.expand(t)))
        chk(f"{name}: D^3(q{gi+1}) eta = eta rho(q{gi+1}) (exact)", lhs == sp.zeros(7, d))
        tr = sp.radsimp(rho.trace())
        want = chars[idx][class_retr.index((q1, q2)[gi][0])].sym()
        chk(f"{name}: tr rho(q{gi+1}) = chi(q{gi+1}) = {sp.nsimplify(want)}", sp.radsimp(tr - want) == 0)
    # planted: a wrong projector (other irreducible's character) must fail the commuting-with-eta test
    print("  eta (columns; rows m = 3..-3), exact:")
    for a in range(7):
        print("   m=%2d:" % (3 - a), [str(sp.nsimplify(eta[a, b])) for b in range(d)])
    print("  P (exact, rows/cols m = 3..-3):")
    for a in range(7):
        print("   ", [str(P[a, b]) for b in range(7)])
    res[name] = {"P": [[str(P[a, b]) for b in range(7)] for a in range(7)],
                 "eta": [[str(eta[a, b]) for b in range(d)] for a in range(7)],
                 "dim": d, "multiplicity": int(trP / d),
                 "rho_q1": [[str(rhos[0][a, b]) for b in range(d)] for a in range(d)],
                 "rho_q2": [[str(rhos[1][a, b]) for b in range(d)] for a in range(d)]}

# planted defect: the intertwining test must fail for a non-intertwiner (perturb eta)
eta4 = sp.Matrix([[sp.sympify(x) for x in r] for r in res["R4"]["eta"]])
bad = eta4.copy(); bad[0, 0] = bad[0, 0] + sp.Rational(1, 10)
def intertw(e):
    rho = e.H * gens_cs[0] * e
    return (gens_cs[0] * e - e * rho).applyfunc(lambda t: sp.radsimp(sp.expand(t))) == sp.zeros(7, e.shape[1])
planted("intertwining test with eta[3,0] shifted by 1/10", lambda: intertw(bad))

# ------------------------------------------------ s_N: Schur (exact) and quadrature (numeric)
print("\n== s_N")
mp.mp.dps = 50
from quad import Dcs_num, quad_nodes, avg_moments

if True:
    nodes, Wb = quad_nodes(25)
    chk("beta-weights integrate the constant to 1", abs(sum(Wb) - 1) < mp.mpf(10) ** -45)
    json.dump(res, open(os.path.join(OUT, "intertwiners.json"), "w"), indent=1)
    import random
    random.seed(1)
    wtest = [mp.mpc(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(7)]
    nw2 = sum(abs(x) ** 2 for x in wtest)
    for name in ("R4", "R5"):
        Pn = mp.matrix([[mp.mpc(complex(sp.N(sp.sympify(x), 60))) if False else mp.mpmathify(sp.N(sp.sympify(x), 60)) for x in r] for r in res[name]["P"]])
        s2, s4 = avg_moments(Pn, wtest)
        sN = sp.Rational(res[name]["dim"], 7)
        print(f"  {name}: s_N (Schur, exact) = {sN};  quadrature <|phi_w|^2>/||w||^2 = {mp.nstr(s2 / nw2, 45)}")
        chk(f"{name}: quadrature s_N agrees with dim/7 to 1e-40", abs(s2 / nw2 - mp.mpf(res[name]['dim']) / 7) < mp.mpf(10) ** -40)
        res[name]["s_N"] = str(sN)
        planted(f"{name}: same quadrature check against a wrong s_N = dim/8",
                lambda: abs(s2 / nw2 - mp.mpf(res[name]['dim']) / 8) < mp.mpf(10) ** -40)
    json.dump(res, open(os.path.join(OUT, "intertwiners.json"), "w"), indent=1)

# ------------------------------------------------ 0(d)
print("\n== 0(d)")
c = cg(3, 3, 3, -3, 6, 0)
print("  <3 3; 3 -3 | 6 0> =", c, "=", sp.N(c, 20), " sign:", "+" if c > 0 else "-")
chk("<3 3;3 -3|6 0> = +sqrt(1/924)", sp.simplify(c - 1 / sp.sqrt(924)) == 0)
# CG orthonormality spot check (columns of the CG matrix for j1=j2=3)
def cg_orth(j, J1, J2, M):
    s = 0
    for m1 in range(-j, j + 1):
        m2 = M - m1
        if abs(m2) <= j:
            s += cg(j, m1, j, m2, J1, M) * cg(j, m1, j, m2, J2, M)
    return sp.nsimplify(s)
ok = all(cg_orth(3, J1, J2, M) == (1 if J1 == J2 else 0) for M in (0, 1, 2) for J1 in range(abs(M), 7) for J2 in range(abs(M), 7))
chk("CG(3x3) orthonormality for M = 0, 1, 2 (exact)", ok)
cg_bad = lambda *a: cg(*a) * (2 if a[4] == 4 and a[1] == 0 else 1)
planted("CG orthonormality with one coefficient doubled", lambda: sp.nsimplify(sum(cg_bad(3, m1, 3, -m1, 4, 0) ** 2 for m1 in range(-3, 4))) == 1)

def theta(u, j2):
    ms = mvals(j2); n = j2 + 1
    # (Theta u)_m = (-1)^{j-m} conj(u_{-m}); index a <-> m = j - a, -m <-> index n-1-a
    return [(-1) ** int(Fr(j2, 2) - ms[a]) * sp.conjugate(u[n - 1 - a]) for a in range(n)]
for j2 in (6, 7):
    u = sp.symbols(f"u0:{j2+1}")
    t2 = theta(theta(list(u), j2), j2)
    ratios = set(sp.simplify(t2[a] / u[a]) for a in range(j2 + 1))
    print(f"  Theta^2 on V_{{{j2}/2}} = {ratios}")
    chk(f"Theta^2 on V_{j2}/2 is (-1)^(2j) = {(-1)**j2}", ratios == {(-1) ** j2})
chk.summary()
