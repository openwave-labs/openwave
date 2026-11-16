"""Items A.4 (critical?, Q, signature) and B.3 (the B.1 list without r6, plus the four members) in R2,
at the eight points of inputs.json "R2 points".  Exact route: sympy over the radicals of the point,
characteristic polynomials over Q after the diagonal similarity N (w = N p); signs of real-rooted
polynomials by exact root isolation.  High-precision route: mpmath (50 digits) eigenvalues, SVD ranks,
and the Krein forms.
Usage: ./py sB3.py [--plant=NAME]   planted defects: thetasign (Theta without its signs), nu (shift a cluster nu by 1e-20),
       charpoly (perturb the exact matrix by 1e-6 in one entry), numeric (perturb the 50-digit Mhat by 1e-10),
       sturm (miscount positive roots), kvec (corrupt a tangent vector)
"""
import sys, json
import mpmath as mpm
from sympy.polys.matrices import DomainMatrix
from lib import *
from quartic import h_data, mat_i, mat_theta, rotation_generators_real, to_real

PLANT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--plant=")), None)
chk = Checker()
n = 7; n1 = 8; dimR = 16; j = sp.Rational(7, 2)
qd = json.load(open(rel("out_A1_R2.json")))["qJ"]
q = {int(J): sp.Rational(v) for J, v in qd.items()}
if PLANT == "q":
    q[3] += sp.Rational(1, 100)
pts = json.load(open(rel("inputs.json")))["R2 points"]["points"]
order = ["weight 7/2", "weight 5/2", "weight 3/2", "weight 1/2",
         "line {7/2, -1/2}", "line {7/2, -3/2}", "line {7/2, -5/2}", "line {5/2, -3/2}"]
assert sorted(order) == sorted(pts.keys())
mpm.mp.dps = 50
Nk = [sp.sqrt(sp.factorial(k) * sp.factorial(n - k)) for k in range(n1)] * 2
I0 = mat_i(n); Th = mat_theta(n)
if PLANT == "thetasign":
    Th = Th.applyfunc(abs) * sp.diag(*([1] * n1 + [-1] * n1))
Kk = Th * I0
Xs = rotation_generators_real(n)
chk("i^2 = J^2 = k^2 = -1 and iJ = -Ji (real 16x16 matrices)",
    I0 * I0 == -sp.eye(dimR) and Th * Th == -sp.eye(dimR) and Kk * Kk == -sp.eye(dimR) and I0 * Th == -Th * I0)
members = {"J": (Th, 1), "k": (Kk, 1), "(i+J)/sqrt2": (I0 + Th, 2), "(i+k)/sqrt2": (I0 + Kk, 2)}
for nm, (A, sc) in members.items():
    chk(f"member {nm}: tau^2 = -1", A * A == -sc * sp.eye(dimR))
group_tangent = Xs + [I0, Th, Kk]
# h is invariant under i, J = Theta and k (the fibre group): check at a random point, 50 digits
from quartic import cg_table
_tab = cg_table(n)
def h_num(wr):
    w = [wr[k] + 1j * 0 + mpm.mpc(0, 1) * wr[n1 + k] for k in range(n1)]
    tot = mpm.mpf(0)
    for J, qJ in q.items():
        for M in range(-J, J + 1):
            G = mpm.fsum(mpm.mpmathify(sp.N(c, 60)) * w[k1] * w[k2] for k1, k2, c in _tab[J][M])
            tot += mpm.mpf(qJ.p) / qJ.q * abs(G) ** 2
    return tot
import random
random.seed(3)
wr = [mpm.mpf(random.randint(-9, 9)) / 7 for _ in range(dimR)]
h0 = h_num(wr)
for nm_, A_ in (("i", I0), ("J", Th), ("k", Kk)):
    Am = [[mpm.mpf(int(A_[a, b])) for b in range(dimR)] for a in range(dimR)]
    w2 = [mpm.fsum(Am[a][b] * wr[b] for b in range(dimR)) for a in range(dimR)]
    chk(f"h(tau w) = h(w) for tau = {nm_} at a random w (50 digits)", abs(h_num(w2) - h0) < mpm.mpf(10) ** -40, mpm.nstr(abs(h_num(w2) - h0), 3))

nu = sp.Symbol("nu")
xi = sp.Symbol("xi")


def sim(Mw):
    """N Mw N^-1: rational after the similarity at these points (checked)."""
    return sp.Matrix(dimR, dimR, lambda a, b: sp.radsimp(Mw[a, b] * Nk[a] / Nk[b]))


def charpoly(A):
    rational = all(x.is_Rational for x in A)
    dm = DomainMatrix.from_Matrix(A) if rational else DomainMatrix.from_Matrix(A, extension=True)
    co = dm.charpoly()
    return [sp.radsimp(dm.domain.to_sympy(c)) for c in co]


def split_zero(co):
    """coefficients of det(x - A), highest first -> (zero multiplicity, list of coefficients of the rest)."""
    z = 0
    while co and co[-1] == 0:
        co = co[:-1]; z += 1
    return z, co


def slow_poly(co_rest, scale=1):
    """co_rest = coefficients of R(xi), even in xi; return the polynomial in nu = xi^2 (with
    nu -> scale*nu: eigenvalues of A/sqrt(scale)), primitive integer when rational, else monic."""
    deg = len(co_rest) - 1
    assert deg % 2 == 0
    for t in range(1, deg + 1, 2):
        assert co_rest[t] == 0, "odd coefficient nonzero"
    co = [sp.radsimp(co_rest[2 * t] * scale ** (deg // 2 - t)) for t in range(deg // 2 + 1)]
    if deg == 0:
        return sp.Poly(1, nu), True
    if all(c.is_Rational for c in co):
        P = sp.Poly(co, nu)
        den = sp.ilcm(*[c.q for c in co])
        P = sp.Poly([c * den for c in co], nu)
        _, P = P.primitive()
        if P.LC() < 0:
            P = -P
        return P, True
    lc = co[0]
    co = [sp.radsimp(c / lc) for c in co]
    return sp.Poly(co, nu, extension=True), False


def sturm_counts(fp):
    """(number of real roots, number of positive roots) of a squarefree fp, exactly, by Sturm."""
    seq = sp.sturm(fp)
    def sgn(e):
        e = sp.radsimp(e)
        return 0 if e == 0 else (1 if e > 0 else -1)
    def changes(signs):
        s_ = [x for x in signs if x != 0]
        return sum(1 for a, b in zip(s_, s_[1:]) if a != b)
    at_pinf = [sgn(g.LC()) for g in seq]
    at_minf = [sgn(g.LC()) * (-1) ** g.degree() for g in seq]
    at_0 = [sgn(g.as_expr().subs(nu, 0)) for g in seq]
    return changes(at_minf) - changes(at_pinf), changes(at_0) - changes(at_pinf)


def root_data(P, rational):
    """exact classification of the roots of P in nu (signs, multiplicities), with 50-digit values."""
    if P.degree() <= 0:
        return dict(roots=[], factored="1", rational=rational)
    roots = []
    if rational:
        _, facs = sp.factor_list(P.as_expr(), nu)
    else:
        _, facs = sp.factor_list(P.as_expr(), nu, extension=True)
    for f, e in facs:
        fp = sp.Poly(f, nu) if rational else sp.Poly(f, nu, extension=True)
        nreal, npos = sturm_counts(fp)
        if PLANT == "sturm":
            npos += 1
        co = [mpm.mpmathify(sp.N(c, 80)) for c in fp.all_coeffs()]
        mpm.mp.dps = 80
        nr = mpm.polyroots(co, maxsteps=500, extraprec=400)
        mpm.mp.dps = 50
        nr = sorted(nr, key=lambda z: abs(mpm.im(z)))
        reals = [mpm.re(z) for z in nr[:nreal]]
        cplx = nr[nreal:]
        chk(f"Sturm count of real roots matches numerics for factor of degree {fp.degree()}",
            all(abs(mpm.im(z)) < mpm.mpf(10) ** -40 for z in nr[:nreal]) and all(abs(mpm.im(z)) > mpm.mpf(10) ** -20 for z in cplx)
            and sum(1 for r in reals if r > 0) == npos)
        ex = None
        if fp.degree() <= 2:
            ex = {mpm.nstr(mpm.mpmathify(sp.N(r_, 60)), 20): str(sp.radsimp(r_)) for r_ in sp.roots(f, nu)}
        for r in reals:
            roots.append(dict(kind="neg" if r < 0 else "pos", mult=e, value=mpm.nstr(r, 50), factor=str(f),
                              exact=(ex or {}).get(mpm.nstr(r, 20))))
        for z in cplx:
            roots.append(dict(kind="nonreal", mult=e, value=mpm.nstr(z, 50), factor=str(f), exact=None))
    fact = " * ".join(f"({f})" + (f"^{e}" if e > 1 else "") for f, e in facs)
    return dict(roots=roots, factored=fact, rational=rational)


def match_roots(numeric_nus, rd):
    """greedy nearest matching of numeric nu (from eigenvalues, each root twice per multiplicity)."""
    ex = []
    for r in rd["roots"]:
        ex += [mpm.mpmathify(r["value"])] * (2 * r["mult"])
    if len(ex) != len(numeric_nus):
        return False, None
    rem = list(numeric_nus); worst = mpm.mpf(0)
    for x in ex:
        i = min(range(len(rem)), key=lambda t: abs(rem[t] - x))
        worst = max(worst, abs(rem[i] - x)); rem.pop(i)
    return worst < mpm.mpf(10) ** -12, worst


def words(rd):
    rs = rd["roots"]
    if all(r["kind"] == "neg" for r in rs):
        mults = [r["mult"] for r in rs]
        if all(m == 1 for m in mults):
            return "elliptic, simple"
        if sorted(mults) == [1] * (len(mults) - 1) + [2] and mults.count(2) == 1:
            return "elliptic, one double frequency"
        return "elliptic (neither 'simple' nor 'one double frequency')"
    nonreal = [r for r in rs if r["kind"] == "nonreal"]
    pos = [r for r in rs if r["kind"] == "pos"]
    if not nonreal and len(pos) == 1 and pos[0]["mult"] == 1:
        return "hyperbolic, one real pair"
    if not nonreal and sum(r["mult"] for r in pos) >= 2:
        return "hyperbolic, real pairs"
    if not pos and len(nonreal) == 2 and all(r["mult"] == 1 for r in nonreal):
        return "hyperbolic, a complex quartet"
    return "hyperbolic (no listed sub-word)"


def to_mp_mat(A):
    return mpm.matrix([[mpm.mpmathify(sp.N(A[a, b], 70)) for b in range(A.shape[1])] for a in range(A.shape[0])])


def nullspace_mp(A, expect=None):
    U, Sg, V = mpm.svd_r(A)
    sv = [Sg[t] for t in range(len(Sg))]
    tol = mpm.mpf(10) ** -30
    null = [t for t, s in enumerate(sv) if s < tol]
    nonnull = [s for s in sv if s >= tol]
    gap = (min(nonnull) if nonnull else None, max([sv[t] for t in null]) if null else None)
    B = mpm.matrix(A.cols, len(null))
    for c, t in enumerate(null):
        for r in range(A.cols):
            B[r, c] = V[t, r]
    return B, gap


def krein(Snum, Mnum, rd, label):
    out = []
    for r in rd["roots"]:
        if r["kind"] != "neg":
            continue
        nu0 = mpm.mpmathify(r["value"])
        if PLANT == "nu":
            nu0 += mpm.mpf(10) ** -20
        A = Snum * Snum - nu0 * mpm.eye(dimR)
        B, gap = nullspace_mp(A)
        dimk = B.cols
        if dimk == 0:
            chk(f"{label}: ker(S^2 + tau^2) has dimension 2*mult (semisimple cluster), SVD gap clear", False, "empty kernel")
            continue
        Mr = B.T * Mnum * B
        ev = mpm.eig(Mr, left=False, right=False)
        ev = sorted([mpm.re(e) for e in ev])
        if all(e > 0 for e in ev):
            sign = "+"
        elif all(e < 0 for e in ev):
            sign = "-"
        else:
            sign = "indefinite"
        out.append(dict(nu=mpm.nstr(nu0, 30), tau=mpm.nstr(mpm.sqrt(-nu0), 30), mult=r["mult"], dim_ker=dimk, sign=sign,
                        min_abs_eig=mpm.nstr(min(abs(e) for e in ev), 5), sv_gap=[mpm.nstr(g, 5) if g is not None else None for g in gap]))
        chk(f"{label}: ker(S^2 + tau^2) has dimension 2*mult (semisimple cluster), SVD gap clear", dimk == 2 * r["mult"] and gap[0] > mpm.mpf(10) ** -10,
            f"dim {dimk}, gap {mpm.nstr(gap[0], 3)} / {mpm.nstr(gap[1], 3)}")
    return out


results = {}
for name in order:
    print("=" * 100); print(name)
    comp = pts[name]
    u = [sp.Integer(0)] * n1
    for m, v in comp.items():
        u[int(sp.Rational(m) + j)] = sp.sympify(v)
    if PLANT == "point" and name.startswith("line {5/2"):
        u[int(sp.Rational(5, 2) + j)] = sp.sqrt(sp.Rational(2, 3))
    nrm = sp.sqrt(sum(sp.Abs(x) ** 2 for x in u))
    u = [sp.radsimp(x / nrm) for x in u]
    ur = sp.Matrix(to_real(u))
    h, g, H = h_data(n, q, u)
    crit_res = [sp.radsimp(g[a] - 4 * h * ur[a]) for a in range(dimR)]
    critical = all(x == 0 for x in crit_res)
    print("  u (m: component):", {str(k - j): str(u[k]) for k in range(n1) if u[k] != 0})
    print("  h(u) = Q(u) =", h, "=", sp.N(h, 30))
    print("  critical (grad h = 4 h u exactly):", critical)
    R = dict(Q=str(h), critical=critical)
    results[name] = R
    if not critical:
        print("  residual:", [str(x) for x in crit_res if x != 0][:4])
        continue
    Mw = H - 4 * h * sp.eye(dimR)
    chk(f"{name}: Mhat u = 8 h u", all(sp.radsimp(x) == 0 for x in (Mw * ur - 8 * h * ur)))
    T = [sp.Matrix(G_ * ur).applyfunc(sp.radsimp) for G_ in group_tangent]
    if PLANT == "kvec":
        T[0] = T[0] + sp.Matrix([sp.Rational(1, 10 ** 6)] + [0] * (dimR - 1))
    Tm = sp.Matrix.hstack(*T)
    kdim = Tm.rank(simplify=True)
    Tnum = to_mp_mat(Tm)
    _, gapT = nullspace_mp(Tnum.T * Tnum)
    k_num = Tm.shape[1] - sum(1 for t in range(Tm.shape[1]) if mpm.svd_r(Tnum.T * Tnum, compute_uv=False)[t] < mpm.mpf(10) ** -30)
    chk(f"{name}: k exact rank = k numeric rank", kdim == k_num, f"{kdim} vs {k_num}")
    chk(f"{name}: Mhat vanishes on K_u", all(sp.radsimp(x) == 0 for x in Mw * Tm))
    chk(f"{name}: K_u orthogonal to u", all(sp.radsimp(x) == 0 for x in ur.T * Tm))
    # independent columns of the tangent vectors
    _, piv = Tm.rref(simplify=True)
    B = Tm.extract(list(range(dimR)), list(piv))
    Gram = (B.T * I0 * B).applyfunc(sp.radsimp)
    r_u = kdim - Gram.rank(simplify=True)
    Ms = sim(Mw)
    if PLANT == "charpoly":
        Ms[0, 1] += sp.Rational(1, 10 ** 6)
    print("  N Mhat N^-1 rational:", all(x.is_Rational for x in Ms))
    coM = charpoly(Ms)
    zM, restM = split_zero(coM)
    rationalM = all(c.is_Rational for c in restM)
    Pm = sp.Poly(restM, xi) if rationalM else sp.Poly(restM, xi, extension=True)
    npos = nneg = nreal = 0
    if Pm.degree() > 0:
        for f, e in sp.sqf_list(Pm)[1]:
            fe = f.as_expr().subs(xi, nu)
            fq = sp.Poly(fe, nu) if rationalM else sp.Poly(fe, nu, extension=True)
            nr_, np_ = sturm_counts(fq)
            nreal += nr_ * e; npos += np_ * e; nneg += (nr_ - np_) * e
    chk(f"{name}: charpoly of Mhat real-rooted (Sturm, exact)", nreal == Pm.degree())
    sig = [nneg, zM - kdim, npos - 1]
    chk(f"{name}: n0 = (zero multiplicity of Mhat) - k is >= 0 and n+ = (positive count) - 1 is >= 0", min(sig) >= 0, str(sig))
    # numeric signature on N_u
    Mnum = to_mp_mat(Mw)
    if PLANT == "numeric":
        Mnum[0, 1] += mpm.mpf(10) ** -10; Mnum[1, 0] += mpm.mpf(10) ** -10
    evM = sorted(mpm.re(e) for e in mpm.eig(Mnum, left=False, right=False))
    tol = mpm.mpf(10) ** -30
    sig_num = [sum(1 for e in evM if e < -tol), sum(1 for e in evM if abs(e) <= tol) - kdim, sum(1 for e in evM if e > tol) - 1]
    chk(f"{name}: signature exact = numeric (50 digits)", sig == sig_num, f"{sig} vs {sig_num}")
    S = I0 * Ms
    coS = charpoly(S)
    zS, restS = split_zero(coS)
    P, ratP = slow_poly(restS)
    rd = root_data(P, ratP)
    Snum = to_mp_mat(I0) * Mnum
    evS = mpm.eig(Snum, left=False, right=False)
    nzero_num = sum(1 for e in evS if abs(e) < mpm.mpf(10) ** -20)
    chk(f"{name}: zero multiplicity of S exact = numeric count of |eig| < 1e-20", zS == nzero_num, f"{zS} vs {nzero_num}")
    agree, worst = match_roots([e * e for e in evS if abs(e) >= mpm.mpf(10) ** -20], rd)
    chk(f"{name}: squares of nonzero numeric eigenvalues of S match the slow polynomial's roots (each twice)", agree, mpm.nstr(worst, 3) if worst is not None else "count")
    kr = krein(Snum, Mnum, rd, name)
    w_ = words(rd)
    print(f"  k = {kdim}, r_u = {r_u}, signature [n-, n0, n+] = {sig}")
    print(f"  zero multiplicity of S_u = {zS}; slow polynomial ({'rational' if ratP else 'monic, algebraic coefficients'}) = {rd['factored']}")
    for r in rd["roots"]:
        print(f"     root nu = {r['value'][:32]}  ({r['kind']}, multiplicity {r['mult']})" + (f", exact {r['exact']}" if r['exact'] else ""))
    print(f"  slow-flow words: {w_}")
    for c in kr:
        print(f"     cluster tau = {c['tau']} (nu = {c['nu']}), Krein sign {c['sign']}, dim ker = {c['dim_ker']}, min|eig of Mhat there| = {c['min_abs_eig']}")
    R.update(k=kdim, r_u=r_u, signature=sig, zero_mult=zS, slow_poly=rd["factored"], slow_poly_rational=ratP, roots=rd["roots"], words=w_, krein=kr)
    R["members"] = {}
    for mn, (A, sc) in members.items():
        Am = A * Ms           # N commutes with i and Theta (N_k = N_{n-k})
        co = charpoly(Am)
        z, rest = split_zero(co)
        Pm_, ratm = slow_poly(rest, scale=sc)   # eigenvalues of (A/sqrt(sc)) Mhat: nu -> sc * nu
        rdm = root_data(Pm_, ratm)
        tau_num = to_mp_mat(A) / mpm.sqrt(sc)
        Sn = tau_num * Mnum
        evn = mpm.eig(Sn, left=False, right=False)
        nz_n = sum(1 for e in evn if abs(e) < mpm.mpf(10) ** -20)
        chk(f"{name} / member {mn}: zero multiplicity exact = numeric", z == nz_n, f"{z} vs {nz_n}")
        ag, worst = match_roots([e * e for e in evn if abs(e) >= mpm.mpf(10) ** -20], rdm)
        chk(f"{name} / member {mn}: numeric eigenvalues match exact slow polynomial", ag, mpm.nstr(worst, 3) if worst is not None else "count")
        krm = krein(Sn, Mnum, rdm, f"{name} / member {mn}")
        wm = words(rdm)
        print(f"  member {mn}: zero mult {z}; slow poly ({'rational' if ratm else 'monic, algebraic'}) {rdm['factored']}")
        print(f"      roots: {[(r['kind'], r['mult'], r['value'][:14]) for r in rdm['roots']]}")
        print(f"      words: {wm}; Krein: {[(c['tau'][:12], c['sign']) for c in krm]}")
        R["members"][mn] = dict(zero_mult=z, slow_poly=rdm["factored"], slow_poly_rational=ratm, roots=rdm["roots"], words=wm, krein=krm)

if PLANT is None:
    json.dump(results, open(rel("out_B3_R2.json"), "w"), indent=1)
chk.summary()
