"""Items 10, 11, 12 (Casimir route), 16, and Frobenius-Schur indicators for 21.

Everything exact over K = Q(sqrt5, i), in the monomial basis e_k = x^k y^(n-k) of Sym^n C^2,
n = 2j, weight m = k - j, Gram <e_k, e_k> = 1/C(n,k) (so v_m = sqrt(C(n,k)) e_k is the
Condon-Shortley orthonormal basis). Uses only the two generators q1, q2 (and their closure).
"""
import json
import pathlib
from fractions import Fraction as Fr
from math import comb, isqrt
from audit_field import (K, R5, ZERO, ONE, Q1, Q2, QONE, closure, nullspace, rank, mat_mul)

HERE = pathlib.Path(__file__).parent
res = {}

G = closure([Q1, Q2])
assert len(G) == 120


def polymul(p, q):
    out = [ZERO] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a.iszero():
            continue
        for k, b in enumerate(q):
            if b.iszero():
                continue
            out[i + k] = out[i + k] + a * b
    return out


def polypow(p, e):
    out = [ONE]
    for _ in range(e):
        out = polymul(out, p)
    return out


def Dmon(q, n):
    """Matrix of q on Sym^n in monomial basis; coefficient lists index = power of x."""
    (a, b), (c, d) = q.su2()
    # image of x: a x + c y  -> coeff list over powers of x: [c (x^0), a (x^1)]
    X = [c, a]
    Y = [d, b]
    M = [[ZERO] * (n + 1) for _ in range(n + 1)]
    for k in range(n + 1):
        col = polymul(polypow(X, k), polypow(Y, n - k))
        for kp in range(n + 1):
            M[kp][k] = col[kp]
    return M


def ident(n):
    return [[ONE if i == k else ZERO for k in range(n)] for i in range(n)]


def madd(A, B, s=ONE):
    return [[a + s * b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def trace(A):
    return sum((A[i][i] for i in range(len(A))), ZERO)


def cheb_U(w, n):
    """U_n(w) exactly (character of spin n/2 at an element with real part w)."""
    if n == 0:
        return R5(1)
    u0, u1 = R5(1), 2 * w
    for _ in range(n - 1):
        u0, u1 = u1, 2 * w * u1 - u0
    return u1


def dagger(A):
    return [[A[k][i].conj() for k in range(len(A))] for i in range(len(A[0]))]


def gram(n):
    return [[K(R5(Fr(1, comb(n, k)))) if i == k else ZERO for k in range(n + 1)] for i in range(n + 1)]


# ---------- exact rep checks ----------
for n in (1, 2, 6):
    Gm = gram(n)
    for q in G[:40]:
        D = Dmon(q, n)
        assert mat_mul(mat_mul(dagger(D), Gm), D) == Gm, "not unitary"
    for a in G[:8]:
        for b in G[:8]:
            assert mat_mul(Dmon(a, n), Dmon(b, n)) == Dmon(a * b, n)
res["rep_checks"] = "Dmon unitary wrt Gram (n=1,2,6; 40 elems) and homomorphic (64 pairs)"

# ---------- item 10 ----------
item10 = {}
for Kspin in range(0, 7):
    n = 2 * Kspin
    s = sum((cheb_U(g.w, n) for g in G), R5(0))
    dim_char = s / 120
    # cross-check: traces of the actual matrices
    s2 = sum((trace(Dmon(g, n)) for g in G), ZERO)
    assert K(s) == s2, (Kspin, s, s2)
    # second method: common fixed space of the two generators
    D1, D2 = Dmon(Q1, n), Dmon(Q2, n)
    I = ident(n + 1)
    stacked = madd(D1, I, K(-1)) + madd(D2, I, K(-1))
    fix = len(nullspace(stacked))
    assert dim_char.b == 0 and dim_char.a.denominator == 1
    assert int(dim_char.a) == fix
    item10[Kspin] = {"char_average": str(dim_char.a), "nullspace_dim": fix}
res["item10_dim_invariants_K0to6"] = item10
extra = {}
for Kspin in range(7, 31):
    s = sum((cheb_U(g.w, 2 * Kspin) for g in G), R5(0)) / 120
    assert s.b == 0 and s.a.denominator == 1
    extra[Kspin] = int(s.a)
res["item10_EXTRA_K7to30_char_average_only"] = extra

# ---------- conjugacy classes ----------
Gs = set(G)
classes = []
seen = set()
for g in G:
    if g in seen:
        continue
    cl = {h * g * h.conj() for h in G}
    seen |= cl
    classes.append(sorted(cl, key=lambda q: q.key()))
res["n_conjugacy_classes"] = len(classes)
res["class_sizes"] = [len(c) for c in classes]
res["class_real_parts"] = [str(c[0].w) for c in classes]


def rsqrt(r):
    """exact sqrt of r in R5 (assumed to exist in R5, r >= 0), else raise."""
    def qsqrt(x):
        if x < 0:
            return None
        p, q = x.numerator, x.denominator
        sp_, sq = isqrt(p), isqrt(q)
        if sp_ * sp_ == p and sq * sq == q:
            return Fr(sp_, sq)
        return None
    a, b = r.a, r.b
    cands = []
    if b == 0:
        c = qsqrt(a)
        if c is not None:
            cands.append(R5(c, 0))
        d = qsqrt(a / 5)
        if d is not None:
            cands.append(R5(0, d))
    else:
        disc = qsqrt(a * a - 5 * b * b)
        if disc is not None:
            for c2 in ((a + disc) / 2, (a - disc) / 2):
                c = qsqrt(c2)
                if c is not None and c != 0:
                    cands.append(R5(c, b / (2 * c)))
    for c in cands:
        if c * c == r:
            return c if c.sign() >= 0 else -c
    raise ValueError(f"no sqrt in Q(sqrt5) for {r}")


def decompose(n):
    """Isotypic decomposition of Sym^n restricted to Gamma. Returns list of projectors (monomial basis)."""
    D1, D2 = Dmon(Q1, n), Dmon(Q2, n)
    N = n + 1
    # commutant equations X D - D X = 0, unknowns X[i][k] flattened
    rows = []
    for D in (D1, D2):
        for i in range(N):
            for k in range(N):
                row = [ZERO] * (N * N)
                for l in range(N):
                    row[i * N + l] = row[i * N + l] + D[l][k]   # (X D)_{ik} = sum_l X_il D_lk
                    row[l * N + k] = row[l * N + k] - D[i][l]   # (D X)_{ik} = sum_l D_il X_lk
                rows.append(row)
    comm = nullspace(rows)
    cdim = len(comm)
    # character-based check of sum m_i^2
    chis = [cheb_U(g.w, n) for g in G]
    s = sum((c * c for c in chis), R5(0)) / 120  # characters real here
    assert s == R5(cdim), (s, cdim)
    I = ident(N)
    if cdim == 1:
        return cdim, [I]
    # class sums, look for one with 2 distinct eigenvalues (commutant dim 2 case)
    assert cdim == 2, "general case not needed"
    for cl in classes:
        Y = [[ZERO] * N for _ in range(N)]
        for g in cl:
            Y = madd(Y, Dmon(g, n))
        Y2 = mat_mul(Y, Y)
        # solve Y2 = al Y + be I using two independent entries
        # find an entry where Y is off-diagonal nonzero
        off = [(i, k) for i in range(N) for k in range(N) if i != k and not Y[i][k].iszero()]
        if not off:
            continue
        i, k = off[0]
        al = Y2[i][k] / Y[i][k]
        be = Y2[0][0] - al * Y[0][0]
        resid = [[Y2[r][c] - al * Y[r][c] - be * I[r][c] for c in range(N)] for r in range(N)]
        if any(not x.iszero() for row in resid for x in row):
            continue
        assert al.q.iszero() and be.q.iszero()
        disc = al.p * al.p + 4 * be.p
        sq = rsqrt(disc)
        l1, l2 = (al.p + sq) / 2, (al.p - sq) / 2
        if l1 == l2:
            continue
        P1 = [[(Y[i][k] - (l2 if i == k else R5(0))) / (l1 - l2) for k in range(N)] for i in range(N)]
        P2 = madd(I, P1, K(-1))
        return cdim, [P1, P2]
    raise RuntimeError("no separating class sum")


def ktostr(x):
    return [str(x.p.a), str(x.p.b), str(x.q.a), str(x.q.b)]


def casimir_apply(w, n):
    """w: dict (k1,k2)->K. Casimir of V_j (x) V_j in monomial basis, j = n/2."""
    j2 = Fr(n, 2)
    out = {}

    def add(key, val):
        if val.iszero():
            return
        out[key] = out.get(key, ZERO) + val

    # Jz^2
    for (k1, k2), val in w.items():
        mz = Fr(k1) - j2 + Fr(k2) - j2
        add((k1, k2), val * K(R5(mz * mz)))

    def Jp(v):
        o = {}
        for (k1, k2), val in v.items():
            if k1 < n:
                o[(k1 + 1, k2)] = o.get((k1 + 1, k2), ZERO) + val * (n - k1)
            if k2 < n:
                o[(k1, k2 + 1)] = o.get((k1, k2 + 1), ZERO) + val * (n - k2)
        return o

    def Jm(v):
        o = {}
        for (k1, k2), val in v.items():
            if k1 > 0:
                o[(k1 - 1, k2)] = o.get((k1 - 1, k2), ZERO) + val * k1
            if k2 > 0:
                o[(k1, k2 - 1)] = o.get((k1, k2 - 1), ZERO) + val * k2
        return o
    for key, val in Jp(Jm(w)).items():
        add(key, val * K(R5(Fr(1, 2))))
    for key, val in Jm(Jp(w)).items():
        add(key, val * K(R5(Fr(1, 2))))
    return out


def proj_spin(w, n, L):
    """Apply the Casimir projector onto total spin L in V_j (x) V_j (spins 0..n)."""
    v = dict(w)
    for Lp in range(0, n + 1):
        if Lp == L:
            continue
        cv = casimir_apply(v, n)
        den = Fr(L * (L + 1) - Lp * (Lp + 1))
        v = {k: (cv.get(k, ZERO) - v.get(k, ZERO) * (Lp * (Lp + 1))) * K(R5(1 / den)) for k in set(cv) | set(v)}
        v = {k: x for k, x in v.items() if not x.iszero()}
    return v


def ip_G(a, b, n):
    s = ZERO
    for key, val in b.items():
        if key in a:
            s = s + a[key].conj() * val * K(R5(Fr(1, comb(n, key[0]) * comb(n, key[1]))))
    return s


def MK_norms(Pm, n, eps=1):
    """||M^{(j)}_K(P)||^2 for K=0..n via w_mon[k1,k2] = eps (-1)^(n-k2) C(n,k2) P_mon[k1, n-k2]."""
    w = {}
    for k1 in range(n + 1):
        for k2 in range(n + 1):
            val = Pm[k1][n - k2] * (eps * (-1) ** (n - k2) * comb(n, k2))
            if not val.iszero():
                w[(k1, k2)] = val
    out = []
    for L in range(0, n + 1):
        pw = proj_spin(w, n, L)
        nv = ip_G(w, pw, n)
        # must equal <pw,pw> (projector is orthogonal and idempotent)
        assert nv == ip_G(pw, pw, n)
        assert nv.q.iszero()
        out.append(nv.p)
    return out


item11, item12, item16, fs = {}, {}, {}, {}
projectors_out = {}
for n in range(1, 7):
    cdim, Ps = decompose(n)
    N = n + 1
    lev = {"commutant_dim": cdim, "summands": []}
    Gm = gram(n)
    for P in Ps:
        assert mat_mul(P, P) == P
        for D in (Dmon(Q1, n), Dmon(Q2, n)):
            assert mat_mul(P, D) == mat_mul(D, P)
        # orthogonal projector wrt the invariant inner product: P^dag G = G P
        assert mat_mul(dagger(P), Gm) == mat_mul(Gm, P)
        d = rank(P)
        assert trace(P) == K(d)
        chi = [trace(mat_mul(P, Dmon(g, n))) for g in G]
        irr = sum((c * c.conj() for c in chi), ZERO) / 120
        assert irr == ONE, "summand not irreducible"
        # second construction: P = (d/|G|) sum conj(chi(g)) D(g)
        Pc = [[ZERO] * N for _ in range(N)]
        for c, g in zip(chi, G):
            Dg = Dmon(g, n)
            Pc = madd(Pc, [[c.conj() * x for x in r] for r in Dg])
        Pc = [[x * K(R5(Fr(d, 120))) for x in r] for r in Pc]
        assert Pc == P, "character-formula projector disagrees"
        # Frobenius-Schur indicator
        gidx = {g: i for i, g in enumerate(G)}
        fsi = sum((chi[gidx[g * g]] for g in G), ZERO) / 120
        chis_by_class = [str(chi[gidx[c[0]]].p) + ("" if chi[gidx[c[0]]].q.iszero() else "+i" + str(chi[gidx[c[0]]].q)) for c in classes]
        norms = MK_norms(P, n, eps=(-1 if n == 6 else 1))
        tot = sum(norms, R5(0))
        assert tot == R5(d), "sum_K ||M_K(P)||^2 != rank"
        lev["summands"].append({"dim": d, "multiplicity": 1, "FS_indicator": str(fsi.p) if fsi.q.iszero() else repr(fsi),
                                "character_on_classes": chis_by_class,
                                "MK_norm2_K0..n": [str(x) for x in norms]})
        if n == 6:
            projectors_out[f"d{d}"] = [[ktostr(x) for x in r] for r in P]
    item16[f"level_{n}_j={Fr(n, 2)}"] = lev
res["class_order_representatives_real_part"] = [str(c[0].w) for c in classes]
res["item11_and_16_levels"] = item16
(HERE / "audit_projectors_level6.json").write_text(json.dumps(projectors_out))
(HERE / "audit_res_gamma.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
