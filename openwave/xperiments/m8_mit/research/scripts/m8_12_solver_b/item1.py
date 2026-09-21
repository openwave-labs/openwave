"""Item 1: fixed spaces V^(H,chi) of dimension 1 and 2, up to rotation.

Enumeration: closed subgroups of SO(3) up to conjugacy are C_n (n>=1), D_n (n>=2),
T, O, I, SO(2), O(2), SO(3) (standard classification; see RETURN.md).  Finite groups
are enumerated explicitly for n <= 12 (n >= 7 is also handled by an argument);
their 1-dimensional characters are found by brute force over generator values with
a homomorphism check.  Fixed-space dimensions come from two routes: the exact
character formula and a numerical rank computation.  Exact bases are then built
and grouped into classes up to rotation, with explicit exact rotations inside a
class and exact rotation invariants separating classes.
"""
import os, sys, json, itertools, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mpmath
import sympy as sp
import core, rot
from core import report, IDX, MS

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
pi = sp.pi
z = (0, 0, 1); x = (1, 0, 0)
n111 = tuple(sp.Integer(1) / sp.sqrt(3) for _ in range(3))
phi = (1 + sp.sqrt(5)) / 2
_n5 = sp.sqrt(1 + phi ** 2)
n5 = (0, 1 / _n5, phi / _n5)

# ------------------------------------------------------------ exact D checks
Rx_exact = rot.D3_exact(x, pi)
Rx_claim = sp.zeros(7)
for m in MS:
    Rx_claim[IDX[-m], IDX[m]] = -1
ok &= report("exact D3(x, pi) equals v_m -> -v_{-m}", (Rx_exact - Rx_claim).applyfunc(sp.simplify) == sp.zeros(7))
ok &= report("exact D3(x, 2 pi) = identity (SO(3) representation)",
             (rot.D3_exact(x, 2 * pi) - sp.eye(7)).applyfunc(sp.simplify) == sp.zeros(7))
C3 = rot.D3_exact(n111, 2 * pi / 3)
ok &= report("exact D3((1,1,1)/sqrt3, 2pi/3) agrees with expm to 1e-12",
             np.abs(np.array(sp.N(C3, 20), dtype=complex) - rot.D3_num(n111, 2 * math.pi / 3)).max() < 1e-12)
ok &= report("exact C3^3 = identity", (C3 ** 3 - sp.eye(7)).applyfunc(sp.simplify) == sp.zeros(7))

# ------------------------------------------------------------ groups
def gens_of(name):
    if name.startswith("C"):
        n = int(name[1:]); return [(z, 2 * pi / n)]
    if name.startswith("D"):
        n = int(name[1:]); return [(z, 2 * pi / n), (x, pi)]
    if name == "T":
        return [(z, pi), (n111, 2 * pi / 3)]
    if name == "O":
        return [(z, pi / 2), (n111, 2 * pi / 3)]
    if name == "I":
        return [(n5, 2 * pi / 5), (n111, 2 * pi / 3)]


def key(M):
    return tuple(np.round(M, 7).ravel() + 0.0)


def close(gm):
    """all elements as 3x3 float matrices, BFS from identity; returns list and word-tree edges."""
    elems = [np.eye(3)]
    index = {key(np.eye(3)): 0}
    edges = []   # (i, gen, j): elems[j] = elems[i] @ gen
    parent = {0: None}
    q = [0]
    while q:
        i = q.pop(0)
        for gi, G in enumerate(gm):
            M = elems[i] @ G
            k = key(M)
            if k not in index:
                index[k] = len(elems); elems.append(M); q.append(index[k]); parent[index[k]] = (i, gi)
            edges.append((i, gi, index[k]))
    close.parent = parent
    return elems, edges


def angles_mp(gens, parent, n_el, dps=40):
    """second-precision identification: rebuild every element along its BFS word at `dps` digits
    (mpmath Rodrigues matrices) and identify theta/pi with denominator bound 60, tolerance 1e-30."""
    mpmath.mp.dps = dps
    def R3mp(n, th):
        n = [mpmath.mpf(sp.N(v, dps + 10)) for v in n]
        nr = mpmath.sqrt(sum(v ** 2 for v in n)); n = [v / nr for v in n]
        K = mpmath.matrix([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        t = mpmath.mpf(sp.N(th, dps + 10))
        return mpmath.eye(3) + mpmath.sin(t) * K + (1 - mpmath.cos(t)) * K * K
    G = [R3mp(n, th) for n, th in gens]
    mats = {0: mpmath.eye(3)}
    for j in range(1, n_el):
        chain = []
        k = j
        while parent[k] is not None:
            chain.append(parent[k][1]); k = parent[k][0]
        M = mpmath.eye(3)
        for gi in reversed(chain):
            M = M * G[gi]
        mats[j] = M
    out = []
    for j in range(n_el):
        M = mats[j]
        c = (M[0, 0] + M[1, 1] + M[2, 2] - 1) / 2
        sn = mpmath.sqrt(((M[2, 1] - M[1, 2]) / 2) ** 2 + ((M[0, 2] - M[2, 0]) / 2) ** 2 + ((M[1, 0] - M[0, 1]) / 2) ** 2)
        t = mpmath.atan2(sn, c) / mpmath.pi
        fr = sp.Rational(str(mpmath.nstr(t, 35))).limit_denominator(60)
        if abs(mpmath.mpf(fr.p) / fr.q - t) > mpmath.mpf(10) ** -30:
            raise RuntimeError("angle not identified at %d digits" % dps)
        out.append(fr * pi)
    return out


def order(M):
    P = np.eye(3)
    for k in range(1, 200):
        P = P @ M
        if np.allclose(P, np.eye(3), atol=1e-9):
            return k


def characters(gm, elems, edges):
    ords = [order(G) for G in gm]
    L = int(np.lcm.reduce(ords))
    chars = []
    for a in itertools.product(*[range(0, L, L // o) for o in ords]):
        val = {0: 0}
        good = True
        # propagate along edges repeatedly
        changed = True
        while changed and good:
            changed = False
            for (i, gi, j) in edges:
                if i in val:
                    v = (val[i] + a[gi]) % L
                    if j in val:
                        if val[j] != v:
                            good = False; break
                    else:
                        val[j] = v; changed = True
        if good and len(val) == len(elems):
            chars.append((a, L, val))
    return chars, L


def angle_of(M):
    c = (np.trace(M) - 1) / 2
    A = (M - M.T) / 2
    sn = math.sqrt(A[2, 1] ** 2 + A[0, 2] ** 2 + A[1, 0] ** 2)
    t = math.atan2(sn, c)
    fr = sp.Rational(t / math.pi).limit_denominator(60)
    assert abs(float(fr) * math.pi - t) < 1e-9
    return fr * pi


_CHI3 = {}


def chi3(theta):
    if theta not in _CHI3:
        _CHI3[theta] = _chi3(theta)
    return _CHI3[theta]


def _chi3(theta):
    return core.exact(sum(sp.exp(-sp.I * m * theta) for m in MS).rewrite(sp.cos))


def dim_exact(elems, val, L):
    """(1/|H|) sum_h conj(chi(h)) chi_3(theta_h) in exact cyclotomic arithmetic: every term is a root of
    unity zeta_D^e (D = common denominator), the sum is reduced modulo the cyclotomic polynomial Phi_D
    and must leave an integer constant."""
    zz = sp.Symbol("zz")
    terms = []
    for i, M in enumerate(elems):
        q = angle_of(M) / (2 * pi)                 # theta_h = 2 pi q, q rational
        for m in MS:
            terms.append(sp.Rational(-val[i], L) - m * q)   # exponent / (2 pi i)
    D = int(sp.ilcm(*[sp.fraction(t_)[1] for t_ in terms]))
    p = sp.Poly(sum(zz ** int((t_ * D) % D) for t_ in terms), zz, domain="ZZ")
    r = p.rem(sp.Poly(sp.cyclotomic_poly(D, zz), zz, domain="ZZ"))
    if r.degree() > 0 or r.eval(0) % len(elems) != 0:
        raise RuntimeError("character formula did not reduce to an integer multiple of |H|: %s" % r)
    return sp.Integer(r.eval(0) // len(elems))


def dim_num(gens, a, L):
    rows = []
    for (n, th), ai in zip(gens, a):
        rows.append(rot.D3_num(n, th) - np.exp(2j * math.pi * ai / L) * np.eye(7))
    S = np.linalg.svd(np.vstack(rows), compute_uv=False)
    return int(sum(S < 1e-9))


def exact_D(n, th):
    if n == z:
        return rot.Dz(th)
    if n == x and th == pi:
        return Rx_claim
    if n == n111:
        return C3 if th == 2 * pi / 3 else None
    return rot.D3_exact(n, th)


def exact_space(gens, a, L):
    rows = []
    for (n, th), ai in zip(gens, a):
        rows.append(exact_D(n, th) - sp.expand_complex(sp.exp(2 * pi * sp.I * ai / L)) * sp.eye(7))
    A = sp.Matrix.vstack(*rows).applyfunc(core.exact)
    ns = A.nullspace(simplify=True)
    return [v.applyfunc(core.exact) for v in ns]


names = ["C%d" % n for n in range(1, 13)] + ["D%d" % n for n in range(2, 13)] + ["T", "O", "I"]
expected_order = {**{"C%d" % n: n for n in range(1, 13)}, **{"D%d" % n: 2 * n for n in range(2, 13)},
                  "T": 12, "O": 24, "I": 60}
expected_nchar = {**{"C%d" % n: n for n in range(1, 13)},
                  **{"D%d" % n: (4 if n % 2 == 0 else 2) for n in range(2, 13)}, "T": 3, "O": 2, "I": 1}
raw = []          # (group, char descr, dim, basis or None)
n_dimcmp = n_dimbad = 0
n_angle_checked = n_angle_bad = 0
dims_seen = {}
for nm in names:
    gens = gens_of(nm)
    gm = [rot.R3(n, th) for n, th in gens]
    elems, edges = close(gm)
    ok &= report("%s has order %d" % (nm, expected_order[nm]), len(elems) == expected_order[nm], "found %d" % len(elems))
    chars, L = characters(gm, elems, edges)
    a64 = [angle_of(M) for M in elems]
    a40 = angles_mp(gens, close.parent, len(elems))
    n_angle_checked += 1
    if a64 != a40:
        n_angle_bad += 1
    ok &= report("%s has %d one-dimensional characters" % (nm, expected_nchar[nm]), len(chars) == expected_nchar[nm],
                 "found %d" % len(chars))
    for a, L, val in chars:
        de = dim_exact(elems, val, L)
        dn = dim_num(gens, a, L)
        n_dimcmp += 1
        if de != dn:
            n_dimbad += 1
            ok &= report("%s chi=%s: exact and numerical dimension agree" % (nm, a), False, "%s vs %s" % (de, dn))
        dims_seen.setdefault(int(de), []).append((nm, list(a), L))
        B = None
        if de in (1, 2):
            if nm == "I":
                raise RuntimeError("unexpected")
            B = exact_space(gens, a, L)
            assert len(B) == de, (nm, a, len(B), de)
        raw.append((nm, list(a), L, int(de), B))
ok &= report("exact (character formula) and numerical (rank) fixed-space dimensions agree for every (H, chi)",
             n_dimbad == 0 and n_dimcmp > 0, "%d pairs compared, %d mismatches" % (n_dimcmp, n_dimbad))

ok &= report("rotation angles of all group elements: float64 identification (denominator <= 60, tol 1e-9) "
             "equals 40-digit identification (tol 1e-30)", n_angle_bad == 0 and n_angle_checked > 0,
             "%d groups, %d disagreements" % (n_angle_checked, n_angle_bad))

# continuous groups
for m in MS:   # SO(2)_z with chi(phi) = e^{-i m phi}: fixed space = ker(J_z - m)
    v = sp.zeros(7, 1); v[IDX[m]] = 1
    raw.append(("SO2", [m], None, 1, [v]))
    dims_seen.setdefault(1, []).append(("SO2", "chi=e^{-i %d phi}" % m, None))
for k in range(4, 8):
    dims_seen.setdefault(0, []).append(("SO2", "chi=e^{-i %d phi}" % k, None))
# O(2): chi must be trivial on SO(2) (s r s = r^-1 forces chi(r)^2 = 1 on a connected group), chi(s) = +-1
v0 = sp.zeros(7, 1); v0[IDX[0]] = 1
ok &= report("R_x v0 = -v0, so O(2) with chi(reflection) = -1 fixes v0 and the trivial character fixes nothing",
             Rx_claim * v0 == -v0)
raw.append(("O2", ["chi(s)=-1"], None, 1, [v0]))
dims_seen.setdefault(1, []).append(("O2", "chi(s)=-1", None))
dims_seen.setdefault(0, []).append(("O2", "trivial", None))
dims_seen.setdefault(0, []).append(("SO3", "trivial", None))

# ------------------------------------------------------------ classes up to rotation
Jz, Jp, Jm, Jx, Jy = core.Jmats()
Js = [Jx, Jy, Jz]


def gram(B):
    Bm = sp.Matrix.hstack(*B)
    return Bm, (Bm.H * Bm).applyfunc(sp.simplify)


def projector(B):
    Bm, G = gram(B)
    return (Bm * G.inv() * Bm.H).applyfunc(core.exact)


def invariants(B):
    Bm, G = gram(B)
    Gi = G.inv()
    lam = sp.Symbol("lam")
    if len(B) == 1:
        u = [Bm[k, 0] for k in range(7)]
        r6 = core.rhat6(u)
        nrm = G[0, 0]
        Jexp = [sp.simplify((Bm.H * J * Bm)[0, 0] / nrm) for J in Js]
        J2 = core.exact(sum(sp.Abs(t) ** 2 for t in Jexp))
        return ("r6=%s" % r6, "|<J>|^2=%s" % J2)
    JW = [(Gi * Bm.H * J * Bm).applyfunc(sp.simplify) for J in Js]
    S1 = sum((A * A for A in JW), sp.zeros(2))
    Qs = []
    for a_ in range(3):
        for b_ in range(3):
            Q = (Js[a_] * Js[b_] + Js[b_] * Js[a_]) / 2 - (4 if a_ == b_ else 0) * sp.eye(7)
            Qs.append((Gi * Bm.H * Q * Bm).applyfunc(sp.simplify))
    S2 = sum((A * A for A in Qs), sp.zeros(2))
    p1 = sp.factor((lam * sp.eye(2) - S1).det())
    p2 = sp.factor((lam * sp.eye(2) - S2).det())
    # dimension of the Lie algebra of the setwise stabiliser {X in so(3): X W in W}
    P = projector(B)
    cs = sp.symbols("ca cb cc")
    X = cs[0] * Jx + cs[1] * Jy + cs[2] * Jz
    eqs = list(((sp.eye(7) - P) * X * P).applyfunc(sp.expand))
    lin = []
    for e in eqs:
        re_, im_ = e.as_real_imag()
        lin += [re_, im_]
    Aeq = sp.Matrix([[sp.diff(e, c_) for c_ in cs] for e in lin])
    stab_dim = 3 - Aeq.rank(simplify=True)
    return ("charpoly sum_a (P J_a P)^2 = %s" % p1, "charpoly sum_ab (P Q_ab P)^2 = %s" % p2,
            "dim Lie(setwise stabiliser) = %d" % stab_dim)


# distinct subspaces
spaces = []   # dicts: dim, B, P, producers
for (nm, a, L, d, B) in raw:
    if d not in (1, 2):
        continue
    P = projector(B)
    for s in spaces:
        if s["dim"] == d and (s["P"] - P).applyfunc(sp.simplify) == sp.zeros(7):
            s["producers"].append((nm, a, L)); break
    else:
        spaces.append({"dim": d, "B": B, "P": P, "producers": [(nm, a, L)]})
print("distinct fixed subspaces of dim 1 or 2 found (before rotation classes):", len(spaces))

# candidate rotations for identifying members of a class: exact factors, numerical products for the
# prefilter, exact product built only for a numerical match
base = {"id": sp.eye(7), "Rx(pi)": Rx_claim, "C3(111)": C3, "C3(111)^2": (C3 * C3).applyfunc(core.exact)}
for k in (2, 4, 6, 8, 10, 12):
    base["Rz(2pi/%d)" % k] = rot.Dz(2 * pi / k)
base_num = {k: np.array(sp.N(v, 30), dtype=complex) for k, v in base.items()}
cands = [((k,), base_num[k]) for k in base]
for k1, k2 in itertools.product(list(base), repeat=2):
    if k1 != "id" and k2 != "id":
        cands.append(((k1, k2), base_num[k1] @ base_num[k2]))


def exact_of(names_):
    A = sp.eye(7)
    for k in names_:
        A = A * base[k]
    return A.applyfunc(core.exact)


def maps_onto(A, P1, P2):
    Pimg = A * P1 * A.H
    return (Pimg - P2).applyfunc(lambda t: sp.simplify(sp.expand(t))) == sp.zeros(7)


for s in spaces:
    s["inv"] = invariants(s["B"])
classes = []
for s in spaces:
    for c in classes:
        if c["dim"] == s["dim"] and c["inv"] == s["inv"]:
            # must exhibit a rotation
            found = None
            Pn = np.array(sp.N(c["rep"]["P"]), dtype=complex); Ps = np.array(sp.N(s["P"]), dtype=complex)
            for names_, An in cands:
                if np.abs(An @ Pn @ An.conj().T - Ps).max() < 1e-9:
                    if maps_onto(exact_of(names_), c["rep"]["P"], s["P"]):
                        found = "*".join(names_); break
            if found is None:
                ok &= report("members with equal invariants are related by an exhibited rotation", False,
                             str(s["producers"]))
                continue
            c["members"].append((s, found)); break
    else:
        classes.append({"dim": s["dim"], "inv": s["inv"], "rep": s, "members": [(s, "id")]})


def show_vec(v):
    terms = []
    for m in MS:
        cf = core.exact(v[IDX[m]])
        if cf != 0:
            terms.append("(%s)v%d" % (cf, m))
    return " + ".join(terms)


out = {"classes": [], "dims_other": sorted(k for k in dims_seen if k not in (1, 2))}
print("\ncomplex dimensions of fixed spaces that occur:", sorted(dims_seen))
print("dimensions other than 1, 2:", out["dims_other"])
for dd in sorted(dims_seen):
    if dd not in (1, 2):
        print("  dim %d produced by e.g." % dd, dims_seen[dd][:4])
for c in classes:
    print("\nCLASS dim %d  rep basis: %s" % (c["dim"], " | ".join(show_vec(v) for v in c["rep"]["B"])))
    print("  invariants:", c["inv"])
    prods = []
    for s, how in c["members"]:
        prods.append({"basis": [show_vec(v) for v in s["B"]], "rotation_from_rep": how,
                      "producers": [[p[0], str(p[1]), p[2]] for p in s["producers"]]})
        print("  member via %s: %s  <- %s" % (how, [show_vec(v) for v in s["B"]], s["producers"][:6]))
    out["classes"].append({"dim": c["dim"], "rep": [show_vec(v) for v in c["rep"]["B"]],
                           "invariants": list(c["inv"]), "members": prods})

n1 = sum(c["dim"] == 1 for c in classes); n2 = sum(c["dim"] == 2 for c in classes)
print("\nclasses: dim1 = %d, dim2 = %d" % (n1, n2))
# pairwise distinctness by exact invariants
for dd in (1, 2):
    cl = [c for c in classes if c["dim"] == dd]
    distinct = len(set(c["inv"] for c in cl)) == len(cl)
    ok &= report("dim-%d classes pairwise separated by exact rotation invariants" % dd, distinct)
out["n_dim1"] = n1; out["n_dim2"] = n2
json.dump(out, open(os.path.join(HERE, "out", "item1.json"), "w"), indent=1)
print("ITEM1", "ALLPASS" if ok else "SOMEFAIL")
