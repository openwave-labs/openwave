"""Block B: B.1 (R4), B.2 (R5, with clauses 3 and 4), B.5.

Everything is exact (sympy, rational / quadratic-field arithmetic) except the Krein-sign
definiteness tests, which are numeric (mpmath, 60 and 90 digits, both reported)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import Checker, planted, cg
from pointdata import *
import mpmath as mp

chk = Checker()
OUT = os.path.join(HERE, "out")
inp = json.load(open(os.path.join(HERE, "..", "inputs.json")))
A = json.load(open(os.path.join(OUT, "blockA.json")))
gmats = [sp.Matrix([[sp.Rational(e) for e in r] for r in M]) for M in inp["generators"]["matrices"]]
xi, nu = sp.symbols("xi nu")

# --- the given generators span the same 4-space as phase + derived spin-3 rotations
def derived_generators():
    """-i J_a in CS components, moved to the chart x: p = F^{-1/2} w."""
    j = 3
    Jp = sp.zeros(7)
    for a, m in enumerate(MS):
        if m < 3:
            Jp[a - 1, a] = sp.sqrt((j - m) * (j + m + 1))      # J+ v_m = sqrt(..) v_{m+1}
    Jm = Jp.T
    Jx = (Jp + Jm) / 2; Jy = (Jp - Jm) / (2 * sp.I); Jz = sp.diag(*MS)
    Fh = sp.diag(*[sp.sqrt(f) for f in FM]); Fhi = Fh.inv()
    out = [sp.I * sp.eye(7)]
    for Ja in (Jx, Jy, Jz):
        out.append((Fhi * (-sp.I * Ja) * Fh).applyfunc(sp.radsimp))
    real = []
    for C in out:
        Re = C.applyfunc(sp.re); Im = C.applyfunc(sp.im)
        real.append(sp.Matrix(sp.BlockMatrix([[Re, -Im], [Im, Re]])))
    return real
dg = derived_generators()
def span_equal(L1, L2):
    v1 = sp.Matrix.hstack(*[M.reshape(196, 1) for M in L1])
    v2 = sp.Matrix.hstack(*[M.reshape(196, 1) for M in L2])
    return v1.rank() == 4 and v2.rank() == 4 and sp.Matrix.hstack(v1, v2).rank() == 4
chk("inputs.json generators span = span(phase, derived infinitesimal rotations), dim 4", span_equal(gmats, dg))
planted("span test with the z-rotation replaced by its square", lambda: span_equal(gmats[:3] + [gmats[3] * gmats[3]], dg))

# --- B.5: HS spin-6 factor vs r6hat, as exact polynomials in the chart
def bilin_norm2(J, sign_fn):
    """sum_M |sum_{m1+m2=M} CG(3 m1;3 m2|J M) sign_fn(m2) w_m1 conj(w_{-m2})|^2 in the ring."""
    P = [(X[a], X[a + 7]) for a in range(7)]
    tot = R(0)
    for M in range(-J, J + 1):
        terms = []
        for i1, m1 in enumerate(MS):
            m2 = M - m1
            if abs(m2) > 3: continue
            i2 = MS.index(-m2)
            K = cg(3, m1, 3, m2, J, M) * sign_fn(m2) * sp.sqrt(FM[i1] * FM[i2])
            if K != 0: terms.append((K, i1, i2))
        if not terms: continue
        c0 = terms[0][0]; re = R(0); im = R(0)
        for K, i1, i2 in terms:
            r = sp.radsimp(K / c0); assert r.is_Rational
            a, b = P[i1]; c, d = P[i2][0], -P[i2][1]
            rr = QQ(int(r.p), int(r.q))
            re += rr * (a * c - b * d); im += rr * (a * d + b * c)
        c02 = sp.radsimp(c0 ** 2); assert c02.is_Rational
        tot += QQ(int(c02.p), int(c02.q)) * (re ** 2 + im ** 2)
    return tot
# r6hat: (Theta w)_{m2} = (-1)^{3-m2} conj(w_{-m2})
r6poly = bilin_norm2(6, lambda m2: (-1) ** (3 - m2))
# HS: tr((T^6_M)^+ w w^+) = sum_{m,m'} (-1)^{3-m'} CG(3 m;3 -m'|6 M) w_m conj(w_m'); put m2 = -m'
hspoly = bilin_norm2(6, lambda m2: (-1) ** (3 + m2))
print("== B.5")
chk("B.5: ||(ww^+)_6||^2 and ||[w (x) Theta w]_6||^2 are the same polynomial (exact)", r6poly == hspoly)
planted("B.5 test against the spin-4 analogue", lambda: r6poly == bilin_norm2(4, lambda m2: (-1) ** (3 + m2)))
aJ = {int(k): sp.Rational(v) for k, v in A["_aJ_HS6"].items()}
hs_from_f = sum(QQ(int(aJ[J].p), int(aJ[J].q)) * channels()[J] for J in (6, 4, 2, 0))
chk("HS6 polynomial = sum_J a_J ||[w(x)w]_J||^2 with the a_J of s03 (exact)", hspoly == hs_from_f)
print("   constant: HS6 = 1 * r6hat on all of V_3")

def r6_at(x):
    return sp.radsimp(sp.expand(to_sym(r6poly, x)))

# --- per-slot analysis
def mp_matrix(M, dps):
    with mp.workdps(dps):
        return mp.matrix([[mp.mpmathify(sp.N(M[a, b], dps + 20)) for b in range(M.shape[1])] for a in range(M.shape[0])])

def krein(d, nu_root, dps):
    """Krein sign of the cluster ker(S^2 + tau^2), tau^2 = -nu_root > 0; numeric at dps digits."""
    with mp.workdps(dps):
        S = mp_matrix(d.Sop, dps); Mo = mp_matrix(d.Mop, dps); G = mp.diag([mp.mpf(g) for g in GDIAG])
        tau2 = -mp.mpmathify(sp.N(nu_root, dps + 20))
        A_ = S * S + tau2 * mp.eye(14)
        U, Sv, V = mp.svd_r(A_)
        sv = [Sv[i] for i in range(14)]
        tol = mp.mpf(10) ** (-dps // 2)
        ker_idx = [i for i in range(14) if sv[i] < tol]
        nonker = [sv[i] for i in range(14) if sv[i] >= tol]
        gap = min(nonker) if nonker else mp.inf
        B = mp.matrix(14, len(ker_idx))
        for c, i in enumerate(ker_idx):
            for r in range(14): B[r, c] = V[i, r]
        F = B.T * G * Mo * B
        F = (F + F.T) / 2
        ev = mp.eigsy(F)[0] if len(ker_idx) > 1 else [F[0, 0]]
        ev = [ev[i] for i in range(len(ker_idx))]
        mx = max(abs(e) for e in ev)
        if all(e > mx * mp.mpf(10) ** -20 for e in ev): s = "+"
        elif all(e < -mx * mp.mpf(10) ** -20 for e in ev): s = "-"
        else: s = "indefinite"
        return s, len(ker_idx), ev, max(sv[i] for i in ker_idx) if ker_idx else None, gap

def analyse(slot, qJ, w, name):
    hpoly = quartic(qJ)
    d = PointData(hpoly, w, gmats)
    out = {"critical": bool(d.critical)}
    out["Q"] = str(d.hu)
    out["924 r6hat"] = str(sp.radsimp(924 * r6_at(d.x) / d.n2 ** 2))
    xv = sp.Matrix(d.x)
    chk(f"{slot} {name}: M-hat vanishes on K_u (exact)", d.MK_zero, quiet=True)
    chk(f"{slot} {name}: M-hat u = 8 h(u) u (exact)", d.Mu == sp.zeros(14, 1), quiet=True)
    chk(f"{slot} {name}: u is G-orthogonal to K_u (exact)", (d.K.T * GM * xv).applyfunc(simp) == sp.zeros(4, 1), quiet=True)
    cpM, fieldM = charpoly_exact(d.Mop, xi)
    PM = sp.Poly(cpM, xi)
    assert all(c.is_Rational for c in PM.all_coeffs()), "charpoly of M-hat not rational"
    z = 0
    while PM.eval(0) == 0:
        PM = sp.Poly(sp.quo(PM.as_expr(), xi), xi); z += 1
    # count roots WITH multiplicity: count_roots counts distinct roots, so go through the factorisation
    npos = nneg = 0
    for f, e in sp.factor_list(PM.as_expr())[1]:
        f = sp.Poly(f, xi)
        npos += e * f.count_roots(0, sp.oo); nneg += e * f.count_roots(-sp.oo, 0)
    out["_Mhat_charpoly"] = str(sp.factor(PM.as_expr()) * xi ** z)
    chk(f"{slot} {name}: charpoly of M-hat is real-rooted (counted with multiplicity)", npos + nneg + z == 14, quiet=True)
    chk(f"{slot} {name}: dim ker M-hat >= k", z >= d.k, quiet=True)
    sig = [int(nneg), int(z - d.k), int(npos - 1)]
    out["signature"] = sig
    out["k"] = int(d.k); out["r_u"] = int(d.r)
    cpS, fieldS = charpoly_exact(d.Sop, xi)
    PS = sp.Poly(cpS, xi)
    assert all(c.is_Rational for c in PS.all_coeffs()), "charpoly of S-hat not rational"
    zS = 0
    while PS.eval(0) == 0:
        PS = sp.Poly(sp.quo(PS.as_expr(), xi), xi); zS += 1
    out["zero multiplicity"] = int(zS)
    chk(f"{slot} {name}: slow char. polynomial is even in xi", all(c == 0 for i, c in enumerate(reversed(PS.all_coeffs())) if i % 2), quiet=True)
    sl = sp.expand(PS.as_expr().subs(xi, sp.sqrt(nu)))
    if sl == 1 or sp.Poly(sl, nu).degree() == 0:
        out["slow polynomial"] = "1 (no nonzero eigenvalues)"
        out["roots"] = []; out["words"] = "none (no slow roots)"; out["krein"] = []
        out["_nu_roots"] = []
        return out, d
    Pn = sp.Poly(sl, nu)
    cont, prim = Pn.primitive()
    prim = sp.Poly(prim.as_expr() * (1 if prim.LC() > 0 else -1), nu)
    fac = sp.factor(prim.as_expr())
    out["slow polynomial"] = str(fac)
    rs = roots_summary(prim.as_expr(), nu)
    out["roots"] = [{"factor": str(r["factor"]), "multiplicity": int(r["mult"]), "negative real": int(r["neg"]),
                     "positive real": int(r["pos"]), "non-real": int(r["nonreal"])} for r in rs]
    out["words"] = words(rs)
    # roots with multiplicity, exact (CRootOf) for pairing
    nr = []
    for r in rs:
        for root in sp.Poly(r["factor"], nu).all_roots():
            nr += [root] * r["mult"]
    out["_nu_roots"] = nr
    # Krein signs for clusters +-i tau (negative real nu)
    kr = []
    seen = []
    for r in rs:
        for root in sp.Poly(r["factor"], nu).all_roots():
            if root.is_real and root < 0:
                s60 = krein(d, root, 60); s90 = krein(d, root, 90)
                chk(f"{slot} {name}: Krein sign at nu={sp.N(root, 12)} same at 60 and 90 digits", s60[0] == s90[0] and s60[1] == s90[1], quiet=True)
                kr.append({"nu": str(root), "nu_num": str(sp.N(root, 30)), "multiplicity": r["mult"],
                           "dim ker(S^2+tau^2)": s90[1], "Krein": s90[0],
                           "restricted M-hat eigenvalues (90 dig)": [mp.nstr(e, 15) for e in s90[2]],
                           "max kernel singular value (90 dig)": mp.nstr(s90[3], 3), "smallest non-kernel s.v.": mp.nstr(s90[4], 5)})
    out["krein"] = kr
    return out, d

def sortkey(root):
    v = complex(sp.N(root, 50))
    re = 0.0 if (root.is_real is False and sp.re(root) == 0) else v.real
    im = 0.0 if root.is_real else v.imag
    if abs(re) < 1e-40: re = 0.0
    return (re, im)

def xi_plus(root):
    if root.is_real:
        return sp.sqrt(root) if root > 0 else sp.I * sp.sqrt(-root)
    return sp.sqrt(root)          # principal branch, Re > 0 for non-real root

results = {"B.1": {}, "B.2": {}, "B.5": {}}
store = {}
qJ = {"R4": {int(k[2:]): v for k, v in A["A.1"]["R4"].items() if k.startswith("q_")},
      "R5": {int(k[2:]): v for k, v in A["A.1"]["R5"].items() if k.startswith("q_")}}
for slot, item in (("R4", "B.1"), ("R5", "B.2")):
    print(f"\n== {item} ({slot}); quartic h = sum_J q_J ||[w(x)w]_J||^2, q_J = {qJ[slot]}")
    for name, comp in inp["census"]["points"].items():
        w = [sp.sympify(comp.get(str(m), "0")) for m in MS]
        o, d = analyse(slot, qJ[slot], w, name)
        store[(slot, name)] = (o, d)
        print(f"-- {name}: critical={o['critical']}  Q={o['Q']}  924 r6hat={o['924 r6hat']}  signature={o['signature']}  "
              f"k={o['k']}  r_u={o['r_u']}  zero mult={o['zero multiplicity']}")
        print(f"   slow polynomial: {o['slow polynomial']}")
        for r in o["roots"]: print("   ", r)
        print(f"   words: {o['words']}")
        for k in o["krein"]: print("   Krein:", k)
        results[item][name] = {k: v for k, v in o.items() if not k.startswith("_")}

# planted defects for B
def crit_test(w):
    return PointData(quartic(qJ["R4"]), w, gmats).critical
planted("criticality test on the pentagonal pyramid with sqrt(13) -> sqrt(14)",
        lambda: crit_test([sp.sqrt(12), 0, 0, 0, 0, sp.sqrt(14), 0]))
o, d = store[("R4", "v1")]
bad_gen = [gmats[0], gmats[1], gmats[2], sp.diag(*range(1, 15))]
planted("'M-hat vanishes on K_u' with a non-symmetry matrix in place of the z-rotation (v1, R4)",
        lambda: PointData(quartic(qJ["R4"]), [0, 0, 1, 0, 0, 0, 0], bad_gen).MK_zero)
# Krein: a hyperbolic-type or mixed subspace must come out indefinite: test the classifier on
# ker(S^2+tau1^2) + ker(S^2+tau2^2) of two clusters of opposite sign if one exists
mixed = None
for (slot, name), (o, d) in store.items():
    sg = {k["Krein"] for k in o["krein"]}
    if "+" in sg and "-" in sg:
        mixed = (slot, name, o, d); break
if mixed:
    slot, name, o, d = mixed
    kp = [k for k in o["krein"] if k["Krein"] == "+"][0]; km = [k for k in o["krein"] if k["Krein"] == "-"][0]
    def mixed_test():
        with mp.workdps(60):
            S = mp_matrix(d.Sop, 60); Mo = mp_matrix(d.Mop, 60); G = mp.diag([mp.mpf(g) for g in GDIAG])
            cols = []
            for k in (kp, km):
                t2 = -mp.mpf(k["nu_num"].split()[0]) if False else -mp.mpmathify(sp.N(sp.sympify(k["nu"]), 80))
                U, Sv, V = mp.svd_r(S * S + t2 * mp.eye(14))
                cols += [[V[i, r] for r in range(14)] for i in range(14) if Sv[i] < mp.mpf(10) ** -30]
            B = mp.matrix(14, len(cols))
            for c, v in enumerate(cols):
                for r in range(14): B[r, c] = v[r]
            F = B.T * G * Mo * B; F = (F + F.T) / 2
            ev = mp.eigsy(F)[0]
            return all(ev[i] > 0 for i in range(len(cols))) or all(ev[i] < 0 for i in range(len(cols)))
    planted(f"Krein definiteness test on the union of a + and a - cluster ({slot}, {name})", mixed_test)
else:
    print("(no point with clusters of both Krein signs; mixed-cluster planted test not available)")

# --- clause 3 pairing and ratios
print("\n== B.2 comparison R5 vs R4 (clause 3 pairing)")
cmp = {}
for name in inp["census"]["points"]:
    o4 = store[("R4", name)][0]; o5 = store[("R5", name)][0]
    n4 = sorted(o4["_nu_roots"], key=sortkey); n5 = sorted(o5["_nu_roots"], key=sortkey)
    entry = {"R4 nonzero eigenvalue count": 2 * len(n4), "R5 nonzero eigenvalue count": 2 * len(n5)}
    if len(n4) != len(n5):
        entry["note"] = "different numbers of slow roots: no pairing possible"
    else:
        pairs = []
        for a, b in zip(n4, n5):
            x4 = xi_plus(a); x5 = xi_plus(b)
            ratio = sp.radsimp(sp.nsimplify(x5 / x4) if False else sp.simplify(x5 / x4))
            num = sp.N(x5 / x4, 40)
            pairs.append({"nu_R4": str(a), "nu_R5": str(b), "+xi_R4": str(sp.N(x4, 25)), "+xi_R5": str(sp.N(x5, 25)),
                          "ratio +xi_R5/+xi_R4 (= ratio of -xi's)": str(ratio), "ratio (40 dig)": str(num)})
        entry["pairs"] = pairs
        # exact test of a common ratio c: slow poly of R5 in nu is proportional to that of R4 at nu / c^2.
        # Then every nu_R5 = c^2 nu_R4, the sort order is preserved (c^2 > 0) and +xi_R5 = c (+xi_R4).
        p4 = sp.Poly(sp.sympify(o4["slow polynomial"]), nu); p5 = sp.Poly(sp.sympify(o5["slow polynomial"]), nu)
        nums = [complex(sp.N(sp.sympify(p["ratio (40 dig)"]), 40)) for p in pairs]
        c = sp.nsimplify(sp.N(sp.sympify(pairs[0]["ratio (40 dig)"]), 40), rational=True, tolerance=1e-30)
        def prop_test(c):
            q = sp.Poly(sp.expand(p4.as_expr().subs(nu, nu / c ** 2)), nu)
            return sp.simplify(q.LC() * p5.as_expr() - p5.LC() * q.as_expr()) == 0
        ok = prop_test(c)
        chk(f"{name}: slow poly R5(nu) proportional to slow poly R4(nu/c^2), c = {c} (exact)", ok)
        if name == "v1":
            planted("same exact ratio test with c = 3/4", lambda: prop_test(sp.Rational(3, 4)))
        entry["exact common ratio"] = str(c) if ok else "not established"
        for p in pairs:
            p["ratio +xi_R5/+xi_R4 (= ratio of -xi's)"] = str(c) if ok else p["ratio +xi_R5/+xi_R4 (= ratio of -xi's)"]
        rats = set(p["ratio +xi_R5/+xi_R4 (= ratio of -xi's)"] for p in pairs)
        entry["distinct ratios"] = sorted(rats)
    cmp[name] = entry
    print(f"-- {name}: {entry.get('distinct ratios', entry.get('note', 'no slow roots'))}")
    for p in entry.get("pairs", []): print("    ", p)
    # same k, r_u, zero multiplicity, signature?
    same = {k: (o4[k], o5[k]) for k in ("critical", "signature", "k", "r_u", "zero multiplicity", "words")}
    print("    R4 vs R5:", same)
results["B.2"]["_comparison_with_R4"] = cmp
results["B.5"] = {"proportional": True, "constant": "1",
                  "statement": "||(ww^+)_6||^2 = ||[w (x) Theta w]_6||^2 identically on V_3 (exact polynomial identity)"}
json.dump(results, open(os.path.join(OUT, "blockB.json"), "w"), indent=1, default=str)
chk.summary()
