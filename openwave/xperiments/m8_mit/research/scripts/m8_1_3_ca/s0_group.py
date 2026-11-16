"""Item 0(a) and 0(b): the group from its two generators, classes, characters (DERIVED here), levels."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from common import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

q1, q2 = generators()
print("q1 =", q1, " q2 =", q2)
check("0(a) |q1|^2 == 1 exactly", q1.norm2() == Q5(1), str(q1.norm2()))
check("0(a) |q2|^2 == 1 exactly", q2.norm2() == Q5(1), str(q2.norm2()))

G = closure([q1, q2])
order = len(G)
print("order of Gamma_0 =", order)
check("0(a) closure is closed under products and inverses",
      all((g * h) in set(G) for g in G for h in (q1, q2)) and all(g.conj() in set(G) for g in G))

def elt_order(g):
    one = Quat(1, 0, 0, 0); p = g; k = 1
    while not (p == one):
        p = p * g; k += 1
    return k

# conjugacy classes by explicit conjugation
Gs = set(G); seen = set(); classes = []
for g in G:
    if g in seen:
        continue
    cl = {h * g * h.conj() for h in G}
    seen |= cl
    classes.append(sorted(cl, key=lambda q: str(q)))
classes.sort(key=lambda cl: (-cl[0].c[0].mpf(), len(cl)))
print("number of classes:", len(classes))
cls_info = []
for k, cl in enumerate(classes):
    re = cl[0].c[0]
    same_re = all(q.c[0] == re for q in cl)
    info = dict(index=k, size=len(cl), order=elt_order(cl[0]), real_part=str(re.sym()),
                representative=str(cl[0]))
    cls_info.append(info)
    print(f"  class {k}: size {len(cl):3d}, element order {info['order']:2d}, Re = {info['real_part']}  (all same Re: {same_re})")
check("0(a) class sizes sum to the order", sum(len(c) for c in classes) == order)
check("0(a) real part is constant on each class and distinct between classes",
      all(all(q.c[0] == cl[0].c[0] for q in cl) for cl in classes)
      and len({cl[0].c[0] for cl in classes}) == len(classes))
sizes = [len(c) for c in classes]

# characters of V_j: trace of D^j, exact, from the weighted-coordinate matrices; cross-check Chebyshev U_{2j}(Re)
def chi_V_trace(j2):
    out = []
    for cl in classes:
        Dw = D_weighted_exact(sp.Rational(j2, 2), cl[0])
        t = C5(0)
        for r in range(j2 + 1):
            t = t + Dw[r][r]
        out.append(t)
    return out

def chi_V_cheb(j2):
    out = []
    for cl in classes:
        w = cl[0].c[0]
        U = [Q5(1), 2 * w]
        for n in range(2, j2 + 1):
            U.append(2 * w * U[-1] - U[-2])
        out.append(C5(U[j2]))
    return out

chiV = {}
for j2 in range(0, 41):
    chiV[j2] = chi_V_cheb(j2)
agree = all(chi_V_trace(j2)[k] == chiV[j2][k] for j2 in range(0, 9) for k in range(len(classes)))
check("0(a) character of V_j: trace of D^j == Chebyshev U_2j(Re g), 2j = 0..8, exact", agree)

def inner(c1, c2):
    s = C5(0)
    for k, n in enumerate(sizes):
        s = s + n * c1[k] * c2[k].conj()
    return C5(s.r / order, s.i / order)

def mul(c1, c2): return [a * b for a, b in zip(c1, c2)]
def sub(c1, c2): return [a - b for a, b in zip(c1, c2)]

irr = {}
# V_j for 2j = 0..5 are irreducible on Gamma_0 (norm 1)
names_V = {0: "1", 1: "2", 2: "3", 3: "4s", 4: "5", 5: "6"}
for j2, nm in names_V.items():
    nrm = inner(chiV[j2], chiV[j2])
    check(f"0(a) <chi_V_{j2}/2, chi_V_{j2}/2> == 1 (irreducible '{nm}')", nrm == C5(1), str(nrm))
    irr[nm] = chiV[j2]
# 2' := V_{7/2} minus its '6' part
m6 = inner(chiV[7], irr["6"]); print("<V_7/2, 6> =", m6)
c = sub(chiV[7], irr["6"]) if m6 == C5(1) else None
check("0(a) V_7/2 - 6 has norm 1 (irreducible 2')", c is not None and inner(c, c) == C5(1))
irr["2'"] = c
# 4 := 2 (x) 2'
c = mul(irr["2"], irr["2'"])
check("0(a) 2 (x) 2' has norm 1 (irreducible '4')", inner(c, c) == C5(1))
irr["4"] = c
# 3' := V_3 minus its '4' part
m4 = inner(chiV[6], irr["4"]); print("<V_3, 4> =", m4)
c = sub(chiV[6], irr["4"])
check("0(a) V_3 - 4 has norm 1 (irreducible 3')", m4 == C5(1) and inner(c, c) == C5(1))
irr["3'"] = c
names = ["1", "2", "2'", "3", "3'", "4", "4s", "5", "6"]
dims = {nm: irr[nm][0] for nm in names}
ok = all(inner(irr[a], irr[b]) == C5(1 if a == b else 0) for a in names for b in names)
check("0(a) the nine characters are orthonormal", ok)
sq = sum((dims[nm] * dims[nm] for nm in names), C5(0))
check("0(a) sum of squared degrees == |Gamma_0| (table complete)", sq == C5(order), str(sq))
check("0(a) number of irreducibles == number of classes", len(names) == len(classes))
# observation: Galois conjugation
check("0(a) observation: 2' and 3' are the Galois (sqrt5 -> -sqrt5) conjugates of 2 and 3",
      [x.conj5() for x in irr["2"]] == irr["2'"] and [x.conj5() for x in irr["3"]] == irr["3'"])
kneg = [k for k, cl in enumerate(classes) if cl[0] == Quat(-1, 0, 0, 0)][0]
check("0(a) the class of -1 is a singleton", len(classes[kneg]) == 1)
faithful = {nm: (irr[nm][kneg] == -irr[nm][0]) for nm in names}

# Frobenius-Schur indicators, exact: (1/|G|) sum_g chi(g^2); -1 <=> quaternionic (C conj(C) = -1), +1 <=> real
cls_index = {}
for k, cl in enumerate(classes):
    for q in cl:
        cls_index[q] = k
fs = {}
for nm in names:
    t = C5(0)
    for g in G:
        t = t + irr[nm][cls_index[g * g]]
    fs[nm] = C5(t.r / order, t.i / order)
print("Frobenius-Schur indicators:", {nm: str(v) for nm, v in fs.items()})
check("0(e) Frobenius-Schur indicator of 2' (slot R2) is -1 exactly: C conj(C) = -1", fs["2'"] == C5(-1))
check("Frobenius-Schur indicators of 3' and 4 (R4, R5) are +1 exactly: C conj(C) = +1", fs["3'"] == C5(1) and fs["4"] == C5(1))

print("\nCharacter table (columns = classes 0..8):")
print("  sizes :", sizes)
print("  orders:", [ci["order"] for ci in cls_info])
for nm in names:
    print(f"  {nm:3s}:", [str(x.sym()) for x in irr[nm]], " (-1 acts as", "-1)" if faithful[nm] else "+1)")

# 0(b): levels <= 20
print("\n0(b) multiplicity of each irreducible in V_{l/2}, l = 0..20 (only nonzero listed):")
levels = {}
for nm in names:
    lv = {}
    for l in range(0, 21):
        m = inner(chiV[l], irr[nm])
        assert m.i.is0() and m.r.b == 0 and m.r.a.denominator == 1
        if m.r.a != 0:
            lv[l] = int(m.r.a)
    levels[nm] = lv
    print(f"  {nm:3s} (dim {dims[nm]}):", lv)
check("0(b) multiplicities sum to dim V_{l/2} for every l <= 20",
      all(sum(levels[nm].get(l, 0) * int(dims[nm].r.a) for nm in names) == l + 1 for l in range(21)))

slots = {"R2": (2, 7), "R4": (3, 6), "R5": (4, 6)}
slot_irr = {}
for s, (dm, n) in slots.items():
    cands = [nm for nm in names if int(dims[nm].r.a) == dm and levels[nm] and min(levels[nm]) == n]
    print(f"slot {s}: dim {dm}, lowest level {n} -> candidates {cands}")
    check(f"0(b) slot {s} names exactly one irreducible", len(cands) == 1)
    slot_irr[s] = cands[0]
    print(f"   multiplicity at level {n}: {levels[cands[0]][n]}")

res = {
    "order": order,
    "classes": cls_info,
    "characters": {nm: [str(x.sym()) for x in irr[nm]] for nm in names},
    "levels": {nm: {str(l): m for l, m in levels[nm].items()} for nm in names},
    "slot_irreducible": slot_irr,
}
json.dump(res, open(os.path.join(OUT, "group.json"), "w"), indent=1)
# character values, exact, for later scripts: also class representatives
json.dump({"classes": [[[str(t.sym()) for t in q.c] for q in cl] for cl in classes],
           "chars": {nm: [[str(x.r.a), str(x.r.b), str(x.i.a), str(x.i.b)] for x in irr[nm]] for nm in names}},
          open(os.path.join(OUT, "group_data.json"), "w"))
print("failures:", n_fails())
