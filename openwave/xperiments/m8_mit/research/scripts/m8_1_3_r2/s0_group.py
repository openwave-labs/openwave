"""Items 0(a), 0(b): group from the two generators, classes, characters (DERIVED), level multiplicities.
Usage: ./py s0_group.py [--plant=NAME]   planted defects: q2norm, extrachar
"""
import sys, json
from fractions import Fraction as Fr
from lib import *

PLANT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--plant=")), None)
chk = Checker()
q1, q2 = GEN_Q1, GEN_Q2
if PLANT == "q2norm":   # planted defect: perturb q2's j-part
    q2 = (q2[0], q2[1], Q5(Fr(51, 100)), q2[3])

n1, n2 = qnorm2(q1), qnorm2(q2)
print("|q1|^2 =", n1, " |q2|^2 =", n2)
chk("0(a) |q1| = |q2| = 1 exactly", n1 == Q5(1) and n2 == Q5(1), f"{n1}, {n2}")
if PLANT == "q2norm":
    chk.summary(); sys.exit(0)

G = generate_group([q1, q2])
print("order |Gamma0| =", len(G))
chk("0(a) group closes and is finite", len(G) > 0)
cls = conj_classes(G)
# order of an element
def order(g):
    one = (Q5(1), Q5(0), Q5(0), Q5(0)); h = g; n = 1
    while h != one:
        h = qmul(h, g); n += 1
    return n
info = []
for c in cls:
    w = c[0][0]
    chk("class has a single real part", all(g[0] == w for g in c))
    info.append(dict(size=len(c), order=order(c[0]), re=w))
info.sort(key=lambda d: (d["order"], -float(d["re"].a + d["re"].b * 5 ** 0.5)))
cls_sorted = sorted(cls, key=lambda c: (order(c[0]), -float(c[0][0].a + c[0][0].b * 5 ** 0.5)))
print("conjugacy classes (order, size, real part w = Re q):")
for d in info:
    print(f"   order {d['order']:2d}  size {d['size']:2d}  w = {d['re'].sym()}")
chk("class sizes sum to |G|", sum(d["size"] for d in info) == len(G))
sizes = [d["size"] for d in info]; ws = [d["re"] for d in info]
nC = len(info)

# characters of V_j restricted (chi = U_{2j}(w)); verify against traces of D^j for small j
for n in range(0, 6):
    for c in cls_sorted:
        U = U_of_q(c[0]); M = Mp(n, U, K(0), K(1))
        tr = sum((M[i][i] for i in range(n + 1)), K(0))
        if not (tr == K(chebU(n, c[0][0]))):
            chk(f"trace D^{n}/2 = U_n(w)", False); break
    else:
        chk(f"trace of D^({n}/2) on a class rep equals U_{n}(w) on every class", True)

def ip(a, b):
    return sum((Q5(sizes[i]) * a[i] * b[i] for i in range(nC)), Q5(0)) / Q5(len(G))

def Vchar(n):
    return [chebU(n, ws[i]) for i in range(nC)]

irr = []  # derived irreducibles
pending = list(range(0, 41))  # levels 0..40
changed = True
while changed:
    changed = False
    for n in pending:
        res = Vchar(n)
        for chi in irr:
            m = ip(res, chi)
            res = [res[i] - m * chi[i] for i in range(nC)]
        nr = ip(res, res)
        if nr == Q5(1) and res[0].b == 0 and res[0].a > 0:
            irr.append(res); changed = True
if PLANT == "extrachar":   # planted defect: duplicate a character
    irr.append(irr[-1])
dims = [int(ch[0].a) for ch in irr]
print("derived irreducible characters (dims):", dims)
chk("number of irreducibles = number of classes", len(irr) == nC, f"{len(irr)} vs {nC}")
chk("sum of dim^2 = |G|", sum(d * d for d in dims) == len(G), str(sum(d * d for d in dims)))
ortho = all(ip(irr[a], irr[b]) == Q5(1 if a == b else 0) for a in range(len(irr)) for b in range(len(irr)))
chk("row orthonormality exact", ortho)
colok = True
for p in range(nC):
    for q in range(nC):
        s = sum((irr[a][p] * irr[a][q] for a in range(len(irr))), Q5(0))
        tgt = Q5(Fr(len(G), sizes[p])) if p == q else Q5(0)
        colok &= (s == tgt)
chk("column orthogonality exact", colok)

# order irreps: by dim, then by the lowest level
def mult(n, chi):
    return ip(Vchar(n), chi)
lev = {}
for a, chi in enumerate(irr):
    lev[a] = [(l, int(mult(l, chi).a)) for l in range(0, 21) if not mult(l, chi).iszero()]
    chk(f"multiplicities integral for irrep {a}", all(mult(l, chi).b == 0 and mult(l, chi).a.denominator == 1 for l in range(21)))
order_ir = sorted(range(len(irr)), key=lambda a: (dims[a], lev[a][0][0]))
print("\ncharacter table (columns = classes in the order above):")
table = []
for a in order_ir:
    vals = [str(sp.nsimplify(irr[a][i].sym())) for i in range(nC)]
    table.append(dict(dim=dims[a], lowest=lev[a][0][0], values=vals, levels=lev[a]))
    print(f"  dim {dims[a]} lowest level {lev[a][0][0]:2d}:", vals)
print("\nlevels l <= 20 with multiplicity (l, mult):")
for t in table:
    print(f"  dim {t['dim']} : {t['levels']}")

names = {"R2": (2, 7), "R4": (3, 6), "R5": (4, 6)}
picked = {}
for nm, (d, l) in names.items():
    hits = [t for t in table if t["dim"] == d and t["lowest"] == l]
    chk(f"0(b) name {nm} (dim {d}, lowest level {l}) picks exactly one irreducible", len(hits) == 1, str(len(hits)))
    if hits:
        picked[nm] = hits[0]
        print(f"  {nm}: character {hits[0]['values']}, multiplicity at lowest level {dict(hits[0]['levels'])[l]}")

out = dict(order=len(G), classes=[dict(order=d["order"], size=d["size"], re=str(d["re"].sym())) for d in info],
           characters=table, slots={k: v["values"] for k, v in picked.items()})
if PLANT is None:
    json.dump(out, open(rel("out_group.json"), "w"), indent=1)
chk.summary()
