"""Items 0(a), 0(b): the group from its two generators, classes, characters (derived), and
multiplicities in V_{l/2} for l <= 20.  Exact arithmetic in Q(sqrt5).

Character table derivation: chi_{V_j}(g) = U_{2j}(Re g) (Chebyshev of the 2nd kind), exact.
Irreducibles are peeled: a virtual character (V_j minus the known irreducibles it contains) of
norm 1 and positive degree is an irreducible character.  Completeness: #irreducibles found =
#classes, and sum dim^2 = |G| (column orthogonality is checked too)."""
import os, sys, json
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import *

chk = Checker()
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)

print("== 0(a)")
n1, n2 = qnorm2(q1), qnorm2(q2)
print("||q1||^2 =", n1, " ||q2||^2 =", n2)
chk("||q1|| = ||q2|| = 1 (exact)", n1 == Q5(1) and n2 == Q5(1))

G = generate_group([q1, q2])
order = len(G)
print("order of Gamma_0 =", order)

def closure_ok(G):
    keys = set(qkey(g) for g in G)
    return all(qkey(qmul(a, b)) in keys for a in G for b in G)
chk("closed under multiplication (all |G|^2 products)", closure_ok(G))
planted("closure check on G minus one element", lambda: closure_ok(G[:-1]))

classes = conj_classes(G)
byk = {qkey(g): g for g in G}
reps = [byk[c[0]] for c in classes]
# sort classes by (real part of the representative, size)
info = []
for c in classes:
    g = byk[c[0]]
    # element order
    o, x = 1, g
    one = qkey((Q5(1), Q5(0), Q5(0), Q5(0)))
    while qkey(x) != one:
        x = qmul(x, g); o += 1
    info.append((-float(g[0]), len(c), o, g[0], c))
info.sort(key=lambda t: (t[0], t[2]))
classes = [t[4] for t in info]
reps = [byk[c[0]] for c in classes]
sizes = [len(c) for c in classes]
retr = [t[3] for t in info]
orders = [t[2] for t in info]
print("number of classes:", len(classes))
for k, (s, o, r) in enumerate(zip(sizes, orders, retr)):
    print(f"  class {k}: size {s:3d}, element order {o:2d}, Re g = {r}")
chk("class sizes sum to |G|", sum(sizes) == order)
chk("each class has a single real part (trace) value",
    all(len(set(tuple(byk[k][0].a for _ in [0]) + tuple([byk[k][0].b]) for k in c)) == 1 for c in classes))

def inner(a, b):
    """<a,b> = (1/|G|) sum_classes size a conj(b); characters here are real (Q(sqrt5))."""
    t = Q5(0)
    for s, x, y in zip(sizes, a, b):
        t = t + s * x * y
    return t / order

def chiV(n):
    return [cheb_U(n, r) for r in retr]

irr = []
peel_log = []
for n in range(0, 21):
    v = chiV(n)
    for c in irr:
        m = inner(v, c)
        v = [x - m * y for x, y in zip(v, c)]
    nv = inner(v, v)
    if nv == Q5(1) and v[0].a > 0:
        irr.append(v); peel_log.append((n, "new irreducible of dim %s" % v[0]))
    if len(irr) == len(classes): break
for n, s in peel_log: print(f"  peeled from V_{{{n}/2}}: {s}")
print("irreducibles found:", len(irr))
chk("#irreducibles = #classes", len(irr) == len(classes))
chk("sum of dim^2 = |G|", sum((c[0] * c[0]) for c in irr[0:]) == Q5(order) if irr else False)
orth = all(inner(a, b) == (Q5(1) if ia == ib else Q5(0)) for ia, a in enumerate(irr) for ib, b in enumerate(irr))
chk("row orthonormality (exact)", orth)
def col_orth(irr):
    for a in range(len(classes)):
        for b in range(len(classes)):
            t = Q5(0)
            for c in irr: t = t + c[a] * c[b]
            want = Q5(Fr(order, sizes[a])) if a == b else Q5(0)
            if not t == want: return False
    return True
chk("column orthogonality (exact)", col_orth(irr))
bad = [list(c) for c in irr]; bad[3] = [x + (Q5(1) if k == 5 else Q5(0)) for k, x in enumerate(bad[3])]
planted("column orthogonality with one table entry shifted by 1", lambda: col_orth(bad))
# integrality check: every character value is an algebraic integer of Q(sqrt5): a+b sqrt5 with 2a,2b integers, a-b integer
def alg_int(x): return (2 * x.a).denominator == 1 and (2 * x.b).denominator == 1 and (x.a - x.b).denominator == 1
chk("all character values are algebraic integers of Q(sqrt5)", all(alg_int(x) for c in irr for x in c))

# order irreducibles by dimension, then by first level
def first_level(c):
    for n in range(0, 61):
        if not inner(chiV(n), c).iszero(): return n
irr.sort(key=lambda c: (c[0].a, first_level(c)))
print("\ncharacter table (rows: irreducibles; columns: classes as above)")
tab = []
for c in irr:
    row = [str(sp.nsimplify(x.sym())) for x in c]
    tab.append(row)
    print(f"  dim {c[0]}  first level {first_level(c):2d}:", row)

print("\n== 0(b)  multiplicity of each irreducible in V_{l/2}, l = 0..20")
mult = {}
for idx, c in enumerate(irr):
    levs = {}
    for l in range(0, 21):
        m = inner(chiV(l), c)
        assert m.b == 0 and m.a.denominator == 1
        if m.a: levs[l] = int(m.a)
    mult[idx] = levs
    print(f"  irr #{idx} dim {c[0]}: " + ", ".join(f"l={l}:x{k}" for l, k in levs.items()))
# dimension bookkeeping: sum mult*dim = l+1
for l in range(21):
    tot = sum(mult[i].get(l, 0) * int(irr[i][0].a) for i in range(len(irr)))
    if tot != l + 1: chk(f"dim count at level {l}", False)
chk("sum over irreducibles of mult*dim = l+1 for every l <= 20", all(
    sum(mult[i].get(l, 0) * int(irr[i][0].a) for i in range(len(irr))) == l + 1 for l in range(21)))

# the slot names
def pick(dim, n):
    return [i for i, c in enumerate(irr) if c[0] == Q5(dim) and first_level(c) == n]
slots = {"R2": pick(2, 7), "R4": pick(3, 6), "R5": pick(4, 6)}
for k, v in slots.items():
    print(f"  {k}: irreducibles with that (dim, lowest level): {v}")
    chk(f"{k} names exactly one irreducible", len(v) == 1)

json.dump({
    "order": order, "class_sizes": sizes, "class_orders": orders,
    "class_retr": [[str(r.a), str(r.b)] for r in retr],
    "class_reps": [[[str(c.a), str(c.b)] for c in g] for g in reps],
    "chars": [[[str(x.a), str(x.b)] for x in c] for c in irr],
    "first_level": [first_level(c) for c in irr],
    "dims": [int(c[0].a) for c in irr],
    "mult": {str(i): {str(l): k for l, k in mult[i].items()} for i in mult},
    "slots": {k: v[0] for k, v in slots.items()},
    "table_str": tab,
}, open(os.path.join(OUT, "group.json"), "w"), indent=1)
chk.summary()
