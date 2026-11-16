"""Item 0(a): norms of the generators, order of Gamma_0, conjugacy classes, irreducible
characters -- all DERIVED here from the two generators, in exact arithmetic over Q(sqrt5)."""
import json, os
from fractions import Fraction as Fr
from lib_alg import (Q5, generators, closure, conjugacy_classes, element_order, qnorm2, qmul,
                     cheb_U, inner)
from lib_check import check, finish

HERE = os.path.dirname(os.path.abspath(__file__))


def derive_character_table(G, classes):
    order = len(G)
    sizes = [len(c) for c in classes]
    reps = [c[0] for c in classes]
    nclass = len(classes)
    irr = []

    def try_add(cand):
        c = list(cand)
        for chi in irr:
            m = inner(cand, chi, sizes, order)
            c = [x - chi_k * m for x, chi_k in zip(c, chi)]
        n = inner(c, c, sizes, order)
        if n == Q5(1) and c[0].b == 0 and c[0].a > 0:
            irr.append(c)
            return True
        return False

    # candidates: characters of V_{n/2} (n = 0..40), their Galois conjugates, and products
    # of irreducibles already found.  Each candidate is a genuine character, so the remainder
    # after removing known constituents is a genuine character; norm 1 => irreducible.
    for rnd in range(4):
        for n in range(0, 41):
            ch = [cheb_U(n, r[0]) for r in reps]
            try_add(ch)
            try_add([x.galois() for x in ch])
        cur = list(irr)
        for a in cur:
            for b in cur:
                try_add([x * y for x, y in zip(a, b)])
        if sum(int(c[0].a) ** 2 for c in irr) == order:
            break
    return irr, sizes, reps


def main():
    q1, q2 = generators()
    out = {}
    n1, n2 = qnorm2(q1), qnorm2(q2)
    check("|q1|^2 == 1 exactly", n1 == Q5(1), repr(n1))
    check("|q2|^2 == 1 exactly", n2 == Q5(1), repr(n2))
    G = closure([q1, q2])
    order = len(G)
    print("order of Gamma_0 =", order)
    # group axioms on the closure: closed under product and inverse (exact)
    Gs = set(G)
    closed = all(qmul(a, b) in Gs for a in G[:40] for b in G)
    check("closure is closed under multiplication (first 40 x all)", closed)
    classes = conjugacy_classes(G)
    classes.sort(key=lambda c: (element_order(c[0]), -float(c[0][0])))
    sizes = [len(c) for c in classes]
    check("class sizes sum to |G|", sum(sizes) == order, str(sizes))
    print("number of classes =", len(classes))
    irr, sizes, reps = derive_character_table(G, classes)
    dims = [int(c[0].a) for c in irr]
    check("sum of squares of derived irreducible degrees == |G|", sum(d * d for d in dims) == order,
          str(dims))
    check("number of derived irreducibles == number of classes", len(irr) == len(classes))
    # orthonormality (rows) and column orthogonality, exact
    ok = all(inner(irr[a], irr[b], sizes, order) == Q5(1 if a == b else 0)
             for a in range(len(irr)) for b in range(len(irr)))
    check("row orthonormality of the derived table (exact)", ok)
    colok = True
    for k in range(len(classes)):
        for l in range(len(classes)):
            s = Q5(0)
            for chi in irr:
                s = s + chi[k] * chi[l]
            want = Q5(Fr(order, sizes[k])) if k == l else Q5(0)
            colok &= (s == want)
    check("column orthogonality of the derived table (exact)", colok)

    cls_out = []
    for c in classes:
        r = c[0]
        cls_out.append({"size": len(c), "element order": element_order(r),
                        "real part w = cos(angle/2)": r[0].sympy_str(),
                        "representative (w,x,y,z)": [t.sympy_str() for t in r]})
    # sort irreducibles by degree then by first level where they occur
    def lowest_level(chi):
        for n in range(0, 61):
            ch = [cheb_U(n, r[0]) for r in reps]
            if inner(ch, chi, sizes, order) != Q5(0):
                return n
        return None
    irr.sort(key=lambda c: (int(c[0].a), lowest_level(c)))
    irr_out = []
    for chi in irr:
        irr_out.append({"dim": int(chi[0].a), "lowest level": lowest_level(chi),
                        "values": [v.sympy_str() for v in chi]})
    print("classes (size, order, real part):")
    for c in cls_out:
        print("  ", c["size"], c["element order"], c["real part w = cos(angle/2)"])
    print("irreducible characters (dim, lowest level: values on the classes in that order):")
    for c in irr_out:
        print("  ", c["dim"], c["lowest level"], c["values"])
    out = {"order": order, "norms": [repr(n1), repr(n2)], "classes": cls_out,
           "characters": irr_out}
    with open(os.path.join(HERE, "out_0a.json"), "w") as f:
        json.dump(out, f, indent=1)
    finish()


if __name__ == "__main__":
    main()
