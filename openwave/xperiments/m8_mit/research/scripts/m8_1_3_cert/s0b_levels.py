"""Item 0(b): for each irreducible representation, every level l <= 20 at which it occurs in
V_{l/2} restricted to Gamma_0, with multiplicity (exact character inner products), and the
check that the names R2, R4, R5 of Section 1.1 each pick out exactly one representation."""
import json, os
from lib_alg import Q5, generators, closure, conjugacy_classes, element_order, cheb_U, inner
from s0a_group import derive_character_table
from lib_check import check, finish

HERE = os.path.dirname(os.path.abspath(__file__))


def setup():
    G = closure(list(generators()))
    classes = conjugacy_classes(G)
    classes.sort(key=lambda c: (element_order(c[0]), -float(c[0][0])))
    irr, sizes, reps = derive_character_table(G, classes)
    return G, classes, irr, sizes, reps


def main():
    G, classes, irr, sizes, reps = setup()
    order = len(G)
    table = []
    for chi in irr:
        mult = {}
        for l in range(0, 21):
            ch = [cheb_U(l, r[0]) for r in reps]
            m = inner(ch, chi, sizes, order)
            assert m.b == 0 and m.a.denominator == 1 and m.a >= 0
            if m.a != 0:
                mult[l] = int(m.a)
        dim = int(chi[0].a)
        table.append({"dim": dim, "lowest level": min(mult) if mult else None,
                      "levels<=20 {level: multiplicity}": mult,
                      "values": [v.sympy_str() for v in chi]})
    table.sort(key=lambda t: (t["dim"], t["lowest level"]))
    for t in table:
        print(f"dim {t['dim']}, lowest level {t['lowest level']}: {t['levels<=20 {level: multiplicity}']}")
    # consistency: sum over irreps of dim*mult == l+1 for every level
    for l in range(21):
        tot = sum(t["dim"] * t["levels<=20 {level: multiplicity}"].get(l, 0) for t in table)
        if tot != l + 1:
            check(f"dimension count at level {l}", False, f"{tot} != {l+1}")
    check("sum_rho dim(rho)*mult(rho, l) == l+1 for l = 0..20", all(
        sum(t["dim"] * t["levels<=20 {level: multiplicity}"].get(l, 0) for t in table) == l + 1
        for l in range(21)))
    names = {"R2": (2, 7), "R4": (3, 6), "R5": (4, 6)}
    pick = {}
    for nm, (d, n) in names.items():
        hits = [t for t in table if t["dim"] == d and t["lowest level"] == n]
        check(f"name {nm} (dim {d}, lowest level {n}) picks exactly one irreducible", len(hits) == 1,
              f"{len(hits)} hits")
        pick[nm] = hits[0] if hits else None
        if hits:
            print(f"{nm}: multiplicity at lowest level {n} = {hits[0]['levels<=20 {level: multiplicity}'][n]}")
    with open(os.path.join(HERE, "out_0b.json"), "w") as f:
        json.dump({"table": table, "names": pick}, f, indent=1)
    finish()


if __name__ == "__main__":
    main()
