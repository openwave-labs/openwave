"""Item E.1: slots rho (trivial included) with V_{n/2}|Gamma_0 = rho at their lowest level n,
the next level where rho occurs, and lambda_1 - lambda_0 (units 1/R^2). Reads out/s01.json (exact table to l = 60)."""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from checks import check, summary, PLANT

d = json.load(open(os.path.join(HERE, "out", "s01.json")))
mult, dims = d["_mult_to_60"], d["_dims"]
if PLANT == "restrict":
    mult = {k: dict(v) for k, v in mult.items()}
    mult["dim5_n4"]["4"] = 2  # planted: multiplicity 2 at the lowest level -> must not count as rigid
if PLANT == "next":
    mult = {k: dict(v) for k, v in mult.items()}
    mult["dim3_n2"]["6"] = 1  # planted: a spurious occurrence at level 6
rows = {}
for lab in sorted(mult, key=lambda s: (dims[s], s)):
    lv = sorted(int(l) for l in mult[lab])
    n = lv[0]
    rigid = mult[lab][str(n)] == 1 and dims[lab] == n + 1  # V_{n/2} (dim n+1) is exactly rho
    print(f"{lab}: dim {dims[lab]}, lowest level {n}, mult {mult[lab][str(n)]}, dim V_(n/2) = {n+1} -> rigid: {rigid}")
    if rigid:
        nxt = lv[1]
        rows[lab] = {"dim": dims[lab], "n": n, "next level": nxt, "lambda1 - lambda0": str(nxt * (nxt + 2) - n * (n + 2))}
print()
for lab, r in rows.items():
    print(f"E.1 {lab}: n = {r['n']}, next level = {r['next level']}, lambda1-lambda0 = {r['lambda1 - lambda0']} /R^2")
# independent criterion from the exact characters (not from the multiplicity table):
# rho is rigid  <=>  chi_rho(g) = U_n(Re g) on every class, n = dim rho - 1;
# and the next level L is the first l > n with <chi_{V_{l/2}}, chi_rho> != 0.
import sympy
chars = {k: [sympy.sympify(v) for v in vals] for k, vals in d["0(a)"]["characters"].items()}
re_g = [sympy.sympify(c["Re g"]) for c in d["0(a)"]["classes"]]
sizes = [c["size"] for c in d["0(a)"]["classes"]]
indep = set()
nxt_indep = {}
for lab, ch in chars.items():
    n = int(ch[0]) - 1
    if all(sympy.simplify(sympy.chebyshevu(n, re_g[k]) - ch[k]) == 0 for k in range(len(ch))):
        indep.add(lab)
        for l in range(n + 1, 61):
            m = sympy.nsimplify(sympy.expand(sum(sizes[k] * sympy.chebyshevu(l, re_g[k]) * ch[k] for k in range(len(ch))) / 120))
            if m != 0:
                nxt_indep[lab] = (l, m)
                break
check("E.1 rigid set equals {rho : chi_rho = chi_{V_{n/2}} exactly} (sympy, from characters)", indep == set(rows), f"{sorted(indep)} vs {sorted(rows)}")
check("E.1 next levels agree with a direct sympy inner product", all(nxt_indep.get(lab, (None,))[0] == r["next level"] for lab, r in rows.items()),
      str({k: v[0] for k, v in nxt_indep.items()}))
if PLANT is None:
    json.dump(rows, open(os.path.join(HERE, "out", "s02.json"), "w"), indent=1)
summary()
