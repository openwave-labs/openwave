"""Items 6 and 10, derived exactly from the outputs of items 4, 5 and 5b.

Item 6: Morse index of E = g r6 at each orbit: n- for g > 0, n+ for g < 0 (on N_u).
Item 10: Q = 1 + w r6 with w = 28/39 or 21/52 (w > 0): values 1 + w r6, Hessian w H_u,
eigenvalues scaled by w, signatures unchanged; the item-5b numbers multiplied by w.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from core import report

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
i5 = json.load(open(os.path.join(HERE, "out", "item5.json")))
i5b = json.load(open(os.path.join(HERE, "out", "item5b.json")))
LAM = sp.Symbol("lam")
out6 = {}
print("item 6: orbit, value, (n-, n0, n+), index g>0, index g<0, extremum")
for name, rec in i5.items():
    nneg, n0, npos = rec["signature"]
    ext_pos = "local min of E (g>0)" if (nneg == 0 and n0 == 0) else ("local max of E (g>0)" if (npos == 0 and n0 == 0) else "")
    ext_neg = "local min of E (g<0)" if (npos == 0 and n0 == 0) else ("local max of E (g<0)" if (nneg == 0 and n0 == 0) else "")
    out6[name] = {"value": rec["value"], "signature": rec["signature"], "index_g_pos": nneg, "index_g_neg": npos,
                  "kernel": n0, "extremum_g_pos": ext_pos, "extremum_g_neg": ext_neg}
    print("  %-10s %-8s %s  %d  %d  %s %s%s" % (name, rec["value"], tuple(rec["signature"]), nneg, npos, ext_pos, ext_neg,
                                              "  [degenerate: kernel %d]" % n0 if n0 else ""))
ok &= report("exactly one orbit has n- = 0 (v3) and exactly one has n+ = 0 (hex), both with n0 = 0",
             [k for k, v in out6.items() if v["signature"][0] == 0] == ["v3"] and
             [k for k, v in out6.items() if v["signature"][2] == 0] == ["hex"] and
             out6["v3"]["kernel"] == 0 and out6["hex"]["kernel"] == 0)

W = {"sector w6=28/39": sp.Rational(28, 39), "sector w6=21/52": sp.Rational(21, 52)}
out10 = {}
for sec, w in W.items():
    rec10 = {"orbit_values_Q": {}, "charpolys_Q": {}, "item5b_times_w": {}}
    for name, rec in i5.items():
        r6 = sp.Rational(rec["value"])
        rec10["orbit_values_Q"][name] = str(1 + w * r6)
        cp = sp.sympify(rec["charpoly"], locals={"lam": LAM})
        # eigenvalues of w H are w * eigenvalues of H: charpoly_Q(lam) = w^n charpoly(lam / w)
        n = sp.Poly(cp, LAM).degree()
        rec10["charpolys_Q"][name] = str(sp.factor(sp.expand(w ** n * cp.subs(LAM, LAM / w))))
    for lab, rb in i5b.items():
        Hm = [[str(sp.Rational(v) * w) for v in row] for row in rb["transverse"]["H"]]
        nulls = {k: str(sp.Rational(v["H_u_unit"]) * w) for k, v in rb["null"].items()}
        rec10["item5b_times_w"][lab] = {"H_transverse_times_w": Hm, "Gram_unchanged": rb["transverse"]["Gram"],
                                        "null_controls_times_w": nulls, "directions": rb["transverse"]["directions"]}
    out10[sec] = rec10
    print("\n%s: Q values:" % sec, rec10["orbit_values_Q"])
    for lab, v in rec10["item5b_times_w"].items():
        print("   5b x w  %-18s H = %s   null = %s" % (lab, v["H_transverse_times_w"], v["null_controls_times_w"]))
ok &= report("ordering of orbit values is the same for r6 and both Q (w > 0)",
             all(sorted(i5, key=lambda k: sp.Rational(i5[k]["value"])) ==
                 sorted(i5, key=lambda k: sp.Rational(out10[s]["orbit_values_Q"][k])) for s in out10))
json.dump({"item6": out6, "item10": out10}, open(os.path.join(HERE, "out", "item6_10.json"), "w"), indent=1)
print("ITEM6_10", "ALLPASS" if ok else "SOMEFAIL")
