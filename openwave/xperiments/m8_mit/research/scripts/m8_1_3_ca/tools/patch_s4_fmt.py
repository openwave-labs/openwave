"""one-off edit: print/store interval endpoints as outward-rounded decimal strings."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "s4_boxes.py"); s = open(p).read()
helper = '''
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, getcontext
getcontext().prec = 400
def dec_out(x, digits, up):
    """decimal string with `digits` significant digits, rounded down (up=False) or up (up=True): exact from the binary value."""
    x = mp.mpf(x)
    if x == 0:
        return "0"
    man, exp = mp.mpf(x).man_exp
    d = Decimal(int(man)) * (Decimal(2) ** int(exp))
    q = Decimal(1).scaleb(d.adjusted() - digits + 1)
    return str(d.quantize(q, rounding=ROUND_CEILING if up else ROUND_FLOOR))
def encl(v, digits=45):
    return [dec_out(v.a, digits, False), dec_out(v.b, digits, True)]
'''
a = "def mid(v):"
s = s.replace(a, helper + "\n" + a, 1)
old = s[s.index('    check(f"C.4 {s} {name}: R0~ has constant sign over the box"'):s.index("if not PLANT:\n    json.dump(res") if False else s.index("# ---- independent 130-digit")]
new = '''    check(f"C.4 {s} {name}: R0~ has constant sign over the box", sign in ("+", "-"), str(encl(R0, 30)))
    lamK = K[14]
    print(f"{s} {name}: certified={ok}  lambda (= 4h at the zero, from K(B)) in {encl(lamK, 40)}")
    print(f"      R0~ over box in {encl(R0, 45)}, width {mp.nstr(mp.mpf(R0.delta), 3)}  sign {sign}")
    res.setdefault(s, {})[name] = {
        "Krawczyk certified (unique zero of the slice system in the box)": bool(ok),
        "critical (c = 0 at the certified zero)": bool(ok),
        "R0~ enclosure over the box": encl(R0, 45),
        "R0~ enclosure width": mp.nstr(mp.mpf(R0.delta), 5),
        "sign": sign,
    }
'''
s = s.replace(old, new)
s = s.replace('res.setdefault("_hp_points", {}).setdefault(s, {})[name] = mp.nstr(mid(R0p), 60)',
              'res.setdefault("_hp_points", {}).setdefault(s, {})[name] = encl(R0p, 60)')
s = s.replace("check(f\"D {s} {name}: Krawczyk K(B) inside int(B)\", inside, f\"max width ratio {mp.nstr(ratio, 3)}, ||I - Y J(B)||_inf <= {mp.nstr(normE, 3)}\")",
              "check(f\"D {s} {name}: Krawczyk K(B) inside int(B)\", inside, f\"max width ratio {mp.nstr(mp.mpf(ratio), 3)}, ||I - Y J(B)||_inf <= {mp.nstr(mp.mpf(normE), 3)}\")")
s = s.replace('f"min diag {mp.nstr(diagmin, 5)}, max off-diag row sum {mp.nstr(offn, 5)}"',
              'f"min diag >= {dec_out(diagmin.a if hasattr(diagmin, \'a\') else diagmin, 8, False)}, max off-diag row sum <= {dec_out(offn.b if hasattr(offn, \'b\') else offn, 3, True)}"')
open(p, "w").write(s)
print("patched")
