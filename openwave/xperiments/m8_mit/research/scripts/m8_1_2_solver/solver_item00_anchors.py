"""Item 0: anchors (a) Theta v_2, (b) CG <3 3; 3 -3 | 6 0>, (c) generator packet."""
import sympy as sp

from solver_lib import (Q1, Q2, ONE, CG, generate_group, theta, save_results, s)

out = {}

# (a) Theta v_2 with global phase +1: Theta v_m = (-1)^m v_{-m}
e = [0] * 7
e[2 + 3] = 1
tv = theta(e)
nz = {int(a - 3): tv[a] for a in range(7) if tv[a] != 0}
out["0a_Theta_v2"] = {f"coeff_of_v_{m}": s(c) for m, c in nz.items()}
print("0(a) Theta v_2 =", " + ".join(f"({c}) v_{m}" for m, c in nz.items()))

# (b)
cg = CG(3, 3, 3, -3, 6, 0)
out["0b_CG_33_3m3_60"] = s(sp.radsimp(cg))
out["0b_float_crosscheck"] = float(cg)
print("0(b) <3 3; 3 -3|6 0> =", sp.radsimp(cg), "=", float(cg), "; 1/cg^2 =", 1 / cg ** 2)

# (c) generator packet
n1, n2 = Q1.norm2(), Q2.norm2()
print("0(c) |q1|^2 =", n1.to_sympy(), " |q2|^2 =", sp.simplify(n2.to_sympy()))
G = generate_group([Q1, Q2])
order = len(G)
print("     |Gamma| =", order)
comms = {g * h * g.conj() * h.conj() for g in G for h in G}
Dsub = generate_group(list(comms))
print("     |[Gamma,Gamma]| =", len(Dsub), " perfect:", len(Dsub) == order)


# element orders (extra, derived)
def order_of(q):
    k, p = 1, q
    while p != ONE:
        p = p * q
        k += 1
    return k


print("     order(q1) =", order_of(Q1), " order(q2) =", order_of(Q2))
from collections import Counter
hist = Counter(order_of(g) for g in G)
print("     element-order histogram:", dict(sorted(hist.items())))
out["0c"] = {
    "norm2_q1": s(n1.to_sympy()), "norm2_q2": s(sp.simplify(n2.to_sympy())),
    "order_Gamma": order, "order_derived_subgroup": len(Dsub),
    "Gamma_equals_derived_subgroup": len(Dsub) == order,
    "order_q1": order_of(Q1), "order_q2": order_of(Q2),
    "element_order_histogram": {str(k): v for k, v in sorted(hist.items())},
}
save_results("item00", out)
