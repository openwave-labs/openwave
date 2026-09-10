"""Item 16, first half: which ranks K can rho^{(j)}_K = [u (x) Theta_j u]_K carry, j = 1/2..3.

Exact: ||rho^{(j)}_K(v_j)||^2 = <j j; j -j | K 0>^2 for every K = 0..2j (eps_j = 1; the norm does not
depend on eps_j). Also at a generic exact state, to show odd K are carried in general.
"""
import json
import pathlib
import sympy as sp
from audit_su2lib import couple, theta, ms, norm2

HERE = pathlib.Path(__file__).parent
res = {}
for n in range(1, 7):
    j = sp.Rational(n, 2)
    mm = ms(j)
    vj = [sp.Integer(0)] * len(mm)
    vj[mm.index(j)] = sp.Integer(1)
    row = [norm2(couple(vj, theta(vj, j, eps=1), j, j, K)) for K in range(0, n + 1)]
    assert sp.nsimplify(sum(row)) == 1
    # generic exact state
    ug = [sp.Integer(k + 1) + sp.I * sp.Integer((-1) ** k * (k % 3)) for k in range(len(mm))]
    nu = sum(sp.expand(x * sp.conjugate(x)) for x in ug)
    rowg = [sp.nsimplify(norm2(couple(ug, theta(ug, j, eps=1), j, j, K)) / nu ** 2) for K in range(0, n + 1)]
    assert sp.nsimplify(sum(rowg)) == 1
    res[f"level_{n}_j={j}"] = {"||rho_K(v_j)||^2 K=0..2j": [str(x) for x in row],
                               "generic_state_rhat_K K=0..2j": [str(x) for x in rowg],
                               "ranks_nonzero_at_v_j": [K for K, x in enumerate(row) if x != 0],
                               "ranks_nonzero_generic": [K for K, x in enumerate(rowg) if x != 0]}
(HERE / "audit_res_item16.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
