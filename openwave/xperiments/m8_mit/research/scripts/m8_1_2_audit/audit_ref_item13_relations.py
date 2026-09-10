"""Item 13 refutation check: all linear relations sum_K c_K ||rho_K(u)||^2 = 0 (K = 0..6).

Numeric kernel at 60 digits, rational identification, then EXACT verification of every relation
as a polynomial identity in (u, conj u). Consequence: w_K in Q_d = sum_K w_K rhat_K is determined
only modulo this kernel.
"""
import json
import pathlib
import sympy as sp
import mpmath as mp
from audit_su2lib import rho

HERE = pathlib.Path(__file__).parent
us = list(sp.symbols("u0:7"))
ub = [sp.conjugate(x) for x in us]
gens = us + ub
polys = [sp.expand(sum(x * sp.conjugate(x) for x in rho(us, K))) for K in range(7)]
mons = sorted(set().union(*[set(sp.Poly(p, *gens).monoms()) for p in polys]))
mp.mp.dps = 80
A = mp.matrix(len(mons), 7)
for K, p in enumerate(polys):
    P = sp.Poly(p, *gens)
    dct = dict(zip(P.monoms(), P.coeffs()))
    for i, mo in enumerate(mons):
        A[i, K] = mp.mpf(str(sp.N(dct.get(mo, 0), 90)))
U, Sv, V = mp.svd_r(A)
sv = [Sv[i] for i in range(7)]
null = [[V[i, k] for k in range(7)] for i in range(7) if Sv[i] < mp.mpf(10) ** -50]
res = {"singular_values": [mp.nstr(x, 8) for x in sv], "kernel_dim": len(null)}
# reduced basis of the kernel: pivot on K = 0, 1, 2 ... (solve for coefficients with c_{K} = 1 on chosen free slots)
Nm = mp.matrix(null)            # rows = kernel vectors
kd = len(null)
# choose free columns 4, 5, 6 (unit vectors) and express c_0..c_3 through them
rels = []
for f in range(7 - kd, 7):
    # find combination of kernel rows with c_f = 1 and c_g = 0 for the other free g
    free = list(range(7 - kd, 7))
    B = mp.matrix(kd, kd)
    for i in range(kd):
        for jj, g in enumerate(free):
            B[jj, i] = Nm[i, g]
    rhs = mp.matrix([1 if g == f else 0 for g in free])
    coef = mp.lu_solve(B, rhs)
    vec_ = [sum(coef[i] * Nm[i, k] for i in range(kd)) for k in range(7)]
    rels.append([sp.nsimplify(mp.nstr(x, 40), rational=True, tolerance=1e-30) for x in vec_])
exact = []
for r in rels:
    e = sp.expand(sum(c * p for c, p in zip(r, polys)))
    P = sp.Poly(e, *gens) if e != 0 else None
    ok = e == 0 or all(sp.simplify(sp.radsimp(c)) == 0 for c in P.coeffs())
    exact.append(ok)
res["relations_c_K_(K=0..6)"] = [[str(c) for c in r] for r in rels]
res["relations_verified_exactly"] = exact
assert all(exact)
# the solver's stated relation (constant shift): sum_K rhat_K = 1 = 7 rhat_0, i.e. c = (-6, 1, 1, 1, 1, 1, 1)
shift = [-6, 1, 1, 1, 1, 1, 1]
res["solver_shift_relation_in_kernel"] = sp.expand(sum(c * p for c, p in zip(shift, polys))) == 0
# is the kernel bigger than the solver's one relation?  (it is if kernel_dim > 1)
res["kernel_larger_than_solver_stated"] = len(null) > 1
# a concrete witness: a different w with the same Q_d in the d=4 sector, having w_6 = 0
w4 = [7, 0, 0, 0, 0, 0, sp.Rational(21, 52)]
# find kernel element with c_6 = -21/52 among the reduced relations (the one with free slot 6)
r6 = rels[-1]
alt = [sp.nsimplify(w4[k] - sp.Rational(21, 52) * r6[k]) for k in range(7)]
res["witness_alternative_w_d4_same_Q"] = [str(x) for x in alt]
res["witness_same_Q_identity_exact"] = sp.expand(sum((a - b) * p for a, b, p in zip(alt, w4, polys))) == 0 or all(
    sp.simplify(sp.radsimp(c)) == 0 for c in sp.Poly(sp.expand(sum((a - b) * p for a, b, p in zip(alt, w4, polys))), *gens).coeffs())
(HERE / "audit_ref_item13_relations.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
