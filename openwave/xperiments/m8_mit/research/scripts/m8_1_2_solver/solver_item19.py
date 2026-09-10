"""Item 19: exact computational support for the characterisation of C(u) = 0.

Checks (all exact, symbolic in u and conj(u) as independent variables):
 1. C(u) := [[u (x) u]_6 (x) Theta u]_8 equals kappa * iota_8^{-1} J(F_u^2, G_u), where
    iota_j(v) = sum_m v_m x^{j+m} y^{j-m}/sqrt((j+m)!(j-m)!) (the equivariant identification of the
    orthonormal weight basis with binary forms used for the representation matrices),
    F_u = iota_3(u), G_u = iota_3(Theta u), J(f,g) = f_x g_y - f_y g_x.
 2. The two couplings [[u (x) Theta u]_L (x) u]_8, L = 5, 6 (which span the V_8-isotypic projection
    of u (x) Theta u (x) u in V_3^{(x)3}) are both constant multiples of C(u).
 3. C vanishes identically on Fix(Theta) and is nonzero at v_3.
"""
import sympy as sp

from solver_lib import CG, save_results, s

MS = range(-3, 4)
a = list(sp.symbols("u0:7"))
b = list(sp.symbols("ub0:7"))
th = [sp.Integer(-1) ** abs(m) * b[-m + 3] for m in MS]      # (Theta u)_m = (-1)^m conj(u_{-m})


def couple(A, ja, B, jb, J):
    o = [sp.S(0)] * (2 * J + 1)
    for m1 in range(-ja, ja + 1):
        if A[m1 + ja] == 0:
            continue
        for m2 in range(-jb, jb + 1):
            M = m1 + m2
            if abs(M) > J or B[m2 + jb] == 0:
                continue
            c = CG(ja, m1, jb, m2, J, M)
            if c:
                o[M + J] += c * A[m1 + ja] * B[m2 + jb]
    return [sp.expand(x) for x in o]


B6 = couple(a, 3, a, 3, 6)
C = couple(B6, 6, th, 3, 8)

x, y = sp.symbols("x y")


def iota(v, j):
    return sum(v[m + j] * x ** (j + m) * y ** (j - m) / sp.sqrt(sp.factorial(j + m) * sp.factorial(j - m))
               for m in range(-j, j + 1))


Fu = iota(a, 3)
Gu = iota(th, 3)
H = sp.expand(sp.diff(Fu ** 2, x) * sp.diff(Gu, y) - sp.diff(Fu ** 2, y) * sp.diff(Gu, x))
PH = sp.Poly(H, x, y)
h = [sp.expand(PH.coeff_monomial(x ** (8 + M) * y ** (8 - M)) * sp.sqrt(sp.factorial(8 + M) * sp.factorial(8 - M)))
     for M in range(-8, 9)]
# kappa from the top component
kappa = sp.cancel(h[16] / C[16])
ok1 = all(sp.expand(h[i] - kappa * C[i]) == 0 for i in range(17))
print("check 1: iota_8^{-1} J(F^2, G) == kappa * C(u) for all 17 components:", ok1, " kappa =", sp.radsimp(kappa))

# check 2: the other couplings
res2 = {}
for L in (5, 6):
    rL = couple(a, 3, th, 3, L)
    XL = couple(rL, L, a, 3, 8)
    kL = sp.cancel(XL[16] / C[16])
    okL = all(sp.expand(XL[i] - kL * C[i]) == 0 for i in range(17))
    res2[L] = (okL, sp.radsimp(kL))
    print(f"check 2: [[u (x) Theta u]_{L} (x) u]_8 == k_{L} * C(u):", okL, " k =", sp.radsimp(kL))
norm_iso_factor = sp.radsimp(sum(k ** 2 for _, k in res2.values()))
print("         ||Pi_8(u (x) Theta u (x) u)||^2 = (k5^2 + k6^2) ||C(u)||^2 with k5^2 + k6^2 =", norm_iso_factor)

# check 3: vanishing on Fix(Theta): u_{-m} = (-1)^m conj(u_m)
p = sp.symbols("p0:4", real=True)
q = sp.symbols("q0:4", real=True)
ufix = {}
for m in range(0, 4):
    if m == 0:
        ufix[0] = p[0]      # (-1)^0 conj(u_0) = u_0 -> real
    else:
        ufix[m] = p[m] + sp.I * q[m]
        ufix[-m] = sp.Integer(-1) ** m * (p[m] - sp.I * q[m])
subs_fix = {a[m + 3]: ufix[m] for m in MS}
subs_fix.update({b[m + 3]: sp.conjugate(ufix[m]) for m in MS})
# Theta u == u on this family:
thu = [sp.expand(e.subs(subs_fix)) for e in th]
assert all(sp.expand(thu[i] - ufix[i - 3]) == 0 for i in range(7))
Cfix = [sp.expand(e.subs(subs_fix)) for e in C]
ok3 = all(e == 0 for e in Cfix)
print("check 3: C(u) == 0 identically on Fix(Theta) (7 real parameters):", ok3)
v3 = {a[i]: (1 if i == 6 else 0) for i in range(7)}
v3.update({b[i]: (1 if i == 6 else 0) for i in range(7)})
Cv3 = [sp.radsimp(e.subs(v3)) for e in C]
print("         C(v_3) =", [e for e in Cv3 if e != 0], " nonzero:", any(e != 0 for e in Cv3))
# bidegree: C(lambda u) = lambda^2 conj(lambda) C(u)
lam, lamb = sp.symbols("lam lamb")
sc = {**{a[i]: lam * a[i] for i in range(7)}, **{b[i]: lamb * b[i] for i in range(7)}}
ok4 = all(sp.expand(e.subs(sc, simultaneous=True) - lam ** 2 * lamb * e) == 0 for e in C)
print("check 4: C(lambda u) = lambda^2 conj(lambda) C(u):", ok4)
# the constellation-free restatement: J(F, G) itself
JFG = sp.expand(sp.diff(Fu, x) * sp.diff(Gu, y) - sp.diff(Fu, y) * sp.diff(Gu, x))
ok5 = sp.expand(H - 2 * Fu * JFG) == 0
print("check 5: J(F^2, G) = 2 F J(F, G):", ok5)
save_results("item19", {
    "check1_C_equals_kappa_times_Jacobian": ok1, "kappa": s(sp.radsimp(kappa)),
    "check2_other_couplings_proportional": {str(L): {"proportional": v[0], "k": s(v[1])} for L, v in res2.items()},
    "isotypic_norm_factor_k5sq_plus_k6sq": s(norm_iso_factor),
    "check3_vanishes_on_Fix_Theta": ok3, "C_v3_nonzero_components": [s(e) for e in Cv3 if e != 0],
    "check4_bidegree_2_1": ok4, "check5_Leibniz": ok5})
