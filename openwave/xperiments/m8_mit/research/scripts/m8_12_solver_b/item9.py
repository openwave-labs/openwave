"""Item 9: exact reasons for computed zeros; symmetry hypotheses verified.

1. N is invariant under phase and rotations (infinitesimally: grad N . X u = 0 identically).
2. Coefficient-wise complex conjugation K (y -> -y) preserves N.
3. Symmetries fixing the item-5b points and reversing exactly one chart tangent (off-diagonal zeros).
4. Theta u = +-u at v0, oct, hex and the CG exchange symmetry (odd-rank multipoles vanish).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
import core, hess, rot
from core import report, IDX, MS
from hess import simp

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
N = core.N_poly()
RV = core.RV
xv = sp.Matrix(RV)
gradN = sp.Matrix(core.grad_N_sym())
Jz, Jp, Jm, Jx, Jy = core.Jmats()
for name, X in (("i", sp.I * sp.eye(7)), ("-i Jx", -sp.I * Jx), ("-i Jy", -sp.I * Jy), ("-i Jz", -sp.I * Jz)):
    Xr = core.realify(X)
    ok &= report("N invariant under the one-parameter group of %s: grad N . (X x) = 0 identically" % name,
                 sp.expand((gradN.T * Xr * xv)[0, 0]) == 0)
Ksub = {RV[7 + k]: -RV[7 + k] for k in range(7)}
ok &= report("N(conj u) = N(u) identically (complex conjugation K preserves r6)", sp.expand(N.xreplace(Ksub) - N) == 0)
# a control: a non-symmetry (swap of v3 and v2 coefficients) does NOT preserve N
Psub = {RV[0]: RV[1], RV[1]: RV[0], RV[7]: RV[8], RV[8]: RV[7]}
ok &= report("control: swapping the v3 and v2 coefficients does not preserve N (check can fail)",
             sp.expand(N.xreplace(Psub, ) - N) != 0)

# 3. off-diagonal zeros of item 5b
x, y = sp.symbols("x y", real=True)
R = sp.Rational


def vec(d):
    c = [sp.Integer(0)] * 7
    for m, val in d.items():
        c[IDX[m]] += sp.sympify(val)
    return c


Kmat = None
cases = [("Wmin in W", vec({0: 1}), vec({3: 1, -3: 1}), sp.sqrt(R(10, 23)), "K", sp.eye(7)),
         ("Umin in U", vec({0: 1}), vec({2: 1, -2: 1}), sp.I * sp.sqrt(R(5, 6)), "K R_z(pi/2)", rot.Dz(sp.pi / 2)),
         ("hex in U", vec({0: 1}), vec({2: 1, -2: 1}), sp.sqrt(R(3, 10)), "K", sp.eye(7)),
         ("oct in W", vec({0: 1}), vec({3: 1, -3: 1}), sp.I * sp.sqrt(R(2, 5)), "K R_z(pi/3)", rot.Dz(sp.pi / 3))]
for label, w0, w1, z0, sname, D in cases:
    zc = x + sp.I * y
    uz = sp.Matrix([a + zc * b for a, b in zip(w0, w1)])
    S = lambda v: (D * v).applyfunc(lambda q: sp.conjugate(q))        # S = K o D (antiunitary, isometry of Re<,>)
    sub = {x: sp.re(z0), y: sp.im(z0)}
    u0 = uz.subs(sub)
    ok &= report("%s: %s fixes u exactly" % (label, sname), (S(u0) - u0).applyfunc(lambda q: sp.simplify(sp.expand_complex(q))) == sp.zeros(7, 1))
    tx = uz.diff(x).subs(sub); ty = uz.diff(y).subs(sub)
    sx = (S(tx) - tx).applyfunc(lambda q: sp.simplify(sp.expand_complex(q))) == sp.zeros(7, 1)
    sx_m = (S(tx) + tx).applyfunc(lambda q: sp.simplify(sp.expand_complex(q))) == sp.zeros(7, 1)
    sy = (S(ty) - ty).applyfunc(lambda q: sp.simplify(sp.expand_complex(q))) == sp.zeros(7, 1)
    sy_m = (S(ty) + ty).applyfunc(lambda q: sp.simplify(sp.expand_complex(q))) == sp.zeros(7, 1)
    ok &= report("%s: %s maps d/dx u to +-itself and d/dy u to -+itself (opposite signs)" % (label, sname),
                 (sx and sy_m) or (sx_m and sy))
    # the normalisation u(z)/|u(z)| is S-compatible because S is an isometry fixing u, so the same
    # sign pattern holds for d/dx u_hat and d/dy u_hat; hence H(tx, ty) = -H(tx, ty) = 0 and Gram(tx, ty) = 0

# 4. odd-rank multipoles
for name, c in (("v0", vec({0: 1})), ("oct", vec({2: 1, -2: -1})), ("hex", vec({3: 1, -3: 1}))):
    tc = core.theta(c)
    sgn = [s_ for s_ in (1, -1) if all(sp.simplify(a - s_ * b) == 0 for a, b in zip(tc, c))]
    ok &= report("%s: Theta u = %s u exactly" % (name, sgn[0] if sgn else "?"), len(sgn) == 1)
bad = 0
for k in range(0, 7):
    for m1 in MS:
        for m2 in MS:
            if abs(m1 + m2) <= k and sp.simplify(core.cg(3, m1, 3, m2, k, m1 + m2) - (-1) ** (6 - k) * core.cg(3, m2, 3, m1, k, m1 + m2)) != 0:
                bad += 1
ok &= report("exchange symmetry <3 m1;3 m2|k Q> = (-1)^(6-k) <3 m2;3 m1|k Q> for all k, m1, m2", bad == 0)

# 5. r2 = 0 at v2 and at oct
# v2: u (x) Theta u = -v2 (x) v-2 has a single rank-2 component, with coefficient <3 2; 3 -2 | 2 0>
ok &= report("<3 2; 3 -2 | 2 0> = 0 exactly (the reason r2(v2) = 0)", core.cg(3, 2, 3, -2, 2, 0) == 0)
# oct: D(h) oct = chi(h) oct for h in O, so oct (x) Theta oct is O-invariant; its rank-2 part is an
# O-invariant vector of V_2, and dim V_2^O = (1/24) sum_h chi_2(theta_h) = 0 (exact class sum)
chi2 = lambda th: sp.simplify(sum(sp.cos(m * th) for m in range(-2, 3)))
classes_O = [(1, 0), (8, 2 * sp.pi / 3), (3, sp.pi), (6, sp.pi / 2), (6, sp.pi)]
dimV2O = sp.simplify(sum(n_ * chi2(th) for n_, th in classes_O) / 24)
ok &= report("dim V_2^O = 0 by the exact character formula (the reason r2(oct) = 0)", dimV2O == 0, "dim = %s" % dimV2O)
# a control on the same formula: V_4 does contain an O-invariant (the cubic harmonic), so this must be 1
chi4 = lambda th: sp.simplify(sum(sp.cos(m * th) for m in range(-4, 5)))
ok &= report("control: dim V_4^O = 1 by the same formula", sp.simplify(sum(n_ * chi4(th) for n_, th in classes_O) / 24) == 1)
print("ITEM9", "ALLPASS" if ok else "SOMEFAIL")
