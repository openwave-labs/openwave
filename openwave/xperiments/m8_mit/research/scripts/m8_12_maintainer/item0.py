"""Worklist item 0 is a warm-up the frozen table does not cover, so compute it here
independently: the maintainer needs its answers to score that part of a return."""
import sys, pathlib
sys.path.insert(0, pathlib.Path(__file__).resolve().parent.as_posix())
from sympy import S, I, Rational, simplify, sqrt, conjugate, nsimplify
from sympy.physics.quantum.cg import CG
from cg import MS, N_of, norm2, theta

print("<3 3; 3 -3 | 6 0> =", CG(S(3), S(3), S(3), S(-3), S(6), S(0)).doit())

# Theta squared, symbolically, on a general vector
from sympy import symbols
cs = symbols('c3 c2 c1 c0 cm1 cm2 cm3')
c = {m: cs[3-m] for m in MS}
tt = theta(theta(c))
print("Theta(Theta u) - u =", {m: simplify(tt[m] - c[m]) for m in MS})

for label, vec in [("2*v3 + v1 - 3*v-2", {3: S(2), 2: S(0), 1: S(1), 0: S(0), -1: S(0), -2: S(-3), -3: S(0)}),
                   ("v3 + I*v0 + 2*v-1", {3: S(1), 2: S(0), 1: S(0), 0: I, -1: S(2), -2: S(0), -3: S(0)})]:
    r = simplify(N_of(vec) / norm2(vec)**2)
    print(f"r6 at {label} = {r}   924*r6 = {simplify(924*r)}")

# convention check: the worklist fixes <3 3; 3 3 | 6 6> = +1
print("<3 3; 3 3 | 6 6> =", CG(S(3), S(3), S(3), S(3), S(6), S(6)).doit())
# second route for the two item-0 values: 50-digit numerics through the mpmath machinery
import mpmath as mp
from hess import Nv, c_to_v14, cplx
mp.mp.dps = 50
for label, vec, want in [("2*v3 + v1 - 3*v-2", {3: S(2), 2: S(0), 1: S(1), 0: S(0), -1: S(0), -2: S(-3), -3: S(0)}, Rational(5931,28)),
                         ("v3 + I*v0 + 2*v-1", {3: S(1), 2: S(0), 1: S(0), 0: I, -1: S(2), -2: S(0), -3: S(0)}, Rational(6329,36))]:
    cn = {m: cplx(vec[m]) for m in MS}
    nrm = mp.sqrt(mp.fsum([mp.re(cn[m]*mp.conj(cn[m])) for m in MS]))
    cn = {m: cn[m]/nrm for m in MS}
    got = 924*Nv(c_to_v14(cn))
    tgt = mp.mpf(str(want.evalf(40)))
    print(f"numeric 924*r6 at {label} = {mp.nstr(got, 25)}  agrees: {mp.fabs(got-tgt) < mp.mpf('1e-30')}")
