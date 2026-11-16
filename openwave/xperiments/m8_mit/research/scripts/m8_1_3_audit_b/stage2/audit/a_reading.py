"""Reading check for C.3 at R2's point of the line {5/2, -3/2}: the worklist says the phase and psi -> conj(psi) C
'generate a group Sp(1)'; read literally, the group generated is U(1) u U(1)J (one-dimensional).  Exact (sympy)
comparison of k, r_u, signature and dim K' under the two readings, with my exact quartic.  Also: h is exactly
invariant under Theta (so Theta u, i Theta u lie in ker Mhat whichever reading is taken)."""
import sys, os
sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from fractions import Fraction as Fr
import sympy as sp
import exact_pt
from exact_pt import ExactPoint, J0, rotation_generators, theta_real
from core import mlist
from hpoly import h_poly
from chk import check, save_json

ms = mlist(7)
x0 = [0] * 16
x0[ms.index(Fr(5, 2))] = 1; x0[ms.index(Fr(-3, 2))] = 1     # p_{5/2} = p_{-3/2}: the line point up to scale
out = {}
orig = exact_pt.group_generators
for reading in ('Sp(1) (as stated)', 'literal: U(1) and J only'):
    if reading.startswith('literal'):
        exact_pt.group_generators = lambda name: [J0(7)] + rotation_generators(7)
    else:
        exact_pt.group_generators = orig
    E = ExactPoint('R2', x0)
    sig = E.signature()
    kp = len(E.kernel_in_N())
    print(f'{reading}: critical {E.critical}, h(u) = {E.hu}, k = {E.k}, r_u = {E.r_u}, signature on N_u {sig}, dim K\' = {kp}')
    out[reading] = {'k': E.k, 'r_u': E.r_u, 'signature': sig, "dim K'": kp}
check('reading: the two readings give different k (the reading matters for C.3)', out['Sp(1) (as stated)']['k'] != out['literal: U(1) and J only']['k'])
# invariance of h under Theta, exactly: h(Theta x) == h(x) as polynomials
xs = sp.symbols('x0:16', real=True)
T = theta_real(7)
pol = h_poly('R2')
def ev(xv):
    tot = 0
    for e, c in pol.items():
        t = sp.Rational(c.numerator, c.denominator)
        for i, k in enumerate(e):
            if k:
                t *= xv[i] ** k
        tot += t
    return sp.expand(tot)
xt = list(T * sp.Matrix(xs))
check('h(Theta x) == h(x) identically (exact polynomial identity)', sp.expand(ev(xt) - ev(list(xs))) == 0)
save_json('a_reading.json', out)
