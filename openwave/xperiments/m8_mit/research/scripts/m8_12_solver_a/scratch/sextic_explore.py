import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sympy as sp, common as C, classes as K
t0 = time.time()
A = list(K.vec({1: 1, -2: sp.Rational(1, 2)}))
D = list(K.vec({2: 1, -3: sp.sqrt(sp.Rational(12, 13))}))
triples = [(2, 2, 2), (2, 2, 4), (2, 4, 4), (4, 4, 4), (1, 1, 2), (1, 2, 3), (3, 3, 2), (3, 3, 4), (6, 6, 2), (6, 6, 4), (5, 5, 2), (1, 5, 6)]
PA = C.multipole_parts(A)
print('tr rho_L^2 check A:', [sp.nsimplify((PA[L] * PA[L]).trace()) for L in range(7)], time.time() - t0)
print('A', C.sextic_invariants(A, triples))
print('D', C.sextic_invariants(D, triples), time.time() - t0)
