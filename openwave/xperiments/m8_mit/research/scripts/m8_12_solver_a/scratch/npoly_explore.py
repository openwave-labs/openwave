import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sympy as sp, common as C
N = C.N_poly()
P = sp.Poly(N, *C.XS)
coeffs = set(P.coeffs())
irr = [c for c in coeffs if not c.is_Rational]
print(len(coeffs), 'irrational coeffs:', irr[:10])
