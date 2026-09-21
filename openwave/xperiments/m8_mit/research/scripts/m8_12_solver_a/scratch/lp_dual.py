import numpy as np
from fractions import Fraction as F
from scipy.optimize import linprog
T = {0: ['1/7'] * 4, 1: ['-3/7', '-9/28', '-1/14', '9/28'], 2: ['5/7', '19/84', '-5/14', '25/84'],
     3: ['-1', '1/6', '1/6', '1/6'], 4: ['9/7', '-9/14', '97/154', '9/154'],
     5: ['-11/7', '55/84', '17/42', '1/84'], 6: ['13/7', '65/84', '13/154', '1/924']}
Tf = {L: [float(F(x)) for x in v] for L, v in T.items()}
# dual: maximize T6.p s.t. p >= 0, sum p = 1, T_L.p >= 0
A_ub = -np.array([Tf[L] for L in range(7)]); b_ub = np.zeros(7)
r = linprog(-np.array(Tf[6]), A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, 4)), b_eq=[1], bounds=[(0, None)] * 4)
print(r.fun, r.x, [F(v).limit_denominator(1000) for v in r.x])
