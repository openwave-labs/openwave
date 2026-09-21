import numpy as np
from fractions import Fraction as F
from scipy.optimize import linprog
T = {0: ['1/7'] * 4, 1: ['-3/7', '-9/28', '-1/14', '9/28'], 2: ['5/7', '19/84', '-5/14', '25/84'],
     3: ['-1', '1/6', '1/6', '1/6'], 4: ['9/7', '-9/14', '97/154', '9/154'],
     5: ['-11/7', '55/84', '17/42', '1/84'], 6: ['13/7', '65/84', '13/154', '1/924']}
T = {L: [float(F(x)) for x in v] for L, v in T.items()}
nv = 1 + 4 + 7
A = np.zeros((4, nv)); b = np.zeros(4)
for K in range(4):
    A[K, 0] = 1; A[K, 1 + K] = -1
    for L in range(7): A[K, 5 + L] = -T[L][K]
    b[K] = T[6][K]
c = np.zeros(nv); c[0] = 1
bounds = [(None, None)] + [(0, None)] * 11
r = linprog(c, A_eq=A, b_eq=b, bounds=bounds)
print(r.status, r.fun, 463 / 924, r.x)
