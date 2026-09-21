from sympy import S, Rational, sqrt, I, conjugate, expand, simplify, nsimplify
from sympy.physics.quantum.cg import CG

MS = [3,2,1,0,-1,-2,-3]

def cg_table():
    t = {}
    for m1 in MS:
        for m2 in MS:
            Q = m1 + m2
            if abs(Q) <= 6:
                t[(m1, m2)] = CG(S(3), S(m1), S(3), S(m2), S(6), S(Q)).doit()
    return t

CGT = cg_table()

def theta(c):
    # (Theta u)_m = (-1)^(3-m) conj(c_{-m})
    return {m: S(-1)**(3-m) * conjugate(c[-m]) for m in MS}

def rho6(c):
    tu = theta(c)
    out = {}
    for Q in range(-6, 7):
        s = S(0)
        for m1 in MS:
            m2 = Q - m1
            if abs(m2) <= 3:
                s += CGT[(m1, m2)] * c[m1] * tu[m2]
        out[Q] = expand(s)
    return out

def N_of(c):
    r = rho6(c)
    tot = S(0)
    for Q in range(-6, 7):
        tot += expand(r[Q] * conjugate(r[Q]))
    return simplify(expand(tot))

def norm2(c):
    return simplify(sum(expand(c[m]*conjugate(c[m])) for m in MS))

def vec(**kw):
    c = {m: S(0) for m in MS}
    for k, v in kw.items():
        m = int(k[1:]) if not k.startswith('vm') else -int(k[2:])
        c[m] = v
    return c

def r6(c):
    return simplify(N_of(c) / norm2(c)**2)
