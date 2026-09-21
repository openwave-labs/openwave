import sys; sys.path.insert(0, __import__('pathlib').Path(__file__).resolve().parent.as_posix())
import mpmath as mp
from sympy import S, sqrt, I, Rational, nsimplify, Poly, symbols, factor
from cg import CGT, MS

mp.mp.dps = 50
IDX = {m: 3-m for m in MS}

def to_vec14(c):
    v = mp.matrix(14,1)
    for m in MS:
        z = mp.mpmathify(complex(mp.mpf(str(float(mp.re(c[m])))), 0)) if False else None
    return v

def cplx(x):
    from sympy import re as sre, im as sim, N
    return mp.mpc(mp.mpf(str(N(sre(x), 40))), mp.mpf(str(N(sim(x), 40))))

CGm = {k: mp.mpf(str(float(v))) if v.is_rational else mp.mpmathify(str(v.evalf(40))) for k, v in CGT.items()}

def rho_num(c):   # c: dict m -> mpc
    tu = {m: ((-1)**(3-m)) * mp.conj(c[-m]) for m in MS}
    out = {}
    for Q in range(-6,7):
        s = mp.mpc(0)
        for m1 in MS:
            m2 = Q-m1
            if abs(m2) <= 3:
                s += CGm[(m1,m2)]*c[m1]*tu[m2]
        out[Q] = s
    return out

def N_num(c):
    r = rho_num(c)
    return mp.fsum([mp.re(r[Q]*mp.conj(r[Q])) for Q in range(-6,7)])

def v14_to_c(v):
    return {m: mp.mpc(v[2*IDX[m]], v[2*IDX[m]+1]) for m in MS}

def c_to_v14(c):
    v = mp.matrix(14,1)
    for m in MS:
        v[2*IDX[m]] = mp.re(c[m]); v[2*IDX[m]+1] = mp.im(c[m])
    return v

def Nv(v):
    return N_num(v14_to_c(v))

def quad_form(u, e):
    """HessN(u)[e,e], exact for a quartic form."""
    f = lambda t: Nv(u + t*e)
    f0 = Nv(u)
    A = (f(mp.mpf(1)) + f(mp.mpf(-1)) - 2*f0)/2
    B = (f(mp.mpf(2)) + f(mp.mpf(-2)) - 2*f0)/8
    return 2*(4*A - B)/3   # HessN[e,e] = 2*a2

def hessian(u):
    H = mp.matrix(14,14)
    E = []
    for i in range(14):
        e = mp.matrix(14,1); e[i] = 1; E.append(e)
    q = [quad_form(u, E[i]) for i in range(14)]
    for i in range(14):
        H[i,i] = q[i]
    for i in range(14):
        for j in range(i+1,14):
            qij = quad_form(u, E[i]+E[j])
            H[i,j] = H[j,i] = (qij - q[i] - q[j])/2
    return H

# J operators on spin 3
def Jz(c):  return {m: m*c[m] for m in MS}
def Jp(c):
    out = {m: mp.mpc(0) for m in MS}
    for m in MS:
        if m+1 in IDX:
            out[m+1] += mp.sqrt(mp.mpf(12 - m*(m+1)))*c[m]
    return out
def Jm(c):
    out = {m: mp.mpc(0) for m in MS}
    for m in MS:
        if m-1 in IDX:
            out[m-1] += mp.sqrt(mp.mpf(12 - m*(m-1)))*c[m]
    return out
def Jx(c):
    a, b = Jp(c), Jm(c); return {m: (a[m]+b[m])/2 for m in MS}
def Jy(c):
    a, b = Jp(c), Jm(c); return {m: (a[m]-b[m])/(2j) for m in MS}

def orbit_dirs(u):
    c = v14_to_c(u)
    out = []
    out.append(c_to_v14({m: 1j*c[m] for m in MS}))
    for Jop in (Jx, Jy, Jz):
        d = Jop(c)
        out.append(c_to_v14({m: -1j*d[m] for m in MS}))
    return out

def gram_schmidt(vs, tol=mp.mpf('1e-25')):
    basis = []
    for v in vs:
        w = v.copy()
        for b in basis:
            w -= (b.T*w)[0]*b
        n = mp.norm(w)
        if n > tol:
            basis.append(w/n)
    return basis

def analyse(c_sym, name, dps=50):
    c = {m: cplx(c_sym[m]) for m in MS}
    nrm = mp.sqrt(mp.fsum([mp.re(c[m]*mp.conj(c[m])) for m in MS]))
    c = {m: c[m]/nrm for m in MS}
    u = c_to_v14(c)
    Nu = Nv(u)
    H = hessian(u)
    M = H - 4*Nu*mp.eye(14)
    # span of u and orbit directions
    killed = gram_schmidt([u] + orbit_dirs(u))
    dimO = len(killed) - 1
    # residual: does M annihilate O_u?
    resid = mp.mpf(0)
    for d in orbit_dirs(u):
        nd = mp.norm(d)
        if nd > mp.mpf('1e-25'):
            resid = max(resid, mp.norm(M*(d/nd)))
    # build N_u
    cand = []
    for i in range(14):
        e = mp.matrix(14,1); e[i] = 1; cand.append(e)
    Nb = gram_schmidt(killed + cand)[len(killed):]
    k = len(Nb)
    R = mp.matrix(k,k)
    for i in range(k):
        for j in range(k):
            R[i,j] = (Nb[i].T*(M*Nb[j]))[0]
    for i in range(k):
        for j in range(i+1,k):
            a = (R[i,j]+R[j,i])/2; R[i,j]=R[j,i]=a
    ev = mp.eigsy(R, eigvals_only=True)
    tol = mp.mpf('1e-20')
    nneg = sum(1 for x in ev if x < -tol)
    nzer = sum(1 for x in ev if abs(x) <= tol)
    npos = sum(1 for x in ev if x > tol)
    return dict(name=name, val924=924*Nu, dimN=k, dimO=dimO, sig=(nneg,nzer,npos),
                ev=sorted([mp.mpf(x) for x in ev]), resid=resid)
