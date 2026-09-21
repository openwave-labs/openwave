"""Audit item 11 (control): u = (v3 + v1 - v-2)/sqrt3."""
import sympy as sp, mpmath as mp
from aud_core import *

w = [sp.Integer(0)] * 7
w[idx(3)] = 1; w[idx(1)] = 1; w[idx(-2)] = -1
s3 = sp.sqrt(3)
u = [z / s3 for z in w]
Nu = sp.expand(N_exact(u))
print('924 rhat6 =', sp.expand(924 * Nu / nrm2(u) ** 2))
g = grad_exact(u)
x = to_real(u)
tg = [sp.expand(gi - 4 * Nu * xi) for gi, xi in zip(g, x)]
gn2 = sp.expand(sum(t ** 2 for t in tg))
print('|tangential grad|^2 =', gn2, '  |grad| =', sp.sqrt(gn2), '=', sp.N(sp.sqrt(gn2), 30))
print('  radial part u.grad =', sp.expand(sum(a * b for a, b in zip(tg, x))))
H = hessN_exact(u)
M = (H - 4 * Nu * sp.eye(14)).applyfunc(sp.expand)
names = ['i u', '-i Jx u', '-i Jy u', '-i Jz u']
for nm, d in zip(names, O_gens(u)):
    dv = sp.Matrix(to_real(d))
    r = (M * dv).applyfunc(sp.expand)
    n2 = sp.radsimp(sp.expand(sum(t ** 2 for t in r)))
    mpoly = sp.minimal_polynomial(sp.sqrt(n2), sp.Symbol('X'))
    print('|M_u d| for d = %-8s : sqrt(%s) = %s ; minpoly %s' % (nm, n2, sp.N(sp.sqrt(n2), 25), mpoly))
# identity M_u X u = X g_u (a consequence of invariance): check exactly for X = i (g_u rotated by i)
gi = from_real(tg); igx = to_real([I * z for z in gi])
r = (M * sp.Matrix(to_real(O_gens(u)[0]))).applyfunc(sp.expand)
print('M_u(iu) - i*g_u == 0 :', all(sp.expand(a - b) == 0 for a, b in zip(r, igx)))

# high-precision independent route: finite-difference-free mpmath gradient via mp.diff of rhat on the sphere
mp.mp.dps = 50
cg = mp_cg()
uc = [mp.mpc(sp.N(sp.re(z), 60), sp.N(sp.im(z), 60)) for z in u]
def f_real(*xs):
    c = [mp.mpc(xs[k], xs[7 + k]) for k in range(7)]
    return mp_rhat(c, cg)
xr = [mp.re(z) for z in uc] + [mp.im(z) for z in uc]
grad = []
for k in range(14):
    def fk(t, k=k):
        xx = list(xr); xx[k] += t; return f_real(*xx)
    grad.append(mp.diff(fk, 0))
gnorm = mp.sqrt(sum(gg ** 2 for gg in grad))
print('mpmath 50-digit |grad rhat6| =', mp.nstr(gnorm, 25), ' diff vs exact:', mp.nstr(abs(gnorm - mp.sqrt(mp.mpf(sp.N(gn2, 70)))), 3))
