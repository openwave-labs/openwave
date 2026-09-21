"""Worklist item 5b: the in-locus second variations at the two interior orbits.

⚠️ The tangent convention is the whole game here. A locus tangent generally has a
component along O_u, and the form must be read on its projection onto N_u,
normalized THERE. Normalizing in the ambient tangent instead rescales the answer by
the squared transverse fraction, which is how a correct value reads as wrong.
"""
import sys, pathlib
sys.path.insert(0, pathlib.Path(__file__).resolve().parent.as_posix())
import mpmath as mp
from sympy import nsimplify, Rational
from cg import MS
from hess import c_to_v14, Nv, hessian, orbit_dirs, gram_schmidt

mp.mp.dps = 50

def unitv(vec):
    c = {m: vec.get(m, mp.mpc(0)) for m in MS}
    n = mp.sqrt(mp.fsum([mp.re(c[m]*mp.conj(c[m])) for m in MS]))
    return c_to_v14({m: c[m]/n for m in MS})

def tangents(param, t0, h=mp.mpf('1e-12')):
    out = []
    for i in range(len(t0)):
        tp = list(t0); tm = list(t0)
        tp[i] += h; tm[i] -= h
        out.append((unitv(param(*tp)) - unitv(param(*tm)))/(2*h))
    return out

def ident(v):
    r = nsimplify(mp.nstr(v, 25), rational=True, tolerance=mp.mpf('1e-20'))
    return r if abs(float(r) - float(v)) < 1e-18 else "(not a small rational)"

def report(name, u, tans, tnames):
    Nu = Nv(u); M = hessian(u) - 4*Nu*mp.eye(14)
    O = gram_schmidt(orbit_dirs(u))
    print(f"\n=== {name}:  924*N(u) = {mp.nstr(924*Nu, 20)},  dim O_u = {len(O)} ===")
    kept, names = [], []
    for d, nm in zip(tans, tnames):
        d = d - (mp.matrix(u).T*d)[0]*mp.matrix(u)          # drop the radial part
        if mp.norm(d) < mp.mpf('1e-20'):
            print(f"  {nm}: pure radial, no projective direction"); continue
        proj = d.copy()
        for b in O:
            proj -= (b.T*d)[0]*b                              # then drop the ORBIT part
        frac = mp.norm(proj)/mp.norm(d)
        if frac < mp.mpf('1e-15'):
            val = (d.T*(M*d))[0]/mp.norm(d)**2
            print(f"  {nm}: projection onto N_u is ZERO, it is an orbit direction. "
                  f"H_u along the unprojected unit tangent = {mp.nstr(val, 12)}  [orbit-null control]")
        else:
            kept.append(proj); names.append(nm)
            print(f"  {nm}: transverse fraction {mp.nstr(frac, 8)}  (squared {mp.nstr(frac**2, 8)} = {ident(frac**2)})")
    if not kept:
        return
    # discard directions that duplicate an earlier one (a real 2-parameter chart on a
    # projective line carries only one independent direction per real dimension)
    ind, ind_names = [], []
    for d, nm in zip(kept, names):
        w = d.copy()
        for b in ind:
            w -= (b.T*d)[0]/mp.norm(b)**2*b
        if mp.norm(w)/mp.norm(d) > mp.mpf('1e-12'):
            ind.append(d); ind_names.append(nm)
        else:
            print(f"  {nm}: dependent on the directions already kept, dropped")
    print(f"  independent transverse directions: {len(ind)}  {ind_names}")
    for i in range(len(ind)):
        for j in range(i, len(ind)):
            v = (ind[i].T*(M*ind[j]))[0]/(mp.norm(ind[i])*mp.norm(ind[j]))
            print(f"    H[{ind_names[i]},{ind_names[j]}] = {mp.nstr(v, 22)}   = {ident(v)}")
    print("  Gram of the normalized projected directions:")
    for i in range(len(ind)):
        print("   ", [mp.nstr((ind[i].T*ind[j])[0]/(mp.norm(ind[i])*mp.norm(ind[j])), 10) for j in range(len(ind))])

# prism: D3 chart, b1 = v3 + v-3, b2 = v0, at z = sqrt(230)/10
def d3(x, y): return {3: mp.mpc(1), -3: mp.mpc(1), 0: mp.mpc(x, y)}
x0 = mp.sqrt(230)/10
report("prism, D3 chart at z = sqrt(230)/10", unitv(d3(x0, mp.mpf(0))),
       tangents(d3, [x0, mp.mpf(0)]), ["d/dx", "d/dy"])

# pyramid: C5 line {v3, v-2}, rep sqrt(12) v3 + sqrt(13) v-2, with the RELATIVE PHASE
def c5(a, b, ph): return {3: mp.mpc(a), -2: mp.mpc(b)*mp.expjpi(ph)}
a0, b0 = mp.sqrt(12), mp.sqrt(13)
report("pyramid, C5 line at s* = 12/25", unitv(c5(a0, b0, mp.mpf(0))),
       tangents(c5, [a0, b0, mp.mpf(0)]), ["d/da", "d/db", "d/dphase"])
