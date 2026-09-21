"""Audit item 5, numerical route: c-coordinates, mpmath, eigenvalues of H_u on an orthonormal N_u basis;
compared against the roots of the exact characteristic polynomials from a5.py."""
import pickle
import sympy as sp, mpmath as mp
from aud_core import *

res = pickle.load(open('out_a5.pkl', 'rb'))
lam = sp.Symbol('lam')
def orbit_reps():
    r2, r3, r5 = mp.sqrt(2), mp.sqrt(3), mp.sqrt(5)
    def vc(d):
        c = [mp.mpc(0)] * 7
        for m, z in d.items(): c[idx(m)] = mp.mpc(z)
        n = mp.sqrt(sum(abs(t) ** 2 for t in c))
        return [t / n for t in c]
    j = mp.mpc(0, 1)
    return {'v3': vc({3: 1}), 'v2': vc({2: 1}), 'v1': vc({1: 1}), 'v0': vc({0: 1}),
            'xyz': vc({2: 1, -2: -1}), 'cat': vc({3: 1, -3: 1}), 'A*': vc({1: 2, -2: 1}),
            'D*': vc({2: mp.sqrt(13), -3: 2 * r3}), 'F*': vc({0: mp.sqrt(6), 2: j * r5, -2: j * r5}),
            'G*': vc({0: mp.sqrt(23), 3: j * mp.sqrt(10), -3: -j * mp.sqrt(10)})}

def run(dps):
    mp.mp.dps = dps
    cg = mp_cg()
    def Bq(xv, yv, Q):
        s = mp.mpc(0)
        for m1 in MS:
            m2 = Q - m1
            if abs(m2) <= 3:
                s += cg[(m1, m2, Q)] * xv[idx(m1)] * (-1) ** (3 - m2) * mp.conj(yv[idx(-m2)])
        return s
    E = []
    for k in range(7):
        e = [mp.mpc(0)] * 7; e[k] = mp.mpc(1); E.append(e)
    for k in range(7):
        e = [mp.mpc(0)] * 7; e[k] = mp.mpc(0, 1); E.append(e)
    Jx, Jy, Jz = [A.evalf(dps + 10) for A in Jmat()]
    def mv(Mx, v): return [sum(mp.mpc(complex(0)) + mp.mpc(sp.re(Mx[i, k]), sp.im(Mx[i, k])) * v[k] for k in range(7)) for i in range(7)]
    def rv(c): return mp.matrix([mp.re(z) for z in c] + [mp.im(z) for z in c])
    worst_all = 0; out = {}
    for name, u in orbit_reps().items():
        rh = [Bq(u, u, Q) for Q in range(-6, 7)]
        Nn = sum(abs(r) ** 2 for r in rh)
        Ls = [[Bq(u, e, Q) + Bq(e, u, Q) for Q in range(-6, 7)] for e in E]
        H = mp.matrix(14, 14)
        for a in range(14):
            for b in range(a, 14):
                s = mp.mpf(0)
                for qi, Q in enumerate(range(-6, 7)):
                    s += 2 * mp.re(mp.conj(Ls[a][qi]) * Ls[b][qi])
                    s += 2 * mp.re(mp.conj(rh[qi]) * (Bq(E[a], E[b], Q) + Bq(E[b], E[a], Q)))
                H[a, b] = s; H[b, a] = s
        M = H - 4 * Nn * mp.eye(14)
        gens = [[mp.mpc(0, 1) * z for z in u]] + [[mp.mpc(0, -1) * z for z in mv(J, u)] for J in (Jx, Jy, Jz)]
        # orthonormal basis of span{u, O_u} by modified Gram-Schmidt with rank detection, then complete
        vecs = [rv(u)] + [rv(g) for g in gens]
        ortho = []
        resid_O = max(mp.norm(M * rv(g)) for g in gens)
        for v in vecs + [mp.matrix([1 if i == k else 0 for i in range(14)]) for k in range(14)]:
            w = v.copy()
            for _ in range(2):
                for q in ortho: w = w - (q.T * w)[0] * q
            if mp.norm(w) > mp.mpf(10) ** (-dps // 2):
                ortho.append(w / mp.norm(w))
        nO = None
        # the first vectors that survived from vecs form span{u,O_u}
        k = 0; basisO = []
        cnt = 0
        for v in vecs:
            pass
        # recompute: dimension of span{u,O} = rank of those five
        Vm = mp.matrix(14, 5)
        for j, v in enumerate(vecs):
            for i in range(14): Vm[i, j] = v[i]
        sv = mp.svd_r(Vm, compute_uv=False)
        rank = sum(1 for s in sv if s > mp.mpf(10) ** (-dps // 2))
        Nb = ortho[rank:]
        assert len(ortho) == 14
        Hn = mp.matrix(len(Nb), len(Nb))
        for i in range(len(Nb)):
            for j in range(len(Nb)):
                Hn[i, j] = (Nb[i].T * M * Nb[j])[0]
        ev = sorted(mp.eigsy(Hn, eigvals_only=True))
        # exact roots
        cp = sp.sympify(res[name]['charpoly'])
        roots = []
        for f_, mult in sp.factor_list(cp)[1]:
            rr = sp.Poly(f_, lam).nroots(n=dps + 15)
            roots += [mp.mpf(str(r)) for r in rr] * mult
        roots = sorted(roots)
        worst = max(abs(a - b) for a, b in zip(ev, roots))
        worst_all = max(worst_all, worst)
        out[name] = (len(Nb), worst, resid_O, min(abs(e) for e in ev))
        print('  %-4s dps=%d dimN=%d  max|eig - exact root| = %s   max|M_u o| over O gens = %s   min|eig| = %s' % (
            name, dps, len(Nb), mp.nstr(worst, 3), mp.nstr(resid_O, 3), mp.nstr(min(abs(e) for e in ev), 3)))
    return worst_all
for dps in (50, 80):
    print('dps', dps, 'worst', mp.nstr(run(dps), 3))
