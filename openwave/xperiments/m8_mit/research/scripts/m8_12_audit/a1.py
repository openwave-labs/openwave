"""Audit item 1: census of fixed spaces V^(H,chi) for finite subgroups (numerical joint eigenspaces,
mpmath 30 digits, SVD rank) + continuous ones by argument; classification of the dim-1 and dim-2 ones."""
import itertools
import numpy as np, mpmath as mp
import sympy as sp
from aud_core import MS, idx, Jmat

mp.mp.dps = 30
Jx, Jy, Jz = [np.array(A.evalf(40).tolist(), dtype=complex) for A in Jmat()]
def Dmat(n, th):
    n = np.array(n, float); n = n / np.linalg.norm(n)
    G = n[0] * Jx + n[1] * Jy + n[2] * Jz
    w, V = np.linalg.eigh(G)
    return V @ np.diag(np.exp(-1j * th * w)) @ V.conj().T
def R3(n, th):
    n = np.array(n, float); n = n / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
phi = (1 + 5 ** 0.5) / 2
GROUPS = {}
for n in range(1, 13):
    GROUPS['C%d' % n] = [((0, 0, 1), 2 * np.pi / n, n)]
for n in range(2, 13):
    GROUPS['D%d' % n] = [((0, 0, 1), 2 * np.pi / n, n), ((1, 0, 0), np.pi, 2)]
GROUPS['T'] = [((0, 0, 1), np.pi, 2), ((1, 1, 1), 2 * np.pi / 3, 3)]
GROUPS['O'] = [((0, 0, 1), np.pi / 2, 4), ((1, 1, 1), 2 * np.pi / 3, 3)]
GROUPS['I'] = [((0, 1, phi), 2 * np.pi / 5, 5), ((1, 1, 1), 2 * np.pi / 3, 3)]
def group_order(gens):
    els = [np.eye(3)]
    frontier = [np.eye(3)]
    G3 = [R3(a, t) for a, t, _ in gens]
    while frontier:
        new = []
        for g in frontier:
            for h in G3:
                k = h @ g
                if not any(np.allclose(k, e, atol=1e-9) for e in els):
                    els.append(k); new.append(k)
        frontier = new
        if len(els) > 200: break
    return len(els)
census = []
for name, gens in GROUPS.items():
    order = group_order(gens)
    Ds = [Dmat(a, t) for a, t, _ in gens]
    # D^3 is a genuine SO(3) rep: generator order check
    for (a, t, o), D in zip(gens, Ds):
        assert np.allclose(np.linalg.matrix_power(D, o), np.eye(7), atol=1e-10)
    for chis in itertools.product(*[range(o) for _, _, o in gens]):
        rows = np.vstack([D - np.exp(2j * np.pi * k / o) * np.eye(7) for D, k, (_, _, o) in zip(Ds, chis, gens)])
        sv = np.linalg.svd(rows, compute_uv=False)
        dim = int(sum(sv < 1e-9))
        gap = min([s for s in sv if s >= 1e-9] or [9])
        if dim:
            _, _, Vh = np.linalg.svd(rows)
            basis = Vh.conj().T[:, -dim:]
        else:
            basis = None
        census.append((name, order, chis, dim, gap, basis))
print('group orders:', {n: group_order(g) for n, g in GROUPS.items()})
print('smallest nonzero singular value over all (H,chi) (rank gap):', min(c[4] for c in census))
from collections import Counter
print('dimension counts:', sorted(Counter(c[3] for c in census).items()))
print('dimensions other than 1,2 and where:')
for d in (3, 4, 7):
    print('  dim %d:' % d, [(c[0], c[2]) for c in census if c[3] == d])
print('  dim 0 count:', sum(1 for c in census if c[3] == 0))

# classify: invariants for lines (rhat6 via mp) and planes (range of rhat6 sampled + exact matching below)
from aud_core import mp_rhat, mp_cg
cg = mp_cg()
def rh(v): return float(mp_rhat([mp.mpc(complex(z)) for z in v], cg))
lines = {}
for c in census:
    if c[3] == 1:
        val = round(rh(c[5][:, 0]) * 924, 6)
        lines.setdefault(val, []).append((c[0], c[2]))
print('dim-1 fixed spaces grouped by 924*rhat6 (a rotation invariant):')
for k, v in sorted(lines.items()): print('  %10.4f  (%d spaces) e.g. %s' % (k, len(v), v[:4]))
# planes: invariant = (min, max) of rhat6 on P(W) (numerical, 400-pt grid + local refine) and the spectrum of P Jz^2...
def plane_inv(Bm):
    a, b = Bm[:, 0], Bm[:, 1]
    vals = []
    for th in np.linspace(0, np.pi, 61):
        for ph in np.linspace(0, 2 * np.pi, 61):
            vals.append(rh(np.cos(th / 2) * a + np.exp(1j * ph) * np.sin(th / 2) * b))
    # rotation-invariant operator spectrum: P (Jx^2+Jy^2+Jz^2 restricted weights) -> use P (sum_a (P J_a P)^2) P
    Pm = Bm @ Bm.conj().T
    S = sum((Pm @ J @ Pm) @ (Pm @ J @ Pm) for J in (Jx, Jy, Jz))
    ev = np.linalg.eigvalsh(Bm.conj().T @ S @ Bm)
    return (round(min(vals) * 924, 1), round(max(vals) * 924, 1), tuple(np.round(ev, 6)))
planes = {}
for c in census:
    if c[3] == 2:
        planes.setdefault(plane_inv(c[5]), []).append((c[0], c[2]))
print('dim-2 fixed spaces grouped by (924*min, 924*max of rhat6 on P(W) [grid, numerical], spectrum of sum (PJP)^2):')
for k, v in sorted(planes.items()): print('  ', k, len(v), v[:6])
