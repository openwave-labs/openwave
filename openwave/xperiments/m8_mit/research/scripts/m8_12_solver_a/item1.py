"""Item 1: fixed spaces V^(H,chi) for all closed subgroups H of SO(3) up to conjugacy.

(a) finite groups C_n (n<=12), D_n (2<=n<=12), T, O, I: every 1-dim character is found by
    brute force over generator values, the fixed-space dimension by the character formula
    dim = (1/|H|) sum_h conj(chi(h)) tr D3(h),  tr D3(rot by t) = 1 + 2cos t + 2cos 2t + 2cos 3t.
(b) C_n, D_n with the standard axes: explicit exact fixed spaces.
(c) continuous groups SO(2), O(2), SO(3): exact Lie-algebra computation.
(d) rotation equivalences exhibited by exact rotations; separations by exact invariants.
"""
import itertools, math
import numpy as np
import sympy as sp
import common as C

ck = C.Checker('item1')
I = sp.I
res = {}

# ------------------------------------------------------------------ (a) finite groups
def rotm(axis, ang):
    a = np.array(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K

def closure(gens):
    """BFS closure; returns list of (matrix, word) with word = tuple of generator indices."""
    elems = [(np.eye(3), ())]
    frontier = [elems[0]]
    while frontier:
        new = []
        for (M, w) in frontier:
            for gi, G in enumerate(gens):
                P = G @ M
                if not any(np.abs(P - E).max() < 1e-9 for E, _ in elems):
                    elems.append((P, (gi,) + w)); new.append((P, (gi,) + w))
        frontier = new
    return elems

def gen_order(G):
    P = np.eye(3)
    for k in range(1, 100):
        P = G @ P
        if np.abs(P - np.eye(3)).max() < 1e-9: return k

def characters(gens, elems):
    """All homomorphisms H -> U(1), as exponent fractions mod 1 per element.
    Generator values range over roots of unity of the generator's order; an assignment
    is kept iff it is consistent on every relation met during a second BFS."""
    orders = [gen_order(G) for G in gens]
    chars = []
    for vals in itertools.product(*[range(o) for o in orders]):
        ex = [sp.Rational(v, o) for v, o in zip(vals, orders)]
        # value on each element via its word, then check multiplicativity on all products g*h
        def val(w): return sum(ex[i] for i in w) % 1
        table = [val(w) for _, w in elems]
        ok = True
        for a, (Ma, wa) in enumerate(elems):
            for gi, G in enumerate(gens):
                P = G @ Ma
                b = next(k for k, (E, _) in enumerate(elems) if np.abs(P - E).max() < 1e-9)
                if (table[a] + ex[gi] - table[b]) % 1 != 0: ok = False; break
            if not ok: break
        if ok: chars.append((ex, table))
    return chars

def trD3(M):
    c = (np.trace(M) - 1) / 2
    c = max(-1.0, min(1.0, c)); t = math.acos(c)
    return 1 + 2 * math.cos(t) + 2 * math.cos(2 * t) + 2 * math.cos(3 * t)

phi = (1 + 5 ** 0.5) / 2
groups = {}
for n in range(1, 13):
    groups['C%d' % n] = [rotm([0, 0, 1], 2 * math.pi / n)]
for n in range(2, 13):
    groups['D%d' % n] = [rotm([0, 0, 1], 2 * math.pi / n), rotm([0, 1, 0], math.pi)]
groups['T'] = [rotm([0, 0, 1], math.pi), rotm([1, 1, 1], 2 * math.pi / 3)]
groups['O'] = [rotm([0, 0, 1], math.pi / 2), rotm([1, 1, 1], 2 * math.pi / 3)]
groups['I'] = [rotm([0, 1, phi], 2 * math.pi / 5), rotm([1, 1, 1], 2 * math.pi / 3)]
expected_order = {**{'C%d' % n: n for n in range(1, 13)}, **{'D%d' % n: 2 * n for n in range(2, 13)},
                  'T': 12, 'O': 24, 'I': 60}

finite = {}
maxres = 0.0
for name, gens in groups.items():
    el = closure(gens)
    ck.check('%s has order %d' % (name, expected_order[name]), len(el) == expected_order[name], str(len(el)))
    chs = characters(gens, el)
    dims = []
    for gvals, ch in chs:
        s = sum(complex(sp.exp(-2 * sp.pi * I * e).evalf(30)) * trD3(M) for (M, _), e in zip(el, ch)) / len(el)
        d = round(s.real); maxres = max(maxres, abs(s - d))
        gv = [str(e) for e in gvals]
        dims.append({'chi_on_generators_as_fraction_of_turn': gv, 'dim': d})
    finite[name] = {'order': len(el), 'n_characters': len(chs), 'fixed_dims': dims}
ck.check('character-formula dimensions are integers (max residual %.1e, double precision)' % maxres, maxres < 1e-9)
res['finite'] = finite
# expected character counts: |H/[H,H]|
exp_nchar = {**{'C%d' % n: n for n in range(1, 13)}, **{'D%d' % n: (2 if n % 2 else 4) for n in range(2, 13)},
             'T': 3, 'O': 2, 'I': 1}
ck.check('number of characters equals |H/[H,H]| for every group',
         all(finite[k]['n_characters'] == exp_nchar[k] for k in finite),
         str({k: finite[k]['n_characters'] for k in finite if finite[k]['n_characters'] != exp_nchar[k]}))

hist = {}
for name, d in finite.items():
    for e in d['fixed_dims']:
        hist.setdefault(e['dim'], []).append((name, e['chi_on_generators_as_fraction_of_turn']))
res['dims_found'] = sorted(hist)
print('fixed-space dimensions occurring over finite groups:', sorted(hist))
for dm in sorted(hist):
    print('  dim %d:' % dm, ', '.join('%s%s' % (n, tuple(v)) for n, v in hist[dm][:40]))

# ------------------------------------------------------------------ (b) explicit C_n, D_n fixed spaces
def vec(d):
    c = [sp.Integer(0)] * 7
    for m, a in d.items(): c[C.idx(m)] += sp.sympify(a)
    return sp.Matrix(c)

def fixed_space_exact(mats_and_chars):
    """nullspace of stacked (D(h) - chi(h)) for generators h."""
    A = sp.Matrix.vstack(*[D - ch * sp.eye(7) for D, ch in mats_and_chars])
    return A.nullspace(simplify=True)

Rz = lambda n: sp.diag(*[sp.exp(-I * 2 * sp.pi * m / n) for m in C.MS])
Ry = C.rot_axis_angle([0, 1, 0], sp.Rational(1, 2))
explicit = {}
for n in range(1, 13):
    for k in range(n):
        ns = fixed_space_exact([(Rz(n), sp.exp(-I * 2 * sp.pi * k / n))])
        explicit[('C%d' % n, k)] = ns
        ms = [m for m in C.MS if (m - k) % n == 0]
        ck.check('C%d k=%d fixed space = span{v_m : m = k mod n} (dim %d)' % (n, k, len(ms)),
                 len(ns) == len(ms) and (len(ms) == 0 or C.same_span(sp.Matrix.hstack(*ns), sp.Matrix.hstack(*[vec({m: 1}) for m in ms]))))
        if n >= 2:
            for eps in (1, -1):
                ns2 = fixed_space_exact([(Rz(n), sp.exp(-I * 2 * sp.pi * k / n)), (Ry, eps)])
                explicit[('D%d' % n, k, eps)] = ns2
dimsC = {key: len(v) for key, v in explicit.items()}

# the seven dimension-2 classes and their producing (H, chi)
s2 = sp.sqrt(2)
classes2 = {
    'A': dict(H='C3', chi='R_z(2pi/3) -> e^{-2pi i/3}', basis=[vec({1: 1}), vec({-2: 1})], key=('C3', 1)),
    'B': dict(H='C4', chi='R_z(pi/2) -> e^{-i pi/2} = -i', basis=[vec({1: 1}), vec({-3: 1})], key=('C4', 1)),
    'C': dict(H='C4', chi='R_z(pi/2) -> e^{-i pi} = -1', basis=[vec({2: 1}), vec({-2: 1})], key=('C4', 2)),
    'D': dict(H='C5', chi='R_z(2pi/5) -> e^{-4pi i/5}', basis=[vec({2: 1}), vec({-3: 1})], key=('C5', 2)),
    'E': dict(H='C6', chi='R_z(pi/3) -> e^{-i pi} = -1', basis=[vec({3: 1}), vec({-3: 1})], key=('C6', 3)),
    'F': dict(H='D2', chi='R_z(pi) -> +1, R_y(pi) -> -1, R_x(pi) -> -1', basis=[vec({0: 1}), vec({2: 1 / s2, -2: 1 / s2})], key=('D2', 0, -1)),
    'G': dict(H='D3', chi='R_z(2pi/3) -> 1, R_y(pi) -> -1', basis=[vec({0: 1}), vec({3: 1 / s2, -3: -1 / s2})], key=('D3', 0, -1)),
}
for nm, d in classes2.items():
    ns = explicit[d['key']]
    ck.check('class %s = V^(%s, %s) exactly' % (nm, d['H'], d['chi']),
             len(ns) == 2 and C.same_span(sp.Matrix.hstack(*ns), sp.Matrix.hstack(*d['basis'])))

classes1 = {
    'v3': dict(H='SO(2)_z', chi='R_z(t) -> e^{-3it}', u=vec({3: 1})),
    'v2': dict(H='SO(2)_z', chi='R_z(t) -> e^{-2it}', u=vec({2: 1})),
    'v1': dict(H='SO(2)_z', chi='R_z(t) -> e^{-it}', u=vec({1: 1})),
    'v0': dict(H='O(2)_z', chi='det-type: R_z(t) -> 1, flips -> -1', u=vec({0: 1})),
    'xyz': dict(H='O (also T with trivial chi)', chi='A2 sign character of O', u=vec({2: 1, -2: -1})),
    'cat': dict(H='D6', chi='R_z(pi/3) -> -1, R_y(pi) -> +1', u=vec({3: 1, -3: 1})),
}

# ------------------------------------------------------------------ (c) continuous groups, exact
# SO(2)_z with chi_k(R_z(t)) = e^{-ikt}: fixed space = ker(Jz - k)
for k in range(-4, 5):
    ns = (C.JZ - k * sp.eye(7)).nullspace()
    ck.check('SO(2) chi_k, k=%d: fixed dim %d' % (k, 1 if abs(k) <= 3 else 0), len(ns) == (1 if abs(k) <= 3 else 0))
# O(2)_z: continuous characters must satisfy chi(R_z(t)) = chi(R_z(-t)) -> trivial on SO(2); chi(flip)=+-1
for eps in (1, -1):
    ns = fixed_space_exact([(C.JZ, 0), (Ry, eps)])
    ck.check('O(2) chi(flip)=%+d: fixed dim %d' % (eps, 0 if eps == 1 else 1), len(ns) == (0 if eps == 1 else 1))
# SO(3): only the trivial character; fixed space = ker Jx cap ker Jy cap ker Jz
ns = fixed_space_exact([(C.JX, 0), (C.JY, 0), (C.JZ, 0)])
ck.check('SO(3): fixed dim 0', len(ns) == 0)

# ------------------------------------------------------------------ (d) equivalences, exhibited exactly
def maps_onto(D, W1, W2):
    return C.same_span(D * sp.Matrix.hstack(*W1), sp.Matrix.hstack(*W2))
equiv = []
def eq(name, D, W1, W2):
    ok = maps_onto(D, W1, W2); equiv.append((name, ok))
    ck.check('exhibited rotation: ' + name, ok)
eq('R_y(pi): V^(C3,k=1)={v1,v-2} -> V^(C3,k=2)={v2,v-1}', Ry, [vec({1: 1}), vec({-2: 1})], [vec({2: 1}), vec({-1: 1})])
eq('R_y(pi): V^(C4,k=1)={v1,v-3} -> V^(C4,k=3)={v3,v-1}', Ry, [vec({1: 1}), vec({-3: 1})], [vec({3: 1}), vec({-1: 1})])
eq('R_y(pi): V^(C5,k=2)={v2,v-3} -> V^(C5,k=3)={v3,v-2}', Ry, [vec({2: 1}), vec({-3: 1})], [vec({3: 1}), vec({-2: 1})])
R111 = C.rot_axis_angle([1, 1, 1], sp.Rational(1, 3))
Fz = classes2['F']['basis']
Wy = explicit[('D2', 1, 1)]; Wx = explicit[('D2', 1, -1)]
eq('R_(111)(2pi/3) maps the D2 character (z:+,y:-) space F onto another D2 character space', R111, Fz, Wy) if maps_onto(R111, Fz, Wy) else eq('R_(111)(2pi/3) maps F onto the (z:-,y:-) space', R111, Fz, Wx)
eq('R_(111)(-2pi/3) maps F onto the remaining D2 character space', R111.H, Fz, Wx if maps_onto(R111, Fz, Wy) else Wy)
Rz12 = C.rot_axis_angle([0, 0, 1], sp.Rational(1, 12))
Rz8 = C.rot_axis_angle([0, 0, 1], sp.Rational(1, 8))
eq('R_z(pi/6): [v3+v-3] -> [v3-v-3]', Rz12, [vec({3: 1, -3: 1})], [vec({3: 1, -3: -1})])
eq('R_z(pi/4): [v2+v-2] -> [v2-v-2]', Rz8, [vec({2: 1, -2: 1})], [vec({2: 1, -2: -1})])
eq('R_y(pi): [v_m] -> [v_-m] for m=3', Ry, [vec({3: 1})], [vec({-3: 1})])
# the T-fixed line and the D2-trivial line are the xyz line; the O line (A2) is the same line
for gname, gens_ex in [('T trivial', [(C.rot_axis_angle([0, 0, 1], sp.Rational(1, 2)), 1), (R111, 1)]),
                       ('O sign', [(C.rot_axis_angle([0, 0, 1], sp.Rational(1, 4)), -1), (R111, 1)]),
                       ('D2 trivial', [(C.rot_axis_angle([0, 0, 1], sp.Rational(1, 2)), 1), (Ry, 1)])]:
    ns = fixed_space_exact(gens_ex)
    ck.check('%s fixed space = [v2 - v-2]' % gname, len(ns) == 1 and C.same_span(ns[0], vec({2: 1, -2: -1})))

# ------------------------------------------------------------------ (e) exact separating invariants
# lines: the multipole invariants r_L (L=1..6) are rotation- and phase-invariant
inv1 = {}
for nm, d in classes1.items():
    inv1[nm] = [C.rhat_exact(list(d['u']), L) for L in range(1, 7)]
    print('line %-4s r_1..r_6 =' % nm, [str(x) for x in inv1[nm]])
names = list(inv1)
ck.check('the six line classes have pairwise distinct (r_1..r_6)',
         all(inv1[a] != inv1[b] for a, b in itertools.combinations(names, 2)))
# every 1-dim fixed space met in the explicit C_n / D_n enumeration (n <= 12) is one of the six classes:
# identify it by its invariants, then confirm by an exhibited rotation (R_y(pi) for v_-m, R_z for v3+-v-3 and v2+-v-2)
cands = {'v3': [vec({3: 1}), vec({-3: 1})], 'v2': [vec({2: 1}), vec({-2: 1})], 'v1': [vec({1: 1}), vec({-1: 1})],
         'v0': [vec({0: 1})], 'xyz': [vec({2: 1, -2: 1}), vec({2: 1, -2: -1})], 'cat': [vec({3: 1, -3: 1}), vec({3: 1, -3: -1})]}
unmatched = []; lines_seen = {}
for key, ns in explicit.items():
    if len(ns) != 1: continue
    v1_ = ns[0]
    hit = [nm for nm, vs in cands.items() if any(C.same_span(v1_, w) for w in vs)]
    if not hit: unmatched.append(key)
    else: lines_seen.setdefault(hit[0], []).append(key)
ck.check('every 1-dim C_n/D_n fixed space (n<=12) is literally one of v_+-m, v2+-v-2, v3+-v-3', not unmatched, str(unmatched[:5]))
print('1-dim fixed spaces by class:', {k: len(v) for k, v in lines_seen.items()})
res['one_dim_fixed_spaces_by_class'] = {k: [str(x) for x in v] for k, v in lines_seen.items()}
# and the listed alternatives are rotation-equivalent to the representatives (exhibited above:
# R_y(pi) v_m -> v_-m, R_z(pi/6) [v3+v-3] -> [v3-v-3], R_z(pi/4) [v2+v-2] -> [v2-v-2])
# every 2-dim C_n/D_n fixed space (n<=12) is literally one of the plane representatives or an exhibited image
planes_known = [classes2[nm]['basis'] for nm in classes2] + [[vec({2: 1}), vec({-1: 1})], [vec({3: 1}), vec({-1: 1})],
               [vec({3: 1}), vec({-2: 1})], list(explicit[('D2', 1, 1)]), list(explicit[('D2', 1, -1)])]
um2 = [key for key, ns in explicit.items() if len(ns) == 2 and not any(
    C.same_span(sp.Matrix.hstack(*ns), sp.Matrix.hstack(*pk)) for pk in planes_known)]
ck.check('every 2-dim C_n/D_n fixed space (n<=12) is a listed plane or one of its exhibited rotation images', not um2, str(um2))
# planes: the image of rhat6 on P(W) is a rotation invariant of W (if gW = W' then
# rhat6(P(W')) = rhat6(P(W)) by rotation invariance).  P(W) is compact, so the range is
# [min, max] over the complete exact critical set of item 2 (classes.critical_set).
import classes as K
rng = {}
for nm, d in classes2.items():
    crit, info = K.critical_set(nm)
    vals = [e['value'] for e in crit]
    rng[nm] = (min(vals), max(vals))
    print('plane %s: stabilizer %s, character %s, range of rhat6 = [%s, %s]' % (nm, d['H'], d['chi'], rng[nm][0], rng[nm][1]))
ck.check('the seven plane classes have pairwise distinct rhat6-ranges',
         len(set(rng.values())) == 7)
res['classes_dim2'] = {nm: {'H': d['H'], 'chi': d['chi'], 'basis': [str(list(v)) for v in d['basis']],
                            'rhat6_range': [str(rng[nm][0]), str(rng[nm][1])]} for nm, d in classes2.items()}
res['classes_dim1'] = {nm: {'H': d['H'], 'chi': d['chi'], 'u': str(list(d['u'])),
                            'r1..r6': [str(v) for v in inv1[nm]]} for nm, d in classes1.items()}
C.save('item1', res)
print('item1 fails:', ck.fails)
