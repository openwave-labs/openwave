"""Item 4: unite critical points of items 2 and 3 modulo rotations and phase."""
import itertools
import sympy as sp, mpmath as mp
import common as C, classes as K
from orbits import ORBITS

ck = C.Checker('item4')
R = sp.Rational; s2 = sp.sqrt(2)
res = {'orbits': {}, 'same_orbit_by_rotation': [], 'separations': []}

inv = {}
for nm, d in ORBITS.items():
    c = list(d['u'])
    ck.check('%s: representative is a unit vector' % nm, sp.simplify(C.norm2(c) - 1) == 0)
    val = C.rhat_exact(c)
    rl = [C.rhat_exact(c, L) for L in range(1, 7)]
    s3 = C.sextic_invariants(c, [(2, 2, 2)])[(2, 2, 2)]
    inv[nm] = dict(val=val, rL=rl, t222=s3)
    mp.mp.dps = 50
    vm = C.mp_rhat(C.mp_vec(c))
    ck.check('%s: 50-digit value agrees with exact %s' % (nm, val), abs(vm - mp.mpf(val.p) / val.q) < mp.mpf(10) ** -45)
    res['orbits'][nm] = {'rhat6': str(val), '924*rhat6': str(924 * val), 'u': str(list(d['u'])), 'stabilizer': d['stab'],
                         'occurs_as': d['where'], 'r1..r6': [str(v) for v in rl], 'tr(rho_2^3)': str(s3)}
    print('%-4s rhat6 = %-8s = %9s/924  stab %-5s r1..r5 = %s  tr(rho2^3) = %s' % (nm, val, 924 * val, d['stab'], [str(v) for v in rl[:5]], s3))

ck.check('the 10 orbits have pairwise distinct invariant vectors (rhat6, r1..r5, tr rho2^3)',
         len({(tuple(v['rL']), v['t222']) for v in inv.values()}) == len(inv))

# ---- same value -> decide
byval = {}
for nm, v in inv.items(): byval.setdefault(v['val'], []).append(nm)
for val, names in byval.items():
    if len(names) > 1:
        for a, b in itertools.combinations(names, 2):
            if inv[a]['rL'] != inv[b]['rL']:
                L = next(k for k in range(6) if inv[a]['rL'][k] != inv[b]['rL'][k]) + 1
                how = 'r_%d: %s vs %s (exact, rotation-invariant)' % (L, inv[a]['rL'][L - 1], inv[b]['rL'][L - 1])
            else:
                how = ('all quartic invariants r_1..r_6 agree; tr(rho_2^3): %s vs %s (exact, rotation-invariant); '
                       'also stabilizers %s vs %s are non-isomorphic' % (inv[a]['t222'], inv[b]['t222'], ORBITS[a]['stab'], ORBITS[b]['stab']))
                ck.check('%s vs %s separated by tr(rho_2^3)' % (a, b), inv[a]['t222'] != inv[b]['t222'])
            res['separations'].append({'pair': [a, b], 'value': str(val), 'separated_by': how})
            print('shared value %s: %s vs %s -> two orbits; %s' % (val, a, b, how))

# ---- critical points found more than once: exhibit exact rotations
def rot_euler(a, cb, sb, g):
    """R_z(a) R_y(beta) R_z(g) exactly, with cos/sin of beta given."""
    return (C.rot_axis_angle([0, 0, 1], a) * C.exact_rot([0, 1, 0], cb, sb) * C.rot_axis_angle([0, 0, 1], g))
def check_map(name, D, src, tgt):
    ok = C.same_span(D * sp.Matrix(src), sp.Matrix(tgt))
    ck.check('rotation exhibited: ' + name, ok)
    res['same_orbit_by_rotation'].append({'map': name, 'verified_exactly': ok})
v = K.vec
Fx = v({0: 1, 2: sp.sqrt(R(3, 5)) / s2, -2: sp.sqrt(R(3, 5)) / s2})
Gx = v({0: 1, 3: sp.sqrt(R(4, 5)) / s2, -3: -sp.sqrt(R(4, 5)) / s2})
Ry = C.rot_axis_angle([0, 1, 0], R(1, 2))
check_map('R_y(pi/2) R_z(pi/2): cat -> F point z = sqrt(3/5)', rot_euler(0, 0, 1, R(1, 4)), ORBITS['cat']['u'], Fx)
check_map('R_z(5pi/3) R_y(arccos(1/sqrt3)) R_z(pi/4): xyz -> G point z = 2/sqrt5', rot_euler(R(5, 6), 1 / sp.sqrt(3), sp.sqrt(R(2, 3)), R(1, 8)), ORBITS['xyz']['u'], Gx)
check_map('R_z(pi/2): F point z -> -z (z = sqrt(3/5))', C.rot_axis_angle([0, 0, 1], R(1, 4)), Fx, v({0: 1, 2: -sp.sqrt(R(3, 5)) / s2, -2: -sp.sqrt(R(3, 5)) / s2}))
check_map('R_z(pi/2): F point z -> -z (z = i sqrt(5/3))', C.rot_axis_angle([0, 0, 1], R(1, 4)), ORBITS['F*']['u'], v({0: 1, 2: -sp.I * sp.sqrt(R(5, 3)) / s2, -2: -sp.I * sp.sqrt(R(5, 3)) / s2}))
check_map('R_z(pi/3): G point z -> -z (z = 2/sqrt5)', C.rot_axis_angle([0, 0, 1], R(1, 6)), Gx, v({0: 1, 3: -sp.sqrt(R(4, 5)) / s2, -3: sp.sqrt(R(4, 5)) / s2}))
check_map('R_z(pi/3): G point z -> -z (z = i sqrt(20/23))', C.rot_axis_angle([0, 0, 1], R(1, 6)), ORBITS['G*']['u'], v({0: 1, 3: -sp.I * sp.sqrt(R(20, 23)) / s2, -3: sp.I * sp.sqrt(R(20, 23)) / s2}))
check_map('R_z(pi/4): (v2+v-2)/sqrt2 (omitted point of F) -> xyz line', C.rot_axis_angle([0, 0, 1], R(1, 8)), v({2: 1, -2: 1}), ORBITS['xyz']['u'])
check_map('R_z(pi/6): (v3-v-3)/sqrt2 (omitted point of G) -> cat line', C.rot_axis_angle([0, 0, 1], R(1, 12)), v({3: 1, -3: -1}), ORBITS['cat']['u'])
for m in (1, 2, 3):
    check_map('R_y(pi): v%d -> v-%d' % (m, m), Ry, v({m: 1}), v({-m: 1}))
# circles: R_z(theta) multiplies the two components of a + z b by different phases, so it moves
# z along its circle |z| = const; representative check at theta = 2pi/7 (generic angle)
Rz7 = C.rot_axis_angle([0, 0, 1], R(1, 7))
for nm, (p, q, zabs) in {'A': (1, -2, R(1, 2)), 'D': (2, -3, sp.sqrt(R(12, 13))), 'C': (2, -2, 1), 'E': (3, -3, 1)}.items():
    src = v({p: 1, q: zabs}); ph = sp.exp(-2 * sp.pi * sp.I * (q - p) / 7)
    check_map('R_z(2pi/7) moves circle point of %s to z = %s * e^{-2pi i (%d)/7}' % (nm, zabs, q - p), Rz7, src, v({p: 1, q: zabs * ph}))
# control: a map that should NOT hold (checks the test can fail)
ok = C.same_span(Ry * sp.Matrix(ORBITS['cat']['u']), sp.Matrix(Fx))
ck.check('control: R_y(pi) does NOT map cat onto the F point (test can fail)', not ok)
C.save('item4', res)
print('item4 fails:', ck.fails)
