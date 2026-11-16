"""0(a), 0(b): the group from its two generators, classes, characters (DERIVED), levels <= 20.
Route: exact closure in Q(sqrt5) quaternions; characters derived as characters of V_j restricted
(Chebyshev in Re g) and their Galois conjugates (sqrt5 -> -sqrt5), then proven irreducible and complete
by exact orthonormality, sum of squares = |G| and count = number of classes.
'plant' argument: corrupts one character value to show the checks can fail."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import *

log = Log(__file__, '_plant' if planted() else '')
G = build_group()
log('0(a)')
n1 = sum((x * x for x in GEN1), ZERO); n2 = sum((x * x for x in GEN2), ZERO)
log('|q1|^2 =', n1, ' |q2|^2 =', n2)
log.check('|q1| = |q2| = 1 exactly', n1 == ONE and n2 == ONE)
log('order of Gamma0 =', len(G))
# homomorphism check of quaternion -> matrix map
ok = all(mm2(quat_to_U(a), quat_to_U(b)) == quat_to_U(qmul(a, b)) for a in G[:20] for b in G[::7])
log.check('quaternion->SU(2) matrix map is multiplicative on sampled pairs', ok)

# conjugacy classes
keys = {qkey(g): g for g in G}
seen = set(); classes = []
for g in G:
    if qkey(g) in seen: continue
    cl = {}
    for h in G:
        c = qmul(qmul(h, g), qinv(h)); cl[qkey(c)] = c
    seen |= set(cl)
    classes.append(list(cl.values()))
classes.sort(key=lambda c: (-float(c[0][0].tofloat().real), len(c)))
def order(g):
    E = (ONE, ZERO, ZERO, ZERO); x = g; n = 1
    while qkey(x) != qkey(E):
        x = qmul(x, g); n += 1
    return n
cinfo = []
for c in classes:
    rep = c[0]
    cinfo.append(dict(size=len(c), order=order(rep), re=rep[0]))
log('number of classes =', len(classes))
for i, c in enumerate(cinfo):
    log(f'  class {i}: size {c["size"]}, element order {c["order"]}, Re(g) = {c["re"]}')
log.check('class sizes sum to |G|', sum(c['size'] for c in cinfo) == len(G))


def chiV(n, re):
    """character of V_{n/2} at an element with real part re: U_n(cos theta), recursion in t = 2 re"""
    t = re * 2
    a, b = ONE, t
    if n == 0: return a
    for _ in range(n - 1):
        a, b = b, t * b - a
    return b


def ip(x, y):
    return sum((K(c['size']) * a * b.conj() for c, a, b in zip(cinfo, x, y)), ZERO) * Fr(1, len(G))

cands = []
for n in range(0, 6):
    ch = [chiV(n, c['re']) for c in cinfo]
    cands.append((f'V_{n}/2', ch))
    gal = [x.galois() for x in ch]
    if gal != ch:
        cands.append((f'gal(V_{n}/2)', gal))
irr = []
def add_if_irr(name, ch):
    if ip(ch, ch) == ONE and ch[0].israt() and ch[0].a > 0 and all(not (ip(ch, c2) == ONE) for _, c2 in irr):
        irr.append((name, ch))
for name, ch in cands:
    add_if_irr(name, ch)
# residuals: V_{n/2} minus its components on the irreducibles found so far; a residual of norm 1 with
# positive degree is a new irreducible (it is a virtual character of norm 1 and positive degree).
for n in range(6, 13):
    ch = [chiV(n, c['re']) for c in cinfo]
    res = list(ch)
    for _, c2 in irr:
        m = ip(ch, c2)
        res = [a - m * b for a, b in zip(res, c2)]
    add_if_irr(f'V_{n}/2-known', res)
if planted():
    irr[1] = (irr[1][0], [irr[1][1][0]] + [irr[1][1][1].galois()] + irr[1][1][2:])
log('irreducible characters (columns = classes above):')
for name, ch in irr:
    log(f'  {name:12s} dim {ch[0]}: ', [str(x) for x in ch])
gram_ok = all(ip(a, b) == (ONE if ia == ib else ZERO) for ia, (_, a) in enumerate(irr) for ib, (_, b) in enumerate(irr))
log.check('exact orthonormality of the derived characters', gram_ok)
log.check('number of irreducibles = number of classes', len(irr) == len(classes))
log.check('sum of dim^2 = |G|', sum((ch[0] * ch[0] for _, ch in irr), ZERO) == K(len(G)))
# column orthogonality
colok = all(sum((a[i] * a[k].conj() for _, a in irr), ZERO) == (K(Fr(len(G), cinfo[i]['size'])) if i == k else ZERO)
            for i in range(len(cinfo)) for k in range(len(cinfo)))
log.check('column orthogonality', colok)

log('\n0(b) multiplicities of each irreducible in V_{l/2}, l = 0..20')
mult = {}
for name, ch in irr:
    row = []
    for l in range(21):
        m = ip([chiV(l, c['re']) for c in cinfo], ch)
        if not (m.israt() and m.a.denominator == 1):
            log.check(f'multiplicity of {name} in V_{l}/2 is an integer', False, str(m)); row.append(-999); continue
        row.append(int(m.a))
    mult[name] = row
    occ = [(l, row[l]) for l in range(21) if row[l]]
    log(f'  {name:12s} dim {ch[0]}: levels (l, mult): {occ}')
log.check('every multiplicity is a nonnegative integer and dims add up',
          all(sum(int(irr[k][1][0].a) * mult[irr[k][0]][l] for k in range(len(irr))) == l + 1 for l in range(21)))

# slot identification
slots = {}
for sname, dim, n in (('R2', 2, 7), ('R4', 3, 6), ('R5', 4, 6)):
    hits = [name for name, ch in irr if ch[0] == K(dim) and min(l for l in range(21) if mult[name][l]) == n]
    log(f'slot {sname}: dim {dim}, lowest level {n} -> {hits}')
    log.check(f'{sname} picks out exactly one irreducible', len(hits) == 1)
    slots[sname] = hits[0] if hits else None
if not planted():
    save_json('group.json', dict(
        classes=[dict(size=c['size'], order=c['order'], re=[str(c['re'].a), str(c['re'].b)]) for c in cinfo],
        irr={name: [[str(x.a), str(x.b)] for x in ch] for name, ch in irr},
        mult=mult, slots=slots))
log.close()
