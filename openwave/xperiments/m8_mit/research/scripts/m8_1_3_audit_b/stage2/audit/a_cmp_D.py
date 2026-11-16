"""Compare solver_cert's Block D / B.4 enclosures with my own (audit/a_D_all.json, produced by my code).
Two enclosures of the same true quantity must intersect; a PASS here is a non-empty intersection, which fails when
the two enclosures are disjoint (a planted defect shifts one of cert's bounds by 1e-60 and must FAIL)."""
import sys, os, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) > 1:
    os.environ['PLANT'] = sys.argv[1]
from fractions import Fraction as Fr
from decimal import Decimal
from chk import check

here = os.path.dirname(os.path.abspath(__file__))
mine = json.load(open(os.path.join(here, 'a_D_all.json')))
cert = json.load(open(os.path.join(here, '..', 'solver_cert', 'results.json')))
PL = os.environ.get('PLANT', '')


def F(s):
    return s if isinstance(s, Fr) else Fr(Decimal(s))


def meet(a, b):
    lo = max(F(a[0]), F(b[0])); hi = min(F(a[1]), F(b[1]))
    return lo <= hi


for slot in ['R4', 'R5']:
    for o in ['O1', 'O2', 'O3']:
        key = f'{o} {slot}'
        m = mine['D'][slot][key]; c = cert['D'][slot][key]
        check(f'{key}: signature equal', m['signature'] == c['signature'], f"{m['signature']} vs {c['signature']}")
        check(f'{key}: k, r_u, zero multiplicity equal', (m['k'], m['r_u'], m['zero multiplicity']) == (c['k'], c['r_u'], c['zero multiplicity']))
        cr6 = list(c['924 r6'])
        if PL == 'shift' and key == 'O2 R5':
            cr6 = [F(cr6[0]) + Fr(1, 10**50), F(cr6[1]) + Fr(1, 10**50)]
        check(f'{key}: 924 r6hat enclosures intersect', meet(m['924 r6hat'], cr6), f"{m['924 r6hat'][0][:40]} / {str(cr6[0])[:40]}")
        check(f'{key}: lambda enclosures intersect', meet(m['lambda enclosure'], c['lambda']))
        mr = [t['enclosure'] for t in m['slow roots (nu)']]
        cr = c['slow roots (nu)']
        check(f'{key}: all 4 slow-root enclosures intersect pairwise in sorted order', len(mr) == len(cr) == 4 and all(meet(a, b) for a, b in zip(mr, cr)))
        check(f'{key}: slow-root signs equal', [t['sign'] for t in m['slow roots (nu)']] == c['slow root signs'])
        cw = cert['B.4'][slot][key]['slow-flow words']
        mw = mine['B.4'][slot][key]['slow-flow words']
        check(f'{key}: slow-flow words equal', cw == mw, f'{mw} / {cw}')
    # widths of cert's printed enclosures
for slot in ['R4', 'R5']:
    for o in ['O1', 'O2', 'O3']:
        key = f'{o} {slot}'
        c = cert['B.4'][slot][key]
        ws = [F(r[1]) - F(r[0]) for r in c['slow roots (nu), sorted']] + [F(c['924 r6'][1]) - F(c['924 r6'][0])]
        check(f'{key}: cert printed B.4 enclosures have width <= 1e-50 (as decimals)', max(ws) <= Fr(1, 10**50), f'max {float(max(ws)):.2e}')
        for rr in c['eigenvalue ratios R5/R4 (clause 3 pairing)']:
            q = rr['ratio +xi(R5)/+xi(R4) = -xi(R5)/-xi(R4)']
            check(f'{key}: cert ratio pair {rr["pair"]} contains 9/16 and width <= 1e-50', F(q[0]) <= Fr(9, 16) <= F(q[1]) and F(q[1]) - F(q[0]) <= Fr(1, 10**50))
