"""Exact comparison of solver_cert's reduced quartic h (quartic_R4.json, quartic_R5.json, used for every Block D / B.4
number) with my own exact quartic (stage-1 hpoly.h_poly, built from my S^3 route).  Also compares the even-J channel
coefficients q_J.  Argument 'plant': perturb one of cert's coefficients by 1/10^30 -> must FAIL."""
import sys, os, json
sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
if len(sys.argv) > 1:
    os.environ['PLANT'] = sys.argv[1]
from fractions import Fraction as Fr
from hpoly import h_poly, q_channels
from chk import check, planted

for slot in ('R4', 'R5'):
    d = json.load(open(os.path.join(here, '..', 'solver_cert', f'quartic_{slot}.json')))
    theirs = {tuple(e): Fr(c) for e, c in d['poly']}
    theirs = {e: c for e, c in theirs.items() if c != 0}
    if planted('plant') and slot == 'R5':
        e0 = sorted(theirs)[0]; theirs[e0] += Fr(1, 10 ** 30)
    mine = h_poly(slot)
    check(f'{slot}: cert quartic == my quartic, coefficient by coefficient (exact)', theirs == mine,
          f'{len(theirs)} vs {len(mine)} monomials')
    q, _ = q_channels(slot)
    ok = all(Fr(d['q_J'][str(int(J))]) == v for J, v in q.items())
    check(f'{slot}: cert even-J q_J == mine (exact)', ok, str({str(J): str(v) for J, v in q.items()}))
