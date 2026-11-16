"""Mechanical comparison of solver_cb's C.5 returns with mine (audit/a_C5.json, from my code): zero multiplicity, k, r_u,
signature, and for every slow group: s_g (matched to 1e-20), multiplicity, and the values (relative 1e-18, i.e. the 20
digits cb prints).  Argument 'plant': negate one of cb's values -> must FAIL."""
import sys, os, json
sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
if len(sys.argv) > 1:
    os.environ['PLANT'] = sys.argv[1]
import sympy as sp
from mpmath import mp
from chk import check, planted

mp.dps = 40
mine = json.load(open(os.path.join(here, 'a_C5.json')))
cb = json.load(open(os.path.join(here, '..', 'solver_cb', 'results.json')))['C.5']


def cz(s):
    v = sp.sympify(str(s).replace('j', '*I') if 'I' not in str(s) else s, locals={'I': sp.I})
    return mp.mpc(str(sp.re(v)), str(sp.im(v)))


def pz(s):   # my mpmath strings like '(a + bj)' or 'a'
    s = s.strip()
    return mp.mpc(complex(0)) + mp.mpmathify(s.replace(' ', '')) if 'j' in s else mp.mpc(mp.mpf(s))


for case, rec in mine.items():
    pname, slot = case.rsplit(' in ', 1)
    c = cb[slot][pname]
    check(f'{case}: zero multiplicity cb {c["zero multiplicity"]} == my exact {rec["zero multiplicity (exact)"]}', c['zero multiplicity'] == rec['zero multiplicity (exact)'])
    check(f'{case}: k, r_u, signature equal', (c['k'], c['r_u'], list(c['signature'])) == (rec['k'], rec['r_u'], rec['signature']),
          f"{(c['k'], c['r_u'], c['signature'])} vs {(rec['k'], rec['r_u'], rec['signature'])}")
    used = set()
    first = True
    for g in rec['groups']:
        s = pz(g['s_g'])
        hits = [i for i, h in enumerate(c['slow groups']) if abs(cz(h['s_g']) - s) < mp.mpf(10) ** -20 and i not in used]
        ok = len(hits) == 1
        if ok:
            h = c['slow groups'][hits[0]]; used.add(hits[0])
            vals = h.get('c_g') or h.get('eigenvalues of C_ab')
            cv = sorted([cz(v) for v in vals], key=lambda z: (float(mp.re(z)), float(mp.im(z))))
            if planted('plant') and first:
                cv[0] = -cv[0]; first = False
            mv = sorted([pz(v) for v in g['values']], key=lambda z: (float(mp.re(z)), float(mp.im(z))))
            dv = max(abs(a - b) / abs(b) for a, b in zip(cv, mv))
            ok = h['multiplicity'] == g['multiplicity'] and len(cv) == len(mv) and dv < mp.mpf(10) ** -18
            check(f'{case}: s_g = {mp.nstr(s, 10)}: cb multiplicity and values equal mine (rel {mp.nstr(dv, 3)})', ok)
        else:
            check(f'{case}: s_g = {mp.nstr(s, 10)} found once in cb', False)
    check(f'{case}: every cb group matched', len(used) == len(c['slow groups']) == len(rec['groups']))
