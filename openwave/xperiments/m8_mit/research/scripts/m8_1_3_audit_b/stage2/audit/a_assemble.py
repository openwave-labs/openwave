"""Assemble audit.json (top level) from the audit outputs, and compare cb's C.3 with my stage-1 C.3 (own/own.json, read
only) mechanically, including an exact span comparison of K' at R2's line point.  Verdicts: AGREE / DISAGREE / DEFECT /
UNCHECKED.  Argument 'plant': alter cb's gamma_u at v1 R5 -> the C.3 comparison must FAIL."""
import sys, os, json
sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
if len(sys.argv) > 1:
    os.environ['PLANT'] = sys.argv[1]
from fractions import Fraction as Fr
import sympy as sp
from core import mlist
from chk import check, planted

top = os.path.join(here, '..')
L = lambda p: json.load(open(os.path.join(top, p)))
own = L('own/own.json'); ca = L('solver_ca/results.json'); cb = L('solver_cb/results.json'); ce = L('solver_cert/results.json')
aC4 = L('audit/a_C4.json'); aC5 = L('audit/a_C5.json'); aD = L('audit/a_D_all.json'); aC1 = L('audit/a_C1.json')
inp = L('inputs.json')
A = 'AGREE'
out = {'_legend': 'AGREE: solver value reproduced by the auditor own route (route stated in AUDIT.md); DISAGREE; DEFECT: demonstrated '
                  'error; UNCHECKED: not recomputed. Solvers: ca = solver_ca, cb = solver_cb, cert = solver_cert. Check-quality '
                  'remarks are in AUDIT.md, not verdicts.'}

# ---------------- anchors (a_anchors.log: 63 PASS, 0 FAIL)
out['0(a)'] = {'ca': A, 'cb': A, 'cert': A, 'mine': 'order 120, 9 classes, character table as own/OWN.md'}
out['0(b)'] = {'ca': A, 'cb': A, 'cert': A, 'mine': 'R2 = dim-2 irrep with levels 7,13,17,19; R4 = 3-dim 6,10,14,16,18,20; R5 = 4-dim 6,8,12,14,16,18(x2),20'}
out['0(c)'] = {s: {k: {'verdict': A} for k in (['ca', 'cb'] + (['cert'] if s != 'R2' else []))} for s in ('R2', 'R4', 'R5')}
for s, v in (('R2', '1/4'), ('R4', '3/7'), ('R5', '4/7')):
    out['0(c)'][s]['mine'] = {'multiplicity': 1, 's_N': v}
    for k in out['0(c)'][s]:
        if k != 'mine':
            out['0(c)'][s][k]['note'] = ('eta is convention-dependent; cb and cert eta give exactly my projector (same D^j); '
                                         "ca eta verified in ca's own convention, rebuilt by me") if k != 'mine' else ''
out['0(d)'] = {'ca': A, 'cb': A, 'cert': A, 'mine': '<3 3;3 -3|6 0> = +1/sqrt(924); Theta^2 = +1 on V_3, -1 on V_7/2'}
out['0(e)'] = {'R2': {'ca': A, 'cb': A, 'mine': 'C conj(C) = -1; map = Theta up to the phase of C',
                      'note': 'ca: phase chosen so map = Theta; cb: C = [[0,1],[-1,0]] gives -Theta; mine: e^{i alpha} Theta. Same statement.'}}
# ---------------- C.1, C.2
out['C.1'] = {}
for s in ('R2', 'R4', 'R5'):
    out['C.1'][s] = {'ca': {'verdict': A, 'basis': "level norms recomputed at ca's own y (orthonormal components) by my CG route, |diff| <= 4e-62; nonzero levels equal"},
                     'cb': {'verdict': A, 'basis': "level norms recomputed at cb's own y by my exact S^3 route: exactly equal at every L <= 20"},
                     'mine (own y, stage 1)': {'nonzero levels': own['C.1'][s]['nonzero levels']}}
out['C.2'] = {s: {'ca': A, 'cb': A, 'mine': own['C.2'][s]} for s in ('R2', 'R4', 'R5')}

# ---------------- C.3: cb vs my stage-1 values
o3 = own['C.3']; c3 = cb['C.3']
ok = []
r2o = o3['R2']['line {5/2, -3/2}']; r2c = c3['R2']['line {5/2, -3/2}']
ok.append(check('C.3 R2 line: |Gamma_u|, dim K\', n_-, k, signature, abs. irreducible equal',
                (r2o['|Gamma_u|'], r2o["dim K'"], r2o['n_-'], r2o['k'], r2o['signature'], r2o["Gamma_u acts on K' absolutely irreducibly"]) ==
                (r2c['|Gamma_u|'], r2c["dim K'"], r2c['n_-'], r2c['k'], r2c['signature'], r2c['absolutely irreducible'])))
ok.append(check('C.3 R2 line: gamma_u exactly equal', Fr(r2o['gamma_u (number, V=R=1)']) == Fr(r2c['gamma_u'].split()[0])))
# K' spans, exact: cb's e1, e2(1), e2(i) in my chart (Re p_7/2..Re p_-7/2, Im ...)
ms = mlist(7); idx = {m: i for i, m in enumerate(ms)}
def vec(entries):
    v = [0] * 16
    for (m, part), val in entries.items():
        v[idx[Fr(m)] + (8 if part == 'im' else 0)] = val
    return v
cbK = [vec({('1/2', 're'): 13, ('-7/2', 're'): 1}),
       vec({('7/2', 're'): 3, ('-1/2', 're'): -1, ('3/2', 're'): 15, ('-5/2', 're'): -5}),
       vec({('7/2', 'im'): 3, ('-1/2', 'im'): -1, ('3/2', 'im'): -15, ('-5/2', 'im'): 5})]
myK = [[sp.Rational(x) for x in row] for row in r2o["K' basis (chart x = (Re p_7/2..Re p_-7/2, Im p_7/2..Im p_-7/2))"]]
Mm = sp.Matrix(myK); Mc = sp.Matrix(cbK)
ok.append(check("C.3 R2 line: cb's exact K' spans the same 3-dim real space as my K' (exact ranks 3, 3, 3)",
                Mm.rank() == 3 and Mc.rank() == 3 and sp.Matrix.vstack(Mm, Mc).rank() == 3))
for s in ('R4', 'R5'):
    a = o3[s]['v1']; b = c3[s]['v1']
    gb = Fr(b['gamma_u'].split()[0])
    if planted('plant') and s == 'R5':
        gb = -gb
    ok.append(check(f'C.3 {s} v1: k, r_u, signature, dim K\', weight, zero multiplicity off K\' equal',
                    (a['k'], a['r_u'], a['signature'], a["dim K'"], int(a['weight under the circle']), a["zero multiplicity off K'"]) ==
                    (b['k'], b['r_u'], b['signature'], b["dim K'"], b["K' weight under the circle"][0], b["off K': zero multiplicity of S^"])))
    ok.append(check(f'C.3 {s} v1: gamma_u exactly equal', Fr(a['gamma_u (number, V=R=1)']) == gb, f"{a['gamma_u (number, V=R=1)']} vs {gb}"))
    nu = sp.Symbol('nu')
    ok.append(check(f'C.3 {s} v1: slow polynomial off K\' identical (exact)',
                    sp.expand(sp.sympify(a["slow polynomial off K'"]) - sp.sympify(b["off K': slow polynomial"])) == 0))
out['C.3'] = {
    'R2': {'line {5/2, -3/2}': {'cb': {'verdict': A, 'items': {'|Gamma_u|': 24, "K'": 'dim 3, same span (exact)', 'absolutely irreducible': True,
                                                               'n_-': 0, 'gamma_u': '115020/20333173 R^2'},
                                       'caveat': '|Gamma_u| = 24 is numerical in both rooms (search + verification at 50 digits); cb closes completeness by maximality of O among finite subgroups of SO(3); not certified exactly by anyone'},
                                'reading': "Sp(1) reading (as stated). Under the literal reading (U(1) and J only) k = 4, signature [0,5,6], dim K' = 5 (audit/a_reading.log): C.3 here would change."}},
    'R4': {'v1': {'cb': {'verdict': A, 'items': {"K'": 'C v_-3 (real dim 2)', 'circle weight': 4, 'gamma_u': '-19919/6697728 R^2', 'zero multiplicity off K\'': 4,
                                                 'slow polynomial off K\'': '(20449*nu + 784)*(20449*nu + 12544)*(1656369*nu + 19600)*(1656369*nu + 1960000)', 'k': 3, 'r_u': 1}}}},
    'R5': {'v1': {'cb': {'verdict': A, 'items': {"K'": 'C v_-3 (real dim 2)', 'circle weight': 4, 'gamma_u': '224487/41628563 R^2', 'zero multiplicity off K\'': 4,
                                                 'slow polynomial off K\'': '(20449*nu + 3969)*(81796*nu + 30625)*(327184*nu + 1225)*(327184*nu + 3969)', 'k': 3, 'r_u': 1}}}}}
if not all(ok):
    for s in out['C.3']:
        for p in out['C.3'][s]:
            out['C.3'][s][p]['cb']['verdict'] = 'see audit/a_assemble.log'

# ---------------- C.4
out['C.4'] = {}
for s in ('R2', 'R4', 'R5'):
    out['C.4'][s] = {}
    for p, rec in aC4[s].items():
        if p.startswith('O'):
            out['C.4'][s][p] = {'ca': A, 'ca sign': ca['C.4'][s][p]['sign'], 'my sign': rec['sign'], 'my enclosure (interval CG route over the Section-5 box)': rec['enclosure'],
                                'ca enclosure': ca['C.4'][s][p]['R0~ enclosure over the box']}
        else:
            e = {'ca': A, 'ca value': rec['ca exact'], 'my sign': rec['sign'], 'my CG route (60 digits)': rec['R0 CG route (60 digits)'], 'critical (my exact check)': rec['critical (exact)']}
            if 'R0 exact (my S^3 route)' in rec:
                e['my exact value'] = rec['R0 exact (my S^3 route)']
            out['C.4'][s][p] = e
out['C.4']['R4']['pentagonal pyramid']['my exact value'] = own['C.4']['R4']['pentagonal pyramid']['R0(u) (V=R=1)']

# ---------------- C.5
out['C.5'] = {}
for case, rec in aC5.items():
    pname, slot = case.rsplit(' in ', 1)
    out['C.5'].setdefault(slot, {})[pname] = {'cb': A, 'my zero multiplicity (exact charpoly)': rec['zero multiplicity (exact)'],
                                              'my dim ker Shat': rec['dim ker Shat (geometric multiplicity of 0)'],
                                              'my slow polynomial': rec['slow polynomial'],
                                              'my groups': [{'s_g': g['s_g'], 'multiplicity': g['multiplicity'], 'values': g['values']} for g in rec['groups']]}

# ---------------- D and B.4
out['D'] = {}; out['B.4'] = {}
for s in ('R4', 'R5'):
    out['D'][s] = {}; out['B.4'][s] = {}
    for o in ('O1', 'O2', 'O3'):
        key = f'{o} {s}'
        m = aD['D'][s][key]
        out['D'][s][key] = {'cert': A, 'ca (Krawczyk test, used for C.4)': A,
                            'mine': {'signature': m['signature'], 'k': m['k'], 'r_u': m['r_u'], 'zero multiplicity': m['zero multiplicity'],
                                     '924 r6hat': m['924 r6hat'], 'slow roots': [t['enclosure'] for t in m['slow roots (nu)']],
                                     'signs': [t['sign'] for t in m['slow roots (nu)']]},
                            'Krawczyk on the Section-3 box': {'cert': 'yes (outward box containing the exact box; K(Z) checked inside the exact box, margin 1e-150)',
                                                              'ca': 'yes (inward-rounded box inside the exact box)', 'mine': 'yes (inner box)'},
                            'enclosures outward-rounded': {'cert': 'yes in interval arithmetic; decimal strings by floor/ceil of a round-to-nearest product (not directed rounding; see AUDIT.md)'},
                            'certified point in the given box': {'cert': 'yes (K(Z) inside the exact box)'}}
        b = aD['B.4'][s][key]
        out['B.4'][s][key] = {'cert': A, 'mine': {'slow-flow words': b['slow-flow words'], 'ratios R5/R4 (all pairs)': '9/16 enclosed, width <= 6.3e-98'}}
if os.environ.get('PLANT'):
    print('planted run: audit.json not written')
else:
    json.dump(out, open(os.path.join(top, 'audit.json'), 'w'), indent=1)
    print('audit.json written')
