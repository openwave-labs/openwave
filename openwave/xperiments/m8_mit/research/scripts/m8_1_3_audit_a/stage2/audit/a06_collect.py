"""Collect verdicts into ../audit.json, keyed by item, slot and point as inputs.json spells them.
Reads only this audit's outputs (census.json, r2.json, rest.json) and the a01 log."""
from au import *

cen = load('census.json'); r2 = load('r2.json'); rest = load('rest.json')
a01 = open(os.path.join(HERE, 'a01_anchors.log')).read()
assert '-- 0 FAIL' in a01
inp = inputs()
A = {}
A['0(a)'] = {'l6': 'AGREE', 'r2': 'AGREE', 'rig': 'AGREE', 'rooms with each other': 'AGREE',
             'note': 'order 120, 9 classes (sizes, element orders, Re g), 9 characters: each row equals one of mine, bijectively'}
A['0(b)'] = {'l6': 'AGREE', 'r2': 'AGREE', 'rig': 'AGREE', 'rooms with each other': 'AGREE',
             'note': 'levels <= 20 with multiplicities; R2 = 2 (lowest level 7), R4 = 3 (lowest 6), R5 = 4 (lowest 6) in all rooms'}
A['0(c)'] = {'R2': {'r2': 'AGREE', 'note': "multiplicity 1, s_N = 1/4; r2's eta verified exactly in r2's own stated convention (my D code); rho(q1), rho(q2) reproduced"},
             'R4': {'l6': 'AGREE', 'note': "multiplicity 1, s_N = 3/7; l6's eta verified exactly in l6's convention (it is NOT an intertwiner in my convention: eta is convention-dependent, not a defect)"},
             'R5': {'l6': 'AGREE', 'note': 'multiplicity 1, s_N = 4/7; eta verified exactly in l6 convention'}}
A['0(d)'] = {'l6': 'AGREE', 'r2': 'AGREE', 'rooms with each other': 'AGREE', 'note': '<3 3; 3 -3|6 0> = sqrt(231)/462 > 0 (my lowering route); Theta^2 = +1 on V_3, -1 on V_7/2'}
A['0(e)'] = {'R2': {'r2': 'AGREE', 'note': 'C = [[0,-1],[1,0]] solves conj(rho)C = C rho (solution space one complex line), C conj(C) = -I; map = w -> -Theta w (50-digit check at random SU(2) points, residual 8e-51)'}}
for item in ('A.1', 'A.2'):
    A[item] = {s: {('r2' if s == 'R2' else 'l6'): rest[f'{item}|{s}']} for s in ('R2', 'R4', 'R5')}
A['A.3'] = {'l6': rest['A.3']}
A['A.4'] = {'R2': {}}
for p in inp['R2 points']['points']:
    v = r2[p]['verdicts']
    A['A.4']['R2'][p] = {'r2': 'AGREE' if all(v[k] == 'AGREE' for k in ('A.4 critical', 'A.4 Q', 'A.4/B.3 signature')) else 'DISAGREE',
                         'mine': {'critical': r2[p]['critical'], 'Q': r2[p]['Q'], 'signature': r2[p]['signature']}}
A['A.5'] = {'R2': {k.split('|')[1]: {'r2': v} for k, v in rest.items() if k.startswith('A.5|')}}
A['A.6'] = {s: {('r2' if s == 'R2' else 'l6'): rest[f'A.6|{s}'], 'changes under nearest rounding': rest[f'A.6|{s}|reading-sensitive']} for s in ('R2', 'R4', 'R5')}
for item, slot in (('B.1', 'R4'), ('B.2', 'R5')):
    A[item] = {slot: {}}
    for p in inp['census']['points']:
        R = cen[f'{slot}|{p}']
        v = R['verdicts']
        A[item][slot][p] = {'l6': 'AGREE' if all(x == 'AGREE' for x in v.values()) else 'DISAGREE', 'fields': v,
                            'mine': {k: R[k] for k in ('critical', 'Q', '924 r6hat', 'signature', 'k', 'r_u', 'zero multiplicity',
                                                        'kernel dim of S', 'slow polynomial', 'words')} | {'krein': [(x['nu'], x['sign']) for x in R['krein']]}
                            | ({'ratio +xi_R5/+xi_R4 (all pairs)': R.get('ratio')} if slot == 'R5' else {})}
A['B.3'] = {'R2': {}}
for p in inp['R2 points']['points']:
    v = r2[p]['verdicts']
    mem = {}
    for lab in ('i', 'J', 'k', '(i+J)/sqrt2', '(i+k)/sqrt2'):
        f = {kk.split(' ', 1)[1]: vv for kk, vv in v.items() if kk.startswith(lab + ' ')}
        M = r2[p][lab]
        mem['S_u = i M_hat' if lab == 'i' else 'member ' + lab] = {
            'r2': 'AGREE' if all(x == 'AGREE' for x in f.values()) else 'DISAGREE', 'fields': f,
            'mine': {'zero multiplicity': M['z'], 'kernel dim': M['kernel'], 'slow polynomial rational': M['rational'],
                     'words': M['words'], 'krein': [(x['nu'], x['sign']) for x in M['krein']]}}
    A['B.3']['R2'][p] = {'r2': 'AGREE' if all(x == 'AGREE' for x in v.values()) else 'DISAGREE',
                         'k': r2[p]['k'], 'r_u': r2[p]['r_u'], 'signature': r2[p]['signature'],
                         'under the Pin(2) reading (no continuous Sp(1))': r2[p]['alt'], 'structures': mem}
A['B.3']['R2']['statement in solver_r2/RETURN.md'] = {
    'r2': 'DEFECT (text only; no reported value affected)',
    'note': "'The smallest |eigenvalue| of M-hat there is >= 0.04 in every case' is false: r2's own out_B3_R2.json records 0.012563 "
            "(line {5/2, -3/2}, members (i+J)/sqrt2 and (i+k)/sqrt2, nu = -0.0163), 0.017596 (weight 3/2, the sqrt2 members) and "
            "0.037419 (line {7/2, -5/2}, member k); my independent computation gives the same three margins. The 0.04 bound holds "
            "only for the clusters of S_u = i M_hat."}
A['B.5'] = {'l6': rest['B.5']}
A['E.1'] = {'rig': rest['E.1']}
A['E.2'] = {'rig': 'AGREE' if all(rest[f'E.2|d={d}'] == 'AGREE' for d in range(1, 7)) else 'DISAGREE',
            'per dimension': {f'd={d}': rest[f'E.2|d={d}'] for d in range(1, 7)}}
A['unchecked'] = ['nothing in scope is left ungraded; solvers\' own quadrature/ladder/CG-orthogonality claims were not re-run (running their code is not verification); their reported values were reproduced by my own routes instead']
with open(os.path.join(ROOT, 'audit.json'), 'w') as f:
    json.dump(A, f, indent=1, default=str)
print('wrote audit.json')
bad = [(i, s, p) for i in ('B.1', 'B.2') for s in A[i] for p in A[i][s] if A[i][s][p]['l6'] != 'AGREE']
bad += [('B.3', 'R2', p) for p in A['B.3']['R2'] if p in inp['R2 points']['points'] and A['B.3']['R2'][p]['r2'] != 'AGREE']
print('non-AGREE:', bad)
