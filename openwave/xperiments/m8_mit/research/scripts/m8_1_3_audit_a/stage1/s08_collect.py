"""Assembles own.json from the JSON outputs of s01..s07 (no new computation)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import load_json, save_json, Log

log = Log(__file__)
g = load_json('group.json'); an = load_json('anchors.json'); a1 = load_json('A1.json'); a2 = load_json('A2.json')
B = load_json('B.json'); b5 = load_json('B5.json'); E = load_json('E.json')

def q5(a, b):
    a = a if a != '0' else ''
    if b == '0': return a or '0'
    return (f'{a} + ' if a else '') + f'({b})*sqrt(5)'

own = {}
own['0(a)'] = dict(norms_q1_q2=['1', '1'], order=120,
                   classes=[dict(size=c['size'], element_order=c['order'], re_g=q5(*c['re'])) for c in g['classes']],
                   characters={k: [q5(*x) for x in v] for k, v in g['irr'].items()},
                   source='DERIVED in this room')
own['0(b)'] = {k: {'dim': int(g['irr'][k][0][0]), 'levels': {str(l): m for l, m in enumerate(v) if m}} for k, v in g['mult'].items()}
own['0(b)']['slot names'] = g['slots']
own['0(c)'] = {s: dict(multiplicity=an[s]['multiplicity'], eta=an[s]['eta'], s_N=an[s]['s_N']) for s in ('R2', 'R4', 'R5')}
own['0(d)'] = {'sign <3 3; 3 -3 | 6 0>': an['CG sign'], 'value': 'sqrt(231)/462 = 1/sqrt(924)',
               'Theta^2 on V_3': an['Theta^2 V_6/2'], 'Theta^2 on V_7/2': an['Theta^2 V_7/2']}
own['0(e)'] = {'C': an['R2 C'], 'C Cbar': an['R2 C Cbar'], 'map on w': an['R2 map on w'],
               'note': 'C is fixed only up to a unit scalar; C -> -C gives +Theta'}
own['A.1'] = {s: {f'q_{J}': v for J, v in a1[s].items()} for s in ('R2', 'R4', 'R5')}
own['A.2'] = {s: dict(proportional=a2[s]['proportional'], beta=a2[s]['beta']) for s in ('R2', 'R4', 'R5')}
own['A.3'] = {'beta_R5/beta_R4': a2['A.3']['ratio'], 'beta_R4 - beta_R5': a2['A.3']['difference']}

def bentry(R, with_r6=True, members=False):
    d = dict(critical=R['critical'], Q=R['Q'])
    if with_r6 and '924 rhat_6' in R: d['924 rhat_6'] = R['924 rhat_6']
    for k in ('signature', 'k', 'r_u', 'zero_multiplicity', 'slow_polynomial', 'roots', 'words', 'krein'):
        if k in R: d[k] = R[k]
    if members and 'members' in R:
        d['members'] = R['members']
    return d

own['A.4'] = {p: dict(critical=B[f'R2|{p}']['critical'], Q=B[f'R2|{p}']['Q'], signature=B[f'R2|{p}']['signature'])
              for p in ('line {5/2, -3/2}', 'weight 7/2')}
own['B.1'] = {p: bentry(B[f'R4|{p}']) for p in ('v1', 'pentagonal pyramid', 'D2 ray')}
own['B.2'] = {}
for p in ('v1', 'pentagonal pyramid', 'D2 ray'):
    d = bentry(B[f'R5|{p}'])
    d['ratio_R5_over_R4'] = [dict(nu_R4=x['nu_R4'], nu_R5=x['nu_R5'], ratio=x['exact'] or x['ratio']) for x in B[f'R5|{p}']['ratio_R5_over_R4']]
    own['B.2'][p] = d
own['B.3'] = {p: bentry(B[f'R2|{p}'], with_r6=False, members=True) for p in ('line {5/2, -3/2}', 'weight 7/2')}
own['B.5'] = dict(proportional=b5['proportional'], constant='rhat_6 = 1 * HS6 (identical functions on V_3)')
own['E.1'] = E['E.1']
own['E.2'] = E['E.2']
save_json('own.json', own)
log('own.json written with items:', list(own))
log.close()
