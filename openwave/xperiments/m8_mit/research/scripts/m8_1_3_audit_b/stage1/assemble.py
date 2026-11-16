"""Assemble own.json from the per-script outputs (out_*.json): keyed by item, then slot, then point/box name as inputs.json spells it."""
from chk import load_json, save_json, check

o0 = load_json('out_0.json'); o1 = load_json('out_C1C2.json'); o3 = load_json('out_C3.json')
o4 = load_json('out_C4.json'); o5 = load_json('out_C5.json'); oD = load_json('out_D_B4.json')
own = {}
for k in ['0(a)', '0(b)', '0(c)', '0(d)', '0(e)']:
    own[k] = o0[k]
own['0(e)'] = {'R2': o0['0(e)']}
own['C.1'] = o1['C.1']
own['C.2'] = o1['C.2']
own['C.3'] = o3['C.3']
own['C.4'] = o4['C.4']
own['C.5'] = o5['C.5']
own['D'] = oD['D']
b4 = oD['B.4']
ratios = b4.pop('ratio R5/R4 (clause 3 pairing; same ratio for +xi and -xi)')
b4['R5']['O3 R5']['ratio xi_R5/xi_R4 (clause 3 pairing, pairs in increasing nu; same for +xi and -xi)'] = ratios
own['B.4'] = b4
for slot in ['R4', 'R5']:
    for item in ['C.3', 'C.4']:
        own[item].setdefault(slot, {})
missing = []
expect = [('C.4', 'R4', 'pentagonal pyramid'), ('C.4', 'R4', 'coherent v3'), ('C.4', 'R5', 'coherent v3'), ('C.4', 'R2', 'weight 7/2'),
          ('C.4', 'R4', 'octahedron'), ('C.4', 'R4', 'O3 R4'), ('C.5', 'R4', 'coherent v3'), ('C.5', 'R2', 'weight 7/2'),
          ('C.3', 'R4', 'v1'), ('C.3', 'R5', 'v1'), ('C.3', 'R2', 'line {5/2, -3/2}'), ('D', 'R4', 'O3 R4'), ('D', 'R5', 'O3 R5'),
          ('B.4', 'R4', 'O3 R4'), ('B.4', 'R5', 'O3 R5')]
for it, sl, pt in expect:
    if pt not in own.get(it, {}).get(sl, {}):
        missing.append((it, sl, pt))
check('assemble: every expected (item, slot, point) entry is present in own.json', not missing, str(missing))
save_json('own.json', own)
print('own.json written with items', list(own))
