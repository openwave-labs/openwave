"""Merge out/item*.json into results.json (exact values as strings)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
names = ['item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item5b', 'item8', 'item9', 'item10', 'item11', 'item12', 'defects']
res = {'conventions': 'basis order v3..v-3; exact values are strings (rationals or sympy expressions); '
                      'items 6 and 7 are stored inside item5 (keys item6, item7_*)'}
for n in names:
    p = os.path.join(OUT, n + '.json')
    res[n] = json.load(open(p)) if os.path.exists(p) else 'MISSING (script did not produce output)'
# a compact headline table
i4 = res['item4']['orbits'] if isinstance(res['item4'], dict) else {}
i5 = res['item5'] if isinstance(res['item5'], dict) else {}
res['headline'] = {
    'cg_33_3m3_60': res['item0'].get('cg_33_3m3_60') if isinstance(res['item0'], dict) else None,
    'rhat6_item0': {k: v['rhat6'] for k, v in res['item0']['points'].items()} if isinstance(res['item0'], dict) else None,
    'orbits': {k: {'rhat6': v['rhat6'], 'dim_N': i5.get(k, {}).get('dim_N'), 'signature': i5.get(k, {}).get('signature_(n-,n0,n+)'),
                   'charpoly': i5.get(k, {}).get('charpoly_monic_factored')} for k, v in i4.items()},
    'global_min': '1/924 at v3 (proved)', 'global_max': '463/924 at (v3+v-3)/sqrt2 (proved)',
}
with open(os.path.join(HERE, 'results.json'), 'w') as f:
    json.dump(res, f, indent=1, default=str)
print('results.json written with', len(res), 'top-level keys')
