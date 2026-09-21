"""Compare each solver's out/<item>.json with the corresponding entry of its results.json."""
import json, os
for solver in ('solver_a', 'solver_b'):
    R = json.load(open(os.path.join(solver, 'results.json')))
    print('==', solver, 'results.json keys:', sorted(R.keys()))
    for f in sorted(os.listdir(os.path.join(solver, 'out'))):
        if not f.endswith('.json'): continue
        key = f[:-5]
        d = json.load(open(os.path.join(solver, 'out', f)))
        if key not in R:
            print('   %-12s not in results.json' % f); continue
        same = json.dumps(d, sort_keys=True) == json.dumps(R[key], sort_keys=True)
        print('   %-12s identical to results.json[%s]: %s' % (f, key, same))
        if not same and isinstance(d, dict):
            for k in d:
                if json.dumps(d[k], sort_keys=True) != json.dumps(R[key].get(k), sort_keys=True):
                    print('        differs at key', k, ':', json.dumps(d[k])[:150], ' VS ', json.dumps(R[key].get(k))[:150])
                    break
