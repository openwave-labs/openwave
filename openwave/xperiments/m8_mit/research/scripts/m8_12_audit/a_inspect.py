"""(1) Exact positive-definiteness of solver A's stored reduced Gram blocks (their data, my check).
(2) Look at how each solver recorded zeros / residuals in its own output files (computed vs absent)."""
import json
import sympy as sp

d = json.load(open('solver_a/out/item8.json'))
blocks = d['max']['Yt_blocks']
allpd = True
for k, M in blocks.items():
    Mm = sp.Matrix([[sp.Rational(x) for x in row] for row in M])
    pd = Mm.is_positive_definite and Mm == Mm.T
    allpd &= pd
    print('A block %-10s size %d  symmetric PD (exact, my check): %s' % (k, Mm.rows, pd))
print('all of A\'s stored blocks PD:', allpd, '  number of stored scalar unknowns:',
      sum(len(M) * (len(M) + 1) // 2 for M in blocks.values()))

def walk(x, path=''):
    if isinstance(x, dict):
        for k, v in x.items(): yield from walk(v, path + '/' + str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from walk(v, path + '[%d]' % i)
    else:
        yield path, x
for solver, files in (('solver_a', ['item5', 'item9', 'item5b']), ('solver_b', ['item5', 'item7', 'item5b'])):
    for f in files:
        try:
            dd = json.load(open('%s/out/%s.json' % (solver, f)))
        except Exception as ex:
            print(solver, f, 'unreadable', ex); continue
        hits = [(p, v) for p, v in walk(dd) if any(s in p.lower() for s in ('resid', 'annih', 'null', 'zero', 'kernel', 'offdiag', 'off_diag'))]
        print('%s/%s: %d residual/zero-type fields; first few:' % (solver, f, len(hits)))
        for p, v in hits[:8]: print('    ', p, '=', str(v)[:80])
