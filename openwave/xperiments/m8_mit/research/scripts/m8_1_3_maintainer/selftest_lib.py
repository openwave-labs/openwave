"""Instrument check for lib.py: its slow-flow reader must return the claims file's own words from the
claims file's own polynomials (B1, B2, B3), and the comparisons must go red on planted defects."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
lib.load_claims(sys.argv[1])
C = lib.CLAIMS
n = bad = 0
for gid in ('B1', 'B2'):
    for pt, v in C[gid]['value'].items():
        sc = v['script']; key = [k for k in sc if k.startswith('slow polynomial')][0]
        w = lib.slow_words(sc[key]); pdf = v['slow flow (PDF)']
        if pt == 'v1': continue
        n += 1; ok = pdf.startswith(w)
        bad += not ok
        print(gid, pt, '|', w, '|', pdf, '|', 'ok' if ok else 'MISMATCH')
# scaling: R5 monic = R4 monic at 256 nu/81
for pt in C['B1']['value']:
    a = C['B1']['value'][pt]['script']['slow polynomial in nu, unit u']
    b = C['B2']['value'][pt]['script']['slow polynomial in nu, unit u, by the pinned routine']
    import sympy as sp
    ok = lib.poly_monic(b) == lib.scale_poly(a, sp.Rational(256, 81))
    n += 1; bad += not ok; print('scaling', pt, ok)
# mutations: each must go red
muts = [lib.eq_rational('28/39', '28/38')[0], lib.eq_sig([5, 0, 4], [4, 0, 5])[0],
        lib.eq_poly_monic('(nu+1)*(nu+2)', '(nu+1)*(nu+3)')[0],
        lib.inside_digits(['215.7757926', '215.77579261'], '215.7757925')[0],
        lib.contains(['238.01', '238.015'], '28800/121')[0],
        lib.slow_words('(nu+1)*(nu-2)') == 'elliptic, simple']
print('mutations red:', [not m for m in muts])
print(f'{n} checks, {bad} mismatches; mutations', 'ALL RED' if not any(muts) else 'SOME GREEN')
