"""M8.1.3 adjudication helpers: exact comparisons against the frozen claims file.

Every comparison returns (ok: bool, detail: str). Nothing here reads a room's value by assumption:
the adapters extract, these functions compare, and the two mutation flags of compare.py show the comparison can fail.
"""
import json, re
import sympy as sp

nu = sp.Symbol('nu')
CLAIMS = None


def load_claims(path):
    global CLAIMS
    d = json.load(open(path, encoding='utf-8'))
    CLAIMS = {c['id']: c for c in d['claims']}
    return d


def Q(x):
    """parse a rational / radical string or number to a sympy number"""
    if isinstance(x, (int,)):
        return sp.Integer(x)
    s = str(x).strip().replace('^', '**').replace('I', 'I')
    return sp.nsimplify(sp.sympify(s), rational=False) if 'sqrt' in s else sp.Rational(sp.sympify(s))


def eq_rational(a, b):
    try:
        A, B = sp.sympify(str(a).replace('^', '**')), sp.sympify(str(b).replace('^', '**'))
    except Exception as e:
        return False, f'unparsable: {a!r} vs {b!r} ({e})'
    ok = sp.simplify(A - B) == 0
    return ok, f'{A} vs {B}'


def poly_monic(s, var=nu):
    """a polynomial string (possibly with a rational prefactor) in `var` -> monic sympy Poly over QQ"""
    s = str(s).replace('^', '**')
    e = sp.sympify(s, locals={'nu': var, 'ν': var})
    P = sp.Poly(sp.expand(e), var, domain='QQ')
    return P.monic()


def eq_poly_monic(a, b):
    try:
        A, B = poly_monic(a), poly_monic(b)
    except Exception as e:
        return False, f'unparsable: {e}'
    return A == B, f'deg {A.degree()} vs {B.degree()}'


def scale_poly(s, factor):
    """P(nu) -> P(factor*nu), monic"""
    P = poly_monic(s)
    return sp.Poly(P.as_expr().subs(nu, factor * nu), nu, domain='QQ').monic()


def slow_words(s):
    """the claims file's slow-flow words, read from a slow polynomial in nu (zero roots removed)"""
    P = poly_monic(s)
    roots = sp.roots(P, nu, multiple=False)
    if sum(roots.values()) != P.degree():
        # fall back to real-root isolation plus counting non-real roots
        rr = sp.Poly(P).intervals()
    real_neg = all(r.is_real and r < 0 for r in roots)
    if real_neg:
        mults = sorted(roots.values())
        if all(m == 1 for m in mults):
            return 'elliptic, simple'
        if mults.count(2) == 1 and all(m in (1, 2) for m in mults):
            return 'elliptic, one double frequency'
        return 'elliptic, other'
    nonreal = {r: m for r, m in roots.items() if not r.is_real}
    pos = {r: m for r, m in roots.items() if r.is_real and r > 0}
    if not nonreal and sum(pos.values()) == 1:
        return 'hyperbolic, one real pair'
    if not nonreal and sum(pos.values()) >= 2:
        return 'hyperbolic, real pairs'
    if not pos and len(nonreal) == 2 and all(m == 1 for m in nonreal.values()):
        return 'hyperbolic, a complex quartet'
    return 'hyperbolic, other'


def eq_sig(a, b):
    a = [int(x) for x in a]; b = [int(x) for x in b]
    return a == b, f'{a} vs {b}'


def eq_int(a, b):
    try:
        return int(a) == int(b), f'{a} vs {b}'
    except Exception as e:
        return False, f'unparsable {a!r} vs {b!r}'


def inside_digits(enc, printed):
    """enc = [lo, hi] decimal strings; printed e.g. '215.7757925' (the ellipsis dropped):
    the enclosure lies in [printed, printed + 1 ulp of the last printed digit)"""
    import mpmath as mp
    mp.mp.dps = 120
    lo, hi = mp.mpf(enc[0]), mp.mpf(enc[1])
    p = printed.rstrip('.').rstrip('…')
    nd = len(p.split('.')[1])
    a = mp.mpf(p); b = a + mp.mpf(10) ** (-nd)
    ok = (lo >= a) and (hi < b) and (lo <= hi)
    return ok, f'[{mp.nstr(lo, 20)}, {mp.nstr(hi, 20)}] in [{p}, {mp.nstr(b, 20)}), width {mp.nstr(hi - lo, 3)}'


def contains(enc, rational):
    import mpmath as mp
    mp.mp.dps = 120
    lo, hi = mp.mpf(enc[0]), mp.mpf(enc[1])
    r = sp.Rational(rational)
    x = mp.mpf(r.p) / r.q
    return lo <= x <= hi, f'{rational} in [{mp.nstr(lo, 25)}, {mp.nstr(hi, 25)}]'
