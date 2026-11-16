"""Audit helpers (stage 2). Reuses my own stage-1 library in ../own (read-only import; nothing there is edited).
Nothing here imports or calls a solver's code; solver files are read only as data (results.json)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'own'))
from lib_common import Log, K, Fr, ZERO, ONE, ms, s2, fact, Dw, build_group, quat_to_U, qmul  # noqa: E402,F401
import mpmath, sympy as sp  # noqa: E402


def planted():
    return len(sys.argv) > 1 and sys.argv[1] == 'plant'


def solver(name):
    with open(os.path.join(ROOT, name, 'results.json')) as f:
        return json.load(f)


def own_json(name):
    with open(os.path.join(ROOT, 'own', name)) as f:
        return json.load(f)


def inputs():
    with open(os.path.join(ROOT, 'inputs.json')) as f:
        return json.load(f)


def save(name, obj):
    with open(os.path.join(HERE, name), 'w') as f:
        json.dump(obj, f, indent=1, default=str)


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def sym(s):
    return sp.sympify(str(s).replace('^', '**'), locals={'I': sp.I, 'sqrt': sp.sqrt, 'nu': sp.Symbol('nu')})


def Dnum(U, j2):
    """numeric CS-basis D^j(U) for (f)(v) -> f(U^T v), i.e. x -> U00 x + U10 y, y -> U01 x + U11 y
    (the same substitution rule as own/lib_common.Dw), U a 2x2 mpmath matrix; rows/cols m = j..-j."""
    n = j2 + 1; ml = ms(j2)
    M = mpmath.matrix(n, n)
    for col, m2 in enumerate(ml):
        A = (j2 + m2) // 2; B = (j2 - m2) // 2
        # (U00 x + U10 y)^A (U01 x + U11 y)^B
        for k1 in range(A + 1):
            c1 = mpmath.binomial(A, k1) * U[0, 0] ** k1 * U[1, 0] ** (A - k1)
            for k2 in range(B + 1):
                c2 = mpmath.binomial(B, k2) * U[0, 1] ** k2 * U[1, 1] ** (B - k2)
                xp = k1 + k2
                row = ml.index(2 * xp - j2)
                M[row, col] += c1 * c2
    sv = [mpmath.sqrt(s2(j2, m2)) for m2 in ml]
    for a in range(n):
        for b in range(n):
            M[a, b] = M[a, b] * sv[a] / sv[b]
    return M


def rand_su2(rng):
    a = [mpmath.mpf(rng.gauss(0, 1)) + mpmath.mpf(rng.random()) * mpmath.mpf(10) ** -20 for _ in range(4)]
    nr = mpmath.sqrt(sum(x * x for x in a))
    w, x, y, z = [t / nr for t in a]
    return mpmath.matrix([[mpmath.mpc(w, x), mpmath.mpc(y, z)], [mpmath.mpc(-y, z), mpmath.mpc(w, -x)]])
