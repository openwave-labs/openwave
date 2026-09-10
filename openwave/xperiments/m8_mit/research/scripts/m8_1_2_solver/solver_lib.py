"""Shared exact machinery for the solver scripts.

Conventions (from handout.md):
  * V_j = Sym^{2j} C^2, orthonormal weight basis v_m = x^{j+m} y^{j-m} / sqrt((j+m)!(j-m)!),
    so J_+ = x d/dy has positive matrix elements sqrt(j(j+1)-m(m+1)) (Condon-Shortley).
  * SU(2) element U acts by x -> U11 x + U21 y, y -> U12 x + U22 y (Sym^{2j} of the defining rep).
  * quaternion w + x i + y j + z k  ->  [[w + x I, y + z I], [-y + z I, w - x I]]  (homomorphism, checked).
  * Clebsch-Gordan: sympy.physics.wigner.clebsch_gordan (Condon-Shortley, real).
  * Theta v_m = (-1)^m v_{-m} (antilinear) on V_3; general spin Theta_j v_m = eps_j (-1)^{j-m} v_{-m}.
  * M_K(P)_N = sum_{n+n'=N} CG(3 n; 3 n' | K N) (-1)^{n'} P_{n,-n'}   (= eps_3 = -1 general form).

Run as ./py solver_lib.py for a self-test of the conventions.
"""
import json
import pathlib
from fractions import Fraction as Fr
from functools import lru_cache

import sympy as sp
from sympy.physics.wigner import clebsch_gordan

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "solver_results.json"
SQ5 = sp.sqrt(5)


# ----------------------------------------------------------------------------------------------
# exact field Q(sqrt5) and Q(sqrt5)(i)
# ----------------------------------------------------------------------------------------------
class Q5:
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = Fr(a)
        self.b = Fr(b)

    def __add__(s, o):
        o = _q5(o)
        return Q5(s.a + o.a, s.b + o.b)

    __radd__ = __add__

    def __neg__(s):
        return Q5(-s.a, -s.b)

    def __sub__(s, o):
        return s + (-_q5(o))

    def __rsub__(s, o):
        return _q5(o) - s

    def __mul__(s, o):
        o = _q5(o)
        return Q5(s.a * o.a + 5 * s.b * o.b, s.a * o.b + s.b * o.a)

    __rmul__ = __mul__

    def inv(s):
        n = s.a * s.a - 5 * s.b * s.b
        return Q5(s.a / n, -s.b / n)

    def __truediv__(s, o):
        return s * _q5(o).inv()

    def __eq__(s, o):
        o = _q5(o)
        return s.a == o.a and s.b == o.b

    def __hash__(s):
        return hash((s.a, s.b))

    def is_zero(s):
        return s.a == 0 and s.b == 0

    def to_sympy(s):
        return sp.Rational(s.a.numerator, s.a.denominator) + sp.Rational(
            s.b.numerator, s.b.denominator) * SQ5

    def __float__(s):
        return float(s.a) + float(s.b) * 5 ** 0.5

    def __repr__(s):
        return f"({s.a}+{s.b}r5)"


def _q5(o):
    return o if isinstance(o, Q5) else Q5(o)


class QC:
    """re + i*im with re, im in Q(sqrt5)."""
    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        self.re = _q5(re)
        self.im = _q5(im)

    def __add__(s, o):
        o = _qc(o)
        return QC(s.re + o.re, s.im + o.im)

    __radd__ = __add__

    def __neg__(s):
        return QC(-s.re, -s.im)

    def __sub__(s, o):
        return s + (-_qc(o))

    def __mul__(s, o):
        o = _qc(o)
        return QC(s.re * o.re - s.im * o.im, s.re * o.im + s.im * o.re)

    __rmul__ = __mul__

    def conj(s):
        return QC(s.re, -s.im)

    def inv(s):
        n = s.re * s.re + s.im * s.im          # in Q(sqrt5), nonzero for nonzero s
        ni = n.inv()
        return QC(s.re * ni, -s.im * ni)

    def __truediv__(s, o):
        return s * _qc(o).inv()

    def __eq__(s, o):
        o = _qc(o)
        return s.re == o.re and s.im == o.im

    def __hash__(s):
        return hash((s.re, s.im))

    def is_zero(s):
        return s.re.is_zero() and s.im.is_zero()

    def to_sympy(s):
        return s.re.to_sympy() + sp.I * s.im.to_sympy()

    def __complex__(s):
        return complex(float(s.re), float(s.im))

    def __repr__(s):
        return f"[{s.re} + i{s.im}]"


def _qc(o):
    return o if isinstance(o, QC) else QC(o)


# ----------------------------------------------------------------------------------------------
# quaternions over Q(sqrt5)
# ----------------------------------------------------------------------------------------------
class Quat:
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w, x, y, z):
        self.w, self.x, self.y, self.z = _q5(w), _q5(x), _q5(y), _q5(z)

    def __mul__(p, q):
        return Quat(
            p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z,
            p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y,
            p.w * q.y - p.x * q.z + p.y * q.w + p.z * q.x,
            p.w * q.z + p.x * q.y - p.y * q.x + p.z * q.w,
        )

    def conj(p):
        return Quat(p.w, -p.x, -p.y, -p.z)

    def norm2(p):
        return p.w * p.w + p.x * p.x + p.y * p.y + p.z * p.z

    def key(p):
        return (p.w, p.x, p.y, p.z)

    def __eq__(p, q):
        return p.key() == q.key()

    def __hash__(p):
        return hash(p.key())

    def su2(p):
        """2x2 matrix [[U11,U12],[U21,U22]] with QC entries."""
        return [[QC(p.w, p.x), QC(p.y, p.z)], [QC(-p.y, p.z), QC(p.w, -p.x)]]

    def __repr__(p):
        return f"Quat({p.w},{p.x},{p.y},{p.z})"


PHI = Q5(Fr(1, 2), Fr(1, 2))          # (1+sqrt5)/2
PHI_INV = Q5(Fr(-1, 2), Fr(1, 2))      # (sqrt5-1)/2 = 1/phi
HALF = Fr(1, 2)
Q1 = Quat(HALF, HALF, HALF, HALF)
Q2 = Quat(PHI * HALF, PHI_INV * HALF, HALF, 0)
ONE = Quat(1, 0, 0, 0)


def generate_group(gens):
    elems = {ONE}
    frontier = [ONE]
    while frontier:
        new = []
        for g in frontier:
            for s in gens:
                for h in (g * s, s * g):
                    if h not in elems:
                        elems.add(h)
                        new.append(h)
        frontier = new
    return list(elems)


def conjugacy_classes(G):
    remaining = set(G)
    classes = []
    while remaining:
        g = next(iter(remaining))
        cls = {h * g * h.conj() for h in G}   # h^{-1} = conj(h) for unit quaternions
        classes.append(sorted(cls, key=lambda q: tuple(float(c) for c in q.key())))
        remaining -= cls
    return classes


def subgroup_generated(G_elems_subset):
    return generate_group(list(G_elems_subset))


# ----------------------------------------------------------------------------------------------
# spin-j representation matrices
# ----------------------------------------------------------------------------------------------
def _poly_mul(p, q):
    r = [QC(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a.is_zero():
            continue
        for k, b in enumerate(q):
            r[i + k] = r[i + k] + a * b
    return r


def _poly_pow(p, e):
    r = [QC(1)]
    for _ in range(e):
        r = _poly_mul(r, p)
    return r


def D_mono(U, n):
    """Spin j = n/2 matrix in the UNnormalised monomial basis e_a = x^a y^(n-a), a = j+m = 0..n.
    Returns list-of-lists M[a'][a] with QC entries: D e_a = sum_a' M[a'][a] e_a'."""
    # x -> U11 x + U21 y ; y -> U12 x + U22 y. Polynomials in x (coefficient index = power of x).
    xp = [U[1][0], U[0][0]]    # U21 + U11 x
    yp = [U[1][1], U[0][1]]    # U22 + U12 x
    M = [[QC(0)] * (n + 1) for _ in range(n + 1)]
    for a in range(n + 1):
        col = _poly_mul(_poly_pow(xp, a), _poly_pow(yp, n - a))
        for ap in range(n + 1):
            M[ap][a] = col[ap]
    return M


def qc_rref(rows):
    """Exact row reduction over Q(sqrt5)(i). rows: list of lists of QC. Returns (rref rows, pivots)."""
    R = [list(r) for r in rows]
    nr, nc = len(R), len(R[0]) if R else 0
    piv = []
    r = 0
    for c in range(nc):
        p = next((i for i in range(r, nr) if not R[i][c].is_zero()), None)
        if p is None:
            continue
        R[r], R[p] = R[p], R[r]
        inv = R[r][c].inv()
        R[r] = [x * inv for x in R[r]]
        for i in range(nr):
            if i != r and not R[i][c].is_zero():
                f = R[i][c]
                R[i] = [x - f * y for x, y in zip(R[i], R[r])]
        piv.append(c)
        r += 1
        if r == nr:
            break
    return R, piv


def qc_rank(rows):
    return len(qc_rref(rows)[1])


def qc_matmul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    return [[sum((A[i][t] * B[t][j] for t in range(k)), QC(0)) for j in range(m)] for i in range(n)]


def qc_eye(n):
    return [[QC(1) if i == j else QC(0) for j in range(n)] for i in range(n)]


def qc_minpoly(Z):
    """Monic minimal polynomial of square QC matrix Z via Krylov dependency of vec(Z^k).
    Returns coefficients [c0, c1, ..., 1] (QC)."""
    n = len(Z)
    powers = [qc_eye(n)]
    while True:
        powers.append(qc_matmul(powers[-1], Z))
        k = len(powers) - 1
        # solve sum_{i<k} c_i vec(Z^i) = -vec(Z^k)
        cols = [[x for row in P for x in row] for P in powers]   # each a vector of length n^2
        # augmented system: rows = coordinates, columns = c_0..c_{k-1} | rhs
        rows = [[cols[i][t] for i in range(k)] + [-cols[k][t]] for t in range(n * n)]
        Rr, piv = qc_rref(rows)
        if k in piv:      # inconsistent -> no dependency yet
            continue
        if len(piv) < k:  # should not happen for first dependency
            raise RuntimeError("non-unique")
        c = [QC(0)] * k
        for i, pc in enumerate(piv):
            c[pc] = Rr[i][k]
        return c + [QC(1)]


def fact_weight(n, a):
    return sp.factorial(a) * sp.factorial(n - a)


def mono_to_orthonormal(M, n):
    """Convert a matrix in the e_a basis to the orthonormal v_m basis (sympy Matrix)."""
    return sp.Matrix(n + 1, n + 1, lambda ap, a: M[ap][a].to_sympy()
                     * sp.sqrt(fact_weight(n, ap) / fact_weight(n, a)))


def D_sym(U_sym, n):
    """Spin j = n/2 matrix in the orthonormal basis, for a sympy 2x2 matrix U_sym."""
    x, y = sp.symbols("x_ y_")
    M = sp.zeros(n + 1, n + 1)
    for a in range(n + 1):
        pol = sp.expand((U_sym[0, 0] * x + U_sym[1, 0] * y) ** a
                        * (U_sym[0, 1] * x + U_sym[1, 1] * y) ** (n - a))
        P = sp.Poly(pol, x, y)
        for ap in range(n + 1):
            c = P.coeff_monomial(x ** ap * y ** (n - ap))
            M[ap, a] = c * sp.sqrt(fact_weight(n, ap) / fact_weight(n, a))
    return M


# ----------------------------------------------------------------------------------------------
# Clebsch-Gordan, Theta, transforms (index conventions: vectors indexed a = j+m, m = a-j)
# ----------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def CG(j1, m1, j2, m2, J, M):
    return sp.nsimplify(clebsch_gordan(sp.S(j1), sp.S(j2), sp.S(J), sp.S(m1), sp.S(m2), sp.S(M)))


def mvals(j):
    j = sp.S(j)
    return [-j + k for k in range(int(2 * j) + 1)]


def theta(u, j=3, eps=None):
    """Theta_j u, u a list indexed by a = j+m. Default at j=3: Section 2.2 with phase +1,
    i.e. Theta v_m = (-1)^m v_{-m}. General: eps_j (-1)^{j-m}."""
    j = sp.S(j)
    ms = mvals(j)
    n = len(ms)
    out = [0] * n
    for a, m in enumerate(ms):
        if eps is None:
            ph = sp.Integer(-1) ** int(m)
        else:
            ph = eps * sp.Integer(-1) ** int(j - m)
        out[int(-m + j)] += ph * sp.conjugate(u[a])
    return out


def bracket(a, ja, b, jb, J):
    """[a (x) b]_J, vector indexed by M = -J..J (index M+J)."""
    ja, jb, J = sp.S(ja), sp.S(jb), sp.S(J)
    out = [sp.S(0)] * int(2 * J + 1)
    for i, m1 in enumerate(mvals(ja)):
        if a[i] == 0:
            continue
        for k, m2 in enumerate(mvals(jb)):
            if b[k] == 0:
                continue
            M = m1 + m2
            if abs(M) > J:
                continue
            c = CG(ja, m1, jb, m2, J, M)
            if c != 0:
                out[int(M + J)] += c * a[i] * b[k]
    return out


def transform(P, K, j=3, eps=None):
    """M^{(j)}_K(P)_N = sum_{n+n'=N} CG(j n; j n'|K N) eps_j (-1)^{j-n'} P_{n,-n'}.
    P: sympy Matrix indexed [a_row][a_col], a = j+m. Default eps: -1 at j=3 (Section 2.3 form)."""
    j = sp.S(j)
    if eps is None:
        eps = -1 if j == 3 else 1
    ms = mvals(j)
    out = [sp.S(0)] * (2 * K + 1)
    for n in ms:
        for npr in ms:
            N = n + npr
            if abs(N) > K:
                continue
            c = CG(j, n, j, npr, K, N)
            if c == 0:
                continue
            out[int(N + K)] += c * eps * sp.Integer(-1) ** int(j - npr) * P[int(n + j), int(-npr + j)]
    return out


def norm2(vec):
    return sp.expand(sum(sp.expand(x * sp.conjugate(x)) for x in vec))


def A_op_matrix(X, K, j=3):
    """Matrix of v -> [X (x) v]_j for X in V_K (the operator A_K with X in the first slot)."""
    j = sp.S(j)
    n = int(2 * j + 1)
    M = sp.zeros(n, n)
    for a in range(n):
        e = [0] * n
        e[a] = 1
        col = bracket(X, K, e, j, j)
        for ap in range(n):
            M[ap, a] = col[ap]
    return M


# ----------------------------------------------------------------------------------------------
# spin-3 multipole maps with u and conj(u) as independent symbol lists (a = u, b = conj u)
# ----------------------------------------------------------------------------------------------
MS3 = list(range(-3, 4))


def rho_ab(a, b, K):
    """rho_K(u)_N = sum CG(3n;3n'|KN) (-1)^{n'} u_n conj(u_{-n'})."""
    o = [sp.S(0)] * (2 * K + 1)
    for n in MS3:
        for npr in MS3:
            N = n + npr
            if abs(N) > K:
                continue
            c = CG(3, n, 3, npr, K, N)
            if c:
                o[N + K] += c * sp.Integer(-1) ** abs(npr) * a[n + 3] * b[-npr + 3]
    return o


def B_ab(a, K):
    """holomorphic square B_K(u)_N = sum CG(3n;3n'|KN) u_n u_n'."""
    o = [sp.S(0)] * (2 * K + 1)
    for n in MS3:
        for npr in MS3:
            N = n + npr
            if abs(N) > K:
                continue
            c = CG(3, n, 3, npr, K, N)
            if c:
                o[N + K] += c * a[n + 3] * a[npr + 3]
    return o


def couple_to_3(X, K, v):
    """[X (x) v]_3 with X in V_K (first slot), v in V_3."""
    o = [sp.S(0)] * 7
    for N in range(-K, K + 1):
        if X[N + K] == 0:
            continue
        for m in MS3:
            M = N + m
            if abs(M) > 3:
                continue
            c = CG(K, N, 3, m, 3, M)
            if c:
                o[M + 3] += c * X[N + K] * v[m + 3]
    return [sp.expand(x) for x in o]


def theta_ab(b):
    """(Theta u)_m = (-1)^m conj(u_{-m}) expressed through b = conj u."""
    return [sp.Integer(-1) ** abs(m) * b[-m + 3] for m in MS3]


def M_ab(a, b, K):
    return couple_to_3(rho_ab(a, b, K), K, a)


def N_ab(a, b, J):
    """[B_J(u) (x) Theta u]_3 (the psi psi^T channel maps of item 21)."""
    return couple_to_3(B_ab(a, J), J, theta_ab(b))


def load_results():
    return json.loads(RESULTS.read_text())


def load_P(key):
    rows = load_results()["group"]["item12"][key]["P_orth_rows"]
    return sp.Matrix([[sp.sympify(e) for e in r] for r in rows])


# ----------------------------------------------------------------------------------------------
# results file
# ----------------------------------------------------------------------------------------------
def save_results(section, data):
    import fcntl
    with open(HERE / ".solver_results.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        res = {}
        if RESULTS.exists():
            res = json.loads(RESULTS.read_text())
        res[section] = data
        RESULTS.write_text(json.dumps(res, indent=1, sort_keys=False))
        fcntl.flock(lk, fcntl.LOCK_UN)


def s(x):
    return str(x)


# ----------------------------------------------------------------------------------------------
# self-test of conventions
# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    # quaternion->SU(2) is a homomorphism (exact, on the generators and a product)
    for p, q in [(Q1, Q2), (Q2, Q1), (Q1 * Q2, Q2 * Q2)]:
        U = sp.Matrix(2, 2, lambda r, c: p.su2()[r][c].to_sympy())
        V = sp.Matrix(2, 2, lambda r, c: q.su2()[r][c].to_sympy())
        W = sp.Matrix(2, 2, lambda r, c: (p * q).su2()[r][c].to_sympy())
        assert sp.simplify(U * V - W) == sp.zeros(2, 2)
    # D^3 homomorphism + unitarity + Theta-commutation, exact at the generators
    n = 6
    D1 = mono_to_orthonormal(D_mono(Q1.su2(), n), n)
    D2 = mono_to_orthonormal(D_mono(Q2.su2(), n), n)
    D12 = mono_to_orthonormal(D_mono((Q1 * Q2).su2(), n), n)
    assert sp.simplify(D1 * D2 - D12) == sp.zeros(7, 7)
    assert sp.simplify(D1 * D1.H - sp.eye(7)) == sp.zeros(7, 7)
    T = sp.zeros(7, 7)
    for a, m in enumerate(mvals(3)):
        T[int(-m + 3), a] = sp.Integer(-1) ** int(m)
    for D in (D1, D2):
        assert sp.simplify(T * D.conjugate() - D * T) == sp.zeros(7, 7)
    # J+ matrix elements positive (CS)
    t = sp.symbols("t")
    Ut = sp.Matrix([[1, t], [0, 1]])   # exp(t E12)
    Dt = D_sym(Ut, n)
    Jp = sp.diff(Dt, t).subs(t, 0)
    for a in range(6):
        assert Jp[a + 1, a] == sp.sqrt(12 - (a - 3) * (a - 2))
    # equivariance of CG coupling with a random rational SU(2)-ish matrix (exact, uses SL2 is enough)
    # use the generator D1, D2 (in SU(2)): [D u (x) D v]_J = D^J [u (x) v]_J
    random.seed(1)
    u = [sp.Rational(random.randint(-5, 5), random.randint(1, 4)) + sp.I * random.randint(-3, 3)
         for _ in range(7)]
    v = [sp.Rational(random.randint(-5, 5), random.randint(1, 4)) + sp.I * random.randint(-3, 3)
         for _ in range(7)]
    for J in range(7):
        DJ = mono_to_orthonormal(D_mono(Q2.su2(), 2 * J), 2 * J)
        lhs = bracket(list(D2 * sp.Matrix(u)), 3, list(D2 * sp.Matrix(v)), 3, J)
        rhs = list(DJ * sp.Matrix(bracket(u, 3, v, 3, J)))
        assert all(sp.simplify(a - b) == 0 for a, b in zip(lhs, rhs)), J
    # rho_K(u) = [u (x) Theta u]_K = M_K(u u^dagger)
    um = sp.Matrix(u)
    for K in range(7):
        a = bracket(u, 3, theta(u), 3, K)
        b = transform(um * um.H, K)
        assert all(sp.simplify(x - y) == 0 for x, y in zip(a, b)), K
    print("solver_lib self-test: all convention checks passed")
