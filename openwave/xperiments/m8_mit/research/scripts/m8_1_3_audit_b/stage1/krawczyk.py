"""Interval linear algebra and Krawczyk tests on mpmath.iv (own code)."""
from mpmath import mp, iv
from ivutil import ends


def ivz():
    return iv.mpf(0)


def mat_mul(A, B):
    n = len(A); m = len(B[0]); l = len(B)
    return [[sum((A[i][k] * B[k][j] for k in range(l)), ivz()) for j in range(m)] for i in range(n)]


def mat_vec(A, x):
    return [sum((A[i][k] * x[k] for k in range(len(x))), ivz()) for i in range(len(A))]


def mid(x):
    lo, hi = ends(x)
    return (lo + hi) / 2


def mp_inverse(Amid):
    M = mp.matrix(Amid)
    return mp.inverse(M)


def krawczyk(F, J, Z, m=None):
    """F(list of iv) -> list of iv; J(list of iv) -> iv matrix; Z: list of iv (box).
    Returns (ok, K) with K = m - Y F(m) + (I - Y J(Z))(Z - m); ok iff K lies in the interior of Z."""
    n = len(Z)
    if m is None:
        m = [iv.mpf(mid(z)) for z in Z]
    Jz = J(Z)
    Jm = [[mid(Jz[i][k]) for k in range(n)] for i in range(n)]
    Y = mp_inverse(Jm)
    Yi = [[iv.mpf(Y[i, k]) for k in range(n)] for i in range(n)]
    Fm = F(m)
    YF = mat_vec(Yi, Fm)
    YJ = mat_mul(Yi, Jz)
    IYJ = [[(iv.mpf(1) if i == k else ivz()) - YJ[i][k] for k in range(n)] for i in range(n)]
    dz = [Z[i] - m[i] for i in range(n)]
    corr = mat_vec(IYJ, dz)
    K = [m[i] - YF[i] + corr[i] for i in range(n)]
    ok = True
    for i in range(n):
        klo, khi = ends(K[i]); zlo, zhi = ends(Z[i])
        if not (klo > zlo and khi < zhi):
            ok = False
    # contraction factor (for the record)
    nrm = max(sum(max(abs(e) for e in ends(IYJ[i][k])) for k in range(n)) for i in range(n))
    return ok, K, nrm


def intersect(a, b):
    alo, ahi = ends(a); blo, bhi = ends(b)
    lo = max(alo, blo); hi = min(ahi, bhi)
    assert lo <= hi
    return iv.mpf([lo, hi])


def charpoly_leverrier(A):
    """coefficients c[0..n] of det(t I - A) = sum c_k t^k (c_n = 1), interval Faddeev-LeVerrier."""
    n = len(A)
    c = [None] * (n + 1)
    c[n] = iv.mpf(1)
    Mk = [[iv.mpf(1) if i == k else ivz() for k in range(n)] for i in range(n)]   # M_1 = I
    for kk in range(1, n + 1):
        AM = mat_mul(A, Mk)
        tr = sum((AM[i][i] for i in range(n)), ivz())
        c[n - kk] = -tr / kk
        if kk < n:
            Mk = [[AM[i][j] + (c[n - kk] if i == j else ivz()) for j in range(n)] for i in range(n)]
    return c


def poly_eval(cs, t):
    acc = ivz() if not isinstance(t, tuple) else None
    r = ivz()
    for c in reversed(cs):
        r = r * t + c
    return r


def poly_deriv(cs):
    return [cs[k] * k for k in range(1, len(cs))]


def root_krawczyk_real(cs, t0, rad):
    """certify a unique (hence simple) real root of every polynomial with coefficients in the enclosures cs
    in I = [t0 - rad, t0 + rad]; returns (ok, K) with K the Krawczyk image."""
    I = iv.mpf([t0 - rad, t0 + rad])
    m = iv.mpf(t0)
    d = poly_deriv(cs)
    dm = mid(poly_eval(d, m))
    Ki = m - poly_eval(cs, m) / dm + (iv.mpf(1) - poly_eval(d, I) / dm) * (I - m)
    klo, khi = ends(Ki); ilo, ihi = ends(I)
    return (klo > ilo and khi < ihi), Ki
