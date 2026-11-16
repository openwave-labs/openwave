"""Exact-for-the-integrand quadrature on SU(2) in Euler angles (ZYZ), mpmath."""
import mpmath as mp
from common import mono_matrix, mvals, fm

def Dcs_num(alpha, beta, gamma, j2):
    """D^j(R_z(alpha) R_y(beta) R_z(gamma)) on CS components, via the monomial construction."""
    ca, sa = mp.cos(alpha / 2), mp.sin(alpha / 2)
    cb, sb = mp.cos(beta / 2), mp.sin(beta / 2)
    cg_, sg = mp.cos(gamma / 2), mp.sin(gamma / 2)
    Uz = lambda c, s: mp.matrix([[mp.mpc(c, -s), 0], [0, mp.mpc(c, s)]])
    Uy = mp.matrix([[cb, -sb], [sb, cb]])
    U = Uz(ca, sa) * Uy * Uz(cg_, sg)
    Ul = [[U[0, 0], U[0, 1]], [U[1, 0], U[1, 1]]]
    M = mono_matrix(Ul, j2, mp.mpc(0), mp.mpc(1))
    ms = mvals(j2)
    n = j2 + 1
    return mp.matrix([[M[a][b] * mp.sqrt(mp.mpf(fm(j2, ms[a])) / fm(j2, ms[b])) for b in range(n)] for a in range(n)])

def quad_nodes(N=25):
    """Exact quadrature for int_SU(2) of trig polynomials of degree < N/2 in each Euler angle
    (Haar, normalised).  alpha, gamma: trapezoid on N points.  beta: f(beta) is a trig polynomial
    of degree K = (N-1)/2; weights W_n = (1/N) int_0^pi sin(b) [1 + 2 sum_k cos k(b - b_n)] db / 2."""
    K = (N - 1) // 2
    nodes = [2 * mp.pi * n / N for n in range(N)]
    def Ic(k):
        if k == 1: return mp.mpf(0)
        return (1 + mp.cos(k * mp.pi)) / (1 - k * k)
    def Is(k): return mp.pi / 2 if k == 1 else mp.mpf(0)
    Wb = []
    for b in nodes:
        t = Ic(0)
        for k in range(1, K + 1):
            t += 2 * (mp.cos(k * b) * Ic(k) + mp.sin(k * b) * Is(k))
        Wb.append(t / N / 2)     # /2 normalises int_0^pi sin = 2
    return nodes, Wb

# D(alpha, beta, 0) cached on the grid; gamma enters as phases on the columns
_grids = {}
def avg_moments(Pnum, w, j2=6, N=25):
    """(<|phi_w|^2>, <|phi_w|^4>) over normalised Haar measure; phi_w(g)_a = sum w_m D_mm'(g) eta_m'a,
    so |phi_w|^2 = u^T P conj(u) with u_m' = sum_m w_m D_mm'(g), P = eta eta^+."""
    nodes, Wb = quad_nodes(N)
    grid = _grids.setdefault((j2, N, mp.mp.dps), {})
    n = j2 + 1
    ms = mvals(j2)
    phs = [[mp.expj(-ms[c] * gmm) for c in range(n)] for gmm in nodes]
    s2 = mp.mpf(0); s4 = mp.mpf(0)
    for ia, a in enumerate(nodes):
        for ib, b in enumerate(nodes):
            key = (ia, ib)
            if key not in grid:
                grid[key] = Dcs_num(a, b, 0, j2)
            D0 = grid[key]
            u0 = [mp.fsum(w[r] * D0[r, c] for r in range(n)) for c in range(n)]
            wt = Wb[ib] / (N * N)
            for ph in phs:
                u = [u0[c] * ph[c] for c in range(n)]
                val = mp.re(mp.fsum(u[r] * Pnum[r, c] * mp.conj(u[c]) for r in range(n) for c in range(n)))
                s2 += wt * val; s4 += wt * val * val
    return s2, s4
