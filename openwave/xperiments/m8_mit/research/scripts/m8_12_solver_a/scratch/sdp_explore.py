"""Exploration: level-2 (sextic) Hermitian-SOS relaxation for max rhat6, numerically (double)."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, math, itertools
import common as C
np.set_printoptions(linewidth=200)
Jx = np.array(C.JX.evalf(), complex); Jy = np.array(C.JY.evalf(), complex); Jz = np.array(C.JZ.evalf(), complex)
Jp = Jx + 1j * Jy
I7 = np.eye(7)
def tot(A):  # A on V (x) V (x) V
    return (np.kron(np.kron(A, I7), I7) + np.kron(np.kron(I7, A), I7) + np.kron(np.kron(I7, I7), A))
Tp = tot(Jp); Tz = tot(Jz)
# symmetrizer on first two factors (for psi2) and on all three (for psi1)
def perm_op(p):
    P = np.zeros((343, 343))
    for i, j, k in itertools.product(range(7), repeat=3):
        idx = [i, j, k]
        src = 49 * i + 7 * j + k
        t = [idx[p[0]], idx[p[1]], idx[p[2]]]
        P[49 * t[0] + 7 * t[1] + t[2], src] = 1
    return P
perms = list(itertools.permutations(range(3)))
S3 = sum(perm_op(p) for p in perms) / 6
S12 = (np.eye(343) + perm_op((1, 0, 2))) / 2
def isotypic_basis(Sproj):
    """for each L: orthonormal highest-weight vectors in range(Sproj), then lowered families."""
    out = {}
    w, V = np.linalg.eigh(Sproj); R = V[:, w > 0.5]          # orthonormal basis of the subspace
    Tz_r = R.conj().T @ Tz @ R
    for L in range(10):
        # vectors in subspace with Jz = L and J+ = 0
        A = np.vstack([Tp @ R, (Tz - L * np.eye(343)) @ R])
        u_, s_, vh = np.linalg.svd(A)
        null = vh[np.sum(s_ > 1e-9):].conj().T
        if null.shape[1] == 0: continue
        H = R @ null                          # highest weight vectors (orthonormal since R, null orthonormal)
        fams = []
        for a in range(H.shape[1]):
            v = H[:, a]; fam = [v]
            Tm = Tp.conj().T
            for M in range(L, -L, -1):
                v = Tm @ v; v = v / np.linalg.norm(v); fam.append(v)
            fams.append(np.array(fam))       # (2L+1) x 343
        out[L] = fams
    return out
B1 = isotypic_basis(S3); B2 = isotypic_basis(S12)
print('Sym3 mults:', {L: len(f) for L, f in B1.items()}, 'dim', sum(len(f) * (2 * L + 1) for L, f in B1.items()))
print('Sym2xV mults:', {L: len(f) for L, f in B2.items()}, 'dim', sum(len(f) * (2 * L + 1) for L, f in B2.items()))
MS = C.MS
def cg(m1, m2, Q): return math.sqrt(math.comb(6, 3 + m1) * math.comb(6, 3 + m2) / math.comb(12, 6 + Q))
def r6N(c):
    th = np.array([(-1) ** (3 - m) * np.conj(c[3 + m]) for m in MS])
    tot_ = 0
    for Q in range(-6, 7):
        s = 0
        for m1 in MS:
            m2 = Q - m1
            if abs(m2) <= 3: s += cg(m1, m2, Q) * c[3 - m1] * th[3 - m2]
        tot_ += abs(s) ** 2
    return tot_
def theta(c): return np.array([(-1) ** (3 - m) * np.conj(c[3 + m]) for m in MS])
def grams(c):
    p1 = np.kron(np.kron(c, c), c); p2 = np.kron(np.kron(c, c), theta(c))
    G = []
    for B, p in ((B1, p1), (B2, p2)):
        for L, fams in sorted(B.items()):
            coeff = np.array([[np.vdot(f[M], p) for M in range(2 * L + 1)] for f in fams])  # mult x (2L+1)
            G.append(coeff.conj() @ coeff.T)   # G_ab = sum_M conj<e^a,p> <e^b,p>  ... hermitian
    return G
# variable layout: lam, then for each block a hermitian Y (real params: diag + re/im upper)
blocks = [len(f) for B in (B1, B2) for L, f in sorted(B.items())]
def herm_params(n): return n * n
nvar = 1 + sum(herm_params(n) for n in blocks)
def Y_from(xv):
    Ys = []; k = 1
    for n in blocks:
        Y = np.zeros((n, n), complex)
        for i in range(n):
            Y[i, i] = xv[k]; k += 1
        for i in range(n):
            for j in range(i + 1, n):
                Y[i, j] = xv[k] + 1j * xv[k + 1]; Y[j, i] = np.conj(Y[i, j]); k += 2
        Ys.append(Y)
    return Ys
rng = np.random.default_rng(5)
rows = []; rhs = []
for s in range(3 * nvar):
    c = rng.normal(size=7) + 1j * rng.normal(size=7)
    c = c / np.linalg.norm(c)
    n2 = np.vdot(c, c).real
    G = grams(c)
    row = np.zeros(nvar); row[0] = n2 ** 3
    k = 1
    for n, Gb in zip(blocks, G):
        # tr(Y G) = sum_ij Y_ij G_ji
        for i in range(n):
            row[k] = Gb[i, i].real; k += 1
        for i in range(n):
            for j in range(i + 1, n):
                # Y_ij G_ji + Y_ji G_ij = 2 Re(Y_ij G_ji) = 2(a Re G_ji - b Im G_ji)
                row[k] = 2 * Gb[j, i].real; row[k + 1] = -2 * Gb[j, i].imag; k += 2
        # F = lam n^6 - N n^2 = sum tr(Y G)  ->  lam n^6 - sum tr(YG) = N n^2
    row[1:] *= -1
    rows.append(row); rhs.append(r6N(c) * n2)
A = np.array(rows); b = np.array(rhs)
# particular solution + nullspace
x0, *_ = np.linalg.lstsq(A, b, rcond=None)
print('consistency residual', np.abs(A @ x0 - b).max())
u_, s_, vh = np.linalg.svd(A)
rank = np.sum(s_ > 1e-8 * s_[0]); Z = vh[rank:].T
print('nvar', nvar, 'rank', rank, 'free', Z.shape[1])
# barrier method: minimize lam(t) - mu * sum logdet Y(t)
def unpack(t): return x0 + Z @ t
def obj(t, mu):
    xv = unpack(t); Ys = Y_from(xv)
    val = xv[0]
    for Y in Ys:
        w = np.linalg.eigvalsh(Y)
        if w.min() <= 0: return np.inf
        val -= mu * np.sum(np.log(w))
    return val

# ---- affine data: lam(v) = lam0 + c.v ; Y_b(v) = Y_b0 + sum v_i Y_bi
k = Z.shape[1]
lam0 = x0[0]; cvec = Z[0]
Y0 = Y_from(x0)
Ycols = [Y_from(Z[:, i]) for i in range(k)]
nb = len(blocks)
# restrict to lam(v) = Lam: v = vp(Lam) + Nc w
_, _, vhc = np.linalg.svd(cvec.reshape(1, -1)); Nc = vhc[1:].T
def setup(Lam):
    vp = cvec * (Lam - lam0) / (cvec @ cvec)
    Yb0 = [Y0[b] + sum(vp[i] * Ycols[i][b] for i in range(k)) for b in range(nb)]
    Ybi = [[sum(Nc[i, j] * Ycols[i][b] for i in range(k)) for b in range(nb)] for j in range(Nc.shape[1])]
    return Yb0, Ybi
def maxmineig(Lam, iters=60):
    Yb0, Ybi = setup(Lam)
    m = len(Ybi)
    def Ys(w): return [Yb0[b] + sum(w[j] * Ybi[j][b] for j in range(m)) for b in range(nb)]
    w = np.zeros(m)
    s_ = min(np.linalg.eigvalsh(Y).min() for Y in Ys(w)) - 1.0
    x = np.concatenate([w, [s_]])
    for mu in [1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 1e-7]:
        for it in range(iters):
            Yl = Ys(x[:m]); sv = x[m]
            g = np.zeros(m + 1); g[m] = -1.0; H = np.zeros((m + 1, m + 1))
            for b in range(nb):
                n = Yl[b].shape[0]
                Yinv = np.linalg.inv(Yl[b] - sv * np.eye(n))
                P = [Yinv @ Ybi[j][b] for j in range(m)] + [-Yinv]
                tr = np.array([np.trace(Pi).real for Pi in P])
                g -= mu * tr
                for i in range(m + 1):
                    for j in range(i, m + 1):
                        h = mu * np.trace(P[i] @ P[j]).real
                        H[i, j] += h
                        if i != j: H[j, i] += h
            step = -np.linalg.solve(H + 1e-15 * np.eye(m + 1), g)
            dec = -g @ step
            def f(xx):
                Yl_ = Ys(xx[:m]); tot_ = -xx[m]
                for Y in Yl_:
                    ww = np.linalg.eigvalsh(Y - xx[m] * np.eye(Y.shape[0]))
                    if ww.min() <= 0: return np.inf
                    tot_ -= mu * np.sum(np.log(ww))
                return tot_
            t = 1.0; f0 = f(x)
            while f(x + t * step) > f0 - 0.25 * t * dec and t > 1e-14: t *= 0.5
            if t <= 1e-14: break
            x = x + t * step
            if dec < 1e-13: break
    return x[m], x
lo, hi = 463 / 924 - 0.01, 11 / 21

# ---- facial reduction from the cat state: Y_b must annihilate w_M = conj(C_{.M}(cat))
cat = np.zeros(7, complex); cat[0] = cat[6] = 1 / np.sqrt(2)
def coeffs(c):
    p1 = np.kron(np.kron(c, c), c); p2 = np.kron(np.kron(c, c), theta(c))
    out = []
    for B, p in ((B1, p1), (B2, p2)):
        for L, fams in sorted(B.items()):
            out.append(np.array([[np.vdot(f[M], p) for M in range(2 * L + 1)] for f in fams]))
    return out
Ccat = coeffs(cat)
Pb = []
for b in range(nb):
    W = Ccat[b].conj()                   # mult x (2L+1), columns w_M
    u_, sv, vh_ = np.linalg.svd(W)
    r = np.sum(sv > 1e-10)
    Pb.append(u_[:, r:])                 # orthonormal basis of the complement of the forced kernel
print('block sizes', blocks)
print('reduced sizes', [P.shape[1] for P in Pb])
def maxmineig_red(Lam, iters=60):
    Yb0, Ybi = setup(Lam)
    Yb0 = [P.conj().T @ Y @ P for P, Y in zip(Pb, Yb0)]
    Ybi = [[P.conj().T @ Y @ P for P, Y in zip(Pb, row)] for row in Ybi]
    keep = [b for b in range(nb) if Pb[b].shape[1] > 0]
    # the forced-kernel parts must vanish identically: constrain Y_b P_perp = 0 -> linear constraints on w
    m = len(Ybi)
    rows_ = []; rhs_ = []
    Yfull0, Yfulli = setup(Lam)
    for b in range(nb):
        K_ = np.linalg.svd(Ccat[b].conj())[0][:, :np.sum(np.linalg.svd(Ccat[b].conj())[1] > 1e-10)]
        if K_.shape[1] == 0: continue
        R0 = Yfull0[b] @ K_
        Ri = [Yfulli[j][b] @ K_ for j in range(m)]
        for (i1, i2) in np.ndindex(R0.shape):
            rows_.append([Ri[j][i1, i2].real for j in range(m)]); rhs_.append(-R0[i1, i2].real)
            rows_.append([Ri[j][i1, i2].imag for j in range(m)]); rhs_.append(-R0[i1, i2].imag)
    Aeq = np.array(rows_); beq = np.array(rhs_)
    w0, *_ = np.linalg.lstsq(Aeq, beq, rcond=None)
    print('   forced-kernel constraints: residual %.2e, rank %d of %d vars' % (np.abs(Aeq @ w0 - beq).max(), np.linalg.matrix_rank(Aeq, 1e-9), m))
    _, sv_, vh2 = np.linalg.svd(Aeq); rk = np.sum(sv_ > 1e-9); Nw = vh2[rk:].T
    Y0r = [Pb[b].conj().T @ (Yfull0[b] + sum(w0[j] * Yfulli[j][b] for j in range(m))) @ Pb[b] for b in keep]
    Yir = [[Pb[b].conj().T @ sum(Nw[j, q] * Yfulli[j][b] for j in range(m)) @ Pb[b] for b in keep] for q in range(Nw.shape[1])]
    mm = len(Yir); nbk = len(keep)
    def Ys(w): return [Y0r[b] + sum(w[j] * Yir[j][b] for j in range(mm)) for b in range(nbk)]
    x = np.concatenate([np.zeros(mm), [min(np.linalg.eigvalsh(Y).min() for Y in Ys(np.zeros(mm))) - 1.0]])
    for mu in [1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6]:
        for it in range(iters):
            Yl = Ys(x[:mm]); sv = x[mm]
            g = np.zeros(mm + 1); g[mm] = -1.0; H = np.zeros((mm + 1, mm + 1))
            for b in range(nbk):
                n = Yl[b].shape[0]
                Yinv = np.linalg.inv(Yl[b] - sv * np.eye(n))
                P = [Yinv @ Yir[j][b] for j in range(mm)] + [-Yinv]
                g -= mu * np.array([np.trace(Pi).real for Pi in P])
                for i in range(mm + 1):
                    for j in range(i, mm + 1):
                        h = mu * np.trace(P[i] @ P[j]).real
                        H[i, j] += h
                        if i != j: H[j, i] += h
            step = -np.linalg.solve(H + 1e-15 * np.eye(mm + 1), g); dec = -g @ step
            def f(xx):
                tot_ = -xx[mm]
                for Y in Ys(xx[:mm]):
                    ww = np.linalg.eigvalsh(Y - xx[mm] * np.eye(Y.shape[0]))
                    if ww.min() <= 0: return np.inf
                    tot_ -= mu * np.sum(np.log(ww))
                return tot_
            t = 1.0; f0 = f(x)
            while f(x + t * step) > f0 - 0.25 * t * dec and t > 1e-14: t *= 0.5
            if t <= 1e-14: break
            x = x + t * step
            if dec < 1e-13: break
    return x[mm], [np.linalg.eigvalsh(Y) for Y in Ys(x[:mm])]
for Lam in [463 / 924]:
    sm, eigs = maxmineig_red(Lam)
    print('REDUCED: Lam = %.10f  max min-eig = %.4e' % (Lam, sm))
    for e in eigs: print('   ', np.round(e, 6))
raise SystemExit
print('singular values', s_[:14])
def resid_at(xv, c):
    c = c / np.linalg.norm(c); G = grams(c); Ys = Y_from(xv)
    return xv[0] - r6N(c) - sum(np.trace(Y @ Gb).real for Y, Gb in zip(Ys, G))
for Lam in [11 / 21, 0.51, 463 / 924, 463 / 924 - 1e-3]:
    sm, xx = maxmineig(Lam)
    Yb0, Ybi = setup(Lam)
    v = cvec * (Lam - lam0) / (cvec @ cvec) + Nc @ xx[:-1]
    xv = x0 + Z @ v
    cat = np.zeros(7, complex); cat[0] = cat[6] = 1
    tests = [resid_at(xv, rng.normal(size=7) + 1j * rng.normal(size=7)) for _ in range(5)] + [resid_at(xv, cat)]
    print('Lam = %.8f  max min-eig = %.3e  lam(x)=%.8f  identity residuals at fresh points + cat: %s' % (Lam, sm, xv[0], np.round(tests, 10)), flush=True)
