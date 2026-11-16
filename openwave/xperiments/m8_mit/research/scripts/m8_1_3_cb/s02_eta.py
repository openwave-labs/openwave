"""Items 0(c) and 0(e): multiplicity at the lowest level (from the projector rank), an
intertwiner eta with eta^dag eta = I, rho(h) = eta^dag D(h) eta, the constant s_N, and in R2
the charge-conjugation matrix C and its action on E_0.  Also the exact projector in the
monomial basis (used by the exact route of s03)."""
import os, sys, json, pickle
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp
from lib_check import check, summary, DEFECT
from lib_exact import Q5, Q5i, qmul, quat_to_ab
from lib_rep import group_elements, slot_character, D_mp, D_mono_exact, theta_apply
from lib_cg import mrange, w as wfac

mp.mp.dps = 60
G, q1, q2 = group_elements()
SLOTS = {"R2": Fr(7, 2), "R4": Fr(3), "R5": Fr(3)}
OUT = {}
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)


def ab_mp(q):
    a, b = quat_to_ab(q)
    return a.mp(mp), b.mp(mp)


def hopf_quadrature(f, nfreq, npoly):
    """Haar mean over SU(2) of f(a, b), a = sqrt(u) e^{i xi}, b = sqrt(1-u) e^{i zeta}, measure
    uniform in (u, xi, zeta).  Exact for trigonometric degree < nfreq in xi, zeta and polynomial
    degree < 2*npoly in u."""
    xs, ws = mp.gauss_legendre_nodes(npoly) if hasattr(mp, 'gauss_legendre_nodes') else _gl(npoly)
    tot = mp.mpf(0)
    for u, wu in zip(xs, ws):
        su, sv = mp.sqrt(u), mp.sqrt(1 - u)
        for k1 in range(nfreq):
            e1 = mp.expj(2 * mp.pi * k1 / nfreq)
            for k2 in range(nfreq):
                e2 = mp.expj(2 * mp.pi * k2 / nfreq)
                tot += wu * f(su * e1, sv * e2)
    return tot / nfreq ** 2


def _gl(n):
    """Gauss-Legendre nodes/weights on [0, 1] at current precision (Newton on P_n)."""
    xs, ws = [], []
    for i in range(1, n + 1):
        x = mp.cos(mp.pi * (i - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
        for _ in range(100):
            p0, p1 = mp.mpf(1), x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < mp.mpf(10) ** (-mp.mp.dps + 5):
                break
        p0, p1 = mp.mpf(1), x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1)
        xs.append((1 - x) / 2)
        ws.append(1 / ((1 - x * x) * dp * dp))  # weights on [-1,1] are 2/(...); on [0,1] halve
    return xs, ws


for slot, j in SLOTS.items():
    print(f"\n===== slot {slot}, j = {j} =====")
    chi, d = slot_character(slot)
    n = int(2 * j + 1)
    ms = mrange(j)
    # projector onto the rho-isotypic part of V_j: (d/|G|) sum chi(h) D(h)   (characters real)
    P = mp.matrix(n, n)
    Pm = [[Q5i(0) for _ in range(n)] for _ in range(n)]
    for h in G:
        c = chi[(h[0].a, h[0].b)]
        a, b = quat_to_ab(h)
        P += c.mp(mp) * D_mp(j, a.mp(mp), b.mp(mp), mp)
        Dm = D_mono_exact(j, a, b)
        cq = Q5i(c)
        for r in range(n):
            for s_ in range(n):
                Pm[r][s_] = Pm[r][s_] + cq * Dm[r][s_]
    P = P * (mp.mpf(d) / len(G))
    fac = Q5i(Q5(Fr(d, len(G))))
    Pm = [[fac * x for x in row] for row in Pm]
    tr = sum(P[k, k] for k in range(n))
    mult = mp.nint(mp.re(tr)) / d
    print(f"0(c) trace of projector = {mp.nstr(tr, 20)}  -> multiplicity at level {int(2*j)} = {mult}")
    check(f"{slot}: projector idempotent and Hermitian", mp.mnorm(P * P - P) < mp.mpf(10) ** -50 and mp.mnorm(P - P.H) < mp.mpf(10) ** -50)
    check(f"{slot}: multiplicity one (trace = dim rho)", abs(tr - d) < mp.mpf(10) ** -50)
    # exact projector checks (monomial basis): idempotent, trace d
    def mmul(A, B):
        return [[sum((A[r][k] * B[k][c] for k in range(n)), Q5i(0)) for c in range(n)] for r in range(n)]
    PP = mmul(Pm, Pm)
    check(f"{slot}: exact projector (monomial basis) idempotent", all(PP[r][c] == Pm[r][c] for r in range(n) for c in range(n)))
    trm = sum((Pm[k][k] for k in range(n)), Q5i(0))
    check(f"{slot}: exact projector trace = {d}", trm == Q5i(d))
    # monomial vs orthonormal: P_o = S P_m S^-1
    err = max(abs(P[r, c] - Pm[r][c].mp(mp) * mp.sqrt(mp.mpf(wfac(j, ms[r])) / wfac(j, ms[c]))) for r in range(n) for c in range(n))
    check(f"{slot}: exact monomial projector agrees with the numeric orthonormal one", err < mp.mpf(10) ** -50)
    # eta: Gram-Schmidt of P v_m, m = j, j-1, ...
    cols = []
    for k in range(n):
        v = P[:, k]
        for c in cols:
            v = v - c * (c.H * v)[0]
        if mp.norm(v) > mp.mpf(10) ** -20:
            cols.append(v / mp.norm(v))
        if len(cols) == d:
            break
    eta = mp.matrix(n, d)
    for a in range(d):
        for r in range(n):
            eta[r, a] = cols[a][r]
    if DEFECT == "eta_perturb" and slot == "R4":
        eta[0, 0] += mp.mpf(10) ** -8
    check(f"{slot}: eta^dag eta = I", mp.mnorm(eta.H * eta - mp.eye(d)) < mp.mpf(10) ** -50)
    rho = {}
    ok_int, ok_hom, ok_chr = True, True, True
    for idx, h in enumerate(G):
        Dh = D_mp(j, *ab_mp(h), mp)
        r_ = eta.H * Dh * eta
        rho[idx] = r_
        ok_int &= mp.mnorm(Dh * eta - eta * r_) < mp.mpf(10) ** -45
        ok_chr &= abs(sum(r_[k, k] for k in range(d)) - chi[(h[0].a, h[0].b)].mp(mp)) < mp.mpf(10) ** -45
    check(f"{slot}: D(h) eta = eta rho(h) for all 120 h", ok_int)
    check(f"{slot}: trace rho(h) = character of the slot for all h", ok_chr)
    # homomorphism on all pairs with q1, q2
    key = {tuple((c.a, c.b) for c in h): i for i, h in enumerate(G)}
    for gen in (q1, q2):
        for idx, h in enumerate(G):
            hg = qmul(h, gen)
            ig = key[tuple((c.a, c.b) for c in hg)]
            ok_hom &= mp.mnorm(rho[idx] * rho[key[tuple((c.a, c.b) for c in gen)]] - rho[ig]) < mp.mpf(10) ** -45
    check(f"{slot}: rho is a homomorphism (all h times each generator)", ok_hom)
    # irreducibility: sum |chi|^2 / |G| = 1 from the matrices themselves
    nrm = sum(abs(sum(rho[i][k, k] for k in range(d))) ** 2 for i in rho) / len(G)
    check(f"{slot}: rho irreducible (<chi,chi> = 1 from the matrices)", abs(nrm - 1) < mp.mpf(10) ** -40)

    # s_N: exact derivation by Schur orthogonality, mean_{SU(2)} D_{mm'} conj D_{nn'} = delta delta/(2j+1)
    # => mean |phi_w|^2 = sum_a sum_{m m'} |w_m|^2 |eta_{m'a}|^2/(2j+1) = ||w||^2 dim(rho)/(2j+1), and the
    # mean over X (normalized) equals the Haar mean since |phi_w| is a function on X.
    sN = Fr(d, n)
    print(f"0(c) s_N (exact, Schur orthogonality) = {d}/{n} = {sN}")
    # high-precision route: quadrature of the definition at two random w
    mp.mp.dps = 35
    eta35 = eta
    import random
    random.seed(7)
    for trial in range(2):
        wv = mp.matrix([mp.mpc(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)])
        def f(a, b):
            row = wv.T * D_mp(j, a, b, mp) * eta35
            return sum(abs(row[0, k]) ** 2 for k in range(d))
        val = hopf_quadrature(f, nfreq=int(4 * j) + 3, npoly=int(2 * j) + 2) / mp.norm(wv) ** 2
        print(f"   quadrature: mean|phi_w|^2/||w||^2 = {mp.nstr(val, 30)}  (trial {trial})")
        check(f"{slot}: quadrature of the definition agrees with s_N = {sN} to 1e-28", abs(val - mp.mpf(sN.numerator) / sN.denominator) < mp.mpf(10) ** -28)
    mp.mp.dps = 60

    rec = dict(j=str(j), dim=d, multiplicity=int(mult), s_N=str(sN),
               eta=[[[mp.nstr(mp.re(eta[r, a]), 50), mp.nstr(mp.im(eta[r, a]), 50)] for a in range(d)] for r in range(n)])
    # charge conjugation: conj(rho(h)) C = C rho(h) for the generators
    i1 = key[tuple((c.a, c.b) for c in q1)]
    i2 = key[tuple((c.a, c.b) for c in q2)]
    rows = []
    for ig in (i1, i2):
        R = rho[ig]
        Rc = mp.matrix(d, d)
        for r in range(d):
            for c in range(d):
                Rc[r, c] = mp.conj(R[r, c])
        # unknown C (d x d), vec index r*d + c ; equation conj(R) C - C R = 0
        for r in range(d):
            for c in range(d):
                row = [mp.mpc(0)] * (d * d)
                for k in range(d):
                    row[k * d + c] += Rc[r, k]
                    row[r * d + k] -= R[k, c]
                rows.append(row)
    A = mp.matrix(rows)
    U, S, V = mp.svd_c(A)
    nullvec = V.H[:, d * d - 1]
    smin = S[d * d - 1]
    s2 = S[d * d - 2]
    check(f"{slot}: solution space of conj(rho)C = C rho is one-dimensional", smin < mp.mpf(10) ** -45 and s2 > mp.mpf(10) ** -5)
    C = mp.matrix(d, d)
    for r in range(d):
        for c in range(d):
            C[r, c] = nullvec[r * d + c]
    C = C * mp.sqrt(d) / mp.norm(C)  # unitary up to phase (Schur): fix the phase on the largest entry
    big = max(((r, c) for r in range(d) for c in range(d)), key=lambda t: abs(C[t[0], t[1]]))
    C = C * (abs(C[big]) / C[big])
    check(f"{slot}: C unitary", mp.mnorm(C.H * C - mp.eye(d)) < mp.mpf(10) ** -45)
    ok = all(mp.mnorm(mp.matrix([[mp.conj(rho[i][r, c]) for c in range(d)] for r in range(d)]) * C - C * rho[i]) < mp.mpf(10) ** -45 for i in rho)
    check(f"{slot}: conj(rho(h)) C = C rho(h) for all h", ok)
    CCb = C * mp.matrix([[mp.conj(C[r, c]) for c in range(d)] for r in range(d)])
    sgn = +1 if mp.mnorm(CCb - mp.eye(d)) < 1e-40 else (-1 if mp.mnorm(CCb + mp.eye(d)) < 1e-40 else 0)
    print(f"0(e)/aux {slot}: C C-bar = {sgn:+d} * I")
    print("   C =", [[mp.nstr(C[r, c], 25) for c in range(d)] for r in range(d)])
    # action on E_0: psibar_w C = c psi_{Theta w}, c from (Theta eta) C = c eta
    Teta = mp.matrix(n, d)
    for a in range(d):
        col = theta_apply(j, eta[:, a], mp)
        for r in range(n):
            Teta[r, a] = col[r]
    TC = Teta * C
    cph = (eta.H * TC)[0, 0]
    check(f"{slot}: (Theta eta) C = c eta with |c| = 1", mp.mnorm(TC - eta * cph) < mp.mpf(10) ** -45 and abs(abs(cph) - 1) < mp.mpf(10) ** -45)
    print(f"   (Theta eta) C = c eta with c = {mp.nstr(cph, 30)}")
    # direct check on functions at two random points g and a random w
    import random
    random.seed(3)
    wv = mp.matrix([mp.mpc(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)])
    Tw = theta_apply(j, wv, mp)
    okf = True
    for h in (G[11], G[99]):
        Dg = D_mp(j, *ab_mp(h), mp)
        lhs = mp.matrix([[mp.conj(x) for x in (wv.T * Dg * eta)]]) * C
        rhs = cph * (Tw.T * Dg * eta)
        okf &= mp.norm(lhs - rhs) < mp.mpf(10) ** -45
    check(f"{slot}: conj(phi_w) C = c phi_(Theta w) pointwise", okf)
    rec.update(C=[[[mp.nstr(mp.re(C[r, c]), 40), mp.nstr(mp.im(C[r, c]), 40)] for c in range(d)] for r in range(d)],
               CCbar=sgn, c_phase=[mp.nstr(mp.re(cph), 40), mp.nstr(mp.im(cph), 40)])
    OUT[slot] = rec
    if not DEFECT: pickle.dump(dict(Pm=[[((x.re.a, x.re.b), (x.im.a, x.im.b)) for x in row] for row in Pm], d=d, j=j),
                open(os.path.join(HERE, "out", f"Pmono_{slot}.pkl"), "wb"))
    print("0(c) eta (columns a, rows m = j..-j), 25 digits:")
    for r in range(n):
        print(f"   m={str(ms[r]):>5}: " + "  ".join(mp.nstr(eta[r, a], 22) for a in range(d)))

if not DEFECT:
    json.dump(OUT, open(os.path.join(HERE, "out", "eta.json"), "w"), indent=1)
summary()
