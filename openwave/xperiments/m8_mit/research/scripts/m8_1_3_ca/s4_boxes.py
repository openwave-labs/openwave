"""C.4 at O1, O2, O3 in R4 and R5: Krawczyk test of the slice system on each box of inputs.json (interval
arithmetic, mpmath.iv), proof that the certified zero has c = 0 (so it is a critical point of h), and an
enclosure of R0~(x/|x|_G) over the whole box (x-part), hence its sign.

h in the chart: w_m = sqrt(N_m) p_m, N_m = (3+m)!(3-m)!,
  h = c0 (x^T G x)^2 + c6 sum_M [ (x^T R_M x)^2 + (x^T I_M x)^2 ],  B6_M = x^T (R_M + i I_M) x = [w (x) Theta w]_{6,M}.
"""
import os, sys, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from common import *

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
iv = mp.iv
iv.dps = 130
mp.mp.dps = 130
consts = pickle.load(open(os.path.join(OUT, "consts.pkl"), "rb"))
inp = json.load(open(os.path.join(HERE, "inputs.json")))
PLANT = os.environ.get("PLANT", "")       # planted defects for the checks (see planted.py)

def ivq(q):
    q = sp.Rational(q); return iv.mpf(int(q.p)) / int(q.q)
def ivsqrtq(q):
    return iv.sqrt(ivq(q))
def signed_sqrt_iv(sign, sq):
    return sign * ivsqrtq(sq) if sign else iv.mpf(0)

def build(j):
    """tables for V_j: weights N_m, metric G, and the quadratic forms R_M, I_M of B6 = [w (x) Theta w]_6."""
    ms = mlist(j); n = len(ms)
    Nm = [sp.factorial(j + m) * sp.factorial(j - m) for m in ms]
    Gdiag = [int(x) for x in Nm] * 2
    RM = [dict() for _ in range(13)]; IM = [dict() for _ in range(13)]
    def addsym(D, r, c, v):
        if r == c:
            D[(r, r)] = D.get((r, r), iv.mpf(0)) + v
        else:
            D[(r, c)] = D.get((r, c), iv.mpf(0)) + v / 2
            D[(c, r)] = D.get((c, r), iv.mpf(0)) + v / 2
    for Mi, M in enumerate(mlist(6)):
        for a, m1 in enumerate(ms):
            m2 = M - m1
            if abs(m2) > j:
                continue
            sgn, sq = cg_rational_square(j, m1, j, m2, 6, M)
            if sgn == 0:
                continue
            th = (-1) ** int(j - m2)
            k = ms.index(-m2)
            kv = signed_sqrt_iv(sgn * th, sq * Nm[a] * Nm[k])   # B6_M = sum kv p_{m1} conj(p_{-m2})
            Xa, Ya, Xk, Yk = a, n + a, k, n + k
            addsym(RM[Mi], Xa, Xk, kv); addsym(RM[Mi], Ya, Yk, kv)     # Re: X_a X_k + Y_a Y_k
            addsym(IM[Mi], Ya, Xk, kv); addsym(IM[Mi], Xa, Yk, -kv)    # Im: Y_a X_k - X_a Y_k
    return dict(j=j, ms=ms, n=n, Nm=Nm, Gdiag=Gdiag, RM=RM, IM=IM)

T3 = build(sp.Integer(3))
j = sp.Integer(3); ms = T3["ms"]; n = 7; Nm = T3["Nm"]; Gdiag = T3["Gdiag"]; RM = T3["RM"]; IM = T3["IM"]

def qf(D, x):
    return sum((v * x[r] * x[c] for (r, c), v in D.items()), iv.mpf(0))
def mv(D, x):
    out = [iv.mpf(0)] * len(x)
    for (r, c), v in D.items():
        out[r] += v * x[c]
    return out

def h_parts(x, c0, c6, want_hess=True):
    Gx = [Gdiag[r] * x[r] for r in range(14)]
    xGx = sum((x[r] * Gx[r] for r in range(14)), iv.mpf(0))
    h = c0 * xGx ** 2
    grad = [4 * c0 * xGx * g for g in Gx]
    H = [[iv.mpf(0)] * 14 for _ in range(14)] if want_hess else None
    if want_hess:
        for r in range(14):
            for c in range(14):
                H[r][c] = 4 * c0 * (2 * Gx[r] * Gx[c] + (xGx * Gdiag[r] if r == c else 0))
    for D in RM + IM:
        q = qf(D, x); Dx = mv(D, x)
        h += c6 * q ** 2
        for r in range(14):
            grad[r] += 4 * c6 * q * Dx[r]
        if want_hess:
            for r in range(14):
                for c in range(14):
                    H[r][c] += 4 * c6 * (2 * Dx[r] * Dx[c] + q * D.get((r, c), 0))
    return h, grad, H, Gx, xGx

gens = [[[ivq(sp.Rational(e)) for e in row] for row in Mx] for Mx in inp["generators"]["matrices"]]
gens_exact = [sp.Matrix([[sp.Rational(e) for e in row] for row in Mx]) for Mx in inp["generators"]["matrices"]]
Gm = sp.diag(*Gdiag)
check("generators: each X_i is G-skew (X^T G + G X = 0), exact", all((X.T * Gm + Gm * X) == sp.zeros(14) for X in gens_exact))
# my own infinitesimal generators: phase, and the derivations of x -> x + t(A11 x + A21 y), y -> y + t(A12 x + A22 y)
def my_gen(A):
    """real 14x14 matrix of the infinitesimal action on weighted coordinates p (A a 2x2 sympy matrix)."""
    x, y = sp.symbols("x y")
    Mc = sp.zeros(7, 7)
    for col, mp_ in enumerate(ms):
        P = x ** int(3 + mp_) * y ** int(3 - mp_)
        dP = sp.expand((A[0, 0] * x + A[1, 0] * y) * sp.diff(P, x) + (A[0, 1] * x + A[1, 1] * y) * sp.diff(P, y))
        for row, m in enumerate(ms):
            Mc[row, col] = sp.Poly(dP, x, y).coeff_monomial(x ** int(3 + m) * y ** int(3 - m))
    R = sp.zeros(14, 14)
    for r in range(7):
        for c in range(7):
            v = Mc[r, c]; re, im = sp.re(v), sp.im(v)
            R[r, c] = re; R[r, 7 + c] = -im; R[7 + r, c] = im; R[7 + r, 7 + c] = re
    return R
Iq, Jq, Kq = sp.Matrix([[sp.I, 0], [0, -sp.I]]), sp.Matrix([[0, 1], [-1, 0]]), sp.Matrix([[0, sp.I], [sp.I, 0]])
mine = [my_gen(sp.I * sp.eye(2)) / 3 * 0 + sp.Matrix(sp.BlockMatrix([[sp.zeros(7), -sp.eye(7)], [sp.eye(7), sp.zeros(7)]])),
        my_gen(Iq), my_gen(Jq), my_gen(Kq)]
stack = sp.Matrix([list(X) for X in gens_exact + mine])
check("generators: span of inputs.json's four == span of my phase + su(2) generators (rank 4 of the 8)", stack.rank() == 4, f"rank {stack.rank()}")


from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, getcontext
getcontext().prec = 400
def dec_out(x, digits, up):
    """decimal string with `digits` significant digits, rounded down (up=False) or up (up=True): exact from the binary value."""
    x = mp.mpf(x)
    if x == 0:
        return "0"
    sgn, man, exp, _ = x._mpf_                 # sign bit, unsigned mantissa, exponent
    d = Decimal(int(man)) * (Decimal(2) ** int(exp))
    if sgn:
        d = -d
    q = Decimal(1).scaleb(d.adjusted() - digits + 1)
    return str(d.quantize(q, rounding=ROUND_CEILING if up else ROUND_FLOOR))
def encl(v, digits=45):
    return [dec_out(v.a, digits, False), dec_out(v.b, digits, True)]

def mid(v):
    return (mp.mpf(v.a) + mp.mpf(v.b)) / 2

def krawczyk(s, name, box):
    c0 = ivq(consts[s]["cJ"][0]); c6 = ivq(consts[s]["cJ"][6])
    xc_str = box["centre, x-part"]; x0_str = box["x0"]
    x0 = [iv.mpf(t) for t in x0_str]
    xc = [iv.mpf(t) for t in xc_str]
    if PLANT == "shift-centre":
        xc[2] = xc[2] + iv.mpf("1e-55")
    # lambda centre from my own h, at the given x (a point value, high precision)
    hc, _, _, _, xGxc = h_parts(xc, c0, c6, want_hess=False)
    lam_c = mid(4 * hc / xGxc)
    centre = [mid(t) for t in xc] + [lam_c] + [mp.mpf(0)] * 4
    if PLANT == "wrong-c6":
        c6 = c6 * (1 + iv.mpf("1e-6"))
    r = iv.mpf("1e-60")
    B = []
    for t in centre:
        lo = mp.mpf((iv.mpf(t) - r).b)      # inward rounding: B is inside the stated box
        hi = mp.mpf((iv.mpf(t) + r).a)
        B.append(iv.mpf([lo, hi]))
    k = [[sum((gens[i][a][b] * x0[b] for b in range(14)), iv.mpf(0)) for a in range(14)] for i in range(4)]
    Gk = [[Gdiag[a] * k[i][a] for a in range(14)] for i in range(4)]
    def F_and_J(X, want_J):
        x = X[:14]; lam = X[14]; c = X[15:]
        h, grad, H, Gx, xGx = h_parts(x, c0, c6, want_hess=want_J)
        F = [grad[a] - lam * Gx[a] - sum((c[i] * Gk[i][a] for i in range(4)), iv.mpf(0)) for a in range(14)]
        F.append(xGx - 1)
        for i in range(4):
            F.append(sum((Gk[i][a] * (x[a] - x0[a]) for a in range(14)), iv.mpf(0)))
        if not want_J:
            return F, None
        J = [[iv.mpf(0)] * 19 for _ in range(19)]
        for a in range(14):
            for b in range(14):
                J[a][b] = H[a][b] - (lam * Gdiag[a] if a == b else 0)
            J[a][14] = -Gx[a]
            for i in range(4):
                J[a][15 + i] = -Gk[i][a]
        for b in range(14):
            J[14][b] = 2 * Gx[b]
            for i in range(4):
                J[15 + i][b] = Gk[i][b]
        return F, J
    m = [iv.mpf(t) for t in centre]
    Fm, _ = F_and_J(m, False)
    _, JB = F_and_J(B, True)
    Jmid = mp.matrix([[mid(JB[a][b]) for b in range(19)] for a in range(19)])
    Y = Jmid ** -1
    Yiv = [[iv.mpf(Y[a, b]) for b in range(19)] for a in range(19)]
    YF = [sum((Yiv[a][b] * Fm[b] for b in range(19)), iv.mpf(0)) for a in range(19)]
    E = [[(1 if a == b else 0) - sum((Yiv[a][c] * JB[c][b] for c in range(19)), iv.mpf(0)) for b in range(19)] for a in range(19)]
    d = [B[b] - m[b] for b in range(19)]
    K = [m[a] - YF[a] + sum((E[a][b] * d[b] for b in range(19)), iv.mpf(0)) for a in range(19)]
    inside = all(K[a].a > B[a].a and K[a].b < B[a].b for a in range(19))
    ratio = max(mp.mpf(K[a].delta) / mp.mpf(B[a].delta) for a in range(19))
    normE = max(sum(abs(mp.mpf(max(abs(E[a][b].a), abs(E[a][b].b)))) for b in range(19)) for a in range(19))
    check(f"D {s} {name}: Krawczyk K(B) inside int(B)", inside, f"max width ratio {mp.nstr(mp.mpf(ratio), 3)}, ||I - Y J(B)||_inf <= {mp.nstr(mp.mpf(normE), 3)}")
    # c = 0 at the zero: A_{ik} = <X_i x, G k_k> nonsingular on B (strict diagonal dominance after midpoint preconditioning)
    xB = B[:14]
    A = [[sum(((sum((gens[i][a][b] * xB[b] for b in range(14)), iv.mpf(0))) * Gk[kk][a] for a in range(14)), iv.mpf(0)) for kk in range(4)] for i in range(4)]
    Am = mp.matrix([[mid(A[i][kk]) for kk in range(4)] for i in range(4)])
    Ya = Am ** -1
    PA = [[sum((iv.mpf(Ya[i, l]) * A[l][kk] for l in range(4)), iv.mpf(0)) for kk in range(4)] for i in range(4)]
    offn = max(sum(max(abs(PA[i][kk].a), abs(PA[i][kk].b)) for kk in range(4) if kk != i) for i in range(4))
    diagmin = min(PA[i][i].a for i in range(4))
    nonsing = diagmin > offn
    check(f"D {s} {name}: <X_i x, G k_k> nonsingular on B (so c = 0 at the zero)", nonsing,
          f"min diag >= {dec_out(diagmin.a if hasattr(diagmin, 'a') else diagmin, 8, False)}, max off-diag row sum <= {dec_out(offn.b if hasattr(offn, 'b') else offn, 3, True)}")
    # invariance identities used for c = 0: grad h . X_i x = 0 and Gx . X_i x = 0 (checked on the box, as intervals)
    _, gB, _, GxB, _ = h_parts(xB, c0, c6, want_hess=False)
    inv_ok = True
    for i in range(4):
        Xx = [sum((gens[i][a][b] * xB[b] for b in range(14)), iv.mpf(0)) for a in range(14)]
        t1 = sum((gB[a] * Xx[a] for a in range(14)), iv.mpf(0))
        t2 = sum((GxB[a] * Xx[a] for a in range(14)), iv.mpf(0))
        inv_ok &= (t1.a <= 0 <= t1.b) and (t2.a <= 0 <= t2.b) and t1.delta < mp.mpf(10) ** -50
    check(f"D {s} {name}: grad h . X_i x and Gx . X_i x enclose 0 on B (invariance; consistent with h's symmetry)", inv_ok)
    # enclosure of lambda and of 4h at the zero
    return inside and nonsing, B, K, c0, c6

def R0_interval(s, xB, T=None):
    """R0~(x/|x|_G) over an x-box in the chart of V_j (real interval arithmetic on (re, im) pairs)."""
    T = T or T3
    c = consts[s]; jj = T["j"]; nn = T["n"]; NmT = T["Nm"]; GT = T["Gdiag"]
    w = [(iv.sqrt(ivq(NmT[a])) * xB[a], iv.sqrt(ivq(NmT[a])) * xB[nn + a]) for a in range(nn)]
    nrm2 = sum((GT[r] * xB[r] ** 2 for r in range(2 * nn)), iv.mpf(0))
    B6 = [(qf(T["RM"][Mi], xB), qf(T["IM"][Mi], xB)) for Mi in range(13)]
    tot = iv.mpf(0); lam0 = int(2 * jj) * (int(2 * jj) + 2)
    norms = {}
    for kk, g in c["gK"].items():
        K = sp.Rational(kk); g = sp.Rational(g)
        if K == jj or g == 0:
            continue
        L = int(2 * K)
        nK = iv.mpf(0)
        for MiK, M in enumerate(mlist(K)):
            re = iv.mpf(0); im = iv.mpf(0)
            for Mi, M1 in enumerate(mlist(6)):
                mm = M - M1
                if abs(mm) > jj:
                    continue
                sg, sq = cg_rational_square(6, M1, jj, mm, K, M)
                if sg == 0:
                    continue
                cgv = signed_sqrt_iv(sg, sq)
                br, bi = B6[Mi]; wr, wi = w[T["ms"].index(mm)]
                re += cgv * (br * wr - bi * wi); im += cgv * (br * wi + bi * wr)
            nK += re ** 2 + im ** 2
        norms[L] = ivq(g) * nK / nrm2 ** 3
        tot += norms[L] / (L * (L + 2) - lam0)
    return -tot, norms

res = {}
for bname, box in inp["boxes"]["boxes"].items():
    name, s = bname.split()
    if PLANT and bname != "O1 R4":
        continue
    ok, B, K, c0, c6 = krawczyk(s, name, box)
    R0, norms = R0_interval(s, B[:14])
    sign = "-" if R0.b < 0 else ("+" if R0.a > 0 else "undecided")
    check(f"C.4 {s} {name}: R0~ has constant sign over the box", sign in ("+", "-"), str(encl(R0, 30)))
    lamK = K[14]
    print(f"{s} {name}: certified={ok}  lambda (= 4h at the zero, from K(B)) in {encl(lamK, 40)}")
    print(f"      R0~ over box in {encl(R0, 45)}, width {mp.nstr(mp.mpf(R0.delta), 3)}  sign {sign}")
    lo_s, hi_s = encl(R0, 45)
    check(f"C.4 {s} {name}: reported decimal enclosure contains the interval and carries its sign",
          mp.mpf(lo_s) <= mp.mpf(R0.a) and mp.mpf(hi_s) >= mp.mpf(R0.b) and ((sign == '-') == hi_s.startswith('-')))
    res.setdefault(s, {})[name] = {
        "Krawczyk certified (unique zero of the slice system in the box)": bool(ok),
        "critical (c = 0 at the certified zero)": bool(ok),
        "R0~ enclosure over the box": encl(R0, 45),
        "R0~ enclosure width": mp.nstr(mp.mpf(R0.delta), 5),
        "sign": sign,
    }
# ---- independent 130-digit evaluation (this file's real-chart code) at the census and R2 points,
#      compared with the exact values of s3_points.py
pts_exact = json.load(open(os.path.join(OUT, "C4_points.json")))
if PLANT:
    print("failures:", n_fails()); sys.exit(0)
T72 = build(sp.Rational(7, 2))
def chart_point(T, comp):
    x = [iv.mpf(0)] * (2 * T["n"])
    for k, val in comp.items():
        v = sp.sympify(val, locals={"I": sp.I, "sqrt": sp.sqrt})
        a = T["ms"].index(sp.Rational(k))
        pv = v / sp.sqrt(T["Nm"][a])                   # p_m = w_m / sqrt(N_m)
        x[a] = iv.mpf(str(sp.N(sp.re(pv), 150))); x[T["n"] + a] = iv.mpf(str(sp.N(sp.im(pv), 150)))
    return x
worst = mp.mpf(0)
for s, T, pts in (("R4", T3, inp["census"]["points"]), ("R5", T3, inp["census"]["points"]), ("R2", T72, inp["R2 points"]["points"])):
    for name, comp in pts.items():
        ex = pts_exact[s][name].get("R0~ exact (V = R = 1)")
        if ex is None:
            continue
        R0p, _ = R0_interval(s, chart_point(T, comp), T)
        exv = sp.Rational(ex)
        d = max(abs(mp.mpf(R0p.a) - mp.mpf(exv.p) / exv.q), abs(mp.mpf(R0p.b) - mp.mpf(exv.p) / exv.q))
        worst = max(worst, d)
        res.setdefault("_hp_points", {}).setdefault(s, {})[name] = encl(R0p, 60)
check("C.4 exact R0~ at all census/R2 points == independent 130-digit real-chart evaluation (to 1e-100)", worst < mp.mpf(10) ** -100, mp.nstr(worst, 3))

if not PLANT:
    json.dump(res, open(os.path.join(OUT, "C4_boxes.json"), "w"), indent=1)
print("failures:", n_fails())
