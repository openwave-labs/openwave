"""one-off edit of s4_boxes.py: generalize the chart tables to any j and add the 130-digit point check."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "s4_boxes.py"); s = open(p).read()
old = s[s.index("j = sp.Integer(3); ms = mlist(j); n = 7"):s.index("def qf(D, x):")]
new = '''def signed_sqrt_iv(sign, sq):
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

'''
s = s.replace(old, new)
s = s.replace("def mv(D, x):\n    out = [iv.mpf(0)] * 14", "def mv(D, x):\n    out = [iv.mpf(0)] * len(x)")
old = s[s.index("def R0_interval(s, xB):"):s.index("res = {}\nfor bname")]
new = '''def R0_interval(s, xB, T=None):
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

'''
s = s.replace(old, new)
old = "if not PLANT:\n    json.dump(res"
new = '''# ---- independent 130-digit evaluation (this file's real-chart code) at the census and R2 points,
#      compared with the exact values of s3_points.py
pts_exact = json.load(open(os.path.join(OUT, "C4_points.json")))
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
        res.setdefault("_hp_points", {}).setdefault(s, {})[name] = mp.nstr(mid(R0p), 60)
check("C.4 exact R0~ at all census/R2 points == independent 130-digit real-chart evaluation (to 1e-100)", worst < mp.mpf(10) ** -100, mp.nstr(worst, 3))

if not PLANT:
    json.dump(res'''
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("patched")
