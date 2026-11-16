"""Independent numeric cross-check of B.1/B.2 (50 digits), on Condon-Shortley components directly.

h(w) = <w(x)w, H w(x)w>, H = sum_J q_J Pi_J on V_3 (x) V_3.  Second derivative along (d, e):
2 Re<d(x)w + w(x)d, H(e(x)w + w(x)e)> + 2 Re<d(x)e + e(x)d, H w(x)w>.
Tangent space K_u from i w and -i J_a w (derived spin matrices, not inputs.json).
Signature on N_u from the projected M-hat; slow roots from eigenvalues of S = i M-hat.
Compared against the exact results of s04 (blockB.json)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import Checker, planted, cg
import sympy as sp
import mpmath as mp

mp.mp.dps = 50
chk = Checker()
OUT = os.path.join(HERE, "out")
inp = json.load(open(os.path.join(HERE, "..", "inputs.json")))
A = json.load(open(os.path.join(OUT, "blockA.json")))
B = json.load(open(os.path.join(OUT, "blockB.json")))
MS = list(range(3, -4, -1))
n = 7
PI = {}
for J in (6, 4, 2, 0):
    P = mp.matrix(49, 49)
    for M in range(-J, J + 1):
        v = mp.matrix(49, 1)
        for i1, m1 in enumerate(MS):
            m2 = M - m1
            if abs(m2) <= 3:
                v[i1 * 7 + MS.index(m2)] = mp.mpf(sp.N(cg(3, m1, 3, m2, J, M), 70))
        P += v * v.T
    PI[J] = P

def kron(a, b):
    return mp.matrix([a[i] * b[k] for i in range(7) for k in range(7)])

def ip(a, b):
    return mp.fsum(mp.conj(a[i]) * b[i] for i in range(len(a)))

def spin():
    Jp = mp.matrix(7, 7)
    for a, m in enumerate(MS):
        if m < 3: Jp[a - 1, a] = mp.sqrt((3 - m) * (3 + m + 1))
    Jm = Jp.T
    return (Jp + Jm) / 2, (Jp - Jm) / (2j), mp.diag([mp.mpf(m) for m in MS])

def analyse(qJ, wsym):
    w = mp.matrix([mp.mpc(*[mp.mpf(str(t)) for t in sp.N(sp.sympify(x), 70).as_real_imag()]) for x in wsym])
    nw = mp.sqrt(mp.re(ip(w, w)))
    u = w / nw
    H = sum((mp.mpf(sp.Rational(q).p) / sp.Rational(q).q) * PI[J] for J, q in qJ.items())
    W = kron(u, u); HW = H * W
    hu = mp.re(ip(W, HW))
    basis = []
    for a in range(7):
        e = mp.matrix(7, 1); e[a] = 1; basis.append(e)
    for a in range(7):
        e = mp.matrix(7, 1); e[a] = 1j; basis.append(e)
    grad = [2 * mp.re(ip(kron(d, u) + kron(u, d), HW)) for d in basis]
    uvec = [mp.re(u[a]) for a in range(7)] + [mp.im(u[a]) for a in range(7)]
    crit = max(abs(grad[a] - 4 * hu * uvec[a]) for a in range(14))
    Hs = mp.matrix(14, 14)
    for a, d in enumerate(basis):
        Dd = kron(d, u) + kron(u, d)
        for b, e in enumerate(basis):
            Ee = kron(e, u) + kron(u, e)
            X = kron(d, e) + kron(e, d)
            Hs[a, b] = 2 * mp.re(ip(Dd, H * Ee)) + 2 * mp.re(ip(X, HW))
    Mh = Hs - 4 * hu * mp.eye(14)
    Jc = mp.matrix(14, 14)
    for a in range(7): Jc[a, a + 7] = -1; Jc[a + 7, a] = 1
    S = Jc * Mh
    # K_u
    Jx, Jy, Jz = spin()
    tang = [1j * u] + [(-1j) * (Ja * u) for Ja in (Jx, Jy, Jz)]
    Kr = mp.matrix(14, 4)
    for c, t in enumerate(tang):
        for a in range(7): Kr[a, c] = mp.re(t[a]); Kr[a + 7, c] = mp.im(t[a])
    sv = mp.svd_r(Kr, compute_uv=False)
    k = sum(1 for i in range(4) if sv[i] > mp.mpf(10) ** -30)
    sig = Kr.T * Jc * Kr
    svs = mp.svd_r(sig, compute_uv=False)
    r = k - sum(1 for i in range(4) if svs[i] > mp.mpf(10) ** -30)
    # projector onto N_u: complement of u and K_u
    U, Sv, V = mp.svd_r(mp.matrix([[Kr[a, c] for c in range(4)] + [uvec[a]] for a in range(14)]), full_matrices=True)
    rk = sum(1 for i in range(5) if Sv[i] > mp.mpf(10) ** -30)
    # orthonormal complement: last 14-rk left singular vectors
    N = mp.matrix(14, 14 - rk)
    for c in range(14 - rk):
        for a in range(14): N[a, c] = U[a, rk + c]
    MN = N.T * Mh * N
    ev = mp.eigsy((MN + MN.T) / 2)[0]
    tol = mp.mpf(10) ** -30
    signature = [sum(1 for i in range(len(ev)) if ev[i] < -tol), sum(1 for i in range(len(ev)) if abs(ev[i]) <= tol),
                 sum(1 for i in range(len(ev)) if ev[i] > tol)]
    eS = mp.eig(S, left=False, right=False)
    zeros = sum(1 for e in eS if abs(e) < mp.mpf(10) ** -20)
    nus = sorted([e * e for e in eS if abs(e) >= mp.mpf(10) ** -20], key=lambda z: (mp.re(z), mp.im(z)))
    return {"S": S, "Mh": Mh, "crit_resid": crit, "Q": hu, "k": k, "r": r, "signature": signature, "zeros": zeros, "nus": nus}

for slot, item in (("R4", "B.1"), ("R5", "B.2")):
    qJ = {int(k[2:]): v for k, v in A["A.1"][slot].items() if k.startswith("q_")}
    for name, comp in inp["census"]["points"].items():
        w = [comp.get(str(m), "0") for m in MS]
        o = analyse(qJ, w)
        ex = B[item][name]
        # exact slow roots with multiplicity, each nu counted twice (for +xi and -xi)
        exn = []
        if ex["roots"]:
            for rr in ex["roots"]:
                for root in sp.Poly(sp.sympify(rr["factor"]), sp.Symbol("nu")).all_roots():
                    exn += [complex(sp.N(root, 40))] * (2 * rr["multiplicity"])
        exn.sort(key=lambda z: (z.real, z.imag))
        num = [complex(z) for z in o["nus"]]
        num.sort(key=lambda z: (round(z.real, 10), z.imag))
        exn.sort(key=lambda z: (round(z.real, 10), z.imag))
        maxd = max([abs(a - b) for a, b in zip(num, exn)] + [0]) if len(num) == len(exn) else float("inf")
        ok = (o["crit_resid"] < mp.mpf(10) ** -40 and abs(o["Q"] - mp.mpf(sp.Rational(ex["Q"]).p) / sp.Rational(ex["Q"]).q) < mp.mpf(10) ** -40
              and o["k"] == ex["k"] and o["r"] == ex["r_u"] and o["signature"] == ex["signature"]
              and o["zeros"] == ex["zero multiplicity"] and maxd < 1e-12)
        print(f"{slot} {name}: crit resid {mp.nstr(o['crit_resid'], 3)}  Q {mp.nstr(o['Q'], 20)}  k {o['k']}  r {o['r']}  sig {o['signature']}  "
              f"zeros {o['zeros']}  max|nu_num - nu_exact| {maxd:.1e}")
        chk(f"{slot} {name}: independent numeric (CS coordinates) agrees with exact s04 results", ok)
        for kr in ex["krein"]:
            t2 = -mp.mpmathify(sp.N(sp.sympify(kr["nu"]), 70))
            Uk, Sk, Vk = mp.svd_r(o["S"] * o["S"] + t2 * mp.eye(14))
            idx = [i for i in range(14) if Sk[i] < mp.mpf(10) ** -25]
            Bk = mp.matrix(14, len(idx))
            for c, i in enumerate(idx):
                for a in range(14): Bk[a, c] = Vk[i, a]
            F = Bk.T * o["Mh"] * Bk; F = (F + F.T) / 2
            ev = mp.eigsy(F)[0]
            sgn = "+" if all(ev[i] > 0 for i in range(len(idx))) else ("-" if all(ev[i] < 0 for i in range(len(idx))) else "indefinite")
            chk(f"{slot} {name}: Krein sign at nu={kr['nu_num'][:12]} independently = {sgn}", sgn == kr["Krein"] and len(idx) == kr["dim ker(S^2+tau^2)"], quiet=True)
# planted: the comparison must fail when the exact record is altered
o = analyse({6: A["A.1"]["R4"]["q_6"], 4: A["A.1"]["R4"]["q_4"], 2: A["A.1"]["R4"]["q_2"], 0: A["A.1"]["R4"]["q_0"]}, ["0", "0", "1", "0", "0", "0", "0"])
planted("v1 in R4 compared against signature [4, 1, 5] instead of the computed one", lambda: o["signature"] == [4, 1, 5])
planted("criticality residual at a non-critical point (v3 + v1)", lambda: analyse({6: "1288/1287", 4: "35/33", 2: "14/9", 0: "7/3"}, ["1", "0", "1", "0", "0", "0", "0"])["crit_resid"] < mp.mpf(10) ** -40)
chk.summary()
