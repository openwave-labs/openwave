"""Item 5b: H_u on the tangent space of a dimension-2 class at an interior critical orbit.

Tangent space of the class sphere at u: {e in W : Re<u, e> = 0}, 3 real dimensions, with
basis t1 = i u, t2 = d/dx u_hat, t3 = d/dy u_hat, where u_hat(x, y) = u(z)/|u(z)| in the
item-2 chart u(z) = w0 + z w1, z = x + i y.  Normalisation: each tangent t is divided by
sqrt(Re<t, t>) (unprojected); H_u(t, t') = t^T M_u t'.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
import core, hess
from core import report, IDX
from hess import simp

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
R = sp.Rational
x, y = sp.symbols("x y", real=True)


def vec(d):
    c = [sp.Integer(0)] * 7
    for m, val in d.items():
        c[IDX[m]] += sp.sympify(val)
    return c


CASES = [  # (label, orbit, class, w0, w1, z0, primary?)
    ("Wmin in W", "Wmin", "W", vec({0: 1}), vec({3: 1, -3: 1}), sp.sqrt(R(10, 23)), True),
    ("Umin in U", "Umin", "U", vec({0: 1}), vec({2: 1, -2: 1}), sp.I * sp.sqrt(R(5, 6)), True),
    ("L12circle in L12", "L12circle", "L12", vec({1: 1}), vec({-2: 1}), R(1, 2), True),
    ("L23circle in L23", "L23circle", "L23", vec({2: 1}), vec({-3: 1}), sp.sqrt(R(12, 13)), True),
    ("hex in L33", "hex", "L33", vec({3: 1}), vec({-3: 1}), sp.Integer(1), False),
    ("hex in U", "hex", "U", vec({0: 1}), vec({2: 1, -2: 1}), sp.sqrt(R(3, 10)), False),
    ("oct in L22", "oct", "L22", vec({2: 1}), vec({-2: 1}), sp.Integer(-1), False),
    ("oct in W", "oct", "W", vec({0: 1}), vec({3: 1, -3: 1}), sp.I * sp.sqrt(R(2, 5)), False),
]
out = {}
for label, orb, cls, w0, w1, z0, primary in CASES:
    zc = x + sp.I * y
    uz = [a + zc * b for a, b in zip(w0, w1)]
    nrm = sp.sqrt(sum(sp.expand(q * sp.conjugate(q)) for q in uz))
    uhat_real = sp.Matrix([v / nrm for v in core.to_real(uz)])
    sub = {x: sp.re(z0), y: sp.im(z0)}
    c0 = [simp(q.subs(sub)) for q in uz]
    cu, r = hess.unit_real(c0)
    ok &= report("%s: chart point is the unit representative" % label,
                 (uhat_real.subs(sub).applyfunc(simp) - r).applyfunc(simp) == sp.zeros(14, 1))
    Mu = hess.M_u(r)
    og = hess.O_gens(cu)
    B = hess.N_basis(r, og)
    Bm = sp.Matrix.hstack(*B)
    PN = (Bm * (Bm.T * Bm).inv() * Bm.T).applyfunc(simp)
    tang = {"i u": hess.cvec_to_real([sp.I * q for q in cu]),
            "d/dx u_hat": uhat_real.diff(x).subs(sub).applyfunc(simp),
            "d/dy u_hat": uhat_real.diff(y).subs(sub).applyfunc(simp)}
    T = sp.Matrix.hstack(*tang.values())
    ok &= report("%s: the 3 tangents are independent and lie in T_u" % label,
                 T.rank(simplify=True) == 3 and all(simp((r.T * t)[0, 0]) == 0 for t in tang.values()))
    rec = {"orbit": orb, "class": cls, "chart_point": str(z0), "n_tangent_directions": 3,
           "normalisation": "t / sqrt(Re<t,t>), unprojected", "null": {}, "transverse": {}}
    trans = []
    print("\n==", label, "(z0 = %s)" % z0)
    for k, t in tang.items():
        p = (PN * t).applyfunc(simp)
        zero = p == sp.zeros(14, 1)
        tn = (t / sp.sqrt(simp((t.T * t)[0, 0]))).applyfunc(simp)
        if zero:
            # identify inside O_u: t = sum a_j d_j
            names = list(og)
            Om = sp.Matrix.hstack(*og.values())
            a = sp.symbols("a0:4")
            sol = sp.solve(list(Om * sp.Matrix(a) - t), a, dict=True)
            Hval = simp((tn.T * Mu * tn)[0, 0])
            s0 = {kk: simp(vv) for kk, vv in sol[0].items()} if sol else {}
            combo = " + ".join("(%s)(%s)" % (s0.get(a[j], a[j]), names[j]) for j in range(4) if s0.get(a[j], a[j]) != 0)
            print("  %-11s projection onto N_u = 0; it is %s ; H_u(unit t, unit t) = %s" % (k, combo, Hval))
            ok &= report("%s: orbit-null control H_u(%s) = 0" % (label, k), Hval == 0)
            rec["null"][k] = {"O_u_combination": combo, "H_u_unit": str(Hval)}
        else:
            print("  %-11s projection onto N_u nonzero (|P_N t|^2/|t|^2 = %s)" % (
                k, simp((p.T * p)[0, 0] / (t.T * t)[0, 0])))
            trans.append((k, tn, (p / sp.sqrt(simp((p.T * p)[0, 0]))).applyfunc(simp)))
    names = [k for k, _, _ in trans]
    Hm = sp.Matrix(len(trans), len(trans), lambda i, j: simp((trans[i][1].T * Mu * trans[j][1])[0, 0]))
    Gm = sp.Matrix(len(trans), len(trans), lambda i, j: simp((trans[i][1].T * trans[j][1])[0, 0]))
    Hp = sp.Matrix(len(trans), len(trans), lambda i, j: simp((trans[i][2].T * Mu * trans[j][2])[0, 0]))
    print("  transverse tangents:", names)
    print("  H_u matrix (unit unprojected tangents):", Hm.tolist())
    print("  Gram matrix Re<t_i, t_j>:            ", Gm.tolist())
    # consistency: H_u(t, t') is unchanged by projecting to N_u (M_u kills O_u); compare with
    # H on the projected, renormalised vectors scaled back by the projection lengths
    scal = [sp.sqrt(simp(((PN * tt).T * (PN * tt))[0, 0])) for _, tt, _ in trans]
    Hp_back = sp.Matrix(len(trans), len(trans), lambda i, j: simp(Hp[i, j] * scal[i] * scal[j]))
    ok &= report("%s: H_u(t, t') = H_u(P_N t, P_N t') exactly" % label, (Hp_back - Hm).applyfunc(simp) == sp.zeros(len(trans)))
    rec["transverse"] = {"directions": names, "H": [[str(v) for v in row] for row in Hm.tolist()],
                         "Gram": [[str(v) for v in row] for row in Gm.tolist()],
                         "H_on_projected_unit_vectors": [[str(v) for v in row] for row in Hp.tolist()]}
    rec["primary"] = primary
    out[label] = rec

json.dump(out, open(os.path.join(HERE, "out", "item5b.json"), "w"), indent=1)
print("ITEM5B", "ALLPASS" if ok else "SOMEFAIL")
