"""Build mutated copies of the solver's scripts under ref_runs/<mutation>/ (solver_copy/ is never
touched). Each textual mutation must match EXACTLY once, else this script fails (so a mutation can
not silently not-apply). Also writes the new stress-test script for sympy's rank(simplify=True).
"""
import pathlib
import shutil

HERE = pathlib.Path(__file__).parent
SRC = HERE / "solver_copy"
RUNS = HERE / "ref_runs"

MUTS = {
    "mut_gen": [("solver_lib.py", "Q2 = Quat(PHI * HALF, PHI_INV * HALF, HALF, 0)", "Q2 = Quat(HALF, HALF, HALF, -HALF)")],
    "mut_cg6": [("solver_lib.py",
                 "    return sp.nsimplify(clebsch_gordan(sp.S(j1), sp.S(j2), sp.S(J), sp.S(m1), sp.S(m2), sp.S(M)))",
                 "    return (-1 if J == 6 else 1) * sp.nsimplify(clebsch_gordan(sp.S(j1), sp.S(j2), sp.S(J), sp.S(m1), sp.S(m2), sp.S(M)))")],
    "mut_cgm": [("solver_lib.py",
                 "    return sp.nsimplify(clebsch_gordan(sp.S(j1), sp.S(j2), sp.S(J), sp.S(m1), sp.S(m2), sp.S(M)))",
                 "    return sp.Integer(-1) ** int(sp.floor(sp.S(m1))) * sp.nsimplify(clebsch_gordan(sp.S(j1), sp.S(j2), sp.S(J), sp.S(m1), sp.S(m2), sp.S(M)))")],
    "mut_theta": [("solver_lib.py", "            ph = sp.Integer(-1) ** int(m)\n", "            ph = sp.Integer(1)\n")],
    "mut_T": [("solver_lib.py", "        T[int(-m + 3), a] = sp.Integer(-1) ** int(m)", "        T[int(-m + 3), a] = 1")],
    "mut_P": [("solver_lib.py",
               "    rows = load_results()[\"group\"][\"item12\"][key][\"P_orth_rows\"]\n    return sp.Matrix([[sp.sympify(e) for e in r] for r in rows])",
               "    d = 4 if key == \"d=4\" else 3\n    return sp.diag(*([1] * d + [0] * (7 - d)))   # MUTATION: non-invariant projector of the same rank")],
    "mut_item19theta": [("solver_item19.py", "th = [sp.Integer(-1) ** abs(m) * b[-m + 3] for m in MS]", "th = [b[-m + 3] for m in MS]")],
    "mut_fmon": [("solver_stage2_roots.py", "- 33*z**8", "- 32*z**8")],
    "mut_quadcoef": [("solver_crosscheck_quadrature.py",
                      "    cK = {K: complex(sp.N((P * A_op_matrix(R[K], K)).trace(), 30)) for K in range(7)}",
                      "    cK = {K: complex(sp.N((P * A_op_matrix(R[K], K)).trace(), 30)) * (1.01 if K == 6 else 1) for K in range(7)}")],
}

for name, edits in MUTS.items():
    dst = RUNS / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in SRC.iterdir():
        if f.suffix in (".py", ".json"):
            shutil.copy(f, dst / f.name)
    for fname, old, new in edits:
        p = dst / fname
        txt = p.read_text()
        n = txt.count(old)
        assert n == 1, (name, fname, n)
        p.write_text(txt.replace(old, new))
    print("built", name)

# stress test of sympy rank(simplify=True), the solver's exact-rank tool (items 14, 20, 21)
dst = RUNS / "mut_rank"
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True)
for f in SRC.iterdir():
    if f.suffix in (".py", ".json"):
        shutil.copy(f, dst / f.name)
(dst / "stress_rank.py").write_text('''"""Feed the solver's rank path inputs whose true rank is known."""
import sympy as sp
from solver_lib import M_ab, N_ab

z, X, Y = sp.symbols("z X Y")


def majorana(vec, j):
    return sp.expand(sum(sp.Integer(-1) ** (j - m) * sp.sqrt(sp.binomial(2 * j, j + m)) * vec[m + j] * z ** (j - m)
                         for m in range(-j, j + 1)))


def homog(f, deg):
    P = sp.Poly(f, z)
    return sp.expand(sum(c * X ** e[0] * Y ** (deg - e[0]) for e, c in P.terms()))


def transvectant_rank(FR):
    rows = []
    for m in range(-3, 4):
        e = [0] * 7
        e[m + 3] = 1
        Fv = homog(majorana(e, 3), 6)
        Jv = sp.expand(sp.diff(FR, X) * sp.diff(Fv, Y) - sp.diff(FR, Y) * sp.diff(Fv, X))
        PJ = sp.Poly(Jv, X, Y)
        rows.append([PJ.coeff_monomial(X ** k * Y ** (16 - k)) for k in range(17)])
    return sp.Matrix(rows).T.rank(simplify=True)


# (a) the solver's own form: expect 7
fmon = (z**12 - 22*sp.sqrt(5)*z**10/5 - 33*z**8 + 44*sp.sqrt(5)*z**6/5 - 33*z**4 - 22*sp.sqrt(5)*z**2/5 + 1)
print("rank, solver form (expect 7):", transvectant_rank(homog(fmon, 12)))
# (b) a perfect square of a sextic with surd coefficients: true rank 6
w = [1, sp.sqrt(2), 0, sp.sqrt(3), sp.I, 0, sp.sqrt(5)]
sq = sp.expand(majorana(w, 3) ** 2)
print("rank, perfect square F_w^2 (true rank 6):", transvectant_rank(homog(sq, 12)))
# (c) square with a denestable radical: sqrt(5+2 sqrt6) = sqrt2 + sqrt3
w2 = [1, sp.sqrt(5 + 2 * sp.sqrt(6)), 0, sp.sqrt(2) + sp.sqrt(3), 0, 0, 1]
sq2 = sp.expand(majorana(w2, 3) ** 2)
print("rank, square with nested radical (true rank 6):", transvectant_rank(homog(sq2, 12)))
# (d) three columns with c3 = sqrt(5+2sqrt6) c1 - (sqrt2+sqrt3) c1 + c2 - c2: true rank 1 (c3 = 0 in disguise) plus c2
c1 = sp.Matrix([1, sp.sqrt(2), sp.sqrt(3), 2])
c2 = sp.Matrix([sp.sqrt(5), 1, 0, sp.sqrt(7)])
c3 = (sp.sqrt(5 + 2 * sp.sqrt(6)) - sp.sqrt(2) - sp.sqrt(3)) * c1 + c2
print("rank, [c1, c2, c3 = 0*c1 + c2 in disguise] (true rank 2):", sp.Matrix.hstack(c1, c2, c3).rank(simplify=True))
# (e) item-21 style: maps with a planted dependency
a = list(sp.symbols("u0:7")); b = list(sp.symbols("ub0:7"))
M0, M6 = M_ab(a, b, 0), M_ab(a, b, 6)
dep = [sp.expand(x + sp.sqrt(91) * y) for x, y in zip(M0, M6)]
keys = sorted({(m, mono) for mp in (M0, M6, dep) for m in range(7) for mono in (sp.Poly(mp[m], *a, *b).monoms() if mp[m] != 0 else [])})
row = lambda mp: [sp.Poly(mp[m], *a, *b).coeff_monomial(mono) if mp[m] != 0 else 0 for (m, mono) in keys]
print("rank [M0, M6, M0 + sqrt91 M6] (true rank 2):", sp.Matrix([row(M0), row(M6), row(dep)]).rank(simplify=True))
''')
print("built mut_rank")
