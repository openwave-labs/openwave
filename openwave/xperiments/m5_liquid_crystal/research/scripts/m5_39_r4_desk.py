"""M5.39 R4: the desk checks. No field of the card is read.

(a) The moment bound. From mu_p - mu_n = (2/3) M_N Lambda_V with Lambda_V <= Lambda and
    Delta - N = 3 / (2 Lambda): (mu_p - mu_n)(Delta - N) <= M_N, and the three printed numbers.
(b) The first-order cranking statement. With H Z = 0, the response equation
    (H - w^2 K) Phi = i w K Z is solved by Phi = -(i / w) Z: symbolically, and numerically on a
    random positive pair with a planted null vector.
(c) The Finkelstein-Rubinstein filter for B = 2 to 8 and 12, as a conditional table: IF the sign is
    the winding field's alone, the allowed spin and isospin of a rigid configuration follow from its
    rational map by the published rule (Krusch 2003, Theorem 3.10, in the form of Manko, Manton and
    Wood 2007, eqs. 34 to 36): a rotation by t2 about n2 that the map turns into an isorotation by
    t1 about n1, with the base point R(z_{-n2}) = R_{-n1}, constrains the states by
    exp(i t2 n2.L) exp(i t1 n1.K) Psi = (-1)^N Psi, N = B (B t2 - t1) / 2 pi.
    Here the isorotation is found numerically from the map itself, N is checked to be an integer,
    and the allowed (J, I) are the common eigenspace of the generators' constraints.

    Known answers, each a row that can fail: the constraint equations the 2007 paper prints for
    B = 2 to 8 (same allowed states from our rule as from the printed operators), the states it
    names, and the ground state and two excited states of the 2003 table for B = 2 to 8 and 12.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.linalg import expm

HALF = sp.Rational(1, 2)
E1, E2, E3 = np.eye(3)
N111 = np.ones(3) / np.sqrt(3)


# ------------------------------------------------------------------ (a)
def moment_bound():
    lam, lam_v, m = sp.symbols("Lambda Lambda_V M_N", positive=True)
    product = sp.simplify(sp.Rational(2, 3) * m * lam_v * 3 / (2 * lam))  # (mu_p - mu_n)(Delta - N)
    bound_holds = sp.simplify(product.subs(lam_v, lam) - m) == 0 and sp.simplify(sp.diff(product, lam_v)) .is_positive
    m_n, m_p, mu = 938.918, 938.272, 4.706  # the nucleon average, the proton, the measured mu_p - mu_n
    rows = [
        dict(quantity="mu_V at the bound, gap 293 MeV (mu_N)", printed=3.20, returned=m_n / 293.0, half=0.005),
        dict(quantity="the same with the proton mass", printed=3.20, returned=m_p / 293.0, half=0.005),
        dict(quantity="gap that 4.706 mu_N would need (MeV)", printed=199.5, returned=m_n / mu, half=0.05),
        dict(quantity="moment missing at 293 MeV (mu_N)", printed=1.50, returned=mu - m_n / 293.0, half=0.005),
        dict(quantity="moment missing at the Breit-Wigner gap 1232 - 938.272 MeV (mu_N)", printed=1.51, returned=mu - m_n / (1232.0 - m_p), half=0.005),
    ]
    for r in rows:
        r["label"] = "REPRODUCED" if abs(r["returned"] - r["printed"]) <= r.pop("half") else "DIFFERS"
    return dict(statement="(mu_p - mu_n)(Delta - N) = M_N Lambda_V / Lambda <= M_N for Lambda_V <= Lambda", algebra_holds=bool(bound_holds),
                conditions=["the isovector moment is (2/3) M_N Lambda_V, i.e. the magnetic current is the minimally gauged rigid-rotor current",
                            "Lambda_V <= Lambda, the isospin inertia", "the gap is the rigid rotor's, 3 / (2 Lambda)"],
                inputs=dict(M_N=m_n, m_p=m_p, mu_p_minus_mu_n=mu), rows=rows,
                passed=bool(bound_holds) and all(r["label"] == "REPRODUCED" for r in rows))


# ------------------------------------------------------------------ (b)
def cranking():
    n = 4
    H = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"h{min(i, j)}{max(i, j)}"))
    K = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"k{min(i, j)}{max(i, j)}"))
    Z = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"z{i}"))
    w = sp.Symbol("omega", positive=True)
    phi = -sp.I / w * Z
    # with H Z = 0 imposed, the residual of the response equation must vanish identically
    residual = (H - w ** 2 * K) * phi - sp.I * w * K * Z + sp.I / w * (H * Z)
    symbolic = all(sp.simplify(x) == 0 for x in residual)
    rng = np.random.default_rng(39)
    worst, unique = 0.0, True
    for _ in range(5):
        m = 12
        a = rng.normal(size=(m, m))
        kin = a @ a.T + m * np.eye(m)
        z = rng.normal(size=m)
        b = rng.normal(size=(m, m - 1))
        proj = np.eye(m) - np.outer(z, z) / (z @ z)
        hess = proj @ (b @ b.T) @ proj  # positive semidefinite with the planted null vector z
        om = 0.37 + rng.random()
        op = hess - om ** 2 * kin
        sol = np.linalg.solve(op.astype(complex), 1j * om * kin @ z)
        worst = max(worst, float(np.abs(sol + 1j / om * z).max() / np.abs(z).max()))
        unique &= abs(np.linalg.det(op)) > 1e-9
    return dict(statement="(H - w^2 K) Phi = i w K Z with H Z = 0 is solved by Phi = -(i / w) Z, a pure zero mode",
                symbolic_residual_zero=bool(symbolic), numeric_largest_difference=worst, operator_invertible_in_every_trial=bool(unique),
                passed=bool(symbolic) and worst < 1e-10 and bool(unique))


# ------------------------------------------------------------------ (c)
def spin_matrices(j):
    d = int(round(2 * j)) + 1
    m = np.array([j - k for k in range(d)])
    jp = np.zeros((d, d), dtype=complex)
    for k in range(1, d):
        jp[k - 1, k] = np.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    return [(jp + jp.conj().T) / 2, (jp - jp.conj().T) / 2j, np.diag(m).astype(complex)]


def rotation_operator(j, vec):
    jx, jy, jz = spin_matrices(j)
    return expm(1j * (vec[0] * jx + vec[1] * jy + vec[2] * jz))


def rotation_matrix(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


class RationalMap:
    """R = p / q of degree B; n_R(x) is the unit vector of R(z), z = (x1 + i x2) / (1 + x3)."""

    def __init__(self, p, q):
        self.B = max(len(p), len(q)) - 1
        self.p = np.concatenate((np.zeros(self.B + 1 - len(p)), np.asarray(p, complex)))
        self.q = np.concatenate((np.zeros(self.B + 1 - len(q)), np.asarray(q, complex)))

    def n(self, x):
        x = np.atleast_2d(x)
        north = x[:, 2] > -0.5
        u = np.where(north, x[:, 0] + 1j * x[:, 1], 1 - x[:, 2])
        v = np.where(north, 1 + x[:, 2], x[:, 0] - 1j * x[:, 1])
        powers = np.stack([u ** (self.B - k) * v ** k for k in range(self.B + 1)], axis=1)
        P, Q = powers @ self.p, powers @ self.q
        norm = np.abs(P) ** 2 + np.abs(Q) ** 2
        w = P * np.conj(Q)
        return np.stack((2 * w.real, 2 * w.imag, np.abs(Q) ** 2 - np.abs(P) ** 2), axis=1) / norm[:, None]

    def induced(self, G, rng):
        """The isorotation G' with n_R(G y) = G' n_R(y), and the misfit (zero for a symmetry)."""
        y = rng.normal(size=(400, 3))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        a, b = self.n(y @ G.T), self.n(y)
        u, _, vt = np.linalg.svd(b.T @ a)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        Gp = vt.T @ np.diag([1, 1, d]) @ u.T
        return Gp, float(np.abs(a - b @ Gp.T).max())


def signed_angle(G, axis):
    """The rotation angle of G about the oriented axis (G fixes it), in (-pi, pi]."""
    c = np.clip((np.trace(G) - 1) / 2, -1, 1)
    theta = float(np.arccos(c))
    if theta < 1e-6:  # arccos near one loses half the digits
        return 0.0
    if abs(theta - np.pi) < 1e-6:
        return np.pi
    u = np.array([G[2, 1] - G[1, 2], G[0, 2] - G[2, 0], G[1, 0] - G[0, 1]]) / (2 * np.sin(theta))
    return theta * float(np.sign(u @ axis))


def constraint_from_map(rmap, axis, angle, rng):
    """The published rule applied to one symmetry: (rotation vector, isorotation vector, sign, record)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    Gp, misfit = rmap.induced(rotation_matrix(axis, angle), rng)
    n1 = -rmap.n(-axis)[0]  # the base point R(z_{-n2}) = R_{-n1}
    theta1 = signed_angle(Gp, n1)
    aligned = float(np.abs(Gp @ n1 - n1).max()) if abs(theta1) > 0 else 0.0
    N = rmap.B * (rmap.B * angle - theta1) / (2 * np.pi)
    return dict(space=(angle * axis).tolist(), iso=(theta1 * n1).tolist(), N=float(N), integer=bool(abs(N - round(N)) < 1e-6),
                sign=int((-1) ** int(round(N))), misfit=misfit, axis_misfit=aligned)


def multiplicity(constraints, J, I):
    """The number of states of spin J and isospin I that satisfy every constraint."""
    dim = (int(round(2 * J)) + 1) * (int(round(2 * I)) + 1)
    rows = [np.kron(rotation_operator(J, np.array(c["space"])), rotation_operator(I, np.array(c["iso"]))) - c["sign"] * np.eye(dim)
            for c in constraints]
    s = np.linalg.svd(np.vstack(rows), compute_uv=False)
    return int(np.sum(s < 1e-6)) + (dim - len(s))


def parities(constraints, parity, J, I):
    """Eigenvalues of a parity operator (rotation vector, isorotation vector) on the allowed states."""
    dim = (int(round(2 * J)) + 1) * (int(round(2 * I)) + 1)
    rows = [np.kron(rotation_operator(J, np.array(c["space"])), rotation_operator(I, np.array(c["iso"]))) - c["sign"] * np.eye(dim)
            for c in constraints]
    _, s, vh = np.linalg.svd(np.vstack(rows))
    basis = vh[np.concatenate((s, np.zeros(dim - len(s)))) < 1e-6].conj().T
    P = np.kron(rotation_operator(J, np.array(parity[0], float)), rotation_operator(I, np.array(parity[1], float)))
    eig = np.linalg.eigvals(basis.conj().T @ P @ basis)
    return sorted(int(round(x.real)) for x in eig), float(np.abs(eig.imag).max() + np.abs(np.abs(eig) - 1).max())


def table(B, constraints, jmax, imax):
    half = B % 2 == 1
    js = [HALF * (2 * k + 1) if half else sp.Integer(k) for k in range(int(jmax) + 1)]
    out = {}
    for I in [x for x in js if x <= imax]:
        out[str(I)] = {str(J): multiplicity(constraints, float(J), float(I)) for J in js if J <= jmax}
    return out


def lowest(tab, I, count=2):
    return [J for J, m in tab[I].items() if m > 0][:count]


def poly(*c):
    return list(c)


def configurations():
    s3 = np.sqrt(3)
    pp, pm = np.poly1d([1, 0, 2j * s3, 0, 1]), np.poly1d([1, 0, -2j * s3, 0, 1])
    cfgs = []

    def add(key, B, name, p, q, gens, printed=None, states=None, source="", parity=None, parity_known=None):
        cfgs.append(dict(key=key, B=B, name=name, map=RationalMap(p, q), gens=gens, printed=printed, states=states or {}, source=source,
                         parity=parity, parity_known=parity_known or {}))

    add("B2", 2, "torus, D_inf_h", [1, 0, 0], [1], [(E3, 1.0), (E1, np.pi)],
        printed=[((1.0 * E3), (2.0 * E3), 1), (np.pi * E1, np.pi * E1, -1)],
        states={"mmw_allowed": [("1", "0"), ("0", "1")], "krusch": ("1", "3", "0")}, source="MMW (57), (58)",
        parity=(0 * E3, np.pi * E3), parity_known={("1", "0"): [1], ("0", "1"): [1]})
    add("B3", 3, "tetrahedron, T_d", [s3 * 1j, 0, -1], [1, 0, -s3 * 1j, 0], [(E3, np.pi), (N111, 2 * np.pi / 3)],
        printed=[(np.pi * E3, np.pi * E3, 1), (2 * np.pi / 3 * N111, 2 * np.pi / 3 * N111, 1)],
        states={"mmw_multiplicity": {("1/2", "1/2"): 1, ("3/2", "3/2"): 2}, "krusch": ("1/2", "5/2", "3/2")}, source="MMW (66), (67)",
        parity=(np.pi / 2 * E3, np.pi / 2 * E3), parity_known={("1/2", "1/2"): [1], ("3/2", "3/2"): [-1, 1]})
    add("B4", 4, "cube, O_h", pp.coeffs, pm.coeffs, [(E3, np.pi / 2), (N111, 2 * np.pi / 3)],
        printed=[(np.pi / 2 * E3, np.pi * E1, 1), (2 * np.pi / 3 * N111, 2 * np.pi / 3 * E3, 1)],
        states={"mmw_allowed": [("0", "0"), ("2", "1"), ("4", "0")], "mmw_forbidden": [("2", "0")], "krusch": ("0", "4", "0")}, source="MMW (79), (80)",
        parity=(0 * E3, np.pi * E3), parity_known={("0", "0"): [1], ("4", "0"): [1], ("2", "1"): [-1]})
    a5, b5 = -3.07, 3.94
    add("B5", 5, "D_2d (the minimal-energy map)", [1, 0, 1j * b5, 0, a5, 0], [a5, 0, 1j * b5, 0, 1], [(E3, np.pi), ("scan", np.pi)],
        states={"krusch": ("1/2", "3/2", "1/2")}, source="Krusch, table 2; the map is MMW (87)")
    add("B5_D4h", 5, "D_4h (the variant the 2007 paper quantizes), a = -2", [1, 0, 0, 0, -2.0, 0], [-2.0, 0, 0, 0, 1], [(E3, np.pi / 2), (E1, np.pi)],
        printed=[(np.pi / 2 * E3, np.pi / 2 * E3, -1), (np.pi * E1, np.pi * E1, 1)],
        states={"mmw_multiplicity": {("3/2", "1/2"): 1, ("5/2", "1/2"): 2, ("1/2", "1/2"): 0}, "mmw_allowed": [("1/2", "3/2")]}, source="MMW (91), (92)",
        parity=(2 * np.pi * E3, 0 * E3), parity_known={("3/2", "1/2"): [-1]})
    a6 = 0.16
    add("B6", 6, "D_4d", [1, 0, 0, 0, 1j * a6], [1j * a6, 0, 0, 0, 1, 0, 0], [(E3, np.pi / 2), (E1, np.pi)],
        printed=[(np.pi / 2 * E3, np.pi * E3, 1), (np.pi * E1, np.pi * E1, -1)],
        states={"mmw_allowed": [("1", "0"), ("3", "0"), ("0", "1"), ("2", "1")], "krusch": ("1", "3", "1")}, source="MMW (105), (106)",
        parity=(np.pi / 4 * E3, -np.pi / 2 * E3), parity_known={("1", "0"): [1], ("3", "0"): [1], ("0", "1"): [1]})
    add("B7", 7, "dodecahedron, Y_h", [7, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, -7, 0, 0], [(E3, 2 * np.pi / 5), (E2, np.pi), ("scan3", 2 * np.pi / 3)],
        states={"krusch": ("7/2", "13/2", "3/2")}, source="Krusch, table 2; MMW (117)")
    add("B7_D5", 7, "D_5 (the variant the 2007 paper quantizes), a = 2", [2, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, -2, 0, 0], [(E3, 2 * np.pi / 5), (E2, np.pi)],
        printed=[(2 * np.pi / 5 * E3, -4 * np.pi / 5 * E3, -1), (np.pi * E2, np.pi * E2, -1)],
        states={"mmw_multiplicity": {("3/2", "1/2"): 1, ("5/2", "1/2"): 1, ("7/2", "1/2"): 2, ("1/2", "1/2"): 0, ("1/2", "3/2"): 1}}, source="MMW (121), (122)",
        parity=(2 * np.pi * E3, 0 * E3), parity_known={("3/2", "1/2"): [-1], ("5/2", "1/2"): [-1], ("7/2", "1/2"): [-1, -1]})
    a8 = 0.14
    add("B8", 8, "D_6d", [1, 0, 0, 0, 0, 0, -1j * a8], [1j * a8, 0, 0, 0, 0, 0, -1, 0, 0], [(E3, np.pi / 3), (E1, np.pi)],
        printed=[(np.pi / 3 * E3, -2 * np.pi / 3 * E3, 1), (np.pi * E1, np.pi * E1, 1)],
        states={"mmw_allowed": [("0", "0"), ("2", "0")], "krusch": ("0", "2", "0")}, source="MMW (150), (151)")
    a12, b12 = -0.53, 0.78
    add("B12", 12, "T_d (the rational-map shell)", (a12 * pp ** 3 + b12 * pm ** 3).coeffs, (pp ** 2 * pm).coeffs, [(E3, np.pi), (N111, 2 * np.pi / 3)],
        states={"krusch": ("0", "3", "0")}, source="Krusch, tables 1 and 2")
    return cfgs


def scan_axis(rmap, angle, polar, rng):
    """The axis at the given polar angle about which a rotation by `angle` is a symmetry of the map."""
    best = (1.0, None)
    for phi in np.linspace(0, 2 * np.pi, 1441)[:-1]:
        axis = np.array([np.sin(polar) * np.cos(phi), np.sin(polar) * np.sin(phi), np.cos(polar)])
        misfit = rmap.induced(rotation_matrix(axis, angle), rng)[1]
        if misfit < best[0]:
            best = (misfit, axis)
    return best[1]


def fr_table():
    rng = np.random.default_rng(1939)
    rows, ok = [], True
    for cfg in configurations():
        rmap, B = cfg["map"], cfg["B"]
        gens = []
        for axis, angle in cfg["gens"]:
            if isinstance(axis, str):
                polar = np.pi / 2 if axis == "scan" else np.arccos(np.sqrt((5 + 2 * np.sqrt(5)) / 15))  # a face center of the icosahedron
                axis = scan_axis(rmap, angle, polar, rng)
            gens.append(constraint_from_map(rmap, axis, angle, rng))
        sound = all(g["misfit"] < 1e-8 and g["integer"] and g["axis_misfit"] < 1e-7 for g in gens)
        jmax, imax = (sp.Rational(13, 2), sp.Rational(3, 2)) if B % 2 else (sp.Integer(6), sp.Integer(2))
        tab = table(B, gens, jmax, imax)
        row = dict(key=cfg["key"], B=B, configuration=cfg["name"], source=cfg["source"], generators=gens, rule_sound=bool(sound), allowed=tab, checks=[])
        if cfg["printed"]:
            printed = [dict(space=np.asarray(s).tolist(), iso=np.asarray(i).tolist(), sign=x) for s, i, x in cfg["printed"]]
            same = table(B, printed, jmax, imax) == tab
            row["checks"].append(dict(check="the printed constraint equations allow the same states as the rule applied to the map", passed=bool(same)))
        st = cfg["states"]
        for J, I in st.get("mmw_allowed", []):
            row["checks"].append(dict(check=f"(J, I) = ({J}, {I}) is allowed (2007 paper)", passed=tab[I][J] > 0, multiplicity=tab[I][J]))
        for J, I in st.get("mmw_forbidden", []):
            row["checks"].append(dict(check=f"(J, I) = ({J}, {I}) is not allowed (2007 paper)", passed=tab[I][J] == 0, multiplicity=tab[I][J]))
        for (J, I), m in st.get("mmw_multiplicity", {}).items():
            row["checks"].append(dict(check=f"(J, I) = ({J}, {I}) has {m} state(s) (2007 paper)", passed=tab[I][J] == m, multiplicity=tab[I][J]))
        if "krusch" in st:
            i0, i1 = ("1/2", "3/2") if B % 2 else ("0", "1")
            ours = tuple((lowest(tab, i0) + [None, None])[:2] + (lowest(tab, i1, 1) + [None])[:1])
            row["lowest_states"] = dict(ground_J=ours[0], second_J=ours[1], lowest_J_at_the_next_isospin=ours[2])
            for label, mine, theirs in zip(("ground state", "second spin at the lowest isospin", "lowest spin at the next isospin"), ours, st["krusch"]):
                row["checks"].append(dict(check=f"{label}: J = {theirs} (2003 table)", passed=mine == theirs, ours=mine))
        if cfg["parity"] is not None:
            row["parity_operator"] = dict(space=np.asarray(cfg["parity"][0]).tolist(), iso=np.asarray(cfg["parity"][1]).tolist(),
                                          note="the operator the 2007 paper prints; where a symmetry element has sign -1 the choice is a convention, as that paper says")
            row["parity"] = {}
            for I, cols in tab.items():
                for J, m in cols.items():
                    if m and sp.nsimplify(J) <= (sp.Rational(7, 2) if B % 2 else 4):
                        vals, defect = parities(gens, cfg["parity"], float(sp.nsimplify(J)), float(sp.nsimplify(I)))
                        row["parity"][f"J={J}, I={I}"] = vals
                        if defect > 1e-6:
                            row["checks"].append(dict(check=f"the parity operator has eigenvalues +-1 on (J, I) = ({J}, {I})", passed=False, defect=defect))
            for (J, I), known in cfg["parity_known"].items():
                mine = row["parity"].get(f"J={J}, I={I}")
                row["checks"].append(dict(check=f"parity of (J, I) = ({J}, {I}) is {known} (2007 paper)", passed=mine == sorted(known), ours=mine))
        row["passed"] = bool(sound) and all(c["passed"] for c in row["checks"])
        ok &= row["passed"]
        rows.append(row)
        print(f"B = {B:2d} {cfg['name']:45s} N = {[round(g['N'], 6) for g in gens]} signs {[g['sign'] for g in gens]} "
              f"{'sound' if sound else 'UNSOUND'}; " + "; ".join(f"I = {I}: " + ", ".join(f"{J}" + (f" x{m}" if m > 1 else "") for J, m in t.items() if m) for I, t in tab.items()))
        if "parity" in row:
            print("     parity:", row["parity"])
        for c in row["checks"]:
            if not c["passed"]:
                print("     DISAGREES:", c)
    return dict(condition="the sign source is the winding field alone, and the configuration is the rational-map shell named in the row",
                rows=rows, passed=bool(ok))


def main():
    out = dict(task="M5.39 R4")
    out["moment_bound"] = moment_bound()
    out["cranking"] = cranking()
    out["finkelstein_rubinstein"] = fr_table()
    print("(a)", out["moment_bound"]["passed"], [(r["quantity"], round(r["returned"], 4), r["label"]) for r in out["moment_bound"]["rows"]])
    print("(b)", out["cranking"])
    path = Path(__file__).resolve().parents[1] / "data" / "m5_39_r4_desk.json"
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("R4", "ALL PASSED" if all(out[k]["passed"] for k in ("moment_bound", "cranking", "finkelstein_rubinstein")) else "SEE THE ROWS")


if __name__ == "__main__":
    main()
