"""M5.32 R19-0 part 2: the form-level certificate of the mixed-bracket
entrants at the vacuum (R1's certificate re-run with the new columns), on
the rigid clock, algebra only.

EQUATIONS FIRST
---------------
Objects (m5_32_r19_c9_c12.objects, the same code): I1 (certified), GG =
(A), (T) the time-index split (F for spatial (mu, nu), G for (0, i), (i, 0)
and (0, 0)), (B-u) at the vacuum (= the fixed-frame (B)), (Gamma) and
(Gamma, tl). Vacuum M_vac = diag(g, 1, delta, 0), u = e_0.

The rigid clock A_0 = omega a0 on top of static jets A_i (the general
symmetric 30-parameter family, 10 per spatial direction): every density is
an even quartic polynomial in omega,
    I_X(omega; x) = A_X(x) + C_X(x) omega^2 + D_X omega^4,
with C_X(x) an exact quadratic form in the static jets x (its 30x30 matrix
Q_X by polarization, checked on random x) and D_X = (1/2) eta^0 eta^0
<X_00, X_00>_eta INDEPENDENT of x (only A_0 enters X_00; checked).
The family L(c) = -4 [(1 - c) I1 + c X] - V4 (c = 0 the certified action,
c = 1 the literal replacement). Its rigid-clock Hamiltonian (the Legendre
transform of an even quartic Lagrangian, H = omega dL/domega - L):
    H(c) = 4 A + V4 - 4 C(c) omega^2 - 12 c D_X omega^4,
    C(c) = (1 - c) C_I1 + c C_X,   D_I1 = 0.
So the kinetic energy is the quadratic form H2(c) = -4 [(1 - c) Q_I1 +
c Q_X] over the static jets (R1's object: PSD on a channel means every
static texture carries non-negative inertia on that clock), and the
omega^4 energy coefficient is E4(c) = -12 c D_X: with c > 0 it is
non-negative only if D_X <= 0. Both are read per channel: the six Lorentz
generators a0 = G M_vac + M_vac G^T (boost_k, rot_k) and the three
notebook local clocks a0 = Gamma M_vac + M_vac Gamma^T (Gamma with a
symmetric time row t and an antisymmetric spatial block r; t only; r only).

Pre-registered readings (ledger § 6.8 R19-0): (i) a c window with H2 PSD
on every channel and E4 >= 0: CANDIDATE for the clock; (ii) no such c:
CANDIDATE_REFUTED at the form level for the constant-coefficient family
(the window is a 1-parameter interval computed exactly from the minimal
eigenvalue as a function of c on a fine grid with bisection at the edges).
Also recorded: the rotation-channel identity (Q_X = Q_I1 and D_X = 0 for
(B-u), (Gamma), (Gamma, tl): a rotation clock at the vacuum has no time
row, so the boost-sector replacement cannot act on it), and D_X on the
boost channels in closed form (D_Gamma = 4 |v|^4 with v the shape-factored
boost vector; D_Bu = 2 |v|^4; D_Gamma_tl = (4/3) |v|^4).
Out: ../data/m5_32_r19_certificate.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(DATA, "m5_32_r19_certificate.json")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


C = _load("m5_32_r19_c9_c12", "m5_32_r19_c9_c12.py")
ETA, ETA_D, W = C.ETA, C.ETA_D, C.W
T0 = time.time()
OBJECTS = ["I1", "GG", "T", "Bu", "Gam", "Gam_tl"]
IU = np.triu_indices(4)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def density_T(A):
    P = C.P_of(A); F = P - P.swapaxes(0, 1); G = P + P.swapaxes(0, 1)
    X = F.copy(); X[0] = G[0]; X[:, 0] = G[:, 0]
    return C.Ifull(X)


def densities(A, u):
    ob = C.objects(A, u)
    return {"I1": ob["I1"], "GG": ob["GG"], "T": density_T(A), "Bu": ob["Bu"], "Gam": ob["Gam"], "Gam_tl": ob["Gam_tl"]}


def jet30(x):
    """x (30,) -> static jets A (4, 1, 4, 4), A[0] = 0."""
    A = np.zeros((4, 1, 4, 4))
    for i in range(3):
        S = np.zeros((4, 4)); S[IU] = x[10 * i:10 * i + 10]
        A[1 + i, 0] = S + S.T - np.diag(np.diag(S))
    return A


def gen_boost(k):
    G = np.zeros((4, 4)); G[0, k] = G[k, 0] = 1.0; return G


def gen_rot(k):
    G = np.zeros((4, 4))
    i, j = [x for x in (1, 2, 3) if x != k]
    s = 1.0 if (k, i, j) in [(1, 2, 3), (2, 3, 1), (3, 1, 2)] else -1.0
    G[i, j] = -s; G[j, i] = s
    return G


def gamma(t, r):
    Gm = np.zeros((4, 4))
    Gm[0, 1:] = t; Gm[1:, 0] = t
    Gm[1, 2], Gm[1, 3] = -r[2], r[1]
    Gm[2, 1], Gm[2, 3] = r[2], -r[0]
    Gm[3, 1], Gm[3, 2] = -r[1], r[0]
    return Gm


def channels(D0):
    ch = {}
    for k in (1, 2, 3):
        G = gen_boost(k); ch[f"boost_{k}"] = G @ D0 + D0 @ G.T
        G = gen_rot(k); ch[f"rot_{k}"] = G @ D0 + D0 @ G.T
    t = np.array([1.0, 0.7, -0.4]); r = np.array([0.5, -0.8, 0.3])
    for name, tt, rr in (("clock_tr", t, r), ("clock_t", t, 0 * r), ("clock_r", 0 * t, r)):
        G = gamma(tt, rr); ch[name] = G @ D0 + D0 @ G.T
    return ch


U0 = np.array([[1.0, 0.0, 0.0, 0.0]])


def omega_coeffs(a0, x):
    """(A, C, D) per object for the clock a0 on the static jets x."""
    A = jet30(x)
    vals = {}
    for wv in (0.0, 1.0, -1.0, 2.0, -2.0):
        Aw = A.copy(); Aw[0, 0] = wv * a0
        vals[wv] = {k: float(v[0]) for k, v in densities(Aw, U0).items()}
    out = {}
    for k in OBJECTS:
        a = vals[0.0][k]
        S1 = 0.5 * (vals[1.0][k] + vals[-1.0][k]); S2 = 0.5 * (vals[2.0][k] + vals[-2.0][k])
        d = (S2 - 4.0 * S1 + 3.0 * a) / 12.0
        c = S1 - a - d
        odd = max(abs(vals[1.0][k] - vals[-1.0][k]), abs(vals[2.0][k] - vals[-2.0][k]))
        out[k] = (a, c, d, odd)
    return out


def Q_matrices(a0, n=30):
    """30x30 matrices of C_X(x) by polarization, plus D_X (x-independent)."""
    e = np.eye(n)
    diag = [omega_coeffs(a0, e[i]) for i in range(n)]
    Q = {k: np.zeros((n, n)) for k in OBJECTS}
    for i in range(n):
        for k in OBJECTS:
            Q[k][i, i] = diag[i][k][1]
    for i in range(n):
        for j in range(i + 1, n):
            c = omega_coeffs(a0, e[i] + e[j])
            for k in OBJECTS:
                Q[k][i, j] = Q[k][j, i] = 0.5 * (c[k][1] - diag[i][k][1] - diag[j][k][1])
    D = {k: diag[0][k][2] for k in OBJECTS}
    Dvar = {k: max(abs(diag[i][k][2] - D[k]) for i in range(n)) for k in OBJECTS}
    return Q, D, Dvar


def check_polarization(Q, a0, rng, n=30):
    worst = {k: 0.0 for k in OBJECTS}; odd = 0.0
    for _ in range(6):
        x = rng.normal(size=n)
        c = omega_coeffs(a0, x)
        for k in OBJECTS:
            worst[k] = max(worst[k], abs(x @ Q[k] @ x - c[k][1]) / max(1.0, abs(c[k][1])))
            odd = max(odd, c[k][3] / max(1.0, abs(c[k][0])))
    return worst, odd


def min_eig(H):
    return float(np.linalg.eigvalsh(0.5 * (H + H.T))[0])


def window_of(QI, QX, cs, tol_rel=1e-9):
    """the set of c with H2(c) = -4[(1-c) QI + c QX] PSD, as intervals on the grid, edges bisected."""
    def me(c):
        H = -4.0 * ((1 - c) * QI + c * QX)
        wv = np.linalg.eigvalsh(0.5 * (H + H.T))
        return wv[0], wv[-1]
    ok = []
    for c in cs:
        lo, hi = me(c)
        ok.append(lo >= -tol_rel * max(1.0, abs(hi)))
    ok = np.array(ok)
    intervals = []
    i = 0
    while i < len(cs):
        if ok[i]:
            j = i
            while j + 1 < len(cs) and ok[j + 1]:
                j += 1
            a, b = cs[i], cs[j]
            # bisect the edges
            if i > 0:
                lo_c, hi_c = cs[i - 1], cs[i]
                for _ in range(30):
                    m = 0.5 * (lo_c + hi_c)
                    l_, h_ = me(m)
                    if l_ >= -tol_rel * max(1.0, abs(h_)):
                        hi_c = m
                    else:
                        lo_c = m
                a = hi_c
            if j + 1 < len(cs):
                lo_c, hi_c = cs[j], cs[j + 1]
                for _ in range(30):
                    m = 0.5 * (lo_c + hi_c)
                    l_, h_ = me(m)
                    if l_ >= -tol_rel * max(1.0, abs(h_)):
                        lo_c = m
                    else:
                        hi_c = m
                b = lo_c
            intervals.append([float(a), float(b)])
            i = j + 1
        else:
            i += 1
    return intervals


def intersect(ints_list):
    """intersection of lists of intervals."""
    cur = [[-np.inf, np.inf]]
    for ints in ints_list:
        new = []
        for a, b in cur:
            for c, d in ints:
                lo, hi = max(a, c), min(b, d)
                if lo <= hi:
                    new.append([lo, hi])
        cur = new
        if not cur:
            return []
    return [[float(a), float(b)] for a, b in cur]


def certify(g, rng, cs):
    D0 = C.vac(g)
    ch = channels(D0)
    out = {"g": g, "channels": list(ch), "per_channel": {}}
    Qs = {}
    for name, a0 in ch.items():
        Q, D, Dvar = Q_matrices(a0)
        pol, odd = check_polarization(Q, a0, rng)
        Qs[name] = Q
        row = {"D": D, "D_x_dependence_max": Dvar, "polarization_worst_rel": pol, "odd_powers_rel": odd,
               "min_eig_minus4Q": {k: min_eig(-4.0 * Q[k]) for k in OBJECTS},
               "max_eig_minus4Q": {k: float(np.linalg.eigvalsh(-4.0 * Q[k])[-1]) for k in OBJECTS},
               "Q_minus_Q_I1_max_abs": {k: float(np.abs(Q[k] - Q["I1"]).max()) for k in OBJECTS},
               "window_c": {k: window_of(Q["I1"], Q[k], cs) for k in OBJECTS if k != "I1"}}
        # the literal c = 1 (the replacement) and the E4 sign per object
        row["c1_PSD"] = {k: min_eig(-4.0 * Q[k]) >= -1e-9 * max(1.0, abs(row["max_eig_minus4Q"][k])) for k in OBJECTS}
        row["E4_at_c1"] = {k: -12.0 * D[k] for k in OBJECTS}
        out["per_channel"][name] = row
        log(f"g={g} {name}: c=1 PSD {row['c1_PSD']}  D {{k: round(v, 3) for k, v in D.items()}}")
    # joint windows
    lor = [n for n in ch if n.startswith("boost") or n.startswith("rot")]
    joint = {}
    for k in OBJECTS:
        if k == "I1":
            continue
        ints_l = intersect([out["per_channel"][n]["window_c"][k] for n in lor])
        ints_all = intersect([out["per_channel"][n]["window_c"][k] for n in ch])
        # E4 >= 0 on every channel: c * D_X(channel) <= 0 for every channel
        Dpos = any(out["per_channel"][n]["D"][k] > 1e-12 for n in ch)
        Dneg = any(out["per_channel"][n]["D"][k] < -1e-12 for n in ch)
        e4_c_range = [-np.inf, 0.0] if Dpos and not Dneg else ([0.0, np.inf] if Dneg and not Dpos else ([0.0, 0.0] if Dpos and Dneg else [-np.inf, np.inf]))
        joint[k] = {"H2_window_lorentz6": ints_l, "H2_window_all9": ints_all,
                    "E4_nonneg_c_range": [float(x) for x in e4_c_range],
                    "joint_window_lorentz6": intersect([ints_l, [e4_c_range]]) if ints_l else [],
                    "joint_window_all9": intersect([ints_all, [e4_c_range]]) if ints_all else []}
    out["joint"] = joint
    # the boost-channel closed forms
    v2 = {f"boost_{k}": (float(np.diag(D0)[k] + np.diag(D0)[0])) ** 2 for k in (1, 2, 3)}
    out["closed_form_D_boost"] = {n: {"v2": v2[n], "Gam_4v4": 4 * v2[n] ** 2, "Bu_2v4": 2 * v2[n] ** 2, "Gam_tl_4_3_v4": (4.0 / 3.0) * v2[n] ** 2,
                                      "measured": {k: out["per_channel"][n]["D"][k] for k in ("Gam", "Bu", "Gam_tl", "GG", "T")}} for n in v2}
    return out


def main():
    rng = np.random.default_rng(23)
    cs = np.round(np.arange(-3.0, 3.0001, 0.005), 4)
    out = {"objects": OBJECTS, "family": "L(c) = -4[(1-c) I1 + c X] - V4", "c_grid": [-3.0, 3.0, 0.005]}
    for g in (8.0, 32.0):
        out[f"g{int(g)}"] = certify(g, rng, cs)
    out["runtime_s"] = time.time() - T0
    r8 = out["g8"]
    v = {}
    v["polarization_exact"] = max(max(r8["per_channel"][n]["polarization_worst_rel"].values()) for n in r8["per_channel"]) < 1e-9
    v["no_odd_powers"] = max(r8["per_channel"][n]["odd_powers_rel"] for n in r8["per_channel"]) < 1e-9
    v["D_independent_of_static_jets"] = max(max(r8["per_channel"][n]["D_x_dependence_max"].values()) for n in r8["per_channel"]) < 1e-6
    lor6 = [f"{t}_{k}" for t in ("boost", "rot") for k in (1, 2, 3)]
    v["I1_indefinite_on_every_lorentz_channel_30jet"] = all(r8["per_channel"][n]["min_eig_minus4Q"]["I1"] < -1e-9 for n in lor6)
    v["min_eig_unchanged_by_every_object_lorentz6"] = all(abs(r8["per_channel"][n]["min_eig_minus4Q"][o] - r8["per_channel"][n]["min_eig_minus4Q"]["I1"]) < 1e-9 * max(1.0, abs(r8["per_channel"][n]["min_eig_minus4Q"]["I1"])) for n in lor6 for o in OBJECTS)
    v["rotation_identity_Bu_Gam_Gam_tl"] = all(r8["per_channel"][f"rot_{k}"]["Q_minus_Q_I1_max_abs"][o] < 1e-9 and abs(r8["per_channel"][f"rot_{k}"]["D"][o]) < 1e-9 for k in (1, 2, 3) for o in ("Bu", "Gam", "Gam_tl"))
    cf = r8["closed_form_D_boost"]
    v["boost_D_closed_form"] = all(abs(cf[n]["measured"]["Gam"] - cf[n]["Gam_4v4"]) < 1e-6 * cf[n]["Gam_4v4"] and abs(cf[n]["measured"]["Bu"] - cf[n]["Bu_2v4"]) < 1e-6 * cf[n]["Bu_2v4"] and abs(cf[n]["measured"]["Gam_tl"] - cf[n]["Gam_tl_4_3_v4"]) < 1e-6 * cf[n]["Gam_tl_4_3_v4"] for n in cf)
    for o in ("GG", "T", "Bu", "Gam", "Gam_tl"):
        v[f"{o}_no_c_window_lorentz6_REFUTED_at_form_level"] = len(r8["joint"][o]["H2_window_lorentz6"]) == 0
        v[f"{o}_boost_E4_negative_at_c1"] = all(r8["per_channel"][f"boost_{k}"]["E4_at_c1"][o] < 0 for k in (1, 2, 3))
    out["verdict"] = {k: bool(x) for k, x in v.items()}
    os.makedirs(DATA, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    for k, x in out["verdict"].items():
        print(f"{'PASS' if x else 'FAIL'} {k}")
    for n in r8["per_channel"]:
        row = r8["per_channel"][n]
        print(n, "min_eig(-4Q):", {k: round(row["min_eig_minus4Q"][k], 4) for k in OBJECTS}, "D:", {k: round(row["D"][k], 4) for k in OBJECTS})
        print("   windows:", {k: row["window_c"][k] for k in ("GG", "T", "Bu", "Gam", "Gam_tl")})
    print(json.dumps(r8["joint"], indent=1))
    return out


if __name__ == "__main__":
    main()
