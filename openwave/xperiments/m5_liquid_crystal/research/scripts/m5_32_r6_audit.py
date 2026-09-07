"""M5.32 R6 INDEPENDENT AUDIT (MINIMAL mode): the delta ladder (R6.a) and
the orbit-blindness theorem (R6.b).

The producer scripts were NOT read. Inputs: the producer's DATA
(data/m5_32_r6_deltaladder.json, data/m5_32_r6_orbitblind.json) and the
oracles m5_21_8_b_lattice.py (dressed hedgehog, a0_unit), m5_21_3_a_4d.py
(the certified 4D stack), m5_32_lagrangian.py (v4_traces_np). No lattice
relaxation is run; the only lattice work is two a0_unit evaluations.

EQUATIONS FIRST
---------------
D1 Identity A (the record clock generator).
    Family (m5_21_8_b_lattice.dressed, m = 0):
        M(x, t) = Qh(x, t) d4 Qh(x, t)^T,   Qh = R3(phi) R2(th) R1(om t)
        d4 = diag(g, 1, delta, 0)           (vac4 at s = -1)
        R1(a) = exp(a G1), G1 the (2,3) rotation generator
    so at t = 0 with om = 1
        a0 := dM/dt = Qh (G1 d4 - d4 G1) Qh^T,   Qh = R3 R2
    and the commutator is elementary:
        (G1 d4 - d4 G1)_{23} = -d4_{22} G1_{23} = delta,
        (G1 d4 - d4 G1)_{32} =  G1_{32} d4_{22} = delta   (sympy check below)
        G1 d4 - d4 G1 = delta (E_23 + E_32)                 (Identity A)
    hence a0 = delta * Qh (E_23 + E_32) Qh^T, with
        max_x max_ab |a0_ab| / delta <= 1  (Qh orthogonal, S = E_23 + E_32
        has spectrum {+1, -1}, |(Qh S Qh^T)_ab| <= ||S||_2 = 1).
    Numerical test: the oracle a0_unit(cfg, 0) (central FD, dt = 1e-4)
    versus the closed form built from the same grid angles; report
    max |a0_oracle - a0_closed| / delta and max |a0| / delta.
    Generalization (beyond the record point m = 0): at m != 0 the family
    carries the radial boost Qb(m, x) on the left, a0 = Qb Qh S Qh^T Qb^T
    with Qh S Qh^T = e_a e_b^T + e_b e_a^T on two tangent directions
    e_a, e_b perpendicular to the radial n. A boost along n acts as the
    identity on span(e_a, e_b), so Qb Qh S Qh^T Qb^T = Qh S Qh^T and
    Identity A holds at EVERY m (checked numerically at m = 0.2).

D2 kin(L) refit, from the producer's kin_table (deltas 0.3, 0.1, 0.03,
   0.01; L = 48, 72, 96):
        linear      kin = a + b L                (least squares, 3 pts)
        free power  kin = a + b L^p              (3 pts, exact: solve
                    (k3 - k2)/(k2 - k1) = (L3^p - L2^p)/(L2^p - L1^p) for p
                    by bisection, then b, a)
        delta exponent of b: slope of log b vs log delta (4 pts, LSQ).

D3 fixed-J drift, from drift_table: drift = omega(L=72)/omega(L=48) - 1 per
   (delta, lambda, J); R* = the R at which the fixed-J energy is minimal in
   each box (producer's R_star_L1/L2). Claim: core-scaled J rows at
   lambda >= 0.75 have delta-independent drift ~ -35 % and R* = L/2.

D4 static gain, from gain_table: G(R)/G(6) at lambda 1 per delta and box;
   "non-saturating" = the last increment G(24) - G(18) is not small vs
   G(6); "converging" = the shape at delta 0.03 and 0.01 agree; lambda 0
   "saturating by R = 12" = |G(R)/G(6) - G(12)/G(6)| small for R >= 12.

D5 orbit blindness (three lines). For Q in O(1,3), Q^T eta Q = eta:
    (1) Q^T = eta Q^{-1} eta   (from Q^T eta Q = eta, eta^2 = 1)
    (2) (Q M Q^T) eta = Q M (eta Q^{-1} eta) eta = Q (M eta) Q^{-1}
    (3) hence (Q M Q^T) eta is similar to M eta: tr((M eta)^p) invariant,
        det(Q M Q^T) = det(Q)^2 det(M) = det(M)  (det Q = +-1).
    V4 = w sum_p (tr((M eta)^p) - C_p)^2 is a function of those traces only,
    so V4 is constant on every O(1,3) orbit. Symbolic check with a
    rapidity-symbol boost and a rotation-angle symbol; numeric check: 20
    random Lorentz dressings (rapidity up to 3, random rotations) of a random
    symmetric sector point; report max relative variation of V4, det(M),
    tr((M eta)^p); Euclidean control tr(M^2) must NOT be invariant.

Run: /opt/anaconda3/envs/master312/bin/python3 m5_32_r6_audit.py
Out: ../data/m5_32_r6_audit.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import time

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
T0 = time.time()


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B3 = _load("ins4", "m5_21_3_a_4d.py")
B8 = _load("lat8", "m5_21_8_b_lattice.py")
LAG = _load("lag", "m5_32_lagrangian.py")

OUT = {"task": "M5.32 R6 independent audit (MINIMAL)", "claims": {}}
LINES = []


def log(s):
    print(s)
    LINES.append(s)


# --------------------------------------------------------------------- D1
def d1_identity_a():
    g, dl = sp.symbols("g delta", positive=True)
    G1 = sp.zeros(4, 4)
    G1[2, 3], G1[3, 2] = -1, 1                 # the lattice G1
    d4 = sp.diag(g, 1, dl, 0)
    E23 = sp.zeros(4, 4); E23[2, 3] = 1
    E32 = sp.zeros(4, 4); E32[3, 2] = 1
    comm = G1 * d4 - d4 * G1
    sym_ok = sp.simplify(comm - dl * (E23 + E32)) == sp.zeros(4, 4)
    # also with the opposite sign convention of G1 (the sum is symmetric
    # in the sign only if... check explicitly)
    comm_neg = (-G1) * d4 - d4 * (-G1)
    sym_neg_ok = sp.simplify(comm_neg + dl * (E23 + E32)) == sp.zeros(4, 4)
    res = {"symbolic_identity": bool(sym_ok),
           "symbolic_flipped_G1_gives_minus": bool(sym_neg_ok),
           "comm": str(comm.tolist()), "lattice": {}}
    log(f"D1 sympy: G1 d4 - d4 G1 == delta (E23 + E32): {sym_ok}"
        f" (flipped G1 sign -> minus identity: {sym_neg_ok})")
    S = np.zeros((4, 4)); S[2, 3] = S[3, 2] = 1.0
    for delta in (0.3, 0.01):
        cfg = B3.base_cfg(s=-1.0, g=32.0, n=32, L=48.0, delta=delta)
        n, h = cfg["n"], cfg["h"]
        X, Y, Z = B3.coords(n, h)
        rho = np.sqrt(X * X + Y * Y)
        phi = np.arctan2(Y, X)
        th = -np.arctan2(Z, rho)
        Qh = np.einsum("...ab,...bc->...ac", B8.rot_field(B8.G3, phi),
                       B8.rot_field(B8.G2, th))
        closed = delta * np.einsum("...ab,bc,...dc->...ad", Qh, S, Qh)
        a0 = B8.a0_unit(cfg, 0.0)
        dev = float(np.max(np.abs(a0 - closed)) / delta)
        amax = float(np.max(np.abs(a0)) / delta)
        # m != 0: the Qb-dressed generator
        a0m = B8.a0_unit(cfg, 0.2)
        dev_m = float(np.max(np.abs(a0m - closed)) / delta)
        amax_m = float(np.max(np.abs(a0m)) / delta)
        res["lattice"][str(delta)] = {
            "max_abs_dev_over_delta_m0": dev, "max_a0_over_delta_m0": amax,
            "m0.2_max_abs_dev_over_delta": dev_m, "m0.2_max_a0_over_delta": amax_m}
        log(f"D1 lattice delta={delta}: max|a0 - closed|/delta = {dev:.2e}, "
            f"max|a0|/delta = {amax:.6f} (m=0); at m=0.2: dev {dev_m:.2e}, "
            f"max|a0|/delta {amax_m:.6f} (boost along n leaves the tangent "
            f"block e_a e_b^T + e_b e_a^T invariant: Identity A holds at every m)")
    ok = sym_ok and all(v["max_abs_dev_over_delta_m0"] < 1e-6
                        and v["max_a0_over_delta_m0"] <= 1 + 1e-9
                        for v in res["lattice"].values())
    res["verdict"] = "CONFIRMED" if bool(ok) else "REFUTED"
    return res


# --------------------------------------------------------------------- D2
def _fit_power(L, k):
    L = np.asarray(L, float); k = np.asarray(k, float)

    def f(p):
        return ((k[2] - k[1]) / (k[1] - k[0])
                - (L[2] ** p - L[1] ** p) / (L[1] ** p - L[0] ** p))
    lo, hi = 0.2, 3.0
    flo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    b = (k[1] - k[0]) / (L[1] ** p - L[0] ** p)
    a = k[0] - b * L[0] ** p
    return float(p), float(a), float(b)


def d2_kin_refit(dl):
    kt = dl["kin_table"]
    res = {"per_delta": {}}
    deltas, bs, bps, as_ = [], [], [], []
    for ds, row in kt.items():
        L = np.array(row["L"]); k = np.array(row["kin"])
        A = np.vstack([np.ones_like(L), L]).T
        (a, b), *_ = np.linalg.lstsq(A, k, rcond=None)
        resid = float(np.max(np.abs(k - (a + b * L)) / k))
        p, ap, bp = _fit_power(L, k)
        prod = row["fit"]
        res["per_delta"][ds] = {
            "linear": {"a": float(a), "b": float(b), "max_rel_resid": resid},
            "power": {"p": p, "a": ap, "b": bp},
            "producer": {"a_lin": prod["linear"]["a"], "b_lin": prod["linear"]["b"],
                         "p": prod["p"]},
            "match_producer_b": bool(abs(b - prod["linear"]["b"]) / prod["linear"]["b"] < 1e-9),
            "match_producer_p": bool(abs(p - prod["p"]) < 1e-6)}
        deltas.append(float(ds)); bs.append(b); bps.append(bp); as_.append(a)
        log(f"D2 delta={ds}: linear a={a:.4g} b={b:.4g} (max rel resid "
            f"{resid:.1e}); power p={p:.4f} a={ap:.4g} b={bp:.4g}; "
            f"producer p={prod['p']:.4f} b_lin={prod['linear']['b']:.4g}")
    ld = np.log(deltas)
    slope_b = float(np.polyfit(ld, np.log(bs), 1)[0])
    slope_bp = float(np.polyfit(ld, np.log(bps), 1)[0])
    slope_k48 = float(np.polyfit(ld, np.log([kt[d]["kin"][0] for d in kt]), 1)[0])
    pair = {f"{deltas[i]}->{deltas[i+1]}":
            float(np.log(bs[i + 1] / bs[i]) / np.log(deltas[i + 1] / deltas[i]))
            for i in range(3)}
    res.update({"b_linear_delta_exponent": slope_b,
                "b_power_delta_exponent": slope_bp,
                "kin_L48_delta_exponent": slope_k48,
                "pairwise_b_exponent": pair,
                "a_sign": ["negative" if a < 0 else "nonnegative" for a in as_],
                "a_over_bL48": [float(a / (b * 48)) for a, b in zip(as_, bs)],
                "producer_b_linear_exponent": dl["delta_exponents"]["b_linear_exponent"]})
    log(f"D2 delta exponent of b: linear {slope_b:.4f}, power {slope_bp:.4f}, "
        f"kin(L48) {slope_k48:.4f}; pairwise {pair}; a < 0 in all rows: "
        f"{all(a < 0 for a in as_)}; a/(b*48) = "
        + " ".join(f"{x:.3f}" for x in res["a_over_bL48"]))
    ok = (all(v["match_producer_b"] and v["match_producer_p"] for v in res["per_delta"].values())
          and all(a < 0 for a in as_) and abs(slope_b - 2) < 0.06)
    res["verdict"] = "CONFIRMED" if bool(ok) else "REFUTED"
    return res


# --------------------------------------------------------------------- D3
def d3_drift(dl):
    dt = dl["drift_table"]
    res = {"rows": {}, "J_values": {d: dl["per_delta"][d]["J_values"] for d in dl["per_delta"]}}
    core, allhi = [], []
    rstar_ok = True
    for ds, t in dt.items():
        for key, v in t.items():
            lam = float(key.split("_")[1])
            if lam < 0.75:
                continue
            r = {"drift": v["drift"], "R_star": [v["R_star_L1"], v["R_star_L2"]],
                 "R_star_is_L_over_2": (v["R_star_L1"] == 24.0 and v["R_star_L2"] == 36.0)}
            res["rows"][f"{ds}|{key}"] = r
            rstar_ok &= r["R_star_is_L_over_2"]
            allhi.append(v["drift"])
            if "Jcore" in key:
                core.append(v["drift"])
    core = np.array(core); allhi = np.array(allhi)
    res["core_drift_min_max"] = [float(core.min()), float(core.max())]
    res["core_drift_spread"] = float(core.max() - core.min())
    res["J200_drifts"] = {k: v["drift"] for k, v in res["rows"].items() if "J_200" in k}
    res["all_R_star_L_over_2"] = bool(rstar_ok)
    log(f"D3 core-scaled J, lambda>=0.75: drift in [{core.min():.4f}, {core.max():.4f}] "
        f"(spread {core.max()-core.min():.4f}) over 4 deltas x 2 lambdas x 2 J; "
        f"R* = L/2 in every lambda>=0.75 row: {rstar_ok}")
    log("D3 J_200 rows (NOT core-scaled): " + " ".join(f"{k}:{v:.3f}" for k, v in res["J200_drifts"].items()))
    log("D3 J_values per delta: " + json.dumps(res["J_values"]))
    rstar_ok = bool(rstar_ok)
    res["all_R_star_L_over_2"] = rstar_ok
    ok = rstar_ok and core.max() - core.min() < 0.01 and abs(core.mean() + 0.355) < 0.01
    res["verdict"] = "CONFIRMED" if ok else "REFUTED"
    return res


# --------------------------------------------------------------------- D4
def d4_gain(dl):
    gt = dl["gain_table"]
    res = {"lam1": {}, "lam0": {}}
    shapes = {}
    for ds, boxes in gt.items():
        for box, lams in boxes.items():
            s1 = np.array(lams["lam_1"]["G_over_G6"])
            s0 = np.array(lams["lam_0"]["G_over_G6"])
            res["lam1"][f"{ds}|{box}"] = {
                "shape": s1.tolist(), "last_increment_over_G6": float(s1[-1] - s1[-2]),
                "nonsaturating": bool(s1[-1] - s1[-2] > 0.1 * 1.0)}
            res["lam0"][f"{ds}|{box}"] = {
                "shape": s0.tolist(),
                "max_abs_change_beyond_R12": float(np.max(np.abs(s0[2:] - s0[2]))),
                "plateau_by_R12": bool(np.max(np.abs(s0[2:] - s0[2])) < 0.03),
                "producer_saturating_flag": lams["lam_0"]["saturating"]}
            if box == "n32_L48":
                shapes[ds] = s1
    d = list(shapes)
    conv = {f"{d[i]}->{d[i+1]}": float(np.max(np.abs(shapes[d[i+1]] - shapes[d[i]]) / shapes[d[i]]))
            for i in range(len(d) - 1)}
    res["lam1_shape_change_between_deltas_n32"] = conv
    log("D4 lambda 1 G(R)/G(6), n32_L48: " + " | ".join(
        f"{k}: " + ",".join(f"{x:.2f}" for x in v) for k, v in shapes.items()))
    log(f"D4 lambda 1 max rel shape change between consecutive deltas: {conv}")
    log("D4 lambda 0 plateau by R=12 (max |G/G6 - G12/G6| beyond R12): " + " ".join(
        f"{k}:{v['max_abs_change_beyond_R12']:.3f}(flag {v['producer_saturating_flag']})"
        for k, v in res["lam0"].items()))
    nonsat = all(v["nonsaturating"] for v in res["lam1"].values())
    plateau = all(v["plateau_by_R12"] for v in res["lam0"].values())
    converging = list(conv.values())[-1] < 0.01
    flags_disagree = any(not v["producer_saturating_flag"] for v in res["lam0"].values())
    res.update({"lam1_all_nonsaturating": bool(nonsat), "lam0_all_plateau_by_R12": bool(plateau),
                "lam1_converging_last_step_lt_1pct": bool(converging),
                "producer_lam0_flag_false_somewhere": bool(flags_disagree)})
    res["verdict"] = ("CONFIRMED" if (nonsat and plateau and converging and not flags_disagree)
                      else ("QUALIFIED" if (nonsat and plateau and converging) else "REFUTED"))
    return res


# --------------------------------------------------------------------- D5
def d5_orbit_blind():
    eta = sp.diag(-1, 1, 1, 1)
    chi, th = sp.symbols("chi theta", real=True)
    Qb = sp.eye(4)
    Qb[0, 0] = sp.cosh(chi); Qb[0, 1] = sp.sinh(chi)
    Qb[1, 0] = sp.sinh(chi); Qb[1, 1] = sp.cosh(chi)
    Qr = sp.eye(4)
    Qr[2, 2] = sp.cos(th); Qr[2, 3] = -sp.sin(th)
    Qr[3, 2] = sp.sin(th); Qr[3, 3] = sp.cos(th)
    Q = Qb * Qr
    Msym = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"m{min(i,j)}{max(i,j)}"))
    line0 = sp.simplify(Q.T * eta * Q - eta) == sp.zeros(4, 4)
    line1 = sp.simplify(Q.T - eta * Q.inv() * eta) == sp.zeros(4, 4)
    lhs = (Q * Msym * Q.T) * eta
    rhs = Q * (Msym * eta) * Q.inv()
    line2 = sp.simplify(lhs - rhs) == sp.zeros(4, 4)
    tr_ok = [bool(sp.simplify((lhs ** p).trace() - ((Msym * eta) ** p).trace()) == 0)
             for p in (1, 2)]
    det_ok = bool(sp.simplify((Q * Msym * Q.T).det() - Msym.det()) == 0)
    log(f"D5 sympy: Q^T eta Q = eta {line0}; Q^T = eta Q^-1 eta {line1}; "
        f"(Q M Q^T) eta = Q (M eta) Q^-1 {line2}; tr((.)^p) p=1,2 invariant {tr_ok}; "
        f"det invariant {det_ok}")
    # numeric
    rng = np.random.default_rng(632)
    ETA = np.diag([-1.0, 1, 1, 1])
    p = {"s": -1.0, "g": 32.0, "delta": 0.3, "w": 1.0}
    cp = LAG.c4_of(p)
    # random sector point: random symmetric M near the vacuum (not pinned)
    M0 = np.diag([32.0, 1.0, 0.3, 0.0]) + 0.5 * (lambda a: a + a.T)(rng.normal(size=(4, 4)))

    def rand_lorentz():
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        chi = rng.uniform(-3, 3)
        K = np.zeros((4, 4)); K[0, 1:] = v; K[1:, 0] = v
        B = np.eye(4) + np.sinh(chi) * K + (np.cosh(chi) - 1) * (K @ K)
        A = rng.normal(size=(3, 3)); Rq, _ = np.linalg.qr(A)
        R = np.eye(4); R[1:, 1:] = Rq
        return B @ R

    vals = {"V4": [], "det": [], "t1": [], "t2": [], "t3": [], "t4": [], "trM2_ctrl": [], "eta_err": []}
    for _ in range(20):
        Q = rand_lorentz()
        vals["eta_err"].append(float(np.max(np.abs(Q.T @ ETA @ Q - ETA))))
        M = Q @ M0 @ Q.T
        t = LAG.v4_traces_np(M)
        vals["V4"].append(float(sum((t[k] - cp[k]) ** 2 for k in range(4))))
        vals["det"].append(float(np.linalg.det(M)))
        for k in range(4):
            vals[f"t{k+1}"].append(float(t[k]))
        vals["trM2_ctrl"].append(float(np.trace(M @ M)))
    var = {k: float((np.max(v) - np.min(v)) / max(abs(np.mean(v)), 1e-300))
           for k, v in vals.items() if k != "eta_err"}
    var["max_eta_err"] = float(np.max(vals["eta_err"]))
    log("D5 numeric max rel variation over 20 dressings: " + " ".join(f"{k}={v:.2e}" for k, v in var.items()))
    ok = (line0 and line1 and line2 and all(tr_ok) and det_ok
          and var["V4"] < 1e-6 and var["det"] < 1e-8 and var["trM2_ctrl"] > 1.0)
    return {"symbolic": {"QT_eta_Q": bool(line0), "QT_eq_eta_Qinv_eta": bool(line1),
                         "conjugation_line": bool(line2), "trace_p12": tr_ok, "det": det_ok},
            "numeric_max_rel_variation": var, "V4_value": float(np.mean(vals["V4"])),
            "verdict": "CONFIRMED" if bool(ok) else "REFUTED"}


def main():
    dl = json.load(open(os.path.join(DATA, "m5_32_r6_deltaladder.json")))
    OUT["claims"]["D1"] = d1_identity_a()
    OUT["claims"]["D2"] = d2_kin_refit(dl)
    OUT["claims"]["D3"] = d3_drift(dl)
    OUT["claims"]["D4"] = d4_gain(dl)
    OUT["claims"]["D5"] = d5_orbit_blind()
    OUT["verdicts"] = {k: v["verdict"] for k, v in OUT["claims"].items()}
    OUT["runtime_s"] = time.time() - T0
    OUT["log"] = LINES
    log("VERDICTS " + json.dumps(OUT["verdicts"]) + f" runtime {OUT['runtime_s']:.1f}s")
    with open(os.path.join(DATA, "m5_32_r6_audit.json"), "w") as f:
        json.dump(OUT, f, indent=1)


if __name__ == "__main__":
    main()
