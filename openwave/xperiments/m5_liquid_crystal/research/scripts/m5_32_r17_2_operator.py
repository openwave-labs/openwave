"""M5.32 R17-2b / R17-3b / R17-4: the clock (doublet) operator of objects (B), (C), (D) with the ANGULAR
DECOMPOSITION the author's replacement request asks for (25.3, 26.4: the lowest mode decomposed by shell
angular index, the +2 / r^2 connection floor separated from the potential's well, the binding region named),
and the 24.3 test (the K_P^proj diagonal on shells under the absolute and the relative weight).

EQUATIONS
---------
The operator (R16-2, verbatim): per cell delta M = a E_a + b E_b in the oriented pair frame (e, f) of the
outward-lifted director, zeta = a + i b of spin weight 2; H zeta = Omega^2 (2 T) zeta with H the Hessian of the
object's circle-averaged static energy on the doublet subspace (matrix-free central differences of the analytic
gradient, m5_32_r17_common.energy_object), T the per-cell 2 x 2 inertia from three kinetic reads (plus, for
object (D), the rank-one c_X l_d l_d^T read from the X_M kinetic on the same three directions); the lowest k
eigenvalues of T^-1/2 H T^-1/2 by Lanczos, Omega^2 = lambda / 2.  Verdict (R16-2): BOUND_DOUBLET iff
0 < Omega_0^2 < mu / c_P and the mode is localized (T-weighted rms radius < 0.25 L, T-weight fraction inside
r < 8 above 0.5); the box bottom from the empty-box control.
The decomposition.  On shells S_s (width 1.5 h) the mode's zeta is projected on the spin-weighted harmonics
2Y_lm, l = 2, 3, 4 (m5_32_r16_0_fields.sY2, the Goldberg formula): P_lm = |<zeta, 2Y_lm>_S|^2, the l = 2
fraction, <m>, the T-weight per shell (where the mode lives), the binding region = the shells holding the
top half of the T-weight, compared with the core radius r_0 (the shell-mean lambda_1 = 0.8 crossing).
The radial effective potential, term by term.  For the pattern zeta_lm(theta, phi) times a shell bump
B_s(r) = exp(-(r - r_s)^2 / (2 (1.5 h)^2)) (a = Re, b = Im), the quadratic form of each term of the static
energy, <zeta, H_term zeta> = [E_term(M + eps zeta) - 2 E_term(M) + E_term(M - eps zeta)] / eps^2 (8 circle
samples), divided by <zeta, 2 T zeta> from the same three kinetic reads, gives the local Omega^2 contribution
of that term on that shell for that pattern: V_eff(r_s; l, m) split into E_h (the bundle / connection part),
V4 + U (the potential's well), K_P (the author's 24.3 barrier), reg.  The connection floor: on the round
hedgehog the transverse bundle of a degree-one director has the connection-Laplacian spectrum l (l + 1) - 4
= 2, 8, 16 for l = 2, 3, 4 (a spin-weight-2 section), so the E_h differences between l obey
    [E_h(l = 3) - E_h(l = 2)] / [E_h(l = 4) - E_h(l = 2)] = 6 / 14
on every shell and every m (the radial part is l-independent and cancels): the gate of the decomposition,
run on the analytic hedgehog; the floor itself is E_h(l = 2) - E_h(l = 0-like radial part), reported as
E_h(l = 2) - [E_h(l = 3) - E_h(l = 2)] / 3 (the l-independent part extrapolated from the l(l+1) - 4 law).
The 24.3 test: K_P(l = 2) per shell under the absolute and the relative weight on the same core.

usage: python3 m5_32_r17_2_operator.py run --field <path.npy> --object v4rel|v4abs|v6 [--gW 1.1] [--cX 0] --label <lab> [--k 4] [--n 32 --L 48]
       python3 m5_32_r17_2_operator.py decompose --field <path.npy> --object ... --label <lab> [--mode <mode.npy>]
       python3 m5_32_r17_2_operator.py gate      (the analytic-hedgehog gate of the decomposition)
       python3 m5_32_r17_2_operator.py collect
out:   checkpoints/m5_32_r17/r17_2op_<label>.json (+ the mode .npy), data/m5_32_r17_2_operator.json, plots/m5_32_r17_2op_<label>.png
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
import m5_32_r17_common as R                              # noqa: E402
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r16_0_fields as F0                           # noqa: E402
import m5_32_r16_2_operator as OP                         # noqa: E402
import m5_32_r17_2_statics as ST                          # noqa: E402
import m5_32_r17_0_record as REC                          # noqa: E402

C15, INS4 = C.C15, C.INS4
RES, DATA, PLOTS = C.RES, C.DATA, C.PLOTS
CK16, CK = C.CK, R.CK
T0 = time.time()
LS = (2, 3, 4)


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def rel(p):
    return os.path.relpath(p, RES)


def make_cfg(obj, n, L, gW, cX=0.0, ns=4):
    cfg = ST.make_cfg(obj, n, L, gW)
    cfg["n_samples"] = ns
    if cX:
        cfg["cX"] = float(cX)
    return cfg


def inertia(M, cfg, Ea, Eb, nref, free):
    """the per-cell 2 x 2 inertia T (h^3-weighted) of the object's kinetic term on the doublet, plus the X_M rank-one part."""
    with R.weight_mode(cfg):
        _, _, kc_a = C.kin_a0_grad(M, cfg, Ea, nref)
        _, _, kc_b = C.kin_a0_grad(M, cfg, Eb, nref)
        _, _, kc_ab = C.kin_a0_grad(M, cfg, Ea + Eb, nref)
        if cfg.get("cX", 0.0):
            fr = C.frame(M, nref)
            kc_a = kc_a + R.xm_kin_cells(M, Ea, cfg, fr, n_ref=nref)
            kc_b = kc_b + R.xm_kin_cells(M, Eb, cfg, fr, n_ref=nref)
            kc_ab = kc_ab + R.xm_kin_cells(M, Ea + Eb, cfg, fr, n_ref=nref)
    Taa, Tbb = kc_a, kc_b
    Tab = 0.5 * (kc_ab - kc_a - kc_b)
    Tm = np.stack([np.stack([Taa, Tab], -1), np.stack([Tab, Tbb], -1)], -2)
    return Tm


def shells_of(cfg, r):
    h, L = cfg["h"], cfg["L"]
    edges = np.arange(0.0, 0.45 * L, 1.5 * h)
    return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:]) if np.sum((r >= a) & (r < b)) >= 8]


def decompose_mode(ab, M, cfg, Tm, r, th, ph, nref):
    """the mode's shell content in 2Y_lm (l = 2, 3, 4), the T-weight per shell, the binding region."""
    zeta = ab[..., 0] + 1j * HAND["sign"] * ab[..., 1]
    wgt = np.maximum(np.einsum("...i,...ij,...j->...", ab, Tm, ab), 0.0)
    wtot = max(float(np.sum(wgt)), 1e-300)
    rows = []
    for a, b in shells_of(cfg, r):
        m_ = (r >= a) & (r < b)
        z = zeta[m_]; nz = float(np.sum(np.abs(z) ** 2) * 4 * np.pi / z.size)
        P = {}
        for l in LS:
            for mm in range(-l, l + 1):
                Y = F0.sY2(2, l, mm, th[m_], ph[m_])
                P[f"{l},{mm}"] = float(abs(complex(4 * np.pi / z.size * np.sum(z * np.conj(Y)))) ** 2)
        tot = max(sum(P.values()), 1e-300)
        Pl = {l: sum(v for k, v in P.items() if k.startswith(f"{l},")) for l in LS}
        P2 = {mm: P[f"2,{mm}"] for mm in range(-2, 3)}
        t2 = max(sum(P2.values()), 1e-300)
        rows.append({"r": [a, b], "n_cells": int(np.sum(m_)), "T_weight_fraction": float(np.sum(wgt[m_]) / wtot), "rms_zeta": float(np.sqrt(np.mean(np.abs(z) ** 2))),
                     "P_l_over_total_power": {str(l): Pl[l] / max(nz, 1e-300) for l in LS}, "l_fraction_of_captured": {str(l): Pl[l] / tot for l in LS}, "P_2m": {str(mm): P2[mm] for mm in P2},
                     "mean_m_l2": float(sum(mm * P2[mm] for mm in P2) / t2), "chirality_l2": float((P2[2] - P2[-2]) / max(P2[2] + P2[-2], 1e-300))})
    # the binding region: the shells (sorted by T-weight) holding the top half
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["T_weight_fraction"])
    acc, top = 0.0, []
    for i in order:
        top.append(rows[i]["r"]); acc += rows[i]["T_weight_fraction"]
        if acc >= 0.5:
            break
    r_rms = float(np.sqrt(np.sum(wgt * r * r) / wtot))
    return {"shells": rows, "binding_region_top_half_T_weight": top, "T_weighted_r_rms": r_rms, "T_weight_fraction_r_lt_8": float(np.sum(wgt[r < 8.0]) / wtot)}


HAND = {"sign": -1.0}      # the doublet basis (E_a, E_b) of m5_32_r16_2_operator has f = J e = -(n x e): the OPPOSITE handedness of
                           # m5_32_r16_0_fields.frame_zeta (f = n x e), so zeta_frame = a - i b there; measured in setup() (the gate found
                           # the m = +-2 patterns 3 x too stiff before this was caught: they were conjugate, spin-weight -2, sections)


def pattern_field(l, mm, r_s, w, Ea, Eb, r, th, ph, free):
    Y = F0.sY2(2, l, mm, th, ph) * np.exp(-(r - r_s) ** 2 / (2 * w * w))
    if HAND["sign"] < 0:
        Y = np.conj(Y)
    a, b = np.real(Y) * free, np.imag(Y) * free
    V = a[..., None, None] * Ea + b[..., None, None] * Eb
    nrm = np.sqrt(np.sum(V * V))
    return V / max(nrm, 1e-300), np.stack([a, b], -1) / max(nrm, 1e-300)


def effective_potential(M, cfg, Ea, Eb, Tm, r, th, ph, nref, free, eps=1e-3, ls=LS, shells=None):
    """<zeta, H_term zeta> / <zeta, 2 T zeta> per term, shell and pattern (l, m); 8 circle samples."""
    cf8 = dict(cfg); cf8["n_samples"] = 8
    E0, _, pp0, _, _ = R.energy_object(M, cf8, None, nref, need_grad=False)
    terms = ["E_h", "V4", "U_v6" if cfg.get("object") == "v6" else "U", "KP", "reg"]
    w = 1.5 * cfg["h"]
    shells = shells or [0.5 * (a + b) for a, b in shells_of(cfg, r)]
    out = {"terms": terms, "shells_r": shells, "rows": {}}
    for l in ls:
        for mm in ([0, 2, -2] if l == 2 else [0]):
            key = f"{l},{mm}"
            out["rows"][key] = []
            for r_s in shells:
                V, ab = pattern_field(l, mm, r_s, w, Ea, Eb, r, th, ph, free)
                Ep = R.energy_object(M + eps * V, cf8, None, nref, need_grad=False)[2]
                Em = R.energy_object(M - eps * V, cf8, None, nref, need_grad=False)[2]
                q = {t: (Ep[t] - 2 * pp0[t] + Em[t]) / (eps * eps) for t in terms}
                tt = 2.0 * float(np.sum(np.einsum("xyzi,xyzij,xyzj->xyz", ab, Tm, ab)))
                q["total"] = sum(q[t] for t in terms)
                q["2T"] = tt
                q["Omega2_by_term"] = {t: q[t] / max(tt, 1e-300) for t in terms}
                q["Omega2_total"] = q["total"] / max(tt, 1e-300)
                q["r_s"] = r_s
                out["rows"][key].append(q)
            log(f"    pattern {key}: Omega2 by shell " + " ".join(f"{qq['Omega2_total']:+.4f}" for qq in out["rows"][key]))
    # the l-law split of E_h on the m = 0 patterns per shell.  LABEL CORRECTION (the R17-3b audit, 2026-09-09): by the law
    # E_h(l) = R + c [l (l + 1) - 4] the CONNECTION term at l = 2 is 2 c = (e3 - e2) / 3 and the l-INDEPENDENT radial part is
    # R = e2 - (e3 - e2) / 3; the JSON keys below were named the other way round in the run (kept as written so the run's
    # JSONs stay readable: 'connection_floor_E_h' HOLDS the radial part R and 'radial_part_E_h' HOLDS the connection term 2 c;
    # 'floor_Omega2' is R / 2T).  The record quotes the corrected labels.
    if all(f"{l},0" in out["rows"] for l in (2, 3, 4)):
        floor = []
        for i, r_s in enumerate(shells):
            e2, e3, e4 = (out["rows"][f"{l},0"][i]["E_h"] for l in (2, 3, 4))
            tt = out["rows"]["2,0"][i]["2T"]
            ratio = (e3 - e2) / (e4 - e2) if abs(e4 - e2) > 1e-300 else None
            floor.append({"r_s": r_s, "ratio_(3-2)/(4-2)_expected_6/14=0.4286": ratio, "E_h_l2": e2, "connection_floor_E_h": e2 - (e3 - e2) / 3.0, "radial_part_E_h": (e3 - e2) / 3.0 * 0 + e2 - (e2 - (e3 - e2) / 3.0),
                          "floor_Omega2": (e2 - (e3 - e2) / 3.0) / max(tt, 1e-300), "well_Omega2 (V4 + U)": (out["rows"]["2,0"][i]["V4"] + out["rows"]["2,0"][i][terms[2]]) / max(tt, 1e-300), "KP_Omega2": out["rows"]["2,0"][i]["KP"] / max(tt, 1e-300)})
        out["connection_floor_by_shell"] = floor
    return out


NREF = {"kind": "radial"}     # 'x' for the empty box: the outward lift flips across x = 0 on a uniform director and the circle samples of a
                              # perturbation then carry a discontinuity plane (R16-2 ran its vacuum control with the x lift; the radial lift
                              # raised the box bottom from 0.0256 to 0.0398 in the first R17 vacuum run, a lift artefact, rerun with x)


def setup(field, cfg):
    M = np.load(field) if isinstance(field, str) else field
    n = cfg["n"]
    nref = C.radial_ref(cfg, NREF["kind"])
    free = ~INS4.pin_shell(n, cfg["h"], 1.6)
    with R.weight_mode(cfg):
        Ea, Eb, fr, r = OP.doublet_basis(M, cfg, NREF["kind"])
    fm = free.astype(float)
    Ea = Ea * fm[..., None, None]; Eb = Eb * fm[..., None, None]
    X, Y, Z = INS4.coords(n, cfg["h"])
    th = np.arccos(np.clip(Z / r, -1, 1)); ph = np.arctan2(Y, X)
    # the handedness of (e, f = J e) against n x e, measured on the field (n the spatial director, e from E_a's structure)
    nn = np.real(fr["n"])[..., 1:]
    eth = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], -1)
    e = eth - np.sum(eth * nn, -1, keepdims=True) * nn; e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-300)
    e4 = np.concatenate([np.zeros(e.shape[:-1] + (1,)), e], -1)
    f4 = np.einsum("...ab,...b->...a", np.real(fr["J"]), e4)
    sgn = np.sum(f4[..., 1:] * np.cross(nn, e), -1)
    mk = (r > 3.0) & (r < 0.4 * cfg["L"])
    HAND["sign"] = float(np.sign(np.mean(sgn[mk])))
    HAND["mean_f_dot_n_cross_e"] = float(np.mean(sgn[mk]))
    return M, nref, free, Ea, Eb, fr, r, th, ph


LANCZOS = {"tol": 1e-5, "maxiter": 4000}


def run(field, obj, gW, cX, label, n, L, k=4, eps=1e-4, ns=4):
    cfg = make_cfg(obj, n, L, gW, cX, ns)
    M, nref, free, Ea, Eb, fr, r, th, ph = setup(field, cfg)
    shape = M.shape[:3]; N = int(np.prod(shape)); fm = free.astype(float)
    rec = {"label": label, "field": rel(field), "object": obj, "gW": gW, "cX": cX, "weight": cfg.get("weight"), "lift": NREF["kind"], "handedness": HAND["sign"], "n": n, "L": L, "h": cfg["h"], "n_samples": ns, "eps": eps, "k": k}
    log(f"{label}: object {obj} gW {gW} cX {cX} weight {cfg.get('weight')}; field {rel(field)}")
    t = time.time()
    Tm = inertia(M, cfg, Ea, Eb, nref, free)
    wT, VT = np.linalg.eigh(Tm[free])
    rec["T"] = {"min_eig_on_free": float(np.min(wT)), "max_eig_on_free": float(np.max(wT)), "read_s": time.time() - t, "vacuum_region_T_aa_mean_r_gt_0.35L": float(np.mean(Tm[..., 0, 0][(r > 0.35 * L) & free])), "expected_vacuum_T_aa_h3": float(cfg["cP"] * cfg["h"] ** 3)}
    if cX:
        Tm0 = inertia(M, dict(cfg, cX=0.0), Ea, Eb, nref, free)
        rec["T"]["X_M_added_inertia_trace_fraction"] = float(np.sum(np.trace(Tm - Tm0, axis1=-2, axis2=-1)[free]) / np.sum(np.trace(Tm0, axis1=-2, axis2=-1)[free]))
        rec["T"]["X_M_added_inertia_max_cell_fraction"] = float(np.max((np.trace(Tm - Tm0, axis1=-2, axis2=-1) / np.maximum(np.trace(Tm0, axis1=-2, axis2=-1), 1e-300))[free]))
    log(f"  T eig range on free [{np.min(wT):.3e}, {np.max(wT):.3e}] ({time.time() - t:.0f} s)" + (f"; X_M inertia fraction {rec['T']['X_M_added_inertia_trace_fraction']:.3e} (max cell {rec['T']['X_M_added_inertia_max_cell_fraction']:.3e})" if cX else ""))
    Tsafe = Tm.copy(); Tsafe[~free] = np.eye(2)
    wT2, VT2 = np.linalg.eigh(Tsafe); wT2 = np.maximum(wT2, 1e-12)
    Tih = VT2 @ (wT2[..., :, None] ** -0.5 * np.swapaxes(VT2, -1, -2))

    def to_field(x):
        ab = x.reshape(shape + (2,))
        return ab[..., 0, None, None] * Ea + ab[..., 1, None, None] * Eb

    def from_field(Gf):
        return np.stack([np.sum(Gf * Ea, axis=(-1, -2)), np.sum(Gf * Eb, axis=(-1, -2))], -1).reshape(-1)

    def Tih_apply(x):
        return np.einsum("...ij,...j->...i", Tih, x.reshape(shape + (2,))).reshape(-1)
    g0 = R.energy_object(M, cfg, None, nref)[1]
    rec["gradient_residual_doublet"] = float(np.sqrt(np.sum(from_field(g0) ** 2)))
    calls = [0]

    def H_apply(x):
        v = to_field(x)
        gp = R.energy_object(M + eps * v, cfg, None, nref)[1]
        gm = R.energy_object(M - eps * v, cfg, None, nref)[1]
        calls[0] += 1
        return from_field((gp - gm) / (2 * eps)) * fm.reshape(-1).repeat(2)

    def Ht_apply(x):
        return Tih_apply(H_apply(Tih_apply(x)))
    rng = np.random.default_rng(7)
    x1, x2 = rng.normal(size=2 * N), rng.normal(size=2 * N)
    h1, h2 = H_apply(x1), H_apply(x2)
    rec["H_symmetry_rel"] = float(abs(np.dot(x2, h1) - np.dot(x1, h2)) / max(abs(np.dot(x2, h1)), 1e-300))
    log(f"  H symmetry {rec['H_symmetry_rel']:.2e}; gradient residual in the doublet sector {rec['gradient_residual_doublet']:.3e}")
    op = LinearOperator((2 * N, 2 * N), matvec=Ht_apply, dtype=float)
    t = time.time()
    vals, vecs = eigsh(op, k=k, which="SA", tol=LANCZOS["tol"], maxiter=LANCZOS["maxiter"], ncv=max(2 * k + 1, 20))
    rec["lanczos"] = dict(LANCZOS)
    order = np.argsort(vals); vals, vecs = vals[order] / 2.0, vecs[:, order]
    log(f"  lowest {k} Omega^2: {vals} ({calls[0]} H applications, {time.time() - t:.0f} s)")
    modes = []
    for i in range(k):
        ab = Tih_apply(vecs[:, i]).reshape(shape + (2,))
        wgt = np.maximum(np.einsum("...i,...ij,...j->...", ab, Tm, ab), 0.0); ws = max(float(np.sum(wgt)), 1e-300)
        r_rms = float(np.sqrt(np.sum(wgt * r * r) / ws)); frac8 = float(np.sum(wgt[r < 8.0]) / ws)
        modes.append({"Omega2": float(vals[i]), "omega2_clock": float(vals[i]) / 4.0, "T_weighted_r_rms": r_rms, "T_weight_fraction_r_lt_8": frac8, "localized": bool(r_rms < 0.25 * L and frac8 > 0.5)})
    rec["modes"] = modes
    mu = cfg.get("mu_v6", cfg["mu"])
    rec["thresholds"] = {"mu": mu, "Omega_c2_infinite_box (mu / c_P)": mu / cfg["cP"], "box_continuum_estimate_Omega2": mu / cfg["cP"] + (np.pi / L) ** 2}
    m0 = modes[0]
    v = "BOUND_DOUBLET" if (0 < m0["Omega2"] < mu / cfg["cP"] and m0["localized"]) else (f"UNSTABLE_DOUBLET (Morse index >= {sum(1 for m in modes if m['Omega2'] < 0)})" if m0["Omega2"] < 0 else "NO_BOUND_MODE")
    rec["verdict"] = v
    mode0 = Tih_apply(vecs[:, 0]).reshape(shape + (2,))
    np.save(os.path.join(CK, f"r17_2op_{label}_mode0.npy"), mode0)
    rec["decomposition_mode0"] = decompose_mode(mode0, M, cfg, Tm, r, th, ph, nref)
    log(f"  VERDICT {v}; mode0 T-weight r_rms {rec['decomposition_mode0']['T_weighted_r_rms']:.2f}, binding region {rec['decomposition_mode0']['binding_region_top_half_T_weight'][:3]}; l=2 fraction on shells " + " ".join(f"{s['l_fraction_of_captured']['2']:.2f}" for s in rec['decomposition_mode0']['shells'][:6]))
    with R.weight_mode(cfg):
        core, _, _, _, _ = REC.core_reads(M, cfg, C.frame(M, nref)); core.pop("central_line", None)
    rec["core"] = core
    # K_coll of mode0 (the 28.1 law) with f peak-normalized
    amp = np.sqrt(mode0[..., 0] ** 2 + mode0[..., 1] ** 2); f = amp / max(np.max(amp), 1e-300)
    om = np.sqrt(max(m0["Omega2"], 0.0)) / 2.0
    rec["K_coll_mode0"] = {"omega_clock": om, "Delta_min": core["Delta_min_free"], "int_f2": float(cfg["h"] ** 3 * np.sum(f ** 2)), "K_coll": 2 * om * core["Delta_min_free"] ** 2 * float(cfg["h"] ** 3 * np.sum(f ** 2)), "r_0_sqrt_mu": core["r_0_profile (shell-mean lambda_1 = 0.8)"] * np.sqrt(mu)}
    t = time.time()
    rec["effective_potential"] = effective_potential(M, cfg, Ea, Eb, Tm, r, th, ph, nref, free)
    log(f"  effective potential ({time.time() - t:.0f} s)")
    rec["wall_s"] = time.time() - T0
    rec["plot"] = plot(rec, mode0, r, cfg, label)
    json.dump(rec, open(os.path.join(CK, f"r17_2op_{label}.json"), "w"), indent=1, default=float)
    log(f"  written checkpoints/m5_32_r17/r17_2op_{label}.json")
    return rec


def plot(rec, mode0, r, cfg, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    amp = np.sqrt(mode0[..., 0] ** 2 + mode0[..., 1] ** 2)
    sh = rec["decomposition_mode0"]["shells"]
    rm = [0.5 * (s["r"][0] + s["r"][1]) for s in sh]
    ax[0].plot(rm, [s["T_weight_fraction"] for s in sh], "o-", ms=3, label="T-weight fraction of mode 0")
    ax[0].plot(rm, [s["l_fraction_of_captured"]["2"] for s in sh], "s--", ms=3, label="l = 2 fraction")
    ax[0].plot(rm, [s["mean_m_l2"] for s in sh], "x:", ms=3, label="<m> (l = 2)")
    ax[0].axvline(rec["core"]["r_0_profile (shell-mean lambda_1 = 0.8)"], color="r", lw=0.7, ls=":", label="r_0")
    ax[0].set_xlabel("r"); ax[0].legend(fontsize=6); ax[0].set_title(f"{label}: mode 0 (Omega^2 {rec['modes'][0]['Omega2']:.4f}), {rec['verdict']}", fontsize=8)
    ep = rec["effective_potential"]
    for key in ("2,0", "2,2", "3,0", "4,0"):
        if key in ep["rows"]:
            ax[1].plot(ep["shells_r"], [q["Omega2_total"] for q in ep["rows"][key]], "o-", ms=3, label=f"pattern ({key}) total")
    ax[1].axhline(rec["thresholds"]["Omega_c2_infinite_box (mu / c_P)"], color="r", ls="--", lw=0.8, label="mu / c_P")
    ax[1].set_xlabel("shell radius"); ax[1].set_ylabel("local Omega^2"); ax[1].legend(fontsize=6); ax[1].set_title("the radial effective potential by pattern", fontsize=8)
    if "connection_floor_by_shell" in ep:
        fl = ep["connection_floor_by_shell"]
        ax[2].plot([q["r_s"] for q in fl], [q["floor_Omega2"] for q in fl], "o-", ms=3, label="E_h connection floor (l-law)")
        ax[2].plot([q["r_s"] for q in fl], [q["well_Omega2 (V4 + U)"] for q in fl], "s-", ms=3, label="V4 + U well")
        ax[2].plot([q["r_s"] for q in fl], [q["KP_Omega2"] for q in fl], "^-", ms=3, label="K_P (24.3 barrier)")
        ax[2].plot([q["r_s"] for q in fl], [q["ratio_(3-2)/(4-2)_expected_6/14=0.4286"] or np.nan for q in fl], "k:", lw=0.8, label="(3-2)/(4-2) ratio (6/14 on the round bundle)")
        ax[2].axhline(0, color="k", lw=0.5); ax[2].set_xlabel("shell radius"); ax[2].legend(fontsize=6); ax[2].set_title("l = 2, m = 0: floor vs well vs K_P", fontsize=8)
    p = os.path.join(PLOTS, f"m5_32_r17_2op_{label}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    return rel(p)


def gate(n=32, L=48.0):
    """the decomposition gate on the analytic hedgehog: the 2Y_lm shell orthonormality (l = 2..4) and the l-law ratio 6/14 of E_h."""
    cfg = make_cfg("v4abs", n, L, 0.0, ns=4)
    M = C15.seed_uniaxial(cfg)
    M, nref, free, Ea, Eb, fr, r, th, ph = setup(M, cfg)
    out = {"n": n, "L": L, "handedness_f_dot_n_cross_e": HAND["mean_f_dot_n_cross_e"], "hand_sign": HAND["sign"]}
    # orthonormality on a shell
    m_ = (r >= 6.0) & (r < 9.0)
    keys = [(l, mm) for l in LS for mm in range(-l, l + 1)]
    G = np.zeros((len(keys), len(keys)), dtype=complex)
    for i, (l1, m1) in enumerate(keys):
        Y1 = F0.sY2(2, l1, m1, th[m_], ph[m_])
        for j, (l2, m2) in enumerate(keys):
            G[i, j] = 4 * np.pi / Y1.size * np.sum(Y1 * np.conj(F0.sY2(2, l2, m2, th[m_], ph[m_])))
    out["shell_gram_max_err_l2_l4"] = float(np.max(np.abs(G - np.eye(len(keys)))))
    log(f"gate: 2Y_lm (l = 2..4) shell Gram max error {out['shell_gram_max_err_l2_l4']:.3e} on [6, 9)")
    Tm = inertia(M, cfg, Ea, Eb, nref, free)
    ep = effective_potential(M, cfg, Ea, Eb, Tm, r, th, ph, nref, free, shells=[6.0, 9.0, 12.0, 15.0])
    out["ratios"] = [q["ratio_(3-2)/(4-2)_expected_6/14=0.4286"] for q in ep["connection_floor_by_shell"]]
    m_indep = []
    for i in range(4):
        e = [ep["rows"][f"2,{mm}"][i]["E_h"] for mm in (0, 2, -2)]
        m_indep.append(float((max(e) - min(e)) / max(abs(np.mean(e)), 1e-300)))
    out["m_independence_l2_E_h_rel_spread"] = m_indep
    out["floor_by_shell"] = ep["connection_floor_by_shell"]
    out["pass_ratio_6_14_within_5pct"] = bool(all(abs(x - 6 / 14) < 0.05 * 6 / 14 for x in out["ratios"][1:]))
    log(f"gate: (3-2)/(4-2) ratios {out['ratios']} (expected 0.4286); m-spread of E_h at l = 2 {m_indep}; PASS {out['pass_ratio_6_14_within_5pct']}")
    json.dump(out, open(os.path.join(DATA, "m5_32_r17_2_operator_gate.json"), "w"), indent=1, default=float)
    return out


def decompose(field, obj, gW, label, n, L, mode_path):
    cfg = make_cfg(obj, n, L, gW, ns=4)
    M, nref, free, Ea, Eb, fr, r, th, ph = setup(field, cfg)
    Tm = inertia(M, cfg, Ea, Eb, nref, free)
    ab = np.load(mode_path)
    rec = {"label": label, "field": rel(field), "mode": rel(mode_path), "object": obj, "decomposition": decompose_mode(ab, M, cfg, Tm, r, th, ph, nref)}
    rec["effective_potential"] = effective_potential(M, cfg, Ea, Eb, Tm, r, th, ph, nref, free)
    json.dump(rec, open(os.path.join(CK, f"r17_2dec_{label}.json"), "w"), indent=1, default=float)
    log(f"written checkpoints/m5_32_r17/r17_2dec_{label}.json")
    return rec


def collect():
    out = {"rung": "R17-2b / R17-3b / R17-4", "runs": {}}
    for p in sorted(glob.glob(os.path.join(CK, "r17_2op_*.json"))):
        rr = json.load(open(p)); out["runs"][rr["label"]] = rr
    out["verdicts"] = {k: v["verdict"] for k, v in out["runs"].items()}
    out["lowest_Omega2"] = {k: v["modes"][0]["Omega2"] for k, v in out["runs"].items()}
    json.dump(out, open(os.path.join(DATA, "m5_32_r17_2_operator.json"), "w"), indent=1, default=float)
    log(f"collected {len(out['runs'])}: {out['verdicts']}; lowest Omega^2 {out['lowest_Omega2']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "decompose", "gate", "collect"])
    ap.add_argument("--field"); ap.add_argument("--object", default="v4rel"); ap.add_argument("--gW", type=float, default=1.1); ap.add_argument("--cX", type=float, default=0.0)
    ap.add_argument("--label", default="x"); ap.add_argument("--k", type=int, default=4); ap.add_argument("--n", type=int, default=32); ap.add_argument("--L", type=float, default=48.0); ap.add_argument("--modefile")
    ap.add_argument("--nref", default="radial", choices=["radial", "x"])
    ap.add_argument("--tol", type=float, default=1e-5); ap.add_argument("--maxiter", type=int, default=4000)
    a = ap.parse_args(ARGS)
    NREF["kind"] = a.nref
    LANCZOS["tol"], LANCZOS["maxiter"] = a.tol, a.maxiter
    if a.mode == "run":
        run(a.field, a.object, a.gW, a.cX, a.label, a.n, a.L, a.k)
    elif a.mode == "decompose":
        decompose(a.field, a.object, a.gW, a.label, a.n, a.L, a.modefile)
    elif a.mode == "gate":
        gate(a.n, a.L)
    else:
        collect()
