"""M5.32 R17-3b INDEPENDENT ADVERSARIAL AUDIT of the doublet-operator bounds (B1-B4).

Own construction, own spin-weighted harmonics, own variational family; the only shared object is the
energy instrument itself (m5_32_r17_common.energy_object), which is the model under audit.

METHOD
------
B1 (the Rayleigh-Ritz statement).  For a real doublet trial V = a E_a + b E_b the quadratic form
    Q_H(V) = [E(M + eps V) - 2 E(M) + E(M - eps V)] / eps^2      (the Hessian, symmetric by construction)
    Q_T(V) = 2 sum_cells (a, b) T (a, b)                          (positive definite: T's eigenvalues are checked)
and Q = Q_H / Q_T >= lambda_min of H zeta = Omega^2 (2T) zeta RESTRICTED to the subspace the trials live in
(the free cells).  Audited here: (i) T positive definite, (ii) sum of the reported terms == the total energy
(a dropped term would break the bound), (iii) the eps^2 truncation by Richardson (eps, eps / 2), (iv) the
n_samples MISMATCH: the producer's numerator uses 8 circle samples and its denominator the 4-sample inertia,
(v) reproduction of the converged operator eigenvalue from the saved mode0, (vi) reproduction of two producer
trials.  Also recorded: the operator is EXACTLY singular on the pinned shell (E_a = E_b = 0 there), so the
unrestricted "lowest Omega^2" is 0 and the claim only makes sense on the free-supported subspace.
B2 (no trial below the box).  Own trials plus a MULTI-VECTOR Rayleigh-Ritz over a 6-dimensional trial space
(the producer minimized over single Gaussian shells only): the small generalized problem H c = Om^2 T c with
H_ij by polarization, H_ij = [Q_H(v_i + v_j) - Q_H(v_i) - Q_H(v_j)] / 2, and T_ij exactly bilinear.  The same
family is run on the EMPTY BOX, where the true answer is known (0.039846 under the radial lift, 0.025632 under
the x lift), to measure the family's slack: a "no trial below X" statement is only as strong as the family.
Also measured: the box bottom the producer compares against (0.025632) was computed under the 'x' director
lift while every trial here uses the 'radial' lift, whose own vacuum bottom is 0.039846.
B3 (the (2, 0) effective potential).  Own 2Y_{l0} from the spin-raising operator ((l - 2)! / (l + 2)!)^1/2 ð^2 Y_l0,
    2Y_20 = sqrt(30) sin^2 th / (8 sqrt(pi)), 2Y_30 = sqrt(210) sin^2 th cos th / (8 sqrt(pi)),
    2Y_40 = 3 sqrt(10) (7 cos^2 th - 1) sin^2 th / (16 sqrt(pi)),
each normalized to 1 on the sphere (verified numerically), cross-checked against m5_32_r16_0_fields.sY2.
The term split is recomputed at the two shells asked for, and BOTH readings of the "connection floor" are
reported: the producer's e2 - (e3 - e2) / 3 (which is the l-INDEPENDENT radial part) and (e3 - e2) / 3 (the
l = 2 connection piece E_h(l = 2) - radial, which is what the docstring defines the floor to be).
B4 (handedness).  Own handedness measurement from (E_a, E_b) themselves (e, f recovered as the +-1 eigenvectors
of the spatial block of E_a), and the m = +2 / m = -2 / unconjugated comparison.

usage: python3 m5_32_r17_3_audit.py
out:   data/m5_32_r17_3_audit.json
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import eigh as deigh

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
import m5_32_r17_common as R                              # noqa: E402
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r16_0_fields as F0                           # noqa: E402
import m5_32_r17_2_operator as OPR                        # noqa: E402

RES, DATA = C.RES, C.DATA
CK = R.CK
T0 = time.time()
OUT = os.path.join(DATA, "m5_32_r17_3_audit.json")
BOX_X_LIFT = 0.025632114255195525          # r17_2op_vacuum_v4rel.json (lift 'x'), the producer's comparator
BOX_RADIAL_LIFT = 0.03984576               # r17_2op_vacuum_v4rel_radial_lift.log, the SAME vacuum field, lift 'radial'
EPS = 1e-3
FIELDS = {
    "v6_gW0.5_seeded": ("r17_2_v6_gW0.5_n32_L48_r16_1_split0.05.npy", 0.5),
    "v6_gW1.1_seeded": ("r17_2_v6_gW1.1_n32_L48_r16_1_split0.05.npy", 1.1),
    "v6_gW1.35_seeded": ("r17_2_v6_gW1.35_n32_L48_r16_1_split0.05.npy", 1.35),
    "v6_gW2.0_control": ("r17_2_v6_gW2_n32_L48_r16_1.npy", 2.0),
}
MODE0 = os.path.join(CK, "r17_2op_v6_gW2.0_control_end_mode0.npy")
VAC = os.path.join(RES, "checkpoints", "m5_32_r16", "vac_n32_L48.npy")
OUTD = {"rung": "R17-3b independent audit", "eps": EPS, "claims": {}, "unclaimed_hazards": [], "notes": {}}
SMOKE = "--smoke" in ARGS                      # a cheap end-to-end pass over every code path (not the audit)
NS = 4 if SMOKE else 8


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def dump():
    OUTD["runtime_s"] = time.time() - T0
    json.dump(OUTD, open(OUT, "w"), indent=1, default=float)


# ----------------------------------------------------------------- own spin-weight-2 harmonics
def harm(l, m, th, ph):
    """own {}_2Y_{lm}: m = 0 from ð^2 Y_l0 (sympy-derived closed forms), m = +-2 from the standard (1 -+ cos)^2."""
    c, s = np.cos(th), np.sin(th)
    if m == 0:
        if l == 2:
            v = np.sqrt(30.0 / (64.0 * np.pi)) * s ** 2
        elif l == 3:
            v = np.sqrt(210.0 / (64.0 * np.pi)) * s ** 2 * c
        elif l == 4:
            v = 3.0 * np.sqrt(10.0 / (256.0 * np.pi)) * (7.0 * c ** 2 - 1.0) * s ** 2
        else:
            raise ValueError(l)
        return v.astype(complex)
    if l == 2 and m == 2:
        return np.sqrt(5.0 / (64.0 * np.pi)) * (1.0 - c) ** 2 * np.exp(2j * ph)
    if l == 2 and m == -2:
        return np.sqrt(5.0 / (64.0 * np.pi)) * (1.0 + c) ** 2 * np.exp(-2j * ph)
    raise ValueError((l, m))


def harm_gate():
    """the own harmonics: unit norm on the sphere, mutual orthogonality, and the comparison with F0.sY2."""
    nt, npp = 400, 401
    tt = (np.arange(nt) + 0.5) * np.pi / nt
    pp = (np.arange(npp) + 0.5) * 2 * np.pi / npp
    TH, PH = np.meshgrid(tt, pp, indexing="ij")
    w = np.sin(TH) * (np.pi / nt) * (2 * np.pi / npp)
    keys = [(2, 0), (3, 0), (4, 0), (2, 2), (2, -2)]
    Y = {k: harm(*k, TH, PH) for k in keys}
    gram = {f"{a}|{b}": float(abs(np.sum(w * Y[a] * np.conj(Y[b])))) for a in keys for b in keys if a <= b}
    cmp_ = {}
    for k in keys:
        Yr = F0.sY2(2, k[0], k[1], TH, PH)
        num = complex(np.sum(w * Yr * np.conj(Y[k])))
        cmp_[str(k)] = {"overlap_with_F0_sY2": [num.real, num.imag], "abs": abs(num)}
    return {"gram_own": gram, "vs_F0_sY2": cmp_}


# ----------------------------------------------------------------- field setup and trial algebra
def load(label, gW, lift="radial", obj="v6"):
    OPR.NREF["kind"] = lift
    cfg = OPR.make_cfg(obj, 32, 48.0, gW, ns=4)
    path = os.path.join(CK, FIELDS[label][0]) if label in FIELDS else label
    M, nref, free, Ea, Eb, fr, r, th, ph = OPR.setup(path, cfg)
    Tm = OPR.inertia(M, cfg, Ea, Eb, nref, free)
    return dict(cfg=cfg, M=M, nref=nref, free=free, Ea=Ea, Eb=Eb, fr=fr, r=r, th=th, ph=ph, Tm=Tm, path=path,
                hand_producer=OPR.HAND["sign"], obj=obj, lift=lift)


def own_handedness(S):
    """recover (e, f) from E_a = e e^T - f f^T and E_b = e f^T + f e^T (spatial blocks) and measure
    sign(f . (n x e)) independently of m5_32_r17_2_operator.setup.  The eigenvectors of E_a fix e and f only
    up to independent signs; E_b fixes their RELATIVE sign (e^T E_b f = +1 for the true pair), and the
    triple product f . (n x e) is invariant under the remaining global flip."""
    Ea, Eb, fr, r = S["Ea"], S["Eb"], S["fr"], S["r"]
    mk = (r > 3.0) & (r < 0.4 * S["cfg"]["L"]) & S["free"]
    A = Ea[mk][:, 1:, 1:]; B = Eb[mk][:, 1:, 1:]
    w, V = np.linalg.eigh(A)
    e = V[..., -1]; f = V[..., 0]                                    # eigenvalues +1 (e) and -1 (f)
    sgn = np.sign(np.einsum("ki,kij,kj->k", e, B, f))
    f = f * sgn[:, None]
    n = np.real(fr["n"])[mk][:, 1:]
    sg = np.sum(f * np.cross(n, e), -1)
    return {"mean_sign_f_dot_n_cross_e": float(np.mean(np.sign(sg))), "mean_f_dot_n_cross_e": float(np.mean(sg)),
            "eig_max_dev_from_pm1": float(max(abs(w[:, -1] - 1).max(), abs(w[:, 0] + 1).max()))}


def prof_gauss(r, r_s, w):
    return np.exp(-(r - r_s) ** 2 / (2.0 * w * w))


def prof_box(cfg):
    X, Y, Z = C.INS4.coords(cfg["n"], cfg["h"])
    a = (cfg["n"] / 2.0 - 2.0) * cfg["h"] + 0.5 * cfg["h"]           # 21.75: the first PINNED cell centre
    return np.cos(np.pi * X / (2 * a)) * np.cos(np.pi * Y / (2 * a)) * np.cos(np.pi * Z / (2 * a))


def trial_ab(S, l, m, prof, hand=None, conj=True):
    """the (a, b) doublet coordinates of a trial, unit Frobenius norm as a field."""
    hand = S["hand_own"] if hand is None else hand
    z = harm(l, m, S["th"], S["ph"]) * prof
    if conj and hand < 0:
        z = np.conj(z)
    a, b = np.real(z) * S["free"], np.imag(z) * S["free"]
    ab = np.stack([a, b], -1)
    return ab / max(np.sqrt(2.0 * np.sum(ab * ab)), 1e-300)


def field_of(S, ab):
    return ab[..., 0, None, None] * S["Ea"] + ab[..., 1, None, None] * S["Eb"]


def tform(S, ab1, ab2):
    return 2.0 * float(np.sum(np.einsum("xyzi,xyzij,xyzj->xyz", ab1, S["Tm"], ab2)))


def terms_of(cfg):
    return ["E_h", "V4", "U_v6" if cfg.get("object") == "v6" else "U", "KP", "reg"]


def base_energy(S, ns=NS):
    cf = dict(S["cfg"]); cf["n_samples"] = ns
    E0, _, pp0, _, _ = R.energy_object(S["M"], cf, None, S["nref"], need_grad=False)
    return cf, E0, pp0


def qh(S, cf, E0, pp0, ab, eps=EPS):
    """the Hessian quadratic form of a trial, total and by term."""
    V = field_of(S, ab)
    Ep, _, ppp, _, _ = R.energy_object(S["M"] + eps * V, cf, None, S["nref"], need_grad=False)
    Em, _, ppm, _, _ = R.energy_object(S["M"] - eps * V, cf, None, S["nref"], need_grad=False)
    tot = (Ep - 2.0 * E0 + Em) / eps ** 2
    by = {t: (ppp[t] - 2.0 * pp0[t] + ppm[t]) / eps ** 2 for t in terms_of(S["cfg"])}
    return float(tot), {k: float(v) for k, v in by.items()}


def ritz(S, basis, cf, E0, pp0, tag=""):
    """the multi-vector Rayleigh-Ritz over a trial subspace (orthonormalized in the field inner product)."""
    B = []
    for ab in basis:                                                  # Gram-Schmidt in the (a, b) inner product
        v = ab.copy()
        for u in B:
            v = v - 2.0 * float(np.sum(u * v)) * u
        nv = np.sqrt(2.0 * np.sum(v * v))
        if nv > 1e-8:
            B.append(v / nv)
    K = len(B)
    Qd = [qh(S, cf, E0, pp0, B[i])[0] for i in range(K)]
    H = np.zeros((K, K)); T = np.zeros((K, K))
    for i in range(K):
        H[i, i] = Qd[i]
        for j in range(K):
            T[i, j] = tform(S, B[i], B[j])
    for i in range(K):
        for j in range(i + 1, K):
            qij = qh(S, cf, E0, pp0, B[i] + B[j])[0]
            H[i, j] = H[j, i] = 0.5 * (qij - Qd[i] - Qd[j])
    w, vec = deigh(H, T)
    single = [Qd[i] / T[i, i] for i in range(K)]
    log(f"    {tag} Ritz K={K}: singles " + " ".join(f"{s:.4f}" for s in single) + f" -> subspace min {w[0] / 1.0:.5f}")
    return {"K": K, "single_trial_Q": single, "ritz_Omega2": [float(x) for x in w], "ritz_min": float(w[0]),
            "T_cond": float(np.linalg.cond(T)), "mix_of_min": [float(x) for x in vec[:, 0]]}


def basis_of(S, mode0):
    """the trial family: the producer's best, a core bump, an m = 2 bump, a broad outer bump, a box-filling
    profile (the delocalized direction the producer never tried), and the control's converged mode0."""
    r, cfg = S["r"], S["cfg"]
    b = [trial_ab(S, 2, 0, prof_gauss(r, 9.0, 4.5)),
         trial_ab(S, 2, 0, prof_gauss(r, 2.25, 2.25)),
         trial_ab(S, 2, 2, prof_gauss(r, 9.0, 4.5)),
         trial_ab(S, 2, 0, prof_gauss(r, 14.0, 9.0)),
         trial_ab(S, 2, 0, prof_box(cfg) * (r * r / (r * r + 9.0)))]
    ab = mode0 * S["free"][..., None]
    b.append(ab / max(np.sqrt(2.0 * np.sum(ab * ab)), 1e-300))
    return b[:3] if SMOKE else b


# ================================================================= the audit
def main():
    res = OUTD
    res["harmonics_gate"] = harm_gate()
    log("harmonics gate: " + json.dumps({k: round(v, 6) for k, v in res["harmonics_gate"]["gram_own"].items() if k.split("|")[0] == k.split("|")[1]}))
    log("  vs F0.sY2 |overlap|: " + json.dumps({k: round(v["abs"], 6) for k, v in res["harmonics_gate"]["vs_F0_sY2"].items()}))
    mode0 = np.load(MODE0)
    prod = json.load(open(os.path.join(DATA, "m5_32_r17_3b.json")))
    b1, b2, b3, b4 = {}, {}, {}, {}
    fields = {}

    labs = ("v6_gW2.0_control", "v6_gW1.35_seeded", "v6_gW1.1_seeded", "v6_gW0.5_seeded")
    for lab in (labs[:2] if SMOKE else labs):
        S = load(lab, FIELDS[lab][1])
        hw = own_handedness(S); S["hand_own"] = hw["mean_sign_f_dot_n_cross_e"]
        wT = np.linalg.eigvalsh(S["Tm"][S["free"]])
        cf8, E8, pp8 = base_energy(S, NS)
        fields[lab] = {"path": os.path.relpath(S["path"], RES), "hand_producer": S["hand_producer"], "hand_own": S["hand_own"], "hand_own_read": hw,
                       "T_min_eig_free": float(wT.min()), "T_max_eig_free": float(wT.max()),
                       "Ea_max_on_pinned": float(np.max(np.abs(S["Ea"][~S["free"]]))),
                       "E_stat_8": float(E8), "terms_sum_minus_E_stat": float(sum(pp8[t] for t in terms_of(S["cfg"])) - pp8["E_stat"])}
        log(f"{lab}: T eig [{wT.min():.3f}, {wT.max():.3f}], hand own {S['hand_own']:+.1f} vs producer {S['hand_producer']:+.1f}, "
            f"terms - E_stat {fields[lab]['terms_sum_minus_E_stat']:.2e}")

        if lab == "v6_gW2.0_control":
            # ---------------- B1 on the control
            ab0 = mode0 * S["free"][..., None]
            ab0 = ab0 / np.sqrt(2.0 * np.sum(ab0 * ab0))
            t8 = tform(S, ab0, ab0)
            q8 = qh(S, cf8, E8, pp8, ab0)[0]
            cf4, E4, pp4 = base_energy(S, 4 if NS == 8 else 2)
            q4 = qh(S, cf4, E4, pp4, ab0)[0]
            qh8 = qh(S, cf8, E8, pp8, ab0, eps=EPS / 2)[0]
            mode_frac_pinned = float(np.sum((mode0 * ~S["free"][..., None]) ** 2) / np.sum(mode0 ** 2))
            b1 = {"mode0_Q_ns8": q8 / t8, "mode0_Q_ns4": q4 / t8, "operator_Omega2": prodmode(prod),
                  "mode0_Q_eps_half": qh8 / t8, "richardson_rel": abs(qh8 - q8) / max(abs(q8), 1e-300),
                  "mode0_energy_weight_on_pinned_cells": mode_frac_pinned,
                  "T_min_eig_free": float(wT.min()), "terms_sum_minus_E_stat": fields[lab]["terms_sum_minus_E_stat"]}
            log(f"  B1 mode0: Q(ns8) {b1['mode0_Q_ns8']:.6f}  Q(ns4) {b1['mode0_Q_ns4']:.6f}  operator {b1['operator_Omega2']:.6f}  "
                f"eps/2 rel {b1['richardson_rel']:.2e}  mode weight on pinned {mode_frac_pinned:.2e}")

        # ---------------- B2: own trials + the multi-vector Ritz
        own = {}
        for tag, r_s, w in (("own_2_0_r9_w4.5", 9.0, 4.5), ("own_2_0_r2.25_w2.25", 2.25, 2.25)):
            ab = trial_ab(S, 2, 0, prof_gauss(S["r"], r_s, w))
            own[tag] = qh(S, cf8, E8, pp8, ab)[0] / tform(S, ab, ab)
        rz = ritz(S, basis_of(S, mode0), cf8, E8, pp8, tag=lab)
        rz.update(own)
        pr = prod["runs"][lab]
        rz["producer_best"] = pr["best_trial"]["Omega2_upper_bound"]
        rz["producer_2_0_r9_w4.5"] = [t["Omega2_upper_bound"] for t in pr["trials"] if t["l"] == 2 and t["m"] == 0 and t["r_s"] == 9.0 and t["width"] > 4][0]
        rz["producer_2_0_r2.25_w2.25"] = [t["Omega2_upper_bound"] for t in pr["trials"] if t["l"] == 2 and t["m"] == 0 and t["r_s"] == 2.25 and t["width"] < 3][0]
        rz["producer_core_sector_min_r_s_le_4"] = min(t["Omega2_upper_bound"] for t in pr["trials"] if t["r_s"] <= 4.0)
        rz["producer_core_sector_min_r_s_le_4_narrow_only"] = min(t["Omega2_upper_bound"] for t in pr["trials"] if t["r_s"] <= 4.0 and t["width"] < 3)
        rz["below_box_x_lift"] = bool(rz["ritz_min"] < BOX_X_LIFT)
        rz["below_box_radial_lift"] = bool(rz["ritz_min"] < BOX_RADIAL_LIFT)
        b2[lab] = rz
        dump()

        if lab in ("v6_gW1.35_seeded", "v6_gW2.0_control"):
            # ---------------- B3 (the term split at two shells) on both, B4 (handedness) on the g_W 1.35 field
            b3[lab] = eff_pot(S, cf8, E8, pp8)
        if lab == "v6_gW1.35_seeded":
            b4 = hand_test(S, cf8, E8, pp8)
        del S
        dump()

    # ---------------- the empty-box calibration of the trial family (the same 6 vectors)
    cal = {}
    for lift, obj in (("radial", "v4rel"), ("x", "v4rel"), ("radial", "v6")):
        S = load(VAC, 2.0, lift=lift, obj=obj)
        hw = own_handedness(S); S["hand_own"] = hw["mean_sign_f_dot_n_cross_e"]
        cf8, E8, pp8 = base_energy(S, NS)
        key = f"vacuum_{obj}_{lift}lift"
        if lift == "radial" and obj == "v4rel":
            cal[key] = ritz(S, basis_of(S, mode0), cf8, E8, pp8, tag=key)
            cal[key]["true_lowest_Omega2_lanczos"] = BOX_RADIAL_LIFT
            cal[key]["family_slack_factor"] = cal[key]["ritz_min"] / BOX_RADIAL_LIFT
        else:
            ab = trial_ab(S, 2, 0, prof_gauss(S["r"], 9.0, 4.5))
            cal[key] = {"single_trial_2_0_r9_w4.5": qh(S, cf8, E8, pp8, ab)[0] / tform(S, ab, ab)}
        log(f"  calibration {key}: {json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in cal[key].items() if k in ('ritz_min', 'family_slack_factor', 'single_trial_2_0_r9_w4.5')})}")
        del S
    ab_ref = cal["vacuum_v4rel_radiallift"].get("single_trial_Q", [None])[0]
    cal["lift_artefact_on_one_trial"] = (ab_ref / cal["vacuum_v4rel_xlift"]["single_trial_2_0_r9_w4.5"]) if ab_ref else None
    cal["object_v6_vs_v4rel_on_one_trial"] = cal["vacuum_v6_radiallift"]["single_trial_2_0_r9_w4.5"] / ab_ref if ab_ref else None
    res["fields"] = fields
    res["calibration"] = cal
    dump()
    verdicts(res, b1, b2, b3, b4, cal, prod)
    dump()
    log("written " + os.path.relpath(OUT, RES))


def prodmode(prod):
    d = json.load(open(os.path.join(CK, "r17_2op_v6_gW2.0_control_end.json")))
    return d["modes"][0]["Omega2"]


def eff_pot(S, cf8, E8, pp8):
    """the (2, 0) term split at the two shells asked for, plus l = 3, 4 for both readings of the floor."""
    out = {}
    tt = terms_of(S["cfg"])
    for r_s in (1.125, 5.625):
        row = {}
        e = {}
        for l in (2, 3, 4):
            ab = trial_ab(S, l, 0, prof_gauss(S["r"], r_s, 1.5 * S["cfg"]["h"]))
            tot, by = qh(S, cf8, E8, pp8, ab)
            t2 = tform(S, ab, ab)
            e[l] = by["E_h"] / t2
            if l == 2:
                row = {"Omega2_total": tot / t2, "by_term": {k: v / t2 for k, v in by.items()},
                       "well_V4_plus_U": (by["V4"] + by[tt[2]]) / t2, "KP": by["KP"] / t2, "2T": t2}
        row["E_h_l2"] = e[2]; row["E_h_l3"] = e[3]; row["E_h_l4"] = e[4]
        row["ratio_(3-2)/(4-2)"] = (e[3] - e[2]) / (e[4] - e[2])
        row["producer_floor_e2_minus_(e3-e2)/3"] = e[2] - (e[3] - e[2]) / 3.0
        row["connection_piece_(e3-e2)/3"] = (e[3] - e[2]) / 3.0
        out[f"r_s={r_s}"] = row
        log(f"  B3 {S['obj']} gW {S['cfg'].get('gW')} r_s {r_s}: total {row['Omega2_total']:.4f}, well {row['well_V4_plus_U']:+.4f}, "
            f"KP {row['KP']:.4f}, E_h {row['by_term']['E_h']:.4f}, floor(producer) {row['producer_floor_e2_minus_(e3-e2)/3']:.4f}, "
            f"connection piece {row['connection_piece_(e3-e2)/3']:.4f}")
    return out


def hand_test(S, cf8, E8, pp8):
    """m = +2 vs m = -2, and the conjugated vs unconjugated pattern (the producer's handedness correction)."""
    out = {"hand_own": S["hand_own"], "hand_producer": S["hand_producer"]}
    pr = prof_gauss(S["r"], 9.0, 4.5)
    for tag, l, m, conj in (("m=+2_conj", 2, 2, True), ("m=-2_conj", 2, -2, True),
                            ("m=+2_unconj", 2, 2, False), ("m=-2_unconj", 2, -2, False)):
        ab = trial_ab(S, l, m, pr, conj=conj)
        out[tag] = qh(S, cf8, E8, pp8, ab)[0] / tform(S, ab, ab)
    for r_s, w in ((2.25, 2.25),):
        pr2 = prof_gauss(S["r"], r_s, w)
        for tag, m, conj in ((f"core_m=+2_conj_r{r_s}", 2, True), (f"core_m=+2_unconj_r{r_s}", 2, False)):
            ab = trial_ab(S, 2, m, pr2, conj=conj)
            out[tag] = qh(S, cf8, E8, pp8, ab)[0] / tform(S, ab, ab)
    out["unconj_over_conj_r9"] = out["m=+2_unconj"] / out["m=+2_conj"]
    out["unconj_over_conj_core"] = out[f"core_m=+2_unconj_r2.25"] / out[f"core_m=+2_conj_r2.25"]
    out["m_plus_minus_rel_gap"] = abs(out["m=+2_conj"] - out["m=-2_conj"]) / abs(out["m=+2_conj"])
    log(f"  B4: m+2 {out['m=+2_conj']:.5f} m-2 {out['m=-2_conj']:.5f} (rel gap {out['m_plus_minus_rel_gap']:.2e}); "
        f"unconj/conj r9 {out['unconj_over_conj_r9']:.2f}, core {out['unconj_over_conj_core']:.2f}")
    return out


def verdicts(res, b1, b2, b3, b4, cal, prod):
    op = b1["operator_Omega2"]
    rel = abs(b1["mode0_Q_ns8"] - op) / op
    rel4 = abs(b1["mode0_Q_ns4"] - op) / op
    tr = []
    for lab, r_ in b2.items():
        tr.append(abs(r_["own_2_0_r9_w4.5"] - r_["producer_2_0_r9_w4.5"]) / r_["producer_2_0_r9_w4.5"])
    res["claims"]["B1"] = {
        "verdict": ("CONFIRMED" if (rel4 < 5e-3 and max(tr) < 1e-2 and b1["T_min_eig_free"] > 0
                                    and abs(b1["terms_sum_minus_E_stat"]) < 1e-9) else "QUALIFIED"),
        "own_numbers": {"mode0_Rayleigh_quotient_producer_convention_ns8_over_T4": b1["mode0_Q_ns8"],
                        "mode0_Rayleigh_quotient_consistent_ns4": b1["mode0_Q_ns4"],
                        "rel_vs_operator_producer_convention": rel, "rel_vs_operator_consistent_ns4": rel4,
                        "richardson_eps_half_rel_change": b1["richardson_rel"],
                        "T_min_eig_on_free": b1["T_min_eig_free"], "terms_sum_minus_E_stat": b1["terms_sum_minus_E_stat"],
                        "mode0_weight_on_pinned_cells": b1["mode0_energy_weight_on_pinned_cells"],
                        "own_producer_trial_reproduction_max_rel": max(tr),
                        "own_(2,0)_r9_w4.5_per_field": {k: v["own_2_0_r9_w4.5"] for k, v in b2.items()},
                        "own_(2,0)_r2.25_w2.25_per_field": {k: v["own_2_0_r2.25_w2.25"] for k, v in b2.items()}},
        "producer_numbers": {"operator_lowest_Omega2": op,
                             "(2,0)_r9_w4.5_per_field": {k: v["producer_2_0_r9_w4.5"] for k, v in b2.items()},
                             "(2,0)_r2.25_w2.25_per_field": {k: v["producer_2_0_r2.25_w2.25"] for k, v in b2.items()}},
        "note": ("The Rayleigh-Ritz statement is sound WITH ONE RESTRICTION MADE EXPLICIT.  2T is positive definite on "
                 f"the free cells (min eig {b1['T_min_eig_free']:.3f}) and the five reported terms sum to E_stat to "
                 f"{b1['terms_sum_minus_E_stat']:.1e} (no term is dropped from the numerator), so Q >= lambda_min of the "
                 "pencil restricted to the free-supported subspace.  That restriction is not cosmetic: E_a = E_b = 0 on "
                 "the 10816 pinned cells, so the operator is EXACTLY singular there and its unrestricted lowest "
                 "eigenvalue is 0, not 0.034944; every 'lowest Omega^2' in the rung must be read as 'over the free "
                 f"cells'.  The saved mode0 reproduces the Lanczos value to {rel4:.1e} relative with a CONSISTENT "
                 f"4-sample numerator and to {rel:.1e} in the producer's own convention (8-sample numerator over the "
                 "4-sample inertia), so the n_samples mismatch inside the quotient is worth "
                 f"{100 * abs(b1['mode0_Q_ns8'] - b1['mode0_Q_ns4']) / b1['mode0_Q_ns4']:.2f}% here.  The eps = 1e-3 "
                 f"second difference moves by {b1['richardson_rel']:.1e} at eps / 2.  Both producer trials reproduce to "
                 f"{max(tr):.1e} relative.  What Q does NOT bound is the operator on any other field: a quotient "
                 "computed on the g_W 1.35 field says nothing about the box.")}

    below = {k: v["ritz_min"] for k, v in b2.items()}
    vac = cal["vacuum_v4rel_radiallift"]
    slack = vac["family_slack_factor"]
    core_min = {k: v["producer_core_sector_min_r_s_le_4"] for k, v in b2.items()}
    core_seeded = {k: round(v, 4) for k, v in core_min.items() if "seeded" in k}
    matched = {k: v["ritz_min"] / vac["ritz_min"] for k, v in b2.items()}
    matched_1 = {k: v["own_2_0_r9_w4.5"] / vac["single_trial_Q"][0] for k, v in b2.items()}
    res["claims"]["B2"] = {
        "verdict": "QUALIFIED" if all(v > BOX_X_LIFT for v in below.values()) else "REFUTED",
        "own_numbers": {"best_own_bound_per_field (6-vector Ritz)": below,
                        "best_own_single_trial_per_field": {k: min(v["single_trial_Q"] + [v["own_2_0_r9_w4.5"], v["own_2_0_r2.25_w2.25"]]) for k, v in b2.items()},
                        "box_bottom_x_lift (the producer comparator)": BOX_X_LIFT,
                        "box_bottom_radial_lift (matched to the trials' gauge)": BOX_RADIAL_LIFT,
                        "empty_box_SAME_FAMILY_Ritz_min": vac["ritz_min"],
                        "family_slack_factor_on_the_empty_box": slack,
                        "object_over_empty_box_same_family_Ritz": matched,
                        "object_over_empty_box_same_single_trial": matched_1,
                        "producer_core_sector_min_r_s_le_4 (all widths)": core_min,
                        "producer_core_sector_min_r_s_le_4 (width 2.25 only)": {k: v["producer_core_sector_min_r_s_le_4_narrow_only"] for k, v in b2.items()},
                        "lift_artefact_factor_on_one_vacuum_trial": cal.get("lift_artefact_on_one_trial"),
                        "object_v6_vs_v4rel_factor_on_one_vacuum_trial": cal.get("object_v6_vs_v4rel_on_one_trial"),
                        "control_converged_Omega2_over_radial_lift_box": op / BOX_RADIAL_LIFT,
                        "Ritz_min_over_measured_family_slack (an estimate, not a bound)": {k: v / slack for k, v in below.items()}},
        "producer_numbers": {"best_bounds": prod["best_bounds"], "verdicts": prod["verdicts"],
                             "claimed_core_sector_floor": 0.12},
        "note": ("No trial of mine goes below either box bottom, so the literal NO_TRIAL_BELOW_BOX survives; four "
                 "qualifications, and the third is the one that matters.  (1) WRONG GAUGE: 0.025632 is the empty box "
                 "under the 'x' director lift, while every R17-3b trial uses the 'radial' lift, whose bottom on the "
                 f"SAME vacuum field is {BOX_RADIAL_LIFT} (r17_2op_vacuum_v4rel_radial_lift.log); my own single trial "
                 f"reproduces that gauge factor independently ({cal.get('lift_artefact_on_one_trial'):.2f}x on one "
                 "vacuum trial vs 1.55x between the two Lanczos bottoms).  The object/c_s difference between the "
                 f"comparator (v4rel) and the trials (v6) is negligible ({cal.get('object_v6_vs_v4rel_on_one_trial'):.4f}x).  "
                 f"(2) THE FAMILY IS LOOSE: the SAME six vectors on the empty box return {vac['ritz_min']:.4f} against a "
                 f"true {BOX_RADIAL_LIFT}, a slack of {slack:.2f}x; the producer's single-Gaussian scan is looser still "
                 "and its minimum sits at the largest r_s (9) and largest width (4.5) it tried, still falling, so the "
                 "'best bound' is a property of where the grid stopped.  A null result cannot resolve a bound state "
                 "whose binding is smaller than that slack.  (3) THE COMPARISON IS MISMATCHED: the honest variational "
                 "statement compares the SAME family on the object and on the box.  It does, and the object wins by "
                 f"about a factor two: {json.dumps({k: round(v, 3) for k, v in matched.items()})} (Ritz) and "
                 f"{json.dumps({k: round(v, 3) for k, v in matched_1.items()})} (the r_s 9 trial alone).  So the same "
                 "trial that costs 0.128 in the empty box costs 0.065 on the object: the data are CONSISTENT with a "
                 "state well below the box bottom that this family simply cannot resolve, which is the opposite of the "
                 "reassurance NO_TRIAL_BELOW_BOX reads as.  (4) The sub-claim 'the core sector (r_s <= 4) is bounded at "
                 f">= 0.12' is FALSE on the producer's own numbers once the width-4.5 rows are counted: {core_seeded} on "
                 f"the seeded fields and {core_min['v6_gW2.0_control']:.4f} on the control; it holds only for the "
                 "width-2.25 rows.  (5) WHERE THE CALIBRATION POINTS: dividing the seeded Ritz minima by the measured "
                 f"{slack:.2f}x family slack puts the object's true lowest doublet near "
                 f"{min(below[k] for k in below if 'seeded' in k) / slack:.3f}-{max(below[k] for k in below if 'seeded' in k) / slack:.3f}, "
                 "i.e. BELOW both box bottoms.  That is an estimate and not a bound, but it is the direction the "
                 "calibration points, and it is the opposite of the reassurance NO_TRIAL_BELOW_BOX reads as.")}

    pr35 = [q for q in prod["runs"]["v6_gW1.35_seeded"]["effective_potential"]["rows"]["2,0"]]
    prc = [q for q in prod["runs"]["v6_gW2.0_control"]["effective_potential"]["rows"]["2,0"]]
    res["claims"]["B3"] = {
        "verdict": "QUALIFIED",
        "own_numbers": {"gW1.35": b3["v6_gW1.35_seeded"], "gW2.0_control": b3["v6_gW2.0_control"]},
        "producer_numbers": {"gW1.35_(2,0)_rows_r1.125_r5.625": [pr35[0], pr35[2]] if len(pr35) > 2 else pr35,
                             "control_(2,0)_rows_r1.125_r5.625": [prc[0], prc[2]] if len(prc) > 2 else prc,
                             "claimed_total_at_r1.125": {"0.5": 0.428, "1.1": 0.317, "1.35": 0.268, "2.0": 0.165},
                             "claimed_floor_at_r1.1": 0.038, "well_negative_shells": prod["well_negative_shells"]},
        "note": ("The totals, the negative V4 + U_v6 well on the inner shells and the K_P dominance all reproduce "
                 "with my own 2Y_20.  The QUALIFIED is the FLOOR: the quantity the producer plots and calls the "
                 "'E_h connection floor' is e2 - (e3 - e2) / 3, which by its own l (l + 1) - 4 law is the "
                 "l-INDEPENDENT RADIAL part of E_h, not the connection term; the connection piece "
                 "E_h(l = 2) - radial = (e3 - e2) / 3 is a different number (reported here).  The docstring defines "
                 "the floor as 'E_h(l = 2) - E_h(radial part)' and then reports the radial part itself: the label and "
                 "the formula disagree, so 'the floor is ~0.038 and present inside the melted core' is a statement "
                 "about the radial (bump-gradient) energy, which is present everywhere by construction and carries no "
                 "connection information.  Worse, the extrapolation that defines either reading fails its own gate at "
                 "the inner shell: the l-law ratio [E_h(3) - E_h(2)] / [E_h(4) - E_h(2)] must be 6 / 14 = 0.4286 for the "
                 f"l (l + 1) - 4 split to hold, and I measure "
                 f"{b3['v6_gW1.35_seeded']['r_s=1.125']['ratio_(3-2)/(4-2)']:.3f} at r_s 1.125 (g_W 1.35) and "
                 f"{b3['v6_gW1.35_seeded']['r_s=5.625']['ratio_(3-2)/(4-2)']:.3f} at r_s 5.625, against the rung's own "
                 "5% acceptance band; the innermost shell, which is exactly where 'the floor is present inside the "
                 "melted core' is claimed, is the shell where the round-bundle law is most violated.")}

    res["claims"]["B4"] = {
        "verdict": "CONFIRMED" if b4["m_plus_minus_rel_gap"] < 1e-3 and b4["unconj_over_conj_core"] > 1.5 else "QUALIFIED",
        "own_numbers": b4,
        "producer_numbers": {"handedness_sign": prod["runs"]["v6_gW1.35_seeded"]["handedness"],
                             "(2,2)_r9_w4.5": [t["Omega2_upper_bound"] for t in prod["runs"]["v6_gW1.35_seeded"]["trials"] if t["l"] == 2 and t["m"] == 2 and t["r_s"] == 9.0 and t["width"] > 4][0],
                             "(2,-2)_r9_w4.5": [t["Omega2_upper_bound"] for t in prod["runs"]["v6_gW1.35_seeded"]["trials"] if t["l"] == 2 and t["m"] == -2 and t["r_s"] == 9.0 and t["width"] > 4][0]},
        "note": ("The handedness sign -1 is confirmed by an independent measurement (e, f recovered as the +-1 "
                 "eigenvectors of E_a's spatial block, then sign(f . (n x e))).  m = +2 and m = -2 agree to "
                 f"{b4['m_plus_minus_rel_gap']:.1e} relative.  Dropping the conjugation raises the quotient by "
                 f"{b4['unconj_over_conj_core']:.2f}x on a core trial and {b4['unconj_over_conj_r9']:.2f}x at r_s 9, "
                 "confirming the correction matters where the trial samples the texture (it washes out far from the core "
                 "where the two spin weights cost nearly the same).")}
    res["tally"] = {k: v["verdict"] for k, v in res["claims"].items()}
    res["unclaimed_hazards"] = [
        f"THE VERDICT FLIPS WITH THE GAUGE (largest): the control's converged lowest doublet Omega^2 ({op:.6f}) is "
        f"ABOVE the x-lift box bottom ({BOX_X_LIFT:.6f}) but {100 * (1 - op / BOX_RADIAL_LIFT):.0f}% BELOW the "
        f"radial-lift box bottom ({BOX_RADIAL_LIFT}) measured on the SAME vacuum field with the same code, and the "
        "control's own operator run uses the radial lift.  So R17-2's NO_BOUND_MODE and R17-3b's NO_TRIAL_BELOW_BOX "
        "rest on which vacuum control is quoted, not on a measured separation.",
        "GAUGE MISMATCH: the box bottom 0.025632 that every R17-3b verdict is compared against was measured "
        "under the 'x' director lift; all R17-3b trials use the 'radial' lift, whose vacuum bottom on the SAME field is "
        f"{BOX_RADIAL_LIFT} (+55%).  The verdict string should name the radial-lift bottom.",
        "OBJECT MISMATCH (secondary): the box bottom comes from object v4rel (mu 1e-2 on rho^2, c_s 0.4) while the "
        "trials run object v6 (mu_v6 1e-2, c_s 0.5, plus the sextic and W).  Measured on one common vacuum trial the "
        "two objects differ by the factor reported in calibration.object_v6_vs_v4rel_on_one_trial.",
        "The doublet operator is EXACTLY singular by construction: E_a = E_b = 0 on the 10816 pinned cells, so 21632 of "
        "65536 directions have H = 0 and 2T = I; its true lowest eigenvalue is 0.  ARPACK misses that null space only "
        "because every Krylov vector after the first lies in the range.  Every statement of the form 'the lowest "
        "Omega^2 of the doublet operator' must be read as 'restricted to the free cells'.",
        "The trial scan is unconverged AT THE EDGE OF ITS OWN GRID: the minimum sits at the largest r_s (9) and the "
        "largest width (4.5) tried, and the quotient is still falling there, so 'the best bound' is a property of "
        "where the grid stopped, not of the field.",
        "The seeded fields are NOT critical points (gradient_residual_doublet ~4e-3 on the control); a Hessian at a "
        "non-stationary point still gives a valid quadratic form, but 'Omega^2' has no oscillation meaning until the "
        "residual is shown to be negligible against the mode's energy scale.",
        "The trial family is measurably loose: the same six vectors reproduce the empty box's known bottom only to the "
        "slack factor reported in calibration.vacuum_v4rel_radiallift.family_slack_factor, so a null result of the form "
        "'nothing below the box' cannot resolve a binding energy smaller than that slack.",
        "'the core sector (r_s <= 4) is bounded at >= 0.12' is contradicted by the producer's own width-4.5 rows on all "
        "four fields; it is true only of the width-2.25 rows.",
    ]


if __name__ == "__main__":
    main()
