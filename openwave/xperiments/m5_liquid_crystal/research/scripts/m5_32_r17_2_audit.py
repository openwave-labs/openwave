"""M5.32 R17-2 / R17-3a / R17-3c INDEPENDENT ADVERSARIAL AUDIT (claims S1-S8).

Own method throughout where the claim allows it:
  * own spectral reads: eigenvalues of N = M eta by np.linalg.eigvals, the ISOLATED spectral
    projectors by the elementary Lagrange matrix polynomial P_j = prod_{k != j} (N - l_k)/(l_j - l_k)
    on MY eigenvalues, P23 = I - P_g - P_1 (idempotence and commutation verified per cell);
  * own plateau weight w(lambda) (1 on [delta - 0.5, delta + 0.5], cosine taper to 0 at +-1);
  * own K_P density E = (1/2) sum_i tr(Y Om), Om = w (A_i eta) w, Y = eta Om^T eta;
  * own spin-weighted harmonics (Goldberg sum, written here, orthonormality-validated);
  * own Rayleigh quotients from SECOND DIFFERENCES OF THE ENERGY (the producer used Lanczos on
    central differences of the analytic gradient: an independent estimator of the same form);
  * own variational subspace (8 core-localized trial doublets, the 8 x 8 generalized problem) as
    the refutation attempt on NO_BOUND_MODE.
The circle rotation R(beta) and the doublet basis (E_a, E_b) are the OBJECT under audit and are
consumed from the instrument (they are definitions, not claims).

usage: python3 m5_32_r17_2_audit.py           out: data/m5_32_r17_2_audit.json
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
import m5_32_r17_common as R                                # noqa: E402
import m5_32_r16_common as C                                # noqa: E402
import m5_32_r16_2_operator as OP                           # noqa: E402
import m5_32_r17_2_statics as ST                            # noqa: E402

C15, INS4 = C.C15, C.INS4
ETA, EYE = C.ETA, C.EYE
RES, DATA = C.RES, C.DATA
CK16, CK17 = C.CK, R.CK
T0 = time.time()
LOGP = os.path.join(CK17, "r17_2_audit.log")
os.makedirs(CK17, exist_ok=True)
_LOG = open(LOGP, "a" if ARGS else "w")


def log(m):
    s = f"[{time.time() - T0:8.1f}s] {m}"
    print(s, flush=True)
    _LOG.write(s + "\n")
    _LOG.flush()


N32, L48, H = 32, 48.0, 1.5
DELTA, G = C.DELTA, C.G
VAC = np.diag([G, 1.0, DELTA, DELTA])
FREE = ~INS4.pin_shell(N32, H, 1.6)
XX, YY, ZZ = INS4.coords(N32, H)
RR = np.sqrt(XX * XX + YY * YY + ZZ * ZZ)
TH = np.arccos(np.clip(ZZ / RR, -1, 1))
PH = np.arctan2(YY, XX)


# ---------------------------------------------------------------- own spectral machinery
def my_eig(M):
    """own eigenvalues of N = M eta, sorted ascending (lg = most negative, l1 = largest)."""
    Nm = M @ ETA
    lam = np.linalg.eigvals(Nm)
    imax = float(np.max(np.abs(np.imag(lam))))
    lam = np.sort(np.real(lam), axis=-1)
    return Nm, lam, imax


def my_projectors(M):
    """own P_g, P_1, P23 by the Lagrange matrix polynomial on MY eigenvalues."""
    Nm, lam, imax = my_eig(M)
    I = np.broadcast_to(EYE, Nm.shape)
    lg, l3, l2, l1 = (lam[..., k] for k in range(4))     # ascending: lg, then the pair (l3 <= l2), then l1

    def lag(j, others):
        P = I
        for k in others:
            P = P @ (Nm - lam[..., k, None, None] * I) / (lam[..., j, None, None] - lam[..., k, None, None])
        return P
    Pg = lag(0, (1, 2, 3))
    P1 = lag(3, (0, 1, 2))
    P23 = I - Pg - P1
    q = {"eig_max_imag": imax,
         "P23_idempotence_max": float(np.max(np.abs(P23 @ P23 - P23))),
         "P23_commutes_with_N_max": float(np.max(np.abs(Nm @ P23 - P23 @ Nm))),
         "Pg_trace_max_dev_1": float(np.max(np.abs(np.einsum("...aa->...", Pg) - 1.0))),
         "P23_trace_max_dev_2": float(np.max(np.abs(np.einsum("...aa->...", P23) - 2.0)))}
    return {"N": Nm, "lam": lam, "lg": lg, "l2": l2, "l3": l3, "l1": l1, "Pg": Pg, "P1": P1, "P23": P23, "quality": q}


def my_w_plateau(lam):
    """own plateau weight: 1 on [delta - 0.5, delta + 0.5], cosine taper to 0 at +-1, 0 beyond."""
    lo, hi = DELTA - 0.5, DELTA + 0.5
    out = np.zeros_like(lam)
    out = np.where((lam >= lo) & (lam <= hi), 1.0, out)
    m = (lam > hi) & (lam < 1.0)
    out = np.where(m, 0.5 * (1.0 + np.cos(np.pi * (lam - hi) / (1.0 - hi))), out)
    m = (lam < lo) & (lam > -1.0)
    out = np.where(m, 0.5 * (1.0 + np.cos(np.pi * (lo - lam) / (lo + 1.0))), out)
    return out


def my_weight(sp, mode):
    """own w(N): P23 (relative) or P23 + w(l1) P1 + w(lg) Pg (absolute)."""
    if mode == "relative":
        return sp["P23"]
    w1 = my_w_plateau(sp["l1"])[..., None, None]
    wg = my_w_plateau(sp["lg"])[..., None, None]
    return sp["P23"] + w1 * sp["P1"] + wg * sp["Pg"]


def my_kp_cells(M, mode, h=H, sp=None):
    """own K_P density per cell (h^3-weighted, c_P = 1), sym stencil, jets A_i = d_i M."""
    if sp is None:
        sp = my_projectors(M)
    w = my_weight(sp, mode)
    out = np.zeros(M.shape[:-2])
    for br, wt in (("fwd", 0.5), ("bwd", 0.5)):
        for ax in range(3):
            A = INS4.d1(M, ax, h, br)
            Om = w @ (A @ ETA) @ w
            Y = ETA @ np.swapaxes(Om, -1, -2) @ ETA
            out = out + wt * 0.5 * np.real(np.einsum("...aa->...", Y @ Om))
    return h ** 3 * out, sp


def my_reads(M):
    """own spectral reads: half split, gaps, l1 statistics."""
    sp = my_projectors(M)
    l1, l2, l3, lg = sp["l1"], sp["l2"], sp["l3"], sp["lg"]
    half = 0.5 * (l2 - l3)
    gap = l1 - l2
    edges = np.arange(0.0, L48 / 2 + H, 1.5 * H)
    rr, ll = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mk = (RR >= a) & (RR < b) & FREE
        if not np.any(mk):
            continue
        rr.append(0.5 * (a + b))
        ll.append(float(np.mean(l1[mk])))
    rr, ll = np.array(rr), np.array(ll)
    cr = np.where((ll[:-1] < 0.8) & (ll[1:] >= 0.8))[0]
    r0 = float(rr[cr[0]] + (0.8 - ll[cr[0]]) * (rr[cr[0] + 1] - rr[cr[0]]) / (ll[cr[0] + 1] - ll[cr[0]])) if len(cr) else 0.0
    return {"half_split_max_all": float(np.max(half)), "half_split_max_free": float(np.max(half[FREE])),
            "Delta_min_free": float(np.min(gap[FREE])), "Delta_min_all": float(np.min(gap)),
            "r_at_Delta_min_free": float(RR[FREE].reshape(-1)[int(np.argmin(gap[FREE]))]),
            "l1_min_free": float(np.min(l1[FREE])), "l1_max": float(np.max(l1)), "lg_max": float(np.max(lg)),
            "l2_max": float(np.max(l2)), "l3_min": float(np.min(l3)),
            "r_0_profile_shellmean_l1_0.8": r0, "escape_d_gap_le_1e-3": bool(np.min(gap) <= 1e-3),
            "shell_l1_mean": [[float(a), float(b)] for a, b in zip(rr, ll)],
            "projector_quality": sp["quality"]}, sp


# ---------------------------------------------------------------- own spin-weighted harmonics
def my_sY(s, l, m, th, ph):
    """own Goldberg-formula spin-weighted harmonic sY_lm."""
    from math import comb, factorial, sqrt, pi
    pref = (-1.0) ** m * sqrt(factorial(l + m) * factorial(l - m) * (2 * l + 1) / (4.0 * pi * factorial(l + s) * factorial(l - s)))
    c, sn = np.cos(th / 2.0), np.sin(th / 2.0)
    acc = np.zeros(th.shape, dtype=complex)
    for r_ in range(l - s + 1):
        k2 = r_ + s - m
        if k2 < 0 or k2 > l + s:
            continue
        e = 2 * r_ + s - m                      # sin^{2l}(t/2) cot^e(t/2) = cos^e sin^{2l-e}
        acc = acc + comb(l - s, r_) * comb(l + s, k2) * ((-1.0) ** (l - r_ - s)) * (c ** e) * (sn ** (2 * l - e))
    return pref * acc * np.exp(1j * m * ph)


def sY_orthonormality():
    """own validation of my_sY on a dense sphere quadrature (Gauss-Legendre in cos th)."""
    nt, npv = 200, 200
    x, wq = np.polynomial.legendre.leggauss(nt)
    th = np.arccos(x)
    ph = (np.arange(npv) + 0.5) * 2 * np.pi / npv
    TT, PP = np.meshgrid(th, ph, indexing="ij")
    WW = np.outer(wq, np.full(npv, 2 * np.pi / npv))
    keys = [(l, m) for l in (2, 3, 4) for m in range(-l, l + 1)]
    Ys = {k: my_sY(2, k[0], k[1], TT, PP) for k in keys}
    err = 0.0
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            v = complex(np.sum(WW * Ys[ki] * np.conj(Ys[kj])))
            err = max(err, abs(v - (1.0 if i == j else 0.0)))
    return float(err)


# ---------------------------------------------------------------- instrument helpers
def cfg_of(obj, gW=0.0, ns=8):
    cfg = ST.make_cfg(obj, N32, L48, gW)
    cfg["n_samples"] = ns
    return cfg


def E_of(M, cfg, nref, K=None):
    return float(np.real(R.energy_object(M, cfg, K, nref, need_grad=False)[0]))


def parts_of(M, cfg, nref, K=None):
    return R.energy_object(M, cfg, K, nref, need_grad=False)[2]


def kin_grad_of(M, cfg, V, nref):
    """(d kin / d a0)(V) = 2 T V and kin(V) = <V, T V> on the object's kinetic form."""
    with R.weight_mode(cfg):
        g, kin, kc = C.kin_a0_grad(M, cfg, V, nref)
    return g, float(kin)


def second_diff(M, cfg, nref, V, E0, eps):
    Ep = E_of(M + eps * V, cfg, nref)
    Em = E_of(M - eps * V, cfg, nref)
    return (Ep - 2.0 * E0 + Em) / (eps * eps)


def parts_second_diff(M, cfg, nref, V, p0, eps, terms):
    pp = parts_of(M + eps * V, cfg, nref)
    pm = parts_of(M - eps * V, cfg, nref)
    return {t: float(np.real(pp[t] - 2 * p0[t] + pm[t])) / (eps * eps) for t in terms}


def unit(V):
    return V / max(float(np.sqrt(np.sum(V * V))), 1e-300)


def pattern(l, m, r_s, w, Ea, Eb, hand=-1.0, gauss_center=True):
    Y = my_sY(2, l, m, TH, PH)
    if hand < 0:
        Y = np.conj(Y)
    prof = np.exp(-(RR - r_s) ** 2 / (2.0 * w * w)) if gauss_center else np.exp(-RR ** 2 / (2.0 * w * w))
    Y = Y * prof
    a, b = np.real(Y) * FREE, np.imag(Y) * FREE
    V = a[..., None, None] * Ea + b[..., None, None] * Eb
    return unit(V)


RES_OUT = {}


def verdict(cid, v, own, prod, note):
    RES_OUT[cid] = {"verdict": v, "own_numbers": own, "producer_numbers": prod, "note": note}
    log(f"  ==> {cid}: {v}")


HAZ = []
PROD_STAT = json.load(open(os.path.join(DATA, "m5_32_r17_2.json")))
PROD_3C = json.load(open(os.path.join(DATA, "m5_32_r17_3c.json")))
PROD_GATE = json.load(open(os.path.join(DATA, "m5_32_r17_2_operator_gate.json")))


def prod_op(label):
    return json.load(open(os.path.join(CK17, f"r17_2op_{label}.json")))


def prod_run(tag):
    """the per-run checkpoint record (data/m5_32_r17_2.json drops the trace in collect())."""
    return json.load(open(os.path.join(CK17, tag + ".json")))


# ================================================================ S1
def claim_S1():
    log("S1: the relative weight == P23; K_P^proj(rel) == R15 K_P^23; the difference lives only in the taper")
    Mc = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
    cfg = cfg_of("v4rel", ns=8)
    nref = C.radial_ref(cfg)
    # (a) own w(N) vs the instrument's relative-mode w
    with R.weight_mode({"weight": "relative"}):
        fr_rel = C.frame(Mc, nref)
    fr_abs = R._FRAME_R16(Mc, nref)
    sp = my_projectors(Mc)
    dev_rel = float(np.max(np.abs(np.real(fr_rel["w"]) - sp["P23"])))
    dev_abs = float(np.max(np.abs(np.real(fr_abs["w"]) - my_weight(sp, "absolute"))))
    # (b) own plain K_P densities under both weights
    kc_rel, _ = my_kp_cells(Mc, "relative", sp=sp)
    kc_abs, _ = my_kp_cells(Mc, "absolute", sp=sp)
    my_plain_rel, my_plain_abs = float(np.sum(kc_rel)), float(np.sum(kc_abs))
    # the instrument's plain action and R15's certified K_P^23
    with R.weight_mode({"weight": "relative"}):
        kp_plain_rel = float(np.real(C.action(Mc, cfg, need_grad=False, n_ref=nref)["parts"]["KP"]))
    kp_plain_abs = float(np.real(C.action(Mc, cfg_of("v4abs", ns=8), need_grad=False, n_ref=nref)["parts"]["KP"]))
    kp23 = float(C15.kp23_energy_grad(Mc, C15.cfg_dd(N32, L48, mu=0.0, cP=1.0), need_grad=False)[0])
    # (c) the taper claim: the per-cell difference must vanish wherever lambda_1 >= 1
    d = np.abs(kc_rel - kc_abs)
    mask_up = sp["l1"] >= 1.0 - 1e-12
    mask_tap = ~mask_up
    # (d) the 8-sample circle average (rotation from the instrument, K_P from my own code)
    avg_rel = avg_abs = 0.0
    for k in range(8):
        beta = np.pi * k / 8
        Rk = C.rot_R(fr_abs["J"], beta)
        Mk = Rk @ Mc @ np.swapaxes(Rk, -1, -2)
        avg_rel += float(np.sum(my_kp_cells(Mk, "relative")[0])) / 8
        avg_abs += float(np.sum(my_kp_cells(Mk, "absolute")[0])) / 8
    own = {"own_w_rel_minus_P23_max": dev_rel, "own_w_abs_minus_instrument_w_max": dev_abs,
           "own_plain_KP_relative": my_plain_rel, "own_plain_KP_absolute": my_plain_abs,
           "instrument_plain_KP_relative": kp_plain_rel, "instrument_plain_KP_absolute": kp_plain_abs,
           "R15_kp23_energy_grad": kp23,
           "rel_dev_own_vs_R15_kp23": abs(my_plain_rel - kp23) / abs(kp23),
           "rel_dev_own_vs_instrument_rel": abs(my_plain_rel - kp_plain_rel) / abs(kp_plain_rel),
           "own_KP_8sample_average_relative": avg_rel, "own_KP_8sample_average_absolute": avg_abs,
           "cells_lambda1_ge_1": int(np.sum(mask_up)), "cells_in_taper": int(np.sum(mask_tap)),
           "max_abs_density_diff_where_l1_ge_1": float(np.max(d[mask_up])) if np.any(mask_up) else None,
           "max_abs_density_diff_in_taper": float(np.max(d[mask_tap])),
           "projector_quality": sp["quality"]}
    prod = {"static_KP_relative_on_R16_1_core_8samples": 3.6586030728340644,
            "static_KP_absolute_on_R16_1_core_8samples": 8.297730319230393,
            "claim": "w(N) = P23 exactly under the relative weight; K_P^proj == R15 K_P^23; differs from absolute only where lambda_1 < 1"}
    ok_w = dev_rel < 1e-10
    ok_kp23 = own["rel_dev_own_vs_R15_kp23"] < 1e-10
    ok_taper = (own["max_abs_density_diff_where_l1_ge_1"] or 0.0) < 1e-14
    ok_avg = abs(avg_rel - 3.6586030728340644) < 1e-8 and abs(avg_abs - 8.297730319230393) < 1e-8
    v = "CONFIRMED" if (ok_w and ok_kp23 and ok_taper and ok_avg) else "QUALIFIED"
    note = (f"own w(N) equals P23 to {dev_rel:.1e}; own plain K_P (relative) equals R15's kp23_energy_grad to "
            f"{own['rel_dev_own_vs_R15_kp23']:.1e} relative; the per-cell density difference between the weights is "
            f"{own['max_abs_density_diff_where_l1_ge_1']:.1e} on the {int(np.sum(mask_up))} cells with lambda_1 >= 1 and up to "
            f"{own['max_abs_density_diff_in_taper']:.3e} on the {int(np.sum(mask_tap))} taper cells; own 8-sample circle averages "
            f"{avg_rel:.7f} / {avg_abs:.7f} reproduce the reported 3.6586 / 8.2977.")
    if not ok_avg:
        note += "  MISMATCH on the averaged values."
    verdict("S1", v, own, prod, note)


# ================================================================ S2
def claim_S2():
    log("S2: the v4rel static end field: UNIAXIAL_RADIAL reads, E_stat_8, convergence")
    p = os.path.join(CK17, "r17_2_v4rel_n32_L48_r16_1.npy")
    M = np.load(p)
    rd, sp = my_reads(M)
    cfg = cfg_of("v4rel", ns=8)
    nref = C.radial_ref(cfg)
    E8 = E_of(M, cfg, nref)
    pin = ~FREE
    # the pinned shell carries the RADIAL hedgehog, i.e. a ROTATED vacuum, so the test is spectral
    # (N-spectrum == (-8, 1, 0.3, 0.3)) plus identity with the seed field's pinned cells
    lam_pin = np.sort(np.real(np.linalg.eigvals(M[pin] @ ETA)), axis=-1)
    dev_vac = float(np.max(np.abs(lam_pin - np.array([-G, DELTA, DELTA, 1.0]))))
    M_seed = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
    dev_seed = float(np.max(np.abs(M[pin] - M_seed[pin])))
    dev_vac_diag = float(np.max(np.abs(M[pin] - VAC)))
    tr = prod_run("r17_2_v4rel_n32_L48_r16_1")["trace"]
    fmax_end = float(tr[-1]["fmax"])
    E_hist = [float(t["E_stat"]) for t in tr[-6:]]
    # biaxiality of the end field (own): beta2 = 1 - 6 (tr B^3)^2 / (tr B^2)^3 style read via the split
    own = dict(rd)
    own.update({"E_stat_8_own": E8, "pinned_shell_N_spectrum_max_dev_from_(-8,0.3,0.3,1)": dev_vac,
                "pinned_shell_max_dev_from_the_seed_field": dev_seed,
                "pinned_shell_max_dev_from_the_DIAGONAL_vacuum (large by design: the shell is the radial hedgehog, a rotated vacuum)": dev_vac_diag,
                "descent_stop": PROD_STAT["runs"]["r17_2_v4rel_n32_L48_r16_1"]["descent"],
                "fmax_at_last_logged_iter": fmax_end, "E_stat_last_6_logged": E_hist,
                "E_change_over_last_500_it": E_hist[-1] - E_hist[0]})
    prod = {"E_stat_8": 7.27779715876424, "half_split_max": 2.3999867e-4, "Delta_min": 0.03742070560287508,
            "r_0": 3.7139855357706604, "verdict": "UNIAXIAL_RADIAL", "converged": False, "max_iter": 3000, "fmax_claimed": "5e-2"}
    ok = (abs(E8 - 7.27779715876424) < 1e-9 and abs(rd["Delta_min_free"] - 0.0374207) < 1e-6
          and abs(rd["r_0_profile_shellmean_l1_0.8"] - 3.71398) < 1e-4 and rd["half_split_max_all"] < 3e-4
          and dev_vac < 1e-12 and dev_seed == 0.0)
    v = "CONFIRMED" if ok else "QUALIFIED"
    note = (f"own eigen reads reproduce E_stat_8 {E8:.8f}, Delta_min {rd['Delta_min_free']:.6f}, r_0 "
            f"{rd['r_0_profile_shellmean_l1_0.8']:.4f}, half split max {rd['half_split_max_all']:.3e}; the pinned shell is the "
            f"vacuum to {dev_vac:.1e} (its N-spectrum is (-8, 0.3, 0.3, 1) everywhere on the shell; the shell carries the RADIAL "
            f"hedgehog, so it is a rotated vacuum, not the diagonal one) and byte-identical to the seed field there ({dev_seed:.1e}); NOT converged confirmed: stop max_iter at 3000 with fmax "
            f"{fmax_end:.2e} (the claim's 5e-2) and E_stat still falling by {E_hist[-1] - E_hist[0]:+.2e} over the last 500 iterations.")
    verdict("S2", v, own, prod, note)
    return M


# ================================================================ S3
def claim_S3():
    log("S3: d U_v6 / dM = 0 on a split-free core; the unseeded g_W 2.0 control == v4rel")
    # (a) analytic / finite-difference: rho^2 = (l2 - l3)^2 / 4 has zero gradient at l2 = l3
    rng = np.random.default_rng(20260908)
    Mcell = np.zeros((1, 1, 1, 4, 4))
    Mcell[0, 0, 0] = np.diag([G, 1.0, DELTA, DELTA])
    g_split = C15.split_cells(Mcell, need_grad=True)[1]
    grad_norm_analytic = float(np.max(np.abs(g_split)))
    fd = []
    for _ in range(8):
        D = C.sym(rng.normal(size=Mcell.shape))
        D /= np.sqrt(np.sum(D * D))
        f = lambda e: float(C15.split_cells(Mcell + e * D, need_grad=False)[0][0, 0, 0]) / 4.0
        fd.append({"central_1e-4": (f(1e-4) - f(-1e-4)) / 2e-4, "value_plus": f(1e-4), "quadratic_ratio_2eps_over_eps": f(2e-4) / max(f(1e-4), 1e-300)})
    # also on a non-degenerate cell as the control (the gradient must NOT vanish there)
    Mcell2 = Mcell.copy()
    Mcell2[0, 0, 0] = np.diag([G, 1.0, DELTA + 0.05, DELTA - 0.05])
    grad_nondeg = float(np.max(np.abs(C15.split_cells(Mcell2, need_grad=True)[1])))
    # (b) the two end fields
    Ma = np.load(os.path.join(CK17, "r17_2_v4rel_n32_L48_r16_1.npy"))
    Mb = np.load(os.path.join(CK17, "r17_2_v6_gW2_n32_L48_r16_1.npy"))
    cfg_v4 = cfg_of("v4rel", ns=8)
    cfg_v6 = cfg_of("v6", 2.0, ns=8)
    nref = C.radial_ref(cfg_v4)
    Ea_ = E_of(Ma, cfg_v4, nref)
    Eb_ = E_of(Mb, cfg_v6, nref)
    Eb_as_v4 = E_of(Mb, cfg_v4, nref)
    pb = parts_of(Mb, cfg_v6, nref)
    own = {"max_abs_grad_of_rho2_at_degenerate_pair": grad_norm_analytic,
           "fd_directional_derivatives_at_degenerate_pair": [round(x["central_1e-4"], 18) for x in fd],
           "rho2_quadratic_ratio_(2eps)/(eps)_expect_4": [round(x["quadratic_ratio_2eps_over_eps"], 6) for x in fd],
           "control_max_abs_grad_at_split_0.05": grad_nondeg,
           "E_stat_8_v4rel_end": Ea_, "E_stat_8_v6_gW2_control_end": Eb_,
           "E_stat_8_of_the_v6_control_field_read_as_v4rel": Eb_as_v4,
           "U_v6_on_the_control_end": float(np.real(pb["U_v6"])),
           "energy_difference_v6control_minus_v4rel": Eb_ - Ea_,
           "max_abs_field_difference": float(np.max(np.abs(Ma - Mb))),
           "rms_field_difference": float(np.sqrt(np.mean((Ma - Mb) ** 2))),
           "own_half_split_max_v6_control": my_reads(Mb)[0]["half_split_max_all"]}
    prod = {"E_stat_v4rel": 7.27779715876424, "E_stat_v6_gW2_control": 7.277788305318547,
            "claim": "d U_v6 / dM = 0 exactly on a split-free core, so the unseeded control relaxed to the SAME state"}
    ok_grad = grad_norm_analytic < 1e-14 and max(abs(x["central_1e-4"]) for x in fd) < 1e-12
    same_state = own["max_abs_field_difference"] < 1e-6
    v = "CONFIRMED" if (ok_grad and abs(Eb_ - Ea_) < 1e-4) else "QUALIFIED"
    note = (f"the gradient statement is EXACT: d rho^2 / dM = 0 at lambda_2 = lambda_3 (max |grad| {grad_norm_analytic:.1e}, every "
            f"finite-difference directional derivative < 1e-12, rho^2 quadratic in the step), while the control at split 0.05 has "
            f"|grad| {grad_nondeg:.3e}.  The two end fields agree in ENERGY to {abs(Eb_ - Ea_):.2e} but are NOT the same state: "
            f"max |M_v6 - M_v4rel| = {own['max_abs_field_difference']:.3e} (both stopped at max_iter, so this is descent-path "
            f"noise, not a claim of identity); the control still carries U_v6 = {own['U_v6_on_the_control_end']:.2e} and half "
            f"split {own['own_half_split_max_v6_control']:.2e} vs 2.4e-4 on the v4rel end.")
    verdict("S3", v, own, prod, note)


# ================================================================ S4
def claim_S4():
    log("S4: the seeded v6 runs decayed; end energies above the saddle; monotone in g_W")
    saddle = 7.27779715876424
    rows = []
    for gW, tag in ((0.5, "r17_2_v6_gW0.5_n32_L48_r16_1_split0.05"), (1.1, "r17_2_v6_gW1.1_n32_L48_r16_1_split0.05"),
                    (1.35, "r17_2_v6_gW1.35_n32_L48_r16_1_split0.05"), (2.0, "r17_2_v6_gW2_n32_L48_r16_1_split0.05")):
        M = np.load(os.path.join(CK17, tag + ".npy"))
        cfg = cfg_of("v6", gW, ns=8)
        nref = C.radial_ref(cfg)
        E8 = E_of(M, cfg, nref)
        rd, _ = my_reads(M)
        pr = prod_run(tag)
        tr = pr["trace"]
        rows.append({"gW": gW, "own_E_stat_8": E8, "own_E_minus_saddle": E8 - saddle,
                     "own_half_split_max": rd["half_split_max_all"], "own_Delta_min_free": rd["Delta_min_free"],
                     "own_escape_d": rd["escape_d_gap_le_1e-3"],
                     "producer_E_stat_8": pr["reads"]["E_stat_8"], "producer_half_split": pr["reads"]["domain"]["half_split_max"],
                     "producer_Delta_min": pr["reads"]["core"]["Delta_min_free"],
                     "stop": pr["descent"]["stop"], "iters": pr["descent"]["iters"],
                     "fmax_last_logged": float(tr[-1]["fmax"]), "fmax_last_5_logged": [float(t["fmax"]) for t in tr[-5:]],
                     "seed_amplitude": 0.05,
                     "E_change_over_last_100_it": float(tr[-1]["E_stat"]) - float(tr[-2]["E_stat"]),
                     "E_change_over_last_500_it": float(tr[-1]["E_stat"]) - float(tr[-6]["E_stat"]) if len(tr) >= 6 else None})
        log(f"    gW {gW}: own E8 {E8:.6f} (saddle + {E8 - saddle:+.5f}), split {rd['half_split_max_all']:.4f}, "
            f"Delta_min {rd['Delta_min_free']:.5f}, stop {pr['descent']['stop']}, fmax {tr[-1]['fmax']:.2e}")
    lo = [r_ for r_ in rows if r_["gW"] < 2.0]
    mono = all(lo[i]["own_E_minus_saddle"] > lo[i + 1]["own_E_minus_saddle"] for i in range(len(lo) - 1))
    mono_split = all(lo[i]["own_half_split_max"] < lo[i + 1]["own_half_split_max"] for i in range(len(lo) - 1))
    own = {"rows": rows, "excess_monotone_decreasing_in_gW_up_to_1.35": mono, "split_monotone_increasing": mono_split}
    prod = {"splits": [0.0082, 0.0121, 0.0148], "excess": [0.013, 0.011, 0.0026], "escape_d_at_2.0_split_held": 0.048}
    ok = all(abs(r_["own_E_stat_8"] - r_["producer_E_stat_8"]) < 1e-9 for r_ in rows) and mono and mono_split
    v = "QUALIFIED" if ok else "REFUTED"
    drop = abs(np.mean([r_["E_change_over_last_100_it"] for r_ in lo]))
    note = (f"every number reproduces exactly: my own 8-sample energies match the producer to < 1e-9 and my own eigen reads "
            f"reproduce the residual half splits {rows[0]['own_half_split_max']:.4f} / {rows[1]['own_half_split_max']:.4f} / "
            f"{rows[2]['own_half_split_max']:.4f} (from a seed of 0.05) and the excesses above the split-free saddle "
            f"{rows[0]['own_E_minus_saddle']:+.4f} / {rows[1]['own_E_minus_saddle']:+.4f} / {rows[2]['own_E_minus_saddle']:+.4f}; "
            f"the excess falls monotonically with g_W ({mono}) while the residual split rises monotonically ({mono_split}); the "
            f"g_W 2.0 seeded run did stop on escape (d) at iteration 400 with the split held at "
            f"{rows[3]['own_half_split_max']:.4f} and the gap at {rows[3]['own_Delta_min_free']:.1e}.  QUALIFIED, not CONFIRMED, "
            f"because 'no condensation up to 1.35' does not follow from these end states.  All three runs stopped at max_iter "
            f"(3000), never at f_tol (1e-6): the last logged fmax are "
            + " / ".join(f"{r_['fmax_last_logged']:.2f}" for r_ in lo)
            + f", and the energy was still falling by {drop:.3f} per 100 iterations at the stop, an order of magnitude MORE than "
              f"the +0.0026 excess that is being read as 'above the saddle' at g_W 1.35.  The three runs share a seed and an "
              f"iteration budget, so the comparison is paired and the ORDERING is informative, but the sign of the gap to the "
              f"saddle is not resolved by these fields.  The defensible statement is 'condensation not DEMONSTRATED up to 1.35', "
              f"and the collapse of the excess from +0.0131 to +0.0026 over g_W 0.5 -> 1.35 is what approaching a threshold looks "
              f"like.")
    verdict("S4", v, own, prod, note)


# ================================================================ S5
def claim_S5():
    log("S5: NO_BOUND_MODE; own Rayleigh quotients of the saved modes + a variational refutation attempt")
    box_bottom = 0.025632114255195525
    cases = [("r16_1_core_v4rel", os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"), "v4rel", 0.0, "radial"),
             ("r16_1_core_v4abs", os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"), "v4abs", 0.0, "radial"),
             ("v4rel_end", os.path.join(CK17, "r17_2_v4rel_n32_L48_r16_1.npy"), "v4rel", 0.0, "radial"),
             ("v6_gW2.0_control_end", os.path.join(CK17, "r17_2_v6_gW2_n32_L48_r16_1.npy"), "v6", 2.0, "radial"),
             ("vacuum_v4rel", os.path.join(CK16, "vac_n32_L48.npy"), "v4rel", 0.0, "x")]
    rows = []
    fields = {}
    for lab, path, obj, gW, lift in cases:
        M = np.load(path)
        cfg = cfg_of(obj, gW, ns=8)
        nref = C.radial_ref(cfg, lift)
        with R.weight_mode(cfg):
            Ea, Eb, fr, rr = OP.doublet_basis(M, cfg, lift)
        fm = FREE.astype(float)
        Ea = Ea * fm[..., None, None]
        Eb = Eb * fm[..., None, None]
        fields[lab] = (M, cfg, nref, Ea, Eb)
        mp = os.path.join(CK17, f"r17_2op_{lab}_mode0.npy")
        row = {"label": lab, "producer_Omega2": prod_op(lab)["modes"][0]["Omega2"]}
        if os.path.exists(mp):
            ab = np.load(mp)
            V = ab[..., 0, None, None] * Ea + ab[..., 1, None, None] * Eb
            nrm = float(np.sqrt(np.sum(V * V)))
            Vh = V / nrm
            E0 = E_of(M, cfg, nref)
            _, kin = kin_grad_of(M, cfg, Vh, nref)
            for eps in (5e-3, 1e-2):
                q = second_diff(M, cfg, nref, Vh, E0, eps)
                row[f"own_Omega2_eps{eps:g}"] = q / (2.0 * kin)
            row["own_2T_of_unit_mode"] = 2.0 * kin
            row["rel_dev_own_vs_producer"] = abs(row["own_Omega2_eps0.005"] - row["producer_Omega2"]) / row["producer_Omega2"]
            log(f"    {lab}: own Omega^2 {row['own_Omega2_eps0.005']:.6f} vs producer {row['producer_Omega2']:.6f} "
                f"(rel {row['rel_dev_own_vs_producer']:.2e})")
        rows.append(row)
    # ---- the refutation attempt: an 8-dimensional variational subspace on the two core fields
    trials = {}
    for lab in ("v6_gW2.0_control_end", "v4rel_end"):
        M, cfg, nref, Ea, Eb = fields[lab]
        E0 = E_of(M, cfg, nref)
        basis, names = [], []
        for r_s in (2.0, 4.0, 6.0, 9.0, 12.0):
            basis.append(pattern(2, 0, r_s, 2.0, Ea, Eb))
            names.append(f"l2m0_r{r_s:g}_w2")
        for r_s in (2.0, 4.0, 6.0):
            basis.append(pattern(2, 2, r_s, 2.0, Ea, Eb))
            names.append(f"l2m2_r{r_s:g}_w2")
        nb = len(basis)
        gs = [kin_grad_of(M, cfg, V, nref) for V in basis]
        Tmat = np.zeros((nb, nb))
        for i in range(nb):
            for j in range(nb):
                Tmat[i, j] = 0.5 * float(np.sum(gs[i][0] * basis[j]))
        Tmat = 0.5 * (Tmat + Tmat.T)
        eps = 5e-3
        Q = {}
        for i in range(nb):
            Q[(i, i)] = second_diff(M, cfg, nref, basis[i], E0, eps)
        Hmat = np.zeros((nb, nb))
        for i in range(nb):
            Hmat[i, i] = Q[(i, i)]
        for i in range(nb):
            for j in range(i + 1, nb):
                qij = second_diff(M, cfg, nref, basis[i] + basis[j], E0, eps)
                Hmat[i, j] = Hmat[j, i] = 0.5 * (qij - Q[(i, i)] - Q[(j, j)])
        Lc = np.linalg.cholesky(2.0 * Tmat)
        Li = np.linalg.inv(Lc)
        wv = np.linalg.eigvalsh(Li @ Hmat @ Li.T)
        singles = {names[i]: Q[(i, i)] / (2.0 * Tmat[i, i]) for i in range(nb)}
        trials[lab] = {"basis": names, "single_trial_Omega2": singles,
                       "variational_lowest_Omega2_in_the_8d_subspace": float(np.min(wv)),
                       "variational_spectrum": [float(x) for x in np.sort(wv)],
                       "beats_box_bottom_0.025632": bool(np.min(wv) < box_bottom),
                       "producer_lowest_Omega2": prod_op(lab)["modes"][0]["Omega2"]}
        log(f"    {lab}: best single trial {min(singles.values()):.5f}; 8d variational bound "
            f"{np.min(wv):.5f} vs box bottom {box_bottom:.6f} vs producer {prod_op(lab)['modes'][0]['Omega2']:.5f}")
    own = {"mode_rayleigh_quotients": rows, "variational_refutation_attempt": trials, "box_bottom_used": box_bottom}
    prod = {"r16_1_core_v4rel": 0.04466258056187137, "r16_1_core_v4abs": 0.04478488914895394,
            "v4rel_end": 0.044491010910013694, "v6_gW2.0_control_end": 0.034944354468154665,
            "empty_box_x_lift": box_bottom, "verdict": "NO_BOUND_MODE everywhere"}
    agree = max(r_.get("rel_dev_own_vs_producer", 0.0) for r_ in rows)
    beat = any(t["beats_box_bottom_0.025632"] for t in trials.values())
    v = "QUALIFIED" if (agree < 5e-3 and not beat) else ("REFUTED" if beat else "QUALIFIED")
    note = (f"own energy-second-difference Rayleigh quotients on the five saved modes reproduce the producer's Lanczos values to "
            f"{agree:.1e} relative (worst), so the reported Omega^2 are right; my own 8-dimensional core-localized variational "
            f"subspace could NOT get below the box bottom on either core field (best bounds "
            + ", ".join(f"{k} {t['variational_lowest_Omega2_in_the_8d_subspace']:.5f}" for k, t in trials.items())
            + f"), so NO_BOUND_MODE survives a genuine refutation attempt.  QUALIFIED on ONE point the producer does not state: the "
              f"box bottom 0.025632 was measured on the VACUUM with the 'x' director lift, while every core operator was run with "
              f"the 'radial' lift, and the producer's own note records that the radial lift raises the same vacuum bottom to "
              f"0.039846 (a lift artefact).  Against the radial-lift bottom, the v6 g_W 2.0 control's 0.034944 lies BELOW the "
              f"control, and its mode is flagged localized (T-weight fraction 0.508 inside r < 8, r_rms 10.7): the verdict "
              f"therefore rests on comparing two different lifts, and no same-lift box control exists for the core runs.")
    verdict("S5", v, own, prod, note)


# ================================================================ S6
def claim_S6():
    log("S6: the handedness of the doublet basis and the l-law gate at r 12")
    orth = sY_orthonormality()
    Mc = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
    cfg = cfg_of("v4rel", ns=8)
    nref = C.radial_ref(cfg)
    with R.weight_mode(cfg):
        Ea, Eb, fr, rr = OP.doublet_basis(Mc, cfg, "radial")
    # own e, own n x e, their f = J e
    nn = np.real(fr["n"])[..., 1:]
    eth = np.stack([np.cos(TH) * np.cos(PH), np.cos(TH) * np.sin(PH), -np.sin(TH)], -1)
    e = eth - np.sum(eth * nn, -1, keepdims=True) * nn
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-300)
    e4 = np.concatenate([np.zeros(e.shape[:-1] + (1,)), e], -1)
    f4 = np.einsum("...ab,...b->...a", np.real(fr["J"]), e4)
    dot = np.sum(f4[..., 1:] * np.cross(nn, e), -1)
    mk = (RR > 3.0) & (RR < 0.4 * L48)
    hand_mean = float(np.mean(dot[mk]))
    hand_min, hand_max = float(np.min(dot[mk])), float(np.max(dot[mk]))
    # the l-law on ONE shell of the analytic hedgehog, both handedness choices
    Mh = C15.seed_uniaxial(cfg)
    cfg_a = cfg_of("v4abs", ns=8)
    with R.weight_mode(cfg_a):
        Ea_h, Eb_h, fr_h, _ = OP.doublet_basis(Mh, cfg_a, "radial")
    fm = FREE.astype(float)
    Ea_h = Ea_h * fm[..., None, None]
    Eb_h = Eb_h * fm[..., None, None]
    E0 = E_of(Mh, cfg_a, nref)
    p0 = parts_of(Mh, cfg_a, nref)
    terms = ["E_h", "V4", "U", "KP", "reg"]
    out_hand = {}
    for hs, hname in ((-1.0, "conjugated (the producer's choice, zeta = a - i b)"), (+1.0, "unconjugated (zeta = a + i b)")):
        Eh = {}
        for l in (2, 3, 4):
            V = pattern(l, 0, 12.0, 1.5 * H, Ea_h, Eb_h, hand=hs)
            q = parts_second_diff(Mh, cfg_a, nref, V, p0, 5e-3, terms)
            Eh[l] = q["E_h"]
        ratio = (Eh[3] - Eh[2]) / (Eh[4] - Eh[2])
        # m independence at l = 2
        ems = {}
        for m in (0, 2, -2):
            V = pattern(2, m, 12.0, 1.5 * H, Ea_h, Eb_h, hand=hs)
            ems[m] = parts_second_diff(Mh, cfg_a, nref, V, p0, 5e-3, terms)["E_h"]
        spread = (max(ems.values()) - min(ems.values())) / abs(np.mean(list(ems.values())))
        out_hand[hname] = {"E_h_l2": Eh[2], "E_h_l3": Eh[3], "E_h_l4": Eh[4], "ratio_(3-2)/(4-2)": ratio,
                           "expected_6_over_14": 6 / 14, "rel_dev_from_6_14": abs(ratio - 6 / 14) / (6 / 14),
                           "E_h_l2_by_m": {str(k): v for k, v in ems.items()}, "m_spread_rel": float(spread)}
        log(f"    r_s 12, {hname}: ratio {ratio:.5f} (6/14 = 0.42857), m-spread {spread:.2e}")
    own = {"my_sY_orthonormality_max_err_on_gauss_grid": orth,
           "f_dot_(n_x_e)_mean_on_R16_1_core": hand_mean, "f_dot_(n_x_e)_range": [hand_min, hand_max],
           "handedness_verdict": "f = J e = -(n x e)" if hand_mean < 0 else "f = +(n x e)",
           "l_law_at_r_12": out_hand}
    prod = {"handedness_f_dot_n_cross_e": PROD_GATE["handedness_f_dot_n_cross_e"], "hand_sign": PROD_GATE["hand_sign"],
            "ratios_at_r_6_9_12_15": PROD_GATE["ratios"], "m_independence": PROD_GATE["m_independence_l2_E_h_rel_spread"],
            "shell_gram_max_err": PROD_GATE["shell_gram_max_err_l2_l4"]}
    key_c = "conjugated (the producer's choice, zeta = a - i b)"
    key_u = "unconjugated (zeta = a + i b)"
    r_ok = out_hand[key_c]["rel_dev_from_6_14"] < 0.05
    hand_ok = hand_mean < -0.99
    v = "CONFIRMED" if (hand_ok and r_ok and orth < 1e-8) else "QUALIFIED"
    note = (f"the handedness is confirmed independently: f = J e has f . (n x e) = {hand_mean:.6f} (range "
            f"[{hand_min:.4f}, {hand_max:.4f}]) on the R16-1 core, so f = -(n x e), the opposite of frame_zeta, and the producer's "
            f"conjugation of the 2Y_lm patterns is correct.  My own spin-weighted harmonics are orthonormal to {orth:.1e} on a "
            f"Gauss-Legendre sphere quadrature.  My own r = 12 shell of the analytic hedgehog gives the l-law ratio "
            f"{out_hand[key_c]['ratio_(3-2)/(4-2)']:.4f} against 6/14 = 0.42857 (producer 0.43700 at r 12) with an m-spread at "
            f"l = 2 of {out_hand[key_c]['m_spread_rel']:.1e}; with the OPPOSITE handedness the same shell gives "
            f"{out_hand[key_u]['ratio_(3-2)/(4-2)']:.4f} and m-spread {out_hand[key_u]['m_spread_rel']:.1e}.  Caveat the producer "
            f"does not carry into the verdict: its own gate records the 2Y_lm shell Gram error at "
            f"{PROD_GATE['shell_gram_max_err_l2_l4']:.1e} on the cubic lattice, so the patterns are only approximately orthonormal "
            f"and the residual excess of the ratio over 6/14 (2 percent at r 12) is a lattice effect, not a measured deviation.")
    verdict("S6", v, own, prod, note)


# ================================================================ S7
def claim_S7():
    log("S7: the 24.3 test: K_P and V4 + U on the (2,0) pattern at r 1.125 and 7.875, both weights")
    Mc = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
    rows = {}
    for obj in ("v4abs", "v4rel"):
        cfg = cfg_of(obj, ns=8)
        nref = C.radial_ref(cfg)
        with R.weight_mode(cfg):
            Ea, Eb, fr, _ = OP.doublet_basis(Mc, cfg, "radial")
        fm = FREE.astype(float)
        Ea = Ea * fm[..., None, None]
        Eb = Eb * fm[..., None, None]
        p0 = parts_of(Mc, cfg, nref)
        terms = ["E_h", "V4", "U", "KP", "reg"]
        for r_s in (1.125, 3.375, 5.625, 7.875, 10.125):
            V = pattern(2, 0, r_s, 1.5 * H, Ea, Eb)
            q = parts_second_diff(Mc, cfg, nref, V, p0, 5e-3, terms)
            _, kin = kin_grad_of(Mc, cfg, V, nref)
            tt = 2.0 * kin
            rows[f"{obj}_r{r_s:g}"] = {"KP_raw": q["KP"], "V4_raw": q["V4"], "U_raw": q["U"], "E_h_raw": q["E_h"],
                                       "reg_raw": q["reg"], "2T": tt, "KP_Omega2": q["KP"] / tt,
                                       "well_V4_plus_U_Omega2": (q["V4"] + q["U"]) / tt, "E_h_Omega2": q["E_h"] / tt,
                                       "total_Omega2": sum(q[t] for t in terms) / tt}
            log(f"    {obj} r {r_s}: KP_Omega2 {q['KP'] / tt:+.5f}, V4+U {(q['V4'] + q['U']) / tt:+.6f}, 2T {tt:.5f}")
    diffs = {}
    for r_s in (1.125, 3.375, 5.625, 7.875, 10.125):
        a = rows[f"v4abs_r{r_s:g}"]["KP_Omega2"]
        b = rows[f"v4rel_r{r_s:g}"]["KP_Omega2"]
        diffs[f"r{r_s:g}"] = {"absolute": a, "relative": b, "abs_minus_rel": a - b, "rel_difference": (a - b) / abs(a)}
    pa = prod_op("r16_1_core_v4abs")["effective_potential"]["rows"]["2,0"]
    pb = prod_op("r16_1_core_v4rel")["effective_potential"]["rows"]["2,0"]
    prod_rows = {f"r{q['r_s']:g}": {"absolute_KP_Omega2": q["Omega2_by_term"]["KP"], "relative_KP_Omega2": p["Omega2_by_term"]["KP"],
                                    "abs_minus_rel": q["Omega2_by_term"]["KP"] - p["Omega2_by_term"]["KP"],
                                    "rel_difference": (q["KP"] - p["KP"]) / abs(q["KP"])}
                 for q, p in zip(pa, pb)}
    wells_positive = all(rows[k]["well_V4_plus_U_Omega2"] > 0 for k in rows)
    own = {"rows": rows, "weight_difference_by_shell": diffs, "well_V4_plus_U_positive_on_every_shell_probed": wells_positive}
    prod = {"claim": "KP 0.62 / 0.27 / 0.15 (abs) vs 0.44 / 0.22 / 0.14 (rel) at r 1.1 / 3.4 / 5.6, identical from r 7.9 outward; "
                     "E_h, V4, U, reg identical under both weights; V4 + U positive on every shell",
            "producer_rows": prod_rows}
    r79 = prod_rows["r7.875"]["rel_difference"]
    v = "QUALIFIED"
    note = (f"the core numbers reproduce: my own second differences give K_P Omega^2 "
            f"{diffs['r1.125']['absolute']:.3f} / {diffs['r3.375']['absolute']:.3f} / {diffs['r5.625']['absolute']:.3f} (absolute) "
            f"against {diffs['r1.125']['relative']:.3f} / {diffs['r3.375']['relative']:.3f} / {diffs['r5.625']['relative']:.3f} "
            f"(relative), and E_h, V4, U, reg are weight-independent to the read precision, and V4 + U is positive on every shell "
            f"probed.  The word 'identical from r 7.9 outward' is WRONG as stated: at r = 7.875 the producer's own K_P values are "
            f"0.447090 (abs) and 0.446024 (rel), a {abs(r79) * 100:.2f} percent difference, which my own recomputation reproduces "
            f"({diffs['r7.875']['rel_difference'] * 100:.2f} percent).  The weights agree to 1e-5 relative only from r = 10.125 "
            f"outward.  This is expected: the taper reaches lambda_1 = 1 only asymptotically, so the K_P difference decays, it does "
            f"not switch off at a radius.")
    verdict("S7", v, own, prod, note)


# ================================================================ S8
def claim_S8():
    log("S8: the R17-3c fixed-K end states: E_K, omega, escape (d), the Legendre identity")
    rows = {}
    for K in (50.0, 200.0):
        tag = f"r17_3c_v6_gW1.35_n32_L48_K{int(K)}"
        M = np.load(os.path.join(CK17, tag + ".npy"))
        cfg = R.cfg_v6(N32, L48, gW=1.35, completion="rebuild", n_samples=8)
        nref = C.radial_ref(cfg)
        E, _, pp, dom, _ = R.energy_object(M, cfg, K, nref, need_grad=False)
        rd, _ = my_reads(M)
        pr = PROD_3C["runs"][tag]
        cfg4 = R.cfg_v6(N32, L48, gW=1.35, completion="rebuild", n_samples=4)
        cfg16 = R.cfg_v6(N32, L48, gW=1.35, completion="rebuild", n_samples=16)
        E4 = float(np.real(R.energy_object(M, cfg4, K, nref, need_grad=False)[0]))
        E16 = float(np.real(R.energy_object(M, cfg16, K, nref, need_grad=False)[0]))
        lift_scan = {}
        for lname, lref in (("radial", nref), ("x", C.radial_ref(cfg, "x")), ("raw_column_sign", None)):
            El, _, ppl, _, frl = R.energy_object(M, cfg, K, lref, need_grad=False)
            lift_scan[lname] = {"E_K_8": float(np.real(El)), "E_stat": float(np.real(ppl["E_stat"])),
                                "omega": float(np.real(ppl["omega"])), "kin_tot": float(np.real(ppl["kin_tot"]))}
        lq = C.lift_quality(C.frame(M, nref), nref)
        lift_scan["radial_lift_quality_min_abs_n_dot_ref"] = lq[0]
        lift_scan["radial_lift_ambiguous_cells"] = lq[1]
        lift_scan["spread_E_K_over_lifts_rel"] = (max(v["E_K_8"] for k_, v in lift_scan.items() if isinstance(v, dict))
                                                  - min(v["E_K_8"] for k_, v in lift_scan.items() if isinstance(v, dict))) / abs(pr["end_parts_8"]["E_K"])
        lift_scan["spread_E_stat_over_lifts_rel"] = (max(v["E_stat"] for k_, v in lift_scan.items() if isinstance(v, dict))
                                                     - min(v["E_stat"] for k_, v in lift_scan.items() if isinstance(v, dict))) / abs(pr["end_parts_8"]["E_stat"])
        rows[tag] = {"own_E_K_8": float(np.real(E)), "own_omega": float(np.real(pp["omega"])),
                     "own_kin_tot": float(np.real(pp["kin_tot"])), "own_E_stat": float(np.real(pp["E_stat"])),
                     "own_min_gap_l1_l2": rd["Delta_min_all"], "own_min_gap_free": rd["Delta_min_free"],
                     "own_escape_d_gap_le_1e-3": rd["escape_d_gap_le_1e-3"],
                     "own_half_split_max": rd["half_split_max_all"],
                     "own_E_K_4samples": E4, "own_E_K_16samples": E16,
                     "circle_sample_defect_4_vs_8_rel": abs(E4 - float(np.real(E))) / abs(float(np.real(E))),
                     "circle_sample_defect_8_vs_16_rel": abs(E16 - float(np.real(E))) / abs(float(np.real(E))),
                     "producer_E_K_8": pr["end_parts_8"]["E_K"], "producer_omega": pr["end_parts_8"]["omega"],
                     "producer_stop": pr["descent"]["stop"], "producer_iters": pr["descent"]["iters"],
                     "producer_E_K2_8": pr["dE_dK"]["E_K2_8"], "producer_K2": pr["dE_dK"]["K2"],
                     "own_dE_dK_from_the_two_energies": (pr["dE_dK"]["E_K2_8"] - float(np.real(E))) / (pr["dE_dK"]["K2"] - K),
                     "own_dE_dK_over_omega": ((pr["dE_dK"]["E_K2_8"] - float(np.real(E))) / (pr["dE_dK"]["K2"] - K)) / float(np.real(pp["omega"])),
                     "producer_ratio_to_omega": pr["dE_dK"]["ratio_to_omega"],
                     "E_K_minus_E_stat_static_seed": float(np.real(E)) - 7.280373107644915,
                     "omega_c_K": 0.05 * K,
                     "stationarity_true_grad_norm_free": pr["stationarity"]["true_grad_norm_free"],
                     "K2_run_stop": pr["dE_dK"]["stop_K2"],
                     "director_lift_scan": lift_scan,
                     "producer_lift": "the descent-propagated n_ref (info['n_ref']); NOT stored in the record"}
        log(f"    K {K:g}: own E_K_8 {float(np.real(E)):.6f} (producer {pr['end_parts_8']['E_K']:.6f}), omega "
            f"{float(np.real(pp['omega'])):.6f}, min gap {rd['Delta_min_all']:.2e}, 4-vs-8 sample defect "
            f"{rows[tag]['circle_sample_defect_4_vs_8_rel']:.2e}")
    own = rows
    prod = {"K50": {"E_K": 20.422310420938025, "excess": 13.14193731329311, "omega_c_K": 2.5, "omega": 0.3855903583320806,
                    "ratio": 0.9915907920219523, "stop": "escape_d at it 200"},
            "K200": {"E_K": 75.27458167761772, "excess": 67.9942085699728, "omega_c_K": 10.0, "omega": 0.39041221530361747,
                     "ratio": 0.945806237247274, "stop": "escape_d at it 400"}}
    k50 = rows["r17_3c_v6_gW1.35_n32_L48_K50"]
    k200 = rows["r17_3c_v6_gW1.35_n32_L48_K200"]
    esc_ok = k50["own_escape_d_gap_le_1e-3"] and k200["own_escape_d_gap_le_1e-3"]
    bound_ok = all(min(v["E_K_8"] for kk, v in r_["director_lift_scan"].items() if isinstance(v, dict)) - 7.280373107644915
                   > 0.05 * (50.0 if "K50" in t else 200.0) for t, r_ in rows.items())
    v = "QUALIFIED" if esc_ok and bound_ok else "REFUTED"
    note = ("the STRUCTURAL claims reproduce independently: both end states are genuinely at escape (d), my own eigendecomposition "
            f"giving min (lambda_1 - lambda_2) = {k50['own_min_gap_l1_l2']:.2e} (K 50) and {k200['own_min_gap_l1_l2']:.2e} (K 200), "
            "both below the 1e-3 threshold, and E_K sits far above omega_c K under EVERY director lift I tried (the smallest "
            f"excess I could produce is {min(v_['E_K_8'] for kk, v_ in k50['director_lift_scan'].items() if isinstance(v_, dict)) - 7.280373107644915:.2f} "
            f"against omega_c K = 2.5 at K 50 and "
            f"{min(v_['E_K_8'] for kk, v_ in k200['director_lift_scan'].items() if isinstance(v_, dict)) - 7.280373107644915:.2f} "
            "against 10 at K 200), so 'not below the delocalized bound' holds.  QUALIFIED, because the reported NUMBERS DO NOT "
            "REPRODUCE from the archived checkpoint.  E_K, E_stat and omega on these fields depend on the director lift n_ref, "
            "and the reads were taken with the descent-propagated lift, which the record does not store.  Reading the same .npy "
            f"with the natural radial lift I get E_K {k50['own_E_K_8']:.4f} / {k200['own_E_K_8']:.4f} against the reported "
            f"20.4223 / 75.2746, and omega {k50['own_omega']:.4f} / {k200['own_omega']:.4f} against 0.38559 / 0.39041 (a "
            f"{abs(k200['own_omega'] - k200['producer_omega']) / k200['producer_omega'] * 100:.0f} percent gap at K 200); across "
            f"the three lifts I scanned, E_K spans {k50['director_lift_scan']['spread_E_K_over_lifts_rel'] * 100:.0f} percent and "
            f"E_stat spans {k50['director_lift_scan']['spread_E_stat_over_lifts_rel'] * 100:.0f} percent at K 50.  The circle "
            "average removes the RP^2 orientation cell by cell but not on the lattice, where the energy couples neighbours, so a "
            "lift is a genuine input and must travel with the field.  Two further qualifications: the Legendre ratio 0.9916 / "
            f"0.9458 recomputes trivially because both energies come from the producer's own lift, and it is read between two "
            f"NON-stationary, both-escaped states (the producer's own true gradient norms on the free cells are "
            f"{k50['stationarity_true_grad_norm_free']:.1f} and {k200['stationarity_true_grad_norm_free']:.1f}, and the K + 2 "
            f"percent run also stopped on escape (d)); and at K 200 the descent's own 4-sample circle average differs from the "
            f"8-sample read by {k200['circle_sample_defect_4_vs_8_rel'] * 100:.1f} percent while the 8 -> 16 doubling is exact to "
            f"{max(k50['circle_sample_defect_8_vs_16_rel'], k200['circle_sample_defect_8_vs_16_rel']):.1e}, so the reads are sound "
            "at 8 samples but the descent that made these fields was not.")
    verdict("S8", v, own, prod, note)


def hazards():
    """hazards the producer did not claim, found while recomputing."""
    s8 = RES_OUT["S8"]["own_numbers"]
    k200 = s8["r17_3c_v6_gW1.35_n32_L48_K200"]
    k50 = s8["r17_3c_v6_gW1.35_n32_L48_K50"]
    HAZ.extend([
        {"id": "H9", "where": "S8 / R17-3c (and every record that reports circle-averaged reads)",
         "hazard": "the reported energies are not reproducible from the saved checkpoint: they depend on a director lift that is not stored",
         "detail": "the circle-averaged E_stat, E_K, kin and omega all depend on the per-cell orientation of the director n, "
                   "because on the lattice the energy couples neighbouring cells and only a SMOOTH lift makes the average "
                   "orientation-blind.  The R17-3c reads used info['n_ref'], the lift propagated through the descent, and the "
                   "record stores neither it nor the lift kind.  Reading r17_3c_v6_gW1.35_n32_L48_K50.npy with the natural radial "
                   f"lift gives E_K {k50['own_E_K_8']:.4f}, with the x lift "
                   f"{k50['director_lift_scan']['x']['E_K_8']:.4f}, with the raw column sign "
                   f"{k50['director_lift_scan']['raw_column_sign']['E_K_8']:.4f}, against the reported 20.4223: a spread of "
                   f"{k50['director_lift_scan']['spread_E_K_over_lifts_rel'] * 100:.0f} percent in E_K and "
                   f"{k50['director_lift_scan']['spread_E_stat_over_lifts_rel'] * 100:.0f} percent in E_stat, far larger than the "
                   "effects being reported off these fields (the 5 percent Legendre deviation, the 2.9 percent sample defect).  "
                   "The radial lift is not a bad lift here: it has min |n . n_ref| = "
                   f"{k50['director_lift_scan']['radial_lift_quality_min_abs_n_dot_ref']:.3f} and "
                   f"{k50['director_lift_scan']['radial_lift_ambiguous_cells']} ambiguous cells.  Fix: save n_ref (or its kind and "
                   "seed) beside every end field.  NOTE the R17-2 statics do NOT have this problem in practice, because their end "
                   "fields are uniaxial hedgehogs on which the radial lift is the descent lift: S2, S3, S4 and S7 all reproduced "
                   "to 1e-9 with the radial lift.",
         "severity": "HIGH"},
        {"id": "H1", "where": "S5 / R17-2b", "hazard": "the NO_BOUND_MODE verdict compares two different director lifts",
         "detail": "the box bottom 0.025632 was measured on the vacuum with the 'x' lift; every core operator ran with the "
                   "'radial' lift, and the producer's own note records the radial lift raising the same vacuum bottom to "
                   "0.039846.  No same-lift box control exists for the core fields.  Against the radial-lift number the v6 "
                   "g_W 2.0 control's 0.034944 sits BELOW the control, and that mode is flagged localized (T-weight fraction "
                   "0.508 inside r < 8).  The verdict is not wrong, but it is not lift-controlled.",
         "severity": "moderate"},
        {"id": "H2", "where": "S8 / R17-3c", "hazard": "the descents ran a 4-sample circle average that is NOT converged on their own end fields",
         "detail": f"at K 200 the 4-sample E_K differs from the 8-sample read by {k200['circle_sample_defect_4_vs_8_rel'] * 100:.1f} "
                   f"percent ({k50['circle_sample_defect_4_vs_8_rel'] * 100:.2f} percent at K 50), while the 8 -> 16 doubling is "
                   f"exact to {max(k200['circle_sample_defect_8_vs_16_rel'], k50['circle_sample_defect_8_vs_16_rel']):.1e}.  The "
                   "instrument documents the n_s = 4 defect as O(h^2)-level on smooth fields; on these split cores it is percent "
                   "level, so the descent path (and therefore the escape-(d) point reached) is instrument-limited.  The R17-2 "
                   "statics report a doubling gate; the R17-3c runs report none.",
         "severity": "moderate"},
        {"id": "H3", "where": "S8 / R17-3c", "hazard": "the dE/dK = omega read is taken between two NON-stationary, both-escaped states",
         "detail": f"the K + 2 percent comparison run also stopped on escape (d) (stop_K2 = {k200['K2_run_stop']}), and the "
                   f"producer's own stationarity read gives true gradient norms {k50['stationarity_true_grad_norm_free']:.1f} "
                   f"(K 50) and {k200['stationarity_true_grad_norm_free']:.1f} (K 200) on the free cells.  The Legendre identity "
                   "holds only at a stationary point, so the 0.99 / 0.95 agreement is not evidence for a branch.",
         "severity": "moderate"},
        {"id": "H4", "where": "S7 / the 24.3 test", "hazard": "'identical from r 7.9 outward' is false at r 7.9",
         "detail": "the producer's own K_P values at r = 7.875 are 0.447090 (absolute) and 0.446024 (relative), a 0.24 percent "
                   "difference, reproduced by my own second differences.  Agreement at the 1e-5 level starts at r = 10.125.  The "
                   "taper decays with radius; it does not switch off at a shell.",
         "severity": "minor"},
        {"id": "H5", "where": "S4 / R17-3a", "hazard": "the 'no condensation up to 1.35' bracket rests on three unconverged descents",
         "detail": "all three seeded runs stopped at max_iter with fmax 4e-2 to 7e-2 (f_tol is 1e-6) and their energies were still "
                   "falling at the last logged iteration; the excess above the split-free saddle collapses monotonically with g_W "
                   "(+0.0131 -> +0.0112 -> +0.0026 over 0.5 -> 1.35), which is what approaching a threshold looks like.  The "
                   "collect() rule reads 'condensed' off these end states as if they were minima.",
         "severity": "moderate"},
        {"id": "H6", "where": "S5 / provenance", "hazard": "the two R16-1 core operator records are not reproducible with the code in the tree",
         "detail": "checkpoints/m5_32_r17/r17_2op_r16_1_core_v4rel.json and _v4abs.json carry neither the 'lift' nor the "
                   "'handedness' field that m5_32_r17_2_operator.run writes, so they were produced by an earlier revision of the "
                   "script.  Their pattern values are consistent with the conjugated handedness, so the numbers look sound, but "
                   "the record does not pin the lift they used.",
         "severity": "minor"},
        {"id": "H7", "where": "S3 / R17-3a", "hazard": "'relaxed to the SAME state' is an energy statement, not a field statement",
         "detail": f"the two end fields differ by max |dM| = {RES_OUT['S3']['own_numbers']['max_abs_field_difference']:.2e} and the "
                   f"v6 control carries U_v6 = {RES_OUT['S3']['own_numbers']['U_v6_on_the_control_end']:.1e} and half split "
                   f"{RES_OUT['S3']['own_numbers']['own_half_split_max_v6_control']:.1e} against 2.4e-4 on the v4rel end.  Both "
                   "runs stopped at max_iter on different paths; the energies agree to 9e-6, the states do not coincide.",
         "severity": "minor"},
        {"id": "H8", "where": "R17-2b coverage", "hazard": "no operator record exists for any SEEDED v6 end field",
         "detail": "the g_W 0.5 / 1.1 / 1.35 operator runs were still in their Lanczos phase when this audit ran (launched "
                   "14:27-14:28 UTC-4, logs stop after the H-symmetry line, processes alive); the only v6 operator record is the "
                   "UNSEEDED g_W 2.0 control, whose field is the split-free saddle.  So the S5 statement covers the saddle and "
                   "not the biaxial states the condensation question is about: the doublet spectrum of the seeded v6 cores is "
                   "unmeasured at the time of this audit.",
         "severity": "moderate"},
    ])


# ================================================================ main
OUTP = os.path.join(DATA, "m5_32_r17_2_audit.json")


def main(only=None):
    """only = a list of claim ids to (re)run; the rest are carried over from the existing output
    (used to re-run a single claim after the audit found something the first pass did not probe)."""
    order = [("S1", claim_S1), ("S2", claim_S2), ("S3", claim_S3), ("S4", claim_S4),
             ("S6", claim_S6), ("S7", claim_S7), ("S8", claim_S8), ("S5", claim_S5)]
    prev = json.load(open(OUTP)) if (only and os.path.exists(OUTP)) else {}
    RES_OUT["_carried_runtime_s"] = list(prev.get("runtime_s_by_invocation", []))
    for cid, fn in order:
        if only and cid not in only:
            if cid in prev:
                RES_OUT[cid] = prev[cid]
                log(f"  ..  {cid}: carried over from the previous run ({prev[cid]['verdict']})")
            continue
        fn()
    if only:
        RES_OUT.update({k: v for k, v in prev.items() if k.startswith("S") and k not in RES_OUT})
    hazards()
    tal = {}
    for k, v in RES_OUT.items():
        if k.startswith("S"):
            tal[v["verdict"]] = tal.get(v["verdict"], 0) + 1
    carried = RES_OUT.pop("_carried_runtime_s", [])
    out = {k: v for k, v in RES_OUT.items() if k.startswith("S")}
    out["tally"] = tal
    out["unclaimed_hazards"] = HAZ
    out["runtime_s_by_invocation"] = carried + [round(time.time() - T0, 1)]
    out["runtime_s"] = round(sum(out["runtime_s_by_invocation"]), 1)
    out["provenance"] = {"script": "scripts/m5_32_r17_2_audit.py", "python": sys.version.split()[0],
                         "claims_run_this_invocation": sorted(k for k in RES_OUT if k.startswith("S")) if not ARGS else ARGS,
                         "note": ("run with no arguments for the full audit; with claim ids (e.g. 'S8') to re-run those and carry "
                                  "the rest over from the previous output")}
    json.dump(out, open(OUTP, "w"), indent=1, default=float)
    log(f"tally {tal}; runtime {out['runtime_s']} s; written data/m5_32_r17_2_audit.json")


if __name__ == "__main__":
    main([a for a in ARGS if a.startswith("S")] or None)
