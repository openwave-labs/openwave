"""M5.32 R16-3: the fixed-K descents of the author's object v4 (ledger 6.5 as amended): E_K =
E_stat + K^2 / (4 kin_tot) on the circle-averaged L_v4, a0 the local clock generator refreshed
each step and frozen in the gradient (the R15 protocol), K in {50, 200}, both completions, the
R16-1 end field as the seed (fallback: the R15-M field, stated), with the four pre-registered
escapes watched every 100 iterations and the R14-B / R15 diagnostics at the end.

EQUATIONS: m5_32_r16_common.py.  The escape reads (per diag; the verdict rule pre-registered here):
    (a) B = 0 director sheet (the split leaves the core): max half-split (lambda_2 - lambda_3)/2 < 1e-3
        on the whole box, or the rho^2 weight's rms radius above 0.35 L (the split delocalizes: the R15
        (iii) route, reported as (a-box), a delocalized split filling the box)
    (b) split sheet with tilt: the rho^2-weighted quadrupole of the split's support oblate below -0.20
        (a planar sheet) AND the rho^2-weighted mean of 1 - (n . r_hat)^2 above 0.3 (the director off radial)
    (c) split sheet with (2,3) twist: the sheet as in (b) AND the spin-2 phase of zeta = S_ee - S_ff + 2 i S_ef
        winding on the sheet (the |<m>| of the shell decomposition above 0.5 on the split's shell)
    (d) the director leaves isolation: min (lambda_1 - lambda_2) <= 1e-3 anywhere (the local circle's
        axis undefined; the descent stops there)
    a relative equilibrium (PERIODIC_ORBIT_EXISTS): the descent plateaus (FIRE plateau or f_tol) with none
        of (a) to (d); the stationarity of the TRUE E_K (a0 refreshed) by directional derivatives along
        6 random symmetric directions relative to the gradient scale; omega = K / (2 kin_tot); dE/dK = omega
        by a 2 percent K-perturbation relaxed 200 iterations from the end state; the a0 inertia split by term
    E(K) < omega_c K: the fixed-K energy above the static end state, E_K - E_stat(R16-1 end), against the
        delocalized bound omega_c K with omega_c = sqrt(mu / (4 c_P)) = 0.05 (R16-0 C2, the clock-rate convention)

usage: python3 m5_32_r16_3_fixedk.py relax --n 32 --L 48 --comp rebuild --K 200 --maxit 3000
       python3 m5_32_r16_3_fixedk.py collect
out:   checkpoints/m5_32_r16/r16_3_<comp>_n<n>_L<L>_K<K>.npy / .json, data/m5_32_r16_3.json, plots/m5_32_r16_3_<tag>.png
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r16_0_fields as F0                           # noqa: E402
import m5_32_r16_1_statics as S1                          # noqa: E402
C15, INS4 = C.C15, C.INS4
RES, DATA, PLOTS, CK = C.RES, C.DATA, C.PLOTS, C.CK
T0 = time.time()
OMEGA_C = float(np.sqrt(1e-2 / 4.0))


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def rel(p):
    return os.path.relpath(p, RES)


def tag_of(comp, n, L, K):
    return f"r16_3_{comp}_n{n}_L{int(L)}_K{int(K)}"


def seed_for(comp, n, L, cfg):
    for kind in ("r15", "analytic"):
        p = os.path.join(CK, S1.tag_of(comp, n, L, kind) + ".npy")
        j = os.path.join(CK, S1.tag_of(comp, n, L, kind) + ".json")
        if os.path.exists(p) and os.path.exists(j):
            r = json.load(open(j))
            if "gates" in r:
                return np.load(p), rel(p), f"the R16-1 end field, seed kind {kind} ({r['descent']['stop']} after {r['descent']['iters']} it)", r["gates"]["reads"][comp]["parts_8"]["E_stat"]
    M, src, how = S1.seed_for(n, L, cfg)
    return M, src, "FALLBACK: " + how + " (the R16-1 end field not available)", None


def escape_reads(M, cfg, fr=None):
    """the four escape diagnostics on a field."""
    n, h, L = cfg["n"], cfg["h"], cfg["L"]
    X, Y, Z = INS4.coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    if fr is None:
        fr = C.frame(M, C.radial_ref(cfg))
    dom = C.domain(fr, cfg)
    trip, lg, disc = F0.spatial_triple(M)
    half = np.sqrt(np.maximum(disc, 0.0)) / 2.0
    rho2 = half * half
    wsum = max(float(np.sum(rho2)), 1e-300)
    xs = np.stack([X, Y, Z], -1) / np.maximum(r, 1e-300)[..., None]
    Qd = np.einsum("xyz,xyzi,xyzj->ij", rho2, xs, xs) / wsum - np.eye(3) / 3.0
    ev, evec = np.linalg.eigh(Qd)
    r_rms = float(np.sqrt(np.sum(rho2 * r * r) / wsum))
    nsp = np.real(fr["n"][..., 1:])
    nsp = nsp / np.maximum(np.linalg.norm(nsp, axis=-1, keepdims=True), 1e-300)
    tilt = 1.0 - np.sum(nsp * xs, -1) ** 2
    tilt_w = float(np.sum(rho2 * tilt) / wsum)
    # the spin-2 phase on the split's shell (the rho^2-weighted mean radius)
    zeta, gap, th, ph, rr = F0.frame_zeta(M, X, Y, Z)
    r_mean = float(np.sum(rho2 * r) / wsum)
    m3 = (r >= r_mean - 0.75 * h) & (r < r_mean + 0.75 * h)
    mean_m, l2f = None, None
    if np.sum(m3) >= 8:
        dec = F0.shell_decomp(zeta[m3], th[m3], ph[m3])
        P = {mm: abs(dec[mm]) ** 2 for mm in dec}
        tot = max(sum(P.values()), 1e-300)
        mean_m = float(sum(mm * P[mm] for mm in P) / tot)
        power = float(np.sum(np.abs(zeta[m3]) ** 2) * 4 * np.pi / np.sum(m3))
        l2f = float(tot / max(power, 1e-300))
    out = {"half_split_max": float(np.max(half)), "r_at_half_split_max": float(r.reshape(-1)[int(np.argmax(half))]), "rho2_r_rms": r_rms, "rho2_r_mean": r_mean,
           "rho2_quadrupole": [float(x) for x in ev], "rho2_quadrupole_axis": [float(x) for x in evec[:, 0]], "tilt_weighted": tilt_w, "spin2_mean_m_on_split_shell": mean_m, "spin2_l2_fraction": l2f,
           "rho2_fraction_r_gt_0.35L": float(np.sum(rho2[r > 0.35 * L]) / wsum), "gap_1_2_min": dom["gap_1_2_min"], "l1_min": dom["l1_min"], "r_d": dom.get("r_d"), "cells_director_in_plateau": dom["cells_director_in_plateau"]}
    sheet = ev[0] < -0.20
    out["escape_a"] = bool(out["half_split_max"] < 1e-3)
    out["escape_a_box"] = bool(r_rms > 0.35 * L)
    out["escape_b"] = bool(sheet and tilt_w > 0.3)
    out["escape_c"] = bool(sheet and mean_m is not None and abs(mean_m) > 0.5)
    out["escape_d"] = bool(dom["escape_d"])
    return out


def make_diag(cfg):
    def diag(M, fr):
        e = escape_reads(M, cfg, fr)
        return {"esc_" + k: v for k, v in e.items() if k.startswith("escape") or k in ("half_split_max", "r_at_half_split_max", "rho2_r_rms", "tilt_weighted", "spin2_mean_m_on_split_shell", "rho2_fraction_r_gt_0.35L")}
    return diag


def stationarity(M, cfg, K, nref, nd=6, eps=1e-4, seed=0):
    """directional derivatives of the TRUE E_K (a0 refreshed) along random symmetric directions,
    the frozen-a0 gradient's scale, and the a0 inertia split by term."""
    rng = np.random.default_rng(seed)
    free = ~INS4.pin_shell(cfg["n"], cfg["h"], 1.6)
    E0, g, pp, dom, fr = C.energy_and_grad(M, cfg, K, nref)
    gs = float(np.sqrt(np.sum((g * free[..., None, None]) ** 2)))
    rows = []
    for _ in range(nd):
        D = C.sym(rng.normal(size=M.shape)) * free[..., None, None]
        D /= np.sqrt(np.sum(D * D))
        Ep = C.energy_and_grad(M + eps * D, cfg, K, nref, need_grad=False)[0]
        Em = C.energy_and_grad(M - eps * D, cfg, K, nref, need_grad=False)[0]
        rows.append({"true_dE": (Ep - Em) / (2 * eps), "frozen_dE": float(np.sum(g * D))})
    return {"E_K": E0, "parts": pp, "frozen_grad_norm": gs, "directions": rows, "true_dE_max": max(abs(r["true_dE"]) for r in rows), "frozen_dE_max": max(abs(r["frozen_dE"]) for r in rows),
            "inertia_split": {k: pp[k] for k in ("kin_h", "kin_KP", "kin_reg", "kin_tot")}, "omega": pp["omega"]}


def plot_run(tag, info, rec, cfg, M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = info["trace"]
    it = [t["it"] for t in tr]
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax[0, 0].plot(it, [t["E_K"] for t in tr], label="E_K"); ax[0, 0].plot(it, [t["E_stat"] for t in tr], label="E_stat"); ax[0, 0].axhline(rec["bound"]["omega_c_K_plus_E_stat_R16_1"], color="r", ls="--", lw=0.8, label="E_stat(R16-1) + omega_c K"); ax[0, 0].set_xlabel("it"); ax[0, 0].legend(fontsize=6); ax[0, 0].set_title(f"{tag}: stop {info['stop']}", fontsize=8)
    ax[0, 1].plot(it, [t["omega"] for t in tr]); ax[0, 1].axhline(OMEGA_C, color="r", ls="--", lw=0.8); ax[0, 1].set_xlabel("it"); ax[0, 1].set_title("omega = K / (2 kin) (red: omega_c 0.05)", fontsize=8)
    ax[0, 2].semilogy(it, [t["kin_tot"] for t in tr], label="kin_tot"); ax[0, 2].semilogy(it, [t["kin_h"] for t in tr], label="kin_h"); ax[0, 2].semilogy(it, [t["kin_KP"] for t in tr], label="kin_KP"); ax[0, 2].legend(fontsize=6); ax[0, 2].set_xlabel("it")
    ax[1, 0].semilogy(it, [t["esc_half_split_max"] for t in tr], label="half split max"); ax[1, 0].plot(it, [t["esc_rho2_r_rms"] / cfg["L"] for t in tr], label="rho^2 rms radius / L"); ax[1, 0].plot(it, [t["esc_rho2_fraction_r_gt_0.35L"] for t in tr], label="rho^2 fraction r > 0.35 L"); ax[1, 0].legend(fontsize=6); ax[1, 0].set_xlabel("it"); ax[1, 0].set_title("escape (a) reads", fontsize=8)
    ax[1, 1].plot(it, [t["esc_tilt_weighted"] for t in tr], label="tilt (rho^2-weighted)"); ax[1, 1].plot(it, [t["esc_spin2_mean_m_on_split_shell"] if t["esc_spin2_mean_m_on_split_shell"] is not None else np.nan for t in tr], label="<m> on the split shell"); ax[1, 1].legend(fontsize=6); ax[1, 1].set_xlabel("it"); ax[1, 1].set_title("escape (b), (c) reads", fontsize=8)
    n, h = cfg["n"], cfg["h"]
    X, Y, Z = INS4.coords(n, h)
    trip, lg, disc = F0.spatial_triple(M)
    half = np.sqrt(np.maximum(disc, 0.0)) / 2.0
    j = n // 2
    ext = [-n * h / 2, n * h / 2, -n * h / 2, n * h / 2]
    im = ax[1, 2].imshow(half[:, :, j].T, origin="lower", extent=ext, cmap="magma"); ax[1, 2].set_title("end half split, plane z", fontsize=8); ax[1, 2].set_xlim(-16, 16); ax[1, 2].set_ylim(-16, 16); fig.colorbar(im, ax=ax[1, 2], shrink=0.7)
    p = os.path.join(PLOTS, f"m5_32_{tag}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return rel(p)


def relax(n, L, comp, K, maxit, seed_split=0.0, dt0=0.01):
    tag = tag_of(comp, n, L, K)
    cfg = C.cfg_v4(n, L, completion=comp, n_samples=4)
    M0, src, how, E_stat_r16_1 = seed_for(comp, n, L, cfg)
    nref = C.radial_ref(cfg)
    free = ~INS4.pin_shell(n, cfg["h"], 1.6)
    if seed_split > 0.0:
        # a nucleated split: a smooth doublet shell (amplitude seed_split at r 5, width 2) in the local pair
        # frame, so that kin_tot is not ~1e-8 at the start (the n64 static core has half split 8e-6 and the
        # fixed-K functional K^2 / (4 kin) starts at 1e10 there: the first FIRE steps blew the field up)
        import m5_32_r16_2_operator as OP
        Ea, Eb, fr0, r0 = OP.doublet_basis(M0, cfg)
        env = seed_split * np.exp(-(r0 - 5.0) ** 2 / 8.0) * free
        M0 = M0 + env[..., None, None] * Ea
        how += f"; a nucleated doublet shell added (amplitude {seed_split} at r 5, width 2, the a-component of the local pair frame)"
    log(f"{tag}: seed {src} ({how}); E_stat(R16-1 end, 8 samples) {E_stat_r16_1}; dt0 {dt0}")
    rec = {"tag": tag, "n": n, "L": L, "h": cfg["h"], "completion": comp, "K": K, "cfg": {k: cfg[k] for k in ("mu", "cP", "cs", "n_samples", "stencil")}, "seed": {"source": src, "how": how, "E_stat_R16_1_end": E_stat_r16_1, "seed_split": seed_split, "dt0": dt0}}
    E0, g0, pp0, dom0, fr0 = C.energy_and_grad(M0, cfg, K, nref)
    rec["seed"]["parts"] = pp0
    rec["seed"]["escapes"] = escape_reads(M0, cfg, fr0)
    log(f"  seed E_K {E0:.6f} omega {pp0['omega']:.5f} kin {pp0['kin_tot']:.5f}; escapes {rec['seed']['escapes']}")
    json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
    ckp = os.path.join(CK, tag + ".npy")
    M, info = C.fire_v4(M0, cfg, free, maxit, K=K, n_ref=nref, log_every=100, tag=tag, diag=make_diag(cfg), ck_path=ckp, ck_every=100, dt0=dt0, dt_max=max(dt0 * 10, 0.1) if dt0 >= 0.01 else dt0 * 10)
    rec["descent"] = {k: info[k] for k in ("stop", "wall_s", "iters")}
    rec["trace"] = info["trace"]
    np.save(ckp, M)
    log(f"  descent {info['stop']} after {info['iters']} it, {info['wall_s']:.0f} s; end reads")
    esc = escape_reads(M, cfg, C.frame(M, info["n_ref"]))
    rec["escapes_end"] = esc
    st = stationarity(M, cfg, K, info["n_ref"])
    rec["stationarity"] = st
    cf8 = dict(cfg); cf8["n_samples"] = 8
    E8, _, pp8, dom8, _ = C.energy_and_grad(M, cf8, K, info["n_ref"], need_grad=False)
    rec["end_parts_8"] = pp8
    rec["end_domain"] = dom8
    # dE/dK = omega by a 2 percent K perturbation relaxed 200 iterations
    K2 = K * 1.02
    M2, info2 = C.fire_v4(M, cfg, free, 200, K=K2, n_ref=info["n_ref"], log_every=100, tag=tag + "_K+2%", diag=None)
    E2 = C.energy_and_grad(M2, cf8, K2, info2["n_ref"], need_grad=False)[0]
    rec["dE_dK"] = {"K": K, "K2": K2, "E_K_8": E8, "E_K2_8": E2, "dE_dK_fd": (E2 - E8) / (K2 - K), "omega_end": pp8["omega"], "ratio": ((E2 - E8) / (K2 - K)) / pp8["omega"], "stop_K2": info2["stop"]}
    base = E_stat_r16_1 if E_stat_r16_1 is not None else rec["seed"]["parts"]["E_stat"]
    rec["bound"] = {"omega_c": OMEGA_C, "omega_c_K": OMEGA_C * K, "E_K_minus_E_stat_R16_1": E8 - base, "E_below_delocalized_bound": bool(E8 - base < OMEGA_C * K), "omega_c_K_plus_E_stat_R16_1": base + OMEGA_C * K, "base_is_R16_1_end": E_stat_r16_1 is not None}
    escapes = [k for k in ("escape_a", "escape_a_box", "escape_b", "escape_c", "escape_d") if esc[k]]
    if escapes:
        v = "CANDIDATE_REFUTED (" + ", ".join(escapes) + ")"
    elif info["stop"] in ("plateau", "f_tol") and st["true_dE_max"] < 1e-3 * max(abs(E8), 1.0):
        v = "PERIODIC_ORBIT_EXISTS"
    elif info["stop"] == "max_iter":
        v = "NUMERICALLY_UNRESOLVED (max_iter, no escape reached)"
    else:
        v = f"NUMERICALLY_UNRESOLVED ({info['stop']})"
    rec["verdict"] = v
    rec["end_field"] = rel(ckp)
    rec["plot"] = plot_run(tag, info, rec, cfg, M)
    json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
    log(f"  VERDICT {v}; E_K {E8:.6f} (- E_stat(R16-1) = {E8 - base:.5f} vs omega_c K {OMEGA_C * K:.4f}); omega {pp8['omega']:.5f}; dE/dK / omega {rec['dE_dK']['ratio']:.4f}; true dE max {st['true_dE_max']:.2e} (frozen {st['frozen_dE_max']:.2e}); escapes {esc}")
    return rec


def collect():
    out = {"rung": "R16-3", "runs": {}}
    for comp in ("rebuild", "norm"):
        for n, L in ((32, 48), (48, 72), (64, 48)):
            for K in (50, 200):
                p = os.path.join(CK, tag_of(comp, n, L, K) + ".json")
                if os.path.exists(p):
                    r = json.load(open(p))
                    r.pop("trace", None)
                    out["runs"][tag_of(comp, n, L, K)] = r
    out["verdicts"] = {t: r.get("verdict") for t, r in out["runs"].items()}
    out["bounds"] = {t: r.get("bound") for t, r in out["runs"].items()}
    out["dE_dK"] = {t: r.get("dE_dK") for t, r in out["runs"].items()}
    json.dump(out, open(os.path.join(DATA, "m5_32_r16_3.json"), "w"), indent=1, default=float)
    log(f"collected {len(out['runs'])}: {out['verdicts']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["relax", "collect"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--L", type=float, default=48.0)
    ap.add_argument("--comp", default="rebuild")
    ap.add_argument("--K", type=float, default=200.0)
    ap.add_argument("--maxit", type=int, default=3000)
    ap.add_argument("--seed_split", type=float, default=0.0)
    ap.add_argument("--dt0", type=float, default=0.01)
    a = ap.parse_args(ARGS)
    if a.mode == "relax":
        relax(a.n, a.L, a.comp, a.K, a.maxit, a.seed_split, a.dt0)
    else:
        collect()
