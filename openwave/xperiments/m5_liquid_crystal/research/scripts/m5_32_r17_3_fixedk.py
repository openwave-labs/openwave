"""M5.32 R17-3c: the fixed-K descent of the author's object v6 with the TRUE gradient (ledger 6.6): E_K = E_stat(v6) +
K^2 / (4 kin) on the circle-averaged v6 action (m5_32_r17_common.energy_object: the relative weight, U_v6 added once, the
a0 chain rule), from the R17-3a static end field at the chosen g_W (the lowest g_W that binds in R17-3b, else the 1.35
field, stated), K in {50, 200}, the four escapes of R16-3, the same reads (omega = K / (2 kin), dE/dK = omega by a 2 percent
K perturbation, the inertia split) plus the v6 reads (W max, mu_eff min, U_v6), and the section-30.1 bookkeeping for the
slope (dE/dK = omega + m Omega on a branch with J = m K; m read from the split shell's <m>).  Predictions on file (26.3):
(4') the Q-ball branch, E / K < Omega_c, dE/dK = omega; the R16-0 C2 prior carried: this sextic's Coleman Q-ball on the
reduced line needs J above 2.6e4 and radius above 69, so a bound state at K 50 or 200 is the core-bound doublet, not the
Coleman Q-ball, and its absence does not refute (4') at the author's J.

usage: python3 m5_32_r17_3_fixedk.py relax --field <static_end.npy> --gW 1.35 --K 200 [--maxit 1500 --seed_split 0 --dt0 0.001]
       python3 m5_32_r17_3_fixedk.py collect
out:   checkpoints/m5_32_r17/r17_3c_v6_gW<g>_n<n>_L<L>_K<K>.npy / .json, data/m5_32_r17_3c.json, plots/m5_32_r17_3c_<tag>.png
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
import m5_32_r17_common as R                              # noqa: E402
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r16_3_fixedk as S3                           # noqa: E402
import m5_32_r17_0_record as REC                          # noqa: E402

C15, INS4 = C.C15, C.INS4
RES, DATA, PLOTS = C.RES, C.DATA, C.PLOTS
CK = R.CK
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def rel(p):
    return os.path.relpath(p, RES)


def tag_of(gW, n, L, K):
    return f"r17_3c_v6_gW{gW:g}_n{n}_L{int(L)}_K{int(K)}"


def stationarity(M, cfg, K, nref, nd=6, eps=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    free = ~INS4.pin_shell(cfg["n"], cfg["h"], 1.6)
    E0, g, pp, dom, fr = R.energy_object(M, cfg, K, nref)
    gfree = g * free[..., None, None]
    rows = []
    for _ in range(nd):
        D = C.sym(rng.normal(size=M.shape)) * free[..., None, None]
        D /= np.sqrt(np.sum(D * D))
        Ep = R.energy_object(M + eps * D, cfg, K, nref, need_grad=False)[0]
        Em = R.energy_object(M - eps * D, cfg, K, nref, need_grad=False)[0]
        rows.append({"true_fd_dE": (Ep - Em) / (2 * eps), "true_analytic_dE": float(np.sum(gfree * D))})
    return {"E_K": E0, "parts": pp, "true_grad_norm_free": float(np.sqrt(np.sum(gfree ** 2))), "true_grad_max_free": float(np.max(np.abs(gfree))), "directions": rows,
            "true_dE_max": max(abs(r_["true_fd_dE"]) for r_ in rows), "analytic_vs_fd_max_rel": max(abs(r_["true_fd_dE"] - r_["true_analytic_dE"]) / max(abs(r_["true_fd_dE"]), 1e-300) for r_ in rows),
            "inertia_split": {k: pp[k] for k in ("kin_h", "kin_KP", "kin_reg", "kin_tot")}, "omega": pp["omega"], "a0_chain_over_frozen_norm": pp["a0_chain_norm"] / max(pp["frozen_grad_norm"], 1e-300)}


def relax(field, gW, K, maxit, n, L, seed_split=0.0, dt0=0.001):
    tag = tag_of(gW, n, L, K)
    cfg = R.cfg_v6(n, L, gW=gW, completion="rebuild", n_samples=4)
    M0 = np.load(field)
    src, how = rel(field), f"the R17-3a v6 static end field at g_W {gW}"
    nref = C.radial_ref(cfg)
    free = ~INS4.pin_shell(n, cfg["h"], 1.6)
    if seed_split > 0.0:
        import m5_32_r16_2_operator as OP
        with R.weight_mode(cfg):
            Ea, Eb, fr0, r0 = OP.doublet_basis(M0, cfg)
        env = seed_split * np.exp(-(r0 - 5.0) ** 2 / 8.0) * free
        M0 = M0 + env[..., None, None] * Ea
        how += f"; a nucleated doublet shell added (amplitude {seed_split} at r 5, width 2)"
    with R.weight_mode(cfg):
        E_stat0 = R.energy_object(M0, dict(cfg, n_samples=8), None, nref, need_grad=False)[0]
    log(f"{tag}: seed {src} ({how}); E_stat(static end, 8 samples) {E_stat0:.6f}; dt0 {dt0}; TRUE gradient, object v6 g_W {gW}")
    rec = {"tag": tag, "rung": "R17-3c", "object": "v6", "gW": gW, "gradient": "true (a0 differentiated)", "n": n, "L": L, "h": cfg["h"], "K": K,
           "cfg": {k: cfg[k] for k in ("mu_v6", "nu", "kappa", "gW", "cP", "cs", "n_samples", "stencil", "weight", "completion")}, "seed": {"source": src, "how": how, "E_stat_static_end": E_stat0, "seed_split": seed_split, "dt0": dt0}}
    E0, g0, pp0, dom0, fr0 = R.energy_object(M0, cfg, K, nref)
    rec["seed"]["parts"] = pp0
    with R.weight_mode(cfg):
        rec["seed"]["escapes"] = S3.escape_reads(M0, cfg, fr0)
    log(f"  seed E_K {E0:.6f} omega {pp0['omega']:.5f} kin {pp0['kin_tot']:.5f}; chain/frozen {pp0['a0_chain_norm'] / max(pp0['frozen_grad_norm'], 1e-300):.3f}; escapes {rec['seed']['escapes']}")
    json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
    ckp = os.path.join(CK, tag + ".npy")
    with R.weight_mode(cfg):
        diag = S3.make_diag(cfg)
    def diag_w(M, fr):
        with R.weight_mode(cfg):
            return diag(M, fr)
    M, info = R.fire_object(M0, cfg, free, maxit, K=K, n_ref=nref, log_every=100, tag=tag, diag=diag_w, ck_path=ckp, ck_every=100, dt0=dt0, dt_max=dt0 * 10)
    rec["descent"] = {k: info[k] for k in ("stop", "wall_s", "iters")}
    rec["trace"] = info["trace"]
    np.save(ckp, M)
    np.save(ckp[:-4] + "_nref.npy", np.real(info["n_ref"]))            # the propagated director lift (the R17-2 audit's H9: the reads depend on it)
    rec["end_lift"] = rel(ckp[:-4] + "_nref.npy")
    log(f"  descent {info['stop']} after {info['iters']} it, {info['wall_s']:.0f} s; end reads (lift saved)")
    if info["stop"] == "non-finite":
        rec["verdict"] = "NUMERICALLY_UNRESOLVED (non-finite)"
        json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
        return rec
    with R.weight_mode(cfg):
        esc = S3.escape_reads(M, cfg, C.frame(M, info["n_ref"]))
        core, _, _, _, _ = REC.core_reads(M, cfg, C.frame(M, info["n_ref"])); core.pop("central_line", None)
    rec["escapes_end"] = esc
    rec["core_end"] = core
    st = stationarity(M, cfg, K, info["n_ref"])
    rec["stationarity"] = st
    cf8 = dict(cfg); cf8["n_samples"] = 8
    E8, _, pp8, dom8, _ = R.energy_object(M, cf8, K, info["n_ref"], need_grad=False)
    rec["end_parts_8"] = pp8
    rec["end_domain"] = dom8
    K2 = K * 1.02
    M2, info2 = R.fire_object(M, cfg, free, 200, K=K2, n_ref=info["n_ref"], log_every=100, tag=tag + "_K+2%", diag=None, dt0=dt0, dt_max=dt0 * 10)
    E2 = R.energy_object(M2, cf8, K2, info2["n_ref"], need_grad=False)[0]
    m_shell = esc.get("spin2_mean_m_on_split_shell")
    rec["dE_dK"] = {"K": K, "K2": K2, "E_K_8": E8, "E_K2_8": E2, "dE_dK_fd": (E2 - E8) / (K2 - K), "omega_end": pp8["omega"], "ratio_to_omega": ((E2 - E8) / (K2 - K)) / pp8["omega"], "stop_K2": info2["stop"],
                    "spin2_mean_m_on_split_shell": m_shell, "note_30_1": "on a branch with J = m K the slope is omega + m Omega (Omega = 2 omega on the doublet): with <m> read on the split shell, the 30.1 slope is omega (1 + 2 <m>)"}
    omega_c = float(np.sqrt(cfg["mu_v6"] / (4 * cfg["cP"])))
    rec["bound"] = {"omega_c": omega_c, "omega_c_K": omega_c * K, "E_K_minus_E_stat_static_end": E8 - E_stat0, "E_below_delocalized_bound": bool(E8 - E_stat0 < omega_c * K), "E_over_K": E8 / K}
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
    rec["bound"]["omega_c_K_plus_E_stat_R16_1"] = E_stat0 + omega_c * K
    rec["plot"] = S3.plot_run(tag, info, rec, cfg, M)
    json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
    log(f"  VERDICT {v}; E_K {E8:.6f} (- E_stat(static) = {E8 - E_stat0:.5f} vs omega_c K {omega_c * K:.4f}); omega {pp8['omega']:.5f}; dE/dK / omega {rec['dE_dK']['ratio_to_omega']:.4f} (<m> {m_shell}); "
        f"true grad norm {st['true_grad_norm_free']:.3e}, true dE max {st['true_dE_max']:.2e}; Delta_min {core['Delta_min_free']:.4f}; escapes {esc}")
    return rec


def collect():
    out = {"rung": "R17-3c", "runs": {}}
    for p in sorted(glob.glob(os.path.join(CK, "r17_3c_*.json"))):
        r_ = json.load(open(p)); r_.pop("trace", None); out["runs"][r_["tag"]] = r_
    out["verdicts"] = {t: r_.get("verdict") for t, r_ in out["runs"].items()}
    out["bounds"] = {t: r_.get("bound") for t, r_ in out["runs"].items()}
    out["dE_dK"] = {t: r_.get("dE_dK") for t, r_ in out["runs"].items()}
    json.dump(out, open(os.path.join(DATA, "m5_32_r17_3c.json"), "w"), indent=1, default=float)
    log(f"collected {len(out['runs'])}: {out['verdicts']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["relax", "collect"])
    ap.add_argument("--field"); ap.add_argument("--gW", type=float, default=1.35); ap.add_argument("--K", type=float, default=200.0); ap.add_argument("--maxit", type=int, default=1500)
    ap.add_argument("--n", type=int, default=32); ap.add_argument("--L", type=float, default=48.0); ap.add_argument("--seed_split", type=float, default=0.0); ap.add_argument("--dt0", type=float, default=0.001)
    a = ap.parse_args(ARGS)
    if a.mode == "relax":
        relax(a.field, a.gW, a.K, a.maxit, a.n, a.L, a.seed_split, a.dt0)
    else:
        collect()
