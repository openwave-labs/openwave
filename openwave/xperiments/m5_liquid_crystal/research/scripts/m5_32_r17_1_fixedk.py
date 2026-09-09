"""M5.32 R17-1: the fixed-K descents of the author's object v4 with the TRUE gradient (ledger 6.6): the
R16-3 protocol re-run with dE_K/dM carrying d a0 / d M (m5_32_r17_common.energy_and_grad_true), the
R16-3 audit's finding that the frozen-a0 gradient omitted a part up to 60 x its own size on the rotating
cores.  Same object (v4: mu 1e-2, c_P 1, c_s 0.4, I_rebuild, the absolute plateau weight), same seeds
(the R16-1 end fields; n64 the nucleated shell), same K in {50, 200}, same four escapes, same reads
(omega = K / (2 kin), dE/dK = omega by a 2 percent K perturbation, the spike radius, the inertia split),
the stationarity now read with the TRUE gradient's norm (the descent's own residual) beside the
directional derivatives of the full E_K; the R16-3 verdicts re-issued, each confirmed with the true
gradient or replaced.

EQUATIONS: m5_32_r17_common.py (section 1) and m5_32_r16_3_fixedk.py (the escape rules, verbatim).
DEVIATION (logged 2026-09-08 14:30 UTC): from the BARE R16-1 static core (half split 6e-4, kin_tot 2e-3) the
fixed-K functional is E_K = K^2 / (4 kin) = 4.9e6 at K 200 and the TRUE gradient (which carries d kin / d M
through the split) is 6000 x the frozen one: the first FIRE step at dt0 0.01 blew the field up (non-finite at
it 1).  The frozen protocol survived the same seed only because it omitted that part.  So every R17-1
descent starts from the R16-1 core plus the nucleated doublet shell of the R16-3 n64 protocol (amplitude
0.05 at r 5, width 2, the a-component of the local pair frame) with dt0 0.001, the n32 runs included: the
bare static core is a 1 / split^2 singularity of E_K under the true gradient, stated as a protocol fact.
The verdict rule (unchanged from R16-3): CANDIDATE_REFUTED (escape a / a-box / b / c / d), PERIODIC_ORBIT_EXISTS
(plateau or f_tol with no escape and the true directional derivatives below 1e-3 |E_K|), else
NUMERICALLY_UNRESOLVED; the true gradient's own residual fmax is reported with it.

usage: python3 m5_32_r17_1_fixedk.py relax --n 32 --L 48 --K 200 --maxit 3000 [--seed_split 0.05 --dt0 0.001]
       python3 m5_32_r17_1_fixedk.py collect
out:   checkpoints/m5_32_r17/r17_1_rebuild_n<n>_L<L>_K<K>.npy / .json, data/m5_32_r17_1.json, plots/m5_32_r17_1_<tag>.png
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
import m5_32_r17_common as R                              # noqa: E402
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r16_3_fixedk as S3                           # noqa: E402

C15, INS4 = C.C15, C.INS4
RES, DATA, PLOTS = C.RES, C.DATA, C.PLOTS
CK16, CK = C.CK, R.CK
T0 = time.time()
OMEGA_C = S3.OMEGA_C


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def rel(p):
    return os.path.relpath(p, RES)


def tag_of(comp, n, L, K):
    return f"r17_1_{comp}_n{n}_L{int(L)}_K{int(K)}"


def stationarity_true(M, cfg, K, nref, nd=6, eps=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    free = ~INS4.pin_shell(cfg["n"], cfg["h"], 1.6)
    E0, g, pp, dom, fr = R.energy_and_grad_true(M, cfg, K, nref)
    gfree = g * free[..., None, None]
    gs = float(np.sqrt(np.sum(gfree ** 2)))
    gf = C.energy_and_grad(M, cfg, K, nref)[1] * free[..., None, None]
    rows = []
    for _ in range(nd):
        D = C.sym(rng.normal(size=M.shape)) * free[..., None, None]
        D /= np.sqrt(np.sum(D * D))
        Ep = R.energy_and_grad_true(M + eps * D, cfg, K, nref, need_grad=False)[0]
        Em = R.energy_and_grad_true(M - eps * D, cfg, K, nref, need_grad=False)[0]
        rows.append({"true_fd_dE": (Ep - Em) / (2 * eps), "true_analytic_dE": float(np.sum(gfree * D)), "frozen_dE": float(np.sum(gf * D))})
    return {"E_K": E0, "parts": pp, "true_grad_norm_free": gs, "true_grad_max_free": float(np.max(np.abs(gfree))), "frozen_grad_norm_free": float(np.sqrt(np.sum(gf ** 2))),
            "directions": rows, "true_dE_max": max(abs(r["true_fd_dE"]) for r in rows), "analytic_vs_fd_max_rel": max(abs(r["true_fd_dE"] - r["true_analytic_dE"]) / max(abs(r["true_fd_dE"]), 1e-300) for r in rows),
            "inertia_split": {k: pp[k] for k in ("kin_h", "kin_KP", "kin_reg", "kin_tot")}, "omega": pp["omega"], "a0_chain_over_frozen_norm": pp["a0_chain_norm"] / max(pp["frozen_grad_norm"], 1e-300)}


def relax(n, L, comp, K, maxit, seed_split=0.0, dt0=0.01):
    tag = tag_of(comp, n, L, K)
    cfg = C.cfg_v4(n, L, completion=comp, n_samples=4)
    M0, src, how, E_stat_r16_1 = S3.seed_for(comp, n, L, cfg)
    nref = C.radial_ref(cfg)
    free = ~INS4.pin_shell(n, cfg["h"], 1.6)
    if seed_split > 0.0:
        import m5_32_r16_2_operator as OP
        Ea, Eb, fr0, r0 = OP.doublet_basis(M0, cfg)
        env = seed_split * np.exp(-(r0 - 5.0) ** 2 / 8.0) * free
        M0 = M0 + env[..., None, None] * Ea
        how += f"; a nucleated doublet shell added (amplitude {seed_split} at r 5, width 2, the a-component of the local pair frame)"
    log(f"{tag}: seed {src} ({how}); E_stat(R16-1 end, 8 samples) {E_stat_r16_1}; dt0 {dt0}; TRUE gradient")
    rec = {"tag": tag, "rung": "R17-1", "gradient": "true (a0 differentiated: m5_32_r17_common.energy_and_grad_true)", "n": n, "L": L, "h": cfg["h"], "completion": comp, "K": K,
           "cfg": {k: cfg[k] for k in ("mu", "cP", "cs", "n_samples", "stencil")}, "seed": {"source": src, "how": how, "E_stat_R16_1_end": E_stat_r16_1, "seed_split": seed_split, "dt0": dt0}}
    E0, g0, pp0, dom0, fr0 = R.energy_and_grad_true(M0, cfg, K, nref)
    rec["seed"]["parts"] = pp0
    rec["seed"]["escapes"] = S3.escape_reads(M0, cfg, fr0)
    log(f"  seed E_K {E0:.6f} omega {pp0['omega']:.5f} kin {pp0['kin_tot']:.5f}; chain/frozen {pp0['a0_chain_norm'] / max(pp0['frozen_grad_norm'], 1e-300):.3f}; escapes {rec['seed']['escapes']}")
    json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
    ckp = os.path.join(CK, tag + ".npy")
    M, info = R.fire_v4(M0, cfg, free, maxit, K=K, n_ref=nref, log_every=100, tag=tag, diag=S3.make_diag(cfg), ck_path=ckp, ck_every=100, dt0=dt0, dt_max=max(dt0 * 10, 0.1) if dt0 >= 0.01 else dt0 * 10, true_gradient=True)
    rec["descent"] = {k: info[k] for k in ("stop", "wall_s", "iters")}
    rec["trace"] = info["trace"]
    np.save(ckp, M)
    np.save(ckp[:-4] + "_nref.npy", np.real(info["n_ref"]))            # the propagated director lift (the R17-2 audit's H9: the reads depend on it)
    rec["end_lift"] = rel(ckp[:-4] + "_nref.npy")
    log(f"  descent {info['stop']} after {info['iters']} it, {info['wall_s']:.0f} s; end reads (lift saved)")
    if info["stop"] == "non-finite":
        rec["verdict"] = "NUMERICALLY_UNRESOLVED (non-finite: the descent blew up)"
        json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
        log(f"  VERDICT {rec['verdict']}")
        return rec
    esc = S3.escape_reads(M, cfg, C.frame(M, info["n_ref"]))
    rec["escapes_end"] = esc
    st = stationarity_true(M, cfg, K, info["n_ref"])
    rec["stationarity"] = st
    cf8 = dict(cfg); cf8["n_samples"] = 8
    E8, _, pp8, dom8, _ = R.energy_and_grad_true(M, cf8, K, info["n_ref"], need_grad=False)
    rec["end_parts_8"] = pp8
    rec["end_domain"] = dom8
    K2 = K * 1.02
    M2, info2 = R.fire_v4(M, cfg, free, 200, K=K2, n_ref=info["n_ref"], log_every=100, tag=tag + "_K+2%", diag=None, dt0=dt0, true_gradient=True)
    E2 = R.energy_and_grad_true(M2, cf8, K2, info2["n_ref"], need_grad=False)[0]
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
    r16 = json.load(open(os.path.join(DATA, "m5_32_r16_3.json"))) if os.path.exists(os.path.join(DATA, "m5_32_r16_3.json")) else {"verdicts": {}}
    rec["r16_3_verdict_same_cell"] = r16["verdicts"].get(S3.tag_of(comp, n, L, K))
    rec["end_field"] = rel(ckp)
    rec["plot"] = S3.plot_run(tag, info, rec, cfg, M)
    json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
    log(f"  VERDICT {v} (R16-3 frozen: {rec['r16_3_verdict_same_cell']}); E_K {E8:.6f} (- E_stat(R16-1) = {E8 - base:.5f} vs omega_c K {OMEGA_C * K:.4f}); omega {pp8['omega']:.5f}; dE/dK / omega {rec['dE_dK']['ratio']:.4f}; "
        f"true grad norm {st['true_grad_norm_free']:.3e} (max {st['true_grad_max_free']:.2e}), true dE max {st['true_dE_max']:.2e}, analytic vs fd {st['analytic_vs_fd_max_rel']:.1e}; escapes {esc}")
    return rec


def collect():
    out = {"rung": "R17-1", "runs": {}}
    for comp in ("rebuild",):
        for n, L in ((32, 48), (48, 72), (64, 48)):
            for K in (50, 200):
                p = os.path.join(CK, tag_of(comp, n, L, K) + ".json")
                if os.path.exists(p):
                    r = json.load(open(p))
                    r.pop("trace", None)
                    out["runs"][tag_of(comp, n, L, K)] = r
    out["verdicts"] = {t: r.get("verdict") for t, r in out["runs"].items()}
    out["r16_3_verdicts"] = {t: r.get("r16_3_verdict_same_cell") for t, r in out["runs"].items()}
    out["bounds"] = {t: r.get("bound") for t, r in out["runs"].items()}
    out["dE_dK"] = {t: r.get("dE_dK") for t, r in out["runs"].items()}
    json.dump(out, open(os.path.join(DATA, "m5_32_r17_1.json"), "w"), indent=1, default=float)
    log(f"collected {len(out['runs'])}: {out['verdicts']} (R16-3: {out['r16_3_verdicts']})")
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
