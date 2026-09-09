"""M5.32 R17-2 / R17-3a: the static relaxations of objects (B) and (C) (ledger 6.6), the R16-1 protocol on the
R17 instrument.  (B) = v4 with the DIRECTOR-RELATIVE plateau weight (w(N) = P23 exactly; the author's 22.4,
our reading: the director never enters the clock block).  (C) = v6 (26.3 / 27.3):
    L_v6 = -4 Ibar_1^h - [V4^dd + (mu - g_W W) rho^2 - nu rho^4 + kappa rho^6] - c_P K_P^proj - c_s bar(rho^2 E2),
    W = [(1 - lambda_1) / (1 - delta)]^2, (mu, nu, kappa) = (1e-2, 1e-2, 0.4), c_s = 0.5, the relative weight,
g_W scanned over {0.5, 1.1, 1.35, 2.0} (the author's two thresholds bracketed: 26.4 bind / condense 0.51 / 0.53
(m = 0) and 1.32 / 1.35 (m = 1) at r0 = 3; 30.1 exact 1.0966 / 1.1218), n32 L48, I_rebuild, from the R16-1 core
(the analytic seed as the control at one value).  Reads = R16-1's (texture verdict, split profile, spin-2 shell
content, exterior) plus the section-28 reads of R17-0c (Delta(r), Delta_min, r_0, mu_eff min, W max) and the
Morse index proxy (the split's growth).  Predictions on file (26.3): (1') uniaxial between the two thresholds,
a biaxial ring or split core above condensation.  Verdicts: UNIAXIAL_RADIAL / BIAXIAL_TORUS / SPLIT_CORE /
BIAXIAL_OTHER (the R16-1 rule, verbatim) per object and g_W, and the bracketed condensation threshold.

usage: python3 m5_32_r17_2_statics.py relax --object v4rel [--n 32 --L 48 --maxit 3000 --seed r15|analytic]
       python3 m5_32_r17_2_statics.py relax --object v6 --gW 1.1 [...]
       python3 m5_32_r17_2_statics.py collect
out:   checkpoints/m5_32_r17/r17_2_<tag>.npy / .json, data/m5_32_r17_2.json, plots/m5_32_r17_2_<tag>.png
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
import m5_32_r16_1_statics as S1                          # noqa: E402
import m5_32_r17_0_record as REC                          # noqa: E402

C15, INS4 = C.C15, C.INS4
RES, DATA, PLOTS = C.RES, C.DATA, C.PLOTS
CK16, CK = C.CK, R.CK
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def rel(p):
    return os.path.relpath(p, RES)


def make_cfg(obj, n, L, gW):
    if obj == "v4rel":
        cfg = C.cfg_v4(n, L, completion="rebuild", n_samples=4)
        cfg["weight"] = "relative"
        cfg["object"] = "v4"
        return cfg
    if obj == "v4abs":
        cfg = C.cfg_v4(n, L, completion="rebuild", n_samples=4)
        cfg["weight"] = "absolute"
        cfg["object"] = "v4"
        return cfg
    return R.cfg_v6(n, L, gW=gW, completion="rebuild", n_samples=4)


def tag_of(obj, n, L, gW, seed, seed_split=0.0):
    g = f"_gW{gW:g}" if obj == "v6" else ""
    s = "" if seed == "r15" else f"_{seed}"
    sp = f"_split{seed_split:g}" if seed_split > 0 else ""
    return f"r17_2_{obj}{g}_n{n}_L{int(L)}{s}{sp}"


def reads_object(M, cfg, nref):
    """the R16-1 reads under the object's energy (weight mode + U_v6) plus the section-28 core reads."""
    with R.weight_mode(cfg):
        tex, b2, trip, half = S1.texture_reads(M, cfg)
        ext = S1.exterior_read(M, cfg)
        exts = S1.spectral_exterior(M, cfg)
        fr = C.frame(M, nref)
        core, hV, gap, l1, hs = REC.core_reads(M, cfg, fr)
        core.pop("central_line", None)
        dom = C.domain(fr, cfg)
        cf8 = dict(cfg); cf8["n_samples"] = 8
        E8, _, pp8, _, _ = R.energy_object(M, cf8, None, nref, need_grad=False)
        # the symmetry-defect gate at 8 samples
        a0 = C.a0_of(M, fr)
        defects = {}
        for beta in (0.4, 1.1):
            Rb = C.rot_R(fr["J"], beta)
            Mb = Rb @ M @ np.swapaxes(Rb, -1, -2)
            pa = R.energy_object(M, cf8, 50.0, nref, need_grad=False)[2]
            pb = R.energy_object(Mb, cf8, 50.0, nref, need_grad=False)[2]
            defects[str(beta)] = {k: abs(pa[k] - pb[k]) / max(abs(pa[k]), 1e-300) for k in pa if isinstance(pa[k], float) and k in ("E_stat", "kin_tot", "E_h", "KP", "reg", "V4", "U_v6", "U")}
        worst = max(v for b in defects for v in defects[b].values())
        p16 = R.energy_object(M, dict(cf8, n_samples=16), 50.0, nref, need_grad=False)[2]
        pk8 = R.energy_object(M, cf8, 50.0, nref, need_grad=False)[2]
        dbl = max(abs(pk8[k] - p16[k]) / max(abs(p16[k]), 1e-300) for k in ("E_stat", "kin_tot", "E_h", "KP", "reg"))
    return {"texture": tex, "exterior": ext, "exterior_spectral": exts, "core": core, "domain": dom, "parts_8": pp8, "E_stat_8": E8,
            "symmetry_gate_worst_rel": worst, "symmetry_gate_pass_1e-10": bool(worst < 1e-10), "doubling_8_16_worst_rel": dbl, "doubling_gate_pass_1e-12": bool(dbl < 1e-12)}, b2, trip, half


def relax(obj, n, L, gW, maxit, seed, seed_split=0.0, seed_r=2.0):
    tag = tag_of(obj, n, L, gW, seed, seed_split)
    cfg = make_cfg(obj, n, L, gW)
    M0, src, how = S1.seed_for(n, L, cfg, "analytic" if seed == "analytic" else "r15")
    if seed == "r16_1":
        p = os.path.join(CK16, S1.tag_of("rebuild", n, L, "r15") + ".npy")
        M0, src, how = np.load(p), rel(p), "the R16-1 end field (v4, absolute weight)"
    nref = C.radial_ref(cfg)
    free = ~INS4.pin_shell(n, cfg["h"], 1.6)
    if seed_split > 0.0:
        # DEVIATION (logged 2026-09-08 14:50 UTC): the split-free R16-1 core is a STATIONARY POINT of U_v6 in the split direction
        # (d rho^2 / dM = 0 at lambda_2 = lambda_3 exactly; the core's residual split 5e-4 sits at r 4.9, outside the melted core
        # where W > 0), so the unseeded v6 statics stayed at half split 5e-4 through 200 iterations at every g_W: the saddle,
        # not the minimizer.  The v6 scan is therefore seeded with a CORE-LOCALIZED doublet split (amplitude seed_split at
        # r = seed_r, width 2, the a-component of the local pair frame); the unseeded g_W 2.0 run is kept as the control.
        import m5_32_r16_2_operator as OP
        with R.weight_mode(cfg):
            Ea, Eb, fr0, r0 = OP.doublet_basis(M0, cfg)
        env = seed_split * np.exp(-(r0 - seed_r) ** 2 / 8.0) * free
        M0 = M0 + env[..., None, None] * Ea
        how += f"; a core doublet split seeded (amplitude {seed_split} at r {seed_r}, width 2, the a-component of the local pair frame)"
    log(f"{tag}: object {obj} gW {gW} weight {cfg.get('weight')}; seed {src} ({how}); h {cfg['h']}")
    rec = {"tag": tag, "rung": "R17-2" if obj != "v6" else "R17-3a", "object": obj, "gW": gW, "seed_split": seed_split, "seed_r": seed_r, "n": n, "L": L, "h": cfg["h"], "cfg": {k: cfg[k] for k in cfg if k in ("mu", "mu_v6", "nu", "kappa", "gW", "cP", "cs", "n_samples", "stencil", "weight", "completion")}, "seed": {"source": src, "how": how}}
    t = time.time()
    rec["seed"]["reads"], _, _, _ = reads_object(M0, cfg, nref)
    log(f"  seed reads {time.time() - t:.0f} s: E_stat_8 {rec['seed']['reads']['E_stat_8']:.6f}; parts {rec['seed']['reads']['parts_8']}; texture {rec['seed']['reads']['texture']['texture_verdict']}; Delta_min {rec['seed']['reads']['core']['Delta_min_free']:.4f}")
    json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
    ckp = os.path.join(CK, tag + ".npy")
    M, info = R.fire_object(M0, cfg, free, maxit, K=None, n_ref=nref, log_every=100, tag=tag, diag=S1.make_diag(cfg), ck_path=ckp, ck_every=200)
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
    t = time.time()
    rec["reads"], b2, trip, half = reads_object(M, cfg, info["n_ref"])
    rec["end_field"] = rel(ckp)
    rec["verdict"] = rec["reads"]["texture"]["texture_verdict"]
    rec["seed_to_end"] = {"E_stat_seed_8": rec["seed"]["reads"]["E_stat_8"], "E_stat_end_8": rec["reads"]["E_stat_8"], "max_abs_change": float(np.max(np.abs(M - M0)))}
    rec["spin2_shells_fixed"] = REC.spin2_shells(M, cfg)
    rec2 = {"texture": rec["reads"]["texture"]}
    rec["plot"] = S1.plot_run(tag, b2, trip, half, cfg, info, rec2)
    json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
    rd = rec["reads"]
    log(f"  end reads {time.time() - t:.0f} s: E_stat_8 {rd['E_stat_8']:.6f}; parts {rd['parts_8']}; VERDICT {rec['verdict']} beta2 max {rd['texture']['beta2_global_max']:.3f} at r {rd['texture']['r_at_beta2_max']:.2f}; half split max {rd['texture']['half_split_max']:.4f}; "
        f"Delta_min {rd['core']['Delta_min_free']:.4f}, r_0 {rd['core']['r_0_profile (shell-mean lambda_1 = 0.8)']:.2f}, W_max {rd['parts_8'].get('W_max')}, mu_eff_min {rd['parts_8'].get('mu_eff_min')}; gates symmetry {rd['symmetry_gate_pass_1e-10']} ({rd['symmetry_gate_worst_rel']:.1e}) doubling {rd['doubling_gate_pass_1e-12']} ({rd['doubling_8_16_worst_rel']:.1e})")
    return rec


def collect():
    out = {"rung": "R17-2 / R17-3a", "runs": {}}
    for p in sorted(glob.glob(os.path.join(CK, "r17_2_*.json"))):
        r = json.load(open(p))
        r.pop("trace", None)
        out["runs"][r["tag"]] = r
    out["verdicts"] = {t: r.get("verdict") for t, r in out["runs"].items()}
    v6 = sorted([(r["gW"], r.get("seed_split", 0.0), r.get("verdict"), r.get("reads", {}).get("texture", {}).get("half_split_max"), r.get("reads", {}).get("core", {}).get("Delta_min_free")) for r in out["runs"].values() if r["object"] == "v6"])
    out["v6_scan"] = [{"gW": a, "seed_split": sp, "verdict": b, "half_split_max": c, "Delta_min": d} for a, sp, b, c, d in v6]
    # condensation by ENERGY and by the seeded split's fate (the texture label of a seeded run reads the residual seed, not the minimizer):
    # the split-free core is the saddle of U_v6 at every g_W (its energy equals the v4rel static's to 1e-5), so a seeded run CONDENSED iff its
    # end energy lies below the saddle's by more than 1e-4 or its split grew above the seed amplitude; the escape-(d) run (the split held, the
    # core left the admissible domain) is reported as UNRESOLVED_ESCAPE_D
    saddle = [r_["reads"]["E_stat_8"] for r_ in out["runs"].values() if r_["object"] != "v6" and "reads" in r_]
    saddle = saddle[0] if saddle else None
    for row in out["v6_scan"]:
        rr = [r_ for r_ in out["runs"].values() if r_["object"] == "v6" and r_["gW"] == row["gW"] and r_.get("seed_split", 0.0) == row["seed_split"]][0]
        row["E_stat_8"] = rr.get("reads", {}).get("E_stat_8")
        row["E_minus_saddle"] = (row["E_stat_8"] - saddle) if (saddle is not None and row["E_stat_8"] is not None) else None
        row["stop"] = rr.get("descent", {}).get("stop")
        if row["stop"] == "escape_d":
            row["condensed"] = "UNRESOLVED_ESCAPE_D (the seeded split held, the core left the admissible domain)"
        elif row["seed_split"] > 0:
            row["condensed"] = bool((row["E_minus_saddle"] is not None and row["E_minus_saddle"] < -1e-4) or (row["half_split_max"] is not None and row["half_split_max"] > row["seed_split"]))
        else:
            row["condensed"] = bool(row["half_split_max"] is not None and row["half_split_max"] > 1e-2)
    out["saddle_E_stat_8 (the v4rel static)"] = saddle
    notc = [row["gW"] for row in out["v6_scan"] if row["condensed"] is False]
    conc = [row["gW"] for row in out["v6_scan"] if row["condensed"] is True]
    unres = [row["gW"] for row in out["v6_scan"] if isinstance(row["condensed"], str)]
    out["condensation_threshold_bracket"] = {"largest_gW_not_condensed": max(notc) if notc else None, "smallest_gW_condensed": min(conc) if conc else None, "unresolved_escape_d": unres,
                                             "rule": "condensed iff E_seeded_end < E_saddle - 1e-4 or the split grew above the seed; the texture label of a seeded run is not the criterion"}
    json.dump(out, open(os.path.join(DATA, "m5_32_r17_2.json"), "w"), indent=1, default=float)
    log(f"collected {len(out['runs'])}: {out['verdicts']}; v6 scan {out['v6_scan']}; bracket {out['condensation_threshold_bracket']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["relax", "collect"])
    ap.add_argument("--object", default="v4rel", choices=["v4rel", "v4abs", "v6"])
    ap.add_argument("--gW", type=float, default=1.1)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--L", type=float, default=48.0)
    ap.add_argument("--maxit", type=int, default=3000)
    ap.add_argument("--seed", default="r16_1", choices=["r15", "analytic", "r16_1"])
    ap.add_argument("--seed_split", type=float, default=0.0)
    ap.add_argument("--seed_r", type=float, default=2.0)
    a = ap.parse_args(ARGS)
    if a.mode == "relax":
        relax(a.object, a.n, a.L, a.gW, a.maxit, a.seed, a.seed_split, a.seed_r)
    else:
        collect()
