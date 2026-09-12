"""M5.32 R19-1: the static Newton read under the boost-sector entrants, on
the R3 arm (ii) instrument (RELAXED boost-dressed like pairs on the pinned
lattice), at the paper's split vacuum.

EQUATIONS FIRST
---------------
Field M(x) real symmetric 4x4 per cell, eta = diag(-1, 1, 1, 1), vacuum
M_vac = diag(g, 1, delta, 0), g = 8, delta = 0.3 (the g = 32 control where
R3 has rows). Jets A_i = d_i M on the certified sym stencil. The entrants
(m5_32_r19_entrants: the definitions, the exact gradients, the gates):
    I1 (the certified action), Gam (the author's boost-sector replacement,
    F + 2 (P^tt)^T), Gam_tl (its traceless literal), Bu (the internal split
    on u), GG (the pure anticommutator, control)
    E_X[M] = 4 h^3 sum_br wt sum_cells f_X + V4        (c = 1, the literal)
Seeds (m5_32_r3_ii_pair.seed_field, unchanged): the M5.21.14 dressed
electron twice; 3x3 part the certified two-center composition ('same' =
the like pair with the +z escape tube, 'single' = one hedgehog),
isotropic-blended cores, embedded with M_00 = g; 4x4 part the per-cell
boost dressing Q_c(x) = exp(b*(r_c) n_c . K) with b*(r) the M5.21.14
record profile, the pair dressing the ordered product Q_top Q_bot. On the
undressed seed (no time row anywhere) Gam, Gam_tl and Bu equal I1
identically (the Coulomb identity, gated), so the undressed rows are run
once under I1 and serve every object.
Relaxation: the R3 FIRE with energy-monotone backtracking (dt0 0.02,
dt_max 0.2, alpha 0.1, dt halved on rejection, dt_min 1e-7), the pinned
Dirichlet shell B3.pin_shell depth 1.6 at the seed values, a fixed budget
of STEPS_ACC accepted steps (iteration cap IT_CAP). Kill rules: RUNAWAY
(grid max |M_0i| above RUNAWAY_FACTOR x seed), DIVERGED (non-finite or
below the dive floor), LOCUS-HIT (the eigenframe lost). The dressing
amplitude is free and recorded (max |M_0i| in a ball of radius BALL_R
about each core) with the R3 amp_trend vocabulary.
Reads (per object, per box):
    E_int(d) = E(pair) - 2 E(single)          (same protocol, same box)
    static part = E_int of the undressed pair (I1 rows, object-independent)
    dressing part = E_int(dressed) - E_int(undressed)
    force sign: attraction <=> E_int INCREASES with d (F = -dE/dd < 0)
    far-field fits over d: A + B/d^p for p in {1, 3, 5} (R^2 each), and
    A + B/d + C ln(d)/d (R0's law); the best exponent by R^2
Pre-registered (ledger § 6.8, revised 2026-09-11), read on the dressed
pair's E_int(d) at n = 32, L = 48, d in {12, 18, 24, 30}:
    NEWTON_SIGN_REVERSED    attractive AND best exponent 1
    ATTRACTIVE_SHORT_RANGE  attractive, best exponent 3 or 5 (the report's
                            own 1/d^5 prediction, not Newton)
    CANDIDATE_REFUTED       repulsive, or no object (a kill rule fired, or
                            the single's dressing vanished)
The n = 48, L = 72 ladder (d 12 and 24) checks the box dependence of the
sign. Controls: the certified action's dressed pair at g = 8 (the R3
calibration at this g), Gam_tl and GG at d 12 and 24, and Gam at g = 32
(d 14 and 24) against R3's lambda = 0 rows.

STAGES (python3 m5_32_r19_1_pair.py STAGE [--workers W]):
    smoke     one tiny job end to end (n = 16, 20 accepted steps)
    relax     the n = 32 batch (main + controls + the g = 32 control)
    ladder    the n = 48, L = 72 batch
    collect   tables, fits, outcomes, plots
    plots     the figures from the saved JSON (no recompute)
    extend_g32  3000 more accepted steps from the saved g 32 end fields (the convergence check)
Out: ../data/m5_32_r19_1_pair.json (partials after every job),
     ../data/m5_32_r19_1/*.npz (local), ../plots/m5_32_r19_1_*.png
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
PLOTS = os.path.join(HERE, "..", "plots")
OUT_JSON = os.path.join(DATA, "m5_32_r19_1_pair.json")
OUT_NPZ = os.path.join(DATA, "m5_32_r19_1")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


EN = _load("m5_32_r19_entrants", "m5_32_r19_entrants.py")
R3 = _load("m5_32_r3_ii_pair", "m5_32_r3_ii_pair.py")
B3 = EN.B3
LAG = EN.LAG
PAIR = R3.PAIR

G_MAIN, DELTA = 8.0, 0.3
N_MAIN, L_MAIN = 32, 48.0
N_LAD, L_LAD = 48, 72.0
DS = (12.0, 18.0, 24.0, 30.0)
DS_CTRL = (12.0, 24.0)
DS_G32 = (14.0, 24.0)
DS_R3 = (10.0, 14.0, 18.0, 24.0)      # R3's separations at g 32 (its lambda = 0 rows are the reference)
STEPS_ACC = 1500
IT_CAP = 3000
BALL_R = 5.0
RUNAWAY_FACTOR = 3.0
DIVE_FLOOR = -1e6
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def cfg_of(n, L, g):
    return B3.base_cfg(s=-1.0, g=g, n=n, L=float(L), delta=DELTA)


def params_of(g):
    return LAG.default_params(s=-1.0, g=g)


def job_tag(obj, kind, d, scale, n, g):
    return f"{obj}_{'dr' if scale else 'un'}_{kind}_d{d:g}_n{n}_g{g:g}"


def descend(M0, cfg, obj, p, steps_acc, it_cap, tag, log_every=100, dt0=0.02, dt_max=0.2):
    """R3's FIRE (m5_32_r3_ii_pair.descend) on the entrant's energy_grad."""
    free = (~B3.pin_shell(cfg["n"], cfg["h"]))[..., None, None].astype(float)
    M = M0.copy()
    E0, G, info = EN.energy_grad(M, cfg, obj, c=1.0, p=p)
    m0i_seed = float(np.max(np.abs(M0[..., 0, 1:])))
    out = {"E0": float(E0), "steps_acc_budget": steps_acc, "it_cap": it_cap,
           "pin": "B3.pin_shell depth 1.6 (Dirichlet at the seed values)",
           "fire": {"dt0": dt0, "dt_max": dt_max, "alpha0": 0.1, "dt_min": 1e-7},
           "max_abs_M0i_seed": m0i_seed, "trace": []}
    if not np.isfinite(E0) or G is None:
        out.update(stop="DIVERGED (seed energy undefined)", verdict="DIVERGED", steps_run=0, accepted=0, E_end=float("nan"))
        return M, out
    v = np.zeros_like(M)
    dt, alpha, n_up = dt0, 0.1, 0
    dt_min = 1e-7
    F = -G * free
    E_prev = E0
    stop = "budget"
    n_rej, n_rej_locus, n_acc = 0, 0, 0
    fmax = float(np.max(np.abs(F)))
    it = 0
    runaway = None
    E = E0
    while it < it_cap and n_acc < steps_acc:
        it += 1
        P = float(np.sum(F * v))
        if P > 0.0:
            n_up += 1
            vn = np.sqrt(np.sum(v * v))
            fn = np.sqrt(np.sum(F * F))
            v = (1 - alpha) * v + alpha * (F / max(fn, 1e-300)) * vn
            if n_up > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            v[:] = 0.0
            alpha, n_up = 0.1, 0
        v_try = v + dt * F
        M_try = M + dt * v_try
        E, G, info = EN.energy_grad(M_try, cfg, obj, c=1.0, p=p)
        locus_loss = not info["ok"]
        reject = locus_loss or not np.isfinite(E) or E > E_prev + 1e-12 * max(abs(E_prev), 1.0)
        if reject:
            n_rej += 1
            if locus_loss:
                n_rej_locus += 1
            dt *= 0.5
            v[:] = 0.0
            alpha, n_up = 0.1, 0
            if dt < dt_min:
                stop = "LOCUS-HIT" if locus_loss else "STALLED (dt collapsed, no descent direction accepted)"
                break
            continue
        n_acc += 1
        M, v, E_prev = M_try, v_try, E
        F = -G * free
        fmax = float(np.max(np.abs(F)))
        m0i = info["max_abs_M0i"]
        if m0i_seed > 0 and m0i > RUNAWAY_FACTOR * m0i_seed:
            runaway = {"step": it, "accepted": n_acc, "max_abs_M0i": m0i, "E": float(E)}
            stop = "RUNAWAY"
            break
        if E < DIVE_FLOOR:
            stop = "DIVERGED (dive floor)"
            break
        if it % log_every == 0 or n_acc == steps_acc:
            row = {"it": it, "acc": n_acc, "E": float(E), "fmax": fmax, "dt": dt, "min_gap": info["min_gap"], "max_abs_M0i": m0i}
            out["trace"].append(row)
            log(f"{tag} it {it:5d} acc {n_acc:5d} E {E:14.5f} fmax {fmax:.3e} dt {dt:.2e} max|M0i| {m0i:.4g} rej {n_rej}")
    if stop == "budget" and n_acc < steps_acc:
        stop = f"IT_CAP ({it_cap} iterations before {steps_acc} accepted)"
    out.update({"stop": stop, "steps_run": it, "accepted": n_acc, "rejected": n_rej, "rejected_locus": n_rej_locus,
                "dt_final": dt, "fmax_end": fmax, "E_end": float(E_prev), "E_drop": float(E0 - E_prev), "runaway": runaway,
                "max_abs_M0i_end": float(np.max(np.abs(M[..., 0, 1:])))})
    tr = out["trace"]
    q = [r for r in tr if r["acc"] >= 0.75 * n_acc] if n_acc else []
    if stop.startswith("budget") or stop.startswith("IT_CAP"):
        if len(q) >= 2:
            dE = q[-1]["E"] - q[0]["E"]
            out["last_quarter_dE"] = float(dE)
            out["last_quarter_rel"] = float(abs(dE) / max(abs(q[-1]["E"]), 1.0))
            out["verdict"] = ("PLATEAU" if out["last_quarter_rel"] <= 1e-3 else "FALLING (still descending at the budget)" if dE < 0 else "RISING")
        else:
            out["verdict"] = "budget (trace too short)"
    else:
        out["verdict"] = stop
    return M, out


def run_job(args):
    obj, kind, d, scale, n, L, g, steps = args[:8]
    resume = args[8] if len(args) > 8 else None      # a saved end field's tag: continue its descent
    t0 = time.time()
    cfg = cfg_of(n, L, g)
    p = params_of(g)
    tag = job_tag(obj, kind, d, scale, n, g) + (f"_x{steps}" if resume else "")
    row = {"obj": obj, "kind": kind, "d": d, "scale": scale, "n": n, "L": L, "g": g, "h": cfg["h"],
           "steps_acc_budget": steps, "tag": tag, "dressed": bool(scale != 0.0), "resumed_from": resume}
    try:
        if resume:
            M0 = np.load(os.path.join(OUT_NPZ, resume + ".npz"))["M"]
        else:
            M0, meta = R3.seed_field(cfg, kind, d, scale)
        row["seed_reads"] = EN.block_reads(M0, cfg, obj)
        row["seed_amp"] = R3.dressing_amplitude(M0, cfg, kind, d)
        M, des = descend(M0, cfg, obj, p, steps, 2 * steps if resume else IT_CAP, tag)
        row["descent"] = des
        finite = bool(np.all(np.isfinite(M))) and np.isfinite(des["E_end"])
        row["status"] = ("OK" if finite and des["stop"].startswith("budget") else des["stop"].split(" ")[0] if not des["stop"].startswith("budget") else "OK")
        if finite:
            row["end_reads"] = EN.block_reads(M, cfg, obj)
            row["E"] = row["end_reads"]["E_total"]
            # the same end field read under every object (the Coulomb identity on undressed rows; the block split on dressed ones)
            row["end_reads_all"] = {o: EN.block_reads(M, cfg, o) for o in ("I1", "Gam", "Gam_tl", "Bu", "GG") if o != obj}
            row["end_amp"] = R3.dressing_amplitude(M, cfg, kind, d)
            row["amp_trend"] = {lab: R3.amp_trend(row["seed_amp"].get(f"{lab}_ball_max_norm_M0i", 0.0), row["end_amp"].get(f"{lab}_ball_max_norm_M0i", 0.0))
                                for lab in ("top", "bot") if f"{lab}_ball_max_norm_M0i" in row["end_amp"]}
            row["amp_trend"]["grid"] = R3.amp_trend(row["seed_amp"]["grid_max_norm_M0i"], row["end_amp"]["grid_max_norm_M0i"])
            try:
                dq = d if d > 0 else 18.0
                row["charge"] = PAIR.charge_suite(M[..., 1:, 1:], cfg, dq)
            except Exception as e:                        # noqa: BLE001
                row["charge"] = f"charge suite failed: {e!r}"
            os.makedirs(OUT_NPZ, exist_ok=True)
            np.savez_compressed(os.path.join(OUT_NPZ, f"{tag}.npz"), M=M.astype(np.float64))
        else:
            row["E"] = None
    except Exception as e:                                # noqa: BLE001
        row["status"] = "DIVERGED"
        row["stop"] = f"exception: {e!r}"
        row["E"] = None
    row["wall_s"] = round(time.time() - t0, 1)
    log(f"DONE {tag} status {row['status']} E {row.get('E')} wall {row['wall_s']}")
    return row


def load_json():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            return json.load(f)
    return {"task": "M5.32 R19-1: the static Newton read under the boost-sector entrants (the R3 arm (ii) instrument)",
            "family": "E_X = 4 h^3 sum f_X + V4 at c = 1 for X in I1, Gam, Gam_tl, Bu, GG",
            "point": {"g": G_MAIN, "s": -1.0, "delta": DELTA}, "ds": list(DS), "rows": []}


def save_json(J):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(J, f, indent=1, default=float)
    os.replace(tmp, OUT_JSON)


def job_list(stage):
    jobs = []
    if stage == "relax":
        n, L, g = N_MAIN, L_MAIN, G_MAIN
        jobs.append(("I1", "single", 0.0, 0.0, n, L, g, STEPS_ACC))
        for d in DS:
            jobs.append(("I1", "same", d, 0.0, n, L, g, STEPS_ACC))
        for obj in ("Gam", "Bu", "I1"):
            jobs.append((obj, "single", 0.0, 1.0, n, L, g, STEPS_ACC))
            for d in DS:
                jobs.append((obj, "same", d, 1.0, n, L, g, STEPS_ACC))
        for obj in ("Gam_tl", "GG"):
            jobs.append((obj, "single", 0.0, 1.0, n, L, g, STEPS_ACC))
            for d in DS_CTRL:
                jobs.append((obj, "same", d, 1.0, n, L, g, STEPS_ACC))
        jobs.append(("Gam", "single", 0.0, 1.0, n, L, 32.0, STEPS_ACC))
        for d in DS_G32:
            jobs.append(("Gam", "same", d, 1.0, n, L, 32.0, STEPS_ACC))
    elif stage == "ladder":
        n, L, g = N_LAD, L_LAD, G_MAIN
        jobs.append(("I1", "single", 0.0, 0.0, n, L, g, STEPS_ACC))
        for d in DS_CTRL:
            jobs.append(("I1", "same", d, 0.0, n, L, g, STEPS_ACC))
        for obj in ("Gam", "Bu", "I1"):
            jobs.append((obj, "single", 0.0, 1.0, n, L, g, STEPS_ACC))
            for d in DS_CTRL:
                jobs.append((obj, "same", d, 1.0, n, L, g, STEPS_ACC))
    elif stage == "relax_g32":
        # the instrument's own vacuum (the M5.21.14 dressing is the g 32 record; at g 8 the
        # certified action itself runs away on it): (Gamma) and (Gamma, tl) at R3's separations,
        # plus this code's reproduction of R3's certified dressed rows
        n, L, g = N_MAIN, L_MAIN, 32.0
        for d in DS_R3:
            if d not in DS_G32:
                jobs.append(("Gam", "same", d, 1.0, n, L, g, STEPS_ACC))
        jobs.append(("Gam_tl", "single", 0.0, 1.0, n, L, g, STEPS_ACC))
        for d in DS_G32:
            jobs.append(("Gam_tl", "same", d, 1.0, n, L, g, STEPS_ACC))
        jobs.append(("I1", "single", 0.0, 1.0, n, L, g, STEPS_ACC))
        for d in DS_R3:
            jobs.append(("I1", "same", d, 1.0, n, L, g, STEPS_ACC))
        jobs.append(("I1", "single", 0.0, 0.0, n, L, g, STEPS_ACC))
        for d in DS_R3:
            jobs.append(("I1", "same", d, 0.0, n, L, g, STEPS_ACC))
    elif stage == "ladder_g32":
        n, L, g = N_LAD, L_LAD, 32.0
        for obj, scale in (("I1", 0.0), ("I1", 1.0), ("Gam", 1.0)):
            jobs.append((obj, "single", 0.0, scale, n, L, g, STEPS_ACC))
            for d in DS_G32:
                jobs.append((obj, "same", d, scale, n, L, g, STEPS_ACC))
    elif stage == "extend_g32":
        # the convergence check the R19-1 audit asked for: 3000 more accepted steps from the saved
        # g 32 end fields (the Gamma single and its four pairs; the certified single and d 10 as controls)
        n, L, g = N_MAIN, L_MAIN, 32.0
        for obj, ds in (("Gam", DS_R3), ("I1", (10.0,))):
            jobs.append((obj, "single", 0.0, 1.0, n, L, g, 3000, job_tag(obj, "single", 0.0, 1.0, n, g)))
            for d in ds:
                jobs.append((obj, "same", d, 1.0, n, L, g, 3000, job_tag(obj, "same", d, 1.0, n, g)))
    elif stage == "extend2_g32":
        # a second round: 6000 more accepted steps from the _x3000 fields (the geometric-tail question)
        n, L, g = N_MAIN, L_MAIN, 32.0
        for obj, ds in (("Gam", DS_R3), ("I1", (10.0,))):
            jobs.append((obj, "single", 0.0, 1.0, n, L, g, 6000, job_tag(obj, "single", 0.0, 1.0, n, g) + "_x3000"))
            for d in ds:
                jobs.append((obj, "same", d, 1.0, n, L, g, 6000, job_tag(obj, "same", d, 1.0, n, g) + "_x3000"))
    elif stage == "extend_ladder_g32":
        # the n48 (Gamma) rows and the certified rows extended by 3000 accepted steps
        n, L, g = N_LAD, L_LAD, 32.0
        for obj in ("Gam", "I1"):
            jobs.append((obj, "single", 0.0, 1.0, n, L, g, 3000, job_tag(obj, "single", 0.0, 1.0, n, g)))
            for d in DS_G32:
                jobs.append((obj, "same", d, 1.0, n, L, g, 3000, job_tag(obj, "same", d, 1.0, n, g)))
    elif stage == "smoke":
        jobs.append(("Gam", "same", 8.0, 1.0, 16, 24.0, G_MAIN, 20))
    return jobs


def stage_relax(stage, workers):
    J = load_json()
    rows = {r["tag"]: r for r in J.get("rows", [])}
    jobs = [j for j in job_list(stage) if job_tag(j[0], j[1], j[2], j[3], j[4], j[6]) + (f"_x{j[7]}" if len(j) > 8 else "") not in rows]
    # the slow objects first so the batch packs well
    order = {"Bu": 0, "Gam_tl": 1, "Gam": 2, "GG": 3, "I1": 4}
    jobs.sort(key=lambda j: (-(j[4]), order.get(j[0], 9)))
    log(f"{stage.upper()} {len(jobs)} jobs on {workers} workers (done already: {len(rows)})")
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                r = fut.result()
            except Exception as e:                        # noqa: BLE001
                r = {"obj": j[0], "kind": j[1], "d": j[2], "scale": j[3], "n": j[4], "L": j[5], "g": j[6],
                     "tag": job_tag(j[0], j[1], j[2], j[3], j[4], j[6]) + (f"_x{j[7]}" if len(j) > 8 else ""),
                     "status": "DIVERGED", "stop": f"exception: {e!r}", "E": None, "resumed_from": j[8] if len(j) > 8 else None}
            rows[r["tag"]] = r
            J = load_json()                       # merge on disk (two stages may run at once)
            disk = {x["tag"]: x for x in J.get("rows", [])}
            disk[r["tag"]] = r
            J["rows"] = list(disk.values())
            J[f"{stage}_wall_s"] = round(time.time() - t0, 1)
            save_json(J)
            log(f"{stage.upper()} [{len(rows)}] saved {r['tag']}")
    return rows


# ================= COLLECT =================
C9 = _load("m5_32_r19_c9_c12", "m5_32_r19_c9_c12.py")


def piece_reads(M, cfg):
    """the static energy split into the sectors of C9c on a field: 4 h^3 sum of
    I(F) = I1, the time-row piece I(F_t) (the R^eg sector, <= 0 on statics), the
    rest of I1, the boost-boost commutator I(X_a), the anticommutator I(X_s) and
    its traceless spatial part, and Gam - I1; plus the eigenframe gap."""
    h3 = cfg["h"] ** 3
    u0, ok, gap = C9.u_of(M)
    uf = u0.reshape(-1, 4)
    acc = {}
    for A, wt in LAG.lattice_jets(M, cfg):
        Af = A.reshape(4, -1, 4, 4)
        N = Af.shape[1]
        for s0 in range(0, N, 32768):
            ob = C9.objects(Af[:, s0:s0 + 32768], uf[s0:s0 + 32768])
            for k in ("I1", "I_Ft_timerow", "I_Xa", "I_Xs", "I_Xs_tl", "I_FminusXa", "Gam", "Gam_tl", "Bu", "GG"):
                acc[k] = acc.get(k, 0.0) + wt * float(ob[k].sum())
    out = {k: 4.0 * h3 * v for k, v in acc.items()}
    out["I1_minus_timerow"] = out["I1"] - out["I_Ft_timerow"]
    out["min_gap"] = float(gap.min()); out["max_abs_M0i"] = float(np.max(np.abs(M[..., 0, 1:])))
    return out


def piece_table():
    """seed and end pieces for every row with a saved end field."""
    J = load_json()
    out = {}
    for r in J.get("rows", []):
        f = os.path.join(OUT_NPZ, f"{r['tag']}.npz")
        if not os.path.exists(f):
            continue
        cfg = cfg_of(r["n"], r["L"], r["g"])
        M0, _ = R3.seed_field(cfg, r["kind"], r["d"], r["scale"])
        M = np.load(f)["M"]
        out[r["tag"]] = {"status": r.get("status"), "seed": piece_reads(M0, cfg), "end": piece_reads(M, cfg)}
        log(f"pieces {r['tag']}: end I1 {out[r['tag']]['end']['I1']:.1f} timerow {out[r['tag']]['end']['I_Ft_timerow']:.1f} Xs {out[r['tag']]['end']['I_Xs']:.1f}")
    J["pieces"] = out
    save_json(J)
    return out

def fit_pow(ds, es, pw):
    ds, es = np.asarray(ds, float), np.asarray(es, float)
    X = np.stack([np.ones_like(ds), ds ** (-pw)], axis=1)
    c, *_ = np.linalg.lstsq(X, es, rcond=None)
    pred = X @ c
    ss = np.sum((es - es.mean()) ** 2)
    r2 = 1.0 - np.sum((es - pred) ** 2) / ss if ss > 0 else float("nan")
    return {"A": float(c[0]), "B": float(c[1]), "R2": float(r2), "p": pw}


def outcome_of(force, fits, no_object):
    if no_object:
        return "CANDIDATE_REFUTED (no object)"
    if force["sign"] != "ATTRACTIVE":
        return "CANDIDATE_REFUTED (repulsive)" if force["sign"] == "REPULSIVE" else "UNDECIDED (flat)"
    best = max((k for k in fits if k.startswith("pow")), key=lambda k: fits[k]["R2"])
    pw = fits[best]["p"]
    return "NEWTON_SIGN_REVERSED" if pw == 1 else f"ATTRACTIVE_SHORT_RANGE (best exponent {pw})"


def collect():
    J = load_json()
    rows = {r["tag"]: r for r in J.get("rows", [])}
    groups = {}
    for r in rows.values():
        key = (r["obj"], int(r["dressed"]), r["n"], r["g"], (1 + r["resumed_from"].count("_x")) if r.get("resumed_from") else 0)
        groups.setdefault(key, {})[(r["kind"], r["d"])] = r
    res = {}
    for (obj, dr, n, g, ext), grp in sorted(groups.items()):
        key = f"{obj}_{'dr' if dr else 'un'}_n{n}_g{g:g}" + (f"_ext{ext}" if ext else "")
        single = grp.get(("single", 0.0))
        out = {"obj": obj + (f" (extended, round {ext})" if ext else ""), "dressed": bool(dr), "n": n, "g": g, "extended": ext, "rows": {}}
        ok_single = single is not None and single.get("status") == "OK"
        no_object = not ok_single
        if ok_single:
            out["E_single"] = single["E"]
            out["single_amp_trend"] = single.get("amp_trend", {})
            out["single_descent"] = {k: single["descent"].get(k) for k in ("verdict", "accepted", "steps_run", "E_drop", "max_abs_M0i_end")}
            if dr and single.get("amp_trend", {}).get("grid") == "vanished":
                no_object = True
        ds, es = [], []
        for (kind, d), r in sorted(grp.items()):
            if kind != "same":
                continue
            rr = {"status": r.get("status"), "E": r.get("E"), "verdict": r.get("descent", {}).get("verdict"),
                  "amp_trend": r.get("amp_trend"), "wall_s": r.get("wall_s")}
            if ok_single and r.get("status") == "OK":
                rr["E_int"] = r["E"] - 2.0 * single["E"]
                ds.append(d); es.append(rr["E_int"])
                er = r.get("end_reads", {})
                rr["end_reads"] = er
            else:
                no_object = True
            out["rows"][f"d{d:g}"] = rr
        if len(ds) >= 3:
            fits = {f"pow{pw}": fit_pow(ds, es, pw) for pw in (1, 3, 5)}
            fits["log"] = R3.fit_log(ds, es)
            out["fits"] = fits
            out["force"] = R3.force_read(ds, es)
            # the Newton vocabulary applies to the DRESSED pair (the ledger's pre-registration); the
            # undressed rows are the static part, whose rising E_int is the record's like-pair string
            out["outcome"] = outcome_of(out["force"], fits, no_object) if dr else f"STATIC_PART ({out['force']['sign']} in the string sense; not a Newton read)"
        else:
            out["outcome"] = "INCOMPLETE" if not no_object else "CANDIDATE_REFUTED (no object)"
        out["no_object"] = no_object
        res[key] = out
    # dressing parts against the undressed static part (I1 rows, object-independent by the Coulomb identity)
    for key, out in res.items():
        if not out["dressed"]:
            continue
        ref = res.get(f"I1_un_n{out['n']}_g{out['g']:g}")
        if ref is None:
            continue
        out["dressing_part"] = {}
        for dk, rr in out["rows"].items():
            if "E_int" in rr and dk in ref["rows"] and "E_int" in ref["rows"][dk]:
                out["dressing_part"][dk] = rr["E_int"] - ref["rows"][dk]["E_int"]
        dd = sorted(out["dressing_part"].items(), key=lambda kv: float(kv[0][1:]))
        if len(dd) >= 3:
            ds = [float(k[1:]) for k, _ in dd]; es = [v for _, v in dd]
            out["dressing_fits"] = {f"pow{pw}": fit_pow(ds, es, pw) for pw in (1, 3, 5)}
            out["dressing_force"] = R3.force_read(ds, es)
    # the Coulomb identity on the undressed end fields
    ident = {}
    for r in rows.values():
        if not r["dressed"] and r.get("status") == "OK":
            era = r.get("end_reads_all", {})
            ident[r["tag"]] = {o: abs(era[o]["E_total"] - r["E"]) / max(abs(r["E"]), 1e-300) for o in era if o in ("Gam", "Gam_tl", "Bu")}
    # R3's record at g 32 (lambda = 0: the certified action on the same instrument, its own descent code)
    r3 = {}
    try:
        with open(os.path.join(DATA, "m5_32_r3_pair.json")) as f:
            R3J = json.load(f)
        for r in R3J.get("rows", []):
            if r.get("lam") == 0.0 and r.get("n") == 32:
                r3[r["tag"]] = {"kind": r["kind"], "d": r["d"], "dressed": bool(r.get("scale")), "status": r.get("status"), "E": r.get("E"),
                                "amp_trend": r.get("amp_trend"), "E_curv_timerow": r.get("end_reads", {}).get("E_curv_timerow")}
        for dr in (0, 1):
            key = f"R3_lam0_{'dr' if dr else 'un'}_n32_g32"
            single = next((v for v in r3.values() if v["kind"] == "single" and v["dressed"] == bool(dr)), None)
            if single and single["E"] is not None:
                rows_ = {}
                for v in r3.values():
                    if v["kind"] == "same" and v["dressed"] == bool(dr) and v["E"] is not None:
                        rows_[f"d{v['d']:g}"] = {"E": v["E"], "E_int": v["E"] - 2.0 * single["E"], "status": v["status"], "amp_trend": v["amp_trend"]}
                res[key] = {"obj": "I1 (R3 record, lambda 0)", "dressed": bool(dr), "n": 32, "g": 32.0, "E_single": single["E"], "rows": rows_, "no_object": False}
                ds = sorted(float(k[1:]) for k in rows_); es = [rows_[f"d{d:g}"]["E_int"] for d in ds]
                if len(ds) >= 3:
                    res[key]["fits"] = {f"pow{pw}": fit_pow(ds, es, pw) for pw in (1, 3, 5)}; res[key]["fits"]["log"] = R3.fit_log(ds, es)
                    res[key]["force"] = R3.force_read(ds, es)
                    res[key]["outcome"] = outcome_of(res[key]["force"], res[key]["fits"], False) if dr else f"STATIC_PART ({res[key]['force']['sign']} in the string sense; not a Newton read)"
    except Exception as e:                                # noqa: BLE001
        res["R3_record_error"] = repr(e)
    J["results"] = res
    J["coulomb_identity_on_undressed_end_fields_rel"] = ident
    save_json(J)
    piece_table()
    J = load_json()
    J["results"] = res
    J["collected_utc"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    save_json(J)
    plots(res, J.get("pieces"))
    for key, out in res.items():
        print(key, out.get("outcome"), out.get("force", {}).get("sign"), {k: round(v["R2"], 3) for k, v in out.get("fits", {}).items()})
    return res


def short(out):
    o = out.get("outcome", "")
    if o.startswith("CANDIDATE_REFUTED (rep"):
        return "repulsive"
    if o.startswith("CANDIDATE_REFUTED (no"):
        return "no object"
    if o.startswith("STATIC_PART"):
        return "static part"
    if o.startswith("NEWTON"):
        return "NEWTON_SIGN_REVERSED"
    if o.startswith("ATTRACTIVE"):
        return "attractive, short range"
    return "two points" if o == "INCOMPLETE" else (o[:16] or "two points")


def plots(res, pieces=None):
    os.makedirs(PLOTS, exist_ok=True)
    for n in sorted({o["n"] for o in res.values() if isinstance(o, dict) and "n" in o}):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
        for key, out in res.items():
            if not isinstance(out, dict) or out.get("n") != n:
                continue
            ds = sorted(float(k[1:]) for k, v in out["rows"].items() if isinstance(v, dict) and "E_int" in v)
            if len(ds) < 2:
                continue
            es = [out["rows"][f"d{d:g}"]["E_int"] for d in ds]
            lab = f"{out['obj']} g{out['g']:g}: {short(out)}"
            if out["dressed"]:
                axs[0].plot(ds, es, "o-", label=lab)
                if "dressing_part" in out:
                    dd = sorted(out["dressing_part"].items(), key=lambda kv: float(kv[0][1:]))
                    axs[2].plot([float(k[1:]) for k, _ in dd], [v for _, v in dd], "s-", label=lab)
            else:
                axs[1].plot(ds, es, "o-", label=lab)
        axs[0].set_title(f"dressed like pair, n = {n}: E_int(d) = E(pair) - 2 E(single)")
        axs[0].set_yscale("symlog", linthresh=100.0)
        axs[1].set_title(f"undressed like pair (the static part), n = {n}")
        axs[2].set_title("the dressing part: E_int(dressed) - E_int(undressed)")
        axs[2].set_yscale("symlog", linthresh=100.0)
        for ax in axs:
            ax.set_xlabel("d"); ax.set_ylabel("E_int"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, f"m5_32_r19_1_eint_n{n}.png"), dpi=130)
        plt.close(fig)
    if pieces:
        # the g 8 kills: the sector pieces on the seed and the end field (symlog)
        keep = ("I1_dr", "Gam_dr", "Gam_tl_dr")
        tags = sorted(t for t in pieces if "g8" in t and t.startswith(keep) and pieces[t]["status"] != "OK") + sorted(t for t in pieces if "g8" in t and t.startswith(keep) and pieces[t]["status"] == "OK")
        if tags:
            fig, ax = plt.subplots(figsize=(13, 5))
            x = np.arange(len(tags)); w = 0.27
            for i, (k, lab) in enumerate((("I_Ft_timerow", "time-row piece I(F_t) (untouched by Gamma)"), ("I1_minus_timerow", "the rest of I1"), ("I_Xs", "boost-boost anticommutator I(X_s)"))):
                ax.bar(x + (i - 1) * w, [pieces[t]["end"][k] for t in tags], w, label=lab + " (end)")
                ax.plot(x + (i - 1) * w, [pieces[t]["seed"][k] for t in tags], "k_", ms=9)
            ax.set_yscale("symlog", linthresh=10.0)
            ax.set_ylim(-3e5, 1e5)
            ax.axhline(0.0, color="k", lw=0.6)
            ax.set_xticks(x); ax.set_xticklabels([t.replace("_n32_g8", "").replace("_same", "").replace("_single_d0", " single").replace("_dr", " dressed") for t in tags], rotation=45, ha="right", fontsize=8)
            ax.set_title("g 8, n 32: sector pieces at the kill (bars) vs the seed (black ticks); the time-row piece dives on every killed row")
            ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS, "m5_32_r19_1_pieces_g8.png"), dpi=130)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "relax", "ladder", "relax_g32", "ladder_g32", "extend_g32", "extend2_g32", "extend_ladder_g32", "collect", "plots"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.stage == "collect":
        collect()
    elif a.stage == "plots":
        J = load_json(); plots(J["results"], J.get("pieces"))
    elif a.stage == "smoke":
        r = run_job(job_list("smoke")[0])
        print(json.dumps({k: r[k] for k in ("status", "E", "wall_s", "amp_trend", "seed_reads", "end_reads")}, indent=1, default=float))
        print(r["descent"]["verdict"], r["descent"]["stop"], r["descent"]["accepted"])
    else:
        stage_relax(a.stage, a.workers)


if __name__ == "__main__":
    main()
