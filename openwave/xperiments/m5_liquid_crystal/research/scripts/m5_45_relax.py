"""M5.45: the delta = 0.05 state at J = 0 on half-box 6 against the spacing, without L1 and with a small L1.

One chain is one L-BFGS-B descent of U (J = 0) from the seed, on the evaluator of M5.42 (m5_42_eq42.py: the
mirror-averaged stencil, the chart fixed, the faces held to the frame). One process, one thread.

    chain            N    spacing   L1
    h0500_L0         25   0.5       0
    h0375_L0         33   0.375     0
    h0250_L0         49   0.25      0
    h01875_L0        65   0.1875    0
    h0500_L0.03      25   0.5       0.03
    h0375_L0.03      33   0.375     0.03
    h0250_L0.03      49   0.25      0.03

What differs from the driver of M5.42 (m5_42_relax.py), and why. There the J = 0 state at spacing 0.375 without
L1 ended at iteration 8364 on the optimizer's function tolerance while its inertia was still rising (1890 at
iteration 3000, 2096 at the end). Here:

  1. an end by the function tolerance is followed by a restart with an empty optimizer memory, until a restart
     ends within 25 iterations (at most 40 restarts);
  2. the cap is 100000 iterations, with stops saved at 1000, 3000, 10000, 30000 and 100000;
  3. U and I are recorded every 100 iterations (the trajectory), from the last evaluation of the optimizer.

The words of a state (m5_45_task_details.md, section 2):

  CONVERGED  as in M5.42: max |dU/dm| / h^3 <= 1e-3 at the last stop, and either the optimizer ended by its
             function tolerance or U and I are within 1e-3 (relative) of the previous target stop
  STEADY     the inertia changed by less than 1e-3 (relative) over the last tenth of the iterations
  RISING     the inertia grew by more than 1e-3 over the last tenth of the iterations

A chain is restartable: an interrupted chain continues from its checkpoint with an empty optimizer memory.

Usage:  python m5_45_relax.py --chain h0250_L0 [--wall-hours 2.5] [--cap 100000]
        python m5_45_relax.py --list
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_42_eq42 as Q  # noqa: E402
import m5_42_relax as R  # noqa: E402

HALF_BOX, DELTA = 6.0, 0.05
TARGETS = (1000, 3000, 10000, 30000, 100000)
CHAINS = {
    "h0500_L0": dict(N=25, L1=0.0), "h0375_L0": dict(N=33, L1=0.0), "h0250_L0": dict(N=49, L1=0.0), "h01875_L0": dict(N=65, L1=0.0),
    "h0500_L0.03": dict(N=25, L1=0.03), "h0375_L0.03": dict(N=33, L1=0.03), "h0250_L0.03": dict(N=49, L1=0.03),
}


def tail_drift(traj, it):
    """The relative change of I over the last tenth of the iterations (None before iteration 1000)."""
    if it < 1000:
        return None
    past = [t for t in traj if t[0] <= 0.9 * it]
    return None if not past else (traj[-1][2] - past[-1][2]) / past[-1][2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_45" / "chains")
    ap.add_argument("--wall-hours", type=float, default=2.5)
    ap.add_argument("--cap", type=int, default=100000)
    a = ap.parse_args()
    if a.list:
        for k, c in CHAINS.items():
            print(k, c, "spacing", 2 * HALF_BOX / (c["N"] - 1))
        return
    chain, c = a.chain, CHAINS[a.chain]
    work = a.work_dir / chain
    work.mkdir(parents=True, exist_ok=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(work / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {chain} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    guard = None
    try:
        sys.path.insert(0, os.path.expanduser("~/Library/Application Support/UPSSentinel"))
        import ups_guard
        ups_guard.register()
        guard = ups_guard
    except Exception as e:  # noqa: BLE001
        log(f"no power guard: {e}")

    jf = a.work_dir / f"{chain}.json"
    if jf.exists() and json.loads(jf.read_text()).get("finished"):
        log("already finished")
        return
    mod = Q.Model(c["N"], HALF_BOX, DELTA, c["L1"], stencil="averaged", fixed_chart=True)
    ckpt, state_f = work / "ckpt.npz", work / "state.json"
    if ckpt.exists() and state_f.exists():
        st = json.loads(state_f.read_text())
        x = np.load(ckpt)["m6"].reshape(-1)
        st["resumed_at"].append(st["iteration"])
        log(f"resumes at it {st['iteration']}")
    else:
        st = dict(iteration=0, stops=[], traj=[], ftol_ends=[], abnormal=[], resumed_at=[], nfev=0, seconds=0.0)
        x = mod.seed6().reshape(-1)
        log(f"starts from the seed: N {c['N']} spacing {mod.h:g} L1 {c['L1']}")
    t0 = time.time() - st["seconds"]
    deadline = time.time() + a.wall_hours * 3600 - st["seconds"]
    box = dict(it=st["iteration"], last=None, last_ckpt=time.time(), cut=None, x=None)

    def fun(z):
        p = mod.evaluate(z, 0.0)
        st["nfev"] += 1
        box["last"] = (p["U"], p["I"])
        return p["U"], p["grad"].ravel()

    def save_stop(z, kind):
        r = R.reads(mod, z, 0.0)
        r.update(iteration=box["it"], kind=kind, seconds=time.time() - t0, drift_of_I_over_the_last_tenth=tail_drift(st["traj"], box["it"]))
        f = a.out_dir / f"m5_45_{chain}_{'s%06d' % box['it'] if kind == 'target' else 'end'}.npz"
        np.savez(f, m6=z.reshape(mod.shape6), x=mod.x, J=0.0)
        r["file"] = f.name
        st["stops"].append(r)
        log(f"{kind} it {box['it']} U {r['U']:.6f} I {r['I']:.3f} in r<2 {r['inertia_inside_r2']:.2f} g {r['gmax_over_h3']:.2e} share {r['largest_share']:.3g} {r['label']}")

    def checkpoint(z):
        np.savez(ckpt, m6=z.reshape(mod.shape6))
        st["iteration"], st["seconds"] = box["it"], time.time() - t0
        state_f.write_text(json.dumps(st))
        box["last_ckpt"] = time.time()

    def cb(z):
        box["it"] += 1
        it = box["it"]
        if it % 100 == 0:
            st["traj"].append([it, box["last"][0], box["last"][1]])
        if it in TARGETS:
            save_stop(z.copy(), "target")
            checkpoint(z)
        if it % 25 == 0:
            (work / "progress.json").write_text(json.dumps(dict(chain=chain, iteration=it, U=box["last"][0], I=box["last"][1],
                                                                 time=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - t0)))
            if time.time() - box["last_ckpt"] > 900:
                checkpoint(z)
            if guard is not None and guard.wrap_up():
                checkpoint(z)
                log("interrupted on a power warning; checkpoint saved")
                sys.exit(75)
            if time.time() > deadline:
                box["cut"], box["x"] = "wall", z.copy()
                raise StopIteration

    message = ""
    while True:
        left = a.cap - box["it"]
        if left <= 0:
            message = "CAP"
            break
        before = box["it"]
        res = minimize(fun, x, jac=True, method="L-BFGS-B", callback=cb, options=dict(maxiter=left, maxfun=20 * a.cap, ftol=1e-15, gtol=0.0, maxcor=20))
        x, message = res.x, str(res.message)
        if box["cut"]:
            message = "CUT BY THE WALL-TIME CAP"
            break
        made = box["it"] - before
        up = message.upper()
        if "REL_REDUCTION_OF_F" in up or "RELATIVE REDUCTION" in up:
            st["ftol_ends"].append(dict(iteration=box["it"], made=made, U=float(res.fun), I=box["last"][1]))
            log(f"function tolerance at it {box['it']} after {made} iterations: U {res.fun:.8f} I {box['last'][1]:.3f}")
            if made <= 25 or len(st["ftol_ends"]) >= 40:
                message = "FUNCTION TOLERANCE, REPEATED AFTER A RESTART"
                break
            continue
        if "ABNORMAL" in up and len(st["abnormal"]) < 3 and made > 0:
            st["abnormal"].append(dict(iteration=box["it"], message=message))
            log(f"restart after ABNORMAL at it {box['it']}")
            continue
        break
    if not st["stops"] or st["stops"][-1]["iteration"] != box["it"]:
        save_stop(x, "end")
    last = st["stops"][-1]
    by_ftol = message.startswith("FUNCTION TOLERANCE")
    prev = [s for s in st["stops"] if s["kind"] == "target" and s["iteration"] < last["iteration"]]
    pair = None if not prev else {k: abs(last[k] - prev[-1][k]) / abs(prev[-1][k]) for k in ("U", "I")}
    gate = last["gmax_over_h3"] <= 1e-3
    drift = tail_drift(st["traj"], box["it"])
    rec = dict(chain=chain, model=dict(N=c["N"], half_box=HALF_BOX, spacing=mod.h, delta=DELTA, L1=c["L1"], stencil="averaged", fixed_chart=True, J=0.0),
               iterations=box["it"], evaluations=st["nfev"], message=message, ended_by_function_tolerance=by_ftol, ended_at_cap=bool(box["it"] >= a.cap),
               cut_by_wall=bool(box["cut"]), gate_met=bool(gate), pair_against_previous_target=pair,
               converged=bool(gate and (by_ftol or (pair is not None and max(pair.values()) <= 1e-3))),
               drift_of_I_over_the_last_tenth=drift, word=None if drift is None else ("STEADY" if abs(drift) < 1e-3 else ("RISING" if drift > 0 else "FALLING")),
               label=last["label"], seconds=time.time() - t0, function_tolerance_ends=st["ftol_ends"], abnormal=st["abnormal"], resumed_at=st["resumed_at"],
               stops=st["stops"], trajectory=st["traj"], end_file=last["file"], finished=True)
    jf.write_text(json.dumps(rec, indent=1) + "\n")
    ckpt.unlink(missing_ok=True)
    state_f.unlink(missing_ok=True)
    log(f"CHAIN END it {box['it']} {message}: U {last['U']:.6f} I {last['I']:.3f} converged {rec['converged']} {rec['word']} ({rec['seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
