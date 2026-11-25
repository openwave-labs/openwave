"""M5.48: the driver of the chains that read the author's three predictions after M5.45.

One chain is one L-BFGS-B descent of E_J = U + J^2/(2 I) (of U at J = 0) on the evaluator of M5.42 (m5_42_eq42.py:
half-box 6, the mirror-averaged stencil, the chart fixed, the faces held to the frame). One process, one thread.

The descent rule is that of M5.45 (m5_45_relax.py), kept line for line: 20 correction pairs, function tolerance
1e-15, the gradient stop off; after an end by the function tolerance a restart with an empty optimizer memory,
until a restart ends within 25 iterations (at most 40 restarts); a stalled line search (ABNORMAL) restarted at most
three times; U and I recorded every 100 iterations; stops saved at 1000, 3000, 10000, 30000 and 100000; a
checkpoint at every stop and every 15 minutes. What this driver adds (m5_48_task_details.md, section 1):

  parameters   delta, J, and a start from a named file on the same grid (m5_45_relax.py fixed delta = 0.05, J = 0)
  (a)          the three chains of P1 at a candidate J (the reference, the chain at J, the return) share one gate:
               min(1e-3, one tenth of max |dE_J/dm| / h^3 on the J = 0 state of M5.45 they start from)
  (b)          an end by the function tolerance repeated within 25 iterations while the gate is not met is recorded
               as STALLED and the chain is restarted from that state with an empty memory, at most three times; a
               fourth is the end of the chain, with the word STALLED

The classes of chains (section 2; `--list` prints every chain):

  p2_h01875_L0              P2: the chain of M5.45 at spacing 0.1875 without L1, continued from its end file (cut at
                            iteration 27200), the counter and the trajectory of M5.45 carried over
  p1_<state>_j<k>_ref       P1: the reference M_0', the J = 0 state of M5.45 (its end file) continued at J = 0 under
                            the gate of the candidate J = J_e/k, k = 40 ... 1280
  p1_<state>_j<k>           P1: E_J at J = J_e/k, from the end of the reference, under the same gate
  p1_<state>_j<k>_back      P1: the return, J = 0, from the end of the chain at J, under the same gate
  p3_h<spacing>             P3: delta = 0.3 without L1, from the seed
  p3_h<a>_from_h<b>         P3: the end file of chain p3_h<b> sampled onto grid a by trilinear interpolation, the faces
                            set back to the frame
  k2_*, k3_*, k4_*, k12_*   the controls of section 3 that are chains
  p1_h0250_L0_j40_back_added  one return added after the audit of the findings (no word; section 8 of the findings)

The words of a state are those of M5.45 (the gate; `converged` = the gate and an end by the function tolerance
repeated after a restart, or U and I within 1e-3 of the previous target stop; STEADY / RISING on the last tenth of
the iterations). The tables of m5_48_reads.py apply them as the tables of M5.45 did.

Every file this driver writes is named m5_48_*; it opens the files of M5.45 and M5.42 for reading only. A chain is
restartable: an interrupted chain continues from its checkpoint with an empty optimizer memory, the resume recorded.
A chain cut by its wall-time cap keeps its checkpoint, and `--resume-cut` continues it under new caps.

Usage:  python m5_48_relax.py --chain p2_h01875_L0
        python m5_48_relax.py --chain p2_h01875_L0 --resume-cut --wall-hours 10 --cap 200000
        python m5_48_relax.py --list
"""

import argparse
import hashlib
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

HALF_BOX = 6.0
TARGETS = (1000, 3000, 10000, 30000, 100000)
GATE = 1e-3
P1_K = (40, 80, 160, 320, 640, 1280)
# the ten J = 0 states of M5.45 that P1 uses (section 4): name of the M5.45 chain -> grid and L1
P1_STATES = {
    "h0375_L0": dict(N=33, L1=0.0), "h0250_L0": dict(N=49, L1=0.0),
    "h0375_L0_from_h0500": dict(N=33, L1=0.0), "h0375_L0_from_h0250": dict(N=33, L1=0.0), "h0375_L0_from_h01875": dict(N=33, L1=0.0),
    "h0250_L0_from_h0375": dict(N=49, L1=0.0), "h0250_L0_from_h01875": dict(N=49, L1=0.0),
    "h0375_L0.03": dict(N=33, L1=0.03), "h0250_L0.03": dict(N=49, L1=0.03), "h0250_L0.03_from_no_L1": dict(N=49, L1=0.03),
}
P1_LABEL_STATES = ("h0375_L0", "h0250_L0", "h0375_L0.03", "h0250_L0.03")
P3_GRIDS = {"h0500": 25, "h0375": 33, "h0250": 49, "h01875": 65}


def chains():
    """Every chain of the task: name -> dict(task, N, delta, L1, J=(kind, number), start, gate, cap, wall_hours, needs).

    The wall-time caps are sized on the pace of the pool of M5.42 (10000 iterations at 65^3 in 1.53 hours there), which was
    slower than the chains of M5.45, so that the cap of iterations is what a chain meets first."""
    C = {}
    C["p2_h01875_L0"] = dict(task="P2", N=65, delta=0.05, L1=0.0, J=("zero", 0), gate=None, cap=100000, wall_hours=4.0, needs=[],
                             start=dict(kind="file", file="m5_45_h01875_L0_end.npz",
                                        carry=dict(summary="m5_45_summary.json", chain="h01875_L0", iteration=27200)))
    for s, m in P1_STATES.items():
        big = m["N"] == 49
        for k in P1_K:
            base = dict(task="P1", N=m["N"], delta=0.05, L1=m["L1"], state=s, k=k)
            C[f"p1_{s}_j{k}_ref"] = dict(base, role="ref", J=("zero", 0), gate=dict(kind="clock", k=k), cap=5000, wall_hours=0.5, needs=[],
                                         start=dict(kind="file", file=f"m5_45_{s}_end.npz"))
            C[f"p1_{s}_j{k}"] = dict(base, role="J", J=("over", k), gate=dict(kind="from", chain=f"p1_{s}_j{k}_ref"), cap=30000, wall_hours=(2.5 if big else 1.0),
                                     needs=[f"p1_{s}_j{k}_ref"], start=dict(kind="chain", chain=f"p1_{s}_j{k}_ref"))
            C[f"p1_{s}_j{k}_back"] = dict(base, role="back", J=("zero", 0), gate=dict(kind="from", chain=f"p1_{s}_j{k}_ref"), cap=20000, wall_hours=(1.75 if big else 0.75),
                                          needs=[f"p1_{s}_j{k}"], start=dict(kind="chain", chain=f"p1_{s}_j{k}"))
    for g, N in P3_GRIDS.items():
        C[f"p3_{g}"] = dict(task="P3", N=N, delta=0.3, L1=0.0, J=("zero", 0), gate=None, cap=(10000 if N == 65 else 100000),
                            wall_hours={25: 1.0, 33: 1.5, 49: 2.5, 65: 2.5}[N], needs=[], start=dict(kind="seed"))
    for g, srcs in (("h0375", ("h0500", "h0250", "h01875")), ("h0250", ("h0375", "h01875")), ("h0500", ("h0375", "h0250", "h01875"))):
        for s in srcs:
            N = P3_GRIDS[g]
            C[f"p3_{g}_from_{s}"] = dict(task="P3", N=N, delta=0.3, L1=0.0, J=("zero", 0), gate=None, cap=100000, wall_hours=(2.0 if N == 49 else 1.0),
                                         needs=[f"p3_{s}"], start=dict(kind="interp", chain=f"p3_{s}"))
    # the controls that are chains (section 3). K3 and K4 take J as M5.42 formed it, (1/40) * J_e, so that the path returns
    C["k2_h0375_L0"] = dict(task="K2", N=33, delta=0.05, L1=0.0, J=("zero", 0), gate=None, cap=10000, wall_hours=0.5, needs=[], start=dict(kind="seed"))
    C["k3_r6_L0.3_j40"] = dict(task="K3", N=33, delta=0.05, L1=0.3, J=("times", 1 / 40), gate=dict(kind="own_start"), cap=10000, wall_hours=0.5, needs=[],
                               start=dict(kind="file", file="m5_42_r6_hb6_L0.3_j0_end.npz"))
    C["k4_r6_L0_j40"] = dict(task="K4", N=33, delta=0.05, L1=0.0, J=("times", 1 / 40), gate=dict(kind="own_start"), cap=10000, wall_hours=0.5, needs=[],
                             start=dict(kind="file", file="m5_42_r6_hb6_L0.0_j0_end.npz"))
    # K12: the chain of K3 under a gate no descent in double precision can meet, so that addition (b) must fire
    C["k12_stalled"] = dict(task="K12", N=33, delta=0.05, L1=0.3, J=("times", 1 / 40), gate=dict(kind="fixed", value=1e-12), cap=10000, wall_hours=0.5, needs=[],
                            start=dict(kind="file", file="m5_42_r6_hb6_L0.3_j0_end.npz"))
    # Added after the audit of the findings, with the word of P1 in view, and carrying no word: the return at J = 0 of the
    # state from the seed at spacing 0.25 at J_e/40, which the pool did not start because that state fails S5 there. Same gate
    # and rule as the other returns; a wall-time cap of 40 minutes. It answers one question (section 8 of the findings).
    C["p1_h0250_L0_j40_back_added"] = dict(task="P1-added", N=49, delta=0.05, L1=0.0, state="h0250_L0", k=40, role="back_added", J=("zero", 0),
                                           gate=dict(kind="from", chain="p1_h0250_L0_j40_ref"), cap=20000, wall_hours=40 / 60, needs=["p1_h0250_L0_j40"],
                                           start=dict(kind="chain", chain="p1_h0250_L0_j40"))
    return C


def gate_of(clock_force_over_h3):
    """Addition (a): the gate of the three chains of P1 at a candidate, from the clock force on the J = 0 state they start from."""
    return min(GATE, 0.1 * clock_force_over_h3)


def after_function_tolerance(made, ends, g_now, gate, stalls):
    """What follows an end by the function tolerance (the rule of M5.45 with addition (b)).

    made: the iterations of the segment that ended; ends: the ends by the function tolerance so far, this one counted;
    g_now: max |dE/dm| / h^3 on the end state; stalls: the stalls recorded before this end. Returns one of
    "restart" (the rule of M5.45: an empty memory, go on), "end" (the end repeated after a restart), "stalled, restart",
    "stalled, end"."""
    if made > 25 and ends < 40:
        return "restart"
    if made <= 25 and g_now > gate:
        return "stalled, restart" if stalls < 3 else "stalled, end"
    return "end"


def interp_start(mod, z):
    """A state of another grid (an npz with m6 and x) sampled onto the grid of mod by trilinear interpolation, the faces set
    back to the frame: the start of the carried chains, as in m5_45_relax.py."""
    from scipy.ndimage import map_coordinates
    xs = z["x"]
    idx = (mod.x - xs[0]) / (xs[1] - xs[0])
    grid = np.meshgrid(idx, idx, idx, indexing="ij")
    m6 = np.stack([map_coordinates(z["m6"][..., k], grid, order=1, mode="nearest") for k in range(6)], -1)
    return np.where(mod.mask[..., None], m6, mod.seed6())


def j_of(c):
    kind, q = c["J"]
    Je = Q.j_electron(c["delta"])
    return 0.0 if kind == "zero" else (Je / q if kind == "over" else q * Je)


def model_of(c):
    return Q.Model(c["N"], HALF_BOX, c["delta"], c["L1"], stencil="averaged", fixed_chart=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def out_file(out_dir, name):
    """The only door to a written array or record: a name that does not begin with m5_48_ is refused (control K10)."""
    if not name.startswith("m5_48_"):
        raise RuntimeError(f"refused: the driver of M5.48 writes only files named m5_48_*, not {name}")
    return Path(out_dir) / name


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
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_48" / "chains")
    ap.add_argument("--wall-hours", type=float, default=None, help="overrides the cap of the plan")
    ap.add_argument("--cap", type=int, default=None, help="overrides the cap of the plan")
    ap.add_argument("--resume-cut", action="store_true", help="continue a chain that was cut by its wall-time cap, from its checkpoint")
    a = ap.parse_args()
    C = chains()
    if a.list:
        for k, c in C.items():
            print(k, {q: c[q] for q in ("task", "N", "delta", "L1", "cap", "wall_hours")}, "J", j_of(c), "spacing", 2 * HALF_BOX / (c["N"] - 1), c["start"], "gate", c["gate"], "needs", c["needs"])
        print(len(C), "chains")
        return
    chain, c = a.chain, C[a.chain]
    cap = a.cap if a.cap is not None else c["cap"]
    wall_hours = a.wall_hours if a.wall_hours is not None else c["wall_hours"]
    J = j_of(c)
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

    jf = a.work_dir / f"m5_48_{chain}.json"
    ckpt, state_f = work / "ckpt.npz", work / "state.json"
    if jf.exists():
        old = json.loads(jf.read_text())
        if old.get("finished") and not (a.resume_cut and old.get("cut_by_wall") and ckpt.exists() and state_f.exists()):
            log("already finished")
            return
    mod = model_of(c)
    h3 = mod.h ** 3
    if ckpt.exists() and state_f.exists():
        st = json.loads(state_f.read_text())
        x = np.load(ckpt, allow_pickle=False)["m6"].reshape(-1)
        st["resumed_at"].append(st["iteration"])
        if a.resume_cut:
            st["seconds_before_the_cut"] = st.get("seconds_before_the_cut", 0.0) + st["seconds"]
            st["seconds"] = 0.0
            st.setdefault("resumed_after_a_cut_at", []).append(st["iteration"])
        log(f"resumes at it {st['iteration']} (cap {cap}, wall {wall_hours} h)")
    else:
        st = dict(iteration=0, stops=[], traj=[], ftol_ends=[], abnormal=[], stalled=[], resumed_at=[], nfev=0, seconds=0.0, start=dict(c["start"]))
        s = c["start"]
        if s["kind"] == "seed":
            x = mod.seed6().reshape(-1)
            log(f"starts from the seed: N {c['N']} spacing {mod.h:g} delta {c['delta']} L1 {c['L1']} J {J:.6f}")
        elif s["kind"] in ("file", "chain"):
            if s["kind"] == "chain":
                src = json.loads((a.work_dir / f"m5_48_{s['chain']}.json").read_text())
                fname = src["end_file"]
                st["start"].update(file=fname, source_converged=src["converged"], source_message=src["message"])
            else:
                fname = s["file"]
            fpath = a.out_dir / fname
            z = np.load(fpath, allow_pickle=False)
            if z["m6"].shape != mod.shape6 or abs(float(z["x"][1] - z["x"][0]) - mod.h) > 1e-12:
                raise RuntimeError(f"{fname} is not on the grid of {chain}")
            # the faces hold this model's frame (as the driver of M5.42 set them for a start read from a file)
            x = Q.to6(np.where(mod.mask[..., None, None], Q.from6(z["m6"]), mod.seed)).reshape(-1)
            st["start"].update(sha256=sha256_of(fpath), faces_already_on_the_frame=bool(np.array_equal(x, z["m6"].reshape(-1))))
            if s.get("carry"):  # P2: the counter, the stops and the trajectory of the chain of M5.45
                k = s["carry"]
                summ = json.loads((a.out_dir / k["summary"]).read_text())
                old = {t["chain"]: t for t in summ["states"]}[k["chain"]]
                if old["iterations"] != k["iteration"] or old["file"] != fname:
                    raise RuntimeError("the record of M5.45 does not match the start file")
                st["iteration"] = k["iteration"]
                st["traj"] = [[int(t[0]), float(t[1]), float(t[2]), float(t[1])] for t in old["trajectory"]]
                st["stops"] = [dict(q, kind=("target" if q["kind"] == "target" else "carried_end"), carried_from="M5.45") for q in old["stops"]]
                st["carried"] = dict(chain=k["chain"], iterations=old["iterations"], evaluations=old["evaluations"], trajectory_points=len(st["traj"]),
                                     U=old["U"], I=old["I"], gmax_over_h3=old["gmax_over_h3"], message=old["message"])
            log(f"starts from {fname} (sha256 {st['start']['sha256'][:8]}): N {c['N']} spacing {mod.h:g} delta {c['delta']} L1 {c['L1']} J {J:.6f}, counter at {st['iteration']}, {len(st['traj'])} trajectory points carried")
        elif s["kind"] == "interp":
            src = json.loads((a.work_dir / f"m5_48_{s['chain']}.json").read_text())
            fname = src["end_file"]
            z = np.load(a.out_dir / fname, allow_pickle=False)
            xs = z["x"]
            x = interp_start(mod, z).reshape(-1)
            st["start"].update(file=fname, source_spacing=float(xs[1] - xs[0]), source_iterations=src["iterations"], source_converged=src["converged"],
                               source_message=src["message"], source_cut_by_wall=src["cut_by_wall"])
            log(f"starts from {fname} (it {src['iterations']}, {src['message']}) sampled onto this grid")
        else:
            raise RuntimeError(f"unknown start {s}")
        p0 = mod.evaluate(x, J)
        g0 = float(np.abs(p0["grad"]).max() / h3)
        st["start"].update(E=p0["E"], U=p0["U"], I=p0["I"], gmax_over_h3=g0)
        if st.get("carried"):  # controls K1 and K10 on the start of P2: the state of the record, read back, or no chain
            k = st["carried"]
            k["read_back"] = dict(U_relative=abs(p0["U"] - k["U"]) / abs(k["U"]), I_relative=abs(p0["I"] - k["I"]) / abs(k["I"]),
                                  gmax_relative=abs(g0 - k["gmax_over_h3"]) / k["gmax_over_h3"])
            if max(k["read_back"]["U_relative"], k["read_back"]["I_relative"]) > 1e-12 or k["read_back"]["gmax_relative"] > 1e-9:
                raise RuntimeError(f"the start file does not return the record of M5.45: {k['read_back']}")
        gs = c.get("gate") or (dict(kind="plain") if J == 0.0 else dict(kind="own_start"))
        if gs["kind"] == "plain":
            st["gate"] = GATE
        elif gs["kind"] == "own_start":  # a chain at J that starts from a J = 0 state: the clock force is its own starting gradient
            st["gate"] = gate_of(g0)
        elif gs["kind"] == "clock":  # a reference: the clock force of the candidate on the J = 0 state it starts from
            gk = float(np.abs(mod.evaluate(x, Q.j_electron(c["delta"]) / gs["k"])["grad"]).max() / h3)
            st["gate"] = gate_of(gk)
            st["start"].update(clock_force_over_h3=gk, clock_J=Q.j_electron(c["delta"]) / gs["k"])
        elif gs["kind"] == "from":  # the chain at J and the return: the gate of their reference
            st["gate"] = json.loads((a.work_dir / f"m5_48_{gs['chain']}.json").read_text())["gate"]
        elif gs["kind"] == "fixed":
            st["gate"] = gs["value"]
        else:
            raise RuntimeError(f"unknown gate {gs}")
        st["gate_as"] = gs
        log(f"at the start: E {p0['E']:.12f} U {p0['U']:.12f} I {p0['I']:.9f} g {g0:.6e}; gate {st['gate']:.3e} ({gs['kind']})")
    gate_value = st["gate"]
    t0 = time.time() - st["seconds"]
    deadline = time.time() + wall_hours * 3600 - st["seconds"]
    box = dict(it=st["iteration"], last=None, last_ckpt=time.time(), cut=None, x=None)

    def fun(z):
        p = mod.evaluate(z, J)
        st["nfev"] += 1
        box["last"] = (p["U"], p["I"], p["E"])
        return p["E"], p["grad"].ravel()

    def save_stop(z, kind):
        r = R.reads(mod, z, J)
        r.update(iteration=box["it"], kind=kind, seconds=time.time() - t0, drift_of_I_over_the_last_tenth=tail_drift(st["traj"], box["it"]))
        f = out_file(a.out_dir, f"m5_48_{chain}_{'s%06d' % box['it'] if kind == 'target' else 'end'}.npz")
        np.savez(f, m6=z.reshape(mod.shape6), x=mod.x, J=J)
        r["file"] = f.name
        st["stops"].append(r)
        log(f"{kind} it {box['it']} E {r['E']:.8f} U {r['U']:.8f} I {r['I']:.4f} in r<2 {r['inertia_inside_r2']:.2f} g {r['gmax_over_h3']:.2e} share {r['largest_share']:.3g} {r['label']}")

    def checkpoint(z):
        tmp = work / "m5_48_ckpt_tmp.npz"
        np.savez(tmp, m6=z.reshape(mod.shape6))
        os.replace(tmp, ckpt)
        st["iteration"], st["seconds"] = box["it"], time.time() - t0
        tmp_s = work / "state_tmp.json"
        tmp_s.write_text(json.dumps(st))
        os.replace(tmp_s, state_f)
        box["last_ckpt"] = time.time()

    def progress():
        (work / "progress.json").write_text(json.dumps(dict(chain=chain, iteration=box["it"], U=box["last"][0], I=box["last"][1], E=box["last"][2],
                                                             I_times_h=box["last"][1] * mod.h, time=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - t0)))

    def cb(z):
        box["it"] += 1
        it = box["it"]
        if it % 100 == 0:
            st["traj"].append([it, box["last"][0], box["last"][1], box["last"][2]])
        if it in TARGETS:
            save_stop(z.copy(), "target")
            checkpoint(z)
        if it % 25 == 0:
            progress()
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
        left = cap - box["it"]
        if left <= 0:
            message = "CAP"
            break
        before = box["it"]
        res = minimize(fun, x, jac=True, method="L-BFGS-B", callback=cb, options=dict(maxiter=left, maxfun=20 * cap, ftol=1e-15, gtol=0.0, maxcor=20))
        x, message = res.x, str(res.message)
        if box["cut"]:
            message = "CUT BY THE WALL-TIME CAP"
            break
        made = box["it"] - before
        up = message.upper()
        if "REL_REDUCTION_OF_F" in up or "RELATIVE REDUCTION" in up:
            pe = mod.evaluate(x, J)  # the end state itself, evaluated once more (not counted; it does not touch the optimizer)
            g_now = float(np.abs(pe["grad"]).max() / h3)
            st["ftol_ends"].append(dict(iteration=box["it"], made=made, E=pe["E"], U=pe["U"], I=pe["I"], gmax_over_h3=g_now, fun=float(res.fun)))
            log(f"function tolerance at it {box['it']} after {made} iterations: E {pe['E']:.12f} U {pe['U']:.12f} I {pe['I']:.9f} g {g_now:.2e}")
            what = after_function_tolerance(made, len(st["ftol_ends"]), g_now, gate_value, len(st["stalled"]))
            if what.startswith("stalled"):  # addition (b): a stall, not an end
                st["stalled"].append(dict(iteration=box["it"], made=made, gmax_over_h3=g_now))
                if what == "stalled, restart":
                    log(f"STALLED at it {box['it']} (g {g_now:.2e} above the gate {gate_value:.2e}): restart {len(st['stalled'])}")
                    continue
                message = "STALLED"
                break
            if what == "end":
                message = "FUNCTION TOLERANCE, REPEATED AFTER A RESTART"
                break
            continue
        if "ABNORMAL" in up and len(st["abnormal"]) < 3 and made > 0:
            st["abnormal"].append(dict(iteration=box["it"], message=message))
            log(f"restart after ABNORMAL at it {box['it']}")
            continue
        break
    if not st["stops"] or st["stops"][-1]["iteration"] != box["it"] or st["stops"][-1].get("carried_from"):
        save_stop(x, "end")
    if box["cut"]:
        checkpoint(x)  # kept, so that --resume-cut continues the chain
    last = st["stops"][-1]
    by_ftol = message.startswith("FUNCTION TOLERANCE")
    prev = [s for s in st["stops"] if s["kind"] == "target" and s["iteration"] < last["iteration"]]
    pair = None if not prev else {k: abs(last[k] - prev[-1][k]) / abs(prev[-1][k]) for k in ("U", "I")}
    gate = last["gmax_over_h3"] <= gate_value
    drift = tail_drift(st["traj"], box["it"])
    seconds = time.time() - t0
    rec = dict(chain=chain, task=c["task"],
               model=dict(N=c["N"], half_box=HALF_BOX, spacing=mod.h, delta=c["delta"], L1=c["L1"], stencil="averaged", fixed_chart=True, J=J, J_e=mod.J_e, J_as=list(c["J"])),
               start=st["start"], carried=st.get("carried"), cap=cap, wall_hours=wall_hours, gate=gate_value, gate_as=st.get("gate_as"),
               iterations=box["it"], evaluations=st["nfev"], message=message, ended_by_function_tolerance=by_ftol, ended_at_cap=bool(box["it"] >= cap),
               cut_by_wall=bool(box["cut"]), stalled_end=bool(message == "STALLED"), gate_met=bool(gate), pair_against_previous_target=pair,
               converged=bool(gate and (by_ftol or (pair is not None and max(pair.values()) <= 1e-3))),
               drift_of_I_over_the_last_tenth=drift, word=None if drift is None else ("STEADY" if abs(drift) < 1e-3 else ("RISING" if drift > 0 else "FALLING")),
               label=last["label"], seconds=seconds, seconds_before_the_cut=st.get("seconds_before_the_cut", 0.0),
               function_tolerance_ends=st["ftol_ends"], abnormal=st["abnormal"], stalled=st["stalled"], resumed_at=st["resumed_at"],
               resumed_after_a_cut_at=st.get("resumed_after_a_cut_at", []), stops=st["stops"], trajectory=st["traj"], end_file=last["file"], finished=True)
    if st["start"].get("sha256") and st["start"].get("file") and not st["start"]["file"].startswith("m5_48_"):
        rec["start_file_sha256_at_the_end"] = sha256_of(a.out_dir / st["start"]["file"])
        rec["start_file_unchanged"] = bool(rec["start_file_sha256_at_the_end"] == st["start"]["sha256"])
    out_file(a.work_dir, jf.name).write_text(json.dumps(rec, indent=1) + "\n")
    if not box["cut"]:
        ckpt.unlink(missing_ok=True)
        state_f.unlink(missing_ok=True)
    log(f"CHAIN END it {box['it']} {message}: E {last['E']:.8f} U {last['U']:.8f} I {last['I']:.4f} converged {rec['converged']} {rec['word']} ({seconds / 60:.1f} min)")


if __name__ == "__main__":
    main()
