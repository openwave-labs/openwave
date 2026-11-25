"""M5.46: one chain of the pull: the relaxed unit dilated about its loop, then released.

The action is that of M5.42 (m5_42_eq42.py) at J = 0, on the mirror-averaged stencil with the chart fixed:

    U = h^3 sum_n [ L1 sum_i |d_i M|^2 + kappa sum_{i<j} |[d_i M, d_j M]|^2 + v sum_k (tr M^k - c_k)^2 ]

with kappa = v = 1, delta = 0.05, L1 = 0.03, c_k = 1 + delta^k.

    THE BASE     the descent of U from the seed with the outer shell held at the seed (chains B0a to B0d), by
                 the rule of m5_45_relax.py: L-BFGS-B, 20 correction pairs, a restart with an empty memory after
                 every end by the function tolerance until a restart ends within 25 iterations.
    THE PULL     M'(x) = M(c + (x - c) / s),  s = d / (2 R),  c and R the center and radius of the base loop
                 (the reader of M5.44), d the separation asked for (10 or 13.8); the six components sampled by
                 trilinear interpolation.
    THE RELEASE  free faces: every node a variable (F, P and G chains). H10a: the outer shell held at the
                 pulled field's own values. L-BFGS-B as above, except G10a:
                     m <- m - a grad U / h^3          (a fixed; chosen by the pilot described in the record)
    THE STOPS    of a free chain: EXIT (no set of 8 pierced plaquettes or more clear of the faces, and the box
                 surface liftable with |q| = 0, on three reads in a row), GATE (max |dU/dm| / h^3 <= 1e-3),
                 CAP (20000 iterations), WALL (one hour). The base chains and H10a run by the rule of M5.45
                 (function tolerance repeated after a restart), under the same cap and wall time; the
                 iteration at which the gate is first met is recorded.
    THE READS    m5_46_degree.read_state at iteration 0, every 10 iterations up to 300, every 25 after, and at
                 the end: the sets of the line with their place against the faces, and |q| on every nested
                 cube surface.
    ADDED        chains X0a, X0b, X14c, X14d (outside the plan, no label): the end state of F0a, F0b, P14c, P14d
                 continued with free faces and the gate stop off.
    THE CURVE    U(s) of the dilated base, s from 1 to the largest pull, before any release (--dilation).

One process, one thread per chain. A chain is restartable from its checkpoint (empty optimizer memory).

Usage:  python m5_46_pull.py --list
        python m5_46_pull.py --chain P10a
        python m5_46_pull.py --pool base --cores 4        (then)   python m5_46_pull.py --dilation
        python m5_46_pull.py --pool release --cores 9
        python m5_46_pull.py --pool added --cores 4
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_42_eq42 as Q  # noqa: E402
import m5_46_degree as D  # noqa: E402

DELTA, L1 = 0.05, 0.03
GATE, CAP, WALL_HOURS = 1e-3, 20000, 1.0
SNAPS = (50, 100, 200, 300, 500, 1000, 2000, 5000, 10000)
CHAINS = {
    "B0a": dict(kind="base", hb=8.0, N=33, faces="held"),
    "B0b": dict(kind="base", hb=8.0, N=65, faces="held"),
    "B0c": dict(kind="base", hb=10.0, N=41, faces="held"),
    "B0d": dict(kind="base", hb=10.0, N=81, faces="held"),
    "F0a": dict(kind="release", hb=8.0, N=33, base="B0a", pull=None, faces="free", descent="lbfgs"),
    "F0b": dict(kind="release", hb=8.0, N=65, base="B0b", pull=None, faces="free", descent="lbfgs"),
    "P10a": dict(kind="release", hb=8.0, N=33, base="B0a", pull=10.0, faces="free", descent="lbfgs"),
    "P10b": dict(kind="release", hb=8.0, N=65, base="B0b", pull=10.0, faces="free", descent="lbfgs"),
    "P14c": dict(kind="release", hb=10.0, N=41, base="B0c", pull=13.8, faces="free", descent="lbfgs"),
    "P14d": dict(kind="release", hb=10.0, N=81, base="B0d", pull=13.8, faces="free", descent="lbfgs"),
    "P14a": dict(kind="release", hb=8.0, N=33, base="B0a", pull=13.8, faces="free", descent="lbfgs"),
    "H10a": dict(kind="release", hb=8.0, N=33, base="B0a", pull=10.0, faces="held", descent="lbfgs"),
    "G10a": dict(kind="release", hb=8.0, N=33, base="B0a", pull=10.0, faces="free", descent="steepest"),
    # Added during the run (m5_46_task_details.md, deviations): a free chain that ended at the gate, continued from its
    # end state with the gate stop off (EXIT, the function tolerance repeated after a restart, CAP or WALL end it).
    "X0a": dict(kind="added", hb=8.0, N=33, base="F0a", pull=None, faces="free", descent="lbfgs", gate_stop=False),
    "X0b": dict(kind="added", hb=8.0, N=65, base="F0b", pull=None, faces="free", descent="lbfgs", gate_stop=False),
    "X14c": dict(kind="added", hb=10.0, N=41, base="P14c", pull=None, faces="free", descent="lbfgs", gate_stop=False),
    "X14d": dict(kind="added", hb=10.0, N=81, base="P14d", pull=None, faces="free", descent="lbfgs", gate_stop=False),
}
PILOT_STEPS, PILOT_A = 60, (0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625)


def due(it):
    return it % 10 == 0 if it <= 300 else it % 25 == 0


def dilate(m6, x, center, s):
    """M'(x) = M(c + (x - c) / s) on the same grid, by trilinear interpolation of the six components."""
    h = x[1] - x[0]
    idx = [(center[a] + (x - center[a]) / s - x[0]) / h for a in range(3)]
    g = np.meshgrid(*idx, indexing="ij")
    return np.stack([map_coordinates(m6[..., k], g, order=1, mode="nearest") for k in range(6)], -1)


def base_loop(m6, x):
    line = D.line_read(m6, x)[0]
    if len(line["sets"]) != 1:
        raise SystemExit(f"the base state has {len(line['sets'])} sets of 8 plaquettes or more; one is needed")
    return line["sets"][0]


def compact(it, U, gmax, r, seconds):
    """One read as a small row: the sets, and one character per nested surface."""
    code = dict(pierced="x", gap="g", lift="l")
    q = "".join(str(min(s["abs_q"], 9)) if s["liftable"] else code[s["reason"]] for s in r["surfaces"])
    sets = [[s["count"], round(s["radius"], 4), *[round(v, 3) for v in s["center"]], *[round(v, 3) for v in s["normal"]],
             round(s["least_distance_to_faces"], 3), sum(1 << f for f in s["faces"]), round(s["max_norm"], 3), round(s["spread"], 3),
             round(s["off_plane"], 3)] for s in r["line"]["sets"][:8]]
    return dict(it=it, U=float(f"{U:.9g}"), g=float(f"{gmax:.4g}"), t=round(seconds, 1), loop=r["loop"], n=len(r["line"]["sets"]),
                clear=len(r["clear"]), sets=sets, frag=[r["line"]["fragments"], r["line"]["fragment_plaquettes"]], q=q,
                pt=[s["pierced_triangles"] for s in r["surfaces"]], ar="".join("1" if s["around_the_loop"] else "0" for s in r["surfaces"]),
                en="".join(str(min(len(s["surrounds"]), 9)) for s in r["surfaces"]),
                mg=round(min(s["min_gap"] for s in r["surfaces"]), 3), dev=float(f"{max((s.get('q_off_whole') or 0.0) for s in r['surfaces']):.2g}"))


def run_chain(name, a):
    c = CHAINS[name]
    work = a.work_dir / "chains" / name
    work.mkdir(parents=True, exist_ok=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(work / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {name} {msg}"
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

    jf = a.out_dir / f"m5_46_chain_{name}.json"
    if jf.exists() and json.loads(jf.read_text()).get("finished"):
        log("already finished")
        return
    free, base, steepest = c["faces"] == "free", c["kind"] == "base", c.get("descent") == "steepest"
    exit_rule, gate_stops = free and not base, free and not base and c.get("gate_stop", True)
    mod = Q.Model(c["N"], c["hb"], DELTA, L1, stencil="averaged", fixed_chart=True, free_faces=free)
    surfaces = D.nested(mod.x)
    h3 = mod.h ** 3
    ckpt, state_f = work / "ckpt.npz", work / "state.json"
    if ckpt.exists() and state_f.exists():
        st = json.loads(state_f.read_text())
        x = np.load(ckpt)["m6"].reshape(-1)
        st["resumed_at"].append(st["iteration"])
        log(f"resumes at it {st['iteration']}")
    else:
        st = dict(iteration=0, reads=[], events=[], resumed_at=[], nfev=0, seconds=0.0, exit_run=0, gate_first=None, snaps=[], step=None, pilot=None)
        if base:
            x = mod.seed6().reshape(-1).copy()
            st["start"] = dict(kind="seed")
            log(f"starts from the seed: N {c['N']} spacing {mod.h:g} half-box {c['hb']:g}")
        else:
            z = np.load(a.out_dir / f"m5_46_{c['base']}_end.npz", allow_pickle=False)
            assert len(z["x"]) == c["N"] and abs(z["x"][-1] - c["hb"]) < 1e-12
            lp = base_loop(z["m6"], z["x"])
            s = 1.0 if c["pull"] is None else c["pull"] / (2.0 * lp["radius"])
            m6 = z["m6"].copy() if c["pull"] is None else dilate(z["m6"], z["x"], lp["center"], s)
            pb = mod.evaluate(z["m6"].reshape(-1), 0.0, need_grad=False, need_I=False)
            x = m6.reshape(-1).copy()
            np.savez(a.out_dir / f"m5_46_{name}_start.npz", m6=m6, x=mod.x)
            st["start"] = dict(kind=("continued" if c["kind"] == "added" else "base") if c["pull"] is None else "pulled", base=c["base"], base_radius=lp["radius"], base_center=lp["center"],
                               base_normal=lp["normal"], separation=c["pull"], s=s, U_base=pb["U"], file=f"m5_46_{name}_start.npz")
            log(f"starts from {c['base']} (loop radius {lp['radius']:.4f}), separation {c['pull']}, s {s:.4f}, faces {c['faces']}")
    t0 = time.time() - st["seconds"]
    deadline = time.time() + a.wall_hours * 3600 - st["seconds"]
    box = dict(it=st["iteration"], last=(np.inf, np.inf), last_ckpt=time.time())
    stop = dict(reason=None, x=None)

    def evaluate(z, grad=True):
        p = mod.evaluate(z, 0.0, need_grad=grad, need_I=False)
        if grad:
            p["gmax"] = float(np.abs(p["grad"]).max() / h3)
        return p

    def take_read(z, it):
        p = evaluate(z)
        r = D.read_state(z.reshape(mod.shape6), mod.x, surfaces)
        row = compact(it, p["U"], p["gmax"], r, time.time() - t0)
        row["Ut"] = [float(f"{p[k]:.9g}") for k in ("U2", "U4", "UV")]
        st["reads"].append(row)
        return row, r

    def checkpoint(z):
        np.savez(ckpt, m6=z.reshape(mod.shape6))
        st["iteration"], st["seconds"] = box["it"], time.time() - t0
        state_f.write_text(json.dumps(st))
        box["last_ckpt"] = time.time()

    def gate_seen(it):
        if st["gate_first"] is None:
            st["gate_first"] = it
            log(f"gate first met at it {it}")

    def after(z):
        """Once per iteration, with the new iterate. Returns True when the chain stops."""
        box["it"] += 1
        it = box["it"]
        if due(it):
            row, r = take_read(z, it)
            if exit_rule:
                bs = r["surfaces"][-1]
                st["exit_run"] = st["exit_run"] + 1 if (not r["clear"] and bs["liftable"] and bs["abs_q"] == 0) else 0
                if st["exit_run"] >= 3:
                    stop.update(reason="EXIT", x=z.copy())
            if row["g"] <= GATE:
                gate_seen(it)
                if gate_stops and not stop["reason"]:
                    stop.update(reason="GATE", x=z.copy())
            moved = False
            if c["kind"] == "added" and r["loop"] is not None:      # an added chain also saves the loop on its way: every unit of travel
                cen = np.array(r["line"]["sets"][r["loop"]]["center"])
                ref = np.array(st.setdefault("snap_center", st["start"]["base_center"]))
                moved = bool(np.linalg.norm(cen - ref) >= 1.0)
                if moved:
                    st["snap_center"] = cen.tolist()
            if it in SNAPS or moved:
                f = a.out_dir / f"m5_46_{name}_s{it:05d}.npz"
                np.savez(f, m6=z.reshape(mod.shape6).astype(np.float32), x=mod.x)
                st["snaps"].append(f.name)
            (work / "progress.json").write_text(json.dumps(dict(chain=name, iteration=it, U=row["U"], g=row["g"], sets=row["n"], clear=row["clear"],
                                                                 q=row["q"], radius=(row["sets"][row["loop"]][1] if row["loop"] is not None else None),
                                                                 time=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - t0)))
        elif box["last"][1] <= GATE and (gate_stops or st["gate_first"] is None):
            if evaluate(z)["gmax"] <= GATE:
                gate_seen(it)
                if gate_stops:
                    stop.update(reason="GATE", x=z.copy())
        if it % 25 == 0 and not stop["reason"]:
            if time.time() - box["last_ckpt"] > 900:
                checkpoint(z)
            if guard is not None and guard.wrap_up():
                checkpoint(z)
                log("interrupted on a power warning; checkpoint saved")
                sys.exit(75)
            if time.time() > deadline:
                stop.update(reason="WALL", x=z.copy())
        return bool(stop["reason"])

    if not st["reads"]:
        row, r = take_read(x, 0)
        st["start"].update(U=row["U"], U2=row["Ut"][0], U4=row["Ut"][1], UV=row["Ut"][2], gmax_over_h3=row["g"],
                           radius=(row["sets"][row["loop"]][1] if row["loop"] is not None else None))
        log(f"read 0: U {row['U']:.6f} (UV {row['Ut'][2]:.6f}) g {row['g']:.3g} sets {row['n']} clear {row['clear']} q {row['q']}")

    message = ""
    if steepest:
        if st["step"] is None:
            pilot = []
            for al in PILOT_A:
                y, ups, prev = x.copy(), 0, None
                for _ in range(PILOT_STEPS):
                    p = evaluate(y)
                    ups += int(prev is not None and p["U"] > prev)
                    prev = p["U"]
                    y -= al * p["grad"].ravel() / h3
                pilot.append(dict(step=al, steps=PILOT_STEPS, rises_of_U=ups, U_last=prev))
                if ups == 0 and np.isfinite(prev):
                    break
            st["pilot"], st["step"] = pilot, pilot[-1]["step"] / 2.0
            log(f"pilot: the largest step without a rise of U over {PILOT_STEPS} steps is {pilot[-1]['step']:g}; the chain uses {st['step']:g}")
        rises, prev = st.get("rises", 0), None
        while box["it"] < a.cap:
            p = evaluate(x)
            st["nfev"] += 1
            rises += int(prev is not None and p["U"] > prev)
            prev = p["U"]
            box["last"] = (p["U"], p["gmax"])
            x = x - st["step"] * p["grad"].ravel() / h3
            st["rises"] = rises
            if after(x):
                break
        message = stop["reason"] or "CAP"
    else:
        def fun(z):
            p = evaluate(z)
            st["nfev"] += 1
            box["last"] = (p["U"], p["gmax"])
            return p["U"], p["grad"].ravel()

        def cb(z):
            if after(z):
                raise StopIteration

        ftol_ends = 0
        while True:
            left = a.cap - box["it"]
            if left <= 0:
                message = "CAP"
                break
            before = box["it"]
            res = minimize(fun, x, jac=True, method="L-BFGS-B", callback=cb, options=dict(maxiter=left, maxfun=20 * a.cap, ftol=1e-15, gtol=0.0, maxcor=20))
            if stop["reason"]:
                x, message = stop["x"], stop["reason"]
                break
            x, msg = res.x, str(res.message)
            made, up = box["it"] - before, msg.upper()
            if "REL_REDUCTION_OF_F" in up or "RELATIVE REDUCTION" in up:
                ftol_ends += 1
                st["events"].append(dict(iteration=box["it"], event="function tolerance", made=made, U=float(res.fun)))
                log(f"function tolerance at it {box['it']} after {made} iterations: U {res.fun:.8f}")
                if made <= 25 or ftol_ends >= 40:
                    message = "FUNCTION TOLERANCE, REPEATED AFTER A RESTART"
                    break
                continue
            if "ABNORMAL" in up and sum(e["event"] == "stalled line search" for e in st["events"]) < 3 and made > 0:
                st["events"].append(dict(iteration=box["it"], event="stalled line search", made=made))
                log(f"restart after a stalled line search at it {box['it']}")
                continue
            if "ABNORMAL" in up:
                st["events"].append(dict(iteration=box["it"], event="stalled line search", made=made))
                msg = "FUNCTION TOLERANCE, THEN A STALLED LINE SEARCH" if ftol_ends else "STALLED LINE SEARCH"
            message = "CAP" if box["it"] >= a.cap else msg
            break

    if st["reads"][-1]["it"] != box["it"]:
        take_read(x, box["it"])
    last = st["reads"][-1]
    if last["g"] <= GATE:
        gate_seen(box["it"])
    end_file = f"m5_46_{name}_end.npz"
    np.savez(a.out_dir / end_file, m6=x.reshape(mod.shape6), x=mod.x)
    rec = dict(chain=name, spec=c, model=dict(N=c["N"], half_box=c["hb"], spacing=mod.h, delta=DELTA, L1=L1, kappa=1.0, v=1.0, stencil="averaged",
                                               fixed_chart=True, free_faces=free, J=0.0),
               half_widths=[s.W for s in surfaces], start=st["start"], descent="steepest" if steepest else "L-BFGS-B", step=st["step"], pilot=st["pilot"],
               rises_of_U=st.get("rises"), ended=message, iterations=box["it"], evaluations=st["nfev"], seconds=time.time() - t0,
               gate=GATE, gate_first_met_at=st["gate_first"], gate_met_at_the_end=bool(last["g"] <= GATE), gmax_over_h3_at_the_end=last["g"],
               U_at_the_end=last["U"], events=st["events"], resumed_at=st["resumed_at"], end_file=end_file, snapshots=st["snaps"],
               row_legend=dict(sets="count, radius, center (3), normal (3), least distance to the faces, faces touched (bits: 2 axis + side), max norm, spread, off plane",
                               q="one character per nested surface, inner to outer: |q| when liftable, x pierced, g gap at or under 0.2, l lift not consistent",
                               pt="pierced triangles per surface", ar="1 where the surface surrounds the loop", en="number of sets the surface surrounds",
                               loop="index of the loop in sets (largest clear set, else largest set)", Ut="U2, U4, UV", mg="least gap over the surfaces",
                               dev="largest distance of |q| from a whole number", g="max |dU/dm| / h^3", t="seconds"),
               reads=st["reads"], finished=True)
    jf.write_text(json.dumps(rec, separators=(",", ":")) + "\n")
    (a.work_dir / "chains").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(jf, a.work_dir / "chains" / f"{name}.json")
    ckpt.unlink(missing_ok=True)
    state_f.unlink(missing_ok=True)
    log(f"CHAIN END it {box['it']} {message}: U {last['U']:.6f} g {last['g']:.3g} sets {last['n']} clear {last['clear']} q {last['q']} ({rec['seconds'] / 60:.1f} min)")


def dilation(a):
    """U along the dilation of each base state, s from 1 to the pull of 13.8."""
    out = {}
    for b in ("B0a", "B0b", "B0c", "B0d"):
        f = a.out_dir / f"m5_46_{b}_end.npz"
        if not f.exists():
            continue
        c = CHAINS[b]
        z = np.load(f, allow_pickle=False)
        mod = Q.Model(c["N"], c["hb"], DELTA, L1, stencil="averaged", fixed_chart=True, free_faces=True)
        lp = base_loop(z["m6"], z["x"])
        marks = {10.0: 10.0 / (2 * lp["radius"]), 13.8: 13.8 / (2 * lp["radius"])}
        ss = sorted(set(np.linspace(1.0, marks[13.8], 25).tolist()) | set(marks.values()))
        rows = []
        for s in ss:
            m6 = z["m6"] if s == 1.0 else dilate(z["m6"], z["x"], lp["center"], s)
            p = mod.evaluate(m6.reshape(-1), 0.0, need_grad=False, need_I=False)
            line = D.line_read(m6, z["x"])[0]
            k = D.the_loop(line["sets"])
            rows.append(dict(s=s, separation=2 * lp["radius"] * s, U=p["U"], U2=p["U2"], U4=p["U4"], UV=p["UV"], sets=len(line["sets"]),
                             radius=(line["sets"][k]["radius"] if k is not None else None),
                             least_distance_to_faces=(line["sets"][k]["least_distance_to_faces"] if k is not None else None)))
            print(f"{b} s {s:.4f} separation {rows[-1]['separation']:.3f} U {p['U']:.5f} (U2 {p['U2']:.4f} U4 {p['U4']:.4f} UV {p['UV']:.4f}) radius {rows[-1]['radius']}", flush=True)
        out[b] = dict(half_box=c["hb"], N=c["N"], spacing=mod.h, base_radius=lp["radius"], base_center=lp["center"], s_at_separation_10=marks[10.0],
                      s_at_separation_13p8=marks[13.8], curve=rows)
    (a.out_dir / "m5_46_dilation.json").write_text(json.dumps(out, indent=1) + "\n")
    shutil.copyfile(a.out_dir / "m5_46_dilation.json", a.work_dir / "m5_46_dilation.json")


def pool(a):
    names = [n for n, c in CHAINS.items() if c["kind"] == a.pool and (not a.only or n in a.only)]
    names.sort(key=lambda n: -CHAINS[n]["N"])
    t0, running, state = time.time(), {}, {n: "pending" for n in names}
    for n in names:
        f = a.out_dir / f"m5_46_chain_{n}.json"
        if f.exists() and json.loads(f.read_text()).get("finished"):
            state[n] = "done"
    while True:
        for n, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[n]
            state[n] = "done" if rc == 0 else ("interrupted" if rc == 75 else f"failed {rc}")
            print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - t0) / 60:5.1f} min] {n}: {state[n]}", flush=True)
        for n in names:
            if len(running) >= a.cores:
                break
            if state[n] == "pending":
                w = a.work_dir / "chains" / n
                w.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ)
                for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
                    env[var] = "1"
                cmd = [sys.executable, "-I", str(HERE / "m5_46_pull.py"), "--chain", n, "--out-dir", str(a.out_dir), "--work-dir", str(a.work_dir),
                       "--wall-hours", str(a.wall_hours), "--cap", str(a.cap)]
                running[n] = subprocess.Popen(cmd, stdout=open(w / "stdout.txt", "a"), stderr=subprocess.STDOUT, env=env)
                state[n] = "running"
                print(f"[{time.strftime('%H:%M:%S')}] launch {n}", flush=True)
        (a.work_dir / f"pool_{a.pool}.json").write_text(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), minutes=(time.time() - t0) / 60, states=state), indent=1) + "\n")
        if not running:
            break
        time.sleep(a.poll)
    print(f"pool {a.pool} end after {(time.time() - t0) / 60:.1f} min: {state}", flush=True)
    sys.exit(0 if all(s == "done" for s in state.values()) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pool", choices=("base", "release", "added"))
    ap.add_argument("--only", nargs="*", help="with --pool: these chains only")
    ap.add_argument("--dilation", action="store_true")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_46")
    ap.add_argument("--wall-hours", type=float, default=WALL_HOURS)
    ap.add_argument("--cap", type=int, default=CAP)
    a = ap.parse_args()
    if a.list:
        for k, c in CHAINS.items():
            print(k, c, "spacing", 2 * c["hb"] / (c["N"] - 1))
    elif a.pool:
        pool(a)
    elif a.dilation:
        dilation(a)
    elif a.chain:
        run_chain(a.chain, a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
