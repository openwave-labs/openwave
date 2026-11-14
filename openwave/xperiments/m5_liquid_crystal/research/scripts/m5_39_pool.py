"""M5.39 R2 and R3: the pool of relaxations, one process per job (m5_39_relax.py).

Jobs (the plan of m5_39_task_details.md, section 4):

  R2, mesh ladder   33^3 at half-box 3.6; 45^3, 55^3, 65^3, 77^3 at 6.075. Five starts per grid:
                    linked, smooth, iso (seeds); chain (the hand-off of the next coarser stage,
                    refined) and reset (the same refined state with the card's frame reset).
                    Starting stages 17^3, 25^3 (box 3.6) and 29^3, 37^3 (box 6.075) stop at hand-off.
  R2, box ladder    73^3 at 8.1: the three seeds, and the box continuation of the lowest stationary
                    55^3 state (padded), with and without the frame reset. 91^3 at 10.125: the two
                    continuations of the lowest stationary 73^3 state, only if there is one.
  R3                the record ablation: on 33^3, 45^3, 55^3 a pair of continuations (the card and
                    the ablated card) from the lowest stationary state of the grid, and one ablated
                    run from the linked seed on 33^3.

Scheduling: at most --cores cores busy. The chains and the box continuation launch as soon as they
are ready; while a chain stage is still to come, two cores are kept free for it and its sibling.
Every job runs on one thread, except that a job launched when fewer jobs remain than there are cores
takes an equal share of the free cores as threads, fixed for the life of the job.

The pool is restartable: finished jobs are read from their JSON, an interrupted job continues from
its checkpoint with a fresh optimizer memory (recorded in its JSON).
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = ("linked", "smooth", "iso")
BOX_A, BOX_B, BOX_73, BOX_91 = 3.6, 6.075, 8.1, 10.125


def build_jobs():
    jobs = []

    def add(name, cls, grid, box, start, source=None, cap=6000, polish=2000, handoff=0, lam=1.0, needs=None):
        jobs.append(dict(name=name, cls=cls, grid=grid, box=box, start=start, source=source, cap=cap, polish=polish,
                         handoff=handoff, lam=lam, needs=needs))

    # the two chains
    add("r2_stage17", "C", 17, BOX_A, "linked", cap=2000, polish=0, handoff=2000)
    add("r2_stage25", "C", 25, BOX_A, "refine", ("handoff", "r2_stage17"), cap=2000, polish=0, handoff=2000)
    add("r2_g33_chain", "C", 33, BOX_A, "refine", ("handoff", "r2_stage25"))
    add("r2_g33_reset", "D", 33, BOX_A, "refine-reset", ("handoff", "r2_stage25"))
    add("r2_stage29", "C", 29, BOX_B, "linked", cap=2000, polish=0, handoff=2000)
    add("r2_stage37", "C", 37, BOX_B, "refine", ("handoff", "r2_stage29"), cap=2000, polish=0, handoff=2000)
    prev = "r2_stage37"
    for grid in (45, 55, 65, 77):
        add(f"r2_g{grid}_chain", "C", grid, BOX_B, "refine", ("handoff", prev), handoff=2000 if grid < 77 else 0)
        add(f"r2_g{grid}_reset", "D", grid, BOX_B, "refine-reset", ("handoff", prev))
        prev = f"r2_g{grid}_chain"
    # the box continuation and its tail
    add("r2_g73_cont", "C", 73, BOX_73, "pad", ("census", "g55"))
    add("r2_g73_contreset", "D", 73, BOX_73, "pad-reset", ("census", "g55"))
    add("r2_g91_cont", "C", 91, BOX_91, "pad", ("census", "g73"), cap=3000, polish=1000, needs="stationary")
    add("r2_g91_contreset", "D", 91, BOX_91, "pad-reset", ("census", "g73"), cap=3000, polish=1000, needs="stationary")
    # fillers: the short jobs
    for kind in SEEDS:
        add(f"r2_g33_{kind}", "F", 33, BOX_A, kind)
    add("r3_a33_unpaired", "F", 33, BOX_A, "linked", lam=0.0)
    for grid, box in ((33, BOX_A), (45, BOX_B), (55, BOX_B)):
        add(f"r3_c{grid}_pair", "F", grid, box, "state", ("census", f"g{grid}"), cap=2000, polish=1000)
        add(f"r3_a{grid}_pair", "F", grid, box, "state", ("census", f"g{grid}"), cap=2000, polish=1000, lam=0.0)
    # the seed starts, in the order of what waits on them
    for grid, box in ((55, BOX_B), (77, BOX_B), (73, BOX_73), (65, BOX_B), (45, BOX_B)):
        for kind in SEEDS:
            add(f"r2_g{grid}_{kind}", "S", grid, box, kind)
    return jobs


CENSUS = {f"g{g}": [f"r2_g{g}_{k}" for k in SEEDS + ("chain", "reset")] for g in (33, 45, 55, 65, 77)}
CENSUS["g73"] = [f"r2_g73_{k}" for k in SEEDS + ("cont", "contreset")]
CHAIN = {"r2_stage25", "r2_stage37"} | {f"r2_g{g}_{k}" for g in (33, 45, 55, 65, 77) for k in ("chain", "reset")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_39")
    ap.add_argument("--only", nargs="*", help="run these jobs only (for tests)")
    ap.add_argument("--poll", type=float, default=5.0)
    a = ap.parse_args()
    a.work_dir.mkdir(parents=True, exist_ok=True)
    log = open(a.work_dir / "pool.log", "a")
    t0 = time.time()

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')} +{(time.time() - t0) / 3600:6.2f}h] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    jobs = build_jobs()
    if a.only:
        jobs = [j for j in jobs if j["name"] in a.only]
    by_name = {j["name"]: j for j in jobs}
    order = {"C": 0, "D": 1, "F": 2, "S": 3}
    state = {j["name"]: "pending" for j in jobs}
    running, threads_of, notes = {}, {}, {}
    stopping = dict(flag=False)

    def result_path(name):
        return a.out_dir / f"m5_39_{name}.json"

    def load_result(name):
        p = result_path(name)
        return json.loads(p.read_text()) if p.exists() else None

    for j in jobs:
        if load_result(j["name"]) is not None:
            state[j["name"]] = "done"

    def census(key):
        """The lowest stationary state of a grid, once its five jobs have ended; None before that."""
        names = CENSUS[key]
        if any(state.get(n, "done") not in ("done", "failed", "skipped") for n in names):
            return None
        rows = [r for r in (load_result(n) for n in names) if r is not None and r.get("energy") == r.get("energy") and r.get("energy") is not None]
        stationary = [r for r in rows if r["label"] == "STATIONARY"]
        pool = stationary or rows
        if not pool:
            return dict(grid=key, winner=None, from_stationary=False)
        best = min(pool, key=lambda r: r["energy"])
        sel = dict(grid=key, winner=best["name"], from_stationary=bool(stationary), energy=best["energy"], branch=best["branch"],
                   jobs={r["name"]: dict(label=r["label"], energy=r["energy"], residual=r["residual"], branch=r["branch"],
                                         center_eigenvalues=r["center_eigenvalues"], iterations=r["iterations"]) for r in rows})
        (a.out_dir / f"m5_39_r2_census_{key}.json").write_text(json.dumps(sel, indent=1) + "\n")
        return sel

    def source_of(job):
        """The path of the job's starting state, or None while it does not exist; 'skip' if never."""
        src = job["source"]
        if src is None:
            return ""
        kind, key = src
        if kind == "handoff":
            p = a.work_dir / key / "handoff.npz"
            return p if p.exists() else None
        sel = census(key)
        if sel is None:
            return None
        if sel["winner"] is None or (job["needs"] == "stationary" and not sel["from_stationary"]):
            return "skip"
        notes[job["name"]] = dict(start_from=sel["winner"], start_is_stationary=sel["from_stationary"])
        return a.out_dir / f"m5_39_{sel['winner']}.npz"

    def launch(job, source, threads):
        name = job["name"]
        work = a.work_dir / name
        work.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(HERE / "m5_39_relax.py"), "--name", name, "--grid", str(job["grid"]), "--box", str(job["box"]),
               "--start", job["start"], "--cap", str(job["cap"]), "--polish", str(job["polish"]), "--handoff", str(job["handoff"]),
               "--lam", str(job["lam"]), "--threads", str(threads), "--out-dir", str(a.out_dir), "--work-dir", str(a.work_dir)]
        if source:
            cmd += ["--source", str(source)]
        env = dict(os.environ)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[var] = "1"
        env["NUMBA_NUM_THREADS"] = str(threads)
        out = open(work / "log.txt", "a")
        running[name] = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
        (work / "pid").write_text(str(running[name].pid))
        threads_of[name] = threads
        state[name] = "running"
        say(f"LAUNCH {name} threads {threads}" + (f" from {Path(source).name}" if source else ""))

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            return False

    # one pool at a time, and no job of an earlier pool still running
    lock = a.work_dir / "pool.pid"
    if lock.exists() and alive(int(lock.read_text() or 0)):
        sys.exit(f"another pool is running (pid {lock.read_text()})")
    orphans = [j["name"] for j in jobs if (a.work_dir / j["name"] / "pid").exists()
               and state[j["name"]] != "done" and alive(int((a.work_dir / j["name"] / "pid").read_text() or 0))]
    if orphans:
        sys.exit(f"jobs of an earlier pool are still running: {orphans}")
    lock.write_text(str(os.getpid()))

    def on_signal(*_):
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    say(f"pool start: {sum(s == 'pending' for s in state.values())} pending, {sum(s == 'done' for s in state.values())} done, cores {a.cores}")

    while True:
        for name, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[name]
            if rc == 0 and load_result(name) is not None:
                r = load_result(name)
                state[name] = "done"
                say(f"DONE {name}: {r['label']} E {r.get('energy')} residual {r.get('residual')} iterations {r.get('iterations')} "
                    f"center {[round(x, 4) for x in r.get('center_eigenvalues', [])]} wall {r.get('wall_time_s', 0) / 3600:.2f}h")
            elif rc == 75:
                state[name] = "pending"
                stopping["flag"] = True
                say(f"INTERRUPTED {name}")
            else:
                state[name] = "failed"
                say(f"FAILED {name} exit {rc}")
        if stopping["flag"]:
            for proc in running.values():
                proc.send_signal(signal.SIGTERM)
            for proc in running.values():
                proc.wait()
            say("pool stopped on a signal or an interrupted job; rerun to continue")
            sys.exit(75)

        in_use = sum(threads_of[n] for n in running)
        free = a.cores - in_use
        spare = 2 if any(state.get(n) == "pending" for n in CHAIN) else 0
        ready = []
        for job in sorted((j for j in jobs if state[j["name"]] == "pending"), key=lambda j: order[j["cls"]]):
            src = source_of(job)
            if src == "skip":
                state[job["name"]] = "skipped"
                result_path(job["name"]).write_text(json.dumps(dict(name=job["name"], label="NOT RUN", reason="the smaller box holds no stationary state")) + "\n")
                say(f"SKIP {job['name']}")
            elif src is not None:
                ready.append((job, src))
        chosen, budget = [], free
        for job, src in ready:
            need_free = 1 if job["cls"] in ("C", "D") else 1 + spare
            if budget >= need_free:
                chosen.append((job, src))
                budget -= 1
        unfinished = sum(s in ("pending", "running") for s in state.values())
        extra = 0
        if chosen and unfinished < a.cores:
            extra = max(0, (budget - spare) // len(chosen))
        for job, src in chosen:
            launch(job, src, 1 + extra)

        status = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_h=(time.time() - t0) / 3600, states=state, notes=notes,
                      running={n: threads_of[n] for n in running})
        prog = {}
        for n in running:
            p = a.work_dir / n / "progress.json"
            if p.exists():
                try:
                    prog[n] = json.loads(p.read_text())
                except Exception:  # noqa: BLE001
                    pass
        status["progress"] = prog
        (a.work_dir / "pool_status.json").write_text(json.dumps(status, indent=1) + "\n")
        if not running and not chosen:
            break
        time.sleep(a.poll)

    left = [n for n, s in state.items() if s == "pending"]
    say(f"pool end: done {sum(s == 'done' for s in state.values())}, failed {[n for n, s in state.items() if s == 'failed']}, "
        f"skipped {[n for n, s in state.items() if s == 'skipped']}, never ready {left}")
    (a.work_dir / "pool_finished.json").write_text(json.dumps(dict(states=state, notes=notes, elapsed_h=(time.time() - t0) / 3600), indent=1) + "\n")


if __name__ == "__main__":
    main()
