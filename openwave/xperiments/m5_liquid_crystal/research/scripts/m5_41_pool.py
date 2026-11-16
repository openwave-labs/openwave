"""M5.41 R2: the pool of the twelve routes, one process per route (m5_39_relax.py with --stops).

Routes (m5_41_task_details.md, section 4), all at half-box 6.075 on Card C:

  77^3, 65^3   iso, chain, smooth          four threads each
  55^3         iso, chain, smooth, author  two threads each
  45^3         iso, chain                  two threads each

Every route runs the stops 1000, 3000, 10000, 30000 with the cap 33000, the optimizer's own gradient
stop off (--stop-gate 1e-9), the evaluation limit above the cap, and at most three restarts after an
abnormal end. The start control of each route is the energy of its first evaluation, to 1e-10.

Scheduling: a queue in the order above, finest mesh first; a route is launched when its threads fit
under --cores busy cores. A route keeps its thread count for its whole life.

The pool is restartable: a finished route is read from its JSON, an interrupted one continues from
its checkpoint with a fresh optimizer memory (recorded in its JSON as a resume).
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
BOX = 6.075
STOPS = "1000,3000,10000,30000"
CAP = 33000
MAXFUN = 400000
RESTARTS = 3
CHAIN_SOURCE = {45: "r2_stage37", 55: "r2_g45_chain", 65: "r2_g55_chain", 77: "r2_g65_chain"}
START_ENERGY = {
    (45, "iso"): 951.1869569484667, (45, "chain"): 76.17486674698928,
    (55, "iso"): 960.2457216651314, (55, "chain"): 76.10589655857467,
    (55, "smooth"): 931.1781080672494, (55, "author"): 75.56196067295708,
    (65, "iso"): 966.5931374279587, (65, "chain"): 75.97938170352795, (65, "smooth"): 939.0364814034779,
    (77, "iso"): 972.1235440994741, (77, "chain"): 76.09773221330995, (77, "smooth"): 945.730922837993,
}
QUEUE = ([(77, k, 4) for k in ("iso", "chain", "smooth")] + [(65, k, 4) for k in ("iso", "chain", "smooth")]
         + [(55, k, 2) for k in ("iso", "chain", "smooth", "author")] + [(45, k, 2) for k in ("iso", "chain")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_41")
    ap.add_argument("--chain-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_39",
                    help="the folder holding the hand-off states of the refinement chain of M5.39")
    ap.add_argument("--only", nargs="*", help="run these routes only")
    ap.add_argument("--poll", type=float, default=10.0)
    a = ap.parse_args()
    a.work_dir.mkdir(parents=True, exist_ok=True)
    log = open(a.work_dir / "pool.log", "a")
    t0 = time.time()

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')} +{(time.time() - t0) / 3600:6.2f}h] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    jobs = []
    for grid, kind, threads in QUEUE:
        name = f"r2_g{grid}_{kind}"
        if a.only and name not in a.only:
            continue
        if kind == "chain":
            start, source = "refine", a.chain_dir / CHAIN_SOURCE[grid] / "handoff.npz"
        elif kind == "author":
            start, source = "state", a.out_dir / "m5_40_author_g55.npz"
        else:
            start, source = kind, None
        if source is not None and not source.exists():
            sys.exit(f"the start of {name} is missing: {source}")
        jobs.append(dict(name=name, grid=grid, kind=kind, threads=threads, start=start, source=source,
                         expect=START_ENERGY[(grid, kind)]))

    def result_of(name):
        p = a.out_dir / f"m5_41_{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    state = {j["name"]: ("done" if result_of(j["name"]) is not None else "pending") for j in jobs}
    running, stopping = {}, dict(flag=False)

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            return False

    lock = a.work_dir / "pool.pid"
    if lock.exists() and alive(int(lock.read_text() or 0)):
        sys.exit(f"another pool is running (pid {lock.read_text()})")
    orphans = [j["name"] for j in jobs if (a.work_dir / j["name"] / "pid").exists() and state[j["name"]] != "done"
               and alive(int((a.work_dir / j["name"] / "pid").read_text() or 0))]
    if orphans:
        sys.exit(f"routes of an earlier pool are still running: {orphans}")
    lock.write_text(str(os.getpid()))

    def launch(job):
        name = job["name"]
        work = a.work_dir / name
        work.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(HERE / "m5_39_relax.py"), "--prefix", "m5_41", "--name", name, "--grid", str(job["grid"]),
               "--box", str(BOX), "--start", job["start"], "--cap", str(CAP), "--polish", "0", "--stops", STOPS,
               "--stop-gate", "1e-9", "--maxfun", str(MAXFUN), "--restarts", str(RESTARTS),
               "--expect-start-energy", repr(job["expect"]), "--threads", str(job["threads"]),
               "--out-dir", str(a.out_dir), "--work-dir", str(a.work_dir)]
        if job["source"] is not None:
            cmd += ["--source", str(job["source"])]
        env = dict(os.environ)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[var] = "1"
        env["NUMBA_NUM_THREADS"] = str(job["threads"])
        out = open(work / "log.txt", "a")
        running[name] = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
        (work / "pid").write_text(str(running[name].pid))
        state[name] = "running"
        say(f"LAUNCH {name} threads {job['threads']}" + (f" from {job['source'].name} of {job['source'].parent.name}" if job["source"] else ""))

    signal.signal(signal.SIGTERM, lambda *_: stopping.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stopping.update(flag=True))
    by_name = {j["name"]: j for j in jobs}
    say(f"pool start: {sum(s == 'pending' for s in state.values())} pending, {sum(s == 'done' for s in state.values())} done, cores {a.cores}")

    while True:
        for name, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[name]
            r = result_of(name)
            if rc == 0 and r is not None:
                state[name] = "done"
                last = r["stops"][-1] if r.get("stops") else {}
                say(f"DONE {name}: end {r.get('route_end')} iterations {r.get('iterations')} E {r.get('energy')} "
                    f"last stop at {last.get('iteration')} gate {last.get('gate')} met {last.get('gate_met')} "
                    f"restarts {len(r.get('restarts', []))} resumes {len(r.get('resumed_at', []))} wall {r.get('wall_time_s', 0) / 3600:.2f}h")
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
            say("pool stopped on a signal or an interrupted route; rerun to continue")
            lock.unlink(missing_ok=True)
            sys.exit(75)

        busy = sum(by_name[n]["threads"] for n in running)
        for job in jobs:
            if state[job["name"]] == "pending" and busy + job["threads"] <= a.cores:
                launch(job)
                busy += job["threads"]

        prog = {}
        for n in running:
            p = a.work_dir / n / "progress.json"
            if p.exists():
                try:
                    prog[n] = json.loads(p.read_text())
                except Exception:  # noqa: BLE001
                    pass
        (a.work_dir / "pool_status.json").write_text(json.dumps(dict(
            time=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_h=(time.time() - t0) / 3600, states=state,
            running={n: by_name[n]["threads"] for n in running}, progress=prog), indent=1) + "\n")
        if not running:
            break
        time.sleep(a.poll)

    say(f"pool end: done {sum(s == 'done' for s in state.values())}, failed {[n for n, s in state.items() if s == 'failed']}, "
        f"never launched {[n for n, s in state.items() if s == 'pending']}")
    (a.work_dir / "pool_finished.json").write_text(json.dumps(dict(states=state, elapsed_h=(time.time() - t0) / 3600), indent=1) + "\n")
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
