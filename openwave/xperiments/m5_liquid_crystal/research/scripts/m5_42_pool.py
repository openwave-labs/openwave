"""M5.42 R2 to R8: the pool of the chains, one single-thread process per chain (m5_42_relax.py).

Scheduling: the chains of the largest grids first; a chain is launched when a core is free under --cores
and the paths it starts from exist (the band chains of R8 start from states of R3 and R4). The pool is
restartable: a finished chain is read from its JSON, an interrupted one continues from its checkpoint.
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
sys.path.insert(0, str(HERE))
from m5_42_relax import chains  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_42" / "chains")
    ap.add_argument("--wall-hours", type=float, default=12.0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--poll", type=float, default=15.0)
    a = ap.parse_args()
    a.work_dir.mkdir(parents=True, exist_ok=True)
    log = open(a.work_dir / "pool.log", "a")
    t0 = time.time()

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')} +{(time.time() - t0) / 3600:6.2f}h] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    C = chains()
    names = [n for n in C if not a.only or n in a.only]
    names.sort(key=lambda n: (-C[n]["model"]["N"], n))

    def record(name):
        p = a.work_dir / f"{name}.json"
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except Exception:  # noqa: BLE001
            return None

    def finished(name):
        r = record(name)
        return bool(r and r.get("finished"))

    def ready(name):
        for src, what in C[name].get("needs", []):
            r = record(src)
            if r is None:
                return False
            if what == "DONE":
                if not r.get("finished"):
                    return False
            elif what not in [p["name"] for p in r["paths"]]:
                return False
        return True

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            return False

    lock = a.work_dir / "pool.pid"
    if lock.exists() and alive(int(lock.read_text() or 0)):
        sys.exit(f"another pool is running (pid {lock.read_text()})")
    lock.write_text(str(os.getpid()))
    state = {n: ("done" if finished(n) else "pending") for n in names}
    running, stopping = {}, dict(flag=False)
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stopping.update(flag=True))
    say(f"pool start: {sum(s == 'pending' for s in state.values())} pending, {sum(s == 'done' for s in state.values())} done, cores {a.cores}")

    def launch(name):
        work = a.work_dir / name
        work.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
            env[var] = "1"
        out = open(work / "stdout.txt", "a")
        cmd = [sys.executable, "-I", str(HERE / "m5_42_relax.py"), "--chain", name, "--out-dir", str(a.out_dir), "--work-dir", str(a.work_dir),
               "--wall-hours", str(a.wall_hours)]
        running[name] = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
        (work / "pid").write_text(str(running[name].pid))
        state[name] = "running"
        say(f"LAUNCH {name} (N {C[name]['model']['N']})")

    while True:
        for name, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[name]
            r = record(name)
            if rc == 0 and r and r.get("finished"):
                state[name] = "done"
                labels = [f"{p['name']}:{'skip' if p.get('skipped') else p['label'] + ('/C' if p['converged'] else '')}" for p in r["paths"]]
                say(f"DONE {name}: {r['ended']}; {' '.join(labels)}; {r['wall_seconds'] / 3600:.2f} h")
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
            say("pool stopped on a signal or an interrupted chain; rerun to continue")
            lock.unlink(missing_ok=True)
            sys.exit(75)
        for name in names:
            if len(running) >= a.cores:
                break
            if state[name] == "pending" and ready(name):
                launch(name)
        blocked = [n for n in names if state[n] == "pending"]
        prog = {}
        for n in running:
            p = a.work_dir / n / "progress.json"
            if p.exists():
                try:
                    prog[n] = json.loads(p.read_text())
                except Exception:  # noqa: BLE001
                    pass
        (a.work_dir / "pool_status.json").write_text(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_h=(time.time() - t0) / 3600,
                                                                       states=state, progress=prog), indent=1) + "\n")
        if not running:
            if blocked:
                say(f"nothing running and {len(blocked)} chains cannot start (a source failed or was skipped): {blocked}")
            break
        time.sleep(a.poll)

    say(f"pool end: done {sum(s == 'done' for s in state.values())}, failed {[n for n, s in state.items() if s == 'failed']}, "
        f"never launched {[n for n, s in state.items() if s == 'pending']}")
    (a.work_dir / "pool_finished.json").write_text(json.dumps(dict(states=state, elapsed_h=(time.time() - t0) / 3600), indent=1) + "\n")
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
