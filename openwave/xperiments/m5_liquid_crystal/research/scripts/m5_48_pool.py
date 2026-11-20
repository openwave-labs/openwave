"""M5.48: the pool. Starts the chains of m5_48_relax.py in the order of the plan, one single-threaded process per chain.

Order (m5_48_task_details.md, section 2): the largest grids first; on one grid the chains of P3 from the seed, the carried
chains of P3, the references of P1, the chains at J without L1, those with L1 = 0.03, then the return chains. A chain starts
when a slot is free and what it needs has ended:

    a carried chain of P3      its source chain has a finished record
    a chain of P1 at J         its reference has a finished record and is CONVERGED under the gate of the candidate; if it is
                               not, the state is UNDECIDED at that candidate and the chain at J is not run
    a return chain of P1       its chain at J has a finished record, is CONVERGED and passes S2, S3 and S5
                               (m5_48_reads.state_at); a chain at J that is not CONVERGED leaves the state UNDECIDED, one that
                               fails a clause has failed, and neither needs a return: the pool records which

The pool is restartable: a finished chain is read from its record, a chain still running from an earlier pool is found by
its process id, and an interrupted chain continues from its checkpoint when the driver is started again. The chain of P2
is not started here (it runs by itself; `--tasks` names what the pool starts). A chain that exits with code 75 (the power
guard asked to wrap up) stops the pool, which exits 75 once the running chains have ended. When no chain is left to start
or to wait for, the pool writes checkpoints/m5_48/pool_done.json.

The ceiling is `--max` chains at once; a number written in checkpoints/m5_48/pool_max.txt replaces it while the pool runs.

Usage:  python m5_48_pool.py --tasks P1,P3 --max 11
        python m5_48_pool.py --tasks P1,P3 --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_48_relax as D  # noqa: E402


def rank(name, c):
    if c["task"] == "P3":
        return 0 if c["start"]["kind"] == "seed" else 1
    if c.get("role") == "ref":
        return 2
    if c.get("role") == "back":
        return 5
    return 3 if c["L1"] == 0.0 else 4


def order(C, tasks):
    names = [n for n, c in C.items() if c["task"] in tasks]
    return sorted(names, key=lambda n: (-C[n]["N"], rank(n, C[n]), names.index(n)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="P1,P3")
    ap.add_argument("--max", type=int, default=11)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="a regular expression: start only the chains whose name matches (for a test)")
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_48" / "chains")
    a = ap.parse_args()
    C = D.chains()
    names = order(C, set(a.tasks.split(",")))
    if a.only:
        import re
        names = [n for n in names if re.search(a.only, n)]
    if a.dry_run:
        for i, n in enumerate(names):
            print(i + 1, n, "N", C[n]["N"], "needs", C[n]["needs"], "cap", C[n]["cap"], "wall", C[n]["wall_hours"])
        print(len(names), "chains at most (a chain at J runs where its reference is CONVERGED, a return where the chain at J is CONVERGED and passes S2, S3 and S5)")
        return
    import m5_48_reads as RD  # the survival test; loads the compiled kernel

    a.work_dir.mkdir(parents=True, exist_ok=True)
    state_f = a.work_dir.parent / "m5_48_pool.json"
    max_f = a.work_dir.parent / "pool_max.txt"
    st = json.loads(state_f.read_text()) if state_f.exists() else dict(launched={}, skipped={}, exits={})
    logf = open(a.work_dir.parent / "m5_48_pool.log", "a")

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] pool {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    def save():
        tmp = state_f.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1) + "\n")
        os.replace(tmp, state_f)

    def finished(n):
        f = a.work_dir / f"m5_48_{n}.json"
        return f.exists() and json.loads(f.read_text()).get("finished", False)

    def alive(n):
        """A chain started by an earlier pool and still running."""
        q = st["launched"].get(n)
        if not q:
            return False
        try:
            os.kill(q["pid"], 0)
        except OSError:
            return False
        cmd = subprocess.run(["ps", "-p", str(q["pid"]), "-o", "command="], capture_output=True, text=True).stdout
        return f"--chain {n}" in cmd

    def ready(n):
        """True, False (wait), or a string (the chain is not needed, with the reason)."""
        for need in C[n]["needs"]:
            if st["skipped"].get(need) or (need in st["exits"] and st["exits"][need] not in (0,)):
                return f"its source {need} did not run to a record"
            if not finished(need):
                return False
        c = C[n]
        if c.get("role") == "J":
            ref = RD.load_rec(a.work_dir, n + "_ref")
            how, conv, _, _ = RD.table_word(ref)
            if not conv:
                return f"UNDECIDED: the reference is not CONVERGED under its gate {ref['gate']:.2e} ({how})"
        if c.get("role") == "back":
            t = RD.state_at(c["state"], c["k"], a.work_dir, a.out_dir)
            if t["converged"] is None:
                return False
            if not t["converged"]:
                return "UNDECIDED: " + t["why"]
            if not t["needs_a_return"]:
                return "the state " + t["why"] + " (I/I0 %.4f, share on W0 %.4f)" % (t["I_over_I0"], t["share_on_W0"])
        return True

    env = dict(os.environ)
    running, ext, stop, failed = {}, set(n for n in names if alive(n) and not finished(n)), False, set()
    if ext:
        log(f"found running from an earlier pool: {sorted(ext)}")
    log(f"starts: tasks {a.tasks}, {len(names)} chains at most, ceiling {a.max}")
    while True:
        for n, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[n]
                st["exits"][n] = rc
                save()
                log(f"{n} exited {rc}" + ("" if finished(n) else " WITHOUT a finished record"))
                if rc == 75:
                    stop = True
                elif rc != 0:
                    failed.add(n)
        ext = set(n for n in ext if alive(n) and not finished(n))
        ceiling = a.max
        if max_f.exists():
            try:
                ceiling = int(max_f.read_text().split()[0])
            except (ValueError, IndexError):
                pass
        pending = [n for n in names if not finished(n) and n not in running and n not in ext and n not in st["skipped"] and n not in failed]
        if not pending and not running and not ext:
            break
        waiting = 0
        if not stop:
            for n in pending:
                if len(running) + len(ext) >= ceiling:
                    break
                r = ready(n)
                if r is True:
                    out = open(a.work_dir / f"{n}.out", "a")
                    p = subprocess.Popen([sys.executable, "-I", str(HERE / "m5_48_relax.py"), "--chain", n, "--out-dir", str(a.out_dir), "--work-dir", str(a.work_dir)],
                                         stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                    running[n] = p
                    st["launched"][n] = dict(pid=p.pid, time=time.strftime("%Y-%m-%d %H:%M:%S"))
                    save()
                    log(f"started {n} (pid {p.pid}); {len(running) + len(ext)} running, ceiling {ceiling}")
                elif r is False:
                    waiting += 1
                else:
                    st["skipped"][n] = r
                    save()
                    log(f"{n} not run: {r}")
        if stop and not running:
            log("stopped on a power warning (a chain exited 75); the chains continue from their checkpoints when the pool is started again")
            sys.exit(75)
        if not running and not ext and not stop and waiting == len(pending) and waiting > 0:
            log(f"nothing can start: {len(pending)} chains wait for sources that have no record ({pending[:5]} ...)")
            sys.exit(2)
        time.sleep(a.poll)
    summary = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), tasks=a.tasks, chains_with_a_record=sum(finished(n) for n in names), not_run=st["skipped"], failed=sorted(failed), exits=st["exits"])
    done_f = a.work_dir.parent / "pool_done.json"
    done_f.write_text(json.dumps(summary, indent=1) + "\n")
    log(f"done: {summary['chains_with_a_record']} chains with a record, {len(st['skipped'])} not run, {len(failed)} failed {sorted(failed)}; wrote {done_f.name}")


if __name__ == "__main__":
    main()
