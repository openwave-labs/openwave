"""M5.39: the pilot of the pool forecast. One evaluation of the Card C energy and gradient
(m5_39_cardc.py) timed alone and with several copies running at once, and with threads.

    python3 m5_39_forecast_pilot.py --grid 45 --box 6.075 --copies 1 12
    python3 m5_39_forecast_pilot.py --grid 91 --box 10.125 --copies 1 --threads 1 3 6

The seconds belong to the machine they were taken on; the ratio between the rows is what the
forecast uses (how much concurrent jobs slow each other, and what a thread buys).
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"


def worker(grid, box, threads, evals, start_at, queue):
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numba as nb
    import m5_39_cardc as C

    nb.set_num_threads(threads)
    state = C.seed(grid, box, "linked")[0]
    card = C.CardC(grid, box)
    card(state)  # compile and warm
    while time.time() < start_at:
        time.sleep(0.01)
    t0 = time.perf_counter()
    for _ in range(evals):
        card(state)
    queue.put((time.perf_counter() - t0) / evals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--box", type=float, required=True)
    ap.add_argument("--copies", type=int, nargs="+", default=[1])
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--evals", type=int, default=10)
    a = ap.parse_args()
    ctx = mp.get_context("spawn")
    rows = []
    for copies in a.copies:
        for threads in a.threads:
            queue = ctx.Queue()
            start_at = time.time() + 25 + 0.5 * copies
            procs = [ctx.Process(target=worker, args=(a.grid, a.box, threads, a.evals, start_at, queue)) for _ in range(copies)]
            for p in procs:
                p.start()
            times = [queue.get() for _ in procs]
            for p in procs:
                p.join()
            rows.append(dict(grid=a.grid, copies=copies, threads=threads, seconds_per_evaluation=sum(times) / len(times), worst=max(times)))
            print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
