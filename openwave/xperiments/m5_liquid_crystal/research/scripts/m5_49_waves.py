"""M5.49: the base state on half-box 13 and the six wave jobs, run with the instruments of M5.46 and M5.47.

Nothing of the two runners is edited. This script gives them its own job list and renames what they write:

    B0e          m5_46_pull.run_chain on an added entry of its chain table: the base chain from the seed on
                 half-box 13 at spacing 0.25 (N = 105), the outer shell held, the descent rule and the gate of
                 M5.45, as B0a to B0d
    Ue, Ne, Se   m5_47_waves.run_job about B0e: noise in the mid polarization, the unperturbed twin, a second draw
    Sd           m5_47_waves.run_job about B0d of M5.46: a second draw of the noise of Ud
    Vq4, Vq55    m5_47_waves.run_job on the uniform vacuum of V2 (half-box 6, N = 49) with v scaled so that the
                 mid mass is 16 and 30.25 (the masses are linear in v), with the seventeen targets of the unit jobs

The runners write files named m5_46_* and m5_47_* into the directory they are given. Here that directory is a
staging folder under the checkpoints of this task; when a job has ended, its files are moved to the data folder
under the names m5_49_*. The base states the runners load by name (m5_46_B0d_end.npz, m5_46_B0e_end.npz) are
links in the staging folder.

The settings of the wave jobs are those of record of M5.47 (its section 7): length 72, the Hann window, step
0.02 and amplitude 1e-4 on the unit, step 0.01 and amplitude 1e-5 in the vacuum, smoothing 0.35.

One process, one thread per job; a job is restartable from the checkpoint of its runner.

Usage:  python m5_49_waves.py --list
        python m5_49_waves.py --chain B0e
        python m5_49_waves.py --job Sd
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import m5_46_pull as P
import m5_47_evolve as E
import m5_47_waves as W

M_MID2 = 60.0891  # the mid mass of the coded Hessian at the vacuum (M5.47, its section 8)
CHAINS = dict(B0e=dict(kind="base", hb=13.0, N=105, faces="held"))
BASES = dict(B0e=(13.0, 105))
JOBS = dict(
    Vq4=dict(kind="vacuum", hb=6.0, N=49, v=0.266271, amp=W.VAC_AMP, dt=W.VAC_STEP, seed=4722),
    Vq55=dict(kind="vacuum", hb=6.0, N=49, v=0.503419, amp=W.VAC_AMP, dt=W.VAC_STEP, seed=4722),
    Sd=dict(kind="unit", base="B0d", hb=10.0, N=81, v=1.0, amp=W.UNIT_AMP, dt=W.STEP, seed=4714),
    Ue=dict(kind="unit", base="B0e", hb=13.0, N=105, v=1.0, amp=W.UNIT_AMP, dt=W.STEP, seed=4705),
    Ne=dict(kind="unit", base="B0e", hb=13.0, N=105, v=1.0, amp=0.0, dt=W.STEP, seed=0),
    Se=dict(kind="unit", base="B0e", hb=13.0, N=105, v=1.0, amp=W.UNIT_AMP, dt=W.STEP, seed=4715),
)
WALL_HOURS = 3.0  # one hour in M5.46, on grids of at most 81^3; a cap of the machine, not of the descent rule


def register():
    """The job list of this task, entered in the tables of the two runners (in this process only)."""
    P.CHAINS.update(CHAINS)
    E.BASE.update(BASES)
    W.JOBS.update(JOBS)


def link(stage, data, name, source):
    dst = stage / name
    if not dst.exists():
        dst.symlink_to(data / source)


def run_chain(name, a):
    register()
    a.stage.mkdir(parents=True, exist_ok=True)
    done = a.data / f"m5_49_chain_{name}.json"
    if done.exists() and json.loads(done.read_text()).get("finished"):
        print(f"{name}: already finished", flush=True)
        return
    P.run_chain(name, SimpleNamespace(work_dir=a.work, out_dir=a.stage, wall_hours=a.wall_hours, cap=P.CAP))
    rec = json.loads((a.stage / f"m5_46_chain_{name}.json").read_text())
    rec["end_file"] = f"m5_49_{name}_end.npz"
    rec["runner"] = "m5_46_pull.run_chain, called by m5_49_waves.py"
    rec["wall_hours_allowed"] = a.wall_hours
    wall = dict(seconds=rec.pop("seconds", None), reads=[x.pop("t", None) for x in rec.get("reads", [])])
    (a.work / f"wall_{name}.json").write_text(json.dumps(wall) + "\n")   # wall times stay with the checkpoints
    shutil.move(a.stage / f"m5_46_{name}_end.npz", a.data / f"m5_49_{name}_end.npz")
    link(a.stage, a.data, f"m5_46_{name}_end.npz", f"m5_49_{name}_end.npz")
    done.write_text(json.dumps(rec, separators=(",", ":")) + "\n")


def run_job(name, a):
    register()
    a.stage.mkdir(parents=True, exist_ok=True)
    job = JOBS[name]
    done = a.data / f"m5_49_{name}.json"
    if done.exists() and json.loads(done.read_text()).get("finished"):
        print(f"{name}: already finished", flush=True)
        return
    if job["kind"] == "vacuum":
        W.VAC_TARGETS = list(W.UNIT_TARGETS)  # the seventeen targets of the unit jobs
    else:
        src = "m5_46_B0d_end.npz" if job["base"] == "B0d" else f"m5_49_{job['base']}_end.npz"
        link(a.stage, a.data, f"m5_46_{job['base']}_end.npz", src)
    W.run_job(name, SimpleNamespace(work_dir=a.work, out_dir=a.stage))
    rec = json.loads((a.stage / f"m5_47_{name}.json").read_text())
    arrays = []
    for f in rec["arrays"]:
        new = f.replace("m5_47_", "m5_49_", 1)
        shutil.move(a.stage / f, a.data / new)
        arrays.append(new)
    rec["arrays"] = arrays
    rec["runner"] = "m5_47_waves.run_job, called by m5_49_waves.py"
    (a.work / f"wall_{name}.json").write_text(json.dumps(dict(seconds=rec.pop("seconds", None))) + "\n")   # wall times stay with the checkpoints
    if job["kind"] == "vacuum":
        rec["mid_mass_squared"] = job["v"] * M_MID2
    done.write_text(json.dumps(rec) + "\n")
    (a.stage / f"m5_47_{name}.json").unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain")
    ap.add_argument("--job")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    ap.add_argument("--work", type=Path, default=HERE.parent / "checkpoints" / "m5_49")
    ap.add_argument("--wall-hours", type=float, default=WALL_HOURS)
    a = ap.parse_args()
    a.stage = a.work / "stage"
    if a.list:
        for k, c in {**CHAINS, **JOBS}.items():
            print(k, c, "spacing", 2 * c["hb"] / (c["N"] - 1))
    elif a.chain:
        run_chain(a.chain, a)
    elif a.job:
        run_job(a.job, a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
