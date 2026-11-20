"""M5.47: the wave jobs. Small noise about the uniform vacuum and about the relaxed unit, evolved in time.

The equations of motion and the integrator are those of m5_47_evolve.py (the action of M5.42 with its time part,
classical Runge-Kutta on (M, p), the outer shell held). Every job starts at rest (p = 0).

    V1, V2, V3   the uniform vacuum, its frame turned by a fixed rotation; noise in all six components
    Ua to Ud     the base states B0a to B0d of M5.46; noise times the mid polarization W(x)
    Na to Nd     the same states with no perturbation
    Ha, Hb       the noise of Ua, Ub at half the amplitude
    Da           the noise of Ua at half the step
    Sa, Sb       a second draw of the noise

The noise is seeded Gaussian white noise smoothed by a Gaussian of standard deviation 0.35 in length units,
of unit root mean square over the interior, zero on the shell, times the amplitude.

At every step j = 0 .. n the job adds to its Fourier sums, per target frequency w,

    A(x, w) += hann(j) exp(i w t_j) (M(x, t_j) - M0(x)),        hann(j) = sin^2(pi j / n)

over the six components, and records E, T and U every 20 steps. The vacuum jobs also keep, at every step, the
projections of M - M0 on twenty sine modes of the box.

The settings are those of the third and last try of the controls. Length 72 (48 in the plan), step 0.01 in
the vacuum jobs (0.005 in V3, whose heavy branch is faster; 0.0125 in the plan), step 0.02 on the unit (0.01 in
Da; 0.03 and 0.015 in the plan), noise amplitude 1e-5 in the vacuum (1e-4 in the plan) and 1e-4 on the unit
(1e-3 in the plan; Ha and Hb at half). What each change answered is in the task record: the energy control K11
(the step), the branches K6 and the linearity K9 (the amplitudes), the linearized equations K12 (the length).

One process, one thread per job; a job is restartable from its checkpoint.

Usage:  python m5_47_waves.py --list
        python m5_47_waves.py --job Ua
        python m5_47_waves.py --pool --cores 12
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from numba import njit

import m5_42_eq42 as Q
import m5_47_evolve as E

LENGTH = 72.0  # 48 at the first two tries (K12: the window mixes frequencies near a cutoff)
UNIT_AMP = 1e-4  # 1e-3 at the first two tries (K9: second-order terms on the waves outside W)
VAC_STEP = 0.01  # 0.0125 at the first two tries, over the length 48
VAC_AMP = 1e-5  # the vacuum jobs; 1e-4 at the first try, where second-order terms moved a few soft modes (K6)
STEP = 0.02  # the unit jobs; 0.03 at the first try, where the scheme took 2.3 % of the perturbation energy (K11)
SIGMA = 0.35
UNIT_TARGETS = [3.0 + 0.25 * i for i in range(13)] + [6.5, 7.0, 7.5, 8.0]
VAC_TARGETS = UNIT_TARGETS + [8.25, 8.5, 8.75, 9.0, 10.0, 11.0]
MODES = [(1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2), (3, 1, 1), (3, 2, 1), (3, 3, 1), (3, 3, 3), (4, 1, 1), (4, 2, 2),
         (4, 4, 1), (4, 4, 4), (5, 1, 1), (5, 3, 1), (5, 5, 2), (6, 1, 1), (6, 3, 2), (6, 6, 1), (7, 2, 1), (8, 1, 1)]
ROTVEC = (0.3, -0.5, 0.7)


def _unit(base, amp, dt, seed):
    hb, N = E.BASE[base]
    return dict(kind="unit", base=base, hb=hb, N=N, v=1.0, amp=amp, dt=dt, seed=seed)


JOBS = dict(
    V1=dict(kind="vacuum", hb=8.0, N=33, v=1.0, amp=VAC_AMP, dt=VAC_STEP, seed=4721),
    V2=dict(kind="vacuum", hb=6.0, N=49, v=1.0, amp=VAC_AMP, dt=VAC_STEP, seed=4722),
    V3=dict(kind="vacuum", hb=8.0, N=33, v=2.0, amp=VAC_AMP, dt=VAC_STEP / 2, seed=4721),
    Ua=_unit("B0a", UNIT_AMP, STEP, 4701), Ub=_unit("B0b", UNIT_AMP, STEP, 4702),
    Uc=_unit("B0c", UNIT_AMP, STEP, 4703), Ud=_unit("B0d", UNIT_AMP, STEP, 4704),
    Na=_unit("B0a", 0.0, STEP, 0), Nb=_unit("B0b", 0.0, STEP, 0),
    Nc=_unit("B0c", 0.0, STEP, 0), Nd=_unit("B0d", 0.0, STEP, 0),
    Ha=_unit("B0a", UNIT_AMP / 2, STEP, 4701), Hb=_unit("B0b", UNIT_AMP / 2, STEP, 4702),
    Da=_unit("B0a", UNIT_AMP, STEP / 2, 4701),
    Sa=_unit("B0a", UNIT_AMP, STEP, 4711), Sb=_unit("B0b", UNIT_AMP, STEP, 4712),
)


def vacuum_matrix(delta=E.DELTA):
    from scipy.spatial.transform import Rotation

    R = Rotation.from_rotvec(ROTVEC).as_matrix()
    return R @ np.diag([1.0, delta, 0.0]) @ R.T


def start_state(job, data):
    """(M0, M at t = 0) over the six components."""
    N, hb = job["N"], job["hb"]
    h = 2 * hb / (N - 1)
    interior = np.zeros((N, N, N), bool)
    interior[1:-1, 1:-1, 1:-1] = True
    if job["kind"] == "vacuum":
        m0 = np.ascontiguousarray(np.broadcast_to(Q.to6(vacuum_matrix()), (N, N, N, 6)))
        rng = np.random.default_rng(job["seed"])
        noise = np.stack([E.smooth_noise(rng, (N, N, N), SIGMA / h, interior) for _ in range(6)], -1)
        return m0, m0 + job["amp"] * noise
    m0, hb0, N0 = E.load_base(job["base"], data)
    assert (hb0, N0) == (hb, N)
    if job["amp"] == 0.0:
        return m0, m0.copy()
    rng = np.random.default_rng(job["seed"])
    noise = E.smooth_noise(rng, (N, N, N), SIGMA / h, interior)
    W = Q.to6(E.polarization(m0)[0])
    return m0, m0 + job["amp"] * noise[..., None] * W


@njit(cache=True)
def accumulate(m, m0, cr, ci, Ar, Ai):
    N = m.shape[0]
    nt = cr.shape[0]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for c in range(6):
                    d = m[i, j, k, c] - m0[i, j, k, c]
                    for t in range(nt):
                        Ar[i, j, k, c, t] += cr[t] * d
                        Ai[i, j, k, c, t] += ci[t] * d


def sine_table(N, nmax=8):
    i = np.arange(N)
    return np.array([np.sin(n * np.pi * i / (N - 1)) for n in range(1, nmax + 1)])


def mode_projections(S, d):
    t1 = np.tensordot(S, d, axes=(1, 0))
    return np.array([np.einsum("j,k,jkc->c", S[ny - 1], S[nz - 1], t1[nx - 1]) for nx, ny, nz in MODES])


def run_job(name, a):
    job = JOBS[name]
    N, hb, dt = job["N"], job["hb"], job["dt"]
    n = int(round(LENGTH / dt))
    targets = np.array(VAC_TARGETS if job["kind"] == "vacuum" else UNIT_TARGETS)
    work = a.work_dir / "jobs" / name
    work.mkdir(parents=True, exist_ok=True)
    summary = a.out_dir / f"m5_47_{name}.json"
    if summary.exists() and json.loads(summary.read_text()).get("finished"):
        print(f"{name}: already finished", flush=True)
        return
    guard = None
    try:
        sys.path.insert(0, os.path.expanduser("~/Library/Application Support/UPSSentinel"))
        import ups_guard

        ups_guard.register()
        guard = ups_guard
    except Exception:
        guard = None
    F = E.Field(N, hb, v=job["v"])
    rk = E.RK4(F)
    m0, m = start_state(job, a.out_dir)
    f = np.zeros(F.shape6)
    U_base = F.static(m0, f)["U"]
    gmax_base = float(np.abs(f[F.interior]).max() / F.h ** 3)
    start_max = float(np.abs(m - m0).max())
    p = np.zeros(F.shape6)
    nt = len(targets)
    Ar = np.zeros(F.shape6 + (nt,))
    Ai = np.zeros(F.shape6 + (nt,))
    vac = job["kind"] == "vacuum"
    S = sine_table(N) if vac else None
    series = np.zeros((n + 1, len(MODES), 6)) if vac else None
    energy, bad, j0, spent = [], 0, 0, 0.0
    ck = work / "state.npz"
    if ck.exists():
        z = np.load(ck, allow_pickle=False)
        m, p, Ar, Ai, j0 = z["m"].copy(), z["p"].copy(), z["Ar"].copy(), z["Ai"].copy(), int(z["j"])
        energy, bad, spent = z["energy"].tolist(), int(z["bad"]), float(z["spent"])
        if vac:
            series = z["series"].copy()
        print(f"{name}: resumed at step {j0}", flush=True)
    every = 400 if N >= 65 else 10 ** 9
    t0 = time.time()

    def save_state(j):
        tmp = work / "state_tmp.npz"
        extra = dict(series=series) if vac else {}
        np.savez(tmp, m=m, p=p, Ar=Ar, Ai=Ai, j=j, energy=np.array(energy), bad=bad, spent=spent + time.time() - t0, **extra)
        tmp.replace(ck)

    def sample(j):
        ph = targets * (j * dt)
        wj = np.sin(np.pi * j / n) ** 2
        accumulate(m, m0, wj * np.cos(ph), wj * np.sin(ph), Ar, Ai)
        if vac:
            series[j] = mode_projections(S, m - m0)

    for j in range(j0, n):
        sample(j)
        e = rk.step(m, p, dt)
        bad += e["bad"]
        if j % 20 == 0:
            energy.append((j * dt, e["E"], e["T"], e["U"], float(np.abs(m - m0).max())))
        if (j + 1) % 100 == 0:
            el = spent + time.time() - t0
            (work / "progress.json").write_text(json.dumps(dict(job=name, step=j + 1, of=n, E=e["E"], T=e["T"], U=e["U"],
                                                                 not_positive=bad, minutes=el / 60)) + "\n")
            if guard is not None and guard.wrap_up():
                save_state(j + 1)
                print(f"{name}: power warning, checkpoint at step {j + 1}", flush=True)
                sys.exit(75)
            if (j + 1) % every == 0 and j + 1 < n:
                save_state(j + 1)
        if not np.isfinite(e["E"]):
            raise SystemExit(f"{name}: the energy is not finite at step {j}")
    sample(n)
    e = F.rhs(m, p, rk.r, rk.f)
    energy.append((n * dt, e["E"], e["T"], e["U"], float(np.abs(m - m0).max())))
    seconds = spent + time.time() - t0
    np.savez(a.out_dir / f"m5_47_{name}_fourier.npz", Ar=Ar, Ai=Ai, targets=targets, dt=dt, steps=n, m_end=m, p_end=p)
    if vac:
        np.savez(a.out_dir / f"m5_47_{name}_modes.npz", series=series, modes=np.array(MODES), dt=dt, m0=Q.from6(m0[0, 0, 0]))
    en = np.array(energy)
    out = dict(job=name, spec=job, spacing=F.h, steps=n, length=n * dt, targets=targets.tolist(), window="hann",
               couplings=dict(delta=F.delta, L1=F.L1, kappa=F.kappa, v=F.v), U_base=U_base, gmax_base_over_h3=gmax_base,
               start_max=start_max, E0=float(en[0, 1]), perturbation_energy=float(en[0, 1] - U_base),
               drift=float(np.abs(en[:, 1] - en[0, 1]).max()), not_positive=bad, seconds=seconds,
               energy=dict(t=en[:, 0].tolist(), E=en[:, 1].tolist(), T=en[:, 2].tolist(), U=en[:, 3].tolist(),
                           max_displacement=en[:, 4].tolist()),
               arrays=[f"m5_47_{name}_fourier.npz"] + ([f"m5_47_{name}_modes.npz"] if vac else []), finished=True)
    summary.write_text(json.dumps(out) + "\n")
    if ck.exists():
        ck.unlink()
    print(f"{name}: end after {seconds / 60:.1f} min, drift {out['drift']:.3e}, perturbation energy {out['perturbation_energy']:.3e}", flush=True)


def pool(a):
    names = [n for n in JOBS if not a.only or n in a.only]
    names.sort(key=lambda n: -JOBS[n]["N"])
    t0, running, state = time.time(), {}, {n: "pending" for n in names}
    for n in names:
        f = a.out_dir / f"m5_47_{n}.json"
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
                w = a.work_dir / "jobs" / n
                w.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ)
                for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
                    env[var] = "1"
                cmd = [sys.executable, "-I", str(HERE / "m5_47_waves.py"), "--job", n, "--out-dir", str(a.out_dir), "--work-dir", str(a.work_dir)]
                running[n] = subprocess.Popen(cmd, stdout=open(w / "stdout.txt", "a"), stderr=subprocess.STDOUT, env=env)
                state[n] = "running"
                print(f"[{time.strftime('%H:%M:%S')}] launch {n}", flush=True)
        (a.work_dir / f"pool_{names[0]}.json").write_text(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), minutes=(time.time() - t0) / 60, states=state), indent=1) + "\n")
        if not running:
            break
        time.sleep(a.poll)
    print(f"pool end after {(time.time() - t0) / 60:.1f} min: {state}", flush=True)
    sys.exit(0 if all(s == "done" for s in state.values()) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--only", nargs="*", help="with --pool: these jobs only")
    ap.add_argument("--cores", type=int, default=12)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_47")
    a = ap.parse_args()
    if a.list:
        for k, c in JOBS.items():
            print(k, c, "spacing", 2 * c["hb"] / (c["N"] - 1), "steps", int(round(LENGTH / c["dt"])))
    elif a.pool:
        pool(a)
    elif a.job:
        run_job(a.job, a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
