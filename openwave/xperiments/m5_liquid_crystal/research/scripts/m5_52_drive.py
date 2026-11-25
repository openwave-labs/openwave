"""M5.52: the driven wave. The prediction, the driver and the jobs.

The stepper of m5_47_evolve.py is imported and not edited. Its Field.rhs returns, off the shell, the rate
R = A(M)^-1 p / h^3 and the force f(M, p) = dT/dM - dU/dM of the action of M5.42 with the time part of M5.47.
The driver writes the classical Runge-Kutta step of RK4.step again, with the same order of accumulation, and adds
to the force of each of the four stages

    dp/dt = f(M, p) - f0 + s(x, t) - gamma p,      f0 = f(M0, 0)  (the force of the base state at rest)
    s(x, t) = a_s r(t) (sin 4t + sin 5.5t) g(x) h^3 A(M0(x)) W(x)
    r(t) = sin^2(pi t / 16) for t < 8, 1 after
    g(x) = sum_(+-) exp(-((rho - R_s)^2 + (z -+ z_s)^2) / (2 sigma_s^2)),  zero on the shell

with rho, z the cylindrical coordinates about the loop's center and normal, R_s the loop's radius (3.3 in the
vacuum jobs), z_s = 3, sigma_s = 0.5, W the mid polarization of M5.47 (six components).

    H' = T + U(M) + f0 . (M - M0),     H'(t) - H'(0) = int ( sum s . R - 2 gamma T ) dt

The step is 4 pi / 628; a job runs 3356 steps; the lock-in amplitude of window k (628 steps from j_k = 2100, 2728)
at a tone w is A_k = 2 / sum w_j  sum_j w_j exp(i w t_j) (M(t_j) - M0), w_j = sin^2(pi (j - j_k) / 628).

Usage:  python m5_52_drive.py --stage0                 (C1, C14, the prediction, C13, C2, C3, then Ga0, C8, C12)
        python m5_52_drive.py --predict                (the prediction alone; opens no lock-in amplitude)
        python m5_52_drive.py --job Ge                 (one driven job)
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import m5_42_eq42 as Q
import m5_47_evolve as E
import m5_47_waves as WV
import m5_49_predict as PR

DATA = HERE.parent / "data"
WORK = HERE.parent / "checkpoints" / "m5_52"
PRED = DATA / "m5_52_drive_predicted.json"

TONES = (4.0, 5.5)
DT = 4.0 * np.pi / 628.0
STEPS, J1, J2, NWIN = 3356, 2100, 2728, 628
GAMMA, AMP = 0.3, 1e-4
RAMP = 8.0
ZS, SIGMA_S, RS_VAC = 3.0, 0.5, 3.3
EVERY = 20
CKPT_EVERY = 200
M_MID2 = PR.M_MID2
VAC_V = dict(VD1=0.553761, VD2=0.250295, VD3=0.282248, VD4=0.473214)
UNIT_GRIDS = dict(a="B0a", b="B0b", d="B0d", e="B0e")


def _unit(grid, cap, **kw):
    d = dict(kind="unit", grid=grid, gamma=GAMMA, amp=AMP, sub=True, mult=1, steps=STEPS, j1=J1, j2=J2, cap_hours=cap)
    d.update(kw)
    return d


JOBS = dict(
    Ge=_unit("e", 7.0), Gd=_unit("d", 4.0), Gb=_unit("b", 3.0),
    GbG=_unit("b", 4.0, gamma=0.15, steps=5456, j1=4200, j2=4828),
    Ga0=_unit("a", 1.0), GaH=_unit("a", 1.0, amp=5e-5), GaS=_unit("a", 1.0, mult=2),
    GaP=_unit("a", 1.0, sub=False), GaN=_unit("a", 1.0, sub=False, amp=0.0),
    **{k: dict(kind="vacuum", grid="V2", v=v, gamma=GAMMA, amp=AMP, sub=True, mult=1, steps=STEPS, j1=J1, j2=J2, cap_hours=2.0) for k, v in VAC_V.items()},
)


def ramp(t):
    return np.sin(np.pi * t / (2 * RAMP)) ** 2 if t < RAMP else 1.0


def drive_time(t):
    """r(t) (sin 4t + sin 5.5t), without the amplitude."""
    return ramp(t) * (np.sin(TONES[0] * t) + np.sin(TONES[1] * t))


# ----------------------------------------------------------------------------- the geometry of a job

def source_profile(X, center, normal, Rs, interior):
    d = X - center
    nu = np.asarray(normal, float) / np.linalg.norm(normal)
    z = d @ nu
    rho = np.sqrt(np.maximum((d * d).sum(-1) - z * z, 0.0))
    g = sum(np.exp(-((rho - Rs) ** 2 + (z - s * ZS) ** 2) / (2 * SIGMA_S ** 2)) for s in (1.0, -1.0))
    return g * interior


_BG = {}


def background(grid, data=DATA):
    """The base state of a job's grid with everything the driver and the read need (cached in the process)."""
    if grid in _BG:
        return _BG[grid]
    if grid == "V2":
        G = PR.vacuum_background("V2")
        m0 = np.ascontiguousarray(np.broadcast_to(Q.to6(WV.vacuum_matrix()), (G["N"],) * 3 + (6,)))
        W, lam, _ = E.polarization(m0)
        G.update(m0=m0, W6=Q.to6(W), lam=lam, radius=RS_VAC, Rs=RS_VAC)
    else:
        G = PR.background(grid, data, with_A=False)
        G["Rs"] = G["radius"]
    N = G["N"]
    interior = np.zeros((N, N, N), bool)
    interior[1:-1, 1:-1, 1:-1] = True
    G["interior"] = interior
    G["g"] = source_profile(G["X"], G["center"], G["normal"], G["Rs"], interior)
    _BG[grid] = G
    return G


# ----------------------------------------------------------------------------- the driver

class Driver:
    """The Runge-Kutta step of m5_47_evolve.RK4.step, written again with the added terms, in place."""

    def __init__(self, field, f0=None, S0=None, gamma=0.0):
        self.F, self.f0, self.S0, self.gamma = field, f0, S0, gamma
        z = lambda: np.zeros(field.shape6)
        self.r, self.f, self.am, self.ap, self.mt, self.pt = z(), z(), z(), z(), z(), z()
        self.tmp = z() if gamma != 0.0 else None

    def _terms(self, f, pc, t):
        """f -= f0; f += s(t); f -= gamma p (each only where the job has it). Returns sigma(t), the factor of S0."""
        if self.f0 is not None:
            f -= self.f0
        sig = 0.0
        if self.S0 is not None:
            sig = drive_time(t)
            f += sig * self.S0
        if self.gamma != 0.0:
            np.multiply(pc, self.gamma, out=self.tmp)
            f -= self.tmp
        return sig

    def _work(self, sig):
        return sig * float(np.dot(self.S0.ravel(), self.r.ravel())) if self.S0 is not None else 0.0

    def step(self, m, p, t, dt):
        """Advances (m, p) by dt from time t. Returns the energies of the state before the step, the work of the
        source and the loss over the step (the stage weights (1, 2, 2, 1)/6)."""
        F, r, f, am, ap, mt, pt = self.F, self.r, self.f, self.am, self.ap, self.mt, self.pt
        e = F.rhs(m, p, r, f)
        sig = self._terms(f, p, t)
        work = self._work(sig)
        loss = 2.0 * self.gamma * e["T"]
        np.copyto(am, r)
        np.copyto(ap, f)
        for w, c in ((2.0, 0.5 * dt), (2.0, 0.5 * dt), (1.0, dt)):
            np.multiply(r, c, out=mt)
            mt += m
            np.multiply(f, c, out=pt)
            pt += p
            e2 = F.rhs(mt, pt, r, f)
            e["bad"] += e2["bad"]
            sig = self._terms(f, pt, t + c)
            work += w * self._work(sig)
            loss += w * 2.0 * self.gamma * e2["T"]
            np.multiply(r, w, out=mt)
            am += mt
            np.multiply(f, w, out=pt)
            ap += pt
        am *= dt / 6.0
        ap *= dt / 6.0
        m += am
        p += ap
        return e, work * dt / 6.0, loss * dt / 6.0


def lockin_coefficients(j, jk, nwin, dt, tone):
    """The weight w_j exp(i tone t_j) of step j in the window that starts at jk (0 outside the window)."""
    u = j - jk
    if u < 0 or u >= nwin:
        return 0.0, 0.0
    w = np.sin(np.pi * u / nwin) ** 2
    ph = tone * j * dt
    return w * np.cos(ph), w * np.sin(ph)


def window_norm(nwin):
    return 2.0 / float(np.sum(np.sin(np.pi * np.arange(nwin) / nwin) ** 2))


def job_setup(name, data=DATA):
    job = JOBS[name]
    G = background(job["grid"], data)
    v = job.get("v", E.V)
    F = E.Field(G["N"], G["hb"], v=v)
    assert abs(F.h - G["h"]) < 1e-12
    m0 = G["m0"]
    f0 = None
    if job["sub"]:
        f0 = np.zeros(F.shape6)
        F.rhs(m0, np.zeros(F.shape6), np.zeros(F.shape6), f0)
    S0 = None
    if job["amp"] != 0.0:
        A = F.a_matrices(m0)
        S0 = job["amp"] * F.h ** 3 * np.einsum("...ab,...b->...a", A, G["g"][..., None] * G["W6"])
        del A
        S0[~G["interior"]] = 0.0
    return job, G, F, m0, f0, S0


def run_job(name, data=DATA, work=WORK, max_steps=None, out=None):
    """One driven job. Writes into work/jobs/<name>/: lockin.npz (both windows, both tones), series.json, end.npz,
    and a restartable state every 200 steps on grids of 65^3 and above."""
    job, G, F, m0, f0, S0 = job_setup(name, data)
    mult = job["mult"]
    dt = DT / mult
    n, j1, j2, nwin = job["steps"] * mult, job["j1"] * mult, job["j2"] * mult, NWIN * mult
    every = EVERY * mult
    wd = out or (work / "jobs" / name)
    wd.mkdir(parents=True, exist_ok=True)
    guard = None
    try:
        sys.path.insert(0, os.path.expanduser("~/Library/Application Support/UPSSentinel"))
        import ups_guard

        ups_guard.register()
        guard = ups_guard
    except Exception:
        guard = None
    drv = Driver(F, f0, S0, job["gamma"])
    m = m0.copy()
    p = np.zeros(F.shape6)
    shape = (2, 2) + F.shape6
    Ar, Ai = np.zeros(shape), np.zeros(shape)
    d = np.zeros(F.shape6)
    tmp = np.zeros(F.shape6)
    series, Wk, Lk, bad, j0, spent = [], 0.0, 0.0, 0, 0, 0.0
    ck = wd / "state.npz"
    if ck.exists() and max_steps is None:
        z = np.load(ck, allow_pickle=False)
        m, p, Ar, Ai, j0 = z["m"].copy(), z["p"].copy(), z["Ar"].copy(), z["Ai"].copy(), int(z["j"])
        series, Wk, Lk, bad, spent = z["series"].tolist(), float(z["W"]), float(z["L"]), int(z["bad"]), float(z["spent"])
        print(f"{name}: resumed at step {j0}", flush=True)
    f0dot = (lambda mm: float(np.dot(f0.ravel(), (mm - m0).ravel()))) if f0 is not None else (lambda mm: 0.0)
    big = G["N"] >= 65
    t0 = time.time()
    cap = job["cap_hours"] * 3600.0
    starts = (j1, j2)

    def save_state(j):
        tmpf = wd / "state_tmp.npz"
        np.savez(tmpf, m=m, p=p, Ar=Ar, Ai=Ai, j=j, series=np.array(series), W=Wk, L=Lk, bad=bad, spent=spent + time.time() - t0)
        tmpf.replace(ck)

    def accumulate(j):
        active = [(k, jk) for k, jk in enumerate(starts) if 0 <= j - jk < nwin]
        if not active:
            return
        np.subtract(m, m0, out=d)
        for k, jk in active:
            for t_, tone in enumerate(TONES):
                cr, ci = lockin_coefficients(j, jk, nwin, dt, tone)
                np.multiply(d, cr, out=tmp)
                Ar[k, t_] += tmp
                np.multiply(d, ci, out=tmp)
                Ai[k, t_] += tmp

    def write_window(k):
        c = window_norm(nwin)
        np.savez(wd / f"lockin_w{k + 1}.npz", Ar=Ar[k] * c, Ai=Ai[k] * c, tones=np.array(TONES), dt=dt, start=starts[k], nwin=nwin)

    capped = False
    nend = n if max_steps is None else min(n, max_steps)
    j = j0
    while j < nend:
        accumulate(j)
        row = None
        if j % every == 0:
            row = [j * dt, None, None, None, f0dot(m), Wk, Lk, float(np.abs(m - m0).max())]
        e, dw, dl = drv.step(m, p, j * dt, dt)
        bad += e["bad"]
        if row is not None:
            row[1], row[2], row[3] = e["E"], e["T"], e["U"]
            row[4] = e["T"] + e["U"] + row[4]
            series.append(row)
            if not np.isfinite(e["E"]):
                raise SystemExit(f"{name}: the energy is not finite at step {j}")
        Wk += dw
        Lk += dl
        j += 1
        for k, jk in enumerate(starts):
            if j == jk + nwin:
                write_window(k)
        if j % 20 == 0:
            el = spent + time.time() - t0
            (wd / "progress.json").write_text(json.dumps(dict(job=name, step=j, of=n, E=e["E"], not_positive=bad, minutes=el / 60)) + "\n")
            if guard is not None and j % 100 == 0 and guard.wrap_up():
                save_state(j)
                print(f"{name}: power warning, state saved at step {j}", flush=True)
                sys.exit(75)
            if big and j % (CKPT_EVERY * mult) == 0 and j < n:
                save_state(j)
            if el > cap and j < n:
                save_state(j)
                capped = True
                break
    e = F.rhs(m, p, drv.r, drv.f)
    if True:
        series.append([j * dt, e["E"], e["T"], e["U"], e["T"] + e["U"] + f0dot(m), Wk, Lk, float(np.abs(m - m0).max())])
    seconds = spent + time.time() - t0
    S = np.array(series)
    dH = S[:, 4] - S[0, 4]
    resid = dH - S[:, 5] + S[:, 6]
    res = dict(job=name, spec=job, N=G["N"], half_box=G["hb"], spacing=F.h, dt=dt, steps=n, steps_done=j, windows=[j1, j2], nwin=nwin,
               tones=list(TONES), capped=capped, finished=bool(j >= n), not_positive=bad, H0=float(S[0, 4]),
               largest_dH=float(np.abs(dH).max()), largest_residual=float(np.abs(resid).max()),
               C8_ratio=float(np.abs(resid).max() / max(np.abs(dH).max(), 1e-300)), C8_passed=bool(np.abs(resid).max() <= 0.02 * np.abs(dH).max()),
               work_end=float(S[-1, 5]), loss_end=float(S[-1, 6]), largest_displacement=float(S[:, 7].max()),
               series=dict(zip(("t", "E", "T", "U", "Hprime", "work", "loss", "max_disp"), S.T.tolist())))
    if max_steps is None:
        np.savez(wd / "end.npz", m_end=m, p_end=p)
        (wd / "series.json").write_text(json.dumps(res) + "\n")
        (wd / "wall.json").write_text(json.dumps(dict(job=name, seconds=seconds, capped=capped)) + "\n")
        if res["finished"] and ck.exists():
            ck.unlink()
    print(f"{name}: steps {j} of {n}, capped {capped}, C8 ratio {res['C8_ratio']:.3e}, largest |M - M0| {res['largest_displacement']:.3e}, "
          f"not positive {bad}, minutes {seconds / 60:.1f}", flush=True)
    return res, m, p


# ----------------------------------------------------------------------------- the prediction (opens base states and the tracked prediction of M5.49 only)

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict(data=DATA, out=PRED):
    import m5_52_reads as RD

    doc49 = json.loads((data / "m5_49_bands_predicted.json").read_text())
    for g in ("b", "d", "e", "V2"):
        have = PR.block_hash(doc49["blocks"][g])
        if have != doc49["hashes"][g]:
            raise SystemExit(f"block {g} of m5_49_bands_predicted.json: hash {have} is not the recorded {doc49['hashes'][g]}")
    blocks = {}
    for g in ("b", "d", "e"):
        blocks[g] = RD.predict_grid(g, doc49["blocks"][g], background(g, data))
    blocks["V"] = RD.predict_vacuum(doc49["blocks"]["V2"], background("V2", data))
    blocks["settings"] = RD.settings()
    blocks = {k: PR.jsonable(v) for k, v in blocks.items()}
    doc = dict(task="M5.52", what="the prediction F of the driven read, its domains and the source, written before any driven job",
               m5_49_block_hashes={g: doc49["hashes"][g] for g in ("b", "d", "e", "V2")},
               hash_rule="SHA-256 of the JSON text of a block: keys sorted, separators ',' and ':', UTF-8, floats with seven significant digits",
               blocks=blocks, hashes={k: PR.block_hash(v) for k, v in blocks.items()})
    out.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
    return doc, _sha(out)


# ----------------------------------------------------------------------------- stage 0

def log(msg, work=WORK):
    with open(work / "run_log.md", "a") as fh:
        fh.write(f"- {time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def control_c1(work=WORK, data=DATA):
    """m5_47_evolve.py --controls, unchanged, its output written outside data/; leaves compared bit for bit."""
    cd = work / "c1" / "data"
    cw = work / "c1" / "work"
    cd.mkdir(parents=True, exist_ok=True)
    cw.mkdir(parents=True, exist_ok=True)
    link = cd / "m5_46_B0a_end.npz"
    if not link.exists():
        link.symlink_to(data / "m5_46_B0a_end.npz")
    rc = subprocess.call([sys.executable, "-I", str(HERE / "m5_47_evolve.py"), "--controls", "--data", str(cd), "--work", str(cw)],
                         stdout=open(work / "c1" / "stdout.txt", "w"), stderr=subprocess.STDOUT)
    new = json.loads((cd / "m5_47_controls_stepper.json").read_text())
    ref = json.loads((data / "m5_47_controls_stepper.json").read_text())

    def leaves(o, pre=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("seconds", "series"):
                    continue
                yield from leaves(v, f"{pre}/{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from leaves(v, f"{pre}/{i}")
        else:
            yield pre, o

    a, b = dict(leaves(new)), dict(leaves(ref))
    diff = [k for k in set(a) | set(b) if a.get(k, "missing") != b.get(k, "missing")]
    return dict(returncode=rc, leaves=len(b), differing=sorted(diff), passed=bool(rc == 0 and not diff and new.get("all_passed")))


def control_c2(data=DATA):
    m0, hb, N = E.load_base("B0a", data)
    F = E.Field(N, hb)
    rk = E.RK4(F)
    drv = Driver(F, None, None, 0.0)
    m1, p1 = m0.copy(), np.zeros(F.shape6)
    m2, p2 = m0.copy(), np.zeros(F.shape6)
    eq = []
    for j in range(50):
        rk.step(m1, p1, DT)
        drv.step(m2, p2, j * DT, DT)
        eq.append(bool(np.array_equal(m1, m2) and np.array_equal(p1, p2)))
    moved = float(np.abs(m2 - m0).max())
    return dict(steps=50, equal_after_every_step=all(eq), first_unequal=(eq.index(False) if not all(eq) else None), largest_move_from_B0a=moved, passed=all(eq))


def control_c3(data=DATA):
    m0, hb, N = E.load_base("B0a", data)
    F = E.Field(N, hb)
    f0 = np.zeros(F.shape6)
    F.rhs(m0, np.zeros(F.shape6), np.zeros(F.shape6), f0)
    drv = Driver(F, f0, None, GAMMA)
    m, p = m0.copy(), np.zeros(F.shape6)
    for j in range(500):
        drv.step(m, p, j * DT, DT)
    dev = float(np.abs(m - m0).max())
    return dict(steps=500, largest_deviation=dev, passed=bool(dev <= 1e-12))


def stage0(a):
    import m5_52_reads as RD

    work = WORK
    res = {}
    t0 = time.time()
    log("stage 0 started (one process): C1, C14, the prediction and its hashes, C13, C2, C3, then Ga0 with C8 and C12")
    res["C1"] = control_c1()
    log(f"C1 (try 1): {json.dumps(res['C1'])}")
    if not res["C1"]["passed"]:
        raise SystemExit("C1 not met")
    res["C14"] = RD.control_c14({g: background(g) for g in ("b", "d", "e")}, json.loads((DATA / "m5_49_bands_predicted.json").read_text()))
    log(f"C14 (try 1): passed {res['C14']['passed']}; {json.dumps(res['C14']['summary'])}")
    if not res["C14"]["passed"]:
        raise SystemExit("C14 not met")
    if PRED.exists():
        raise SystemExit("the prediction file exists already; it is not rewritten")
    doc, fh = predict()
    log(f"PREDICTION written {PRED.name}: file SHA-256 {fh}; block hashes {json.dumps(doc['hashes'], sort_keys=True)}; "
        f"M5.49 blocks matched {json.dumps(doc['m5_49_block_hashes'], sort_keys=True)}")
    (work / "prediction_hashes.json").write_text(json.dumps(dict(file=fh, blocks=doc["hashes"], written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1) + "\n")
    res["C13"] = RD.control_c13()
    log(f"C13 (try 1): passed {res['C13']['passed']}; {json.dumps(res['C13']['summary'])}")
    (work / "c13.json").write_text(json.dumps(PR.jsonable(res["C13"]), indent=1) + "\n")
    if not res["C13"]["passed"]:
        raise SystemExit("C13 not met")
    res["C2"] = control_c2()
    log(f"C2 (try 1): {json.dumps(res['C2'])}")
    if not res["C2"]["passed"]:
        raise SystemExit("C2 not met")
    res["C3"] = control_c3()
    log(f"C3 (try 1): {json.dumps(res['C3'])}")
    if not res["C3"]["passed"]:
        raise SystemExit("C3 not met")
    log("Ga0 started in the stage-0 process (the first driven job)")
    r, _, _ = run_job("Ga0")
    res["Ga0_C8"] = dict(ratio=r["C8_ratio"], passed=r["C8_passed"])
    log(f"Ga0 ended: steps {r['steps_done']} of {r['steps']}, capped {r['capped']}; C8 ratio {r['C8_ratio']:.4e} ({'met' if r['C8_passed'] else 'NOT met'})")
    res["Ga0_C12"] = RD.control_c12("Ga0")
    log(f"C12 on Ga0: passed {res['Ga0_C12']['passed']}; worst {res['Ga0_C12']['worst']:.4e}")
    (work / "stage0.json").write_text(json.dumps(PR.jsonable(res), indent=1) + "\n")
    ok = all(res[k]["passed"] for k in ("C1", "C14", "C13", "C2", "C3", "Ga0_C8", "Ga0_C12"))
    log(f"stage 0 {'MET' if ok else 'NOT MET'} after {(time.time() - t0) / 60:.1f} min")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage0", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--job")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.stage0:
        stage0(a)
    elif a.predict:
        if PRED.exists():
            raise SystemExit("the prediction file exists already; it is not rewritten")
        doc, fh = predict()
        print("file", fh, doc["hashes"])
    elif a.job:
        r, _, _ = run_job(a.job)
        sys.exit(0 if r["finished"] else 3)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
