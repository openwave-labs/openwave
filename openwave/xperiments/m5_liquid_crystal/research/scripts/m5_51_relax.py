"""M5.51: one chain (or one re-relaxation of K5) per process, single thread, on m5_51_action.py.

A chain `<grid>_<weight>_<start>` is one L-BFGS-B descent of U over the six components of the free nodes (faces held
at M_0), from

  a   M_0 plus, on each free node, six independent normal draws of standard deviation 1e-4
      (numpy.random.default_rng, seeds 51001 to 51012 in the order G1 to G6 and, within a grid, end then mid)
  b   the author's stored array (G1, G2, G4, G6), or the end of this task's chain `a` of the same weight on the
      coarser grid of the same half-box (G2 for G3, G4 for G5) interpolated by map_coordinates with order 3;
      faces reset to M_0

Run rules (plan, section 3): maxcor 10, ftol 1e-12, gtol 1e-7 (on the gradient with its factor h^3), maxfun 1.25
times the cap; cap 30000 iterations (10000 for K5); stops at 1000, 3000, 10000, 20000, 30000 and at the end, each
with every read of section 4; checkpoints every 500 iterations (the last three kept) and at each stop; an
abnormal line-search end restarted from the last iterate at most three times; a wall cap of 5 hours.

  CONVERGED  gate g = max |dU/dm| / h^3 <= 1e-3 at the end, and either the optimizer ended by its own tolerance, or
             the last two stops lie at least 5000 iterations apart and differ by at most 1e-6 relative in U and
             0.002 in R_L (a chain that ends by its tolerance before its first stop meets the first clause only)
  CAPPED     any other end

K5 (`--k5 <chain>`): the chain's end, perturbed by 1e-4 normal noise on each component of each free node (seeds
51101 to 51112 for chains a, 51201 to 51212 for chains b, in the order of the chains), relaxed again by the same
rules with a cap of 10000.

Usage:  python m5_51_relax.py --chain G1_end_a | --k5 G1_end_a | --list
"""

import argparse
import json
import os
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
import m5_51_action as A  # noqa: E402
import m5_51_reads as RD  # noqa: E402

CK = HERE.parent / "checkpoints" / "m5_51"
WORK = CK / "chains"
GRIDS = RD.GRIDS
ORDER = [(g, w) for g in ("G1", "G2", "G3", "G4", "G5", "G6") for w in ("end", "mid")]
STOPS_CHAIN = (1000, 3000, 10000, 20000, 30000)
STOPS_K5 = (1000, 3000, 10000)
WALL_HOURS = 5.0


def chain_names():
    return [f"{g}_{w}_{s}" for s in ("a", "b") for g, w in ORDER]


def parse(name):
    g, w, s = name.split("_")
    return g, w, s


def noise_seed(g, w, base):
    return base + ORDER.index((g, w)) + 1


def source_of_b(g, w):
    return dict(G3=f"G2_{w}_a", G5=f"G4_{w}_a").get(g)


def load_end(name):
    rec = json.loads((WORK / f"{name}.json").read_text())
    z = np.load(WORK / name / rec["end_file"], allow_pickle=False)
    return np.asarray(z["m6"], float), np.asarray(z["x"], float), rec


def start_state(name, mod):
    g, w, s = parse(name)
    N = mod.N
    info = {}
    if s == "a":
        sd = noise_seed(g, w, 51000)
        rng = np.random.default_rng(sd)
        m6 = mod.seed6().copy()
        m6[1:-1, 1:-1, 1:-1] += 1e-4 * rng.standard_normal((N - 2, N - 2, N - 2, 6))
        info = dict(start_state="seed with noise", noise_seed=sd)
    elif g in A.STORED:
        m6, x, _ = A.load_stored(g)
        assert np.allclose(x, mod.x)
        m6 = mod.hold_faces(m6)
        info = dict(start_state="the author's stored array", file=A.STORED[g][0])
    else:
        src = source_of_b(g, w)
        m6s, xs, _ = load_end(src)
        hs = xs[1] - xs[0]
        I = np.meshgrid(*([(mod.x - xs[0]) / hs] * 3), indexing="ij")
        m6 = np.stack([map_coordinates(m6s[..., k], I, order=3) for k in range(6)], -1)
        dev = float(np.abs(m6 - mod.seed6())[~mod.mask].max())
        m6 = mod.hold_faces(m6)
        info = dict(start_state=f"end of {src}, interpolated with order 3", source=src, face_deviation_before_reset=dev)
    return m6, info


class Stop(Exception):
    pass


def run(name, m6_0, mod, cap, stops, work, rec, log, guard, deadline):
    """One descent with its stops, checkpoints, restarts and resume."""
    N = mod.N
    h3 = mod.h ** 3
    state_f = work / "state.json"
    st = json.loads(state_f.read_text()) if state_f.exists() else dict(iteration=0, nfev=0, stops=[], restarts=[], resumes=[], ckpts=[])
    x = m6_0.reshape(-1).copy()
    if st["iteration"] > 0:
        last = st["ckpts"][-1]
        x = np.load(work / last["file"], allow_pickle=False)["m6"].reshape(-1)
        st["resumes"].append(st["iteration"])
        st["nfev"] += st.pop("nfev_this_process", 0)
        log(f"resumes at it {st['iteration']} from {last['file']}")
    box = dict(it=st["iteration"], nfev=0, cut=None)
    free = np.broadcast_to(mod.mask[..., None], mod.shape6).reshape(-1)

    def fun(z):
        p = mod.evaluate(z)
        box["nfev"] += 1
        return p["U"], p["grad"].reshape(-1)

    def save(z, it, kind):
        f = f"{'s%05d' % it if kind == 'stop' else ('end' if kind == 'end' else 'ckpt_%05d' % it)}.npz"
        np.savez(work / f, m6=z.reshape(mod.shape6), x=mod.x)
        return f

    def checkpoint(z, it, kind="ckpt"):
        f = save(z, it, kind)
        if kind == "ckpt":
            st["ckpts"].append(dict(iteration=it, file=f))
            while len([c for c in st["ckpts"] if c["file"].startswith("ckpt_")]) > 3:
                old = [c for c in st["ckpts"] if c["file"].startswith("ckpt_")][0]
                (work / old["file"]).unlink(missing_ok=True)
                st["ckpts"].remove(old)
        else:
            st["ckpts"].append(dict(iteration=it, file=f))
        st["iteration"] = it
        st["nfev"] = st.get("nfev", 0)
        state_f.write_text(json.dumps(dict(st, nfev_this_process=box["nfev"]), default=RD._jsonable))

    def stop_read(z, it, kind):
        f = save(z, it, "stop" if kind == "stop" else "end")
        r = RD.read_state(z.reshape(mod.shape6), mod.x, mod.half_box, mod.weight)
        r.update(iteration=it, kind=kind, file=f)
        st["stops"].append(r)
        log(f"{kind} it {it} U {r['U']:.10f} gate {r['gate']:.3e} RL {r.get('RL')} cls {r.get('cls')} label {r['label']} one_ring {r.get('one_ring')} deg {r['degree']['values']} N/U {r['N_over_U']:.5f}")
        return r

    def cb(z):
        box["it"] += 1
        it = box["it"]
        if it in stops:
            stop_read(z.copy(), it, "stop")
            st["ckpts"].append(dict(iteration=it, file="s%05d.npz" % it))
        if it % 500 == 0:
            checkpoint(z.copy(), it)
            (work / "progress.json").write_text(json.dumps(dict(chain=name, iteration=it, time=time.strftime("%Y-%m-%d %H:%M:%S"))))
            if guard is not None:
                try:
                    if guard.wrap_up():
                        box["cut"] = "power"
                        raise StopIteration
                except StopIteration:
                    raise
                except Exception:  # noqa: BLE001
                    pass
        if it % 25 == 0 and time.time() > deadline:
            box["cut"] = "wall"
            box["x"] = z.copy()
            raise StopIteration

    message = ""
    by_tol = False
    while True:
        left = cap - box["it"]
        used = st["nfev"] + box["nfev"]
        if left <= 0:
            message = "CAP"
            break
        maxfun = int(1.25 * cap) - used
        if maxfun <= 0:
            message = "MAXFUN"
            break
        res = minimize(fun, x, jac=True, method="L-BFGS-B", callback=cb,
                       options=dict(maxiter=left, maxfun=maxfun, ftol=1e-12, gtol=1e-7, maxcor=10))
        x = res.x
        message = str(res.message)
        if box["cut"] == "power":
            checkpoint(x, box["it"])
            st["nfev"] += box["nfev"]
            state_f.write_text(json.dumps(st, default=RD._jsonable))
            log("power warning: checkpoint saved, exit 75")
            sys.exit(75)
        if box["cut"] == "wall":
            x = box["x"]
            message = "CUT BY THE WALL-TIME CAP"
            break
        by_tol = message.upper().startswith("CONVERGENCE")
        stalled = bool(st["restarts"]) and st["restarts"][-1]["iteration"] == box["it"]
        if "ABNORMAL" in message.upper() and len(st["restarts"]) < 3 and not stalled:
            st["restarts"].append(dict(iteration=box["it"], message=message))
            log(f"restart {len(st['restarts'])} at it {box['it']}: {message}")
            continue
        break
    it = box["it"]
    st["nfev"] += box["nfev"]
    if not st["stops"] or st["stops"][-1]["iteration"] != it:
        end = stop_read(x, it, "end")
        end_file = end["file"]
    else:
        end = st["stops"][-1]
        end_file = end["file"]
        np.savez(work / "end.npz", m6=x.reshape(mod.shape6), x=mod.x)
        end_file = "end.npz"
    # the earlier state: the last saved state at least 500 iterations before the end, else the start
    cands = [c for c in st["ckpts"] if c["iteration"] <= it - 500 and (work / c["file"]).exists()]
    if cands:
        c = max(cands, key=lambda q: q["iteration"])
        ze = np.load(work / c["file"], allow_pickle=False)["m6"]
        earlier_it, earlier_kind = c["iteration"], "saved state"
    else:
        ze = m6_0.reshape(mod.shape6)
        earlier_it, earlier_kind = 0, "the start state (no saved state lies 500 iterations or more before the end)"
    er = RD.read_state(ze, mod.x, mod.half_box, mod.weight)
    er.update(iteration=earlier_it, kind=earlier_kind)
    stop_list = st["stops"]
    gate_ok = end["gate"] <= 1e-3
    pair = None
    if len(stop_list) >= 2:
        p, q = stop_list[-2], stop_list[-1]
        if p["iteration"] != q["iteration"]:
            pair = dict(iterations=[p["iteration"], q["iteration"]], apart=q["iteration"] - p["iteration"], dU_rel=abs(q["U"] / p["U"] - 1),
                        dRL=(abs(q["RL"] - p["RL"]) if (q.get("RL") is not None and p.get("RL") is not None) else None))
    pair_ok = bool(pair and pair["apart"] >= 5000 and pair["dU_rel"] <= 1e-6 and pair["dRL"] is not None and pair["dRL"] <= 0.002)
    first_clause_only = bool(by_tol and it < min(stops))
    converged = bool(gate_ok and (by_tol or pair_ok))
    rec.update(iterations=it, evaluations=st["nfev"], message=message, ended_by_tolerance=by_tol, cut_by_wall=bool(box["cut"] == "wall"),
               ended_at_cap=bool(it >= cap), restarts=st["restarts"], resumes=st["resumes"], pair=pair, gate_met=bool(gate_ok),
               converged=converged, end_state="CONVERGED" if converged else "CAPPED", only_first_clause=first_clause_only,
               stops=stop_list, end=end, end_file=end_file, earlier=er)
    return rec


def post(src):
    """After the audit (a descriptor, no word): continue the end of a chain with the relative-reduction tolerance
    switched off (ftol 0), the other options unchanged, a cap of 30000 iterations from the continuation's start, a
    wall cap of 40 minutes, a saved state with its reads every 1000 iterations and at the end."""
    name = f"post_{src}"
    work = WORK / name
    work.mkdir(parents=True, exist_ok=True)
    jf = WORK / f"{name}.json"
    if jf.exists():
        print("exists; never overwritten")
        return
    g, w, s = parse(src)
    hb, h, N = GRIDS[g]
    mod = A.Model(N, hb, weight=w)
    m6e, _, srec = load_end(src)
    cap = 30000
    deadline = time.time() + 40 * 60
    logf = open(work / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {name} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    keys = ("U", "gate", "RL", "cls", "angle_axis", "angle_diag", "normal", "line_regime", "halfturn_all", "one_ring", "label")

    def read(z, it, kind):
        np.savez(work / (f"s{it:05d}.npz" if kind == "stop" else "end.npz"), m6=z.reshape(mod.shape6), x=mod.x)
        r = RD.read_state(z.reshape(mod.shape6), mod.x, mod.half_box, mod.weight)
        q = {k: r.get(k) for k in keys}
        q.update(iteration=it, kind=kind)
        log(f"{kind} it {it} U {q['U']:.10f} gate {q['gate']:.2e} RL {q['RL']} cls {q['cls']} angles {q['angle_axis']:.3f} {q['angle_diag']:.3f} regime {q['line_regime']} halfturn {q['halfturn_all']}")
        return q

    rows = [read(m6e.reshape(-1), 0, "stop")]
    box = dict(it=0, cut=None, nfev=0)

    def fun(z):
        p = mod.evaluate(z)
        box["nfev"] += 1
        return p["U"], p["grad"].reshape(-1)

    def cb(z):
        box["it"] += 1
        if box["it"] % 1000 == 0:
            rows.append(read(z.copy(), box["it"], "stop"))
        if box["it"] % 25 == 0 and time.time() > deadline:
            box["cut"] = "wall"
            box["x"] = z.copy()
            raise StopIteration

    log(f"starts from the end of {src} (iteration {srec['iterations']}), ftol 0, cap {cap}, wall 40 min")
    res = minimize(fun, m6e.reshape(-1), jac=True, method="L-BFGS-B", callback=cb,
                   options=dict(maxiter=cap, maxfun=int(1.25 * cap), ftol=0.0, gtol=1e-7, maxcor=10))
    x = box["x"] if box["cut"] else res.x
    msg = "CUT BY THE 40-MINUTE WALL CAP" if box["cut"] else str(res.message)
    if rows[-1]["iteration"] != box["it"]:
        rows.append(read(x, box["it"], "end"))
    rec = dict(name=name, source=src, grid=g, weight=w, start=s, iterations=box["it"], evaluations=box["nfev"], message=msg,
               ftol=0.0, gtol=1e-7, maxcor=10, cap=cap, reads=rows, finished=True)
    jf.write_text(json.dumps(rec, indent=1, default=RD._jsonable) + "\n")
    log(f"END it {box['it']} {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain")
    ap.add_argument("--k5")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--post", help="after the audit: continue the end of a G2 chain with ftol 0 (a descriptor)")
    a = ap.parse_args()
    if a.post:
        post(a.post)
        return
    if a.list:
        for n in chain_names():
            print(n)
        return
    name = a.chain if a.chain else f"k5_{a.k5}"
    WORK.mkdir(parents=True, exist_ok=True)
    work = WORK / name
    work.mkdir(exist_ok=True)
    jf = WORK / f"{name}.json"
    if jf.exists() and json.loads(jf.read_text()).get("finished"):
        print("already finished")
        return
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
    t0 = time.time()
    prev = json.loads((work / "wall.json").read_text())["seconds"] if (work / "wall.json").exists() else 0.0
    deadline = t0 + WALL_HOURS * 3600 - prev
    if a.chain:
        g, w, s = parse(name)
        hb, h, N = GRIDS[g]
        mod = A.Model(N, hb, weight=w)
        m6_0, info = start_state(name, mod)
        cap, stops = 30000, STOPS_CHAIN
        rec = dict(name=name, grid=g, weight=w, start=s, half_box=hb, h=h, N=N, cap=cap, **info)
    else:
        src = a.k5
        g, w, s = parse(src)
        hb, h, N = GRIDS[g]
        mod = A.Model(N, hb, weight=w)
        m6e, _, srec = load_end(src)
        sd = noise_seed(g, w, 51100 if s == "a" else 51200)
        rng = np.random.default_rng(sd)
        m6_0 = m6e.copy()
        m6_0[1:-1, 1:-1, 1:-1] += 1e-4 * rng.standard_normal((N - 2, N - 2, N - 2, 6))
        cap, stops = 10000, STOPS_K5
        rec = dict(name=name, k5_of=src, grid=g, weight=w, start=f"end of {src} with 1e-4 noise", noise_seed=sd, half_box=hb, h=h, N=N, cap=cap)
    rec["finished"] = False
    log(f"starts: {json.dumps({k: v for k, v in rec.items() if k not in ('finished',)}, default=RD._jsonable)}")
    U0 = mod.evaluate(m6_0, False)["U"]
    rec["U_start"] = U0
    try:
        rec = run(name, m6_0, mod, cap, stops, work, rec, log, guard, deadline)
    finally:
        (work / "wall.json").write_text(json.dumps(dict(seconds=prev + time.time() - t0)))
    if a.k5:
        e = rec["end"]
        src_end = srec["end"]
        dU = abs(e["U"] / src_end["U"] - 1)
        dR = abs(e["RL"] - src_end["RL"]) if (e.get("RL") is not None and src_end.get("RL") is not None) else None
        same_cls = e.get("cls") == src_end.get("cls")
        same_deg = e["degree"]["values"] == src_end["degree"]["values"]
        ok = bool(dU <= 1e-6 and dR is not None and dR <= 0.002 and same_cls and same_deg)
        rec["k5"] = dict(dU_rel=dU, dRL=dR, same_class=same_cls, same_degree=same_deg, passed=ok, U_chain=src_end["U"], U_k5=e["U"])
        log(f"K5 {'PASS' if ok else 'FAIL'} dU_rel {dU:.3e} dRL {dR} class {same_cls} degree {same_deg}")
    rec["finished"] = True
    jf.write_text(json.dumps(rec, indent=1, default=RD._jsonable) + "\n")
    log(f"END it {rec['iterations']} {rec['message']} {rec['end_state']} gate {rec['end']['gate']:.3e} U {rec['end']['U']:.10f}")


if __name__ == "__main__":
    main()
