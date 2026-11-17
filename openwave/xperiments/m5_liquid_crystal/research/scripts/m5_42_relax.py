"""M5.42 R2 to R8: one chain of relaxations at fixed J (one process, one thread), on m5_42_eq42.py.

A chain is a list of paths on one model (box, spacing, delta, L1, stencil, chart, faces, band). A path is one
L-BFGS-B descent of E_J = U + J^2/2I over the six components of every free node, from the seed, from the end
of the path before it, or from the end of a path of another chain (optionally with its eigenvalues clipped
into the band). Stops are saved at 1000, 3000 and 10000 iterations (the cap) and at the optimizer's own end.

The labels of a stop (m5_42_task_details.md, section 2, row 4):

  SMOOTH     no node holds more than 1 percent of the inertia, and no node holding more than 1e-4 of it holds
             more than 50 times the mean of its 26 neighbors
  ONE-CELL   a node holds more than half of the inertia
  MIXED      neither
  CONVERGED  max |dE/dm| / h^3 <= 1e-3 at the last stop, and either the optimizer ended by its function
             tolerance, or U, I and 2 J Omega / E are within 1e-3 (relative) of the stop of the previous target;
             a path that ends before its first target has the first clause only, and its record says so

The chains (section 4): r2 (the author's box, forward stencil, deposited frame), g3 (gate 3: forward stencil,
fixed chart), r3 (the one-cell states against the spacing), r4 and r4b (the smooth branch against the box),
r5 (free faces), r6 (delta = 0.05), r8a and r8b (the band control). `--list` prints them.

A chain is restartable: finished paths are read from its JSON; an interrupted path continues from its
checkpoint with a fresh optimizer memory, recorded as a resume.
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
from scipy.optimize import minimize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_42_eq42 as Q  # noqa: E402

TARGETS = (1000, 3000, 10000)
CAP = 10000
FRACTION = dict(j0=0.0, j40=1 / 40, j20=1 / 20, j10=1 / 10, j5=1 / 5, j2p5=1 / 2.5, j1=1.0)
STEPS = ("j0", "j40", "j20", "j10", "j5", "j2p5", "j1")
BAND = dict(w=1e6, lo=-0.5, hi=1.5)


def chains():
    """Every chain of the task: name -> dict(model=..., paths=[...], stop_rule=..., needs=[(chain, path)])."""
    C = {}

    def model(N, hb, delta=0.3, L1=0.3, stencil="averaged", fixed_chart=True, free_faces=False, band=None):
        return dict(N=N, half_box=hb, delta=delta, L1=L1, stencil=stencil, fixed_chart=fixed_chart, free_faces=free_faces, band=band)

    def path(name, j, start, clip=False, only_if_not_smooth=None, plant=None):
        return dict(name=name, j=j, start=start, clip=clip, only_if_not_smooth=only_if_not_smooth, plant=plant)

    def ladder(steps, first="seed"):
        return [path(s, s, first if i == 0 else "prev") for i, s in enumerate(steps)]

    # R2: the author's box, the forward stencil, the frame as deposited
    C["r2_g33"] = dict(rung="r2", model=model(33, 6.0, stencil="forward", fixed_chart=False),
                       paths=ladder(("j0", "j10", "j1")) + [path("j10x", "j10", ("self", "j1"), only_if_not_smooth="j1"),
                                                             path("j0x", "j0", ("self", "j1"), only_if_not_smooth="j1")])
    # gate 3: the forward stencil on the fixed chart, half-box 6, three spacings
    for N in (33, 49, 65):
        C[f"g3_g{N}"] = dict(rung="g3", model=model(N, 6.0, stencil="forward"), paths=ladder(("j0", "j10", "j1")))
    # R3: the one-cell states against the spacing
    for N in (33, 49, 65):
        for L1 in (0.3, 0.1, 0.0):
            C[f"r3_g{N}_L{L1}"] = dict(rung="r3", model=model(N, 6.0, L1=L1),
                                       paths=ladder(("j0", "j10", "j1")) + [path("j10x", "j10", ("self", "j1"), only_if_not_smooth="j1"),
                                                                             path("j1x", "j1", "prev", only_if_not_smooth="j1")])
    # R4: the smooth branch against the box
    for hb, N in ((6.0, 33), (9.0, 49), (12.0, 65), (18.0, 97)):
        for L1 in (0.3, 0.1):
            C[f"r4_hb{hb:g}_L{L1}"] = dict(rung="r4", model=model(N, hb, L1=L1), paths=ladder(STEPS[:4] if hb == 18.0 else STEPS), stop_rule="smooth")
    # Added during the run. g3x: each stencil started from the J = 0 state of the other (half-box 6, L1 = 0.3), after the
    # two stencils relaxed to different J = 0 states. r3s: without L1 no one-cell state formed from the smooth start, so
    # a spike is planted at one node (eigenvalues set to -A, mid, +A along the node's own axes), the author's own test.
    for N in (33, 49):
        C[f"g3x_avg_g{N}"] = dict(rung="g3", model=model(N, 6.0), paths=[path("j0f", "j0", (f"g3_g{N}", "j0"))], needs=[(f"g3_g{N}", "j0")])
        C[f"g3x_fwd_g{N}"] = dict(rung="g3", model=model(N, 6.0, stencil="forward"), paths=[path("j0a", "j0", (f"r3_g{N}_L0.3", "j0"))], needs=[(f"r3_g{N}_L0.3", "j0")])
    for N, z in ((33, 0.375), (49, 0.25), (65, 0.375)):
        C[f"r3s_g{N}_L0.0"] = dict(rung="r3", model=model(N, 6.0, L1=0.0), needs=[(f"r3_g{N}_L0.0", "j10")],
                                    paths=[path("j10s", "j10", (f"r3_g{N}_L0.0", "j10"), plant=dict(position=(3.0, -1.5, z), A=2.0)), path("j1s", "j1", "prev")])
    # Added after the audit of the findings: the symmetry-broken state on half-box 9 (the forward stencil finds it from the
    # seed; the mirror-averaged functional is then started from it), to compare the two sectors on a second box.
    C["g3b_fwd_hb9"] = dict(rung="g3", model=model(49, 9.0, stencil="forward"), paths=ladder(("j0",)))
    C["g3b_avg_hb9"] = dict(rung="g3", model=model(49, 9.0), paths=[path("j0b", "j0", ("g3b_fwd_hb9", "j0")), path("j10b", "j10", "prev")], needs=[("g3b_fwd_hb9", "DONE")])
    # Added after the audit of the findings: is the smooth state of half-box 12 at J_e stable against a planted spike?
    C["r4s_hb12_L0.3"] = dict(rung="r4", model=model(65, 12.0, L1=0.3), needs=[("r4_hb12_L0.3", "DONE")],
                              paths=[path("j1s", "j1", ("r4_hb12_L0.3", "j1"), plant=dict(position=(-3.375, 3.0, 0.375), A=3.4))])
    # R4 extension (added during the run, after half-box 12 stayed SMOOTH up to J_e at L1 = 0.3): half-box 18 beyond J_e/10
    C["r4x_hb18_L0.3"] = dict(rung="r4", model=model(97, 18.0, L1=0.3), paths=ladder(("j5", "j2p5", "j1"), first=("r4_hb18_L0.3", "j10")),
                              stop_rule="smooth", needs=[("r4_hb18_L0.3", "DONE")])
    # R4b: without L1 (half-box 6 at both spacings is r3_g33_L0.0 and r3_g49_L0.0)
    for name, hb, N in (("r4b_hb9", 9.0, 49), ("r4b_hb12", 12.0, 65), ("r4b_hb9_h025", 9.0, 73)):
        C[name] = dict(rung="r4b", model=model(N, hb, L1=0.0), paths=ladder(("j0", "j10")))
    # R5: free faces
    C["r5_g33_L0.3"] = dict(rung="r5", model=model(33, 6.0, L1=0.3, free_faces=True), paths=ladder(STEPS))
    C["r5_g33_L0.0"] = dict(rung="r5", model=model(33, 6.0, L1=0.0, free_faces=True), paths=ladder(STEPS))
    C["r5_g49_L0.3"] = dict(rung="r5", model=model(49, 9.0, L1=0.3, free_faces=True), paths=ladder(("j0", "j10")))
    # R6: delta = 0.05
    for hb, N in ((6.0, 33), (12.0, 65)):
        for L1 in (0.3, 0.0):
            C[f"r6_hb{hb:g}_L{L1}"] = dict(rung="r6", model=model(N, hb, delta=0.05, L1=L1), paths=ladder(("j0", "j40", "j10")))
    # R8: the band control (a change of the model)
    for tag, hb, N, L1, src in (("hb6", 6.0, 33, 0.3, "r3_g33_L0.3"), ("hb9", 9.0, 49, 0.3, "r4_hb9_L0.3"), ("hb12", 12.0, 65, 0.3, "r4_hb12_L0.3"),
                                ("hb6_h025", 6.0, 49, 0.3, "r3_g49_L0.3"), ("hb6_L0", 6.0, 33, 0.0, "r3_g33_L0.0"), ("hb6_h025_L0", 6.0, 49, 0.0, "r3_g49_L0.0")):
        m = model(N, hb, L1=L1, band=BAND)
        C[f"r8a_{tag}"] = dict(rung="r8", model=m, paths=ladder(STEPS[1:], first=(src, "j0")), needs=[(src, "j0")])
        last = "j1" if src.startswith("r3") else "LAST"
        C[f"r8b_{tag}"] = dict(rung="r8", model=m, needs=[(src, "DONE")],
                               paths=[path("j10c", "j10", (src, last), clip=True, only_if_not_smooth=(src, last)), path("j1c", "j1", "prev", only_if_not_smooth=(src, last))])
    return C


def reads(mod, m6, J, with_grad=True):
    """The scalars of a state and its label inputs."""
    p = mod.evaluate(m6, J, need_grad=with_grad, need_I=True)
    idn = mod.idn
    h3 = mod.h ** 3
    I = p["I"]
    share = h3 * idn / I
    n_max = np.unravel_index(int(np.argmax(share)), share.shape)
    pad = np.pad(idn, 1)
    cnt = np.pad(np.ones_like(idn), 1)
    S = np.zeros_like(idn)
    Cn = np.zeros_like(idn)
    Nn = idn.shape[0]
    for a in range(3):
        for b in range(3):
            for c in range(3):
                S += pad[a:a + Nn, b:b + Nn, c:c + Nn]
                Cn += cnt[a:a + Nn, b:b + Nn, c:c + Nn]
    nb_mean = (S - idn) / (Cn - 1)
    big = share > 1e-4
    ratio_nb = float((idn[big] / np.maximum(nb_mean[big], 1e-300)).max()) if big.any() else 0.0
    M = Q.from6(m6.reshape(mod.shape6))
    lam = np.linalg.eigvalsh(M)
    out = dict(E=p["E"], U=p["U"], U2=p["U2"], U4=p["U4"], UV=p["UV"], UB=p["UB"], I=I, Omega=(J / I if J else 0.0),
               two_J_Omega_over_E=(2 * J * J / I / p["E"] if J else None),
               largest_share=float(share[n_max]), largest_node=[int(q) for q in n_max],
               largest_position=[float(mod.x[q]) for q in n_max], largest_eigenvalues=[float(q) for q in lam[n_max]],
               largest_over_neighbors=ratio_nb, eigenvalue_min=float(lam.min()), eigenvalue_max=float(lam.max()),
               inertia_inside_r2=float(h3 * idn[mod.r < 2].sum()))
    if with_grad:
        out["gmax_over_h3"] = float(np.abs(p["grad"]).max() / h3)
    if mod.band:
        ex = np.maximum(lam - mod.band_hi, 0) + np.maximum(mod.band_lo - lam, 0)
        near = (np.abs(lam - mod.band_hi) < 0.05) | (np.abs(lam - mod.band_lo) < 0.05)
        out.update(band_largest_excursion=float(ex.max()), band_share_of_nodes_near_an_edge=float(near.any(-1).mean()))
    s, r = out["largest_share"], ratio_nb
    out["label"] = "SMOOTH" if (s <= 0.01 and r <= 50.0) else ("ONE-CELL" if s > 0.5 else "MIXED")
    return out


class Interrupted(Exception):
    pass


def run_path(mod, spec, x0, J, work, out_dir, chain, state, deadline, guard, log):
    """One descent. state carries a resume (iteration offset, stops so far)."""
    name = spec["name"]
    stops = state.get("stops", [])
    offset = state.get("iteration", 0)
    resumed = state.get("resumed_at", [])
    restarts = state.get("restarts", [])
    nfev = state.get("nfev", 0)
    t_start = time.time() - state.get("seconds", 0.0)
    box = dict(it=offset, x=None, last_ckpt=time.time(), nfev=0, cut=None)

    def fun(z):
        p = mod.evaluate(z, J)
        box["nfev"] += 1
        return p["E"], p["grad"].ravel()

    def save_stop(z, it, kind):
        r = reads(mod, z, J)
        r.update(iteration=it, kind=kind, seconds=time.time() - t_start)
        f = out_dir / f"m5_42_{chain}_{name}_{'s%05d' % it if kind == 'target' else 'end'}.npz"
        np.savez(f, m6=z.reshape(mod.shape6), x=mod.x, J=J)
        r["file"] = f.name
        stops.append(r)
        log(f"{name} {kind} it {it} E {r['E']:.8f} U {r['U']:.6f} I {r['I']:.4f} ratio {r['two_J_Omega_over_E']} g {r['gmax_over_h3']:.2e} share {r['largest_share']:.3g} {r['label']}")
        return r

    def checkpoint(z):
        np.savez(work / f"{name}_ckpt.npz", m6=z.reshape(mod.shape6))
        (work / f"{name}_state.json").write_text(json.dumps(dict(iteration=box["it"], stops=stops, resumed_at=resumed, restarts=restarts,
                                                                  nfev=nfev + box["nfev"], seconds=time.time() - t_start)))
        box["last_ckpt"] = time.time()

    def cb(z):
        box["it"] += 1
        it = box["it"]
        if it in TARGETS:
            save_stop(z.copy(), it, "target")
            checkpoint(z)
        if it % 25 == 0:
            (work / "progress.json").write_text(json.dumps(dict(chain=chain, path=name, iteration=it, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                 seconds=time.time() - t_start)))
            if time.time() - box["last_ckpt"] > 900:
                checkpoint(z)
            if guard is not None and guard.wrap_up():
                checkpoint(z)
                raise Interrupted("power")
            if time.time() > deadline:
                box["cut"] = "wall"
                box["x"] = z.copy()
                raise StopIteration

    x = x0.copy()
    message = ""
    while True:
        left = CAP - box["it"]
        if left <= 0:
            message = "CAP"
            break
        res = minimize(fun, x, jac=True, method="L-BFGS-B", callback=cb, options=dict(maxiter=left, maxfun=20 * CAP, ftol=1e-15, gtol=0.0, maxcor=20))
        x = res.x
        message = str(res.message)
        if box["cut"]:
            message = "CUT BY THE WALL-TIME CAP"
            break
        stalled = bool(restarts) and restarts[-1]["iteration"] == box["it"]  # a restart that made no iteration
        if "ABNORMAL" in message.upper() and len(restarts) < 3 and not stalled:
            restarts.append(dict(iteration=box["it"], message=message))
            log(f"{name} restart {len(restarts)} at it {box['it']}: {message}")
            continue
        break
    it = box["it"]
    by_ftol = "REL_REDUCTION_OF_F" in message.upper() or "RELATIVE REDUCTION" in message.upper()
    if not stops or stops[-1]["iteration"] != it:
        save_stop(x, it, "end")
    last = stops[-1]
    targets = [s for s in stops if s["kind"] == "target"]
    pair = None
    if not by_ftol:
        prev = [s for s in targets if s["iteration"] < last["iteration"]]
        if prev:
            q = prev[-1]
            keys = ["U", "I"] + (["two_J_Omega_over_E"] if J else [])
            pair = dict(against_iteration=q["iteration"], drift={k: abs(last[k] - q[k]) / abs(q[k]) for k in keys})
    gate = last["gmax_over_h3"] <= 1e-3
    first_clause_only = bool(not by_ftol and pair is None and not box["cut"])  # ended before its first target
    fixed = by_ftol or (pair is not None and max(pair["drift"].values()) <= 1e-3) or first_clause_only
    converged = bool(gate and fixed)
    result = dict(name=name, J=J, j=spec["j"], start=spec["start"] if isinstance(spec["start"], str) else list(spec["start"]), clip=spec["clip"], plant=spec.get("plant"),
                  iterations=it, evaluations=nfev + box["nfev"], message=message, ended_by_function_tolerance=by_ftol,
                  ended_at_cap=bool(it >= CAP), cut_by_wall=bool(box["cut"]), restarts=restarts, resumed_at=resumed, pair=pair,
                  gate_met=bool(gate), converged=converged, only_first_clause=first_clause_only,
                  label=last["label"], seconds=time.time() - t_start, stops=stops, end_file=last["file"])
    return x, result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_42" / "chains")
    ap.add_argument("--wall-hours", type=float, default=12.0)
    a = ap.parse_args()
    C = chains()
    if a.list:
        for k, c in C.items():
            print(k, c["model"], [p["name"] for p in c["paths"]], c.get("stop_rule", ""), c.get("needs", ""))
        print(len(C), "chains,", sum(len(c["paths"]) for c in C.values()), "paths at most")
        return
    chain = a.chain
    c = C[chain]
    work = a.work_dir / chain
    work.mkdir(parents=True, exist_ok=True)
    logf = open(work / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {chain} {msg}"
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

    jf = a.work_dir / f"{chain}.json"
    rec = json.loads(jf.read_text()) if jf.exists() else dict(chain=chain, rung=c["rung"], model=c["model"], paths=[], finished=False, wall_seconds=0.0)
    if rec.get("finished"):
        log("already finished")
        return
    m = c["model"]
    mod = Q.Model(m["N"], m["half_box"], m["delta"], m["L1"], stencil=m["stencil"], fixed_chart=m["fixed_chart"], free_faces=m["free_faces"], band=m["band"])
    Je = mod.J_e
    t0 = time.time()
    deadline = t0 + a.wall_hours * 3600 - rec["wall_seconds"]
    done = {p["name"]: p for p in rec["paths"]}

    def end_of(chain_name, path_name):
        """The end field and label of a path of this or another chain."""
        if chain_name == "self":
            p = done[path_name]
        else:
            other = json.loads((a.work_dir / f"{chain_name}.json").read_text())
            p = other["paths"][-1] if path_name == "LAST" else {q["name"]: q for q in other["paths"]}.get(path_name)
        if p is None or p.get("skipped"):
            return None, None
        return np.load(a.out_dir / p["end_file"])["m6"].reshape(-1), p["label"]

    x_prev = None
    ended = None
    for spec in c["paths"]:
        name = spec["name"]
        if name in done:
            if not done[name].get("skipped"):
                x_prev = np.load(a.out_dir / done[name]["end_file"])["m6"].reshape(-1)
            continue
        cond = spec["only_if_not_smooth"]
        if cond is not None:
            src = ("self", cond) if isinstance(cond, str) else tuple(cond)
            _, lab = end_of(*src)
            if lab is None or lab == "SMOOTH":
                rec["paths"].append(dict(name=name, skipped=True, reason=f"the source {src} ended {lab}: there is no such start"))
                done[name] = rec["paths"][-1]
                jf.write_text(json.dumps(rec, indent=1) + "\n")
                log(f"{name} skipped: source {src} ended {lab}")
                continue
        J = FRACTION[spec["j"]] * Je
        state_f = work / f"{name}_state.json"
        state = {}
        if state_f.exists() and (work / f"{name}_ckpt.npz").exists():
            state = json.loads(state_f.read_text())
            x0 = np.load(work / f"{name}_ckpt.npz")["m6"].reshape(-1)
            state.setdefault("resumed_at", []).append(state["iteration"])
            log(f"{name} resumes at it {state['iteration']}")
        elif spec["start"] == "seed":
            x0 = mod.seed6().reshape(-1)
        elif spec["start"] == "prev":
            x0 = x_prev
        else:
            x0, _ = end_of(*spec["start"])
            if spec["clip"]:
                x0 = Q.clip_eigenvalues(x0.reshape(mod.shape6), BAND["lo"], BAND["hi"]).reshape(-1)
        if x0 is None:
            rec["paths"].append(dict(name=name, skipped=True, reason="no start state"))
            done[name] = rec["paths"][-1]
            continue
        if spec.get("plant"):
            node = tuple(int(np.argmin(np.abs(mod.x - q))) for q in spec["plant"]["position"])
            Mx = Q.from6(x0.reshape(mod.shape6))
            lam, vec = np.linalg.eigh(Mx[node])
            lam[0], lam[2] = -spec["plant"]["A"], spec["plant"]["A"]
            Mx[node] = (vec * lam) @ vec.T
            x0 = Q.to6(Mx).reshape(-1)
        if not mod.free_faces:  # the faces hold this model's frame
            x0 = Q.to6(np.where(mod.mask[..., None, None], Q.from6(x0.reshape(mod.shape6)), mod.seed)).reshape(-1)
        log(f"{name} starts: J {J:.4f} ({spec['j']}), start {spec['start']}")
        try:
            x_prev, result = run_path(mod, spec, x0, J, work, a.out_dir, chain, state, deadline, guard, log)
        except Interrupted:
            rec["wall_seconds"] += time.time() - t0
            jf.write_text(json.dumps(rec, indent=1) + "\n")
            log("interrupted on a power warning; checkpoint saved")
            sys.exit(75)
        rec["paths"].append(result)
        done[name] = result
        rec["wall_seconds_so_far"] = rec["wall_seconds"] + time.time() - t0
        jf.write_text(json.dumps(rec, indent=1) + "\n")
        for f in (work / f"{name}_ckpt.npz", state_f):
            f.unlink(missing_ok=True)
        log(f"{name} END it {result['iterations']} {result['message']} label {result['label']} converged {result['converged']}")
        if result["cut_by_wall"]:
            ended = f"cut by the wall-time cap in {name}"
            break
        if c.get("stop_rule") == "smooth":
            if result["label"] != "SMOOTH":
                ended = f"{name} is {result['label']}: the chain left the smooth branch"
                break
            if result["ended_at_cap"] and not result["converged"]:
                ended = f"{name} reached its cap without being CONVERGED"
                break
    rec["finished"] = True
    rec["ended"] = ended or "all paths run"
    rec["wall_seconds"] += time.time() - t0
    jf.write_text(json.dumps(rec, indent=1) + "\n")
    log(f"CHAIN END: {rec['ended']} ({rec['wall_seconds'] / 3600:.2f} h)")


if __name__ == "__main__":
    main()
