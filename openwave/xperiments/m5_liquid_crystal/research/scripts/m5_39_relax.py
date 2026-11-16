"""M5.39 R2 and R3: one relaxation of Card C (or of its record ablation) per process.

The optimizer and its settings are the card's own driver's: L-BFGS-B on y with the state
start + scale * y (scale 1 on the winding unknowns, 2 on the factor), maxcor 28, maxls 35, ftol 1e-15,
gtol = 0.7 * 1e-3 * h^3, and a wall (1e100) when a simplex of the first Kuhn triangulation leaves its
open hemisphere. The energy and gradient are ours (m5_39_cardc.py, gated in m5_39_r1_gates.py).

A job is one segment up to --cap iterations and, if the force gate is not met on the stopped state,
one declared polish of up to --polish iterations with a fresh optimizer memory. The gate is read on
the final state in the original variables: the largest gradient entry over h^3, at most 1e-3.

Starts (--start): the card's `linked` or `smooth` seed; `iso` (the linked seed with an isotropic
center); `refine` (a coarser state of the same box, by the card's cubic refinement path); `pad` (a
state of a smaller box at the same spacing, set in the larger one with the card's boundary values);
`state` (a state of the same grid); and the last three with `-reset` appended, the card's frame reset
applied after the map. A stage with --handoff N writes its state at the first of its own stop or
iteration N, for the next finer stage to start from.

Labels: STATIONARY (gate met, winding one, every simplex of the four triangulations in an open
hemisphere), NOT STATIONARY (the caps reached, with the residual), INADMISSIBLE (the start or the end
inadmissible, or the wall met during the run).

Stops (--stops, added for M5.41; without the option the job runs as before). A route is one optimizer
path with a list of stop targets counted in its own iterations. The gate is read at every iterate, in
the projected reading: the larger of the winding gradient on the unit sphere (grad_p times |p|, node
by node) and the factor gradient, largest entry over h^3, at most 1e-3. The state saved for a target
is the first iterate at or after it that is under the gate; a target with no such iterate before the
next target keeps the iterate at its own count. The route ends when its last stop is saved: at the
first iterate under the gate at or after the last target, or at --cap, whose iterate is then kept.
The optimizer is never restarted at a stop. If the optimizer ends by its function tolerance, the
final iterate is the last stop; an abnormal end is restarted from the last iterate with a fresh
memory, at most --restarts times, each one recorded. --maxfun lifts the library's limit on function
evaluations, and --expect-start-energy is the start control (1e-10 relative, exit 3 when not met).
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m5_39_cardc as C  # noqa: E402

GATE = 1e-3
REPORT = 50
FRAME_SCALE = 2.0
WALL = 1e100


def power_guard():
    """Optional: a local power-loss watcher, if one is installed. The run does not depend on it."""
    try:
        sys.path.insert(0, os.path.expanduser("~/Library/Application Support/UPSSentinel"))
        import ups_guard as ug

        ug.register()
        return ug
    except Exception:  # noqa: BLE001
        return None


def build_start(a):
    """The starting state on the job's grid, and a record of where it came from."""
    kind, reset = (a.start[:-6], True) if a.start.endswith("-reset") else (a.start, False)
    info = dict(kind=a.start)
    if kind in ("linked", "smooth", "iso"):
        state, coords, h = C.seed(a.grid, a.box, kind, a.delta)
    else:
        with np.load(a.source) as data:
            src, src_coords = data["state"], data["coords"]
            info.update(source=str(a.source.name), source_grid=int(src.shape[0]),
                        source_iteration=int(data["iteration"]) if "iteration" in data else None,
                        source_energy=float(data["energy"]) if "energy" in data else None)
        coords, h = C.coordinates(a.grid, a.box)
        if kind == "refine":
            state, _, _ = C.refine(src, src_coords, a.grid, a.box)
        elif kind == "pad":
            if abs((src_coords[1, 0, 0, 0] - src_coords[0, 0, 0, 0]) - h) > 1e-12:
                raise ValueError("the box continuation requires the same spacing")
            state = C.pad(src, a.grid)
        elif kind == "state":
            if src.shape[0] != a.grid or np.max(np.abs(src_coords - coords)) > 1e-12:
                raise ValueError("a state start requires the same grid and box")
            state = src.copy()
        else:
            raise ValueError(a.start)
        # the card's resume path renormalizes the winding unknowns of the interior
        inner = C.interior(state)
        inner[..., :4] /= np.linalg.norm(inner[..., :4], axis=-1, keepdims=True)
        if reset:
            state = C.reset_frame(state, a.delta)
    info["center_eigenvalues"] = C.center_eigenvalues(state).tolist()
    return state, coords, h, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--box", type=float, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--source", type=Path)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--polish", type=int, default=2000)
    ap.add_argument("--handoff", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1.0, help="1 the card, 0 its record ablation (a a^T / G_time removed from N)")
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--kx", type=float, default=0.25)
    ap.add_argument("--ko", type=float, default=0.02)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--stop-gate", type=float, default=GATE, help="the optimizer's own stop (the card's driver: the gate); "
                    "a smaller value lets a run go on past the gate. The label is always read at the card's gate")
    ap.add_argument("--stops", default="", help="comma list of stop targets in iterations (M5.41); the last window ends at --cap")
    ap.add_argument("--maxfun", type=int, default=15000, help="the optimizer's limit on function evaluations (default: the library's)")
    ap.add_argument("--restarts", type=int, default=0, help="restarts allowed after an abnormal end of the optimizer (with --stops)")
    ap.add_argument("--expect-start-energy", type=float, help="the start control: the first evaluation must return this to 1e-10 relative")
    ap.add_argument("--prefix", default="m5_39", help="prefix of the output files (a later task reuses the driver under its own)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    a = ap.parse_args()
    targets = [int(x) for x in a.stops.split(",") if x.strip()]
    if targets and (targets != sorted(set(targets)) or a.cap <= targets[-1]):
        ap.error("--stops must be increasing and --cap must lie above the last target")

    import numba as nb

    nb.set_num_threads(a.threads)
    guard = power_guard()
    work = a.work_dir / a.name
    work.mkdir(parents=True, exist_ok=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    base, coords, h, start_info = build_start(a)
    card = C.CardC(a.grid, a.box, a.delta, a.kx, a.ko, a.lam)
    n_in = a.grid - 2
    shape_in = (n_in, n_in, n_in, 13)
    scale = np.tile([1.0] * 4 + [FRAME_SCALE] * 9, n_in ** 3)
    full = base.copy()
    st = dict(iteration=0, segment="main", evaluations=0, wall_hits=0, resumed_at=[], segments=[],
              trace_it=[], trace_e=[], trace_res=[], rows=[], handoff_iteration=None, last=None,
              trace_gate=[], stops=[], restarts=[], extra_evaluations=0, route_end=None, final_state=None)
    META_KEYS = ("evaluations", "wall_hits", "resumed_at", "segments", "rows", "handoff_iteration") + (
        ("stops", "restarts", "extra_evaluations") if targets else ())

    ckpt = work / "checkpoint.npz"
    if ckpt.exists():  # an interrupted job: continue from its last saved state, optimizer memory fresh
        with np.load(ckpt, allow_pickle=False) as data:
            full = data["state"].copy()
            st["iteration"] = int(data["iteration"])
            st["segment"] = str(data["segment"])
            st["trace_it"], st["trace_e"], st["trace_res"] = data["trace_it"].tolist(), data["trace_e"].tolist(), data["trace_res"].tolist()
            if "trace_gate" in data:
                st["trace_gate"] = data["trace_gate"].tolist()
        meta = json.loads((work / "checkpoint.json").read_text())
        for key in META_KEYS:
            st[key] = meta[key]
        start_info = meta["start"]
        st["resumed_at"].append(st["iteration"])

    def evaluate(state_full):
        card.fields(state_full)
        if card.margins(all_four=False)[0] <= 0:
            return None
        terms, grad = card(state_full, fields_done=True)
        return terms, C.interior(grad).ravel().copy()

    first = evaluate(full)
    if first is None:
        result = dict(name=a.name, label="INADMISSIBLE", reason="the start is not admissible at this spacing", start=start_info)
        (a.out_dir / f"{a.prefix}_{a.name}.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result), flush=True)
        return
    terms0, g0 = first
    deg0 = C.degree(card.q)[0]
    if not st["resumed_at"]:
        start_info.update(energy=float(terms0.sum()), terms=terms0.tolist(), residual=float(np.max(np.abs(g0)) / h ** 3), degree=int(deg0))
        # the card's own start-up check of the gradient, three random directions
        rng = np.random.default_rng(65)
        errors = []
        for _ in range(3):
            d = rng.normal(size=g0.size)
            d /= np.linalg.norm(d)
            up, dn = full.copy(), full.copy()
            C.interior(up)[...] += 2e-5 * d.reshape(shape_in)
            C.interior(dn)[...] -= 2e-5 * d.reshape(shape_in)
            fd = (card(up)[0].sum() - card(dn)[0].sum()) / 4e-5
            errors.append(abs(float(fd - d @ g0)))
        start_info["gradient_check"] = errors
        if max(errors) > 3e-7:
            raise RuntimeError(f"gradient check failed: {errors}")
        if a.expect_start_energy is not None:
            diff = abs(start_info["energy"] - a.expect_start_energy) / abs(a.expect_start_energy)
            start_info["start_control"] = dict(expected=a.expect_start_energy, relative_difference=diff, bound=1e-10, met=bool(diff <= 1e-10))
            if diff > 1e-10:
                result = dict(name=a.name, label="START CONTROL NOT MET", start=start_info)
                (a.out_dir / f"{a.prefix}_{a.name}.json").write_text(json.dumps(result, indent=1) + "\n")
                print(json.dumps(result), flush=True)
                sys.exit(3)

    stop = dict(flag=False)
    signal.signal(signal.SIGUSR1, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    def save_checkpoint(state_full):
        tmp = work / "checkpoint_tmp.npz"
        extra = dict(trace_gate=np.array(st["trace_gate"])) if targets else {}
        np.savez(tmp, state=state_full, coords=coords, iteration=st["iteration"], segment=st["segment"],
                 trace_it=np.array(st["trace_it"]), trace_e=np.array(st["trace_e"]), trace_res=np.array(st["trace_res"]), **extra)
        os.replace(tmp, ckpt)
        meta = {k: st[k] for k in META_KEYS}
        meta["start"] = start_info
        (work / "checkpoint.json").write_text(json.dumps(meta) + "\n")

    def write_handoff(state_full, energy):
        if st["handoff_iteration"] is None and a.handoff > 0:
            tmp = work / "handoff_tmp.npz"
            np.savez(tmp, state=state_full, coords=coords, iteration=st["iteration"], energy=energy)
            os.replace(tmp, work / "handoff.npz")
            st["handoff_iteration"] = st["iteration"]

    class Interrupted(Exception):
        pass

    class LastStop(Exception):
        pass

    def stop_record(target, kind, state_full, last, it):
        """One saved stop: the state in the work folder, and what was read on it."""
        name = f"stop_{target}_{kind}.npz"
        tmp = work / f"stop_{target}_{kind}_tmp.npz"
        np.savez(tmp, state=state_full, coords=coords, iteration=it, energy=float(last[1].sum()))
        os.replace(tmp, work / name)
        gate = max(last[5], last[4])
        return dict(target=target, iteration=it, kind=kind, file=name, open=False, gate_met=bool(gate <= GATE), gate=gate,
                    energy=float(last[1].sum()), terms=last[1].tolist(), residual=last[2], residual_winding=last[3],
                    residual_factor=last[4], residual_winding_unit_sphere=last[5], winding_length_min=last[6],
                    winding_length_max=last[7], center_eigenvalues=C.center_eigenvalues(state_full).tolist())

    def stop_step(it, state_full, last):
        """The stops of the route at one iterate. True when the last stop is saved."""
        gate = max(last[5], last[4])
        recs = st["stops"]
        n_open = len(recs)  # index of the next target not yet opened
        if n_open < len(targets) and it >= targets[n_open]:
            if recs and recs[-1]["open"]:
                recs[-1]["open"] = False  # its window ended with no iterate under the gate: it keeps the iterate at its count
            target = targets[n_open]
            if gate <= GATE:
                recs.append(stop_record(target, "gate", state_full, last, it))
            else:
                recs.append(stop_record(target, "count", state_full, last, it))
                recs[-1]["open"] = True
        elif recs and recs[-1]["open"] and gate <= GATE:
            count = dict(iteration=recs[-1]["iteration"], energy=recs[-1]["energy"], gate=recs[-1]["gate"])
            recs[-1] = stop_record(recs[-1]["target"], "gate", state_full, last, it)
            recs[-1]["iterate_at_the_count"] = count
        if len(recs) == len(targets):
            if not recs[-1]["open"]:
                return True
            if it >= a.cap:  # the last window ended: the iterate at the cap is kept
                count = dict(iteration=recs[-1]["iteration"], energy=recs[-1]["energy"], gate=recs[-1]["gate"])
                recs[-1] = stop_record(recs[-1]["target"], "cap", state_full, last, it)
                recs[-1]["iterate_at_the_count"] = count
                return True
        return False

    def run_segment(start_full, maxiter, tag):
        x0 = C.interior(start_full).ravel().copy()
        work_state = start_full.copy()
        t_seg, it_seg0 = time.time(), st["iteration"]

        def state_of(y):
            C.interior(work_state)[...] = (x0 + y * scale).reshape(shape_in)
            return work_state

        def fun(y):
            st["evaluations"] += 1
            res = evaluate(state_of(y))
            if res is None:
                st["wall_hits"] += 1
                return WALL, np.zeros_like(y)
            terms, grad = res
            # the length of the winding unknowns, and the winding gradient read on the unit sphere
            # (grad_p times |p| node by node); added for M5.40, the optimizer does not see them
            plen = np.linalg.norm(C.interior(work_state)[..., :4], axis=-1).reshape(-1, 1)
            st["last"] = (y.copy(), terms, float(np.max(np.abs(grad)) / h ** 3),
                          float(np.max(np.abs(grad.reshape(-1, 13)[:, :4])) / h ** 3),
                          float(np.max(np.abs(grad.reshape(-1, 13)[:, 4:])) / h ** 3),
                          float(np.max(np.abs(grad.reshape(-1, 13)[:, :4]) * plen) / h ** 3),
                          float(plen.min()), float(plen.max()))
            return float(terms.sum()), grad * scale

        def callback(y):
            st["iteration"] += 1
            it = st["iteration"]
            last = st["last"]
            known = last is not None and np.array_equal(last[0], y)
            if targets and not known:  # the gate is read at every iterate: one evaluation outside the optimizer's count
                fun(y)
                st["extra_evaluations"] += 1
                last = st["last"]
                known = last is not None and np.array_equal(last[0], y)
                if not known:
                    raise RuntimeError(f"iterate {it} could not be evaluated")
            if known:
                st["trace_it"].append(it)
                st["trace_e"].append(float(last[1].sum()))
                st["trace_res"].append(last[2])
            if a.handoff and it == a.handoff and st["handoff_iteration"] is None:
                write_handoff(state_of(y), float(last[1].sum()) if known else float("nan"))
            ended = False
            if targets:
                st["trace_gate"].append(max(last[5], last[4]))
                ended = stop_step(it, state_of(y), last)
            if it % REPORT == 0 or ended:
                if not known:
                    fun(y)
                    last = st["last"]
                eig = C.center_eigenvalues(state_of(y))
                per = (time.time() - t_seg) / max(1, it - it_seg0)
                row = dict(iteration=it, segment=tag, energy=float(last[1].sum()), residual=last[2], residual_winding=last[3],
                           residual_factor=last[4], residual_winding_unit_sphere=last[5], winding_length_min=last[6],
                           winding_length_max=last[7], center_eigenvalues=eig.tolist(), seconds_per_iteration=per)
                st["rows"].append(row)
                save_checkpoint(state_of(y))
                (work / "progress.json").write_text(json.dumps(dict(name=a.name, **row, evaluations=st["evaluations"],
                                                                    wall_time_s=time.time() - t_start, threads=a.threads)) + "\n")
                print(a.name, json.dumps(row), flush=True)
                if ended:
                    st["route_end"] = "the last stop"
                    st["final_state"] = state_of(y).copy()
                    raise LastStop()
                if stop["flag"] or (guard is not None and guard.wrap_up()):
                    raise Interrupted()

        try:
            opt = minimize(fun, np.zeros_like(x0), jac=True, method="L-BFGS-B", callback=callback,
                           options=dict(maxiter=maxiter, maxfun=a.maxfun, ftol=1e-15, gtol=a.stop_gate * h ** 3 * 0.7,
                                        maxcor=28, maxls=35))
        except LastStop:
            st["segments"].append(dict(segment=tag, iterations=st["iteration"] - it_seg0, optimizer_success=None,
                                       optimizer_message="ended by the driver at the last stop", optimizer_value=None,
                                       wall_time_s=time.time() - t_seg))
            return st["final_state"], None
        final = state_of(opt.x).copy()
        st["segments"].append(dict(segment=tag, iterations=st["iteration"] - it_seg0, optimizer_success=bool(opt.success),
                                   optimizer_message=str(opt.message), optimizer_value=float(opt.fun),
                                   optimizer_status=int(opt.status), optimizer_evaluations=int(opt.nfev),
                                   wall_time_s=time.time() - t_seg))
        return final, opt

    def gate_of(state_full):
        res = evaluate(state_full)
        if res is None:
            return None, None, float("inf")
        return res[0], res[1], float(np.max(np.abs(res[1])) / h ** 3)

    def end_without_last_stop(state_full, reason):
        """The optimizer ended by itself: the final iterate is the route's last stop."""
        res = evaluate(state_full)
        recs = st["stops"]
        if recs and recs[-1]["open"]:
            recs[-1]["open"] = False
        if res is not None:
            terms_e, grad_e = res
            gq_e = np.abs(grad_e.reshape(-1, 13))
            plen = np.linalg.norm(C.interior(state_full)[..., :4], axis=-1).reshape(-1, 1)
            last = (None, terms_e, float(gq_e.max() / h ** 3), float(gq_e[:, :4].max() / h ** 3), float(gq_e[:, 4:].max() / h ** 3),
                    float((gq_e[:, :4] * plen).max() / h ** 3), float(plen.min()), float(plen.max()))
            rec = stop_record(targets[-1], "end", state_full, last, st["iteration"])
            rec["ended_by"] = reason
            if recs and recs[-1]["target"] == targets[-1]:
                rec["iterate_at_the_count"] = dict(iteration=recs[-1]["iteration"], energy=recs[-1]["energy"], gate=recs[-1]["gate"])
                recs[-1] = rec
            else:
                recs.append(rec)
        st["route_end"] = reason

    try:
        if targets:
            while st["route_end"] is None:
                remaining = a.cap - st["iteration"]
                if remaining <= 0:
                    end_without_last_stop(full, "the cap, with no stop step at it")
                    break
                full, opt = run_segment(full, remaining, "main")
                if st["route_end"] is not None:
                    break
                if opt.status == 0:
                    end_without_last_stop(full, f"the optimizer's tolerance ({opt.message})")
                elif opt.status == 2 and len(st["restarts"]) < a.restarts:
                    st["restarts"].append(dict(iteration=st["iteration"], message=str(opt.message)))
                    save_checkpoint(full)
                    print(a.name, "RESTART after an abnormal end at iteration", st["iteration"], flush=True)
                else:
                    end_without_last_stop(full, f"the optimizer ended with no restart left ({opt.message})")
        elif st["segment"] == "main":
            remaining = a.cap - st["iteration"]
            if remaining > 0:
                full, opt = run_segment(full, remaining, "main")
            terms, grad, residual = gate_of(full)
            write_handoff(full, float(terms.sum()) if terms is not None else float("nan"))
            if residual > GATE and a.polish > 0 and terms is not None:
                st["segment"] = "polish"
                save_checkpoint(full)
        if st["segment"] == "polish":
            remaining = a.cap + a.polish - st["iteration"]
            if remaining > 0:
                full, opt = run_segment(full, remaining, "polish")
    except Interrupted:
        print(a.name, "INTERRUPTED at iteration", st["iteration"], flush=True)
        sys.exit(75)

    # the final reads
    terms, grad, residual = gate_of(full)
    admissible_end = terms is not None
    if not admissible_end:
        card.fields(full)
    m0, m4 = card.margins(all_four=True)
    deg, hits = C.degree(card.q)
    eig = C.center_eigenvalues(full)
    energy = float(terms.sum()) if admissible_end else float("nan")
    te, ti = np.array(st["trace_e"]), np.array(st["trace_it"])
    drift = None
    if len(te) > 10:
        cut = st["iteration"] - max(1, st["iteration"] // 10)
        before = te[ti <= cut]
        if len(before):
            drift = float(before[-1] - te[-1])
    under = [r["iteration"] for r in st["rows"] if r["residual"] <= GATE]
    if not admissible_end or st["wall_hits"] > 0 or m0 <= 0:
        label = "INADMISSIBLE"
    elif residual <= GATE and deg == 1 and m4 > 0:
        label = "STATIONARY"
    else:
        label = "NOT STATIONARY"
    gq = grad.reshape(-1, 13) if admissible_end else None
    plen_end = np.linalg.norm(C.interior(full)[..., :4], axis=-1).reshape(-1, 1)
    result = dict(
        name=a.name, card="C" if a.lam == 1.0 else f"C with a a^T / G_time scaled by {a.lam:g}", lam=a.lam,
        grid=a.grid, half_box=a.box, spacing=h, delta=a.delta, kx=a.kx, ko=a.ko, start=start_info,
        caps=dict(main=a.cap, polish=a.polish, handoff=a.handoff), label=label, gate=GATE, optimizer_stop_gate=a.stop_gate,
        energy=energy, terms=terms.tolist() if admissible_end else None, term_names=C.TERM_NAMES,
        composites=list(card.composites) if admissible_end else None,
        residual=residual,
        residual_winding=float(np.max(np.abs(gq[:, :4])) / h ** 3) if admissible_end else None,
        residual_factor=float(np.max(np.abs(gq[:, 4:])) / h ** 3) if admissible_end else None,
        residual_winding_unit_sphere=float(np.max(np.abs(gq[:, :4]) * plen_end) / h ** 3) if admissible_end else None,
        winding_length_range=[float(plen_end.min()), float(plen_end.max())],
        degree=int(deg), degree_preimages=int(hits), hemisphere_margin_first_triangulation=float(m0),
        hemisphere_margin_four_triangulations=float(m4), center_eigenvalues=eig.tolist(), branch=C.branch(eig),
        iterations=st["iteration"], evaluations=st["evaluations"], wall_hits=st["wall_hits"],
        energy_drift_last_tenth=drift, saved_states_under_the_gate=under, handoff_iteration=st["handoff_iteration"],
        segments=st["segments"], resumed_at=st["resumed_at"], threads=a.threads, wall_time_s=time.time() - t_start,
        history=st["rows"],
    )
    if targets:
        tg = np.array(st["trace_gate"])
        edges = targets + [a.cap + 1]
        for rec in st["stops"]:
            rec.pop("open", None)
            src = work / rec["file"]
            with np.load(src, allow_pickle=False) as data:
                if int(data["iteration"]) != rec["iteration"]:
                    raise RuntimeError(f"stop file {src} holds iteration {int(data['iteration'])}, the record {rec['iteration']}")
                out_name = f"{a.prefix}_{a.name}_s{rec['target']}.npz"
                C.save_state(a.out_dir / out_name, data["state"], data["coords"], compressed=True, iteration=rec["iteration"],
                             energy=rec["energy"], target=rec["target"])
            rec["state_file"] = out_name
        result.update(
            stop_targets=targets, stops=st["stops"], route_end=st["route_end"], restarts=st["restarts"], maxfun=a.maxfun,
            evaluations_outside_the_optimizer=st["extra_evaluations"],
            gate_reading="max(|grad_p| |p|, |grad_f|) over the interior nodes, over h^3 (unit-sphere winding gradient, factor gradient)",
            targets_not_reached=[t for t in targets if t not in [r["target"] for r in st["stops"]]],
            iterates_under_the_gate=[dict(window=[edges[i], edges[i + 1] - 1],
                                          count=int(np.sum(tg[edges[i] - 1:edges[i + 1] - 1] <= GATE)),
                                          iterates=int(len(tg[edges[i] - 1:edges[i + 1] - 1]))) for i in range(len(targets))]
            + [dict(window=[1, targets[0] - 1], count=int(np.sum(tg[:targets[0] - 1] <= GATE)), iterates=int(len(tg[:targets[0] - 1])))],
        )
    C.save_state(a.out_dir / f"{a.prefix}_{a.name}.npz", full, coords, compressed=True, iteration=st["iteration"], energy=energy,
                 trace_it=ti, trace_e=te, trace_res=np.array(st["trace_res"]),
                 **(dict(trace_gate=np.array(st["trace_gate"])) if targets else {}))
    tmp = a.out_dir / f"{a.prefix}_{a.name}.json.tmp"
    tmp.write_text(json.dumps(result, indent=1) + "\n")
    os.replace(tmp, a.out_dir / f"{a.prefix}_{a.name}.json")
    if ckpt.exists():
        ckpt.unlink()
    print(a.name, "DONE", label, f"E {energy:.6f} residual {residual:.2e} iterations {st['iteration']} center {np.round(eig, 4).tolist()}", flush=True)


if __name__ == "__main__":
    main()
