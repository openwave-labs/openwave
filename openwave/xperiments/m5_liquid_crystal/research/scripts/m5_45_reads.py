"""M5.45: the reads on the end states of m5_45_relax.py, the tables of the task document and its figures.

For every chain record (checkpoints/m5_45/chains/<chain>.json) and its end state (data/m5_45_<chain>_*.npz):

  the scalars     U and its terms, I, the inertia by shells (r < 2, 2 to 4, 4 to the half-box, beyond the
                  inscribed sphere), from the evaluator of M5.42 on the end state
  the set W       the smallest set of nodes that holds 95 percent of the inertia (nodes ranked by the inertia
                  density of the functional), its share of the nodes and its volume
  q               per node, with the one-sided differences d_i^s M of the functional's stencil and the mean over
                  the orientations s that exist at the node:
                      C = < sum_{i<j} |[d_i M, d_j M]|^2 >_s,   G = < sum_i |d_i M|^2 >_s,   q = C / G^2
                  reported as the mean weighted by the inertia density, on W and on the rest
  two controls    h^3 sum_n sum_s w_s C_s against U4 / kappa of the evaluator, and L1 h^3 sum_n sum_s w_s G_s
                  against its U2 (the difference code of this script against the compiled kernel)
  h * grad2       the spacing times h^3 sum_n sum_s w_s G_s
  the loop        the reader of M5.44 (m5_44_loops.read_state)

and, from the tracked reads of M5.42 and M5.44, the inertia by box of the J = 0 ladders at L1 = 0.1 and 0.3.

The chains added during the run have their own tables: a kicked end state relaxed again, and a state carried from
another grid (or from the state without L1 of the same grid) and relaxed. Every state found on a grid is then
listed lowest U first.

Writes data/m5_45_summary.json (tracked) and the two figures, and prints the tables in markdown.

Usage:  python m5_45_reads.py [--work-dir DIR] [--out-dir DIR] [--plots-dir DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_42_eq42 as Q  # noqa: E402
import m5_44_loops as LP  # noqa: E402
from m5_45_relax import CHAINS, DELTA, HALF_BOX  # noqa: E402

IT_REF = 8364  # M5.42, chain r6_hb6_L0.0, path j0: the iteration of its end (m5_42_task_details.md, Table 9 of M5.44); U and I are read from m5_42_r7_reads.json
AUTHOR = {(0.5, 0.0): 1233.0, (0.375, 0.0): 2120.0, (0.25, 0.0): 3460.0, (0.5, 0.03): 119.0, (0.375, 0.03): 119.2}


def comm2(A, B):
    C = A @ B - B @ A
    return (C ** 2).sum((-2, -1))


def stencil_sums(M, h):
    """Per node: the sums over the orientations that exist of C_s and G_s, and their number."""
    N = M.shape[0]
    Cs, Gs, cnt = np.zeros((N, N, N)), np.zeros((N, N, N)), np.zeros((N, N, N))
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                s = (sx, sy, sz)
                here = tuple(slice(0, N - 1) if q > 0 else slice(1, N) for q in s)
                D = []
                for a in range(3):
                    there = list(here)
                    there[a] = slice(1, N) if s[a] > 0 else slice(0, N - 1)
                    D.append((M[tuple(there)] - M[here]) / h)
                Cs[here] += comm2(D[0], D[1]) + comm2(D[0], D[2]) + comm2(D[1], D[2])
                Gs[here] += sum((d ** 2).sum((-2, -1)) for d in D)
                cnt[here] += 1
    return Cs, Gs, cnt


def read(rec, out_dir):
    m = rec["model"]
    z = np.load(out_dir / rec["end_file"], allow_pickle=False)
    m6, x = z["m6"], z["x"]
    mod = Q.Model(m["N"], m["half_box"], m["delta"], m["L1"], stencil="averaged", fixed_chart=True)
    p = mod.evaluate(m6.reshape(-1), 0.0, need_grad=True, need_I=True)
    h, h3 = mod.h, mod.h ** 3
    dens = h3 * mod.idn.copy()
    I = p["I"]
    r = mod.r
    shells = dict(inside_r2=float(dens[r < 2].sum()), r2_to_4=float(dens[(r >= 2) & (r < 4)].sum()),
                  r4_to_half_box=float(dens[(r >= 4) & (r < m["half_box"])].sum()), beyond_inscribed_sphere=float(dens[r >= m["half_box"]].sum()))
    order = np.argsort(dens.ravel())[::-1]
    cum = np.cumsum(dens.ravel()[order])
    sets = {}
    for f in (0.5, 0.9, 0.95):
        k = int(np.searchsorted(cum, f * I) + 1)
        sets[f] = k
    inW = np.zeros(dens.size, bool)
    inW[order[:sets[0.95]]] = True
    inW = inW.reshape(dens.shape)
    M = Q.from6(m6)
    Cs, Gs, cnt = stencil_sums(M, h)
    C, G = Cs / cnt, Gs / cnt
    q = np.where(G > 0, C / np.maximum(G, 1e-300) ** 2, 0.0)
    u4_check = h3 * Cs.sum() / 8.0  # times kappa = 1
    g2 = h3 * Gs.sum() / 8.0

    def wmean(mask):
        w = dens[mask]
        return float((q[mask] * w).sum() / w.sum()) if w.sum() > 0 else None

    live = G > 0
    other = dict(unweighted=float(q[~inW & live].mean() / q[inW].mean()), pooled=float((C[~inW].sum() / (G[~inW] ** 2).sum()) / (C[inW].sum() / (G[inW] ** 2).sum())))

    lp = LP.read_state(m6, x, second=False)
    lp.pop("_points", None)
    out = dict(chain=rec["chain"], N=m["N"], spacing=m["spacing"], L1=m["L1"], file=rec["end_file"],
               U=p["U"], U2=p["U2"], U4=p["U4"], UV=p["UV"], I=I, gmax_over_h3=float(np.abs(p["grad"]).max() / h3), shells=shells,
               nodes=int(dens.size), nodes_for_half=sets[0.5], nodes_for_90=sets[0.9], nodes_for_95=sets[0.95],
               W_share_of_nodes=sets[0.95] / dens.size, W_volume=sets[0.95] * h3, volume_for_half=sets[0.5] * h3, q_on_W=wmean(inW), q_on_the_rest=wmean(~inW),
               q_ratio_rest_over_W=None, q_ratio_other_weightings=other, h_times_grad2=h * g2, grad2=g2,
               control_U4=dict(this_script=u4_check, evaluator=p["U4"], relative=abs(u4_check - p["U4"]) / max(abs(p["U4"]), 1e-300)),
               control_U2=None if m["L1"] == 0 else dict(this_script=m["L1"] * g2, evaluator=p["U2"], relative=abs(m["L1"] * g2 - p["U2"]) / abs(p["U2"])),
               loops=lp["loops"], fragments=lp["fragments"], pierced=lp["pierced"], loop=lp["loop"])
    if out["q_on_W"] and out["q_on_the_rest"]:
        out["q_ratio_rest_over_W"] = out["q_on_the_rest"] / out["q_on_W"]
    mid = m["N"] // 2
    return out, dict(x=x, dens_z0=mod.idn[:, :, mid].copy(), inW_z0=inW[:, :, mid].copy())


def thin(traj, n=300):
    if len(traj) <= n:
        return traj
    idx = np.unique(np.round(np.linspace(0, len(traj) - 1, n)).astype(int))
    return [traj[i] for i in idx]


def ending(rec):
    """How a chain ended, in the words of the tables, and whether the tables call it CONVERGED.

    The events of the record are listed in order: every end by the function tolerance and every stop of the line
    search (L-BFGS-B's message ABNORMAL: the line search found no lower point). The driver's own record keeps its
    words. A chain with a stalled line search in its record is reported as CONVERGED on the gradient gate alone
    (deviation 1); a chain without one keeps the driver's word."""
    msg = rec["message"].upper()
    ev = [(f["iteration"], "function tolerance") for f in rec["function_tolerance_ends"]] + [(q["iteration"], "a stalled line search") for q in rec["abnormal"]]
    if "ABNORMAL" in msg:
        ev.append((rec["iterations"], "a stalled line search"))
    ev = sorted(set(ev), key=lambda t: (t[0], t[1] != "function tolerance"))
    stalled = any(k != "function tolerance" for _, k in ev)
    how = ", then ".join("%s at %d" % (k, it) for it, k in ev)
    if rec["cut_by_wall"]:
        return (how + ", then " if how else "") + "cut by the wall-time cap at %d" % rec["iterations"], False, stalled
    if rec["ended_at_cap"]:
        return (how + ", then " if how else "") + "the cap of iterations", bool(rec["converged"]), stalled
    if stalled:
        return how + " (the driver's record: converged %s)" % rec["converged"], bool(rec["gate_met"]), True
    return how, bool(rec["converged"]), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_45" / "chains")
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--plots-dir", type=Path, default=HERE.parent / "plots")
    a = ap.parse_args()
    R42 = {(t["chain"], t["path"]): t for t in json.load(open(a.out_dir / "m5_42_r7_reads.json"))["states"]}
    U_REF, I_REF = R42[("r6_hb6_L0.0", "j0")]["U"], R42[("r6_hb6_L0.0", "j0")]["I"]
    WINDOW = (0.97 * U_REF, 1.03 * U_REF)
    states, slices, chains, kicks = [], {}, {}, []
    for name in CHAINS:
        jf = a.work_dir / f"{name}.json"
        if not jf.exists():
            print(f"(no record yet for {name})", file=sys.stderr)
            continue
        rec = json.loads(jf.read_text())
        how, conv, stalled = ending(rec)
        if CHAINS[name].get("start"):  # a kicked end state relaxed again (deviation 2): its own table
            k, last = rec["kick"], rec["stops"][-1]
            kicks.append(dict(chain=name, source=k["source"], spacing=rec["model"]["spacing"], amplitude=k["amplitude"], seed=k["seed"], U_before=k["U_before"], I_before=k["I_before"],
                              U_after_the_kick=k["U_after_the_kick"], I_after_the_kick=k["I_after_the_kick"], U=last["U"], I=last["I"], inertia_inside_r2=last["inertia_inside_r2"],
                              gmax_over_h3=last["gmax_over_h3"], iterations=rec["iterations"], how_it_ended=how, converged=conv, stalled_line_search=stalled, word=rec["word"],
                              U_relative=(last["U"] - k["U_before"]) / k["U_before"], I_relative=(last["I"] - k["I_before"]) / k["I_before"],
                              stops=[{q: t[q] for q in ("iteration", "kind", "U", "I")} for t in rec["stops"]]))
            continue
        st, sl = read(rec, a.out_dir)
        st.update(iterations=rec["iterations"], evaluations=rec["evaluations"], message=rec["message"], converged=conv, converged_by_the_driver=rec["converged"], how_it_ended=how, stalled_line_search=stalled,
                  gate_met=rec["gate_met"], cut_by_wall=rec["cut_by_wall"], ended_at_cap=rec["ended_at_cap"], word=rec["word"],
                  drift_of_I_over_the_last_tenth=rec["drift_of_I_over_the_last_tenth"], label=rec["label"], function_tolerance_ends=rec["function_tolerance_ends"],
                  abnormal=rec["abnormal"], resumed_at=rec["resumed_at"], start="seed",
                  stops=[{k: s[k] for k in ("iteration", "kind", "U", "U2", "U4", "UV", "I", "inertia_inside_r2", "gmax_over_h3", "largest_share", "label", "file")} for s in rec["stops"]],
                  trajectory=thin(rec["trajectory"]))
        if rec.get("interp"):  # a state carried from another grid or another L1 (deviation 3)
            k = rec["interp"]
            src = k["source_file"].replace("m5_45_", "").split("_end")[0].split("_s0")[0]
            same = abs(k["source_spacing"] - rec["model"]["spacing"]) < 1e-9
            st.update(start=("the state without L₁ of this grid" if same else "carried from spacing %g" % k["source_spacing"]) + (" (its stop at iteration 10000)" if "_s010000" in k["source_file"] else ""),
                      source_chain=src, source_file=k["source_file"], source_spacing=k["source_spacing"], U_at_the_start=k["U_at_the_start"], I_at_the_start=k["I_at_the_start"])
        states.append(st)
        slices[name] = sl
        chains[name] = rec

    seeds = [s for s in states if s["start"] == "seed"]
    crosses = [s for s in states if s["start"] != "seed"]
    by = {(s["spacing"], s["L1"]): s for s in seeds}
    for t in crosses:
        ref = by.get((t["spacing"], t["L1"]))
        if ref:
            t.update(U_minus_the_seed_state=t["U"] - ref["U"], I_over_the_seed_state=t["I"] / ref["I"])
    # the control of section 3: the chain at spacing 0.375 without L1 at its first end by the function tolerance
    control = None
    c = chains.get("h0375_L0")
    if c and c["function_tolerance_ends"]:
        f = c["function_tolerance_ends"][0]
        control = dict(iteration=f["iteration"], U=f["U"], I=f["I"], reference=dict(iteration=IT_REF, U=U_REF, I=I_REF),
                       U_relative=abs(f["U"] - U_REF) / U_REF, I_relative=abs(f["I"] - I_REF) / I_REF)
        control["met"] = bool(f["iteration"] == IT_REF and control["U_relative"] < 1e-10 and control["I_relative"] < 1e-10)

    def exponents(pick, tag):
        out = []
        for L1 in (0.0, 0.03):
            hs = sorted((h for (h, l) in pick if l == L1), reverse=True)
            for h1, h2 in zip(hs, hs[1:]):
                s1, s2 = pick[(h1, L1)], pick[(h2, L1)]
                out.append(dict(states=tag, L1=L1, coarse=h1, fine=h2, I_coarse=s1["I"], I_fine=s2["I"], ratio_of_I=s2["I"] / s1["I"],
                                exponent=float(np.log(s2["I"] / s1["I"]) / np.log(h2 / h1)), a_bound=bool(not s1["converged"] or not s2["converged"] or "RISING" in (s1["word"], s2["word"]))))
        return out

    lowest = {}
    for s in states:
        k = (s["spacing"], s["L1"])
        if k not in lowest or s["U"] < lowest[k]["U"]:
            lowest[k] = s
    expo = exponents(by, "from the seed") + exponents(lowest, "the lowest U found on each grid")
    # the author's prediction, on every state found at spacing 0.25 without L1
    scores = []
    for s25 in sorted((s for s in states if s["spacing"] == 0.25 and s["L1"] == 0.0), key=lambda s: s["start"] != "seed"):
        inside = WINDOW[0] <= s25["U"] <= WINDOW[1]
        steady = s25["word"] == "STEADY" or s25["converged"]
        word = "HELD" if (s25["I"] > 3000 and inside) else ("FAILED" if (not inside or (s25["I"] <= 3000 and steady)) else "OPEN")
        scores.append(dict(chain=s25["chain"], start=s25["start"], I=s25["I"], U=s25["U"], U_window=list(WINDOW), U_over_reference=s25["U"] / U_REF, word=word,
                           state_word=s25["word"], converged=s25["converged"]))
    # the inertia by box, from the tracked reads of M5.42 and M5.44
    L44 = json.load(open(a.out_dir / "m5_44_loops.json"))["states"]
    ladder = []
    for s in L44:
        if s["chain"].startswith("r4_hb") and s["path"] == "j0":
            r = R42[(s["chain"], s["path"])]
            ladder.append(dict(chain=s["chain"], L1=s["L1"], half_box=s["half_box"], N=s["N"], converged=s["converged"], label=s["label"], U=r["U"], I=r["I"],
                               **r["inertia"], loop_radius=s["loop"]["radius"]))
    ladder.sort(key=lambda t: (t["L1"], t["half_box"]))
    out = dict(task="M5.45", model=dict(half_box=HALF_BOX, delta=DELTA, J=0.0, stencil="averaged", fixed_chart=True), states=states, control=control,
               exponents=expo, kicks=kicks, author_prediction=scores, author_values={f"{k[0]}|{k[1]}": v for k, v in AUTHOR.items()}, inertia_by_box=ladder)
    (a.out_dir / "m5_45_summary.json").write_text(json.dumps(out, indent=1) + "\n")

    # ------------------------------------------------------------------ tables
    def cw(s):
        return "%s, %s" % ("CONVERGED" if s["converged"] else "not converged", s["word"] or "under 1000 iterations")

    order = lambda s: (s["L1"], -s["spacing"], s["start"] != "seed", s["U"])  # noqa: E731
    print("**Table 1. The state reached from the seed at J = 0, against the spacing (δ = 0.05, half-box 6, mirror-averaged stencil).**\n")
    print("| L₁ | Spacing (grid) | U | U₂ | U₄ | U_V | I | I inside r < 2 (share) | Iterations | How it ended | Words | max \\|dU/dm\\|/h³ | Drift of I over the last tenth | The author's I |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in sorted(seeds, key=order):
        d = s["drift_of_I_over_the_last_tenth"]
        au = AUTHOR.get((s["spacing"], s["L1"]))
        print("| %g | %g (%d³) | %.4f | %.4f | %.4f | %.4f | %.1f | %.1f (%.3f) | %d | %s | %s | %.1e | %s | %s |" % (
            s["L1"], s["spacing"], s["N"], s["U"], s["U2"], s["U4"], s["UV"], s["I"], s["shells"]["inside_r2"], s["shells"]["inside_r2"] / s["I"], s["iterations"], s["how_it_ended"],
            cw(s), s["gmax_over_h3"], "n/a" if d is None else "%+.1e" % d, "n/a" if au is None else "%g" % au))
    print("\n**Table 2. The inertia at the stops of the chains from the seed.**\n")
    print("| L₁ | Spacing | " + " | ".join(f"I at {t}" for t in (1000, 3000, 10000, 30000, 100000)) + " | I at the end (iteration) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in sorted(seeds, key=order):
        at = {t["iteration"]: t["I"] for t in s["stops"] if t["kind"] == "target"}
        print("| %g | %g | %s | %.1f (%d) |" % (s["L1"], s["spacing"], " | ".join("%.1f" % at[t] if t in at else "n/a" for t in (1000, 3000, 10000, 30000, 100000)), s["I"], s["iterations"]))
    if control:
        print("\n**The control of the driver:** first end by the function tolerance at iteration %d (M5.42: %d), U %.15f against %.15f (relative %.1e), I %.12f against %.12f (relative %.1e): %s." % (
            control["iteration"], IT_REF, control["U"], U_REF, control["U_relative"], control["I"], I_REF, control["I_relative"], "MET" if control["met"] else "NOT MET"))
    st_list = [t["chain"] for t in states + kicks if t["stalled_line_search"]]
    print("\n**The chains with a stalled line search in their record (%d of %d):** %s. The largest max \\|dU/dm\\|/h³ among them is %.1e." % (
        len(st_list), len(states) + len(kicks), ", ".join("`%s`" % c for c in st_list), max(t["gmax_over_h3"] for t in states + kicks if t["stalled_line_search"])))
    if kicks:
        print("\n**Table 3. The states from the seed without L₁, kicked and relaxed again (independent Gaussian noise of standard deviation %g on the six components of every free node).**\n" % kicks[0]["amplitude"])
        print("| Spacing | U and I before the kick | U and I after the kick | U and I after relaxing again | Change of U | Change of I | Iterations | How it ended |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for t in sorted(kicks, key=lambda t: -t["spacing"]):
            print("| %g | %.6f, %.3f | %.4f, %.1f | %.6f, %.3f | %+.1e | %+.1e | %d | %s |" % (t["spacing"], t["U_before"], t["I_before"], t["U_after_the_kick"], t["I_after_the_kick"],
                  t["U"], t["I"], t["U_relative"], t["I_relative"], t["iterations"], t["how_it_ended"]))
    print("\n**Table 4. Every state found on each grid, lowest U first (δ = 0.05, half-box 6, J = 0).**\n")
    print("| L₁ | Spacing | Start | U at the start | U | I | I inside r < 2 (share) | U minus the seed state's | I over the seed state's | Iterations | How it ended | Words | max \\|dU/dm\\|/h³ |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in sorted(states, key=lambda s: (s["L1"], -s["spacing"], s["U"])):
        print("| %g | %g | %s | %s | %.6f | %.3f | %.1f (%.3f) | %s | %s | %d | %s | %s | %.1e |" % (
            s["L1"], s["spacing"], s["start"], "n/a" if s["start"] == "seed" else "%.2f" % s["U_at_the_start"], s["U"], s["I"], s["shells"]["inside_r2"], s["shells"]["inside_r2"] / s["I"],
            "n/a" if s["start"] == "seed" else "%+.4f" % s.get("U_minus_the_seed_state", float("nan")), "n/a" if s["start"] == "seed" else "%.4f" % s.get("I_over_the_seed_state", float("nan")),
            s["iterations"], s["how_it_ended"], cw(s), s["gmax_over_h3"]))
    print("\n**Table 5. The local exponent of the inertia in the spacing.**\n")
    print("| States | L₁ | From spacing (I) | To spacing (I) | Ratio of I | Exponent | Read as |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for e in expo:
        print("| %s | %g | %g (%.1f) | %g (%.1f) | %.3f | %.2f | %s |" % (e["states"], e["L1"], e["coarse"], e["I_coarse"], e["fine"], e["I_fine"], e["ratio_of_I"], e["exponent"],
              "a bound (a state not converged or RISING enters)" if e["a_bound"] else "a value"))
    if scores:
        print("\n**Table 6. The author's prediction at spacing 0.25 (I above 3000; the energy within 3 percent, read here as U against the %.4f of the state at spacing 0.375, a window of %.3f to %.3f; that reading of \"the energy\" is this task's).**\n" % (U_REF, WINDOW[0], WINDOW[1]))
        print("| Start | I | Above 3000 | U | U over the reference | In the window | Word | The state |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for t in scores:
            print("| %s | %.1f | %s | %.4f | %.4f | %s | %s | %s, %s |" % (t["start"], t["I"], "yes" if t["I"] > 3000 else "no", t["U"], t["U_over_reference"],
                  "yes" if WINDOW[0] <= t["U"] <= WINDOW[1] else "no", t["word"], "CONVERGED" if t["converged"] else "not converged", t["state_word"]))
    print("\n**Table 7. Where the inertia sits (W: the smallest set of nodes holding 95 percent of I; q as defined in § 2 row 4).**\n")
    print("| L₁ | Spacing | Start | Volume for 50 % of I | W: nodes (share of the grid) | W: volume | q on W | q on the rest | Rest over W: weighted by the inertia, unweighted, pooled | h × ∫‖∇M‖² dV | U₄ of this script against the evaluator |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in sorted(states, key=order):
        o = s["q_ratio_other_weightings"]
        print("| %g | %g | %s | %.1f | %d (%.4f) | %.1f | %.5f | %.4f | %.3g, %.3g, %.3g | %.1f | %.1e |" % (
            s["L1"], s["spacing"], s["start"], s["volume_for_half"], s["nodes_for_95"], s["W_share_of_nodes"], s["W_volume"], s["q_on_W"], s["q_on_the_rest"], s["q_ratio_rest_over_W"],
            o["unweighted"], o["pooled"], s["h_times_grad2"], s["control_U4"]["relative"]))
    print("\n**Table 8. The loop (the reader of M5.44).**\n")
    print("| L₁ | Spacing | Start | Pierced plaquettes | Loops / fragments | Radius (spread) | Rms off the plane | Normal |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in sorted(states, key=order):
        lp = s["loop"]
        print("| %g | %g | %s | %d | %d / %d | %s | %s | %s |" % (s["L1"], s["spacing"], s["start"], s["pierced"], s["loops"], s["fragments"],
              "n/a" if not lp else "%.3f (%.2f)" % (lp["radius"], lp["spread"]), "n/a" if not lp else "%.2f" % lp["off_plane"],
              "n/a" if not lp else "(%.2f, %.2f, %.2f)" % tuple(lp["normal"])))
    print("\n**Table 9. The inertia by box of the J = 0 ladders of M5.42 (δ = 0.3, mirror-averaged, spacing 0.375).**\n")
    print("| L₁ | Half-box (grid) | I | Inside r < 2 | r from 2 to 4 | r from 4 to the half-box | Beyond the inscribed sphere | Loop radius (M5.44) | Label |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for t in ladder:
        print("| %g | %g (%d³) | %.2f | %.2f | %.2f | %.2f | %.2f | %.3f | %s, %s |" % (t["L1"], t["half_box"], t["N"], t["I"], t["inside_r2"], t["r2_to_4"], t["r4_to_half_box"],
              t["beyond_inscribed_sphere"], t["loop_radius"], t["label"], "CONVERGED" if t["converged"] else "not converged"))

    # ------------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a.plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for L1, col, lab in ((0.0, "#b3471d", "without L₁"), (0.03, "#1f5fa8", "L₁ = 0.03")):
        ss = sorted((s for s in seeds if s["L1"] == L1), key=lambda s: s["spacing"])
        if not ss:
            continue
        ax[0].plot([s["spacing"] for s in ss], [s["I"] for s in ss], "o-", color=col, lw=2, ms=8, label=f"{lab}, from the seed")
        cc = [s for s in crosses if s["L1"] == L1]
        ax[0].plot([s["spacing"] for s in cc], [s["I"] for s in cc], "s", color=col, ms=6, mfc="none", mew=1.5, label=f"{lab}, carried from another state")
        for s in ss + cc:
            if not s["converged"]:
                ax[0].annotate("not converged", (s["spacing"], s["I"]), textcoords="offset points", xytext=(6, 4), fontsize=7)
        au = sorted((k[0], v) for k, v in AUTHOR.items() if k[1] == L1)
        ax[0].plot([q[0] for q in au], [q[1] for q in au], "x", color="k", ms=8, mew=1.8, label="the author's post" if L1 == 0.0 else None)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks([0.1875, 0.25, 0.375, 0.5])
    ax[0].set_xticklabels(["0.1875", "0.25", "0.375", "0.5"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("spacing")
    ax[0].set_ylabel("inertia I")
    ax[0].set_title("The inertia against the spacing (δ = 0.05, half-box 6, J = 0)", fontsize=10)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    for s, col in zip(sorted((s for s in seeds if s["L1"] == 0.0), key=lambda s: -s["spacing"]), ("#e0a36b", "#cf7a3a", "#b3471d", "#6e2206")):
        t = np.array(chains[s["chain"]]["trajectory"])
        if len(t):
            ax[1].plot(t[:, 0], t[:, 2], color=col, lw=2, label=f"spacing {s['spacing']:g}")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("inertia I")
    ax[1].set_title("Without L₁, from the seed: the inertia along the descent", fontsize=10)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.plots_dir / "m5_45_inertia_against_spacing.png", dpi=140)
    plt.close(fig)

    names = [s["chain"] for s in sorted(seeds, key=order)]
    fig, axs = plt.subplots(2, len(names), figsize=(2.6 * len(names), 5.6), squeeze=False)
    for k, n in enumerate(names):
        c = CHAINS[n]
        mod = Q.Model(c["N"], HALF_BOX, DELTA, c["L1"], stencil="averaged", fixed_chart=True)
        mod.evaluate(mod.seed6().reshape(-1), 0.0, need_grad=False, need_I=True)
        seed = mod.idn[:, :, c["N"] // 2].copy()
        for row, (img, ttl) in enumerate(((seed, "seed"), (slices[n]["dens_z0"], "end"))):
            axx = axs[row, k]
            im = axx.imshow(np.log10(np.maximum(img.T, 1e-6)), origin="lower", extent=[-HALF_BOX, HALF_BOX, -HALF_BOX, HALF_BOX], cmap="magma", vmin=-3, vmax=3)
            axx.set_title(f"{ttl}: spacing {2 * HALF_BOX / (c['N'] - 1):g}, L₁ = {c['L1']:g}", fontsize=7.5)
            axx.set_xticks([])
            axx.set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.7, label="log₁₀ of the inertia density on the plane z = 0")
    fig.savefig(a.plots_dir / "m5_45_inertia_density_slices.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    low = [lowest[k] for k in sorted(lowest, key=lambda k: (k[1], -k[0])) if k[1] == 0.0]
    if low:
        fig, axs = plt.subplots(1, len(low), figsize=(2.9 * len(low), 3.3), squeeze=False)
        for k, s in enumerate(low):
            axx = axs[0, k]
            im = axx.imshow(np.log10(np.maximum(slices[s["chain"]]["dens_z0"].T, 1e-6)), origin="lower", extent=[-HALF_BOX, HALF_BOX, -HALF_BOX, HALF_BOX], cmap="magma", vmin=-3, vmax=3)
            axx.set_title("spacing %g: %s\nU %.3f, I %.0f" % (s["spacing"], s["start"], s["U"], s["I"]), fontsize=7)
            axx.set_xticks([])
            axx.set_yticks([])
        fig.colorbar(im, ax=axs, shrink=0.8, label="log₁₀ of the inertia density, z = 0")
        fig.savefig(a.plots_dir / "m5_45_lowest_states_slices.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
