"""M5.48: the controls and the reads of the author's three predictions after M5.45 (m5_48_task_details.md, sections 3 and 4).

Reads the chain records of m5_48_relax.py and the saved states, and writes

    data/m5_48_controls.json    the controls K1 to K13 (`--controls` writes this file alone)
    data/m5_48_summary.json     every number of the tables (tracked)
    plots/m5_48_*.png           the excess and omega against J; the two falls against J; the inertia and U along the chain
                                of P2; the inertia against the spacing at delta = 0.3; sections of the inertia per node

and prints the tables in markdown. CONVERGED is the word of the tables of M5.45 (m5_45_reads.ending) with the gate of the
chain's own record; a chain that stalled above its gate (addition (b)) is never CONVERGED. The words STEADY and RISING
describe a chain that is not CONVERGED and enter nothing.

    P1   per state and candidate J = J_e/k, three chains: the reference M_0' (the state of M5.45 continued at J = 0 under the
         gate of the candidate), the chain at J from M_0', the return at J = 0 from M_J.
             dE = E_J[M_J] - U[M_0'],  omega = J / I[M_J],  rho = 1 - 2 I[M_0'] dE / J^2
         a state at a candidate is UNDECIDED when one of its chains is not CONVERGED, with one exception (the amendment
         after the fold): a chain at J that is not CONVERGED and whose last state fails S3 or S5 has "failed at the cut,
         not converged" and counts as not surviving. Decided, a state SURVIVES when
             S2  U[M_J] >= (1 - 1e-9) U[M_0']
             S3  at least 0.90 of the inertia of M_J sits on W_0, the smallest set of nodes holding 95 percent of the inertia of M_0'
             S4  the return closes nine tenths of the way: |I_ret - I_0'| <= 0.1 |I_J - I_0'| and |U_ret - U_0'| <= 0.1 |U_J - U_0'|
                 (floors: 1e-6 I_0' and 1e-10 U_0', or the change the reference itself made from the state of M5.45, if larger)
             S5  the loop reader of M5.44 finds as many loops on M_J as on M_0', the radius of the first within 10 percent
         per pair (one state at 0.375, one at 0.25):  f_X = 1 - X(0.25) / X(0.375);  f_0 = 1 - I_0(0.375) / I_0(0.25) on M5.45's states
         J* = the largest candidate at which the four states from the seed survive; the word is given only if no larger
         candidate is UNDECIDED.
    P2   HELD inside both forms of the window (I h in [1000, 1300] and I in [5300, 6900]) with U within 3 percent of 14.69;
         EDGE inside exactly one form; NOT HELD outside both, or U outside; no label for a chain that is not CONVERGED
    P3   G = I(seed, 0.25) / I(seed, 0.375), S = max I / min I over the CONVERGED states found at 0.375

Per state, three descriptors of how many nodes carry the inertia, with I_c the inertia of node c (the inertia density of the
functional times h^3): the largest share max I_c / sum I_c, the participation number N_I = (sum I_c)^2 / sum I_c^2, and the
participation volume V_I = h^3 N_I. No word and no threshold reads them.

The numeric thresholds are the constants below; nothing else in this file decides a word.

Usage:  python m5_48_reads.py [--controls] [--work-dir DIR] [--out-dir DIR] [--plots-dir DIR]
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
import m5_45_reads as R45  # noqa: E402
import m5_48_relax as D  # noqa: E402

# P1 (section 4)
S2_U, S3_SHARE, S4_FRACTION, S4_FLOOR_I, S4_FLOOR_U, S5_RADIUS = 1e-9, 0.90, 0.1, 1e-6, 1e-10, 0.10
P1_FALL, P1_L1_CHANGE, P1_FALSIFIER = 0.20, 0.02, 0.05
ARMS = {"without L1": ("h0375_L0", "h0250_L0"), "L1 = 0.03": ("h0375_L0.03", "h0250_L0.03")}
NO_L1_0375 = ("h0375_L0", "h0375_L0_from_h0500", "h0375_L0_from_h0250", "h0375_L0_from_h01875")
NO_L1_0250 = ("h0250_L0", "h0250_L0_from_h0375", "h0250_L0_from_h01875")
# P2: the two forms of the window in the post, and the window of U
P2_IH, P2_I, P2_U_CENTER, P2_U_REL = (1000.0, 1300.0), (5300.0, 6900.0), 14.69, 0.03
# P3: the author's thresholds, and nine tenths of the 1.609 and 1.376 of M5.45
P3_G, P3_S, P3_G_AS, P3_S_AS = 1.4, 1.2, 1.448, 1.239

_models = {}


def model(N, delta, L1):
    k = (N, delta, L1)
    if k not in _models:
        _models[k] = Q.Model(N, D.HALF_BOX, delta, L1, stencil="averaged", fixed_chart=True)
    return _models[k]


def load_rec(work_dir, chain):
    f = work_dir / f"m5_48_{chain}.json"
    if not f.exists():
        return None
    rec = json.loads(f.read_text())
    return rec if rec.get("finished") else None


def table_word(rec):
    """(how it ended, CONVERGED by the tables of M5.45, a stalled line search in the record, stalled above the gate)."""
    how, conv, line_search = R45.ending(rec)
    stalled = bool(rec.get("stalled_end") or (rec.get("stalled") and not rec["gate_met"]))
    if stalled:
        return how + ", STALLED above its gate (addition (b), %d times)" % len(rec["stalled"]), False, line_search, True
    return how, conv, line_search, False


def words(rec):
    """CONVERGED, or what a chain that is not was doing (the drift words describe such a chain only)."""
    _, conv, _, stalled = table_word(rec)
    if conv:
        return "CONVERGED"
    return "not converged, %s" % ("STALLED" if stalled else (rec["word"] or "under 1000 iterations"))


def inertia_set(mod, m6, frac=0.95):
    """The smallest set of nodes that holds frac of the inertia (as m5_45_reads.read), the inertia per node, U and I, and the
    three descriptors of how many nodes carry the inertia."""
    p = mod.evaluate(np.ascontiguousarray(m6).reshape(-1), 0.0, need_grad=False, need_I=True)
    dens = mod.h ** 3 * mod.idn.copy()
    order = np.argsort(dens.ravel())[::-1]
    cum = np.cumsum(dens.ravel()[order])
    k = int(np.searchsorted(cum, frac * p["I"]) + 1)
    inW = np.zeros(dens.size, bool)
    inW[order[:k]] = True
    tot = float(dens.sum())
    n_i = float(tot * tot / (dens * dens).sum())
    part = dict(largest_share=float(dens.max() / tot), N_I=n_i, V_I=float(mod.h ** 3 * n_i), W_nodes=k)
    return inW.reshape(dens.shape), dens, p, part


def s2(U_J, U_0):
    return bool(U_J >= (1 - S2_U) * U_0)


def s3(mod, m6_0, m6_J):
    W0, d0, p0, part0 = inertia_set(mod, m6_0)
    WJ, dJ, pJ, partJ = inertia_set(mod, m6_J)
    share = float(dJ[W0].sum() / dJ.sum())
    return dict(S3=bool(share >= S3_SHARE), share_on_W0=share, share_of_I0_on_W0=float(d0[W0].sum() / d0.sum()), jaccard=float((W0 & WJ).sum() / (W0 | WJ).sum()),
                largest_change_of_a_component=float(np.abs(np.asarray(m6_J) - np.asarray(m6_0)).max()), I0=p0["I"], U0=p0["U"], I=pJ["I"], U=pJ["U"],
                participation_of_M0=part0, participation=partJ)


def s4(I_ret, U_ret, I_0, U_0, I_J, U_J, I_45=None, U_45=None):
    """The return closes nine tenths of the way in I and in U. The floors are the level at which a reference is itself
    reproduced: 1e-6 I_0 and 1e-10 U_0 (the kicked states of M5.45 returned within 6e-7 and 1e-12), or the change the reference
    made from the state of M5.45, if that is larger."""
    floor_I = max(S4_FLOOR_I * I_0, 0.0 if I_45 is None else abs(I_0 - I_45))
    floor_U = max(S4_FLOOR_U * abs(U_0), 0.0 if U_45 is None else abs(U_0 - U_45))
    ok_I = abs(I_ret - I_0) <= max(S4_FRACTION * abs(I_J - I_0), floor_I)
    ok_U = abs(U_ret - U_0) <= max(S4_FRACTION * abs(U_J - U_0), floor_U)
    return dict(S4=bool(ok_I and ok_U), I_part=bool(ok_I), U_part=bool(ok_U), I_left_over=float(abs(I_ret - I_0) / max(abs(I_J - I_0), 1e-300)),
                U_left_over=float(abs(U_ret - U_0) / max(abs(U_J - U_0), 1e-300)), floor_I=floor_I, floor_U=floor_U)


def loops(m6, x):
    lp = LP.read_state(m6, x, second=False)
    return dict(loops=lp["loops"], fragments=lp["fragments"], pierced=lp["pierced"], radius=None if not lp["loop"] else lp["loop"]["radius"])


def s5(l0, lJ):
    """As many loops on M_J as on the reference, the radius of the first within 10 percent."""
    same = l0["loops"] == lJ["loops"]
    rad = None if (l0["radius"] is None or lJ["radius"] is None) else abs(lJ["radius"] - l0["radius"]) / l0["radius"]
    return dict(S5=bool(same and (rad is None or rad <= S5_RADIUS)), same_number_of_loops=bool(same), radius_change=rad)


def decide(reference_converged, converged, S2, S3, S5, return_converged=None, S4=None):
    """The status of a state at a candidate from its clauses: (status, why), status None while a return is still owed.

    The amendment after the fold: a chain at J that is not CONVERGED (cut by a cap, or STALLED) is read at its last state
    against the two clauses that need no convergence. Failing S3 or S5 there it has failed at the cut and does not survive;
    passing both it is UNDECIDED."""
    if not reference_converged:
        return "undecided", "the reference is not CONVERGED under its gate"
    if not converged:
        gone = [q for q, v in (("S3", S3), ("S5", S5)) if v is False]
        if gone:
            return "fails", "failed at the cut, not converged (its last state fails %s)" % " and ".join(gone)
        return "undecided", "the chain at J is not CONVERGED and its last state passes S3 and S5"
    failed = [q for q, v in (("S2", S2), ("S3", S3), ("S5", S5)) if not v]
    if failed:
        return "fails", "fails " + ", ".join(failed)
    if return_converged is None:
        return None, "the return has no record"
    if not return_converged:
        return "undecided", "the return is not CONVERGED"
    return ("survives", None) if S4 else ("fails", "fails S4")


def state_at(s, k, work_dir, out_dir):
    """One state of P1 at one candidate: its three chains, the clauses, and its status: None (a chain not run yet),
    "undecided" (a chain not CONVERGED), "fails" or "survives"."""
    m = D.P1_STATES[s]
    name = f"p1_{s}_j{k}"
    J = Q.j_electron(0.05) / k
    out = dict(state=s, k=k, chain=name, J=J, N=m["N"], spacing=2 * D.HALF_BOX / (m["N"] - 1), L1=m["L1"], status=None, why="the reference has no record", S2=None, S3=None, S4=None, S5=None,
               reference_converged=None, converged=None, needs_a_return=None)
    ref = load_rec(work_dir, name + "_ref")
    if ref is None:
        return out
    how_r, conv_r, _, _ = table_word(ref)
    lr = ref["stops"][-1]
    I45, U45 = ref["start"]["I"], ref["start"]["U"]
    I0, U0 = lr["I"], lr["U"]
    out.update(I0_of_M5_45=I45, U0_of_M5_45=U45, I0=I0, U0=U0, gate=ref["gate"], clock_force_over_h3=ref["start"].get("clock_force_over_h3"), reference_converged=conv_r,
               reference=dict(iterations=ref["iterations"], how_it_ended=how_r, words=words(ref), gmax_over_h3=lr["gmax_over_h3"], U_minus_M5_45=U0 - U45, I_relative_to_M5_45=(I0 - I45) / I45))
    if not conv_r:
        out.update(status="undecided", why="the reference is not CONVERGED under the gate %.2e (%s)" % (ref["gate"], how_r))
        return out
    rj = load_rec(work_dir, name)
    if rj is None:
        out["why"] = "the chain at J has no record"
        return out
    how, conv, line_search, _ = table_word(rj)
    last = rj["stops"][-1]
    dE = last["E"] - U0
    out.update(E=last["E"], U=last["U"], I=last["I"], dE=dE, omega=J / last["I"], rigid_excess=J * J / (2 * I0), rho=1 - 2 * I0 * dE / (J * J), I_over_I0=last["I"] / I0,
               iterations=rj["iterations"], how_it_ended=how, converged=conv, words=words(rj), stalled_line_search=line_search, gmax_over_h3=last["gmax_over_h3"], label=last["label"])
    z0 = np.load(out_dir / ref["end_file"], allow_pickle=False)
    zJ = np.load(out_dir / rj["end_file"], allow_pickle=False)
    mod = model(m["N"], 0.05, m["L1"])
    r3 = s3(mod, z0["m6"], zJ["m6"])
    l0, lJ = loops(z0["m6"], z0["x"]), loops(zJ["m6"], zJ["x"])
    r5 = s5(l0, lJ)
    out.update(S2=s2(last["U"], U0), S3=r3["S3"], share_on_W0=r3["share_on_W0"], jaccard=r3["jaccard"], largest_change_of_a_component=r3["largest_change_of_a_component"],
               S5=r5["S5"], loops_on_the_reference=l0, loops_at_J=lJ, radius_change=r5["radius_change"], participation_of_the_reference=r3["participation_of_M0"], participation=r3["participation"])
    if not conv:  # the amendment after the fold: read at its last state against S3 and S5
        st, why = decide(True, False, out["S2"], out["S3"], out["S5"])
        out.update(status=st, why=why + " (%s)" % how, needs_a_return=False, failed_at_the_cut=bool(st == "fails"),
                   at_the_cut=dict(U=last["U"], I=last["I"], gmax_over_h3=last["gmax_over_h3"], share_on_W0=r3["share_on_W0"], S3=r3["S3"], S5=r5["S5"],
                                   loops_on_the_reference=l0, loops=lJ, radius_change=r5["radius_change"], how_it_ended=how))
        return out
    st, why = decide(True, True, out["S2"], out["S3"], out["S5"])
    out["needs_a_return"] = st is None
    if st is not None:
        out.update(status=st, why=why)
        return out
    rb = load_rec(work_dir, name + "_back")
    if rb is None:
        out["why"] = "the return has no record"
        return out
    bhow, bconv, _, _ = table_word(rb)
    bl = rb["stops"][-1]
    r4 = s4(bl["I"], bl["U"], I0, U0, last["I"], last["U"], I45, U45)
    out["back"] = dict(iterations=rb["iterations"], how_it_ended=bhow, converged=bconv, words=words(rb), U=bl["U"], I=bl["I"], **r4)
    st, why = decide(True, True, True, True, True, bconv, r4["S4"])
    out.update(S4=r4["S4"] if bconv else None, status=st, why=(why + " (%s)" % bhow) if st == "undecided" else why)
    return out


# ----------------------------------------------------------------------------- P1: the reader (a function of the table alone)

def pair(a, b):
    """The falls for a pair of decided states: a at spacing 0.375, b at 0.25, the same J. The band is what the measured rises of
    I leave to the fall of the excess when S2 holds: J^2/2I_J <= dE <= J^2/2I_0 on each grid."""
    r, ac, af = a["I0"] / b["I0"], a["I"] / a["I0"], b["I"] / b["I0"]
    return dict(coarse=a["state"], fine=b["state"], f_dE=1 - b["dE"] / a["dE"], f_omega=1 - b["omega"] / a["omega"], f_0=1 - a["I0_of_M5_45"] / b["I0_of_M5_45"],
                f_0_on_the_references=1 - r, rise_of_I_coarse=ac, rise_of_I_fine=af, band_of_f_dE=[1 - r * ac, 1 - r / af], both_survive=bool(a["status"] == "survives" and b["status"] == "survives"))


def decided(t):
    return t is not None and t["status"] in ("survives", "fails")


def status_of(T, states, k):
    sts = [None if T.get((s, k)) is None else T[(s, k)]["status"] for s in states]
    if "fails" in sts:
        return "fails"
    return "survives" if all(q == "survives" for q in sts) else "undecided"


def p1_word(f0, f1):
    """The word at J* from the falls without L1 (f0) and with L1 = 0.03 (f1), and the clauses that failed."""
    held0 = f0["f_dE"] >= P1_FALL and f0["f_omega"] >= P1_FALL
    held1 = abs(f1["f_dE"]) < P1_L1_CHANGE and abs(f1["f_omega"]) < P1_L1_CHANGE
    if abs(f0["f_dE"]) < P1_FALSIFIER:
        return "FALSIFIED", ["the absolute value of f_dE under 0.05 without L1"]
    if held0 and held1:
        return "HELD", []
    failed = []
    if P1_FALSIFIER <= f0["f_dE"] < P1_FALL:
        failed.append("f_dE from 0.05 up to 0.20")
    if f0["f_dE"] <= -P1_FALSIFIER:
        failed.append("f_dE at or under -0.05 (the excess rose)")
    if f0["f_omega"] < P1_FALL:
        failed.append("f_omega under 0.20")
    if not held1:
        failed.append("an absolute value of 0.02 or more with L1 = 0.03")
    return "NOT HELD", failed


def p1_reading(T, ks=D.P1_K):
    """Everything P1 reads from the table T[(state, k)] of state_at: the candidates, J*, the word, the pairs, the reading by arm."""
    cands = []
    for k in ks:
        row = dict(k=k, J=Q.j_electron(0.05) / k, status=status_of(T, D.P1_LABEL_STATES, k),
                   states={s: (None if T.get((s, k)) is None else dict(status=T[(s, k)]["status"], why=T[(s, k)]["why"])) for s in D.P1_LABEL_STATES})
        for arm, (a, b) in ARMS.items():
            row[arm] = pair(T[(a, k)], T[(b, k)]) if decided(T.get((a, k))) and decided(T.get((b, k))) else None
            row[arm + ", status"] = status_of(T, (a, b), k)
        cands.append(row)
    by_k = {r["k"]: r for r in cands}
    surv = [k for k in ks if by_k[k]["status"] == "survives"]
    und = [k for k in ks if by_k[k]["status"] == "undecided"]
    out = dict(k_star=None, J_star=None, word="no label", reason=None, clauses_failed=[], qualifier=None, pairs=[], pairs_left_out=[], pairs_at_or_above_in_both=None, by_arm={})
    if not surv:
        out["reason"] = ("undecided at " + ", ".join("J_e/%d" % k for k in und)) if und else "no candidate survives"
    else:
        ks_ = min(surv)
        larger = [k for k in und if k < ks_]
        out.update(k_star=ks_, J_star=by_k[ks_]["J"])
        f0, f1 = by_k[ks_]["without L1"], by_k[ks_]["L1 = 0.03"]
        if larger:
            out["reason"] = "undecided at " + ", ".join("J_e/%d" % k for k in larger) + ", larger than the largest surviving candidate"
        else:
            out["word"], out["clauses_failed"] = p1_word(f0, f1)
        side = (f0["f_dE"] >= P1_FALL, f0["f_omega"] >= P1_FALL)
        for a in NO_L1_0375:
            for b in NO_L1_0250:
                ta, tb = T.get((a, ks_)), T.get((b, ks_))
                if decided(ta) and decided(tb):
                    out["pairs"].append(pair(ta, tb))
                else:
                    out["pairs_left_out"].append([a, b])
        out["pairs_at_or_above_in_both"] = sum(p["f_dE"] >= P1_FALL and p["f_omega"] >= P1_FALL for p in out["pairs"])
        if any((p["f_dE"] >= P1_FALL, p["f_omega"] >= P1_FALL) != side for p in out["pairs"]):
            out["qualifier"] = "START-DEPENDENT"
        tilted = T.get(("h0250_L0.03_from_no_L1", ks_))
        out["pair_with_the_tilted_state"] = pair(T[("h0375_L0.03", ks_)], tilted) if decided(tilted) else None
    for arm in ARMS:
        ok = [k for k in ks if by_k[k][arm + ", status"] == "survives"]
        ua = [k for k in ks if by_k[k][arm + ", status"] == "undecided"]
        out["by_arm"][arm] = None if not ok else dict(k=min(ok), larger_undecided=[k for k in ua if k < min(ok)], **by_k[min(ok)][arm])
    return dict(candidates=cands, reading=out)


def added_return(work_dir, T):
    """The one chain added after the audit of the findings (no word): the return at J = 0 of the state from the seed at spacing
    0.25 at J_e/40, which fails S5 there by the radius alone. Read with S4, it says what that state would be with S5 read as a
    loop count alone."""
    t = T[("h0250_L0", 40)]
    rec = load_rec(work_dir, "p1_h0250_L0_j40_back_added")
    if rec is None:
        pf = work_dir / "p1_h0250_L0_j40_back_added" / "progress.json"
        return dict(finished=False, progress=json.loads(pf.read_text()) if pf.exists() else None)
    how, conv, _, _ = table_word(rec)
    bl = rec["stops"][-1]
    r4 = s4(bl["I"], bl["U"], t["I0"], t["U0"], t["I"], t["U"], t["I0_of_M5_45"], t["U0_of_M5_45"])
    count = bool(t["S2"] and t["S3"] and t["loops_on_the_reference"]["loops"] == t["loops_at_J"]["loops"])
    # the U part as decision 6 of the plan audit first wrote it, one-sided (the frozen S4 takes absolute values): a return that
    # ends lower in U than its reference passes this form. Printed beside the frozen one; it decides nothing.
    one_sided = bool(r4["I_part"] and (bl["U"] - t["U0"]) <= max(S4_FRACTION * (t["U"] - t["U0"]), r4["floor_U"]))
    return dict(finished=True, S4_with_the_U_part_one_sided=one_sided, U_minus_U_of_the_reference=bl["U"] - t["U0"], I_relative_to_the_reference=(bl["I"] - t["I0"]) / t["I0"], chain=rec["chain"], iterations=rec["iterations"], seconds=rec["seconds"], how_it_ended=how, converged=conv, words=words(rec), cut_by_wall=rec["cut_by_wall"],
                ended_at_cap=rec["ended_at_cap"], gate=rec["gate"], gmax_over_h3=bl["gmax_over_h3"], U=bl["U"], I=bl["I"], label=bl["label"], U_of_the_reference=t["U0"], I_of_the_reference=t["I0"],
                U_at_J=t["U"], I_at_J=t["I"], **r4, with_S5_as_a_loop_count_alone=("undecided" if not conv else ("survives" if (count and r4["S4"]) else "fails")))


def readings(T, added):
    """Three readings of P1 side by side, each also with S5 read as a loop count alone. Descriptors: the word of record is the
    first one, as the frozen rules give it.

      the frozen rules         one J for both arms, the clauses S2 to S5 as frozen
      by arm                   the same clauses, each value of L1 taking its own largest surviving candidate
      with the label SMOOTH    as the frozen rules, a state that is not SMOOTH by the label of M5.42 counted as failing

    With S5 as a loop count alone, a state that failed S5 by the radius alone owes a return; only one such return exists (the
    added chain), so the others are undecided in that variant."""
    def variant(count_only, smooth, one_sided=False):
        V = {}
        for key, t in T.items():
            t = dict(t)
            if count_only and t["status"] == "fails" and t.get("S5") is False and t["S2"] and t["S3"] and t["loops_on_the_reference"]["loops"] == t["loops_at_J"]["loops"]:
                if key == ("h0250_L0", 40) and added.get("finished"):
                    t["status"] = added["with_S5_as_a_loop_count_alone"]
                    if one_sided and added["converged"]:
                        t["status"] = "survives" if added["S4_with_the_U_part_one_sided"] else "fails"
                    t["why"] = "the added return: " + ("S4 holds" if t["status"] == "survives" else ("S4 fails" if t["status"] == "fails" else "not CONVERGED"))
                else:
                    t["status"], t["why"] = "undecided", "no return was run"
            if smooth and t.get("label") is not None and t["label"] != "SMOOTH" and t["status"] in ("survives", "undecided"):
                t["status"], t["why"] = "fails", "not SMOOTH (" + t["label"] + ")"
            V[key] = t
        return p1_reading(V)

    def row(res, by_arm=False):
        r = res["reading"]
        if not by_arm:
            return dict(k_star=r["k_star"], word=r["word"], reason=r["reason"], qualifier=r["qualifier"], clauses_failed=r["clauses_failed"],
                        candidates={c["k"]: c["status"] for c in res["candidates"]})
        a, b = r["by_arm"]["without L1"], r["by_arm"]["L1 = 0.03"]
        if a is None or b is None:
            return dict(k_star=None, word="no label", reason="an arm with no surviving candidate", clauses_failed=[])
        w, failed = p1_word(a, b)
        blocked = a["larger_undecided"] or b["larger_undecided"]
        return dict(k_star=dict(without_L1=a["k"], with_L1=b["k"]), word="no label" if blocked else w, reason=("a larger candidate undecided on an arm" if blocked else None), clauses_failed=failed,
                    f_without_L1=[a["f_dE"], a["f_omega"]], f_with_L1=[b["f_dE"], b["f_omega"]], larger_undecided=dict(without_L1=a["larger_undecided"], with_L1=b["larger_undecided"]))

    A, B, C, Dv = variant(False, False), variant(True, False), variant(False, True), variant(True, True)
    E, F = variant(True, False, True), variant(True, True, True)  # as B and Dv, with the U part of S4 one-sided on the added return
    key = "with_S5_as_a_loop_count_alone_and_the_U_part_of_S4_one_sided"
    return [{"reading": "the frozen rules (the word of record)", "as_frozen": row(A), "with_S5_as_a_loop_count_alone": row(B), key: row(E)},
            {"reading": "by arm: each value of L1 takes its own largest surviving candidate", "as_frozen": row(A, True), "with_S5_as_a_loop_count_alone": row(B, True), key: row(E, True)},
            {"reading": "with the label SMOOTH as a clause of the survival test", "as_frozen": row(C), "with_S5_as_a_loop_count_alone": row(Dv), "by_arm": row(C, True),
             "by_arm_with_S5_as_a_loop_count_alone": row(Dv, True), key: row(F), "by_arm_" + key: row(F, True)}]


def p1(work_dir, out_dir):
    T = {(s, k): state_at(s, k, work_dir, out_dir) for s in D.P1_STATES for k in D.P1_K}
    res = p1_reading(T)
    res["added_return"] = added_return(work_dir, T)
    res["readings"] = readings(T, res["added_return"])
    # the ratio of the participation volume between the two spacings, on the pairs from the seed
    vi = []
    for k in D.P1_K:
        for arm, (a, b) in ARMS.items():
            ta, tb = T[(a, k)], T[(b, k)]
            if ta.get("participation") and tb.get("participation"):
                vi.append(dict(arm=arm, k=k, V_I_ratio_at_J=tb["participation"]["V_I"] / ta["participation"]["V_I"],
                               V_I_ratio_on_the_references=tb["participation_of_the_reference"]["V_I"] / ta["participation_of_the_reference"]["V_I"]))
    res.update(states=list(T.values()), participation_volume_ratios=vi)
    return res


# ----------------------------------------------------------------------------- P2

def p2_word(converged, I, U, h=0.1875):
    """(word, the numbers beside it). `converged` is the CONVERGED of the tables; a state that is not gets no label."""
    Ih = I * h
    lo, hi = P2_U_CENTER * (1 - P2_U_REL), P2_U_CENTER * (1 + P2_U_REL)
    in_U, in_Ih, in_I = bool(lo <= U <= hi), bool(P2_IH[0] <= Ih <= P2_IH[1]), bool(P2_I[0] <= I <= P2_I[1])
    edge = P2_IH[0] if Ih < P2_IH[0] else (P2_IH[1] if Ih > P2_IH[1] else min(P2_IH, key=lambda e: abs(Ih - e)))
    num = dict(I=I, I_times_h=Ih, U=U, U_window=[lo, hi], U_in_window=in_U, inside_the_form_in_I_times_h=in_Ih, inside_the_form_in_I=in_I,
               I_times_h_from_the_nearer_edge_percent=100 * (Ih - edge) / edge, U_from_center_percent=100 * (U - P2_U_CENTER) / P2_U_CENTER)
    if not converged:
        return "no label", num
    if not in_U or not (in_Ih or in_I):
        return "NOT HELD", num
    return ("HELD" if (in_Ih and in_I) else "EDGE"), num


def p2(work_dir, out_dir):
    rec = load_rec(work_dir, "p2_h01875_L0")
    if rec is None:
        pf = work_dir / "p2_h01875_L0" / "progress.json"
        return dict(finished=False, progress=json.loads(pf.read_text()) if pf.exists() else None)
    how, conv, line_search, stalled = table_word(rec)
    last = rec["stops"][-1]
    word, num = p2_word(conv, last["I"], last["U"], rec["model"]["spacing"])
    m = rec["model"]
    _, _, _, part = inertia_set(model(m["N"], m["delta"], m["L1"]), np.load(out_dir / rec["end_file"], allow_pickle=False)["m6"])
    return dict(finished=True, chain=rec["chain"], iterations=rec["iterations"], iterations_new=rec["iterations"] - rec["carried"]["iterations"], how_it_ended=how,
                converged=conv, converged_by_the_driver=rec["converged"], stalled_line_search=line_search, stalled_above_the_gate=stalled, words=words(rec),
                drift_of_I_over_the_last_tenth=rec["drift_of_I_over_the_last_tenth"], gmax_over_h3=last["gmax_over_h3"], gate=rec["gate"], label=last["label"],
                cut_by_wall=rec["cut_by_wall"], ended_at_cap=rec["ended_at_cap"], seconds=rec["seconds"], function_tolerance_ends=rec["function_tolerance_ends"],
                abnormal=rec["abnormal"], stalled=rec["stalled"], resumed_at=rec["resumed_at"], start=rec["start"], carried=rec["carried"],
                start_file_unchanged=rec.get("start_file_unchanged"), word=word, word_printed=word + ", on the continued chain", participation=part, **num,
                stops=[{k: s.get(k) for k in ("iteration", "kind", "U", "I", "gmax_over_h3", "carried_from")} for s in rec["stops"]], trajectory=R45.thin(rec["trajectory"], 400))


# ----------------------------------------------------------------------------- P3

def p3_ratios(states):
    """G, S and the ratios of the two families of M5.45 from a list of states (spacing, start, U, I, converged). A ratio is a
    value only when both of its states are CONVERGED; S is taken over the CONVERGED states of a grid."""
    hs = sorted({s["spacing"] for s in states}, reverse=True)
    seed = {s["spacing"]: s for s in states if s["start"] == "seed"}
    lowest = {}
    for s in states:
        if s["spacing"] not in lowest or s["U"] < lowest[s["spacing"]]["U"]:
            lowest[s["spacing"]] = s

    def ladder(pick, tag):
        out = []
        for h1, h2 in zip(hs, hs[1:]):
            if h1 in pick and h2 in pick:
                s1, s2_ = pick[h1], pick[h2]
                out.append(dict(states=tag, coarse=h1, fine=h2, I_coarse=s1["I"], I_fine=s2_["I"], ratio_of_I=s2_["I"] / s1["I"], a_value=bool(s1["converged"] and s2_["converged"])))
        return out

    spread = {}
    for h in hs:
        every = [s for s in states if s["spacing"] == h]
        conv = [s for s in every if s["converged"]]
        spread[h] = dict(states=len(every), converged=len(conv), I_min=min((s["I"] for s in conv), default=None), I_max=max((s["I"] for s in conv), default=None),
                         S=(max(s["I"] for s in conv) / min(s["I"] for s in conv)) if len(conv) >= 2 else None)
    return dict(from_the_seed=ladder(seed, "from the seed"), lowest_U_found=ladder(lowest, "the lowest U found on each grid"), spread=spread)


def p3_word(ratios, coarse=0.375, fine=0.25, starts=4):
    g = [r for r in ratios["from_the_seed"] if r["coarse"] == coarse and r["fine"] == fine]
    G, G_value = (g[0]["ratio_of_I"], g[0]["a_value"]) if g else (None, False)
    sp = ratios["spread"].get(coarse, dict(states=0, converged=0, S=None))
    S = sp["S"]
    if (G_value and G > P3_G) or (S is not None and S > P3_S):
        word = "NOT HELD"
    elif G_value and G <= P3_G and sp["states"] == starts and sp["converged"] == starts and S <= P3_S:
        word = "HELD"
    else:
        word = "no label"
    if word == "HELD":
        line = "BELOW delta = 0.05"
    elif G_value and S is not None and G >= P3_G_AS and S >= P3_S_AS:
        line = "AS AT delta = 0.05"
    else:
        line = "BETWEEN"
    return dict(G=G, G_is_a_value=bool(G_value), S=S, states_at_coarse=sp["states"], converged_at_coarse=sp["converged"], word=word, second_reading=line,
                the_inertia_fell_as_the_spacing_fell=bool(G_value and G < 1))


def p3(work_dir, out_dir):
    states, missing = [], []
    for name, c in D.chains().items():
        if c["task"] != "P3":
            continue
        rec = load_rec(work_dir, name)
        if rec is None:
            missing.append(name)
            continue
        how, conv, line_search, stalled = table_word(rec)
        st, _ = R45.read(rec, out_dir)
        m = rec["model"]
        _, _, _, part = inertia_set(model(m["N"], m["delta"], m["L1"]), np.load(out_dir / rec["end_file"], allow_pickle=False)["m6"])
        # the chain at 0.1875 is a start only (it ends by its count of 10000 iterations): a state that is not CONVERGED enters no ratio as a value
        st.update(start="seed" if c["start"]["kind"] == "seed" else "carried from spacing %g" % rec["start"]["source_spacing"], source=rec["start"].get("file"),
                  source_converged=rec["start"].get("source_converged"), source_cut_by_wall=rec["start"].get("source_cut_by_wall"),
                  iterations=rec["iterations"], how_it_ended=how, converged=conv, converged_by_the_driver=rec["converged"], stalled_line_search=line_search, stalled_above_the_gate=stalled,
                  words=words(rec), label=rec["label"], cut_by_wall=rec["cut_by_wall"], ended_at_cap=rec["ended_at_cap"], drift_of_I_over_the_last_tenth=rec["drift_of_I_over_the_last_tenth"],
                  function_tolerance_ends=rec["function_tolerance_ends"], participation=part, stops=[{k: s[k] for k in ("iteration", "kind", "U", "I", "gmax_over_h3")} for s in rec["stops"]],
                  trajectory=R45.thin(rec["trajectory"]))
        states.append(st)
    ratios = p3_ratios(states) if states else None
    seed = {s["spacing"]: s for s in states if s["start"] == "seed"}
    vi = seed[0.25]["participation"]["V_I"] / seed[0.375]["participation"]["V_I"] if 0.25 in seed and 0.375 in seed else None
    return dict(states=states, missing=missing, ratios=ratios, reading=p3_word(ratios) if ratios else None, V_I_ratio_from_the_seed_0375_to_0250=vi)


# ----------------------------------------------------------------------------- every chain

def chains_table(work_dir):
    """One row per chain of the registry: how it ended, or why it was not run (the reason the pool recorded)."""
    pool_f = work_dir.parent / "m5_48_pool.json"
    skipped = json.loads(pool_f.read_text())["skipped"] if pool_f.exists() else {}
    rows = []
    for name, c in D.chains().items():
        rec = load_rec(work_dir, name)
        row = dict(chain=name, task=c["task"], role=c.get("role"), N=c["N"], spacing=2 * D.HALF_BOX / (c["N"] - 1), delta=c["delta"], L1=c["L1"], J=D.j_of(c), cap=c["cap"], wall_hours=c["wall_hours"])
        if rec is None:
            row.update(ran=False, why=skipped.get(name, "no record at this read"))
        else:
            how, conv, line_search, stalled = table_word(rec)
            last = rec["stops"][-1]
            row.update(ran=True, iterations=rec["iterations"], iterations_new=rec["iterations"] - (rec["carried"]["iterations"] if rec.get("carried") else 0), seconds=rec["seconds"],
                       how_it_ended=how, converged=conv, words=words(rec), cut_by_wall=rec["cut_by_wall"], ended_at_cap=rec["ended_at_cap"], stalled_above_the_gate=stalled,
                       stalls=len(rec["stalled"]), stalled_line_search=line_search, gate=rec["gate"], gmax_over_h3=last["gmax_over_h3"], U=last["U"], I=last["I"], label=last["label"], resumed_at=rec["resumed_at"])
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- the controls

def synthetic_table(f0=None, f1=None, status=None, tabled=False):
    """A made-up table for the P1 reader. The states from the seed get the falls f0 = (f_dE, f_omega) without L1 and f1 with
    L1 = 0.03 at every candidate; None gives the rigid values J^2/2I_0 and J/I_0 on the inertias of M5.45. With `tabled` the
    six other states are present, rigid. status(s, k) gives the status of a state from the seed (default: survives)."""
    I45 = {"h0375_L0": 2095.989, "h0250_L0": 3372.312, "h0375_L0_from_h0500": 1977.478, "h0375_L0_from_h0250": 2469.492, "h0375_L0_from_h01875": 2721.793,
           "h0250_L0_from_h0375": 3434.668, "h0250_L0_from_h01875": 3936.182, "h0375_L0.03": 119.1005, "h0250_L0.03": 119.3205, "h0250_L0.03_from_no_L1": 119.167}
    T = {}
    for s, I0 in I45.items():
        if s not in D.P1_LABEL_STATES and not tabled:
            continue
        for k in D.P1_K:
            J = Q.j_electron(0.05) / k
            t = dict(state=s, k=k, status="survives", why=None, I0=I0, I0_of_M5_45=I0, I=I0, dE=J * J / (2 * I0), omega=J / I0)
            if s in D.P1_LABEL_STATES:
                f = f0 if D.P1_STATES[s]["L1"] == 0 else f1
                if f is not None:
                    fine = s.startswith("h0250")
                    t.update(dE=(1 - f[0]) if fine else 1.0, omega=(1 - f[1]) if fine else 1.0)
                if status is not None:
                    t["status"] = status(s, k)
                    t["why"] = None if t["status"] == "survives" else "made up"
            T[(s, k)] = t
    return T


def controls(work_dir, out_dir):
    K = {}
    s45 = json.loads((out_dir / "m5_45_summary.json").read_text())
    by45 = {t["chain"]: t for t in s45["states"]}
    paths42 = json.loads((out_dir / "m5_42_paths.json").read_text())
    Je = Q.j_electron(0.05)

    def end42(chain, path):
        p = {q["name"]: q for q in paths42[chain]["paths"]}[path]
        return p["iterations"], p["stops"][-1]

    # K1: the J = 0 states of M5.45 read back by the evaluator with the settings of the driver
    rows = []
    for s, m in list(D.P1_STATES.items()) + [("h01875_L0", dict(N=65, L1=0.0))]:
        mod = model(m["N"], 0.05, m["L1"])
        z = np.load(out_dir / f"m5_45_{s}_end.npz", allow_pickle=False)
        p = mod.evaluate(z["m6"].reshape(-1), 0.0)
        ref = by45[s]
        rows.append(dict(state=s, U=p["U"], I=p["I"], gmax_over_h3=float(np.abs(p["grad"]).max() / mod.h ** 3), U_relative=abs(p["U"] - ref["U"]) / abs(ref["U"]),
                         I_relative=abs(p["I"] - ref["I"]) / abs(ref["I"]), converged_by_the_tables_of_M5_45=ref["converged"]))
    K["K1"] = dict(what="the ten J = 0 end states of M5.45 used by P1 and the start of P2, read back: U and I of m5_45_summary.json to 1e-12 (relative)", rows=rows,
                   largest_relative=max(max(r["U_relative"], r["I_relative"]) for r in rows), met=bool(all(max(r["U_relative"], r["I_relative"]) <= 1e-12 for r in rows)))

    def first_end(chain, ref_it, ref, keys, tol=1e-10):
        rec = load_rec(work_dir, chain)
        if rec is None or not rec["function_tolerance_ends"]:
            return dict(met=None, note="no record, or no end by the function tolerance, at the time of this read")
        f = rec["function_tolerance_ends"][0]
        rel = {k: abs(f[k] - ref[k]) / abs(ref[k]) for k in keys}
        return dict(iteration=f["iteration"], reference_iteration=ref_it, **{k: f[k] for k in keys}, reference={k: ref[k] for k in keys}, relative=rel,
                    end_iteration=rec["iterations"], end_message=rec["message"], met=bool(f["iteration"] == ref_it and max(rel.values()) <= tol))

    it, ref = end42("r6_hb6_L0.0", "j0")
    K["K2"] = dict(what="k2_h0375_L0 (delta 0.05, no L1, spacing 0.375, from the seed): its first end by the function tolerance against chain r6_hb6_L0.0 path j0 of M5.42 (8364), and its end at 8383",
                   **first_end("k2_h0375_L0", it, ref, ("U", "I")))
    if K["K2"]["met"] is not None:
        K["K2"]["end_iteration_of_M5_45"] = by45["h0375_L0"]["iterations"]
        K["K2"]["met"] = bool(K["K2"]["met"] and K["K2"]["end_iteration"] == by45["h0375_L0"]["iterations"])
    it, ref = end42("r6_hb6_L0.3", "j40")
    K["K3"] = dict(what="k3_r6_L0.3_j40 (L1 = 0.3, J = J_e/40, from the J = 0 end of M5.42): first end by the function tolerance against path j40 of chain r6_hb6_L0.3",
                   **first_end("k3_r6_L0.3_j40", it, ref, ("E", "U", "I")))
    it, ref = end42("r6_hb6_L0.0", "j40")
    K["K4"] = dict(what="k4_r6_L0_j40 (no L1, J = J_e/40): first end by the function tolerance against path j40 of chain r6_hb6_L0.0", **first_end("k4_r6_L0_j40", it, ref, ("E", "U", "I")))
    it, ref = end42("r3_g33_L0.0", "j0")
    K["K5"] = dict(what="p3_h0375 (delta 0.3, no L1, from the seed): first end by the function tolerance against path j0 of chain r3_g33_L0.0 of M5.42", **first_end("p3_h0375", it, ref, ("U", "I")))

    # K6: the gradient of E_J against central differences at 200 random interior entries (the best of three steps).
    # As rewritten at the fold of the plan audit: the clock term alone on the two J = 0 states of spacing 0.375 (the difference
    # of J^2/2I against the part of the gradient that J adds), and the whole E_J on the seed of each grid, which is not a
    # stationary state, each within 1e-6 of its largest entry. The first wording (the whole E_J on the two J = 0 states,
    # within 1e-6 of the largest entry) could not be met by a correct gradient at the smallest J: there the largest entry of a
    # converged state is the clock force alone, under the round-off of the central difference of U. Its result is kept.
    rows = []
    rng = np.random.default_rng(2026111948)
    for s in ("h0375_L0", "h0375_L0.03"):
        m = D.P1_STATES[s]
        mod = model(m["N"], 0.05, m["L1"])
        free = np.flatnonzero(np.repeat(mod.mask.reshape(-1), 6))
        pick = rng.choice(free, 200, replace=False)
        for tag, x0 in (("the J = 0 state", np.load(out_dir / f"m5_45_{s}_end.npz", allow_pickle=False)["m6"].reshape(-1).copy()), ("the seed", mod.seed6().reshape(-1).copy())):
            g0 = mod.evaluate(x0, 0.0)["grad"].ravel()[pick].copy()
            an, gmax = {}, {}
            for k in (80, 1280):
                g = mod.evaluate(x0, Je / k)["grad"]
                an[k], gmax[k] = g.ravel()[pick].copy(), float(np.abs(g).max())
            for eps in (1e-3, 1e-4, 1e-5):
                Up, Um, Ip, Im = (np.zeros(200) for _ in range(4))
                for n, i in enumerate(pick):
                    x = x0.copy()
                    x[i] += eps
                    q = mod.evaluate(x, 0.0, need_grad=False, need_I=True)
                    Up[n], Ip[n] = q["U"], q["I"]
                    x[i] -= 2 * eps
                    q = mod.evaluate(x, 0.0, need_grad=False, need_I=True)
                    Um[n], Im[n] = q["U"], q["I"]
                for k in (80, 1280):
                    J = Je / k
                    fd_clock = (J * J / 2) * (1 / Ip - 1 / Im) / (2 * eps)
                    fd = (Up - Um) / (2 * eps) + fd_clock
                    an_clock = an[k] - g0
                    rows.append(dict(state=s, at=tag, k=k, J=J, step=eps, largest_gradient_entry=gmax[k], largest_difference=float(np.abs(fd - an[k]).max()),
                                     over_the_largest_entry=float(np.abs(fd - an[k]).max() / gmax[k]),
                                     clock_term_alone_largest_difference=float(np.abs(fd_clock - an_clock).max()),
                                     clock_term_alone_over_its_largest_entry=float(np.abs(fd_clock - an_clock).max() / np.abs(an_clock).max()),
                                     round_off_of_the_difference_of_U=float(2.2e-16 * abs(Up[0]) / eps)))

    def best(at, key):
        out = {}
        for r in rows:
            if r["at"] == at:
                q = f"{r['state']}|J_e/{r['k']}"
                out[q] = min(out.get(q, np.inf), r[key])
        return out

    first_wording, clock, seed = best("the J = 0 state", "over_the_largest_entry"), best("the J = 0 state", "clock_term_alone_over_its_largest_entry"), best("the seed", "over_the_largest_entry")
    K["K6"] = dict(what="central differences against the gradient at 200 random interior entries, each within 1e-6 of its largest entry: the clock term alone on the two J = 0 states of spacing 0.375, "
                        "and the whole E_J on the seed of each grid, at J_e/80 and J_e/1280",
                   rows=rows, clock_term_alone_on_the_J0_states=clock, whole_E_J_on_the_seed=seed,
                   met=bool(all(v <= 1e-6 for v in clock.values()) and all(v <= 1e-6 for v in seed.values())),
                   first_wording=dict(what="the whole E_J on the two J = 0 states, within 1e-6 of the largest entry of the gradient (the wording of the plan before the fold of its audit)",
                                      result=first_wording, met=bool(all(v <= 1e-6 for v in first_wording.values())),
                                      note="NOT MET at J_e/1280: the largest gradient entry of a converged state there is the clock force alone, under the round-off of the central difference of U"))

    # K7: the clauses of the survival test on known answers
    mod = model(33, 0.05, 0.0)
    za = np.load(out_dir / "m5_45_h0375_L0_end.npz", allow_pickle=False)
    zo = np.load(out_dir / "m5_45_h0375_L0_from_h01875_end.npz", allow_pickle=False)
    zl = np.load(out_dir / "m5_45_h0375_L0.03_end.npz", allow_pickle=False)
    b0 = np.load(out_dir / "m5_42_r6_hb6_L0.0_j0_end.npz", allow_pickle=False)
    bJ = np.load(out_dir / "m5_42_r6_hb6_L0.0_j40_end.npz", allow_pickle=False)
    k7 = dict(what="the clauses of the survival test on known answers: S3 on a state against itself (passes), on the state of M5.42 at J_e/40 (passes, share 0.928) and on a state of another start (fails); "
                   "S2 on a U lowered by hand; S4 on returns that do not move, close 80, 95 and 100 percent of the way, and on a tiny excursion; S5 on a state against itself and against the state with L1 = 0.03")
    k7["S3_itself"] = {k: v for k, v in s3(mod, za["m6"], za["m6"]).items() if not k.startswith("participation")}
    k7["S3_M5_42_at_J_e_over_40"] = {k: v for k, v in s3(mod, b0["m6"], bJ["m6"]).items() if not k.startswith("participation")}
    k7["S3_another_start"] = {k: v for k, v in s3(mod, za["m6"], zo["m6"]).items() if not k.startswith("participation")}
    U0 = by45["h0375_L0"]["U"]
    k7["S2"] = dict(equal=s2(U0, U0), higher=s2(U0 * (1 + 1e-3), U0), lowered_by_1e_6=s2(U0 * (1 - 1e-6), U0), lowered_by_1e_10=s2(U0 * (1 - 1e-10), U0))
    I0, IJ, UJ = 2000.0, 2200.0, U0 + 0.01
    ret = lambda frac: s4(I0 + (1 - frac) * (IJ - I0), U0 + (1 - frac) * (UJ - U0), I0, U0, IJ, UJ)["S4"]  # noqa: E731
    k7["S4"] = dict(does_not_move=ret(0.0), closes_80_percent=ret(0.80), closes_95_percent=ret(0.95), closes_all=ret(1.0),
                    tiny_excursion_inside_the_floor=s4(I0 * (1 + 5e-7), U0, I0, U0, I0 * (1 + 1e-9), U0)["S4"],
                    another_state_as_the_return=s4(by45["h0375_L0_from_h01875"]["I"], by45["h0375_L0_from_h01875"]["U"], by45["h0375_L0"]["I"], U0, 3171.28, 14.9144)["S4"])
    la, ll, lb0, lbJ = loops(za["m6"], za["x"]), loops(zl["m6"], zl["x"]), loops(b0["m6"], b0["x"]), loops(bJ["m6"], bJ["x"])
    k7["S5"] = dict(itself=s5(la, la), against_the_state_with_L1=s5(la, ll), M5_42_at_J_e_over_40=dict(s5(lb0, lbJ), reference=lb0, at_J=lbJ, note="reported, no answer fixed in advance"),
                    radius_without_L1=la["radius"], radius_with_L1=ll["radius"])
    k7["met"] = bool(k7["S3_itself"]["S3"] and k7["S3_itself"]["share_on_W0"] >= 0.95 and k7["S3_M5_42_at_J_e_over_40"]["S3"] and abs(k7["S3_M5_42_at_J_e_over_40"]["share_on_W0"] - 0.928) < 1e-3
                     and not k7["S3_another_start"]["S3"]
                     and k7["S2"]["equal"] and k7["S2"]["higher"] and not k7["S2"]["lowered_by_1e_6"] and k7["S2"]["lowered_by_1e_10"]
                     and not k7["S4"]["does_not_move"] and not k7["S4"]["closes_80_percent"] and k7["S4"]["closes_95_percent"] and k7["S4"]["closes_all"]
                     and k7["S4"]["tiny_excursion_inside_the_floor"] and not k7["S4"]["another_state_as_the_return"]
                     and k7["S5"]["itself"]["S5"] and not k7["S5"]["against_the_state_with_L1"]["S5"])
    K["K7"] = k7

    # K8: the ratios of P3 on the records of M5.45
    st = [dict(spacing=t["spacing"], start=t["start"], U=t["U"], I=t["I"], converged=t["converged"]) for t in s45["states"] if t["L1"] == 0.0]
    r8 = p3_ratios(st)
    g = {(r["coarse"], r["fine"]): r for r in r8["from_the_seed"]}
    lo = {(r["coarse"], r["fine"]): r for r in r8["lowest_U_found"]}
    got = dict(growth_from_the_seed=g[(0.375, 0.25)]["ratio_of_I"], growth_lowest_U=lo[(0.375, 0.25)]["ratio_of_I"], spread_0375=r8["spread"][0.375]["S"], spread_0500=r8["spread"][0.5]["S"],
               spread_0250=r8["spread"][0.25]["S"], to_01875_is_no_value=bool(not g[(0.25, 0.1875)]["a_value"] and not lo[(0.25, 0.1875)]["a_value"]),
               converged_pairs_are_values=bool(g[(0.375, 0.25)]["a_value"] and g[(0.5, 0.375)]["a_value"]))
    want = dict(growth_from_the_seed=1.609, growth_lowest_U=1.446, spread_0375=1.376, spread_0500=1.735, spread_0250=1.167)
    w8 = p3_word(r8)
    made = lambda G, S, n=4, conv=4, val=True: p3_word(dict(from_the_seed=[dict(coarse=0.375, fine=0.25, ratio_of_I=G, a_value=val)], spread={0.375: dict(states=n, converged=conv, S=S)}))  # noqa: E731
    cases = dict(held=made(1.2, 1.1), not_held_on_G=made(1.5, 1.1), not_held_on_S_alone=made(1.2, 1.3, 4, 2, False), no_label_G_not_a_value=made(1.2, 1.1, 4, 4, False),
                 no_label_a_state_missing=made(1.2, 1.1, 4, 3), as_at=made(1.449, 1.24), between=made(1.447, 1.24), fell=made(0.8, 1.1))
    K["K8"] = dict(what="the reads of P3 applied to the records of M5.45 (delta = 0.05): 1.609 and 1.446; 1.376, 1.735, 1.167; the ratio to 0.1875 no value; and the words on made-up ratios", got=got,
                   expected=want, word_on_M5_45=w8, made_up={k: dict(word=v["word"], second_reading=v["second_reading"], fell=v["the_inertia_fell_as_the_spacing_fell"]) for k, v in cases.items()},
                   met=bool(all(abs(got[k] - v) < 5e-4 for k, v in want.items()) and got["to_01875_is_no_value"] and got["converged_pairs_are_values"]
                            and w8["word"] == "NOT HELD" and w8["second_reading"] == "AS AT delta = 0.05"
                            and cases["held"]["word"] == "HELD" and cases["held"]["second_reading"] == "BELOW delta = 0.05" and cases["not_held_on_G"]["word"] == "NOT HELD"
                            and cases["not_held_on_S_alone"]["word"] == "NOT HELD" and cases["no_label_G_not_a_value"]["word"] == "no label" and cases["no_label_a_state_missing"]["word"] == "no label"
                            and cases["as_at"]["second_reading"] == "AS AT delta = 0.05" and cases["between"]["second_reading"] == "BETWEEN"
                            and cases["fell"]["the_inertia_fell_as_the_spacing_fell"] and cases["fell"]["word"] == "HELD"))

    # K9: the window of P2, both edges
    cut = by45["h01875_L0"]
    cases = [("the cut state of M5.45", cut["converged"], cut["I"], cut["U"], "no label"), ("made up", True, 5400.0, 14.70, "HELD"), ("made up, the lower edge", True, 5310.0, 14.70, "EDGE"),
             ("made up, the upper edge", True, 6920.0, 14.70, "EDGE"), ("made up", True, 5290.0, 14.70, "NOT HELD"), ("made up", True, 6950.0, 14.70, "NOT HELD"),
             ("made up", True, 5400.0, 15.20, "NOT HELD"), ("made up, an edge with U outside", True, 5310.0, 14.20, "NOT HELD")]
    rows = []
    for tag, conv, I, U, want_w in cases:
        w, num = p2_word(conv, I, U)
        rows.append(dict(case=tag, converged=conv, I=I, U=U, I_times_h=num["I_times_h"], word=w, expected=want_w, met=bool(w == want_w)))
    K["K9"] = dict(what="the window of P2 on the cut state of M5.45 and seven made-up records, both edges of the two forms", rows=rows, cut_state_I_times_h=rows[0]["I_times_h"],
                   met=bool(all(r["met"] for r in rows) and abs(rows[0]["I_times_h"] - 993.09) < 5e-3))

    # K10: the continuation
    k10 = dict(what="the start of p2_h01875_L0: U and I of the record, the counter at 27200, 272 trajectory points; the start file unchanged; a name outside m5_48_ refused")
    rec = load_rec(work_dir, "p2_h01875_L0")
    sf = work_dir / "p2_h01875_L0" / "state.json"
    src = rec if rec is not None else (json.loads(sf.read_text()) if sf.exists() else None)
    try:
        D.out_file(out_dir, "m5_45_h01875_L0_end.npz")
        refused = False
    except RuntimeError:
        refused = True
    k10["refuses_a_name_outside_m5_48"] = refused
    if src is None:
        k10.update(met=None, note="the chain has no checkpoint or record at the time of this read")
    else:
        car, start = src["carried"], src["start"]
        first_new = [t for t in (src.get("trajectory") or src.get("traj")) if t[0] > car["iterations"]]
        now = D.sha256_of(out_dir / start["file"])
        k10.update(start_file=start["file"], sha256_at_the_start=start["sha256"], sha256_at_this_read=now, unchanged=bool(now == start["sha256"]), chain_finished=bool(rec is not None),
                   counter_at_the_start=car["iterations"], trajectory_points_carried=car["trajectory_points"], read_back=car["read_back"], U_at_the_start=start["U"], I_at_the_start=start["I"],
                   first_new_trajectory_iteration=(first_new[0][0] if first_new else None),
                   met=bool(refused and now == start["sha256"] and car["iterations"] == 27200 and car["trajectory_points"] == 272
                            and max(car["read_back"]["U_relative"], car["read_back"]["I_relative"]) <= 1e-12 and (not first_new or first_new[0][0] == 27300)))
    K["K10"] = k10

    # K11: the reader of P1 on made-up tables, each word once
    def read(**kw):
        r = p1_reading(synthetic_table(**kw))["reading"]
        return dict(word=r["word"], reason=r["reason"], clauses_failed=r["clauses_failed"], qualifier=r["qualifier"], k_star=r["k_star"],
                    pairs=len(r["pairs"]), pairs_left_out=len(r["pairs_left_out"]), pairs_at_or_above_in_both=r["pairs_at_or_above_in_both"])

    c11 = dict(
        held=read(), held_start_dependent=read(tabled=True), falsified=read(f0=(0.03, 0.30)), falsified_from_below=read(f0=(-0.03, 0.30)),
        not_held_between=read(f0=(0.12, 0.25)), not_held_the_excess_rose=read(f0=(-0.10, 0.25)), not_held_on_omega=read(f0=(0.30, 0.10)), not_held_on_the_arm_with_L1=read(f1=(0.05, 0.001)),
        no_candidate_survives=read(status=lambda s, k: "fails"),
        undecided_above=read(status=lambda s, k: "fails" if k < 160 else ("undecided" if (k == 160 and s == "h0250_L0") else "survives")),
        undecided_everywhere=read(status=lambda s, k: "undecided"),
        a_failing_state_beats_an_undecided_one=read(status=lambda s, k: "survives" if k >= 320 else ("fails" if s == "h0375_L0" else "undecided")),
        the_largest_survivor_is_taken=read(status=lambda s, k: "fails" if k < 320 else "survives"),
        cut_failing_S3_does_not_block=read(status=lambda s, k: decide(True, False, True, False, True)[0] if (k == 40 and s == "h0375_L0.03") else "survives"),
        cut_failing_S5_does_not_block=read(status=lambda s, k: decide(True, False, True, True, False)[0] if (k == 40 and s == "h0250_L0.03") else "survives"),
        cut_passing_S3_and_S5_above_blocks=read(status=lambda s, k: decide(True, False, True, True, True)[0] if (k == 80 and s == "h0375_L0.03") else ("fails" if k == 40 else "survives")),
        cut_passing_S3_and_S5_below_does_not_block=read(status=lambda s, k: decide(True, False, True, True, True)[0] if (k == 1280 and s == "h0375_L0.03") else "survives"),
        start_dependent_on_omega_alone=p1_reading({**synthetic_table(tabled=True), **{("h0375_L0_from_h0500", k): dict(state="h0375_L0_from_h0500", k=k, status="survives", why=None, I0=1977.478,
                                                  I0_of_M5_45=1977.478, I=1977.478, dE=(Q.j_electron(0.05) / k) ** 2 / (2 * 1977.478), omega=(Q.j_electron(0.05) / k) / 3000.0) for k in D.P1_K},
                                                   **{("h0375_L0_from_h01875", k): None for k in D.P1_K}})["reading"])
    sd = c11["start_dependent_on_omega_alone"]
    c11["start_dependent_on_omega_alone"] = dict(word=sd["word"], qualifier=sd["qualifier"], pairs=len(sd["pairs"]), pairs_left_out=len(sd["pairs_left_out"]),
                                                 omega_sides=sorted({bool(p["f_omega"] >= P1_FALL) for p in sd["pairs"]}), dE_sides=sorted({bool(p["f_dE"] >= P1_FALL) for p in sd["pairs"]}))
    ok11 = (c11["held"]["word"] == "HELD" and c11["held"]["qualifier"] is None and c11["held"]["k_star"] == 40 and c11["held"]["pairs"] == 1 and c11["held"]["pairs_left_out"] == 11
            and c11["held_start_dependent"]["word"] == "HELD" and c11["held_start_dependent"]["qualifier"] == "START-DEPENDENT" and c11["held_start_dependent"]["pairs"] == 12
            and c11["held_start_dependent"]["pairs_at_or_above_in_both"] == 11
            and c11["falsified"]["word"] == "FALSIFIED" and c11["falsified_from_below"]["word"] == "FALSIFIED"
            and c11["not_held_between"]["word"] == "NOT HELD" and c11["not_held_between"]["clauses_failed"] == ["f_dE from 0.05 up to 0.20"]
            and c11["not_held_the_excess_rose"]["word"] == "NOT HELD" and c11["not_held_the_excess_rose"]["clauses_failed"] == ["f_dE at or under -0.05 (the excess rose)"]
            and c11["not_held_on_omega"]["word"] == "NOT HELD" and c11["not_held_on_omega"]["clauses_failed"] == ["f_omega under 0.20"]
            and c11["not_held_on_the_arm_with_L1"]["word"] == "NOT HELD" and c11["not_held_on_the_arm_with_L1"]["clauses_failed"] == ["an absolute value of 0.02 or more with L1 = 0.03"]
            and c11["no_candidate_survives"]["word"] == "no label" and c11["no_candidate_survives"]["reason"] == "no candidate survives"
            and c11["undecided_above"]["word"] == "no label" and c11["undecided_above"]["k_star"] == 320 and "J_e/160" in c11["undecided_above"]["reason"]
            and c11["undecided_everywhere"]["word"] == "no label" and "undecided at" in c11["undecided_everywhere"]["reason"]
            and c11["a_failing_state_beats_an_undecided_one"]["word"] == "HELD" and c11["a_failing_state_beats_an_undecided_one"]["k_star"] == 320
            and c11["the_largest_survivor_is_taken"]["k_star"] == 320 and c11["the_largest_survivor_is_taken"]["word"] == "HELD"
            and decide(True, False, True, False, True) == ("fails", "failed at the cut, not converged (its last state fails S3)")
            and decide(True, False, True, True, False)[0] == "fails" and decide(True, False, False, True, True)[0] == "undecided" and decide(False, False, None, None, None)[0] == "undecided"
            and decide(True, True, True, True, True, None)[0] is None and decide(True, True, True, True, True, False, True)[0] == "undecided"
            and decide(True, True, True, True, True, True, True)[0] == "survives" and decide(True, True, True, True, True, True, False)[0] == "fails" and decide(True, True, True, False, True)[0] == "fails"
            and c11["cut_failing_S3_does_not_block"]["word"] == "HELD" and c11["cut_failing_S3_does_not_block"]["k_star"] == 80
            and c11["cut_failing_S5_does_not_block"]["word"] == "HELD" and c11["cut_failing_S5_does_not_block"]["k_star"] == 80
            and c11["cut_passing_S3_and_S5_above_blocks"]["word"] == "no label" and c11["cut_passing_S3_and_S5_above_blocks"]["k_star"] == 160 and "J_e/80" in c11["cut_passing_S3_and_S5_above_blocks"]["reason"]
            and c11["cut_passing_S3_and_S5_below_does_not_block"]["word"] == "HELD" and c11["cut_passing_S3_and_S5_below_does_not_block"]["k_star"] == 40
            and c11["start_dependent_on_omega_alone"]["qualifier"] == "START-DEPENDENT" and c11["start_dependent_on_omega_alone"]["dE_sides"] == [True]
            and c11["start_dependent_on_omega_alone"]["omega_sides"] == [False, True] and c11["start_dependent_on_omega_alone"]["pairs_left_out"] == 3)
    K["K11"] = dict(what="the reader of P1 on made-up tables: HELD; HELD, START-DEPENDENT (the pair at 0.193 of M5.45's inertias); FALSIFIED; each clause of NOT HELD alone; no label with no survivor; "
                         "no label with a larger candidate UNDECIDED; a decided failure not hidden by an undecided state; the largest survivor taken; START-DEPENDENT on the fall of omega alone; and the amendment after the fold: a cut chain that fails S3, or S5, at its last state "
                         "does not block the word, one that passes both blocks it above J* and not below", cases=c11, met=bool(ok11))

    # K12: addition (b) and the gate, on constructed inputs and on a chain that cannot meet its gate
    f = D.after_function_tolerance
    c12 = dict(a_long_segment_restarts=f(300, 1, 1.0, 1e-3, 0), a_repeat_under_the_gate_ends=f(3, 2, 1e-5, 1e-3, 0), a_repeat_above_the_gate_stalls=f(3, 2, 1e-2, 1e-3, 0),
               the_third_stall_restarts=f(3, 4, 1e-2, 1e-3, 2), the_fourth_stall_ends=f(3, 5, 1e-2, 1e-3, 3), forty_ends_end=f(300, 40, 1e-5, 1e-3, 0),
               gate_of_a_small_clock_force=D.gate_of(5.31e-3), gate_of_a_large_clock_force=D.gate_of(0.34))
    mod = model(33, 0.05, 0.0)
    gk = float(np.abs(mod.evaluate(za["m6"].reshape(-1), Je / 320)["grad"]).max() / mod.h ** 3)
    c12["clock_force_on_the_seed_state_at_J_e_over_320"] = gk
    c12["its_gate"] = D.gate_of(gk)
    rec = load_rec(work_dir, "k12_stalled")
    if rec is None:
        c12["chain"] = None
    else:
        how, conv, _, stalled = table_word(rec)
        c12["chain"] = dict(gate=rec["gate"], stalls_recorded=len(rec["stalled"]), message=rec["message"], how_it_ended=how, converged=conv, stalled_above_the_gate=stalled, gmax_over_h3=rec["stops"][-1]["gmax_over_h3"])
    K["K12"] = dict(what="addition (b) and the gate: the rule after an end by the function tolerance on six constructed inputs; the gate on two clock forces; and the chain of K3 under a gate of 1e-12, "
                         "which must record a stall and must not be CONVERGED", cases=c12,
                    met=None if rec is None else bool(c12["a_long_segment_restarts"] == "restart" and c12["a_repeat_under_the_gate_ends"] == "end" and c12["a_repeat_above_the_gate_stalls"] == "stalled, restart"
                                                      and c12["the_third_stall_restarts"] == "stalled, restart" and c12["the_fourth_stall_ends"] == "stalled, end" and c12["forty_ends_end"] == "end"
                                                      and abs(c12["gate_of_a_small_clock_force"] - 5.31e-4) < 1e-12 and c12["gate_of_a_large_clock_force"] == 1e-3 and abs(c12["its_gate"] - 0.1 * gk) < 1e-15 and gk < 1e-2
                                                      and c12["chain"]["stalls_recorded"] >= 1 and not c12["chain"]["converged"] and c12["chain"]["stalled_above_the_gate"]))

    # K13: the interpolated start of the driver against the starts that the carried chains of M5.45 recorded
    rows = []
    for t in s45["states"]:
        if t.get("source_file"):
            mod = model(t["N"], 0.05, t["L1"])
            x = D.interp_start(mod, np.load(out_dir / t["source_file"], allow_pickle=False)).reshape(-1)
            p = mod.evaluate(x, 0.0, need_grad=False, need_I=True)
            rows.append(dict(chain=t["chain"], source=t["source_file"], U=p["U"], I=p["I"], U_relative=abs(p["U"] - t["U_at_the_start"]) / t["U_at_the_start"], I_relative=abs(p["I"] - t["I_at_the_start"]) / t["I_at_the_start"]))
    K["K13"] = dict(what="the interpolated start of the driver on the %d carried chains of M5.45: U and I at the start as M5.45 recorded them, to 1e-12 (relative)" % len(rows), rows=rows,
                    largest_relative=max(max(r["U_relative"], r["I_relative"]) for r in rows), met=bool(len(rows) == 14 and all(max(r["U_relative"], r["I_relative"]) <= 1e-12 for r in rows)))
    return K


# ----------------------------------------------------------------------------- tables and figures

def fmt(v, f="%.4f"):
    return "n/a" if v is None else f % v


def print_tables(out):
    P2, P3, P1 = out["P2"], out["P3"], out["P1"]
    yn = {True: "yes", False: "no", None: "n/a"}
    print("\n**The controls.**\n")
    print("| Control | Met | What |")
    print("| --- | --- | --- |")
    for k, v in out["controls"].items():
        print("| %s | %s | %s |" % (k, {True: "yes", False: "NO", None: "not read yet"}[v["met"]], v["what"]))
    print("\n**P2. The chain from the seed at spacing 0.1875, continued.**\n")
    if not P2["finished"]:
        print("Not finished at this read; progress:", P2.get("progress"))
    else:
        print("| Iterations (new) | How it ended | Words | Largest gradient entry over h³ | Drift of I over the last tenth | U | I | I·h | I·h from the nearer edge | Inside the form in I·h, in I | Largest share, N_I, V_I | Word |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        pt = P2["participation"]
        print("| %d (%d) | %s | %s | %.2e | %s | %.6f | %.3f | %.3f | %+.2f %% | %s, %s | %.5f, %.0f, %.2f | %s |" % (
            P2["iterations"], P2["iterations_new"], P2["how_it_ended"], P2["words"], P2["gmax_over_h3"], fmt(P2["drift_of_I_over_the_last_tenth"], "%+.1e"), P2["U"], P2["I"], P2["I_times_h"],
            P2["I_times_h_from_the_nearer_edge_percent"], yn[P2["inside_the_form_in_I_times_h"]], yn[P2["inside_the_form_in_I"]], pt["largest_share"], pt["N_I"], pt["V_I"], P2["word_printed"]))
    print("\n**P3. Every state found at δ = 0.3 without L₁, lowest U first on each grid.**\n")
    print("| Spacing | Start | U | I | I inside r < 2 (share) | W: share of the nodes | q on W | q on the rest | Largest share | N_I | V_I | Iterations | How it ended | Words | Largest gradient entry over h³ |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in sorted(P3["states"], key=lambda s: (-s["spacing"], s["U"])):
        pt = s["participation"]
        print("| %g | %s | %.6f | %.3f | %.1f (%.3f) | %.4f | %s | %s | %.5f | %.0f | %.2f | %d | %s | %s | %.1e |" % (
            s["spacing"], s["start"], s["U"], s["I"], s["shells"]["inside_r2"], s["shells"]["inside_r2"] / s["I"], s["W_share_of_nodes"], fmt(s["q_on_W"], "%.5f"), fmt(s["q_on_the_rest"]),
            pt["largest_share"], pt["N_I"], pt["V_I"], s["iterations"], s["how_it_ended"], s["words"], s["gmax_over_h3"]))
    if P3["ratios"]:
        print("\n| States | From spacing (I) | To spacing (I) | Ratio of I | Read as |")
        print("| --- | --- | --- | --- | --- |")
        for e in P3["ratios"]["from_the_seed"] + P3["ratios"]["lowest_U_found"]:
            print("| %s | %g (%.1f) | %g (%.1f) | %.3f | %s |" % (e["states"], e["coarse"], e["I_coarse"], e["fine"], e["I_fine"], e["ratio_of_I"], "a value" if e["a_value"] else "no value (a state not CONVERGED)"))
        print("\n| Spacing | States found | CONVERGED | I from | I to | Spread S |")
        print("| --- | --- | --- | --- | --- | --- |")
        for h, v in sorted(P3["ratios"]["spread"].items(), reverse=True):
            print("| %g | %d | %d | %s | %s | %s |" % (h, v["states"], v["converged"], fmt(v["I_min"], "%.1f"), fmt(v["I_max"], "%.1f"), fmt(v["S"], "%.3f")))
        r = P3["reading"]
        print("\nP3: G = %s (%s), S = %s on %d CONVERGED of %d states: **%s**; the second reading: %s%s; the ratio of V_I from 0.375 to 0.25 on the states from the seed: %s." % (
            fmt(r["G"], "%.3f"), "a value" if r["G_is_a_value"] else "no value", fmt(r["S"], "%.3f"), r["converged_at_coarse"], r["states_at_coarse"], r["word"], r["second_reading"],
            "; the inertia fell as the spacing fell" if r["the_inertia_fell_as_the_spacing_fell"] else "", fmt(P3["V_I_ratio_from_the_seed_0375_to_0250"], "%.3f")))
    if P3["missing"]:
        print("\n(no record: %s)" % ", ".join(P3["missing"]))
    print("\n**P1. The references: the states of M5.45 continued at J = 0 under the gate of each candidate.**\n")
    print("| L₁ | Spacing | State | J_e/k | Gate | Iterations | How it ended | U − U of M5.45 | I, relative to M5.45 | Largest gradient entry over h³ |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    order = lambda s: (s["L1"], -s["spacing"], list(D.P1_STATES).index(s["state"]), s["k"])  # noqa: E731
    for s in sorted(P1["states"], key=order):
        if s.get("reference"):
            q = s["reference"]
            print("| %g | %g | `%s` | %d | %.2e | %d | %s | %+.1e | %+.1e | %.1e |" % (s["L1"], s["spacing"], s["state"], s["k"], s["gate"], q["iterations"], q["how_it_ended"], q["U_minus_M5_45"], q["I_relative_to_M5_45"], q["gmax_over_h3"]))
    print("\n**P1. Every state at every candidate.**\n")
    print("| L₁ | Spacing | State | J_e/k | J | Iterations | How it ended | ΔE | ω | ρ | I/I₀′ | Share on W₀ | S2 | S3 | S4 (left over in I, in U) | S5 (loops, radius: reference, at J) | SMOOTH | Largest share | N_I | V_I | Status |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in sorted(P1["states"], key=order):
        if s.get("dE") is None:
            print("| %g | %g | `%s` | %d | %.4f | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | %s |" % (s["L1"], s["spacing"], s["state"], s["k"], s["J"], (s["status"] or "no record") + (": " + s["why"] if s["why"] else "")))
            continue
        b, pt = s.get("back"), s.get("participation")
        l0, lJ = s.get("loops_on_the_reference"), s.get("loops_at_J")
        print("| %g | %g | `%s` | %d | %.4f | %d | %s | %.6f | %.6f | %.4f | %.4f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            s["L1"], s["spacing"], s["state"], s["k"], s["J"], s["iterations"], s["how_it_ended"], s["dE"], s["omega"], s["rho"], s["I_over_I0"], fmt(s.get("share_on_W0")), yn[s["S2"]], yn[s["S3"]],
            "not run" if not b else "%s (%.3f, %.3f)" % (yn[s["S4"]] if b["converged"] else "return not CONVERGED", b["I_left_over"], b["U_left_over"]),
            "n/a" if not l0 else "%s (%d, %s; %d, %s)" % (yn[s["S5"]], l0["loops"], fmt(l0["radius"], "%.3f"), lJ["loops"], fmt(lJ["radius"], "%.3f")), yn[s["label"] == "SMOOTH"],
            fmt(pt and pt["largest_share"], "%.5f"), fmt(pt and pt["N_I"], "%.0f"), fmt(pt and pt["V_I"], "%.2f"), (s["status"] or "no record") + (": " + s["why"] if s["why"] else "")))
    cut = [s for s in sorted(P1["states"], key=order) if s.get("at_the_cut")]
    if cut:
        print("\n**P1. The chains at J that are not CONVERGED, read at their last state (the amendment after the fold).**\n")
        print("| L₁ | Spacing | State | J_e/k | How it ended | U | I | Largest gradient entry over h³ | Share on W₀ (S3) | Loops, radius: reference, last state (S5) | Read as |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for s in cut:
            c = s["at_the_cut"]
            print("| %g | %g | `%s` | %d | %s | %.6f | %.3f | %.1e | %.4f (%s) | %d, %s; %d, %s (%s) | %s |" % (
                s["L1"], s["spacing"], s["state"], s["k"], c["how_it_ended"], c["U"], c["I"], c["gmax_over_h3"], c["share_on_W0"], yn[c["S3"]], c["loops_on_the_reference"]["loops"],
                fmt(c["loops_on_the_reference"]["radius"], "%.3f"), c["loops"]["loops"], fmt(c["loops"]["radius"], "%.3f"), yn[c["S5"]],
                "failed at the cut, not converged" if s["failed_at_the_cut"] else "UNDECIDED"))
    print("\n**P1. The candidates, on the pairs from the seed.**\n")
    print("| J_e/k | J | Status | Without L₁: f_ΔE | f_ω | f₀ | Rise of I at 0.375, at 0.25 | Band left to f_ΔE | L₁ = 0.03: f_ΔE | f_ω | f₀ | Rise of I at 0.375, at 0.25 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in P1["candidates"]:
        a, b = r["without L1"], r["L1 = 0.03"]
        why = "; ".join("%s %s" % (s, v["why"]) for s, v in r["states"].items() if v and v["why"])
        print("| %d | %.4f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["k"], r["J"], r["status"] + (" (" + why + ")" if why else ""), *(fmt(a[q]) if a else "n/a" for q in ("f_dE", "f_omega", "f_0")),
            "n/a" if not a else "%.4f, %.4f" % (a["rise_of_I_coarse"], a["rise_of_I_fine"]), "n/a" if not a else "%.4f to %.4f" % tuple(a["band_of_f_dE"]),
            *(fmt(b[q], "%.5f") if b else "n/a" for q in ("f_dE", "f_omega", "f_0")), "n/a" if not b else "%.4f, %.4f" % (b["rise_of_I_coarse"], b["rise_of_I_fine"])))
    r = P1["reading"]
    print("\nP1: J* = %s (J_e/%s): **%s**%s%s%s; pairs at or above 0.20 in both falls: %s of %d decided (%d left out); by arm: %s." % (
        fmt(r["J_star"]), r["k_star"], r["word"], (" (" + r["reason"] + ")") if r["reason"] else "", (", " + r["qualifier"]) if r["qualifier"] else "",
        (" (" + "; ".join(r["clauses_failed"]) + ")") if r["clauses_failed"] else "", r["pairs_at_or_above_in_both"], len(r["pairs"]), len(r["pairs_left_out"]),
        {k: (None if v is None else "J_e/%d" % v["k"]) for k, v in r["by_arm"].items()}))
    if r["pairs"]:
        print("\n| State at 0.375 | State at 0.25 | f_ΔE | f_ω | f₀ | Both survive |")
        print("| --- | --- | --- | --- | --- | --- |")
        for p in r["pairs"]:
            print("| `%s` | `%s` | %.4f | %.4f | %.4f | %s |" % (p["coarse"], p["fine"], p["f_dE"], p["f_omega"], p["f_0"], yn[p["both_survive"]]))
    if r["pairs_left_out"]:
        print("\nPairs left out (a state not decided at J*): " + "; ".join("`%s` with `%s`" % tuple(p) for p in r["pairs_left_out"]) + ".")
    if P1["participation_volume_ratios"]:
        print("\nThe ratio of V_I from spacing 0.375 to 0.25 on the pairs from the seed: " + "; ".join(
            "%s, J_e/%d: %.3f at J, %.3f on the references" % (v["arm"], v["k"], v["V_I_ratio_at_J"], v["V_I_ratio_on_the_references"]) for v in P1["participation_volume_ratios"]) + ".")


def figures(out, plots_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots_dir.mkdir(parents=True, exist_ok=True)
    P1, P2, P3 = out["P1"], out["P2"], out["P3"]
    have = [t for t in P1["states"] if t.get("dE") is not None]
    if have:
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for s, col, mk in (("h0375_L0", "#cf7a3a", "o"), ("h0250_L0", "#6e2206", "o"), ("h0375_L0.03", "#6aa3dd", "s"), ("h0250_L0.03", "#1f5fa8", "s")):
            pts = sorted((t for t in have if t["state"] == s), key=lambda t: t["J"])
            if not pts:
                continue
            lab = "%s, spacing %g" % ("without L₁" if pts[0]["L1"] == 0 else "L₁ = 0.03", pts[0]["spacing"])
            J = np.array([t["J"] for t in pts])
            ax[0].plot(J, [t["dE"] for t in pts], mk + "-", color=col, label=lab)
            ax[0].plot(J, [t["rigid_excess"] for t in pts], ":", color=col)
            ax[1].plot(J, [t["omega"] for t in pts], mk + "-", color=col, label=lab)
            for t in pts:
                if t["status"] != "survives":
                    ax[0].plot(t["J"], t["dE"], "x", color="k", ms=9)
                    ax[1].plot(t["J"], t["omega"], "x", color="k", ms=9)
        for q, yl in zip(ax, ("the excess E(J) − E(0); dotted: J²/2I₀′", "ω = J/I")):
            q.set_xscale("log")
            q.set_yscale("log")
            q.set_xlabel("J")
            q.set_ylabel(yl)
            q.grid(alpha=0.3)
            q.legend(fontsize=7)
        ax[0].set_title("δ = 0.05, half-box 6, states from the seed (a cross: not a surviving state)", fontsize=9)
        fig.tight_layout()
        fig.savefig(plots_dir / "m5_48_p1_excess_and_omega.png", dpi=140)
        plt.close(fig)
    cands = [r for r in P1["candidates"] if r["without L1"] or r["L1 = 0.03"]]
    if cands:
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for q, arm, lines in ((ax[0], "without L1", ((P1_FALL, "0.20, the author's threshold"), (P1_FALSIFIER, "0.05, the author's falsifier"))), (ax[1], "L1 = 0.03", ((P1_L1_CHANGE, "0.02, the author's threshold"),))):
            rows = [r for r in cands if r[arm]]
            J = np.array([r["J"] for r in rows])
            for key, col, lab in (("f_dE", "#b3471d", "the fall of the excess"), ("f_omega", "#1f5fa8", "the fall of ω")):
                y = np.array([r[arm][key] for r in rows])
                y = np.abs(y) if arm != "without L1" else y
                q.plot(J, y, "-", color=col, lw=1)
                for r, yy in zip(rows, y):
                    q.plot(r["J"], yy, "^" if r["status"] == "undecided" else "o", color=col, ms=8, mfc=col if r["status"] == "survives" else "none", mew=1.5)
                q.plot([], [], "-", color=col, label=lab)
            if arm == "without L1":
                for r in rows:
                    q.plot([r["J"]] * 2, r[arm]["band_of_f_dE"], "-", color="#b3471d", lw=5, alpha=0.2)
                q.plot(J, [r[arm]["f_0"] for r in rows], "k:", lw=1, label="f₀, the J = 0 arithmetic of M5.45")
            for v, lab in lines:
                q.axhline(v, color="k", lw=0.8, ls="--")
                q.annotate(lab, (J.min(), v), textcoords="offset points", xytext=(2, 3), fontsize=7)
            for mk, mfc, lab in (("o", "k", "the candidate survives"), ("o", "none", "the candidate fails"), ("^", "none", "the candidate is undecided")):
                q.plot([], [], mk, color="k", mfc=mfc, ms=7, mew=1.5, label=lab)
            q.set_xscale("log")
            q.set_xticks(J)
            q.set_xticklabels(["J_e/%d" % r["k"] for r in rows], fontsize=8)
            q.minorticks_off()
            q.set_xlabel("J")
            q.grid(alpha=0.3)
            if arm == "without L1":
                q.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.075))
            else:
                q.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(0.01, 0.72))
        ax[0].set_ylabel("fall from spacing 0.375 to 0.25")
        ax[0].set_title("Without L₁, the pair from the seed (shaded: the band the rises of I leave)", fontsize=9)
        ax[1].set_yscale("log")
        ax[1].set_ylabel("absolute value of the fall")
        ax[1].set_title("L₁ = 0.03, the pair from the seed", fontsize=9)
        fig.tight_layout()
        fig.savefig(plots_dir / "m5_48_p1_falls.png", dpi=140)
        plt.close(fig)
    if P2.get("finished"):
        t = np.array([q[:3] for q in P2["trajectory"]], float)
        fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.2))
        ax[0].plot(t[:, 0], t[:, 2] * 0.1875, color="#6e2206", lw=2)
        ax[0].axvline(27200, color="k", ls=":", lw=1)
        ax[0].axhspan(P2_IH[0], P2_IH[1], color="#1f5fa8", alpha=0.12, label="the author's window, I·h from 1000 to 1300")
        ax[0].set_xlabel("iteration (dotted: the cut of M5.45)")
        ax[0].set_ylabel("I·h")
        ax[0].legend(fontsize=8)
        tail = t[t[:, 0] >= 20000]
        ax[1].plot(tail[:, 0], tail[:, 2] * 0.1875, color="#6e2206", lw=2)
        ax[1].axvline(27200, color="k", ls=":", lw=1)
        ax[1].axhline(P2_IH[0], color="#1f5fa8", lw=1.5, label="I·h = 1000")
        ax[1].axhline(P2_I[0] * 0.1875, color="#1f5fa8", lw=1, ls="--", label="I = 5300")
        ax[1].set_xlabel("iteration, from 20000")
        ax[1].set_ylabel("I·h")
        ax[1].legend(fontsize=8)
        ax[2].plot(t[:, 0], t[:, 1], color="#2f7d4f", lw=2)
        ax[2].axvline(27200, color="k", ls=":", lw=1)
        ax[2].axhspan(P2_U_CENTER * (1 - P2_U_REL), P2_U_CENTER * (1 + P2_U_REL), color="#1f5fa8", alpha=0.12, label="U within 3 percent of 14.69")
        ax[2].set_ylim(14.0, 16.0)
        ax[2].set_xlabel("iteration")
        ax[2].set_ylabel("U")
        ax[2].legend(fontsize=8)
        for q in ax:
            q.grid(alpha=0.3)
        ax[0].set_title("Spacing 0.1875 without L₁, from the seed: I·h along the descent", fontsize=10)
        ax[2].set_title("U along the same descent", fontsize=10)
        fig.tight_layout()
        fig.savefig(plots_dir / "m5_48_p2_inertia_along_the_chain.png", dpi=140)
        plt.close(fig)
    sec = out.get("sections")
    if sec:
        fig, axs = plt.subplots(2, 3, figsize=(13.5, 8.6), squeeze=False)
        for i, row in enumerate(sec):
            for j, pan in enumerate(row):
                q = axs[i][j]
                d = np.array(pan["inertia_per_node"])
                im = q.imshow(np.log10(np.maximum(d.T, 1e-12)), origin="lower", extent=[-6, 6, -6, 6], cmap="magma", vmin=pan["vmin"], vmax=pan["vmax"])
                q.set_title(pan["title"], fontsize=8)
                q.set_xlabel("x")
                q.set_ylabel("y")
                fig.colorbar(im, ax=q, fraction=0.046, pad=0.04)
        fig.suptitle("log₁₀ of the inertia per node on a plane of constant z (spacing 0.375, states from the seed)", fontsize=10)
        fig.tight_layout()
        fig.savefig(plots_dir / "m5_48_inertia_sections.png", dpi=140)
        plt.close(fig)
    if P3["states"]:
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for s in P3["states"]:
            seed = s["start"] == "seed"
            ax[0].plot(s["spacing"], s["I"], "o" if seed else "s", color="#2f7d4f", ms=9 if seed else 6, mfc="#2f7d4f" if seed else "none", mew=1.5)
            if not s["converged"]:
                ax[0].annotate("not converged", (s["spacing"], s["I"]), textcoords="offset points", xytext=(6, 4), fontsize=7)
            if seed:
                t = np.array([q[:3] for q in s["trajectory"]], float)
                if len(t):
                    ax[1].plot(t[:, 0], t[:, 2], lw=2, label="spacing %g" % s["spacing"])
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_xticks([0.1875, 0.25, 0.375, 0.5])
        ax[0].set_xticklabels(["0.1875", "0.25", "0.375", "0.5"])
        ax[0].minorticks_off()
        ax[0].set_xlabel("spacing")
        ax[0].set_ylabel("inertia I")
        ax[0].set_title("δ = 0.3 without L₁, half-box 6 (filled: from the seed; open: carried)", fontsize=10)
        ax[1].set_xscale("log")
        ax[1].set_xlabel("iteration")
        ax[1].set_ylabel("inertia I")
        ax[1].set_title("From the seed: the inertia along the descent", fontsize=10)
        ax[1].legend(fontsize=8)
        for q in ax:
            q.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "m5_48_p3_inertia_against_spacing.png", dpi=140)
        plt.close(fig)


def sections(P1, work_dir, out_dir, coarse=("h0375_L0", "h0375_L0.03")):
    """For the two states from the seed at spacing 0.375: the inertia per node on one plane of constant z, for the reference, the
    state at J* (at J_e/320 if there is no J*) and the state at the largest candidate. The plane is the one through the node
    that holds the most inertia at the largest candidate. Arrays for the figure only; they do not enter the JSON."""
    kstar = P1["reading"]["k_star"] or 320
    rows = []
    for s in coarse:
        m = D.P1_STATES[s]
        mod = model(m["N"], 0.05, m["L1"])
        recs = [(f"p1_{s}_j{kstar}_ref", "the reference at J = 0"), (f"p1_{s}_j{kstar}", "at J_e/%d" % kstar), (f"p1_{s}_j{min(D.P1_K)}", "at J_e/%d" % min(D.P1_K))]
        dens = []
        for name, tag in recs:
            rec = load_rec(work_dir, name)
            if rec is None:
                dens.append(None)
                continue
            mod.evaluate(np.load(out_dir / rec["end_file"], allow_pickle=False)["m6"].reshape(-1), 0.0, need_grad=False, need_I=True)
            dens.append((mod.h ** 3 * mod.idn.copy(), tag, rec["stops"][-1]["I"], rec["stops"][-1]["label"]))
        if any(d is None for d in dens):
            return None
        iz = int(np.unravel_index(int(np.argmax(dens[-1][0])), dens[-1][0].shape)[2])
        vmax = float(np.log10(max(d[0][:, :, iz].max() for d in dens)))
        rows.append([dict(inertia_per_node=d[:, :, iz].tolist(), vmin=vmax - 6, vmax=vmax,
                          title="%s, %s\nI = %.1f, %s, z = %.3f" % ("without L₁" if m["L1"] == 0 else "L₁ = 0.03", tag, I, lab, mod.x[iz])) for d, tag, I, lab in dens])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", action="store_true", help="the controls only: writes data/m5_48_controls.json")
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_48" / "chains")
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--plots-dir", type=Path, default=HERE.parent / "plots")
    a = ap.parse_args()
    K = controls(a.work_dir, a.out_dir)
    D.out_file(a.out_dir, "m5_48_controls.json").write_text(json.dumps(dict(task="M5.48", controls=K), indent=1) + "\n")
    if a.controls:
        for k, v in K.items():
            print(k, {True: "MET", False: "NOT MET", None: "not read yet"}[v["met"]], "-", v["what"])
        return
    out = dict(task="M5.48", controls={k: dict(met=v["met"], what=v["what"]) for k, v in K.items()},
               thresholds=dict(S2_U=S2_U, S3_share=S3_SHARE, S4_fraction=S4_FRACTION, S4_floor_I=S4_FLOOR_I, S4_floor_U=S4_FLOOR_U, S5_radius=S5_RADIUS, P1_fall=P1_FALL, P1_L1_change=P1_L1_CHANGE,
                               P1_falsifier=P1_FALSIFIER, P2_I_times_h=list(P2_IH), P2_I=list(P2_I), P2_U_center=P2_U_CENTER, P2_U_relative=P2_U_REL, P3_G=P3_G, P3_S=P3_S, P3_G_as=P3_G_AS, P3_S_as=P3_S_AS),
               P1=p1(a.work_dir, a.out_dir), P2=p2(a.work_dir, a.out_dir), P3=p3(a.work_dir, a.out_dir))
    out["chains"] = chains_table(a.work_dir)
    D.out_file(a.out_dir, "m5_48_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print_tables(out)
    out["sections"] = sections(out["P1"], a.work_dir, a.out_dir)
    figures(out, a.plots_dir)


if __name__ == "__main__":
    main()
