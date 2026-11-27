"""M5.57: the words of sections 3 to 5, the controls C5 to C8 and C10 as read, the level rule and M_c, the readings of
the merged end, the scores of OW57-1 and OW57-2 (strict, and as written before the amendment) and of X1 to X6, the
summary JSON and the plots.

Every word is given by the frozen rule as written; this script reads the chain records of m5_57_relax.py
(checkpoints/m5_57/chains/), the control records of m5_57_hold.py and the stored ends of M5.56, and computes nothing a
rule does not name, besides the descriptors it prints.

Usage:
  python m5_57_reads.py --summary       data/m5_57_summary.json and the plots
  python m5_57_reads.py --status        a short status of every row (no file written)
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
import m5_51_reads as RD  # noqa: E402
import m5_56_turn as T  # noqa: E402
import m5_57_hold as H  # noqa: E402
import m5_57_relax as X  # noqa: E402

CK = H.CK
WORK = X.WORK
DATA = HERE.parent / "data"
PLOTS = HERE.parent / "plots"
PI = np.pi
E_UNIT_HB12 = 19.497928342958982
MARGIN = 1e-4


def rec(n, k5=False):
    return X.rec(n, k5)


def k5_of(n):
    k = rec(n, k5=True)
    if k is None:
        return dict(state="not run")
    st = X.k5_state(n)
    return dict(state=st, iterations=k["iterations"], end_state=k["end_state"], checks=k["k5"]["checks"],
                values={a: (b if not isinstance(b, dict) else b.get("mu")) for a, b in k["k5"]["values"].items()})


def stored_k5(row):
    """K5 of a stored end of M5.56 (rows 30, 31, 33): met under M5.56's strict reading (CONVERGED and passed)."""
    d = json.loads((H.CK56 / f"k5_{H.STORED[row]}.json").read_text())
    return "met" if (d["k5"]["passed"] and d["end_state"] == "CONVERGED") else "not met"


def summ(n):
    """The reads of one row's end."""
    r = rec(n)
    out = dict(row=n, name=X.cname(n), grid=X.ROWS[n]["grid"], kind=X.ROWS[n]["kind"])
    if r is None:
        out["state"] = "not finished"
        return out
    if r.get("end_state") == "NOT RUN":
        out["state"] = r.get("reason")
        return out
    e = r["end"]
    out.update(state=r["end_state"], iterations=r["iterations"], message=r["message"], restarts=len(r["restarts"]), resumes=len(r["resumes"]),
               U=e["U"], E=e["E"], gate=e["gate"], gate_threshold=e["gate_threshold"], torque=e["torque"], Dprime=e["Dprime"],
               converged_by=[k for k in ("rule_a", "rule_b", "rule_c") if r["converged_check"].get(k)], target=r.get("target"), mu=r.get("mu"))
    if "f" in e:
        out.update(f=e["f"], hold_energy=e["hold_energy"], F=e["F"])
    if "Dprime_plus" in e:
        out.update(Dprime_plus=e["Dprime_plus"], Dprime_minus=e["Dprime_minus"], offset=e["offset"])
    out["dprime_regime"] = dict(in_regime=e["dprime_regime"]["in_regime"], share=e["dprime_regime"]["share"],
                                least_g23=e["dprime_regime"]["least_g23_where_w_gt_1e6"], components=e["dprime_regime"]["components"])
    if "pair" in e:
        p = e["pair"]
        out.update(word=p["word"], components=p["components"], sizes=p["sizes"], s=p.get("separation"), circle_distance=p.get("circle_distance"))
        if p.get("rings"):
            out["rings"] = [dict(c=q["c"], RL=q["RL"], nu=q["nu"], regime=q["regime"], one_ring=q.get("one_ring"), cube_q=(q.get("cube_q") or {}).get("abs_q"),
                                 cls=q.get("cls")) for q in p["rings"]]
        if p.get("one_object"):
            out["one_object"] = p["one_object"]
        out["g_faces"] = e.get("g_faces")
        dm = e["dmoment"]
        out["D"] = dict(D=dm["D"], Dp=dm["Dp"], Dm=dm["Dm"], Qp=dm["Qp"], Qm=dm["Qm"], dc=dm["dc"], in_regime=dm["in_regime"])
    else:
        out.update(RL=e.get("RL"), center=e.get("center"), normal=e.get("normal"), one_ring=e.get("one_ring"), line_regime=e.get("line_regime"))
    ss = r.get("second_step") or {}
    out["second_step"] = dict(source=ss.get("source"), ok=X.second_step_ok(n), checks=ss.get("checks"),
                              values={a: (b if not isinstance(b, dict) else b.get("mu")) for a, b in (ss.get("values") or {}).items()})
    out["K5"] = k5_of(n)
    if X.ROWS[n]["grid"] != "H1" and X.ROWS[n]["kind"] == "held":
        out.update(in_hold=X.in_hold(n), evaded=X.evaded(n), usable=X.usable(n))
    return out


def gives_word(n):
    """A pair word is given only on a CONVERGED end whose second step is met."""
    r = rec(n)
    return bool(r is not None and r.get("end_state") == "CONVERGED" and X.second_step_ok(n))


# ----------------------------------------------------------------------------- rung A

def rungA():
    rows = {n: summ(n) for n in (1, 2, 3, 4, 5)}
    for n in rows:
        r = rows[n]
        if r.get("g_faces"):
            r["g_mean"] = float(np.mean(r["g_faces"]))
        r["qualifies"] = bool(gives_word(n) and r["K5"]["state"] == "met")
    r30 = H.load56(30)[2]["end"]["pair"]
    r31 = H.load56(31)[2]["end"]["pair"]
    s18 = 0.5 * (r30["separation"] + r31["separation"])
    g18 = float(np.mean([18.0 - abs(q["c"][0]) for q in r30["rings"] + r31["rings"]]))
    boxes = {18: dict(counts=True, word="TWO RINGS", s=s18, g=g18, rows="M5.56 rows 30, 31 (record)", K5=[stored_k5(30), stored_k5(31)])}
    for L, (a, b) in ((24, (1, 2)), (30, (4, 5))):
        ra, rb = rows[a], rows[b]
        d = dict(rows=[a, b])
        if ra["qualifies"] and rb["qualifies"]:
            wa, wb = ra["word"], rb["word"]
            dU = abs(ra["U"] / rb["U"] - 1)
            d["dU_rel"] = dU
            if wa == wb == "TWO RINGS":
                d["ds"] = abs(ra["s"] - rb["s"])
                d["counts"] = bool(d["ds"] <= 0.05 and dU <= 1e-6)
            elif wa == wb == "ONE OBJECT":
                d["counts"] = bool(dU <= 1e-6)
            else:
                d["counts"] = False
            d["word"] = wa if d["counts"] else None
            if d["counts"] and wa == "TWO RINGS":
                d["s"] = 0.5 * (ra["s"] + rb["s"])
                d["g"] = 0.5 * (ra["g_mean"] + rb["g_mean"])
            if not d["counts"]:
                d["why"] = "the two starts disagree"
        else:
            d["counts"] = False
            d["why"] = "a main start does not end CONVERGED with its second step and K5 met"
        boxes[L] = d
    b24, b30 = boxes[24], boxes[30]
    word, follows = "no label", False
    why = ""
    if all(boxes[L]["counts"] and boxes[L]["word"] == "TWO RINGS" for L in (18, 24, 30)):
        d1, d2 = b24["s"] - s18, b30["s"] - b24["s"]
        if d1 > 1.0 and d2 > 1.0 and d2 >= 0.5 * d1:
            word = "KEEPS SEPARATING"
            follows = bool(abs(b24["g"] - 5.237) <= 0.5 and abs(b30["g"] - 5.237) <= 0.5)
        elif abs(d2) <= 1.0:
            word = "STOPS"
        else:
            why = f"s24 - s18 = {d1:.4f}, s30 - s24 = {d2:.4f}: neither KEEPS SEPARATING nor STOPS"
    elif b24["counts"] and b30["counts"] and b24["word"] == b30["word"] == "ONE OBJECT":
        word = "MERGES"
    else:
        why = "a box does not count, or the boxes carry mixed words"
    # the face control A3
    a3 = rows[3]
    ctl = dict(converged=bool(a3.get("state") == "CONVERGED"), word=a3.get("word"), s=a3.get("s"))
    if a3.get("state") == "CONVERGED" and b24.get("counts"):
        moved = (a3.get("word") != b24["word"]) or (a3.get("word") == "TWO RINGS" and abs(a3["s"] - b24["s"]) > 1.0)
        ctl["ds_to_box24"] = None if a3.get("s") is None or b24.get("s") is None else abs(a3["s"] - b24["s"])
        ctl["moves_the_word"] = bool(moved)
        if moved and word != "no label":
            why = "the far field moves the separation"
            word = "no label"
            follows = False
    elif a3.get("state") != "CONVERGED":
        ctl["note"] = "A3 not CONVERGED: the word stands, printed beside"
    return dict(rows=rows, boxes=boxes, word=word, follows_the_box=follows, why=why, face_control=ctl, s18=s18, g18=g18)


# ----------------------------------------------------------------------------- rung B

def ladder_table(spacing, which):
    rows = X.ladder_rows(spacing, which)
    out = []
    bad = 0
    capped_before = False
    for n in rows:
        s = summ(n)
        s["target"] = X.ROWS[n]["target"]
        s["after_a_capped_rung"] = capped_before
        if spacing == 0.375 and bad >= 2 and s.get("state") not in (None, "not finished"):
            s["post_hoc"] = "not run: the ladder broke (two consecutive children not CONVERGED before it; read after the fact)"
        r = rec(n)
        if r is not None:
            conv = r.get("end_state") == "CONVERGED"
            if r.get("end_state") != "NOT RUN":
                bad = bad + 1 if not conv else 0
                if not conv:
                    capped_before = True
            elif spacing == 0.375:
                pass
        out.append(s)
    return out


def rungB(spacing):
    fe, me = X.free_end(spacing), X.merged_end(spacing)
    if spacing == 0.5:
        refs = dict(free=dict(fe, state="CONVERGED", K5=stored_k5(30), second_step=True),
                    merged=dict(me, state="CONVERGED", K5=stored_k5(33), second_step=True))
    else:
        refs = {}
        for lab, n, d in (("free", 6, fe), ("merged", 7, me)):
            s = summ(n)
            refs[lab] = dict(d or {}, state=s.get("state"), K5=s.get("K5", {}).get("state"), second_step=s.get("second_step", {}).get("ok"),
                             s=s.get("s"), word=s.get("word"), iterations=s.get("iterations"), gate=s.get("gate"))
    tabs = dict(inward=ladder_table(spacing, "in"), outward=ladder_table(spacing, "out"))
    rel = 0.05 if spacing == 0.5 else 0.10
    usable_ = {}
    for lab, tb in list(tabs.items()):
        for s in tb:
            s["usable_here"] = bool(s.get("usable") and not s.get("post_hoc"))
            s["in_hold_here"] = bool(s.get("in_hold") and not s.get("post_hoc"))
        # step identity between neighboring rungs in hold that read TWO RINGS and are not EVADED
        q = [s for s in tb if s["in_hold_here"] and s.get("word") == "TWO RINGS" and not s.get("evaded")]
        steps = []
        for a, b in zip(q[:-1], q[1:]):
            si = X.step_identity(a["U"], b["U"], a["f"], b["f"], a["Dprime"], b["Dprime"], rel, 2e-4)
            si.update(targets=[a["target"], b["target"]], jump=not si["passes"])
            steps.append(si)
        tabs[lab + "_steps"] = steps
        for s in tb:
            if s["usable_here"]:
                usable_.setdefault(s["target"], []).append(s)
    # branches
    branches = []
    tin = {s["target"]: s for s in tabs["inward"] if s["in_hold_here"]}
    tout = {s["target"]: s for s in tabs["outward"] if s["in_hold_here"]}
    for t in sorted(set(tin) & set(tout)):
        dU = tin[t]["U"] - tout[t]["U"]
        if abs(dU) > 1e-4:
            low = tin[t] if dU < 0 else tout[t]
            branches.append(dict(target=t, U_inward=tin[t]["U"], U_outward=tout[t]["U"], words=[tin[t].get("word"), tout[t].get("word")],
                                 least=low["name"]))
    out = dict(spacing=spacing, references=refs, tables=tabs, branches=branches)
    # the guard on the references
    guard_ok = bool(fe is not None and me is not None and refs["free"].get("word") == "TWO RINGS" and refs["merged"].get("word") == "ONE OBJECT")
    ref_states_ok = bool(all(refs[k].get("state") == "CONVERGED" and refs[k].get("second_step") for k in ("free", "merged")))
    out["guard"] = dict(references_are_what_they_are_called=guard_ok, references_converged_with_second_step=ref_states_ok)
    if fe is None or me is None:
        out["Mc"] = dict(label="no label", why="a reference has no end")
        return out
    U_f, U_m, D_f, D_m = fe["U"], me["U"], fe["Dprime"], me["Dprime"]
    targets = sorted({X.ROWS[n]["target"] for n in X.ladder_rows(spacing, "in") + X.ladder_rows(spacing, "out")})
    rng = [t for t in targets if D_m < t < D_f]
    E_low = {t: (min(usable_[t], key=lambda s: s["U"]) if t in usable_ else None) for t in targets}
    us_all = [s["U"] for lst in usable_.values() for s in lst]
    E_min = min([U_f, U_m] + us_all)
    E_ring = 0.5 * U_f
    lv = []
    for t in rng:
        b = E_low[t]
        lv.append(dict(target=t, E_low=None if b is None else b["U"], rung=None if b is None else b["name"], row=None if b is None else b["row"],
                       ladder=None if b is None else ("inward" if b["row"] in X.ladder_rows(spacing, "in") else "outward"),
                       word=None if b is None else b.get("word")))
    # skipped targets and the no-label rule of two neighboring empty targets
    ext = [dict(target=D_m, E_low=U_m, end="merged")] + lv + [dict(target=D_f, E_low=U_f, end="free")]
    skipped, two_empty = [], False
    for i in range(1, len(ext) - 1):
        if ext[i]["E_low"] is None:
            if ext[i - 1]["E_low"] is not None and ext[i + 1]["E_low"] is not None:
                skipped.append(ext[i]["target"])
            else:
                two_empty = True
    have = [d for d in lv if d["E_low"] is not None]
    E_star = max(have, key=lambda d: d["E_low"]) if have else None
    top = max([U_m] + [d["E_low"] for d in have])
    Mc = (top - E_min) / E_ring
    at_merged = bool(E_star is None or E_star["E_low"] <= U_m)
    if at_merged:
        k5_max = refs["merged"]["K5"]
        max_holder = dict(kind="the merged end", name=me["name"], K5=k5_max)
    else:
        k5_max = X.k5_state(E_star["row"])
        max_holder = dict(kind="rung", name=E_star["rung"], target=E_star["target"], ladder=E_star["ladder"], word=E_star["word"], K5=k5_max)
    why = []
    if two_empty:
        why.append("two neighboring targets hold no usable rung")
    if k5_max != "met":
        why.append(f"the rung or end of the maximum lacks K5 met ({k5_max})")
    if not guard_ok:
        why.append("a reference does not read what it is called (the amendment's guard)")
    if not ref_states_ok:
        why.append("a reference is not CONVERGED with its second step met (this task's stricter reading)")
    # with no level of the range holding a rung, no interior word is read and M_c has no value (the reading of the
    # findings, section 15; read literally, the "otherwise" of section 4 would print NO INTERIOR MAXIMUM)
    if not have:
        interior_word = None
    else:
        interior_word = "INTERIOR MAXIMUM" if E_star["E_low"] > U_m + MARGIN else "NO INTERIOR MAXIMUM"
    out["levels"] = lv
    out["Mc"] = dict(value=Mc if have else None, label="no label" if why else "number", why=why, E_min=E_min, E_ring=E_ring, E_ring_hb12=E_UNIT_HB12,
                     Mc_in_hb12_unit=(top - E_min) / E_UNIT_HB12 if have else None, literal_interior_word_on_an_empty_range="NO INTERIOR MAXIMUM" if not have else None, top=top, U_free=U_f, U_merged=U_m, Dprime_free=D_f, Dprime_merged=D_m,
                     range=rng, skipped=skipped, max_holder=max_holder, interior_word=interior_word,
                     E_star=None if E_star is None else dict(E_star))
    # the merged end: two readings
    if spacing == 0.5:
        r1 = "A LOCAL MINIMUM" if stored_k5(33) == "met" else "NOT A LOCAL MINIMUM"
    else:
        s7 = summ(7)
        k = s7.get("K5", {})
        if s7.get("state") == "CONVERGED" and k.get("state") == "met":
            r1 = "A LOCAL MINIMUM"
        elif k.get("end_state") == "CONVERGED" and k.get("state") != "met":
            r1 = "NOT A LOCAL MINIMUM"
        else:
            r1 = "no label"
    above = [t for t in targets if t > D_m]
    below = [t for t in targets if t < D_m]
    ta, tb = (min(above) if above else None), (max(below) if below else None)
    ea = E_low.get(ta) if ta is not None else None
    eb = E_low.get(tb) if tb is not None else None
    if ea is None or eb is None:
        r2 = "no label"
    elif ea["U"] > U_m + MARGIN and eb["U"] > U_m + MARGIN:
        r2 = "A HELD MINIMUM"
    elif ea["U"] <= U_m or eb["U"] <= U_m:
        r2 = "NOT A HELD MINIMUM"
    else:
        r2 = "no label"
    if not guard_ok:
        r1 = r2 = "no label"
    out["merged_end"] = dict(reading_i=r1, reading_ii=r2, targets_next=[ta, tb], E_low_next=[None if ea is None else ea["U"], None if eb is None else eb["U"]])
    # strict score, per spacing
    lab = out["Mc"]["label"] == "number"
    if E_star is not None:
        dm, df = E_star["E_low"] - U_m, E_star["E_low"] - U_f
    else:
        dm = df = None
    if lab and dm is not None and dm > MARGIN and df > MARGIN:
        interior = "a maximum before the merged end"
    elif lab and dm is not None and (dm < -MARGIN or df < -MARGIN):
        interior = "none"
    else:
        interior = "unresolved"
    if interior == "a maximum before the merged end" and Mc < 0.1 and r1 == "A LOCAL MINIMUM":
        strict = "met"
    elif interior == "none" or (lab and Mc >= 0.1) or r1 == "NOT A LOCAL MINIMUM":
        strict = "not met"
    else:
        strict = "no word"
    if lab and Mc < 0.1 and r1 == "A LOCAL MINIMUM":
        written = "met"
    elif (lab and Mc >= 0.1) or r1 == "NOT A LOCAL MINIMUM":
        written = "not met"
    else:
        written = "no word"
    out["score"] = dict(interior=interior, Delta_m=dm, Delta_f=df, strict=strict, as_written=written)
    # releases
    rels = {}
    for j in ((62, 64, 65) if spacing == 0.5 else (63, 66, 67)):
        s = summ(j)
        r = rec(j)
        rels[X.cname(j)] = dict(state=s.get("state"), source=(r or {}).get("info", {}).get("source"), U=s.get("U"), word=s.get("word"), s=s.get("s"),
                                second_step=s.get("second_step", {}).get("ok"))
    out["releases"] = rels
    return out


# ----------------------------------------------------------------------------- rung C

def rungC():
    out = dict(decisions={k: v for k, v in X.decisions().items() if int(k) >= 68}, chains=[])
    for j in (68, 69, 70):
        r = rec(j)
        d = dict(row=j, name=X.cname(j))
        if r is None:
            d["state"] = "not finished"
            out["chains"].append(d)
            continue
        prep = r.get("prep") or {}
        d["reference"] = r.get("reference")
        d["prep"] = dict(collar=prep.get("collar"), C10=prep.get("C10"), start_rule=prep.get("start_rule"),
                         paths={k: dict(word=v["word"], largest_rise=v["largest_rise"], dU=[q["dU"] for q in v["rows"]]) for k, v in (prep.get("paths") or {}).items()})
        if r.get("end_state") == "NOT RUN":
            d["state"] = r.get("reason")
            d["word"] = r.get("reason")
            out["chains"].append(d)
            continue
        refn = X.decisions()[str(j)]["ref"]
        e = r["end"]
        er = r.get("earlier")
        s = summ(j)
        d.update(state=r["end_state"], iterations=r["iterations"], U=e["U"], U_ref=r["U_ref"], dU=e["U"] - r["U_ref"], dU_start=r["stops"][0]["U"] - r["U_ref"],
                 gate=e["gate"], word_pair=e["pair"]["word"], turn=dict(sigma=e["turn"]["sigma"], chi_max=e["turn"]["chi_max"], in_regime=e["turn"]["in_regime"],
                 q0_sign_shell=e["turn"]["q0_sign_shell"], rho_t=e["turn"]["rho_t_cross"]), mu_free=e["mu_free"], K5=s["K5"], ref_K5=X.k5_state(refn))
        ss = s["second_step"]["ok"]
        if er is not None and "turn" in er:
            ss = bool(ss and er["turn"]["sigma"] == e["turn"]["sigma"] and all((e["turn"]["chi_max"] <= b) == (er["turn"]["chi_max"] <= b) for b in (0.05, PI / 2)))
        d["second_step"] = ss
        why = []
        if r["end_state"] != "CONVERGED":
            why.append("CAPPED")
        if not ss:
            why.append("the second step")
        if s["K5"]["state"] != "met":
            why.append(f"K5 {s['K5']['state']}")
        if d["ref_K5"] != "met":
            why.append(f"the reference's K5 {d['ref_K5']}")
        if not e["turn"]["in_regime"]:
            why.append("the turn reader out of regime at the end")
        ref = rec(refn)["end"]
        cl = {}
        cl["U"] = bool(abs(e["U"] / r["U_ref"] - 1) <= 1e-5)
        cl["two_rings"] = bool(e["pair"]["word"] == "TWO RINGS")
        if cl["two_rings"]:
            a, b = e["pair"]["rings"], ref["pair"]["rings"]
            cl["RL"] = bool(all(abs(p["RL"] - q["RL"]) <= 0.01 for p, q in zip(a, b)))
            cl["center"] = bool(all(np.linalg.norm(np.asarray(p["c"]) - np.asarray(q["c"])) <= 0.05 for p, q in zip(a, b)))
            cl["class"] = bool(all(p["cls"] == q["cls"] for p, q in zip(a, b)))
            cl["degree"] = bool(all((p.get("cube_q") or {}).get("abs_q") == (q.get("cube_q") or {}).get("abs_q") for p, q in zip(a, b))
                                and (e["pair"].get("box_q") or {}).get("abs_q") == (ref["pair"].get("box_q") or {}).get("abs_q"))
        cl["mu_free"] = bool(e["mu_free"]["in_regime"] and e["mu_free"]["mu"] <= 0.03)
        agree = all(cl.values())
        d["readers_agree"] = dict(agree=agree, clauses=cl)
        tears = [st["iteration"] for st in r["stops"] if st.get("tear", {}).get("tear")]
        d["tear_stops"] = tears
        if why:
            w = "no label"
        elif agree and e["turn"]["sigma"] == 1 and e["turn"]["chi_max"] <= 0.05:
            w = "RELEASED"
        else:
            held = bool(er is not None and all(q["turn"]["chi_max"] >= PI / 2 and q["turn"]["sigma"] == -1 for q in (e, er))
                        and all(q["tear"]["z_nodes_outside"] == 0 and q["tear"]["jump_links"] == 0 for q in (e, er)))
            w = "HELD" if held else ("TORN" if tears else "MOVED")
        d["word"] = w
        d["why_no_label"] = why
        mono = all(v["word"] == "MONOTONE" for v in d["prep"]["paths"].values()) if d["prep"]["paths"] else False
        sep = rec(refn)["end"]["pair"]["separation"]
        d["separation"] = sep
        d["line"] = ("the turned class is not a state at this separation under the hold: the turn falls away" if (mono and w == "RELEASED") else "tabled")
        out["chains"].append(d)
    return out


# ----------------------------------------------------------------------------- controls, scores, expectations

def controls():
    out = {}
    for nm in ("C1", "C2", "C3", "C4", "C9"):
        tries = sorted(CK.glob(f"control_{nm}_try*.json"))
        if tries:
            last = json.loads(tries[-1].read_text())
            out[nm] = dict(met=last["passed"], tries=len(tries), file=tries[-1].name)
    out["C5"] = X.c5_result()
    out["C6"] = X.c6_result()
    out["C7"] = {str(s): X.c7_result(s) for s in (0.5, 0.375)}
    out["C8"] = {str(s): X.c8_result(s) for s in (0.5, 0.375)}
    c10 = []
    for j in (68, 69, 70):
        r = rec(j)
        if r is not None and r.get("prep", {}).get("C10"):
            c10.append(dict(row=j, reference=r.get("reference"), **{k: v for k, v in r["prep"]["C10"].items()}))
    out["C10"] = c10
    return out


def scores(A_, B, C_, ctl):
    ow1 = {"KEEPS SEPARATING": "met", "STOPS": "not met", "MERGES": "not met"}.get(A_["word"], "not scored")
    s5, s375 = B["0.5"]["score"], B["0.375"]["score"]

    def comb(a, b):
        if a == "met" and b == "met":
            return "met"
        if a in ("met", "not met") and b in ("met", "not met") and a == "not met" and b == "not met":
            return "not met"
        return "not scored"
    ow2 = dict(strict=dict(spacing_05=s5["strict"], spacing_0375=s375["strict"], together=comb(s5["strict"], s375["strict"])),
               as_written=dict(spacing_05=s5["as_written"], spacing_0375=s375["as_written"], together=comb(s5["as_written"], s375["as_written"])),
               limit_9=bool(A_["word"] == "KEEPS SEPARATING"),
               skipped={sp: B[sp]["Mc"].get("skipped") for sp in ("0.5", "0.375")})
    ex = []
    ex.append(dict(id="X1", outcome={"KEEPS SEPARATING": "right", "STOPS": "wrong", "MERGES": "wrong"}.get(A_["word"], "not scored")))
    fc = A_["face_control"]
    b24 = A_["boxes"][24]
    if fc.get("converged") and b24.get("counts") and b24.get("s") is not None and fc.get("s") is not None:
        x2 = "right" if (fc["word"] == "TWO RINGS" and abs(fc["s"] - b24["s"]) <= 1.0) else "wrong"
    elif fc.get("converged") and b24.get("counts") and fc.get("word") != b24.get("word"):
        x2 = "wrong"
    else:
        x2 = "not scored"
    ex.append(dict(id="X2", outcome=x2))
    c5, c6 = ctl["C5"]["met"], ctl["C6"]["met"]
    ex.append(dict(id="X3", outcome="right" if (c5 and c6 and ctl.get("C5_tries", 1) == 1 and ctl.get("C6_tries", 1) == 1) else ("wrong" if (c5 is not None and c6 is not None) else "not scored")))
    tb = B["0.5"]["tables"]["inward"]
    one = [s for s in tb if s.get("word") == "ONE OBJECT" and gives_word(s["row"])]
    if one:
        x4 = "right"
    elif all((s.get("word") == "TWO RINGS" and gives_word(s["row"])) or s.get("evaded") for s in tb):
        x4 = "wrong"
    else:
        x4 = "not scored"
    ex.append(dict(id="X4", outcome=x4))
    mc = B["0.5"]["Mc"]
    if mc.get("label") == "number":
        x5 = "right" if mc["interior_word"] == "INTERIOR MAXIMUM" else "wrong"
    else:
        x5 = "not scored"
    ex.append(dict(id="X5", outcome=x5))
    ran = [c for c in C_["chains"] if c.get("state") in ("CONVERGED", "CAPPED")]
    if not ran:
        x6 = "not scored"
    else:
        allmono = all(all(v["word"] == "MONOTONE" for v in c["prep"]["paths"].values()) for c in C_["chains"] if c.get("prep", {}).get("paths"))
        allrel = all(c.get("word") == "RELEASED" for c in ran)
        x6 = "right" if (allmono and allrel) else "wrong"
    ex.append(dict(id="X6", outcome=x6))
    return dict(OW57_1=ow1, OW57_2=ow2, expectations=ex)


def summary(write=True):
    ctl = controls()
    A_ = rungA()
    B = {str(s): rungB(s) for s in (0.5, 0.375)}
    C_ = rungC()
    S = scores(A_, B, C_, ctl)
    out = dict(task="M5.57", controls=ctl, rungA=A_, rungB=B, rungC=C_, scores=S,
               rows={X.cname(n): summ(n) for n in X.ROWS if n not in (1, 2, 3, 4, 5)})
    if write:
        DATA.mkdir(exist_ok=True)
        (DATA / "m5_57_summary.json").write_text(json.dumps(out, indent=1, default=T._jsonable) + "\n")
        plots(A_, B, C_)
    return out


# ----------------------------------------------------------------------------- plots

def plots(A_, B, C_):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                         "grid.linewidth": 0.6})
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#c94f7c", "#7a7a75"]
    # 1. rung A: s and the face distance against the half-box
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    L = [18, 24, 30]
    for n, mk in ((1, "o"), (2, "s"), (3, "^"), (4, "o"), (5, "s")):
        r = A_["rows"][n]
        if r.get("s") is None:
            continue
        Lb = 24 if n <= 3 else 30
        lab = f"{r['name']} ({r.get('state')}, {r.get('word')})"
        axs[0].plot([Lb], [r["s"]], mk, color=cols[n - 1], label=lab, ms=5, mfc="none" if r.get("state") != "CONVERGED" else cols[n - 1])
        if r.get("g_mean") is not None:
            axs[1].plot([Lb], [r["g_mean"]], mk, color=cols[n - 1], ms=5, mfc="none" if r.get("state") != "CONVERGED" else cols[n - 1])
    axs[0].plot([18], [A_["s18"]], "D", color="#333", label="half-box 18, M5.56 rows 30, 31", ms=5)
    axs[1].plot([18], [A_["g18"]], "D", color="#333", ms=5)
    axs[0].plot(L, [2 * (q - 5.237) for q in L], ":", color="#999", lw=1, label="rings 5.237 from the faces")
    axs[0].plot(L, [25.525] * 3, "--", color="#bbb", lw=1, label="s = 25.525")
    axs[1].axhline(5.237, ls=":", color="#999", lw=1)
    axs[0].set_xlabel("half-box L")
    axs[0].set_ylabel("separation s of the fitted circles")
    axs[1].set_xlabel("half-box L")
    axs[1].set_ylabel("face distance (mean of L - |c_x|)")
    axs[0].legend(fontsize=5.5, frameon=False)
    axs[0].set_title(f"rung A: {A_['word']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "m5_57_rungA.png", dpi=130)
    plt.close(fig)
    # 2. rung B: U against s with D' and D, both ladders, both spacings (drawn only when a rung ran)
    any_rung = any(s_.get("U") is not None for sp in ("0.5", "0.375") for key in ("inward", "outward") for s_ in B[sp]["tables"][key])
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 5.6))
    for k, sp in enumerate(("0.5", "0.375")):
        b = B[sp]
        ax, ax2 = axs[0, k], axs[1, k]
        for i, (lab, key) in enumerate((("inward", "inward"), ("outward", "outward"))):
            tb = b["tables"][key]
            pts = [s for s in tb if s.get("in_hold_here") and s.get("word") == "TWO RINGS" and not s.get("evaded")]
            ax.plot([s["s"] for s in pts], [s["U"] for s in pts], marker="o", ms=3, color=cols[i], label=f"{lab}, TWO RINGS in hold")
            one = [s for s in tb if s.get("in_hold_here") and s.get("word") == "ONE OBJECT"]
            ax.plot([s["Dprime"] for s in one], [s["U"] for s in one], marker="x", ms=4, ls="none", color=cols[i], label=f"{lab}, ONE OBJECT at D'")
            oth = [s for s in tb if s.get("U") is not None and not s.get("in_hold_here")]
            ax.plot([s["Dprime"] for s in oth], [s["U"] for s in oth], marker="o", ms=3, ls="none", mfc="none", color=cols[i], alpha=0.6,
                    label=f"{lab}, not in hold (at D')")
            allr = [s for s in tb if s.get("U") is not None]
            ax2.plot([s["target"] for s in allr], [s["Dprime"] for s in allr], marker="o", ms=3, color=cols[i], label=f"{lab}: D'")
            ax2.plot([s["target"] for s in allr if s.get("D")], [s["D"]["D"] for s in allr if s.get("D")], marker="^", ms=3, ls=":", color=cols[i],
                     label=f"{lab}: D (read)")
        r = b["references"]
        if r["free"].get("U") is not None:
            ax.plot([r["free"].get("s")], [r["free"]["U"]], "D", color="#333", ms=4, label="free end")
        if r["merged"].get("U") is not None:
            ax.plot([r["merged"].get("Dprime")], [r["merged"]["U"]], "*", color="#333", ms=7, label="merged end (at D')")
        mc = b.get("Mc", {})
        v = mc.get("value")
        ttl = f"spacing {sp}: M_c = {'none' if v is None else f'{v:.4f}'} ({mc.get('label')}), {mc.get('interior_word') or ''}"
        ax.set_title(ttl, fontsize=7)
        ax.set_xlabel("separation s (or D' where marked)")
        ax.set_ylabel("U")
        ax.legend(fontsize=5, frameon=False)
        lim = [0, 26]
        ax2.plot(lim, lim, ":", color="#999", lw=1)
        ax2.set_xlabel("target of D'")
        ax2.set_ylabel("D' and D at the rung end")
        ax2.legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    if any_rung:
        fig.savefig(PLOTS / "m5_57_rungB.png", dpi=130)
    plt.close(fig)
    # 3. rung C: the path tables and the turn profiles
    ch = [c for c in C_["chains"] if c.get("prep", {}).get("paths")]
    if ch:
        fig, axs = plt.subplots(2, len(ch), figsize=(3.2 * len(ch), 5.0), squeeze=False)
        for k, c in enumerate(ch):
            ax = axs[0, k]
            psi = [2 - q / 8 for q in range(17)]
            for i, (an, v) in enumerate(c["prep"]["paths"].items()):
                ax.plot(psi, v["dU"], marker="o", ms=2.5, color=cols[i], label=f"{an} ({v['word']})")
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xlabel("psi / pi")
            ax.set_ylabel("dU along T[psi]")
            ax.set_title(f"{c['name']} on {c.get('reference')}", fontsize=7)
            ax.legend(fontsize=6, frameon=False)
            ax = axs[1, k]
            r = rec(c["row"])
            if r is not None and r.get("stops"):
                stops = [s for s in r["stops"] if "turn" in s]
                pick = stops[:: max(1, len(stops) // 8)] + [stops[-1]]
                for i, s in enumerate(pick):
                    ax.plot([q["rho"] for q in s["turn"]["shells"]], [q["q"][0] for q in s["turn"]["shells"]], lw=1,
                            color=plt.cm.viridis(i / max(1, len(pick) - 1)), label=f"it {s['iteration']}" if i in (0, len(pick) - 1) else None)
                ax.set_xlabel("shell radius about ring 1")
                ax.set_ylabel("q0 of the shell's turn")
                ax.set_title(f"{c['name']}: {c.get('word')}", fontsize=7)
                ax.legend(fontsize=6, frameon=False)
            else:
                ax.text(0.5, 0.5, c.get("state", ""), ha="center", fontsize=7, transform=ax.transAxes, wrap=True)
                ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(PLOTS / "m5_57_rungC.png", dpi=130)
        plt.close(fig)
    # 4. field cross-sections: g12 on the plane z = 0
    panels = []
    for n, lab, f in ((1, "start", "s00000.npz"), (1, "end", "end.npz"), (2, "end", "end.npz"), (4, "end", "end.npz"), (5, "end", "end.npz"),
                      (6, "end", "end.npz")):
        if rec(n) is not None and rec(n).get("end_state") != "NOT RUN":
            panels.append((n, lab, f))
    for sp in ("0.5", "0.375"):
        mh = B[sp].get("Mc", {}).get("max_holder", {})
        if mh.get("kind") == "rung":
            n = [s["row"] for s in B[sp]["tables"]["inward"] + B[sp]["tables"]["outward"] if s["name"] == mh["name"]]
            if n:
                panels.append((n[0], "end", "end.npz"))
    if panels:
        fig, axs = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.6), squeeze=False)
        for ax, (n, lab, f) in zip(axs[0], panels):
            z = np.load(WORK / X.cname(n) / f, allow_pickle=False)
            m6, x = np.asarray(z["m6"]), np.asarray(z["x"])
            w, _ = T.eigs(m6)
            g = (w[..., 0] - w[..., 1])[:, :, len(x) // 2]
            ax.imshow(g.T, origin="lower", extent=[x[0], x[-1], x[0], x[-1]], cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"{X.cname(n)} {lab}", fontsize=7)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(False)
        fig.tight_layout()
        fig.savefig(PLOTS / "m5_57_fields.png", dpi=110)
        plt.close(fig)


def status():
    for n in X.ROWS:
        s = summ(n)
        extra = ""
        if s.get("U") is not None:
            extra = f"U {s['U']:.7f} D' {s['Dprime']:.4f} word {s.get('word')} s {s.get('s')} ss {s['second_step']['ok']} K5 {s['K5']['state']}"
            if "in_hold" in s:
                extra += f" hold {s['in_hold']} ev {s['evaded']}"
        print(f"{X.cname(n):16s} {str(s.get('state'))[:40]:40s} {s.get('iterations', '')} {extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        status()
    elif a.summary:
        summary()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
