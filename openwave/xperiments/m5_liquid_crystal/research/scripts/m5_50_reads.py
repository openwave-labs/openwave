"""M5.50: the readings R1 to R5 from the chain records, by the frozen rules of m5_50_task_details.md section 4.

    THE STEP IDENTITY   T_k = (f_k + f_k+1)(D_k+1 - D_k) / 2,   e_k = |(U_k+1 - U_k) - T_k| / |U_k+1 - U_k|
    ELIGIBLE            a rung CONVERGED, not EVADED, with one clear set and a room of at least the room rule
    EVADED              CONVERGED, one clear set, and the reader's separation more than 0.75 from D or less than
                        0.80 of the weight within 1.5 of the loop's plaquettes
    THE COUNTED RUN     the unbroken run of eligible rungs from the first outward target; it ends before a rung that
                        is not eligible, or at a step with e above 10 percent (5 percent at spacing 0.5) that does
                        not pass its bisection (two bisections per ladder at most)
    A FALL OF U         between two eligible rungs at neighboring targets: the upper rung's state is continued down
                        to the lower target (the branch check); a lower state found there takes the lower rung's
                        place (LOWER STATE); BRANCH only when U still falls
    THE ROOM RULE       the smallest room of half-box 10 such that its rung and every shared rung with more room have
                        a rise within 5 percent of the rise on half-box 14; never under 2
    R1 to R5            the table of section 4; NOT HELD outranks no label, no label outranks HELD

The same functions name the chains that the outcomes call for (dynamic_jobs, used by the pool of m5_50_hold.py),
read the made-up records of K10 (--k10), and write the tracked summary (--summary).

Usage:  python m5_50_reads.py --k10
        python m5_50_reads.py --summary
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

LOWER = 1e-6
_CACHE = {}


# ----------------------------------------------------------------------------- records

def load(work_dir, name):
    f = Path(work_dir) / "chains" / f"{name}.json"
    if not f.exists():
        return None
    m = f.stat().st_mtime
    if _CACHE.get(str(f), (None,))[0] != m:
        r = json.loads(f.read_text())
        _CACHE[str(f)] = (m, r if r.get("finished") else None)
    return _CACHE[str(f)][1]


def bad_surfaces(rec):
    """Reads at which a liftable surface is outside 0 and 1, or one around the loop does not read 1."""
    bad = []
    for row in rec["reads"] + rec.get("dense", []):
        for k, ch in enumerate(row["cls"]):
            if ch.isdigit() and (ch not in "01" or (row["ar"][k] == "1" and ch != "1")):
                bad.append([row["it"], rec["half_widths"][k], ch, row["ar"][k]])
    return bad


def end_how(rec):
    """How a chain ended, from its events (the driver's own word can say REPEATED AFTER A RESTART where no restart
    followed an end by the function tolerance): printed beside the word."""
    ev = rec.get("events", [])
    nf = sum(1 for e in ev if e["event"] == "function tolerance")
    ns = sum(1 for e in ev if e["event"] == "stalled line search")
    if rec["word"] != "CONVERGED":
        return rec["ended"]
    if ns:
        return "the gate alone, a stalled line search in its record"
    if nf >= 2:
        return "the function tolerance repeated after a restart"
    return "one end by the function tolerance within 25 iterations of the start, no restart"


def rung_of(rec):
    """The numbers of a held state that the readings use."""
    e, sp = rec["end"], rec["spec"]
    one = e["clear_sets"] == 1
    share = None
    if one:
        k = [i for i, t in enumerate(e["set_touches"]) if not t][0]
        share = e["weight_share_per_set"][k]
    hold = sp.get("hold") or {}
    conv = bool(rec["converged"])
    evaded = bool(conv and one and hold.get("kind") == "D" and (abs(e["separation"] - e["D"]) > 0.75 or share < 0.80))
    return dict(name=rec["chain"], d=hold.get("d"), mu=hold.get("mu"), D=e["D"], sep=e["separation"] if one else None, U=e["U"], Ut=[e["U2"], e["U4"], e["UV"]],
                f=e["f"], Eh=e["Eh"], converged=conv, word=rec["word"], ended=rec["ended"], evaded=evaded, clear_sets=e["clear_sets"], sets=e["sets"],
                room=e["room"], radius=e["radius"], share=share, reads=len(rec["reads"]), bad=bad_surfaces(rec), iterations=rec["iterations"],
                gmax=e["gmax_over_h3"], gate=e["gate"], cubes=e.get("cubes"), set_radii=e.get("set_radii"), shares=e.get("weight_share_per_set"),
                E=e["E"], line_search_end=rec.get("line_search_end", False), stalls=rec.get("stalls", 0), end_how=end_how(rec))


# ----------------------------------------------------------------------------- R1

def ident(a, b):
    dU = b["U"] - a["U"]
    T = 0.5 * (a["f"] + b["f"]) * (b["D"] - a["D"])
    return abs(dU - T) / abs(dU) if dU != 0 else float("inf")


def why_not(r, room_rule):
    if not r["converged"]:
        return "a rung that is not CONVERGED"
    if r["evaded"]:
        return "an EVADED rung"
    if r["clear_sets"] != 1:
        return "two sets" if r["clear_sets"] >= 2 else "a rung with no clear set"
    if room_rule is not None and r["room"] < room_rule - 1e-9:
        return "the room rule"
    return None


def analyze_ladder(lad):
    """One ladder read by the rules of R1. lad: tol, targets, free_U, rungs {d: rung}, mids {d: rung}, checks
    {(lo, hi, name of the upper state): rung}, room_rule (None when no shared target agrees), second_hold_fall,
    family_lower (a rung or None), cont {d: [U, ...]}. Returns the word, the counted run, where it ends and why,
    and the chains still needed (bisections and branch checks) before the reading is final."""
    tol, targets = lad["tol"], list(lad["targets"])
    rungs = {d: dict(r) for d, r in lad["rungs"].items()}
    mids, checks = lad.get("mids", {}), lad.get("checks", {})
    room_rule, no_room = lad.get("room_rule"), lad.get("room_rule") is None and lad.get("room_rule_needed", True)
    needs, notes, lower_at, branch, undecided, first_print = [], [], [], [], [], None

    def eligible(r):
        return r is not None and why_not(r, room_rule) is None

    def falls(nb):
        """Resolve the falls of U between neighbors; returns False when a needed chain is missing."""
        again = True
        while again:
            again = False
            for lo, hi in zip(nb[:-1], nb[1:]):
                a, b = at(lo), at(hi)
                if not (eligible(a) and eligible(b)) or not b["U"] < a["U"]:
                    continue
                key = (lo, hi, b["name"])
                if any(x[0] == key for x in branch) or key in undecided:
                    continue
                c = checks.get(key)
                if c is None:
                    needs.append(dict(kind="check", lo=lo, hi=hi, upper=b["name"]))
                    return False
                # an EVADED check gives no point of the curve (section 1): neither a lower state nor a branch, the step is not decided
                if c["converged"] and c["clear_sets"] == 1 and not c.get("evaded") and a["U"] - c["U"] > LOWER * abs(a["U"]):
                    lower_at.append(dict(at=lo, first=dict(a), lower=dict(c)))
                    put(lo, dict(c, d=a["d"], lower_state=True))
                    again = True
                    break
                if c["converged"] and not c.get("evaded"):
                    branch.append((key, dict(lo=lo, hi=hi, U_lo=a["U"], U_hi=b["U"], U_check=c["U"], check_sets=c["clear_sets"])))
                else:
                    undecided.append(key)
        return True

    def at(d):
        return rungs.get(d) if d in rungs else mids_used.get(d)

    def put(d, r):
        if d in rungs:
            rungs[d] = r
        else:
            mids_used[d] = r

    mids_used = {}
    if lad.get("family_lower") is not None:
        first_print = True
    missing = [d for d in targets if d not in rungs]
    if missing:
        return dict(final=False, needs=[], word=None, waiting_for=missing)
    ok = falls(targets)
    counted, ended_by, ends_at, bis, classes, pending = [], None, None, 0, [], not ok
    if ok:
        for d in targets:
            r = rungs[d]
            if not eligible(r):
                ended_by = why_not(r, room_rule)
                break
            if counted:
                p = at(counted[-1])
                fell = [k for k in [x[0] for x in branch] + undecided if k[0] == counted[-1] and k[1] == d]
                if fell:        # a step on which U falls after its branch check is not bisected: the check has read it
                    ended_by = "a fall of U (BRANCH)" if any(x[0] == fell[0] for x in branch) else "a fall of U that is not decided"
                    break
                e = ident(p, r)
                if e > tol:
                    if bis >= 2:
                        ended_by = "the identity (a third failing step, which is not bisected)"
                        classes.append(dict(step=[p["d"], d], e=e, cls="not bisected"))
                        break
                    md = 0.5 * (counted[-1] + d)
                    m = mids.get(md)
                    if m is None:
                        needs.append(dict(kind="bisect", d=md))
                        pending = True
                        break
                    bis += 1
                    e1, e2 = (ident(p, m), ident(m, r)) if eligible(m) else (None, None)
                    if eligible(m) and e1 <= tol and e2 <= tol:
                        mids_used[md] = dict(m)
                        counted += [md, d]
                        notes.append(f"the step from {counted[-3]:g} to {d:g} (e = {e:.3f}) passes on its bisection at {md:g} (e = {e1:.3f}, {e2:.3f})")
                        continue
                    if eligible(m):
                        for x, y, ee in ((p, m, e1), (m, r, e2)):
                            if ee > tol:
                                classes.append(dict(step=[x["d"], y["d"]], e=ee, f_lower=x["f"], f_upper=y["f"],
                                                    cls="a fall of the force inside the step" if y["f"] < 0.8 * x["f"] else "two branches"))
                    else:
                        classes.append(dict(step=[p["d"], d], e=e, f_lower=p["f"], f_upper=r["f"], middle=why_not(m, room_rule),
                                            cls="a fall of the force inside the step" if r["f"] < 0.8 * p["f"] else "two branches"))
                    ended_by = "the identity"
                    break
            counted.append(d)
        else:
            ended_by = "the top of the ladder"
        if not pending and mids_used:
            nb = sorted(set(targets) | set(mids_used))
            pending = not falls(nb)
    if pending:
        return dict(final=False, needs=needs, word=None)
    C = [at(d) for d in counted]
    # a member of the family below the top rung takes its place, and the run is read again
    fl = lad.get("family_lower")
    if fl is not None and C and not lad.get("_second"):
        first = analyze_ladder(dict(lad, family_lower=None))
        lad2 = dict(lad, _second=True, rungs=dict(lad["rungs"]), mids=dict(mids))
        top = counted[-1]
        rep = dict(fl, d=at(top)["d"], lower_state=True)
        if top in lad2["rungs"]:
            lad2["rungs"][top] = rep
        else:
            lad2["mids"][top] = rep
        out = analyze_ladder(lad2)
        if out["final"]:
            out["first_print"] = dict(word=first["word"], label=first["label"])
            out["family_lower"] = dict(at=top, U_first=at(top)["U"], U_lower=fl["U"], name=fl["name"])
            if out["word"] == "HELD":
                out["label"] = out["label"].replace("HELD", "HELD (LOWER STATE)", 1)
        return out
    ends_at = counted[-1] if counted else None
    fmax = max((r["f"] for r in C), default=None)
    last2 = float(np.mean([r["f"] for r in C[-2:]])) if len(C) >= 2 else None
    ratio = last2 / fmax if (last2 is not None and fmax) else None
    reasons_not, reasons_none = [], []
    for key, b in branch:
        reasons_not.append(f"BRANCH: U falls from {b['U_lo']:.6f} at {b['lo']:g} to {b['U_hi']:.6f} at {b['hi']:g}, and the state continued down from {b['hi']:g} ends at "
                           f"{b['U_check']:.6f} with {b['check_sets']} clear set(s): no lower state at {b['lo']:g}")
    level = [r for r in C if fmax and r["f"] <= 0.02 * fmax]
    if level:
        reasons_not.append("LEVEL: f at " + ", ".join(f"{r['d']:g}" for r in level) + f" is at or under 0.02 of the largest ({fmax:.4f})")
    if ratio is not None and ratio < 0.1:
        reasons_not.append(f"LEVEL: the mean of f over the last two counted rungs is {ratio:.3f} of the largest")
    if no_room:
        reasons_none.append("no shared target agrees between the two boxes: no room rule")
    for key in undecided:
        reasons_none.append(f"the step from {key[0]:g} to {key[1]:g} is not decided: U falls and the branch check is not CONVERGED")
    if len(C) < 3:
        reasons_none.append(f"{['no', 'one', 'two'][len(C)]} counted rung{'s' if len(C) != 1 else ''}")
    if ratio is not None and 0.1 <= ratio < 0.5:
        reasons_none.append(f"the mean of f over the last two counted rungs is {ratio:.2f} of the largest (between one tenth and one half)")
    under = [r for r in C if lad.get("free_U") is not None and r["U"] <= lad["free_U"]]
    if under:
        reasons_none.append("a rung under the free state: U at " + ", ".join(f"{r['d']:g}" for r in under) + " is at or under the free state's")
    if lad.get("second_hold_fall"):
        reasons_none.append("the two kinds of hold disagree in sign: the second hold gives a fall of U between its two larger disks")
    where = f"the run ending at {ends_at:g}, " if ends_at is not None else "no counted rung, "
    by = {"the top of the ladder": "the top of the ladder", "the identity": "on the identity", "a fall of U (BRANCH)": "on a fall of U (BRANCH)",
          "a fall of U that is not decided": "on a fall of U that is not decided"}.get(ended_by, "on " + ended_by if ended_by and ended_by.startswith(("a ", "an ", "two")) else
                                                                                                    "by " + str(ended_by))
    if classes and ended_by == "the identity":
        by += ", the step classed as " + " and ".join(sorted({c["cls"] for c in classes}))
    tail = where + by
    if reasons_not:
        word = "NOT HELD"
        kind = "BRANCH" if branch else "LEVEL"
        label = f"NOT HELD ({kind}): " + "; ".join(reasons_not) + "; " + tail
    elif reasons_none:
        word = "no label"
        label = "no label: " + "; ".join(reasons_none) + "; " + tail
    else:
        word = "HELD"
        label = "HELD" + (" (LOWER STATE at " + ", ".join(f"{x['at']:g}" for x in lower_at) + ")" if lower_at else "") + ", " + tail
    # start dependence, at spacing 0.5: a continuation rung against its fresh partner
    sd = []
    for d in counted:
        for Uc in lad.get("cont", {}).get(d, []):
            rise = at(d)["U"] - lad["free_U"]
            if rise > 0 and abs(Uc - at(d)["U"]) > 0.01 * rise:
                sd.append(dict(d=d, U_fresh=at(d)["U"], U_continued=Uc, share_of_the_rise=abs(Uc - at(d)["U"]) / rise))
    if sd and word == "HELD":
        label = label.replace(",", ", START-DEPENDENT,", 1) if not lower_at else label + ", START-DEPENDENT"
    steps = [dict(lo=a["d"], hi=b["d"], e=ident(a, b), dU=b["U"] - a["U"], T=0.5 * (a["f"] + b["f"]) * (b["D"] - a["D"])) for a, b in zip(C[:-1], C[1:])]
    return dict(final=True, needs=[], word=word, label=label, counted=counted, ends_at=ends_at, ended_by=ended_by, classes=classes, f_max=fmax, f_last_two=last2,
                ratio_last_two=ratio, lower_states=lower_at, branch=[b for _, b in branch], undecided=[list(k) for k in undecided], notes=notes, start_dependent=sd,
                steps=steps, counted_rungs=C, bisections_used=bis, top=(C[-1] if C else None), reasons_not=reasons_not, reasons_none=reasons_none)


def uncut_print(lad, with_evaded=False):
    """R1 printed on all CONVERGED rungs with one clear set, with no cut by the identity or the room rule: a descriptor.
    The plan removes only those two cuts; with_evaded=True keeps the EVADED rungs in as well (the second print)."""
    R = [lad["rungs"][d] for d in lad["targets"] if d in lad["rungs"]]
    R = [r for r in R if r["converged"] and r["clear_sets"] == 1 and (with_evaded or not r["evaded"])]
    if not R:
        return dict(rungs=0)
    fmax = max(r["f"] for r in R)
    last2 = float(np.mean([r["f"] for r in R[-2:]]))
    rises = all(b["U"] > a["U"] for a, b in zip(R[:-1], R[1:]))
    return dict(rungs=len(R), targets=[r["d"] for r in R], f_max=fmax, f_last_two=last2, ratio_last_two=last2 / fmax if fmax else None, U_rises=rises,
                f_min_over_max=min(r["f"] for r in R) / fmax if fmax else None)


def room_rule(l10, l14, free10, free14):
    """The room rule of one spacing from the shared targets; None when no shared target agrees."""
    rows = []
    for d in sorted(set(l10) & set(l14)):
        a, b = l10[d], l14[d]
        if all(r["converged"] and not r["evaded"] and r["clear_sets"] == 1 for r in (a, b)):
            ra, rb = a["U"] - free10, b["U"] - free14
            rows.append(dict(d=d, room10=a["room"], rise10=ra, rise14=rb, gap=abs(ra - rb) / abs(rb) if rb else float("inf")))
    for r in rows:
        r["agrees"] = bool(r["gap"] <= 0.05)
    rule = None
    for r in sorted(rows, key=lambda r: -r["room10"]):
        if not r["agrees"]:
            break
        rule = r["room10"]
    if rule is not None:
        rule = max(rule, 2.0)
        # a room is a length: rungs within a hair of the rule's own rung are on it
        rule -= 1e-9
    return rule, rows


# ----------------------------------------------------------------------------- R2 to R5

def split_word(state):
    """The word of a CONVERGED state in two clear sets or more, from its release."""
    rel = state.get("release")
    if rel is None:
        return "release pending"
    if not rel["converged"]:
        return "release not CONVERGED"
    return "PERSISTENT SPLIT" if rel["clear_sets"] >= 2 else "MADE BY THE HOLD"


def read_R2(rungs, counted_n):
    """rungs: the outward rung chains of a ladder (converged, clear_sets, release, bad, reads)."""
    nots, nones = [], []
    for r in rungs:
        if r.get("bad"):
            nots.append(f"{r['name']}: a liftable surface outside 0 and 1, or one around the loop that does not read 1 ({len(r['bad'])} reads)")
        if r["converged"] and r["clear_sets"] >= 2:
            w = split_word(r)
            if w == "PERSISTENT SPLIT":
                nots.append(f"PERSISTENT SPLIT: {r['name']} ends with {r['clear_sets']} clear sets and its release keeps {r['release']['clear_sets']}")
            else:
                nones.append(f"{r['name']} ends with {r['clear_sets']} clear sets: {w}")
        if r.get("reads", 99) < 10:
            nones.append(f"{r['name']}: {r['reads']} counted reads (fewer than ten)")
    if counted_n < 3:
        nones.append("fewer than three counted rungs")
    if nots:
        return "NOT HELD", "NOT HELD" + (" (PERSISTENT SPLIT)" if any(n.startswith("PERSISTENT") for n in nots) else "") + ": " + "; ".join(nots)
    if nones:
        made = any("MADE BY THE HOLD" in n for n in nones)
        return "no label", "no label" + (" (MADE BY THE HOLD)" if made else "") + ": " + "; ".join(nones)
    return "HELD", "HELD: every CONVERGED outward rung ends with one clear set; every liftable surface reads 0 or 1, and 1 around the loop"


def member_word(m):
    if not m["converged"]:
        return "not CONVERGED"
    if m["clear_sets"] == 1:
        return "one clear set"
    if m["clear_sets"] == 0:
        return "no clear set"
    return split_word(m)


def read_R3(members, has_top, expected=None):
    if not has_top:
        return "no label", "no label: the grid has no top counted rung"
    if not members or (expected is not None and len(members) < expected):
        return "no label", f"no label: {len(members)} of {expected if expected is not None else 'the'} chains of the family have a record"
    words = {m["name"]: member_word(m) for m in members}
    if any(w == "PERSISTENT SPLIT" for w in words.values()):
        return "NOT HELD", "NOT HELD (PERSISTENT SPLIT): " + ", ".join(n for n, w in words.items() if w == "PERSISTENT SPLIT")
    rest = {n: w for n, w in words.items() if w != "one clear set"}
    if rest:
        made = any(w == "MADE BY THE HOLD" for w in rest.values())
        return "no label", "no label" + (" (MADE BY THE HOLD)" if made else "") + ": " + "; ".join(f"{n}: {w}" for n, w in rest.items())
    return "HELD", f"HELD: the {len(members)} chains of the family are CONVERGED and end with one clear set"


def seq_word(seq):
    """One surface's reads (characters 0, 1, other digits, L, B) against parts (a) and (c)."""
    if any(c.isdigit() and c not in "01" for c in seq):
        return "NOT HELD", "a liftable read outside 0 and 1"
    i1 = max((i for i, c in enumerate(seq) if c == "1"), default=None)
    i0 = min((i for i, c in enumerate(seq) if c == "0"), default=None)
    if i1 is not None and i0 is not None and i0 < i1:
        return "NOT HELD", "a 1 after a 0"
    if i1 is not None and any(c == "L" for c in seq[:i1]):
        return "NOT HELD", "a read at the loop before the last 1"
    if i1 is not None and i0 is not None and any(c == "B" for c in seq[i1 + 1:i0]):
        return "NOT HELD", "a blind read between the last 1 and the first 0"
    if i1 is None or i0 is None:
        return "no label", "the surface does not go from 1 to 0 on this chain"
    if any(c != "1" for c in seq[:i1]) or any(c != "0" for c in seq[i0:]):
        return "no label", "a read outside the order 1, at the loop, 0 (a blind read in the run of 1, or a read that is not 0 after the first 0)"
    return "HELD", "a run of 1, then reads at the loop, then a run of 0"


def read_R4(chain):
    """chain: ended, faces (those touched at the last read with a set), seqs {half-width: string}, bad (a liftable
    surface outside 0 and 1 anywhere)."""
    if chain.get("bad"):
        return "NOT HELD", "NOT HELD: a liftable surface outside 0 and 1"
    words = {w: seq_word(s) for w, s in chain["seqs"].items()}
    nots = {w: x[1] for w, x in words.items() if x[0] == "NOT HELD"}
    if nots:
        return "NOT HELD", "NOT HELD: " + "; ".join(f"surface {w}: {why}" for w, why in nots.items())
    if chain["ended"] != "EXIT":
        return "no label", f"no label: the chain does not end by EXIT ({chain['ended']})"
    if len(chain["faces"]) >= 2:
        return "no label", f"no label: cut by the box ({len(chain['faces'])} faces touched)"
    if len(chain["faces"]) == 0:
        return "no label", "no label: vanished inside (no face touched at the last read with a set)"
    nones = {w: x[1] for w, x in words.items() if x[0] == "no label"}
    if nones:
        return "no label", "no label: " + "; ".join(f"surface {w}: {why}" for w, why in nones.items())
    if not words:
        return "no label", "no label: no surface is liftable and around the loop at the first read"
    return "HELD", f"HELD: {len(words)} surfaces around the loop at the first read, each a run of 1, then reads at the loop, then a run of 0"


def base_word(w):
    return "NOT HELD" if w.startswith("NOT HELD") else "HELD" if w.startswith("HELD") else "no label"


def read_R5(a, b):
    """a, b: the words of the two spacings of a box: R1, R2 and R3m {member: word}."""
    diff, none = [], []
    for k in ("R1", "R2"):
        x, y = base_word(a[k]), base_word(b[k])
        if "no label" in (x, y):
            none.append(k)
        elif x != y:
            diff.append(f"{k}: {x} at spacing 0.5, {y} at spacing 0.25")
    for m in sorted(set(a["R3m"]) & set(b["R3m"])):
        x, y = a["R3m"][m], b["R3m"][m]
        if x != y:
            (none if "not CONVERGED" in (x, y) or "pending" in x + y else diff).append(f"R3 on {m}: {x} at spacing 0.5, {y} at spacing 0.25")
    if a.get("R3") == "no label: the grid has no top counted rung" or b.get("R3") == "no label: the grid has no top counted rung":
        none.append("R3")
    if diff:
        return "NOT HELD", "NOT HELD: " + "; ".join(diff)
    if none:
        return "no label", "no label: a grid with no label in " + ", ".join(none)
    return "HELD", "HELD: R1 and R2 carry the same word at the two spacings, and R3 the same word on the members both grids ran"


# ----------------------------------------------------------------------------- the ladders from the records

def tname(d):
    return f"d{int(round(d * 100)):04d}"


def ladder_of(a, g, G, rule, second_hold_fall=None, family_lower=None, with_cont=True):
    """The ladder of a grid as analyze_ladder reads it, from the records present."""
    free = load(a.work_dir, f"R{g}_free_m20")
    rungs, mids, checks, cont = {}, {}, {}, {}
    for d in G["targets"]:
        r = load(a.work_dir, f"R{g}_{tname(d)}_m20")
        if r is not None:
            rungs[d] = rung_of(r)
    lo, hi = min(G["targets"]), max(G["targets"])
    for f in sorted((Path(a.work_dir) / "chains").glob(f"R{g}_d*_m20.json")):
        r = load(a.work_dir, f.stem)
        if r is None:
            continue
        # the target is read from the chain's own hold: the name keeps two decimals, and the middle of a step at
        # spacing 0.5 has three (8.125 is named d0812)
        d = float(r["spec"]["hold"]["d"])
        if lo < d < hi and all(abs(d - t) > 1e-9 for t in G["targets"]):
            mids[d] = rung_of(r)
    for f in sorted((Path(a.work_dir) / "chains").glob(f"C{g}_d*_from_*.json")):
        r = load(a.work_dir, f.stem)
        if r is not None:
            sp = r["spec"]
            checks[(sp["check"]["lo"], sp["check"]["hi"], sp["check"]["upper"])] = rung_of(r)
    if with_cont:
        for d in G["targets"]:
            for pre in ("U", "V"):
                r = load(a.work_dir, f"{pre}{g}_{tname(d)}")
                # an EVADED continuation rung gives no point of the curve (section 1), so it is not compared with its fresh partner
                if r is not None and r["converged"] and r["end"]["clear_sets"] == 1 and not rung_of(r)["evaded"]:
                    cont.setdefault(d, []).append(r["end"]["U"])
    return dict(tol=G["tol"], targets=G["targets"], free_U=(free["end"]["U"] if free else None), rungs=rungs, mids=mids, checks=checks, room_rule=rule,
                second_hold_fall=second_hold_fall, family_lower=family_lower, cont=cont)


def k67(a, GRIDS):
    """K6 per grid and K7 on half-box 10 at spacing 0.5, from the records present. Returns (rows, all present, all met)."""
    out, present, met = {}, True, True
    for g in GRIDS:
        ref, fr = load(a.work_dir, f"K6ref_{g}"), load(a.work_dir, f"R{g}_free_m20")
        if ref is None or fr is None:
            present = False
            continue
        gap = abs(fr["end"]["U"] - ref["end"]["U"]) / abs(ref["end"]["U"])
        same = fr["end"]["clear_sets"] == 1 and abs(fr["end"]["radius"] - ref["end"]["radius"]) <= 1e-3
        ok = bool(gap <= 1e-8 and fr["end"]["Eh"] < 1e-12 and same and fr["converged"] and ref["converged"])
        out["K6 " + g] = dict(passed=ok, U_free_rung=fr["end"]["U"], U_reference=ref["end"]["U"], relative_gap=gap, hold_energy=fr["end"]["Eh"],
                              radius_free_rung=fr["end"]["radius"], radius_reference=ref["end"]["radius"], D_target=fr["spec"]["hold"]["d"], D_reached=fr["end"]["D"],
                              words=[fr["word"], ref["word"]], ended=[end_how(fr), end_how(ref)], iterations=[fr["iterations"], ref["iterations"]])
        met &= ok
    ks = [load(a.work_dir, f"K7_10a_p{j}") for j in (1, 2, 3, 4)]
    if any(k is None for k in ks):
        present = False
    else:
        R = [rung_of(k) for k in ks]
        es = [ident(x, y) for x, y in zip(R[:-1], R[1:])]
        wrong = [abs((y["U"] - x["U"]) + 0.5 * (x["f"] + y["f"]) * (y["D"] - x["D"])) / abs(y["U"] - x["U"]) for x, y in zip(R[:-1], R[1:])]
        ok = bool(all(e < 0.01 for e in es) and all(r["converged"] and r["clear_sets"] == 1 for r in R))
        out["K7"] = dict(passed=ok, e=es, e_with_the_other_sign_of_f=wrong, targets=[r["d"] for r in R], D=[r["D"] for r in R], U=[r["U"] for r in R], f=[r["f"] for r in R],
                         words=[r["word"] for r in R])
        met &= ok
    return out, present, met


def spacing_state(a, s, GRIDS):
    """The two ladders of one spacing: the room rule, the analysis of each, and whether every reading chain is in."""
    g10, g14 = "10" + s, "14" + s
    L = {}
    for g in (g10, g14):
        L[g] = ladder_of(a, g, GRIDS[g], None)
    if any(len(L[g]["rungs"]) < len(GRIDS[g]["targets"]) or L[g]["free_U"] is None for g in L):
        return dict(ready=False, grids=L)
    rule, rows = room_rule(L[g10]["rungs"], L[g14]["rungs"], L[g10]["free_U"], L[g14]["free_U"])
    out = dict(ready=True, room_rule=rule, room_rows=rows, grids={})
    for g in (g10, g14):
        lad = dict(L[g], room_rule=rule)
        out["grids"][g] = dict(lad=lad, an=analyze_ladder(lad))
    return out


def dynamic_jobs(a, J, GRIDS, FAMILY, FAMILY_113):
    """The chains that the outcomes call for. Keys starting with __ carry a message for the pool."""
    out = {}
    MU = 20.0
    rows, present, met = k67(a, {"10a": GRIDS["10a"]})
    if present and not met:
        out["__halt__"] = "K6 or K7 is not met on half-box 10 at spacing 0.5: no rung is started (" + json.dumps({k: v["passed"] for k, v in rows.items()}) + ")"
        return out
    notes = []
    for s in ("a", "b"):
        st = spacing_state(a, s, GRIDS)
        if not st["ready"]:
            continue
        final = True
        for g, x in st["grids"].items():
            G, an = GRIDS[g], x["an"]
            bneeds = [f"B{g}"] if g.startswith("14") else []
            for nd in an.get("needs", []):
                final = False
                if nd["kind"] == "bisect":
                    d = nd["d"]
                    out[f"R{g}_{tname(d)}_m20"] = dict(N=G["N"], hb=G["hb"], faces="held", wall=G["wall"], power=a.power, needs=bneeds, cls="bisect", target=d,
                                                      start=dict(kind="state", file=G["base"], dilate=dict(to_D=d)), hold=dict(kind="D", mu=MU, d=d))
                else:
                    up = nd["upper"]
                    out[f"C{g}_{tname(nd['lo'])}_from_{up}"] = dict(N=G["N"], hb=G["hb"], faces="held", wall=G["wall"], power=a.power, needs=[up], cls="check", target=nd["lo"],
                                                                    check=dict(lo=nd["lo"], hi=nd["hi"], upper=up), start=dict(kind="state", file=f"m5_50_{up}_end.npz"),
                                                                    hold=dict(kind="D", mu=MU, d=nd["lo"]))
            if not an.get("final"):
                final = False
        mu2 = [n for n, c in J.items() if c.get("cls") == "mu2" and n[1:4] in st["grids"]]
        if not final or any(load(a.work_dir, n) is None for n in mu2):
            continue
        for g, x in st["grids"].items():
            G, an = GRIDS[g], x["an"]
            top = an["top"]
            if top is None:
                notes.append(f"{g}: no counted rung, so no family and no exit of the top rung")
            else:
                src = f"m5_50_{top['name']}_end.npz"
                for m in (FAMILY_113 if G["N"] == 113 else FAMILY):
                    out[f"P{g}_{m}"] = dict(N=G["N"], hb=G["hb"], faces="held", wall=G["wall"], power=a.power, needs=[top["name"]], cls="family", member=m, target=top["d"],
                                            start=dict(kind="state", file=src, map=FAMILY[m]), hold=dict(kind="D", mu=MU, d=top["d"]))
                out[f"Z{g}"] = dict(N=G["N"], hb=G["hb"], faces="free", wall=G["wall"], power=a.power, needs=[top["name"]], cls="exit", exit_rule=True, dense=True,
                                    start=dict(kind="state", file=src), hold=None, top=top["name"])
            # the releases: every CONVERGED held state of this grid in two clear sets or more
            names = [n for n, c in list(J.items()) + list(out.items()) if not n.startswith("__") and c.get("hold") and c["hold"].get("kind") == "D"
                     and c["N"] == G["N"] and c["hb"] == G["hb"] and c.get("cls") in ("rung", "bisect", "check", "mu2", "family", "cont_up", "cont_down")]
            for n in names:
                r = load(a.work_dir, n)
                if r is not None and r["converged"] and r["end"]["clear_sets"] >= 2:
                    out[f"L_{n}"] = dict(N=G["N"], hb=G["hb"], faces="held", wall=G["wall"], power=a.power, needs=[n], cls="release", of=n,
                                         start=dict(kind="state", file=f"m5_50_{n}_end.npz"), hold=None)
    out["__note__"] = "; ".join(notes)
    return out


# ----------------------------------------------------------------------------- K10: the made-up records

def _A(f=None, D=(8.5, 10.0, 11.5, 13.0, 14.5, 16.0), tol=0.10, U0=31.0, U_first=0.9):
    """A ladder with U built from the step identity."""
    f = list(f or (0.40, 0.42, 0.44, 0.45, 0.46, 0.46))
    U = [U0 + U_first]
    for k in range(1, len(f)):
        U.append(U[-1] + 0.5 * (f[k - 1] + f[k]) * (D[k] - D[k - 1]))
    rungs = {d: dict(name=f"r{d:g}", d=d, D=d, U=u, f=ff, converged=True, evaded=False, clear_sets=1, room=20.0 - d / 2) for d, u, ff in zip(D, U, f)}
    return dict(tol=tol, targets=list(D), free_U=U0, rungs=rungs, mids={}, checks={}, room_rule=2.0, second_hold_fall=False, family_lower=None, cont={})


def k10(out_file=None):
    sys.path.insert(0, str(HERE))
    import m5_50_hold as H
    res = []

    def rec(n, got, must):
        ok = all(m in got for m in must)
        res.append(dict(record=n, passed=bool(ok), returned=got, must_hold=list(must)))
        print(f"K10 record {n:2d}: {'PASS' if ok else 'FAIL'}  {got}")

    def r1(lad):
        an = analyze_ladder(lad)
        return an["label"] if an["final"] else "NOT FINAL: needs " + json.dumps(an["needs"])

    rec(1, r1(_A()), ["HELD, the run ending at 16, the top of the ladder"])
    rec(2, r1(_A(f=(0.40, 0.30, 0.20, 0.10, 0.04, 0.02))), ["NOT HELD (LEVEL)", "0.075 of the largest"])
    rec(3, r1(_A(f=(0.40, 0.36, 0.30, 0.24, 0.18, 0.14))), ["no label", "0.40 of the largest"])
    rec(4, r1(_A(f=(0.40, 0.42, 0.44, 0.45, 0.005, 0.46))), ["NOT HELD (LEVEL)", "at or under 0.02"])
    L = _A()
    for d in (13.0, 14.5, 16.0):
        L["rungs"][d]["U"] -= 0.6675 + 0.3
    need = analyze_ladder(L)
    L["checks"][(11.5, 13.0, "r13")] = dict(L["rungs"][11.5], name="c", clear_sets=1, converged=True)
    rec(5, r1(L), ["NOT HELD (BRANCH)"])
    assert need["needs"] == [dict(kind="check", lo=11.5, hi=13.0, upper="r13")], need
    L = _A()
    L["rungs"][14.5]["converged"] = False
    rec(6, r1(L), ["HELD, the run ending at 13, on a rung that is not CONVERGED"])
    L = _A()
    for d in (11.5, 13.0, 14.5, 16.0):
        L["rungs"][d]["converged"] = False
    rec(7, r1(L), ["no label", "two counted rungs"])
    L["rungs"][10.0]["U"] = L["rungs"][8.5]["U"] - 0.1
    L["checks"][(8.5, 10.0, "r10")] = dict(L["rungs"][8.5], name="c")
    rec(8, r1(L), ["NOT HELD (BRANCH)"])
    L = _A()
    L["rungs"][13.0]["evaded"] = True
    rec(9, r1(L), ["HELD, the run ending at 11.5, on an EVADED rung"])
    L = _A()
    L["rungs"][16.0]["room"] = 1.9
    rec(10, r1(L), ["HELD, the run ending at 14.5, by the room rule"])
    L = _A()
    for d in (13.0, 14.5, 16.0):
        L["rungs"][d]["U"] += 0.2
    need = analyze_ladder(L)
    L["mids"][12.25] = dict(name="m", d=12.25, D=12.25, f=0.445, U=L["rungs"][11.5]["U"] + 0.5 * (0.44 + 0.445) * 0.75, converged=True, evaded=False, clear_sets=1, room=10.0)
    got = analyze_ladder(L)
    rec(11, got["label"] + " | e of the step and of the failing half: " + f"{ident(L['rungs'][11.5], L['rungs'][13.0]):.2f}, {got['classes'][0]['e']:.2f}",
        ["HELD, the run ending at 11.5, on the identity, the step classed as two branches", "0.23, 0.37"])
    assert need["needs"] == [dict(kind="bisect", d=12.25)], need
    L = _A(f=(0.40, 0.42, 0.44, 0.10, 0.10, 0.10))
    for d in (13.0, 14.5, 16.0):
        L["rungs"][d]["U"] += 0.4725 - 0.405
    e12 = ident(L["rungs"][11.5], L["rungs"][13.0])
    L["mids"][12.25] = dict(name="m", d=12.25, D=12.25, f=0.36, U=L["rungs"][11.5]["U"] + 0.30, converged=True, evaded=False, clear_sets=1, room=10.0)
    got = analyze_ladder(L)
    rec(12, got["label"] + f" | e = {e12:.2f}; counted {got['counted']}", ["no label", "the run ending at 16", "0.23 of the largest", "e = 0.14", "12.25"])
    L = _A()
    D = L["targets"]
    for k in (1, 2, 4):          # the steps into the 2nd, 3rd and 5th rungs fail (U raised by a fifth of the step from there on)
        a_, b_ = L["rungs"][D[k - 1]], L["rungs"][D[k]]
        T = 0.5 * (a_["f"] + b_["f"]) * 1.5
        for d in D[k:]:
            L["rungs"][d]["U"] += 0.2 * T
        if k in (1, 2):
            fm = 0.5 * (4 * 1.2 * T / 1.5 - a_["f"] - b_["f"])
            md = 0.5 * (D[k - 1] + D[k])
            L["mids"][md] = dict(name=f"m{md:g}", d=md, D=md, f=fm, U=a_["U"] + 0.5 * (a_["f"] + fm) * 0.75, converged=True, evaded=False, clear_sets=1, room=10.0)
    got = analyze_ladder(L)
    rec(13, got["label"] + f" | bisections used {got['bisections_used']}; counted {got['counted']}",
        ["HELD, the run ending at 13, by the identity (a third failing step, which is not bisected)", "bisections used 2"])
    L = _A()
    L["second_hold_fall"] = True
    rec(14, r1(L), ["no label", "the two kinds of hold disagree in sign"])
    L = _A()
    L["family_lower"] = dict(L["rungs"][16.0], name="member", U=L["rungs"][16.0]["U"] - 0.05)
    got = analyze_ladder(L)
    rec(15, got["label"] + " | first print: " + got["first_print"]["label"], ["HELD (LOWER STATE), the run ending at 16", "first print: HELD, the run ending at 16"])
    L = _A(tol=0.05)
    rise = L["rungs"][11.5]["U"] - L["free_U"]
    L["cont"] = {11.5: [L["rungs"][11.5]["U"] + 0.02 * rise]}
    rec(16, r1(L), ["HELD, START-DEPENDENT"])
    L = _A(U_first=-0.01)
    rec(17, r1(L), ["no label", "a rung under the free state"])

    def chains(n=6, **kw):
        return [dict(name=f"r{k}", converged=True, clear_sets=1, release=None, bad=[], reads=30, **kw) for k in range(n)]

    rec(18, read_R2(chains(), 6)[1], ["HELD"])
    C = chains()
    C[5].update(clear_sets=2, cubes=[[[2.0, "1"]], [[1.0, "0"]]], release=dict(converged=True, clear_sets=2))
    rec(19, read_R2(C, 5)[1], ["NOT HELD (PERSISTENT SPLIT)"])
    C[5]["release"] = dict(converged=True, clear_sets=1)
    rec(20, read_R2(C, 5)[1], ["no label (MADE BY THE HOLD)"])
    C[5]["release"] = dict(converged=False, clear_sets=2)
    rec(21, read_R2(C, 5)[1], ["no label", "release not CONVERGED"])
    C = chains()
    fake = dict(reads=[dict(it=0, cls="LL1111", ar="001111"), dict(it=10, cls="LL1211", ar="001111")], half_widths=[2, 2.5, 3, 3.5, 4, 4.5])
    C[2]["bad"] = bad_surfaces(fake)
    rec(22, read_R2(C, 6)[1] + f" | reads flagged {C[2]['bad']}", ["NOT HELD", "[10, 3.5, '2', '1']"])
    C[5].update(clear_sets=2, release=dict(converged=True, clear_sets=1))
    rec(23, read_R2(C, 5)[1], ["NOT HELD"])
    C = chains()
    C[1]["reads"] = 8
    rec(24, read_R2(C, 6)[1], ["no label", "8 counted reads"])
    M = [dict(name=f"m{k}", converged=True, clear_sets=1, release=None) for k in range(6)]
    rec(25, read_R3(M, True)[1], ["HELD"])
    M[2].update(clear_sets=2, release=dict(converged=True, clear_sets=2))
    rec(26, read_R3(M, True)[1], ["NOT HELD (PERSISTENT SPLIT)"])
    M[2]["release"] = dict(converged=True, clear_sets=1)
    rec(27, read_R3(M, True)[1], ["no label (MADE BY THE HOLD)"])
    M[2].update(clear_sets=1, release=None, converged=False)
    rec(28, read_R3(M, True)[1], ["no label", "not CONVERGED"])
    M[3].update(clear_sets=2, release=dict(converged=True, clear_sets=2))
    rec(29, read_R3(M, True)[1], ["NOT HELD"])
    rec(30, read_R3([], False)[1], ["no label", "no top counted rung"])

    def ex(seq, ended="EXIT", faces=(4,), bad=False):
        return read_R4(dict(ended=ended, faces=list(faces), seqs={"7.0": seq}, bad=bad))[1]

    rec(31, ex("111LL000"), ["HELD"])
    rec(32, ex("111000"), ["HELD"])
    # record 33: the class of a surface that is not liftable, with no pierced triangle and a fragment of three plaquettes 0.4 from it
    P = np.array([[6.6, 0.1, 0.2], [6.6, 0.6, 0.2], [6.6, 0.1, 0.7]])
    near = float(H.dist_to_cube(P, 7.0).min()) <= 1.0
    far = float(H.dist_to_cube(P - np.array([4.0, 0, 0]), 7.0).min()) <= 1.0
    rec(33, ex("11" + ("L" if near else "B") + "00") + f" | fragment within 1.0: {near}; the same fragment 4 further in: {far}", ["HELD", "within 1.0: True", "further in: False"])
    rec(34, ex("11B00"), ["NOT HELD", "blind"])
    rec(35, ex("1L1L0"), ["NOT HELD", "before the last 1"])
    rec(36, ex("101"), ["NOT HELD", "a 1 after a 0"])
    rec(37, ex("11200", bad=True), ["NOT HELD", "outside 0 and 1"])
    rec(38, ex("11L00", faces=(4, 1)), ["no label", "cut by the box"])
    rec(39, ex("11L00", faces=()), ["no label", "vanished inside"])
    rec(40, ex("11L", ended="CAP"), ["no label", "does not end by EXIT"])
    same = dict(R1="HELD, the run ending at 13", R2="HELD: x", R3m=dict(pin2e80="one clear set", noise20="one clear set"))
    rec(41, read_R5(same, dict(same))[1], ["HELD"])
    rec(42, read_R5(same, dict(same, R1="NOT HELD (LEVEL): x"))[1], ["NOT HELD", "R1"])
    rec(43, read_R5(same, dict(same, R1="no label: two counted rungs"))[1], ["no label"])
    L = _A()
    L["rungs"][11.5]["U"] += 0.9
    fall = L["rungs"][11.5]["U"] - L["rungs"][13.0]["U"]
    L["checks"][(11.5, 13.0, "r13")] = dict(_A()["rungs"][11.5], name="c")
    got = analyze_ladder(L)
    rec(44, got["label"] + f" | U falls by {fall:.4f}; steps read again: {[round(s['e'], 6) for s in got['steps']]}", ["HELD (LOWER STATE at 11.5), the run ending at 16", "0.2325"])
    L["checks"][(11.5, 13.0, "r13")]["converged"] = False
    need = analyze_ladder(L)
    # with the lower state not found, the step from 10 to 11.5 fails the identity (the fresh state is 0.9 above): the record is
    # completed with the bisection rung that this step calls for, on the curve of ladder A
    L["mids"][10.75] = dict(name="m", d=10.75, D=10.75, f=0.43, U=L["rungs"][10.0]["U"] + 0.5 * (0.42 + 0.43) * 0.75, converged=True, evaded=False, clear_sets=1, room=10.0)
    rec(45, r1(L) + f" | before the bisection rung the reader asks for {json.dumps(need['needs'])}", ["no label", "the step from 11.5 to 13 is not decided"])
    # the counter-input of the control: a reader that returns HELD on every record
    always = sum(1 for r in res if "HELD" in r["must_hold"][0] and not r["must_hold"][0].startswith(("NOT", "no")))
    passed = all(r["passed"] for r in res)
    out = dict(control="K10 the readers of the labels", passed=passed, records=res,
               counter_input=f"a reader that returns HELD on every record matches {always} of the {len(res)} listed words, so the check fails on it")
    print("K10:", "PASS" if passed else "FAIL: records " + ", ".join(str(r["record"]) for r in res if not r["passed"]))
    print("      counter-input:", out["counter_input"])
    if out_file:
        Path(out_file).write_text(json.dumps(out, indent=1) + "\n")
    return passed


# ----------------------------------------------------------------------------- the summary

def hermite(nodes, D, reach=0.1):
    """The cubic through the points and slopes (D_k, U_k, f_k), at D; None outside the nodes.
    The end segments are extended by `reach` in D: at the top target a point at mu = 80 reaches D = d - f/80, which lies
    above the point at mu = 20 (D = d - f/20) by about 0.04, outside the nodes (a defect of the first build, which left
    the top point of every ladder with no curve)."""
    nodes = sorted(nodes)
    if len(nodes) < 2:
        return None
    segs = list(zip(nodes[:-1], nodes[1:]))
    for k, ((d0, u0, f0), (d1, u1, f1)) in enumerate(segs):
        lo = d0 - (reach if k == 0 else 1e-12)
        hi = d1 + (reach if k == len(segs) - 1 else 1e-12)
        if lo <= D <= hi:
            h = d1 - d0
            t = (D - d0) / h
            return ((2 * t ** 3 - 3 * t ** 2 + 1) * u0 + (t ** 3 - 2 * t ** 2 + t) * h * f0 + (-2 * t ** 3 + 3 * t ** 2) * u1 + (t ** 3 - t ** 2) * h * f1)
    return None


def seen(rec):
    """The distinct reads of a chain: class string and around-the-loop string, with the number of reads of each."""
    out = {}
    for row in rec["reads"]:
        k = row["cls"] + "|" + row["ar"] + "|" + str(row["clear"])
        out[k] = out.get(k, 0) + 1
    return out


def chain_row(rec):
    e, sp = rec["end"], rec["spec"]
    return dict(cls=sp.get("cls"), N=sp["N"], half_box=sp["hb"], faces=sp["faces"], hold=sp.get("hold"), target=sp.get("target"), start=rec["start"],
                ended=rec["ended"], end_how=end_how(rec), word=rec["word"], converged=rec["converged"], iterations=rec["iterations"], evaluations=rec["evaluations"],
                stalls=rec.get("stalls", 0), resumed_at=rec.get("resumed_at", []), U=e["U"], Ut=[e["U2"], e["U4"], e["UV"]], E=e["E"], D=e["D"], f=e["f"], Eh=e["Eh"],
                Eo=e.get("Eo"), separation=e["separation"], radius=e["radius"], room=e["room"], center=e["center"], sets=e["sets"], clear_sets=e["clear_sets"],
                fragments=e["fragments"], set_radii=e["set_radii"], set_counts=e["set_counts"], set_centers=e.get("set_centers"),
                weight_share_per_set=e["weight_share_per_set"], weight_share_within_1p5=e["weight_share_within_1p5"],
                gmax_over_h3=e["gmax_over_h3"], gate=e["gate"], gate_met=e["gate_met"], cubes=e["cubes"], reads=len(rec["reads"]), dense_reads=len(rec.get("dense", [])),
                dense_from=rec.get("dense_from"), reads_seen=seen(rec), bad_reads=bad_surfaces(rec), q_at_the_end=e["q"], cls_at_the_end=e["cls"],
                two_clear_sets_at_reads=[r["it"] for r in rec["reads"] if r["clear"] >= 2][:40], end_file=rec["end_file"])


def exit_chain(rec):
    """The reads of an exit chain for R4: per surface liftable and around the loop at the first read, its classes in the order of the iterations."""
    rows = sorted(rec["reads"] + rec.get("dense", []), key=lambda r: r["it"])
    first = rows[0]
    idx = [k for k, ch in enumerate(first["cls"]) if ch.isdigit() and first["ar"][k] == "1"]
    seqs = {f"{rec['half_widths'][k]:g}": "".join(r["cls"][k] for r in rows) for k in idx}
    allseq = {f"{rec['half_widths'][k]:g}": "".join(r["cls"][k] for r in rows) for k in range(len(rec["half_widths"]))}
    with_set = [r for r in rows if r["n"] >= 1]
    bits = 0
    for srow in (with_set[-1]["sets"] if with_set else []):
        bits |= int(srow[9])
    faces = [["-x", "+x", "-y", "+y", "-z", "+z"][b] for b in range(6) if bits >> b & 1]
    bad = any(ch.isdigit() and ch not in "01" for r in rows for ch in r["cls"])
    ch = dict(ended=rec["ended"], faces=faces, seqs=seqs, bad=bad)
    word, label = read_R4(ch)
    return dict(ended=rec["ended"], iterations=rec["iterations"], faces_at_the_last_read_with_a_set=faces, last_read_with_a_set=(with_set[-1]["it"] if with_set else None),
                reads=len(rec["reads"]), dense_reads=len(rec.get("dense", [])), dense_from=rec.get("dense_from"), its=[r["it"] for r in rows], seqs=seqs, all_surfaces=allseq,
                per_surface={w: list(seq_word(q)) for w, q in seqs.items()}, word=word, label=label, U_end=rec["end"]["U"],
                radius_by_read=[(r["sets"][r["loop"]][1] if r["loop"] is not None else None) for r in rec["reads"]],
                distance_to_faces_by_read=[(r["sets"][r["loop"]][8] if r["loop"] is not None else None) for r in rec["reads"]], cadence_its=[r["it"] for r in rec["reads"]])


def reread_m5_46(out_dir, name):
    """The desk read (i): the per-read record of an added chain of M5.46, 'at the loop' under the gap taken from the fitted circle."""
    import m5_50_hold as H
    rec = json.loads((Path(out_dir) / f"m5_46_chain_{name}.json").read_text())
    hw = rec["half_widths"]
    th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    rows = []
    for r in rec["reads"]:
        pts = []
        for srow in r["sets"]:
            c, n, R = np.array(srow[2:5]), np.array(srow[5:8]), srow[1]
            n = n / (np.linalg.norm(n) or 1.0)
            e1 = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            pts.append(c + R * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2))
        P = np.concatenate(pts) if pts else np.zeros((0, 3))
        cls = ""
        for k, ch in enumerate(r["q"]):
            if ch.isdigit():
                cls += ch
            elif ch == "x":
                cls += "L"
            else:
                cls += "L" if (len(P) and float(H.dist_to_cube(P, hw[k]).min()) <= 1.0 + 1e-9) else "B"
        rows.append(dict(it=r["it"], cls=cls, ar=r["ar"], n=r["n"], sets=r["sets"]))
    first = rows[0]
    idx = [k for k, ch in enumerate(first["cls"]) if ch.isdigit() and first["ar"][k] == "1"]
    seqs = {f"{hw[k]:g}": "".join(r["cls"][k] for r in rows) for k in idx}
    with_set = [r for r in rows if r["n"] >= 1]
    bits = 0
    for srow in (with_set[-1]["sets"] if with_set else []):
        bits |= int(srow[9])
    faces = [["-x", "+x", "-y", "+y", "-z", "+z"][b] for b in range(6) if bits >> b & 1]
    bad = any(ch.isdigit() and ch not in "01" for r in rows for ch in r["cls"])
    word, label = read_R4(dict(ended=rec["ended"], faces=faces, seqs=seqs, bad=bad))
    return dict(ended=rec["ended"], iterations=rec["iterations"], reads=len(rows), faces_at_the_last_read_with_a_set=faces, its=[r["it"] for r in rows], seqs=seqs,
                per_surface={w: list(seq_word(q)) for w, q in seqs.items()}, reread_word=word, reread_label=label,
                note="a re-read of the stored record of M5.46 at its cadence of 25 iterations; at the loop under the gap is taken from the fitted circle; no label of R4 is given here")


def summary(a):
    import m5_50_hold as H
    G = H.GRIDS
    W = Path(a.work_dir)
    S = dict(task="M5.50", plan_commit="7de7d19e", model=dict(delta=H.DELTA, L1=H.L1, kappa=1.0, v=1.0, J=0.0, stencil="averaged", fixed_chart=True),
             instrument=dict(power=a.power, mu=H.MU, mu_second=H.MU2, mu_obstacle=H.MU_O, tolerance_of_the_identity={g: G[g]["tol"] for g in G}),
             grids={g: dict(half_box=G[g]["hb"], N=G[g]["N"], spacing=2 * G[g]["hb"] / (G[g]["N"] - 1), targets=G[g]["targets"], inward=G[g]["inward"]) for g in G})
    names = sorted(f.stem for f in (W / "chains").glob("*.json"))
    recs = {n: load(W, n) for n in names}
    recs = {n: r for n, r in recs.items() if r is not None}
    S["chains"] = {n: chain_row(r) for n, r in recs.items()}
    C = {}
    # --- the controls from the chains
    for fn in ("controls_stored_states.json",):
        if (W / fn).exists():
            for c in json.loads((W / fn).read_text())["controls"]:
                C[c["control"].split()[0]] = dict(passed=c["passed"], name=c["control"], numbers=c["numbers"], counter_input=c["counter_input"])
    k2 = {}
    for g, ref in (("14a", "10a"), ("14b", "10b")):
        b, r0 = recs.get(f"B{g}"), recs.get(f"K6ref_{ref}")
        if b is None or r0 is None:
            k2[g] = dict(passed=False, why="the base chain has no record")
            continue
        e = b["end"]
        cen = max(abs(v) for v in e["center"]) if e["center"] else None
        gap = abs(e["radius"] - r0["end"]["radius"]) if e["radius"] else None
        k2[g] = dict(passed=bool(b["converged"] and e["clear_sets"] == 1 and e["sets"] == 1 and cen <= 0.01 and gap <= 0.03), word=b["word"], ended=b["ended"],
                     iterations=b["iterations"], sets=e["sets"], center_largest_coordinate=cen, radius=e["radius"], radius_of_half_box_10=r0["end"]["radius"], radius_gap=gap,
                     D=e["D"], D_of_half_box_10=r0["end"]["D"], half_gap_of_D=0.5 * abs(e["D"] - r0["end"]["D"]), U=e["U"], gmax_over_h3=e["gmax_over_h3"])
    C["K2"] = dict(passed=all(v["passed"] for v in k2.values()), name="K2 the new bases", numbers=k2,
                   counter_input="a base with no loop, two loops or a radius more than 0.03 from the base of half-box 10 fails it")
    rows, present, met = k67(a, G)
    C["K6"] = dict(passed=bool(all(rows.get("K6 " + g, {}).get("passed") for g in G)), name="K6 the free state returned", numbers={k: v for k, v in rows.items() if k != "K7"})
    C["K7"] = dict(passed=bool(rows.get("K7", {}).get("passed")), name="K7 the force of the hold", numbers=rows.get("K7"),
                   counter_input="with the other sign of f the error of the three steps is " + ", ".join(f"{x:.3f}" for x in rows.get("K7", {}).get("e_with_the_other_sign_of_f", [])))
    if (W / "k10_readers.json").exists():
        k = json.loads((W / "k10_readers.json").read_text())
        C["K10"] = dict(passed=k["passed"], name=k["control"], numbers=dict(records=len(k["records"]), failed=[r["record"] for r in k["records"] if not r["passed"]]),
                        counter_input=k["counter_input"], records=k["records"])
    k11 = {}
    for n, src, its in (("E0a", "X0a", 775), ("E14c", "X14c", 975)):
        r, old = recs.get(n), json.loads((Path(a.out_dir) / f"m5_46_chain_{src}.json").read_text())
        if r is None:
            k11[n] = dict(passed=False, why="no record")
            continue
        Uend = float(f"{r['end']['U']:.9g}")
        k11[n] = dict(passed=bool(r["iterations"] == its == old["iterations"] and Uend == old["U_at_the_end"] and r["ended"] == "EXIT" and len(r["dense"]) > 0),
                      iterations=r["iterations"], iterations_of_m5_46=old["iterations"], U_at_the_end=Uend, U_of_m5_46=old["U_at_the_end"], ended=r["ended"],
                      dense_reads=len(r["dense"]), dense_from=r["dense_from"], evaluations=r["evaluations"], evaluations_of_m5_46=old["evaluations"])
    C["K11"] = dict(passed=all(v["passed"] for v in k11.values()), name="K11 the exits repeated", numbers=k11,
                    counter_input="a dense read counted by the EXIT rule would end E0a before iteration 775")
    # --- the ladders
    S["spacings"], S["ladders"], words = {}, {}, {}
    k2_fail = {g: not k2[g]["passed"] for g in k2}
    for s in ("a", "b"):
        st = spacing_state(a, s, G)
        if not st["ready"]:
            S["spacings"][s] = dict(ready=False)
            continue
        S["spacings"][s] = dict(ready=True, room_rule=st["room_rule"], shared_targets=st["room_rows"])
        for g, x in st["grids"].items():
            lad0 = x["lad"]
            free = recs[f"R{g}_free_m20"]
            # the second kind of hold
            obst = []
            for Rd in G[g]["obstacle"]:
                r = recs.get(f"O{g}_r{int(round(Rd * 10)):02d}")
                if r is not None:
                    obst.append(dict(Rd=Rd, name=r["chain"], converged=r["converged"], word=r["word"], U=r["end"]["U"], Eo=r["end"]["Eo"], Eo_at_the_start=r["reads"][0].get("Eo"),
                                     separation=r["end"]["separation"], radius=r["end"]["radius"], D=r["end"]["D"], clear_sets=r["end"]["clear_sets"], iterations=r["iterations"]))
            fall = None
            big = [o for o in obst if o["Rd"] > 1.5]
            if len(big) == 2 and all(o["converged"] and o["clear_sets"] == 1 for o in big):
                fall = bool(big[1]["U"] < big[0]["U"])
            # a member of the family below the top rung
            an0 = analyze_ladder(dict(lad0, second_hold_fall=fall))
            fam, fl = [], None
            for m in (H.FAMILY_113 if G[g]["N"] == 113 else H.FAMILY):
                r = recs.get(f"P{g}_{m}")
                if r is None:
                    continue
                row = dict(rung_of(r), member=m, start_sets=r["start"].get("sets"), start_U=r["start"].get("U"), sources_outside_the_box=r["start"].get("sources_outside_the_box"))
                rel = recs.get(f"L_P{g}_{m}")
                row["release"] = dict(converged=rel["converged"], clear_sets=rel["end"]["clear_sets"], U=rel["end"]["U"], word=rel["word"], set_radii=rel["end"]["set_radii"]) if rel else None
                row["word"] = member_word(row)
                fam.append(row)
                top = an0.get("top") if an0.get("final") else None
                if top and row["converged"] and row["clear_sets"] == 1 and not row["evaded"] and top["U"] - row["U"] > LOWER * abs(top["U"]):
                    if fl is None or row["U"] < fl["U"]:
                        fl = row
            lad = dict(lad0, second_hold_fall=fall, family_lower=fl)
            an = analyze_ladder(lad)
            # the rung rows
            rr = []
            everyd = sorted(set(G[g]["targets"]) | set(lad["mids"]) | set(G[g]["inward"]))
            for d in everyd:
                r = recs.get(f"R{g}_{tname(d)}_m20")
                if r is None:
                    continue
                q = rung_of(r)
                rel = recs.get(f"L_R{g}_{tname(d)}_m20")
                q["release"] = dict(converged=rel["converged"], clear_sets=rel["end"]["clear_sets"], U=rel["end"]["U"], word=rel["word"], set_radii=rel["end"]["set_radii"]) if rel else None
                q.update(kind="rung" if d in G[g]["targets"] else "bisection" if d in lad["mids"] else "inward", rise=q["U"] - free["end"]["U"],
                         counted=bool(an.get("final") and d in an["counted"]), why_not=why_not(q, st["room_rule"]))
                rr.append(q)
            # K8
            nodes = [(free["end"]["D"], free["end"]["U"], 0.0)] + [(q["D"], q["U"], q["f"]) for q in rr if q["kind"] != "inward" and q["converged"] and q["clear_sets"] == 1 and not q["evaded"]]
            # the named failure of K8, the hold energy counted into U at both strengths
            nodes_h = [(free["end"]["D"], free["end"]["U"], 0.0)] + [(q["D"], q["U"] + q["Eh"], q["f"]) for q in rr if q["kind"] != "inward" and q["converged"] and q["clear_sets"] == 1 and not q["evaded"]]
            k8 = []
            for d in (G[g]["targets"] if G[g]["mu2"] is None else G[g]["mu2"]):
                r = recs.get(f"R{g}_{tname(d)}_m80")
                if r is None:
                    k8.append(dict(d=d, passed=False, why="no record"))
                    continue
                q = rung_of(r)
                Hc = hermite(nodes, q["D"]) if (q["converged"] and q["clear_sets"] == 1) else None
                tol = max(0.02 * (q["U"] - free["end"]["U"]), 0.005)
                k8.append(dict(d=d, D=q["D"], U=q["U"], f=q["f"], Eh=q["Eh"], converged=q["converged"], clear_sets=q["clear_sets"], curve=Hc, gap=(abs(q["U"] - Hc) if Hc is not None else None),
                               tolerance=tol, gap_with_the_hold_energy_counted=(abs(q["U"] + q["Eh"] - Hc) if Hc is not None else None),
                               gap_with_the_hold_energy_counted_at_both=(abs(q["U"] + q["Eh"] - hermite(nodes_h, q["D"])) if Hc is not None else None),
                               passed=bool(Hc is not None and abs(q["U"] - Hc) <= tol), word=q["word"], evaded=q["evaded"]))
            k8_ok = bool(k8) and all(p["passed"] for p in k8)
            # K9
            k9 = dict(rows=obst, passed=None)
            small = [o for o in obst if o["Rd"] == 1.5]
            if small:
                o = small[0]
                k9["passed"] = bool(o["converged"] and o["Eo"] <= o["Eo_at_the_start"] + 1e-15 and abs(o["U"] - free["end"]["U"]) <= 1e-3 * abs(free["end"]["U"])
                                    and abs(o["radius"] - free["end"]["radius"]) <= 0.25 * 2 * G[g]["hb"] / (G[g]["N"] - 1) * 2)
                ok1 = [q for q in rr if q["kind"] != "inward" and q["converged"] and q["clear_sets"] == 1 and not q["evaded"]]
                xs = [free["end"]["separation"]] + [q["sep"] for q in ok1]
                ys = [free["end"]["U"]] + [q["U"] for q in ok1]
                for o in obst:
                    if o["Rd"] > 1.5 and o["separation"] is not None and min(xs) <= o["separation"] <= max(xs):
                        order = np.argsort(xs)
                        Uc = float(np.interp(o["separation"], np.array(xs)[order], np.array(ys)[order]))
                        o["U_first_hold_at_the_same_separation"] = Uc
                        o["gap_over_the_rise"] = (o["U"] - Uc) / (Uc - free["end"]["U"]) if Uc > free["end"]["U"] else None
                        Hd = hermite(nodes, o["D"])
                        o["U_first_hold_at_the_same_D"] = Hd
                        o["gap_over_the_rise_at_the_same_D"] = (o["U"] - Hd) / (Hd - free["end"]["U"]) if Hd is not None and Hd > free["end"]["U"] else None
            k9["fall_between_the_two_larger_disks"] = fall
            # continuation
            cont = []
            for pre in ("U", "V"):
                for d in G[g]["targets"]:
                    r = recs.get(f"{pre}{g}_{tname(d)}")
                    fr = lad["rungs"].get(d)
                    if r is not None:
                        q = rung_of(r)
                        rise = (fr["U"] - free["end"]["U"]) if fr else None
                        cont.append(dict(chain=r["chain"], direction="up" if pre == "U" else "down", d=d, D=q["D"], U=q["U"], f=q["f"], converged=q["converged"], clear_sets=q["clear_sets"], evaded=q["evaded"], share=q["share"],
                                         sep=q["sep"], U_fresh=(fr["U"] if fr else None), gap=(q["U"] - fr["U"] if fr else None),
                                         gap_over_the_rise=((q["U"] - fr["U"]) / rise if fr and rise else None), iterations=q["iterations"]))
            # R1 to R3
            r1_label = an["label"] if an.get("final") else "not final: " + json.dumps(an.get("needs"))
            r1_word = an.get("word") or "no label"
            if not k8_ok and an.get("final"):
                r1_label = "no label: K8 is not met on this grid (no R1); the rules applied to the chains as run give: " + an["label"]
                r1_word = "no label"
            if k9["passed"] is False and an.get("final"):
                r1_label += " [one kind of hold: K9 is not met]"
            outward = [q for q in rr if q["kind"] in ("rung", "bisection")]
            r2_word, r2_label = read_R2(outward, len(an.get("counted", [])) if an.get("final") else 0)
            r3_word, r3_label = read_R3(fam, bool(an.get("final") and an.get("top")), expected=len(H.FAMILY_113 if G[g]["N"] == 113 else H.FAMILY))
            blocked = None
            box14 = "14" + s
            if k2_fail.get(box14):
                blocked = ("K2 is not met on the base of half-box 14 at this spacing (its cost in the plan: no ladder on that base, and with it no room rule for "
                           "either box at this spacing), so every label of this spacing is withheld. As run, not a label: ")
            # the literal reading of eligibility: with no room rule no rung is eligible, so no rung is counted
            if st["room_rule"] is None:
                lit = dict(R1="no label: no room rule, so no rung is eligible and none is counted (the literal reading)",
                           R2=read_R2(outward, 0)[1] + " (the literal reading: no counted rung)",
                           R3=read_R3(fam, False)[1] + " (the literal reading: no counted rung)")
            else:
                lit = dict(R1=r1_label, R2=r2_label, R3=r3_label)
            S["ladders"][g] = dict(as_run_literal=lit, free=dict(U=free["end"]["U"], D=free["end"]["D"], separation=free["end"]["separation"], radius=free["end"]["radius"]), rungs=rr, analysis={k: v for k, v in an.items() if k not in ("counted_rungs",)},
                                   uncut=uncut_print(lad), uncut_with_evaded=uncut_print(lad, True), K8=dict(passed=k8_ok, rows=k8), K9=k9, continuation=cont, family=fam, withheld=blocked,
                                   R1=dict(word="no label" if blocked else r1_word, label=("no label: " + blocked + r1_label) if blocked else r1_label, as_run=r1_label),
                                   R2=dict(word="no label" if blocked else r2_word, label=("no label: " + blocked + r2_label) if blocked else r2_label, as_run=r2_label),
                                   R3=dict(word="no label" if blocked else r3_word, label=("no label: " + blocked + r3_label) if blocked else r3_label, as_run=r3_label,
                                           members={m["member"]: m["word"] for m in fam}))
            words[g] = dict(R1=S["ladders"][g]["R1"]["label"], R2=S["ladders"][g]["R2"]["label"], R3=S["ladders"][g]["R3"]["label"], R3m=S["ladders"][g]["R3"]["members"],
                            literal=dict(lit, R3m=S["ladders"][g]["R3"]["members"]),
                            as_run=dict(R1=r1_label, R2=r2_label, R3=r3_label, R3m=S["ladders"][g]["R3"]["members"]))
    C["K8"] = dict(passed=bool(S["ladders"]) and all(L["K8"]["passed"] for L in S["ladders"].values()) and len(S["ladders"]) == 4, name="K8 two strengths on one curve",
                   numbers={g: L["K8"] for g, L in S["ladders"].items()}, counter_input="with the hold energy counted into U the gaps are those of the column gap_with_the_hold_energy_counted")
    k9s = {g: L["K9"] for g, L in S["ladders"].items() if L["K9"]["rows"]}
    C["K9"] = dict(passed=bool(k9s) and all(v["passed"] for v in k9s.values()), name="K9 the second hold", numbers=k9s)
    S["controls"] = C
    # --- R4
    S["R4"] = {}
    for n in ("E0a", "E0b", "E14c", "E14d", "Z10a", "Z10b", "Z14a", "Z14b"):
        if n in recs:
            S["R4"][n] = exit_chain(recs[n])
            if not C["K11"]["passed"]:
                S["R4"][n]["label"] = "no label: K11 is not met (no R4); as run: " + S["R4"][n]["label"]
                S["R4"][n]["word"] = "no label"
            if n.startswith("Z") and k2_fail.get("14" + n[-1]):
                S["R4"][n]["label"] = "no label: K2 is not met on the base of half-box 14 at this spacing, so every label of this spacing is withheld. As run, not a label: " + S["R4"][n]["label"]
                S["R4"][n]["word"] = "no label"
    S["R4_reread_of_m5_46"] = {n: reread_m5_46(a.out_dir, n) for n in ("X0a", "X0b", "X14c", "X14d")}
    # --- R5
    S["R5"] = {}
    for box in ("10", "14"):
        if box + "a" in words and box + "b" in words:
            w, lab = read_R5(words[box + "a"], words[box + "b"])
            w2, lab2 = read_R5(words[box + "a"]["as_run"], words[box + "b"]["as_run"])
            w3, lab3 = read_R5(words[box + "a"]["literal"], words[box + "b"]["literal"])
            S["R5"][box] = dict(word=w, label=lab, as_run=lab2, as_run_literal=lab3)
    S["hashes_of_m5_46_end_files"] = "see checkpoints/m5_50/m5_46_hashes_before.txt and m5_46_hashes_after.txt (identical)"
    out = Path(a.out_dir) / "m5_50_summary.json"
    out.write_text(json.dumps(S, indent=None, separators=(",", ":"), default=float) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    for k in sorted(C, key=lambda x: int(x[1:])):
        print(k, "PASS" if C[k]["passed"] else "FAIL")
    for g, L in S["ladders"].items():
        print(g, "R1:", L["R1"]["label"][:300])
        print(g, "R2:", L["R2"]["label"][:300])
        print(g, "R3:", L["R3"]["label"][:300])
    for n, r in S["R4"].items():
        print(n, "R4:", r["label"][:300])
    for b, r in S["R5"].items():
        print("R5 half-box", b, r["label"][:300])
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k10", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--power", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_50")
    a = ap.parse_args()
    if a.k10:
        sys.exit(0 if k10(a.work_dir / "k10_readers.json") else 1)
    elif a.summary:
        summary(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
