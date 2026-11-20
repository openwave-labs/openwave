"""M5.46: the labels R1 to R4 of every chain from its saved reads, the tables and the plots.

A read (m5_46_degree.read_state, saved by m5_46_pull.py) holds the sets of 8 pierced plaquettes or more with
their radius, least distance d to the faces and faces touched (a set with d <= h touches; otherwise it is
clear), and for every nested cube surface one character: |q| when liftable, x when pierced, g when the gap is
at or under 0.2. The loop of a read is the largest clear set, else the largest set. The box surface is the
outermost cube (half-width L - h).

    q(S) = (1 / 4 pi) sum_t 2 atan2( n_a.(n_b x n_c), 1 + n_a.n_b + n_b.n_c + n_c.n_a )

The frozen readings (m5_46_task_details.md, section 4) as this script applies them, with R_0 the radius of the
loop at read 0 (the pulled value), R_b the radius of the base loop, R_c the radius at the last read with a
clear loop, and tol = h / 2:

    R1  SPLIT      a read with two clear sets or more
        EXPANDED   R_c - R_0 > tol (the radius grows from the pulled value), or a read at which the sets that
                   touch the faces touch two faces or more
        NOT HELD   SPLIT or EXPANDED (both are named when both occur)
        HELD       not NOT HELD; exactly one clear set at every read up to the end of the chain (for a chain
                   that ends by EXIT: up to the last read before the first read without a clear set, with no
                   clear set after it); and R_c - R_0 < -tol (the radius falls from the pulled value)
        no label   anything else (the outcome is stated beside it); the chains without a pull (F0) cannot be
                   HELD, since the HELD clause needs a pulled value
    R2  a read counts when a liftable surface surrounds its loop; any other read is "no read"
        NOT HELD   a counted read with a liftable surface around the loop that is not 1, or any read with a
                   liftable surface that is neither 0 nor 1
        no label   fewer than ten reads in the chain, or (without a NOT HELD read) fewer than ten counted reads
        HELD       otherwise
    R3  read only for a chain that ends by EXIT with its sets touching one face in all
        NOT HELD   a read with the box surface at 0 and a clear set
        HELD       not NOT HELD; at the last read with the box surface at 1 the loop is within 2 of a face;
                   and no later read has a clear set
        no label   anything else, and every chain that does not exit through one face
    R4  per pair of spacings (F0a F0b, P10a P10b, P14c P14d): HELD when R1, R2 and R3 carry the same label
        (with SPLIT or EXPANDED) at both; NOT HELD when one differs

The base chains (B0) carry no label. The chains added during the run (X0a, X0b, X14c, X14d: a free chain
that ended at the gate, continued with the gate stop off) are outside the frozen table and carry no label;
their reads are tabled by themselves. Nothing here sets an icon.

Usage:  python m5_46_reads.py [--data DIR] [--plots DIR] [--tables FILE]
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

ORDER = ("B0a", "B0b", "B0c", "B0d", "F0a", "F0b", "P10a", "P10b", "P14c", "P14d", "P14a", "H10a", "G10a")
ADDED = ("X0a", "X0b", "X14c", "X14d")
PAIRS = (("F0a", "F0b"), ("P10a", "P10b"), ("P14c", "P14d"))
SNAPS_STD = (50, 100, 200, 300, 500, 1000, 2000, 5000, 10000)
FACE_NAMES = ("-x", "+x", "-y", "+y", "-z", "+z")
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#d9d8d3"


def faces_of(mask):
    return [FACE_NAMES[b] for b in range(6) if mask >> b & 1]


def derive(rec):
    """Per-read quantities of a chain record."""
    h = rec["model"]["spacing"]
    out = []
    for r in rec["reads"]:
        sets = r["sets"]
        touch = [s for s in sets if s[8] <= h + 1e-9]
        clear = [s for s in sets if s[8] > h + 1e-9]
        mask = 0
        for s in touch:
            mask |= s[9]
        lp = sets[r["loop"]] if r["loop"] is not None else None
        around = [int(c) for c, a in zip(r["q"], r["ar"]) if a == "1" and c.isdigit()]
        out.append(dict(it=r["it"], U=r["U"], g=r["g"], n_sets=len(sets), n_clear=len(clear), n_touch=len(touch), faces=faces_of(mask),
                        loop=lp, loop_clear=bool(lp is not None and lp[8] > h + 1e-9), radius=(lp[1] if lp else None),
                        distance=(lp[8] if lp else None), around=around, liftable=[int(c) for c in r["q"] if c.isdigit()],
                        box=r["q"][-1], q=r["q"], ar=r["ar"], frag=r["frag"], Ut=r.get("Ut"),
                        others=[int(c) for c, a in zip(r["q"], r["ar"]) if a == "0" and c.isdigit()]))
    return out


def label_chain(rec):
    h = rec["model"]["spacing"]
    tol = h / 2
    d = derive(rec)
    base = rec["spec"]["kind"] in ("base", "added")
    pulled = rec["start"].get("separation") is not None
    its = [x["it"] for x in d]
    res = dict(chain=rec["chain"], reads=len(d), ended=rec["ended"], iterations=rec["iterations"])
    clear_r = [(x["it"], x["radius"]) for x in d if x["loop_clear"]]
    R0 = d[0]["radius"]
    Rb = rec["start"].get("base_radius", None)
    Rc = clear_r[-1][1] if clear_r else None
    all_r = [x["radius"] for x in d if x["radius"] is not None]
    res.update(radius_at_read_0=R0, base_radius=Rb, radius_at_the_last_clear_read=Rc, last_clear_read=(clear_r[-1][0] if clear_r else None),
               radius_min_clear=(min(r for _, r in clear_r) if clear_r else None), radius_max_clear=(max(r for _, r in clear_r) if clear_r else None),
               radius_at_the_last_read=d[-1]["radius"], sets_at_the_last_read=d[-1]["n_sets"], clear_at_the_last_read=d[-1]["n_clear"],
               distance_at_the_last_read=d[-1]["distance"], box_at_the_last_read=d[-1]["box"], q_at_the_last_read=d[-1]["q"],
               radius_max_any=(max(all_r) if all_r else None))
    split = [x["it"] for x in d if x["n_clear"] >= 2]
    multi = [x["it"] for x in d if len(x["faces"]) >= 2]
    first_touch = next((x["it"] for x in d if x["n_touch"] > 0), None)
    first_no_clear = next((x["it"] for x in d if x["n_clear"] == 0), None)
    no_set = [x["it"] for x in d if x["n_sets"] == 0]
    faces_all = sorted({f for x in d for f in x["faces"]}, key=FACE_NAMES.index)
    grows = bool(Rc is not None and R0 is not None and Rc - R0 > tol)
    falls = bool(Rc is not None and R0 is not None and Rc - R0 < -tol)
    res.update(reads_with_two_clear_sets=len(split), first_read_with_two_clear_sets=(split[0] if split else None),
               reads_with_two_faces_touched=len(multi), first_read_with_two_faces_touched=(multi[0] if multi else None),
               sets_at_the_first_read_with_two_faces=(next(x["n_sets"] for x in d if len(x["faces"]) >= 2) if multi else None),
               first_read_touching_a_face=first_touch, first_read_without_a_clear_set=first_no_clear, reads_without_a_set=len(no_set),
               first_read_without_a_set=(no_set[0] if no_set else None), faces_touched=faces_all, radius_change=(None if Rc is None or R0 is None else Rc - R0),
               tolerance=tol)
    # R2 numbers (computed for every chain; a label only for the released ones)
    counted = [x for x in d if x["around"]]
    bad_a = [x["it"] for x in counted if any(v != 1 for v in x["around"])]
    bad_b = [x["it"] for x in d if any(v not in (0, 1) for v in x["liftable"])]
    res.update(reads_counted=len(counted), reads_no_read=len(d) - len(counted), reads_around_not_1=len(bad_a), first_around_not_1=(bad_a[0] if bad_a else None),
               reads_outside_0_and_1=len(bad_b), first_outside_0_and_1=(bad_b[0] if bad_b else None),
               surfaces_around_per_counted_read=(float(np.mean([len(x["around"]) for x in counted])) if counted else 0.0),
               last_counted_read=(counted[-1]["it"] if counted else None))
    allq = "".join(x["q"] for x in d)
    res.update(surface_reads=len(allq), surface_reads_liftable=sum(c.isdigit() for c in allq), surface_reads_pierced=allq.count("x"),
               surface_reads_gap=allq.count("g"), surface_reads_lift_not_consistent=allq.count("l"),
               values_on_liftable_surfaces_not_around_the_loop=sorted({v for x in d for v in x["others"]}),
               most_fragments_at_a_read=max(x["frag"][0] for x in d), reads_with_fragments=sum(x["frag"][0] > 0 for x in d),
               most_fragment_plaquettes_at_a_read=max(x["frag"][1] for x in d))
    box0_clear = [x["it"] for x in d if x["box"] == "0" and x["n_clear"] >= 1]
    box1 = [x for x in d if x["box"] == "1"]
    res.update(reads_box_0_with_a_clear_set=len(box0_clear), first_box_0_with_a_clear_set=(box0_clear[0] if box0_clear else None),
               last_read_with_the_box_at_1=(box1[-1]["it"] if box1 else None), distance_at_that_read_box=(box1[-1]["distance"] if box1 else None),
               first_read_with_the_box_at_0=next((x["it"] for x in d if x["box"] == "0"), None))
    if rec["spec"]["kind"] == "added":
        withset = [x for x in d if x["loop"] is not None]
        c0 = np.array(d[0]["loop"][2:5])
        gone = next((x["it"] for x in withset if np.linalg.norm(np.array(x["loop"][2:5]) - c0) >= 0.5 and x["it"] > 0 and
                     all(np.linalg.norm(np.array(y["loop"][2:5]) - c0) >= 0.5 for y in withset if y["it"] >= x["it"])), None)
        plateau = [x for x in d if gone is None or x["it"] < gone]
        low = min(plateau, key=lambda x: x["g"])
        last = withset[-1]
        cen = np.array(last["loop"][2:5])
        ax = int(np.argmax(np.abs(cen)))
        res.update(least_gmax_before_the_departure=low["g"], iteration_of_the_least_gmax=low["it"], U_at_the_least_gmax=low["U"],
                   departure_iteration=gone, last_read_with_a_set=last["it"], center_at_that_read=cen.tolist(), radius_at_that_read=last["radius"],
                   distance_at_that_read=last["distance"], sets_at_that_read=last["n_sets"], q_at_that_read=last["q"],
                   face_nearest_to_the_loop_at_that_read=FACE_NAMES[2 * ax + (1 if cen[ax] > 0 else 0)],
                   radius_min_with_a_set=min(x["radius"] for x in withset), radius_max_with_a_set=max(x["radius"] for x in withset),
                   most_sets_at_a_read=max(x["n_sets"] for x in d))
    if base:
        note = "base chain" if rec["spec"]["kind"] == "base" else "added chain, outside the frozen table"
        res.update(R1="no label", R1_note=note, R2="no label", R2_note=note, R3="no label", R3_note=note)
        return res, d
    # R1
    kinds = (["SPLIT"] if split else []) + (["EXPANDED"] if (grows or multi) else [])
    if kinds:
        why = []
        if split:
            why.append(f"two clear sets at {len(split)} reads, first at iteration {split[0]}")
        if grows:
            why.append(f"the radius of the clear loop went from {R0:.3f} to {Rc:.3f} (last clear read, iteration {clear_r[-1][0]})")
        if multi:
            why.append(f"two faces or more touched from iteration {multi[0]} ({res['sets_at_the_first_read_with_two_faces']} sets at that read; faces touched in all: {', '.join(faces_all)})")
        res.update(R1="NOT HELD (" + ", ".join(kinds) + ")", R1_note="; ".join(why))
    else:
        if rec["ended"] == "EXIT" and first_no_clear is not None:
            before = [x for x in d if x["it"] < first_no_clear]
            after = [x for x in d if x["it"] >= first_no_clear]
            one = all(x["n_clear"] == 1 for x in before) and all(x["n_clear"] == 0 for x in after)
        else:
            one = all(x["n_clear"] == 1 for x in d)
        if not pulled:
            res.update(R1="no label", R1_note=f"no pull, so the HELD clause has no pulled value; neither SPLIT nor EXPANDED; "
                                              f"{'one clear set at every read' if one else 'not one clear set at every read'}; radius {fmt(R0)} at read 0, {fmt(Rc)} at the last clear read")
        elif one and falls:
            res.update(R1="HELD", R1_note=f"one clear set at every read; the radius fell from {R0:.3f} to {Rc:.3f} (base {Rb:.3f})")
        else:
            res.update(R1="no label", R1_note=f"neither column: {'one clear set at every read' if one else 'not one clear set at every read'}; radius {fmt(R0)} at read 0, "
                                              f"{fmt(Rc)} at the last clear read (base {fmt(Rb)}); reads without a set {len(no_set)}; first read touching a face {fmt(first_touch)}")
    # R2
    if len(d) < 10:
        res.update(R2="no label", R2_note=f"{len(d)} reads in the chain")
    elif bad_a or bad_b:
        res.update(R2="NOT HELD", R2_note=f"{len(bad_a)} counted reads with a surface around the loop that is not 1 (first at {bad_a[0] if bad_a else None}); "
                                           f"{len(bad_b)} reads with a liftable surface outside 0 and 1 (first at {bad_b[0] if bad_b else None})")
    elif len(counted) < 10:
        res.update(R2="no label", R2_note=f"{len(counted)} counted reads of {len(d)}")
    else:
        res.update(R2="HELD", R2_note=f"{len(counted)} counted reads of {len(d)}, every surface around the loop at 1, every liftable surface at 0 or 1")
    # R3
    if rec["ended"] == "EXIT" and len(faces_all) == 1:
        later_clear = [x["it"] for x in d if box1 and x["it"] > box1[-1]["it"] and x["n_clear"] >= 1]
        if box0_clear:
            res.update(R3="NOT HELD", R3_note=f"the box surface at 0 with a clear set at {len(box0_clear)} reads, first at {box0_clear[0]}")
        elif box1 and box1[-1]["distance"] is not None and box1[-1]["distance"] <= 2.0 and not later_clear:
            res.update(R3="HELD", R3_note=f"last read with the box surface at 1: iteration {box1[-1]['it']}, the loop {box1[-1]['distance']:.2f} from a face ({faces_all[0]}); no clear set after it")
        else:
            res.update(R3="no label", R3_note=f"exit through {faces_all[0]}; last box-at-1 read {box1[-1]['it'] if box1 else None} with the loop at distance "
                                              f"{box1[-1]['distance'] if box1 else None}; later reads with a clear set {len(later_clear)}")
    else:
        how = "did not end by EXIT" if rec["ended"] != "EXIT" else f"exit with {len(faces_all)} faces touched"
        res.update(R3="no label", R3_note=f"{how}; reads with the box surface at 0 and a clear set: {len(box0_clear)}")
    return res, d


def fmt(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def duration(sec):
    m, s = divmod(int(round(sec)), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def tables(recs, labs, ders, dil, ctl):
    L = []
    L.append("**Table 1. Every chain: how it ran, how it ended, the loop and the labels.**\n")
    L.append("| Chain | Half-box, spacing (grid) | Start | Faces | Descent | Iterations | How it ended | max \\|dU/dm\\|/h³ at the end | U at read 0 | U at the end | "
             "Radius of the loop at read 0 | Radius at the last read | R1 | R2 | R3 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for n in ORDER:
        if n not in recs:
            continue
        r, s, a = recs[n], recs[n]["start"], labs[n]
        start = "seed" if s["kind"] == "seed" else (f"{s['base']}, no pull" if s["kind"] == "base" else f"{s['base']} pulled to {s['separation']:g} (s = {s['s']:.4f})")
        faces = "free" if r["model"]["free_faces"] else ("shell held at the seed" if s["kind"] == "seed" else "shell held at the pulled field")
        desc = "L-BFGS-B" if not r["step"] else f"steepest, step {r['step']:g}"
        rl = "no set" if a["radius_at_the_last_read"] is None else f"{a['radius_at_the_last_read']:.3f}"
        r0 = "no set" if a["radius_at_read_0"] is None else f"{a['radius_at_read_0']:.3f}"
        L.append(f"| {n} | {r['model']['half_box']:g}, {r['model']['spacing']:g} ({r['model']['N']}³) | {start} | {faces} | {desc} | {r['iterations']} | {r['ended']} | "
                 f"{r['gmax_over_h3_at_the_end']:.1e} | {s['U']:.4f} | {r['U_at_the_end']:.4f} | {r0} | {rl} | {a['R1']} | {a['R2']} | {a['R3']} |")
    L.append("\n**Table 2. What each label rests on (the released chains).**\n")
    L.append("| Chain | R1 | R2 | R3 |")
    L.append("| --- | --- | --- | --- |")
    for n in ORDER:
        if n in labs and not n.startswith("B0"):
            a = labs[n]
            L.append(f"| {n} | {a['R1']}: {a['R1_note']} | {a['R2']}: {a['R2_note']} | {a['R3']}: {a['R3_note']} |")
    L.append("\n**Table 3. The reads of the line and of the charge, per chain.**\n")
    L.append("| Chain | Reads | Reads with one clear set, two or more, none | Faces touched | Counted reads (a liftable surface around the loop) | No read | "
             "Mean surfaces around the loop per counted read | Counted reads with a surface around the loop not at 1 | Reads with a liftable surface outside 0 and 1 | "
             "Box surface at the last read | Reads with the box surface at 0 and a clear set | Surface reads: liftable, pierced, gap | "
             "Values on liftable surfaces that do not surround the loop | Reads with fragments (most fragments, most plaquettes in them) |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for n in ORDER + ADDED:
        if n in labs:
            a, d = labs[n], ders[n]
            one, two, none = sum(x["n_clear"] == 1 for x in d), sum(x["n_clear"] >= 2 for x in d), sum(x["n_clear"] == 0 for x in d)
            L.append(f"| {n} | {a['reads']} | {one}, {two}, {none} | {', '.join(a['faces_touched']) or 'none'} | {a['reads_counted']} | {a['reads_no_read']} | "
                     f"{a['surfaces_around_per_counted_read']:.1f} | {a['reads_around_not_1']} | {a['reads_outside_0_and_1']} | {a['box_at_the_last_read']} | "
                     f"{a['reads_box_0_with_a_clear_set']} | {a['surface_reads_liftable']}, {a['surface_reads_pierced']}, {a['surface_reads_gap']} | "
                     f"{', '.join(str(v) for v in a['values_on_liftable_surfaces_not_around_the_loop']) or 'none'} | "
                     f"{a['reads_with_fragments']} ({a['most_fragments_at_a_read']}, {a['most_fragment_plaquettes_at_a_read']}) |")
    L.append("\n**Table 4. R4, the two spacings.**\n")
    L.append("| Pair | R1 at 0.5 | R1 at 0.25 | R2 at 0.5 | R2 at 0.25 | R3 at 0.5 | R3 at 0.25 | R4 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    r4 = {}
    for a, b in PAIRS:
        if a in labs and b in labs:
            same = all(labs[a][k] == labs[b][k] for k in ("R1", "R2", "R3"))
            diff = [k for k in ("R1", "R2", "R3") if labs[a][k] != labs[b][k]]
            r4[f"{a} {b}"] = dict(label="HELD" if same else "NOT HELD", differing=diff)
            L.append(f"| {a}, {b} | {labs[a]['R1']} | {labs[b]['R1']} | {labs[a]['R2']} | {labs[b]['R2']} | {labs[a]['R3']} | {labs[b]['R3']} | "
                     f"{'HELD' if same else 'NOT HELD (' + ', '.join(diff) + ' differ)'} |")
    L.append("\n**Table 5. The pulled state and its first reads: U, the potential term U_V, the radius of the loop.**\n")
    L.append("| Chain | Base | Read 0 (pulled) | Iteration 10 | Iteration 20 | Iteration 50 | Iteration 100 | The end |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for n in ORDER:
        if n in recs and recs[n]["start"]["kind"] == "pulled":
            st, d = recs[n]["start"], ders[n]
            bd = ders[st["base"]][-1] if st["base"] in ders else None

            def cell(x):
                return "n/a" if x is None else f"{x['U']:.3f}, {x['Ut'][2]:.3f}, {'no set' if x['radius'] is None else format(x['radius'], '.3f')}"

            at = {x["it"]: x for x in d}
            L.append(f"| {n} | {cell(bd)} | {cell(d[0])} | {cell(at.get(10))} | {cell(at.get(20))} | {cell(at.get(50))} | {cell(at.get(100))} | {cell(d[-1])} |")
    if dil:
        L.append("\n**Table 6. U along the dilation of each base state.**\n")
        L.append("| Base | Half-box, spacing | Radius of the base loop | U at s = 1 | U at separation 10 (s) | U at separation 13.8 (s) | Largest U on the curve (separation) | "
                 "U_V at s = 1, at 10, at 13.8 | L₁ term at s = 1, at 10, at 13.8 | Commutator term at s = 1, at 10, at 13.8 | Steps of the curve on which U falls |")
        L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for b, v in dil.items():
            c = v["curve"]
            at = lambda s: min(c, key=lambda r: abs(r["s"] - s))  # noqa: E731
            a1, a10, a14 = at(1.0), at(v["s_at_separation_10"]), at(v["s_at_separation_13p8"])
            top = max(c, key=lambda r: r["U"])
            falls = sum(c[i + 1]["U"] < c[i]["U"] for i in range(len(c) - 1))
            L.append(f"| {b} | {v['half_box']:g}, {v['spacing']:g} | {v['base_radius']:.4f} | {a1['U']:.4f} | {a10['U']:.4f} ({a10['s']:.4f}) | {a14['U']:.4f} ({a14['s']:.4f}) | "
                     f"{top['U']:.4f} ({top['separation']:.2f}) | {a1['UV']:.3f}, {a10['UV']:.3f}, {a14['UV']:.3f} | {a1['U2']:.3f}, {a10['U2']:.3f}, {a14['U2']:.3f} | "
                     f"{a1['U4']:.3f}, {a10['U4']:.3f}, {a14['U4']:.3f} | {falls} of {len(c) - 1} |")
    xs = [n for n in ADDED if n in labs]
    if xs:
        L.append("\n**Table 7. The chains added during the run (outside the frozen table; no label).**\n")
        L.append("| Quantity | " + " | ".join(xs) + " |")
        L.append("| --- |" + " --- |" * len(xs))

        def row(name, f):
            L.append(f"| {name} | " + " | ".join(f(labs[n], recs[n]) for n in xs) + " |")

        row("Starts from the end state of", lambda a, r: r["start"]["base"])
        row("Half-box, spacing (grid)", lambda a, r: f"{r['model']['half_box']:g}, {r['model']['spacing']:g} ({r['model']['N']}³)")
        row("U and max \\|dU/dm\\|/h³ at read 0", lambda a, r: f"{r['start']['U']:.4f}, {r['start']['gmax_over_h3']:.1e}")
        row("Iterations, how it ended", lambda a, r: f"{r['iterations']}, {r['ended']}")
        row("Least max \\|dU/dm\\|/h³ before the departure (iteration)", lambda a, r: f"{a['least_gmax_before_the_departure']:.1e} ({a['iteration_of_the_least_gmax']})")
        row("U at that read", lambda a, r: f"{a['U_at_the_least_gmax']:.4f}")
        row("Departure: first read from which the center stays 0.5 or more from its start", lambda a, r: fmt(a["departure_iteration"]))
        row("Last read with a set: iteration", lambda a, r: str(a["last_read_with_a_set"]))
        row("Center of the loop at that read", lambda a, r: "({:.2f}, {:.2f}, {:.2f})".format(*a["center_at_that_read"]))
        row("Radius, least distance to the faces at that read", lambda a, r: f"{a['radius_at_that_read']:.3f}, {a['distance_at_that_read']:.2f}")
        row("Face nearest to the loop at that read", lambda a, r: a["face_nearest_to_the_loop_at_that_read"])
        row("Faces touched by a set at any read", lambda a, r: ", ".join(a["faces_touched"]) or "none")
        row("Radius range while a set exists", lambda a, r: f"{a['radius_min_with_a_set']:.3f} to {a['radius_max_with_a_set']:.3f}")
        row("Most sets at a read", lambda a, r: str(a["most_sets_at_a_read"]))
        row("Reads with two clear sets", lambda a, r: str(a["reads_with_two_clear_sets"]))
        row("Reads with two faces touched", lambda a, r: str(a["reads_with_two_faces_touched"]))
        row("Counted reads; of them with a surface around the loop not at 1", lambda a, r: f"{a['reads_counted']}; {a['reads_around_not_1']}")
        row("Reads with a liftable surface outside 0 and 1", lambda a, r: str(a["reads_outside_0_and_1"]))
        row("Last read with the box surface at 1: iteration, distance of the loop to the faces", lambda a, r: f"{fmt(a['last_read_with_the_box_at_1'])}, {fmt(a['distance_at_that_read_box'], 2)}")
        row("First read with the box surface at 0", lambda a, r: fmt(a["first_read_with_the_box_at_0"]))
        row("Reads with the box surface at 0 and a clear set", lambda a, r: str(a["reads_box_0_with_a_clear_set"]))
        row("U at the end", lambda a, r: f"{r['U_at_the_end']:.4f}")
    L.append("\n**Table 8. Wall time per chain.**\n")
    L.append("| Chain | Iterations | Evaluations of U by the descent | Wall time |")
    L.append("| --- | --- | --- | --- |")
    for n in ORDER + ADDED:
        if n in recs:
            L.append(f"| {n} | {recs[n]['iterations']} | {recs[n]['evaluations']} | {duration(recs[n]['seconds'])} |")
    if ctl:
        L.append(f"\n**Controls:** {'ALL PASS' if ctl['passed'] else 'FAIL'} on " + ", ".join(ctl["base_states"]) + "\n")
        for c in ctl["controls"]:
            L.append(f"- {c['control']}: {'PASS' if c['passed'] else 'FAIL'}")
    return "\n".join(L) + "\n", r4


# ----------------------------------------------------------------------------- plots

def _axes_style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _panels(names, ncols=5, w=3.3, hgt=2.6):
    import matplotlib.pyplot as plt
    nrows = int(np.ceil(len(names) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(w * ncols, hgt * nrows), squeeze=False)
    for ax in axs.ravel()[len(names):]:
        ax.set_visible(False)
    return fig, axs.ravel()[:len(names)]


def plot_series(recs, ders, labs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter
    names = [n for n in ORDER + ADDED if n in recs]
    # loop radius against iteration
    fig, axs = _panels(names)
    for ax, n in zip(axs, names):
        d, a = ders[n], labs[n]
        it = np.array([x["it"] for x in d]) + 1
        rad = np.array([np.nan if x["radius"] is None else x["radius"] for x in d])
        cl = np.array([x["loop_clear"] for x in d])
        has = ~np.isnan(rad)
        ax.plot(it[has], rad[has], color=GRID, lw=1.0, zorder=1)
        ax.plot(it[has & cl], rad[has & cl], "o", color=BLUE, ms=3.2, zorder=3)
        ax.plot(it[has & ~cl], rad[has & ~cl], "s", mfc="white", mec=ORANGE, ms=3.4, mew=1.0, zorder=2)
        if a["base_radius"]:
            ax.axhline(a["base_radius"], color=MUTED, lw=0.9, ls=":")
        ax.axhline(recs[n]["model"]["half_box"], color=MUTED, lw=0.9, ls="--")
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0, recs[n]["model"]["half_box"] * 1.08)
        ax.set_title(f"{n}: {recs[n]['ended'].split(',')[0].lower()} at {recs[n]['iterations']}", fontsize=9, color=INK)
        _axes_style(ax)
    for ax in axs[-5:]:
        ax.set_xlabel("iteration + 1", fontsize=8, color=MUTED)
    for ax in axs[::5]:
        ax.set_ylabel("radius of the loop", fontsize=8, color=MUTED)
    fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", ms=5, label="the loop, clear of the faces"),
                        Line2D([], [], marker="s", mfc="white", mec=ORANGE, ls="", ms=5, label="largest set, touching a face"),
                        Line2D([], [], color=MUTED, ls=":", label="radius of the base loop"), Line2D([], [], color=MUTED, ls="--", label="half-box")],
               loc="lower right", frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.04))
    fig.suptitle("M5.46: radius of the loop at every read (the reader of M5.44)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "m5_46_radius.png", dpi=110)
    plt.close(fig)
    # least distance to the faces
    fig, axs = _panels(names)
    for ax, n in zip(axs, names):
        d = ders[n]
        it = np.array([x["it"] for x in d]) + 1
        dist = np.array([np.nan if x["distance"] is None else x["distance"] for x in d])
        has = ~np.isnan(dist)
        ax.plot(it[has], dist[has], "-", color=BLUE, lw=1.4)
        ax.axhline(recs[n]["model"]["spacing"], color=ORANGE, lw=0.9, ls="--")
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0, None)
        ax.set_title(n, fontsize=9, color=INK)
        _axes_style(ax)
    for ax in axs[-5:]:
        ax.set_xlabel("iteration + 1", fontsize=8, color=MUTED)
    for ax in axs[::5]:
        ax.set_ylabel("least distance to the faces", fontsize=8, color=MUTED)
    fig.legend(handles=[Line2D([], [], color=BLUE, label="the loop of the read"), Line2D([], [], color=ORANGE, ls="--", label="one cell: at or under it the set touches")],
               loc="lower right", frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.06))
    fig.suptitle("M5.46: least distance of the loop to the faces at every read", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "m5_46_distance.png", dpi=110)
    plt.close(fig)
    # U against iteration
    fig, axs = _panels(names)
    for ax, n in zip(axs, names):
        d = ders[n]
        ax.plot(np.array([x["it"] for x in d]) + 1, [x["U"] for x in d], "-", color=BLUE, lw=1.4)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0, None)
        ax.set_title(f"{n}: U {d[0]['U']:.2f} to {d[-1]['U']:.2f}", fontsize=9, color=INK)
        _axes_style(ax)
    for ax in axs[-5:]:
        ax.set_xlabel("iteration + 1", fontsize=8, color=MUTED)
    for ax in axs[::5]:
        ax.set_ylabel("U", fontsize=8, color=MUTED)
    fig.suptitle("M5.46: U at every read", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "m5_46_energy.png", dpi=110)
    plt.close(fig)
    # the degree on each nested surface
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cols = ["#e9e8e3", BLUE, "#6250d6", ORANGE, "#f6c7b3"]            # 0, 1, another whole number, pierced, gap
    fig, axs = _panels(names, ncols=4, w=4.2, hgt=2.9)
    for ax, n in zip(axs, names):
        r, d = recs[n], ders[n]
        W = r["half_widths"]
        img = np.zeros((len(W), len(d)))
        for j, x in enumerate(d):
            for i, c in enumerate(x["q"]):
                img[i, j] = {"0": 0, "1": 1, "x": 3, "g": 4, "l": 3}.get(c, 2)
        ax.imshow(img, origin="lower", aspect="auto", cmap=ListedColormap(cols), vmin=-0.5, vmax=4.5, interpolation="nearest",
                  extent=(-0.5, len(d) - 0.5, -0.5, len(W) - 0.5))
        mn = [np.nan if x["loop"] is None else np.searchsorted(W, x["loop"][10] + 1e-9) - 0.5 for x in d]
        ax.plot(range(len(d)), mn, color=INK, lw=0.9)
        ticks = np.unique(np.linspace(0, len(d) - 1, 5).astype(int))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(d[t]["it"]) for t in ticks])
        yt = [i for i, w in enumerate(W) if abs(w - round(w)) < 1e-9 and int(round(w)) % 2 == 0] + [len(W) - 1]
        ax.set_yticks(sorted(set(yt)))
        ax.set_yticklabels([f"{W[i]:g}" for i in sorted(set(yt))])
        ax.set_title(f"{n}: R2 {labs[n]['R2']}" if n[0] not in "BX" else n, fontsize=9, color=INK)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(MUTED)
    for ax in axs[-4:]:
        ax.set_xlabel("iteration of the read (one column per read)", fontsize=8, color=MUTED)
    for ax in axs[::4]:
        ax.set_ylabel("half-width of the cube", fontsize=8, color=MUTED)
    fig.legend(handles=[Patch(fc=cols[0], ec=MUTED, lw=0.5, label="liftable, |q| = 0"), Patch(fc=cols[1], label="liftable, |q| = 1"),
                        Patch(fc=cols[2], label="liftable, another whole number"), Patch(fc=cols[3], label="pierced: no charge"),
                        Patch(fc=cols[4], label="gap at or under 0.2: no charge"), Line2D([], [], color=INK, lw=0.9, label="cubes above the line surround the loop")],
               loc="lower right", frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.03), ncol=2)
    fig.suptitle("M5.46: the read on each nested cube surface at every read", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "m5_46_degree.png", dpi=110)
    plt.close(fig)


def plot_dilation(dil, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, len(dil), figsize=(3.6 * len(dil), 3.4), squeeze=False, sharey=True)
    for ax, (b, v) in zip(axs[0], dil.items()):
        c = v["curve"]
        sep = [r["separation"] for r in c]
        ax.plot(sep, [r["U"] for r in c], "-", color=INK, lw=2.0)
        ax.plot(sep, [r["U2"] for r in c], "-", color=BLUE, lw=1.4)
        ax.plot(sep, [r["U4"] for r in c], "-", color=ORANGE, lw=1.4)
        ax.plot(sep, [r["UV"] for r in c], "-", color=AQUA, lw=1.4)
        for m in (10.0, 13.8):
            ax.axvline(m, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"{b}: half-box {v['half_box']:g}, spacing {v['spacing']:g}", fontsize=9, color=INK)
        ax.set_xlabel("separation of the two points (2 R s)", fontsize=8, color=MUTED)
        ax.set_ylim(0, None)
        _axes_style(ax)
    axs[0][0].set_ylabel("energy of the dilated base state", fontsize=8, color=MUTED)
    fig.legend(handles=[Line2D([], [], color=INK, lw=2, label="U"), Line2D([], [], color=BLUE, label="L₁ term"), Line2D([], [], color=ORANGE, label="commutator term"),
                        Line2D([], [], color=AQUA, label="potential term"), Line2D([], [], color=MUTED, ls=":", label="the pulls: 10 and 13.8")],
               loc="lower center", frameon=False, fontsize=9, ncol=5)
    fig.suptitle("M5.46: U along the dilation M′(x) = M(c + (x − c)/s), before any release", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out / "m5_46_dilation.png", dpi=110)
    plt.close(fig)


def plot_sections(recs, data, out, rows, fname, title):
    """Per chain: the base, the start and the end state on the plane y = 0 and on the plane of the loop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import m5_46_degree as D
    panels = []
    for n in rows:
        if n not in recs:
            continue
        r = recs[n]
        items = []
        if r["spec"]["kind"] == "added":
            items.append((f"{n} at read 0 (end of {r['start']['base']})", data / r["start"]["file"]))
            travel = [s for s in r.get("snapshots", []) if int(s.split("_s")[-1].split(".")[0]) not in SNAPS_STD]
            for k in (np.unique(np.linspace(0, len(travel) - 1, min(4, len(travel))).astype(int)) if travel else []):
                items.append((f"{n} at iteration {int(travel[k].split('_s')[-1].split('.')[0])}", data / travel[k]))
            items.append((f"{n} at the end ({r['iterations']})", data / r["end_file"]))
            panels.append([it for it in items if it[1].exists()])
            continue
        if r["start"]["kind"] != "seed":
            items.append((f"{r['start']['base']} (base)", data / f"m5_46_{r['start']['base']}_end.npz"))
            items.append((f"{n} at read 0" + ("" if r["start"]["separation"] is None else f" (pulled to {r['start']['separation']:g})"), data / r["start"]["file"]))
        snaps = [s for s in r.get("snapshots", [])]
        if snaps:
            mid = snaps[(len(snaps) - 1) // 2]
            items.append((f"{n} at iteration {int(mid.split('_s')[-1].split('.')[0])}", data / mid))
        items.append((f"{n} at the end ({r['iterations']})", data / r["end_file"]))
        panels.append([it for it in items if it[1].exists()])
    panels = [p for p in panels if p]
    if not panels:
        return
    ncols = max(len(p) for p in panels)
    fig, axs = plt.subplots(2 * len(panels), ncols, figsize=(3.3 * ncols, 3.5 * 2 * len(panels)), squeeze=False)
    im = None
    for i, items in enumerate(panels):
        for j in range(ncols):
            a1, a2 = axs[2 * i, j], axs[2 * i + 1, j]
            if j >= len(items):
                a1.set_visible(False)
                a2.set_visible(False)
                continue
            name, f = items[j]
            z = np.load(f, allow_pickle=False)
            m6, x = z["m6"].astype(float), z["x"]
            h, mid = x[1] - x[0], len(x) // 2
            n_, gap = D.director(m6)
            line, P, lab = D.line_read(m6, x)
            k = D.the_loop(line["sets"])
            zc = line["sets"][k]["center"][2] if k is not None else 0.0
            kz = int(np.clip(np.rint((zc - x[0]) / h), 0, len(x) - 1))
            st = max(1, int(round(len(x) / 17)))
            for ax, sl, (u, v), plane in ((a1, (slice(None), mid, slice(None)), (0, 2), 1), (a2, (slice(None), slice(None), kz), (0, 1), 2)):
                im = ax.pcolormesh(x, x, gap[sl].T, cmap="Blues_r", vmin=0, vmax=1, shading="nearest", rasterized=True)
                nn = n_[sl][::st, ::st]
                xs = x[::st]
                XX, YY = np.meshgrid(xs, xs, indexing="ij")
                ax.quiver(XX, YY, nn[..., u], nn[..., v], headwidth=0, headlength=0, headaxislength=0, pivot="mid", scale=28, width=0.004, color=INK, alpha=0.75)
                if len(P):
                    pos = 0.0 if plane == 1 else x[kz]
                    near = np.abs(P[:, plane] - pos) <= (0.75 * h if plane == 1 else 1e9)
                    ax.plot(P[near, u], P[near, v], ".", color=ORANGE, ms=3.5)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(MUTED)
            rad = "no set" if k is None else f"radius {line['sets'][k]['radius']:.2f}, {len(line['sets'])} set" + ("s" if len(line["sets"]) > 1 else "")
            a1.set_title(f"{name}\nplane y = 0 (x, z)\n{rad}", fontsize=8, color=INK)
            a2.set_title(f"plane z = {x[kz]:g} (x, y)\nevery pierced plaquette projected", fontsize=8, color=INK)
    H = 3.5 * 2 * len(panels)
    fig.subplots_adjust(top=1 - 1.15 / H, hspace=0.42, right=0.93)
    fig.colorbar(im, cax=fig.add_axes([0.945, 0.35, 0.010, 0.30]), label="gap of the two larger eigenvalues")
    fig.suptitle(title + "\nsticks: the long axis in the plane; orange: pierced plaquettes", fontsize=10, color=INK, y=1 - 0.2 / H)
    fig.savefig(out / fname, dpi=72, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    ap.add_argument("--plots", type=Path, default=HERE.parent / "plots")
    ap.add_argument("--tables", type=Path, default=HERE.parent / "checkpoints" / "m5_46" / "tables.md")
    ap.add_argument("--no-sections", action="store_true")
    a = ap.parse_args()
    recs, labs, ders = {}, {}, {}
    for n in ORDER + ADDED:
        f = a.data / f"m5_46_chain_{n}.json"
        if f.exists():
            recs[n] = json.loads(f.read_text())
            labs[n], ders[n] = label_chain(recs[n])
    dil = json.loads((a.data / "m5_46_dilation.json").read_text()) if (a.data / "m5_46_dilation.json").exists() else None
    ctl = json.loads((a.data / "m5_46_controls.json").read_text()) if (a.data / "m5_46_controls.json").exists() else None
    text, r4 = tables(recs, labs, ders, dil, ctl)
    print(text)
    a.tables.parent.mkdir(parents=True, exist_ok=True)
    a.tables.write_text(text)
    chains = {n: dict(model=recs[n]["model"], start={k: v for k, v in recs[n]["start"].items()}, descent=recs[n]["descent"], step=recs[n]["step"],
                      pilot=recs[n]["pilot"], rises_of_U=recs[n]["rises_of_U"], ended=recs[n]["ended"], iterations=recs[n]["iterations"],
                      evaluations=recs[n]["evaluations"], seconds=recs[n]["seconds"], gate_first_met_at=recs[n]["gate_first_met_at"],
                      gate_met_at_the_end=recs[n]["gate_met_at_the_end"], gmax_over_h3_at_the_end=recs[n]["gmax_over_h3_at_the_end"],
                      U_at_the_end=recs[n]["U_at_the_end"], events=recs[n]["events"], half_widths=recs[n]["half_widths"],
                      outside_the_plan=bool(n in ADDED), labels=labs[n]) for n in recs}
    summary = dict(task="M5.46", readings="m5_46_task_details.md section 4, applied as in the docstring of m5_46_reads.py", chains=chains, R4=r4,
                   controls_passed=(ctl["passed"] if ctl else None), reads_files={n: f"m5_46_chain_{n}.json" for n in recs})
    (a.data / "m5_46_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    a.plots.mkdir(parents=True, exist_ok=True)
    plot_series(recs, ders, labs, a.plots)
    if dil:
        plot_dilation(dil, a.plots)
    if not a.no_sections:
        plot_sections(recs, a.data, a.plots, ("P10a", "P10b"), "m5_46_sections_p10.png", "M5.46: the base, the pulled state and the release, separation 10 on half-box 8")
        plot_sections(recs, a.data, a.plots, ("P14c", "P14d"), "m5_46_sections_p14.png", "M5.46: the base, the pulled state and the release, separation 13.8 on half-box 10")
        plot_sections(recs, a.data, a.plots, ("F0a", "F0b"), "m5_46_sections_f0.png", "M5.46: the base released with free faces, no pull")
        plot_sections(recs, a.data, a.plots, ("H10a", "G10a", "P14a"), "m5_46_sections_other.png", "M5.46: the shell-held pull, the steepest descent and the tight box")
        plot_sections(recs, a.data, a.plots, ("X0a", "X14c"), "m5_46_sections_added_h05.png", "M5.46: added chains at spacing 0.5, a gate-ended free state continued")
        plot_sections(recs, a.data, a.plots, ("X0b", "X14d"), "m5_46_sections_added_h025.png", "M5.46: added chains at spacing 0.25, a gate-ended free state continued")
    print("written:", a.data / "m5_46_summary.json", "and the plots in", a.plots)


if __name__ == "__main__":
    main()
