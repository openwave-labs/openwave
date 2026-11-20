"""M5.48: the tables of the findings, printed in markdown from the tracked summaries.

Reads data/m5_48_summary.json and data/m5_48_controls.json (written by m5_48_reads.py) and nothing else: no state, no
chain record. Every table of m5_48_task_details.md, sections 6 to 11, is printed here. Tables 18 and 19 (the return added
after the audit of the findings, and three readings of P1 side by side) were added with the fold of that audit.

Usage:  python m5_48_report.py [--out-dir DIR]
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
YN = {True: "yes", False: "no", None: "n/a"}
ORDER = ["h0375_L0", "h0250_L0", "h0375_L0_from_h0500", "h0375_L0_from_h0250", "h0375_L0_from_h01875", "h0250_L0_from_h0375", "h0250_L0_from_h01875",
         "h0375_L0.03", "h0250_L0.03", "h0250_L0.03_from_no_L1"]
START = {"h0375_L0": "seed", "h0250_L0": "seed", "h0375_L0_from_h0500": "carried from 0.5", "h0375_L0_from_h0250": "carried from 0.25", "h0375_L0_from_h01875": "carried from 0.1875",
         "h0250_L0_from_h0375": "carried from 0.375", "h0250_L0_from_h01875": "carried from 0.1875", "h0375_L0.03": "seed", "h0250_L0.03": "seed", "h0250_L0.03_from_no_L1": "the state without L₁"}


def f(v, fm="%.4f"):
    return "n/a" if v is None else fm % v


def short(how):
    """How a chain ended, in a few words (the full list of events is in the JSON)."""
    if how is None:
        return "n/a"
    if "STALLED above" in how:
        return "STALLED above its gate"
    if "cut by the wall-time cap" in how:
        return "cut by the wall-time cap"
    if "the cap of iterations" in how:
        return "the cap of iterations"
    if "stalled line search" in how:
        return "a stalled line search, the gate met" if "function tolerance" not in how else "function tolerance, then a stalled line search"
    return "function tolerance, repeated"


def radius_alone(s):
    """A state that fails S5 with as many loops as its reference fails it by the radius alone (the plan, section 5)."""
    return bool(s.get("S5") is False and s.get("loops_on_the_reference") and s["loops_on_the_reference"]["loops"] == s["loops_at_J"]["loops"])


def controls(K):
    def num(k, v):
        if k == "K1":
            return "largest relative difference %.1e over %d states" % (v["largest_relative"], len(v["rows"]))
        if k in ("K2", "K3", "K4", "K5"):
            keys = [q for q in ("E", "U", "I") if q in v]
            return "first end at iteration %d (reference %d); %s; largest relative difference %.1e; the chain ended at %d" % (
                v["iteration"], v["reference_iteration"], ", ".join("%s = %.12f" % (q, v[q]) for q in keys), max(v["relative"].values()), v["end_iteration"])
        if k == "K6":
            c, s, w = v["clock_term_alone_on_the_J0_states"], v["whole_E_J_on_the_seed"], v["first_wording"]["result"]
            return "clock term alone %.1e to %.1e; whole E_J on the seeds %.1e to %.1e. First wording: NOT MET, %.1e and %.1e at J_e/1280 (%.1e and %.1e at J_e/80)" % (
                min(c.values()), max(c.values()), min(s.values()), max(s.values()), w["h0375_L0|J_e/1280"], w["h0375_L0.03|J_e/1280"], w["h0375_L0|J_e/80"], w["h0375_L0.03|J_e/80"])
        if k == "K7":
            return "S3 shares %.4f (itself), %.4f (M5.42 at J_e/40), %.4f (another start, fails); S2 %s; S4 %s; S5 itself %s, against the state with L₁ %s (radius %.3f against %.3f)" % (
                v["S3_itself"]["share_on_W0"], v["S3_M5_42_at_J_e_over_40"]["share_on_W0"], v["S3_another_start"]["share_on_W0"], "/".join(YN[q] for q in v["S2"].values()),
                "/".join(YN[q] for q in v["S4"].values()), YN[v["S5"]["itself"]["S5"]], YN[v["S5"]["against_the_state_with_L1"]["S5"]], v["S5"]["radius_without_L1"], v["S5"]["radius_with_L1"])
        if k == "K8":
            g = v["got"]
            return "growth %.4f and %.4f; spread %.4f, %.4f, %.4f; on M5.45: %s, %s; %d made-up cases" % (
                g["growth_from_the_seed"], g["growth_lowest_U"], g["spread_0375"], g["spread_0500"], g["spread_0250"], v["word_on_M5_45"]["word"], v["word_on_M5_45"]["second_reading"], len(v["made_up"]))
        if k == "K9":
            return "; ".join("(%g, %g): %s" % (r["I"], r["U"], r["word"]) if r["case"].startswith("made") else "the cut state (I·h %.3f): %s" % (r["I_times_h"], r["word"]) for r in v["rows"])
        if k == "K10":
            return "counter %d, %d trajectory points, first new point at %s; read-back relative %.1e; SHA-256 unchanged at the end of the chain: %s; a name outside the prefix refused: %s" % (
                v["counter_at_the_start"], v["trajectory_points_carried"], v["first_new_trajectory_iteration"], max(v["read_back"]["U_relative"], v["read_back"]["I_relative"]),
                YN[v["unchanged"] and v["chain_finished"]], YN[v["refuses_a_name_outside_m5_48"]])
        if k == "K11":
            return "%d made-up tables, every word and every branch once" % len(v["cases"])
        if k == "K12":
            c = v["cases"]
            return "six rule cases; gates %.3g and %.3g; the chain under a gate of 1e-12: %d stall recorded, CONVERGED %s" % (
                c["gate_of_a_small_clock_force"], c["gate_of_a_large_clock_force"], c["chain"]["stalls_recorded"], YN[c["chain"]["converged"]])
        if k == "K13":
            return "largest relative difference %.1e over %d carried starts" % (v["largest_relative"], len(v["rows"]))
        return ""

    print("\n**Table 1. The controls (every one read from `m5_48_controls.json`).**\n")
    print("| Control | Result | The numbers |")
    print("| --- | --- | --- |")
    for k, v in K.items():
        print("| %s | %s | %s |" % (k, {True: "PASS", False: "NOT MET", None: "not read"}[v["met"]], num(k, v)))


def chains(C):
    def cls(c):
        if c["task"] == "P1":
            return "P1, " + {"ref": "the reference", "J": "at J", "back": "the return"}[c["role"]] + (", 33³" if c["N"] == 33 else ", 49³")
        if c["task"] == "P3":
            return "P3, %d³" % c["N"]
        return c["task"]

    groups = {}
    for c in C:
        groups.setdefault(cls(c), []).append(c)
    print("\n**Table 2. The chains by class (all grids of half-box 6; iterations and wall time over the chains that ran).**\n")
    print("| Class | In the plan | Ran | CONVERGED | At the cap of iterations, not CONVERGED | Cut by the wall-time cap | STALLED above the gate | A line search with no lower point, above the gate | Not run | Iterations, from to | Wall time per chain, minutes, from to | Process-hours |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    tot = 0.0
    for g, L in groups.items():
        ran = [c for c in L if c["ran"]]
        its = [c["iterations_new"] for c in ran]
        sec = [c["seconds"] for c in ran]
        tot += sum(sec) / 3600
        other = sum(not c["converged"] and not c["stalled_above_the_gate"] and not c["cut_by_wall"] and not c["ended_at_cap"] for c in ran)
        print("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %s | %s | %.2f |" % (
            g, len(L), len(ran), sum(c["converged"] for c in ran), sum(c["ended_at_cap"] and not c["converged"] for c in ran), sum(c["cut_by_wall"] for c in ran),
            sum(c["stalled_above_the_gate"] for c in ran), other, len(L) - len(ran), "n/a" if not its else "%d to %d" % (min(its), max(its)),
            "n/a" if not sec else "%.1f to %.1f" % (min(sec) / 60, max(sec) / 60), sum(sec) / 3600))
    print("\nProcess-hours in all: %.1f." % tot)
    print("\n**Table 3. The chains that ran and are not CONVERGED.**\n")
    print("| Chain | Grid | Iterations | Wall time, minutes | How it ended | Gate | Largest gradient entry over h³ | I |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in C:
        if c["ran"] and not c["converged"]:
            how = short(c["how_it_ended"]) + (" (%d stalls recorded)" % c["stalls"] if c["stalls"] else "") + (", no iteration made" if c["iterations_new"] == 0 else "")
            how = how.replace("a stalled line search, the gate met", "a line search with no lower point")
            print("| `%s` | %d³ | %d | %.1f | %s | %.2e | %.2e | %.3f |" % (c["chain"], c["N"], c["iterations_new"], c["seconds"] / 60, how, c["gate"], c["gmax_over_h3"], c["I"]))
    print("\n**Table 4. The chains of the plan that were not run, with the reason the pool recorded.**\n")
    print("| Chain | Reason |")
    print("| --- | --- |")
    for c in C:
        if not c["ran"]:
            why = c["why"].split(" (")[0].replace("UNDECIDED: ", "undecided: ").replace("its source ", "its chain at J, ").replace(" did not run to a record", ", was not run")
            print("| `%s` | %s |" % (c["chain"], why))


def p1(P1):
    by = {(s["state"], s["k"]): s for s in P1["states"]}
    ks = sorted({s["k"] for s in P1["states"]})
    print("\n**Table 5. P1, the references: each J = 0 state of M5.45 continued at J = 0 under the gate of each candidate (δ = 0.05, half-box 6).**\n")
    print("| L₁ | Spacing | State (start in M5.45) | " + " | ".join("J_e/%d: gate; iterations; CONVERGED" % k for k in ks) + " | Largest change of U, of I (relative) over the six |")
    print("| --- | --- | --- | " + " | ".join("---" for _ in ks) + " | --- |")
    for st in ORDER:
        rows = [by[(st, k)] for k in ks]
        s0 = rows[0]
        cells = ["%.2e; %d; %s" % (r["gate"], r["reference"]["iterations"], YN[r["reference_converged"]]) for r in rows]
        print("| %g | %g | `%s` (%s) | %s | %.1e, %.1e |" % (s0["L1"], s0["spacing"], st, START[st], " | ".join(cells), max(abs(r["reference"]["U_minus_M5_45"]) for r in rows),
                                                          max(abs(r["reference"]["I_relative_to_M5_45"]) for r in rows)))
    print("\n**Table 6. P1, every state at every candidate: the chain at J, the clauses and the status (δ = 0.05, half-box 6; ΔE, ω, ρ and a against the reference of the same candidate).**\n")
    print("| L₁ | Spacing | State | J_e/k | Iterations, end | ΔE | ω | ρ | a = I/I₀′ | S2: U − U₀′ | S3: share on W₀ | S5: loops, radius on the reference; at J; change | S4: left over in I, in U; return iterations, end | Label | Status |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for st in ORDER:
        for k in ks:
            s = by[(st, k)]
            if s.get("dE") is None:
                print("| %g | %g | `%s` | %d | not run | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | %s: %s |" % (s["L1"], s["spacing"], st, k, s["status"], "the reference is not CONVERGED under its gate"))
                continue
            l0, lJ, b = s["loops_on_the_reference"], s["loops_at_J"], s.get("back")
            why = "" if not s["why"] else s["why"].split(" (")[0]
            if radius_alone(s):
                why = why.replace("S5", "S5 (radius alone)")
            status = why if why.startswith("fails") else s["status"] + ("" if not why else ": " + why)
            print("| %g | %g | `%s` | %d | %d, %s | %.6f | %.6f | %.4f | %.4f | %+.2e (%s) | %.4f (%s) | %d, %s; %d, %s; %s (%s) | %s | %s | %s |" % (
                s["L1"], s["spacing"], st, k, s["iterations"], short(s["how_it_ended"]), s["dE"], s["omega"], s["rho"], s["I_over_I0"], s["U"] - s["U0"], YN[s["S2"]], s["share_on_W0"], YN[s["S3"]],
                l0["loops"], f(l0["radius"], "%.3f"), lJ["loops"], f(lJ["radius"], "%.3f"), f(s["radius_change"] and 100 * s["radius_change"], "%.1f %%") if s["radius_change"] is not None else "n/a", YN[s["S5"]],
                "not run" if not b else "%.3f, %.3f; %d, %s (%s)" % (b["I_left_over"], b["U_left_over"], b["iterations"], short(b["how_it_ended"]), YN[s["S4"]] if b["converged"] else "return not CONVERGED"),
                s["label"], status))
    cut = [s for st in ORDER for k in ks for s in [by[(st, k)]] if s.get("at_the_cut")]
    print("\n**Table 7. P1, the chains at J that are not CONVERGED, read at their last state against S3 and S5 (the amendment after the fold).**\n")
    print("| L₁ | Spacing | State | J_e/k | How it ended | U | I | Largest gradient entry over h³ (gate) | Share on W₀ (S3) | Loops, radius: reference; last state (S5) | Read as |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in cut:
        c = s["at_the_cut"]
        print("| %g | %g | `%s` | %d | %s | %.6f | %.3f | %.1e (%.1e) | %.4f (%s) | %d, %s; %d, %s (%s) | %s |" % (
            s["L1"], s["spacing"], s["state"], s["k"], short(c["how_it_ended"]), c["U"], c["I"], c["gmax_over_h3"], s["gate"], c["share_on_W0"], YN[c["S3"]], c["loops_on_the_reference"]["loops"],
            f(c["loops_on_the_reference"]["radius"], "%.3f"), c["loops"]["loops"], f(c["loops"]["radius"], "%.3f"), YN[c["S5"]], "failed at the cut, not converged" if s["failed_at_the_cut"] else "UNDECIDED"))
    print("\n**Table 8. P1, the candidates: the four states from the seed and the falls on their two pairs (spacing 0.375 against 0.25).**\n")
    print("| J_e/k | J | Status | `h0375_L0` | `h0250_L0` | `h0375_L0.03` | `h0250_L0.03` | Without L₁: f_ΔE | f_ω | f₀ | a at 0.375, at 0.25 | Band left to f_ΔE | L₁ = 0.03: f_ΔE | f_ω | f₀ | a at 0.375, at 0.25 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in P1["candidates"]:
        a, b = r["without L1"], r["L1 = 0.03"]
        cell = lambda st: (lambda v: v["status"] + ("" if not v["why"] else " (" + v["why"].split(" (")[0].replace("fails ", "").replace("S5", "S5, radius alone" if radius_alone(by[(st, r["k"])]) else "S5") + ")"))(r["states"][st])  # noqa: E731
        print("| %d | %.4f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["k"], r["J"], r["status"].upper(), *(cell(st) for st in ("h0375_L0", "h0250_L0", "h0375_L0.03", "h0250_L0.03")), *(f(a[q]) if a else "n/a" for q in ("f_dE", "f_omega", "f_0")),
            "n/a" if not a else "%.4f, %.4f" % (a["rise_of_I_coarse"], a["rise_of_I_fine"]), "n/a" if not a else "%.4f to %.4f" % tuple(a["band_of_f_dE"]),
            *(f(b[q], "%.5f") if b else "n/a" for q in ("f_dE", "f_omega", "f_0")), "n/a" if not b else "%.4f, %.4f" % (b["rise_of_I_coarse"], b["rise_of_I_fine"])))
    r = P1["reading"]
    print("\nThe reading: J\\* = %s (J_e/%s); the word **%s**%s%s%s; pairs at or above 0.20 in both falls: %s of %d counted, %d left out; by arm: %s." % (
        f(r["J_star"]), r["k_star"], r["word"], (" (" + r["reason"] + ")") if r["reason"] else "", (", " + r["qualifier"]) if r["qualifier"] else "",
        (" (" + "; ".join(r["clauses_failed"]) + ")") if r["clauses_failed"] else "", r["pairs_at_or_above_in_both"], len(r["pairs"]), len(r["pairs_left_out"]),
        "; ".join("%s: %s" % (k, "none" if v is None else "J_e/%d (f_ΔE %.4f, f_ω %.4f)" % (v["k"], v["f_dE"], v["f_omega"])) for k, v in r["by_arm"].items())))
    print("\n**Table 9. P1, the twelve pairs of states without L₁ at J\\* = J_e/%s (four states at spacing 0.375 times three at 0.25).**\n" % r["k_star"])
    print("| State at 0.375 | State at 0.25 | f_ΔE | f_ω | f₀ | At or above 0.20 in both | Both states survive |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for p in r["pairs"]:
        print("| `%s` | `%s` | %.4f | %.4f | %.4f | %s | %s |" % (p["coarse"], p["fine"], p["f_dE"], p["f_omega"], p["f_0"], YN[p["f_dE"] >= 0.2 and p["f_omega"] >= 0.2], YN[p["both_survive"]]))
    if r.get("pair_with_the_tilted_state"):
        t = r["pair_with_the_tilted_state"]
        print("\nWith L₁ = 0.03 at J\\*, the state from the seed at 0.375 against the tilted state at 0.25: f_ΔE = %.5f, f_ω = %.5f (f₀ = %.5f)." % (t["f_dE"], t["f_omega"], t["f_0"]))


def added(P1):
    a = a_ret = P1["added_return"]
    print("\n**Table 18. P1, the one return added after the audit of the findings, with the word in view and carrying no word: the state from the seed at spacing 0.25 without L₁ at J_e/40, "
          "descended at J = 0 under the gate and the rule of the other returns (wall-time cap 40 minutes).**\n")
    if not a["finished"]:
        print("Not ended at this read; progress:", a.get("progress"))
    else:
        print("| Iterations, wall time | How it ended | Words | Gradient (gate) | U: reference; at J; return | I: reference; at J; return | Left over in I, in U (S4 allows 0.1) | U of the return minus U of the reference | S4, as frozen (absolute values) | S4 with the U part one-sided | The state at J_e/40 with S5 read as a loop count alone |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        print("| %d, %.1f minutes | %s | %s | %.2e (%.0e) | %.6f; %.6f; %.6f | %.3f; %.3f; %.3f | %.4f, %.4f | %+.4e | %s | %s | %s |" % (
            a["iterations"], a["seconds"] / 60, short(a["how_it_ended"]), a["words"], a["gmax_over_h3"], a["gate"], a["U_of_the_reference"], a["U_at_J"], a["U"], a["I_of_the_reference"], a["I_at_J"], a["I"],
            a["I_left_over"], a["U_left_over"], a["U_minus_U_of_the_reference"], YN[a["S4"]] if a["converged"] else "n/a (the return is not CONVERGED)",
            YN[a["S4_with_the_U_part_one_sided"]] if a["converged"] else "n/a", a["with_S5_as_a_loop_count_alone"]))

    def jstar(r):
        k = r["k_star"]
        if k is None:
            return "none"
        return "J_e/%d" % k if not isinstance(k, dict) else "without L₁ J_e/%d; with L₁ J_e/%d" % (k["without_L1"], k["with_L1"])

    def word(r):
        w = r["word"] + ((", " + r["qualifier"]) if (r.get("qualifier") and r["word"] != "no label") else "")
        if r.get("reason"):
            w += " (" + r["reason"] + ")"
        elif r.get("clauses_failed"):
            w += " (" + "; ".join(r["clauses_failed"]) + ")"
        if r.get("f_with_L1"):
            w += "; changes with L₁ %.4f and %.4f, falls without L₁ %.4f and %.4f" % (*r["f_with_L1"], *r["f_without_L1"])
        return w.replace("L1", "L₁")

    print("\n**Table 19. P1, three readings side by side, each also with S5 read as a loop count alone (descriptors, added after the audit of the findings: the word of record is the first, as the frozen rules give it).**\n")
    print("| Reading | J\\* | Word | With S5 read as a loop count alone: J\\* | Word |")
    print("| --- | --- | --- | --- | --- |")
    for r in P1["readings"]:
        a, b = r["as_frozen"], r["with_S5_as_a_loop_count_alone"]
        extra = ""
        if r.get("by_arm"):
            extra = "; by arm the same: %s, %s" % (jstar(r["by_arm"]), r["by_arm"]["word"])
            extra_b = "; by arm: %s, %s" % (jstar(r["by_arm_with_S5_as_a_loop_count_alone"]), r["by_arm_with_S5_as_a_loop_count_alone"]["word"] + (
                " (" + r["by_arm_with_S5_as_a_loop_count_alone"]["reason"] + ")" if r["by_arm_with_S5_as_a_loop_count_alone"].get("reason") else ""))
        print("| %s | %s | %s%s | %s | %s%s |" % (r["reading"].replace("L1", "L₁"), jstar(a), word(a), extra, jstar(b), word(b), extra_b if r.get("by_arm") else ""))
    key = "with_S5_as_a_loop_count_alone_and_the_U_part_of_S4_one_sided"
    if a_ret["finished"] and a_ret["converged"] and a_ret["S4"] != a_ret["S4_with_the_U_part_one_sided"]:
        print("\nThe last two columns rest on S4 as frozen, with absolute values. With S5 read as a loop count alone and the U part of S4 one-sided, the form in which the plan audit first wrote it, "
              "the added return passes and the three readings give: " + "; ".join("%s: %s, %s" % (r["reading"].replace("L1", "L₁"), jstar(r[key]), word(r[key])) for r in P1["readings"]) + ".")


def p2(P2):
    print("\n**Table 10. P2, the chain from the seed at spacing 0.1875 without L₁ (δ = 0.05, 65³): the stops of M5.45, carried, and those of the continuation.**\n")
    print("| Iteration | Stop | U | I | I·h | Largest gradient entry over h³ |")
    print("| --- | --- | --- | --- | --- | --- |")
    for s in P2["stops"]:
        kind = ("M5.45, " if s.get("carried_from") else "M5.48, ") + {"target": "target", "carried_end": "its cut", "end": "the end"}[s["kind"]]
        print("| %d | %s | %.9f | %.4f | %.3f | %.2e |" % (s["iteration"], kind, s["U"], s["I"], s["I"] * 0.1875, s["gmax_over_h3"]))
    print("\n**Table 11. P2, the end state against the two forms of the author's window.**\n")
    print("| Iterations (new) | How it ended | Words | Gradient (gate) | Drift of I over the last tenth | I | I·h | Inside I·h from 1000 to 1300 | Inside I from 5300 to 6900 | I·h from the nearer edge | U | U inside 14.2493 to 15.1307 | Word |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    print("| %d (%d) | %s | %s | %.2e (%.0e) | %+.1e | %.4f | %.3f | %s | %s | %+.2f %% | %.9f | %s | %s |" % (
        P2["iterations"], P2["iterations_new"], P2["how_it_ended"], P2["words"], P2["gmax_over_h3"], P2["gate"], P2["drift_of_I_over_the_last_tenth"], P2["I"], P2["I_times_h"],
        YN[P2["inside_the_form_in_I_times_h"]], YN[P2["inside_the_form_in_I"]], P2["I_times_h_from_the_nearer_edge_percent"], P2["U"], YN[P2["U_in_window"]], P2["word_printed"]))


def p3(P3):
    print("\n**Table 12. P3, every state found at δ = 0.3 without L₁ on half-box 6, lowest U first on each grid.**\n")
    print("| Spacing | Start | U | I | I inside r < 2 (share) | W: share of the nodes | q on W | q on the rest | Iterations | How it ended | Words | Largest gradient entry over h³ |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in sorted(P3["states"], key=lambda s: (-s["spacing"], s["U"])):
        print("| %g | %s | %.6f | %.3f | %.1f (%.3f) | %.4f | %s | %s | %d | %s | %s | %.1e |" % (
            s["spacing"], s["start"], s["U"], s["I"], s["shells"]["inside_r2"], s["shells"]["inside_r2"] / s["I"], s["W_share_of_nodes"], f(s["q_on_W"], "%.5f"), f(s["q_on_the_rest"]),
            s["iterations"], short(s["how_it_ended"]), s["words"], s["gmax_over_h3"]))
    print("\n**Table 13. P3, the ratios of the inertia between spacings, for the two families of M5.45 (δ = 0.3; beside them the ratios of M5.45 at δ = 0.05).**\n")
    print("| States | From spacing (I) | To spacing (I) | Ratio of I | Read as | At δ = 0.05 (M5.45) |")
    print("| --- | --- | --- | --- | --- | --- |")
    m545 = {("from the seed", 0.5): "1.763", ("from the seed", 0.375): "1.609", ("from the seed", 0.25): "1.571, no value", ("the lowest U found on each grid", 0.5): "1.717",
            ("the lowest U found on each grid", 0.375): "1.446", ("the lowest U found on each grid", 0.25): "1.346, no value"}
    for e in P3["ratios"]["from_the_seed"] + P3["ratios"]["lowest_U_found"]:
        print("| %s | %g (%.1f) | %g (%.1f) | %.3f | %s | %s |" % (e["states"], e["coarse"], e["I_coarse"], e["fine"], e["I_fine"], e["ratio_of_I"], "a value" if e["a_value"] else "no value (a state not CONVERGED)",
                                                              m545[(e["states"], e["coarse"])]))
    print("\n**Table 14. P3, the spread of the inertia over the CONVERGED states found on each grid (δ = 0.3; beside it the spread of M5.45 at δ = 0.05).**\n")
    print("| Spacing | States found | CONVERGED | I from | I to | Spread S | At δ = 0.05 (M5.45) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for h, v in sorted(P3["ratios"]["spread"].items(), key=lambda q: -float(q[0])):
        print("| %g | %d | %d | %s | %s | %s | %s |" % (float(h), v["states"], v["converged"], f(v["I_min"], "%.1f"), f(v["I_max"], "%.1f"), f(v["S"], "%.3f"), {"0.5": "1.735", "0.375": "1.376", "0.25": "1.167"}.get(str(float(h)), "n/a")))
    r = P3["reading"]
    print("\nThe reading: G = %s (%s), S = %s on %d CONVERGED of %d states at spacing 0.375: the word **%s**; the second reading: %s%s." % (
        f(r["G"], "%.3f"), "a value" if r["G_is_a_value"] else "no value", f(r["S"], "%.3f"), r["converged_at_coarse"], r["states_at_coarse"], r["word"], r["second_reading"],
        "; the inertia fell as the spacing fell" if r["the_inertia_fell_as_the_spacing_fell"] else ""))
    if P3["missing"]:
        print("\n(no record at this read: %s)" % ", ".join("`%s`" % q for q in P3["missing"]))


def descriptors(S):
    P1, P2, P3 = S["P1"], S["P2"], S["P3"]
    by = {(s["state"], s["k"]): s for s in P1["states"]}
    ks = sorted({s["k"] for s in P1["states"]})
    print("\n**Table 15. How many nodes carry the inertia, P1: the ten states, on the reference and at J (δ = 0.05, half-box 6; in each cell the largest share of one node; N_I; V_I).**\n")
    print("| L₁ | Spacing | State | On the reference | " + " | ".join("At J_e/%d" % k for k in ks) + " |")
    print("| --- | --- | --- | --- | " + " | ".join("---" for _ in ks) + " |")
    pt = lambda q: "n/a" if not q else "%.5f; %.0f; %.2f" % (q["largest_share"], q["N_I"], q["V_I"])  # noqa: E731
    for st in ORDER:
        rows = [by[(st, k)] for k in ks]
        ref = next((r["participation_of_the_reference"] for r in rows if r.get("participation_of_the_reference")), None)
        print("| %g | %g | `%s` | %s | %s |" % (rows[0]["L1"], rows[0]["spacing"], st, pt(ref), " | ".join(pt(r.get("participation")) for r in rows)))
    print("\n**Table 16. The ratio of V_I from spacing 0.375 to 0.25 on the pairs from the seed (P1), at J and on the references.**\n")
    print("| Arm | " + " | ".join("J_e/%d: at J; on the references" % k for k in ks) + " |")
    print("| --- | " + " | ".join("---" for _ in ks) + " |")
    for arm in ("without L1", "L1 = 0.03"):
        v = {q["k"]: q for q in P1["participation_volume_ratios"] if q["arm"] == arm}
        print("| %s | %s |" % (arm.replace("L1", "L₁"), " | ".join("n/a" if k not in v else "%.3g; %.3f" % (v[k]["V_I_ratio_at_J"], v[k]["V_I_ratio_on_the_references"]) for k in ks)))
    print("\n**Table 17. How many nodes carry the inertia, P2 (δ = 0.05) and P3 (δ = 0.3), without L₁ on half-box 6.**\n")
    print("| Task | δ | Spacing | State | I | Largest share | N_I | V_I |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    q = P2["participation"]
    print("| P2 | 0.05 | 0.1875 | from the seed, continued | %.3f | %.5f | %.0f | %.2f |" % (P2["I"], q["largest_share"], q["N_I"], q["V_I"]))
    for s in sorted(P3["states"], key=lambda s: (-s["spacing"], s["U"])):
        q = s["participation"]
        print("| P3 | 0.3 | %g | %s%s | %.3f | %.5f | %.0f | %.2f |" % (s["spacing"], s["start"], "" if s["converged"] else " (not CONVERGED)", s["I"], q["largest_share"], q["N_I"], q["V_I"]))
    print("\nThe ratio of V_I from spacing 0.375 to 0.25 on the states from the seed of P3: %s." % f(P3["V_I_ratio_from_the_seed_0375_to_0250"], "%.3f"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    a = ap.parse_args()
    S = json.loads((a.out_dir / "m5_48_summary.json").read_text())
    K = json.loads((a.out_dir / "m5_48_controls.json").read_text())["controls"]
    controls(K)
    chains(S["chains"])
    p1(S["P1"])
    added(S["P1"])
    p2(S["P2"])
    p3(S["P3"])
    descriptors(S)


if __name__ == "__main__":
    main()
