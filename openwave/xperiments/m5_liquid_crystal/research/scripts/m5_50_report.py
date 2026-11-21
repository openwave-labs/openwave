"""M5.50: the tables and the plots of the findings, from the tracked summary data/m5_50_summary.json.

    --tables    the markdown tables of the findings, written to <work-dir>/tables.md
    --plots     plots/m5_50_ladders.png        the rise of U, the force of the hold and the error of the step identity
                                               against the reader's separation, four ladders
                plots/m5_50_sections_<g>.png   the potential density u in the plane of the loop and in a plane through
                                               its axis: the base, the top counted rung, each end state of the family
                                               (needs the local end states data/m5_50_*_end.npz)
                plots/m5_50_exits.png          the class of every nested surface at every read of the exit chains

Usage:  python m5_50_report.py --tables --plots
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
GN = {"10a": "half-box 10, spacing 0.5 (41³)", "10b": "half-box 10, spacing 0.25 (81³)", "14a": "half-box 14, spacing 0.5 (57³)", "14b": "half-box 14, spacing 0.25 (113³)"}
COL = {"10a": "#1f77b4", "10b": "#17becf", "14a": "#d62728", "14b": "#ff7f0e"}


def fm(v, n=4):
    return "n/a" if v is None else f"{v:.{n}f}"


def rle(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}×{j - i}")
        i = j
    return ", ".join(out)


def tables(S, a):
    T = []
    C = S["controls"]
    T.append("**Table 1. The controls.**\n")
    T.append("| Control | Word | The numbers |")
    T.append("| --- | --- | --- |")

    def crow(k, text):
        c = C.get(k)
        T.append(f"| {k} | {'not run' if c is None else 'met' if c['passed'] else 'NOT MET'} | {text} |")

    c = C.get("K1", {}).get("numbers", [])
    crow("K1", "; ".join(f"{r['state']}: U {r['U']:.12f}, relative gap {r['relative_gap']:.1e}, {r['sets']} loop of radius {r['radius']:.4f}" for r in c))
    c = C.get("K2", {}).get("numbers", {})
    crow("K2", "; ".join(f"B{g}: {r.get('word')}, {r.get('sets')} loop, center within {fm(r.get('center_largest_coordinate'), 4)}, radius {fm(r.get('radius'))} against {fm(r.get('radius_of_half_box_10'))} "
                         f"(gap {fm(r.get('radius_gap'))}, bound 0.03); D {fm(r.get('D'))} against {fm(r.get('D_of_half_box_10'))} (half the gap {fm(r.get('half_gap_of_D'))})" for g, r in c.items()))
    c = C.get("K3", {}).get("numbers", {})
    crow("K3", "the seven controls of M5.46 on the four base states: " + ", ".join(f"{n} {'met' if p else 'NOT MET'}" for n, p in c.get("controls", [])) if c.get("controls") else json.dumps(c))
    c = C.get("K4", {}).get("numbers", [])
    crow("K4", "; ".join(f"{r['state']}: D {r['D']:.4f}, reader {r['reader_separation']:.4f} (D above it by {r['D_minus_reader']:.3f}); shifts of two cells change D by "
                         + ", ".join(f"{s['change']:.1e}" for s in r["shifts"]) + " (the shift alone " + ", ".join(f"{s['change_from_the_shift_alone']:.0e}" for s in r["shifts"])
                         + "); dilations 1.5 and 2: ratio " + ", ".join(f"{d['ratio'] / d['s']:.4f}" for d in r["dilations"]) + " of the dilation, D above the reader by "
                         + ", ".join(f"{d['D_minus_reader']:.3f}" for d in r["dilations"]) for r in c))
    c = C.get("K5", {}).get("numbers", [])
    crow("K5", "worst gap over the largest entry, 200 entries each: " + ", ".join(f"{r['term']} {r['worst_gap_over_largest']:.1e}" for r in c) + " (bound 1e-6)")
    c = C.get("K6", {}).get("numbers", {})
    crow("K6", "; ".join(f"{k[3:]}: U {r['U_free_rung']:.10f} against {r['U_reference']:.10f} (relative gap {r['relative_gap']:.1e}), hold energy {r['hold_energy']:.1e}, radius "
                         f"{r['radius_free_rung']:.4f} and {r['radius_reference']:.4f}, iterations {r['iterations'][0]} and {r['iterations'][1]} ({r['ended'][0]})" for k, r in c.items()))
    c = C.get("K7", {}).get("numbers") or {}
    crow("K7", "e of the three steps: " + ", ".join(f"{e:.4f}" for e in c.get("e", [])) + " (bound 0.01); f at the four rungs: " + ", ".join(f"{f:.4f}" for f in c.get("f", []))
         + "; U: " + ", ".join(f"{u:.6f}" for u in c.get("U", [])))
    c = C.get("K8", {}).get("numbers", {})
    crow("K8", "; ".join(f"{g}: " + ("met" if r["passed"] else "NOT MET") + ", largest gap " + (f"{max((p.get('gap') or 0) for p in r['rows']):.1e}" if r["rows"] else "n/a")
                         + f" on {sum(1 for p in r['rows'] if p.get('gap') is not None)} of {len(r['rows'])} points" for g, r in c.items())
         + "; with the hold energy counted into U at both strengths (the failure the control names) the gaps would be "
         + ", ".join(f"{g} {min(p['gap_with_the_hold_energy_counted_at_both'] for p in r['rows'] if p.get('gap_with_the_hold_energy_counted_at_both') is not None):.4f} to "
                     f"{max(p['gap_with_the_hold_energy_counted_at_both'] for p in r['rows'] if p.get('gap_with_the_hold_energy_counted_at_both') is not None):.4f}"
                     for g, r in c.items() if any(p.get('gap_with_the_hold_energy_counted_at_both') is not None for p in r['rows']))
         + ", every one under its tolerance: as built the control could not fail for its named failure")
    c = C.get("K9", {}).get("numbers", {})
    crow("K9", "; ".join(f"{g}: " + "; ".join(f"R_d {o['Rd']:g}: {o['word']}, U {o['U']:.5f}, obstacle energy {o['Eo']:.2e} (start {o['Eo_at_the_start']:.2e}), radius {fm(o['radius'])}"
                                              for o in r["rows"]) for g, r in c.items()))
    c = C.get("K10", {})
    crow("K10", f"{c.get('numbers', {}).get('records')} made-up records, failed: {c.get('numbers', {}).get('failed')}")
    c = C.get("K11", {}).get("numbers", {})
    crow("K11", "; ".join(f"{n}: {r.get('iterations')} iterations against {r.get('iterations_of_m5_46')}, U at the end {r.get('U_at_the_end')} against {r.get('U_of_m5_46')}, "
                          f"{r.get('dense_reads')} dense reads from iteration {r.get('dense_from')}" for n, r in c.items()))
    c = C.get("K12", {}).get("numbers", [])
    crow("K12", "; ".join(f"{r['state']}: largest change at zero amplitude {max(r['largest_change_at_zero_amplitude']):.0e}, source points outside the box "
                          f"{sum(r['source_points_outside_the_box'].values())}, the pinch at 0.5 gives {r['sets_after_the_pinch']} set with radii in the ratio {r['radii_min_over_max']:.3f}" for r in c))
    T.append("")
    n = 2
    for g, L in S["ladders"].items():
        T.append(f"**Table {n}{'abcd'[list(S['ladders']).index(g)]}. The ladder of {GN[g]}.** Free state: U {L['free']['U']:.6f}, D {L['free']['D']:.4f}, reader's separation {L['free']['separation']:.4f}.\n")
        T.append("| Target d | Kind | D reached | Reader's separation | U | Rise of U | f | Hold energy | Room | Word | Clear sets | Weight within 1.5 | Counted | e of the step from the counted rung below |")
        T.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        es = {s["hi"]: s["e"] for s in L["analysis"].get("steps", [])}
        for r in L["rungs"]:
            wd = r["word"] + (", EVADED" if r["evaded"] else "") + (f" ({r['end_how']})" if r.get("end_how") and r["converged"] else "")
            T.append(f"| {r['d']:g} | {r['kind']} | {r['D']:.4f} | {fm(r['sep'])} | {r['U']:.6f} | {r['rise']:.6f} | {fm(r['f'])} | {r['Eh']:.2e} | {fm(r['room'], 3)} | {wd} | {r['clear_sets']} | "
                     f"{fm(r['share'], 3)} | {'yes' if r['counted'] else 'no' + (': ' + r['why_not'] if r['why_not'] else '')} | {fm(es.get(r['d']), 4) if r['d'] in es else ''} |")
        T.append("")
    n += 1
    T.append(f"**Table {n}. The room rule: the shared targets of the two boxes.**\n")
    T.append("| Spacing | Target | Room on half-box 10 | Rise on half-box 10 | Rise on half-box 14 | Gap | Agrees (5 percent) | Room rule |")
    T.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for s, sp in S["spacings"].items():
        if not sp.get("ready"):
            continue
        for r in sp["shared_targets"]:
            T.append(f"| {'0.5' if s == 'a' else '0.25'} | {r['d']:g} | {r['room10']:.3f} | {r['rise10']:.6f} | {r['rise14']:.6f} | {r['gap']:.4f} | {'yes' if r['agrees'] else 'no'} | "
                     f"{fm(sp['room_rule'], 3) if sp['room_rule'] is not None else 'none'} |")
    T.append("")
    n += 1
    T.append(f"**Table {n}. R1 to R3 per grid.** Where a label is withheld, the words that the frozen rules give on the chains as run are printed beside it; they are not labels.\n")
    T.append("Two readings of the room clause are printed when no room rule exists (deviation 26): the reads script skips the clause (\"skipped\"), and the literal reading, in which eligibility needs a room rule and no rung is counted (\"literal\"). The print with no cut is given as the plan defines it (EVADED rungs out) and with the EVADED rungs in (deviation 27).\n")
    T.append("| Grid | Labels R1, R2, R3 | R1 as run, skipped | R1 as run, literal | R1 with no cut | R1 with no cut, EVADED rungs in | R2 as run, skipped / literal | R3 as run, skipped / literal |")
    T.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for g, L in S["ladders"].items():
        u = L["uncut"]
        ud = (f"{u['rungs']} rungs, U rises at every step: {'yes' if u.get('U_rises') else 'no'}, largest f {fm(u.get('f_max'))}, mean of the last two {fm(u.get('f_last_two'))} "
              f"({fm(u.get('ratio_last_two'), 3)} of the largest), least f {fm(u.get('f_min_over_max'), 3)} of the largest") if u.get("rungs") else "no rung"
        lab = ("no label (all three): K2 is not met on the base of half-box 14 at this spacing" if L.get("withheld")
               else f"R1 {L['R1']['label']}; R2 {L['R2']['label']}; R3 {L['R3']['label']}")
        u2 = L["uncut_with_evaded"]
        ue = (f"{u2['rungs']} rungs, U rises at every step: {'yes' if u2.get('U_rises') else 'no'}, largest f {fm(u2.get('f_max'))}, mean of the last two {fm(u2.get('f_last_two'))} "
              f"({fm(u2.get('ratio_last_two'), 3)} of the largest)") if u2.get("rungs") else "no rung"
        li = L["as_run_literal"]
        T.append(f"| {GN[g]} | {lab} | {L['R1']['as_run']} | {li['R1']} | {ud} | {ue} | {L['R2']['as_run']} / {li['R2']} | {L['R3']['as_run']} / {li['R3']} |")
    T.append("")
    n += 1
    T.append(f"**Table {n}. The counted runs: the numbers under R1.**\n")
    T.append("| Grid | Counted rungs | Ends at | Ended by | Largest f | Mean of f on the last two | Ratio | Largest e of a counted step | Bisections | Lower states | Start dependence |")
    T.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for g, L in S["ladders"].items():
        an = L["analysis"]
        if not an.get("final"):
            T.append(f"| {GN[g]} | not final | | | | | | | | | |")
            continue
        T.append(f"| {GN[g]} | {len(an['counted'])} ({', '.join(f'{d:g}' for d in an['counted'])}) | {an['ends_at'] if an['ends_at'] is not None else 'n/a'} | {an['ended_by']} | {fm(an['f_max'])} | "
                 f"{fm(an['f_last_two'])} | {fm(an['ratio_last_two'], 3)} | {fm(max((s['e'] for s in an['steps']), default=None), 4)} | {an['bisections_used']}"
                 f"{' (' + '; '.join(an['notes']) + ')' if an['notes'] else ''}{' ' + json.dumps(an['classes']) if an['classes'] else ''} | "
                 f"{'; '.join(str(x['at']) for x in an['lower_states']) or 'none'} | "
                 f"{'; '.join(str(x['d']) + ': ' + format(x['share_of_the_rise'], '.3f') for x in an['start_dependent']) or 'none'} |")
    T.append("")
    n += 1
    T.append(f"**Table {n}. K8: the second strength (μ = 80) against the cubic through the points and slopes at μ = 20.**\n")
    T.append("| Grid | Target | D reached | U | Cubic at that D | Gap | Tolerance | Gap with the hold energy counted into U | Word |")
    T.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for g, L in S["ladders"].items():
        for p in L["K8"]["rows"]:
            if "D" not in p:
                T.append(f"| {GN[g]} | {p['d']:g} | no record | | | | | | NOT MET |")
                continue
            T.append(f"| {GN[g]} | {p['d']:g} | {p['D']:.4f} | {p['U']:.6f} | {fm(p['curve'], 6)} | {fm(p['gap'], 6)} | {p['tolerance']:.4f} | {fm(p['gap_with_the_hold_energy_counted'], 4)} | "
                     f"{p['word']}{', EVADED' if p['evaded'] else ''}: {'met' if p['passed'] else 'NOT MET'} |")
    T.append("")
    n += 1
    T.append(f"**Table {n}. K9 and the second kind of hold (the obstacle), spacing 0.5.**\n")
    T.append("| Grid | R_d | Word | U | Reader's separation | D | U of the first hold at that separation | Gap over the rise | U of the first hold at that D | Gap over the rise |")
    T.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for g, L in S["ladders"].items():
        for o in L["K9"]["rows"]:
            T.append(f"| {GN[g]} | {o['Rd']:g} | {o['word']} | {o['U']:.6f} | {fm(o['separation'])} | {o['D']:.4f} | {fm(o.get('U_first_hold_at_the_same_separation'), 6)} | "
                     f"{fm(o.get('gap_over_the_rise'), 4)} | {fm(o.get('U_first_hold_at_the_same_D'), 6)} | {fm(o.get('gap_over_the_rise_at_the_same_D'), 4)} |")
    T.append("")
    n += 1
    T.append(f"**Table {n}. The continuation chains at spacing 0.5 against the fresh rungs.** Up: each rung from the end of the rung below, from the free state; down: from the top, each from the end of the rung above. The share is that of the weight within 1.5 of the loop's plaquettes.\n")
    T.append("| Grid | Direction | Target d | Word | D reached | Reader's separation | Share | U | U of the fresh rung | Gap over the rise |")
    T.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for g, L in S["ladders"].items():
        for c in L["continuation"]:
            wd = ("CONVERGED" if c["converged"] else "not CONVERGED") + (", EVADED" if c.get("evaded") else "") + ("" if c["clear_sets"] == 1 else f", {c['clear_sets']} clear sets")
            T.append(f"| {GN[g]} | {c['direction']} | {c['d']:g} | {wd} | {c['D']:.4f} | {fm(c['sep'])} | {fm(c.get('share'), 3)} | {c['U']:.6f} | {fm(c['U_fresh'], 6)} | "
                     f"{fm(c['gap_over_the_rise'], 4)} |")
    T.append("")
    n += 1
    T.append(f"**Table {n}. The family that seeks a split, on the top counted rung of each grid.**\n")
    T.append("| Grid | Member | Sets of the start | Word of the chain | Clear sets at the end | U at the end | U of the rung | D | Reader's separation | Release | Word |")
    T.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for g, L in S["ladders"].items():
        top = L["analysis"].get("top") or {}
        for m in L["family"]:
            rel = m["release"]
            T.append(f"| {GN[g]} | {m['member']} | {m['start_sets']} | {m['end_how']} [{'CONVERGED' if m['converged'] else 'not CONVERGED'}{', EVADED' if m.get('evaded') else ''}] | {m['clear_sets']} | {m['U']:.6f} | {fm(top.get('U'), 6)} | "
                     f"{m['D']:.4f} | {fm(m['sep'])} | {('CONVERGED' if rel['converged'] else 'not CONVERGED') + ', ' + str(rel['clear_sets']) + ' clear sets, U ' + format(rel['U'], '.6f') if rel else 'none'} | "
                     f"{m['word']} |")
    T.append("")
    n += 1
    T.append(f"**Table {n}. R4: the exit chains.** One character per read: 1 or 0 on a liftable surface, L at the loop, B blind; runs are written as class × number of reads.\n")
    T.append("| Chain | How it ended | Iterations | Cadence reads, dense reads (from) | Faces touched at the last read with a set | Surfaces around the loop at the first read | The box surface, read by read | R4 |")
    T.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for nme, r in S["R4"].items():
        box = list(r["all_surfaces"].values())[-1]
        T.append(f"| {nme} | {r['ended']} | {r['iterations']} | {r['reads']}, {r['dense_reads']} ({r['dense_from']}) | {', '.join(r['faces_at_the_last_read_with_a_set']) or 'none'} | {len(r['seqs'])} | "
                 f"{rle(box)} | {r['label']} |")
    T.append("")
    n += 1
    T.append(f"**Table {n}. The desk re-read of the four added chains of M5.46 (no label of R4).**\n")
    T.append("| Chain of M5.46 | How it ended | Reads | Faces touched at the last read with a set | The outermost surface around the loop, read by read | The rules of R4 applied to the stored record |")
    T.append("| --- | --- | --- | --- | --- | --- |")
    for nme, r in S["R4_reread_of_m5_46"].items():
        last = list(r["seqs"].values())[-1] if r["seqs"] else ""
        T.append(f"| {nme} | {r['ended']} | {r['reads']} | {', '.join(r['faces_at_the_last_read_with_a_set']) or 'none'} | {rle(last)} | {r['reread_label']} |")
    T.append("")
    n += 1
    T.append(f"**Table {n}. R5: the two spacings of a box.**\n")
    T.append("| Box | R5 | As run, the room clause skipped | As run, literal |")
    T.append("| --- | --- | --- | --- |")
    for b, r in S["R5"].items():
        T.append(f"| half-box {b} | {r['label']} | {r['as_run']} | {r.get('as_run_literal')} |")
    T.append("")
    n += 1
    T.append(f"**Table {n}. The chains by class.**\n")
    T.append("| Class | Chains | CONVERGED | EXIT | Other ends | With two clear sets or more at the end |")
    T.append("| --- | --- | --- | --- | --- | --- |")
    by = {}
    for nme, c in S["chains"].items():
        by.setdefault(c["cls"], []).append(c)
    for k, cs in sorted(by.items(), key=lambda kv: str(kv[0])):
        other = sorted({c["word"] for c in cs if c["word"] not in ("CONVERGED", "EXIT")})
        T.append(f"| {k} | {len(cs)} | {sum(c['word'] == 'CONVERGED' for c in cs)} | {sum(c['word'] == 'EXIT' for c in cs)} | {', '.join(other) or 'none'} | {sum(c['clear_sets'] >= 2 for c in cs)} |")
    T.append("")
    out = Path(a.work_dir) / "tables.md"
    out.write_text("\n".join(T) + "\n")
    print(f"wrote {out} ({len(T)} lines)")


def plots(S, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm
    import m5_50_hold as H
    P = Path(a.plot_dir)
    P.mkdir(exist_ok=True)

    # 1. the ladders
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    for g, L in S["ladders"].items():
        ev = [r for r in L["rungs"] if r["kind"] != "inward" and r["converged"] and r["clear_sets"] == 1 and r["evaded"]]
        ok = [r for r in L["rungs"] if r["kind"] != "inward" and r["converged"] and r["clear_sets"] == 1 and not r["evaded"]]
        cn = [r for r in ok if r["counted"]]
        un = [r for r in ok if not r["counted"]]
        ax[0].plot([L["free"]["separation"]] + [r["sep"] for r in ok], [0.0] + [r["rise"] for r in ok], "-", color=COL[g], lw=0.8, alpha=0.6)
        ax[0].plot([r["sep"] for r in cn], [r["rise"] for r in cn], "o", color=COL[g], ms=5, label=GN[g])
        ax[0].plot([r["sep"] for r in un], [r["rise"] for r in un], "o", mfc="none", color=COL[g], ms=5)
        ax[0].plot([r["D"] for r in ev], [r["rise"] for r in ev], "v", color=COL[g], ms=6)
        k8 = [p for p in L["K8"]["rows"] if p.get("gap") is not None]
        ax[1].plot([r["sep"] for r in ok], [r["f"] for r in ok], "-", color=COL[g], lw=0.8, alpha=0.6)
        ax[1].plot([r["sep"] for r in cn], [r["f"] for r in cn], "o", color=COL[g], ms=5, label=GN[g])
        ax[1].plot([r["sep"] for r in un], [r["f"] for r in un], "o", mfc="none", color=COL[g], ms=5)
        if k8:
            seps = {r["d"]: r["sep"] for r in ok}
            ax[1].plot([seps.get(p["d"], np.nan) for p in k8], [p["f"] for p in k8], "x", color=COL[g], ms=6)
        ob = [o for o in L["K9"]["rows"] if o["converged"] and o["separation"]]
        if ob:
            ax[0].plot([o["separation"] for o in ob], [o["U"] - L["free"]["U"] for o in ob], "s", mfc="none", color=COL[g], ms=8)
        st = L["analysis"].get("steps", [])
        seps = {r["d"]: r["sep"] for r in ok}
        ax[2].semilogy([seps[s_["hi"]] for s_ in st], [max(s_["e"], 1e-6) for s_ in st], "o-", color=COL[g], ms=4, lw=0.8, label=GN[g])
    ax[2].axhline(0.10, color="0.4", ls="--", lw=0.8)
    ax[2].axhline(0.05, color="0.4", ls=":", lw=0.8)
    ax[0].set(xlabel="separation of the reader (twice the loop's radius); triangles: EVADED, at D", ylabel="U − U of the free state",
              title="The cost of the separation\n(filled: counted run; open: not counted; squares: second hold)")
    ax[1].set(xlabel="separation of the reader", ylabel="f = μ (d − D)", title="The force of the hold\n(crosses: μ = 80)")
    ax[2].set(xlabel="separation of the reader at the upper rung", ylabel="e of the step", title="The step identity on the counted runs\n(dashed 0.10, dotted 0.05)")
    for x in ax:
        x.grid(alpha=0.3)
    ax[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(P / "m5_50_ladders.png", dpi=110)
    plt.close(fig)
    print("wrote m5_50_ladders.png")

    # 2. the cross-sections
    for g, L in S["ladders"].items():
        top = L["analysis"].get("top")
        names = [("base", H.GRIDS[g]["base"])]
        if top:
            names.append((f"top rung, d = {top['d']:g}", f"m5_50_{top['name']}_end.npz"))
        for m in L["family"]:
            names.append((f"{m['member']}: {m['word']}{', EVADED' if m.get('evaded') else ''}", f"m5_50_{m['name']}_end.npz"))
            if m.get("release"):
                names.append((f"{m['member']}, released", f"m5_50_L_{m['name']}_end.npz"))
        names = [(t, f) for t, f in names if (Path(a.out_dir) / f).exists()]
        if not names:
            continue
        fig, ax = plt.subplots(2, len(names), figsize=(max(2.6 * len(names), 7.0), 6.0), squeeze=False)
        for j, (t, f) in enumerate(names):
            m6, x = H.load_state(Path(a.out_dir) / f)
            u = H.hold_D(m6, x)["u"]
            k0 = (len(x) - 1) // 2
            for i, (sl, lab) in enumerate(((u[:, :, k0], "plane z = 0"), (u[:, k0, :], "plane y = 0"))):
                ax[i, j].imshow(np.maximum(sl.T, 1e-8), origin="lower", extent=[x[0], x[-1], x[0], x[-1]], norm=LogNorm(1e-6, 1.0), cmap="magma")
                ax[i, j].set_title(t if i == 0 else lab, fontsize=7)
                ax[i, j].tick_params(labelsize=6)
                ax[i, j].set_xticks([x[0], 0.0, x[-1]])
                ax[i, j].set_yticks([x[0], 0.0, x[-1]])
        fig.suptitle(f"M5.50, {GN[g]}\nthe potential density u (log scale, 1e-6 to 1): plane of the loop (top), through its axis (bottom)", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(P / f"m5_50_sections_{g}.png", dpi=95)
        plt.close(fig)
        print(f"wrote m5_50_sections_{g}.png")

    # 3. the exit chains
    R4 = S["R4"]
    if R4:
        cmap = ListedColormap(["#2c7bb6", "#fdae61", "#f0f0f0", "#d7191c", "#000000"])
        code = {"1": 0, "L": 1, "0": 2, "B": 3}
        nr = (len(R4) + 3) // 4
        fig, ax = plt.subplots(nr, 4, figsize=(17, 3.4 * nr + 0.4), squeeze=False)
        for k, (nme, r) in enumerate(R4.items()):
            x_ = ax[k // 4, k % 4]
            hw = list(r["all_surfaces"])
            M = np.array([[code.get(ch, 4) for ch in r["all_surfaces"][w]] for w in hw])
            x_.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
            tk = np.linspace(0, len(r["its"]) - 1, 6).astype(int)
            x_.set_xticks(tk)
            x_.set_xticklabels([r["its"][t] for t in tk], fontsize=6)
            x_.set_yticks(range(0, len(hw), 2))
            x_.set_yticklabels([hw[i] for i in range(0, len(hw), 2)], fontsize=6)
            blind = sorted({j for w in hw for j, ch in enumerate(r["all_surfaces"][w]) if ch == "B"})
            for j in blind:
                x_.axvline(j, color="#d7191c", lw=0.8, ls="--")
            if r.get("dense_from") is not None and r["dense_from"] in r["its"]:
                x_.axvline(r["its"].index(r["dense_from"]), color="0.2", lw=0.8, ls=":")
            bl = f"; blind at it {', '.join(str(r['its'][j]) for j in blind)}" if blind else ""
            x_.set_title(f"{nme}: {r['ended']} at {r['iterations']}; R4 {r['word']}{bl}", fontsize=7.5)
            x_.set_xlabel("read, in order (labels: its iteration; dotted line: dense reads from here)", fontsize=6.5)
            x_.set_ylabel("half-width of the surface", fontsize=7)
        for k in range(len(R4), 4 * nr):
            ax[k // 4, k % 4].axis("off")
        fig.suptitle("M5.50: the class of every nested surface at every read of the exit chains (blue 1, orange at the loop, light 0, red blind, black another whole number; red dashes mark a read with a blind surface)", fontsize=9)
        fig.tight_layout()
        fig.savefig(P / "m5_50_exits.png", dpi=100)
        plt.close(fig)
        print("wrote m5_50_exits.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--plot-dir", type=Path, default=HERE.parent / "plots")
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_50")
    a = ap.parse_args()
    S = json.loads((a.out_dir / "m5_50_summary.json").read_text())
    if a.tables:
        tables(S, a)
    if a.plots:
        plots(S, a)


if __name__ == "__main__":
    main()
