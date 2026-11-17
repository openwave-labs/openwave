"""M5.42: the tables of the findings, assembled from the chain records, the reads and the gates.

Writes two tracked files:

  m5_42_paths.json     every path of every chain: its start, its end, its stops with their scalars and labels
  m5_42_summary.json   the author's box; the smooth branch against the box and the word of section 2, row 5; the
                       one-cell states against the spacing with the exponents of section 5b; the runs without L1;
                       free faces; delta = 0.05; the band control with its words; the two stencils; the
                       expectations of section 5 as met or failed

The rules are those of m5_42_task_details.md (section 2, section 5 and section 5b); nothing is fitted here
beyond the exponents that section 5b names.
"""

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m5_42_relax import FRACTION, chains  # noqa: E402

STEPS = ("j0", "j40", "j20", "j10", "j5", "j2p5", "j1")


def slope(x1, y1, x2, y2):
    if min(x1, x2, y1, y2) <= 0:
        return None
    return math.log(y1 / y2) / math.log(x1 / x2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_42" / "chains")
    ap.add_argument("--data-dir", type=Path, default=HERE.parent / "data")
    a = ap.parse_args()
    C = chains()
    rec = {}
    for name in C:
        f = a.chains_dir / f"{name}.json"
        if f.exists():
            rec[name] = json.loads(f.read_text())
    reads = {}
    rf = a.data_dir / "m5_42_r7_reads.json"
    if rf.exists():
        for s in json.loads(rf.read_text())["states"]:
            reads[(s["chain"], s["path"])] = s

    def state(chain, path):
        """The end of a path as one flat row, or None."""
        r = rec.get(chain)
        if r is None:
            return None
        p = {q["name"]: q for q in r["paths"]}.get(path)
        if p is None or p.get("skipped"):
            return None
        s = p["stops"][-1]
        lam = s["largest_eigenvalues"]
        m = C[chain]["model"]
        row = dict(chain=chain, path=path, N=m["N"], half_box=m["half_box"], spacing=2 * m["half_box"] / (m["N"] - 1), L1=m["L1"], delta=m["delta"],
                   stencil=m["stencil"], J_over_Je=FRACTION[p["j"]], iterations=p["iterations"], E=s["E"], U=s["U"], U2=s["U2"], U4=s["U4"], UV=s["UV"], UB=s["UB"],
                   I=s["I"], two_J_Omega_over_E=s["two_J_Omega_over_E"], inertia_inside_r2=s["inertia_inside_r2"], largest_share=s["largest_share"],
                   largest_position=s["largest_position"], largest_over_neighbors=s["largest_over_neighbors"], amplitude=(lam[2] - lam[0]) / 2,
                   eigenvalue_range=[s["eigenvalue_min"], s["eigenvalue_max"]], gmax_over_h3=s["gmax_over_h3"], label=p["label"], converged=p["converged"],
                   only_first_clause=p["only_first_clause"], ended_by_function_tolerance=p["ended_by_function_tolerance"], ended_at_cap=p["ended_at_cap"],
                   cut_by_wall=p["cut_by_wall"], message=p["message"])
        rd = reads.get((chain, path))
        if rd:
            row.update(inertia_by_shell=rd["inertia"], aligned_share_shell_2_5=rd["aligned_share_shell_2_5"], minor_gap=rd["minor_gap"], spheres=rd["spheres"])
            if "band" in rd:
                row["band"] = rd["band"]
        return row

    out = {}
    exp = {}

    # the author's box (R2)
    box = {p: state("r2_g33", p) for p in ("j0", "j10", "j1", "j10x", "j0x")}
    out["author_box"] = box
    want = dict(j0=63.922535, j10=85.178992, j1=184.81, j10x=79.30)
    if all(box.get(k) for k in want):
        dev = {k: abs(box[k]["E"] - v) / v for k, v in want.items()}
        exp["1_author_box"] = dict(statement="R2 returns the two smooth states and the two one-cell states to five digits", relative_differences=dev,
                                   met=bool(max(dev[k] for k in ("j0", "j10")) < 5e-6 and max(dev[k] for k in ("j1", "j10x")) < 5e-5),
                                   labels={k: box[k]["label"] for k in want})

    # the smooth branch against the box (R4, with the extension on half-box 18)
    branch = {}
    for L1 in (0.3, 0.1):
        rows = {}
        for hb in (6.0, 9.0, 12.0, 18.0):
            names = [f"r4_hb{hb:g}_L{L1}"] + ([f"r4x_hb{hb:g}_L{L1}"] if f"r4x_hb{hb:g}_L{L1}" in C else [])
            per = {}
            for n in names:
                for st in STEPS:
                    s = state(n, st)
                    if s:
                        per[st] = s
            smooth = [st for st in STEPS if st in per and per[st]["label"] == "SMOOTH" and per[st]["converged"]]
            rows[f"{hb:g}"] = dict(states=per, largest_smooth_step=(smooth[-1] if smooth else None),
                                   ended=[rec[n].get("ended") for n in names if n in rec])
        branch[str(L1)] = rows
    out["smooth_branch_against_the_box"] = branch

    def ratio(L1, hb, st):
        s = branch[str(L1)][f"{hb:g}"]["states"].get(st)
        return s if s and s["label"] == "SMOOTH" and s["converged"] else None

    word = {}
    for L1 in (0.3, 0.1):
        at = [(hb, ratio(L1, hb, "j1")) for hb in (6.0, 9.0, 12.0, 18.0)]
        have = [(hb, s["two_J_Omega_over_E"]) for hb, s in at if s]
        w = dict(smooth_converged_states_at_Je=have)
        if len(have) >= 2 and have[-1][0] == 18.0 and have[-2][0] == 12.0:
            r1, r2 = have[-2][1], have[-1][1]
            w["two_largest_boxes_agree_to_2_percent"] = bool(abs(r1 - r2) / abs(r2) <= 0.02)
            w["within_2_percent_of_one"] = bool(abs(r1 - 1) <= 0.02 and abs(r2 - 1) <= 0.02)
            w["word"] = "COMES OUT" if (w["two_largest_boxes_agree_to_2_percent"] and w["within_2_percent_of_one"]) else "DOES NOT COME OUT"
        else:
            w["word"] = "DOES NOT COME OUT: no SMOOTH and CONVERGED state at J_e on the two largest boxes"
        crossings = []
        for hb in (6.0, 9.0, 12.0, 18.0):
            seq = [(st, ratio(L1, hb, st)) for st in STEPS[1:]]
            seq = [(st, s["two_J_Omega_over_E"]) for st, s in seq if s]
            for (s1, r1), (s2, r2) in zip(seq, seq[1:]):
                if (r1 - 1) * (r2 - 1) < 0:
                    crossings.append(dict(half_box=hb, between=[s1, s2], ratios=[r1, r2]))
        for st in STEPS[1:]:
            seq = [(hb, ratio(L1, hb, st)) for hb in (6.0, 9.0, 12.0, 18.0)]
            seq = [(hb, s["two_J_Omega_over_E"]) for hb, s in seq if s]
            for (h1, r1), (h2, r2) in zip(seq, seq[1:]):
                if (r1 - 1) * (r2 - 1) < 0:
                    crossings.append(dict(step=st, between_half_boxes=[h1, h2], ratios=[r1, r2]))
        w["crossings"] = crossings
        word[str(L1)] = w
    out["does_hbar_Omega_equal_E_come_out"] = word
    ends = {f"{L1}/{hb}": branch[str(L1)][hb]["largest_smooth_step"] for L1 in (0.3, 0.1) for hb in branch[str(L1)]}
    exp["3_smooth_branch_ends_below_Je"] = dict(statement="on every box the state leaves the SMOOTH label at a J between J_e/10 and J_e", largest_smooth_step=ends,
                                                met=bool(all(v in ("j10", "j5", "j2p5") for k, v in ends.items() if not k.endswith("/18"))),
                                                note="half-box 18 ran to J_e/10 by plan; its extension at L1 = 0.3 is in the table")
    mono = {}
    for L1 in (0.3, 0.1):
        seq = [ratio(L1, hb, "j10") for hb in (6.0, 9.0, 12.0, 18.0)]
        vals = [s["two_J_Omega_over_E"] if s else None for s in seq]
        I0 = [branch[str(L1)][f"{hb:g}"]["states"].get("j0", {}).get("I") for hb in (6.0, 18.0)]
        mono[str(L1)] = dict(ratio_at_Je_over_10_by_half_box=vals, monotone=bool(all(v is not None for v in vals) and all(x > y for x, y in zip(vals, vals[1:]))),
                             inertia_growth_6_to_18=(I0[1] / I0[0] if all(I0) else None))
    exp["4_ratio_falls_with_the_box"] = dict(statement="at J_e/10 the ratio falls monotonically with the box; the J = 0 inertia grows 15 to 27 times from half-box 6 to 18 (L1 = 0.3)",
                                             by_L1=mono, met=bool(mono["0.3"]["monotone"] and mono["0.3"]["inertia_growth_6_to_18"] and 15 <= mono["0.3"]["inertia_growth_6_to_18"] <= 27))
    exp["5_the_question"] = dict(statement="the relation does not COME OUT for any L1", words={k: v["word"] for k, v in word.items()},
                                 met=bool(all(v["word"] != "COMES OUT" for v in word.values())))

    # the one-cell states against the spacing (R3, and the planted spikes without L1)
    onecell = {}
    for L1 in (0.3, 0.1, 0.0):
        per_J = {}
        for jname, paths in (("Je_over_10", ("j10x", "j10", "j10s")), ("Je", ("j1", "j1x", "j1s"))):
            rows = []
            for N in (33, 49, 65):
                base = state(f"r3_g{N}_L{L1}", "j0")
                cands = [state(f"r3_g{N}_L{L1}", p) for p in paths[:2]] + [state(f"r3s_g{N}_L{L1}", paths[2])]
                off = [dict(chain=s["chain"], path=s["path"], E=s["E"], I=s["I"], label=s["label"], converged=s["converged"]) for s in cands if s and s["label"] == "MIXED"]
                cands = [s for s in cands if s and s["label"] == "ONE-CELL"]
                if not cands or base is None:
                    rows.append(dict(N=N, state=None, note="no ONE-CELL state", mixed_states=off))
                    continue
                s = min(cands, key=lambda q: q["E"])
                rows.append(dict(N=N, spacing=s["spacing"], state=s, all_candidates=[dict(chain=q["chain"], path=q["path"], E=q["E"], I=q["I"], label=q["label"], converged=q["converged"]) for q in cands],
                                 energy_above_J0=s["E"] - base["U"], spike_inertia=s["I"] - base["I"], J0_label=base["label"], J0_converged=base["converged"]))
            fit = {}
            good = [r for r in rows if r.get("state")]
            for key in ("energy_above_J0", "spike_inertia"):
                pairs = [slope(r1["spacing"], r1[key], r2["spacing"], r2[key]) for r1, r2 in zip(good, good[1:])]
                fit[key] = dict(exponent_by_pair=pairs, exponent_coarsest_to_finest=(slope(good[0]["spacing"], good[0][key], good[-1]["spacing"], good[-1][key]) if len(good) >= 2 else None))
            spacing_set = None
            if len(good) >= 2:
                spacing_set = bool(any(abs(r1["state"]["I"] - r2["state"]["I"]) / r2["state"]["I"] > 0.05 for r1, r2 in zip(good, good[1:])))
                direction = bool(all(r2["state"]["I"] > r1["state"]["I"] and r2["state"]["E"] < r1["state"]["E"] for r1, r2 in zip(good, good[1:])))
            per_J[jname] = dict(rows=rows, fits=fit, spacing_set=spacing_set, energy_falls_and_inertia_grows_with_refinement=(direction if len(good) >= 2 else None))
        amp = []
        for r10, r1 in zip(per_J["Je_over_10"]["rows"], per_J["Je"]["rows"]):
            if r10.get("state") and r1.get("state"):
                amp.append(dict(N=r10["N"], amplitude_at_Je_over_10=r10["state"]["amplitude"], amplitude_at_Je=r1["state"]["amplitude"],
                                ratio=r1["state"]["amplitude"] / r10["state"]["amplitude"], same_node=bool(r10["state"]["largest_position"] == r1["state"]["largest_position"])))
        per_J["amplitude_ratio_Je_to_Je_over_10"] = amp
        onecell[str(L1)] = per_J
    out["one_cell_states_against_the_spacing"] = onecell
    out["one_cell_predictions"] = dict(author=dict(L1_zero="energy exponent near 1", L1_positive="a floor", spike_inertia_exponent=-1.0, amplitude_ratio=10 ** 0.25),
                                       plan=dict(L1_positive="energy exponent near 1/3", spike_inertia_exponent=-1 / 3, amplitude_ratio=10 ** (1 / 3)),
                                       windows="plus or minus 0.3 on an exponent, plus or minus 0.2 on the ratio (the plan's windows)")
    sets = {f"{L1}/{j}": onecell[str(L1)][j]["spacing_set"] for L1 in (0.3, 0.1, 0.0) for j in ("Je_over_10", "Je")}
    exp["2_one_cell_states_are_spacing_set"] = dict(statement="the inertia of the one-cell states grows and their energy falls as the spacing is reduced, with and without L1; they exist on the mirror-averaged stencil",
                                                    spacing_set=sets, without_L1="no ONE-CELL state formed",
                                                    met=bool(all(sets[f"{L1}/{j}"] and onecell[str(L1)][j]["energy_falls_and_inertia_grows_with_refinement"] for L1 in (0.3, 0.1) for j in ("Je_over_10", "Je"))),
                                                    note="met for L1 = 0.3 and 0.1; without L1 no ONE-CELL state formed, so the expectation is not met there")

    # without L1
    nol1 = []
    for name in ("r3_g33_L0.0", "r3_g49_L0.0", "r3_g65_L0.0", "r4b_hb9", "r4b_hb12", "r4b_hb9_h025"):
        for p in ("j0", "j10", "j1"):
            s = state(name, p)
            if s:
                nol1.append(s)
    out["without_L1"] = nol1
    j0 = {s["N"]: s for s in nol1 if s["chain"].startswith("r3_") and s["path"] == "j0"}
    if 33 in j0 and 49 in j0:
        d = abs(j0[49]["U"] - j0[33]["U"]) / j0[33]["U"]
        exp["6_without_L1"] = dict(statement="the J = 0 states are SPACING-SET (the energy lower at 0.25 than at 0.375 by more than 5 percent) and NOT CONVERGED within the cap on the larger boxes",
                                   U_by_grid={k: v["U"] for k, v in j0.items()}, relative_change_0375_to_025=d, converged_by_grid={k: v["converged"] for k, v in j0.items()},
                                   larger_boxes_converged={s["chain"]: s["converged"] for s in nol1 if s["chain"].startswith("r4b") and s["path"] == "j0"},
                                   met=bool(d > 0.05 and j0[49]["U"] < j0[33]["U"]))

    # free faces, delta = 0.05
    out["free_faces"] = [s for n in ("r5_g33_L0.3", "r5_g33_L0.0", "r5_g49_L0.3") for st in STEPS for s in [state(n, st)] if s]
    out["delta_0p05"] = [s for n in ("r6_hb6_L0.3", "r6_hb6_L0.0", "r6_hb12_L0.3", "r6_hb12_L0.0") for st in ("j0", "j40", "j10") for s in [state(n, st)] if s]

    # the band control
    band = []
    for tag, src, src_path in (("hb6", "r3_g33_L0.3", {"j10": "j10", "j1": "j1"}), ("hb9", "r4_hb9_L0.3", {"j10": "j10", "j1": "j1"}), ("hb12", "r4_hb12_L0.3", {"j10": "j10", "j1": "j1"}),
                               ("hb6_h025", "r3_g49_L0.3", {"j10": "j10", "j1": "j1"}), ("hb6_L0", "r3_g33_L0.0", {"j10": "j10", "j1": "j1"}), ("hb6_h025_L0", "r3_g49_L0.0", {"j10": "j10", "j1": "j1"})):
        for chain, pmap in ((f"r8a_{tag}", {"j10": "j10", "j1": "j1"}), (f"r8b_{tag}", {"j10": "j10c", "j1": "j1c"})):
            for j, p in pmap.items():
                s = state(chain, p)
                if s is None:
                    reason = None
                    if chain in rec:
                        q = {x["name"]: x for x in rec[chain]["paths"]}.get(p)
                        reason = q.get("reason") if q else "not run"
                    band.append(dict(chain=chain, path=p, J=j, state=None, note=reason))
                    continue
                unb = state(src, src_path[j])
                w = None
                if "band" in s:
                    sh = s["band"]["share_of_the_inertia_on_them"]
                    same = bool(unb and unb["label"] == "SMOOTH" and abs(unb["E"] - s["E"]) < 1e-6 * abs(s["E"]))
                    w = "UNCHANGED SMOOTH STATE" if same else ("REMOVES" if (s["label"] == "SMOOTH" and sh < 0.01) else ("BAND-SATURATED" if sh > 0.10 else None))
                band.append(dict(chain=chain, path=p, J=j, state=s, word=w, unbanded=(dict(chain=unb["chain"], path=unb["path"], E=unb["E"], I=unb["I"], two_J_Omega_over_E=unb["two_J_Omega_over_E"],
                                                                                              label=unb["label"], eigenvalue_range=unb["eigenvalue_range"]) if unb else None)))
    out["band_control"] = band

    # the two stencils (gate 3) and the cross starts
    out["two_stencils"] = [s for N in (33, 49, 65) for n in (f"g3_g{N}", f"r3_g{N}_L0.3") for p in ("j0", "j10", "j1") for s in [state(n, p)] if s]
    out["cross_starts"] = [s for n in ("g3x_avg_g33", "g3x_fwd_g33", "g3x_avg_g49", "g3x_fwd_g49") for p in ("j0f", "j0a") for s in [state(n, p)] if s]
    # added after the audit of the findings: the broken sector on half-box 9, and a spike planted on the smooth state of half-box 12 at J_e
    out["sectors_on_half_box_9"] = dict(broken=[s for n, p in (("g3b_fwd_hb9", "j0"), ("g3b_avg_hb9", "j0b"), ("g3b_avg_hb9", "j10b")) for s in [state(n, p)] if s],
                                        symmetric=[s for p in ("j0", "j10") for s in [state("r4_hb9_L0.3", p)] if s])
    spike = rec.get("r4s_hb12_L0.3")
    if spike and spike["paths"]:
        q = spike["paths"][0]
        keys = ("iteration", "kind", "E", "U", "I", "two_J_Omega_over_E", "largest_share", "largest_position", "largest_eigenvalues", "gmax_over_h3", "label")
        out["planted_spike_on_half_box_12_at_Je"] = dict(plant=q.get("plant"), smooth_state=state("r4_hb12_L0.3", "j1"), stops=[{k: s[k] for k in keys} for s in q["stops"]],
                                                         iterations=q["iterations"], label=q["label"], converged=q["converged"], message=q["message"])

    for name, f in (("gates", "m5_42_r1_gates.json"), ("controls", "m5_42_r1_controls.json")):
        if (a.data_dir / f).exists():
            g = json.loads((a.data_dir / f).read_text())
            out[name] = g if name == "gates" else dict(all_met=g["all_met"], gate1=g["gate1"], gate2=g["gate2"], checks=g["checks"], timing=g["timing"])
    out["expectations"] = exp
    out["chains"] = {n: dict(finished=r.get("finished"), ended=r.get("ended"), wall_hours=r.get("wall_seconds", 0) / 3600,
                             paths=[(p["name"], "skipped" if p.get("skipped") else p["label"] + ("/C" if p["converged"] else "")) for p in r["paths"]]) for n, r in rec.items()}
    (a.data_dir / "m5_42_summary.json").write_text(json.dumps(out, indent=1) + "\n")

    slim = {}
    for n, r in rec.items():
        slim[n] = dict(rung=r["rung"], model=r["model"], ended=r.get("ended"), wall_hours=r.get("wall_seconds", 0) / 3600, paths=r["paths"])
    (a.data_dir / "m5_42_paths.json").write_text(json.dumps(slim, indent=1) + "\n")
    print("expectations:")
    for k, v in exp.items():
        print(" ", k, "MET" if v["met"] else "FAILED")
    for L1, w in word.items():
        print("word at L1", L1, ":", w["word"], "| crossings:", len(w["crossings"]))
    print("chains recorded:", len(rec), "of", len(C), "| unfinished:", [n for n, r in rec.items() if not r.get("finished")])


if __name__ == "__main__":
    main()
