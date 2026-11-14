"""M5.39 R2: the census, the reads and the declared comparisons, from the job files of the pool.

    python3 m5_39_r2_summary.py --archive <hashed copy of the card's code folder> [--threads 1]

Per grid: the five jobs with their labels, the stationary states and the lowest of them, the gap
between states against the drift of each over its last tenth of iterations and against the spread of
gate-passing states of one branch, the count of record lumps. Per stationary state (and per lowest
state of a grid without one, labeled so): the reads of m5_39_reads.py.

Units. The card's length unit is hbar c / (e f_pi) = 0.5014 fm, so e f_pi = 393.55 MeV under every
calibration; an inertia I in card units is I / (e^3 f_pi) in physical ones. Three scales are used:
the inherited one of the printed 33^3 row (e = 4.259, f_pi = 92.4 MeV); the printed 55^3 calibration
(e = 4.311, f_pi = 91.29 MeV); and the frozen rule applied to our own largest stationary box rung,
e^2 = 2 I Delta / (3 e f_pi) with Delta = 293 MeV.

Physical reads: the rotor gap 3 / (2 I) and the rigid moment (2/3) m_p I (the same as m_p / gap); the
radii; g_piNN = (8 pi M_N f_pi / 3) A / (e f_pi)^2 from the fitted tail amplitude A (card units), with
M_N = 938.272 MeV, the mass the author's reply of 13 November (20:38 UTC) names for this formula; and the
axial coupling that this tail implies through the Goldberger-Treiman relation, f_pi g_piNN / M_N =
(8 pi / 3) f_pi^2 A. The text's g_A = 0.904 is the axial integral, a route not computed here: the text
does not define that integral on the card's fields.

Comparisons declared before the run (m5_39_task_details.md, section 4), none adjusted here.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_39_reads as R  # noqa: E402

DATA = HERE.parent / "data"
CACHE = HERE.parent / "checkpoints" / "m5_39" / "reads"
HBARC, LENGTH = 197.3269804, 0.5014
EF = HBARC / LENGTH                      # e f_pi in MeV
M_N, M_P, DELTA = 938.272, 938.272, 293.0
SCALES = {"inherited": (4.259, 92.4), "printed_55": (4.311, 91.29)}
SEEDS = ("linked", "smooth", "iso")
GRIDS = {33: (3.6, SEEDS + ("chain", "reset")), 45: (6.075, SEEDS + ("chain", "reset")), 55: (6.075, SEEDS + ("chain", "reset")),
         65: (6.075, SEEDS + ("chain", "reset")), 77: (6.075, SEEDS + ("chain", "reset")),
         73: (8.1, SEEDS + ("cont", "contreset")), 91: (10.125, ("cont", "contreset"))}
PRINTED_33 = dict(energy=76.05003, center=(0.333, 0.376, 0.394), gap=311.9, mu_v=3.004, r2_s=0.2303, r2_v=0.4519)
PRINTED_55 = dict(inertia=37.44, g_a=0.904, g_pinn=(9.54, 9.55), tail_fm2_implied_by_g_a=0.504, tail_fm2_fitted=0.518, g_a_implied_by_the_tail=0.929)


def job(name):
    p = DATA / f"m5_39_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def unit_residual(name):
    """The force residual with the winding unknowns rescaled to unit length. The gradient on p is the
    gradient on q over |p|, and a relaxed state carries |p| above one, so the card's gate is not
    invariant under that rescaling; this is the same read at |p| = 1."""
    import m5_39_cardc as C

    with np.load(DATA / f"m5_39_{name}.npz") as d:
        state, coords = d["state"].copy(), d["coords"]
    norm = np.linalg.norm(state[..., :4], axis=-1)
    state[..., :4] /= norm[..., None]
    n = state.shape[0]
    h = float(coords[1, 0, 0, 0] - coords[0, 0, 0, 0])
    _, grad = C.CardC(n, float(coords[-1, 0, 0, 0]))(state)
    inner = C.interior(norm)
    return dict(residual=float(np.abs(C.interior(grad)).max() / h ** 3), largest_norm_of_p=float(inner.max()), smallest_norm_of_p=float(inner.min()))


def state_reads(name, archive):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{name}.json"
    out = json.loads(cache.read_text()) if cache.exists() else R.reads(DATA / f"m5_39_{name}.npz", archive)
    if "unit_p" not in out:
        out["unit_p"] = unit_residual(name)
    cache.write_text(json.dumps(out) + "\n")
    return out


def physical(inertia, scale):
    e, fpi = scale
    gap = 1.5 / inertia * e ** 3 * fpi  # the scale's own e and f_pi (their product is 393.5 MeV to 0.005 percent on every scale)
    return dict(e=e, f_pi=fpi, gap_MeV=gap, mu_v_rigid=M_P / gap)


def couplings(amplitude_card, scale):
    e, fpi = scale
    a_fm2 = amplitude_card * LENGTH ** 2
    g_a = 8.0 * np.pi / 3.0 * (fpi / HBARC) ** 2 * a_fm2
    g_pinn = 8.0 * np.pi / 3.0 * M_N * fpi * amplitude_card / (e * fpi) ** 2
    return dict(tail_amplitude_fm2=a_fm2, g_a_implied_by_the_tail=g_a, g_pinn_tail=g_pinn)


def outer_amplitude(tail, box):
    """The tail amplitude over the outer shells: the mean of the per-shell fits between half the box
    and 0.9 of it, and the spread there (the text calls the fit flat over the outer shells)."""
    sel = [r for r in tail if 0.5 * box <= r["r_lo"] and r["r_hi"] <= 0.9 * box + 1e-9]
    amps = np.array([r["amplitude"] for r in sel])
    return dict(shells=len(sel), mean=float(amps.mean()), spread=float(amps.max() - amps.min()),
                continuum_dipole_mean=float(np.mean([r["amplitude_against_the_continuum_dipole"] for r in sel])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--grids", type=int, nargs="*", default=sorted(GRIDS))
    a = ap.parse_args()
    summary = dict(task="M5.39 R2", units=dict(length_fm=LENGTH, e_f_pi_MeV=EF, M_N=M_N, m_p=M_P, Delta=DELTA, scales=SCALES), grids={})
    reads_out = {}

    for grid in a.grids:
        box, starts = GRIDS[grid]
        rows, missing = [], []
        for s in starts:
            j = job(f"r2_g{grid}_{s}")
            if j is None:
                missing.append(s)
                continue
            if j["label"] == "NOT RUN":
                rows.append(dict(start=s, label="NOT RUN", reason=j.get("reason")))
                continue
            rows.append(dict(start=s, name=j["name"], label=j["label"], energy=j["energy"], residual=j["residual"], iterations=j["iterations"],
                             evaluations=j["evaluations"], drift_last_tenth=j["energy_drift_last_tenth"], branch=j["branch"],
                             center_eigenvalues=j["center_eigenvalues"], degree=j["degree"], wall_hits=j["wall_hits"],
                             hemisphere_margin=j["hemisphere_margin_four_triangulations"], segments=[(x["segment"], x["iterations"], x["optimizer_message"]) for x in j["segments"]],
                             saved_states_under_the_gate=len(j["saved_states_under_the_gate"]), started_from=j["start"].get("source"),
                             start_energy=j["start"].get("energy"), wall_time_h=j["wall_time_s"] / 3600, threads=j["threads"]))
        g = dict(half_box=box, spacing=2 * box / (grid - 1), jobs=rows, missing=missing)
        done = [r for r in rows if r["label"] not in ("NOT RUN",)]
        stationary = [r for r in done if r["label"] == "STATIONARY"]
        pool = stationary or [r for r in done if r["energy"] == r["energy"]]
        if pool and not missing:
            best = min(pool, key=lambda r: r["energy"])
            g["state_of_the_grid"] = dict(name=best["name"], from_stationary=bool(stationary), energy=best["energy"], branch=best["branch"])
        # the reads, on every stationary state and on the lowest state of a grid without one
        for r in (stationary or pool[:0]) + ([min(pool, key=lambda r: r["energy"])] if pool and not stationary else []):
            rd = state_reads(r["name"], a.archive)
            left = rd["inertia_audit"]["actions"]["left"]
            inertia = left["isospin_mean"]
            amp = outer_amplitude(rd["tail"], box)
            r["reads"] = dict(
                lumps=rd["record_lumps"]["count"], lump_energy=rd["record_lumps"]["potential_energy_of_the_lumps"],
                residual_at_unit_p=rd["unit_p"]["residual"], largest_norm_of_p=rd["unit_p"]["largest_norm_of_p"],
                isospin_inertia=inertia, isospin_eigenvalues=left["isospin_eigenvalues"], spin_inertia=left["spin_mean"], mixed_inertia=left["mixed_mean"],
                isospin_inertia_other_actions={k: v["isospin_mean"] for k, v in rd["inertia_audit"]["actions"].items()},
                ours_against_the_audit=rd["isospin_block_ours_against_the_audit"], inertia_inside_radius=rd["isospin_inertia_ours"]["inertia_inside_radius"],
                winding_total=rd["winding"]["total"], r2_s_card=rd["winding"]["isoscalar_r2"], r2_v_card=rd["isospin_inertia_ours"]["isovector_r2"],
                r2_s_fm2=rd["winding"]["isoscalar_r2"] * LENGTH ** 2, r2_v_fm2=rd["isospin_inertia_ours"]["isovector_r2"] * LENGTH ** 2,
                tail=amp, physical={k: dict(physical(inertia, sc), **couplings(amp["mean"], sc)) for k, sc in SCALES.items()},
                mirror=[dict(axis=m["axis"], max_q=m["max_q"], max_N=m["max_N"], max_s=m["max_s"], energy_difference=m["energy_difference"],
                             record_quartic_difference=m["term_differences"][2]) for m in rd.get("mirror", [])])
            reads_out[r["name"]] = rd
        # ranking: the gap to the lowest stationary state against the drifts and the same-branch spread
        if len(stationary) > 1:
            low = min(stationary, key=lambda r: r["energy"])
            for r in stationary:
                r["gap_to_the_lowest_stationary"] = r["energy"] - low["energy"]
            by_branch = {}
            for r in stationary:
                if r.get("reads", {}).get("lumps", 0) == 0:
                    by_branch.setdefault(r["branch"], []).append(r["energy"])
            g["spread_within_branch"] = {b: max(v) - min(v) for b, v in by_branch.items() if len(v) > 1}
        summary["grids"][str(grid)] = g

    G = summary["grids"]

    def grid_state(grid):
        g = G.get(str(grid), {})
        sel = g.get("state_of_the_grid")
        if not sel:
            return None, None
        row = next(r for r in g["jobs"] if r.get("name") == sel["name"])
        return sel, row

    # the 33^3 control
    sel, row = grid_state(33)
    if sel and "reads" in row:
        rd = row["reads"]
        ph = rd["physical"]["inherited"]
        checks = dict(
            a_33_state_is_stationary=bool(sel["from_stationary"]),
            energy=dict(ours=row["energy"], printed=PRINTED_33["energy"], difference=row["energy"] - PRINTED_33["energy"], within=bool(abs(row["energy"] - PRINTED_33["energy"]) <= 3e-4)),
            center=dict(ours=row["center_eigenvalues"], printed=PRINTED_33["center"],
                        largest_difference=float(np.abs(np.array(row["center_eigenvalues"]) - np.array(PRINTED_33["center"])).max()),
                        within=bool(np.abs(np.array(row["center_eigenvalues"]) - np.array(PRINTED_33["center"])).max() <= 0.01)),
            gap=dict(ours=ph["gap_MeV"], printed=PRINTED_33["gap"], within=bool(abs(ph["gap_MeV"] - PRINTED_33["gap"]) <= 0.05), inertia=rd["isospin_inertia"]),
            mu_v=dict(ours_rigid_relation=ph["mu_v_rigid"], printed=PRINTED_33["mu_v"], within_printed_digits=bool(abs(ph["mu_v_rigid"] - PRINTED_33["mu_v"]) <= 0.0005),
                      relative_difference=ph["mu_v_rigid"] / PRINTED_33["mu_v"] - 1,
                      note="the printed moment is an integral whose definition is neither in the text nor in the archive; the text puts it 0.12 percent from the rigid relation"),
            r2_s=dict(ours=rd["r2_s_fm2"], printed=PRINTED_33["r2_s"], within_printed_digits=bool(abs(rd["r2_s_fm2"] - PRINTED_33["r2_s"]) <= 0.00005)),
            r2_v=dict(ours=rd["r2_v_fm2"], printed=PRINTED_33["r2_v"], within_printed_digits=bool(abs(rd["r2_v_fm2"] - PRINTED_33["r2_v"]) <= 0.00005)))
        passed = checks["a_33_state_is_stationary"] and all(checks[k].get("within", checks[k].get("within_printed_digits")) for k in ("energy", "center", "gap", "mu_v", "r2_s", "r2_v"))
        summary["control_33"] = dict(state=sel["name"], checks=checks, passed=bool(passed))

    # the 55^3 comparison and the calibration of our largest stationary box rung
    sel55, row55 = grid_state(55)
    if sel55 and "reads" in row55:
        rd = row55["reads"]
        ph = rd["physical"]["printed_55"]
        summary["comparison_55"] = dict(
            state=sel55["name"], stationary=bool(sel55["from_stationary"]), branch=sel55["branch"],
            inertia=dict(ours=rd["isospin_inertia"], printed=PRINTED_55["inertia"], within_printed_digits=bool(abs(rd["isospin_inertia"] - PRINTED_55["inertia"]) <= 0.005)),
            tail_amplitude_fm2=dict(ours=ph["tail_amplitude_fm2"], the_authors_fitted=PRINTED_55["tail_fm2_fitted"], implied_by_the_printed_g_a=PRINTED_55["tail_fm2_implied_by_g_a"],
                                    spread_over_the_outer_shells_card=rd["tail"]["spread"]),
            g_a_implied_by_the_tail=dict(ours=ph["g_a_implied_by_the_tail"], the_authors=PRINTED_55["g_a_implied_by_the_tail"],
                                         printed_axial_integral_not_computed_here=PRINTED_55["g_a"]),
            g_pinn_tail=dict(ours=ph["g_pinn_tail"], printed=PRINTED_55["g_pinn"], within_printed_range=bool(PRINTED_55["g_pinn"][0] - 0.005 <= ph["g_pinn_tail"] <= PRINTED_55["g_pinn"][1] + 0.005)),
            calibration_rule_on_our_inertia=dict(e=float(np.sqrt(2 * rd["isospin_inertia"] * DELTA / (3 * EF))), f_pi=float(EF / np.sqrt(2 * rd["isospin_inertia"] * DELTA / (3 * EF)))))

    # the box rung
    box_rows = {g: grid_state(g) for g in (55, 73, 91)}
    largest = None
    for g in (91, 73, 55):
        sel, row = box_rows[g]
        if sel and sel["from_stationary"] and "reads" in row:
            largest = (g, row)
            break
    if largest:
        g, row = largest
        inertia = row["reads"]["isospin_inertia"]
        e = float(np.sqrt(2 * inertia * DELTA / (3 * EF)))
        summary["own_calibration"] = dict(grid=g, state=row["name"], inertia=inertia, e=e, f_pi=EF / e)
        own = (e, EF / e)
        for gg in G.values():
            for r in gg["jobs"]:
                if "reads" in r:
                    r["reads"]["physical"]["own_box_rung"] = dict(physical(r["reads"]["isospin_inertia"], own), **couplings(r["reads"]["tail"]["mean"], own))
    (s55, r55), (s73, r73), (s91, r91) = box_rows[55], box_rows[73], box_rows[91]
    if s55 and s73 and "reads" in r55 and "reads" in r73:
        rise = r73["reads"]["isospin_inertia"] / r55["reads"]["isospin_inertia"] - 1
        reasons = []
        if not (s55["from_stationary"] and s73["from_stationary"]):
            reasons.append("a state of the pair is not stationary")
        if s55["branch"] != s73["branch"]:
            reasons.append("the two states sit on different branches")
        if abs(r55["reads"]["isospin_inertia"] - PRINTED_55["inertia"]) > 0.005:
            reasons.append("our 55^3 inertia misses the printed 37.44 beyond its printed digits")
        label = "NOT SCORED" if reasons else ("AGREES" if 0.008 <= rise <= 0.014 else "DIFFERS")
        summary["box_number"] = dict(inertia_55=r55["reads"]["isospin_inertia"], inertia_73=r73["reads"]["isospin_inertia"], rise_percent=100 * rise,
                                     recorded_by_the_reply=dict(inertia_73=37.85, rise_percent=1.1, still_outside_percent=0.3), window_percent=(0.8, 1.4),
                                     label=label, reasons=reasons)
        if s91 and "reads" in r91:
            summary["box_number"]["rise_73_to_91_percent"] = 100 * (r91["reads"]["isospin_inertia"] / r73["reads"]["isospin_inertia"] - 1)

    # the ladder words
    def ladder(grids):
        states = [grid_state(g) for g in grids]
        if any(s is None for s, _ in states):
            return dict(word="INCOMPLETE")
        if not states[-1][0]["from_stationary"]:
            return dict(word="NOT STATIONARY", detail="the finest grid holds no stationary state")
        if not all(s["from_stationary"] for s, _ in states):
            return dict(word="NOT STATIONARY", detail="a grid of the ladder holds no stationary state")
        branches = [s["branch"] for s, _ in states]
        if len(set(branches)) > 1:
            return dict(word="BRANCH-DEPENDENT", branches=branches)
        e = [r["energy"] for _, r in states]
        i = [r["reads"]["isospin_inertia"] for _, r in states]
        falling = abs(e[2] - e[1]) < abs(e[1] - e[0]) and abs(i[2] - i[1]) < abs(i[1] - i[0])
        return dict(word="CONVERGENT" if falling else "NOT CONVERGENT", branches=branches, energies=e, inertias=i,
                    energy_differences=[e[1] - e[0], e[2] - e[1]], inertia_differences=[i[1] - i[0], i[2] - i[1]])

    summary["ladders"] = dict(mesh=ladder((55, 65, 77)), box=ladder((55, 73, 91)))

    # the mesh ladder against the square of the spacing (both rules of the card are second order)
    mesh = [(g, grid_state(g)) for g in (45, 55, 65, 77)]
    if all(s and "reads" in r for _, (s, r) in mesh):
        h2 = np.array([G[str(g)]["spacing"] ** 2 for g, _ in mesh])
        fits = {}
        for key, get in (("isospin_inertia", lambda r: r["reads"]["isospin_inertia"]), ("energy", lambda r: r["energy"]),
                         ("tail_amplitude", lambda r: r["reads"]["tail"]["mean"]), ("r2_v_card", lambda r: r["reads"]["r2_v_card"]),
                         ("r2_s_card", lambda r: r["reads"]["r2_s_card"])):
            y = np.array([get(r) for _, (s, r) in mesh])
            slope, intercept = np.polyfit(h2[1:], y[1:], 1)
            fits[key] = dict(values=y.tolist(), slopes_between_neighbors=(np.diff(y) / np.diff(h2)).tolist(),
                             extrapolated_from_the_three_finest=float(intercept), slope=float(slope))
        summary["mesh_ladder_in_h2"] = dict(grids=[g for g, _ in mesh], spacing_squared=h2.tolist(), states=[s["name"] for _, (s, r) in mesh],
                                            lumps=[r["reads"]["lumps"] for _, (s, r) in mesh], fits=fits)

    # the gating of the definition of done: no read is labeled unless the 33^3 control passed
    gate = summary.get("control_33", {}).get("passed", False)
    summary["labels"] = dict(control_33_passed=bool(gate), note="the comparison words below are what the declared rules return; by the definition of done "
                             "they are not applied as labels while the 33^3 control has not passed")
    if not gate:
        if "box_number" in summary:
            summary["box_number"]["what_the_declared_rule_returns"] = summary["box_number"].pop("label")
            summary["box_number"]["label"] = "NOT LABELED: the 33^3 control did not pass"
        for lad in summary["ladders"].values():
            lad["what_the_declared_rule_returns"] = lad.pop("word")
            lad["word"] = "NOT LABELED: the 33^3 control did not pass"
    (DATA / "m5_39_r2_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (DATA / "m5_39_r2_reads.json").write_text(json.dumps(reads_out) + "\n")

    for grid, g in G.items():
        print(f"--- {grid}^3 (half-box {g['half_box']}, missing {g['missing']})")
        for r in g["jobs"]:
            if r["label"] == "NOT RUN":
                print(f"  {r['start']:10s} NOT RUN")
                continue
            extra = ""
            if "reads" in r:
                rd = r["reads"]
                extra = (f" | I {rd['isospin_inertia']:.4f} r2S {rd['r2_s_fm2']:.4f} r2V {rd['r2_v_fm2']:.4f} A {rd['tail']['mean']:.4f}+-{rd['tail']['spread']:.4f} "
                         f"lumps {rd['lumps']} mirror ds {max(m['max_s'] for m in rd['mirror']):.2e} dE {max(abs(m['energy_difference']) for m in rd['mirror']):.1e}")
            print(f"  {r['start']:10s} {r['label']:15s} E {r['energy']:.6f} res {r['residual']:.2e} it {r['iterations']:5d} drift {r['drift_last_tenth'] or 0:.1e} "
                  f"{r['branch']:17s} {[round(x, 4) for x in r['center_eigenvalues']]}{extra}")
        if "state_of_the_grid" in g:
            print("  state of the grid:", g["state_of_the_grid"], "spread", g.get("spread_within_branch"))
    for key in ("control_33", "comparison_55", "own_calibration", "box_number", "ladders", "mesh_ladder_in_h2"):
        if key in summary:
            print(key, json.dumps(summary[key], indent=1))


if __name__ == "__main__":
    main()
