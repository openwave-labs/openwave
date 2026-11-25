"""M5.41 R3 and R4: the labels of the routes, the spread between routes, the zero-spacing lines and
the calibration, from the route records (m5_39_relax.py --stops) and the reads (m5_41_reads.py).

The rule is the one frozen in m5_41_task_details.md, section 2, before any route ran:

  row 3   the pair that is read: the last stop and the latest earlier stop whose iteration count is
          at most a tenth of the last stop's (the counts of the iterates actually saved);
  row 4   FIXED: for each of the inertia, g_A e^2 and mu_V, the relative difference from the last
          stop is at most 1e-4 both for the stop of row 3 and for the stop of the 10000 target;
  row 6   MEETS THE RULE: the last stop under the gate, no lump on it, the three readouts FIXED;
  row 7   the spread on a mesh: (max - min) / 2 over the routes' last stops, relative to their mean;
          over all routes, over the routes that meet the rule, and with one start family left out;
          routes on another branch than the declared one are not combined;
  row 8   a straight line in h^2 through 55^3, 65^3, 77^3; the central value is the intercept on the
          means of the routes that meet the rule; the route error is the half range of the intercepts
          over every combination of one such route per mesh. Beside it: the two finest meshes alone,
          all four meshes, the 10000 stops, each start alone. The word calibration is used only when
          each of the three meshes has two routes that meet the rule.

Every number of the findings is in the JSON this script writes.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_41_calibration as CAL  # noqa: E402

DATA = HERE.parent / "data"
WORK = HERE.parent / "checkpoints" / "m5_41"
GRIDS = (45, 55, 65, 77)
FINE = (55, 65, 77)
KINDS = {45: ("iso", "chain"), 55: ("iso", "chain", "smooth", "author"), 65: ("iso", "chain", "smooth"), 77: ("iso", "chain", "smooth")}
FAMILY = dict(iso="linked", chain="linked", smooth="smooth", author="author")
TARGETS = (1000, 3000, 10000, 30000)
FIXED_BOUND = 1e-4
DECLARED = "ISOTROPIC CENTER"
READOUTS = ("inertia", "gA_e2", "muV")
EXTRA = ("rS2_fm2", "rV2_fm2", "energy", "muV_calibrated", "gA_own_e", "e_own")
# the trend of M5.40, section 13, recorded before the run (m5_41_task_details.md, section 5, item 5)
# mu_V there is the script's value at the inherited constants, not the value at the calibrated gap
EXPECTED_INTERCEPT = dict(inertia=35.92, gA_e2=16.28, muV=3.148, e=4.222, f_pi_MeV=93.2, gA=0.913)


def rel(a, b):
    return abs(a - b) / abs(b)


def half_range(values):
    values = np.asarray(values, dtype=float)
    return float((values.max() - values.min()) / 2.0)


def line(h2, y):
    """Least-squares straight line y = a + b h^2; returns (intercept, slope)."""
    A = np.stack([np.ones(len(h2)), np.asarray(h2, dtype=float)], axis=1)
    coef = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)[0]
    return float(coef[0]), float(coef[1])


def intercept_weights(h2):
    A = np.stack([np.ones(len(h2)), np.asarray(h2, dtype=float)], axis=1)
    return np.linalg.pinv(A)[0].tolist()


def quantities(read):
    """The numbers of one stop that enter spreads and lines."""
    r = read["readouts"]
    e_own = CAL.calibrate(r["inertia"])["e"]
    return dict(inertia=r["inertia"], gA_e2=r["gA_e2"], muV=r["muV"], rS2_fm2=r["rS2_fm2"], rV2_fm2=r["rV2_fm2"], energy=read["energy"],
                muV_calibrated=CAL.muV_calibrated(r["muV"], r["inertia"]), gA_own_e=r["gA_e2"] / e_own ** 2, e_own=e_own)


def load_route(grid, kind, reads_dir):
    name = f"r2_g{grid}_{kind}"
    path = DATA / f"m5_41_{name}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    route = dict(name=name, grid=grid, kind=kind, family=FAMILY[kind], spacing=rec.get("spacing"), record=rec, stops={})
    for s in rec.get("stops", []):
        p = reads_dir / f"m5_41_{name}_s{s['target']}.json"
        if not p.exists():
            raise FileNotFoundError(f"no reads for {p.name}: run m5_41_reads.py")
        read = json.loads(p.read_text())
        if rel(read["energy"], s["energy"]) > 1e-12 or read["stored"].get("iteration") != s["iteration"]:
            raise RuntimeError(f"the reads of {p.name} are not of the state the route recorded")
        route["stops"][s["target"]] = dict(record=s, read=read, q=quantities(read))
    return route


def label_route(route, pair_mode="frozen"):
    """The label of section 2, row 6, with every number it rests on.

    pair_mode: "frozen" is section 2 row 3 as written (the latest earlier saved stop whose iteration
    is at most a tenth of the last stop's, whether or not it was under the gate). Two other readings
    are computed beside it, after the audit, and decide nothing: "under_the_gate" (the same, among the
    stops that were under the gate) and "by_target" (the stop of the target a factor 10 below the
    last one, whatever iteration it was saved at)."""
    rec, stops = route["record"], route["stops"]
    out = dict(name=route["name"], grid=route["grid"], kind=route["kind"], family=route["family"],
               route_end=rec.get("route_end"), iterations=rec.get("iterations"), evaluations=rec.get("evaluations"),
               restarts=rec.get("restarts", []), resumed_at=rec.get("resumed_at", []), wall_hits=rec.get("wall_hits"),
               threads=rec.get("threads"), wall_time_h=rec.get("wall_time_s", 0.0) / 3600.0,
               start=dict(kind=rec["start"]["kind"], energy=rec["start"]["energy"], control=rec["start"].get("start_control"),
                          gradient_check=rec["start"].get("gradient_check")),
               energy_drift_last_tenth=rec.get("energy_drift_last_tenth"),
               iterates_under_the_gate=rec.get("iterates_under_the_gate"), targets_not_reached=rec.get("targets_not_reached", []))
    table = []
    for t in TARGETS:
        if t not in stops:
            continue
        s, r, q = stops[t]["record"], stops[t]["read"], stops[t]["q"]
        table.append(dict(target=t, iteration=s["iteration"], kind=s["kind"], gate=r["gate"], gate_met=r["gate_met"],
                          residual_as_stored=r["residual_as_stored"], winding_length_range=r["winding_length_range"],
                          terms=r["terms"], center_eigenvalues=r["center_eigenvalues"], branch=r["branch"],
                          lumps=r["record_lumps"]["count"], largest_record_excess=r["record_lumps"]["largest_excess"],
                          degree=r["degree"], hemisphere_margin=r["hemisphere_margin_four_triangulations"],
                          muV_rigid=r["readouts"]["muV_rigid"], gA_e2_N_term=r["readouts"]["gA_e2_N_term"],
                          gA_e2_wedge=r["readouts"]["gA_e2_wedge"], ended_by=s.get("ended_by"), **q))
    out["stops"] = table
    fails = []
    last_target = TARGETS[-1]
    if last_target not in stops:
        out.update(label="NO LAST STOP", fails=["the route has no last stop"], meets=False, branch=None)
        return out
    last = [row for row in table if row["target"] == last_target][0]
    out["branch"] = last["branch"]
    if not last["gate_met"]:
        fails.append("GATE NOT MET")
    if last["lumps"] > 0:
        fails.append("LUMP")
    earlier = [row for row in table if row["target"] != last_target and row["iteration"] <= last["iteration"] / 10.0]
    if pair_mode == "under_the_gate":
        earlier = [row for row in earlier if row["gate_met"]]
    elif pair_mode == "by_target":
        earlier = [row for row in table if row["target"] * 10 == last_target]
    pair = max(earlier, key=lambda row: row["iteration"]) if earlier else None
    ten = [row for row in table if row["target"] == TARGETS[2]]
    ten = ten[0] if ten else None
    fixed = {}
    for key in READOUTS:
        d_pair = rel(pair[key], last[key]) if pair else None
        d_ten = rel(ten[key], last[key]) if ten else None
        ok = d_pair is not None and d_ten is not None and d_pair <= FIXED_BOUND and d_ten <= FIXED_BOUND
        fixed[key] = dict(drift_from_the_pair_stop=d_pair, drift_from_the_10000_stop=d_ten, fixed=bool(ok))
    out["pair_stop"] = dict(target=pair["target"], iteration=pair["iteration"], kind=pair["kind"], gate=pair["gate"], gate_met=pair["gate_met"]) if pair else None
    out["last_stop"] = dict(target=last["target"], iteration=last["iteration"], kind=last["kind"])
    out["fixed"] = fixed
    # the same drifts for the quantities that carry no label
    out["unlabeled_drifts"] = {key: dict(from_the_pair_stop=rel(pair[key], last[key]) if pair else None,
                                         from_the_10000_stop=rel(ten[key], last[key]) if ten else None)
                               for key in ("rS2_fm2", "rV2_fm2", "energy")}
    not_fixed = [k for k in READOUTS if not fixed[k]["fixed"]]
    if pair is None:
        fails.append("NO PAIR STOP")
    elif not_fixed:
        fails.append("NOT FIXED: " + ", ".join(not_fixed))
    out["meets"] = not fails
    out["fails"] = fails
    out["label"] = "MEETS THE RULE" if not fails else "; ".join(fails)
    return out


def spread_block(rows, keys):
    out = {}
    for key in keys:
        vals = [r[key] for r in rows]
        if len(vals) < 2:
            out[key] = dict(routes=len(vals), values=vals)
            continue
        mean = float(np.mean(vals))
        out[key] = dict(routes=len(vals), mean=mean, half_range=half_range(vals), relative_half_range=half_range(vals) / abs(mean),
                        min=float(min(vals)), max=float(max(vals)))
    return out


def mesh_block(grid, labels):
    """The spread between the routes of one mesh, on their last stops."""
    routes = [lab for lab in labels if lab["grid"] == grid and lab.get("last_stop")]
    last = {lab["name"]: [row for row in lab["stops"] if row["target"] == TARGETS[-1]][0] for lab in routes}
    keys = READOUTS + EXTRA
    on_branch = [lab for lab in routes if lab["branch"] == DECLARED]
    meeting = [lab for lab in on_branch if lab["meets"]]
    block = dict(grid=grid, routes=[lab["name"] for lab in routes], on_the_declared_branch=[lab["name"] for lab in on_branch],
                 on_another_branch={lab["name"]: lab["branch"] for lab in routes if lab["branch"] != DECLARED},
                 meeting_the_rule=[lab["name"] for lab in meeting],
                 all_routes=spread_block([last[lab["name"]] for lab in on_branch], keys),
                 routes_meeting_the_rule=spread_block([last[lab["name"]] for lab in meeting], keys), one_family_left_out={})
    for fam in sorted({lab["family"] for lab in on_branch}):
        rest = [lab for lab in on_branch if lab["family"] != fam]
        block["one_family_left_out"][fam] = dict(routes=[lab["name"] for lab in rest], spread=spread_block([last[lab["name"]] for lab in rest], READOUTS))
    energies = {lab["name"]: last[lab["name"]]["energy"] for lab in routes}
    if energies:
        low = min(energies.values())
        block["energy"] = dict(lowest=low, lowest_route=min(energies, key=energies.get),
                               above_the_lowest={k: v - low for k, v in energies.items()},
                               more_than_5e_4_above=[k for k, v in energies.items() if v - low > 5e-4])
    return block


def calibration_of(inertia0, gA_e2_0):
    c = CAL.calibrate(inertia0, gA_e2_0)
    return dict(e=c["e"], f_pi_MeV=c["f_pi_MeV"], action_energy_unit_MeV=c["action_energy_unit_MeV"], gA=c["gA"])


def lines_block(labels, pick, grids=None, target=None):
    """Lines in h^2 on the routes selected by pick(label); central values, the route range, the calibration."""
    grids = FINE if grids is None else grids
    target = TARGETS[-1] if target is None else target
    per_mesh = {}
    for g in grids:
        rows = []
        for lab in labels:
            if lab["grid"] != g or not pick(lab):
                continue
            row = [r for r in lab["stops"] if r["target"] == target]
            if row:
                rows.append((lab["name"], lab["spacing"], row[0]))
        per_mesh[g] = rows
    if any(len(per_mesh[g]) == 0 for g in grids):
        return dict(available=False, routes={g: [n for n, _, _ in per_mesh[g]] for g in grids})
    h2 = [per_mesh[g][0][1] ** 2 for g in grids]
    keys = ("inertia", "gA_e2", "muV_calibrated", "gA_own_e", "rS2_fm2", "rV2_fm2", "energy", "muV")
    out = dict(available=True, grids=list(grids), h2=h2, intercept_weights=intercept_weights(h2),
               routes={g: [n for n, _, _ in per_mesh[g]] for g in grids}, central={}, route_range={})
    means = {k: [float(np.mean([row[k] for _, _, row in per_mesh[g]])) for g in grids] for k in keys}
    for k in keys:
        a, b = line(h2, means[k])
        fit = a + b * np.asarray(h2)
        out["central"][k] = dict(intercept=a, slope=b, mesh_means=means[k], largest_misfit=float(np.max(np.abs(fit - means[k]))),
                                 intercept_minus_finest=a - means[k][-1])
    out["central"]["calibration"] = calibration_of(out["central"]["inertia"]["intercept"], out["central"]["gA_e2"]["intercept"])
    out["central"]["calibration"]["muV"] = out["central"]["muV_calibrated"]["intercept"]
    out["central"]["calibration"]["gA_from_the_line_of_gA_at_own_e"] = out["central"]["gA_own_e"]["intercept"]
    out["central"]["calibration"]["muV_rigid"] = CAL.rigid_muV()
    combos = list(itertools.product(*[per_mesh[g] for g in grids]))
    per_combo = []
    for combo in combos:
        vals = {k: line(h2, [row[k] for _, _, row in combo])[0] for k in keys}
        cal = calibration_of(vals["inertia"], vals["gA_e2"])
        per_combo.append(dict(routes=[n for n, _, _ in combo], intercepts=vals, calibration=cal))
    out["combinations"] = len(combos)
    for k in keys:
        v = [c["intercepts"][k] for c in per_combo]
        out["route_range"][k] = dict(half_range=half_range(v), min=min(v), max=max(v),
                                     relative_half_range=half_range(v) / abs(out["central"][k]["intercept"]))
    for k in ("e", "f_pi_MeV", "action_energy_unit_MeV", "gA"):
        v = [c["calibration"][k] for c in per_combo]
        out["route_range"][k] = dict(half_range=half_range(v), min=min(v), max=max(v),
                                     relative_half_range=half_range(v) / abs(out["central"]["calibration"][k]))
    out["per_combination"] = per_combo
    return out


def brief(block):
    """The intercepts of a lines block, without its combinations."""
    if not block.get("available"):
        return dict(available=False, routes=block.get("routes"))
    return dict(available=True, grids=block["grids"], routes=block["routes"],
                intercepts={k: v["intercept"] for k, v in block["central"].items() if k != "calibration"},
                calibration=block["central"]["calibration"],
                route_half_range={k: v["half_range"] for k, v in block["route_range"].items()})


def expectations(labels, meshes, lines):
    """The five lines recorded before the run (section 5), each as met or not with its numbers."""
    out = {}
    prior = {}
    for lab in labels:
        p = DATA / f"m5_39_{lab['name']}.json"
        if p.exists():
            prior[lab["name"]] = json.loads(p.read_text()).get("energy")
    item1 = {}
    for lab in labels:
        if not lab.get("last_stop"):
            continue
        end = [r for r in lab["stops"] if r["target"] == TARGETS[-1]][0]["energy"]
        ref = prior.get(lab["name"])
        item1[lab["name"]] = dict(end_energy=end, end_state_of_m5_39=ref, below=(end < ref) if ref is not None else None)
    e55 = meshes.get(55, {}).get("energy")
    all55 = None
    if e55:
        ends = [item1[n]["end_energy"] for n in meshes[55]["routes"]]
        all55 = dict(ends=ends, largest_distance_from_75_5617=float(max(abs(e - 75.5617) for e in ends)),
                     within_2e_4=bool(all(abs(e - 75.5617) <= 2e-4 for e in ends)))
    out["1_energy"] = dict(routes=item1, every_route_below_its_m5_39_end=bool(all(v["below"] for v in item1.values() if v["below"] is not None)),
                           the_four_55_routes=all55, routes_more_than_5e_4_above_the_lowest={g: m["energy"]["more_than_5e_4_above"] for g, m in meshes.items() if "energy" in m})
    lower = {}
    for g, m in meshes.items():
        if "energy" not in m:
            continue
        for p in sorted(DATA.glob(f"m5_39_r2_g{g}_*.json")):
            rec = json.loads(p.read_text())
            e = rec.get("energy")
            if e is not None and e == e and rec.get("half_box") == 6.075 and e < m["energy"]["lowest"]:
                lower.setdefault(str(g), []).append(dict(state=p.stem, energy=e, below_the_lowest_last_stop=m["energy"]["lowest"] - e, label=rec.get("label"),
                                                         residual=rec.get("residual"), center_eigenvalues=rec.get("center_eigenvalues")))
    out["states_of_m5_39_below_the_lowest_last_stop"] = lower
    out["2_gate_at_the_last_stop"] = {lab["name"]: [r for r in lab["stops"] if r["target"] == TARGETS[-1]][0]["gate_met"] for lab in labels if lab.get("last_stop")}
    out["3_within_a_route"] = {lab["name"]: {k: lab["fixed"][k]["fixed"] for k in READOUTS} for lab in labels if lab.get("fixed")}
    out["4_across_routes"] = {g: {k: m["all_routes"].get(k, {}).get("relative_half_range") for k in READOUTS} for g, m in meshes.items()}
    main = lines.get("routes_meeting_the_rule")
    if main and main.get("available"):
        vals = dict(inertia=main["central"]["inertia"]["intercept"], gA_e2=main["central"]["gA_e2"]["intercept"],
                    muV=main["central"]["muV"]["intercept"], e=main["central"]["calibration"]["e"],
                    f_pi_MeV=main["central"]["calibration"]["f_pi_MeV"], gA=main["central"]["calibration"]["gA"])
        out["5_zero_spacing"] = {k: dict(value=v, expected=EXPECTED_INTERCEPT[k], relative_difference=rel(v, EXPECTED_INTERCEPT[k]),
                                         within_half_a_percent=bool(rel(v, EXPECTED_INTERCEPT[k]) <= 5e-3)) for k, v in vals.items()}
    return out


def main():
    global DATA, GRIDS, FINE, TARGETS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads-dir", type=Path, default=WORK / "reads")
    ap.add_argument("--out", type=Path, default=DATA / "m5_41_summary.json")
    ap.add_argument("--reads-out", type=Path, default=DATA / "m5_41_r3_reads.json")
    ap.add_argument("--data-dir", type=Path, help="a test of the pipeline on another folder of route records")
    ap.add_argument("--grids", type=int, nargs=4, help="with --data-dir: the four meshes, the last three being the fitted ones")
    ap.add_argument("--targets", type=int, nargs=4, help="with --data-dir: the four stop targets")
    a = ap.parse_args()
    if a.data_dir:
        DATA = a.data_dir
        a.out, a.reads_out = DATA / "m5_41_summary.json", DATA / "m5_41_r3_reads.json"
        if a.grids:
            GRIDS, FINE = tuple(a.grids), tuple(a.grids[1:])
        if a.targets:
            TARGETS = tuple(a.targets)

    kinds = {g: KINDS.get(g, ("iso", "chain", "smooth")) for g in GRIDS}
    routes = [load_route(g, k, a.reads_dir) for g in GRIDS for k in kinds[g]]
    missing = [f"r2_g{g}_{k}" for g in GRIDS for k in kinds[g] if not (DATA / f"m5_41_r2_g{g}_{k}.json").exists()]
    routes = [r for r in routes if r is not None]
    labels = []
    for r in routes:
        lab = label_route(r)
        lab["spacing"] = r["spacing"]
        labels.append(lab)
    meshes = {g: mesh_block(g, labels) for g in GRIDS if any(lab["grid"] == g for lab in labels)}

    def on_branch(lab):
        return lab.get("branch") == DECLARED

    def meeting(lab):
        return on_branch(lab) and lab["meets"]

    def no_lump(lab):
        last = [r for r in lab["stops"] if r["target"] == TARGETS[-1]]
        return on_branch(lab) and bool(last) and last[0]["lumps"] == 0

    lines = dict(
        routes_meeting_the_rule=lines_block(labels, meeting),
        # three variants added after the audit, each a subset chosen after the results were seen
        added_after_the_audit=dict(
            meeting_without_the_authors_field=brief(lines_block(labels, lambda lab: meeting(lab) and lab["kind"] != "author")),
            meeting_with_chain_alone_on_the_coarsest_fitted_mesh=brief(lines_block(labels, lambda lab: meeting(lab) and (lab["grid"] != FINE[0] or lab["kind"] == "chain"))),
            all_routes_without_a_lump=brief(lines_block(labels, no_lump))),
        all_routes_on_the_declared_branch=lines_block(labels, on_branch),
        fit_form=dict(two_finest_meshes=brief(lines_block(labels, meeting, grids=FINE[1:])),
                      all_four_meshes=brief(lines_block(labels, meeting, grids=GRIDS)),
                      all_four_meshes_all_routes=brief(lines_block(labels, on_branch, grids=GRIDS)),
                      two_finest_meshes_all_routes=brief(lines_block(labels, on_branch, grids=FINE[1:]))),
        from_the_10000_stops=brief(lines_block(labels, meeting, target=TARGETS[2])),
        from_the_10000_stops_all_routes=brief(lines_block(labels, on_branch, target=TARGETS[2])),
        each_start_alone={k: brief(lines_block(labels, lambda lab, k=k: on_branch(lab) and lab["kind"] == k)) for k in ("iso", "chain", "smooth")},
        each_family_alone={f: brief(lines_block(labels, lambda lab, f=f: on_branch(lab) and lab["family"] == f)) for f in ("linked", "smooth")},
    )
    # the two other readings of the pair stop (section 2 row 3), computed after the audit; they decide nothing
    other = {}
    for mode in ("under_the_gate", "by_target"):
        alt = []
        for r in routes:
            lab = label_route(r, pair_mode=mode)
            lab["spacing"] = r["spacing"]
            alt.append(lab)
        per_mesh = {g: [lab["name"] for lab in alt if lab["grid"] == g and lab.get("branch") == DECLARED and lab["meets"]] for g in GRIDS}
        blk = lines_block(alt, lambda lab: lab.get("branch") == DECLARED and lab["meets"])
        other[mode] = dict(labels={lab["name"]: dict(label=lab["label"], pair_stop=lab.get("pair_stop"),
                                                     fixed={k: v for k, v in (lab.get("fixed") or {}).items()}) for lab in alt},
                           meeting_the_rule=per_mesh, two_per_fitted_mesh={g: len(per_mesh[g]) >= 2 for g in FINE},
                           word="calibration" if all(len(per_mesh[g]) >= 2 for g in FINE) else "trend",
                           lines=dict(brief(blk), combinations=blk.get("combinations")))
    # labels that hinge on a value within 20 percent of a bound
    near = []
    for lab in labels:
        for k, v in (lab.get("fixed") or {}).items():
            for which in ("drift_from_the_pair_stop", "drift_from_the_10000_stop"):
                d = v[which]
                if d is not None and 0.8 * FIXED_BOUND <= d <= 1.2 * FIXED_BOUND:
                    near.append(dict(route=lab["name"], quantity=k, which=which, value=d, bound=FIXED_BOUND, label=lab["label"]))
        last = [r for r in lab["stops"] if r["target"] == TARGETS[-1]]
        if last and 0.8e-3 <= last[0]["gate"] <= 1.2e-3:
            near.append(dict(route=lab["name"], quantity="gate at the last stop", value=last[0]["gate"], bound=1e-3, label=lab["label"]))
    two_per_mesh = {g: len(meshes[g]["meeting_the_rule"]) >= 2 for g in FINE if g in meshes}
    word = "calibration" if len(two_per_mesh) == 3 and all(two_per_mesh.values()) else "trend"
    summary = dict(
        task="M5.41", rule=dict(gate=1e-3, fixed_bound=FIXED_BOUND, stop_targets=list(TARGETS), declared_branch=DECLARED,
                                lump_threshold=0.2, readouts=list(READOUTS)),
        constants=dict(delta_MeV=CAL.DELTA, frequency_unit_MeV=CAL.FREQUENCY_UNIT, m_p_MeV=CAL.M_P),
        routes_missing=missing, labels=labels, meshes=meshes, lines=lines,
        word=word, two_routes_meeting_the_rule_per_mesh=two_per_mesh,
        failing_clause_for_the_word=None if word == "calibration" else {g: dict(meeting=meshes[g]["meeting_the_rule"], labels={lab["name"]: lab["label"] for lab in labels if lab["grid"] == g}) for g in FINE if g in meshes and not two_per_mesh[g]},
    )
    summary["other_readings_of_the_pair_stop"] = other
    summary["values_within_20_percent_of_a_bound"] = near
    summary["expectations"] = expectations(labels, meshes, lines)
    a.out.write_text(json.dumps(summary, indent=1) + "\n")
    reads = {}
    for r in routes:
        for t, s in r["stops"].items():
            reads[f"{r['name']}_s{t}"] = s["read"]
    a.reads_out.write_text(json.dumps(reads, indent=1) + "\n")

    # the terminal tables
    print(f"routes read: {len(labels)}; missing: {missing}")
    for lab in labels:
        print(f"\n{lab['name']}: {lab['label']}  (end: {lab['route_end']}, iterations {lab['iterations']}, restarts {len(lab['restarts'])}, "
              f"resumes {len(lab['resumed_at'])}, branch {lab.get('branch')}, drift of E over the last tenth {lab['energy_drift_last_tenth']})")
        for row in lab["stops"]:
            print(f"  s{row['target']:<6} it {row['iteration']:<6} {row['kind']:<5} gate {row['gate']:.2e} lumps {row['lumps']} E {row['energy']:.7f} "
                  f"I {row['inertia']:.6f} gAe2 {row['gA_e2']:.6f} muV {row['muV']:.6f} rS2 {row['rS2_fm2']:.6f} rV2 {row['rV2_fm2']:.6f} "
                  f"center {[round(x, 4) for x in row['center_eigenvalues']]}")
        if lab.get("fixed"):
            print("  fixed: " + "; ".join(f"{k} {v['drift_from_the_pair_stop']:.1e} / {v['drift_from_the_10000_stop']:.1e} {'FIXED' if v['fixed'] else 'NOT FIXED'}"
                                          for k, v in lab["fixed"].items() if v["drift_from_the_pair_stop"] is not None and v["drift_from_the_10000_stop"] is not None)
                  + f"  (pair stop {lab['pair_stop']})")
    for g, m in meshes.items():
        print(f"\nmesh {g}: meeting {m['meeting_the_rule']}; other branch {m['on_another_branch']}")
        for k in READOUTS:
            s_all, s_meet = m["all_routes"].get(k, {}), m["routes_meeting_the_rule"].get(k, {})
            print(f"  {k}: all routes {s_all.get('relative_half_range')}  meeting {s_meet.get('relative_half_range')}")
        if "energy" in m:
            print(f"  energy above the lowest: {m['energy']['above_the_lowest']}")
    for tag in ("routes_meeting_the_rule", "all_routes_on_the_declared_branch"):
        blk = lines[tag]
        print(f"\nlines, {tag}: available {blk.get('available')}")
        if blk.get("available"):
            print(f"  weights {[round(w, 3) for w in blk['intercept_weights']]}, combinations {blk['combinations']}")
            for k in ("inertia", "gA_e2", "muV_calibrated", "gA_own_e"):
                print(f"  {k}: intercept {blk['central'][k]['intercept']:.6f}, route half range {blk['route_range'][k]['half_range']:.2e}, means {[round(x, 6) for x in blk['central'][k]['mesh_means']]}")
            c = blk["central"]["calibration"]
            rr = blk["route_range"]
            print(f"  route half ranges in brackets: e {c['e']:.5f} [{rr['e']['half_range']:.1e}]; f_pi {c['f_pi_MeV']:.4f} [{rr['f_pi_MeV']['half_range']:.1e}]; "
                  f"unit {c['action_energy_unit_MeV']:.5f} [{rr['action_energy_unit_MeV']['half_range']:.1e}]; gA {c['gA']:.5f} [{rr['gA']['half_range']:.1e}]; "
                  f"muV {c['muV']:.5f}; rigid {c['muV_rigid']:.5f}")
    print(f"\nthe word: {word}; two meeting routes per mesh: {two_per_mesh}")


if __name__ == "__main__":
    main()
