"""M5.40: the four rungs gathered into one summary, from the files of the run.

    python3 m5_40_summary.py --archive <hashed copy of the card's code folder> --iso <folder of the isolated run>

Inputs: m5_40_r1_payload.json (R1), m5_40_r2_reads_author_g*.json (the readers of M5.39 on the author's
fields), the job files of M5.39, the two continuation jobs m5_40_r3_g*_cont.json with their states,
m5_40_r4_readouts.json, and, in the isolated folder, the printed outputs of the author's scripts run
unmodified on the author's fields (f33/, f55/: magnetic.jsonl, zero_q2.json, static_preflight.json).

The words. REPRODUCED: the read of the author's field rounds to the printed digits. DIFFERS: it does
not (the number is given). The expectations of R3 were recorded in the plan before the run and are
only evaluated here.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_39_cardc as C  # noqa: E402
import m5_39_reads as R  # noqa: E402
import m5_40_readouts as M  # noqa: E402

DATA = HERE.parent / "data"
WORK = HERE.parent / "checkpoints" / "m5_40"
HBARC, LENGTH = 197.3269804, 0.5014
SCALE = (4.259, 92.4)
PRINTED = {
    33: dict(center=(0.333, 0.376, 0.394), inertia=34.33, gap_MeV=311.9, rV2_fm2=0.4519, rS2_fm2=0.2303, muV=3.004, muV_rigid=3.008),
    55: dict(inertia=37.44, tail_fm2=0.518, gA=0.904),
}
PRINTED_AXIAL_33 = dict(gA=0.957, N_term=0.457, wedge=0.500)
BOX = {33: 3.6, 55: 6.075}
M539 = {33: ["r2_g33_iso", "r2_g33_chain", "r2_g33_linked_ext", "r2_g33_linked_ext2"],
        55: ["r2_g55_linked", "r2_g55_smooth", "r2_g55_iso", "r2_g55_chain", "r2_g55_reset"]}
REFERENCE = {33: "r2_g33_iso", 55: "r2_g55_smooth"}
MESH = [("m5_39_r2_g45_chain", 45), ("m5_39_r2_g55_smooth", 55), ("m5_39_r2_g65_chain", 65), ("m5_39_r2_g77_smooth", 77)]
BOXES = [("m5_39_r2_g55_smooth", 55), ("m5_39_r2_g73_cont", 73), ("m5_39_r2_g91_cont", 91)]
EXPECTED = dict(energy_fall=(1e-4, 2e-3), center_move=0.01, inertia_move_percent=0.1, valley_floor_fall=1e-5)


def load(path):
    return json.loads(Path(path).read_text())


def word(value, printed, decimals):
    return "REPRODUCED" if round(float(value), decimals) == round(float(printed), decimals) else "DIFFERS"


def cell_winding_density(q, h):
    n = q.shape[0]
    out = np.zeros((n - 1, n - 1, n - 1))
    weight = h ** 3 / (192.0 * np.pi ** 2)
    for r in range(4):
        for o in range(6):
            sl = M.simplex_slices(n, r, o)
            v = [q[s] for s in sl]
            orient = np.linalg.det(np.array([C.OFFSETS[r, o, m + 1] - C.OFFSETS[r, o, m] for m in range(3)], dtype=float))
            D = [(v[m + 1] - v[m]) / h for m in range(3)]
            S = sum(v)
            for i in range(4):
                P = C.SMALL * S + (C.LARGE - C.SMALL) * v[i]
                out += -orient * weight * np.linalg.det(np.stack([P] + D, axis=-2)) / np.sum(P * P, axis=-1) ** 2
    return out / h ** 3


def profiles(path, width=0.25):
    with np.load(path, allow_pickle=False) as d:
        state, x = d["state"], d["coords"]
    h = float(x[1, 0, 0, 0] - x[0, 0, 0, 0])
    q, f = M.unpack(state)
    s = np.swapaxes(f, -1, -2) @ f
    lam = np.linalg.eigvalsh(s)
    r = np.linalg.norm(x, axis=-1)
    dens = cell_winding_density(q, h)
    rc = np.linalg.norm(R.midpoint(x, h)[0], axis=-1)
    edges = np.arange(0.0, float(x[-1, 0, 0, 0]) + 1e-9, width)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel, selc = (r >= lo) & (r < hi), (rc >= lo) & (rc < hi)
        if sel.sum() == 0 or selc.sum() == 0:
            continue
        rows.append(dict(r_lo=float(lo), r_hi=float(hi), q0=float(q[sel, 0].mean()), trace_s=float(np.trace(s, axis1=-2, axis2=-1)[sel].mean()),
                         eig_low=float(lam[sel, 0].mean()), eig_mid=float(lam[sel, 1].mean()), eig_high=float(lam[sel, 2].mean()),
                         winding_density=float(dens[selc].mean())))
    return rows, q, s


def compare_profiles(a, b):
    out = {}
    for key in ("q0", "trace_s", "eig_low", "eig_mid", "eig_high", "winding_density"):
        va, vb = np.array([r[key] for r in a]), np.array([r[key] for r in b])
        out[key] = dict(largest_difference=float(np.abs(va - vb).max()), largest_value=float(np.abs(va).max()))
    return out


def r2_block(grid, archive):
    tag = f"g{grid}"
    rd = load(DATA / f"m5_40_r2_reads_author_{tag}.json")
    p = PRINTED[grid]
    inertia = rd["inertia_audit"]["actions"]["left"]["isospin_mean"]
    gap = 1.5 / inertia * SCALE[0] ** 3 * SCALE[1]
    rows = [dict(quantity="isospin inertia", printed=p["inertia"], read=inertia, word=word(inertia, p["inertia"], 2))]
    if grid == 33:
        eig = rd["center_eigenvalues"]
        rows.append(dict(quantity="center eigenvalues of the record", printed=list(p["center"]), read=eig,
                         word="REPRODUCED" if all(round(a, 3) == b for a, b in zip(eig, p["center"])) else "DIFFERS"))
        rows.append(dict(quantity="rotor gap (MeV)", printed=p["gap_MeV"], read=gap, word=word(gap, p["gap_MeV"], 1)))
        rv = rd["isospin_inertia_ours"]["isovector_r2"] * LENGTH ** 2
        rows.append(dict(quantity="isovector radius squared (fm^2), reader of M5.39", printed=p["rV2_fm2"], read=rv, word=word(rv, p["rV2_fm2"], 4)))
        rs = rd["winding"]["isoscalar_r2"] * LENGTH ** 2
        rows.append(dict(quantity="isoscalar radius squared (fm^2), reader of M5.39 (Gauss rule, first triangulation, about the origin)",
                         printed=p["rS2_fm2"], read=rs, word=word(rs, p["rS2_fm2"], 4)))
    else:
        tail = rd["tail"]
        sel = [t for t in tail if 0.5 * BOX[grid] <= t["r_lo"] and t["r_hi"] <= 0.9 * BOX[grid] + 1e-9]
        mean = float(np.mean([t["amplitude"] for t in sel])) * LENGTH ** 2
        last = [t["amplitude"] * LENGTH ** 2 for t in tail[-4:]]
        rows.append(dict(quantity="tail amplitude (fm^2), window of M5.39 (half the box to 0.9 of it)", printed=p["tail_fm2"], read=mean,
                         word=word(mean, p["tail_fm2"], 3), last_four_shells=last, last_four_shells_word=word(np.mean(last), p["tail_fm2"], 3)))
    mine = dict(name=f"the author's {grid}^3 field", energy=rd["energy"], terms=rd["terms"], residual=rd["residual"], center_eigenvalues=rd["center_eigenvalues"],
                lumps=rd["record_lumps"]["count"], largest_eigenvalue_excess=rd["record_lumps"]["largest_excess"], inertia=inertia,
                isospin_eigenvalues=rd["inertia_audit"]["actions"]["left"]["isospin_eigenvalues"],
                spin_inertia=rd["inertia_audit"]["actions"]["left"]["spin_mean"], mixed_inertia=rd["inertia_audit"]["actions"]["left"]["mixed_mean"],
                ours_against_the_audit=rd["isospin_block_ours_against_the_audit"], winding_total=rd["winding"]["total"],
                mirror=[dict(axis=m["axis"], max_q=m["max_q"], max_s=m["max_s"], energy_difference=m["energy_difference"]) for m in rd["mirror"]])
    states = []
    cache = load(DATA / "m5_39_r2_reads.json")
    for name in M539[grid]:
        j = load(DATA / f"m5_39_{name}.json")
        row = dict(name=name, label=j["label"], energy=j["energy"], terms=j["terms"], residual=j["residual"], iterations=j["iterations"],
                   center_eigenvalues=j["center_eigenvalues"], energy_minus_the_authors=j["energy"] - rd["energy"],
                   term_differences=(np.array(j["terms"]) - np.array(rd["terms"])).tolist())
        if name in cache:
            row["inertia"] = cache[name]["inertia_audit"]["actions"]["left"]["isospin_mean"]
            row["lumps"] = cache[name]["record_lumps"]["count"]
        states.append(row)
    pa, qa, sa = profiles(DATA / f"m5_40_author_{tag}.npz")
    pb, qb, sb = profiles(DATA / f"m5_39_{REFERENCE[grid]}.npz")
    node = dict(reference=REFERENCE[grid], max_q=float(np.abs(qa - qb).max()), rms_q=float(np.sqrt(np.mean(np.sum((qa - qb) ** 2, -1)))),
                max_s=float(np.abs(sa - sb).max()), rms_s=float(np.sqrt(np.mean(np.sum((sa - sb) ** 2, (-2, -1))))))
    # the same distance to the eight images of the author's field under the coordinate mirrors (an image
    # of a state is a state of the same energy; rotations of the lattice are not tried)
    with np.load(DATA / f"m5_40_author_{tag}.npz", allow_pickle=False) as d:
        base = d["state"]
    images = []
    for axes in ((), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)):
        img = base
        for ax in axes:
            img = C.mirror(img, ax)
        qi, fi = M.unpack(img)
        si = np.swapaxes(fi, -1, -2) @ fi
        images.append(dict(mirrors=list(axes), max_q=float(np.abs(qi - qb).max()), max_s=float(np.abs(si - sb).max()),
                           rms_s=float(np.sqrt(np.mean(np.sum((si - sb) ** 2, (-2, -1)))))))
    node["mirror_images_of_the_authors_field"] = images
    node["smallest_rms_s_over_the_images"] = min(i["rms_s"] for i in images)
    return dict(printed_rows=rows, the_authors_field=mine, states_of_m5_39=states, term_names=C.TERM_NAMES,
                profiles=dict(the_authors=pa, reference=pb, reference_name=REFERENCE[grid], shell_differences=compare_profiles(pa, pb)),
                node_by_node_diagnostic=node)


def state_reads(path, archive, tag):
    cache = WORK / f"reads_{tag}.json"
    if cache.exists():
        return load(cache)
    rd = R.reads(path, archive, mirror=False)
    ro = M.readouts(path)
    out = dict(energy=rd["energy"], residual=rd["residual"], center_eigenvalues=rd["center_eigenvalues"],
               inertia=rd["inertia_audit"]["actions"]["left"]["isospin_mean"] if "actions" in rd.get("inertia_audit", {}) else rd["isospin_inertia_ours"]["mean"],
               lumps=rd["record_lumps"]["count"], largest_eigenvalue_excess=rd["record_lumps"]["largest_excess"],
               rS2_fm2=ro["rS2_fm2"], rV2_fm2=ro["rV2_fm2"], muV=ro["muV"], muV_rigid=ro["muV_rigid"], e2_gA=ro["axial"]["e2_gA"],
               e2_gA_N_term=ro["axial"]["e2_gA_N_term"], e2_gA_wedge=ro["axial"]["e2_gA_wedge"], frozen_rule_e=ro["frozen_rule_e"],
               winding_centroid=ro["winding_centroid"], tail=[dict(r_lo=t["r_lo"], amplitude=t["amplitude"]) for t in rd["tail"]])
    cache.write_text(json.dumps(out) + "\n")
    return out


def r3_block(grid, archive, r1, readouts):
    name = f"r3_g{grid}_cont"
    path = DATA / f"m5_40_{name}.json"
    if not path.exists():
        return dict(status="NOT FINISHED")
    j = load(path)
    recorded = r1["K1"]["recorded"]
    hist = j["history"]
    e0 = j["start"]["energy"]
    it = np.array([r["iteration"] for r in hist])
    en = np.array([r["energy"] for r in hist])
    res = np.array([r["residual"] for r in hist])
    unit = np.array([r["residual_winding_unit_sphere"] for r in hist])
    wind = np.array([r["residual_winding"] for r in hist])
    eig = np.array([r["center_eigenvalues"] for r in hist])
    eig0 = np.array(j["start"]["center_eigenvalues"])
    out = dict(
        name=name, label=j["label"], iterations=j["iterations"], evaluations=j["evaluations"], wall_time_min=j["wall_time_s"] / 60, threads=j["threads"],
        optimizer=[(s["segment"], s["iterations"], s["optimizer_message"]) for s in j["segments"]], wall_hits=j["wall_hits"], resumed_at=j["resumed_at"],
        start_control=dict(energy_at_the_first_evaluation=e0, recorded=recorded, relative_difference=abs(e0 - recorded) / recorded, met=bool(abs(e0 - recorded) / recorded <= 1e-10),
                           residual_at_the_start=j["start"]["residual"], gradient_check=j["start"]["gradient_check"]),
        end=dict(energy=j["energy"], fall_from_the_recorded_energy=recorded - j["energy"], residual=j["residual"], residual_winding=j["residual_winding"],
                 residual_factor=j["residual_factor"], residual_winding_unit_sphere=j["residual_winding_unit_sphere"], winding_length_range=j["winding_length_range"],
                 center_eigenvalues=j["center_eigenvalues"], branch=j["branch"], degree=j["degree"], margin_four_triangulations=j["hemisphere_margin_four_triangulations"],
                 drift_last_tenth=j["energy_drift_last_tenth"], terms=j["terms"]),
        saved_rows=dict(count=len(hist), under_the_gate=int((res <= 1e-3).sum()), smallest_residual=float(res.min()), largest_residual=float(res.max()),
                        median_residual=float(np.median(res)), lowest_energy_under_the_gate=float(en[res <= 1e-3].min()) if (res <= 1e-3).any() else None,
                        first_under_the_gate_after_the_start=int(it[res <= 1e-3][0]) if (res <= 1e-3).any() else None,
                        largest_relative_gap_between_the_two_winding_readings=float(np.max(np.abs(unit - wind) / wind)),
                        largest_winding_length=float(max(r["winding_length_max"] for r in hist)),
                        largest_center_eigenvalue_move=float(np.abs(eig - eig0).max())),
        fall_at=[dict(iteration=int(i), fall=float(recorded - e)) for i, e in zip(it, en) if i in (500, 1000, 1500, 2000, 3000, 5000, 7500, 10000)],
    )
    base = load(DATA / f"m5_40_r2_reads_author_g{grid}.json")
    base_inertia = base["inertia_audit"]["actions"]["left"]["isospin_mean"]
    base_ro = readouts[f"m5_40_author_g{grid}"]
    stops = {}
    hand = WORK / name / "handoff.npz"
    if hand.exists():
        with np.load(hand) as d:
            hit = int(d["iteration"])
        stops[f"iteration {hit}"] = state_reads(hand, archive, f"{name}_handoff")
        stops[f"iteration {hit}"]["iteration"] = hit
    stops["end"] = state_reads(DATA / f"m5_40_{name}.npz", archive, f"{name}_end")
    stops["end"]["iteration"] = j["iterations"]
    for s in stops.values():
        s["fall_from_the_recorded_energy"] = recorded - s["energy"]
        s["inertia_change_percent"] = 100.0 * (s["inertia"] / base_inertia - 1.0)
        s["center_move"] = float(np.abs(np.array(s["center_eigenvalues"]) - eig0).max())
        s["displacement_of_the_winding_centroid"] = float(np.linalg.norm(np.array(s["winding_centroid"]) - np.array(base_ro["winding_centroid"])))
        s["changes_percent"] = {k: 100.0 * (s[k] / base_ro[k2] - 1.0) for k, k2 in (("rS2_fm2", "rS2_fm2"), ("rV2_fm2", "rV2_fm2"), ("muV", "muV"))}
        s["changes_percent"]["e2_gA"] = 100.0 * (s["e2_gA"] / base_ro["axial"]["e2_gA"] - 1.0)
    out["reads_at_the_two_stops"] = stops
    fall = out["end"]["fall_from_the_recorded_energy"]
    lo, hi = EXPECTED["energy_fall"]
    out["expectations_recorded_before_the_run"] = dict(
        energy_fall=dict(expected=[lo, hi], found=fall, met=bool(lo <= fall <= hi),
                         valley_floor_outcome=bool(fall < EXPECTED["valley_floor_fall"])),
        center_move=dict(expected_under=EXPECTED["center_move"], found=stops["end"]["center_move"], met=bool(stops["end"]["center_move"] < EXPECTED["center_move"]),
                         largest_along_the_run=out["saved_rows"]["largest_center_eigenvalue_move"]),
        inertia_move=dict(expected_under_percent=EXPECTED["inertia_move_percent"], found_percent=stops["end"]["inertia_change_percent"],
                          met=bool(abs(stops["end"]["inertia_change_percent"]) < EXPECTED["inertia_move_percent"])))
    return out


def rel(a, b):
    return float(abs(a - b) / abs(b))


def r4_block(iso, readouts):
    out = dict(gate_a={}, gate_b={})
    for grid in (33, 55):
        ours = readouts[f"m5_40_author_g{grid}"]
        folder = iso / f"f{grid}"
        g = dict(as_shipped=dict(zero_q2="NOT RUNNABLE: reads magnetic.jsonl beside the field, a file the payload does not hold",
                                 static_preflight="NOT RUNNABLE: reads calibration33.json beside the script, a file neither the payload nor the release holds"))
        if (folder / "zero_q2.json").exists() and (folder / "zero_q2.json").stat().st_size:
            z = load(folder / "zero_q2.json")[0]
            mag = load(folder / "magnetic.jsonl")
            pairs = dict(winding_by_the_rule=(ours["winding_by_the_rule"], z["B_quadrature"]), rS2_fm2=(ours["rS2_fm2"], z["rS2_fm2"]),
                         rV2_fm2=(ours["rV2_fm2"], z["rV2_fm2"]), gap_MeV=(ours["gap_MeV"], z["gap_MeV"]), m_z=(ours["magnetic"]["m_z"], mag["dE_d_eB"][2]),
                         muV=(ours["muV"], z["muV_muN"]), muV_rigid=(ours["muV_rigid"], z["muV_rigid_muN"]),
                         Q_transition_fm2=(ours["Q_transition_fm2"], z["Q_transition_fm2"]), inertia_mean=(ours["inertia_mean"], float(np.trace(np.array(z["inertia"])) / 3)))
            g["zero_q2_after_lattice_source"] = {k: dict(ours=a, the_authors_script=b, relative_difference=rel(a, b)) for k, (a, b) in pairs.items()}
            g["zero_q2_met"] = bool(max(v["relative_difference"] for v in g["zero_q2_after_lattice_source"].values()) <= 1e-8)
            g["the_authors_magnetic_line"] = mag
        sp = folder / "static_preflight.json"
        if sp.exists() and sp.stat().st_size:
            a = load(sp)[0]["axial"]
            pairs = dict(e2_gA_N_term=(ours["axial"]["e2_gA_N_term"], a["signed_gA_N_term"]), e2_gA_wedge=(ours["axial"]["e2_gA_wedge"], a["signed_gA_wedge"]),
                         e2_gA=(ours["axial"]["e2_gA"], a["gA_spurion_diagnostic"]))
            g["static_preflight_with_a_calibration_file_of_ours_e_equal_one"] = {k: dict(ours=x, the_authors_script=y, relative_difference=rel(x, y)) for k, (x, y) in pairs.items()}
            g["static_preflight_met"] = bool(max(v["relative_difference"] for v in g["static_preflight_with_a_calibration_file_of_ours_e_equal_one"].values()) <= 1e-8)
            g["the_authors_tail_fit"] = [dict(radius=t["radius"], amplitude=t["dirichlet"]["amplitude_lattice"], misfit=t["dirichlet"]["residual"]) for t in load(sp)[0]["tail"]]
        out["gate_a"][str(grid)] = g
    o33, o55 = readouts["m5_40_author_g33"], readouts["m5_40_author_g55"]
    p = PRINTED[33]
    e33 = o33["frozen_rule_e"]
    out["gate_b"] = dict(
        from_the_33_field=[
            dict(quantity="r_S^2 (fm^2)", printed=p["rS2_fm2"], read=o33["rS2_fm2"], word=word(o33["rS2_fm2"], p["rS2_fm2"], 4)),
            dict(quantity="r_V^2 (fm^2)", printed=p["rV2_fm2"], read=o33["rV2_fm2"], word=word(o33["rV2_fm2"], p["rV2_fm2"], 4)),
            dict(quantity="gap (MeV)", printed=p["gap_MeV"], read=o33["gap_MeV"], word=word(o33["gap_MeV"], p["gap_MeV"], 1)),
            dict(quantity="mu_V", printed=p["muV"], read=o33["muV"], word=word(o33["muV"], p["muV"], 3)),
            dict(quantity="mu_V, rigid relation", printed=p["muV_rigid"], read=o33["muV_rigid"], word=word(o33["muV_rigid"], p["muV_rigid"], 3)),
            dict(quantity="g_A at the rule's e of this field", e=e33, printed=PRINTED_AXIAL_33["gA"], read=o33["axial"]["e2_gA"] / e33 ** 2,
                 word=word(o33["axial"]["e2_gA"] / e33 ** 2, PRINTED_AXIAL_33["gA"], 3)),
            dict(quantity="g_A, N-gradient part", e=e33, printed=PRINTED_AXIAL_33["N_term"], read=abs(o33["axial"]["e2_gA_N_term"]) / e33 ** 2,
                 word=word(abs(o33["axial"]["e2_gA_N_term"]) / e33 ** 2, PRINTED_AXIAL_33["N_term"], 3)),
            dict(quantity="g_A, wedge part", e=e33, printed=PRINTED_AXIAL_33["wedge"], read=abs(o33["axial"]["e2_gA_wedge"]) / e33 ** 2,
                 word=word(abs(o33["axial"]["e2_gA_wedge"]) / e33 ** 2, PRINTED_AXIAL_33["wedge"], 3)),
        ],
        from_the_55_field=[
            dict(quantity="g_A at e = 4.311", e=4.311, printed=0.904, read=o55["axial"]["e2_gA"] / 4.311 ** 2, word=word(o55["axial"]["e2_gA"] / 4.311 ** 2, 0.904, 3)),
            dict(quantity="g_A at the rule's e of this field", e=o55["frozen_rule_e"], printed=0.904, read=o55["axial"]["e2_gA"] / o55["frozen_rule_e"] ** 2,
                 word=word(o55["axial"]["e2_gA"] / o55["frozen_rule_e"] ** 2, 0.904, 3)),
            dict(quantity="g_A at the rule's e of the 33^3 field (the text: the larger field alone)", e=e33, printed=0.986, read=o55["axial"]["e2_gA"] / e33 ** 2,
                 word=word(o55["axial"]["e2_gA"] / e33 ** 2, 0.986, 3)),
        ])

    def ladder(rows, spacing_of):
        got = [(name, grid, readouts[name]) for name, grid in rows if name in readouts]
        if len(got) < len(rows):
            return dict(status="INCOMPLETE", have=[g for _, g, _ in got])
        h2 = np.array([spacing_of(r) ** 2 for _, _, r in got])
        table = dict(grids=[g for _, g, _ in got], states=[n for n, _, _ in got], spacing_squared=h2.tolist(), half_boxes=[r["half_box"] for _, _, r in got])
        series = {
            "inertia": [r["inertia_mean"] for _, _, r in got], "rS2_fm2": [r["rS2_fm2"] for _, _, r in got], "rV2_fm2": [r["rV2_fm2"] for _, _, r in got],
            "muV": [r["muV"] for _, _, r in got], "muV_over_rigid_minus_one": [r["muV_deviation_fraction"] for _, _, r in got],
            "e2_gA": [r["axial"]["e2_gA"] for _, _, r in got], "frozen_rule_e": [r["frozen_rule_e"] for _, _, r in got],
            "gA_at_own_rule_e": [r["axial"]["e2_gA"] / r["frozen_rule_e"] ** 2 for _, _, r in got],
            "gA_at_4.311": [r["axial"]["e2_gA"] / 4.311 ** 2 for _, _, r in got],
            "N_term_share": [r["axial"]["e2_gA_N_term"] / (r["axial"]["e2_gA_N_term"] + r["axial"]["e2_gA_wedge"]) for _, _, r in got],
        }
        table["series"] = series
        return table

    mesh = ladder(MESH, lambda r: r["spacing"])
    if "series" in mesh:
        h2 = np.array(mesh["spacing_squared"])
        fits = {}
        for key, y in mesh["series"].items():
            y = np.array(y)
            slope, intercept = np.polyfit(h2[1:], y[1:], 1)
            slope2, intercept2 = np.polyfit(h2[2:], y[2:], 1)
            fits[key] = dict(slopes_between_neighbors=(np.diff(y) / np.diff(h2)).tolist(), extrapolated_from_the_three_finest=float(intercept),
                             extrapolated_from_the_two_finest=float(intercept2), value_at_55=float(y[1]),
                             the_55_value_over_the_extrapolation_percent=float(100 * (y[1] / intercept - 1)))
        mesh["fits_in_h2"] = fits
    out["mesh_ladder"] = mesh
    out["box_ladder"] = ladder(BOXES, lambda r: r["spacing"])
    out["same_grid_pairs"] = {}
    for grid, ref in ((33, "m5_39_r2_g33_iso"), (55, "m5_39_r2_g55_smooth")):
        if ref in readouts:
            a, b = readouts[f"m5_40_author_g{grid}"], readouts[ref]
            out["same_grid_pairs"][str(grid)] = {k: dict(the_authors_field=a[k], state_of_m5_39=b[k], relative_difference=rel(b[k], a[k]))
                                                 for k in ("inertia_mean", "rS2_fm2", "rV2_fm2", "muV")}
            out["same_grid_pairs"][str(grid)]["e2_gA"] = dict(the_authors_field=a["axial"]["e2_gA"], state_of_m5_39=b["axial"]["e2_gA"],
                                                              relative_difference=rel(b["axial"]["e2_gA"], a["axial"]["e2_gA"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--iso", type=Path, required=True)
    a = ap.parse_args()
    r1 = load(DATA / "m5_40_r1_payload.json")
    readouts = load(DATA / "m5_40_r4_readouts.json")
    by_grid = {f["grid"]: f for f in r1["fields"]}
    out = dict(task="M5.40", units=dict(length_fm=LENGTH, e=SCALE[0], f_pi_MeV=SCALE[1], hbar_c=HBARC),
               R1={str(g): dict(digests_all_match=r1["digests"]["all_match"], layout=f["layout"], K1=f["K1"], K2=f["K2"], K3=f["K3"]) for g, f in by_grid.items()},
               R2={str(g): r2_block(g, a.archive) for g in (33, 55)},
               R3={str(g): r3_block(g, a.archive, by_grid[g], readouts) for g in (33, 55)},
               R4=r4_block(a.iso, readouts))
    (DATA / "m5_40_summary.json").write_text(json.dumps(out, indent=1) + "\n")

    for g in ("33", "55"):
        print(f"=== {g}^3")
        for row in out["R2"][g]["printed_rows"]:
            print("  R2", row["quantity"], "| printed", row["printed"], "| read", row["read"], "|", row["word"], row.get("last_four_shells", ""), row.get("last_four_shells_word", ""))
        print("  R2 node diagnostic", out["R2"][g]["node_by_node_diagnostic"])
        print("  R2 shell differences", {k: round(v["largest_difference"], 6) for k, v in out["R2"][g]["profiles"]["shell_differences"].items()})
        for s in out["R2"][g]["states_of_m5_39"]:
            print(f"     {s['name']:22s} {s['label']:15s} E {s['energy']:.6f} dE {s['energy_minus_the_authors']:+.2e} res {s['residual']:.2e} I {s.get('inertia', float('nan')):.4f}")
        r3 = out["R3"][g]
        if "end" in r3:
            print("  R3", json.dumps({k: r3[k] for k in ("label", "iterations", "optimizer", "start_control", "end", "saved_rows", "fall_at", "expectations_recorded_before_the_run")}, indent=1))
            for k, s in r3["reads_at_the_two_stops"].items():
                print("  R3 stop", k, {kk: s[kk] for kk in ("energy", "fall_from_the_recorded_energy", "residual", "inertia", "inertia_change_percent", "lumps", "center_move", "rS2_fm2", "rV2_fm2", "muV", "e2_gA", "changes_percent")})
        else:
            print("  R3", r3)
    print("R4", json.dumps({k: out["R4"][k] for k in ("gate_a", "gate_b", "same_grid_pairs")}, indent=1))
    for lad in ("mesh_ladder", "box_ladder"):
        print(lad, json.dumps(out["R4"][lad], indent=1))


if __name__ == "__main__":
    main()
