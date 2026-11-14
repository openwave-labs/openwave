"""M5.39 R3: the static half of the record ablation, read on the paired continuations.

The ablated card is Card C with the term a a^T / G_time removed from the tensor N (lam = 0 in
m5_39_cardc.py); the record stays in every other term. Per grid, two continuations start from one
state, one of the card and one of the ablated card (m5_39_pool.py); every difference below is the card
minus its ablation.

Reads per grid:
  energy     (E_card - E_ablated) / E_ablated, and the difference by term;
  inertia    as fractions of the card's isospin inertia: the direct part (the card's end state read
             with and without a a^T / G_time in N), the relaxed change (each end state read with its
             own N), and the chiral part (both end states read without it);
  radius     the rms radius of the winding density, in fm (the card's length unit 0.5014 fm);
  split      N = -P + lam Ga / G_time with P = q q^T and Ga = a a^T makes the tensor term
             E_PP - (2 lam / G) E_PG + (lam / G)^2 E_GG on the card's two-sided difference rule: the
             three parts are computed separately and must return the evaluator's term to round-off;
  cross      the cross part against its continuum form, eps sum_i (d_i q)^T A (d_i q) with A = a a^T
             and eps = 1 / G_time. Read on the card's own rule with A averaged over each edge the two
             are the same number (a^T q = 0 on every node makes the identity exact on the lattice);
             read at the nodes with centered differences the remainder is second order in the
             spacing, checked on a smooth state at four spacings and reported on the relaxed states;
  share      the cross part over the winding part E_PP, and the largest eigenvalue of the record (the
             operator norm of A), on which the bound 2 / G_time of the share is conditional.

Known answers, printed in the v3.4 text from another team's replication on the author's fields and
read before this run: the energy change +0.0425, +0.0433, +0.0438 percent on 33^3, 45^3, 55^3; the
direct part +2.7 to +2.8e-4; the relaxed change -2.7 to -3.6e-4; the rms radius smaller by 1.5e-4 fm.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_39_cardc as C  # noqa: E402
import m5_39_reads as R  # noqa: E402

DATA = HERE.parent / "data"
FM = 0.5014
PRINTED = {33: 0.0425, 45: 0.0433, 55: 0.0438}
DIRECT = (2.7e-4, 2.8e-4)
RELAXED = (-3.6e-4, -2.7e-4)


def two_sided(X, Y, h, coefficient):
    """The card's two-sided difference rule as a bilinear form: (c h^3 / 2) sum <D X, D Y>, forward
    differences on the cells based at [0, n-2]^3 and backward ones on those based at [1, n-1]^3."""
    total = 0.0
    for box, step in (((slice(None, -1),) * 3, 1), ((slice(1, None),) * 3, -1)):
        for axis in range(3):
            nb = list(box)
            nb[axis] = slice(1, None) if step == 1 else slice(None, -1)
            nb = tuple(nb)
            total += float(np.sum((X[nb] - X[box]) * (Y[nb] - Y[box])))
    return 0.5 * coefficient * h * total  # h^3 / h^2


def node_fields(state):
    p = state[..., :4]
    q = p / np.linalg.norm(p, axis=-1, keepdims=True)
    f = state[..., 4:].reshape(state.shape[:3] + (3, 3))
    q0, q1, q2, q3 = np.moveaxis(q, -1, 0)
    tau = np.stack((np.stack((-q1, q0, q3, -q2), -1), np.stack((-q2, -q3, q0, q1), -1), np.stack((-q3, q2, -q1, q0), -1)), -1)
    a = tau @ f
    return q, a, q[..., :, None] * q[..., None, :], a @ np.swapaxes(a, -1, -2), np.swapaxes(f, -1, -2) @ f


def cross_continuum(q, A, h):
    """eps sum_i (d_i q)^T A (d_i q) on the two-sided rule, A averaged over each edge."""
    total = 0.0
    for box, step in (((slice(None, -1),) * 3, 1), ((slice(1, None),) * 3, -1)):
        for axis in range(3):
            nb = list(box)
            nb[axis] = slice(1, None) if step == 1 else slice(None, -1)
            nb = tuple(nb)
            d = q[nb] - q[box]
            Am = 0.5 * (A[nb] + A[box])
            total += float(np.sum(np.einsum("...i,...ij,...j->...", d, Am, d)))
    return 0.5 * h * total / C.G_TIME


def cross_centered(q, A, h):
    """The same continuum form read at the nodes with centered differences."""
    grads = np.gradient(q, h, axis=(0, 1, 2), edge_order=2)
    return float(sum(np.sum(np.einsum("...i,...ij,...j->...", d, A, d)) for d in grads)) * h ** 3 / C.G_TIME


def split(state, h, grid, box):
    q, a, P, Ga, s = node_fields(state)
    e_pp = two_sided(P, P, h, C.TENSOR)
    e_pg = two_sided(P, Ga, h, C.TENSOR)
    e_gg = two_sided(Ga, Ga, h, C.TENSOR)
    cross = -2.0 * e_pg / C.G_TIME
    pure = e_gg / C.G_TIME ** 2
    card, ablated = C.CardC(grid, box), C.CardC(grid, box, lam=0.0)
    t1, t0 = card(state)[0][0], ablated(state)[0][0]
    cont = cross_continuum(q, Ga, h)
    centered = cross_centered(q, Ga, h)
    return dict(cross_centered_form=centered, cross_remainder_centered=(cross - centered) / cross, winding_part=e_pp, cross_part=cross, pure_record_part=pure, tensor_term_card=float(t1), tensor_term_ablated=float(t0),
                split_misfit_card=abs(e_pp + cross + pure - t1) / t1, split_misfit_ablated=abs(e_pp - t0) / t0,
                cross_continuum_form=cont, cross_remainder=(cross - cont) / cross, share_of_the_winding_part=cross / e_pp,
                largest_record_eigenvalue=float(np.linalg.eigvalsh(s).max()), bound_on_the_share=2.0 / C.G_TIME)


def load(name):
    with np.load(DATA / f"m5_39_{name}.npz") as d:
        return d["state"], d["coords"]


def inertia_parts(card_state, abl_state, coords, h):
    i_c1 = R.isospin_inertia(card_state, coords, h, 1.0)["mean"]
    i_c0 = R.isospin_inertia(card_state, coords, h, 0.0)["mean"]
    i_a0 = R.isospin_inertia(abl_state, coords, h, 0.0)["mean"]
    return dict(card=i_c1, card_read_without_the_term=i_c0, ablated=i_a0, direct=(i_c1 - i_c0) / i_c1, relaxed=(i_c1 - i_a0) / i_c1,
                chiral=(i_c0 - i_a0) / i_c1)


def smooth_fixture():
    """The order of the cross remainder on one smooth state at four spacings."""
    rows = []
    for grid in (17, 33, 65, 129):
        state, coords, h = C.smooth_test_state(grid, 3.6)
        q, a, P, Ga, s = node_fields(state)
        cross = -2.0 * two_sided(P, Ga, h, C.TENSOR) / C.G_TIME
        rows.append(dict(grid=grid, spacing=h, remainder_edge_averaged=(cross - cross_continuum(q, Ga, h)) / cross,
                         remainder=(cross - cross_centered(q, Ga, h)) / cross))
    orders = [float(np.log2(abs(rows[k]["remainder"] / rows[k + 1]["remainder"]))) for k in range(3)]
    return dict(rows=rows, orders=orders, second_order=bool(abs(orders[-1] - 2.0) < 0.3),
                edge_averaged_form_is_exact=bool(max(abs(r["remainder_edge_averaged"]) for r in rows) < 1e-12),
                review_fixture=dict(points=(32, 64, 128), remainder=(6.9e-4, 1.8e-4, 4.4e-5)))


def main():
    out = dict(task="M5.39 R3", printed=dict(energy_percent=PRINTED, direct=DIRECT, relaxed=RELAXED, rms_radius_fm=-1.5e-4), grids=[])
    out["cross_remainder_fixture"] = smooth_fixture()
    print("fixture orders", [round(x, 2) for x in out["cross_remainder_fixture"]["orders"]], [f"{r['remainder']:.2e}" for r in out["cross_remainder_fixture"]["rows"]])
    for grid in (33, 45, 55):
        pc, pa = DATA / f"m5_39_r3_c{grid}_pair.json", DATA / f"m5_39_r3_a{grid}_pair.json"
        census = DATA / f"m5_39_r2_census_g{grid}.json"
        if not (pc.exists() and pa.exists() and census.exists()):
            out["grids"].append(dict(grid=grid, label="NOT RUN YET"))
            continue
        jc, ja, sel = json.loads(pc.read_text()), json.loads(pa.read_text()), json.loads(census.read_text())
        sc, coords = load(f"r3_c{grid}_pair")
        sa, _ = load(f"r3_a{grid}_pair")
        h, box = jc["spacing"], jc["half_box"]
        row = dict(grid=grid, start=sel["winner"], start_is_stationary=sel["from_stationary"], start_branch=sel["branch"],
                   card=dict(label=jc["label"], energy=jc["energy"], residual=jc["residual"], iterations=jc["iterations"], drift=jc["energy_drift_last_tenth"], branch=jc["branch"]),
                   ablated=dict(label=ja["label"], energy=ja["energy"], residual=ja["residual"], iterations=ja["iterations"], drift=ja["energy_drift_last_tenth"], branch=ja["branch"]))
        row["energy_change_percent"] = 100.0 * (jc["energy"] - ja["energy"]) / ja["energy"]
        row["energy_change_by_term"] = (np.array(jc["terms"]) - np.array(ja["terms"])).tolist()
        row["inertia"] = inertia_parts(sc, sa, coords, h)
        rc, ra = R.winding_density(sc, coords, h), R.winding_density(sa, coords, h)
        row["rms_radius_change_fm"] = FM * (np.sqrt(rc["isoscalar_r2"]) - np.sqrt(ra["isoscalar_r2"]))
        row["isovector_r2_change_fm2"] = FM ** 2 * (R.isospin_inertia(sc, coords, h, 1.0)["isovector_r2"] - R.isospin_inertia(sa, coords, h, 0.0)["isovector_r2"])
        row["split"] = split(sc, h, grid, box)
        scored = sel["from_stationary"] and sel["branch"] == "ISOTROPIC CENTER" and jc["label"] == "STATIONARY"
        e_ok = abs(row["energy_change_percent"] / PRINTED[grid] - 1.0) <= 0.05
        d, r = row["inertia"]["direct"], row["inertia"]["relaxed"]
        i_ok = DIRECT[0] / 1.5 <= d <= DIRECT[1] * 1.5 and RELAXED[0] * 1.5 <= r <= RELAXED[1] / 1.5
        if not scored:
            row["label"] = "NOT SCORED"
        elif ja["label"] != "STATIONARY":
            row["label"] = "FAILS: the ablated continuation is not stationary where the card's is"
        else:
            row["label"] = "AGREES" if e_ok and i_ok else "DIFFERS"
        row["energy_within_5_percent"], row["inertia_parts_within_a_factor_1_5"] = bool(e_ok), bool(i_ok)
        out["grids"].append(row)
        print(f"{grid}: dE {row['energy_change_percent']:+.4f} % (printed {PRINTED[grid]:+.4f}) direct {d:+.2e} relaxed {r:+.2e} chiral {row['inertia']['chiral']:+.2e} "
              f"d r_rms {row['rms_radius_change_fm']:+.2e} fm split {row['split']['split_misfit_card']:.1e} remainder {row['split']['cross_remainder_centered']:+.2e} "
              f"share {row['split']['share_of_the_winding_part']:.2e} |A| {row['split']['largest_record_eigenvalue']:.3f} {row['label']}")
    # the unpaired check on 33^3
    pu, c33 = DATA / "m5_39_r3_a33_unpaired.json", DATA / "m5_39_r2_census_g33.json"
    if pu.exists() and c33.exists() and out["grids"] and out["grids"][0].get("card"):
        ju, sel = json.loads(pu.read_text()), json.loads(c33.read_text())
        g = out["grids"][0]
        paired = g["card"]["energy"] - g["ablated"]["energy"]
        unpaired = sel["energy"] - ju["energy"]
        floor = sum(abs(x or 0.0) for x in (g["card"]["drift"], g["ablated"]["drift"], ju["energy_drift_last_tenth"]))
        out["unpaired_check_33"] = dict(ablated_from_the_linked_seed=dict(label=ju["label"], energy=ju["energy"], residual=ju["residual"], branch=ju["branch"],
                                                                         center_eigenvalues=ju["center_eigenvalues"], iterations=ju["iterations"]),
                                        paired_difference=paired, unpaired_difference=unpaired, stopping_floor=floor,
                                        returned=bool(abs(paired - unpaired) <= floor))
        print("unpaired check:", out["unpaired_check_33"])
    (DATA / "m5_39_r3_ablation.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
