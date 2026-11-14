"""M5.40 R1: the author's two Card C fields under the evaluator of M5.39.

    python3 -I m5_40_payload.py --payload <folder of CardC_payload_recovered.zip> --out-dir ../data

The payload is read as data only (arrays without pickles, JSON, the digest list); none of its code is
imported here. For each field (33^3 at half-box 3.6, 55^3 at half-box 6.075):

  digests   every file of the payload's SHA256SUMS, recomputed;
  layout    `state` is (p, f) per node, 13 reals, in the order of the unknowns of m5_39_cardc.py; the
            stored `q` against p / |p|, the stored `record` against f^T f, the stored `coords` against
            the grid of the card, the boundary against the card's boundary values, and the range of
            |p| over the nodes;
  K1        the energy of our evaluator against the energy of the run record (1e-10 relative);
  K2        the winding number, the two margins that the deposited code defines (the smallest dot
            product of two vertices of a simplex of the first Kuhn triangulation, `topology` of
            dn87_relax.py; the smallest v . (sum of the simplex's vertices), `hemisphere_margin` of
            bq_core.py), and candidates for the record's "max_record_amplitude", a name that the
            deposited code does not define;
  K3        the largest gradient entry over h^3, whole, on the winding unknowns and on the factor, in
            two readings: (A) on p as stored, grad_p = (dq - q (q . dq)) / |p|; (B) on the unit
            sphere, dq - q (q . dq), which is reading A times |p| node by node. Each against the three
            residuals of the run record (1 percent). No other reading and no fitted factor.

The fields are then written in the layout of m5_39_cardc.save_state (local-only arrays) for the
readers and the driver of M5.39.
"""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m5_39_cardc as C  # noqa: E402

FIELDS = [
    dict(tag="g33", folder="bq_c33/residual_edge", grid=33, box=3.6),
    dict(tag="g55", folder="bq_rc76/gate55", grid=55, box=6.075),
]


def digests(payload):
    rows = []
    for line in (payload / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        want, name = line.split(maxsplit=1)
        got = hashlib.sha256((payload / name.strip()).read_bytes()).hexdigest()
        rows.append(dict(file=name.strip().removeprefix("./"), match=bool(got == want)))
    return rows


def pair_margin(q):
    """Smallest dot product of two vertices of one simplex, first Kuhn triangulation."""
    n = q.shape[0]
    best = 1.0
    for o in range(6):
        v = [q[tuple(slice(int(C.OFFSETS[0, o, m, a]), n - 1 + int(C.OFFSETS[0, o, m, a])) for a in range(3))] for m in range(4)]
        for i, j in itertools.combinations(range(4), 2):
            best = min(best, float(np.min(np.sum(v[i] * v[j], axis=-1))))
    return best


def rel(a, b):
    return float(abs(a - b) / abs(b))


def read_field(payload, spec, out_dir):
    folder = payload / spec["folder"]
    record = json.loads((folder / "run.json").read_text())
    with np.load(folder / "final.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    state, coords = arrays["state"], arrays["coords"]
    n = state.shape[0]
    grid_coords, h = C.coordinates(spec["grid"], spec["box"])
    p = state[..., :4]
    f = state[..., 4:].reshape(state.shape[:3] + (3, 3))
    plen = np.linalg.norm(p, axis=-1)
    q = p / plen[..., None]
    s = np.swapaxes(f, -1, -2) @ f
    boundary = np.ones(state.shape[:3], dtype=bool)
    boundary[1:-1, 1:-1, 1:-1] = False
    vac = np.zeros(13)
    vac[0] = 1.0
    layout = dict(
        arrays={k: list(v.shape) for k, v in arrays.items()}, dtype=str(state.dtype),
        coords_against_the_card_grid=float(np.abs(coords - grid_coords).max()),
        stored_q_against_p_over_length=float(np.abs(arrays["q"] - q).max()),
        stored_record_against_f_transpose_f=float(np.abs(arrays["record"] - s).max()),
        boundary_against_the_card_values=float(np.abs(state[boundary] - vac).max()),
        winding_vector_length=dict(min=float(plen.min()), max=float(plen.max()),
                                   interior_min=float(C.interior(plen).min()), interior_max=float(C.interior(plen).max()),
                                   largest_departure_from_one=float(np.abs(plen - 1.0).max())),
    )

    final = record["history"][-1]
    card = C.CardC(n, spec["box"], record["delta"], record["kx"], record["ko"])
    terms, grad = card(state)
    energy = float(terms.sum())
    k1 = dict(energy=energy, recorded=final["energy"], relative_difference=rel(energy, final["energy"]), bound=1e-10)
    k1["met"] = bool(k1["relative_difference"] <= k1["bound"])

    g = C.interior(grad).reshape(-1, 13)
    pl = C.interior(plen).reshape(-1, 1)
    h3 = h ** 3
    readings = {
        "A, on the winding vector as stored": dict(
            residual=float(np.abs(g).max() / h3), residual_q=float(np.abs(g[:, :4]).max() / h3), residual_f=float(np.abs(g[:, 4:]).max() / h3)),
        "B, on the unit sphere": dict(
            residual=float(max(np.abs(g[:, :4] * pl).max(), np.abs(g[:, 4:]).max()) / h3),
            residual_q=float(np.abs(g[:, :4] * pl).max() / h3), residual_f=float(np.abs(g[:, 4:]).max() / h3)),
    }
    k3 = dict(recorded={k: final[k] for k in ("residual", "residual_q", "residual_f")}, readings=readings, bound=0.01)
    for name, r in readings.items():
        r["relative_differences"] = {k: rel(r[k], final[k]) for k in ("residual", "residual_q", "residual_f")}
        r["met"] = bool(max(r["relative_differences"].values()) <= k3["bound"])
    k3["identified_with"] = [name for name, r in readings.items() if r["met"]]
    # the per-node 4-vector norm of the winding gradient, a number for the record, not a reading of K3
    k3["largest_node_norm_of_the_unit_sphere_winding_gradient_over_h3"] = float(np.linalg.norm(g[:, :4] * pl, axis=-1).max() / h3)

    deg, hits = C.degree(card.q)
    m0, m4 = card.margins(all_four=True)
    lam = np.linalg.eigvalsh(s)
    sv = np.linalg.svd(f, compute_uv=False)
    center = lam[n // 2, n // 2, n // 2]
    candidates = {
        "largest record eigenvalue over the grid": float(lam.max()),
        "largest record eigenvalue at the center node": float(center.max()),
        "largest entry of the record in magnitude": float(np.abs(s).max()),
        "largest singular value of the factor": float(sv.max()),
        "largest entry of the factor in magnitude": float(np.abs(f).max()),
        "largest Frobenius norm of the record": float(np.sqrt(np.sum(s * s, axis=(-2, -1))).max()),
    }
    where = np.unravel_index(int(np.argmax(lam[..., 2])), lam.shape[:3])
    k2 = dict(
        degree=int(deg), degree_preimages=int(hits), recorded_degree=final["degree"],
        pair_margin=pair_margin(q), hemisphere_margin_first_triangulation=float(m0), hemisphere_margin_four_triangulations=float(m4),
        recorded_margin=final["margin"], recorded_max_record_amplitude=final["max_record_amplitude"],
        max_record_amplitude_candidates=candidates,
        node_of_the_largest_record_eigenvalue=dict(index=[int(x) for x in where], radius=float(np.linalg.norm(coords[where]))),
        center_eigenvalues=center.tolist(), branch=C.branch(center),
    )
    k2["pair_margin_relative_difference"] = rel(k2["pair_margin"], final["margin"])
    k2["max_record_amplitude_matches"] = [name for name, v in candidates.items() if rel(v, final["max_record_amplitude"]) <= 5e-6]
    k2["met"] = bool(deg == 1 and k2["pair_margin_relative_difference"] <= 5e-6 and len(k2["max_record_amplitude_matches"]) > 0)

    out_path = out_dir / f"m5_40_author_{spec['tag']}.npz"
    C.save_state(out_path, state, coords, compressed=True, iteration=0, energy=energy)
    return dict(tag=spec["tag"], source=f"{spec['folder']}/final.npz", grid=n, half_box=spec["box"], spacing=h,
                run_record={k: v for k, v in record.items() if k != "history"}, run_record_history=record["history"],
                layout=layout, terms=terms.tolist(), term_names=C.TERM_NAMES, K1=k1, K2=k2, K3=k3, converted_copy=out_path.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    rows = digests(a.payload)
    out = dict(task="M5.40 R1", digests=dict(files=len(rows), all_match=bool(all(r["match"] for r in rows)), rows=rows),
               fields=[read_field(a.payload, spec, a.out_dir) for spec in FIELDS])
    (a.out_dir / "m5_40_r1_payload.json").write_text(json.dumps(out, indent=1) + "\n")
    for fld in out["fields"]:
        print(json.dumps({k: fld[k] for k in ("tag", "layout", "K1", "K2", "K3")}, indent=1))
    print("digests", out["digests"]["files"], out["digests"]["all_match"])


if __name__ == "__main__":
    main()
