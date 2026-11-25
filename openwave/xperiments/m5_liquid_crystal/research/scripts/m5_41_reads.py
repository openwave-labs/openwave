"""M5.41 R3: the reads at every saved stop of a route, one process per state.

For each state (a stop file of m5_39_relax.py --stops, or any state in that layout):

  the energy and its seven terms, recomputed from the file (m5_39_cardc.py);
  the residual in both readings, largest entry over the interior nodes over h^3: on the winding
  vector as stored, and the gate's reading, the larger of the winding gradient on the unit sphere
  (grad_p times |p|) and the factor gradient;
  the range of the winding vector's length, the winding number, the hemisphere margins;
  the center eigenvalues and the branch label;
  the record lumps (record_lumps of m5_39_reads.py, unchanged);
  the readouts of m5_40_readouts.py, unchanged: the inertia, the two radii, mu_V with its rigid
  value, g_A e^2 with its two parts.

Nothing here decides a label; m5_41_summary.py reads these files.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GATE = 1e-3


def read_state(path, out_dir):
    import m5_39_cardc as C
    import m5_39_reads as R39
    import m5_40_readouts as R40

    t0 = time.time()
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        state, coords = data["state"], data["coords"]
        stored = {k: (int(data[k]) if k in ("iteration", "target") else float(data[k])) for k in ("iteration", "energy", "target") if k in data}
    n = state.shape[0]
    h = float(coords[1, 0, 0, 0] - coords[0, 0, 0, 0])
    box = float(coords[-1, 0, 0, 0])
    card = C.CardC(n, box)
    terms, grad = card(state)
    g = np.abs(C.interior(grad).reshape(-1, 13))
    plen = np.linalg.norm(C.interior(state)[..., :4], axis=-1).reshape(-1, 1)
    m0, m4 = card.margins(all_four=True)
    deg, hits = C.degree(card.q)
    eig = C.center_eigenvalues(state)
    winding_unit, factor = float((g[:, :4] * plen).max() / h ** 3), float(g[:, 4:].max() / h ** 3)
    out = dict(
        file=path.name, grid=n, half_box=box, spacing=h, stored=stored,
        energy=float(terms.sum()), terms=terms.tolist(), term_names=C.TERM_NAMES,
        residual_as_stored=float(g.max() / h ** 3), residual_winding_as_stored=float(g[:, :4].max() / h ** 3),
        residual_winding_unit_sphere=winding_unit, residual_factor=factor, gate=max(winding_unit, factor),
        gate_met=bool(max(winding_unit, factor) <= GATE),
        winding_length_range=[float(plen.min()), float(plen.max())],
        degree=int(deg), degree_preimages=int(hits), hemisphere_margin_first_triangulation=float(m0),
        hemisphere_margin_four_triangulations=float(m4),
        center_eigenvalues=eig.tolist(), branch=C.branch(eig),
        record_lumps=R39.record_lumps(state, coords),
    )
    rd = R40.readouts(path, axial=True)
    out["readouts"] = dict(
        inertia=rd["inertia_mean"], inertia_diagonal=np.diag(np.array(rd["inertia_matrix"])).tolist(),
        rS2_fm2=rd["rS2_fm2"], rV2_fm2=rd["rV2_fm2"], muV=rd["muV"], muV_rigid=rd["muV_rigid"],
        muV_deviation_fraction=rd["muV_deviation_fraction"], m_z=rd["magnetic"]["m_z"], gap_MeV=rd["gap_MeV"],
        gA_e2=rd["axial"]["e2_gA"], gA_e2_N_term=rd["axial"]["e2_gA_N_term"], gA_e2_wedge=rd["axial"]["e2_gA_wedge"],
        gA_e2_closed_form_relative_difference=rd["axial"]["closed_form_relative_difference"],
        frozen_rule_e=rd["frozen_rule_e"], winding_by_the_rule=rd["winding_by_the_rule"],
        tensor_quadratic=rd["tensor_quadratic"], quaternion_quartic=rd["quaternion_quartic"], constants=rd["constants"])
    out["seconds"] = time.time() - t0
    target = Path(out_dir) / (path.stem + ".json")
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(tmp, target)
    return path.name, out["seconds"], out["energy"], out["gate"], out["record_lumps"]["count"], out["readouts"]["inertia"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", type=Path, nargs="*", help="state files; default: every data/m5_41_r2_*_s<target>.npz")
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_41" / "reads")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--redo", action="store_true", help="read again the states that already have a JSON")
    a = ap.parse_args()
    states = a.states or sorted((HERE.parent / "data").glob("m5_41_r2_*_s[0-9]*.npz"))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in states if a.redo or not (a.out_dir / (p.stem + ".json")).exists()
            or (a.out_dir / (p.stem + ".json")).stat().st_mtime < p.stat().st_mtime]
    # the largest grids first, so that the pool ends on the short ones
    todo.sort(key=lambda p: -p.stat().st_size)
    print(f"{len(states)} states, {len(todo)} to read, {a.workers} workers", flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(read_state, p, a.out_dir): p for p in todo}
        for fut in as_completed(futures):
            try:
                name, sec, energy, gate, lumps, inertia = fut.result()
                print(f"{name}: E {energy:.8f} gate {gate:.3e} lumps {lumps} inertia {inertia:.6f} ({sec:.0f} s)", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAILED {futures[fut].name}: {exc!r}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
