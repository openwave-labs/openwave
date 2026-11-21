"""M5.50, a descriptor outside the plan: the box surface at the blind reads of the exit chains E0a, E0b and E14d.

The exit chains are run again exactly as the pool ran them (the same spec, the same descent, the same reads), in a
separate work folder, with one addition to every read that is not part of the chain: for each nested surface whose
class is B (not liftable, no pierced triangle, no pierced plaquette within 1.0), the least gap l3 - l2 over the nodes
of the surface, the node where it sits (its coordinates, the face of the cube and the distance to the nearest edge
of that face), and the number of nodes of the surface at or under the gap bound 0.2. The pierced plaquettes of the
line reader (all of them, fragments included) are taken from the chain's own rows at the reads before and after.

Nothing here enters a label: R4 is read by m5_50_reads.py on the records of the pool.

Usage:  python m5_50_diag_r4.py E0a E0b E14d        (work folder: ../checkpoints/m5_50/diag/)
"""

import json
import os
import sys
import types
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_46_degree as DG  # noqa: E402
import m5_50_hold as H  # noqa: E402

WORK = HERE.parent / "checkpoints" / "m5_50"
DIAG = WORK / "diag"
FACE = ["-x", "+x", "-y", "+y", "-z", "+z"]


def main(name):
    work, data = DIAG / "work", DIAG / "data"
    (work / "chains" / name).mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    spec = json.loads((WORK / "chains" / name / "spec.json").read_text())
    (work / "chains" / name / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    src = data / spec["start"]["file"]
    if not src.exists():
        src.symlink_to(HERE.parent / "data" / spec["start"]["file"])
    for f in (work / "chains" / f"{name}.json", work / "chains" / name / "ckpt.npz", work / "chains" / name / "state.json"):
        f.unlink(missing_ok=True)
    calls = []
    orig = H.read_plus

    def read_plus(m6, x, surfaces):
        r = orig(m6, x, surfaces)
        info = dict(cls=r["cls"], pierced=r["line"]["pierced"], fragments=r["line"]["fragments"], blind=[])
        if "B" in r["cls"]:
            n, gap = DG.director(m6)
            N, h = len(x), x[1] - x[0]
            for k, (S, ch) in enumerate(zip(surfaces, r["cls"])):
                if ch != "B":
                    continue
                gv = gap.reshape(-1)[S.nodes]
                j = int(gv.argmin())
                ijk = np.array(np.unravel_index(S.nodes[j], (N, N, N)))
                p = x[ijk]
                ax = int(np.argmax(np.abs(p)))
                face = FACE[2 * ax + (1 if p[ax] > 0 else 0)]
                other = [abs(p[a]) for a in range(3) if a != ax]
                info["blind"].append(dict(half_width=S.W, least_gap=float(gv.min()), at=[float(v) for v in p], face=face,
                                          distance_to_the_nearest_edge_of_the_face=float(S.W - max(other)),
                                          nodes_at_or_under_0p2=int((gv <= DG.GAP_MIN).sum()), nodes=int(len(gv)),
                                          surface_reason=r["surfaces"][k]["reason"]))
        calls.append(info)
        return r

    H.read_plus = read_plus
    a = types.SimpleNamespace(work_dir=work, out_dir=data, power=2)
    H.run_chain(name, a)
    rec = json.loads((work / "chains" / f"{name}.json").read_text())
    old = json.loads((WORK / "chains" / f"{name}.json").read_text())
    rows = sorted(rec["reads"] + rec["dense"], key=lambda q: q["it"])
    for q, c in zip(rows, calls):
        assert q["cls"] == c["cls"], (q["it"], q["cls"], c["cls"])

    def plaq(q):
        return int(sum(s[0] for s in q["sets"]) + q["frag"][1])

    out = []
    for k, (q, c) in enumerate(zip(rows, calls)):
        if not c["blind"]:
            continue
        prev = next((rows[j] for j in range(k - 1, -1, -1) if "B" not in rows[j]["cls"]), None)
        nxt = next((rows[j] for j in range(k + 1, len(rows)) if "B" not in rows[j]["cls"]), None)
        out.append(dict(it=q["it"], cls=q["cls"], q=q["q"], pierced_plaquettes=c["pierced"], sets=q["n"], fragments=q["frag"], blind=c["blind"],
                        read_before=dict(it=prev["it"], cls=prev["cls"], pierced_plaquettes=plaq(prev), sets=prev["n"], fragments=prev["frag"]) if prev else None,
                        read_after=dict(it=nxt["it"], cls=nxt["cls"], pierced_plaquettes=plaq(nxt), sets=nxt["n"], fragments=nxt["frag"]) if nxt else None))
    same = dict(iterations=[rec["iterations"], old["iterations"]], U=[rec["end"]["U"], old["end"]["U"]], ended=[rec["ended"], old["ended"]],
                dense=[len(rec["dense"]), len(old["dense"])], same_classes=[r_["cls"] for r_ in sorted(old["reads"] + old["dense"], key=lambda q: q["it"])]
                == [q["cls"] for q in rows])
    res = dict(chain=name, the_rerun_against_the_pool_record=same, blind_reads=out,
               note="a descriptor outside the plan: no label is read from it; the least gap is over the nodes of the surface, the bound of a liftable read is 0.2")
    (DIAG / f"{name}_blind_reads.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(same))
    for b in out:
        print(f"it {b['it']}: cls {b['cls']} pierced {b['pierced_plaquettes']} (before: it {b['read_before']['it'] if b['read_before'] else None} "
              f"{b['read_before']['pierced_plaquettes'] if b['read_before'] else None}; after: it {b['read_after']['it'] if b['read_after'] else None} "
              f"{b['read_after']['pierced_plaquettes'] if b['read_after'] else None})")
        for s in b["blind"]:
            print(f"   surface {s['half_width']}: least gap {s['least_gap']:.4f} at {s['at']} (face {s['face']}, {s['distance_to_the_nearest_edge_of_the_face']:.2f} from its edge), "
                  f"{s['nodes_at_or_under_0p2']} of {s['nodes']} nodes at or under 0.2")


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        main(nm)
