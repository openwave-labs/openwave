"""M5.41 R3, a read added during the run: record spikes that the lump check cannot flag.

The lump check of the rule (record_lumps of m5_39_reads.py) flags a node whose largest record
eigenvalue stands more than 0.2 above the card's target scale c(q) = (1 - q0) / 2. Near the core
q0 is close to -1 and c is close to 1, so a node left at an eigenvalue near one is not flagged
there. This read uses no target scale: a spike is an interior node whose largest record eigenvalue
stands more than 0.2 above the largest of its six neighbours. It is reported beside the lump
check and changes no label.

    python3 m5_41_spikes.py [states ...] --out data/m5_41_r3_spikes.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
THRESHOLD = 0.2


def spikes(path):
    with np.load(path, allow_pickle=False) as d:
        st, co = d["state"], d["coords"]
    n = st.shape[0]
    p = st[..., :4]
    q0 = p[..., 0] / np.linalg.norm(p, axis=-1)
    f = st[..., 4:].reshape(n, n, n, 3, 3)
    lam = np.linalg.eigvalsh(np.swapaxes(f, -1, -2) @ f)[..., 2]
    r = np.linalg.norm(co, axis=-1)
    excess = lam - 0.5 * (1.0 - q0)
    inner = lam[1:-1, 1:-1, 1:-1]
    around = np.maximum.reduce([lam[2:, 1:-1, 1:-1], lam[:-2, 1:-1, 1:-1], lam[1:-1, 2:, 1:-1], lam[1:-1, :-2, 1:-1],
                                lam[1:-1, 1:-1, 2:], lam[1:-1, 1:-1, :-2]])
    idx = np.argwhere(inner - around > THRESHOLD) + 1
    flagged = excess > THRESHOLD
    nodes = [dict(index=[int(x) for x in i], radius=float(r[tuple(i)]), eigenvalue=float(lam[tuple(i)]), q0=float(q0[tuple(i)]),
                  excess_over_the_target_scale=float(excess[tuple(i)]), flagged_by_the_lump_check=bool(flagged[tuple(i)])) for i in idx]
    return dict(file=Path(path).name, grid=n, largest_eigenvalue=float(lam.max()), largest_excess_over_the_neighbours=float((inner - around).max()),
                flagged_by_the_lump_check=int(flagged.sum()),
                spikes=len(nodes), spikes_not_flagged=sum(not x["flagged_by_the_lump_check"] for x in nodes), nodes=nodes[:40])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", type=Path, nargs="*")
    ap.add_argument("--out", type=Path, default=HERE.parent / "data" / "m5_41_r3_spikes.json")
    a = ap.parse_args()
    states = a.states or sorted((HERE.parent / "data").glob("m5_41_r2_*_s[0-9]*.npz"))
    out = {}
    for p in states:
        out[p.stem] = spikes(p)
        s = out[p.stem]
        print(f"{p.stem}: lump check {s['flagged_by_the_lump_check']}, spikes {s['spikes']}, not flagged {s['spikes_not_flagged']}, largest eigenvalue {s['largest_eigenvalue']:.3f}", flush=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
