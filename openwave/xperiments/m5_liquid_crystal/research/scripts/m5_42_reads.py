"""M5.42 R7: the reads on the end state of every path (m5_42_task_details.md, section 2, rows 7 and 9; section 5).

For each end state:

  where the inertia sits   inside r < 2, in 2 <= r < 4, in 4 <= r < half-box, beyond the inscribed sphere;
                           the share within one tenth of the half-box of a face
  alignment                the share of the nodes of the shell 2 < r < 5 whose long axis (the eigenvector of
                           the largest eigenvalue) lies within 26 degrees of rhat
  the minor gap            lam_mid - lam_min on the three central planes: its largest value and where
  eigenvalues on lines     along +x, +z and the diagonal (1, 1, 1), at the nodes of the 0.375 spacing
  winding and punctures    on the spheres r = 3, 4, 5 and 0.95 of the half-box (a cubed sphere of 6 x 15 x 15 cells, the field
                           interpolated linearly): the degree of the long axis as a line field (lifted to a
                           vector field face by face; defined up to its sign), the count of +1/2 and -1/2
                           turns of the minor axis around the cells, and the smallest gap between the two
                           larger eigenvalues on the sphere (the long axis is not defined where it closes)

Controls (--controls): the winding reads return degree 1 and four +1/2 punctures on the author's frame, and
degree 0 with no puncture on a uniform field.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_42_eq42 as Q  # noqa: E402
from m5_42_relax import chains  # noqa: E402

NC = 15  # cells per edge of a face of the cubed sphere


def cubed_sphere():
    """Vertices (6, NC+1, NC+1, 3) on the unit sphere and, per face, the two in-face directions."""
    u = np.linspace(-1, 1, NC + 1)
    A, B = np.meshgrid(u, u, indexing="ij")
    one = np.ones_like(A)
    faces = []
    for ax in range(3):
        for sgn in (1.0, -1.0):
            P = [None, None, None]
            P[ax], P[(ax + 1) % 3], P[(ax + 2) % 3] = sgn * one, A, B
            if sgn < 0:  # keep every face's vertex order counterclockwise seen from outside
                P[(ax + 1) % 3], P[(ax + 2) % 3] = B, A
            V = np.stack(P, -1)
            faces.append(V / np.linalg.norm(V, axis=-1, keepdims=True))
    return np.stack(faces)


def sample(M6, x, pts):
    """The six components interpolated linearly at pts (..., 3), as 3 x 3 matrices."""
    h = x[1] - x[0]
    c = [(pts[..., a] - x[0]) / h for a in range(3)]
    out = np.stack([map_coordinates(M6[..., k], c, order=1, mode="nearest") for k in range(6)], -1)
    return Q.from6(out)


def solid_angle(a, b, c):
    num = np.einsum("...i,...i->...", a, np.cross(b, c))
    den = 1 + (a * b).sum(-1) + (b * c).sum(-1) + (c * a).sum(-1)
    return 2 * np.arctan2(num, den)


def sphere_reads(M6, x, radius, V=None):
    V = cubed_sphere() if V is None else V
    M = sample(M6, x, radius * V)
    lam, vec = np.linalg.eigh(M)
    long_axis, minor = vec[..., 2], vec[..., 1]
    # the degree of the long axis. A line field on a sphere without defects lifts to a vector field: each
    # face is oriented by a sweep (first row, then down the columns), and the faces are matched to the first
    # one at the vertices they share. A defect of the long axis on the sphere breaks the lift and shows as a
    # degree that is not an integer.
    long_axis = long_axis.copy()
    for f in range(6):
        for a in range(1, NC + 1):
            if (long_axis[f, a, 0] * long_axis[f, a - 1, 0]).sum() < 0:
                long_axis[f, a, 0] *= -1
        for b in range(1, NC + 1):
            flip = (long_axis[f, :, b] * long_axis[f, :, b - 1]).sum(-1) < 0
            long_axis[f, flip, b] *= -1
    done = [0]
    todo = [1, 2, 3, 4, 5]
    lift_failed = False
    while todo:
        before = len(todo)
        for f in list(todo):
            votes = 0.0
            for g in done:
                dist = np.linalg.norm(V[f].reshape(-1, 1, 3) - V[g].reshape(1, -1, 3), axis=-1)
                i, j = np.nonzero(dist < 1e-9)
                votes += np.sign((long_axis[f].reshape(-1, 3)[i] * long_axis[g].reshape(-1, 3)[j]).sum(-1)).sum()
            if votes != 0.0:
                long_axis[f] *= np.sign(votes)
                done.append(f)
                todo.remove(f)
        if len(todo) == before:  # the shared vertices vote evenly: the long axis has a defect on this sphere
            lift_failed = True
            done += todo
            todo = []
    total = 0.0
    for tri in (((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1))):
        n = [long_axis[:, da:da + NC, db:db + NC] for da, db in tri]
        total += solid_angle(n[0], n[1], n[2]).sum()
    degree = total / (4 * np.pi)
    # the punctures of the minor axis: the turn of its tangent-plane angle (mod pi) around each cell
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    P = [V[:, da:da + NC, db:db + NC] for da, db in corners]
    center = sum(P)
    center /= np.linalg.norm(center, axis=-1, keepdims=True)
    e1 = P[1] - P[0]
    e1 -= (e1 * center).sum(-1, keepdims=True) * center
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(center, e1)
    ang = []
    for da, db in corners:
        d = minor[:, da:da + NC, db:db + NC]
        ang.append(np.arctan2((d * e2).sum(-1), (d * e1).sum(-1)))
    turn = np.zeros_like(ang[0])
    for q in range(4):
        dphi = ang[(q + 1) % 4] - ang[q]
        turn += (dphi + np.pi / 2) % np.pi - np.pi / 2
    idx = np.rint(turn / np.pi).astype(int)  # in units of 1/2
    return dict(radius=radius, degree_of_long_axis=float(degree), lift_failed=lift_failed, punctures_plus_half=int((idx == 1).sum()), punctures_minus_half=int((idx == -1).sum()),
                punctures_other=int((np.abs(idx) > 1).sum()), smallest_gap_of_the_two_larger_eigenvalues=float((lam[..., 2] - lam[..., 1]).min()),
                smallest_minor_gap=float((lam[..., 1] - lam[..., 0]).min()))


def state_reads(job):
    chain, path, spec, f = job
    z = np.load(f)
    m6, x, J = z["m6"], z["x"], float(z["J"])
    mod = Q.Model(spec["N"], spec["half_box"], spec["delta"], spec["L1"], stencil=spec["stencil"], fixed_chart=spec["fixed_chart"],
                  free_faces=spec["free_faces"], band=spec["band"])
    p = mod.evaluate(m6, J, need_grad=False, need_I=True)
    h3, r, hb, idn = mod.h ** 3, mod.r, mod.half_box, mod.idn
    I = p["I"]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    near_face = np.maximum(np.maximum(np.abs(X), np.abs(Y)), np.abs(Z)) > 0.9 * hb
    lam, vec = np.linalg.eigh(Q.from6(m6))
    shell = (r > 2) & (r < 5)
    cosang = np.abs((vec[..., 2] * mod.n).sum(-1))
    c = spec["N"] // 2
    gap = lam[..., 1] - lam[..., 0]
    planes = dict(x=gap[c], y=gap[:, c], z=gap[:, :, c])
    worst = max(planes, key=lambda k: planes[k].max())
    at = np.unravel_index(int(np.argmax(planes[worst])), planes[worst].shape)
    diag = [(c + q, c + q, c + q) for q in range(c + 1)]
    out = dict(chain=chain, path=path, file=Path(f).name, J=J, I=I, U=p["U"],
               inertia=dict(inside_r2=float(h3 * idn[r < 2].sum()), r2_to_4=float(h3 * idn[(r >= 2) & (r < 4)].sum()),
                            r4_to_half_box=float(h3 * idn[(r >= 4) & (r < hb)].sum()), beyond_inscribed_sphere=float(h3 * idn[r >= hb].sum()),
                            share_within_a_tenth_of_a_face=float(h3 * idn[near_face].sum() / I)),
               aligned_share_shell_2_5=float((cosang[shell] > np.cos(np.radians(26))).mean()),
               minor_gap=dict(largest=float(planes[worst].max()), plane=worst, at=[float(x[q]) for q in at], on_the_frame=spec["delta"]),
               eigenvalues_along_plus_x=np.round(lam[c:, c, c][::max(1, (spec["N"] - 1) // 32)], 5).tolist(),
               eigenvalues_along_plus_z=np.round(lam[c, c, c:][::max(1, (spec["N"] - 1) // 32)], 5).tolist(),
               eigenvalues_along_diagonal=[np.round(lam[q], 5).tolist() for q in diag[::max(1, (spec["N"] - 1) // 32)]],
               spheres=[sphere_reads(m6, x, rad) for rad in (3.0, 4.0, 5.0, 0.95 * hb) if rad < hb])
    if spec["band"]:
        b = spec["band"]
        edge = ((np.abs(lam - b["hi"]) < 0.05) | (np.abs(lam - b["lo"]) < 0.05) | (lam > b["hi"]) | (lam < b["lo"])).any(-1)
        out["band"] = dict(nodes_at_an_edge=int(edge.sum()), share_of_the_inertia_on_them=float(h3 * idn[edge].sum() / I),
                           largest_excursion=float(max((lam - b["hi"]).max(), (b["lo"] - lam).max(), 0.0)))
    return out


def controls():
    mod = Q.Model(33, 6.0, 0.3, 0.3, fixed_chart=True)
    frame = sphere_reads(Q.to6(mod.frame), mod.x, 4.0)
    uni = np.zeros(mod.shape6)
    uni[..., 0], uni[..., 3] = 1.0, 0.3  # eigenvalues (1, 0.3, 0) along fixed axes
    flat = sphere_reads(uni, mod.x, 4.0)
    ok = (abs(abs(frame["degree_of_long_axis"]) - 1) < 1e-6 and frame["punctures_plus_half"] - frame["punctures_minus_half"] == 4 and frame["punctures_plus_half"] == 4
          and abs(flat["degree_of_long_axis"]) < 1e-6 and flat["punctures_plus_half"] + flat["punctures_minus_half"] == 0)
    return dict(frame=frame, uniform=flat, met=bool(ok))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--chains-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_42" / "chains")
    ap.add_argument("--out", type=Path, default=HERE.parent / "data" / "m5_42_r7_reads.json")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    ctl = controls()
    print("winding controls:", json.dumps(ctl, indent=1))
    if a.controls:
        sys.exit(0 if ctl["met"] else 1)
    C = chains()
    jobs = []
    for name, c in C.items():
        f = a.chains_dir / f"{name}.json"
        if not f.exists():
            continue
        for p in json.loads(f.read_text())["paths"]:
            if not p.get("skipped"):
                jobs.append((name, p["name"], c["model"], str(a.data_dir / p["end_file"])))
    if a.workers <= 1:
        rows = []
        for q, job in enumerate(jobs):
            rows.append(state_reads(job))
            if q % 25 == 0:
                print(q, "of", len(jobs), flush=True)
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(state_reads, jobs))
    text = json.dumps(dict(controls=ctl, states=rows), indent=1)
    text = re.sub(r"\[\s+(-?[0-9.e-]+),\s+(-?[0-9.e-]+),\s+(-?[0-9.e-]+)\s+\]", r"[\1, \2, \3]", text)  # eigenvalue triples on one line
    a.out.write_text(text + "\n")
    print(len(rows), "states ->", a.out)


if __name__ == "__main__":
    main()
