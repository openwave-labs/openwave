"""M5.39 R2: the reads on a relaxed Card C state (a state file of m5_39_relax.py).

    python3 m5_39_reads.py <state.npz> --archive <hashed copy of the card's code folder> [--out reads.json]

Reads, in the card's own units (length 1 / (e f_pi), the unit the text gives as 0.5014 fm):

  energy     the seven terms, our evaluator (m5_39_cardc.py);
  inertia    the six-generator kinetic matrix by the card's own audit (inertia.py of the archive, run
             as an isolated subprocess on our file), and our own isospin block for the covariant left
             action, written from the same definition on the same cell-midpoint rule: the two must
             agree, and ours carries the density that the isovector radius needs;
  radii      <r^2> weighted by the winding density of the card's interpolant (isoscalar) and by the
             isospin-inertia density (isovector);
  tail       the dipole amplitude A of the static pion field (the vector part of q) against the free
             massive dipole of the lattice operator, - Laplacian + beta^2 on the same cube with
             Dirichlet faces, normalized so that its continuum form is x (1 + m r) exp(-m r) / r^3;
             fitted shell by shell;
  lumps      the nodes whose largest record eigenvalue stands above the card's target scale (see
             `record_lumps`): a leftover of the seed that the force gate does not see;
  mirror     for each coordinate reflection, the distance between the state and its mirror image on
             the fields q, N and s, and the energy of the image by term (m5_39_cardc.mirror).

Each read is a definition stated here; where the text prints a number without its definition, the
task document says which of these it is compared with.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m5_39_cardc as C  # noqa: E402

OFFSETS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
EPS = np.zeros((3, 3, 3))
for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    EPS[i, j, k], EPS[i, k, j] = 1.0, -1.0
GEN = np.zeros((3, 4, 4))  # the isospin generators on the winding field: rotations of its vector part
GEN[:, 1:, 1:] = -EPS


def cells(x):
    """The eight corner views of every cell."""
    n = x.shape[0]
    return [x[a:a + n - 1, b:b + n - 1, c:c + n - 1] for a, b, c in OFFSETS]


def midpoint(x, h):
    corners = cells(x)
    value = sum(corners) / 8.0
    deriv = [sum((2 * o[k] - 1) * z for o, z in zip(OFFSETS, corners)) / (4.0 * h) for k in range(3)]
    return value, deriv


def isospin_inertia(state, coords, h, lam=1.0):
    """Our isospin block for the left action, on the cell-midpoint rule, with its density.

    The kinetic metric is the Lorentz completion of the two terms that the left action moves:
    2 x (1/4) |delta N|^2 and 2 x (1/2) sum_i (|delta q|^2 |d_i q|^2 - (delta q . d_i q)^2), with
    delta q = T_a q and delta N = [T_a, N].
    """
    p = state[..., :4]
    f = state[..., 4:].reshape(state.shape[:3] + (3, 3))
    qm, dq = midpoint(p / np.linalg.norm(p, axis=-1, keepdims=True), h)
    norm = np.linalg.norm(qm, axis=-1, keepdims=True)
    qm = qm / norm
    dq = [(d - qm * np.sum(qm * d, axis=-1, keepdims=True)) / norm for d in dq]
    fm, _ = midpoint(f, h)
    xm, _ = midpoint(coords, h)
    q0, q1, q2, q3 = np.moveaxis(qm, -1, 0)
    tau = np.stack((np.stack((-q1, q0, q3, -q2), -1), np.stack((-q2, -q3, q0, q1), -1), np.stack((-q3, q2, -q1, q0), -1)), -1)
    a = tau @ fm
    N = -qm[..., :, None] * qm[..., None, :] + lam * a @ np.swapaxes(a, -1, -2) / C.G_TIME
    zq = [qm @ GEN[k].T for k in range(3)]
    zN = [GEN[k] @ N - N @ GEN[k] for k in range(3)]
    dens = np.zeros(qm.shape[:3] + (3, 3))
    for i in range(3):
        for j in range(i, 3):
            d = 2.0 * C.TENSOR * np.sum(zN[i] * zN[j], axis=(-2, -1))
            for u in dq:
                d = d + 2.0 * C.QUARTIC * (np.sum(zq[i] * zq[j], -1) * np.sum(u * u, -1) - np.sum(u * zq[i], -1) * np.sum(u * zq[j], -1))
            dens[..., i, j] = dens[..., j, i] = d * h ** 3
    trace = np.trace(dens, axis1=-2, axis2=-1) / 3.0
    r2 = np.sum(xm * xm, axis=-1)
    r = np.sqrt(r2)
    shells = np.arange(0.0, r.max() + 0.5, 0.5)
    inside = [float(trace[r <= s].sum()) for s in shells]
    return dict(matrix=dens.sum(axis=(0, 1, 2)).tolist(), mean=float(trace.sum()), isovector_r2=float(np.sum(trace * r2) / trace.sum()),
                inertia_inside_radius=dict(radius=shells.tolist(), inertia=inside))


def tetrahedron_rule(k=3):
    """A Gauss rule on the reference tetrahedron (Duffy map of the cube), weights summing to 1/6."""
    x, w = np.polynomial.legendre.leggauss(k)
    x, w = (x + 1) / 2, w / 2
    pts, wts = [], []
    for (a, wa), (b, wb), (c, wc) in itertools.product(zip(x, w), repeat=3):
        l1, l2, l3 = a, b * (1 - a), c * (1 - a) * (1 - b)
        pts.append((1 - l1 - l2 - l3, l1, l2, l3))
        wts.append(wa * wb * wc * (1 - a) ** 2 * (1 - b))
    return np.array(pts), np.array(wts)


def winding_density(state, coords, h):
    """The winding density of the card's own interpolant, its total and its mean square radius.

    On each simplex of the first Kuhn triangulation the field is the affine interpolant P of the four
    vertex values, projected to the sphere; its winding density is det[v0, v1, v2, v3] / (2 pi^2 |P|^4)
    per unit of the simplex's reference volume form, integrated here by a Gauss rule. The total is the
    winding number, an integer for an admissible state.
    """
    p = state[..., :4]
    q = p / np.linalg.norm(p, axis=-1, keepdims=True)
    n = q.shape[0]
    pts, wts = tetrahedron_rule()
    total, second = 0.0, 0.0
    for o in range(6):
        sl = [tuple(slice(int(C.OFFSETS[0, o, m, a]), n - 1 + int(C.OFFSETS[0, o, m, a])) for a in range(3)) for m in range(4)]
        v = [q[s] for s in sl]
        x = [coords[s] for s in sl]
        det = np.linalg.det(np.stack(v, axis=-1))
        sign = -np.sign(np.linalg.det(np.array([C.OFFSETS[0, o, m + 1] - C.OFFSETS[0, o, 0] for m in range(3)], dtype=float)))
        for lam, w in zip(pts, wts):
            P = sum(c * vv for c, vv in zip(lam, v))
            X = sum(c * xx for c, xx in zip(lam, x))
            dens = sign * w * det / np.sum(P * P, axis=-1) ** 2 / (2.0 * np.pi ** 2)
            total += float(dens.sum())
            second += float(np.sum(dens * np.sum(X * X, axis=-1)))
    return dict(total=total, isoscalar_r2=second / total)


def record_lumps(state, coords, threshold=0.2):
    """Nodes whose largest record eigenvalue stands above the card's target scale c(q) = (1 - q0) / 2.

    The seeds start with a record eigenvalue of one on every interior node. A node left near one
    after the relaxation is a lattice-scale lump, a local minimum of its own: the composite fields
    x1 = s (s - 1)(s - delta c) and x2 = s x1 vanish at an eigenvalue of one as they do at zero, so
    their difference terms resist the node's way down while the potential pulls it, and the two
    balance near 0.96. The node then has a small gradient and carries about h^3 (lambda^2 +
    lambda^6) / 4 of potential energy. A state with lumps is not comparable with one without at that
    level, so every read reports their number.
    """
    p = state[..., :4]
    q0 = p[..., 0] / np.linalg.norm(p, axis=-1)
    f = state[..., 4:].reshape(state.shape[:3] + (3, 3))
    lam = np.linalg.eigvalsh(np.swapaxes(f, -1, -2) @ f)[..., 2]
    excess = lam - 0.5 * (1.0 - q0)
    idx = np.argwhere(excess > threshold)
    h = float(coords[1, 0, 0, 0] - coords[0, 0, 0, 0])
    r = np.linalg.norm(coords, axis=-1)
    cost = float(sum(0.25 * h ** 3 * (lam[tuple(i)] ** 2 + lam[tuple(i)] ** 6) for i in idx))
    return dict(count=int(len(idx)), largest_excess=float(excess.max()), potential_energy_of_the_lumps=cost,
                nodes=[dict(index=[int(x) for x in i], radius=float(r[tuple(i)]), eigenvalue=float(lam[tuple(i)])) for i in idx[:12]])


def lattice_dipole(n, h, beta2):
    """The three free massive dipole fields of - Laplacian_h + beta^2 on the cube, Dirichlet faces.

    Source at the center node: 4 pi times the centered difference, in the source point, of a unit point
    source along the axis, so that the continuum, infinite-box field is x_i (1 + m r) exp(-m r) / r^3.
    The mass is the card's: far from the core the energy is (1/2)(|d pi|^2 + beta^2 pi^2), m^2 = beta^2.
    """
    m = n - 2
    k = np.arange(1, m + 1)
    lam = (2.0 - 2.0 * np.cos(np.pi * k / (m + 1))) / h ** 2
    denom = lam[:, None, None] + lam[None, :, None] + lam[None, None, :] + beta2
    c = (n - 1) // 2 - 1  # the center in interior indexing
    fields = []
    for axis in range(3):
        src = np.zeros((m, m, m))
        up, dn = [c, c, c], [c, c, c]
        up[axis] += 1
        dn[axis] -= 1
        src[tuple(up)] = 4.0 * np.pi / (2.0 * h * h ** 3)
        src[tuple(dn)] = -4.0 * np.pi / (2.0 * h * h ** 3)
        phi = idstn(dstn(src, type=1) / denom, type=1)
        full = np.zeros((n, n, n))
        full[1:-1, 1:-1, 1:-1] = phi
        fields.append(full)
    return np.stack(fields, axis=-1)


def tail_amplitude(state, coords, h):
    """A per shell: the least-squares amplitude of the vector part of q on the lattice dipole."""
    n = state.shape[0]
    p = state[..., :4]
    pion = (p / np.linalg.norm(p, axis=-1, keepdims=True))[..., 1:]
    phi = lattice_dipole(n, h, C.BETA2)
    r = np.linalg.norm(coords, axis=-1)
    box = coords[-1, 0, 0, 0]
    edges = np.arange(1.0, box + 1e-9, 0.25)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (r >= lo) & (r < hi)
        if sel.sum() < 20:
            continue
        num, den = float(np.sum(pion[sel] * phi[sel])), float(np.sum(phi[sel] ** 2))
        amp = num / den
        resid = float(np.sqrt(np.sum((pion[sel] - amp * phi[sel]) ** 2) / np.sum(pion[sel] ** 2)))
        # the same shell against the continuum, infinite-box dipole, for the size of the box effect
        m = np.sqrt(C.BETA2)
        cont = coords[sel] * ((1 + m * r[sel]) * np.exp(-m * r[sel]) / r[sel] ** 3)[:, None]
        rows.append(dict(r_lo=float(lo), r_hi=float(hi), nodes=int(sel.sum()), amplitude=amp, relative_misfit=resid,
                         amplitude_against_the_continuum_dipole=float(np.sum(pion[sel] * cont) / np.sum(cont ** 2))))
    return rows


def mirror_reads(state, card):
    q0, s0 = card.fields(state)
    q0, s0, N0 = q0.copy(), s0.copy(), card.N.copy()
    t0 = card(state, fields_done=True)[0].copy()
    out = []
    for axis in range(3):
        image = C.mirror(state, axis)
        t1 = card(image)[0].copy()
        dq, dN, ds = card.q - q0, card.N - N0, card.s - s0
        out.append(dict(axis=axis, max_q=float(np.abs(dq).max()), max_N=float(np.abs(dN).max()), max_s=float(np.abs(ds).max()),
                        rms_q=float(np.sqrt(np.mean(np.sum(dq * dq, -1)))), rms_s=float(np.sqrt(np.mean(np.sum(ds * ds, (-2, -1))))),
                        energy_of_the_image=float(t1.sum()), terms_of_the_image=t1.tolist(),
                        energy_difference=float(t1.sum() - t0.sum()), term_differences=(t1 - t0).tolist()))
    return out


def archived_inertia(path, grid, archive):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    cmd = [sys.executable, "-E", "-s", str(archive / "inertia.py"), str(path), "--card", "C", "--delta", "0.2",
           "--boxes", str(grid), str(grid + 8), "--actions", "left", "q_only", "q_left_right", "record_right"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(archive))
    if p.returncode != 0:
        return dict(error=p.stderr.strip().splitlines()[-1:] or ["failed"])
    doc = json.loads(p.stdout)
    run = doc["runs"][0]
    out = dict(order=doc["order"], padding_check="passed: eight more nodes of vacuum change no entry", actions={})
    for name, act in run["actions"].items():
        full = np.array(act["full"])
        out["actions"][name] = dict(full=act["full"], terms=act["terms"],
                                    isospin_mean=float(np.trace(full[3:, 3:]) / 3), spin_mean=float(np.trace(full[:3, :3]) / 3),
                                    mixed_mean=float(np.trace(full[:3, 3:]) / 3), isospin_eigenvalues=np.linalg.eigvalsh(full[3:, 3:]).tolist(),
                                    spin_eigenvalues=np.linalg.eigvalsh(full[:3, :3]).tolist(),
                                    grand_spin_largest_entry=float(np.abs(np.array(act["grand_spin"])).max()))
    return out


def reads(path, archive=None, mirror=True):
    with np.load(path) as data:
        state, coords = data["state"], data["coords"]
    n = state.shape[0]
    h = float(coords[1, 0, 0, 0] - coords[0, 0, 0, 0])
    box = float(coords[-1, 0, 0, 0])
    card = C.CardC(n, box)
    terms, grad = card(state)
    out = dict(file=Path(path).name, grid=n, half_box=box, spacing=h, energy=float(terms.sum()), terms=terms.tolist(),
               residual=float(np.abs(C.interior(grad)).max() / h ** 3), center_eigenvalues=C.center_eigenvalues(state).tolist())
    out["isospin_inertia_ours"] = isospin_inertia(state, coords, h)
    out["winding"] = winding_density(state, coords, h)
    out["record_lumps"] = record_lumps(state, coords)
    out["tail"] = tail_amplitude(state, coords, h)
    if mirror:
        out["mirror"] = mirror_reads(state, card)
    if archive is not None:
        out["inertia_audit"] = archived_inertia(Path(path).resolve(), n, archive)
        if "actions" in out["inertia_audit"]:
            theirs = np.array(out["inertia_audit"]["actions"]["left"]["full"])[3:, 3:]
            ours = np.array(out["isospin_inertia_ours"]["matrix"])
            out["isospin_block_ours_against_the_audit"] = float(np.abs(ours - theirs).max() / np.abs(theirs).max())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state", type=Path)
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-mirror", action="store_true")
    a = ap.parse_args()
    out = reads(a.state, a.archive, not a.no_mirror)
    text = json.dumps(out, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    brief = {k: out[k] for k in ("file", "grid", "energy", "residual", "center_eigenvalues")}
    brief["isospin_inertia_ours"] = out["isospin_inertia_ours"]["mean"]
    brief["isovector_r2"] = out["isospin_inertia_ours"]["isovector_r2"]
    brief["winding"] = out["winding"]
    if "inertia_audit" in out and "actions" in out["inertia_audit"]:
        brief["audit_left"] = {k: out["inertia_audit"]["actions"]["left"][k] for k in ("isospin_mean", "spin_mean", "mixed_mean")}
        brief["ours_against_audit"] = out["isospin_block_ours_against_the_audit"]
    brief["tail_outer"] = [(r["r_lo"], round(r["amplitude"], 4), round(r["amplitude_against_the_continuum_dipole"], 4), round(r["relative_misfit"], 3)) for r in out["tail"]]
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
