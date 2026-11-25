"""M5.49: the bands predicted from the background, written before any wave is read along a ray.

This script opens base states only (the end states of the base chains B0a to B0d of M5.46 and B0e of this task)
and the tracked summary of M5.47. It has no code path that opens a Fourier sum.

THE LOCAL MASS (prediction F), at a frequency w in (4, 5.5):

    m_F^2(x) = m_mid^2 / q(x),   m_mid^2 = 60.0891 (the mid mass of the coded Hessian at the vacuum, M5.47)
    q(x)   = 1 + (kappa / L1) sum_j ( |[W, d_j^+ M0]|^2 + |[W, d_j^- M0]|^2 ) / 2
    W(x)   = w1 n n^T + w2 (1 - n n^T)/sqrt 2          the mid polarization of M5.47 (two weights)
    q_P(x) = the same sum with P(x) = p1 e_major e_major^T + p2 e_middle e_middle^T + p3 e_minor e_minor^T,
             (p1, p2, p3) the middle eigenvector of the Hessian of the potential in the three eigenvalues at
             the vacuum (1, delta, 0), in the eigenframe of the background (three weights)

    the band of F at w: where q is above m_mid^2 / w^2 (3.7556 at 4, 1.9864 at 5.5).

THE LINES, from the loop's center c, unit normal nu and radius R (the reader of M5.44 on the base state):

    A    the two rays along nu and -nu
    PF   the four rays in the plane through c normal to nu, along the two box axes of that plane, both ways
    PD   the four rays of that plane at 45 degrees from those of PF
    O    the eight rays at 45 degrees from nu, above and below the four directions of PF

THE PIECES. With s the coordinate along a ray and t the distance from it, a line holds the nodes with s >= 0
and t <= max(t0, s tan 15 deg) for one of its rays; a node inside the sets of two rays is counted once, with the
ray it is nearest to (the smaller s on a tie). Piece i holds s from i D to (i + 1) D, center (i + 1/2) D.
Spacing 0.25: D = 0.5, t0 = 1.0, floor 150 nodes. Spacing 0.5: D = 1.0, t0 = 1.5, floor 100 nodes.
The domain: nodes at least 1.5 inside every face where the two largest eigenvalues of the background are at
least 0.5 apart. A piece is read when its nodes in the domain reach the floor and are at least 90 percent of
its nodes. Only the nodes of the domain enter a mean.

THE FILE m5_49_bands_predicted.json, per grid and line:

    F on the pieces   m_F(s_i) = sqrt(m_mid^2 / mean q over the piece), joined by straight lines between centers
    F on the ray      q interpolated trilinearly at points half a spacing apart on each ray, averaged over the rays
    F with q_P        both forms again
    F' (no label)     sqrt of the mean over the piece of v V''_WW / (2 L1 q), the local mass of M5.47
    C (no label)      the cutoffs M5.47 fitted per bin of q on the same grid (grid d for grid e), interpolated
                      linearly in the mean q of the piece; no value outside the range of the fitted bins
    the geometry      the loop; the nodes of every piece and how many lie in the domain; where each line stops
                      being read; for every edge of F its window and whether the read pieces hold it

Every grid is one block of the file. The hash of a block is the SHA-256 of its JSON text, keys sorted, separators
"," and ":", UTF-8, floats written with seven significant digits. m5_49_reads.py takes the hash of each grid it
opens as an argument and stops on a mismatch.

Usage:  python m5_49_predict.py --grids a b c d V1 V2
        python m5_49_predict.py --grids e
        python m5_49_predict.py --hashes
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from scipy.ndimage import map_coordinates

import m5_42_eq42 as Q
import m5_44_loops as L44
import m5_47_evolve as E

M_MID2 = 60.0891
OMEGAS = (4.0, 5.5)
INSET = 1.5
GAP = 0.5
TAN = float(np.tan(np.radians(15.0)))
SHARE = 0.9
SPEC = {0.25: dict(D=0.5, t0=1.0, floor=150), 0.5: dict(D=1.0, t0=1.5, floor=100)}
GRIDS = dict(a=dict(base="B0a", hb=8.0, N=33, file="m5_46_B0a_end.npz", cut="a"), b=dict(base="B0b", hb=8.0, N=65, file="m5_46_B0b_end.npz", cut="b"),
             c=dict(base="B0c", hb=10.0, N=41, file="m5_46_B0c_end.npz", cut="c"), d=dict(base="B0d", hb=10.0, N=81, file="m5_46_B0d_end.npz", cut="d"),
             e=dict(base="B0e", hb=13.0, N=105, file="m5_49_B0e_end.npz", cut="d"))
VACUUM = dict(V1=dict(hb=8.0, N=33), V2=dict(hb=6.0, N=49))
LINES = ("A", "PF", "PD", "O")
REGISTERED = {"4.0": dict(A=None, plane=(2.77, 6.31)), "5.5": dict(A=(0.81, 8.38), plane=(1.56, 9.74))}


# ----------------------------------------------------------------------------- pure functions (also used by the reads)

def three_weights(delta=E.DELTA):
    """The middle eigenvector of the Hessian of the potential in the three eigenvalues at the vacuum (1, delta, 0):
    its weights on (major, middle, minor), the first one positive."""
    lam = np.array([1.0, delta, 0.0])
    J = np.array([k * lam ** (k - 1) for k in (1, 2, 3)])
    _, vec = np.linalg.eigh(2 * J.T @ J)
    p = vec[:, 1]
    return p * np.sign(p[0])


def line_rays(nu):
    """The unit directions of the rays of the four lines, for a loop of unit normal nu."""
    nu = np.asarray(nu, float) / np.linalg.norm(nu)
    k = int(np.argmax(np.abs(nu)))
    ax = [np.eye(3)[i] for i in range(3) if i != k]
    u = ax[0] - (ax[0] @ nu) * nu
    u /= np.linalg.norm(u)
    v = np.cross(nu, u)
    if v @ ax[1] < 0:
        v = -v
    r2 = np.sqrt(2.0)
    pf = [u, -u, v, -v]
    return dict(A=[nu, -nu], PF=pf, PD=[(u + v) / r2, (u - v) / r2, (-u + v) / r2, (-u - v) / r2],
                O=[(s * nu + d) / r2 for s in (1.0, -1.0) for d in pf])


def assign(d, rays, D, t0):
    """Per node: the ray of the line it is counted with (-1: none) and its piece along that ray."""
    r2 = (d * d).sum(-1)
    best_t = np.full(r2.shape, np.inf)
    best_s = np.full(r2.shape, np.inf)
    ray = np.full(r2.shape, -1, dtype=np.int16)
    for k, e in enumerate(rays):
        s = d @ e
        t = np.sqrt(np.maximum(r2 - s * s, 0.0))
        mem = (s >= -1e-9) & (t <= np.maximum(t0, s * TAN) + 1e-9)
        better = mem & ((t < best_t - 1e-9) | ((np.abs(t - best_t) <= 1e-9) & (s < best_s - 1e-9)))
        best_t = np.where(better, t, best_t)
        best_s = np.where(better, s, best_s)
        ray = np.where(better, k, ray)
    piece = np.where(ray >= 0, np.floor(np.where(ray >= 0, best_s, 0.0) / D + 1e-9), -1).astype(np.int32)
    return ray, piece


def crossings(s, y, level, max_gap=None):
    """The radii where the profile y(s), joined by straight lines between consecutive finite points (no more than
    max_gap apart when given), crosses the level. 'outer': y goes from under to over as s grows."""
    s, y = np.asarray(s, float), np.asarray(y, float)
    idx = [i for i in range(len(s)) if np.isfinite(y[i])]
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        if max_gap is not None and s[b] - s[a] > max_gap + 1e-9:
            continue
        oa, ob = y[a] >= level, y[b] >= level
        if oa != ob:
            r = s[a] + (level - y[a]) / (y[b] - y[a]) * (s[b] - s[a])
            out.append(dict(r=float(r), kind="outer" if ob else "inner", between=[int(a), int(b)], gap=float(s[b] - s[a])))
    return out


def window(r0, centers, D):
    """The pieces whose centers lie from 0.9 r0 - D to 1.1 r0 + D; complete when the line has pieces out to its end."""
    lo, hi = 0.9 * r0 - D, 1.1 * r0 + D
    idx = [i for i, c in enumerate(centers) if lo - 1e-9 <= c <= hi + 1e-9]
    complete = bool(len(centers) and centers[-1] + D >= hi - 1e-9)
    return idx, complete


def f_statement(m_pieces, read, m_ray, w, margin):
    """The statement of F on a line at w: NO BAND when the least m_F is at least (1 + margin) w in both forms; BAND when
    two consecutive read pieces have m_F at most (1 - margin) w and the ray form has a band; else None."""
    mp = np.array([m for m, r in zip(m_pieces, read) if r and m is not None and np.isfinite(m)], float)
    mr = np.asarray([m for m in m_ray if m is not None and np.isfinite(m)], float)
    if mp.size == 0 or mr.size == 0:
        return None
    if mp.min() >= (1 + margin) * w and mr.min() >= (1 + margin) * w:
        return "NO BAND"
    under = [bool(r and m is not None and np.isfinite(m) and m <= (1 - margin) * w) for m, r in zip(m_pieces, read)]
    if any(a and b for a, b in zip(under[:-1], under[1:])) and (mr < w).any():
        return "BAND"
    return None


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(f"{float(o):.7g}")
    return o


def block_hash(block):
    return hashlib.sha256(json.dumps(block, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- the background of a grid

def comm2(A, B):
    C = A @ B - B @ A
    return (C * C).sum((-1, -2))


def kinetic(M, Wd, h):
    s = np.zeros(M.shape[:3])
    for j in range(3):
        Dp = (np.roll(M, -1, axis=j) - M) / h
        Dm = (np.roll(M, 1, axis=j) - M) / h
        s += 0.5 * (comm2(Wd, Dp) + comm2(Wd, Dm))
    return 1 + E.KAPPA * s / E.L1


def background(g, data, with_A=True):
    """Everything the prediction and the read need from the base state of a grid."""
    spec = GRIDS[g]
    z = np.load(data / spec["file"], allow_pickle=False)
    m0 = np.ascontiguousarray(z["m6"])
    hb, N = spec["hb"], spec["N"]
    assert m0.shape == (N, N, N, 6) and abs(z["x"][-1] - hb) < 1e-12
    x = np.linspace(-hb, hb, N)
    h = x[1] - x[0]
    M = Q.from6(m0)
    W, lam, n = E.polarization(m0)
    lam_e, vec = np.linalg.eigh(M)
    p = three_weights()
    P = sum(p[k] * np.einsum("...i,...j->...ij", vec[..., 2 - k], vec[..., 2 - k]) for k in range(3))
    q, qP = kinetic(M, W, h), kinetic(M, P, h)
    eps = 1e-4
    Vww = (E.potential(M + eps * W) - 2 * E.potential(M) + E.potential(M - eps * W)) / eps ** 2
    m2loc = E.V * Vww / (2 * E.L1 * q)
    loop = L44.read_state(m0, x, second=False)
    lp = loop["loop"]
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    inside = (np.abs(X) <= hb - INSET + 1e-9).all(-1)
    dom = inside & ((lam[..., 2] - lam[..., 1]) >= GAP)
    out = dict(grid=g, base=spec["base"], m0=m0, hb=hb, N=N, h=h, x=x, X=X, W=W, W6=Q.to6(W), lam=lam, q=q, qP=qP, m2loc=m2loc, inside=inside, dom=dom,
               center=np.array(lp["center"]), normal=np.array(lp["normal"]), radius=float(lp["radius"]), loops=int(loop["loops"]),
               three_weights=p, two_weights=E.mid_weights()[0])
    if with_A:
        F = E.Field(N, hb)
        A = F.a_matrices(m0)
        qA = np.einsum("...a,...ab,...b->...", out["W6"], A, out["W6"]) / (2 * E.L1)
        out["q_from_A"] = float(np.abs(qA - q)[dom].max())
        out["q_from_A_rel"] = float((np.abs(qA - q) / q)[dom].max())
    return out


def vacuum_background(name):
    hb, N = VACUUM[name]["hb"], VACUUM[name]["N"]
    x = np.linspace(-hb, hb, N)
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    inside = (np.abs(X) <= hb - INSET + 1e-9).all(-1)
    return dict(grid=name, hb=hb, N=N, h=x[1] - x[0], x=x, X=X, inside=inside, dom=inside, center=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
                q=np.ones((N, N, N)), qP=np.ones((N, N, N)), m2loc=np.full((N, N, N), M_MID2))


def pieces_of(G):
    """Per line: the ray and the piece of every node, and the table of the pieces."""
    sp = SPEC[round(G["h"], 6)]
    rays = line_rays(G["normal"])
    d = G["X"] - G["center"]
    out = {}
    for ln in LINES:
        ray, piece = assign(d, rays[ln], sp["D"], sp["t0"])
        n = int(piece.max()) + 1
        on = piece >= 0
        tot = np.bincount(piece[on], minlength=n)
        ind = on & G["dom"]
        cnt = np.bincount(piece[ind], minlength=n)
        read = (cnt >= sp["floor"]) & (cnt >= SHARE * tot)
        out[ln] = dict(rays=rays[ln], ray=ray, piece=piece, n=n, nodes=tot, nodes_dom=cnt, read=read, centers=(np.arange(n) + 0.5) * sp["D"], D=sp["D"],
                       sel=ind)
    return out


def piece_mean(L, field):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.bincount(L["piece"][L["sel"]], weights=field[L["sel"]], minlength=L["n"]) / L["nodes_dom"]


def ray_profile(G, rays, field):
    """The field interpolated trilinearly at points half a spacing apart on each ray, inside the inset, averaged over the rays."""
    h, hb = G["h"], G["hb"]
    smax = min(float(np.min((hb - INSET - np.sign(e) * G["center"]) / np.maximum(np.abs(e), 1e-12))) for e in rays)
    s = np.arange(0.0, smax + 1e-9, h / 2)
    vals = []
    for e in rays:
        pts = G["center"] + s[:, None] * e
        idx = ((pts - G["x"][0]) / h).T
        vals.append(map_coordinates(field, idx, order=1, mode="nearest"))
    return s, np.mean(vals, 0), np.array(vals)


def cutoffs_of_m5_47(data, g):
    """(mean q, cutoff) of the fitted bins of M5.47 (its section 11) on a grid."""
    r = json.loads((data / "m5_47_reads.json").read_text())
    fits = [f for f in r["added_cut"]["grids"][g]["fits"] if f["fitted"] and f["m2"] > 0]
    return np.array([f["q_mean"] for f in fits]), np.sqrt(np.array([f["m2"] for f in fits]))


def edges_block(s, m, D, read=None):
    out = {}
    for w in OMEGAS:
        y = np.where(read, m, np.nan) if read is not None else m
        out[str(w)] = crossings(s, y, w)
    return out


def predict_grid(g, data):
    G = background(g, data)
    L = pieces_of(G)
    qb, cb = cutoffs_of_m5_47(data, GRIDS[g]["cut"])
    order = np.argsort(qb)
    block = dict(grid=g, base=G["base"], N=G["N"], half_box=G["hb"], spacing=G["h"], m_mid2=M_MID2,
                 loop=dict(center=G["center"], normal=G["normal"], radius=G["radius"], loops=G["loops"]),
                 two_weights=G["two_weights"], three_weights=G["three_weights"], domain_nodes=int(G["dom"].sum()),
                 inset_nodes=int(G["inside"].sum()), C3=dict(max_abs=G["q_from_A"], max_rel=G["q_from_A_rel"], passed=bool(G["q_from_A"] <= 1e-12)),
                 cutoffs_of_m5_47=dict(grid=GRIDS[g]["cut"], q_mean=qb[order], cutoff=cb[order]), lines={})
    for ln in LINES:
        Ln = L[ln]
        qm, qPm, f2 = piece_mean(Ln, G["q"]), piece_mean(Ln, G["qP"]), piece_mean(Ln, G["m2loc"])
        with np.errstate(divide="ignore", invalid="ignore"):
            mF, mFP = np.sqrt(M_MID2 / qm), np.sqrt(M_MID2 / qPm)
            mFp = np.sqrt(np.where(f2 > 0, f2, np.nan))
        C = np.where((qm >= qb.min()) & (qm <= qb.max()), np.interp(qm, qb[order], cb[order]), np.nan)
        nr = len(Ln["rays"])
        rq = np.full((nr, Ln["n"]), np.nan)
        rn = np.zeros((nr, Ln["n"]), int)
        for k in range(nr):
            sel = Ln["sel"] & (Ln["ray"] == k)
            rn[k] = np.bincount(Ln["piece"][sel], minlength=Ln["n"])
            with np.errstate(divide="ignore", invalid="ignore"):
                rq[k] = np.bincount(Ln["piece"][sel], weights=G["q"][sel], minlength=Ln["n"]) / rn[k]
        s, qr, _ = ray_profile(G, Ln["rays"], G["q"])
        _, qPr, _ = ray_profile(G, Ln["rays"], G["qP"])
        mr, mPr = np.sqrt(M_MID2 / qr), np.sqrt(M_MID2 / qPr)
        read = Ln["read"]
        last = int(np.flatnonzero(read).max()) if read.any() else -1
        ed = dict(pieces=edges_block(Ln["centers"], mF, Ln["D"], read), ray=edges_block(s, mr, Ln["D"]),
                  pieces_qP=edges_block(Ln["centers"], mFP, Ln["D"], read), ray_qP=edges_block(s, mPr, Ln["D"]))
        for w in OMEGAS:
            for e in ed["pieces"][str(w)]:
                idx, complete = window(e["r"], Ln["centers"], Ln["D"])
                e["window"] = idx
                e["window_read"] = bool(complete and all(read[i] for i in idx))
        block["lines"][ln] = dict(rays=np.array(Ln["rays"]), D=Ln["D"], pieces=Ln["n"], read_to=(last + 1) * Ln["D"],
                                  centers=Ln["centers"], nodes=Ln["nodes"], nodes_domain=Ln["nodes_dom"], read=read, q=qm, qP=qPm, mF=mF, mF_qP=mFP,
                                  mF_prime=mFp, C=C, ray_q=rq, ray_nodes=rn, ray_form=dict(s=s, q=qr, qP=qPr, mF=mr, mF_qP=mPr), edges=ed,
                                  least_mF=dict(pieces=float(np.nanmin(np.where(read, mF, np.nan))) if read.any() else None, ray=float(mr.min()),
                                                pieces_qP=float(np.nanmin(np.where(read, mFP, np.nan))) if read.any() else None, ray_qP=float(mPr.min()),
                                                ray_at=float(s[int(np.argmin(mr))])))
    return jsonable(block)


def predict_vacuum(name):
    G = vacuum_background(name)
    L = pieces_of(G)
    block = dict(grid=name, N=G["N"], half_box=G["hb"], spacing=G["h"], lines={})
    for ln in LINES:
        Ln = L[ln]
        last = int(np.flatnonzero(Ln["read"]).max()) if Ln["read"].any() else -1
        block["lines"][ln] = dict(rays=np.array(Ln["rays"]), D=Ln["D"], pieces=Ln["n"], read_to=(last + 1) * Ln["D"], centers=Ln["centers"],
                                  nodes=Ln["nodes"], nodes_domain=Ln["nodes_dom"], read=Ln["read"])
    return jsonable(block)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", nargs="*", default=[])
    ap.add_argument("--hashes", action="store_true")
    ap.add_argument("--force", action="store_true", help="rewrite a block that is already in the file")
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    a = ap.parse_args()
    f = a.data / "m5_49_bands_predicted.json"
    doc = json.loads(f.read_text()) if f.exists() else dict(
        task="M5.49", what="the bands predicted from the background of each grid (prediction F), with the geometry of the lines and pieces",
        m_mid2=M_MID2, omegas=list(OMEGAS), q_threshold={str(w): M_MID2 / w ** 2 for w in OMEGAS}, registered=REGISTERED,
        pieces=dict(half_angle_degrees=15.0, share=SHARE, inset=INSET, eigenvalue_gap=GAP, per_spacing={str(k): v for k, v in SPEC.items()}),
        hash_rule="SHA-256 of the JSON text of a block: keys sorted, separators ',' and ':', UTF-8, floats with seven significant digits",
        blocks={}, hashes={})
    for g in a.grids:
        if g in doc["blocks"] and not a.force:
            raise SystemExit(f"the block of grid {g} is already in the file; it is not rewritten without --force")
        block = predict_vacuum(g) if g in VACUUM else predict_grid(g, a.data)
        doc["blocks"][g] = block
        doc["hashes"][g] = block_hash(block)
        print(g, "block", doc["hashes"][g], flush=True)
        if g in GRIDS:
            print("   loop", block["loop"], "C3", block["C3"], flush=True)
            for ln in LINES:
                Lb = block["lines"][ln]
                print("   ", ln, "pieces", Lb["pieces"], "read to", Lb["read_to"], "first nodes", Lb["nodes"][:3], "least mF", Lb["least_mF"])
                for w in OMEGAS:
                    for k in ("pieces", "ray", "pieces_qP", "ray_qP"):
                        print("       w", w, k, [(round(e["r"], 3), e["kind"]) for e in Lb["edges"][k][str(w)]])
        else:
            for ln in LINES:
                Lb = block["lines"][ln]
                print("   ", ln, "pieces", Lb["pieces"], "read to", Lb["read_to"], "first nodes", Lb["nodes"][:3])
    if a.grids:
        f.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
    if a.grids or a.hashes:
        for g, hsh in doc["hashes"].items():
            assert block_hash(doc["blocks"][g]) == hsh, g
            print("hash", g, hsh)
        print("file", hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
