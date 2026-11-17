"""M5.44: the half-turn line of the long axis in the saved end states of M5.42.

The field is a real symmetric 3 x 3 matrix M(x) on an N^3 grid of spacing h, stored as six components
(xx, xy, xz, yy, yz, zz). With eigenvalues l1 <= l2 <= l3 and n the unit eigenvector of l3 (the long axis,
defined up to sign):

    PIERCED   a plaquette with corners a, b, c, d (in order) is pierced when
              sign(n_a.n_b) sign(n_b.n_c) sign(n_c.n_d) sign(n_d.n_a) = -1
              (the director comes back reversed: a half turn of the long axis around the plaquette)
    loop      a connected set of pierced plaquettes, two being adjacent when their centers are at most
              1.05 h apart; fewer than 8 plaquettes is a FRAGMENT
    center    c = mean of the plaquette centers p_k of the loop
    normal    nu = the direction of least spread of p_k - c (last right singular vector)
    radius    R = mean_k | (p_k - c) - ((p_k - c).nu) nu |          spread = its standard deviation
    off       sqrt( mean_k ((p_k - c).nu)^2 )
    gap       g = l3 - l2, read on the loop from the mean of M over the four corners of each plaquette
    second    for 72 azimuths in the loop's plane, the place of the smallest g of the trilinearly
    method    interpolated field on a half plane (rho, zeta) through the axis c + zeta nu; its mean rho

Nothing is relaxed. Usage:  python m5_44_loops.py [--research-dir DIR] [--out-dir DIR]
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

IDX = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])
PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
FRAGMENT = 8
SMALL_GAP = 0.05


def to6(M):
    return np.stack([M[..., a, b] for a, b in PAIRS], -1)


def eigen(m6):
    """Eigenvalues (ascending) and the long-axis director at every node."""
    w, v = np.linalg.eigh(m6[..., IDX])
    return w, v[..., :, 2]


def pierced_plaquettes(m6, x):
    """Centers of the pierced plaquettes, and the gap l3 - l2 of the corner-mean field on each."""
    _, n = eigen(m6)
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    centers, gaps = [], []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        def corner(arr, da, db):
            sl = [slice(None)] * 3
            sl[a] = slice(da, arr.shape[a] - 1 + da)
            sl[b] = slice(db, arr.shape[b] - 1 + db)
            return arr[tuple(sl)]
        ring = [(0, 0), (1, 0), (1, 1), (0, 1)]
        c = [corner(n, *d) for d in ring]
        s = np.ones(c[0].shape[:3])
        for i in range(4):
            s *= np.sign(np.einsum("...k,...k", c[i], c[(i + 1) % 4]))
        hit = s < 0
        centers.append(((corner(X, 0, 0) + corner(X, 1, 1)) / 2)[hit])
        mean6 = sum(corner(m6, *d) for d in ring)[hit] / 4
        w = np.linalg.eigvalsh(mean6[..., IDX]) if hit.any() else np.zeros((0, 3))
        gaps.append(w[:, 2] - w[:, 1])
    return np.concatenate(centers), np.concatenate(gaps)


def group(P, h):
    """Labels of the connected sets of pierced plaquettes (adjacent: centers at most 1.05 h apart)."""
    if len(P) == 0:
        return np.zeros(0, int)
    pr = cKDTree(P).query_pairs(1.05 * h, output_type="ndarray")
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(P), len(P)))
    return connected_components(g, directed=False)[1]


def loop_stats(Q):
    c = Q.mean(0)
    d = Q - c
    nu = np.linalg.svd(d, full_matrices=False)[2][2]
    nu = nu * np.sign(nu[np.argmax(np.abs(nu))])
    off = d @ nu
    rad = np.linalg.norm(d - np.outer(off, nu), axis=1)
    return dict(count=int(len(Q)), center=c.tolist(), normal=nu.tolist(), radius=float(rad.mean()),
                spread=float(rad.std()), off_plane=float(np.sqrt((off ** 2).mean())),
                radius_3d=float(np.linalg.norm(d, axis=1).mean()))


def sample6(m6, x, pts):
    h = x[1] - x[0]
    c = [(pts[..., a] - x[0]) / h for a in range(3)]
    return np.stack([map_coordinates(m6[..., k], c, order=1, mode="nearest") for k in range(6)], -1)


def second_method(m6, x, loop):
    """Mean radius of the smallest gap l3 - l2 of the interpolated field over 72 half planes."""
    h = x[1] - x[0]
    c, nu, R = np.array(loop["center"]), np.array(loop["normal"]), loop["radius"]
    e1 = np.cross(nu, [1.0, 0.0, 0.0] if abs(nu[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nu, e1)
    zmax = 3 * loop["off_plane"] + 1.0
    rho = np.arange(max(0.5, R - 2.5), R + 2.5 + 1e-9, h / 4)
    zeta = np.arange(-zmax, zmax + 1e-9, h / 4)
    RHO, ZETA = np.meshgrid(rho, zeta, indexing="ij")
    out = []
    for phi in np.arange(72) * 2 * np.pi / 72:
        u = np.cos(phi) * e1 + np.sin(phi) * e2
        pts = c + RHO[..., None] * u + ZETA[..., None] * nu
        inside = (np.abs(pts) <= x[-1]).all(-1)
        w = np.linalg.eigvalsh(sample6(m6, x, pts)[..., IDX])
        g = np.where(inside, w[..., 2] - w[..., 1], np.inf)
        k = np.unravel_index(np.argmin(g), g.shape)
        out.append((RHO[k], ZETA[k], g[k]))
    out = np.array(out)
    return dict(radius=float(out[:, 0].mean()), spread=float(out[:, 0].std()),
                off_plane=float(np.sqrt((out[:, 1] ** 2).mean())), gap_mean=float(out[:, 2].mean()),
                gap_max=float(out[:, 2].max()))


def read_state(m6, x, second=True):
    h = x[1] - x[0]
    w, _ = eigen(m6)
    P, g = pierced_plaquettes(m6, x)
    lab = group(P, h)
    sets = sorted((np.flatnonzero(lab == i) for i in range(lab.max() + 1 if len(lab) else 0)), key=lambda s: -len(s))
    loops = []
    for s in sets:
        if len(s) >= FRAGMENT:
            st = loop_stats(P[s])
            st.update(gap_mean=float(g[s].mean()), gap_max=float(g[s].max()))
            loops.append(st)
    res = dict(pierced=int(len(P)), loops=len(loops), fragments=int(sum(len(s) < FRAGMENT for s in sets)),
               fragment_plaquettes=int(sum(len(s) for s in sets if len(s) < FRAGMENT)),
               small_gap_share=float(((w[..., 2] - w[..., 1]) < SMALL_GAP).mean()), loop=None, other_loops=loops[1:4])
    if loops:
        res["loop"] = loops[0]
        res["loop"]["radius_over_half_box"] = loops[0]["radius"] / x[-1]
        if second:
            res["loop"]["second_method"] = second_method(m6, x, loops[0])
        res["_points"] = P[sets[0]]
    return res


# ----------------------------------------------------------------------------- controls

def manufactured(R0, nu, h, half_box=9.0, delta=0.3):
    """M = n n^T + delta m m^T with n = cos(psi/2) rho_hat + sin(psi/2) nu, psi the angle around the circle."""
    N = int(round(2 * half_box / h)) + 1
    x = np.linspace(-half_box, half_box, N)
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    nu = np.asarray(nu, float) / np.linalg.norm(nu)
    zeta = X @ nu
    inpl = X - zeta[..., None] * nu
    rho = np.linalg.norm(inpl, axis=-1)
    rho_hat = inpl / (rho[..., None] + 1e-300)
    psi = np.arctan2(zeta, rho - R0)
    n = np.cos(psi / 2)[..., None] * rho_hat + np.sin(psi / 2)[..., None] * nu
    m = np.cross(nu, rho_hat)
    M = np.einsum("...i,...j->...ij", n, n) + delta * np.einsum("...i,...j->...ij", m, m)
    return to6(M), x


def reflect(m6, s):
    """The mirror image M'(x) = R M(Rx) R for R = diag(s)."""
    out = m6
    for a in range(3):
        if s[a] < 0:
            out = np.flip(out, axis=a)
    sign = np.array([s[a] * s[b] for a, b in PAIRS], float)
    return np.ascontiguousarray(out * sign)


def refine(m6, x):
    x2 = np.linspace(x[0], x[-1], 2 * len(x) - 1)
    pts = np.stack(np.meshgrid(x2, x2, x2, indexing="ij"), -1)
    return sample6(m6, x, pts), x2


def controls(research, data):
    checks = []

    def add(name, value, bound, met, note=""):
        checks.append(dict(name=name, value=value, bound=bound, met=bool(met), note=note))

    for h, R0, nu in itertools.product((0.375, 0.25), (3.0, 4.25, 7.0), ((0, 0, 1), (1, -1, 0))):
        m6, x = manufactured(R0, nu, h)
        r = read_state(m6, x)
        lp = r["loop"]
        tag = "manufactured R0=%g normal=%s h=%g" % (R0, "z" if nu[2] else "(1,-1,0)/sqrt2", h)
        nun = np.asarray(nu, float) / np.linalg.norm(nu)
        ang = float(np.degrees(np.arccos(min(1.0, abs(np.dot(lp["normal"], nun)))))) if lp else None
        add(tag + ": one loop, no fragment", [r["loops"], r["fragments"]], [1, 0], r["loops"] == 1 and r["fragments"] == 0)
        add(tag + ": |radius - R0|", abs(lp["radius"] - R0) if lp else None, h / 2, lp and abs(lp["radius"] - R0) <= h / 2)
        add(tag + ": normal angle (deg)", ang, 2.0, lp and ang <= 2.0)
        add(tag + ": second method |radius - R0|", abs(lp["second_method"]["radius"] - R0) if lp else None, h,
            lp and abs(lp["second_method"]["radius"] - R0) <= h, "the gap of this field closes only on the circle")

    x = np.linspace(-6, 6, 33)
    uni = np.zeros((33, 33, 33, 6))
    uni[..., 3], uni[..., 5] = 0.3, 1.0
    r = read_state(uni, x, second=False)
    add("uniform field: pierced plaquettes", r["pierced"], 0, r["pierced"] == 0)
    sys.path.insert(0, os.path.join(research, "scripts"))
    import m5_42_eq42 as Q
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    P, _ = pierced_plaquettes(to6(Q.vacuum_frame(X, Y, Z, 0.3, True)[0]), x)
    outside = int((np.abs(P).max(-1) > (x[1] - x[0]) + 1e-9).sum()) if len(P) else 0
    add("the frame continued inward: pierced plaquettes outside the eight central cells", outside, 0, outside == 0,
        "%d pierced plaquettes inside them" % (len(P) - outside))

    for f in ("m5_42_r4_hb6_L0.3_j0_end.npz", "m5_42_r2_g33_j0_end.npz"):
        z = np.load(os.path.join(data, f), allow_pickle=False)
        m6, x = z["m6"], z["x"]
        base = read_state(m6, x, second=False)["loop"]
        worst_r, worst_n = 0.0, 0.0
        for s in itertools.product((1, -1), repeat=3):
            lp = read_state(reflect(m6, s), x, second=False)["loop"]
            worst_r = max(worst_r, abs(lp["radius"] - base["radius"]))
            worst_n = max(worst_n, 1 - abs(np.dot(np.array(lp["normal"]), np.array(s) * np.array(base["normal"]))))
        add(f + ": radius under the eight reflections", worst_r, 1e-12, worst_r <= 1e-12)
        add(f + ": 1 - |normal . reflected normal|", worst_n, 1e-12, worst_n <= 1e-12)
        m2, x2 = refine(m6, x)
        lp2 = read_state(m2, x2, second=False)
        add(f + ": radius on the sample of half the spacing", abs(lp2["loop"]["radius"] - base["radius"]), (x[1] - x[0]) / 2,
            lp2["loops"] == 1 and abs(lp2["loop"]["radius"] - base["radius"]) <= (x[1] - x[0]) / 2,
            "refined radius %.4f against %.4f" % (lp2["loop"]["radius"], base["radius"]))
    return checks


# ----------------------------------------------------------------------------- the read

def nearest(P, pos):
    return float(np.linalg.norm(P - np.asarray(pos), axis=1).min()) if P is not None and len(P) else None


def read_all(research, data):
    paths = json.load(open(os.path.join(data, "m5_42_paths.json")))
    states, points = [], {}
    for chain, rec in paths.items():
        mod = rec["model"]
        for p in rec["paths"]:
            f = p.get("end_file")
            if not f or not os.path.exists(os.path.join(data, f)):
                continue
            z = np.load(os.path.join(data, f), allow_pickle=False)
            r = read_state(z["m6"], z["x"])
            points[(chain, p["name"])] = r.pop("_points", None)
            last = p["stops"][-1]
            states.append(dict(chain=chain, path=p["name"], file=f, N=mod["N"], half_box=mod["half_box"],
                               spacing=2 * mod["half_box"] / (mod["N"] - 1), delta=mod["delta"], L1=mod["L1"],
                               stencil=mod["stencil"], fixed_chart=mod["fixed_chart"], free_faces=mod["free_faces"],
                               band=mod["band"] is not None, J=p["J"], label=p["label"], converged=p["converged"],
                               start=p.get("start"), largest_share=last["largest_share"],
                               largest_position=last["largest_position"], ratio=last["two_J_Omega_over_E"], **r))
    for s in states:
        if s["label"] != "ONE-CELL":
            continue
        s["node_to_own_loop"] = nearest(points.get((s["chain"], s["path"])), s["largest_position"])
        for ref in ("j0", "j10"):
            t = next((q for q in states if q["chain"] == s["chain"] and q["path"] == ref), None)
            if t is not None and t["label"] == "SMOOTH":
                s["node_to_%s_loop" % ref] = nearest(points.get((s["chain"], ref)), s["largest_position"])
    return states, points


def word(states):
    a = next(s for s in states if s["chain"] == "r4_hb12_L0.3" and s["path"] == "j0")["loop"]
    b = next(s for s in states if s["chain"] == "r4_hb18_L0.3" and s["path"] == "j0")["loop"]
    dr = b["radius"] / a["radius"] - 1
    ds = b["radius_over_half_box"] / a["radius_over_half_box"] - 1
    w = "SATURATES" if abs(dr) < 0.05 else "SCALES WITH THE BOX" if abs(ds) < 0.05 else "NEITHER"
    return dict(word=w, radius_change=dr, share_change=ds, radius_12=a["radius"], radius_18=b["radius"])


def plots(states, points, data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    series = (("r4_hb%d_L0.3", "L₁ = 0.3, symmetric sector", "#1f5fa8", "o"), ("r4_hb%d_L0.1", "L₁ = 0.1, symmetric sector", "#c2571a", "s"))
    for pat, name, col, mk in series:
        pts = [(hb, s["loop"]) for hb in (6, 9, 12, 18) for s in states if s["chain"] == pat % hb and s["path"] == "j0" and s["loop"]]
        ax[0].plot([p[0] for p in pts], [p[1]["radius"] for p in pts], mk + "-", color=col, lw=2, ms=7, label=name)
        ax[1].plot([p[0] for p in pts], [p[1]["radius_over_half_box"] for p in pts], mk + "-", color=col, lw=2, ms=7, label=name)
    br = [s for s in states if (s["chain"], s["path"]) in (("g3x_avg_g33", "j0f"), ("g3b_avg_hb9", "j0b")) and s["loop"]]
    ax[0].plot([s["half_box"] for s in br], [s["loop"]["radius"] for s in br], "^", color="#3a8a4a", ms=8, label="L₁ = 0.3, broken sector")
    ax[1].plot([s["half_box"] for s in br], [s["loop"]["radius_over_half_box"] for s in br], "^", color="#3a8a4a", ms=8, label="L₁ = 0.3, broken sector")
    ax[0].set_ylabel("loop radius")
    ax[1].set_ylabel("loop radius over the half-box")
    for a in ax:
        a.set_xlabel("half-box")
        a.set_xticks([6, 9, 12, 18])
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylim(0)
    ax[1].set_ylim(0, 1)
    ax[0].legend(frameon=False)
    fig.suptitle("The half-turn loop of the long axis at J = 0 (mirror-averaged stencil, spacing 0.375)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "m5_44_radius_against_box.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    for a, hb in zip(ax, (6, 9, 12, 18)):
        s = next(q for q in states if q["chain"] == "r4_hb%d_L0.3" % hb and q["path"] == "j0")
        z = np.load(os.path.join(data, s["file"]), allow_pickle=False)
        w, _ = eigen(z["m6"][:, :, len(z["x"]) // 2])
        im = a.pcolormesh(z["x"], z["x"], (w[..., 2] - w[..., 1]).T, cmap="Blues_r", vmin=0, vmax=1, shading="nearest")
        P = points[(s["chain"], s["path"])]
        a.plot(P[:, 0], P[:, 1], ".", color="#c2571a", ms=3)
        a.set_aspect("equal")
        a.set_title("half-box %d: radius %.2f" % (hb, s["loop"]["radius"]))
        a.set_xlabel("x")
    ax[0].set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.8, label="gap of the two larger eigenvalues on the plane z = 0")
    fig.suptitle("Pierced plaquettes (orange, projected on z = 0) over the gap map, J = 0, L₁ = 0.3", y=0.99)
    fig.savefig(os.path.join(out, "m5_44_gap_and_loop.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    research = here
    while not os.path.isdir(os.path.join(research, "data")) and research != "/":
        research = os.path.dirname(research)
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-dir", default=research)
    ap.add_argument("--out-dir", default=None, help="where the JSON and the plots go (default: data/ and plots/)")
    a = ap.parse_args()
    data = os.path.join(a.research_dir, "data")
    jdir = a.out_dir or data
    pdir = a.out_dir or os.path.join(a.research_dir, "plots")
    chk = controls(a.research_dir, data)
    for c in chk:
        print("%s  %s: %s (bound %s) %s" % ("met   " if c["met"] else "FAILED", c["name"], c["value"], c["bound"], c["note"]))
    states, points = read_all(a.research_dir, data)
    out = dict(controls=chk, controls_met=sum(c["met"] for c in chk), controls_total=len(chk), word=word(states), states=states)
    json.dump(out, open(os.path.join(jdir, "m5_44_loops.json"), "w"), indent=1)
    plots(states, points, data, pdir)
    print("controls met: %d of %d; states read: %d; word: %s" % (out["controls_met"], len(chk), len(states), out["word"]))


if __name__ == "__main__":
    main()
