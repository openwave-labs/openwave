"""M5.50: the relaxed unit with the size of its loop held and raised on a ladder: the hold, the chains, the pool.

The action is that of M5.42 (m5_42_eq42.py) at J = 0, on the mirror-averaged stencil with the chart fixed:

    U = h^3 sum_n [ L1 sum_i |d_i M|^2 + kappa sum_{i<j} |[d_i M, d_j M]|^2 + v sum_k (tr M^k - c_k)^2 ]

with kappa = v = 1, delta = 0.05, L1 = 0.03, c_k = 1 + delta^k.

    THE HOLD     u_n = sum_k (tr M_n^k - c_k)^2,   w_n = u_n^p (p = 2),   xbar = sum w x / sum w
                 D[M] = 2 sum_n w_n |x_n - xbar| / sum_n w_n
                 E_mu = U + (mu / 2) (D - d)^2,      f = mu (d - D)
                 dD/dw_n = (2 / W) ( r_n - D / 2 - a . (x_n - xbar) ),   a = (1 / W) sum_m w_m (x_m - xbar) / r_m
                 dw_n/dM_n = p u_n^(p-1) ( 2 t1 1 + 4 t2 M + 6 t3 M^2 ),   t_k = tr M^k - c_k
    THE SECOND   E_o = U + (mu_o / 2) h^3 sum_{n in Delta(R_d)} (1 - (M_n)_zz)^2,  Delta(R_d) the nodes of the plane
    HOLD         z = 0 at a distance of at most R_d from the z axis of the box;  mu_o = 100
    THE DESCENT  L-BFGS-B over the six components of every free node, 20 correction pairs, function tolerance 1e-15;
                 after an end by the function tolerance a restart with an empty memory, until a restart ends within
                 25 iterations (40 restarts at most); a stalled line search restarted three times at most.
    THE GATE     max |dE/dm| / h^3 at the end  <=  max( 1e-4, min( 1e-3, 0.1 max |f dD/dm| / h^3 ) )
    THE WORDS    CONVERGED: not cut by a cap, the gate met, and an end by the function tolerance repeated after a
                 restart (or, with a stalled line search at the end, the gate alone). STALLED: the function
                 tolerance repeated within 25 iterations with the gate not met, after three more restarts.
    THE START    of a rung: the base dilated about its loop's center by d / D(base) (the dilation of M5.46), the
                 faces set back to the base's own face values (the frame, to the seed's 1e-7).
    THE MAPS     of the family: the radial pinch (the identity beyond R + 2) and the windowed noise, section 4 of
                 m5_50_task_details.md.
    THE READS    at iteration 0, every 10 iterations up to 300, every 25 after, and at the end (the cadence of
                 M5.46), with m5_46_degree.py unchanged; per nested surface one class: its whole number when
                 liftable, L when not liftable with a pierced triangle or a pierced plaquette within 1.0 of it
                 (at the loop), B otherwise (blind). At a read with two sets or more, three cubes centered on
                 each set. An exit chain is read in addition at every iteration from the first cadence read at
                 which its loop stands within 2 of a face or the box surface does not read 1; the EXIT rule
                 counts the cadence reads alone.

One process, one thread per chain. A chain is restartable from its checkpoint (empty optimizer memory).

Usage:  python m5_50_hold.py --controls                 (K1, K3, K4, K5, K12; needs the four base states)
        python m5_50_hold.py --chain NAME               (the spec is read from <work-dir>/chains/NAME/spec.json)
        python m5_50_hold.py --pool [--no-hold]         (the whole job list with its dependencies; restartable)
        python m5_50_hold.py --list
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402
from scipy.ndimage import gaussian_filter, map_coordinates  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_42_eq42 as Q  # noqa: E402
import m5_46_degree as DG  # noqa: E402

DELTA, L1 = 0.05, 0.03
CK = (1 + DELTA, 1 + DELTA ** 2, 1 + DELTA ** 3)
CAP = 20000
MU, MU2, MU_O, POWER = 20.0, 80.0, 100.0, 2
GRIDS = {
    "10a": dict(hb=10.0, N=41, base="m5_46_B0c_end.npz", targets=[7.75 + 0.75 * i for i in range(8)], inward=[5.5, 4.0], mu2=None, wall=0.5,
                tol=0.05, obstacle=[1.5, 4.5, 5.5]),
    "10b": dict(hb=10.0, N=81, base="m5_46_B0d_end.npz", targets=[8.5, 10.0, 11.5, 13.0], inward=[5.5, 4.0], mu2=[10.0, 13.0], wall=2.0, tol=0.10,
                obstacle=[]),
    "14a": dict(hb=14.0, N=57, base="m5_50_B14a_end.npz", targets=[7.75 + 0.75 * i for i in range(18)], inward=[], mu2=None, wall=1.0, tol=0.05,
                obstacle=[1.5, 5.5, 8.0]),
    "14b": dict(hb=14.0, N=113, base="m5_50_B14b_end.npz", targets=[8.5 + 1.5 * i for i in range(9)], inward=[], mu2=[14.5], wall=3.0, tol=0.10,
                obstacle=[]),
}
FAMILY = {"pin2e50": dict(kind="pinch", m=2, eps=0.5), "pin2e80": dict(kind="pinch", m=2, eps=0.8), "pin2e95": dict(kind="pinch", m=2, eps=0.95),
          "pin3e80": dict(kind="pinch", m=3, eps=0.8), "noise05": dict(kind="noise", amp=0.05, seed=5001), "noise20": dict(kind="noise", amp=0.2, seed=5002)}
FAMILY_113 = ("pin2e80", "pin2e95", "noise20")


# ----------------------------------------------------------------------------- the hold

@njit(cache=True)
def _hold_kernel(m6, x, c1, c2, c3, power, need_grad, g6):
    """D and, with need_grad, dD/dm over the six components (off-diagonal entries counted twice)."""
    N = m6.shape[0]
    u = np.empty((N, N, N))
    W = 0.0
    sx = 0.0
    sy = 0.0
    sz = 0.0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                a = m6[i, j, k, 0]
                b = m6[i, j, k, 1]
                c = m6[i, j, k, 2]
                d = m6[i, j, k, 3]
                e = m6[i, j, k, 4]
                f = m6[i, j, k, 5]
                xx = a * a + b * b + c * c
                xy = a * b + b * d + c * e
                xz = a * c + b * e + c * f
                yy = b * b + d * d + e * e
                yz = b * c + d * e + e * f
                zz = c * c + e * e + f * f
                t1 = a + d + f - c1
                t2 = xx + yy + zz - c2
                t3 = xx * a + yy * d + zz * f + 2.0 * (xy * b + xz * c + yz * e) - c3
                uu = t1 * t1 + t2 * t2 + t3 * t3
                u[i, j, k] = uu
                w = uu ** power
                W += w
                sx += w * x[i]
                sy += w * x[j]
                sz += w * x[k]
    cx = sx / W
    cy = sy / W
    cz = sz / W
    S = 0.0
    ax = 0.0
    ay = 0.0
    az = 0.0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                w = u[i, j, k] ** power
                dx = x[i] - cx
                dy = x[j] - cy
                dz = x[k] - cz
                r = np.sqrt(dx * dx + dy * dy + dz * dz)
                S += w * r
                if r > 1e-300:
                    ax += w * dx / r
                    ay += w * dy / r
                    az += w * dz / r
    Dv = 2.0 * S / W
    ax /= W
    ay /= W
    az /= W
    if need_grad:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    uu = u[i, j, k]
                    dx = x[i] - cx
                    dy = x[j] - cy
                    dz = x[k] - cz
                    r = np.sqrt(dx * dx + dy * dy + dz * dz)
                    co = (2.0 / W) * (r - 0.5 * Dv - (ax * dx + ay * dy + az * dz)) * power * uu ** (power - 1)
                    a = m6[i, j, k, 0]
                    b = m6[i, j, k, 1]
                    c = m6[i, j, k, 2]
                    d = m6[i, j, k, 3]
                    e = m6[i, j, k, 4]
                    f = m6[i, j, k, 5]
                    xx = a * a + b * b + c * c
                    xy = a * b + b * d + c * e
                    xz = a * c + b * e + c * f
                    yy = b * b + d * d + e * e
                    yz = b * c + d * e + e * f
                    zz = c * c + e * e + f * f
                    t1 = a + d + f - c1
                    t2 = xx + yy + zz - c2
                    t3 = xx * a + yy * d + zz * f + 2.0 * (xy * b + xz * c + yz * e) - c3
                    g6[i, j, k, 0] = co * (2.0 * t1 + 4.0 * t2 * a + 6.0 * t3 * xx)
                    g6[i, j, k, 1] = co * 2.0 * (4.0 * t2 * b + 6.0 * t3 * xy)
                    g6[i, j, k, 2] = co * 2.0 * (4.0 * t2 * c + 6.0 * t3 * xz)
                    g6[i, j, k, 3] = co * (2.0 * t1 + 4.0 * t2 * d + 6.0 * t3 * yy)
                    g6[i, j, k, 4] = co * 2.0 * (4.0 * t2 * e + 6.0 * t3 * yz)
                    g6[i, j, k, 5] = co * (2.0 * t1 + 4.0 * t2 * f + 6.0 * t3 * zz)
    return Dv, W, cx, cy, cz, u


def hold_D(m6, x, power=POWER, need_grad=False, g6=None, about_box_center=False):
    """D of the field, the weight sum, the weight's center, the density u and (with need_grad) dD/dm in g6."""
    if g6 is None:
        g6 = np.zeros(m6.shape if need_grad else (1, 1, 1, 6))
    Dv, W, cx, cy, cz, u = _hold_kernel(np.ascontiguousarray(m6), x, CK[0], CK[1], CK[2], power, need_grad, g6)
    if about_box_center:        # the wrong form, for the counter-input of K4: distances taken from the box center
        X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
        w = u ** power
        Dv = float(2.0 * (w * np.linalg.norm(X, axis=-1)).sum() / w.sum())
    return dict(D=float(Dv), W=float(W), center=[float(cx), float(cy), float(cz)], u=u, grad=g6 if need_grad else None)


def obstacle(m6, x, Rd, mu_o=MU_O, need_grad=False):
    """(mu_o / 2) h^3 sum over the disk of (1 - M_zz)^2 and its gradient (the zz component alone)."""
    N, h = len(x), x[1] - x[0]
    k0 = (N - 1) // 2
    X, Y = np.meshgrid(x, x, indexing="ij")
    disk = (X * X + Y * Y) <= Rd * Rd + 1e-12
    dz = (1.0 - m6[:, :, k0, 5]) * disk
    E = 0.5 * mu_o * h ** 3 * float((dz * dz).sum())
    g = None
    if need_grad:
        g = np.zeros(m6.shape)
        g[:, :, k0, 5] = -mu_o * h ** 3 * dz
    return E, g, int(disk.sum())


# ----------------------------------------------------------------------------- the starts and the maps

def dilate(m6, x, center, s):
    """M'(x) = M(c + (x - c) / s) on the same grid, by trilinear interpolation of the six components (M5.46)."""
    h = x[1] - x[0]
    idx = [(center[a] + (x - center[a]) / s - x[0]) / h for a in range(3)]
    g = np.meshgrid(*idx, indexing="ij")
    return np.stack([map_coordinates(m6[..., k], g, order=1, mode="nearest") for k in range(6)], -1)


def set_faces(m6, src):
    """The six faces of m6 set to those of src."""
    for sl in ((0,), (-1,), (slice(None), 0), (slice(None), -1), (slice(None), slice(None), 0), (slice(None), slice(None), -1)):
        m6[sl] = src[sl]
    return m6


def pinch(m6, x, center, R, eps, m, width=2.0):
    """The radial pinch: the new field at (rho, phi, z) is the old one at (sigma, phi, z); the identity beyond R + width.
    Returns (field, number of source points outside the box)."""
    h = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    dx, dy = X - center[0], Y - center[1]
    rho, phi = np.hypot(dx, dy), np.arctan2(dy, dx)
    s = 1.0 - eps * np.cos(m * phi / 2.0) ** 2
    inner = rho <= s * R
    mid = (~inner) & (rho <= R + width)
    sig = rho.copy()
    sig[inner] = rho[inner] / s[inner]
    sig[mid] = R + (rho[mid] - s[mid] * R) * width / (R + width - s[mid] * R)
    sx, sy = center[0] + sig * np.cos(phi), center[1] + sig * np.sin(phi)
    out = (np.abs(sx) > x[-1] + 1e-9) | (np.abs(sy) > x[-1] + 1e-9)
    if eps == 0.0:
        return m6.copy(), int(out.sum())
    idx = [(sx - x[0]) / h, (sy - x[0]) / h, (Z - x[0]) / h]
    new = np.stack([map_coordinates(m6[..., k], idx, order=1, mode="nearest") for k in range(6)], -1)
    return new, int(out.sum())


def noise(m6, x, center, normal, R, amp, seed):
    """Six white Gaussian fields smoothed by a Gaussian of width 1, windowed by exp(-d^2 / 4.5) about the loop's
    fitted circle, scaled to the rms amp over the six components of the nodes with d < 1.5."""
    h, N = x[1] - x[0], len(x)
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1) - np.asarray(center)
    nrm = np.asarray(normal) / np.linalg.norm(normal)
    zp = X @ nrm
    rp = np.linalg.norm(X - zp[..., None] * nrm, axis=-1)
    d = np.hypot(rp - R, zp)
    rng = np.random.default_rng(seed)
    f = np.stack([gaussian_filter(rng.standard_normal((N, N, N)), 1.0 / h, mode="nearest") for _ in range(6)], -1) * np.exp(-d * d / 4.5)[..., None]
    rms = float(np.sqrt((f[d < 1.5] ** 2).mean()))
    return m6 + (amp / rms) * f if amp != 0.0 else m6.copy()


def apply_map(m6, x, mp, lp):
    if mp["kind"] == "pinch":
        return pinch(m6, x, lp["center"], lp["radius"], mp["eps"], mp["m"])
    return noise(m6, x, lp["center"], lp["normal"], lp["radius"], mp["amp"], mp["seed"]), 0


# ----------------------------------------------------------------------------- the reads

def sets_of(lab, P):
    """The plaquettes of each set of 8 or more, in the order of m5_46_degree.line_read (largest first)."""
    idx = sorted((np.flatnonzero(lab == i) for i in range(lab.max() + 1 if len(lab) else 0)), key=lambda s: -len(s))
    return [P[s] for s in idx if len(s) >= DG.FRAGMENT]


def dist_to_cube(P, w):
    """Distance of each point to the surface of the centered cube of half-width w."""
    a = np.abs(P)
    mx = a.max(1)
    return np.where(mx <= w, w - mx, np.linalg.norm(np.maximum(a - w, 0.0), axis=1))


def read_plus(m6, x, surfaces):
    """m5_46_degree.read_state with the plaquettes kept; adds the class of every surface and the cubes about each set."""
    n, gap = DG.director(m6)
    line, P, lab = DG.line_read(m6, x)
    k = DG.the_loop(line["sets"])
    surf, cls = [], []
    for S in surfaces:
        r = S.read(n, gap)
        r["surrounds"] = [i for i, s in enumerate(line["sets"]) if s["max_norm"] < S.W - 1e-9]
        r["around_the_loop"] = bool(k is not None and k in r["surrounds"])
        r["line_crossings"] = DG.crossings(P, S.W)
        if r["liftable"]:
            cls.append(str(min(r["abs_q"], 9)))
        else:
            near = r["pierced_triangles"] > 0 or (len(P) > 0 and float(dist_to_cube(P, S.W).min()) <= 1.0 + 1e-9)
            cls.append("L" if near else "B")
        surf.append(r)
    out = dict(line=line, loop=k, clear=[i for i, s in enumerate(line["sets"]) if not s["touches"]], surfaces=surf, cls="".join(cls), P=P, lab=lab, cubes=None)
    if len(line["sets"]) >= 2:
        N, h = len(x), x[1] - x[0]
        cubes = []
        for st, Pk in zip(line["sets"], sets_of(lab, P)):
            c = np.rint((np.asarray(st["center"]) - x[0]) / h).astype(int)
            ext = float(np.abs(Pk - (x[0] + c * h)).max())
            row = []
            for w0 in (0.5, 1.0, 1.5):
                w = np.ceil((ext + w0) / 0.5 - 1e-9) * 0.5
                mm = int(round(w / h))
                if (c - mm).min() < 0 or (c + mm).max() > N - 1:
                    row.append([float(w), "out"])
                    continue
                r = DG.Surface(N, c - mm, c + mm, half_width=float(w)).read(n, gap)
                row.append([float(w), str(min(r["abs_q"], 9)) if r["liftable"] else ("x" if r["pierced_triangles"] else "g" if r["reason"] == "gap" else "l")])
            cubes.append(row)
        out["cubes"] = cubes
    return out


def weight_shares(u, x, sets_P, power=POWER, radius=1.5):
    """The share of the weight u^p within `radius` of the plaquettes of each set, and of all of them."""
    w = u ** power
    tot = float(w.sum())
    big = np.argwhere(w > 1e-14 * w.max())
    xyz = x[big]
    wv = w[big[:, 0], big[:, 1], big[:, 2]]
    shares, anyset = [], np.zeros(len(big), bool)
    for Pk in sets_P:
        dmin, _ = cKDTree(Pk).query(xyz, distance_upper_bound=radius)
        inside = np.isfinite(dmin)
        anyset |= inside
        shares.append(float(wv[inside].sum() / tot))
    return shares, float(wv[anyset].sum() / tot)


def compact(it, p, r, seconds):
    code = dict(pierced="x", gap="g", lift="l")
    q = "".join(str(min(s["abs_q"], 9)) if s["liftable"] else code[s["reason"]] for s in r["surfaces"])
    sets = [[s["count"], round(s["radius"], 4), *[round(v, 3) for v in s["center"]], *[round(v, 3) for v in s["normal"]],
             round(s["least_distance_to_faces"], 3), sum(1 << f for f in s["faces"]), round(s["max_norm"], 3), round(s["spread"], 3),
             round(s["off_plane"], 3)] for s in r["line"]["sets"][:8]]
    row = dict(it=it, U=float(f"{p['U']:.9g}"), g=float(f"{p['gmax']:.4g}"), t=round(seconds, 1), loop=r["loop"], n=len(r["line"]["sets"]),
               clear=len(r["clear"]), sets=sets, frag=[r["line"]["fragments"], r["line"]["fragment_plaquettes"]], q=q, cls=r["cls"],
               ar="".join("1" if s["around_the_loop"] else "0" for s in r["surfaces"]), Ut=[float(f"{p[k]:.9g}") for k in ("U2", "U4", "UV")])
    if p.get("D") is not None:
        row.update(D=float(f"{p['D']:.9g}"), f=float(f"{p['f']:.6g}"), Eh=float(f"{p['Eh']:.6g}"))
    if p.get("Eo") is not None:
        row.update(Eo=float(f"{p['Eo']:.6g}"))
    if r["cubes"] is not None:
        row["cubes"] = r["cubes"]
    return row


def due(it):
    return it % 10 == 0 if it <= 300 else it % 25 == 0


# ----------------------------------------------------------------------------- one chain

def after_function_tolerance(made, n_ends, g_now, gate, stalls):
    """The rule of M5.45 as M5.48 states it: restart, end, or a stall (restarted three times at most)."""
    if made <= 25 and g_now > gate:
        return "stalled, restart" if stalls < 3 else "stalled, end"
    if made <= 25 or n_ends >= 40:
        return "end"
    return "restart"


def load_state(f):
    z = np.load(f, allow_pickle=False)
    return z["m6"].astype(float), z["x"]


def one_loop(m6, x):
    line = DG.line_read(m6, x)[0]
    k = DG.the_loop(line["sets"])
    if k is None:
        raise SystemExit("the source state has no set of 8 plaquettes or more")
    return line["sets"][k], len(line["sets"])


def build_start(spec, mod, out_dir):
    """The start field of a chain, with what it was made from."""
    s = spec["start"]
    if s["kind"] == "seed":
        return mod.seed6().copy(), dict(kind="seed")
    m6, x = load_state(out_dir / s["file"])
    assert len(x) == spec["N"] and abs(x[-1] - spec["hb"]) < 1e-12
    info = dict(kind="state", file=s["file"])
    src = m6
    if s.get("dilate") or s.get("map"):
        lp, nsets = one_loop(m6, x)
        info.update(source_radius=lp["radius"], source_center=lp["center"], source_normal=lp["normal"], source_sets=nsets)
    if s.get("dilate"):
        d = s["dilate"]
        D0 = hold_D(m6, x, spec.get("power", POWER))["D"]
        ratio = d["to_D"] / D0 if "to_D" in d else d["to_radius"] / lp["radius"] if "to_radius" in d else d["ratio"]
        info.update(dilation=ratio, source_D=D0)
        if abs(ratio - 1.0) > 1e-15:
            m6 = dilate(m6, x, lp["center"], ratio)
    if s.get("map"):
        m6, nout = apply_map(m6, x, s["map"], lp)
        info.update(map=s["map"], sources_outside_the_box=nout)
    if spec["faces"] == "held" and m6 is not src:
        m6 = set_faces(m6.copy(), src)
    return m6.copy(), info


def run_chain(name, a):
    work = a.work_dir / "chains" / name
    work.mkdir(parents=True, exist_ok=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads((work / "spec.json").read_text())
    logf = open(work / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {name} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    guard = None
    try:
        sys.path.insert(0, os.path.expanduser("~/Library/Application Support/UPSSentinel"))
        import ups_guard
        ups_guard.register()
        guard = ups_guard
    except Exception as e:  # noqa: BLE001
        log(f"no power guard: {e}")

    jf = a.work_dir / "chains" / f"{name}.json"
    if jf.exists() and json.loads(jf.read_text()).get("finished"):
        log("already finished")
        return
    (work / "pid").write_text(str(os.getpid()))
    free = spec["faces"] == "free"
    hold = spec.get("hold")
    exit_rule, dense_allowed = bool(spec.get("exit_rule")), bool(spec.get("dense"))
    power = spec.get("power", POWER)
    cap, wall_hours = spec.get("cap", CAP), spec["wall"]
    mod = Q.Model(spec["N"], spec["hb"], DELTA, L1, stencil="averaged", fixed_chart=True, free_faces=free)
    surfaces = DG.nested(mod.x)
    h3 = mod.h ** 3
    g6 = np.zeros(mod.shape6)
    ckpt, state_f = work / "ckpt.npz", work / "state.json"
    if ckpt.exists() and state_f.exists():
        st = json.loads(state_f.read_text())
        x = np.load(ckpt)["m6"].reshape(-1)
        st["resumed_at"].append(st["iteration"])
        log(f"resumes at it {st['iteration']}")
    else:
        m6, info = build_start(spec, mod, a.out_dir)
        x = m6.reshape(-1).copy()
        st = dict(iteration=0, reads=[], dense=[], events=[], ftol_ends=0, stalls=0, abnormal=0, resumed_at=[], nfev=0, seconds=0.0, exit_run=0,
                  dense_from=None, start=info)
        log(f"starts: {info.get('kind')} {info.get('file', '')} dilation {info.get('dilation')} map {info.get('map')} hold {hold} faces {spec['faces']}")
    t0 = time.time() - st["seconds"]
    deadline = time.time() + wall_hours * 3600 - st["seconds"]
    box = dict(it=st["iteration"], last=(np.inf, np.inf), last_ckpt=time.time())
    stop = dict(reason=None, x=None)

    def evaluate(z, grad=True):
        p = mod.evaluate(z, 0.0, need_grad=grad, need_I=False)
        E, g = p["U"], (p["grad"] if grad else None)
        p["fmax"] = 0.0
        if hold and hold["kind"] == "D":
            hd = hold_D(z.reshape(mod.shape6), mod.x, power, need_grad=grad, g6=g6)
            p["D"], p["f"] = hd["D"], hold["mu"] * (hold["d"] - hd["D"])
            p["Eh"] = 0.5 * hold["mu"] * (hd["D"] - hold["d"]) ** 2
            E = E + p["Eh"]
            if grad:
                gh = g6 * mod.mask[..., None]
                g = g - p["f"] * gh
                p["fmax"] = float(abs(p["f"]) * np.abs(gh).max() / h3)
        elif hold and hold["kind"] == "obstacle":
            Eo, go, _ = obstacle(z.reshape(mod.shape6), mod.x, hold["Rd"], hold["mu_o"], need_grad=grad)
            p["Eo"] = Eo
            E = E + Eo
            if grad:
                go = go * mod.mask[..., None]
                g = g + go
                p["fmax"] = float(np.abs(go).max() / h3)
        p["E"] = E
        if grad:
            p["grad"] = g
            p["gmax"] = float(np.abs(g).max() / h3)
            p["gate"] = max(1e-4, min(1e-3, 0.1 * p["fmax"])) if hold else 1e-4
        return p

    def take_read(z, it, dense=False):
        p = evaluate(z)
        r = read_plus(z.reshape(mod.shape6), mod.x, surfaces)
        row = compact(it, p, r, time.time() - t0)
        (st["dense"] if dense else st["reads"]).append(row)
        return row, r, p

    def checkpoint(z):
        np.savez(ckpt, m6=z.reshape(mod.shape6))
        st["iteration"], st["seconds"] = box["it"], time.time() - t0
        state_f.write_text(json.dumps(st))
        box["last_ckpt"] = time.time()

    def dense_trigger(row, r):
        if not dense_allowed or st["dense_from"] is not None:
            return
        bs = r["surfaces"][-1]
        near = r["loop"] is not None and r["line"]["sets"][r["loop"]]["least_distance_to_faces"] <= 2.0 + 1e-9
        if near or not (bs["liftable"] and bs["abs_q"] == 1):
            st["dense_from"] = row["it"]
            log(f"dense reads from it {row['it']} (loop within 2 of a face: {near}; box surface {row['q'][-1]})")

    def after(z):
        box["it"] += 1
        it = box["it"]
        if due(it):
            row, r, p = take_read(z, it)
            if exit_rule:
                bs = r["surfaces"][-1]
                st["exit_run"] = st["exit_run"] + 1 if (not r["clear"] and bs["liftable"] and bs["abs_q"] == 0) else 0
                if st["exit_run"] >= 3:
                    stop.update(reason="EXIT", x=z.copy())
            dense_trigger(row, r)
            (work / "progress.json").write_text(json.dumps(dict(chain=name, iteration=it, U=row["U"], g=row["g"], D=row.get("D"), f=row.get("f"),
                                                                 sets=row["n"], clear=row["clear"], q=row["q"], time=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                 seconds=time.time() - t0)))
        elif st["dense_from"] is not None and not stop["reason"]:
            take_read(z, it, dense=True)
        if it % 25 == 0 and not stop["reason"]:
            if time.time() - box["last_ckpt"] > 900:
                checkpoint(z)
            if guard is not None and guard.wrap_up():
                checkpoint(z)
                log("interrupted on a power warning; checkpoint saved")
                sys.exit(75)
            if time.time() > deadline:
                stop.update(reason="WALL", x=z.copy())
        return bool(stop["reason"])

    if not st["reads"]:
        row, r, p = take_read(x, 0)
        st["start"].update(U=row["U"], Ut=row["Ut"], D=row.get("D"), f=row.get("f"), gmax_over_h3=row["g"], sets=row["n"],
                           radius=(row["sets"][row["loop"]][1] if row["loop"] is not None else None))
        dense_trigger(row, r)
        log(f"read 0: U {row['U']:.6f} D {row.get('D')} f {row.get('f')} g {row['g']:.3g} sets {row['n']} clear {row['clear']} q {row['q']}")

    def fun(z):
        p = evaluate(z)
        st["nfev"] += 1
        box["last"] = (p["E"], p["gmax"])
        return p["E"], p["grad"].ravel()

    def cb(z):
        if after(z):
            raise StopIteration

    message = ""
    while True:
        left = cap - box["it"]
        if left <= 0:
            message = "CAP"
            break
        before = box["it"]
        res = minimize(fun, x, jac=True, method="L-BFGS-B", callback=cb, options=dict(maxiter=left, maxfun=20 * cap, ftol=1e-15, gtol=0.0, maxcor=20))
        if stop["reason"]:
            x, message = stop["x"], stop["reason"]
            break
        x, msg = res.x, str(res.message)
        made, up = box["it"] - before, msg.upper()
        if "REL_REDUCTION_OF_F" in up or "RELATIVE REDUCTION" in up:
            st["ftol_ends"] += 1
            pe = evaluate(x)
            st["events"].append(dict(iteration=box["it"], event="function tolerance", made=made, E=float(res.fun), U=pe["U"], g=pe["gmax"], gate=pe["gate"]))
            what = "end" if (free and made <= 25) else after_function_tolerance(made, st["ftol_ends"], pe["gmax"], pe["gate"], st["stalls"])
            log(f"function tolerance at it {box['it']} after {made} iterations: E {res.fun:.8f} g {pe['gmax']:.3g} gate {pe['gate']:.3g}: {what}")
            if what.startswith("stalled"):
                st["stalls"] += 1
                st["events"].append(dict(iteration=box["it"], event="stalled", made=made, g=pe["gmax"]))
                if what == "stalled, restart":
                    continue
                message = "STALLED"
                break
            if what == "end":
                message = "FUNCTION TOLERANCE, REPEATED AFTER A RESTART"
                break
            continue
        if "ABNORMAL" in up and st["abnormal"] < 3 and made > 0:
            st["abnormal"] += 1
            st["events"].append(dict(iteration=box["it"], event="stalled line search", made=made))
            log(f"restart after a stalled line search at it {box['it']}")
            continue
        if "ABNORMAL" in up:
            st["events"].append(dict(iteration=box["it"], event="stalled line search", made=made))
            msg = "FUNCTION TOLERANCE, THEN A STALLED LINE SEARCH" if st["ftol_ends"] else "STALLED LINE SEARCH"
        message = "CAP" if box["it"] >= cap else msg
        break

    if st["reads"][-1]["it"] != box["it"]:
        take_read(x, box["it"])
    last = st["reads"][-1]
    m6 = x.reshape(mod.shape6)
    pe = evaluate(x)
    r = read_plus(m6, mod.x, surfaces)
    hd = hold_D(m6, mod.x, power)
    sp = sets_of(r["lab"], r["P"])
    shares, share_all = weight_shares(hd["u"], mod.x, sp, power) if sp else ([], 0.0)
    clear = r["clear"]
    lp = r["line"]["sets"][r["loop"]] if r["loop"] is not None else None
    gate_met = bool(pe["gmax"] <= pe["gate"])
    line_search_end = "STALLED LINE SEARCH" in message
    converged = bool(gate_met and (message == "FUNCTION TOLERANCE, REPEATED AFTER A RESTART" or line_search_end))
    word = "CONVERGED" if converged else ("STALLED" if message == "STALLED" else message if message in ("CAP", "WALL", "EXIT") else "NOT CONVERGED")
    end = dict(U=pe["U"], U2=pe["U2"], U4=pe["U4"], UV=pe["UV"], E=pe["E"], D=hd["D"], weight_center=hd["center"], f=pe.get("f"), Eh=pe.get("Eh"), Eo=pe.get("Eo"),
               gmax_over_h3=pe["gmax"], gate=pe["gate"], gate_met=gate_met, hold_force_max_over_h3=pe["fmax"], sets=len(r["line"]["sets"]), clear_sets=len(clear),
               fragments=r["line"]["fragments"], radius=(lp["radius"] if lp else None), separation=(2 * lp["radius"] if lp else None),
               center=(lp["center"] if lp else None), normal=(lp["normal"] if lp else None), room=(spec["hb"] - lp["radius"] if lp else None),
               least_distance_to_faces=(lp["least_distance_to_faces"] if lp else None), weight_share_per_set=shares, weight_share_within_1p5=share_all,
               set_radii=[s["radius"] for s in r["line"]["sets"]], set_counts=[s["count"] for s in r["line"]["sets"]],
               set_centers=[s["center"] for s in r["line"]["sets"]], set_touches=[s["touches"] for s in r["line"]["sets"]],
               set_faces=[s["faces"] for s in r["line"]["sets"]], cubes=r["cubes"], cls=r["cls"], q=last["q"])
    end_file = f"m5_50_{name}_end.npz"
    np.savez(a.out_dir / end_file, m6=m6, x=mod.x)
    rec = dict(chain=name, spec=spec, model=dict(N=spec["N"], half_box=spec["hb"], spacing=mod.h, delta=DELTA, L1=L1, kappa=1.0, v=1.0, stencil="averaged",
                                                 fixed_chart=True, free_faces=free, J=0.0, power=power),
               half_widths=[s.W for s in surfaces], start=st["start"], ended=message, word=word, converged=converged, line_search_end=line_search_end,
               iterations=box["it"], evaluations=st["nfev"], seconds=time.time() - t0, events=st["events"], stalls=st["stalls"], resumed_at=st["resumed_at"],
               dense_from=st["dense_from"], end=end, end_file=end_file, reads=st["reads"], dense=st["dense"],
               row_legend=dict(sets="count, radius, center (3), normal (3), least distance to the faces, faces touched (bits: 2 axis + side), max norm, spread, off plane",
                               q="one character per nested surface, inner to outer: |q| when liftable, x pierced, g gap at or under 0.2, l lift not consistent",
                               cls="one character per nested surface: |q| when liftable, L at the loop, B blind", ar="1 where the surface surrounds the loop",
                               cubes="per set, three cubes centered on it: half-width and its read", Ut="U2, U4, UV", g="max |dE/dm| / h^3", t="seconds"),
               finished=True)
    jf.write_text(json.dumps(rec, separators=(",", ":")) + "\n")
    ckpt.unlink(missing_ok=True)
    state_f.unlink(missing_ok=True)
    (work / "pid").unlink(missing_ok=True)
    log(f"CHAIN END it {box['it']} {message} [{word}]: U {pe['U']:.6f} D {hd['D']:.4f} f {pe.get('f')} g {pe['gmax']:.3g} (gate {pe['gate']:.3g}) sets {end['sets']} "
        f"clear {end['clear_sets']} q {last['q']} ({rec['seconds'] / 60:.1f} min)")


# ----------------------------------------------------------------------------- the controls on stored states

def controls(a):
    res = []

    def report(name, ok, numbers, counter):
        res.append(dict(control=name, passed=bool(ok), numbers=numbers, counter_input=counter))
        print(f"{name}: {'PASS' if ok else 'FAIL'}  {json.dumps(numbers)[:1500]}")
        print(f"      counter-input: {counter}", flush=True)

    bases = {g: a.out_dir / GRIDS[g]["base"] for g in GRIDS}
    have = {g: f for g, f in bases.items() if f.exists()}

    # K1: the states of M5.46 read back
    summ = json.loads((a.out_dir / "m5_46_summary.json").read_text())["chains"]
    ok, nums = True, []
    for g, ch, rad in (("10a", "B0c", 3.293), ("10b", "B0d", 3.316)):
        m6, x = load_state(bases[g])
        mod = Q.Model(GRIDS[g]["N"], GRIDS[g]["hb"], DELTA, L1, stencil="averaged", fixed_chart=True)
        U = mod.evaluate(m6.reshape(-1), 0.0, need_grad=False, need_I=False)["U"]
        ref = [e["U"] for e in summ[ch]["events"] if e["event"] == "function tolerance"][-1]
        lp, ns = one_loop(m6, x)
        good = abs(U - ref) <= 1e-12 * abs(ref) and ns == 1 and abs(lp["radius"] - rad) < 5e-4
        ok &= good
        nums.append(dict(state=ch, U=U, U_of_the_summary=ref, relative_gap=abs(U - ref) / abs(ref), sets=ns, radius=lp["radius"]))
    modw = Q.Model(41, 10.0, DELTA, 0.3, stencil="averaged", fixed_chart=True)
    Uw = modw.evaluate(load_state(bases["10a"])[0].reshape(-1), 0.0, need_grad=False, need_I=False)["U"]
    report("K1 the states of M5.46 read back", ok, nums, f"with L1 = 0.3 the same file gives U = {Uw:.6f}, a gap of {abs(Uw - nums[0]['U_of_the_summary']):.3f}, so the check fails on it")

    # K3: the instrument of M5.46 on the four base states
    if len(have) == 4:
        k3 = DG.selftest([str(f) for f in bases.values()], str(a.work_dir / "k3_instrument_of_m5_46.json"))
        inner = json.loads((a.work_dir / "k3_instrument_of_m5_46.json").read_text())
        report("K3 the seven controls of M5.46 on the four base states", k3, dict(controls=[(c["control"].split()[0], c["passed"]) for c in inner["controls"]], states=inner["base_states"]),
               "each of the seven prints its own counter-input (k3_instrument_of_m5_46.json)")
    else:
        report("K3 the seven controls of M5.46 on the four base states", False, dict(missing=[g for g in GRIDS if g not in have]), "not run: a base state is missing")

    # K4: the held quantity
    ok, nums, counter = True, [], ""
    for g in ("10a", "10b"):
        m6, x = load_state(bases[g])
        h, N = x[1] - x[0], len(x)
        mod = Q.Model(N, GRIDS[g]["hb"], DELTA, L1, stencil="averaged", fixed_chart=True)
        frame = Q.to6(mod.frame)
        lp, _ = one_loop(m6, x)
        D0 = hold_D(m6, x)["D"]
        D0c = hold_D(m6, x, about_box_center=True)["D"]
        row = dict(state=g, D=D0, reader_separation=2 * lp["radius"], D_minus_reader=D0 - 2 * lp["radius"], shifts=[], dilations=[])
        # try 1 shifted by 1, 1.5 and -2 length units (changes up to 1.9e-8, the far-field weight that leaves the box);
        # try 2 shifts by two cells, the input measured at plan time
        for axis, cells in a.k4_shifts:
            sh = frame.copy()
            cut = m6.copy()
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            out = [slice(None)] * 3
            if cells > 0:
                src[axis], dst[axis], out[axis] = slice(0, N - cells), slice(cells, N), slice(N - cells, N)
            else:
                src[axis], dst[axis], out[axis] = slice(-cells, N), slice(0, N + cells), slice(0, -cells)
            sh[tuple(dst)] = m6[tuple(src)]
            cut[tuple(out)] = frame[tuple(out)]
            Ds, Dsc, Dcut = hold_D(sh, x)["D"], hold_D(sh, x, about_box_center=True)["D"], hold_D(cut, x)["D"]
            row["shifts"].append(dict(axis="xyz"[axis], cells=cells, D=Ds, change=abs(Ds - D0), change_about_the_box_center=abs(Dsc - D0c),
                                      change_from_the_layers_that_leave_the_box=abs(Dcut - D0), change_from_the_shift_alone=abs(Ds - Dcut)))
            ok &= abs(Ds - D0) <= 1e-8
        for s in (1.5, 2.0):
            md = dilate(m6, x, lp["center"], s)
            Dd = hold_D(md, x)["D"]
            lpd, nsd = one_loop(md, x)
            row["dilations"].append(dict(s=s, D=Dd, ratio=Dd / D0, reader_separation=2 * lpd["radius"], D_minus_reader=Dd - 2 * lpd["radius"], sets=nsd))
            ok &= abs(Dd / D0 / s - 1.0) <= 0.02 and 0.1 <= Dd - 2 * lpd["radius"] <= 0.6
        ok &= 0.1 <= row["D_minus_reader"] <= 0.6
        nums.append(row)
        counter = (f"the same sum taken about the box center moves by {max(s['change_about_the_box_center'] for s in row['shifts']):.3f} under the shifts of {g} "
                   f"(above 0.1), so the check fails on it")
    report("K4 the held quantity: unchanged by a shift, in the ratio of a dilation, above the reader by 0.1 to 0.6", ok, nums, counter)

    # K5: the gradients against central differences, on a dilated base
    ok, nums = True, []
    rng = np.random.default_rng(20261120)
    for g in ("10a",):
        m6, x = load_state(bases[g])
        N, hb = GRIDS[g]["N"], GRIDS[g]["hb"]
        mod = Q.Model(N, hb, DELTA, L1, stencil="averaged", fixed_chart=True)
        lp, _ = one_loop(m6, x)
        md = set_faces(dilate(m6, x, lp["center"], 1.3), m6)
        Dd = hold_D(md, x)["D"]
        d, mu, Rd = Dd + 0.5, MU, 4.5
        gD = np.zeros(md.shape)
        hold_D(md, x, need_grad=True, g6=gD)
        pU = mod.evaluate(md.reshape(-1), 0.0, need_grad=True, need_I=False)
        f = mu * (d - Dd)
        go = obstacle(md, x, Rd, need_grad=True)[1]
        an = dict(D=gD, hold=-f * gD, E=pU["grad"] - f * gD * mod.mask[..., None], obstacle=go)

        def val(z, what):
            if what == "D":
                return hold_D(z, x)["D"]
            if what == "hold":
                return 0.5 * mu * (hold_D(z, x)["D"] - d) ** 2
            if what == "E":
                return mod.evaluate(z.reshape(-1), 0.0, need_grad=False, need_I=False)["U"] + 0.5 * mu * (hold_D(z, x)["D"] - d) ** 2
            return obstacle(z, x, Rd)[0]

        k0 = (N - 1) // 2
        for what in ("D", "hold", "E", "obstacle"):
            big = float(np.abs(an[what]).max())
            if what == "obstacle":
                ii = np.argwhere(np.abs(an[what]) > 0)
                pick = ii[rng.choice(len(ii), 150)].tolist() + [[*rng.integers(1, N - 1, 3).tolist(), int(rng.integers(0, 6))] for _ in range(50)]
            else:
                wn = np.argwhere(np.abs(gD).max(-1)[1:-1, 1:-1, 1:-1] > 1e-3 * np.abs(gD).max()) + 1
                pick = [[*wn[rng.integers(len(wn))].tolist(), int(rng.integers(0, 6))] for _ in range(150)] + \
                       [[*rng.integers(1, N - 1, 3).tolist(), int(rng.integers(0, 6))] for _ in range(50)]
            worst = 0.0
            for i, j, k, c in pick:
                best = np.inf
                for step in (1e-3, 1e-4, 1e-5):
                    zp, zm = md.copy(), md.copy()
                    zp[i, j, k, c] += step
                    zm[i, j, k, c] -= step
                    num = (val(zp, what) - val(zm, what)) / (2 * step)
                    best = min(best, abs(num - an[what][i, j, k, c]))
                worst = max(worst, best / big)
            ok &= worst <= 1e-6
            nums.append(dict(term=what, entries=len(pick), largest_entry=big, worst_gap_over_largest=worst))
        # the counter-input: the term of the moving center left out, and a wrong sign
        gno = gD.copy()
        hd = hold_D(md, x)
        X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1) - np.array(hd["center"])
        r = np.linalg.norm(X, axis=-1)
        w = hd["u"] ** POWER
        avec = (w[..., None] * X / np.maximum(r, 1e-300)[..., None]).sum((0, 1, 2)) / w.sum()
        co_full = r - 0.5 * hd["D"] - X @ avec
        co_no = r - 0.5 * hd["D"]
        nz = np.abs(co_full) > 1e-12
        gno[nz] = gD[nz] * (co_no[nz] / co_full[nz])[..., None]
        miss = float(np.abs(gno - gD).max() / np.abs(gD).max())
    # an off-center state makes the moving-center term visible
    sh = Q.to6(mod.frame).copy()
    c3 = int(round(3.0 / (x[1] - x[0])))
    sh[c3:, :, :] = md[:N - c3, :, :]
    gS = np.zeros(sh.shape)
    hs = hold_D(sh, x, need_grad=True, g6=gS)
    i, j, k = np.unravel_index(np.abs(gS[..., 0]).argmax(), gS.shape[:3])
    zp, zm = sh.copy(), sh.copy()
    zp[i, j, k, 0] += 1e-4
    zm[i, j, k, 0] -= 1e-4
    numS = (hold_D(zp, x)["D"] - hold_D(zm, x)["D"]) / 2e-4
    report("K5 the gradients of D, of the hold term, of E_mu and of the obstacle against central differences", ok, nums,
           f"the opposite sign of the hold term is off by 2 of its largest entry; with the term of the moving center left out the gradient of D changes by {miss:.2e} of its "
           f"largest entry on this state (a shifted copy: analytic {gS[i, j, k, 0]:.6e} against {numS:.6e} by differences)")

    # K12: the maps of the family
    ok, nums = True, []
    for g in ("10a", "10b"):
        m6, x = load_state(bases[g])
        lp, _ = one_loop(m6, x)
        z0 = [float(np.abs(apply_map(m6, x, dict(kind="pinch", m=mm, eps=0.0), lp)[0] - m6).max()) for mm in (2, 3)]
        z0.append(float(np.abs(noise(m6, x, lp["center"], lp["normal"], lp["radius"], 0.0, 5001) - m6).max()))
        outs = {k: apply_map(m6, x, v, lp)[1] for k, v in FAMILY.items() if v["kind"] == "pinch"}
        new, _ = pinch(m6, x, lp["center"], lp["radius"], 0.5, 2)
        line, P, lab = DG.line_read(new, x)
        sp = sets_of(lab, P)
        rr = np.hypot(sp[0][:, 0] - lp["center"][0], sp[0][:, 1] - lp["center"][1]) if sp else np.array([0.0, 1.0])
        ratio = float(rr.min() / rr.max())
        nz = noise(m6, x, lp["center"], lp["normal"], lp["radius"], 0.2, 5002) - m6
        good = max(z0) <= 1e-14 and all(v == 0 for v in outs.values()) and len(line["sets"]) == 1 and abs(ratio - 0.5) <= 0.1
        ok &= good
        nums.append(dict(state=g, largest_change_at_zero_amplitude=z0, source_points_outside_the_box=outs, sets_after_the_pinch=len(line["sets"]),
                         radii_min_over_max=ratio, noise_rms_near_the_loop=float(np.sqrt((nz[_near_circle(x, lp, 1.5)] ** 2).mean())),
                         noise_largest_on_the_faces=float(max(np.abs(nz[0]).max(), np.abs(nz[-1]).max(), np.abs(nz[:, 0]).max(), np.abs(nz[:, -1]).max(),
                                                              np.abs(nz[:, :, 0]).max(), np.abs(nz[:, :, -1]).max()))))
    m6, x = load_state(bases["10a"])
    lp, _ = one_loop(m6, x)
    c2 = [lp["center"][0] + 2.0, lp["center"][1], lp["center"][2]]
    new2 = pinch(m6, x, c2, lp["radius"], 0.5, 2)[0]
    l2, P2, lab2 = DG.line_read(new2, x)
    sp2 = sets_of(lab2, P2)
    rr2 = np.hypot(sp2[0][:, 0] - lp["center"][0], sp2[0][:, 1] - lp["center"][1]) if sp2 else np.array([0.0, 1.0])
    report("K12 the maps of the family: the identity at zero amplitude, no source outside the box, the pinch at 0.5", ok, nums,
           f"the same pinch applied about an axis 2 to the side gives {len(l2['sets'])} set(s) with radii in the ratio {rr2.min() / rr2.max():.2f}, so the check fails on it")

    passed = all(r["passed"] for r in res)
    out = dict(passed=passed, controls=res, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    (a.work_dir / (a.controls_out or "controls_stored_states.json")).write_text(json.dumps(out, indent=1) + "\n")
    print("CONTROLS:", "ALL PASS" if passed else "FAIL: " + ", ".join(r["control"].split()[0] for r in res if not r["passed"]))
    return passed


def _near_circle(x, lp, radius):
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1) - np.asarray(lp["center"])
    nrm = np.asarray(lp["normal"]) / np.linalg.norm(lp["normal"])
    zp = X @ nrm
    rp = np.linalg.norm(X - zp[..., None] * nrm, axis=-1)
    return np.hypot(rp - lp["radius"], zp) < radius


# ----------------------------------------------------------------------------- the job list and the pool

def tname(d):
    return f"d{int(round(d * 100)):04d}"


def rung_name(g, d, mu=MU):
    return f"R{g}_{tname(d)}_m{int(mu)}"


def base_D(a, g, cache={}):
    if g not in cache:
        f = a.out_dir / GRIDS[g]["base"]
        if not f.exists() or (g.startswith("14") and rec_of(a, f"B{g}") is None):
            return None
        m6, x = load_state(f)
        cache[g] = hold_D(m6, x, a.power)["D"]
    return cache[g]


def rec_of(a, n):
    f = a.work_dir / "chains" / f"{n}.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return r if r.get("finished") else None


def static_jobs(a):
    """Every chain whose spec does not depend on an outcome. A spec with needs waits for those chains."""
    J = {}

    def add(name, g_or_grid, **kw):
        G = GRIDS[g_or_grid] if isinstance(g_or_grid, str) else g_or_grid
        J[name] = dict(N=G["N"], hb=G["hb"], faces=kw.pop("faces", "held"), wall=kw.pop("wall", G.get("wall")), power=a.power, needs=kw.pop("needs", []), **kw)

    add("B14a", "14a", start=dict(kind="seed"), hold=None, wall=1.0, cls="base")
    add("B14b", "14b", start=dict(kind="seed"), hold=None, wall=2.5, cls="base")
    for n, src, N, hb, wall in (("E0a", "F0a", 33, 8.0, 0.5), ("E0b", "F0b", 65, 8.0, 1.0), ("E14c", "P14c", 41, 10.0, 0.5), ("E14d", "P14d", 81, 10.0, 2.0)):
        add(n, dict(N=N, hb=hb), faces="free", start=dict(kind="state", file=f"m5_46_{src}_end.npz"), hold=None, exit_rule=True, dense=True, wall=wall, cls="exit")
    for g, G in GRIDS.items():
        bneeds = [f"B{g}"] if g.startswith("14") else []
        add(f"K6ref_{g}", g, start=dict(kind="state", file=G["base"]), hold=None, needs=bneeds, cls="k6")
        D0 = base_D(a, g)
        if D0 is None:
            continue
        add(f"R{g}_free_m20", g, start=dict(kind="state", file=G["base"]), hold=dict(kind="D", mu=MU, d=D0), needs=bneeds, cls="k6", target="free")
        if g == "10a":
            for j in (1, 2, 3, 4):
                d = D0 + 0.1 * j
                add(f"K7_10a_p{j}", g, start=dict(kind="state", file=G["base"], dilate=dict(to_D=d)), hold=dict(kind="D", mu=MU, d=d), cls="k7", target=d)
        k7 = [f"K7_10a_p{j}" for j in (1, 2, 3, 4)] + ["R10a_free_m20", "K6ref_10a"]
        for d in G["targets"] + G["inward"]:
            add(rung_name(g, d), g, start=dict(kind="state", file=G["base"], dilate=dict(to_D=d)), hold=dict(kind="D", mu=MU, d=d), needs=bneeds + k7, cls="rung", target=d,
                outward=d in G["targets"])
        for d in (G["targets"] if G["mu2"] is None else G["mu2"]):
            add(rung_name(g, d, MU2), g, start=dict(kind="state", file=G["base"], dilate=dict(to_D=d)), hold=dict(kind="D", mu=MU2, d=d), needs=bneeds + k7, cls="mu2",
                target=d)
        for Rd in G["obstacle"]:
            st = dict(kind="state", file=G["base"]) if Rd == 1.5 else dict(kind="state", file=G["base"], dilate=dict(to_radius=Rd + 0.5))
            add(f"O{g}_r{int(round(Rd * 10)):02d}", g, start=st, hold=dict(kind="obstacle", mu_o=MU_O, Rd=Rd), needs=bneeds + k7, cls="obstacle", Rd=Rd)
        if g in ("10a", "14a"):
            prev = f"R{g}_free_m20"
            ups = []
            for d in G["targets"]:
                n = f"U{g}_{tname(d)}"
                add(n, g, start=dict(kind="state", file=f"m5_50_{prev}_end.npz"), hold=dict(kind="D", mu=MU, d=d), needs=bneeds + k7 + [prev], cls="cont_up", target=d)
                prev = n
                ups.append(n)
            for d in reversed(G["targets"][:-1]):
                n = f"V{g}_{tname(d)}"
                add(n, g, start=dict(kind="state", file=f"m5_50_{prev}_end.npz"), hold=dict(kind="D", mu=MU, d=d), needs=bneeds + k7 + [prev], cls="cont_down", target=d)
                prev = n
    return J


def all_jobs(a):
    """The static list and the chains that the outcomes call for (bisections, branch checks, the family, the exits of
    the top rung, the releases), by the rules of m5_50_reads.py."""
    J = static_jobs(a)
    if a.no_hold:
        return {n: c for n, c in J.items() if c["hold"] is None}
    import m5_50_reads as RD
    for n, c in RD.dynamic_jobs(a, J, GRIDS, FAMILY, FAMILY_113).items():
        J[n] = c
    return J


def chain_alive(a, n):
    f = a.work_dir / "chains" / n / "pid"
    if not f.exists():
        return False
    try:
        pid = int(f.read_text())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
    return f"--chain {n}" in cmd


def ceiling(a):
    try:
        n49 = len([ln for ln in subprocess.run(["pgrep", "-f", "scripts/m5_49_"], capture_output=True, text=True).stdout.split() if ln.strip()])
    except Exception:  # noqa: BLE001
        n49 = 4
    c = min(10, 12 - min(4, n49))
    f = a.work_dir / "pool_max"
    if f.exists():
        try:
            c = min(c, int(f.read_text().split()[0]))
        except (ValueError, IndexError):
            pass
    return c, n49


def pool(a):
    t0 = time.time()
    logf = open(a.work_dir / "pool.log", "a")

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')} +{(time.time() - t0) / 60:6.1f} min] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    running, failed, stop, launched_at = {}, {}, False, {}
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        env[var] = "1"
    log(f"pool starts (no_hold {a.no_hold}, power {a.power})")
    last_note = ""
    while True:
        for n, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[n]
                r = rec_of(a, n)
                log(f"{n} exited {rc}" + (f": {r['word']} it {r['iterations']} U {r['end']['U']:.6f} D {r['end']['D']:.4f} sets {r['end']['clear_sets']}" if r else " WITHOUT a record"))
                if rc == 75:
                    stop = True
                elif rc != 0 or r is None:
                    failed[n] = rc
        if (a.work_dir / "pool_stop").exists():
            stop = True
        J = all_jobs(a)
        halt = J.pop("__halt__", None)
        if halt and not running:
            log("HALT: " + str(halt))
            break
        if halt:
            stop = True
        note_end = J.pop("__note__", "")
        pending = [n for n in J if rec_of(a, n) is None and n not in running and n not in failed]
        ext = [n for n in pending if chain_alive(a, n)]
        pending = [n for n in pending if n not in ext]
        if not pending and not running and not ext:
            log("nothing left to run. " + str(note_end))
            break
        cmax, n49 = ceiling(a)
        ready = [n for n in pending if all(rec_of(a, m) is not None for m in J[n]["needs"]) and not any(m in failed for m in J[n]["needs"])]
        blocked = [n for n in pending if any(m in failed for m in J[n]["needs"])]
        # the bases first; then the largest grids, two slots kept for the small grids while any is ready
        # the continuation chains go before the other rungs: each is a serial chain of up to 35 links, the longest path
        order = dict(base=0, k6=1, k7=1, exit=2, cont_up=2, cont_down=2)
        ready.sort(key=lambda n: (order.get(J[n]["cls"], 3), -J[n]["N"]))
        small_ready = any(J[n]["N"] <= 57 for n in ready)
        if not stop:
            for n in ready:
                if len(running) + len(ext) >= cmax:
                    break
                big_running = sum(1 for m in list(running) + ext if J.get(m, {}).get("N", 0) >= 65)
                if J[n]["N"] >= 81 and small_ready and big_running >= max(1, cmax - 2) and order.get(J[n]["cls"], 3) == 3:
                    continue
                w = a.work_dir / "chains" / n
                w.mkdir(parents=True, exist_ok=True)
                spec = {k: v for k, v in J[n].items() if k != "needs"}
                (w / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
                p = subprocess.Popen([sys.executable, "-I", str(HERE / "m5_50_hold.py"), "--chain", n, "--out-dir", str(a.out_dir), "--work-dir", str(a.work_dir)],
                                     stdout=open(w / "stdout.txt", "a"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
                running[n] = p
                launched_at[n] = time.time()
                log(f"started {n} (pid {p.pid}); {len(running) + len(ext)} running, ceiling {cmax} (M5.49 processes seen: {n49})")
        note = f"running {sorted(running)} ext {ext} ready {len(ready)} waiting {len(pending) - len(ready)} blocked {blocked} failed {failed}"
        (a.work_dir / "pool_status.json").write_text(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), minutes=(time.time() - t0) / 60, running=sorted(running),
                                                                    external=ext, ready=len(ready), pending=len(pending), blocked=blocked, failed=failed,
                                                                    done=sum(1 for n in J if not n.startswith("__") and rec_of(a, n) is not None),
                                                                    total=sum(1 for n in J if not n.startswith("__")), ceiling=cmax), indent=1) + "\n")
        if not running and not ext and (stop or not ready):
            log("stops: " + ("a stop was asked" if stop else "nothing can start: " + note))
            break
        time.sleep(a.poll)
    log(f"pool ends after {(time.time() - t0) / 60:.1f} min; failed {failed}")
    sys.exit(0 if not failed else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--no-hold", action="store_true", help="with --pool: only the chains with no hold (the bases, the exits of M5.46, the K6 references)")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--controls-out", default=None)
    ap.add_argument("--power", type=int, default=POWER)
    ap.add_argument("--k4-try1", action="store_true", help="K4 with the shifts of the first try (1, 1.5 and -2 length units)")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--work-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_50")
    a = ap.parse_args()
    a.k4_shifts = ((2, 2), (0, 3), (2, -4)) if a.k4_try1 else ((2, 2), (0, 2), (2, -2))
    if a.list:
        for k, c in all_jobs(a).items():
            if not k.startswith("__"):
                print(k, {kk: vv for kk, vv in c.items() if kk in ("N", "hb", "faces", "hold", "needs", "cls", "wall")})
    elif a.pool:
        pool(a)
    elif a.controls:
        sys.exit(0 if controls(a) else 1)
    elif a.chain:
        run_chain(a.chain, a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
