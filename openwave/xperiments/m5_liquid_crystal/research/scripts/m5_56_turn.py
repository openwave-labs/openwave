"""M5.56: the collar turn T, the twist W about the long axis, the seeds (the biaxial hedgehog, the pair with its
ladder, the four-line field), the start rule, the path tables and the readers of sections 2, 10 and 12 of the plan
(the turn reader, the interior fit, the tear reads, the two-ring reader, the minor reads), with the controls of
sections 3 and 14 under --selftest.

The action is that of M5.51 (m5_51_action.py, unchanged): kappa = v = L_V = 1, L_1 = 0, the midpoint link weight,
faces held, delta = 0 (sections 1 to 9) or delta = 0.05 (sections 10 to 12). With f the linear profile

    f(rho) = min(1, max(0, (R2 - rho) / (R2 - R1))),   rho = |x - c|

    T[Theta, a, c, R1, R2]:  M(n) -> Q(n) M(n) Q(n)^T,   Q(n) = exp(Theta f(rho_n) [a]_x)
    W[Theta, c, R1, R2]:     the same with a = e1(n), the top eigenvector of the reference at the node, its sign fixed
                             outward (e1 . (x - c) > 0), or by a breadth-first fill over the collar on a pair collar

The turn reader fits per shell k (rho in [k h - h/2, k h + h/2)) the rotation q_k and its radial rate w_k,
min sum |M(n) - Q_n M_ref(n) Q_n^T|^2 with Q_n = Q(q_k) exp((rho_n - rho_k) [w_k]_x), by BFGS (gtol 1e-12) from the
shell outside it, q continued (q_k . q_{k+1} > 0) from q = +1 at the outermost shell; the sign sigma is the sign of
q_0 at the sign shell, the innermost with rho_k - h/2 >= R1.

Usage:
  python m5_56_turn.py --selftest d0 [--try N]       C1, C2, C3 (a) and (b) on G6_mid_a, C8, C6
  python m5_56_turn.py --selftest d5 [--try N]       C3 (a) and C8 at delta = 0.05, C9 to C17
  python m5_56_turn.py --c3a GRID --ref FILE.npz [--try N]   C3 (a) on the reference of H2 or H3 (fixed fields)
  python m5_56_turn.py --paths ROW                   the path tables of a reference (fixed fields)
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_51_action as A  # noqa: E402
import m5_51_reads as RD  # noqa: E402
import m5_46_degree as DG  # noqa: E402

CK = HERE.parent / "checkpoints" / "m5_56"
CK51 = HERE.parent / "checkpoints" / "m5_51" / "chains"
DELTA = 0.05
GRIDS = dict(H1=(12.0, 0.5, 49), H2=(12.0, 0.375, 65), H3=(15.0, 0.5, 61), P=(18.0, 0.5, 73))
TUBE = 2.0
GAP_Z = 0.5


# ----------------------------------------------------------------------------- basics

def ck(delta):
    return (1.0 + delta, 1.0 + delta ** 2, 1.0 + delta ** 3)


def axis_of(grid):
    hb, h, N = GRIDS[grid]
    return np.linspace(-hb, hb, N)


def mesh(x):
    return np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)


def unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def skew(a):
    a = np.asarray(a, float)
    K = np.zeros(a.shape[:-1] + (3, 3))
    K[..., 0, 1] = -a[..., 2]
    K[..., 0, 2] = a[..., 1]
    K[..., 1, 0] = a[..., 2]
    K[..., 1, 2] = -a[..., 0]
    K[..., 2, 0] = -a[..., 1]
    K[..., 2, 1] = a[..., 0]
    return K


def rot(axes, ang):
    """exp(ang [axis]_x) for one unit axis or per-node unit axes (..., 3), angles (...)."""
    ang = np.asarray(ang, float)
    a = np.asarray(axes, float)
    if a.ndim == 1:
        a = np.broadcast_to(a, ang.shape + (3,))
    K = skew(a)
    return np.eye(3) + np.sin(ang)[..., None, None] * K + (1.0 - np.cos(ang))[..., None, None] * (K @ K)


def rotvec(w):
    """exp([w]_x) for rotation vectors w (..., 3)."""
    t = np.linalg.norm(w, axis=-1)
    return rot(w / np.maximum(t, 1e-300)[..., None], t)


def conj(Q, M):
    return Q @ M @ np.swapaxes(Q, -1, -2)


def fprof(rho, R1, R2):
    return np.clip((R2 - rho) / (R2 - R1), 0.0, 1.0)


def eigs(m6):
    """Eigenvalues descending (l1 >= l2 >= l3) and their eigenvectors as columns."""
    w, V = np.linalg.eigh(A.from6(m6))
    return w[..., ::-1], V[..., :, ::-1]


def faces_mask(N):
    f = np.ones((N, N, N), bool)
    f[1:-1, 1:-1, 1:-1] = False
    return f


def make_model(grid, delta=0.0, hold6=None, held=None):
    """M5.51's model on a grid of this task; hold6 replaces the attribute M0 (the field the held nodes keep), held
    adds held nodes (a ball) to the faces."""
    hb, h, N = GRIDS[grid]
    mod = A.Model(N, hb, weight="mid", delta=delta)
    if hold6 is not None:
        mod.M0 = A.from6(np.asarray(hold6, float).reshape(N, N, N, 6)).copy()
    if held is not None:
        mod.mask &= ~held
    return mod


def energy(mod, m6, grad=False):
    p = mod.evaluate(np.asarray(m6, float), need_grad=grad)
    out = dict(U=p["U"], U4=p["U4"], UV=p["UV"], ULV=p["ULV"])
    if grad:
        out["gate"] = float(np.abs(p["grad"]).max() / mod.h ** 3)
    return out


def load51(name):
    rec = json.loads((CK51 / f"{name}.json").read_text())
    z = np.load(CK51 / name / rec["end_file"], allow_pickle=False)
    return np.asarray(z["m6"], float), np.asarray(z["x"], float), rec["end"]


def ring_of(end):
    return dict(c=np.asarray(end["center"], float), nu=unit(end["normal"]), RL=float(end["RL"]))


def inplane_axis(nu):
    nu = unit(nu)
    t = np.array([1.0, 0, 0])
    if RD.angle(nu, t) <= 25.0:
        t = np.array([0, 1.0, 0])
    a = t - (t @ nu) * nu
    return unit(a)


def ring_distance(X, c, nu, RL):
    q = X - np.asarray(c, float)
    nu = unit(nu)
    z = q @ nu
    rp = np.sqrt(np.maximum((q * q).sum(-1) - z * z, 0.0))
    return np.sqrt((rp - RL) ** 2 + z * z)


def tubes(x, rings, tube=TUBE):
    X = mesh(x)
    t = np.zeros(X.shape[:-1], bool)
    for r in rings:
        t |= ring_distance(X, r["c"], r["nu"], r["RL"]) <= tube
    return t


# ----------------------------------------------------------------------------- the turns

def T_turn(m6, x, Th, a, c, R1, R2, fmap=None):
    X = mesh(x) - np.asarray(c, float)
    rho = np.linalg.norm(X, axis=-1)
    f = fprof(rho, R1, R2) if fmap is None else fmap
    Q = rot(unit(a), Th * f)
    return A.to6(conj(Q, A.from6(m6)))


def e1_outward(ref6, x, c):
    _, V = eigs(ref6)
    e1 = V[..., :, 0].copy()
    sg = np.sign((e1 * (mesh(x) - np.asarray(c, float))).sum(-1))
    sg[sg == 0] = 1.0
    return e1 * sg[..., None]


def e1_fill(ref6, x, c, R1, R2):
    """Section 12: from the collar node of largest |e1 . rhat|, oriented outward, a breadth-first fill over the collar
    nodes (0 < f < 1) gives each node the sign that makes e1 . e1' >= 0 with the node it was reached from. Returns the
    oriented e1 (zero outside the collar), the collar mask, the count reached and the least e1 . e1' over collar
    links."""
    X = mesh(x) - np.asarray(c, float)
    rho = np.linalg.norm(X, axis=-1)
    f = fprof(rho, R1, R2)
    col = (f > 0) & (f < 1)
    _, V = eigs(ref6)
    e1 = V[..., :, 0]
    rh = X / np.maximum(rho, 1e-300)[..., None]
    dot = np.abs((e1 * rh).sum(-1))
    N = len(x)
    idx = np.argwhere(col)
    start = tuple(idx[np.argmax(dot[col])])
    ori = np.zeros(col.shape)
    ori[start] = 1.0 if (e1[start] * rh[start]).sum() > 0 else -1.0
    dq = deque([start])
    while dq:
        p = dq.popleft()
        v = ori[p] * e1[p]
        for ax in range(3):
            for st in (-1, 1):
                nb = list(p)
                nb[ax] += st
                if not (0 <= nb[ax] < N):
                    continue
                nb = tuple(nb)
                if col[nb] and ori[nb] == 0:
                    ori[nb] = 1.0 if (e1[nb] * v).sum() >= 0 else -1.0
                    dq.append(nb)
    eo = e1 * ori[..., None]
    least = 1.0
    for ax in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[ax] = slice(0, -1)
        s1[ax] = slice(1, None)
        both = col[tuple(s0)] & col[tuple(s1)]
        if both.any():
            least = min(least, float((eo[tuple(s0)] * eo[tuple(s1)]).sum(-1)[both].min()))
    return eo, col, int((ori != 0).sum()), int(col.sum()), least


def e1_pair_profile(ref6, x, c, R1, R2):
    """The oriented e1 the minor turn profile uses on a pair collar: the fill of section 12 on the collar nodes, the
    outward sign of section 10 on the other nodes of the profile's shells."""
    eo, col, _, _, _ = e1_fill(ref6, x, c, R1, R2)
    return np.where(col[..., None], eo, e1_outward(ref6, x, c))


def W_turn(m6, ref6, x, Th, c, R1, R2, fill=False):
    X = mesh(x) - np.asarray(c, float)
    rho = np.linalg.norm(X, axis=-1)
    f = fprof(rho, R1, R2)
    if fill:
        e1, col, _, _, _ = e1_fill(ref6, x, c, R1, R2)
        Q = rot(np.where(col[..., None], e1, np.array([0, 0, 1.0])), Th * f)
        Q[~col] = np.eye(3)
    else:
        e1 = e1_outward(ref6, x, c)
        Q = rot(e1, Th * f)
    return A.to6(conj(Q, A.from6(m6)))


def W_regime(ref6, x, c, R1, R2):
    X = mesh(x) - np.asarray(c, float)
    rho = np.linalg.norm(X, axis=-1)
    col = (rho > R1) & (rho < R2)
    w, V = eigs(ref6)
    rh = X / np.maximum(rho, 1e-300)[..., None]
    dot = np.abs((V[..., :, 0] * rh).sum(-1))[col]
    g12 = (w[..., 0] - w[..., 1])[col]
    return dict(least_e1_rhat=float(dot.min()), least_g12=float(g12.min()), nodes_under_0p9=int((dot < 0.9).sum()),
                regime=bool(dot.min() >= 0.9 and g12.min() >= 0.5))


# ----------------------------------------------------------------------------- the seeds

def seed_bi(x, delta=DELTA, iso=None):
    """M_0delta of section 10 (iso: the leading isotropic part, (1 + delta)/3 unless given; C9's failing input)."""
    X = mesh(x)
    r = np.linalg.norm(X, axis=-1) + 1e-12
    n = X / r[..., None]
    zc = np.cross(np.array([0, 0, 1.0]), n)
    sz = np.linalg.norm(zc, axis=-1)
    m = zc / np.maximum(sz, 1e-300)[..., None]
    m[sz < 1e-12] = [1.0, 0, 0]
    p = np.cross(n, m)
    rp = r * sz
    nn = np.einsum("...i,...j->...ij", n, n)
    B = 0.5 * delta * (np.eye(3) - nn) + 0.5 * delta * np.tanh(rp / 1.2)[..., None, None] * (
        np.einsum("...i,...j->...ij", m, m) - np.einsum("...i,...j->...ij", p, p))
    mb = (1 + delta) / 3
    M = (mb if iso is None else iso) * np.eye(3) + np.tanh(r / 1.2)[..., None, None] * (nn + B - mb * np.eye(3))
    return A.to6(M)


RIN, ROUT, DFAR = 11.35, 17.0, 14.0


def ladder(r, d0):
    t = np.clip((r - RIN) / (ROUT - RIN), 0.0, 1.0)
    return d0 + (DFAR - d0) * (3 * t * t - 2 * t ** 3)


def pair_bi(x, d0, delta=DELTA, use_ladder=True):
    """M_0delta^(2) of section 12, centers (+-d(|x|)/2, 0, 0)."""
    X = mesh(x)
    r = np.linalg.norm(X, axis=-1)
    dd = ladder(r, d0) if use_ladder else np.full(r.shape, float(d0))
    zs, amp, rhos = [], 1.0, []
    for s in (+1, -1):
        c = np.zeros(X.shape)
        c[..., 0] = s * dd / 2
        Y = X - c
        rk = np.linalg.norm(Y, axis=-1) + 1e-12
        u = Y / rk[..., None]
        den = 1 - u[..., 2]
        zk = (u[..., 0] + 1j * u[..., 1]) / np.where(np.abs(den) < 1e-15, 1e-15, den)
        zs.append(zk)
        amp = amp * np.tanh(rk / 1.2)
        rhos.append(rk)
    z = zs[0] * zs[1]
    a2 = np.abs(z) ** 2
    big = ~np.isfinite(a2) | (a2 > 1e30)
    with np.errstate(invalid="ignore", over="ignore"):
        n = np.stack([2 * z.real, 2 * z.imag, a2 - 1], -1) / (a2 + 1)[..., None]
    n[big] = [0, 0, 1.0]
    zc = np.cross(np.array([0, 0, 1.0]), n)
    sz = np.linalg.norm(zc, axis=-1)
    m = zc / np.maximum(sz, 1e-300)[..., None]
    m[sz < 1e-12] = [1.0, 0, 0]
    p = np.cross(n, m)
    nn = np.einsum("...i,...j->...ij", n, n)
    core = np.tanh(np.minimum(rhos[0], rhos[1]) * sz / 1.2)
    B = 0.5 * delta * (np.eye(3) - nn) + 0.5 * delta * core[..., None, None] * (
        np.einsum("...i,...j->...ij", m, m) - np.einsum("...i,...j->...ij", p, p))
    mb = (1 + delta) / 3
    M = mb * np.eye(3) + amp[..., None, None] * (nn + B - mb * np.eye(3))
    return A.to6(M)


def stereo_frame(n):
    k = np.array([0, 0, 1.0])
    mv = n - k
    m2 = (mv * mv).sum(-1, keepdims=True) + 1e-300

    def refl(v):
        return v - 2 * (v * mv).sum(-1, keepdims=True) * mv / m2
    ex = np.broadcast_to(np.array([1.0, 0, 0]), n.shape)
    ey = np.broadcast_to(np.array([0, 1.0, 0]), n.shape)
    return refl(ex), refl(ey)


def four_line(x, alpha_deg=16.0, delta=DELTA):
    """The four-line field of section 14 (the input of C12 and C13)."""
    Xg = mesh(x)
    Xc = Xg.copy()
    onp = (np.abs(Xg[..., 0]) < 1e-12) & (np.abs(Xg[..., 1]) < 1e-12) & (Xg[..., 2] > 0)
    r = np.linalg.norm(Xg, axis=-1) + 1e-12
    Xc[..., 0] = np.where(onp, Xg[..., 0] + 1e-3 * r, Xg[..., 0])
    rc = np.linalg.norm(Xc, axis=-1) + 1e-12
    n = Xg / r[..., None]
    nc = Xc / rc[..., None]
    w = (Xc[..., 0] + 1j * Xc[..., 1]) / (rc - Xc[..., 2] + 1e-300)
    a = np.radians(alpha_deg)
    ta, ca = np.tan(a / 2), 1 / np.tan(a / 2)
    psi = 0.5 * np.angle((w * w - ta * ta) * (w * w + ca * ca))
    t1, t2 = stereo_frame(nc)
    d = np.cos(psi)[..., None] * t1 + np.sin(psi)[..., None] * t2
    d = d - (d * n).sum(-1, keepdims=True) * n
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    s_, c_ = np.sin(a), np.cos(a)
    rays = [np.array([s_, 0, -c_]), np.array([-s_, 0, -c_]), np.array([0, s_, c_]), np.array([0, -s_, c_])]
    amp = np.ones_like(r)
    for u in rays:
        t = np.maximum((Xg * u).sum(-1), 0.0)
        amp = amp * np.tanh(np.linalg.norm(Xg - t[..., None] * u, axis=-1) / 1.2)
    p = np.cross(n, d)
    nn = np.einsum("...i,...j->...ij", n, n)
    B = 0.5 * delta * (np.eye(3) - nn) + 0.5 * delta * amp[..., None, None] * (
        np.einsum("...i,...j->...ij", d, d) - np.einsum("...i,...j->...ij", p, p))
    mb = (1 + delta) / 3
    return A.to6(mb * np.eye(3) + np.tanh(r / 1.2)[..., None, None] * (nn + B - mb * np.eye(3)))


# ----------------------------------------------------------------------------- links

def _pairs(N):
    for ax in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[ax] = slice(0, N - 1)
        s1[ax] = slice(1, N)
        yield tuple(s0), tuple(s1)


def jump_links(m6, x, delta, region):
    """Links with both nodes near the vacuum (s < 1e-4 at this delta) and both in `region`: the largest angle of the
    top eigenvectors, and the jump links (angle above 45 degrees) with their radii."""
    M = A.from6(m6)
    s = A.sfun(M, ck(delta))
    _, V = eigs(m6)
    n = V[..., :, 0]
    X = mesh(x)
    big, J, rad = 0.0, 0, []
    for s0, s1 in _pairs(len(x)):
        both = (s[s0] < 1e-4) & (s[s1] < 1e-4) & region[s0] & region[s1]
        ang = np.degrees(np.arccos(np.clip(np.abs((n[s0] * n[s1]).sum(-1)), 0, 1)))
        if both.any():
            big = max(big, float(ang[both].max()))
        Jm = both & (ang > 45.0)
        J += int(Jm.sum())
        if Jm.any():
            rad += np.linalg.norm(X[s0][Jm], axis=-1).tolist()
    return dict(largest_deg=big, jumps=J, radii=[float(min(rad)), float(max(rad))] if rad else None)


def minor_jumps(m6, delta, region):
    """Links with both nodes g23 >= delta/2 and in `region` whose e2 differ by more than 45 degrees."""
    w, V = eigs(m6)
    e2 = V[..., :, 1]
    g23 = w[..., 1] - w[..., 2]
    big, J = 0.0, 0
    for s0, s1 in _pairs(m6.shape[0]):
        ok = (g23[s0] >= delta / 2) & (g23[s1] >= delta / 2) & region[s0] & region[s1]
        ang = np.degrees(np.arccos(np.clip(np.abs((e2[s0] * e2[s1]).sum(-1)), 0, 1)))
        if ok.any():
            big = max(big, float(ang[ok].max()))
        J += int((ok & (ang > 45.0)).sum())
    return dict(largest_deg=big, jumps=J)


def minor_jump_split(m6, ref6, x, delta, rings, c, R1, R2):
    """The supplement's record of clause (iv): minor jump links (both nodes g23 >= delta/2 and outside the tubes, e2
    more than 45 degrees apart) of the start and of its reference, by the region of the link's midpoint about the
    collar's center (inside R1, the collar, outside R2), and whether they are the same links."""
    out_t = ~tubes(x, rings)
    X = mesh(x) - np.asarray(c, float)
    res = {}
    sets = {}
    for lab, f6 in (("start", m6), ("reference", ref6)):
        w, V = eigs(f6)
        e2 = V[..., :, 1]
        g23 = w[..., 1] - w[..., 2]
        reg = dict(inside_R1=[0, 0.0], collar=[0, 0.0], outside_R2=[0, 0.0])
        links = {"inside_R1": set(), "collar": set(), "outside_R2": set()}
        places = []
        for ax, (s0, s1) in enumerate(_pairs(len(x))):
            ok = (g23[s0] >= delta / 2) & (g23[s1] >= delta / 2) & out_t[s0] & out_t[s1]
            ang = np.degrees(np.arccos(np.clip(np.abs((e2[s0] * e2[s1]).sum(-1)), 0, 1)))
            J = ok & (ang > 45.0)
            if not J.any():
                continue
            mid = 0.5 * (X[s0] + X[s1])
            rho = np.linalg.norm(mid, axis=-1)
            for idx in np.argwhere(J):
                t = tuple(idx)
                rr = float(rho[t])
                k = "inside_R1" if rr <= R1 else ("collar" if rr < R2 else "outside_R2")
                reg[k][0] += 1
                reg[k][1] = max(reg[k][1], float(ang[t]))
                links[k].add((ax,) + t)
                places.append(dict(axis=ax, rho=rr, angle=float(ang[t]), g23=[float(g23[s0][t]), float(g23[s1][t])]))
        res[lab] = dict(counts={k: v[0] for k, v in reg.items()}, largest_deg={k: v[1] for k, v in reg.items()}, links=places)
        sets[lab] = links
    res["outside_collar_equal"] = bool(sets["start"]["inside_R1"] == sets["reference"]["inside_R1"]
                                       and sets["start"]["outside_R2"] == sets["reference"]["outside_R2"])
    res["new_in_collar"] = len(sets["start"]["collar"] - sets["reference"]["collar"])
    return res


def start_rule(m6, x, delta, Th, R1, R2, rings, minor=False):
    """(i) nominal largest turn per link |Theta| h / (R2 - R1) <= pi/4; (ii) no jump link with both nodes outside
    the tubes; (iv) at delta > 0, no minor jump link outside the tubes."""
    h = x[1] - x[0]
    width = R2 - R1
    nominal = abs(Th) * h / width if width > 0 else None
    out_t = ~tubes(x, rings)
    jl = jump_links(m6, x, delta, out_t)
    out = dict(collar_width=float(width), nominal_deg=None if nominal is None else float(np.degrees(nominal)),
               clause_i=bool(nominal is not None and nominal <= np.pi / 4 + 1e-12), largest_vacuum_deg=jl["largest_deg"],
               jump_links=jl["jumps"], jump_radii=jl["radii"], clause_ii=bool(jl["jumps"] == 0))
    ok = out["clause_i"] and out["clause_ii"]
    if minor:
        mj = minor_jumps(m6, delta, out_t)
        out.update(minor_largest_deg=mj["largest_deg"], minor_jump_links=mj["jumps"], clause_iv=bool(mj["jumps"] == 0))
        ok = ok and out["clause_iv"]
    out["passes"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- the turn reader

def qmul(p, q):
    a1, b1, c1, d1 = p
    a2, b2, c2, d2 = q
    return np.array([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                     a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])


def qrot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def qexp(v):
    t = np.linalg.norm(v)
    if t < 1e-15:
        return np.array([1.0, *(0.5 * v)])
    return np.array([np.cos(t / 2), *(np.sin(t / 2) * v / t)])


def _fit(Ms, Rs, dr, q, w0, first=True):
    def cost(p):
        Q0 = qrot(qmul(q, qexp(p[:3])))
        Qs = Q0 @ rotvec(dr[:, None] * p[3:][None, :]) if first else np.broadcast_to(Q0, (len(Ms), 3, 3))
        D = Ms - Qs @ Rs @ np.swapaxes(Qs, -1, -2)
        return float((D * D).sum())
    p0 = np.r_[np.zeros(3), w0] if first else np.zeros(3)
    res = minimize(cost, p0, method="BFGS", options=dict(gtol=1e-12))
    qn = qmul(q, qexp(res.x[:3]))
    qn /= np.linalg.norm(qn)
    if qn @ q < 0:
        qn = -qn
    return qn, (res.x[3:] if first else None), float(np.sqrt(res.fun / float((Ms * Ms).sum())))


def unwrap_angle(q, a, prev):
    p = 2 * np.arctan2(q[1:] @ a, q[0])
    while p - prev > np.pi:
        p -= 2 * np.pi
    while p - prev < -np.pi:
        p += 2 * np.pi
    return p


def shells(x, c, R1, outer):
    h = x[1] - x[0]
    kmax = int(np.floor(outer / h + 1e-9))
    return [k for k in range(kmax, 0, -1) if k * h - h / 2 >= R1 - 1e-9]


def turn_reader(m6, ref6, x, c, R1, outer=None, exclude=None, axis=None, no_continuation=False, every=1, dens=None):
    """The turn profile of section 2 with its regime. `dens` = (edens of the state, edens of the reference) gives the
    share of dU by shell."""
    h = x[1] - x[0]
    L = x[-1]
    outer = L - 2 * h if outer is None else outer
    X = mesh(x) - np.asarray(c, float)
    rho = np.linalg.norm(X, axis=-1)
    M = A.from6(m6)
    Rf = A.from6(ref6)
    w, _ = eigs(m6)
    g12 = w[..., 0] - w[..., 1]
    keep = np.ones(rho.shape, bool) if exclude is None else ~exclude
    ks = shells(x, c, R1, outer)[::every]
    q = np.array([1.0, 0, 0, 0])
    wv = np.zeros(3)
    rows = []
    prev_ang = 0.0
    a = None if axis is None else unit(axis)
    for k in ks:
        rk = k * h
        S = (rho >= rk - h / 2) & (rho < rk + h / 2) & keep
        qn, wv, r = _fit(M[S], Rf[S], rho[S] - rk, q, wv)
        if no_continuation and qn[0] < 0:
            qn = -qn
        row = dict(rho=float(rk), n=int(S.sum()), q=qn.tolist(), r=r, step=float(np.linalg.norm(qn - q)) if rows else float(np.linalg.norm(qn - np.array([1.0, 0, 0, 0]))),
                   chi=float(2 * np.arccos(min(1.0, abs(qn[0])))), g12_min=float(g12[S].min()))
        if a is not None:
            prev_ang = unwrap_angle(qn, a, prev_ang)
            row["angle"] = float(prev_ang)
        if dens is not None:
            row["dU_shell"] = float((dens[0] - dens[1])[S].sum())
        rows.append(row)
        q = qn
    q0s = np.array([r["q"][0] for r in rows])
    sigma = int(np.sign(q0s[-1])) if q0s[-1] != 0 else 0
    steps = [r["step"] for r in rows[1:]]
    regime = dict(r_ok=bool(max(r["r"] for r in rows) <= 0.95), g12_ok=bool(min(r["g12_min"] for r in rows) >= 0.5),
                  step_ok=bool((max(steps) if steps else 0.0) <= 0.8), outer_ok=bool(q0s[0] >= 0.3), sign_ok=bool(abs(q0s[-1]) >= 0.5))
    in_regime = all(regime.values())
    # transition radius: the outermost crossing of q0 through 0 (linear between shells), and the radius of least q0
    rho_t = None
    for i in range(len(rows) - 1):
        if q0s[i] > 0 >= q0s[i + 1] or q0s[i] >= 0 > q0s[i + 1]:
            t = q0s[i] / (q0s[i] - q0s[i + 1]) if q0s[i] != q0s[i + 1] else 0.0
            rho_t = float(rows[i]["rho"] + t * (rows[i + 1]["rho"] - rows[i]["rho"]))
            break
    out = dict(rows=rows, sigma=sigma, q0_sign_shell=float(q0s[-1]), sign_shell=rows[-1]["rho"], chi_max=float(max(r["chi"] for r in rows)),
               rho_t_cross=rho_t, rho_least_q0=float(rows[int(np.argmin(q0s))]["rho"]), r_max=float(max(r["r"] for r in rows)),
               step_max=float(max(steps) if steps else 0.0), q0_outer=float(q0s[0]), regime=regime, in_regime=bool(in_regime))
    if dens is not None:
        dU = float((dens[0] - dens[1]).sum())
        out["dU_total"] = dU
        out["dU_inside_sign_shell"] = float((dens[0] - dens[1])[rho < rows[-1]["rho"] - h / 2].sum())
    return out


def interior_fit(m6, ref6, x, mask, q_start, axis, angle_start):
    """One rotation fitted on the nodes of `mask` against the reference, continued from the sign shell's q."""
    M = A.from6(m6)
    Rf = A.from6(ref6)
    qi, _, ri = _fit(M[mask], Rf[mask], None, np.asarray(q_start, float), None, first=False)
    psi = unwrap_angle(qi, unit(axis), angle_start)
    return dict(r_int=ri, psi_int=float(psi), q=qi.tolist(), nodes=int(mask.sum()), in_regime=bool(ri <= 0.1))


# ----------------------------------------------------------------------------- the tear reads

def tear_reads(m6, x, delta, tube_mask, region):
    """Z = {g12 < 0.5} in 26-neighborhood components; the ring's component holds the node of least g12 in the tube.
    Reads over `region` minus the tube (rung 1), or over `region` (rungs 2 and 3, tube_mask None)."""
    w, _ = eigs(m6)
    g12 = w[..., 0] - w[..., 1]
    X = mesh(x)
    Z = g12 < GAP_Z
    lab, k = ndimage.label(Z, structure=np.ones((3, 3, 3)))
    outside = region.copy() if tube_mask is None else (region & ~tube_mask)
    ring_lab = 0
    if tube_mask is not None and tube_mask.any():
        gi = np.where(tube_mask, g12, np.inf)
        i = np.unravel_index(np.argmin(gi), gi.shape)
        ring_lab = int(lab[i]) if Z[i] else 0
    comps = []
    for j in range(1, k + 1):
        n_out = int(((lab == j) & outside).sum())
        if n_out:
            comps.append(dict(label=j, ring=bool(j == ring_lab), nodes_outside=n_out))
    go = np.where(outside, g12, np.inf)
    i = np.unravel_index(np.argmin(go), go.shape)
    jl = jump_links(m6, x, delta, outside)
    other = [c for c in comps if not c["ring"]]
    ringout = [c for c in comps if c["ring"]]
    return dict(components_outside=comps, other_components_outside=len(other), other_nodes_outside=int(sum(c["nodes_outside"] for c in other)),
                ring_nodes_outside=int(sum(c["nodes_outside"] for c in ringout)), least_g12_outside=float(go[i]), least_g12_place=X[i].tolist(),
                least_g12_radius=float(np.linalg.norm(X[i])), jump_links=jl["jumps"], largest_vacuum_deg=jl["largest_deg"],
                z_nodes_outside=int((Z & outside).sum()),
                tear=bool((Z & outside).any() or jl["jumps"] > 0))


# ----------------------------------------------------------------------------- the two-ring reader

def _cube_q(m6, x, center, hw):
    N = len(x)
    h = x[1] - x[0]
    ci = np.rint((np.asarray(center, float) - x[0]) / h).astype(int)
    m = int(round(hw / h))
    lo, hi = ci - m, ci + m
    if (lo < 0).any() or (hi > N - 1).any():
        return dict(read=False, reason="outside the box")
    n, g = DG.director(m6)
    r = DG.Surface(N, lo, hi, half_width=hw).read(n, g)
    return dict(read=True, liftable=r["liftable"], abs_q=r["abs_q"], q=r["q"], reason=r["reason"], lo=lo.tolist(), hi=hi.tolist())


def circle_points(c, nu, R, n=360):
    nu, e1, e2 = RD.frame(nu)
    ph = 2 * np.pi * np.arange(n) / n
    return np.asarray(c) + R * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2)


def two_ring_reader(m6, x, thr=GAP_Z):
    h = x[1] - x[0]
    L = x[-1]
    M = A.from6(m6)
    w, _ = eigs(m6)
    g = w[..., 0] - w[..., 1]
    X = mesh(x)
    r = np.linalg.norm(X, axis=-1)
    Z = (g < thr) & (r < 0.93 * L)
    lab, k = ndimage.label(Z, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel(), minlength=k + 1)[1:]
    comps = [j + 1 for j in np.argsort(sizes)[::-1] if sizes[j] >= 4]
    out = dict(components=len(comps), sizes=[int(sizes[j - 1]) for j in comps], all_components=int(k), rings=[])
    for j in comps[:2] if len(comps) <= 2 else []:
        msk = lab == j
        pts = X[msk]
        wt = thr - g[msk]
        cc = (pts * wt[:, None]).sum(0) / wt.sum()
        q = pts - cc
        C = (q[:, :, None] * q[:, None, :] * wt[:, None, None]).sum(0) / wt.sum()
        nu = np.linalg.eigh(C)[1][:, 0]
        R = float((np.sqrt(np.maximum((q ** 2).sum(1) - (q @ nu) ** 2, 0)) * wt).sum() / wt.sum())
        for _ in range(2):
            mp = RD.meridian_pass(m6, x, cc, nu, R)
            P = np.array([o["p"] for o in mp])
            cc, nu, R = RD.fit_circle(P)
        gs = np.array([o["g"] for o in mp])
        ht = [o["halfturn"] for o in mp]
        dcons = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        out["rings"].append(dict(label=int(j), nodes=int(msk.sum()), c=np.asarray(cc).tolist(), nu=unit(nu).tolist(), RL=float(R),
                                 regime=bool((gs <= 0.02).all() and all(ht)), gmax=float(gs.max()), halfturn_all=bool(all(ht)),
                                 all_crossings=bool((gs <= 0.02).all()), max_consecutive=float(dcons.max()), cls=RD.orientation(nu)["cls"]))
    rings = out["rings"]
    if len(rings) == 2:
        for i, rg in enumerate(rings):
            other = rings[1 - i]
            keep = ~(lab == other["label"])
            P = X[(g < 0.15) & keep] - np.asarray(rg["c"])
            nu = np.asarray(rg["nu"])
            z = P @ nu
            rp = np.sqrt(np.maximum((P ** 2).sum(1) - z ** 2, 0.0))
            dist = np.sqrt((rp - rg["RL"]) ** 2 + z ** 2)
            far = int((dist > 1.5).sum())
            rg["far_nodes"] = far
            rg["one_ring"] = bool(rg["all_crossings"] and rg["max_consecutive"] <= 2 * h and far == 0)
            hw = float(np.ceil((rg["RL"] + 1.5) / h - 1e-9) * h)
            ci = np.rint((np.asarray(rg["c"]) - x[0]) / h).astype(int)
            cnode = x[0] + ci * h
            m = int(round(hw / h))
            inside = np.all(np.abs(X - cnode) <= hw + 1e-9, axis=-1)
            dother = ring_distance(X[inside], other["c"], other["nu"], other["RL"])
            if (dother <= 2.0).any():
                rg["cube_q"] = dict(read=False, reason="a node within 2.0 of the other circle", half_width=hw)
            else:
                rg["cube_q"] = _cube_q(m6, x, cnode, hw)
                rg["cube_q"]["half_width"] = hw
        P0 = circle_points(rings[0]["c"], rings[0]["nu"], rings[0]["RL"])
        P1 = circle_points(rings[1]["c"], rings[1]["nu"], rings[1]["RL"])
        out["circle_distance"] = float(np.min(np.linalg.norm(P0[:, None] - P1[None], axis=-1)))
        out["separation"] = float(np.linalg.norm(np.asarray(rings[0]["c"]) - np.asarray(rings[1]["c"])))
        balls = np.zeros(g.shape, bool)
        for rg in rings:
            balls |= np.linalg.norm(X - np.asarray(rg["c"]), axis=-1) <= rg["RL"] + 1.5
        Mr = M[::-1, ::-1, :]
        out["D"] = float(np.linalg.norm(Mr - M, axis=(-1, -2))[balls].max())
        # ring 1: c_x > 0
        if rings[0]["c"][0] < rings[1]["c"][0]:
            out["rings"] = rings = [rings[1], rings[0]]
    out["box_q"] = _cube_q(m6, x, np.zeros(3), L - h)
    if len(comps) == 2 and len(rings) == 2 and all(rg["regime"] and rg["one_ring"] for rg in rings) and out["circle_distance"] > 3.0 \
            and all(rg["cube_q"].get("read") and rg["cube_q"].get("liftable") and rg["cube_q"].get("abs_q") == 1 for rg in rings):
        out["word"] = "TWO RINGS"
    elif len(comps) == 1:
        out["word"] = "ONE OBJECT"
        msk = lab == comps[0]
        pts = X[msk]
        wt = thr - g[msk]
        cc = (pts * wt[:, None]).sum(0) / wt.sum()
        out["one_object"] = dict(nodes=int(msk.sum()), centroid=cc.tolist(), least_g12=float(g[msk].min()))
    else:
        out["word"] = "neither"
    return out


# ----------------------------------------------------------------------------- the minor reads (delta > 0)

def minor_part(m6):
    M = A.from6(m6)
    w, V = eigs(m6)
    e1 = V[..., :, 0]
    P = np.einsum("...i,...j->...ij", e1, e1)
    return M - w[..., 0, None, None] * P - 0.5 * (w[..., 1] + w[..., 2])[..., None, None] * (np.eye(3) - P)


def mu(m6, ref6, region):
    B = minor_part(m6)[region]
    Br = minor_part(ref6)[region]
    w, _ = eigs(m6)
    wr, _ = eigs(ref6)
    reg = bool(min((w[..., 0] - w[..., 1])[region].min(), (wr[..., 0] - wr[..., 1])[region].min()) >= 0.5)
    return dict(mu=float(np.sqrt(((B - Br) ** 2).sum() / (Br ** 2).sum())), in_regime=reg)


def minor_profile(m6, ref6, x, c, R1, outer=None, exclude=None, e1_ref=None, signless=False):
    """The minor turn profile of section 10: alpha_k per shell, the coherence, the alignment, alpha_s."""
    h = x[1] - x[0]
    L = x[-1]
    outer = L - 2 * h if outer is None else outer
    Xc = mesh(x) - np.asarray(c, float)
    rr = np.linalg.norm(Xc, axis=-1)
    B = minor_part(m6)
    Br = minor_part(ref6)
    if e1_ref is None:
        if signless:
            e1 = eigs(ref6)[1][..., :, 0]
        else:
            e1 = e1_outward(ref6, x, c)
    else:
        e1 = e1_ref
    _, Vn = eigs(m6)
    _, Vr = eigs(ref6)
    al = np.abs((Vn[..., :, 0] * Vr[..., :, 0]).sum(-1))
    t = np.where(np.abs(e1[..., :1]) < 0.9, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    u = t - (t * e1).sum(-1, keepdims=True) * e1
    u /= np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-300)
    v = np.cross(e1, u)

    def spin2(Bm):
        return 0.5 * (np.einsum("...i,...ij,...j->...", u, Bm, u) - np.einsum("...i,...ij,...j->...", v, Bm, v)) + \
            1j * np.einsum("...i,...ij,...j->...", u, Bm, v)
    b = spin2(B)
    br = spin2(Br)
    keep = np.ones(rr.shape, bool) if exclude is None else ~exclude
    rows = []
    prev = None
    for k in shells(x, c, R1, outer):
        S = (rr >= k * h - h / 2) & (rr < k * h + h / 2) & keep
        z = (b[S] * np.conj(br[S])).sum()
        coh = float(abs(z) / max((np.abs(b[S]) * np.abs(br[S])).sum(), 1e-300))
        ph = float(np.angle(z))
        if prev is not None:
            ph = prev + float(np.angle(np.exp(1j * (ph - prev))))
        prev = ph
        rows.append(dict(rho=float(k * h), alpha=ph / 2, coherence=coh, align=float(al[S].min())))
    steps = [abs(rows[i + 1]["alpha"] - rows[i]["alpha"]) for i in range(len(rows) - 1)]
    reg = dict(align_ok=bool(min(r["align"] for r in rows) >= 0.9), coherence_ok=bool(min(r["coherence"] for r in rows) >= 0.5),
               step_ok=bool((max(steps) if steps else 0.0) <= 1.0), outer_ok=bool(abs(rows[0]["alpha"]) <= np.pi / 4))
    return dict(rows=rows, alpha_s=rows[-1]["alpha"], sign_shell=rows[-1]["rho"], regime=reg, in_regime=bool(all(reg.values())),
                min_coherence=float(min(r["coherence"] for r in rows)), min_align=float(min(r["align"] for r in rows)))


def lines_read(m6, x, c, nu, radii, delta, tube_mask=None):
    h = x[1] - x[0]
    X = mesh(x) - np.asarray(c, float)
    rho = np.linalg.norm(X, axis=-1)
    w, _ = eigs(m6)
    g23 = w[..., 1] - w[..., 2]
    thr = delta / 2
    keep = np.ones(rho.shape, bool) if tube_mask is None else ~tube_mask
    nu = unit(nu)
    out = []
    for Rr in radii:
        S = (np.abs(rho - Rr) < h / 2) & keep
        Zs = S & (g23 < thr)
        lab, k = ndimage.label(Zs, structure=np.ones((3, 3, 3)))
        comps = []
        for j in range(1, k + 1):
            sel = lab == j
            P = X[sel]
            wts = thr - g23[sel]
            cen = (P * wts[:, None]).sum(0) / wts.sum()
            ang = float(np.degrees(np.arccos(np.clip(cen @ nu / np.linalg.norm(cen), -1, 1))))
            comps.append(dict(polar_deg=ang, nodes=int(sel.sum())))
        comps.sort(key=lambda q: q["polar_deg"])
        nS = int(S.sum())
        reg = bool(all(q["nodes"] <= 0.1 * nS for q in comps))
        out.append(dict(R=float(Rr), count=int(k) if reg else None, comps=comps, shell_nodes=nS, in_regime=reg))
    return out


def minor_torque(mod, m6):
    """max over free nodes of |<dU/dM(n), [K_n, M(n)]>| / h^3, K_n the generator about the node's own e1."""
    out = mod.evaluate(np.asarray(m6, float), need_grad=True)
    G = mod.gU
    M = A.from6(m6)
    _, V = eigs(m6)
    K = skew(V[..., :, 0])
    tq = (G * (K @ M - M @ K)).sum((-1, -2)) * mod.mask
    return dict(torque=float(np.abs(tq).max() / mod.h ** 3), gate=float(np.abs(out["grad"]).max() / mod.h ** 3), U=out["U"], U4=out["U4"],
                UV=out["UV"], ULV=out["ULV"])


# ----------------------------------------------------------------------------- the path tables

def path_table(mod, ref6, x, axis, c, R1, R2, top=2.0, sign=1.0, kind="T"):
    """U along T[psi] (or W[psi]) for psi/pi from top down to 0 in steps of 0.125, against the reference."""
    p0 = mod.evaluate(ref6, need_grad=False)
    rows = []
    for k in range(int(round(top * 8)), -1, -1):
        psi = sign * k * np.pi / 8
        f6 = T_turn(ref6, x, psi, axis, c, R1, R2) if kind == "T" else W_turn(ref6, ref6, x, psi, c, R1, R2)
        p = mod.evaluate(f6, need_grad=False)
        rows.append(dict(psi_over_pi=sign * k / 8, dU=p["U"] - p0["U"], dUV=p["UV"] - p0["UV"]))
    dU = [r["dU"] for r in rows]
    rises = [dU[i + 1] - dU[i] for i in range(len(dU) - 1)]
    mono = all(q < 0 for q in rises)
    return dict(rows=rows, word="MONOTONE" if mono else "BARRIER", largest_rise=float(max(rises)), U_ref=p0["U"],
                max_abs_dUV=float(max(abs(r["dUV"]) for r in rows)))


# ----------------------------------------------------------------------------- the controls

PLAN_H1_PATH = dict(normal=[668.0, 485.3, 347.5, 245.5, 171.4, 118.5, 81.3, 55.6, 37.9, 25.7, 17.2, 11.2, 6.9, 3.8, 1.7, 0.4, 0],
                    inplane=[587.1, 427.0, 306.3, 216.8, 151.6, 105.0, 72.3, 49.5, 33.8, 23.0, 15.4, 10.0, 6.2, 3.4, 1.5, 0.4, 0])
PLAN_D5_PATH = dict(z=[498.31, 363.65, 261.91, 186.35, 131.20, 91.61, 63.61, 44.02, 30.38, 20.85, 14.11, 9.26, 5.71, 3.15, 1.39, 0.34, 0],
                    x=[587.03, 425.76, 304.29, 214.44, 149.22, 102.75, 70.22, 47.76, 32.39, 21.87, 14.60, 9.48, 5.81, 3.19, 1.40, 0.35, 0])


def H1_reference():
    m6, x, e = load51("G6_mid_a")
    return m6, x, ring_of(e), e


def c3a_check(m6, x, ring, R2, label, extra_fail=None, delta=0.0):
    """C3 (a) on one reference: T[2pi], T[4pi], T[1.3] about both axes and the untouched reference."""
    h = x[1] - x[0]
    c, nu = ring["c"], ring["nu"]
    R1 = ring["RL"] + 1.5
    ip = inplane_axis(nu)
    cases = []
    ok = True
    for Th in (2 * np.pi, 4 * np.pi, 1.3):
        for an, a in (("normal", nu), ("inplane", ip)):
            f6 = T_turn(m6, x, Th, a, c, R1, R2)
            tr = turn_reader(f6, m6, x, c, R1, axis=a)
            inner = [r for r in tr["rows"] if r["rho"] - h >= R1 - 1e-9 and r["rho"] + h <= R2 + 1e-9]
            e_in = max(abs(r["angle"] - Th * fprof(r["rho"], R1, R2)) for r in inner)
            r_in = max(r["r"] for r in inner)
            e_all = max(abs(r["angle"] - Th * fprof(r["rho"], R1, R2)) for r in tr["rows"])
            want = -1 if abs(Th - 2 * np.pi) < 1e-12 else 1
            good = bool(r_in <= 1e-6 and e_in <= 1e-5 and e_all <= 0.15 and tr["r_max"] <= 0.15 and tr["sigma"] == want and tr["in_regime"])
            ok &= good
            cases.append(dict(Theta=Th, axis=an, r_inner_max=r_in, angle_err_inner_max=e_in, angle_err_all_max=e_all, r_max=tr["r_max"],
                              sigma=tr["sigma"], q0_sign_shell=tr["q0_sign_shell"], sign_shell=tr["sign_shell"], in_regime=tr["in_regime"],
                              regime=tr["regime"], step_max=tr["step_max"], chi_max=tr["chi_max"], passed=good))
    tr = turn_reader(m6, m6, x, c, R1, axis=nu)
    good = bool(tr["sigma"] == 1 and tr["in_regime"] and tr["r_max"] <= 0.15)
    ok &= good
    cases.append(dict(Theta=0.0, axis="untouched", r_max=tr["r_max"], sigma=tr["sigma"], q0_sign_shell=tr["q0_sign_shell"], in_regime=tr["in_regime"],
                      chi_max=tr["chi_max"], passed=good))
    fails = []
    f6 = T_turn(m6, x, 2 * np.pi, nu, c, R1, R2)
    tr = turn_reader(f6, m6, x, c, R1, axis=nu, no_continuation=True)
    fails.append(dict(input="a reader without continuation on T[2pi]", sigma=tr["sigma"], fails_as_planned=bool(tr["sigma"] == 1)))
    Nu = unit([0.2, 0.3, 0.93])
    uni = np.broadcast_to(A.to6(np.outer(Nu, Nu)), m6.shape).copy()
    tr = turn_reader(uni, m6, x, c, R1)
    fails.append(dict(input="a uniform field", r_min=float(min(r["r"] for r in tr["rows"])), r_max=tr["r_max"],
                      fails_as_planned=bool(min(r["r"] for r in tr["rows"]) > 0.95)))
    if extra_fail is not None:
        fails.append(extra_fail)
    return dict(control=f"C3 (a) {label}", passed=bool(ok), cases=cases, failing_inputs=fails)


def c3a_h2_fail(m6h2, xh2, ring):
    """Failing input (ii): the H2 4pi start read one shell in four."""
    R1 = ring["RL"] + 1.5
    f6 = T_turn(m6h2, xh2, 4 * np.pi, ring["nu"], ring["c"], R1, 11.0)
    tr = turn_reader(f6, m6h2, xh2, ring["c"], R1, axis=ring["nu"], every=4)
    return dict(input="the H2 4pi start read one shell in four", step_max=tr["step_max"], r_max=tr["r_max"], regime=tr["regime"],
                fails_the_control=bool(not tr["in_regime"]), fails_as_planned=bool(tr["step_max"] > 0.8))


def interp_to(m6, xs, x, order=3):
    hs = xs[1] - xs[0]
    I = np.meshgrid(*([(x - xs[0]) / hs] * 3), indexing="ij")
    return np.stack([map_coordinates(m6[..., k], I, order=order) for k in range(6)], -1)


def control_c1():
    rows = []
    ok = True
    for name, grid in (("G6_mid_a", (12.0, 49)), ("G4_mid_a", (9.0, 37)), ("G5_mid_a", (9.0, 49))):
        m6, x, e = load51(name)
        mod = A.Model(len(x), grid[0], weight="mid")
        p0 = mod.evaluate(m6, False)
        Q = rot(unit([0.3, -0.5, 0.8]), np.array(1.234))
        mc = A.to6(conj(Q, A.from6(m6)))
        pc = mod.evaluate(mc, False)
        rel = {k: abs(pc[k] - p0[k]) / abs(p0[k]) for k in ("U4", "UV", "ULV", "U")}
        h = x[1] - x[0]
        P = mesh(x) @ Q
        idx = [(P[..., k] - x[0]) / h for k in range(3)]
        mp = np.stack([map_coordinates(m6[..., k], idx, order=1, mode="nearest") for k in range(6)], -1)
        mp = mod.hold_faces(mp)
        fr = abs(mod.evaluate(mp, False)["U"] / p0["U"] - 1)
        good = bool(max(rel.values()) <= 1e-13)
        ok &= good
        rows.append(dict(state=name, rel=rel, passed=good, failing_input_positions_turned_rel=fr, fails_as_planned=bool(fr > 1e-6)))
    return dict(control="C1 the constant turn", passed=bool(ok), rows=rows)


def control_c2():
    rows = []
    ok = True
    for name, hb in (("G6_mid_a", 12.0), ("G4_mid_a", 9.0), ("G5_mid_a", 9.0)):
        m6, x, e = load51(name)
        h = x[1] - x[0]
        rg = ring_of(e)
        R1, R2 = rg["RL"] + 1.5, hb - 2 * h
        mod = A.Model(len(x), hb, weight="mid")
        p0 = mod.evaluate(m6, False)
        rho = np.linalg.norm(mesh(x) - rg["c"], axis=-1)
        s0 = A.sfun(A.from6(m6))
        sel = (rho <= R1) | (rho >= R2)
        for Th in (2 * np.pi, 4 * np.pi):
            f6 = T_turn(m6, x, Th, rg["nu"], rg["c"], R1, R2)
            d_io = float(np.abs(f6 - m6)[sel].max())
            d_s = float(np.abs(A.sfun(A.from6(f6)) - s0).max())
            p = mod.evaluate(f6, False)
            d_uv = abs(p["UV"] - p0["UV"]) / p0["UV"]
            good = bool(d_io <= 1e-15 and d_s <= 1e-15 and d_uv <= 1e-14)
            ok &= good
            rows.append(dict(state=name, Theta=Th, inside_outside=d_io, s=d_s, UV_rel=d_uv, passed=good))
        fl = []
        for Th in (1.3, np.pi):
            f6 = T_turn(m6, x, Th, rg["nu"], rg["c"], R1, R2)
            fl.append(float(np.abs(f6 - m6)[rho <= R1].max()))
        rows.append(dict(state=name, failing_inputs_inside_change=fl, fails_as_planned=bool(min(fl) > 0.1)))
    return dict(control="C2 the collar turn is exact", passed=bool(ok), rows=rows)


def control_c3b():
    m6, x, ring, e = H1_reference()
    c, nu, R1 = ring["c"], ring["nu"], ring["RL"] + 1.5
    L = x[-1]
    X = mesh(x) - c
    rho = np.linalg.norm(X, axis=-1)
    cos_t = (X @ nu) / np.maximum(rho, 1e-12)
    inf = np.abs(X + c).max(-1)
    cases = dict(
        shifted=fprof(np.linalg.norm(X - nu, axis=-1), R1, 11.0),
        P2=fprof(rho * (1 + 0.25 * 0.5 * (3 * cos_t ** 2 - 1)), R1, 11.0),
        to_faces=1 - np.clip((rho - R1) / (L * rho / np.maximum(inf, 1e-12) - R1), 0, 1),
        cube_edges=1 - np.clip((inf - R1) / (L - R1), 0, 1))
    rows = []
    ok = True
    worst = 0.0
    for nm, f in cases.items():
        f6 = T_turn(m6, x, 2 * np.pi, nu, c, R1, 11.0, fmap=f)
        tr = turn_reader(f6, m6, x, c, R1, axis=nu)
        good = bool(tr["sigma"] == -1 and tr["in_regime"])
        ok &= good
        worst = max(worst, tr["r_max"])
        rows.append(dict(case=nm, sigma=tr["sigma"], q0_sign_shell=tr["q0_sign_shell"], r_max=tr["r_max"], step_max=tr["step_max"],
                         q0_outer=tr["q0_outer"], in_regime=tr["in_regime"], regime=tr["regime"], passed=good))
    return dict(control="C3 (b) deformed collars", passed=bool(ok), rows=rows, largest_residual=worst,
                note="the cube-edged collar reads f at the infinity norm with R2 = L (the faces)")


def control_c8(d5=False):
    if not d5:
        m6, x, ring, e = H1_reference()
        mod = A.Model(49, 12.0, weight="mid")
        axes = dict(normal=ring["nu"], inplane=inplane_axis(ring["nu"]))
        plan, tol = PLAN_H1_PATH, 0.05
        c, R1 = ring["c"], ring["RL"] + 1.5
    else:
        x = axis_of("H1")
        m6 = seed_bi(x)
        mod = make_model("H1", DELTA, hold6=m6)
        axes = dict(z=np.array([0, 0, 1.0]), x=np.array([1.0, 0, 0]))
        plan, tol = PLAN_D5_PATH, 0.005
        c, R1 = np.zeros(3), 4.354
    rows = {}
    ok = True
    for an, a in axes.items():
        pt = path_table(mod, m6, x, a, c, R1, 11.0)
        got = [r["dU"] for r in pt["rows"]]
        err = max(abs(g - p) for g, p in zip(got, plan[an]))
        good = bool(err <= tol)
        ok &= good
        rows[an] = dict(dU=got, max_err=err, word=pt["word"], max_abs_dUV=pt["max_abs_dUV"], passed=good)
    # failing input: the turn applied to the interior only (the ball rho <= R1 turned by psi, the collar left)
    an, a = list(axes.items())[0]
    rho = np.linalg.norm(mesh(x) - c, axis=-1)
    p0 = mod.evaluate(m6, False)["U"]
    fl = []
    for k in range(16, -1, -1):
        f = np.where(rho <= R1, 1.0, 0.0)
        fl.append(mod.evaluate(T_turn(m6, x, k * np.pi / 8, a, c, R1, 11.0, fmap=f), False)["U"] - p0)
    ferr = max(abs(g - p) for g, p in zip(fl, plan[an]))
    if d5:
        rows["W_on_M0delta"] = path_table(mod, m6, x, None, c, R1, 11.0, kind="W")
    return dict(control="C8 at delta = 0.05" if d5 else "C8 the path tables", passed=bool(ok), rows=rows,
                failing_input_interior_only_max_err=ferr, fails_as_planned=bool(ferr > tol))


def constructed_pair(x, centers, nu=(0, 0, 1.0), e1=(1.0, 0, 0), R0=2.75, delta=0.0):
    X = mesh(x)
    fields = []
    for cc in centers:
        r6 = RD.ring_field(x, R0, list(nu), list(e1), c=cc)
        if delta:
            Y = X - np.asarray(cc)
            nuu = unit(nu)
            zt = Y @ nuu
            rp = Y - zt[..., None] * nuu
            ph = np.cross(nuu, rp) / np.maximum(np.linalg.norm(rp, axis=-1), 1e-300)[..., None]
            r6 = A.to6(A.from6(r6) + delta * np.einsum("...i,...j->...ij", ph, ph))
        fields.append(r6)
    d0 = np.linalg.norm(X - np.asarray(centers[0]), axis=-1)
    d1 = np.linalg.norm(X - np.asarray(centers[1]), axis=-1)
    return np.where((d0 <= d1)[..., None], fields[0], fields[1])


def control_c6():
    x = axis_of("P")
    m6 = constructed_pair(x, [(7.0, 0, 0), (-7.0, 0, 0)])
    tr = two_ring_reader(m6, x)
    ok = tr["word"] == "TWO RINGS" and tr["components"] == 2
    for rg, cx in zip(tr["rings"], (7.0, -7.0)):
        ok &= abs(rg["RL"] - 2.75) <= 0.01 and RD.angle(rg["nu"], [0, 0, 1.0]) <= 0.5 and np.linalg.norm(np.asarray(rg["c"]) - [cx, 0, 0]) <= 0.05
    ok &= tr["circle_distance"] > 3.0
    one = RD.ring_field(x, 2.75, [0, 0, 1.0], [1.0, 0, 0])
    t1 = two_ring_reader(one, x)
    co = constructed_pair(x, [(0, 0, 1.25), (0, 0, -1.25)])
    t2 = two_ring_reader(co, x)
    fails = [dict(input="one ring", components=t1["components"], word=t1["word"], fails_as_planned=bool(t1["word"] != "TWO RINGS")),
             dict(input="two coaxial rings at (0, 0, +-1.25)", components=t2["components"], sizes=t2["sizes"], circle_distance=t2.get("circle_distance"),
                  word=t2["word"], fails_as_planned=bool(t2["word"] != "TWO RINGS"))]
    return dict(control="C6 the two-ring reader", passed=bool(ok), read=_trim_tr(tr), failing_inputs=fails)


def _trim_tr(tr):
    return {k: v for k, v in tr.items()}


def control_c9():
    rows = []
    ok = True
    d = DELTA
    for grid in ("H1", "H2", "H3"):
        x = axis_of(grid)
        N = len(x)
        face = faces_mask(N)
        m6 = seed_bi(x)
        w, _ = eigs(m6)
        M = A.from6(m6)
        s = A.sfun(M, ck(d))
        X = mesh(x)
        rp = np.sqrt(X[..., 0] ** 2 + X[..., 1] ** 2)
        e1 = float(np.abs(w[..., 0][face] - 1).max())
        e2 = float(np.abs(0.5 * (w[..., 1] + w[..., 2])[face] - d / 2).max())
        e3 = float(np.abs((w[..., 1] - w[..., 2])[face] - d * np.tanh(rp[face] / 1.2)).max())
        smax = float(s[face].max())
        tr = float(np.abs(np.trace(M, axis1=-2, axis2=-1)[face] - 1 - d).max())
        tr_all = float(np.abs(np.trace(M, axis1=-2, axis2=-1) - 1 - d).max())
        cn = float(np.abs(M[N // 2, N // 2, N // 2] - (1 + d) / 3 * np.eye(3)).max())
        lr = lines_read(m6, x, np.zeros(3), [0, 0, 1.0], (6.0, 8.0, 10.0), d)
        lines_ok = all(q["in_regime"] and q["count"] == 2 and abs(q["comps"][0]["polar_deg"]) <= 1.0 and abs(q["comps"][1]["polar_deg"] - 180) <= 1.0 for q in lr)
        good = bool(e1 <= 1e-8 and e2 <= 1e-8 and e3 <= 1e-8 and smax <= 2e-6 and tr <= 1e-12 and cn <= 1e-12 and lines_ok)
        ok &= good
        rows.append(dict(grid=grid, lam1=e1, mean_minor=e2, minor_gap=e3, s_faces=smax, trace_faces=tr, trace_all_nodes_descriptor=tr_all, center=cn,
                         lines=[(q["R"], q["count"], [round(c["polar_deg"], 2) for c in q["comps"]]) for q in lr], passed=good))
    x = axis_of("H1")
    bad = seed_bi(x, iso=1.0 / 3.0)
    wb, _ = eigs(bad)
    face = faces_mask(len(x))
    sb = A.sfun(A.from6(bad), ck(DELTA))
    return dict(control="C9 the biaxial seed", passed=bool(ok), rows=rows,
                failing_input=dict(input="the isotropic part 1/3 in place of (1 + delta)/3", face_eig=np.median(wb[face], 0).tolist(), s_faces_max=float(sb[face].max()),
                                   fails_as_planned=bool(float(np.abs(wb[..., 0][face] - 1).max()) > 1e-8)))


def control_c10_c11_c12(x, refs):
    """refs: name -> (field, ring or None). C10 (W exact), C11 (the minor turn profile), C12 (mu)."""
    c, R1, R2 = np.zeros(3), 4.354, 11.0
    X = mesh(x)
    rho = np.linalg.norm(X - c, axis=-1)
    face = faces_mask(len(x))
    res10, res11 = [], []
    ok10 = ok11 = True
    for nm, (ref, rg) in refs.items():
        w2 = W_turn(ref, ref, x, 2 * np.pi, c, R1, R2)
        wp = W_turn(ref, ref, x, np.pi, c, R1, R2)
        sel = (rho <= R1) | (rho >= R2)
        a1 = float(np.abs(w2 - ref)[sel].max())
        wa, Va = eigs(ref)
        wb, Vb = eigs(w2)
        a2 = float(np.abs(wa - wb).max())
        a3 = float(np.abs((Va[..., :, 0] * Vb[..., :, 0]).sum(-1)).min())
        a4 = float(np.abs(wp - ref)[rho <= R1].max())
        w13 = W_turn(ref, ref, x, 1.3, c, R1, R2)
        f13 = float(np.abs(w13 - ref)[rho <= R1].max())
        good = bool(a1 <= 1e-15 and a2 <= 1e-14 and a3 >= 1 - 1e-14 and a4 <= 1e-14)
        ok10 &= good
        res10.append(dict(reference=nm, W2pi_ball_outside=a1, W2pi_eigenvalues=a2, W2pi_e1_dot_min=a3, Wpi_ball=a4, passed=good,
                          failing_W1p3_ball_change=f13, fails_as_planned=bool(f13 > 1e-3)))
        for Th in (2 * np.pi, np.pi, 1.3):
            wt = W_turn(ref, ref, x, Th, c, R1, R2)
            mp = minor_profile(wt, ref, x, c, R1)
            h = x[1] - x[0]
            inner = [r for r in mp["rows"] if r["rho"] - h >= R1 - 1e-9 and r["rho"] + h <= R2 + 1e-9]
            err = max(abs(r["alpha"] - Th * fprof(r["rho"], R1, R2)) for r in inner)
            good = bool(err <= 0.1 and mp["min_coherence"] >= 0.9 and mp["in_regime"])
            ok11 &= good
            res11.append(dict(reference=nm, Theta=Th, max_err=err, min_coherence=mp["min_coherence"], alpha_s=mp["alpha_s"], in_regime=mp["in_regime"],
                              regime=mp["regime"], passed=good))
        mp = minor_profile(ref, ref, x, c, R1)
        good = bool(abs(mp["alpha_s"]) <= 1e-12 and mp["min_coherence"] >= 1 - 1e-12)
        ok11 &= good
        res11.append(dict(reference=nm, Theta="untouched", alpha_s=mp["alpha_s"], min_coherence=mp["min_coherence"], passed=good))
        w2 = W_turn(ref, ref, x, 2 * np.pi, c, R1, R2)
        mp_f = minor_profile(w2, ref, x, c, R1, signless=True)
        res11.append(dict(reference=nm, failing="e1 with its sign not fixed", min_coherence=mp_f["min_coherence"],
                          fails_as_planned=bool(not mp_f["in_regime"])))
        t2 = T_turn(ref, x, 2 * np.pi, [0, 0, 1.0], c, R1, R2)
        mp_t = minor_profile(t2, ref, x, c, R1)
        res11.append(dict(reference=nm, failing="the T[2pi] copy", min_align=mp_t["min_align"], fails_as_planned=bool(not mp_t["in_regime"])))
    # C12
    res12 = {}
    ok12 = True
    col = (rho > R1) & (rho < R2)
    for nm, (ref, rg) in refs.items():
        tube = tubes(x, [rg]) if rg is not None else np.zeros(rho.shape, bool)
        free = ~face & ~tube
        w2 = W_turn(ref, ref, x, 2 * np.pi, c, R1, R2)
        wp = W_turn(ref, ref, x, np.pi, c, R1, R2)
        inside = (rho <= R1) & ~face
        mi2 = mu(w2, ref, inside)["mu"]
        mip = mu(wp, ref, inside)["mu"]
        mc = mu(w2, ref, col)
        rng = np.random.default_rng(56200 if nm == "M0delta" else 56201)
        nz = ref.copy()
        nz[~face] += 1e-4 * rng.standard_normal((int((~face).sum()), 6))
        mn = mu(nz, ref, free)
        rel = float(np.sqrt(((A.from6(w2) - A.from6(ref)) ** 2)[free].sum() / (A.from6(ref) ** 2)[free].sum()))
        good = bool(mi2 <= 1e-13 and mip <= 1e-13 and mc["mu"] >= 1.0 and mn["mu"] <= 0.01)
        ok12 &= good
        res12[nm] = dict(inside_W2pi=mi2, inside_Wpi=mip, mu_col_W2pi=mc["mu"], mu_col_regime=mc["in_regime"], mu_free_noise=mn["mu"],
                         mu_free_noise_regime=mn["in_regime"], failing_relative_norm_W2pi=rel, fails_as_planned=bool(rel <= 0.06), passed=good)
    f4 = four_line(x, 16.0)
    f4m = four_line(x, 16.0 + np.degrees(0.5 / 8.0))
    free = ~face
    mm = mu(f4m, f4, free)
    m42 = mu(f4, seed_bi(x), free)
    good = bool(mm["mu"] >= 0.05 and m42["mu"] >= 1.0)
    ok12 &= good
    res12["four_line"] = dict(mu_free_moved_copy=mm["mu"], mu_free_four_vs_two=m42["mu"], regime=[mm["in_regime"], m42["in_regime"]], passed=good)
    return (dict(control="C10 W is exact", passed=bool(ok10), rows=res10), dict(control="C11 the minor turn profile", passed=bool(ok11), rows=res11),
            dict(control="C12 mu", passed=bool(ok12), rows=res12))


def control_c13(x):
    d = DELTA
    res = []
    ok = True
    for nm, f6, want in (("M0delta", seed_bi(x), (0.0, 180.0)), ("four_16", four_line(x, 16.0), (16.0, 164.0)),
                         ("four_moved", four_line(x, 16.0 + np.degrees(0.5 / 8.0)), (16.0 + np.degrees(0.5 / 8.0), 180 - 16.0 - np.degrees(0.5 / 8.0)))):
        lr = lines_read(f6, x, np.zeros(3), [0, 0, 1.0], (6.0, 8.0, 10.0), d)
        good = True
        for q in lr:
            if not q["in_regime"]:
                good = False
                continue
            angs = [c["polar_deg"] for c in q["comps"]]
            if nm == "M0delta":
                good &= q["count"] == 2 and abs(angs[0] - want[0]) <= 1.0 and abs(angs[1] - want[1]) <= 1.0
            else:
                good &= q["count"] == 4 and all(abs(a - want[0]) <= 1.0 for a in angs[:2]) and all(abs(a - want[1]) <= 1.0 for a in angs[2:])
        ok &= bool(good)
        res.append(dict(field=nm, lines=[(q["R"], q["count"], [round(c["polar_deg"], 2) for c in q["comps"]], [c["nodes"] for c in q["comps"]]) for q in lr],
                        minor_jumps=minor_jumps(f6, d, ~faces_mask(len(x)))["jumps"], passed=bool(good)))
    sb = seed_bi(x)
    w0, V0 = eigs(sb)
    P = np.einsum("...i,...j->...ij", V0[..., :, 0], V0[..., :, 0])
    uni = A.to6(w0[..., 0, None, None] * P + 0.5 * (w0[..., 1] + w0[..., 2])[..., None, None] * (np.eye(3) - P))
    lu = lines_read(uni, x, np.zeros(3), [0, 0, 1.0], (6.0, 8.0, 10.0), d)
    X = mesh(x)
    r = np.linalg.norm(X, axis=-1) + 1e-12
    n = X / r[..., None]
    nn = np.einsum("...i,...j->...ij", n, n)
    m = np.zeros_like(n)
    sz = np.sqrt(n[..., 0] ** 2 + n[..., 1] ** 2)
    m[..., 0] = -n[..., 1] / np.maximum(sz, 1e-300)
    m[..., 1] = n[..., 0] / np.maximum(sz, 1e-300)
    m[sz < 1e-12] = [1.0, 0, 0]
    p = np.cross(n, m)
    B = 0.5 * d * (np.eye(3) - nn) + 0.5 * d * (np.einsum("...i,...j->...ij", m, m) - np.einsum("...i,...j->...ij", p, p))
    mb = (1 + d) / 3
    nocore = A.to6(mb * np.eye(3) + np.tanh(r / 1.2)[..., None, None] * (nn + B - mb * np.eye(3)))
    ln = lines_read(nocore, x, np.zeros(3), [0, 0, 1.0], (6.0, 8.0, 10.0), d)
    fails = [dict(input="the minor block made uniaxial", regime=[q["in_regime"] for q in lu], fails_as_planned=bool(not any(q["in_regime"] for q in lu))),
             dict(input="the seed with no core", counts=[q["count"] for q in ln], minor_jumps=minor_jumps(nocore, d, ~faces_mask(len(x)))["jumps"],
                  fails_as_planned=bool(all(q["count"] == 0 for q in ln)))]
    return dict(control="C13 the lines read", passed=bool(ok), rows=res, failing_inputs=fails)


def control_c14():
    x = axis_of("P")
    N = len(x)
    face = faces_mask(N)
    ref14 = pair_bi(x, 14.0)
    ci = (N - 1) // 2
    h = 0.5
    rows = []
    ok = True
    allreg = np.ones((N, N, N), bool)
    for d0 in (14.0, 10.0, 6.0, 0.0):
        m6 = pair_bi(x, d0)
        fd = float(np.abs(m6 - ref14)[face].max())
        fl = float(np.abs(pair_bi(x, d0, use_ladder=False) - ref14)[face].max())
        box = _cube_q(m6, x, np.zeros(3), 18.0 - h)
        cubes = []
        if d0 > 0:
            hw = min(4.0, d0 / 2 - h)
            for s in (+1, -1):
                cubes.append(_cube_q(m6, x, np.array([s * d0 / 2, 0, 0]), hw))
            want_c = 1
        else:
            cubes.append(_cube_q(m6, x, np.zeros(3), 4.0))
            want_c = 2
        jl = jump_links(m6, x, DELTA, allreg)
        mj = minor_jumps(m6, DELTA, allreg)
        D = float(np.linalg.norm(A.from6(m6[::-1, ::-1, :]) - A.from6(m6), axis=(-1, -2)).max())
        lr = lines_read(m6, x, np.zeros(3), [0, 0, 1.0], (12.0, 14.0, 16.0), DELTA)
        good = bool(fd <= 1e-15 and box["liftable"] and box["abs_q"] == 2 and all(c["liftable"] and c["abs_q"] == want_c for c in cubes)
                    and jl["jumps"] == 0 and mj["jumps"] == 0 and D < 1e-14)
        ok &= good
        rows.append(dict(d0=d0, faces_vs_d14=fd, faces_without_ladder=fl, box_abs_q=box["abs_q"], cubes_abs_q=[c["abs_q"] for c in cubes],
                         largest_vacuum_deg=jl["largest_deg"], jumps=jl["jumps"], minor_largest_deg=mj["largest_deg"], minor_jumps=mj["jumps"], halfturn_D=D,
                         lines=[(q["R"], q["count"], [round(c["polar_deg"], 1) for c in q["comps"]]) for q in lr], passed=good))
    return dict(control="C14 the ladder of the pair", passed=bool(ok), rows=rows,
                failing_input=dict(input="the starts without the ladder", faces_off=[r["faces_without_ladder"] for r in rows[1:]],
                                   fails_as_planned=bool(all(r["faces_without_ladder"] > 1e-3 for r in rows[1:]))))


def control_c15(x, b4b):
    vac = A.to6(np.diag([1.0, DELTA, 0.0]) * np.ones((len(x),) * 3 + (1, 1)))
    mod = make_model("H1", DELTA, hold6=vac)
    tv = minor_torque(mod, vac)
    good = bool(tv["gate"] <= 1e-3 and tv["torque"] <= 1e-5)
    fails = []
    s0 = seed_bi(x)
    mod0 = make_model("H1", DELTA, hold6=s0)
    for nm, f6 in (("M0delta", s0), ("W2pi of M0delta", W_turn(s0, s0, x, 2 * np.pi, np.zeros(3), 4.354, 11.0)),
                   ("W2pi of the B4b start", W_turn(b4b, b4b, x, 2 * np.pi, np.zeros(3), 4.354, 11.0))):
        t = minor_torque(mod0, f6)
        fails.append(dict(input=nm, gate=t["gate"], torque=t["torque"], fails_as_planned=bool(t["gate"] > 1e-3 or t["torque"] > 1e-5)))
    return dict(control="C15 the convergence rule at delta > 0 (fixed fields)", passed=good, vacuum=tv, failing_inputs=fails)


def control_c16():
    x = axis_of("P")
    ref = pair_bi(x, 14.0)
    c1 = np.array([7.0, 0, 0])
    R1c, R2c = 4.35, 9.65
    eo, col, reached, ncol, least = e1_fill(ref, x, c1, R1c, R2c)
    wt = W_turn(ref, ref, x, 2 * np.pi, c1, R1c, R2c, fill=True)
    outside = float(np.abs(wt - ref)[~col].max())
    X = mesh(x)
    excl = np.linalg.norm(X + c1, axis=-1) <= 2.85 + 2.0
    mp = minor_profile(wt, ref, x, c1, R1c, outer=R2c + 0.5, exclude=excl, e1_ref=e1_pair_profile(ref, x, c1, R1c, R2c))
    fs = fprof(mp["sign_shell"], R1c, R2c)
    good = bool(reached == ncol and least >= np.cos(np.radians(45)) and outside <= 1e-15 and mp["in_regime"] and abs(mp["alpha_s"] - 2 * np.pi * fs) <= 0.1)
    wr = W_regime(ref, x, c1, R1c, R2c)
    mod = make_model("P", DELTA, hold6=ref)
    U0 = mod.evaluate(ref, False)["U"]
    return dict(control="C16 W on a pair collar", passed=good, reached=reached, collar_nodes=ncol, least_link_dot=least, outside_change=outside,
                alpha_s=mp["alpha_s"], expected=2 * np.pi * fs, sign_shell=mp["sign_shell"], min_coherence=mp["min_coherence"], min_align=mp["min_align"],
                regime=mp["regime"], dU_W2pi=mod.evaluate(wt, False)["U"] - U0,
                failing_input=dict(input="the regime of section 11 on the same collar", **wr, fails_as_planned=bool(not wr["regime"])))


def control_c17():
    rows = []
    ok = True
    for h in (0.5, 0.375):
        N = int(round(12 / h)) + 1
        xx = np.linspace(-6, 6, N)
        for on, (nu, e1v) in RD.ORIENT.items():
            r6 = RD.ring_field(xx, 2.75, nu, e1v)
            nuu = unit(nu)
            XX = mesh(xx)
            zt = XX @ nuu
            rp = XX - zt[..., None] * nuu
            ph = np.cross(nuu, rp) / np.maximum(np.linalg.norm(rp, axis=-1), 1e-300)[..., None]
            m6 = A.to6(A.from6(r6) + DELTA * np.einsum("...i,...j->...ij", ph, ph))
            L = RD.line_reader(m6, xx, 6.0)
            orr = RD.one_ring(m6, xx, L)
            ang = RD.angle(L["nu"], nuu)
            good = bool(abs(L["RL"] - 2.75) <= 0.01 and ang <= 0.5 and L["regime"] and orr["one_ring"])
            ok &= good
            rows.append(dict(h=h, orientation=on, RL=L["RL"], normal_err_deg=ang, regime=L["regime"], one_ring=orr["one_ring"], passed=good))
    x = axis_of("P")
    m6p = constructed_pair(x, [(7.0, 0, 0), (-7.0, 0, 0)], delta=DELTA)
    w, _ = eigs(m6p)
    g12 = w[..., 0] - w[..., 1]
    Z = (g12 < 0.5) & (np.linalg.norm(mesh(x), axis=-1) < 0.93 * 18)
    lab, k = ndimage.label(Z, structure=np.ones((3, 3, 3)))
    sizes = [int((lab == j).sum()) for j in range(1, k + 1)]
    pair_ok = bool(k == 2)
    return dict(control="C17 the ring readers at delta > 0", passed=bool(ok and pair_ok), single_ring_passed=bool(ok), pair_passed=pair_ok, rows=rows,
                pair=dict(components=k, sizes=sizes))


def control_c3a_d5():
    x = axis_of("H1")
    m6 = seed_bi(x)
    ring = dict(c=np.zeros(3), nu=np.array([0, 0, 1.0]), RL=4.354 - 1.5)
    R1, R2 = 4.354, 11.0
    h = 0.5
    cases = []
    ok = True
    for Th, a, an in ((2 * np.pi, [0, 0, 1.0], "z"), (2 * np.pi, [1.0, 0, 0], "x"), (1.3, [0, 0, 1.0], "z")):
        f6 = T_turn(m6, x, Th, a, ring["c"], R1, R2)
        tr = turn_reader(f6, m6, x, ring["c"], R1, axis=a)
        inner = [r for r in tr["rows"] if r["rho"] - h >= R1 - 1e-9 and r["rho"] + h <= R2 + 1e-9]
        e_in = max(abs(r["angle"] - Th * fprof(r["rho"], R1, R2)) for r in inner)
        r_in = max(r["r"] for r in inner)
        e_all = max(abs(r["angle"] - Th * fprof(r["rho"], R1, R2)) for r in tr["rows"])
        want = -1 if abs(Th - 2 * np.pi) < 1e-12 else 1
        good = bool(r_in <= 1e-6 and e_in <= 1e-5 and e_all <= 0.15 and tr["r_max"] <= 0.15 and tr["sigma"] == want and tr["in_regime"])
        ok &= good
        cases.append(dict(Theta=Th, axis=an, r_inner_max=r_in, angle_err_inner_max=e_in, angle_err_all_max=e_all, r_max=tr["r_max"], sigma=tr["sigma"],
                          q0_sign_shell=tr["q0_sign_shell"], in_regime=tr["in_regime"], passed=good))
    w2 = W_turn(m6, m6, x, 2 * np.pi, ring["c"], R1, R2)
    tr = turn_reader(w2, m6, x, ring["c"], R1)
    good = bool(tr["sigma"] == 1 and tr["chi_max"] <= 0.05 and tr["in_regime"])
    ok &= good
    cases.append(dict(Theta="W2pi", sigma=tr["sigma"], q0_sign_shell=tr["q0_sign_shell"], chi_max=tr["chi_max"], r_max=tr["r_max"], in_regime=tr["in_regime"], passed=good))
    f6 = T_turn(m6, x, 2 * np.pi, [0, 0, 1.0], ring["c"], R1, R2)
    trn = turn_reader(f6, m6, x, ring["c"], R1, axis=[0, 0, 1.0], no_continuation=True)
    return dict(control="C3 (a) at delta = 0.05", passed=bool(ok), cases=cases,
                failing_inputs=[dict(input="a reader without continuation on T[2pi]", sigma=trn["sigma"], fails_as_planned=bool(trn["sigma"] == 1))])


def write_control(name, res, tryn):
    CK.mkdir(parents=True, exist_ok=True)
    f = CK / f"control_{name}_try{tryn}.json"
    f.write_text(json.dumps(res, indent=1, default=_jsonable) + "\n")
    print(f"{res['control']}: {'MET' if res['passed'] else 'NOT MET'}  -> {f.name}", flush=True)
    return res["passed"]


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def selftest_d0(tryn, only=None):
    out = {}
    jobs = [("C1", control_c1), ("C2", control_c2), ("C3a_H1", None), ("C3b", control_c3b), ("C8", control_c8), ("C6", control_c6)]
    for nm, fn in jobs:
        if only and nm not in only:
            continue
        t0 = time.time()
        if nm == "C3a_H1":
            m6, x, ring, e = H1_reference()
            xh2 = axis_of("H2")
            m2 = interp_to(m6, x, xh2)
            mod2 = make_model("H2")
            m2 = mod2.hold_faces(m2)
            res = c3a_check(m6, x, ring, 11.0, "on G6_mid_a (H1)", extra_fail=c3a_h2_fail(m2, xh2, ring))
            res["note"] = "failing input (ii) read on the B2 start (G6_mid_a interpolated to H2, faces reset), before B2 has ended"
        else:
            res = fn()
        res["seconds"] = time.time() - t0
        out[nm] = write_control(nm, res, tryn)
    return out


def selftest_d5(tryn, only=None):
    out = {}
    x = axis_of("H1")
    g6, _, ring, e = H1_reference()
    s0 = seed_bi(x)
    face = faces_mask(len(x))
    b4b = np.where(face[..., None], s0, g6)
    jobs = [("C3a_d5", control_c3a_d5), ("C8_d5", lambda: control_c8(True)), ("C9", control_c9), ("C10_C11_C12", None), ("C13", lambda: control_c13(x)),
            ("C14", control_c14), ("C15", lambda: control_c15(x, b4b)), ("C16", control_c16), ("C17", control_c17)]
    for nm, fn in jobs:
        if only and nm not in only:
            continue
        t0 = time.time()
        if nm == "C10_C11_C12":
            r10, r11, r12 = control_c10_c11_c12(x, {"M0delta": (s0, None), "B4b_start": (b4b, ring)})
            for k, r in (("C10", r10), ("C11", r11), ("C12", r12)):
                r["seconds"] = time.time() - t0
                out[k] = write_control(k, r, tryn)
            continue
        res = fn()
        res["seconds"] = time.time() - t0
        out[nm] = write_control(nm, res, tryn)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", choices=["d0", "d5"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--try", dest="tryn", type=int, default=1)
    ap.add_argument("--c3a", choices=["H2", "H3"])
    ap.add_argument("--ref")
    a = ap.parse_args()
    if a.selftest == "d0":
        r = selftest_d0(a.tryn, a.only)
    elif a.selftest == "d5":
        r = selftest_d5(a.tryn, a.only)
    elif a.c3a:
        rec = json.loads(Path(a.ref).read_text())
        z = np.load(Path(a.ref).parent / rec["name"] / rec["end_file"], allow_pickle=False)
        m6 = np.asarray(z["m6"], float)
        x = np.asarray(z["x"], float)
        ring = ring_of(rec["end"])
        R2 = 11.0 if a.c3a == "H2" else 14.0
        extra = c3a_h2_fail(m6, x, ring) if a.c3a == "H2" else None
        res = c3a_check(m6, x, ring, R2, f"on the reference of {a.c3a} ({rec['name']})", extra_fail=extra)
        r = {a.c3a: write_control(f"C3a_{a.c3a}", res, a.tryn)}
    else:
        ap.print_help()
        return
    sys.exit(0 if all(r.values()) else 1)


if __name__ == "__main__":
    main()
