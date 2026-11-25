"""M5.46: the charge of the long axis on nested cube surfaces, and the half-turn line against the faces.

The field is a real symmetric 3 x 3 matrix M(x) on an N^3 grid of spacing h in the box [-L, L]^3, stored as
six components (xx, xy, xz, yy, yz, zz). With eigenvalues l1 <= l2 <= l3, n is the unit eigenvector of l3
(the long axis, defined up to sign) and g = l3 - l2 is the gap.

    THE CHARGE   on a closed cube surface S of grid nodes, every plaquette cut into two triangles (a, b, c)
                 by one diagonal, corners ordered counterclockwise seen from outside:

        pierced(t)   sign(n_a.n_b) sign(n_b.n_c) sign(n_c.n_a) = -1        (independent of the signs of n)
        lift         a breadth-first spanning tree over the nodes of S (edges: the triangle sides), root sign
                     +1, each node given the sign that makes n.n > 0 along its tree edge; each surface alone
        liftable     no pierced triangle on S, and g > 0.2 at every node of S
        Omega_t  =   2 atan2( n_a . (n_b x n_c) ,  1 + n_a.n_b + n_b.n_c + n_c.n_a )     (lifted vectors)
        q(S)     =   (1 / 4 pi) sum_t Omega_t           the read is |q|, a whole number; which one is the read

                 A surface with a pierced triangle gets no charge; its count of pierced triangles is reported.
                 The nested surfaces are the cubes centered on the box with half-widths 2, 2.5, 3, ... and
                 L - h (one cell inside the faces); the last one is called the box surface.

    THE LINE     the functions of m5_44_loops.py (pierced_plaquettes, group, loop_stats; its second method is
                 not called): a plaquette (a, b, c, d) is pierced when the four sign products around it
                 multiply to -1; pierced plaquettes whose centers are at most 1.05 h apart form a set; a set
                 under 8 plaquettes is a fragment. Added here, per set of 8 or more with centers p_k:

        distance     d = min_k min_axis ( L - |p_k,axis| )                 the least distance to the faces
        touches      d <= h (a plaquette within one cell of a face); the faces touched are listed; a set
                     that touches no face is clear of the faces
        max norm     max_k max_axis |p_k,axis|; a centered cube of half-width W surrounds the set when
                     max norm < W

    THE LOOP     of a read: the largest set clear of the faces, or the largest set when none is clear.

The controls K1 to K7 of m5_46_task_details.md (section 3) run with --selftest; each prints PASS or FAIL with
its numbers and an input on which the same check returns another value. The exit code is nonzero on any FAIL.

Usage:  python m5_46_degree.py --selftest [--base-state FILE.npz ...] [--out FILE.json]
"""

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import breadth_first_order  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_44_loops as L  # noqa: E402

IDX = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])
GAP_MIN = 0.2
FRAGMENT = 8
WHOLE_TOL = 1e-6


def director(m6):
    """The long axis n (unit eigenvector of the largest eigenvalue) and the gap l3 - l2 at every node."""
    w, v = np.linalg.eigh(m6[..., IDX])
    return np.ascontiguousarray(v[..., :, 2]), w[..., 2] - w[..., 1]


def uniaxial(n):
    """M = n n^T as six components (eigenvalues 1, 0, 0: the gap is 1)."""
    M = np.einsum("...i,...j->...ij", n, n)
    return np.stack([M[..., a, b] for a, b in L.PAIRS], -1)


@njit(cache=True)
def _lift(order, pred, v):
    s = np.ones(v.shape[0])
    for t in range(1, order.shape[0]):
        k = order[t]
        p = pred[k]
        d = v[k, 0] * v[p, 0] + v[k, 1] * v[p, 1] + v[k, 2] * v[p, 2]
        s[k] = s[p] if d >= 0.0 else -s[p]
    return s


def _sgn(a):
    return np.where(a >= 0.0, 1.0, -1.0)


class Surface:
    """A closed cube surface of grid nodes, index bounds lo[a] .. hi[a] (inclusive) per axis, triangulated."""

    def __init__(self, N, lo, hi, half_width=None):
        lo, hi = np.asarray(lo, int), np.asarray(hi, int)
        assert (lo >= 0).all() and (hi <= N - 1).all() and (hi > lo).all()
        self.N, self.lo, self.hi, self.W = N, lo, hi, half_width
        strides = np.array([N * N, N, 1])
        tris = []
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):          # e_b x e_c = e_a
            jb, jc = np.meshgrid(np.arange(lo[b], hi[b]), np.arange(lo[c], hi[c]), indexing="ij")
            jb, jc = jb.ravel(), jc.ravel()
            for pos, outward in ((hi[a], True), (lo[a], False)):
                base = pos * strides[a]
                A = base + jb * strides[b] + jc * strides[c]
                B = A + strides[b]
                C = A + strides[b] + strides[c]
                D = A + strides[c]
                if outward:
                    t = np.stack([np.stack([A, B, C], -1), np.stack([A, C, D], -1)], 1)
                else:
                    t = np.stack([np.stack([A, C, B], -1), np.stack([A, D, C], -1)], 1)
                tris.append(t.reshape(-1, 3))                        # rows 2p and 2p + 1: the plaquette p
        flat = np.concatenate(tris)
        self.nodes = np.unique(flat)
        self.tri = np.searchsorted(self.nodes, flat)
        K = len(self.nodes)
        e = np.concatenate([self.tri[:, [0, 1]], self.tri[:, [1, 2]], self.tri[:, [2, 0]]])
        g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(K, K)).tocsr()
        order, pred = breadth_first_order(g, 0, directed=False, return_predecessors=True)
        assert len(order) == K
        self.order, self.pred = order.astype(np.int64), pred.astype(np.int64)

    def vectors(self, n):
        return np.ascontiguousarray(n.reshape(-1, 3)[self.nodes])

    def degree_of(self, w):
        """(1 / 4 pi) sum of the triangle solid angles of the vectors w (K x 3), taken as given."""
        a, b, c = w[self.tri[:, 0]], w[self.tri[:, 1]], w[self.tri[:, 2]]
        num = np.einsum("ti,ti->t", a, np.cross(b, c))
        den = 1.0 + (a * b).sum(-1) + (b * c).sum(-1) + (c * a).sum(-1)
        return float((2.0 * np.arctan2(num, den)).sum() / (4.0 * np.pi))

    def read(self, n, gap):
        v = self.vectors(n)
        a, b, c = v[self.tri[:, 0]], v[self.tri[:, 1]], v[self.tri[:, 2]]
        pt = (_sgn((a * b).sum(-1)) * _sgn((b * c).sum(-1)) * _sgn((c * a).sum(-1))) < 0
        pair = pt.reshape(-1, 2)
        min_gap = float(gap.reshape(-1)[self.nodes].min())
        out = dict(half_width=self.W, pierced_triangles=int(pt.sum()), plaquettes_one_triangle=int((pair.sum(1) == 1).sum()),
                   plaquettes_both_triangles=int((pair.sum(1) == 2).sum()), min_gap=min_gap, q=None, abs_q=None,
                   liftable=False, frustrated=None, reason=None)
        if out["pierced_triangles"]:
            out["reason"] = "pierced"
            return out
        w = v * _lift(self.order, self.pred, v)[:, None]
        a, b, c = w[self.tri[:, 0]], w[self.tri[:, 1]], w[self.tri[:, 2]]
        out["frustrated"] = int((((a * b).sum(-1) < 0) | ((b * c).sum(-1) < 0) | ((c * a).sum(-1) < 0)).sum())
        q = self.degree_of(w)
        out["q"] = q
        out["abs_q"] = int(round(abs(q)))
        out["q_off_whole"] = abs(abs(q) - round(abs(q)))
        if min_gap <= GAP_MIN:
            out["reason"] = "gap"
        elif out["frustrated"] or out["q_off_whole"] > WHOLE_TOL:
            out["reason"] = "lift"
        else:
            out["liftable"] = True
        return out


def half_widths(half_box, h):
    """2, 2.5, 3, ... and half_box - h (one cell inside the faces, the box surface)."""
    ws = [float(w) for w in np.arange(2.0, half_box - h + 1e-9, 0.5)]
    if abs(ws[-1] - (half_box - h)) > 1e-9:
        ws.append(float(half_box - h))
    return ws


def nested(x):
    """The centered cube surfaces of a grid with axis x."""
    N, h = len(x), x[1] - x[0]
    c = (N - 1) // 2
    out = []
    for w in half_widths(x[-1], h):
        m = int(round(w / h))
        out.append(Surface(N, [c - m] * 3, [c + m] * 3, half_width=w))
    return out


def cube_at(x, center, half_width):
    """A cube surface centered on the node nearest to `center`."""
    N, h = len(x), x[1] - x[0]
    c = np.rint((np.asarray(center, float) - x[0]) / h).astype(int)
    m = int(round(half_width / h))
    return Surface(N, c - m, c + m, half_width=half_width)


def line_read(m6, x):
    """The sets of pierced plaquettes with their place against the faces. Returns (dict, centers, labels)."""
    h, hb = x[1] - x[0], x[-1]
    P, g = L.pierced_plaquettes(m6, x)
    lab = L.group(P, h)
    idx = sorted((np.flatnonzero(lab == i) for i in range(lab.max() + 1 if len(lab) else 0)), key=lambda s: -len(s))
    sets = []
    for s in idx:
        if len(s) < FRAGMENT:
            continue
        Q = P[s]
        st = L.loop_stats(Q)
        d = hb - np.abs(Q)
        near = d <= h + 1e-9
        faces = sorted({2 * a + (1 if sg > 0 else 0) for a in range(3) for sg in (-1, 1) if (near[:, a] & (sg * Q[:, a] > 0)).any()})
        st.update(least_distance_to_faces=float(d.min()), touches=bool(near.any()), faces=faces, max_norm=float(np.abs(Q).max()),
                  gap_mean=float(g[s].mean()), gap_max=float(g[s].max()))
        sets.append(st)
    small = [s for s in idx if len(s) < FRAGMENT]
    return dict(pierced=int(len(P)), sets=sets, fragments=len(small), fragment_plaquettes=int(sum(len(s) for s in small))), P, lab


def crossings(P, w):
    """The pierced plaquettes of the line reader that lie on the centered cube of half-width w."""
    if len(P) == 0:
        return 0
    a = np.abs(P)
    on = np.zeros(len(P), bool)
    for ax in range(3):
        o = [k for k in range(3) if k != ax]
        on |= (np.abs(a[:, ax] - w) < 1e-9) & (a[:, o[0]] < w) & (a[:, o[1]] < w)
    return int(on.sum())


def the_loop(sets):
    """Index of the loop of a read: the largest set clear of the faces, else the largest set, else None."""
    clear = [i for i, s in enumerate(sets) if not s["touches"]]
    return clear[0] if clear else (0 if sets else None)


def read_state(m6, x, surfaces, n_gap=None):
    """One read: the line, the loop, and the charge on every nested surface with the sets it surrounds."""
    n, gap = director(m6) if n_gap is None else n_gap
    line, P, _ = line_read(m6, x)
    k = the_loop(line["sets"])
    surf = []
    for S in surfaces:
        r = S.read(n, gap)
        r["surrounds"] = [i for i, s in enumerate(line["sets"]) if s["max_norm"] < S.W - 1e-9]
        r["around_the_loop"] = bool(k is not None and k in r["surrounds"])
        r["line_crossings"] = crossings(P, S.W)
        surf.append(r)
    return dict(line=line, loop=k, clear=[i for i, s in enumerate(line["sets"]) if not s["touches"]], surfaces=surf)


# ----------------------------------------------------------------------------- analytic inputs

def grid(half_box, h):
    N = int(round(2 * half_box / h)) + 1
    x = np.linspace(-half_box, half_box, N)
    return x, np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)


def degree_map(X, k):
    """n(x) = f_k(x / |x|), f_k the map w -> w^k of the sphere (k = 1: the hedgehog; k = 2: degree 2)."""
    r = np.linalg.norm(X, axis=-1) + 1e-300
    ct = X[..., 2] / r
    phi = np.arctan2(X[..., 1], X[..., 0])
    s2, c2 = (1 - ct) / 2, (1 + ct) / 2                     # sin^2 and cos^2 of theta / 2
    th = 2 * np.arctan2(s2 ** (k / 2.0), c2 ** (k / 2.0))    # tan(theta' / 2) = tan(theta / 2)^k
    return np.stack([np.sin(th) * np.cos(k * phi), np.sin(th) * np.sin(k * phi), np.cos(th)], -1)


def pair_field(X, p_plus, p_minus, same=False):
    """The unit field of a point source at p_plus and a sink (or, with same, a second source) at p_minus."""
    a, b = X - np.asarray(p_plus), X - np.asarray(p_minus)
    v = a / np.linalg.norm(a, axis=-1, keepdims=True) ** 3 + (1 if same else -1) * b / np.linalg.norm(b, axis=-1, keepdims=True) ** 3
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ring_field(X, rings, R0, axis_xy=(0.11, 0.17), delta=0.05):
    """M = n n^T + delta m m^T with n = cos(psi/2) rho_hat + sin(psi/2) z_hat and, for k rings at heights z_i,

        psi = arg[ (-1)^(k+1) prod_i (zeta - zeta_i)(zeta + conj(zeta_i)) ],   zeta = rho + i z,  zeta_i = R0 + i z_i

    psi winds once around each circle (a half turn of n) and equals pi on the axis, where n = z_hat: no other
    line. The axis is set off the nodes. One ring is the hedgehog opened into a ring (n -> r_hat far away)."""
    Y = X - np.array([axis_xy[0], axis_xy[1], 0.0])
    rho = np.hypot(Y[..., 0], Y[..., 1])
    rho_hat = np.stack([Y[..., 0] / rho, Y[..., 1] / rho, np.zeros_like(rho)], -1)
    zh = np.array([0.0, 0.0, 1.0])
    zeta = rho + 1j * Y[..., 2]
    f = (-1.0) ** (len(rings) + 1) * np.ones_like(zeta)
    for z0 in rings:
        zi = R0 + 1j * z0
        f = f * (zeta - zi) * (zeta + np.conj(zi))
    psi = np.angle(f)
    n = np.cos(psi / 2)[..., None] * rho_hat + np.sin(psi / 2)[..., None] * zh
    m = np.cross(zh, rho_hat)
    M = np.einsum("...i,...j->...ij", n, n) + delta * np.einsum("...i,...j->...ij", m, m)
    return np.stack([M[..., a, b] for a, b in L.PAIRS], -1)


# ----------------------------------------------------------------------------- the controls

def _absq(surf):
    return [(s["abs_q"] if s["liftable"] else None) for s in surf]


def selftest(base_states, out_file):
    import m5_42_eq42 as Q
    res = []

    def report(name, ok, numbers, counter):
        res.append(dict(control=name, passed=bool(ok), numbers=numbers, counter_input=counter))
        print(f"{name}: {'PASS' if ok else 'FAIL'}  {numbers}")
        print(f"      counter-input: {counter}")

    def charges(m6, x):
        n, gap = director(m6)
        return [S.read(n, gap) for S in nested(x)]

    uni = {}
    for hb, N in ((8.0, 33), (8.0, 65), (10.0, 41)):
        x = np.linspace(-hb, hb, N)
        u = np.zeros((N, N, N, 6))
        u[..., 0], u[..., 3] = 1.0, 0.05
        uni[(hb, N)] = (u, x)

    # K1: the seed (a point hedgehog of the long axis, no loop)
    ok, nums = True, []
    for hb, N in ((8.0, 33), (8.0, 65), (10.0, 41)):
        mod = Q.Model(N, hb, 0.05, 0.03)
        m6 = mod.seed6()
        s = charges(m6, mod.x)
        line = line_read(m6, mod.x)[0]
        good = all(r["liftable"] and r["abs_q"] == 1 and r["pierced_triangles"] == 0 for r in s) and len(line["sets"]) == 0
        ok &= good
        nums.append(dict(half_box=hb, N=N, surfaces=len(s), abs_q=_absq(s), pierced_triangles=sum(r["pierced_triangles"] for r in s),
                         q_off_whole_max=max(r["q_off_whole"] for r in s), min_gap=min(r["min_gap"] for r in s), loops=len(line["sets"]),
                         fragments=line["fragments"]))
    cu = _absq(charges(*uni[(8.0, 33)]))
    report("K1 seed reads 1 on every cube, no pierced triangle, no loop", ok, nums, f"a uniform M returns {sorted(set(cu))} on the same cubes, so the check fails on it")

    # K2: a uniform M
    ok, nums = True, []
    for key, (u, x) in uni.items():
        s = charges(u, x)
        ok &= all(r["liftable"] and r["abs_q"] == 0 for r in s)
        nums.append(dict(half_box=key[0], N=key[1], abs_q=_absq(s), q_abs_max=max(abs(r["q"]) for r in s)))
    mod = Q.Model(33, 8.0, 0.05, 0.03)
    report("K2 uniform M reads 0 on every cube", ok, nums, f"the seed returns {sorted(set(_absq(charges(mod.seed6(), mod.x))))}, so the check fails on it")

    # K3: the map w -> w^2
    ok, nums = True, []
    for hb, h in ((8.0, 0.5), (8.0, 0.25)):
        x, X = grid(hb, h)
        s = charges(uniaxial(degree_map(X, 2)), x)
        ok &= all(r["liftable"] and r["abs_q"] == 2 for r in s)
        nums.append(dict(half_box=hb, spacing=h, abs_q=_absq(s), pierced_triangles=sum(r["pierced_triangles"] for r in s),
                         q_off_whole_max=max((r.get("q_off_whole") or 0.0) for r in s)))
    x, X = grid(8.0, 0.5)
    report("K3 degree-2 director reads 2 on every cube", ok, nums,
           f"the map w -> w (degree 1) returns {sorted(set(_absq(charges(uniaxial(degree_map(X, 1)), x))))}, so the check fails on it")

    # K4: a hedgehog and an anti-hedgehog on the axis
    ok, nums = True, []
    pp, pm = (0.1, 0.07, 3.2), (0.1, 0.07, -3.2)
    for hb, h in ((8.0, 0.5), (8.0, 0.25)):
        x, X = grid(hb, h)
        m6 = uniaxial(pair_field(X, pp, pm))
        n, gap = director(m6)
        s = [S.read(n, gap) for S in nested(x)]
        outer = [r for r in s if r["half_width"] >= 4.5]
        up, dn = cube_at(x, (0, 0, 3.0), 2.0).read(n, gap), cube_at(x, (0, 0, -3.0), 2.0).read(n, gap)
        good = all(r["liftable"] and r["abs_q"] == 0 for r in outer) and up["liftable"] and dn["liftable"] and up["abs_q"] == 1 and dn["abs_q"] == 1
        ok &= good
        nums.append(dict(spacing=h, around_both_half_width_4p5_up=_absq(outer), around_the_upper=up["abs_q"], around_the_lower=dn["abs_q"],
                         q_upper=up["q"], q_lower=dn["q"], all_centered=[(r["half_width"], r["abs_q"] if r["liftable"] else r["reason"]) for r in s]))
    x, X = grid(8.0, 0.5)
    n, gap = director(uniaxial(pair_field(X, pp, pm, same=True)))
    cs = sorted({r["abs_q"] for r in (S.read(n, gap) for S in nested(x)) if r["liftable"] and r["half_width"] >= 4.5})
    report("K4 pair of opposite points: 0 around both, 1 around each", ok, nums, f"two points of the same sign return {cs} around both (a zero of index -1 sits between them), so the check fails on it")

    # K5: cubes cut by the base loop
    ok, nums, counter = bool(base_states), [], "no base state given"
    for f in base_states:
        z = np.load(f, allow_pickle=False)
        m6, x = z["m6"], z["x"]
        r = read_state(m6, x, nested(x))
        cut = [s for s in r["surfaces"] if s["line_crossings"] > 0]
        good = len(cut) > 0 and all((not s["liftable"]) and s["abs_q"] is None and s["pierced_triangles"] == s["line_crossings"] for s in cut)
        not_cut_pierced = [s["half_width"] for s in r["surfaces"] if s["line_crossings"] == 0 and s["pierced_triangles"] > 0]
        good &= not not_cut_pierced
        ok &= good
        nums.append(dict(state=Path(f).name, loops=len(r["line"]["sets"]), radius=(r["line"]["sets"][0]["radius"] if r["line"]["sets"] else None),
                         cut=[dict(half_width=s["half_width"], pierced_triangles=s["pierced_triangles"], line_crossings=s["line_crossings"],
                                   plaquettes_both_triangles=s["plaquettes_both_triangles"], charge=s["abs_q"]) for s in cut],
                         pierced_without_a_crossing=not_cut_pierced))
        around = [s for s in r["surfaces"] if s["around_the_loop"] and s["liftable"]]
        counter = (f"on the {len(around)} liftable cubes around the loop of {Path(f).name} the charge is {sorted({s['abs_q'] for s in around})} and the count is "
                   f"{sorted({s['pierced_triangles'] for s in around})}: a charge is assigned there, so the check fails on them")
    report("K5 cubes cut by the base loop are not liftable; piercings equal the line reader's crossings", ok, nums, counter)

    # K6: one half-turn ring, then two rings at z = +-3
    ok, nums = True, []
    for hb, h in ((8.0, 0.5), (8.0, 0.25)):
        x, X = grid(hb, h)
        for R0 in (3.0, 5.0):
            m6 = ring_field(X, (0.0,), R0)
            r = read_state(m6, x, nested(x))
            one = r["line"]
            g1 = len(one["sets"]) == 1 and abs(one["sets"][0]["radius"] - R0) <= h / 2
            ok &= g1
            nums.append(dict(spacing=h, rings=1, R0=R0, loops=len(one["sets"]), radius=[s["radius"] for s in one["sets"]], fragments=one["fragments"],
                             charge_by_cube=[(s["half_width"], s["abs_q"] if s["liftable"] else s["reason"]) for s in r["surfaces"]]))
        two = line_read(ring_field(X, (-3.0, 3.0), 3.0), x)[0]
        g2 = len(two["sets"]) == 2 and all(abs(s["radius"] - 3.0) <= h / 2 for s in two["sets"]) and sorted(round(s["center"][2]) for s in two["sets"]) == [-3, 3]
        ok &= g2
        nums.append(dict(spacing=h, rings=2, R0=3.0, loops=len(two["sets"]), radius=[s["radius"] for s in two["sets"]],
                         center_z=[s["center"][2] for s in two["sets"]], fragments=two["fragments"]))
    report("K6 one ring: one loop at its radius; two rings: two loops", ok, nums,
           "the ring of radius 5 reads 5 and the two rings read two loops, so a check for one loop of radius 3 fails on both")

    # K7: the sign of n flipped at random nodes before the lift
    ok, nums, counter = bool(base_states), [], "no base state given"
    rng = np.random.default_rng(20261119)
    for f in base_states:
        z = np.load(f, allow_pickle=False)
        m6, x = z["m6"], z["x"]
        n, gap = director(m6)
        S = nested(x)
        a = [s.read(n, gap) for s in S]
        worst, moved, flipped = 0.0, 0, 0
        for _ in range(5):
            sg = np.where(rng.random(n.shape[:3]) < 0.5, -1.0, 1.0)
            flipped += int((sg < 0).sum())
            b = [s.read(n * sg[..., None], gap) for s in S]
            for p, q in zip(a, b):
                same = p["liftable"] == q["liftable"] and p["pierced_triangles"] == q["pierced_triangles"] and p["abs_q"] == q["abs_q"]
                moved += 0 if same else 1
                if p["q"] is not None and q["q"] is not None:
                    worst = max(worst, abs(abs(p["q"]) - abs(q["q"])))
        ok &= moved == 0 and worst <= 1e-9
        lift = [s for s, r in zip(S, a) if r["liftable"]]
        raw = [s.degree_of(s.vectors(n * sg[..., None])) for s in lift]
        nums.append(dict(state=Path(f).name, surfaces=len(S), liftable=len(lift), trials=5, nodes_flipped=flipped, reads_moved=moved, largest_change_of_abs_q=worst))
        counter = (f"without the lift the same flipped field of {Path(f).name} sums to {[round(v, 3) for v in raw[:4]]} on its first liftable cubes "
                   f"(lifted: {[r['abs_q'] for r in a if r['liftable']][:4]}), so the read moves when the lift is skipped")
    report("K7 random sign flips of n before the lift leave every read unchanged", ok, nums, counter)

    passed = all(r["passed"] for r in res)
    if out_file:
        Path(out_file).write_text(json.dumps(dict(passed=passed, base_states=[Path(f).name for f in base_states], controls=res), indent=1) + "\n")
    print("CONTROLS:", "ALL PASS" if passed else "FAIL: " + ", ".join(r["control"].split()[0] for r in res if not r["passed"]))
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--base-state", nargs="*", default=None, help="relaxed states (npz with m6, x) for K5 and K7")
    ap.add_argument("--out", default=None, help="where the JSON of the controls goes")
    a = ap.parse_args()
    if not a.selftest:
        ap.print_help()
        return
    data = HERE.parent / "data"
    states = a.base_state
    if states is None:
        states = [str(data / f"m5_46_{c}_end.npz") for c in ("B0a", "B0b", "B0c", "B0d") if (data / f"m5_46_{c}_end.npz").exists()]
        if not states and (data / "m5_45_h0500_L0.03_end.npz").exists():
            states = [str(data / "m5_45_h0500_L0.03_end.npz")]
    sys.exit(0 if selftest(states, a.out) else 1)


if __name__ == "__main__":
    main()
