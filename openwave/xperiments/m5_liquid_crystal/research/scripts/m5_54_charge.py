"""M5.54: the author's charge integral Q/4pi ported (plan, section 1), with its controls C1 to C4 and the regime
row (plan, section 2).

With l1 >= l2 >= l3 the eigenvalues of M, g12 = l1 - l2 and n the top eigenvector (defined up to sign), x the node
position about the center node of the grid:

    nt  = s n,   s = sgn(n . x + 1e-12)                     the oriented lift (branch sheet where n . x = 0)
    b_1 = nt . (d_2 nt x d_3 nt), and cyclically           d_a: second-order central difference inside, first-order
                                                            one-sided on a face (numpy.gradient, edge_order 1)
    rho = sum_a d_a b_a,    Q = h^3 sum_inner rho          inner = the nodes off the faces

The volume sum telescopes exactly into a flux through the two outermost node layers:

    Q = h^2 sum_a sum_perp 1/2 [ b_a(N-1) + b_a(N-2) - b_a(1) - b_a(0) ]

and, for a half-width l = m h, Q(l) = h^3 sum_{|x|_inf <= l - h} rho is the same two-layer flux at half-widths
l - h and l. The second surface is l2 = L - 2h.

Regime of a Q read: on both two-layer shells (half-widths L - h, L and L - 3h, L - 2h) no lift flip (nt_i . nt_j < 0)
between two neighboring nodes of the shell, g12 > 0.2 on every node of the shell, and l2 - h >= R_L + 1.5.

Shares (descriptors): with d the distance to a circle (center c, normal nu, radius R), over the inner nodes,
S_delta = sum_{d < delta} |rho| / sum |rho| (delta = 0.5, 1, 1.5), the signed share sum_{d < 1} rho / sum rho, and the
branch share, |rho| on nodes with d >= 1 that have a lift flip with a neighbor.

The archive's distance (control C1 only, and beside the shares of the rotated G1 array): the least g12 on the half
plane y = 0, x >= 0 through the box center, refined by a parabola in each direction, gives (R, z0); d is the distance
to the circle of radius R about the z axis at height z0.

Usage:
  python m5_54_charge.py --selftest [--try-tag try1]      the controls C1 to C4 and the regime row
"""

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_51_action as A  # noqa: E402  (read-only import)

CK = HERE.parent / "checkpoints" / "m5_54"
DATA = HERE.parent / "data"
SUMMARY_51 = DATA / "m5_51_summary.json"

GRIDS = dict(G1=(6.0, 0.5, 25), G2=(6.0, 0.375, 33), G3=(6.0, 0.25, 49), G4=(9.0, 0.5, 37), G5=(9.0, 0.375, 49), G6=(12.0, 0.5, 49))

# the archive's printed values (hodge.json, folders soleve20b and rethink20, identical copies), 16 digits as printed
ARCHIVE = {
    "G1": dict(R=2.7998614713535988, z0=0.00251192956045454, gap_min=0.1830353720051825, Q_over_4pi=0.9986035356904263,
               share_05=0.4185542111000723, share_10=0.9308763429914628, share_15=0.9713645274767495, signed_10=0.955951440134341),
    "G2": dict(R=2.7724598297373455, z0=-0.017321051324735492, gap_min=0.1302238793986663, Q_over_4pi=0.999082418482884,
               share_05=0.8134637359594177, share_10=0.9632566442000612, share_15=0.983232892479753, signed_10=0.9821357810120291),
    "G4": dict(R=2.859061719213898, z0=0.0018107220227925325, gap_min=0.1516223172645058, Q_over_4pi=0.9994012088895041,
               share_05=0.434937806795898, share_10=0.9279953410479458, share_15=0.9688433782239733, signed_10=0.9532344779945496),
}
ARCHIVE_SHA256 = {"hodge.py": "600d1b10eb28e825e1c8cd9415fb7a945f622b15efcff36dab047df26e5798c4",
                  "hodge.json": "b9462467bedeaddbc72333e83599f61c60f39a0d87e64c8b453c9f42b9ac5bce"}
ONE_MINUS_SQRT2 = 1.0 - np.sqrt(2.0)


# ----------------------------------------------------------------------------- fields and lattice rotations

def full(m6):
    return A.from6(np.asarray(m6, float))


def cyclic(M, k=1):
    """The lattice rotation new (x, y, z) = old (y, z, x), applied k times, to a field of full matrices (N, N, N, 3, 3)."""
    p = np.array([1, 2, 0])
    for _ in range(k % 3):
        M = np.transpose(M, (1, 2, 0, 3, 4))
        M = M[..., p, :][..., :, p]
    return np.ascontiguousarray(M)


def cyclic_vec(v, k=1):
    v = np.asarray(v, float)
    for _ in range(k % 3):
        v = np.array([v[1], v[2], v[0]])
    return v


def turns_to_z(nu):
    """The number of cyclic rotations that put the box axis nearest to nu on z."""
    a = int(np.argmax(np.abs(np.asarray(nu, float))))
    return {2: 0, 0: 1, 1: 2}[a]


def positions(x):
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return np.stack([X, Y, Z], -1)


def seed_M(N, L):
    return A.seed(N, L)[0]


def zz_field(N, L):
    M = np.zeros((N, N, N, 3, 3))
    M[..., 2, 2] = 1.0
    return M


def c4c_field(N, L):
    """M = 0.999 n n^T + 0.001 * 1/3, n = (-x, -y, z) / r, r = |x| + 1e-12: charge 1 - sqrt 2 under the lift."""
    x = np.linspace(-L, L, N)
    P = positions(x)
    r = np.linalg.norm(P, axis=-1) + 1e-12
    n = P * np.array([-1.0, -1.0, 1.0]) / r[..., None]
    return 0.999 * np.einsum("...i,...j->...ij", n, n) + 0.001 * np.eye(3) / 3.0


# ----------------------------------------------------------------------------- the read

def _grad(f, h, axis, mode):
    if mode == "central":
        return np.gradient(f, h, axis=axis)
    if mode == "forward":
        g = (np.roll(f, -1, axis=axis) - f) / h
        idx = [slice(None)] * f.ndim
        idx[axis] = -1
        last = [slice(None)] * f.ndim
        last[axis] = -1
        prev = [slice(None)] * f.ndim
        prev[axis] = -2
        g[tuple(idx)] = (f[tuple(last)] - f[tuple(prev)]) / h
        return g
    raise ValueError(mode)


def bfield(nt, h, mode="central", wrong_cyclic=False):
    d = [_grad(nt, h, a, mode) for a in range(3)]
    b0 = np.einsum("...i,...i", nt, np.cross(d[1], d[2]))
    if wrong_cyclic:
        b0 = np.einsum("...i,...i", nt, np.cross(d[2], d[1]))
    b1 = np.einsum("...i,...i", nt, np.cross(d[2], d[0]))
    b2 = np.einsum("...i,...i", nt, np.cross(d[0], d[1]))
    return np.stack([b0, b1, b2], -1)


def divergence(b, h, mode="central", mixed=False):
    if mixed:
        return _grad(b[..., 0], h, 0, "forward") + _grad(b[..., 1], h, 1, "central") + _grad(b[..., 2], h, 2, "central")
    return sum(_grad(b[..., a], h, a, mode) for a in range(3))


def cube_mask(N, m):
    """The nodes with |index - center|_inf <= m."""
    c = (N - 1) // 2
    i = np.abs(np.arange(N) - c)
    I, J, K = np.meshgrid(i, i, i, indexing="ij")
    return np.maximum(np.maximum(I, J), K) <= m


def shell_mask(N, m):
    """The two-layer shell at half-widths (m - 1) h and m h."""
    c = (N - 1) // 2
    i = np.abs(np.arange(N) - c)
    I, J, K = np.meshgrid(i, i, i, indexing="ij")
    r = np.maximum(np.maximum(I, J), K)
    return (r == m) | (r == m - 1)


def flux(b, h, m):
    """The two-layer flux at half-widths (m - 1) h and m h, and the sum of its absolute terms."""
    N = b.shape[0]
    c = (N - 1) // 2
    sel = slice(c - m + 1, c + m)
    tot = 0.0
    ab = 0.0
    for a in range(3):
        ba = b[..., a]
        ix = [sel, sel, sel]

        def lay(i):
            q = list(ix)
            q[a] = i
            return ba[tuple(q)]
        t = 0.5 * (lay(c + m) + lay(c + m - 1) - lay(c - m + 1) - lay(c - m))
        tot += float(t.sum())
        ab += float(np.abs(t).sum())
    return h * h * tot, h * h * ab


def flips(nt, mask):
    """Lift flips (nt_i . nt_j < 0) between neighboring nodes that both lie in mask; and the per-node flag."""
    cnt = 0
    flag = np.zeros(mask.shape, bool)
    for a in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[a] = slice(0, -1)
        s1[a] = slice(1, None)
        s0, s1 = tuple(s0), tuple(s1)
        dot = (nt[s0] * nt[s1]).sum(-1)
        both = mask[s0] & mask[s1]
        f = (dot < 0) & both
        cnt += int(f.sum())
        flag[s0] |= f
        flag[s1] |= f
    return cnt, flag


def raw_sign_pairs(n, X, mask):
    """A wrong regime test (the failing input of C4 (b)): neighbor pairs of opposite sign of n . x on the raw eigenvector."""
    sg = np.sign((n * X).sum(-1))
    cnt = 0
    for a in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[a] = slice(0, -1)
        s1[a] = slice(1, None)
        s0, s1 = tuple(s0), tuple(s1)
        cnt += int(((sg[s0] * sg[s1] < 0) & mask[s0] & mask[s1]).sum())
    return cnt


def archive_gap_read(g, x):
    """The archive's reader: least g12 on the half plane y = 0, x >= 0 (index c on y), parabolic refinement."""
    N = len(x)
    h = float(x[1] - x[0])
    c = N // 2
    sl = g[c:, c, :]
    i, k = np.unravel_index(np.argmin(sl), sl.shape)

    def par(a, j):
        if 0 < j < len(a) - 1:
            den = a[j - 1] - 2 * a[j] + a[j + 1]
            return j + (0.5 * (a[j - 1] - a[j + 1]) / den if den != 0 else 0)
        return j
    R = x[c] + par(sl[:, k], i) * h
    z0 = x[0] + par(sl[i, :], k) * h
    return float(R), float(z0), float(sl[i, k])


def shares(rho, X, inner, c, nu, R, flag=None):
    nu = np.asarray(nu, float)
    nu = nu / np.linalg.norm(nu)
    q = X - np.asarray(c, float)
    z = q @ nu
    rc = np.sqrt(np.maximum((q * q).sum(-1) - z * z, 0.0))
    d = np.sqrt((rc - R) ** 2 + z * z)
    a = np.abs(rho[inner]).sum()
    out = {f"share_{int(dd * 10):02d}": float(np.abs(rho[inner & (d < dd)]).sum() / a) for dd in (0.5, 1.0, 1.5)}
    out["signed_10"] = float(rho[inner & (d < 1)].sum() / rho[inner].sum())
    if flag is not None:
        out["branch"] = float(np.abs(rho[inner & (d >= 1) & flag]).sum() / a)
    return out


def archive_shares(rho, X, inner, R, z0):
    rc = np.sqrt(X[..., 0] ** 2 + X[..., 1] ** 2)
    d = np.sqrt((rc - R) ** 2 + (X[..., 2] - z0) ** 2)
    a = np.abs(rho[inner]).sum()
    out = {f"share_{int(dd * 10):02d}": float(np.abs(rho[inner & (d < dd)]).sum() / a) for dd in (0.5, 1.0, 1.5)}
    out["signed_10"] = float(rho[inner & (d < 1)].sum() / rho[inner].sum())
    return out


def charge_read(M, x, RL=None, circle=None, archive=False, eig="top", lift=+1, no_lift=False, mode="central",
                wrong_cyclic=False, include_faces=False, mixed=False, flip_center=False, keep=False):
    """Every quantity of section 1 on one field of full matrices M (N, N, N, 3, 3).

    RL      the radius of the state's ring for the regime clause l2 - h >= R_L + 1.5 (0 for a field with no ring)
    circle  (c, nu, R) of M5.51's line reader on the same state, for the shares
    archive the archive's gap read and its distance, beside
    The other options are the failing inputs of the controls."""
    N = len(x)
    h = float(x[1] - x[0])
    L = float(x[-1])
    Mh = (N - 1) // 2
    X = positions(x)
    w, V = np.linalg.eigh(M)
    g = w[..., 2] - w[..., 1]
    n = V[..., :, 2] if eig == "top" else V[..., :, 0]
    if no_lift:
        nt = n.copy()
    else:
        s = np.sign((n * X).sum(-1) + 1e-12)
        nt = lift * n * s[..., None]
    if flip_center:
        c0 = (N - 1) // 2
        nt[c0, c0, c0] = -nt[c0, c0, c0]
    b = bfield(nt, h, mode, wrong_cyclic)
    rho = divergence(b, h, mode, mixed)
    inner = np.zeros((N, N, N), bool)
    inner[1:-1, 1:-1, 1:-1] = True
    summ = inner if not include_faces else np.ones((N, N, N), bool)
    h3 = h ** 3
    Q = float(rho[summ].sum() * h3)
    out = dict(N=N, h=h, L=L, Q_over_4pi=Q / (4 * np.pi))
    # the profile and the identity (C3) at every half-width
    prof = []
    ident = []
    for m in range(2, Mh + 1):
        cm = cube_mask(N, m - 1)
        Qv = float(rho[cm].sum() * h3)
        Qf, ab = flux(b, h, m)
        av = float(np.abs(rho[cm]).sum() * h3)
        prof.append(dict(l=m * h, Q_over_4pi=Qv / (4 * np.pi)))
        ident.append(dict(l=m * h, Q_vol=Qv, Q_flux=Qf, diff=Qv - Qf, abs_vol=av, abs_flux=ab, rel=abs(Qv - Qf) / min(av, ab) if min(av, ab) > 0 else abs(Qv - Qf)))
    out["profile"] = prof
    out["Q_l2_over_4pi"] = prof[-3]["Q_over_4pi"]  # l = L - 2h
    out["l2"] = L - 2 * h
    out["identity"] = dict(max_rel=float(max(q["rel"] for q in ident)), at_L=ident[-1], at_l2=ident[-3],
                           box_sum_vs_flux=dict(Q_box=Q, Q_flux_L=ident[-1]["Q_flux"], diff=Q - ident[-1]["Q_flux"]))
    # the regime
    reg = {}
    ok = True
    for name, m in (("L", Mh), ("l2", Mh - 2)):
        sm = shell_mask(N, m)
        nf, _ = flips(nt, sm)
        gmin = float(g[sm].min())
        nlow = int((g[sm] <= 0.2).sum())
        reg[name] = dict(half_widths=[(m - 1) * h, m * h], lift_flips=nf, least_g12=gmin, nodes_g12_le_02=nlow,
                         raw_sign_pairs=raw_sign_pairs(n, X, sm), nt_dot_x_negative=int(((nt * X).sum(-1)[sm] < 0).sum()))
        ok &= (nf == 0 and nlow == 0)
    rad = None
    if RL is not None:
        rad = dict(l2_minus_h=L - 3 * h, RL_plus_15=RL + 1.5, holds=bool(L - 3 * h >= RL + 1.5))
        ok &= rad["holds"]
    reg["radius_clause"] = rad
    reg["in_regime"] = bool(ok and rad is not None)
    out["regime"] = reg
    _, flag = flips(nt, np.ones((N, N, N), bool))
    out["lift_flip_nodes"] = int(flag.sum())
    if circle is not None:
        c, nu, R = circle
        out["shares"] = shares(rho, X, inner, c, nu, R, flag)
        out["circle"] = dict(center=list(map(float, c)), normal=list(map(float, nu)), R=float(R))
    if archive:
        R, z0, gmin = archive_gap_read(g, x)
        out["archive_read"] = dict(R=R, z0=z0, gap_min=gmin)
        out["archive_shares"] = archive_shares(rho, X, inner, R, z0)
    out["least_g12_node"] = float(g.min())
    if keep:
        out["_rho"] = rho
        out["_nt"] = nt
    return out


# ----------------------------------------------------------------------------- the controls

def _stored_raw(tag):
    m6, x, hb = A.load_stored(tag)
    return full(m6), x, hb


def _s51():
    return json.loads(SUMMARY_51.read_text())


def control_c3(reads):
    """C3 on every field read: the volume sum and the two-layer flux within 1e-12 of the sum of the absolute terms."""
    worst = max(r["identity"]["max_rel"] for r in reads.values())
    return dict(passed=bool(worst <= 1e-12), worst_rel=worst, per_field={k: r["identity"]["max_rel"] for k, r in reads.items()})


def selftest(tag):
    CK.mkdir(parents=True, exist_ok=True)
    s51 = _s51()
    res = dict(archive_sha256=ARCHIVE_SHA256)
    reads = {}
    allok = True

    # C4 (a), (b), (c) on the grids of G1, G2, G4, G5
    c4 = dict(a={}, b={}, c={})
    ok_a = ok_b = ok_c = True
    for gname in ("G1", "G2", "G4", "G5"):
        L, h, N = GRIDS[gname]
        x = np.linspace(-L, L, N)
        M0 = seed_M(N, L)
        r = charge_read(M0, x, RL=0.0)
        reads[f"seed_{gname}"] = r
        fail = dict(opposite_lift=charge_read(M0, x, RL=0.0, lift=-1)["Q_over_4pi"],
                    wrong_cyclic=charge_read(M0, x, RL=0.0, wrong_cyclic=True)["Q_over_4pi"],
                    least_eigenvector=charge_read(M0, x, RL=0.0, eig="least")["Q_over_4pi"])
        pa = bool(abs(r["Q_over_4pi"] - 1) <= 0.002 and abs(r["Q_l2_over_4pi"] - 1) <= 0.0044 and r["regime"]["in_regime"])
        fails_fail = {k: bool(abs(v - 1) > 0.002) for k, v in fail.items()}
        c4["a"][gname] = dict(Q_over_4pi=r["Q_over_4pi"], Q_l2_over_4pi=r["Q_l2_over_4pi"], in_regime=r["regime"]["in_regime"], regime=r["regime"],
                              passed=pa, failing_inputs=fail, failing_inputs_miss=fails_fail)
        ok_a &= pa and all(fails_fail.values())
        print("C4a", gname, f"{r['Q_over_4pi']:.6f}", f"{r['Q_l2_over_4pi']:.6f}", r["regime"]["in_regime"], json.dumps(fail), flush=True)

        Mz = zz_field(N, L)
        rz = charge_read(Mz, x, RL=0.0)
        reads[f"zz_{gname}"] = rz
        wrong_in = bool(all(rz["regime"][s]["nt_dot_x_negative"] == 0 for s in ("L", "l2")))
        wrong_raw_in = bool(all(rz["regime"][s]["raw_sign_pairs"] == 0 for s in ("L", "l2")))
        pb = bool(not rz["regime"]["in_regime"])
        c4["b"][gname] = dict(Q_over_4pi=rz["Q_over_4pi"], in_regime=rz["regime"]["in_regime"], regime=rz["regime"], passed=pb,
                              failing_input_test_on_nt_dot_x_reads_in_regime=wrong_in, failing_input_test_on_raw_sign_reads_in_regime=wrong_raw_in)
        ok_b &= pb and wrong_in
        print("C4b", gname, f"{rz['Q_over_4pi']:.3e}", "in regime" if rz["regime"]["in_regime"] else "out of regime",
              "flips", rz["regime"]["L"]["lift_flips"], rz["regime"]["l2"]["lift_flips"], "nt.x test in:", wrong_in, "raw test in:", wrong_raw_in, flush=True)

        Mc = c4c_field(N, L)
        rc = charge_read(Mc, x, RL=0.0)
        reads[f"c4c_{gname}"] = rc
        failc = dict(no_lift=charge_read(Mc, x, RL=0.0, no_lift=True)["Q_over_4pi"], opposite_lift=charge_read(Mc, x, RL=0.0, lift=-1)["Q_over_4pi"])
        pc = bool(abs(rc["Q_over_4pi"] - ONE_MINUS_SQRT2) <= 0.01 and not rc["regime"]["in_regime"])
        failc_miss = {k: bool(abs(v - ONE_MINUS_SQRT2) > 0.01) for k, v in failc.items()}
        c4["c"][gname] = dict(Q_over_4pi=rc["Q_over_4pi"], target=ONE_MINUS_SQRT2, diff=rc["Q_over_4pi"] - ONE_MINUS_SQRT2,
                              in_regime=rc["regime"]["in_regime"], regime=rc["regime"], passed=pc, failing_inputs=failc, failing_inputs_miss=failc_miss)
        ok_c &= pc and all(failc_miss.values())
        print("C4c", gname, f"{rc['Q_over_4pi']:.6f}", f"diff {rc['Q_over_4pi'] - ONE_MINUS_SQRT2:+.5f}", rc["regime"]["in_regime"], json.dumps(failc), flush=True)
    res["C4a"] = dict(passed=bool(ok_a), grids=c4["a"])
    res["C4b"] = dict(passed=bool(ok_b), grids=c4["b"])
    res["C4c"] = dict(passed=bool(ok_c), grids=c4["c"])
    allok &= ok_a and ok_b and ok_c

    # the regime row: the four stored arrays, as stored (the archive read them as stored)
    reg_rows = {}
    ok_r = True
    for gname in ("G1", "G2", "G4", "G6"):
        M, x, hb = _stored_raw(gname)
        st = s51["stored"][gname]
        face_dev = float(np.abs(A.to6(M) - A.to6(seed_M(len(x), hb)))[~cube_mask(len(x), (len(x) - 1) // 2 - 1)].max())
        r = charge_read(M, x, RL=st["RL"], circle=(st["center"], st["normal"], st["RL"]))
        reads[f"author_{gname}"] = r
        reg_rows[gname] = dict(in_regime=r["regime"]["in_regime"], regime=r["regime"], Q_over_4pi=r["Q_over_4pi"], Q_l2_over_4pi=r["Q_l2_over_4pi"],
                               face_deviation_from_seed=face_dev)
        ok_r &= r["regime"]["in_regime"]
        print("regime", gname, r["regime"]["in_regime"], "flips", r["regime"]["L"]["lift_flips"], r["regime"]["l2"]["lift_flips"],
              f"least g12 {r['regime']['L']['least_g12']:.4f} {r['regime']['l2']['least_g12']:.4f}", "raw-sign pairs on the box shell",
              r["regime"]["L"]["raw_sign_pairs"], f"Q {r['Q_over_4pi']:.7f} {r['Q_l2_over_4pi']:.5f}", f"face dev {face_dev:.2e}", flush=True)
    zz_out = all(not c4["b"][g]["in_regime"] for g in c4["b"])
    res["regime_row"] = dict(passed=bool(ok_r and zz_out), arrays=reg_rows, failing_input_C4b_reads_out=bool(zz_out))
    allok &= ok_r and zz_out

    # C1: the port against the archive, on the stored G1 array rotated (new x, y, z = old y, z, x)
    M, x, hb = _stored_raw("G1")
    st = s51["stored"]["G1"]
    Mr = cyclic(M, 1)
    circ = (cyclic_vec(st["center"]), cyclic_vec(st["normal"]), st["RL"])
    r = charge_read(Mr, x, RL=st["RL"], circle=circ, archive=True)
    reads["author_G1_rotated"] = r
    ru = reads["author_G1"]
    ar = ARCHIVE["G1"]
    got = dict(Q_over_4pi=r["Q_over_4pi"], gap_min=r["archive_read"]["gap_min"], R=r["archive_read"]["R"], z0=r["archive_read"]["z0"],
               **r["archive_shares"])
    diffs = {k: got[k] - ar[k] for k in ar}
    c1_ok = all(abs(v) <= 1e-10 for v in diffs.values())
    rot_vs_unrot = r["Q_over_4pi"] - ru["Q_over_4pi"]
    c1_ok &= abs(rot_vs_unrot) <= 1e-12
    center_retry = None
    share_keys = ("share_05", "share_10", "share_15", "signed_10")
    if not c1_ok and all(abs(diffs[k]) <= 1e-10 for k in ar if k not in share_keys) and abs(rot_vs_unrot) <= 1e-12:
        rf = charge_read(Mr, x, RL=st["RL"], archive=True, flip_center=True)
        center_retry = {k: rf["archive_shares"][k] - ar[k] for k in share_keys}
    # the failing inputs
    fin = {}
    q = charge_read(Mr, x, lift=-1)["Q_over_4pi"]
    fin["opposite_lift"] = dict(Q_over_4pi=q, misses=bool(abs(q - ar["Q_over_4pi"]) > 1e-10))
    q = charge_read(Mr, x, eig="least")["Q_over_4pi"]
    fin["least_eigenvector"] = dict(Q_over_4pi=q, misses=bool(abs(q - ar["Q_over_4pi"]) > 1e-10))
    q = charge_read(Mr, x, mode="forward")["Q_over_4pi"]
    fin["forward_differences"] = dict(Q_over_4pi=q, misses=bool(abs(q - ar["Q_over_4pi"]) > 1e-10))
    q = charge_read(Mr, x, include_faces=True)["Q_over_4pi"]
    fin["sum_with_face_nodes"] = dict(Q_over_4pi=q, misses=bool(abs(q - ar["Q_over_4pi"]) > 1e-10))
    ru_a = charge_read(M, x, archive=True)
    sd = {k: ru_a["archive_shares"][k] - ar[k] for k in share_keys}
    fin["shares_on_unrotated_array"] = dict(diffs=sd, misses=bool(any(abs(v) > 1e-10 for v in sd.values())))
    fin_ok = all(v["misses"] for v in fin.values())
    res["C1"] = dict(passed=bool(c1_ok and fin_ok), values=got, archive=ar, diffs=diffs, Q_rotated_minus_unrotated=rot_vs_unrot,
                     fitted_circle_shares=r["shares"], center_node_retry=center_retry, failing_inputs=fin, failing_inputs_all_miss=bool(fin_ok))
    allok &= c1_ok and fin_ok
    print("C1", "PASS" if (c1_ok and fin_ok) else "FAIL", json.dumps({k: f"{v:+.3e}" for k, v in diffs.items()}), f"rot-unrot {rot_vs_unrot:+.2e}",
          json.dumps({k: v.get("Q_over_4pi", v.get("diffs")) for k, v in fin.items()}, default=float), flush=True)

    # C2: the archive's other two arrays (stored G2 and G4, as stored)
    c2 = {}
    ok2 = True
    for gname, post in (("G2", 0.9991), ("G4", 0.9994)):
        r = reads[f"author_{gname}"]
        d = r["Q_over_4pi"] - post
        ok = abs(d) <= 0.001
        ok2 &= ok
        row = dict(Q_over_4pi=r["Q_over_4pi"], archive_printed=post, archive_16=ARCHIVE[gname]["Q_over_4pi"], diff=d,
                   diff_to_16_digits=r["Q_over_4pi"] - ARCHIVE[gname]["Q_over_4pi"], passed=bool(ok), fitted_circle_shares=r["shares"],
                   archive_shares_on_their_file={k: ARCHIVE[gname][k] for k in share_keys})
        st = s51["stored"][gname]
        if st["cls"] == "axis":
            Mg, xg, _ = _stored_raw(gname)
            k = turns_to_z(st["normal"])
            rr = charge_read(cyclic(Mg, k), xg, archive=True)
            row["rotated_to_z"] = dict(turns=k, archive_read=rr["archive_read"], archive_shares=rr["archive_shares"])
        c2[gname] = row
        print("C2", gname, f"{r['Q_over_4pi']:.7f}", f"against {post}: {d:+.2e}", "PASS" if ok else "FAIL", flush=True)
    res["C2"] = dict(passed=bool(ok2), arrays=c2)
    allok &= ok2

    # C3 on every field read here
    res["C3"] = control_c3(reads)
    allok &= res["C3"]["passed"]
    # the failing inputs of C3: a sum over the face nodes; a stencil that mixes orders. Try 1 read them on the seed of
    # G1, whose mirror symmetry makes the forward and central telescopes agree, so the mixed stencil could not fail
    # there; from try 2 they are read on the author's stored G1 array (a field with no mirror symmetry)
    M0, x, _ = _stored_raw("G1")
    rf = charge_read(M0, x, include_faces=True)
    rm = charge_read(M0, x, mixed=True)
    res["C3"]["failing_inputs"] = dict(
        sum_with_face_nodes=dict(Q_box=rf["identity"]["box_sum_vs_flux"]["Q_box"], Q_flux_L=rf["identity"]["box_sum_vs_flux"]["Q_flux_L"],
                                 rel=abs(rf["identity"]["box_sum_vs_flux"]["diff"]) / rf["identity"]["at_L"]["abs_vol"]),
        mixed_stencil=dict(max_rel=rm["identity"]["max_rel"]))
    fi = res["C3"]["failing_inputs"]
    res["C3"]["failing_inputs_miss"] = bool(fi["sum_with_face_nodes"]["rel"] > 1e-12 and fi["mixed_stencil"]["max_rel"] > 1e-12)
    res["C3"]["passed"] = bool(res["C3"]["passed"] and res["C3"]["failing_inputs_miss"])
    allok &= res["C3"]["passed"]
    print("C3", "PASS" if res["C3"]["passed"] else "FAIL", f"worst {res['C3']['worst_rel']:.2e}", json.dumps(fi), flush=True)

    res["passed"] = bool(allok)
    res["seed_values"] = {g: dict(Q_over_4pi=reads[f"seed_{g}"]["Q_over_4pi"], Q_l2_over_4pi=reads[f"seed_{g}"]["Q_l2_over_4pi"]) for g in ("G1", "G2", "G4", "G5")}
    res["reads"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in reads.items()}
    f = CK / f"controls_{tag}.json"
    f.write_text(json.dumps(res, indent=1, default=_jsonable) + "\n")
    print("controls:", "ALL PASS" if allok else "FAIL", "->", f.name, flush=True)
    for k in ("C1", "C2", "C3", "C4a", "C4b", "C4c", "regime_row"):
        print(" ", k, "met" if res[k]["passed"] else "NOT MET", flush=True)
    return allok


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--try-tag", default="try1")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.try_tag) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
