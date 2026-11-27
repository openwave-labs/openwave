"""M5.57: the line moment D' (held), the degree moment D (read) with its lift, the hold, the far field and the starts
of rung A, with the fixed-field controls C1 to C4 and C9 of the plan under --selftest.

The action is that of M5.51 (m5_51_action.py, unchanged) with the midpoint weight at delta = 0.05, kappa = v = L_V = 1,
L_1 = 0, faces held at the far field (the attribute M0 of each model is replaced before any hold_faces). With
l1 >= l2 >= l3 the eigenvalues of M, e1, e2 their eigenvectors, g12 = l1 - l2, |x| the distance of a node to the plane
x = 0, and the + (-) half the inner nodes with x > 0 (x < 0):

    w = exp(-g12^2 / g0^2),  g0 = 0.2,       g12^2 = (l1 + l2)^2 - 4 l1 l2
    D'_pm = sum_pm |x| w / sum_pm w,          D' = D'_+ + D'_-           (one ring: D'_1 = sum x w / sum w, inner nodes)
    dw/dM = -2 w g12 / g0^2 (e1 e1^T - e2 e2^T)

    n_ref: the direction field of z = z1 z2 with centers (+-d_c/2, 0, 0) (no ladder, no core)
    nt = s n,  s = sgn(n . n_ref + 1e-12),  b_a = nt . (d_b nt x d_c nt) (cyclic),  rho = sum_a d_a b_a   (M5.54's differences)
    Q_pm = h^3 / (4 pi) sum_pm rho,           D_pm = h^3 / (4 pi) sum_pm |x| rho,     D = D_+ + D_-

    E_mu = U + mu/2 (D' - d)^2,   f = mu (d - D')

Usage:
  python m5_57_hold.py --selftest [--only C1 C2 ...] [--try N]      C1 to C4 and C9 (fixed fields)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_51_action as A  # noqa: E402
import m5_54_charge as C54  # noqa: E402
import m5_56_turn as T  # noqa: E402

CK = HERE.parent / "checkpoints" / "m5_57"
CK56 = HERE.parent / "checkpoints" / "m5_56" / "chains"
DELTA = 0.05
G0 = 0.2
DFAR = 14.0
GRIDS = dict(H1=(12.0, 0.5, 49), P=(18.0, 0.5, 73), P375=(18.0, 0.375, 97), A24=(24.0, 0.5, 97), A30=(30.0, 0.5, 121))
STORED = {22: "r22_B4a", 24: "r24_B4-H2", 30: "r30_P4-14", 31: "r31_P4-10", 33: "r33_P4-0"}


# ----------------------------------------------------------------------------- grids, stored ends, fields

def axis_of(grid):
    hb, h, N = GRIDS[grid]
    return np.linspace(-hb, hb, N)


def load56(row):
    """A stored end of M5.56 (read-only): field, axis, end reads."""
    name = STORED[row]
    rec = json.loads((CK56 / f"{name}.json").read_text())
    z = np.load(CK56 / name / rec["end_file"], allow_pickle=False)
    return np.asarray(z["m6"], float), np.asarray(z["x"], float), rec


def _stereo_n(X, dd):
    """n from z = z1 z2 with centers (+-dd/2, 0, 0) (dd per node), as M5.56's pair seed; also rho_1, rho_2."""
    zs, rhos = [], []
    for s in (+1, -1):
        c = np.zeros(X.shape)
        c[..., 0] = s * dd / 2
        Y = X - c
        rk = np.linalg.norm(Y, axis=-1) + 1e-12
        u = Y / rk[..., None]
        den = 1 - u[..., 2]
        zs.append((u[..., 0] + 1j * u[..., 1]) / np.where(np.abs(den) < 1e-15, 1e-15, den))
        rhos.append(rk)
    z = zs[0] * zs[1]
    a2 = np.abs(z) ** 2
    big = ~np.isfinite(a2) | (a2 > 1e30)
    with np.errstate(invalid="ignore", over="ignore"):
        n = np.stack([2 * z.real, 2 * z.imag, a2 - 1], -1) / (a2 + 1)[..., None]
    n[big] = [0, 0, 1.0]
    return n, rhos


def pair_field(x, dd, delta=DELTA):
    """M_0delta^(2) of M5.56 section 12 with a separation d per node (a scalar for no ladder)."""
    X = T.mesh(x)
    dd = np.broadcast_to(np.asarray(dd, float), X.shape[:-1])
    n, rhos = _stereo_n(X, dd)
    amp = np.tanh(rhos[0] / 1.2) * np.tanh(rhos[1] / 1.2)
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
    return A.to6(mb * np.eye(3) + amp[..., None, None] * (nn + B - mb * np.eye(3)))


def far_field(x, dfar=DFAR):
    return pair_field(x, dfar)


def ladder_d(x, d0, dfar, rin, rout):
    r = np.linalg.norm(T.mesh(x), axis=-1)
    t = np.clip((r - rin) / (rout - rin), 0.0, 1.0)
    return d0 + (dfar - d0) * (3 * t * t - 2 * t ** 3)


def rungA_start(x, d0, dfar):
    """Plan section 3: the pair seed with the ladder recast for the box, R_in = d0/2 + 4.35, R_out = L - h; no ladder
    when R_out - R_in < 4. Faces reset to the far field of dfar."""
    L, h = x[-1], x[1] - x[0]
    rin, rout = d0 / 2 + 4.35, L - h
    if rout - rin < 4:
        m6 = pair_field(x, d0)
        lad = None
    else:
        m6 = pair_field(x, ladder_d(x, d0, dfar, rin, rout))
        lad = [rin, rout]
    hold = far_field(x, dfar)
    face = T.faces_mask(len(x))
    m6 = np.where(face[..., None], hold, m6)
    return m6, hold, lad


def make_model(x, hold6):
    N = len(x)
    mod = A.Model(N, float(x[-1]), weight="mid", delta=DELTA)
    mod.M0 = A.from6(np.asarray(hold6, float).reshape(N, N, N, 6)).copy()
    return mod


def moved(m6, k, ax, hold6):
    """The field moved by k cells along axis ax: the field at the node k cells back, the vacated layers filled by the
    nearest stored layer, the faces reset to hold6."""
    N = m6.shape[0]
    idx = np.clip(np.arange(N) - k, 0, N - 1)
    out = np.take(m6, idx, axis=ax)
    face = T.faces_mask(N)
    return np.where(face[..., None], hold6, out)


def interp(m6, xs, x):
    return T.interp_to(m6, xs, x)


# ----------------------------------------------------------------------------- the line moment D'

def halves(x):
    N = len(x)
    inner = ~T.faces_mask(N)
    X = T.mesh(x)
    plus = inner & (X[..., 0] > 1e-9)
    minus = inner & (X[..., 0] < -1e-9)
    return inner, plus, minus, X


def dprime(m6, x, grad=False, one=False, g0=G0, normalized=True, drop_norm_term=False, eig=None):
    """D'_+, D'_- and D' (one=True: D'_1 over the inner nodes); with grad the gradient of D' (or D'_1) over the six
    components (faces zero). The other options are the failing inputs of C1 and C2."""
    inner, plus, minus, X = halves(x)
    w_, V = T.eigs(m6) if eig is None else eig
    g12 = w_[..., 0] - w_[..., 1]
    wt = np.exp(-g12 * g12 / (g0 * g0))
    out = dict()
    coef = np.zeros(wt.shape)
    if one:
        a = X[..., 0]
        W = wt[inner].sum()
        D1 = float((a * wt)[inner].sum() / W)
        out.update(D1=D1, W=float(W), nodes_equiv=float(W))
        coef[inner] = ((a - D1) / W)[inner] if not drop_norm_term else (a / W)[inner]
        val = D1
    else:
        a = np.abs(X[..., 0])
        res = {}
        for lab, H in (("p", plus), ("m", minus)):
            W = wt[H].sum()
            S = (a * wt)[H].sum()
            Dh = float(S / W) if normalized else float(S)
            res[lab] = (Dh, float(W))
            if normalized:
                coef[H] = ((a - Dh) / W)[H] if not drop_norm_term else (a / W)[H]
            else:
                coef[H] = a[H]
        out.update(Dp=res["p"][0], Dm=res["m"][0], D=res["p"][0] + res["m"][0], Wp=res["p"][1], Wm=res["m"][1])
        val = out["D"]
    out["value"] = val
    if grad:
        e1 = V[..., :, 0]
        e2 = V[..., :, 1]
        P = np.einsum("...i,...j->...ij", e1, e1) - np.einsum("...i,...j->...ij", e2, e2)
        dw = (-2.0 * wt * g12 / (g0 * g0))[..., None, None] * P
        G = coef[..., None, None] * dw
        out["grad"] = A.grad_to6(G) * inner[..., None]
    return out


def dprime_regime(m6, x, eig=None, g0=G0, min_size=20):
    """Regime of a D' read: (i) the share of w within 1.5 of the line (the components of at least 20 nodes of
    {g12 < 0.5, r < 0.93 L}, 26-neighborhood) at least 0.99 over the inner nodes; (ii) l2 - l3 >= 0.1 on every node
    with w > 1e-6."""
    h = x[1] - x[0]
    L = x[-1]
    inner, plus, minus, X = halves(x)
    w_, V = T.eigs(m6) if eig is None else eig
    g12 = w_[..., 0] - w_[..., 1]
    g23 = w_[..., 1] - w_[..., 2]
    wt = np.exp(-g12 * g12 / (g0 * g0))
    r = np.linalg.norm(X, axis=-1)
    Z = (g12 < 0.5) & (r < 0.93 * L)
    lab, k = ndimage.label(Z, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel(), minlength=k + 1)
    sizes[0] = 0
    big = np.flatnonzero(sizes >= min_size)
    line = np.isin(lab, big) if len(big) else np.zeros(Z.shape, bool)
    if line.any():
        dist = ndimage.distance_transform_edt(~line, sampling=h)
    else:
        dist = np.full(Z.shape, np.inf)
    near = dist <= 1.5 + 1e-9
    W = wt[inner].sum()
    share = float(wt[inner & near].sum() / W)
    shares_half = [float(wt[H & near].sum() / wt[H].sum()) for H in (plus, minus)]
    heavy = wt > 1e-6
    g23_min = float(g23[heavy].min()) if heavy.any() else None
    ok_i = bool(share >= 0.99)
    ok_ii = bool(g23_min is not None and g23_min >= 0.1)
    return dict(share=share, share_per_half=shares_half, components=[int(sizes[j]) for j in big], line_nodes=int(line.sum()),
                least_g23_where_w_gt_1e6=g23_min, clause_i=ok_i, clause_ii=ok_ii, in_regime=bool(ok_i and ok_ii))


# ----------------------------------------------------------------------------- the degree moment D

def nref_field(x, dc):
    X = T.mesh(x)
    n, _ = _stereo_n(X, np.full(X.shape[:-1], float(dc)))
    return n


def dmoment(m6, x, dc, radial=False, eig=None, ref_vec=None, keep=False):
    """Q_pm, D_pm, D and the regime of section 1 (radial: the failing input of C4, the lift radial about each center
    with no half-turn for the - half; ref_vec: the lift taken from a given oriented field, the failing input of C3)."""
    N = len(x)
    h = float(x[1] - x[0])
    c = (N - 1) // 2
    inner, plus, minus, X = halves(x)
    w_, V = T.eigs(m6) if eig is None else eig
    g12 = w_[..., 0] - w_[..., 1]
    n = V[..., :, 0]
    if ref_vec is not None:
        ref = ref_vec
    elif radial:
        ref = np.zeros(X.shape)
        for s, H in ((+1, X[..., 0] >= 0), (-1, X[..., 0] < 0)):
            Y = X - np.array([s * dc / 2, 0, 0])
            ref[H] = (Y / (np.linalg.norm(Y, axis=-1) + 1e-12)[..., None])[H]
    else:
        ref = nref_field(x, dc)
    sg = np.sign((n * ref).sum(-1) + 1e-12)
    nt = n * sg[..., None]
    b = C54.bfield(nt, h)
    rho = C54.divergence(b, h)
    a = np.abs(X[..., 0])
    k4 = h ** 3 / (4 * np.pi)
    Qp, Qm = float(k4 * rho[plus].sum()), float(k4 * rho[minus].sum())
    Dp, Dm = float(k4 * (a * rho)[plus].sum()), float(k4 * (a * rho)[minus].sum())
    # the regime
    i = np.arange(N)
    I, J, K = np.meshgrid(i, i, i, indexing="ij")
    outer2 = (np.minimum(np.minimum(I, J), K) <= 1) | (np.maximum(np.maximum(I, J), K) >= N - 2)
    zlay = (K <= 1) | (K >= N - 2)
    pole = np.zeros((N, N, N), bool)
    for s in (+1, -1):
        pole |= zlay & (np.sqrt((X[..., 0] - s * dc / 2) ** 2 + X[..., 1] ** 2) <= 2 * h + 1e-9)
    reg = {}
    ok = True
    for lab, side in (("plus", +1), ("minus", -1)):
        part = (outer2 & ((I > c) if side > 0 else (I < c))) | (I == c) | (I == c + side)
        cnt = 0
        cnt_pole = 0
        for ax in range(3):
            s0 = [slice(None)] * 3
            s1 = [slice(None)] * 3
            s0[ax] = slice(0, -1)
            s1[ax] = slice(1, None)
            s0, s1 = tuple(s0), tuple(s1)
            both = part[s0] & part[s1]
            fl = both & ((nt[s0] * nt[s1]).sum(-1) < 0)
            ex = pole[s0] | pole[s1]
            cnt += int((fl & ~ex).sum())
            cnt_pole += int((fl & ex).sum())
        reg[lab] = dict(flips=cnt, flips_at_the_poles_left_out=cnt_pole)
        ok &= cnt == 0
    planes = (I == c) | (I == c + 1) | (I == c - 1)
    gmin = float(g12[planes].min())
    reg["least_g12_on_planes"] = gmin
    reg["Q_ok"] = bool(abs(Qp - 1) <= 0.01 and abs(Qm - 1) <= 0.01)
    reg["g12_ok"] = bool(gmin >= 0.2)
    reg["flips_ok"] = bool(ok)
    inr = bool(reg["Q_ok"] and reg["g12_ok"] and ok)
    out = dict(dc=float(dc), Qp=Qp, Qm=Qm, Dp=Dp, Dm=Dm, D=Dp + Dm, regime=reg, in_regime=inr)
    if keep:
        out["_nt"] = nt
    return out


# ----------------------------------------------------------------------------- the hold

class Held:
    """E = U (+ mu/2 (D' - d)^2 when held); the gradient over the six components with the faces zero. form = 'pair'
    (D' = D'_+ + D'_-) or 'one' (D'_1)."""

    def __init__(self, mod, x, target=None, mu=None, form="pair"):
        self.mod, self.x, self.target, self.mu, self.form = mod, x, target, mu, form
        self.h = x[1] - x[0]

    @property
    def held(self):
        return self.target is not None

    def evaluate(self, m6, need_grad=True):
        p = self.mod.evaluate(m6, need_grad=need_grad)
        out = dict(U=p["U"], U4=p["U4"], UV=p["UV"], ULV=p["ULV"], E=p["U"])
        if need_grad:
            out["grad"] = p["grad"]
        if self.held:
            dp = dprime(np.asarray(m6).reshape(p["grad"].shape if need_grad else (len(self.x),) * 3 + (6,)), self.x, grad=need_grad,
                        one=self.form == "one")
            Dv = dp["value"]
            f = self.mu * (self.target - Dv)
            out.update(Dprime=Dv, f=f, hold_energy=0.5 * self.mu * (Dv - self.target) ** 2)
            out["E"] = p["U"] + out["hold_energy"]
            if need_grad:
                out["grad"] = p["grad"] - f * dp["grad"]
                out["F"] = float(np.abs(f * dp["grad"]).max() / self.h ** 3)
        return out

    def gate_threshold(self, F):
        if not self.held:
            return 1e-3
        return max(1e-5, min(1e-3, 0.1 * F))


def torque(mod, grad6_unused, m6, gU):
    """The minor torque of M5.56 section 10 from the gradient over the nine entries gU (of U: the hold adds none, since
    D' depends on the eigenvalues only)."""
    M = A.from6(m6)
    _, V = T.eigs(m6)
    K = T.skew(V[..., :, 0])
    tq = (gU * (K @ M - M @ K)).sum((-1, -2)) * mod.mask
    return float(np.abs(tq).max() / mod.h ** 3)


# ----------------------------------------------------------------------------- the start rule of rung A

def rungA_rule(m6, x):
    """No jump link anywhere (both nodes s < 1e-4, top eigenvectors more than 45 degrees apart), the box surface
    liftable with |q| = 2."""
    N = len(x)
    allreg = np.ones((N, N, N), bool)
    jl = T.jump_links(m6, x, DELTA, allreg)
    box = T._cube_q(m6, x, np.zeros(3), x[-1] - (x[1] - x[0]))
    ok = bool(jl["jumps"] == 0 and box.get("liftable") and box.get("abs_q") == 2)
    return dict(largest_vacuum_deg=jl["largest_deg"], jump_links=jl["jumps"], box_liftable=box.get("liftable"), box_abs_q=box.get("abs_q"), passes=ok)


# ----------------------------------------------------------------------------- the controls C1 to C4, C9

def _seed(grid, d0):
    x = axis_of(grid)
    return pair_field(x, d0), x


def control_c1():
    rows = []
    ok = True
    want = {9: 4.5, 15: 7.5, 21: 10.5}
    for grid in ("P", "P375"):
        for d0, hv in want.items():
            m6, x = _seed(grid, d0)
            dp = dprime(m6, x)
            good = bool(abs(dp["Dp"] - hv) <= 0.01 and abs(dp["Dm"] - hv) <= 0.01)
            ok &= good
            rows.append(dict(input=f"seed d0 = {d0} on {grid}", Dp=dp["Dp"], Dm=dp["Dm"], passed=good))
    m30, x, rec30 = load56(30)
    s30 = rec30["end"]["pair"]["separation"]
    hold = far_field(x)
    d30 = dprime(m30, x)
    reg30 = dprime_regime(m30, x)
    good = bool(abs(d30["D"] - s30) <= 0.05)
    ok &= good
    rows.append(dict(input="row 30", Dp=d30["Dp"], Dm=d30["Dm"], D=d30["D"], s=s30, regime=reg30, W_per_half=[d30["Wp"], d30["Wm"]], passed=good))
    mx = dprime(moved(m30, 2, 0, hold), x)
    good = bool(abs((mx["Dp"] - d30["Dp"]) - 1) <= 0.005 and abs((mx["Dm"] - d30["Dm"]) + 1) <= 0.005)
    ok &= good
    rows.append(dict(input="row 30 moved by 2 cells along x", dDp=mx["Dp"] - d30["Dp"], dDm=mx["Dm"] - d30["Dm"], passed=good))
    my = dprime(moved(m30, 4, 1, hold), x)
    good = bool(abs(my["Dp"] - d30["Dp"]) <= 0.005 and abs(my["Dm"] - d30["Dm"]) <= 0.005)
    ok &= good
    rows.append(dict(input="row 30 moved by 4 cells along y", dDp=my["Dp"] - d30["Dp"], dDm=my["Dm"] - d30["Dm"], passed=good))
    un = dprime(m30, x, normalized=False)
    r03 = dprime_regime(m30, x, g0=0.3)
    # the plan-time failing reads of the share (descriptors of the regime)
    m30b = m30.copy()
    i17 = int(round((17.0 - x[0]) / (x[1] - x[0])))
    ic = (len(x) - 1) // 2
    m30b[i17, ic, ic] = A.to6((1 + DELTA) / 3 * np.eye(3))
    rb = dprime_regime(m30b, x)
    db = dprime(m30b, x)
    m22, x22, _ = load56(22)
    m22b = m22.copy()
    i6 = int(round((6.0 - x22[0]) / (x22[1] - x22[0])))
    ic22 = (len(x22) - 1) // 2
    m22b[i6, ic22, ic22] = A.to6((1 + DELTA) / 3 * np.eye(3))
    r22b = dprime_regime(m22b, x22)
    m33, _, _ = load56(33)
    r33 = dprime_regime(m33, x)
    d33 = dprime(m33, x)
    fails = [dict(input="the weight not normalized, sum |x| w on the + half", value=un["Dp"], fails_as_planned=bool(abs(un["Dp"] - d30["Dp"]) > 0.01)),
             dict(input="g0 = 0.3: the share clause", share=r03["share"], fails_as_planned=bool(not r03["clause_i"]))]
    desc = dict(row30_node_17_isotropic=dict(share=rb["share"], dDp=db["Dp"] - d30["Dp"]), row22_node_6_isotropic=dict(share=r22b["share"]),
                row33=dict(Dp=d33["Dp"], Dm=d33["Dm"], D=d33["D"], regime=r33))
    return dict(control="C1 D' known answers", passed=bool(ok), rows=rows, failing_inputs=fails, descriptors=desc)


def control_c2():
    m30, x, rec30 = load56(30)
    h = x[1] - x[0]
    X = T.mesh(x)
    rings = rec30["end"]["pair"]["rings"]
    dist = np.min([T.ring_distance(X, q["c"], q["nu"], q["RL"]) for q in rings], axis=0)
    inner = ~T.faces_mask(len(x))
    classes = dict(near=inner & (dist <= 0.5), mid=inner & (dist > 0.5) & (dist <= 1.5), far=inner & (dist > 5) & (X[..., 0] > 0))
    rng = np.random.default_rng(57002)
    dp = dprime(m30, x, grad=True)
    bad = dprime(m30, x, grad=True, drop_norm_term=True)
    G = dp["grad"]
    picks = []
    for nm, msk in classes.items():
        idx = np.argwhere(msk)
        for k in rng.choice(len(idx), 4, replace=False):
            picks.append((nm, tuple(idx[k])))
    rows = []
    errs, errs_bad, an = [], [], []
    for nm, p in picks:
        for comp in range(6):
            e = 1e-6
            zp = m30.copy()
            zp[p + (comp,)] += e
            zm = m30.copy()
            zm[p + (comp,)] -= e
            fd = (dprime(zp, x)["value"] - dprime(zm, x)["value"]) / (2 * e)
            a = float(G[p + (comp,)])
            ab = float(bad["grad"][p + (comp,)])
            errs.append(abs(fd - a))
            errs_bad.append(abs(fd - ab))
            an.append(abs(a))
            rows.append(dict(cls=nm, node=list(p), comp=comp, fd=fd, analytic=a, err=abs(fd - a)))
    big = max(an)
    ok = bool(max(errs) <= 1e-6 * big)
    return dict(control="C2 the gradient of D'", passed=ok, worst_error=max(errs), largest_analytic_entry=big, rows=rows,
                failing_input=dict(input="the derivative of the normalization left out", worst_error=max(errs_bad),
                                   fails_as_planned=bool(max(errs_bad) > 1e-6 * big)))


def control_c3():
    m30, x, rec30 = load56(30)
    hold = far_field(x)
    m1 = moved(m30, 1, 0, hold)
    s30 = rec30["end"]["pair"]["separation"]
    inner, plus, minus, X = halves(x)
    vals, grads, Dvals, nts = [], [], [], []
    path = []
    for k in range(41):
        t = k / 40
        f6 = (1 - t) * m30 + t * m1
        path.append(f6)
        d = dprime(f6, x, grad=True)
        vals.append(d["Dp"])
        grads.append(d["grad"] * plus[..., None])
        dm_ = dmoment(f6, x, s30, keep=True)
        Dvals.append(dm_["Dp"])
        nts.append(dm_.pop("_nt"))
    steps, trap, dsteps, dfixed = [], [], [], []
    for k in range(40):
        dm = path[k + 1] - path[k]
        steps.append(vals[k + 1] - vals[k])
        trap.append(float(0.5 * ((grads[k] + grads[k + 1]) * dm).sum()))
        dsteps.append(Dvals[k + 1] - Dvals[k])
        # D_+ with the lift held fixed across the step (the smooth part its gradient sees), forward and backward averaged
        fwd = dmoment(path[k + 1], x, s30, ref_vec=nts[k])["Dp"] - dmoment(path[k], x, s30, ref_vec=nts[k])["Dp"]
        bwd = dmoment(path[k + 1], x, s30, ref_vec=nts[k + 1])["Dp"] - dmoment(path[k], x, s30, ref_vec=nts[k + 1])["Dp"]
        dfixed.append(0.5 * (fwd + bwd))
    err = max(abs(a - b) for a, b in zip(steps, trap))
    tot = vals[-1] - vals[0]
    ok = bool(err <= 1e-4 and abs(tot - 0.5) <= 0.005)
    derr = max(abs(a - b) for a, b in zip(dsteps, dfixed))
    return dict(control="C3 a smooth hold", passed=ok, worst_step_error=err, total_change=tot, Dp_path=vals,
                failing_input=dict(input="D_+ on the same path, its steps against the steps with the lift held fixed", steps=dsteps,
                                   fixed_lift_steps=dfixed, worst_step_error=float(derr), total_change=float(Dvals[-1] - Dvals[0]),
                                   fails_as_planned=bool(derr > 1e-4)))


def control_c4():
    rows = []
    ok = True
    for grid in ("P", "P375"):
        for d0 in (9, 15, 21):
            m6, x = _seed(grid, d0)
            r = dmoment(m6, x, d0)
            good = bool(abs(r["D"] - d0) <= 0.1 and abs(r["Qp"] - 1) <= 0.002 and abs(r["Qm"] - 1) <= 0.002 and r["regime"]["flips_ok"])
            ok &= good
            rows.append(dict(input=f"seed d0 = {d0} on {grid}", **r, passed=good))
    for row in (30, 31):
        m6, x, rec = load56(row)
        s = rec["end"]["pair"]["separation"]
        r = dmoment(m6, x, s)
        good = bool(abs(r["D"] - s) <= 0.1 and abs(r["Qp"] - 1) <= 0.002 and abs(r["Qm"] - 1) <= 0.002 and r["in_regime"])
        ok &= good
        rows.append(dict(input=f"row {row}", s=s, **r, passed=good))
    m33, x, _ = load56(33)
    for dc in (0.0, 2.0, 7.7, 12.0):
        r = dmoment(m33, x, dc)
        good = bool(not r["in_regime"])
        ok &= good
        rows.append(dict(input=f"row 33, d_c = {dc}", **r, passed=good))
    m30, x, rec30 = load56(30)
    rf = dmoment(m30, x, rec30["end"]["pair"]["separation"], radial=True)
    fails = [dict(input="the lift radial about each center, no half-turn for the - half", Qm=rf["Qm"], flips=rf["regime"]["minus"]["flips"],
                  fails_as_planned=bool(not rf["in_regime"]))]
    return dict(control="C4 the degree read", passed=bool(ok), rows=rows, failing_inputs=fails)


def control_c9():
    c6 = T.control_c6()
    c17 = T.control_c17()
    ok = bool(c6["passed"] and c17["pair_passed"])
    return dict(control="C9 the two-ring reader (M5.56 C6 and C17 pair part, unchanged)", passed=ok, C6=dict(passed=c6["passed"], read=dict(
        word=c6["read"]["word"], components=c6["read"]["components"], RL=[q["RL"] for q in c6["read"]["rings"]],
        circle_distance=c6["read"].get("circle_distance")), failing_inputs=c6["failing_inputs"]), C17_pair=c17["pair"])


def control_seed_check():
    """Descriptor: this script's pair field against M5.56's pair_bi with its ladder (the same formula)."""
    x = axis_of("P")
    lad = ladder_d(x, 6.0, 14.0, T.RIN, T.ROUT)
    return float(np.abs(pair_field(x, lad) - T.pair_bi(x, 6.0)).max())


def write_control(name, res, tryn):
    CK.mkdir(parents=True, exist_ok=True)
    f = CK / f"control_{name}_try{tryn}.json"
    f.write_text(json.dumps(res, indent=1, default=T._jsonable) + "\n")
    print(f"{res['control']}: {'MET' if res['passed'] else 'NOT MET'} -> {f.name}", flush=True)
    return res["passed"]


def selftest(tryn, only=None):
    jobs = [("C1", control_c1), ("C2", control_c2), ("C3", control_c3), ("C4", control_c4), ("C9", control_c9)]
    out = {}
    for nm, fn in jobs:
        if only and nm not in only:
            continue
        t0 = time.time()
        res = fn()
        res["seconds"] = time.time() - t0
        if nm == "C1":
            res["descriptors"]["pair_field_vs_m5_56_pair_bi"] = control_seed_check()
        out[nm] = write_control(nm, res, tryn)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--try", dest="tryn", type=int, default=1)
    a = ap.parse_args()
    if not a.selftest:
        ap.print_help()
        return
    r = selftest(a.tryn, a.only)
    sys.exit(0 if all(r.values()) else 1)


if __name__ == "__main__":
    main()
