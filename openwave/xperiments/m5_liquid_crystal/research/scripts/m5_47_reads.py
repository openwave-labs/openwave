"""M5.47: the reads of the wave jobs. Controls K6 to K12, the four frozen readings, the tables and the plots.

The jobs are those of m5_47_waves.py; the action and its time part are those of m5_47_evolve.py.

THE VACUUM (jobs V1, V2, V3; no label)
    K6   every one of the twenty sine modes, projected on the six eigen-directions e of the Hessian of the coded
         potential at the vacuum (H e = lam g e, g the metric of the six components): one frequency per
         direction from x(t + tau) + x(t - tau) = c x(t), c = 2 cos(w tau), tau = 0.01 for the heavy
         direction and 0.2 for the others (one step and twenty steps at the step 0.01). PASS: w^2 within 1 % of khat^2 + v lam / (2 L1).
    K7   the estimator below with W on the nodes at least 1.5 inside the faces, targets 8 to 11.
         PASS: w^2 - K^2 within 1.0 of the mid mass at every fit point.

THE UNIT (jobs U, N, H, D, S on the base states B0a to B0d of M5.46)
    a(x, w) = <W(x), A(x, w)>,   A the six-component Fourier sum of a job minus that of its unperturbed twin,
    W(x) = w1 n n^T + w2 (1 - n n^T)/sqrt 2 the mid polarization (m5_47_evolve.polarization).

    K^2(B, w)      = - sum_B Re(conj(a) Lap_h a) / sum_B |a|^2            Lap_h the seven-point Laplacian
    rho(B, w)      = sum_B |rhat . grad a|^2 / sum_B |grad a|^2            central differences; K_r^2 = rho K^2
    w_act^2(B)     = sum_B [ L1 |grad a|^2 + kappa (sum_j |C_j|^2 |grad a|^2 - |sum_j C_j d_j a|^2)
                             + v V''_WW |a|^2 / 2 ] / sum_B L1 q |a|^2,    C_j = [W, d_j M0]
    q(x)           = 1 + (kappa / L1) sum_j ( |[W, d_j^+ M0]|^2 + |[W, d_j^- M0]|^2 ) / 2

    The read region: nodes at least 1.5 from the loop line and at least the loop's radius from its center
    (center, radius, normal from m5_44_loops.read_state), at least 1.5 inside every face, the two largest
    eigenvalues of the background at least 0.5 apart. Bins of q with the edges EDGES; a bin under 500 nodes is
    not read. A fit point is a pair (bin, target) with 1 <= K^2 <= 4 / h^2.

    K8   in the bin of every fit point, the Fourier power of the twin at most 1 % of that of the difference
         (six components; the same ratio in W alone is tabled beside it)
    K9   |a(U) - 2 a(H)| over the bin at most 2 % of |a(U)|, at every fit point of U
    K10  K^2 of Da against Ua: at most 2 % apart at every fit point of Ua
    K11  max |E(t) - E(0)| at most 2 % of E(0) - U(M0), every perturbed job
    K12  sum_B Re<conj(A), H A> / sum_B Re<conj(A), h^3 A(M0) A> within 3 % of w^2 at every fit point,
         H A the central difference of the coded gradient about M0

    R1   one law: with m^2 the median of w^2 - K^2 over all fit points, |w^2 - K^2 - m^2| <= 0.05 w^2 at each
    R2   a local law: at least three bins fitted (four fit points at least); in each, every fit point within
         0.05 w^2 of the bin's plane w^2 = cr^2 Kr^2 + ct^2 Kt^2 + m^2 (six points at least) or of its line
    R3   |w_act^2 - w^2| <= 0.1 w^2 at every fit point
    R4   pairs (Ua, Ub), (Uc, Ud), (Ub, Ud): a bin is compared when both grids read it and four targets at
         least are fit points on both; the line is refitted on the shared targets; HELD when three bins at
         least are compared and in each c^2 agrees within 0.05 and m^2 within 10 % (of the mean of the two)

    What withholds a label: K6, K7, K9 or K10 failed (every grid); K8, K11 or K12 failed (that grid); the
    scatter rule on the second draws (Sa, Sb). R1 to R3 are labeled on the 0.25 grids. Nothing here sets an icon.

Usage:  python m5_47_reads.py --compute      (writes data/m5_47_reads.json)
        python m5_47_reads.py --tables       (writes the markdown tables)
        python m5_47_reads.py --plots
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import m5_42_eq42 as Q
import m5_44_loops as L44
import m5_47_evolve as E
import m5_47_waves as WV

EDGES = np.array([1.25, 1.5, 1.75, 2.0, 2.25, 2.75, 3.25, 3.75, 4.25, 5.0, 6.0, 7.0])
NB = len(EDGES) - 1
MIN_NODES = 500
SHELL_W = 0.5
INSET = 1.5
GRIDS = dict(a=("B0a", "Ua", "Na"), b=("B0b", "Ub", "Nb"), c=("B0c", "Uc", "Nc"), d=("B0d", "Ud", "Nd"))
EXTRA = dict(Ha="a", Hb="b", Da="a", Sa="a", Sb="b")
PAIRS = (("a", "b", "two spacings, half-box 8"), ("c", "d", "two spacings, half-box 10"), ("b", "d", "two boxes, spacing 0.25"))
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#d9d8d3"


def comm2(A, B):
    C = A @ B - B @ A
    return (C * C).sum((-1, -2))


def geometry(base, data):
    """Everything the read needs from a base state."""
    m0, hb, N = E.load_base(base, data)
    x = np.linspace(-hb, hb, N)
    h = x[1] - x[0]
    W, lam, n = E.polarization(m0)
    w = E.mid_weights()[0]
    P = np.einsum("...i,...j->...ij", n, n)
    Wh6 = Q.to6(-w[1] * P + w[0] * (np.eye(3) - P) / np.sqrt(2.0))
    M = Q.from6(m0)
    loop = L44.read_state(m0, x, second=False)["loop"]
    c, nu, R = np.array(loop["center"]), np.array(loop["normal"]), loop["radius"]
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    d = X - c
    r = np.sqrt((d * d).sum(-1))
    zz = d @ nu
    rho = np.sqrt(((d - zz[..., None] * nu) ** 2).sum(-1))
    dist = np.sqrt((rho - R) ** 2 + zz ** 2)
    rhat = d / (r[..., None] + 1e-12)
    inside = (np.abs(X) <= hb - INSET + 1e-9).all(-1)
    gap = lam[..., 2] - lam[..., 1]
    region = (dist >= INSET) & (r >= R) & inside & (gap >= 0.5)
    s = np.zeros((N, N, N))
    Cc = np.zeros((3, N, N, N, 3, 3))
    Cs = np.zeros((3, 2, N, N, N, 3, 3))
    for j in range(3):
        Dp = (np.roll(M, -1, axis=j) - M) / h
        Dm = (np.roll(M, 1, axis=j) - M) / h
        s += 0.5 * (comm2(W, Dp) + comm2(W, Dm))
        Cs[j, 0] = W @ Dp - Dp @ W
        Cs[j, 1] = W @ Dm - Dm @ W
        Dc = (np.roll(M, -1, axis=j) - np.roll(M, 1, axis=j)) / (2 * h)
        Cc[j] = W @ Dc - Dc @ W
    q = 1 + E.KAPPA * s / E.L1
    sumC2 = (Cc * Cc).sum((0, -1, -2))
    Cr = np.einsum("xyzj,jxyzab->xyzab", rhat, Cc)
    Cr2 = (Cr * Cr).sum((-1, -2))
    cr2 = 1 - E.KAPPA * Cr2 / (E.L1 * q)
    ct2 = 1 - 0.5 * E.KAPPA * (sumC2 - Cr2) / (E.L1 * q)
    eps = 1e-4
    Vww = (E.potential(M + eps * W) - 2 * E.potential(M) + E.potential(M - eps * W)) / eps ** 2
    m2loc = E.V * Vww / (2 * E.L1 * q)
    idx = np.digitize(q, EDGES) - 1
    F = E.Field(N, hb)
    W6 = Q.to6(W)
    A = F.a_matrices(m0)
    qA = np.einsum("...a,...ab,...b->...", W6, A, W6) / (2 * E.L1)
    return dict(base=base, m0=m0, hb=hb, N=N, h=h, x=x, W=W, W6=W6, Wh6=Wh6, lam=lam, n=n, loop=loop, r=r, rhat=rhat, dist=dist,
                region=region, q=q, q_from_A=float(np.abs(qA - q)[region].max()), Cc=Cc, Cs=Cs, sumC2=sumC2, cr2=cr2, ct2=ct2,
                Vww=Vww, m2loc=m2loc, idx=idx, sidx=np.floor(r / SHELL_W).astype(int), F=F, center=c, normal=nu, radius=R)


def load_A(job, data):
    z = np.load(data / f"m5_47_{job}_fourier.npz", allow_pickle=False)
    return z["Ar"] + 1j * z["Ai"], z["targets"]


def lap(a, h):
    out = -6.0 * a
    for j in range(3):
        out = out + np.roll(a, -1, axis=j) + np.roll(a, 1, axis=j)
    return out / (h * h)


def grad(a, h):
    return np.stack([(np.roll(a, -1, axis=j) - np.roll(a, 1, axis=j)) / (2 * h) for j in range(3)])


def grad_sided(a, h):
    """The one-sided differences of the action's stencil: [axis, 0] forward, [axis, 1] backward."""
    return np.stack([np.stack([(np.roll(a, -1, axis=j) - a) / h, (np.roll(a, 1, axis=j) - a) / h]) for j in range(3)])


def sums(field, sel, index, size):
    return np.bincount(index[sel], weights=field[sel], minlength=size)[:size]


def read_job(job, twin, G, data, k12=True):
    """Per bin of q and per shell of r, per target: the sums the reads are made of."""
    A, targets = load_A(job, data)
    An, _ = load_A(twin, data)
    dA = A - An
    del A
    h, reg, F, m0 = G["h"], G["region"], G["F"], G["m0"]
    g6 = G["W6"] * E.METRIC
    a = np.einsum("xyzc,xyzct->xyzt", g6, dA)
    aN = np.einsum("xyzc,xyzct->xyzt", g6, An)
    aH = np.einsum("xyzc,xyzct->xyzt", G["Wh6"] * E.METRIC, dA)
    nt = len(targets)
    ns = int(G["sidx"][reg].max()) + 1
    selb = reg & (G["idx"] >= 0) & (G["idx"] < NB)
    names = ("e0", "e1", "g2", "gr2", "num", "num_os", "den", "pN", "pH", "p6", "p6N", "k12n", "k12d")
    B = {k: np.zeros((NB, nt)) for k in names}
    S = {k: np.zeros((ns, nt)) for k in names}
    f1, f2, p1 = np.zeros(F.shape6), np.zeros(F.shape6), np.zeros(F.shape6)
    for t in range(nt):
        at = a[..., t]
        e0 = np.abs(at) ** 2
        gr = grad(at, h)
        g2 = (np.abs(gr) ** 2).sum(0)
        gr2 = np.abs(np.einsum("xyzj,jxyz->xyz", G["rhat"], gr)) ** 2
        Sm = np.einsum("jxyz,jxyzab->xyzab", gr, G["Cc"])
        num = E.L1 * g2 + E.KAPPA * (G["sumC2"] * g2 - (np.abs(Sm) ** 2).sum((-1, -2))) + 0.5 * E.V * G["Vww"] * e0
        gs = grad_sided(at, h)
        num_os = E.L1 * 0.5 * (np.abs(gs) ** 2).sum((0, 1)) + 0.5 * E.V * G["Vww"] * e0
        for i in range(3):
            for j in range(i + 1, 3):
                for si in range(2):
                    for sj in range(2):
                        X = gs[i, si][..., None, None] * G["Cs"][j, sj] - gs[j, sj][..., None, None] * G["Cs"][i, si]
                        num_os = num_os + 0.25 * E.KAPPA * (np.abs(X) ** 2).sum((-1, -2))
        dens = dict(e0=e0, e1=-(np.conj(at) * lap(at, h)).real, g2=g2, gr2=gr2, num=num, num_os=num_os, den=E.L1 * G["q"] * e0,
                    pN=np.abs(aN[..., t]) ** 2, pH=np.abs(aH[..., t]) ** 2, p6=(np.abs(dA[..., t]) ** 2 * E.METRIC).sum(-1),
                    p6N=(np.abs(An[..., t]) ** 2 * E.METRIC).sum(-1))
        if k12:
            kn = np.zeros(reg.shape)
            kd = np.zeros(reg.shape)
            for part in (np.ascontiguousarray(dA[..., t].real), np.ascontiguousarray(dA[..., t].imag)):
                eps = 1e-4 / max(np.abs(part).max(), 1e-300)
                F.static(m0 + eps * part, f1)
                F.static(m0 - eps * part, f2)
                HA = -(f1 - f2) / (2 * eps)
                F.momentum(m0, part, p1)
                kn += (part * HA).sum(-1)
                kd += (part * p1).sum(-1)
            dens.update(k12n=kn, k12d=kd)
        for k, v in dens.items():
            B[k][:, t] = sums(v, selb, G["idx"], NB)
            S[k][:, t] = sums(v, reg, G["sidx"], ns)

    def derive(D, counts):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = dict(nodes=counts.tolist(), K2=D["e1"] / D["e0"], rho=D["gr2"] / D["g2"], w2_act=D["num"] / D["den"], w2_act_os=D["num_os"] / D["den"],
                       twin_share=D["p6N"] / D["p6"], twin_share_W=D["pN"] / D["e0"], outside_W=1 - D["e0"] / D["p6"], heavy_share=D["pH"] / D["p6"],
                       power=D["e0"], k12=(D["k12n"] / D["k12d"]) if k12 else np.full_like(D["e0"], np.nan))
        return out

    nodes_b = np.bincount(G["idx"][selb], minlength=NB)[:NB]
    nodes_s = np.bincount(G["sidx"][reg], minlength=ns)[:ns]
    out = dict(job=job, twin=twin, targets=targets, bins=derive(B, nodes_b), shells=derive(S, nodes_s), a=a)
    out["fit"] = (nodes_b[:, None] >= MIN_NODES) & (out["bins"]["K2"] >= 1.0) & (out["bins"]["K2"] <= 4.0 / h ** 2)
    out["fit_shells"] = (nodes_s[:, None] >= MIN_NODES) & (out["shells"]["K2"] >= 1.0) & (out["shells"]["K2"] <= 4.0 / h ** 2)
    return out


def fit_line(K2, w2):
    c2, m2 = np.polyfit(K2, w2, 1)
    return float(c2), float(m2), c2 * K2 + m2


def fit_plane(Kr2, Kt2, w2):
    X = np.stack([Kr2, Kt2, np.ones_like(Kr2)], 1)
    co, *_ = np.linalg.lstsq(X, w2, rcond=None)
    return float(co[0]), float(co[1]), float(co[2]), X @ co


def bin_fits(R, G):
    """Per bin: the line, the plane, the residuals and the descriptors."""
    w2 = R["targets"] ** 2
    out = []
    for b in range(NB):
        sel = R["fit"][b]
        rec = dict(bin=b, q_lo=float(EDGES[b]), q_hi=float(EDGES[b + 1]), nodes=int(R["bins"]["nodes"][b]), points=int(sel.sum()),
                   read=bool(R["bins"]["nodes"][b] >= MIN_NODES), fitted=False)
        m = G["region"] & (G["idx"] == b)
        if rec["read"]:
            rec.update(q_mean=float(G["q"][m].mean()), r_median=float(np.median(G["r"][m])),
                       formula=dict(cr2=float(G["cr2"][m].mean()), ct2=float(G["ct2"][m].mean()), m2_median=float(np.median(G["m2loc"][m])),
                                    m2_mean=float(G["m2loc"][m].mean())))
        if sel.sum() >= 4:
            K2, rho = R["bins"]["K2"][b, sel], R["bins"]["rho"][b, sel]
            c2, m2, pred = fit_line(K2, w2[sel])
            rec.update(fitted=True, targets=R["targets"][sel].tolist(), c2=c2, m2=m2, m2_q=m2 * rec["q_mean"],
                       line_max_dev=float(np.abs(pred - w2[sel]).max() / 1.0), line_max_rel=float((np.abs(pred - w2[sel]) / w2[sel]).max()),
                       c2_within_005_of_1=bool(abs(c2 - 1) <= 0.05), rho_mean=float(rho.mean()))
            rec["judged_by"], rec["max_rel"] = "line", rec["line_max_rel"]
            if sel.sum() >= 6:
                cr2, ct2, mp2, predp = fit_plane(rho * K2, (1 - rho) * K2, w2[sel])
                rec.update(plane=dict(cr2=cr2, ct2=ct2, m2=mp2, max_rel=float((np.abs(predp - w2[sel]) / w2[sel]).max()),
                                      cond=float(np.linalg.cond(np.stack([rho * K2, (1 - rho) * K2, np.ones_like(K2)], 1)))))
                rec["judged_by"], rec["max_rel"] = "plane", rec["plane"]["max_rel"]
        out.append(rec)
    return out


def reading_r1(R):
    w2 = np.broadcast_to(R["targets"] ** 2, R["fit"].shape)
    sel = R["fit"]
    if sel.sum() == 0:
        return dict(points=0, held=None)
    res = (w2 - R["bins"]["K2"])[sel]
    m2 = float(np.median(res))
    rel = np.abs(res - m2) / w2[sel]
    return dict(points=int(sel.sum()), m2_median=m2, max_rel=float(rel.max()), outside=int((rel > 0.05).sum()), held=bool((rel <= 0.05).all()),
                residual_min=float(res.min()), residual_max=float(res.max()))


def reading_r2(fits):
    fb = [f for f in fits if f["fitted"]]
    if len(fb) < 3:
        return dict(fitted_bins=len(fb), held=None)
    return dict(fitted_bins=len(fb), max_rel=max(f["max_rel"] for f in fb), bins_outside=[f["bin"] for f in fb if f["max_rel"] > 0.05],
                held=bool(all(f["max_rel"] <= 0.05 for f in fb)))


def restricted(R, limit=0.5):
    """The same read with the fit points kept only where the power outside W is under the limit (added)."""
    out = dict(R)
    out["fit"] = R["fit"] & (R["bins"]["outside_W"] < limit)
    return out


def mid_branch(R):
    """The same read with the fit points kept only at the targets above the bin's target of least K^2 (added)."""
    out = dict(R)
    K2 = np.where(np.isfinite(R["bins"]["K2"]), R["bins"]["K2"], np.inf)
    cut = np.argmin(K2, axis=1)
    out["fit"] = R["fit"] & (np.arange(K2.shape[1])[None, :] > cut[:, None])
    out["cutoff_target"] = R["targets"][cut]
    return out


def reading_r3(R, key="w2_act"):
    w2 = np.broadcast_to(R["targets"] ** 2, R["fit"].shape)
    sel = R["fit"]
    if sel.sum() == 0:
        return dict(points=0, held=None)
    rel = (R["bins"][key] - w2)[sel] / w2[sel]
    return dict(points=int(sel.sum()), max_rel=float(np.abs(rel).max()), min=float(rel.min()), max=float(rel.max()), median=float(np.median(rel)),
                outside=int((np.abs(rel) > 0.10).sum()), held=bool((np.abs(rel) <= 0.10).all()))


def compare(R1, R2):
    """The bins compared between two reads: the line refitted on the targets that are fit points on both."""
    w2 = R1["targets"] ** 2
    rows = []
    for b in range(NB):
        both = R1["fit"][b] & R2["fit"][b]
        if both.sum() >= 4:
            c1, m1, _ = fit_line(R1["bins"]["K2"][b, both], w2[both])
            c2, m2, _ = fit_line(R2["bins"]["K2"][b, both], w2[both])
            rows.append(dict(bin=b, shared=int(both.sum()), c2=(c1, c2), m2=(m1, m2), dc2=abs(c1 - c2),
                             dm2_rel=abs(m1 - m2) / (0.5 * abs(m1 + m2))))
    return rows


def reading_r4(R1, R2):
    rows = compare(R1, R2)
    if len(rows) < 3:
        return dict(compared=len(rows), rows=rows, held=None)
    return dict(compared=len(rows), rows=rows, max_dc2=max(r["dc2"] for r in rows), max_dm2_rel=max(r["dm2_rel"] for r in rows),
                held=bool(all(r["dc2"] <= 0.05 and r["dm2_rel"] <= 0.10 for r in rows)))


def scatter(RU, RS, fitsU, fitsS):
    w2 = np.broadcast_to(RU["targets"] ** 2, RU["fit"].shape)
    both = RU["fit"] & RS["fit"]
    out = dict(common_points=int(both.sum()))
    if both.sum():
        out["median_dK2_over_w2"] = float(np.median(np.abs(RU["bins"]["K2"] - RS["bins"]["K2"])[both] / w2[both]))
        out["median_dw2act_over_w2"] = float(np.median(np.abs(RU["bins"]["w2_act"] - RS["bins"]["w2_act"])[both] / w2[both]))
    pairs = [(u, s) for u, s in zip(fitsU, fitsS) if u["fitted"] and s["fitted"]]
    out["bins_fitted_in_both"] = len(pairs)
    if pairs:
        out["median_dc2"] = float(np.median([abs(u["c2"] - s["c2"]) for u, s in pairs]))
        out["median_dm2_rel"] = float(np.median([abs(u["m2"] - s["m2"]) / (0.5 * abs(u["m2"] + s["m2"])) for u, s in pairs]))
    return out


# ----------------------------------------------------------------------------- the vacuum

def khat2(mode, N, h):
    return float((4 / h ** 2) * sum(np.sin(n * np.pi / (2 * (N - 1))) ** 2 for n in mode))


def read_modes(job, data):
    spec = json.loads((data / f"m5_47_{job}.json").read_text())
    z = np.load(data / f"m5_47_{job}_modes.npz", allow_pickle=False)
    series, dt, M0 = z["series"], float(z["dt"]), z["m0"]
    N, h, v = spec["spec"]["N"], spec["spacing"], spec["spec"]["v"]
    H, lam, vec = E.vacuum_hessian(M0)
    masses = v * lam / (2 * E.L1)
    x = series @ (np.diag(E.METRIC) @ vec)
    rows, worst = [], np.zeros(6)
    for mi, mode in enumerate(WV.MODES):
        k2 = khat2(mode, N, h)
        rec = dict(mode=list(mode), khat2=k2, w2=[], expected=[], rel=[])
        for l in range(6):
            tau = max(1, int(round((0.01 if l == 5 else 0.2) / dt)))
            y = x[:, mi, l]
            c = float((y[tau:-tau] * (y[2 * tau:] + y[:-2 * tau])).sum() / (y[tau:-tau] ** 2).sum())
            w = np.arccos(np.clip(c / 2, -1, 1)) / (tau * dt)
            ex = k2 + masses[l]
            rec["w2"].append(float(w * w))
            rec["expected"].append(float(ex))
            rec["rel"].append(float(abs(w * w - ex) / ex))
            worst[l] = max(worst[l], rec["rel"][-1])
        rows.append(rec)
    read_mass = [float(np.median([r["w2"][l] - r["khat2"] for r in rows])) for l in range(6)]
    return dict(job=job, v=v, masses=masses.tolist(), read_masses=read_mass, worst_rel_per_direction=worst.tolist(),
                worst_rel=float(worst.max()), passed=bool(worst.max() <= 0.01), modes=rows)


def read_vacuum_estimator(job, data):
    spec = json.loads((data / f"m5_47_{job}.json").read_text())
    A, targets = load_A(job, data)
    N, h, hb = spec["spec"]["N"], spec["spacing"], spec["spec"]["hb"]
    z = np.load(data / f"m5_47_{job}_modes.npz", allow_pickle=False)
    M0 = z["m0"]
    W = E.polarization(Q.to6(M0))[0]
    mid = spec["spec"]["v"] * E.vacuum_hessian(M0)[1][4] / (2 * E.L1)
    a = np.einsum("c,xyzct->xyzt", Q.to6(W) * E.METRIC, A)
    x = np.linspace(-hb, hb, N)
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    sel = (np.abs(X) <= hb - INSET + 1e-9).all(-1)
    rows = []
    for t, w in enumerate(targets):
        at = a[..., t]
        K2 = float(-(np.conj(at) * lap(at, h)).real[sel].sum() / (np.abs(at) ** 2)[sel].sum())
        p6 = float(((np.abs(A[..., t]) ** 2 * E.METRIC).sum(-1))[sel].sum())
        rows.append(dict(target=float(w), K2=K2, fit=bool(1.0 <= K2 <= 4 / h ** 2), w2_minus_K2=float(w * w - K2),
                         outside_W=float(1 - (np.abs(at) ** 2)[sel].sum() / p6), power=float((np.abs(at) ** 2)[sel].sum())))
    used = [r for r in rows if r["fit"] and r["target"] >= 8.0]
    dev = [abs(r["w2_minus_K2"] - mid) for r in used]
    return dict(job=job, mid_mass=float(mid), nodes=int(sel.sum()), rows=rows, fit_points=len(used), worst=float(max(dev)) if dev else None,
                passed=bool(dev and max(dev) <= 1.0))


# ----------------------------------------------------------------------------- compute

def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(f"{float(o):.7g}")
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def control_at_fit(R, key, limit, center=None):
    """The worst value of a per-(bin, target) quantity over the fit points, and whether it stays inside."""
    sel = R["fit"]
    if sel.sum() == 0:
        return dict(points=0, worst=None, passed=None)
    v = R["bins"][key][sel]
    if center is not None:
        v = np.abs(v / center[sel] - 1)
    return dict(points=int(sel.sum()), worst=float(np.nanmax(v)), median=float(np.nanmedian(v)), passed=bool(np.nanmax(v) <= limit))


def compute(a):
    data = a.data
    res = dict(edges=EDGES.tolist(), min_nodes=MIN_NODES)
    full = set(a.grids) == set(GRIDS)
    GR = {g: v for g, v in GRIDS.items() if g in a.grids}
    EX = {j: g for j, g in EXTRA.items() if g in a.grids}
    VJ = [j for j in ("V1", "V2", "V3") if (data / f"m5_47_{j}.json").exists()]
    # the vacuum
    res["K6"] = {j: read_modes(j, data) for j in VJ}
    res["K7"] = {j: read_vacuum_estimator(j, data) for j in VJ}
    k6 = all(res["K6"][j]["passed"] for j in VJ)
    if "V3" in res["K7"]:
        # K7 is read on V1 and V2: at v = 2 the mid mass is above every target
        res["K7"]["V3"].update(passed=None, applies=False)
    k7 = all(res["K7"][j]["passed"] for j in VJ if j != "V3")
    print("K6", {j: (res["K6"][j]["passed"], round(res["K6"][j]["worst_rel"], 5)) for j in res["K6"]}, flush=True)
    print("K7", {j: (res["K7"][j]["passed"], res["K7"][j]["worst"], res["K7"][j]["fit_points"]) for j in res["K7"]}, flush=True)
    # the unit
    jobs = {j: json.loads((data / f"m5_47_{j}.json").read_text()) for j in WV.JOBS if (data / f"m5_47_{j}.json").exists()}
    res["K11"] = {j: dict(drift=s["drift"], perturbation_energy=s["perturbation_energy"], ratio=s["drift"] / s["perturbation_energy"],
                          passed=bool(s["drift"] / s["perturbation_energy"] <= 0.02))
                  for j, s in jobs.items() if s["perturbation_energy"] > 0}
    res["twins"] = {j: dict(drift=s["drift"], U_base=s["U_base"], gmax_base_over_h3=s["gmax_base_over_h3"],
                            max_displacement=max(s["energy"]["max_displacement"])) for j, s in jobs.items() if j.startswith("N")}
    res["grids"], reads, fits, geo = {}, {}, {}, {}
    for g, (base, U, Nn) in GR.items():
        G = geometry(base, data)
        geo[g] = G
        reg = G["region"]
        far = reg & (G["r"] > 5)
        info = dict(base=base, N=G["N"], spacing=G["h"], half_box=G["hb"], loop=dict(center=G["center"].tolist(), normal=G["normal"].tolist(), radius=G["radius"]),
                    region_nodes=int(reg.sum()), q_min=float(G["q"][reg].min()), q_max=float(G["q"][reg].max()),
                    q_tail=float(np.median(((G["q"] - 1) * G["r"] ** 2)[far])), q_from_A_max_diff=G["q_from_A"],
                    nodes_below_first_edge=int((reg & (G["idx"] < 0)).sum()), nodes_above_last_edge=int((reg & (G["idx"] >= NB)).sum()))
        R = read_job(U, Nn, G, data)
        reads[U] = R
        fits[U] = bin_fits(R, G)
        w2 = np.broadcast_to(R["targets"] ** 2, R["fit"].shape)
        info.update(job=U, twin=Nn, targets=R["targets"], bins=R["bins"], fit=R["fit"], shells=R["shells"], fit_shells=R["fit_shells"], fits=fits[U],
                    K8=control_at_fit(R, "twin_share", 0.01), K8_in_W=control_at_fit(R, "twin_share_W", 0.01),
                    K12=control_at_fit(R, "k12", 0.03, center=w2), R1=reading_r1(R), R2=reading_r2(fits[U]), R3=reading_r3(R))
        res["grids"][g] = info
        print(g, "fit points", int(R["fit"].sum()), "K8", info["K8"], "K12", info["K12"], flush=True)
        print("   R1", info["R1"], "\n   R2", info["R2"], "\n   R3", info["R3"], flush=True)
    res["extra"] = {}
    for j, g in EX.items():
        G = geo[g]
        R = read_job(j, GRIDS[g][2], G, data, k12=False)
        reads[j] = R
        fits[j] = bin_fits(R, G)
        res["extra"][j] = dict(grid=g, bins=R["bins"], fit=R["fit"], fits=fits[j])
    # K9, K10
    res["K9"] = {}
    for H, U, g in (("Ha", "Ua", "a"), ("Hb", "Ub", "b")):
        if g not in GR:
            continue
        G, RU, RH = geo[g], reads[U], reads[H]
        selb = G["region"] & (G["idx"] >= 0) & (G["idx"] < NB)
        ratio = np.zeros(RU["fit"].shape)
        for t in range(len(RU["targets"])):
            num = sums(np.abs(RU["a"][..., t] - 2 * RH["a"][..., t]) ** 2, selb, G["idx"], NB)
            den = sums(np.abs(RU["a"][..., t]) ** 2, selb, G["idx"], NB)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio[:, t] = np.sqrt(num / den)
        v = ratio[RU["fit"]]
        vw = ratio[restricted(RU)["fit"]]
        res["K9"][H] = dict(points=int(RU["fit"].sum()), worst=float(v.max()) if v.size else None, median=float(np.median(v)) if v.size else None,
                            passed=bool(v.size and v.max() <= 0.02), ratio=ratio,
                            in_W=dict(points=int(vw.size), worst=float(vw.max()) if vw.size else None, median=float(np.median(vw)) if vw.size else None))
    if "a" in GR:
        RU, RD = reads["Ua"], reads["Da"]
        v = (np.abs(RD["bins"]["K2"] - RU["bins"]["K2"]) / RU["bins"]["K2"])[RU["fit"]]
        res["K10"] = dict(points=int(RU["fit"].sum()), worst=float(v.max()) if v.size else None, median=float(np.median(v)) if v.size else None,
                          passed=bool(v.size and v.max() <= 0.02))
        vw = (np.abs(RD["bins"]["K2"] - RU["bins"]["K2"]) / RU["bins"]["K2"])[restricted(RU)["fit"]]
        res["K10"]["in_W"] = dict(points=int(vw.size), worst=float(vw.max()) if vw.size else None, median=float(np.median(vw)) if vw.size else None)
    print("K9", {k: (v["passed"], v["worst"], v["in_W"]) for k, v in res["K9"].items()}, "K10", res.get("K10"), flush=True)
    # the scatter rule
    res["scatter"] = {g: scatter(reads["U" + g], reads["S" + g], fits["U" + g], fits["S" + g]) for g in ("a", "b") if g in GR}
    print("scatter", res["scatter"], flush=True)
    if not full:
        res["added"] = {g: dict(R1_like=reading_r1(restricted(reads[GR[g][1]])), R2_like=reading_r2(bin_fits(restricted(reads[GR[g][1]]), geo[g])),
                                R3_like=reading_r3(restricted(reads[GR[g][1]])), R3_like_one_sided=reading_r3(restricted(reads[GR[g][1]]), "w2_act_os"),
                                fits=bin_fits(restricted(reads[GR[g][1]]), geo[g])) for g in GR}
        for g in GR:
            print(g, "added", {k: v for k, v in res["added"][g].items() if k != "fits"}, flush=True)
            for f in res["added"][g]["fits"]:
                if f["fitted"]:
                    print("    q", f["q_lo"], f["q_hi"], "pts", f["points"], "c2", round(f["c2"], 3), "m2", round(f["m2"], 2), "formula", round(f["formula"]["m2_median"], 2),
                          "plane", {k: round(v, 3) for k, v in f.get("plane", {}).items()}, "gap", round(f["max_rel"], 4))
        (data.parent / "checkpoints" / "m5_47" / "reads_partial.json").write_text(json.dumps(jsonable(res)) + "\n")
        print("partial read written")
        return
    sb, sa = res["scatter"]["b"], res["scatter"]["a"]
    # R4
    res["R4"] = {}
    for g1, g2, what in PAIRS:
        r = reading_r4(reads[GRIDS[g1][1]], reads[GRIDS[g2][1]])
        r["what"] = what
        res["R4"][g1 + g2] = r
    # added after the first reads (no label): the same reads on two subsets of the fit points, and the quotient on the
    # action's own one-sided differences. "added": the power outside W under 0.5 (defined after the first read of
    # grid a). "added_cut": the targets above the bin's target of least K^2 (defined after the reads of a, b, c).
    for key, pick in (("added", restricted), ("added_cut", mid_branch)):
        res[key] = dict(grids={}, pairs={})
        for g, (base, U, Nn) in GRIDS.items():
            Rw = pick(reads[U])
            fw = bin_fits(Rw, geo[g])
            res[key]["grids"][g] = dict(fit_sel=Rw["fit"], points=int(Rw["fit"].sum()), fits=fw, R1_like=reading_r1(Rw), R2_like=reading_r2(fw),
                                        R3_like=reading_r3(Rw), R3_like_one_sided=reading_r3(Rw, "w2_act_os"),
                                        cutoff_target=Rw.get("cutoff_target"),
                                        outside_W_at_fit_points=dict(over_090=int((reads[U]["fit"] & (reads[U]["bins"]["outside_W"] > 0.9)).sum()),
                                                                     under_050=int((reads[U]["fit"] & (reads[U]["bins"]["outside_W"] < 0.5)).sum()),
                                                                     all=int(reads[U]["fit"].sum())),
                                        outside_W_on_selection=dict(least=float(reads[U]["bins"]["outside_W"][Rw["fit"]].min()) if Rw["fit"].any() else None,
                                                                    largest=float(reads[U]["bins"]["outside_W"][Rw["fit"]].max()) if Rw["fit"].any() else None))
            x = res[key]["grids"][g]
            print(g, key, "points", x["points"], "\n   R1-like", x["R1_like"], "\n   R2-like", x["R2_like"], "\n   R3-like", x["R3_like"],
                  "\n   R3-like one-sided", x["R3_like_one_sided"], flush=True)
        for g1, g2, what in PAIRS:
            r = reading_r4(pick(reads[GRIDS[g1][1]]), pick(reads[GRIDS[g2][1]]))
            r["what"] = what
            res[key]["pairs"][g1 + g2] = r
        res[key]["scatter"] = {g: scatter(pick(reads["U" + g]), pick(reads["S" + g]), bin_fits(pick(reads["U" + g]), geo[g]), bin_fits(pick(reads["S" + g]), geo[g]))
                               for g in ("a", "b")}
    for H, U in (("Ha", "Ua"), ("Hb", "Ub")):
        vm = res["K9"][H]["ratio"][mid_branch(reads[U])["fit"]]
        res["K9"][H]["above_cutoff"] = dict(points=int(vm.size), worst=float(vm.max()) if vm.size else None)
    # the labels
    glob_fail = [k for k, ok in (("K6", k6), ("K7", k7), ("K9", all(v["passed"] for v in res["K9"].values())), ("K10", res["K10"]["passed"])) if not ok]
    grid_fail = {}
    for g, (base, U, Nn) in GRIDS.items():
        info = res["grids"][g]
        f = [k for k in ("K8", "K12") if info[k]["passed"] is False]
        bad11 = [j for j in res["K11"] if (j == U or EXTRA.get(j) == g) and not res["K11"][j]["passed"]]
        if bad11:
            f.append("K11 (" + ", ".join(bad11) + ")")
        grid_fail[g] = f

    def word(held, withheld):
        if withheld:
            return "no label (" + "; ".join(withheld) + ")"
        if held is None:
            return "no label (too few bins or points)"
        return "HELD" if held else "NOT HELD"

    sc12 = ["scatter of K^2"] if sb.get("median_dK2_over_w2", 0) > 0.025 else []
    sc3 = ["scatter of the quotient"] if sb.get("median_dw2act_over_w2", 0) > 0.05 else []
    labels = {}
    for g in GRIDS:
        info = res["grids"][g]
        base_w = [k + " failed" for k in glob_fail + grid_fail[g]]
        labels[g] = dict(R1=word(info["R1"]["held"], base_w + sc12), R2=word(info["R2"]["held"], base_w + sc12),
                         R3=word(info["R3"]["held"], base_w + sc3), labeled=bool(g in ("b", "d")))

    def sc4(s):
        return s.get("median_dc2", 0) > 0.025 or s.get("median_dm2_rel", 0) > 0.05

    for g1, g2, what in PAIRS:
        w = [k + " failed" for k in glob_fail] + [f"{k} failed on {g}" for g in (g1, g2) for k in grid_fail[g]]
        spac = {res["grids"][g]["spacing"] for g in (g1, g2)}
        if 0.5 in spac and sc4(sa):
            w.append("scatter at spacing 0.5")
        if 0.25 in spac and sc4(sb):
            w.append("scatter at spacing 0.25")
        labels[g1 + g2] = dict(R4=word(res["R4"][g1 + g2]["held"], w))
    res["labels"], res["withheld_globally"], res["withheld_per_grid"] = labels, glob_fail, grid_fail
    print("labels", json.dumps(labels, indent=1), flush=True)
    (data / "m5_47_reads.json").write_text(json.dumps(jsonable(res)) + "\n")
    print("written", data / "m5_47_reads.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--grids", nargs="*", default=list(GRIDS), help="with --compute: a partial read of these grids (a preview, written to the checkpoints)")
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    ap.add_argument("--plots-dir", type=Path, default=HERE.parent / "plots")
    ap.add_argument("--work", type=Path, default=HERE.parent / "checkpoints" / "m5_47")
    a = ap.parse_args()
    if a.compute:
        compute(a)
    if a.tables:
        import m5_47_report as RP

        RP.tables(a)
    if a.plots:
        import m5_47_report as RP

        RP.plots(a)
    if not (a.compute or a.tables or a.plots):
        ap.print_help()


if __name__ == "__main__":
    main()
