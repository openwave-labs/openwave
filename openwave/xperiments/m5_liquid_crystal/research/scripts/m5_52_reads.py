"""M5.52: the controls, the reads, the words, the tables and the plots of the driven wave.

THE READ of a piece S_i of a line (its nodes in the domain D), at a tone w, on the lock-in amplitude A_k of window k:

    a(x, w)  = <W(x), A_k(x, w)>                         the inner product of M5.47 section 3 (metric 1, 2, 2, 1, 2, 1)
    Z_i      = - sum_S conj(a) Lap_h a / sum_S |a|^2     Lap_h the seven-point Laplacian, unfiltered
    m2_read  = w^2 - w gamma Re Z / Im Z,   c2_read = w gamma / Im Z,   y_i = w^2 - m2_read

NOT READ: a node of the piece with g >= 1e-3 (the source); the root mean square of a over its nodes in D under
1e-12; c2_read outside 0.4 to 2.5; the soft rule m2_read < 0.5 (0.87 m_F)^2; and the pieces M5.49 does not read.

THE BUDGET   eps_i = max(0.01 w^2, 2 |y - y(window 1)|, 2 |y - y(filtered)|, 2 b_gamma(w, bin), 2 b_vac(w))
with the filtered term only where the filtered g is under 1e-5 on every node of the piece; b_gamma the 90th
percentile of |y(Gb) - y(GbG)| over the read pieces of grid b in the bin of x = |y| / (w gamma) (I under 1, II 1 to
4, III 4 and above; a bin of fewer than five pieces takes the largest of the bins); b_vac the worst
|m2_read - m^2| of the vacuum cases of C9 that carry a bound at w.

PROFILES y, y + eps, y - eps, joined between the centers of consecutive read pieces no more than 2 D apart. A piece
is PROP when y - eps >= 0, EVAN when y + eps < 0, MARGIN otherwise. An edge is a radius where a profile crosses 0
(outer: from positive to negative as s grows). Phi[p] = int sqrt(max(p, 0)) ds.

Usage:  python m5_52_reads.py --c13                          (C13 alone; stage 0 runs it in its own process)
        python m5_52_reads.py --read Ge --hash b=.. d=.. e=.. V=.. settings=..
        python m5_52_reads.py --words --hash ...             (C4, C6, C7, C9, C10, C11 and the words; writes the summary)
        python m5_52_reads.py --plots
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from scipy.ndimage import map_coordinates

import m5_47_evolve as E
import m5_49_predict as PR
import m5_52_drive as D

TONES = D.TONES
FLOOR = 0.01
C2_LO, C2_HI = 0.4, 2.5
SOFT_SHARE, SOFT_RATIO = 0.5, 0.87
RMS_FLOOR = 1e-12
G_EXCL, G_FILT = 1e-3, 1e-5
LAMBDAS = (1.0, 0.935, 0.87)
DOM_BOUND = 1.2
TOL = 0.10
RMIN = 2.5
FORMS_TOL = 0.03
COVER = 0.9
BIN_EDGES = (1.0, 4.0)
C11_BOUND = (0.01, 0.06, 0.12)
LABELED = ("b", "d", "e")
GRID_JOB = dict(b="Gb", d="Gd", e="Ge")
LINES = ("A", "PF", "PD", "O")
WORD_LINES = ("A", "PF", "PD")
REG = {4.0: dict(plane=dict(inner=2.77, outer=6.31)), 5.5: dict(A=dict(inner=0.81, outer=8.38), plane=dict(inner=1.56, outer=9.74))}
C9_BOUNDED = {4.0: ("VD2", "VD3"), 5.5: ("VD1", "VD4", "VD2", "VD3")}
C9_TABLED = {4.0: ("VD1", "VD4")}
READS = D.WORK / "reads"


# ----------------------------------------------------------------------------- pure functions: the field and a piece

def inner_W(W6, A6):
    return np.einsum("...c,...c->...", W6 * E.METRIC, A6)


def power6(A6):
    return (np.abs(A6) ** 2 * E.METRIC).sum(-1)


def lap(a, h):
    out = -6.0 * a
    for j in range(3):
        out = out + np.roll(a, -1, axis=j) + np.roll(a, 1, axis=j)
    return out / (h * h)


def binomial_filter(a, passes=8):
    """[1, 2, 1]/4 along each axis, `passes` times per axis, a extended past the shell by odd reflection about the
    shell nodes (a = 0 on the shell)."""
    b = np.array(a, copy=True)
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = 0
        b[tuple(sl)] = 0
        sl[ax] = -1
        b[tuple(sl)] = 0
    N = b.shape[0]
    for ax in range(3):
        for _ in range(passes):
            pad = np.concatenate([-np.take(b, [1], axis=ax), b, -np.take(b, [N - 2], axis=ax)], axis=ax)
            b = 0.25 * np.take(pad, range(0, N), axis=ax) + 0.5 * np.take(pad, range(1, N + 1), axis=ax) + 0.25 * np.take(pad, range(2, N + 2), axis=ax)
    return b


def ratio_read(Z, w, gam):
    Z = np.asarray(Z, complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = w * gam * Z.real / Z.imag
        c2 = w * gam / Z.imag
    return w * w - y, c2, y


def piece_max(L, field, sel=None):
    out = np.zeros(L["n"])
    on = (L["piece"] >= 0) if sel is None else sel
    np.maximum.at(out, L["piece"][on], field[on])
    return out


def piece_sums(L, field):
    return np.bincount(L["piece"][L["sel"]], weights=field[L["sel"]], minlength=L["n"])


def read_field(a, P6, h, L, w, gam, mF2, gmax, read49):
    """The read of every piece of a line from the scalar a (and the six-component power P6)."""
    La = lap(a, h)
    num = -np.conj(a) * La
    nr, ni = piece_sums(L, num.real), piece_sums(L, num.imag)
    den = piece_sums(L, np.abs(a) ** 2)
    pout = piece_sums(L, P6 - np.abs(a) ** 2) if P6 is not None else np.full(L["n"], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = (nr + 1j * ni) / den
        rms = np.sqrt(den / L["nodes_dom"])
    m2, c2, y = ratio_read(Z, w, gam)
    reason = []
    for i in range(L["n"]):
        if not read49[i]:
            reason.append("not read by the geometry of M5.49")
        elif gmax[i] >= G_EXCL:
            reason.append("source")
        elif not (rms[i] >= RMS_FLOOR):
            reason.append("floor")
        elif not (C2_LO <= c2[i] <= C2_HI):
            reason.append("c2")
        elif mF2[i] is None or not np.isfinite(mF2[i]) or m2[i] < SOFT_SHARE * SOFT_RATIO ** 2 * mF2[i]:
            reason.append("soft")
        else:
            reason.append("")
    ok = np.array([r == "" for r in reason])
    return dict(Z_re=Z.real, Z_im=Z.imag, m2=m2, c2=c2, y=y, rms=rms, p_in=den, p_out=pout, reason=reason, ok=ok)


# ----------------------------------------------------------------------------- pure functions: profiles, edges, phase

def joined_pairs(centers, ok, D_):
    idx = [i for i in range(len(centers)) if ok[i]]
    return [(a, b) for a, b in zip(idx[:-1], idx[1:]) if centers[b] - centers[a] <= 2 * D_ + 1e-9]


def edges0(centers, p, ok, D_):
    """The radii where the profile p (on the read pieces, joined) crosses 0. outer: from >= 0 to < 0 as s grows."""
    out = []
    for a, b in joined_pairs(centers, ok, D_):
        pa, pb = p[a], p[b]
        if (pa >= 0) != (pb >= 0):
            r = centers[a] + (0 - pa) / (pb - pa) * (centers[b] - centers[a])
            out.append(dict(r=float(r), kind="outer" if pa >= 0 else "inner", between=[int(a), int(b)]))
    return out


def phi_linear(s0, s1, p0, p1):
    """int_{s0}^{s1} sqrt(max(p, 0)) ds for p linear from p0 to p1."""
    if s1 <= s0:
        return 0.0
    k = (p1 - p0) / (s1 - s0)
    G = lambda x: (2.0 / 3.0) * max(x, 0.0) ** 1.5
    if abs(k) < 1e-14:
        return float(np.sqrt(max(p0, 0.0)) * (s1 - s0))
    return float((G(p1) - G(p0)) / k)


def phi_on(centers, p, pairs, lo, hi):
    """Phi of the joined profile over [lo, hi] (only on joined segments; a point stretch gives 0)."""
    tot = 0.0
    for a, b in pairs:
        s0, s1 = centers[a], centers[b]
        u0, u1 = max(s0, lo), min(s1, hi)
        if u1 <= u0:
            continue
        f = lambda s: p[a] + (p[b] - p[a]) * (s - s0) / (s1 - s0)
        tot += phi_linear(u0, u1, f(u0), f(u1))
    return tot


def runs_nonneg(dom, ok, p):
    """Runs of consecutive domain pieces, each read, with p >= 0 (lists of piece indices)."""
    runs, cur = [], []
    for i in dom:
        if ok[i] and p[i] >= 0:
            if cur and i != cur[-1] + 1:
                runs.append(cur)
                cur = []
            cur.append(i)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def run_phi(centers, p, ok, D_, run, dom):
    """Phi of a run extended to its crossings on the joined profile and cut at the first and last domain centers."""
    pairs = joined_pairs(centers, ok, D_)
    lo, hi = centers[run[0]], centers[run[-1]]
    for a, b in pairs:
        if b == run[0] and p[a] < 0:
            lo = centers[a] + (0 - p[a]) / (p[b] - p[a]) * (centers[b] - centers[a])
        if a == run[-1] and p[b] < 0:
            hi = centers[a] + (0 - p[a]) / (p[b] - p[a]) * (centers[b] - centers[a])
    lo, hi = max(lo, centers[dom[0]]), min(hi, centers[dom[-1]])
    return phi_on(centers, p, pairs, lo, hi), (float(lo), float(hi))


def statement(centers, D_, dom, ok, y, eps, purity=None):
    """The statement of W1 on one window: WAVE, NO WAVE, ZONE or None (undecided), with its numbers."""
    y = np.asarray(y, float)
    eps = np.asarray(eps, float)
    prof = dict(y=y, hi=y + eps, lo=y - eps)
    n_ok = sum(bool(ok[i]) for i in dom)
    cover = n_ok / len(dom) if dom else 0.0
    info = dict(cover=cover, n_domain=len(dom), n_read=n_ok, runs={})
    for k, p in prof.items():
        info["runs"][k] = [dict(pieces=r, phi=run_phi(centers, p, ok, D_, r, dom)[0], span=run_phi(centers, p, ok, D_, r, dom)[1]) for r in runs_nonneg(dom, ok, p)]
    if not dom or cover < COVER - 1e-12:
        return None, dict(info, why="cover under nine in ten")
    big = {k: [r for r in info["runs"][k] if r["phi"] >= np.pi] for k in prof}
    if all(big[k] for k in prof):
        pure = [r for r in big["hi"] if purity is None or all(purity[i] for i in r["pieces"])]
        if pure:
            return "WAVE", dict(info, why="")
        info["why_purity"] = "a run meets every clause of WAVE but the purity clause"
    if all(prof["hi"][i] < 0 for i in dom if ok[i]):
        return "NO WAVE", dict(info, why="")
    if info["runs"]["y"]:
        return "ZONE", dict(info, why=info.get("why_purity", "a run at or above 0 on y, neither WAVE nor NO WAVE"))
    return None, dict(info, why="no run on y, not every piece EVAN")


def w1_word(st2, st1):
    st = st2 if st2 == st1 else None
    if st == "WAVE":
        return st, "NOT HELD", ""
    if st == "NO WAVE":
        return st, "HELD", ""
    if st2 != st1:
        return None, "no label", f"window 1 says {st1}, window 2 says {st2}"
    return st, "no label", "ZONE" if st == "ZONE" else "undecided"


def w2_word(F_st, read_st):
    if F_st in ("WAVE", "NO WAVE") and read_st in ("WAVE", "NO WAVE"):
        return ("HELD", "") if F_st == read_st else ("NOT HELD", f"F says {F_st}, the read says {read_st}")
    return "no label", f"F says {F_st}, the read says {read_st}"


def window_cover(r0, centers, D_, ok):
    idx, complete = PR.window(r0, centers, D_)
    return bool(complete and idx and all(ok[i] for i in idx)), idx


def edge_entry(r0, kind, centers, D_, ok, edges3, F_forms=None):
    """HELD / NOT HELD / no label for one expected edge; edges3: the read edges on the profiles y, y + eps, y - eps.
    F_forms: (edge on the pieces or None, edge on the ray or None) for an R entry."""
    near = [min([e["r"] for e in E3 if e["kind"] == kind], key=lambda r: abs(r - r0), default=None) for E3 in edges3] if r0 is not None else [None] * 3
    if r0 < RMIN:
        return "no label", "the radius is under 2.5", near
    if F_forms is not None:
        fp, fr = F_forms
        if fp is None or fr is None:
            return "no label", "only one form of F has the edge", near
        if abs(fp - fr) > FORMS_TOL * fp:
            return "no label", "the two forms of F differ by more than 3 percent of the radius", near
    cov, _ = window_cover(r0, centers, D_, ok)
    if not cov:
        return "no label", "the window is not covered", near
    found = [any(e["kind"] == kind and (1 - TOL) * r0 <= e["r"] <= (1 + TOL) * r0 for e in E3) for E3 in edges3]
    if all(found):
        return "HELD", "", near
    if not any(found):
        return "NOT HELD", "", near
    return "no label", "the three profiles disagree", near


def p_stretch(centers, D_, ok, dom_mask, r_edge):
    """The longest run of joined read pieces inside the domain that contains the outer edge of F."""
    okd = [bool(ok[i] and dom_mask[i]) for i in range(len(centers))]
    runs, cur = [], []
    for a, b in [(i, i) for i in range(len(centers))]:
        pass
    idx = [i for i in range(len(centers)) if okd[i]]
    for i in idx:
        if cur and centers[i] - centers[cur[-1]] <= 2 * D_ + 1e-9:
            cur.append(i)
        else:
            if cur:
                runs.append(cur)
            cur = [i]
    if cur:
        runs.append(cur)
    hold = [r for r in runs if r_edge is not None and centers[r[0]] <= r_edge <= centers[r[-1]]]
    return max(hold, key=len) if hold else None, runs


def p_entry(centers, D_, ok, dom_mask, y3, yF, r_edge):
    """P: Phi of the read (three profiles) against [0.9 Phi_F(1), 1.1 Phi_F(0.87)] on the stretch S."""
    if r_edge is None:
        return dict(word="no label", why="F has no outer edge on the pieces of this line", stretch=None)
    S, _ = p_stretch(centers, D_, ok, dom_mask, r_edge)
    if S is None:
        return dict(word="no label", why="the outer edge of F lies beyond the stretch", stretch=None)
    okS = [i in S for i in range(len(centers))]
    pairs = joined_pairs(centers, okS, D_)
    lo, hi = centers[S[0]], centers[S[-1]]
    phF = {lam: phi_on(centers, yF[lam], pairs, lo, hi) for lam in yF}
    phR = [phi_on(centers, p, pairs, lo, hi) for p in y3]
    out = dict(stretch=[float(lo), float(hi)], pieces=S, phi_F={str(k): v for k, v in phF.items()}, phi_read=phR,
               interval=[0.9 * phF[1.0], 1.1 * phF[0.87]])
    if phF[1.0] < 1.0:
        return dict(out, word="no label", why="Phi_F(1) is under 1 radian")
    if phF[0.87] > 2 * phF[1.0]:
        return dict(out, word="no label", why="Phi_F(0.87) exceeds 2 Phi_F(1)")
    inside = [out["interval"][0] <= v <= out["interval"][1] for v in phR]
    if all(inside):
        return dict(out, word="HELD", why="")
    if not any(inside):
        return dict(out, word="NOT HELD", why="")
    return dict(out, word="no label", why="the three profiles disagree")


def kappa_bar(centers, ok, p, r_edge, reach=1.5):
    sel = [i for i in range(len(centers)) if ok[i] and r_edge is not None and r_edge < centers[i] <= r_edge + reach]
    if not sel:
        return None
    return float(np.mean([np.sqrt(max(-p[i], 0.0)) for i in sel]))


def reading_word(words):
    if any(w == "NOT HELD" for w in words):
        return "NOT HELD"
    if any(w == "HELD" for w in words):
        return "HELD"
    return "no label"


def bin_of(y, w, gam=D.GAMMA):
    x = abs(y) / (w * gam)
    return 0 if x < BIN_EDGES[0] else (1 if x < BIN_EDGES[1] else 2)


# ----------------------------------------------------------------------------- the prediction (called by m5_52_drive.py --predict)

def lines_of(G):
    return PR.pieces_of(G)


def gfilt_of(G):
    if "gf" not in G:
        G["gf"] = binomial_filter(G["g"])
    return G["gf"]


def source_on_pieces(G, Ls):
    gf = gfilt_of(G)
    out = {}
    for ln, L in Ls.items():
        gm, gfm = piece_max(L, G["g"]), piece_max(L, gf)
        out[ln] = dict(g_max=gm, g_filtered_max=gfm, excluded=(gm >= G_EXCL), filter_allowed=(gfm < G_FILT))
    return out


def ray_statement(Lb, w, lam=1.0):
    """F's statement by the rule of W1 on the ray form, over the s-range of the domain on the pieces."""
    s = np.array(Lb["ray_form"]["s"], float)
    mr = np.array(Lb["ray_form"]["mF"], float)
    return s, w * w - lam ** 2 * mr ** 2


def domain_of(Lb, w):
    mF = np.array([np.nan if v is None else v for v in Lb["mF"]], float)
    return [i for i in range(len(mF)) if Lb["read"][i] and np.isfinite(mF[i]) and mF[i] ** 2 <= DOM_BOUND * w * w]


def F_statement(Lb, w):
    """F on the axis: the statement by the rule of W1 on y_F(1) with eps = 0.01 w^2, on the pieces and on the ray."""
    centers = np.array(Lb["centers"], float)
    mF = np.array([np.nan if v is None else v for v in Lb["mF"]], float)
    dom = domain_of(Lb, w)
    ok = np.array(Lb["read"], bool) & np.isfinite(mF)
    yF = w * w - mF ** 2
    eps = np.full(len(centers), FLOOR * w * w)
    st_p, info_p = statement(centers, Lb["D"], dom, ok, np.where(ok, yF, -1e9), eps)
    s, yr = ray_statement(Lb, w)
    lo, hi = centers[dom[0]], centers[dom[-1]]
    rd = [i for i in range(len(s)) if lo - 1e-9 <= s[i] <= hi + 1e-9]
    st_r, info_r = statement(s, (s[1] - s[0]) / 2, rd, np.ones(len(s), bool), yr, np.full(len(s), FLOOR * w * w))
    st = st_p if st_p == st_r else "inside its margin"
    return dict(pieces=st_p, ray=st_r, statement=st, info_pieces=info_p, info_ray=dict(info_r, runs={k: [dict(phi=r["phi"], span=r["span"]) for r in v] for k, v in info_r["runs"].items()}),
                largest_yF=float(np.nanmax(np.where(ok, yF, np.nan)[dom])), domain=dom)


def predicted_decay(Lb, w, lam):
    centers = np.array(Lb["centers"], float)
    mF = np.array([np.nan if v is None else v for v in Lb["mF"]], float)
    dom = domain_of(Lb, w)
    ok = [i in dom for i in range(len(centers))]
    p = -(w * w - lam ** 2 * mF ** 2)
    pairs = joined_pairs(centers, ok, Lb["D"])
    imin = min(dom, key=lambda i: mF[i])
    return max(phi_on(centers, np.nan_to_num(p), pairs, centers[dom[0]], centers[imin]), phi_on(centers, np.nan_to_num(p), pairs, centers[imin], centers[dom[-1]]))


def m49_reference(g, ln, Lb, w):
    """Phi_F(lambda) and kappa_F on the stretch that the fitted pieces of M5.49 give inside the domain, around the
    outer edge of F (reference only), and Phi with the cutoffs M5.49 read in place of m_F."""
    r49 = json.loads((D.DATA / "m5_49_reads.json").read_text())["grids"][g]["lines"][ln]
    centers = np.array(Lb["centers"], float)
    mF = np.array([np.nan if v is None else v for v in Lb["mF"]], float)
    wc = np.array([np.nan if v is None else v for v in r49["wc"]], float)
    ok = np.array(Lb["read"], bool) & np.isfinite(mF)
    dom = domain_of(Lb, w)
    dmask = [i in dom for i in range(len(centers))]
    outer = [e["r"] for e in Lb["edges"]["pieces"][str(w)] if e["kind"] == "outer"]
    out = dict(outer_edge_F=outer[0] if outer else None)
    if not dom:
        return out
    if outer:
        S, _ = p_stretch(centers, Lb["D"], ok, dmask, outer[0])
    else:
        S = None
    if S is None:
        okd = [bool(ok[i] and dmask[i]) for i in range(len(centers))]
        S = [i for i in range(len(centers)) if okd[i]]
    if not S:
        return out
    okS = [i in S for i in range(len(centers))]
    pairs = joined_pairs(centers, okS, Lb["D"])
    lo, hi = centers[S[0]], centers[S[-1]]
    out.update(stretch=[lo, hi], phi_F={str(l): phi_on(centers, np.nan_to_num(w * w - l ** 2 * mF ** 2), pairs, lo, hi) for l in LAMBDAS},
               phi_m5_49_cutoffs=phi_on(centers, np.where(np.isfinite(wc), w * w - wc ** 2, np.nan_to_num(w * w - mF ** 2)), pairs, lo, hi),
               m5_49_cutoffs_missing=[float(centers[i]) for i in S if not np.isfinite(wc[i])])
    if outer:
        out["kappa_F"] = {str(l): kappa_bar(centers, np.isfinite(mF) & np.array(Lb["read"], bool), w * w - l ** 2 * mF ** 2, outer[0]) for l in LAMBDAS}
    return out


def q_on_circles(G, n=64):
    c, nu, Rs = G["center"], G["normal"] / np.linalg.norm(G["normal"]), G["Rs"]
    u = np.array([1.0, 0, 0]) if abs(nu[0]) < 0.9 else np.array([0, 1.0, 0])
    u = u - (u @ nu) * nu
    u /= np.linalg.norm(u)
    v = np.cross(nu, u)
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out = {}
    for sgn in (1.0, -1.0):
        pts = c + Rs * (np.cos(ph)[:, None] * u + np.sin(ph)[:, None] * v) + sgn * D.ZS * nu
        idx = ((pts - G["x"][0]) / G["h"]).T
        q = map_coordinates(G["q"], idx, order=1)
        out["+z" if sgn > 0 else "-z"] = [float(q.min()), float(q.max())]
    return out


def predict_grid(g, block, G):
    Ls = lines_of(G)
    src = source_on_pieces(G, Ls)
    out = dict(grid=g, base=block["base"], loop=block["loop"], source_radius=G["Rs"], q_on_circles=q_on_circles(G), lines={})
    for ln in LINES:
        Lb = block["lines"][ln]
        mF = np.array([np.nan if v is None else v for v in Lb["mF"]], float)
        rec = dict(centers=Lb["centers"], D=Lb["D"], read_m5_49=Lb["read"], mF=Lb["mF"], mF_qP=Lb["mF_qP"], C=Lb["C"],
                   ray_form=dict(s=Lb["ray_form"]["s"], mF=Lb["ray_form"]["mF"]), edges_F=Lb["edges"], source=src[ln], at={})
        for w in TONES:
            dom = domain_of(Lb, w)
            at = dict(y_F={str(l): np.where(np.array(Lb["read"], bool), w * w - l ** 2 * mF ** 2, np.nan) for l in LAMBDAS},
                      domain=dom, domain_s=[Lb["centers"][i] for i in dom] if dom else [],
                      reference=m49_reference(g, ln, Lb, w) if ln in WORD_LINES else None)
            if ln == "A" and w == 4.0 and dom:
                at["statement_F"] = F_statement(Lb, w)
                at["largest_decay"] = {str(l): predicted_decay(Lb, w, l) for l in (1.0, 0.935)}
            rec["at"][str(w)] = at
        out["lines"][ln] = rec
    return out


def predict_vacuum(block, G):
    Ls = lines_of(G)
    src = source_on_pieces(G, Ls)
    out = dict(geometry="V2 of M5.49 (half-box 6, spacing 0.25), center of the box, normal along the third axis", source_radius=D.RS_VAC,
               source=src, jobs={})
    for name, v in D.VAC_V.items():
        m2 = v * PR.M_MID2
        rec = dict(v=v, m2=m2, at={})
        for w in TONES:
            y = w * w - m2
            ph = {}
            for ln in WORD_LINES:
                Lb = block["lines"][ln]
                ok = [bool(Lb["read"][i] and not src[ln]["excluded"][i]) for i in range(Lb["pieces"])]
                idx = [i for i in range(len(ok)) if ok[i]]
                ph[ln] = dict(read_pieces=len(idx), stretch=[Lb["centers"][idx[0]], Lb["centers"][idx[-1]]],
                              phi=float(np.sqrt(max(y, 0)) * (Lb["centers"][idx[-1]] - Lb["centers"][idx[0]])))
            rec["at"][str(w)] = dict(y=y, lines=ph, bounded=name in C9_BOUNDED[w])
        out["jobs"][name] = rec
    return out


def settings():
    return dict(tones=TONES, step=D.DT, steps=D.STEPS, windows=[D.J1, D.J2], window_length=D.NWIN, gamma=D.GAMMA, amplitude=D.AMP, ramp=D.RAMP,
                z_s=D.ZS, sigma_s=D.SIGMA_S, R_s_vacuum=D.RS_VAC, jobs=D.JOBS, floor=FLOOR, c2_bounds=[C2_LO, C2_HI], soft=[SOFT_SHARE, SOFT_RATIO],
                rms_floor=RMS_FLOOR, g_excluded=G_EXCL, g_filter=G_FILT, filter=dict(taps=[0.25, 0.5, 0.25], passes=8), lambdas=LAMBDAS,
                domain_bound=DOM_BOUND, tolerance=TOL, rmin=RMIN, forms=FORMS_TOL, cover=COVER, bins=BIN_EDGES, C11_bounds=C11_BOUND,
                C9_bounded=C9_BOUNDED, registered=REG, P_rule=dict(low=0.9, high=1.1, phi_min=1.0, factor=2.0), W1_phase=float(np.pi))


# ----------------------------------------------------------------------------- C14

def control_c14(Gs, doc49):
    out, ok = [], True
    for g, G in Gs.items():
        block = doc49["blocks"][g]
        lp = block["loop"]
        dc = float(np.abs(np.array(G["center"]) - np.array(lp["center"])).max())
        dn = float(np.abs(np.array(G["normal"]) - np.array(lp["normal"])).max())
        dr = abs(G["radius"] - lp["radius"])
        Ls = lines_of(G)
        geo, qok = True, True
        for ln in LINES:
            L, Lb = Ls[ln], block["lines"][ln]
            geo &= bool(L["n"] == Lb["pieces"] and L["read"].tolist() == Lb["read"] and L["nodes_dom"].tolist() == Lb["nodes_domain"])
            qm = PR.piece_mean(L, G["q"])
            for i, qb in enumerate(Lb["q"]):
                if qb is None:
                    qok &= not np.isfinite(qm[i])
                else:
                    qok &= float(f"{qm[i]:.7g}") == qb
        r = dict(grid=g, center=dc, normal=dn, radius=dr, geometry_identical=geo, q_seven_digits=qok,
                 passed=bool(dc <= 1e-6 and dn <= 1e-6 and dr <= 1e-6 and geo and qok))
        ok &= r["passed"]
        out.append(r)
    return dict(grids=out, passed=bool(ok), summary=[(r["grid"], r["passed"], r["center"], r["normal"], r["radius"]) for r in out])


# ----------------------------------------------------------------------------- C13, the readers on made-up inputs

def lockin_series(x, j0, nwin, dt, tone):
    cr = ci = 0.0
    for j in range(j0, j0 + nwin):
        a, b = D.lockin_coefficients(j, j0, nwin, dt, tone)
        cr += a * x[j - j0]
        ci += b * x[j - j0]
    return (cr + 1j * ci) * D.window_norm(nwin)


def lockin_analytic(B, w_read, w_tone, j0, nwin, dt):
    def H(theta):
        u = np.arange(nwin)
        G = lambda th: (1 - np.exp(1j * th * nwin)) / (1 - np.exp(1j * th)) if abs(np.exp(1j * th) - 1) > 1e-14 else complex(nwin)
        return 0.5 * G(theta) - 0.25 * G(theta + 2 * np.pi / nwin) - 0.25 * G(theta - 2 * np.pi / nwin)
    th1, th2 = (w_read - w_tone) * dt, (w_read + w_tone) * dt
    S = B * np.exp(1j * th1 * j0) * H(th1) + np.conj(B) * np.exp(1j * th2 * j0) * H(th2)
    return S * D.window_norm(nwin) / 2.0


def lattice_K(kappa2, c2, e, h):
    """The complex K with c2 (4/h^2) sum_i sin^2(K e_i h / 2) = kappa2, by Newton from the continuum root."""
    K = np.sqrt(kappa2 / c2 + 0j)
    for _ in range(60):
        f = c2 * 4 / h ** 2 * np.sum(np.sin(K * e * h / 2) ** 2) - kappa2
        fp = c2 * 2 / h * np.sum(e * np.sin(K * e * h))
        dK = f / fp
        K -= dK
        if abs(dK) < 1e-15 * max(1.0, abs(K)):
            break
    return K


def c13(seed=52013):
    res, ok = {}, True
    # (a) lock-in
    dt, nwin, j0 = D.DT, D.NWIN, D.J2
    t = np.arange(j0, j0 + nwin) * dt
    A4, A55 = 0.7 - 0.2j, -0.3 + 0.5j
    x = (A4 * np.exp(-4j * t)).real + (A55 * np.exp(-5.5j * t)).real
    a1, a2 = lockin_series(x, j0, nwin, dt, 4.0), lockin_series(x, j0, nwin, dt, 5.5)
    x55 = (A55 * np.exp(-5.5j * t)).real
    a3 = lockin_series(x55, j0, nwin, dt, 4.0)
    B = 0.4 + 0.9j
    x43 = (B * np.exp(-4.3j * t)).real
    a4, an = lockin_series(x43, j0, nwin, dt, 4.0), lockin_analytic(B, 4.0, 4.3, j0, nwin, dt)
    ra = dict(two_tone_4=abs(a1 - A4), two_tone_55=abs(a2 - A55), tone55_at_4=abs(a3), tone43_vs_analytic=abs(a4 - an), tone43_read=[a4.real, a4.imag])
    ra["passed"] = bool(ra["two_tone_4"] <= 1e-12 and ra["two_tone_55"] <= 1e-12 and ra["tone55_at_4"] <= 1e-12 and ra["tone43_vs_analytic"] <= 1e-10)
    res["a"] = ra
    ok &= ra["passed"]
    # (b) the estimator on a 49^3 lattice of spacing 0.25
    N, hb, h, w, gam = 49, 6.0, 0.25, 4.0, 0.3
    xg = np.linspace(-hb, hb, N)
    X = np.stack(np.meshgrid(xg, xg, xg, indexing="ij"), -1)
    inner = (np.abs(X) <= hb - 2.5 + 1e-9).all(-1)

    def readZ(a, sel):
        La = lap(a, h)
        Z = -(np.conj(a[sel]) * La[sel]).sum() / (np.abs(a[sel]) ** 2).sum()
        return ratio_read(Z, w, gam)

    def shell0(a):
        b = a.copy()
        b[0], b[-1], b[:, 0], b[:, -1], b[:, :, 0], b[:, :, -1] = 0, 0, 0, 0, 0, 0
        return b

    rb, okb = [], True
    for m2 in (14.0, 18.0):
        kap2 = w * w - m2 + 1j * w * gam
        ex, ed = np.array([1.0, 0, 0]), np.ones(3) / np.sqrt(3)
        Kx, Kd = lattice_K(kap2, 1.0, ex, h), lattice_K(kap2, 1.0, ed, h)
        cases = {"axis": np.exp(1j * Kx * X[..., 0]), "diagonal": np.exp(1j * Kd * (X @ ed)),
                 "two opposite": np.exp(1j * Kx * X[..., 0]) + 0.7 * np.exp(-1j * Kx * X[..., 0])}
        for nm, a in cases.items():
            a = shell0(a)
            m2r, c2r, yr = readZ(a, inner)
            m2f, _, _ = readZ(binomial_filter(a), inner)
            st = "PROP" if yr - FLOOR * w * w >= 0 else ("EVAN" if yr + FLOOR * w * w < 0 else "MARGIN")
            want = "PROP" if m2 < w * w else "EVAN"
            rw = np.cos(4 * X[..., 1] + 0.3)
            sc = np.sqrt(0.1 / 0.9 * (np.abs(a[inner]) ** 2).sum() / (rw[inner] ** 2).sum())
            b = shell0(a + sc * rw)
            m2b, _, _ = readZ(b, inner)
            m2bf, _, _ = readZ(binomial_filter(b), inner)
            r = dict(m2=m2, case=nm, raw=float(m2r), c2=float(c2r), status=st, filtered=float(m2f), added_raw=float(m2b), added_filtered=float(m2bf))
            r["passed"] = bool(abs(m2r - m2) <= 1e-9 and st == want and abs(m2f - m2r) <= 1e-6 and abs(m2bf - m2) <= FLOOR * w * w and abs(m2b - m2) > 0.05 * w * w)
            okb &= r["passed"]
            rb.append(r)
    # the soft input: a massless damped wave with 90 % of the power over the evanescent wave of m^2 = 18
    kap18 = w * w - 18.0 + 1j * w * gam
    ev = np.exp(1j * lattice_K(kap18, 1.0, np.array([1.0, 0, 0]), h) * X[..., 0])
    sw = np.exp(1j * lattice_K(w * w + 1j * w * gam, 1.0, np.array([0, 1.0, 0]), h) * X[..., 1])
    sc = np.sqrt(0.9 / 0.1 * (np.abs(ev[inner]) ** 2).sum() / (np.abs(sw[inner]) ** 2).sum())
    a = shell0(ev + sc * sw)
    m2s, c2s, _ = readZ(a, inner)
    reason = "c2" if not (C2_LO <= c2s <= C2_HI) else ("soft" if m2s < SOFT_SHARE * SOFT_RATIO ** 2 * 18.0 else "")
    soft = dict(m2_read=float(m2s), c2_read=float(c2s), reason=reason, passed=bool(reason == "soft"))
    okb &= soft["passed"]
    # the two-speed input
    rng = np.random.default_rng(seed)
    loc = np.arange(-6, 7) * h
    LX = np.stack(np.meshgrid(loc, loc, np.arange(-2, 4) * h, indexing="ij"), -1)
    piece = (LX[..., 2] >= -1e-9) & (LX[..., 2] < 0.5 - 1e-9) & ((LX[..., 0] ** 2 + LX[..., 1] ** 2) <= 1.0 + 1e-9)

    def lap_local(a):
        out = np.zeros_like(a)
        out[1:-1, 1:-1, 1:-1] = (a[2:, 1:-1, 1:-1] + a[:-2, 1:-1, 1:-1] + a[1:-1, 2:, 1:-1] + a[1:-1, :-2, 1:-1] + a[1:-1, 1:-1, 2:] + a[1:-1, 1:-1, :-2]
                                 - 6 * a[1:-1, 1:-1, 1:-1]) / h ** 2
        return out

    two = {}
    ys = {4.0: (-3.2, -1, -0.5, 0.5, 1, 3, 6, 7), 5.5: (-3.2, -1, 1, 3, 6, 10, 14)}
    for wt, ylist in ys.items():
        rows = []
        for y in ylist:
            for _ in range(300):
                dirs, amps, c2s_ = [], [], []
                for _k in range(3):
                    e = rng.standard_normal(3)
                    dirs.append(e / np.linalg.norm(e))
                    amps.append(rng.standard_normal() + 1j * rng.standard_normal())
                    c2s_.append(0.7 if rng.random() < 0.5 else 0.9)
                out = []
                for g_ in (gam, gam / 2):
                    kap = wt * wt - (wt * wt - y) + 1j * wt * g_
                    a = sum(A_ * np.exp(1j * lattice_K(kap, c_, e_, h) * (LX @ e_)) for A_, c_, e_ in zip(amps, c2s_, dirs))
                    Z = -(np.conj(a[piece]) * lap_local(a)[piece]).sum() / (np.abs(a[piece]) ** 2).sum()
                    m2r, _, yr = ratio_read(Z, wt, g_)
                    out.append((float(m2r), float(yr)))
                rows.append(dict(y=y, m2_set=wt * wt - y, m2_read=out[0][0], y_read=out[0][1], y_half=out[1][1]))
        per = {}
        for b in range(3):
            sel = [r for r in rows if bin_of(r["y_read"], wt) == b]
            if not sel:
                continue
            bg = float(np.percentile([abs(r["y_read"] - r["y_half"]) for r in sel], 90))
            eps = max(FLOOR * wt * wt, 2 * bg)
            err = np.array([abs(r["m2_read"] - r["m2_set"]) for r in sel])
            share = float((err > eps).mean())
            per[["I", "II", "III"][b]] = dict(draws=len(sel), b_gamma=bg, eps=eps, share_over_eps=share, p90_error=float(np.percentile(err, 90)), passed=bool(share <= 0.1))
        p90 = {str(y): float(np.percentile([abs(r["m2_read"] - r["m2_set"]) for r in rows if r["y"] == y], 90)) for y in ylist}
        two[str(wt)] = dict(bins=per, p90_error_by_y=p90, passed=all(v["passed"] for v in per.values()))
        okb &= two[str(wt)]["passed"]
    res["b"] = dict(plane_waves=rb, soft=soft, two_speed=two, passed=bool(okb))
    ok &= okb
    # (c) made-up profiles through every outcome of section 6
    res["c"] = c13_profiles()
    ok &= res["c"]["passed"]
    res["passed"] = bool(ok)
    res["summary"] = dict(a=ra["passed"], b=okb, c=res["c"]["passed"], c_cases=f"{res['c']['met']} of {res['c']['cases']}",
                          two_speed_p90={k: v["p90_error_by_y"] for k, v in two.items()})
    return res


def c13_profiles():
    D_ = 0.5
    centers = (np.arange(14) + 0.5) * D_
    dom = list(range(3, 10))
    eps = np.full(14, 0.16)
    cases = []

    def W1(y, y1=None, ok=None, pur=None, pur1=None):
        ok = np.ones(14, bool) if ok is None else ok
        s2, _ = statement(centers, D_, dom, ok, y, eps, pur)
        s1, _ = statement(centers, D_, dom, ok, y if y1 is None else y1, eps, pur if pur1 is None else pur1)
        return w1_word(s2, s1)

    def add(name, got, want):
        cases.append(dict(case=name, got=got, want=want, met=bool(got == want)))

    neg = np.full(14, -1.0)
    wave = np.where(np.isin(np.arange(14), dom), 2.0, -1.0)
    add("W1 HELD (NO WAVE)", W1(neg)[1:2], ("HELD",))
    add("W1 NOT HELD (WAVE)", W1(wave)[1:2], ("NOT HELD",))
    z = neg.copy()
    z[5:7] = 0.5
    add("W1 no label by a zone", W1(z)[0:2], ("ZONE", "no label"))
    add("W1 no label by the margin", W1(np.full(14, -0.1))[0:2], (None, "no label"))
    okc = np.ones(14, bool)
    okc[6] = False
    add("W1 no label by the cover", W1(neg, ok=okc)[0:2], (None, "no label"))
    tp = np.where(np.isin(np.arange(14), dom), 1.2, -1.0)
    add("W1 no label by the three profiles", W1(tp)[0:2], ("ZONE", "no label"))
    add("W1 no label by window 1", W1(wave, y1=z)[0:2], (None, "no label"))
    pur = np.ones(14, bool)
    pur[5] = False
    add("W1 no label by the purity clause", W1(wave, pur=pur)[0:2], ("ZONE", "no label"))
    add("W2 HELD", w2_word("NO WAVE", "NO WAVE")[0], "HELD")
    add("W2 NOT HELD", w2_word("NO WAVE", "WAVE")[0], "NOT HELD")
    add("W2 no label", w2_word("ZONE", "NO WAVE")[0], "no label")
    # edges: a profile y = 3 - s (outer edge at 3.0), read on every piece of a line of 14 pieces
    def prof(r):
        return 2.0 * (r - centers)

    def E3(r, shift=0.0):
        y = prof(r)
        return [edges0(centers, y + d, np.ones(14, bool), D_) for d in (0.0, 0.16 + shift, -0.16 - shift)]

    ok14 = np.ones(14, bool)
    add("edge HELD", edge_entry(4.0, "outer", centers, D_, ok14, E3(4.1))[0], "HELD")
    add("edge NOT HELD", edge_entry(4.0, "outer", centers, D_, ok14, E3(5.0))[0], "NOT HELD")
    okw = ok14.copy()
    okw[7] = False
    add("edge no label by the cover", edge_entry(4.0, "outer", centers, D_, okw, E3(4.1))[0:2], ("no label", "the window is not covered"))
    add("edge no label by 2.5", edge_entry(2.0, "outer", centers, D_, ok14, E3(2.0))[0:2], ("no label", "the radius is under 2.5"))
    add("edge no label by the three profiles", edge_entry(4.0, "outer", centers, D_, ok14, E3(4.38, shift=0.0))[0:2], ("no label", "the three profiles disagree"))
    add("edge no label by the two forms of F", edge_entry(4.0, "outer", centers, D_, ok14, E3(4.0), F_forms=(4.0, 4.3))[0:2],
        ("no label", "the two forms of F differ by more than 3 percent of the radius"))
    add("edge no label by one form of F", edge_entry(4.0, "outer", centers, D_, ok14, E3(4.0), F_forms=(4.0, None))[0:2], ("no label", "only one form of F has the edge"))
    # P: F with an outer edge at 4.0 (y_F(1) = 2 (4 - s)), the stretch the whole line
    dmask = [True] * 14
    yF = {1.0: 2.0 * (4.0 - centers), 0.935: 2.4 * (4.2 - centers), 0.87: 2.8 * (4.4 - centers)}
    ph = lambda p: phi_on(centers, p, joined_pairs(centers, ok14, D_), centers[0], centers[-1])
    base = 2.2 * (4.1 - centers)
    add("P HELD", p_entry(centers, D_, ok14, dmask, [base, base + 0.05, base - 0.05], yF, 4.0)["word"], "HELD")
    lowp = 0.5 * (4.0 - centers)
    add("P NOT HELD", p_entry(centers, D_, ok14, dmask, [lowp, lowp + 0.01, lowp - 0.01], yF, 4.0)["word"], "NOT HELD")
    small = {k: 0.02 * v for k, v in yF.items()}
    add("P no label by 1 radian", p_entry(centers, D_, ok14, dmask, [base] * 3, small, 4.0)["why"], "Phi_F(1) is under 1 radian")
    wide = dict(yF)
    wide[0.87] = 6.0 * (6.0 - centers)
    add("P no label by the factor of two", p_entry(centers, D_, ok14, dmask, [base] * 3, wide, 4.0)["why"], "Phi_F(0.87) exceeds 2 Phi_F(1)")
    hiF = 1.1 * ph(yF[0.87])
    scale = (hiF / ph(base)) ** 2
    add("P no label by the three profiles", p_entry(centers, D_, ok14, dmask, [base * scale, base * scale * 1.2, base * scale * 0.8], yF, 4.0)["why"],
        "the three profiles disagree")
    okp = ok14.copy()
    okp[8] = False
    okp[9] = False
    add("P no label by the edge beyond the stretch", p_entry(centers, D_, okp, dmask, [base] * 3, yF, 4.0)["why"], "the outer edge of F lies beyond the stretch")
    met = sum(c["met"] for c in cases)
    return dict(cases=len(cases), met=met, rows=cases, passed=bool(met == len(cases)))


def control_c13():
    return c13()


# ----------------------------------------------------------------------------- reading a job

def load_lockin(name):
    wd = D.WORK / "jobs" / name
    out = {}
    for k in (1, 2):
        z = np.load(wd / f"lockin_w{k}.npz", allow_pickle=False)
        A = z["Ar"] + 1j * z["Ai"]
        out[k] = A
    return out


def control_c12(name):
    """The steady state of the linearized equations, every tone and window, on the nodes of D with g < 1e-3."""
    job = D.JOBS[name]
    G = D.background(job["grid"])
    F = E.Field(G["N"], G["hb"], v=job.get("v", E.V))
    m0 = G["m0"]
    Amat = F.a_matrices(m0)
    h3 = F.h ** 3
    sel = G["dom"] & (G["g"] < G_EXCL)
    S = job["amp"] * G["g"][..., None] * h3 * np.einsum("...ab,...b->...a", Amat, G["W6"])
    A = load_lockin(name)
    f1, f2 = np.zeros(F.shape6), np.zeros(F.shape6)
    rows = []
    for k in (1, 2):
        for t, w in enumerate(TONES):
            Ak = A[k][t]
            HA = np.zeros(F.shape6, complex)
            for part, unit in ((Ak.real, 1.0), (Ak.imag, 1j)):
                part = np.ascontiguousarray(part)
                eps = 1e-5 / max(float(np.abs(part).max()), 1e-300)
                F.static(m0 + eps * part, f1)
                F.static(m0 - eps * part, f2)
                HA += unit * (-(f1 - f2) / (2 * eps))
            KA = h3 * np.einsum("...ab,...b->...a", Amat, Ak)
            R = HA - (w * w + 1j * w * job["gamma"]) * KA - 1j * S
            num = float(np.sqrt((np.abs(R[sel]) ** 2).sum()))
            den = float(np.sqrt((np.abs(HA[sel]) ** 2).sum()))
            rows.append(dict(window=k, tone=w, ratio=num / den, passed=bool(num <= 0.03 * den)))
    return dict(job=name, rows=rows, worst=max(r["ratio"] for r in rows), passed=all(r["passed"] for r in rows))


def check_hashes(hashes):
    doc = json.loads(D.PRED.read_text())
    for k, hsh in doc["hashes"].items():
        have = PR.block_hash(doc["blocks"][k])
        if hashes.get(k) != have or hsh != have:
            raise SystemExit(f"block {k}: the hash given ({hashes.get(k)}) is not that of the prediction block ({have}); no lock-in amplitude is opened")
    with open(D.WORK / "hash_checks.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} matched {sorted(hashes)}\n")
    return doc


def read_job(name, doc):
    """The read of every piece of every line, both tones and windows, and window 2 filtered; written to reads/<name>.json."""
    job = D.JOBS[name]
    G = D.background(job["grid"])
    Ls = lines_of(G)
    gam = job["gamma"]
    A = load_lockin(name)
    gf = gfilt_of(G)
    out = dict(job=name, grid=job["grid"], gamma=gam, lines={})
    vac = job["kind"] == "vacuum"
    if job["grid"] in LABELED:
        blk = doc["blocks"][job["grid"]]
    for ln in LINES:
        L = Ls[ln]
        gmax = piece_max(L, G["g"])
        gfmax = piece_max(L, gf)
        if vac:
            mF2 = np.full(L["n"], job["v"] * PR.M_MID2)
        elif job["grid"] in LABELED:
            mF2 = np.array([np.nan if v is None else v ** 2 for v in blk["lines"][ln]["mF"]], float)
        else:
            mF2 = PR.M_MID2 / PR.piece_mean(L, G["q"])
        rec = dict(centers=L["centers"], D=L["D"], read49=L["read"], g_max=gmax, g_filtered_max=gfmax, filter_allowed=gfmax < G_FILT, at={})
        for t, w in enumerate(TONES):
            at = {}
            for k in (1, 2):
                Ak = A[k][t]
                a = inner_W(G["W6"], Ak)
                at[f"w{k}"] = read_field(a, power6(Ak), G["h"], L, w, gam, mF2, gmax, L["read"])
                if k == 2:
                    at["filtered"] = read_field(binomial_filter(a), None, G["h"], L, w, gam, mF2, gfmax * 0, L["read"])
            if ln in WORD_LINES:
                a2 = inner_W(G["W6"], A[2][t])
                sr, rays = ray_samples(G, L, a2)
                at["rays"] = rays
                at["ray_s"] = sr
            rec["at"][str(w)] = at
        out["lines"][ln] = rec
    out = PR.jsonable(out)
    READS.mkdir(parents=True, exist_ok=True)
    (READS / f"{name}.json").write_text(json.dumps(out) + "\n")
    return out


def ray_samples(G, L, a):
    """The author's read, a descriptor: a (window 2) on each ray of the line, at points a spacing apart, inside the
    inset: modulus, phase (unwrapped along the ray) and minus its gradient."""
    h, hb = G["h"], G["hb"]
    smax = min(float(np.min((hb - PR.INSET - np.sign(e) * G["center"]) / np.maximum(np.abs(e), 1e-12))) for e in L["rays"])
    s = np.arange(0.0, smax + 1e-9, h)
    rays = []
    for e in L["rays"]:
        pts = G["center"] + s[:, None] * e
        idx = ((pts - G["x"][0]) / h).T
        v = map_coordinates(a.real, idx, order=1) + 1j * map_coordinates(a.imag, idx, order=1)
        ph = np.unwrap(np.angle(v))
        rays.append(dict(modulus=np.abs(v), phase=ph, k_phase=-np.gradient(ph, s)))
    return s, rays


def norm6(A, sel):
    return float(np.sqrt((np.abs(A[sel]) ** 2 * E.METRIC).sum()))


def grid_a_controls():
    """C4, C6 and C7 on the nodes of D of grid a with g < 1e-3 (the inner product of M5.47 section 3)."""
    G = D.background("a")
    sel = G["dom"] & (G["g"] < G_EXCL)
    L = {n: load_lockin(n) for n in ("Ga0", "GaH", "GaS", "GaP", "GaN")}
    out = dict(C4=[], C6=[], C7=[])
    for t, w in enumerate(TONES):
        A0 = L["Ga0"][2][t]
        d = (L["GaP"][2][t] - L["GaN"][2][t]) - A0
        out["C4"].append(dict(tone=w, window=2, rel=norm6(d, sel) / norm6(A0, sel)))
        for k in (1, 2):
            A0 = L["Ga0"][k][t]
            out["C6"].append(dict(tone=w, window=k, rel=norm6(A0 - 2 * L["GaH"][k][t], sel) / norm6(A0, sel)))
            out["C7"].append(dict(tone=w, window=k, rel=norm6(A0 - L["GaS"][k][t], sel) / norm6(A0, sel)))
    for key, bound in (("C4", 0.01), ("C6", 0.02), ("C7", 0.01)):
        for r in out[key]:
            r["passed"] = bool(r["rel"] <= bound)
        out[key] = dict(rows=out[key], worst=max(r["rel"] for r in out[key]), bound=bound, passed=all(r["passed"] for r in out[key]))
    out["nodes"] = int(sel.sum())
    return out


def arr(v):
    return np.array([np.nan if x is None else x for x in v], float)


def piece_arrays(rj, ln, w, key="w2"):
    r = rj["lines"][ln]["at"][str(w)][key]
    return dict(y=arr(r["y"]), m2=arr(r["m2"]), c2=arr(r["c2"]), ok=np.array(r["ok"], bool), reason=r["reason"], Z_re=arr(r["Z_re"]),
                rms=arr(r["rms"]), p_in=arr(r["p_in"]), p_out=arr(r["p_out"]))


def control_c9(R):
    """The vacuum jobs on the pieces of A, PF and PD."""
    rows, bvac = [], {}
    for w in TONES:
        errs = []
        for name in C9_BOUNDED[w]:
            m2set = D.VAC_V[name] * PR.M_MID2
            for ln in WORD_LINES:
                P = piece_arrays(R[name], ln, w)
                errs += [abs(P["m2"][i] - m2set) for i in range(len(P["ok"])) if P["ok"][i]]
        bvac[str(w)] = float(max(errs)) if errs else float("nan")
    for w in TONES:
        for name in C9_BOUNDED[w] + C9_TABLED.get(w, ()):
            m2set = D.VAC_V[name] * PR.M_MID2
            yset = w * w - m2set
            bounded = name in C9_BOUNDED[w]
            eps = max(FLOOR * w * w, 2 * bvac[str(w)])
            n_read, n_soft, n_geo, worst, c2dev, wrong, n_edges, phis = 0, 0, 0, 0.0, 0.0, 0, 0, {}
            per_line = {}
            for ln in WORD_LINES:
                P = piece_arrays(R[name], ln, w)
                c = np.array(R[name]["lines"][ln]["centers"], float)
                Dl = R[name]["lines"][ln]["D"]
                geo = np.array(R[name]["lines"][ln]["read49"], bool)
                n_geo += int(geo.sum())
                ok = P["ok"]
                n_read += int(ok.sum())
                n_soft += sum(r == "soft" for r in P["reason"])
                idx = [i for i in range(len(ok)) if ok[i]]
                if idx:
                    worst = max(worst, max(abs(P["m2"][i] - m2set) for i in idx))
                    c2dev = max(c2dev, max(abs(P["c2"][i] - 1) for i in idx))
                st = ["PROP" if P["y"][i] - eps >= 0 else ("EVAN" if P["y"][i] + eps < 0 else "MARGIN") for i in idx]
                want = "PROP" if yset > 0 else "EVAN"
                wrong += sum(x != want for x in st)
                yy = np.where(ok, P["y"], np.nan)
                ne = sum(len(edges0(c, yy + d, ok, Dl)) for d in (0.0, eps, -eps))
                n_edges += ne
                ph = None
                if yset > 0 and idx:
                    pairs = joined_pairs(c, ok, Dl)
                    ph = dict(read=phi_on(c, yy, pairs, c[idx[0]], c[idx[-1]]), set=float(np.sqrt(yset) * (c[idx[-1]] - c[idx[0]])))
                    ph["rel"] = abs(ph["read"] - ph["set"]) / ph["set"] if ph["set"] > 0 else None
                per_line[ln] = dict(read=int(ok.sum()), reasons=[r for r in P["reason"] if r and r != "not read by the geometry of M5.49"], edges=ne, phi=ph,
                                    m2=[float(P["m2"][i]) for i in idx], c2=[float(P["c2"][i]) for i in idx])
            row = dict(job=name, tone=w, y_set=yset, m2_set=m2set, bounded=bounded, geometry_pieces=n_geo, read=n_read, soft=n_soft, worst_m2_error=worst,
                       tolerance=FLOOR * w * w, worst_c2_dev=c2dev, eps=eps, wrong_status=wrong, edges=n_edges, lines=per_line)
            if bounded:
                phi_ok = all(v["phi"] is None or (v["phi"]["rel"] is not None and v["phi"]["rel"] <= 0.02) for v in per_line.values())
                row["clauses"] = dict(read=bool(n_read >= 24 and n_soft == 0), cutoff=bool(worst <= FLOOR * w * w), speed=bool(c2dev <= 0.01),
                                      statuses=bool(wrong == 0), edges=bool(n_edges == 0), phase=bool(phi_ok))
                row["passed"] = all(row["clauses"].values())
            rows.append(row)
    passed = {str(w): all(r["passed"] for r in rows if r["bounded"] and r["tone"] == w) for w in TONES}
    return dict(rows=rows, b_vac=bvac, passed=passed)


def control_c11(Rb, RbG):
    out, bg = {}, {}
    for w in TONES:
        diffs = {0: [], 1: [], 2: []}
        for ln in LINES:
            P, Q_ = piece_arrays(Rb, ln, w), piece_arrays(RbG, ln, w)
            for i in range(len(P["ok"])):
                if P["ok"][i] and Q_["ok"][i]:
                    diffs[bin_of(P["y"][i], w)].append(abs(P["y"][i] - Q_["y"][i]))
        p90 = {b: (float(np.percentile(v, 90)) if v else None) for b, v in diffs.items()}
        rows = []
        for b in range(3):
            bound = C11_BOUND[b] * w * w
            rows.append(dict(bin=["I", "II", "III"][b], pieces=len(diffs[b]), p90=p90[b], bound=bound, passed=(None if p90[b] is None else bool(p90[b] <= bound))))
        largest = max(v for v in p90.values() if v is not None)
        bg[str(w)] = [p90[b] if len(diffs[b]) >= 5 else largest for b in range(3)]
        for b in range(3):
            rows[b].update(value_used=bg[str(w)][b], took_largest=bool(len(diffs[b]) < 5))
        out[str(w)] = rows
    return dict(per_tone=out, b_gamma=bg)


def eps_of(P, P1, Pf, allowed, w, bgam, bvac):
    n = len(P["y"])
    eps, terms = np.full(n, np.nan), []
    for i in range(n):
        if not P["ok"][i]:
            terms.append(None)
            continue
        t = dict(floor=FLOOR * w * w)
        t["window"] = 2 * abs(P["y"][i] - P1["y"][i]) if P1["ok"][i] else np.inf
        if allowed[i]:
            t["filter"] = 2 * abs(P["y"][i] - Pf["y"][i]) if np.isfinite(Pf["y"][i]) else np.inf
        b = bin_of(P["y"][i], w)
        t["damping"] = 2 * bgam[b]
        t["bin"] = ["I", "II", "III"][b]
        t["vacuum"] = 2 * bvac
        eps[i] = max(v for k, v in t.items() if k != "bin")
        terms.append(t)
    return eps, terms


def words(doc, trial=False):
    jobs = [n for n in D.JOBS if (READS / f"{n}.json").exists()]
    R = {n: json.loads((READS / f"{n}.json").read_text()) for n in jobs}
    S = {n: json.loads((D.WORK / "jobs" / n / "series.json").read_text()) for n in D.JOBS if (D.WORK / "jobs" / n / "series.json").exists()}
    C12 = {n: json.loads((D.WORK / "c12" / f"{n}.json").read_text()) for n in D.JOBS if (D.WORK / "c12" / f"{n}.json").exists()}
    st0 = json.loads((D.WORK / "stage0.json").read_text())
    out = dict(task="M5.52", prediction_hashes=doc["hashes"], stage0={k: dict(passed=v["passed"]) for k, v in st0.items()})
    out["C8"] = {n: dict(ratio=s["C8_ratio"], passed=s["C8_passed"], steps_done=s["steps_done"], steps=s["steps"], capped=s["capped"],
                         largest_displacement=s["largest_displacement"], not_positive=s["not_positive"]) for n, s in S.items()}
    out["C12"] = {n: dict(worst=c["worst"], passed=c["passed"], rows=c["rows"]) for n, c in C12.items()}
    try:
        ga = grid_a_controls()
    except FileNotFoundError:
        if not trial:
            raise
        ga = {k: dict(passed=True, trial_missing=True) for k in ("C4", "C6", "C7")}
        ga["nodes"] = None
    out.update(C4=ga["C4"], C6=ga["C6"], C7=ga["C7"], grid_a_nodes=ga["nodes"])
    if trial and not all(v in R for v in D.VAC_V):
        c9 = dict(rows=[], b_vac={str(w): 0.0 for w in TONES}, passed={str(w): True for w in TONES}, trial_missing=True)
    else:
        c9 = control_c9(R)
    out["C9"] = c9
    if trial and not ("Gb" in R and "GbG" in R):
        c11 = dict(per_tone={str(w): [dict(passed=None)] * 3 for w in TONES}, b_gamma={str(w): [0.0] * 3 for w in TONES}, trial_missing=True)
    else:
        c11 = control_c11(R["Gb"], R["GbG"])
    out["C11"] = c11
    glob = []
    for key in ("C4", "C6", "C7"):
        if not out[key]["passed"]:
            glob.append(f"{key} not met")
    for k in ("C1", "C2", "C3", "C13", "C14"):
        if not st0[k]["passed"]:
            glob.append(f"{k} not met")
    out["withheld_globally"] = glob
    grids, entries = {}, []
    for g in LABELED:
        job = GRID_JOB[g]
        if trial and (job not in R or job not in C12):
            continue
        blk = doc["blocks"][g]
        rj = R[job]
        gw = dict(job=job, capped=S[job]["capped"], C8=S[job]["C8_passed"], lines={}, C10={}, C12={})
        withhold_grid = []
        if S[job]["capped"]:
            withhold_grid.append("capped")
        if not S[job]["C8_passed"]:
            withhold_grid.append("C8 not met on the job")
        per_w = {}
        for w in TONES:
            diffs = []
            for ln in LINES:
                P, P1 = piece_arrays(rj, ln, w), piece_arrays(rj, ln, w, "w1")
                diffs += [abs(P["y"][i] - P1["y"][i]) for i in range(len(P["ok"])) if P["ok"][i] and P1["ok"][i]]
            med = float(np.median(diffs)) if diffs else float("nan")
            gw["C10"][str(w)] = dict(median=med, bound=FLOOR * w * w, pieces=len(diffs), passed=bool(med <= FLOOR * w * w))
            c12w = all(r["passed"] for r in C12[job]["rows"] if r["tone"] == w)
            gw["C12"][str(w)] = c12w
            wh = list(withhold_grid) + glob
            if not gw["C10"][str(w)]["passed"]:
                wh.append("C10 not met on this grid and frequency")
            if not c12w:
                wh.append("C12 not met on this grid and frequency")
            if not c9["passed"][str(w)]:
                wh.append("C9 not met at this frequency")
            fb = [r["bin"] for r in c11["per_tone"][str(w)] if r["passed"] is False]
            if fb:
                wh.append(f"C11 not met at this frequency in bin(s) {', '.join(fb)}: their pieces are not read on every grid")
            per_w[w] = wh
        for ln in LINES:
            Lb = blk["lines"][ln]
            c = np.array(Lb["centers"], float)
            Dl = Lb["D"]
            rec = dict(centers=c, at={})
            for w in TONES:
                P, P1, Pf = piece_arrays(rj, ln, w), piece_arrays(rj, ln, w, "w1"), piece_arrays(rj, ln, w, "filtered")
                allowed = np.array(rj["lines"][ln]["filter_allowed"], bool)
                bins_fail = [b for b, r in enumerate(c11["per_tone"][str(w)]) if r["passed"] is False]
                ok_pre = P["ok"].copy()
                eps, terms = eps_of(P, P1, Pf, allowed, w, c11["b_gamma"][str(w)], c9["b_vac"][str(w)])
                removed = np.array([bool(ok_pre[i] and bin_of(P["y"][i], w) in bins_fail) for i in range(len(ok_pre))])
                ok = ok_pre & ~removed
                P["ok"] = ok
                y = np.where(ok, P["y"], np.nan)
                prof = [y, y + eps, y - eps]
                status = [None if not ok[i] else ("PROP" if y[i] - eps[i] >= 0 else ("EVAN" if y[i] + eps[i] < 0 else "MARGIN")) for i in range(len(ok))]
                edges = [edges0(c, p, ok, Dl) for p in prof]
                purity = [bool(ok[i] and P["p_out"][i] < P["p_in"][i]) for i in range(len(ok))]
                y_pre = np.where(ok_pre, P["y"], np.nan)
                status_pre = [None if not ok_pre[i] else ("PROP" if y_pre[i] - eps[i] >= 0 else ("EVAN" if y_pre[i] + eps[i] < 0 else "MARGIN")) for i in range(len(ok_pre))]
                rec["at"][str(w)] = dict(y=y, eps=eps, eps_terms=terms, status=status, ok=ok, reason=P["reason"], removed_by_C11=removed, y_before_C11=y_pre,
                                         status_before_C11_no_word=status_pre, ok_window1=P1["ok"] & ~removed, y_w1=np.where(P1["ok"], P1["y"], np.nan),
                                         y_filtered=np.where(allowed, Pf["y"], np.nan), c2=np.where(ok, P["c2"], np.nan), Z_re=P["Z_re"], rms=P["rms"],
                                         outside_W_share=P["p_out"] / (P["p_out"] + P["p_in"]), purity=purity, edges=edges, m2=P["m2"],
                                         domain=Lb["at"][str(w)]["domain"], bins_failed_C11=bins_fail)
                if ln in WORD_LINES:
                    yF = {lam: arr(Lb["at"][str(w)]["y_F"][str(lam)]) for lam in LAMBDAS}
                    outerF = [e["r"] for e in Lb["edges_F"]["pieces"][str(w)] if e["kind"] == "outer"]
                    rec["at"][str(w)]["kappa_read"] = kappa_bar(c, ok, y, [e["r"] for e in edges[0] if e["kind"] == "outer"][0]) if any(e["kind"] == "outer" for e in edges[0]) else None
                    rec["at"][str(w)]["kappa_F"] = {str(l): kappa_bar(c, np.isfinite(yF[l]), yF[l], outerF[0]) for l in (1.0, 0.87)} if outerF else None
            gw["lines"][ln] = rec

        def entry(reading, ln, w, kind, word, why, **kw):
            wh = per_w[w]
            e = dict(reading=reading, grid=g, line=ln, tone=w, kind=kind, word_as_read=word, why_as_read=why, **kw)
            if wh and word != "no label":
                e.update(word="no label", why="; ".join(wh))
            elif wh:
                e.update(word="no label", why=why + ("; " if why else "") + "; ".join(wh))
            else:
                e.update(word=word, why=why)
            entries.append(e)
            return e

        # W1 and W2
        A4 = gw["lines"]["A"]["at"]["4.0"]
        dom = A4["domain"]
        c = gw["lines"]["A"]["centers"]
        P1 = piece_arrays(rj, "A", 4.0, "w1")
        P1["ok"] = P1["ok"] & ~np.array(A4["removed_by_C11"], bool)
        pur1 = [bool(P1["ok"][i] and P1["p_out"][i] < P1["p_in"][i]) for i in range(len(P1["ok"]))]
        st2, info2 = statement(c, 0.5, dom, A4["ok"], np.nan_to_num(A4["y"], nan=-1e9), np.nan_to_num(A4["eps"]), A4["purity"])
        st1, info1 = statement(c, 0.5, dom, P1["ok"], np.nan_to_num(P1["y"], nan=-1e9), np.nan_to_num(A4["eps"]), pur1)
        stf, word, why = w1_word(st2, st1)
        if stf is None and st1 == st2:
            why = f"undecided ({info2['why']})"
        purity_ratio = [float(A4["outside_W_share"][i] / (1 - A4["outside_W_share"][i])) if np.isfinite(A4["outside_W_share"][i]) else None for i in dom]
        entry("W1", "A", 4.0, "statement", word, why, statement=stf, window2=st2, window1=st1, info_w2=info2, info_w1=info1, domain=dom,
              domain_s=[float(c[i]) for i in dom], purity_ratio_out_over_in=purity_ratio)
        Fst = blk["lines"]["A"]["at"]["4.0"]["statement_F"]["statement"]
        w2w, w2why = w2_word(Fst, stf)
        entry("W2", "A", 4.0, "statement", w2w, w2why, F_statement=Fst, read_statement=stf)
        # R and N
        for reading, nread, w, lines in (("R2", "N2", 4.0, ("PF", "PD")), ("R3", "N3", 5.5, ("A",)), ("R4", "N4", 5.5, ("PF", "PD"))):
            for ln in lines:
                L_ = gw["lines"][ln]
                at = L_["at"][str(w)]
                Lb = blk["lines"][ln]
                c = L_["centers"]
                Fp = Lb["edges_F"]["pieces"][str(w)]
                Fr = Lb["edges_F"]["ray"][str(w)]
                for kind in ("inner", "outer"):
                    fp = [e["r"] for e in Fp if e["kind"] == kind]
                    fr = [e["r"] for e in Fr if e["kind"] == kind]
                    if fp:
                        for r0 in fp:
                            rr = min(fr, key=lambda x: abs(x - r0)) if fr else None
                            wd, why, near = edge_entry(r0, kind, c, 0.5, at["ok"], at["edges"], F_forms=(r0, rr))
                            entry(reading, ln, w, kind, wd, why, r0=r0, F_ray=rr, nearest_read=near)
                    elif fr:
                        entry(reading, ln, w, kind, "no label", "only one form of F has the edge", r0=None, F_ray=fr[0], nearest_read=None)
                    else:
                        entry(reading, ln, w, kind, "no label", "F has no such edge on this line (beyond the line or none)", r0=None, F_ray=None, nearest_read=None)
                    reg = REG[w]["A" if ln == "A" else "plane"][kind]
                    wd, why, near = edge_entry(reg, kind, c, 0.5, at["ok"], at["edges"])
                    entry(nread, ln, w, kind, wd, why, r0=reg, nearest_read=near)
        # P
        for reading, w, lines in (("P1", 4.0, ("PF", "PD")), ("P2", 5.5, ("A",)), ("P3", 5.5, ("PF", "PD"))):
            for ln in lines:
                L_ = gw["lines"][ln]
                at = L_["at"][str(w)]
                Lb = blk["lines"][ln]
                c = L_["centers"]
                outerF = [e["r"] for e in Lb["edges_F"]["pieces"][str(w)] if e["kind"] == "outer"]
                dmask = [i in at["domain"] for i in range(len(c))]
                yF = {lam: np.nan_to_num(arr(Lb["at"][str(w)]["y_F"][str(lam)]), nan=-1e9) for lam in LAMBDAS}
                y3 = [np.nan_to_num(p, nan=-1e9) for p in (at["y"], at["y"] + at["eps"], at["y"] - at["eps"])]
                pe = p_entry(c, 0.5, at["ok"], dmask, y3, yF, outerF[0] if outerF else None)
                entry(reading, ln, w, "phase", pe.pop("word"), pe.pop("why"), **pe)
        grids[g] = gw
    out["grids"] = grids
    out["entries"] = entries
    # the budget terms and the controls by name (an input of a later task)
    out["b_gamma"] = {str(w): [dict(bin=r["bin"], pieces=r["pieces"], p90=r["p90"], value=r["value_used"], took_largest=r["took_largest"])
                               for r in c11["per_tone"][str(w)]] for w in TONES} if "trial_missing" not in c11 else None
    out["b_vac"] = c9["b_vac"]
    c9c = {}
    for w in TONES:
        rows9 = [r for r in c9["rows"] if r["bounded"] and r["tone"] == w]
        c9c[str(w)] = {k: all(r["clauses"][k] for r in rows9) for k in ("read", "cutoff", "speed", "statuses", "edges", "phase")} if rows9 else None
        if rows9:
            c9c[str(w)]["cases"] = {r["job"]: r["clauses"] for r in rows9}
    c10c = {g: {str(w): grids[g]["C10"][str(w)]["passed"] for w in TONES} for g in grids}
    out["controls"] = dict(
        C1=st0["C1"]["passed"], C2=st0["C2"]["passed"], C3=st0["C3"]["passed"], C4=out["C4"]["passed"], C5=True,
        C6=out["C6"]["passed"], C7=out["C7"]["passed"],
        C8={n: v["passed"] for n, v in out["C8"].items()},
        C9={str(w): dict(passed=c9["passed"][str(w)], clauses=c9c[str(w)]) for w in TONES},
        C10=c10c,
        C11={str(w): {r["bin"]: r["passed"] for r in c11["per_tone"][str(w)]} for w in TONES} if "trial_missing" not in c11 else None,
        C12={n: {f"{r['tone']:g} w{r['window']}": r["passed"] for r in v["rows"]} for n, v in out["C12"].items()},
        C13=st0["C13"]["passed"], C14=st0["C14"]["passed"],
        note="C5: the prediction file and its hashes were written before Ga0 (run log); every read matched the block hashes (hash_checks.log)")
    names = ("W1", "W2", "R2", "R3", "R4", "N2", "N3", "N4", "P1", "P2", "P3")
    out["words"] = {n: dict(word=reading_word([e["word"] for e in entries if e["reading"] == n]),
                            per_grid={g: reading_word([e["word"] for e in entries if e["reading"] == n and e["grid"] == g]) for g in LABELED},
                            behind=[f"{e['grid']} {e['line']} {e['kind']}" for e in entries if e["reading"] == n and e["word"] != "no label"]) for n in names}
    # descriptor: the author's read along each ray (window 2), every 0.5 in s, averaged over the rays of the line
    auth = {}
    for g in grids:
        auth[g] = {}
        rj = R[GRID_JOB[g]]
        for ln in WORD_LINES:
            auth[g][ln] = {}
            for w in TONES:
                at = rj["lines"][ln]["at"][str(w)]
                if "rays" not in at:
                    continue
                sr = arr(at["ray_s"])
                pick = [i for i in range(len(sr)) if abs(sr[i] / 0.5 - round(sr[i] / 0.5)) < 1e-6]
                kk = np.array([arr(r["k_phase"]) for r in at["rays"]])
                mo = np.array([arr(r["modulus"]) for r in at["rays"]])
                auth[g][ln][str(w)] = dict(s=sr[pick], k_phase_mean=kk[:, pick].mean(0), k_phase_min=kk[:, pick].min(0), k_phase_max=kk[:, pick].max(0),
                                           modulus_mean=mo[:, pick].mean(0))
    out["author_read_descriptor"] = auth
    # descriptors made after the words: the free-noise Re Z of M5.49 at the targets 4 and 5.5 on the same pieces
    r49 = json.loads((D.DATA / "m5_49_reads.json").read_text())["grids"]
    fn = {}
    for g in LABELED:
        fn[g] = {}
        for ln in WORD_LINES:
            L49 = r49[g]["lines"][ln]
            tg = L49["targets"]
            fn[g][ln] = {str(w): [None if row is None else row[tg.index(w)] for row in L49["K2"]] for w in TONES}
    out["free_noise_ReZ_m5_49"] = fn
    out = PR.jsonable(out)
    txt = json.dumps(out, sort_keys=True, separators=(",", ":")) + "\n"
    (D.WORK / "trial_summary.json" if trial else D.DATA / "m5_52_summary.json").write_text(txt)
    print("summary", hashlib.sha256(txt.encode()).hexdigest(), len(txt), "bytes")
    for n, v in out["words"].items():
        print(n, v["word"], v["per_grid"], v["behind"])
    return out


COL_READ, COL_F, COL_49 = "#1f4e9c", "#2a9d8f", "#d9822b"


def _panel(ax, gl, Lb, r49, w, title):
    c = np.array(gl["centers"], float)
    at = gl["at"][str(w)]
    y, eps = arr(at["y_before_C11"]), arr(at["eps"])
    yF1, yF87 = arr(Lb["at"][str(w)]["y_F"]["1.0"]), arr(Lb["at"][str(w)]["y_F"]["0.87"])
    dom = Lb["at"][str(w)]["domain"]
    if dom:
        ax.axvspan(c[dom[0]] - 0.25, c[dom[-1]] + 0.25, color="0.92", lw=0, label="near-threshold domain")
    ax.axhline(0, color="0.45", lw=0.8)
    m = np.isfinite(yF1)
    ax.fill_between(c[m], yF1[m], yF87[m], color=COL_F, alpha=0.18, lw=0, label=r"F, $\lambda$ from 0.87 to 1")
    ax.plot(c[m], yF1[m], color=COL_F, lw=1.6, label=r"F, $\lambda = 1$")
    wc = arr(r49["wc"]) if r49 is not None else None
    if wc is not None:
        mm = np.isfinite(wc)
        ax.plot(c[mm], w * w - wc[mm] ** 2, "s", ms=4, color=COL_49, label=r"M5.49 cutoffs, $\omega^2 - \omega_c^2$")
    k = np.isfinite(y)
    ax.errorbar(c[k], y[k], yerr=np.where(np.isfinite(eps[k]), eps[k], 0), fmt="o", ms=4, color=COL_READ, ecolor=COL_READ, elinewidth=1.0, capsize=2,
                label=r"read $y$ with $\varepsilon$ (before the C11 removal)")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("s along the line")
    ax.grid(alpha=0.25, lw=0.5)


def plots(summary=None, plots_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    doc = json.loads(D.PRED.read_text())
    S = json.loads(Path(summary or (D.DATA / "m5_52_summary.json")).read_text())
    r49 = json.loads((D.DATA / "m5_49_reads.json").read_text())["grids"]
    plots_dir = Path(plots_dir or (HERE.parent / "plots"))
    for g in LABELED:
        fig, axs = plt.subplots(3, 2, figsize=(11, 10.5), constrained_layout=True)
        for i, ln in enumerate(WORD_LINES):
            for j, w in enumerate(TONES):
                ax = axs[i, j]
                _panel(ax, S["grids"][g]["lines"][ln], doc["blocks"][g]["lines"][ln], r49[g]["lines"][ln], w, f"grid {g}, line {ln}, $\\omega$ = {w:g}")
                yy = arr(S["grids"][g]["lines"][ln]["at"][str(w)]["y_before_C11"])
                lo = np.nanmin(np.r_[yy[np.isfinite(yy)], -5.0])
                ax.set_ylim(max(lo - 2, -40), w * w * 0.75 if w == 5.5 else 8)
                ax.set_ylabel(r"$y = \omega^2 - m^2$")
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"M5.52, grid {g}: the local law read on the driven field (window 2) beside F; numbers only, no word (C9, C11)", fontsize=11)
        fig.savefig(plots_dir / f"m5_52_profiles_{g}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True, sharey=True)
    for j, g in enumerate(LABELED):
        e = [x for x in S["entries"] if x["reading"] == "W1" and x["grid"] == g][0]
        _panel(axs[j], S["grids"][g]["lines"]["A"], doc["blocks"][g]["lines"]["A"], r49[g]["lines"]["A"], 4.0,
               f"grid {g}, axis, $\\omega$ = 4 (W1: {e['word']})")
        axs[j].set_xlim(0, 7)
        axs[j].set_ylim(-14, 6)
    axs[0].set_ylabel(r"$y = 16 - m^2$")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.savefig(plots_dir / "m5_52_axis_w4.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("plots written", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c13", action="store_true")
    ap.add_argument("--read", nargs="*")
    ap.add_argument("--c12", nargs="*")
    ap.add_argument("--words", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--trial", action="store_true", help="with --words: partial data allowed, written to the checkpoints")
    ap.add_argument("--hash", nargs="*", default=[])
    a = ap.parse_args()
    hashes = dict(h.split("=", 1) for h in a.hash)
    if a.c13:
        r = control_c13()
        print(json.dumps(PR.jsonable(r["summary"])), r["passed"])
    if a.read:
        doc = check_hashes(hashes)
        for name in a.read:
            read_job(name, doc)
            print("read", name, flush=True)
    if a.c12:
        for name in a.c12:
            r = control_c12(name)
            (D.WORK / "c12").mkdir(exist_ok=True)
            (D.WORK / "c12" / f"{name}.json").write_text(json.dumps(PR.jsonable(r), indent=1) + "\n")
            print("C12", name, r["worst"], r["passed"], flush=True)
    if a.words:
        doc = check_hashes(hashes)
        words(doc, a.trial)
    if a.plots:
        plots()


if __name__ == "__main__":
    main()
