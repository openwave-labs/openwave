"""M5.49: the controls and the readings. The mid wave read along lines through the relaxed unit.

The prediction (m5_49_predict.py) is written first; this script takes the hash of the block of every grid it
opens and stops on a mismatch, before it opens a Fourier sum of that grid.

THE ESTIMATOR, on a piece S of a line (the nodes of S inside the domain), at a target w of the job:

    a(x, w)   = <W(x), A_job(x, w) - A_twin(x, w)>          A the six-component Fourier sum, W the mid polarization
    K^2(S, w) = - sum_S Re(conj(a) Lap_h a) / sum_S |a|^2    Lap_h the seven-point Laplacian

THE CUTOFF OF A PIECE (the rule of M5.47, its section 11). t* is the target of least K^2 over the targets of the
job. The kept targets are those above t* with 1 <= K^2 <= 4 / h^2. With four kept targets or more the line
w_t^2 = c^2 K^2 + m^2 is fitted by least squares; the piece is fitted when m^2 > 0 and every kept target lies
within 0.10 w_t^2 of the line; w_c = sqrt(m^2). A fitted piece whose kept targets fail C7 (twin power over 1 %)
or C8 (the linearized quotient more than 3 % off w^2) is treated as not fitted.

THE STATUS of a read piece against a test frequency w (4 or 5.5), with the margin d of its grid:

    UNDER   fitted, t* not the first target, w_c <= (1 - d) w
    OVER    fitted, t* not the first target, w_c >= (1 + d) w; or fewer than four kept targets and t* >= w + 0.5
    OPEN    anything else

    d = max(0.01, 2 g_a, 2 d_10, 2 |beta(4)|, 2 |beta(5.5)|)     g_a the worst gap of C6 (a), d_10 the median of
    C10 on the grid, beta(w) the bias of C6 (b). Every edge and statement is found on three profiles: w_c,
    w_c (1 + b) and w_c (1 - b), b = |beta| at the test frequency.

AN EDGE of the read: a radius where w_c(s_i), joined by straight lines between the centers of consecutive
fitted pieces no more than 2 D apart, crosses w ('outer': from under to over as s grows). Where a fitted UNDER
piece is next to an OVER piece that is not fitted, the midpoint of the two centers is an edge by its midpoint.
Two edges of one line and frequency closer than 2 D carry no label.

A RUN is a set of consecutive UNDER pieces. With c^2 and w_c of its middle piece (of two, the higher cutoff),
K^2 = (w^2 - w_c^2) / c^2 and the half wavelength is pi / K. BAND: a run at least that long. ZONE: a run, none
that long. NO BAND: nine pieces in ten of the line read (the pieces from the center to where the line stops
being read) and every read piece OVER. Otherwise undecided.

Stages (each writes its sums to the checkpoints; the words are made from the sums alone):

    python m5_49_reads.py --pure                       C2, the readers on made-up inputs
    python m5_49_reads.py --bins --hash a=.. b=..      C13, the reader on the bins of M5.47 (grids a to d)
    python m5_49_reads.py --vacuum V1 V2 Vq4 Vq55      C6, the vacuum on the pieces
    python m5_49_reads.py --pieces b --hash b=..       the sums of the unit on the pieces (after C6)
    python m5_49_reads.py --words                      the controls C7 to C14 and the readings; writes
                                                       data/m5_49_reads.json
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import m5_42_eq42 as Q
import m5_47_evolve as E
import m5_49_predict as PR

OMEGAS = PR.OMEGAS
M_MID2 = PR.M_MID2
TOL = 0.10            # the tolerance of the registration on an edge
RMIN = 2.5            # no edge under this radius is placed by pieces of 0.5
LINE_TOL = 0.10       # every kept target within 0.10 w^2 of the line
FORMS_TOL = 0.03      # the two forms of F
QP_TOL = 0.02         # C14
RAY_TOL = 0.02        # C11
DRAW_MEDIAN = 0.015   # C10
DRAW_EDGE = 0.03      # C10
MARGIN_FLOOR, MARGIN_CAP = 0.01, 0.03
GRID_JOBS = dict(a=("Ua", "Na", "Sa"), b=("Ub", "Nb", "Sb"), c=("Uc", "Nc", None), d=("Ud", "Nd", "Sd"), e=("Ue", "Ne", "Se"))
OWN = ("Sd", "Ue", "Ne", "Se", "Vq4", "Vq55")
VAC_JOBS = dict(V1=dict(block="V1", mass2=M_MID2), V2=dict(block="V2", mass2=M_MID2), Vq4=dict(block="V2", mass2=16.0), Vq55=dict(block="V2", mass2=30.25))
LABELED = ("b", "d", "e")
LINES = PR.LINES
WORD_LINES = ("A", "PF", "PD")
M47_EDGES = np.array([1.25, 1.5, 1.75, 2.0, 2.25, 2.75, 3.25, 3.75, 4.25, 5.0, 6.0, 7.0])
M47_MIN_NODES = 500
REG = {4.0: dict(A=None, plane=dict(inner=2.77, outer=6.31)), 5.5: dict(A=dict(inner=0.81, outer=8.38), plane=dict(inner=1.56, outer=9.74))}


# ----------------------------------------------------------------------------- pure functions: a piece

def cutoff(targets, K2, h):
    """The cutoff of a piece (or of a bin) from its K^2 at the targets of the job."""
    targets, K2 = np.asarray(targets, float), np.asarray(K2, float)
    K = np.where(np.isfinite(K2), K2, np.inf)
    rec = dict(t_star=None, first=False, n_kept=0, kept=[], fitted=False, wc=None, c2=None, m2=None, max_rel=None)
    if not np.isfinite(K).any():
        return rec
    cut = int(np.argmin(K))
    keep = (np.arange(len(K)) > cut) & (K >= 1.0) & (K <= 4.0 / h ** 2)
    rec.update(t_star=float(targets[cut]), first=bool(cut == 0), n_kept=int(keep.sum()), kept=targets[keep].tolist(), kept_index=np.flatnonzero(keep).tolist())
    if keep.sum() >= 4:
        w2 = targets[keep] ** 2
        c2, m2 = np.polyfit(K2[keep], w2, 1)
        rel = float((np.abs(c2 * K2[keep] + m2 - w2) / w2).max())
        rec.update(c2=float(c2), m2=float(m2), max_rel=rel, fitted=bool(m2 > 0 and rel <= LINE_TOL), wc=float(np.sqrt(m2)) if m2 > 0 else None)
    return rec


def status(rec, fit, w, margin, scale=1.0):
    if fit and not rec["first"]:
        wc = rec["wc"] * scale
        if wc <= (1 - margin) * w:
            return "UNDER"
        if wc >= (1 + margin) * w:
            return "OVER"
        return "OPEN"
    if not fit and rec["n_kept"] < 4 and rec["t_star"] is not None and rec["t_star"] >= w + 0.5:
        return "OVER"
    return "OPEN"


# ----------------------------------------------------------------------------- pure functions: a line

def read_line(centers, D, read, recs, w, margin, scale=1.0):
    """The statuses, the edges, the runs and the statement of one line at one test frequency, on one profile."""
    n = len(centers)
    fit = [bool(read[i] and recs[i].get("fitted_eff", recs[i]["fitted"])) for i in range(n)]
    wc = np.array([recs[i]["wc"] * scale if fit[i] else np.nan for i in range(n)])
    stat = [status(recs[i], fit[i], w, margin, scale) if read[i] else None for i in range(n)]
    edges = [dict(e, midpoint=False) for e in PR.crossings(centers, wc, w, max_gap=2 * D)]
    for i in range(n - 1):
        if not (read[i] and read[i + 1]):
            continue
        mid = 0.5 * (centers[i] + centers[i + 1])
        if fit[i] and stat[i] == "UNDER" and stat[i + 1] == "OVER" and not fit[i + 1]:
            edges.append(dict(r=float(mid), kind="outer", between=[i, i + 1], midpoint=True, centers=[float(centers[i]), float(centers[i + 1])]))
        if fit[i + 1] and stat[i + 1] == "UNDER" and stat[i] == "OVER" and not fit[i]:
            edges.append(dict(r=float(mid), kind="inner", between=[i, i + 1], midpoint=True, centers=[float(centers[i]), float(centers[i + 1])]))
    edges.sort(key=lambda e: e["r"])
    for e in edges:
        e["close"] = any(o is not e and abs(o["r"] - e["r"]) < 2 * D - 1e-9 for o in edges)
    runs, i = [], 0
    while i < n:
        if stat[i] == "UNDER":
            j = i
            while j + 1 < n and stat[j + 1] == "UNDER":
                j += 1
            k = j - i + 1
            mids = [i + k // 2] if k % 2 else [i + k // 2 - 1, i + k // 2]
            m = max(mids, key=lambda t: wc[t])
            c2 = recs[m]["c2"]
            K2 = (w * w - wc[m] ** 2) / c2 if c2 and c2 > 0 else np.nan
            half = float(np.pi / np.sqrt(K2)) if np.isfinite(K2) and K2 > 0 else np.inf
            runs.append(dict(first=i, last=j, pieces=k, length=k * D, start=float(centers[i] - D / 2), end=float(centers[j] + D / 2), middle=int(m),
                             wc=float(wc[m]), c2=c2, half_wavelength=half, band=bool(k * D >= half - 1e-9)))
            i = j + 1
        else:
            i += 1
    idx_read = [i for i in range(n) if read[i]]
    last = max(idx_read) if idx_read else -1
    share = len(idx_read) / (last + 1) if last >= 0 else 0.0
    if any(r["band"] for r in runs):
        statement = "BAND"
    elif runs:
        statement = "ZONE"
    elif idx_read and share >= 0.9 and all(stat[i] == "OVER" for i in idx_read):
        statement = "NO BAND"
    else:
        statement = None
    return dict(w=w, margin=margin, scale=scale, fit=fit, wc=wc, status=stat, edges=edges, runs=runs, statement=statement, read_share=share)


def covered(r0, centers, D, read, RL):
    idx, complete = PR.window(r0, centers, D)
    ok = bool(complete and idx and all(read[i] and (RL["fit"][i] or RL["status"][i] == "OVER") for i in idx))
    return ok, idx


def edge_word(r0, kind, centers, D, read, RL):
    """HELD, NOT HELD or no label for one expected edge on one profile. Returns (word, reason, read radius)."""
    same = [e for e in RL["edges"] if e["kind"] == kind]
    near = min(same, key=lambda e: abs(e["r"] - r0))["r"] if same else None
    if r0 < RMIN:
        return "no label", "the radius is under 2.5", near
    ok, _ = covered(r0, centers, D, read, RL)
    if not ok:
        return "no label", "the window is not covered", near
    lo, hi = (1 - TOL) * r0, (1 + TOL) * r0
    inside = [e for e in same if not e["midpoint"] and lo <= e["r"] <= hi]
    mid_in = [e for e in same if e["midpoint"] and all(lo <= c <= hi for c in e["centers"])]
    mid_half = [e for e in same if e["midpoint"] and sum(lo <= c <= hi for c in e["centers"]) == 1]
    found = inside + mid_in
    if found:
        best = min(found, key=lambda e: abs(e["r"] - r0))
        if all(e["close"] for e in found):
            return "no label", "two edges closer than 2 D", best["r"]
        best = min([e for e in found if not e["close"]], key=lambda e: abs(e["r"] - r0))
        return "HELD", "an edge by its midpoint" if best["midpoint"] else "", best["r"]
    if mid_half:
        return "no label", "an edge by its midpoint with one center inside the range", mid_half[0]["r"]
    return "NOT HELD", "", near


def three(fn, scales):
    """A word found on the three profiles of the bias: the same on the three, or no label."""
    out = [fn(s) for s in scales]
    if all(o[0] == out[0][0] for o in out):
        return out[0] + ([o[0] for o in out],)
    return ("no label", "inside the bias of the estimator", out[0][2], [o[0] for o in out])


def match(edges, kind, r0):
    same = [e for e in edges if e["kind"] == kind]
    return min(same, key=lambda e: abs(e["r"] - r0)) if same else None


def bands_of(edges, last_radius):
    """The bands of F on a line from its edges: (inner or 0, outer or the end of the line)."""
    out, lo = [], 0.0
    opened = True
    for e in sorted(edges, key=lambda e: e["r"]):
        if e["kind"] == "inner":
            lo, opened = e["r"], True
        else:
            if opened:
                out.append((lo, e["r"]))
            opened = False
    if opened and edges and sorted(edges, key=lambda e: e["r"])[-1]["kind"] == "inner":
        out.append((lo, last_radius))
    return out


def band_entry(F_statement, read_statements, margin, edge_words, second=None):
    """The word of a band-or-none entry. read_statements: the statement of the read on the three profiles."""
    rs = read_statements[0]
    if margin > MARGIN_CAP:
        return "no label", "the margin is above 0.03"
    if any(s != rs for s in read_statements):
        return "no label", "inside the bias of the estimator"
    if second is not None and second != rs:
        return "no label", "the second draw does not repeat the statement"
    if F_statement is None:
        return "no label", "the statement of F lies inside the margin"
    if rs is None:
        return "no label", "the read is undecided"
    if rs == "ZONE":
        return "no label", "the read says ZONE"
    if F_statement != rs:
        return "NOT HELD", f"F says {F_statement}, the read says {rs}"
    if rs == "BAND":
        if any(w == "NOT HELD" for w in edge_words):
            return "NOT HELD", "an edge of the band is NOT HELD"
        if not any(w == "HELD" for w in edge_words):
            return "no label", "both say BAND and no edge carries a word"
    return "HELD", ""


def n1_word(read_statements, margin, second=None):
    rs = read_statements[0]
    if margin > MARGIN_CAP:
        return "no label", "the margin is above 0.03"
    if any(s != rs for s in read_statements):
        return "no label", "inside the bias of the estimator"
    if second is not None and second != rs:
        return "no label", "the second draw does not repeat the statement"
    if rs == "NO BAND":
        return "HELD", ""
    if rs == "BAND":
        return "NOT HELD", "the read says BAND"
    return "no label", "the read says ZONE" if rs == "ZONE" else "the read is undecided"


def reading_word(words):
    if any(w == "NOT HELD" for w in words):
        return "NOT HELD"
    if any(w == "HELD" for w in words):
        return "HELD"
    return "no label"


# ----------------------------------------------------------------------------- the sums

def prefix(job):
    return "m5_49" if job in OWN else "m5_47"


def load_sum(job, data):
    z = np.load(data / f"{prefix(job)}_{job}_fourier.npz", allow_pickle=False)
    A = z["Ar"].astype(np.complex128)
    A.imag = z["Ai"]
    return A, z["targets"]


def lap(a, h):
    out = -6.0 * a
    for j in range(3):
        out = out + np.roll(a, -1, axis=j) + np.roll(a, 1, axis=j)
    return out / (h * h)


def densities(G, job, twin, data, k12):
    """Per node and target: |a|^2, -Re(conj a Lap a), the six-component powers of the difference and of the twin,
    and the two sums of the linearized quotient."""
    A, targets = load_sum(job, data)
    if twin is not None:
        An, _ = load_sum(twin, data)
        A -= An
    else:
        An = None
    N, h, nt = G["N"], G["h"], len(targets)
    g6 = G["W6"] * E.METRIC
    names = ["e0", "e1", "p6"] + (["p6N"] if An is not None else []) + (["k12n", "k12d"] if k12 else [])
    out = {k: np.zeros((nt, N, N, N)) for k in names}
    if k12:
        F = E.Field(N, G["hb"])
        f1, f2, p1 = np.zeros(F.shape6), np.zeros(F.shape6), np.zeros(F.shape6)
    sections = {}
    for t in range(nt):
        at = np.einsum("xyzc,xyzc->xyz", g6, A[..., t])
        if float(targets[t]) in OMEGAS:
            sections[f"sec_y_{targets[t]:g}"] = at[:, N // 2, :].copy()
            sections[f"sec_z_{targets[t]:g}"] = at[:, :, N // 2].copy()
        out["e0"][t] = np.abs(at) ** 2
        out["e1"][t] = -(np.conj(at) * lap(at, h)).real
        out["p6"][t] = (np.abs(A[..., t]) ** 2 * E.METRIC).sum(-1)
        if An is not None:
            out["p6N"][t] = (np.abs(An[..., t]) ** 2 * E.METRIC).sum(-1)
        if k12:
            for part in (np.ascontiguousarray(A[..., t].real), np.ascontiguousarray(A[..., t].imag)):
                eps = 1e-4 / max(np.abs(part).max(), 1e-300)
                F.static(G["m0"] + eps * part, f1)
                F.static(G["m0"] - eps * part, f2)
                F.momentum(G["m0"], part, p1)
                out["k12n"][t] += (part * (-(f1 - f2) / (2 * eps))).sum(-1)
                out["k12d"][t] += (part * p1).sum(-1)
    return out, targets, sections


def sum_pieces(dens, L):
    """(pieces, targets) sums of every density over the nodes of each piece inside the domain."""
    sel, idx, n = L["sel"], L["piece"], L["n"]
    return {k: np.stack([np.bincount(idx[sel], weights=v[t][sel], minlength=n) for t in range(v.shape[0])], 1) for k, v in dens.items()}


def check_hash(g, hashes, doc):
    want = hashes.get(g)
    have = PR.block_hash(doc["blocks"][g])
    if want is None or want != have or doc["hashes"][g] != have:
        raise SystemExit(f"grid {g}: the hash given ({want}) is not that of the prediction block ({have}); no Fourier sum is opened")
    return have


def geometry_matches(L, block_line):
    assert L["n"] == block_line["pieces"] and L["read"].tolist() == block_line["read"] and L["nodes_dom"].tolist() == block_line["nodes_domain"]


def stage_bins(a, doc):
    """C13: the reader given the read region and the bins of q of M5.47, on grids a to d."""
    ref = json.loads((a.data / "m5_47_reads.json").read_text())
    res = {}
    for g in ("a", "b", "c", "d"):
        check_hash(g, a.hashes, doc)
        G = PR.background(g, a.data, with_A=False)
        d = G["X"] - G["center"]
        r = np.sqrt((d * d).sum(-1))
        zz = d @ G["normal"]
        rho = np.sqrt(((d - zz[..., None] * G["normal"]) ** 2).sum(-1))
        dist = np.sqrt((rho - G["radius"]) ** 2 + zz ** 2)
        region = (dist >= PR.INSET) & (r >= G["radius"]) & G["dom"]
        idx = np.digitize(G["q"], M47_EDGES) - 1
        nb = len(M47_EDGES) - 1
        sel = region & (idx >= 0) & (idx < nb)
        U, Nn, _ = GRID_JOBS[g]
        dens, targets, sections = densities(G, U, Nn, a.data, k12=False)
        nodes = np.bincount(idx[sel], minlength=nb)[:nb]
        e0 = np.stack([np.bincount(idx[sel], weights=dens["e0"][t][sel], minlength=nb)[:nb] for t in range(len(targets))], 1)
        e1 = np.stack([np.bincount(idx[sel], weights=dens["e1"][t][sel], minlength=nb)[:nb] for t in range(len(targets))], 1)
        rows, worst = [], 0.0
        for f in ref["added_cut"]["grids"][g]["fits"]:
            b = f["bin"]
            if nodes[b] < M47_MIN_NODES:
                continue
            rec = cutoff(targets, e1[b] / e0[b], G["h"])
            if f["fitted"]:
                want = np.sqrt(f["m2"]) if f["m2"] > 0 else None
                have = np.sqrt(rec["m2"]) if rec["m2"] and rec["m2"] > 0 else None
                rel = abs(have / want - 1) if want and have else np.inf
                worst = max(worst, rel)
                rows.append(dict(bin=b, nodes=int(nodes[b]), nodes_m5_47=f["nodes"], cutoff=have, cutoff_m5_47=want, rel=rel, kept=rec["n_kept"], kept_m5_47=f["points"]))
            elif rec["n_kept"] >= 4:
                worst = np.inf
                rows.append(dict(bin=b, note="fitted here and not in M5.47"))
        res[g] = dict(rows=rows, worst_rel=worst, bins=len(rows), passed=bool(rows and worst <= 1e-6))
        print("C13", g, "bins", len(rows), "worst relative gap", worst, res[g]["passed"], flush=True)
    res["passed"] = all(v["passed"] for v in res.values())
    (a.work / "c13.json").write_text(json.dumps(PR.jsonable(res)) + "\n")
    print("C13", "MET" if res["passed"] else "NOT MET")


def stage_vacuum(a, doc):
    """The sums of the vacuum jobs on the pieces of the four lines about the center of the box and a box axis."""
    for job in a.vacuum:
        block = VAC_JOBS[job]["block"]
        G = PR.vacuum_background(block)
        z = np.load(a.data / f"{prefix(job)}_{job}_modes.npz", allow_pickle=False)
        G["W6"] = np.broadcast_to(Q.to6(E.polarization(Q.to6(z["m0"]))[0]), (G["N"],) * 3 + (6,))
        dens, targets, sections = densities(G, job, None, a.data, k12=False)
        L = PR.pieces_of(G)
        out = dict(targets=targets)
        for ln in LINES:
            geometry_matches(L[ln], doc["blocks"][block]["lines"][ln])
            for k, v in sum_pieces(dens, L[ln]).items():
                out[f"{ln}_{k}"] = v
        np.savez(a.work / f"sums_{job}.npz", **out)
        print("vacuum sums", job, "targets", len(targets), flush=True)


def stage_pieces(a, doc):
    """The sums of the unit jobs on the pieces. Runs only after C6 is met."""
    c6 = json.loads((a.work / "c6.json").read_text())
    if not (c6["a"]["passed"] and (c6["b"]["Vq4"]["passed"] or c6["b"]["Vq55"]["passed"])):
        raise SystemExit("C6 is not met; no read of the unit along a ray is made")
    for g in a.pieces:
        hsh = check_hash(g, a.hashes, doc)
        (a.work / "hash_checks.log").open("a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} grid {g}: block hash {hsh} matched before its Fourier sums were opened\n")
        G = PR.background(g, a.data, with_A=False)
        L = PR.pieces_of(G)
        U, Nn, S = GRID_JOBS[g]
        for job in [j for j in (U, S) if j is not None and (not a.jobs or j in a.jobs)]:
            if not (a.data / f"{prefix(job)}_{job}_fourier.npz").exists():
                print("pieces", g, job, "no Fourier sum yet", flush=True)
                continue
            dens, targets, sections = densities(G, job, Nn, a.data, k12=True)
            out = dict(targets=targets, block_hash=np.array(hsh), **sections)
            if job == U:
                # the field at the start and at the end of the job, projected on W, on the section y = 0 (for the figures)
                import m5_47_waves as W47
                import m5_49_waves as W49

                W49.register()
                zf = np.load(a.data / f"{prefix(job)}_{job}_fourier.npz", allow_pickle=False)
                src = (a.work / "stage") if g == "e" else a.data
                _, m_start = W47.start_state(W47.JOBS[job], src)
                g6 = G["W6"] * E.METRIC
                jy = G["N"] // 2
                out["start_y"] = np.einsum("xzc,xzc->xz", g6[:, jy], (m_start - G["m0"])[:, jy])
                out["end_y"] = np.einsum("xzc,xzc->xz", g6[:, jy], (zf["m_end"] - G["m0"])[:, jy])
            for ln in LINES:
                geometry_matches(L[ln], doc["blocks"][g]["lines"][ln])
                for k, v in sum_pieces(dens, L[ln]).items():
                    out[f"{ln}_{k}"] = v
            np.savez(a.work / f"sums_{job}.npz", **out)
            print("pieces", g, job, "written", flush=True)


# ----------------------------------------------------------------------------- the tables of a job

def piece_records(z, ln, h, twin=True):
    """Per piece of a line: the cutoff record, with C7 and C8 at its kept targets."""
    t = z["targets"]
    with np.errstate(divide="ignore", invalid="ignore"):
        K2 = z[f"{ln}_e1"] / z[f"{ln}_e0"]
        c7 = z[f"{ln}_p6N"] / z[f"{ln}_p6"] if twin else None
        c8 = np.abs(z[f"{ln}_k12n"] / z[f"{ln}_k12d"] / t ** 2 - 1) if twin else None
        outside = 1 - z[f"{ln}_e0"] / z[f"{ln}_p6"]
    recs = []
    for i in range(K2.shape[0]):
        r = cutoff(t, K2[i], h)
        r.update(K2=K2[i], outside_W=outside[i], power=z[f"{ln}_e0"][i])
        if twin:
            k = r.get("kept_index", [])
            r["c7"] = float(np.nanmax(c7[i][k])) if r["fitted"] and k else None
            r["c8"] = float(np.nanmax(c8[i][k])) if r["fitted"] and k else None
            r["c7_ok"] = bool(r["c7"] is None or r["c7"] <= 0.01)
            r["c8_ok"] = bool(r["c8"] is None or r["c8"] <= 0.03)
            r["fitted_eff"] = bool(r["fitted"] and r["c7_ok"] and r["c8_ok"])
        else:
            r["fitted_eff"] = r["fitted"]
        recs.append(r)
    return recs


def control_c6(a, doc):
    """C6: the vacuum on the pieces. (a) V2 (V1 tabled); (b) Vq4 and Vq55, with the bias beta."""
    out = dict(a={}, b={})
    for job, spec in VAC_JOBS.items():
        f = a.work / f"sums_{job}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        blk = doc["blocks"][spec["block"]]
        m = float(np.sqrt(spec["mass2"]))
        wcs, n_read, n_fit, all_over, edges, per_line = [], 0, 0, True, 0, {}
        for ln in LINES:
            Lb = blk["lines"][ln]
            recs = piece_records(z, ln, blk["spacing"], twin=False)
            read = Lb["read"]
            fit = [r["wc"] for r, rd in zip(recs, read) if rd and r["fitted"]]
            wcs += fit
            n_read += sum(read)
            n_fit += len(fit)
            per_line[ln] = dict(read=int(sum(read)), fitted=len(fit), cutoffs=[r["wc"] if rd and r["fitted"] else None for r, rd in zip(recs, read)],
                                t_star=[r["t_star"] for r in recs], kept=[r["n_kept"] for r in recs])
            if job in ("V1", "V2"):
                for w in OMEGAS:
                    RL = read_line(Lb["centers"], Lb["D"], read, recs, w, MARGIN_CAP)
                    all_over &= all(s == "OVER" for s, rd in zip(RL["status"], read) if rd)
                    edges += len(RL["edges"])
        wcs = np.array(wcs)
        gap = float(np.abs(wcs / m - 1).max()) if wcs.size else None
        rec = dict(job=job, expected=m, read=n_read, fitted=n_fit, share_fitted=n_fit / max(n_read, 1), worst_gap=gap,
                   median=float(np.median(wcs)) if wcs.size else None, beta=float(np.median(wcs) / m - 1) if wcs.size else None, lines=per_line)
        if job in ("V1", "V2"):
            rec.update(every_read_piece_over=bool(all_over), edges=edges,
                       passed=bool(wcs.size and gap <= 0.01 and rec["share_fitted"] >= 0.9 and all_over and edges == 0))
            out["a"][job] = rec
        else:
            rec.update(passed=bool(wcs.size and gap <= 0.01 and rec["share_fitted"] >= 0.9))
            out["b"][job] = rec
        print("C6", job, {k: rec[k] for k in ("expected", "read", "fitted", "worst_gap", "median", "beta", "passed")}, flush=True)
    if "V2" in out["a"]:
        out["a"]["passed"] = out["a"]["V2"]["passed"]
    if out["a"].get("passed") is not None and len(out["b"]) == 2:
        (a.work / "c6.json").write_text(json.dumps(PR.jsonable(out)) + "\n")
    return out


# ----------------------------------------------------------------------------- the words

def margins(c6, c10_medians):
    # V1 (spacing 0.5) is tabled: with K^2 <= 4 / h^2 = 16 it keeps fewer than four targets, so the gap of V2 stands in for it
    v1 = c6["a"].get("V1", {}).get("worst_gap")
    g_a = {0.25: c6["a"]["V2"]["worst_gap"], 0.5: v1 if v1 is not None else c6["a"]["V2"]["worst_gap"]}
    b4, b55 = abs(c6["b"]["Vq4"]["beta"]), abs(c6["b"]["Vq55"]["beta"])
    return g_a, {4.0: b4, 5.5: b55}


def line_read_all(Lb, recs, margin, beta):
    """The read of a line at the two frequencies on the three profiles of the bias."""
    out = {}
    for w in OMEGAS:
        b = beta[w]
        out[w] = [read_line(Lb["centers"], Lb["D"], Lb["read"], recs, w, margin, s) for s in (1.0, 1.0 + b, 1.0 - b)]
    return out


def words(a, doc):
    c6 = control_c6(a, doc)
    c13 = json.loads((a.work / "c13.json").read_text())
    pure = json.loads((a.work / "c2.json").read_text())
    c1 = json.loads((a.work / "c1.json").read_text()) if (a.work / "c1.json").exists() else None
    g_a, beta = margins(c6, None)
    jobs_json = {}
    for j in ("Sd", "Ue", "Ne", "Se", "Vq4", "Vq55"):
        f = a.data / f"m5_49_{j}.json"
        if f.exists():
            jobs_json[j] = json.loads(f.read_text())
    c9 = {j: dict(drift=s["drift"], perturbation_energy=s["perturbation_energy"], ratio=s["drift"] / s["perturbation_energy"],
                  passed=bool(s["drift"] / s["perturbation_energy"] <= 0.02)) for j, s in jobs_json.items() if s["perturbation_energy"] > 0}
    res = dict(task="M5.49", omegas=list(OMEGAS), registered={str(k): v for k, v in REG.items()}, C1=c1, C2=pure, C6=c6, C9=c9, C13=c13,
               prediction_hashes=doc["hashes"], beta={str(k): v for k, v in beta.items()}, grids={}, entries=[])
    glob_fail = [k for k, ok in (("C1", c1 is None or c1["passed"]), ("C2", pure["passed"]), ("C13", c13["passed"]), ("C6 (a)", c6["a"]["passed"])) if not ok]
    freq_fail = {4.0: not c6["b"]["Vq4"]["passed"], 5.5: not c6["b"]["Vq55"]["passed"]}
    for g, (U, Nn, S) in GRID_JOBS.items():
        if g not in doc["blocks"] or not (a.work / f"sums_{U}.npz").exists():
            continue
        blk = doc["blocks"][g]
        h = blk["spacing"]
        zU = np.load(a.work / f"sums_{U}.npz", allow_pickle=False)
        assert str(zU["block_hash"]) == doc["hashes"][g]
        zS = np.load(a.work / f"sums_{S}.npz", allow_pickle=False) if S and (a.work / f"sums_{S}.npz").exists() else None
        recU = {ln: piece_records(zU, ln, h) for ln in LINES}
        recS = {ln: piece_records(zS, ln, h) for ln in LINES} if zS is not None else None
        # C10: the median over the pieces fitted in both draws
        c10 = dict(job=S, median=None, per_line={})
        if recS is not None:
            rel_all = []
            for ln in LINES:
                rd = blk["lines"][ln]["read"]
                rel = [abs(u["wc"] - s["wc"]) / (0.5 * (u["wc"] + s["wc"])) for u, s, r in zip(recU[ln], recS[ln], rd) if r and u["fitted_eff"] and s["fitted_eff"]]
                c10["per_line"][ln] = dict(pieces=len(rel), median=float(np.median(rel)) if rel else None, largest=float(max(rel)) if rel else None)
                rel_all += rel
            c10.update(pieces=len(rel_all), median=float(np.median(rel_all)) if rel_all else None, median_passed=bool(rel_all and np.median(rel_all) <= DRAW_MEDIAN))
        d10 = c10["median"] if c10["median"] is not None else 0.0
        gap = g_a[round(h, 6)]
        margin = max(MARGIN_FLOOR, 2 * gap, 2 * d10, 2 * beta[4.0], 2 * beta[5.5])
        info = dict(base=blk["base"], spacing=h, half_box=blk["half_box"], job=U, twin=Nn, second=S, labeled=bool(g in LABELED), margin=margin,
                    margin_terms=dict(floor=MARGIN_FLOOR, twice_C6a=2 * gap, twice_C10=2 * d10, twice_beta4=2 * beta[4.0], twice_beta55=2 * beta[5.5]),
                    C10=c10, C3=blk["C3"], loop=blk["loop"], lines={}, C7=dict(points=0, worst=0.0, pieces_failed=0), C8=dict(points=0, worst=0.0, pieces_failed=0),
                    C9={j: c9[j] for j in (U, S) if j in c9}, C11=[], C12=[], C14=[], F={})
        RLU, RLS = {}, {}
        for ln in LINES:
            Lb = blk["lines"][ln]
            RLU[ln] = line_read_all(Lb, recU[ln], margin, beta)
            RLS[ln] = line_read_all(Lb, recS[ln], margin, beta) if recS is not None else None
            for r, rd in zip(recU[ln], Lb["read"]):
                if rd and r["fitted"]:
                    for key in ("C7", "C8"):
                        v = r[key.lower()]
                        info[key]["points"] += len(r["kept"])
                        info[key]["worst"] = max(info[key]["worst"], v)
                        info[key]["pieces_failed"] += int(not r[key.lower() + "_ok"])
            info["lines"][ln] = dict(
                centers=Lb["centers"], read=Lb["read"], nodes_domain=Lb["nodes_domain"], q=Lb["q"], mF=Lb["mF"], mF_qP=Lb["mF_qP"], mF_prime=Lb["mF_prime"], C=Lb["C"],
                wc=[r["wc"] if rd and r["fitted_eff"] else None for r, rd in zip(recU[ln], Lb["read"])],
                wc_raw=[r["wc"] for r in recU[ln]], c2=[r["c2"] for r in recU[ln]], t_star=[r["t_star"] for r in recU[ln]], first=[r["first"] for r in recU[ln]],
                kept=[r["n_kept"] for r in recU[ln]], max_rel=[r["max_rel"] for r in recU[ln]], fitted=[r["fitted"] for r in recU[ln]],
                c7=[r["c7"] for r in recU[ln]], c8=[r["c8"] for r in recU[ln]],
                wc_second=[r["wc"] if rd and r["fitted_eff"] else None for r, rd in zip(recS[ln], Lb["read"])] if recS is not None else None,
                targets=zU["targets"], K2=np.array([r["K2"] for r in recU[ln]]),
                at={str(w): dict(status=RLU[ln][w][0]["status"], status_plus=RLU[ln][w][1]["status"], status_minus=RLU[ln][w][2]["status"],
                                 edges=RLU[ln][w][0]["edges"], edges_plus=RLU[ln][w][1]["edges"], edges_minus=RLU[ln][w][2]["edges"],
                                 runs=RLU[ln][w][0]["runs"], statement=[x["statement"] for x in RLU[ln][w]], read_share=RLU[ln][w][0]["read_share"],
                                 second=dict(edges=RLS[ln][w][0]["edges"], runs=RLS[ln][w][0]["runs"], statement=RLS[ln][w][0]["statement"],
                                             status=RLS[ln][w][0]["status"]) if RLS[ln] is not None else None,
                                 K2=np.array([r["K2"][int(np.argmin(np.abs(zU["targets"] - w)))] for r in recU[ln]]),
                                 outside_W=np.array([r["outside_W"][int(np.argmin(np.abs(zU["targets"] - w)))] for r in recU[ln]]),
                                 power=np.array([r["power"][int(np.argmin(np.abs(zU["targets"] - w)))] for r in recU[ln]])) for w in OMEGAS})
        # the entries of the two tables
        withheld = list(glob_fail)
        for j in (U, S):
            if j in c9 and not c9[j]["passed"] and j == U:
                withheld.append(f"C9 ({j})")
        entries = []
        f_edges_by = {}
        for w in OMEGAS:
            for ln in WORD_LINES:
                Lb = blk["lines"][ln]
                cen, D, read = Lb["centers"], Lb["D"], Lb["read"]
                base = RLU[ln][w][0]
                sec = RLS[ln][w][0] if RLS[ln] is not None else None
                Rname = {(4.0, "A"): "R1", (4.0, "PF"): "R2", (4.0, "PD"): "R2", (5.5, "A"): "R3", (5.5, "PF"): "R4", (5.5, "PD"): "R4"}[(w, ln)]
                Nname = "N" + Rname[1]

                def one_edge(r0, kind):
                    fn = lambda k: edge_word(r0, kind, cen, D, read, RLU[ln][w][k])
                    word, why, rr, trio = three(lambda k: fn(k), (0, 1, 2))
                    s_word, s_r = None, None
                    if sec is not None:
                        s_word, _, s_r = edge_word(r0, kind, cen, D, read, sec)
                    if word == "HELD" and sec is not None:
                        again = [e for e in sec["edges"] if e["kind"] == kind and abs(e["r"] - rr) <= DRAW_EDGE * rr]
                        if not again:
                            word, why = "no label", "the edge is not found again within 3 % in the second draw"
                    # C11 on the window
                    idx, _ = PR.window(r0, cen, D)
                    worst = 0.0
                    for i in idx:
                        if read[i]:
                            for k in range(len(Lb["ray_q"])):
                                if Lb["ray_nodes"][k][i] > 0 and Lb["ray_q"][k][i] is not None:
                                    worst = max(worst, abs(Lb["ray_q"][k][i] / Lb["q"][i] - 1))
                    c11 = dict(line=ln, w=w, r0=r0, kind=kind, worst=worst, passed=bool(worst <= RAY_TOL))
                    info["C11"].append(c11)
                    if word in ("HELD", "NOT HELD") and not c11["passed"]:
                        word, why = "no label", "C11: the rays of the line differ by more than 2 % in the window"
                    return word, why, rr, trio, s_word, s_r

                # ---- the R column
                Fp = Lb["edges"]["pieces"][str(w)]
                Fr = Lb["edges"]["ray"][str(w)]
                FpP = Lb["edges"]["pieces_qP"][str(w)]
                Fs = PR.f_statement(Lb["mF"], read, Lb["ray_form"]["mF"], w, margin)
                FsP = PR.f_statement(Lb["mF_qP"], read, Lb["ray_form"]["mF_qP"], w, margin)
                info["F"][f"{ln} {w}"] = dict(pieces=Fp, ray=Fr, pieces_qP=FpP, ray_qP=Lb["edges"]["ray_qP"][str(w)], statement=Fs, statement_qP=FsP,
                                              least=Lb["least_mF"])
                f_edges_by[(w, ln)] = Fp
                stm = [x["statement"] for x in RLU[ln][w]]
                # The routing of an R entry (ruling on R1 after the audit of the record): the edges of F are scored by the
                # edge rule only where F says BAND by the definition of section 5 (two consecutive pieces with m_F at most
                # (1 - margin) w, and a band in the ray form); everywhere else the line's one R entry is its band-or-none
                # entry. The run first routed by a literal crossing of w on the pieces (edges whenever F on the pieces had
                # one); the word of that routing is kept in the field as_run_literal_crossing.
                edge_mode = bool(Fp) and Fs == "BAND"
                edge_entries = []
                edge_ws = []
                for n_e, e in enumerate(Fp):
                    r0, kind = e["r"], e["kind"]
                    word, why, rr, trio, s_word, s_r = one_edge(r0, kind)
                    mr = match(Fr, kind, r0)
                    mP = match(FpP, kind, r0)
                    c14 = dict(line=ln, w=w, kind=kind, r_q=r0, r_qP=mP["r"] if mP else None, rel=abs(mP["r"] - r0) / r0 if mP else None,
                               passed=bool(mP is not None and abs(mP["r"] - r0) / r0 <= QP_TOL))
                    info["C14"].append(c14)
                    if word in ("HELD", "NOT HELD"):
                        if mr is None:
                            word, why = "no label", "only the piece form of F has the edge"
                        elif abs(mr["r"] - r0) / r0 > FORMS_TOL:
                            word, why = "no label", "the two forms of F differ by more than 3 % of the radius"
                        elif not c14["passed"]:
                            word, why = "no label", "C14: F with q_P and F with q differ by more than 2 % of the radius"
                    edge_entries.append(dict(reading=Rname, column="R", grid=g, line=ln, w=w, what=f"{kind} edge" + (f" {n_e + 1}" if len(Fp) > 2 else ""), r0=r0,
                                             r0_ray=mr["r"] if mr else None, r0_qP=mP["r"] if mP else None, word=word, why=why, read=rr, profiles=trio, second_word=s_word,
                                             second_read=s_r, as_run_literal_crossing=word))
                    edge_ws.append(word)
                if edge_mode:
                    entries.extend(edge_entries)
                for e in Fr:
                    if match(Fp, e["kind"], e["r"]) is None:
                        entries.append(dict(reading=Rname, column="R", grid=g, line=ln, w=w, what=f"{e['kind']} edge", r0=None, r0_ray=e["r"], r0_qP=None, word="no label",
                                            why="only the ray form of F has the edge", read=None, profiles=None, second_word=None, second_read=None))
                rd_idx = [i for i in range(len(read)) if read[i]]
                if rd_idx and Lb["mF"][rd_idx[-1]] < w and not any(e["kind"] == "outer" and e["r"] > cen[rd_idx[-1]] for e in Fr):
                    entries.append(dict(reading=Rname, column="R", grid=g, line=ln, w=w, what="outer edge", r0=None, r0_ray=None, r0_qP=None, word="no label",
                                        why="F is under the frequency out to where the line stops being read: its outer edge lies beyond the line", read=None, profiles=None,
                                        second_word=None, second_read=None))
                if not edge_mode:
                    c14s = dict(line=ln, w=w, kind="statement", statement_q=Fs, statement_qP=FsP, passed=bool(Fs == FsP))
                    info["C14"].append(c14s)
                    word, why = band_entry(Fs, stm, margin, [], sec["statement"] if sec is not None else None)
                    if word in ("HELD", "NOT HELD") and not c14s["passed"]:
                        word, why = "no label", "C14: the statement of F differs between q and q_P"
                    if word in ("HELD", "NOT HELD") and recS is not None and not c10["median_passed"]:
                        word, why = "no label", "C10: the median of the second draw is above 0.015"
                    literal = ([dict(what=x["what"], r0=x["r0"], read=x["read"], word=x["word"], why=x["why"]) for x in edge_entries] if Fp else word)
                    entries.append(dict(reading=Rname, column="R", grid=g, line=ln, w=w, what="band or none", r0=None, F_statement=Fs, read_statement=stm, word=word, why=why,
                                        runs=base["runs"], second_statement=sec["statement"] if sec is not None else None, as_run_literal_crossing=literal))
                else:
                    bands = bands_of(Fp, Lb["read_to"])
                    for r in base["runs"]:
                        if r["band"] and all(r["end"] < 0.9 * lo or r["start"] > 1.1 * hi for lo, hi in bands):
                            entries.append(dict(reading=Rname, column="R", grid=g, line=ln, w=w, what="band not predicted", r0=None, word="NOT HELD",
                                                why=f"a BAND of the read from {r['start']:g} to {r['end']:g} outside every band of F", runs=[r]))
                # ---- the N column
                if (w, ln) == (4.0, "A"):
                    word, why = n1_word(stm, margin, sec["statement"] if sec is not None else None)
                    if word in ("HELD", "NOT HELD") and recS is not None and not c10["median_passed"]:
                        word, why = "no label", "C10: the median of the second draw is above 0.015"
                    entries.append(dict(reading="N1", column="N", grid=g, line=ln, w=w, what="band or none", r0=None, registered="no band", read_statement=stm, word=word,
                                        why=why, runs=base["runs"], second_statement=sec["statement"] if sec is not None else None))
                else:
                    regs = REG[w]["A"] if ln == "A" else REG[w]["plane"]
                    for kind in ("outer", "inner"):
                        r0 = regs[kind]
                        word, why, rr, trio, s_word, s_r = one_edge(r0, kind)
                        entries.append(dict(reading=Nname, column="N", grid=g, line=ln, w=w, what=f"{kind} edge", r0=r0, word=word, why=why, read=rr, profiles=trio,
                                            second_word=s_word, second_read=s_r))
        # C12, the swap: two lines whose edges of F at one w differ by more than 10 % of their mean
        for w in OMEGAS:
            for kind in ("outer", "inner"):
                for i1, l1 in enumerate(WORD_LINES):
                    for l2 in WORD_LINES[i1 + 1:]:
                        e1, e2 = match(f_edges_by[(w, l1)], kind, 1e9 if kind == "outer" else 0.0), match(f_edges_by[(w, l2)], kind, 1e9 if kind == "outer" else 0.0)
                        if e1 is None or e2 is None or abs(e1["r"] - e2["r"]) <= 0.1 * 0.5 * (e1["r"] + e2["r"]):
                            continue
                        r1, r2 = match(RLU[l1][w][0]["edges"], kind, e1["r"]), match(RLU[l2][w][0]["edges"], kind, e2["r"])
                        rec = dict(w=w, kind=kind, lines=[l1, l2], F=[e1["r"], e2["r"]], read=[r1["r"] if r1 else None, r2["r"] if r2 else None])
                        if r1 is None or r2 is None:
                            rec.update(passed=None, note="a read edge is missing; the edge rule already scores it")
                        else:
                            same = (r1["r"] - r2["r"]) * (e1["r"] - e2["r"]) > 0
                            rec.update(passed=bool(same and abs(r1["r"] - r2["r"]) >= 0.5 * abs(e1["r"] - e2["r"])))
                        info["C12"].append(rec)
                        if rec["passed"] is False:
                            for en in entries:
                                if en["column"] == "R" and en["w"] == w and en["line"] in (l1, l2) and en["what"].startswith(kind) and en["word"] in ("HELD", "NOT HELD"):
                                    en["word"], en["why"] = "no label", "C12: the read does not tell the two lines apart"
        for en in entries:
            if en["word"] in ("HELD", "NOT HELD"):
                if withheld:
                    en["word"], en["why"] = "no label", "withheld: " + ", ".join(withheld) + " not met"
                elif freq_fail[en["w"]]:
                    en["word"], en["why"] = "no label", "withheld: C6 (b) not met at this frequency"
            en["labeled"] = bool(g in LABELED)
        info["words"] = {rd: reading_word([en["word"] for en in entries if en["reading"] == rd]) for rd in ("R1", "R2", "R3", "R4", "N1", "N2", "N3", "N4")}
        res["grids"][g] = info
        res["entries"] += entries
        print(f"grid {g}: margin {margin:.4f} C10 {c10.get('median')} words {info['words']}", flush=True)
    lab = [en for en in res["entries"] if en["labeled"]]
    res["words"] = {rd: reading_word([en["word"] for en in lab if en["reading"] == rd]) for rd in ("R1", "R2", "R3", "R4", "N1", "N2", "N3", "N4")}
    res["withheld_globally"] = glob_fail
    print("words over the labeled grids:", res["words"])
    (a.data / "m5_49_reads.json").write_text(json.dumps(PR.jsonable(res), separators=(",", ":")) + "\n")
    print("written", a.data / "m5_49_reads.json", (a.data / "m5_49_reads.json").stat().st_size, "bytes")
    return res


# ----------------------------------------------------------------------------- C2: the readers on made-up inputs

def control_pure(a):
    """C2. Every reader on an input with a known answer, and each branch of the rules once."""
    T17 = np.array([3.0 + 0.25 * i for i in range(13)] + [6.5, 7.0, 7.5, 8.0])
    T23 = np.concatenate([T17, [8.25, 8.5, 8.75, 9.0, 10.0, 11.0]])
    checks = []

    def ok(name, cond, got=None):
        checks.append(dict(check=name, passed=bool(cond), got=got))

    # (a) profiles given as functions of s on pieces of 0.5
    s = np.arange(0.25, 13.0, 0.5)
    sr = np.arange(0.0, 13.0, 0.125)
    read = [True] * len(s)
    mk = lambda q: np.sqrt(M_MID2 / q)
    tail = mk(1 + 88 / s ** 2)
    e4, e55 = PR.crossings(s, tail, 4.0), PR.crossings(s, tail, 5.5)
    ok("a1 tail q = 1 + 88/s^2: outer edge at 5.65 at w = 4", len(e4) == 1 and e4[0]["kind"] == "outer" and abs(e4[0]["r"] - 5.65) <= 0.02, e4)
    ok("a2 tail: outer edge at 9.45 at w = 5.5", len(e55) == 1 and e55[0]["kind"] == "outer" and abs(e55[0]["r"] - 9.45) <= 0.02, e55)
    flat = mk(np.full(len(s), 1.5))
    ok("a3 flat q = 1.5: no edge", not PR.crossings(s, flat, 4.0) and not PR.crossings(s, flat, 5.5))
    for plateau, want in ((3.60, {0.01: "NO BAND", 0.03: None}), (3.95, {0.01: "BAND", 0.03: None})):
        qq = np.where((s >= 1.5) & (s <= 4.5), plateau, 2.0)
        qr = np.where((sr >= 1.5) & (sr <= 4.5), plateau, 2.0)
        for dl, st in want.items():
            got = PR.f_statement(mk(qq), read, mk(qr), 4.0, dl)
            ok(f"a4 plateau {plateau} at margin {dl}: {st}", got == st, got)
    # (b) tables of K^2 against the targets
    h = 0.25

    def table(targets, wc, c2=0.87):
        """K^2 on a line of slope c2 above the cutoff, large under it. The first target above the cutoff has the least
        K^2: it is t*, and the kept targets are those above it."""
        return np.where(targets > wc, (targets ** 2 - wc ** 2) / c2, 20.0 + targets)

    r = cutoff(T17, table(T17, 5.5), h)
    ok("b1 a line of slope 0.87 and cutoff 5.5: cutoff within 0.01, kept targets as built",
       r["fitted"] and abs(r["wc"] - 5.5) <= 0.01 and r["kept"] == [6.0, 6.5, 7.0, 7.5, 8.0] and abs(r["c2"] - 0.87) < 1e-9, dict(wc=r["wc"], kept=r["kept"], t_star=r["t_star"]))
    r3 = cutoff(T17, table(T17, 6.4), h)
    ok("b2 three kept targets: not fitted", not r3["fitted"] and r3["n_kept"] == 3 and r3["t_star"] == 6.5, dict(kept=r3["kept"]))
    rf = cutoff(T17, (T17 ** 2 - 2.0 ** 2) / 0.87, h)
    ok("b3 least K^2 at the first target: OPEN", rf["first"] and status(rf, rf["fitted"], 4.0, 0.01) == "OPEN" and status(rf, rf["fitted"], 5.5, 0.01) == "OPEN", dict(t_star=rf["t_star"], wc=rf["wc"]))
    r23 = cutoff(T23, table(T23, 7.75, c2=1.0), h)
    ok("b4 twenty-three targets, cutoff 7.75, slope 1: the rule on its own targets",
       r23["fitted"] and abs(r23["wc"] - 7.75) <= 0.01 and r23["t_star"] == 8.0 and r23["kept"] == [8.25, 8.5, 8.75, 9.0, 10.0, 11.0],
       dict(wc=r23["wc"], kept=r23["kept"]))
    rb = cutoff(T17, np.where(T17 > 5.5, (T17 ** 2 - 30.25) / 0.87 * (1 + 0.5 * np.sin(7 * T17)), 30.0), h)
    ok("b5 kept targets more than 0.10 w^2 off the line: not fitted", not rb["fitted"] and rb["n_kept"] >= 4, dict(max_rel=rb["max_rel"]))
    r_few = cutoff(T17, table(T17, 7.2), h)
    ok("b6 fewer than four kept targets and t* at least w + 0.5: OVER", status(r_few, False, 5.5, 0.01) == "OVER" and status(r_few, False, 4.0, 0.01) == "OVER", dict(t_star=r_few["t_star"], kept=r_few["kept"]))
    r_few2 = cutoff(T17, table(T17, 7.2), h)
    r_few2["t_star"] = 5.75
    ok("b7 fewer than four kept targets and t* under w + 0.5: OPEN", status(r_few2, False, 5.5, 0.01) == "OPEN")

    # (c) made-up profiles of w_c, with slopes, for every branch
    def recs_of(wc, c2=0.87, fitted=None, t_star=None, kept=None):
        out = []
        for i, v in enumerate(wc):
            f = True if fitted is None else fitted[i]
            out.append(dict(wc=float(v) if v is not None else None, c2=c2, fitted=f, fitted_eff=f, first=False, n_kept=(6 if f else (2 if kept is None else kept[i])),
                            t_star=(float(v) if v is not None else None) if t_star is None else t_star[i], kept=[], m2=None, max_rel=0.0))
        return out

    D = 0.5
    cen = np.arange(0.25, 12.0, D)
    rd = [True] * len(cen)
    prof = lambda r_out, lo=3.0, hi=7.0, r_in=None: np.where(cen < r_out, lo, hi) if r_in is None else np.where((cen > r_in) & (cen < r_out), lo, hi)
    # a smooth outer edge at 6.0: w_c rises linearly through 4 at s = 6
    smooth = 4.0 + 0.4 * (cen - 6.0)
    RL = read_line(cen, D, rd, recs_of(smooth), 4.0, 0.01)
    ok("c1 HELD: an outer edge at 6.0 against r0 = 6.2", edge_word(6.2, "outer", cen, D, rd, RL)[0] == "HELD", [e["r"] for e in RL["edges"]])
    ok("c2 NOT HELD: the same read against r0 = 8.0", edge_word(8.0, "outer", cen, D, rd, RL)[0] == "NOT HELD")
    ok("c3 no label by the cover: r0 = 11.5, the window runs past the line", edge_word(11.5, "outer", cen, D, rd, RL)[1] == "the window is not covered")
    rd2 = list(rd)
    rd2[12] = False
    RL2 = read_line(cen, D, rd2, recs_of(smooth), 4.0, 0.01)
    ok("c4 no label by the cover: a piece of the window not read", edge_word(6.2, "outer", cen, D, rd2, RL2)[1] == "the window is not covered")
    ok("c5 no label by the radius of 2.5", edge_word(2.2, "inner", cen, D, rd, RL)[1] == "the radius is under 2.5")
    wig = np.where((cen < 5.5) | (np.abs(cen - 6.25) < 0.1), 3.5, 4.5)   # edges at 5.5 (outer), 6.0 (inner), 6.5 (outer)
    RLw = read_line(cen, D, rd, recs_of(wig), 4.0, 0.01)
    ok("c6 no label by two edges too close", edge_word(6.0, "outer", cen, D, rd, RLw)[1] == "two edges closer than 2 D" and len(RLw["edges"]) == 3,
       [(e["r"], e["kind"], e["close"]) for e in RLw["edges"]])
    # the bias: an edge at 6.0 against r0 = 6.6 is at the lower end of the range; a profile shifted up moves it out
    near = 4.0 + 0.4 * (cen - 5.96)
    recs_n = recs_of(near)
    trio = three(lambda sc: edge_word(6.6, "outer", cen, D, rd, read_line(cen, D, rd, recs_n, 4.0, 0.01, sc)), (1.0, 1.03, 0.97))
    ok("c7 no label by the bias: the word differs on the three profiles", trio[1] == "inside the bias of the estimator", trio[3])
    # an edge by its midpoint: a fitted UNDER piece next to an OVER piece that is not fitted
    wc_m = [3.0 if c < 6.0 else None for c in cen]
    fit_m = [c < 6.0 for c in cen]
    ts_m = [3.0 if c < 6.0 else 6.5 for c in cen]
    RLm = read_line(cen, D, rd, recs_of(wc_m, fitted=fit_m, t_star=ts_m), 4.0, 0.01)
    em = [e for e in RLm["edges"] if e["midpoint"]]
    ok("c8 an edge by its midpoint is found", len(em) == 1 and em[0]["kind"] == "outer" and abs(em[0]["r"] - 6.0) < 1e-9, em)
    ok("c9 midpoint, both centers in the range: HELD", edge_word(6.0, "outer", cen, D, rd, RLm)[0] == "HELD")
    ok("c10 midpoint, one center in the range: no label", edge_word(6.5, "outer", cen, D, rd, RLm)[1] == "an edge by its midpoint with one center inside the range")
    ok("c11 midpoint, both centers outside the range: NOT HELD", edge_word(8.0, "outer", cen, D, rd, RLm)[0] == "NOT HELD")
    # statuses
    one = recs_of([3.9, 4.1, 4.0, 3.0, 5.0])
    ok("c12 UNDER, OVER and OPEN", [status(r, True, 4.0, 0.01) for r in one] == ["UNDER", "OVER", "OPEN", "UNDER", "OVER"])
    # BAND, ZONE, NO BAND, undecided
    long_run = prof(6.0, lo=3.0)            # twelve pieces under: length 6; half wavelength pi / sqrt((16 - 9) / 0.87) = 1.11
    RLb = read_line(cen, D, rd, recs_of(long_run), 4.0, 0.01)
    ok("c13 BAND: a run longer than its half wavelength", RLb["statement"] == "BAND" and abs(RLb["runs"][0]["half_wavelength"] - np.pi / np.sqrt(7 / 0.87)) < 1e-9, RLb["runs"])
    short = np.where((cen > 3.0) & (cen < 4.0), 3.9, 7.0)   # two pieces under at 3.9: half wavelength pi / sqrt(0.79 / 0.87) = 3.3
    RLz = read_line(cen, D, rd, recs_of(short), 4.0, 0.01)
    ok("c14 ZONE: a run shorter than its half wavelength", RLz["statement"] == "ZONE" and RLz["runs"][0]["length"] == 1.0 and RLz["runs"][0]["half_wavelength"] > 3.0, RLz["runs"])
    RLn = read_line(cen, D, rd, recs_of(np.full(len(cen), 7.0)), 4.0, 0.01)
    ok("c15 NO BAND: every read piece OVER", RLn["statement"] == "NO BAND")
    RLu = read_line(cen, D, rd, recs_of(np.where(cen < 2.0, 4.0, 7.0)), 4.0, 0.01)
    ok("c16 undecided: OPEN pieces and no run", RLu["statement"] is None)
    rd3 = [c > 2.0 for c in cen]
    RLs = read_line(cen, D, rd3, recs_of(np.full(len(cen), 7.0)), 4.0, 0.01)
    ok("c17 undecided: fewer than nine pieces in ten read", RLs["statement"] is None and RLs["read_share"] < 0.9)
    # the band-or-none entries and N1
    ok("c18 band entry HELD: F and the read say NO BAND", band_entry("NO BAND", ["NO BAND"] * 3, 0.01, [])[0] == "HELD")
    ok("c19 band entry NOT HELD: F says NO BAND, the read says BAND", band_entry("NO BAND", ["BAND"] * 3, 0.01, [])[0] == "NOT HELD")
    ok("c20 band entry: the read says ZONE, no label", band_entry("NO BAND", ["ZONE"] * 3, 0.01, [])[1] == "the read says ZONE")
    ok("c21 band entry: F inside the margin, no label", band_entry(None, ["BAND"] * 3, 0.01, [])[1] == "the statement of F lies inside the margin")
    ok("c22 band entry: margin above 0.03, no label", band_entry("NO BAND", ["NO BAND"] * 3, 0.031, [])[1] == "the margin is above 0.03")
    ok("c23 a statement the second draw does not repeat: no label", band_entry("NO BAND", ["NO BAND"] * 3, 0.01, [], second="ZONE")[1] == "the second draw does not repeat the statement")
    ok("c24 band entry: the bias changes the statement, no label", band_entry("NO BAND", ["NO BAND", "NO BAND", "ZONE"], 0.01, [])[1] == "inside the bias of the estimator")
    ok("c25 both say BAND, no edge with a word: no label", band_entry("BAND", ["BAND"] * 3, 0.01, [])[0] == "no label")
    ok("c26 both say BAND, an edge HELD: HELD; an edge NOT HELD: NOT HELD",
       band_entry("BAND", ["BAND"] * 3, 0.01, ["HELD"])[0] == "HELD" and band_entry("BAND", ["BAND"] * 3, 0.01, ["HELD", "NOT HELD"])[0] == "NOT HELD")
    ok("c27 N1: HELD on NO BAND, NOT HELD on BAND, no label on ZONE and undecided",
       [n1_word([x] * 3, 0.01)[0] for x in ("NO BAND", "BAND", "ZONE", None)] == ["HELD", "NOT HELD", "no label", "no label"])
    # F with no band, with one, with two; a band not predicted; the two forms of F; C14
    two = np.where(((s > 2.0) & (s < 4.0)) | ((s > 7.0) & (s < 9.0)), 3.0, 7.0)
    e_two = PR.crossings(s, two, 4.0)
    ok("c28 F with two bands: four edges, in order", [e["kind"] for e in e_two] == ["inner", "outer", "inner", "outer"] and len(bands_of(e_two, 12.0)) == 2, [e["r"] for e in e_two])
    ok("c29 F with no band: no edge, statement NO BAND", not PR.crossings(s, np.full(len(s), 7.0), 4.0) and PR.f_statement(np.full(len(s), 7.0), read, np.full(len(sr), 7.0), 4.0, 0.01) == "NO BAND")
    bands = bands_of(PR.crossings(s, np.where((s > 2.0) & (s < 4.0), 3.0, 7.0), 4.0), 12.0)
    run_far = dict(start=8.0, end=10.0, band=True)
    run_in = dict(start=3.0, end=4.2, band=True)
    outside = lambda r: all(r["end"] < 0.9 * lo or r["start"] > 1.1 * hi for lo, hi in bands)
    ok("c30 a band not predicted: a BAND of the read wholly outside every band of F", outside(run_far) and not outside(run_in), bands)
    ep = PR.crossings(s, mk(1 + 88 / s ** 2), 4.0)[0]
    er_far = dict(r=ep["r"] * 1.05, kind="outer")
    ok("c31 the two forms of F apart by more than 3 %: no label", abs(er_far["r"] - ep["r"]) / ep["r"] > FORMS_TOL and match([er_far], "outer", ep["r"]) is not None)
    ok("c32 one form of F lacking the edge: no match", match([], "outer", ep["r"]) is None and match([dict(r=5.0, kind="inner")], "outer", ep["r"]) is None)
    ok("c33 C14: an edge of F with q_P more than 2 % away fails", abs(ep["r"] * 1.03 - ep["r"]) / ep["r"] > QP_TOL and abs(ep["r"] * 1.01 - ep["r"]) / ep["r"] <= QP_TOL)
    ok("c34 the word of a reading: NOT HELD outranks HELD; no word, no label",
       reading_word(["HELD", "no label", "NOT HELD"]) == "NOT HELD" and reading_word(["HELD", "no label"]) == "HELD" and reading_word(["no label"]) == "no label")
    # the inner edge
    inner = np.where(cen < 3.0, 7.0, 3.0)
    RLi = read_line(cen, D, rd, recs_of(4.0 - 0.8 * (cen - 3.0)), 4.0, 0.01)
    ok("c35 an inner edge at 3.0 against r0 = 2.9: HELD", edge_word(2.9, "inner", cen, D, rd, RLi)[0] == "HELD", [(e["r"], e["kind"]) for e in RLi["edges"]])
    # C7 or C8 failed: the piece is treated as not fitted
    recs_c = recs_of(smooth)
    recs_c[12]["fitted_eff"] = False
    RLc = read_line(cen, D, rd, recs_c, 4.0, 0.01)
    ok("c36 a piece that fails C7 or C8 is not fitted: its status is OPEN and the window is not covered", RLc["status"][12] == "OPEN" and edge_word(6.2, "outer", cen, D, rd, RLc)[1] == "the window is not covered")
    res = dict(checks=checks, passed=all(c["passed"] for c in checks), count=len(checks))
    for c in checks:
        print("PASS" if c["passed"] else "FAIL", c["check"], "" if c["passed"] else c["got"])
    (a.work / "c2.json").write_text(json.dumps(PR.jsonable(res)) + "\n")
    print("C2", "MET" if res["passed"] else "NOT MET", f"({sum(c['passed'] for c in checks)} of {len(checks)})")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pure", action="store_true")
    ap.add_argument("--bins", action="store_true")
    ap.add_argument("--vacuum", nargs="*", default=[])
    ap.add_argument("--c6", action="store_true")
    ap.add_argument("--pieces", nargs="*", default=[])
    ap.add_argument("--jobs", nargs="*", default=[])
    ap.add_argument("--words", action="store_true")
    ap.add_argument("--hash", nargs="*", default=[], help="g=<sha256 of the prediction block of grid g>")
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    ap.add_argument("--work", type=Path, default=HERE.parent / "checkpoints" / "m5_49")
    a = ap.parse_args()
    a.hashes = dict(x.split("=", 1) for x in a.hash)
    f = a.data / "m5_49_bands_predicted.json"
    doc = json.loads(f.read_text()) if f.exists() else None
    if a.pure:
        control_pure(a)
    if a.bins:
        stage_bins(a, doc)
    if a.vacuum:
        stage_vacuum(a, doc)
    if a.c6:
        control_c6(a, doc)
    if a.pieces:
        stage_pieces(a, doc)
    if a.words:
        words(a, doc)


if __name__ == "__main__":
    main()
