"""M5.55: the author's rung T4 scored on the stored driven fields of M5.52 at omega = 5.5.

No job runs in this task. The read is that of M5.52 section 3 (read Z, the piece estimator), on the lock-in
amplitudes that the jobs Ge, Gd, Gb and VD1 to VD4 of M5.52 store, opened read-only:

    a(x)     = <W(x), A_k(x, 5.5)>                       (tone index 1, window k = 2; window 1 repeats every statement)
    Z_i      = - sum_S conj(a) Lap_h a / sum_S |a|^2,    y_i = w gamma Re Z / Im Z,    c2_read = w gamma / Im Z
    eps_i    = the budget of M5.52 (eps_of), with b_gamma and b_vac from m5_52_summary.json
    status   PROP when y - eps >= 0, EVAN when y + eps < 0, MARGIN otherwise
    kappa_A  = sqrt(-y_i),   kappa_P = sqrt(-Re Z_i),   kappa_F = sqrt(m_F^2 - w^2),   kappa_C = sqrt(C^2 - w^2)

THE ENTRIES (plan section 3): T4a' (the scored pieces inside the band of F, at least 2 D from its edges, all PROP),
T4b1 (the outer edge of y, y + eps, y - eps against the interval 8.4 to 8.8 on A, 9.7 to 10.1 on PF and PD),
T4b2 (the scored pieces within 1.5 beyond the interval, all EVAN; NOT READ unless the purity clause, the second
window and the faces rule hold). T4a (the literal phase) and T4c (the rate) are descriptors without a word.

From m5_52_reads.py this script calls only the pure functions ratio_read, read_field, joined_pairs, edges0, bin_of,
eps_of, kappa_bar and the read-only loader load_lockin. It writes only under this task's own paths.

Usage:  python m5_55_reads.py --predict                       (the prediction file; opens no lock-in amplitude)
        python m5_55_reads.py --k4                            (the rules on made-up profiles and pieces)
        python m5_55_reads.py --expect                        (the expectations of section 6 from tracked files)
        python m5_55_reads.py --read --hash b=.. d=.. e=.. V=.. settings=..
        python m5_55_reads.py --words --hash b=.. d=.. e=.. V=.. settings=..
        python m5_55_reads.py --plots
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import calendar
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
import m5_52_reads as RD

DATA = HERE.parent / "data"
WORK = HERE.parent / "checkpoints" / "m5_55"
PRED = DATA / "m5_55_predicted.json"
SUMMARY = DATA / "m5_55_summary.json"
M52_PRED = DATA / "m5_52_drive_predicted.json"
M52_SUMMARY = DATA / "m5_52_summary.json"
M52_READS = HERE.parent / "checkpoints" / "m5_52" / "reads"

# the prediction of M5.52 as committed in 44660227 (its file and the hashes of its blocks)
M52_PRED_SHA = "13593daa600de3bee404912cf647d1e1a668a077ca29c8ee03932461eba40c51"
M52_BLOCK_HASHES = dict(V="3aa643bd66cb8115c8c0168d28d33b9ed5fcc514f637eca74c42c57fcdbdc3fd",
                        b="95b4e39d2d536ad8a45151871aa3003bb6eb50084cbca74342d409397d0edff7",
                        d="fc6cea4907e93f58697decbab8cacdc84421c57562a69b92b1f1a8a3f99491ae",
                        e="8ac1a9f47736caf666ed662e25df4e66e3e6d07419c505005081fee536eb3bb9",
                        settings="d2263ae50762710843dae5ee14eb2fc20c464d145bc1a2b88c1789909fc96b6f")
PLAN_COMMIT = "44660227"
PLAN_PUSH_UTC = "2026-11-23 21:20:46"

W = 5.5
TONE_INDEX = D.TONES.index(W)
GAMMA = D.GAMMA
DELTA = 0.5
GRIDS = ("b", "d", "e")
GRID_JOB = dict(b="Gb", d="Gd", e="Ge")
VAC = ("VD1", "VD2", "VD3", "VD4")
LINES = ("A", "PF", "PD")
INTERVAL = dict(A=(8.4, 8.8), PF=(9.7, 10.1), PD=(9.7, 10.1))
REACH = 1.5
COVER = 0.9
WIN2_BOUND = 0.02
FACE_BOUND = 0.01
FLOOR = RD.FLOOR
T4C_TOL = 0.15
T4C_MARGIN = (1 - (1 - T4C_TOL) ** 2) / 2
K2_TOL = 0.02
WORD_GRIDS = dict(T4a=("d", "e"), T4b1=("e",), T4b2=("e",))
# known at freeze from the disclosed outcome of C11 (plan section 2); reference only, no word uses these numbers
KNOWN_DAMPING_TERM = dict(I=6.88, II=0.96, III=6.88)
COL_READ, COL_F, COL_C, COL_P = "#1f4e9c", "#2a9d8f", "#d9822b", "#8e44ad"


def log(msg):
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "run_log.md", "a") as fh:
        fh.write(f"- {time.strftime('%Y-%m-%d %H:%M:%S %Z')}: {msg}\n")
    print(msg, flush=True)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr(v):
    return np.array([np.nan if x is None else x for x in v], float)


# ----------------------------------------------------------------------------- the prediction of M5.52, checked

def m52_prediction():
    """The prediction file of M5.52, its file hash and every block hash checked; stops on a mismatch."""
    fh = sha_file(M52_PRED)
    if fh != M52_PRED_SHA:
        raise SystemExit(f"m5_52_drive_predicted.json: SHA-256 {fh} is not that of commit {PLAN_COMMIT} ({M52_PRED_SHA})")
    doc = json.loads(M52_PRED.read_text())
    for k, want in M52_BLOCK_HASHES.items():
        have = PR.block_hash(doc["blocks"][k])
        if have != want or doc["hashes"].get(k) != want:
            raise SystemExit(f"m5_52_drive_predicted.json block {k}: hash {have} (recorded {doc['hashes'].get(k)}) is not {want}")
    return doc


# ----------------------------------------------------------------------------- geometry

def face_distance(center, rays, s, hb):
    """The least distance to a face of the box over the points s along each ray of a line."""
    c = np.asarray(center, float)
    return float(min(np.min(hb - np.abs(c + s * np.asarray(e, float))) for e in rays))


def wkb_w(centers, kappa):
    """w = |d kappa / ds| / kappa^2 on the pieces, central differences between consecutive pieces with kappa > 0."""
    n = len(centers)
    out = np.full(n, np.nan)
    good = np.isfinite(kappa) & (kappa > 0)
    for i in range(n):
        if not good[i]:
            continue
        lo = i - 1 if i > 0 and good[i - 1] else i
        hi = i + 1 if i < n - 1 and good[i + 1] else i
        if hi == lo:
            continue
        out[i] = abs(kappa[hi] - kappa[lo]) / (centers[hi] - centers[lo]) / kappa[i] ** 2
    return out


def span_samples(lo, hi, h):
    k = int(round((hi - lo) / (h / 2)))
    return lo + np.arange(k + 1) * (h / 2)


def predict_line(g, ln, Lb, G):
    centers = arr(Lb["centers"])
    n = len(centers)
    read49 = np.array(Lb["read_m5_49"], bool)
    excl = np.array(Lb["source"]["excluded"], bool)
    mF, C = arr(Lb["mF"]), arr(Lb["C"])
    with np.errstate(invalid="ignore"):
        yF, yC = W * W - mF ** 2, W * W - C ** 2
        kF = np.where(yF < 0, np.sqrt(-yF), np.nan)
        kC = np.where(yC < 0, np.sqrt(-yC), np.nan)
    edges = [e for e in Lb["edges_F"]["pieces"][str(W)]]
    outer = [e["r"] for e in edges if e["kind"] == "outer"]
    rays = PR.line_rays(G["normal"])[ln]
    hb = G["hb"]
    s_out = centers + DELTA / 2
    d_face = np.array([face_distance(G["center"], rays, s, hb) for s in s_out])
    with np.errstate(invalid="ignore", over="ignore"):
        face_factor = np.where(np.isfinite(kF), np.exp(-2 * np.nan_to_num(kF) * d_face), 1.0)
    faces_ok = np.isfinite(kF) & (face_factor <= FACE_BOUND)
    geo = read49 & ~excl
    rec = dict(centers=centers, D=DELTA, read_m5_49=read49, source_excluded=excl, filter_allowed=Lb["source"]["filter_allowed"],
               mF=mF, C=C, y_F=np.where(read49, yF, np.nan), y_C=np.where(read49, yC, np.nan), kappa_F=np.where(read49, kF, np.nan),
               kappa_C=np.where(read49, kC, np.nan), w_F=wkb_w(centers, np.where(read49, kF, np.nan)), w_C=wkb_w(centers, np.where(read49, kC, np.nan)),
               edges_F=edges, outer_edge_F=outer[0] if outer else None, face_distance=d_face, face_factor=face_factor, faces_ok=faces_ok)
    # T4a': inside the band of F, at least 2 D from each of its edges, read by the geometry, not excluded by the source
    sc = [i for i in range(n) if geo[i] and np.isfinite(yF[i]) and yF[i] > 0 and all(abs(centers[i] - e["r"]) >= 2 * DELTA - 1e-9 for e in edges)]
    rec["T4a_scored"] = sc if g in WORD_GRIDS["T4a"] else []
    rec["T4a_scored_s"] = [float(centers[i]) for i in sc]
    lo, hi = INTERVAL[ln]
    win = [i for i in range(n) if lo - DELTA - 1e-9 <= centers[i] <= hi + DELTA + 1e-9]
    rec["T4b1_window"] = win
    rec["T4b1_window_s"] = [float(centers[i]) for i in win]
    rec["T4b1_window_geometry_read"] = bool(win and all(geo[i] for i in win))
    b2 = [i for i in range(n) if hi + DELTA - 1e-9 <= centers[i] <= hi + REACH + 1e-9]
    rec["T4b2_scored"] = b2
    rec["T4b2_scored_s"] = [float(centers[i]) for i in b2]
    rec["T4b2_geometry_read"] = bool(b2 and all(geo[i] for i in b2))
    rec["T4b2_faces_ok"] = [bool(faces_ok[i]) for i in b2]
    rec["beyond_reach"] = [i for i in range(n) if geo[i] and centers[i] > hi + REACH + 1e-9]
    rec["beyond_reach_s"] = [float(centers[i]) for i in rec["beyond_reach"]]
    # read S: the span of the T4b2 pieces, samples h/2 apart
    if b2:
        slo, shi = centers[b2[0]] - DELTA / 2, centers[b2[-1]] + DELTA / 2
        ss = span_samples(slo, shi, G["h"])
        inset = all(face_distance(G["center"], rays, s, hb) >= PR.INSET - 1e-9 for s in ss)
        dom = True
        for e in rays:
            pts = np.asarray(G["center"]) + ss[:, None] * np.asarray(e)
            idx = (pts - G["x"][0]) / G["h"]
            for p in idx:
                f = np.floor(p).astype(int)
                for dx in (0, 1):
                    for dy in (0, 1):
                        for dz in (0, 1):
                            q = (min(f[0] + dx, G["N"] - 1), min(f[1] + dy, G["N"] - 1), min(f[2] + dz, G["N"] - 1))
                            dom &= bool(G["dom"][q])
        rec["read_S"] = dict(span=[float(slo), float(shi)], samples=ss, inside_inset=bool(inset), nodes_in_domain=bool(dom), readable=bool(inset and dom))
    else:
        rec["read_S"] = None
    # the descriptor pieces: read by the geometry, beyond the outer edge of F
    rec["descriptor_pieces"] = [i for i in range(n) if geo[i] and outer and centers[i] > outer[0]]
    rec["descriptor_pieces_s"] = [float(centers[i]) for i in rec["descriptor_pieces"]]
    return rec


def predict(write=True):
    if write and PRED.exists():
        raise SystemExit("m5_55_predicted.json exists already; it is not rewritten")
    m52 = m52_prediction()
    blocks = {}
    for g in GRIDS:
        blk = m52["blocks"][g]
        G = D.background(g)
        lp = blk["loop"]
        dc = float(np.abs(np.asarray(G["center"]) - np.asarray(lp["center"])).max())
        dn = float(np.abs(np.asarray(G["normal"]) - np.asarray(lp["normal"])).max())
        dr = abs(G["radius"] - lp["radius"])
        if max(dc, dn, dr) > 1e-6:
            raise SystemExit(f"grid {g}: the base state's loop is not that of the hashed block ({dc}, {dn}, {dr})")
        Ls = PR.pieces_of(G)
        lines = {}
        for ln in LINES:
            Lb = blk["lines"][ln]
            if Ls[ln]["read"].tolist() != Lb["read_m5_49"] or len(Ls[ln]["centers"]) != len(Lb["centers"]):
                raise SystemExit(f"grid {g} line {ln}: the piece geometry of the base state is not that of the hashed block")
            lines[ln] = predict_line(g, ln, Lb, G)
        blocks[g] = dict(grid=g, job=GRID_JOB[g], half_box=G["hb"], spacing=G["h"], loop=lp, lines=lines,
                         words=[k for k, v in WORD_GRIDS.items() if g in v])
    V = m52["blocks"]["V"]
    vac = {}
    for name in VAC:
        m2 = D.VAC_V[name] * PR.M_MID2
        y = W * W - m2
        z = np.sqrt(complex(y, W * GAMMA))
        rec = dict(v=D.VAC_V[name], m2=m2, y=y, status="PROP" if y > 0 else "EVAN", read_pieces={ln: V["jobs"][name]["at"][str(W)]["lines"][ln]["read_pieces"] for ln in LINES})
        if y < 0:
            rec.update(kappa=float(np.sqrt(-y)), Im_sqrtZ=float(z.imag), Re_sqrtZ=float(z.real))
        vac[name] = rec
    blocks["V"] = dict(jobs=vac, K2=dict(job="VD1", y=vac["VD1"]["y"], kappa=vac["VD1"]["kappa"], Im_sqrtZ=vac["VD1"]["Im_sqrtZ"], Re_sqrtZ=vac["VD1"]["Re_sqrtZ"],
                                         tolerance=K2_TOL, pieces=sum(vac["VD1"]["read_pieces"].values())),
                       K3=dict(prop=["VD4", "VD2", "VD3"], evan=["VD1"], eps="max(0.01 w^2, 2 b_vac), no damping term (the status clause of C9)"))
    blocks["settings"] = dict(
        omega=W, tone_index=TONE_INDEX, gamma=GAMMA, piece=DELTA, intervals=INTERVAL, reach=REACH, cover=COVER, second_window=WIN2_BOUND,
        purity="power outside W under power on W on the piece (each window on its own field)", faces=FACE_BOUND, floor=FLOOR * W * W,
        t4c=dict(tolerance=T4C_TOL, margin_fraction_of_abs_yX=T4C_MARGIN, word=None, domain_bound_abs_yF=0.2 * W * W,
                 largest_margin=T4C_MARGIN * 0.2 * W * W),
        known_damping_term_at_freeze=KNOWN_DAMPING_TERM, least_eps_at_freeze=KNOWN_DAMPING_TERM["II"],
        prop_gaps="0 < y < 1.65 or 6.6 <= y < 6.88 cannot be PROP; -1.65 < y < 0 or -6.88 < y <= -6.6 cannot be EVAN",
        bins=[W * GAMMA * b for b in RD.BIN_EDGES], m52_functions=["ratio_read", "read_field", "joined_pairs", "edges0", "bin_of", "eps_of", "kappa_bar", "load_lockin"],
        k0="every cost rule of M5.52 at 5.5 inherited except the clause of C9 on the read speed", plane="HELD when one of PF, PD is HELD and none NOT HELD; NOT HELD when either is",
        words_on=WORD_GRIDS, read_S=dict(step="h/2", fit="least squares of -ln|a| against s, per ray, then averaged", second_window=WIN2_BOUND))
    blocks = {k: PR.jsonable(v) for k, v in blocks.items()}
    doc = dict(task="M5.55", what="the scored pieces, windows, spans, F and C on the pieces at omega = 5.5 and every bound, written before any read of grids d and e",
               plan_commit=PLAN_COMMIT, m5_52_prediction=dict(file_sha256=M52_PRED_SHA, block_hashes=M52_BLOCK_HASHES),
               hash_rule="SHA-256 of the JSON text of a block: keys sorted, separators ',' and ':', UTF-8, floats with seven significant digits",
               blocks=blocks, hashes={k: PR.block_hash(v) for k, v in blocks.items()})
    if write:
        PRED.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
        fh = sha_file(PRED)
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "prediction_hashes.json").write_text(json.dumps(dict(file=fh, blocks=doc["hashes"], written_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                                                                      epoch=time.time()), indent=1) + "\n")
        return doc, fh
    return doc, None


def check_own_hashes(hashes):
    doc = json.loads(PRED.read_text())
    for k, hsh in doc["hashes"].items():
        have = PR.block_hash(doc["blocks"][k])
        if hashes.get(k) != have or hsh != have:
            raise SystemExit(f"m5_55_predicted.json block {k}: the hash given ({hashes.get(k)}) is not that of the block ({have}); no lock-in amplitude is opened")
    return doc


# ----------------------------------------------------------------------------- the rules (pure)

def status_of(y, eps):
    if y is None or eps is None or not (np.isfinite(y) and np.isfinite(eps)):
        return None
    return "PROP" if y - eps >= 0 else ("EVAN" if y + eps < 0 else "MARGIN")


def second_window_ratio(a2, a1, L):
    """||a2 - a1|| / ||a2|| over the nodes of each piece in D."""
    num = np.bincount(L["piece"][L["sel"]], weights=(np.abs(a2 - a1) ** 2)[L["sel"]], minlength=L["n"])
    den = np.bincount(L["piece"][L["sel"]], weights=(np.abs(a2) ** 2)[L["sel"]], minlength=L["n"])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(num / den)


def purity_ok(p_in, p_out):
    return bool(np.isfinite(p_in) and np.isfinite(p_out) and p_out < p_in)


def t4a_window(scored, readable, status):
    n = len(scored)
    nr = sum(bool(readable[i]) for i in scored)
    if n == 0:
        return None, "no scored piece"
    if nr < COVER * n - 1e-12:
        return None, f"{nr} of {n} scored pieces read, under nine in ten"
    evan = [i for i in scored if readable[i] and status[i] == "EVAN"]
    if evan:
        return "NOT HELD", "EVAN: pieces " + ", ".join(str(i) for i in evan)
    if all(status[i] == "PROP" for i in scored if readable[i]):
        return "HELD", f"{nr} of {n} read, all PROP"
    return None, "a read scored piece is MARGIN, none EVAN: pieces " + ", ".join(str(i) for i in scored if readable[i] and status[i] != "PROP")


def nearest_to(rs, lo, hi):
    return min(rs, key=lambda r: 0.0 if lo <= r <= hi else min(abs(r - lo), abs(r - hi)))


def t4b1_window(centers, readable, y, eps, interval, win):
    lo, hi = interval
    if not win or not all(readable[i] for i in win):
        return None, "the window is not covered", None
    ok = np.array(readable, bool)
    yy = np.where(ok, y, np.nan)
    got, sides = [], []
    for nm, p in (("y", yy), ("y+eps", yy + eps), ("y-eps", yy - eps)):
        outs = [e["r"] for e in RD.edges0(centers, p, ok, DELTA) if e["kind"] == "outer"]
        if not outs:
            got.append(None)
            return None, f"the profile {nm} has no outer edge", got
        r = nearest_to(outs, lo, hi)
        got.append(float(r))
        sides.append("inside" if lo <= r <= hi else ("below" if r < lo else "above"))
    if all(s == "inside" for s in sides):
        return "HELD", "", got
    if all(s == "below" for s in sides) or all(s == "above" for s in sides):
        return "NOT HELD", f"all three {sides[0]} the interval", got
    return None, "the three edges straddle the interval (" + ", ".join(sides) + ")", got


def t4b2_window(scored, readable, status, why_not):
    if not scored:
        return None, "no scored piece"
    nr = [i for i in scored if not readable[i]]
    if nr:
        return None, "NOT READ: " + "; ".join(f"piece {i} ({why_not[i]})" for i in nr)
    prop = [i for i in scored if status[i] == "PROP"]
    if prop:
        return "NOT HELD", "PROP: pieces " + ", ".join(str(i) for i in prop)
    if all(status[i] == "EVAN" for i in scored):
        return "HELD", "every scored piece read and EVAN"
    return None, "a scored piece is MARGIN, none PROP"


def both_windows(r2, r1):
    w2, w1 = r2[0], r1[0]
    if w2 != w1:
        return "no label", f"window 1 says {w1 or 'no label'}, window 2 says {w2 or 'no label'}"
    return (w2 or "no label"), r2[1]


def plane_word(wpf, wpd):
    ws = (wpf, wpd)
    if "NOT HELD" in ws:
        return "NOT HELD"
    if "HELD" in ws:
        return "HELD"
    return "no label"


def line_words(x, pl, budget_ok, withhold):
    """The words of one line of one grid. x: per piece, for each window k in (2, 1): ok{k} (M5.52's rules and the
    C11 bins), y{k}; eps (window 2's budget, None when an input is missing); for T4b2: purity{k}, win2 (the second
    window ratio), faces_ok. pl: the prediction of the line (scored pieces, windows, interval). withhold: reasons of K0
    or of blindness; budget_ok: False when a term of eps is missing."""
    centers = np.asarray(pl["centers"], float)
    n = len(centers)
    eps = np.asarray(x["eps"], float) if x.get("eps") is not None else np.full(n, np.nan)
    res = {}
    per = {}
    for k in (2, 1):
        ok = np.asarray(x[f"ok{k}"], bool) & np.isfinite(eps)
        y = np.asarray(x[f"y{k}"], float)
        st = [status_of(y[i], eps[i]) if ok[i] else None for i in range(n)]
        why_not, rd2 = [], []
        for i in range(n):
            if not ok[i]:
                why_not.append("not read under the rules of M5.52" if np.isfinite(eps[i]) or not x[f"ok{k}"][i] else "no budget")
                rd2.append(False)
            elif not x[f"purity{k}"][i]:
                why_not.append("purity")
                rd2.append(False)
            elif not (np.isfinite(x["win2"][i]) and x["win2"][i] <= WIN2_BOUND):
                why_not.append("second window")
                rd2.append(False)
            elif not x["faces_ok"][i]:
                why_not.append("faces")
                rd2.append(False)
            else:
                why_not.append("")
                rd2.append(True)
        per[k] = dict(T4a=t4a_window(pl["T4a_scored"], ok, st), T4b1=t4b1_window(centers, ok, y, eps, tuple(pl["interval"]), pl["T4b1_window"]),
                      T4b2=t4b2_window(pl["T4b2_scored"], rd2, st, why_not), status=st, readable_T4b2=rd2, why_not=why_not)
    for ent in ("T4a", "T4b1", "T4b2"):
        if ent == "T4a" and not pl["T4a_scored"]:
            continue
        if ent != "T4a" and not pl.get(f"{ent}_scored_or_window"):
            continue
        word, why = both_windows(per[2][ent][:2], per[1][ent][:2])
        rec = dict(word_as_read=word, why_as_read=why, window2=per[2][ent][0] or "no label", window1=per[1][ent][0] or "no label",
                   why_window2=per[2][ent][1], why_window1=per[1][ent][1])
        if ent == "T4b1":
            rec["edges_window2"], rec["edges_window1"] = per[2][ent][2], per[1][ent][2]
        if not budget_ok:
            rec.update(word="no label", why="no label (budget input missing)")
        elif withhold:
            rec.update(word="no label", why="; ".join(withhold) + (f" (as read: {word})" if word != "no label" else ""))
        else:
            rec.update(word=word, why=why)
        res[ent] = rec
    res["status_window2"] = per[2]["status"]
    res["status_window1"] = per[1]["status"]
    res["readable_T4b2_window2"] = per[2]["readable_T4b2"]
    res["why_not_T4b2_window2"] = per[2]["why_not"]
    return res


# ----------------------------------------------------------------------------- K0, K1 and blindness (from M5.52's summary and files)

def k0_reasons(s52, g):
    """The cost rules of M5.52 at 5.5 on grid g, inherited except the speed clause of C9."""
    out = []
    st0 = s52.get("stage0", {})
    for k in ("C1", "C2", "C3", "C13", "C14"):
        if not st0.get(k, {}).get("passed", False):
            out.append(f"M5.52 {k} not met" if k in st0 else f"M5.52 {k} outcome missing")
    for k in ("C4", "C6", "C7"):
        if not s52.get(k, {}).get("passed", False):
            out.append(f"M5.52 {k} not met" if k in s52 else f"M5.52 {k} outcome missing")
    job = GRID_JOB[g]
    c8 = s52.get("C8", {}).get(job)
    if c8 is None:
        out.append(f"M5.52 C8 of {job} missing")
    else:
        if c8.get("capped"):
            out.append(f"{job} capped")
        if not c8.get("passed"):
            out.append(f"M5.52 C8 not met on {job}")
    gw = s52.get("grids", {}).get(g)
    if gw is None:
        out.append(f"M5.52 grid {g} missing from its summary")
    else:
        if not gw.get("C10", {}).get(str(W), {}).get("passed", False):
            out.append("M5.52 C10 not met on this grid at 5.5")
        if not gw.get("C12", {}).get(str(W), False):
            out.append("M5.52 C12 not met on this grid at 5.5")
    c9 = c9_without_speed(s52)
    if not c9["passed"]:
        out.append("M5.52 C9 at 5.5 not met on a clause other than the read speed: " + "; ".join(c9["failed"]))
    return out


def c9_without_speed(s52):
    rows = [r for r in s52.get("C9", {}).get("rows", []) if r.get("bounded") and abs(r.get("tone", 0) - W) < 1e-9]
    failed = []
    if len(rows) != 4:
        failed.append(f"{len(rows)} bounded vacuum cases at 5.5 in the summary, not 4")
    for r in rows:
        nm = r["job"]
        if r["read"] < 24:
            failed.append(f"{nm}: {r['read']} pieces read")
        if r["soft"] != 0:
            failed.append(f"{nm}: {r['soft']} soft")
        if not r["worst_m2_error"] <= FLOOR * W * W:
            failed.append(f"{nm}: cutoff error {r['worst_m2_error']}")
        if r["wrong_status"] != 0:
            failed.append(f"{nm}: {r['wrong_status']} statuses wrong")
        if r["edges"] != 0:
            failed.append(f"{nm}: {r['edges']} edges")
        for ln, v in r.get("lines", {}).items():
            ph = v.get("phi")
            if ph is not None and not (ph.get("rel") is not None and ph["rel"] <= 0.02):
                failed.append(f"{nm} {ln}: Phi off by {ph.get('rel')}")
    speed = [dict(job=r["job"], worst_c2_dev=r["worst_c2_dev"]) for r in rows]
    return dict(passed=not failed, failed=failed, speed_clause_not_inherited=speed)


def blind_check():
    """The order of the record: the commit, this task's prediction hashes, then M5.52's first read of d or e."""
    push = calendar.timegm(time.strptime(PLAN_PUSH_UTC, "%Y-%m-%d %H:%M:%S"))
    ph = json.loads((WORK / "prediction_hashes.json").read_text())
    utc = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) if t is not None else None
    out = dict(plan_push_utc=PLAN_PUSH_UTC, prediction_written_utc=utc(ph["epoch"]), grids={})
    for g in ("d", "e"):
        f = M52_READS / f"{GRID_JOB[g]}.json"
        t = f.stat().st_mtime if f.exists() else None
        out["grids"][g] = dict(m52_read_mtime_utc=utc(t), before_commit=bool(t is not None and t < push), before_prediction=bool(t is not None and t < ph["epoch"]))
    out["K1"] = not any(v["before_prediction"] for v in out["grids"].values())
    return out


# ----------------------------------------------------------------------------- read Z on the stored fields (phase 2)

def inner_W(W6, A6):
    return np.einsum("...c,...c->...", W6 * E.METRIC, A6)


def power6(A6):
    return (np.abs(A6) ** 2 * E.METRIC).sum(-1)


def binomial_filter(a, passes=8):
    """The filter of M5.52 section 3 (the same code as its reads script)."""
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


def piece_max(L, field):
    out = np.zeros(L["n"])
    on = L["piece"] >= 0
    np.maximum.at(out, L["piece"][on], field[on])
    return out


def sample_rays(G, rays, a, s):
    out = []
    for e in rays:
        pts = np.asarray(G["center"]) + s[:, None] * np.asarray(e)
        idx = ((pts - G["x"][0]) / G["h"]).T
        out.append(map_coordinates(a.real, idx, order=1) + 1j * map_coordinates(a.imag, idx, order=1))
    return np.array(out)


def read_job(name, pred):
    job = D.JOBS[name]
    G = D.background(job["grid"])
    Ls = PR.pieces_of(G)
    gam = job["gamma"]
    A = RD.load_lockin(name)
    A2, A1 = A[2][TONE_INDEX], A[1][TONE_INDEX]
    del A
    a2, a1 = inner_W(G["W6"], A2), inner_W(G["W6"], A1)
    P2, P1 = power6(A2), power6(A1)
    del A2, A1
    af = binomial_filter(a2)
    gf = binomial_filter(G["g"])
    out = dict(job=name, grid=job["grid"], gamma=gam, lines={})
    vac = job["kind"] == "vacuum"
    for ln in LINES:
        L = Ls[ln]
        gmax, gfmax = piece_max(L, G["g"]), piece_max(L, gf)
        if vac:
            mF2 = np.full(L["n"], job["v"] * PR.M_MID2)
        else:
            mF2 = arr(pred["blocks"][job["grid"]]["lines"][ln]["mF"]) ** 2
        r2 = RD.read_field(a2, P2, G["h"], L, W, gam, mF2, gmax, L["read"])
        r1 = RD.read_field(a1, P1, G["h"], L, W, gam, mF2, gmax, L["read"])
        rf = RD.read_field(af, None, G["h"], L, W, gam, mF2, gfmax * 0, L["read"])
        Z = r2["Z_re"] + 1j * r2["Z_im"]
        rec = dict(centers=L["centers"], read49=L["read"], g_max=gmax, filter_allowed=gfmax < RD.G_FILT,
                   w2={k: r2[k] for k in ("Z_re", "Z_im", "m2", "c2", "y", "rms", "p_in", "p_out", "reason", "ok")},
                   w1={k: r1[k] for k in ("Z_re", "Z_im", "m2", "c2", "y", "rms", "p_in", "p_out", "reason", "ok")},
                   filtered=dict(y=rf["y"], ok=rf["ok"]), win2=second_window_ratio(a2, a1, L), Im_sqrtZ=np.sqrt(Z).imag)
        rays = L["rays"]
        smax = min(float(np.min((G["hb"] - PR.INSET - np.sign(e) * G["center"]) / np.maximum(np.abs(e), 1e-12))) for e in rays)
        s = np.arange(0.0, smax + 1e-9, G["h"])
        v2 = sample_rays(G, rays, a2, s)
        ph = np.unwrap(np.angle(v2), axis=1)
        rec["rays"] = dict(s=s, modulus=np.abs(v2), phase=ph, dphase=np.gradient(ph, s, axis=1))
        if not vac:
            rs = pred["blocks"][job["grid"]]["lines"][ln]["read_S"]
            if rs is not None and rs["inside_inset"]:
                ss = arr(rs["samples"])
                l2 = np.log(np.abs(sample_rays(G, rays, a2, ss)))
                l1 = np.log(np.abs(sample_rays(G, rays, a1, ss)))
                rec["read_S"] = dict(s=ss, ln_a2=l2, ln_a1=l1)
        out["lines"][ln] = rec
    out = PR.jsonable(out)
    (WORK / "reads").mkdir(parents=True, exist_ok=True)
    (WORK / "reads" / f"{name}.json").write_text(json.dumps(out) + "\n")
    return out


def piece_dict(r):
    return dict(y=arr(r["y"]), ok=np.array(r["ok"], bool))


# ----------------------------------------------------------------------------- the words, the descriptors (phase 2)

def read_S_of(rS, readable):
    if rS is None:
        return None
    s = arr(rS["s"])
    per = []
    for k in range(len(rS["ln_a2"])):
        l2, l1 = arr(rS["ln_a2"][k]), arr(rS["ln_a1"][k])
        dev = float(np.max(np.abs(l2 - l1)))
        slope = float(np.polyfit(s, -l2, 1)[0])
        per.append(dict(ray=k, kappa_S=slope, second_window_max=dev, passed=bool(dev <= WIN2_BOUND)))
    allp = all(p["passed"] for p in per)
    return dict(span=[float(s[0]), float(s[-1])], samples=len(s), rays=per, kappa_S=float(np.mean([p["kappa_S"] for p in per])) if allp and readable else None,
                why=None if (allp and readable) else ("the span leaves the inset (the line does not reach it)" if not readable else "the second window on a ray"))


def words(pred, s52, s52_sha):
    R = {n: json.loads((WORK / "reads" / f"{n}.json").read_text()) for n in [GRID_JOB[g] for g in GRIDS] + list(VAC)}
    bgam = s52.get("C11", {}).get("b_gamma", {}).get(str(W))
    bvac = s52.get("C9", {}).get("b_vac", {}).get(str(W))
    budget_ok = bool(bgam is not None and len(bgam) == 3 and all(v is not None and np.isfinite(v) for v in bgam) and bvac is not None and np.isfinite(bvac))
    c11rows = s52.get("C11", {}).get("per_tone", {}).get(str(W), [])
    bins_fail = [b for b, r in enumerate(c11rows) if r.get("passed") is False]
    blind = blind_check()
    out = dict(task="M5.55", prediction_hashes=pred["hashes"], m5_52_summary_sha256=s52_sha, b_gamma=bgam, b_vac=bvac, budget_inputs_present=budget_ok,
               C11_bins_failed_at_5_5=bins_fail, blindness=blind, C9_without_speed=c9_without_speed(s52), grids={}, entries=[])
    k1_ok = blind["K1"]
    for g in GRIDS:
        job = GRID_JOB[g]
        rj = R[job]
        blk = pred["blocks"][g]
        wh = k0_reasons(s52, g)
        if not k1_ok:
            wh.append("K1 not met: a read of d or e before this task's prediction hashes")
        if g in ("d", "e") and blind["grids"][g]["before_commit"]:
            wh.append("not blind")
        gw = dict(job=job, K0_reasons=wh, lines={})
        for ln in LINES:
            pl = dict(blk["lines"][ln])
            pl["interval"] = INTERVAL[ln]
            pl["T4b1_scored_or_window"] = bool(g in WORD_GRIDS["T4b1"])
            pl["T4b2_scored_or_window"] = bool(g in WORD_GRIDS["T4b2"])
            if g not in WORD_GRIDS["T4b2"]:
                pl["T4b2_scored"] = []
            rl = rj["lines"][ln]
            P, P1, Pf = piece_dict(rl["w2"]), piece_dict(rl["w1"]), piece_dict(rl["filtered"])
            allowed = np.array(rl["filter_allowed"], bool)
            n = len(P["y"])
            for i in range(n):
                if P["ok"][i] and RD.bin_of(P["y"][i], W) in bins_fail:
                    P["ok"][i] = False
                if P1["ok"][i] and RD.bin_of(P1["y"][i], W) in bins_fail:
                    P1["ok"][i] = False
            if budget_ok:
                eps, terms = RD.eps_of(P, P1, Pf, allowed, W, bgam, bvac)
            else:
                eps, terms = None, [None] * n
            pin2, pout2 = arr(rl["w2"]["p_in"]), arr(rl["w2"]["p_out"])
            pin1, pout1 = arr(rl["w1"]["p_in"]), arr(rl["w1"]["p_out"])
            x = dict(ok2=P["ok"], ok1=P1["ok"] & P["ok"], y2=P["y"], y1=P1["y"], eps=eps,
                     purity2=[purity_ok(pin2[i], pout2[i]) for i in range(n)], purity1=[purity_ok(pin1[i], pout1[i]) for i in range(n)],
                     win2=arr(rl["win2"]), faces_ok=blk["lines"][ln]["faces_ok"])
            res = line_words(x, pl, budget_ok, wh)
            epsv = np.full(n, np.nan) if eps is None else np.asarray(eps)
            y2 = np.where(P["ok"], P["y"], np.nan)
            ZR = arr(rl["w2"]["Z_re"])
            c2 = arr(rl["w2"]["c2"])
            with np.errstate(invalid="ignore"):
                kA = np.where(P["ok"] & (y2 < 0), np.sqrt(-y2), np.where(P["ok"], 0.0, np.nan))
                kP = np.where(P["ok"] & (ZR < 0), np.sqrt(-ZR), np.where(P["ok"], 0.0, np.nan))
            kF, kC = arr(blk["lines"][ln]["kappa_F"]), arr(blk["lines"][ln]["kappa_C"])
            centers = arr(blk["lines"][ln]["centers"])
            okv = P["ok"] & np.isfinite(epsv)
            read_edges = [RD.edges0(centers, p, okv, DELTA) for p in (y2, y2 + epsv, y2 - epsv)]
            outer_read = [e["r"] for e in read_edges[0] if e["kind"] == "outer"]
            r_out = nearest_to(outer_read, *INTERVAL[ln]) if outer_read else None
            desc = []
            for i in blk["lines"][ln]["descriptor_pieces"]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    desc.append(dict(s=float(centers[i]), piece=i, read=bool(P["ok"][i]), reason=rl["w2"]["reason"][i], y=y2[i], eps=epsv[i],
                                     bin=["I", "II", "III"][RD.bin_of(y2[i], W)] if np.isfinite(y2[i]) else None, status=res["status_window2"][i],
                                     kappa_A=kA[i], kappa_P=kP[i], kappa_F=kF[i], kappa_C=kC[i], A_over_F=kA[i] / kF[i], P_over_F=kP[i] / kF[i],
                                     A_over_C=kA[i] / kC[i], P_over_C=kP[i] / kC[i], Im_sqrtZ=arr(rl["Im_sqrtZ"])[i], c2_read=c2[i],
                                     purity_ratio=pout2[i] / pin2[i], second_window=arr(rl["win2"])[i], faces_ok=bool(blk["lines"][ln]["faces_ok"][i]),
                                     readable_T4b2_rules=bool(res["readable_T4b2_window2"][i]), why_not=res["why_not_T4b2_window2"][i],
                                     regime="beyond 1.5 past the interval (break of regime)" if centers[i] > INTERVAL[ln][1] + REACH + 1e-9 else "within 1.5 past the interval"))
            b2 = blk["lines"][ln]["T4b2_scored"]
            rd2 = [i for i in b2 if res["readable_T4b2_window2"][i]]
            # the plan gates read S by the second window alone; the prediction's flag 'readable' (inset and domain of the
            # interpolation nodes, counting a node of zero weight at the end of the span) is tabled, never used as a gate
            rS = read_S_of(rl.get("read_S"), bool(blk["lines"][ln]["read_S"] and blk["lines"][ln]["read_S"]["inside_inset"]))
            if rS is not None:
                rS["prediction_flag_nodes_in_domain"] = blk["lines"][ln]["read_S"]["nodes_in_domain"]
                rS["mean_Im_sqrtZ_T4b2_read"] = float(np.mean(arr(rl["Im_sqrtZ"])[rd2])) if rd2 else None
                rS["mean_Im_sqrtZ_T4b2_all"] = float(np.nanmean(arr(rl["Im_sqrtZ"])[b2])) if b2 else None
            rays = rl["rays"]
            sr = arr(rays["s"])
            pick = [j for j in range(len(sr)) if abs(sr[j] / DELTA - round(sr[j] / DELTA)) < 1e-6]
            dph = np.array([arr(v) for v in rays["dphase"]])
            phase = dict(s=sr[pick], dphase_mean=dph[:, pick].mean(0), dphase_min=dph[:, pick].min(0), dphase_max=dph[:, pick].max(0),
                         modulus_mean=np.array([arr(v) for v in rays["modulus"]])[:, pick].mean(0))
            okF = np.isfinite(arr(blk["lines"][ln]["y_F"]))
            kbar = dict(read=RD.kappa_bar(centers, okv, y2, r_out) if r_out is not None else None,
                        F=RD.kappa_bar(centers, okF, arr(blk["lines"][ln]["y_F"]), blk["lines"][ln]["outer_edge_F"]) if blk["lines"][ln]["outer_edge_F"] is not None else None)
            gw["lines"][ln] = dict(centers=centers, y=y2, y_w1=np.where(P1["ok"], P1["y"], np.nan), eps=epsv, eps_terms=terms, ok=P["ok"], reason=rl["w2"]["reason"],
                                   c2=np.where(P["ok"], c2, np.nan), Z_re=ZR, Im_sqrtZ=arr(rl["Im_sqrtZ"]), kappa_A=kA, kappa_P=kP, purity_ratio=pout2 / pin2,
                                   second_window=arr(rl["win2"]), status=res["status_window2"], status_w1=res["status_window1"], read_edges=read_edges,
                                   outer_edge_read_nearest=r_out, kappa_bar=kbar, descriptor=desc, read_S=rS, phase=phase,
                                   words={k: res[k] for k in ("T4a", "T4b1", "T4b2") if k in res})
            for k in ("T4a", "T4b1", "T4b2"):
                if k in res:
                    out["entries"].append(dict(entry=k, grid=g, line=ln, **res[k]))
        for k in ("T4a", "T4b1", "T4b2"):
            if g not in WORD_GRIDS[k]:
                continue
            wA = gw["lines"]["A"]["words"].get(k, {}).get("word", "no label")
            wPF = gw["lines"]["PF"]["words"].get(k, {}).get("word", "no label")
            wPD = gw["lines"]["PD"]["words"].get(k, {}).get("word", "no label")
            gw[k] = dict(axis=wA, plane=plane_word(wPF, wPD), PF=wPF, PD=wPD)
        out["grids"][g] = gw
    # K2 and K3 on the vacuum jobs
    k2rows, k3 = [], {}
    eps_v = max(FLOOR * W * W, 2 * bvac) if bvac is not None and np.isfinite(bvac) else None
    for name in VAC:
        rv = R[name]
        yset = pred["blocks"]["V"]["jobs"][name]["y"]
        sts, n_edges, n_outer = [], 0, 0
        for ln in LINES:
            rl = rv["lines"][ln]
            P = piece_dict(rl["w2"])
            c = arr(rl["centers"])
            y = np.where(P["ok"], P["y"], np.nan)
            ZR = arr(rl["w2"]["Z_re"])
            if eps_v is not None:
                sts += [status_of(y[i], eps_v) for i in range(len(y)) if P["ok"][i]]
                ee = [RD.edges0(c, y + d, P["ok"], DELTA) for d in (0.0, eps_v, -eps_v)]
                n_edges += sum(len(e) for e in ee)
                n_outer += sum(1 for e3 in ee for e in e3 if e["kind"] == "outer")
            if name == "VD1":
                K = pred["blocks"]["V"]["K2"]
                for i in range(len(y)):
                    if P["ok"][i]:
                        kA = float(np.sqrt(max(-y[i], 0)))
                        kP = float(np.sqrt(max(-ZR[i], 0)))
                        im = float(arr(rl["Im_sqrtZ"])[i])
                        k2rows.append(dict(line=ln, s=float(c[i]), kappa_A=kA, kappa_P=kP, Im_sqrtZ=im, c2=float(arr(rl["w2"]["c2"])[i]),
                                           rel_A=kA / K["kappa"] - 1, rel_P=kP / K["kappa"] - 1, rel_Im=im / K["Im_sqrtZ"] - 1))
        want = "PROP" if yset > 0 else "EVAN"
        if eps_v is None:
            k3[name] = dict(met=False, why="budget input missing")
        else:
            met = bool(all(s == want for s in sts) and (n_edges == 0 if want == "EVAN" else (n_outer == 0 and "EVAN" not in sts)))
            k3[name] = dict(y_set=yset, want=want, read=len(sts), statuses={s: sts.count(s) for s in set(sts)}, edges=n_edges, outer_edges=n_outer, met=met)
    k2met = bool(k2rows and all(abs(r["rel_A"]) <= K2_TOL and abs(r["rel_P"]) <= K2_TOL and abs(r["rel_Im"]) <= K2_TOL for r in k2rows))
    out["K2"] = dict(rows=k2rows, read=len(k2rows), worst_A=max((abs(r["rel_A"]) for r in k2rows), default=None), worst_P=max((abs(r["rel_P"]) for r in k2rows), default=None),
                     worst_Im=max((abs(r["rel_Im"]) for r in k2rows), default=None), met=k2met, cost="none (reported beside the descriptor)")
    out["K3"] = dict(cases=k3, met=all(v["met"] for v in k3.values()))
    if not out["K3"]["met"]:
        for e in out["entries"]:
            if e["word"] != "no label":
                e.update(word="no label", why="K3 not met")
        for g in out["grids"]:
            for k in ("T4a", "T4b1", "T4b2"):
                if k in out["grids"][g]:
                    out["grids"][g][k] = dict(axis="no label", plane="no label", PF="no label", PD="no label", why="K3 not met")
    k4 = json.loads((WORK / "k4.json").read_text()) if (WORK / "k4.json").exists() else dict(passed=False)
    out["K4"] = dict(passed=k4.get("passed"), cases=k4.get("cases"), met=k4.get("met"))
    if not k4.get("passed"):
        for e in out["entries"]:
            e.update(word="no label", why="K4 not met")
    out = PR.jsonable(out)
    txt = json.dumps(out, sort_keys=True, separators=(",", ":")) + "\n"
    SUMMARY.write_text(txt)
    print("summary", hashlib.sha256(txt.encode()).hexdigest(), len(txt), "bytes", flush=True)
    for g, gw in out["grids"].items():
        for k in ("T4a", "T4b1", "T4b2"):
            if k in gw:
                print(g, k, gw[k], flush=True)
    return out


# ----------------------------------------------------------------------------- K4, the rules on made-up profiles and pieces

def k4():
    cases = []

    def add(name, got, want):
        cases.append(dict(case=name, got=got, want=want, met=bool(got == want)))

    n = 30
    centers = (np.arange(n) + 0.5) * DELTA
    sc = list(range(4, 14))
    b1w = [i for i in range(n) if 8.4 - DELTA - 1e-9 <= centers[i] <= 8.8 + DELTA + 1e-9]
    b2 = [i for i in range(n) if 8.8 + DELTA - 1e-9 <= centers[i] <= 8.8 + REACH + 1e-9]
    pl = dict(centers=centers, T4a_scored=sc, T4b1_window=b1w, T4b2_scored=b2, interval=(8.4, 8.8), T4b1_scored_or_window=True, T4b2_scored_or_window=True)

    def base(slope=20.0, r0=8.6, eps=0.16):
        y = slope * (r0 - centers)
        return dict(ok2=np.ones(n, bool), ok1=np.ones(n, bool), y2=y.copy(), y1=y.copy(), eps=np.full(n, eps), purity2=[True] * n, purity1=[True] * n,
                    win2=np.full(n, 0.005), faces_ok=[True] * n)

    def run(x, budget_ok=True, withhold=()):
        return line_words(x, pl, budget_ok, list(withhold))

    # T4a'
    x = base()
    add("T4a' HELD", run(x)["T4a"]["word"], "HELD")
    x = base()
    x["y2"][7] = x["y1"][7] = -5.0
    add("T4a' NOT HELD (one EVAN, named)", (run(x)["T4a"]["word"], "7" in run(x)["T4a"]["why"]), ("NOT HELD", True))
    x = base()
    x["ok2"][5] = x["ok2"][6] = x["ok1"][5] = x["ok1"][6] = False
    r = run(x)["T4a"]
    add("T4a' no label by the nine in ten (8 of 10)", (r["word"], "under nine in ten" in r["why_window2"]), ("no label", True))
    x = base()
    x["ok2"][5] = False
    x["ok1"][5] = False
    add("T4a' HELD at nine in ten exactly", run(x)["T4a"]["word"], "HELD")
    x = base()
    x["y2"][8] = x["y1"][8] = 0.1
    add("T4a' no label by a MARGIN piece", run(x)["T4a"]["word"], "no label")
    x = base()
    x["y1"][7] = -5.0
    add("T4a' no label by window 1", run(x)["T4a"]["word"], "no label")
    add("T4a' no label by a missing budget term", run(base(), budget_ok=False)["T4a"]["why"], "no label (budget input missing)")
    x = base()
    x["eps"] = None
    add("T4a' no label by an eps that is absent", run(x)["T4a"]["word"], "no label")
    # T4b1
    add("T4b1 HELD (all three inside)", run(base(20.0, 8.6))["T4b1"]["word"], "HELD")
    add("T4b1 NOT HELD (all three above)", run(base(20.0, 9.4))["T4b1"]["word"], "NOT HELD")
    add("T4b1 NOT HELD (all three below)", run(base(20.0, 7.9))["T4b1"]["word"], "NOT HELD")
    x = base(20.0, 8.6)
    x["ok2"][b1w[1]] = x["ok1"][b1w[1]] = False
    add("T4b1 no label by the cover", (run(x)["T4b1"]["word"], run(x)["T4b1"]["why_window2"]), ("no label", "the window is not covered"))
    r = run(base(2.6, 8.6, 0.96))["T4b1"]
    add("T4b1 no label by the three profiles straddling", (r["word"], r["why_window2"].startswith("the three edges straddle")), ("no label", True))
    x = base()
    x["y2"] = np.full(n, -0.5)
    x["y1"] = np.full(n, -0.5)
    x["eps"] = np.full(n, 1.0)
    r = run(x)["T4b1"]
    add("T4b1 no label by a profile with no edge", (r["word"], "has no outer edge" in r["why_window2"]), ("no label", True))
    x = base(20.0, 8.6)
    x["y1"] = 20.0 * (9.4 - centers)
    add("T4b1 no label by window 1", run(x)["T4b1"]["word"], "no label")
    add("T4b1 no label by a missing budget term", run(base(20.0, 8.6), budget_ok=False)["T4b1"]["why"], "no label (budget input missing)")
    # T4b2
    add("T4b2 HELD", run(base())["T4b2"]["word"], "HELD")
    x = base()
    x["y2"][b2[1]] = x["y1"][b2[1]] = 5.0
    r = run(x)["T4b2"]
    add("T4b2 NOT HELD (one PROP, named)", (r["word"], str(b2[1]) in r["why"]), ("NOT HELD", True))
    # purity and the second window from set powers and amplitudes
    p_in, p_out = 1.0, 1.2
    x = base()
    x["purity2"] = [purity_ok(p_in, p_out) if i == b2[0] else True for i in range(n)]
    x["purity1"] = list(x["purity2"])
    r = run(x)["T4b2"]
    add("T4b2 no label by the purity clause (power outside W 1.2 against 1.0 on W)", (r["word"], "purity" in r["why"]), ("no label", True))
    Lm = dict(piece=np.array([0] * 8 + [1] * 8), sel=np.ones(16, bool), n=2)
    a2 = np.ones(16, complex)
    a1 = np.r_[np.full(8, 0.99 + 0j), np.full(8, 0.97 + 0j)]
    wr = second_window_ratio(a2, a1, Lm)
    add("second window ratio of set amplitudes (0.01, 0.03)", [round(float(v), 12) for v in wr], [0.01, 0.03])
    x = base()
    x["win2"] = np.where(np.arange(n) == b2[0], wr[1], wr[0])
    r = run(x)["T4b2"]
    add("T4b2 no label by the second window", (r["word"], "second window" in r["why"]), ("no label", True))
    x = base()
    x["win2"] = np.where(np.arange(n) == b2[0], wr[0], wr[0])
    add("T4b2 HELD with a second window of 0.01", run(x)["T4b2"]["word"], "HELD")
    x = base()
    x["faces_ok"] = [i != b2[1] for i in range(n)]
    r = run(x)["T4b2"]
    add("T4b2 no label by the faces", (r["word"], "faces" in r["why"]), ("no label", True))
    x = base()
    x["ok2"][b2[0]] = x["ok1"][b2[0]] = False
    r = run(x)["T4b2"]
    add("T4b2 no label by a piece not read under M5.52's rules", (r["word"], "M5.52" in r["why"]), ("no label", True))
    x = base()
    x["y2"][b2[0]] = x["y1"][b2[0]] = -0.1
    add("T4b2 no label by a MARGIN piece", run(x)["T4b2"]["word"], "no label")
    x = base()
    x["purity1"] = [i != b2[0] for i in range(n)]
    add("T4b2 no label by window 1 (purity on window 1)", run(x)["T4b2"]["word"], "no label")
    x = base()
    x["y1"][b2[1]] = 5.0
    add("T4b2 no label by window 1 (status)", run(x)["T4b2"]["word"], "no label")
    add("T4b2 no label by a missing budget term", run(base(), budget_ok=False)["T4b2"]["why"], "no label (budget input missing)")
    # withholding: K0 and blindness
    r = run(base(), withhold=["M5.52 C10 not met on this grid at 5.5"])
    add("withheld by K0 (C10)", (r["T4a"]["word"], r["T4b1"]["word"], r["T4b2"]["word"]), ("no label",) * 3)
    r = run(base(), withhold=["not blind"])
    add("withheld: not blind", (r["T4a"]["word"], "not blind" in r["T4a"]["why"]), ("no label", True))
    s52 = dict(stage0={k: dict(passed=True) for k in ("C1", "C2", "C3", "C13", "C14")}, C4=dict(passed=True), C6=dict(passed=True), C7=dict(passed=True),
               C8=dict(Ge=dict(passed=True, capped=False)), grids=dict(e=dict(C10={"5.5": dict(passed=True)}, C12={"5.5": True})),
               C9=dict(rows=[dict(job=j, tone=5.5, bounded=True, read=26, soft=0, worst_m2_error=0.05, wrong_status=0, edges=0, worst_c2_dev=0.0244 if j == "VD1" else 0.001,
                                  lines={}) for j in VAC]))
    add("K0: the speed clause of C9 alone withholds nothing", k0_reasons(s52, "e"), [])
    s52b = json.loads(json.dumps(s52))
    s52b["C9"]["rows"][0]["worst_m2_error"] = 0.5
    add("K0: the cutoff clause of C9 withholds", len(k0_reasons(s52b, "e")), 1)
    s52c = json.loads(json.dumps(s52))
    s52c["grids"]["e"]["C12"]["5.5"] = False
    add("K0: C12 withholds", k0_reasons(s52c, "e"), ["M5.52 C12 not met on this grid at 5.5"])
    s52d = json.loads(json.dumps(s52))
    s52d["C8"]["Ge"]["capped"] = True
    add("K0: a capped job withholds", k0_reasons(s52d, "e"), ["Ge capped"])
    # the plane
    add("plane HELD and no label: HELD", plane_word("HELD", "no label"), "HELD")
    add("plane HELD and NOT HELD: NOT HELD", plane_word("HELD", "NOT HELD"), "NOT HELD")
    add("plane no label and no label: no label", plane_word("no label", "no label"), "no label")
    # the status rule at its bounds
    add("status at y - eps = 0: PROP", status_of(0.96, 0.96), "PROP")
    add("status at y + eps = 0: MARGIN", status_of(-0.96, 0.96), "MARGIN")
    add("status of a missing budget: none", status_of(-3.0, None), None)
    met = sum(c["met"] for c in cases)
    return dict(cases=len(cases), met=met, rows=cases, passed=bool(met == len(cases)))


# ----------------------------------------------------------------------------- the expectations of section 6, from tracked files

def expect(pred):
    r49 = json.loads((DATA / "m5_49_reads.json").read_text())["grids"]
    damp = [KNOWN_DAMPING_TERM["I"], KNOWN_DAMPING_TERM["II"], KNOWN_DAMPING_TERM["III"]]
    out = {}
    for g in ("d", "e"):
        out[g] = {}
        for ln in LINES:
            pl = pred["blocks"][g]["lines"][ln]
            c = arr(pl["centers"])
            wc = arr(r49[g]["lines"][ln]["wc"])
            y = W * W - wc ** 2
            ok = np.isfinite(y)
            eps = np.array([max(FLOOR * W * W, damp[RD.bin_of(v, W)]) if np.isfinite(v) else np.nan for v in y])
            st = [status_of(y[i], eps[i]) for i in range(len(c))]
            rec = dict(T4a=[dict(s=float(c[i]), y=y[i], eps=eps[i], status=st[i]) for i in pl["T4a_scored"]])
            prof = {"y-eps": y - eps, "y": y, "y+eps": y + eps}
            rec["edges"] = {k: [round(e["r"], 3) for e in RD.edges0(c, p, ok, DELTA) if e["kind"] == "outer"] for k, p in prof.items()}
            rec["T4b2"] = [dict(s=float(c[i]), y=y[i], status=st[i]) for i in pl["T4b2_scored"]] if g == "e" else []
            kF = arr(pl["kappa_F"])
            kC = arr(pl["kappa_C"])
            c2 = arr(r49[g]["lines"][ln]["c2"])
            with np.errstate(invalid="ignore", divide="ignore"):
                kA = np.where(ok, np.sqrt(np.maximum(-y, 0)), np.nan)
                rec["ratios"] = [dict(s=float(c[i]), A_over_F=kA[i] / kF[i], P_over_F=kA[i] / np.sqrt(c2[i]) / kF[i], A_over_C=kA[i] / kC[i], c2_m5_49=c2[i])
                                 for i in pl["descriptor_pieces"]]
            rec["T4a_word_expected"] = t4a_window(pl["T4a_scored"], ok, st)[0] if pl["T4a_scored"] else None
            out[g][ln] = rec
    # the axis at omega = 4 on grid e: share of the power outside W on the pieces M5.49 read under 4 (UNDER)
    L = r49["e"]["lines"]["A"]
    sts = L["at"]["4.0"]["status"]
    ow = L["at"]["4.0"]["outside_W"]
    out["e"]["axis_outside_W_at_4_on_pieces_under_4"] = [ow[i] for i in range(len(sts)) if sts[i] == "UNDER"]
    # section 1 and 2 numbers
    sec = {}
    rat = []
    for g in GRIDS:
        for ln in ("A", "PF", "PD", "O"):
            Lr = r49[g]["lines"][ln]
            for i in range(len(Lr["centers"])):
                if Lr["read"][i] and Lr["q"][i] is not None and Lr["q"][i] < 2.25 and Lr["C"][i] is not None and Lr["mF"][i] is not None:
                    rat.append(Lr["C"][i] / Lr["mF"][i] - 1)
    sec["C_under_F_qbar_under_2_25"] = dict(pieces=len(rat), least=min(rat), most=max(rat))
    rr = []
    for ln in LINES:
        pl = pred["blocks"]["e"]["lines"][ln]
        Lr = r49["e"]["lines"][ln]
        for i in pl["descriptor_pieces"]:
            if Lr["wc"][i] is not None:
                rr.append(dict(line=ln, s=pl["centers"][i], wc_over_F=Lr["wc"][i] / Lr["mF"][i] - 1))
    sec["m5_49_cutoffs_beyond_outer_edges_e"] = rr
    ef = {}
    for ln in LINES:
        pl = pred["blocks"]["e"]["lines"][ln]
        c = arr(pl["centers"])
        mF = arr(pl["mF"])
        yF = W * W - mF ** 2
        dom = [i for i in range(len(c)) if pl["read_m5_49"][i] and np.isfinite(mF[i]) and mF[i] ** 2 <= 1.2 * W * W]
        r0 = pl["outer_edge_F"]
        last = max(i for i in dom if c[i] > r0)
        okd = np.array([i in dom for i in range(len(c))])
        pairs = RD.joined_pairs(c, okd, DELTA)
        fF = RD.phi_on(c, np.nan_to_num(-yF, nan=-1e9), pairs, r0, c[last])
        wc = arr(r49["e"]["lines"][ln]["wc"])
        y49 = W * W - wc ** 2
        ok49 = okd & np.isfinite(wc)
        p49 = RD.joined_pairs(c, ok49, DELTA)
        f49 = RD.phi_on(c, np.nan_to_num(-y49, nan=-1e9), p49, r0, c[last])
        e49 = [e["r"] for e in RD.edges0(c, np.where(ok49, y49, np.nan), ok49, DELTA) if e["kind"] == "outer"]
        f49e = RD.phi_on(c, np.nan_to_num(-y49, nan=-1e9), p49, e49[-1], c[last]) if e49 else None
        i0 = [i for i in range(len(c)) if abs(c[i] - 8.25) < 1e-9 or abs(c[i] - 8.75) < 1e-9] if ln == "A" else [i for i in range(len(c)) if abs(c[i] - 9.75) < 1e-9 or abs(c[i] - 10.25) < 1e-9]
        slope = (yF[i0[0]] - yF[i0[1]]) / DELTA
        ef[ln] = dict(outer_edge_F=r0, domain_end=float(c[last]), efolds_F=fF, efolds_m5_49_from_F_edge=f49, efolds_m5_49_from_its_edge=f49e,
                      largest_abs_yF_beyond_edge_in_domain=float(np.nanmax(np.abs(yF[[i for i in dom if c[i] > r0]]))), slope_yF_across_interval=float(slope))
    sec["efolds_and_slopes_e"] = ef
    z = np.sqrt(complex(-3.025, W * GAMMA))
    sec["vacuum_VD1"] = dict(kappa=float(np.sqrt(3.025)), Im_sqrtZ=float(z.imag), Re_K=float(z.real))
    ss = np.linspace(2, 4.5, 201)
    sec["spreading_ln_s_slope_2_to_4_5"] = dict(slope=float(np.polyfit(ss, np.log(ss), 1)[0]), inverse_mean_s=float(1 / ss.mean()))
    yy = np.array([5.8, 2.2])
    r = W * GAMMA / yy
    sec["D_of_r"] = dict(abs_y=yy.tolist(), D=np.sqrt((np.sqrt(1 + r ** 2) + 1) / 2).tolist())
    out["sections_1_to_3"] = sec
    out = PR.jsonable(out)
    (WORK / "expectations.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


# ----------------------------------------------------------------------------- plots (phase 2)

def plots(summary=None, plots_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = json.loads(PRED.read_text())
    S = json.loads(Path(summary or SUMMARY).read_text())
    R = json.loads((WORK / "reads" / "Ge.json").read_text())
    plots_dir = Path(plots_dir or (HERE.parent / "plots"))
    for ln in LINES:
        gl = S["grids"]["e"]["lines"][ln]
        pl = pred["blocks"]["e"]["lines"][ln]
        c = arr(gl["centers"])
        y, eps = arr(gl["y"]), arr(gl["eps"])
        lo, hi = INTERVAL[ln]
        fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
        ax = axs[0, 0]
        ax.axvspan(lo, hi, color="0.85", lw=0, label="the author's interval")
        ax.axhline(0, color="0.45", lw=0.8)
        yF, yC = arr(pl["y_F"]), arr(pl["y_C"])
        m = np.isfinite(yF)
        ax.plot(c[m], yF[m], color=COL_F, lw=1.6, label=r"F: $\omega^2 - m_F^2$")
        mc = np.isfinite(yC)
        ax.plot(c[mc], yC[mc], "--", color=COL_C, lw=1.4, label=r"C: $\omega^2 - C^2$")
        k = np.isfinite(y)
        ax.errorbar(c[k], y[k], yerr=eps[k], fmt="o", ms=4, color=COL_READ, ecolor=COL_READ, elinewidth=1.0, capsize=2, label=r"read $y$ with $\varepsilon$ (window 2)")
        for E3, ls_, nm in zip(gl["read_edges"], ("-", ":", "-."), ("y", "y + eps", "y - eps")):
            outs = [e["r"] for e in E3 if e["kind"] == "outer"]
            if outs:
                ax.axvline(nearest_to(outs, lo, hi), color=COL_READ, ls=ls_, lw=1.0, label=f"outer edge of {nm}")
        ax.set_xlim(max(0, lo - 4), min(c[-1] + 0.5, hi + 4))
        ax.set_ylim(-14, 12)
        ax.set_xlabel("s along the line")
        ax.set_ylabel(r"$y = \omega^2 - m^2$")
        ax.set_title(f"T4b1 ({gl['words'].get('T4b1', {}).get('word', '-')}): y with its budget and the outer edges", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.25, lw=0.5)
        ax = axs[0, 1]
        kA, kP = arr(gl["kappa_A"]), arr(gl["kappa_P"])
        sel = c > (pl["outer_edge_F"] or 0) - 1.0
        with np.errstate(invalid="ignore"):
            klo = np.sqrt(np.maximum(-(y + eps), 0))
            khi = np.sqrt(np.maximum(-(y - eps), 0))
        kk = sel & np.isfinite(kA)
        ax.fill_between(c[kk], klo[kk], khi[kk], color=COL_READ, alpha=0.15, lw=0, label=r"$\kappa_A$ over $y \pm \varepsilon$")
        ax.plot(c[kk], kA[kk], "o-", ms=4, lw=1.0, color=COL_READ, label=r"$\kappa_A = \sqrt{-y}$")
        ax.plot(c[kk], kP[kk], "s-", ms=4, lw=1.0, color=COL_P, label=r"$\kappa_P = \sqrt{-\mathrm{Re}\,Z}$")
        kF, kC = arr(pl["kappa_F"]), arr(pl["kappa_C"])
        ax.plot(c[sel & np.isfinite(kF)], kF[sel & np.isfinite(kF)], "^-", ms=4, lw=1.4, color=COL_F, label=r"$\kappa_F$")
        ax.plot(c[sel & np.isfinite(kC)], kC[sel & np.isfinite(kC)], "v--", ms=4, lw=1.4, color=COL_C, label=r"$\kappa_C$")
        b2s = pl["T4b2_scored_s"]
        ax.axvspan(b2s[0] - DELTA / 2, b2s[-1] + DELTA / 2, color="0.9", lw=0, label="the T4b2 pieces")
        ax.set_xlabel("s along the line")
        ax.set_ylabel("decay rate")
        ax.set_title("T4c (no word): the rate profile beyond the outer edge of F", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25, lw=0.5)
        rl = R["lines"][ln]
        rays = rl["rays"]
        sr = arr(rays["s"])
        ax = axs[1, 0]
        for j, mo in enumerate(rays["modulus"]):
            ax.plot(sr, np.log(arr(mo)), lw=0.9, color=COL_READ, alpha=0.6, label="ln |a| on each ray (window 2)" if j == 0 else None)
        rS = gl.get("read_S")
        if rS is not None and "read_S" in rl:
            ss = arr(rl["read_S"]["s"])
            for j, l2 in enumerate(rl["read_S"]["ln_a2"]):
                p = np.polyfit(ss, arr(l2), 1)
                ax.plot(ss, np.polyval(p, ss), color=COL_C, lw=2.0, label="read S: least-squares line on the span" if j == 0 else None)
        ax.axvspan(lo, hi, color="0.85", lw=0)
        ax.set_xlabel("s along the ray")
        ax.set_ylabel("ln |a|")
        kS = rS["kappa_S"] if rS else None
        ax.set_title(f"read S (no word): kappa_S = {kS:.3f}" if kS is not None else "read S (no word): not read", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.25, lw=0.5)
        ax = axs[1, 1]
        for j, ph in enumerate(rays["phase"]):
            ax.plot(sr, arr(ph), lw=0.9, color=COL_READ, alpha=0.6, label=r"$\varphi = \arg a$ on each ray (window 2)" if j == 0 else None)
        ax.axvspan(lo, hi, color="0.85", lw=0)
        ax.set_xlabel("s along the ray")
        ax.set_ylabel("phase (radians, unwrapped)")
        ax.set_title("T4a (no word): the literal phase", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25, lw=0.5)
        fig.suptitle(f"M5.55, grid e, line {ln}, omega = 5.5", fontsize=11)
        fig.savefig(plots_dir / f"m5_55_e_{ln}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print("plots written", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--k4", action="store_true")
    ap.add_argument("--expect", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--words", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--hash", nargs="*", default=[])
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    hashes = dict(h.split("=", 1) for h in a.hash)
    if a.predict:
        doc, fh = predict()
        print("file", fh)
        print("blocks", json.dumps(doc["hashes"], sort_keys=True))
    if a.k4:
        r = k4()
        (WORK / "k4.json").write_text(json.dumps(PR.jsonable(r), indent=1) + "\n")
        print(f"K4: {r['met']} of {r['cases']} met, passed {r['passed']}")
        for c in r["rows"]:
            if not c["met"]:
                print("   NOT MET:", c)
    if a.expect:
        r = expect(json.loads(PRED.read_text()))
        print(json.dumps(r, indent=1)[:20000])
    if a.read:
        pred = check_own_hashes(hashes)
        if not M52_SUMMARY.exists():
            raise SystemExit("m5_52_summary.json does not exist; the reads wait for it")
        for name in [GRID_JOB[g] for g in ("e", "d", "b")] + list(VAC):
            read_job(name, pred)
            print("read", name, flush=True)
    if a.words:
        pred = check_own_hashes(hashes)
        s52_sha = sha_file(M52_SUMMARY)
        words(pred, json.loads(M52_SUMMARY.read_text()), s52_sha)
    if a.plots:
        plots()


if __name__ == "__main__":
    main()
