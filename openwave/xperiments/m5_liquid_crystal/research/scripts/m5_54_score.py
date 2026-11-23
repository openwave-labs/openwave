"""M5.54: the charge read of m5_54_charge.py on the stored states of M5.51, the words of section 3 of the plan against
rung T2, the tally, the tables, the summary JSON and the plots.

Reads (no relaxation): the 24 end states of M5.51 (`checkpoints/m5_51/chains/<chain>/end.npz`), their earlier states
(the saved state of M5.51's second step, or the start state rebuilt by `m5_51_relax.start_state`), the author's four
stored arrays and the seeds. M5.51's files are imported and read, never written.

Words (plan, section 3), on the midpoint chain `a` of the T2 grids G1, G2, G4, G5, with chain `b` as a veto:

  U        |U / U_T2 - 1| <= 0.00966 INSIDE, >= 0.01034 OUTSIDE (G1, G2, G4; U_T2 = 20.50, 20.53, 21.50)
  radius   every available reader (R_L, R_A with 4 nodes or more, R_rel on G1, G2, G4) inside [2.76, 2.86] by more
           than its budget: INSIDE; every one outside by more than its budget: OUTSIDE; otherwise no label (readers)
  Q/4pi    Q(L)/4pi >= 0.992 and Q(l2)/4pi >= 0.9944: INSIDE (fixed by the held faces); Q(L)/4pi < 0.988 and
           Q(l2)/4pi < 0.9856: OUTSIDE; otherwise, or out of regime, no label
  N/U      in [0.941, 0.999] INSIDE, under 0.939 or above 1.001 OUTSIDE

Usage:
  python m5_54_score.py --reads            the charge reads of every state -> checkpoints/m5_54/reads.json
  python m5_54_score.py --summary          the words, the tally, data/m5_54_summary.json and the plots
"""

import argparse
import hashlib
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
import m5_51_reads as RD  # noqa: E402  (read-only import)
import m5_51_relax as RX  # noqa: E402  (read-only import: the start states)
import m5_54_charge as CH  # noqa: E402

CK = HERE.parent / "checkpoints" / "m5_54"
CK51 = HERE.parent / "checkpoints" / "m5_51" / "chains"
DATA = HERE.parent / "data"
PLOTS = HERE.parent / "plots"
GRIDS = CH.GRIDS
CHAINS = [f"{g}_{w}_{s}" for g in GRIDS for w in ("mid", "end") for s in ("a", "b")]
T2_GRIDS = ("G1", "G2", "G4", "G5")
U_T2 = dict(G1=20.50, G2=20.53, G4=21.50)
RANGE = (2.76, 2.86)
RELEASE_R = dict(G1=2.73, G2=2.74, G4=2.83)
RA_BUDGET_PLAN = {(0.5, "axis"): 0.020, (0.375, "axis"): 0.033, (0.5, "diag"): 0.089, (0.375, "diag"): 0.048,
                  (0.5, "neither"): 0.123, (0.375, "neither"): 0.048}
FIXED_TABLE_PLAN = dict(
    G1=dict(U="INSIDE", radius="no label (readers)", N_over_U="OUTSIDE"),
    G2=dict(U="no label (second step)", radius="no label (second step)", N_over_U="no label (second step)"),
    G4=dict(U="INSIDE", radius="no label (readers)", N_over_U="INSIDE"),
    G5=dict(U=None, radius="no label (readers)", N_over_U="INSIDE"))
COLORS = dict(end="#2a78d6", mid="#eb6834", author="#1baf7a", ref="#8a8a85")


def s51():
    return json.loads((DATA / "m5_51_summary.json").read_text())


def controls():
    """The latest try of the controls (try2 if present)."""
    for t in ("try3", "try2", "try1"):
        f = CK / f"controls_{t}.json"
        if f.exists():
            return t, json.loads(f.read_text())
    raise SystemExit("no controls file: run m5_54_charge.py --selftest first")


# ----------------------------------------------------------------------------- the states

def chain_rec(name):
    return json.loads((CK51 / f"{name}.json").read_text())


def end_state(name):
    rec = chain_rec(name)
    z = np.load(CK51 / name / rec["end_file"], allow_pickle=False)
    return np.asarray(z["m6"], float), np.asarray(z["x"], float), rec


def earlier_state(name):
    rec = chain_rec(name)
    it = rec["earlier"]["iteration"]
    g, w, s = name.split("_")
    hb, h, N = GRIDS[g]
    if rec["earlier"]["kind"] == "saved state":
        for f in (f"ckpt_{it:05d}.npz", f"s{it:05d}.npz"):
            p = CK51 / name / f
            if p.exists():
                z = np.load(p, allow_pickle=False)
                return np.asarray(z["m6"], float), np.asarray(z["x"], float), dict(iteration=it, source=f)
        raise FileNotFoundError(f"{name}: no saved state at {it}")
    mod = A.Model(N, hb, weight=w)
    m6, info = RX.start_state(name, mod)
    return np.asarray(m6, float), mod.x, dict(iteration=0, source="start state rebuilt by m5_51_relax.start_state", info=info)


def energy(m6, x, hb, w):
    return A.Model(len(x), hb, weight=w).evaluate(m6, need_grad=False)["U"]


def read_one(M, x, RL, center, normal, cls):
    circ = (center, normal, RL) if (RL is not None and center is not None and normal is not None) else None
    r = CH.charge_read(M, x, RL=RL if RL is not None else None, circle=circ)
    out = {k: v for k, v in r.items() if not k.startswith("_")}
    if cls == "axis" and normal is not None:
        k = CH.turns_to_z(normal)
        rr = CH.charge_read(CH.cyclic(M, k), x, archive=True)
        out["fourth_reader"] = dict(turns=k, **rr["archive_read"])
    return out


def reads(out_file):
    S = s51()
    res = dict(chains={}, author={}, seeds={})
    for name in CHAINS:
        g, w, s = name.split("_")
        hb, h, N = GRIDS[g]
        c = S["chains"][name]
        m6, x, rec = end_state(name)
        U_chk = energy(m6, x, hb, w)
        re = read_one(A.from6(m6), x, c["RL"], c["center"], c["normal"], c["cls"])
        re["U_recomputed"] = U_chk
        re["U_summary"] = c["U"]
        er_m6, er_x, er_info = earlier_state(name)
        Ue = energy(er_m6, er_x, hb, w)
        ere = rec["earlier"]
        rr = read_one(A.from6(er_m6), er_x, ere.get("RL"), ere.get("center"), ere.get("normal"), ere.get("cls"))
        rr.update(U_recomputed=Ue, U_record=ere.get("U"), info=er_info, RA=ere.get("RA"), RA_nodes=ere.get("RA_nodes"), RL=ere.get("RL"),
                  cls=ere.get("cls"), line_regime=ere.get("line_regime"), halfturn_all=ere.get("halfturn_all"))
        res["chains"][name] = dict(end=re, earlier=rr)
        print(name, f"U {U_chk:.9f} (summary {c['U']:.9f})", f"Q {re['Q_over_4pi']:.6f} {re['Q_l2_over_4pi']:.6f}", "regime", re["regime"]["in_regime"],
              f"| earlier it {er_info['iteration']}: U {Ue:.9f} (record {ere.get('U')})", f"Q {rr['Q_over_4pi']:.6f} {rr['Q_l2_over_4pi']:.6f}",
              "regime", rr["regime"]["in_regime"], flush=True)
    for g in ("G1", "G2", "G4", "G6"):
        M, x, hb = CH._stored_raw(g)
        st = S["stored"][g]
        r = read_one(M, x, st["RL"], st["center"], st["normal"], st["cls"])
        res["author"][g] = r
        print("author", g, f"Q {r['Q_over_4pi']:.7f} {r['Q_l2_over_4pi']:.6f}", r["regime"]["in_regime"], flush=True)
    for g in GRIDS:
        hb, h, N = GRIDS[g]
        x = np.linspace(-hb, hb, N)
        r = CH.charge_read(CH.seed_M(N, hb), x, RL=0.0)
        res["seeds"][g] = {k: v for k, v in r.items() if not k.startswith("_")}
        print("seed", g, f"Q {r['Q_over_4pi']:.6f} {r['Q_l2_over_4pi']:.6f}", flush=True)
    out_file.write_text(json.dumps(res, indent=1, default=CH._jsonable) + "\n")
    print("reads written", out_file.name, flush=True)


# ----------------------------------------------------------------------------- the R_A budgets (K4 (a) of M5.51, recomputed)

def ra_budgets():
    L = 6.0
    rows = {}
    for h in (0.5, 0.375):
        N = int(round(2 * L / h)) + 1
        x = np.linspace(-L, L, N)
        for on, (nu, e1) in RD.ORIENT.items():
            errs = []
            for off in (0.0, 1 / 3, 2 / 3):
                R0 = 2.75 + off * h
                ra = RD.R_A(RD.ring_field(x, R0, nu, e1), x, L)
                errs.append(ra["R"] - R0)
            rows[(h, on)] = max(abs(e) for e in errs)
    out = {}
    for h in (0.5, 0.375):
        out[(h, "axis")] = rows[(h, "axis")]
        out[(h, "diag")] = rows[(h, "diag")]
        out[(h, "neither")] = max(rows[(h, o)] for o in ("axis", "diag", "tilt16"))
    return out, rows


# ----------------------------------------------------------------------------- the words

def _radius_rule(vals):
    """vals: list of (name, value, budget). INSIDE, OUTSIDE or no label (readers), and the margins."""
    lo, hi = RANGE
    ins, outs, margins = [], [], []
    for n, v, b in vals:
        ins.append(lo + b < v < hi - b)
        outs.append(v < lo - b or v > hi + b)
        margins.append(dict(reader=n, value=v, budget=b, inside_by=min(v - lo, hi - v), inside=ins[-1], outside=outs[-1]))
    if ins and all(ins):
        return "INSIDE", margins
    if outs and all(outs):
        return "OUTSIDE", margins
    return "no label (readers)", margins


def _u_rule(U, UT):
    d = abs(U / UT - 1)
    if d <= 0.00966:
        return "INSIDE", d
    if d >= 0.01034:
        return "OUTSIDE", d
    return "no label (between)", d


def _q_rule(q):
    if not q["regime"]["in_regime"]:
        return "no label (out of regime)"
    a, b = q["Q_over_4pi"], q["Q_l2_over_4pi"]
    if a >= 0.992 and b >= 0.9944:
        return "INSIDE (fixed by the held faces)"
    if a < 0.988 and b < 0.9856:
        return "OUTSIDE"
    return "no label (between)"


def _nu_rule(v):
    if 0.941 <= v <= 0.999:
        return "INSIDE"
    if v < 0.939 or v > 1.001:
        return "OUTSIDE"
    return "no label (between)"


def gives(w):
    return not w.startswith("no label")


def radius_readers(g, c, RA_B, stored):
    h = GRIDS[g][1]
    vals = [("R_L", c["RL"], 0.01)]
    if c.get("RA_nodes") is not None and c["RA_nodes"] >= 4 and c.get("RA") is not None:
        cls = c["cls"]
        vals.append(("R_A", c["RA"], RA_B[(h, cls)]))
    if g in RELEASE_R:
        rrel = c["RL"] - (stored[g]["RL"] - RELEASE_R[g])
        vals.append(("R_rel", rrel, 0.015))
    return vals


def words(S, R, ctl_ok, RA_B):
    out = {}
    stored = S["stored"]
    for g in T2_GRIDS:
        a = S["chains"][f"{g}_mid_a"]
        b = S["chains"][f"{g}_mid_b"]
        ra = R["chains"][f"{g}_mid_a"]
        rb = R["chains"][f"{g}_mid_b"]
        ss = a["second_step"]
        qual = bool(a["qualifies"] and a.get("line_regime") and a.get("halfturn_all"))
        ent = {}

        def veto(word_a, word_b):
            if gives(word_a) and word_b != word_a:
                return "no label (starts disagree)"
            return word_a

        # U
        if g in U_T2:
            UT = U_T2[g]
            wa, da = _u_rule(a["U"], UT)
            wb, db = _u_rule(b["U"], UT)
            if not qual:
                w = "no label (not qualifying)"
            elif not ss["U"]["ok"]:
                w = "no label (second step)"
            else:
                w = veto(wa, wb)
            ent["U"] = dict(word=w, value_a=a["U"], U_T2=UT, rel_a=a["U"] / UT - 1, value_b=b["U"], rel_b=b["U"] / UT - 1, rule_word_a=wa, rule_word_b=wb,
                            second_step=ss["U"])
        else:
            ent["U"] = dict(word="tabled, no entry", value_a=a["U"], value_b=b["U"])
        # radius
        va = radius_readers(g, a, RA_B, stored)
        vb = radius_readers(g, b, RA_B, stored)
        wa, ma = _radius_rule(va)
        wb, mb = _radius_rule(vb)
        ra_step = None
        if any(n == "R_A" for n, _, _ in va):
            ea = R["chains"][f"{g}_mid_a"]["earlier"]
            er_RA = ea.get("RA") if (ea.get("RA_nodes") or 0) >= 4 else None
            ra_step = dict(end=a["RA"], earlier=er_RA, ok=bool(er_RA is not None and abs(a["RA"] - er_RA) <= 0.001))
        step_ok = bool(ss["RL"]["ok"] and ss["cls"]["ok"] and (ra_step is None or ra_step["ok"]))
        if not qual:
            w = "no label (not qualifying)"
        elif not step_ok:
            w = "no label (second step)"
        else:
            w = veto(wa, wb)
        ent["radius"] = dict(word=w, readers_a=ma, readers_b=mb, rule_word_a=wa, rule_word_b=wb, second_step=dict(RL=ss["RL"], cls=ss["cls"], RA=ra_step))
        # Q
        qa, qb = ra["end"], rb["end"]
        qe = ra["earlier"]
        q_step = dict(Q_L=dict(end=qa["Q_over_4pi"], earlier=qe["Q_over_4pi"], ok=bool(abs(qa["Q_over_4pi"] - qe["Q_over_4pi"]) <= 1e-4)),
                      Q_l2=dict(end=qa["Q_l2_over_4pi"], earlier=qe["Q_l2_over_4pi"], ok=bool(abs(qa["Q_l2_over_4pi"] - qe["Q_l2_over_4pi"]) <= 1e-4)),
                      earlier_in_regime=qe["regime"]["in_regime"], one_ring=ss["one_ring"])
        q_ok = bool(q_step["Q_L"]["ok"] and q_step["Q_l2"]["ok"] and q_step["earlier_in_regime"] and ss["one_ring"]["ok"])
        wqa, wqb = _q_rule(qa), _q_rule(qb)
        if not ctl_ok:
            w = "tabled (a control failed)"
        elif not qual:
            w = "no label (not qualifying)"
        elif not q_ok:
            w = "no label (second step)"
        else:
            w = veto(wqa, wqb)
        ent["Q"] = dict(word=w, Q_L_a=qa["Q_over_4pi"], Q_l2_a=qa["Q_l2_over_4pi"], Q_L_b=qb["Q_over_4pi"], Q_l2_b=qb["Q_l2_over_4pi"],
                        in_regime_a=qa["regime"]["in_regime"], in_regime_b=qb["regime"]["in_regime"], rule_word_a=wqa, rule_word_b=wqb,
                        second_step=q_step, seed=dict(Q_L=R["seeds"][g]["Q_over_4pi"], Q_l2=R["seeds"][g]["Q_l2_over_4pi"]))
        # N/U
        wa, wb = _nu_rule(a["N_over_U"]), _nu_rule(b["N_over_U"])
        if not qual:
            w = "no label (not qualifying)"
        elif not ss["N_over_U"]["ok"]:
            w = "no label (second step)"
        else:
            w = veto(wa, wb)
        ent["N_over_U"] = dict(word=w, value_a=a["N_over_U"], value_b=b["N_over_U"], rule_word_a=wa, rule_word_b=wb, second_step=ss["N_over_U"])
        out[g] = dict(chain_a=a["chain"], chain_b=b["chain"], qualifies=qual, entries=ent)
    return out


def tally(W):
    cols = ("INSIDE", "INSIDE (fixed by the held faces)", "OUTSIDE", "no label")
    per = {}
    tot = {c: 0 for c in cols}
    for read in ("U", "radius", "Q", "N_over_U"):
        row = {c: 0 for c in cols}
        for g in T2_GRIDS:
            w = W[g]["entries"][read]["word"]
            if w.startswith("tabled"):
                continue
            if w == "INSIDE (fixed by the held faces)":
                k = cols[1]
            elif w in ("INSIDE", "OUTSIDE"):
                k = w
            else:
                k = "no label"
            row[k] += 1
            tot[k] += 1
        per[read] = row
    return dict(per_read=per, total=tot, entries=sum(tot.values()))


def fixed_table_check(W):
    diffs = []
    for g, d in FIXED_TABLE_PLAN.items():
        for read, plan_w in d.items():
            got = W[g]["entries"][read]["word"]
            if plan_w is None:
                plan_w = "tabled, no entry"
            if got != plan_w:
                diffs.append(dict(grid=g, read=read, plan=plan_w, run=got))
    return diffs


# ----------------------------------------------------------------------------- the plots

def plots(S, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import map_coordinates
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6})
    files = {}
    # 1. the profile Q(l)/4pi against l, per grid
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2), sharey=True)
    for k, g in enumerate(GRIDS):
        ax = axs[k // 3, k % 3]
        for w in ("mid", "end"):
            for s in ("a", "b"):
                p = R["chains"][f"{g}_{w}_{s}"]["end"]["profile"]
                ax.plot([q["l"] for q in p], [q["Q_over_4pi"] for q in p], color=COLORS[w], ls="-" if s == "a" else "--", lw=1.4 if s == "a" else 1.1,
                        marker="o" if s == "a" else None, ms=3, label=f"{w} weight, chain {s}")
        if g in R["author"]:
            p = R["author"][g]["profile"]
            ax.plot([q["l"] for q in p], [q["Q_over_4pi"] for q in p], color=COLORS["author"], ls=":", lw=1.4, label="the author's stored array")
        p = R["seeds"][g]["profile"]
        ax.plot([q["l"] for q in p], [q["Q_over_4pi"] for q in p], color=COLORS["ref"], lw=1.0, label="the seed (no ring)")
        RL = S["chains"][f"{g}_mid_a"]["RL"]
        ax.axvline(RL, color=COLORS["ref"], lw=0.8, ls="-.")
        ax.axvline(GRIDS[g][0] - 2 * GRIDS[g][1], color="#444441", lw=0.8, ls=(0, (1, 2)))
        ax.set_title(f"{g} (half-box {GRIDS[g][0]:g}, h {GRIDS[g][1]:g})", fontsize=8)
        ax.set_xlabel("cube half-width l")
        if k % 3 == 0:
            ax.set_ylabel("Q(l) / 4 pi")
    axs[0, 0].legend(fontsize=6, frameon=False, loc="lower right")
    fig.suptitle("Q(l)/4pi on centered cubes; dash-dot: R_L of the midpoint chain a; dotted: l2 = L - 2h", fontsize=8)
    fig.tight_layout()
    f = PLOTS / "m5_54_profile.png"
    fig.savefig(f, dpi=100)
    plt.close(fig)
    files["profile"] = f.name
    # 2. rho on a meridian plane of the four T2 midpoint states (chain a)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, g in zip(axs, T2_GRIDS):
        name = f"{g}_mid_a"
        c = S["chains"][name]
        m6, x, _ = end_state(name)
        r = CH.charge_read(A.from6(m6), x, RL=c["RL"], keep=True)
        rho = r["_rho"]
        nu, e1, _ = RD.frame(c["normal"])
        cc = np.array(c["center"])
        Lp = min(x[-1] - 1.0, 5.0)
        t = np.linspace(-Lp, Lp, 241)
        Sg, Zg = np.meshgrid(t, t, indexing="ij")
        P = cc + Sg[..., None] * e1 + Zg[..., None] * nu
        hh = x[1] - x[0]
        idx = [(P[..., a] - x[0]) / hh for a in range(3)]
        val = map_coordinates(rho, idx, order=1, mode="nearest")
        vm = float(np.percentile(np.abs(rho), 99.9))
        im = ax.imshow(val.T, origin="lower", extent=(-Lp, Lp, -Lp, Lp), cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax.plot([-c["RL"], c["RL"]], [0, 0], ls="none", marker="+", color="#222", ms=8)
        ax.set_title(f"{name}  R_L {c['RL']:.3f}, h {hh:g}", fontsize=8)
        ax.set_xlabel("in-plane distance from the fitted center")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.75)
    axs[0].set_ylabel("along the fitted normal")
    fig.suptitle("rho = div b on the meridian plane through the fitted axis (trilinear on the nodes; color scale at the 99.9th percentile of |rho|; + marks the fitted circle)", fontsize=8)
    fig.tight_layout()
    f = PLOTS / "m5_54_rho_meridian.png"
    fig.savefig(f, dpi=100)
    plt.close(fig)
    files["rho_meridian"] = f.name
    return files


# ----------------------------------------------------------------------------- the summary

def _trim(r):
    return {k: v for k, v in r.items() if k not in ("identity",)} | dict(identity_max_rel=r["identity"]["max_rel"])


def summary():
    S = s51()
    R = json.loads((CK / "reads.json").read_text())
    tag, C = controls()
    ctl_ok = bool(C["passed"])
    RA_B, RA_rows = ra_budgets()
    ra_check = {f"{h}_{c}": dict(run=v, plan=RA_BUDGET_PLAN[(h, c)], diff=v - RA_BUDGET_PLAN[(h, c)], agrees=bool(abs(v - RA_BUDGET_PLAN[(h, c)]) <= 0.001))
                for (h, c), v in RA_B.items()}
    # the budget used is the larger of the recomputed value and the plan's (stricter in both directions)
    RA_use = {k: max(v, RA_BUDGET_PLAN[k]) for k, v in RA_B.items()}
    W = words(S, R, ctl_ok, RA_use)
    T = tally(W)
    diffs = fixed_table_check(W)
    # C3 on every field read here
    worst = max([r["end"]["identity"]["max_rel"] for r in R["chains"].values()] + [r["earlier"]["identity"]["max_rel"] for r in R["chains"].values()]
                + [r["identity"]["max_rel"] for r in R["author"].values()] + [r["identity"]["max_rel"] for r in R["seeds"].values()])
    # the U checks of the loaded states against M5.51's record
    ucheck = {n: dict(end=abs(r["end"]["U_recomputed"] / r["end"]["U_summary"] - 1),
                      earlier=(abs(r["earlier"]["U_recomputed"] / r["earlier"]["U_record"] - 1) if r["earlier"].get("U_record") else None))
              for n, r in R["chains"].items()}
    tabled = {}
    for n in CHAINS:
        c = S["chains"][n]
        e = R["chains"][n]["end"]
        er = R["chains"][n]["earlier"]
        tabled[n] = dict(U=c["U"], RL=c["RL"], RA=c["RA"], RA_nodes=c["RA_nodes"], cls=c["cls"], N_over_U=c["N_over_U"], qualifies=c["qualifies"],
                         Q_L=e["Q_over_4pi"], Q_l2=e["Q_l2_over_4pi"], in_regime=e["regime"]["in_regime"],
                         regime=dict(L=e["regime"]["L"], l2=e["regime"]["l2"], radius_clause=e["regime"]["radius_clause"]),
                         shares=e.get("shares"), fourth_reader=e.get("fourth_reader"), lift_flip_nodes=e["lift_flip_nodes"],
                         profile=[[q["l"], q["Q_over_4pi"]] for q in e["profile"]],
                         earlier=dict(iteration=er["info"]["iteration"], source=er["info"]["source"], Q_L=er["Q_over_4pi"], Q_l2=er["Q_l2_over_4pi"],
                                      in_regime=er["regime"]["in_regime"], shares=er.get("shares"), RA=er.get("RA"), RA_nodes=er.get("RA_nodes")))
    author = {}
    for g, r in R["author"].items():
        author[g] = dict(Q_L=r["Q_over_4pi"], Q_l2=r["Q_l2_over_4pi"], in_regime=r["regime"]["in_regime"],
                         regime=dict(L=r["regime"]["L"], l2=r["regime"]["l2"], radius_clause=r["regime"]["radius_clause"]),
                         shares=r.get("shares"), fourth_reader=r.get("fourth_reader"), profile=[[q["l"], q["Q_over_4pi"]] for q in r["profile"]],
                         archive=CH.ARCHIVE.get(g), RL=S["stored"][g]["RL"])
    seeds = {g: dict(Q_L=r["Q_over_4pi"], Q_l2=r["Q_l2_over_4pi"], in_regime=r["regime"]["in_regime"]) for g, r in R["seeds"].items()}
    c1 = C["C1"]
    files = plots(S, R)
    out = dict(task="M5.54", plan_commit="44660227", controls_try=tag, controls_passed=ctl_ok,
               archive_sha256=CH.ARCHIVE_SHA256,
               controls=dict(C1=dict(passed=c1["passed"], diffs=c1["diffs"], Q_rotated_minus_unrotated=c1["Q_rotated_minus_unrotated"],
                                     failing_inputs=c1["failing_inputs"], fitted_circle_shares=c1["fitted_circle_shares"], values=c1["values"]),
                             C2={g: {k: v for k, v in d.items()} for g, d in C["C2"]["arrays"].items()} | dict(passed=C["C2"]["passed"]),
                             C3=dict(passed=C["C3"]["passed"], worst_rel_controls=C["C3"]["worst_rel"], worst_rel_states=worst, failing_inputs=C["C3"]["failing_inputs"]),
                             C4a=C["C4a"], C4b={k: v for k, v in C["C4b"].items()}, C4c=C["C4c"], regime_row=C["regime_row"]),
               RA_budgets=dict(recomputed=ra_check, per_orientation={f"{h}_{o}": v for (h, o), v in RA_rows.items()},
                               used={f"{h}_{c}": v for (h, c), v in RA_use.items()}),
               U_check_of_loaded_states=ucheck,
               words=W, tally=T, fixed_table_differences=diffs, tabled_states=tabled, author_arrays=author, seeds=seeds, plots=files,
               posted_shares_end_weight="93 to 96 percent within 1 of the line (post of 2026-11-20, item 2)", confidence_T2="80 percent (tabled, not scored)")
    f = DATA / "m5_54_summary.json"
    f.write_text(json.dumps(out, indent=1, default=CH._jsonable) + "\n")
    print("summary written", f.name, f.stat().st_size, "bytes, sha256", hashlib.sha256(f.read_bytes()).hexdigest())
    print("controls", tag, "passed" if ctl_ok else "FAILED")
    print("R_A budgets", json.dumps(ra_check))
    for g in T2_GRIDS:
        print(g, {k: v["word"] for k, v in W[g]["entries"].items()})
    print("tally", json.dumps(T))
    print("fixed-table differences", diffs)
    print("C3 worst on the states", worst)
    print("U check worst", max(v["end"] for v in ucheck.values()), max(v["earlier"] for v in ucheck.values() if v["earlier"] is not None))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.reads:
        CK.mkdir(parents=True, exist_ok=True)
        reads(CK / "reads.json")
        return
    if a.summary:
        summary()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
