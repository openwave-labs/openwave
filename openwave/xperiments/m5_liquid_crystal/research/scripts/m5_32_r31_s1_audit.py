"""M5.32 R31 = S1 independent audit: recompute the headline claims from the raw record, not the run's collect code.

DEFINITIONS (every symbol as used below)
  accepted(state): label-free; degree d = cond1_degree_top["12"]["degree"] (None = undefined), accepted iff
      d is not None and |abs(d) - 1| <= 0.1, E_J finite, |E_stat| <= 1e3.
  Per point: E1 <= E2 <= E3 = the three lowest E_J among accepted states (J >= 0, all seeds, all J).
      ratios E2/E1, E3/E1; Koide Q = (E1 + E2 + E3) / (sqrt E1 + sqrt E2 + sqrt E3)^2.
  Degree census over J > 0 rows: for each row the set of cubes {6, 9, 12} with a defined degree.
  Sorted-spectrum field: lam(x) = sort(eigvalsh(S(x))), S = M[..., 1:, 1:], over free cells (outside the pinned shell).
      rms_sorted(A, B) = sqrt(mean over free cells and 3 slots of (lamA - lamB)^2); also the per-cell-norm variant
      sqrt(mean over cells of |lamA - lamB|_2^2) (a factor sqrt 3 larger).
      rms_raw(A, B) = sqrt(mean over free cells and 9 tensor entries of (SA - SB)^2) and the per-cell Frobenius variant.
  Frame-jump fraction: nearest-neighbor links (+x, +y, +z) with both ends at r < 14; big = |dS|_F > 0.3;
      jump = big and |d lam|_2 < f |dS|_F; fraction = jump / n_links (the run's definition, f = 0.5) and also
      jump / big. Reported at f = 0.3, 0.5, 1.0.
  Ladder: ratio E_J(J_1)/E_J(0) and E_J(2 J_1)/E_J(0), per seed.
  Closure: slope = (E(1.05 J_1) - E(0.95 J_1)) / (0.1 J_1), compared with omega* = J_1 / (2 kin_rad(J_1)) read from rows.
SECOND SECTION (lifts, R31-1b), same record, six points with kind "lift":
  lift densities per cell on the full 4x4 field, N = M eta, matrix polynomials (independent of the core's eigen form):
      sx : tr[p(N)^2] / (2 D^2),  p(z) = (z + g)(z - 1)(z - delta)             (eq. 46)
      sel: tr[q(N)^2] / 2,        q(z) = (z + g)^2 (z - 1)^2 (z - delta) / D^2  (eq. 48),  D = (g + delta)(1 - delta)
      E_lift = eps * w * h^3 * sum_cells density, w the base weight of the core's Terms.
  lift share = E_lift / E_stat. kin recomputed by Terms.kin(M); omega* = J / (2 kin).
  Distinct count: greedy over states in ascending E_J, a state is a duplicate when its sorted-spectrum rms to an
      already-kept state is below the threshold (per-element rms, and the per-cell-norm variant, sqrt 3 larger).
  The run's rms is the per-cell-norm variant sqrt(mean_cells sum_3 (dlam)^2) (its spectrum_rms), so the cellnorm count
  is the one comparable to the run's; the per-element variant is reported beside it.
  Best cubic image: min over the 48 axis permutations/flips of the raw spatial tensor cell-Frobenius rms.
  Distinct read: the three lowest accepted states mutually distinct at 0.05 cellnorm rms (across J); Koide Q as above.
Reads data/m5_32_r31_s1.json once (a snapshot copy is not made; it is only read) and writes data/m5_32_r31_s1_audit.json.
"""
import importlib.util
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
rec = json.load(open(os.path.join(DATA, "m5_32_r31_s1.json")))
rows, points, center = rec["rows"], rec["points"], rec["center"]
out = {}


def lift(tag):
    return "closure" in tag or "_sx" in tag or any(("_sel" + str(d)) in tag for d in range(10))


def ok(v):
    f = v["final"]
    d = f["cond1_degree_top"]["12"]["degree"]
    return (d is not None and abs(abs(d) - 1) <= 0.1 and np.isfinite(f["E_J"]) and abs(f["E_stat"]) <= 1e3)


pids = ["c0"] + [p for p, q in points.items() if q["box"] == [24, 36.0] and q.get("kind") != "lift"]
sel = {p: {t: v for t, v in rows.items() if v["pid"] == p and v["n"] == 24 and not lift(t)} for p in pids}
if len(pids) != 15:
    print("WARNING pid count", len(pids))


def p(*a):
    print(*a)


# ---- claims 1, 2
p("== CLAIM 1/2: three lowest accepted per point ==")
c1 = {}
for pid in pids:
    acc = sorted([(v["final"]["E_J"], t, v) for t, v in sel[pid].items() if ok(v)], key=lambda x: x[0])
    nacc = len(acc)
    if nacc < 3:
        p(f"{pid:12s} accepted={nacc}  (fewer than 3)")
        c1[pid] = {"n_acc": nacc}
        continue
    E = np.array([a[0] for a in acc[:3]])
    Q = E.sum() / np.sqrt(E).sum() ** 2
    Js = [a[2]["J"] for a in acc[:3]]
    # alternate: acceptance loosened to |E_stat|<=1e3 only and degree ignored
    allf = sorted([v["final"]["E_J"] for v in sel[pid].values() if abs(v["final"]["E_stat"]) <= 1e3])[:3]
    alt = [allf[1] / allf[0], allf[2] / allf[0]] if len(allf) == 3 else None
    c1[pid] = dict(n_acc=nacc, E=E.tolist(), r2=E[1] / E[0], r3=E[2] / E[0], Q=Q, J_of_three=Js,
                   alt_no_degree_r2_r3=alt)
    p(f"{pid:12s} acc={nacc:2d} E={np.round(E, 4)} r2={E[1]/E[0]:.4f} r3={E[2]/E[0]:.4f} Q={Q:.5f} Js={[round(j) for j in Js]}"
      f" noDeg={None if alt is None else np.round(alt, 4)}")
g = [v for v in c1.values() if "r2" in v]
p("r2 range", min(v["r2"] for v in g), max(v["r2"] for v in g), "| r3 range", min(v["r3"] for v in g),
  max(v["r3"] for v in g), "| Q range", min(v["Q"] for v in g), max(v["Q"] for v in g),
  "| max r2", max(v["r2"] for v in g), "| any J>0 in three:", any(j > 0 for v in g for j in v["J_of_three"]))
out["claim1"] = c1

# ---- claim 3
p("== CLAIM 3: degree census ==")
cnt = Counter()
tot = 0
for pid in pids:
    for t, v in sel[pid].items():
        if v["J"] > 0:
            tot += 1
            dt = v["final"]["cond1_degree_top"]
            cnt[tuple(k for k in ("6", "9", "12") if dt[k]["degree"] is not None)] += 1
p("J>0 states", tot, dict(cnt))
j0 = Counter()
for pid in pids:
    for t, v in sel[pid].items():
        if v["J"] == 0 and ok(v):
            dt = v["final"]["cond1_degree_top"]
            j0[tuple((k, None if dt[k]["degree"] is None else round(dt[k]["degree"], 3)) for k in ("6", "9", "12"))] += 1
p("J=0 accepted degree patterns", dict(j0))
out["claim3"] = {"n_Jpos": tot, "census": {str(k): c for k, c in cnt.items()}, "J0_patterns": {str(k): c for k, c in j0.items()}}

# ---- claim 5
p("== CLAIM 5: negative c6 ==")
for pid in ("ax_c6_-1", "ax_c6_-0.3"):
    rr = sel[pid].values()
    a0 = [(v["seed"], round(v["final"]["E_J"], 3)) for v in rr if v["J"] == 0 and ok(v)]
    lad = [(v["seed"], v["J"] > 0 and round(v["final"]["E_stat"]), ) for v in rr if v["J"] > 0]
    big = sum(1 for v in rr if v["J"] > 0 and abs(v["final"]["E_stat"]) > 1e3)
    nl = sum(1 for v in rr if v["J"] > 0)
    nacc = sum(1 for v in rr if ok(v))
    p(pid, "accepted total", nacc, "accepted J=0", a0, "| ladder |E_stat|>1e3:", big, "/", nl)
    p("   J=0 all rows:", [(v["seed"], round(v["final"]["E_J"], 3), round(v["final"]["E_stat"], 3),
                           v["final"]["cond1_degree_top"]["12"]["degree"]) for v in rr if v["J"] == 0])
    p("   ladder E_stat:", [(v["seed"], round(v["J"]), round(v["final"]["E_stat"], 1)) for v in rr if v["J"] > 0])
    out["claim5_" + pid] = {"accepted": nacc, "acc_J0": a0, "ladder_big": big, "ladder_n": nl}

# ---- claim 7
p("== CLAIM 7: c0 ladder ==")
J1 = center["J_1"]
E0, E1_, E2_ = {}, {}, {}
for t, v in sel["c0"].items():
    s = v["seed"]
    if v["J"] == 0:
        E0[s] = v["final"]["E_J"]
    elif abs(v["J"] - J1) < 1:
        E1_[s] = v["final"]["E_J"]
    elif abs(v["J"] - 2 * J1) < 1:
        E2_[s] = v["final"]["E_J"]
for s in E0:
    p(f"{s:5s} E0={E0[s]:.3f} E(J1)={E1_.get(s, np.nan):.3f} E(2J1)={E2_.get(s, np.nan):.3f} "
      f"r1={E1_.get(s, np.nan)/E0[s]:.3f} r2={E2_.get(s, np.nan)/E0[s]:.3f}")
p("E(J1) range", min(E1_.values()), max(E1_.values()), "E(2J1) range", min(E2_.values()), max(E2_.values()))
out["claim7"] = {"E0": E0, "E1": E1_, "E2": E2_}

# ---- claim 8
p("== CLAIM 8: closure ==")
cl = {v["J"]: v for t, v in rows.items() if v["pid"] == "c0_closure" or ("closure" in t)}
hi = [v for v in cl.values() if abs(v["J"] - 1.05 * J1) < 1][0]
lo = [v for v in cl.values() if abs(v["J"] - 0.95 * J1) < 1][0]
slope = (hi["final"]["E_J"] - lo["final"]["E_J"]) / (hi["J"] - lo["J"])
radJ1 = [v for v in sel["c0"].values() if v["seed"] == "rad" and abs(v["J"] - J1) < 1][0]
omega_row = radJ1["final"]["omega"]
omega_kin = J1 / (2 * radJ1["final"]["kin"])
p(f"slope={slope:.5f} omega(row)={omega_row:.5f} J1/(2kin)={omega_kin:.5f} mismatch={(slope-omega_row)/omega_row*100:.2f}%"
  f"  (E hi {hi['final']['E_J']:.4f}, lo {lo['final']['E_J']:.4f}, J {hi['J']:.2f}, {lo['J']:.2f})")
out["claim8"] = dict(slope=slope, omega_row=omega_row, omega_kin=omega_kin)

# ---- claims 4, 6 from the npz
spec = importlib.util.spec_from_file_location("core", os.path.join(HERE, "m5_32_r31_s1_core.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
T = core.Terms(24, 36.0, 8.0, 0.3, 25.0)
free, r = T.free, T.r


def load(tag):
    return np.load(os.path.join(DATA, "m5_32_r31_s1", tag + ".npz"))["M"][..., 1:, 1:]


def sortspec(S):
    return np.sort(np.linalg.eigvalsh(S), axis=-1)


c0 = {v["seed"]: v["tag"] for v in sel["c0"].values() if v["J"] == 0}
S0 = {s: load(t) for s, t in c0.items()}
L0 = {s: sortspec(S) for s, S in S0.items()}
p("== CLAIM 4: c0 J=0 field comparisons (free cells:", int(free.sum()), ") ==")
c4 = {}
for a, b in [("obl", "perm"), ("obl", "bia"), ("rad", "node"), ("obl", "rad"), ("perm", "bia"), ("bia", "rad")]:
    dl = (L0[a] - L0[b])[free]
    ds = (S0[a] - S0[b])[free]
    v = dict(sorted_rms_elem=float(np.sqrt((dl**2).mean())), sorted_rms_cellnorm=float(np.sqrt((dl**2).sum(-1).mean())),
             raw_rms_elem=float(np.sqrt((ds**2).mean())), raw_rms_cellfro=float(np.sqrt((ds**2).sum((-1, -2)).mean())))
    c4[f"{a}-{b}"] = v
    p(f"{a:5s}-{b:5s} " + " ".join(f"{k}={x:.4f}" for k, x in v.items()))
out["claim4"] = c4


def jumps(S, L, factors=(0.3, 0.5, 1.0), rmax=14.0):
    n = S.shape[0]
    res = {f: [0, 0, 0] for f in factors}
    nl = 0
    inr = r < rmax
    for ax in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[ax] = slice(0, n - 1)
        sl1[ax] = slice(1, n)
        sl0, sl1 = tuple(sl0), tuple(sl1)
        m = inr[sl0] & inr[sl1]
        dS = np.sqrt(((S[sl0] - S[sl1]) ** 2).sum((-1, -2)))[m]
        dL = np.sqrt(((L[sl0] - L[sl1]) ** 2).sum(-1))[m]
        nl += m.sum()
        big = dS > 0.3
        for f in factors:
            res[f][0] += int((big & (dL < f * dS)).sum())
            res[f][1] += int(big.sum())
    return {f: dict(jump=a, big=b, links=int(nl), frac_links=a / nl, frac_big=a / max(b, 1)) for f, (a, b, _) in res.items()}


p("== CLAIM 6: frame jumps (fraction of links; fraction of big links) at factors 0.3 / 0.5 / 1.0 ==")
c6 = {}
for pid in pids:
    for s_, v in sorted(((v["seed"], v) for v in sel[pid].values() if v["J"] == 0 and ok(v))):
        S = load(v["tag"])
        res = jumps(S, sortspec(S))
        c6[f"{pid}/{s_}"] = res
        p(f"{pid:11s} {s_:5s} " + " | ".join(f"f{f}: {100*res[f]['frac_links']:.2f}% ({100*res[f]['frac_big']:.1f}% of big)"
                                              for f in (0.3, 0.5, 1.0)) + f"  links={res[0.5]['links']} big={res[0.5]['big']}")
fr = {pid: [c6[k][0.5]["frac_links"] for k in c6 if k.startswith(pid + "/")] for pid in pids}
for pid in pids:
    if fr[pid]:
        p(f"  range {pid:11s} {100*min(fr[pid]):.2f}% to {100*max(fr[pid]):.2f}%")
out["claim6"] = {k: {str(f): x for f, x in v.items()} for k, v in c6.items()}

# =================== LIFTS (R31-1b) ===================
p("\n\n================ LIFTS ================")
LP = [q_ for q_, v in points.items() if v.get("kind") == "lift"]
ALLP = ["c0"] + LP
lrows = {q_: {t: v for t, v in rows.items() if v["pid"] == q_ and v["n"] == 24 and "closure" not in t} for q_ in ALLP}
LO = {}
EIGH = None


def Tof(pid):
    pr = points[pid]["params"] if pid != "c0" else center["params"]
    return core.Terms(24, 36.0, pr["g"], pr["delta"], pr["w1s"], pr.get("c6", 0.0), pr.get("lam_s", 0.0),
                      pr.get("lam_sel", 0.0), pr.get("eps_sx", 0.0), pr.get("eps_sel", 0.0))


def loadM(tag):
    return np.load(os.path.join(DATA, "m5_32_r31_s1", tag + ".npz"))["M"]


ETA4 = np.diag([-1.0, 1.0, 1.0, 1.0])


def lift_energy(M, T, kind, eps):
    N = M @ ETA4
    I4 = np.eye(4)
    g, dl = T.g, T.delta
    D = (g + dl) * (1.0 - dl)
    a, b, c = N + g * I4, N - I4, N - dl * I4
    if kind == "sx":
        P = a @ b @ c
        dens = np.einsum("...ij,...ji->...", P, P) / (2 * D * D)
    else:
        P = a @ a @ b @ b @ c / (D * D)
        dens = np.einsum("...ij,...ji->...", P, P) / 2
    return eps * T.w * T.h3 * float(dens.sum())


def rms_pair(LA, LB, mask, cell=False):
    d = (LA - LB)[mask]
    return float(np.sqrt((d**2).sum(-1).mean())) if cell else float(np.sqrt((d**2).mean()))


def greedy_distinct(items, thr, cell=False):
    """items: list of (E, tag, spec-field); returns kept tags and the duplicate graph edges."""
    kept = []
    edges = []
    for E, t, L in sorted(items, key=lambda x: x[0]):
        dup = None
        for kt, kL in kept:
            if rms_pair(L, kL, free, cell) < thr:
                dup = kt
                break
        if dup is None:
            kept.append((t, L))
        else:
            edges.append((t, dup))
    return [k for k, _ in kept], edges


FS = {}   # sorted-spectrum fields cache
def spec_of(tag):
    if tag not in FS:
        FS[tag] = sortspec(load(tag))
    return FS[tag]


import itertools


def cubic_images(S):
    """the 48 cubic-group images of a spatial-tensor field S[n,n,n,3,3] on the centered even grid."""
    for P in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            G = np.transpose(S, P + (3, 4))
            G = G[..., list(P), :][..., list(P)]
            for a in range(3):
                if sg[a] < 0:
                    G = np.flip(G, axis=a)
            sv = np.array(sg, float)
            yield G * (sv[:, None] * sv[None, :])


def aligned_raw(SA, SB):
    best = np.inf
    for G in cubic_images(SA):
        best = min(best, float(np.sqrt(((G - SB)[free] ** 2).sum((-1, -2)).mean())))
    return best


for pid in ALLP:
    rr = lrows[pid]
    T = Tof(pid)
    res = {}
    J1 = center["J_1"]
    st0 = {v["seed"]: v for v in rr.values() if v["J"] == 0}
    st1 = {v["seed"]: v for v in rr.values() if abs(v["J"] - J1) < 1}
    st2 = {v["seed"]: v for v in rr.values() if abs(v["J"] - 2 * J1) < 1}
    E0s = {s_: v["final"]["E_J"] for s_, v in st0.items()}
    res["n_states"] = len(rr)
    res["labels"] = dict(Counter(v["label"] for v in rr.values()))
    res["status"] = dict(Counter(v["status"] for v in rr.values()))
    res["n_accepted"] = sum(1 for v in rr.values() if ok(v))
    res["n_at_gate"] = sum(1 for v in rr.values() if v["label"] == "AT_GATE")
    res["J0_E"] = E0s
    mn, mx = min(E0s.values()), max(E0s.values())
    res["J0_spread_pct"] = 100 * (mx - mn) / mn
    # 1. distinct counts
    items = [(v["final"]["E_J"], v["tag"], spec_of(v["tag"])) for v in st0.values()]
    res["distinct"] = {}
    for thr in (0.02, 0.05, 0.1):
        res["distinct"][str(thr)] = {"elem": len(greedy_distinct(items, thr)[0]),
                                      "cellnorm": len(greedy_distinct(items, thr, True)[0])}
    # pairwise raw/sorted rms between the J=0 seeds
    seeds = sorted(st0)
    pr_sorted, pr_raw, pr_full = [], [], []
    Mf = {s_: loadM(st0[s_]["tag"]) for s_ in seeds}
    for i_ in range(len(seeds)):
        for j_ in range(i_ + 1, len(seeds)):
            a_, b_ = seeds[i_], seeds[j_]
            pr_sorted.append(rms_pair(spec_of(st0[a_]["tag"]), spec_of(st0[b_]["tag"]), free))
            pr_raw.append(float(np.sqrt(((Mf[a_][..., 1:, 1:] - Mf[b_][..., 1:, 1:])[free] ** 2).mean())))
            pr_full.append(float(np.sqrt(((Mf[a_] - Mf[b_])[free] ** 2).mean())))
    res["J0_pair_rms_sorted_elem"] = [min(pr_sorted), max(pr_sorted)]
    res["J0_pair_rms_raw_spatial_elem"] = [min(pr_raw), max(pr_raw)]
    res["J0_pair_rms_raw_full4x4_elem"] = [min(pr_full), max(pr_full)]
    PM = {}
    for i_, a_ in enumerate(seeds):
        for b_ in seeds[i_ + 1:]:
            ds_ = (Mf[a_][..., 1:, 1:] - Mf[b_][..., 1:, 1:])[free]
            PM[f"{a_}-{b_}"] = dict(sorted_cellnorm=rms_pair(spec_of(st0[a_]["tag"]), spec_of(st0[b_]["tag"]), free, True),
                                    raw_spatial_cellfro=float(np.sqrt((ds_**2).sum((-1, -2)).mean())),
                                    dE_rel=abs(E0s[a_] - E0s[b_]) / E0s[a_])
    for i_, a_ in enumerate(seeds):
        for b_ in seeds[i_ + 1:]:
            PM[f"{a_}-{b_}"]["raw_spatial_cellfro_after_best_cubic_image"] = aligned_raw(Mf[a_][..., 1:, 1:], Mf[b_][..., 1:, 1:])
    res["J0_pair_matrix"] = PM
    res["J0_max_pair_aligned_raw_cellfro"] = max(v["raw_spatial_cellfro_after_best_cubic_image"] for v in PM.values())
    res["J0_min_pair_aligned_raw_cellfro"] = min(v["raw_spatial_cellfro_after_best_cubic_image"] for v in PM.values())
    acc1 = [v for v in st1.values() if ok(v)]
    pm1 = [rms_pair(spec_of(a_["tag"]), spec_of(b_["tag"]), free, True) for i_, a_ in enumerate(acc1) for b_ in acc1[i_ + 1:]]
    res["J1_accepted_pair_sorted_cellnorm"] = [min(pm1), max(pm1)] if pm1 else None
    res["J1_accepted_E_spread_pct"] = (100 * (max(v["final"]["E_J"] for v in acc1) / min(v["final"]["E_J"] for v in acc1) - 1)) if acc1 else None
    if acc1:
        lowJ0 = min(st0.values(), key=lambda v: v["final"]["E_J"])
        res["J1_vs_lowestJ0_sorted_cellnorm"] = [min(rms_pair(spec_of(a_["tag"]), spec_of(lowJ0["tag"]), free, True) for a_ in acc1),
                                                  max(rms_pair(spec_of(a_["tag"]), spec_of(lowJ0["tag"]), free, True) for a_ in acc1)]
    # 3. lift share from rows; recomputed from field for the lowest J=0 state
    sh = []
    for v in rr.values():
        f_ = v["final"]
        sh.append(((f_.get("E_lift_sx", 0.0) + f_.get("E_lift_sel", 0.0)) / f_["E_stat"], v["J"]))
    s0 = [x for x, jj in sh if jj == 0]
    res["lift_share_J0_pct"] = [100 * min(s0), 100 * max(s0)]
    sl = [x for x, jj in sh if jj > 0 and np.isfinite(x)]
    res["lift_share_ladder_pct"] = [100 * min(sl), 100 * max(sl)]
    lowseed = min(E0s, key=E0s.get)
    Mlow = Mf[lowseed]
    eps_sx, eps_sel = T.eps_sx, T.eps_sel
    row_f = st0[lowseed]["final"]
    rec_sx = lift_energy(Mlow, T, "sx", eps_sx) if eps_sx else 0.0
    rec_sel = lift_energy(Mlow, T, "sel", eps_sel) if eps_sel else 0.0
    res["lift_from_field_lowestJ0"] = dict(seed=lowseed, sx=rec_sx, sel=rec_sel, row_sx=row_f.get("E_lift_sx", 0.0),
                                           row_sel=row_f.get("E_lift_sel", 0.0), share_from_field_pct=100 * (rec_sx + rec_sel) / row_f["E_stat"])
    # 4. kin
    kin_rows = {s_: v["final"]["kin"] for s_, v in st0.items()}
    kin_rec = {s_: float(T.kin(Mf[s_], need_grad=False)[0]) for s_ in seeds}
    res["kin_J0_row"] = [min(kin_rows.values()), max(kin_rows.values())]
    res["kin_J0_recomputed"] = [min(kin_rec.values()), max(kin_rec.values())]
    res["kin_J0_max_rel_diff"] = max(abs(kin_rec[s_] - kin_rows[s_]) / kin_rows[s_] for s_ in seeds)
    om = [v["final"]["omega"] for v in st1.values()]
    kj1 = [v["final"]["kin"] for v in st1.values()]
    res["omega_J1_row"] = [min(om), max(om)]
    res["kin_J1_row"] = [min(kj1), max(kj1)]
    res["omega_check_max_rel"] = max(abs(v["final"]["omega"] - v["J"] / (2 * v["final"]["kin"])) / v["final"]["omega"]
                                     for v in st1.values())
    kj1_rec = [float(T.kin(loadM(v["tag"]), need_grad=False)[0]) for v in st1.values()]
    res["kin_J1_recomputed_max_rel_diff"] = max(abs(a_ - b_) / b_ for a_, b_ in zip(kj1_rec, kj1))
    res["omega_J1_from_recomputed_kin"] = [J1 / (2 * max(kj1_rec)), J1 / (2 * min(kj1_rec))]
    # 5. ladder degree census
    lad = [v for v in rr.values() if v["J"] > 0]
    res["ladder_n"] = len(lad)
    res["ladder_r12_defined"] = sum(1 for v in lad if v["final"]["cond1_degree_top"]["12"]["degree"] is not None)
    res["J0_r12_degrees"] = {s_: v["final"]["cond1_degree_top"]["12"]["degree"] for s_, v in st0.items()}
    # 6. ladder ratios
    E0min = min(E0s.values())
    res["ratio_J1"] = [min(v["final"]["E_J"] for v in st1.values()) / E0min, max(v["final"]["E_J"] for v in st1.values()) / E0min]
    res["ratio_2J1"] = [min(v["final"]["E_J"] for v in st2.values()) / E0min, max(v["final"]["E_J"] for v in st2.values()) / E0min]
    acc = [v for v in rr.values() if ok(v)]
    itemsA = [(v["final"]["E_J"], v["tag"], spec_of(v["tag"])) for v in acc]
    kept = greedy_distinct(itemsA, 0.05, True)[0]
    kept_elem = greedy_distinct(itemsA, 0.05)[0]
    kept_s = sorted(kept, key=lambda t: rows[t]["final"]["E_J"])
    triple = []
    if len(kept_s) >= 3:
        E3 = np.array([rows[t]["final"]["E_J"] for t in kept_s[:3]])
        res["distinct_triple"] = dict(tags_J=[rows[t]["J"] for t in kept_s[:3]], seeds=[rows[t]["seed"] for t in kept_s[:3]],
                                      E=E3.tolist(), r2=E3[1] / E3[0], r3=E3[2] / E3[0],
                                      Q=E3.sum() / np.sqrt(E3).sum() ** 2)
        named = set(kept_s[:3])
        near = []
        for t, v in rr.items():
            if t in named:
                continue
            for nt in kept_s[:3]:
                if abs(v["final"]["E_J"] / rows[nt]["final"]["E_J"] - 1) < 0.01:
                    near.append((v["seed"], v["J"] > 0 and round(v["J"] / J1, 2) or 0, ok(v), v["label"],
                                 round(v["final"]["E_J"], 4), "near", rows[nt]["seed"], round(rows[nt]["J"] / J1, 2)))
        res["within_1pct_of_named"] = near
        # J=0-only plan read
    ke = sorted(kept_elem, key=lambda t: rows[t]["final"]["E_J"])[:3]
    res["distinct_triple_elem_variant"] = [(rows[t]["seed"], round(rows[t]["J"] / J1, 2), rows[t]["final"]["E_J"]) for t in ke]
    acc0 = sorted(v["final"]["E_J"] for v in acc if v["J"] == 0)
    if len(acc0) >= 3:
        res["J0_only_accepted_ratios"] = [acc0[1] / acc0[0], acc0[2] / acc0[0]]
    # all lowest accepted (not distinct), top 6
    res["lowest_accepted_6"] = [(rows[t]["seed"], round(rows[t]["J"] / J1, 2), round(rows[t]["final"]["E_J"], 4))
                                for t in sorted([v["tag"] for v in acc], key=lambda t: rows[t]["final"]["E_J"])[:6]]
    # 7. frame jumps
    def jr(sub, f_):
        vals = []
        for v in sub:
            S = load(v["tag"])
            vals.append(jumps(S, spec_of(v["tag"]), factors=(0.3, 0.5, 1.0))[f_]["frac_links"])
        return [100 * min(vals), 100 * max(vals)] if vals else None
    J0all = list(st0.values())
    J0acc = [v for v in J0all if ok(v)]
    ladall = lad
    ladacc = [v for v in lad if ok(v)]
    res["jumps_pct"] = {f"f{f_}": {"J0_all": jr(J0all, f_), "J0_acc": jr(J0acc, f_), "lad_all": jr(ladall, f_),
                                    "lad_acc": jr(ladacc, f_)} for f_ in (0.3, 0.5, 1.0)}
    # observations
    res["max_abs_E_stat"] = max(abs(v["final"]["E_stat"]) for v in rr.values())
    res["near_runaway_bar"] = [(v["seed"], round(v["J"] / J1, 2), v["final"]["E_stat"]) for v in rr.values()
                               if 100 < abs(v["final"]["E_stat"]) < 1e4]
    res["not_accepted"] = [(v["seed"], round(v["J"] / J1, 2), v["label"],
                            v["final"]["cond1_degree_top"]["12"]["degree"] if v["final"]["cond1_degree_top"]["12"]["degree"] is None else round(v["final"]["cond1_degree_top"]["12"]["degree"], 3),
                            round(v["final"]["E_stat"], 2)) for v in rr.values() if not ok(v)]
    res["iters_J0"] = {s_: v["iters"] for s_, v in st0.items()}
    LO[pid] = res
    p(f"\n--- {pid} ---")
    for k_, v_ in res.items():
        p(f"  {k_}: {v_}")
out["lifts"] = LO

json.dump(out, open(os.path.join(DATA, "m5_32_r31_s1_audit.json"), "w"), indent=1, default=float)
