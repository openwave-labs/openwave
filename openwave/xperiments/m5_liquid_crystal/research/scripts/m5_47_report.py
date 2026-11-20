"""M5.47: the tables and the plots, from data/m5_47_reads.json and the summaries of the jobs.

Called by m5_47_reads.py (--tables, --plots). The numbers are those of the reads; nothing is fitted here except
the lines on the shells of r, which carry no label.
"""

import json
from pathlib import Path

import numpy as np

import m5_42_eq42 as Q
import m5_47_evolve as E
import m5_47_waves as WV

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#d9d8d3"
GRID_NAME = dict(a="half-box 8, spacing 0.5", b="half-box 8, spacing 0.25", c="half-box 10, spacing 0.5", d="half-box 10, spacing 0.25")
DIRS = ("massless 1", "massless 2", "massless 3", "light", "mid", "heavy")


def _arr(x):
    return np.array([[np.nan if v is None else v for v in row] for row in x], float)


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        return f"{v:.2e}"
    return f"{v:.{nd}f}"


def pf(ok):
    return "n/a" if ok is None else ("PASS" if ok else "FAIL")


def load(a):
    res = json.loads((a.data / "m5_47_reads.json").read_text())
    ctl = json.loads((a.data / "m5_47_controls_stepper.json").read_text())
    jobs = {j: json.loads((a.data / f"m5_47_{j}.json").read_text()) for j in WV.JOBS}
    return res, ctl, jobs


def shell_fits(info):
    """A line per shell of r (width 0.5), with four fit points at least. No label."""
    w2 = np.array(info["targets"]) ** 2
    K2 = np.array([[np.nan if v is None else v for v in row] for row in info["shells"]["K2"]])
    fit = np.array(info["fit_shells"], bool)
    rows = []
    for s in range(K2.shape[0]):
        if fit[s].sum() >= 4:
            c2, m2 = np.polyfit(K2[s, fit[s]], w2[fit[s]], 1)
            dev = np.abs(c2 * K2[s, fit[s]] + m2 - w2[fit[s]]) / w2[fit[s]]
            rows.append(dict(r_lo=0.5 * s, r_hi=0.5 * (s + 1), nodes=info["shells"]["nodes"][s], points=int(fit[s].sum()), c2=float(c2), m2=float(m2), max_rel=float(dev.max())))
    return rows


def tables(a):
    res, ctl, jobs = load(a)
    L = []
    add = L.append
    add("### Controls\n")
    add("| Control | What was read | Result | |")
    add("| --- | --- | --- | --- |")
    k0, k1, k2, k3, k4, k5 = (ctl[k] for k in ("K0", "K1", "K2", "K3", "K4", "K5"))
    add(f"| K0 (added) | U and force of the fused stage against `m5_42_eq42` (seed, B0a) | relative gaps {fmt(max(x['rel_U'] for x in k0))} and {fmt(max(x['rel_force'] for x in k0))} | {pf(ctl['passed']['K0'])} |")
    add(f"| K1 | T against ½ I Ω² (seed, B0a) | relative gaps {fmt(k1[0]['rel'])}, {fmt(k1[1]['rel'])} | {pf(ctl['passed']['K1'])} |")
    add(f"| K2 | ∂T/∂Ṁ and ∂T/∂M against central differences, 200 entries | {fmt(k2['rel_dT_dR'])} and {fmt(k2['rel_dT_dM'])} of the largest entry | {pf(ctl['passed']['K2'])} |")
    add(f"| K3 | 𝒜 at every interior node (seed, B0a) | least eigenvalue {fmt(min(x['least_eigenvalue'] for x in k3), 4)} (2 L₁ = 0.06), asymmetry {fmt(max(x['asymmetry'] for x in k3))}, solve residual {fmt(max(x['solve_residual'] for x in k3))} | {pf(ctl['passed']['K3'])} |")
    add(f"| K4 | E1: drift of E over the starting T | {fmt(k4['drift_over_T0_at_0015'])} at step 0.015, {fmt(k4['drift_over_T0_at_003'])} at 0.03, ratio {fmt(k4['ratio'], 1)} | {pf(ctl['passed']['K4'])} |")
    add(f"| K5 | E1 run back: return error over the displacement at the turn | {fmt(k5['return_over_turn_at_0015'])} at step 0.015; {fmt(k5['gain'], 1)} times smaller than at 0.03 | {pf(ctl['passed']['K5'])} |")
    k6 = res["K6"]
    add(f"| K6 | twenty sine modes, six directions: worst gap of ω² from k̂² + m² | V1 {fmt(100 * k6['V1']['worst_rel'])} %, V2 {fmt(100 * k6['V2']['worst_rel'])} %, V3 (v = 2) {fmt(100 * k6['V3']['worst_rel'])} % | {pf(all(k6[j]['passed'] for j in k6))} |")
    k7 = res["K7"]
    add(f"| K7 | the estimator in the vacuum, targets 8 to 11: worst gap of ω² − K² from {fmt(k7['V1']['mid_mass'], 2)} | V1 {fmt(k7['V1']['worst'])} over {k7['V1']['fit_points']} fit points, V2 {fmt(k7['V2']['worst'])} over {k7['V2']['fit_points']} | {pf(k7['V1']['passed'] and k7['V2']['passed'])} |")
    g = res["grids"]
    add("| K8 | twin power over difference power, worst fit point (six components; in W alone) | " + "; ".join(f"{k}: {fmt(g[k]['K8']['worst'])} ({fmt(g[k]['K8_in_W']['worst'])})" for k in g) + f" | {pf(all(g[k]['K8']['passed'] for k in g))} |")
    k9 = res["K9"]
    add("| K9 | ‖a(ε) − 2 a(ε/2)‖ over ‖a(ε)‖, worst fit point | " + "; ".join(f"{k}: {fmt(v['worst'])}" for k, v in k9.items()) + f" | {pf(all(v['passed'] for v in k9.values()))} |")
    add(f"| K10 | K² at half the step (Da against Ua), worst fit point | {fmt(res['K10']['worst'])} over {res['K10']['points']} fit points | {pf(res['K10']['passed'])} |")
    k11 = res["K11"]
    add("| K11 | largest \\|E(t) − E(0)\\| over the perturbation's energy | " + "; ".join(f"{k}: {fmt(100 * v['ratio'])} %" for k, v in k11.items()) + f" | {pf(all(v['passed'] for v in k11.values()))} |")
    add("| K12 | linearized equations on the Fourier sums: worst gap of the quotient from ω² | " + "; ".join(f"{k}: {fmt(100 * g[k]['K12']['worst'])} %" for k in g) + f" | {pf(all(g[k]['K12']['passed'] for k in g))} |")
    add("")
    add("### The vacuum branches (K6)\n")
    add("| Job | Direction | m² of the coded Hessian | m² read (median of ω² − k̂² over the modes) | Worst gap of ω² |")
    add("| --- | --- | --- | --- | --- |")
    for j in ("V1", "V2", "V3"):
        for l in range(6):
            add(f"| {j} | {DIRS[l]} | {fmt(k6[j]['masses'][l], 4)} | {fmt(k6[j]['read_masses'][l], 4)} | {fmt(100 * k6[j]['worst_rel_per_direction'][l])} % |")
    add("")
    add("### The estimator in the vacuum (K7)\n")
    add("| Job | Target ω | K² | Fit point | ω² − K² | Power outside W |")
    add("| --- | --- | --- | --- | --- | --- |")
    for j in ("V1", "V2"):
        for r in k7[j]["rows"]:
            if r["target"] >= 7.5:
                add(f"| {j} | {r['target']:g} | {fmt(r['K2'], 2)} | {fmt(r['fit'])} | {fmt(r['w2_minus_K2'], 2)} | {fmt(100 * r['outside_W'], 1)} % |")
    add("")
    add("### The grids of the unit\n")
    add("| Grid | Base | N | Spacing | Loop radius | Nodes read | q, least to largest | Median of (q − 1) r² beyond r = 5 | Fit points |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for k, info in g.items():
        add(f"| {k} | {info['base']} | {info['N']} | {info['spacing']:g} | {fmt(info['loop']['radius'])} | {info['region_nodes']} | {fmt(info['q_min'], 2)} to {fmt(info['q_max'], 2)} | {fmt(info['q_tail'], 1)} | {int(np.sum(info['fit']))} |")
    add("")
    for k, info in g.items():
        add(f"### Bins of q, grid {k} ({GRID_NAME[k]}, job {info['job']})\n")
        add("| q | Nodes | Median r | Fit points | Line: c², m² | m² q̄ | Plane: c_r², c_t², m² | Formula: c_r², c_t², m² (median) | Largest gap from the fit |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for f in info["fits"]:
            if not f["read"]:
                continue
            fo = f["formula"]
            line = f"{fmt(f['c2'], 2)}, {fmt(f['m2'], 1)}" if f["fitted"] else "no fit"
            plane = f"{fmt(f['plane']['cr2'], 2)}, {fmt(f['plane']['ct2'], 2)}, {fmt(f['plane']['m2'], 1)}" if f.get("plane") else "n/a"
            dev = f"{fmt(100 * f['max_rel'], 1)} % ({f['judged_by']})" if f["fitted"] else "n/a"
            add(f"| {f['q_lo']:g} to {f['q_hi']:g} | {f['nodes']} | {fmt(f['r_median'], 1)} | {f['points']} | {line} | {fmt(f.get('m2_q'), 1)} | {plane} | {fmt(fo['cr2'], 2)}, {fmt(fo['ct2'], 2)}, {fmt(fo['m2_median'], 1)} | {dev} |")
        add("")
    add("### The readings\n")
    add("| Grid | R1: median m², largest gap, points outside | R2: bins fitted, largest gap | R3: quotient over ω² − 1 (least, median, largest), points outside |")
    add("| --- | --- | --- | --- |")
    for k, info in g.items():
        r1, r2, r3 = info["R1"], info["R2"], info["R3"]
        add(f"| {k} | {fmt(r1.get('m2_median'), 1)}, {fmt(100 * r1['max_rel'], 1) if r1.get('max_rel') is not None else 'n/a'} %, {r1.get('outside', 'n/a')} of {r1['points']} | {r2['fitted_bins']}, {fmt(100 * r2['max_rel'], 1) if r2.get('max_rel') is not None else 'n/a'} % | {fmt(100 * r3['min'], 1) if r3.get('min') is not None else 'n/a'} %, {fmt(100 * r3['median'], 1) if r3.get('median') is not None else 'n/a'} %, {fmt(100 * r3['max'], 1) if r3.get('max') is not None else 'n/a'} %; {r3.get('outside', 'n/a')} of {r3['points']} |")
    add("")
    add("| Grid or pair | R1 | R2 | R3 | R4 |")
    add("| --- | --- | --- | --- | --- |")
    lab = res["labels"]
    for k in g:
        note = "" if lab[k]["labeled"] else " (spacing 0.5: tabled, not labeled)"
        add(f"| {k}{note} | {lab[k]['R1']} | {lab[k]['R2']} | {lab[k]['R3']} | |")
    for k, r in res["R4"].items():
        add(f"| {k[0]} with {k[1]} ({r['what']}) | | | | {lab[k]['R4']} |")
    add("")
    add("### The pairs of R4\n")
    add("| Pair | Bin of q | Shared targets | c² on the two grids | m² on the two grids | Gap of c² | Gap of m² |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for k, r in res["R4"].items():
        for row in r["rows"]:
            b = row["bin"]
            add(f"| {k[0]} with {k[1]} | {res['edges'][b]:g} to {res['edges'][b + 1]:g} | {row['shared']} | {fmt(row['c2'][0], 3)}, {fmt(row['c2'][1], 3)} | {fmt(row['m2'][0], 1)}, {fmt(row['m2'][1], 1)} | {fmt(row['dc2'])} | {fmt(100 * row['dm2_rel'], 1)} % |")
        if not r["rows"]:
            add(f"| {k[0]} with {k[1]} | none compared | | | | | |")
    add("")
    add("### The scatter of the read (second draws)\n")
    add("| Spacing | Common fit points | Median of \\|ΔK²\\| / ω² (limit 0.025) | Median of \\|Δ quotient\\| / ω² (limit 0.05) | Bins fitted in both | Median \\|Δc²\\| (limit 0.025) | Median relative Δm² (limit 0.05) |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for k, sp in (("a", 0.5), ("b", 0.25)):
        s = res["scatter"][k]
        add(f"| {sp:g} | {s['common_points']} | {fmt(s.get('median_dK2_over_w2'), 4)} | {fmt(s.get('median_dw2act_over_w2'), 4)} | {s['bins_fitted_in_both']} | {fmt(s.get('median_dc2'), 4)} | {fmt(s.get('median_dm2_rel'), 4)} |")
    add("")
    add("### The fit points by polarization\n")
    add("| Grid | q | Fit points with the power outside W over 90 %: targets, K² | Fit points with it under 50 %: targets, K² | Fit points between | Largest heavy share at a fit point |")
    add("| --- | --- | --- | --- | --- | --- |")
    for k, info in g.items():
        fit, tg = np.array(info["fit"], bool), np.array(info["targets"])
        ow, K2, hs = _arr(info["bins"]["outside_W"]), _arr(info["bins"]["K2"]), _arr(info["bins"]["heavy_share"])
        for f in info["fits"]:
            b = f["bin"]
            if not f["read"] or fit[b].sum() == 0:
                continue
            soft, inw = fit[b] & (ow[b] > 0.9), fit[b] & (ow[b] < 0.5)

            def span(m):
                return "none" if m.sum() == 0 else f"{int(m.sum())}: ω {tg[m].min():g} to {tg[m].max():g}, K² {K2[b, m].min():.1f} to {K2[b, m].max():.1f}"

            add(f"| {k} | {f['q_lo']:g} to {f['q_hi']:g} | {span(soft)} | {span(inw)} | {int((fit[b] & ~soft & ~inw).sum())} | {fmt(100 * float(hs[b, fit[b]].max()), 1)} % |")
    add("")
    def pc(v):
        return "n/a" if v is None else f"{100 * v:.1f} %"

    mvac = res["K6"]["V1"]["masses"][4]
    for key, tag in (("added", "in W"), ("added_cut", "above the cutoff")):
        ad = res[key]
        add(f"### Added ({tag}): summary\n")
        add("| Grid | Fit points: all, kept | Power outside W on the kept points, least to largest | Median m² of ω² − K², largest gap, points outside 5 % | Bins fitted, largest gap from the fit | Quotient over ω² − 1, central differences (least, median, largest; outside 10 %) | The same on the action's one-sided differences |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for k, x in ad["grids"].items():
            o, r1, r2, r3, r3o, ow = x["outside_W_at_fit_points"], x["R1_like"], x["R2_like"], x["R3_like"], x["R3_like_one_sided"], x["outside_W_on_selection"]
            add(f"| {k} | {o['all']}, {x['points']} | {pc(ow['least'])} to {pc(ow['largest'])} | {fmt(r1.get('m2_median'), 1)}, {pc(r1.get('max_rel'))}, {r1.get('outside', 'n/a')} of {r1['points']} | {r2['fitted_bins']}, {pc(r2.get('max_rel'))} | {pc(r3.get('min'))}, {pc(r3.get('median'))}, {pc(r3.get('max'))}; {r3.get('outside', 'n/a')} of {r3['points']} | {pc(r3o.get('min'))}, {pc(r3o.get('median'))}, {pc(r3o.get('max'))}; {r3o.get('outside', 'n/a')} of {r3o['points']} |")
        add("")
        for k, x in ad["grids"].items():
            add(f"### Added ({tag}): bins, grid {k}\n")
            add("| q | Points kept | Targets | Line: c², m² | Plane: c_r², c_t², m² (condition number) | Formula: c_r², c_t², m² (median) | m² over the formula | m² q̄ | Largest gap from the fit |")
            add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for f in x["fits"]:
                if not f["read"]:
                    continue
                fo = f["formula"]
                if f["fitted"]:
                    plane = f"{fmt(f['plane']['cr2'], 2)}, {fmt(f['plane']['ct2'], 2)}, {fmt(f['plane']['m2'], 1)} ({f['plane']['cond']:.0f})" if f.get("plane") else "n/a"
                    add(f"| {f['q_lo']:g} to {f['q_hi']:g} | {f['points']} | {f['targets'][0]:g} to {f['targets'][-1]:g} | {fmt(f['c2'], 2)}, {fmt(f['m2'], 1)} | {plane} | {fmt(fo['cr2'], 2)}, {fmt(fo['ct2'], 2)}, {fmt(fo['m2_median'], 1)} | {fmt(f['m2'] / fo['m2_median'], 2)} | {fmt(f['m2_q'], 1)} | {fmt(100 * f['max_rel'], 1)} % ({f['judged_by']}) |")
                else:
                    add(f"| {f['q_lo']:g} to {f['q_hi']:g} | {f['points']} | | no fit | | {fmt(fo['cr2'], 2)}, {fmt(fo['ct2'], 2)}, {fmt(fo['m2_median'], 1)} | | | |")
            add("")
        add(f"### Added ({tag}): the cutoff of each bin against the vacuum mass over the kinetic factor\n")
        add("| Grid | q | Mean q of the bin | √(m²_vac / q̄) | Cutoff of the bin: √m² of its line | Ratio | Lowest target kept |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for k, x in ad["grids"].items():
            fit, tg = np.array(x["fit_sel"], bool), np.array(g[k]["targets"])
            for f in x["fits"]:
                if f["fitted"] and f["m2"] > 0:
                    me = float(np.sqrt(mvac / f["q_mean"]))
                    add(f"| {k} | {f['q_lo']:g} to {f['q_hi']:g} | {fmt(f['q_mean'], 2)} | {fmt(me, 2)} | {fmt(float(np.sqrt(f['m2'])), 2)} | {fmt(float(np.sqrt(f['m2'])) / me, 3)} | {tg[fit[f['bin']]].min():g} |")
        add("")
        add(f"### Added ({tag}): the pairs\n")
        add("| Pair | Bin of q | Shared targets | c² on the two grids | m² on the two grids | Gap of c² | Gap of m² |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for k, r in ad["pairs"].items():
            for row in r["rows"]:
                b = row["bin"]
                add(f"| {k[0]} with {k[1]} | {res['edges'][b]:g} to {res['edges'][b + 1]:g} | {row['shared']} | {fmt(row['c2'][0], 3)}, {fmt(row['c2'][1], 3)} | {fmt(row['m2'][0], 1)}, {fmt(row['m2'][1], 1)} | {fmt(row['dc2'])} | {fmt(100 * row['dm2_rel'], 1)} % |")
            if not r["rows"]:
                add(f"| {k[0]} with {k[1]} | none compared | | | | | |")
        add("")
        add(f"### Added ({tag}): the second draws\n")
        add("| Spacing | Common points | Median of \\|ΔK²\\| / ω² | Median of \\|Δ quotient\\| / ω² | Bins fitted in both | Median \\|Δc²\\| | Median relative Δm² |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for k, sp in (("a", 0.5), ("b", 0.25)):
            sc = ad["scatter"][k]
            add(f"| {sp:g} | {sc['common_points']} | {fmt(sc.get('median_dK2_over_w2'), 4)} | {fmt(sc.get('median_dw2act_over_w2'), 4)} | {sc['bins_fitted_in_both']} | {fmt(sc.get('median_dc2'), 4)} | {fmt(sc.get('median_dm2_rel'), 4)} |")
        add("")
    add("### Shells of r (no label)\n")
    add("| Grid | r | Nodes | Fit points | c² | m² | Largest gap from the line |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for k in ("b", "d"):
        for s in shell_fits(g[k]):
            add(f"| {k} | {s['r_lo']:g} to {s['r_hi']:g} | {s['nodes']} | {s['points']} | {fmt(s['c2'], 2)} | {fmt(s['m2'], 1)} | {fmt(100 * s['max_rel'], 1)} % |")
    add("")
    add("### The jobs\n")
    add("| Job | N | Step | Steps | Perturbation energy | Largest \\|E(t) − E(0)\\| | Minutes |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for j, s in jobs.items():
        add(f"| {j} | {s['spec']['N']} | {s['spec']['dt']:g} | {s['steps']} | {fmt(s['perturbation_energy'])} | {fmt(s['drift'])} | {fmt(s['seconds'] / 60, 1)} |")
    add("")
    out = a.work / "tables.md"
    out.write_text("\n".join(L) + "\n")
    print("written", out)


# ----------------------------------------------------------------------------- plots

def _style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plots(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    res, ctl, jobs = load(a)
    out = a.plots_dir
    seq = LinearSegmentedColormap.from_list("seq", ["#b9d3f2", BLUE, "#0d2f5e"])
    g = res["grids"]
    nb = len(res["edges"]) - 1

    # 1. the vacuum branches
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    marks = dict(V1=("o", BLUE, "V1: spacing 0.5"), V2=("s", ORANGE, "V2: spacing 0.25"), V3=("^", AQUA, "V3: spacing 0.5, v = 2"))
    for ax, dirs, title in zip(axs[:3], ((0, 1, 2, 3), (4,), (5,)), ("Massless and light directions", "Mid direction", "Heavy direction")):
        for j, (mk, col, lab) in marks.items():
            if j == "V3" and dirs != (0, 1, 2, 3):
                continue
            k6 = res["K6"][j]
            for l in dirs:
                ax.plot([m["khat2"] for m in k6["modes"]], [m["w2"][l] for m in k6["modes"]], mk, ms=4.5, mfc="none" if l < 3 else col, mec=col, mew=1.0,
                        label=lab + (" (hollow: massless, filled: light)" if dirs == (0, 1, 2, 3) and l == 0 else "") if l == dirs[0] else None)
        kk = np.linspace(0, max(m["khat2"] for m in res["K6"]["V2"]["modes"]) * 1.05, 50)
        for l in dirs:
            ax.plot(kk, kk + res["K6"]["V1"]["masses"][l], color=MUTED, lw=1.0, label="ω² = k̂² + m² of the coded Hessian" if l == dirs[0] else None)
        if dirs == (0, 1, 2, 3):
            ax.plot(kk, kk + res["K6"]["V3"]["masses"][3], color=MUTED, lw=1.0, ls="--", label="the same for the light direction at v = 2")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("k̂² of the sine mode", fontsize=9, color=MUTED)
        ax.set_ylabel("ω² read", fontsize=9, color=MUTED)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left" if dirs != (0, 1, 2, 3) else "lower right")
        _style(ax)
    ax = axs[3]
    for i, (j, (mk, col, lab)) in enumerate(marks.items()):
        k6 = res["K6"][j]
        ax.plot(np.arange(3) + 0.12 * (i - 1), k6["read_masses"][3:], mk, ms=7, mfc=col, mec=col, label=lab + ": read")
        ax.plot(np.arange(3) + 0.12 * (i - 1), k6["masses"][3:], "_", ms=16, mew=1.4, color=INK, label="coded Hessian" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(range(3), ["light", "mid", "heavy"])
    ax.set_title("The three masses m², read against coded", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.suptitle("M5.47 vacuum jobs: twenty sine modes on the six directions of the vacuum Hessian", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "m5_47_vacuum_branches.png", dpi=140)
    plt.close(fig)

    # 2. the dispersion on the unit, per bin of q
    fig, axs = plt.subplots(2, 2, figsize=(16.5, 9.5), sharex=False)
    mid = res["K7"]["V1"]["mid_mass"]
    for ax, k in zip(axs.ravel(), ("b", "d", "a", "c")):
        info = g[k]
        K2, fit, w2 = _arr(info["bins"]["K2"]), np.array(info["fit"], bool), np.array(info["targets"]) ** 2
        inw = np.array(res["added_cut"]["grids"][k]["fit_sel"], bool)
        for f, fw in zip(info["fits"], res["added_cut"]["grids"][k]["fits"]):
            b = f["bin"]
            if not f["read"] or fit[b].sum() == 0:
                continue
            col = seq((f["q_mean"] - 1.25) / (7 - 1.25))
            out_w = fit[b] & ~inw[b]
            ax.plot(K2[b, out_w], w2[out_w], "o", ms=4, mfc="none", mec=col, mew=0.9)
            ax.plot(K2[b, inw[b]], w2[inw[b]], "o", ms=4, color=col, label=f"q {f['q_lo']:g} to {f['q_hi']:g}" + (f": c² {fw['c2']:.2f}, m² {fw['m2']:.1f}" if fw["fitted"] else ": no fit"))
            if fw["fitted"]:
                xx = np.array([K2[b, inw[b]].min(), K2[b, inw[b]].max()])
                ax.plot(xx, fw["c2"] * xx + fw["m2"], color=col, lw=1.2)
        xx = np.array([0, 4 / info["spacing"] ** 2])
        ax.plot(xx, xx + mid, "--", color=ORANGE, lw=1.4, label=f"vacuum law: ω² = K² + {mid:.2f}")
        ax.set_xlim(0, min(4 / info["spacing"] ** 2, 64) * 1.02)
        ax.set_ylim(0, 75)
        ax.set_title(f"Grid {k}: {GRID_NAME[k]} (job {info['job']})", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("K² of the bin (local wavenumber)", fontsize=9, color=MUTED)
        ax.set_ylabel("ω² of the target", fontsize=9, color=MUTED)
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        _style(ax)
    fig.suptitle("M5.47 relaxed unit: ω² against the local K², per bin of the kinetic factor q. Every fit point is drawn; filled: above the bin's target of least K²;\nhollow: at or under it. The lines are the added fits on the filled points (no label)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "m5_47_unit_dispersion.png", dpi=140)
    plt.close(fig)

    # 3. the fitted law against q, with the formula of the linearized action
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    gm = dict(a=("o", BLUE, False), b=("o", BLUE, True), c=("s", ORANGE, False), d=("s", ORANGE, True))
    for k, (mk, col, filled) in gm.items():
        fb = [f for f in res["added_cut"]["grids"][k]["fits"] if f["fitted"]]
        fp = [f for f in fb if f.get("plane")]
        kw = dict(ms=6, mfc=col if filled else "none", mec=col, mew=1.2, ls="none")
        axs[0].plot([f["q_mean"] for f in fb], [f["m2"] for f in fb], mk, label=f"grid {k}: {GRID_NAME[k]}", **kw)
        axs[1].plot([f["q_mean"] for f in fb], [f["c2"] for f in fb], mk, label=f"grid {k}", **kw)
        axs[2].plot([f["q_mean"] for f in fp], [f["plane"]["cr2"] for f in fp], mk, label=f"grid {k}: c_r²", **kw)
        axs[2].plot([f["q_mean"] for f in fp], [f["plane"]["ct2"] for f in fp], mk, alpha=0.45, label=f"grid {k}: c_t² (faded)", **kw)
    fr = [f for f in g["d"]["fits"] if f["read"]]
    qq = np.array([f["q_mean"] for f in fr])
    axs[0].plot(qq, [f["formula"]["m2_median"] for f in fr], color=MUTED, lw=1.2, label="formula v V″ / (2 L₁ q), bin median (grid d)")
    axs[0].plot(qq, mid / qq, ":", color=MUTED, lw=1.2, label=f"{mid:.2f} / q")
    axs[2].plot(qq, [f["formula"]["cr2"] for f in fr], color=MUTED, lw=1.2, label="formula, along r̂ (grid d)")
    axs[2].plot(qq, [f["formula"]["ct2"] for f in fr], "--", color=MUTED, lw=1.2, label="formula, across r̂ (grid d)")
    axs[1].axhline(1.0, color=MUTED, lw=1.0, ls=":")
    for ax, t, yl in zip(axs, ("m² of the line, per bin", "c² of the line, per bin", "The two speeds of the plane, per bin"), ("m²", "c²", "c_r², c_t²")):
        ax.set_title(t, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("mean kinetic factor q of the bin", fontsize=9, color=MUTED)
        ax.set_ylabel(yl, fontsize=9, color=MUTED)
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
    fig.suptitle("M5.47 relaxed unit: the law fitted in each bin above its cutoff (added, no label), against the kinetic factor (filled: spacing 0.25; hollow: 0.5)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "m5_47_local_law.png", dpi=140)
    plt.close(fig)

    # 4. the quotient of the linearized action and the control K12
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4), sharey=True)
    for ax, k in zip(axs, ("b", "d")):
        info = g[k]
        fit, tg = np.array(info["fit"], bool), np.array(info["targets"])
        inw = np.array(res["added_cut"]["grids"][k]["fit_sel"], bool)
        wa, wo, k12 = _arr(info["bins"]["w2_act"]), _arr(info["bins"]["w2_act_os"]), _arr(info["bins"]["k12"])
        first = True
        for f in info["fits"]:
            b = f["bin"]
            if not f["read"] or fit[b].sum() == 0:
                continue
            col = seq((f["q_mean"] - 1.25) / (7 - 1.25))
            ax.plot(tg[inw[b]], wa[b, inw[b]] / tg[inw[b]] ** 2, "o", ms=4, mfc="none", mec=col, mew=0.9, label="one polarization, central differences (hollow; darker: larger q)" if first else None)
            ax.plot(tg[inw[b]], wo[b, inw[b]] / tg[inw[b]] ** 2, "o", ms=4, color=col, label="one polarization, the action's one-sided differences (filled)" if first else None)
            ax.plot(tg[fit[b]], k12[b, fit[b]] / tg[fit[b]] ** 2, "x", ms=4, color=ORANGE, label="K12: six components, every fit point" if first else None)
            first = False
        ax.set_ylim(0.5, 1.5)
        for y, ls in ((1.0, "-"), (0.9, ":"), (1.1, ":")):
            ax.axhline(y, color=MUTED, lw=0.9, ls=ls)
        ax.set_title(f"Grid {k}: {GRID_NAME[k]}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("target ω", fontsize=9, color=MUTED)
        ax.legend(fontsize=7.5, frameon=False)
        _style(ax)
    axs[0].set_ylabel("quotient / ω²", fontsize=9, color=MUTED)
    fig.suptitle("M5.47 relaxed unit: what the linearized action gives on the measured waves (circles: fit points above the cutoff; dotted: the ±10 % of R3)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "m5_47_action_quotient.png", dpi=140)
    plt.close(fig)

    # 5. the energies
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    order = [j for j in jobs if jobs[j]["perturbation_energy"] > 0]
    for ax, sel, title in ((axs[0], [j for j in order if j.startswith("V")], f"Vacuum jobs (step {jobs['V1']['spec']['dt']:g}; V3 {jobs['V3']['spec']['dt']:g})"),
                           (axs[1], [j for j in order if not j.startswith("V")], f"Unit jobs (step {jobs['Ua']['spec']['dt']:g}; Da {jobs['Da']['spec']['dt']:g})")):
        for i, j in enumerate(sel):
            en = jobs[j]["energy"]
            ax.plot(en["t"], 100 * (np.array(en["E"]) - en["E"][0]) / jobs[j]["perturbation_energy"], lw=1.3,
                    color=(BLUE, ORANGE, AQUA, MUTED)[i % 4], ls=("-", "--", ":")[i // 4 % 3], label=j)
        t1 = a.work / "try1_step003" / "m5_47_Ua.json"
        if ax is axs[1] and t1.exists():
            s = json.loads(t1.read_text())
            ax.plot(s["energy"]["t"], 100 * (np.array(s["energy"]["E"]) - s["energy"]["E"][0]) / s["perturbation_energy"], color=INK, lw=1.0, ls="-.", label="Ua at step 0.03 (first try)")
        ax.axhline(-2, color=MUTED, lw=0.9, ls=":")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("time", fontsize=9, color=MUTED)
        ax.set_ylabel("(E(t) − E(0)) / perturbation energy, %", fontsize=9, color=MUTED)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        _style(ax)
    ax = axs[2]
    for i, j in enumerate(("Ua", "Ub", "Uc", "Ud")):
        en = jobs[j]["energy"]
        ax.plot(en["t"], 100 * np.array(en["T"]) / jobs[j]["perturbation_energy"], lw=1.1, color=(BLUE, ORANGE, AQUA, MUTED)[i], label=j)
    ax.set_title("Kinetic share of the perturbation's energy", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("time", fontsize=9, color=MUTED)
    ax.set_ylabel("T / perturbation energy, %", fontsize=9, color=MUTED)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.suptitle("M5.47: the energy over the length of each job (dotted: the 2 % of K11)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "m5_47_energy.png", dpi=140)
    plt.close(fig)

    # 6. the kinetic factor and the field, in the plane y = 0
    import m5_47_reads as RD

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for row, k in enumerate(("b", "d")):
        G = RD.geometry(g[k]["base"], a.data)
        N, x, hb = G["N"], G["x"], G["hb"]
        jy = N // 2
        ext = [-hb, hb, -hb, hb]
        qmap = np.clip(G["q"][:, jy, :], 1, 7)
        qmap[[0, -1], :] = np.nan
        qmap[:, [0, -1]] = np.nan
        im = axs[row, 0].imshow(qmap.T, origin="lower", extent=ext, cmap=seq, vmin=1, vmax=7)
        axs[row, 0].contour(x, x, G["region"][:, jy, :].T.astype(float), levels=[0.5], colors=[ORANGE], linewidths=1.0)
        fig.colorbar(im, ax=axs[row, 0], fraction=0.046, pad=0.02)
        axs[row, 0].set_title(f"Grid {k}: q (clipped at 7); orange: read region", fontsize=8.5, color=INK, loc="left")
        U = WV.JOBS[g[k]["job"]]
        m0, mstart = WV.start_state(U, a.data)
        z = np.load(a.data / f"m5_47_{g[k]['job']}_fourier.npz", allow_pickle=False)
        zn = np.load(a.data / f"m5_47_{g[k]['twin']}_fourier.npz", allow_pickle=False)
        g6 = G["W6"] * E.METRIC
        s0 = ((mstart - m0) * g6).sum(-1)[:, jy, :]
        s1 = ((z["m_end"] - zn["m_end"]) * g6).sum(-1)[:, jy, :]
        ti = int(np.argmin(np.abs(z["targets"] - 6.0)))
        amp = np.abs((((z["Ar"][..., ti] - zn["Ar"][..., ti]) + 1j * (z["Ai"][..., ti] - zn["Ai"][..., ti])) * g6).sum(-1))[:, jy, :]
        vm = float(np.abs(s0).max())
        tend = jobs[g[k]["job"]]["length"]
        for ax, fld, title, cm, vmn, vmx in ((axs[row, 1], s0, "⟨W, M − M₀⟩ at t = 0", "RdBu_r", -vm, vm), (axs[row, 2], s1, f"⟨W, M − M(twin)⟩ at t = {tend:g}", "RdBu_r", -vm, vm),
                                             (axs[row, 3], amp, "|a(x, ω = 6)|, Fourier sum in W", seq, 0, float(amp.max()))):
            im = ax.imshow(fld.T, origin="lower", extent=ext, cmap=cm, vmin=vmn, vmax=vmx)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_title(f"Grid {k}: {title}", fontsize=8.5, color=INK, loc="left")
        for ax in axs[row]:
            ax.set_xlabel("x", fontsize=8, color=MUTED)
            ax.set_ylabel("z", fontsize=8, color=MUTED)
            ax.tick_params(colors=MUTED, labelsize=7)
    fig.suptitle("M5.47 relaxed unit, sections y = 0 (the loop lies in the plane z = 0): the kinetic factor, the start, the end, one Fourier sum", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "m5_47_sections.png", dpi=130)
    plt.close(fig)
    print("plots written to", out)
