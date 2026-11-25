"""M5.49: the tables and the plots of the record, from the prediction file and the summary of the reads.

    python m5_49_report.py --tables      writes checkpoints/m5_49/tables.md (the tables of the findings)
    python m5_49_report.py --plots       writes plots/m5_49_*.png

Nothing here opens a Fourier sum: the tables come from data/m5_49_bands_predicted.json and data/m5_49_reads.json,
the maps of q from the base states, and the sections of the wave from the slices the reads script saved.
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

import m5_49_predict as PR

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#d9d8d3"
LINE_NAME = dict(A="A (axis)", PF="PF (plane, toward the faces)", PD="PD (plane, diagonals)", O="O (oblique)")
GRID_NAME = dict(a="half-box 8, spacing 0.5", b="half-box 8, spacing 0.25", c="half-box 10, spacing 0.5", d="half-box 10, spacing 0.25", e="half-box 13, spacing 0.25")
REG = {"4.0": dict(A={}, plane=dict(inner=2.77, outer=6.31)), "5.5": dict(A=dict(inner=0.81, outer=8.38), plane=dict(inner=1.56, outer=9.74))}


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "yes" if v else "no"
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def pct(a, b):
    return "n/a" if a is None or b is None else f"{100 * (a / b - 1):+.1f} %"


def arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


def load(a):
    return json.loads((a.data / "m5_49_bands_predicted.json").read_text()), json.loads((a.data / "m5_49_reads.json").read_text())


def edges_text(edges, kind=None):
    es = [e for e in edges if kind is None or e["kind"] == kind]
    return ", ".join(f"{e['r']:.2f} ({e['kind']}{', midpoint' if e.get('midpoint') else ''})" for e in es) if es else "none"


def tables(a):
    doc, res = load(a)
    out = []
    w = out.append
    G = res["grids"]
    # the margins
    w("#### The margin of each grid\n")
    w("| Grid | Twice the worst gap of C6 (a) | Twice the median of C10 | Twice the bias at 4 | Twice the bias at 5.5 | Margin δ |")
    w("| --- | --- | --- | --- | --- | --- |")
    for g, info in G.items():
        t = info["margin_terms"]
        w(f"| {g} ({GRID_NAME[g]}) | {t['twice_C6a']:.4f} | {t['twice_C10']:.4f} | {t['twice_beta4']:.4f} | {t['twice_beta55']:.4f} | {info['margin']:.4f} |")
    w("")
    # the two tables of entries
    for col, title in (("R", "Against F, the prediction from the background of the same grid"), ("N", "Against the registered numbers")):
        w(f"#### {title}\n")
        w("| Reading | Grid | Line | ω | Entry | Expected at | Read at | Word | What withheld it, or the note | The three profiles | Second draw |")
        w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for en in res["entries"]:
            if en["column"] != col:
                continue
            if en["what"].startswith("band"):
                exp = en.get("F_statement", en.get("registered")) if col == "R" else "no band"
                rd = "; ".join(dict.fromkeys(x or "undecided" for x in en.get("read_statement", [])))
                runs = "; ".join(f"run {r['start']:g} to {r['end']:g}, half wavelength {r['half_wavelength']:.2f}" for r in en.get("runs", []))
                sec = (en.get("second_statement") or "undecided") if G[en["grid"]]["C10"].get("median") is not None else "n/a"
                w(f"| {en['reading']} | {en['grid']}{'' if en['labeled'] else ' (tabled)'} | {en['line']} | {en['w']:g} | {en['what']} | {exp if exp is not None else 'inside the margin'} | "
                  f"{rd}{' (' + runs + ')' if runs else ''} | {en['word']} | {en['why']} | {rd} | {sec} |")
            else:
                r0 = en["r0"] if en["r0"] is not None else en.get("r0_ray")
                prof = ", ".join(en["profiles"]) if en.get("profiles") else "n/a"
                sec = f"{en['second_word']} ({fmt(en['second_read'], 2)})" if en.get("second_word") else "n/a"
                extra = f" (ray form {fmt(en.get('r0_ray'), 2)}, q_P {fmt(en.get('r0_qP'), 2)})" if col == "R" and en["r0"] is not None else (" (ray form only)" if col == "R" and r0 is not None else "")
                w(f"| {en['reading']} | {en['grid']}{'' if en['labeled'] else ' (tabled)'} | {en['line']} | {en['w']:g} | {en['what']} | {fmt(r0, 2)}{extra} | {fmt(en.get('read'), 2)} | "
                  f"{en['word']} | {en['why']} | {prof} | {sec} |")
        w("")
    w("#### The words per grid and over the labeled grids\n")
    names = ("R1", "R2", "R3", "R4", "N1", "N2", "N3", "N4")
    w("| Grid | " + " | ".join(names) + " |")
    w("| --- |" + " --- |" * len(names))
    for g, info in G.items():
        w(f"| {g}{'' if info['labeled'] else ' (tabled)'} | " + " | ".join(info["words"][n] for n in names) + " |")
    w("| **b, d, e** | " + " | ".join(f"**{res['words'][n]}**" for n in names) + " |")
    w("")
    # the controls
    w("#### Controls on the pieces\n")
    w("| Grid | C3: q against the matrix of the stepper | C7: twin share, worst over kept targets | C8: quotient, worst gap | Pieces set aside by C7 or C8 | C9: energy | C10: pieces, median | C11: worst over the windows | C12 | C14: worst edge, statements |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for g, info in G.items():
        c9 = ", ".join(f"{j} {100 * v['ratio']:.2f} %" for j, v in info["C9"].items()) or "stored jobs of M5.47"
        c10 = info["C10"]
        c11 = max([c["worst"] for c in info["C11"]], default=0.0)
        c12 = ", ".join(f"{c['lines'][0]}/{c['lines'][1]} {c['kind']} at {c['w']:g}: {'met' if c['passed'] else ('n/a' if c['passed'] is None else 'NOT MET')}" for c in info["C12"]) or "no pair of lines apart by 10 %"
        e14 = [c for c in info["C14"] if c["kind"] != "statement"]
        s14 = [c for c in info["C14"] if c["kind"] == "statement"]
        w14 = max([c["rel"] for c in e14 if c["rel"] is not None], default=0.0)
        c14 = f"{100 * w14:.2f} % ({sum(c['passed'] for c in e14)} of {len(e14)} edges met); statements {sum(c['passed'] for c in s14)} of {len(s14)} alike"
        w(f"| {g} | {info['C3']['max_abs']:.1e} | {info['C7']['worst']:.2e} | {100 * info['C8']['worst']:.2f} % | {info['C7']['pieces_failed']} and {info['C8']['pieces_failed']} | {c9} | "
          f"{c10.get('pieces', 'n/a')}, {fmt(c10.get('median'), 4)} | {100 * c11:.2f} % | {c12} | {c14} |")
    w("")
    # descriptor B
    w("#### Descriptor B: the edges of F against the registered numbers, and the two forms of F\n")
    w("| Grid | Line | ω | Edge | F on the pieces | F on the ray | F on the pieces, q_P | Registered | F on the ray against the registered number |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for g, info in G.items():
        for ln in ("A", "PF", "PD"):
            Lb = doc["blocks"][g]["lines"][ln]
            for om in ("4.0", "5.5"):
                reg = REG[om]["A"] if ln == "A" else REG[om]["plane"]
                for kind in ("inner", "outer"):
                    p = [e["r"] for e in Lb["edges"]["pieces"][om] if e["kind"] == kind]
                    r = [e["r"] for e in Lb["edges"]["ray"][om] if e["kind"] == kind]
                    pp = [e["r"] for e in Lb["edges"]["pieces_qP"][om] if e["kind"] == kind]
                    rg = reg.get(kind)
                    if not (p or r or rg):
                        continue
                    rr = r[-1] if (r and kind == "outer") else (r[0] if r else None)
                    w(f"| {g} | {ln} | {float(om):g} | {kind} | {', '.join(f'{x:.2f}' for x in p) or 'none'} | {', '.join(f'{x:.2f}' for x in r) or 'none'} | "
                      f"{', '.join(f'{x:.2f}' for x in pp) or 'none'} | {fmt(rg, 2) if rg else 'no band' if (ln == 'A' and om == '4.0') else 'n/a'} | {pct(rr, rg) if rg else 'n/a'} |")
    w("")
    w("| Grid | Least m_F² on the axis, ray form | At radius | Least m_F² on the axis, pieces | Registered |")
    w("| --- | --- | --- | --- | --- |")
    for g in G:
        l = doc["blocks"][g]["lines"]["A"]["least_mF"]
        w(f"| {g} | {l['ray'] ** 2:.3f} | {l['ray_at']:.2f} | {l['pieces'] ** 2:.3f} | 16.577 at 2.50 |")
    w("")
    # the read edges, side by side, with C
    w("#### The edges of the read, with the edges of F and of C on the same line\n")
    w("| Grid | Line | ω | Edges of the read | Second draw | Edges of F on the pieces | Edges of C | Read against C | Statement of the read | Runs: from, to, half wavelength |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for g, info in G.items():
        for ln in PR.LINES:
            L = info["lines"][ln]
            Lb = doc["blocks"][g]["lines"][ln]
            for om in ("4.0", "5.5"):
                at = L["at"][om]
                ce = PR.crossings(arr(L["centers"]), np.where(np.array(L["read"], bool), arr(L["C"]), np.nan), float(om))
                comp = []
                for e in at["edges"]:
                    same = [c for c in ce if c["kind"] == e["kind"]]
                    if same:
                        c = min(same, key=lambda c: abs(c["r"] - e["r"]))
                        comp.append(f"{e['kind']} {pct(e['r'], c['r'])}")
                runs = "; ".join(f"{r['start']:g} to {r['end']:g}, {r['half_wavelength']:.2f}{' (band)' if r['band'] else ''}" for r in at["runs"]) or "none"
                st = "; ".join(dict.fromkeys(s or "undecided" for s in at["statement"]))
                w(f"| {g} | {ln} | {float(om):g} | {edges_text(at['edges'])} | {edges_text(at['second']['edges']) if at['second'] else 'n/a'} | {edges_text(Lb['edges']['pieces'][om])} | "
                  f"{edges_text(ce)} | {', '.join(comp) or 'n/a'} | {st} | {runs} |")
    w("")
    # the profiles of the labeled lines
    for g in a.profile_grids:
        if g not in G:
            continue
        info = G[g]
        for ln in PR.LINES:
            L = info["lines"][ln]
            w(f"#### Grid {g}, line {ln}: the pieces\n")
            w("| s | Nodes in the domain | Read | q̄ | m_F | m_F′ | C | ω_c | Second draw | c² | t\\* | Kept | Status at 4 | Status at 5.5 | K² at 4 | Outside W at 4 | K² at 5.5 | Outside W at 5.5 |")
            w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for i, s in enumerate(L["centers"]):
                if s > doc["blocks"][g]["lines"][ln]["read_to"] + 1e-9 and not L["read"][i]:
                    continue
                sec = L["wc_second"][i] if L["wc_second"] else None
                w(f"| {s:.2f} | {L['nodes_domain'][i]} | {fmt(bool(L['read'][i]))} | {fmt(L['q'][i])} | {fmt(L['mF'][i])} | {fmt(L['mF_prime'][i])} | {fmt(L['C'][i])} | "
                  f"{fmt(L['wc'][i])}{' (first target)' if L['first'][i] and L['wc'][i] is not None else ''} | {fmt(sec)} | {fmt(L['c2'][i])} | {fmt(L['t_star'][i], 2)} | {L['kept'][i]} | "
                  f"{L['at']['4.0']['status'][i] or 'not read'} | {L['at']['5.5']['status'][i] or 'not read'} | {fmt(L['at']['4.0']['K2'][i], 2)} | {fmt(L['at']['4.0']['outside_W'][i])} | "
                  f"{fmt(L['at']['5.5']['K2'][i], 2)} | {fmt(L['at']['5.5']['outside_W'][i])} |")
            w("")
    (a.work / "tables.md").write_text("\n".join(out) + "\n")
    print("tables written:", a.work / "tables.md", len(out), "lines")


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plots(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    doc, res = load(a)
    out = a.plots_dir
    seq = LinearSegmentedColormap.from_list("seq", ["#eaf1fb", "#b9d3f2", BLUE, "#0d2f5e"])
    G = res["grids"]
    # 1. the profiles: the cutoff read on each line beside the predictions
    for g, info in G.items():
        fig, axs = plt.subplots(2, 2, figsize=(14, 8.4), sharey=True)
        for ax, ln in zip(axs.ravel(), PR.LINES):
            L, Lb = info["lines"][ln], doc["blocks"][g]["lines"][ln]
            s, read = arr(L["centers"]), np.array(L["read"], bool)
            rf = Lb["ray_form"]
            ax.plot(rf["s"], rf["mF"], color=BLUE, lw=0.9, alpha=0.55, label="F on the ray")
            # pieces that are not read are gaps of the curves (no line is drawn across them)
            ax.plot(s, np.where(read, arr(L["mF"]), np.nan), "-", color=BLUE, lw=1.8, label="F on the pieces: √(60.09 / q̄)")
            ax.plot(s, np.where(read, arr(L["mF_prime"]), np.nan), "--", color=MUTED, lw=1.1, label="F′ (no label)")
            ax.plot(s, np.where(read, arr(L["C"]), np.nan), ":", color=AQUA, lw=2.0, label="C: the cutoffs of M5.47 at q̄ (no label)")
            wc, first = arr(L["wc"]), np.array(L["first"], bool)
            ok = np.isfinite(wc)
            ax.plot(s[ok & ~first], wc[ok & ~first], "o", ms=5.5, color=ORANGE, mec="white", mew=0.6, label="read: cutoff ω_c of the piece")
            if (ok & first).any():
                ax.plot(s[ok & first], wc[ok & first], "o", ms=5.5, mfc="none", mec=ORANGE, mew=1.2, label="read, least K² at the first target (OPEN)")
            if L["wc_second"]:
                ax.plot(s, arr(L["wc_second"]), "x", ms=4.5, color=INK, mew=0.9, alpha=0.7, label="second draw")
            for om in (4.0, 5.5):
                ax.axhline(om, color=INK, lw=0.8)
                ax.text(0.995, om + 0.06, f"ω = {om:g}", fontsize=7.5, color=INK, va="bottom", ha="right", transform=ax.get_yaxis_transform())
                reg = REG[f"{om:.1f}"]["A"] if ln == "A" else REG[f"{om:.1f}"]["plane"]
                if ln != "O":
                    for r0 in reg.values():
                        ax.plot([r0], [om], marker="|", ms=16, mew=2.0, color=INK)
            ax.set_title(f"Line {LINE_NAME[ln]}", fontsize=10, color=INK, loc="left")
            ax.set_xlabel("s along the line", fontsize=9, color=MUTED)
            ax.set_ylim(2.4, 8.3)
            ax.set_xlim(0, (s[read][-1] + 1.2) if read.any() else 1)
            _style(ax)
        axs[0, 0].set_ylabel("frequency, cutoff, local mass", fontsize=9, color=MUTED)
        axs[1, 0].set_ylabel("frequency, cutoff, local mass", fontsize=9, color=MUTED)
        hd, lb = [], []
        for ax in axs.ravel():
            for h_, l_ in zip(*ax.get_legend_handles_labels()):
                if l_ not in lb:
                    hd.append(h_)
                    lb.append(l_)
        fig.legend(hd, lb, fontsize=8, frameon=False, loc="lower center", ncol=4)
        fig.suptitle(f"M5.49 grid {g} ({GRID_NAME[g]}): the cutoff of the mid wave read on each piece, beside the local mass predicted from the background.\n"
                     f"A band at ω is where the profile lies under the line of ω. Vertical ticks: the registered edges. Margin of the grid δ = {info['margin']:.3f}",
                     fontsize=10.5, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        fig.savefig(out / f"m5_49_profiles_{g}.png", dpi=130)
        plt.close(fig)
        print("plot", f"m5_49_profiles_{g}.png")
    # 2. the map of q with the lines drawn on it
    grids = [g for g in ("b", "d", "e") if g in G]
    fig, axs = plt.subplots(len(grids), 2, figsize=(11.5, 5.2 * len(grids)), squeeze=False)
    for row, g in enumerate(grids):
        B = PR.background(g, a.data, with_A=False)
        x, N, hb = B["x"], B["N"], B["hb"]
        j = N // 2
        ext = [-hb, hb, -hb, hb]
        blk = doc["blocks"][g]
        for col, (sl, dm, name, lines, axes) in enumerate(((B["q"][:, j, :], B["dom"][:, j, :], "section y = 0 (the axis is vertical)", ("A", "PF", "O"), (0, 2)),
                                                            (B["q"][:, :, j], B["dom"][:, :, j], "plane z = 0 (the plane of the loop)", ("PF", "PD"), (0, 1)))):
            ax = axs[row, col]
            sl = sl.copy()
            sl[[0, -1], :] = np.nan      # the outermost layer of nodes: the periodic wrap of the difference, not the field
            sl[:, [0, -1]] = np.nan
            im = ax.imshow(sl.T, origin="lower", extent=ext, cmap=seq, vmin=1, vmax=7)
            ax.contour(x, x, sl.T, levels=[PR.M_MID2 / 5.5 ** 2, PR.M_MID2 / 4.0 ** 2], colors=[INK, INK], linewidths=[0.8, 1.4], linestyles=["--", "-"])
            ax.contour(x, x, dm.T.astype(float), levels=[0.5], colors=[ORANGE], linewidths=1.0)
            for ln, ls in zip(lines, ("-", "-", ":")):
                Lb = blk["lines"][ln]
                for e in Lb["rays"]:
                    e = np.array(e)
                    if abs(e[3 - sum(axes)]) > 1e-9:
                        continue
                    end = Lb["read_to"] * e
                    ax.plot([0, end[axes[0]]], [0, end[axes[1]]], ls, color="white", lw=1.6)
                    ax.plot([0, end[axes[0]]], [0, end[axes[1]]], ls, color=AQUA, lw=0.9)
                    ax.text(end[axes[0]] * 1.04, end[axes[1]] * 1.04, ln, fontsize=7.5, color=INK, ha="center", va="center")
            ax.set_title(f"Grid {g}, {name}", fontsize=10, color=INK, loc="left")
            ax.set_xlabel("x", fontsize=9, color=MUTED)
            ax.set_ylabel("z" if col == 0 else "y", fontsize=9, color=MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("kinetic factor q (clipped at 7)", fontsize=8, color=MUTED)
            cb.ax.tick_params(labelsize=7, colors=MUTED)
    fig.suptitle("M5.49: the kinetic factor q of the base state, with the lines of the read drawn to where they stop being read.\n"
                 "Black contours: q = 3.756 (solid; inside it the local mass of F is under 4) and q = 1.986 (dashed; under 5.5). Orange: the edge of the domain.",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (5.2 * len(grids))))   # room for the two lines of the title
    fig.savefig(out / "m5_49_qmap.png", dpi=120)
    plt.close(fig)
    print("plot m5_49_qmap.png")
    # 3. the field of the job on the widest grid read: the start, the end, and the Fourier modulus at the two frequencies
    for g in [x for x in ("e", "d") if x in G][:1]:
        z = np.load(a.work / f"sums_{G[g]['job']}.npz", allow_pickle=False)
        hb = G[g]["half_box"]
        ext = [-hb, hb, -hb, hb]
        fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
        div = LinearSegmentedColormap.from_list("div", [ORANGE, "#f4f3ef", BLUE])
        for ax, key, title in ((axs[0], "start_y", "⟨W, M − M₀⟩ at t = 0 (the noise)"), (axs[1], "end_y", "⟨W, M − M₀⟩ at the end, t = 72")):
            v = z[key]
            m = float(np.abs(v).max())
            im = ax.imshow(v.T, origin="lower", extent=ext, cmap=div, vmin=-m, vmax=m)
            ax.set_title(title, fontsize=9.5, color=INK, loc="left")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7, colors=MUTED)
        for ax, om in ((axs[2], 4.0), (axs[3], 5.5)):
            v = np.log10(np.abs(z[f"sec_y_{om:g}"]) + 1e-300)
            top = float(v.max())
            im = ax.imshow(v.T, origin="lower", extent=ext, cmap=seq, vmin=top - 3, vmax=top)
            ax.set_title(f"log₁₀ |a(x, ω = {om:g})|, top three decades", fontsize=9.5, color=INK, loc="left")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7, colors=MUTED)
        for ax in axs:
            ax.set_xlabel("x", fontsize=9, color=MUTED)
            ax.set_ylabel("z", fontsize=9, color=MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
        fig.suptitle(f"M5.49 job {G[g]['job']} (grid {g}), section y = 0: the perturbation in the mid polarization at the start and at the end of the job, "
                     "and the modulus of its Fourier sum (job minus twin) at the two test frequencies", fontsize=10.5, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"m5_49_sections_{g}.png", dpi=120)
        plt.close(fig)
        print("plot", f"m5_49_sections_{g}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--profile-grids", nargs="*", default=["e"])
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    ap.add_argument("--plots-dir", type=Path, default=HERE.parent / "plots")
    ap.add_argument("--work", type=Path, default=HERE.parent / "checkpoints" / "m5_49")
    a = ap.parse_args()
    if a.tables:
        tables(a)
    if a.plots:
        plots(a)


if __name__ == "__main__":
    main()
