"""M5.40: the figures, from the job files and the summary (headless).

    python3 m5_40_plots.py

Figures (research/plots/): m5_40_r3_fields.png (the central slice of the author's two fields and what
the continuation changed), m5_40_r3_continuation.png (energy and force residual along the two
continuations), m5_40_r2_profiles.png (the record's eigenvalues against the radius, the author's field
beside the state of M5.39 on the same grid), m5_40_r4_ladder.png (the readouts of the author's scripts
on the mesh ladder of M5.39, against the square of the spacing).
Colors: one fixed hue per source on every figure (the author's field, the state of M5.39, the
continuation), with a marker or a line style as the second encoding; a single-hue ramp for magnitudes
and two hues around a neutral midpoint for signed fields.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATA, PLOTS = HERE.parent / "data", HERE.parent / "plots"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e3e3", "#fcfcfb"
AUTHOR, OURS, CONT = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f0a27f", "#b8431a"])
GATE = 1e-3


def style(ax, title=None):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")


def load(name):
    p = DATA / name
    return json.loads(p.read_text()) if p.exists() else None


def slice_fields(path):
    with np.load(path, allow_pickle=False) as d:
        state = d["state"]
    n = state.shape[0]
    sl = state[:, :, n // 2]
    p = sl[..., :4]
    q0 = p[..., 0] / np.linalg.norm(p, axis=-1)
    f = sl[..., 4:].reshape(n, n, 3, 3)
    lam = np.linalg.eigvalsh(np.swapaxes(f, -1, -2) @ f)[..., 2]
    return q0, lam


def fig_fields():
    rows = [(33, 3.6), (55, 6.075)]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6), facecolor="white")
    for row, (grid, box) in enumerate(rows):
        start = DATA / f"m5_40_author_g{grid}.npz"
        end = DATA / f"m5_40_r3_g{grid}_cont.npz"
        q0, lam = slice_fields(start)
        ext = (-box, box, -box, box)
        panels = [(q0, DIV, (-1, 1), f"{grid}³, the author's field: q0"), (lam, SEQ, (0, max(0.4, float(lam.max()))), "largest record eigenvalue")]
        if end.exists():
            q1, lam1 = slice_fields(end)
            dq, dl = q1 - q0, lam1 - lam
            mq, ml = float(np.abs(dq).max()) or 1e-12, float(np.abs(dl).max()) or 1e-12
            panels += [(dq, DIV, (-mq, mq), "change of q0 along the continuation"), (dl, DIV, (-ml, ml), "change of the largest eigenvalue")]
        for col, (field, cmap, lim, title) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(field.T, origin="lower", extent=ext, cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_title(title, color=INK, fontsize=10, loc="left")
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_xlabel("x (card lengths)", color=MUTED, fontsize=8.5)
            ax.set_ylabel("y", color=MUTED, fontsize=8.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=8, colors=MUTED)
        for col in range(len(panels), 4):
            axes[row, col].axis("off")
    fig.suptitle("The author's Card C fields on the central slice z = 0, and what the continuation changed", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOTS / "m5_40_r3_fields.png", dpi=140)
    plt.close(fig)


def fig_continuation(summary):
    refs = {33: [("m5_39_r2_g33_linked_ext2.json", "M5.39, the linked route's end"), ("m5_39_r2_g33_iso.json", "M5.39, the census state")],
            55: [("m5_39_r2_g55_smooth.json", "M5.39, the smooth start"), ("m5_39_r2_g55_chain.json", "M5.39, the chain")]}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), facecolor="white", sharex="col")
    for col, grid in enumerate((33, 55)):
        j = load(f"m5_40_r3_g{grid}_cont.json")
        rec = summary["R1"][str(grid)]["K1"]["recorded"]
        ax, bx = axes[0, col], axes[1, col]
        style(ax, f"{grid}³: energy minus the energy of the author's run record ({rec:.5f})")
        style(bx, f"{grid}³: largest gradient entry over h³")
        if j:
            it = np.array([0] + [r["iteration"] for r in j["history"]])
            e = np.array([j["start"]["energy"]] + [r["energy"] for r in j["history"]]) - rec
            res = np.array([j["start"]["residual"]] + [r["residual"] for r in j["history"]])
            ax.plot(it, e * 1e4, color=CONT, linewidth=2)
            ax.annotate(f"end of the continuation: {e[-1] * 1e4:+.2f}", (it[-1], e[-1] * 1e4), textcoords="offset points", xytext=(-120, 75), ha="right", color=INK, fontsize=9,
                        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
            bx.semilogy(it, res, color=CONT, linewidth=1.3)
            under = res <= GATE
            bx.scatter(it[under], res[under], s=14, color=CONT, edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.axhline(0.0, color=AUTHOR, linewidth=1.4)
        ax.annotate("the author's field", (0.99, 0.0), xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points", ha="right", color=AUTHOR, fontsize=9)
        for name, text in refs[grid]:
            r = load(name)
            if r:
                y = (r["energy"] - rec) * 1e4
                ax.axhline(y, color=OURS, linewidth=1.2, linestyle=(0, (5, 3)))
                ax.annotate(f"{text} ({y:+.2f})", (0.99, y), xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points", ha="right", color=OURS, fontsize=9)
        ax.set_ylabel("energy difference (10⁻⁴ card units)", color=MUTED, fontsize=9)
        bx.axhline(GATE, color=MUTED, linewidth=1.1, linestyle=(0, (2, 2)))
        bx.annotate("the card's gate, 10⁻³", (0.99, 0.94), xycoords="axes fraction", ha="right", color=MUTED, fontsize=9)
        bx.plot([0.66, 0.72], [0.955, 0.955], transform=bx.transAxes, color=MUTED, linewidth=1.1, linestyle=(0, (2, 2)))
        bx.set_xlabel("iteration of the continuation (saved every 50; dots: saved states under the gate)", color=MUTED, fontsize=9)
        bx.set_ylabel("residual", color=MUTED, fontsize=9)
    fig.suptitle("Continuation from the author's fields under the card's optimizer settings, stop at a tenth of the gate", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOTS / "m5_40_r3_continuation.png", dpi=140)
    plt.close(fig)


def fig_profiles(summary):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), facecolor="white", sharex="col")
    for col, grid in enumerate(("33", "55")):
        pr = summary["R2"][grid]["profiles"]
        a, b = pr["the_authors"], pr["reference"]
        ra = np.array([(r["r_lo"] + r["r_hi"]) / 2 for r in a])
        rb = np.array([(r["r_lo"] + r["r_hi"]) / 2 for r in b])
        ax, bx = axes[0, col], axes[1, col]
        style(ax, f"{grid}³: shell means of the record's three eigenvalues")
        for key, ls in (("eig_high", "-"), ("eig_mid", "--"), ("eig_low", ":")):
            ax.plot(ra, [r[key] for r in a], color=AUTHOR, linewidth=2, linestyle=ls)
            ax.plot(rb, [r[key] for r in b], color=OURS, linewidth=0, marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.6)
        ax.plot([], [], color=AUTHOR, linewidth=2, label="the author's field (lines: largest, middle, smallest)")
        ax.plot([], [], color=OURS, linewidth=0, marker="o", markersize=5, label=f"M5.39, {pr['reference_name']} (markers)")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
        ax.set_ylabel("eigenvalue", color=MUTED, fontsize=9)
        style(bx, f"{grid}³: the state of M5.39 minus the author's field, shell by shell")
        for key, color, marker, text in (("trace_s", OURS, "o", "trace of the record"), ("q0", CONT, "s", "q0")):
            d = np.array([rb_[key] for rb_ in b]) - np.array([ra_[key] for ra_ in a])
            bx.plot(ra, d * 1e3, color=color, linewidth=1.6, marker=marker, markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.6, label=text)
        bx.axhline(0.0, color=MUTED, linewidth=1.0)
        bx.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
        bx.set_xlabel("radius (card lengths), shells of 0.25", color=MUTED, fontsize=9)
        bx.set_ylabel("difference (10⁻³)", color=MUTED, fontsize=9)
    fig.suptitle("The author's fields beside the states of M5.39 on the same grids", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOTS / "m5_40_r2_profiles.png", dpi=140)
    plt.close(fig)


def fig_ladder(summary):
    mesh = summary["R4"]["mesh_ladder"]
    if "series" not in mesh:
        return
    ro = load("m5_40_r4_readouts.json")
    h2 = np.array(mesh["spacing_squared"])
    a55 = ro["m5_40_author_g55"]
    keys = [("rS2_fm2", "r_S² (fm²)", a55["rS2_fm2"]), ("rV2_fm2", "r_V² (fm²)", a55["rV2_fm2"]), ("muV", "μ_V (nuclear magnetons, inherited scale)", a55["muV"]),
            ("e2_gA", "g_A e² (the axial twist, before any e)", a55["axial"]["e2_gA"])]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), facecolor="white")
    for ax, (key, title, author_value) in zip(axes.ravel(), keys):
        style(ax, title)
        y = np.array(mesh["series"][key])
        fit = mesh["fits_in_h2"][key]
        slope = (y[-1] - fit["extrapolated_from_the_three_finest"]) / h2[-1]
        xs = np.array([0.0, h2[1]])
        ax.plot(xs, fit["extrapolated_from_the_three_finest"] + np.polyfit(h2[1:], y[1:], 1)[0] * xs, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.plot(h2, y, color=OURS, linewidth=1.6, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=0.8, label="states of M5.39 (45³, 55³, 65³, 77³)")
        ax.plot([h2[1]], [author_value], color=AUTHOR, linewidth=0, marker="D", markersize=8, markerfacecolor="none", markeredgewidth=2, label="the author's 55³ field")
        ax.plot([0.0], [fit["extrapolated_from_the_three_finest"]], color=MUTED, linewidth=0, marker="x", markersize=8, markeredgewidth=2, label="straight line through the three finest, at h² = 0")
        ax.annotate(f"{fit['extrapolated_from_the_three_finest']:.4f}", (0.0, fit["extrapolated_from_the_three_finest"]), textcoords="offset points", xytext=(8, -3), color=MUTED, fontsize=9)
        for g, x, v in zip(mesh["grids"], h2, y):
            ax.annotate(f"{g}³", (x, v), textcoords="offset points", xytext=(0, 9), ha="center", color=INK, fontsize=8.5)
        ax.set_xlabel("h² (card lengths squared), half-box 6.075", color=MUTED, fontsize=9)
        ax.set_xlim(-0.004, h2[0] * 1.06)
        del slope
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.suptitle("The readouts of the author's scripts on the mesh ladder of M5.39 (gate-passing states, not converged ones)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOTS / "m5_40_r4_ladder.png", dpi=140)
    plt.close(fig)


def main():
    summary = load("m5_40_summary.json")
    fig_fields()
    fig_continuation(summary)
    fig_profiles(summary)
    fig_ladder(summary)
    print("figures written to", PLOTS)


if __name__ == "__main__":
    main()
