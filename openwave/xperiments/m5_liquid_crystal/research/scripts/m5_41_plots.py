"""M5.41: the figures, from the route files and the summary (headless).

    python3 m5_41_plots.py

Figures (research/plots/):
  m5_41_r2_descent.png   the energy above the lowest last stop of each mesh, and the gate, along
                         the twelve routes, with the saved stops marked;
  m5_41_r3_readouts.png  the three labeled readouts at each stop, as their relative distance from
                         the route's last stop, against the bound of the rule;
  m5_41_r4_lines.png     the readouts at the last stops against the square of the spacing, the
                         straight line through the three finest meshes, the intercept with its
                         route range, and the residuals of the routes about the line;
  m5_41_r2_fields.png    the central slice of the three 77^3 starts and of their last stops.
Colors: one fixed hue per start on every figure (iso, chain, smooth, the author's field), with the
marker as the second encoding; a single-hue ramp for magnitudes and two hues around a neutral
midpoint for signed fields.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATA, PLOTS = HERE.parent / "data", HERE.parent / "plots"
CHAIN_DIR = HERE.parent / "checkpoints" / "m5_39"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e3e3", "#fcfcfb"
COLOR = dict(iso="#eb6834", chain="#1baf7a", smooth="#c2417f", author="#2a78d6")
MARK = dict(iso="o", chain="s", smooth="^", author="D")
NAME = dict(iso="iso", chain="chain", smooth="smooth", author="the author's field")
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f0a27f", "#b8431a"])
GATE, BOUND = 1e-3, 1e-4
GRIDS = (45, 55, 65, 77)
BOX = 6.075


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


def legend(fig, kinds, loc="upper right", extra=()):
    handles = [Line2D([0], [0], color=COLOR[k], marker=MARK[k], markersize=7, linewidth=2, label=NAME[k]) for k in kinds] + list(extra)
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=(0.995, 0.962), ncol=len(handles), frameon=False, fontsize=9.5, labelcolor=INK)


def fig_descent(summary):
    labels = {lab["name"]: lab for lab in summary["labels"]}
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.6), facecolor="white", sharex=True)
    for col, grid in enumerate(GRIDS):
        mesh = summary["meshes"].get(str(grid))
        ax, bx = axes[0, col], axes[1, col]
        style(ax, f"{grid}³: energy above the lowest last stop of the mesh")
        style(bx, f"{grid}³: the gate, lowest value in each 50 iterations")
        if mesh is None:
            continue
        low = mesh["energy"]["lowest"]
        for name in mesh["routes"]:
            kind = labels[name]["kind"]
            p = DATA / f"m5_41_{name}.npz"
            if not p.exists():
                continue
            with np.load(p, allow_pickle=False) as d:
                it, e, gate = d["trace_it"], d["trace_e"], d["trace_gate"]
            ax.loglog(it, np.maximum(e - low, 1e-9), color=COLOR[kind], linewidth=1.5)
            block = 50
            m = len(gate) // block
            gmin = gate[:m * block].reshape(m, block).min(axis=1)
            bx.loglog((np.arange(m) + 0.5) * block, gmin, color=COLOR[kind], linewidth=1.2)
            for row in labels[name]["stops"]:
                bx.scatter([row["iteration"]], [row["gate"]], s=46, marker=MARK[kind], color=COLOR[kind], edgecolor=SURFACE, linewidth=0.9, zorder=3)
                ax.scatter([row["iteration"]], [max(row["energy"] - low, 1e-9)], s=46, marker=MARK[kind], color=COLOR[kind], edgecolor=SURFACE, linewidth=0.9, zorder=3)
        bx.axhline(GATE, color=MUTED, linewidth=1.1, linestyle=(0, (2, 2)))
        bx.set_xlabel("iteration of the route", color=MUTED, fontsize=9)
        ax.set_ylim(1e-9, None)
    axes[0, 0].set_ylabel("energy difference (card units; floor of the axis 10⁻⁹)", color=MUTED, fontsize=9)
    axes[1, 0].set_ylabel("gate (unit-sphere winding and factor gradients over h³)", color=MUTED, fontsize=9)
    legend(fig, ("iso", "chain", "smooth", "author"),
           extra=[Line2D([0], [0], color=MUTED, linewidth=1.1, linestyle=(0, (2, 2)), label="the gate, 10⁻³")])
    fig.suptitle("The twelve routes on Card C at half-box 6.075: descent and gate, the saved stops marked", color=INK, fontsize=12.5, x=0.01, y=0.995, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(PLOTS / "m5_41_r2_descent.png", dpi=140)
    plt.close(fig)


def fig_readouts(summary):
    keys = [("inertia", "the inertia"), ("gA_e2", "g_A e²"), ("muV", "μ_V")]
    fig, axes = plt.subplots(len(GRIDS), 3, figsize=(15.5, 13.5), facecolor="white", sharex=True, sharey=True)
    for r, grid in enumerate(GRIDS):
        for c, (key, title) in enumerate(keys):
            ax = axes[r, c]
            style(ax, f"{grid}³, {title}: distance from the route's last stop")
            for lab in summary["labels"]:
                if lab["grid"] != grid or not lab.get("last_stop"):
                    continue
                last = [row for row in lab["stops"] if row["target"] == 30000][0]
                its = [row["iteration"] for row in lab["stops"] if row["target"] != 30000]
                d = [max(abs(row[key] - last[key]) / abs(last[key]), 1e-9) for row in lab["stops"] if row["target"] != 30000]
                ax.loglog(its, d, color=COLOR[lab["kind"]], marker=MARK[lab["kind"]], markersize=7, linewidth=1.5,
                          markeredgecolor=SURFACE, markeredgewidth=0.9)
            ax.axhline(BOUND, color=MUTED, linewidth=1.1, linestyle=(0, (2, 2)))
            if r == len(GRIDS) - 1:
                ax.set_xlabel("iteration of the saved stop", color=MUTED, fontsize=9)
            if c == 0:
                ax.set_ylabel("relative difference", color=MUTED, fontsize=9)
    legend(fig, ("iso", "chain", "smooth", "author"),
           extra=[Line2D([0], [0], color=MUTED, linewidth=1.1, linestyle=(0, (2, 2)), label="the bound of the rule, 10⁻⁴")])
    fig.suptitle("The labeled readouts at the earlier stops, relative to the last stop of the same route", color=INK, fontsize=12.5, x=0.01, y=0.995, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PLOTS / "m5_41_r3_readouts.png", dpi=140)
    plt.close(fig)


def fig_lines(summary):
    keys = [("inertia", "the inertia"), ("gA_e2", "g_A e²"), ("muV_calibrated", "μ_V at the gap Δ = 293 MeV"), ("gA_own_e", "g_A at each state's own e")]
    block = summary["lines"]["routes_meeting_the_rule"]
    tag = "the routes that meet the rule"
    if not block.get("available"):
        block, tag = summary["lines"]["all_routes_on_the_declared_branch"], "all routes on the declared branch"
    used = {n for names in block["routes"].values() for n in names} if block.get("available") else set()
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.8), facecolor="white")
    for c, (key, title) in enumerate(keys):
        ax, bx = axes[0, c], axes[1, c]
        style(ax, f"{title} against h²")
        style(bx, "residual about the line (relative, units of 10⁻⁴)")
        a = b = None
        if block.get("available"):
            a, b = block["central"][key]["intercept"], block["central"][key]["slope"]
            x = np.linspace(0.0, max(block["h2"]) * 1.02, 50)
            ax.plot(x, a + b * x, color=INK, linewidth=1.3)
            rr = block["route_range"][key]["half_range"]
            ax.errorbar([0.0], [a], yerr=[rr], color=INK, marker="_", markersize=12, capsize=5, linewidth=1.4)
            ax.annotate(f"{a:.5f}, route range {rr:.1e}", (0.0, a), textcoords="offset points", xytext=(10, -4), color=INK, fontsize=9.5)
        for lab in summary["labels"]:
            if not lab.get("last_stop"):
                continue
            last = [row for row in lab["stops"] if row["target"] == 30000][0]
            h2 = lab["spacing"] ** 2
            filled = lab["name"] in used
            ax.scatter([h2], [last[key]], s=58, marker=MARK[lab["kind"]], color=COLOR[lab["kind"]] if filled else SURFACE,
                       edgecolor=COLOR[lab["kind"]], linewidth=1.4, zorder=3)
            if a is not None:
                bx.scatter([h2], [(last[key] - (a + b * h2)) / abs(a) * 1e4], s=58, marker=MARK[lab["kind"]],
                           color=COLOR[lab["kind"]] if filled else SURFACE, edgecolor=COLOR[lab["kind"]], linewidth=1.4, zorder=3)
        bx.axhline(0.0, color=INK, linewidth=1.0)
        bx.set_xlabel("h² (card lengths squared)", color=MUTED, fontsize=9)
        ax.set_xlim(left=-0.002)
        bx.set_xlim(left=-0.002)
    legend(fig, ("iso", "chain", "smooth", "author"),
           extra=[Line2D([0], [0], color=INK, linewidth=1.3, label=f"line through 55³, 65³, 77³ on {tag}"),
                  Line2D([0], [0], color=MUTED, marker="o", markerfacecolor=SURFACE, linewidth=0, label="open: not in the line")])
    fig.suptitle("The readouts at the last stops against the square of the spacing; the bar at h² = 0 is the route range of the intercept",
                 color=INK, fontsize=12.5, x=0.01, y=0.995, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(PLOTS / "m5_41_r4_lines.png", dpi=140)
    plt.close(fig)


def slice_of(state):
    n = state.shape[0]
    sl = state[:, :, n // 2]
    p = sl[..., :4]
    q0 = p[..., 0] / np.linalg.norm(p, axis=-1)
    f = sl[..., 4:].reshape(n, n, 3, 3)
    lam = np.linalg.eigvalsh(np.swapaxes(f, -1, -2) @ f)[..., 2]
    return q0, lam


def fig_fields(grid=77):
    import m5_39_relax as RX

    kinds = ("iso", "chain", "smooth")
    fig, axes = plt.subplots(3, 4, figsize=(16.5, 11.6), facecolor="white")
    ext = (-BOX, BOX, -BOX, BOX)
    for r, kind in enumerate(kinds):
        if kind == "chain":
            args = SimpleNamespace(start="refine", source=CHAIN_DIR / "r2_g65_chain" / "handoff.npz", grid=grid, box=BOX, delta=0.2)
        else:
            args = SimpleNamespace(start=kind, source=None, grid=grid, box=BOX, delta=0.2)
        panels = []
        if args.source is None or args.source.exists():
            start = RX.build_start(args)[0]
            q0, lam = slice_of(start)
            panels += [(q0, DIV, (-1, 1), f"{grid}³ {kind}, the start: q0"), (lam, SEQ, (0, float(lam.max())), "the start: largest record eigenvalue")]
        end = DATA / f"m5_41_r2_g{grid}_{kind}_s30000.npz"
        if end.exists():
            with np.load(end, allow_pickle=False) as d:
                q1, lam1 = slice_of(d["state"])
                it = int(d["iteration"])
            panels += [(q1, DIV, (-1, 1), f"the last stop (iteration {it}): q0"), (lam1, SEQ, (0, max(0.4, float(lam1.max()))), "the last stop: largest record eigenvalue")]
        for c, (field, cmap, lim, title) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(field.T, origin="lower", extent=ext, cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_title(title, color=INK, fontsize=10, loc="left")
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_xlabel("x (card lengths)", color=MUTED, fontsize=8.5)
            ax.set_ylabel("y", color=MUTED, fontsize=8.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=8, colors=MUTED)
        for c in range(len(panels), 4):
            axes[r, c].axis("off")
    fig.suptitle(f"The three {grid}³ routes on the central slice z = 0: each start and its last stop", color=INK, fontsize=12.5, x=0.01, y=0.995, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOTS / "m5_41_r2_fields.png", dpi=140)
    plt.close(fig)


def main():
    PLOTS.mkdir(exist_ok=True)
    summary = json.loads((DATA / "m5_41_summary.json").read_text())
    fig_descent(summary)
    fig_readouts(summary)
    fig_lines(summary)
    fig_fields()
    print("figures written to", PLOTS)


if __name__ == "__main__":
    main()
