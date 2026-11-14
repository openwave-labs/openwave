"""M5.39: the figures of the Card C run, from the job files and the summary (headless).

    python3 m5_39_plots.py

Figures (research/plots/): m5_39_r2_fields.png (the seed and the endpoint of the 33^3 state on the
central slice), m5_39_r2_traces.png (energy and force residual against the iteration, per grid),
m5_39_r2_census.png (where the five starts of each grid end), m5_39_r2_profiles.png (the record's
eigenvalues against the radius), m5_39_r2_tail.png (the tail amplitude shell by shell).
Colors: one fixed hue per start on every figure (a validated five-slot categorical set), with a marker
shape per start as the second encoding; a single-hue ramp for magnitudes, two hues around a neutral
midpoint for the signed field q0.
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
import m5_39_cardc as C  # noqa: E402

DATA, PLOTS = HERE.parent / "data", HERE.parent / "plots"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e3e3", "#fcfcfb"
STARTS = {"linked": ("#2a78d6", "o"), "smooth": ("#eb6834", "s"), "iso": ("#1baf7a", "^"), "chain": ("#eda100", "D"), "reset": ("#e87ba4", "v"),
          "cont": ("#eda100", "D"), "contreset": ("#e87ba4", "v")}
NAMES = {"linked": "linked seed", "smooth": "smooth seed", "iso": "isotropic-center seed", "chain": "chain", "reset": "chain, frame reset",
         "cont": "box continuation", "contreset": "box continuation, frame reset"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f0a27f", "#b8431a"])
PRINTED = {33: 76.05003}
GRIDS = {33: 3.6, 45: 6.075, 55: 6.075, 65: 6.075, 77: 6.075, 73: 8.1, 91: 10.125}


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


def job(grid, start):
    p = DATA / f"m5_39_r2_g{grid}_{start}.json"
    return json.loads(p.read_text()) if p.exists() else None


def jobs_of(grid):
    out = []
    for s in ("linked", "smooth", "iso", "chain", "reset", "cont", "contreset"):
        j = job(grid, s)
        if j and j.get("label") != "NOT RUN":
            out.append((s, j))
    return out


def eigen_slice(state):
    n = state.shape[0]
    sl = state[:, :, n // 2]
    q0 = sl[..., 0] / np.linalg.norm(sl[..., :4], axis=-1)
    f = sl[..., 4:].reshape(sl.shape[:2] + (3, 3))
    lam = np.linalg.eigvalsh(np.swapaxes(f, -1, -2) @ f)
    return q0, lam[..., 0], lam[..., 2]


def fields(summary):
    sel = summary["grids"].get("33", {}).get("state_of_the_grid")
    if not sel:
        return
    start = sel["name"].split("_")[-1]
    seed = C.seed(33, 3.6, start if start in ("linked", "smooth", "iso") else "linked")[0]
    with np.load(DATA / f"m5_39_{sel['name']}.npz") as d:
        end = d["state"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), facecolor=SURFACE)
    ext = [-3.6, 3.6, -3.6, 3.6]
    for row, (label, st) in enumerate((("the seed (iteration 0)", seed), (f"the endpoint ({sel['name'].replace('r2_', '')}, gate met)", end))):
        q0, lo, hi = eigen_slice(st)
        for col, (arr, cmap, lim, name) in enumerate(((q0, DIV, (-1, 1), "q0"), (lo, SEQ, (0, 1), "smallest record eigenvalue"), (hi, SEQ, (0, 1), "largest record eigenvalue"))):
            ax = axes[row, col]
            im = ax.imshow(arr.T, origin="lower", extent=ext, cmap=cmap, vmin=lim[0], vmax=lim[1])
            if row == 0:
                ax.set_title(name, color=INK, fontsize=10, loc="left")
            if col == 0:
                ax.set_ylabel(label, color=INK, fontsize=9.5)
            ax.tick_params(colors=MUTED, labelsize=8)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(colors=MUTED, labelsize=8)
            cb.outline.set_edgecolor(GRID)
    fig.suptitle("Card C at 33³, central slice z = 0 (lengths in the card's unit)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOTS / "m5_39_r2_fields.png", dpi=150)
    plt.close(fig)


LEGEND = [("linked", "linked seed"), ("smooth", "smooth seed"), ("iso", "isotropic-center seed"), ("chain", "chain, or box continuation"),
          ("reset", "the same with the frame reset")]


def start_legend(fig, y=0.0):
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=STARTS[k][0], marker=STARTS[k][1], markersize=8, linewidth=1.8) for k, _ in LEGEND]
    handles += [Line2D([], [], color=MUTED, marker="o", markersize=8, linestyle="none"),
                Line2D([], [], color=MUTED, marker="o", markersize=8, linestyle="none", fillstyle="none", markeredgewidth=1.6)]
    fig.legend(handles, [n for _, n in LEGEND] + ["end state: gate met", "end state: gate not met at the caps"], loc="lower center", ncol=4, fontsize=9,
               frameon=False, labelcolor=MUTED, bbox_to_anchor=(0.5, y))


def traces(grids):
    grids = [g for g in grids if jobs_of(g)]
    if not grids:
        return
    fig, axes = plt.subplots(len(grids), 2, figsize=(11, 2.9 * len(grids) + 0.8), facecolor=SURFACE, squeeze=False)
    for row, grid in enumerate(grids):
        js = jobs_of(grid)
        floor = PRINTED.get(grid, min(j["energy"] for _, j in js if j["energy"] == j["energy"]) - 2e-6)
        for s, j in js:
            col, mk = STARTS[s]
            it = [r["iteration"] for r in j["history"]]
            axes[row, 0].plot(it, [max(r["energy"] - floor, 1e-7) for r in j["history"]], color=col, linewidth=1.8)
            axes[row, 1].plot(it, [r["residual"] for r in j["history"]], color=col, linewidth=1.2)
            axes[row, 0].plot(j["iterations"], max(j["energy"] - floor, 1e-7), marker=mk, markersize=8, color=col, markeredgecolor=col, markeredgewidth=1.6,
                              fillstyle="full" if j["label"] == "STATIONARY" else "none", linestyle="none")
        axes[row, 1].axhline(1e-3, color=INK, linewidth=1, linestyle="--")
        axes[row, 1].text(0.99, 1.15e-3, "the force gate", color=INK, fontsize=8.5, ha="right", transform=axes[row, 1].get_yaxis_transform())
        for ax in axes[row]:
            ax.set_yscale("log")
        style(axes[row, 0], f"{grid}³: energy above " + ("the printed 76.05003" if grid in PRINTED else "the lowest end energy of the grid"))
        style(axes[row, 1], f"{grid}³: force residual at the saved states")
        axes[row, 0].set_ylim(1e-6, 1e-1)
        axes[row, 1].set_ylim(2e-4, 1e0)
    for ax in axes[-1]:
        ax.set_xlabel("iteration", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.07 * 4 / len(grids) / 1.6, 1, 1))
    start_legend(fig)
    fig.savefig(PLOTS / "m5_39_r2_traces.png", dpi=150)
    plt.close(fig)


def census(grids):
    grids = [g for g in grids if jobs_of(g)]
    if not grids:
        return
    fig, axes = plt.subplots(1, len(grids), figsize=(2.3 * len(grids) + 1.2, 4.4), facecolor=SURFACE, sharey=True, squeeze=False)
    for ax, grid in zip(axes[0], grids):
        js = jobs_of(grid)
        floor = min(j["energy"] for _, j in js if j["energy"] == j["energy"])
        for k, (s, j) in enumerate(js):
            col, mk = STARTS[s]
            ax.plot(k, max(j["energy"] - floor, 3e-7), marker=mk, markersize=9, color=col, markeredgecolor=col, markeredgewidth=1.6,
                    fillstyle="full" if j["label"] == "STATIONARY" else "none", linestyle="none")
        ax.set_yscale("log")
        ax.set_xlim(-0.6, 4.6)
        ax.set_xticks([])
        style(ax, f"{grid}³")
    axes[0, 0].set_ylabel("end energy above the lowest of the grid\n(the lowest itself drawn at the bottom)", color=INK, fontsize=9.5)
    axes[0, 0].set_ylim(2e-7, 1e-1)
    fig.suptitle("Where the starts of each grid end", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    start_legend(fig)
    fig.savefig(PLOTS / "m5_39_r2_census.png", dpi=150)
    plt.close(fig)


def profiles(summary):
    names = [(g, v["state_of_the_grid"]["name"]) for g, v in summary["grids"].items() if "state_of_the_grid" in v]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names) + 0.6, 3.6), facecolor=SURFACE, sharey=True, squeeze=False)
    for ax, (grid, name) in zip(axes[0], names):
        with np.load(DATA / f"m5_39_{name}.npz") as d:
            st, co = d["state"], d["coords"]
        q0 = st[..., 0] / np.linalg.norm(st[..., :4], axis=-1)
        f = st[..., 4:].reshape(st.shape[:3] + (3, 3))
        lam = np.linalg.eigvalsh(np.swapaxes(f, -1, -2) @ f)
        r = np.linalg.norm(co, axis=-1)
        edges = np.arange(0, 3.01, 0.15)
        mid, rows, cq = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (r >= lo) & (r < hi)
            if sel.sum():
                mid.append(0.5 * (lo + hi))
                rows.append(lam[sel].mean(axis=0))
                cq.append(0.5 * (1 - q0[sel]).mean())
        rows = np.array(rows)
        for k, (lab, mk) in enumerate((("smallest", "v"), ("middle", "s"), ("largest", "^"))):
            ax.plot(mid, rows[:, k], color=["#86b6ef", "#3987e5", "#104281"][k], linewidth=1.8, marker=mk, markersize=4.5, label=f"{lab} eigenvalue")
        ax.plot(mid, cq, color=INK, linewidth=1.1, linestyle="--", label="c(q) = (1 − q0)/2")
        ax.plot(mid, 0.2 * np.array(cq), color=MUTED, linewidth=1.1, linestyle=":", label="δ c(q)")
        style(ax, f"{grid}³: {name.replace('r2_', '')}")
        ax.set_xlabel("radius (card length)", color=INK, fontsize=9.5)
    axes[0, 0].set_ylabel("record eigenvalues, shell means", color=INK, fontsize=10)
    axes[0, -1].legend(fontsize=8, frameon=False, labelcolor=MUTED, loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS / "m5_39_r2_profiles.png", dpi=150)
    plt.close(fig)


def tail():
    p = DATA / "m5_39_r2_reads.json"
    if not p.exists():
        return
    reads = json.loads(p.read_text())
    keep = {}
    for name, rd in reads.items():
        keep.setdefault(rd["grid"], (name, rd))
    if not keep:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=SURFACE, sharey=True)
    blues = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b", "#0b0b0b"]
    for k, grid in enumerate(sorted(keep)):
        name, rd = keep[grid]
        r = [0.5 * (t["r_lo"] + t["r_hi"]) for t in rd["tail"]]
        lab = f"{grid}³ (half-box {rd['half_box']:g})"
        axes[0].plot(r, [t["amplitude"] for t in rd["tail"]], color=blues[k % 7], linewidth=1.8, marker="osD^vP*"[k % 7], markersize=4.5, label=lab)
        axes[1].plot(r, [t["amplitude_against_the_continuum_dipole"] for t in rd["tail"]], color=blues[k % 7], linewidth=1.8, marker="osD^vP*"[k % 7], markersize=4.5, label=lab)
    style(axes[0], "tail amplitude against the lattice dipole of the same cube")
    style(axes[1], "the same field against the continuum, infinite-box dipole")
    axes[0].set_ylabel("A per shell (card units)", color=INK, fontsize=10)
    for ax in axes:
        ax.set_xlabel("radius (card length)", color=INK, fontsize=10)
        ax.set_ylim(0.8, 2.6)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=MUTED, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "m5_39_r2_tail.png", dpi=150)
    plt.close(fig)


def main():
    PLOTS.mkdir(exist_ok=True)
    summary = json.loads((DATA / "m5_39_r2_summary.json").read_text())
    fields(summary)
    traces([33, 55, 77, 73])
    census([33, 45, 55, 65, 77, 73, 91])
    profiles(summary)
    tail()
    print("figures written:", sorted(p.name for p in PLOTS.glob("m5_39_*.png")))


if __name__ == "__main__":
    main()
