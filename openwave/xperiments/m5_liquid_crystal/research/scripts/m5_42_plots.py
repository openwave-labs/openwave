"""M5.42: the four figures of the findings, from m5_42_summary.json and the saved end fields.

  m5_42_r4_ratio_against_box.png   2 J Omega / E on the smooth branch against the half-box, one line per J
  m5_42_r3_one_cell_spacing.png    the one-cell states against the spacing: energy above the J = 0 state and
                                   the inertia of the spike, with the two recorded exponents as guide lines
  m5_42_r2_minor_gap.png           the minor gap lam_mid - lam_min on the plane z = 0 of the author's box at
                                   J = 0, J_e/10 and J_e
  m5_42_r2_inertia_slices.png      the inertia density on the plane of its largest node, for the smooth and
                                   the one-cell state of the author's box at J_e/10
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_42_eq42 as Q  # noqa: E402
from m5_42_relax import chains  # noqa: E402

INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#c2417f", "#8b6fd6", "#b8860b"]
STEP_NAME = dict(j40="J_e/40", j20="J_e/20", j10="J_e/10", j5="J_e/5", j2p5="J_e/2.5", j1="J_e")
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def fig_ratio(S, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, L1 in zip(axes, ("0.3", "0.1")):
        rows = S["smooth_branch_against_the_box"][L1]
        for c, (st, name) in zip(COLORS, STEP_NAME.items()):
            xs, ys, xo, yo = [], [], [], []
            for hb, r in rows.items():
                s = r["states"].get(st)
                if not s:
                    continue
                if s["label"] == "SMOOTH" and s["converged"]:
                    xs.append(float(hb))
                    ys.append(s["two_J_Omega_over_E"])
                else:
                    xo.append(float(hb))
                    yo.append(s["two_J_Omega_over_E"])
            if xs:
                ax.plot(xs, ys, "-o", color=c, lw=2, ms=6, label=name)
                ax.annotate(name, (xs[-1], ys[-1]), textcoords="offset points", xytext=(7, 0), color=INK, fontsize=8, va="center")
            if xo:
                ax.plot(xo, yo, "x", color=c, ms=8, mew=2)
        ax.axhline(1.0, color=INK, lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_xticks([6, 9, 12, 18])
        ax.set_xlim(5, 21)
        ax.set_xlabel("half-box")
        ax.set_title(f"L₁ = {L1}", color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("2JΩ/E")
    axes[0].plot([], [], "x", color=MUTED, ms=8, mew=2, label="not SMOOTH (one-cell or mixed)")
    axes[0].plot([], [], "--", color=INK, lw=1, label="ħΩ = E")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("The ratio on the smooth branch is set by the box (mirror-averaged stencil, spacing 0.375)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.savefig(out / "m5_42_r4_ratio_against_box.png")
    plt.close(fig)


def fig_onecell(S, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    O = S["one_cell_states_against_the_spacing"]
    for ax, key, ylabel, guides in ((axes[0], "energy_above_J0", "energy above the J = 0 state", ((1.0, "h¹ (author, L₁ = 0)"), (1 / 3, "h^(1/3) (plan, L₁ > 0)"))),
                                    (axes[1], "spike_inertia", "inertia of the spike, I − I(J = 0)", ((-1.0, "h⁻¹ (author)"), (-1 / 3, "h^(−1/3) (plan, L₁ > 0)")))):
        k = 0
        ally = []
        for L1 in ("0.3", "0.1", "0.0"):
            for jn, ls, mk in (("Je_over_10", "-", "o"), ("Je", "--", "s")):
                rows = [r for r in O[L1][jn]["rows"] if r.get("state")]
                if not rows:
                    continue
                x = [r["spacing"] for r in rows]
                y = [r[key] for r in rows]
                conv = [r["state"]["converged"] for r in rows]
                ally += [v for v in y if v > 0]
                ax.plot(x, y, ls, color=COLORS[k // 2], lw=2)
                for xi, yi, ci in zip(x, y, conv):
                    ax.plot(xi, yi, mk, color=COLORS[k // 2], ms=7, mfc=COLORS[k // 2] if ci else "white", mew=1.8)
                ax.annotate(f"L₁ = {L1}, {'J_e/10' if jn == 'Je_over_10' else 'J_e'}", (x[0], y[0]), textcoords="offset points", xytext=(6, 0), fontsize=8, color=INK, va="center")
            k += 2
        xs = np.array([0.1875, 0.375])
        for p, name in guides:
            ref = np.exp(np.mean(np.log(ally))) * (xs / xs.mean()) ** p if ally else xs * 0 + 1
            ax.plot(xs, ref, ":", color=MUTED, lw=1.5)
            ax.annotate(name, (xs[0], ref[0]), textcoords="offset points", xytext=(-4, 6), fontsize=8, color=MUTED, ha="left")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.1875, 0.25, 0.375])
        ax.set_xticklabels(["0.1875", "0.25", "0.375"])
        ax.minorticks_off()
        ax.set_xlim(0.17, 0.52)
        ax.set_xlabel("spacing")
        ax.set_ylabel(ylabel)
    axes[0].plot([], [], "-o", color=MUTED, label="J_e/10")
    axes[0].plot([], [], "--s", color=MUTED, label="J_e")
    axes[0].plot([], [], "o", color=MUTED, mfc="white", mew=1.8, label="open marker: not CONVERGED")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("The one-cell states against the spacing (half-box 6, mirror-averaged stencil)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.savefig(out / "m5_42_r3_one_cell_spacing.png")
    plt.close(fig)


def load(S, data, path):
    s = S["author_box"][path]
    f = data / f"m5_42_r2_g33_{path}_end.npz"
    cands = sorted(data.glob(f"m5_42_r2_g33_{path}_*.npz"))
    z = np.load(f if f.exists() else cands[-1])
    return s, z["m6"], z["x"], float(z["J"])


def fig_gap(S, data, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.subplots_adjust(wspace=0.38)
    for ax, path, name in zip(axes, ("j0", "j10", "j1"), ("J = 0", "J_e/10", "J_e")):
        s, m6, x, _ = load(S, data, path)
        lam = np.linalg.eigvalsh(Q.from6(m6))
        c = len(x) // 2
        gap = lam[:, :, c, 1] - lam[:, :, c, 0]
        im = ax.imshow(gap.T, origin="lower", extent=[x[0], x[-1], x[0], x[-1]], cmap="viridis", vmin=0, vmax=max(0.6, float(gap.max())))
        ax.set_title(f"{name}: {s['label']}, largest gap {gap.max():.2f}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("x")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("y")
    fig.suptitle("The minor gap λ_mid − λ_min on the plane z = 0 of the author's box (forward stencil; the frame has 0.3)", color=INK, fontsize=11, x=0.01, y=1.0, ha="left")
    fig.savefig(out / "m5_42_r2_minor_gap.png")
    plt.close(fig)


def fig_slices(S, data, out):
    C = chains()
    m = C["r2_g33"]["model"]
    mod = Q.Model(m["N"], m["half_box"], m["delta"], m["L1"], stencil=m["stencil"], fixed_chart=m["fixed_chart"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, path, name in zip(axes, ("j10", "j10x"), ("the smooth state", "the one-cell state")):
        s, m6, x, J = load(S, data, path)
        mod.evaluate(m6, J, need_grad=False)
        k = int(np.argmin(np.abs(x - S["author_box"]["j10x"]["largest_position"][2])))
        dens = np.maximum(mod.idn[:, :, k], 1e-6)
        im = ax.imshow(np.log10(dens).T, origin="lower", extent=[x[0], x[-1], x[0], x[-1]], cmap="magma", vmin=-2, vmax=5)
        ax.set_title(f"{name}: E = {s['E']:.3f}, I = {s['I']:.1f}, 2JΩ/E = {s['two_J_Omega_over_E']:.3f}\nlargest node holds {100 * s['largest_share']:.2g} % of I", color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("x")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="log₁₀ inertia density")
    axes[0].set_ylabel("y")
    fig.suptitle(f"Inertia density on the plane z = {x[k]:.3f} of the author's box at J_e/10 (forward stencil)", color=INK, fontsize=11, x=0.01, y=1.04, ha="left")
    fig.savefig(out / "m5_42_r2_inertia_slices.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--summary", type=Path, default=HERE.parent / "data" / "m5_42_summary.json")
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "plots")
    a = ap.parse_args()
    S = json.loads(a.summary.read_text())
    fig_ratio(S, a.out_dir)
    fig_onecell(S, a.out_dir)
    fig_gap(S, a.data_dir, a.out_dir)
    fig_slices(S, a.data_dir, a.out_dir)
    print("four figures ->", a.out_dir)


if __name__ == "__main__":
    main()
