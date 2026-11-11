"""M5.38: the panel plot, from the two JSONs of the run (no field is re-read except the planted control of panel 2).

Panel 1: the summed inertia density over the vacuum closed form against the radius, off the column, on three states.
Panel 2: the trace integral I(R) of the proton-analog against the vacuum-volume law and the planted compact record.
Panel 3: the electron-ID electron over half a clock cycle in the tilt plane: the degree of the unit director on a
         closed surface, the flux off the axis over 4 pi against cos(phi), and the instrument's lattice charge.

Run from this directory: python3 m5_38_panel.py   ->  ../plots/m5_38_panel.png
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
PLOTS = os.path.join(HERE, "..", "plots")

BLUE, ORANGE, GREEN, INK, MUTED, GRID = "#0072B2", "#D55E00", "#009E73", "#222222", "#6b6b6b", "#e3e3e3"

gate = json.load(open(os.path.join(DATA, "m5_38_inertia_gate.json")))
clock = json.load(open(os.path.join(DATA, "m5_38_clock_checks.json")))

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
for ax in axes:
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)

# ---- panel 1
ax = axes[0]
series = [("P-0.5_plane_sc6_n48_pinned_d0.3", "proton-analog, n 48", BLUE, "o"),
          ("P-1_plane_sc6_n48_pinned_d0.3", "neutron-analog, n 48", ORANGE, "s"),
          ("E+0.5_plane_sc6_n32_pinned_d0.3", "lepton reference, n 32", GREEN, "^")]
for tag, lab, col, mk in series:
    pr = [p for p in gate["states"][tag]["profile_summed_density_over_closed_form"] if p["off_column"] is not None]
    r = [0.5 * (p["r1"] + p["r2"]) for p in pr]
    ax.plot(r, [p["off_column"] for p in pr], color=col, marker=mk, markersize=6, linewidth=2, label=lab)
ax.axhline(1.0, color=INK, linewidth=1, linestyle="--"); ax.text(6.3, 1.015, "the vacuum closed form (the pinned shell reads 1.000)", fontsize=8.5, color=INK)
ax.axhline(0.0, color=MUTED, linewidth=1, linestyle=":"); ax.text(6.3, 0.02, "a record that vanishes outside its core", fontsize=8.5, color=MUTED)
ax.axvspan(18, 22, color="#f1f1f1", zorder=0); ax.text(18.15, 0.45, "the read\nwindow", fontsize=8.5, color=MUTED)
ax.set_xlim(6, 24); ax.set_ylim(-0.05, 1.12)
ax.set_xlabel("radius r (off the column, cylindrical radius above 6)", color=INK, fontsize=10)
ax.set_ylabel("summed inertia density over the vacuum value", color=INK, fontsize=10)
ax.set_title("R1: no exterior where the density falls", color=INK, fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 0.08))

# ---- panel 2
ax = axes[1]
st = gate["states"]["P-0.5_plane_sc6_n48_pinned_d0.3"]
R = np.array(st["radii"]); I = np.array(st["I_trace"]); cells = np.array(st["cells_in_ball"])
vac = st["closed_form_vacuum_sum"] * cells * st["h"] ** 3
spec = importlib.util.spec_from_file_location("g", os.path.join(HERE, "m5_38_inertia_gate.py"))
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
n, h, d = 32, 1.5, 0.3
X, Y, Z = g.grid(n, h); rr = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
Mv = np.broadcast_to(np.diag([1.0, d, 0.0]), (n, n, n, 3, 3)).copy()
w = np.clip((8.0 - rr) / 2.0, 0.0, 1.0); w = 0.5 - 0.5 * np.cos(np.pi * w)
Mc = w[..., None, None] * Mv + (1.0 - w[..., None, None]) * np.broadcast_to(np.eye(3) * (1.0 + d) / 3.0, Mv.shape)
ctl = g.read_state(Mc, h, d)
ax.plot(R, vac, color=INK, linewidth=1.2, linestyle="--", label="vacuum density times the enclosed volume")
ax.plot(R, I, color=BLUE, marker="o", markersize=6, linewidth=2, label="proton-analog, n 48")
ax.plot(ctl["radii"], ctl["I_trace"], color=MUTED, marker="D", markersize=5, linewidth=2, label="control: a record that vanishes outside r = 8")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([4, 6, 8, 12, 16, 22]); ax.set_xticklabels(["4", "6", "8", "12", "16", "22"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("ball radius R", color=INK, fontsize=10)
ax.set_ylabel("trace of the inertia integral I(R)", color=INK, fontsize=10)
ax.set_title("R1: the integral grows with the volume", color=INK, fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")

# ---- panel 3
ax = axes[2]
topo = clock["topology_added_on_the_audit"]["n48"]
tw = clock["rows"]["twist_n48_b0.00"]
f = np.array([r["phi_over_pi"] for r in topo])
fine = np.linspace(0, 1, 201)
ax.plot(fine, np.cos(fine * np.pi), color=INK, linewidth=1.2, linestyle="--", label="cos φ")
ax.plot(f, [r["degree_cube_k2"] for r in topo], color=GREEN, marker="^", markersize=6, linewidth=2, label="degree of the unit director on a closed surface")
ax.plot(f, [r["flux_off_axis"] / (4 * np.pi) for r in topo], color=BLUE, marker="o", markersize=6, linewidth=2, label="flux off the axis, over 4π")
ax.plot(f, [r["charge_raw_director"] / topo[0]["charge_raw_director"] for r in topo], color=ORANGE, marker="s", markersize=6, linewidth=2, label="lattice charge of the instrument, over its φ = 0 value")
ax.axhline(0.0, color=MUTED, linewidth=1)
ax.set_xlim(0, 1); ax.set_ylim(-1.15, 1.75)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "π/4", "π/2", "3π/4", "π"])
ax.set_xlabel("clock phase φ, tilt plane (the twist plane is flat at 1)", color=INK, fontsize=10)
ax.set_ylabel("value over its reference (n 48)", color=INK, fontsize=10)
ax.set_title("R2: the degree holds; the flux moves onto the axis line", color=INK, fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper right")

fig.tight_layout()
os.makedirs(PLOTS, exist_ok=True)
fig.savefig(os.path.join(PLOTS, "m5_38_panel.png"), dpi=150)
print("wrote ../plots/m5_38_panel.png")
