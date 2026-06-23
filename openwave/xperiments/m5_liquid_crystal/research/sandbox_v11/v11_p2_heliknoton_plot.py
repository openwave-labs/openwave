"""Compact figure for the P2 heliknoton build (run 3): the chiral+Frank machinery works,
but the biaxial M5 tensor does not host a clean stable helix / localized smooth heliknoton.
Reads the two checkpoint JSONs (no Taichi). Run: python v11_p2_heliknoton_plot.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "_checkpoints")

sweep = json.load(open(os.path.join(CKPT, "p2_heliknoton.json")))
calib = json.load(open(os.path.join(CKPT, "p2_helix_calib.json")))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel 1: the L-sweep -- L=0 combs out (knot dissolves), L>0 -> delocalized blue-phase texture
Ls = [r["L"] for r in sweep["L_sweep"]]
loc = [r["localization"] for r in sweep["L_sweep"]]
ecv = [r["excess_curv_over_helix"] for r in sweep["L_sweep"]]
a0 = ax[0]
b = a0.bar([str(L) for L in Ls], loc, color="#4477aa", label="localization (peak/mean)")
a0.axhline(4.0, ls="--", c="#cc6677", lw=1, label="localized-core threshold")
a0.set_xlabel("chiral strength L  (Frank K = L/2)")
a0.set_ylabel("excess localization (peak/mean)")
a0.set_title("knot localization vs chiral strength\n(L=0 control dissolves; L>0 stays delocalized)")
a02 = a0.twinx()
a02.plot(range(len(Ls)), ecv, "o-", c="#228833", label="excess curvature E")
a02.set_ylabel("excess curvature over helix", color="#228833")
a0.legend(loc="upper left", fontsize=8)

# Panel 2: the helix is never stationary -- relaxed Ecurv > 0 for every calibrated case
labels = [f"h{r['hand']:+.0f}\nLc={r['Lc']}" for r in calib["results"]]
ecr = [r["Ecurv_relaxed"] for r in calib["results"]]
dms = [r["max_dMsp"] for r in calib["results"]]
a1 = ax[1]
a1.bar(labels, ecr, color="#ee8866", label="relaxed Ecurv (>0 = 3D texture)")
a1.set_ylabel("relaxed curvature energy")
a1.set_title("no stable simple helix exists\n(every case relaxes to a 3D modulated texture)")
a12 = a1.twinx()
a12.plot(range(len(labels)), dms, "s--", c="#aa3377", label="max|dMsp| (director reorganization)")
a12.set_ylabel("max director change", color="#aa3377")
a1.legend(loc="upper right", fontsize=8)

fig.suptitle("M5.11 P2 build: chiral Lifshitz + Frank validated; biaxial M5 tensor "
             "drives blue-phase texture, not a localized heliknoton", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(HERE, "p2_heliknoton.png")
fig.savefig(out, dpi=110)
print(f"saved {out}")
