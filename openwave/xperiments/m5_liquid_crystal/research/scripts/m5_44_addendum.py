"""M5.44 addendum: two reads asked by the author's post of 2026-11-17 on the saved end states of M5.42.

A. The loop radius by the author's measure, reimplemented from its published definition (capstone 3.97 source,
   checks/v397/ring17/ring_tools.py). With eigenvalues l1 <= l2 <= l3 of M, gap g = l3 - l2, half-box L:

       nodes    the grid nodes with g < 0.15 and |x| < 0.93 L
       weight   w = 0.15 - g
       center   c = sum(w p) / sum(w)
       normal   nu = eigenvector of the smallest eigenvalue of sum(w q q^T) / sum(w),  q = p - c
       radius   R = sum(w rho) / sum(w),  rho = sqrt(|q|^2 - (q.nu)^2)

   It is a weighted cloud of nodes and not a line; the plaquette reader and the smallest-gap method of
   m5_44_loops.py are printed beside it from m5_44_loops.json.

B. The delta = 0.05 states: the degree of the long axis on the spheres r = 3 and r = 4 and the inertia inside
   r < 2 (both from the tracked m5_42_r7_reads.json), beside the loop of m5_44_loops.json.

Nothing is relaxed.  Usage:  python m5_44_addendum.py [--research-dir DIR]
"""
import argparse
import json
import os

import numpy as np

IDX = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])
THR, CUT = 0.15, 0.93
JE = {0.3: 1687.604, 0.05: 3108.3}


def cloud(m6, x):
    """The author's measure of the loop on one state."""
    w = np.linalg.eigvalsh(m6[..., IDX])
    g = w[..., 2] - w[..., 1]
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    r = np.sqrt((X ** 2).sum(-1))
    L = float(x[-1])
    inside = r < CUT * L
    m = (g < THR) & inside
    out = dict(nodes=int(m.sum()), gap_min=float(g[inside].min()))
    if m.sum() < 8:
        return out
    p, wt = X[m], THR - g[m]
    c = (p * wt[:, None]).sum(0) / wt.sum()
    q = p - c
    C = (q[:, :, None] * q[:, None, :] * wt[:, None, None]).sum(0) / wt.sum()
    ev, evec = np.linalg.eigh(C)
    nu = evec[:, 0]
    rho = np.sqrt(np.maximum((q ** 2).sum(1) - (q @ nu) ** 2, 0.0))
    R = float((rho * wt).sum() / wt.sum())
    out.update(center=c.tolist(), normal=nu.tolist(), radius=R, radius_over_half_box=R / L,
               radius_min=float(rho.min()), radius_max=float(rho.max()),
               off_plane=float(np.sqrt(((q @ nu) ** 2 * wt).sum() / wt.sum())))
    return out


def jlabel(delta, J):
    if J == 0:
        return "0"
    return "J_e/%g" % round(JE[delta] / J, 2)


def normal_name(v):
    a = [abs(t) for t in v]
    if a[2] > 0.99:
        return "z"
    if abs(a[0] - a[1]) < 0.03 and a[2] < 0.06 and v[0] * v[1] < 0:
        return "(1, −1, 0)/√2"
    return "(%.2f, %.2f, %.2f)" % tuple(v)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    research = here
    while not os.path.isdir(os.path.join(research, "data")) and research != "/":
        research = os.path.dirname(research)
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-dir", default=research)
    a = ap.parse_args()
    data = os.path.join(a.research_dir, "data")
    loops = json.load(open(os.path.join(data, "m5_44_loops.json")))["states"]
    reads = json.load(open(os.path.join(data, "m5_42_r7_reads.json")))["states"]

    # A. the J = 0 states with fixed faces, no band, delta = 0.3 and L1 > 0
    rows_a = []
    for s in loops:
        if not (s["J"] == 0 and s["delta"] == 0.3 and s["L1"] > 0 and not s["free_faces"] and not s["band"]):
            continue
        z = np.load(os.path.join(data, s["file"]), allow_pickle=False)
        c = cloud(z["m6"], z["x"])
        lp = s["loop"]
        rows_a.append(dict(chain=s["chain"], path=s["path"], file=s["file"], N=s["N"], half_box=s["half_box"],
                           spacing=s["spacing"], L1=s["L1"], stencil=s["stencil"], start=s["start"],
                           converged=s["converged"], plaquette_radius=lp["radius"],
                           plaquette_share=lp["radius_over_half_box"], smallest_gap_radius=lp["second_method"]["radius"],
                           smallest_gap_share=lp["second_method"]["radius"] / s["half_box"],
                           plaquette_normal=lp["normal"], cloud=c))

    # B. the delta = 0.05 states
    rows_b = []
    for s in loops:
        if s["delta"] != 0.05:
            continue
        r = next(t for t in reads if t["chain"] == s["chain"] and t["path"] == s["path"])
        sph = {round(q["radius"], 2): q for q in r["spheres"]}
        lp = s["loop"] or {}
        rows_b.append(dict(chain=s["chain"], path=s["path"], half_box=s["half_box"], L1=s["L1"], J=s["J"],
                           label=s["label"], converged=s["converged"], I=r["I"], I_inside_r2=r["inertia"]["inside_r2"],
                           degree_r3=abs(sph[3.0]["degree_of_long_axis"]), degree_r4=abs(sph[4.0]["degree_of_long_axis"]),
                           gap_r3=sph[3.0]["smallest_gap_of_the_two_larger_eigenvalues"],
                           gap_r4=sph[4.0]["smallest_gap_of_the_two_larger_eigenvalues"],
                           loops=s["loops"], fragments=s["fragments"], plaquettes=lp.get("count"),
                           loop_radius=lp.get("radius"), loop_share=lp.get("radius_over_half_box"),
                           loop_spread=lp.get("spread"), spacing=s["spacing"]))

    out = dict(author_measure=dict(threshold=THR, cut=CUT, states=rows_a), delta_005=rows_b)
    json.dump(out, open(os.path.join(data, "m5_44_addendum.json"), "w"), indent=1)

    print("**Table 8. The loop radius at J = 0 by three measures (δ = 0.3, fixed faces).**\n")
    print("| Chain of M5.42 | L₁ | Half-box (grid) | Functional | Start | Normal | Plaquettes: radius (share) "
          "| Smallest gap: radius (share) | The author's measure: radius (share) | Its nodes | Its off-plane rms |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for t in sorted(rows_a, key=lambda t: (-t["L1"], abs(t["plaquette_normal"][2]) < 0.9, t["half_box"], t["N"], t["stencil"])):
        c = t["cloud"]
        st = "seed" if t["start"] == "seed" else ("continued" if t["start"][0] == "self" else "the other stencil's J = 0 state")
        print("| `%s` %s | %g | %g (%d³) | %s | %s | %s | %.3f (%.3f) | %.3f (%.3f) | %.3f (%.3f) | %d | %.2f |" % (
            t["chain"], t["path"], t["L1"], t["half_box"], t["N"], t["stencil"], st, normal_name(t["plaquette_normal"]),
            t["plaquette_radius"], t["plaquette_share"], t["smallest_gap_radius"], t["smallest_gap_share"],
            c["radius"], c["radius_over_half_box"], c["nodes"], c["off_plane"]))

    print("\n**Table 9. The δ = 0.05 states (mirror-averaged stencil, spacing 0.375).**\n")
    print("| L₁ | Half-box | J | Label | Degree on r = 3 (gap) | Degree on r = 4 (gap) | I inside r < 2 of I (share) "
          "| Loops / fragments | Loop radius (share of the half-box) | Plaquettes |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for t in sorted(rows_b, key=lambda t: (-t["L1"], t["half_box"], t["J"])):
        print("| %g | %g | %s | %s | %.2f (%.3f) | %.2f (%.3f) | %.1f of %.1f (%.3f) | %d / %d | %.2f (%.3f) | %d |" % (
            t["L1"], t["half_box"], jlabel(0.05, t["J"]), t["label"] + (", CONVERGED" if t["converged"] else ", not converged"),
            t["degree_r3"], t["gap_r3"], t["degree_r4"], t["gap_r4"], t["I_inside_r2"], t["I"], t["I_inside_r2"] / t["I"],
            t["loops"], t["fragments"], t["loop_radius"], t["loop_share"], t["plaquettes"]))


if __name__ == "__main__":
    main()
