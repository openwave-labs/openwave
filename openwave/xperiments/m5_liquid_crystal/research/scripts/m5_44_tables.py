"""M5.44: the tables of the findings, written from m5_44_loops.json.  Usage: python m5_44_tables.py [JSON]"""
import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "m5_44_loops.json"))
S = d["states"]
JE = {0.3: 1687.604, 0.05: 3108.3}
JN = {"0": 0, "J_e/40": 1, "J_e/20": 2, "J_e/10": 3, "J_e/5": 4, "J_e/2.5": 5, "J_e": 6}


def jl(s):
    if s["J"] == 0:
        return "0"
    r = JE[s["delta"]] / s["J"]
    return "J_e" if abs(r - 1) < 1e-3 else "J_e/%g" % round(r, 2)


def get(chain, path):
    return next(s for s in S if s["chain"] == chain and s["path"] == path)


def nrm(v):
    a = [abs(x) for x in v]
    if a[2] > 0.99:
        return "z"
    if abs(a[0] - a[1]) < 0.02 and a[2] < 0.05 and v[0] * v[1] < 0:
        return "(1, −1, 0)/√2"
    return "(%.2f, %.2f, %.2f)" % tuple(v)


def lab(s):
    return s["label"] + (", CONVERGED" if s["converged"] else ", not converged")


def cells(s):
    l = s["loop"]
    return "%d / %d | %s | %.3f | %.3f | %.2f | %.2f | %.3f | %.3f" % (
        s["loops"], s["fragments"], nrm(l["normal"]), l["radius"], l["radius_over_half_box"], l["spread"],
        l["off_plane"], l["second_method"]["radius"], l["gap_max"])


HEAD = "Loops / fragments | Normal | Radius | Radius over half-box | Spread | Rms off the plane | Second method | Largest gap on the loop |"
SEP = "| --- " * 9 + "|"

print("**Table 1. The loop against the box at J = 0 (mirror-averaged stencil, spacing 0.375, the symmetric sector).**\n")
print("| L₁ | Half-box (grid) | Half-box minus radius | " + HEAD + "\n| --- | --- " + SEP)
for L in ("0.3", "0.1"):
    for hb in (6, 9, 12, 18):
        s = get("r4_hb%d_L%s" % (hb, L), "j0")
        print("| %s | %d (%d³) | %.2f | %s |" % (L, hb, s["N"], hb - s["loop"]["radius"], cells(s)))

print("\n**Table 2. The two sectors (L₁ = 0.3).**\n")
print("| Half-box (grid) | Functional | Start | J | " + HEAD + "\n| --- | --- | --- " + SEP)
rows = [("r4_hb6_L0.3", "j0", "seed"), ("r3_g49_L0.3", "j0", "seed"), ("r3_g65_L0.3", "j0", "seed"),
        ("g3x_fwd_g33", "j0a", "the averaged J = 0 state"), ("g3x_fwd_g49", "j0a", "the averaged J = 0 state"),
        ("r2_g33", "j0", "seed, frame as deposited"), ("g3_g33", "j0", "seed"), ("g3_g49", "j0", "seed"), ("g3_g65", "j0", "seed"),
        ("g3x_avg_g33", "j0f", "the forward J = 0 state"), ("g3x_avg_g49", "j0f", "the forward J = 0 state"),
        ("r4_hb9_L0.3", "j0", "seed"), ("r4_hb9_L0.3", "j10", "the state before"),
        ("g3b_fwd_hb9", "j0", "seed"), ("g3b_avg_hb9", "j0b", "the forward J = 0 state"), ("g3b_avg_hb9", "j10b", "the state before")]
for c, p, st in rows:
    s = get(c, p)
    print("| %g (%d³) | %s | %s | %s | %s |" % (s["half_box"], s["N"], "averaged" if s["stencil"] != "forward" else "forward", st, jl(s), cells(s)))

print("\n**Table 3. The spacing on half-box 6 (mirror-averaged stencil, fixed chart).**\n")
print("| L₁ | J | Grid (spacing) | Label | " + HEAD + "\n| --- | --- | --- " + SEP)
for L in ("0.3", "0.1"):
    for p in ("j0", "j10"):
        for g in (33, 49, 65):
            s = get("r3_g%d_L%s" % (g, L), p)
            print("| %s | %s | %d³ (%g) | %s | %s |" % (L, jl(s), g, s["spacing"], lab(s), cells(s)))

print("\n**Table 4. The loop against J on the smooth branch: radius (radius over the half-box); SMOOTH and CONVERGED states only.**\n")
for L in ("0.3", "0.1"):
    print("L₁ = %s:\n\n| J | Half-box 6 | Half-box 9 | Half-box 12 | Half-box 18 |\n| --- | --- | --- | --- | --- |" % L)
    for j in JN:
        row = []
        for hb in (6, 9, 12, 18):
            c = [s for s in S if s["chain"] in ("r4_hb%d_L%s" % (hb, L), "r4x_hb%d_L%s" % (hb, L)) and jl(s) == j]
            c = [s for s in c if s["label"] == "SMOOTH" and s["converged"]]
            row.append("%.3f (%.3f)" % (c[0]["loop"]["radius"], c[0]["loop"]["radius_over_half_box"]) if c else "")
        if any(row):
            print("| %s | %s |" % (j, " | ".join(row)))
    print()

print("**Table 5. The author's box (33³, half-box 6, L₁ = 0.3, forward stencil, frame as deposited).**\n")
print("| Path | J | Label | " + HEAD + "\n| --- | --- " + SEP)
for p, st in (("j0", "from the seed"), ("j10", "from the state before"), ("j1", "from the state before"), ("j10x", "from the J_e state"), ("j0x", "from the J_e state")):
    s = get("r2_g33", p)
    print("| %s | %s | %s | %s |" % (st, jl(s), lab(s), cells(s)))

print("\n**Table 6. The one-cell states of M5.42 Table 6 and the loop: distance from the largest node to the nearest pierced plaquette, in length units (and in spacings).**\n")
print("| L₁ | J | Grid (spacing) | Largest node | Its distance from the z axis | Loops / fragments of the state | To the state's own loop | To the loop of the J = 0 state | To the loop of the smooth J_e/10 state |\n" + "| --- " * 9 + "|")
for c, p in (("r3_g33_L0.3", "j10x"), ("r3_g49_L0.3", "j10x"), ("r3_g65_L0.3", "j10x"), ("r3_g33_L0.3", "j1"), ("r3_g49_L0.3", "j1"), ("r3_g65_L0.3", "j1x"),
             ("r3_g33_L0.1", "j10"), ("r3_g49_L0.1", "j10x"), ("r3_g65_L0.1", "j10x"), ("r3_g33_L0.1", "j1"), ("r3_g49_L0.1", "j1x"), ("r3_g65_L0.1", "j1x")):
    s = get(c, p)
    h = s["spacing"]
    pos = s["largest_position"]
    f = lambda v: "%.2f (%.1f)" % (v, v / h) if v is not None else "no such state"
    print("| %g | %s | %d³ (%g) | (%g, %g, %g) | %.2f | %d / %d | %s | %s | %s |" % (
        s["L1"], jl(s), s["N"], h, *pos, (pos[0] ** 2 + pos[1] ** 2) ** 0.5, s["loops"], s["fragments"],
        f(s.get("node_to_own_loop")), f(s.get("node_to_j0_loop")), f(s.get("node_to_j10_loop"))))

print("\n**Table 7. The other states at J = 0: the largest loop as found.**\n")
print("| State | Half-box (grid, spacing) | Label | Loops / fragments | Radius | Radius over half-box | Rms off the plane |\n" + "| --- " * 7 + "|")
for c, name in (("r3_g33_L0.0", "L₁ = 0"), ("r3_g49_L0.0", "L₁ = 0"), ("r3_g65_L0.0", "L₁ = 0"), ("r4b_hb9", "L₁ = 0"), ("r4b_hb9_h025", "L₁ = 0"), ("r4b_hb12", "L₁ = 0"),
                ("r6_hb6_L0.3", "δ = 0.05, L₁ = 0.3"), ("r6_hb12_L0.3", "δ = 0.05, L₁ = 0.3"), ("r6_hb6_L0.0", "δ = 0.05, L₁ = 0"), ("r6_hb12_L0.0", "δ = 0.05, L₁ = 0"),
                ("r5_g33_L0.3", "free faces, L₁ = 0.3"), ("r5_g49_L0.3", "free faces, L₁ = 0.3"), ("r5_g33_L0.0", "free faces, L₁ = 0")):
    s = get(c, "j0")
    l = s["loop"]
    print("| %s | %g (%d³, %g) | %s | %d / %d | %s | %s | %s |" % (name, s["half_box"], s["N"], s["spacing"], lab(s), s["loops"], s["fragments"],
          "%.2f" % l["radius"] if l else "no pierced plaquette", "%.3f" % l["radius_over_half_box"] if l else "", "%.2f" % l["off_plane"] if l else ""))
