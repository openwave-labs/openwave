#!/usr/bin/env python3
"""M10.3, the maintainers' independent recomputation of the free two-load sector of the pin.

It imports nothing from the author's package and executes none of it. It reads one
data file, the carrier's seam list, as arrays, and optionally the author's receipt as
JSON on stdin.

    python3 -I m10_3_maintainer_recompute.py BOARD.npz [--receipt-stdin] [--output OUT.json]

BOARD.npz is `openwave/interval_echo/row_04/FEDERATION_LEVEL0.npz` of the author
package at commit 4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c (tag
interval-echo-m10.3-source). Its SHA-256 is checked before anything is computed from
it. With --receipt-stdin, stdin carries `row_04_receipt.json` from the same folder at
the same commit, also checked by SHA-256.

The equations (the generator the author's spec states for the pinned law, on two unit
loads at distinct ports a < b of the 240-port carrier, 28,680 configurations):

  H(kappa) = (1/2) sum_e (I - SWAP_e) - 2 kappa sum_e n_a n_b
      e runs over the 630 seams; SWAP_e exchanges the occupations of the seam's two
      ports; n_a n_b is 1 when both loads sit on the seam's ends. K denotes the first
      sum, P the projector on the 630 adjacent configurations, so H = K - 2 kappa P.
  psi(t) = exp(-i t H) psi0,   psi0 = the two loads on the launch seam (156, 157)
      propagated by a Chebyshev expansion: with H~ = (H - c) / r mapping the spectrum
      into [-1, 1],  exp(-i t H) = exp(-i c t) sum_k a_k (-i)^k J_k(r t) T_k(H~),
      a_0 = 1 and a_k = 2 for k >= 1; the phase exp(-i c t) is omitted (probabilities unchanged).
  P_home(t) = sum over the 30 seams inside the launch cell of |<s|psi(t)>|^2
  P_adj(t)  = sum over all 630 seams of |<s|psi(t)>|^2
  Var H = <psi0|H^2|psi0> - <psi0|H|psi0>^2,   Delta H = sqrt(Var H)
  window bound: the angle to the home subspace, arccos sqrt(P_home), moves at most
      Delta H per unit exposure, and no time in [0, 37] is further than 1/16 from the
      1/8 grid, so  P_home(t) >= cos^2( arccos sqrt(min_grid P_home - 1e-6) + Delta H / 16 )
  pair records: 4,096 categorical draws per setting from the 31-outcome law (the 30
      home seams and quiet) with numpy's default_rng([917421, arm, j]); a record is a
      draw on a home seam
  Hoeffding radius (the author's simultaneous one-sided form):
      r = sqrt(ln(8 / 0.01) / (2 * 4096));  common lower bound = min native count / 4096 - r

Armed before any comparison: the norm at exposure 37, energy conservation over
[0, 37], and agreement with scipy.sparse.linalg.expm_multiply at exposure 1.

Compared with the author's numbers (CLAIMED below): the carrier size and launch seam,
P_home at exposures 1, 7, 37 at kappa = 8, its minimum on the 1/8 grid over [0, 37],
Var H, P_home in cell 0 at exposure 150 for kappa = 8 and 0, the pair records, the
Hoeffding radius and the common lower bound, the 94% window floor; with the receipt,
its source, addresses, 186 outcome probabilities and record counts.

Reported, not compared: P_home and P_adj at exposure 37 for kappa in
(0, 0.5, 1, 2, 4, 8, 16, -8), and the longer times at kappa = 8 on a grid of step 4
up to 2,000.

Every comparison below can fail: section ARMS feeds each one a wrong value.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.special import jv

BOARD_SHA256 = "99870398e3d516f899c600e9bf25ba53c7f513142ba8facbe1cedd94b23f915b"
RECEIPT_SHA256 = "7f6e458fd0b7cf9b9e50fb15a5c4b77f944a2457a9c53cdcca88efdf4d4c3bdd"
PORTS, CELL_PORTS, SHOTS, SEED = 240, 12, 4096, 917421
KAPPA_TABLE = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, -8.0)
LONG_MARKS = (300, 1000, 2000)

# The author's numbers, as the task (M10.3) and its replay state them.
CLAIMED = {
    "ports": 240,
    "seams": 630,
    "launch": [156, 157],
    "home_t1_t7_t37_kappa8": [0.990982505, 0.989653268, 0.980271948],
    "grid_min_home_kappa8": 0.978883326,
    "grid_min_at": 34.0,
    "variance": 2.25,
    "cell0_home_t150": {"8.0": 0.983880525, "0.0": 0.149558605},
    "pair_records": {"8.0": [4048, 4052, 4028], "0.0": [1780, 1182, 824]},
    "hoeffding_radius": 0.02857,
    "common_lower_bound": 0.9548,
    "window_floor": 0.94,
}
NINE_DIGITS = 0.5e-9  # half a unit in the ninth decimal of a value rounded to nine digits
ROUTE = 1e-11  # allowance for this route's own arithmetic on top of the rounding

# ---- inputs ------------------------------------------------------------------------------------


def load_board(path):
    """The seam list of the pinned carrier, read as arrays after its SHA-256 is checked."""
    if not os.path.isfile(path):
        raise SystemExit(f"board file not found: {path}")
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if digest != BOARD_SHA256:
        raise SystemExit(f"board file is not the pinned one: sha256 {digest}")
    with np.load(path, allow_pickle=False) as data:
        edges = np.array(data["edges"], dtype=np.int64)
    return sorted({(int(min(a, b)), int(max(a, b))) for a, b in edges}), digest


def read_receipt():
    """The author's receipt from stdin, checked by SHA-256."""
    raw = sys.stdin.buffer.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != RECEIPT_SHA256:
        raise SystemExit(f"receipt on stdin is not the pinned one: sha256 {digest}")
    return json.loads(raw), digest


def structure(seams):
    """Ports, seams, degrees, and how the 20 cells are joined."""
    deg = np.zeros(PORTS, int)
    for a, b in seams:
        deg[a] += 1
        deg[b] += 1
    outside = [(a, b) for a, b in seams if a // CELL_PORTS != b // CELL_PORTS]
    ends = sorted(p for e in outside for p in e)
    inside = {
        tuple(
            sorted(
                (a % CELL_PORTS, b % CELL_PORTS)
                for a, b in seams
                if a // CELL_PORTS == b // CELL_PORTS == c
            )
        )
        for c in range(PORTS // CELL_PORTS)
    }
    values, counts = np.unique(deg, return_counts=True)
    return (
        {
            "ports": PORTS,
            "seams": len(seams),
            "seams_between_cells": len(outside),
            "degrees": {str(int(v)): int(c) for v, c in zip(values, counts)},
            "between_cells_form_a_matching": len(set(ends)) == 2 * len(outside),
            "between_cell_ends_per_cell": sorted(
                {sum(1 for p in ends if p // CELL_PORTS == c) for c in range(PORTS // CELL_PORTS)}
            ),
            "cells_identical": len(inside) == 1,
        },
        deg,
        set(ends),
    )


# ---- the generator and the propagation ---------------------------------------------------------


def build(seams):
    """Configurations {a < b}; K = (1/2) sum_e (I - SWAP_e); the adjacency indicator P."""
    conf = [(a, b) for a in range(PORTS) for b in range(a + 1, PORTS)]
    idx = {c: i for i, c in enumerate(conf)}
    rows, cols, vals = [], [], []
    diag = np.zeros(len(conf))
    adj = np.zeros(len(conf))
    for a, b in seams:
        adj[idx[(a, b)]] = 1.0
        for c in range(PORTS):
            if c in (a, b):
                continue
            i = idx[tuple(sorted((a, c)))]
            j = idx[tuple(sorted((b, c)))]
            diag[i] += 0.5
            diag[j] += 0.5
            rows += [i, j]
            cols += [j, i]
            vals += [-0.5, -0.5]
    kinetic = sparse.csr_matrix((vals, (rows, cols)), shape=(len(conf),) * 2)
    return idx, (kinetic + sparse.diags(diag)).tocsr(), adj


def generator(kinetic, adj, kappa):
    """H(kappa) = K - 2 kappa P."""
    return (kinetic - 2.0 * kappa * sparse.diags(adj)).tocsr()


def chebyshev(h, psi0, times, keep, full_at=()):
    """Amplitudes of exp(-i t H) psi0 on the configurations `keep`, for every t in times.

    Returns (amps[T, len(keep)], {t: full vector} for t in full_at). Global phase dropped.
    """
    d = h.diagonal()
    off = np.asarray(abs(h).sum(axis=1)).ravel() - np.abs(d)
    lo, hi = float((d - off).min()), float((d + off).max())
    centre, radius = (hi + lo) / 2, (hi - lo) / 2 * 1.0005
    scaled = ((h - centre * sparse.identity(h.shape[0])) / radius).tocsr()
    times = np.asarray(times, float)
    x = radius * times
    kmax = int(x.max() + 30 * max(x.max(), 1.0) ** (1 / 3) + 80)
    amps = np.zeros((len(times), len(keep)), complex)
    fulls = {t: np.zeros(h.shape[0], complex) for t in full_at}
    t_prev, t_cur = None, psi0.astype(float)
    chunk_k, chunk_v = [], []

    def flush():
        if not chunk_k:
            return
        k = np.array(chunk_k)
        coef = jv(k[:, None], x[None, :]) * ((-1j) ** (k % 4))[:, None]
        coef[k == 0] *= 0.5
        amps[:] += 2 * coef.T @ np.array(chunk_v)
        chunk_k.clear()
        chunk_v.clear()

    for k in range(kmax + 1):
        if k == 1:
            t_prev, t_cur = t_cur, scaled @ t_cur
        elif k > 1:
            t_prev, t_cur = t_cur, 2 * (scaled @ t_cur) - t_prev
        chunk_k.append(k)
        chunk_v.append(t_cur[keep].copy())
        for t in fulls:
            fulls[t] += jv(k, radius * t) * (-1j) ** (k % 4) * (1.0 if k == 0 else 2.0) * t_cur
        if len(chunk_k) == 512:
            flush()
    flush()
    return amps, fulls


def probabilities(amps, home_pos):
    """P_home and P_adj per time, and the per-seam probabilities."""
    p = np.abs(amps) ** 2
    return p[:, home_pos].sum(axis=1), p.sum(axis=1), p


# ---- the quantities ----------------------------------------------------------------------------


class Carrier:
    """The seam list, the generator's pieces, and the cells' home seams."""

    def __init__(self, seams):
        self.seams = seams
        self.idx, self.kinetic, self.adj = build(seams)
        self.keep = np.array([self.idx[s] for s in seams])  # the 630 adjacent configurations
        self.pos = {s: i for i, s in enumerate(seams)}

    def home(self, cell):
        hs = [s for s in self.seams if s[0] // CELL_PORTS == cell and s[1] // CELL_PORTS == cell]
        return hs, np.array([self.pos[s] for s in hs])

    def start(self, seam):
        v = np.zeros(len(self.idx))
        v[self.idx[seam]] = 1.0
        return v

    def h(self, kappa):
        return generator(self.kinetic, self.adj, kappa)


def arming(car, launch):
    """Norm at exposure 37, energy conservation, and a second algorithm at exposure 1."""
    h8, psi0 = car.h(8.0), car.start(launch)
    _, fulls = chebyshev(h8, psi0, [1.0, 37.0], car.keep, full_at=(1.0, 37.0))
    ref = expm_multiply(-1j * h8, psi0.astype(complex))
    e0 = float(psi0 @ (h8 @ psi0))
    f37 = fulls[37.0]
    return {
        "norm_t37_minus_1": float(np.vdot(f37, f37).real - 1.0),
        "energy_t37_minus_t0": float(np.vdot(f37, h8 @ f37).real - e0),
        "max_prob_difference_vs_expm_multiply_t1": float(
            np.abs(np.abs(fulls[1.0]) ** 2 - np.abs(ref) ** 2).max()
        ),
        "energy_t0": e0,
        "variance_t0": float((h8 @ psi0) @ (h8 @ psi0) - e0**2),
    }


def free_pair(car, launch, home_pos, kappa):
    """P_home and P_adj on the 1/8 grid over [0, 37] from the launch seam."""
    grid = np.arange(0, 297) / 8.0
    amps, _ = chebyshev(car.h(kappa), car.start(launch), grid, car.keep)
    p_home, p_adj, p = probabilities(amps, home_pos)
    return grid, p_home, p_adj, p


def cell0(car, kappa):
    """The older experiment: loads on (0, 1), P_home of cell 0 at exposure 150."""
    _, home_pos = car.home(0)
    amps, _ = chebyshev(car.h(kappa), car.start((0, 1)), [150.0], car.keep)
    p_home, p_adj, _ = probabilities(amps, home_pos)
    return float(p_home[0]), float(p_adj[0])


def outcome_law(p, home_pos, step):
    """The 31-outcome law at one grid step: the 30 home seams, then quiet."""
    ph = p[step, home_pos]
    return np.r_[ph, 1.0 - ph.sum()]


def pair_records(laws):
    """Seeded categorical draws from each law; a record is a draw on a home seam."""
    records = {}
    for arm, kappa in enumerate(("8.0", "0.0")):
        records[kappa] = []
        for j, law in enumerate(laws[kappa]):
            rng = np.random.default_rng([SEED, arm, j])
            draws = rng.choice(len(law), size=SHOTS, p=law / law.sum())
            records[kappa].append(int((draws < len(law) - 1).sum()))
    return records


def hoeffding(records):
    """The author's simultaneous one-sided radius and the common lower bound."""
    radius = math.sqrt(math.log(8 / 0.01) / (2 * SHOTS))
    return radius, min(records["8.0"]) / SHOTS - radius


def window_bound(grid_min, variance):
    """The angle argument over [0, 37] from the 1/8 grid."""
    angle = math.acos(math.sqrt(grid_min - 1e-6)) + math.sqrt(variance) * (1 / 8) / 2
    return math.cos(angle) ** 2


def receipt_comparison(rec, launch, home13, laws, records):
    """The author's receipt against this route: source, addresses, 186 values, counts."""
    worst, counts = 0.0, {}
    for arm, kappa in zip(rec["arms"], ("8.0", "0.0")):
        for j, reading in enumerate(arm["readings"]):
            mine = laws[kappa][j]
            worst = max(worst, float(np.abs(mine - np.array(reading["probabilities"])).max()))
            counts[f"{kappa}_t{reading['exposure']}"] = [
                int(sum(reading["counts"][:-1])),
                records[kappa][j],
            ]
    return {
        "source_matches": rec["source"] == list(launch),
        "addresses_match": rec["addresses"][:-1] == [list(s) for s in home13],
        "values_compared": sum(
            len(r["probabilities"]) for a in rec["arms"] for r in a["readings"]
        ),
        "max_probability_difference": worst,
        "record_counts_receipt_vs_recomputed": counts,
    }


def coupling_table(car, launch, home_pos):
    """Reported: P_home and P_adj at exposure 37 for each coupling of the table."""
    out = {}
    for kappa in KAPPA_TABLE:
        amps, _ = chebyshev(car.h(kappa), car.start(launch), [37.0], car.keep)
        p_home, p_adj, _ = probabilities(amps, home_pos)
        out[str(kappa)] = {"home_t37": float(p_home[0]), "adjacent_t37": float(p_adj[0])}
    return out


def long_times(car, launch, home_pos):
    """Reported: kappa = 8 on a grid of step 4 up to 2,000; marks lie on the grid."""
    grid = np.arange(0, 2001, 4.0)
    amps, _ = chebyshev(car.h(8.0), car.start(launch), grid, car.keep)
    p_home, p_adj, _ = probabilities(amps, home_pos)
    at = {t: int(np.flatnonzero(grid == t)[0]) for t in LONG_MARKS}
    return {
        "grid_step": 4.0,
        "marks": list(LONG_MARKS),
        "home": [float(p_home[at[t]]) for t in LONG_MARKS],
        "adjacent": [float(p_adj[at[t]]) for t in LONG_MARKS],
        "adjacent_min_over_grid": float(p_adj.min()),
    }


# ---- comparisons -------------------------------------------------------------------------------


def arming_checks(a):
    """The route is armed when these hold; they precede every comparison."""
    return {
        "arming: norm at exposure 37 within 1e-10": abs(a["norm_t37_minus_1"]) < 1e-10,
        "arming: energy conserved to 1e-10 over [0, 37]": abs(a["energy_t37_minus_t0"]) < 1e-10,
        "arming: agrees with expm_multiply at exposure 1 to 1e-12": a[
            "max_prob_difference_vs_expm_multiply_t1"
        ]
        < 1e-12,
    }


def verdicts(r):
    """One named comparison per claim. Each returns True only on agreement."""
    c, s = r["claims"], r["structure"]
    checks = arming_checks(r["arming"])
    checks |= {
        "carrier: 240 ports, 630 seams": (s["ports"], s["seams"])
        == (CLAIMED["ports"], CLAIMED["seams"]),
        "launch seam (156, 157)": r["launch"] == CLAIMED["launch"],
        "home-cell probability at exposures 1, 7, 37, kappa 8": all(
            abs(x - y) <= NINE_DIGITS + ROUTE
            for x, y in zip(c["home_t1_t7_t37_kappa8"], CLAIMED["home_t1_t7_t37_kappa8"])
        ),
        "grid minimum over [0, 37] and where": abs(
            c["grid_min_home_kappa8"] - CLAIMED["grid_min_home_kappa8"]
        )
        <= NINE_DIGITS + ROUTE
        and c["grid_min_at"] == CLAIMED["grid_min_at"],
        "energy variance of the launch state": abs(c["variance"] - CLAIMED["variance"]) < 1e-12,
        "cell 0 at exposure 150, kappa 8 and 0": all(
            abs(c["cell0_home_t150"][k] - CLAIMED["cell0_home_t150"][k]) <= NINE_DIGITS + ROUTE
            for k in ("8.0", "0.0")
        ),
        "pair records of 4,096, kappa 8": c["pair_records"]["8.0"]
        == CLAIMED["pair_records"]["8.0"],
        "pair records of 4,096, kappa 0": c["pair_records"]["0.0"]
        == CLAIMED["pair_records"]["0.0"],
        "Hoeffding radius about 0.02857": abs(c["hoeffding_radius"] - CLAIMED["hoeffding_radius"])
        < 5e-6,
        "common lower bound above 95.48%": c["common_lower_bound"] > CLAIMED["common_lower_bound"],
        "window bound at or above the 94% floor": c["window_bound"] >= CLAIMED["window_floor"],
    }
    if r["receipt"] is not None:
        q = r["receipt"]
        checks.update(
            {
                "receipt: source is the launch seam": q["source_matches"],
                "receipt: addresses are the 30 home seams": q["addresses_match"],
                "receipt: 186 outcome probabilities to nine digits": q["values_compared"] == 186
                and q["max_probability_difference"] <= NINE_DIGITS + ROUTE,
                "receipt: record counts equal the recomputed draws": all(
                    x == y for x, y in q["record_counts_receipt_vs_recomputed"].values()
                ),
            }
        )
    return checks


def arms(r):
    """Feed each comparison a wrong value. Every one must turn false."""
    w = copy.deepcopy(r)
    w["arming"].update(
        norm_t37_minus_1=1e-9,
        energy_t37_minus_t0=1e-9,
        max_prob_difference_vs_expm_multiply_t1=1e-11,
    )
    w["structure"]["seams"] = 629
    w["launch"] = [156, 158]
    c = w["claims"]
    c["home_t1_t7_t37_kappa8"][2] += 2e-9
    c["grid_min_at"] = 34.125
    c["variance"] += 1e-9
    c["cell0_home_t150"]["0.0"] += 2e-9
    c["pair_records"]["8.0"][2] += 1
    c["pair_records"]["0.0"][0] -= 1
    c["hoeffding_radius"] += 2e-5
    c["common_lower_bound"] = 0.9547
    c["window_bound"] = 0.939
    if w["receipt"] is not None:
        q = w["receipt"]
        q.update(source_matches=False, addresses_match=False, max_probability_difference=1e-9)
        key = next(iter(q["record_counts_receipt_vs_recomputed"]))
        q["record_counts_receipt_vs_recomputed"][key][1] += 1
    return [name for name, passed in verdicts(w).items() if passed]


# ---- run ---------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("board", help="FEDERATION_LEVEL0.npz of the pinned author package")
    parser.add_argument("--receipt-stdin", action="store_true", help="read row_04_receipt.json")
    parser.add_argument("--output", help="write the result as JSON")
    args = parser.parse_args()

    seams, board_sha = load_board(args.board)
    rec, receipt_sha = read_receipt() if args.receipt_stdin else (None, None)
    struct, deg, ends = structure(seams)
    car = Carrier(seams)
    home13, home13_pos = car.home(13)
    launch = next(s for s in home13 if s[0] in ends or s[1] in ends)

    armed = arming(car, launch)
    r = {"arming": armed, "structure": struct, "launch": list(launch), "receipt": None}
    failed = [name for name, ok in arming_checks(armed).items() if not ok]
    if failed:
        print("FAIL  " + "\nFAIL  ".join(failed) + "\nnot armed: no comparison is made")
        return 1

    grids, laws = {}, {}
    for kappa in ("8.0", "0.0"):
        grid, p_home, p_adj, p = free_pair(car, launch, home13_pos, float(kappa))
        steps = [int(np.flatnonzero(grid == t)[0]) for t in (1.0, 7.0, 37.0)]
        grids[kappa] = {
            "home_t1_t7_t37": [float(p_home[i]) for i in steps],
            "adjacent_t1_t7_t37": [float(p_adj[i]) for i in steps],
            "grid_min_home": float(p_home.min()),
            "grid_min_at": float(grid[p_home.argmin()]),
        }
        laws[kappa] = [outcome_law(p, home13_pos, i) for i in steps]
    records = pair_records(laws)
    radius, lower = hoeffding(records)
    cell0_values = {k: cell0(car, float(k)) for k in ("8.0", "0.0")}
    r["claims"] = {
        "variance": armed["variance_t0"],
        "home_t1_t7_t37_kappa8": grids["8.0"]["home_t1_t7_t37"],
        "grid_min_home_kappa8": grids["8.0"]["grid_min_home"],
        "grid_min_at": grids["8.0"]["grid_min_at"],
        "cell0_home_t150": {k: v[0] for k, v in cell0_values.items()},
        "pair_records": records,
        "hoeffding_radius": radius,
        "common_lower_bound": lower,
        "window_bound": window_bound(grids["8.0"]["grid_min_home"], armed["variance_t0"]),
    }
    if rec is not None:
        r["receipt"] = receipt_comparison(rec, launch, home13, laws, records)
    r["reported"] = {
        "launch_degrees": [int(deg[launch[0]]), int(deg[launch[1]])],
        "free_pair_grid": grids,
        "cell0_adjacent_t150": {k: v[1] for k, v in cell0_values.items()},
        "coupling_table_t37": coupling_table(car, launch, home13_pos),
        "long_times_kappa8": long_times(car, launch, home13_pos),
    }

    checks = verdicts(r)
    survivors = arms(r)
    for name, passed in checks.items():
        print(("PASS  " if passed else "FAIL  ") + name)
    print(
        f"comparisons: {len(checks)}; arms: {len(checks) - len(survivors)} of {len(checks)}"
        " comparisons turn false on a wrong value"
    )
    result = {
        "task": "M10.3",
        "inputs": {"board_sha256": board_sha, "receipt_sha256": receipt_sha},
        "claimed": CLAIMED,
        **r,
        "checks": checks,
        "arms_surviving": survivors,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, sort_keys=True)
            handle.write("\n")
    return 0 if all(checks.values()) and not survivors else 1


if __name__ == "__main__":
    sys.exit(main())
