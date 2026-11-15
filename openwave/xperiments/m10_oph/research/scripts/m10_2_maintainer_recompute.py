#!/usr/bin/env python3
"""M10.2, the maintainers' independent recomputation of the task's headline numbers.

It imports nothing from the author's package and executes none of it. Part A needs
no input. Part B reads one data file, the candidate's graph definition, as JSON.

    python3 -I m10_2_maintainer_recompute.py                      # part A only
    python3 -I m10_2_maintainer_recompute.py GRAPH.json [--output OUT.json]

GRAPH.json is `evidence/openwave/shared/localized_rest_join/evidence/graph.json`
of the author package at commit c330c6e7f23506b5f87c339cd73906928c1a7cd9. Its
SHA-256 is checked before anything is computed from it.

The equations (the author's, from the M10.2 method note):

  endpoint rule on a seam (a, b) with coin c in {0, 1}, loads q:
      d = q_a - q_b,   j = (d - (2c - 1)(|d| mod 2)) / 2,   q' = q - j e_a + j e_b
  mean one-event map and the energy convention, on a graph with |E| seams:
      T = I - L / (2|E|),   H = -2|E| log T,   so a mode L f = x f has E = -2|E| log(1 - x / 2|E|)
  first-current moments on one cell, over all 30 seams and both coins:
      Delta = j (v_b - v_a),   m = E[Delta],   C = E[(Delta - m)(Delta - m)^T]

PART A, one icosahedral cell (12 ports, 30 seams), ports at (0, +-1, +-phi) and
cyclic, source = +1 on a port v and -1 on its antipode:
  A1  m = (phi - 3)/15 v,  C_par = (3 - phi)/9,  C_perp = (phi + 2)/15,  C uniaxial about v
  A2  the half turn diag(-1, -1, 1) permutes the ports and sends four to their antipodes
  A3  expected events until a +1/-1 pair on any two ports cancels: 66, 84 or 90
  A4  the structure of A1 (mean along v, uniaxial C, half-turn behaviour) under two other
      rules: one that respects the cell's symmetry, one that does not

PART B, the full graph (240 ports, 630 seams):
  B1  20 cells joined by 30 seams, three to a cell, every cell seeing the others at the
      distances of a dodecahedron's vertices (1, 3, 6, 6, 3, 1)
  B2  L f = 6 f exactly for the two compact rays f and g, neither on an outside seam
  B3  E6, E12, the pair energy at the 511 keV anchor, and its excess over 2 x 511
  B4  how many modes of L lie below 6 and at 6 (reported, not compared with a claim)

Every comparison below can fail: section ARMS feeds each one a wrong value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys

import numpy as np
import sympy as sp

GRAPH_SHA256 = "2e8be4c030e8e04ab903adb5a8a786f11654ecca0380ee54d17509a707f49085"
CLAIMED = {"E6": 6.014331228348673, "E12": 12.057508280349891, "pair_keV": 1024.4508487023415}
PHI = (1 + sp.sqrt(5)) / 2
HALF_TURN = sp.diag(-1, -1, 1)


def cell():
    """The twelve ports and thirty seams of one icosahedral cell."""
    ports = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            ports += [sp.Matrix([0, s1, s2 * PHI]), sp.Matrix([s1, s2 * PHI, 0])]
            ports += [sp.Matrix([s2 * PHI, 0, s1])]
    seams = [
        (i, j)
        for i in range(12)
        for j in range(i + 1, 12)
        if sp.simplify((ports[i] - ports[j]).dot(ports[i] - ports[j]) - 4) == 0
    ]
    assert len(seams) == 30
    return ports, seams


def native(d, c):
    """The endpoint rule's current j for load difference d and coin c."""
    return sp.Rational(d - (2 * c - 1) * (abs(d) % 2), 2)


def always_move(d, c):
    """A changed rule that respects the cell's symmetry: a touched seam always moves a unit."""
    return sp.sign(d)


def orientation_biased(d, c):
    """A changed rule that breaks the symmetry: the weight depends on the seam's label order."""
    return native(d, c) * (sp.Rational(17, 10) if c else sp.Rational(2, 5))


def moments(ports, seams, axis, rule=native):
    """Mean and covariance of the first current for +1 on `axis`, -1 on its antipode."""
    anti = next(k for k in range(12) if ports[k] == -ports[axis])
    q = [0] * 12
    q[axis], q[anti] = 1, -1
    steps = [rule(q[a] - q[b], c) * (ports[b] - ports[a]) for a, b in seams for c in (0, 1)]
    mean = sum(steps, sp.zeros(3, 1)) / len(steps)
    second = sum((s * s.T for s in steps), sp.zeros(3, 3)) / len(steps)
    return sp.simplify(mean), sp.simplify(second - mean * mean.T), len(steps)


def structure(ports, seams, rule):
    """On how many of the 12 axes: mean along v, C uniaxial about v, half turn carries (m, C)."""
    along = uniaxial = carried = 0
    for axis in range(12):
        v = ports[axis]
        m, cov, _ = moments(ports, seams, axis, rule)
        v2 = v.dot(v)
        along += sp.simplify(m - (m.dot(v) / v2) * v) == sp.zeros(3, 1) and m != sp.zeros(3, 1)
        par = sp.simplify((v.T * cov * v)[0] / v2)
        perp = sp.simplify((cov.trace() - par) / 2)
        shape = par * v * v.T / v2 + perp * (sp.eye(3) - v * v.T / v2)
        uniaxial += sp.simplify(cov - shape) == sp.zeros(3, 3)
        image = next(k for k in range(12) if ports[k] == HALF_TURN * v)
        m2, cov2, _ = moments(ports, seams, image, rule)
        carried += sp.simplify(m2 - HALF_TURN * m) == sp.zeros(3, 1) and sp.simplify(
            cov2 - HALF_TURN * cov * HALF_TURN.T
        ) == sp.zeros(3, 3)
    return {"mean_along_v": along, "uniaxial": uniaxial, "half_turn_carries": carried}


def stopping_times(seams):
    """Expected events until a +1/-1 pair cancels, for all 132 ordered pairs, exactly."""
    index = {
        pair: i for i, pair in enumerate((p, r) for p in range(12) for r in range(12) if p != r)
    }
    a_matrix, ones = sp.eye(132), sp.ones(132, 1)
    for (p, r), i in index.items():
        for a, b in seams:
            for c in (0, 1):
                q = [0] * 12
                q[p], q[r] = 1, -1
                j = native(q[a] - q[b], c)
                q[a] -= j
                q[b] += j
                if any(q):
                    a_matrix[i, index[(q.index(1), q.index(-1))]] -= sp.Rational(1, 60)
    return sorted({int(t) for t in a_matrix.LUsolve(ones)})


def part_a():
    ports, seams = cell()
    out = {"seams": len(seams)}
    exact = 0
    for axis in range(12):
        v = ports[axis]
        m, cov, count = moments(ports, seams, axis)
        v2 = v.dot(v)
        par = sp.simplify((v.T * cov * v)[0] / v2)
        perp = sp.simplify((cov.trace() - par) / 2)
        exact += (
            sp.simplify(m - (PHI - 3) / 15 * v) == sp.zeros(3, 1)
            and sp.simplify(par - (3 - PHI) / 9) == 0
            and sp.simplify(perp - (PHI + 2) / 15) == 0
        )
    out["A1_axes_matching_claimed_moments"] = exact
    out["A1_alternatives_per_axis"] = count
    out["A2_half_turn_is_a_symmetry"] = all(HALF_TURN * p in ports for p in ports)
    out["A2_ports_sent_to_antipode"] = sum(HALF_TURN * p == -p for p in ports)
    out["A3_stopping_times"] = stopping_times(seams)
    out["A4_native"] = structure(ports, seams, native)
    out["A4_always_move"] = structure(ports, seams, always_move)
    out["A4_orientation_biased"] = structure(ports, seams, orientation_biased)
    m_native, _, _ = moments(ports, seams, 0, native)
    m_changed, _, _ = moments(ports, seams, 0, always_move)
    out["A4_always_move_mean_over_native"] = str(
        sp.simplify(m_changed.dot(ports[0]) / m_native.dot(ports[0]))
    )
    return out


def energy(x, seams):
    return -2 * seams * math.log(1 - x / (2 * seams))


def part_b(path):
    raw = open(path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != GRAPH_SHA256:
        raise SystemExit(f"graph file is not the pinned one: sha256 {digest}")
    graph = json.loads(raw)
    n, seams = graph["ports"], [tuple(e) for e in graph["seams"]]
    lap = np.zeros((n, n))
    for a, b in seams:
        lap[a, a] += 1
        lap[b, b] += 1
        lap[a, b] -= 1
        lap[b, a] -= 1
    outside = [(a, b) for a, b in seams if a // 12 != b // 12]
    links = {}
    for a, b in outside:
        links.setdefault(a // 12, set()).add(b // 12)
        links.setdefault(b // 12, set()).add(a // 12)
    spectrum = np.linalg.eigvalsh(lap)
    profiles = set()
    for start in links:
        distance = {start: 0}
        frontier = [start]
        while frontier:
            frontier = [n2 for n1 in frontier for n2 in links[n1] if n2 not in distance]
            distance.update({n2: distance.get(n2, max(distance.values()) + 1) for n2 in frontier})
            frontier = list(dict.fromkeys(frontier))
        counts = [list(distance.values()).count(k) for k in range(max(distance.values()) + 1)]
        profiles.add(tuple(counts))
    out = {
        "B1_cell_distance_profiles": sorted(map(list, profiles)),
        "B1_ports": n,
        "B1_seams": len(set(map(frozenset, seams))),
        "B1_cells": len(links),
        "B1_seams_between_cells": len(outside),
        "B1_links_per_cell": sorted({len(v) for v in links.values()}),
        "B1_ports_with_outside_seam": len({p for e in outside for p in e}),
        "B4_modes_below_6": int((spectrum < 6 - 1e-9).sum()),
        "B4_modes_at_6": int((abs(spectrum - 6) < 1e-9).sum()),
        "B4_lowest_nonzero": float(spectrum[1]),
    }
    for name in ("f", "g"):
        ray = np.zeros(n)
        for port, value in graph[name].items():
            ray[int(port)] = value
        out[f"B2_residual_{name}"] = float(abs(lap @ ray - 6 * ray).max())
        support = {int(port) for port in graph[name]}
        out[f"B2_outside_seams_on_{name}"] = sum(a in support or b in support for a, b in outside)
    e6, e12 = energy(6, len(seams)), energy(12, len(seams))
    out.update(
        B3_E6=e6, B3_E12=e12, B3_pair_keV=511 * e12 / e6, B3_excess_keV=511 * e12 / e6 - 1022
    )
    return out


def verdicts(a, b):
    """One named comparison per claim. Each returns True only on agreement."""
    checks = {
        "A1 moments on all 12 axes": a["A1_axes_matching_claimed_moments"] == 12,
        "A1 60 alternatives per axis": a["A1_alternatives_per_axis"] == 60,
        "A2 half turn is a symmetry, four ports to antipodes": a["A2_half_turn_is_a_symmetry"]
        and a["A2_ports_sent_to_antipode"] == 4,
        "A3 stopping times 66, 84, 90": a["A3_stopping_times"] == [66, 84, 90],
        "A4 native rule: structure on 12 of 12 axes": set(a["A4_native"].values()) == {12},
        "A4 symmetric changed rule: same structure": set(a["A4_always_move"].values()) == {12},
        "A4 symmetric changed rule: different magnitude": a["A4_always_move_mean_over_native"]
        != "1",
        "A4 symmetry-breaking rule fails": a["A4_orientation_biased"]["half_turn_carries"] < 12,
    }
    if b:
        checks.update(
            {
                "B1 240 ports, 630 seams, 20 cells, 30 links, 3 per cell": (
                    b["B1_ports"],
                    b["B1_seams"],
                    b["B1_cells"],
                    b["B1_seams_between_cells"],
                    b["B1_links_per_cell"],
                )
                == (240, 630, 20, 30, [3]),
                "B1 the 20 cells are linked as a dodecahedron": b["B1_cell_distance_profiles"]
                == [[1, 3, 6, 6, 3, 1]],
                "B2 L f = 6 f and L g = 6 g": max(b["B2_residual_f"], b["B2_residual_g"]) < 1e-12,
                "B2 neither ray touches an outside seam": b["B2_outside_seams_on_f"]
                + b["B2_outside_seams_on_g"]
                == 0,
                "B3 E6": abs(b["B3_E6"] - CLAIMED["E6"]) < 1e-12,
                "B3 E12": abs(b["B3_E12"] - CLAIMED["E12"]) < 1e-12,
                "B3 pair energy": abs(b["B3_pair_keV"] - CLAIMED["pair_keV"]) < 1e-9,
            }
        )
    return checks


def arms(a, b):
    """Feed each comparison a wrong value. Every one must turn false."""
    wrong_a = dict(a, A1_axes_matching_claimed_moments=11, A1_alternatives_per_axis=30)
    wrong_a.update(A2_ports_sent_to_antipode=12, A3_stopping_times=[66, 90])
    wrong_a.update(
        A4_native={"x": 11}, A4_always_move={"x": 11}, A4_always_move_mean_over_native="1"
    )
    wrong_a.update(A4_orientation_biased={"half_turn_carries": 12})
    wrong_b = None
    if b:
        wrong_b = dict(b, B1_cells=19, B2_residual_f=1e-3, B3_E6=b["B3_E6"] + 1e-9)
        wrong_b.update(B1_cell_distance_profiles=[[1, 3, 6, 10]], B2_outside_seams_on_f=1)
        wrong_b.update(B3_E12=b["B3_E12"] + 1e-9, B3_pair_keV=b["B3_pair_keV"] + 1e-6)
    return [name for name, passed in verdicts(wrong_a, wrong_b).items() if passed]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("graph", nargs="?", help="graph.json of the pinned author package")
    parser.add_argument("--output", help="write the result as JSON")
    args = parser.parse_args()
    a = part_a()
    b = part_b(args.graph) if args.graph else None
    checks = verdicts(a, b)
    survivors = arms(a, b)
    for name, passed in checks.items():
        print(("PASS  " if passed else "FAIL  ") + name)
    print(
        f"arms: {len(checks) - len(survivors)} of {len(checks)} comparisons turn false on a wrong value"
    )
    result = {
        "task": "M10.2",
        "part_a": a,
        "part_b": b,
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
