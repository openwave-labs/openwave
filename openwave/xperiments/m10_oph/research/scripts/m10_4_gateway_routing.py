"""Reproduce the retrospective routing test on the filed M10.4 board."""

import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m10_4_connected_pair as filed


def routing_masks(seams, home):
    """Compare every binary mask on contacts incident on the station and its relays."""
    neighbours = defaultdict(set)
    gateways = set()
    for a, b in seams:
        neighbours[a].add(b)
        neighbours[b].add(a)
        if a // 12 != b // 12:
            gateways.update((a, b))
    local = {p for p in neighbours if p // 12 == home}
    source = min(local - gateways, key=lambda p: (-len(gateways & neighbours[p]), p))
    relays = sorted(neighbours[source] & gateways)
    filed.require(len(relays) == 2, "station requires two relays")
    incident = [e for e in seams if {source, *relays}.intersection(e)]
    strict, bypass = [], []
    for switches in product((False, True), repeat=len(incident)):
        live = defaultdict(set)
        for (a, b), written in zip(incident, switches, strict=True):
            if not written:
                live[a].add(b)
                live[b].add(a)
        if not set(relays) <= live[source]:
            continue
        # Disable one arm on departure. The surviving relay must have exactly
        # one nonreturn outcome, identical on both arms and within this home.
        left, right = (live[p] - {source} for p in relays)
        if len(left) != 1 or left != right or not left <= local - {source, *relays}:
            continue
        receiver = next(iter(left))
        other = live[source] - set(relays)
        mask = {e for e, written in zip(incident, switches, strict=True) if written}
        if not other:
            strict.append((receiver, mask))
        elif other == {receiver}:
            bypass.append((receiver, mask))
    return neighbours, source, relays, 2 ** len(incident), strict, bypass


def delivery_bins(neighbours, source, relays, receiver, written, *, stop):
    """Two uniformly chosen attempts. A written contact leaves the load at its port."""
    mass = {source: Fraction(1)}
    for _ in range(2):
        after = defaultdict(Fraction)
        for p, weight in mass.items():
            if stop and p == receiver:
                after[p] += weight
                continue
            for q in neighbours[p]:
                dest = p if tuple(sorted((p, q))) in written else q
                after[dest] += weight / len(neighbours[p])
        mass = after
    filed.require(sum(mass.values()) == 1, "incomplete one-load distribution")
    main = [mass[source], sum(mass[p] for p in relays), mass[receiver]]
    return [str(x) for x in (*main, 1 - sum(main))]


def replay(board):
    seams, digest = filed.m103.load_board(board)
    neighbours, b, relays, count, strict, bypass = routing_masks(seams, 13)
    filed.require(count == 32768 and len(strict) == len(bypass) == 1, "routing multiplicity")
    (d, cut), (other_d, shortcut) = strict[0], bypass[0]
    direct = tuple(sorted((b, d)))
    filed.require(d == other_d and direct in cut and shortcut == cut - {direct}, "shortcut")
    _, degree, _ = filed.m103.structure(seams)
    filed_mask, _ = filed.installation(seams, degree)
    filed.require((b, *relays, d) == (filed.B, filed.A, filed.C, filed.EXIT), "station")
    filed.require(cut == filed_mask, "routing did not select the filed installation")
    trials = {}
    for rule, stop in (("fixed_attempts", False), ("stop_on_receipt", True)):
        trials[rule] = {
            name: delivery_bins(neighbours, b, relays, d, mask, stop=stop)
            for name, mask in (("blank", set()), ("strict", cut), ("shortcut", shortcut))
        }
    filed.require(
        trials
        == {
            "fixed_attempts": {
                "blank": ["14/75", "4/25", "1/15", "44/75"],
                "strict": ["32/75", "38/75", "1/15", "0"],
                "shortcut": ["4/15", "38/75", "11/75", "2/25"],
            },
            "stop_on_receipt": {
                "blank": ["11/75", "2/25", "4/15", "38/75"],
                "strict": ["32/75", "38/75", "1/15", "0"],
                "shortcut": ["17/75", "32/75", "26/75", "0"],
            },
        },
        "two-attempt routing distribution",
    )
    # Commission the ten-write control with the direct seam blank from the start.
    # The occupied-port query still has a neutral contact to a vacant reference.
    source_ports = {p for word in filed.WORDS for p in word}
    meters = [e for e in shortcut if b in e and set(e) - source_ports]
    filed.require(bool(meters), "no vacancy meter shared by the two preparations")
    filed.local_law()
    weights, exhaustion, _ = filed.acquisition(degree)
    car, _, _, cost = filed.sector([e for e in seams if e not in shortcut], degree)
    home = [e for e in seams if e[0] // 12 == e[1] // 12 == 13]
    detector = np.array([car.idx[e] for e in home])
    rows = filed.appointments(car, cost, detector, weights, exhaustion)
    observed = [sum(row["bins"]["coherent"][:-1]) for row in rows if row["exposure"] == 37]
    np.testing.assert_allclose(
        observed, [0.526840130, 0.940265361, 0.996086764], atol=1e-9, rtol=0, equal_nan=False
    )
    return {
        "board_sha256": digest,
        "source": b,
        "relays": relays,
        "receiver": d,
        "masks_tested": count,
        "strict_mask": sorted(cut),
        "restored_contact": direct,
        "shared_vacancy_meter": min(meters),
        "one_load_bins_source_relays_receiver_other": trials,
        "outcome_order": [*home, "quiet"],
        "shortcut_appointments": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", type=Path)
    print(json.dumps(replay(parser.parse_args().board), allow_nan=False))
