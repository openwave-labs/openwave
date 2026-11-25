#!/usr/bin/env python3
"""M10.4, the maintainers' independent recomputation of the connected two-load release.

It imports nothing from the author's package or from the pull request's scripts, and
executes none of them. It reads one data file of the author's, the carrier's seam list, as
arrays, and the run summary the pull request ships, `m10_4_reproduction.json`, as JSON.

    python3 -I m10_4_maintainer_recompute.py BOARD.npz [--output OUT.json]

BOARD.npz is `model/executable_model/carriers/FEDERATION_LEVEL0.npz` of the author package
at commit 09f062f77f7f540f2783f128ae6314c5d8b818ea (tag
interval-echo-m10.4-connected-pair-proposal), the same bytes as the M10.3 board. Its
SHA-256 is checked before anything is computed from it, and so is the run summary's.

The equations. Two unit loads at distinct ports i < j of the 240-port carrier, 28,680
configurations. A seam is live when its memory is blank. On a live seam a load moves into
an empty port.

  m_ij          = the number of such moves from configuration ij
  L             = the Laplacian of those moves (diagonal m_ij, -1 per move)
  K = L / 2,    C_ij,ij = m_ij + (6 - deg i) + (6 - deg j)     (deg: the board's original degrees)
  H(kappa)      = K + kappa C
  written board = every seam live except the eleven that touch exactly one of the ports
                  a = 157, b = 156, c = 166, other than (157, 163) and (163, 166)
  blank board   = all 630 seams live
  X = ab, Y = ac, Z = bc

  The step from the seam law to this generator rests on one fact, checked here in exact
  arithmetic on one seam from the law's own definition (the docstring of
  `model/candidate_laws/contrast_memory_spherical_v1/law.py` at the pin): with
  U = [[0, W*], [W, I - W W*]] and G = 0_blank (+) [(I - R)/2 + kappa D^2] (x) I, the local
  term h = U G U* has no element between blank and written memory, acts on blank memory as
  (I - SWAP)/2 + kappa (n_a - n_b)^2, and annihilates the neutral record (the equal coin
  superposition at d = 0) beside each of the nine load words, for every kappa.

  Acquisition (the method note's three-word model): K_src on (X, Y, Z) with
  2 K_src = [[1, -1, 0], [-1, 2, -1], [0, -1, 1]], three pulses of length 2 pi / 3, each
  followed by a read that removes Y; f = the failed weight of one attempt, at most eight
  attempts. The accepted state is v = (-13 X + 14 Z) / 27 per attempt, the density
  rho = (sum_{j<8} f^j) v v*, and exhaustion f^8.

  The floor. D = (X - Z) / sqrt 2 is an eigenvector of K and of C on the written board
  (2 K D = 2 D, C D = 3 D, checked in integers), and lies inside the 30 home seams, so
  <D | U_t B> = 0 for any B orthogonal to D and the home probability is at least the
  weight of D in rho: (1 - f^8) * 729 / 730, for every real kappa and every t.

  The four arms at (kappa, t) in {0, 1/2, 8} x {1, 7, 37}, on both boards:
    coherent        | sqrt(w_X) U_t X + sqrt(w_Z) U_t Z |^2 summed over the home seams
    dephased        w_X |U_t X|^2 + w_Z |U_t Z|^2, summed the same way
    stochastic_swap exp(-t K) applied to the populations (w_X, w_Z)
    unfiltered      | U_t X |^2
  with (w_X, w_Z) the diagonal of rho. Propagation by scipy's expm_multiply, with the
  mean diagonal removed (a common phase).

Armed before any comparison: the two flown states stay unit and orthogonal to 1e-10 at
every setting, the energy of X is conserved over [0, 37] to 1e-10 of the largest diagonal
entry of H, and the route returns the three
values on record for the M10.3 sharp pair at kappa = 8 on the blank board, computed there
by a Chebyshev route ([m10_3_maintainer_recompute.py]).

Compared with the author's numbers (CLAIMED below): the eleven written seams and the two
open contacts, the acquisition, the floor, the dark identity and its two failing
controls, the escape ranks, the method note's table at exposure 37, the blank-memory
comparison, the stochastic control at exposure 1, the ten-write control, the one-load
delivery table, the seam fact; and the 72 home probabilities of the shipped run summary.

Reported, not compared: the coherent arm at exposure 37 for kappa in
(0, 0.5, 1, 2, 4, 8, 16, -8) on the written board; one load alone on the written board;
the one-load leak of each written seam restored on its own.

Every comparison below can fail: section ARMS feeds each one a wrong value.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from fractions import Fraction

import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

BOARD_SHA256 = "99870398e3d516f899c600e9bf25ba53c7f513142ba8facbe1cedd94b23f915b"
SUMMARY_SHA256 = "b4cfcfee71a11cae6ba908bcba32141c7d5f181342e5ebdbeb0f3b3573c232dd"
SUMMARY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "m10_4_reproduction.json"
)
PORTS, CELL_PORTS, HOME_CELL = 240, 12, 13
A, B, C, EXIT = 157, 156, 166, 163
KAPPAS, TIMES = (0.0, 0.5, 8.0), (1.0, 7.0, 37.0)
ARMS = ("coherent", "dephased", "stochastic_swap", "unfiltered")
KAPPA_TABLE = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, -8.0)

# The authors' numbers, as the method note, the task and the pull request state them.
CLAIMED = {
    "written_seams": [
        [152, 166], [156, 161], [156, 163], [156, 167], [157, 161], [157, 164],
        [157, 165], [157, 170], [158, 166], [162, 166], [166, 167],
    ],  # fmt: skip
    "written_between_cells": [[152, 166], [157, 170]],
    "open_contacts": [[157, 163], [163, 166]],
    "first_vacancies": ["4/9", "4/81", "4/729"],
    "accepted_times_27": [-13, 14],
    "exhaustion": 0.0038635883,
    "floor": 0.994771841,
    "dark_eigenvalues": {"twice_exchange": 2, "contrast": 3},
    "escape_ranks": {"written": 29, "blank": 30},
    "table_t37": {
        "coherent": [0.995196058, 0.995945068, 0.996109886],
        "dephased": [0.652907216, 0.926296116, 0.986454527],
        "stochastic_swap": [0.123196093, 0.123196093, 0.123196093],
        "unfiltered": [0.655439565, 0.929888823, 0.990280564],
    },
    "blank_coherent_k05_t37": 0.440929105,
    "blank_unfiltered_k05_t37": 0.416819359,
    "stochastic_t1": 0.753088868,
    "ten_write_coherent_t37": [0.526840130, 0.940265361, 0.996086764],
    "one_load_receiver_other": {
        "fixed_attempts": {"blank": ["1/15", "44/75"], "eleven": ["1/15", "0"],
                           "ten": ["11/75", "2/25"]},
        "stop_on_receipt": {"blank": ["4/15", "38/75"], "eleven": ["1/15", "0"],
                            "ten": ["26/75", "0"]},
    },  # fmt: skip
}
# The values on record for the M10.3 sharp pair on the blank board at kappa = 8.
M10_3_RECORD = [0.990982505, 0.989653268, 0.980271948]
NINE_DIGITS = 0.5e-9  # half a unit in the ninth decimal of a value rounded to nine digits
TEN_DIGITS = 0.5e-10
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


def load_summary():
    """The run summary the pull request ships, read as JSON after its SHA-256 is checked."""
    with open(SUMMARY, "rb") as handle:
        raw = handle.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SUMMARY_SHA256:
        raise SystemExit(f"run summary is not the merged one: sha256 {digest}")
    return json.loads(raw), digest


# ---- the seam law on one seam, exact -----------------------------------------------------------


def seam_fact():
    """The local term of the pinned law on one seam, in exact arithmetic, symbolic kappa."""
    words = [(a, b) for a in range(3) for b in range(3)]
    labels = [(c, d) for c in (0, 1) for d in range(-2, 3)]
    kappa = sp.Symbol("kappa", real=True)
    w = sp.zeros(90, 9)
    for col, (a, b) in enumerate(words):
        s = a + b
        for c in (0, 1):
            y = (s // 2 + c * (s % 2), s // 2 + (1 - c) * (s % 2))
            w[labels.index((c, a - b)) * 9 + words.index(y), col] += 1 / sp.sqrt(2)
    u = sp.BlockMatrix([[sp.zeros(9), w.T], [w, sp.eye(90) - w * w.T]]).as_explicit()
    r = sp.zeros(10)
    for i, (c, d) in enumerate(labels):
        r[labels.index((c, -d)), i] = 1
    d2 = sp.diag(*[d * d for _, d in labels])
    g = sp.diag(sp.zeros(9), sp.kronecker_product((sp.eye(10) - r) / 2 + kappa * d2, sp.eye(9)))
    h = (u * g * u).applyfunc(sp.expand)
    swap = sp.zeros(9)
    for i, (a, b) in enumerate(words):
        swap[words.index((b, a)), i] = 1
    contrast = sp.diag(*[(a - b) ** 2 for a, b in words])

    def zero(m):
        return all(sp.expand(x) == 0 for x in m)

    def record(d):
        m = sp.zeros(99, 9)
        for k in range(9):
            for c in (0, 1):
                m[9 + labels.index((c, d)) * 9 + k, k] = 1 / sp.sqrt(2)
        return m

    empty = sp.zeros(99, 1)
    empty[words.index((0, 0))] = 1
    return {
        "echo_squares_to_identity": zero(u * u - sp.eye(99)),
        "no_element_between_blank_and_written": zero(h[9:, :9]) and zero(h[:9, 9:]),
        "blank_block_is_exchange_plus_contrast": zero(
            h[:9, :9] - ((sp.eye(9) - swap) / 2 + kappa * contrast)
        ),
        "neutral_record_switches_the_seam_off": zero(h * record(0)),
        "echo_of_an_empty_blank_seam_is_the_neutral_record": zero(
            u * empty - record(0)[:, words.index((0, 0))]
        ),
        "control_record_at_d1_switches_the_seam_off": zero(h * record(1)),
    }


# ---- the board ---------------------------------------------------------------------------------


def degrees(seams):
    deg = np.zeros(PORTS, dtype=np.int64)
    for a, b in seams:
        deg[a] += 1
        deg[b] += 1
    return deg


def written_seams(seams):
    trio, open_contacts = {A, B, C}, {(A, EXIT), (EXIT, C)}
    return [e for e in seams if len(trio & set(e)) == 1 and e not in open_contacts]


def sector(live, deg):
    """Configurations i < j, the move Laplacian L (integers) and the contrast diagonal C."""
    conf = [(i, j) for i in range(PORTS) for j in range(i + 1, PORTS)]
    idx = {c: k for k, c in enumerate(conf)}
    nb = [[] for _ in range(PORTS)]
    for a, b in live:
        nb[a].append(b)
        nb[b].append(a)
    rows, cols = [], []
    moves = np.zeros(len(conf), dtype=np.int64)
    for k, (i, j) in enumerate(conf):
        for p, q in ((i, j), (j, i)):
            for r in nb[p]:
                if r != q:
                    rows.append(idx[(min(r, q), max(r, q))])
                    cols.append(k)
                    moves[k] += 1
    lap = sparse.csr_matrix(
        (-np.ones(len(rows), dtype=np.int64), (rows, cols)), shape=(len(conf),) * 2
    ) + sparse.diags(moves, format="csr", dtype=np.int64)
    cost = moves + np.array([(6 - deg[i]) + (6 - deg[j]) for i, j in conf])
    return conf, idx, lap.tocsr(), cost


def rank_exact(rows):
    """Rank over the rationals, by fraction-free elimination on Python integers."""
    m = [list(map(int, r)) for r in rows]
    rank = 0
    for col in range(len(m[0])):
        piv = next((r for r in range(rank, len(m)) if m[r][col] != 0), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        for r in range(rank + 1, len(m)):
            if m[r][col]:
                f, g = m[rank][col], m[r][col]
                m[r] = [f * x - g * y for x, y in zip(m[r], m[rank])]
        rank += 1
    return rank


# ---- acquisition and flight --------------------------------------------------------------------


def acquisition():
    twice_k = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float)
    vals, vecs = np.linalg.eigh(twice_k / 2)
    pulse = (vecs * np.exp(-1j * (2 * np.pi / 3) * vals)) @ vecs.conj().T
    state = np.array([1, 0, 0], dtype=complex)
    fails = []
    for _ in range(3):
        state = pulse @ state
        fails.append(abs(state[1]) ** 2)
        state[1] = 0
    phase = state[0] / abs(state[0])
    accepted = (state[[0, 2]] / -phase * 27).real  # common phase fixed so X carries -13
    f = sum(fails)
    retry = sum(f**k for k in range(8))
    return {
        "first_vacancies": fails,
        "accepted_times_27": accepted.tolist(),
        "failed_per_attempt": f,
        "exhaustion": f**8,
        "weights": (retry * np.abs(state[[0, 2]]) ** 2).tolist(),
        "amplitudes": (np.sqrt(retry) * state[[0, 2]]).tolist(),
    }


def floor_exact():
    f = Fraction(364, 729)
    return (1 - f**8) * Fraction(729, 730)


def flown(lap, cost, kets, kappa, times):
    h = (lap.astype(float) / 2 + sparse.diags(kappa * cost.astype(float))).tocsr()
    h = h - h.diagonal().mean() * sparse.identity(h.shape[0], format="csr")
    out, block, prev = {}, np.array(kets, dtype=complex).T, 0.0
    for t in times:
        block = expm_multiply(-1j * (t - prev) * h, block)
        out[t] = block.copy()
        prev = t
    return out, h


def arms_at(board, acq, kappas, times, energy=False):
    conf, idx, lap, cost, det = board
    x, z = np.zeros(len(conf)), np.zeros(len(conf))
    x[idx[(B, A)]], z[idx[(B, C)]] = 1, 1
    amp, wts = np.array(acq["amplitudes"]), np.array(acq["weights"])
    stoch, pop, prev = {}, wts[0] * x + wts[1] * z, 0.0
    for t in times:
        pop = expm_multiply(-(t - prev) * (lap.astype(float) / 2), pop)
        stoch[t] = float(pop[det].sum())
        prev = t
    rows, unit_err, energy_drift = {}, 0.0, 0.0
    for kappa in kappas:
        fl, h = flown(lap, cost, [x, z], kappa, times)
        for t in times:
            px, pz = fl[t][:, 0], fl[t][:, 1]
            unit_err = max(
                unit_err,
                abs(np.vdot(px, px) - 1),
                abs(np.vdot(pz, pz) - 1),
                abs(np.vdot(px, pz)),
            )
            rows[(kappa, t)] = {
                "coherent": float((np.abs(amp[0] * px + amp[1] * pz) ** 2)[det].sum()),
                "dephased": float(
                    (wts[0] * np.abs(px) ** 2 + wts[1] * np.abs(pz) ** 2)[det].sum()
                ),
                "stochastic_swap": stoch[t],
                "unfiltered": float((np.abs(px) ** 2)[det].sum()),
            }
        if energy:
            p37 = fl[times[-1]][:, 0]
            scale = abs(h.diagonal()).max() + 1.0  # bounds the spectral radius up to the hops
            energy_drift = max(
                energy_drift,
                abs(np.vdot(p37, h @ p37).real - np.vdot(x, h @ x).real) / scale,
            )
    return rows, unit_err, energy_drift


# ---- the one-load facts ------------------------------------------------------------------------


def one_load(seams, written):
    nb = {}
    for a, b in seams:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)

    def bins(closed, stop):
        mass = {B: Fraction(1)}
        for _ in range(2):
            nxt = {}
            for p, wgt in mass.items():
                if stop and p == EXIT:
                    nxt[p] = nxt.get(p, 0) + wgt
                    continue
                for q in nb[p]:
                    dest = p if (min(p, q), max(p, q)) in closed else q
                    nxt[dest] = nxt.get(dest, 0) + wgt / len(nb[p])
            mass = nxt
        receiver = mass.get(EXIT, Fraction(0))
        known = receiver + sum(mass.get(p, 0) for p in (A, B, C))
        return [str(receiver), str(1 - known)]

    boards = {"blank": set(), "eleven": set(written), "ten": set(written) - {(B, EXIT)}}
    table = {
        rule: {name: bins(closed, stop) for name, closed in boards.items()}
        for rule, stop in (("fixed_attempts", False), ("stop_on_receipt", True))
    }

    def lap1(live):
        m = np.zeros((PORTS, PORTS), dtype=np.int64)
        for a, b in live:
            m[a, a] += 1
            m[b, b] += 1
            m[a, b] -= 1
            m[b, a] -= 1
        return m

    v = np.zeros(PORTS, dtype=np.int64)
    v[A], v[C] = 1, -1
    live = [e for e in seams if e not in written]
    leaks = {}
    for e in written:
        r = lap1(live + [e]) @ v
        r[A] = r[C] = 0
        leaks[f"{e[0]}-{e[1]}"] = str(Fraction(int(r @ r), 8))
    return table, {
        "a_minus_c_is_an_eigenvector_on_the_written_board": bool(
            np.array_equal(lap1(live) @ v, 2 * v)
        ),
        "a_minus_c_is_an_eigenvector_on_the_blank_board": bool(
            np.array_equal(lap1(seams) @ v, 2 * v)
        ),
        "leak_per_written_seam_restored_alone": leaks,
        "contacts_touching_a_b_c": sum(1 for e in seams if {A, B, C} & set(e)),
    }


# ---- comparisons -------------------------------------------------------------------------------


def arming_checks(a):
    """The route is armed when these hold; they precede every comparison."""
    return {
        "arming: flown states unit and orthogonal to 1e-10": a["unit_err"] < 1e-10,
        "arming: energy of X conserved over [0, 37], 1e-10 of the spectral scale": a[
            "energy_drift"
        ]
        < 1e-10,
        "arming: the M10.3 record for the sharp pair at kappa 8 on blank memory": all(
            abs(x - y) <= NINE_DIGITS + ROUTE for x, y in zip(a["m10_3_values"], M10_3_RECORD)
        ),
    }


def close(x, y, tol=NINE_DIGITS + ROUTE):
    return all(abs(p - q) <= tol for p, q in zip(x, y))


def verdicts(r):
    """One named comparison per claim. Each returns True only on agreement."""
    c, s = r["claims"], r["seam"]
    checks = arming_checks(r["arming"])
    checks |= {
        "seam law: no element between blank and written memory": s[
            "no_element_between_blank_and_written"
        ]
        and s["echo_squares_to_identity"]
        and s["blank_block_is_exchange_plus_contrast"],
        "seam law: a neutral record switches its seam off, every kappa": s[
            "neutral_record_switches_the_seam_off"
        ]
        and s["echo_of_an_empty_blank_seam_is_the_neutral_record"],
        "seam law: the control record at d = 1 does not": not s[
            "control_record_at_d1_switches_the_seam_off"
        ],
        "the eleven written seams, two between cells": c["written_seams"]
        == CLAIMED["written_seams"]
        and c["written_between_cells"] == CLAIMED["written_between_cells"],
        "the two open contacts are live seams": c["open_contacts"] == CLAIMED["open_contacts"],
        "acquisition: failures 4/9, 4/81, 4/729": close(
            c["first_vacancies"], [float(Fraction(q)) for q in CLAIMED["first_vacancies"]], 1e-14
        ),
        "acquisition: accepted state (-13 X + 14 Z) / 27": close(
            c["accepted_times_27"], CLAIMED["accepted_times_27"], 1e-12
        ),
        "acquisition: exhaustion 0.0038635883": abs(c["exhaustion"] - CLAIMED["exhaustion"])
        <= TEN_DIGITS + ROUTE,
        "the floor (1 - f^8) 729 / 730": abs(float(c["floor_exact"]) - CLAIMED["floor"])
        <= NINE_DIGITS
        and abs(c["dark_weight"] - float(c["floor_exact"])) < 1e-12,
        "dark identity: 2K D = 2 D and C D = 3 D on the written board": c["dark"]
        == CLAIMED["dark_eigenvalues"],
        "dark identity fails for X + Z": not c["dark_wrong_phase"],
        "dark identity fails on the blank board": not c["dark_blank_board"],
        "escape ranks 29 written, 30 blank": c["escape_ranks"] == CLAIMED["escape_ranks"],
        "the method note's table at exposure 37": all(
            close(c["table_t37"][arm], CLAIMED["table_t37"][arm]) for arm in ARMS
        ),
        "coherent arm at or above the floor, nine settings": c["coherent_min"]
        >= float(c["floor_exact"]) - 1e-12,
        "blank memories at kappa 1/2, exposure 37": close(
            [c["blank_coherent_k05_t37"], c["blank_unfiltered_k05_t37"]],
            [CLAIMED["blank_coherent_k05_t37"], CLAIMED["blank_unfiltered_k05_t37"]],
        ),
        "stochastic control at exposure 1, below 0.94": close(
            [c["stochastic_t1"]], [CLAIMED["stochastic_t1"]]
        )
        and c["stochastic_t1"] < 0.94,
        "ten writes, coherent, exposure 37": close(
            c["ten_write_coherent_t37"], CLAIMED["ten_write_coherent_t37"]
        ),
        "one-load delivery table": c["one_load_receiver_other"]
        == CLAIMED["one_load_receiver_other"],
        "run summary: 72 home probabilities to nine digits": c["summary_compared"] == 72
        and c["summary_max_difference"] <= NINE_DIGITS + ROUTE,
    }
    return {name: bool(passed) for name, passed in checks.items()}


def arms(r):
    """Feed each comparison a wrong value. Every one must turn false."""
    w = copy.deepcopy(r)
    w["arming"].update(unit_err=2e-10, energy_drift=2e-10)
    w["arming"]["m10_3_values"][2] += 2e-9
    w["seam"].update(
        no_element_between_blank_and_written=False,
        neutral_record_switches_the_seam_off=False,
        control_record_at_d1_switches_the_seam_off=True,
    )
    c = w["claims"]
    c["written_seams"] = c["written_seams"][:-1]
    c["open_contacts"] = [[156, 163], [163, 166]]
    c["first_vacancies"][2] += 1e-12
    c["accepted_times_27"][0] = -12.0
    c["exhaustion"] += 1e-9
    c["floor_exact"] = Fraction(99477185, 10**8)
    c["dark"] = {"twice_exchange": 2, "contrast": 2}
    c["dark_wrong_phase"] = True
    c["dark_blank_board"] = True
    c["escape_ranks"] = {"written": 30, "blank": 30}
    c["table_t37"]["dephased"][1] += 2e-9
    c["coherent_min"] = 0.9947
    c["blank_coherent_k05_t37"] += 2e-9
    c["stochastic_t1"] += 2e-9
    c["ten_write_coherent_t37"][0] += 2e-9
    c["one_load_receiver_other"]["fixed_attempts"]["ten"] = ["11/75", "1/25"]
    c["summary_max_difference"] = 1e-9
    return [name for name, passed in verdicts(w).items() if passed]


# ---- run ---------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("board", help="FEDERATION_LEVEL0.npz of the pinned author package")
    parser.add_argument("--output", help="write the result as JSON")
    args = parser.parse_args()

    seams, board_sha = load_board(args.board)
    summary, summary_sha = load_summary()
    deg = degrees(seams)
    written = written_seams(seams)
    home = [e for e in seams if e[0] // CELL_PORTS == e[1] // CELL_PORTS == HOME_CELL]
    acq = acquisition()

    boards = {}
    for name, live in (
        ("written", [e for e in seams if e not in written]),
        ("blank", seams),
        ("ten", [e for e in seams if e not in written or e == (B, EXIT)]),
    ):
        conf, idx, lap, cost = sector(live, deg)
        boards[name] = (conf, idx, lap, cost, np.array([idx[e] for e in home]))

    tables, unit_err, drift = {}, 0.0, 0.0
    for name in ("written", "blank"):
        tables[name], u, d = arms_at(boards[name], acq, KAPPAS, TIMES, energy=True)
        unit_err, drift = max(unit_err, u), max(drift, d)
    armed = {
        "unit_err": unit_err,
        "energy_drift": drift,
        # the sharp pair of M10.3 on blank memory at kappa = 8
        "m10_3_values": [tables["blank"][(8.0, t)]["unfiltered"] for t in TIMES],
    }
    r = {"arming": armed}
    failed = [name for name, ok in arming_checks(armed).items() if not ok]
    if failed:
        print("FAIL  " + "\nFAIL  ".join(failed) + "\nnot armed: no comparison is made")
        return 1

    ten, _, _ = arms_at(boards["ten"], acq, KAPPAS, (37.0,))
    conf, idx, lap, cost, det = boards["written"]
    dark = np.zeros(len(conf), dtype=np.int64)
    dark[idx[(B, A)]], dark[idx[(B, C)]] = 1, -1
    bconf, bidx, blap, bcost, bdet = boards["blank"]
    bdark = np.zeros(len(bconf), dtype=np.int64)
    bdark[bidx[(B, A)]], bdark[bidx[(B, C)]] = 1, -1

    def escape_rank(lp, dt):
        outside = np.ones(lp.shape[0], dtype=bool)
        outside[dt] = False
        block = lp[:, dt][outside]
        return rank_exact(block[block.getnnz(axis=1) > 0].toarray().tolist())

    differences = []
    for row in summary["home_probabilities"]:
        key = (float(row["kappa"]), float(row["exposure"]))
        for name, label in (("written", "marked"), ("blank", "blank")):
            for arm in ARMS:
                differences.append(abs(tables[name][key][arm] - row[label][arm]))

    routing, one_load_facts = one_load(seams, written)
    amp = np.array(acq["amplitudes"])
    r["seam"] = seam_fact()
    r["claims"] = {
        "written_seams": [list(e) for e in written],
        "written_between_cells": [
            list(e) for e in written if e[0] // CELL_PORTS != e[1] // CELL_PORTS
        ],
        "open_contacts": [list(e) for e in sorted({(A, EXIT), (EXIT, C)}) if e in set(seams)],
        "first_vacancies": acq["first_vacancies"],
        "accepted_times_27": acq["accepted_times_27"],
        "exhaustion": acq["exhaustion"],
        "floor_exact": floor_exact(),
        "dark_weight": float(abs(amp[0] - amp[1]) ** 2 / 2),
        "dark": {
            "twice_exchange": 2 if np.array_equal(lap @ dark, 2 * dark) else None,
            "contrast": 3 if np.array_equal(cost * dark, 3 * dark) else None,
        },
        "dark_wrong_phase": bool(np.array_equal(lap @ abs(dark), 2 * abs(dark))),
        "dark_blank_board": bool(np.array_equal(blap @ bdark, 2 * bdark)),
        "escape_ranks": {"written": escape_rank(lap, det), "blank": escape_rank(blap, bdet)},
        "table_t37": {arm: [tables["written"][(k, 37.0)][arm] for k in KAPPAS] for arm in ARMS},
        "coherent_min": min(tables["written"][(k, t)]["coherent"] for k in KAPPAS for t in TIMES),
        "blank_coherent_k05_t37": tables["blank"][(0.5, 37.0)]["coherent"],
        "blank_unfiltered_k05_t37": tables["blank"][(0.5, 37.0)]["unfiltered"],
        "stochastic_t1": tables["written"][(0.0, 1.0)]["stochastic_swap"],
        "ten_write_coherent_t37": [ten[(k, 37.0)]["coherent"] for k in KAPPAS],
        "one_load_receiver_other": routing,
        "summary_compared": len(differences),
        "summary_max_difference": max(differences),
    }
    coupling, _, _ = arms_at(boards["written"], acq, KAPPA_TABLE, (37.0,))
    r["reported"] = {
        "coherent_written_t37_by_kappa": {
            str(k): coupling[(k, 37.0)]["coherent"] for k in KAPPA_TABLE
        },
        "blank_coherent_t37_by_kappa": {
            str(k): tables["blank"][(k, 37.0)]["coherent"] for k in KAPPAS
        },
        "one_load": one_load_facts,
        "tables": {
            name: [
                {"kappa": k, "exposure": t, **tables[name][(k, t)]} for k in KAPPAS for t in TIMES
            ]
            for name in tables
        },
    }

    checks = verdicts(r)
    survivors = arms(r)
    for name, passed in checks.items():
        print(("PASS  " if passed else "FAIL  ") + name)
    print(
        f"comparisons: {len(checks)}; arms: {len(checks) - len(survivors)} of {len(checks)}"
        " comparisons turn false on a wrong value"
    )
    r["claims"]["floor_exact"] = str(r["claims"]["floor_exact"])
    result = {
        "task": "M10.4",
        "inputs": {"board_sha256": board_sha, "run_summary_sha256": summary_sha},
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
