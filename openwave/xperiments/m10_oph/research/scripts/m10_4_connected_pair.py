#!/usr/bin/env python3
"""Recompute the registered M10.4 detector from the M10.3 board.

python -I -B m10_4_connected_pair.py BOARD.npz [--archive connected_records.json] [--summary | --json]

The source design, archive and historical gate were known when the experiment was chosen.
Only data are read from the author's package. The sector and Chebyshev routine belong to
the OpenWave M10.3 replay. All nine appointments include the marked and blank flight arms.
"""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from math import factorial, pi
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm_multiply
from sympy import Matrix

# Isolated Python omits even the script's directory. Resolve this platform helper locally.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m10_3_maintainer_recompute as m103

ARCHIVE_SHA = "bf0f1d94c8d3d82809b315db814aa3332ca806f3d6a11eb9b8695a88fa2eaf58"
STRENGTHS, TIMES = (0, 0.5, 8), (1, 7, 37)
ARMS = ("coherent", "dephased", "stochastic_swap", "unfiltered")
TOL = 2e-10
A, B, C, EXIT = 157, 156, 166, 163
WORDS = ((B, A), (A, C), (B, C))  # X, Y, Z


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_law():
    """Exact 99-state native echo, blank/neutral reductions and selective instruments."""
    words = [(a, b) for a in range(3) for b in range(3)]
    labels = [(coin, d) for coin in range(2) for d in range(-2, 3)]
    basis = np.eye(99, dtype=np.int64)
    pattern = np.zeros((90, 9), dtype=np.int64)  # sqrt(2) times the write isometry
    for col, (a, b) in enumerate(words):
        lo, odd = divmod(a + b, 2)
        for coin in range(2):
            output = (lo + coin * odd, lo + (1 - coin) * odd)
            row = 9 * labels.index((coin, a - b)) + words.index(output)
            pattern[row, col] = 1
    # In the basis with blank columns scaled by sqrt(2), V = 2 U has integer entries.
    v = np.block(
        [[0 * basis[:9, :9], pattern.T], [2 * pattern, 2 * basis[9:, 9:] - pattern @ pattern.T]]
    )
    metric = np.r_[np.full(9, 2), np.ones(90, dtype=int)]
    require(np.array_equal(metric[:, None] * v, v.T * metric), "echo is not Hermitian")
    require(np.array_equal(v @ v, 4 * basis), "echo fails reversal")
    blank = basis[:, :9]
    # P selects blank memory, regardless of the endpoint word. P is diagonal
    # also in this scaled basis. The preparation echo is a positive control.
    is_blank = np.arange(99) < 9
    changes_blank = is_blank[:, None] != is_blank[None, :]
    require(v[changes_blank].any(), "preparation echo failed to cross the blank boundary")
    reflection = [labels.index((coin, -d)) for coin, d in labels]
    identity = np.eye(10, dtype=int)
    swap = [words.index((b, a)) for a, b in words]
    neutral_rows = [9 * (1 + labels.index((coin, 0))) for coin in range(2)]
    neutral = [basis[:, start : start + 9] for start in neutral_rows]
    coefficients = (
        (identity - identity[:, reflection], basis[:9, :9] - basis[:9, swap]),
        (np.diag([d * d for _, d in labels]), np.diag([(a - b) ** 2 for a, b in words])),
    )
    for label_op, blank_op in coefficients:
        # 4 h in this basis, for twice exchange and for contrast separately.
        local = v[:, 9:] @ np.kron(label_op, basis[:9, :9]) @ v[9:, :]
        require(not local[changes_blank].any(), "free term mixes blank and written memory")
        require(np.array_equal(local @ blank, 4 * blank @ blank_op), "blank reduction")
        require(all(not (local @ cols).any() for cols in neutral), "neutral seam acts")
    written = neutral[0] + neutral[1]
    require(np.array_equal(v @ blank[:, 0], 2 * written[:, 0]), "empty write")
    pair = np.arange(99) % 9 == words.index((1, 1))
    for loads in ((0, 0), (0, 1), (1, 0), (1, 1)):
        j = words.index(loads)
        for state in (blank[:, j], written[:, j]):
            require(
                np.array_equal(v @ (pair * (v @ state)), 4 * (loads == (1, 1)) * state),
                "pair click changes its material word or collar",
            )
        if sum(loads) <= 1:  # exterior end vacant during the monitored source
            state = written[:, j]
            require(
                np.array_equal(v[:, :9] @ (v @ state)[:9], 4 * (loads == (0, 0)) * state),
                "vacancy read changes the source",
            )
    # The other branch of each binary flag follows by V^2 = 4 I.
    return {
        "dimension": 99,
        "blank_columns_per_coefficient": 9,
        "neutral_columns_per_coefficient": 18,
    }


def installation(seams, degree):
    path = {A, B, C}
    neighbors = {v: {y if x == v else x for x, y in seams if v in (x, y)} for v in path}
    peers = {p for p in neighbors[B] - {A} - neighbors[A] if degree[p] == degree[A]}
    require(peers == {C} and neighbors[A] & neighbors[C] == {B, EXIT}, "source geometry")
    openings = {tuple(sorted((v, EXIT))) for v in (A, C)}
    cut = {e for e in seams if len(path.intersection(e)) == 1} - openings
    loads, written, swaps = {A, B}, set(), []
    for hole in (C, B, A, B, C):
        if hole in loads:
            edge = tuple(sorted((hole, (path - loads).pop())))
            require(edge in (WORDS[0], WORDS[2]), "setup swap outside source")
            loads.symmetric_difference_update(edge)
            swaps.append(edge)
        for edge in sorted(cut - written):
            if hole in edge:
                require(not loads.intersection(edge), "write requires two vacant ends")
                written.add(edge)
    require(
        len(cut) == 11 and written == cut and loads == {A, B} and len(swaps) == 4,
        "source installation incomplete",
    )
    return cut, swaps


def acquisition(degree):
    """Derive the addressed source and count each failed prefix and retry."""
    twice_k = np.zeros((3, 3), dtype=int)
    cost = np.zeros(3, dtype=int)
    for j, word in enumerate(WORDS):
        for edge in (WORDS[0], WORDS[2]):
            if len(set(edge).intersection(word)) == 1:
                dest = tuple(sorted(set(word).symmetric_difference(edge)))
                twice_k[j, j] += 1
                twice_k[WORDS.index(dest), j] -= 1
                cost[j] += 1
        cost[j] += (6 - degree[B]) * (B in word)
    require(
        np.array_equal(twice_k, [[1, -1, 0], [-1, 2, -1], [0, -1, 1]]) and np.all(cost == 2),
        "addressed source Hamiltonian",
    )
    # A vacancy selects Y. A pi exchange on b,c returns Y to X up to a common phase.
    require(set(WORDS[1]).symmetric_difference(WORDS[2]) == set(WORDS[0]), "retry swap")
    pulse = expm(-1j * pi / 3 * twice_k)  # uniform contrast phase cancels in each branch
    state = np.array([1.0, 0.0, 0.0], dtype=complex)
    # Integer eigenvectors with pulse phases 1, -1 and exp(-i*pi/3). The unnormalized
    # symmetric pair S=(1,0,1) goes to (S + 4Y)/3, and the occupied flag removes Y.
    modes = np.array([[1, 1, 1], [1, -2, 1], [1, 0, -1]]).T
    require(np.array_equal(twice_k @ modes, modes * [0, 3, 1]), "source pulse spectrum")
    require(
        np.array_equal(2 * modes[:, 0] + modes[:, 1], [3, 0, 3])
        and np.array_equal(2 * modes[:, 0] - modes[:, 1], [1, 4, 1]),
        "bright pulse",
    )
    bright = Fraction(1, 2)  # X=(S+D)/2
    first_vacancies = []
    for _ in range(3):
        bright /= 3
        first_vacancies.append((4 * bright) ** 2)
    dark = -Fraction(1, 2)  # three dark pulse phases give exp(-i*pi)
    signed = (bright + dark, bright - dark)
    for failure in first_vacancies:
        state = pulse @ state
        require(abs(abs(state[1]) ** 2 - float(failure)) < 2e-14, "monitored pulse failure")
        state[1] = 0
    require(signed == (Fraction(-13, 27), Fraction(14, 27)), "signed accepted state")
    source = np.asarray(signed, dtype=float)
    require(
        np.allclose(
            np.outer(state[[0, 2]], state[[0, 2]].conj()),
            np.outer(source, source),
            atol=2e-14,
            rtol=0,
        ),
        "three-read source density",
    )
    failed = sum(first_vacancies)
    retry = sum(failed**i for i in range(8))
    exhaustion = failed**8
    require(retry * sum(q * q for q in signed) + exhaustion == 1, "lost source mass")
    overlap = (signed[0] - signed[1]) ** 2 / 2
    return (
        source * np.sqrt(float(retry)),
        float(exhaustion),
        {
            "signed_one_attempt_numerators": [-13, 14],
            "denominator": 27,
            "first_vacancies": list(map(str, first_vacancies)),
            "exhaustion": str(exhaustion),
            "exhaustion_probability": float(exhaustion),
            "protected_weight": str(retry * overlap),
            "protected_probability": float(retry * overlap),
        },
    )


def sector(seams, degree):
    car = m103.Carrier(seams)
    pairs = np.array(sorted(car.idx, key=car.idx.__getitem__))
    # Twice K counts live exchanges. Each occupied port also keeps its original degree reference.
    twice = car.kinetic * 2
    require(np.array_equal(twice.data, np.rint(twice.data)), "fractional hop count")
    twice = twice.astype(np.int64)
    cost = twice.diagonal() + (6 - degree)[pairs].sum(axis=1)
    require((twice - twice.T).nnz == 0 and not np.any(twice.sum(axis=0)), "exchange Laplacian")
    x, y = np.transpose(seams)
    graph = sparse.csr_matrix((np.ones(len(x)), (x, y)), shape=(240, 240))
    require(
        connected_components(graph, directed=False, return_labels=False) == 1, "port graph splits"
    )
    require(
        len(pairs) == 28680
        and connected_components(twice, directed=False, return_labels=False) == 1,
        "two-load graph splits",
    )
    return car, pairs, twice, cost


def protected(twice, cost, dark, detector):
    return (
        np.array_equal(twice @ dark, 2 * dark)
        and np.array_equal(cost * dark, 3 * dark)
        and np.dot(dark[detector], dark[detector]) == 2
    )


def home_escape_rank(twice_exchange, detector):
    """Exact rank of the block from the thirty home outcomes into quiet."""
    outside = np.ones(twice_exchange.shape[0], dtype=bool)
    outside[detector] = False
    block = twice_exchange[:, detector][outside]
    block.eliminate_zeros()
    reachable = block[block.getnnz(axis=1) > 0]
    return Matrix(reachable.toarray().tolist()).rank()


def flight_certificate(car, pairs, twice, cost, detector):
    dark = np.zeros(len(pairs), dtype=np.int64)
    dark[[car.idx[WORDS[0]], car.idx[WORDS[2]]]] = [1, -1]
    require(protected(twice, cost, dark, detector), "full-sector dark identity")
    bright = abs(dark)
    require(not protected(twice, cost, bright, detector), "wrong phase survived dark check")
    shifted = cost.copy()
    shifted[car.idx[WORDS[2]]] += 1
    require(not protected(twice, shifted, dark, detector), "unequal contrast survived dark check")
    graph = -twice.copy()
    graph.setdiag(0)
    graph.eliminate_zeros()
    require(np.all(graph.data == 1), "stochastic exchange has a negative or repeated hop")
    outside = np.any(pairs // 12 != 13, axis=1)
    paths = bright
    # At the first nonzero exterior amplitude no diagonal step fits. Every shortest walk
    # contributes the same signed (1/2)^length / length! to its amplitude before normalization.
    for length in range(1, 5):
        paths = graph @ paths
        if np.any(paths[outside]):
            square = sum(int(count) ** 2 for count in paths[outside])
            coeff = Fraction(square, 2 * (2**length * factorial(length)) ** 2)
            require(length == 4 and coeff == Fraction(1, 18432), "bright exterior coefficient")
            return {
                "twice_exchange_eigenvalue": 2,
                "contrast_eigenvalue": 3,
                "first_exterior_probability_power": 2 * length,
                "coefficient": str(coeff),
            }
    raise ValueError("bright pair cannot reach exterior in four exchanges")


def read_bins(population, detector, exhaustion):
    missed = np.ones(len(population), dtype=bool)
    missed[detector] = False
    bins = np.r_[population[detector], population[missed].sum() + exhaustion]
    require(
        np.isfinite(bins).all() and bins.min() >= 0 and abs(bins.sum() - 1) < TOL,
        "negative bin or incomplete joint record",
    )
    return bins.tolist()


def appointments(car, cost, detector, weights, exhaustion):
    start = [car.start(word) for word in (WORDS[0], WORDS[2])]
    population = weights**2 @ np.array(start)
    stochastic = {}
    previous = 0
    for time in TIMES:
        # -K has rate 1/2 along every legal jump and zero column sum.
        population = expm_multiply(-(time - previous) * car.kinetic, population)
        stochastic[time] = read_bins(population, detector, exhaustion)
        previous = time
    rows = []
    for kappa in STRENGTHS:
        h = car.kinetic + sparse.diags(kappa * cost, format="csr", dtype=float)
        columns = [m103.chebyshev(h, ket, TIMES, detector, full_at=TIMES)[1] for ket in start]
        for time in TIMES:
            x, z = (column[time] for column in columns)
            require(
                max(
                    abs((x.conj() * x).sum() - 1),
                    abs((z.conj() * z).sum() - 1),
                    abs((x.conj() * z).sum()),
                )
                < 1e-10,
                "unitary flight lost orthogonality",
            )
            px, pz = np.abs(x) ** 2, np.abs(z) ** 2
            bins = {
                "coherent": read_bins(
                    np.abs(weights[0] * x + weights[1] * z) ** 2, detector, exhaustion
                ),
                "dephased": read_bins(
                    weights[0] ** 2 * px + weights[1] ** 2 * pz, detector, exhaustion
                ),
                "stochastic_swap": stochastic[time],
                "unfiltered": read_bins(px, detector, 0.0),
            }
            rows.append({"kappa": kappa, "exposure": time, "bins": bins})
    return rows


def archive_comparison(rows, home, document):
    require(
        document["units_per_probability"] == 10**12
        and document["outcome_order"] == [*map(list, home), "quiet"],
        "archive outcome order",
    )
    records = {(row["kappa"], row["exposure"]): row["bins"] for row in document["appointments"]}
    keys = {(row["kappa"], row["exposure"]) for row in rows}
    require(
        len(document["appointments"]) == len(keys) == 9 and set(records) == keys,
        "archive appointments",
    )
    count, worst = 0, 0.0
    for row in rows:
        archived = records[row["kappa"], row["exposure"]]
        require(set(archived) == set(ARMS), "archive arms")
        for arm in ARMS:
            data = archived[arm]
            require(
                len(data) == 31 and all(type(v) is int and 0 <= v <= 10**12 for v in data),
                "archive integer probability units",
            )
            error = np.max(np.abs(np.asarray(data) / 10**12 - row["bins"][arm]))
            require(
                error <= TOL,
                f"archive mismatch at {row['kappa']}, {row['exposure']}, {arm}: {error}",
            )
            worst = max(worst, float(error))
            count += len(data)
    return {"bins_compared": count, "maximum_absolute_error": worst, "sha256": ARCHIVE_SHA}


def replay(board, archive):
    seams, digest = m103.load_board(board)
    structure, degree, _ = m103.structure(seams)
    require(structure["seams"] == 630, "carrier seam count")
    cut, swaps = installation(seams, degree)
    home = [e for e in seams if e[0] // 12 == e[1] // 12 == 13]
    require(len(home) == 30, "original home detector")
    report = {
        "board_sha256": digest,
        "local_law": local_law(),
        "neutral_edges": sorted(cut),
        "open_contacts": sorted(tuple(sorted((v, EXIT))) for v in (A, C)),
        "setup_swaps": swaps,
        "outcome_order": [*home, "quiet"],
        "home_escape_ranks": {},
    }
    weights, exhaustion, report["source"] = acquisition(degree)
    floor = report["source"]["protected_probability"]
    require(floor > 0.94, "guaranteed joint home weight misses the historical floor")
    for memory, live in (("marked", [e for e in seams if e not in cut]), ("blank", seams)):
        car, pairs, twice, cost = sector(live, degree)
        detector = np.array([car.idx[e] for e in home])
        report["home_escape_ranks"][memory] = home_escape_rank(twice, detector)
        if memory == "marked":
            report["certificate"] = flight_certificate(car, pairs, twice, cost, detector)
            report["certificate"].update(
                connected_ports=len(degree), connected_pair_states=len(pairs)
            )
        else:
            require(np.array_equal(cost, 12 - 2 * car.adj), "blank M10.3 reduction")
        rows = appointments(car, cost, detector, weights, exhaustion)
        report[memory] = rows
    require(report["home_escape_ranks"] == {"marked": 29, "blank": 30}, "home escape ranks")
    for row in report["marked"]:
        bins = row["bins"]
        require(sum(bins["coherent"][:-1]) >= floor - TOL, "protected weight escaped")
        if row["exposure"] == 1:
            require(sum(bins["stochastic_swap"][:-1]) < 0.94, "swap control passes the gate")
    # Dropping the exhausted branch must fail the joint normalization check.
    try:
        read_bins(weights**2, np.array([0, 1]), 0.0)
    except ValueError:
        pass
    else:
        raise ValueError("missing exhaustion escaped normalization")
    report["archive"] = None
    if archive:
        raw = archive.read_bytes()
        require(sha256(raw).hexdigest() == ARCHIVE_SHA, "unexpected archive SHA-256")
        document = json.loads(raw)
        report["archive"] = archive_comparison(report["marked"], home, document)
        # Exchange two assigned probabilities. This preserves normalization and the home sum.
        mutant = document["appointments"][0]["bins"]["coherent"]
        low, high = np.argmin(mutant[:-1]), np.argmax(mutant[:-1])
        mutant[low], mutant[high] = mutant[high], mutant[low]
        try:
            archive_comparison(report["marked"], home, document)
        except ValueError:
            pass
        else:
            raise ValueError("permuted home addresses escaped archive comparison")
    return report


def summary(result):
    receipt = {
        key: result[key]
        for key in (
            "board_sha256",
            "local_law",
            "neutral_edges",
            "open_contacts",
            "setup_swaps",
            "source",
            "certificate",
            "home_escape_ranks",
            "archive",
        )
    }
    rows = []
    for marked, blank in zip(result["marked"], result["blank"], strict=True):
        row = {key: marked[key] for key in ("kappa", "exposure")}
        for memory, appointment in (("marked", marked), ("blank", blank)):
            row[memory] = {arm: round(sum(appointment["bins"][arm][:-1]), 9) for arm in ARMS}
        rows.append(row)
    receipt["home_probabilities"] = rows
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", type=Path)
    parser.add_argument("--archive", type=Path, help="SHA-256 pinned connected_records.json")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--summary", action="store_true", help="print the compact JSON receipt")
    output.add_argument("--json", action="store_true", help="print every ordered bin")
    args = parser.parse_args()
    result = replay(args.board, args.archive)
    if args.json or args.summary:
        print(json.dumps(summary(result) if args.summary else result, indent=1, allow_nan=False))
    else:
        print("Protected joint home weight", result["source"]["protected_probability"])
        print("Source exhaustion", result["source"]["exhaustion_probability"])
        print("Memory  strength  exposure  coherent  dephased  stochastic_swap  unfiltered")
        for memory in ("marked", "blank"):
            for row in result[memory]:
                home = [sum(row["bins"][name][:-1]) for name in ARMS]
                print(memory, row["kappa"], row["exposure"], *(f"{p:.9f}" for p in home))
        if result["archive"]:
            print("Archive comparison", result["archive"])


if __name__ == "__main__":
    main()
