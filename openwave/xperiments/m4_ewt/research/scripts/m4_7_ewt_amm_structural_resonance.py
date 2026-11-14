#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From manuscript version: 5.2.4 or leter
Nodal shell hierarchy and Fibonacci latch generator for the EWT lepton model.

Generates:
    K_n        total nodal count for generation n
    Delta K_n  nodal increment for generation n
    L_n        Fibonacci latch for generation n

The operator is:
    Delta K_k = round(10^k * pi^2)       for k = 0 (electron core)
    Delta K_k = round(10^k * 2*pi^2)     for k >= 1 (shells)

The Fibonacci latch is F_{4n-3}, advancing by 2^2 = 4 per generation.

Also prints the closure criterion K_n / L_n^2 > 1
(tab:closure_criterion in the manuscript).
"""

import math

PI = math.pi


# ----------------------------------------------------------------------
# Fibonacci and latch
# ----------------------------------------------------------------------

def _fib(n: int) -> int:
    """Standard Fibonacci number F_n, with F_1 = F_2 = 1."""
    if n < 1:
        raise ValueError("Fibonacci index must be >= 1")
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a


def get_L_invariant(generation: int) -> int:
    """Fibonacci latch for generation n: F_{4n-3}."""
    if generation < 1:
        raise ValueError("Generation number must be >= 1")
    return _fib(4 * generation - 3)


# ----------------------------------------------------------------------
# Nodal increments and total
# ----------------------------------------------------------------------

def get_delta_K(k: int) -> int:
    """Nodal increment for shell index k."""
    if k < 0:
        raise ValueError("Shell index must be >= 0")
    if k == 0:
        return round(10 ** k * PI ** 2)
    return round(10 ** k * 2.0 * PI ** 2)


def get_AMMi_K(n: int) -> int:
    """Total nodal count for generation n (sum of increments up to n-1)."""
    if n < 1:
        raise ValueError("Generation number must be >= 1")
    return sum(get_delta_K(k) for k in range(n))


# ----------------------------------------------------------------------
# Main: print the hierarchy
# ----------------------------------------------------------------------

GENERATIONS = {
    1: "Electron (core)",
    2: "Muon (shell 1)",
    3: "Tau (shell 2)",
    4: "4th gen. (hypothetical)",
}


def print_hierarchy_table():
    """Print the nodal shell parameters (tab:shell_parameters_verified)."""
    print()
    print(f"{'Lepton Shell':<24} {'Operator':<20} "
          f"{'Delta K':>8} {'Total K_n':>10} {'L_dim':>6}")
    print("-" * 76)

    rows = [
        (1, "Core (Electron)",     "10^0 * pi^2"),
        (2, "Shell 1 (Muon)",      "10^1 * 2pi^2"),
        (3, "Shell 2 (Tau)",       "10^2 * 2pi^2"),
        (4, "Shell 3 (4th gen.)*", "10^3 * 2pi^2"),
    ]

    for n, label, op in rows:
        k = n - 1
        dK = get_delta_K(k)
        K = get_AMMi_K(n)
        L = get_L_invariant(n)
        print(f"{label:<24} {op:<20} {dK:>8} {K:>10} {L:>6}")

    print()
    print("* Hypothetical.")
    print()


def print_closure_table():
    """Print the closure criterion (tab:closure_criterion)."""
    print("=" * 76)
    print("CLOSURE CRITERION: NODE DENSITY AT THE SHELL INTERFACE")
    print("=" * 76)
    print()
    print(f"{'Shell':<24} {'L_n':>6} {'L_n^2':>8} {'K_n':>8} "
          f"{'K_n/L_n^2':>10} {'Closed':>8}")
    print("-" * 76)

    for n, label in GENERATIONS.items():
        L = get_L_invariant(n)
        K = get_AMMi_K(n)
        L_sq = L * L
        ratio = K / L_sq
        closed = "yes" if ratio > 1.0 else "NO"

        if n == 4:
            label = label.replace("(hypothetical)", "*(hypo.)")

        print(f"{label:<24} {L:>6} {L_sq:>8} {K:>8} "
              f"{ratio:>10.3f} {closed:>8}")

    print("-" * 76)
    print("  Criterion: K_n / L_n^2 > 1  (node density at interface)")
    print()


def main():
    print_hierarchy_table()
    print_closure_table()


if __name__ == "__main__":
    main()