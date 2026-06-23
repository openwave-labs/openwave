"""
M5.8.2s, SPIN-1/2 FROM APOLARITY: the double-cover check.

Duda (2026-06-10, coverage-matrix review): "liquid crystals don't need magical
tricks to get spin 1/2; instead there is used symmetric field of e.g.
ellipsoids: after 2pi rotation they rotate by pi, but ellipsoid rotated by pi
is the same ellipsoid."

The check, run on the validated biaxial-hedgehog seed (the 2r/2q stack):
the FRAME W(phi) has period 2pi in the clock phase, but the FIELD
M = W D W^T is apolar (quadratic in the frame axes), so a pi rotation flips
two eigen-axes' signs and returns M EXACTLY. Therefore one full 2pi frame
rotation = TWO complete M-field periods: the observable field oscillates at
omega_M = 2 omega_frame. That factor 2 is the spin-1/2 double cover, and it
is the same machine-exact apolar doubling recorded as the G7 bookkeeping
rule (omega_M = 2 omega_clock, m5_8_2j).

VERDICT (2026-06-10): PASS, machine-exact on both clock planes.
  max|M(phi+pi)   - M(phi)| ~ 1e-16   (the field returns at pi)
  max|M(phi+pi/2) - M(phi)| ~ O(delta..1) (NOT trivial periodicity)
  max|W(phi+pi)   - W(phi)| ~ O(1)    (the frame does NOT return at pi)

USAGE:  python m5_8_2s_spin_half_apolar.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openwave.xperiments.m5_liquid_crystal.research.sandbox_vn.m5_8_2r_electron_id import (  # noqa: E402
    grid_and_seed,
)


def main():
    n = 24
    print("=" * 74)
    print("M5.8.2s - spin-1/2 from apolarity: the double-cover check  [24^3 seed]")
    print("=" * 74)
    print("\n  | clock plane | max|M(φ+π)−M(φ)| | max|M(φ+π/2)−M(φ)| |"
          " max|W(φ+π)−W(φ)| | verdict |")
    print("  | --- | --- | --- | --- | --- |")
    ok_all = True
    for plane, tag in (((2, 3), "twist"), ((1, 2), "tilt")):
        _, W0, M0, _, act = grid_and_seed(n, b=0.0, plane=plane, phi=0.3)
        _, Wpi, Mpi, _, _ = grid_and_seed(n, b=0.0, plane=plane, phi=0.3 + np.pi)
        _, _, Mh, _, _ = grid_and_seed(n, b=0.0, plane=plane,
                                       phi=0.3 + np.pi / 2)
        dM_pi = float(np.abs(Mpi - M0)[act].max())
        dM_h = float(np.abs(Mh - M0)[act].max())
        dW_pi = float(np.abs(Wpi - W0)[act].max())
        ok = dM_pi < 1e-12 and dM_h > 1e-3 and dW_pi > 1e-3
        ok_all = ok_all and ok
        print(f"  | {tag} | {dM_pi:.2e} | {dM_h:.2e} | {dW_pi:.2e} |"
              f" {'PASS ✓' if ok else 'FAIL ✗'} |")
    print("\n  READ: the field M returns EXACTLY at a π rotation while the frame")
    print("  needs 2π, so one frame revolution is TWO field periods: the spin-1/2")
    print("  double cover, with no belt-trick machinery required (Duda). This is")
    print("  the same factor 2 as the G7 apolar-doubling rule (ω_M = 2ω_clock).")
    print("=" * 74)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
