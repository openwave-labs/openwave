import numpy as np

from openwave.common import constants, equations

import openwave.spacetime.L0_granule_grid as data_grid


# From constants
EWAVE_FREQUENCY = constants.EWAVE_FREQUENCY

# Current simulation parameters
UNIVERSE_EDGE = 4 * constants.EWAVE_LENGTH  # m, simulation domain, edge length of cubic universe
TARGET_GRANULES = 1e6  # target granule count
SLOW_MO = constants.EWAVE_FREQUENCY  # slows frequency down to 1Hz for human visibility
STIFFNESS = 1e-13  # N/m (already reduced!)

# Calculate granule properties (simplified BCC lattice estimate)
granules_per_edge = TARGET_GRANULES ** (1 / 3) * 0.8  # 80% fill factor
unit_cell_edge = UNIVERSE_EDGE / granules_per_edge
# unit_cell_edge = 2 * np.e * constants.PLANCK_LENGTH  # ~5.4e-35 m, override to Planck scale
granule = data_grid.BCCGranule(unit_cell_edge, UNIVERSE_EDGE)

# Spring-mass system natural frequency
frequency = equations.compute_natural_frequency(STIFFNESS, granule.mass)  # Hz
omega = 2 * np.pi * frequency  # rad/s
period = 1 / frequency

# Critical timestep for stability (explicit integrators)
# For semi-implicit Euler: dt < 2/omega
dt_critical = 2 / omega

# Typical frame time at 30 FPS
dt_frame = 1 / 30  # 33ms

# How many substeps needed?
substeps_needed = dt_frame / dt_critical

print("\n" + "=" * 64)
print("STABILITY ANALYSIS")
print("=" * 64)

print(f"Unit Cell edge: {unit_cell_edge:.2e} m")
print(f"Granule radius: {granule.radius:.2e} m")
print(f"Granule mass: {granule.mass:.2e} kg")

print(f"\nCurrent Spring stiffness: {STIFFNESS:.2e} N/m")
print(f"Natural frequency: {frequency:.2e} Hz (period: {period:.2e} s)")
print(f"Energy-Wave frequency: {EWAVE_FREQUENCY:.2e} Hz")
print(f"Energy-Wave slowed: {EWAVE_FREQUENCY / SLOW_MO:.2e} Hz")
print(
    f"Stiffness to match: {equations.compute_stiffness_from_frequency(EWAVE_FREQUENCY, granule.mass):.2e} N/m"
)

print(f"\nCritical timestep (stability limit): {dt_critical:.2e} s")
print(f"Frame timestep (30 FPS): {dt_frame:.2e} s")
print(f"Substeps needed for stability: {substeps_needed:.0f}")
print(f"You're using: 200 substeps")

if substeps_needed > 200:
    print(
        f"\n❌ UNSTABLE! Need {substeps_needed:.0e} substeps ({30*substeps_needed:.0e} iterations/second), only using 200 ({30*200:.0e} i/s)"
    )
else:
    print(f"\n✓ Should be stable")

# What stiffness would work with 200 substeps?
dt_sub = dt_frame / 200
omega_max = 2 / dt_sub
stiffness_max = (omega_max**2) * granule.mass
print(f"Maximum stiffness: {stiffness_max:.2e} N/m, for 200 substeps to be stable")
print(f"=" * 64)
