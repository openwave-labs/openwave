"""
Stability Analysis for Wave Equation Simulation.
"""

import numpy as np
import taichi as ti

from openwave.common import constants

import openwave.spacetime.L1_field_grid as data_grid

ti.init(arch=ti.gpu)

UNIVERSE_SIZE = [
    6e-15,
    6e-15,
    6e-15,
]  # m, simulation domain [x, y, z] dimensions (can be asymmetric)

wave_field = data_grid.WaveField(UNIVERSE_SIZE, target_voxels=3.5e8)


# ================================================================
# Stability Analysis - Wave Equation CFL Condition
# ================================================================

print("\n" + "=" * 64)
print("STABILITY ANALYSIS - WAVE EQUATION")
print("=" * 64)

# Wave propagation parameters
c = constants.EWAVE_SPEED  # m/s, speed of light
dx = wave_field.voxel_edge  # m, voxel edge length
ewave_freq = constants.EWAVE_FREQUENCY  # Hz

# CFL (Courant-Friedrichs-Lewy) stability condition for 3D wave equation
# For 6-connectivity (face neighbors only): dt ≤ dx / (c√3)
dt_critical = dx / (c * np.sqrt(3))  # Maximum stable timestep

# Frame timestep (typical screen refresh rates)
dt_frame_60fps = 1 / 60  # 60 FPS
dt_frame_30fps = 1 / 30  # 30 FPS

print(f"\nVoxel count: {wave_field.voxel_count:.1e}")
print(f"Voxel edge (dx_am): {dx / constants.ATTOMETER:.1f} am")
print(f"Wave speed (c): {c:.2e} m/s")
print(f"Energy-wave frequency: {ewave_freq:.2e} Hz")

# SLO_MO mitigation
c_slowed = c / constants.EWAVE_FREQUENCY  # Wave speed after SLO_MO to 1Hz
freq_slowed = ewave_freq / constants.EWAVE_FREQUENCY  # Frequency after SLO_MO to 1Hz
dt_critical_slowed = dx / (c_slowed * np.sqrt(3))  # CFL with slowed wave speed

print(f"\nCFL Stability (WITHOUT SLO_MO):")
print(f"  Critical timestep: {dt_critical:.2e} s")
print(f"  Frame timestep (60 FPS): {dt_frame_60fps:.3f} s")
print(f"  Frame timestep (30 FPS): {dt_frame_30fps:.3f} s")
print(f"  Violation ratio (60 FPS): {dt_frame_60fps / dt_critical:.2e}×")
print(f"  Violation ratio (30 FPS): {dt_frame_30fps / dt_critical:.2e}×")

print(f"\nCFL Stability (WITH SLO_MO = {constants.EWAVE_FREQUENCY:.2e}):")
print(f"  Slowed wave speed: {c_slowed:.2e} m/s")
print(f"  Slowed frequency: {freq_slowed:.2e} Hz")
print(f"  Critical timestep (slowed): {dt_critical_slowed:.3f} s")
print(f"  Frame timestep (60 FPS): {dt_frame_60fps:.3f} s")
print(f"  Frame timestep (30 FPS): {dt_frame_30fps:.3f} s")

# Check stability with SLO_MO
if dt_frame_60fps <= dt_critical_slowed:
    print(f"\n✓ STABLE at 60 FPS (dt={dt_frame_60fps:.3f} s ≤ dt_crit={dt_critical_slowed:.3f} s)")
    safety_margin_60 = dt_critical_slowed / dt_frame_60fps
    print(f"  Safety margin: {safety_margin_60:.2f}× (CFL factor = {1/safety_margin_60:.3f})")
else:
    violation_60 = dt_frame_60fps / dt_critical_slowed
    print(f"\n❌ UNSTABLE at 60 FPS!")
    print(f"  Violation: {violation_60:.3f}× over critical timestep")
    print(f"  Need SLO_MO ≥ {constants.EWAVE_FREQUENCY * violation_60:.3f} for stability")

if dt_frame_30fps <= dt_critical_slowed:
    print(f"\n✓ STABLE at 30 FPS (dt={dt_frame_30fps:.3f} s ≤ dt_crit={dt_critical_slowed:.3f} s)")
    safety_margin_30 = dt_critical_slowed / dt_frame_30fps
    print(f"  Safety margin: {safety_margin_30:.3f}× (CFL factor = {1/safety_margin_30:.3f})")
else:
    violation_30 = dt_frame_30fps / dt_critical_slowed
    print(f"\n❌ UNSTABLE at 30 FPS!")
    print(f"  Violation: {violation_30:.2e}× over critical timestep")
    print(f"  Need SLO_MO ≥ {constants.EWAVE_FREQUENCY * violation_30:.2e} for stability")

# Summary and recommendations
print(f"\nMitigation Strategy:")
print(f"  1. Apply SLO_MO to wave speed: c_slowed = c / SLO_MO")
print(f"  2. Use SIM_SPEED parameter for visualization control")
print(f"  3. Monitor CFL factor: (c·dt/dx)² should be ≤ 1/3 for 3D")
print(f"  4. Use fixed timestep strategy (not elapsed time)")

print("=" * 64)
