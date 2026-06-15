"""
XPERIMENT PARAMETERS – GOLDEN ANGLE (SPHERICAL PHYLLOTAXIS) VERSION

K-selektivity test based on the EWT prediction:
- Wave centers arranged by golden angle (~137.5°) on a sphere.
- For K=10 the pattern closes → first stable electron core.
- For K<10 the pattern does not close → unstable.
- For K>10 the energy surplus forces recursive shells (muon/tau).

SPIN NOTE:
Geometric rotation (spin) stabilises the configuration gyroscopically.
To include initial spin, add a VELOCITIES list in wave_centers dict
with tangential velocities for each WC. This requires extending the
motion model; for now we test pure golden-angle placement.
"""

import math
import random
from openwave.common import constants

# ── Universe ────────────────────────────────────────────────────────
UNIVERSE_EDGE = 1e-15              # m
TARGET_VOXELS = 100_000_000

EWAVE_LENGTH   = constants.EWAVE_LENGTH
LOCK_SPACING   = EWAVE_LENGTH / UNIVERSE_EDGE   # λ in normalised coords

# ── Select K ────────────────────────────────────────────────────────
K = 10                     # change to test 2..12
PERTURBATION = 0.1         # fraction of λ, 0.0 = perfect golden‑angle

# ── Geometry ────────────────────────────────────────────────────────
CENTER = (0.5, 0.5, 0.5)
# Radius chosen so that mean inter‑WC distance ≈ λ/2 (the first lock‑in well).
# For K=10 a radius ~0.35*LOCK_SPACING works well. Adjust if needed.
SPHERE_RADIUS = 0.35 * LOCK_SPACING

# ── Helper functions ─────────────────────────────────────────────────
def golden_angle_positions(K, radius, center):
    """
    K points on a sphere via Fibonacci spiral (golden angle phyllotaxis).
    Returns list of (x, y, z) in normalised coords.
    """
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))   # golden angle ~2.399963 rad
    for i in range(K):
        y = 1.0 - (2.0 * i + 1.0) / K         # y uniformly from 1 to -1
        r_y = math.sqrt(1.0 - y * y)          # radius of circle at that y
        theta = phi * i
        x = math.cos(theta) * r_y
        z = math.sin(theta) * r_y
        points.append((
            center[0] + x * radius,
            center[1] + y * radius,
            center[2] + z * radius
        ))
    return points

def apply_perturbation(points, fraction, lock_spacing):
    """Random jitter ±fraction of lock_spacing for each coordinate."""
    jittered = []
    for (x, y, z) in points:
        dx = random.uniform(-fraction, fraction) * lock_spacing
        dy = random.uniform(-fraction, fraction) * lock_spacing
        dz = random.uniform(-fraction, fraction) * lock_spacing
        jittered.append((x + dx, y + dy, z + dz))
    return jittered

# ── Generate positions ──────────────────────────────────────────────
raw_positions = golden_angle_positions(K, SPHERE_RADIUS, CENTER)
POSITIONS = apply_perturbation(raw_positions, PERTURBATION, LOCK_SPACING)

PHASES = [180] * K   # all same phase (electron‑like)

# ── XPARAMETERS ─────────────────────────────────────────────────────
XPARAMETERS = {
    "meta": {
        "X_NAME": f"  /Golden-Angle K={K}",
        "DESCRIPTION": f"K={K} phyllotaxis stability test — "
                       f"{'expect STABLE' if K == 10 else 'expect UNSTABLE'}",
    },
    "camera": {
        "INITIAL_POSITION": [0.94, 0.91, 0.69],
    },
    "universe": {
        "SIZE": [UNIVERSE_EDGE, UNIVERSE_EDGE, UNIVERSE_EDGE],
        "TARGET_VOXELS": TARGET_VOXELS,
    },
    "wave_centers": {
        "COUNT": K,
        "POSITION": POSITIONS,
        "PHASE_OFFSETS_DEG": PHASES,
        "APPLY_MOTION": True,
        # To add initial spin, uncomment and supply tangential velocities:
        # "VELOCITIES": compute_spin_velocities(raw_positions, CENTER, omega=...),
    },
    "ui_defaults": {
        "SHOW_AXIS": False,
        "TICK_SPACING": 0.25,
        "SHOW_GRID": False,
        "SHOW_EDGES": False,
        "FLUX_MESH_PLANES": [0.5, 0.5, 0.5],
        "SHOW_FLUX_MESH": 3,
        "WARP_MESH": 150,
        "PARTICLE_SHELL": True,
        "TIMESTEP": 5.0,
        "PAUSED": False,
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",
        "WAVE_MENU": 4,
    },
    "analytics": {
        "INSTRUMENTATION": True,
        "EXPORT_VIDEO": False,
        "VIDEO_FRAMES": 24,
    },
}