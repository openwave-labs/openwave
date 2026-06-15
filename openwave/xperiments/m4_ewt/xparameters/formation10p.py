"""
XPERIMENT PARAMETERS

K=2 through K=10 progressive particle formation test.
Verifies prediction: K=2..9 are UNSTABLE (decay/fly apart),
K=10 is the first stable standalone particle (electron tetrahedron).

Switch K by uncommenting the desired configuration below.
Each K places WCs at λ/2 spacing in the simplest possible geometry:
  K=2:  line (2 WCs along x-axis)
  K=3:  equilateral triangle
  K=4:  regular tetrahedron (4 vertices)
  K=5:  trigonal bipyramid
  K=6:  octahedron
  K=7:  pentagonal bipyramid
  K=8:  cube (dual tetrahedra — should be neutral/unstable differently)
  K=9:  tricapped trigonal prism
  K=10: 1-3-6 tetrahedron (STABLE — the electron)
"""

from openwave.common import constants
from openwave.xperiments.m4_ewt.xparameters.formation02 import generate_K_positions

UNIVERSE_EDGE = 1e-15  # m, universe edge length in meters
TARGET_VOXELS = 100_000_000  # Target voxel count (impacts performance)

# Lock spacing: λ in normalized coords (first lock-in well for Combined W-L)
# The envelope 2|sin(kr/2)|/r has zeros at r=nλ — these are the energy minima
# where same-phase WCs lock in. (Wolff-original sinc had wells at λ/2.)
EWAVE_LENGTH = constants.EWAVE_LENGTH
LOCK_SPACING = EWAVE_LENGTH / UNIVERSE_EDGE

# ════════════════════════════════════════════════════════════════════════════
# SELECT K VALUE HERE
# ════════════════════════════════════════════════════════════════════════════
K = 10
# K = 2    # Line — EXPECT: STABLE
# K = 3    # Triangle — EXPECT: unstable
# K = 4    # Tetrahedron (4) — EXPECT: unstable
# K = 5    # Trigonal bipyramid — EXPECT: unstable
# K = 6    # Octahedron — EXPECT: unstable
# K = 7    # Pentagonal bipyramid — EXPECT: unstable
# K = 8    # Cube (dual tetra) — EXPECT: STABLE (neutral)
# K = 9    # Tricapped prism — EXPECT: unstable
# K = 10   # 1-3-6 tetrahedron — EXPECT: STABLE (electron)

# Perturbation: shift each WC by random ±PERTURBATION fraction of λ.
# At 0.0: perfect lattice (all K stable). At 0.2+: real test.
PERTURBATION = 0.1  # fraction of λ (0.0 = perfect, 0.3 = 30% random displacement)

POSITIONS = generate_K_positions(
    UNIVERSE_EDGE, K, center=(0.5, 0.5, 0.5), rotation=(45, 45, 45), perturbation=PERTURBATION
)
PHASES = [0] * K  # all same phase (electron-like)

XPARAMETERS = {
    "meta": {
        "X_NAME": f"  /Positron (K={K})",
        "DESCRIPTION": f"K={K} stability test — {'STABLE' if K == 10 else 'expect UNSTABLE'}",
    },
    "camera": {
        "INITIAL_POSITION": [0.29, 1.28, 0.22],  # [x, y, z] in normalized coordinates
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
    },
    "ui_defaults": {
        "SHOW_AXIS": False,
        "TICK_SPACING": 0.25,
        "SHOW_GRID": False,
        "SHOW_EDGES": False,
        "FLUX_MESH_PLANES": [0.5, 0.5, 0.5],
        "SHOW_FLUX_MESH": 1,
        "WARP_MESH": 150,
        "SHOW_GRANULES": False,  # Toggle to show/hide granule particles (rendered as points)
        "PARTICLE_SHELL": True,
        "TIMESTEP": 5.0,
        "PAUSED": False,
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",
        "WAVE_MENU": 4,
    },
    "analytics": {
        "INSTRUMENTATION": False,
        "EXPORT_VIDEO": False,
        "VIDEO_FRAMES": 24,
    },
}
