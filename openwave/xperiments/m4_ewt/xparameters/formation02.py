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

import numpy as np
from openwave.common import constants

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
K = 2
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


def tetrahedron_10(univ_edge, center=(0.5, 0.5, 0.5), rotation=(0, 0, 0)):
    """Generate 10 positions in 1-3-6 tetrahedral arrangement.

    Args:
        center: (x, y, z) center position in normalized coords
        rotation: (rx, ry, rz) rotation in degrees around x, y, z axes
    """
    # Lock spacing calibrated to lambda (first standing wave node)
    # In normalized coords: lambda / UNIVERSE_EDGE
    LOCK_SPACING = constants.EWAVE_LENGTH / univ_edge  # lambda in normalized coords

    layer_h = LOCK_SPACING * np.sqrt(2 / 3)  # tetrahedral layer spacing
    cx, cy, cz = center

    Rb = LOCK_SPACING * 2 / np.sqrt(3)  # base vertex radius
    Rm = LOCK_SPACING / np.sqrt(3)  # midpoint/middle layer radius

    angles_v = np.radians([90, 210, 330])  # vertex angles
    angles_m = np.radians([30, 150, 270])  # midpoint angles

    # Generate positions relative to center (0,0,0)
    local_positions = [
        # Base: 3 vertices + 3 midpoints
        *[[Rb * np.cos(a), Rb * np.sin(a), 0] for a in angles_v],
        *[[Rm * np.cos(a), Rm * np.sin(a), 0] for a in angles_m],
        # Middle: 3 elements
        *[[Rm * np.cos(a), Rm * np.sin(a), layer_h] for a in angles_v],
        # Apex: 1 element
        [0, 0, 2 * layer_h],
    ]

    # Apply rotation if specified
    rx, ry, rz = np.radians(rotation)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    return [[cx + p[0], cy + p[1], cz + p[2]] for p in (R @ np.array(local_positions).T).T]


def generate_K_positions(
    univ_edge, K, center=(0.5, 0.5, 0.5), rotation=(0, 0, 0), perturbation=0.0
):
    """Generate K WC positions in a compact geometry at λ spacing.

    Uses simple symmetric arrangements. Not all are the natural geometry
    (only K=10 has the 1-3-6 tetrahedron), but they test whether arbitrary
    K-body arrangements are stable or decay.
    """
    cx, cy, cz = center
    s = LOCK_SPACING  # shorthand in normalized coords

    if K == 2:
        # Line along x
        positions = [[cx - s / 2, cy, cz], [cx + s / 2, cy, cz]]

    elif K == 3:
        # Equilateral triangle in xy plane
        angles = np.radians([90, 210, 330])
        r = s / np.sqrt(3)  # circumradius for edge = s
        positions = [[cx + r * np.cos(a), cy + r * np.sin(a), cz] for a in angles]

    elif K == 4:
        # Regular tetrahedron (4 vertices)
        h = s * np.sqrt(2 / 3)
        r = s / np.sqrt(3)
        positions = [
            [cx, cy + r, cz],
            [cx - s / 2, cy - r / 2, cz],
            [cx + s / 2, cy - r / 2, cz],
            [cx, cy, cz + h],
        ]

    elif K == 5:
        # Trigonal bipyramid: triangle + 2 poles
        angles = np.radians([90, 210, 330])
        r = s / np.sqrt(3)
        positions = [[cx + r * np.cos(a), cy + r * np.sin(a), cz] for a in angles]
        positions.append([cx, cy, cz + s / 2])  # top pole
        positions.append([cx, cy, cz - s / 2])  # bottom pole

    elif K == 6:
        # Octahedron: 6 vertices along axes
        d = s / np.sqrt(2)
        positions = [
            [cx + d, cy, cz],
            [cx - d, cy, cz],
            [cx, cy + d, cz],
            [cx, cy - d, cz],
            [cx, cy, cz + d],
            [cx, cy, cz - d],
        ]

    elif K == 7:
        # Pentagonal bipyramid: pentagon + 2 poles
        angles = np.radians([i * 72 for i in range(5)])
        r = s / (2 * np.sin(np.pi / 5))  # circumradius for edge = s
        positions = [[cx + r * np.cos(a), cy + r * np.sin(a), cz] for a in angles]
        positions.append([cx, cy, cz + s / 2])
        positions.append([cx, cy, cz - s / 2])

    elif K == 8:
        # Cube: 8 vertices (dual tetrahedra — K=8 muon neutrino)
        d = s / 2
        positions = [
            [cx + d, cy + d, cz + d],
            [cx + d, cy + d, cz - d],
            [cx + d, cy - d, cz + d],
            [cx + d, cy - d, cz - d],
            [cx - d, cy + d, cz + d],
            [cx - d, cy + d, cz - d],
            [cx - d, cy - d, cz + d],
            [cx - d, cy - d, cz - d],
        ]

    elif K == 9:
        # Tricapped trigonal prism: 6 (trigonal prism) + 3 caps
        angles = np.radians([90, 210, 330])
        r = s / np.sqrt(3)
        h = s / 2
        positions = []
        for z_off in [-h, h]:
            for a in angles:
                positions.append([cx + r * np.cos(a), cy + r * np.sin(a), cz + z_off])
        cap_angles = np.radians([30, 150, 270])
        cap_r = r * 1.5
        for a in cap_angles:
            positions.append([cx + cap_r * np.cos(a), cy + cap_r * np.sin(a), cz])

    elif K == 10:
        # 1-3-6 tetrahedron
        positions = tetrahedron_10(univ_edge=UNIVERSE_EDGE, center=center, rotation=rotation)

    else:
        raise ValueError(f"K={K} not implemented")

    # Apply perturbation (random displacement from ideal positions)
    if perturbation > 0:
        rng = np.random.default_rng(seed=42)  # reproducible
        noise = rng.uniform(-perturbation, perturbation, size=(K, 3)) * LOCK_SPACING
        positions = [[p[0] + n[0], p[1] + n[1], p[2] + n[2]] for p, n in zip(positions, noise)]

    return positions


POSITIONS = generate_K_positions(
    UNIVERSE_EDGE, K, center=(0.5, 0.5, 0.5), rotation=(0, 0, 0), perturbation=PERTURBATION
)
PHASES = [180] * K  # all same phase (electron-like)

XPARAMETERS = {
    "meta": {
        "X_NAME": f"  /Particle (K={K})",
        "DESCRIPTION": f"K={K} stability test — {'STABLE' if K == 10 else 'expect UNSTABLE'}",
    },
    "camera": {
        "INITIAL_POSITION": [0.36, 1.20, 0.75],
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
        "WAVE_MENU": 1,
    },
    "analytics": {
        "INSTRUMENTATION": False,
        "EXPORT_VIDEO": False,
        "VIDEO_FRAMES": 24,
    },
}
