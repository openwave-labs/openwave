"""
XPERIMENT PARAMETERS — Vacuum only (the medium's ground state, NO defect)

Companion to _topo_biaxial1_von.py / _topo_biaxial1_voff.py, but with the particle
REMOVED: this seeds only the uniform medium in its ground state, so you can study
the vacuum by itself, before adding a topological defect.

WHAT THIS SEEDS.  MODE = "vacuum" → seeds.seed_vacuum_M(wave_field, lc_delta):

    M(x) = δ·I + (1−δ)·ẑ⊗ẑ   at EVERY voxel   (uniform, no winding, no core)

i.e. every ellipsoid identical and aligned the same way — the director n̂ = ẑ
everywhere, the "perfectly straight grain" picture (L1). No particle, no charge,
no clock. Press Evolve PDE and nothing should move: a uniform field is already
at the potential's minimum, so it's the stable ground state.

⚠️ UNIAXIAL vs BIAXIAL vacuum (read before editing).  seed_vacuum_M writes the
*uniaxial* spectrum diag(1, δ, δ) (the M5.4 placeholder vacuum; δ = LC_DELTA in
medium.py, default 0.5). The biaxial vacuum diag(1, δ, 0) that _topo_biaxial1_*.py
shows AWAY from the core is NOT seedable as a uniform field today — only the
biaxial *hedgehog* seeder builds diag(1, δ, 0), and it always carries a defect.
A uniform-biaxial-vacuum seeder (diag(1, δ, 0) everywhere) would be a small new
kernel; flag it if you want that exact spectrum here.

VIEW.  Boots PAUSED with the director glyphs on, so you see the uniform aligned
medium (all directors parallel to ẑ). With no gradients, the EM observables are
flat: ∇·n̂ = 0 (no charge, WAVE_MENU 6) and ∇×n̂ = 0 (no B, WAVE_MENU 7) — the
"boring ground state" of L1, by construction.

SANITY CHECKLIST (all healthy = EVERYTHING FLAT): parallel ẑ glyphs; NO δ cross-bar
(uniaxial → degenerate minor axes, no distinct δ-axis); NO E/B glyphs (∇·n̂ = ∇×n̂ = 0);
Evolve PDE moves nothing (ground state); every WAVE_MENU 1–7 mesh blank (all are
deviation/gradient measures, zero in vacuum). Any non-flat reading ⇒ a bug or a stray seed.
"""

UNIVERSE_EDGE = 1e-15  # m
TARGET_VOXELS = 64**3  # ~262k voxels — small for fast smoke-test

TOPOLOGY_SEED = {
    "MODE": "vacuum",  # seed_vacuum_M — uniform M = δI + (1−δ)ẑ⊗ẑ, no defect
    "AUTO_RELAX_STEPS": 0,  # nothing to relax — the uniform field is already the ground state
}


XPARAMETERS = {
    "meta": {
        "X_NAME": "Vacuum, Ground State",
        "DESCRIPTION": "Uniform medium, no defect — M = δI + (1−δ)ẑ⊗ẑ everywhere (the vacuum of L1/L2)",
    },
    "camera": {
        "INITIAL_POSITION": [1.10, 1.46, 0.81],
    },
    "universe": {
        "SIZE": [UNIVERSE_EDGE, UNIVERSE_EDGE, UNIVERSE_EDGE],
        "TARGET_VOXELS": TARGET_VOXELS,
    },
    "ui_defaults": {
        "SHOW_AXIS": False,
        "TICK_SPACING": 0.25,
        "SHOW_GRID": False,
        "SHOW_EDGES": False,
        "VIZ_STRIDE": 2,
        "SHOW_GLYPHS": 3,  # see the uniform director field — all glyphs parallel ẑ
        "FLUX_MESH_PLANES": [0.5, 0.5, 0.5],
        "SHOW_FLUX_MESH": 2,
        "WARP_MESH": 0,
        "SHOW_GRANULES": False,
        "SIM_SPEED": 1.0,
        "PAUSED": True,
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",
        "WAVE_MENU": 4,  # Hamiltonian energy — flat/zero for the uniform vacuum
    },
    "analytics": {
        "INSTRUMENTATION": False,
        "EXPORT_VIDEO": False,
        "VIDEO_FRAMES": 24,
    },
    "topology_seed": TOPOLOGY_SEED,
}
