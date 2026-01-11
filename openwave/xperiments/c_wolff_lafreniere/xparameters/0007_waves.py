"""
XPERIMENT PARAMETERS

This XPERIMENT showcases:
-
"""

UNIVERSE_EDGE = 2e-16  # m, universe edge length in meters
TARGET_VOXELS = 25_000_000  # Target voxel count (impacts performance)

XPARAMETERS = {
    "meta": {
        "X_NAME": f"7 waves, {TARGET_VOXELS/1e6:.0f}M voxels",
        "DESCRIPTION": "Energy Wave Charging, Propagation and Interaction",
    },
    "camera": {
        "INITIAL_POSITION": [1.40, 1.40, 1.20],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "SIZE": [UNIVERSE_EDGE, UNIVERSE_EDGE, UNIVERSE_EDGE],  # m, simulation domain [x, y, z]
        "TARGET_VOXELS": TARGET_VOXELS,  # Simulation voxel count (impacts performance)
    },
    "wave_centers": {
        "COUNT": 1,  # Number of wave-centers for this xperiment
        # Wave-Center positions: normalized coordinates (0-1 range, relative to max universe edge)
        "POSITION": [
            [0.50, 0.50, 0.50],
        ],
        # Phase offsets for each wave-center (integer degrees, converted to radians internally)
        "PHASE_OFFSETS_DEG": [0],
        "APPLY_FORCE": True,  # Toggle to apply force at wave-centers each iteration
    },
    "ui_defaults": {
        "SHOW_AXIS": False,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "SHOW_GRID": False,  # Toggle to show/hide the voxel data-grid
        "SHOW_EDGES": False,  # Toggle to show/hide universe edges
        "FLUX_MESH_PLANES": [0.5, 0.5, 0.5],  # [x, y, z] positions relative to universe size
        "SHOW_FLUX_MESH": 1,  # Flux Mesh toggle, 0: none, 1: xy, 2: xy+xz, 3: xy+xz+yz
        "WARP_MESH": 300,  # Visual warp mesh effect intensity
        "PARTICLE_SHELL": False,  # Toggle to enable/disable particle shell rendering
        "SIM_SPEED": 1.0,  # Simulation speed multiplier
        "PAUSED": False,  # Pause/Start simulation toggle
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
        "COLOR_PALETTE": 1,  # yellowgreen (1), redblue (2), viridis (4), ironbow (5), blueprint (6)
    },
    "analytics": {
        "INSTRUMENTATION": False,  # Toggle data acquisition and analytics
        "EXPORT_VIDEO": False,  # Toggle frame image export to video directory
        "VIDEO_FRAMES": 24,  # Target frame number to stop recording and finalize video export
    },
}
