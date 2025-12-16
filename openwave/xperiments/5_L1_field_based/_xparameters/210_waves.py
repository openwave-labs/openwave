"""
XPERIMENT PARAMETERS

This XPERIMENT showcases:
-
"""

UNIVERSE_EDGE = 6e-15  # m, universe edge length in meters

XPARAMETERS = {
    "meta": {
        "X_NAME": "210 waves, 350M voxels",
        "DESCRIPTION": "Energy Wave Charging, Propagation and Interaction",
    },
    "camera": {
        "INITIAL_POSITION": [1.40, 1.40, 1.20],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "SIZE": [UNIVERSE_EDGE, UNIVERSE_EDGE, UNIVERSE_EDGE],  # m, simulation domain [x, y, z]
        "TARGET_VOXELS": 3.5e8,  # Simulation voxel count (impacts performance)
    },
    "charging": {
        "STATIC_BOOST": 1.00,  # One-Time charger amplitude boost multiplier
    },
    "ui_defaults": {
        "SHOW_AXIS": False,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "SHOW_GRID": False,  # Toggle to show/hide the voxel data-grid
        "SHOW_EDGES": False,  # Toggle to show/hide universe edges
        "SHOW_FLUX_MESH": 1,  # Flux Mesh toggle, 0: none, 1: xy, 2: xy+xz, 3: xy+xz+yz
        "FLUX_MESH_PLANES": [0.5, 0.5, 0.5],  # [x, y, z] positions relative to universe size
        "SIM_SPEED": 1.0,  # Simulation speed multiplier
        "PAUSED": False,  # Pause/Start simulation toggle
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
        "COLOR_PALETTE": 1,  # redblue (1), ironbow (3), blueprint (5), viridis (4)
    },
    "analytics": {
        "INSTRUMENTATION": False,  # Toggle data acquisition and analytics
        "EXPORT_VIDEO": False,  # Toggle frame image export to video directory
        "VIDEO_FRAMES": 24,  # Target frame number to stop recording and finalize video export
    },
}
