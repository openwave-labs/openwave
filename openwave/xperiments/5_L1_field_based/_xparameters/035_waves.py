"""
XPERIMENT PARAMETERS

This XPERIMENT showcases:
-
"""

UNIVERSE_EDGE = 1e-15  # m, universe edge length in meters

XPARAMETERS = {
    "meta": {
        "X_NAME": "035 waves, 100M voxels",
        "DESCRIPTION": "Energy Wave Charging, Propagation and Interaction",
    },
    "camera": {
        "INITIAL_POSITION": [0.50, 0.84, 1.80],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "SIZE": [UNIVERSE_EDGE, UNIVERSE_EDGE, UNIVERSE_EDGE],  # m, simulation domain [x, y, z]
        "TARGET_VOXELS": 1e8,  # Simulation voxel count (impacts performance)
    },
    "charging": {
        "STATIC_BOOST": 0.8,  # One-Time charger amplitude boost multiplier
        "DYNAMIC_BOOST": 1.0,  # Dynamic charger amplitude boost multiplier
        "DAMPING_FACTOR": 0.999,  # Dynamic damping factor per frame (0.0-1.0)
    },
    "ui_defaults": {
        "SHOW_AXIS": False,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "SHOW_GRID": False,  # Toggle to show/hide the voxel data-grid
        "FLUX_MESH_SHOW": 1,  # Flux Mesh toggle, 0: none, 1: xy, 2: xy+xz, 3: xy+xz+yz
        "FLUX_MESH_PLANES": [0.5, 0.5, 0.5],  # Normalized positions [x, y, z]
        "SIM_SPEED": 1.0,  # Frequency boost multiplier
        "PAUSED": False,  # Pause/Start simulation toggle
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
        "COLOR_PALETTE": 1,  # Color palette list: redshift (1), ironbow (2), blueprint (3), viridis (4), orange (5)
    },
    "analytics": {
        "INSTRUMENTATION": False,  # Toggle data acquisition and analytics
        "EXPORT_VIDEO": False,  # Toggle frame image export to video directory
        "VIDEO_FRAMES": 24,  # Target frame number to stop recording and finalize video export
    },
}
