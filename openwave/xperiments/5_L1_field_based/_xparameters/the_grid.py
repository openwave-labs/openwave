"""
XPERIMENT PARAMETERS: Wave Field Grid of Voxels

This XPERIMENT showcases:
-
"""

from openwave.common import constants

UNIVERSE_EDGE = 1e-16  # m, universe edge length in meters

XPARAMETERS = {
    "meta": {
        "X_NAME": "The Grid",
        "DESCRIPTION": "Wave Field Grid of Voxels",
    },
    "camera": {
        "INITIAL_POSITION": [2.50, 1.50, 1.50],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "SIZE": [UNIVERSE_EDGE, UNIVERSE_EDGE, UNIVERSE_EDGE],  # m, simulation domain [x, y, z]
        "TARGET_VOXELS": 1e3,  # Simulation voxel count (impacts performance)
    },
    "ui_defaults": {
        "SHOW_AXIS": True,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "SHOW_GRID": True,  # Toggle to show/hide the voxel data-grid
        "FLUX_MESH_SHOW": 0,  # Flux Mesh toggle, 0: none, 1: xy, 2: xy+xz, 3: xy+xz+yz
        "FLUX_MESH_PLANES": [0.5, 0.5, 0.5],  # Normalized positions of flux mesh planes [x, y, z]
        "SIM_SPEED": 1.0,  # Frequency boost multiplier
        "PAUSED": True,  # Pause/Start simulation toggle
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
        "COLOR_PALETTE": 1,  # Color palette list: redshift (1), ironbow (2), blueprint (3), viridis (4)
    },
    "analytics": {
        "INSTRUMENTATION": False,  # Toggle data collection
        "EXPORT_VIDEO": False,  # Toggle frame image export to video directory
        "VIDEO_FRAMES": 24,  # Target frame number to stop recording and finalize video export
    },
}
