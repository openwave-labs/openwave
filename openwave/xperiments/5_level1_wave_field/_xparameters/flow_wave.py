"""
XPERIMENT PARAMETERS: Flow Wave

This XPERIMENT showcases:
-
"""

from openwave.common import constants

UNIVERSE_EDGE = 6 * constants.EWAVE_LENGTH  # m, universe edge length in meters

XPARAMETERS = {
    "meta": {
        "X_NAME": "[WIP] Flow Wave",
        "DESCRIPTION": "Energy Wave Charging, Propagation and Interaction",
    },
    "camera": {
        "INITIAL_POSITION": [1.50, 1.50, 1.11],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "SIZE": [UNIVERSE_EDGE, UNIVERSE_EDGE, UNIVERSE_EDGE],  # m, simulation domain [x, y, z]
        "TARGET_VOXELS": 1e6,  # Simulation voxel count (impacts performance)
        "SLO_MO": constants.EWAVE_FREQUENCY,  # SLO_MO factor to reduce wave speed / frequency for visibility
    },
    "ui_defaults": {
        "SHOW_AXIS": False,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "SHOW_GRID": False,  # Toggle to show/hide the voxel data-grid
        "FLUX_MESH_OPTION": 3,  # Flux Mesh toggle, 0: none, 1: xy, 2: xy+xz, 3: xy+xz+yz
        "SIM_SPEED": 1.0,  # Frequency boost multiplier
        "AMP_BOOST": 1.0,  # Amplitude boost multiplier
        "PAUSED": True,  # Pause/Start simulation toggle
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
        "COLOR_PALETTE": 2,  # Color palette list: ironbow (1), blueprint (2), redshift (3), viridis (4)
        "VAR_AMP": False,  # Displacement vs amplitude toggle
    },
    "diagnostics": {
        "WAVE_DIAGNOSTICS": False,  # Toggle wave diagnostics (speed & wavelength measurements)
        "EXPORT_VIDEO": False,  # Toggle frame image export to video directory
        "VIDEO_FRAMES": 24,  # Target frame number to stop recording and finalize video export
    },
}
