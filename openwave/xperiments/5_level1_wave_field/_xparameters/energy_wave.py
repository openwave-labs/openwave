"""
XPERIMENT PARAMETERS: Energy Wave

This XPERIMENT showcases:
-
"""

from openwave.common import constants

XPARAMETERS = {
    "meta": {
        "X_NAME": "[WIP] Energy Wave",
        "DESCRIPTION": "Energy Wave Charging, Propagation and Interaction",
    },
    "camera": {
        "INITIAL_POSITION": [1.50, 1.50, 1.11],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "SIZE": [1e-16, 1e-16, 1e-16],  # m, simulation domain [x, y, z]
        "TARGET_VOXELS": 1e8,  # Simulation voxel count (impacts performance)
        "SLOW_MO": constants.EWAVE_FREQUENCY,  # SLOW_MO factor to reduce wave speed / frequency for visibility
    },
    "ui_defaults": {
        "SHOW_AXIS": True,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "SHOW_GRID": False,  # Toggle to show/hide the voxel grid
        "SHOW_FLUX_MESH": True,  # Flux Mesh toggle
        "RADIUS_FACTOR": 1.0,  # Granule radius scaling factor
        "FREQ_BOOST": 10.0,  # Frequency boost multiplier
        "AMP_BOOST": 1.0,  # Amplitude boost multiplier
        "PAUSED": False,  # Pause/Start simulation toggle
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
