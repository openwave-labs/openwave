"""
XPERIMENT PARAMETERS: Flow Wave

This XPERIMENT showcases:
-
"""

from openwave.common import constants

XPARAMETERS = {
    "meta": {
        "name": "[WIP] Flow Wave",
        "description": "Energy Wave Charging, Propagation and Interaction",
    },
    "camera": {
        "initial_position": [2.00, 1.50, 1.75],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "size": [1e-16, 1e-16, 1e-16],  # m, simulation domain [x, y, z]
        "target_voxels": 1e5,  # Simulation voxel count (impacts performance)
        "slow_mo": constants.EWAVE_FREQUENCY,  # SLOW_MO factor to reduce wave speed / frequency for visibility
        "show_grid": False,  # Toggle to show/hide the voxel grid
        "tick_spacing": 0.25,  # Axis tick marks spacing for position reference
        "color_theme": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
    },
    "ui_defaults": {
        "show_axis": False,  # Toggle to show/hide axis lines
        "flux_films": True,  # Flux Films toggle
        "radius_factor": 1.0,  # Granule radius scaling factor
        "freq_boost": 10.0,  # Frequency boost multiplier
        "amp_boost": 1.0,  # Amplitude boost multiplier
        "paused": False,  # Pause/Start simulation toggle
        "color_palette": 3,  # Color palette list: ironbow (1), blueprint (2), redshift (3), viridis (4)
        "var_amp": False,  # Displacement vs amplitude toggle
    },
    "diagnostics": {
        "wave_diagnostics": False,  # Toggle wave diagnostics (speed & wavelength measurements)
        "export_video": False,  # Toggle frame image export to video directory
        "video_frames": 24,  # Target frame number to stop recording and finalize video export
    },
}
