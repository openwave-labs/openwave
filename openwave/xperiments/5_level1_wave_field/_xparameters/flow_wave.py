"""
XPERIMENT PARAMETERS: Flow Wave

This XPERIMENT showcases:
-
"""

XPARAMETERS = {
    "meta": {
        "name": "Flow Wave",
        "description": "Energy Wave Charging, Propagation and Interaction",
    },
    "camera": {
        "initial_position": [2.00, 1.50, 1.75],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "size": [1e-16, 1e-16, 1e-16],  # m, simulation domain [x, y, z]
        "tick_spacing": 0.25,  # Axis tick marks spacing for position reference
        "color_theme": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
    },
    "ui_defaults": {
        "show_axis": False,  # Toggle to show/hide axis lines
        "flux_detector": False,  # Flux detector toggle
        "radius_factor": 1.0,  # Granule radius scaling factor
        "freq_boost": 10.0,  # Frequency boost multiplier
        "amp_boost": 1.0,  # Amplitude boost multiplier
        "paused": False,  # Pause/Start simulation toggle
        "granule_type": False,  # Granule type color
        "ironbow": False,  # Ironbow color scheme toggle
        "blueprint": False,  # Blueprint color scheme toggle
        "var_displacement": True,  # Displacement vs amplitude toggle
    },
    "diagnostics": {
        "wave_diagnostics": False,  # Toggle wave diagnostics (speed & wavelength measurements)
        "export_video": False,  # Toggle frame image export to video directory
        "video_frames": 24,  # Target frame number to stop recording and finalize video export
    },
}
