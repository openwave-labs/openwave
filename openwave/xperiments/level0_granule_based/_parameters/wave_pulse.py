"""
XPERIMENT PARAMETERS: Wave Pulse

Demonstrates radiation from a single wave source.
A single source generates spherical longitudinal waves that propagate radially outward.

This XPERIMENT showcases:
- Single wave source at center
- Wave pulse radiation
- Wave diagnostics enabled for speed and wavelength measurements
"""

PARAMETERS = {
    "meta": {
        "name": "Wave Pulse",
        "description": "Single source radiation with wave diagnostics",
    },
    "camera": {
        "initial_position": [1.35, 0.91, 0.68],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "size": [1e-16, 1e-16, 1e-16],  # m, simulation domain [x, y, z] (cubic)
        "tick_spacing": 0.25,  # Axis tick marks spacing for position reference
        "color_theme": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
    },
    "wave_sources": {
        "count": 1,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Single source at center for pure spherical wave pulse radiation
        "positions": [[0.5, 0.5, 0.5]],  # Wave Source position - Center
        # Phase offset in degrees (0° = in phase with base frequency)
        "phase_offsets_deg": [0],
    },
    "ui_defaults": {
        "show_axis": True,  # Toggle to show/hide axis lines
        "block_slice": False,  # Block-slicing toggle
        "show_sources": False,  # Toggle to show/hide wave source markers
        "radius_factor": 0.1,  # Granule radius scaling factor
        "freq_boost": 10.0,  # Frequency boost multiplier
        "amp_boost": 5.0,  # Amplitude boost multiplier
        "paused": False,  # Pause/Start simulation toggle
        "granule_type": False,  # Granule type color
        "ironbow": False,  # Ironbow color scheme toggle
        "blueprint": False,  # Blueprint color scheme toggle
        "var_displacement": True,  # Displacement vs amplitude toggle
    },
    "diagnostics": {
        "wave_diagnostics": True,  # Toggle wave diagnostics (speed & wavelength measurements)
        "export_video": False,  # Toggle frame image export to video directory
        "video_frames": 24,  # Target frame number to stop recording and finalize video export
    },
}
