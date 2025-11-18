"""
XPERIMENT PARAMETERS: 3D Spherical Wave

Demonstrates wave interference from multiple sources in the lattice.
Each source generates spherical longitudinal waves that superpose at each granule,
creating constructive and destructive interference patterns.

This XPERIMENT showcases:
- Single wave source at center
- Pure spherical wave propagation
- Radial wave patterns
"""

XPARAMETERS = {
    "meta": {
        "name": "3D Spherical Wave",
        "description": "Single source spherical wave demonstrating radial propagation",
    },
    "camera": {
        "initial_position": [0.97, 2.06, 0.82],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "size": [1e-16, 1e-16, 1e-16],  # m, simulation domain [x, y, z]
        "target_granules": 1e6,  # Simulation particle count (impacts performance)
        "tick_spacing": 0.25,  # Axis tick marks spacing for position reference
        "color_theme": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
    },
    "wave_sources": {
        "count": 1,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Single source at center for pure spherical wave propagation
        "positions": [[0.5, 0.5, 0.5]],  # Wave Source position - Center
        # Phase offset in degrees (0° = in phase with base frequency)
        "phase_offsets_deg": [0],
    },
    "ui_defaults": {
        "show_axis": True,  # Toggle to show/hide axis lines
        "block_slice": True,  # Block-slicing toggle
        "show_sources": False,  # Toggle to show/hide wave source markers
        "radius_factor": 0.4,  # Granule radius scaling factor
        "freq_boost": 0.5,  # Frequency boost multiplier
        "amp_boost": 5.0,  # Amplitude boost multiplier
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
