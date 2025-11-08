"""
XPERIMENT PARAMETERS: Crossing Waves

Demonstrates wave interference from multiple sources in the lattice.
Each source generates spherical longitudinal waves that superpose at each granule,
creating constructive and destructive interference patterns with crossing wave fronts.

This XPERIMENT showcases:
- Multiple wave sources (9 sources: center + 8 corners)
- Crossing wave patterns
- Phase control between sources (constructive/destructive interference)
- DESERT color theme for different visual experience
"""

PARAMETERS = {
    "meta": {
        "name": "Crossing Waves",
        "description": "Crossing Waves Harmonic Oscillations with 9 sources",
    },
    "camera": {
        "initial_position": [2.00, 1.50, 1.75],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "size": [1e-16, 1e-16, 1e-16],  # m, simulation domain [x, y, z]
        "tick_spacing": 0.25,  # Axis tick marks spacing for position reference
        "color_theme": "DESERT",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
    },
    "wave_sources": {
        "count": 9,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Each row represents [x, y, z] coordinates for one source (Z-up coordinate system)
        "positions": [
            [0.5, 0.5, 0.5],  # Wave Source 0 - Center
            [0.0, 1.0, 1.0],  # Wave Source 1 - Back-top-left corner
            [1.0, 0.0, 1.0],  # Wave Source 2 - Front-top-right corner
            [0.0, 1.0, 0.0],  # Wave Source 3 - Back-bottom-left corner
            [1.0, 0.0, 0.0],  # Wave Source 4 - Front-bottom-right corner
            [0.0, 0.0, 1.0],  # Wave Source 5 - Front-top-left corner
            [1.0, 1.0, 1.0],  # Wave Source 6 - Back-top-right corner
            [0.0, 0.0, 0.0],  # Wave Source 7 - Front-bottom-left corner
            [1.0, 1.0, 0.0],  # Wave Source 8 - Back-bottom-right corner
        ],
        # Phase offsets for each source (integer degrees, converted to radians internally)
        # Center source at 180° creates crossing wave patterns with corner sources at 0°
        "phase_offsets_deg": [180, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "ui_defaults": {
        "show_axis": False,  # Toggle to show/hide axis lines
        "block_slice": False,  # Block-slicing toggle
        "show_sources": True,  # Toggle to show/hide wave source markers
        "radius_factor": 1.0,  # Granule radius scaling factor
        "freq_boost": 1.0,  # Frequency boost multiplier
        "amp_boost": 5.0,  # Amplitude boost multiplier
        "paused": False,  # Pause/Start simulation toggle
        "granule_type": True,  # Granule type color
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
