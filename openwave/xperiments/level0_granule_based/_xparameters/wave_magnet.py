"""
XPERIMENT PARAMETERS: Wave Sourced Magnetic Field-Like Interference

Demonstrates wave interference from 2 sources in a linear arrangement.
Each source generates spherical longitudinal waves that superpose at each granule,
creating constructive and destructive interference patterns.

This XPERIMENT showcases:
- 2 wave sources pattern creating a magnetic field-like interference
- Wave superposition and interference patterns
- Thin Z dimension for 2.5D visualization
"""

# Generate 2 sources in a linear pattern
Z_POSITION = 0.07  # Z-axis position for all sources

XPARAMETERS = {
    "meta": {
        "name": "Wave Magnet",
        "description": "2 sources in linear pattern demonstrating wave superposition",
    },
    "camera": {
        "initial_position": [0.44, 0.94, 1.22],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "size": [5e-16, 5e-16, 0.1e-16],  # m, simulation domain [x, y, z]
        "tick_spacing": 0.25,  # Axis tick marks spacing for position reference
        "color_theme": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
    },
    "wave_sources": {
        "count": 2,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Arranged in equilateral triangle for symmetric interference pattern
        "positions": [
            [0.25, 0.50, Z_POSITION],  # left
            [0.75, 0.50, Z_POSITION],  # right
        ],
        # Phase offsets for each source (integer degrees, converted to radians internally)
        # All sources in phase (0°) to create symmetric interference
        "phase_offsets_deg": [0, 0],
    },
    "ui_defaults": {
        "show_axis": False,  # Toggle to show/hide axis lines
        "block_slice": False,  # Block-slicing toggle
        "show_sources": False,  # Toggle to show/hide wave source markers
        "radius_factor": 1.0,  # Granule radius scaling factor
        "amp_boost": 1.0,  # Amplitude boost multiplier
        "freq_boost": 1.0,  # Frequency boost multiplier
        "paused": False,  # Pause/Start simulation toggle
        "granule_type": False,  # Granule type color
        "ironbow": True,  # Ironbow color scheme toggle
        "blueprint": False,  # Blueprint color scheme toggle
        "var_displacement": False,  # Displacement vs amplitude toggle
    },
    "diagnostics": {
        "wave_diagnostics": False,  # Toggle wave diagnostics (speed & wavelength measurements)
        "export_video": False,  # Toggle frame image export to video directory
        "video_frames": 24,  # Target frame number to stop recording and finalize video export
    },
}
