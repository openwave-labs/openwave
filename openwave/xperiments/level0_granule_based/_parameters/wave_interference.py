"""
XPERIMENT PARAMETERS: Wave Interference

Demonstrates wave interference from three sources arranged in an equilateral triangle.
Each source generates spherical longitudinal waves that superpose at each granule,
creating constructive and destructive interference patterns.

This XPERIMENT showcases:
- 3 wave sources in equilateral triangle pattern
- Wave superposition and interference patterns
- Thin Z dimension for 2.5D visualization
"""

import math

# Calculate equilateral triangle positions for symmetric interference pattern
EQUILATERAL = math.sqrt(3) / 6  # Height factor for equilateral triangle geometry
Z_POSITION = 0.05  # Z-axis position for all sources (thin slice)

PARAMETERS = {
    "meta": {
        "name": "Wave Interference",
        "description": "3 sources in triangular pattern demonstrating wave superposition",
    },
    "camera": {
        "initial_position": [0.25, 0.91, 1.00],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "size": [2e-16, 1.5e-16, 1e-17],  # m, simulation domain [x, y, z] (thin Z, elongated X)
        "tick_spacing": 0.25,  # Axis tick marks spacing for position reference
        "color_theme": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
    },
    "wave_sources": {
        "count": 3,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Arranged in equilateral triangle for symmetric interference pattern
        "positions": [
            [0.5, 0.45 - EQUILATERAL, Z_POSITION],  # Bottom center
            [0.25, 0.45 + EQUILATERAL / 2, Z_POSITION],  # Top left
            [0.75, 0.45 + EQUILATERAL / 2, Z_POSITION],  # Top right
        ],
        # Phase offsets for each source (integer degrees, converted to radians internally)
        # All sources in phase (0°) to create symmetric interference
        "phase_offsets_deg": [0, 0, 0],
    },
    "ui_defaults": {
        "show_axis": False,  # Toggle to show/hide axis lines
        "block_slice": False,  # Block-slicing toggle
        "show_sources": True,  # Toggle to show/hide wave source markers
        "radius_factor": 1.0,  # Granule radius scaling factor
        "freq_boost": 1.0,  # Frequency boost multiplier
        "amp_boost": 1.0,  # Amplitude boost multiplier
        "paused": False,  # Pause/Start simulation toggle
        "granule_type": False,  # Granule type color
        "ironbow": True,  # Ironbow color scheme toggle
        "blueprint": False,  # Blueprint color scheme toggle
        "var_displacement": True,  # Displacement vs amplitude toggle
    },
    "diagnostics": {
        "wave_diagnostics": False,  # Toggle wave diagnostics (speed & wavelength measurements)
        "export_video": False,  # Toggle frame image export to video directory
        "video_frames": 24,  # Target frame number to stop recording and finalize video export
    },
}
