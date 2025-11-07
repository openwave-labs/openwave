"""
XPERIMENT PARAMETERS: Wave Interference

Demonstrates wave interference from three sources arranged in an equilateral triangle.
Each source generates spherical longitudinal waves that superpose at each granule,
creating constructive and destructive interference patterns.

This XPERIMENT showcases:
- 3 wave sources in equilateral triangle pattern
- Wave superposition and interference patterns
- Thin Z dimension for 2.5D visualization
- No spring coupling (pure wave propagation)
"""

import math

# Calculate equilateral triangle positions
EQUILATERAL = math.sqrt(3) / 6  # Height factor for equilateral triangle
Z_POSITION = 0.12

PARAMETERS = {
    "meta": {
        "name": "Wave Interference",
        "description": "3 sources in triangular pattern demonstrating wave superposition",
    },
    "camera": {
        "initial_position": [0.25, 0.91, 1.00],
    },
    "universe": {
        "size_multipliers": [8, 6, 1],  # Multiplies EWAVE_LENGTH
        "tick_spacing": 0.25,
        "color_theme": "OCEAN",
    },
    "wave_sources": {
        "count": 3,
        "positions": [
            [0.5, 0.45 - EQUILATERAL, Z_POSITION],  # Bottom center
            [0.25, 0.45 + EQUILATERAL / 2, Z_POSITION],  # Top left
            [0.75, 0.45 + EQUILATERAL / 2, Z_POSITION],  # Top right
        ],
        "phase_offsets_deg": [0, 0, 0],
    },
    "ui_defaults": {
        "show_axis": False,
        "block_slice": False,
        "show_sources": True,
        "radius_factor": 1.0,
        "freq_boost": 1.0,
        "amp_boost": 1.0,
        "paused": False,
        "granule_type": False,
        "ironbow": True,
        "blueprint": False,
        "var_displacement": True,
    },
    "diagnostics": {
        "wave_diagnostics": False,
        "export_video": False,
        "video_frames": 24,
    },
}
