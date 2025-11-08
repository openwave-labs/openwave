"""
XPERIMENT PARAMETERS: Standing Wave

Demonstrates standing wave patterns from multiple sources arranged in a circle.
64 sources generate spherical longitudinal waves that superpose at each granule,
creating complex standing wave interference patterns.

This XPERIMENT showcases:
- 64 wave sources arranged in a circular pattern
- Standing wave formation
- Complex interference patterns
- Thin Z dimension for 2.5D visualization
"""

import numpy as np

# Generate 64 sources in a circular pattern
NUM_SOURCES = 64  # Number of wave sources for this xperiment
CIRCLE_RADIUS = 0.333  # Normalized radius of circular arrangement (0-1 range)
Z_POSITION = 0.06  # Z-axis position for all sources (thin slice)

# Calculate source positions in a circle around center
SOURCES_POSITION = [
    [
        0.5 + CIRCLE_RADIUS * np.cos(2 * np.pi * i / NUM_SOURCES),
        0.5 + CIRCLE_RADIUS * np.sin(2 * np.pi * i / NUM_SOURCES),
        Z_POSITION,
    ]
    for i in range(NUM_SOURCES)
]

# All sources in phase (0°) to create standing wave pattern
SOURCES_PHASE_DEG = [0] * NUM_SOURCES

PARAMETERS = {
    "meta": {
        "name": "Standing Wave",
        "description": "64 sources in circular pattern creating standing wave patterns",
    },
    "camera": {
        "initial_position": [1.33, 0.67, 1.52],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "size": [2e-16, 2e-16, 0.1e-16],  # m, simulation domain [x, y, z]
        "tick_spacing": 0.25,  # Axis tick marks spacing for position reference
        "color_theme": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
    },
    "wave_sources": {
        "count": NUM_SOURCES,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Each row represents [x, y, z] coordinates for one source (Z-up coordinate system)
        "positions": SOURCES_POSITION,
        # Phase offsets for each source (integer degrees, converted to radians internally)
        # Allows creating constructive/destructive interference patterns
        # Common patterns: 0° = in phase, 180° = opposite phase, 90° = quarter-cycle offset
        "phase_offsets_deg": SOURCES_PHASE_DEG,
    },
    "ui_defaults": {
        "show_axis": False,  # Toggle to show/hide axis lines
        "block_slice": False,  # Block-slicing toggle
        "show_sources": False,  # Toggle to show/hide wave source markers
        "radius_factor": 1.0,  # Granule radius scaling factor
        "freq_boost": 0.5,  # Frequency boost multiplier
        "amp_boost": 0.1,  # Amplitude boost multiplier
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
