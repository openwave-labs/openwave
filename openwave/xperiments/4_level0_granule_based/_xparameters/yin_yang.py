"""
XPERIMENT PARAMETERS: Yin-Yang Effect from Golden-Ratio Spiral Wave

Demonstrates spiral wave patterns from multiple sources arranged in a golden ratio radius.
12 sources generate spherical longitudinal waves that superpose at each granule,
creating beautiful spiral interference patterns inspired by the Yin-Yang symbol.

This XPERIMENT showcases:
- 12 wave sources arranged in golden ratio pattern
- Progressive phase offsets (30° increments)
- Spiral wave interference patterns
"""

import numpy as np

from openwave.common import config, constants

# Generate 12 sources in a golden ratio pattern for spiral Yin-Yang effect
NUM_SOURCES = 12  # Number of wave sources for this xperiment
Z_POSITION = 0.0  # Z-axis position for all sources (flat plane)

# Calculate source positions in a golden ratio pattern
UNIVERSE_EDGE = 12 * constants.EWAVE_LENGTH  # m, universe edge length in meters
GOLDEN_RADIUS = constants.EWAVE_LENGTH / constants.GOLDEN_RATIO  # m, r = λ / φ, for spiral effect
NORMALIZED_RADIUS = GOLDEN_RADIUS / UNIVERSE_EDGE  # normalized radius

# Positions relative to universe center (0.5, 0.5, Z_POSITION)
SOURCES_POSITION = [
    [
        np.cos(i * 2 * np.pi / NUM_SOURCES) * NORMALIZED_RADIUS + 0.5,
        np.sin(i * 2 * np.pi / NUM_SOURCES) * NORMALIZED_RADIUS + 0.5,
        Z_POSITION,
    ]
    for i in range(NUM_SOURCES)
]

# Phase offsets: 30° increments (0°, 30°, 60°, ..., 330°) for progressive spiral wave pattern
SOURCES_PHASE_DEG = [i * 30 for i in range(NUM_SOURCES)]

XPARAMETERS = {
    "meta": {
        "name": "Golden-Ratio Spiral",
        "description": "12 sources in golden ratio pattern with progressive phase offsets",
    },
    "camera": {
        "initial_position": [1.43, 0.74, 1.41],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "size": [
            UNIVERSE_EDGE,
            UNIVERSE_EDGE,
            UNIVERSE_EDGE / 6,
        ],  # m, simulation domain [x, y, z]
        "target_granules": 1e6,  # Simulation particle count (impacts performance)
        "tick_spacing": 0.25,  # Axis tick marks spacing for position reference
        "color_theme": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
    },
    "wave_sources": {
        "count": NUM_SOURCES,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Arranged in golden ratio spiral for Yin-Yang pattern
        "positions": SOURCES_POSITION,
        # Phase offsets for each source (integer degrees, converted to radians internally)
        # Progressive 30° increments create spiral wave interference pattern
        "phase_offsets_deg": SOURCES_PHASE_DEG,
    },
    "ui_defaults": {
        "show_axis": False,  # Toggle to show/hide axis lines
        "block_slice": False,  # Block-slicing toggle
        "show_sources": True,  # Toggle to show/hide wave source markers
        "radius_factor": 2.0,  # Granule radius scaling factor
        "freq_boost": 0.5,  # Frequency boost multiplier
        "amp_boost": 1.0,  # Amplitude boost multiplier
        "paused": False,  # Pause/Start simulation toggle
        "granule_type": True,  # Granule type color
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
