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

from openwave.common import constants

# Generate 64 sources in a circular pattern
NUM_SOURCES = 64  # Number of wave sources for this xperiment
Z_POSITION = 0.05  # Z-axis position for all sources (thin slice)

# Calculate source positions in a circle around center
UNIVERSE_EDGE = 6 * constants.EWAVE_LENGTH  # m, universe edge length in meters
SOURCES_RADIUS = 2 * constants.EWAVE_LENGTH  # m, r = n * λ, interference radius
NORMALIZED_RADIUS = SOURCES_RADIUS / UNIVERSE_EDGE  # normalized radius

# Positions relative to universe center (0.5, 0.5, Z_POSITION)
SOURCES_POSITION = [
    [
        np.cos(i * 2 * np.pi / NUM_SOURCES) * NORMALIZED_RADIUS + 0.5,
        np.sin(i * 2 * np.pi / NUM_SOURCES) * NORMALIZED_RADIUS + 0.5,
        Z_POSITION,
    ]
    for i in range(NUM_SOURCES)
]

# All sources in phase (0°) to create standing wave pattern
SOURCES_PHASE_DEG = [0] * NUM_SOURCES

XPARAMETERS = {
    "meta": {
        "X_NAME": "Standing Wave",
        "DESCRIPTION": "64 sources in circular pattern creating standing wave patterns",
    },
    "camera": {
        "INITIAL_POSITION": [1.33, 0.67, 1.52],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "SIZE": [
            UNIVERSE_EDGE,
            UNIVERSE_EDGE,
            UNIVERSE_EDGE / 20,
        ],  # m, simulation domain [x, y, z]
        "TARGET_GRANULES": 1e6,  # Simulation particle count (impacts performance)
    },
    "wave_sources": {
        "COUNT": NUM_SOURCES,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Each row represents [x, y, z] coordinates for one source (Z-up coordinate system)
        "POSITIONS": SOURCES_POSITION,
        # Phase offsets for each source (integer degrees, converted to radians internally)
        # Allows creating constructive/destructive interference patterns
        # Common patterns: 0° = in phase, 180° = opposite phase, 90° = quarter-cycle offset
        "PHASE_OFFSETS_DEG": SOURCES_PHASE_DEG,
    },
    "ui_defaults": {
        "SHOW_AXIS": False,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "BLOCK_SLICE": False,  # Block-slicing toggle
        "SHOW_SOURCES": False,  # Toggle to show/hide wave source markers
        "RADIUS_FACTOR": 1.0,  # Granule radius scaling factor
        "FREQ_BOOST": 0.5,  # Frequency boost multiplier
        "AMP_BOOST": 0.1,  # Amplitude boost multiplier
        "PAUSED": False,  # Pause/Start simulation toggle
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
        "COLOR_PALETTE": 1,  # Color palette list: default (99), granule-type (0), ironbow (1), blueprint (2)
        "VAR_AMP": False,  # Displacement vs amplitude toggle
    },
    "diagnostics": {
        "WAVE_DIAGNOSTICS": False,  # Toggle wave diagnostics (speed & wavelength measurements)
        "EXPORT_VIDEO": False,  # Toggle frame image export to video directory
        "VIDEO_FRAMES": 24,  # Target frame number to stop recording and finalize video export
    },
}
