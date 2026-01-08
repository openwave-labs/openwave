"""
XPERIMENT PARAMETERS

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
NUM_SOURCES = 1  # Number of wave sources for this xperiment, legacy = 64
Z_POSITION = 0.05  # Z-axis position for all sources (thin slice)

# Calculate source positions in a circle around center
UNIVERSE_EDGE = 6 * constants.EWAVE_LENGTH  # m, universe edge length in meters
SOURCES_RADIUS = 1 * constants.EWAVE_LENGTH  # m, r = n * λ, interference radius, legacy = 2
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
            UNIVERSE_EDGE * Z_POSITION,
        ],  # m, simulation domain [x, y, z]
        "TARGET_GRANULES": 1e6,  # Simulation particle count (impacts performance)
    },
    "wave_sources": {
        "COUNT": NUM_SOURCES,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Each row represents [x, y, z] coordinates for one source (Z-up coordinate system)
        "POSITION": SOURCES_POSITION,
        # Phase offsets for each source (integer degrees, converted to radians internally)
        # Allows creating constructive/destructive interference patterns
        # Common patterns: 0° = in phase, 180° = opposite phase, 90° = quarter-cycle offset
        "PHASE_OFFSETS_DEG": SOURCES_PHASE_DEG,
        "IN_WAVE_TOGGLE": 1,  # 1 = enable in_wave, 0 = disable in_wave
        "OUT_WAVE_TOGGLE": 1,  # 1 = enable out_wave, 0 = disable out_wave
    },
    "ui_defaults": {
        "SHOW_AXIS": False,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "BLOCK_SLICE": False,  # Block-slicing toggle
        "SHOW_SOURCES": True,  # Toggle to show/hide wave source markers
        "RADIUS_FACTOR": 1.0,  # Granule radius scaling factor
        "FREQ_BOOST": 0.5,  # Frequency boost multiplier
        "AMP_BOOST": 1.0,  # Amplitude boost multiplier, legacy = 0.1
        "PAUSED": False,  # Pause/Start simulation toggle
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
        "COLOR_PALETTE": 6,  # default (99), granule-type (0), ironbow (3), orange (6)
    },
    "analytics": {
        "INSTRUMENTATION": False,  # Toggle data collection (speed & wavelength measurements)
        "EXPORT_VIDEO": False,  # Toggle frame image export to video directory
        "VIDEO_FRAMES": 24,  # Target frame number to stop recording and finalize video export
    },
}
