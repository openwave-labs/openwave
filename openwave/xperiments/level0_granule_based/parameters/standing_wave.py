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
- No spring coupling (pure wave propagation)
"""

import math

# Generate 64 sources in a circular pattern
NUM_SOURCES = 64
CIRCLE_RADIUS = 0.333
Z_POSITION = 0.09

SOURCES_POSITION = [
    [
        0.5 + CIRCLE_RADIUS * math.cos(2 * math.pi * i / NUM_SOURCES),
        0.5 + CIRCLE_RADIUS * math.sin(2 * math.pi * i / NUM_SOURCES),
        Z_POSITION,
    ]
    for i in range(NUM_SOURCES)
]

SOURCES_PHASE_DEG = [0] * NUM_SOURCES

PARAMETERS = {
    "meta": {
        "name": "Standing Wave",
        "description": "64 sources in circular pattern creating standing wave patterns",
    },
    "camera": {
        "initial_position": [1.33, 0.67, 1.52],
    },
    "universe": {
        "size_multipliers": [6, 6, 0.5],  # Multiplies EWAVE_LENGTH
        "tick_spacing": 0.25,
        "color_theme": "OCEAN",
    },
    "wave_sources": {
        "count": NUM_SOURCES,
        "positions": SOURCES_POSITION,
        "phase_offsets_deg": SOURCES_PHASE_DEG,
    },
    "ui_defaults": {
        "show_axis": False,
        "block_slice": False,
        "show_sources": False,
        "radius_factor": 1.0,
        "freq_boost": 0.5,
        "amp_boost": 0.1,
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
