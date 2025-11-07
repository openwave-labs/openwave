"""
XPERIMENT CONFIG: Yin-Yang Spiral Wave

Demonstrates spiral wave patterns from multiple sources arranged in a golden ratio spiral.
12 sources generate spherical longitudinal waves that superpose at each granule,
creating beautiful spiral interference patterns inspired by the Yin-Yang symbol.

This XPERIMENT showcases:
- 12 wave sources arranged in golden ratio spiral pattern
- Progressive phase offsets (30° increments)
- Spiral wave interference patterns
- No spring coupling (pure wave propagation)
"""

import math

# Generate 12 sources in a golden ratio spiral
NUM_SOURCES = 12
GOLDEN_RATIO = 1.618
Z_POSITION = 0.0

# Original formula: r = λ / φ, where φ = golden ratio ~1.618, for yin-yang spiral effect
# Position divides by (6 * golden_ratio) to scale the radius
SOURCES_POSITION = [
    [
        math.cos(i * 2 * math.pi / (NUM_SOURCES - 1)) / (6 * GOLDEN_RATIO) + 0.5,
        math.sin(i * 2 * math.pi / (NUM_SOURCES - 1)) / (6 * GOLDEN_RATIO) + 0.5,
        Z_POSITION,
    ]
    for i in range(NUM_SOURCES)
]

# Phase offsets: 30° increments (0°, 30°, 60°, ..., 330°)
SOURCES_PHASE_DEG = [i * 30 for i in range(NUM_SOURCES)]

CONFIG = {
    "meta": {
        "name": "Yin-Yang Spiral Wave",
        "description": "12 sources in golden ratio spiral with progressive phase offsets",
    },
    "camera": {
        "initial_position": [1.50, 0.80, 1.50],
    },
    "universe": {
        "size_multipliers": [6, 6, 2],  # Multiplies EWAVE_LENGTH
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
        "show_sources": True,
        "radius_factor": 2.0,
        "freq_boost": 0.1,
        "amp_boost": 5.0,
        "paused": False,
        "granule_type": True,
        "ironbow": False,
        "blueprint": False,
        "var_displacement": True,
    },
    "diagnostics": {
        "wave_diagnostics": False,
        "export_video": False,
        "video_frames": 24,
    },
}
