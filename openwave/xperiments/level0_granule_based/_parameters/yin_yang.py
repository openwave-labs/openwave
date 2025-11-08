"""
XPERIMENT PARAMETERS: Yin-Yang Spiral Wave

Demonstrates spiral wave patterns from multiple sources arranged in a golden ratio spiral.
12 sources generate spherical longitudinal waves that superpose at each granule,
creating beautiful spiral interference patterns inspired by the Yin-Yang symbol.

This XPERIMENT showcases:
- 12 wave sources arranged in golden ratio spiral pattern
- Progressive phase offsets (30° increments)
- Spiral wave interference patterns
"""

import math

from openwave.common import constants

# Generate 12 sources in a golden ratio spiral for Yin-Yang pattern
NUM_SOURCES = 12  # Number of wave sources for this xperiment
GOLDEN_RATIO = 1.618  # Golden ratio φ ≈ 1.618 for spiral geometry
Z_POSITION = 0.0  # Z-axis position for all sources (flat plane)

# Original formula: r = λ / φ, where φ = golden ratio ~1.618, for yin-yang spiral effect
# Position divides by (6 * golden_ratio) to scale the radius to universe size
SOURCES_POSITION = [
    [
        math.cos(i * 2 * math.pi / NUM_SOURCES) / (6 * GOLDEN_RATIO) + 0.5,
        math.sin(i * 2 * math.pi / NUM_SOURCES) / (6 * GOLDEN_RATIO) + 0.5,
        Z_POSITION,
    ]
    for i in range(NUM_SOURCES)
]

# Phase offsets: 30° increments (0°, 30°, 60°, ..., 330°) for progressive spiral wave pattern
SOURCES_PHASE_DEG = [i * 30 for i in range(NUM_SOURCES)]

PARAMETERS = {
    "meta": {
        "name": "Yin-Yang Spiral Wave",
        "description": "12 sources in golden ratio spiral with progressive phase offsets",
    },
    "camera": {
        "initial_position": [1.50, 0.80, 1.50],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "size": [
            6 * constants.EWAVE_LENGTH,
            6 * constants.EWAVE_LENGTH,
            2 * constants.EWAVE_LENGTH,
        ],  # m, simulation domain [x, y, z] (thin Z for 2.5D visualization)
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
        "freq_boost": 0.1,  # Frequency boost multiplier
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
