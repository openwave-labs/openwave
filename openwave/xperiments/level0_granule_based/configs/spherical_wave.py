"""
XPERIMENT CONFIG: 3D Spherical Wave

Demonstrates wave interference from multiple sources in the lattice.
Each source generates spherical longitudinal waves that superpose at each granule,
creating constructive and destructive interference patterns.

This XPERIMENT showcases:
- Single wave source at center
- Pure spherical wave propagation
- Radial wave patterns
- No spring coupling (pure wave propagation)
"""

CONFIG = {
    "meta": {
        "name": "3D Spherical Wave",
        "description": "Single source spherical wave demonstrating radial propagation",
    },
    "camera": {
        "initial_position": [0.97, 2.06, 0.82],
    },
    "universe": {
        "size_multipliers": [4, 4, 4],  # Multiplies EWAVE_LENGTH
        "tick_spacing": 0.25,
        "color_theme": "OCEAN",
    },
    "wave_sources": {
        "count": 1,
        "positions": [[0.5, 0.5, 0.5]],  # Wave Source position - Center
        "phase_offsets_deg": [0],
    },
    "ui_defaults": {
        "show_axis": True,
        "block_slice": True,
        "show_sources": False,
        "radius_factor": 0.4,
        "freq_boost": 1.0,
        "amp_boost": 5.0,
        "paused": False,
        "granule_type": False,
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
