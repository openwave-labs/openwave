# Level 0 Granule-Based Xperiments

## Overview

Interactive wave propagation simulations demonstrating interference patterns from multiple harmonic oscillators in a BCC lattice medium. Features a unified launcher with UI-based xperiment selection and real-time parameter control.

## Quick Start

Run the launcher from command line:

```bash
python launcher_L0.py
```

Or use the OpenWave CLI menu to select from available xperiments.

## Available Xperiments

1. **Spacetime Vibration** (default) - 9 sources demonstrating wave interference
2. **3D Spherical Wave** - Single source with pure radial propagation
3. **Standing Wave** - 64 sources in circular pattern creating standing waves
4. **Wave Interference** - 3 sources in triangular arrangement
5. **Wave Pulse** - Single source with wave diagnostics enabled
6. **Crossing Waves** - 9 sources with desert color theme
7. **Yin-Yang Spiral Wave** - 12 sources in golden ratio spiral pattern

## Keyboard Controls

- **ESC** - Close window and exit
- **Q** - Zoom camera in/out
- **Cmd+Q** - Not available (use ESC instead, see Technical Notes below)

## Switching Between Xperiments

The xperiment launcher appears at the **top-left** of the window:

1. Click any xperiment checkbox to select it
2. Program automatically restarts with the new xperiment
3. All parameters reset to the selected xperiment's defaults

**Note**: Switching requires a program restart (2-5 seconds) due to Taichi's field initialization constraints. This prevents crashes when changing lattice dimensions.

## UI Controls

All xperiments share these interactive controls:

### Display Options

- **Axis** - Show/hide coordinate axes with tick marks
- **Block Slice** - Enable see-through slicing effect
- **Show Wave Sources** - Display wave source markers
- **Granule Size** - Adjust particle radius (0.1-2.0x)

### Wave Parameters

- **f Boost** - Frequency multiplier for slow-motion visibility (0.1-10.0x)
- **Amp Boost** - Amplitude multiplier for visualization (0.1-5.0x)
- **Pause/Continue** - Freeze/resume simulation

### Color Schemes

- **Displacement (ironbow)** - Color by granule displacement from rest
- **Amplitude (ironbow)** - Color by oscillation amplitude
- **Amplitude (blueprint)** - Blueprint-style amplitude coloring
- **Granule Type Color** - Color by lattice position type
- **Medium Default Color** - Uniform medium color

## Creating Custom Xperiments

### 1. Create Parameter File

Create a new file in `_parameters/` directory (e.g., `my_wave.py`):

```python
"""
XPERIMENT PARAMETERS: My Custom Wave

Brief description of what this xperiment demonstrates.
"""

PARAMETERS = {
    "meta": {
        "name": "My Custom Wave",
        "description": "Custom wave interference pattern",
    },
    "camera": {
        "initial_position": [2.0, 1.5, 1.75],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "size": [1e-16, 1e-16, 1e-16],  # m, simulation domain [x, y, z]
        "tick_spacing": 0.25,  # Axis tick marks spacing
        "color_theme": "OCEAN",  # OCEAN, DESERT, or FOREST
    },
    "wave_sources": {
        "count": 4,  # Number of wave sources
        "positions": [
            [0.25, 0.25, 0.5],  # Source positions in normalized coords (0-1)
            [0.75, 0.25, 0.5],
            [0.25, 0.75, 0.5],
            [0.75, 0.75, 0.5],
        ],
        "phase_offsets_deg": [0, 90, 180, 270],  # Phase offsets in degrees
    },
    "ui_defaults": {
        "show_axis": True,
        "block_slice": False,
        "show_sources": True,
        "radius_factor": 0.5,
        "freq_boost": 1.0,
        "amp_boost": 1.0,
        "paused": False,
        "granule_type": False,
        "ironbow": True,
        "blueprint": False,
        "var_displacement": True,
    },
    "diagnostics": {
        "wave_diagnostics": False,  # Enable wave speed/wavelength measurements
        "export_video": False,  # Export frames for video
        "video_frames": 24,  # Number of frames to capture
    },
}
```

### 2. Run Launcher

Your new xperiment automatically appears in the launcher UI!

## Parameter Customization Guide

### Positioning Wave Sources

Use Python code for computed positions:

```python
import math

# Circular pattern
NUM_SOURCES = 8
CIRCLE_RADIUS = 0.4
SOURCES_POSITION = [
    [
        0.5 + CIRCLE_RADIUS * math.cos(2 * math.pi * i / NUM_SOURCES),
        0.5 + CIRCLE_RADIUS * math.sin(2 * math.pi * i / NUM_SOURCES),
        0.5,
    ]
    for i in range(NUM_SOURCES)
]

# Equilateral triangle
EQUILATERAL = math.sqrt(3) / 6
SOURCES_POSITION = [
    [0.5, 0.5 - EQUILATERAL, 0.5],      # Bottom
    [0.25, 0.5 + EQUILATERAL/2, 0.5],   # Top left
    [0.75, 0.5 + EQUILATERAL/2, 0.5],   # Top right
]

# Grid pattern
SOURCES_POSITION = [
    [x/3, y/3, 0.5]
    for x in range(1, 4)
    for y in range(1, 4)
]
```

### Phase Offset Patterns

Create interference effects with phase control:

```python
# All in phase (constructive interference)
SOURCES_PHASE_DEG = [0, 0, 0, 0]

# Alternating phase (standing waves)
SOURCES_PHASE_DEG = [0, 180, 0, 180]

# Progressive spiral
SOURCES_PHASE_DEG = [i * 30 for i in range(12)]  # 0°, 30°, 60°, ..., 330°

# Random phase (chaotic patterns)
import random
SOURCES_PHASE_DEG = [random.randint(0, 360) for _ in range(NUM_SOURCES)]
```

### Universe Dimensions

Adjust simulation volume:

```python
from openwave.common import constants

# Cubic volume
"size": [1e-16, 1e-16, 1e-16]

# Thin slice (2.5D visualization)
"size": [2e-16, 2e-16, 1e-17]

# Elongated volume
"size": [2e-16, 1.5e-16, 1e-16]

# Using wavelength multiples
"size": [
    6 * constants.EWAVE_LENGTH,
    6 * constants.EWAVE_LENGTH,
    2 * constants.EWAVE_LENGTH,
]
```

### Camera Positioning

Set initial camera view:

```python
# Top-down view
"initial_position": [1.0, 0.0, 2.0]

# Angled view
"initial_position": [2.0, 1.5, 1.75]

# Side view
"initial_position": [0.0, 2.0, 1.0]

# Close-up
"initial_position": [0.5, 0.5, 0.8]
```

## Parameter Reference

### Required Fields

All parameter files must include:

- `meta` - Xperiment name and description
- `camera` - Initial camera position
- `universe` - Simulation domain size and theme
- `wave_sources` - Source count, positions, and phases
- `ui_defaults` - Default UI control states
- `diagnostics` - Diagnostic and export settings

### Coordinate System

- **Positions**: Normalized coordinates (0.0-1.0 range)
  - `0.0` = minimum universe edge
  - `1.0` = maximum universe edge
  - `0.5` = center
- **Axes**: Z-up coordinate system (X-right, Y-forward, Z-up)
- **Camera**: Free-rotation orbit camera (mouse drag to rotate)

### Color Themes

Available universe color themes:

- `OCEAN` - Blue/cyan tones (default)
- `DESERT` - Warm earth tones
- `FOREST` - Green/natural tones

## Directory Structure

```text
level0_granule_based/
├── launcher_L0.py          # Main launcher (run this)
├── _parameters/            # Xperiment parameter files
│   ├── spacetime_vibration.py
│   ├── spherical_wave.py
│   ├── standing_wave.py
│   ├── wave_interference.py
│   ├── wave_pulse.py
│   ├── xwaves.py
│   └── yin_yang.py
├── _legacy/                # Archived standalone files
└── README.md               # This file
```

**Note**: The underscore prefix on `_parameters/` prevents individual parameter files from appearing in the CLI menu. Only `launcher_L0.py` is user-facing.

## Troubleshooting

### Xperiment not appearing in launcher

- Ensure file is in `_parameters/` directory
- Filename must end with `.py` (not `__init__.py`)
- Must contain a `PARAMETERS` dictionary
- Check console for import errors

### Incorrect interference patterns

- Verify `count` matches number of entries in `positions` and `phase_offsets_deg`
- Ensure positions are in normalized coordinates (0-1 range)
- Check phase offsets are in degrees (not radians)

### Performance issues

- Reduce universe size for faster simulation
- Decrease granule count by using smaller universe
- Lower `freq_boost` to reduce computation per frame
- Disable `wave_diagnostics` when not needed

### Import errors

- Parameter files are imported as Python modules
- Import path: `openwave.xperiments.level0_granule_based._parameters.<filename>`
- Ensure all required imports are at top of parameter file

## Technical Notes

### Xperiment Switching Mechanism

Due to Taichi's constraints, switching xperiments uses `os.execv()` to replace the current process. This has two side effects on macOS:

1. **Warning message**: "Task policy set failed: 4" (harmless, can be ignored)
2. **Cmd+Q disabled**: Use ESC key or window X button to exit instead

This approach ensures clean lattice initialization and prevents segmentation faults when changing universe dimensions.

### State Management

The launcher maintains:

- `XperimentManager` - Auto-discovers and loads parameter files
- `SimulationState` - Tracks simulation time, frames, and UI states
- Fresh initialization on each xperiment switch

### Performance Optimization

- Data sampling updates every 30 frames (not every frame)
- Parameter files loaded once at startup
- Module reloading enabled for development
- No performance impact during rendering

## Example Patterns

### Interference Node Pattern

```python
# 4 sources at corners create central null point
"wave_sources": {
    "count": 4,
    "positions": [
        [0.25, 0.25, 0.5],
        [0.75, 0.25, 0.5],
        [0.25, 0.75, 0.5],
        [0.75, 0.75, 0.5],
    ],
    "phase_offsets_deg": [0, 180, 180, 0],  # Opposite corners in phase
}
```

### Rotating Wave Pattern

```python
# Progressive phase creates rotating interference
"wave_sources": {
    "count": 6,
    "positions": [
        [0.5 + 0.3 * math.cos(i * math.pi/3),
         0.5 + 0.3 * math.sin(i * math.pi/3),
         0.5]
        for i in range(6)
    ],
    "phase_offsets_deg": [i * 60 for i in range(6)],  # 60° increments
}
```

### Focused Wave Beam

```python
# Linear array creates directional beam
"wave_sources": {
    "count": 5,
    "positions": [[0.2 + i*0.15, 0.5, 0.5] for i in range(5)],
    "phase_offsets_deg": [0, 0, 0, 0, 0],  # All in phase
}
```

## Additional Resources

- **Physics Background**: See `/research_requirements/` for Energy Wave Theory papers
- **Code Documentation**: See `launcher_L0.py` docstrings for implementation details
- **Legacy Files**: See `_legacy/` for original standalone xperiment files

---

**Directory**: `openwave/xperiments/level0_granule_based/`

**Purpose**: Educational simulations of wave interference in granular spacetime medium
