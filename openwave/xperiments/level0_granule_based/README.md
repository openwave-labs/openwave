# Level 0 Granule-Based Xperiments - Refactored Structure

## Overview

This directory has been refactored to eliminate code duplication and improve maintainability. All xperiments now share a single unified launcher with xperiment-specific parameters stored in separate config files.

## Directory Structure

```text
level0_granule_based/
├── launcher_L0.py               # Universal xperiment launcher (run this!)
├── configs/                     # Xperiment configuration files
│   ├── __init__.py
│   ├── spacetime_vibration.py   # Default xperiment config
│   ├── spherical_wave.py
│   ├── standing_wave.py
│   ├── wave_interference.py
│   ├── wave_pulse.py
│   ├── xwaves.py
│   └── yin_yang.py
├── _legacy/                     # Original xperiment files (archived)
│   └── [old standalone files]
└── README_REFACTORING.md        # This file
```

## How to Use

### Running Xperiments

Simply run the universal launcher:

```bash
python launcher_L0.py
```

### Selecting Xperiments

The xperiment launcher UI appears at the **top-left** of the window, just above the xperiment specs panel. Features include:

- **Current xperiment display**: Shows which xperiment is currently running
- **Xperiment list**: All available xperiments as checkboxes
- **Xperiment switching**: Click any xperiment to switch to it
- **Automatic restart**: The program automatically restarts with the new xperiment

**Note**: Due to Taichi's field initialization constraints, switching xperiments requires restarting the program. When you select a new xperiment, the current window closes and a new instance launches with the selected xperiment. This happens automatically and takes just a few seconds.

### Available Xperiments

1. **Spacetime Vibration** (default) - 9 sources with interference patterns
2. **3D Spherical Wave** - Single source demonstrating radial propagation
3. **Standing Wave** - 64 sources in circular pattern
4. **Wave Interference** - 3 sources in triangular arrangement
5. **Wave Pulse** - Single source with diagnostics enabled
6. **Crossing Waves** - 9 sources with DESERT color theme
7. **Yin-Yang Spiral Wave** - 12 sources in golden ratio spiral

## Architecture Benefits

### Before Refactoring

- **7 separate files** with ~270 lines of duplicated code each
- **~1,890 lines** of duplicated UI/rendering code total
- Any bug fix required updating all 7 files
- Inconsistent behavior if files got out of sync

### After Refactoring

- **Single launcher** with shared UI/rendering code (~540 lines)
- **7 config files** containing only xperiment-specific parameters (~60 lines each)
- **~960 lines total** (50% reduction in code)
- Single source of truth for all UI and rendering logic
- Easy to add new xperiments (just create a config file)
- Bugs fixed once affect all xperiments

## Creating New Xperiments

To create a new xperiment:

1. Create a new config file in `configs/` directory (e.g., `my_xperiment.py`)
2. Define the `CONFIG` dictionary with required parameters:

```python
"""
XPERIMENT CONFIG: My Custom Xperiment

Description of what this xperiment demonstrates.
"""

CONFIG = {
    "meta": {
        "name": "My Custom Xperiment",
        "description": "Brief description",
    },
    "camera": {
        "initial_position": [x, y, z],  # Normalized coordinates
    },
    "universe": {
        "size_multipliers": [x, y, z],  # Multiplies EWAVE_LENGTH
        "tick_spacing": 0.25,
        "color_theme": "OCEAN",  # OCEAN, DESERT, or FOREST
    },
    "wave_sources": {
        "count": n,
        "positions": [[x1, y1, z1], [x2, y2, z2], ...],  # Normalized
        "phase_offsets_deg": [0, 180, ...],
    },
    "ui_defaults": {
        "show_axis": False,
        "block_slice": False,
        "show_sources": False,
        "radius_factor": 0.5,
        "freq_boost": 10.0,
        "amp_boost": 1.0,
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
```

3. Run `launcher_L0.py` - your new xperiment will automatically appear in the launcher!

## Config File Features

### Computed Positions

Config files can use Python code to compute positions:

```python
import math

NUM_SOURCES = 64
SOURCES_POSITION = [
    [
        0.5 + 0.3 * math.cos(2 * math.pi * i / NUM_SOURCES),
        0.5 + 0.3 * math.sin(2 * math.pi * i / NUM_SOURCES),
        0.5,
    ]
    for i in range(NUM_SOURCES)
]
```

### Phase Patterns

Create interesting interference patterns with phase offsets:

```python
SOURCES_PHASE_DEG = [i * 30 for i in range(12)]  # 0°, 30°, 60°, ..., 330°
```

## Legacy Files

The original standalone xperiment files are preserved in `_legacy/` for reference. These files still work but are no longer maintained.

## Technical Details

### State Management

The `SimulationState` class manages all xperiment parameters and simulation state:

- Lattice initialization
- Time tracking
- UI control variables
- Diagnostics settings

### XperimentManager

The `XperimentManager` class handles:

- Auto-discovery of config files
- Dynamic loading of configurations
- Module reloading for development

### Xperiment Switching Process

When switching xperiments:

1. Current window closes
2. Program automatically restarts with new xperiment
3. New instance loads selected config
4. Lattice is initialized with new parameters
5. Fresh time tracking begins
6. Rendering starts with new xperiment

**Why restart?** Taichi doesn't support reinitializing fields with different dimensions after they've been created. Attempting to do so causes segmentation faults. The restart approach ensures clean initialization and prevents crashes.

## Performance Considerations

- Switching takes 2-5 seconds (program restart)
- Config files are loaded dynamically on startup
- No performance impact during normal rendering
- Each xperiment runs in a fresh Taichi instance

## Troubleshooting

### Xperiment not appearing in launcher

- Ensure config file is in `configs/` directory
- Filename must end with `.py` (not `__init__.py`)
- Must contain a `CONFIG` dictionary

### Import errors

- Config files are imported as modules
- Import path: `openwave.xperiments.level0_granule_based.configs.<name>`
- Ensure all required fields are present in CONFIG

### Switching fails

- Check console for error messages
- Verify config dictionary structure
- Ensure all list lengths match (e.g., NUM_SOURCES vs positions)

## Migration Notes

If you have custom modifications to the old standalone files:

1. Locate your modified file in `_legacy/`
2. Extract the parameter values (lines 1-95)
3. Create a new config file with these parameters
4. Test with `launcher_level0.py`

## Future Enhancements

Potential improvements:

- Config validation with helpful error messages
- Preset library with example patterns
- Config file templates
- Export current state as new config
- Xperiment descriptions in UI tooltips

---

**Last Updated**: November 7, 2025
**Refactored By**: Claude Code
**Reason**: Eliminate code duplication, improve maintainability, enhance user experience
