# OpenWave Coordinate System

## Current System: Z-up (Right-Handed)

OpenWave uses a **right-handed, Z-up coordinate system** for all simulations and visualizations.

### Axis Conventions

- **X-axis** (Red): Horizontal, left-right (positive = right)
- **Y-axis** (Green): Horizontal, front-back (positive = forward/north)
- **Z-axis** (Blue): Vertical, up-down (positive = up)

### Normalized Coordinates

All positions in experiments are specified using normalized coordinates in the range [0, 1]:

- `[0, 0, 0]` = Bottom-front-left corner
- `[1, 1, 1]` = Top-back-right corner
- `[0.5, 0.5, 0.5]` = Center of the simulation space

### Wave Source Position Format

When defining wave sources in experiment files:

```python
SOURCES_POSITION = [
    [x, y, z],  # where z ∈ [0,1] represents vertical position
    [0.5, 0.5, 1.0],  # Example: center top (Z=1 is top)
    [0.5, 0.5, 0.0],  # Example: center bottom (Z=0 is bottom)
]
```

## Rationale for Z-up

This coordinate system was chosen to:

1. **Align with physics conventions** - Z typically represents vertical dimension in scientific work
2. **Match modern 3D tools** - Blender, Unreal Engine, and many CAD programs use Z-up
3. **Simplify exports** - Easier integration with scientific visualization tools (Matplotlib, Mayavi, PyVista)
4. **Improve intuition** - Users expect Z to represent "height" when defining 3D positions

## Compatibility with Other Software

### Z-up Software (Direct Compatibility)

- Blender
- Shapr3D
- Unreal Engine
- 3ds Max
- Most scientific visualization libraries (Matplotlib 3D, Mayavi, PyVista)

### Y-up Software (Requires Transformation)

- SolidWorks
- Fusion 360 (default, but configurable)
- Maya
- Unity

When exporting to Y-up software, apply this transformation:

```python
# Transform from OpenWave Z-up to Y-up
def to_yup(pos):
    x, y, z = pos
    return [x, z, y]  # Swap Y and Z
```

## Camera Controls

- **Orbit**: Right-click + drag (object rotates in same direction as mouse movement for intuitive control)
- **Zoom**: Q (in) / Z (out) keys
- **Pan**: Arrow keys (screen-space panning)

### Orbit Behavior

When orbiting with right-click drag:

- Drag **right** → Object rotates right
- Drag **left** → Object rotates left
- Drag **up** → Object rotates up
- Drag **down** → Object rotates down

This matches the standard behavior in Blender, Maya, and most 3D software.

### Pan Behavior

Arrow keys pan in screen-space (relative to current camera view):

- **Up arrow** → Pan up (Z+ in world space)
- **Down arrow** → Pan down (Z- in world space)
- **Left arrow** → Pan left (perpendicular to camera view in XY plane)
- **Right arrow** → Pan right (perpendicular to camera view in XY plane)

Left/right panning is **camera-relative**: the orbit center moves sideways on your screen regardless of camera angle. This provides intuitive screen-space control similar to 2D image viewers.

## History

**2025-10**: Changed from Y-up to Z-up coordinate system to align with modern standards and physics conventions. All wave source positions in experiments were updated accordingly.
