# FLUX FILM VISUALIZATION

## Table of Contents

1. [Overview](#overview)
1. [Concept and Terminology](#concept-and-terminology)
1. [Physical Analogy](#physical-analogy)
1. [Geometry and Positioning](#geometry-and-positioning)
   - [Initial Configuration](#initial-configuration)
   - [Intersection Design](#intersection-design)
   - [Universe Coordinate System](#universe-coordinate-system)
1. [Mesh Implementation](#mesh-implementation)
   - [Taichi Mesh Rendering](#taichi-mesh-rendering)
   - [Mesh Resolution](#mesh-resolution)
   - [Vertex-to-Voxel Mapping](#vertex-to-voxel-mapping)
1. [Wave Property Visualization](#wave-property-visualization)
   - [Property Sampling](#property-sampling)
   - [Color Gradient Mapping](#color-gradient-mapping)
   - [Supported Properties](#supported-properties)
1. [Color Palettes](#color-palettes)
   - [Amplitude Visualization (Ironbow)](#amplitude-visualization-ironbow)
   - [Displacement Visualization (Redshift)](#displacement-visualization-redshift)
   - [Blueprint Palette](#blueprint-palette)
1. [User Interface Controls](#user-interface-controls)
   - [Toggle Controls](#toggle-controls)
   - [Camera Interaction](#camera-interaction)
1. [Implementation Details](#implementation-details)
   - [Coordinate Transformation](#coordinate-transformation)
   - [Two-Sided Rendering](#two-sided-rendering)
   - [Performance Considerations](#performance-considerations)
1. [Future Extensions](#future-extensions)

## Overview

**Flux Films** are 2D cross-sectional detector surfaces that provide real-time views of wave propagation through the universe simulation domain. They function as photographic films that react to wave flux and convert wave properties into visible color gradients.

**Purpose**:

- Monitor wave propagation in real-time
- Visualize wave interference patterns
- Display force vector fields
- Show wave mode distributions
- Analyze energy flow through space

**Key Features**:

- Three orthogonal films (XY, XZ, YZ)
- Intersect at center of universe domain
- Mesh resolution matches simulation resolution
- Dynamic color mapping of wave properties
- Two-sided rendering for full visibility

**Current Implementation Status** (as of 2025-11-17):

- ✅ **UI Toggle**: Implemented in `launcher_L1.py`
  - Location: `launcher_L1.py:222` (CONTROLS window)
  - Variable: `state.flux_films` (Boolean)
  - Control: Single checkbox "Flux Films" for all three films
  - Default: `False` (films hidden)
  - Configuration: Via xparameters `ui_defaults["flux_films"]`

- 🔄 **In Progress**: Core mesh rendering and wave property visualization
  - Mesh generation functions (to be implemented)
  - Property sampling kernels (to be implemented)
  - Color gradient mapping (redshift gradient needed)
  - Integration with render loop (to be wired up)

## Concept and Terminology

### Naming Convention

The feature is called **"Flux Films"** (plural):

- **Flux Films**: Primary term - three 2D detector films visualizing wave flux
- **Flux Film**: Singular - referring to the system/feature or an individual film
- **Alternative names considered**: Flux Screens, Flux Sensors, Flux Detectors, Plane Slice

**Flux** represents the energy wave field passing through the films, while **film** represents the detector surface that captures wave properties and converts them to visual information (like photographic film, X-ray film, or nuclear emulsion).

**Singular vs Plural Usage**:

- **Plural "Flux Films"**: When referring to the three detector surfaces (UI label, multiple objects)
- **Singular "Flux Film"**: When referring to the feature/system or a specific individual film

### Inspiration

Based on digital image sensor technology:

- CCD/CMOS sensor arrays in cameras
- Each pixel samples light intensity and color
- 2D array captures 3D scene information
- Reference: [Canon Image Sensors](https://www.canon.com.cy/pro/infobank/image-sensors-explained)

### Relation to LEVEL-0

Similar to LEVEL-0's `block_slice` feature but adapted for voxel-based wave fields:

- LEVEL-0: Sliced granule lattice to show interior structure
- LEVEL-1: Flux films showing wave field properties

### Reserved Terminology

**"Field Sensor"** is reserved for future point measurement features:

- **Flux Films**: 2D surface detectors (current feature - plural for three films)
- **Field Sensor**: Point probe measuring specific voxel properties (future feature)

## Physical Analogy

**Flux films act like**:

1. **Photographic Film**: Detecting light (energy waves)
2. **Detector Screens**: In particle physics experiments (bubble chambers, cloud chambers)
3. **Ultrasound Imaging Planes**: Medical imaging cross-sections
4. **Oscilloscope Screen**: Displaying waveforms in 2D
5. **Schlieren Photography**: Visualizing density gradients in fluids

## Geometry and Positioning

### Initial Configuration

Three orthogonal flux films intersecting at the universe center:

```python
# Universe domain: [0, 1] in normalized coordinates
# Real universe: [0, universe_size] in meters (attometers)

# Center position (real coordinates)
center_pos = universe_size * 0.5  # Center of domain

# Flux film positions
film_xy_z = center_pos  # XY film at z = 0.5
film_xz_y = center_pos  # XZ film at y = 0.5
film_yz_x = center_pos  # YZ film at x = 0.5
```

### Intersection Design

All three flux films intersect at the central voxel:

```text
        Y
        ↑
        |
        +------- X
       /|
      / |
     Z  |

    [0.5, 0.5, 0.5] - Intersection point

    XY Film: spans (0→1, 0→1, 0.5)
    XZ Film: spans (0→1, 0.5, 0→1)
    YZ Film: spans (0.5, 0→1, 0→1)
```

**Visual Result**: Three mutually perpendicular flux films forming a 3D cross at the universe center.

### Universe Coordinate System

Two coordinate systems are used:

1. **Real Universe Coordinates** (meters/attometers)
   - Used for physics calculations
   - Voxel positions in physical space
   - Wave property calculations

2. **Normalized Screen Coordinates** [0, 1]
   - Used for rendering
   - Camera and display positioning
   - User interaction

**Conversion**:

```python
# Real to normalized
normalized_pos = real_pos / universe_size

# Normalized to real
real_pos = normalized_pos * universe_size
```

## Mesh Implementation

### Taichi Mesh Rendering

Each flux film is rendered as a **Taichi mesh**:

```python
import taichi as ti

# Mesh for XY film
mesh_resolution_x = target_voxels_x
mesh_resolution_y = target_voxels_y

# Vertex positions (3D coordinates)
film_xy_vertices = ti.Vector.field(
    n=3,
    dtype=ti.f32,
    shape=(mesh_resolution_x, mesh_resolution_y)
)

# Vertex colors (RGB)
film_xy_colors = ti.Vector.field(
    n=3,
    dtype=ti.f32,
    shape=(mesh_resolution_x, mesh_resolution_y)
)

# Indices for triangulation (two triangles per quad)
film_xy_indices = ti.field(
    dtype=ti.i32,
    shape=(mesh_resolution_x-1, mesh_resolution_y-1, 6)
)
```

### Mesh Resolution

**Resolution matches simulation voxel grid**:

```python
# If simulation has target_voxels = 64³
# XY film mesh: 64 × 64 vertices
# XZ film mesh: 64 × 64 vertices
# YZ film mesh: 64 × 64 vertices

# Each quad (voxel face) = 2 triangles
# Total triangles per film: (64-1) × (64-1) × 2 = 7,938 triangles
```

**Benefits**:

- Each mesh vertex corresponds to a voxel center
- Direct 1:1 mapping of wave properties
- No interpolation needed
- Efficient property lookup

### Vertex-to-Voxel Mapping

**Mesh Construction** (in real coordinates, then normalized):

```python
@ti.kernel
def create_film_xy_mesh():
    """Create XY flux film mesh at z = center_z."""
    for i, j in ti.ndrange(mesh_resolution_x, mesh_resolution_y):
        # Real universe coordinates
        x = (i / mesh_resolution_x) * universe_size
        y = (j / mesh_resolution_y) * universe_size
        z = center_z  # Fixed at 0.5 * universe_size

        # Store vertex position (real coordinates)
        film_xy_vertices[i, j] = ti.Vector([x, y, z])

        # Map to voxel index for property lookup
        voxel_i = i  # Direct mapping
        voxel_j = j
        voxel_k = mesh_resolution_z // 2  # Center slice

        # Sample wave property at voxel
        property_value = wave_field[voxel_i, voxel_j, voxel_k]

        # Convert to color
        film_xy_colors[i, j] = property_to_color(property_value)
```

**Rendering** (normalized coordinates):

```python
@ti.kernel
def normalize_film_vertices():
    """Convert real coordinates to normalized [0,1] for rendering."""
    for i, j in film_xy_vertices:
        real_pos = film_xy_vertices[i, j]
        film_xy_vertices[i, j] = real_pos / universe_size
```

## Wave Property Visualization

### Property Sampling

Flux films can visualize any voxel-based property:

**Wave Field Properties**:

- `displacement`: Current wave displacement (signed, can be negative)
- `amplitude`: Wave amplitude envelope (absolute value)
- `velocity`: Wave velocity field (signed)
- `energy`: Energy density (positive)
- `frequency`: Local frequency (for multi-frequency simulations)
- `phase`: Wave phase (0 to 2π)
- `wave_mode`: Wave type indicator (longitudinal, transverse)

**Force Field Properties**:

- `force_gravity`: Gravitational force magnitude/direction
- `force_electric`: Electric force magnitude/direction
- `force_magnetic`: Magnetic force magnitude/direction
- `force_total`: Total force vector

### Color Gradient Mapping

**Mapping Process**:

1. Sample property value at voxel center
2. Normalize value to [0, 1] range
3. Apply color gradient function
4. Assign RGB color to mesh vertex

```python
@ti.func
def property_to_color(value: ti.f32,
                      min_val: ti.f32,
                      max_val: ti.f32,
                      gradient_type: ti.i32) -> ti.math.vec3:
    """Convert property value to RGB color using specified gradient.

    Args:
        value: Property value to visualize
        min_val: Minimum expected value
        max_val: Maximum expected value
        gradient_type: 0=ironbow, 1=redshift, 2=blueprint

    Returns:
        RGB color vector [0,1]³
    """
    # Normalize to [0,1]
    norm_value = (value - min_val) / (max_val - min_val)
    norm_value = ti.math.clamp(norm_value, 0.0, 1.0)

    color = ti.Vector([0.0, 0.0, 0.0])

    if gradient_type == 0:  # Ironbow
        color = ironbow_gradient(norm_value)
    elif gradient_type == 1:  # Redshift
        color = redshift_gradient(norm_value)
    elif gradient_type == 2:  # Blueprint
        color = blueprint_gradient(norm_value)

    return color
```

### Supported Properties

**Initial Implementation**:

1. **Displacement** (primary)
   - Signed value (positive/negative)
   - Use redshift gradient (red=positive, blue=negative)
   - Shows wave fronts and interference

2. **Amplitude** (primary)
   - Absolute value (always positive)
   - Use ironbow gradient (black=zero, white=maximum)
   - Shows energy distribution

**Future Properties**:

- Force vectors (requires arrow rendering)
- Wave mode (requires discrete color map)
- Phase (requires cyclic colormap)

## Color Palettes

### Amplitude Visualization (Ironbow)

**Use Case**: Visualizing absolute-valued properties (amplitude, energy)

**Gradient**: Black → Dark Blue → Magenta → Red-Orange → Yellow-White

**Implementation**: Already defined in `config.py` as `get_ironbow_color()`

```python
# From config.py
@ti.func
def get_ironbow_color(value, min_value, max_value, saturation=1.0):
    """Maps value to ironbow thermal gradient.

    Gradient: black → dark blue → magenta → red-orange → yellow-white
    Good for: amplitude, energy, absolute values
    """
    # See openwave/common/config.py for full implementation
    pass
```

**Color Stops**:

```python
ironbow = [
    ["#000000", (0.0, 0.0, 0.0)],        # black (zero amplitude)
    ["#20008A", (0.125, 0.0, 0.54)],     # dark blue (low)
    ["#91009C", (0.57, 0.0, 0.61)],      # magenta (medium)
    ["#E64616", (0.90, 0.27, 0.09)],     # red-orange (high)
    ["#FFFFF6", (1.0, 1.0, 0.96)],       # yellow-white (maximum)
]
```

**Visual Mapping**:

- **Black**: Zero amplitude (no wave)
- **Blue/Magenta**: Low to medium amplitude
- **Orange/Yellow**: High amplitude
- **White**: Maximum amplitude (saturation)

### Displacement Visualization (Redshift)

**Use Case**: Visualizing signed properties (displacement, velocity)

**Gradient**: Red ← Zero → Blue (inspired by Doppler shift)

**New Gradient Definition** (to be added to `config.py`):

```python
# Redshift Doppler-Inspired Palette
# ================================================================
# 5-color gradient for signed wave displacement
redshift = [
    ["#8B0000", (0.545, 0.0, 0.0)],     # dark red (maximum negative)
    ["#FF6347", (1.0, 0.39, 0.28)],     # red-orange (negative)
    ["#1C1C1C", (0.11, 0.11, 0.11)],    # dark gray (zero)
    ["#4169E1", (0.255, 0.41, 0.88)],   # blue (positive)
    ["#00008B", (0.0, 0.0, 0.545)],     # dark blue (maximum positive)
]

@ti.func
def get_redshift_color(value, min_value, max_value, saturation=1.0):
    """Maps signed value to redshift gradient (red-blue).

    Gradient: dark red → red → gray → blue → dark blue
    Red = negative displacement (redshift)
    Blue = positive displacement (blueshift)
    Gray = zero displacement

    Good for: displacement, velocity, signed values

    Args:
        value: Property value (can be negative)
        min_value: Expected minimum (negative)
        max_value: Expected maximum (positive)
        saturation: Headroom factor (> 1.0 prevents saturation)

    Returns:
        RGB color vector [0,1]³
    """
    # Normalize to [-1, 1] then to [0, 1]
    scale = (max_value - min_value) * saturation
    norm_value = (value - min_value) / scale
    norm_value = ti.math.clamp(norm_value, 0.0, 1.0)

    r, g, b = 0.0, 0.0, 0.0

    if norm_value < 0.25:  # dark red to red
        blend = norm_value / 0.25
        r = 0.545 * (1.0 - blend) + 1.0 * blend
        g = 0.0 * (1.0 - blend) + 0.39 * blend
        b = 0.0 * (1.0 - blend) + 0.28 * blend

    elif norm_value < 0.5:  # red to gray (zero)
        blend = (norm_value - 0.25) / 0.25
        r = 1.0 * (1.0 - blend) + 0.11 * blend
        g = 0.39 * (1.0 - blend) + 0.11 * blend
        b = 0.28 * (1.0 - blend) + 0.11 * blend

    elif norm_value < 0.75:  # gray to blue
        blend = (norm_value - 0.5) / 0.25
        r = 0.11 * (1.0 - blend) + 0.255 * blend
        g = 0.11 * (1.0 - blend) + 0.41 * blend
        b = 0.11 * (1.0 - blend) + 0.88 * blend

    else:  # blue to dark blue
        blend = (norm_value - 0.75) / 0.25
        r = 0.255 * (1.0 - blend) + 0.0 * blend
        g = 0.41 * (1.0 - blend) + 0.0 * blend
        b = 0.88 * (1.0 - blend) + 0.545 * blend

    return ti.Vector([r, g, b])
```

**Physical Interpretation**:

- **Red**: Wave compressed (moving away, redshift)
- **Blue**: Wave expanded (moving toward, blueshift)
- **Gray**: Equilibrium position (no displacement)

### Blueprint Palette

**Use Case**: Alternative low-intensity visualization

**Gradient**: Dark Blue → Medium Blue → Light Blue → Extra-Light Blue

**Implementation**: Already defined in `config.py` as `blueprint4`

```python
# From config.py
blueprint4 = [
    ["#192C64", (0.1, 0.17, 0.39)],      # dark blue
    ["#405CB1", (0.25, 0.36, 0.69)],     # medium blue
    ["#98AEDD", (0.6, 0.68, 0.87)],      # light blue
    ["#E4EAF6", (0.9, 0.94, 0.98)],      # extra-light blue
]
```

**Use**: Subtle background visualization or amplitude alternative

## User Interface Controls

### Toggle Controls

**L1 Launcher UI Implementation**:

The Flux Film system is implemented in `launcher_L1.py` with a **single toggle for all three films**:

```python
# SimulationState class (launcher_L1.py:123)
self.flux_films = False  # Single toggle for all flux films

# Apply from xparameters (launcher_L1.py:164)
self.flux_films = ui["flux_films"]

# UI Control (launcher_L1.py:222)
with render.gui.sub_window("CONTROLS", 0.00, 0.34, 0.15, 0.22) as sub:
    state.flux_films = sub.checkbox("Flux Films", state.flux_films)
```

**Configuration in Xparameters**:

```python
# From _xparameters/energy_wave.py:27
"ui_defaults": {
    "flux_films": False,  # Flux Films toggle (all three films)
    # ... other UI defaults
}
```

**Current Implementation**:

- **Single toggle controls all three films** (XY, XZ, YZ)
- Default state: `False` (films hidden)
- Toggleable via checkbox in CONTROLS window
- Simple on/off functionality for initial implementation

**Future Enhancements** (optional):

```python
# Individual film toggles (if needed)
state.show_film_xy = True   # XY film at z=0.5
state.show_film_xz = True   # XZ film at y=0.5
state.show_film_yz = True   # YZ film at x=0.5

# Property selection (future)
state.property_mode = 0  # 0=displacement, 1=amplitude, 2=energy

# Color gradient selection (future)
state.gradient_mode = 0  # 0=ironbow, 1=redshift, 2=blueprint
```

### Camera Interaction

**Orbiting Camera**:

- User can orbit camera around universe cube
- View flux films from any angle
- See both sides of each film (two-sided rendering)
- Zoom in/out for detail inspection

**Benefits of Two-Sided Rendering**:

- Front face shows wave properties from one side
- Back face shows wave properties from other side
- Useful when camera is behind a film
- No occlusion issues

## Implementation Details

### Coordinate Transformation

**Two-Stage Process**:

1. **Construction** (real coordinates):
   - Compute vertex positions in real universe coordinates
   - Sample wave properties at voxel centers
   - Apply color gradients

2. **Rendering** (normalized coordinates):
   - Transform vertices to [0,1]³ for display
   - Pass to Taichi rendering pipeline

```python
@ti.kernel
def update_flux_film(film_id: ti.i32):
    """Update flux film vertices and colors.

    Args:
        film_id: 0=XY, 1=XZ, 2=YZ
    """
    # 1. Sample wave properties at voxel centers (real coords)
    for i, j in film_vertices[film_id]:
        # Get real position
        pos_real = get_film_vertex_position(film_id, i, j)

        # Map to voxel indices
        voxel_idx = position_to_voxel_index(pos_real)

        # Sample property
        prop_value = wave_field[voxel_idx]

        # Convert to color
        film_colors[film_id][i, j] = property_to_color(prop_value)

    # 2. Normalize positions for rendering
    for i, j in film_vertices[film_id]:
        film_vertices[film_id][i, j] /= universe_size
```

### Two-Sided Rendering

**Taichi Mesh Two-Sided Mode**:

```python
# When rendering with GGUI
scene.mesh(
    vertices=film_xy_vertices,
    colors=film_xy_colors,
    indices=film_xy_indices,
    two_sided=True  # Enable two-sided rendering
)
```

**Effect**:

- Both front and back faces are rendered
- No backface culling
- Visible from all viewing angles
- Essential for cross-sectional flux films

### Performance Considerations

**Efficiency**:

- Mesh resolution = simulation resolution (no extra overhead)
- Direct voxel-to-vertex mapping (no interpolation)
- GPU-accelerated mesh rendering (Taichi)
- Only visible films are rendered (toggle control)

**Memory**:

```python
# Per film (64³ simulation):
# Vertices: 64 × 64 × 3 floats = 49,152 bytes ≈ 48 KB
# Colors: 64 × 64 × 3 floats = 49,152 bytes ≈ 48 KB
# Indices: 63 × 63 × 6 ints = 95,256 bytes ≈ 93 KB
# Total per film: ~189 KB
# Total for 3 films: ~567 KB (negligible)
```

**Update Frequency**:

- Update every frame for real-time visualization
- Can reduce update frequency for performance (every N frames)
- Interpolation possible for smooth animation

## Future Extensions

### Dynamic Positioning

- **User-Controlled Film Position**: Slider to move films along their normal axis
- **Scanning Animation**: Automatically sweep films through domain
- **Multiple Films**: More than 3 films for detailed analysis

### Advanced Properties

- **Vector Field Visualization**: Display force arrows on films
- **Streamline Overlay**: Show energy flow paths on films
- **Phase Visualization**: Cyclic colormap for wave phase

### Universe Wall Visualization

**Same Technology**:

- Universe boundary walls can also react to wave properties
- Six wall faces (±X, ±Y, ±Z) as flux films
- Shows wave reflections at boundaries
- User can observe from outside cube

**Future Implementation**:

```python
# Six wall faces as flux films
wall_films = [
    "x_min", "x_max",  # YZ films at x=0 and x=1
    "y_min", "y_max",  # XZ films at y=0 and y=1
    "z_min", "z_max",  # XY films at z=0 and z=1
]

# Render wave properties on walls
for wall in wall_films:
    render_wall_as_flux_film(wall)
```

### Interactive Selection

- **Click to Place Film**: User clicks to position films
- **Drag to Orient**: Custom film orientations (not just orthogonal)
- **Region of Interest**: Zoom and clip to specific regions

### Field Sensors (Future Feature)

**Point Measurement Devices**:

- Place individual field sensors at specific voxel locations
- Read scalar values (amplitude, displacement, energy)
- Display values in real-time
- Track sensor readings over time (graphs/plots)
- Different visualization from 2D flux films

---

**Status**: Documented with UI toggle implemented

**Implementation Status**:

- ✅ **UI Toggle**: Implemented in `launcher_L1.py` (line 222)
  - Single checkbox "Flux Films" controls all three films
  - Integrated with xparameters system
  - Default: `flux_films = False`

- ⏳ **Remaining Tasks**:
  1. Implement `get_redshift_color()` function in `config.py`
  2. Create flux film mesh generation functions in `config.py`
  3. Implement property sampling and color mapping kernels
  4. Wire up `state.flux_films` toggle to mesh rendering
  5. Test with wave propagation simulations

**Integration Points**:

```python
# launcher_L1.py - render_elements() function (line 382)
def render_elements(state):
    """Render spacetime elements with appropriate coloring."""
    # Grid Visualization
    if state.SHOW_GRID:
        render.scene.lines(state.wave_field.wire_frame, width=1, color=config.COLOR_MEDIUM[1])

    # Flux Films Visualization
    if state.flux_films:
        config.render_flux_films(state.wave_field, render.scene)
```

**Module Structure**:

All flux film functionality will be implemented in existing modules (no new files):

```text
openwave/
└── common/
    └── config.py  (ADD flux film functions)
        ├── get_redshift_color()         # NEW: Redshift gradient for signed values
        ├── create_flux_film_meshes()    # NEW: Initialize 3 film meshes
        ├── update_flux_film_colors()    # NEW: Sample wave properties and apply colors
        └── render_flux_films()          # NEW: Render meshes to scene
```

**Rationale**:

- Keep flux film rendering functions with other color/rendering utilities
- Similar to existing `get_ironbow_color()` and `ironbow_palette()` functions
- Avoids creating new module for visualization helpers
- Maintains consistency with project's current structure

**Related Documentation**:

- [`07_VISUALIZATION.md`](./07_VISUALIZATION.md) - Complete visualization systems overview
- [`01b_WAVE_FIELD_properties.md`](./01b_WAVE_FIELD_properties.md) - Wave properties to visualize
- [`02_WAVE_ENGINE.md`](./02_WAVE_ENGINE.md) - Wave field data structure
- `launcher_L1.py:222` - Plane-slice UI toggle implementation
