# VISUALIZATION SYSTEMS

## Table of Contents

1. [Overview](#overview)
1. [Flux Detector Films/Planes](#flux-detector-filmsplanes)
   - [Purpose and Concept](#purpose-and-concept)
   - [Implementation](#implementation)
   - [Properties and Interactions](#properties-and-interactions)
1. [Universe Boundaries](#universe-boundaries)
   - [Outer Walls](#outer-walls)
   - [Wall as Flux Detectors](#wall-as-flux-detectors)
   - [User Interaction](#user-interaction)
1. [3D Wave Visualization Techniques](#3d-wave-visualization-techniques)
   - [Particle Spray Method](#particle-spray-method)
   - [Vector Fields](#vector-fields)
   - [Streamlines](#streamlines)
   - [Wave Fronts](#wave-fronts)
   - [Ether Visualization](#ether-visualization)
1. [Particle/Wave Center Visualization](#particlewave-center-visualization)
   - [Wave Centers](#wave-centers)
   - [Standing Wave Shells](#standing-wave-shells)
   - [Toggle Options](#toggle-options)
1. [Electron Visualization](#electron-visualization)
   - [Spin Representation](#spin-representation)
   - [Formation Events](#formation-events)
   - [Energy Charging Animation](#energy-charging-animation)
1. [Reference Infrastructure](#reference-infrastructure)
   - [Grid Lines](#grid-lines)
   - [Coordinate Indicators](#coordinate-indicators)
   - [Spatial Orientation](#spatial-orientation)
1. [Implementation Strategy](#implementation-strategy)

## Overview

LEVEL-1 visualization systems convert wave field data into observable visual representations. Unlike LEVEL-0's direct particle rendering, LEVEL-1 requires specialized techniques to visualize field-based waves, interference patterns, and emergent phenomena.

**Visualization Goals**:

- Make wave fields visible (amplitude, phase, energy)
- Show interference patterns and standing waves
- Visualize wave-particle duality
- Provide intuitive understanding of wave mechanics
- Support scientific analysis and debugging

**Primary Rendering Technologies**:

1. **WAVE VIEWING**:
   - **Taichi Meshes**: Flux detector films/planes display wave properties
   - **Color Gradients**: Defined in `config.py` for energy wave visualization
   - **Wave Properties**: Render wave fronts, amplitude, energy density

2. **PARTICLE VIEWING**:
   - **Taichi Particle Rendering**: GPU-accelerated point/sphere rendering
   - **Spray/Cloud Visualization**: Tiny particles display standing waves
   - **Standing Wave Rings**: Positioned at λ/2 intervals around wave centers
   - **Particle Radius**: Visualize one or two wavelengths around particles

3. **VECTOR/FLOW VISUALIZATION**:
   - **Taichi Lines**: Display vector fields (force, direction, amplitude gradient)
   - **Streamlines**: Follow flow/energy propagation paths
   - **Polylines**: Smooth curves showing wave dynamics

## Flux Detector Films/Planes

### Purpose and Concept

**Flux Detector Films** are 2D sensor planes that:

- React to wave parameters passing through them
- Convert wave properties into visible colors
- Act like camera sensors or photographic plates
- Provide 2D "slices" through 3D wave field

**Physical Analogy**:

- Like photographic film detecting light
- Like detector screens in particle physics experiments
- Like ultrasound imaging planes in medical imaging

### Implementation

**Geometry**:

```python
# Detector plane as 2D mesh
detector_plane = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))
detector_color = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

@ti.kernel
def create_detector_plane(position: ti.math.vec3, normal: ti.math.vec3):
    """Create detector plane at position with given normal."""
    for i, j in ti.ndrange(res_x, res_y):
        # Position on plane
        u = (i / res_x - 0.5) * plane_width
        v = (j / res_y - 0.5) * plane_height

        # Tangent vectors
        tangent1, tangent2 = compute_tangents(normal)

        # Point on plane
        detector_plane[i, j] = position + u * tangent1 + v * tangent2
```

**Positioning**:

- Can be placed anywhere in 3D space
- Can be oriented in any direction
- Multiple planes can show different slices
- Can animate plane position for scanning

**Sampling Wave Field**:

```python
@ti.kernel
def sample_field_on_detector():
    """Sample wave field values onto detector plane."""
    for i, j in detector_plane:
        pos = detector_plane[i, j]

        # Sample amplitude at this position (interpolate from grid)
        amp = sample_amplitude_field(pos)

        # Convert to color (colormap)
        detector_color[i, j] = amplitude_to_color(amp)
```

### Properties and Interactions

**Detection Properties**:

- **Amplitude**: Intensity/brightness
- **Frequency**: Color hue (if multi-frequency)
- **Energy**: Overall brightness
- **Phase**: Can show as color variation

**Wave Interactions**:

- Show interference patterns (bright/dark fringes)
- Show wave reflections from particles
- Show standing wave nodes/antinodes
- Show traveling wave motion

**Colormap Options**:

```python
@ti.func
def amplitude_to_color(amp: ti.f32) -> ti.math.vec3:
    """Convert amplitude to RGB color."""
    # Option 1: Grayscale
    intensity = (amp + 1.0) / 2.0  # Map [-1,1] to [0,1]
    return ti.Vector([intensity, intensity, intensity])

    # Option 2: Heatmap (blue-red)
    # Blue = negative, Red = positive
    if amp > 0:
        return ti.Vector([amp, 0.0, 0.0])
    else:
        return ti.Vector([0.0, 0.0, -amp])

    # Option 3: Viridis/Plasma colormap (better for scientific viz)
```

**Rendering with Taichi Meshes**:

- Use **Taichi mesh rendering** for detector planes
- Define **energy wave color gradient** in `config.py`
- Apply color gradient to display wave properties:
  - **Wave fronts**: Surfaces of constant phase
  - **Amplitude**: Intensity mapped to color
  - **Energy density**: Brightness/saturation
- Mesh vertices updated each frame with wave field data

## Universe Boundaries

### Outer Walls

**Boundary Geometry**:

- Cube or rectangular box enclosing simulation domain
- Each face is a potential visualization surface
- Walls are at `x=0, x=L_x, y=0, y=L_y, z=0, z=L_z`

**Mesh Representation**:

```python
# Six wall faces as meshes
wall_faces = []
for face in ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']:
    wall_faces.append(create_wall_mesh(face))
```

**Wave Reflection**:

- Walls reflect waves without energy loss
- Hard boundary condition (phase inversion)
- Creates standing wave patterns near walls

### Wall as Flux Detectors

**Dual Purpose**:

1. Physical boundary (wave reflection)
2. Visualization surface (flux detection)

**Painting Wave Properties**:

```python
@ti.kernel
def render_wall_face(wall_id: ti.i32):
    """Render wave properties on wall face."""
    for i, j in wall_mesh[wall_id]:
        # Get 3D position on wall
        pos = wall_mesh[wall_id].vertices[i, j]

        # Sample amplitude near wall
        amp = sample_amplitude_near_boundary(pos)

        # Color based on amplitude
        wall_mesh[wall_id].colors[i, j] = amplitude_to_color(amp)
```

**Benefits**:

- See wave reflections at boundaries
- Visualize energy distribution on walls
- No need for internal detector planes
- User can observe from outside

### User Interaction

**Camera Orbit**:

- User orbits around universe cube
- Can view from any angle
- See different wall faces
- Zoom in/out for detail

**Wall Transparency**:

- Walls can be semi-transparent
- See waves inside while viewing boundaries
- Adjustable opacity

**Wall Highlighting**:

- Highlight specific walls
- Show/hide individual walls
- Focus on regions of interest

## 3D Wave Visualization Techniques

### Particle Spray Method

**Concept**: Visualize waves using small particles positioned at wave nodes/antinodes.

**Implementation**:

```python
# Small particles for visualization (NOT physics particles)
viz_particles = ti.Vector.field(3, dtype=ti.f32, shape=num_viz_particles)

@ti.kernel
def update_viz_particles():
    """Position particles at wave nodes (high amplitude regions)."""
    count = 0
    for i, j, k in amplitude:
        if ti.abs(amplitude[i,j,k]) > threshold:
            if count < num_viz_particles:
                viz_particles[count] = get_position(i, j, k)
                count += 1
```

**Characteristics**:

- Particles track wave fronts dynamically
- Creates cloud-like or gas-like appearance
- Density of particles = wave amplitude
- Similar to smoke/dust visualization in sound waves

**Rendering with Taichi Particles**:

- Use **Taichi particle rendering system**
- Very small spheres or points
- Color by amplitude or energy
- Transparency for layered view
- GPU-accelerated for millions of visualization particles

**Use Cases**:

- **Standing waves**: Display rings around wave centers at wave nodes
  - Rings positioned at integer multiples of λ/2
  - One or two wavelengths radius for particle visualization
  - Tiny particle spray/cloud forming concentric shells
- **Wave fronts**: Track surfaces of constant phase
- **Energy density**: Particle density represents wave intensity

### Vector Fields

**Arrow Visualization**:

- Display force vectors throughout space
- Arrow direction = force direction
- Arrow length = force magnitude

**Implementation with Taichi Lines**:

```python
@ti.kernel
def render_vector_field(stride: ti.i32):
    """Render arrows for force field (subsampled for clarity)."""
    for i, j, k in ti.ndrange((0, nx, stride),
                               (0, ny, stride),
                               (0, nz, stride)):
        pos = get_position(i, j, k)
        vec = force[i, j, k]

        # Draw arrow from pos in direction vec using Taichi lines
        render_arrow(pos, vec)
```

**Rendering Options**:

- **Taichi lines**: For vector arrows (direction + magnitude)
- **Line primitives**: Efficient GPU rendering
- **Color coding**: Direction (hue) or magnitude (saturation)
- **Subsampling**: Show every Nth vector for clarity

**Use Cases**:

- Visualize force fields (F = -∇A)
- Show wave propagation directions
- Understand particle motion (MAP)
- Display amplitude gradient vectors

### Streamlines

**Concept**: Lines following flow/propagation direction.

**Implementation**:

```python
@ti.kernel
def compute_streamline(start_pos: ti.math.vec3):
    """Trace streamline from start position."""
    pos = start_pos
    for step in range(max_steps):
        # Get direction at current position
        direction = sample_wave_direction(pos)

        # Step along direction
        pos += direction * step_size

        # Store for rendering
        streamline_points[step] = pos
```

**Rendering Flow/Stream Lines**:

- **Taichi lines**: Connect streamline points
- **Polyline rendering**: Smooth curves through points
- **Color coding**:
  - By speed (faster = brighter)
  - By time (gradient from start to end)
  - By energy (local amplitude)
- **Animation**: Time-evolving streamlines showing wave motion

**Use Cases**:

- Show wave propagation paths
- Visualize energy flow directions
- Understand wave dynamics and interference
- Display Poynting vector (energy flux)

### Wave Fronts

**Concept**: Surfaces of constant phase.

**Implementation**:

```python
@ti.kernel
def extract_wavefront(phase_value: ti.f32):
    """Extract isosurface of constant phase."""
    # Marching cubes algorithm
    for i, j, k in ti.ndrange(nx-1, ny-1, nz-1):
        # Check if wavefront passes through this voxel
        if crosses_isosurface(i, j, k, phase_value):
            add_triangle_to_mesh(i, j, k)
```

**Rendering**:

- Semi-transparent surfaces
- Color by amplitude or frequency
- Animate over time to show propagation

### Ether Visualization

**Concept**: Visualize the "ether" (medium) itself.

**Appearance**:

- Bright blue color
- Clear, slightly luminous
- Subtle glow effect
- Fog-like when disturbed by waves

**"Spray" Effect**:

- Medium appears denser where waves are strong
- Reveals 3D wave structure
- Like seeing sound waves in fog

**Implementation**:

```python
@ti.kernel
def render_ether():
    """Render ether with density proportional to wave energy."""
    for i, j, k in amplitude:
        energy_density = amplitude[i,j,k]**2

        # Ether opacity ∝ energy density
        opacity = energy_density / max_energy

        # Blue luminous color
        color = ti.Vector([0.3, 0.5, 1.0]) * opacity

        render_voxel(i, j, k, color, opacity)
```

## Particle/Wave Center Visualization

### Wave Centers

**Appearance**:

- **White particles** (like infrastructure elements)
- Distinct from wave visualization
- Small spheres or points
- Always visible (high contrast)

**Purpose**:

- Show where particles (wave centers) are located
- Track particle motion over time
- Distinguish from wave field visualization

**Rendering**:

```python
@ti.kernel
def render_wave_centers():
    for p in particles:
        if particles.active[p]:
            pos = particles.pos[p]
            # White sphere
            render_sphere(pos, radius=particle_radius,
                         color=ti.Vector([1.0, 1.0, 1.0]))
```

### Standing Wave Shells

**Concept**: Visualize standing wave nodes around wave centers.

**Implementation**:

```python
@ti.kernel
def render_standing_wave_shells(particle_id: ti.i32):
    """Render concentric shells at standing wave nodes."""
    center = particles.pos[particle_id]
    wavelength = get_particle_wavelength(particle_id)

    # Shell radii at λ/2 intervals
    for n in range(num_shells):
        radius = (n + 1) * wavelength / 2.0

        # Render transparent shell
        render_sphere_shell(center, radius,
                           opacity=0.3 / (n+1),  # Fade with distance
                           color=ti.Vector([0.8, 0.9, 1.0]))
```

**Characteristics**:

- Transparent shells
- Progressively smaller particles/opacity at each layer
- Can see through to internal structure
- Shows particle "size" and wave structure

### Toggle Options

**Visibility Controls**:

1. **Wave field only**: No particles visible
2. **Particles only**: No wave field, just centers
3. **Shells only**: Standing wave patterns only
4. **Everything combined**: Full visualization
5. **Detector plate only**: 2D slice view

**User Interface**:

```python
# Visualization flags
show_wave_field = True
show_particles = True
show_shells = False
show_detector = False

@ti.kernel
def render_scene():
    if show_wave_field:
        render_wave_field()
    if show_particles:
        render_wave_centers()
    if show_shells:
        render_standing_wave_shells()
    if show_detector:
        render_detector_plane()
```

## Electron Visualization

### Spin Representation

**2D Simplified Model**:

- Electron spin can be represented in 2D
- Rotating arrow or vector field
- Color coding for spin up/down

**Implementation**:

```python
@ti.kernel
def render_electron_spin(electron_id: ti.i32):
    """Render electron with spin visualization."""
    pos = particles.pos[electron_id]
    spin_axis = particles.spin[electron_id]

    # Render rotating field around electron
    render_spinning_pattern(pos, spin_axis)
```

### Formation Events

**"Click" Visualization**:

- Show two wave centers approaching
- Visualize standing wave pattern changing
- Sudden transition to bound state
- New particle properties emerge

**Animation Sequence**:

1. Two separate wave centers (distinct shells)
2. Centers approach, shells overlap
3. Interference patterns appear
4. Sudden "snap" - shells merge
5. New electron pattern forms (different shell structure)

**Implementation**:

```python
@ti.kernel
def detect_electron_formation():
    """Detect when two centers form electron."""
    for p1, p2 in particle_pairs:
        distance = (particles.pos[p1] - particles.pos[p2]).norm()
        if distance < critical_distance:
            # Check energy and phase conditions
            if formation_conditions_met(p1, p2):
                trigger_formation_event(p1, p2)
```

### Energy Charging Animation

**Show Transformation**:

- Visualize energy being Charged
- Wave pattern changes
- Color shifts during transformation
- Particle properties update

## Reference Infrastructure

### Grid Lines

**Purpose**: Provide spatial reference frame.

**Implementation**:

```python
@ti.kernel
def render_grid_lines():
    """Render reference grid lines."""
    # XY plane grid
    for i in range(0, nx, grid_spacing):
        for j in range(0, ny, grid_spacing):
            draw_line(get_position(i, j, 0), get_position(i, j, nz-1),
                     color=ti.Vector([0.3, 0.3, 0.3]))
```

**Options**:

- Adjustable spacing
- Toggle on/off
- Different colors for different planes

### Coordinate Indicators

**Axes**:

- X, Y, Z axes in different colors
- Origin marker
- Scale indicators

**Labels**:

- Axis labels (X, Y, Z)
- Distance markers (in meters or attometers)
- Coordinate values

### Spatial Orientation

**Orientation Aids**:

- Compass rose (show viewing direction)
- Rotation indicator
- Scale bar (show physical size)

**Implementation**:

```python
# Corner overlay showing orientation
render_axes_overlay(corner='bottom_left')
render_scale_bar(corner='bottom_right')
```

## Implementation Strategy

### Rendering Pipeline

1. **Compute**: Update wave field (physics)
2. **Sample**: Extract visualization data from fields
3. **Process**: Convert data to colors/geometries
4. **Render**: Display using graphics API (OpenGL/Vulkan)

### Performance Considerations

**Subsampling**:

- Don't visualize every voxel (too many)
- Subsample at stride for vector fields
- Adaptive resolution based on zoom

**Level-of-Detail**:

- Detailed rendering near camera
- Simplified rendering far from camera
- Adjust particle counts dynamically

**GPU Acceleration**:

- All visualization kernels run on GPU
- Minimal CPU-GPU transfers
- Direct rendering from Taichi fields

### Recommended Stack

**Graphics Library**:

- **GGUI** (Taichi native): Simple, integrated
- **OpenGL** (via moderngl): More control
- **Vulkan**: Maximum performance (complex)

**Colormaps**:

- Matplotlib colormaps (viridis, plasma)
- Custom physical colormaps
- Perceptually uniform scales

---

**Status**: Visualization systems defined

**Next Steps**: Implement basic detector plane and wave field rendering

**Related Documentation**:

- [`02_WAVE_PROPERTIES.md`](./02_WAVE_PROPERTIES.md) - Properties to visualize
- [`03_WAVE_ENGINE.md`](./03_WAVE_ENGINE.md) - Wave field to visualize
- [`05_MATTER.md`](./05_MATTER.md) - Particles to visualize
