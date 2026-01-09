# GGUI Mesh Rendering Concepts

## Summary

This document explains fundamental concepts for rendering 3D meshes using Taichi GGUI's `scene.mesh()` API. Key topics include mesh topology (vertices, indices, triangles), surface normals for lighting, per-vertex color interpolation, and wireframe visualization. Understanding these concepts is critical for creating realistic 3D visualizations in physics simulations.

## What is a Mesh?

A **mesh** is a collection of vertices, edges, and faces that define a 3D object's surface. In computer graphics, meshes are typically composed of **triangles** because:

- Triangles are always planar (3 points define a plane)
- GPUs are optimized for triangle rasterization
- Any polygon can be decomposed into triangles

## Mesh Components

### 1. Vertices

**Definition**: 3D points in space that define the mesh's geometry.

**In Taichi**:

```python
# 8 vertices for a cube (but see "Normals" section for why we use 24)
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
vertices[0] = ti.Vector([0.0, 0.0, 0.0])  # Corner at origin
vertices[1] = ti.Vector([1.0, 0.0, 0.0])  # Corner at X=1
# ... etc
```

### 2. Indices (Triangle Definitions)

**Definition**: Triplets of vertex indices that define triangles.

**In Taichi**:

```python
# 12 triangles for cube (2 per face � 6 faces)
indices = ti.field(ti.i32, shape=12 * 3)  # 36 indices total

# Front face, first triangle
indices[0] = 0  # Vertex 0
indices[1] = 1  # Vertex 1
indices[2] = 2  # Vertex 2

# Front face, second triangle
indices[3] = 0  # Vertex 0
indices[4] = 2  # Vertex 2
indices[5] = 3  # Vertex 3
```

**Important**: Vertex order matters! Counter-clockwise winding (when viewed from outside) is the standard convention.

### 3. Normals (Surface Orientation)

**Definition**: Unit vectors perpendicular to surfaces, used for lighting calculations.

**Why they matter**: Normals determine how light reflects off surfaces, creating the perception of 3D depth and shape.

**Key insight**: For proper lighting on objects with sharp edges (like cubes), **each face needs its own set of vertices with consistent normals**.

**Example - Wrong approach (shared vertices)**:

```python
# PROBLEM: 8 vertices, 8 normals
# Corner vertices belong to 3 faces with different normals
# GPU will average normals, creating smooth/rounded appearance
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
normals = ti.Vector.field(3, dtype=ti.f32, shape=8)
# Result: Cube looks like a sphere (smooth shading)
```

**Example - Correct approach (duplicated vertices)**:

```python
# SOLUTION: 24 vertices (4 per face � 6 faces)
# Each face has its own vertices with consistent normals
vertices = ti.Vector.field(3, dtype=ti.f32, shape=24)
normals = ti.Vector.field(3, dtype=ti.f32, shape=24)

# Front face (vertices 0-3) - all normals point in -Z direction
vertices[0] = ti.Vector([0.0, 0.0, 0.0])
vertices[1] = ti.Vector([1.0, 0.0, 0.0])
vertices[2] = ti.Vector([1.0, 1.0, 0.0])
vertices[3] = ti.Vector([0.0, 1.0, 0.0])
normals[0] = ti.Vector([0.0, 0.0, -1.0])
normals[1] = ti.Vector([0.0, 0.0, -1.0])
normals[2] = ti.Vector([0.0, 0.0, -1.0])
normals[3] = ti.Vector([0.0, 0.0, -1.0])

# Back face (vertices 4-7) - all normals point in +Z direction
# ... different positions, but consistent +Z normals
```

**Rule of thumb**: Duplicate vertices at edges/corners where surface orientation changes abruptly.

### 4. Colors (Per-Vertex)

**Definition**: RGB color values assigned to each vertex.

**How it works**: The GPU performs **barycentric interpolation** across triangle surfaces.

**Interpolation math**:

```text
For any point P inside triangle (V1, V2, V3):
  color(P) = w1 * color(V1) + w2 * color(V2) + w3 * color(V3)

where w1, w2, w3 are barycentric weights that sum to 1.0
```

**Result**: Smooth color gradients across triangle surfaces, even with only 3 color values per triangle.

**In Taichi**:

```python
# Per-vertex colors (same count as vertices)
colors = ti.Vector.field(3, dtype=ti.f32, shape=24)

@ti.kernel
def update_colors(t: ti.f32):
    for i in range(24):
        # Each vertex gets its own color based on position/time
        wave = ti.sin(t * 3.0 + vertices[i][0] * 5.0)
        brightness = 0.3 + (wave + 1.0) * 0.35
        colors[i] = ti.Vector([1.0 * brightness, 0.5 * brightness, 0.0])

# GPU automatically interpolates between vertex colors
scene.mesh(vertices, indices=indices, normals=normals, per_vertex_color=colors)
```

**Why it looks realistic**: The smooth gradients from interpolation mimic how light naturally varies across surfaces. This same technique is used for:

- Lighting calculations (Gouraud shading)
- Texture coordinate interpolation
- Normal mapping
- All modern 3D rendering

## Taichi GGUI Mesh API

### Basic Usage

```python
scene.mesh(
    vertices,              # ti.Vector.field(3, dtype=ti.f32)
    indices=indices,       # ti.field(ti.i32) - triangle definitions
    normals=normals,       # ti.Vector.field(3, dtype=ti.f32) - per-vertex normals
    color=(1.0, 0.5, 0.0), # Uniform color (alternative to per_vertex_color)
)
```

### Advanced Usage with Per-Vertex Colors

```python
scene.mesh(
    vertices,
    indices=indices,
    normals=normals,
    per_vertex_color=colors,  # ti.Vector.field(3, dtype=ti.f32)
    show_wireframe=False,     # Optional: show triangle edges
)
```

### Wireframe Visualization

**Parameter**: `show_wireframe` (boolean, default: `False`)

**What it does**:

- `show_wireframe=False`: Renders only filled triangles with smooth shading
- `show_wireframe=True`: Renders **both** filled triangles AND lines along all triangle edges

**Use cases**:

1. **Debugging mesh topology** - Verify vertex connections, identify gaps/overlaps
2. **Visualizing mesh density** - Understand triangle distribution and resolution
3. **Technical visualization** - CAD-like views showing mesh structure
4. **Artistic effects** - "Blueprint" or technical drawing aesthetics

**Example**:

```python
# Debug view - see mesh structure
scene.mesh(vertices, indices=indices, normals=normals,
           per_vertex_color=colors, show_wireframe=True)
```

**Performance**: Small overhead from rendering additional line geometry, but negligible for typical mesh sizes.

## Best Practices

### 1. Pre-Allocate Fields

Create Taichi fields **once** during initialization, not per-frame:

```python
# GOOD - in initialization
def init_mesh():
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=24)
    normals = ti.Vector.field(3, dtype=ti.f32, shape=24)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=24)
    populate_geometry()  # Taichi kernel

# BAD - in render loop
while window.running:
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=24)  # Memory leak!
```

### 2. Use Taichi Kernels for Updates

Avoid numpy operations and CPU-to-GPU transfers in render loops:

```python
# GOOD - Taichi kernel (parallel GPU execution)
@ti.kernel
def update_mesh_colors(t: ti.f32):
    for i in range(num_vertices):
        colors[i] = calculate_color(vertices[i], t)

# BAD - numpy/Python loops (slow, CPU-to-GPU transfer)
colors_np = np.zeros((num_vertices, 3))
for i in range(num_vertices):
    colors_np[i] = calculate_color(vertices[i].to_numpy(), t)
colors.from_numpy(colors_np)  # Expensive transfer!
```

### 3. Proper Normal Calculation

For sharp edges, duplicate vertices at corners and assign consistent normals per face.

**Cube example**: 24 vertices (4 per face � 6 faces), not 8 shared vertices.

**Sphere example**: Shared vertices are fine because surface is smooth everywhere.

### 4. Winding Order Consistency

Use counter-clockwise vertex order (viewed from outside) for proper backface culling:

```python
# Counter-clockwise when viewed from outside (+Z looking at front face)
indices[0] = 0  # Bottom-left
indices[1] = 1  # Bottom-right
indices[2] = 2  # Top-right
```

## Common Pitfalls

### Flat/Unrealistic Appearance

**Problem**: Mesh looks flat despite being 3D.

**Cause**: Shared vertices with averaged normals at sharp edges.

**Solution**: Duplicate vertices at edges/corners, assign per-face normals.

### Performance Degradation

**Problem**: FPS drops when rendering mesh.

**Cause**: Creating fields or transferring data in render loop.

**Solution**: Pre-allocate fields, use Taichi kernels for updates.

### Missing Triangles

**Problem**: Some faces don't render or appear inside-out.

**Cause**: Incorrect winding order or index definitions.

**Solution**: Verify counter-clockwise winding, check index triplets.

## Reference Implementation

See `test_mesh.py` for a complete working example:

- Cube mesh with 24 vertices and proper per-face normals
- Dynamic per-vertex color wave effect
- Performance testing against 1.2M particle simulation
- No performance degradation observed

## Related Documentation

- `lines_vs_particles.md` - Investigation of GGUI rendering performance
- `test_mesh.py` - Mesh performance test script
- `/openwave/i_o/render.py` - Production rendering engine

## Environment

- Taichi version: 1.7.4
- Platform: macOS (Darwin 24.6.0)
- Architecture: Apple Silicon (Metal backend)
- Backend: ti.gpu
