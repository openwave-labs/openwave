# Loop Optimization Patterns

## Overview

Loops are critical performance bottlenecks in physics simulations. This guide provides specific patterns and techniques for optimizing loops in OpenWave, with detailed loop patterns, vectorization techniques, Taichi kernel patterns, memory access optimization, and common anti-patterns.

## Loop Optimization Principles

### Hierarchy of Optimization

1. **Eliminate loops** - Use vectorization when possible
2. **Reduce iterations** - Optimize algorithm complexity
3. **Optimize loop body** - Minimize work per iteration
4. **Parallelize** - Use Taichi kernels or multiprocessing
5. **Hardware optimization** - Cache-friendly access patterns

## Python Loop Patterns

### Vectorization with NumPy

```python
# Bad: Python loop
result = []
for i in range(len(array)):
    result.append(array[i] * 2 + 1)

# Good: Vectorized
result = array * 2 + 1

# Bad: Nested loops for matrix operations
for i in range(n):
    for j in range(m):
        C[i,j] = A[i,j] + B[i,j]

# Good: NumPy operations
C = A + B
```

### List Comprehensions vs Loops

```python
# Bad: Accumulating in loop
result = []
for item in items:
    if condition(item):
        result.append(transform(item))

# Better: List comprehension
result = [transform(item) for item in items if condition(item)]

# Best for large data: Generator expression
result = (transform(item) for item in items if condition(item))
```

## Taichi Kernel Patterns

### Basic Parallel Loop

```python
@ti.kernel
def process_particles():
    # Parallel iteration over all particles
    for i in range(n_particles):
        # Process particle i
        position[i] += velocity[i] * dt
```

### Nested Loop Optimization

```python
# Good: Parallel outer loop
@ti.kernel
def compute_interactions():
    for i in range(n_particles):  # Parallelized
        force = ti.Vector([0.0, 0.0, 0.0])
        for j in range(n_particles):  # Sequential
            if i != j:
                force += calculate_force(i, j)
        total_force[i] = force

# Better: Tiling for cache efficiency
@ti.kernel
def compute_interactions_tiled():
    tile_size = 32
    for tile_i in range(0, n_particles, tile_size):
        for tile_j in range(0, n_particles, tile_size):
            for i in range(tile_i, min(tile_i + tile_size, n_particles)):
                for j in range(tile_j, min(tile_j + tile_size, n_particles)):
                    if i != j:
                        # Process interaction
                        pass
```

### Reduction Patterns

```python
# Sum reduction
@ti.kernel
def sum_energy() -> ti.f32:
    total = 0.0
    for i in range(n_particles):
        total += compute_particle_energy[i]
    return total

# Parallel reduction with atomic operations
@ti.kernel
def parallel_sum():
    for i in range(n_particles):
        ti.atomic_add(total_energy[None], compute_particle_energy[i])
```

## Memory Access Patterns

### Spatial Locality

```python
# Bad: Column-major access in row-major array
for j in range(cols):
    for i in range(rows):
        process(array[i, j])  # Poor cache usage

# Good: Row-major access
for i in range(rows):
    for j in range(cols):
        process(array[i, j])  # Better cache usage
```

### Loop Fusion

```python
# Bad: Multiple passes over data
for i in range(n):
    a[i] = b[i] * 2

for i in range(n):
    c[i] = a[i] + d[i]

# Good: Single pass
for i in range(n):
    temp = b[i] * 2
    a[i] = temp
    c[i] = temp + d[i]
```

### Loop Tiling/Blocking

```python
# Cache-friendly matrix multiplication
@ti.kernel
def matmul_tiled(A: ti.template(), B: ti.template(), C: ti.template()):
    tile_size = 16  # Adjust based on cache size
    
    for i_tile in range(0, n, tile_size):
        for j_tile in range(0, m, tile_size):
            for k_tile in range(0, p, tile_size):
                # Process tile
                for i in range(i_tile, min(i_tile + tile_size, n)):
                    for j in range(j_tile, min(j_tile + tile_size, m)):
                        for k in range(k_tile, min(k_tile + tile_size, p)):
                            C[i, j] += A[i, k] * B[k, j]
```

## Boundary Condition Handling

### Avoid Conditionals in Inner Loops

```python
# Bad: Conditional in inner loop
@ti.kernel
def update_field():
    for i in range(n):
        for j in range(m):
            if i > 0 and i < n-1 and j > 0 and j < m-1:
                field[i, j] = compute_interior(i, j)
            else:
                field[i, j] = compute_boundary(i, j)

# Good: Separate loops
@ti.kernel
def update_field_optimized():
    # Interior points (no conditionals)
    for i in range(1, n-1):
        for j in range(1, m-1):
            field[i, j] = compute_interior(i, j)
    
    # Boundaries (handled separately)
    for i in range(n):
        field[i, 0] = compute_boundary(i, 0)
        field[i, m-1] = compute_boundary(i, m-1)
    for j in range(1, m-1):
        field[0, j] = compute_boundary(0, j)
        field[n-1, j] = compute_boundary(n-1, j)
```

## Unrolling and Pipelining

### Manual Loop Unrolling

```python
# Original loop
for i in range(0, n):
    process(data[i])

# Unrolled by factor of 4
for i in range(0, n - 3, 4):
    process(data[i])
    process(data[i + 1])
    process(data[i + 2])
    process(data[i + 3])

# Handle remainder
for i in range(n - (n % 4), n):
    process(data[i])
```

## Particle-Particle Interactions

### Symmetric Interaction Optimization

```python
@ti.kernel
def compute_pairwise_forces():
    # Exploit Newton's third law
    for i in range(n_particles):
        for j in range(i + 1, n_particles):  # Only compute upper triangle
            force_ij = calculate_force(i, j)
            forces[i] += force_ij
            forces[j] -= force_ij  # Newton's third law
```

### Spatial Hashing for Range Queries

```python
@ti.kernel
def update_spatial_hash():
    # Build spatial hash for neighbor searches
    for i in range(n_particles):
        cell_id = get_cell_id(position[i])
        # Add particle to cell
        cell_particles[cell_id].append(i)

@ti.kernel
def compute_short_range_forces():
    for i in range(n_particles):
        cell_id = get_cell_id(position[i])
        # Only check neighboring cells
        for neighbor_cell in get_neighbor_cells(cell_id):
            for j in cell_particles[neighbor_cell]:
                if i != j and distance(i, j) < cutoff:
                    # Compute interaction
                    pass
```

## Common Anti-Patterns to Avoid

### Don't Do This

```python
# Anti-pattern 1: Repeated expensive calculations
for i in range(n):
    for j in range(m):
        value = expensive_function(i) * data[j]  # Recalculating for each j

# Anti-pattern 2: Poor memory access
for i in range(n):
    sum += array[random_indices[i]]  # Random access pattern

# Anti-pattern 3: Unnecessary allocations in loops
for i in range(n):
    temp_array = np.zeros(1000)  # Allocation in loop
    # Use temp_array

# Anti-pattern 4: String operations in numerical loops
for i in range(n):
    result[i] = float(str(values[i]))  # Unnecessary conversions
```

## Performance Measurement

### Loop Profiling Template

```python
import time

# Simple timing
start = time.perf_counter()
for i in range(n):
    # Loop body
    pass
elapsed = time.perf_counter() - start
print(f"Loop time: {elapsed:.6f} seconds")

# Detailed profiling with Taichi
ti.profiler.start()
kernel_function()
ti.profiler.stop()
ti.profiler.print()
```

## Checklist for Loop Optimization

Before optimizing a loop, consider:

- Can this loop be eliminated through vectorization?
- Is the algorithm optimal (O(n) vs O(n²))?
- Are memory accesses cache-friendly?
- Can the loop be parallelized?
- Are there redundant calculations?
- Is the loop boundary checking efficient?
- Are data types appropriate (float32 vs float64)?
- Can loop fusion or fission improve performance?
- Would loop tiling improve cache usage?
- Is the compiler able to optimize this loop?

## References

- [Taichi Kernel Optimization](https://docs.taichi-lang.org/docs/kernel_optimization)
- [NumPy Performance Guide](https://numpy.org/doc/stable/reference/random/performance.html)
- [Loop Optimization Techniques](https://en.wikipedia.org/wiki/Loop_optimization)
