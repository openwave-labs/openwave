# Performance Guidelines

## Overview

OpenWave simulates quantum physics at the Planck scale, requiring careful attention to performance optimization. This guide provides best practices for writing high-performance code in the project. It contains general performance principles, Taichi optimization, GPU best practices, and benchmarking guidelines.

## General Performance Principles

### Profile Before Optimizing

- Always measure performance before and after optimization
- Use profiling tools to identify actual bottlenecks
- Focus optimization efforts on hot paths

### Algorithm Selection

- Choose appropriate algorithms for the scale of the problem
- Consider time vs. space complexity tradeoffs
- Use established scientific computing libraries when appropriate

## Python Performance

### NumPy Best Practices

- Vectorize operations instead of using Python loops
- Use appropriate dtypes (float32 vs float64)
- Avoid unnecessary array copies
- Use views and slices when possible

### Memory Management

- Pre-allocate arrays when size is known
- Reuse buffers in iterative calculations
- Be mindful of memory layout (C vs Fortran order)

## Taichi Optimization

### Kernel Design

- Minimize kernel launches by batching operations
- Use Taichi's parallel for-loops effectively
- Avoid excessive atomic operations
- Structure data for coalesced memory access

### Data Layout

```python
# Good: Structure of Arrays (SoA)
@ti.data_oriented
class Particles:
    def __init__(self, n):
        self.x = ti.field(dtype=ti.f32, shape=n)
        self.y = ti.field(dtype=ti.f32, shape=n)
        self.z = ti.field(dtype=ti.f32, shape=n)

# Consider: Array of Structures (AoS) when accessing all properties together
@ti.data_oriented
class Particles:
    def __init__(self, n):
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
```

### GPU Optimization

- Ensure sufficient parallelism for GPU utilization
- Minimize CPU-GPU data transfers
- Use appropriate block sizes for GPU kernels
- Avoid divergent branching in GPU code

## Parallel Computing

### Threading Considerations

- Use thread-safe data structures when necessary
- Minimize lock contention
- Consider using thread-local storage for temporary data

### Vectorization

- Write code that compiler can auto-vectorize
- Use SIMD-friendly data layouts
- Align data structures appropriately

## Caching Strategies

### Computation Caching

- Cache expensive calculations when inputs don't change
- Use memoization for recursive functions
- Implement lazy evaluation where appropriate

### Data Caching

- Keep frequently accessed data in cache-friendly layouts
- Use spatial and temporal locality principles
- Consider cache line sizes (typically 64 bytes)

## Performance Benchmarking

### Benchmark Guidelines

- Create reproducible benchmarks
- Test with realistic data sizes
- Measure both average and worst-case performance
- Document hardware specifications for benchmarks

### Performance Targets

- Granule simulation: >1M particles at 30 FPS
- Wave propagation: Real-time for 2D, near real-time for 3D
- Memory usage: Scale linearly with particle count

## Common Pitfalls to Avoid

### Python-Specific

- Avoid creating objects in inner loops
- Don't use global variables in performance-critical code
- Minimize string operations in numerical code
- Avoid repeated attribute lookups in loops

### Numerical Computing

- Check for numerical stability in iterative methods
- Use appropriate precision for calculations
- Avoid unnecessary type conversions
- Be careful with division by small numbers

## Performance Monitoring

### Metrics to Track

- Frame rate / simulation steps per second
- Memory usage and allocation patterns
- GPU utilization (if applicable)
- Cache hit rates

### Profiling Tools

- Python: cProfile, line_profiler, memory_profiler
- Taichi: Built-in profiler (`ti.profiler`)
- System: perf, Intel VTune, NVIDIA Nsight

## Optimization Workflow

1. **Establish baseline**: Measure current performance
2. **Profile**: Identify bottlenecks
3. **Optimize**: Apply targeted optimizations
4. **Verify**: Ensure correctness is maintained
5. **Measure**: Quantify improvement
6. **Document**: Record optimization rationale and results

## References

- [Taichi Performance Tuning Guide](https://docs.taichi-lang.org/docs/performance)
- [NumPy Performance Tips](https://numpy.org/doc/stable/user/c-info.beyond-basics.html)
- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
