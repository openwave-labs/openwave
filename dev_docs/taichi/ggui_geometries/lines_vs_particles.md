# GGUI Axis Rendering: Lines vs Particles Investigation

## Summary

This document records our investigation into apparent performance differences between `scene.lines()` and `scene.particles()` when rendering axis lines in OpenWave's physics simulator. **The issue turned out to be GPU/system state, not a fundamental Taichi GGUI problem.**

## Initial Observations

During development, we observed significant FPS drops when enabling axis rendering:

- **Without axis**: 48 FPS
- **With axis using scene.lines()**: 38 FPS (10 FPS loss, ~21% drop)
- **With axis using scene.particles()**: ~47 FPS (minimal loss)

The performance impact scaled with base framerate:
- Experiment A (33 FPS baseline): 33 → 32 FPS with lines (1 FPS loss)
- Experiment B (48 FPS baseline): 48 → 38 FPS with lines (10 FPS loss)

This suggested a fixed ~2-3ms overhead per `scene.lines()` call.

## Investigation Process

### 1. Initial Hypothesis: scene.lines() Performance Issue

We initially suspected `scene.lines()` had inherent overhead compared to `scene.particles()`.

### 2. Optimization Attempts

Created optimized version moving field allocation to initialization:
- Pre-allocate Taichi field once in `init_UI()`
- Populate using Taichi kernel (no numpy transfers)
- Reuse same field every frame

### 3. Particle-Based Workaround

Developed alternative using `scene.particles()` to render dense point-sampled lines:
- 1000 points per axis line
- 50 points per tick mark
- Total: ~4200 particles for axis visualization
- See: `render_particles.py`

### 4. Reproduction Attempts

Created minimal test case (`lines_vs_particles.py`) to reproduce:
- 1.2M particles with physics simulation
- Per-vertex coloring
- Multiple GUI windows
- **Result**: Could NOT reproduce the issue - both lines and particles performed identically

## Root Cause: GPU/System State

After MacBook reboot, the performance issue **completely disappeared**. Both `scene.lines()` and `scene.particles()` now perform identically at ~48 FPS.

**Conclusion**: The FPS degradation was caused by transient GPU/Metal driver state, not the code itself.

### Possible Causes

1. **Metal driver memory leak** - GPU memory fragmentation from previous sessions
2. **Thermal throttling** - GPU overheating from extended simulation runs
3. **Background processes** - Other applications using GPU resources
4. **GPU cache corruption** - Metal shader cache or pipeline state issues
5. **System memory pressure** - macOS resource constraints affecting GPU

## Lessons Learned

### ✅ Best Practices Confirmed

1. **Pre-allocate Taichi fields** - Create fields once during initialization, not per-frame
2. **Use Taichi kernels** - Populate fields with kernels, avoid numpy transfers in render loop
3. **System health matters** - GPU performance can degrade from system state, not just code
4. **Reboot when needed** - If performance degrades unexpectedly, try system reboot first

### 📝 Final Implementation

OpenWave uses **optimized `scene.lines()`** for axis rendering:
- Clean, simple code using the right primitive for the job
- Field pre-allocated and pre-populated once
- 54 points (3 axes + tick marks) vs 4200+ for particle workaround
- Sharp, accurate geometric lines

### 📁 Reference Files

- **lines_vs_particles.py** - Test script for performance comparison
- **render_particles.py** - Alternative particle-based axis implementation (kept as reference)
- **render.py** - Production implementation using optimized `scene.lines()`

## Environment

- Taichi version: 1.7.4
- Platform: macOS (Darwin 24.6.0)
- Architecture: Apple Silicon (Metal backend)
- Backend: ti.gpu
- Hardware: MacBook Pro with 120Hz display

## Recommendations

1. **Monitor system state** - Watch for performance degradation over long sessions
2. **Reboot periodically** - Clear GPU state if FPS drops unexpectedly
3. **Use appropriate primitives** - Lines for lines, particles for particles, meshes for meshes
4. **Always pre-allocate** - Never create Taichi fields or transfer data in render loops
