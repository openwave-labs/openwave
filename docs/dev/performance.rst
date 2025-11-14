Performance Guidelines
======================

.. note::
   For detailed performance guidelines, see ``dev_docs/PERFORMANCE_GUIDELINES.md``
   and ``dev_docs/LOOP_OPTIMIZATION.md`` in the repository.

This page covers performance optimization strategies for OpenWave.

GPU Acceleration
----------------

Taichi Kernels
~~~~~~~~~~~~~~

Use ``@ti.kernel`` for GPU parallelization:

.. code-block:: python

   @ti.kernel
   def update_wave():
       for i in range(granule_count):
           # Runs in parallel on GPU
           velocity[i] = calculate_velocity(i)

Single Outermost Loop
~~~~~~~~~~~~~~~~~~~~~

**Critical:** Taichi parallelizes only the outermost loop:

.. code-block:: python

   # Good: Single outermost loop
   @ti.kernel
   def update_good():
       for i in range(N):
           # Parallel across GPU threads
           result[i] = compute(i)

   # Bad: Nested loops
   @ti.kernel
   def update_bad():
       for i in range(N):
           for j in range(M):  # Sequential!
               result[i, j] = compute(i, j)

1D Arrays vs 3D Arrays
~~~~~~~~~~~~~~~~~~~~~~

**Use 1D arrays with 3D vectors:**

.. code-block:: python

   # Good: 1D array, single loop
   position = ti.Vector.field(3, dtype=ti.f32, shape=N)

   @ti.kernel
   def update():
       for i in range(N):
           position[i] += velocity[i] * dt

   # Bad: 3D array, nested loops
   position = ti.field(dtype=ti.f32, shape=(nx, ny, nz))

   @ti.kernel
   def update():
       for i in range(nx):
           for j in range(ny):  # Sequential
               for k in range(nz):  # Sequential
                   position[i, j, k] += velocity[i, j, k]

Memory Optimization
-------------------

Data Structures
~~~~~~~~~~~~~~~

**Use appropriate data types:**

.. code-block:: python

   # Use f32 for GPU efficiency
   field = ti.field(dtype=ti.f32, shape=N)

   # Use attometer units for precision
   position_am = position / constants.ATTOMETER

Minimize Allocations
~~~~~~~~~~~~~~~~~~~~

Reuse fields instead of creating new ones:

.. code-block:: python

   # Good: Reuse field
   temp_field = ti.field(dtype=ti.f32, shape=N)

   @ti.kernel
   def compute():
       for i in range(N):
           temp_field[i] = calculate(i)

   # Bad: Allocate in loop
   for iteration in range(steps):
       temp = ti.field(dtype=ti.f32, shape=N)  # Slow!

Cache Efficiency
~~~~~~~~~~~~~~~~

Access memory contiguously:

.. code-block:: python

   # Good: Sequential access
   for i in range(N):
       sum += array[i]

   # Bad: Random access
   for i in random.shuffle(indices):
       sum += array[i]

Algorithmic Optimization
------------------------

Vectorization
~~~~~~~~~~~~~

Use NumPy/Taichi vector operations:

.. code-block:: python

   # Good: Vectorized
   result = ti.math.sqrt(x**2 + y**2 + z**2)

   # Bad: Manual
   result = 0.0
   result += x * x
   result += y * y
   result += z * z
   result = ti.math.sqrt(result)

Avoid Branches
~~~~~~~~~~~~~~

Minimize conditionals in kernels:

.. code-block:: python

   # Good: Use math instead of branches
   @ti.kernel
   def update():
       for i in range(N):
           # Branchless
           value[i] = max(0.0, calculate(i))

   # Worse: Branch in tight loop
   @ti.kernel
   def update():
       for i in range(N):
           temp = calculate(i)
           if temp > 0:
               value[i] = temp
           else:
               value[i] = 0.0

Profiling
---------

Taichi Profiler
~~~~~~~~~~~~~~~

Use built-in profiler:

.. code-block:: python

   import taichi as ti

   ti.init(arch=ti.gpu, kernel_profiler=True)

   # Run simulation
   for step in range(steps):
       update_wave()

   # Print profile
   ti.profiler.print_kernel_profiler_info()

Python Profiler
~~~~~~~~~~~~~~~

For Python code:

.. code-block:: python

   import cProfile
   import pstats

   profiler = cProfile.Profile()
   profiler.enable()

   # Run code
   run_simulation()

   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(10)

GPU Memory Monitoring
~~~~~~~~~~~~~~~~~~~~~

Check GPU memory usage:

.. code-block:: bash

   # NVIDIA GPUs
   nvidia-smi

   # In Python
   import taichi as ti
   ti.init(arch=ti.gpu)
   print(ti.profiler.get_kernel_profiler_info())

Common Optimizations
--------------------

Loop Unrolling
~~~~~~~~~~~~~~

Let compiler unroll small loops:

.. code-block:: python

   @ti.kernel
   def compute():
       for i in range(N):
           # Small fixed loop - compiler can unroll
           for dim in ti.static(range(3)):
               position[i][dim] += velocity[i][dim]

Precompute Constants
~~~~~~~~~~~~~~~~~~~~

Calculate once, reuse many times:

.. code-block:: python

   # Good: Precompute
   omega = 2 * ti.math.pi * frequency
   k = 2 * ti.math.pi / wavelength

   @ti.kernel
   def update():
       for i in range(N):
           phase = k * distance[i] - omega * time
           displacement[i] = amplitude * ti.sin(phase)

Reduce Function Calls
~~~~~~~~~~~~~~~~~~~~~

Inline simple calculations:

.. code-block:: python

   # Good: Inline
   @ti.kernel
   def update():
       for i in range(N):
           result[i] = x[i] * x[i] + y[i] * y[i]

   # Worse: Function call overhead
   @ti.func
   def square(val):
       return val * val

   @ti.kernel
   def update():
       for i in range(N):
           result[i] = square(x[i]) + square(y[i])

Scalability
-----------

Grid Size vs Performance
~~~~~~~~~~~~~~~~~~~~~~~~

**Trade-offs:**

.. list-table::
   :header-rows: 1

   * - Grid Size
     - Memory
     - Compute Time
     - Resolution
   * - 10³ granules
     - ~MB
     - Real-time
     - Low
   * - 10⁶ granules
     - ~GB
     - Seconds/frame
     - Medium
   * - 10⁹ granules
     - ~TB
     - Minutes/frame
     - High

Batch Processing
~~~~~~~~~~~~~~~~

For large simulations:

.. code-block:: python

   # Process in batches
   batch_size = 1000000
   for start in range(0, granule_count, batch_size):
       end = min(start + batch_size, granule_count)
       process_batch(start, end)

Adaptive Resolution
~~~~~~~~~~~~~~~~~~~

Use high resolution only where needed:

.. code-block:: python

   # Fine grid near particles
   # Coarse grid far away
   # (Implementation depends on specific needs)

Platform-Specific
-----------------

CUDA (NVIDIA)
~~~~~~~~~~~~~

.. code-block:: python

   ti.init(
       arch=ti.cuda,
       device_memory_GB=4.0,
       packed=True
   )

Vulkan (Cross-platform)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ti.init(
       arch=ti.vulkan,
       device_memory_GB=2.0
   )

CPU (Fallback)
~~~~~~~~~~~~~~

.. code-block:: python

   ti.init(
       arch=ti.cpu,
       cpu_max_num_threads=16
   )

Benchmarking
------------

Timing Code
~~~~~~~~~~~

.. code-block:: python

   import time

   start = time.perf_counter()
   run_simulation()
   duration = time.perf_counter() - start

   print(f"Simulation took {duration:.3f} seconds")
   print(f"FPS: {frames / duration:.1f}")

Performance Targets
~~~~~~~~~~~~~~~~~~~

**Target Performance:**

- Interactive visualization: 30-60 FPS
- Research simulations: Minutes per timestep acceptable
- Batch processing: Optimize for throughput

Best Practices
--------------

1. **Profile First:** Measure before optimizing
2. **GPU First:** Use kernels for parallel work
3. **Memory Aware:** Monitor GPU memory usage
4. **Vectorize:** Use vector operations when possible
5. **Test:** Verify correctness after optimization

Resources
---------

- **Taichi Docs:** https://docs.taichi-lang.org/docs/performance
- **Loop Optimization:** See ``dev_docs/LOOP_OPTIMIZATION.md``
- **GPU Computing:** CUDA/Vulkan documentation

Next Steps
----------

- Review :doc:`coding_standards` for code quality
- See :doc:`architecture` for system design
- Check :doc:`../contributing` for how to contribute optimizations
