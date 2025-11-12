Computational Approaches
========================

OpenWave implements Energy Wave Theory through two complementary computational methods.

Overview
--------

Two Levels of Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Level-0:** Granule-based medium (educational)
2. **Level-1:** Field-based medium (research) [WIP]

Both approaches model wave dynamics but with different computational strategies.

Level-0: Granule-Based Medium
------------------------------

Description
~~~~~~~~~~~

The granule-based approach represents the wave medium as discrete particles (granules) that oscillate
harmonically. This creates an intuitive, visually comprehensible model of wave mechanics.

Key Features
~~~~~~~~~~~~

- **Discrete Particles:** Individual granules with position and velocity
- **BCC/SC Lattices:** Body-centered cubic or simple cubic arrangements
- **GPU-Accelerated:** Parallel processing via Taichi
- **Visual:** Direct 3D rendering of granule motion

Implementation
~~~~~~~~~~~~~~

The medium is modeled as a crystal lattice:

.. code-block:: python

   from openwave.spacetime.medium_level0 import BCCLattice

   # Create BCC lattice with granules
   lattice = BCCLattice(
       universe_size=[L_x, L_y, L_z],
       theme="OCEAN"
   )

Granule Properties
~~~~~~~~~~~~~~~~~~

Each granule has:

- **Position:** 3D coordinates in space
- **Velocity:** Displacement from equilibrium
- **Mass:** Based on medium density
- **Radius:** Derived from unit cell size

Lattice Structures
~~~~~~~~~~~~~~~~~~

**Body-Centered Cubic (BCC):**

- 68% space filling efficiency
- 2 granules per unit cell
- Higher resolution for given memory

**Simple Cubic (SC):**

- 52% space filling efficiency
- 1 granule per unit cell
- Simpler geometry

Advantages
~~~~~~~~~~

✅ **Intuitive:** Easy to visualize and understand
✅ **Educational:** Great for teaching wave mechanics
✅ **Fast Prototyping:** Quick to implement new ideas
✅ **Visual Appeal:** Beautiful 3D animations

Limitations
~~~~~~~~~~~

⚠️ **Resolution Limited:** Discrete particles limit spatial resolution
⚠️ **Memory Intensive:** Large lattices require significant memory
⚠️ **Research Scope:** Less suitable for complex force calculations

Level-1: Wave-Field Medium
----------------------------

Description
~~~~~~~~~~~

The wave-field approach treats the medium as a continuous 3D vector field, similar to lattice QCD
(quantum chromodynamics) but with classical wave field equations.

Key Features
~~~~~~~~~~~~

- **Continuous Field:** Vector field values at grid points
- **PDE Solver:** Partial differential equations
- **Scalable:** Handle complex phenomena
- **Research-Grade:** Suitable for quantitative predictions

Implementation [WIP]
~~~~~~~~~~~~~~~~~~~~

The medium is a discretized field:

.. code-block:: python

   from openwave.spacetime.medium_level1 import FieldMedium  # WIP

   # Create wave-field medium
   field = FieldMedium(
       grid_size=[nx, ny, nz],
       grid_spacing=dx
   )

Field Representation
~~~~~~~~~~~~~~~~~~~~

Each grid point stores:

- **Field Value:** Vector amplitude [ψx, ψy, ψz]
- **Field Gradient:** Spatial derivatives
- **Field Velocity:** Time derivatives
- **Energy Density:** Local energy

Numerical Methods
~~~~~~~~~~~~~~~~~

**Finite Difference Method:**

Discretize spatial and temporal derivatives:

.. math::

   \\frac{\\partial^2 \\psi}{\\partial x^2} \\approx \\frac{\\psi_{i+1} - 2\\psi_i + \\psi_{i-1}}{\\Delta x^2}

**Time Integration:**

Use explicit or implicit schemes:

- **Explicit:** Forward Euler, RK4
- **Implicit:** Backward Euler, Crank-Nicolson

Advantages
~~~~~~~~~~

✅ **Scalable:** Handle complex multi-particle systems
✅ **Accurate:** High-order numerical methods
✅ **Flexible:** Easy to add new physics
✅ **Research-Ready:** Quantitative predictions

Limitations
~~~~~~~~~~~

⚠️ **Computational Cost:** More intensive than granules
⚠️ **Less Intuitive:** Field values less visual than particles
⚠️ **Implementation Complexity:** Requires sophisticated numerics

Comparison Table
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - Level-0 (Granule)
     - Level-1 (Field)
   * - **Representation**
     - Discrete particles
     - Continuous field
   * - **Best For**
     - Education, visualization
     - Research, predictions
   * - **Memory Usage**
     - O(N granules)
     - O(N grid points)
   * - **Compute Cost**
     - Lower
     - Higher
   * - **Resolution**
     - Limited by granule count
     - Limited by grid spacing
   * - **Accuracy**
     - Qualitative
     - Quantitative
   * - **Status**
     - Released
     - Work in Progress

Computational Performance
-------------------------

GPU Acceleration
~~~~~~~~~~~~~~~~

Both approaches use Taichi for GPU parallelization:

.. code-block:: python

   import taichi as ti

   @ti.kernel
   def update_wave():
       for i in range(total_points):
           # Parallel update on GPU
           pass

Memory Optimization
~~~~~~~~~~~~~~~~~~~

**Level-0:**

- Use attometer units for float32 precision
- 1D arrays for cache efficiency
- Minimize field allocations

**Level-1:**

- Sparse grids for localized phenomena
- Block-based decomposition
- Adaptive mesh refinement [planned]

Scalability
~~~~~~~~~~~

**Level-0:**

- 10³ - 10⁶ granules typical
- Limited by VRAM
- Real-time visualization

**Level-1:**

- 10² - 10³ per dimension
- Batch processing
- Post-processing visualization

Choosing an Approach
--------------------

Use Level-0 When...
~~~~~~~~~~~~~~~~~~~

- Learning wave mechanics
- Creating visualizations
- Rapid prototyping
- Teaching/presentations

Use Level-1 When...
~~~~~~~~~~~~~~~~~~~

- Conducting research
- Need quantitative results
- Modeling complex phenomena
- Validating against experiments

Future Directions
-----------------

Planned Enhancements
~~~~~~~~~~~~~~~~~~~~

**Level-0:**

- Adaptive granule density
- Better force calculations
- Improved rendering

**Level-1:**

- Full PDE implementation
- Multi-particle simulations
- Force field calculations
- Matter formation studies

Next Steps
----------

- Try :doc:`../xperiments` to see Level-0 in action
- Read :doc:`energy_wave_theory` for theoretical background
- Check :doc:`../api/spacetime` for technical details
- See :doc:`../contributing` to help develop Level-1
