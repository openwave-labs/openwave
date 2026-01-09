Architecture Overview
====================

This page describes OpenWave's system architecture and design principles.

System Modules
--------------

OpenWave is organized into modular components:

.. code-block:: text

   openwave/
   ├── spacetime/        # Wave medium and energy waves
   ├── matter/           # Particle models (WIP)
   ├── forces/           # Force calculations (WIP)
   ├── motion/           # Kinetic energy (WIP)
   ├── photon/           # Light/photons (WIP)
   ├── heat/             # Thermal energy (WIP)
   ├── xperiments/       # Virtual lab experiments
   ├── common/           # Shared utilities
   ├── validations/      # Diagnostics and analysis
   └── i_o/              # CLI and rendering

Module Relationships
--------------------

.. code-block:: text

   SPACETIME (core)
       ↓
   ├── MATTER ←→ FORCES
   ├── MOTION ←→ FORCES
   ├── PHOTON
   └── HEAT
       ↓
   XPERIMENTS (demonstrations)
       ↓
   I/O (visualization)

Spacetime Module
----------------

**Purpose:** Wave medium and energy wave dynamics

**Components:**

- ``medium_level0.py`` - Granule-Motion lattice (released)
- ``medium_level1.py`` - Field-based PDE solver (WIP)
- ``energy_wave_level0.py`` - Wave physics for granules
- ``energy_wave_level1.py`` - Wave physics for fields (WIP)

**Design:**

- 1D arrays for GPU efficiency
- Taichi kernels for parallelization
- BCC/SC lattice structures

Matter Module (WIP)
-------------------

**Purpose:** Particle models from standing waves

**Planned Components:**

- ``fundamental_particle.py`` - Electron, proton models
- ``standalone_particle.py`` - Individual particle dynamics
- ``composite_particle.py`` - Multi-particle systems
- ``atom.py`` - Atomic structures
- ``molecule.py`` - Molecular structures

Forces Module (WIP)
-------------------

**Purpose:** Force calculations from field gradients

**Planned Components:**

- ``electric.py`` - Electric forces
- ``magnetic.py`` - Magnetic forces
- ``gravitational.py`` - Gravitational forces
- ``strong.py`` - Strong nuclear forces
- ``orbital.py`` - Orbital mechanics

Common Module
-------------

**Purpose:** Shared constants, configuration, equations

**Components:**

- ``constants.py`` - Physical constants
- ``config.py`` - Configuration and themes
- ``equations.py`` - Physical equations

**Design Principle:** Single source of truth for constants

Xperiments Module
-----------------

**Purpose:** Educational demonstrations and virtual lab

**Structure:**

Each xperiment is a self-contained simulation:

.. code-block:: python

   class Xperiment:
       def __init__(self, lattice, **params):
           # Initialize
           pass

       @ti.kernel
       def update(self):
           # Physics update
           pass

       def run(self):
           # Rendering loop
           pass

I/O Module
----------

**Purpose:** User interface and visualization

**Components:**

- ``cli.py`` - Command-line interface
- ``render.py`` - 3D rendering utilities
- ``file_export.py`` - Data export (WIP)
- ``video_manager.py`` - Animation export (WIP)

Validations Module
------------------

**Purpose:** Wave diagnostics and analysis

**Components:**

- ``wave_diagnostics.py`` - Energy, momentum calculations
- ``derivations.py`` - Equation derivations

Design Principles
-----------------

Modularity
~~~~~~~~~~

- Each module has single responsibility
- Clean interfaces between modules
- Easy to extend and test

Performance First
~~~~~~~~~~~~~~~~~

- GPU acceleration via Taichi
- Memory-efficient data structures
- Profile-guided optimization

Scientific Rigor
~~~~~~~~~~~~~~~~

- Document assumptions
- Validate against theory
- Reproducible results

Computational Levels
--------------------

Level-0: Granule-Motion
~~~~~~~~~~~~~~~~~~~~~~

**Status:** Released

**Characteristics:**

- Discrete particles
- Visual and intuitive
- Educational focus
- Real-time rendering

**Use Cases:**

- Learning wave mechanics
- Creating visualizations
- Rapid prototyping

Level-1: Wave-Field
~~~~~~~~~~~~~~~~~~~~

**Status:** Work in Progress

**Characteristics:**

- Continuous fields
- PDE-based
- Research-grade
- Quantitative predictions

**Use Cases:**

- Research simulations
- Complex phenomena
- Validation studies

Data Flow
---------

Typical Simulation Flow
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   1. Initialize Taichi
   2. Create lattice (spacetime)
   3. Configure wave sources
   4. Simulation loop:
       a. Update physics (kernels)
       b. Render frame
       c. Handle input
   5. Export data/visualizations

Example:

.. code-block:: python

   ti.init(arch=ti.gpu)
   lattice = BCCLattice(universe_size)
   wave = SphericalWave(lattice)

   while running:
       wave.update(dt)
       render(lattice)

GPU Architecture
----------------

Parallelization Strategy
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @ti.kernel
   def update_parallel():
       # Single outermost loop
       for i in range(N):
           # Each iteration runs on different GPU thread
           result[i] = compute(position[i], velocity[i])

Memory Layout
~~~~~~~~~~~~~

.. code-block:: text

   CPU Memory
       ↓ (initialization)
   GPU Memory (fields)
       ↓ (computation)
   Taichi Kernels (parallel)
       ↓ (rendering)
   Display

Extensibility
-------------

Adding New Physics
~~~~~~~~~~~~~~~~~~

To add new physics:

1. Create module in appropriate directory
2. Implement using Taichi kernels
3. Add configuration to ``common/config.py``
4. Create xperiment for demonstration
5. Document in API reference

Example:

.. code-block:: python

   # openwave/forces/electric.py
   @ti.kernel
   def compute_electric_force(charges, positions):
       for i in range(N):
           force[i] = calculate_coulomb(charges, positions, i)

Testing Strategy
----------------

Levels of Testing
~~~~~~~~~~~~~~~~~

1. **Smoke Tests:** Basic functionality
2. **Unit Tests:** Individual functions
3. **Integration Tests:** Module interactions
4. **Physics Tests:** Validate against theory
5. **Performance Tests:** Benchmarks

Smoke Test Example
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   if __name__ == "__main__":
       print("SMOKE TEST: Module Name")
       ti.init(arch=ti.gpu)
       lattice = BCCLattice([1e-17, 1e-17, 1e-17])
       print(f"Granules: {lattice.granule_count}")
       print("SMOKE TEST PASSED")

Future Architecture
-------------------

Planned Enhancements
~~~~~~~~~~~~~~~~~~~~

**Short Term:**

- Level-1 field implementation
- Matter module basics
- Force calculations

**Medium Term:**

- Multi-particle simulations
- Atomic structure models
- Video export functionality

**Long Term:**

- Distributed computing
- Cloud deployment
- Web visualization

Scalability Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Adaptive mesh refinement
- Multi-GPU support
- Checkpoint/restart capability
- Parallel I/O

Contributing
------------

To contribute to architecture:

1. Discuss design in GitHub issues
2. Follow existing patterns
3. Document design decisions
4. Update this page

See :doc:`../contributing` for details.

Resources
---------

- **Taichi Lang:** https://docs.taichi-lang.org/
- **Design Patterns:** Gang of Four
- **Scientific Computing:** NumPy, SciPy docs

Next Steps
----------

- Review :doc:`coding_standards` for implementation
- See :doc:`performance` for optimization
- Check :doc:`../api/openwave` for API details
