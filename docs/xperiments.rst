Xperiments (Virtual Lab)
========================

Xperiments are pre-built simulations that demonstrate various wave mechanics phenomena.

Overview
--------

The xperiments module provides interactive experiments for exploring Energy Wave Theory concepts.

Launch all xperiments via the CLI:

.. code-block:: bash

   openwave -x

Available Xperiments
--------------------

Spacetime Vibration
~~~~~~~~~~~~~~~~~~~

**Module:** ``openwave.xperiments.level0_granule_medium.spacetime_vibration``

Demonstrates fundamental oscillations of the wave medium.

**Features:**

- Visualizes granule vibrations
- Shows harmonic motion
- Demonstrates energy distribution

**Parameters:**

.. code-block:: python

   from openwave.xperiments.level0_granule_medium.spacetime_vibration import spacetime_vibration

   # Run via CLI: openwave -x
   # Select "Spacetime Vibration" from menu

Spherical Wave
~~~~~~~~~~~~~~

**Module:** ``openwave.xperiments.level0_granule_medium.spherical_wave``

Demonstrates wave propagation from a point source.

**Features:**

- Expanding spherical wave front
- Energy conservation visualization
- Wave speed measurement

**Parameters:**

.. code-block:: python

   from openwave.xperiments.level0_granule_medium.spherical_wave import spherical_wave

   # Run via CLI: openwave -x
   # Select "Spherical Wave" from menu

Standing Wave
~~~~~~~~~~~~~

**Module:** ``openwave.xperiments.level0_granule_medium.standing_wave``

Shows wave interference and standing wave patterns.

**Features:**

- Interference patterns
- Nodes and antinodes
- Energy localization

**Parameters:**

.. code-block:: python

   from openwave.xperiments.level0_granule_medium.standing_wave import standing_wave

   # Run via CLI: openwave -x
   # Select "Standing Wave" from menu

Superposing Wave
~~~~~~~~~~~~~~~~

**Module:** ``openwave.xperiments.level0_granule_medium.superposing_wave``

Demonstrates wave superposition with multiple sources.

**Features:**

- Multiple wave sources
- Constructive/destructive interference
- Complex interference patterns

**Parameters:**

.. code-block:: python

   from openwave.xperiments.level0_granule_medium.superposing_wave import superposing_wave

   # Run via CLI: openwave -x
   # Select "Superposing Wave" from menu

The Pulse
~~~~~~~~~

**Module:** ``openwave.xperiments.level0_granule_medium.the_pulse``

Simulates energy pulse propagation through the medium.

**Features:**

- Localized energy transfer
- Wave packet dynamics
- Group velocity visualization

**Parameters:**

.. code-block:: python

   from openwave.xperiments.level0_granule_medium.the_pulse import the_pulse

   # Run via CLI: openwave -x
   # Select "The Pulse" from menu

X-Waves
~~~~~~~

**Module:** ``openwave.xperiments.level0_granule_medium.xwaves``

Demonstrates crossing wave patterns.

**Features:**

- Perpendicular wave interaction
- Complex interference patterns
- Energy exchange visualization

**Parameters:**

.. code-block:: python

   from openwave.xperiments.level0_granule_medium.xwaves import xwaves

   # Run via CLI: openwave -x
   # Select "X-Waves" from menu

Yin Yang
~~~~~~~~

**Module:** ``openwave.xperiments.level0_granule_medium.yin_yang``

Shows complementary wave patterns.

**Features:**

- Symmetrical wave patterns
- Phase relationships
- Energy balance visualization

**Parameters:**

.. code-block:: python

   from openwave.xperiments.level0_granule_medium.yin_yang import yin_yang

   # Run via CLI: openwave -x
   # Select "Yin Yang" from menu

Creating Custom Xperiments
---------------------------

Structure
~~~~~~~~~

Custom xperiments should follow this structure:

.. code-block:: python

   from openwave.spacetime.medium_level0 import BCCLattice
   from openwave.common import constants
   import taichi as ti

   class MyCustomXperiment:
       """Custom experiment description."""

       def __init__(self, lattice: BCCLattice, **kwargs):
           self.lattice = lattice
           # Initialize parameters

       @ti.kernel
       def update(self):
           """Update simulation state."""
           # Implement physics update
           pass

       def run(self):
           """Main simulation loop."""
           # Implement rendering loop
           pass

Example: Custom Wave Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openwave.spacetime.medium_level0 import BCCLattice
   from openwave.common import constants
   import taichi as ti
   import numpy as np

   class CustomWavePattern:
       """Creates a custom wave interference pattern."""

       def __init__(
           self,
           lattice: BCCLattice,
           num_sources: int = 3,
           amplitude: float = 0.5
       ):
           self.lattice = lattice
           self.num_sources = num_sources
           self.amplitude = amplitude
           self.time = 0.0

           # Create wave sources in a circle
           self.sources = []
           for i in range(num_sources):
               angle = 2 * np.pi * i / num_sources
               radius = lattice.max_universe_edge * 0.3
               center = lattice.max_universe_edge / 2
               x = center + radius * np.cos(angle)
               y = center + radius * np.sin(angle)
               z = center
               self.sources.append([x, y, z])

       @ti.kernel
       def update(self, dt: float):
           """Update wave positions."""
           omega = 2 * np.pi * constants.EWAVE_FREQUENCY

           for i in range(self.lattice.granule_count):
               # Sum contributions from all sources
               displacement = ti.Vector([0.0, 0.0, 0.0])

               for source in ti.static(self.sources):
                   # Calculate distance to source
                   pos = self.lattice.position_am[i]
                   source_pos = ti.Vector(source)
                   r = (pos - source_pos).norm()

                   # Calculate wave contribution
                   k = 2 * np.pi / constants.EWAVE_LENGTH
                   phase = k * r - omega * self.time
                   amplitude = self.amplitude / (1 + r)  # Decay with distance

                   # Add to total displacement
                   displacement += amplitude * ti.math.sin(phase)

               # Update granule position
               self.lattice.velocity_am[i] = displacement

           self.time += dt

       def run(self):
           """Run the simulation."""
           window = ti.ui.Window("Custom Wave Pattern", (1920, 1080))
           canvas = window.get_canvas()
           scene = ti.ui.Scene()
           camera = ti.ui.Camera()

           while window.running:
               # Update physics
               self.update(0.016)  # ~60 FPS

               # Render
               camera.position(10, 10, 10)
               camera.lookat(0, 0, 0)
               scene.set_camera(camera)

               scene.particles(
                   self.lattice.position_am,
                   radius=0.02,
                   color=self.lattice.granule_color
               )

               canvas.scene(scene)
               window.show()

Xperiment Best Practices
-------------------------

1. **Initialize Properly**

   .. code-block:: python

      ti.init(arch=ti.gpu)
      lattice = BCCLattice(UNIVERSE_SIZE)

2. **Use Taichi Kernels**

   For performance, implement physics in ``@ti.kernel`` functions.

3. **Handle Time Steps**

   Use consistent time steps for stable simulations:

   .. code-block:: python

      dt = 1.0 / 60.0  # 60 FPS

4. **Clean Up Resources**

   .. code-block:: python

      ti.reset()  # Reset Taichi after each experiment

5. **Document Parameters**

   Always document configurable parameters.

Performance Tips
----------------

For Large Simulations
~~~~~~~~~~~~~~~~~~~~~

- Reduce granule count by increasing wavelength
- Use lower resolution for prototyping
- Profile using ``ti.profiler``

For Real-Time Rendering
~~~~~~~~~~~~~~~~~~~~~~~

- Target 30-60 FPS
- Adjust granule radius for visibility
- Use level-of-detail techniques

Contributing Xperiments
------------------------

To contribute new xperiments:

1. Create your experiment in ``openwave/xperiments/``
2. Follow the naming convention: ``my_experiment.py``
3. Add to the CLI menu in ``openwave/i_o/cli.py``
4. Document in this page
5. Submit a pull request

See :doc:`contributing` for details.

Next Steps
----------

- Try modifying existing xperiments
- Create your own experiments
- Read :doc:`api/openwave` for API details
- Check :doc:`theory/energy_wave_theory` for physics background
