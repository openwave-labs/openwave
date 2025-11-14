Usage Guide
===========

This guide provides examples and patterns for using OpenWave.

Command Line Interface
----------------------

OpenWave provides a CLI for quick access to experiments:

.. code-block:: bash

   # Launch xperiments menu
   openwave -x

   # Show help
   openwave --help

   # Show version
   openwave --version

Basic Usage Patterns
--------------------

Creating a Lattice
~~~~~~~~~~~~~~~~~~

The lattice represents the wave medium:

.. code-block:: python

   from openwave.spacetime.medium_level0 import BCCLattice
   from openwave.common import constants
   import taichi as ti

   # Initialize Taichi for GPU
   ti.init(arch=ti.gpu)

   # Define universe size
   UNIVERSE_SIZE = [
       4 * constants.EWAVE_LENGTH,
       4 * constants.EWAVE_LENGTH,
       4 * constants.EWAVE_LENGTH
   ]

   # Create BCC lattice
   lattice = BCCLattice(UNIVERSE_SIZE, theme="OCEAN")

   # Print lattice properties
   print(f"Total granules: {lattice.granule_count:,}")
   print(f"Grid size: {lattice.grid_size}")
   print(f"Unit cell edge: {lattice.unit_cell_edge:.2e} m")

Available Themes
^^^^^^^^^^^^^^^^

- ``OCEAN``: Blue tones (default)
- ``DESERT``: Warm earth tones
- ``FOREST``: Green tones
- ``SUNSET``: Orange/red tones

Simple Cubic Lattice
^^^^^^^^^^^^^^^^^^^^

Alternatively, use a Simple Cubic (SC) lattice:

.. code-block:: python

   from openwave.spacetime.medium_level0 import SCLattice

   lattice = SCLattice(UNIVERSE_SIZE, theme="DESERT")

Working with Waves
------------------

Spherical Wave
~~~~~~~~~~~~~~

Create an expanding spherical wave:

.. code-block:: python

   from openwave.spacetime.energy_wave_level0 import SphericalWave

   # Initialize lattice first
   lattice = BCCLattice(UNIVERSE_SIZE)

   # Create spherical wave at center
   wave = SphericalWave(
       lattice=lattice,
       amplitude=0.5,  # Wave amplitude
       frequency=constants.EWAVE_FREQUENCY,
       source_position=None  # None = center
   )

   # Simulate
   wave.run()

Standing Wave
~~~~~~~~~~~~~

Create interference patterns:

.. code-block:: python

   from openwave.spacetime.energy_wave_level0 import StandingWave

   wave = StandingWave(
       lattice=lattice,
       amplitude=0.7,
       frequency=constants.EWAVE_FREQUENCY
   )

   wave.run()

The Pulse
~~~~~~~~~

Simulate energy transfer:

.. code-block:: python

   from openwave.xperiments.the_pulse import ThePulse

   pulse = ThePulse(
       lattice=lattice,
       pulse_width=10,  # Number of granules
       initial_velocity=0.5
   )

   pulse.run()

Visualization Options
---------------------

Rendering Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Control visualization parameters:

.. code-block:: python

   from openwave._io.render import RenderConfig

   config = RenderConfig(
       window_size=(1920, 1080),
       fps=60,
       show_grid=True,
       show_axes=True,
       camera_position=(10, 10, 10),
       background_color=(0.1, 0.1, 0.1)
   )

Interactive Controls
~~~~~~~~~~~~~~~~~~~~

During simulation, you can use:

- **Mouse drag**: Rotate view
- **Mouse wheel**: Zoom in/out
- **Arrow keys**: Pan camera
- **Space**: Pause/resume
- **R**: Reset view
- **ESC**: Exit simulation

Exporting Data
--------------

Export Simulation Data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   # Export granule positions
   positions = lattice.position_am.to_numpy()
   np.save('positions.npy', positions)

   # Export velocities
   velocities = lattice.velocity_am.to_numpy()
   np.save('velocities.npy', velocities)

Export Visualization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openwave._io.render import export_frame

   # Export current frame as image
   export_frame('output.png')

   # Export animation as GIF (if supported)
   # Coming soon in future versions

Analysis and Diagnostics
-------------------------

Wave Diagnostics
~~~~~~~~~~~~~~~~

Analyze wave properties:

.. code-block:: python

   from openwave.validations.wave_diagnostics import WaveDiagnostics

   diagnostics = WaveDiagnostics(lattice)

   # Calculate total energy
   total_energy = diagnostics.total_energy()
   print(f"Total energy: {total_energy:.2e} J")

   # Calculate momentum
   momentum = diagnostics.total_momentum()
   print(f"Total momentum: {momentum}")

   # Energy distribution
   energy_dist = diagnostics.energy_distribution()

Custom Configurations
---------------------

Custom Universe Size
~~~~~~~~~~~~~~~~~~~~

Create asymmetric universes:

.. code-block:: python

   # Asymmetric universe (different dimensions)
   UNIVERSE_SIZE = [
       2 * constants.EWAVE_LENGTH,  # X dimension
       4 * constants.EWAVE_LENGTH,  # Y dimension
       8 * constants.EWAVE_LENGTH   # Z dimension
   ]

   lattice = BCCLattice(UNIVERSE_SIZE)

Custom Wave Sources
~~~~~~~~~~~~~~~~~~~

Place wave sources at specific positions:

.. code-block:: python

   # Create wave source at custom position
   source_pos = [
       lattice.universe_size[0] * 0.3,
       lattice.universe_size[1] * 0.5,
       lattice.universe_size[2] * 0.7
   ]

   wave = SphericalWave(
       lattice=lattice,
       source_position=source_pos
   )

Performance Optimization
------------------------

GPU Selection
~~~~~~~~~~~~~

For multi-GPU systems:

.. code-block:: python

   import taichi as ti

   # Use specific GPU
   ti.init(arch=ti.cuda, device_memory_GB=4.0)

Memory Management
~~~~~~~~~~~~~~~~~

For large simulations:

.. code-block:: python

   # Reduce precision for larger grids
   ti.init(arch=ti.gpu, default_fp=ti.f32)

   # Or use CPU for very large simulations
   ti.init(arch=ti.cpu)

Batch Processing
~~~~~~~~~~~~~~~~

Run multiple simulations:

.. code-block:: python

   import taichi as ti

   for amplitude in [0.1, 0.3, 0.5, 0.7]:
       ti.reset()  # Reset Taichi
       ti.init(arch=ti.gpu)

       lattice = BCCLattice(UNIVERSE_SIZE)
       wave = SphericalWave(lattice, amplitude=amplitude)
       wave.run()

       # Export results
       np.save(f'wave_{amplitude}.npy', lattice.velocity_am.to_numpy())

Advanced Examples
-----------------

Wave Superposition
~~~~~~~~~~~~~~~~~~

Create multiple wave sources:

.. code-block:: python

   from openwave.xperiments.superposing_wave import SuperposingWave

   wave = SuperposingWave(
       lattice=lattice,
       num_sources=4,
       amplitude=0.5
   )

   wave.run()

Custom Physics
~~~~~~~~~~~~~~

Modify physical constants:

.. code-block:: python

   from openwave.common import constants

   # Custom wavelength
   CUSTOM_WAVELENGTH = 5e-17  # meters

   UNIVERSE_SIZE = [
       4 * CUSTOM_WAVELENGTH,
       4 * CUSTOM_WAVELENGTH,
       4 * CUSTOM_WAVELENGTH
   ]

   lattice = BCCLattice(UNIVERSE_SIZE)

Best Practices
--------------

1. **Always initialize Taichi** before creating lattices
2. **Use appropriate GPU memory** for your simulation size
3. **Export data regularly** for long simulations
4. **Profile performance** for optimization
5. **Document custom configurations** for reproducibility

Next Steps
----------

- Explore :doc:`xperiments` for pre-built experiments
- Read :doc:`api/openwave` for detailed API reference
- Check :doc:`theory/energy_wave_theory` for physics background
- See :doc:`contributing` to add your own experiments
